
import math

import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

from ActiveLearner import ActiveLearner

from typing import Dict, Any, Tuple, List

import logging
logger = logging.getLogger(__name__)


class AlphaMix(ActiveLearner):
  
    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any]) -> None:
        
        super().__init__(ct_p, t_p, self.__class__.__name__)
        
        self.alpha_closed_form_approx = False
        self.alpha_cap = 0.03125
        self.alpha_learning_rate = 0.1
        self.alpha_opt = False
        self.alpha_learning_iters = 5
        self.alpha_clf_coef = 1.0
        self.alpha_l2_coef = 0.01
        self.alpha_learn_batch_size = 1000000
        
        
    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
        
        dl_dict = dict( batch_size=self.batch_size, shuffle=False, pin_memory=True )
            
        unlab_train_dl = DataLoader(sample_unlab_subset, **dl_dict)
        lab_train_dl = DataLoader(Subset(self.dataset.unlab_train_ds, self.labelled_indices), **dl_dict)
            
        logger.info(' => Getting the labelled and unlabelled embeddings')
        self.lab_embedds_dict = {
            'embedds': torch.empty((0, self.model.backbone.get_embedding_dim()), dtype=torch.float32, device=self.device),
            'labels': torch.empty(0, dtype=torch.int8, device=torch.device('cpu'))
        }
        self.unlab_embedds_dict = {
            'outs': torch.empty((0, self.dataset.n_classes), dtype=torch.float32, device=torch.device('cpu')),
            'embedds': torch.empty((0, self.model.backbone.get_embedding_dim()), dtype=torch.float32, device=self.device),
        }

        self.get_embeddings(lab_train_dl, self.lab_embedds_dict)
        self.get_embeddings(unlab_train_dl, self.unlab_embedds_dict)
        
        idxs_unlabelled = torch.tensor(self.rand_unlab_sample) 

        pred_1 = F.softmax(self.unlab_embedds_dict["outs"], dim=1).sort(descending=True)[1][:, 0]
        ulb_embedding = self.unlab_embedds_dict["embedds"]
        lb_embedding = self.lab_embedds_dict["embedds"]

        unlabelled_size = ulb_embedding.size(0)
        embedding_size = ulb_embedding.size(1)

        min_alphas = torch.ones((unlabelled_size, embedding_size), dtype=torch.float, device=self.device)
        candidate = torch.zeros(unlabelled_size, dtype=torch.bool, device=self.device)

        if self.alpha_closed_form_approx:
            var_emb = torch.clone(ulb_embedding).to(self.device)
            var_emb.requires_grad_(True)
            out, _ = self.model.backbone(var_emb, embedding=True)
            loss = F.cross_entropy(out, pred_1.to(self.device))
            grads = torch.autograd.grad(loss, var_emb)[0].data.cpu()
            del loss, var_emb, out
        else:
            grads = None

        alpha_cap = 0.
        while alpha_cap < 1.0:
            alpha_cap += self.alpha_cap

            tmp_pred_change, tmp_min_alphas = \
                self.find_candidate_set(
                    lb_embedding, ulb_embedding, pred_1, alpha_cap=alpha_cap,
                    Y=self.labelled_indices,
                    grads=grads)

            is_changed = min_alphas.norm(dim=1) >= tmp_min_alphas.norm(dim=1)

            min_alphas[is_changed] = tmp_min_alphas[is_changed]
            candidate += tmp_pred_change

            logger.info('With alpha_cap set to %f, number of inconsistencies: %d' % (alpha_cap, int(tmp_pred_change.sum().item())))

            if candidate.sum() > n_top_k_obs: break

        if candidate.sum() > 0:
            logger.info('Number of inconsistencies: %d' % (int(candidate.sum().item())))

            logger.info('alpha_mean_mean: %f' % min_alphas[candidate].mean(dim=1).mean().item())
            logger.info('alpha_std_mean: %f' % min_alphas[candidate].mean(dim=1).std().item())
            logger.info('alpha_mean_std %f' % min_alphas[candidate].std(dim=1).mean().item())

            c_alpha = F.normalize(self.unlab_embedds_dict["embedds"][candidate].view(int(candidate.sum()), -1), p=2, dim=1).detach().cpu()

            selected_idxs = torch.as_tensor(
                self.sample(min(n_top_k_obs, candidate.sum().item()), feats=c_alpha)
            )
            selected_idxs = np.array(idxs_unlabelled[candidate.bool().cpu()][selected_idxs].numpy(), dtype=np.int32)
        else:
            selected_idxs = np.array([], dtype=np.int32)


        if len(selected_idxs) < n_top_k_obs:

            remained = n_top_k_obs - len(selected_idxs)
            bool_idx_unlb = np.zeros(len(self.rand_unlab_sample))
                    
            bool_idx_unlb[[self.rand_unlab_sample.index(int(idx)) for idx in selected_idxs]] = 1           
                    
            selected_idxs = np.concatenate([
                selected_idxs, 
                np.random.choice(
                    np.array(self.rand_unlab_sample, dtype=np.int32)[np.where(bool_idx_unlb == 0)[0]], 
                    remained, replace=False
                )
            ])
            logger.info('picked %d samples from RandomSampling.' % (remained))
            
        selected_idxs = np.array(selected_idxs).tolist()
        
        return [self.rand_unlab_sample.index(idx) for idx in selected_idxs], selected_idxs
    


    def find_candidate_set(self, lb_embedding, ulb_embedding, pred_1, alpha_cap, Y, grads):

        unlabelled_size = ulb_embedding.size(0)
        embedding_size = ulb_embedding.size(1)

        min_alphas = torch.ones((unlabelled_size, embedding_size), dtype=torch.float, device=self.device)
        pred_change = torch.zeros(unlabelled_size, dtype=torch.bool, device=self.device)

        if self.alpha_closed_form_approx:
            alpha_cap /= math.sqrt(embedding_size)
            grads = grads.to(self.device)
            
        for i in range(self.dataset.n_classes):
            emb = lb_embedding[Y == i]
            if emb.size(0) == 0:
                emb = lb_embedding
            anchor_i = emb.mean(dim=0).view(1, -1).repeat(unlabelled_size, 1)

            if self.alpha_closed_form_approx:
                embed_i, ulb_embed = anchor_i.to(self.device), ulb_embedding.to(self.device)
                alpha = self.calculate_optimum_alpha(alpha_cap, embed_i, ulb_embed, grads)

                embedding_mix = (1 - alpha) * ulb_embed + alpha * embed_i
                out, _ = self.model.backbone(embedding_mix, embedding=True)
                out = out.detach().cpu()
                alpha = alpha.cpu()

                pc = out.argmax(dim=1) != pred_1
            else:
                alpha = self.generate_alpha(unlabelled_size, embedding_size, alpha_cap).to(self.device)
                if self.alpha_opt:
                    alpha, pc = self.learn_alpha(ulb_embedding, pred_1, anchor_i, alpha, alpha_cap,
                                                 log_prefix=str(i))
                else:
                    embedding_mix = (1 - alpha) * ulb_embedding + alpha * anchor_i
                    out, _ = self.model.backbone(embedding_mix.to(self.device), embedding=True)                    
                    
                    out = out.detach().cpu()

                    pc = out.argmax(dim=1) != pred_1

            torch.cuda.empty_cache()

            alpha[~pc] = 1.
            pred_change[pc] = True
            is_min = min_alphas.norm(dim=1) > alpha.norm(dim=1)
            min_alphas[is_min] = alpha[is_min]
            
        return pred_change, min_alphas


    def calculate_optimum_alpha(self, eps, lb_embedding, ulb_embedding, ulb_grads):
        z = (lb_embedding - ulb_embedding)
        return (eps * z.norm(dim=1) / ulb_grads.norm(dim=1)).unsqueeze(dim=1).repeat(1, z.size(1)) * ulb_grads / (z + 1e-8)
         


    def sample(self, n, feats):
        feats = feats.numpy()
        cluster_learner = KMeans(n_clusters=n, n_init='auto')
        cluster_learner.fit(feats)

        cluster_idxs = cluster_learner.predict(feats)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (feats - centers) ** 2
        dis = dis.sum(axis=1)
        return np.array(
            [np.arange(feats.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(n) if
             (cluster_idxs == i).sum() > 0])


    def retrieve_anchor(self, embeddings, count):
        return embeddings.mean(dim=0).view(1, -1).repeat(count, 1)


    def generate_alpha(self, size, embedding_size, alpha_cap):
        alpha = torch.normal(
            mean=alpha_cap / 2.0,
            std=alpha_cap / 2.0,
            size=(size, embedding_size))

        alpha[torch.isnan(alpha)] = 1
        return self.clamp_alpha(alpha, alpha_cap)


    def clamp_alpha(self, alpha, alpha_cap):
        return torch.clamp(alpha, min=1e-8, max=alpha_cap)


    def learn_alpha(self, org_embed, labels, anchor_embed, alpha, alpha_cap, log_prefix=''):
        labels = labels.to(self.device)
        min_alpha = torch.ones(alpha.size(), dtype=torch.float, device=self.device)
        pred_changed = torch.zeros(labels.size(0), dtype=torch.bool, device=self.device)

        loss_func = torch.nn.CrossEntropyLoss(reduction='none')

        self.model.backbone.eval()

        for i in range(self.alpha_learning_iters):
            tot_nrm, tot_loss, tot_clf_loss = 0., 0., 0.
            for b in range(math.ceil(float(alpha.size(0)) / self.alpha_learn_batch_size)):
                self.model.backbone.zero_grad()
                start_idx = b * self.alpha_learn_batch_size
                end_idx = min((b + 1) * self.alpha_learn_batch_size, alpha.size(0))

                l = alpha[start_idx:end_idx].to(self.device)
                l.requires_grad_(True)
                opt = torch.optim.Adam([l], lr=self.alpha_learning_rate / (1. if i < self.alpha_learning_iters * 2 / 3 else 10.))
                e = org_embed[start_idx:end_idx].to(self.device)
                c_e = anchor_embed[start_idx:end_idx].to(self.device)
                embedding_mix = (1 - l) * e + l * c_e

                out, _ = self.model.backbone(embedding_mix, embedding=True)

                label_change = out.argmax(dim=1) != labels[start_idx:end_idx]

                tmp_pc = torch.zeros(labels.size(0), dtype=torch.bool, device=self.device)
                tmp_pc[start_idx:end_idx] = label_change
                pred_changed[start_idx:end_idx] += tmp_pc[start_idx:end_idx].detach().cpu()

                tmp_pc[start_idx:end_idx] = tmp_pc[start_idx:end_idx] * (l.norm(dim=1) < min_alpha[start_idx:end_idx].norm(dim=1).to(self.device))
                min_alpha[tmp_pc] = l[tmp_pc[start_idx:end_idx]].detach().cpu()

                clf_loss = loss_func(out, labels[start_idx:end_idx].to(self.device))

                l2_nrm = torch.norm(l, dim=1)

                clf_loss *= -1

                loss = self.alpha_clf_coef * clf_loss + self.alpha_l2_coef * l2_nrm
                loss.sum().backward(retain_graph=True)
                opt.step()

                l = self.clamp_alpha(l, alpha_cap)

                alpha[start_idx:end_idx] = l.detach().cpu()

                tot_clf_loss += clf_loss.mean().item() * l.size(0)
                tot_loss += loss.mean().item() * l.size(0)
                tot_nrm += l2_nrm.mean().item() * l.size(0)

                del l, e, c_e, embedding_mix
                torch.cuda.empty_cache()

        return min_alpha.cpu(), pred_changed.cpu()