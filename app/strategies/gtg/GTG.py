
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset#, RandomSampler

from ActiveLearner import ActiveLearner
from config import al_params

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)


class GTG(ActiveLearner):
    
    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any], gtg_p) -> None:
                
        if gtg_p["am_ts"] != None:
            
            if gtg_p["am_ts"][gtg_p["id_am_ts"]] == 'mean': str_tresh_strat = '_ts-mean'
            elif gtg_p["am_ts"][gtg_p["id_am_ts"]] == 'threshold': str_tresh_strat = f'_{gtg_p["am_ts"][gtg_p["id_am_ts"]]}_{gtg_p["am_t"]}'
            else: str_tresh_strat = ''
            
            strategy_name = f'{self.__class__.__name__}_{gtg_p["am_s"]}_{gtg_p["am"][gtg_p["id_am"]]}{str_tresh_strat}_es-{gtg_p["e_s"]}'
        else: strategy_name = f'{self.__class__.__name__}_{gtg_p["am_s"]}_{gtg_p["am"][gtg_p["id_am"]]}_es-{gtg_p["e_s"]}'
                
        super().__init__(ct_p, t_p, strategy_name)
        #self.perc_labelled_batch: int = gtg_p["plb"]
        self.batch_size_gtg_online: int = gtg_p["bsgtgo"]
        self.gtg_model = gtg_p["gtg_model"]
        if self.model.only_module_name != None: self.model.added_module.define_idx_params(gtg_p["id_am_ts"], gtg_p["id_am"]) # type: ignore # -> for GTG module only
        


    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
            
        lab_query_bs = self.batch_size_gtg_online * self.iter
            
        lab_train_dl = DataLoader(
            dataset=Subset(self.dataset.unlab_train_ds, self.labelled_indices), 
            batch_size=lab_query_bs, shuffle=False, pin_memory=True
        )
            
        unlab_train_dl = DataLoader(
            dataset=sample_unlab_subset,
            batch_size=self.batch_size_gtg_online * al_params["al_iters"], shuffle=False, pin_memory=True
        )
            
        pred = torch.empty(0, dtype=torch.float32, device=self.device)
        true = torch.empty(0, dtype=torch.float32, device=self.device)
            
        logger.info(' => Running GTG in inference mode...')
            
        self.load_best_checkpoint()
        if self.model.added_module == None:
            raise ValueError('The model does not have an added module')
        self.model.added_module.eval()
            
        with torch.inference_mode():
                
            for (_, lab_images, lab_labels), (_, unlab_images, unlab_labels) in zip(lab_train_dl, unlab_train_dl):
                                                
                lab_outs, lab_embedds = self.model.backbone(lab_images.to(self.device))
                lab_features = self.model.backbone.get_features()
                unlab_outs, unlab_embedds = self.model.backbone(unlab_images.to(self.device))
                unlab_features = self.model.backbone.get_features()

                (y_pred, y_true) ,_ = self.model.added_module(
                    features=[
                        torch.cat((lab_feature, unlab_feature), dim=0) for lab_feature, unlab_feature in zip(lab_features, unlab_features)
                    ], 
                    embedds=torch.cat((lab_embedds, unlab_embedds), dim=0), 
                    outs=torch.cat((lab_outs, unlab_outs), dim=0),
                    labels=torch.cat((lab_labels, unlab_labels), dim=0).to(self.device),
                    iteration=self.iter
                )
                    
                # save only the unalbelled entropies
                pred = torch.cat((pred, y_pred[lab_query_bs:]), dim=0) 
                true = torch.cat((true, y_true[lab_query_bs:]), dim=0)
                    
        logger.info(' DONE\n')
            
        ############# DEBUG PORPUSE ############# 
        largest_topk = self.gtg_model != 'lstmbc'
            
        top_k_pred = torch.topk(pred, n_top_k_obs, largest=largest_topk).indices
        top_k_true = torch.topk(true, n_top_k_obs, largest=largest_topk).indices
            
        logger.info(torch.topk(pred, n_top_k_obs, largest=largest_topk))
        logger.info(torch.topk(true, n_top_k_obs, largest=largest_topk))
            
        intersection_count = len(set(top_k_pred.cpu().tolist()) & set(top_k_true.cpu().tolist()))
        intersection_ratio = intersection_count / len(top_k_pred)
            
        logger.info(f' => Intersection ratio between predicted and true entropies: {intersection_ratio}')
            
        logger.info(f'MSE true_entr: {F.mse_loss(pred[top_k_true], true[top_k_true])}')
        logger.info(f'MSE pred_entr: {F.mse_loss(pred[top_k_pred], true[top_k_pred])}')
        ############# DEBUG PORPUSE ############# 
            
            
        logger.info(f' => Extracting the Top-k unlabelled observations')
        overall_topk = top_k_pred.tolist()
        ######################################################
        #overall_topk = top_k_true.tolist() # -> DEBUB PORPUSE
        ######################################################
        logger.info(' DONE\n')
            
        return overall_topk, [self.rand_unlab_sample[id] for id in overall_topk]
        
        
    '''def query_llmlp(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
        self.unlab_train_dl = DataLoader(sample_unlab_subset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
                
        logger.info(' => Evaluating unlabelled observations')
        embeds_dict = { 'module_out': torch.empty(0, dtype=torch.float32) }
        
        self.load_best_checkpoint()
        
        self.get_embeddings(self.unlab_train_dl, embeds_dict)
        
        logger.info(f' => Extracting the Top-k unlabelled observations')
        overall_topk = torch.topk(embeds_dict["module_out"], n_top_k_obs).indices.tolist()
        logger.info(' DONE\n')
        
        return overall_topk, [self.rand_unlab_sample[id] for id in overall_topk]'''


        #if self.gtg_model == 'llmlp':
        #    return self.query_llmlp(sample_unlab_subset, n_top_k_obs)
        #else:
        
    '''query_bs = int(self.batch_size * self.perc_labelled_batch)
            
            unlab_train_dl = DataLoader(sample_unlab_subset, batch_size=query_bs, shuffle=False, pin_memory=True)
            labelled_subset = Subset(self.dataset.unlab_train_ds, self.labelled_indices)
            lab_train_dl = DataLoader(
                dataset=labelled_subset, batch_size=query_bs,
                sampler=RandomSampler(labelled_subset, num_samples=len(sample_unlab_subset), replacement=False), 
                #random sample with replacement, each batch has different set of labelled observation drown ad random from the entire set
            )'''