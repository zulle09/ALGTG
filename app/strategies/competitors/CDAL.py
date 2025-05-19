
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ActiveLearner import ActiveLearner

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)



class CDAL(ActiveLearner):
    
    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any]) -> None:
        self.strategy_name = self.__class__.__name__
        
        super().__init__(ct_p, t_p, self.__class__.__name__)
        
        
    def kl_pairwise_distances(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        dist = torch.zeros((a.size(0), b.size(0)))
        
        for i in range(b.size(0)):
            b_i = b[i]
            kl1 = a * torch.log(a / b_i)
            kl2 = b_i * torch.log(b_i / a)
            dist[:, i] = 0.5 * (torch.sum(kl1, dim=1)) + 0.5 * (torch.sum(kl2, dim=1))
            
        return dist


    def furthest_first(self, X: torch.Tensor, X_set: torch.Tensor, n_top_k_obs: int) -> List[int]:
        m = X.size(0)
        if X_set.size(0) == 0:
            min_dist = float('inf') * torch.ones(m)
        else:
            dist_ctr = self.kl_pairwise_distances(X, X_set)
            min_dist, _ = torch.min(dist_ctr, dim=1)
        
        overall_topk = []

        while len(overall_topk) < n_top_k_obs:
            idx = torch.argmax(min_dist)
            overall_topk.append(idx.item())
            dist_new_ctr = self.kl_pairwise_distances(X, X[idx].unsqueeze(0))
            min_dist = torch.min(min_dist, dist_new_ctr[:, 0])

        return overall_topk


    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
                                
        dl_dict = dict( batch_size=self.batch_size, shuffle=False, pin_memory=True )
            
        unlab_train_dl = DataLoader(sample_unlab_subset, **dl_dict)
        lab_train_dl = DataLoader(Subset(self.dataset.unlab_train_ds, self.labelled_indices), **dl_dict)
            
        logger.info(' => Getting the labelled and unlabelled logits')
        lab_embedds_dict = { 'outs': torch.empty((0, self.dataset.n_classes), dtype=torch.float32, device=torch.device('cpu')) }
        unlab_embedds_dict = { 'outs': torch.empty((0, self.dataset.n_classes), dtype=torch.float32, device=torch.device('cpu')) }
        
        self.load_best_checkpoint()
        
        self.get_embeddings(lab_train_dl, lab_embedds_dict)
        self.get_embeddings(unlab_train_dl, unlab_embedds_dict)
        
        lab_probs = F.softmax(lab_embedds_dict["outs"], dim=1)
        unlab_probs = F.softmax(unlab_embedds_dict["outs"], dim=1)
        
        logger.info(' DONE\n')
        
        topk_idx_obs = self.furthest_first(unlab_probs, lab_probs, n_top_k_obs)
                    
        
        return topk_idx_obs, [self.rand_unlab_sample[id] for id in topk_idx_obs]
    