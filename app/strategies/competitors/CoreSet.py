
import torch
from torch.utils.data import DataLoader, Subset

from ActiveLearner import ActiveLearner

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)


class CoreSet(ActiveLearner):
    
    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any]) -> None:
        
        super().__init__(ct_p, t_p, self.__class__.__name__)
        
    
    def furthest_first(self, n_top_k_obs: int) -> List[int]:
        unlabelled_size = self.unlab_embedds_dict["embedds"].size(0)
        if self.lab_embedds_dict["embedds"].size(0) == 0:
            min_dist = float('inf') * torch.ones(unlabelled_size)
        else:
            dist_ctr = torch.cdist(self.unlab_embedds_dict["embedds"], self.lab_embedds_dict["embedds"])
            min_dist, _ = torch.min(dist_ctr, dim=1)
        
        overall_topk = []

        while len(overall_topk) < n_top_k_obs:
            idx = torch.argmax(min_dist)
            overall_topk.append(idx.item())
            dist_new_ctr = torch.cdist(self.unlab_embedds_dict["embedds"],
                                       self.unlab_embedds_dict["embedds"][idx].unsqueeze(0))
            min_dist = torch.min(min_dist, dist_new_ctr[:, 0])

        return overall_topk


    
    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
        
        dl_dict = dict( batch_size=self.batch_size, shuffle=False, pin_memory=True )
            
        unlab_train_dl = DataLoader(sample_unlab_subset, **dl_dict)
        lab_train_dl = DataLoader(Subset(self.dataset.unlab_train_ds, self.labelled_indices), **dl_dict)
            
        logger.info(' => Getting the labelled and unlabelled embeddings')
        self.lab_embedds_dict = { 'embedds': torch.empty((0, self.model.backbone.get_embedding_dim()), dtype=torch.float32, device=self.device) }
        self.unlab_embedds_dict = { 'embedds': torch.empty((0, self.model.backbone.get_embedding_dim()), dtype=torch.float32, device=self.device) }
        
        self.load_best_checkpoint()
            
        self.get_embeddings(lab_train_dl, self.lab_embedds_dict)
        self.get_embeddings(unlab_train_dl, self.unlab_embedds_dict)
        logger.info(' DONE\n')
                        
        logger.info(' => Top K extraction')
        topk_idx_obs = self.furthest_first(n_top_k_obs)
        logger.info(' DONE\n')
        
        del self.lab_embedds_dict
        torch.cuda.empty_cache()
        
        return topk_idx_obs, [self.rand_unlab_sample[id] for id in topk_idx_obs]
    