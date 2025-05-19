
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torch.distributed as dist

from ActiveLearner import ActiveLearner
from utils import entropy

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)



class BALD(ActiveLearner):
    
    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any]) -> None:
        
        super().__init__(ct_p, t_p, self.__class__.__name__)
        
        
        
    def evaluate_unlabelled_train(self, n_drop=5) -> Tuple[torch.Tensor, torch.Tensor]:
        
        self.load_best_checkpoint()

        self.model.backbone.train()
        
        prob_dist_drop = torch.zeros((n_drop, len(self.unlab_train_dl.dataset), self.dataset.n_classes), dtype=torch.float32, device=self.device)  # type: ignore
        indices = torch.empty(0, device=self.device) 
        
        for drop in range(n_drop):
            with torch.inference_mode(): # Allow inference mode
                for idx_dl, (idxs, images, _) in enumerate(self.unlab_train_dl):
                    
                    idxs, images = idxs.to(self.device), images.to(self.device)
                    
                    outputs, _ = self.model.backbone(images)
                                                         
                    prob_dist_drop[drop][idx_dl * idxs.shape[0] : (idx_dl + 1) * idxs.shape[0]] += F.softmax(outputs, dim=1)

                    if(drop == 0): indices = torch.cat((indices, idxs), dim = 0)
                    
        return indices, prob_dist_drop 
        
    

    def disagreement_dropout(self) -> Tuple[torch.Tensor, torch.Tensor]:
        indices, prob_dist_drop = self.evaluate_unlabelled_train()
        
        mean_pb = torch.mean(prob_dist_drop, dim=0)
        
        entropy1 = entropy(mean_pb)
        entropy2 = torch.mean(entropy(prob_dist_drop, dim=2), dim=0)
        
        del prob_dist_drop
        torch.cuda.empty_cache()
        
        return indices, entropy2 - entropy1
        
    
    
    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
                        
        self.unlab_train_dl = DataLoader(
            sample_unlab_subset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True
        )
            
        logger.info(' => Performing the disagreement dropout')
        indices, res_entropy = self.disagreement_dropout()
        logger.info(' DONE\n')
            
        overall_topk = torch.topk(res_entropy, n_top_k_obs).indices.tolist()
        
        del res_entropy
        torch.cuda.empty_cache()
                
        return overall_topk, [int(indices[id].item()) for id in overall_topk]
    