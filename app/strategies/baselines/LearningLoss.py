
import torch
from torch.utils.data import DataLoader, Subset

from ActiveLearner import ActiveLearner

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)


class LearningLoss(ActiveLearner):
    
    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any]) -> None:
        
        super().__init__(ct_p, t_p, self.__class__.__name__)
                
                
    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
        """
        Select the top-k most informative unlabeled observations for annotation.
        This method is part of an active learning strategy. It evaluates the model's
        embeddings on the unlabeled subset and selects the samples with the highest
        scores (based on the module output) to be labeled next.
        
        Args:
            sample_unlab_subset (Subset): A PyTorch Subset containing the unlabeled
                training data to be evaluated.
            n_top_k_obs (int): The number of top observations to select from the
                unlabeled subset.
        
        Returns:
            Tuple[List[int], List[int]]: A tuple containing:
                - List of indices (from the unlabeled subset) corresponding to the
                  top-k observations with highest scores.
                - List of corresponding random unlabeled samples associated with
                  each selected index.
        
        Example:
            >>> top_indices, top_samples = query(unlab_subset, n_top_k_obs=10)
            >>> print(f"Selected {len(top_indices)} observations")
        """
        self.unlab_train_dl = DataLoader(
            sample_unlab_subset,
            batch_size=self.batch_size, shuffle=False, pin_memory=True
        )
                
        logger.info(' => Evaluating unlabelled observations')
        embeds_dict = { 'module_out': torch.empty(0, dtype=torch.float32) }
        
        self.load_best_checkpoint()
        
        self.get_embeddings(self.unlab_train_dl, embeds_dict)
        
        logger.info(f' => Extracting the Top-k unlabelled observations')
        overall_topk = torch.topk(embeds_dict["module_out"], n_top_k_obs).indices.tolist()
        logger.info(' DONE\n')
        
        return overall_topk, [self.rand_unlab_sample[id] for id in overall_topk]