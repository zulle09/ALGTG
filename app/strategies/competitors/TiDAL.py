
import torch
from torch.utils.data import DataLoader, Subset

from ActiveLearner import ActiveLearner
from utils import entropy

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)


class TiDAL(ActiveLearner):
    
    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any]) -> None:
        
        super().__init__(ct_p, t_p, self.__class__.__name__)
                
                
    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
        """
        Query the most informative unlabelled observations for active learning.
        This method selects the top-k unlabelled observations with the highest
        entropy (i.e., most uncertain predictions) from the unlabelled subset.
        It loads the best model checkpoint, computes embeddings for the unlabelled
        data, calculates softmax probabilities, and identifies the observations
        with maximum uncertainty based on entropy.
        
        Args:
            sample_unlab_subset: A Subset containing the unlabelled observations
                to query from.
            n_top_k_obs: The number of top-k most uncertain observations to select.
        
        Returns:
            Tuple[List[int], List[int]]: A tuple containing:
                - List[int]: Indices of the top-k most uncertain unlabelled observations.
                - List[int]: Corresponding random unlabelled samples for the selected indices.
        Example:
            >>> topk_indices, rand_samples = query(unlab_subset, n_top_k_obs=10)
        """

        self.unlab_train_dl = DataLoader(
            sample_unlab_subset,
            batch_size=self.batch_size, shuffle=False, pin_memory=True
        )
                
        logger.info(' => Evaluating unlabelled observations')
        embeds_dict = { 'module_out': torch.empty((0, self.dataset.n_classes), dtype=torch.float32, device=torch.device('cpu')) }
        
        self.load_best_checkpoint()
        
        self.get_embeddings(self.unlab_train_dl, embeds_dict)
        sub_prob = torch.softmax(embeds_dict["module_out"], dim=1)
        logger.info(' DONE\n')
        
        logger.info(f' => Extracting the Top-k unlabelled observations')
        overall_topk = torch.topk(entropy(sub_prob), n_top_k_obs).indices.tolist()
        logger.info(' DONE\n')
        
        return overall_topk, [self.rand_unlab_sample[id] for id in overall_topk]