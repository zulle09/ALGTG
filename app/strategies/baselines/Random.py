
import random

from ActiveLearner import ActiveLearner

from torch.utils.data import Subset

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)


class Random(ActiveLearner):
        
    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any]) -> None:
        
        super().__init__(ct_p, t_p, self.__class__.__name__)
        
        
    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
        """Select random observations from an unlabelled subset for querying.

        Randomly samples a specified number of indices from the unlabelled subset.
        If the subset contains fewer observations than requested, all available
        indices are returned.

        Args:
            sample_unlab_subset (Subset): The subset of unlabelled data from which
                to sample observations.
            n_top_k_obs (int): The number of observations to sample (top K).

        Returns:
            Tuple[List[int], List[int]]: A tuple containing:
                - List[int]: Indices of the sampled items in the original 
                  unlabelled subset indices list.
                - List[int]: The sampled indices from the unlabelled subset.

        Note:
            If the number of available unlabelled indices is less than n_top_k_obs,
            all available indices will be returned without raising an error.
        """
        logger.info(f' => Sampling K unlabelled observations')
        if(len(sample_unlab_subset.indices) > n_top_k_obs):           
            sample = list(random.sample(list(sample_unlab_subset.indices), n_top_k_obs))
        else:
            sample = list(sample_unlab_subset.indices)
        logger.info(' DONE\n')
        return [list(sample_unlab_subset.indices).index(item) for item in sample], sample