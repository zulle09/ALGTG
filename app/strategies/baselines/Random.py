
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
        logger.info(f' => Sampling K unlabelled observations')
        if(len(sample_unlab_subset.indices) > n_top_k_obs):           
            sample = list(random.sample(list(sample_unlab_subset.indices), n_top_k_obs))
        else:
            sample = list(sample_unlab_subset.indices)
        logger.info(' DONE\n')
        return [list(sample_unlab_subset.indices).index(item) for item in sample], sample