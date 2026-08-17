
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
        """
        Select unlabelled observations using the Furthest First strategy.
        This method implements a greedy active learning selection algorithm that
        iteratively chooses the unlabelled sample that is furthest from the
        currently labelled set. It computes pairwise distances between unlabelled
        and labelled embeddings, then selects the point with the maximum minimum
        distance to the existing labelled points. After each selection, the
        minimum distances are updated to include the newly selected point.
        
        Args:
            n_top_k_obs (int): The number of unlabelled observations to select.
        
        Returns:
            List[int]: A list of indices corresponding to the selected unlabelled
                observations, ordered by the Furthest First selection strategy.
        
        Raises:
            ValueError: If n_top_k_obs is greater than the number of unlabelled
                embeddings available.
        
        Example:
            >>> strategy = SomeStrategy(...)
            >>> selected_indices = strategy.furthest_first(n_top_k_obs=10)
            >>> print(f"Selected {len(selected_indices)} observations")
        """
        unlabelled_size = self.unlab_embedds_dict["embedds"].size(0)
        if self.lab_embedds_dict["embedds"].size(0) == 0:
            min_dist = float('inf') * torch.ones(unlabelled_size)
        else:
            dist_ctr = torch.cdist(self.unlab_embedds_dict["embedds"], self.lab_embedds_dict["embedds"])
            min_dist, _ = torch.min(dist_ctr, dim=1)
        
        overall_topk = []

        while len(overall_topk) < n_top_k_obs:
            idx = torch.argmax(min_dist)
            overall_topk.append(int(idx.item()))
            dist_new_ctr = torch.cdist(self.unlab_embedds_dict["embedds"],
                                       self.unlab_embedds_dict["embedds"][idx].unsqueeze(0))
            min_dist = torch.min(min_dist, dist_new_ctr[:, 0])
            # Explicitly retire the chosen point: with duplicate embeddings several points tie at
            # distance 0 and argmax would hand back an index already in the batch, which later
            # fails as "list.remove(x): x not in list" when the labelled set is updated.
            min_dist[idx] = float('-inf')

        return overall_topk


    
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
    