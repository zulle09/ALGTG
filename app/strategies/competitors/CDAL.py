
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
        """Compute pairwise symmetric KL divergence distances between two sets of distributions.
        Calculates the symmetric KL divergence (also known as Jensen-Shannon divergence
        multiplied by a constant) between each row vector in tensor `a` and each row vector
        in tensor `b`. The symmetric KL divergence is computed as:
        0.5 * (KL(a || b_i) + KL(b_i || a))
        
        Args:
            a (torch.Tensor): A 2D tensor of shape (n, d) representing n distributions,
                each with d dimensions.
            b (torch.Tensor): A 2D tensor of shape (m, d) representing m distributions,
                each with d dimensions.
        
        Returns:
            torch.Tensor: A 2D tensor of shape (n, m) containing the pairwise symmetric
                KL divergence distances between each row in `a` and each row in `b`.
        
        Raises:
            ValueError: If tensors `a` and `b` have different number of dimensions.
            ValueError: If any element in `a` or `b` is zero or negative (KL divergence
                requires positive probabilities).
        
        Example:
            >>> a = torch.tensor([[0.1, 0.9], [0.5, 0.5]])
            >>> b = torch.tensor([[0.2, 0.8], [0.3, 0.7]])
            >>> dist = kl_pairwise_distances(a, b)
            >>> print(dist.shape)
            torch.Size([2, 2])
        """
        dist = torch.zeros((a.size(0), b.size(0)))

        # Softmax outputs can underflow to exactly 0 in float32 when logits are far apart.
        # log(0) = -inf then makes `p * log(p/q)` evaluate to 0 * -inf = NaN, and a NaN in the
        # distance makes argmax in furthest_first return the same index forever -> duplicate
        # selections -> "list.remove(x): x not in list" when the sets are updated.
        a = a.clamp(min=1e-12)
        b = b.clamp(min=1e-12)

        for i in range(b.size(0)):
            b_i = b[i]
            kl1 = a * torch.log(a / b_i)
            kl2 = b_i * torch.log(b_i / a)
            dist[:, i] = 0.5 * (torch.sum(kl1, dim=1)) + 0.5 * (torch.sum(kl2, dim=1))

        return dist


    def furthest_first(
        self,
        X: torch.Tensor,
        X_set: torch.Tensor,
        n_top_k_obs: int
    ) -> List[int]:
        """Selects the top-k most diverse samples using Furthest-First traversal.

        This method implements a greedy farthest-first algorithm to select
        diverse samples from X based on KL divergence distances. It starts
        with either the minimum distances to X_set or initializes with infinity,
        then iteratively selects the sample with maximum distance to all
        previously selected samples.

        Args:
            X: Tensor of shape (m, n_classes) containing unlabelled sample
                probabilities.
            X_set: Tensor of shape (k, n_classes) containing labelled sample
                probabilities.
            n_top_k_obs: The number of samples to select.

        Returns:
            List[int]: Indices of the selected top-k diverse samples.

        Raises:
            ValueError: If n_top_k_obs exceeds the number of samples in X.
        """

        m = X.size(0)
        if n_top_k_obs > m:
            raise ValueError(
                f"n_top_k_obs ({n_top_k_obs}) cannot exceed "
                f"the number of samples in X ({m})"
            )

        if X_set.size(0) == 0:
            min_dist = float('inf') * torch.ones(m)
        else:
            dist_ctr = self.kl_pairwise_distances(X, X_set)
            min_dist, _ = torch.min(dist_ctr, dim=1)

        overall_topk = []

        while len(overall_topk) < n_top_k_obs:
            idx = torch.argmax(min_dist)
            overall_topk.append(int(idx.item()))
            dist_new_ctr = self.kl_pairwise_distances(X, X[idx].unsqueeze(0))
            min_dist = torch.min(min_dist, dist_new_ctr[:, 0])
            # Explicitly retire the chosen point. Relying on "distance to itself is 0" breaks
            # whenever several points tie at the running minimum, which would re-select an index
            # already in the batch.
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
    