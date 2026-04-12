
import torch
from torch.utils.data import DataLoader, Subset
from torch.distributions import Categorical
import pdb

from ActiveLearner import ActiveLearner

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)



class BADGE(ActiveLearner):
    
    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any]) -> None:
                
        super().__init__(ct_p, t_p, self.__class__.__name__)
    
    
    def init_centers(self, n_top_k_obs: int) -> List[int]:
        """
        Initialize cluster centers using a probabilistic approach.
        Implements k-means++ style initialization to select initial cluster
        centers from embedded data points. Selection is based on a weighted
        probability distribution where points with larger distances from
        existing centers have higher probabilities of being selected.
        
        Args:
            n_top_k_obs: The number of centers to initialize.
        
        Returns:
            List[int]: A list of indices representing the selected center
                indices from the embedding dictionary.
        
        Example:
            >>> centers = init_centers(n_top_k_obs=10)
            >>> # Returns list of 10 indices corresponding to selected centers
        """
        ind = torch.argmax(torch.norm(self.embedds_dict["embedds"], dim=1))
        mu = self.embedds_dict["embedds"][ind].unsqueeze(0)
        indsAll = [int(ind.item())]
        centInds = [0] * len(self.embedds_dict["embedds"])
        cent = 0
        D2 = torch.Tensor()
        while len(mu) < n_top_k_obs:
            if len(mu) == 1:
                D2 = torch.cdist(self.embedds_dict["embedds"], mu).ravel().float()
            else:
                newD = torch.cdist(self.embedds_dict["embedds"], mu[-1].unsqueeze(0)).ravel().float()
                for i in range(len(self.embedds_dict["embedds"])):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            if torch.sum(D2) == 0.0: pdb.set_trace()
            Ddist = (D2 ** 2) / torch.sum(D2 ** 2)
            customDist = Categorical(Ddist)
            ind = customDist.sample()
            while ind.item() in indsAll:
                ind = customDist.sample()
            
            mu = torch.cat((mu, self.embedds_dict["embedds"][ind].unsqueeze(0)))
            
            indsAll.append(int(ind.item()))
            cent += 1
            
        return indsAll



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
                        
        unlab_train_dl = DataLoader(
            sample_unlab_subset, batch_size=self.batch_size,
            shuffle=False, pin_memory=True
        )
            
        logger.info(' => Getting the unlabelled embeddings')
        self.embedds_dict = { 'embedds': torch.empty((0, self.model.backbone.get_embedding_dim()), dtype=torch.float32, device=self.device) }
        
        self.load_best_checkpoint()
        
        self.get_embeddings(unlab_train_dl, self.embedds_dict)
        logger.info(' DONE\n')
                        
        logger.info(' => Top K extraction using init_centers')
        topk_idx_obs = self.init_centers(n_top_k_obs)
        logger.info(' DONE\n')
        
        return topk_idx_obs, [self.rand_unlab_sample[id] for id in topk_idx_obs]
    