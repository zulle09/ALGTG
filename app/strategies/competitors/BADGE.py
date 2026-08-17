
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.distributions import Categorical
import random

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
            if torch.sum(D2) == 0.0:
                # Degenerate case (all remaining points coincide with a centre): the
                # k-means++ sampling distribution is undefined. Fill the rest with random
                # unpicked points instead of dropping into a debugger, which would hang a
                # headless/cluster run indefinitely.
                remaining = [j for j in range(len(self.embedds_dict["embedds"])) if j not in set(indsAll)]
                random.shuffle(remaining)
                indsAll.extend(remaining[:n_top_k_obs - len(indsAll)])
                break
            Ddist = (D2 ** 2) / torch.sum(D2 ** 2)
            customDist = Categorical(Ddist)
            ind = customDist.sample()
            # Rejection-sample an as-yet-unpicked centre, but bounded. Once many points have
            # collapsed onto the chosen centres their D2 is exactly 0, so the k-means++
            # distribution can carry no numerically representable mass on the unpicked points
            # and this loop would never terminate (observed: 106/128 points at D2 == 0, mass on
            # unpicked = -1e-07). Fall back to the deterministic limit of k-means++ — the
            # unpicked point farthest from its nearest centre.
            tries = 0
            while ind.item() in indsAll:
                if tries >= 1000:
                    masked = D2.clone()
                    masked[torch.tensor(indsAll, dtype=torch.long, device=D2.device)] = -1.
                    ind = torch.argmax(masked)
                    break
                ind = customDist.sample()
                tries += 1

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
            
        logger.info(' => Getting the unlabelled embeddings and logits')
        self.embedds_dict = {
            'embedds': torch.empty((0, self.model.backbone.get_embedding_dim()), dtype=torch.float32, device=self.device),
            'outs': torch.empty((0, self.dataset.n_classes), dtype=torch.float32, device=torch.device('cpu'))
        }

        self.load_best_checkpoint()

        self.get_embeddings(unlab_train_dl, self.embedds_dict)
        logger.info(' DONE\n')

        # Build BADGE gradient embeddings: g_x = (p - onehot(argmax p)) (x) z, flattened to (N, C*D).
        # The magnitude encodes uncertainty and the direction encodes diversity, which is what makes
        # this BADGE rather than plain k-means++ on penultimate features. Kept on CPU because C*D can
        # be large (e.g. 200*512 for tiny-imagenet).
        embs = self.embedds_dict["embedds"].cpu()                                          # (N, D)
        probs = F.softmax(self.embedds_dict["outs"], dim=1)                                 # (N, C)
        scale = probs - F.one_hot(probs.argmax(dim=1), num_classes=probs.size(1)).float()  # (N, C)
        self.embedds_dict["embedds"] = (scale.unsqueeze(2) * embs.unsqueeze(1)).view(embs.size(0), -1)  # (N, C*D)

        logger.info(' => Top K extraction using init_centers')
        topk_idx_obs = self.init_centers(n_top_k_obs)
        logger.info(' DONE\n')
        
        return topk_idx_obs, [self.rand_unlab_sample[id] for id in topk_idx_obs]
    