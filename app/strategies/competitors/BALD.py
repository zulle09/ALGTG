
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
        """
        Evaluate the unlabelled training data using Monte Carlo dropout.
        This method performs inference on the unlabelled training dataset using
        multiple forward passes with dropout enabled (MC dropout). It collects
        the predicted probability distributions across all dropout iterations
        to compute the disagreement metric for the BALD acquisition strategy.
        
        Args:
            n_drop (int, optional): Number of dropout iterations to perform.
                Defaults to 5. More iterations provide better uncertainty
                estimates but increase computational cost.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - indices (torch.Tensor): The indices of the unlabelled samples
                  in the training dataset. Shape: (num_samples,).
                - prob_dist_drop (torch.Tensor): Predicted probability
                  distributions for each dropout iteration. Shape:
                  (n_drop, num_samples, num_classes).
        
        Raises:
            RuntimeError: If the model checkpoint cannot be loaded.
        
        Note:
            This method sets the backbone to training mode to enable dropout
            during inference (MC dropout). The model should have dropout layers
            in the backbone for this to work correctly.
        
        Example:
            >>> indices, probs = strategy.evaluate_unlabelled_train(n_drop=10)
            >>> print(f"Evaluated {len(indices)} samples with 10 dropout passes")
        """
        
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
        """
        Computes disagreement score using dropout-based uncertainty estimation.
        This method evaluates the disagreement between predictions across different
        dropout masks to estimate uncertainty. It calculates the difference between
        the average entropy of individual predictions and the entropy of the average
        prediction, which serves as a measure of model disagreement.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - indices: Indices of the unlabelled training samples.
                - Tensor: Disagreement scores (entropy2 - entropy1) for each sample.
                    Higher values indicate greater uncertainty.
        
        Raises:
            RuntimeError: If CUDA memory allocation fails during computation.
        
        Example:
            >>> indices, disagreement = model.disagreement_dropout()
            >>> selected_indices = indices[disagreement > threshold]
        """
        indices, prob_dist_drop = self.evaluate_unlabelled_train()
        
        mean_pb = torch.mean(prob_dist_drop, dim=0)
        
        entropy1 = entropy(mean_pb)
        entropy2 = torch.mean(entropy(prob_dist_drop, dim=2), dim=0)
        
        del prob_dist_drop
        torch.cuda.empty_cache()
        
        return indices, entropy2 - entropy1
        
    
    
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
    