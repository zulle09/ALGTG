
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset#, RandomSampler

from ActiveLearner import ActiveLearner
from config import al_params

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)


class GTG(ActiveLearner):
    
    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any], gtg_p) -> None:
                
        if gtg_p["am_ts"] != None:
            
            if gtg_p["am_ts"][gtg_p["id_am_ts"]] == 'mean': str_tresh_strat = '_ts-mean'
            elif gtg_p["am_ts"][gtg_p["id_am_ts"]] == 'threshold': str_tresh_strat = f'_{gtg_p["am_ts"][gtg_p["id_am_ts"]]}_{gtg_p["am_t"]}'
            else: str_tresh_strat = ''
            
            strategy_name = f'{self.__class__.__name__}_{gtg_p["am_s"]}_{gtg_p["am"][gtg_p["id_am"]]}{str_tresh_strat}_es-{gtg_p["e_s"]}'
        else: strategy_name = f'{self.__class__.__name__}_{gtg_p["am_s"]}_{gtg_p["am"][gtg_p["id_am"]]}_es-{gtg_p["e_s"]}'
                
        super().__init__(ct_p, t_p, strategy_name)
        #self.perc_labelled_batch: int = gtg_p["plb"]
        self.batch_size_gtg_online: int = gtg_p["bsgtgo"]
        self.gtg_model = gtg_p["gtg_model"]
        if self.model.only_module_name != None: 
            self.model.added_module.define_idx_params(gtg_p["id_am_ts"], gtg_p["id_am"]) # type: ignore # -> for GTG module only
        


    def query(
            self, 
            sample_unlab_subset: Subset, 
            n_top_k_obs: int
        ) -> Tuple[List[int], List[int]]:
        """
        Query the top-k unlabelled observations using the GTG module.
        This method performs active learning query selection by computing predicted
        and true entropies for both labelled and unlabelled data using the GTG 
        (Guess That Grows) module. It selects the top-k unlabelled observations 
        based on predicted entropy scores.
        Args:
            sample_unlab_subset (Subset): The unlabelled data subset to query from.
            n_top_k_obs (int): The number of top observations to select.
        Returns:
            Tuple[List[int], List[int]]: A tuple containing:
                - List of indices of the top-k unlabelled observations selected based 
                    on predicted entropy.
                - List of the corresponding sample data for those indices.
        Raises:
            ValueError: If the model does not have an added module (GTG module).
        Note:
            - The method creates two DataLoaders: one for labelled data and one for 
                unlabelled data, which are iterated simultaneously.
            - It loads the best checkpoint before inference.
            - The GTG module is run in inference mode to compute predicted and true 
                entropies.
            - Only unlabelled entropies are saved for selection.
            - Debug information is logged including intersection ratio between 
                predicted and true top-k, and MSE losses.
        """
            
        # I need to create two dataloaders, one for the labelled data and one for the unlabelled data, 
        # since I need to pass them separately to the GTG module to get the predicted entropies and true entropies
        lab_train_dl = DataLoader(
            dataset=Subset(self.dataset.unlab_train_ds, self.labelled_indices), 
            batch_size=self.batch_size_gtg_online * self.iter, shuffle=False, pin_memory=True
        )
            
        unlab_train_dl = DataLoader(
            dataset=sample_unlab_subset,
            batch_size=self.batch_size_gtg_online * al_params["al_iters"], shuffle=False, pin_memory=True
        )
            
        pred = torch.empty(0, dtype=torch.float32, device=self.device)
        true = torch.empty(0, dtype=torch.float32, device=self.device)

        if len(lab_train_dl) != len(unlab_train_dl):
            logger.warning(
                f'GTG query dataloaders have different lengths: lab={len(lab_train_dl)} unlab={len(unlab_train_dl)}. '
                'zip() will stop at the shortest length.'
            )
            
        logger.info(' => Running GTG in inference mode...')
            
        self.load_best_checkpoint()
        if self.model.added_module == None:
            raise ValueError('The model does not have an added module')
        self.model.added_module.eval()
            
        with torch.inference_mode():
            # I need to iterate over both dataloaders at the same time, since I need to concatenate the features and embeddings of the labelled and unlabelled data before passing them to the GTG module
            for (_, lab_images, lab_labels), (_, unlab_images, unlab_labels) in zip(lab_train_dl, unlab_train_dl):

                # Get the outputs, features and embeddings of the labelled and unlabelled data                  
                lab_outs, lab_embedds = self.model.backbone(lab_images.to(self.device))
                lab_features = self.model.backbone.get_features()
                unlab_outs, unlab_embedds = self.model.backbone(unlab_images.to(self.device))
                unlab_features = self.model.backbone.get_features()

                # pass the concatenated features and embeddings to the GTG module to get the predicted entropies and true entropies
                (y_pred, y_true) ,_ = self.model.added_module(
                    features=[
                        torch.cat((lab_feature, unlab_feature), dim=0) for lab_feature, unlab_feature in zip(lab_features, unlab_features)
                    ], 
                    embedds=torch.cat((lab_embedds, unlab_embedds), dim=0), 
                    outs=torch.cat((lab_outs, unlab_outs), dim=0),
                    labels=torch.cat((lab_labels, unlab_labels), dim=0).to(self.device),
                    iteration=self.iter,
                    n_lab_obs=lab_images.shape[0]
                )
                    
                # save only the unalbelled entropies
                n_curr_lab = lab_images.shape[0]
                pred = torch.cat((pred, y_pred[n_curr_lab:]), dim=0)
                true = torch.cat((true, y_true[n_curr_lab:]), dim=0)
                    
        logger.info(' DONE\n')
            
        ############# DEBUG PORPUSE ############# 
        largest_topk = self.gtg_model != 'lstmbc'
            
        top_k_pred = torch.topk(pred, n_top_k_obs, largest=largest_topk).indices
        top_k_true = torch.topk(true, n_top_k_obs, largest=largest_topk).indices
            
        logger.info(torch.topk(pred, n_top_k_obs, largest=largest_topk))
        logger.info(torch.topk(true, n_top_k_obs, largest=largest_topk))
            
        intersection_count = len(set(top_k_pred.cpu().tolist()) & set(top_k_true.cpu().tolist()))
        intersection_ratio = intersection_count / len(top_k_pred)
            
        logger.info(f' => Intersection ratio between predicted and true entropies: {intersection_ratio}')
            
        logger.info(f'MSE true_entr: {F.mse_loss(pred[top_k_true], true[top_k_true])}')
        logger.info(f'MSE pred_entr: {F.mse_loss(pred[top_k_pred], true[top_k_pred])}')
        ############# DEBUG PORPUSE ############# 
            
            
        logger.info(f' => Extracting the Top-k unlabelled observations')
        overall_topk = top_k_pred.tolist()
        logger.info(' DONE\n')
            
        return overall_topk, [self.rand_unlab_sample[id] for id in overall_topk]
