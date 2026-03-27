
from ActiveLearner import ActiveLearner
from utils import plot_tsne_A, create_directory, entropy, plot_gtg_entropy_tensor

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

import gc
from typing import Dict, Any, List, Tuple
    
import logging
logger = logging.getLogger(__name__)


class GTG_off(ActiveLearner):
    
    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any], gtg_p: Dict[str, Any]) -> None:
        
        self.get_A_fn = {
            'cos_sim': self.get_A_cos_sim,
            'corr': self.get_A_corr,
            'rbfk': self.get_A_rbfk,
        }

        self.gtg_tol: float = gtg_p["gtg_t"]
        self.gtg_max_iter: int = gtg_p["gtg_i"]
        
        self.AM_function: str = gtg_p["am"][gtg_p["id_am"]]
        self.AM_strategy: str = gtg_p["am_s"]
        self.AM_threshold_strategy: str = gtg_p["am_ts"][gtg_p["id_am_ts"]]
        self.AM_threshold: float = gtg_p["am_t"]
        
        self.AM_mixed_distance = gtg_p["am_md"]
        self.AM_self_loop = gtg_p["am_sl"]
        
        self.ent_strategy: str = gtg_p["e_s"]

        if self.AM_threshold_strategy != 'none':
            
            if self.AM_threshold_strategy == 'mean': str_tresh_strat = '_ts-mean'
            elif self.AM_threshold_strategy == 'threshold': str_tresh_strat = f'_{self.AM_threshold_strategy}_{self.AM_threshold}'
            else: str_tresh_strat = ''
            
            strategy_name = f'{self.__class__.__name__}_{self.AM_function}_{self.AM_strategy}{str_tresh_strat}_es-{self.ent_strategy}'
        else:
            strategy_name = f'{self.__class__.__name__}_{self.AM_function}_{self.AM_strategy}_es-{self.ent_strategy}'
                
        super().__init__(ct_p, t_p, strategy_name)
        
        create_directory(self.path + '/gtg_entropies_plots')
        
        
    
    def get_A_treshold(self, A: torch.Tensor) -> Any:
        if self.AM_threshold_strategy == 'mean': return torch.mean(A)
        else: return self.AM_threshold
    
        
    # select the relative choosen affinity matrix method
    def get_A(self) -> None:
        
        logger.info(' => Computing Affinity Matrix...')

        concat_embedds = torch.cat((self.lab_embedds_dict["embedds"], self.unlab_embedds_dict["embedds"]))
                
        # compute the affinity matrix
        A = self.get_A_fn[self.AM_function](concat_embedds, to_cpu=True)
    
        initial_A = torch.clone(A)
               
        if self.AM_threshold_strategy != 'none':
            # remove weak connections with the choosen threshold strategy and value
            logger.info(f' Affinity Matrix Threshold to be used: {self.AM_threshold_strategy} -> {self.get_A_treshold(A) if self.AM_threshold_strategy == "mean" else self.AM_threshold}')
                
            if self.AM_function != 'rbfk': A = torch.where(A < self.get_A_treshold(A), 0, A)
            else: A = torch.where(A > self.get_A_treshold(A), 1, A)
        
        if self.AM_strategy == 'diversity':
            # set the whole matrix as a distance matrix and not similarity matrix
            A = 1 - A
        elif self.AM_strategy == 'mixed':    
            # set the unlabelled submatrix as distance matrix and not similarity matrix
            n_lab_obs = len(self.labelled_indices)
            
            # LL -> DISTANCE
            if self.AM_function == 'rbfk': A = 1 - A
            for AM_m_d in self.AM_mixed_distance:
                if AM_m_d == 'LL': A[:n_lab_obs, :n_lab_obs] = 1 - A[:n_lab_obs, :n_lab_obs] # LL -> DISTANCE
                elif AM_m_d == 'UU': A[n_lab_obs:, n_lab_obs:] = 1 - A[n_lab_obs:, n_lab_obs:] # UU -> DISTANCE
                elif AM_m_d == 'LU': A[:n_lab_obs, n_lab_obs:] = 1 - A[:n_lab_obs, n_lab_obs:] # LU -> DISTANCE
                else: A[n_lab_obs:, :n_lab_obs] = 1 - A[n_lab_obs:, :n_lab_obs] # UL -> DISTANCE
            
               
                
        # plot the TSNE fo the original and modified affinity matrix
        logger.info('Plotting TSNE of the original and modified Affinity Matrix...')
        plot_tsne_A(
            (initial_A, A),
            (self.lab_embedds_dict["labels"], self.unlab_embedds_dict["labels"]), self.dataset.classes,
            self.ct_p["exp_path"], self.ct_p["dataset_name"], self.ct_p["trial"], self.strategy_name, self.AM_function, self.AM_strategy, self.iter
        )
        
        self.A = A.to(self.device)
        logger.info(' DONE\n')


    def get_A_rbfk(self, concat_embedds: torch.Tensor, to_cpu = False) -> torch.Tensor:
        
        device = torch.device('cpu') if to_cpu else self.device
        A_matrix = self.get_A_e_d(concat_embedds, to_cpu)
        seventh_neigh = concat_embedds[torch.argsort(A_matrix, dim=1)[:, 6]]            
            
        sigmas = torch.unsqueeze(torch.tensor([
            self.get_A_fn[self.AM_function](torch.cat((
                torch.unsqueeze(concat_embedds[i], dim=0), torch.unsqueeze(seventh_neigh[i], dim=0)
            )), to_cpu=to_cpu)[0,1].item()
            for i in range(concat_embedds.shape[0]) 
        ], device=device), dim=0)
        
        A = torch.exp(-A_matrix.pow(2) / (torch.mm(sigmas.T, sigmas))).to(device)
        A = torch.clamp(A, min=0., max=1.)
        
        return A.fill_diagonal_(0. if self.AM_self_loop else 1.)
    
    
    def get_A_e_d(self, concat_embedds: torch.Tensor, to_cpu = False) -> torch.Tensor:
        A = torch.cdist(concat_embedds, concat_embedds).to(torch.device('cpu') if to_cpu else self.device)
        return torch.clamp(A, min=0., max=1.).fill_diagonal_(0. if self.AM_self_loop else 1.)


    def get_A_cos_sim(self, concat_embedds: torch.Tensor, to_cpu = False) -> torch.Tensor:
        device = torch.device('cpu') if to_cpu else self.device
        normalized_embedding = F.normalize(concat_embedds, dim=-1).to(device)
        
        A = torch.matmul(normalized_embedding, normalized_embedding.transpose(-1, -2)).to(device)
        A = torch.clamp(A, min=0., max=1.)

        return A.fill_diagonal_(1. if self.AM_self_loop else 0.)
        
        
    def get_A_corr(self, concat_embedds: torch.Tensor, to_cpu = False) -> torch.Tensor:
        A = torch.corrcoef(concat_embedds).to(torch.device('cpu') if to_cpu else self.device)
        return torch.clamp(A, min=0., max=1.).fill_diagonal_(1. if self.AM_self_loop else 0.)
        

    def get_X(self) -> None:
        logger.info(' => Computing X Matrix...')
        
        self.X: torch.Tensor = torch.zeros(
            (len(self.labelled_indices) + self.len_unlab_sample, self.dataset.n_classes),
            dtype=torch.float32, device=self.device
        )
        
        # Set the labels for the labelled samples
        for idx, label in enumerate(self.lab_embedds_dict["labels"]): self.X[idx][int(label.item())] = 1.
        
        # Set the initial probabilities for the unlabelled samples
        self.X[
            torch.arange(len(self.labelled_indices), len(self.labelled_indices) + self.len_unlab_sample), :
        ] = torch.ones(self.dataset.n_classes, device=self.device) * (1. / self.dataset.n_classes)
        
        
        logger.info(' DONE\n')
        
        
        
    def graph_trasduction_game(self) -> None:
        
        self.get_A()
        self.get_X()
                
        n_lab_obs = len(self.lab_embedds_dict["embedds"])
                        
        err = float('Inf')
        i = 0
        
        logger.info(' => Running GTG')
        logger.info(f'starting X: {self.X.argmax(dim=1)}')
        
        while err > self.gtg_tol and i < self.gtg_max_iter:
            X_old = torch.clone(self.X)
            
            self.X *= torch.mm(self.A, self.X)
            self.X /= torch.sum(self.X, dim=1, keepdim=True)
            
            logger.info(f'X: {self.X[len(self.labelled_indices):].argmax(dim=1).unique(return_counts=True)}')
        
            self.unlab_entropy_hist[:, i] = entropy(self.X)[len(self.labelled_indices):]
                        
            err = torch.norm(self.X - X_old)
            i += 1
            
        logger.info(f'end X: {self.X[len(self.labelled_indices):].argmax(dim=1).unique(return_counts=True)}')
        logger.info(f'GTG accuracy: {self.X[n_lab_obs:].cpu().argmax(dim=1).eq(self.unlab_embedds_dict["labels"]).sum().item() / len(self.unlab_embedds_dict["embedds"])}')
        
        logger.info(' DONE\n')
        
        


    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:            
        self.len_unlab_sample = len(sample_unlab_subset)

        dl_dict = dict(batch_size=self.batch_size, shuffle=False, pin_memory=True)           
        
        unlab_train_dl = DataLoader(sample_unlab_subset, **dl_dict)
        lab_train_dl = DataLoader(Subset(self.dataset.unlab_train_ds, self.labelled_indices), **dl_dict)
                        
        logger.info(' => Getting the labelled and unlabelled embeddings')
        self.lab_embedds_dict = {
            'embedds': torch.empty((0, self.model.backbone.get_embedding_dim()), dtype=torch.float32, device=torch.device('cpu')),
            'labels': torch.empty(0, dtype=torch.int8, device=torch.device('cpu'))
        }
        self.unlab_embedds_dict = {
            'embedds': torch.empty((0, self.model.backbone.get_embedding_dim()), dtype=torch.float32, device=torch.device('cpu')),
            'labels': torch.empty(0, dtype=torch.int8, device=torch.device('cpu'))
        }        
        self.load_best_checkpoint()
                
        self.get_embeddings(lab_train_dl, self.lab_embedds_dict, embedds2cpu=True)
        self.get_embeddings(unlab_train_dl, self.unlab_embedds_dict, embedds2cpu=True)
        logger.info(' DONE\n')
        
        
        # I save the entropy history in order to be able to plot it
        self.unlab_entropy_hist = torch.zeros((self.len_unlab_sample, self.gtg_max_iter), device=self.device)
                
        logger.info(' => Execution of the Graph Trasduction Game')
        self.graph_trasduction_game()
        logger.info(' DONE\n')
                        
        # TRUE / FALSE np.ndarray
        self.gtg_result_prediction = (torch.argmax(self.X[len(self.labelled_indices):], dim=1).cpu() == self.unlab_embedds_dict["labels"]).numpy()

        
        del self.A
        del self.X
        gc.collect()
        torch.cuda.empty_cache()
        
                            
        logger.info(f' => Extracting the Top-k unlabelled observations using {self.ent_strategy}')
        
        if self.ent_strategy not in ['mean', 'integral']:
            logger.exception('Unrecognized derivates computation strategy')
            raise Exception('Unrecognized derivates computation strategy')
        
        if self.ent_strategy == 'mean': # computing the mean of the entropis history
            entropy_measures = torch.mean(self.unlab_entropy_hist, dim=1)
          
        else: # computing the area of the entropis history using trapezoid formula 
            entropy_measures = torch.trapezoid(self.unlab_entropy_hist, dim=1)
           
        overall_topk = torch.topk(entropy_measures, k=n_top_k_obs).indices.tolist()
                
        # plot entropy history tensor
        plot_gtg_entropy_tensor(
            tensor=self.unlab_entropy_hist, topk=overall_topk, lab_unlabels=self.unlab_embedds_dict["labels"].tolist(), classes=self.dataset.classes, 
            path=self.path, iter=self.iter - 1, max_x=self.gtg_max_iter, dir='history'
        )
                    
        del self.unlab_entropy_hist
        gc.collect()
        torch.cuda.empty_cache() 
        
        logger.info(' DONE\n')
        
        return overall_topk, [self.rand_unlab_sample[id] for id in overall_topk]
    