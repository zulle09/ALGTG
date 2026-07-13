
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
        """
        Calculate the threshold value for tensor A.

        Determines the threshold based on the configured threshold strategy.
        If the strategy is set to 'mean', computes the mean of the input tensor.
        Otherwise, returns the predefined AM_threshold value.

        Args:
            A (torch.Tensor): Input tensor for which to calculate the threshold.

        Returns:
            Any: The threshold value. Either the mean of tensor A if strategy
                is 'mean', or self.AM_threshold otherwise.

        Raises:
            Exception: Any exceptions raised by torch.mean for invalid inputs.
        """
        if self.AM_threshold_strategy == 'mean': return torch.mean(A)
        else: return self.AM_threshold
    
        
    # select the relative choosen affinity matrix method
    def get_A(self) -> None:
        """
        Compute and process the affinity matrix for graph-based semi-supervised learning.
        This method constructs an affinity matrix by combining labeled and unlabeled
        embeddings, applies thresholding to remove weak connections, and optionally
        converts similarity to distance based on the specified strategy.
        The processing pipeline includes:
            1. Concatenation of labeled and unlabeled embeddings
            2. Computation of the affinity matrix using the configured function
            3. Optional thresholding to remove weak connections
            4. Optional conversion to distance matrix based on strategy
        The affinity matrix can be processed in different modes:
            - 'diversity': Converts entire matrix to distance (1 - similarity)
            - 'mixed': Converts specific submatrices (LL, UU, LU, UL) to distance
        Finally, generates t-SNE visualizations comparing the original and processed
        affinity matrices.
        
        Returns:
            None: The computed affinity matrix is stored in self.A.
        
        Raises:
            ValueError: If an invalid AM_threshold_strategy or AM_strategy is provided.
        
        Note:
            - The threshold value is determined by get_A_threshold() method when
              AM_threshold_strategy is 'mean', otherwise uses self.AM_threshold
            - For 'rbfk' function, thresholding logic is inverted (values > threshold
              become 1, else 0)
        """
        
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
            
            # corr, cos_sim and rbfk all return a similarity in [0, 1]; the requested blocks are
            # converted to distances uniformly below. (rbfk used to invert the whole matrix first,
            # which flipped its block geometry relative to corr/cos_sim.)
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
        """
        Compute the RBF (Radial Basis Function) kernel adjacency matrix.
        This method constructs an adaptive RBF kernel matrix based on the 
        concatenated embeddings. The bandwidth (sigma) for each sample is 
        determined by its distance to its 7th nearest neighbor, enabling 
        adaptive kernel scaling.
        
        Args:
            concat_embedds (torch.Tensor): A tensor of concatenated embeddings 
                of shape (n_samples, embedding_dim), where n_samples is the 
                number of samples and embedding_dim is the dimensionality 
                of the embeddings.
            to_cpu (bool, optional): If True, forces computation on CPU regardless 
                of the model's device. If False (default), uses the model's 
                configured device.
        
        Returns:
            torch.Tensor: A square adjacency matrix of shape (n_samples, n_samples) 
                with values in the range [0, 1]. The diagonal is filled with 0 
                if self-loops are disabled (AM_self_loop=False), or 1 if 
                self-loops are enabled.
        
        Raises:
            ValueError: If concat_embedds has invalid dimensions.
        
        Note:
            The sigma values are computed adaptively using the distance to the 
            7th nearest neighbor, which is a common technique for creating 
            data-dependent kernel bandwidths in graph-based learning tasks.
        """
        
        device = torch.device('cpu') if to_cpu else self.device
        A_matrix = self.get_A_e_d(concat_embedds, to_cpu)
        # Use a raw distance matrix (diagonal=0) for neighbour search so the self-distance
        # is always at sorted index 0 regardless of the AM_self_loop setting.
        raw_dist = torch.clamp(torch.cdist(concat_embedds, concat_embedds), min=0.).to(device)
        neigh_idx = min(7, max(concat_embedds.shape[0] - 1, 0))
        seventh_neigh = concat_embedds[torch.argsort(raw_dist, dim=1)[:, neigh_idx]]
            
        sigmas = torch.unsqueeze(torch.tensor([
            self.get_A_e_d(torch.cat((
                torch.unsqueeze(concat_embedds[i], dim=0), torch.unsqueeze(seventh_neigh[i], dim=0)
            )), to_cpu=to_cpu)[0,1].item()
            for i in range(concat_embedds.shape[0]) 
        ], device=device), dim=0)
        
        # clamp(min=1e-8) guards against sigma_i * sigma_j == 0 (duplicate embeddings at
        # distance 0), which would otherwise yield NaNs that the clamp below cannot repair.
        A = torch.exp(-A_matrix.pow(2) / torch.mm(sigmas.T, sigmas).clamp(min=1e-8)).to(device)
        A = torch.clamp(A, min=0., max=1.)
        
        return A.fill_diagonal_(1. if self.AM_self_loop else 0.)
    
    
    def get_A_e_d(self, concat_embedds: torch.Tensor, to_cpu = False) -> torch.Tensor:
        """
        Computes the adjacency matrix from concatenated embeddings.

        Calculates pairwise Euclidean distances between all embeddings in the
        concatenated tensor using torch.cdist, clamps negative values to 0
        (distances are unbounded above), and sets the diagonal via the self-loop flag.

        Args:
            concat_embedds (torch.Tensor): A tensor of concatenated embeddings
                with shape (n, d) where n is the number of embeddings and d
                is the embedding dimension.
            to_cpu (bool, optional): If True, moves the resulting adjacency
                matrix to CPU. If False, keeps it on the device. Defaults
                to False.

        Returns:
            torch.Tensor: The adjacency matrix with values clamped to [0, 1]
                and diagonal set to 0 if self-loops are disabled (AM_self_loop
                is False), or 1 if self-loops are enabled (AM_self_loop is True).
        """
        A = torch.cdist(concat_embedds, concat_embedds).to(torch.device('cpu') if to_cpu else self.device)
        # Distances are unbounded above; only clamp the lower bound. Clamping to max=1 here
        # would saturate every pair farther than 1.0 and collapse the RBF kernel / neighbour
        # search into an almost-constant matrix.
        return torch.clamp(A, min=0.).fill_diagonal_(1. if self.AM_self_loop else 0.)



    def get_A_cos_sim(self, concat_embedds: torch.Tensor, to_cpu = False) -> torch.Tensor:
        """
        Computes the cosine similarity matrix from concatenated embeddings.
        
        Args:
            concat_embedds (torch.Tensor): Concatenated embeddings tensor.
            to_cpu (bool, optional): If True, forces computation on CPU. Defaults to False.
        
        Returns:
            torch.Tensor: Cosine similarity matrix (adjacency matrix) with values clamped to [0, 1].
                The diagonal is filled with 1s if self-loops are enabled, else 0s.
        """
        device = torch.device('cpu') if to_cpu else self.device
        normalized_embedding = F.normalize(concat_embedds, dim=-1).to(device)
        
        A = torch.matmul(normalized_embedding, normalized_embedding.transpose(-1, -2)).to(device)
        A = torch.clamp(A, min=0., max=1.)

        return A.fill_diagonal_(1. if self.AM_self_loop else 0.)
        
        

    def get_A_corr(self, concat_embedds: torch.Tensor, to_cpu = False) -> torch.Tensor:
        """
        Computes the correlation-based adjacency matrix from concatenated embeddings.

        Calculates the Pearson correlation coefficient matrix for the given concatenated
        embeddings, clamps the values to [0, 1] range, and sets the diagonal values
        based on the self-loop configuration.

        Args:
            concat_embedds (torch.Tensor): Tensor of concatenated embeddings with shape
                (num_nodes, feature_dim) used to compute pairwise correlations.
            to_cpu (bool, optional): If True, moves the resulting adjacency matrix to CPU.
                Defaults to False, keeping it on the device (self.device).

        Returns:
            torch.Tensor: Adjacency matrix of shape (num_nodes, num_nodes) with values
                clamped to [0, 1]. The diagonal is set to 1.0 if self-loops are enabled
                (AM_self_loop=True), otherwise 0.0.

        Note:
            The correlation matrix is computed using torch.corrcoef which treats rows
            as observations and computes the correlation between columns.
        """
        A = torch.corrcoef(concat_embedds).to(torch.device('cpu') if to_cpu else self.device)
        return torch.clamp(A, min=0., max=1.).fill_diagonal_(1. if self.AM_self_loop else 0.)
        


    def get_X(self) -> None:
        """
        Compute and initialize the X matrix for the GTG algorithm.
        The X matrix serves as the initial probability/label matrix containing
        information for both labelled and unlabelled samples:
        - For labelled samples: one-hot encoded ground truth labels
        - For unlabelled samples: uniform probability distribution across all classes
        The matrix dimensions are:
            - Rows: (num_labelled_samples + num_unlabelled_samples)
            - Columns: number of classes in the dataset
        
        Returns:
            None: The computed X matrix is stored in self.X as a torch.Tensor.
        
        Example:
            >>> self.get_X()
            >>> print(self.X.shape)  # (n_labelled + n_unlabelled, n_classes)
        """
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
        """Runs the Graph Transduction Game (GTG) algorithm for semi-supervised learning.
        This method performs label propagation on a graph structure, iteratively
        updating the feature matrix X by propagating information from labeled
        nodes to unlabeled nodes through the adjacency matrix A.
        The algorithm runs until either:
            - The convergence tolerance (gtg_tol) is reached, meaning the change in X
            between iterations is below the threshold, or
            - The maximum number of iterations (gtg_max_iter) is reached.
        During each iteration:
            - The feature matrix X is multiplied by the adjacency matrix A to propagate
            information across the graph.
            - X is row-normalized to maintain probability distributions.
            - The entropy of unlabeled node predictions is recorded for analysis.
        Logs the following information:
            - Starting and ending predictions for unlabeled nodes (unique label counts).
            - GTG accuracy on unlabeled nodes (computed as argmax comparison with true labels).
        
        Returns:
            None: This method updates self.X in place and populates
            self.unlab_entropy_hist with entropy values for each iteration.
        
        Raises:
            None: Explicitly. Errors from tensor operations may propagate.
        """
        
        self.get_A()
        self.get_X()

        n_lab_obs = len(self.lab_embedds_dict["embedds"])
        lab_X_init = self.X[:n_lab_obs].clone()

        err = float('Inf')
        i = 0

        logger.info(' => Running GTG')
        logger.info(f'starting X: {self.X.argmax(dim=1)}')

        while err > self.gtg_tol and i < self.gtg_max_iter:
            X_old = torch.clone(self.X)

            self.X *= torch.mm(self.A, self.X)
            self.X /= torch.sum(self.X, dim=1, keepdim=True).clamp(min=1e-8)

            # Keep labelled observations fixed as hard anchors during propagation.
            if n_lab_obs > 0:
                self.X[:n_lab_obs] = lab_X_init

            logger.info(f'X: {self.X[len(self.labelled_indices):].argmax(dim=1).unique(return_counts=True)}')

            self.unlab_entropy_hist[:, i] = entropy(self.X)[len(self.labelled_indices):]

            err = torch.norm(self.X - X_old)
            i += 1
            
        logger.info(f'end X: {self.X[len(self.labelled_indices):].argmax(dim=1).unique(return_counts=True)}')
        logger.info(f'GTG accuracy: {self.X[n_lab_obs:].cpu().argmax(dim=1).eq(self.unlab_embedds_dict["labels"]).sum().item() / len(self.unlab_embedds_dict["embedds"])}')
        
        logger.info(' DONE\n')
        
        


    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:            
        """Select top-k unlabeled observations based on Graph Transduction Game entropy.
        This method performs active learning selection by executing the Graph Transduction
        Game (GTG) algorithm on unlabeled data and selecting the observations with the
        highest entropy measures.
        
        Args:
            sample_unlab_subset (Subset): A PyTorch Subset containing the unlabeled
                samples to be considered for selection.
            n_top_k_obs (int): The number of top observations to select based on
                entropy measures.
        
        Returns:
            Tuple[List[int], List[int]]: A tuple containing:
                - List[int]: Indices of the top-k selected unlabeled observations.
                - List[int]: Corresponding random samples from the unlabeled subset.
        
        Raises:
            Exception: If an unrecognized entropy computation strategy is specified.
        
        Notes:
            - The method loads the best checkpoint before computing embeddings.
            - Entropy history is computed over self.gtg_max_iter iterations.
            - Two entropy strategies are supported: 'mean' (average entropy) and
              'integral' (area under the entropy curve using trapezoid formula).
            - The method saves an entropy history tensor for visualization purposes.
            - GPU memory is cleaned up after processing by deleting intermediate
              tensors and calling garbage collection.
        """
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
    