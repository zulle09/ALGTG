
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import entropy
from config import al_params
from .GTG_Modules import (
    Module_LL, Module_LSs_GTG_MLP,
    Module_LS_GTG_LSTM, Module_LSs_GTG_MLP_alternative
)

from typing import Any, List, Tuple

import logging
logger = logging.getLogger(__name__)


        

class GTGModule(nn.Module):
    def __init__(self, params) -> None:
        """
        Initialize the GTG (Graph-Tensor-Graph) module.
        
        Args:
            params: A tuple containing (gtg_p, ll_p) configuration parameters.
                - gtg_p (dict): Dictionary with GTG-specific parameters:
                    - gtg_t (float): GTG convergence tolerance.
                    - gtg_i (int): GTG maximum number of iterations.
                    - am_s (str): Adjacency matrix strategy.
                    - am_sl (bool): Adjacency matrix self-loop flag.
                    - am_t (float): Adjacency matrix threshold value.
                    - am_ts (List[str]): List of adjacency matrix threshold strategies.
                    - am (List[str]): List of adjacency matrix functions.
                    - e_s (str): Entity strategy.
                    - bsgtgo (int): Batch size for GTG online processing.
                    - n_classes (int): Number of classes.
                    - device (str): Device for computation (e.g., 'cpu', 'cuda').
                    - gtg_module (str): GTG model identifier.
                - ll_p (dict): Dictionary with additional learning parameters.
        
        Attributes:
            get_A_fn (dict): Mapping of adjacency matrix function names to their implementations.
            gtg_tol (float): GTG tolerance threshold.
            gtg_max_iter (int): Maximum iterations for GTG.
            AM_strategy (str): Strategy for adjacency matrix computation.
            AM_self_loop (bool): Whether to include self-loops in adjacency matrix.
            AM_threshold (float): Threshold value for adjacency matrix.
            list_AM_threshold_strategy (List[str]): List of threshold strategies.
            list_AM_function (List[str]): List of adjacency matrix functions.
            ent_strategy (str): Entity strategy.
            batch_size_gtg_online (int): Batch size for online GTG processing.
            n_top_k_obs (int): Number of top-k observations.
            n_classes (int): Number of classes.
            device (torch.device): Computation device.
            ll_p (dict): Learning parameters.
            GTG_Model (str): GTG model name.
            name (str): Full module name combining class and model names.
        
        Note:
            The adjacency matrix functions available are: 'cos_sim' (cosine similarity),
            'corr' (correlation), and 'rbfk' (RBF kernel).
        """
        
        super(GTGModule, self).__init__()
        
        gtg_p, ll_p = params

        self.get_A_fn = { 'cos_sim': self.get_A_cos_sim, 'corr': self.get_A_corr, 'rbfk': self.get_A_rbfk }
        
        self.gtg_tol: float = gtg_p["gtg_t"]
        self.gtg_max_iter: int = gtg_p["gtg_i"]
        
        self.AM_strategy: str = gtg_p["am_s"]
        self.AM_self_loop = gtg_p["am_sl"]
        self.AM_threshold: float = gtg_p["am_t"]
        
        self.list_AM_threshold_strategy: List[str] = gtg_p["am_ts"]
        self.list_AM_function: List[str] = gtg_p["am"]
        
        self.ent_strategy: str = gtg_p["e_s"]
        #self.perc_labelled_batch: int = gtg_p["plb"]
        self.batch_size_gtg_online: int = gtg_p["bsgtgo"]
        
        self.n_top_k_obs: int = al_params["n_top_k_obs"]
        self.n_classes: int = gtg_p["n_classes"]
        self.device: torch.device = torch.device(gtg_p["device"])
        
        self.ll_p = ll_p
        self.GTG_Model = gtg_p["gtg_module"]
        self.name = f'{self.__class__.__name__}_{self.GTG_Model}'
        self.set_additional_module()
                
    
    def set_additional_module(self) -> None:
        """
        Set the additional GTG module based on the GTG_Model configuration.
        Initializes the appropriate Graph Transduction Game (GTG) module based on
        the value of self.GTG_Model. The method supports different model architectures
        including LL MLP, LS MLP variants, and LSTM-based models.
        
        Raises:
            AttributeError: If GTG_Model is not one of the valid options:
                'llmlp', 'lsmlps', 'lsmlpsA', 'lstmreg', 'lstmbc'.
        
        Note:
            The supported GTG models are:
            - 'llmlp': Module_LL for standard MLP-based GTG
            - 'lsmlps': Module_LSs_GTG_MLP for LS MLP GTG
            - 'lsmlpsA': Module_LSs_GTG_MLP_alternative for alternative LS MLP GTG
            - 'lstmreg': Module_LS_GTG_LSTM for regression
            - 'lstmbc': Module_LS_GTG_LSTM for binary classification
        """
        logger.info(self.GTG_Model)
        
        if self.GTG_Model == 'llmlp':
            self.gtg_module = Module_LL(self.ll_p).to(self.device)
        
        elif self.GTG_Model in ['lsmlps', 'lsmlpsA']:
            module = Module_LSs_GTG_MLP if self.GTG_Model == 'lsmlps' else Module_LSs_GTG_MLP_alternative
            self.gtg_module = module(
                params=self.ll_p,  
                gtg_func=self.graph_trasduction_game
            ).to(self.device)
        
        elif self.GTG_Model in ['lstmreg', 'lstmbc']:
            self.gtg_module = Module_LS_GTG_LSTM(
                params=self.ll_p, input_size=1,
                seq_length=self.gtg_max_iter,
                hidden_size=1, num_layers=1,
                output_size=1, bidirectional=True, device=self.device,
                gtg_func=self.graph_trasduction_game,
                is_bin_class=self.GTG_Model == 'lstmbc'
            ).to(self.device)
            
        else:
            logger.exception(' Invalid GTG Model') 
            raise AttributeError(' Invalid GTG Model')


        
    def define_idx_params(self, id_am_ts: int, id_am: int) -> None:
        """Define the indices for adjacency matrix threshold strategy and function.

        Args:
            id_am_ts: Index into the list of AM threshold strategies.
            id_am: Index into the list of AM functions.

        Attributes:
            AM_threshold_strategy: The selected adjacency matrix threshold strategy.
            AM_function: The selected adjacency matrix function.
        """
        self.AM_threshold_strategy: str = self.list_AM_threshold_strategy[id_am_ts]
        self.AM_function: str = self.list_AM_function[id_am]
    
    
    def get_A_treshold(self, A: torch.Tensor) -> Any:
        """Calculates the threshold value for the attention matrix.

        Determines the threshold to use for the attention matrix based on the
        configured threshold strategy.

        Args:
            A (torch.Tensor): The attention matrix tensor used for computing
                the threshold when the strategy is set to 'mean'.

        Returns:
            Any: The computed threshold value. Returns the mean of the attention
                matrix if `AM_threshold_strategy` is set to 'mean', otherwise
                returns the predefined `AM_threshold` value.

        Raises:
            Exception: Any exceptions that may be raised by torch.mean() when
                computing the mean of the attention matrix.
        """
        if self.AM_threshold_strategy == 'mean': return torch.mean(A)
        else: return self.AM_threshold
        
    
    # sim (1.) dist (0.) -> self loop 
    # sim (0.) dist (1.) -> no self loop 
    
    def get_A_rbfk(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        """
        Computes an RBF (Radial Basis Function) kernel-based adjacency matrix.
        This method calculates a similarity matrix between concatenated embeddings using
        the Gaussian RBF kernel. The bandwidth parameter (sigma) is dynamically computed
        for each sample based on the distance to its 7th nearest neighbor, providing
        adaptive kernel bandwidths.
        
        Args:
            concat_embedds (torch.Tensor): A tensor of concatenated embeddings with shape
                (batch_size, embedding_dim) for which to compute the adjacency matrix.
        
        Returns:
            torch.Tensor: A symmetric adjacency matrix of shape (batch_size, batch_size)
                with values in the range [0, 1]. The diagonal is set to 1 if self-loops
                are enabled (AM_self_loop=True), otherwise 0.
        
        Note:
            The sigma parameter is computed as the distance between each embedding and
            its 7th nearest neighbor, allowing the kernel to adapt to local density
            variations in the embedding space. This is based on the k-NN bandwidth
            selection method for Gaussian kernels.
        """
        A_matrix = self.get_A_e_d(concat_embedds)
        # Use a raw distance matrix (diagonal=0) for neighbour search so the self-distance
        # is always at sorted index 0 regardless of the AM_self_loop setting.
        raw_dist = torch.clamp(torch.cdist(concat_embedds, concat_embedds), min=0., max=1.)
        # Guard small batches: use the farthest available neighbour when < 8 samples.
        neigh_idx = min(7, max(concat_embedds.shape[0] - 1, 0))
        seventh_neigh = concat_embedds[torch.argsort(raw_dist, dim=1)[:, neigh_idx]]

        sigmas = torch.unsqueeze(torch.tensor([
            self.get_A_e_d(torch.cat((
                torch.unsqueeze(concat_embedds[i], dim=0), torch.unsqueeze(seventh_neigh[i], dim=0)
            )))[0,1].item()
            for i in range(concat_embedds.shape[0]) 
        ], device=self.device), dim=0)
        
        A_m_2 = -A_matrix.pow(2)
        sigma_T = sigmas.T
        sigma_mm = (torch.mm(sigma_T, sigmas))
        A = torch.exp(A_m_2 / sigma_mm)
        return torch.clamp(A.to(self.device), min=0., max=1.).fill_diagonal_(1. if self.AM_self_loop else 0.)
    
        
    def get_A_e_d(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        """
        Computes the adjacency matrix from concatenated embeddings.

        Calculates pairwise Euclidean distances between the input embeddings,
        clamps the values to the range [0, 1], and sets the diagonal elements
        based on the self-loop configuration.

        Args:
            concat_embedds (torch.Tensor): Concatenated embedding tensor of shape
                (n_nodes, embedding_dim).

        Returns:
            torch.Tensor: Adjacency matrix of shape (n_nodes, n_nodes) with
                values clamped to [0, 1]. The diagonal is set to 1.0 if
                self-loops are enabled (AM_self_loop=True), otherwise 0.0.

        Raises:
            ValueError: If concat_embedds has invalid dimensions.
        """
        A = torch.cdist(concat_embedds, concat_embedds).to(self.device)
        return torch.clamp(A, min=0., max=1.).fill_diagonal_(1. if self.AM_self_loop else 0.)
    

    def get_A_cos_sim(self, concat_embedds: torch.Tensor) -> torch.Tensor: 
        """
        Computes the cosine similarity matrix from concatenated embeddings.

        This method normalizes the input embeddings along the last dimension and
        computes the pairwise cosine similarity between all embeddings by performing
        matrix multiplication of the normalized embedding with its transpose.

        Args:
            concat_embedds (torch.Tensor): A tensor of concatenated embeddings
                to compute pairwise cosine similarity from.

        Returns:
            torch.Tensor: A cosine similarity matrix (adjacency matrix) with values
                clamped to the range [0, 1]. The diagonal is filled with 1.0 if
                self-loops are enabled (AM_self_loop is True), otherwise 0.0.

        Note:
            - The computation is performed on the device specified by self.device.
            - The output matrix has shape (N, N) where N is the number of embeddings.
        """
        normalized_embedding = F.normalize(concat_embedds, dim=-1).to(self.device)
        A = torch.matmul(normalized_embedding, normalized_embedding.transpose(-1, -2)).to(self.device)   
        return torch.clamp(A, min=0., max=1.).fill_diagonal_(1. if self.AM_self_loop else 0.)

        
    def get_A_corr(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        """
        Computes the correlation-based adjacency matrix.

        Calculates the Pearson correlation coefficient matrix from the concatenated
        embeddings, clamps the values to the [0, 1] range, and optionally adds self-loops
        by setting the diagonal elements based on the AM_self_loop configuration.

        Args:
            concat_embedds (torch.Tensor): Concatenated embedding tensors from which
                to compute the correlation matrix.

        Returns:
            torch.Tensor: The correlation-based adjacency matrix with values clamped
                to [0, 1] and diagonal elements set to 1.0 if self-loops are enabled,
                otherwise 0.0.

        Note:
            The correlation coefficient is computed using torch.corrcoef, which
            standardizes the input before calculating the Pearson correlation.
        """
        A = torch.corrcoef(concat_embedds).to(self.device)
        return torch.clamp(A, min=0., max=1.).fill_diagonal_(1. if self.AM_self_loop else 0.)
    
    
    def get_A(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Computes the affinity matrix from the given embedding tensor.
        This method generates an affinity matrix based on the input embedding using
        the affinity matrix function specified by `self.AM_function`. It applies
        thresholding based on `self.AM_threshold_strategy` and transforms the matrix
        according to `self.AM_strategy` (e.g., 'diversity', 'mixed').
        
        Args:
            embedding (torch.Tensor): The input embedding tensor used to compute
                the affinity matrix.
        
        Returns:
            torch.Tensor: The computed affinity matrix after applying threshold
                and strategy transformations.
        
        Raises:
            AttributeError: If `self.AM_function` is None.
        """
        if self.AM_function != None: A = self.get_A_fn[self.AM_function](embedding) # compute the affinity matrix
        else: raise AttributeError('A Fucntion is None')
                
        if self.AM_threshold_strategy != 'none':
            if self.AM_function != 'rbfk': A = torch.where(A < self.get_A_treshold(A), 0, A)
            else: A = torch.where(A > self.get_A_treshold(A), 1, A)
        if self.AM_strategy == 'diversity':
            A = 1 - A # set the whole matrix as a distance matrix and not similarity matrix
        elif self.AM_strategy == 'mixed':    
            if self.AM_function == 'rbfk': A = 1 - A # set the unlabelled submatrix as distance matrix and not similarity matrix
            A[:self.n_lab_obs, :self.n_lab_obs] = 1 - A[:self.n_lab_obs, :self.n_lab_obs] # -> inverse of the similarity matrix / distance matrix
            
        return A
        
        

    def get_X(self, probs) -> None:
        """
        Initialize the label distribution matrix X.
        Creates and populates a tensor X of shape (batch_size, n_classes) representing
        the label distribution or confidence scores for each observation in the batch.
        For labeled observations (indices 0 to n_lab_obs-1):
            - If GTG_Model is not 'lstmbc': Sets binary indicators (1.0) at positions
              corresponding to the true labels in self.lab_labels.
            - If GTG_Model is 'lstmbc': Copies probability values from the provided
              probs tensor.
        For unlabeled observations (indices n_lab_obs to batch_size-1):
            - Initializes uniform probability distribution (1 / n_classes) across
              all classes.
        
        Args:
            probs (torch.Tensor): Probability tensor used when GTG_Model is 'lstmbc'.
                Expected shape: (n_lab_obs, n_classes).
        
        Returns:
            None. The result is stored in self.X attribute.
        
        Note:
            - self.X has shape (batch_size, n_classes)
            - Labeled observations occupy the first n_lab_obs rows
            - Unlabeled observations occupy the remaining rows
        """
        self.X: torch.Tensor = torch.zeros((self.batch_size, self.n_classes), dtype=torch.float32, device=self.device)
       
        if self.GTG_Model != 'lstmbc': self.X[torch.arange(self.n_lab_obs), self.lab_labels] = 1.
        else: self.X[torch.arange(self.n_lab_obs), :] = probs[torch.arange(self.n_lab_obs)]
        
        self.X[self.n_lab_obs:, :] = torch.ones(self.n_classes, device=self.device) * (1. / self.n_classes)
    
    
    
    def graph_trasduction_game(self, features_embedding: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs Graph Transduction Game (GTG) for semi-supervised learning.
        This method implements an iterative graph-based transductive learning algorithm
        that propagates labels from labeled to unlabeled observations through a graph
        constructed from feature embeddings.
        
        Args:
            features_embedding (torch.Tensor): Feature embeddings of shape (batch_size, feature_dim)
                used to construct the adjacency matrix.
            labels (torch.Tensor): Ground truth labels of shape (batch_size,) for the labeled
                observations. Only the first n_lab_obs are used as labeled seeds.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Xs: Tensor of shape (batch_size, n_classes, gtg_max_iter) containing the
                    probability distributions over classes at each iteration.
                - entropy_hist: Tensor of shape (batch_size, gtg_max_iter) containing the
                    entropy of the predictions for unlabeled observations at each iteration.
        
        Raises:
            Warning: Logs warnings when NaN values are detected in features_embedding or
                adjacency matrix A.
        
        Note:
            The algorithm stops early if the L2 norm between consecutive iterations falls
            below gtg_tol or when max iterations (gtg_max_iter) are reached. Labeled
            observations (first n_lab_obs) remain fixed during the propagation process.
        """
        
        logger.info(f'Non zero cell in features_embedding {torch.count_nonzero(features_embedding)} / {features_embedding.numel()}')
        if torch.any(torch.isnan(features_embedding)):  logger.info(f'NaN cell in features_embedding  {torch.isnan(features_embedding).sum()} / {features_embedding.numel()}')
                        
        A = self.get_A(features_embedding) # -> nxn

        logger.info(f'Non zero cell in A {torch.count_nonzero(A)} / {A.numel()}')
        if torch.any(torch.isnan(A)): logger.info(f'NaN cell in A {torch.isnan(A).sum()} / {A.numel()}')

        if(torch.isnan(A).sum()):
            nan_indices = torch.isnan(A).nonzero(as_tuple=True)
            for i, j in zip(nan_indices[0], nan_indices[1]):
                logger.info(f'NaN cell in A: {i} {j} {A[i, j]}')

        X = self.X.clone()
        if A.requires_grad: X.requires_grad_(True)
        Xs = torch.empty((self.batch_size, self.n_classes, 0), device=self.device, dtype=torch.float32, requires_grad=True if A.requires_grad else False)

        # Accumulate unlabeled entropy per-iteration into a list so that torch.stack
        # preserves the autograd graph (in-place assignment into a pre-allocated tensor
        # would silently break gradients from the adjacency matrix to mod_ls).
        unlab_ent_list = []

        err = float('Inf')
        i = 0


        while err > self.gtg_tol and i < self.gtg_max_iter:

            if X.requires_grad: X.register_hook(lambda grad: logger.info(f'1- X grad: {grad.sum()}\n'))

            X_old = X.detach().clone()

            X = X * torch.mm(A, X)
            if X.requires_grad: X.register_hook(lambda grad: logger.info(f'2- X grad: {grad.sum()}'))
            X = X / torch.sum(X, dim=1, keepdim=True)
            if X.requires_grad: X.register_hook(lambda grad: logger.info(f'3- X grad: {grad.sum()}'))

            # Keep labelled observations fixed as hard anchors during propagation.
            if self.n_lab_obs > 0:
                X = torch.cat((self.X[:self.n_lab_obs], X[self.n_lab_obs:]), dim=0)

            logger.info(f'X: {X[self.n_lab_obs:].argmax(dim=1).unique(return_counts=True)}')

            unlab_ent_list.append(entropy(X[self.n_lab_obs:, :]).to(self.device))

            err = torch.norm(X.detach() - X_old)
            Xs = torch.cat((Xs, X.unsqueeze(dim=-1)), dim=-1)

            i += 1

        # Build entropy_hist as a differentiable tensor by stacking the per-iteration
        # unlabeled entropies, then padding remaining iterations with zeros.
        n_unlab = self.batch_size - self.n_lab_obs
        actual_iters = len(unlab_ent_list)
        if actual_iters > 0:
            unlab_hist = torch.stack(unlab_ent_list, dim=1)  # (n_unlab, actual_iters)
            if actual_iters < self.gtg_max_iter:
                pad = torch.zeros(n_unlab, self.gtg_max_iter - actual_iters, device=self.device)
                unlab_hist = torch.cat([unlab_hist, pad], dim=1)
        else:
            unlab_hist = torch.zeros(n_unlab, self.gtg_max_iter, device=self.device)
        lab_zeros = torch.zeros(self.n_lab_obs, self.gtg_max_iter, device=self.device)
        entropy_hist = torch.cat([lab_zeros, unlab_hist], dim=0)  # (B, gtg_max_iter)

        # fill in case we early exit by the norm err
        for _ in range(self.gtg_max_iter - Xs.shape[-1]):
            Xs = torch.cat((
                Xs, torch.zeros((self.batch_size, self.n_classes, 1), device=self.device, dtype=torch.float32, requires_grad=True if A.requires_grad else False)
            ), dim=-1)
            
        logger.info(f'end X: {X[self.n_lab_obs:].argmax(dim=1).unique(return_counts=True)}')
        unlab_obs = self.batch_size - self.n_lab_obs
        if unlab_obs > 0:
            gtg_acc = X[self.n_lab_obs:].argmax(dim=1).eq(labels[self.n_lab_obs:]).sum().item() / unlab_obs
            logger.info(f'GTG accuracy: {gtg_acc}')
        else:
            logger.info('GTG accuracy: N/A (no unlabelled observations in this batch)')

        return Xs, entropy_hist
    
          
        
    # List[torch.Tensor] -> detection
    def get_entropies(self, embedding: torch.Tensor, labels: torch.Tensor) -> torch.Tensor: #Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute entropies from the Graph Transduction Game.
        Args:
            embedding (torch.Tensor): The input embedding tensor.
            labels (torch.Tensor): The input labels tensor.
        Returns:
            torch.Tensor: The computed quantity result based on the entropy strategy.
                If strategy is 'mean', returns the mean of entropy history.
                If strategy is 'integral', returns the trapezoid area under the entropy history.
        Raises:
            AttributeError: If an invalid entropy strategy is specified.
        Note:
            The method uses the graph transduction game to compute entropy history.
            If NaN values are present in the entropy history, they are logged for debugging.
        """
                  
        entropy_hist = self.graph_trasduction_game(embedding, labels)[1]
        
        if torch.any(torch.isnan(entropy_hist)):
            logger.info(f'NaN cell in entropy_hist {torch.isnan(entropy_hist).sum()} / {entropy_hist.numel()}')
 
        if self.ent_strategy == 'mean': quantity_result = torch.mean(entropy_hist, dim=1)
        # computing the mean of the entropis history    
        elif self.ent_strategy == 'integral': quantity_result = torch.trapezoid(entropy_hist, dim=1)
        # computing the area of the entropis history using trapezoid formula    
        else:
            logger.exception(' Invalid GTG Strategy') 
            raise AttributeError(' Invalid GTG Strategy')
            
        return quantity_result
    
    
    
    def forward(
            self, 
            features: List[torch.Tensor], 
            embedds: torch.Tensor, 
            outs: torch.Tensor, 
            labels: torch.Tensor, 
            iteration: int,  
            n_lab_obs: int | None = None,
            weight: int = 1
        ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor | None]: 
        """
        Forward pass for the GTG (Guess Then Guess) model.
        Performs a forward pass through the GTG module to compute predictions
        and ground truth labels for active learning scenarios.
        Args:
            features: List of input feature tensors.
            embedds: Embedded representations.
            outs: Output logits from the model.
            labels: Ground truth labels.
            iteration: Current active learning iteration.
            weight: Weight factor for the GTG module. Defaults to 1.
        Returns:
            A tuple containing:
                - A tuple of (y_pred, y_true):
                    - y_pred: Predicted values from the GTG module.
                    - y_true: Target values based on entropies or labels.
                - labelled_mask: Boolean tensor indicating which samples are
                    labeled, or None during inference.
        Raises:
            AssertionError: If y_pred or y_true contains NaN values.
        Note:
            - The number of labeled observations is computed based on the
                iteration and batch size parameters.
            - If weight is 0, predictions are detached from the computation graph.
            - For 'lstmbc' model, labeled mask is derived from y_true.
            - A gradient hook is registered when predictions require gradients.
        """
        
        self.batch_size = len(embedds)

        if n_lab_obs is not None:
            self.n_lab_obs = min(max(n_lab_obs, 0), self.batch_size)
        else:
            # Fallback when explicit labelled count is not provided by the caller.
            expected_n_lab_obs = self.batch_size_gtg_online * iteration
            self.n_lab_obs = min(max(expected_n_lab_obs, 0), self.batch_size)
        
        '''self.batch_size = len(embedds)
        #int(self.batch_size * self.perc_labelled_batch)
        #                          10               *            32              +          32                * iteration
        if self.batch_size == al_params["al_iters"] * self.batch_size_gtg_online + self.batch_size_gtg_online * iteration:
            self.n_lab_obs = self.batch_size_gtg_online * iteration 
        else:
            #                   batch_size   - (           10000              % (          10          *    batch_size_gtg_online  )) 
            self.n_lab_obs = self.batch_size - (al_params["unlab_sample_dim"] % (al_params["al_iters"] * self.batch_size_gtg_online)) '''
        

        self.lab_labels = labels[:self.n_lab_obs]
        self.get_X(F.softmax(outs, dim=1))
        
        if self.GTG_Model == 'llmlp': y_pred = self.gtg_module(features).squeeze()
        else: y_pred = self.gtg_module(features, weight, labels).squeeze()

        if weight == 0: y_pred = y_pred.detach()

        if torch.any(torch.isnan(y_pred)):
            print(features)
            print(y_pred)
        assert not torch.any(torch.isnan(y_pred)), 'y_pred is nan'
        
        
        if self.GTG_Model != 'lstmbc':
            y_true = self.get_entropies(embedds.detach(), labels)
        else: 
            y_true = torch.zeros(self.batch_size, device=self.device)
            y_true[:self.n_lab_obs] = 1.
        
        if torch.any(torch.isnan(y_true)):
            print(embedds)
            print(y_true)
        assert not torch.any(torch.isnan(y_true)), 'y_true is nan'


        if self.training:
            if self.GTG_Model != 'lstmbc':
                labelled_mask = torch.zeros(self.batch_size, device=self.device)
                labelled_mask[:self.n_lab_obs] = 1.
                labelled_mask = labelled_mask.bool()
            else:
                labelled_mask = y_true.bool()
            
        else: labelled_mask = None
        
        if y_pred.requires_grad: y_pred.register_hook(lambda grad: logger.info(f'y_pred grad: {grad.sum()}'))
        
        return (y_pred, y_true), labelled_mask

