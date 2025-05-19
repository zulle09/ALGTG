
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import entropy
from config import al_params

from typing import Any, List, Tuple, Dict

import logging
logger = logging.getLogger(__name__)



class Module_LS_GTG_LSTM(nn.Module):
    def __init__(self, params: Dict[str, Any], input_size: int, hidden_size: int, num_layers: int, output_size: int, \
                bidirectional: bool, device: torch.device, seq_length: int, gtg_func, is_bin_class=False) -> None:
        super(Module_LS_GTG_LSTM, self).__init__()

        self.mod_ls = Module_LS(params)
        self.is_bin_class = is_bin_class
        self.gtg_func = gtg_func
        self.sigmoid = nn.Sigmoid()
        
        self.hidden_size = hidden_size
        self.device = device
        
        in_features_cls = seq_length * hidden_size * (2 if bidirectional else 1) #num_layers
        self.hiddens_dim = num_layers * (2 if bidirectional else 1)
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional)
        
        self.classifier = nn.Linear(in_features_cls, output_size)
        
        
    def forward(self, features: List[torch.Tensor], weight: int, labels: torch.Tensor) -> torch.Tensor:
        
        ls_features = self.mod_ls(features)
        if weight == 0: ls_features = [ls.detach() for ls in ls_features]
        ls_features = torch.cat(ls_features, dim=1)
        
        ent_history = self.gtg_func(ls_features, labels)[1].unsqueeze(dim=-1)
        
        hiddens = (
            torch.zeros(self.hiddens_dim, ent_history.shape[0], self.hidden_size, device=self.device),
            torch.zeros(self.hiddens_dim, ent_history.shape[0], self.hidden_size, device=self.device)
        )
        out, _ = self.lstm(ent_history, hiddens)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return self.sigmoid(out) if self.is_bin_class else out
      



class Module_LSs_GTG_MLP_alternative(nn.Module):
    def __init__(self, params: Dict[str, Any], gtg_func) -> None:
        super(Module_LSs_GTG_MLP_alternative, self).__init__()

        self.mod_ls = Module_LS(params)
        self.linear = nn.Linear(4, 1)
        self.gtg_func = gtg_func

        
    def forward(self, features: List[torch.Tensor], weight: int, labels: torch.Tensor) -> torch.Tensor:
        outs = []
        for id, features_embedding in enumerate(features):
            emb_ls = self.mod_ls(features_embedding, id)
            if weight == 0: emb_ls = emb_ls.detach()
            latent_feature = self.gtg_func(emb_ls, labels)[0]
            out1 = latent_feature.view(latent_feature.size(0), -1)
            out2 = torch.mean(out1, dim=1)
            outs.append(out2.unsqueeze(1))
        outs = torch.cat(outs, dim=1)
        out = self.linear(outs)
        return out


class Module_LSs_GTG_MLP(nn.Module):
    def __init__(self, params: Dict[str, Any], gtg_func) -> None:
        super(Module_LSs_GTG_MLP, self).__init__()

        self.mod_ls = Module_LS(params)
        self.gtg_func = gtg_func

        
    def forward(self, features: List[torch.Tensor], weight: int, labels: torch.Tensor) -> torch.Tensor:
        ls_features = self.mod_ls(features)
        if weight == 0: ls_features = [ls.detach() for ls in ls_features]
        ls_features = torch.cat(ls_features, dim=1)
        
        latent_feature = self.gtg_func(ls_features, labels)[0]
        out = latent_feature.view(latent_feature.size(0), -1)
        out = torch.mean(out, dim=1)
        
        return out


class Module_LS(nn.Module):
    def __init__(self, params: Dict[str, Any]) -> None:
        super(Module_LS, self).__init__()

        # same parameters of loss net
        feature_sizes = params["feature_sizes"]
        num_channels = params["num_channels"]
        interm_dim = params["interm_dim"]

        self.linears, self.gaps = [], []

        for n_c, e_d in zip(num_channels, feature_sizes):
            self.gaps.append(nn.AvgPool2d(e_d))
            self.linears.append(nn.Sequential(
                nn.Linear(n_c, interm_dim), nn.ReLU(),# nn.BatchNorm1d(interm_dim)
            ))

        self.linears = nn.ModuleList(self.linears)
        self.gaps = nn.ModuleList(self.gaps)


    def forward(self, features: torch.Tensor | List[torch.Tensor], id_feat: int =-1) -> torch.Tensor | List[torch.Tensor]:
        if id_feat != -1: 
            out = self.gaps[id_feat](features)
            out = out.view(out.size(0), -1)
            return self.linears[id_feat](out)
        else:
            outs = []
            for i in range(len(features)):
                out = self.gaps[i](features[i])
                out = out.view(out.size(0), -1)
                out = self.linears[i](out)
                outs.append(out)
            return outs
    
    
    
class Module_LL(nn.Module):
    def __init__(self, params: Dict[str, Any]) -> None:
        super(Module_LL, self).__init__()

        # same parameters of loss net
        feature_sizes = params["feature_sizes"]
        num_channels = params["num_channels"]
        interm_dim = params["interm_dim"]

        self.linears, self.gaps = [], []

        for n_c, e_d in zip(num_channels, feature_sizes):
            self.gaps.append(nn.AvgPool2d(e_d))
            self.linears.append(nn.Sequential(
                nn.Linear(n_c, interm_dim), nn.ReLU(),
            ))

        self.linears = nn.ModuleList(self.linears)
        self.gaps = nn.ModuleList(self.gaps)
        self.classifier = nn.Linear(interm_dim * len(num_channels), 1)


    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        outs = []
        for i in range(len(features)):
            out = self.gaps[i](features[i])
            out = out.view(out.size(0), -1)
            out = self.linears[i](out)
            outs.append(out)
        out = torch.cat(outs, 1)
        return self.classifier(out)


        

class GTGModule(nn.Module):
    def __init__(self, params) -> None:
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


        
    def define_idx_params(self, id_am_ts: int, id_am: int) -> None: # in order to define the indices of AM_threshold_strategy and AM_function
        self.AM_threshold_strategy: str = self.list_AM_threshold_strategy[id_am_ts]
        self.AM_function: str = self.list_AM_function[id_am]
    
    
    def get_A_treshold(self, A: torch.Tensor) -> Any:
        if self.AM_threshold_strategy == 'mean': return torch.mean(A)
        else: return self.AM_threshold
        
    
    # sim (1.) dist (0.) -> self loop 
    # sim (0.) dist (1.) -> no self loop 
    
    def get_A_rbfk(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        A_matrix = self.get_A_e_d(concat_embedds)
        seventh_neigh = concat_embedds[torch.argsort(A_matrix, dim=1)[:, 6]]            

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
        A = torch.cdist(concat_embedds, concat_embedds).to(self.device)
        return torch.clamp(A, min=0., max=1.).fill_diagonal_(1. if self.AM_self_loop else 0.)
    

    def get_A_cos_sim(self, concat_embedds: torch.Tensor) -> torch.Tensor: 
        normalized_embedding = F.normalize(concat_embedds, dim=-1).to(self.device)
        A = torch.matmul(normalized_embedding, normalized_embedding.transpose(-1, -2)).to(self.device)   
        return torch.clamp(A, min=0., max=1.).fill_diagonal_(1. if self.AM_self_loop else 0.)

        
    def get_A_corr(self, concat_embedds: torch.Tensor) -> torch.Tensor:
        A = torch.corrcoef(concat_embedds).to(self.device)
        return torch.clamp(A, min=0., max=1.).fill_diagonal_(1. if self.AM_self_loop else 0.)
    
    
    def get_A(self, embedding: torch.Tensor) -> torch.Tensor:
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
        self.X: torch.Tensor = torch.zeros((self.batch_size, self.n_classes), dtype=torch.float32, device=self.device)
       
        if self.GTG_Model != 'lstmbc': self.X[torch.arange(self.n_lab_obs), self.lab_labels] = 1.
        else: self.X[torch.arange(self.n_lab_obs), :] = probs[torch.arange(self.n_lab_obs)]
        
        self.X[self.n_lab_obs:, :] = torch.ones(self.n_classes, device=self.device) * (1. / self.n_classes)
    
    
    
    def graph_trasduction_game(self, features_embedding: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        logger.info(f'Non zero cell in features_embedding {torch.count_nonzero(features_embedding)} / {features_embedding.numel()}')
        if torch.any(torch.isnan(features_embedding)):  logger.info(f'NaN cell in features_embedding  {torch.isnan(features_embedding).sum()} / {features_embedding.numel()}')
                        
        entropy_hist = torch.zeros((self.batch_size, self.gtg_max_iter), device=self.device)        
        
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
        
        err = float('Inf')
        i = 0
        
                             
        while err > self.gtg_tol and i < self.gtg_max_iter:
            
            if X.requires_grad: X.register_hook(lambda grad: logger.info(f'1- X grad: {grad.sum()}\n'))
            
            X_old = X.detach().clone()
                        
            X = X * torch.mm(A, X)
            if X.requires_grad: X.register_hook(lambda grad: logger.info(f'2- X grad: {grad.sum()}'))
            X = X / torch.sum(X, dim=1, keepdim=True)
            if X.requires_grad: X.register_hook(lambda grad: logger.info(f'3- X grad: {grad.sum()}'))

            logger.info(f'X: {X[self.n_lab_obs:].argmax(dim=1).unique(return_counts=True)}')
            
            entropy_hist[self.n_lab_obs:, i] = entropy(X[self.n_lab_obs:, :]).to(self.device)

            err = torch.norm(X.detach() - X_old)
            Xs = torch.cat((Xs, X.unsqueeze(dim=-1)), dim=-1)
            
            i += 1
        
        # fill in case we early extit by the norm err
        for _ in range(self.gtg_max_iter - Xs.shape[-1]): 
            Xs = torch.cat((
                Xs, torch.zeros((self.batch_size, self.n_classes, 1), device=self.device, dtype=torch.float32, requires_grad=True if A.requires_grad else False)
            ), dim=-1)
            
        logger.info(f'end X: {X[self.n_lab_obs:].argmax(dim=1).unique(return_counts=True)}')
        logger.info(f'GTG accuracy: {X[self.n_lab_obs:].argmax(dim=1).eq(labels[self.n_lab_obs:]).sum().item() / (self.batch_size - self.n_lab_obs)}')

        return Xs, entropy_hist
    
          
        
    # List[torch.Tensor] -> detection
    def get_entropies(self, embedding: torch.Tensor, labels: torch.Tensor) -> torch.Tensor: #Tuple[torch.Tensor, torch.Tensor]:
                  
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
    
    
    
    def forward(self, features: List[torch.Tensor], embedds: torch.Tensor, outs: torch.Tensor, labels: torch.Tensor, iteration: int,  weight: int = 1) \
        -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor | None]: 
        
        self.batch_size = len(embedds)
        #int(self.batch_size * self.perc_labelled_batch)
        #                          10               *            32              +          32                * iteration
        if self.batch_size == al_params["al_iters"] * self.batch_size_gtg_online + self.batch_size_gtg_online * iteration:
            self.n_lab_obs = self.batch_size_gtg_online * iteration 
        else:
            #                   batch_size   - (           10000              % (          10          *    batch_size_gtg_online  )) 
            self.n_lab_obs = self.batch_size - (al_params["unlab_sample_dim"] % (al_params["al_iters"] * self.batch_size_gtg_online)) 
        
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





             
'''class Module_LS_GTG_MLP(nn.Module):
    def __init__(self, params: Dict[str, Any], gtg_iter, gtg_func) -> None:
        super(Module_LS_GTG_MLP, self).__init__()

        self.mod_ls = Module_LS(params)
        self.gtg_func = gtg_func        
        self.classifier = nn.Linear(gtg_iter, 1)
        
        
    def forward(self, features: List[torch.Tensor], weight: int) -> torch.Tensor:
        
        ls_features = self.mod_ls(features)
        if weight == 0: ls_features = [ls.detach() for ls in ls_features]
        ls_features = torch.cat(ls_features, dim=1)
        
        ent_history = self.gtg_func(ls_features)[1].unsqueeze(dim=1)
        ent_history = ent_history.view(ent_history.size(0), -1)
        
        return self.classifier(ent_history)'''

