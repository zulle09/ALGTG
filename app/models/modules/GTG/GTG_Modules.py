
import torch.nn as nn
import torch
from typing import List, Dict, Any

#################################################################################################################################
# Define the other GTG modules (MLP, LSTM, etc.) here. They should have the same input and output of the GTG_LS_GTG_LSTM module, 
# so that they can be easily interchanged in the training loop.
#################################################################################################################################


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

