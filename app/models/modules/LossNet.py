
import torch.nn as nn
import torch.nn.functional as F
import torch

import logging
logger = logging.getLogger(__name__)

    
class LossPredLoss(nn.Module):
    def __init__(self, device, margin=1.0, reduction='mean'):
        super(LossPredLoss, self).__init__()

        assert reduction == 'mean' or reduction == 'none', 'Wrong reduction'
        self.device = device
        self.margin = margin
        self.criterion = nn.BCELoss(reduction=reduction)

    def forward(self, input, target):
        assert len(input) % 2 == 0, 'The batch size is not even.'
        assert input.shape == input.flip(0).shape
        
        input = input.view(input.size(0))
        
        input = (input - input.flip(0))[:len(input) // 2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
        target = (target - target.flip(0))[:len(target) // 2]
        target = target.detach()
        
        diff = torch.sigmoid(input)
        one = torch.sign(torch.clamp(target, min=0)).to(self.device) # 1 operation which is defined by the authors
        
        return self.criterion(diff,one)



# Loss Prediction Network
class LossNet(nn.Module):
    def __init__(self, LL_params):
        super(LossNet, self).__init__()
        
        self.name = f'{self.__class__.__name__}_{LL_params["module_out"]}' # -> or TiDAL LL module

        feature_sizes = LL_params["feature_sizes"]
        num_channels = LL_params["num_channels"]
        interm_dim = LL_params["interm_dim"]
        task = LL_params["task"]
        
        self.GAP = []
        for feature_size in feature_sizes:
            if task == 'detection': self.GAP.append(nn.AdaptiveAvgPool2d((1, 1)))
            else: self.GAP.append(nn.AvgPool2d(feature_size))
            
        self.GAP = nn.ModuleList(self.GAP)

        self.FC = []
        for num_channel in num_channels: self.FC.append(nn.Linear(num_channel, interm_dim))
        self.FC = nn.ModuleList(self.FC)

        self.linear = nn.Linear(len(num_channels) * interm_dim, LL_params["module_out"])
                

    def forward(self, features):
        outs = []
        for i in range(len(features)):
            out = self.GAP[i](features[i])
            out = out.view(out.size(0), -1)
            out = F.relu(self.FC[i](out))
            outs.append(out)

        out = self.linear(torch.cat(outs, 1))
        return out
        