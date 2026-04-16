
import torch.nn as nn
import torch.nn.functional as F
import torch

import logging
logger = logging.getLogger(__name__)

    
class LossPredLoss(nn.Module):
    def __init__(self, device, margin=1.0, reduction='mean'):
        """
        Initialize the LossPredLoss module.

        Args:
            device: The device (cpu/cuda) to run the model on.
            margin (float): The margin value for the loss. Defaults to 1.0.
            reduction (str): The reduction method for the loss computation.
                Must be either 'mean' or 'none'. Defaults to 'mean'.

        Raises:
            AssertionError: If reduction is not 'mean' or 'none'.
        """
        super(LossPredLoss, self).__init__()

        assert reduction == 'mean' or reduction == 'none', 'Wrong reduction'
        self.device = device
        self.margin = margin
        self.criterion = nn.BCELoss(reduction=reduction)

    def forward(self, input, target):
        """
        Forward pass of the LossNet module.
        Computes the loss between the input and target tensors by comparing
        mirrored pairs in the batch. The input and target are split into pairs
        (element i and element 2B-1-i), and the differences are computed.
        
        Args:
            input: Input tensor of shape (batch_size,) where batch_size must be even.
            target: Target tensor of shape (batch_size,) with same batch size as input.
        
        Returns:
            Tensor: The computed loss value between the sigmoid-transformed input
                differences and the sign of the target differences.
        
        Raises:
            AssertionError: If batch size is not even or input shapes don't match
                when flipped.
        """
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
        """
        Initialize the LossNet module.
        
        Args:
            LL_params (dict): Dictionary containing the following parameters:
                - feature_sizes (list of int): List of feature map sizes for each
                  Global Average Pooling layer.
                - num_channels (list of int): List of number of input channels for
                  each fully connected layer.
                - interm_dim (int): Intermediate dimension size for the fully
                  connected layers.
                - task (str): Task identifier (currently unused in initialization).
                - module_out (int): Output dimension for the final linear layer.
        
        Attributes:
            name (str): Name of the module in the format '{ClassName}_{module_out}'.
            GAP (nn.ModuleList): List of Global Average Pooling layers, one for
                each feature size.
            FC (nn.ModuleList): List of fully connected layers that map from
                num_channels to interm_dim.
            linear (nn.Linear): Final linear layer that combines all intermediate
                dimensions to produce the module output.
        """
        super(LossNet, self).__init__()
        
        self.name = f'{self.__class__.__name__}_{LL_params["module_out"]}' # -> or TiDAL LL module

        feature_sizes = LL_params["feature_sizes"]
        num_channels = LL_params["num_channels"]
        interm_dim = LL_params["interm_dim"]
        
        self.GAP = []
        for feature_size in feature_sizes:
            self.GAP.append(nn.AvgPool2d(feature_size))
            
        self.GAP = nn.ModuleList(self.GAP)

        self.FC = []
        for num_channel in num_channels: self.FC.append(nn.Linear(num_channel, interm_dim))
        self.FC = nn.ModuleList(self.FC)

        self.linear = nn.Linear(len(num_channels) * interm_dim, LL_params["module_out"])
                

    def forward(self, features):
        """Performs forward pass through the LossNet.

        Args:
            features (list of torch.Tensor): List of feature tensors from different
                layers of the network. Each tensor should have shape (batch_size,
                channels, height, width).

        Returns:
            torch.Tensor: Output tensor after processing all features through GAP,
                fully connected layers, ReLU activation, and a final linear layer.
                Shape is (batch_size, num_classes) or (batch_size, output_dim).

        Example:
            >>> features = [feat1, feat2, feat3]  # List of feature maps
            >>> output = forward(features)
        """
        outs = []
        for i in range(len(features)):
            out = self.GAP[i](features[i])
            out = out.view(out.size(0), -1)
            out = F.relu(self.FC[i](out))
            outs.append(out)

        out = self.linear(torch.cat(outs, 1))
        return out
        