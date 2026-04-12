
import torch
import torch.nn as nn

from models.backbones.ResNet18 import ResNet
from models.backbones.VGG import VGG

from models.modules.GTG.GTG_Cls import GTGModule
from models.modules.LossNet import LossNet

import logging
logger = logging.getLogger(__name__)


class Master_Model(nn.Module):
    def __init__(self, backbone: ResNet | VGG, added_module: LossNet | GTGModule | None, dataset_name: str) -> None:
        """
        Initialize the Master_Model.
        
        Args:
            backbone: The backbone network, either ResNet or VGG architecture.
            added_module: Optional additional module, can be LossNet, GTGModule,
                or None if no extra module is used.
            dataset_name: Name of the dataset used for saving the checkpoint.
        
        Note:
            The model name is automatically constructed from the backbone class
            name and the added module name (if provided). An initial checkpoint
            containing the model state dict is saved to
            'app/checkpoints/{dataset_name}/{model_name}_init.pth.tar' after
            initialization.
        """
        
        super(Master_Model, self).__init__()
        
        self.backbone = backbone
        self.added_module = added_module
        if added_module != None:
            self.added_module_name = added_module.name
            self.only_module_name = self.added_module_name.split('_')[0]
        else: self.added_module_name, self.only_module_name = None, None
        
        
        if added_module != None: self.name = f'{self.backbone.__class__.__name__}_{self.added_module_name}' # type: ignore
        else: self.name = f'{self.backbone.__class__.__name__}'
            
        # backbone and additional module have initialized their respecive layers, so I can save the initial checkpoint
        logger.info(f' => Saving Initial {self.name} checkpoint app/checkpoints/{dataset_name}/{self.name}_init.pth.tar')
        torch.save(dict(state_dict = self.state_dict()), f'app/checkpoints/{dataset_name}/{self.name}_init.pth.tar')
        logger.info(' DONE\n')
        
        

    def forward(self, x, labels=None, weight=1, mode='all', iteration=None):
        """
        Performs forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor to the backbone network.
            labels (torch.Tensor, optional): Ground truth labels for GTGModule.
                Defaults to None.
            weight (float, optional): Weight factor for feature detachment. If set
                to 0, features are detached from the computational graph. Defaults to 1.
            mode (str, optional): Operation mode determining what to return. Options are:
                - 'all': Returns both outputs and module output (training mode only).
                - 'outs': Returns only model outputs (evaluation mode only).
                - 'embedds': Returns only embeddings (evaluation mode only).
                - 'module_out': Returns only module output (evaluation mode only).
                Defaults to 'all'.
            iteration (int, optional): Current iteration number for GTGModule.
                Defaults to None.
        
        Returns:
            tuple or torch.Tensor: Varies depending on mode:
                - 'all': Tuple of (outs, module_out) during training; raises error in eval.
                - 'outs': Model outputs tensor in evaluation mode.
                - 'embedds': Embeddings tensor in evaluation mode.
                - 'module_out': Module output tensor in evaluation mode.
        
        Raises:
            AttributeError: If mode is not valid, if called in wrong training/evaluation
                state, or if no added_module exists when required.
        
        Note:
            In 'all' mode during training, assertions verify that embeddings contain no
            NaN values and have a standard deviation greater than zero.
        """

        if mode == 'all':
            if self.training:
                outs, embedds = self.backbone(x)
                
                if torch.any(torch.isnan(embedds)): print(embedds)
                assert not torch.any(torch.isnan(embedds)), 'embedding is nan'
                assert torch.std(embedds) > 0, 'std is zero or negative'

                if self.added_module != None:
                    features = self.backbone.get_features()
                    if weight == 0: features = [feature.detach() for feature in features]                        
                    if self.only_module_name == 'GTGModule':
                        module_out = self.added_module(features=features, embedds=embedds, outs=outs, labels=labels, weight=weight, iteration=iteration)
                    else: module_out = self.added_module(features=features)
                    return outs, module_out
                else: return outs, None
            else: raise AttributeError("The Master_Model is in evaluation mode, so it can't return everything")
        
        elif mode == 'outs': 
            if not self.training: return self.backbone(x)[0]
            else: raise AttributeError("The Master_Model is in training mode, so it can't return the probabilities") 
            
        elif mode == 'embedds': 
            if not self.training: return self.backbone(x)[1]
            else: raise AttributeError("The Master_Model is in training mode, so it can't return the embeddings")
            
        elif mode == 'module_out': # -> it is only used for the learning loss
            if not self.training:
                if self.added_module != None:
                    outs, embedds = self.backbone(x)
                    features = self.backbone.get_features()
                    if self.only_module_name == 'GTGModule' and iteration != None:
                        return self.added_module(features=features, embedds=embedds, outs=outs, labels=labels, iteration=iteration)[0][0]
                    else: return self.added_module(features=features)
                else: raise AttributeError("The Master_Model hasn't got any additional module")   
            else: raise AttributeError("The Master_Model is in training mode, so it can't return the module_out")
        else: raise AttributeError("The mode is not valid")