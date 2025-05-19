import torch
import torch.nn as nn
import torch.nn.init as init

from typing import List, Tuple


def init_weights_apply(m: torch.nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None: init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=0.01)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    


class VGG(nn.Module):

    def __init__(self, features: nn.Sequential, n_classes: int):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.linears = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout()
        )
            
        self.classifier = nn.Linear(4096, n_classes)

        self.apply(init_weights_apply)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = []
        y = x
        for i, model in enumerate(self.features):
            y = model(y)
            if i in {3,5,10,15}: feat.append(y) #(y.view(y.size(0),-1))
        self.feat = feat
        
        x = self.features(x)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        embedds = self.linears(x)
        x = self.classifier(embedds)
        
        return x, embedds
    
    def get_embedding_dim(self) -> int:
        return self.classifier.in_features
    
    def get_features(self) -> List[torch.Tensor]:
        return self.feat
    
    def get_rich_features_shape(self) -> List[int]:
        return [64, 64, 128, 256]



def make_layers(n_channels: int, cfg, batch_norm=False) -> nn.Sequential:
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(n_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            n_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(cfg, batch_norm, n_channels, n_classes):
    return VGG(make_layers(n_channels, cfgs[cfg], batch_norm=batch_norm), n_classes)

def VGG16_bn(n_channels, n_classes): return _vgg('D', True, n_channels, n_classes)
