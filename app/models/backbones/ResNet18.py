'''ResNet in PyTorch.

Reference:
https://github.com/kuangliu/pytorch-cifar
'''

import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F

from utils import init_weights_apply



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, img_size, n_channels=3, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.img_size = img_size

        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        
        self.apply(init_weights_apply)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, embedding=False):
        if embedding: # -> performed for alphamix strategy
            embedds = x
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out1 = self.layer1(out)
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out4 = self.layer4(out3)
            self.features = [out1, out2, out3, out4]
            
            out = F.avg_pool2d(out4, 4 if self.img_size == 32 else 8)

            embedds = out.view(out.size(0), -1)
        
        out = self.linear(embedds)
        
        return out, embedds
    
    
    def get_rich_features_shape(self) -> List[int]:
        return [64, 128, 256, 512]
    
    def get_features(self) -> List[torch.Tensor]:
        return self.features
    
    def get_embedding_dim(self) -> int:
        return self.linear.in_features



def ResNet18(n_classes=10, n_channels=3, img_size=32) -> ResNet:
    return ResNet(block=BasicBlock, num_blocks=[2,2,2,2], n_channels=n_channels, num_classes=n_classes, img_size=img_size)
