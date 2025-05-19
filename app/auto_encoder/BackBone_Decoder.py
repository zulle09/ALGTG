import torch.nn as nn

from auto_encoder.ResNet18_decoder import ResNet18_Decoder
from models.backbones.ResNet18 import ResNet18

class BackBone_Decoder(nn.Module):
    def __init__(self, classes, n_channels, image_size):
        super().__init__()

        # initial checks
        self.encoder = ResNet18(n_classes=classes, n_channels=n_channels, img_size=image_size)
        self.decoder = ResNet18_Decoder()
        
        
    def forward(self, x):
        _, encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded