import torch.nn as nn

from models.backbones.ResNet18 import ResNet18
from auto_encoder.ResNet18_Decoder import ResNet18_Decoder

class BackBone_Decoder(nn.Module):
    """
    A backbone decoder model combining an encoder and decoder architecture.
    This class integrates a ResNet18-based encoder and a ResNet18 decoder
    to form an autoencoder structure. It encodes input images into a
    latent representation and then decodes them back to reconstruct
    the original input.
    
    Args:
        classes (int): Number of classes for the encoder classification head.
        n_channels (int): Number of input channels (e.g., 3 for RGB images).
        image_size (int or tuple): Size of the input images.
    
    Attributes:
        encoder (ResNet18): The encoder network for feature extraction.
        decoder (ResNet18_Decoder): The decoder network for reconstruction.
    
    Example:
        >>> model = BackBone_Decoder(classes=10, n_channels=3, image_size=224)
        >>> encoded, decoded = model(input_tensor)
        >>> print(f"Encoded shape: {encoded.shape}")
        >>> print(f"Decoded shape: {decoded.shape}")
    """
    def __init__(self, classes, n_channels, image_size):
        super().__init__()

        # initial checks
        self.encoder = ResNet18(n_classes=classes, n_channels=n_channels, img_size=image_size)
        self.decoder = ResNet18_Decoder()
        
        
    def forward(self, x):
        _, encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded