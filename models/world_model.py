import torch
import torch.nn as nn
from torchvision import models

class WorldModel(nn.Module):
    def __init__(self, embed_dim: int = 512, img_channels: int = 3):
        super().__init__()
        # Using a pre-trained ResNet as the image encoder
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]  # Remove the last two layers
        self.backbone = nn.Sequential(*modules)
        self.conv = nn.Conv2d(2048, embed_dim, kernel_size=1)  # Adjust dimensions
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encodes images into feature representations.

        Args:
            images (torch.Tensor): Input images of shape [batch, 3, H, W]

        Returns:
            torch.Tensor: Encoded image features [batch, embed_dim, H', W']
        """
        features = self.backbone(images)  # [batch, 2048, H', W']
        features = self.conv(features)    # [batch, embed_dim, H', W']
        return features