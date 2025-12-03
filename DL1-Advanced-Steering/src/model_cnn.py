# TODO: CNN backbone
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class ConvNeXtTinySteering(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pretrained ConvNeXt-Tiny backbone
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

        # Replace classifier with smaller layer
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1)   # single steering output
        )

    def forward(self, x):
        return self.backbone(x)
