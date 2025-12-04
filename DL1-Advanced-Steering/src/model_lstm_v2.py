import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class ConvNeXtLSTMV2(nn.Module):
    def __init__(self, seq_len=15, hidden=512, dropout=0.3):
        super().__init__()
        self.seq_len = seq_len

        self.backbone = convnext_tiny(
            weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )
        self.backbone.classifier = nn.Flatten()
        self.feature_dim = 768

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden,
            batch_first=True,
            dropout=dropout
        )
        self.ln = nn.LayerNorm(hidden)

        # multi-task heads
        self.head_steer = nn.Linear(hidden, 1)
        self.head_speed = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)           # [B*T, 768]
        feats = feats.view(B, T, -1)       # [B, T, 768]

        out, _ = self.lstm(feats)          # [B, T, hidden]
        h = self.ln(out[:, -1, :])         # last timestep

        steer = self.head_steer(h)         # [B,1]
        speed = self.head_speed(h)         # [B,1]
        return steer, speed
