from __future__ import annotations

from torch import nn


class ConvFeaturePredictor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        hidden = max(256, min(1024, out_channels // 2))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class MLPPooledPredictor(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        hidden = max(256, min(1024, out_features))
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x):
        return self.net(x)
