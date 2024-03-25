import torch
from torch import nn


class FDA_plus(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(FDA_plus, self).__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.GMP = nn.AdaptiveMaxPool2d(1)
        self.sig = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(3)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels, bias=False)
        )

    def forward(self, x):
        y1 = self.GAP(x)
        y2 = self.GMP(x).squeeze(-1)
        scatter = x.gt(y1).to(x.dtype)
        y5 = self.GAP(x * scatter).squeeze(-1)
        y = torch.cat((y1.squeeze(-1), y2, y5), dim=-1)  # ed.7
        y = self.bn(y.transpose(-1, -2)).transpose(-1, -2)
        y1 = self.mlp(y[:, :, 0])
        y2 = self.mlp(y[:, :, 1])
        y5 = self.mlp(y[:, :, 2])
        y = y1 + y2 + y5
        y = self.sig(y).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * y