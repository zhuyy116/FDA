import torch
from torch import nn


class FDA(nn.Module):
    def __init__(self, channels):
        super(FDA, self).__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.GMP = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=0, groups=channels, bias=False)
        self.sig = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(3)

    def forward(self, x):
        y1 = self.GAP(x)
        y2 = self.GMP(x).squeeze(-1)
        scatter = x.gt(y1).to(x.dtype)
        y5 = self.GAP(x * scatter).squeeze(-1)
        y = torch.cat((y1.squeeze(-1), y2, y5), dim=-1)
        y = self.bn(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.conv(y)
        y = self.sig(y).unsqueeze(-1)
        return x * y


# Optimized in speed, mathematically equivalent to the original
class FDA_Optimized(nn.Module):
    def __init__(self, channels):
        super(FDA_Optimized, self).__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=0, groups=channels, bias=False)
        self.sig = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(3)

    def forward(self, x):
        b, c, h, w = x.shape
        y1 = x.mean(dim=(2, 3), keepdim=True)
        y2 = x.view(b, c, -1).max(dim=2, keepdim=True)[0].view(b, c, 1, 1)
        y5 = (x * (x > y1)).sum(dim=(2, 3), keepdim=True) / (h * w)
        y = torch.cat([y1.flatten(2), y2.flatten(2), y5.flatten(2)], dim=2)
        y = self.bn(y.transpose(1, 2)).transpose(1, 2)
        y = self.conv(y)
        y = self.sig(y).unsqueeze(-1)
        return x * y
