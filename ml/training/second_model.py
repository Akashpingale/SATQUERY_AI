"""Compact U-Net for SECOND binary change detection (learned, Phase 9).

Input is the channel-concatenation of the co-registered T1 and T2 RGB images (6 channels);
output is a single-channel change logit map. Deliberately small so it trains on a 4 GB GPU at
256x256. This is a LEARNED detector added ALONGSIDE the Phase 6 deterministic CVA, never a
replacement.
"""

from __future__ import annotations

import torch
from torch import nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SiameseUNet(nn.Module):
    def __init__(self, in_channels: int = 6, base: int = 32) -> None:
        super().__init__()
        self.e1 = _block(in_channels, base)
        self.e2 = _block(base, base * 2)
        self.e3 = _block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _block(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = _block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = _block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = _block(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.e1(x)
        c2 = self.e2(self.pool(c1))
        c3 = self.e3(self.pool(c2))
        b = self.bottleneck(self.pool(c3))
        d3 = self.d3(torch.cat([self.up3(b), c3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3), c2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), c1], dim=1))
        return self.head(d1)  # (B, 1, H, W) logits
