"""
ResNet baselines adapted for face recognition (MPCNN/AutoFHE baselines).
ResNet32 (0.53M params) and ResNet44 (0.72M params) from the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResNetBlock, CryptoFaceNetBlock, AESPABlock


class ResNet(nn.Module):
    """
    ResNet for face recognition (baseline).
    Adapted from He et al., 2016.

    Paper uses ResNet32 (0.53M) and ResNet44 (0.72M).
    """

    def __init__(self, block_class, n_blocks: list, feature_dim: int = 256,
                 in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.feature_dim = feature_dim

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        C = base_channels
        self.layer1 = self._make_layer(block_class, C,    C,     n_blocks[0], stride=1)
        self.layer2 = self._make_layer(block_class, C,    C*2,   n_blocks[1], stride=2)
        self.layer3 = self._make_layer(block_class, C*2,  C*4,   n_blocks[2], stride=2)
        self.layer4 = self._make_layer(block_class, C*4,  C*8,   n_blocks[3], stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(C*8, feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim)

    def _make_layer(self, block_cls, in_ch, out_ch, n, stride):
        layers = [block_cls(in_ch, out_ch, stride=stride)]
        for _ in range(1, n):
            layers.append(block_cls(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        x = self.bn(self.fc(x))
        return F.normalize(x, p=2, dim=1)

    def extract_features(self, x):
        return self.forward(x)

    def get_model_name(self):
        return 'ResNet'


def build_resnet32(feature_dim=256, **kwargs):
    """ResNet32: 0.53M params (MPCNN baseline)."""
    return ResNet(ResNetBlock, [3, 4, 6, 3], feature_dim=feature_dim, **kwargs)


def build_resnet44(feature_dim=256, **kwargs):
    """ResNet44: 0.72M params (MPCNN baseline)."""
    return ResNet(ResNetBlock, [3, 4, 8, 3], feature_dim=feature_dim, **kwargs)
