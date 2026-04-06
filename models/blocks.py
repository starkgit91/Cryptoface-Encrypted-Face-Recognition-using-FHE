"""
CryptoFaceNet convolutional blocks.

Implements the depth-optimal shifted AESPA blocks shown in Figure 4 of the paper.

FHE Level Consumption:
  Standard AESPA block: 8 levels (2 per Conv + 2 per HerPN × 2 pairs)
  CryptoFaceNet block:  6 levels (fuse 'a' into Conv → HerPN costs 1 instead of 2)

Block topology (stride=1, following Figure 4 right):
  Input (Level ℓ)
    → HerPN (x^2 + b1/a1 * x + c1/a1)      [Level ℓ-1]
    → Conv × a1                              [Level ℓ-2; 'a1' fused into Conv]
       ↓                                     [×a1 branch for residual scaling]
    → HerPN (x^2 + b2/a2 * x + c2/a2)      [Level ℓ-3]
    → Conv × a2                              [Level ℓ-4]
    → ⊕ residual (scaled by a1)             [Level ℓ-4 → ℓ-6 after add+rescale]
  Output (Level ℓ-6)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .activations import HerPN, get_activation


class FHEConv2d(nn.Module):
    """
    Convolutional layer optimized for FHE.

    In FHE, standard Conv2d costs 2 multiplicative levels.
    BatchNorm is fused into Conv weights at inference time.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, bias: bool = False,
                 fhe_depth_cost: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.fhe_depth_cost = fhe_depth_cost

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def fhe_depth(self) -> int:
        return self.fhe_depth_cost


class CryptoFaceNetBlock(nn.Module):
    """
    CryptoFaceNet block (depth-optimal shifted AESPA block).

    Achieves depth 6 vs AESPA's depth 8, saving 2 FHE levels.

    The key optimization: fuse coefficient 'a' into Conv weights,
    changing polynomial ax^2 + bx + c  →  x^2 + (b/a)x + (c/a)
    which reduces HerPN depth from 2 to 1 per block.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        stride: Convolution stride (1 or 2)
        activation: Activation type ('herp' recommended)
    """

    FHE_DEPTH = 6  # vs AESPA's 8

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, activation: str = 'herp'):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ── Main path ─────────────────────────────────────────────────────────
        # HerPN1: x^2 + (b1/a1)x + (c1/a1)   [depth: 1]
        self.act1 = HerPN(in_channels, fused=True)

        # Conv1 (absorbs a1 coefficient)       [depth: 2]
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # HerPN2: x^2 + (b2/a2)x + (c2/a2)   [depth: 1]
        self.act2 = HerPN(out_channels, fused=True)

        # Conv2 (absorbs a2 coefficient)       [depth: 2]
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # ── Residual path ─────────────────────────────────────────────────────
        # For stride=2 or channel mismatch, need projection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if stride == 2:
                # stride=2 block has extra BN (visible in Figure 4 right)
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=stride,
                              padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1,
                              padding=0, bias=False),
                )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        CryptoFaceNet block forward pass.
        Mirrors the FHE computation: HerPN → Conv → HerPN → Conv → ⊕
        """
        residual = self.shortcut(x)

        # HerPN → Conv → BN (main path, first half)
        out = self.act1(x)
        out = self.conv1(out)
        out = self.bn1(out)

        # HerPN → Conv → BN (main path, second half)
        out = self.act2(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Residual addition (FHE: cheap ciphertext-ciphertext add)
        out = out + residual
        return out

    def fhe_depth(self) -> int:
        return self.FHE_DEPTH

    def extra_repr(self) -> str:
        return (f'in={self.in_channels}, out={self.out_channels}, '
                f'stride={self.stride}, fhe_depth={self.FHE_DEPTH}')


class AESPABlock(nn.Module):
    """
    Standard AESPA block (for comparison with CryptoFaceNet).
    Depth: 8 (2 per Conv + 2 per HerPN, × 2 pairs).
    """

    FHE_DEPTH = 8

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        # Standard HerPN (not fused, depth=2)
        self.act1 = HerPN(out_channels, fused=False)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.act2 = HerPN(out_channels, fused=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        return out + residual

    def fhe_depth(self) -> int:
        return self.FHE_DEPTH


class ResNetBlock(nn.Module):
    """
    Standard ResNet block (baseline for comparison).
    Uses BN + ReLU, not directly FHE compatible.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class AdaptiveAvgPool2x2(nn.Module):
    """
    Adaptive average pooling with output size (2, 2).

    From paper Section 3.3:
    'We customized it to have an output size of (2, 2), resulting
     in 256-dimensional features. Our (2, 2) pooling is implemented
     as four separate (1, 1) pooling operations, which permute elements
     of the feature vector.'

    In FHE: implemented as 4 separate (1,1) pooling ops to handle
    feature vector permutation efficiently.
    """

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)
