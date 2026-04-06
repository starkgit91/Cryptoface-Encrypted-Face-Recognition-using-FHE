"""
Polynomial activation functions for FHE-compatible neural networks.

HerPN (Hermite Polynomial Normalization) from AESPA [Park et al., 2022]:
  - ax^2 + bx + c  (degree-2 Hermite polynomial)
  - In CryptoFaceNet, we fuse coefficient 'a' into Conv weights,
    yielding x^2 + (b/a)x + (c/a) with multiplicative depth 1 (saves 1 level)

In cleartext training we keep full learnable coefficients (a, b, c).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HerPN(nn.Module):
    """
    Hermite Polynomial Normalization activation.

    Computes: a*x^2 + b*x + c per channel (or globally).

    In FHE context:
      - Depth cost: 1 (CryptoFaceNet fused) or 2 (standard AESPA)
      - 'a' is fused into preceding Conv weights in CryptoFaceNet to save 1 level

    Args:
        num_features (int): Number of feature channels for per-channel coefficients.
        fused (bool): If True, simulate fused mode (a=1, only learn b, c).
                      This mimics the CryptoFaceNet optimization.
        init_a (float): Initial value for coefficient a.
        init_b (float): Initial value for coefficient b.
        init_c (float): Initial value for coefficient c.
    """

    def __init__(self, num_features: int, fused: bool = True,
                 init_a: float = 1.0, init_b: float = 0.0, init_c: float = 0.0):
        super().__init__()
        self.num_features = num_features
        self.fused = fused

        # Per-channel learnable coefficients
        if not fused:
            self.a = nn.Parameter(torch.full((1, num_features, 1, 1), init_a))
        self.b = nn.Parameter(torch.full((1, num_features, 1, 1), init_b))
        self.c = nn.Parameter(torch.full((1, num_features, 1, 1), init_c))

        self._init_weights()

    def _init_weights(self):
        """Initialize with small perturbation around identity-like behavior."""
        if not self.fused:
            nn.init.constant_(self.a, 1.0)
        nn.init.constant_(self.b, 0.01)
        nn.init.constant_(self.c, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Polynomial-activated tensor
        """
        if self.fused:
            # x^2 + b*x + c  (a is fused into preceding conv)
            return x * x + self.b * x + self.c
        else:
            # a*x^2 + b*x + c
            return self.a * x * x + self.b * x + self.c

    def fhe_depth(self) -> int:
        """Return multiplicative depth consumed by this activation in FHE."""
        return 1 if self.fused else 2

    def extra_repr(self) -> str:
        return f'num_features={self.num_features}, fused={self.fused}'


class HerPN1D(nn.Module):
    """HerPN for 1D features (after flatten/pooling)."""

    def __init__(self, num_features: int, fused: bool = True):
        super().__init__()
        self.num_features = num_features
        self.fused = fused

        if not fused:
            self.a = nn.Parameter(torch.ones(1, num_features))
        self.b = nn.Parameter(torch.zeros(1, num_features))
        self.c = nn.Parameter(torch.zeros(1, num_features))

        self._init_weights()

    def _init_weights(self):
        if not self.fused:
            nn.init.constant_(self.a, 1.0)
        nn.init.constant_(self.b, 0.01)
        nn.init.constant_(self.c, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            return x * x + self.b * x + self.c
        else:
            return self.a * x * x + self.b * x + self.c

    def fhe_depth(self) -> int:
        return 1 if self.fused else 2


class SquaredActivation(nn.Module):
    """Simple x^2 activation (used in some LHENets like CryptoNets)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x

    def fhe_depth(self) -> int:
        return 1


class ChebyshevActivation(nn.Module):
    """
    Chebyshev polynomial approximation of ReLU.
    Used as a smooth reference implementation.
    In FHE, this has high multiplicative depth.
    """

    def __init__(self, degree: int = 27):
        super().__init__()
        self.degree = degree
        self._compute_coefficients()

    def _compute_coefficients(self):
        """Minimax Chebyshev coefficients for ReLU on [-1,1]."""
        # Approximate coefficients for ReLU(x) ≈ Σ c_k T_k(x)
        # For training we just use actual ReLU, Chebyshev is for FHE
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)  # In cleartext, use true ReLU

    def fhe_depth(self) -> int:
        # depth ≈ log2(degree) for Chebyshev evaluation
        return int(math.ceil(math.log2(self.degree))) + 1


def get_activation(name: str, num_features: int, fused: bool = True):
    """
    Factory function for FHE-compatible activations.

    Args:
        name: 'herp', 'square', 'relu', 'silu'
        num_features: Number of feature channels
        fused: Whether to use fused mode for HerPN

    Returns:
        nn.Module activation
    """
    name = name.lower()
    if name in ('herp', 'herpn', 'hermite'):
        return HerPN(num_features, fused=fused)
    elif name == 'square':
        return SquaredActivation()
    elif name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'silu':
        return nn.SiLU()
    elif name == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f'Unknown activation: {name}')


# ── Depth registry ────────────────────────────────────────────────────────────
ACTIVATION_DEPTH = {
    'herp': 1,      # fused CryptoFaceNet
    'herpn_full': 2,  # standard AESPA
    'square': 1,
    'relu': 14,     # Chebyshev composite approximation
    'silu': 7,      # degree-63 Chebyshev
}
