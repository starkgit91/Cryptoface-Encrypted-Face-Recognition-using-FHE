"""
ArcFace: Additive Angular Margin Loss (Equation 4 from the paper).

Reference: Deng et al., ArcFace: Additive Angular Margin Loss for
           Deep Face Recognition, TPAMI 2022.

L_ArcFace(ω) = -log [e^{s·cos(θ_i + m)} /
               (e^{s·cos(θ_i + m)} + Σ_{j≠i} e^{s·cos(θ_j)})]

Where:
  θ_i: angle between feature y and its identity center M[i]
  m:   additive angular margin (default 0.5 from paper)
  s:   scale factor (default 64.0 from paper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFace(nn.Module):
    """
    ArcFace loss with learnable feature centers.

    Args:
        feature_dim: Embedding dimension (d)
        num_classes: Number of identities (M) — 205,990 for WebFace4M
        s:           Scale factor (64.0 from paper)
        m:           Angular margin in radians (0.5 from paper)
        easy_margin: Whether to use easy margin variant
    """

    def __init__(self, feature_dim: int = 256, num_classes: int = 205990,
                 s: float = 64.0, m: float = 0.5, easy_margin: bool = False):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        # Feature centers W ∈ R^{M × d}
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin trig values
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: L2-normalized embeddings (B, d)
            labels:   Identity labels (B,)
        Returns:
            loss: Scalar ArcFace loss
        """
        # Normalize feature centers
        normed_weight = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity: cos(θ_j) = features · weight^T
        cosine = F.linear(features, normed_weight)  # (B, M)
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)

        # Compute sin(θ_i)
        sine = torch.sqrt(1.0 - cosine ** 2)

        # cos(θ_i + m) = cos(θ_i)cos(m) - sin(θ_i)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Select phi for true class, cosine for others
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        loss = F.cross_entropy(output, labels.long())
        return loss

    def get_centers(self) -> torch.Tensor:
        """Return normalized feature centers."""
        return F.normalize(self.weight, p=2, dim=1)

    def extra_repr(self) -> str:
        return (f'feature_dim={self.feature_dim}, '
                f'num_classes={self.num_classes}, '
                f's={self.s}, m={self.m}')


class CosFace(nn.Module):
    """
    CosFace (Large Margin Cosine Loss) — alternative to ArcFace.
    L = -log [e^{s·(cos(θ_i) - m)} / (e^{s·(cos(θ_i) - m)} + Σ_{j≠i} e^{s·cos(θ_j)})]
    """

    def __init__(self, feature_dim: int = 256, num_classes: int = 205990,
                 s: float = 64.0, m: float = 0.35):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        normed_weight = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(features, normed_weight)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = cosine - one_hot * self.m
        output *= self.s
        return F.cross_entropy(output, labels.long())
