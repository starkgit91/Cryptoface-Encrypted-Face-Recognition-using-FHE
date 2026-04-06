"""
Jigsaw Puzzle Auxiliary Loss (Section 3.2 of the paper).

From paper:
'We introduce a jigsaw puzzle auxiliary task to supplement positional
information. Specifically, we use the local features y1, y2, ..., yL
to predict their original positions (1, 2, ..., L) via a fully
connected layer.'

Training objective (Equation 6):
  L(ω, W, A, b) = L_ArcFace(ω, W, A, b) + α * L_Jigsaw(ω)

where α = 0.005 from the paper.

The jigsaw task forces each PCNN to encode positional information in its
local features, which naturally complements the patch-based design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import random


class JigsawLoss(nn.Module):
    """
    Jigsaw Puzzle auxiliary loss.

    Given L local features {y_i}, predict the original position of each patch.
    This is a classification task: predict which of the L positions each
    local feature came from.

    During training, we optionally shuffle the patches to make the task
    non-trivial (the model must predict the original un-shuffled position).

    Args:
        n_patches:   Number of patches (L) = number of positions to classify
        feature_dim: Dimension of each local feature
        alpha:       Loss weight (0.005 from paper)
    """

    def __init__(self, n_patches: int, feature_dim: int, alpha: float = 0.005):
        super().__init__()
        self.n_patches = n_patches
        self.feature_dim = feature_dim
        self.alpha = alpha

        # Classification head (attached to CryptoFaceNet's jigsaw_head)
        # Just stores the loss computation logic here

    def forward(self, position_logits: torch.Tensor,
                position_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute jigsaw position prediction loss.

        Args:
            position_logits: (B*L, n_patches) logits from JigsawHead
            position_labels: (B*L,) ground-truth positions

        Returns:
            Weighted jigsaw loss: α * CE(logits, labels)
        """
        loss = F.cross_entropy(position_logits, position_labels.long())
        return self.alpha * loss

    def create_position_labels(self, batch_size: int,
                               device: torch.device) -> torch.Tensor:
        """
        Create ground-truth position labels.

        Each image has L patches at positions 0, 1, ..., L-1.
        Labels are repeated for each image in the batch.

        Returns:
            labels: (B*L,) position labels
        """
        positions = torch.arange(self.n_patches, device=device)  # [0, 1, ..., L-1]
        labels = positions.repeat(batch_size)  # (B*L,)
        return labels

    def extra_repr(self) -> str:
        return f'n_patches={self.n_patches}, alpha={self.alpha}'


class CombinedLoss(nn.Module):
    """
    Combined ArcFace + Jigsaw training loss (Equation 6).

    L = L_ArcFace + α * L_Jigsaw

    Args:
        arcface_loss: ArcFace loss module
        jigsaw_loss:  Jigsaw loss module
        n_patches:    Number of patches
    """

    def __init__(self, arcface_loss: nn.Module, jigsaw_loss: JigsawLoss,
                 n_patches: int):
        super().__init__()
        self.arcface = arcface_loss
        self.jigsaw = jigsaw_loss
        self.n_patches = n_patches

    def forward(self, global_features: torch.Tensor,
                local_features: List[torch.Tensor],
                labels: torch.Tensor,
                jigsaw_head: nn.Module) -> dict:
        """
        Args:
            global_features: (B, d) L2-normalized global features
            local_features:  List of L tensors (B, d) local features
            labels:          (B,) identity labels
            jigsaw_head:     JigsawHead module from CryptoFaceNet

        Returns:
            dict with 'total', 'arcface', 'jigsaw' losses
        """
        B = global_features.shape[0]
        device = global_features.device

        # ArcFace loss on global features
        arc_loss = self.arcface(global_features, labels)

        # Jigsaw loss on local features
        position_logits = jigsaw_head(local_features)  # (B*L, L)
        position_labels = self.jigsaw.create_position_labels(B, device)
        jig_loss = self.jigsaw(position_logits, position_labels)

        total_loss = arc_loss + jig_loss

        return {
            'total': total_loss,
            'arcface': arc_loss.item(),
            'jigsaw': jig_loss.item(),
        }
