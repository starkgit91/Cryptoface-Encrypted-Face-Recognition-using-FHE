"""
CryptoFaceNet: Mixture of Patch Convolutional Neural Networks.

Architecture from Section 3.2 and Figure 3, 5 of the paper:

Training:
  x (C×H×W) → split into L patches (C×P×P)
  [PCNN_1, ..., PCNN_L] → [y1, ..., yL] (L × d local features)
  y = y'A^T + b               (feature fusion, Equation 5)
  + JigsawPuzzleHead          (auxiliary positional task)
  → ArcFace loss

Inference (FHE):
  Parallel evaluation of L PCNNs
  Fusion: y = Σ_i (y_i A_i^T + b/L)   (Equation 7, decomposed for FHE)
  L2 Norm via polynomial approximation
  Match via polynomial cosine similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math

from .blocks import CryptoFaceNetBlock, AdaptiveAvgPool2x2
from .activations import HerPN


class PatchCNN(nn.Module):
    """
    Shallow Patch CNN (PCNN) that processes a single image patch.

    Architecture (Figure 5):
      Conv → Block1 → Block2 → Block3 → Block4 → Block5
           → HerPN → BN1D → AvgPool(2×2) → Linear → [L2Norm]

    Chosen to be shallow (low multiplicative depth) so each PCNN
    requires only ONE bootstrapping operation in FHE.

    Args:
        patch_size:    Spatial size of input patch (P×P)
        in_channels:   Number of input channels (3 for RGB)
        feature_dim:   Output feature dimension
        base_channels: Base channel count for conv layers
        patch_idx:     Index of this patch (for weight non-sharing)
    """

    def __init__(self, patch_size: int = 32, in_channels: int = 3,
                 feature_dim: int = 256, base_channels: int = 64,
                 patch_idx: int = 0):
        super().__init__()
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.patch_idx = patch_idx

        C = base_channels  # e.g. 64

        # ── Initial Conv (stem) ───────────────────────────────────────────────
        # stride=2 to reduce spatial size from P to P/2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        # ── Backbone blocks (5 blocks matching Figure 5) ──────────────────────
        # Block 1: C → C, stride=1
        self.block1 = CryptoFaceNetBlock(C, C, stride=1)

        # Block 2: C → 2C, stride=2
        self.block2 = CryptoFaceNetBlock(C, C * 2, stride=2)

        # Block 3: 2C → 2C, stride=1
        self.block3 = CryptoFaceNetBlock(C * 2, C * 2, stride=1)

        # Block 4: 2C → 4C, stride=2
        self.block4 = CryptoFaceNetBlock(C * 2, C * 4, stride=2)

        # Block 5: 4C → 4C, stride=1
        self.block5 = CryptoFaceNetBlock(C * 4, C * 4, stride=1)

        # ── Head ──────────────────────────────────────────────────────────────
        # HerPN activation before pooling
        self.head_act = HerPN(C * 4, fused=True)

        # BatchNorm1D applied after flattening (fused with Linear in FHE)
        self.head_bn = nn.BatchNorm1d(C * 4 * 4)  # 4×4 spatial × C*4 channels

        # Adaptive Average Pooling (2×2) → 4 × C*4 features
        self.pool = AdaptiveAvgPool2x2()

        # Final linear projection: 4*C*4 → feature_dim
        # In FHE: BatchNorm1D fused into Linear
        self.fc = nn.Linear(C * 4 * 4, feature_dim, bias=False)
        self.bn_out = nn.BatchNorm1d(feature_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patch tensor (B, C, P, P)
        Returns:
            Local feature (B, feature_dim)
        """
        # Stem
        out = self.stem(x)

        # Backbone
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)

        # HerPN activation
        out = self.head_act(out)

        # Pool (2×2 adaptive)
        out = self.pool(out)   # (B, C*4, 2, 2)

        # Flatten + BN + Linear
        out = out.flatten(1)   # (B, C*4*4)
        out = self.head_bn(out)
        out = self.fc(out)
        out = self.bn_out(out)

        return out  # (B, feature_dim)

    def fhe_depth(self) -> int:
        """
        Total multiplicative depth of this PCNN.

        Each CryptoFaceNetBlock: 6 levels
        HerPN: 1 level
        AvgPool, Linear: 0 levels (implemented via rotations/additions)
        L2Norm: 2 levels (computed separately)

        Total ≈ 5 blocks × 6 + 1 HerPN = 31 levels
        With bootstrapping budget L_eff, only ONE bootstrap needed.
        """
        depth = 5 * CryptoFaceNetBlock.FHE_DEPTH + 1  # blocks + head HerPN
        return depth


class JigsawHead(nn.Module):
    """
    Jigsaw Puzzle Auxiliary Head (Section 3.2).

    Uses local features to predict original patch positions.
    This auxiliary task forces the PCNNs to learn positional information.

    From paper:
    'We use the local features y1, y2, ..., yL to predict their original
     positions (1, 2, ..., L) via a fully connected layer.'
    """

    def __init__(self, feature_dim: int, n_patches: int):
        super().__init__()
        self.n_patches = n_patches
        # Predict position for each patch (classification over n_patches positions)
        self.fc = nn.Linear(feature_dim, n_patches)

    def forward(self, local_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            local_features: List of L tensors, each (B, feature_dim)
        Returns:
            position_logits: (B*L, n_patches) logits for position prediction
        """
        # Concatenate all patch features
        all_features = torch.stack(local_features, dim=1)  # (B, L, D)
        B, L, D = all_features.shape
        # Flatten batch and patch dimensions
        all_features = all_features.view(B * L, D)
        return self.fc(all_features)


class FeatureFusion(nn.Module):
    """
    Feature fusion module (Equations 5 & 7 from the paper).

    Training (Equation 5):
      y = y'A^T + b    where y' = [y1, y2, ..., yL] stacked row-wise

    FHE Inference (Equation 7 - decomposed for parallel evaluation):
      y = Σ_i (y_i A_i^T + b/L)
      This decomposes the large rectangular matrix A ∈ R^{d×dL}
      into L square matrices A_i ∈ R^{d×d}, enabling efficient FHE
      matrix-vector products and parallel evaluation.

    Args:
        feature_dim:  Local feature dimension (d)
        n_patches:    Number of patches (L)
    """

    def __init__(self, feature_dim: int, n_patches: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_patches = n_patches

        # Full fusion matrix A ∈ R^{d × dL} decomposed as L square matrices
        # A = [A_1, A_2, ..., A_L] where each A_i ∈ R^{d×d}
        self.fusion_layers = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim, bias=False)
            for _ in range(n_patches)
        ])

        # Global bias b ∈ R^d (split as b/L per patch in FHE mode)
        self.bias = nn.Parameter(torch.zeros(feature_dim))

        self._init_weights()

    def _init_weights(self):
        for layer in self.fusion_layers:
            nn.init.kaiming_normal_(layer.weight)
        nn.init.zeros_(self.bias)

    def forward(self, local_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            local_features: List of L tensors, each (B, d)
        Returns:
            global_feature: (B, d)
        """
        # FHE-decomposed fusion (Equation 7):
        # y = Σ_i (y_i A_i^T + b/L)
        bias_per_patch = self.bias / self.n_patches

        fused = sum(
            self.fusion_layers[i](feat) + bias_per_patch
            for i, feat in enumerate(local_features)
        )
        return fused  # (B, d)


class CryptoFaceNet(nn.Module):
    """
    CryptoFaceNet: Mixture of Patch CNNs for FHE-compatible face recognition.

    From paper Section 3.2:
    - Face image split into L = H*W/P^2 non-overlapping patches
    - L separate PCNNs process patches independently (no weight sharing)
    - Feature fusion via decomposed linear projection
    - Training: ArcFace loss + Jigsaw Puzzle auxiliary loss
    - FHE: PCNNs evaluated in parallel, fusion = cheap addition

    Variants:
      CryptoFaceNet4:  resolution=64,  patch_size=32, L=4  patches
      CryptoFaceNet9:  resolution=96,  patch_size=32, L=9  patches
      CryptoFaceNet16: resolution=128, patch_size=32, L=16 patches

    Args:
        resolution:   Input image size (H=W)
        patch_size:   Patch size (P×P), default 32 from paper
        in_channels:  Input channels (3 for RGB)
        feature_dim:  Output embedding dimension (256 from paper)
        base_channels: Base channel count (64)
        num_classes:  Number of training identities
    """

    def __init__(self, resolution: int = 64, patch_size: int = 32,
                 in_channels: int = 3, feature_dim: int = 256,
                 base_channels: int = 64, num_classes: int = 205990):
        super().__init__()
        assert resolution % patch_size == 0, \
            f'Resolution {resolution} must be divisible by patch_size {patch_size}'

        self.resolution = resolution
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.n_patches = (resolution // patch_size) ** 2  # L
        self.num_classes = num_classes

        print(f'CryptoFaceNet: resolution={resolution}, patch_size={patch_size}, '
              f'n_patches={self.n_patches} (CryptoFaceNet{self.n_patches})')

        # ── Mixture of PCNNs (no weight sharing) ─────────────────────────────
        self.pcnns = nn.ModuleList([
            PatchCNN(patch_size=patch_size, in_channels=in_channels,
                     feature_dim=feature_dim, base_channels=base_channels,
                     patch_idx=i)
            for i in range(self.n_patches)
        ])

        # ── Feature Fusion (Equations 5 & 7) ─────────────────────────────────
        self.fusion = FeatureFusion(feature_dim, self.n_patches)

        # ── Jigsaw Puzzle Auxiliary Head ──────────────────────────────────────
        self.jigsaw_head = JigsawHead(feature_dim, self.n_patches)

        self._print_model_info()

    def _print_model_info(self):
        total = sum(p.numel() for p in self.parameters())
        pcnn_params = sum(p.numel() for p in self.pcnns.parameters())
        print(f'Total params: {total/1e6:.2f}M')
        print(f'PCNN params (×{self.n_patches}): {pcnn_params/1e6:.2f}M')

    def split_patches(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Split image into L non-overlapping patches.

        Args:
            x: Input image (B, C, H, W)
        Returns:
            List of L patches, each (B, C, P, P)
        """
        B, C, H, W = x.shape
        P = self.patch_size
        n_h = H // P
        n_w = W // P

        patches = []
        for i in range(n_h):
            for j in range(n_w):
                patch = x[:, :, i*P:(i+1)*P, j*P:(j+1)*P]  # (B, C, P, P)
                patches.append(patch)
        return patches  # List of L patches

    def forward(self, x: torch.Tensor, return_local: bool = False
                ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass for training.

        Args:
            x: Input image (B, C, H, W)
            return_local: Whether to return local features for jigsaw loss

        Returns:
            global_feature: (B, feature_dim) L2-normalized feature
            local_features: (optional) List of L tensors (B, feature_dim)
        """
        # 1. Split into L patches
        patches = self.split_patches(x)  # List[L × (B, C, P, P)]

        # 2. Process each patch with its dedicated PCNN (no weight sharing)
        #    In FHE: evaluated in parallel
        local_features = [
            self.pcnns[i](patches[i]) for i in range(self.n_patches)
        ]  # List[L × (B, D)]

        # 3. Feature fusion (Equation 7)
        global_feature = self.fusion(local_features)  # (B, D)

        # 4. L2 normalization
        #    In FHE: polynomial approximation of 1/||y||
        global_feature = F.normalize(global_feature, p=2, dim=1)

        if return_local:
            return global_feature, local_features
        return global_feature, None

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Clean feature extraction (no auxiliary outputs).

        Args:
            x: Input image (B, C, H, W)
        Returns:
            feature: (B, feature_dim) L2-normalized feature
        """
        feature, _ = self.forward(x, return_local=False)
        return feature

    def fhe_depth_per_pcnn(self) -> int:
        """Total FHE multiplicative depth per PCNN."""
        return self.pcnns[0].fhe_depth()

    def fhe_bootstrap_count(self) -> int:
        """Number of bootstrapping operations needed in FHE (=1 from paper)."""
        return 1

    def get_model_name(self) -> str:
        return f'CryptoFaceNet{self.n_patches}'

    def extra_repr(self) -> str:
        return (f'resolution={self.resolution}, patch_size={self.patch_size}, '
                f'n_patches={self.n_patches}, feature_dim={self.feature_dim}')


def build_cryptofacenet(resolution: int = 64, patch_size: int = 32,
                        num_classes: int = 205990,
                        **kwargs) -> CryptoFaceNet:
    """
    Build CryptoFaceNet for a given resolution.

    Resolution → CryptoFaceNet variant:
      64  → CryptoFaceNet4  (4 patches of 32×32)
      96  → CryptoFaceNet9  (9 patches of 32×32)
      128 → CryptoFaceNet16 (16 patches of 32×32)
    """
    return CryptoFaceNet(
        resolution=resolution,
        patch_size=patch_size,
        num_classes=num_classes,
        **kwargs
    )
