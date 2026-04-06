"""
Dataset loading for CryptoFace.

Training: WebFace4M (subset of WebFace260M)
  - 205,990 identities, 4,235,242 face images

Evaluation benchmarks (standard 10-fold cross-validation):
  - LFW    (6,000 pairs)
  - AgeDB  (6,000 pairs)
  - CALFW  (6,000 pairs)
  - CPLFW  (6,000 pairs)
  - CFP-FP (7,000 pairs)
  - IJB-B  (mixed quality)
  - IJB-C  (mixed quality)
"""

import os
import random
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image


# ── Transforms ───────────────────────────────────────────────────────────────

def get_train_transforms(resolution: int = 64) -> T.Compose:
    """
    Training augmentation pipeline.
    Follows AdaFace [Kim et al., 2022] augmentation protocol.
    """
    return T.Compose([
        T.Resize((resolution, resolution)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_eval_transforms(resolution: int = 64) -> T.Compose:
    """Evaluation transforms (no augmentation)."""
    return T.Compose([
        T.Resize((resolution, resolution)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# ── Training Dataset ──────────────────────────────────────────────────────────

class WebFace4MDataset(Dataset):
    """
    WebFace4M dataset for training.

    Expected directory structure:
        data_root/
            identity_0001/
                img_001.jpg
                img_002.jpg
                ...
            identity_0002/
                ...

    Also supports InsightFace format (data_root/label.txt).

    Args:
        data_root:  Path to WebFace4M root directory
        resolution: Input resolution
        transform:  Optional custom transform
    """

    def __init__(self, data_root: str, resolution: int = 64,
                 transform: Optional[T.Compose] = None):
        self.data_root = Path(data_root)
        self.resolution = resolution
        self.transform = transform or get_train_transforms(resolution)

        self.samples: List[Tuple[Path, int]] = []
        self.num_classes = 0

        self._load_dataset()

    def _load_dataset(self):
        """Load dataset from directory structure or label file."""
        label_file = self.data_root / 'label.txt'

        if label_file.exists():
            # InsightFace format: image_path label_id
            self._load_from_label_file(label_file)
        else:
            # Directory format: each subdirectory is an identity
            self._load_from_dirs()

        print(f'Loaded {len(self.samples)} samples, '
              f'{self.num_classes} identities from {self.data_root}')

    def _load_from_label_file(self, label_file: Path):
        identity_set = set()
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                img_path, label = parts[0], int(parts[1])
                full_path = self.data_root / img_path
                if full_path.exists():
                    self.samples.append((full_path, label))
                    identity_set.add(label)
        self.num_classes = len(identity_set)

    def _load_from_dirs(self):
        id_dirs = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        for label, id_dir in enumerate(id_dirs):
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
                for img_path in id_dir.glob(ext):
                    self.samples.append((img_path, label))
        self.num_classes = len(id_dirs)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


class SyntheticFaceDataset(Dataset):
    """
    Synthetic face dataset for testing without real data.
    Generates random tensors as placeholder.
    Use this to test the pipeline without downloading WebFace4M.
    """

    def __init__(self, num_samples: int = 10000, num_classes: int = 1000,
                 resolution: int = 64):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.resolution = resolution

        # Pre-generate labels
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Random face-like tensor
        img = torch.randn(3, self.resolution, self.resolution)
        label = self.labels[idx].item()
        return img, label


# ── Evaluation Datasets ───────────────────────────────────────────────────────

class FacePairDataset(Dataset):
    """
    Face verification dataset (pairs of images + same/different label).

    Supports standard benchmarks: LFW, AgeDB, CALFW, CPLFW, CFP-FP.

    Expected pair file format (space-separated):
        img1_path img2_path label (1=same, 0=different)

    Args:
        data_root:  Path to dataset root
        pairs_file: Path to pairs file (or will look for standard names)
        resolution: Target resolution
        transform:  Optional custom transform
    """

    def __init__(self, data_root: str, pairs_file: str,
                 resolution: int = 64,
                 transform: Optional[T.Compose] = None):
        self.data_root = Path(data_root)
        self.transform = transform or get_eval_transforms(resolution)

        self.pairs: List[Tuple[Path, Path, int]] = []
        self._load_pairs(pairs_file)

    def _load_pairs(self, pairs_file: str):
        with open(pairs_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    img1 = self.data_root / parts[0]
                    img2 = self.data_root / parts[1]
                    label = int(parts[2])
                    self.pairs.append((img1, img2, label))

        print(f'Loaded {len(self.pairs)} pairs from {pairs_file}')

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img1_path, img2_path, label = self.pairs[idx]

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label


class SyntheticPairDataset(Dataset):
    """Synthetic pair dataset for testing the evaluation pipeline."""

    def __init__(self, num_pairs: int = 6000, resolution: int = 64):
        self.num_pairs = num_pairs
        self.resolution = resolution
        self.labels = torch.randint(0, 2, (num_pairs,))

    def __len__(self) -> int:
        return self.num_pairs

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img1 = torch.randn(3, self.resolution, self.resolution)
        img2 = torch.randn(3, self.resolution, self.resolution)
        if self.labels[idx] == 1:
            # Same pair: add small noise to img1
            img2 = img1 + torch.randn_like(img1) * 0.1
        label = self.labels[idx].item()
        return img1, img2, label


# ── DataLoader Builders ───────────────────────────────────────────────────────

def build_train_loader(data_root: str, resolution: int = 64,
                       batch_size: int = 256, num_workers: int = 8,
                       synthetic: bool = False) -> Tuple[DataLoader, int]:
    """
    Build training DataLoader.

    Args:
        data_root:   Path to WebFace4M
        resolution:  Target resolution
        batch_size:  Training batch size
        num_workers: DataLoader workers
        synthetic:   Use synthetic data if True (for testing)

    Returns:
        (dataloader, num_classes)
    """
    if synthetic or not Path(data_root).exists():
        print(f'[WARNING] data_root {data_root} not found. '
              f'Using synthetic data for testing.')
        dataset = SyntheticFaceDataset(
            num_samples=50000, num_classes=1000, resolution=resolution)
    else:
        dataset = WebFace4MDataset(data_root, resolution=resolution)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, dataset.num_classes


def build_eval_loader(data_root: str, pairs_file: str,
                      resolution: int = 64, batch_size: int = 512,
                      num_workers: int = 4,
                      synthetic: bool = False) -> DataLoader:
    """Build evaluation DataLoader for a face pair dataset."""
    if synthetic or not Path(data_root).exists():
        dataset = SyntheticPairDataset(num_pairs=6000, resolution=resolution)
    else:
        dataset = FacePairDataset(data_root, pairs_file, resolution=resolution)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


# ── Standard Benchmark Paths ──────────────────────────────────────────────────

BENCHMARK_CONFIGS = {
    'lfw': {
        'num_pairs': 6000,
        'pairs_file': 'pairs.txt',
        'description': 'Labeled Faces in the Wild',
    },
    'agedb': {
        'num_pairs': 6000,
        'pairs_file': 'pairs.txt',
        'description': 'AgeDB-30',
    },
    'calfw': {
        'num_pairs': 6000,
        'pairs_file': 'pairs.txt',
        'description': 'Cross-Age LFW',
    },
    'cplfw': {
        'num_pairs': 6000,
        'pairs_file': 'pairs.txt',
        'description': 'Cross-Pose LFW',
    },
    'cfpfp': {
        'num_pairs': 7000,
        'pairs_file': 'pairs.txt',
        'description': 'CFP Frontal-Profile',
    },
}
