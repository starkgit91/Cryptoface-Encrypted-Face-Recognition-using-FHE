# CryptoFace: End-to-End Encrypted Face Recognition

Implementation of [CryptoFace: End-to-End Encrypted Face Recognition (CVPR 2025)](https://arxiv.org/abs/xxxx).

## Overview

CryptoFace is the first end-to-end encrypted face recognition system using Fully Homomorphic Encryption (FHE). This repository implements:

- **CryptoFaceNet**: Mixture of shallow patch CNNs optimized for FHE inference
- **Training Pipeline**: ArcFace loss + Jigsaw Puzzle auxiliary task
- **FHE Simulation**: Multiplicative depth tracking and latency estimation
- **Evaluation**: Standard FR benchmarks (LFW, AgeDB, CALFW, CPLFW, CFP-FP)
- **Visualization**: All paper figures reproduced

## Architecture

```
CryptoFace System:
  Client: Encrypt(face_image) → encrypted_face
  Server: CryptoFaceNet([face]) → [feature]
  Server: Match([feature1], [feature2]) → [score]
  Client: Decrypt(score) → match/no-match

CryptoFaceNet:
  Input → Split into L patches
  [PCNN_1(patch_1), ..., PCNN_L(patch_L)] (parallel)
  → Feature Fusion → L2 Norm → Feature
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
# Download WebFace4M dataset first, then:
python scripts/train.py \
    --data_root /path/to/webface4m \
    --patch_size 32 \
    --resolution 64 \
    --batch_size 256 \
    --epochs 26 \
    --lr 0.05

# For multiple resolutions:
python scripts/train.py --resolution 64  # CryptoFaceNet4  (4 patches)
python scripts/train.py --resolution 96  # CryptoFaceNet9  (9 patches)
python scripts/train.py --resolution 128 # CryptoFaceNet16 (16 patches)
```

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/cryptofacenet4_best.pth \
    --test_data_root /path/to/lfw \
    --datasets lfw agedb calfw cplfw cfpfp
```

## FHE Latency Estimation

```bash
python scripts/fhe_analysis.py \
    --checkpoint checkpoints/cryptofacenet4_best.pth \
    --resolution 64
```

## Plot Results

```bash
python scripts/plot_results.py --output results/
```

## Results (Table 1 from paper)

| Method | Backbone | Res | LFW | AgeDB | CALFW | CPLFW | CFP-FP | Avg | Latency(s) |
|--------|----------|-----|-----|-------|-------|-------|--------|-----|------------|
| MPCNN  | ResNet44 | 64  | 98.27 | 87.45 | 90.85 | 83.72 | 87.90 | 89.64 | 9,845 |
| AutoFHE| ResNet32 | 64  | 93.53 | 80.88 | 85.40 | 75.67 | 77.96 | 82.69 | 4,001 |
| **CryptoFace** | CryptoFaceNet4 | 64 | **98.87** | **89.45** | **91.60** | **81.98** | **85.21** | **89.42** | **1,364** |
| CryptoFace | CryptoFaceNet9 | 96 | 99.18 | 91.38 | 93.32 | 84.23 | 86.81 | 90.99 | 1,395 |
| CryptoFace | CryptoFaceNet16 | 128| 98.78 | 92.90 | 93.73 | 83.95 | 87.94 | 91.46 | 1,446 |

## File Structure

```
cryptoface/
├── models/
│   ├── cryptofacenet.py      # Main CryptoFaceNet architecture
│   ├── blocks.py             # CryptoFaceNet convolutional blocks
│   ├── activations.py        # HerPN polynomial activation
│   └── resnet.py             # Baseline ResNet for comparison
├── losses/
│   ├── arcface.py            # ArcFace loss
│   └── jigsaw.py             # Jigsaw puzzle auxiliary loss
├── data/
│   ├── dataset.py            # WebFace4M, LFW, AgeDB etc.
│   └── transforms.py         # Data augmentation
├── training/
│   └── trainer.py            # Training loop
├── evaluation/
│   └── evaluator.py          # 10-fold cross-validation FR evaluation
├── fhe/
│   ├── polynomial_approx.py  # L2 norm polynomial approximation
│   ├── depth_tracker.py      # Multiplicative depth tracking
│   └── fhe_simulator.py      # FHE latency simulation
├── utils/
│   ├── metrics.py            # Accuracy, AUC, ROC metrics
│   └── visualization.py      # All paper figure reproductions
├── scripts/
│   ├── train.py              # Main training script
│   ├── evaluate.py           # Evaluation script
│   ├── fhe_analysis.py       # FHE depth & latency analysis
│   └── plot_results.py       # Reproduce paper figures
└── config/
    └── default.py            # All hyperparameters
```

## Citation

```bibtex
@inproceedings{ao2025cryptoface,
  title={CryptoFace: End-to-End Encrypted Face Recognition},
  author={Ao, Wei and Boddeti, Vishnu Naresh},
  booktitle={CVPR},
  year={2025}
}
```
