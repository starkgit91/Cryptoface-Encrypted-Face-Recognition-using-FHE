#!/usr/bin/env python3
"""
Main CryptoFace Training Script.

Usage:
  # Train CryptoFaceNet4 (64×64, 4 patches):
  python scripts/train.py --resolution 64 --data_root /path/to/webface4m

  # Train CryptoFaceNet9 (96×96, 9 patches):
  python scripts/train.py --resolution 96 --data_root /path/to/webface4m

  # Test pipeline with synthetic data:
  python scripts/train.py --synthetic --epochs 3

Paper training settings (Section 4):
  - SGD lr=0.05, momentum=0.9, weight_decay=0.0005
  - Epochs=26, LR decay at [12, 20, 24] by ×0.1
  - ArcFace s=64, m=0.5
  - Jigsaw α=0.005
  - Gradient clip=1.0
"""

import sys
import os
import random
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.default import get_config
from models.cryptofacenet import build_cryptofacenet
from training.trainer import CryptoFaceTrainer
from data.dataset import build_train_loader, build_eval_loader, BENCHMARK_CONFIGS
from evaluation.evaluator import FREvaluator, format_results_table
from utils.visualization import generate_all_figures


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def build_eval_fn(config, model, evaluator, test_data_root):
    """Build evaluation function for use during training."""
    def eval_fn(model):
        results = {}
        for dataset_name in config.datasets:
            pairs_file = os.path.join(
                test_data_root, dataset_name, 'pairs.txt')

            loader = build_eval_loader(
                data_root=os.path.join(test_data_root, dataset_name),
                pairs_file=pairs_file if os.path.exists(pairs_file) else '',
                resolution=config.resolution,
                batch_size=config.eval_batch_size,
                synthetic=True,  # Use synthetic if data not found
            )

            dataset_results = evaluator.evaluate_dataset(
                loader, dataset_name.upper())
            results[dataset_name] = dataset_results['accuracy']

        return results
    return eval_fn


def main():
    parser = get_config()
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data (for testing without real dataset)')
    parser.add_argument('--eval_during_training', action='store_true',
                        help='Run evaluation after each epoch')
    config = parser.parse_args()

    # Setup
    set_seed(config.seed)
    device = get_device(config.device)
    print(f'Device: {device}')
    print(f'Resolution: {config.resolution}×{config.resolution}')
    print(f'Patch size: {config.patch_size}×{config.patch_size}')

    # ── Build DataLoader ──────────────────────────────────────────────────────
    print(f'\nLoading training data from: {config.data_root}')
    train_loader, num_classes = build_train_loader(
        data_root=config.data_root,
        resolution=config.resolution,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        synthetic=config.synthetic,
    )
    print(f'Train samples: {len(train_loader.dataset):,}')
    print(f'Identities: {num_classes:,}')

    # ── Build Model ───────────────────────────────────────────────────────────
    print(f'\nBuilding CryptoFaceNet...')
    model = build_cryptofacenet(
        resolution=config.resolution,
        patch_size=config.patch_size,
        num_classes=num_classes,
        feature_dim=config.feature_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Parameters: {total_params:.2f}M')
    print(f'FHE depth per PCNN: {model.fhe_depth_per_pcnn()}')
    print(f'FHE bootstraps needed: {model.fhe_bootstrap_count()}')

    # ── Build Trainer ─────────────────────────────────────────────────────────
    cfg_dict = vars(config)
    trainer = CryptoFaceTrainer(
        model=model,
        num_classes=num_classes,
        device=device,
        config=cfg_dict,
        log_dir=os.path.join(config.log_dir, model.get_model_name()),
        checkpoint_dir=config.checkpoint_dir,
    )

    # ── Load checkpoint if provided ───────────────────────────────────────────
    start_epoch = 0
    if config.checkpoint:
        start_epoch = trainer.load_checkpoint(config.checkpoint)

    # ── Build eval function ───────────────────────────────────────────────────
    eval_fn = None
    if config.eval_during_training:
        evaluator = FREvaluator(model=model, device=device,
                                n_folds=config.num_folds)
        eval_fn = build_eval_fn(config, model, evaluator,
                                config.test_data_root)

    # ── Train ─────────────────────────────────────────────────────────────────
    history = trainer.train(
        train_loader=train_loader,
        epochs=config.epochs,
        eval_fn=eval_fn,
        start_epoch=start_epoch,
    )

    # ── Final Evaluation ──────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('Running final evaluation...')
    print('='*60)

    evaluator = FREvaluator(model=model, device=device, n_folds=config.num_folds)

    eval_results = {}
    for dataset_name in config.datasets:
        loader = build_eval_loader(
            data_root=os.path.join(config.test_data_root, dataset_name),
            pairs_file=os.path.join(config.test_data_root, dataset_name, 'pairs.txt'),
            resolution=config.resolution,
            batch_size=config.eval_batch_size,
            synthetic=config.synthetic,
        )
        results = evaluator.evaluate_dataset(loader, dataset_name.upper())
        eval_results[dataset_name] = results

    print(format_results_table(eval_results, model.get_model_name()))

    # ── Generate figures ──────────────────────────────────────────────────────
    print('\nGenerating paper figures...')
    generate_all_figures(
        output_dir=config.results_dir,
        history=history,
        eval_results=eval_results,
    )

    print(f'\nTraining complete!')
    print(f'Checkpoints: {config.checkpoint_dir}')
    print(f'Results: {config.results_dir}')


if __name__ == '__main__':
    main()
