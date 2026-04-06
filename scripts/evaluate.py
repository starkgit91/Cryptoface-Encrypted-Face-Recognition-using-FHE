#!/usr/bin/env python3
"""
CryptoFace Evaluation Script.

Evaluates a trained CryptoFaceNet checkpoint on standard FR benchmarks.

Usage:
  # Evaluate on all benchmarks:
  python scripts/evaluate.py --checkpoint checkpoints/CryptoFaceNet4_best.pth

  # Evaluate with synthetic data (test pipeline):
  python scripts/evaluate.py --synthetic

  # Evaluate on specific datasets:
  python scripts/evaluate.py --checkpoint ... --datasets lfw agedb
"""

import sys
import os
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.default import get_config, PAPER_PARAMS
from models.cryptofacenet import build_cryptofacenet
from data.dataset import build_eval_loader
from evaluation.evaluator import FREvaluator, format_results_table
from utils.visualization import generate_all_figures


def load_model(checkpoint_path: str, device: torch.device):
    """Load CryptoFaceNet from checkpoint."""
    state = torch.load(checkpoint_path, map_location=device)
    cfg = state.get('config', {})

    resolution = cfg.get('resolution', 64)
    patch_size = cfg.get('patch_size', 32)
    feature_dim = cfg.get('feature_dim', 256)

    model = build_cryptofacenet(
        resolution=resolution,
        patch_size=patch_size,
        feature_dim=feature_dim,
    )
    model.load_state_dict(state['model_state'])
    model = model.to(device)
    model.eval()

    print(f'Loaded {model.get_model_name()} from {checkpoint_path}')
    print(f'Epoch: {state.get("epoch", "unknown")}')
    print(f'Best avg acc: {state.get("best_avg_acc", 0):.2f}%')

    return model, cfg


def main():
    parser = get_config()
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data for testing')
    parser.add_argument('--save_embeddings', action='store_true',
                        help='Save embeddings to disk')
    parser.add_argument('--compare_paper', action='store_true',
                        help='Compare results with paper numbers')
    config = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Build/Load Model ──────────────────────────────────────────────────────
    if config.checkpoint and os.path.exists(config.checkpoint):
        model, model_cfg = load_model(config.checkpoint, device)
        resolution = model_cfg.get('resolution', config.resolution)
    else:
        print(f'No checkpoint found. Building model with resolution={config.resolution}')
        resolution = config.resolution
        model = build_cryptofacenet(
            resolution=resolution,
            patch_size=config.patch_size,
            feature_dim=config.feature_dim,
        ).to(device)

    evaluator = FREvaluator(model=model, device=device, n_folds=config.num_folds)

    # ── Evaluate on each benchmark ────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'Evaluating {model.get_model_name()} on {config.datasets}')
    print(f'{"="*60}')

    all_results = {}
    for dataset_name in config.datasets:
        data_root = os.path.join(config.test_data_root, dataset_name)
        pairs_file = os.path.join(data_root, 'pairs.txt')

        loader = build_eval_loader(
            data_root=data_root,
            pairs_file=pairs_file if os.path.exists(pairs_file) else '',
            resolution=resolution,
            batch_size=config.eval_batch_size,
            synthetic=config.synthetic,
        )

        results = evaluator.evaluate_dataset(loader, dataset_name.upper())
        all_results[dataset_name] = results

    # ── Print results table ───────────────────────────────────────────────────
    print(format_results_table(all_results, model.get_model_name()))

    # ── Compare with paper ────────────────────────────────────────────────────
    if config.compare_paper:
        model_key = model.get_model_name()
        if model_key in PAPER_PARAMS['paper_results']:
            paper = PAPER_PARAMS['paper_results'][model_key]
            print(f'\n{"="*60}')
            print(f'Comparison with Paper (Table 1):')
            print(f'{"Dataset":<12} {"Ours":>10} {"Paper":>10} {"Diff":>8}')
            print(f'{"-"*42}')

            ds_map = {
                'lfw': 'LFW', 'agedb': 'AgeDB', 'calfw': 'CALFW',
                'cplfw': 'CPLFW', 'cfpfp': 'CFP-FP',
            }
            for ds, paper_key in ds_map.items():
                if ds in all_results and paper_key in paper:
                    ours = all_results[ds].get('accuracy', 0)
                    paper_val = paper[paper_key]
                    diff = ours - paper_val
                    marker = '↑' if diff > 0 else ('↓' if diff < 0 else '=')
                    print(f'{paper_key:<12} {ours:>9.2f}% {paper_val:>9.2f}% '
                          f'{marker}{abs(diff):>6.2f}%')

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs(config.results_dir, exist_ok=True)
    results_file = os.path.join(config.results_dir, f'{model.get_model_name()}_results.json')

    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for ds, r in all_results.items():
        json_results[ds] = {
            k: float(v) if isinstance(v, (float, np.floating)) else
               v.tolist() if hasattr(v, 'tolist') else v
            for k, v in r.items()
            if k not in ('fpr', 'tpr', 'fold_accuracies')  # skip large arrays
        }

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f'\nResults saved to {results_file}')

    # ── Generate figures ──────────────────────────────────────────────────────
    generate_all_figures(
        output_dir=config.results_dir,
        eval_results=all_results,
    )


if __name__ == '__main__':
    main()
