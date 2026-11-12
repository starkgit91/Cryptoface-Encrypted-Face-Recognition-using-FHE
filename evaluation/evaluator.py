"""
Face Recognition Evaluation with 10-fold Cross-Validation.

From paper Section 4.1:
'We follow the standard 10-fold cross-validation to benchmark CryptoFace
and baselines on the encrypted face datasets. For each split, nine groups
(cleartext) are used to estimate the threshold and polynomial approximation
of ℓ2 normalization, while the standalone group (ciphertext) is used to test
performance.'

Evaluation metrics:
  - Verification accuracy (%)
  - ROC curve (TPR vs FPR)
  - AUC score
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d


class FREvaluator:
    """
    Face Recognition evaluator with 10-fold cross-validation.

    Matches the exact evaluation protocol from the paper:
    - 9 folds for threshold estimation
    - 1 fold for testing
    - Report average accuracy across all 10 splits

    Args:
        model:    CryptoFaceNet model
        device:   Torch device
        n_folds:  Number of cross-validation folds (10)
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 n_folds: int = 10):
        self.model = model
        self.device = device
        self.n_folds = n_folds

    @torch.no_grad()
    def extract_embeddings(self, loader: DataLoader
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract face embeddings for all pairs in the dataset.

        Args:
            loader: DataLoader yielding (img1, img2, label) tuples

        Returns:
            emb1, emb2: (N, D) numpy arrays of embeddings
            labels:     (N,) numpy array of 0/1 labels
        """
        self.model.eval()
        all_emb1, all_emb2, all_labels = [], [], []

        for batch in loader:
            img1, img2, label = batch
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)

            # Extract L2-normalized features
            emb1 = self.model.extract_features(img1)
            emb2 = self.model.extract_features(img2)

            all_emb1.append(emb1.cpu().numpy())
            all_emb2.append(emb2.cpu().numpy())
            all_labels.append(label.numpy())

        emb1 = np.concatenate(all_emb1, axis=0)  # (N, D)
        emb2 = np.concatenate(all_emb2, axis=0)  # (N, D)
        labels = np.concatenate(all_labels, axis=0)  # (N,)

        return emb1, emb2, labels

    def cosine_similarity(self, emb1: np.ndarray,
                          emb2: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity (or equivalently, L2 distance between
        normalized features).

        Score(y1, y2) = ||y1/||y1|| - y2/||y2||||_2^2 = 2 - 2*cos(θ)

        For normalized features, dot product = cosine similarity.
        Higher score → more similar → same person.

        Returns:
            scores: (N,) array, range [-1, 1] for cosine similarity
        """
        # Both emb1, emb2 are already L2-normalized by model
        scores = np.sum(emb1 * emb2, axis=1)  # dot product = cosine similarity
        return scores

    def evaluate_fold(self, scores: np.ndarray, labels: np.ndarray,
                      train_idx: np.ndarray, test_idx: np.ndarray
                      ) -> float:
        """
        Evaluate one cross-validation fold.

        Train threshold on train_idx, evaluate on test_idx.

        Returns:
            accuracy: Verification accuracy on test fold (%)
        """
        train_scores = scores[train_idx]
        train_labels = labels[train_idx]
        test_scores = scores[test_idx]
        test_labels = labels[test_idx]

        # Find optimal threshold on training fold
        threshold = self._find_optimal_threshold(train_scores, train_labels)

        # Evaluate on test fold
        predictions = (test_scores >= threshold).astype(int)
        accuracy = np.mean(predictions == test_labels) * 100.0

        return accuracy

    def _find_optimal_threshold(self, scores: np.ndarray,
                                labels: np.ndarray) -> float:
        """
        Find threshold that maximizes accuracy on the training fold.
        Also used for polynomial ℓ2 normalization threshold estimation in FHE.
        """
        # Search over candidate thresholds
        thresholds = np.linspace(scores.min(), scores.max(), 1000)
        best_acc = 0.0
        best_thresh = 0.0

        for t in thresholds:
            preds = (scores >= t).astype(int)
            acc = np.mean(preds == labels)
            if acc > best_acc:
                best_acc = acc
                best_thresh = t

        return best_thresh

    def evaluate_dataset(self, loader: DataLoader,
                         dataset_name: str = '') -> Dict[str, float]:
        """
        Run full 10-fold cross-validation evaluation on a dataset.

        Returns:
            Dict with 'accuracy', 'std', 'auc', 'eer'
        """
        print(f'  Evaluating {dataset_name}...')

        # Extract all embeddings
        emb1, emb2, labels = self.extract_embeddings(loader)
        scores = self.cosine_similarity(emb1, emb2)

        N = len(labels)
        fold_size = N // self.n_folds

        # 10-fold cross-validation
        fold_accuracies = []
        kf = KFold(n_splits=self.n_folds, shuffle=True)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(range(N))):
            acc = self.evaluate_fold(scores, labels, train_idx, test_idx)
            fold_accuracies.append(acc)

        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)

        # ROC and AUC
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        # Equal Error Rate (EER)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        print(f'    {dataset_name}: {mean_acc:.2f}% ± {std_acc:.2f}%  '
              f'AUC={roc_auc:.3f}  EER={eer*100:.2f}%')

        return {
            'accuracy': mean_acc,
            'std': std_acc,
            'auc': roc_auc,
            'eer': eer * 100,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'fold_accuracies': fold_accuracies,
        }

    def evaluate_all(self, loaders: Dict[str, DataLoader]
                     ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate on all datasets and compute average.

        Args:
            loaders: Dict mapping dataset_name → DataLoader

        Returns:
            results: Dict mapping dataset_name → metrics
        """
        results = {}
        accuracies = []

        for name, loader in loaders.items():
            results[name] = self.evaluate_dataset(loader, name)
            accuracies.append(results[name]['accuracy'])

        avg_acc = np.mean(accuracies)
        print(f'\n  Average accuracy: {avg_acc:.2f}%')
        results['average'] = {'accuracy': avg_acc}

        return results


def compute_roc_metrics(scores: np.ndarray, labels: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute ROC curve metrics for IJB-B/IJB-C evaluation (Figure 7).

    Returns:
        fpr, tpr, auc_score, eer
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # EER
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    return fpr, tpr, roc_auc, eer


def format_results_table(results: Dict[str, Dict[str, float]],
                         model_name: str = 'CryptoFace') -> str:
    """Format results as a table matching Table 1 from the paper."""
    datasets = ['lfw', 'agedb', 'calfw', 'cplfw', 'cfpfp']
    names_map = {
        'lfw': 'LFW', 'agedb': 'AgeDB', 'calfw': 'CALFW',
        'cplfw': 'CPLFW', 'cfpfp': 'CFP-FP'
    }

    lines = [
        f'\n{"="*70}',
        f'Results for {model_name}',
        f'{"="*70}',
        f'{"Dataset":<12} {"Accuracy":>10} {"±Std":>8} {"AUC":>8}',
        f'{"-"*40}',
    ]

    accs = []
    for ds in datasets:
        if ds in results:
            r = results[ds]
            acc = r.get('accuracy', 0.0)
            std = r.get('std', 0.0)
            a = r.get('auc', 0.0)
            lines.append(f'{names_map[ds]:<12} {acc:>9.2f}% {std:>7.2f}% {a:>8.3f}')
            accs.append(acc)

    if accs:
        avg = np.mean(accs)
        lines.append(f'{"-"*40}')
        lines.append(f'{"Average":<12} {avg:>9.2f}%')
    lines.append('='*70)

    return '\n'.join(lines)
