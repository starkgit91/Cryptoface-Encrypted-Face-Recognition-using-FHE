"""Evaluation metrics utilities."""
import numpy as np
from sklearn.metrics import roc_curve, auc as sklearn_auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def compute_accuracy(scores: np.ndarray, labels: np.ndarray,
                     threshold: float) -> float:
    preds = (scores >= threshold).astype(int)
    return float(np.mean(preds == labels) * 100)


def find_best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    best_acc, best_t = 0.0, 0.0
    for t in thresholds:
        acc = np.mean((scores >= t).astype(int) == labels)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t


def compute_roc(scores: np.ndarray, labels: np.ndarray):
    fpr, tpr, _ = roc_curve(labels, scores)
    auc_score = sklearn_auc(fpr, tpr)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return fpr, tpr, auc_score, eer


def count_params(model) -> float:
    """Return model parameter count in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6
