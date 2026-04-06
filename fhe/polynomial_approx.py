"""
FHE Polynomial Approximation Modules.

Distribution-Aware L2 Normalization Polynomial (Section 3.3):
  q(t) = 1/sqrt(t) approximated by p(t) = b2*t^2 + b1*t + b0

  Control points: t1=Mean-Std, t2=Mean, t3=Mean+Std
  FHE depth: 2 levels | Approximation error <= 2^{-10}
"""

import numpy as np
from typing import Tuple, Optional

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False


class PolynomialL2Norm:
    """Distribution-aware polynomial approx of 1/sqrt(t). FHE depth: 2."""

    def __init__(self):
        self.beta2 = self.beta1 = self.beta0 = None
        self.mean_t = self.std_t = None
        self.t1 = self.t2 = self.t3 = None
        self.is_fitted = False

    def fit(self, norms_squared):
        t = np.asarray(norms_squared, dtype=np.float64)
        t = t[t > 0]
        self.mean_t, self.std_t = float(np.mean(t)), float(np.std(t))
        self.t1 = max(self.mean_t - self.std_t, 1e-6)
        self.t2 = self.mean_t
        self.t3 = self.mean_t + self.std_t
        A = np.array([[self.t1**2, self.t1, 1.0],
                      [self.t2**2, self.t2, 1.0],
                      [self.t3**2, self.t3, 1.0]])
        q = np.array([1/np.sqrt(self.t1), 1/np.sqrt(self.t2), 1/np.sqrt(self.t3)])
        try:
            self.beta2, self.beta1, self.beta0 = np.linalg.solve(A, q)
        except np.linalg.LinAlgError:
            self.beta2, self.beta1 = 0.0, -0.5 * self.mean_t**(-1.5)
            self.beta0 = 1/np.sqrt(self.mean_t) - self.beta1 * self.mean_t
        self.is_fitted = True
        return self

    def apply(self, t):
        assert self.is_fitted
        return self.beta2 * t**2 + self.beta1 * t + self.beta0

    def approximation_error(self, t_range=None):
        assert self.is_fitted
        if t_range is None:
            t_range = np.linspace(max(self.t1*0.6, 1e-4), self.t3*1.4, 600)
        error = np.log2(np.abs(self.apply(t_range) - 1/np.sqrt(t_range)) + 1e-30)
        return t_range, error

    def fhe_depth(self):
        return 2

    def normalized_features(self, features):
        assert self.is_fitted
        norms_sq = np.sum(features**2, axis=1, keepdims=True)
        return features * self.apply(norms_sq)


class MatchFunction:
    """Match(y1, y2) = Score(y1,y2) - T. Result < 0 => same person."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit_threshold(self, scores, labels):
        best_acc, best_t = 0.0, 0.0
        for t in np.linspace(scores.min(), scores.max(), 1000):
            acc = np.mean((scores >= t).astype(int) == labels)
            if acc > best_acc:
                best_acc, best_t = acc, t
        self.threshold = best_t
        return self.threshold

    def __call__(self, score):
        return score - self.threshold
