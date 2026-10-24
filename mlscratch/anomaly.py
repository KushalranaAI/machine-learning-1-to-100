"""Anomaly detection: statistical scores and Isolation Forest."""

from __future__ import annotations

import numpy as np

from .base import BaseEstimator, as_2d, rng_from


def zscore_outliers(x, threshold=3.0) -> np.ndarray:
    x = np.asarray(x, float)
    return np.abs((x - x.mean()) / x.std()) > threshold


def iqr_outliers(x, k=1.5) -> np.ndarray:
    """Tukey's fences: outside [Q1 - k·IQR, Q3 + k·IQR]."""
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return (x < q1 - k * iqr) | (x > q3 + k * iqr)


def mahalanobis(X) -> np.ndarray:
    """d(x) = √((x-μ)ᵀ Σ⁻¹ (x-μ)): distance that accounts for feature scales and correlations."""
    X = as_2d(X)
    diff = X - X.mean(0)
    inv = np.linalg.pinv(np.cov(X, rowvar=False))
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff))


def _c(n):
    """Average path length of an unsuccessful BST search: c(n) = 2H(n-1) - 2(n-1)/n."""
    n = np.asarray(n, float)
    safe = np.maximum(n, 3.0)  # avoid log(0) in the branch np.where does not select
    return np.where(n > 2, 2 * (np.log(safe - 1) + np.euler_gamma) - 2 * (safe - 1) / safe, np.where(n == 2, 1.0, 0.0))


class IsolationForest(BaseEstimator):
    """Anomalies are few and different, so random splits isolate them in fewer steps.

    Build trees with random feature + random threshold on subsamples of size ψ.
    score(x) = 2^(-E[h(x)] / c(ψ)) - close to 1 → anomaly, ≲ 0.5 → normal.
    """

    def __init__(self, n_estimators=100, max_samples=256, contamination=0.1, random_state=None):
        self.n_estimators, self.max_samples, self.contamination, self.random_state = n_estimators, max_samples, contamination, random_state

    def _build(self, X, depth, limit, rng):
        if depth >= limit or len(X) <= 1 or np.all(X.max(0) == X.min(0)):
            return ("leaf", len(X))
        f = rng.integers(X.shape[1])
        lo, hi = X[:, f].min(), X[:, f].max()
        if lo == hi:
            return ("leaf", len(X))
        t = rng.uniform(lo, hi)
        m = X[:, f] < t
        return ("node", f, t, self._build(X[m], depth + 1, limit, rng), self._build(X[~m], depth + 1, limit, rng))

    def _path(self, x, node, depth=0):
        if node[0] == "leaf":
            return depth + float(_c(node[1]))
        _, f, t, left, right = node
        return self._path(x, left if x[f] < t else right, depth + 1)

    def fit(self, X, y=None):
        X = as_2d(X)
        rng = rng_from(self.random_state)
        self.psi_ = min(self.max_samples, len(X))
        limit = int(np.ceil(np.log2(self.psi_)))
        self.trees_ = [self._build(X[rng.choice(len(X), self.psi_, replace=False)], 0, limit, rng)
                       for _ in range(self.n_estimators)]
        self.threshold_ = np.quantile(self.score_samples(X), 1 - self.contamination)
        return self

    def score_samples(self, X):
        X = as_2d(X)
        h = np.array([[self._path(x, t) for t in self.trees_] for x in X]).mean(1)
        return 2 ** (-h / float(_c(self.psi_)))

    def predict(self, X):
        """+1 normal, -1 anomaly (same convention as scikit-learn)."""
        return np.where(self.score_samples(X) > self.threshold_, -1, 1)
