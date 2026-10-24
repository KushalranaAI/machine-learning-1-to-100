"""k-nearest neighbours: no training, just memory + a distance function."""

from __future__ import annotations

import numpy as np

from .base import BaseEstimator, ClassifierMixin, RegressorMixin, as_2d, check_Xy


def pairwise_distances(A: np.ndarray, B: np.ndarray, metric: str = "euclidean", p: float = 2) -> np.ndarray:
    """D[i, j] = distance(A[i], B[j]).

    Euclidean uses ||a - b||² = ||a||² + ||b||² - 2a·b, which turns the whole matrix into one matrix product.
    """
    if metric == "euclidean":
        sq = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
        return np.sqrt(np.maximum(sq, 0))
    if metric == "manhattan":
        return np.abs(A[:, None, :] - B[None, :, :]).sum(-1)
    if metric == "minkowski":
        return (np.abs(A[:, None, :] - B[None, :, :]) ** p).sum(-1) ** (1 / p)
    if metric == "cosine":
        An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
        Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
        return 1 - An @ Bn.T
    raise ValueError(metric)


class _KNNBase(BaseEstimator):
    def __init__(self, n_neighbors: int = 5, weights: str = "uniform", metric: str = "euclidean"):
        self.n_neighbors, self.weights, self.metric = n_neighbors, weights, metric

    def _neighbors(self, X):
        # Queries go in batches so the distance matrix stays at ~20M entries (160 MB) instead of n_test × n_train.
        X = as_2d(X)
        k = min(self.n_neighbors, len(self.X_))
        batch_size = max(1, min(1024, 20_000_000 // len(self.X_)))
        idx_parts, d_parts = [], []
        for start in range(0, len(X), batch_size):
            D = pairwise_distances(X[start : start + batch_size], self.X_, self.metric)
            idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
            d = np.take_along_axis(D, idx, axis=1)
            order = np.argsort(d, axis=1)
            idx_parts.append(np.take_along_axis(idx, order, 1))
            d_parts.append(np.take_along_axis(d, order, 1))
        idx, d = np.vstack(idx_parts), np.vstack(d_parts)
        w = np.ones_like(d) if self.weights == "uniform" else 1.0 / np.maximum(d, 1e-12)
        return idx, w


class KNeighborsClassifier(_KNNBase, ClassifierMixin):
    """Predict the (optionally distance-weighted) majority class among the k closest training points."""

    def fit(self, X, y):
        self.X_, y = check_Xy(X, y)
        self.y_ = self._encode_labels(y)
        return self

    def predict_proba(self, X):
        idx, w = self._neighbors(X)
        votes = np.zeros((len(idx), len(self.classes_)))
        for k in range(len(self.classes_)):
            votes[:, k] = (w * (self.y_[idx] == k)).sum(axis=1)
        return votes / votes.sum(axis=1, keepdims=True)


class KNeighborsRegressor(_KNNBase, RegressorMixin):
    """Predict the (optionally distance-weighted) mean target of the k closest training points."""

    def fit(self, X, y):
        self.X_, y = check_Xy(X, y)
        self.y_ = y.astype(float)
        return self

    def predict(self, X):
        idx, w = self._neighbors(X)
        return (w * self.y_[idx]).sum(axis=1) / w.sum(axis=1)
