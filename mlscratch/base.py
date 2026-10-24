"""Shared helpers: input validation and the estimator interface every model follows.

Every model in this library has the same tiny API, borrowed from scikit-learn so the
code is easy to compare:

    model.fit(X, y)          learn parameters from data
    model.predict(X)         use them
    model.predict_proba(X)   (classifiers) class probabilities
    model.transform(X)       (transformers) change the features
"""

from __future__ import annotations

import numpy as np


def as_2d(X) -> np.ndarray:
    """Return X as a float 2-D array of shape (n_samples, n_features)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {X.shape}")
    return X


def as_1d(y) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1:
        y = y.ravel()
    return y


def check_Xy(X, y):
    X, y = as_2d(X), as_1d(y)
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
    return X, y


def add_bias(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones so the intercept becomes an ordinary weight."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable 1 / (1 + e^-z)."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softmax(Z: np.ndarray) -> np.ndarray:
    """Row-wise softmax; subtracting the row max avoids overflow without changing the result."""
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def one_hot(y: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    n_classes = n_classes or int(y.max()) + 1
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


def rng_from(seed) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


class BaseEstimator:
    """Gives every model a readable repr and get/set_params (used by grid search)."""

    def get_params(self) -> dict:
        return {k: v for k, v in vars(self).items() if not k.endswith("_") and not k.startswith("_")}

    def set_params(self, **params):
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"{type(self).__name__} has no parameter {key!r}")
            setattr(self, key, value)
        return self

    def clone(self):
        return type(self)(**self.get_params())

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"


class ClassifierMixin:
    """Classifiers map arbitrary labels to 0..K-1 internally and back on predict."""

    def _encode_labels(self, y):
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        return y_idx

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


class RegressorMixin:
    def score(self, X, y) -> float:
        """Coefficient of determination R^2."""
        y = np.asarray(y, dtype=float)
        residual = np.sum((y - self.predict(X)) ** 2)
        total = np.sum((y - y.mean()) ** 2)
        return float(1 - residual / total)
