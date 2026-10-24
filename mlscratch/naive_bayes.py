"""Naive Bayes: Bayes' rule + the 'naive' assumption that features are independent given the class.

    P(y | x) ∝ P(y) · Π_j P(x_j | y)        →  work in log space:  log P(y) + Σ_j log P(x_j | y)
"""

from __future__ import annotations

import numpy as np

from .base import BaseEstimator, ClassifierMixin, as_2d, check_Xy


class _NB(BaseEstimator, ClassifierMixin):
    def predict_proba(self, X):
        J = self._joint_log_likelihood(as_2d(X))
        J -= J.max(axis=1, keepdims=True)          # log-sum-exp trick for stable normalisation
        P = np.exp(J)
        return P / P.sum(axis=1, keepdims=True)

    def predict_log_joint(self, X):
        return self._joint_log_likelihood(as_2d(X))


class GaussianNB(_NB):
    """Continuous features: P(x_j | y=k) = N(x_j; μ_kj, σ²_kj)."""

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X, y = check_Xy(X, y)
        y = self._encode_labels(y)
        K = len(self.classes_)
        self.theta_ = np.array([X[y == k].mean(axis=0) for k in range(K)])
        eps = self.var_smoothing * X.var(axis=0).max()
        self.var_ = np.array([X[y == k].var(axis=0) for k in range(K)]) + eps
        self.class_log_prior_ = np.log(np.bincount(y, minlength=K) / len(y))
        return self

    def _joint_log_likelihood(self, X):
        # log N(x; μ, σ²) = -½ log(2πσ²) - (x - μ)² / (2σ²)
        ll = -0.5 * np.log(2 * np.pi * self.var_).sum(axis=1)[None, :]
        ll = ll - 0.5 * (((X[:, None, :] - self.theta_[None]) ** 2) / self.var_[None]).sum(axis=2)
        return ll + self.class_log_prior_


class MultinomialNB(_NB):
    """Count features (word counts): P(x | y=k) ∝ Π_j θ_kj^{x_j}, θ_kj = (N_kj + α) / (N_k + α·d)  (Laplace smoothing)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X, y = check_Xy(X, y)
        y = self._encode_labels(y)
        K = len(self.classes_)
        counts = np.array([X[y == k].sum(axis=0) for k in range(K)]) + self.alpha
        self.feature_log_prob_ = np.log(counts / counts.sum(axis=1, keepdims=True))
        self.class_log_prior_ = np.log(np.bincount(y, minlength=K) / len(y))
        return self

    def _joint_log_likelihood(self, X):
        return X @ self.feature_log_prob_.T + self.class_log_prior_


class BernoulliNB(_NB):
    """Binary features (word present / absent): absent words count as evidence too."""

    def __init__(self, alpha: float = 1.0, binarize: float = 0.0):
        self.alpha, self.binarize = alpha, binarize

    def fit(self, X, y):
        X, y = check_Xy(X, y)
        X = (X > self.binarize).astype(float)
        y = self._encode_labels(y)
        K = len(self.classes_)
        n_k = np.bincount(y, minlength=K)
        self.feature_prob_ = (np.array([X[y == k].sum(0) for k in range(K)]) + self.alpha) / (n_k[:, None] + 2 * self.alpha)
        self.class_log_prior_ = np.log(n_k / len(y))
        return self

    def _joint_log_likelihood(self, X):
        X = (X > self.binarize).astype(float)
        p = self.feature_prob_
        return X @ np.log(p).T + (1 - X) @ np.log(1 - p).T + self.class_log_prior_
