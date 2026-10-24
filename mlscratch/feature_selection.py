"""Feature selection: filter (statistics), wrapper (search with a model) and embedded (L1, trees) methods."""

from __future__ import annotations

import numpy as np

from .base import BaseEstimator, as_2d, check_Xy
from .model_selection import cross_val_score


class VarianceThreshold(BaseEstimator):
    """Drop features whose variance ≤ threshold (constant or near-constant columns carry no information)."""

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.variances_ = as_2d(X).var(0)
        self.support_ = self.variances_ > self.threshold
        return self

    def transform(self, X):
        return as_2d(X)[:, self.support_]

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def correlation_filter(X, threshold=0.9) -> np.ndarray:
    """Boolean mask that drops one feature of every pair with |Pearson r| > threshold."""
    C = np.abs(np.corrcoef(as_2d(X), rowvar=False))
    keep = np.ones(C.shape[0], bool)
    for i in range(C.shape[0]):
        if keep[i]:
            keep[(np.arange(C.shape[0]) > i) & (C[i] > threshold)] = False
    return keep


def chi2(X, y):
    """χ² statistic between non-negative features (counts/frequencies) and the class:
    Σ_k (O_k - E_k)² / E_k, with O = per-class feature totals and E = what independence predicts."""
    X, y = check_Xy(X, y)
    classes = np.unique(y)
    observed = np.array([X[y == c].sum(0) for c in classes])
    class_prob = np.array([np.mean(y == c) for c in classes])[:, None]
    expected = class_prob * X.sum(0)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nansum((observed - expected) ** 2 / expected, axis=0)


def f_classif(X, y):
    """One-way ANOVA F = (between-group variance) / (within-group variance) for each feature."""
    X, y = check_Xy(X, y)
    classes = np.unique(y)
    grand = X.mean(0)
    ssb = sum(np.sum(y == c) * (X[y == c].mean(0) - grand) ** 2 for c in classes)
    ssw = sum(((X[y == c] - X[y == c].mean(0)) ** 2).sum(0) for c in classes)
    dfb, dfw = len(classes) - 1, len(y) - len(classes)
    return (ssb / dfb) / np.maximum(ssw / dfw, 1e-12)


def mutual_info_classif(X, y, n_bins=10):
    """I(X;Y) = Σ p(x,y) log p(x,y)/(p(x)p(y)) with each continuous feature discretised into quantile bins."""
    X, y = check_Xy(X, y)
    _, y_idx = np.unique(y, return_inverse=True)
    out = []
    for col in X.T:
        edges = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)))
        b = np.clip(np.searchsorted(edges[1:-1], col, side="right"), 0, None)
        joint = np.zeros((b.max() + 1, y_idx.max() + 1))
        np.add.at(joint, (b, y_idx), 1)
        joint /= joint.sum()
        px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
        nz = joint > 0
        out.append(float(np.sum(joint[nz] * np.log(joint[nz] / (px @ py)[nz]))))
    return np.array(out)


class SelectKBest(BaseEstimator):
    def __init__(self, score_func=f_classif, k=10):
        self.score_func, self.k = score_func, k

    def fit(self, X, y):
        self.scores_ = self.score_func(X, y)
        self.support_ = np.zeros(len(self.scores_), bool)
        self.support_[np.argsort(-np.nan_to_num(self.scores_))[: self.k]] = True
        return self

    def transform(self, X):
        return as_2d(X)[:, self.support_]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class RFE(BaseEstimator):
    """Recursive Feature Elimination: fit, drop the least important feature(s), repeat."""

    def __init__(self, estimator, n_features_to_select=5, step=1):
        self.estimator, self.n_features_to_select, self.step = estimator, n_features_to_select, step

    @staticmethod
    def _importance(model):
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        return np.abs(np.atleast_2d(model.coef_)).sum(0)

    def fit(self, X, y):
        X, y = check_Xy(X, y)
        remaining = list(range(X.shape[1]))
        self.ranking_ = np.ones(X.shape[1], int)
        rank = X.shape[1]
        while len(remaining) > self.n_features_to_select:
            model = self.estimator.clone().fit(X[:, remaining], y)
            order = np.argsort(self._importance(model))
            for i in order[: min(self.step, len(remaining) - self.n_features_to_select)]:
                self.ranking_[remaining[i]] = rank
                rank -= 1
            drop = {remaining[i] for i in order[: min(self.step, len(remaining) - self.n_features_to_select)]}
            remaining = [f for f in remaining if f not in drop]
        self.support_ = np.zeros(X.shape[1], bool)
        self.support_[remaining] = True
        return self

    def transform(self, X):
        return as_2d(X)[:, self.support_]


class SequentialFeatureSelector(BaseEstimator):
    """Greedy forward selection: add the feature that most improves the cross-validated score."""

    def __init__(self, estimator, n_features_to_select=5, cv=5):
        self.estimator, self.n_features_to_select, self.cv = estimator, n_features_to_select, cv

    def fit(self, X, y):
        X, y = check_Xy(X, y)
        chosen, self.history_ = [], []
        while len(chosen) < self.n_features_to_select:
            best = max((f for f in range(X.shape[1]) if f not in chosen),
                       key=lambda f: cross_val_score(self.estimator, X[:, chosen + [f]], y, self.cv).mean())
            chosen.append(best)
            self.history_.append((best, cross_val_score(self.estimator, X[:, chosen], y, self.cv).mean()))
        self.support_ = np.zeros(X.shape[1], bool)
        self.support_[chosen] = True
        return self

    def transform(self, X):
        return as_2d(X)[:, self.support_]


def permutation_importance(model, X, y, n_repeats=5, random_state=0, scoring=None):
    """Importance = drop in score when one feature's column is randomly shuffled (model-agnostic)."""
    X, y = check_Xy(X, y)
    rng = np.random.default_rng(random_state)
    score = (lambda: scoring(y, model.predict(Xp))) if scoring else (lambda: model.score(Xp, y))
    Xp = X
    base = score()
    out = np.zeros((X.shape[1], n_repeats))
    for j in range(X.shape[1]):
        for r in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            out[j, r] = base - score()
    return out.mean(1), out.std(1)
