"""Splitting data, cross-validation and hyper-parameter search."""

from __future__ import annotations

import itertools

import numpy as np

from .base import rng_from


def train_test_split(*arrays, test_size: float = 0.25, stratify=None, random_state=None, shuffle: bool = True):
    """Split arrays into train and test parts. With `stratify`, class proportions are preserved."""
    n = len(arrays[0])
    rng = rng_from(random_state)
    n_test = int(np.ceil(test_size * n)) if isinstance(test_size, float) else int(test_size)
    if stratify is not None:
        stratify = np.asarray(stratify)
        test_idx = []
        for cls in np.unique(stratify):
            idx = np.flatnonzero(stratify == cls)
            rng.shuffle(idx)
            test_idx.extend(idx[: int(round(len(idx) * n_test / n))])
        test_idx = np.array(sorted(test_idx))
        train_idx = np.setdiff1d(np.arange(n), test_idx)
    else:
        idx = rng.permutation(n) if shuffle else np.arange(n)
        test_idx, train_idx = idx[:n_test], idx[n_test:]
    out = []
    for a in arrays:
        a = np.asarray(a) if not hasattr(a, "iloc") else a
        pick = (lambda ix: a.iloc[ix]) if hasattr(a, "iloc") else (lambda ix: a[ix])
        out += [pick(train_idx), pick(test_idx)]
    return out


class KFold:
    """Split indices into k folds; each fold is the validation set once."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state=None):
        self.n_splits, self.shuffle, self.random_state = n_splits, shuffle, random_state

    def split(self, X, y=None):
        n = len(X)
        idx = rng_from(self.random_state).permutation(n) if self.shuffle else np.arange(n)
        for fold in np.array_split(idx, self.n_splits):
            yield np.setdiff1d(idx, fold), fold


class StratifiedKFold(KFold):
    """K-fold that keeps the class ratio of y in every fold (important for imbalanced data)."""

    def split(self, X, y):
        y = np.asarray(y)
        rng = rng_from(self.random_state)
        folds = [[] for _ in range(self.n_splits)]
        for cls in np.unique(y):
            idx = np.flatnonzero(y == cls)
            if self.shuffle:
                rng.shuffle(idx)
            for i, part in enumerate(np.array_split(idx, self.n_splits)):
                folds[i].extend(part)
        all_idx = np.arange(len(y))
        for fold in folds:
            fold = np.array(sorted(fold))
            yield np.setdiff1d(all_idx, fold), fold


def cross_val_score(model, X, y, cv=5, scoring=None) -> np.ndarray:
    """Fit a fresh copy of `model` on each training fold and score it on the held-out fold."""
    X, y = np.asarray(X), np.asarray(y)
    splitter = cv if hasattr(cv, "split") else KFold(cv, random_state=0)
    scores = []
    for train_idx, val_idx in splitter.split(X, y):
        m = model.clone().fit(X[train_idx], y[train_idx])
        scores.append(scoring(y[val_idx], m.predict(X[val_idx])) if scoring else m.score(X[val_idx], y[val_idx]))
    return np.array(scores)


class GridSearchCV:
    """Try every combination in `param_grid` with cross-validation; refit the best on all data."""

    def __init__(self, model, param_grid: dict, cv=5, scoring=None):
        self.model, self.param_grid, self.cv, self.scoring = model, param_grid, cv, scoring

    def fit(self, X, y):
        keys = list(self.param_grid)
        self.results_ = []
        for values in itertools.product(*(self.param_grid[k] for k in keys)):
            params = dict(zip(keys, values))
            scores = cross_val_score(self.model.clone().set_params(**params), X, y, self.cv, self.scoring)
            self.results_.append({"params": params, "mean": scores.mean(), "std": scores.std()})
        best = max(self.results_, key=lambda r: r["mean"])
        self.best_params_, self.best_score_ = best["params"], best["mean"]
        self.best_estimator_ = self.model.clone().set_params(**self.best_params_).fit(X, y)
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)


class RandomizedSearchCV(GridSearchCV):
    """Sample `n_iter` random combinations - usually finds a good region far faster than a full grid."""

    def __init__(self, model, param_distributions: dict, n_iter: int = 10, cv=5, scoring=None, random_state=None):
        super().__init__(model, param_distributions, cv, scoring)
        self.n_iter, self.random_state = n_iter, random_state

    def fit(self, X, y):
        rng = rng_from(self.random_state)
        self.results_ = []
        for _ in range(self.n_iter):
            params = {k: (v.rvs(random_state=rng) if hasattr(v, "rvs") else v[rng.integers(len(v))])
                      for k, v in self.param_grid.items()}
            scores = cross_val_score(self.model.clone().set_params(**params), X, y, self.cv, self.scoring)
            self.results_.append({"params": params, "mean": scores.mean(), "std": scores.std()})
        best = max(self.results_, key=lambda r: r["mean"])
        self.best_params_, self.best_score_ = best["params"], best["mean"]
        self.best_estimator_ = self.model.clone().set_params(**self.best_params_).fit(X, y)
        return self


def learning_curve(model, X, y, train_sizes=(0.1, 0.25, 0.5, 0.75, 1.0), cv=5):
    """Train/validation score as a function of training-set size (diagnoses bias vs variance)."""
    X, y = np.asarray(X), np.asarray(y)
    rows = []
    for frac in train_sizes:
        tr, va = [], []
        for train_idx, val_idx in KFold(cv, random_state=0).split(X):
            sub = train_idx[: max(2, int(frac * len(train_idx)))]
            m = model.clone().fit(X[sub], y[sub])
            tr.append(m.score(X[sub], y[sub]))
            va.append(m.score(X[val_idx], y[val_idx]))
        rows.append((frac, float(np.mean(tr)), float(np.mean(va))))
    return rows
