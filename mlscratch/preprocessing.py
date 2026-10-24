"""Scaling, encoding, imputation and feature-construction transformers (fit on train, apply anywhere)."""

from __future__ import annotations

import itertools

import numpy as np

from .base import BaseEstimator, as_2d


class Transformer(BaseEstimator):
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# --------------------------------------------------------------------------------- scaling


class StandardScaler(Transformer):
    """z = (x - μ) / σ  per feature. Needed by every distance- or gradient-based model."""

    def fit(self, X, y=None):
        X = as_2d(X)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0  # constant columns stay constant
        return self

    def transform(self, X):
        return (as_2d(X) - self.mean_) / self.scale_

    def inverse_transform(self, Z):
        return as_2d(Z) * self.scale_ + self.mean_


class MinMaxScaler(Transformer):
    """x' = (x - min) / (max - min) → [0, 1]. Sensitive to outliers."""

    def fit(self, X, y=None):
        X = as_2d(X)
        self.min_, self.max_ = X.min(axis=0), X.max(axis=0)
        self.range_ = np.where(self.max_ > self.min_, self.max_ - self.min_, 1.0)
        return self

    def transform(self, X):
        return (as_2d(X) - self.min_) / self.range_


class RobustScaler(Transformer):
    """x' = (x - median) / IQR. Outliers barely move the median or the quartiles."""

    def fit(self, X, y=None):
        X = as_2d(X)
        self.center_ = np.median(X, axis=0)
        q1, q3 = np.percentile(X, [25, 75], axis=0)
        self.scale_ = np.where(q3 > q1, q3 - q1, 1.0)
        return self

    def transform(self, X):
        return (as_2d(X) - self.center_) / self.scale_


class PowerTransformer(Transformer):
    """Yeo-Johnson transform; λ per feature chosen by maximising the Gaussian log-likelihood (grid search)."""

    def __init__(self, standardize: bool = True):
        self.standardize = standardize

    @staticmethod
    def _yeo_johnson(x, lam):
        out = np.empty_like(x)
        pos = x >= 0
        out[pos] = np.log1p(x[pos]) if abs(lam) < 1e-8 else ((x[pos] + 1) ** lam - 1) / lam
        out[~pos] = (-np.log1p(-x[~pos]) if abs(lam - 2) < 1e-8
                     else -(((-x[~pos] + 1) ** (2 - lam)) - 1) / (2 - lam))
        return out

    def _loglik(self, x, lam):
        y = self._yeo_johnson(x, lam)
        return -len(x) / 2 * np.log(y.var() + 1e-12) + (lam - 1) * np.sum(np.sign(x) * np.log1p(np.abs(x)))

    def fit(self, X, y=None):
        X = as_2d(X)
        grid = np.linspace(-2, 4, 241)
        self.lambdas_ = np.array([grid[np.argmax([self._loglik(col, g) for g in grid])] for col in X.T])
        if self.standardize:
            self._scaler = StandardScaler().fit(self._apply(X))
        return self

    def _apply(self, X):
        return np.column_stack([self._yeo_johnson(col, lam) for col, lam in zip(X.T, self.lambdas_)])

    def transform(self, X):
        Z = self._apply(as_2d(X))
        return self._scaler.transform(Z) if self.standardize else Z


# -------------------------------------------------------------------------------- encoding


class LabelEncoder(Transformer):
    def fit(self, y, _=None):
        self.classes_ = np.unique(np.asarray(y))
        return self

    def transform(self, y):
        index = {c: i for i, c in enumerate(self.classes_)}
        return np.array([index[v] for v in np.asarray(y)])

    def inverse_transform(self, idx):
        return self.classes_[np.asarray(idx)]


class OrdinalEncoder(Transformer):
    """Map each category to an integer. Pass `categories` to impose a meaningful order (low < mid < high)."""

    def __init__(self, categories=None, unknown_value: float = -1):
        self.categories, self.unknown_value = categories, unknown_value

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=object).reshape(len(X), -1)
        self.categories_ = self.categories or [sorted(set(col), key=str) for col in X.T]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=object).reshape(len(X), -1)
        maps = [{c: i for i, c in enumerate(cats)} for cats in self.categories_]
        return np.array([[m.get(v, self.unknown_value) for v, m in zip(row, maps)] for row in X], dtype=float)


class OneHotEncoder(Transformer):
    """One binary column per category. `drop_first` avoids the dummy-variable trap for linear models."""

    def __init__(self, drop_first: bool = False, handle_unknown: str = "ignore"):
        self.drop_first, self.handle_unknown = drop_first, handle_unknown

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=object).reshape(len(X), -1)
        self.categories_ = [sorted(set(col), key=str) for col in X.T]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=object).reshape(len(X), -1)
        blocks = []
        for j, cats in enumerate(self.categories_):
            cols = cats[1:] if self.drop_first else cats
            block = np.column_stack([(X[:, j] == c).astype(float) for c in cols]) if cols else np.zeros((len(X), 0))
            if self.handle_unknown == "error" and not np.isin(X[:, j], cats).all():
                raise ValueError(f"unknown category in column {j}")
            blocks.append(block)
        return np.hstack(blocks)

    def get_feature_names(self, names=None) -> list[str]:
        names = names or [f"x{j}" for j in range(len(self.categories_))]
        return [f"{n}={c}" for n, cats in zip(names, self.categories_) for c in (cats[1:] if self.drop_first else cats)]


class TargetEncoder(Transformer):
    """Replace a category by a smoothed mean of the target:  (n·mean_c + m·global_mean) / (n + m).

    Fit it inside cross-validation folds only - otherwise the target leaks into the features.
    """

    def __init__(self, smoothing: float = 10.0):
        self.smoothing = smoothing

    def fit(self, X, y):
        X = np.asarray(X, dtype=object).reshape(len(X), -1)
        y = np.asarray(y, dtype=float)
        self.global_mean_ = y.mean()
        self.maps_ = []
        for col in X.T:
            m = {}
            for c in set(col):
                mask = col == c
                n = mask.sum()
                m[c] = (n * y[mask].mean() + self.smoothing * self.global_mean_) / (n + self.smoothing)
            self.maps_.append(m)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=object).reshape(len(X), -1)
        return np.column_stack([[m.get(v, self.global_mean_) for v in col] for col, m in zip(X.T, self.maps_)])


# ------------------------------------------------------------------------------ imputation


class SimpleImputer(Transformer):
    """Fill NaNs with the column mean, median, most frequent value or a constant."""

    def __init__(self, strategy: str = "mean", fill_value: float = 0.0):
        self.strategy, self.fill_value = strategy, fill_value

    def fit(self, X, y=None):
        X = as_2d(X)
        stats = []
        for col in X.T:
            vals = col[~np.isnan(col)]
            if self.strategy == "mean":
                stats.append(vals.mean() if len(vals) else self.fill_value)
            elif self.strategy == "median":
                stats.append(np.median(vals) if len(vals) else self.fill_value)
            elif self.strategy == "most_frequent":
                v, c = np.unique(vals, return_counts=True)
                stats.append(v[np.argmax(c)] if len(v) else self.fill_value)
            else:
                stats.append(self.fill_value)
        self.statistics_ = np.array(stats)
        return self

    def transform(self, X):
        X = as_2d(X).copy()
        rows, cols = np.where(np.isnan(X))
        X[rows, cols] = self.statistics_[cols]
        return X


class KNNImputer(Transformer):
    """Fill a missing value with the mean of that feature over the k nearest rows (distance on observed features)."""

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        self.X_ = as_2d(X)
        return self

    def transform(self, X):
        X = as_2d(X).copy()
        ref = self.X_
        for i in np.where(np.isnan(X).any(axis=1))[0]:
            obs = ~np.isnan(X[i])
            both = ~np.isnan(ref[:, obs]).any(axis=1)
            cand = ref[both]
            d = np.sqrt(((cand[:, obs] - X[i, obs]) ** 2).sum(axis=1))
            for j in np.where(~obs)[0]:
                usable = ~np.isnan(cand[:, j])
                nearest = np.argsort(d[usable])[: self.n_neighbors]
                X[i, j] = cand[usable][nearest, j].mean() if usable.any() else np.nanmean(ref[:, j])
        return X


# --------------------------------------------------------------------- feature construction


class PolynomialFeatures(Transformer):
    """All products of features up to `degree` (e.g. x1, x2, x1², x1·x2, x2² for degree 2)."""

    def __init__(self, degree: int = 2, include_bias: bool = False, interaction_only: bool = False):
        self.degree, self.include_bias, self.interaction_only = degree, include_bias, interaction_only

    def fit(self, X, y=None):
        n = as_2d(X).shape[1]
        combo = itertools.combinations if self.interaction_only else itertools.combinations_with_replacement
        self.powers_ = [c for d in range(1, self.degree + 1) for c in combo(range(n), d)]
        return self

    def transform(self, X):
        X = as_2d(X)
        cols = [np.prod(X[:, list(c)], axis=1) for c in self.powers_]
        if self.include_bias:
            cols.insert(0, np.ones(len(X)))
        return np.column_stack(cols)


class KBinsDiscretizer(Transformer):
    """Bin continuous features: 'uniform' = equal width, 'quantile' = equal frequency."""

    def __init__(self, n_bins: int = 5, strategy: str = "quantile"):
        self.n_bins, self.strategy = n_bins, strategy

    def fit(self, X, y=None):
        X = as_2d(X)
        if self.strategy == "uniform":
            self.edges_ = [np.linspace(c.min(), c.max(), self.n_bins + 1) for c in X.T]
        else:
            self.edges_ = [np.unique(np.quantile(c, np.linspace(0, 1, self.n_bins + 1))) for c in X.T]
        return self

    def transform(self, X):
        X = as_2d(X)
        return np.column_stack([np.clip(np.searchsorted(e[1:-1], c, side="right"), 0, len(e) - 2)
                                for c, e in zip(X.T, self.edges_)]).astype(float)


class Pipeline(BaseEstimator):
    """Chain transformers and a final estimator so preprocessing is learned only from training folds."""

    def __init__(self, steps: list):
        self.steps = steps

    def clone(self):
        return Pipeline([(name, step.clone()) for name, step in self.steps])

    def set_params(self, **params):
        for key, value in params.items():
            name, attr = key.split("__", 1)
            dict(self.steps)[name].set_params(**{attr: value})
        return self

    def fit(self, X, y=None):
        for _, step in self.steps[:-1]:
            X = step.fit_transform(X, y)
        self.steps[-1][1].fit(X, y)
        return self

    def _transform(self, X):
        for _, step in self.steps[:-1]:
            X = step.transform(X)
        return X

    def predict(self, X):
        return self.steps[-1][1].predict(self._transform(X))

    def predict_proba(self, X):
        return self.steps[-1][1].predict_proba(self._transform(X))

    def score(self, X, y):
        return self.steps[-1][1].score(self._transform(X), y)
