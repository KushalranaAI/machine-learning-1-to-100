"""CART decision trees (classification and regression), grown greedily.

At every node we try each feature f and threshold t and keep the split that most reduces impurity:

    gain = I(parent) - (n_L/n) I(left) - (n_R/n) I(right)

Impurities:  Gini = 1 - Σ p_k²     Entropy = -Σ p_k log₂ p_k     MSE = mean (y - ȳ)²

Finding the best threshold is O(n log n) per feature: sort once, then sweep left→right while
updating running class counts (or running sums of y and y²) - no re-scanning per threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BaseEstimator, ClassifierMixin, RegressorMixin, as_2d, check_Xy, rng_from


@dataclass
class Node:
    value: np.ndarray            # class distribution (classifier) or [mean] (regressor)
    n_samples: float
    impurity: float
    feature: int = -1
    threshold: float = 0.0
    left: "Node | None" = None
    right: "Node | None" = None
    leaf_id: int = -1

    @property
    def is_leaf(self) -> bool:
        return self.left is None


def gini_from_counts(counts: np.ndarray) -> np.ndarray:
    total = counts.sum(axis=-1, keepdims=True)
    p = counts / np.maximum(total, 1e-12)
    return 1.0 - (p**2).sum(axis=-1)


def entropy_from_counts(counts: np.ndarray) -> np.ndarray:
    total = counts.sum(axis=-1, keepdims=True)
    p = counts / np.maximum(total, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=-1)


class _BaseTree(BaseEstimator):
    def __init__(self, criterion=None, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0, max_features=None, random_state=None):
        self.criterion, self.max_depth = criterion, max_depth
        self.min_samples_split, self.min_samples_leaf = min_samples_split, min_samples_leaf
        self.min_impurity_decrease, self.max_features, self.random_state = min_impurity_decrease, max_features, random_state

    # -- subclass hooks: node statistics and the best split along one sorted feature
    def _n_features_to_try(self, d):
        mf = self.max_features
        if mf is None:
            return d
        if mf == "sqrt":
            return max(1, int(np.sqrt(d)))
        if mf == "log2":
            return max(1, int(np.log2(d)))
        return max(1, int(mf * d)) if isinstance(mf, float) else min(d, int(mf))

    def fit(self, X, y, sample_weight=None):
        X, y = check_Xy(X, y)
        y = self._prepare_y(y)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        self.n_features_ = X.shape[1]
        self._rng = rng_from(self.random_state)
        self._importances = np.zeros(X.shape[1])
        self._n_leaves = 0
        self.root_ = self._grow(X, y, w, depth=0)
        total = self._importances.sum()
        self.feature_importances_ = self._importances / total if total > 0 else self._importances
        self.n_leaves_ = self._n_leaves
        return self

    def _grow(self, X, y, w, depth):
        value, impurity = self._node_stats(y, w)
        node = Node(value=value, n_samples=w.sum(), impurity=impurity)
        if (len(y) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth)
                or impurity <= 1e-12):
            return self._leaf(node)
        best = None  # (gain, feature, threshold)
        features = self._rng.choice(X.shape[1], self._n_features_to_try(X.shape[1]), replace=False)
        for f in features:
            order = np.argsort(X[:, f], kind="mergesort")
            res = self._best_split(X[order, f], y[order], w[order])
            if res and (best is None or res[0] > best[0]):
                best = (res[0], f, res[1])
        if best is None or best[0] * node.n_samples < self.min_impurity_decrease * node.n_samples or best[0] <= 1e-12:
            return self._leaf(node)
        gain, f, t = best
        mask = X[:, f] <= t
        self._importances[f] += gain * node.n_samples
        node.feature, node.threshold = int(f), float(t)
        node.left = self._grow(X[mask], y[mask], w[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], w[~mask], depth + 1)
        return node

    def _leaf(self, node):
        node.leaf_id = self._n_leaves
        self._n_leaves += 1
        return node

    def _valid_positions(self, x_sorted):
        """Split positions i (left = [:i]) where the value changes and both sides respect min_samples_leaf."""
        n = len(x_sorted)
        pos = np.flatnonzero(np.diff(x_sorted) > 1e-12) + 1
        m = self.min_samples_leaf
        return pos[(pos >= m) & (pos <= n - m)]

    def _route(self, x):
        node = self.root_
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def apply(self, X):
        """Leaf index for every sample (used by gradient boosting to update leaf values)."""
        return np.array([self._route(x).leaf_id for x in as_2d(X)])

    def leaves(self):
        out, stack = [], [self.root_]
        while stack:
            n = stack.pop()
            if n.is_leaf:
                out.append(n)
            else:
                stack += [n.right, n.left]
        return sorted(out, key=lambda n: n.leaf_id)

    def get_depth(self):
        def depth(n):
            return 0 if n.is_leaf else 1 + max(depth(n.left), depth(n.right))
        return depth(self.root_)

    def export_text(self, feature_names=None, decimals=3) -> str:
        names = feature_names or [f"x{i}" for i in range(self.n_features_)]
        lines = []

        def walk(n, indent):
            pad = "|   " * indent
            if n.is_leaf:
                lines.append(f"{pad}|--- value: {np.round(n.value, decimals).tolist()}")
                return
            lines.append(f"{pad}|--- {names[n.feature]} <= {n.threshold:.{decimals}f}")
            walk(n.left, indent + 1)
            lines.append(f"{pad}|--- {names[n.feature]} >  {n.threshold:.{decimals}f}")
            walk(n.right, indent + 1)

        walk(self.root_, 0)
        return "\n".join(lines)


class DecisionTreeClassifier(_BaseTree, ClassifierMixin):
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0, max_features=None, random_state=None):
        super().__init__(criterion, max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease,
                         max_features, random_state)

    def _prepare_y(self, y):
        y = self._encode_labels(y)
        self.n_classes_ = len(self.classes_)
        return y

    def _impurity(self, counts):
        return gini_from_counts(counts) if self.criterion == "gini" else entropy_from_counts(counts)

    def _node_stats(self, y, w):
        counts = np.bincount(y, weights=w, minlength=self.n_classes_)
        return counts / counts.sum(), float(self._impurity(counts))

    def _best_split(self, x, y, w):
        pos = self._valid_positions(x)
        if len(pos) == 0:
            return None
        onehot = np.zeros((len(y), self.n_classes_))
        onehot[np.arange(len(y)), y] = w
        left = np.cumsum(onehot, axis=0)[pos - 1]          # class weights left of each candidate split
        total = onehot.sum(axis=0)
        right = total - left
        nl, nr = left.sum(1), right.sum(1)
        n = total.sum()
        child = (nl * self._impurity(left) + nr * self._impurity(right)) / n
        gain = self._impurity(total) - child
        k = int(np.argmax(gain))
        return float(gain[k]), float((x[pos[k] - 1] + x[pos[k]]) / 2)

    def predict_proba(self, X):
        return np.array([self._route(x).value for x in as_2d(X)])


class DecisionTreeRegressor(_BaseTree, RegressorMixin):
    def __init__(self, criterion="mse", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0, max_features=None, random_state=None):
        super().__init__(criterion, max_depth, min_samples_split, min_samples_leaf, min_impurity_decrease,
                         max_features, random_state)

    def _prepare_y(self, y):
        return y.astype(float)

    def _node_stats(self, y, w):
        mean = np.average(y, weights=w)
        return np.array([mean]), float(np.average((y - mean) ** 2, weights=w))

    def _best_split(self, x, y, w):
        pos = self._valid_positions(x)
        if len(pos) == 0:
            return None
        # Var = E[y²] - E[y]²; running sums give every candidate's left/right variance in O(n).
        cw, cy, cyy = np.cumsum(w), np.cumsum(w * y), np.cumsum(w * y * y)
        W, Y, YY = cw[-1], cy[-1], cyy[-1]
        nl, sl, ssl = cw[pos - 1], cy[pos - 1], cyy[pos - 1]
        nr, sr, ssr = W - nl, Y - sl, YY - ssl
        sse_left = ssl - sl**2 / nl
        sse_right = ssr - sr**2 / nr
        parent = YY / W - (Y / W) ** 2
        gain = parent - (sse_left + sse_right) / W
        k = int(np.argmax(gain))
        return float(gain[k]), float((x[pos[k] - 1] + x[pos[k]]) / 2)

    def predict(self, X):
        return np.array([self._route(x).value[0] for x in as_2d(X)])
