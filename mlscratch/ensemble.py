"""Ensembles: many weak/unstable models combined into one strong model.

Bagging / Random Forest  - average many high-variance trees trained on bootstrap samples → lower variance.
AdaBoost                 - re-weight samples so each new stump focuses on previous mistakes.
Gradient Boosting        - each new tree fits the negative gradient of the loss (the "residuals").
XGBoost-style boosting   - second-order (gradient + Hessian) trees with L2 leaf penalty and split cost γ.
"""

from __future__ import annotations

import numpy as np

from .base import BaseEstimator, ClassifierMixin, RegressorMixin, as_2d, check_Xy, rng_from, sigmoid, softmax
from .tree import DecisionTreeClassifier, DecisionTreeRegressor


class BaggingClassifier(BaseEstimator, ClassifierMixin):
    """Bootstrap aggregating with any base classifier; prediction = average of probabilities."""

    def __init__(self, base_estimator=None, n_estimators=50, max_samples=1.0, random_state=None):
        self.base_estimator, self.n_estimators = base_estimator, n_estimators
        self.max_samples, self.random_state = max_samples, random_state

    def _make(self, seed):
        return (self.base_estimator.clone() if self.base_estimator is not None
                else DecisionTreeClassifier(random_state=seed))

    def fit(self, X, y):
        X, y = check_Xy(X, y)
        y_idx = self._encode_labels(y)
        rng = rng_from(self.random_state)
        n = len(y)
        m = int(self.max_samples * n)
        K = len(self.classes_)
        self.estimators_, oob_votes = [], np.zeros((n, K))
        for i in range(self.n_estimators):
            idx = rng.integers(0, n, m)                          # sample WITH replacement
            est = self._make(int(rng.integers(1 << 31))).fit(X[idx], y_idx[idx])
            self.estimators_.append(est)
            oob = np.setdiff1d(np.arange(n), idx)                 # ~36.8% of rows are never drawn
            if len(oob):
                oob_votes[oob] += self._proba(est, X[oob], K)
        seen = oob_votes.sum(1) > 0
        self.oob_score_ = float(np.mean(np.argmax(oob_votes[seen], 1) == y_idx[seen])) if seen.any() else np.nan
        return self

    @staticmethod
    def _proba(est, X, K):
        P = np.zeros((len(X), K))
        P[:, est.classes_.astype(int)] = est.predict_proba(X)
        return P

    def predict_proba(self, X):
        X = as_2d(X)
        K = len(self.classes_)
        return np.mean([self._proba(e, X, K) for e in self.estimators_], axis=0)


class RandomForestClassifier(BaggingClassifier):
    """Bagging + a random subset of features at every split (de-correlates the trees)."""

    def __init__(self, n_estimators=100, max_depth=None, max_features="sqrt", min_samples_leaf=1,
                 criterion="gini", random_state=None):
        super().__init__(None, n_estimators, 1.0, random_state)
        self.max_depth, self.max_features, self.min_samples_leaf, self.criterion = max_depth, max_features, min_samples_leaf, criterion

    def get_params(self):
        return {k: getattr(self, k) for k in ("n_estimators", "max_depth", "max_features", "min_samples_leaf",
                                              "criterion", "random_state")}

    def _make(self, seed):
        return DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth, max_features=self.max_features,
                                      min_samples_leaf=self.min_samples_leaf, random_state=seed)

    def fit(self, X, y):
        super().fit(X, y)
        imp = np.mean([e.feature_importances_ for e in self.estimators_], axis=0)
        self.feature_importances_ = imp / imp.sum() if imp.sum() else imp
        return self


class RandomForestRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=None, max_features=1.0, min_samples_leaf=1, random_state=None):
        self.n_estimators, self.max_depth, self.max_features = n_estimators, max_depth, max_features
        self.min_samples_leaf, self.random_state = min_samples_leaf, random_state

    def fit(self, X, y):
        X, y = check_Xy(X, y)
        rng = rng_from(self.random_state)
        n = len(y)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, n)
            self.estimators_.append(DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.max_features,
                                                          min_samples_leaf=self.min_samples_leaf,
                                                          random_state=int(rng.integers(1 << 31))).fit(X[idx], y[idx]))
        imp = np.mean([e.feature_importances_ for e in self.estimators_], axis=0)
        self.feature_importances_ = imp / imp.sum() if imp.sum() else imp
        return self

    def predict(self, X):
        return np.mean([e.predict(X) for e in self.estimators_], axis=0)


class AdaBoostClassifier(BaseEstimator, ClassifierMixin):
    """SAMME AdaBoost with decision stumps.

    err_m  = Σ wᵢ [h_m(xᵢ) ≠ yᵢ] / Σ wᵢ
    α_m    = η · (log((1 - err_m)/err_m) + log(K - 1))
    wᵢ    ← wᵢ · exp(α_m [h_m(xᵢ) ≠ yᵢ])         then normalise
    H(x)   = argmax_k Σ_m α_m [h_m(x) = k]
    """

    def __init__(self, n_estimators=50, learning_rate=1.0, max_depth=1):
        self.n_estimators, self.learning_rate, self.max_depth = n_estimators, learning_rate, max_depth

    def fit(self, X, y):
        X, y = check_Xy(X, y)
        y_idx = self._encode_labels(y)
        n, K = len(y), len(self.classes_)
        w = np.full(n, 1.0 / n)
        self.estimators_, self.alphas_ = [], []
        for _ in range(self.n_estimators):
            stump = DecisionTreeClassifier(max_depth=self.max_depth).fit(X, y_idx, sample_weight=w)
            pred = stump.classes_[np.argmax(stump.predict_proba(X), 1)]
            wrong = pred != y_idx
            err = w[wrong].sum() / w.sum()
            if err >= 1 - 1 / K:   # no better than chance: stop
                break
            alpha = self.learning_rate * (np.log((1 - err) / max(err, 1e-12)) + np.log(K - 1))
            self.estimators_.append(stump)
            self.alphas_.append(alpha)
            if err <= 0:
                break
            w *= np.exp(alpha * wrong)
            w /= w.sum()
        return self

    def decision_function(self, X):
        X = as_2d(X)
        votes = np.zeros((len(X), len(self.classes_)))
        for stump, alpha in zip(self.estimators_, self.alphas_):
            pred = stump.classes_[np.argmax(stump.predict_proba(X), 1)].astype(int)
            votes[np.arange(len(X)), pred] += alpha
        return votes

    def predict_proba(self, X):
        v = self.decision_function(X)
        return softmax(v / max(sum(self.alphas_), 1e-12) * len(self.classes_))


class GradientBoostingRegressor(BaseEstimator, RegressorMixin):
    """F₀ = mean(y);  F_m = F_{m-1} + η·h_m, where h_m is a tree fit to the residuals r = y - F_{m-1}
    (the negative gradient of ½(y - F)²). With loss="absolute", r = sign(y - F) and leaves take the median."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0, loss="squared",
                 min_samples_leaf=1, random_state=None):
        self.n_estimators, self.learning_rate, self.max_depth = n_estimators, learning_rate, max_depth
        self.subsample, self.loss, self.min_samples_leaf, self.random_state = subsample, loss, min_samples_leaf, random_state

    def fit(self, X, y):
        X, y = check_Xy(X, y)
        y = y.astype(float)
        rng = rng_from(self.random_state)
        self.init_ = float(np.median(y) if self.loss == "absolute" else y.mean())
        F = np.full(len(y), self.init_)
        self.estimators_, self.train_loss_ = [], []
        for _ in range(self.n_estimators):
            r = np.sign(y - F) if self.loss == "absolute" else y - F
            idx = rng.choice(len(y), int(self.subsample * len(y)), replace=False) if self.subsample < 1 else np.arange(len(y))
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf).fit(X[idx], r[idx])
            if self.loss == "absolute":  # optimal leaf value for L1 is the median residual in that leaf
                leaf = tree.apply(X[idx])
                for node in tree.leaves():
                    in_leaf = leaf == node.leaf_id
                    if in_leaf.any():
                        node.value = np.array([np.median((y - F)[idx][in_leaf])])
            F += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)
            self.train_loss_.append(float(np.mean(np.abs(y - F)) if self.loss == "absolute" else np.mean((y - F) ** 2)))
        return self

    def predict(self, X):
        X = as_2d(X)
        return self.init_ + self.learning_rate * np.sum([t.predict(X) for t in self.estimators_], axis=0)

    def staged_predict(self, X):
        X = as_2d(X)
        F = np.full(len(X), self.init_)
        for t in self.estimators_:
            F = F + self.learning_rate * t.predict(X)
            yield F.copy()


class GradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    """Log-loss gradient boosting.

    Binary: F = log-odds; residual r = y - σ(F); Newton leaf value γ = Σr / Σ p(1-p).
    K classes: K trees per round on r_k = y_k - softmax(F)_k with γ = (K-1)/K · Σr / Σ|r|(1-|r|) (Friedman 2001).
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0, min_samples_leaf=1,
                 random_state=None):
        self.n_estimators, self.learning_rate, self.max_depth = n_estimators, learning_rate, max_depth
        self.subsample, self.min_samples_leaf, self.random_state = subsample, min_samples_leaf, random_state

    def _tree(self, X, r, denom, idx, scale=1.0):
        tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf).fit(X[idx], r[idx])
        leaf = tree.apply(X[idx])
        for node in tree.leaves():
            m = leaf == node.leaf_id
            node.value = np.array([scale * r[idx][m].sum() / max(denom[idx][m].sum(), 1e-12)]) if m.any() else np.array([0.0])
        return tree

    def fit(self, X, y):
        X, y = check_Xy(X, y)
        y_idx = self._encode_labels(y)
        K, n = len(self.classes_), len(y)
        rng = rng_from(self.random_state)
        self.estimators_ = []
        if K == 2:
            p0 = np.clip(y_idx.mean(), 1e-6, 1 - 1e-6)
            self.init_ = np.array([np.log(p0 / (1 - p0))])
            F = np.full(n, self.init_[0])
            for _ in range(self.n_estimators):
                p = sigmoid(F)
                idx = rng.choice(n, int(self.subsample * n), replace=False) if self.subsample < 1 else np.arange(n)
                tree = self._tree(X, y_idx - p, p * (1 - p), idx)
                F += self.learning_rate * tree.predict(X)
                self.estimators_.append([tree])
        else:
            prior = np.bincount(y_idx, minlength=K) / n
            self.init_ = np.log(np.clip(prior, 1e-12, None))
            F = np.tile(self.init_, (n, 1))
            Y = np.eye(K)[y_idx]
            for _ in range(self.n_estimators):
                P = softmax(F)
                idx = rng.choice(n, int(self.subsample * n), replace=False) if self.subsample < 1 else np.arange(n)
                round_trees = []
                for k in range(K):
                    r = Y[:, k] - P[:, k]
                    tree = self._tree(X, r, np.abs(r) * (1 - np.abs(r)), idx, scale=(K - 1) / K)
                    F[:, k] += self.learning_rate * tree.predict(X)
                    round_trees.append(tree)
                self.estimators_.append(round_trees)
        return self

    def decision_function(self, X):
        X = as_2d(X)
        F = np.tile(self.init_, (len(X), 1))
        for round_trees in self.estimators_:
            for k, tree in enumerate(round_trees):
                F[:, k] += self.learning_rate * tree.predict(X)
        return F[:, 0] if len(self.classes_) == 2 else F

    def predict_proba(self, X):
        F = self.decision_function(X)
        if F.ndim == 1:
            p = sigmoid(F)
            return np.column_stack([1 - p, p])
        return softmax(F)


class _XGBNode:
    __slots__ = ("feature", "threshold", "left", "right", "weight")

    def __init__(self, weight=0.0):
        self.feature, self.threshold, self.left, self.right, self.weight = -1, 0.0, None, None, weight


class XGBoostLike(BaseEstimator):
    """Second-order boosting with the XGBoost objective.

    Each round, for loss l(y, F): g = ∂l/∂F, h = ∂²l/∂F². For a leaf containing samples I:
        optimal weight   w* = -G / (H + λ)            (G = Σg, H = Σh)
        its score        -½ G² / (H + λ)
    split gain = ½ [G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ)] - γ   → only split if gain > 0.
    """

    def __init__(self, objective="reg:squarederror", n_estimators=100, learning_rate=0.3, max_depth=6,
                 reg_lambda=1.0, gamma=0.0, min_child_weight=1.0, subsample=1.0, colsample=1.0, random_state=None):
        self.objective, self.n_estimators, self.learning_rate, self.max_depth = objective, n_estimators, learning_rate, max_depth
        self.reg_lambda, self.gamma, self.min_child_weight = reg_lambda, gamma, min_child_weight
        self.subsample, self.colsample, self.random_state = subsample, colsample, random_state

    def _grad_hess(self, y, F):
        if self.objective == "binary:logistic":
            p = sigmoid(F)
            return p - y, p * (1 - p)
        return F - y, np.ones_like(y)

    def _build(self, X, g, h, depth, features):
        G, H = g.sum(), h.sum()
        node = _XGBNode(weight=-G / (H + self.reg_lambda))
        if depth >= self.max_depth or len(g) < 2:
            return node
        best_gain, best = 0.0, None
        parent = G**2 / (H + self.reg_lambda)
        for f in features:
            order = np.argsort(X[:, f], kind="mergesort")
            xs, gs, hs = X[order, f], np.cumsum(g[order]), np.cumsum(h[order])
            pos = np.flatnonzero(np.diff(xs) > 1e-12)  # split after index pos
            if len(pos) == 0:
                continue
            GL, HL = gs[pos], hs[pos]
            GR, HR = G - GL, H - HL
            ok = (HL >= self.min_child_weight) & (HR >= self.min_child_weight)
            gain = 0.5 * (GL**2 / (HL + self.reg_lambda) + GR**2 / (HR + self.reg_lambda) - parent) - self.gamma
            gain = np.where(ok, gain, -np.inf)
            k = int(np.argmax(gain))
            if gain[k] > best_gain:
                best_gain, best = gain[k], (f, (xs[pos[k]] + xs[pos[k] + 1]) / 2)
        if best is None:
            return node
        node.feature, node.threshold = int(best[0]), float(best[1])
        mask = X[:, node.feature] <= node.threshold
        node.left = self._build(X[mask], g[mask], h[mask], depth + 1, features)
        node.right = self._build(X[~mask], g[~mask], h[~mask], depth + 1, features)
        return node

    @staticmethod
    def _predict_tree(node, X):
        out = np.empty(len(X))
        for i, x in enumerate(X):
            n = node
            while n.left is not None:
                n = n.left if x[n.feature] <= n.threshold else n.right
            out[i] = n.weight
        return out

    def fit(self, X, y):
        X, y = check_Xy(X, y)
        y = y.astype(float)
        rng = rng_from(self.random_state)
        self.base_score_ = 0.0 if self.objective == "binary:logistic" else float(y.mean())
        F = np.full(len(y), self.base_score_)
        self.trees_ = []
        d = X.shape[1]
        for _ in range(self.n_estimators):
            g, h = self._grad_hess(y, F)
            rows = rng.choice(len(y), int(self.subsample * len(y)), replace=False) if self.subsample < 1 else np.arange(len(y))
            feats = rng.choice(d, max(1, int(self.colsample * d)), replace=False)
            tree = self._build(X[rows], g[rows], h[rows], 0, feats)
            F += self.learning_rate * self._predict_tree(tree, X)
            self.trees_.append(tree)
        return self

    def decision_function(self, X):
        X = as_2d(X)
        return self.base_score_ + self.learning_rate * np.sum([self._predict_tree(t, X) for t in self.trees_], axis=0)

    def predict_proba(self, X):
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        F = self.decision_function(X)
        return (sigmoid(F) > 0.5).astype(int) if self.objective == "binary:logistic" else F
