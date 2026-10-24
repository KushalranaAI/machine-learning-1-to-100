"""Support Vector Machines.

Primal (soft margin):   min_w,b  ½||w||² + C Σ max(0, 1 - yᵢ(wᵀxᵢ + b))       (hinge loss)
Dual:                   max_α  Σαᵢ - ½ ΣΣ αᵢαⱼ yᵢyⱼ K(xᵢ, xⱼ)   s.t. 0 ≤ αᵢ ≤ C,  Σ αᵢyᵢ = 0
Decision function:      f(x) = Σ αᵢ yᵢ K(xᵢ, x) + b      (only support vectors have αᵢ > 0)
"""

from __future__ import annotations

import numpy as np

from .base import BaseEstimator, ClassifierMixin, as_2d, check_Xy, rng_from


class LinearSVC(BaseEstimator, ClassifierMixin):
    """Linear SVM by (sub)gradient descent on the primal hinge-loss objective.

    Subgradient: ∂/∂w = w - C Σ_{margin violators} yᵢxᵢ ,  ∂/∂b = -C Σ_{violators} yᵢ   (averaged over n)
    """

    def __init__(self, C: float = 1.0, lr: float = 0.01, n_iter: int = 2000):
        self.C, self.lr, self.n_iter = C, lr, n_iter

    def fit(self, X, y):
        X, y = check_Xy(X, y)
        t = np.where(self._encode_labels(y) == 1, 1.0, -1.0)
        n, d = X.shape
        w, b = np.zeros(d), 0.0
        for _ in range(self.n_iter):
            margin = t * (X @ w + b)
            v = margin < 1
            w -= self.lr * (w / n - self.C * (t[v] @ X[v]) / n)
            b -= self.lr * (-self.C * t[v].sum() / n)
        self.coef_, self.intercept_ = w, b
        return self

    def decision_function(self, X):
        return as_2d(X) @ self.coef_ + self.intercept_

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]


def kernel_matrix(A, B, kernel="rbf", gamma=1.0, degree=3, coef0=1.0):
    if kernel == "linear":
        return A @ B.T
    if kernel == "poly":
        return (gamma * A @ B.T + coef0) ** degree
    if kernel == "rbf":  # K(a, b) = exp(-γ ||a - b||²): an infinite-dimensional feature map
        sq = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
        return np.exp(-gamma * np.maximum(sq, 0))
    raise ValueError(kernel)


class SVC(BaseEstimator, ClassifierMixin):
    """Kernel SVM solved in the dual with Platt's SMO (Sequential Minimal Optimisation).

    SMO repeatedly picks two multipliers (αᵢ, αⱼ), solves the 2-variable QP analytically,
    clips to the box [L, H] and updates the bias. Binary problems; multiclass via one-vs-rest.
    """

    def __init__(self, C: float = 1.0, kernel: str = "rbf", gamma: float | str = "scale", degree: int = 3,
                 tol: float = 1e-3, max_passes: int = 10, max_iter: int = 10000, random_state=0):
        self.C, self.kernel, self.gamma, self.degree = C, kernel, gamma, degree
        self.tol, self.max_passes, self.max_iter, self.random_state = tol, max_passes, max_iter, random_state

    def _smo(self, K, t):
        n = len(t)
        alpha, b = np.zeros(n), 0.0
        rng = rng_from(self.random_state)
        passes = it = 0
        while passes < self.max_passes and it < self.max_iter:
            changed = 0
            for i in range(n):
                Ei = (alpha * t) @ K[:, i] + b - t[i]
                if (t[i] * Ei < -self.tol and alpha[i] < self.C) or (t[i] * Ei > self.tol and alpha[i] > 0):
                    j = int(rng.integers(n - 1))
                    j += j >= i
                    Ej = (alpha * t) @ K[:, j] + b - t[j]
                    ai, aj = alpha[i], alpha[j]
                    if t[i] != t[j]:
                        L, H = max(0, aj - ai), min(self.C, self.C + aj - ai)
                    else:
                        L, H = max(0, ai + aj - self.C), min(self.C, ai + aj)
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]          # second derivative along the constraint line
                    if L == H or eta >= 0:
                        continue
                    alpha[j] = np.clip(aj - t[j] * (Ei - Ej) / eta, L, H)
                    if abs(alpha[j] - aj) < 1e-7:
                        continue
                    alpha[i] = ai + t[i] * t[j] * (aj - alpha[j])
                    b1 = b - Ei - t[i] * (alpha[i] - ai) * K[i, i] - t[j] * (alpha[j] - aj) * K[i, j]
                    b2 = b - Ej - t[i] * (alpha[i] - ai) * K[i, j] - t[j] * (alpha[j] - aj) * K[j, j]
                    b = b1 if 0 < alpha[i] < self.C else b2 if 0 < alpha[j] < self.C else (b1 + b2) / 2
                    changed += 1
            passes = passes + 1 if changed == 0 else 0
            it += 1
        return alpha, b

    def fit(self, X, y):
        X, y = check_Xy(X, y)
        y_idx = self._encode_labels(y)
        self.gamma_ = 1.0 / (X.shape[1] * X.var()) if self.gamma == "scale" else float(self.gamma)
        self.X_ = X
        K = kernel_matrix(X, X, self.kernel, self.gamma_, self.degree)
        targets = [y_idx == 1] if len(self.classes_) == 2 else [y_idx == k for k in range(len(self.classes_))]
        self.models_ = []
        for pos in targets:
            t = np.where(pos, 1.0, -1.0)
            alpha, b = self._smo(K, t)
            sv = alpha > 1e-8
            self.models_.append((alpha[sv] * t[sv], np.flatnonzero(sv), b))
        self.support_ = np.unique(np.concatenate([m[1] for m in self.models_]))
        return self

    def decision_function(self, X):
        X = as_2d(X)
        out = []
        for coef, sv, b in self.models_:
            out.append(kernel_matrix(X, self.X_[sv], self.kernel, self.gamma_, self.degree) @ coef + b)
        out = np.column_stack(out)
        return out.ravel() if out.shape[1] == 1 else out

    def predict(self, X):
        s = self.decision_function(X)
        return self.classes_[(s > 0).astype(int)] if s.ndim == 1 else self.classes_[np.argmax(s, axis=1)]
