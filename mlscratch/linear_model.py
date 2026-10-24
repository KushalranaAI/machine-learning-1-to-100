"""Linear models from first principles.

Regression:      ŷ = Xw + b
Classification:  p(y=1|x) = σ(xᵀw + b)   (logistic)     p(y=k|x) = softmax(xᵀW + b)_k   (multinomial)
"""

from __future__ import annotations

import numpy as np

from .base import BaseEstimator, ClassifierMixin, RegressorMixin, add_bias, as_2d, check_Xy, one_hot, rng_from, sigmoid, softmax


class LinearRegression(BaseEstimator, RegressorMixin):
    """Ordinary least squares.

    solver="normal": closed form w = (XᵀX)⁻¹Xᵀy, computed stably with a least-squares solve.
    solver="gd":     batch gradient descent on MSE, gradient = (2/n) Xᵀ(Xw - y).
    """

    def __init__(self, solver: str = "normal", lr: float = 0.01, n_iter: int = 5000, tol: float = 1e-10):
        self.solver, self.lr, self.n_iter, self.tol = solver, lr, n_iter, tol

    def fit(self, X, y):
        X, y = check_Xy(X, y)
        Xb = add_bias(X)
        if self.solver == "normal":
            # lstsq solves min ||Xb w - y||² via SVD: same answer as the normal equation, but it
            # also works when XᵀX is singular (collinear features) and is more accurate.
            w, *_ = np.linalg.lstsq(Xb, y.astype(float), rcond=None)
            self.loss_history_ = []
        else:
            w = np.zeros(Xb.shape[1])
            self.loss_history_ = []
            n = len(y)
            for _ in range(self.n_iter):
                residual = Xb @ w - y
                grad = (2 / n) * Xb.T @ residual
                w -= self.lr * grad
                self.loss_history_.append(float(np.mean(residual**2)))
                if np.linalg.norm(grad) < self.tol:
                    break
        self.intercept_, self.coef_ = w[0], w[1:]
        return self

    def predict(self, X):
        return as_2d(X) @ self.coef_ + self.intercept_


class Ridge(BaseEstimator, RegressorMixin):
    """L2-regularised least squares: min ||y - Xw - b||² + α||w||².

    Closed form (intercept not penalised, so we centre first): w = (XᵀX + αI)⁻¹ Xᵀy.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X, y = check_Xy(X, y)
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - x_mean, y - y_mean
        A = Xc.T @ Xc + self.alpha * np.eye(X.shape[1])
        self.coef_ = np.linalg.solve(A, Xc.T @ yc)
        self.intercept_ = y_mean - x_mean @ self.coef_
        return self

    def predict(self, X):
        return as_2d(X) @ self.coef_ + self.intercept_


def _soft_threshold(rho: float, lam: float) -> float:
    """S(ρ, λ) = sign(ρ)·max(|ρ| - λ, 0): the proximal operator of λ|w| - it is what makes Lasso weights exactly 0."""
    return np.sign(rho) * max(abs(rho) - lam, 0.0)


class ElasticNet(BaseEstimator, RegressorMixin):
    """min (1/2n)||y - Xw - b||² + α·l1_ratio·||w||₁ + (α/2)(1 - l1_ratio)||w||²  by coordinate descent.

    One coordinate at a time the problem is 1-D and has a closed-form solution:
        w_j = S(ρ_j, α·l1_ratio) / (z_j + α(1 - l1_ratio)),
        ρ_j = (1/n) x_jᵀ(r + x_j w_j),  z_j = (1/n) x_jᵀx_j,  r = residual.
    """

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, n_iter: int = 1000, tol: float = 1e-6):
        self.alpha, self.l1_ratio, self.n_iter, self.tol = alpha, l1_ratio, n_iter, tol

    def fit(self, X, y):
        X, y = check_Xy(X, y)
        n, d = X.shape
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - x_mean, y - y_mean
        w = np.zeros(d)
        z = (Xc**2).sum(axis=0) / n
        l1, l2 = self.alpha * self.l1_ratio, self.alpha * (1 - self.l1_ratio)
        r = yc - Xc @ w
        for self.n_iter_ in range(1, self.n_iter + 1):
            max_change = 0.0
            for j in range(d):
                if z[j] == 0:
                    continue
                old = w[j]
                rho = Xc[:, j] @ (r + Xc[:, j] * old) / n
                w[j] = _soft_threshold(rho, l1) / (z[j] + l2)
                if w[j] != old:
                    r -= Xc[:, j] * (w[j] - old)
                    max_change = max(max_change, abs(w[j] - old))
            if max_change < self.tol:
                break
        self.coef_, self.intercept_ = w, y_mean - x_mean @ w
        return self

    def predict(self, X):
        return as_2d(X) @ self.coef_ + self.intercept_


class Lasso(ElasticNet):
    """L1-regularised least squares (sparse weights → automatic feature selection)."""

    def __init__(self, alpha: float = 1.0, n_iter: int = 1000, tol: float = 1e-6):
        super().__init__(alpha=alpha, l1_ratio=1.0, n_iter=n_iter, tol=tol)

    def get_params(self):
        return {"alpha": self.alpha, "n_iter": self.n_iter, "tol": self.tol}


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """Binary logistic regression trained by gradient descent on the (L2-regularised) cross-entropy.

    p = σ(Xw + b),   L = -(1/n) Σ [y log p + (1-y) log(1-p)] + (λ/2n)||w||²
    ∂L/∂w = (1/n) Xᵀ(p - y) + (λ/n) w,     ∂L/∂b = (1/n) Σ (p - y)
    Multiclass input is handled one-vs-rest.
    """

    def __init__(self, lr: float = 0.1, n_iter: int = 2000, C: float = 1.0, fit_intercept: bool = True,
                 batch_size: int | None = None, random_state=None):
        self.lr, self.n_iter, self.C, self.fit_intercept = lr, n_iter, C, fit_intercept
        self.batch_size, self.random_state = batch_size, random_state

    def _fit_binary(self, X, y):
        n, d = X.shape
        w, b = np.zeros(d), 0.0
        lam = 1.0 / self.C
        rng = rng_from(self.random_state)
        losses = []
        for _ in range(self.n_iter):
            idx = rng.choice(n, self.batch_size, replace=False) if self.batch_size else slice(None)
            Xi, yi = X[idx], y[idx]
            m = len(yi)
            p = sigmoid(Xi @ w + b)
            w -= self.lr * (Xi.T @ (p - yi) / m + lam * w / n)
            if self.fit_intercept:
                b -= self.lr * np.mean(p - yi)
            pf = np.clip(sigmoid(X @ w + b), 1e-12, 1 - 1e-12)
            losses.append(float(-np.mean(y * np.log(pf) + (1 - y) * np.log(1 - pf)) + lam * w @ w / (2 * n)))
        return w, b, losses

    def fit(self, X, y):
        X, y = check_Xy(X, y)
        y_idx = self._encode_labels(y)
        if len(self.classes_) == 2:
            w, b, self.loss_history_ = self._fit_binary(X, y_idx.astype(float))
            self.coef_, self.intercept_ = w[None, :], np.array([b])
        else:  # one-vs-rest
            fits = [self._fit_binary(X, (y_idx == k).astype(float)) for k in range(len(self.classes_))]
            self.coef_ = np.array([f[0] for f in fits])
            self.intercept_ = np.array([f[1] for f in fits])
            self.loss_history_ = fits[0][2]
        return self

    def decision_function(self, X):
        scores = as_2d(X) @ self.coef_.T + self.intercept_
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict_proba(self, X):
        s = self.decision_function(X)
        if s.ndim == 1:
            p = sigmoid(s)
            return np.column_stack([1 - p, p])
        P = sigmoid(s)
        return P / P.sum(axis=1, keepdims=True)


class SoftmaxRegression(BaseEstimator, ClassifierMixin):
    """Multinomial logistic regression: P = softmax(XW + b); gradient of cross-entropy is Xᵀ(P - Y)/n."""

    def __init__(self, lr: float = 0.1, n_iter: int = 2000, reg: float = 0.0):
        self.lr, self.n_iter, self.reg = lr, n_iter, reg

    def fit(self, X, y):
        X, y = check_Xy(X, y)
        Y = one_hot(self._encode_labels(y), len(np.unique(y)))
        n, d = X.shape
        W, b = np.zeros((d, Y.shape[1])), np.zeros(Y.shape[1])
        self.loss_history_ = []
        for _ in range(self.n_iter):
            P = softmax(X @ W + b)
            W -= self.lr * (X.T @ (P - Y) / n + self.reg * W)
            b -= self.lr * (P - Y).mean(axis=0)
            self.loss_history_.append(float(-np.mean(np.log(np.clip((P * Y).sum(1), 1e-12, None)))))
        self.coef_, self.intercept_ = W, b
        return self

    def predict_proba(self, X):
        return softmax(as_2d(X) @ self.coef_ + self.intercept_)


class Perceptron(BaseEstimator, ClassifierMixin):
    """Rosenblatt's perceptron: on each mistake, w ← w + η·y·x (labels in {-1, +1}).

    Converges in finite steps iff the data is linearly separable (Novikoff's theorem).
    """

    def __init__(self, lr: float = 1.0, n_iter: int = 100, random_state=None):
        self.lr, self.n_iter, self.random_state = lr, n_iter, random_state

    def fit(self, X, y):
        X, y = check_Xy(X, y)
        t = np.where(self._encode_labels(y) == 1, 1.0, -1.0)
        w, b = np.zeros(X.shape[1]), 0.0
        rng = rng_from(self.random_state)
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for i in rng.permutation(len(t)):
                if t[i] * (X[i] @ w + b) <= 0:
                    w += self.lr * t[i] * X[i]
                    b += self.lr * t[i]
                    errors += 1
            self.errors_.append(errors)
            if errors == 0:
                break
        self.coef_, self.intercept_ = w, b
        return self

    def predict(self, X):
        return self.classes_[(as_2d(X) @ self.coef_ + self.intercept_ > 0).astype(int)]
