"""Feature extraction / dimensionality reduction: PCA, Kernel PCA, LDA (Fisher), truncated SVD, t-SNE."""

from __future__ import annotations

import numpy as np

from .base import BaseEstimator, as_2d, check_Xy, rng_from
from .svm import kernel_matrix


class PCA(BaseEstimator):
    """Project onto the directions of maximum variance.

    Covariance  Σ = XcᵀXc / (n-1).  Its eigenvectors are the principal axes; eigenvalues = variance along them.
    We use the SVD Xc = U S Vᵀ instead (numerically better): axes = rows of Vᵀ, variances = S² / (n-1).
    """

    def __init__(self, n_components=None, whiten=False):
        self.n_components, self.whiten = n_components, whiten

    def fit(self, X, y=None):
        X = as_2d(X)
        self.mean_ = X.mean(0)
        U, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        # deterministic sign: largest-magnitude loading of each component is positive
        signs = np.sign(Vt[np.arange(len(Vt)), np.argmax(np.abs(Vt), axis=1)])
        Vt *= signs[:, None]
        var = S**2 / (len(X) - 1)
        k = self.n_components
        if isinstance(k, float) and 0 < k < 1:      # keep enough components to explain k of the variance
            k = int(np.searchsorted(np.cumsum(var) / var.sum(), k) + 1)
        k = k or len(S)
        self.n_components_ = k
        self.components_ = Vt[:k]
        self.explained_variance_ = var[:k]
        self.explained_variance_ratio_ = var[:k] / var.sum()
        self.singular_values_ = S[:k]
        return self

    def transform(self, X):
        Z = (as_2d(X) - self.mean_) @ self.components_.T
        return Z / np.sqrt(self.explained_variance_) if self.whiten else Z

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        Z = as_2d(Z) * (np.sqrt(self.explained_variance_) if self.whiten else 1)
        return Z @ self.components_ + self.mean_


class TruncatedSVD(BaseEstimator):
    """SVD without centring (works on sparse/count data, a.k.a. LSA for text)."""

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y=None):
        U, S, Vt = np.linalg.svd(as_2d(X), full_matrices=False)
        self.components_, self.singular_values_ = Vt[: self.n_components], S[: self.n_components]
        return self

    def transform(self, X):
        return as_2d(X) @ self.components_.T

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class KernelPCA(BaseEstimator):
    """PCA in the feature space of a kernel: eigen-decompose the centred kernel matrix K̃ = HKH."""

    def __init__(self, n_components=2, kernel="rbf", gamma=1.0, degree=3):
        self.n_components, self.kernel, self.gamma, self.degree = n_components, kernel, gamma, degree

    def fit(self, X, y=None):
        self.X_ = as_2d(X)
        K = kernel_matrix(self.X_, self.X_, self.kernel, self.gamma, self.degree)
        n = len(K)
        one = np.full((n, n), 1 / n)
        Kc = K - one @ K - K @ one + one @ K @ one
        vals, vecs = np.linalg.eigh(Kc)
        order = np.argsort(vals)[::-1][: self.n_components]
        self.lambdas_, self.alphas_ = vals[order], vecs[:, order]
        self._K_fit_rows = K.mean(0)
        self._K_fit_all = K.mean()
        return self

    def transform(self, X):
        K = kernel_matrix(as_2d(X), self.X_, self.kernel, self.gamma, self.degree)
        Kc = K - K.mean(1, keepdims=True) - self._K_fit_rows + self._K_fit_all
        return Kc @ (self.alphas_ / np.sqrt(np.maximum(self.lambdas_, 1e-12)))

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class LinearDiscriminantAnalysis(BaseEstimator):
    """Fisher LDA: directions maximising between-class scatter relative to within-class scatter.

    S_W = Σ_k Σ_{i∈k} (xᵢ-μ_k)(xᵢ-μ_k)ᵀ,   S_B = Σ_k n_k (μ_k-μ)(μ_k-μ)ᵀ
    Solve S_W⁻¹ S_B v = λ v; at most K-1 useful directions. Also a classifier (shared-covariance Gaussians).
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y):
        X, y = check_Xy(X, y)
        self.classes_ = np.unique(y)
        mu = X.mean(0)
        d = X.shape[1]
        Sw, Sb = np.zeros((d, d)), np.zeros((d, d))
        self.means_, priors = [], []
        for c in self.classes_:
            Xc = X[y == c]
            m = Xc.mean(0)
            self.means_.append(m)
            priors.append(len(Xc) / len(X))
            Sw += (Xc - m).T @ (Xc - m)
            Sb += len(Xc) * np.outer(m - mu, m - mu)
        self.means_, self.priors_ = np.array(self.means_), np.array(priors)
        vals, vecs = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
        order = np.argsort(vals.real)[::-1]
        k = self.n_components or min(len(self.classes_) - 1, d)
        self.scalings_ = vecs[:, order[:k]].real
        self.explained_variance_ratio_ = (vals.real[order] / vals.real.sum())[:k]
        self.mean_ = mu
        self._cov = Sw / (len(X) - len(self.classes_))
        return self

    def transform(self, X):
        return (as_2d(X) - self.mean_) @ self.scalings_

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def predict(self, X):
        inv = np.linalg.pinv(self._cov)
        X = as_2d(X)
        # linear discriminant δ_k(x) = xᵀΣ⁻¹μ_k - ½μ_kᵀΣ⁻¹μ_k + log π_k
        scores = X @ inv @ self.means_.T - 0.5 * np.sum(self.means_ @ inv * self.means_, 1) + np.log(self.priors_)
        return self.classes_[np.argmax(scores, 1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


class TSNE(BaseEstimator):
    """t-SNE (exact, O(n²)): match neighbourhood probabilities in high-D (Gaussian) and 2-D (Student-t).

    p_{j|i} ∝ exp(-||xᵢ-xⱼ||²/2σᵢ²) with σᵢ found by binary search so perplexity = 2^H(Pᵢ);  P = (P + Pᵀ)/2n
    q_ij ∝ (1 + ||yᵢ-yⱼ||²)⁻¹;   minimise KL(P||Q);   ∂KL/∂yᵢ = 4 Σⱼ (p_ij - q_ij)(yᵢ-yⱼ)(1+||yᵢ-yⱼ||²)⁻¹
    """

    def __init__(self, n_components=2, perplexity=30.0, n_iter=750, learning_rate=200.0, random_state=None):
        self.n_components, self.perplexity, self.n_iter, self.learning_rate, self.random_state = n_components, perplexity, n_iter, learning_rate, random_state

    def _p_matrix(self, X):
        D = np.maximum(((X[:, None] - X[None]) ** 2).sum(-1), 0)
        n = len(X)
        P = np.zeros((n, n))
        target = np.log(self.perplexity)
        for i in range(n):
            lo, hi, beta = 1e-20, 1e20, 1.0
            d = np.delete(D[i], i)
            for _ in range(60):
                p = np.exp(-(d - d.min()) * beta)
                p /= p.sum()
                H = -np.sum(p * np.log(np.maximum(p, 1e-12)))
                if abs(H - target) < 1e-5:
                    break
                if H > target:
                    lo = beta
                    beta = beta * 2 if hi == 1e20 else (beta + hi) / 2
                else:
                    hi = beta
                    beta = (beta + lo) / 2
            P[i, np.arange(n) != i] = p
        return (P + P.T) / (2 * n)

    def fit_transform(self, X, y=None):
        X = as_2d(X)
        rng = rng_from(self.random_state)
        P = np.maximum(self._p_matrix(X), 1e-12)
        Y = rng.normal(0, 1e-4, (len(X), self.n_components))
        velocity = np.zeros_like(Y)
        for it in range(self.n_iter):
            exaggeration = 12.0 if it < 250 else 1.0           # early exaggeration forms tight clusters first
            sq = np.maximum(((Y[:, None] - Y[None]) ** 2).sum(-1), 0)
            num = 1 / (1 + sq)
            np.fill_diagonal(num, 0)
            Q = np.maximum(num / num.sum(), 1e-12)
            PQ = (exaggeration * P - Q) * num
            grad = 4 * (np.diag(PQ.sum(1)) - PQ) @ Y
            momentum = 0.5 if it < 250 else 0.8
            velocity = momentum * velocity - self.learning_rate * grad
            Y += velocity
        self.kl_divergence_ = float(np.sum(P * np.log(P / Q)))
        self.embedding_ = Y
        return Y
