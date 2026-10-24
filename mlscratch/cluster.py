"""Clustering: k-means(++), agglomerative (hierarchical), DBSCAN and Gaussian mixtures (EM)."""

from __future__ import annotations

import numpy as np

from .base import BaseEstimator, as_2d, rng_from
from .neighbors import pairwise_distances


class KMeans(BaseEstimator):
    """Lloyd's algorithm minimising inertia J = Σᵢ ||xᵢ - μ_c(i)||².

    1. init centroids (k-means++: pick each next centre with probability ∝ D(x)²)
    2. assign:  c(i) = argmin_k ||xᵢ - μ_k||
    3. update:  μ_k = mean of points assigned to k
    repeat 2-3 until assignments stop changing. Each step can only lower J, so it converges
    (to a local minimum - hence n_init restarts).
    """

    def __init__(self, n_clusters=8, init="k-means++", n_init=10, max_iter=300, tol=1e-6, random_state=None):
        self.n_clusters, self.init, self.n_init, self.max_iter, self.tol, self.random_state = n_clusters, init, n_init, max_iter, tol, random_state

    def _init_centroids(self, X, rng):
        if self.init == "random":
            return X[rng.choice(len(X), self.n_clusters, replace=False)].copy()
        centers = [X[rng.integers(len(X))]]
        for _ in range(1, self.n_clusters):
            d2 = np.min(pairwise_distances(X, np.array(centers)) ** 2, axis=1)
            centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        return np.array(centers)

    def _run(self, X, rng):
        C = self._init_centroids(X, rng)
        history = [C.copy()]
        for it in range(1, self.max_iter + 1):
            labels = np.argmin(pairwise_distances(X, C), axis=1)
            newC = np.array([X[labels == k].mean(0) if np.any(labels == k) else C[k] for k in range(self.n_clusters)])
            shift = np.linalg.norm(newC - C)
            C = newC
            history.append(C.copy())
            if shift < self.tol:
                break
        labels = np.argmin(pairwise_distances(X, C), axis=1)
        inertia = float(((X - C[labels]) ** 2).sum())
        return C, labels, inertia, it, history

    def fit(self, X, y=None):
        X = as_2d(X)
        rng = rng_from(self.random_state)
        best = min((self._run(X, rng) for _ in range(self.n_init)), key=lambda r: r[2])
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_, self.history_ = best
        return self

    def predict(self, X):
        return np.argmin(pairwise_distances(as_2d(X), self.cluster_centers_), axis=1)

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


class AgglomerativeClustering(BaseEstimator):
    """Bottom-up hierarchical clustering: start with n singletons, repeatedly merge the closest pair.

    Cluster distance (linkage): single = min, complete = max, average = mean, ward = increase in SSE.
    Lance-Williams update keeps merging O(n²) per step with a distance matrix.
    """

    def __init__(self, n_clusters=2, linkage="ward"):
        self.n_clusters, self.linkage = n_clusters, linkage

    def fit(self, X, y=None):
        X = as_2d(X)
        n = len(X)
        D = pairwise_distances(X, X) ** (2 if self.linkage == "ward" else 1)
        np.fill_diagonal(D, np.inf)
        size = np.ones(n)
        active = np.ones(n, bool)
        members = {i: [i] for i in range(n)}
        self.children_, self.distances_ = [], []
        ids = list(range(n))
        next_id = n
        while active.sum() > 1:
            Dm = np.where(active[:, None] & active[None, :], D, np.inf)
            i, j = np.unravel_index(np.argmin(Dm), D.shape)
            dij = D[i, j]
            self.children_.append((ids[i], ids[j]))
            self.distances_.append(float(np.sqrt(dij) if self.linkage == "ward" else dij))
            # Lance-Williams: distance from the merged cluster (i ∪ j) to every other cluster k
            ni, nj, nk = size[i], size[j], size
            if self.linkage == "single":
                new = np.minimum(D[i], D[j])
            elif self.linkage == "complete":
                new = np.maximum(D[i], D[j])
            elif self.linkage == "average":
                new = (ni * D[i] + nj * D[j]) / (ni + nj)
            else:  # ward (on squared distances)
                new = ((ni + nk) * D[i] + (nj + nk) * D[j] - nk * dij) / (ni + nj + nk)
            D[i, :], D[:, i] = new, new
            D[i, i] = np.inf
            active[j] = False
            size[i] += size[j]
            members[i] = members[i] + members.pop(j)
            ids[i] = next_id
            next_id += 1
            if active.sum() == self.n_clusters:
                self.labels_ = np.empty(n, int)
                for label, k in enumerate(np.flatnonzero(active)):
                    self.labels_[members[k]] = label
        if self.n_clusters == 1:
            self.labels_ = np.zeros(n, int)
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


class DBSCAN(BaseEstimator):
    """Density-based clustering. Core point: ≥ min_samples neighbours within eps.
    Clusters = core points connected through neighbourhoods (+ their border points). Others = noise (-1)."""

    def __init__(self, eps=0.5, min_samples=5):
        self.eps, self.min_samples = eps, min_samples

    def fit(self, X, y=None):
        X = as_2d(X)
        D = pairwise_distances(X, X)
        neighbours = [np.flatnonzero(row <= self.eps) for row in D]
        core = np.array([len(nb) >= self.min_samples for nb in neighbours])
        labels = np.full(len(X), -1)
        cluster = 0
        for i in range(len(X)):
            if labels[i] != -1 or not core[i]:
                continue
            labels[i] = cluster
            stack = [i]
            while stack:                     # breadth of the density-connected region
                p = stack.pop()
                if not core[p]:
                    continue
                for q in neighbours[p]:
                    if labels[q] == -1:
                        labels[q] = cluster
                        stack.append(q)
            cluster += 1
        self.labels_, self.core_sample_indices_ = labels, np.flatnonzero(core)
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


class GaussianMixture(BaseEstimator):
    """Mixture of K Gaussians fit by Expectation-Maximisation.

    E-step: responsibilities r_ik = π_k N(xᵢ|μ_k,Σ_k) / Σ_j π_j N(xᵢ|μ_j,Σ_j)
    M-step: N_k = Σᵢ r_ik,  π_k = N_k/n,  μ_k = Σᵢ r_ik xᵢ / N_k,  Σ_k = Σᵢ r_ik (xᵢ-μ_k)(xᵢ-μ_k)ᵀ / N_k
    The log-likelihood never decreases between iterations.
    """

    def __init__(self, n_components=2, max_iter=200, tol=1e-6, reg_covar=1e-6, random_state=None):
        self.n_components, self.max_iter, self.tol, self.reg_covar, self.random_state = n_components, max_iter, tol, reg_covar, random_state

    def _log_gauss(self, X):
        n, d = X.shape
        out = np.empty((n, self.n_components))
        for k in range(self.n_components):
            L = np.linalg.cholesky(self.covariances_[k])
            z = np.linalg.solve(L, (X - self.means_[k]).T)          # Mahalanobis via Cholesky
            out[:, k] = -0.5 * (d * np.log(2 * np.pi) + (z**2).sum(0)) - np.log(np.diag(L)).sum()
        return out

    def _estep(self, X):
        lw = self._log_gauss(X) + np.log(self.weights_)
        m = lw.max(1, keepdims=True)
        log_norm = m + np.log(np.exp(lw - m).sum(1, keepdims=True))
        return np.exp(lw - log_norm), float(log_norm.mean())

    def fit(self, X, y=None):
        X = as_2d(X)
        n, d = X.shape
        init = KMeans(self.n_components, n_init=1, random_state=self.random_state).fit(X)
        R = np.eye(self.n_components)[init.labels_]
        self.lower_bound_history_ = []
        prev = -np.inf
        for self.n_iter_ in range(1, self.max_iter + 1):
            Nk = R.sum(0) + 1e-12
            self.weights_ = Nk / n
            self.means_ = (R.T @ X) / Nk[:, None]
            self.covariances_ = np.array([((R[:, k, None] * (X - self.means_[k])).T @ (X - self.means_[k])) / Nk[k]
                                          + self.reg_covar * np.eye(d) for k in range(self.n_components)])
            R, ll = self._estep(X)
            self.lower_bound_history_.append(ll)
            if abs(ll - prev) < self.tol:
                break
            prev = ll
        self.lower_bound_ = ll
        return self

    def predict_proba(self, X):
        return self._estep(as_2d(X))[0]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), 1)

    def score(self, X):
        return self._estep(as_2d(X))[1]

    def bic(self, X):
        X = as_2d(X)
        d = X.shape[1]
        params = self.n_components * (d + d * (d + 1) / 2) + self.n_components - 1
        return -2 * self.score(X) * len(X) + params * np.log(len(X))
