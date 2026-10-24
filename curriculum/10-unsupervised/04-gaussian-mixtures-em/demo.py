"""Gaussian mixtures from scratch: EM's monotone likelihood, soft assignments, BIC for K, GMM vs k-means."""

import numpy as np

from mlscratch.cluster import GaussianMixture, KMeans
from mlscratch.viz import plt, savefig

rng = np.random.default_rng(0)
# three elongated, overlapping clusters - hard for k-means
A = rng.multivariate_normal([0, 0], [[4, 3.4], [3.4, 3.2]], 300)
B = rng.multivariate_normal([5, -2], [[0.6, 0], [0, 3]], 200)
C = rng.multivariate_normal([3, 4], [[2.5, -1.5], [-1.5, 1.5]], 250)
X = np.vstack([A, B, C])
truth = np.r_[np.zeros(300), np.ones(200), 2 * np.ones(250)].astype(int)

gmm = GaussianMixture(3, random_state=0).fit(X)
ll = gmm.lower_bound_history_
print(f"EM ran {gmm.n_iter_} iterations; mean log-likelihood {ll[0]:.4f} → {ll[-1]:.4f}; ever decreased? {bool(np.any(np.diff(ll) < -1e-12))}")
print("mixing weights π:", np.round(gmm.weights_, 3), " (true 0.40, 0.27, 0.33)")


def agreement(labels):
    return max(np.mean(np.array([p[l] for l in labels]) == truth) for p in __import__("itertools").permutations(range(3)))


km = KMeans(3, random_state=0).fit(X)
print(f"label agreement with the true components: GMM {agreement(gmm.predict(X)):.3f}   k-means {agreement(km.labels_):.3f}")

R = gmm.predict_proba(X)
unsure = np.sum(R.max(1) < 0.8)
print(f"points with no component above 80% responsibility (genuinely ambiguous): {unsure}")

print("\nBIC by number of components (lower is better):")
for k in range(1, 7):
    print(f"  K={k}: BIC {GaussianMixture(k, random_state=0).fit(X).bic(X):9.1f}")

pyplot = plt()
fig, axes = pyplot.subplots(1, 3, figsize=(16, 4.5))
axes[0].plot(ll, "o-")
axes[0].set(xlabel="EM iteration", ylabel="mean log-likelihood", title="EM never decreases the likelihood")
axes[1].scatter(X[:, 0], X[:, 1], c=R, s=6)
xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 150), np.linspace(X[:, 1].min(), X[:, 1].max(), 150))
dens = np.exp(gmm._log_gauss(np.c_[xx.ravel(), yy.ravel()])) @ gmm.weights_      # p(x) = Σ π_k N(x)
axes[1].contour(xx, yy, dens.reshape(xx.shape), levels=8, colors="k", linewidths=0.6)
axes[1].set_title("GMM: soft assignments (RGB = responsibilities) + density")
axes[2].scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="tab10", s=6)
axes[2].set_title("k-means: straight Voronoi boundaries")
savefig(fig, "gmm")
