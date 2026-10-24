"""PCA vs LDA on iris, kernel PCA on circles, and t-SNE on digits - all from scratch."""

import numpy as np

from mlscratch.datasets import load_iris, make_circles
from mlscratch.decomposition import PCA, TSNE, KernelPCA, LinearDiscriminantAnalysis
from mlscratch.linear_model import LogisticRegression
from mlscratch.metrics import silhouette_score
from mlscratch.preprocessing import StandardScaler
from mlscratch.viz import plt, savefig

pyplot = plt()
X, y, names = load_iris()
Xs = StandardScaler().fit_transform(X)
Zp = PCA(2).fit_transform(Xs)
lda = LinearDiscriminantAnalysis(2).fit(Xs, y)
Zl = lda.transform(Xs)
print(f"iris → 2-D   silhouette by species: PCA {silhouette_score(Zp, y):.3f}   LDA {silhouette_score(Zl, y):.3f}")
print(f"LDA as a classifier, training accuracy: {lda.score(Xs, y):.3f}")

Xc, yc = make_circles(400, noise=0.05, factor=0.4, random_state=0)
lin = PCA(2).fit_transform(Xc)
kpca = KernelPCA(3, gamma=2.0).fit_transform(Xc)
acc = lambda Z: LogisticRegression(n_iter=2000).fit(Z, yc).score(Z, yc)  # noqa: E731
print(f"circles, linear classifier on linear PCA (2 comps): {acc(lin):.3f}  - no linear direction separates rings")
for k in range(3):
    print(f"circles, linear classifier on kernel PC{k + 1} alone : {acc(kpca[:, [k]]):.3f}")
print("→ kernel PCs 1-2 form a pair that encodes the ANGLE around the circle; PC3 encodes the RADIUS and separates the rings.")
ker = kpca[:, [2]]

try:
    from sklearn.datasets import load_digits
    d = load_digits()
    idx = np.random.default_rng(0).choice(len(d.data), 500, replace=False)
    Xd, yd = d.data[idx], d.target[idx]
    T = TSNE(perplexity=30, n_iter=600, random_state=0).fit_transform(Xd)
    P = PCA(2).fit_transform(Xd)
    print(f"digits (500 images) silhouette by digit: PCA {silhouette_score(P, yd):.3f}  t-SNE {silhouette_score(T, yd):.3f}")
except ImportError:
    Xd = None

fig, axes = pyplot.subplots(1, 4 if Xd is not None else 3, figsize=(18, 4))
for ax, Z, title in [(axes[0], Zp, "iris: PCA"), (axes[1], Zl, "iris: LDA")]:
    ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="viridis", s=12)
    ax.set_title(title)
axes[2].scatter(ker[:, 0], np.random.default_rng(0).normal(0, 0.01, len(ker)), c=yc, cmap="coolwarm", s=8)
axes[2].set_title("circles → kernel PC3 alone (separable)")
if Xd is not None:
    axes[3].scatter(T[:, 0], T[:, 1], c=yd, cmap="tab10", s=8)
    axes[3].set_title("digits: t-SNE")
savefig(fig, "projections")
