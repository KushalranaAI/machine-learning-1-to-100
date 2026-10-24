"""DBSCAN from scratch: shapes k-means can't handle, the k-distance plot for ε, and noise detection."""

import numpy as np

from mlscratch.cluster import DBSCAN, KMeans
from mlscratch.datasets import make_circles, make_moons
from mlscratch.neighbors import pairwise_distances
from mlscratch.viz import plt, savefig

rng = np.random.default_rng(0)
Xm, ym = make_moons(300, noise=0.06, random_state=0)
noise = rng.uniform(-1.5, 2.5, (20, 2))
X = np.vstack([Xm, noise])

D = pairwise_distances(X, X)
k = 5
kdist = np.sort(np.sort(D, axis=1)[:, k])
# "kneedle": the knee is the point farthest below the straight line joining the first and last points
xs = np.linspace(0, 1, len(kdist))
ys = (kdist - kdist.min()) / (kdist.max() - kdist.min())
knee = kdist[int(np.argmax(xs - ys))]
print(f"k-distance plot suggests ε ≈ {knee:.3f} (the knee where the curve starts to shoot up)")

for eps in (0.05, 0.15, 0.3, 0.6):
    labels = DBSCAN(eps, k).fit_predict(X)
    print(f"ε={eps:<5} clusters {labels.max() + 1:>2}   noise points {int(np.sum(labels == -1)):>3}")

pyplot = plt()
fig, axes = pyplot.subplots(1, 4, figsize=(18, 4))
axes[0].plot(kdist)
axes[0].axhline(knee, ls="--", color="#e8590c")
axes[0].set(title=f"sorted {k}-distance plot", xlabel="points", ylabel=f"distance to {k}th neighbour")
for ax, (Xi, title) in zip(axes[1:3], [(X, "moons + noise"), (make_circles(300, noise=0.04, factor=0.5, random_state=1)[0], "circles")]):
    labels = DBSCAN(0.15 if title != "circles" else 0.2, k).fit_predict(Xi)
    ax.scatter(Xi[labels >= 0, 0], Xi[labels >= 0, 1], c=labels[labels >= 0], cmap="tab10", s=8)
    ax.scatter(Xi[labels < 0, 0], Xi[labels < 0, 1], c="k", marker="x", s=20, label="noise")
    ax.set_title(f"DBSCAN on {title}")
    ax.legend()
axes[3].scatter(X[:, 0], X[:, 1], c=KMeans(2, random_state=0).fit_predict(X), cmap="coolwarm", s=8)
axes[3].set_title("k-means on the same data")
savefig(fig, "dbscan")
