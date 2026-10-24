"""Agglomerative clustering from scratch: linkages compared, and a dendrogram from our merge history."""

import numpy as np

from mlscratch.cluster import AgglomerativeClustering
from mlscratch.datasets import make_blobs, make_moons
from mlscratch.metrics import silhouette_score
from mlscratch.viz import plt, savefig


def purity(labels, truth):
    return sum(np.bincount(truth[labels == c]).max() for c in np.unique(labels)) / len(truth)


Xm, ym = make_moons(200, noise=0.05, random_state=0)
Xb, yb = make_blobs(200, centers=3, cluster_std=1.5, random_state=1)
print(f"{'linkage':<10}{'moons purity':>14}{'blobs purity':>14}{'blobs silhouette':>18}")
for link in ("single", "complete", "average", "ward"):
    lm = AgglomerativeClustering(2, link).fit_predict(Xm)
    lb = AgglomerativeClustering(3, link).fit_predict(Xb)
    print(f"{link:<10}{purity(lm, ym):>14.3f}{purity(lb, yb):>14.3f}{silhouette_score(Xb, lb):>18.3f}")

# Dendrogram from our own merge history (children_ + distances_)
X = make_blobs(25, centers=3, cluster_std=0.8, random_state=4)[0]
model = AgglomerativeClustering(1, "ward").fit(X)
n = len(X)
pos = {i: float(i) for i in range(n)}
order = []


def leaves(node):
    if node < n:
        return [node]
    a, b = model.children_[node - n]
    return leaves(a) + leaves(b)


for idx, leaf in enumerate(leaves(2 * n - 2)):
    pos[leaf] = float(idx)
height = {i: 0.0 for i in range(n)}
pyplot = plt()
fig, ax = pyplot.subplots(figsize=(10, 4))
for k, ((a, b), d) in enumerate(zip(model.children_, model.distances_)):
    node = n + k
    xa, xb = pos[a], pos[b]
    ax.plot([xa, xa, xb, xb], [height[a], d, d, height[b]], color="#4263eb")
    pos[node], height[node] = (xa + xb) / 2, d
gaps = np.diff(sorted(model.distances_))
cut = sorted(model.distances_)[int(np.argmax(gaps))] + gaps.max() / 2
ax.axhline(cut, ls="--", color="#e8590c", label=f"cut at the largest gap → {int(np.sum(np.array(model.distances_) > cut)) + 1} clusters")
ax.set(title="dendrogram (Ward linkage), drawn from our merge history", ylabel="merge distance")
ax.set_xticks([])
ax.legend()
savefig(fig, "dendrogram")
print(f"\nlast 4 merge heights: {np.round(model.distances_[-4:], 2).tolist()} → the big jump suggests 3 clusters")
