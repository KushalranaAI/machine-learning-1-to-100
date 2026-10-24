"""kNN from scratch: choosing k by CV, distance weighting, and the curse of dimensionality."""

import numpy as np

from mlscratch.datasets import make_moons
from mlscratch.model_selection import cross_val_score
from mlscratch.neighbors import KNeighborsClassifier, pairwise_distances
from mlscratch.viz import decision_boundary, plt, savefig

X, y = make_moons(400, noise=0.3, random_state=0)
print("5-fold CV accuracy by k:")
for k in (1, 3, 5, 9, 15, 25, 51, 101, 201):
    u = cross_val_score(KNeighborsClassifier(k), X, y, cv=5).mean()
    w = cross_val_score(KNeighborsClassifier(k, weights="distance"), X, y, cv=5).mean()
    print(f"  k={k:<4} uniform {u:.3f}   distance-weighted {w:.3f}")

pyplot = plt()
fig, axes = pyplot.subplots(1, 3, figsize=(15, 4.3))
for ax, k in zip(axes, (1, 15, 201)):
    m = KNeighborsClassifier(k).fit(X, y)
    decision_boundary(ax, m, X, y)
    ax.set_title(f"k = {k}  (train acc {m.score(X, y):.2f})")
savefig(fig, "knn_k")

try:
    from sklearn.datasets import load_digits
    d = load_digits()
    print(f"\ndigits, 5-fold CV accuracy with k=3: {cross_val_score(KNeighborsClassifier(3), d.data, d.target, cv=5).mean():.3f}")
except ImportError:
    pass

print("\ncurse of dimensionality - 500 uniform points, ratio of farthest to nearest neighbour distance:")
rng = np.random.default_rng(0)
for dim in (2, 10, 100, 1000):
    P = rng.random((500, dim))
    D = pairwise_distances(P[:1], P[1:])[0]
    print(f"  d={dim:<5} nearest {D.min():6.2f}  farthest {D.max():6.2f}  ratio {D.max() / D.min():6.2f}")
print("→ as d grows the ratio approaches 1: every point is 'about as far' as every other, so 'nearest' means little.")
