"""k-means from scratch: convergence of the objective, random vs k-means++ init, elbow and silhouette."""

import numpy as np

from mlscratch.cluster import KMeans
from mlscratch.datasets import make_blobs, make_moons
from mlscratch.metrics import silhouette_score
from mlscratch.viz import plt, savefig

X, y = make_blobs(600, centers=5, cluster_std=0.8, random_state=7)

m = KMeans(5, n_init=1, init="random", random_state=0).fit(X)
inertias = [float(((X - C[np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)]) ** 2).sum()) for C in m.history_]
print("inertia after each Lloyd iteration (never increases):", np.round(inertias, 1).tolist())

runs = {"random": [], "k-means++": []}
for seed in range(30):
    for init in runs:
        runs[init].append(KMeans(5, init=init, n_init=1, random_state=seed).fit(X).inertia_)
best = min(min(v) for v in runs.values())
for init, vals in runs.items():
    bad = sum(v > best * 1.05 for v in vals)
    print(f"{init:<10} init, 30 single runs: mean inertia {np.mean(vals):8.1f}   runs stuck in a worse local minimum: {bad}")

print(f"\n{'k':>3}{'inertia':>10}{'silhouette':>12}")
ks = range(2, 10)
stats = []
for k in ks:
    km = KMeans(k, random_state=0).fit(X)
    stats.append((km.inertia_, silhouette_score(X, km.labels_)))
    print(f"{k:>3}{stats[-1][0]:>10.1f}{stats[-1][1]:>12.3f}")
print(f"silhouette is highest at k = {list(ks)[int(np.argmax([s for _, s in stats]))]}")

pyplot = plt()
fig, axes = pyplot.subplots(1, 3, figsize=(15, 4.2))
axes[0].plot(list(ks), [i for i, _ in stats], "o-")
axes[0].set(xlabel="k", ylabel="inertia", title="elbow method")
km = KMeans(5, random_state=0).fit(X)
axes[1].scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="tab10", s=6)
axes[1].scatter(*km.cluster_centers_.T, c="k", marker="x", s=120)
axes[1].set_title("k-means, k=5")
Xm, _ = make_moons(400, noise=0.05, random_state=0)
axes[2].scatter(Xm[:, 0], Xm[:, 1], c=KMeans(2, random_state=0).fit_predict(Xm), cmap="coolwarm", s=6)
axes[2].set_title("k-means fails on non-convex shapes")
savefig(fig, "kmeans")
