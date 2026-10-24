"""Step 2 - how many segments? Elbow, silhouette, BIC and a dendrogram should roughly agree before we commit."""

import numpy as np

from common import FEATURES, load
from mlscratch.cluster import AgglomerativeClustering, GaussianMixture, KMeans
from mlscratch.metrics import silhouette_score
from mlscratch.preprocessing import StandardScaler
from mlscratch.viz import plt, savefig

df = load()
print(df.groupby("gender")[FEATURES].mean().round(1).to_string(), "\n")
print("correlations:\n", df[FEATURES].corr().round(2).to_string(), "\n")

# Two feature sets: the classic 2-D (income, score) view and all three numeric features.
views = {"income+score": ["income", "score"], "age+income+score": FEATURES}
results = {}
for view, cols in views.items():
    X = StandardScaler().fit_transform(df[cols].to_numpy(float))   # without scaling, income (15-137) would dominate age
    ks = range(2, 11)
    inertia = [KMeans(1, random_state=0).fit(X).inertia_] + [KMeans(k, n_init=10, random_state=0).fit(X).inertia_ for k in ks]
    sil = [silhouette_score(X, KMeans(k, n_init=10, random_state=0).fit(X).labels_) for k in ks]
    bic = [GaussianMixture(k, random_state=0).fit(X).bic(X) for k in ks]
    results[view] = (inertia, sil, bic)
    best_sil = list(ks)[int(np.argmax(sil))]
    best_bic = list(ks)[int(np.argmin(bic))]
    print(f"[{view}]")
    print("  k   inertia  silhouette        BIC")
    for i, k in enumerate(ks):
        print(f"  {k:>2} {inertia[i + 1]:>9.1f} {sil[i]:>11.3f} {bic[i]:>10.1f}")
    print(f"  → silhouette peaks at k = {best_sil}; GMM BIC is lowest at k = {best_bic}\n")

# Ward's dendrogram: the merge distance jumps when two genuinely different groups are forced together.
X2 = StandardScaler().fit_transform(df[["income", "score"]].to_numpy(float))
agg = AgglomerativeClustering(1, linkage="ward").fit(X2)
last = np.asarray(agg.distances_)[-10:][::-1]
print("Ward merge heights for the last 10 merges (big gap = natural number of clusters):")
for k, h in enumerate(last, start=1):
    print(f"  going from {k + 1:>2} to {k:>2} clusters costs {h:.2f}")
ratio = last[:-1] / last[1:]          # ratio[i] = cost(i+2 → i+1) / cost(i+3 → i+2)
k_star = int(np.argmax(ratio)) + 2
print(f"  the biggest relative jump ({ratio.max():.1f}×) comes when merging below {k_star} clusters → cut the tree at k = {k_star}")

P = plt()
fig, axes = P.subplots(1, 3, figsize=(17, 4.3))
for view, (inertia, sil, bic) in results.items():
    axes[0].plot(range(1, 11), inertia, "o-", label=view)
    axes[1].plot(range(2, 11), sil, "o-", label=view)
    axes[2].plot(range(2, 11), bic, "o-", label=view)
axes[0].set(title="Elbow: k-means inertia", xlabel="k")
axes[1].set(title="Silhouette (higher is better)", xlabel="k")
axes[2].set(title="Gaussian mixture BIC (lower is better)", xlabel="k")
for ax in axes:
    ax.legend()
savefig(fig, "01_choose_k")

fig, ax = P.subplots(figsize=(7, 4))
ax.bar(range(2, 12), last, color="#4c78a8")
ax.set(title="Ward merge height when going from k to k-1 clusters", xlabel="k (before the merge)")
savefig(fig, "02_ward_heights")
