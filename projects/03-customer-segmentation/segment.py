"""Step 3 - build the segments with four algorithms, check that they agree and are stable, then profile them.

Output: segments.csv (one row per customer with its segment) plus a profile table marketing can act on.
"""

from pathlib import Path

import numpy as np

from common import FEATURES, adjusted_rand_index, load
from mlscratch.cluster import DBSCAN, AgglomerativeClustering, GaussianMixture, KMeans
from mlscratch.decomposition import PCA
from mlscratch.metrics import silhouette_score
from mlscratch.neighbors import pairwise_distances
from mlscratch.preprocessing import StandardScaler
from mlscratch.viz import plt, savefig

K = 5  # from choose_k.py: silhouette, BIC and the Ward dendrogram all point to 5 on (income, score)
df = load()
X = StandardScaler().fit_transform(df[["income", "score"]].to_numpy(float))

km = KMeans(K, n_init=20, random_state=0).fit(X)
labels = {
    "k-means": km.labels_,
    "Ward hierarchical": AgglomerativeClustering(K, linkage="ward").fit_predict(X),
    "Gaussian mixture": GaussianMixture(K, random_state=0).fit(X).predict(X),
}
# DBSCAN: eps from the knee of the sorted 4th-nearest-neighbour distance (min_samples = 5 counts the point itself)
D = np.sort(pairwise_distances(X, X), axis=1)[:, 4]
eps = float(np.quantile(D, 0.9))
db = DBSCAN(eps=eps, min_samples=5).fit(X)
labels["DBSCAN"] = db.labels_
print(f"DBSCAN with eps={eps:.2f}: {len(set(db.labels_) - {-1})} clusters, {np.sum(db.labels_ == -1)} points marked as noise")
print("  (the dense middle group and the sparse corners need different eps values - one global density threshold\n"
      "   cannot separate them. DBSCAN is the wrong tool here; it shines on arbitrary shapes with similar density.)\n")

print("Agreement with k-means (adjusted Rand index, 1 = identical):")
for name, lab in labels.items():
    sil = silhouette_score(X, lab) if len(set(lab)) > 1 else float("nan")
    print(f"  {name:<18} ARI {adjusted_rand_index(labels['k-means'], lab):.3f}   silhouette {sil:.3f}")

# Stability: recluster bootstrap samples; a real segment structure survives resampling.
rng = np.random.default_rng(0)
aris = []
for _ in range(30):
    idx = rng.choice(len(X), len(X), replace=True)
    boot = KMeans(K, n_init=5, random_state=int(rng.integers(1_000_000))).fit(X[idx])
    aris.append(adjusted_rand_index(km.labels_, boot.predict(X)))
print(f"\nBootstrap stability of k-means (30 resamples): ARI {np.mean(aris):.3f} ± {np.std(aris):.3f}")


def describe(row, inc_mid, score_mid):
    inc = "high income" if row.income > inc_mid + 15 else "low income" if row.income < inc_mid - 15 else "mid income"
    spend = "high spenders" if row.score > score_mid + 15 else "low spenders" if row.score < score_mid - 15 else "average spenders"
    return f"{inc}, {spend}"


ACTIONS = {
    "high income, high spenders": "VIP programme, early access, premium brands",
    "high income, low spenders": "biggest upside: find out why - personalised offers, quality messaging",
    "low income, high spenders": "loyalty points, buy-now-pay-later; watch credit risk",
    "low income, low spenders": "value bundles, discount days",
    "mid income, average spenders": "mainstream campaigns, cross-sell",
}
df["segment"] = km.labels_
profile = df.groupby("segment").agg(size=("age", "size"), age=("age", "mean"), income=("income", "mean"),
                                    score=("score", "mean"), female=("gender", lambda g: (g == "Female").mean()))
profile["description"] = [describe(r, df.income.median(), df.score.median()) for r in profile.itertuples()]
profile["action"] = profile.description.map(ACTIONS).fillna("-")
profile = profile.sort_values("income")
print("\nSegment profiles (k-means on income + score):")
print(profile.round(1).to_string())

# Does age separate anything the 2-D view missed? Cluster all three features and project with PCA to look.
X3 = StandardScaler().fit_transform(df[FEATURES].to_numpy(float))
km3 = KMeans(6, n_init=20, random_state=0).fit(X3)
pca = PCA(2).fit(X3)
Z = pca.transform(X3)
print(f"\n3-feature k-means (k=6): PCA keeps {pca.explained_variance_ratio_.sum():.0%} of the variance in 2-D")
df["segment_3d"] = km3.labels_
print(df.groupby("segment_3d")[FEATURES].mean().round(1).sort_values("income").to_string())
print("Adding age splits the big 'mid income, average spenders' group into a younger and an older half -\n"
      "same wallet, very different life stage, so different channels and products.")

out = Path(__file__).resolve().parent / "outputs" / "segments.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"\nwrote outputs/{out.name}")

P = plt()
fig, axes = P.subplots(2, 2, figsize=(13, 10))
raw = df[["income", "score"]].to_numpy()
for ax, (name, lab) in zip(axes.ravel(), labels.items()):
    ax.scatter(raw[:, 0], raw[:, 1], c=lab, cmap="tab10", s=25, edgecolor="k", linewidth=0.3)
    if name == "DBSCAN":
        ax.scatter(raw[lab == -1, 0], raw[lab == -1, 1], marker="x", c="k", s=30, label="noise")
        ax.legend()
    ax.set(title=f"{name} (ARI vs k-means {adjusted_rand_index(labels['k-means'], lab):.2f})",
           xlabel="annual income (k$)", ylabel="spending score")
savefig(fig, "03_four_algorithms")

fig, axes = P.subplots(1, 2, figsize=(13, 4.6))
centers = km.cluster_centers_ * raw.std(0) + raw.mean(0)
axes[0].scatter(raw[:, 0], raw[:, 1], c=km.labels_, cmap="tab10", s=25)
for k, (cx, cy) in enumerate(centers):
    axes[0].annotate(profile.loc[k, "description"], (cx, cy), ha="center", fontsize=8,
                     bbox=dict(boxstyle="round", fc="white", alpha=0.8))
axes[0].set(title="Named segments", xlabel="annual income (k$)", ylabel="spending score")
axes[1].scatter(Z[:, 0], Z[:, 1], c=km3.labels_, cmap="tab10", s=25)
axes[1].set(title="age + income + score, k = 6, projected on 2 principal components", xlabel="PC1", ylabel="PC2")
savefig(fig, "04_segments")
