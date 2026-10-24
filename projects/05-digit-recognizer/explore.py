"""Step 2 - look at the pixels: class balance, the 'average' digit, and how much of the 784-D space PCA needs."""

import numpy as np

from common import load, show_digits
from mlscratch.decomposition import PCA
from mlscratch.viz import plt, savefig

xtr, ytr, xval, yval, xte, yte = load()
print(f"train {xtr.shape} · validation {xval.shape} · test {xte.shape}")
print("class balance (train):", np.round(np.bincount(ytr) / len(ytr), 3).tolist())
print(f"pixel values in [{xtr.min():.0f}, {xtr.max():.0f}]; {np.mean(xtr.std(0) == 0):.0%} of pixels are constant (the border) "
      "and carry no information")

pca = PCA().fit(xtr[:10000])
cum = np.cumsum(pca.explained_variance_ratio_)
for target in (0.80, 0.90, 0.95, 0.99):
    print(f"  {target:.0%} of the variance needs {int(np.searchsorted(cum, target)) + 1:>3} of 784 components")

P = plt()
fig, axes = P.subplots(3, 10, figsize=(13, 4.4))
first = [np.flatnonzero(ytr == d)[:2] for d in range(10)]
show_digits(axes[:2].T, xtr[np.concatenate(first)], [str(ytr[i]) for i in np.concatenate(first)])
show_digits(axes[2], [xtr[ytr == d].mean(0) for d in range(10)], [f"mean {d}" for d in range(10)])
savefig(fig, "01_samples_and_means")

fig, axes = P.subplots(1, 2, figsize=(13, 4.4))
axes[0].plot(np.arange(1, len(cum) + 1), cum)
axes[0].set(title="PCA cumulative explained variance", xlabel="components", ylabel="fraction", xlim=(0, 300))
Z = pca.transform(xtr[:3000])[:, :2]
sc = axes[1].scatter(Z[:, 0], Z[:, 1], c=ytr[:3000], cmap="tab10", s=4)
fig.colorbar(sc, ax=axes[1])
axes[1].set(title="First two principal components (3,000 digits)")
savefig(fig, "02_pca")

fig, axes = P.subplots(4, 6, figsize=(9, 6.4))
for row, k in enumerate((5, 20, 50, 150)):
    p = PCA(k).fit(xtr[:10000])
    rec = p.inverse_transform(p.transform(xte[:6]))
    show_digits(axes[row], rec, [f"k={k}"] * 6)
savefig(fig, "03_pca_reconstruction")
