"""PCA from scratch on 8×8 digit images: explained variance, 2-D projection and reconstructions."""

import numpy as np

from mlscratch.decomposition import PCA
from mlscratch.viz import plt, savefig

try:
    from sklearn.datasets import load_digits   # only used to LOAD the small bundled digits dataset
    digits = load_digits()
    X, y = digits.data, digits.target
except ImportError:
    raise SystemExit("pip install scikit-learn (used here only to load the 8×8 digits dataset)")

print("data:", X.shape, "(1797 images × 64 pixels)")
pca = PCA().fit(X)
cum = np.cumsum(pca.explained_variance_ratio_)
for k in (2, 10, 20, 30):
    print(f"  {k:>2} components keep {cum[k - 1]:.1%} of the variance")
print("components needed for 95%:", PCA(0.95).fit(X).n_components_)

# Check the math: eigenvalues of the covariance = explained variance from the SVD
Xc = X - X.mean(0)
eig = np.sort(np.linalg.eigvalsh(Xc.T @ Xc / (len(X) - 1)))[::-1]
print("eigen-decomposition and SVD agree:", np.allclose(eig[:10], pca.explained_variance_[:10]))

pyplot = plt()
fig, axes = pyplot.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(np.arange(1, 65), cum, marker=".")
axes[0].axhline(0.95, ls="--", color="grey")
axes[0].set_xlabel("number of components")
axes[0].set_ylabel("cumulative explained variance")
axes[0].set_title("scree / cumulative variance")
Z = PCA(2).fit_transform(X)
sc = axes[1].scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=8)
axes[1].set_title("64-D digits projected onto PC1/PC2")
fig.colorbar(sc, ax=axes[1])
savefig(fig, "pca_digits")

fig, axes = pyplot.subplots(4, 8, figsize=(10, 5.2))
for row, k in enumerate([2, 8, 20, 64]):
    model = PCA(k).fit(X)
    recon = model.inverse_transform(model.transform(X[:8]))
    err = np.mean((model.inverse_transform(model.transform(X)) - X) ** 2)
    for col in range(8):
        axes[row, col].imshow(recon[col].reshape(8, 8), cmap="gray_r")
        axes[row, col].axis("off")
    axes[row, 0].set_title(f"k={k}  MSE={err:.2f}", fontsize=8, loc="left")
savefig(fig, "pca_reconstruction")

top = np.argsort(-np.abs(pca.components_[0]))[:5]
print("PC1's largest pixel loadings (pixel index: weight):", {int(j): round(float(pca.components_[0, j]), 3) for j in top})
