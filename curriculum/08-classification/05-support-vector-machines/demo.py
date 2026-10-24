"""SVMs from scratch: margin and support vectors, C and γ, kernels, and hinge vs logistic loss."""

import numpy as np

from mlscratch.datasets import make_blobs, make_circles, make_moons
from mlscratch.model_selection import cross_val_score
from mlscratch.preprocessing import StandardScaler
from mlscratch.svm import SVC, LinearSVC
from mlscratch.viz import decision_boundary, plt, savefig

pyplot = plt()

# linear SVM: the margin is 2/||w||
X, y = make_blobs(120, centers=[[-1.5, -1.5], [1.5, 1.5]], cluster_std=0.8, random_state=0)
for C in (0.01, 1, 100):
    m = LinearSVC(C=C, lr=0.01, n_iter=5000).fit(X, y)
    t = np.where(y == 1, 1, -1)
    margins = t * m.decision_function(X)
    print(f"linear SVM C={C:<5} margin width 2/||w|| = {2 / np.linalg.norm(m.coef_):.2f}   points inside margin: {int(np.sum(margins < 1))}")

# kernels on non-linear data
Xc, yc = make_circles(200, noise=0.08, factor=0.4, random_state=0)
Xc = StandardScaler().fit_transform(Xc)
for kernel, kw in [("linear", {}), ("poly", {"degree": 2, "gamma": 1.0}), ("rbf", {"gamma": 1.0})]:
    acc = cross_val_score(SVC(C=1.0, kernel=kernel, **kw), Xc, yc, cv=4).mean()
    print(f"circles, {kernel:<6} kernel: CV accuracy {acc:.3f}")

# C and γ for the RBF kernel
Xm, ym = make_moons(200, noise=0.25, random_state=1)
Xm = StandardScaler().fit_transform(Xm)
fig, axes = pyplot.subplots(2, 3, figsize=(15, 9))
for ax, (C, g) in zip(axes.ravel(), [(0.1, 1), (1, 1), (100, 1), (1, 0.1), (1, 10), (1, 100)]):
    m = SVC(C=C, gamma=g).fit(Xm, ym)
    decision_boundary(ax, m, Xm, ym)
    sv = Xm[m.support_]
    ax.scatter(sv[:, 0], sv[:, 1], s=80, facecolors="none", edgecolors="k", lw=0.8)
    acc = cross_val_score(SVC(C=C, gamma=g), Xm, ym, cv=4).mean()
    ax.set_title(f"C={C}, γ={g}: {len(m.support_)} support vectors, CV acc {acc:.2f}", fontsize=9)
    print(f"RBF C={C:<4} γ={g:<4} support vectors {len(m.support_):>3}   CV accuracy {acc:.3f}")
savefig(fig, "svm_c_gamma")

# KKT check on one model: α=0 points must lie outside the margin
m = SVC(C=1, gamma=1).fit(Xm, ym)
t = np.where(ym == 1, 1, -1)
f = m.decision_function(Xm)
non_sv = np.setdiff1d(np.arange(len(ym)), m.support_)
print(f"\nKKT: min y·f(x) over non-support vectors = {np.min(t[non_sv] * f[non_sv]):.3f} (should be ≥ ~1)")

mgrid = np.linspace(-2, 3, 300)
fig, ax = pyplot.subplots(figsize=(6, 4))
ax.plot(mgrid, np.maximum(0, 1 - mgrid), label="hinge  max(0, 1-m)")
ax.plot(mgrid, np.log1p(np.exp(-mgrid)) / np.log(2), label="logistic (scaled)")
ax.plot(mgrid, (mgrid < 0).astype(float), label="0-1 loss")
ax.set(xlabel="margin m = y·f(x)", ylabel="loss", title="surrogate losses for classification")
ax.legend()
savefig(fig, "losses")
