"""Filter-based selection: variance, correlation, ANOVA F, chi-square and mutual information - and the XOR blind spot."""

import numpy as np

from mlscratch.datasets import make_classification
from mlscratch.feature_selection import SelectKBest, VarianceThreshold, chi2, correlation_filter, f_classif, mutual_info_classif
from mlscratch.linear_model import LogisticRegression
from mlscratch.model_selection import cross_val_score

rng = np.random.default_rng(0)
X, y = make_classification(600, n_features=10, n_informative=3, random_state=0)     # features 0-2 informative
X = np.c_[X, X[:, 0] * 0.98 + rng.normal(0, 0.05, 600), np.full(600, 7.0)]          # 10: near-copy of 0, 11: constant
names = [f"inf{i}" for i in range(3)] + [f"noise{i}" for i in range(3, 10)] + ["copy_of_inf0", "constant"]

vt = VarianceThreshold(1e-8).fit(X)
print("variance threshold drops:", [n for n, keep in zip(names, vt.support_) if not keep])
keep = correlation_filter(X[:, vt.support_], threshold=0.9)
kept = [n for n, s in zip([n for n, k in zip(names, vt.support_) if k], keep) if s]
print("correlation filter drops:", sorted(set(names) - set(kept) - {"constant"}))

Xk = X[:, :10]
F = f_classif(Xk, y)
MI = mutual_info_classif(Xk, y)
C = chi2(Xk - Xk.min(0), y)                       # χ² needs non-negative inputs
print(f"\n{'feature':<9}{'ANOVA F':>10}{'χ²':>10}{'MI':>8}")
for j in np.argsort(-F):
    print(f"{names[j]:<9}{F[j]:>10.1f}{C[j]:>10.1f}{MI[j]:>8.3f}")

sel = SelectKBest(f_classif, k=3).fit(Xk, y)
print("\nSelectKBest(k=3) keeps:", [names[j] for j in np.flatnonzero(sel.support_)])
for label, Z in [("all 10 features", Xk), ("top-3 by F", sel.transform(Xk))]:
    print(f"CV accuracy with {label:<16}: {cross_val_score(LogisticRegression(n_iter=500), Z, y, cv=5).mean():.3f}")

# The XOR blind spot: each feature alone looks useless
a, b = rng.normal(size=600), rng.normal(size=600)
y_xor = ((a > 0) ^ (b > 0)).astype(int)
Xx = np.c_[a, b, rng.normal(size=600)]
print("\nXOR data - F-scores:", np.round(f_classif(Xx, y_xor), 2), " MI:", np.round(mutual_info_classif(Xx, y_xor), 3))
print("→ a and b look as useless as the noise column, yet together they determine y exactly.")
