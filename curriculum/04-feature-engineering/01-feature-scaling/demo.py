"""Scaling in numbers: kNN accuracy, gradient-descent conditioning, and robust vs standard scaling."""

import numpy as np

from mlscratch.datasets import make_classification
from mlscratch.linear_model import LinearRegression
from mlscratch.model_selection import train_test_split
from mlscratch.neighbors import KNeighborsClassifier
from mlscratch.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler, StandardScaler

rng = np.random.default_rng(0)

# 1. kNN with one informative small-scale feature and one noisy large-scale feature
X, y = make_classification(600, n_features=2, n_informative=1, class_sep=1.5, random_state=0)
X[:, 1] *= 1000                                          # e.g. a noise column measured in grams
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
for name, scaler in [("no scaling", None), ("StandardScaler", StandardScaler()), ("MinMaxScaler", MinMaxScaler())]:
    A, B = (Xtr, Xte) if scaler is None else (scaler.fit_transform(Xtr), scaler.transform(Xte))  # fit on TRAIN only
    acc = KNeighborsClassifier(5).fit(A, ytr).score(B, yte)
    print(f"kNN test accuracy with {name:<15}: {acc:.3f}")

# 2. condition number of XᵀX and gradient-descent convergence
Xr = np.c_[rng.normal(50, 10, 300), rng.normal(50_000, 15_000, 300)]
yr = 3 * Xr[:, 0] + 0.002 * Xr[:, 1] + rng.normal(0, 5, 300)
Xs = StandardScaler().fit_transform(Xr)
print(f"\ncondition number of XᵀX: raw {np.linalg.cond(Xr.T @ Xr):.1e}  standardised {np.linalg.cond(Xs.T @ Xs):.1f}")
gd_scaled = LinearRegression(solver="gd", lr=0.1, n_iter=5000).fit(Xs, yr)
gd_raw = LinearRegression(solver="gd", lr=1e-10, n_iter=5000).fit(Xr, yr)
print(f"GD on standardised data: {len(gd_scaled.loss_history_)} iterations to converge, final MSE {gd_scaled.loss_history_[-1]:.2f}")
print(f"GD on raw data (lr must be ~1e-10 to not diverge): MSE after 5000 iterations {gd_raw.loss_history_[-1]:,.1f}")

# 3. outliers: standard vs robust scaling
x = np.r_[rng.normal(50, 5, 200), [500, 800]][:, None]
s, r = StandardScaler().fit_transform(x), RobustScaler().fit_transform(x)
print(f"\nwith 2 huge outliers - spread of the 200 normal points after scaling: standard {s[:200].std():.3f}, robust {r[:200].std():.3f}")

# 4. power transform makes a skewed feature symmetric
skewed = rng.lognormal(0, 1, (1000, 1))
pt = PowerTransformer().fit(skewed)
z = pt.transform(skewed).ravel()
skew = lambda v: np.mean((v - v.mean()) ** 3) / v.std() ** 3  # noqa: E731
print(f"Yeo-Johnson λ = {pt.lambdas_[0]:.2f}; skewness {skew(skewed.ravel()):.2f} → {skew(z):.2f}")
