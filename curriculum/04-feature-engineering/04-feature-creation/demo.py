"""Feature creation: interactions, polynomials, cyclical dates and a leaking feature that looks amazing."""

import numpy as np
import pandas as pd

from mlscratch.linear_model import LinearRegression, LogisticRegression, Ridge
from mlscratch.model_selection import cross_val_score, train_test_split
from mlscratch.preprocessing import PolynomialFeatures, StandardScaler

rng = np.random.default_rng(1)
n = 1500
df = pd.DataFrame({
    "area": rng.uniform(40, 250, n),
    "quality": rng.integers(1, 11, n),
    "age": rng.uniform(0, 80, n),
    "sold_at": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 730 * 24, n), unit="h"),
})
season = np.cos(2 * np.pi * (df.sold_at.dt.month - 7) / 12)                 # summer premium
df["price"] = 900 * df.area * (0.6 + 0.08 * df.quality) - 600 * df.age + 15_000 * season + rng.normal(0, 15_000, n)


def cv_r2(X):
    return cross_val_score(Ridge(1.0), StandardScaler().fit_transform(X), df["price"].to_numpy(), cv=5).mean()


base = df[["area", "quality", "age"]].to_numpy()
inter = np.c_[base, df.area * df.quality]
month = df.sold_at.dt.month.to_numpy()
cyc = np.c_[inter, np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)]
poly = PolynomialFeatures(2).fit_transform(base)
print(f"CV R² raw features                 : {cv_r2(base):.4f}")
print(f"  + area × quality interaction     : {cv_r2(inter):.4f}")
print(f"  + month as number                : {cv_r2(np.c_[inter, month]):.4f}")
print(f"  + month as sin/cos (cyclical)    : {cv_r2(cyc):.4f}")
print(f"all degree-2 polynomial terms ({poly.shape[1]} cols): {cv_r2(poly):.4f}")

print("\nDecember vs January distance:  numeric", abs(12 - 1), " | sin/cos",
      round(float(np.hypot(np.sin(2 * np.pi * 12 / 12) - np.sin(2 * np.pi / 12), np.cos(2 * np.pi * 12 / 12) - np.cos(2 * np.pi / 12))), 3))

# --- a leak: predicting churn with a column recorded AFTER the churn decision -------------
churn = rng.binomial(1, 0.2, n)
honest = np.c_[rng.normal(size=n) + 0.6 * churn, rng.normal(size=n)]         # weakly informative features
refund_issued = (churn & (rng.random(n) < 0.9)).astype(float)                # only happens after churning!
for name, X in [("honest features", honest), ("+ refund_issued (LEAK)", np.c_[honest, refund_issued])]:
    Xtr, Xte, ytr, yte = train_test_split(X, churn, test_size=0.3, random_state=0)
    acc = LogisticRegression(n_iter=1000).fit(Xtr, ytr).score(Xte, yte)
    print(f"churn model with {name:<24} test accuracy {acc:.3f}")
print("→ the leaky model looks brilliant offline but can't work in production: refunds don't exist yet at prediction time.")
_ = LinearRegression
