"""One-hot vs ordinal vs target encoding - and how naive target encoding leaks the label."""

import numpy as np
import pandas as pd

from mlscratch.datasets import DATA_DIR
from mlscratch.linear_model import LogisticRegression
from mlscratch.model_selection import KFold, train_test_split
from mlscratch.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

cust = pd.read_csv(DATA_DIR / "clean_customers.csv").dropna(subset=["plan", "city"])
X = cust[["plan", "city"]].to_numpy(object)

ohe = OneHotEncoder().fit(X)
print("one-hot columns:", ohe.get_feature_names(["plan", "city"]))
print("first row →", ohe.transform(X[:1]).astype(int))
ord_enc = OrdinalEncoder(categories=[["basic", "premium", "enterprise"], sorted(set(X[:, 1]))]).fit(X)
print("ordinal (explicit plan order) first 3 rows →", ord_enc.transform(X[:3]).tolist())
print("unseen city 'Tokyo' one-hot →", ohe.transform(np.array([["basic", "Tokyo"]], dtype=object)).astype(int))

# dummy-variable trap: full one-hot + intercept is rank-deficient
D = np.c_[np.ones(len(X)), OneHotEncoder().fit_transform(X[:, :1])]
print(f"\nrank of [1 | one-hot(plan)] = {np.linalg.matrix_rank(D)} for {D.shape[1]} columns → collinear; drop_first fixes it:",
      np.linalg.matrix_rank(np.c_[np.ones(len(X)), OneHotEncoder(drop_first=True).fit_transform(X[:, :1])]))

# --- target-encoding leakage on a feature that carries NO signal -----------------------------
rng = np.random.default_rng(0)
n = 2000
ids = rng.integers(0, 600, n).astype(str).reshape(-1, 1)     # high-cardinality ID, unrelated to y
y = rng.integers(0, 2, n)
Xtr, Xte, ytr, yte = train_test_split(ids, y, test_size=0.3, random_state=0)

naive = TargetEncoder(smoothing=0).fit(Xtr, ytr)               # uses each row's own label
model = LogisticRegression(n_iter=500).fit(naive.transform(Xtr), ytr)
print(f"\nNAIVE target encoding:  train acc {model.score(naive.transform(Xtr), ytr):.3f}   test acc {model.score(naive.transform(Xte), yte):.3f}")

oof = np.zeros(len(Xtr))                                      # out-of-fold: each row encoded without itself
for fit_idx, enc_idx in KFold(5, random_state=0).split(Xtr):
    oof[enc_idx] = TargetEncoder(smoothing=20).fit(Xtr[fit_idx], ytr[fit_idx]).transform(Xtr[enc_idx]).ravel()
full = TargetEncoder(smoothing=20).fit(Xtr, ytr)
model = LogisticRegression(n_iter=500).fit(oof[:, None], ytr)
print(f"OUT-OF-FOLD encoding:   train acc {model.score(oof[:, None], ytr):.3f}   test acc {model.score(full.transform(Xte), yte):.3f}")
print("→ the feature is pure noise; only the out-of-fold version is honest about it (≈ 0.5 everywhere).")
