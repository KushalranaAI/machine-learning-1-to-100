"""Gradient boosting from scratch: shrinkage vs rounds (early stopping), L1 vs L2 loss with outliers, classification."""

import numpy as np

from mlscratch.datasets import make_classification
from mlscratch.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from mlscratch.metrics import mean_absolute_error, roc_auc_score
from mlscratch.model_selection import train_test_split
from mlscratch.viz import plt, savefig

rng = np.random.default_rng(0)
n = 800
X = rng.uniform(-3, 3, (n, 4))
y = np.sin(X[:, 0]) * 2 + X[:, 1] ** 2 * 0.5 + X[:, 2] * X[:, 3] + rng.normal(0, 0.3, n)
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, random_state=0)

pyplot = plt()
fig, ax = pyplot.subplots(figsize=(8, 4.5))
for lr in (1.0, 0.3, 0.1, 0.05):
    gb = GradientBoostingRegressor(n_estimators=1200, learning_rate=lr, max_depth=3).fit(Xtr, ytr)
    val = [np.mean((yva - p) ** 2) for p in gb.staged_predict(Xva)]
    best = int(np.argmin(val)) + 1
    print(f"η={lr:<5} best validation MSE {min(val):.3f} at round {best:>3}   (after 1200 rounds {val[-1]:.3f})")
    ax.plot(val, label=f"η={lr}")
rf = RandomForestRegressor(200, random_state=0).fit(Xtr, ytr)
print(f"random forest (200 trees) validation MSE {np.mean((yva - rf.predict(Xva)) ** 2):.3f}")
ax.set(yscale="log", xlabel="boosting round", ylabel="validation MSE", title="large η overfits early; small η needs more rounds")
ax.legend()
savefig(fig, "gb_learning_rate")

# Robustness: absolute loss vs squared loss when 5% of training targets are corrupted
y_bad = ytr.copy()
idx = rng.choice(len(y_bad), int(0.05 * len(y_bad)), replace=False)
y_bad[idx] += rng.choice([-1, 1], len(idx)) * 30
for loss in ("squared", "absolute"):
    gb = GradientBoostingRegressor(200, 0.1, 3, loss=loss).fit(Xtr, y_bad)
    print(f"with 5% corrupted targets, loss={loss:<8} validation MAE {mean_absolute_error(yva, gb.predict(Xva)):.3f}")

Xc, yc = make_classification(1500, 10, n_informative=5, class_sep=0.6, random_state=0)
Xctr, Xcte, yctr, ycte = train_test_split(Xc, yc, test_size=0.3, random_state=0)
for sub in (1.0, 0.5):
    gbc = GradientBoostingClassifier(150, 0.1, 3, subsample=sub, random_state=0).fit(Xctr, yctr)
    print(f"classifier subsample={sub}: test accuracy {gbc.score(Xcte, ycte):.3f}   AUC {roc_auc_score(ycte, gbc.predict_proba(Xcte)[:, 1]):.3f}")
