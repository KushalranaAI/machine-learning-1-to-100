"""Forward selection, RFE, the Lasso path and permutation importance on data with known informative features."""

import numpy as np

from mlscratch.datasets import make_regression
from mlscratch.ensemble import RandomForestRegressor
from mlscratch.feature_selection import RFE, SequentialFeatureSelector, permutation_importance
from mlscratch.linear_model import Lasso, LinearRegression
from mlscratch.model_selection import train_test_split
from mlscratch.preprocessing import StandardScaler
from mlscratch.viz import plt, savefig

rng = np.random.default_rng(0)
coef = np.array([8.0, -5.0, 3.0, 1.5, 0, 0, 0, 0, 0, 0])            # only 4 of 10 features matter
X, y, _ = make_regression(400, n_features=10, noise=3.0, coef=coef, random_state=0)
X = StandardScaler().fit_transform(X)
names = [f"x{j}" for j in range(10)]

sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=4, cv=5).fit(X, y)
print("forward selection order:", [(names[f], round(float(score), 3)) for f, score in sfs.history_])
rfe = RFE(LinearRegression(), n_features_to_select=4).fit(X, y)
print("RFE keeps:", [names[j] for j in np.flatnonzero(rfe.support_)], " ranking:", rfe.ranking_.tolist())

alphas = np.logspace(-2, 1, 40)
path = np.array([Lasso(a).fit(X, y).coef_ for a in alphas])
for a in (0.1, 1.0, 3.0, 6.0):
    nz = np.flatnonzero(np.abs(Lasso(a).fit(X, y).coef_) > 1e-10)
    print(f"Lasso α={a:<4} non-zero weights: {[names[j] for j in nz]}")

pyplot = plt()
fig, ax = pyplot.subplots(figsize=(7, 4))
for j in range(10):
    ax.plot(alphas, path[:, j], label=names[j])
ax.set_xscale("log")
ax.set_xlabel("α (stronger L1 penalty →)")
ax.set_ylabel("coefficient")
ax.set_title("Lasso path: features drop to exactly 0 one by one")
ax.legend(ncol=2, fontsize=7)
savefig(fig, "lasso_path")

Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, random_state=0)
rf = RandomForestRegressor(60, max_features=0.5, random_state=0).fit(Xtr, ytr)
mean, std = permutation_importance(rf, Xva, yva, n_repeats=5)
print("\nrandom forest impurity importance  :", np.round(rf.feature_importances_, 3))
print("permutation importance (R² drop)   :", np.round(mean, 3))
