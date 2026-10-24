"""Polynomial basis expansion by OLS, degree selection by CV, and kNN / tree regression for comparison."""

import numpy as np

from mlscratch.linear_model import LinearRegression, Ridge
from mlscratch.metrics import root_mean_squared_error
from mlscratch.model_selection import KFold
from mlscratch.neighbors import KNeighborsRegressor
from mlscratch.preprocessing import PolynomialFeatures
from mlscratch.tree import DecisionTreeRegressor
from mlscratch.viz import plt, savefig

rng = np.random.default_rng(3)
x = np.sort(rng.uniform(-1, 1, 60))
f = lambda t: 0.5 * t + np.sin(3 * t)  # noqa: E731
y = f(x) + rng.normal(0, 0.15, len(x))


def cv_rmse(degree):
    errs = []
    for tr, va in KFold(5, random_state=0).split(x):
        pf = PolynomialFeatures(degree).fit(x[tr, None])
        m = LinearRegression().fit(pf.transform(x[tr, None]), y[tr])
        errs.append(root_mean_squared_error(y[va], m.predict(pf.transform(x[va, None]))))
    return np.mean(errs)


scores = {d: cv_rmse(d) for d in range(1, 16)}
best = min(scores, key=scores.get)
print("5-fold CV RMSE by degree:", {d: round(float(s), 3) for d, s in scores.items()})
print(f"→ best degree {best}")

raw = np.vander(x * 1000, 16, increasing=True)
print(f"\ncondition number of Φᵀ Φ, degree 15: x in [-1000, 1000] → {np.linalg.cond(raw.T @ raw):.1e};  x in [-1, 1] → "
      f"{np.linalg.cond(np.vander(x, 16, increasing=True).T @ np.vander(x, 16, increasing=True)):.1e}")

grid = np.linspace(-1.4, 1.4, 400)
models = {
    f"polynomial deg {best}": (PolynomialFeatures(best), LinearRegression()),
    "polynomial deg 15": (PolynomialFeatures(15), Ridge(1e-9)),
    "kNN (k=5)": (None, KNeighborsRegressor(5)),
    "tree (depth 4)": (None, DecisionTreeRegressor(max_depth=4)),
}
pyplot = plt()
fig, ax = pyplot.subplots(figsize=(9, 5))
ax.scatter(x, y, color="k", s=12, label="training data")
ax.plot(grid, f(grid), "k--", lw=1, label="true function")
for name, (pf, m) in models.items():
    Xtr = pf.fit_transform(x[:, None]) if pf else x[:, None]
    Xg = pf.transform(grid[:, None]) if pf else grid[:, None]
    m.fit(Xtr, y)
    pred = m.predict(Xg)
    out = (grid < -1) | (grid > 1)
    print(f"{name:<20} extrapolation error beyond [-1, 1]: RMSE {root_mean_squared_error(f(grid[out]), pred[out]):8.2f}")
    ax.plot(grid, np.clip(pred, -3, 3), label=name)
ax.axvspan(-1.4, -1, color="grey", alpha=0.1)
ax.axvspan(1, 1.4, color="grey", alpha=0.1)
ax.set_ylim(-3, 3)
ax.set_title("shaded = outside the training range (extrapolation)")
ax.legend(fontsize=8)
savefig(fig, "nonlinear_regression")
