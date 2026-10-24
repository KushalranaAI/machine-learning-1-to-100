"""Linear regression from scratch: normal equation vs gradient descent, coefficient inference, diagnostics."""

import numpy as np
from scipy import stats

from mlscratch.base import add_bias
from mlscratch.linear_model import LinearRegression
from mlscratch.metrics import r2_score, root_mean_squared_error
from mlscratch.model_selection import train_test_split
from mlscratch.preprocessing import StandardScaler
from mlscratch.viz import plt, savefig

rng = np.random.default_rng(0)
n = 500
area = rng.uniform(40, 250, n)
rooms = np.clip(np.round(area / 35 + rng.normal(0, 0.8, n)), 1, 8)
age = rng.uniform(0, 60, n)
price = 50 + 2.0 * area + 8 * rooms - 0.9 * age + rng.normal(0, 25, n)        # in k$
X = np.c_[area, rooms, age]
names = ["area_m2", "rooms", "age_years"]
Xtr, Xte, ytr, yte = train_test_split(X, price, test_size=0.25, random_state=0)

ols = LinearRegression("normal").fit(Xtr, ytr)
sc = StandardScaler().fit(Xtr)
gd = LinearRegression("gd", lr=0.1, n_iter=10000).fit(sc.transform(Xtr), ytr)
gd_coef = gd.coef_ / sc.scale_                                               # back to original units
gd_b = gd.intercept_ - np.sum(gd.coef_ * sc.mean_ / sc.scale_)
print("true      : b=50.00  w=[2.000, 8.000, -0.900]")
print(f"normal eq : b={ols.intercept_:.2f}  w={np.round(ols.coef_, 3)}")
print(f"grad desc : b={gd_b:.2f}  w={np.round(gd_coef, 3)}  ({len(gd.loss_history_)} iterations on scaled features)")
print(f"test RMSE {root_mean_squared_error(yte, ols.predict(Xte)):.2f} k$,  R² {r2_score(yte, ols.predict(Xte)):.3f}")

# Inference: Var(ŵ) = σ² (XᵀX)⁻¹
Xb = add_bias(Xtr)
resid = ytr - ols.predict(Xtr)
dof = len(ytr) - Xb.shape[1]
sigma2 = resid @ resid / dof
se = np.sqrt(np.diag(sigma2 * np.linalg.inv(Xb.T @ Xb)))
coef = np.r_[ols.intercept_, ols.coef_]
t = coef / se
p = 2 * stats.t.sf(np.abs(t), dof)
print(f"\n{'term':<11}{'coef':>9}{'std err':>9}{'t':>8}{'p-value':>10}   95% CI")
for name, c, s, tv, pv in zip(["intercept", *names], coef, se, t, p):
    q = stats.t.ppf(0.975, dof)
    print(f"{name:<11}{c:>9.3f}{s:>9.3f}{tv:>8.2f}{pv:>10.2e}   [{c - q * s:.3f}, {c + q * s:.3f}]")
print("Xᵀ·residuals (≈ 0 → residuals ⟂ features):", np.round(Xb.T @ resid, 6))

# Multicollinearity inflates standard errors
X_col = np.c_[area, area + rng.normal(0, 0.5, n)]
Xc = add_bias(X_col)
fit = LinearRegression().fit(X_col, price)
r = price - fit.predict(X_col)
se_col = np.sqrt(np.diag((r @ r / (n - 3)) * np.linalg.inv(Xc.T @ Xc)))
print(f"\nnear-duplicate feature: coefs {np.round(fit.coef_, 2)} with std errors {np.round(se_col[1:], 2)} (vs {se[1]:.3f} before)")

pyplot = plt()
fig, axes = pyplot.subplots(1, 3, figsize=(15, 4))
axes[0].plot(gd.loss_history_)
axes[0].set(yscale="log", xlabel="iteration", ylabel="MSE", title="gradient descent converges to the OLS optimum")
axes[1].scatter(ols.predict(Xte), yte, s=8)
lims = [yte.min(), yte.max()]
axes[1].plot(lims, lims, "k--")
axes[1].set(xlabel="predicted", ylabel="actual", title="predicted vs actual (test)")
axes[2].scatter(ols.predict(Xtr), resid, s=8)
axes[2].axhline(0, color="k")
axes[2].set(xlabel="fitted", ylabel="residual", title="residuals: no pattern = good")
savefig(fig, "linear_regression")
