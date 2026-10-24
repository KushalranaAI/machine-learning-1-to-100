"""Transforms and binning: a multiplicative relationship, a skewed target and a step-shaped effect."""

import numpy as np

from mlscratch.linear_model import LinearRegression
from mlscratch.metrics import r2_score
from mlscratch.model_selection import train_test_split
from mlscratch.preprocessing import KBinsDiscretizer, OneHotEncoder, PowerTransformer
from mlscratch.viz import plt, savefig

rng = np.random.default_rng(0)

# multiplicative data: price = 300 · area^1.2 · noise
area = rng.uniform(30, 400, 800)
price = 300 * area**1.2 * np.exp(rng.normal(0, 0.25, 800))
a_tr, a_te, p_tr, p_te = train_test_split(area, price, test_size=0.3, random_state=0)

raw = LinearRegression().fit(a_tr, p_tr)
log = LinearRegression().fit(np.log(a_tr), np.log(p_tr))
pred_log = np.exp(log.predict(np.log(a_te)))
print(f"price ~ area           test R² = {raw.score(a_te, p_te):.3f}")
print(f"log price ~ log area   test R² = {r2_score(p_te, pred_log):.3f}  (back on the price scale)")
print(f"fitted exponent b = {log.coef_[0]:.3f} (true 1.2) → 'a 1% larger home costs ~{log.coef_[0]:.2f}% more'")

# relative error is what matters for skewed targets
rel = lambda y, p: np.mean(np.abs(p - y) / y)  # noqa: E731
print(f"mean relative error: raw model {rel(p_te, raw.predict(a_te)):.1%}, log model {rel(p_te, pred_log):.1%}")

# Yeo-Johnson on a skewed feature
pt = PowerTransformer(standardize=False).fit(price[:, None])
print(f"\nYeo-Johnson λ for price = {pt.lambdas_[0]:.3f} (λ = 1: leave as is, λ = 0: take the log)")

# step-shaped effect: binning + one-hot lets a linear model fit it
age = rng.uniform(18, 80, 1000)
risk = np.select([age < 25, age < 60], [3.0, 1.0], 2.0) + rng.normal(0, 0.3, 1000)
bins = KBinsDiscretizer(8, "quantile").fit(age[:, None])
X_binned = OneHotEncoder().fit_transform(bins.transform(age[:, None]))
print(f"\nstep-shaped risk vs age: linear R² {LinearRegression().fit(age, risk).score(age, risk):.3f}, "
      f"binned+one-hot R² {LinearRegression().fit(X_binned, risk).score(X_binned, risk):.3f}")

pyplot = plt()
fig, axes = pyplot.subplots(1, 3, figsize=(14, 3.6))
axes[0].scatter(area, price, s=4)
axes[0].set_title("price vs area (curved, fanning out)")
axes[1].scatter(np.log(area), np.log(price), s=4, color="#2b8a3e")
axes[1].set_title("log-log: straight line, constant spread")
order = np.argsort(age)
axes[2].scatter(age, risk, s=4, alpha=0.4)
axes[2].plot(age[order], LinearRegression().fit(X_binned, risk).predict(X_binned)[order], color="#e8590c")
axes[2].set_title("binned linear model follows the steps")
savefig(fig, "transforms")
