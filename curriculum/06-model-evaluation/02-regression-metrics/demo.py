"""Regression metrics side by side, the mean/median optimality result, and residual patterns."""

import numpy as np

from mlscratch.linear_model import LinearRegression
from mlscratch.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score, root_mean_squared_error
from mlscratch.viz import plt, savefig

rng = np.random.default_rng(0)
y = rng.normal(100, 20, 200)
pred = y + rng.normal(0, 8, 200)


def report(label, yt, yp):
    print(f"{label:<28} RMSE {root_mean_squared_error(yt, yp):7.2f}  MAE {mean_absolute_error(yt, yp):7.2f}  "
          f"R² {r2_score(yt, yp):6.3f}  MAPE {mean_absolute_percentage_error(yt, yp):6.1%}")


report("good model", y, pred)
bad = pred.copy()
bad[0] += 300                                      # one terrible prediction
report("same + one huge miss", y, bad)
report("predict the mean (baseline)", y, np.full_like(y, y.mean()))
report("worse than the mean", y, y[::-1])

# the constant that minimises each loss
skewed = rng.lognormal(3, 0.8, 1000)
grid = np.linspace(skewed.min(), np.percentile(skewed, 95), 4000)
best_mse = grid[np.argmin([np.mean((skewed - c) ** 2) for c in grid])]
best_mae = grid[np.argmin([np.mean(np.abs(skewed - c)) for c in grid])]
print(f"\nskewed data: mean {skewed.mean():.2f}, median {np.median(skewed):.2f}")
print(f"constant minimising MSE {best_mse:.2f} (= mean), constant minimising MAE {best_mae:.2f} (= median)")

# residual patterns
x = rng.uniform(0, 10, 300)
curved = 2 + x**2 + rng.normal(0, 4, 300)
funnel = np.exp(0.3 * x) * (1 + rng.normal(0, 0.25, 300))
pyplot = plt()
fig, axes = pyplot.subplots(1, 3, figsize=(14, 3.6))
for ax, (title, target) in zip(axes, [("good fit", 3 * x + rng.normal(0, 2, 300)), ("curve → missing x²", curved), ("funnel → try log(y)", funnel)]):
    m = LinearRegression().fit(x, target)
    ax.scatter(m.predict(x), target - m.predict(x), s=6)
    ax.axhline(0, color="k", lw=1)
    ax.set_title(title)
    ax.set_xlabel("prediction")
axes[0].set_ylabel("residual")
savefig(fig, "residual_patterns")
_ = mean_squared_error
