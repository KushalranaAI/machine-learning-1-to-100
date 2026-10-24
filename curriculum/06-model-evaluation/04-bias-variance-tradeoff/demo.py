"""Estimate bias² and variance empirically, plus validation and learning curves."""

import numpy as np

from mlscratch.linear_model import LinearRegression, Ridge
from mlscratch.model_selection import learning_curve
from mlscratch.preprocessing import PolynomialFeatures
from mlscratch.tree import DecisionTreeRegressor
from mlscratch.viz import plt, savefig

rng = np.random.default_rng(0)
f = lambda x: np.sin(2 * np.pi * x)  # noqa: E731
sigma, n_train = 0.3, 25
x_test = np.linspace(0.1, 0.9, 100)   # avoid the edges, where high-degree fits explode

print(f"{'degree':>6}{'bias²':>10}{'variance':>10}{'noise':>8}{'total':>9}")
rows = []
for degree in range(1, 13):
    preds = []
    for _ in range(300):                                  # 300 independent training sets
        x = rng.uniform(0, 1, n_train)
        y = f(x) + rng.normal(0, sigma, n_train)
        pf = PolynomialFeatures(degree).fit((2 * x - 1)[:, None])    # map to [-1, 1]: keeps xᵈ well-conditioned
        m = Ridge(1e-6).fit(pf.transform((2 * x - 1)[:, None]), y)
        preds.append(m.predict(pf.transform((2 * x_test - 1)[:, None])))
    preds = np.array(preds)
    bias2 = np.mean((preds.mean(0) - f(x_test)) ** 2)
    var = np.mean(preds.var(0))
    rows.append((degree, bias2, var))
    print(f"{degree:>6}{bias2:>10.4f}{var:>10.4f}{sigma**2:>8.3f}{bias2 + var + sigma**2:>9.4f}")
best = min(rows, key=lambda r: r[1] + r[2])
print(f"→ lowest expected error at degree {best[0]}. Bias² collapses once the curve can bend (degree 3); after that,")
print("  every extra degree mostly adds variance. The even/odd zig-zag is real: sin(2πx) is an ODD function around the")
print("  centre, so each even power is a useless term that only adds variance.")

pyplot = plt()
fig, axes = pyplot.subplots(1, 3, figsize=(15, 4))
d, b, v = zip(*rows)
axes[0].plot(d, b, "o-", label="bias²")
axes[0].plot(d, v, "o-", label="variance")
axes[0].plot(d, np.array(b) + np.array(v) + sigma**2, "o-", label="total expected error")
axes[0].set_yscale("log")
axes[0].set(xlabel="polynomial degree", title="bias-variance decomposition")
axes[0].legend()

X = rng.uniform(0, 1, (400, 1))
y = f(X[:, 0]) + rng.normal(0, sigma, 400)
for ax, model, title in [(axes[1], LinearRegression(), "learning curve: linear (high bias)"),
                         (axes[2], DecisionTreeRegressor(), "learning curve: deep tree (high variance)")]:
    curve = learning_curve(model, X, y, train_sizes=(0.05, 0.1, 0.2, 0.4, 0.7, 1.0))
    sizes, tr, va = zip(*curve)
    ax.plot(sizes, tr, "o-", label="train R²")
    ax.plot(sizes, va, "o-", label="validation R²")
    ax.set(xlabel="fraction of training data", title=title, ylim=(-0.2, 1.05))
    ax.legend()
    print(f"{title:<42} final train R² {tr[-1]:.3f}, validation R² {va[-1]:.3f}")
savefig(fig, "bias_variance")
