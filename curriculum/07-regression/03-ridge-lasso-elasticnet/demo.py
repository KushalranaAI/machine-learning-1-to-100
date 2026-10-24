"""Ridge vs Lasso vs Elastic Net: shrinkage, sparsity, correlated features and choosing α by CV."""

import numpy as np

from mlscratch.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from mlscratch.model_selection import GridSearchCV
from mlscratch.viz import plt, savefig

rng = np.random.default_rng(0)
n, d = 50, 40                                         # few samples, many features → OLS overfits
true_w = np.zeros(d)
true_w[:5] = [4, -3, 2.5, 2, -1.5]                    # only 5 features matter
X = rng.normal(size=(n, d))
y = X @ true_w + rng.normal(0, 2.0, n)
X_test = rng.normal(size=(2000, d))
y_test = X_test @ true_w + rng.normal(0, 2.0, 2000)

mse = lambda m: np.mean((m.predict(X_test) - y_test) ** 2)  # noqa: E731
models = {
    "OLS": LinearRegression().fit(X, y),
    "Ridge (α by CV)": GridSearchCV(Ridge(), {"alpha": list(np.logspace(-2, 3, 20))}, cv=5).fit(X, y).best_estimator_,
    "Lasso (α by CV)": GridSearchCV(Lasso(), {"alpha": list(np.logspace(-3, 0.5, 20))}, cv=5).fit(X, y).best_estimator_,
    "ElasticNet (α by CV)": GridSearchCV(ElasticNet(l1_ratio=0.5), {"alpha": list(np.logspace(-3, 0.5, 20))}, cv=5).fit(X, y).best_estimator_,
}
print(f"{n} samples, {d} features, 5 truly useful")
for name, m in models.items():
    alpha = f"α={m.alpha:.3f}" if hasattr(m, "alpha") else ""
    print(f"{name:<21}{alpha:<10} test MSE {mse(m):6.2f}   non-zero weights {int(np.sum(np.abs(m.coef_) > 1e-8)):>2}   ||w||₂ {np.linalg.norm(m.coef_):.2f}")
lasso = models["Lasso (α by CV)"]
print(f"Lasso kept features {np.flatnonzero(np.abs(lasso.coef_) > 1e-8).tolist()} - the 5 true ones are {[0, 1, 2, 3, 4]}")

# Ridge = shrinkage in the SVD basis
Xc, yc = X - X.mean(0), y - y.mean()
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
alpha = 10
w_svd = Vt.T @ ((S / (S**2 + alpha)) * (U.T @ yc))
print("\nRidge closed form == SVD shrinkage formula:", np.allclose(w_svd, Ridge(alpha).fit(X, y).coef_))


# correlated group: three near-copies of one signal
z = rng.normal(size=n)
Xg = np.c_[z + rng.normal(0, 0.05, n), z + rng.normal(0, 0.05, n), z + rng.normal(0, 0.05, n), rng.normal(size=(n, 3))]
yg = 3 * z + rng.normal(0, 0.5, n)
print("\ncorrelated group of 3 copies of the signal:")
print("  Lasso      weights:", np.round(Lasso(0.05).fit(Xg, yg).coef_[:3], 2), "← picks one, drops the rest")
print("  ElasticNet weights:", np.round(ElasticNet(0.05, 0.3).fit(Xg, yg).coef_[:3], 2), "← spreads the weight")

alphas = np.logspace(-3, 1, 40)
pyplot = plt()
fig, axes = pyplot.subplots(1, 2, figsize=(13, 4))
for ax, Model, title in [(axes[0], Ridge, "Ridge path: shrinks smoothly"), (axes[1], Lasso, "Lasso path: exact zeros")]:
    scale = 50 if Model is Ridge else 1
    path = np.array([Model(a * scale).fit(X, y).coef_ for a in alphas])
    ax.plot(alphas * scale, path[:, :5], lw=2)
    ax.plot(alphas * scale, path[:, 5:], color="grey", lw=0.6)
    ax.set_xscale("log")
    ax.set(xlabel="α", ylabel="coefficient", title=title)
savefig(fig, "regularisation_paths")
