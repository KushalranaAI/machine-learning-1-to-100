"""Decision trees from scratch: impurity functions, depth vs overfitting, readable rules and regression trees."""

import numpy as np

from mlscratch.datasets import load_iris, make_moons
from mlscratch.model_selection import cross_val_score, train_test_split
from mlscratch.tree import DecisionTreeClassifier, DecisionTreeRegressor, entropy_from_counts, gini_from_counts
from mlscratch.viz import decision_boundary, plt, savefig

print("impurity of a node with class counts [p·100, (1-p)·100]:")
for p in (0.0, 0.1, 0.3, 0.5):
    c = np.array([p * 100, (1 - p) * 100])
    print(f"  p={p:.1f}: Gini {gini_from_counts(c):.3f}   entropy {entropy_from_counts(c):.3f}")

X, y, names = load_iris()
tree = DecisionTreeClassifier(max_depth=3).fit(X, y)
print("\niris tree (max_depth=3):")
print(tree.export_text(["sepal_len", "sepal_wid", "petal_len", "petal_wid"]))
print("feature importances:", dict(zip(["sepal_len", "sepal_wid", "petal_len", "petal_wid"], np.round(tree.feature_importances_, 3).tolist())))

Xm, ym = make_moons(500, noise=0.3, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(Xm, ym, test_size=0.3, random_state=0)
print(f"\n{'max_depth':>9}{'leaves':>8}{'train acc':>11}{'test acc':>10}{'5-fold CV':>11}")
for depth in (1, 2, 3, 4, 6, 8, 12, None):
    t = DecisionTreeClassifier(max_depth=depth).fit(Xtr, ytr)
    cv = cross_val_score(DecisionTreeClassifier(max_depth=depth), Xtr, ytr, cv=5).mean()
    print(f"{str(depth):>9}{t.n_leaves_:>8}{t.score(Xtr, ytr):>11.3f}{t.score(Xte, yte):>10.3f}{cv:>11.3f}")

pyplot = plt()
fig, axes = pyplot.subplots(1, 3, figsize=(15, 4.3))
for ax, depth in zip(axes, (2, 5, None)):
    decision_boundary(ax, DecisionTreeClassifier(max_depth=depth).fit(Xtr, ytr), Xm, ym)
    ax.set_title(f"max_depth={depth}")
savefig(fig, "tree_depth")

rng = np.random.default_rng(0)
xr = np.sort(rng.uniform(0, 6, 200))
yr = np.sin(xr) + rng.normal(0, 0.2, 200)
fig, ax = pyplot.subplots(figsize=(8, 4))
ax.scatter(xr, yr, s=8, color="grey")
for depth in (2, 4, 10):
    reg = DecisionTreeRegressor(max_depth=depth).fit(xr, yr)
    ax.plot(xr, reg.predict(xr), label=f"depth {depth} ({reg.n_leaves_} leaves)")
ax.legend()
ax.set_title("regression trees are piecewise constant")
savefig(fig, "regression_tree")
