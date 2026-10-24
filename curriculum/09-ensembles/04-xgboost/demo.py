"""XGBoost-style boosting from scratch: leaf weights, the effect of λ and γ, and a comparison with plain GBM."""

import numpy as np

from mlscratch.datasets import make_classification
from mlscratch.ensemble import GradientBoostingClassifier, XGBoostLike
from mlscratch.metrics import log_loss, roc_auc_score
from mlscratch.model_selection import train_test_split


def count_leaves(node):
    return 1 if node.left is None else count_leaves(node.left) + count_leaves(node.right)


rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, (300, 2))
y = X[:, 0] ** 2 + np.sin(3 * X[:, 1]) + rng.normal(0, 0.2, 300)

# One round, squared loss, λ=0: leaf weight = -G/H = mean residual in the leaf
m = XGBoostLike(n_estimators=1, learning_rate=1.0, max_depth=2, reg_lambda=0.0).fit(X, y)
tree = m.trees_[0]
leaf_pred = m._predict_tree(tree, X)
for w in np.unique(np.round(leaf_pred, 8))[:4]:
    members = np.isclose(leaf_pred, w)
    print(f"leaf weight {w:+.4f}   mean residual of its samples {np.mean(y[members] - y.mean()):+.4f}")

print("\nλ shrinks leaf weights, γ prunes leaves (depth-6 trees, 1 round):")
for lam, gam in [(0, 0), (10, 0), (0, 2), (0, 10)]:
    t = XGBoostLike(n_estimators=1, max_depth=6, reg_lambda=lam, gamma=gam).fit(X, y).trees_[0]
    print(f"  λ={lam:<3} γ={gam:<3} leaves {count_leaves(t):>3}   largest |w| {np.max(np.abs(XGBoostLike._predict_tree(t, X))):.3f}")

Xc, yc = make_classification(2000, 12, n_informative=6, class_sep=0.6, random_state=1)
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.3, random_state=0)
xgb = XGBoostLike("binary:logistic", n_estimators=150, learning_rate=0.1, max_depth=4, reg_lambda=1.0, gamma=0.1,
                  subsample=0.8, colsample=0.8, random_state=0).fit(Xtr, ytr)
gbm = GradientBoostingClassifier(150, 0.1, 4).fit(Xtr, ytr)
for name, model in [("XGBoost-style", xgb), ("plain gradient boosting", gbm)]:
    p = model.predict_proba(Xte)[:, 1]
    print(f"{name:<24} test accuracy {np.mean((p > 0.5) == yte):.3f}   AUC {roc_auc_score(yte, p):.3f}   log-loss {log_loss(yte, p):.3f}")
