"""Logistic regression: gradient descent vs Newton's method, odds ratios, separable data and decision boundaries."""

import numpy as np

from mlscratch.base import add_bias, sigmoid
from mlscratch.datasets import make_blobs, make_moons
from mlscratch.linear_model import LogisticRegression
from mlscratch.metrics import log_loss, roc_auc_score
from mlscratch.model_selection import train_test_split
from mlscratch.preprocessing import PolynomialFeatures, StandardScaler
from mlscratch.viz import decision_boundary, plt, savefig

rng = np.random.default_rng(0)
n = 1000
age = rng.uniform(20, 70, n)
smoker = rng.integers(0, 2, n)
bmi = rng.normal(27, 4, n)
logit = -9 + 0.08 * age + 1.2 * smoker + 0.12 * bmi                 # true log-odds of disease
y = (rng.random(n) < sigmoid(logit)).astype(int)
X = np.c_[age, smoker, bmi]

gd = LogisticRegression(lr=0.5, n_iter=5000, C=1e6)
sc = StandardScaler().fit(X)
gd.fit(sc.transform(X), y)
w_gd = gd.coef_[0] / sc.scale_                                      # undo scaling
b_gd = gd.intercept_[0] - np.sum(gd.coef_[0] * sc.mean_ / sc.scale_)

# Newton's method (IRLS): w ← w - H⁻¹∇,  H = XᵀSX/n
Xb = add_bias(X)
w = np.zeros(4)
for it in range(10):
    p = sigmoid(Xb @ w)
    grad = Xb.T @ (p - y) / n
    H = Xb.T @ (Xb * (p * (1 - p))[:, None]) / n
    w -= np.linalg.solve(H, grad)
print("true    : b=-9.000  w=[0.080, 1.200, 0.120]")
print(f"GD      : b={b_gd:.3f}  w={np.round(w_gd, 3)}   (5000 iterations)")
print(f"Newton  : b={w[0]:.3f}  w={np.round(w[1:], 3)}   (10 iterations)")
print("odds ratios e^w:", {k: round(float(v), 3) for k, v in zip(["age (+1 yr)", "smoker", "bmi (+1)"], np.exp(w[1:]))})
p = sigmoid(Xb @ w)
print(f"log-loss {log_loss(y, p):.4f}   AUC {roc_auc_score(y, p):.3f}")

# Separable data: without regularisation the weights keep growing
Xs, ys = make_blobs(100, centers=[[-2, -2], [2, 2]], cluster_std=0.7, random_state=0)
for iters in (100, 1000, 10000):
    m = LogisticRegression(lr=1.0, n_iter=iters, C=1e9).fit(Xs, ys)
    print(f"separable data, {iters:>5} iterations, no regularisation: ||w|| = {np.linalg.norm(m.coef_):.2f}")
print(f"with C=1 (regularised) after 10000 iterations: ||w|| = {np.linalg.norm(LogisticRegression(lr=1.0, n_iter=10000, C=1).fit(Xs, ys).coef_):.2f}")

# Linear vs polynomial features on a curved boundary
Xm, ym = make_moons(400, noise=0.2, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(Xm, ym, test_size=0.3, random_state=0)
lin = LogisticRegression(n_iter=3000).fit(Xtr, ytr)


class PolyLogistic:
    def __init__(self, degree):
        self.pf, self.sc, self.m = PolynomialFeatures(degree), StandardScaler(), LogisticRegression(lr=0.5, n_iter=5000)

    def fit(self, X, y):
        self.m.fit(self.sc.fit_transform(self.pf.fit_transform(X)), y)
        return self

    def predict(self, X):
        return self.m.predict(self.sc.transform(self.pf.transform(X)))


poly = PolyLogistic(3).fit(Xtr, ytr)
print(f"\nmoons test accuracy: linear features {lin.score(Xte, yte):.3f}, degree-3 features {np.mean(poly.predict(Xte) == yte):.3f}")
pyplot = plt()
fig, axes = pyplot.subplots(1, 2, figsize=(11, 4.5))
decision_boundary(axes[0], lin, Xm, ym)
axes[0].set_title("logistic regression, linear features")
decision_boundary(axes[1], poly, Xm, ym)
axes[1].set_title("logistic regression, degree-3 features")
savefig(fig, "logistic_boundaries")
