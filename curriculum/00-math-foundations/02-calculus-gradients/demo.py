"""Analytic gradients vs. numerical finite differences, for the losses used throughout the course."""

import numpy as np

from mlscratch.base import sigmoid
from mlscratch.neural_network import Dense, Sequential, Tanh, gradient_check

rng = np.random.default_rng(0)
X = rng.normal(size=(50, 4))
y_reg = X @ np.array([1.0, -2.0, 0.5, 3.0]) + rng.normal(size=50)
y_clf = (X[:, 0] + X[:, 1] > 0).astype(float)


def numeric_grad(f, w, eps=1e-6):
    g = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w)
        e[i] = eps
        g[i] = (f(w + e) - f(w - e)) / (2 * eps)
    return g


def check(name, f, grad, w):
    a, n = grad(w), numeric_grad(f, w)
    err = np.max(np.abs(a - n) / np.maximum(1e-8, np.abs(a) + np.abs(n)))
    print(f"{name:<34} max relative error = {err:.2e}  {'✓' if err < 1e-6 else '✗'}")


w = rng.normal(size=4)
lam = 0.3

mse = lambda w: np.mean((X @ w - y_reg) ** 2)  # noqa: E731
check("MSE  ∇ = (2/n)Xᵀ(Xw - y)", mse, lambda w: 2 / len(X) * X.T @ (X @ w - y_reg), w)

ridge = lambda w: np.mean((X @ w - y_reg) ** 2) + lam * w @ w  # noqa: E731
check("Ridge  ∇ = (2/n)Xᵀ(Xw-y) + 2λw", ridge, lambda w: 2 / len(X) * X.T @ (X @ w - y_reg) + 2 * lam * w, w)


def logloss(w):
    p = sigmoid(X @ w)
    return -np.mean(y_clf * np.log(p) + (1 - y_clf) * np.log(1 - p))


check("Log-loss  ∇ = (1/n)Xᵀ(σ(Xw) - y)", logloss, lambda w: X.T @ (sigmoid(X @ w) - y_clf) / len(X), w)

# A whole neural network: backprop through Dense → tanh → Dense, checked parameter by parameter
net = Sequential([Dense(4, 6, rng=1), Tanh(), Dense(6, 2, rng=2)])
Y = np.eye(2)[y_clf.astype(int)]
print(f"{'2-layer network backprop':<34} max relative error = {gradient_check(net, X, Y):.2e}")

# Hessian of the log-loss: XᵀSX/n with S = diag(p(1-p)) - always positive semi-definite → convex
p = sigmoid(X @ w)
H = X.T @ (X * (p * (1 - p))[:, None]) / len(X)
print("\nlog-loss Hessian eigenvalues (all ≥ 0 ⇒ convex):", np.round(np.linalg.eigvalsh(H), 4))

# One Newton step vs one gradient step from the same point
g = X.T @ (p - y_clf) / len(X)
w_newton = w - np.linalg.solve(H, g)
w_gd = w - 0.5 * g
print(f"loss now {logloss(w):.4f} → after one GD step {logloss(w_gd):.4f} → after one Newton step {logloss(w_newton):.4f}")
