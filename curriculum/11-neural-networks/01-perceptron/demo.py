"""The perceptron: convergence on separable data (vs the Novikoff bound), failure on XOR, and a hand-built 2-layer XOR net."""

import numpy as np

from mlscratch.datasets import make_blobs
from mlscratch.linear_model import Perceptron

X, y = make_blobs(200, centers=[[-2, -2], [2, 2]], cluster_std=0.8, random_state=0)
p = Perceptron(random_state=0).fit(X, y)
print(f"separable blobs: converged after {len(p.errors_)} epochs, mistakes per epoch {p.errors_}")
t = np.where(y == 1, 1, -1)
w_star = p.coef_ / np.linalg.norm(p.coef_)
Xb = np.c_[X, np.ones(len(X))]
wb = np.r_[p.coef_, p.intercept_]
gamma = np.min(t * (Xb @ wb)) / np.linalg.norm(wb)
R = np.max(np.linalg.norm(Xb, axis=1))
print(f"total mistakes {sum(p.errors_)}  ≤ Novikoff bound (R/γ)² = ({R:.2f}/{gamma:.3f})² = {(R / gamma) ** 2:.0f} (with the margin of the found separator)")

Xx = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
yx = np.array([0, 1, 1, 0])
px = Perceptron(n_iter=1000, random_state=0).fit(Xx, yx)
print(f"\nXOR: after 1000 epochs mistakes per epoch still {px.errors_[-5:]} → accuracy {px.score(Xx, yx):.2f} (never 1.0)")

step = lambda z: (z > 0).astype(int)  # noqa: E731
h_or = step(Xx @ np.array([1, 1]) - 0.5)
h_nand = step(Xx @ np.array([-1, -1]) + 1.5)
out = step(np.c_[h_or, h_nand] @ np.array([1, 1]) - 1.5)
print("two-layer net: OR =", h_or, " NAND =", h_nand, " AND(OR, NAND) =", out, " target =", yx, " ✓" if np.array_equal(out, yx) else "")

rng = np.random.default_rng(1)
Xn, yn = make_blobs(200, centers=[[-1, -1], [1, 1]], cluster_std=1.5, random_state=1)
pn = Perceptron(n_iter=30, random_state=0).fit(Xn, yn)
print(f"\nnon-separable data, mistakes per epoch (never reaches 0): {pn.errors_[-10:]}")
