"""Compare SGD, momentum, RMSProp and Adam on an elongated (ill-conditioned) quadratic bowl."""

import numpy as np

from mlscratch.optim import SGD, Adam, RMSProp, minimize
from mlscratch.viz import plt, savefig

# f(x, y) = 0.5 * (x² + 25 y²): curvature 1 along x and 25 along y → condition number 25
A = np.diag([1.0, 25.0])


def f(t):
    return 0.5 * t @ A @ t


def grad(t):
    return A @ t


optimisers = {
    "SGD (lr=0.07)": SGD(0.07),
    "Momentum (lr=0.02, β=0.9)": SGD(0.02, momentum=0.9),
    "RMSProp (lr=0.1)": RMSProp(0.1),
    "Adam (lr=0.3)": Adam(0.3),
}

pyplot = plt()
fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(12, 4.5))
xx, yy = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-2, 2, 200))
ax1.contour(xx, yy, 0.5 * (xx**2 + 25 * yy**2), levels=25, cmap="Greys", linewidths=0.6)

print(f"{'optimiser':<28}{'final loss':>14}{'steps to loss<1e-2':>22}")
for name, opt in optimisers.items():
    path = minimize(grad, [-3.5, 1.5], opt, n_steps=150)
    losses = np.array([f(p) for p in path])
    reached = np.argmax(losses < 1e-2) if np.any(losses < 1e-2) else None
    print(f"{name:<28}{losses[-1]:>14.2e}{str(reached):>22}")
    ax1.plot(path[:, 0], path[:, 1], ".-", ms=3, label=name)
    ax2.semilogy(losses, label=name)

ax1.set_title("Paths on f(x,y) = ½(x² + 25y²)")
ax1.legend(fontsize=8)
ax2.set_title("Loss per step (log scale)")
ax2.set_xlabel("step")
ax2.legend(fontsize=8)
savefig(fig, "optimiser_paths")

print("\nLearning-rate sweep for plain GD (theory: diverges when lr > 2/λ_max = 0.08):")
for lr in (0.01, 0.04, 0.07, 0.079, 0.081):
    path = minimize(grad, [-3.5, 1.5], SGD(lr), n_steps=100)
    print(f"  lr={lr:<6} final loss = {f(path[-1]):.3e}")
