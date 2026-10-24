"""Tiny plotting helpers for the lesson demos (matplotlib is imported lazily)."""

from __future__ import annotations

import os
from pathlib import Path


def plt():
    import matplotlib

    matplotlib.use("Agg")  # no window needed; figures are saved as PNG
    import matplotlib.pyplot as pyplot

    return pyplot


def savefig(fig, name: str) -> Path:
    """Save to $ML_OUTPUT_DIR (set by the web app's Run button) or ./outputs next to the demo."""
    out = Path(os.environ.get("ML_OUTPUT_DIR", "outputs"))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt().close(fig)
    print(f"[figure] {path}")
    return path


def decision_boundary(ax, model, X, y, steps=200, alpha=0.25):
    """Shade the plane by predicted class and scatter the data on top."""
    import numpy as np

    x0, x1 = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y0, y1 = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x0, x1, steps), np.linspace(y0, y1, steps))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=alpha, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=25)
