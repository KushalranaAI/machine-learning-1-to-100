"""Loading and a fixed train/validation/test protocol shared by every MNIST script."""

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data" / "mnist.npz"


def load(flat: bool = True, n_train: int | None = None, seed: int = 0):
    """Pixels scaled to [0, 1]. The last 5,000 training images are the validation set; the official 10k are the test set."""
    if not DATA.exists():
        raise SystemExit("data/mnist.npz not found - run `python download_data.py` first.")
    with np.load(DATA) as z:
        xtr, ytr, xte, yte = z["x_train"], z["y_train"].astype(int), z["x_test"], z["y_test"].astype(int)
    xtr, xte = xtr.astype(np.float64) / 255.0, xte.astype(np.float64) / 255.0
    xval, yval, xtr, ytr = xtr[-5000:], ytr[-5000:], xtr[:-5000], ytr[:-5000]
    if n_train:
        idx = np.random.default_rng(seed).choice(len(xtr), n_train, replace=False)
        xtr, ytr = xtr[idx], ytr[idx]
    if flat:
        xtr, xval, xte = (a.reshape(len(a), -1) for a in (xtr, xval, xte))
    return xtr, ytr, xval, yval, xte, yte


def show_digits(ax_grid, images, titles):
    for ax, img, t in zip(ax_grid.ravel(), images, titles):
        ax.imshow(img.reshape(28, 28), cmap="gray_r")
        ax.set_title(t, fontsize=8)
        ax.axis("off")
