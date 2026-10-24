"""Synthetic datasets for the lessons (no downloads needed) and loaders for the bundled CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import rng_from

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def make_regression(n_samples=200, n_features=1, noise=10.0, coef=None, bias=0.0, random_state=None):
    rng = rng_from(random_state)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.uniform(5, 50, n_features) if coef is None else np.asarray(coef, float)
    y = X @ w + bias + rng.normal(0, noise, n_samples)
    return X, y, w


def make_blobs(n_samples=300, centers=3, cluster_std=1.0, n_features=2, random_state=None):
    rng = rng_from(random_state)
    C = rng.uniform(-10, 10, (centers, n_features)) if isinstance(centers, int) else np.asarray(centers, float)
    labels = rng.integers(len(C), size=n_samples)
    return C[labels] + rng.normal(0, cluster_std, (n_samples, n_features)), labels


def make_moons(n_samples=200, noise=0.1, random_state=None):
    rng = rng_from(random_state)
    n1 = n_samples // 2
    t1, t2 = np.linspace(0, np.pi, n1), np.linspace(0, np.pi, n_samples - n1)
    X = np.vstack([np.c_[np.cos(t1), np.sin(t1)], np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]])
    return X + rng.normal(0, noise, X.shape), np.r_[np.zeros(n1, int), np.ones(n_samples - n1, int)]


def make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=None):
    rng = rng_from(random_state)
    n1 = n_samples // 2
    t1, t2 = np.linspace(0, 2 * np.pi, n1, endpoint=False), np.linspace(0, 2 * np.pi, n_samples - n1, endpoint=False)
    X = np.vstack([np.c_[np.cos(t1), np.sin(t1)], factor * np.c_[np.cos(t2), np.sin(t2)]])
    return X + rng.normal(0, noise, X.shape), np.r_[np.zeros(n1, int), np.ones(n_samples - n1, int)]


def make_classification(n_samples=500, n_features=10, n_informative=3, class_sep=1.0, weights=None, random_state=None):
    """Informative features carry the class signal; the rest are pure noise (good for feature-selection demos)."""
    rng = rng_from(random_state)
    p = weights[0] if weights else 0.5
    y = (rng.random(n_samples) > p).astype(int)
    X = rng.normal(size=(n_samples, n_features))
    X[:, :n_informative] += class_sep * (2 * y[:, None] - 1) * rng.uniform(0.5, 1.5, n_informative)
    return X, y


def load_iris():
    """Fisher's iris data (150 flowers, 4 measurements, 3 species) bundled in data/iris.csv."""
    import csv

    rows = list(csv.DictReader(open(DATA_DIR / "iris.csv")))
    X = np.array([[float(r[k]) for k in ("sepal_length", "sepal_width", "petal_length", "petal_width")] for r in rows])
    names = sorted({r["species"] for r in rows})
    y = np.array([names.index(r["species"]) for r in rows])
    return X, y, names
