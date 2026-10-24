"""Loading, scaling and the adjusted Rand index (a label-permutation-invariant agreement score for two clusterings)."""

from __future__ import annotations

from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data" / "Mall_Customers.csv"
FEATURES = ["age", "income", "score"]


def load() -> pd.DataFrame:
    if not DATA.exists():
        raise SystemExit("data/Mall_Customers.csv not found - run `python download_data.py` first.")
    df = pd.read_csv(DATA)
    df.columns = ["customer_id", "gender", "age", "income", "score"]  # Genre/Gender and units vary between copies
    return df


def adjusted_rand_index(a, b) -> float:
    """ARI = (RI - E[RI]) / (max RI - E[RI]) computed from the contingency table n_ij.

    It counts pairs of points the two clusterings put together or apart, corrected for chance:
    1 = identical partitions (up to renaming), about 0 = no better than random labels.
    """
    a, b = np.asarray(a), np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1), int)
    np.add.at(table, (ai, bi), 1)
    sum_ij = sum(comb(int(v), 2) for v in table.ravel())
    sum_a = sum(comb(int(v), 2) for v in table.sum(1))
    sum_b = sum(comb(int(v), 2) for v in table.sum(0))
    expected = sum_a * sum_b / comb(len(a), 2)
    return float((sum_ij - expected) / (0.5 * (sum_a + sum_b) - expected))
