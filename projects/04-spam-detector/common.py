"""Loading plus the hand-made "meta" features that complement the bag of words."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data" / "sms_spam.raw"


def load() -> pd.DataFrame:
    """Return a DataFrame with columns label ('ham'/'spam'), text, y (1 = spam). Handles the Kaggle CSV and UCI TSV."""
    if not DATA.exists():
        raise SystemExit("data/sms_spam.raw not found - run `python download_data.py` first.")
    data = DATA.read_bytes()
    try:
        raw = data.decode("utf-8")      # UCI file and GitHub mirror
    except UnicodeDecodeError:
        raw = data.decode("latin-1")    # Kaggle's spam.csv
    if raw.startswith("v1,v2"):                                          # Kaggle: CSV with 3 junk columns
        df = pd.read_csv(io.StringIO(raw), usecols=[0, 1], names=["label", "text"], header=0)
    else:                                                                # UCI: "ham<TAB>message", no header
        df = pd.read_csv(io.StringIO(raw), sep="\t", names=["label", "text"], quoting=csv.QUOTE_NONE)
    df = df.dropna().drop_duplicates("text").reset_index(drop=True)  # duplicated texts would leak between splits
    df["y"] = (df.label == "spam").astype(int)
    return df


META_NAMES = ["log_length", "digit_ratio", "upper_ratio", "has_url", "has_currency", "has_phone", "exclamations"]


def meta_features(texts) -> np.ndarray:
    """Signals a bag of words misses: spam is long, shouty, full of digits, prices, links and phone numbers."""
    rows = []
    for t in texts:
        n = max(len(t), 1)
        rows.append([
            np.log1p(len(t)),
            sum(c.isdigit() for c in t) / n,
            sum(c.isupper() for c in t) / n,
            float(bool(re.search(r"https?://|www\.|\.com\b", t, re.I))),
            float(bool(re.search(r"[£$€]|\bp(?:ence)?\b|\bcash\b", t, re.I))),
            float(bool(re.search(r"\b\d{5,}\b", t))),
            float(t.count("!")),
        ])
    return np.array(rows)
