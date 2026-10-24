"""Feature engineering for California Housing, shared by explore.py and train.py.

Everything that learns from data (medians for imputation, the one-hot categories, the scaler)
is fitted on the TRAINING rows only and then applied to validation/test rows. Doing it before the split leaks test information.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mlscratch.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler

DATA = Path(__file__).resolve().parent / "data" / "housing.csv"
CAP = 500_001  # the census capped house values at $500,001


def load(drop_capped: bool = True) -> pd.DataFrame:
    if not DATA.exists():
        raise SystemExit("data/housing.csv not found - run `python download_data.py` first.")
    df = pd.read_csv(DATA)
    if drop_capped:  # capped rows have an unknown true value; keeping them teaches the model a false ceiling
        df = df[df.median_house_value < CAP].reset_index(drop=True)
    return df


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Per-household ratios mean more than raw district totals (a big district has more rooms *and* more people)."""
    out = df.copy()
    out["rooms_per_household"] = out.total_rooms / out.households
    out["bedrooms_per_room"] = out.total_bedrooms / out.total_rooms
    out["people_per_household"] = out.population / out.households
    # Distance to the two big job centres - a hand-made "location" feature that linear models can use.
    for name, (lat, lon) in {"dist_sf": (37.77, -122.42), "dist_la": (34.05, -118.24)}.items():
        out[name] = np.hypot(out.latitude - lat, out.longitude - lon)
    return out


NUMERIC = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households",
           "median_income", "rooms_per_household", "bedrooms_per_room", "people_per_household", "dist_sf", "dist_la"]
SKEWED = ["total_rooms", "total_bedrooms", "population", "households", "rooms_per_household", "people_per_household"]


class HousingFeatures:
    """Raw DataFrame → numeric matrix: ratios, log1p of skewed counts, median imputation, one-hot ocean_proximity, scaling."""

    def __init__(self, scale: bool = True):
        self.scale = scale

    def _numeric(self, df: pd.DataFrame) -> np.ndarray:
        df = add_ratios(df)
        for c in SKEWED:
            df[c] = np.log1p(df[c])
        return df[NUMERIC].to_numpy(float)

    def fit(self, df: pd.DataFrame):
        num = self._numeric(df)
        self.imputer_ = SimpleImputer("median").fit(num)
        self.onehot_ = OneHotEncoder().fit(df[["ocean_proximity"]].to_numpy())
        X = np.c_[self.imputer_.transform(num), self.onehot_.transform(df[["ocean_proximity"]].to_numpy())]
        self.scaler_ = StandardScaler().fit(X) if self.scale else None
        self.names_ = NUMERIC + [f"ocean={c}" for c in self.onehot_.categories_[0]]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = np.c_[self.imputer_.transform(self._numeric(df)), self.onehot_.transform(df[["ocean_proximity"]].to_numpy())]
        return self.scaler_.transform(X) if self.scaler_ is not None else X

    def fit_transform(self, df):
        return self.fit(df).transform(df)
