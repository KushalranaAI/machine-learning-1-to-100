"""Titanic feature engineering. Most of the signal is hidden in text columns (Name, Ticket, Cabin).

All statistics (median ages per title, fare medians, ticket-group sizes, category lists) are learned in fit() from
training rows and reused in transform(). The cross-validation in train.py refits them inside every fold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mlscratch.preprocessing import OneHotEncoder, StandardScaler

DATA = Path(__file__).resolve().parent / "data"
TITLE_MAP = {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master", "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
             "Lady": "Rare", "Countess": "Rare", "Capt": "Rare", "Col": "Rare", "Don": "Rare", "Dr": "Rare",
             "Major": "Rare", "Rev": "Rare", "Sir": "Rare", "Jonkheer": "Rare", "Dona": "Rare"}


def load(name: str = "train.csv") -> pd.DataFrame:
    if not (DATA / name).exists():
        raise SystemExit(f"data/{name} not found - run `python download_data.py` first.")
    return pd.read_csv(DATA / name)


def parse(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless parsing: pull structured facts out of the raw strings."""
    out = df.copy()
    # "Braund, Mr. Owen Harris" → "Mr". The title encodes sex, age group and social status at once.
    out["Title"] = out.Name.str.extract(r",\s*([^.]+)\.", expand=False).str.strip().map(TITLE_MAP).fillna("Rare")
    out["FamilySize"] = out.SibSp + out.Parch + 1
    out["IsAlone"] = (out.FamilySize == 1).astype(float)
    out["HasCabin"] = out.Cabin.notna().astype(float)                     # missingness itself is informative
    out["Deck"] = out.Cabin.str[0].fillna("U").replace({"T": "U"})       # deck letter, U = unknown
    out["TicketPrefix"] = out.Ticket.str.contains(r"[A-Za-z]").astype(float)
    return out


class TitanicFeatures:
    def __init__(self, scale: bool = True):
        self.scale = scale

    def fit(self, df: pd.DataFrame):
        d = parse(df)
        self.age_by_title_ = d.groupby("Title").Age.median().to_dict()   # Masters are boys: median ~4, not 28
        self.age_median_ = float(d.Age.median())
        self.fare_median_ = float(d.Fare.median())
        self.embarked_mode_ = d.Embarked.mode()[0]
        self.ticket_counts_ = d.Ticket.value_counts().to_dict()          # people sharing a ticket travelled together
        self.onehot_ = OneHotEncoder().fit(self._cats(d))
        self.scaler_ = StandardScaler().fit(self._matrix(d)) if self.scale else None
        return self

    def _cats(self, d):
        return d[["Sex", "Embarked", "Title", "Deck", "Pclass"]].fillna({"Embarked": self.embarked_mode_}).astype(str).to_numpy()

    def _matrix(self, d):
        age = d.Age.fillna(d.Title.map(self.age_by_title_)).fillna(self.age_median_)
        fare = d.Fare.fillna(self.fare_median_)
        group = d.Ticket.map(self.ticket_counts_).fillna(1)
        num = np.c_[age, d.Age.isna(), np.log1p(fare), np.log1p(fare / group), group, d.FamilySize, d.IsAlone,
                    d.HasCabin, d.TicketPrefix, d.SibSp, d.Parch, age < 14]
        return np.c_[num.astype(float), self.onehot_.transform(self._cats(d))]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self._matrix(parse(df))
        return self.scaler_.transform(X) if self.scaler_ is not None else X

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    @property
    def names_(self):
        num = ["Age", "AgeMissing", "logFare", "logFarePerPerson", "TicketGroup", "FamilySize", "IsAlone", "HasCabin",
               "TicketPrefix", "SibSp", "Parch", "IsChild"]
        return num + self.onehot_.get_feature_names(["Sex", "Embarked", "Title", "Deck", "Pclass"])


class FeaturePipeline:
    """features + model as one estimator, so cross_val_score refits the feature statistics in every fold."""

    def __init__(self, model, scale=True):
        self.model, self.scale = model, scale

    def clone(self):
        return FeaturePipeline(self.model.clone(), self.scale)

    def fit(self, df, y):
        self.features_ = TitanicFeatures(self.scale).fit(df)
        self.model.fit(self.features_.transform(df), y)
        return self

    def predict(self, df):
        return self.model.predict(self.features_.transform(df))

    def predict_proba(self, df):
        return self.model.predict_proba(self.features_.transform(df))
