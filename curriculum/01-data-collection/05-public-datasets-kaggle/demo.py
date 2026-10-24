"""Download a public dataset with fallbacks (Kaggle → original → mirror), then write a quick data card."""

from pathlib import Path

import pandas as pd

from mlscratch.download import fetch, kaggle_available

print("Kaggle token configured:", kaggle_available())
raw = Path("outputs/raw")
path = fetch(
    "iris.csv",
    sources=[
        {"kaggle": "uciml/iris", "file": "Iris.csv"},
        {"url": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"},
        {"url": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"},
    ],
    dest_dir=raw,
)

df = pd.read_csv(path)
if df.shape[1] == 5 and not any(c.lower().startswith("sepal") for c in df.columns):  # UCI file has no header row
    df = pd.read_csv(path, header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
df.columns = [c.lower().replace("cm", "").strip("_ ").replace(" ", "_") for c in df.columns]

print("\n=== Data card ===")
print(f"file      : {path}  ({path.stat().st_size / 1024:.1f} KB)")
print(f"rows      : {len(df)}   columns: {df.shape[1]}")
print(f"columns   : {list(df.columns)}")
print(f"missing   : {int(df.isna().sum().sum())} values")
print(f"target    : {df.columns[-1]} → {df.iloc[:, -1].value_counts().to_dict()}")
print("numeric summary:\n", df.describe().round(2).T[["mean", "std", "min", "max"]])
