"""Clean data/messy_customers.csv step by step and compare imputation strategies."""

import numpy as np
import pandas as pd

from mlscratch.datasets import DATA_DIR
from mlscratch.preprocessing import KNNImputer, SimpleImputer

raw = pd.read_csv(DATA_DIR / "messy_customers.csv", dtype=str)
df = raw.copy()
print("raw:", df.shape)

# 1. missing codes → NaN
df = df.replace({"": np.nan, "N/A": np.nan, "?": np.nan, "unknown": np.nan})

# 2. types
df["annual_income"] = pd.to_numeric(df["annual_income"].str.replace(r"[$,]", "", regex=True), errors="coerce")
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["height_cm"] = pd.to_numeric(df["height_cm"], errors="coerce")
df["monthly_visits"] = pd.to_numeric(df["monthly_visits"], errors="coerce")
df["signup_date"] = pd.to_datetime(df["signup_date"], format="mixed", dayfirst=True)
df["churned"] = df["churned"].str.strip().str.lower().map({"yes": 1, "y": 1, "1": 1, "no": 0, "n": 0, "0": 0})
inches = df["height_cm"] < 100                      # values like 66.9 are inches, not centimetres
df.loc[inches, "height_cm"] = (df.loc[inches, "height_cm"] * 2.54).round(1)
print(f"converted {inches.sum()} heights from inches to cm")

# impossible values → NaN (an age of 999 is a data-entry error, not an outlier)
bad_age = ~df["age"].between(1, 110)
df.loc[bad_age & df["age"].notna(), "age"] = np.nan
print(f"impossible ages set to missing: {(bad_age & raw['age'].notna()).sum()}")

# 3. categories and duplicates
df["name"] = df["name"].str.strip()
df["city"] = df["city"].str.strip().str.title().replace({"Bombay": "Mumbai", "Nyc": "New York"})
df["plan"] = df["plan"].str.strip().str.lower().replace({"prem": "premium"})
print("cities:", sorted(df["city"].dropna().unique()), "| plans:", sorted(df["plan"].dropna().unique()))
before = len(df)
df = df.drop_duplicates()
print(f"dropped {before - len(df)} exact duplicate rows → {len(df)} rows")

print("\nmissing after type fixes:\n", df.isna().sum()[df.isna().sum() > 0])

# 4. is 'age' missing at random? compare missing rate across plans
print("\nage missing rate by plan:\n", df.groupby("plan", dropna=False)["age"].apply(lambda s: s.isna().mean()).round(3))

# 5. compare imputation strategies on income (inject extra MCAR gaps so we know the truth)
num = df[["age", "annual_income", "height_cm", "monthly_visits"]].to_numpy(float)
rng = np.random.default_rng(0)
truth_rows = np.flatnonzero(~np.isnan(num[:, 1]))
hide = rng.choice(truth_rows, 60, replace=False)
truth = num[hide, 1].copy()
masked = num.copy()
masked[hide, 1] = np.nan
log_masked = masked.copy()
log_masked[:, 1] = np.log(log_masked[:, 1])         # income is skewed: impute on the log scale for KNN
results = {
    "mean": SimpleImputer("mean").fit_transform(masked)[hide, 1],
    "median": SimpleImputer("median").fit_transform(masked)[hide, 1],
    "KNN (k=5)": np.exp(KNNImputer(5).fit_transform(log_masked)[hide, 1]),
}
print("\nimputing 60 hidden incomes - mean absolute error vs the true value:")
for name, est in results.items():
    print(f"  {name:<10} MAE = {np.mean(np.abs(est - truth)):>10,.0f}")
full = SimpleImputer("mean").fit_transform(masked)[:, 1]
print(f"\nvariance shrinkage from mean imputation: std before {np.nanstd(masked[:, 1]):,.0f} → after {full.std():,.0f}")

# 6. missing indicator + final imputation
df["age_was_missing"] = df["age"].isna().astype(int)
df["age"] = df["age"].fillna(df["age"].median())
df["city"] = df["city"].fillna("Missing")
print("\nchurn rate when age was missing vs present:", df.groupby("age_was_missing")["churned"].mean().round(3).to_dict())
df.to_csv(DATA_DIR / "clean_customers.csv", index=False)
print(f"\nclean table saved to data/clean_customers.csv: {df.shape}")
