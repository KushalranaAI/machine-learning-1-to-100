"""pandas essentials on the messy customers file: inspect, select, vectorise, group."""

import time

import numpy as np
import pandas as pd

from mlscratch.datasets import DATA_DIR

df = pd.read_csv(DATA_DIR / "messy_customers.csv")
print("shape:", df.shape)
print("\ndtypes (note: age, income and churned are text - they need cleaning):\n", df.dtypes)
print("\nmissing per column:\n", df.isna().sum())
print("\nplan spellings:", df["plan"].value_counts(dropna=False).to_dict())

# selection
print("\nloc - rows 0-2, two columns:\n", df.loc[0:2, ["customer_id", "plan"]])
visits = df["monthly_visits"]
print(f"\nboolean mask - customers with > 10 visits: {(visits > 10).sum()}")

# vectorised vs loop
big = pd.DataFrame({"w": np.random.default_rng(0).uniform(50, 100, 500_000), "h": np.random.default_rng(1).uniform(1.5, 2.0, 500_000)})
t = time.perf_counter(); loop = [w / h**2 for w, h in zip(big.w, big.h)]; t_loop = time.perf_counter() - t
t = time.perf_counter(); big["bmi"] = big.w / big.h**2; t_vec = time.perf_counter() - t
t = time.perf_counter(); big.apply(lambda r: r.w / r.h**2, axis=1); t_apply = time.perf_counter() - t
print(f"\n500k BMI values: python loop {t_loop:.3f}s · apply {t_apply:.2f}s · vectorised {t_vec:.4f}s")

# groupby with agg and transform (quick clean of 'plan' first)
df["plan_clean"] = df["plan"].str.strip().str.lower().replace({"prem": "premium"})
summary = df.groupby("plan_clean").agg(customers=("customer_id", "nunique"), mean_visits=("monthly_visits", "mean"))
print("\nper plan:\n", summary.round(2))
df["visits_vs_plan_mean"] = df["monthly_visits"] / df.groupby("plan_clean")["monthly_visits"].transform("mean")
print("\ntransform keeps one value per row:\n", df[["plan_clean", "monthly_visits", "visits_vs_plan_mean"]].head().round(2))

# memory: object strings vs category
before = df["plan_clean"].memory_usage(deep=True)
after = df["plan_clean"].astype("category").memory_usage(deep=True)
print(f"\nmemory of plan column: object {before} bytes → category {after} bytes")
