"""Generate data/messy_customers.csv - a deliberately dirty dataset for the wrangling lessons.

Problems planted on purpose: mixed date formats, inconsistent category spellings, missing-value codes,
duplicate rows, impossible ages, outlier incomes, currency strings, stray whitespace and mixed units.
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 600
cities = ["Lima", "lima ", "LIMA", "Mumbai", "mumbai", "Bombay", "New York", "NYC", "new york", "Berlin", "berlin"]
plans = ["basic", "Basic", "BASIC", "premium", "Premium ", "prem", "enterprise", "Enterprise"]

signup = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")
fmt = rng.choice(["%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y"], n)
age = rng.normal(38, 11, n).round().astype(int)
age[rng.choice(n, 6, replace=False)] = rng.choice([-1, 0, 150, 999], 6)          # impossible values
income = rng.lognormal(10.8, 0.5, n).round(-2)
income[rng.choice(n, 4, replace=False)] = [2_500_000, 3_100_000, 1_900_000, 4_000_000]  # extreme but real?
height = np.where(rng.random(n) < 0.15, (rng.normal(170, 9, n) / 2.54).round(1), rng.normal(170, 9, n).round(1))  # some in inches!

df = pd.DataFrame({
    "customer_id": np.arange(1000, 1000 + n),
    "name": [f"  Customer {i} " if i % 7 == 0 else f"Customer {i}" for i in range(n)],
    "signup_date": [d.strftime(f) for d, f in zip(signup, fmt)],
    "city": rng.choice(cities, n),
    "plan": rng.choice(plans, n),
    "age": age.astype(object),
    "annual_income": pd.Series([f"${v:,.0f}" if rng.random() < 0.3 else v for v in income], dtype=object),
    "height_cm": height,
    "monthly_visits": rng.poisson(6, n).astype(float),
    "churned": rng.choice(["yes", "no", "Y", "N", "1", "0"], n, p=[0.15, 0.4, 0.05, 0.2, 0.05, 0.15]),
})
# missing values in several disguises
for col, codes in {"age": ["?", "", None], "city": ["N/A", None, "unknown"], "monthly_visits": [np.nan], "plan": [None]}.items():
    idx = rng.choice(n, 45, replace=False)
    df[col] = df[col].astype(object)  # allow mixed codes like "?" next to numbers
    df.loc[idx, col] = rng.choice(np.array(codes, dtype=object), len(idx))
df = pd.concat([df, df.sample(25, random_state=1)], ignore_index=True)               # exact duplicates
df = df.sample(frac=1, random_state=2).reset_index(drop=True)

out = Path(__file__).resolve().parent / "messy_customers.csv"
df.to_csv(out, index=False)
print(f"wrote {out} with {len(df)} rows")
