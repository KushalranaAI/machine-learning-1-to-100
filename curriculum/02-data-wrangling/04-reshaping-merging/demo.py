"""Merge, aggregate, pivot and resample - building one model table from several raw tables."""

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
customers = pd.DataFrame({"customer_id": range(1, 11), "segment": rng.choice(["retail", "business"], 10)})
orders = pd.DataFrame({
    "order_id": range(100, 160),
    "customer_id": rng.integers(1, 13, 60),             # ids 11 and 12 don't exist in `customers`!
    "date": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 120, 60), unit="D"),
    "category": rng.choice(["books", "games", "toys"], 60),
    "amount": rng.gamma(2, 25, 60).round(2),
})

# 1. joins - always check what didn't match
check = orders.merge(customers, on="customer_id", how="left", indicator=True)
print("order rows whose customer is unknown:", (check["_merge"] == "left_only").sum())
for how in ("inner", "left", "right", "outer"):
    print(f"  {how:<5} join → {len(orders.merge(customers, on='customer_id', how=how))} rows")

# 2. aggregation → per-customer features
feats = orders.groupby("customer_id").agg(
    n_orders=("order_id", "count"), total=("amount", "sum"), avg_order=("amount", "mean"),
    last_order=("date", "max"), n_categories=("category", "nunique"),
).round(2)
table = customers.merge(feats, on="customer_id", how="left", validate="one_to_one")
table[["n_orders", "total"]] = table[["n_orders", "total"]].fillna(0)
print("\nmodel table (one row per customer):\n", table.head().to_string(index=False))

# 3. pivot: spend per category as columns; melt it back
wide = orders.pivot_table(index="customer_id", columns="category", values="amount", aggfunc="sum", fill_value=0).round(2)
print("\npivot_table (wide):\n", wide.head())
long = wide.reset_index().melt(id_vars="customer_id", var_name="category", value_name="spend")
print("\nmelt back to long:", long.shape)

# 4. a merge bug: duplicate keys multiply rows
dup_customers = pd.concat([customers, customers.iloc[[0]]])
try:
    orders.merge(dup_customers, on="customer_id", validate="many_to_one")
except pd.errors.MergeError as e:
    print("\nvalidate= caught a duplicated key:", str(e)[:80])

# 5. time series resampling and lag features
daily = orders.set_index("date").sort_index()["amount"].resample("D").sum()
weekly = daily.resample("W").sum()
frame = pd.DataFrame({"weekly_revenue": weekly, "rolling_4w": weekly.rolling(4).mean(), "last_week": weekly.shift(1)})
print("\nweekly revenue with backward-looking features:\n", frame.head(8).round(1))
