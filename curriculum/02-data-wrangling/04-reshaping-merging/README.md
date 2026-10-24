---
title: Reshaping, Merging & Aggregating
order: 4
summary: Combine and reshape tables - merge/join types, concat, pivot and melt, groupby aggregations, and time-based resampling - to build one modelling table.
level: intermediate
minutes: 40
why: Real data is spread across tables (customers, orders, products) and stored in shapes that don't match what a model needs. Most feature engineering is really joining and aggregating.
background: Relational joins come from Codd's relational algebra (1970). Wide vs long ("tidy") data was formalised by Wickham (2014).
use_cases:
  - Joining labels to features from several sources
  - Turning event logs into per-customer features
  - Pivoting survey answers or monthly sales into columns
  - Resampling time series to daily/weekly
where_used:
  - 03-customer-segmentation
demo: demo.py
cons:
  - Wrong join keys or types silently drop or duplicate rows
  - Many-to-many joins can explode the row count
workflow: |
  flowchart LR
    A[(customers)] -->|left merge on id| M[combined]
    B[(orders)] -->|groupby customer → agg| G[per-customer stats] -->|merge| M
    M --> P[pivot / melt to the needed shape]
    P --> T[one row per sample = model table]
---

## Joins
| `how=` | keeps | rows from right without a match |
|--------|-------|-----------------------------------|
| `inner` | only keys in both | dropped |
| `left` | every left row | NaN-filled |
| `right` | every right row | - |
| `outer` | everything | NaN-filled on both sides |

```python
merged = customers.merge(orders_agg, on="customer_id", how="left", validate="one_to_one", indicator=True)
merged["_merge"].value_counts()      # left_only / right_only / both - check what didn't match!
```
`validate=` raises an error if the key isn't unique on the side you expect - the cheapest bug detector in pandas.
**Always compare row counts before and after a merge.**

## Concatenation
`pd.concat([df_2023, df_2024])` stacks rows (same columns); `axis=1` places frames side by side (aligned on the index).

## Wide ↔ long
- **melt** (wide → long): columns `jan, feb, mar` become rows `(month, value)` - tidy, easy to group and plot.
- **pivot / pivot_table** (long → wide): one column per category, with an aggregation for duplicates.

```python
long = sales.melt(id_vars="store", var_name="month", value_name="revenue")
wide = long.pivot_table(index="store", columns="month", values="revenue", aggfunc="sum")
```

## Aggregation = features
```python
feats = orders.groupby("customer_id").agg(
    n_orders=("order_id", "count"),
    total=("amount", "sum"),
    avg=("amount", "mean"),
    first=("date", "min"),
    last=("date", "max"),
    n_categories=("category", "nunique"),
)
```

## Time series resampling
With a `DatetimeIndex`: `daily.resample("W").sum()`, rolling windows `s.rolling(7).mean()`, lags `s.shift(1)`.
Lag/rolling features must only look **backwards** in time.

## Exercises
1. Run the demo and find the orders whose `customer_id` has no matching customer (`indicator=True`).
2. Create a wide table with one column per category containing each customer's total spend.
3. Build a weekly revenue series and a 4-week rolling mean.
