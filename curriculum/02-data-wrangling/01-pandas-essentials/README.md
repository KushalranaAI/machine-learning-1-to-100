---
title: pandas Essentials
order: 1
summary: Series and DataFrames, selecting with loc/iloc and boolean masks, vectorised operations, groupby, dtypes and memory - the toolkit for every later step.
level: beginner
minutes: 45
why: pandas is how Python data scientists hold and manipulate tables. Fluency with selection, vectorisation and groupby makes every other part of the workflow faster and less buggy.
background: Wes McKinney started pandas in 2008 at a hedge fund to bring R-style data frames to Python. It sits on top of NumPy arrays.
use_cases:
  - Loading, inspecting and summarising any tabular dataset
  - Building features with groupby/merge
  - Preparing train/test matrices for models
where_used:
  - 01-house-prices
  - 02-titanic-survival
  - 03-customer-segmentation
demo: demo.py
workflow: |
  flowchart LR
    A[read_csv] --> B[head / info / describe]
    B --> C[select: loc, iloc, masks]
    C --> D[vectorised new columns]
    D --> E[groupby → aggregate]
    E --> F[to_numpy → model]
---

## Two objects
- **Series**: a 1-D labelled array (one column).
- **DataFrame**: a table of Series sharing an **index** (row labels).

## First look at any dataset
```python
df.shape            # (rows, columns)
df.info()           # dtypes, non-null counts, memory
df.head()           # first rows
df.describe()       # count, mean, std, quartiles of numeric columns
df.isna().sum()     # missing values per column
df["col"].value_counts(dropna=False)   # category frequencies, including missing
```

## Selecting data
| Goal | Code |
|------|------|
| one column | `df["age"]` |
| several columns | `df[["age", "plan"]]` |
| rows by label | `df.loc[10:20, "age"]` |
| rows by position | `df.iloc[0:10, 2]` |
| rows by condition | `df[(df.age > 30) & (df.plan == "basic")]` |
| membership | `df[df.city.isin(["Lima", "Berlin"])]` |

Use `&`, `|`, `~` (not `and`/`or`) and parentheses around each condition.

> **SettingWithCopy**: modify data through `df.loc[mask, "col"] = value`, not `df[mask]["col"] = value` (the latter may change a temporary copy).

## Vectorise, don't loop
Column operations run in compiled code. `df["bmi"] = df.weight / df.height**2` is 100-1000× faster than a Python loop or `df.apply(..., axis=1)`.
Use `np.where`, `.str` and `.dt` accessors, `.map(dict)`, and `pd.cut` for most transformations.

## groupby: split → apply → combine
```python
df.groupby("plan").agg(customers=("customer_id", "nunique"),
                       avg_income=("income", "mean"),
                       churn_rate=("churned", "mean"))
df["income_vs_plan_avg"] = df.income / df.groupby("plan").income.transform("mean")   # same shape as df
```
`agg` collapses groups; `transform` broadcasts the group result back to every row - perfect for features.

## dtypes and memory
- `category` dtype for low-cardinality strings saves memory and speeds up groupby.
- Downcast numbers (`pd.to_numeric(s, downcast="integer")`).
- `df.memory_usage(deep=True)` shows the real cost.

## Exercises
1. Find the 5 cities with the highest average monthly visits (after cleaning, next lesson).
2. Add a column with each customer's visits rank within their plan (`groupby(...).rank()`).
3. Time a loop vs a vectorised version of the same column computation.
