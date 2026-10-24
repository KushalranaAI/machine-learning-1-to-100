---
title: Data Cleaning & Missing Values
order: 2
summary: Fix types, parse dates, standardise categories, remove duplicates and handle missing values (MCAR/MAR/MNAR, deletion, mean/median/mode, KNN imputation, missing indicators).
level: beginner
minutes: 55
why: Models can't read "$52,000", "N/A" or "Bombay vs Mumbai". Most libraries crash on NaN, and naive fixes (dropping rows, filling with 0) silently bias your results.
background: Rubin's 1976 framework (MCAR / MAR / MNAR) is still how statisticians reason about missing data. Multiple imputation (Rubin, 1987) and KNN imputation (Troyanskaya et al., 2001) are standard tools.
use_cases:
  - Preparing any real-world dataset for modelling
  - Survey data with skipped questions
  - Sensor data with gaps
  - Merging sources with different conventions
where_used:
  - 02-titanic-survival
  - 01-house-prices
code:
  - mlscratch/preprocessing.py:SimpleImputer
  - mlscratch/preprocessing.py:KNNImputer
demo: demo.py
widgets: []
pros:
  - Imputation keeps every row (more data for the model)
  - Missing-indicator columns let the model learn from missingness itself
cons:
  - Mean imputation shrinks variance and distorts correlations
  - Imputing before the train/test split leaks test information
workflow: |
  flowchart TD
    A[Load raw] --> B[Standardise missing codes → NaN]
    B --> C[Fix dtypes: numbers, dates, booleans]
    C --> D[Normalise categories: strip, lower, map synonyms]
    D --> E[Drop exact duplicates]
    E --> F{Missing values}
    F -->|few rows, MCAR| G[Drop rows]
    F -->|numeric| H[Median / KNN impute + indicator]
    F -->|categorical| I[Mode or 'missing' category]
    G & H & I --> J[Clean table]
---

## Step 1 - make missing values visible
Missing data hides behind many codes: `""`, `"N/A"`, `"?"`, `"unknown"`, `-1`, `999`. Convert them all to `NaN` first:
```python
df = df.replace({"": np.nan, "N/A": np.nan, "?": np.nan, "unknown": np.nan})
```

## Step 2 - fix types
- Numbers stored as text: `pd.to_numeric(df.income.str.replace(r"[$,]", "", regex=True), errors="coerce")`.
- Dates in mixed formats: `pd.to_datetime(s, format="mixed", dayfirst=True)` - then check the result.
- Booleans: map every spelling: `{"yes": 1, "y": 1, "1": 1, "no": 0, "n": 0, "0": 0}`.
- Units: detect and convert (heights below 100 "cm" are probably inches → multiply by 2.54).

## Step 3 - categories and duplicates
```python
df["city"] = df.city.str.strip().str.title().replace({"Bombay": "Mumbai", "Nyc": "New York"})
df = df.drop_duplicates()                       # exact duplicates
df = df.drop_duplicates(subset="customer_id")   # duplicate entities (decide which copy to keep!)
```

## Step 4 - missing values: understand the mechanism
| Mechanism | Meaning | Example | Consequence |
|-----------|---------|---------|-------------|
| **MCAR** - missing completely at random | unrelated to anything | sensor packet lost | dropping rows is unbiased (just loses data) |
| **MAR** - missing at random | depends on *observed* columns | older users skip "income" | impute using the other columns (KNN, model-based) |
| **MNAR** - missing not at random | depends on the *missing value itself* | rich users hide income | no imputation fixes it fully - add a **missing indicator** and think about the domain |

## Step 5 - strategies
| Strategy | Formula / idea | When |
|----------|----------------|------|
| Drop rows | `dropna()` | few rows, MCAR |
| Drop column | | > ~60% missing and not informative |
| Mean | $\hat x = \bar x$ | symmetric numeric, quick baseline |
| **Median** | $\hat x = \text{median}(x)$ | skewed numeric (income) - robust to outliers |
| Mode / "Missing" category | most frequent / new level | categorical |
| **KNN imputation** | mean of the feature over the $k$ most similar rows | MAR, related features |
| Model-based / iterative | predict the column from the others | MAR, many features |
| **+ missing indicator** | extra 0/1 column `x_was_missing` | always worth trying - missingness is often predictive |

**Mean imputation's hidden cost:** every imputed value sits exactly at the mean, so the variance shrinks by a factor
$\frac{n_{obs}-1}{n-1}$ and correlations with other features are weakened. The demo measures this.

## The golden rule: fit on train only
Imputation statistics (means, medians, neighbours) must be **learned from the training set** and then applied to
validation/test data - otherwise information leaks from the test set. Our `SimpleImputer`/`KNNImputer` follow
`fit` → `transform` for exactly this reason, and `Pipeline` (Part 6) makes it automatic inside cross-validation.

## Exercises
1. Run the demo and compare the income distribution after mean vs median vs KNN imputation.
2. Is `age` missing at random with respect to `plan`? Compute the missing rate per plan.
3. Add missing-indicator columns and check whether `age_was_missing` correlates with churn.
