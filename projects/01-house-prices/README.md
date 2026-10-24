---
title: "California House Prices"
level: Beginner
summary: "Predict the median house value of 20,000 Californian districts: data cleaning, feature engineering, linear models, trees and gradient boosting, all from scratch."
problem: |
  A property platform wants an automated valuation for every district: given census facts like income, house age,
  rooms, population and location, predict the **median house value**. This is a **regression** problem. We measure
  error in dollars (RMSE, MAE) and in relative terms, because a $40k miss matters more on a $100k house than on a $450k one.
dataset:
  name: "California Housing Prices (1990 census)"
  source: "Kaggle · camnugent/california-housing-prices (from Pace & Barry, 1997)"
  url: "https://www.kaggle.com/datasets/camnugent/california-housing-prices"
  kaggle: "camnugent/california-housing-prices"
  license: "Public domain (US census data). Mirror: github.com/ageron/handson-ml2"
  size: "20,640 rows × 10 columns · 1.4 MB"
algorithms: [Linear Regression, Ridge, kNN, Decision Tree, Random Forest, Gradient Boosting]
skills:
  - Finding and handling missing values (median imputation fitted on train only)
  - Spotting a censored target (values capped at $500,001)
  - Ratio features, distance features, log transforms of skewed counts
  - One-hot encoding a categorical column
  - Train / validation / test protocol and why the test set is touched once
  - Multicollinearity, and why Ridge stabilises coefficients
  - Reading a boosting learning curve (early stopping)
lessons:
  - 01-data-collection/05-public-datasets-kaggle
  - 02-data-wrangling/02-data-cleaning
  - 03-eda/01-descriptive-statistics-visualization
  - 04-feature-engineering/01-feature-scaling
  - 04-feature-engineering/02-encoding-categorical
  - 04-feature-engineering/03-transformations-binning
  - 04-feature-engineering/04-feature-creation
  - 06-model-evaluation/02-regression-metrics
  - 07-regression/01-linear-regression
  - 07-regression/03-ridge-lasso-elasticnet
  - 08-classification/03-k-nearest-neighbors
  - 09-ensembles/01-bagging-random-forest
  - 09-ensembles/03-gradient-boosting
steps:
  - title: "1 · Download the data"
    script: download_data.py
    description: |
      `mlscratch.download.fetch` tries the Kaggle API first (if you have `~/.kaggle/kaggle.json`). Otherwise it
      downloads the identical CSV from GitHub, then caches it in `data/`.
  - title: "2 · Explore: missing values, the price cap, skew and correlations"
    script: explore.py
    description: |
      You'll find **207 missing `total_bedrooms`**. You'll also find that **4.7% of districts sit exactly at $500,001**,
      a censoring cap: the true value is "≥ $500k", so we drop those rows instead of teaching the model a false ceiling.
      Count columns are heavily right-skewed (skew ≈ 4-5), so we log-transform them. `median_income` is by far the
      strongest single predictor (r = 0.64).
  - title: "3 · Feature engineering (features.py)"
    description: |
      `HousingFeatures` turns the raw table into 18 numbers:
      - **ratios**: rooms / household, bedrooms / room, people / household. A big district has more rooms *and* more people, so raw totals mostly measure size.
      - **distances** to San Francisco and Los Angeles: a location signal linear models can actually use.
      - **log1p** of skewed counts, **median imputation**, **one-hot** `ocean_proximity`, **standardisation**.

      Every statistic (medians, categories, means, std) is learned in `fit()` on training rows only.
  - title: "4 · Train, compare and evaluate"
    script: train.py
    description: |
      Seven models are compared on a validation set: a mean baseline, OLS, Ridge, kNN, a tree, a random forest and
      gradient boosting. Then the winner is refit on train + validation and scored **once** on the test set. The target
      is `log(price)`, so errors are relative. Expect about 2½ minutes: the forests and boosting are pure NumPy.
workflow: |
  flowchart LR
    A[Kaggle / GitHub CSV] --> B[EDA: missing, cap, skew]
    B --> C[Drop capped rows]
    C --> D[Split 60/20/20]
    D --> E[Fit features on train:<br/>ratios, log, impute, one-hot, scale]
    E --> F[7 models on validation]
    F --> G[Boosting curve → n rounds]
    G --> H[Refit on train+val]
    H --> I[Test set, once]
---

# California House Prices: regression end to end

## Results (seeded run on the full dataset)

| Model | Validation RMSE (log) | Validation RMSE ($) | R² |
|---|---|---|---|
| Mean baseline | 0.529 | 101,303 | 0.00 |
| Linear regression | 0.297 | 65,114 | 0.68 |
| Ridge (α = 10) | 0.299 | 62,734 | 0.68 |
| kNN (k = 10, distance-weighted) | 0.259 | 52,871 | 0.76 |
| Decision tree (depth 10) | 0.266 | 53,760 | 0.75 |
| Random forest (40 trees) | 0.216 | 45,009 | 0.83 |
| **Gradient boosting (≈ 390 rounds, depth 5)** | **0.206** | **42,255** | **0.85** |

**Test set (touched once):** RMSE $41,981 · MAE $27,448 · R² 0.837 · median error **10.7%** of the true price.

## What to notice
- **The baseline matters.** Without it, "RMSE = $42k" means nothing. The baseline is $101k.
- **Multicollinearity.** `total_rooms`, `total_bedrooms`, `population` and `households` are all "district size"
  (pairwise r = 0.87-0.97 after the log). OLS gives them large opposite-signed weights (households +2.0, population −1.3), and its CV R²
  swings ±0.10 between folds. Ridge shrinks them to a stable compromise (CV R² ± 0.014). Same predictions, very different
  stories, so be careful when you interpret raw OLS coefficients.
- **Trees beat lines here** because location matters non-linearly: coastal and Bay Area pockets, the
  LA basin. Boosting's importances put income first and then four location features.
- **Validation → test gap is tiny** (0.206 vs 0.215): we didn't overfit the validation set by choosing among 7 models.

## Extend it
1. Replace the two hand-made distance features with **k-means cluster ids of (lat, lon)**, one-hot encoded.
2. Try `XGBoostLike(objective="reg:squarederror")` from `mlscratch.ensemble` and compare with gradient boosting.
3. Use `GradientBoostingRegressor(loss="absolute")`. It optimises MAE, which is more robust to the noisy top end.
4. Instead of dropping capped rows, model them with a **censored (Tobit) loss**. That's a nice maths exercise.
