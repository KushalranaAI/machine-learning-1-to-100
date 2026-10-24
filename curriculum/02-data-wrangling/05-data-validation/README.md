---
title: Data Validation & Quality Checks
order: 5
summary: Write explicit expectations about your data - schema, ranges, uniqueness, allowed values, freshness and distribution drift - and fail loudly when they break.
level: intermediate
minutes: 30
why: Data pipelines break silently - a column changes units, an upstream system starts sending nulls, a category is renamed. Validation catches these before they poison training or production predictions.
background: Software engineering has unit tests; data teams adopted the same idea as "data contracts" and tools like Great Expectations (2017), pandera and TensorFlow Data Validation.
use_cases:
  - Guarding a training pipeline against bad input files
  - Checking production inputs before predicting (train/serve skew)
  - Detecting data drift over time
where_used:
  - 02-titanic-survival
demo: demo.py
workflow: |
  flowchart LR
    A[New data batch] --> B[Schema: columns & dtypes]
    B --> C[Values: ranges, allowed sets, nulls]
    C --> D[Uniqueness & row counts]
    D --> E[Distribution vs reference: drift]
    E -->|all pass| F[Train / predict]
    E -->|fail| G[Stop + alert with a clear report]
---

## Types of checks
| Check | Example |
|-------|---------|
| **Schema** | columns present, dtypes as expected |
| **Completeness** | `customer_id` never null; `age` < 10% null |
| **Validity** | `0 < age < 110`; `plan ∈ {basic, premium, enterprise}`; dates not in the future |
| **Uniqueness** | `customer_id` is unique |
| **Consistency** | `end_date ≥ start_date`; totals add up |
| **Volume / freshness** | row count within ±20% of yesterday; newest record < 24 h old |
| **Distribution drift** | mean/quantiles or category shares close to the training data |

## Measuring drift
- Numeric: compare quantiles, or the **Population Stability Index**
  $\;PSI = \sum_b (p_b - q_b)\ln\frac{p_b}{q_b}$ over bins $b$ (rule of thumb: < 0.1 stable, 0.1-0.25 moderate, > 0.25 large shift).
- Categorical: compare category proportions; flag new, unseen categories.

## Make checks code
A validation function that returns a list of failures, run at the start of every training and prediction job.
Libraries: **pandera** (schemas as Python classes), **Great Expectations** (reports, many checks), **TFDV**.

## Exercises
1. Run the demo; then break the data on purpose (e.g. multiply ages by 12) and confirm the right check fails.
2. Add a "no future dates" check for `signup_date`.
3. Compute PSI between two months of a feature you care about.
