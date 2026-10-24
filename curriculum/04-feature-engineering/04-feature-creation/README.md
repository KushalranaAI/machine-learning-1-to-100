---
title: Feature Creation & Leakage
order: 4
summary: Polynomial and interaction terms, ratios, date/time and cyclical features, group aggregates, domain knowledge - and how to recognise target leakage.
level: intermediate
minutes: 45
why: Models only see the columns you give them. A well-chosen ratio, interaction or aggregate often beats switching to a fancier algorithm - and leakage is the most common reason a model that looked great fails in production.
background: Polynomial regression dates back to Gergonne (1815). Kaggle competitions repeatedly showed that winning solutions owe more to feature engineering (and leak hunting) than to model choice.
use_cases:
  - Price per square metre, debt-to-income ratio, BMI
  - Day of week, hour, holidays, time since last event
  - Customer-level aggregates of transactions
  - Interactions such as area × quality
where_used:
  - 01-house-prices
  - 02-titanic-survival
code:
  - mlscratch/preprocessing.py:PolynomialFeatures
demo: demo.py
cons:
  - Polynomial features grow combinatorially (d features, degree p → C(d+p, p) columns)
  - Every feature must be computable at prediction time with the information available then
workflow: |
  flowchart LR
    A[Domain knowledge] --> B[Ratios & differences]
    C[Dates] --> D[parts + sin/cos cycles + durations]
    E[Transactions] --> F[groupby aggregates]
    G[Numeric features] --> H[interactions / polynomials]
    B & D & F & H --> I{Available at prediction time?}
    I -->|yes| J[Keep & validate with CV]
    I -->|no| K[LEAKAGE - drop it]
---

## Polynomials and interactions
A linear model on $[x_1, x_2]$ can only draw planes. Adding $x_1^2, x_1x_2, x_2^2$ lets it fit curves and
interactions while staying linear **in the parameters** - so it's still solved by least squares.
The number of degree-$\le p$ terms in $d$ features is $\binom{d+p}{p} - 1$: 10 features at degree 3 → 285 columns. Regularise (Ridge/Lasso).

## Domain features
Ask "what would an expert compute?":
- **Ratios**: price per m², debt-to-income, clicks per visit, fare per person
- **Differences**: days between signup and first purchase, temperature minus seasonal average
- **Counts / flags**: number of family members aboard, `has_garage`, `is_weekend`
- **Text-derived**: title from a name (Mr/Mrs/Master), length of a description

## Dates and cycles
Extract year, month, day of week, hour, is_holiday; durations (`now - last_purchase`).
Cyclical values (hour 23 is next to hour 0) should be encoded on a circle:
$$
\text{hour}_{\sin} = \sin\!\left(\frac{2\pi\cdot\text{hour}}{24}\right), \qquad \text{hour}_{\cos} = \cos\!\left(\frac{2\pi\cdot\text{hour}}{24}\right)
$$

## Group aggregates
"How does this customer compare to similar customers?" - `groupby(...).transform("mean")` features: income relative
to the city average, a product's price relative to its category median.

## Target leakage - the silent killer
A feature leaks if it contains information that **won't be available when you predict**, or that is derived from the target:
- `refund_issued` when predicting churn (refunds happen *after* churning),
- target/mean encodings fitted on the full dataset (previous lesson),
- scaling/imputation fitted on train + test,
- random train/test split on time-series data (training on the future),
- duplicate entities across train and test (the same patient in both).

Symptoms: suspiciously high validation scores, one feature with enormous importance, a big drop in production.

## Exercises
1. Run the demo; which created feature helps the most?
2. Encode month as 1-12 vs sin/cos; plot both encodings - which keeps December next to January?
3. Find the leak in the demo's "too good to be true" feature and explain it.
