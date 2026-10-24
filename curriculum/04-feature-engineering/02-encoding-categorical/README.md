---
title: Encoding Categorical Variables
order: 2
summary: One-hot, ordinal, frequency and target encoding - when each is right, the dummy-variable trap, high cardinality, unseen categories and target leakage.
level: beginner
minutes: 35
why: Models need numbers, but "city", "plan" and "colour" are labels. The wrong encoding invents fake orderings (red < green < blue) or leaks the target; the right one can be the biggest single improvement to a model.
background: Dummy (indicator) variables are as old as regression. Target encoding was formalised by Micci-Barreca (2001) and is the core idea behind CatBoost's ordered target statistics.
use_cases:
  - Country, product category, plan type, colour
  - Ordinal levels like education or "low/medium/high"
  - High-cardinality IDs (zip code, merchant) via target or frequency encoding
where_used:
  - 02-titanic-survival
  - 01-house-prices
code:
  - mlscratch/preprocessing.py:OneHotEncoder
  - mlscratch/preprocessing.py:OrdinalEncoder
  - mlscratch/preprocessing.py:TargetEncoder
demo: demo.py
workflow: |
  flowchart TD
    A{Categorical feature} -->|natural order| B[Ordinal encoding with explicit order]
    A -->|no order, few levels| C[One-hot]
    A -->|many levels| D{Target available?}
    D -->|yes| E[Target encoding inside CV folds + smoothing]
    D -->|no| F[Frequency / hashing]
---

## The encoders
| Encoding | Result for `colour ∈ {red, green, blue}` | Good for | Danger |
|----------|------------------------------------------|----------|--------|
| **One-hot** | 3 binary columns | nominal, few levels | many columns for high cardinality |
| **Ordinal** | red=0, green=1, blue=2 | truly ordered levels, trees | fake order for nominal data in linear/distance models |
| **Frequency** | share of rows with that value | high cardinality | different categories with the same frequency collide |
| **Target (mean)** | smoothed mean of $y$ for that value | high cardinality, strong signal | **target leakage** if done on the full data |

## One-hot and the dummy-variable trap
With an intercept, $K$ one-hot columns are perfectly collinear: they always sum to 1 (= the intercept column).
$X^\top X$ becomes singular. Drop one column (`drop_first=True`) for unregularised linear models; tree models and
regularised models don't need to.

## Target encoding with smoothing
For category $c$ with $n_c$ rows and mean target $\bar y_c$:
$$
\text{enc}(c) = \frac{n_c\,\bar y_c + m\,\bar y}{n_c + m}
$$
Rare categories are pulled towards the global mean $\bar y$; $m$ controls how strongly.
**Leakage**: if a row's own target contributes to its encoding, the model sees the answer. Fit the encoder on
training folds only (out-of-fold encoding) - the demo shows how badly the naive version overfits.

## Unseen categories at prediction time
New categories will appear in production. One-hot → all zeros (`handle_unknown="ignore"`), target encoding → global mean,
ordinal → a reserved value (-1).

## Exercises
1. Run the demo: compare naive vs out-of-fold target encoding on a random high-cardinality feature.
2. Encode `plan` as ordinal (basic < premium < enterprise) and as one-hot; when could each be better?
3. What happens to `OneHotEncoder` with 10 000 zip codes? Propose two alternatives.
