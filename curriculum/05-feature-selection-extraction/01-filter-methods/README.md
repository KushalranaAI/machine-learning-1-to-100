---
title: Filter Methods
order: 1
summary: Rank features by a statistic computed independently of any model - variance, correlation, ANOVA F, chi-square and mutual information.
level: intermediate
minutes: 35
why: Filters are fast, model-agnostic first passes that remove constant, redundant and irrelevant columns before expensive modelling - essential when you start with thousands of features (text, genomics, sensors).
background: The statistics are classic (Pearson, Fisher, Shannon's information theory, 1948); using them for selection became standard practice in text classification in the 1990s.
use_cases:
  - Removing constant / near-constant columns
  - Dropping one of each pair of highly correlated features
  - Keeping the top-k words for a text classifier (χ²)
  - Quick relevance ranking on a new dataset
where_used:
  - 04-spam-detector
  - 02-titanic-survival
code:
  - mlscratch/feature_selection.py:VarianceThreshold
  - mlscratch/feature_selection.py:correlation_filter
  - mlscratch/feature_selection.py:f_classif
  - mlscratch/feature_selection.py:chi2
  - mlscratch/feature_selection.py:mutual_info_classif
demo: demo.py
pros:
  - Very fast, scales to huge feature counts
  - Independent of the model
cons:
  - Scores each feature alone - misses interactions (XOR features look useless individually)
  - Must be computed on training data only
workflow: |
  flowchart LR
    A[All features] --> B[Variance threshold]
    B --> C[Correlation filter: drop redundant]
    C --> D[Score vs target: F / χ² / MI]
    D --> E[Keep top-k on TRAIN folds]
---

## The statistics
**Variance threshold** - a feature that never changes can't help: drop features with $\mathrm{Var}(x_j) \le \tau$.

**Correlation filter** - if $|r(x_i, x_j)| > 0.9$, keep one of the pair (they carry the same information).

**ANOVA F (numeric feature, categorical target):**
$$
F_j = \frac{\sum_k n_k(\bar x_{jk} - \bar x_j)^2/(K-1)}{\sum_k\sum_{i\in k}(x_{ij}-\bar x_{jk})^2/(n-K)}
$$
large F ⇒ class means are far apart relative to the spread within classes.

**χ² (non-negative counts, categorical target):** compares observed per-class feature totals with what independence
predicts: $\chi^2_j = \sum_k \frac{(O_{jk}-E_{jk})^2}{E_{jk}}$ - the standard way to pick words for text models.

**Mutual information** - how much knowing $x$ reduces uncertainty about $y$, capturing any (non-linear) dependence:
$$
I(X;Y) = \sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)\,p(y)} \;\ge 0, \quad = 0 \iff \text{independent}
$$
We estimate it by discretising $x$ into quantile bins.

## The blind spot: interactions
In an XOR problem ($y = 1$ iff exactly one of $x_1, x_2$ is positive) each feature alone is useless - every filter
gives it a score of ~0 - yet together they determine $y$ perfectly. Wrapper and embedded methods (next lesson) can see this.

## Exercises
1. Run the demo and check which method ranks the informative features highest.
2. Add an XOR pair to the data; how do the F-scores and MI rank it?
3. Why must the selection be done inside the cross-validation loop? (Hint: select on all data, then CV - the score is inflated.)
