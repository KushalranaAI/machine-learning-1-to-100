---
title: Bagging & Random Forests
order: 1
summary: Bootstrap aggregation, out-of-bag error, random feature subsets at each split, why decorrelated trees generalise, and feature importances.
level: intermediate
minutes: 45
why: A single deep tree overfits; hundreds of decorrelated deep trees averaged together are among the most accurate, robust and low-maintenance models available - a strong default for almost any tabular problem.
background: Breiman introduced bagging in 1996 and random forests in 2001, building on Ho's random subspace method (1995).
use_cases:
  - Credit scoring, churn, fraud - strong tabular baselines
  - Feature importance and selection
  - Bioinformatics with many features
  - Out-of-the-box models when tuning time is limited
where_used:
  - 02-titanic-survival
  - 01-house-prices
prerequisites:
  - 08-classification/06-decision-trees
  - 06-model-evaluation/04-bias-variance-tradeoff
code:
  - mlscratch/ensemble.py:BaggingClassifier
  - mlscratch/ensemble.py:RandomForestClassifier
  - mlscratch/ensemble.py:RandomForestRegressor
demo: demo.py
pros:
  - Accurate with little tuning; hard to overfit by adding trees
  - Free validation estimate (out-of-bag)
  - Parallelisable
cons:
  - Large models, slower prediction than one tree
  - Less interpretable than a single tree
  - Can't extrapolate beyond the training target range
workflow: |
  flowchart LR
    A[training data n rows] --> B[B bootstrap samples, n rows with replacement]
    B --> C[grow a deep tree on each; at every split try only √d random features]
    C --> D[average probabilities / vote]
    B --> E[out-of-bag rows ≈ 36.8% per tree → free validation score]
---

## Bagging = Bootstrap AGGregatING
1. Draw $B$ **bootstrap samples**: $n$ rows sampled **with replacement** from the training set.
2. Train one model (usually a deep, low-bias tree) on each.
3. Predict by averaging (regression) or voting / averaging probabilities (classification).

Each tree is noisy, but their errors partly cancel, so the ensemble has much lower variance with about the same bias.

## Out-of-bag (OOB) error
The probability that a given row is **not** drawn in a bootstrap sample of size $n$ is
$$
\left(1 - \frac1n\right)^n \xrightarrow{n\to\infty} e^{-1} \approx 0.368
$$
So every tree has ~37% unseen rows. Predict each row using only the trees that didn't see it - an honest validation
estimate without a separate split (`oob_score_`).

## Random forests: decorrelate the trees
If one feature is very strong, every bagged tree splits on it first and the trees become highly correlated - averaging
correlated trees helps little ($\rho\sigma^2$ term). Random forests add one rule: **at each split, consider only a random
subset of features** (`max_features`, typically $\sqrt d$ for classification, $d/3$ or all for regression). Trees become
more different, the average better.

## Hyper-parameters
| Parameter | Effect | Typical |
|-----------|--------|---------|
| `n_estimators` | more trees → lower variance, never overfits, slower | 200-1000 |
| `max_features` | lower → more decorrelation, more bias | √d (clf), 0.33-1.0 (reg) |
| `max_depth`, `min_samples_leaf` | limit individual tree complexity | often unlimited / 1-5 |

## Feature importance
Mean impurity decrease across all trees (fast, but biased towards high-cardinality features), or permutation
importance on OOB/validation data (fairer).

## Exercises
1. Run the demo: plot test accuracy vs number of trees for bagging and for a random forest.
2. Verify the 36.8% out-of-bag fraction empirically.
3. Set `max_features=1.0` (pure bagging) vs `"sqrt"`; compare tree correlation and accuracy.
