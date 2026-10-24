---
title: Wrapper & Embedded Methods
order: 2
summary: Select features with a model in the loop - forward selection, recursive feature elimination, L1 (Lasso) sparsity, tree importances and permutation importance.
level: intermediate
minutes: 40
why: Filters judge features one at a time; wrappers and embedded methods judge them by how much they help an actual model, so they capture interactions and redundancy - at a higher computational cost.
background: Stepwise regression dates to the 1960s; RFE was introduced with SVMs for gene selection (Guyon et al., 2002); Lasso (Tibshirani, 1996) made selection part of training; permutation importance comes from Breiman's random forests (2001).
use_cases:
  - Finding a small, interpretable set of predictors
  - Reducing cost when features are expensive to collect
  - Explaining which inputs drive a model
where_used:
  - 01-house-prices
  - 02-titanic-survival
code:
  - mlscratch/feature_selection.py:SequentialFeatureSelector
  - mlscratch/feature_selection.py:RFE
  - mlscratch/feature_selection.py:permutation_importance
  - mlscratch/linear_model.py:ElasticNet
demo: demo.py
pros:
  - Account for interactions and redundancy
  - Optimise what you actually care about (validation score)
cons:
  - Expensive - many model fits
  - Can overfit the selection to the validation data (use nested CV)
workflow: |
  flowchart LR
    A[Features] --> W{Method}
    W -->|wrapper| B[Forward selection / RFE: fit model repeatedly]
    W -->|embedded| C[Lasso: weights driven to 0 / tree importances]
    W -->|model-agnostic| D[Permutation importance on validation data]
    B & C & D --> E[Subset → validate with CV]
---

## Wrapper methods
**Forward selection:** start empty; at each step add the feature that most improves the cross-validated score; stop at $k$ features or when nothing helps. Cost: $O(d \cdot k)$ model fits.

**Recursive Feature Elimination (RFE):** fit the model on all features, drop the least important (smallest $|w_j|$ or importance), refit, repeat. Cost: $O(d)$ fits.

## Embedded methods
**L1 regularisation (Lasso):**
$$
\min_w \frac{1}{2n}\|y - Xw\|^2 + \alpha\|w\|_1
$$
The L1 penalty has corners at 0, so the optimum often lies exactly on an axis: many $w_j = 0$. Coordinate descent
makes this explicit through the **soft-threshold** update $w_j = S(\rho_j, \alpha)/z_j$ with
$S(\rho, \alpha) = \text{sign}(\rho)\max(|\rho| - \alpha, 0)$ - any coordinate whose correlation with the residual is
below α is set to exactly zero. Increasing α drops features one by one (the "Lasso path").

**Tree importances:** total impurity decrease contributed by each feature across all splits (Part 9). Fast, but biased
towards high-cardinality and continuous features.

## Permutation importance (model-agnostic)
Shuffle one column in the **validation** set and measure how much the score drops. A feature the model relies on
causes a big drop; an unused or redundant one causes none. Works for any model, but correlated features share
(and hide) importance.

## Avoid selection bias
Selecting features on the full dataset and then cross-validating gives optimistic scores. Do the selection **inside**
each training fold (`Pipeline`), or keep a final untouched test set.

## Exercises
1. Run the demo and plot the Lasso path: at which α does each feature enter?
2. Compare RFE with logistic regression vs a decision tree as the inner model.
3. Duplicate a feature and recompute permutation importance - what happens to the two copies?
