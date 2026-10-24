---
title: Hyper-parameter Tuning
order: 5
summary: Parameters vs hyper-parameters, grid search, random search, successive halving and Bayesian optimisation ideas, nested cross-validation and pipelines.
level: intermediate
minutes: 40
why: Defaults are rarely optimal. Systematic, cross-validated tuning of k, depth, learning rate or regularisation strength often gives large gains - if done without leaking the test set.
background: Grid search is the classic approach; Bergstra & Bengio (2012) showed random search is far more efficient when only a few hyper-parameters matter. Bayesian optimisation (e.g. Optuna, Hyperopt) and Hyperband (2017) are modern refinements.
use_cases:
  - Choosing k for kNN, C and γ for SVMs, depth and learning rate for boosting
  - Picking regularisation strength
  - Comparing preprocessing choices inside a pipeline
where_used:
  - 02-titanic-survival
  - 04-spam-detector
code:
  - mlscratch/model_selection.py:GridSearchCV
  - mlscratch/model_selection.py:RandomizedSearchCV
  - mlscratch/preprocessing.py:Pipeline
demo: demo.py
workflow: |
  flowchart LR
    A[Define search space] --> B{Strategy}
    B --> C[Grid: every combination]
    B --> D[Random: n samples]
    B --> E[Bayesian / halving: adaptive]
    C & D & E --> F[Each candidate: k-fold CV score]
    F --> G[Best params → refit on all training data]
    G --> H[Test set once]
---

## Parameters vs hyper-parameters
**Parameters** are learned by `fit` (weights, split thresholds, centroids). **Hyper-parameters** are chosen before
training (k, max_depth, λ, learning rate) and control *how* learning happens. They are chosen by validation performance.

## Search strategies
| Strategy | Idea | Cost | Good when |
|----------|------|------|-----------|
| Grid search | all combinations of listed values | $\prod_i |values_i|$ fits × k folds | 1-2 hyper-parameters |
| **Random search** | sample combinations from distributions | budget you choose | several hyper-parameters, few matter |
| Successive halving / Hyperband | many configs on small budgets, keep the best half, repeat | much cheaper | expensive models |
| Bayesian optimisation | model score = f(params), sample where improvement is likely | fewest fits | expensive fits, continuous spaces |

**Why random beats grid:** with 9 trials and 2 hyper-parameters where only one matters, a 3×3 grid tests only 3
distinct values of the important one; random search tests 9.

Sample scale-like parameters (λ, learning rate, C, γ) **log-uniformly**: 0.001, 0.01, 0.1, 1 matter equally.

## Nested cross-validation
The best CV score found by tuning is optimistically biased (you picked the max of many noisy estimates). For an
honest estimate of the *whole procedure*, wrap the tuning (inner CV) inside an outer CV loop - or keep a final test set.

## Tune pipelines, not models
Put preprocessing in a `Pipeline` and tune its parameters too (`"knn__n_neighbors"`, `"scale__..."`). The pipeline is
refit inside every fold, so nothing leaks.

## Exercises
1. Run the demo; compare the number of fits and the best scores of grid vs random search.
2. Implement successive halving on top of `cross_val_score`.
3. Show the optimism of the best inner-CV score compared with the nested-CV estimate.
