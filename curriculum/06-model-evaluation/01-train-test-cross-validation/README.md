---
title: Train/Test Splits & Cross-Validation
order: 1
summary: Hold-out splits, stratification, k-fold and stratified k-fold, time-series splits, group splits, nested CV - and the leakage they prevent.
level: beginner
minutes: 35
why: Training error is always optimistic. Without a clean estimate of performance on unseen data you can't compare models, tune hyper-parameters or know whether the model is ready for production.
background: Cross-validation was formalised by Stone (1974) and Geisser (1975). k-fold CV became the default model-selection tool in the 1990s.
use_cases:
  - Estimating generalisation error
  - Comparing algorithms and feature sets
  - Tuning hyper-parameters without touching the test set
where_used:
  - 01-house-prices
  - 02-titanic-survival
  - 04-spam-detector
code:
  - mlscratch/model_selection.py:train_test_split
  - mlscratch/model_selection.py:KFold
  - mlscratch/model_selection.py:StratifiedKFold
  - mlscratch/model_selection.py:cross_val_score
demo: demo.py
widgets:
  - cross-validation
workflow: |
  flowchart TD
    A[All data] --> B[Test set 15-20%: lock it away]
    A --> C[Development set]
    C --> D[k-fold CV: k models, k validation scores]
    D --> E[mean ± std → choose model]
    E --> F[Refit on all development data]
    F --> G[Evaluate ONCE on test]
---

## Hold-out split
Shuffle, then split into **train** (fit), **validation** (choose), **test** (final report). Typical: 60/20/20 or
80/20 with CV on the 80. **Stratify** classification splits so each part keeps the class proportions - essential when a class is rare.

## k-fold cross-validation
Split the development data into $k$ folds; train on $k-1$, validate on the remaining one, rotate. The CV estimate is
$$
\text{CV}_k = \frac{1}{k}\sum_{i=1}^k \text{score}(\text{model trained without fold } i,\ \text{fold } i)
$$
Report the mean **and the standard deviation** across folds. $k = 5$ or $10$ balances bias (small training sets)
against cost. Leave-one-out ($k = n$) has low bias but high variance and cost.

## Special splits
| Data | Split | Why |
|------|-------|-----|
| imbalanced classes | **stratified** k-fold | every fold has positives |
| time series | **forward-chaining** (expanding window) | never train on the future |
| repeated entities (patients, users) | **group** k-fold | the same person can't be in train and validation |
| small data + tuning | **nested** CV | inner loop tunes, outer loop evaluates the tuning procedure |

## Leakage through preprocessing
Any step that learns from data (scaling, imputation, feature selection, target encoding) must be fit **inside**
each training fold. `Pipeline` + `cross_val_score` does this automatically: the pipeline is cloned and refit per fold.

## Exercises
1. In the widget, switch between schemes and count how often each sample is used for validation.
2. Run the demo: compare the spread of hold-out estimates vs 5-fold CV estimates over many random seeds.
3. Show that feature selection before CV inflates the score on pure-noise data.
