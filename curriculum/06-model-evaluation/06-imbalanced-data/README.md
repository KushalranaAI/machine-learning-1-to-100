---
title: Imbalanced Data
order: 6
summary: When positives are rare - the right metrics, class weights, random over/under-sampling, SMOTE from scratch, and threshold moving.
level: intermediate
minutes: 40
why: Fraud, disease, churn and defects are rare by nature. A model can reach 99% accuracy by never predicting the rare class - and be worthless. Handling imbalance is essential for most real classification problems.
background: SMOTE (Chawla et al., 2002) is the best-known oversampling method. Cost-sensitive learning and threshold moving are older and often just as effective.
use_cases:
  - Fraud detection (≪1% fraud)
  - Medical diagnosis of rare conditions
  - Predictive maintenance (failures are rare)
  - Spam detection (moderately imbalanced)
where_used:
  - 04-spam-detector
  - 02-titanic-survival
code:
  - mlscratch/metrics.py:precision_recall_curve
  - mlscratch/model_selection.py:StratifiedKFold
demo: demo.py
workflow: |
  flowchart LR
    A[Rare positive class] --> B[Stratified splits]
    B --> C[Metric: recall, precision, F1, PR-AUC]
    C --> D{Fix}
    D --> E[Class weights in the loss]
    D --> F[Resample TRAIN only: under / over / SMOTE]
    D --> G[Move the decision threshold]
    E & F & G --> H[Compare on an untouched, imbalanced test set]
---

## Step 1: measure correctly
Use stratified splits, and recall / precision / F1 / PR-AUC instead of accuracy. Keep the **test set imbalanced** -
it must look like reality.

## Step 2: change the training signal
**Class weights** - multiply each sample's loss by a weight inversely proportional to its class frequency,
$w_k = \frac{n}{K\,n_k}$. The rare class's errors count more, pushing the boundary towards the majority.

**Random undersampling** - drop majority samples (fast; loses information).
**Random oversampling** - duplicate minority samples (no information loss; can overfit duplicates).

**SMOTE** - create *synthetic* minority samples by interpolating between a minority point and one of its $k$ nearest minority neighbours:
$$
x_{new} = x_i + u\,(x_{nn} - x_i), \qquad u \sim U(0,1)
$$
Resample **only the training folds**, never validation or test data (otherwise the scores are fiction).

## Step 3: move the threshold
A model trained on imbalanced data often ranks well (good AUC) but its default 0.5 threshold is too high. Choosing the
threshold that maximises F1 (or minimises business cost) on validation data is simple and frequently the most effective fix.

## Exercises
1. Run the demo: which technique gives the best F1? the best recall?
2. Combine SMOTE with threshold moving.
3. Why does resampling before splitting leak information? Demonstrate it.
