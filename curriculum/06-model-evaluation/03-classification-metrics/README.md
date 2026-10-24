---
title: Classification Metrics
order: 3
summary: Confusion matrix, accuracy, precision, recall, F1, ROC curve and AUC, precision-recall curve, log-loss, calibration, and multi-class averaging.
level: beginner
minutes: 45
why: "95% accurate" means nothing if 95% of emails are not spam. The confusion matrix and the metrics built on it tell you what kinds of mistakes a model makes - and the threshold lets you choose which mistakes you can afford.
background: Precision and recall come from information retrieval (1950s-60s). ROC analysis was developed for radar signal detection in WWII and adopted by medicine and then ML.
use_cases:
  - Spam, fraud and disease detection (rare positives)
  - Ranking problems (AUC measures ranking quality)
  - Choosing an operating threshold from business costs
where_used:
  - 02-titanic-survival
  - 04-spam-detector
  - 05-digit-recognizer
code:
  - mlscratch/metrics.py:confusion_matrix
  - mlscratch/metrics.py:precision_score
  - mlscratch/metrics.py:roc_curve
  - mlscratch/metrics.py:roc_auc_score
  - mlscratch/metrics.py:log_loss
demo: demo.py
widgets:
  - roc
workflow: |
  flowchart LR
    A[Model scores / probabilities] --> B[Choose threshold t]
    B --> C[Confusion matrix TP FP FN TN]
    C --> D[precision, recall, F1, accuracy]
    A --> E[Sweep all thresholds]
    E --> F[ROC curve + AUC, PR curve]
    D & F --> G[Pick t from the cost of FP vs FN]
---

## The confusion matrix
| | predicted + | predicted − |
|-|-------------|-------------|
| **actual +** | TP (hit) | FN (miss) |
| **actual −** | FP (false alarm) | TN |

| Metric | Formula | Question it answers |
|--------|---------|--------------------|
| Accuracy | $\frac{TP+TN}{n}$ | how often are we right? (misleading when imbalanced) |
| **Precision** | $\frac{TP}{TP+FP}$ | when we say "positive", how often is it true? |
| **Recall** (sensitivity, TPR) | $\frac{TP}{TP+FN}$ | what fraction of real positives do we catch? |
| Specificity (TNR) | $\frac{TN}{TN+FP}$ | what fraction of negatives do we leave alone? |
| FPR | $\frac{FP}{FP+TN}$ | false-alarm rate |
| **F1** | $\frac{2PR}{P+R}$ | balance of P and R (harmonic mean punishes a low one) |
| $F_\beta$ | $\frac{(1+\beta^2)PR}{\beta^2P+R}$ | β > 1 favours recall |

**Which matters?** Cancer screening: missing a case (FN) is far worse → high recall. Spam filter: deleting a real
email (FP) is worse → high precision.

## Thresholds, ROC and AUC
Most classifiers output a score; the threshold turns it into a label. Sweeping the threshold traces the **ROC curve**
(TPR vs FPR). The **AUC** has a beautiful meaning:
$$
\text{AUC} = P(\text{score of a random positive} > \text{score of a random negative})
$$
0.5 = random ranking, 1.0 = perfect. It is threshold-free and insensitive to class balance - which also means it can
look good while precision at any useful threshold is poor on very rare positives. Then use the
**precision-recall curve** and its area (average precision).

## Log-loss and calibration
$\text{log-loss} = -\frac1n\sum[y\log p + (1-y)\log(1-p)]$ scores the probabilities themselves; a confident wrong
prediction costs a lot. A model is **calibrated** if among cases with $p \approx 0.8$, about 80% are positive.
Check with a reliability plot (demo).

## Multi-class averaging
- **macro**: compute the metric per class, average equally (every class matters equally).
- **weighted**: average weighted by class support.
- **micro**: pool all TP/FP/FN (equals accuracy for single-label problems).

## Choosing the threshold from costs
If a false negative costs $c_{FN}$ and a false positive $c_{FP}$, pick the $t$ minimising $c_{FN}\cdot FN(t) + c_{FP}\cdot FP(t)$
on validation data. With calibrated probabilities the optimum is $t^* = \frac{c_{FP}}{c_{FP}+c_{FN}}$.

## Exercises
1. In the widget, lower the positive rate to 2% and see accuracy vs F1 diverge.
2. Run the demo and verify that AUC equals the fraction of correctly ordered (positive, negative) pairs.
3. Find the cost-optimal threshold when a missed fraud costs 50× a false alarm.
