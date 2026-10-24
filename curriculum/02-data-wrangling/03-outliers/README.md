---
title: Outlier Detection & Treatment
order: 3
summary: Find unusual values with z-scores, the IQR rule, MAD, Mahalanobis distance and Isolation Forest - then decide whether to fix, cap, transform, keep or remove them.
level: intermediate
minutes: 40
why: A few extreme values can drag a regression line, inflate a standard deviation and dominate a squared-error loss. But outliers are also sometimes the most important data (fraud, failures) - handling them is a judgement call you must make deliberately.
background: Tukey introduced box-plot fences in 1977. Robust statistics (Huber, Hampel, 1960s-70s) studied estimators that resist outliers. Isolation Forest (Liu, Ting & Zhou, 2008) brought tree-based anomaly detection.
use_cases:
  - Catching data-entry errors (age 999, negative prices)
  - Protecting regression models from extreme points
  - Fraud and fault detection (where the outliers ARE the target)
where_used:
  - 01-house-prices
  - 03-customer-segmentation
code:
  - mlscratch/anomaly.py:iqr_outliers
  - mlscratch/anomaly.py:mahalanobis
  - mlscratch/anomaly.py:IsolationForest
demo: demo.py
widgets:
  - outliers
workflow: |
  flowchart TD
    A[Detect: plot + rules] --> B{Why is it extreme?}
    B -->|impossible value / typo| C[Fix or set to NaN]
    B -->|real but extreme| D{Does the model care?}
    D -->|squared loss, linear model| E[Cap / winsorise or log-transform]
    D -->|trees, robust loss| F[Keep]
    B -->|the thing we want to find| G[Keep - anomaly detection problem]
---

## Univariate rules
**z-score:** $z_i = \frac{x_i - \bar x}{s}$, flag $|z| > 3$. Simple, but $\bar x$ and $s$ are themselves pulled by the
outliers - with many outliers, $s$ inflates and the rule stops seeing them (**masking**).

**IQR rule (Tukey's fences):** flag $x < Q_1 - k\cdot IQR$ or $x > Q_3 + k\cdot IQR$, with $IQR = Q_3 - Q_1$, usually $k = 1.5$ (3 for "extreme").
Quartiles barely move when outliers are added, so this rule is robust.

**Modified z-score with MAD:** $\;M_i = 0.6745\,\frac{x_i - \text{median}}{\text{MAD}}$, $\text{MAD} = \text{median}(|x_i - \text{median}|)$, flag $|M_i| > 3.5$.

## Multivariate outliers
A 1.95 m tall person is fine, a 40 kg person is fine, a 1.95 m / 40 kg person is suspicious. Univariate rules miss this.

**Mahalanobis distance** measures distance in units of the data's own covariance:
$$
D_M(x) = \sqrt{(x-\mu)^\top \Sigma^{-1} (x-\mu)}
$$
Under a multivariate Gaussian, $D_M^2 \sim \chi^2_d$ - flag points above the 97.5% quantile.

**Isolation Forest** (Part 10) needs no distribution assumption: anomalies are isolated by fewer random splits.

## What to do with them
| Situation | Action |
|-----------|--------|
| Impossible value (age 999, negative price) | it's an error → set to NaN and impute |
| Unit/format mistake | fix it (inches → cm) |
| Real but extreme, model uses squared loss | **winsorise** (cap at the 1st/99th percentile) or **log-transform** |
| Real and extreme, tree-based model | usually keep - trees split on order, not distance |
| The outliers are what you're looking for | keep them - it's an anomaly-detection problem |

Always decide on the **training set** and apply the same caps to new data.

## Exercises
1. In the widget, inject 30 outliers and observe the masking effect on the z-score rule.
2. Run the demo and compare a linear regression slope with and without winsorising.
3. Compute Mahalanobis distances on (height, weight) and list the top 5 suspicious people.
