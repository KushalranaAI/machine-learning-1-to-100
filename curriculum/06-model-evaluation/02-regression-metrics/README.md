---
title: Regression Metrics
order: 2
summary: MSE, RMSE, MAE, R², adjusted R², MAPE and RMSLE - what each measures, which loss it corresponds to, and how to read residual plots.
level: beginner
minutes: 30
why: The metric defines what "good" means. RMSE punishes big misses, MAE treats all errors alike, R² compares with a trivial baseline, RMSLE cares about relative error. Choosing wrong optimises the wrong behaviour.
background: Least squares (Gauss/Legendre, ~1805) made squared error the default. Robust statistics (Laplace, then Huber 1964) motivated absolute and hybrid losses.
use_cases:
  - Price, demand and sales forecasting
  - Any continuous target
where_used:
  - 01-house-prices
code:
  - mlscratch/metrics.py:mean_squared_error
  - mlscratch/metrics.py:mean_absolute_error
  - mlscratch/metrics.py:r2_score
demo: demo.py
workflow: |
  flowchart LR
    A[Predictions ŷ] --> B[Residuals y - ŷ]
    B --> C[Aggregate: RMSE / MAE / R²]
    B --> D[Plot residuals vs ŷ: patterns = missing structure]
    C --> E[Compare with a baseline: predict the mean]
---

## The metrics
| Metric | Formula | Units | Optimal constant prediction | Sensitive to outliers |
|--------|---------|-------|------------------------------|------------------------|
| MSE | $\frac1n\sum(y-\hat y)^2$ | squared | mean | very |
| **RMSE** | $\sqrt{\text{MSE}}$ | same as $y$ | mean | very |
| **MAE** | $\frac1n\sum\lvert y-\hat y\rvert$ | same as $y$ | **median** | little |
| **R²** | $1 - \frac{\sum(y-\hat y)^2}{\sum(y-\bar y)^2}$ | none | - | very |
| Adjusted R² | $1-(1-R^2)\frac{n-1}{n-p-1}$ | none | - | very |
| MAPE | $\frac1n\sum\left\lvert\frac{y-\hat y}{y}\right\rvert$ | % | - | explodes near $y=0$ |
| RMSLE | $\sqrt{\frac1n\sum(\log(1+y)-\log(1+\hat y))^2}$ | ≈ relative | - | little |

**Why does MAE prefer the median?** Minimising $\sum|y_i - c|$ over a constant $c$: the derivative is
$\sum \text{sign}(c - y_i)$, zero when half the points are above and half below - the median. For squared error the
derivative $2\sum(c - y_i)$ is zero at the mean.

**R²** = fraction of the target's variance explained. 0 means "no better than predicting the mean"; it can be
**negative** on test data for a bad model. Adding any feature never lowers training R² - use adjusted R² or CV.

## Read the residuals
Plot residuals $y - \hat y$ against predictions:
- random cloud around 0 → good;
- a curve → missing non-linearity (add features / non-linear model);
- a funnel → error grows with the prediction (try a log target);
- a few huge residuals → outliers or data errors.

## Exercises
1. Run the demo: one outlier changes RMSE far more than MAE - by how much?
2. Show numerically that the constant minimising MAE is the median.
3. When would you report MAPE to a business stakeholder, and when would it mislead?
