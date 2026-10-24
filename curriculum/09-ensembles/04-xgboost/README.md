---
title: XGBoost: Second-Order Boosting
order: 4
summary: The regularised objective, the second-order Taylor expansion, optimal leaf weights w* = −G/(H+λ), the split-gain formula with γ, and the engineering that makes XGBoost/LightGBM fast.
level: advanced
minutes: 50
why: XGBoost (and its cousins LightGBM and CatBoost) are the go-to tools for tabular ML in industry and competitions. Their core is one elegant derivation that you can implement in ~80 lines.
background: Tianqi Chen and Carlos Guestrin released XGBoost in 2014 (paper 2016); it dominated Kaggle for years. LightGBM (Microsoft, 2017) and CatBoost (Yandex, 2017) followed with histogram-based splits and ordered target statistics.
use_cases:
  - Most tabular competition-winning solutions
  - Credit risk, fraud, click-through rate, ranking in production
where_used:
  - 02-titanic-survival
  - 01-house-prices
prerequisites:
  - 09-ensembles/03-gradient-boosting
  - 00-math-foundations/02-calculus-gradients
code:
  - mlscratch/ensemble.py:XGBoostLike
demo: demo.py
pros:
  - Principled regularisation of tree structure (λ, γ)
  - Any twice-differentiable loss with the same split code
cons:
  - Many hyper-parameters; exact split search is slow on huge data (real libraries use histograms)
workflow: |
  flowchart LR
    A[current F] --> B[per-sample gradient gᵢ and Hessian hᵢ]
    B --> C[grow tree: split gain from G, H sums]
    C --> D[leaf weight w* = −G / H + λ]
    D --> E[F ← F + η·tree]
    E --> A
---

## The regularised objective
At round $t$ we add a tree $f_t$ to the current prediction $\hat y^{(t-1)}$:
$$
\mathcal{L}^{(t)} = \sum_i l\big(y_i,\ \hat y_i^{(t-1)} + f_t(x_i)\big) + \Omega(f_t), \qquad \Omega(f) = \gamma T + \tfrac{\lambda}{2}\sum_{j=1}^T w_j^2
$$
$T$ = number of leaves, $w_j$ = leaf weights. $\gamma$ charges for every extra leaf, $\lambda$ shrinks the weights.

## Second-order Taylor expansion
With $g_i = \partial_{\hat y} l(y_i, \hat y^{(t-1)})$ and $h_i = \partial^2_{\hat y} l(y_i, \hat y^{(t-1)})$:
$$
\mathcal{L}^{(t)} \approx \sum_i\Big[g_if_t(x_i) + \tfrac12 h_if_t(x_i)^2\Big] + \gamma T + \tfrac\lambda2\sum_j w_j^2 + \text{const}
$$
A tree assigns weight $w_j$ to every sample in leaf $I_j$, so group by leaf with $G_j = \sum_{i\in I_j}g_i$, $H_j = \sum_{i\in I_j}h_i$:
$$
\mathcal{L}^{(t)} \approx \sum_{j=1}^T\Big[G_jw_j + \tfrac12(H_j + \lambda)w_j^2\Big] + \gamma T
$$
## Optimal leaf weights and structure score
Each bracket is a 1-D quadratic in $w_j$. Setting its derivative to zero:
$$
\boxed{w_j^* = -\frac{G_j}{H_j + \lambda}}, \qquad \mathcal{L}^* = -\frac12\sum_j\frac{G_j^2}{H_j + \lambda} + \gamma T
$$
## The split gain
Splitting a leaf into left/right changes the score by
$$
\text{Gain} = \frac12\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma
$$
Split only if Gain > 0 - $\gamma$ is a built-in pruning threshold. Evaluating all thresholds uses the same sorted
cumulative-sum trick as CART, now on $g$ and $h$. `min_child_weight` requires $H \ge$ a minimum in each child.

## Losses through g and h
| Objective | $g_i$ | $h_i$ |
|-----------|-------|-------|
| squared error | $\hat y_i - y_i$ | 1 |
| logistic | $p_i - y_i$ | $p_i(1-p_i)$ |

With squared error and λ = 0, $w^* = $ mean residual - plain gradient boosting is a special case.

## What the real libraries add
- **Histogram split finding** (bucket each feature into ~256 bins) - O(bins) instead of O(n) per split.
- **Sparsity-aware splits** - learn a default direction for missing values.
- **Column subsampling**, **leaf-wise growth** (LightGBM), **ordered boosting** for categoricals (CatBoost), GPU training.

## Exercises
1. Run the demo: confirm that with squared loss and λ=0 the leaf weights equal the mean residuals.
2. Increase γ and count how many leaves the trees keep.
3. Add the "missing value default direction" to `_build`: try sending NaNs left and right, keep the better gain.
