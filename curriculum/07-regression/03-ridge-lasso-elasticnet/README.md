---
title: Ridge, Lasso & Elastic Net
order: 3
summary: Penalised least squares - Ridge's closed form and shrinkage, Lasso's soft-thresholding coordinate descent and sparsity, Elastic Net, and the Bayesian interpretation.
level: intermediate
minutes: 55
why: With many or correlated features, plain least squares overfits and its coefficients become unstable. Penalties trade a little bias for a large reduction in variance; Lasso additionally selects features by setting weights exactly to zero.
background: Ridge regression (Hoerl & Kennard, 1970; Tikhonov regularisation in maths, 1943). Lasso (Tibshirani, 1996). Elastic Net (Zou & Hastie, 2005) combines both. Coordinate descent made them fast for huge problems (Friedman et al., 2010 - glmnet).
use_cases:
  - Many features relative to samples (genomics, text)
  - Correlated predictors (economic indicators)
  - Automatic feature selection with Lasso
  - Default regularised baseline for regression
where_used:
  - 01-house-prices
prerequisites:
  - 07-regression/01-linear-regression
  - 06-model-evaluation/04-bias-variance-tradeoff
code:
  - mlscratch/linear_model.py:Ridge
  - mlscratch/linear_model.py:_soft_threshold
  - mlscratch/linear_model.py:ElasticNet
  - cpp/include/mlcpp/linear_regression.hpp
demo: demo.py
pros:
  - Ridge - always solvable, stable with collinearity, closed form
  - Lasso - sparse, interpretable models
cons:
  - Must standardise features (the penalty depends on scale)
  - Lasso picks one feature arbitrarily from a correlated group (Elastic Net fixes this)
workflow: |
  flowchart LR
    A[standardise X] --> B{penalty}
    B -->|L2: Ridge| C[w = XᵀX + αI ⁻¹ Xᵀy]
    B -->|L1: Lasso| D[coordinate descent + soft-threshold]
    B -->|L1+L2: Elastic Net| D
    C & D --> E[choose α by CV]
    E --> F[coefficient path, sparsity]
---

## Ridge (L2)
$$
\min_w\; \|y - Xw\|^2 + \alpha\|w\|_2^2
$$
Gradient $= -2X^\top(y - Xw) + 2\alpha w = 0 \Rightarrow$
$$
\boxed{\hat w_{ridge} = (X^\top X + \alpha I)^{-1}X^\top y}
$$
Adding $\alpha I$ makes the matrix invertible even when features are collinear - Ridge's original motivation.
(Centre $X$ and $y$ first so the intercept isn't penalised; that's what `Ridge.fit` does.)

**Shrinkage in the SVD basis:** with $X = USV^\top$,
$$
\hat w_{ridge} = \sum_j \frac{s_j^2}{s_j^2 + \alpha}\cdot\frac{u_j^\top y}{s_j}\,v_j
$$
Directions with small singular values (the noisy, poorly determined ones) are shrunk the most. As $\alpha\to\infty$, $w \to 0$.

## Lasso (L1)
$$
\min_w\; \frac{1}{2n}\|y - Xw\|^2 + \alpha\|w\|_1
$$
No closed form ($|w|$ isn't differentiable at 0), but **coordinate descent** solves it one weight at a time.
Fix all weights except $w_j$; with $r$ = current residual, the 1-D problem is
$$
\min_{w_j}\; \frac{1}{2n}\|r + x_jw_j^{old} - x_jw_j\|^2 + \alpha|w_j|
$$
whose exact solution is the **soft-threshold**:
$$
w_j = \frac{S(\rho_j, \alpha)}{z_j}, \quad \rho_j = \frac1n x_j^\top(r + x_jw_j^{old}),\quad z_j = \frac1n x_j^\top x_j, \quad S(\rho,\alpha) = \text{sign}(\rho)\max(|\rho|-\alpha, 0)
$$
If a feature's correlation with the residual $|\rho_j|$ is below $\alpha$, its weight becomes **exactly 0**. Cycle
through features until nothing changes - `ElasticNet.fit` is this loop, ~20 lines.

**Why corners give zeros:** the L1 ball $\{w : \|w\|_1 \le t\}$ is a diamond; the elliptical loss contours usually
touch it at a corner, where some coordinates are 0. The L2 ball is round - no corners, so Ridge shrinks but never zeroes.

## Elastic Net
$$
\min_w \frac{1}{2n}\|y - Xw\|^2 + \alpha\,\rho\,\|w\|_1 + \frac{\alpha(1-\rho)}{2}\|w\|_2^2
$$
Coordinate update: $w_j = S(\rho_j, \alpha\rho)/(z_j + \alpha(1-\rho))$. Groups of correlated features enter together instead of one being picked arbitrarily.

## Bayesian view
Ridge = MAP estimate with a Gaussian prior $w_j \sim \mathcal N(0, \tau^2)$; Lasso = MAP with a Laplace prior
(sharply peaked at 0). $\alpha$ = noise variance / prior variance.

## Choosing α
Standardise features, try a log-spaced grid, pick α by cross-validation. Plot the **coefficient path** (coefficients vs α).

## Exercises
1. Run the demo; compare OLS with the CV-tuned Ridge and Lasso. Which extra features does Lasso keep, and how large are their weights?
2. Verify the SVD shrinkage formula numerically against `Ridge.fit`.
3. Create 3 almost-identical features; compare how Lasso and Elastic Net distribute the weight.
