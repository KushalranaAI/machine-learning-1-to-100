---
title: Polynomial & Non-linear Regression
order: 2
summary: Fit curves with polynomial basis expansions (still linear least squares!), compare with kNN and tree regression, and choose model complexity by cross-validation.
level: intermediate
minutes: 40
why: Real relationships curve. Basis expansions let the trusted linear-regression machinery fit curves, and comparing them with kNN and trees shows the different ways models can be flexible - and overfit.
background: Polynomial fitting was used by Gergonne (1815). Splines (Schoenberg, 1946) and later kernel methods generalise the same "linear in the parameters, non-linear in the inputs" idea.
use_cases:
  - Dose-response and growth curves
  - Physical relationships (drag ∝ velocity²)
  - Any curved trend where interpretability matters
where_used:
  - 01-house-prices
prerequisites:
  - 07-regression/01-linear-regression
  - 06-model-evaluation/04-bias-variance-tradeoff
code:
  - mlscratch/preprocessing.py:PolynomialFeatures
  - mlscratch/neighbors.py:KNeighborsRegressor
  - mlscratch/tree.py:DecisionTreeRegressor
demo: demo.py
widgets:
  - bias-variance
cons:
  - High-degree polynomials oscillate wildly at the edges (Runge's phenomenon) and extrapolate terribly
workflow: |
  flowchart LR
    A[x] --> B[basis expansion: 1, x, x², …, xᵖ]
    B --> C[ordinary least squares]
    C --> D[choose p by CV]
    A --> E[alternatives: kNN / tree regression]
    D & E --> F[compare validation RMSE]
---

## Linear in the parameters
$$
\hat y = w_0 + w_1x + w_2x^2 + \dots + w_px^p = \phi(x)^\top w, \qquad \phi(x) = [1, x, x^2, \dots, x^p]
$$
The model is non-linear in $x$ but **linear in $w$**, so the normal equation still applies with the expanded design
matrix $\Phi$: $\hat w = (\Phi^\top\Phi)^{-1}\Phi^\top y$. With several features, `PolynomialFeatures` adds all products
up to degree $p$ (interactions included).

## Choosing the degree
Higher $p$ ⇒ lower training error, always. Pick $p$ by **validation error** - the U-shaped curve from the bias-variance lesson.
Numerical tip: scale $x$ to $[-1, 1]$ first; $x^{10}$ of values around 1000 overflows and makes $\Phi^\top\Phi$ singular.
Even better: orthogonal bases (Legendre, Chebyshev) or splines.

## Runge's phenomenon
Fitting a high-degree polynomial through equally spaced points of a smooth function can oscillate wildly near the
interval's ends. Polynomials also **extrapolate** badly - $x^{9}$ explodes just outside the training range.

## Other non-linear regressors (preview)
- **kNN regression**: $\hat y(x) = $ mean of the $k$ nearest training targets - local, no formula, step-like.
- **Regression trees**: piecewise-constant predictions from recursive splits (Part 8).
- **Ensembles and neural networks**: Parts 9 and 11.

## Exercises
1. In the widget, find the degree where the test error is lowest for 15 and for 80 points.
2. Run the demo: which model extrapolates most sensibly beyond the training range?
3. Fit degree 15 with and without scaling x to [-1, 1]; compare the condition numbers.
