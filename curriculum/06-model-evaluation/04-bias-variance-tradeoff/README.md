---
title: Bias-Variance Trade-off & Regularisation
order: 4
summary: Decompose expected error into bias², variance and noise; recognise under- and over-fitting from learning and validation curves; control complexity with regularisation.
level: intermediate
minutes: 45
why: Every modelling decision - depth of a tree, k in kNN, degree of a polynomial, strength of regularisation - moves along the bias-variance trade-off. Understanding it tells you whether to add complexity, add data or add regularisation.
background: The decomposition was popularised for ML by Geman, Bienenstock and Doursat (1992). Regularisation goes back to Tikhonov (1943, ridge-like penalties). "Double descent" (2019) revisits the picture for very large models.
use_cases:
  - Diagnosing why a model performs badly
  - Choosing model complexity and regularisation strength
  - Deciding whether more data will help
where_used:
  - 01-house-prices
  - 02-titanic-survival
code:
  - mlscratch/model_selection.py:learning_curve
  - mlscratch/linear_model.py:Ridge
demo: demo.py
widgets:
  - bias-variance
workflow: |
  flowchart TD
    A[Compare train vs validation error] --> B{Pattern}
    B -->|both high, close together| C[High bias: underfit → more features, more complex model, less regularisation]
    B -->|train low, validation high| D[High variance: overfit → more data, regularise, simpler model, ensembles]
    B -->|both low| E[Good - check on test]
---

## The decomposition
Suppose $y = f(x) + \varepsilon$ with noise variance $\sigma^2$, and $\hat f$ is learned from a random training set.
The expected squared error at a point $x$ (over training sets and noise) splits exactly into three parts:
$$
\mathbb{E}\big[(y - \hat f(x))^2\big] = \underbrace{\big(\mathbb{E}[\hat f(x)] - f(x)\big)^2}_{\text{bias}^2} + \underbrace{\mathbb{E}\big[(\hat f(x) - \mathbb{E}[\hat f(x)])^2\big]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{irreducible noise}}
$$
*Proof sketch:* add and subtract $\mathbb{E}[\hat f(x)]$ inside the square; the cross terms vanish because the noise
is independent with mean zero and $\hat f - \mathbb{E}\hat f$ has mean zero.

- **Bias**: error from wrong assumptions (a line fit to a curve). Simple models → high bias.
- **Variance**: how much the model changes with a different training sample. Flexible models → high variance.
- **Noise**: no model can remove it.

The demo estimates bias² and variance directly by training 300 polynomial models on 300 independent training sets.

## Diagnosing with curves
- **Validation curve** - error vs a complexity knob (degree, depth, 1/λ): training error falls monotonically, validation error is U-shaped.
- **Learning curve** - error vs training-set size: if both curves plateau high and close → bias (more data won't help); if there's a big gap → variance (more data will help).

## Regularisation: buying variance with bias
Ridge adds $\lambda\|w\|^2$ to the loss. In closed form $w = (X^\top X + \lambda I)^{-1}X^\top y$: larger $\lambda$ shrinks
the weights, makes the fit smoother and less sensitive to the particular sample (lower variance) at the cost of some
bias. Choose $\lambda$ by cross-validation. Other regularisers: L1 (sparsity), early stopping, dropout, max tree depth, min leaf size, bagging.

## Exercises
1. In the widget, find the degree with the lowest test error for 15 points, then for 80 points.
2. Run the demo; where do bias² and variance cross?
3. Fit Ridge with degree-15 features and increase λ - watch the overfit curve smooth out.
