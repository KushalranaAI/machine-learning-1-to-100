---
title: Feature Scaling
order: 1
summary: Standardisation, min-max, robust scaling and power transforms - why distance- and gradient-based models need them, and why trees don't.
level: beginner
minutes: 30
why: Features measured in different units (dollars vs years) dominate distance calculations and make gradient descent zig-zag. Scaling puts features on a comparable footing and can turn a useless kNN or a non-converging logistic regression into a good model.
background: Standardisation (z-scores) comes from classical statistics. Its importance for optimisation follows from the condition number of the loss (see the gradient descent lesson).
use_cases:
  - Before kNN, k-means, SVM, PCA, neural networks
  - Before regularised linear models (so the penalty treats all weights fairly)
  - Not needed for decision trees and tree ensembles
where_used:
  - 01-house-prices
  - 03-customer-segmentation
  - 05-digit-recognizer
prerequisites:
  - 00-math-foundations/04-optimization-gradient-descent
code:
  - mlscratch/preprocessing.py:StandardScaler
  - mlscratch/preprocessing.py:MinMaxScaler
  - mlscratch/preprocessing.py:RobustScaler
  - mlscratch/preprocessing.py:PowerTransformer
demo: demo.py
widgets:
  - scaling
pros:
  - Huge accuracy gains for distance-based models
  - Faster, more stable gradient descent
cons:
  - Must be fitted on training data only
  - Makes coefficients harder to read in original units (keep the scaler to invert)
workflow: |
  flowchart LR
    A[X_train] -->|fit: μ, σ / min, max / median, IQR| S[Scaler]
    S -->|transform| B[X_train scaled]
    T[X_test] -->|transform with TRAIN stats| C[X_test scaled]
    B --> M[Model]
    C --> M
---

## The scalers
| Scaler | Formula | Output | Use when |
|--------|---------|--------|----------|
| **Standard** | $z = \frac{x-\mu}{\sigma}$ | mean 0, std 1 | default for most models |
| **Min-max** | $x' = \frac{x - \min}{\max - \min}$ | $[0, 1]$ | bounded inputs (pixels), neural nets |
| **Robust** | $x' = \frac{x - \text{median}}{IQR}$ | median 0 | outliers present |
| **Power (Yeo-Johnson / Box-Cox)** | $\lambda$ chosen by max likelihood | ≈ Gaussian | skewed features, linear models |
| **Log** | $\log(1+x)$ | compressed tail | positive, right-skewed (income, counts) |

## Who needs scaling?
| Model | Needs scaling? | Why |
|-------|----------------|-----|
| kNN, k-means, SVM (RBF) | **yes** | distances are dominated by large-scale features |
| PCA | **yes** | variance is dominated by large-scale features |
| Linear/logistic regression with GD | **yes** | ill-conditioned loss → slow or divergent training |
| Ridge / Lasso | **yes** | the penalty $\lambda\|w\|$ should treat features equally |
| Neural networks | **yes** | stable activations and gradients |
| Decision trees, random forests, boosting | no | splits depend only on the order of values |

## Why gradient descent cares (the math)
For linear regression, the Hessian of the MSE is $H = \frac{2}{n}X^\top X$. With features of scale 1 and 10 000,
its eigenvalues differ by ~$10^8$: the condition number $\kappa = \lambda_{max}/\lambda_{min}$ is huge, the
learning rate must be tiny ($\eta < 2/\lambda_{max}$), and progress along the flat direction is glacial.
Standardising makes $X^\top X/n$ close to the correlation matrix, with $\kappa$ near 1. The demo measures it.

## Implementation
`StandardScaler.fit` stores `mean_` and `scale_` (constant columns get scale 1); `transform` applies them;
`inverse_transform` undoes them - useful to report predictions or coefficients in original units.

## Exercises
1. In the widget, switch scalers with and without the outlier and watch the nearest neighbours change.
2. Run the demo; compare kNN accuracy and logistic-regression convergence with and without scaling.
3. Why is it a leak to scale before splitting into train and test? Try both and compare the test score.
