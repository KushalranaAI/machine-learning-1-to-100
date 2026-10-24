---
title: Transformations & Binning
order: 3
summary: Log, square-root, Box-Cox/Yeo-Johnson and target transforms; equal-width and quantile binning; when non-linear transforms help a linear model.
level: intermediate
minutes: 35
why: Linear models assume straight-line relationships and roughly symmetric errors. A log turns a multiplicative effect into an additive one, binning captures thresholds, and transforming a skewed target often cuts error dramatically.
background: Box and Cox proposed their power transform in 1964; Yeo and Johnson extended it to negative values in 2000.
use_cases:
  - Log of prices, incomes, counts and populations
  - Age groups, income brackets for interpretable models
  - Log-transforming a skewed regression target (house prices)
where_used:
  - 01-house-prices
  - 03-customer-segmentation
code:
  - mlscratch/preprocessing.py:PowerTransformer
  - mlscratch/preprocessing.py:KBinsDiscretizer
demo: demo.py
workflow: |
  flowchart LR
    A[Skewed feature / curved relation] --> B{Shape?}
    B -->|multiplicative, right-skew| C[log1p]
    B -->|counts| D[sqrt]
    B -->|unknown skew| E[Yeo-Johnson, λ by MLE]
    B -->|thresholds / steps| F[Quantile bins → one-hot]
    C & D & E & F --> G[Linear model fits better]
---

## Monotone transforms
| Transform | Formula | Effect |
|-----------|---------|--------|
| log | $\log(1+x)$ | compresses large values; multiplicative → additive |
| square root | $\sqrt{x}$ | milder; stabilises variance of counts (Poisson) |
| reciprocal | $1/x$ | very strong; reverses order |
| Box-Cox ($x>0$) | $\frac{x^\lambda - 1}{\lambda}$ (log when $\lambda=0$) | λ chosen to make data most Gaussian |
| Yeo-Johnson | Box-Cox extended to $x \le 0$ | works for any real values |

**Why log helps:** if $\text{price} = a \cdot \text{area}^{b} \cdot e^{\varepsilon}$, then
$\log \text{price} = \log a + b \log \text{area} + \varepsilon$ - a linear model in the logs, with additive noise.

## Transforming the target
For a right-skewed target (prices), fit on $\log(1+y)$ and predict $\exp(\hat z) - 1$. Squared error on the log scale
≈ relative error on the original scale - an error of \$20k matters more on a \$100k house than on a \$2M one. Kaggle's
House Prices competition scores exactly this (RMSE of log prices).

## Choosing λ by maximum likelihood
Yeo-Johnson picks the λ that maximises the Gaussian log-likelihood of the transformed data:
$$
\ell(\lambda) = -\frac{n}{2}\log\hat\sigma^2(\lambda) + (\lambda - 1)\sum_i \operatorname{sign}(x_i)\log(1+|x_i|)
$$
The second term is the log-Jacobian of the transform. `PowerTransformer.fit` searches a grid of λ values.

## Binning (discretisation)
- **Equal width**: bins of the same size - sensitive to outliers.
- **Equal frequency (quantile)**: each bin has ~the same number of samples.
- **Domain bins**: age groups, tax brackets.

Then one-hot encode the bins and a linear model can learn any step-shaped relationship. The cost: information
inside a bin is lost, and the edges are arbitrary. Trees learn bins by themselves.

## Exercises
1. Run the demo; compare a linear model's R² on raw vs log-transformed target.
2. Plot the transformed income for λ = 0, 0.5 and the fitted λ.
3. Bin age into quantiles and plot the churn rate per bin.
