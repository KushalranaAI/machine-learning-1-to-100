---
title: Linear Regression (OLS)
order: 1
summary: The model, the least-squares loss, the normal equation derived step by step, gradient descent, the probabilistic view, assumptions, interpretation and diagnostics.
level: beginner
minutes: 60
why: Linear regression is the simplest useful model and the foundation for almost everything else - logistic regression, neural networks and boosting all reuse its ideas. It is also still the right choice whenever interpretability and robustness matter.
background: Legendre (1805) and Gauss (1809) invented least squares to fit planetary orbits. Galton coined "regression" (1886) studying heights of parents and children ("regression to the mean").
use_cases:
  - House price, salary and demand prediction
  - Estimating the effect of a factor while controlling for others (economics, medicine)
  - Baseline model for any regression problem
  - Trend lines and forecasting components
where_used:
  - 01-house-prices
prerequisites:
  - 00-math-foundations/01-linear-algebra
  - 00-math-foundations/02-calculus-gradients
code:
  - mlscratch/linear_model.py:LinearRegression
  - cpp/include/mlcpp/linear_regression.hpp
demo: demo.py
widgets:
  - linear-regression
pros:
  - Fast, closed-form solution, no hyper-parameters
  - Coefficients are directly interpretable
  - Works well with little data when the relationship is roughly linear
cons:
  - Only straight-line (hyperplane) relationships unless you engineer features
  - Sensitive to outliers (squared loss) and to collinear features
workflow: |
  flowchart LR
    A[X n×d, y] --> B[add bias column]
    B --> C{n·d small?}
    C -->|yes| D[solve XᵀX w = Xᵀy  via lstsq]
    C -->|huge data| E[gradient descent on MSE]
    D & E --> F[w, b]
    F --> G[ŷ = Xw + b]
    G --> H[RMSE, R², residual plots]
---

## 1. The model
$$
\hat y = w_1x_1 + w_2x_2 + \dots + w_dx_d + b = w^\top x + b
$$
Trick: append a constant feature $x_0 = 1$ so the intercept becomes an ordinary weight: $\hat y = Xw$ with
$X \in \mathbb{R}^{n\times(d+1)}$. (`add_bias` in `mlscratch/base.py`.)

**Interpretation:** $w_j$ = change in $\hat y$ when $x_j$ increases by one unit **with all other features held fixed**.

## 2. The loss
$$
L(w) = \frac{1}{n}\sum_{i=1}^n (y_i - x_i^\top w)^2 = \frac{1}{n}\|y - Xw\|^2
$$
Why squares? Smooth and differentiable, a unique minimum (convex), a closed-form solution - and, as shown below, it is the
maximum-likelihood choice under Gaussian noise.

## 3. Solution A - the normal equation (derived)
Expand the loss: $\;\|y - Xw\|^2 = y^\top y - 2w^\top X^\top y + w^\top X^\top X w$.

Take the gradient using $\nabla_w(a^\top w) = a$ and $\nabla_w(w^\top A w) = 2Aw$ for symmetric $A$:
$$
\nabla_w L = \frac{1}{n}\left(-2X^\top y + 2X^\top X w\right)
$$
Set it to zero:
$$
\boxed{X^\top X\, w = X^\top y \quad\Rightarrow\quad \hat w = (X^\top X)^{-1}X^\top y}
$$
The Hessian $\frac{2}{n}X^\top X$ is positive semi-definite, so this stationary point is the global minimum.

**Geometry:** $X\hat w$ is the orthogonal projection of $y$ onto the space spanned by the feature columns; the residual
$y - X\hat w$ is orthogonal to every column ($X^\top(y - X\hat w) = 0$ is literally the normal equation).

**In practice** never invert $X^\top X$. `LinearRegression(solver="normal")` calls `np.linalg.lstsq`, which uses the SVD:
stable, and it still works when features are collinear and $X^\top X$ is singular. Cost: $O(nd^2 + d^3)$.

## 4. Solution B - gradient descent
When $d$ or $n$ is huge, iterate instead:
$$
w \leftarrow w - \eta\,\frac{2}{n}X^\top(Xw - y)
$$
Each step costs $O(nd)$; mini-batches make it scale to any data size. The loss is convex, so GD reaches the same
answer as the normal equation (the tests check this to 4 decimals) - provided the features are scaled and $\eta < 2/\lambda_{max}$.

## 5. The probabilistic view
Assume $y_i = x_i^\top w + \varepsilon_i$ with $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$ independent. The log-likelihood is
$$
\log p(y\mid X, w) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_i(y_i - x_i^\top w)^2
$$
Maximising it over $w$ is exactly minimising the squared error. This view also gives **standard errors**:
$\mathrm{Var}(\hat w) = \sigma^2(X^\top X)^{-1}$, so we can build confidence intervals and t-tests for each coefficient (demo).

## 6. Assumptions (and what breaks when they fail)
| Assumption | Check | If violated |
|------------|-------|-------------|
| Linearity | residuals vs fitted - no curve | add polynomial/log features, or a non-linear model |
| Independent errors | no pattern over time / groups | time-series or mixed models |
| Constant variance (homoscedasticity) | no funnel in residuals | log the target, weighted least squares |
| Normal errors | Q-Q plot | matters for CIs, not for predictions |
| No perfect multicollinearity | condition number, VIF | drop/merge features, use Ridge |

## 7. From scratch
`mlscratch/linear_model.py → LinearRegression` implements both solvers in ~30 lines; the C++ version in
`cpp/include/mlcpp/linear_regression.hpp` solves the normal equation with a Cholesky decomposition and also offers GD.

## Exercises
1. In the interactive tab, train with gradient descent and compare the final line with the normal-equation solution.
2. Run the demo; verify the residuals are orthogonal to every feature column.
3. Make two features nearly identical and watch the coefficients' standard errors explode.
