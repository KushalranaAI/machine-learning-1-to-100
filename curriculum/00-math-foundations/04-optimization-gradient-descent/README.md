---
title: Optimisation & Gradient Descent
order: 4
summary: How models learn - by repeatedly stepping downhill on a loss surface. SGD, momentum, RMSProp and Adam from their update rules.
level: beginner
minutes: 45
why: Almost every model in this course is trained by minimising a loss function. Gradient descent is the engine; knowing how it behaves (learning rate, momentum, scaling) explains most training problems you will ever debug.
background: Cauchy proposed gradient descent in 1847. Stochastic gradient descent (Robbins & Monro, 1951) made it practical for big data, and adaptive methods (AdaGrad 2011, RMSProp 2012, Adam 2014) made deep learning trainable.
use_cases:
  - Training linear & logistic regression on millions of rows
  - Training every neural network (CNNs, transformers)
  - Gradient boosting (each tree is a gradient step in function space)
  - Fitting any differentiable model - recommendation, forecasting, physics
where_used:
  - 01-house-prices
  - 05-digit-recognizer
prerequisites:
  - 00-math-foundations/02-calculus-gradients
code:
  - mlscratch/optim.py:SGD
  - mlscratch/optim.py:Adam
  - mlscratch/optim.py:minimize
  - mlscratch/linear_model.py:LinearRegression.fit
demo: demo.py
widgets:
  - gradient-descent
pros:
  - Works for any differentiable loss, any number of parameters
  - Memory-cheap - only the gradient is needed
  - Mini-batches scale to datasets that don't fit in memory
cons:
  - Learning rate must be tuned - too big diverges, too small crawls
  - Converges to a local minimum on non-convex losses
  - Sensitive to feature scaling (elongated valleys)
workflow: |
  flowchart LR
    A[Initialise θ] --> B[Compute loss L θ on a batch]
    B --> C[Compute gradient ∇L θ]
    C --> D[Update θ ← θ - η·∇L]
    D --> E{Converged?}
    E -- no --> B
    E -- yes --> F[Trained model]
---

## The idea in one picture

Imagine standing on a foggy hillside and wanting to reach the valley floor. You can't see far, but you can feel
the slope under your feet. Take a step **downhill**, feel again, repeat. That's gradient descent.

The "hillside" is the **loss function** $L(\theta)$ - how wrong the model is with parameters $\theta$.
The slope is the **gradient** $\nabla L(\theta)$, the vector of partial derivatives, which points *uphill*.

## The update rule

$$
\theta_{t+1} = \theta_t - \eta \, \nabla L(\theta_t)
$$

- $\eta$ (eta) is the **learning rate** - the step size.
- Near the minimum the gradient shrinks, so steps shrink automatically.

### Why it works (one line of calculus)
A first-order Taylor expansion says $L(\theta - \eta g) \approx L(\theta) - \eta\, g^\top g = L(\theta) - \eta \|g\|^2$.
For small enough $\eta$ the loss **must** go down, because $\|g\|^2 \ge 0$.

### The learning rate decides everything
For the 1-D quadratic $L(\theta) = \tfrac{a}{2}\theta^2$, the update is $\theta_{t+1} = (1 - \eta a)\,\theta_t$:

| $\eta$ | behaviour |
|--------|-----------|
| $\eta < 1/a$ | smooth, monotone convergence |
| $1/a < \eta < 2/a$ | converges while oscillating around the minimum |
| $\eta > 2/a$ | **diverges** - every step overshoots further |

In $d$ dimensions the curvature differs per direction (the eigenvalues $\lambda_i$ of the Hessian). The largest
$\lambda_{max}$ limits the learning rate ($\eta < 2/\lambda_{max}$), and the smallest $\lambda_{min}$ sets how slowly
the flat direction converges. Their ratio, the **condition number** $\kappa = \lambda_{max}/\lambda_{min}$, is why
**unscaled features make training slow**: the loss becomes a long, narrow valley and GD zig-zags across it.
Standardising features (Part 4) makes the valley round.

## Batch, stochastic and mini-batch

For a dataset, the loss is an average over samples: $L(\theta) = \frac{1}{n}\sum_i \ell_i(\theta)$.

| Variant | Gradient from | Trade-off |
|---------|---------------|-----------|
| **Batch GD** | all $n$ samples | exact but slow per step |
| **Stochastic GD** | 1 random sample | noisy but cheap; noise can escape bad minima |
| **Mini-batch GD** | $b$ samples (32-512) | the standard: vectorised and still noisy enough |

## Momentum, RMSProp and Adam

Plain GD forgets the past. Better optimisers keep a running memory of gradients.

**Momentum** - a heavy ball that keeps rolling:
$$
v_t = \beta v_{t-1} + g_t, \qquad \theta_{t+1} = \theta_t - \eta\, v_t
$$
Oscillations across the valley cancel out; steps along the valley add up.

**RMSProp** - a per-parameter learning rate:
$$
s_t = \rho\, s_{t-1} + (1-\rho)\, g_t^2, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t} + \epsilon}\, g_t
$$
Directions with big gradients get smaller steps, so badly scaled problems behave better.

**Adam** - momentum + RMSProp + bias correction:
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,\quad
\hat m_t = \frac{m_t}{1-\beta_1^t},\ \hat v_t = \frac{v_t}{1-\beta_2^t},\quad
\theta_{t+1} = \theta_t - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$
The bias correction matters early on: $m$ and $v$ start at 0, so without it the first steps would be far too small.

## From scratch

Each optimiser in `mlscratch/optim.py` is a few lines that update a dictionary of parameters in place -
exactly the formulas above. The neural network (Part 11) uses the same objects.

```python
opt = Adam(lr=0.1)
params = {"theta": np.array([3.0, 2.0])}
for step in range(200):
    opt.step(params, {"theta": grad(params["theta"])})
```

## Common pitfalls
- **Loss goes up or becomes NaN** → learning rate too high, or features not scaled.
- **Loss decreases painfully slowly** → learning rate too low, or an ill-conditioned problem (scale features, use momentum/Adam).
- **Loss plateaus then drops** → saddle points; momentum helps.
- Always **plot the loss curve** - it is the single most useful debugging tool in ML.

## Exercises
1. In the interactive tab, find the largest learning rate that still converges on the elongated bowl. Compare it with $2/\lambda_{max}$.
2. Run the demo and compare how many steps each optimiser needs to get within 0.01 of the minimum.
3. Implement **Nesterov momentum** (evaluate the gradient at $\theta - \eta\beta v$) in `mlscratch/optim.py`.
