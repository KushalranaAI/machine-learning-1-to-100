---
title: Calculus & Gradients
order: 2
summary: Derivatives, partial derivatives, the gradient, the chain rule and the Hessian - everything needed to train a model by minimising its loss.
level: beginner
minutes: 40
why: Training a model means finding parameters that minimise a loss. The gradient tells you which way is downhill; the chain rule (backpropagation) computes it for models of any depth.
background: Newton and Leibniz invented calculus in the 1680s. Backpropagation - the chain rule applied systematically to neural networks - was popularised by Rumelhart, Hinton and Williams in 1986.
use_cases:
  - Deriving gradient-descent updates for regression and classification
  - Backpropagation in neural networks
  - Newton's method / second-order optimisers (XGBoost uses the Hessian)
  - Checking your own gradient code numerically
where_used:
  - 01-house-prices
  - 05-digit-recognizer
prerequisites:
  - 00-math-foundations/01-linear-algebra
code:
  - mlscratch/linear_model.py:LogisticRegression._fit_binary
  - mlscratch/neural_network.py:gradient_check
  - mlscratch/ensemble.py:XGBoostLike._grad_hess
demo: demo.py
workflow: |
  flowchart LR
    A[Model f x;θ] --> B[Loss L θ]
    B --> C[Derivatives ∂L/∂θᵢ]
    C --> D[Gradient ∇L]
    D --> E[Gradient descent step]
    C --> F[Chain rule → backprop]
---

## Derivatives: the slope
$$
f'(x) = \lim_{h\to 0} \frac{f(x+h) - f(x)}{h}
$$
Rules you'll use: $(x^n)' = n x^{n-1}$, $(e^x)' = e^x$, $(\ln x)' = 1/x$, product rule, and above all the **chain rule**:
$$
\frac{d}{dx} f(g(x)) = f'(g(x))\, g'(x)
$$

## Partial derivatives and the gradient
For $f(\theta_1, \dots, \theta_d)$, the partial derivative $\partial f/\partial \theta_i$ changes one variable and freezes the rest.
Stacking them gives the **gradient**:
$$
\nabla f(\theta) = \left[\frac{\partial f}{\partial \theta_1}, \dots, \frac{\partial f}{\partial \theta_d}\right]^\top
$$
The gradient points in the direction of **steepest increase**; $-\nabla f$ is steepest descent. At a minimum, $\nabla f = 0$.

## Matrix calculus you need (memorise these four)
| Function | Gradient w.r.t. $w$ |
|----------|-----------------------|
| $a^\top w$ | $a$ |
| $w^\top A w$ | $(A + A^\top) w$ (= $2Aw$ if $A$ symmetric) |
| $\|Xw - y\|^2$ | $2X^\top(Xw - y)$ |
| $\lambda\|w\|^2$ | $2\lambda w$ |

### Worked example 1 - MSE for linear regression
$L(w) = \frac{1}{n}\|Xw - y\|^2 \Rightarrow \nabla L = \frac{2}{n} X^\top (Xw - y)$ - exactly the line in `LinearRegression.fit(solver="gd")`.

### Worked example 2 - logistic regression (the beautiful cancellation)
With $p = \sigma(z)$, $z = w^\top x$, and loss $\ell = -[y\ln p + (1-y)\ln(1-p)]$:
- $\sigma'(z) = \sigma(z)(1-\sigma(z))$
- $\frac{\partial \ell}{\partial p} = -\frac{y}{p} + \frac{1-y}{1-p}$
- chain rule: $\frac{\partial \ell}{\partial z} = \frac{\partial \ell}{\partial p}\cdot p(1-p) = p - y$
- so $\nabla_w \ell = (p - y)\,x$ and over a dataset $\nabla_w L = \frac{1}{n}X^\top(p - y)$.

The same "prediction minus target" form appears for softmax + cross-entropy and is why these losses pair so well with these activations.

## The chain rule on a computation graph = backpropagation
A neural network is a composition $L = \ell(f_3(f_2(f_1(x))))$. Backprop computes
$\frac{\partial L}{\partial \theta}$ for every layer by walking the graph backwards, multiplying local derivatives.
Each layer only needs to know its own derivative (Part 11).

## Second derivatives: curvature and the Hessian
$$
H_{ij} = \frac{\partial^2 f}{\partial \theta_i \partial \theta_j}
$$
- $H$ positive definite at a point with $\nabla f = 0$ ⇒ local minimum. If $H$ is PSD everywhere, $f$ is **convex** - any local minimum is global (linear/logistic regression, SVM).
- **Newton's method** uses curvature: $\theta \leftarrow \theta - H^{-1}\nabla f$. XGBoost uses the per-sample second derivative $h_i$ of the loss to size each leaf: $w^* = -G/(H+\lambda)$.

## Checking gradients numerically
Bugs in hand-written gradients are common. Compare with central finite differences:
$$
\frac{\partial f}{\partial \theta_i} \approx \frac{f(\theta + \epsilon e_i) - f(\theta - \epsilon e_i)}{2\epsilon}
$$
`mlscratch.neural_network.gradient_check` does exactly this; relative error below $10^{-6}$ means your backprop is right.

## Exercises
1. Derive $\nabla_w$ of Ridge loss $\frac1n\|Xw-y\|^2 + \lambda\|w\|^2$ and solve $\nabla = 0$ for the closed form.
2. Derive $\sigma'(z) = \sigma(z)(1-\sigma(z))$ from $\sigma(z) = 1/(1+e^{-z})$.
3. Run the demo, then break the logistic gradient on purpose (drop the $1/n$) and see the gradient check fail.
