---
title: Logistic Regression
order: 1
summary: Model the log-odds linearly, squash with the sigmoid, train by maximum likelihood (cross-entropy) with gradient descent; interpretation as odds ratios, regularisation and one-vs-rest.
level: beginner
minutes: 55
why: Logistic regression is the workhorse of binary classification - credit scoring, medicine, marketing - because it is fast, gives calibrated probabilities and interpretable coefficients. It is also a one-neuron neural network, so understanding it unlocks deep learning.
background: The logistic function was introduced by Verhulst (1838) for population growth. Berkson (1944) used it for dose-response modelling; Cox (1958) developed logistic regression as we know it.
use_cases:
  - Credit default and fraud risk scores
  - Disease risk from clinical measurements
  - Click-through and conversion prediction
  - Survival on the Titanic (Project 2)
where_used:
  - 02-titanic-survival
  - 04-spam-detector
prerequisites:
  - 00-math-foundations/02-calculus-gradients
  - 07-regression/01-linear-regression
code:
  - mlscratch/base.py:sigmoid
  - mlscratch/linear_model.py:LogisticRegression
  - cpp/include/mlcpp/logistic_regression.hpp
demo: demo.py
widgets:
  - logistic-regression
pros:
  - Probabilistic output, usually well calibrated
  - Convex loss → unique optimum, no local minima
  - Coefficients = log-odds ratios, easy to explain
cons:
  - Linear boundary unless you add features
  - Perfectly separable data → weights grow without bound (regularise!)
workflow: |
  flowchart LR
    A[x] --> B[z = wᵀx + b]
    B --> C[p = σ z]
    C --> D[loss: −y log p − 1−y log 1−p]
    D --> E[gradient Xᵀ p − y / n]
    E --> F[update w]
    F --> B
    C --> G[predict 1 if p ≥ threshold]
---

## 1. From a line to a probability
A linear score $z = w^\top x + b$ ranges over $(-\infty, \infty)$; a probability must be in $[0, 1]$.
Logistic regression models the **log-odds** linearly:
$$
\log\frac{p}{1-p} = w^\top x + b \quad\Longleftrightarrow\quad p = \sigma(z) = \frac{1}{1+e^{-z}}
$$
The decision boundary $p = 0.5$ is the hyperplane $w^\top x + b = 0$.

**Sigmoid facts:** $\sigma(0) = 0.5$, $\sigma(-z) = 1 - \sigma(z)$, $\sigma'(z) = \sigma(z)(1-\sigma(z))$.
`mlscratch.base.sigmoid` computes it in two branches so $e^{-z}$ never overflows for very negative $z$.

## 2. Maximum likelihood → cross-entropy
Each label is a Bernoulli draw: $P(y\mid x) = p^y(1-p)^{1-y}$. The negative log-likelihood of the dataset is
$$
L(w) = -\frac{1}{n}\sum_{i=1}^n \Big[y_i\log p_i + (1-y_i)\log(1-p_i)\Big]
$$
the **binary cross-entropy / log-loss**. Why not MSE? With a sigmoid, MSE is non-convex and its gradient vanishes when
the model is confidently wrong; cross-entropy is convex in $w$ and punishes confident mistakes hard.

## 3. The gradient (derived)
For one sample, with $z = w^\top x$:
$$
\frac{\partial \ell}{\partial p} = -\frac{y}{p} + \frac{1-y}{1-p}, \qquad \frac{\partial p}{\partial z} = p(1-p), \qquad \frac{\partial z}{\partial w} = x
$$
Chain rule: $\frac{\partial\ell}{\partial w} = \left(-\frac{y}{p} + \frac{1-y}{1-p}\right)p(1-p)\,x = (p - y)\,x$.
Over the dataset, with L2 regularisation $\frac{\lambda}{2n}\|w\|^2$:
$$
\nabla_w L = \frac{1}{n}X^\top(p - y) + \frac{\lambda}{n}w, \qquad \frac{\partial L}{\partial b} = \frac{1}{n}\sum_i(p_i - y_i)
$$
"Prediction minus target, times input" - the same shape as linear regression's gradient. There's no closed form, so we
iterate (gradient descent here; Newton's method / IRLS converges in a handful of steps because the Hessian
$\frac1n X^\top S X$, $S = \mathrm{diag}(p_i(1-p_i))$, is available).

## 4. Separable data and regularisation
If a hyperplane separates the classes perfectly, increasing $\|w\|$ always lowers the loss (probabilities → 0/1), so
unregularised weights diverge. The L2 penalty (`C = 1/λ`) keeps them finite and improves generalisation.

## 5. Interpreting coefficients
$e^{w_j}$ is an **odds ratio**: one unit more of $x_j$ multiplies the odds $\frac{p}{1-p}$ by $e^{w_j}$.
Example: $w_{\text{female}} = 2.5$ on Titanic ⇒ odds of survival ×12 for women, other features fixed.

## 6. More than two classes
**One-vs-rest:** train $K$ binary models ("class $k$ vs everything else"), predict the class with the highest score -
what `LogisticRegression` does for multiclass input. The principled alternative is softmax regression (next lesson).

## Exercises
1. In the widget, train on "moons" with degree 1, then degree 3 - why does the polynomial version work?
2. Run the demo; compare our gradient-descent weights with a Newton-method fit.
3. Remove the regularisation on a separable dataset and watch $\|w\|$ grow every epoch.
