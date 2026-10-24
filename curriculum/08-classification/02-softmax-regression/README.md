---
title: Softmax (Multinomial) Regression
order: 2
summary: Generalise logistic regression to K classes with the softmax function and categorical cross-entropy - the output layer of every neural-network classifier.
level: intermediate
minutes: 35
why: Many problems have more than two classes (digits, species, product categories). Softmax regression handles them in one coherent probabilistic model, and its gradient is the same one that trains the last layer of every deep classifier.
background: The multinomial logit model was developed in econometrics (McFadden, 1974 - Nobel Prize 2000). The softmax name comes from it being a smooth version of argmax.
use_cases:
  - Multi-class classification (digits, species, categories)
  - The final layer of neural-network classifiers
  - Discrete choice modelling in economics
where_used:
  - 05-digit-recognizer
prerequisites:
  - 08-classification/01-logistic-regression
code:
  - mlscratch/base.py:softmax
  - mlscratch/linear_model.py:SoftmaxRegression
demo: demo.py
workflow: |
  flowchart LR
    A[x] --> B[K scores z = Wᵀx + b]
    B --> C[softmax → K probabilities]
    C --> D[cross-entropy with one-hot y]
    D --> E[gradient Xᵀ P − Y / n]
    E --> B
---

## The softmax function
$$
P(y = k\mid x) = \text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}, \qquad z_k = w_k^\top x + b_k
$$
All outputs are positive and sum to 1. With $K = 2$ it reduces to the sigmoid of $z_1 - z_0$.

**Numerical stability:** $e^{1000}$ overflows. Since softmax is unchanged by subtracting a constant from every score,
compute $\text{softmax}(z - \max_j z_j)$ - exactly what `mlscratch.base.softmax` does.

## Loss: categorical cross-entropy
With one-hot targets $Y \in \{0,1\}^{n\times K}$:
$$
L = -\frac{1}{n}\sum_i\sum_k Y_{ik}\log P_{ik}
$$
## Gradient
For one sample, $\frac{\partial L}{\partial z_k} = P_k - Y_k$ (the softmax Jacobian cancels against the log, just like
sigmoid + log-loss). Therefore
$$
\nabla_W L = \frac{1}{n}X^\top(P - Y), \qquad \nabla_b L = \frac1n\sum_i (P_i - Y_i)
$$
## Softmax vs one-vs-rest
| | One-vs-rest logistic | Softmax |
|-|----------------------|---------|
| models | K independent binaries | one joint model |
| probabilities | don't sum to 1 (normalised afterwards) | sum to 1 by construction |
| when better | classes not mutually exclusive (multi-label) | exactly one class per sample |

## Exercises
1. Show that softmax with K = 2 equals a sigmoid of the score difference.
2. Run the demo: compare softmax with one-vs-rest on the digits data.
3. Add a "temperature" $T$ (softmax(z/T)); what happens to the probabilities as $T \to 0$ and $T \to \infty$?
