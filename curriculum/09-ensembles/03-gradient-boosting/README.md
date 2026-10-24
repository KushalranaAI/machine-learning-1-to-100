---
title: Gradient Boosting
order: 3
summary: Boosting as gradient descent in function space - fit trees to pseudo-residuals, shrinkage, subsampling, losses for regression and classification, and Newton leaf values.
level: advanced
minutes: 60
why: Gradient-boosted trees are the single most effective algorithm family for tabular data (XGBoost, LightGBM, CatBoost). Understanding the core algorithm - which is surprisingly short - demystifies all of them.
background: Friedman's "Greedy Function Approximation - A Gradient Boosting Machine" (1999/2001) unified boosting under gradient descent; stochastic gradient boosting (2002) added subsampling.
use_cases:
  - Tabular regression and classification (credit, insurance, sales)
  - Ranking (search engines use boosted trees)
  - Kaggle-style competitions on structured data
where_used:
  - 01-house-prices
  - 02-titanic-survival
prerequisites:
  - 09-ensembles/02-adaboost
  - 00-math-foundations/04-optimization-gradient-descent
code:
  - mlscratch/ensemble.py:GradientBoostingRegressor
  - mlscratch/ensemble.py:GradientBoostingClassifier
demo: demo.py
widgets:
  - boosting
pros:
  - State-of-the-art accuracy on tabular data
  - Any differentiable loss; handles mixed feature types
cons:
  - Sequential training; several interacting hyper-parameters
  - Overfits if trees are too deep or too many rounds without shrinkage
workflow: |
  flowchart LR
    A[F₀ = best constant] --> B[pseudo-residuals rᵢ = −∂L/∂F xᵢ]
    B --> C[fit small regression tree hₘ to r]
    C --> D[optimal leaf values: line search / Newton step]
    D --> E[Fₘ = Fₘ₋₁ + η·hₘ]
    E --> B
    E --> F[stop by validation loss]
---

## Boosting = gradient descent over functions
We want $F(x)$ minimising $\sum_i L(y_i, F(x_i))$. Gradient descent on the vector of predictions $F = (F(x_1), \dots, F(x_n))$ would step
$$
F \leftarrow F - \eta\,\nabla_F L, \qquad (\nabla_F L)_i = \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}
$$
But that only defines $F$ at training points. So we **fit a regression tree** $h_m$ to the negative gradient (the
*pseudo-residuals*) - a function that generalises to new $x$ - and step along it.

## The algorithm
1. $F_0(x) = \arg\min_c \sum_i L(y_i, c)$ (the mean for squared loss, the log-odds for log-loss).
2. For $m = 1..M$:
   - pseudo-residuals $r_i = -\left[\frac{\partial L(y_i, F)}{\partial F}\right]_{F = F_{m-1}(x_i)}$
   - fit a small regression tree $h_m$ to $\{(x_i, r_i)\}$
   - for each leaf $R_j$, choose the best value $\gamma_j = \arg\min_\gamma\sum_{x_i\in R_j} L(y_i, F_{m-1}(x_i) + \gamma)$
   - update $F_m(x) = F_{m-1}(x) + \eta\,\gamma_{j(x)}$

## Losses and their pseudo-residuals
| Loss | $L(y, F)$ | pseudo-residual $r$ | leaf value |
|------|-----------|---------------------|------------|
| squared | $\tfrac12(y - F)^2$ | $y - F$ (the ordinary residual!) | mean of $r$ |
| absolute | $\lvert y - F\rvert$ | $\text{sign}(y - F)$ | median of $y - F$ |
| log-loss (binary) | $-[y\log p + (1-y)\log(1-p)]$, $p = \sigma(F)$ | $y - p$ | Newton step $\frac{\sum r_i}{\sum p_i(1-p_i)}$ |
| multinomial (K classes) | cross-entropy with softmax | $y_k - p_k$ | $\frac{K-1}{K}\frac{\sum r_i}{\sum\lvert r_i\rvert(1-\lvert r_i\rvert)}$ |

For log-loss there's no closed-form leaf optimum, so one **Newton step** is used: gradient over Hessian, summed over the leaf.
Our `GradientBoostingClassifier` does exactly this and matches scikit-learn's probabilities to 1e-10 for the first rounds.

## Regularisation knobs
| Knob | Effect |
|------|--------|
| learning rate η (shrinkage) | smaller steps → need more trees, generalise better (η ≈ 0.01-0.1) |
| `n_estimators` | number of rounds - choose by validation / early stopping |
| `max_depth` (3-8) | interaction order each tree can capture |
| `subsample` < 1 | stochastic gradient boosting: each tree sees a random fraction of rows (less variance, faster) |
| `min_samples_leaf` | smoother leaves |

## Bagging vs boosting
Random forests average **deep, independent** trees to cut variance; boosting adds **shallow, sequential** trees to cut bias.
Boosting usually wins on accuracy, forests on robustness and ease of tuning.

## Exercises
1. In the widget, step through the stages with η = 1 and η = 0.1 and compare how the residuals shrink.
2. Run the demo: find the number of rounds where validation loss is lowest (early stopping) for each learning rate.
3. Switch the regression loss to `absolute` and add outliers - which loss is more robust?
