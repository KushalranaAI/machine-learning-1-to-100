---
title: Regression
icon: "📈"
description: Predict numbers - ordinary least squares from both the normal equation and gradient descent, polynomial and non-linear regression, and Ridge / Lasso / Elastic Net regularisation.
---

Regression learns a function $f: \mathbb{R}^d \to \mathbb{R}$ from examples $(x_i, y_i)$. Every algorithm here answers
the same three questions - and you'll see them in every later algorithm too:

| Question | Linear regression's answer |
|----------|---------------------------|
| **Model** - what functions can it represent? | $\hat y = w^\top x + b$ |
| **Loss** - how do we measure a bad fit? | mean squared error |
| **Optimiser** - how do we find good parameters? | closed form (normal equation) or gradient descent |

```mermaid
flowchart LR
  D[Training data] --> M[Model family]
  M --> L[Loss function]
  L --> O[Optimiser]
  O --> P[Fitted parameters]
  P --> E[Evaluate on new data]
```
