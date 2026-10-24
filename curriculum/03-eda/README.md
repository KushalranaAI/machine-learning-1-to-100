---
title: Exploratory Data Analysis
icon: "🔍"
description: Understand the data before modelling it - distributions, relationships, correlations, statistical tests and the questions that shape your features.
---

EDA is a conversation with the data. Its goals:
1. **Understand** each variable (type, range, distribution, missingness).
2. **Find relationships** between features and with the target.
3. **Spot problems** - leakage, outliers, imbalance, data errors - before they reach the model.
4. **Generate hypotheses** for feature engineering.

```mermaid
flowchart LR
  U[Univariate: histograms, counts] --> B[Bivariate: feature vs target]
  B --> M[Multivariate: correlations, pair plots]
  M --> T[Test hypotheses: is the difference real?]
  T --> F[Ideas for features & model choice]
```
