---
title: Feature Engineering
icon: "🛠️"
description: Turn clean data into informative model inputs - scaling, encoding categories, transformations, binning, interactions, date and text features - without leaking the target.
---

> "Coming up with features is difficult, time-consuming, requires expert knowledge. Applied machine learning is basically feature engineering." - Andrew Ng

A good feature makes the pattern **easy** for the model to learn. A linear model can't learn $\text{price} \propto \text{area} \times \text{quality}$ unless you give it that product; a distance-based model can't compare dollars with years unless you scale them.

```mermaid
flowchart LR
  C[Clean table] --> N[Numeric: scale, transform, bin]
  C --> K[Categorical: one-hot, ordinal, target encoding]
  C --> D[Dates: parts, cycles, durations]
  C --> T[Text: counts, TF-IDF]
  N & K & D & T --> X[Feature matrix X]
  X --> F[fit on TRAIN only → apply to test]
```

**Rule #1 of feature engineering:** every statistic (mean, scale, category list, target mean) is learned from the
**training data only**, then applied unchanged to validation/test data. Breaking this rule is called *leakage*.
