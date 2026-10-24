---
title: Feature Selection & Extraction
icon: "🎯"
description: Keep the features that matter (filter, wrapper, embedded selection) or build better ones (PCA, LDA, kernel PCA, t-SNE) - fewer, more informative inputs.
---

Two different ideas, often confused:

| | **Feature selection** | **Feature extraction** |
|-|------------------------|------------------------|
| Output | a **subset** of the original columns | **new** columns, combinations of the originals |
| Interpretable? | yes - same features | less - "PC1 = 0.6·area + 0.4·rooms…" |
| Examples | variance threshold, χ², mutual information, RFE, Lasso | PCA, LDA, kernel PCA, t-SNE, autoencoders |

Why reduce dimensions at all?
- **Curse of dimensionality**: in high dimensions all points are far apart, so distance-based methods weaken and you need exponentially more data.
- **Overfitting**: noise features give the model more ways to memorise the training set.
- **Speed and cost**: fewer features = faster training, cheaper data collection.
- **Visualisation**: you can only look at 2-3 dimensions.

```mermaid
flowchart TD
  A[Many features] --> B{Goal?}
  B -->|keep originals, interpretability| S[Selection]
  S --> F[Filter: statistics, fast]
  S --> W[Wrapper: search with a model, slow]
  S --> E[Embedded: L1, tree importance]
  B -->|compress / visualise| X[Extraction]
  X --> P[PCA: max variance, linear]
  X --> L[LDA: max class separation, supervised]
  X --> K[Kernel PCA / t-SNE: non-linear]
```
