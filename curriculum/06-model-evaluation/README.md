---
title: Model Evaluation & Selection
icon: "📏"
description: Measure models honestly - splits and cross-validation, the right metric for the job, the bias-variance trade-off, hyper-parameter tuning and imbalanced data.
---

A model is only as good as the evidence that it works on **data it has never seen**. This part comes *before* the
algorithms on purpose: every algorithm lesson uses these tools to judge itself.

```mermaid
flowchart LR
  D[(Data)] --> S[Split: train / validation / test]
  S --> T[Train candidates on train folds]
  T --> V[Score on validation - the right metric]
  V --> H[Tune hyper-parameters, compare models]
  H --> F[Final model: retrain on train+val]
  F --> X[Report ONE score on the untouched test set]
```

| Problem | Typical metrics |
|---------|-----------------|
| Regression | RMSE, MAE, R², MAPE |
| Balanced classification | accuracy, log-loss, ROC-AUC |
| Imbalanced classification | precision, recall, F1, PR-AUC, cost-weighted |
| Clustering | silhouette, inertia, adjusted Rand (if labels exist) |
