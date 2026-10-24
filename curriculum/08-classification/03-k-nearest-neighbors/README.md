---
title: k-Nearest Neighbours
order: 3
summary: Classify by majority vote of the k closest training points - distance metrics, choosing k, weighting, the curse of dimensionality and fast neighbour search.
level: beginner
minutes: 40
why: kNN makes almost no assumptions about the shape of the boundary - "similar inputs have similar outputs" - so it's the simplest non-linear classifier, a great baseline, and the core of recommendation and similarity search.
background: Fix and Hodges introduced nearest-neighbour rules in 1951; Cover and Hart (1967) proved that with infinite data the 1-NN error is at most twice the optimal (Bayes) error.
use_cases:
  - Recommendation ("customers similar to you bought…")
  - Similarity search over embeddings (images, documents)
  - Imputation (KNNImputer) and anomaly detection
  - Quick non-linear baseline
where_used:
  - 05-digit-recognizer
  - 03-customer-segmentation
prerequisites:
  - 04-feature-engineering/01-feature-scaling
code:
  - mlscratch/neighbors.py:pairwise_distances
  - mlscratch/neighbors.py:KNeighborsClassifier
  - cpp/include/mlcpp/knn.hpp
demo: demo.py
widgets:
  - knn
pros:
  - No training, any boundary shape, naturally multi-class
  - Easy to update - just add data
cons:
  - Prediction is slow (compare with every training point) and memory-hungry
  - Needs feature scaling; suffers in high dimensions
workflow: |
  flowchart LR
    A[store X_train, y_train] --> B[new point x]
    B --> C[distances to all training points]
    C --> D[take k smallest]
    D --> E[majority / weighted vote]
    E --> F[class + vote share as probability]
---

## The algorithm
To classify $x$: compute $d(x, x_i)$ for every training point, take the $k$ nearest $N_k(x)$, and predict
$$
\hat y = \arg\max_c \sum_{i\in N_k(x)} w_i\,[y_i = c], \qquad w_i = 1 \text{ (uniform) or } \frac{1}{d(x, x_i)} \text{ (distance-weighted)}
$$
The vote shares serve as class probabilities. For regression, average the neighbours' targets instead.

## Distance metrics
| Metric | Formula | Notes |
|--------|---------|-------|
| Euclidean | $\sqrt{\sum_j (a_j-b_j)^2}$ | default; scale features first |
| Manhattan | $\sum_j \lvert a_j-b_j\rvert$ | more robust to single large differences |
| Minkowski | $(\sum_j \lvert a_j-b_j\rvert^p)^{1/p}$ | p=1 Manhattan, p=2 Euclidean |
| Cosine | $1 - \frac{a^\top b}{\|a\|\|b\|}$ | direction only - text, embeddings |

`pairwise_distances` computes the whole matrix at once with $\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2a^\top b$ - one matrix product instead of a double loop.

## Choosing k (bias-variance again)
- $k = 1$: memorises the training data - jagged boundary, high variance (training accuracy 100%).
- large $k$: smooth boundary, high bias; $k = n$ predicts the majority class everywhere.
Pick $k$ by cross-validation; odd $k$ avoids ties in binary problems.

## The curse of dimensionality
In $d$ dimensions, to capture a fraction $r$ of uniformly spread data a neighbourhood must span $r^{1/d}$ of each
axis: for 1% of the data in 100-D that is $0.01^{1/100} \approx 95\%$ of the range - "neighbours" are no longer local.
Distances also **concentrate**: the nearest and farthest points become almost equally far (demo). Remedies: feature
selection, PCA, metric learning.

## Making it fast
Brute force costs $O(nd)$ per query. KD-trees and ball trees prune the search in low dimensions; approximate
methods (locality-sensitive hashing, HNSW graphs as in vector databases) scale to millions of points.

## Exercises
1. In the widget (moons), move k from 1 to 51; when does the boundary become too smooth?
2. Run the demo and look at the distance-concentration table - why does kNN struggle in high dimensions?
3. Add weights="distance" and compare cross-validated accuracy on the digits.
