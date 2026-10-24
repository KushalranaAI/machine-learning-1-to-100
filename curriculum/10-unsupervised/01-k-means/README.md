---
title: k-Means Clustering
order: 1
summary: Lloyd's algorithm, the inertia objective and why each step decreases it, k-means++ initialisation, choosing k (elbow, silhouette) and limitations.
level: beginner
minutes: 45
why: k-means is the most widely used clustering algorithm - fast, simple and surprisingly effective for customer segmentation, image compression and as a building block for other methods.
background: Lloyd devised the algorithm at Bell Labs in 1957 (published 1982); MacQueen named it k-means in 1967. Arthur and Vassilvitskii's k-means++ (2007) fixed its sensitivity to initialisation with a provable guarantee.
use_cases:
  - Customer segmentation (Project 3)
  - Colour quantisation / image compression
  - Vector quantisation, document grouping
  - Initialising Gaussian mixtures
where_used:
  - 03-customer-segmentation
prerequisites:
  - 04-feature-engineering/01-feature-scaling
code:
  - mlscratch/cluster.py:KMeans
  - cpp/include/mlcpp/kmeans.hpp
demo: demo.py
widgets:
  - kmeans
pros:
  - Fast - O(n·k·d) per iteration; scales to millions of points
  - Simple and interpretable centroids
cons:
  - Must choose k; finds only convex, similarly sized, roughly spherical clusters
  - Sensitive to initialisation, outliers and feature scale
workflow: |
  flowchart LR
    A[scale features] --> B[init k centroids: k-means++]
    B --> C[assign each point to nearest centroid]
    C --> D[move centroid to mean of its points]
    D --> E{assignments changed?}
    E -->|yes| C
    E -->|no| F[clusters + inertia]
    F --> G[repeat with n_init seeds, keep lowest inertia]
---

## The objective
Partition the data into $k$ clusters $C_1..C_k$ with centroids $\mu_1..\mu_k$ minimising the **inertia** (within-cluster sum of squares):
$$
J = \sum_{j=1}^k\sum_{x\in C_j}\|x - \mu_j\|^2
$$
## Lloyd's algorithm
1. **Initialise** $k$ centroids.
2. **Assign**: each point to its nearest centroid, $c(i) = \arg\min_j\|x_i - \mu_j\|^2$.
3. **Update**: each centroid to the mean of its points, $\mu_j = \frac{1}{|C_j|}\sum_{x\in C_j}x$.
4. Repeat 2-3 until assignments stop changing.

## Why it converges
- The assignment step can only lower $J$: each point moves to a closer centroid.
- The update step can only lower $J$: for a fixed set of points, $\sum\|x - \mu\|^2$ is minimised at $\mu = $ mean (derivative $-2\sum(x - \mu) = 0$).
- $J \ge 0$ and there are finitely many partitions, so the algorithm stops - at a **local** minimum. Hence multiple restarts (`n_init`).

## k-means++ initialisation
Pick the first centroid uniformly at random; pick each next one with probability proportional to $D(x)^2$, the squared
distance to the nearest centroid chosen so far. Spread-out starts avoid two centroids sharing a blob, and the expected
inertia is within $O(\log k)$ of the optimum.

## Choosing k
- **Elbow**: plot inertia vs k; look for the bend where adding clusters stops helping much.
- **Silhouette**: for each point, $s = \frac{b - a}{\max(a, b)}$ with $a$ = mean distance to its own cluster and $b$ = mean distance to the nearest other cluster. Average ≈ 1 is great, ≈ 0 means overlapping clusters.
- **Domain needs**: 4 marketing segments might simply be what the business can act on.

## Limitations
Assumes spherical clusters of similar size (it draws straight Voronoi boundaries), so it fails on moons, rings and very
elongated clusters. Scale your features first. Outliers drag centroids - consider k-medoids or removing outliers.

## Exercises
1. In the widget, re-initialise random init several times until you find a bad local minimum; then try k-means++.
2. Run the demo; which k do the elbow and silhouette methods suggest?
3. Compress an image to 8 colours with k-means on its RGB pixels.
