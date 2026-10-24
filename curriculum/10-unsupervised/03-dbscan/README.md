---
title: DBSCAN
order: 3
summary: Density-based clustering - core, border and noise points, ε-neighbourhoods, density-reachability, choosing ε with the k-distance plot.
level: intermediate
minutes: 35
why: Real clusters are often irregular and data contains noise. DBSCAN finds clusters of any shape, labels outliers as noise, and doesn't need the number of clusters.
background: Ester, Kriegel, Sander and Xu introduced DBSCAN in 1996 (KDD test-of-time award 2014). OPTICS (1999) and HDBSCAN (2013) extend it to varying densities.
use_cases:
  - Geographic clustering (GPS points, crime hot-spots)
  - Anomaly detection (noise points)
  - Clusters with irregular shapes in feature space
where_used:
  - 03-customer-segmentation
code:
  - mlscratch/cluster.py:DBSCAN
demo: demo.py
widgets:
  - dbscan
pros:
  - Arbitrary cluster shapes, automatic number of clusters
  - Built-in noise detection
cons:
  - One global ε - struggles when clusters have very different densities
  - Sensitive to ε and minPts; needs scaled features
workflow: |
  flowchart LR
    A[scaled X] --> B[ε-neighbourhood of every point]
    B --> C[core points: ≥ minPts neighbours]
    C --> D[connect core points within ε → clusters]
    D --> E[attach border points]
    E --> F[everything else = noise −1]
---

## Definitions
- **ε-neighbourhood** of $p$: all points within distance $\varepsilon$ of $p$.
- **Core point**: has at least `minPts` points (itself included) in its ε-neighbourhood.
- **Border point**: not core, but inside the ε-neighbourhood of a core point.
- **Noise**: neither.
- $q$ is **density-reachable** from $p$ if there's a chain of core points $p = p_1, \dots, p_m = q$, each within ε of the next.

A cluster = a maximal set of density-connected points.

## Algorithm
For each unvisited core point, start a new cluster and expand it with a breadth-first search: add every neighbour; if a
neighbour is also a core point, continue expanding from it. Points never reached stay noise (label −1).
With a precomputed distance matrix this is $O(n^2)$; spatial indexes bring it to $\approx O(n\log n)$.

## Choosing parameters
- `minPts`: ≥ dimension + 1; 4-10 is common (higher = smoother, more noise).
- ε: the **k-distance plot** - sort every point's distance to its $k$-th nearest neighbour (k = minPts); the "knee"
  where it starts to shoot up is a good ε. A simple way to find it automatically: normalise both axes to [0, 1] and take the
  point farthest below the straight line joining the first and last points (the "kneedle" method).

## Exercises
1. In the widget, find ε and minPts that separate the two moons with few noise points.
2. Run the demo; read ε off the k-distance plot and compare with the clustering result.
3. Create two blobs with very different densities. Can any single ε handle both?
