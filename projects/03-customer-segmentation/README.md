---
title: "Customer Segmentation"
level: Intermediate
summary: "Unsupervised learning on mall customers: choose k with four criteria, cluster with k-means, Ward, Gaussian mixtures and DBSCAN, measure agreement and stability, and turn clusters into marketing personas."
problem: |
  A shopping mall has a loyalty-card database: gender, age, annual income and a 1-100 **spending score**. There are
  no labels. Marketing wants a handful of **segments** they can target differently. There is no "accuracy" here. The
  hard questions are *how many* groups there are, whether they are *real* (stable, found by different methods), and
  whether they are *useful* (explainable, actionable).
dataset:
  name: "Mall Customer Segmentation Data"
  source: "Kaggle · vjchoudhary7/customer-segmentation-tutorial-in-python"
  url: "https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python"
  kaggle: "vjchoudhary7/customer-segmentation-tutorial-in-python"
  license: "Educational dataset. GitHub mirrors are used as a fallback"
  size: "200 customers × 5 columns · 4 KB"
algorithms: [K-Means, Agglomerative (Ward), Gaussian Mixture (EM), DBSCAN, PCA]
skills:
  - Scaling before distance-based methods
  - Choosing k with the elbow, silhouette, BIC and dendrogram gaps
  - Comparing clusterings with the adjusted Rand index (implemented from its formula)
  - Bootstrap stability analysis
  - Profiling clusters into named personas with business actions
lessons:
  - 04-feature-engineering/01-feature-scaling
  - 05-feature-selection-extraction/03-pca
  - 10-unsupervised/01-k-means
  - 10-unsupervised/02-hierarchical-clustering
  - 10-unsupervised/03-dbscan
  - 10-unsupervised/04-gaussian-mixtures-em
steps:
  - title: "1 · Download"
    script: download_data.py
    description: "200 rows from Kaggle or one of two GitHub mirrors. `common.load()` normalises the column names (`Genre` vs `Gender`)."
  - title: "2 · How many segments?"
    script: choose_k.py
    description: |
      For k = 2…10 on standardised (income, score) and on (age, income, score), we compute k-means **inertia**
      (elbow), the **silhouette** score, the **BIC** of a Gaussian mixture, and the **Ward merge heights**. On
      (income, score), all four point to **k = 5**: silhouette peaks at 0.555, BIC is lowest at 5, and the dendrogram's
      biggest relative jump (2.2×) comes right below 5 clusters.
  - title: "3 · Segment, validate, profile"
    script: segment.py
    description: |
      Four algorithms with k = 5 and their agreement (ARI), a 30× bootstrap stability check, cluster profiles with
      names and suggested actions, and a 3-feature (with age) clustering viewed through PCA. Writes `outputs/segments.csv`.
workflow: |
  flowchart LR
    A[Mall_Customers.csv] --> B[Standardise]
    B --> C{How many?<br/>elbow · silhouette · BIC · dendrogram}
    C -->|k = 5| D[k-means]
    C --> E[Ward] & F[GMM] & G[DBSCAN]
    D & E & F & G --> H[Agreement: ARI]
    D --> I[Bootstrap stability]
    D --> J[Profiles → personas → actions]
---

# Customer Segmentation: clustering you can defend

## Results

| Method (k = 5) | ARI vs k-means | Silhouette |
|---|---|---|
| k-means | 1.000 | 0.555 |
| Ward hierarchical | 0.942 | 0.554 |
| Gaussian mixture | 0.942 | 0.554 |
| DBSCAN (eps from the k-distance curve) | 0.273 | 0.357 |

**Stability:** k-means on 30 bootstrap resamples reproduces the full-data segments with ARI **0.975 ± 0.017**.

| Segment | Size | Age | Income (k$) | Score | Suggested action |
|---|---|---|---|---|---|
| low income, high spenders | 22 | 25 | 26 | 79 | loyalty points; watch credit risk |
| low income, low spenders | 23 | 45 | 26 | 21 | value bundles, discount days |
| mid income, average spenders | 81 | 43 | 55 | 50 | mainstream campaigns |
| high income, high spenders | 39 | 33 | 87 | 82 | VIP programme |
| high income, low spenders | 35 | 41 | 88 | 17 | **biggest upside**: find out why |

## What to notice
- **Three different algorithms agree (ARI 0.94)**, and resampling barely changes the answer. That is the evidence that
  the five groups are a property of the data and not an artefact of k-means.
- **DBSCAN fails, and that's the lesson.** The middle segment is dense and the corners are sparse. One global `eps`
  either merges the corners or marks them as noise. Density-based methods need similar densities.
- **Adding age** splits the 81-person "average" segment into a younger (≈ 27) and an older (≈ 56) group with the same
  wallet. Whether that matters is a business question, not a maths one.
- `adjusted_rand_index` in `common.py` is about 15 lines built from its contingency-table formula. Read it.

## Extend it
1. Gaussian mixtures give **soft** memberships: list the customers whose top probability is below 0.6. These are
   the borderline cases to keep out of aggressive campaigns.
2. Apply the same pipeline to the much larger UCI **Online Retail** data using **RFM features** (recency, frequency,
   monetary value), which is the standard in industry.
3. Train a decision tree to *predict* the segment from the raw features. The tree you get is a human-readable rulebook for staff.
