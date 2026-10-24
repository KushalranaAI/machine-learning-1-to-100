---
title: Anomaly Detection
order: 5
summary: Statistical scores (z, IQR, Mahalanobis), density-based detection with Gaussian models, and Isolation Forest - with evaluation when labels are rare.
level: intermediate
minutes: 45
why: Fraud, intrusions, equipment failures and data errors are rare and often unlabelled. Anomaly detection finds them by modelling what "normal" looks like and flagging what doesn't fit.
background: Grubbs' test (1950) and Mahalanobis distance (1936) are classical. Isolation Forest (Liu, Ting & Zhou, 2008) introduced the idea that anomalies are easier to isolate than to describe.
use_cases:
  - Credit-card fraud and insurance claims
  - Network intrusion and bot detection
  - Predictive maintenance from sensor readings
  - Data-quality monitoring
where_used:
  - 03-customer-segmentation
code:
  - mlscratch/anomaly.py:mahalanobis
  - mlscratch/anomaly.py:IsolationForest
  - mlscratch/cluster.py:GaussianMixture
demo: demo.py
pros:
  - Works without labels
  - Isolation Forest scales well and needs no distance metric
cons:
  - "\"Anomalous\" ≠ \"interesting\" - many flags are just rare-but-normal"
  - Hard to evaluate without (some) labels; threshold choice is a business decision
workflow: |
  flowchart LR
    A[mostly normal data] --> B{method}
    B --> C[statistical: z, Mahalanobis]
    B --> D[density: Gaussian / GMM likelihood]
    B --> E[isolation: random trees, path length]
    C & D & E --> F[anomaly score]
    F --> G[threshold = expected contamination / alert budget]
    G --> H[review flags, feed confirmed cases back]
---

## Three families
**Statistical distance** - how many "standard deviations" from the centre, accounting for correlation:
$D_M(x) = \sqrt{(x-\mu)^\top\Sigma^{-1}(x-\mu)}$. Under a Gaussian, $D_M^2\sim\chi^2_d$.

**Density** - fit $p(x)$ (Gaussian, GMM, kernel density) on normal data; flag $p(x) < \varepsilon$. Andrew Ng's classic
recipe: product of per-feature Gaussians, choose ε on a small labelled validation set by F1.

**Isolation** - anomalies are *few and different*, so random splits isolate them quickly:
1. Build trees on random subsamples of size $\psi$ (256): pick a random feature and a random threshold between its min and max, recurse until points are isolated or a depth limit $\lceil\log_2\psi\rceil$.
2. The path length $h(x)$ from root to $x$'s leaf is short for anomalies.
3. Normalise by the average path length of an unsuccessful binary-search-tree search, $c(\psi) = 2H(\psi-1) - \frac{2(\psi-1)}{\psi}$:
$$
s(x) = 2^{-\frac{\mathbb E[h(x)]}{c(\psi)}} \qquad (s\to 1: \text{anomaly},\ s \lesssim 0.5: \text{normal})
$$
No distances, no density - just counting splits, so it handles many features and scales linearly.

## Evaluating with few labels
Use precision@k (of the top-k scored points, how many are real anomalies?), PR-AUC, and alert budget: if the team can
review 50 cases a day, the threshold is the 50th highest score.

## Exercises
1. Run the demo; which method finds the "correlation breaking" anomalies that look normal on each axis?
2. Change the contamination parameter and plot precision vs recall.
3. Use a GMM's log-density as the score on multi-modal normal data. Why does a single Gaussian fail there?
