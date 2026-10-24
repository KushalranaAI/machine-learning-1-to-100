"""Anomaly detection: z-score vs Mahalanobis vs GMM density vs Isolation Forest, scored with precision@k."""

import numpy as np

from mlscratch.anomaly import IsolationForest, mahalanobis
from mlscratch.cluster import GaussianMixture
from mlscratch.metrics import roc_auc_score

rng = np.random.default_rng(0)
# normal data: two correlated modes (e.g. two kinds of customers)
normal = np.vstack([rng.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], 500),
                    rng.multivariate_normal([6, 6], [[1, 0.9], [0.9, 1]], 500)])
far = rng.uniform(-6, 12, (10, 2))                                   # obvious outliers
breaking = np.array([[1.5, -1.5], [-1.4, 1.6], [7.5, 4.5], [4.6, 7.4], [1.8, -1.2]])   # normal on each axis, wrong combination
X = np.vstack([normal, far, breaking])
y = np.r_[np.zeros(len(normal)), np.ones(len(far) + len(breaking))]

z = np.abs((X - X.mean(0)) / X.std(0)).max(1)
scores = {
    "max |z-score|": z,
    "Mahalanobis (1 Gaussian)": mahalanobis(X),
    "-log GMM density (2 comps)": -np.log(np.exp(GaussianMixture(2, random_state=0).fit(normal)._log_gauss(X)) @ GaussianMixture(2, random_state=0).fit(normal).weights_ + 1e-300),
    "Isolation Forest": IsolationForest(200, random_state=0).fit(X).score_samples(X),
}
k = int(y.sum())
print(f"{len(X)} points, {k} true anomalies ({len(breaking)} of them only break the correlation)\n")
print(f"{'method':<28}{'precision@15':>13}{'ROC-AUC':>9}{'correlation-breakers found':>28}")
for name, s in scores.items():
    top = np.argsort(-s)[:k]
    found_breakers = int(np.sum(top >= len(normal) + len(far)))
    print(f"{name:<28}{y[top].mean():>13.2f}{roc_auc_score(y, s):>9.3f}{found_breakers:>22}/{len(breaking)}")
