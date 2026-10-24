"""Imbalanced classification: class weights (via resampling), undersampling, SMOTE from scratch and threshold moving."""

import numpy as np

from mlscratch.datasets import make_classification
from mlscratch.linear_model import LogisticRegression
from mlscratch.metrics import f1_score, precision_score, recall_score, roc_auc_score
from mlscratch.model_selection import train_test_split
from mlscratch.neighbors import pairwise_distances
from mlscratch.preprocessing import StandardScaler

rng = np.random.default_rng(0)
X, y = make_classification(6000, 6, n_informative=3, class_sep=0.45, weights=[0.97], random_state=1)   # ~3% positives, overlapping classes
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
print(f"train positives {ytr.mean():.2%}  ({ytr.sum()} of {len(ytr)})")


def smote(X, y, k=5, ratio=1.0, rng=rng):
    """Synthesise minority points by interpolating towards random minority neighbours."""
    minority = X[y == 1]
    n_new = int(ratio * (np.sum(y == 0) - len(minority)))
    D = pairwise_distances(minority, minority)
    np.fill_diagonal(D, np.inf)
    neighbours = np.argsort(D, axis=1)[:, :k]
    i = rng.integers(len(minority), size=n_new)
    j = neighbours[i, rng.integers(k, size=n_new)]
    new = minority[i] + rng.random((n_new, 1)) * (minority[j] - minority[i])
    return np.vstack([X, new]), np.r_[y, np.ones(n_new, int)]


def undersample(X, y, rng=rng):
    neg = np.flatnonzero(y == 0)
    keep = np.r_[rng.choice(neg, np.sum(y == 1), replace=False), np.flatnonzero(y == 1)]
    return X[keep], y[keep]


def oversample(X, y, rng=rng):   # duplicating positives = class weighting for a GD-trained model
    pos = np.flatnonzero(y == 1)
    extra = rng.choice(pos, np.sum(y == 0) - len(pos))
    return np.vstack([X, X[extra]]), np.r_[y, y[extra]]


def evaluate(name, Xa, ya, threshold=0.5):
    m = LogisticRegression(n_iter=1500).fit(Xa, ya)
    p = m.predict_proba(Xte)[:, 1]
    pred = (p >= threshold).astype(int)
    print(f"{name:<34} recall {recall_score(yte, pred):.3f}  precision {precision_score(yte, pred):.3f}  "
          f"F1 {f1_score(yte, pred):.3f}  AUC {roc_auc_score(yte, p):.3f}  acc {np.mean(pred == yte):.3f}")
    return {"recall": recall_score(yte, pred), "f1": f1_score(yte, pred), "acc": float(np.mean(pred == yte))}


results = {"baseline": evaluate("baseline (imbalanced)", Xtr, ytr)}
results["undersample"] = evaluate("random undersampling", *undersample(Xtr, ytr))
results["oversample"] = evaluate("random oversampling (≈ class weights)", *oversample(Xtr, ytr))
results["SMOTE"] = evaluate("SMOTE", *smote(Xtr, ytr))

# threshold moving: choose t on a validation split of the TRAINING data
Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.3, stratify=ytr, random_state=1)
pv = LogisticRegression(n_iter=1500).fit(Xa, ya).predict_proba(Xv)[:, 1]
ts = np.linspace(0.02, 0.9, 89)
t_best = ts[int(np.argmax([f1_score(yv, (pv >= t).astype(int)) for t in ts]))]
results["threshold"] = evaluate(f"baseline + threshold {t_best:.2f}", Xtr, ytr, threshold=t_best)
best_f1 = max(results, key=lambda k: results[k]["f1"])
print(f"\nhighest accuracy: {max(results, key=lambda k: results[k]['acc'])} · highest recall: "
      f"{max(results, key=lambda k: results[k]['recall'])} · highest F1: {best_f1}")
print("→ accuracy rewards ignoring the rare class; pick the method by the metric that matches the business cost.")
