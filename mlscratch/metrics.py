"""Evaluation metrics, each written straight from its formula."""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------- regression


def mean_squared_error(y_true, y_pred) -> float:
    """MSE = (1/n) Σ (y - ŷ)²"""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true, y_pred) -> float:
    """MAE = (1/n) Σ |y - ŷ|  (robust to outliers, median-optimal)"""
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def r2_score(y_true, y_pred) -> float:
    """R² = 1 - SS_res / SS_tot : share of the variance in y explained by the model."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def mean_absolute_percentage_error(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-12))))


# ------------------------------------------------------------------------- classification


def confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    """C[i, j] = number of samples with true class i predicted as class j."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred])) if labels is None else np.asarray(labels)
    index = {label: i for i, label in enumerate(labels)}
    C = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        C[index[t], index[p]] += 1
    return C


def accuracy_score(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def _binary_counts(y_true, y_pred, pos_label=1):
    y_true, y_pred = np.asarray(y_true) == pos_label, np.asarray(y_pred) == pos_label
    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)
    tn = np.sum(~y_true & ~y_pred)
    return tp, fp, fn, tn


def precision_score(y_true, y_pred, pos_label=1, average="binary") -> float:
    """precision = TP / (TP + FP): of the predicted positives, how many are right."""
    if average != "binary":
        return _averaged(precision_score, y_true, y_pred, average)
    tp, fp, _, _ = _binary_counts(y_true, y_pred, pos_label)
    return float(tp / (tp + fp)) if tp + fp else 0.0


def recall_score(y_true, y_pred, pos_label=1, average="binary") -> float:
    """recall = TP / (TP + FN): of the real positives, how many we found."""
    if average != "binary":
        return _averaged(recall_score, y_true, y_pred, average)
    tp, _, fn, _ = _binary_counts(y_true, y_pred, pos_label)
    return float(tp / (tp + fn)) if tp + fn else 0.0


def f1_score(y_true, y_pred, pos_label=1, average="binary") -> float:
    """F1 = harmonic mean of precision and recall = 2PR / (P + R)."""
    if average != "binary":
        return _averaged(f1_score, y_true, y_pred, average)
    p, r = precision_score(y_true, y_pred, pos_label), recall_score(y_true, y_pred, pos_label)
    return 2 * p * r / (p + r) if p + r else 0.0


def _averaged(metric, y_true, y_pred, average):
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    scores = np.array([metric(y_true, y_pred, pos_label=c) for c in labels])
    if average == "macro":
        return float(scores.mean())
    if average == "weighted":
        support = np.array([np.sum(np.asarray(y_true) == c) for c in labels])
        return float(np.sum(scores * support) / support.sum())
    raise ValueError(f"unknown average {average!r}")


def roc_curve(y_true, scores):
    """Return (fpr, tpr, thresholds) sweeping the threshold from high to low."""
    y_true, scores = np.asarray(y_true), np.asarray(scores, float)
    order = np.argsort(-scores, kind="mergesort")
    y_sorted, s_sorted = y_true[order], scores[order]
    distinct = np.where(np.diff(s_sorted))[0]           # last index of each tied score group
    idx = np.r_[distinct, len(y_sorted) - 1]
    tps = np.cumsum(y_sorted)[idx]
    fps = (1 + idx) - tps
    tpr = np.r_[0, tps / max(tps[-1], 1)]
    fpr = np.r_[0, fps / max(fps[-1], 1)]
    return fpr, tpr, np.r_[np.inf, s_sorted[idx]]


def roc_auc_score(y_true, scores) -> float:
    """Area under the ROC curve = P(score of a random positive > score of a random negative)."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.trapezoid(tpr, fpr))


def precision_recall_curve(y_true, scores):
    y_true, scores = np.asarray(y_true), np.asarray(scores, float)
    thresholds = np.unique(scores)[::-1]
    precision = np.array([precision_score(y_true, (scores >= t).astype(int)) for t in thresholds])
    recall = np.array([recall_score(y_true, (scores >= t).astype(int)) for t in thresholds])
    return precision, recall, thresholds


def log_loss(y_true, proba, eps: float = 1e-15) -> float:
    """Cross-entropy: -(1/n) Σ log p(true class). `proba` is (n,) for binary or (n, K)."""
    y_true, proba = np.asarray(y_true), np.clip(np.asarray(proba, float), eps, 1 - eps)
    if proba.ndim == 1:
        return float(-np.mean(y_true * np.log(proba) + (1 - y_true) * np.log(1 - proba)))
    return float(-np.mean(np.log(proba[np.arange(len(y_true)), y_true.astype(int)])))


def classification_report(y_true, y_pred) -> str:
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    lines = [f"{'class':>10} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}"]
    for c in labels:
        support = int(np.sum(np.asarray(y_true) == c))
        lines.append(f"{str(c):>10} {precision_score(y_true, y_pred, c):>10.3f} "
                     f"{recall_score(y_true, y_pred, c):>10.3f} {f1_score(y_true, y_pred, c):>10.3f} {support:>10}")
    lines.append(f"{'accuracy':>10} {accuracy_score(y_true, y_pred):>32.3f} {len(y_true):>10}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- clustering


def silhouette_score(X, labels) -> float:
    """Mean over points of (b - a) / max(a, b); a = mean intra-cluster distance, b = nearest other cluster."""
    X, labels = np.asarray(X, float), np.asarray(labels)
    D = np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0))
    clusters = np.unique(labels)
    s = np.zeros(len(X))
    for i in range(len(X)):
        same = labels == labels[i]
        if same.sum() <= 1:
            continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = min(D[i, labels == c].mean() for c in clusters if c != labels[i])
        s[i] = (b - a) / max(a, b)
    return float(s.mean())
