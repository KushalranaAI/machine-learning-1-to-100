"""Confusion-matrix metrics, ROC/AUC as a ranking probability, PR curves, calibration and cost-based thresholds."""

import numpy as np

from mlscratch.datasets import make_classification
from mlscratch.linear_model import LogisticRegression
from mlscratch.metrics import classification_report, confusion_matrix, log_loss, precision_recall_curve, roc_auc_score, roc_curve
from mlscratch.model_selection import train_test_split
from mlscratch.preprocessing import StandardScaler
from mlscratch.viz import plt, savefig

X, y = make_classification(4000, 8, n_informative=4, class_sep=0.8, weights=[0.9], random_state=0)  # ~10% positives
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)
sc = StandardScaler().fit(Xtr)
model = LogisticRegression(n_iter=2000).fit(sc.transform(Xtr), ytr)
p = model.predict_proba(sc.transform(Xte))[:, 1]
pred = (p >= 0.5).astype(int)

print(f"positives in test set: {yte.mean():.1%}")
print("confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(yte, pred))
print(classification_report(yte, pred))
print(f"always predicting 'negative' would score accuracy {1 - yte.mean():.3f} with recall 0 - accuracy alone is useless here.")

auc = roc_auc_score(yte, p)
pos, neg = p[yte == 1], p[yte == 0]
pairs = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
print(f"\nROC-AUC {auc:.4f}  =  fraction of (positive, negative) pairs ranked correctly {pairs:.4f}")
print(f"log-loss {log_loss(yte, p):.4f}")

# cost-based threshold: a missed positive costs 20, a false alarm costs 1
thresholds = np.linspace(0.01, 0.99, 99)
cost = [20 * np.sum((p < t) & (yte == 1)) + 1 * np.sum((p >= t) & (yte == 0)) for t in thresholds]
t_best = thresholds[int(np.argmin(cost))]
print(f"cost-optimal threshold {t_best:.2f} (theory for calibrated p: 1/(1+20) = {1 / 21:.3f}); "
      f"cost at 0.5 = {cost[49]}, at best = {min(cost)}")

pyplot = plt()
fig, axes = pyplot.subplots(1, 3, figsize=(15, 4.2))
fpr, tpr, _ = roc_curve(yte, p)
axes[0].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
axes[0].plot([0, 1], [0, 1], "--", color="grey")
axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC curve")
axes[0].legend()
prec, rec, _ = precision_recall_curve(yte, np.round(p, 3))
axes[1].plot(rec, prec)
axes[1].axhline(yte.mean(), ls="--", color="grey", label="random classifier")
axes[1].set(xlabel="recall", ylabel="precision", title="precision-recall curve")
axes[1].legend()
bins = np.linspace(0, 1, 11)
idx = np.digitize(p, bins) - 1
frac = [yte[idx == b].mean() if np.any(idx == b) else np.nan for b in range(10)]
axes[2].plot((bins[:-1] + bins[1:]) / 2, frac, "o-", label="model")
axes[2].plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
axes[2].set(xlabel="predicted probability", ylabel="observed positive rate", title="reliability diagram")
axes[2].legend()
savefig(fig, "classification_metrics")
