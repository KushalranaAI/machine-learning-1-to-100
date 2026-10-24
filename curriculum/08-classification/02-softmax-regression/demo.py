"""Softmax regression from scratch on 8×8 digits; comparison with one-vs-rest; temperature and stability."""

import numpy as np

from mlscratch.base import sigmoid, softmax
from mlscratch.linear_model import LogisticRegression, SoftmaxRegression
from mlscratch.metrics import confusion_matrix, log_loss
from mlscratch.model_selection import train_test_split
from mlscratch.preprocessing import StandardScaler
from mlscratch.viz import plt, savefig

z = np.array([[1000.0, 1001.0, 1002.0]])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(z) / np.exp(z).sum()
print("naive softmax of [1000, 1001, 1002]:", naive, " → stable version:", np.round(softmax(z), 4))
print("softmax of 2 scores == sigmoid of the difference:", np.allclose(softmax(np.array([[0.3, 1.7]]))[0, 1], sigmoid(np.array([1.4]))[0]))
for T in (0.1, 1, 10):
    print(f"temperature {T:>4}: softmax([2, 1, 0]/T) = {np.round(softmax(np.array([[2.0, 1.0, 0.0]]) / T), 3)}")

try:
    from sklearn.datasets import load_digits     # only to load the bundled digits data
except ImportError:
    raise SystemExit("pip install scikit-learn (used only to load the digits dataset)")
d = load_digits()
Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.3, stratify=d.target, random_state=0)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

soft = SoftmaxRegression(lr=0.3, n_iter=1500, reg=1e-3).fit(Xtr, ytr)
ovr = LogisticRegression(lr=0.3, n_iter=1500).fit(Xtr, ytr)
print(f"\ndigits test accuracy: softmax {soft.score(Xte, yte):.3f}   one-vs-rest {ovr.score(Xte, yte):.3f}")
print(f"softmax test log-loss {log_loss(yte, soft.predict_proba(Xte)):.4f}")
print("confusion matrix (rows = true digit):\n", confusion_matrix(yte, soft.predict(Xte)))

pyplot = plt()
fig, axes = pyplot.subplots(1, 10, figsize=(14, 1.8))
for k in range(10):
    axes[k].imshow(soft.coef_[:, k].reshape(8, 8), cmap="RdBu")
    axes[k].set_title(str(k))
    axes[k].axis("off")
fig.suptitle("learned weight templates (red = pixel votes for the digit, blue = against)", fontsize=9)
savefig(fig, "softmax_weights")
