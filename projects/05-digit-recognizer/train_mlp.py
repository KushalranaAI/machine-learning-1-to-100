"""Step 4 - a multi-layer perceptron from scratch (backprop + Adam), with and without regularisation."""

import time

import numpy as np

from common import load, show_digits
from mlscratch.metrics import accuracy_score, confusion_matrix
from mlscratch.neural_network import MLPClassifier
from mlscratch.viz import plt, savefig

xtr, ytr, xval, yval, xte, yte = load()
configs = {
    "784-256-128-10, no regularisation": dict(hidden_layers=(256, 128)),
    "784-256-128-10, dropout 0.2": dict(hidden_layers=(256, 128), dropout=0.2),
}
runs = {}
for name, cfg in configs.items():
    t = time.time()
    m = MLPClassifier(**cfg, optimizer="adam", lr=1e-3, epochs=15, batch_size=128, random_state=0, verbose=True)
    m.fit(xtr, ytr, xval, yval)
    acc = {s: accuracy_score(y, m.predict(X)) for s, X, y in (("train", xtr, ytr), ("val", xval, yval), ("test", xte, yte))}
    runs[name] = (m, acc)
    n_params = sum(p.size for p in m.model_.parameters()[0].values())
    print(f"→ {name}: {n_params:,} parameters, train {acc['train']:.4f} · val {acc['val']:.4f} · test {acc['test']:.4f} "
          f"({time.time() - t:.0f}s)\n", flush=True)

name, (best, acc) = max(runs.items(), key=lambda kv: kv[1][1]["val"])
print(f"Chosen on validation: {name} → test accuracy {acc['test']:.4f} ({(1 - acc['test']) * 10000:.0f} of 10,000 wrong)")
pred, proba = best.predict(xte), best.predict_proba(xte)
cm = confusion_matrix(yte, pred)
off = cm - np.diag(np.diag(cm))
pairs = sorted(((off[i, j], i, j) for i in range(10) for j in range(10) if off[i, j]), reverse=True)[:5]
print("most frequent confusions (true → predicted):", ", ".join(f"{i}→{j} ({c})" for c, i, j in pairs))
print("per-class recall:", " ".join(f"{d}:{cm[d, d] / cm[d].sum():.3f}" for d in range(10)))

P = plt()
fig, axes = P.subplots(1, 2, figsize=(13, 4.3))
for name, (m, _) in runs.items():
    axes[0].plot(m.loss_history_, label=f"train - {name.split(', ')[1]}")
    axes[0].plot(m.val_history_, "--", label=f"val - {name.split(', ')[1]}")
axes[0].set(title="Cross-entropy per epoch", xlabel="epoch", yscale="log")
axes[0].legend(fontsize=8)
axes[1].imshow(np.log1p(off), cmap="Reds")
axes[1].set(title="Confusion matrix off-diagonal (log counts)", xlabel="predicted", ylabel="true", xticks=range(10), yticks=range(10))
savefig(fig, "04_mlp_training")

wrong = np.flatnonzero(pred != yte)
worst = wrong[np.argsort(-proba[wrong].max(1))][:12]
fig, axes = P.subplots(2, 6, figsize=(10, 4))
show_digits(axes, xte[worst], [f"true {yte[i]} → {pred[i]} ({proba[i].max():.0%})" for i in worst])
savefig(fig, "05_confident_mistakes")

W1 = best.model_.layers[0].params["W"]
fig, axes = P.subplots(2, 8, figsize=(11, 3))
show_digits(axes, W1.T[:16], [f"unit {i}" for i in range(16)])
savefig(fig, "06_first_layer_weights")
