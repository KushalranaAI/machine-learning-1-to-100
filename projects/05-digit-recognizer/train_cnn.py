"""Step 5 - a convolutional network from scratch, with the training loop written out by hand.

conv(1→8, 3×3) → ReLU → maxpool 2 → conv(8→16, 3×3) → ReLU → maxpool 2 → flatten(16·7·7) → dense 64 → ReLU → dense 10
Convolutions run as im2col + one matrix product (mlscratch/neural_network.py). It's pure NumPy, so expect ~30 s per epoch.
"""

import argparse
import time

import numpy as np

from common import load
from mlscratch.base import one_hot
from mlscratch.metrics import accuracy_score
from mlscratch.neural_network import Conv2D, Dense, Flatten, MaxPool2D, ReLU, Sequential, softmax_cross_entropy
from mlscratch.optim import Adam
from mlscratch.viz import plt, savefig

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--n-train", type=int, default=None, help="use a random subset for a quicker run")
args = parser.parse_args()

xtr, ytr, xval, yval, xte, yte = load(flat=False, n_train=args.n_train)
xtr, xval, xte = xtr[:, None], xval[:, None], xte[:, None]          # (n, channels=1, 28, 28)
rng = np.random.default_rng(0)
net = Sequential([
    Conv2D(1, 8, 3, pad=1, rng=rng), ReLU(), MaxPool2D(2),
    Conv2D(8, 16, 3, pad=1, rng=rng), ReLU(), MaxPool2D(2),
    Flatten(), Dense(16 * 7 * 7, 64, rng=rng), ReLU(), Dense(64, 10, "xavier", rng),
])
params, _ = net.parameters()
n_conv = sum(p.size for k, p in params.items() if k.split(".")[0] in ("0", "3"))
print(f"{sum(p.size for p in params.values()):,} parameters, only {n_conv:,} of them in the two conv layers: a 3×3 filter is\n"
      f"reused at every position (weight sharing). The MLP in train_mlp.py needs 235,146.")


def predict(X, batch=500):
    net.train(False)
    return np.concatenate([np.argmax(net.forward(X[i : i + batch]), 1) for i in range(0, len(X), batch)])


opt, Y = Adam(1e-3), one_hot(ytr, 10)
history, best = [], (0.0, None)
for epoch in range(1, args.epochs + 1):
    t, net_loss = time.time(), 0.0
    net.train(True)
    order = rng.permutation(len(xtr))
    for start in range(0, len(xtr), 64):
        idx = order[start : start + 64]
        loss, grad = softmax_cross_entropy(net.forward(xtr[idx]), Y[idx])   # forward pass + dLoss/dlogits
        net.backward(grad)                                                   # backprop through every layer
        opt.step(*net.parameters())                                          # Adam update, in place
        net_loss += loss * len(idx)
    val = accuracy_score(yval, predict(xval))
    history.append((net_loss / len(xtr), val))
    if val > best[0]:  # keep the best weights seen on validation (early stopping without stopping)
        best = (val, {k: v.copy() for k, v in net.parameters()[0].items()})
    print(f"epoch {epoch}: train loss {history[-1][0]:.4f} · val accuracy {val:.4f} · {time.time() - t:.0f}s", flush=True)

for k, v in best[1].items():
    net.parameters()[0][k][...] = v
pred = predict(xte)
print(f"\nCNN test accuracy {accuracy_score(yte, pred):.4f} ({np.sum(pred != yte)} of 10,000 wrong) using the best-validation weights")

P = plt()
fig, axes = P.subplots(1, 2, figsize=(12, 3.8))
axes[0].plot([h[0] for h in history], "o-")
axes[0].set(title="CNN training loss", xlabel="epoch")
axes[1].plot([h[1] for h in history], "o-")
axes[1].set(title="CNN validation accuracy", xlabel="epoch")
savefig(fig, "07_cnn_training")

filters = net.layers[0].params["W"][:, 0]
net.train(False)
maps = net.layers[1].forward(net.layers[0].forward(xte[:1]))[0]
fig, axes = P.subplots(2, 8, figsize=(11, 3))
for i in range(8):
    axes[0, i].imshow(filters[i], cmap="RdBu")
    axes[0, i].set_title(f"filter {i}", fontsize=8)
    axes[1, i].imshow(maps[i], cmap="gray_r")
    axes[1, i].set_title("its feature map", fontsize=8)
for ax in axes.ravel():
    ax.axis("off")
savefig(fig, "08_cnn_filters")
