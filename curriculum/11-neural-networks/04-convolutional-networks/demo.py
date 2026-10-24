"""Convolution from scratch: hand-made filters, a tiny CNN trained on 8×8 digits, and parameter counts."""

import time

import numpy as np

from mlscratch.model_selection import train_test_split
from mlscratch.neural_network import Conv2D, Dense, Flatten, MaxPool2D, MLPClassifier, ReLU, Sequential, softmax_cross_entropy
from mlscratch.optim import Adam
from mlscratch.viz import plt, savefig

try:
    from sklearn.datasets import load_digits
except ImportError:
    raise SystemExit("pip install scikit-learn (used only to load the digits data)")
d = load_digits()
X = (d.data / 16.0).reshape(-1, 1, 8, 8)
y = d.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

# 1. hand-made filters
edges = Conv2D(1, 3, k=3, pad=1)
edges.params["W"] = np.array([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],     # vertical edges (Sobel x)
                              [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],     # horizontal edges (Sobel y)
                              [[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], float)  # Laplacian (blobs / corners)
maps = edges.forward(X[:1])
print("feature maps shape for 1 image, 3 filters:", maps.shape)

# 2. a tiny CNN: conv(8 filters 3×3) → ReLU → maxpool → flatten → dense(10)
rng = np.random.default_rng(0)
cnn = Sequential([Conv2D(1, 8, k=3, pad=1, rng=0), ReLU(), MaxPool2D(2), Flatten(), Dense(8 * 4 * 4, 10, init="xavier", rng=1)])
opt = Adam(0.01)
Ytr = np.eye(10)[ytr]
t0 = time.perf_counter()
for epoch in range(15):
    order = rng.permutation(len(Xtr))
    for s in range(0, len(Xtr), 32):
        idx = order[s:s + 32]
        _, g = softmax_cross_entropy(cnn.forward(Xtr[idx]), Ytr[idx])
        cnn.backward(g)
        opt.step(*cnn.parameters())
    if epoch % 5 == 4:
        print(f"epoch {epoch + 1:>2}: test accuracy {np.mean(cnn.forward(Xte).argmax(1) == yte):.3f}")
print(f"trained in {time.perf_counter() - t0:.1f}s")

n_cnn = sum(p.size for p in cnn.parameters()[0].values())
mlp = MLPClassifier((40,), lr=0.005, epochs=60, random_state=0).fit(Xtr.reshape(len(Xtr), -1), ytr)
n_mlp = sum(p.size for p in mlp.model_.parameters()[0].values())
print(f"\nCNN: {n_cnn} parameters, test acc {np.mean(cnn.forward(Xte).argmax(1) == yte):.3f}")
print(f"MLP: {n_mlp} parameters, test acc {mlp.score(Xte.reshape(len(Xte), -1), yte):.3f}")
print(f"conv layer alone: {cnn.layers[0].params['W'].size + cnn.layers[0].params['b'].size} parameters shared across all 64 positions")
print("→ the CNN reaches about the same accuracy with less than half the parameters; on larger images the gap is enormous.")

pyplot = plt()
fig, axes = pyplot.subplots(2, 4, figsize=(11, 5.5))
axes[0, 0].imshow(X[0, 0], cmap="gray_r")
axes[0, 0].set_title(f"input digit {y[0]}")
for k, name in enumerate(["vertical edges", "horizontal edges", "Laplacian"]):
    axes[0, k + 1].imshow(maps[0, k], cmap="RdBu")
    axes[0, k + 1].set_title(name)
for k in range(4):
    axes[1, k].imshow(cnn.layers[0].params["W"][k, 0], cmap="RdBu")
    axes[1, k].set_title(f"learned filter {k + 1}")
for ax in axes.ravel():
    ax.axis("off")
savefig(fig, "cnn_filters")
