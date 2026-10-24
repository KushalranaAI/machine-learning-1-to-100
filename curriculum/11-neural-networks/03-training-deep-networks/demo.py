"""Initialisation, vanishing gradients, optimisers, dropout and early stopping - measured on our NumPy networks."""

import numpy as np

from mlscratch.model_selection import train_test_split
from mlscratch.neural_network import Dense, MLPClassifier, ReLU, Sequential, Sigmoid, softmax_cross_entropy
from mlscratch.preprocessing import StandardScaler
from mlscratch.viz import plt, savefig

rng = np.random.default_rng(0)
X = rng.normal(size=(512, 256))

print("activation std through 10 ReLU layers (width 256):")
for name, scale in [("too small (0.01)", lambda n: 0.01), ("Xavier", lambda n: np.sqrt(2 / (n + n))), ("He", lambda n: np.sqrt(2 / n))]:
    a, stds = X, []
    for _ in range(10):
        a = np.maximum(0, a @ rng.normal(0, scale(256), (256, 256)))
        stds.append(a.std())
    print(f"  {name:<17} layer1 {stds[0]:.3f}  layer5 {stds[4]:.3e}  layer10 {stds[-1]:.3e}")

print("\ngradient size reaching the FIRST layer of a 10-layer net:")
Y = np.eye(10)[rng.integers(0, 10, 512)]
for act_name, Act in [("sigmoid", Sigmoid), ("ReLU", ReLU)]:
    layers = []
    for i in range(10):
        layers += [Dense(256 if i else 256, 256, init="he" if Act is ReLU else "xavier", rng=i), Act()]
    net = Sequential(layers + [Dense(256, 10, init="xavier", rng=99)])
    _, g = softmax_cross_entropy(net.forward(X), Y)
    net.backward(g)
    print(f"  {act_name:<8} |∂L/∂W₁| = {np.abs(net.layers[0].grads['W']).mean():.2e}   |∂L/∂W₁₀| = {np.abs(net.layers[18].grads['W']).mean():.2e}")

try:
    from sklearn.datasets import load_digits
except ImportError:
    raise SystemExit("pip install scikit-learn (used only to load the digits data)")
d = load_digits()
Xtr, Xva, ytr, yva = train_test_split(d.data, d.target, test_size=0.3, stratify=d.target, random_state=0)
sc = StandardScaler().fit(Xtr)
Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)

pyplot = plt()
fig, axes = pyplot.subplots(1, 2, figsize=(14, 4.2))
print("\noptimiser comparison on digits (MLP 64-128-10, 40 epochs):")
for opt, lr in [("sgd", 0.05), ("momentum", 0.02), ("rmsprop", 0.001), ("adam", 0.001)]:
    m = MLPClassifier((128,), optimizer=opt, lr=lr, epochs=40, random_state=0).fit(Xtr, ytr, Xva, yva)
    axes[0].plot(m.val_history_, label=opt)
    print(f"  {opt:<9} validation accuracy {m.score(Xva, yva):.3f}   final val loss {m.val_history_[-1]:.3f}")
axes[0].set(xlabel="epoch", ylabel="validation loss", title="optimisers", yscale="log")
axes[0].legend()

print("\ndropout vs overfitting (MLP 64-256-256-10 on a small subset, 150 epochs):")
small = slice(0, 300)
for p in (0.0, 0.3, 0.5):
    m = MLPClassifier((256, 256), dropout=p, lr=0.001, epochs=150, random_state=0).fit(Xtr[small], ytr[small], Xva, yva)
    best = int(np.argmin(m.val_history_))
    axes[1].plot(m.val_history_, label=f"dropout {p}")
    print(f"  dropout {p}: train acc {m.score(Xtr[small], ytr[small]):.3f}  val acc {m.score(Xva, yva):.3f}  "
          f"best val loss at epoch {best + 1} (early stopping point)")
axes[1].set(xlabel="epoch", ylabel="validation loss", title="dropout and early stopping")
axes[1].legend()
savefig(fig, "training")
