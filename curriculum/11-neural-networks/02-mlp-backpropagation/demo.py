"""Backprop from scratch: gradient check, a manual training loop, and MLP vs logistic regression on curved data."""

import numpy as np

from mlscratch.datasets import make_circles, make_moons
from mlscratch.linear_model import LogisticRegression
from mlscratch.neural_network import Dense, MLPClassifier, ReLU, Sequential, Tanh, gradient_check, softmax_cross_entropy
from mlscratch.optim import Adam
from mlscratch.viz import decision_boundary, plt, savefig

rng = np.random.default_rng(0)
Xs, Ys = rng.normal(size=(16, 3)), np.eye(4)[rng.integers(0, 4, 16)]
net = Sequential([Dense(3, 8, rng=1), Tanh(), Dense(8, 8, rng=2), Tanh(), Dense(8, 4, rng=3)])
print(f"gradient check (backprop vs finite differences), max relative error: {gradient_check(net, Xs, Ys):.2e}")

# A manual training loop - exactly what MLPClassifier does inside
X, y = make_circles(400, noise=0.08, factor=0.45, random_state=0)
Y = np.eye(2)[y]
model = Sequential([Dense(2, 16, rng=0), ReLU(), Dense(16, 16, rng=1), ReLU(), Dense(16, 2, init="xavier", rng=2)])
opt = Adam(0.01)
for epoch in range(301):
    order = rng.permutation(len(X))
    for start in range(0, len(X), 32):
        idx = order[start:start + 32]
        loss, grad = softmax_cross_entropy(model.forward(X[idx]), Y[idx])   # forward + dL/dlogits
        model.backward(grad)                                                # backprop through every layer
        opt.step(*model.parameters())                                       # update all W, b
    if epoch % 75 == 0:
        full_loss, _ = softmax_cross_entropy(model.forward(X), Y)
        print(f"epoch {epoch:>3}  loss {full_loss:.4f}  accuracy {np.mean(model.forward(X).argmax(1) == y):.3f}")

Xm, ym = make_moons(500, noise=0.2, random_state=1)
lin = LogisticRegression(n_iter=2000).fit(Xm, ym)
mlp = MLPClassifier((32, 32), lr=0.01, epochs=150, random_state=0).fit(Xm, ym)
print(f"\nmoons: logistic regression {lin.score(Xm, ym):.3f}   MLP(32, 32) {mlp.score(Xm, ym):.3f}")
for h in (1, 2, 3, 8):
    m = MLPClassifier((h,), activation="tanh", lr=0.02, epochs=300, random_state=0).fit(X, y)
    print(f"circles with {h} hidden unit(s): accuracy {m.score(X, y):.3f}")

pyplot = plt()
fig, axes = pyplot.subplots(1, 3, figsize=(15, 4.3))
decision_boundary(axes[0], lin, Xm, ym)
axes[0].set_title("logistic regression")
decision_boundary(axes[1], mlp, Xm, ym)
axes[1].set_title("MLP (32, 32)")
axes[2].plot(mlp.loss_history_)
axes[2].set(xlabel="epoch", ylabel="training loss", title="MLP loss curve")
savefig(fig, "mlp")
