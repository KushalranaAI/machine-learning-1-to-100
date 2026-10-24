"""A vanilla RNN from scratch with BPTT: sine-wave prediction, and the 'remember the first input' task vs sequence length."""

import numpy as np

from mlscratch.neural_network import Dense, SimpleRNN, mse_loss
from mlscratch.optim import Adam

rng = np.random.default_rng(0)


def train(X, Y, hidden=16, epochs=200, lr=0.01, clip=None):
    rnn, head = SimpleRNN(X.shape[2], hidden, rng=0), Dense(hidden, Y.shape[1], init="xavier", rng=1)
    opt = Adam(lr)
    for _ in range(epochs):
        h = rnn.forward(X)
        loss, g = mse_loss(head.forward(h), Y)
        rnn.backward(head.backward(g))
        grads = {**{f"r{k}": v for k, v in rnn.grads.items()}, **{f"h{k}": v for k, v in head.grads.items()}}
        if clip:
            norm = np.sqrt(sum((v**2).sum() for v in grads.values()))
            grads = {k: v * min(1, clip / (norm + 1e-12)) for k, v in grads.items()}
        params = {**{f"r{k}": v for k, v in rnn.params.items()}, **{f"h{k}": v for k, v in head.params.items()}}
        opt.step(params, grads)
    return rnn, head, loss


# 1. predict the next value of a noisy sine wave from the previous 20
t = np.arange(0, 400, 0.1)
series = np.sin(t) + 0.05 * rng.normal(size=len(t))
idx = rng.choice(len(series) - 21, 800, replace=False)
X = np.stack([series[i:i + 20] for i in idx])[:, :, None]
Y = series[idx + 20][:, None]
rnn, head, loss = train(X, Y, epochs=150)
print(f"sine wave, next-step prediction: final training MSE {loss:.4f} (noise variance 0.0025)")

# 2. long-range memory: the target is the FIRST element of a random sequence
print("\n'remember the first value' task (predicting 0 gives MSE ≈ 1.0):")
print(f"{'T':>4}{'trained MSE':>13}{'grad at x₁ / grad at x_T (untrained net)':>42}")
for T in (2, 5, 10, 20, 40):
    Xs = rng.normal(size=(400, T, 1))
    # gradient flow at initialisation: how much signal from the loss reaches the first vs the last input?
    rnn0, head0 = SimpleRNN(1, 16, rng=0), Dense(16, 1, init="xavier", rng=1)
    _, g = mse_loss(head0.forward(rnn0.forward(Xs)), Xs[:, 0, :])
    dx = rnn0.backward(head0.backward(g))
    ratio = np.abs(dx[:, 0]).mean() / np.abs(dx[:, -1]).mean()
    _, _, loss = train(Xs, Xs[:, 0, :], epochs=300, clip=1.0)
    print(f"{T:>4}{loss:>13.3f}{ratio:>42.2e}")
print("→ the learning signal reaching x₁ shrinks exponentially with T, so the vanilla RNN fails to learn long-range")
print("  memory (MSE rises towards 1). LSTMs and GRUs were designed to fix exactly this.")
