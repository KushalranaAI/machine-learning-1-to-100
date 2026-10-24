"""A small deep-learning framework in NumPy: layers with forward/backward, losses, and an MLP wrapper.

Backpropagation is the chain rule applied layer by layer, from the loss back to the inputs:
    forward:   a⁰ = x,  zˡ = aˡ⁻¹Wˡ + bˡ,  aˡ = φ(zˡ)
    backward:  δᴸ = ∂L/∂zᴸ;  ∂L/∂Wˡ = (aˡ⁻¹)ᵀδˡ;  ∂L/∂bˡ = Σ δˡ;  δˡ⁻¹ = (δˡ Wˡᵀ) ⊙ φ'(zˡ⁻¹)
Every Layer stores what it needs during forward() and returns ∂L/∂input from backward().
"""

from __future__ import annotations

import numpy as np

from .base import BaseEstimator, ClassifierMixin, RegressorMixin, as_2d, check_Xy, one_hot, rng_from, softmax
from .optim import SGD, Adam, RMSProp


class Layer:
    params: dict = {}
    grads: dict = {}
    training = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class Dense(Layer):
    """z = xW + b. He init for ReLU nets (var 2/fan_in), Xavier/Glorot otherwise (var 2/(fan_in+fan_out))."""

    def __init__(self, n_in, n_out, init="he", rng=None):
        rng = rng_from(rng)
        scale = np.sqrt(2 / n_in) if init == "he" else np.sqrt(2 / (n_in + n_out))
        self.params = {"W": rng.normal(0, scale, (n_in, n_out)), "b": np.zeros(n_out)}
        self.grads = {}

    def forward(self, x):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad):
        self.grads = {"W": self.x.T @ grad, "b": grad.sum(0)}
        return grad @ self.params["W"].T


class ReLU(Layer):
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


class LeakyReLU(Layer):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, grad):
        return grad * np.where(self.x > 0, 1, self.alpha)


class Sigmoid(Layer):
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return self.out

    def backward(self, grad):
        return grad * self.out * (1 - self.out)      # σ' = σ(1-σ)


class Tanh(Layer):
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad):
        return grad * (1 - self.out**2)


class Dropout(Layer):
    """Inverted dropout: zero a fraction p of activations during training and scale the rest by 1/(1-p)."""

    def __init__(self, p=0.5, rng=None):
        self.p, self.rng = p, rng_from(rng)

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        self.mask = (self.rng.random(x.shape) >= self.p) / (1 - self.p)
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask if self.training and self.p else grad


class BatchNorm(Layer):
    """x̂ = (x - μ_B)/√(σ²_B + ε),  y = γx̂ + β. Running averages are used at inference time."""

    def __init__(self, n, momentum=0.9, eps=1e-5):
        self.params = {"gamma": np.ones(n), "beta": np.zeros(n)}
        self.grads, self.momentum, self.eps = {}, momentum, eps
        self.running_mean, self.running_var = np.zeros(n), np.ones(n)

    def forward(self, x):
        if self.training:
            mu, var = x.mean(0), x.var(0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mu, var = self.running_mean, self.running_var
        self.std = np.sqrt(var + self.eps)
        self.xhat = (x - mu) / self.std
        return self.params["gamma"] * self.xhat + self.params["beta"]

    def backward(self, grad):
        n = grad.shape[0]
        self.grads = {"gamma": (grad * self.xhat).sum(0), "beta": grad.sum(0)}
        dxhat = grad * self.params["gamma"]
        return (n * dxhat - dxhat.sum(0) - self.xhat * (dxhat * self.xhat).sum(0)) / (n * self.std)


# ------------------------------------------------------------------------------------- losses


def softmax_cross_entropy(logits, Y):
    """Returns (loss, ∂loss/∂logits). The famous simplification: ∂L/∂z = (softmax(z) - y) / n."""
    P = softmax(logits)
    n = len(Y)
    loss = -np.sum(Y * np.log(np.clip(P, 1e-12, None))) / n
    return loss, (P - Y) / n


def mse_loss(pred, y):
    n = len(y)
    diff = pred - y
    return float(np.mean(diff**2)), 2 * diff / (n * diff.shape[1])


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def train(self, mode=True):
        for layer in self.layers:
            layer.training = mode

    def parameters(self):
        """Flat views {"3.W": array, ...} so optimisers can update every layer's params in place."""
        params, grads = {}, {}
        for i, layer in enumerate(self.layers):
            for k in layer.params:
                params[f"{i}.{k}"] = layer.params[k]
                grads[f"{i}.{k}"] = layer.grads.get(k, np.zeros_like(layer.params[k]))
        return params, grads


def gradient_check(model: Sequential, X, Y, eps=1e-5, n_checks=10, rng=0) -> float:
    """Compare backprop gradients with finite differences (f(θ+ε) - f(θ-ε)) / 2ε. Returns max relative error."""
    rng = rng_from(rng)
    model.train(False)
    loss, g = softmax_cross_entropy(model.forward(X), Y)
    model.backward(g)
    params, grads = model.parameters()
    worst = 0.0
    for name in params:
        p = params[name]
        for _ in range(n_checks):
            idx = tuple(rng.integers(s) for s in p.shape)
            old = p[idx]
            p[idx] = old + eps
            lp, _ = softmax_cross_entropy(model.forward(X), Y)
            p[idx] = old - eps
            lm, _ = softmax_cross_entropy(model.forward(X), Y)
            p[idx] = old
            num = (lp - lm) / (2 * eps)
            ana = grads[name][idx]
            worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    return worst


class _MLPBase(BaseEstimator):
    def __init__(self, hidden_layers=(64,), activation="relu", optimizer="adam", lr=1e-3, epochs=100,
                 batch_size=32, dropout=0.0, batch_norm=False, weight_decay=0.0, random_state=None, verbose=False):
        self.hidden_layers, self.activation, self.optimizer, self.lr = hidden_layers, activation, optimizer, lr
        self.epochs, self.batch_size, self.dropout, self.batch_norm = epochs, batch_size, dropout, batch_norm
        self.weight_decay, self.random_state, self.verbose = weight_decay, random_state, verbose

    def _build(self, n_in, n_out, rng):
        act = {"relu": ReLU, "tanh": Tanh, "sigmoid": Sigmoid, "leaky_relu": LeakyReLU}[self.activation]
        layers, prev = [], n_in
        for h in self.hidden_layers:
            layers.append(Dense(prev, h, "he" if self.activation in ("relu", "leaky_relu") else "xavier", rng))
            if self.batch_norm:
                layers.append(BatchNorm(h))
            layers.append(act())
            if self.dropout:
                layers.append(Dropout(self.dropout, rng))
            prev = h
        layers.append(Dense(prev, n_out, "xavier", rng))
        return Sequential(layers)

    def _optimizer(self):
        if self.optimizer == "adam":
            return Adam(self.lr, weight_decay=self.weight_decay)
        if self.optimizer == "rmsprop":
            return RMSProp(self.lr)
        return SGD(self.lr, momentum=0.9 if self.optimizer == "momentum" else 0.0, weight_decay=self.weight_decay)

    def _train(self, X, T, loss_fn, X_val=None, T_val=None):
        rng = rng_from(self.random_state)
        self.model_ = self._build(X.shape[1], T.shape[1], rng)
        opt = self._optimizer()
        self.loss_history_, self.val_history_ = [], []
        for epoch in range(self.epochs):
            self.model_.train(True)
            order = rng.permutation(len(X))
            total = 0.0
            for start in range(0, len(X), self.batch_size):
                idx = order[start : start + self.batch_size]
                out = self.model_.forward(X[idx])
                loss, grad = loss_fn(out, T[idx])
                self.model_.backward(grad)
                params, grads = self.model_.parameters()
                opt.step(params, grads)
                total += loss * len(idx)
            self.loss_history_.append(total / len(X))
            if X_val is not None:
                self.model_.train(False)
                self.val_history_.append(loss_fn(self.model_.forward(X_val), T_val)[0])
            if self.verbose and (epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1):
                extra = f"  val={self.val_history_[-1]:.4f}" if self.val_history_ else ""
                print(f"epoch {epoch + 1:>4}/{self.epochs}  loss={self.loss_history_[-1]:.4f}{extra}")
        self.model_.train(False)
        return self


class MLPClassifier(_MLPBase, ClassifierMixin):
    def fit(self, X, y, X_val=None, y_val=None):
        X, y = check_Xy(X, y)
        Y = one_hot(self._encode_labels(y), len(self.classes_))
        Yv = None
        if X_val is not None:
            Yv = one_hot(np.searchsorted(self.classes_, y_val), len(self.classes_))
        return self._train(X, Y, softmax_cross_entropy, None if X_val is None else as_2d(X_val), Yv)

    def predict_proba(self, X):
        return softmax(self.model_.forward(as_2d(X)))


class MLPRegressor(_MLPBase, RegressorMixin):
    def fit(self, X, y):
        X, y = check_Xy(X, y)
        return self._train(X, y.reshape(-1, 1).astype(float), mse_loss)

    def predict(self, X):
        return self.model_.forward(as_2d(X)).ravel()


# ---------------------------------------------------------------- convolution (for the CNN lesson)


def im2col(x, kh, kw, stride=1, pad=0):
    """Unroll every kh×kw patch into a row so convolution becomes one matrix product."""
    n, c, h, w = x.shape
    x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    oh, ow = (h + 2 * pad - kh) // stride + 1, (w + 2 * pad - kw) // stride + 1
    cols = np.empty((n, c, kh, kw, oh, ow))
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j] = x[:, :, i : i + stride * oh : stride, j : j + stride * ow : stride]
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(n * oh * ow, -1), oh, ow


def col2im(cols, x_shape, kh, kw, oh, ow, stride=1, pad=0):
    n, c, h, w = x_shape
    cols = cols.reshape(n, oh, ow, c, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    x = np.zeros((n, c, h + 2 * pad, w + 2 * pad))
    for i in range(kh):
        for j in range(kw):
            x[:, :, i : i + stride * oh : stride, j : j + stride * ow : stride] += cols[:, :, i, j]
    return x[:, :, pad : pad + h, pad : pad + w]


class Conv2D(Layer):
    """out[n, f, i, j] = Σ_c Σ_u Σ_v x[n, c, i·s+u, j·s+v] · W[f, c, u, v] + b[f]   (via im2col)."""

    def __init__(self, in_ch, out_ch, k=3, stride=1, pad=1, rng=None):
        rng = rng_from(rng)
        self.params = {"W": rng.normal(0, np.sqrt(2 / (in_ch * k * k)), (out_ch, in_ch, k, k)), "b": np.zeros(out_ch)}
        self.grads, self.k, self.stride, self.pad = {}, k, stride, pad

    def forward(self, x):
        self.x_shape = x.shape
        self.cols, oh, ow = im2col(x, self.k, self.k, self.stride, self.pad)
        self.oh, self.ow = oh, ow
        W = self.params["W"].reshape(len(self.params["W"]), -1)
        out = self.cols @ W.T + self.params["b"]
        return out.reshape(x.shape[0], oh, ow, -1).transpose(0, 3, 1, 2)

    def backward(self, grad):
        F = self.params["W"].shape[0]
        g = grad.transpose(0, 2, 3, 1).reshape(-1, F)
        self.grads = {"W": (g.T @ self.cols).reshape(self.params["W"].shape), "b": g.sum(0)}
        dcols = g @ self.params["W"].reshape(F, -1)
        return col2im(dcols, self.x_shape, self.k, self.k, self.oh, self.ow, self.stride, self.pad)


class MaxPool2D(Layer):
    def __init__(self, size=2):
        self.size = size

    def forward(self, x):
        n, c, h, w = x.shape
        s = self.size
        xr = x[:, :, : h // s * s, : w // s * s].reshape(n, c, h // s, s, w // s, s)
        out = xr.max(axis=(3, 5))
        self.mask = xr == out[:, :, :, None, :, None]
        self.x_shape = x.shape
        return out

    def backward(self, grad):
        n, c, h, w = self.x_shape
        s = self.size
        g = self.mask * grad[:, :, :, None, :, None]
        out = np.zeros(self.x_shape)
        out[:, :, : h // s * s, : w // s * s] = g.reshape(n, c, h // s * s, w // s * s)
        return out


class Flatten(Layer):
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(len(x), -1)

    def backward(self, grad):
        return grad.reshape(self.shape)


class SimpleRNN(Layer):
    """hₜ = tanh(xₜWx + hₜ₋₁Wh + b); returns the last hidden state. Backprop through time (BPTT)."""

    def __init__(self, n_in, n_hidden, rng=None):
        rng = rng_from(rng)
        self.params = {"Wx": rng.normal(0, np.sqrt(1 / n_in), (n_in, n_hidden)),
                       "Wh": rng.normal(0, np.sqrt(1 / n_hidden), (n_hidden, n_hidden)), "b": np.zeros(n_hidden)}
        self.grads = {}

    def forward(self, x):  # x: (n, T, n_in)
        n, T, _ = x.shape
        self.x, self.hs = x, [np.zeros((n, self.params["Wh"].shape[0]))]
        for t in range(T):
            self.hs.append(np.tanh(x[:, t] @ self.params["Wx"] + self.hs[-1] @ self.params["Wh"] + self.params["b"]))
        return self.hs[-1]

    def backward(self, grad):
        gWx, gWh, gb = (np.zeros_like(self.params[k]) for k in ("Wx", "Wh", "b"))
        dx = np.zeros_like(self.x)
        dh = grad
        for t in reversed(range(self.x.shape[1])):
            dz = dh * (1 - self.hs[t + 1] ** 2)
            gWx += self.x[:, t].T @ dz
            gWh += self.hs[t].T @ dz
            gb += dz.sum(0)
            dx[:, t] = dz @ self.params["Wx"].T
            dh = dz @ self.params["Wh"].T
        self.grads = {"Wx": gWx, "Wh": gWh, "b": gb}
        return dx
