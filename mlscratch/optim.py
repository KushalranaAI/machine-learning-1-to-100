"""Gradient-based optimisers, written as plain update rules (used by the neural network and the lessons).

All of them minimise f(θ) using the gradient g = ∇f(θ):
    SGD:       θ ← θ - η g
    Momentum:  v ← βv + g;            θ ← θ - η v
    RMSProp:   s ← ρs + (1-ρ)g²;      θ ← θ - η g / (√s + ε)
    Adam:      m ← β₁m + (1-β₁)g;  v ← β₂v + (1-β₂)g²;  m̂ = m/(1-β₁ᵗ), v̂ = v/(1-β₂ᵗ);  θ ← θ - η m̂/(√v̂ + ε)
"""

from __future__ import annotations

import numpy as np


class SGD:
    def __init__(self, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.lr, self.momentum, self.weight_decay, self.v = lr, momentum, weight_decay, {}

    def step(self, params: dict, grads: dict):
        for k in params:
            g = grads[k] + self.weight_decay * params[k]
            if self.momentum:
                self.v[k] = self.momentum * self.v.get(k, 0) + g
                g = self.v[k]
            params[k] -= self.lr * g


class RMSProp:
    def __init__(self, lr=0.001, rho=0.9, eps=1e-8):
        self.lr, self.rho, self.eps, self.s = lr, rho, eps, {}

    def step(self, params, grads):
        for k in params:
            self.s[k] = self.rho * self.s.get(k, 0) + (1 - self.rho) * grads[k] ** 2
            params[k] -= self.lr * grads[k] / (np.sqrt(self.s[k]) + self.eps)


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr, self.b1, self.b2, self.eps, self.weight_decay = lr, beta1, beta2, eps, weight_decay
        self.m, self.v, self.t = {}, {}, 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m.get(k, 0) + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v.get(k, 0) + (1 - self.b2) * g**2
            m_hat = self.m[k] / (1 - self.b1**self.t)          # bias correction: m, v start at 0
            v_hat = self.v[k] / (1 - self.b2**self.t)
            params[k] -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * params[k])


def minimize(grad_fn, theta0, optimizer, n_steps=100):
    """Run an optimiser on a function given its gradient; returns the path (for the lessons' plots)."""
    params = {"theta": np.array(theta0, float)}
    path = [params["theta"].copy()]
    for _ in range(n_steps):
        optimizer.step(params, {"theta": grad_fn(params["theta"])})
        path.append(params["theta"].copy())
    return np.array(path)
