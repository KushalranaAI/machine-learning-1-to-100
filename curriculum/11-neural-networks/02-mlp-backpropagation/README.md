---
title: Multi-Layer Perceptrons & Backpropagation
order: 2
summary: Dense layers, activation functions, the forward pass, backpropagation derived layer by layer, softmax + cross-entropy, gradient checking, and the universal approximation theorem.
level: advanced
minutes: 70
why: Backpropagation is the algorithm that trains every neural network, from a 2-layer MLP to a large language model. Deriving it once - and checking it numerically - removes the magic from deep learning.
background: Backprop was discovered several times (Linnainmaa 1970 as reverse-mode autodiff, Werbos 1974) and popularised for neural nets by Rumelhart, Hinton and Williams (1986). Cybenko (1989) and Hornik (1991) proved the universal approximation theorem.
use_cases:
  - Classification and regression on tabular, image, audio and text data
  - The core of every deep learning model
where_used:
  - 05-digit-recognizer
prerequisites:
  - 11-neural-networks/01-perceptron
  - 00-math-foundations/02-calculus-gradients
  - 08-classification/02-softmax-regression
code:
  - mlscratch/neural_network.py:Dense
  - mlscratch/neural_network.py:ReLU
  - mlscratch/neural_network.py:softmax_cross_entropy
  - mlscratch/neural_network.py:Sequential
  - mlscratch/neural_network.py:gradient_check
  - mlscratch/neural_network.py:MLPClassifier
  - cpp/include/mlcpp/neural_network.hpp
demo: demo.py
widgets:
  - neural-network
pros:
  - Learns its own features; approximates any continuous function
  - Same training algorithm for any architecture
cons:
  - Many hyper-parameters, non-convex loss, needs more data
  - Less interpretable than linear models or trees
workflow: |
  flowchart LR
    A[mini-batch X] --> B[forward: zˡ = aˡ⁻¹Wˡ + bˡ, aˡ = φ zˡ]
    B --> C[loss L]
    C --> D[δᴸ = ∂L/∂zᴸ = P − Y / n]
    D --> E[backward each layer: ∂L/∂W = aᵀδ, δ ← δWᵀ ⊙ φ′]
    E --> F[optimizer step on every W, b]
    F --> A
---

## 1. Layers
A **dense (fully connected) layer** maps a batch $A^{(l-1)} \in \mathbb{R}^{m\times n_{l-1}}$ to
$$
Z^{(l)} = A^{(l-1)}W^{(l)} + b^{(l)}, \qquad A^{(l)} = \varphi(Z^{(l)})
$$
with an element-wise non-linearity $\varphi$. Without it, stacked layers collapse to a single linear map ($W_2W_1x$).

| Activation | $\varphi(z)$ | $\varphi'(z)$ | Notes |
|------------|-------------|---------------|-------|
| sigmoid | $\frac{1}{1+e^{-z}}$ | $\sigma(1-\sigma)$ | saturates; max gradient 0.25 → vanishing gradients |
| tanh | $\tanh z$ | $1-\tanh^2 z$ | zero-centred, still saturates |
| **ReLU** | $\max(0, z)$ | $[z > 0]$ | cheap, no saturation for $z>0$; default |
| Leaky ReLU | $\max(\alpha z, z)$ | $1$ or $\alpha$ | avoids "dead" ReLUs |

## 2. Output layer and loss
For $K$ classes: logits $Z^{(L)}$, probabilities $P = \text{softmax}(Z^{(L)})$, cross-entropy $L = -\frac1m\sum_i\sum_k Y_{ik}\log P_{ik}$.
The combined gradient is the famous
$$
\delta^{(L)} = \frac{\partial L}{\partial Z^{(L)}} = \frac{1}{m}(P - Y)
$$
(`softmax_cross_entropy` returns exactly this). For regression: linear output + MSE, $\delta^{(L)} = \frac{2}{m}(\hat y - y)$.

## 3. Backpropagation, derived
Define $\delta^{(l)} = \partial L/\partial Z^{(l)}$. Given $\delta^{(l)}$, the chain rule gives three things:
$$
\frac{\partial L}{\partial W^{(l)}} = A^{(l-1)\top}\delta^{(l)}, \qquad \frac{\partial L}{\partial b^{(l)}} = \sum_{\text{batch}}\delta^{(l)}, \qquad \frac{\partial L}{\partial A^{(l-1)}} = \delta^{(l)}W^{(l)\top}
$$
and then through the activation of the previous layer:
$$
\delta^{(l-1)} = \Big(\delta^{(l)}W^{(l)\top}\Big)\odot\varphi'\big(Z^{(l-1)}\big)
$$
Why the transposes? $Z = AW$ means each $Z_{ij} = \sum_k A_{ik}W_{kj}$, so $\partial L/\partial W_{kj} = \sum_i A_{ik}\,\delta_{ij}$ = $(A^\top\delta)_{kj}$. Shapes always tell you where the transpose goes.

**In code**, every layer stores what it needs in `forward()` (Dense keeps its input, ReLU keeps its mask) and
`backward(grad)` turns $\partial L/\partial\text{output}$ into $\partial L/\partial\text{input}$, filling `self.grads`.
`Sequential.backward` just calls the layers in reverse. That's the whole algorithm.

Cost: backward ≈ 2× forward - computing all gradients costs about as much as evaluating the network. That efficiency
(reverse-mode automatic differentiation) is why deep learning is feasible.

## 4. Check your gradients
Always compare backprop with finite differences on a tiny network: $\frac{L(\theta+\epsilon) - L(\theta-\epsilon)}{2\epsilon}$.
`gradient_check` returns the maximum relative error; ~$10^{-7}$ means correct, ~$10^{-2}$ means a bug.

## 5. Universal approximation
A network with one hidden layer of enough units (and a non-polynomial activation) can approximate any continuous
function on a compact set arbitrarily well. Depth makes it **efficient**: some functions need exponentially many units
in one layer but only polynomially many with more layers.

## Exercises
1. In the widget, train on "circles" with 2 hidden units, then 8. Why does 2 fail?
2. Run the demo; break `ReLU.backward` on purpose and watch the gradient check fail.
3. Derive $\delta^{(L)} = P - Y$ for softmax + cross-entropy by hand.
