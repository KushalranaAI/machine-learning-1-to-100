---
title: Recurrent Neural Networks
order: 5
summary: Sequences with a hidden state - the vanilla RNN, unrolling, backpropagation through time, vanishing gradients over long sequences, and why LSTMs/GRUs (and transformers) followed.
level: advanced
minutes: 45
why: Text, speech, sensor streams and time series are sequences whose meaning depends on order. RNNs process them step by step with a memory, and understanding their training (and its limits) explains the design of LSTMs, GRUs and the attention mechanism.
background: Elman (1990) and Jordan (1986) introduced simple recurrent networks. Hochreiter's 1991 thesis analysed vanishing gradients; LSTMs (Hochreiter & Schmidhuber, 1997) solved them. Transformers (Vaswani et al., 2017) replaced recurrence with attention.
use_cases:
  - Time-series forecasting and anomaly detection
  - Character/word-level language modelling
  - Speech and handwriting recognition (historically)
where_used:
  - 05-digit-recognizer
prerequisites:
  - 11-neural-networks/02-mlp-backpropagation
code:
  - mlscratch/neural_network.py:SimpleRNN
demo: demo.py
cons:
  - Sequential computation (slow); vanishing/exploding gradients over long sequences
workflow: |
  flowchart LR
    X1[x₁] --> H1[h₁]
    H0[h₀=0] --> H1
    X2[x₂] --> H2[h₂]
    H1 --> H2
    X3[x_T] --> H3[h_T]
    H2 -.-> H3
    H3 --> O[Dense → prediction]
---

## The recurrence
The same weights are applied at every time step, carrying a hidden state $h_t$ (the memory):
$$
h_t = \tanh(x_tW_x + h_{t-1}W_h + b), \qquad \hat y = h_TW_o + b_o
$$
Weight sharing over time is to sequences what convolution is to images.

## Backpropagation through time (BPTT)
Unroll the network over $T$ steps - it becomes a deep feed-forward network with shared weights - and apply ordinary
backprop. With $\delta_t = \partial L/\partial h_t$ and $z_t$ the pre-activation:
$$
\frac{\partial L}{\partial z_t} = \delta_t\odot(1 - h_t^2), \quad \frac{\partial L}{\partial W_x} \mathrel{+}= x_t^\top\frac{\partial L}{\partial z_t}, \quad \frac{\partial L}{\partial W_h} \mathrel{+}= h_{t-1}^\top\frac{\partial L}{\partial z_t}, \quad \delta_{t-1} = \frac{\partial L}{\partial z_t}W_h^\top
$$
Gradients for the shared weights are **summed** over time steps (`SimpleRNN.backward`).

## Vanishing and exploding gradients
The gradient reaching step $t-k$ contains the product $\prod_{j}\big(\text{diag}(1-h_j^2)W_h^\top\big)$ of $k$ Jacobians.
If the largest singular value of $W_h$ (times the tanh slopes) is < 1, it shrinks exponentially - the network can't
learn long-range dependencies; if > 1 it explodes (fix: gradient clipping).

**LSTM** cells add a separate cell state updated additively and controlled by gates (forget, input, output), so
information and gradients can flow over many steps. **GRUs** are a simpler variant. **Transformers** drop recurrence
entirely and let every position attend to every other directly.

## Exercises
1. Run the demo: at what sequence length does the vanilla RNN fail the "remember the first value" task?
2. Measure the gradient norm reaching the first time step as the sequence grows.
3. Add gradient clipping to the training loop and check whether long sequences train more stably.
