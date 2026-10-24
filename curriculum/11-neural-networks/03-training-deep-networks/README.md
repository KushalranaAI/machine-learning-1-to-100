---
title: Training Deep Networks
order: 3
summary: Weight initialisation (Xavier/He), vanishing and exploding gradients, SGD vs momentum vs Adam in practice, dropout, batch normalisation, weight decay and early stopping.
level: advanced
minutes: 55
why: A correct network can still fail to train - activations explode, gradients vanish, the loss plateaus, or the model memorises the training set. These techniques are what make deep networks trainable and generalise.
background: Glorot & Bengio (2010) explained initialisation; He et al. (2015) adapted it for ReLU. Dropout (Srivastava, Hinton et al., 2014), batch normalisation (Ioffe & Szegedy, 2015) and Adam (Kingma & Ba, 2014) made deep training routine.
use_cases:
  - Every deep-learning training run
  - Debugging a network that won't learn or overfits
where_used:
  - 05-digit-recognizer
prerequisites:
  - 11-neural-networks/02-mlp-backpropagation
  - 00-math-foundations/04-optimization-gradient-descent
code:
  - mlscratch/neural_network.py:Dense.__init__
  - mlscratch/neural_network.py:Dropout
  - mlscratch/neural_network.py:BatchNorm
  - mlscratch/optim.py:Adam
demo: demo.py
widgets:
  - gradient-descent
workflow: |
  flowchart TD
    A[loss not decreasing] --> B{check}
    B --> C[gradients ~0 in early layers → vanishing: ReLU, He init, batch norm]
    B --> D[loss NaN / exploding → lower lr, gradient clipping, scale inputs]
    E[train ≪ validation loss] --> F[overfitting → dropout, weight decay, early stopping, more data]
---

## Initialisation
If weights are too small, activations shrink layer after layer to 0; too large and they explode. Choose the variance so
signals keep roughly constant scale through the network:
$$
\text{Xavier/Glorot (tanh, sigmoid)}: \mathrm{Var}(W) = \frac{2}{n_{in} + n_{out}}, \qquad \text{He (ReLU)}: \mathrm{Var}(W) = \frac{2}{n_{in}}
$$
*Derivation sketch (He):* for $z = \sum_{j=1}^{n_{in}} w_ja_j$ with independent zero-mean weights,
$\mathrm{Var}(z) = n_{in}\mathrm{Var}(w)\mathbb{E}[a^2]$. ReLU zeroes half its inputs, so $\mathbb E[a^2] = \tfrac12\mathrm{Var}(z_{prev})$;
keeping $\mathrm{Var}(z)$ constant needs $\mathrm{Var}(w) = 2/n_{in}$. Never initialise all weights to zero - every
unit in a layer would compute the same thing forever (symmetry).

## Vanishing and exploding gradients
Backprop multiplies by $W^\top$ and $\varphi'$ at every layer. Sigmoid's $\varphi' \le 0.25$, so after 10 layers the
gradient is scaled by ≤ $0.25^{10}\approx 10^{-6}$ - early layers stop learning. Fixes: ReLU-family activations,
proper initialisation, batch normalisation, residual connections; for explosions, gradient clipping.

## Optimisers (see Part 0)
SGD with momentum and **Adam** are the defaults. Adam adapts a per-parameter step size and works with little tuning
(lr ≈ 1e-3); well-tuned SGD+momentum sometimes generalises slightly better. Use a learning-rate schedule (decay or
warm-up + cosine) for long runs.

## Regularisation
| Technique | How | Implementation |
|-----------|-----|----------------|
| **Weight decay (L2)** | add $\lambda\|W\|^2$ → gradient $+2\lambda W$ | `weight_decay=` in SGD/Adam |
| **Dropout** | during training, zero each activation with probability $p$, scale the rest by $\frac{1}{1-p}$ ("inverted dropout") | `Dropout` - identity at inference |
| **Early stopping** | stop when validation loss stops improving; keep the best weights | track `val_history_` |
| Data augmentation | random shifts/rotations/noise of inputs | (images) |

Dropout trains an implicit ensemble of thinned networks; at test time using all units with the $\frac{1}{1-p}$ scaling
approximates averaging them.

## Batch normalisation
Normalise each feature over the mini-batch, then let the network rescale it:
$$
\hat x = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y = \gamma\hat x + \beta
$$
At inference, running averages of $\mu$ and $\sigma^2$ replace the batch statistics. Allows higher learning rates and
reduces sensitivity to initialisation. Its backward pass (in `BatchNorm.backward`) is a nice exercise in the chain rule.

## A debugging checklist
1. Overfit a tiny subset (e.g. 50 samples) first - if you can't reach ~0 loss, there's a bug.
2. Check gradients numerically on a tiny model.
3. Plot training and validation loss every epoch.
4. Scale inputs; start with Adam lr=1e-3.

## Exercises
1. Run the demo: compare activation statistics through 10 layers for bad, Xavier and He initialisation.
2. Find the epoch where early stopping would stop training on the digits.
3. Train the same network with dropout 0, 0.2 and 0.5; compare the train-validation gap.
