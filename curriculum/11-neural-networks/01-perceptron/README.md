---
title: The Perceptron
order: 1
summary: Rosenblatt's neuron - weighted sum, step activation, the mistake-driven learning rule, the convergence theorem, and the XOR problem that motivated multi-layer networks.
level: beginner
minutes: 30
why: The perceptron is the ancestor of every neural network. Its learning rule, its convergence guarantee and its famous failure on XOR explain why we need hidden layers and smooth activations.
background: McCulloch and Pitts modelled a neuron in 1943; Rosenblatt built the perceptron (and the Mark I hardware) in 1958. Minsky and Papert's 1969 book showed its limits (XOR), contributing to the first "AI winter".
use_cases:
  - Historical foundation of neural networks
  - Online linear classification of streaming data
  - The building block (a neuron) of every MLP
where_used:
  - 05-digit-recognizer
prerequisites:
  - 08-classification/01-logistic-regression
code:
  - mlscratch/linear_model.py:Perceptron
demo: demo.py
cons:
  - Only linearly separable problems; no probabilities
  - Never converges (keeps oscillating) on non-separable data
workflow: |
  flowchart LR
    A[x] --> B[z = wᵀx + b]
    B --> C[ŷ = step z]
    C --> D{mistake?}
    D -->|yes| E[w ← w + η y x]
    D -->|no| F[next sample]
    E --> F
---

## The model
$$
\hat y = \text{sign}(w^\top x + b) \in \{-1, +1\}
$$
A weighted sum of inputs followed by a hard threshold - a crude model of a biological neuron firing when stimulation exceeds a threshold.

## The learning rule
For each training sample $(x_i, y_i)$, if it is misclassified ($y_i(w^\top x_i + b) \le 0$):
$$
w \leftarrow w + \eta\,y_ix_i, \qquad b \leftarrow b + \eta\,y_i
$$
Intuition: the update increases $y_i(w^\top x_i + b)$ by $\eta(\|x_i\|^2 + 1) > 0$ - it pushes the boundary towards classifying $x_i$ correctly.
This is stochastic subgradient descent on the "perceptron loss" $\max(0, -y\,f(x))$.

## Convergence theorem (Novikoff, 1962)
If the data is linearly separable with margin $\gamma$ (some unit $w^*$ has $y_i\,w^{*\top}x_i \ge \gamma$) and all $\|x_i\|\le R$,
the perceptron makes at most $(R/\gamma)^2$ mistakes. *Proof idea:* every mistake increases $w^\top w^*$ by at least
$\gamma$ but increases $\|w\|^2$ by at most $R^2$; since $w^\top w^* \le \|w\|$, the number of mistakes $k$ satisfies
$k\gamma \le \sqrt{k}R$.

## The XOR problem
No single line separates $\{(0,0), (1,1)\}$ from $\{(0,1), (1,0)\}$. A perceptron can never learn XOR.
**Two layers can:** $\text{XOR}(a, b) = \text{AND}(\text{OR}(a,b), \text{NAND}(a,b))$ - three perceptrons. Hidden layers
create new features in which the problem becomes linearly separable. Training them requires a differentiable
activation instead of the step function - which leads to backpropagation (next lesson).

## Exercises
1. Run the demo; compare the number of mistakes with the $(R/\gamma)^2$ bound.
2. Hand-pick weights for the three-perceptron XOR network and verify them.
3. Run the perceptron on non-separable data and plot errors per epoch - does it settle?
