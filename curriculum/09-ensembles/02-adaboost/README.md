---
title: AdaBoost
order: 2
summary: Adaptive boosting - re-weight misclassified samples, weight each weak learner by its accuracy, and why it minimises exponential loss. SAMME for multi-class.
level: intermediate
minutes: 40
why: AdaBoost was the first practical boosting algorithm and the clearest illustration of the boosting idea - a sequence of weak learners, each fixing the mistakes of the previous ones, combined into a strong learner.
background: Freund and Schapire introduced AdaBoost in 1995 (Gödel Prize 2003), answering a theoretical question of whether "weak" learners can be boosted into strong ones. Friedman, Hastie and Tibshirani (2000) showed it is stagewise additive modelling with exponential loss. Viola-Jones face detection (2001) used it in every digital camera.
use_cases:
  - Face detection (Viola-Jones cascades)
  - Classification with simple, fast weak learners
  - Teaching boosting before gradient boosting
where_used:
  - 02-titanic-survival
prerequisites:
  - 08-classification/06-decision-trees
code:
  - mlscratch/ensemble.py:AdaBoostClassifier
demo: demo.py
cons:
  - Sensitive to noisy labels and outliers (they get exponentially large weights)
  - Sequential - can't be parallelised across rounds
workflow: |
  flowchart LR
    A[equal weights wᵢ = 1/n] --> B[fit stump on weighted data]
    B --> C[weighted error errₘ]
    C --> D[αₘ = log 1−err / err + log K−1]
    D --> E[up-weight misclassified: wᵢ ← wᵢ e^αₘ]
    E --> B
    D --> F[final: weighted vote Σ αₘ hₘ x]
---

## The algorithm (SAMME, works for K classes; K = 2 is classic AdaBoost)
Start with uniform sample weights $w_i = 1/n$. For $m = 1..M$:
1. Fit a weak learner $h_m$ (usually a depth-1 tree, a "stump") to the **weighted** data.
2. Weighted error: $\;\text{err}_m = \frac{\sum_i w_i\,[h_m(x_i)\ne y_i]}{\sum_i w_i}$
3. Learner weight: $\;\alpha_m = \log\frac{1-\text{err}_m}{\text{err}_m} + \log(K-1)$ - better learners get a bigger say.
4. Re-weight: $\;w_i \leftarrow w_i\cdot e^{\alpha_m[h_m(x_i)\ne y_i]}$, then normalise - mistakes become more important.

Prediction: $\;H(x) = \arg\max_k \sum_m \alpha_m\,[h_m(x) = k]$.

The learner must beat chance: $\text{err}_m < 1 - 1/K$, otherwise $\alpha_m \le 0$ and boosting stops.

## Why it works: exponential loss
For $K = 2$ with $y\in\{-1, 1\}$ and $F(x) = \sum_m \alpha_m h_m(x)$, AdaBoost is **forward stagewise** minimisation of
$$
L = \sum_i e^{-y_iF(x_i)}
$$
Adding $\alpha h$ to $F$: the weights $w_i \propto e^{-y_iF(x_i)}$ appear naturally, the best $h$ minimises weighted error,
and solving $\frac{\partial L}{\partial\alpha} = 0$ gives exactly $\alpha = \tfrac12\log\frac{1-\text{err}}{\text{err}}$ (SAMME's $\alpha$ is twice that, which doesn't change the vote).
The exponential loss also explains the weakness: a mislabelled point with large negative margin gets an exponentially large weight.

## Training error bound
If every learner has error $\le \tfrac12 - \gamma$, the training error after $M$ rounds is at most $e^{-2\gamma^2M}$ - it falls exponentially.

## Exercises
1. Run the demo: watch the weights of hard examples grow and the training error fall round by round.
2. Flip 10% of the labels (noise) and compare AdaBoost with a random forest.
3. Change the weak learner to depth-2 trees. How many rounds do you need now?
