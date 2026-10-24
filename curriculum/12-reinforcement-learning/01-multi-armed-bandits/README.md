---
title: Multi-Armed Bandits
order: 1
summary: The exploration-exploitation dilemma in its purest form - action values, incremental means, ε-greedy, UCB1, Thompson sampling and regret.
level: intermediate
minutes: 40
why: Every A/B test, recommendation slot and ad auction is a bandit problem - and the exploration strategies here (optimism, randomisation, posterior sampling) reappear in all of reinforcement learning.
background: Thompson proposed posterior sampling in 1933 for clinical trials. Robbins formalised bandits in 1952; Lai & Robbins (1985) proved regret lower bounds; Auer et al. introduced UCB1 (2002).
use_cases:
  - Adaptive A/B testing (send more traffic to the winning variant)
  - News/ads/recommendation slot selection
  - Clinical trials, dynamic pricing
where_used:
  - 05-digit-recognizer
code:
  - mlscratch/rl.py:MultiArmedBandit
demo: demo.py
workflow: |
  flowchart LR
    A[estimates Q a, counts N a] --> B{choose arm}
    B -->|ε-greedy| C[random with prob ε else argmax Q]
    B -->|UCB| D[argmax Q + c√ ln t / N]
    B -->|Thompson| E[sample from posterior, argmax]
    C & D & E --> F[pull, observe reward r]
    F --> G[Q ← Q + r − Q / N]
    G --> A
---

## The problem
$k$ slot machines ("arms"), each with an unknown mean reward $q_*(a)$. Each round you pull one and observe a noisy reward.
Goal: maximise total reward - equivalently minimise **regret**
$$
\text{Regret}(T) = T\,q_* - \sum_{t=1}^T q_*(a_t), \qquad q_* = \max_a q_*(a)
$$
## Estimating action values incrementally
The sample mean $Q_n = \frac1n\sum_{i=1}^n r_i$ can be updated without storing rewards:
$$
Q_{n+1} = Q_n + \frac{1}{n+1}\big(r_{n+1} - Q_n\big) \qquad \text{new estimate = old + step × (target − old)}
$$
This "move towards the target" form is the template for every RL update that follows.

## Strategies
**Greedy** - always pull the arm with the best estimate. Can lock onto a mediocre arm forever after an unlucky start.

**ε-greedy** - explore uniformly with probability ε, otherwise exploit. Simple; keeps exploring forever (linear regret unless ε decays).

**UCB1 - optimism in the face of uncertainty:**
$$
a_t = \arg\max_a\left[Q_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}}\right]
$$
The bonus is large for rarely tried arms and shrinks as they're sampled; arms are explored only as long as they *might*
be best. Regret grows only logarithmically.

**Thompson sampling** - keep a posterior over each arm's mean, sample a plausible mean from each, play the best sample.
Arms are explored in proportion to the probability that they're optimal. Often the best in practice.

## Exercises
1. Run the demo; compare the cumulative regret curves. Which strategies have logarithmic regret?
2. Make two arms very close in value; how does that change the regret?
3. Implement ε-greedy with a decaying ε = 1/t and compare.
