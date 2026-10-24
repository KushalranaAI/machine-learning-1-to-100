---
title: MDPs, Bellman Equations & Dynamic Programming
order: 2
summary: Markov decision processes, returns and discounting, value functions, the Bellman expectation and optimality equations, value iteration and policy iteration with convergence arguments.
level: advanced
minutes: 55
why: Every RL algorithm is an approximation of the Bellman equations. When the environment's model is known, dynamic programming solves them exactly - the gold standard that learning methods like Q-learning try to reach from experience.
background: Bellman formulated dynamic programming and his equation in the 1950s; Howard introduced policy iteration (1960). Sutton & Barto's textbook (1998, 2018) is the classic reference.
use_cases:
  - Planning with a known model (inventory control, routing, robotics in simulation)
  - Computing optimal policies to benchmark learning agents
where_used:
  - 05-digit-recognizer
prerequisites:
  - 12-reinforcement-learning/01-multi-armed-bandits
code:
  - mlscratch/rl.py:GridWorld
  - mlscratch/rl.py:value_iteration
  - mlscratch/rl.py:policy_iteration
demo: demo.py
workflow: |
  flowchart LR
    A[MDP: S, A, P, R, γ] --> B{method}
    B --> C[value iteration: V ← max_a Σ P r + γV]
    B --> D[policy iteration: evaluate π exactly, then act greedily]
    C & D --> E[V* and π*]
---

## Markov decision process
$(\mathcal S, \mathcal A, P, R, \gamma)$: states, actions, transition probabilities $P(s'\mid s, a)$, rewards $R(s, a, s')$
and a discount $\gamma\in[0, 1)$. **Markov property:** the future depends only on the current state, not the history.

The agent maximises the **return** $G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots$. Discounting keeps the sum finite
and makes near rewards worth more ($\gamma = 0.99$ ≈ "care about the next ~100 steps").

## Value functions
$$
V^\pi(s) = \mathbb E_\pi[G_t\mid s_t = s], \qquad Q^\pi(s, a) = \mathbb E_\pi[G_t\mid s_t = s, a_t = a]
$$
## Bellman equations
Splitting the return into the first reward plus the discounted rest gives recursive equations:
$$
V^\pi(s) = \sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)\big[R(s,a,s') + \gamma V^\pi(s')\big] \qquad\text{(expectation)}
$$
$$
V^*(s) = \max_a\sum_{s'}P(s'\mid s,a)\big[R(s,a,s') + \gamma V^*(s')\big] \qquad\text{(optimality)}
$$
The optimal policy acts greedily with respect to $V^*$: $\pi^*(s) = \arg\max_a\sum_{s'}P(s'\mid s,a)[R + \gamma V^*(s')]$.

## Value iteration
Turn the optimality equation into an update and sweep until it stops changing:
$$
V_{k+1}(s) \leftarrow \max_a\sum_{s'}P(s'\mid s,a)\big[R(s,a,s') + \gamma V_k(s')\big]
$$
**Why it converges:** the Bellman operator is a $\gamma$-contraction in the max-norm, $\|TV - TU\|_\infty \le \gamma\|V - U\|_\infty$,
so by Banach's fixed-point theorem the iterates converge to the unique fixed point $V^*$ - the error shrinks by a factor $\gamma$ per sweep.

## Policy iteration
1. **Evaluate** the current policy exactly - the expectation equation is linear, so solve $(I - \gamma P_\pi)V = R_\pi$.
2. **Improve**: make the policy greedy with respect to $V$.
3. Repeat until the policy stops changing. The policy improvement theorem guarantees each step is at least as good,
   and there are finitely many policies, so it converges - typically in very few iterations.

## The grid world used here
`mlscratch.rl.GridWorld`: `S` start, `G` goal (+1), `H` holes (−1), `#` walls, a small cost per step, and optional
**slip** probability (the agent moves sideways). Slip makes the optimal policy keep away from the holes.

## Exercises
1. Run the demo; compare the policies for slip = 0 and slip = 0.3. Why do they differ?
2. Plot $\|V_{k+1} - V_k\|_\infty$ per sweep on a log scale - is the decay rate close to γ?
3. Change γ to 0.5. What happens to the agent's willingness to take a long safe route?
