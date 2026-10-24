---
title: Q-Learning & SARSA
order: 3
summary: Learning optimal behaviour from experience without a model - temporal-difference learning, the Q-learning and SARSA updates, ε-greedy exploration, on- vs off-policy, and the bridge to deep Q-networks.
level: advanced
minutes: 50
why: In most real problems you don't know the transition probabilities. Q-learning learns optimal action values purely from trial and error, and with neural networks in place of the table it became DQN, which learned to play Atari games from pixels.
background: Temporal-difference learning (Sutton, 1988), Q-learning (Watkins, 1989; convergence proof 1992), SARSA (Rummery & Niranjan, 1994). DeepMind's DQN (Mnih et al., 2015) scaled Q-learning with deep networks.
use_cases:
  - Game playing, robotics control, resource allocation
  - Recommendation and ad bidding as sequential decisions
  - Any problem with a simulator but no explicit model
where_used:
  - 05-digit-recognizer
prerequisites:
  - 12-reinforcement-learning/02-mdps-dynamic-programming
code:
  - mlscratch/rl.py:epsilon_greedy
  - mlscratch/rl.py:q_learning
demo: demo.py
widgets:
  - q-learning
pros:
  - Model-free; converges to the optimal Q* (tabular, enough exploration, decaying α)
cons:
  - Tables don't scale to large/continuous states (→ function approximation, DQN)
  - Sample-inefficient; sensitive to α, γ, ε
workflow: |
  flowchart LR
    A[state s] --> B[ε-greedy action a]
    B --> C[env: reward r, next state s′]
    C --> D[TD target: r + γ max Q s′,·  or  r + γ Q s′,a′]
    D --> E[Q s,a ← Q s,a + α target − Q s,a]
    E --> A
---

## Temporal-difference learning
Monte Carlo methods wait until the end of an episode to see the return. TD methods **bootstrap**: they update an estimate
using another estimate, after a single step. The core idea is the TD error
$$
\delta = \underbrace{r + \gamma\,V(s')}_{\text{TD target}} - V(s)
$$
and the update $V(s) \leftarrow V(s) + \alpha\,\delta$ - the same "move towards the target" form as the bandit average.

## Q-learning (off-policy)
$$
Q(s, a) \leftarrow Q(s, a) + \alpha\Big[r + \gamma\max_{a'}Q(s', a') - Q(s, a)\Big]
$$
It is a sampled, incremental version of the Bellman **optimality** equation: the target uses the best next action,
whatever the agent actually does next. So Q-learning learns $Q^*$ (the greedy policy's values) while behaving
ε-greedily - **off-policy**. Watkins & Dayan proved convergence to $Q^*$ if every pair is visited infinitely often and
$\sum\alpha = \infty$, $\sum\alpha^2 < \infty$.

## SARSA (on-policy)
$$
Q(s, a) \leftarrow Q(s, a) + \alpha\Big[r + \gamma\,Q(s', a') - Q(s, a)\Big]
$$
The target uses the action $a'$ the agent **actually takes** next (State-Action-Reward-State-Action). It learns the value of
the exploring policy itself. In risky environments ("cliff walking"), SARSA learns a safer path because it accounts
for its own random exploratory moves; Q-learning learns the optimal-but-risky path.

## Exploration
ε-greedy with ε decaying from 1 towards a small floor: explore a lot early, exploit later. Optimistic initial values
(start Q high) also encourage exploration.

## From tables to deep RL
Replace the table with a network $Q(s, a;\theta)$ trained on the TD error $(r + \gamma\max_{a'}Q(s', a';\theta^-) - Q(s,a;\theta))^2$.
DQN adds an **experience replay** buffer (break correlations between consecutive samples) and a slowly updated
**target network** $\theta^-$ (stabilise the moving target).

## Exercises
1. In the widget, train with Q-learning, then switch to SARSA with ε = 0.3. Compare the routes near the holes.
2. Run the demo: how many episodes does Q-learning need to match value iteration's policy?
3. Set ε to 0 from the start. Does the agent still find the goal?
