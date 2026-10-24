"""Reinforcement learning: bandits, MDP planning (value/policy iteration) and model-free control (Q-learning, SARSA).

Bellman optimality:  V*(s) = max_a Σ_s' P(s'|s,a)[R(s,a,s') + γV*(s')]
Q-learning update:   Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]      (off-policy)
SARSA update:        Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]              (on-policy)
"""

from __future__ import annotations

import numpy as np

from .base import rng_from

ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
ARROWS = "↑→↓←"


class GridWorld:
    """A small grid. 'S' start, 'G' goal (+1, terminal), 'H' hole (-1, terminal), '#' wall, '.' free (step cost)."""

    def __init__(self, layout=("S..#.", ".#...", "...#H", ".#..G"), step_reward=-0.04, slip=0.0, random_state=None):
        self.grid = [list(r) for r in layout]
        self.h, self.w = len(self.grid), len(self.grid[0])
        self.step_reward, self.slip = step_reward, slip
        self.rng = rng_from(random_state)
        self.states = [(r, c) for r in range(self.h) for c in range(self.w) if self.grid[r][c] != "#"]
        self.index = {s: i for i, s in enumerate(self.states)}
        self.start = next(s for s in self.states if self.grid[s[0]][s[1]] == "S")
        self.n_states, self.n_actions = len(self.states), 4

    def is_terminal(self, s):
        return self.grid[s[0]][s[1]] in "GH"

    def _move(self, s, a):
        r, c = s[0] + ACTIONS[a][0], s[1] + ACTIONS[a][1]
        if 0 <= r < self.h and 0 <= c < self.w and self.grid[r][c] != "#":
            return (r, c)
        return s

    def reward(self, s):
        return {"G": 1.0, "H": -1.0}.get(self.grid[s[0]][s[1]], self.step_reward)

    def transitions(self, s, a):
        """[(prob, next_state, reward)]: with prob `slip` the agent moves perpendicular to its intent."""
        if self.is_terminal(s):
            return [(1.0, s, 0.0)]
        outcomes = [(1 - self.slip, a)] + ([(self.slip / 2, (a + 1) % 4), (self.slip / 2, (a + 3) % 4)] if self.slip else [])
        return [(p, self._move(s, b), self.reward(self._move(s, b))) for p, b in outcomes]

    def reset(self):
        self.s = self.start
        return self.index[self.s]

    def step(self, a):
        u, acc = self.rng.random(), 0.0
        for p, s2, r in self.transitions(self.s, a):
            acc += p
            if u <= acc:
                break
        self.s = s2
        return self.index[s2], r, self.is_terminal(s2)

    def render_policy(self, policy) -> str:
        rows = []
        for r in range(self.h):
            row = ""
            for c in range(self.w):
                ch = self.grid[r][c]
                row += ch if ch in "#GH" else ARROWS[policy[self.index[(r, c)]]]
            rows.append(row)
        return "\n".join(rows)


def value_iteration(env: GridWorld, gamma=0.99, theta=1e-8):
    V = np.zeros(env.n_states)
    sweeps = 0
    while True:
        sweeps += 1
        delta = 0.0
        for s in env.states:
            i = env.index[s]
            q = [sum(p * (r + gamma * V[env.index[s2]]) for p, s2, r in env.transitions(s, a)) for a in range(4)]
            best = 0.0 if env.is_terminal(s) else max(q)
            delta = max(delta, abs(best - V[i]))
            V[i] = best
        if delta < theta:
            break
    policy = np.array([int(np.argmax([sum(p * (r + gamma * V[env.index[s2]]) for p, s2, r in env.transitions(s, a))
                                      for a in range(4)])) for s in env.states])
    return V, policy, sweeps


def policy_iteration(env: GridWorld, gamma=0.99):
    policy = np.zeros(env.n_states, int)
    for it in range(1, 1000):
        # policy evaluation: solve the linear system V = R_π + γ P_π V exactly
        P, R = np.zeros((env.n_states, env.n_states)), np.zeros(env.n_states)
        for s in env.states:
            i = env.index[s]
            for p, s2, r in env.transitions(s, policy[i]):
                P[i, env.index[s2]] += p
                R[i] += p * r
        V = np.linalg.solve(np.eye(env.n_states) - gamma * P, R)
        # policy improvement: act greedily w.r.t. V
        new = np.array([int(np.argmax([sum(p * (r + gamma * V[env.index[s2]]) for p, s2, r in env.transitions(s, a))
                                       for a in range(4)])) for s in env.states])
        if np.array_equal(new, policy):
            return V, policy, it
        policy = new
    return V, policy, it


def epsilon_greedy(Q_row, eps, rng):
    return int(rng.integers(len(Q_row))) if rng.random() < eps else int(np.argmax(Q_row))


def q_learning(env: GridWorld, episodes=2000, alpha=0.1, gamma=0.99, eps=1.0, eps_min=0.05, eps_decay=0.995, random_state=0,
               sarsa=False, max_steps=200):
    rng = rng_from(random_state)
    Q = np.zeros((env.n_states, env.n_actions))
    returns = []
    for _ in range(episodes):
        s = env.reset()
        a = epsilon_greedy(Q[s], eps, rng)
        total = 0.0
        for _ in range(max_steps):
            s2, r, done = env.step(a)
            a2 = epsilon_greedy(Q[s2], eps, rng)
            target = r if done else r + gamma * (Q[s2, a2] if sarsa else Q[s2].max())
            Q[s, a] += alpha * (target - Q[s, a])
            total += r
            s, a = s2, a2
            if done:
                break
        returns.append(total)
        eps = max(eps_min, eps * eps_decay)
    return Q, np.argmax(Q, 1), returns


class MultiArmedBandit:
    """k arms with Gaussian rewards; compare ε-greedy, UCB1 and Thompson-style strategies."""

    def __init__(self, means, random_state=0):
        self.means, self.rng = np.asarray(means, float), rng_from(random_state)

    def pull(self, a):
        return self.rng.normal(self.means[a], 1.0)

    def run(self, strategy="ucb", steps=1000, eps=0.1, c=2.0):
        k = len(self.means)
        counts, values, rewards = np.zeros(k), np.zeros(k), []
        for t in range(1, steps + 1):
            if strategy == "greedy" or (strategy == "eps" and self.rng.random() >= eps):
                a = int(np.argmax(values))
            elif strategy == "eps":
                a = int(self.rng.integers(k))
            elif strategy == "thompson":   # Gaussian posterior with unit-variance likelihood
                a = int(np.argmax(self.rng.normal(values, 1 / np.sqrt(counts + 1))))
            else:                           # UCB1: optimism in the face of uncertainty
                a = int(np.argmax(values + c * np.sqrt(np.log(t) / np.maximum(counts, 1e-12)))) if counts.min() > 0 else int(np.argmin(counts))
            r = self.pull(a)
            counts[a] += 1
            values[a] += (r - values[a]) / counts[a]        # incremental mean
            rewards.append(r)
        return np.array(rewards), counts, values
