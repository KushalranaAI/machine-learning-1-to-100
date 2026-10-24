"""Bandits: greedy vs ε-greedy vs UCB1 vs Thompson sampling, with exact regret averaged over many runs."""

import numpy as np

from mlscratch.rl import MultiArmedBandit
from mlscratch.viz import plt, savefig

means = np.array([0.2, 0.5, 0.45, 0.8, 0.3, 0.75, 0.1, 0.6, 0.35, 0.55])   # rewards ~ N(mean, 1)
steps, runs = 5000, 30
strategies = [("greedy", "greedy", {}), ("ε-greedy (ε=0.1)", "eps", {"eps": 0.1}), ("UCB1 (c=1)", "ucb", {"c": 1.0}),
              ("UCB1 (c=2)", "ucb", {"c": 2.0}), ("Thompson sampling", "thompson", {})]

pyplot = plt()
fig, ax = pyplot.subplots(figsize=(8, 4.5))
print(f"{'strategy':<20}{'% pulls of best arm':>21}{'regret @1000':>14}{'regret @5000':>14}")
for label, name, kw in strategies:
    curves, best = [], []
    for seed in range(runs):
        bandit = MultiArmedBandit(means, random_state=seed)
        bandit.run(name, steps, **kw)
        chosen = np.array(bandit.actions_)
        curves.append(np.cumsum(means.max() - means[chosen]))       # exact (expected) regret of the choices made
        best.append(np.mean(chosen == means.argmax()))
    R = np.mean(curves, axis=0)
    ax.plot(R, label=label)
    print(f"{label:<20}{100 * np.mean(best):>20.1f}%{R[999]:>14.1f}{R[-1]:>14.1f}")
ax.set(xlabel="step", ylabel="cumulative regret", title=f"regret, average of {runs} runs (lower is better)")
ax.legend()
savefig(fig, "bandits")
print("\nε-greedy's regret keeps growing linearly (it explores forever); UCB and Thompson flatten out (≈ logarithmic).")
print("UCB's c matters: with noisy rewards c=2 over-explores - tune it like any hyper-parameter.")
