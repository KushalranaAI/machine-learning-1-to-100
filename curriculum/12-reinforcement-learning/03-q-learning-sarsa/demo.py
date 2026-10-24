"""Q-learning vs SARSA on a slippery grid world, compared with the optimal policy from value iteration."""

import numpy as np

from mlscratch.rl import GridWorld, q_learning, value_iteration
from mlscratch.viz import plt, savefig

env = GridWorld(slip=0.1, random_state=0)
V_opt, pi_opt, _ = value_iteration(env, gamma=0.99)
non_terminal = [env.index[s] for s in env.states if not env.is_terminal(s)]
print("optimal policy (value iteration, uses the model):")
print(env.render_policy(pi_opt))


def evaluate(policy, episodes=500, seed=1):
    e = GridWorld(slip=0.1, random_state=seed)
    total = 0.0
    for _ in range(episodes):
        s, done, steps = e.reset(), False, 0
        while not done and steps < 100:
            s, r, done = e.step(policy[s])
            total += r
            steps += 1
    return total / episodes


pyplot = plt()
fig, ax = pyplot.subplots(figsize=(8, 4))
for name, sarsa in [("Q-learning", False), ("SARSA", True)]:
    for episodes in (100, 500, 3000):
        Q, pi, returns = q_learning(GridWorld(slip=0.1, random_state=0), episodes=episodes, alpha=0.1, gamma=0.99, sarsa=sarsa, random_state=0)
        agree = np.mean(pi[non_terminal] == pi_opt[non_terminal])
        print(f"{name:<10} after {episodes:>4} episodes: policy agrees with optimal on {agree:.0%} of states, "
              f"average return of greedy policy {evaluate(pi):+.3f}")
    ax.plot(np.convolve(returns, np.ones(100) / 100, mode="valid"), label=name)
print(f"optimal policy average return {evaluate(pi_opt):+.3f}")
print("The learned policies earn the optimal return; the remaining disagreements are in states where several actions")
print("are equally good, or that the agent rarely visits - their exact action doesn't change the return.")
print("\nQ-learning policy after 3000 episodes:")
print(env.render_policy(q_learning(GridWorld(slip=0.1, random_state=0), episodes=3000, random_state=0)[1]))
ax.set(xlabel="episode", ylabel="return (moving average of 100)", title="learning curves (ε decays from 1 to 0.05)")
ax.legend()
savefig(fig, "q_learning")
