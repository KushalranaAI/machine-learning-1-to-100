"""Value iteration and policy iteration on a grid world, with and without slippery moves."""

import numpy as np

from mlscratch.rl import GridWorld, policy_iteration, value_iteration

for slip in (0.0, 0.3):
    env = GridWorld(slip=slip)
    V, pi, sweeps = value_iteration(env, gamma=0.95)
    V2, pi2, iters = policy_iteration(env, gamma=0.95)
    print(f"=== slip = {slip} ===")
    print(f"value iteration: {sweeps} sweeps   policy iteration: {iters} policy improvements   same values: {np.allclose(V, V2, atol=1e-6)}")
    print(env.render_policy(pi))
    print(f"V(start) = {V[env.index[env.start]]:.3f}\n")

# contraction: the max change per sweep shrinks roughly by γ each time
env = GridWorld(slip=0.3)
V = np.zeros(env.n_states)
gamma = 0.95
deltas = []
for _ in range(40):
    new = np.array([0.0 if env.is_terminal(s) else max(sum(p * (r + gamma * V[env.index[s2]]) for p, s2, r in env.transitions(s, a)) for a in range(4))
                    for s in env.states])
    deltas.append(np.max(np.abs(new - V)))
    V = new
ratios = np.array(deltas[11:40]) / np.array(deltas[10:39])
print(f"‖V_k+1 − V_k‖∞ shrinks by a factor ≈ {np.median(ratios):.3f} per sweep, never more than γ = {gamma} (max ratio {ratios.max():.3f})")
print("The Bellman operator is a γ-contraction: γ is an upper bound on the shrink factor. Absorbing goal/hole states")
print("end returns early, so convergence here is faster than the worst case.")
