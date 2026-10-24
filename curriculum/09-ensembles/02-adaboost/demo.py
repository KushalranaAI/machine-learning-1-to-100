"""AdaBoost from scratch: round-by-round errors and learner weights, sample re-weighting, and label noise."""

import numpy as np

from mlscratch.datasets import make_moons
from mlscratch.ensemble import AdaBoostClassifier, RandomForestClassifier
from mlscratch.model_selection import train_test_split
from mlscratch.tree import DecisionTreeClassifier
from mlscratch.viz import decision_boundary, plt, savefig

X, y = make_moons(600, noise=0.25, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

# Re-implement the loop inline to show the weights evolving
n = len(ytr)
w = np.full(n, 1 / n)
F = np.zeros((n, 2))
print(f"{'round':>5}{'weighted err':>14}{'alpha':>8}{'max weight':>12}{'train err':>11}")
for m in range(1, 31):
    stump = DecisionTreeClassifier(max_depth=1).fit(Xtr, ytr, sample_weight=w)
    pred = stump.predict(Xtr)
    wrong = pred != ytr
    err = w[wrong].sum() / w.sum()
    alpha = np.log((1 - err) / err) + np.log(2 - 1)
    w *= np.exp(alpha * wrong)
    w /= w.sum()
    F[np.arange(n), pred] += alpha
    if m in (1, 2, 3, 5, 10, 20, 30):
        print(f"{m:>5}{err:>14.3f}{alpha:>8.3f}{w.max() * n:>11.1f}×{np.mean(F.argmax(1) != ytr):>11.3f}")

ada = AdaBoostClassifier(n_estimators=200).fit(Xtr, ytr)
print(f"\nAdaBoost (200 stumps): train {ada.score(Xtr, ytr):.3f}  test {ada.score(Xte, yte):.3f}")
print(f"single stump: test {DecisionTreeClassifier(max_depth=1).fit(Xtr, ytr).score(Xte, yte):.3f}")

rng = np.random.default_rng(0)
noisy = ytr.copy()
flip = rng.choice(len(noisy), int(0.15 * len(noisy)), replace=False)
noisy[flip] = 1 - noisy[flip]
ada_noisy = AdaBoostClassifier(n_estimators=200, max_depth=2).fit(Xtr, noisy)
rf_noisy = RandomForestClassifier(100, min_samples_leaf=5, random_state=0).fit(Xtr, noisy)
print(f"\nwith 15% flipped training labels → test accuracy: AdaBoost(depth-2) {ada_noisy.score(Xte, yte):.3f}   random forest {rf_noisy.score(Xte, yte):.3f}")

pyplot = plt()
fig, axes = pyplot.subplots(1, 3, figsize=(15, 4.3))
for ax, rounds in zip(axes, (1, 10, 200)):
    decision_boundary(ax, AdaBoostClassifier(n_estimators=rounds).fit(Xtr, ytr), X, y)
    ax.set_title(f"AdaBoost with {rounds} stump(s)")
savefig(fig, "adaboost_rounds")
