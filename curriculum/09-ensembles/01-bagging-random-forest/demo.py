"""Bagging and random forests from scratch: OOB fraction, accuracy vs number of trees, decorrelation."""

import numpy as np

from mlscratch.datasets import make_classification
from mlscratch.ensemble import BaggingClassifier, RandomForestClassifier
from mlscratch.model_selection import train_test_split
from mlscratch.tree import DecisionTreeClassifier
from mlscratch.viz import plt, savefig

rng = np.random.default_rng(0)
n = 1000
draws = np.array([len(np.unique(rng.integers(0, n, n))) for _ in range(200)])
print(f"bootstrap samples contain on average {draws.mean() / n:.1%} unique rows → out-of-bag {1 - draws.mean() / n:.1%} (theory 36.8%)")

X, y = make_classification(1200, 20, n_informative=6, class_sep=0.7, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
single = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
print(f"\nsingle deep tree: train {single.score(Xtr, ytr):.3f}  test {single.score(Xte, yte):.3f}")

sizes = [1, 5, 10, 25, 50, 100, 200]
results = {}
for name, make in [("bagging", lambda b: BaggingClassifier(n_estimators=b, random_state=0)),
                   ("random forest", lambda b: RandomForestClassifier(n_estimators=b, max_features="sqrt", random_state=0))]:
    accs = []
    for b in sizes:
        m = make(b).fit(Xtr, ytr)
        accs.append(m.score(Xte, yte))
    results[name] = accs
    print(f"{name:<14} test accuracy by #trees {dict(zip(sizes, np.round(accs, 3).tolist()))}   OOB score (200 trees) {m.oob_score_:.3f}")

# correlation between individual trees' predictions
def mean_corr(model):
    P = np.array([e.predict_proba(Xte)[:, 1] for e in model.estimators_[:30]])
    C = np.corrcoef(P)
    return C[np.triu_indices_from(C, 1)].mean()

bag = BaggingClassifier(n_estimators=30, random_state=1).fit(Xtr, ytr)
rf = RandomForestClassifier(n_estimators=30, max_features="sqrt", random_state=1).fit(Xtr, ytr)
print(f"\naverage correlation between trees' predictions: bagging {mean_corr(bag):.3f}  random forest {mean_corr(rf):.3f}")
print("top-5 features by RF importance:", np.argsort(-rf.feature_importances_)[:5].tolist(), "(features 0-5 are informative)")

pyplot = plt()
fig, ax = pyplot.subplots(figsize=(7, 4))
for name, accs in results.items():
    ax.plot(sizes, accs, "o-", label=name)
ax.axhline(single.score(Xte, yte), ls="--", color="grey", label="single tree")
ax.set(xscale="log", xlabel="number of trees", ylabel="test accuracy", title="more trees → lower variance")
ax.legend()
savefig(fig, "forest_size")
