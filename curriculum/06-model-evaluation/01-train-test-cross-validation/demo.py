"""Hold-out vs k-fold variance, stratification, and the classic 'select features before CV' leak."""

import numpy as np

from mlscratch.datasets import load_iris, make_classification
from mlscratch.feature_selection import SelectKBest, f_classif
from mlscratch.linear_model import LogisticRegression
from mlscratch.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from mlscratch.neighbors import KNeighborsClassifier
from mlscratch.preprocessing import Pipeline, StandardScaler

X, y, _ = load_iris()

holdout, kfold = [], []
for seed in range(30):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    holdout.append(KNeighborsClassifier(5).fit(Xtr, ytr).score(Xte, yte))
    kfold.append(cross_val_score(KNeighborsClassifier(5), X, y, cv=KFold(5, random_state=seed)).mean())
print(f"30 random hold-out splits: accuracy {np.mean(holdout):.3f} ± {np.std(holdout):.3f}  (range {min(holdout):.2f}-{max(holdout):.2f})")
print(f"30 repetitions of 5-fold : accuracy {np.mean(kfold):.3f} ± {np.std(kfold):.3f}  → much more stable")

# stratification matters for rare classes
y_rare = (np.random.default_rng(0).random(200) < 0.05).astype(int)
plain = [int(y_rare[val].sum()) for _, val in KFold(5, random_state=1).split(np.zeros(200))]
strat = [int(y_rare[val].sum()) for _, val in StratifiedKFold(5, random_state=1).split(np.zeros(200), y_rare)]
print(f"\npositives per validation fold (rare class, {y_rare.sum()} total): KFold {plain}  StratifiedKFold {strat}")

# leakage: selecting features on ALL data, then cross-validating, on pure noise
rng = np.random.default_rng(0)
Xn, yn = rng.normal(size=(100, 2000)), rng.integers(0, 2, 100)
leaky_X = SelectKBest(f_classif, k=20).fit_transform(Xn, yn)
leaky = cross_val_score(LogisticRegression(n_iter=300), leaky_X, yn, cv=5).mean()
honest = cross_val_score(Pipeline([("select", SelectKBest(f_classif, k=20)), ("model", LogisticRegression(n_iter=300))]), Xn, yn, cv=5).mean()
print(f"\npure noise, 2000 features: select-then-CV accuracy {leaky:.3f} (LEAK!)  vs  select-inside-CV {honest:.3f} (honest ≈ 0.5)")

Xc, yc = make_classification(500, 5, random_state=0)
pipe = Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier(7))])
scores = cross_val_score(pipe, Xc, yc, cv=StratifiedKFold(5, random_state=0))
print(f"\nPipeline(scaler → kNN) 5-fold scores: {np.round(scores, 3)} → {scores.mean():.3f} ± {scores.std():.3f}")
