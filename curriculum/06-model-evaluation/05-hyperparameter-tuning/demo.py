"""Grid search vs random search on an RBF SVM, tuning a Pipeline, and nested CV."""

import time

import numpy as np

from mlscratch.datasets import make_moons
from mlscratch.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_val_score
from mlscratch.neighbors import KNeighborsClassifier
from mlscratch.preprocessing import Pipeline, StandardScaler
from mlscratch.svm import SVC


class LogUniform:
    """Sample 10^U(a, b) - the right prior for scale parameters like C and γ."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def rvs(self, random_state):
        return float(10 ** random_state.uniform(self.a, self.b))


X, y = make_moons(300, noise=0.25, random_state=0)

t = time.perf_counter()
grid = GridSearchCV(SVC(), {"C": [0.1, 1, 10], "gamma": [0.1, 1, 10]}, cv=5).fit(X, y)
print(f"grid search  : {len(grid.results_)} configs × 5 folds in {time.perf_counter() - t:.1f}s → best {grid.best_params_}  CV acc {grid.best_score_:.3f}")
t = time.perf_counter()
rand = RandomizedSearchCV(SVC(), {"C": LogUniform(-2, 2), "gamma": LogUniform(-2, 1.5)}, n_iter=9, cv=5, random_state=0).fit(X, y)
print(f"random search: {len(rand.results_)} configs × 5 folds in {time.perf_counter() - t:.1f}s → best "
      f"{ {k: round(v, 3) for k, v in rand.best_params_.items()} }  CV acc {rand.best_score_:.3f}")

pipe = Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier())])
gs = GridSearchCV(pipe, {"knn__n_neighbors": [1, 3, 5, 9, 15, 25, 41]}, cv=5).fit(X, y)
print("\nk-NN pipeline: CV accuracy by k →", {r["params"]["knn__n_neighbors"]: round(float(r["mean"]), 3) for r in gs.results_})

# nested CV: honest estimate of "tune k by CV, then train"
outer = []
for tr, te in KFold(5, random_state=1).split(X):
    inner = GridSearchCV(pipe, {"knn__n_neighbors": [1, 3, 5, 9, 15, 25, 41]}, cv=5).fit(X[tr], y[tr])
    outer.append(np.mean(inner.predict(X[te]) == y[te]))
print(f"best inner-CV score {gs.best_score_:.3f}  vs  nested-CV estimate {np.mean(outer):.3f} ± {np.std(outer):.3f}")
_ = cross_val_score
