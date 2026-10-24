"""Step 3 - classical baselines. Every neural network has to beat these to justify its complexity."""

import time

import numpy as np

from common import load
from mlscratch.decomposition import PCA
from mlscratch.linear_model import SoftmaxRegression
from mlscratch.metrics import accuracy_score
from mlscratch.naive_bayes import GaussianNB
from mlscratch.neighbors import KNeighborsClassifier
from mlscratch.ensemble import RandomForestClassifier

xtr, ytr, xval, yval, xte, yte = load()
pca = PCA(50).fit(xtr)                          # 784 → 50 dims keeps ~83% of the variance and makes kNN 15× cheaper
Ztr, Zval, Zte = pca.transform(xtr), pca.transform(xval), pca.transform(xte)
print(f"PCA-50 keeps {pca.explained_variance_ratio_.sum():.1%} of the pixel variance\n")

rows = []


def run(name, model, Xa, ya, Xv, Xt):
    t = time.time()
    model.fit(Xa, ya)
    val, test = accuracy_score(yval, model.predict(Xv)), accuracy_score(yte, model.predict(Xt))
    rows.append((name, val, test, time.time() - t))
    print(f"{name:<42}val {val:.4f}   test {test:.4f}   ({time.time() - t:.0f}s)", flush=True)


run("Gaussian naive Bayes (PCA-50)", GaussianNB(), Ztr, ytr, Zval, Zte)
run("Softmax regression (raw pixels, 15k imgs)", SoftmaxRegression(lr=0.5, n_iter=300), xtr[:15000], ytr[:15000], xval, xte)
run("Random forest 30 trees (PCA-50, 15k imgs)",
    RandomForestClassifier(n_estimators=30, max_depth=16, random_state=0), Ztr[:15000], ytr[:15000], Zval, Zte)
for k in (1, 3):
    run(f"kNN k={k} (PCA-50, all 55k)", KNeighborsClassifier(k, weights="distance"), Ztr, ytr, Zval, Zte)

print("\nTake-aways: Gaussian naive Bayes models each class as one axis-aligned Gaussian blob - too simple even on")
print("decorrelated PCA features. A linear model reaches ~91%. kNN in a good feature space is remarkably strong but")
print("must keep all 55k images and compare against them at prediction time. The MLP and CNN in the next steps")
print("should beat kNN while being far cheaper to query (a few matrix products per image).")
