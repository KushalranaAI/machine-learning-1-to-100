"""Step 3 - compare 8 classifiers with stratified 5-fold CV, tune the best, evaluate once on a held-out test set."""

import numpy as np

from features import FeaturePipeline, TitanicFeatures, load
from mlscratch.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from mlscratch.linear_model import LogisticRegression
from mlscratch.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve
from mlscratch.model_selection import StratifiedKFold, train_test_split
from mlscratch.naive_bayes import GaussianNB
from mlscratch.neighbors import KNeighborsClassifier
from mlscratch.svm import SVC
from mlscratch.tree import DecisionTreeClassifier
from mlscratch.viz import plt, savefig

df = load()
y = df.Survived.to_numpy()
idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=7)
train, test = df.iloc[idx_train].reset_index(drop=True), df.iloc[idx_test].reset_index(drop=True)
y_train, y_test = y[idx_train], y[idx_test]
print(f"train {len(train)} · test {len(test)} (stratified: {y_train.mean():.1%} / {y_test.mean():.1%} survived)\n")


def cv_scores(model, k=5):
    """Stratified k-fold on the TRAIN rows. The feature statistics are refitted inside each fold, so no leakage."""
    accs = []
    for tr, va in StratifiedKFold(k, random_state=0).split(np.zeros(len(train)), y_train):
        m = FeaturePipeline(model.clone()).fit(train.iloc[tr], y_train[tr])
        accs.append(accuracy_score(y_train[va], m.predict(train.iloc[va])))
    return np.array(accs)


models = {
    "LogisticRegression": LogisticRegression(lr=0.1, n_iter=2000, C=1.0),
    "kNN (k=9)": KNeighborsClassifier(9),
    "GaussianNB": GaussianNB(),
    "SVC (RBF)": SVC(C=1.0, kernel="rbf"),
    "DecisionTree (d=4)": DecisionTreeClassifier(max_depth=4),
    "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=7, min_samples_leaf=2, random_state=0),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.5),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=0),
}
print(f"{'model':<22}{'CV accuracy':>18}")
cv = {}
for name, model in models.items():
    s = cv_scores(model)
    cv[name] = s
    print(f"{name:<22}{s.mean():>10.3f} ± {s.std():.3f}")

print("\nGrid search for the SVM (C × gamma) and the random forest (depth × leaf size):")
grids = {
    "SVC (RBF)": [SVC(C=C, kernel="rbf", gamma=g) for C in (1, 3, 10) for g in (0.01, 0.05)],
    "RandomForest": [RandomForestClassifier(n_estimators=100, max_depth=d, min_samples_leaf=l, random_state=0) for d in (4, 6, 8) for l in (1, 3)],
}
best = {}
for name, cands in grids.items():
    scored = [(cv_scores(m).mean(), m) for m in cands]
    score, model = max(scored, key=lambda t: t[0])
    keys = ("C", "gamma") if name.startswith("SVC") else ("max_depth", "min_samples_leaf")
    print(f"  {name:<14} best CV {score:.3f} with {', '.join(f'{k}={model.get_params()[k]}' for k in keys)}")
    best[name] = (score, model)

name, (score, model) = max(best.items(), key=lambda kv: kv[1][0])
print(f"\nFinal model: {name}. Fit on all {len(train)} training rows, then evaluate on the test set once:")
final = FeaturePipeline(model.clone()).fit(train, y_train)
pred = final.predict(test)
print(f"  test accuracy {accuracy_score(y_test, pred):.3f} · F1 {f1_score(y_test, pred):.3f}")
print(classification_report(y_test, pred))
print("confusion matrix [rows = true died/survived, cols = predicted]:")
print(confusion_matrix(y_test, pred))

# A probabilistic model for the ROC curve and the threshold discussion
proba_model = FeaturePipeline(GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=0)).fit(train, y_train)
p = proba_model.predict_proba(test)[:, 1]
fpr, tpr, thr = roc_curve(y_test, p)
print(f"\nGradient boosting test ROC-AUC = {roc_auc_score(y_test, p):.3f}")
for t in (0.3, 0.5, 0.7):
    print(f"  threshold {t}: accuracy {accuracy_score(y_test, (p >= t).astype(int)):.3f}, "
          f"recall {np.mean(p[y_test == 1] >= t):.2f}, precision {np.mean(y_test[p >= t]):.2f}")

# What did a simple, interpretable model learn?
lr = FeaturePipeline(LogisticRegression(lr=0.1, n_iter=2000, C=1.0)).fit(train, y_train)
coef = lr.model.coef_.ravel()
names = lr.features_.names_
order = np.argsort(coef)
print("\nLogistic regression - strongest effects on the log-odds of survival (standardized features):")
for i in list(order[:4]) + list(order[-4:]):
    print(f"  {names[i]:<18}{coef[i]:+.3f}")

if (test_csv := load.__globals__["DATA"] / "test.csv").exists():  # Kaggle competition file: write a submission
    import pandas as pd

    kaggle_test = pd.read_csv(test_csv)
    full = FeaturePipeline(model.clone()).fit(df, y)
    pd.DataFrame({"PassengerId": kaggle_test.PassengerId, "Survived": full.predict(kaggle_test)}).to_csv("submission.csv", index=False)
    print("\nwrote submission.csv - upload it at kaggle.com/competitions/titanic/submit")

P = plt()
fig, axes = P.subplots(1, 3, figsize=(17, 4.5))
axes[0].boxplot(list(cv.values()), orientation="horizontal", tick_labels=list(cv))
axes[0].set(title="5-fold CV accuracy")
axes[1].plot(fpr, tpr, label=f"AUC {roc_auc_score(y_test, p):.3f}")
axes[1].plot([0, 1], [0, 1], "k--")
axes[1].set(title="ROC curve (test, gradient boosting)", xlabel="false positive rate", ylabel="true positive rate")
axes[1].legend()
axes[2].barh(np.array(names)[order], coef[order], color=["#e45756" if c < 0 else "#54a24b" for c in coef[order]])
axes[2].set(title="Logistic regression coefficients")
savefig(fig, "03_models")
