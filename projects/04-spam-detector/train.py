"""Step 3 - bag-of-words models from scratch, an operating threshold chosen for high precision, error analysis.

Losing a real message is worse than letting a spam through, so the threshold is tuned on out-of-fold predictions
to reach ≥ 98% precision. The test set is used once at the end. The model is saved to data/spam_model.json for predict.py.
"""

import json
from pathlib import Path

import numpy as np

from common import META_NAMES, load, meta_features
from mlscratch.linear_model import LogisticRegression
from mlscratch.metrics import (confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score,
                               roc_auc_score)
from mlscratch.model_selection import StratifiedKFold, train_test_split
from mlscratch.naive_bayes import BernoulliNB, MultinomialNB
from mlscratch.preprocessing import StandardScaler
from mlscratch.svm import LinearSVC
from mlscratch.text import CountVectorizer, TfidfVectorizer
from mlscratch.viz import plt, savefig

df = load()
texts, y = df.text.to_numpy(), df.y.to_numpy()
tr, te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=0)
print(f"train {len(tr):,} · test {len(te):,} · spam rate {y[tr].mean():.1%} / {y[te].mean():.1%}\n")


class TextModel:
    """vectorizer (+ optional scaled meta features) → classifier, refitted from raw text so CV never leaks."""

    def __init__(self, kind, model, meta=False):
        self.kind, self.model, self.meta = kind, model, meta

    def clone(self):
        return TextModel(self.kind, self.model.clone(), self.meta)

    def _vec(self):
        if self.kind == "counts":
            return CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=3000)
        if self.kind == "binary":
            return CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=3000, binary=True)
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=3000, sublinear_tf=True)

    def _X(self, docs, fit=False):
        X = self.vec_.fit_transform(docs) if fit else self.vec_.transform(docs)
        if self.meta:
            M = meta_features(docs)
            M = self.scaler_.fit_transform(M) if fit else self.scaler_.transform(M)
            X = np.c_[X, 0.3 * M]           # down-weighted so a few dense columns don't swamp 3,000 sparse ones
        return X

    def fit(self, docs, y):
        self.vec_, self.scaler_ = self._vec(), StandardScaler()
        self.model.fit(self._X(docs, fit=True), y)
        return self

    def score(self, docs):
        X = self._X(docs)
        return self.model.predict_proba(X)[:, 1] if hasattr(self.model, "predict_proba") else self.model.decision_function(X)


models = {
    "MultinomialNB (counts)": TextModel("counts", MultinomialNB(alpha=0.1)),
    "BernoulliNB (binary)": TextModel("binary", BernoulliNB(alpha=0.1)),
    "LogisticRegression (tf-idf)": TextModel("tfidf", LogisticRegression(lr=2.0, n_iter=800, C=50.0)),
    "LogReg (tf-idf + meta)": TextModel("tfidf", LogisticRegression(lr=2.0, n_iter=800, C=50.0), meta=True),
    "LinearSVC (tf-idf + meta)": TextModel("tfidf", LinearSVC(C=10.0, lr=0.5, n_iter=800), meta=True),
}


def out_of_fold(model, k=5):
    """Scores for every training message from a model that never saw it."""
    oof = np.zeros(len(tr))
    for a, b in StratifiedKFold(k, random_state=0).split(np.zeros(len(tr)), y[tr]):
        oof[b] = model.clone().fit(texts[tr][a], y[tr][a]).score(texts[tr][b])
    return oof


def threshold_for_precision(scores, labels, target=0.98):
    """Lowest threshold whose precision ≥ target - the most recall we can buy at that precision."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    ok = np.flatnonzero(precision[: len(thresholds)] >= target)
    return float(thresholds[ok].min()) if len(ok) else float(np.max(scores))


print(f"{'model':<30}{'OOF ROC-AUC':>12}{'F1@default':>12}{'thr→P≥98%':>11}{'recall there':>14}")
oof_scores, thresholds = {}, {}
for name, model in models.items():
    s = out_of_fold(model)
    default = 0.0 if isinstance(model.model, LinearSVC) else 0.5
    t = threshold_for_precision(s, y[tr])
    oof_scores[name], thresholds[name] = s, t
    print(f"{name:<30}{roc_auc_score(y[tr], s):>12.4f}{f1_score(y[tr], (s >= default).astype(int)):>12.3f}"
          f"{t:>11.3f}{recall_score(y[tr], (s >= t).astype(int)):>14.3f}")

best = max(models, key=lambda n: recall_score(y[tr], (oof_scores[n] >= thresholds[n]).astype(int)))
print(f"\nChosen: {best} (most recall at ≥ 98% precision). Refit on all training data, evaluate on test once:")
final = models[best].clone().fit(texts[tr], y[tr])
s_test = final.score(texts[te])
pred = (s_test >= thresholds[best]).astype(int)
cm = confusion_matrix(y[te], pred)
print(f"  precision {precision_score(y[te], pred):.3f} · recall {recall_score(y[te], pred):.3f} · F1 {f1_score(y[te], pred):.3f}"
      f" · ROC-AUC {roc_auc_score(y[te], s_test):.4f}")
print(f"  confusion matrix: {cm[0, 0]} ham kept, {cm[0, 1]} ham lost (false positives), "
      f"{cm[1, 0]} spam missed, {cm[1, 1]} spam caught")
print(f"  (the threshold targeted 98% precision on out-of-fold training scores; the test set has only {y[te].sum()} spam, so\n"
      f"   each false positive moves test precision by ~{100 / max(pred.sum(), 1):.1f} points - small test sets are noisy)")

print("\nError analysis - read the mistakes, they tell you what feature to add next:")
for label, mask in [("false positives (ham flagged as spam)", (pred == 1) & (y[te] == 0)),
                    ("false negatives (spam that got through)", (pred == 0) & (y[te] == 1))]:
    print(f"  {label}:")
    for i in np.flatnonzero(mask)[:5]:
        print(f"    [{s_test[i]:.2f}] {texts[te][i][:100]}")

# Interpretability: for multinomial NB the evidence of word w is log θ_spam,w - log θ_ham,w.
nb = TextModel("counts", MultinomialNB(alpha=0.1)).fit(texts[tr], y[tr])
words = nb.vec_.get_feature_names_out()
evidence = nb.model.feature_log_prob_[1] - nb.model.feature_log_prob_[0]
print("\nNaive Bayes: strongest spam n-grams:", ", ".join(words[np.argsort(-evidence)[:12]]))

model_path = Path(__file__).resolve().parent / "data" / "spam_model.json"
json.dump({"vocabulary": nb.vec_.vocabulary_, "feature_log_prob": nb.model.feature_log_prob_.tolist(),
           "class_log_prior": nb.model.class_log_prior_.tolist(), "ngram_range": [1, 2],
           "threshold": thresholds["MultinomialNB (counts)"]}, model_path.open("w"))
print(f"\nsaved the naive Bayes model ({len(words):,} n-grams) to data/{model_path.name} - try `python predict.py`")

P = plt()
fig, axes = P.subplots(1, 2, figsize=(13, 4.6))
for name, s in oof_scores.items():
    p, r, _ = precision_recall_curve(y[tr], s)
    axes[0].plot(r, p, label=name)
axes[0].axhline(0.98, color="k", ls=":", lw=1)
axes[0].set(title="Precision-recall (out-of-fold, training set)", xlabel="recall", ylabel="precision", ylim=(0.5, 1.01))
axes[0].legend(fontsize=8)
order = np.argsort(-evidence)[:15][::-1]
axes[1].barh(words[order], evidence[order], color="#e45756")
axes[1].set(title="Naive Bayes evidence log θ_spam/θ_ham")
savefig(fig, "02_models")
