"""Naive Bayes from scratch: a tiny spam filter with word counts, Laplace smoothing, and Gaussian NB on iris."""

import numpy as np

from mlscratch.datasets import load_iris
from mlscratch.model_selection import cross_val_score
from mlscratch.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from mlscratch.text import CountVectorizer

train = [
    ("win a free prize now", 1), ("free cash claim now", 1), ("you won a free holiday", 1), ("claim your prize today", 1),
    ("urgent call now to win cash", 1), ("are we meeting today", 0), ("lunch at noon?", 0), ("send me the report", 0),
    ("call me when you are free", 0), ("the meeting is moved to monday", 0), ("can you review my report today", 0),
]
docs, y = [d for d, _ in train], np.array([lab for _, lab in train])
cv = CountVectorizer().fit(docs)
X = cv.transform(docs)
vocab = [str(w) for w in cv.get_feature_names_out()]

nb = MultinomialNB(alpha=1.0).fit(X, y)
print("class priors P(ham), P(spam):", np.round(np.exp(nb.class_log_prior_), 3))
for w in ("free", "prize", "meeting", "report"):
    j = vocab.index(w)
    ps, ph = np.exp(nb.feature_log_prob_[1, j]), np.exp(nb.feature_log_prob_[0, j])
    print(f"  P('{w}'|spam) = {ps:.3f}   P('{w}'|ham) = {ph:.3f}   likelihood ratio {ps / ph:5.2f}")

tests = ["free prize meeting", "are you free for lunch", "claim cash now", "monday report review"]
P = nb.predict_proba(cv.transform(tests))
for t, p in zip(tests, P[:, 1]):
    print(f"P(spam | '{t}') = {p:.3f}")

# By hand for one message: log P(spam) + Σ count·log θ
x = cv.transform(["free prize"])[0]
scores = nb.class_log_prior_ + nb.feature_log_prob_ @ x
print("by hand: log-scores", np.round(scores, 3), "→ P(spam) =", round(float(np.exp(scores[1]) / np.exp(scores).sum()), 3))

# Without smoothing, a single unseen word vetoes everything
nb0 = MultinomialNB(alpha=1e-10).fit(X, y)
msg = "free free free free call meeting"   # spam-leaning words + 'meeting', which never appears in training spam
print(f"\nα≈0: P(spam | '{msg}') = {nb0.predict_proba(cv.transform([msg]))[0, 1]:.3f}  (P('meeting'|spam) = 0 vetoes all other evidence)")
print(f"α=1: P(spam | '{msg}') = {nb.predict_proba(cv.transform([msg]))[0, 1]:.3f}")
print(f"Bernoulli NB: P(spam | 'free prize') = {BernoulliNB().fit(X, y).predict_proba(cv.transform(['free prize']))[0, 1]:.3f}")

Xi, yi, _ = load_iris()
print(f"\nGaussian NB on iris, 5-fold CV accuracy: {cross_val_score(GaussianNB(), Xi, yi, cv=5).mean():.3f}")
g = GaussianNB().fit(Xi, yi)
print("per-class petal-length means:", np.round(g.theta_[:, 2], 2), " variances:", np.round(g.var_[:, 2], 3))
