"""Bag-of-words and TF-IDF from scratch on a handful of messages."""

import numpy as np

from mlscratch.text import CountVectorizer, TfidfVectorizer, tokenize

docs = [
    "WIN a FREE prize now!!! Call now to claim your free cash",
    "Are we still meeting for lunch today?",
    "Congratulations, you have won a free holiday. Text WIN to claim",
    "Can you send me the report before the meeting",
    "Free entry in our weekly competition, claim your prize now",
    "I'll call you after lunch, the meeting ran late",
]
labels = np.array([1, 0, 1, 0, 1, 0])  # 1 = spam

print("tokens of doc 0:", tokenize(docs[0]))
cv = CountVectorizer(stop_words="english").fit(docs)
counts = cv.transform(docs)
vocab = cv.get_feature_names_out()
print(f"\nvocabulary size {len(vocab)}; count matrix {counts.shape}, {100 * (counts == 0).mean():.0f}% zeros")
print("doc 0 counts:", {str(w): int(c) for w, c in zip(vocab, counts[0]) if c})

tfidf = TfidfVectorizer(stop_words="english").fit(docs)
T = tfidf.transform(docs)
idf = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
print("\nidf: common words get low weight →", {w: round(float(idf[w]), 2) for w in ["free", "claim", "meeting", "report"]})
top = np.argsort(-T[0])[:5]
print("top TF-IDF terms in doc 0:", [(str(tfidf.get_feature_names_out()[j]), round(float(T[0, j]), 3)) for j in top])

sim = T @ T.T                     # rows are L2-normalised → dot product = cosine similarity
np.fill_diagonal(sim, 0)
i, j = np.unravel_index(np.argmax(sim), sim.shape)
print(f"\nmost similar pair (cosine {sim[i, j]:.2f}):\n  {docs[i]}\n  {docs[j]}")

spam_words = counts[labels == 1].sum(0)
ham_words = counts[labels == 0].sum(0)
print("\nwords used only in spam:", [str(w) for w, s, h in zip(vocab, spam_words, ham_words) if s >= 2 and h == 0])

bigrams = CountVectorizer(ngram_range=(1, 2), stop_words="english").fit(docs)
print("with bigrams the vocabulary grows to", len(bigrams.vocabulary_), "e.g.", [str(t) for t in bigrams.get_feature_names_out() if " " in t][:5])
