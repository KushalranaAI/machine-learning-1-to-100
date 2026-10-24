"""Turning text into numbers: tokenisation, bag-of-words counts and TF-IDF."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from .base import BaseEstimator

ENGLISH_STOP_WORDS = frozenset("""a about above after again against all am an and any are as at be because been before being
below between both but by can did do does doing down during each few for from further had has have having he her here hers
herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or
other our ours ourselves out over own same she should so some such than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where which while who whom why will with you your
yours yourself yourselves""".split())


def tokenize(text: str, lowercase=True) -> list[str]:
    text = text.lower() if lowercase else text
    return re.findall(r"(?u)\b\w\w+\b", text)


class CountVectorizer(BaseEstimator):
    """Bag of words: column j counts how often vocabulary term j occurs in the document. Supports n-grams."""

    def __init__(self, ngram_range=(1, 1), min_df=1, max_features=None, stop_words=None, binary=False):
        self.ngram_range, self.min_df, self.max_features, self.stop_words, self.binary = ngram_range, min_df, max_features, stop_words, binary

    def _terms(self, doc):
        toks = tokenize(doc)
        if self.stop_words == "english":
            toks = [t for t in toks if t not in ENGLISH_STOP_WORDS]
        lo, hi = self.ngram_range
        return [" ".join(toks[i : i + n]) for n in range(lo, hi + 1) for i in range(len(toks) - n + 1)]

    def fit(self, docs, y=None):
        df = Counter()
        tf = Counter()
        for d in docs:
            terms = self._terms(d)
            df.update(set(terms))
            tf.update(terms)
        vocab = [t for t in df if df[t] >= self.min_df]
        if self.max_features:
            vocab = sorted(vocab, key=lambda t: (-tf[t], t))[: self.max_features]
        self.vocabulary_ = {t: i for i, t in enumerate(sorted(vocab))}
        return self

    def transform(self, docs):
        X = np.zeros((len(docs), len(self.vocabulary_)))
        for i, d in enumerate(docs):
            for term, c in Counter(self._terms(d)).items():
                j = self.vocabulary_.get(term)
                if j is not None:
                    X[i, j] = 1 if self.binary else c
        return X

    def fit_transform(self, docs, y=None):
        return self.fit(docs).transform(docs)

    def get_feature_names_out(self):
        return np.array(sorted(self.vocabulary_, key=self.vocabulary_.get), dtype=object)


class TfidfVectorizer(CountVectorizer):
    """tf-idf(t, d) = tf(t, d) · idf(t),  idf(t) = ln((1 + n) / (1 + df(t))) + 1, then L2-normalise each row.

    Words frequent in a document but rare in the corpus get high weight; 'the' gets almost none.
    """

    def __init__(self, ngram_range=(1, 1), min_df=1, max_features=None, stop_words=None, sublinear_tf=False):
        super().__init__(ngram_range, min_df, max_features, stop_words)
        self.sublinear_tf = sublinear_tf

    def get_params(self):
        return {k: getattr(self, k) for k in ("ngram_range", "min_df", "max_features", "stop_words", "sublinear_tf")}

    def fit(self, docs, y=None):
        super().fit(docs)
        counts = super().transform(docs)
        df = (counts > 0).sum(0)
        self.idf_ = np.log((1 + len(docs)) / (1 + df)) + 1
        return self

    def transform(self, docs):
        tf = super().transform(docs)
        if self.sublinear_tf:
            tf = np.where(tf > 0, 1 + np.log(np.maximum(tf, 1e-12)), 0)
        X = tf * self.idf_
        return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    def fit_transform(self, docs, y=None):
        return self.fit(docs).transform(docs)
