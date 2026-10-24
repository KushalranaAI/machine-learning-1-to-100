---
title: Text Features: Bag of Words & TF-IDF
order: 5
summary: Turn documents into vectors - tokenisation, stop words, n-grams, count vectors and TF-IDF - built from scratch and used for spam detection.
level: intermediate
minutes: 40
why: Text is everywhere - emails, reviews, tickets, descriptions. Classical vectorisers plus a linear model or naive Bayes are fast, interpretable and still hard to beat on many text-classification tasks.
background: The vector-space model (Salton, 1975) and TF-IDF (Spärck Jones, 1972) powered search engines for decades and remain strong baselines next to modern embeddings.
use_cases:
  - Spam and phishing detection
  - Sentiment analysis of reviews
  - Routing support tickets
  - Document search and similarity
where_used:
  - 04-spam-detector
code:
  - mlscratch/text.py:tokenize
  - mlscratch/text.py:CountVectorizer
  - mlscratch/text.py:TfidfVectorizer
demo: demo.py
cons:
  - Ignores word order (beyond n-grams) and meaning (synonyms)
  - Huge sparse matrices for large vocabularies
workflow: |
  flowchart LR
    A[Raw text] --> B[lowercase + tokenise]
    B --> C[remove stop words, n-grams]
    C --> D[vocabulary from TRAIN docs]
    D --> E[count matrix]
    E --> F[× idf, L2-normalise]
    F --> G[linear model / naive Bayes]
---

## Bag of words
1. **Tokenise**: `"Win a FREE prize!!"` → `["win", "free", "prize"]` (lowercase, split on non-word characters).
2. **Vocabulary**: every distinct token in the training documents gets a column.
3. **Count matrix**: $X_{d,t}$ = number of times term $t$ appears in document $d$. Most entries are 0 (sparse).

**n-grams** add word pairs ("not good", "free prize") to capture a little word order.

## TF-IDF
Frequent-everywhere words ("the", "you") carry little information. TF-IDF down-weights them:
$$
\text{tfidf}(t, d) = \text{tf}(t,d)\cdot\text{idf}(t), \qquad \text{idf}(t) = \ln\frac{1+n}{1+\text{df}(t)} + 1
$$
where $\text{df}(t)$ is the number of documents containing $t$. Each row is then L2-normalised so long documents
don't dominate. Cosine similarity between TF-IDF vectors is a classic document-similarity measure.

## Which model?
- **Multinomial naive Bayes** on counts - very fast, great with little data (Part 8).
- **Logistic regression / linear SVM** on TF-IDF - usually the strongest classical baseline.

## Exercises
1. Run the demo; print the top-weighted terms of a spam message.
2. Add bigrams; which new features appear in the top 10?
3. Compute cosine similarity between all pairs of demo documents - which are most similar?
