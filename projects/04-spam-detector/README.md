---
title: "SMS Spam Detector"
level: Intermediate+
summary: "NLP from scratch: tokenise 5,000 real SMS messages, build bag-of-words and tf-idf features, train naive Bayes, logistic regression and a linear SVM, choose a high-precision threshold, then analyse the errors."
problem: |
  Flag spam SMS messages automatically. Classes are imbalanced (≈ 13% spam), and the two kinds of error don't cost the
  same: **a lost real message is worse than a spam that gets through**. So we optimise *recall at a fixed high precision*
  rather than accuracy. "Always ham" already scores 87%.
dataset:
  name: "SMS Spam Collection v.1"
  source: "UCI Machine Learning Repository · Almeida & Hidalgo (2011)"
  url: "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
  kaggle: "uciml/sms-spam-collection-dataset"
  license: "CC BY 4.0"
  size: "5,574 messages (5,171 unique) · 0.5 MB"
algorithms: [Multinomial Naive Bayes, Bernoulli Naive Bayes, Logistic Regression, Linear SVM, TF-IDF]
skills:
  - Text cleaning, tokenisation and n-grams
  - Bag-of-words vs binary vs tf-idf representations
  - Hand-made meta features (digits, currency, phone numbers, URLs)
  - De-duplication to avoid train/test leakage
  - Precision-recall trade-offs and choosing a threshold from out-of-fold scores
  - Error analysis and model interpretation (naive Bayes log-odds)
  - Shipping a model as a small JSON file plus a predict script
lessons:
  - 04-feature-engineering/05-text-features
  - 06-model-evaluation/03-classification-metrics
  - 06-model-evaluation/06-imbalanced-data
  - 08-classification/01-logistic-regression
  - 08-classification/04-naive-bayes
  - 08-classification/05-support-vector-machines
steps:
  - title: "1 · Download"
    script: download_data.py
    description: |
      Kaggle's CSV (latin-1, columns v1/v2), the UCI zip or a GitHub mirror. `common.load()` detects the format and
      encoding, and **drops 403 duplicate messages**. Duplicates would otherwise land on both sides of the split and inflate the scores.
  - title: "2 · Explore the text"
    script: explore.py
    description: |
      74% of spam contains a 5+ digit number (vs 0.1% of ham), and 36% mentions money. The most spam-indicative words
      (*claim, prize, 150p, guaranteed, www*) come from the same smoothed log-odds ratio that naive Bayes learns. On
      the ham side, *lt / gt* are HTML-escaped `&lt;#&gt;` placeholders: real-world data is never clean.
  - title: "3 · Train, pick a threshold, analyse errors"
    script: train.py
    description: |
      Five models with 5-fold **out-of-fold** scores on the training set. Each model's threshold is set to reach
      **≥ 98% precision**, and we pick the model with the most recall there. Then one evaluation on the test set, and
      a printout of every kind of mistake. Takes about 1 minute.
  - title: "4 · Try your own messages"
    script: predict.py
    description: |
      Loads `data/spam_model.json` (vocabulary + naive Bayes log-probabilities) and scores messages with one dot
      product: `log P(c) + Σ count_w · log θ_c,w`. Pass your own text: `python predict.py "your message"`.
workflow: |
  flowchart LR
    A[Raw SMS] --> B[Decode + de-duplicate]
    B --> C[Stratified 80/20]
    C --> D[Tokenise · 1-2 grams · min_df 2]
    D --> E[counts / binary / tf-idf<br/>+ meta features]
    E --> F[NB · LogReg · LinearSVC<br/>out-of-fold scores]
    F --> G[Threshold for P ≥ 98%]
    G --> H[Test once + error analysis]
    H --> I[spam_model.json → predict.py]
---

# SMS Spam Detector: classic NLP, no libraries

## Results (out-of-fold on 4,136 training messages)

| Model | ROC-AUC | F1 at default threshold | Recall at ≥ 98% precision |
|---|---|---|---|
| Multinomial NB (counts) | 0.978 | 0.940 | 0.914 |
| Bernoulli NB (binary) | 0.988 | 0.941 | 0.939 |
| Logistic regression (tf-idf) | 0.989 | 0.822 | 0.799 |
| Logistic regression (tf-idf + meta) | 0.990 | 0.913 | 0.854 |
| **Linear SVM (tf-idf + meta)** | **0.995** | **0.954** | **0.944** |

**Test set (1,035 messages, once):** precision 0.969 · recall 0.947 · F1 0.958 · ROC-AUC 0.993. That's 4 real
messages lost and 7 spams missed.

## What to notice
- **Naive Bayes is a superb baseline for text.** It trains in milliseconds and is within a few points of the best
  model. The "naive" independence assumption is wrong, but the *ranking* it produces is still good.
- **Meta features help the discriminative models** (logistic regression recall 0.80 → 0.85). Spam is long, shouty
  and full of digits, which bag-of-words only partly captures.
- **The default threshold is arbitrary.** Logistic regression's F1 at 0.5 looks poor (0.82), yet its ROC-AUC is 0.99: it ranks well but is
  under-confident because of regularisation. Always choose the threshold for your cost structure.
- **Read the errors.** The false positives are personal messages that contain phone numbers. The missed spam includes
  chain-letter jokes the annotators labelled spam. The next features to add follow directly from those mistakes.

## Extend it
1. Add character n-grams (3-5 chars). They catch obfuscations like "fr33" and "w1n".
2. Calibrate the SVM's scores into probabilities (Platt scaling = a 1-D logistic regression on the scores).
3. Compare with sentence embeddings + logistic regression (see the companion RAG course).
