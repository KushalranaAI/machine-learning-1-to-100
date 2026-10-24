---
title: Naive Bayes
order: 4
summary: Bayes' rule plus conditional independence - Gaussian, multinomial and Bernoulli variants, Laplace smoothing, log-space computation and why "naive" still works.
level: beginner
minutes: 45
why: Naive Bayes trains in one pass over the data, needs very little of it, scales to huge vocabularies and is a remarkably strong text classifier - the original spam filters were naive Bayes.
background: Bayes' theorem (1763). Naive Bayes appeared in early information retrieval (Maron, 1961) and became famous for spam filtering (Sahami et al., 1998; Paul Graham's "A Plan for Spam", 2002).
use_cases:
  - Spam and phishing filters
  - Document categorisation and sentiment
  - Medical diagnosis from symptoms
  - Fast baselines on large, high-dimensional data
where_used:
  - 04-spam-detector
prerequisites:
  - 00-math-foundations/03-probability-statistics
code:
  - mlscratch/naive_bayes.py:GaussianNB
  - mlscratch/naive_bayes.py:MultinomialNB
  - mlscratch/naive_bayes.py:BernoulliNB
  - cpp/include/mlcpp/naive_bayes.hpp
demo: demo.py
widgets:
  - naive-bayes
pros:
  - Extremely fast training and prediction, tiny memory
  - Works with very little data and thousands of features
cons:
  - The independence assumption is usually false → probabilities are over-confident
  - Gaussian NB can't model correlated features or complex boundaries
workflow: |
  flowchart LR
    A[training data] --> B[class priors P y]
    A --> C[per-class feature likelihoods P xⱼ given y]
    D[new x] --> E[log P y + Σⱼ log P xⱼ given y]
    B & C --> E
    E --> F[argmax / softmax → class probabilities]
---

## Bayes' rule for classification
$$
P(y\mid x) = \frac{P(x\mid y)\,P(y)}{P(x)} \;\propto\; P(y)\,P(x\mid y)
$$
Estimating $P(x\mid y)$ for a full $d$-dimensional $x$ needs an enormous amount of data. The **naive** assumption:
features are conditionally independent given the class,
$$
P(x\mid y) = \prod_{j=1}^d P(x_j\mid y)
$$
so we only estimate $d$ one-dimensional distributions per class. Prediction:
$$
\hat y = \arg\max_c\; \Big[\log P(y=c) + \sum_j \log P(x_j\mid y=c)\Big]
$$
Work in **log space** - multiplying thousands of small probabilities underflows to 0. Normalise with the log-sum-exp trick.

## Three variants
**Gaussian NB** (continuous features): $P(x_j\mid c) = \mathcal{N}(x_j;\,\mu_{cj}, \sigma^2_{cj})$ with per-class means and variances
(+ a tiny `var_smoothing` so no variance is 0). The boundary is quadratic.

**Multinomial NB** (counts, e.g. word counts):
$$
P(x\mid c) \propto \prod_j \theta_{cj}^{x_j}, \qquad \hat\theta_{cj} = \frac{N_{cj} + \alpha}{N_c + \alpha d}
$$
$N_{cj}$ = total count of word $j$ in class $c$. $\log P(x\mid c) = \sum_j x_j\log\theta_{cj}$ is **linear** in the counts.

**Bernoulli NB** (presence/absence): $P(x\mid c) = \prod_j \theta_{cj}^{x_j}(1-\theta_{cj})^{1-x_j}$ - absent words count as evidence too. Good for short texts.

## Laplace (additive) smoothing
Without it, a word never seen in spam during training gets $\hat\theta = 0$, so $\log 0 = -\infty$ and one word vetoes
every other piece of evidence. Adding $\alpha$ (usually 1) to every count fixes that.

## Why it works despite being "naive"
Classification only needs the **ranking** of $P(y\mid x)$ to be right, not the exact values. Even when the
independence assumption inflates the probabilities, the argmax is often correct. (Don't trust NB's probabilities
without calibration.)

## Exercises
1. In the widget, add points to make the classes correlated (diagonal). How does Gaussian NB's boundary cope?
2. Run the demo; set α = 0 for multinomial NB and find the message that breaks it.
3. Compute by hand the posterior for "free prize" using the demo's word counts.
