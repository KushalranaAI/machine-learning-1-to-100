---
title: Probability & Statistics
order: 3
summary: Random variables, distributions, expectation and variance, Bayes' rule, maximum likelihood, the central limit theorem and hypothesis tests.
level: beginner
minutes: 55
why: Data is noisy and models are uncertain. Probability explains where loss functions come from (MSE = Gaussian likelihood, log-loss = Bernoulli likelihood), powers naive Bayes and Gaussian mixtures, and statistics tells you whether a model improvement is real or luck.
background: Bayes' theorem (1763), Gauss's normal distribution (1809) and Fisher's maximum likelihood (1912-1922) are the foundations. Statistical learning theory connects them to modern ML.
use_cases:
  - Choosing a loss function by maximum likelihood
  - Naive Bayes spam filters and Gaussian mixture clustering
  - A/B tests and comparing two models
  - Confidence intervals for metrics
  - Detecting anomalies as low-probability points
where_used:
  - 04-spam-detector
  - 03-customer-segmentation
code:
  - mlscratch/naive_bayes.py:GaussianNB
  - mlscratch/cluster.py:GaussianMixture
demo: demo.py
widgets:
  - distributions
workflow: |
  flowchart LR
    A[Assume a data distribution p x θ] --> B[Write the likelihood]
    B --> C[Take -log → a loss]
    C --> D[Minimise → MLE parameters]
    D --> E[Quantify uncertainty: CIs, tests]
---

## Random variables and distributions
A random variable $X$ maps outcomes to numbers. It is described by a **PMF** $P(X=k)$ (discrete) or a **PDF** $p(x)$ (continuous).

| Distribution | Models | Parameters | Mean | Variance |
|--------------|--------|------------|------|----------|
| Bernoulli | yes/no (click, spam) | $p$ | $p$ | $p(1-p)$ |
| Binomial | # successes in $n$ trials | $n, p$ | $np$ | $np(1-p)$ |
| Poisson | counts per interval | $\lambda$ | $\lambda$ | $\lambda$ |
| Gaussian | measurement noise, heights | $\mu, \sigma^2$ | $\mu$ | $\sigma^2$ |
| Exponential | waiting times | $\lambda$ | $1/\lambda$ | $1/\lambda^2$ |
| Multinomial | word counts | $n, \theta_1..\theta_K$ | $n\theta_k$ | $n\theta_k(1-\theta_k)$ |

Gaussian density: $\;p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$.

## Expectation, variance, covariance
$$
\mathbb{E}[X] = \sum_x x\,p(x), \quad \mathrm{Var}(X) = \mathbb{E}[(X-\mu)^2], \quad \mathrm{Cov}(X,Y) = \mathbb{E}[(X-\mu_X)(Y-\mu_Y)]
$$
Correlation $\rho = \mathrm{Cov}/(\sigma_X\sigma_Y) \in [-1, 1]$ measures **linear** association only.

## Conditional probability and Bayes' rule
$$
P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)} \qquad \text{posterior} = \frac{\text{likelihood}\times\text{prior}}{\text{evidence}}
$$
Example: 1% of emails contain "lottery"; 40% of spam does and 0.5% of ham does; 20% of email is spam.
$P(\text{spam}\mid\text{lottery}) = \frac{0.4 \cdot 0.2}{0.4\cdot 0.2 + 0.005 \cdot 0.8} = 0.952$. That's naive Bayes in one line.

## Maximum likelihood: where loss functions come from
Given data $x_1..x_n$ and a model $p(x\mid\theta)$, choose $\theta$ making the data most probable:
$$
\hat\theta = \arg\max_\theta \prod_i p(x_i \mid \theta) = \arg\min_\theta \; -\sum_i \log p(x_i\mid\theta)
$$
- **Gaussian noise** $y = w^\top x + \varepsilon$, $\varepsilon\sim\mathcal N(0,\sigma^2)$: $-\log p = \frac{(y - w^\top x)^2}{2\sigma^2} + \text{const}$ ⇒ **MSE**.
- **Bernoulli labels** $y\sim\text{Bern}(\sigma(w^\top x))$: $-\log p = -[y\log p + (1-y)\log(1-p)]$ ⇒ **log-loss**.
- **Laplace noise** ⇒ **MAE**. A Gaussian **prior** on $w$ ⇒ **Ridge**; a Laplace prior ⇒ **Lasso** (MAP estimation).

## Law of large numbers & central limit theorem
- **LLN**: the sample mean $\bar X_n \to \mu$ as $n$ grows.
- **CLT**: whatever the original distribution (finite variance), $\bar X_n \approx \mathcal{N}(\mu, \sigma^2/n)$.
The standard error shrinks like $1/\sqrt{n}$ - to halve your uncertainty you need **4×** the data.
Try it in the interactive tab with a skewed distribution.

## Confidence intervals and hypothesis tests
- 95% CI for a mean: $\bar x \pm 1.96\,\frac{s}{\sqrt n}$. For a model's accuracy $\hat p$ on $n$ test samples: $\hat p \pm 1.96\sqrt{\hat p(1-\hat p)/n}$.
  With 200 test samples and 85% accuracy, the CI is ±5% - small test sets can't distinguish 85% from 88%!
- **Hypothesis test** recipe: null hypothesis $H_0$ (no difference) → test statistic → p-value $= P(\text{statistic this extreme}\mid H_0)$. Small p (< 0.05) ⇒ reject $H_0$.
- **Bootstrap**: resample the data with replacement many times and recompute the statistic - a distribution-free CI (also the basis of bagging, Part 9).

## Exercises
1. In the widget, set exponential with λ=1 and n=30. Is the mean's histogram symmetric? Compare its std with $1/\sqrt{30}$.
2. Show that maximising the Gaussian likelihood for $\mu$ gives the sample mean.
3. Use the demo's bootstrap to compute a 95% CI for the median of a skewed sample.
