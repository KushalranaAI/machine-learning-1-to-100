---
title: Gaussian Mixtures & the EM Algorithm
order: 4
summary: Soft clustering with a mixture of Gaussians, the Expectation-Maximisation algorithm derived, why the likelihood never decreases, choosing k with BIC, and k-means as a special case.
level: advanced
minutes: 55
why: Gaussian mixtures give probabilistic, overlapping, elliptical clusters and a density model you can sample from or use for anomaly detection. EM is a general recipe for fitting any model with hidden variables (HMMs, topic models, missing data).
background: Pearson fitted a two-component mixture to crab measurements in 1894. Dempster, Laird and Rubin formalised the EM algorithm in 1977 - one of the most cited papers in statistics.
use_cases:
  - Soft customer segmentation with membership probabilities
  - Density estimation and anomaly scoring
  - Speaker identification (GMM-UBM), image segmentation
  - Any latent-variable model (EM generalises)
where_used:
  - 03-customer-segmentation
prerequisites:
  - 00-math-foundations/03-probability-statistics
  - 10-unsupervised/01-k-means
code:
  - mlscratch/cluster.py:GaussianMixture
demo: demo.py
pros:
  - Elliptical, overlapping clusters with probabilities
  - Principled model selection (likelihood, BIC)
cons:
  - Local optima - initialise from k-means, restart
  - Full covariances need many parameters (d² per component)
workflow: |
  flowchart LR
    A[init from k-means] --> B[E-step: responsibilities r_ik]
    B --> C[M-step: weights, means, covariances]
    C --> D[log-likelihood]
    D --> E{improved > tol?}
    E -->|yes| B
    E -->|no| F[soft clusters + density p x]
---

## The model
$$
p(x) = \sum_{k=1}^K \pi_k\,\mathcal N(x\mid\mu_k, \Sigma_k), \qquad \pi_k \ge 0,\ \sum_k\pi_k = 1
$$
Generative story: pick component $z \sim \text{Categorical}(\pi)$, then draw $x \sim \mathcal N(\mu_z, \Sigma_z)$. The
component $z$ is a **hidden (latent) variable**.

## Why not just maximise the likelihood directly?
$\log p(X) = \sum_i\log\sum_k\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)$ - the log of a sum has no closed-form maximiser.
If we *knew* each point's component, we'd just fit one Gaussian per group. EM alternates between guessing the
components (softly) and fitting the Gaussians.

## E-step: responsibilities
The posterior probability that component $k$ generated $x_i$:
$$
r_{ik} = \frac{\pi_k\,\mathcal N(x_i\mid\mu_k,\Sigma_k)}{\sum_j\pi_j\,\mathcal N(x_i\mid\mu_j,\Sigma_j)}
$$
(computed in log space with log-sum-exp; the Gaussian log-density uses a Cholesky factor of $\Sigma_k$).

## M-step: weighted maximum likelihood
With $N_k = \sum_i r_{ik}$:
$$
\pi_k = \frac{N_k}{n}, \qquad \mu_k = \frac{1}{N_k}\sum_i r_{ik}x_i, \qquad \Sigma_k = \frac{1}{N_k}\sum_i r_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top + \epsilon I
$$
## Why the likelihood never decreases
For any distribution $q(z)$ over the latent variables, Jensen's inequality gives the **evidence lower bound**
$$
\log p(X\mid\theta) \ge \sum_z q(z)\log\frac{p(X, z\mid\theta)}{q(z)} = \text{ELBO}(q, \theta)
$$
The E-step sets $q$ to the exact posterior, making the bound tight; the M-step maximises the bound over $\theta$.
So each iteration can only increase $\log p(X\mid\theta)$ (the tests check this on every run).

## k-means as a limit
With all $\Sigma_k = \sigma^2I$ and $\sigma\to 0$, responsibilities become hard 0/1 assignments to the nearest mean and
EM becomes Lloyd's algorithm.

## Choosing K: BIC
$\text{BIC} = -2\log\hat L + p\log n$ with $p$ = number of free parameters. Lower is better; the penalty stops the
likelihood from always preferring more components.

## Exercises
1. Run the demo: compare GMM and k-means on elongated, overlapping clusters.
2. Plot the log-likelihood per EM iteration - is it ever decreasing?
3. Use the fitted density $p(x)$ to flag the 1% least likely points as anomalies.
