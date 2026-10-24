"""Probability in practice: MLE = familiar losses, Bayes' rule, the CLT, confidence intervals and a bootstrap."""

import numpy as np

from mlscratch.viz import plt, savefig

rng = np.random.default_rng(0)

# --- maximum likelihood for a Gaussian = sample mean and (biased) variance -------------------
x = rng.normal(5.0, 2.0, size=500)
mus = np.linspace(3, 7, 401)
nll = [0.5 * np.sum((x - m) ** 2) for m in mus]          # -log-likelihood (σ fixed) up to constants
print(f"MLE of μ by grid search: {mus[np.argmin(nll)]:.3f}   sample mean: {x.mean():.3f}")

# --- Bayes' rule: the spam example from the lesson ------------------------------------------
p_spam, p_word_spam, p_word_ham = 0.2, 0.4, 0.005
posterior = p_word_spam * p_spam / (p_word_spam * p_spam + p_word_ham * (1 - p_spam))
print(f"P(spam | 'lottery') = {posterior:.3f}")

# --- central limit theorem with a skewed (exponential) distribution --------------------------
pyplot = plt()
fig, axes = pyplot.subplots(1, 4, figsize=(14, 3))
for ax, n in zip(axes, [1, 2, 10, 50]):
    means = rng.exponential(1.0, size=(20000, n)).mean(axis=1)
    ax.hist(means, bins=60, density=True, color="#4263eb", alpha=0.7)
    ax.set_title(f"mean of n={n} draws\nstd={means.std():.3f} (theory {1 / np.sqrt(n):.3f})")
savefig(fig, "central_limit_theorem")

# --- confidence interval for a classifier's accuracy -----------------------------------------
for n_test in (100, 1000, 10000):
    acc = 0.85
    half = 1.96 * np.sqrt(acc * (1 - acc) / n_test)
    print(f"accuracy 85% on {n_test:>5} test samples → 95% CI [{acc - half:.3f}, {acc + half:.3f}]")

# --- bootstrap CI for the median of skewed data -----------------------------------------------
income = rng.lognormal(10, 0.8, size=300)
boot = np.array([np.median(rng.choice(income, len(income))) for _ in range(5000)])
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"median income {np.median(income):,.0f}, bootstrap 95% CI [{lo:,.0f}, {hi:,.0f}]")

# --- a two-sample permutation test: is model B really better than model A? -------------------
a = rng.binomial(1, 0.80, 400)  # per-sample correctness of model A
b = rng.binomial(1, 0.84, 400)  # ... and model B
observed = b.mean() - a.mean()
pooled = np.concatenate([a, b])
perm = np.array([np.diff([s[:400].mean(), s[400:].mean()])[0] for s in (rng.permutation(pooled) for _ in range(5000))])
p_value = np.mean(perm >= observed)
print(f"B - A accuracy = {observed:+.3f}, permutation-test p-value = {p_value:.3f}"
      f" → {'significant' if p_value < 0.05 else 'could be luck'} at the 5% level")
