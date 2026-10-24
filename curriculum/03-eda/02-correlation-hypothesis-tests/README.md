---
title: Correlation & Hypothesis Testing
order: 2
summary: Pearson, Spearman and Cramér's V; t-tests, ANOVA and chi-square tests computed from their formulas; correlation vs causation and Simpson's paradox.
level: intermediate
minutes: 45
why: EDA produces many apparent patterns. Correlation measures and statistical tests tell you which relationships are strong and which differences are real rather than noise - and filter-based feature selection (Part 5) is built directly on these statistics.
background: Pearson's correlation (1896), Student's t-test (Gosset at the Guinness brewery, 1908), Fisher's ANOVA (1918) and Pearson's chi-square test (1900) are the workhorses of applied statistics.
use_cases:
  - Ranking features by their association with the target
  - Checking whether two groups (A/B test) really differ
  - Finding redundant features
  - Testing independence of two categorical variables
where_used:
  - 02-titanic-survival
code:
  - mlscratch/feature_selection.py:f_classif
  - mlscratch/feature_selection.py:chi2
demo: demo.py
workflow: |
  flowchart TD
    A{Variable types?} -->|numeric-numeric| B[Pearson linear / Spearman monotonic]
    A -->|numeric-categorical| C[t-test 2 groups / ANOVA k groups]
    A -->|categorical-categorical| D[chi-square test, Cramér's V]
    B & C & D --> E[effect size + p-value]
    E --> F[Correlation ≠ causation: check confounders]
---

## Correlation coefficients
**Pearson** (linear): $\;r = \frac{\sum (x_i-\bar x)(y_i - \bar y)}{\sqrt{\sum(x_i-\bar x)^2}\sqrt{\sum(y_i-\bar y)^2}}$

**Spearman** (monotonic): Pearson computed on the **ranks** - robust to outliers and captures any monotone curve.

**Cramér's V** (two categorical variables): $\;V = \sqrt{\frac{\chi^2/n}{\min(r-1,\,c-1)}} \in [0, 1]$.

$r = 0$ means "no **linear** relation", not "no relation": $y = x^2$ on symmetric $x$ has $r \approx 0$.

## Hypothesis testing in one picture
1. Null hypothesis $H_0$: no effect (means equal, variables independent).
2. Compute a **test statistic** that grows with evidence against $H_0$.
3. **p-value**: probability of a statistic at least this extreme if $H_0$ were true.
4. p < α (often 0.05) ⇒ reject $H_0$. Also report the **effect size** - with huge $n$, tiny meaningless differences become "significant".

### Welch's t-test (two group means)
$$
t = \frac{\bar x_1 - \bar x_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}
$$
### One-way ANOVA (k group means)
$$
F = \frac{\text{between-group variance}}{\text{within-group variance}} = \frac{\sum_k n_k(\bar x_k - \bar x)^2/(K-1)}{\sum_k\sum_{i\in k}(x_i-\bar x_k)^2/(n-K)}
$$
### Chi-square test of independence (two categorical variables)
$$
\chi^2 = \sum_{cells} \frac{(O - E)^2}{E}, \qquad E_{ij} = \frac{\text{row}_i \cdot \text{col}_j}{n}
$$

## Correlation is not causation
- **Confounders**: ice-cream sales and drownings both rise in summer.
- **Simpson's paradox**: a trend in every subgroup reverses when the groups are combined. The demo reproduces the
  classic kidney-stone example - treatment A wins for small stones *and* for large stones, but loses overall,
  because A was given mostly for the harder (large) cases.
- **Multiple testing**: test 100 random features at α=0.05 and ~5 will be "significant". Correct with Bonferroni (α/m) or use a held-out set.

## Exercises
1. Run the demo; verify our t and F statistics against `scipy.stats`.
2. Generate $y = x^2 + \text{noise}$ and compare Pearson vs Spearman vs mutual information.
3. Test 50 pure-noise features against a random target; how many have p < 0.05?
