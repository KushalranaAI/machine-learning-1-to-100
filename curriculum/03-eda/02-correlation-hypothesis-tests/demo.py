"""Correlation and hypothesis tests computed from their formulas (scipy only for p-values / checks)."""

import numpy as np
import pandas as pd
from scipy import stats

from mlscratch.datasets import DATA_DIR, load_iris

rng = np.random.default_rng(0)


def pearson(x, y):
    xc, yc = x - x.mean(), y - y.mean()
    return float(xc @ yc / np.sqrt((xc @ xc) * (yc @ yc)))


def spearman(x, y):
    rank = lambda v: pd.Series(v).rank().to_numpy()  # noqa: E731  (average ranks for ties)
    return pearson(rank(x), rank(y))


x = rng.uniform(-3, 3, 300)
for name, y in [("linear y = 2x + noise", 2 * x + rng.normal(0, 1, 300)),
                ("monotone y = eˣ", np.exp(x) + rng.normal(0, 0.5, 300)),
                ("non-monotone y = x²", x**2 + rng.normal(0, 0.5, 300))]:
    print(f"{name:<24} Pearson {pearson(x, y):+.3f}   Spearman {spearman(x, y):+.3f}")

# --- Welch t-test by hand ---------------------------------------------------------------------
X, y, names = load_iris()
a, b = X[y == 1, 2], X[y == 2, 2]          # petal length: versicolor vs virginica
t = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
print(f"\nWelch t (petal length, versicolor vs virginica): ours {t:.3f}, scipy {stats.ttest_ind(a, b, equal_var=False).statistic:.3f}")

# --- one-way ANOVA by hand ----------------------------------------------------------------
groups = [X[y == k, 1] for k in range(3)]     # sepal width across species
grand = X[:, 1].mean()
ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
F = (ssb / 2) / (ssw / (len(X) - 3))
print(f"ANOVA F (sepal width by species): ours {F:.3f}, scipy {stats.f_oneway(*groups).statistic:.3f}, p = {stats.f.sf(F, 2, len(X) - 3):.2e}")

# --- chi-square test + Cramér's V on churn vs plan -----------------------------------------
cust = pd.read_csv(DATA_DIR / "clean_customers.csv").dropna(subset=["plan", "churned"])
O = pd.crosstab(cust["plan"], cust["churned"]).to_numpy()
E = O.sum(1, keepdims=True) * O.sum(0, keepdims=True) / O.sum()
chi2 = ((O - E) ** 2 / E).sum()
dof = (O.shape[0] - 1) * (O.shape[1] - 1)
v = np.sqrt(chi2 / O.sum() / min(O.shape[0] - 1, O.shape[1] - 1))
print(f"\nchurn vs plan: χ² = {chi2:.2f}, dof = {dof}, p = {stats.chi2.sf(chi2, dof):.3f}, Cramér's V = {v:.3f}")

# --- Simpson's paradox: kidney stones (Charig et al., 1986) ---------------------------------
stones = pd.DataFrame({"treatment": ["A", "A", "B", "B"], "stone": ["small", "large", "small", "large"],
                       "success": [81, 192, 234, 55], "total": [87, 263, 270, 80]})
by_group = stones.assign(rate=stones.success / stones.total).pivot(index="stone", columns="treatment", values="rate")
overall = stones.groupby("treatment")[["success", "total"]].sum()
print("\nSimpson's paradox - success rate by stone size:\n", by_group.round(3))
print("overall:", (overall.success / overall.total).round(3).to_dict(), "→ B looks better overall, A is better in every group!")

# --- multiple testing --------------------------------------------------------------------------
target = rng.integers(0, 2, 200)
noise = rng.normal(size=(200, 50))
p = [stats.ttest_ind(noise[target == 0, j], noise[target == 1, j]).pvalue for j in range(50)]
print(f"\n50 pure-noise features: {sum(pv < 0.05 for pv in p)} 'significant' at 0.05, {sum(pv < 0.05 / 50 for pv in p)} after Bonferroni")
