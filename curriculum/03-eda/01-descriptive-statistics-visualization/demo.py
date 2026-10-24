"""EDA essentials: summary statistics, Anscombe's quartet, class-wise plots and a correlation heatmap."""

import numpy as np
import pandas as pd

from mlscratch.datasets import DATA_DIR, load_iris
from mlscratch.viz import plt, savefig

pyplot = plt()

# --- Anscombe's quartet: identical statistics, different stories --------------------------
x = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
quartet = {
    "I": (x, np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])),
    "II": (x, np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])),
    "III": (x, np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])),
    "IV": (np.array([8] * 7 + [19] + [8] * 3), np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89])),
}
fig, axes = pyplot.subplots(1, 4, figsize=(15, 3.3), sharey=True)
print(f"{'set':<5}{'mean x':>8}{'mean y':>8}{'var y':>8}{'corr':>7}   fitted line")
for ax, (name, (xs, ys)) in zip(axes, quartet.items()):
    slope, intercept = np.polyfit(xs, ys, 1)
    print(f"{name:<5}{xs.mean():>8.2f}{ys.mean():>8.2f}{ys.var(ddof=1):>8.2f}{np.corrcoef(xs, ys)[0, 1]:>7.3f}   y = {slope:.2f}x + {intercept:.2f}")
    ax.scatter(xs, ys, color="#4263eb")
    grid = np.linspace(3, 20, 10)
    ax.plot(grid, slope * grid + intercept, color="#e8590c")
    ax.set_title(f"Anscombe {name}")
savefig(fig, "anscombe")

# --- iris: which feature separates the species? ------------------------------------------
X, y, names = load_iris()
cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
iris = pd.DataFrame(X, columns=cols).assign(species=[names[i] for i in y])
print("\nper-species means:\n", iris.groupby("species").mean().round(2))
fig, axes = pyplot.subplots(1, 4, figsize=(15, 3.2))
for ax, c in zip(axes, cols):
    ax.boxplot([iris.loc[iris.species == s, c] for s in names], tick_labels=names)
    ax.set_title(c)
savefig(fig, "iris_boxplots")

corr = iris[cols].corr()
fig, ax = pyplot.subplots(figsize=(5, 4))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(4), cols, rotation=45, ha="right")
ax.set_yticks(range(4), cols)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im)
ax.set_title("correlation heatmap")
savefig(fig, "iris_correlation")

# --- skew and the log transform on income -------------------------------------------------
cust = pd.read_csv(DATA_DIR / "clean_customers.csv")
inc = cust["annual_income"].dropna()
skew = lambda s: ((s - s.mean()) ** 3).mean() / s.std() ** 3  # noqa: E731
print(f"\nincome: mean {inc.mean():,.0f} vs median {inc.median():,.0f} → skewness {skew(inc):.2f}; log-income skewness {skew(np.log(inc)):.2f}")
print("churn rate by plan:\n", cust.groupby("plan")["churned"].mean().round(3))
