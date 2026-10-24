"""Compare outlier rules on real-looking income data, then show what outliers do to a regression."""

import numpy as np
import pandas as pd

from mlscratch.anomaly import IsolationForest, iqr_outliers, mahalanobis, zscore_outliers
from mlscratch.datasets import DATA_DIR
from mlscratch.linear_model import LinearRegression
from mlscratch.viz import plt, savefig

df = pd.read_csv(DATA_DIR / "clean_customers.csv") if (DATA_DIR / "clean_customers.csv").exists() else None
if df is None:
    raise SystemExit("data/clean_customers.csv is missing - run the data-cleaning lesson demo to recreate it.")
income = df["annual_income"].dropna().to_numpy()

mad = np.median(np.abs(income - np.median(income)))
modified_z = 0.6745 * (income - np.median(income)) / mad
rules = {
    "z-score > 3": zscore_outliers(income, 3),
    "IQR k=1.5": iqr_outliers(income, 1.5),
    "IQR k=3": iqr_outliers(income, 3),
    "modified z (MAD) > 3.5": np.abs(modified_z) > 3.5,
    "IQR on log(income)": iqr_outliers(np.log(income), 1.5),
}
print(f"{len(income)} incomes, median {np.median(income):,.0f}, max {income.max():,.0f}")
for name, mask in rules.items():
    print(f"  {name:<24} flags {mask.sum():>3}  (smallest flagged: {income[mask].min() if mask.any() else float('nan'):,.0f})")

# Multivariate: height vs income
X = df[["height_cm", "annual_income"]].dropna().to_numpy()
d = mahalanobis(np.c_[X[:, 0], np.log(X[:, 1])])
print("\ntop-5 Mahalanobis distances (height, log income):", np.round(np.sort(d)[-5:], 2))
iso = IsolationForest(contamination=0.02, random_state=0).fit(np.c_[X[:, 0], np.log(X[:, 1])])
print("Isolation Forest flags", int((iso.predict(np.c_[X[:, 0], np.log(X[:, 1])]) == -1).sum()), "points")

# Effect on a regression: visits ~ income, with and without winsorising
data = df[["annual_income", "monthly_visits"]].dropna()
x, y = data["annual_income"].to_numpy() / 1000, data["monthly_visits"].to_numpy()
lo, hi = np.percentile(x, [1, 99])
x_w = np.clip(x, lo, hi)
raw_fit = LinearRegression().fit(x, y)
win_fit = LinearRegression().fit(x_w, y)
print(f"\nslope of visits on income (k$): raw {raw_fit.coef_[0]:.5f}  winsorised {win_fit.coef_[0]:.5f}")

pyplot = plt()
fig, axes = pyplot.subplots(1, 3, figsize=(14, 3.8))
axes[0].hist(income, bins=60, color="#4263eb")
axes[0].set_title("income: long right tail")
axes[1].hist(np.log(income), bins=60, color="#2b8a3e")
axes[1].set_title("log(income): roughly symmetric")
axes[2].boxplot([x, x_w], tick_labels=["raw", "winsorised 1-99%"])
axes[2].set_title("income (k$) before / after capping")
savefig(fig, "outliers")
