"""Step 2 - exploratory data analysis: what is in the data, what is wrong with it, what predicts price?"""

import numpy as np

from features import CAP, add_ratios, load
from mlscratch.viz import plt, savefig

raw = load(drop_capped=False)
print(f"rows: {len(raw):,}")
print("\nmissing values per column:")
print(raw.isna().sum()[raw.isna().sum() > 0].to_string())
capped = (raw.median_house_value >= CAP).mean()
print(f"\n{capped:.1%} of districts sit exactly at the ${CAP:,} cap - the true value is unknown (censored). "
      "train.py drops them.")
print("\nocean_proximity counts:")
print(raw.ocean_proximity.value_counts().to_string())

df = add_ratios(load())
print("\nskewness (|skew| > 1 → consider a log transform):")
skew = df.select_dtypes("number").skew().sort_values(ascending=False)
print(skew.round(2).to_string())

corr = df.select_dtypes("number").corr()["median_house_value"].drop("median_house_value").sort_values()
print("\nPearson correlation with median_house_value:")
print(corr.round(3).to_string())

P = plt()
fig, axes = P.subplots(1, 3, figsize=(16, 4.5))
axes[0].hist(raw.median_house_value / 1000, bins=60, color="#4c78a8")
axes[0].axvline(CAP / 1000, color="red", ls="--", label="census cap")
axes[0].set(title="Target: median house value (k$)", xlabel="k$")
axes[0].legend()
sc = axes[1].scatter(df.longitude, df.latitude, c=df.median_house_value / 1000, s=df.population / 300, cmap="viridis", alpha=0.5)
fig.colorbar(sc, ax=axes[1], label="k$")
axes[1].set(title="Location × price (size = population)", xlabel="longitude", ylabel="latitude")
axes[2].scatter(df.median_income, df.median_house_value / 1000, s=3, alpha=0.2)
axes[2].set(title=f"Strongest single predictor: income (r = {corr['median_income']:.2f})", xlabel="median income (×$10k)", ylabel="k$")
savefig(fig, "01_eda_overview")

fig, ax = P.subplots(figsize=(7, 5))
ax.barh(corr.index, corr.values, color=["#e45756" if v < 0 else "#54a24b" for v in corr.values])
ax.set(title="Correlation with price", xlabel="Pearson r")
savefig(fig, "02_correlations")

fig, axes = P.subplots(1, 2, figsize=(11, 4))
axes[0].hist(df.people_per_household, bins=80, range=(0, 10))
axes[0].set(title=f"people_per_household (skew {skew['people_per_household']:.0f})")
axes[1].hist(np.log1p(df.people_per_household), bins=80)
axes[1].set(title="after log1p - much closer to symmetric")
savefig(fig, "03_log_transform")
