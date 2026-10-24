"""Step 2 - EDA: who survived, and which raw columns hide useful features?"""

from features import load, parse
from mlscratch.viz import plt, savefig

df = parse(load())
print(f"{len(df)} passengers, overall survival rate {df.Survived.mean():.1%} (the 'always predict died' baseline gets {1 - df.Survived.mean():.1%})\n")
print("missing values:")
print(df[["Age", "Cabin", "Embarked", "Fare"]].isna().sum().to_string())

for col in ["Sex", "Pclass", "Title", "Embarked", "FamilySize", "HasCabin", "Deck"]:
    t = df.groupby(col).Survived.agg(["mean", "count"]).rename(columns={"mean": "survival"})
    print(f"\nsurvival by {col}:")
    print(t.round(3).to_string())

print("\nWomen and children first? survival by sex × class:")
print(df.pivot_table(index="Sex", columns="Pclass", values="Survived").round(2).to_string())
print("\nAge median by title (used to impute missing ages - far better than one global median):")
print(df.groupby("Title").Age.median().round(1).to_string())

P = plt()
fig, axes = P.subplots(1, 3, figsize=(16, 4.3))
for ax, col in zip(axes, ["Sex", "Pclass", "Title"]):
    rates = df.groupby(col).Survived.mean()
    ax.bar(rates.index.astype(str), rates.values, color="#4c78a8")
    ax.set(title=f"Survival rate by {col}", ylim=(0, 1))
savefig(fig, "01_survival_by_group")

fig, ax = P.subplots(figsize=(8, 4))
ax.hist([df.Age[df.Survived == 1].dropna(), df.Age[df.Survived == 0].dropna()], bins=30, stacked=True, label=["survived", "died"])
ax.set(title="Age distribution by outcome - young children were prioritised", xlabel="age")
ax.legend()
savefig(fig, "02_age")
