"""A tiny data-validation framework: declarative checks + drift (PSI), run on clean vs corrupted data."""

import numpy as np
import pandas as pd

from mlscratch.datasets import DATA_DIR

SCHEMA = {
    "customer_id": {"dtype": "int", "unique": True, "nullable": False},
    "age": {"dtype": "float", "min": 1, "max": 110, "max_null_frac": 0.1},
    "annual_income": {"dtype": "float", "min": 0, "max_null_frac": 0.2},
    "plan": {"allowed": {"basic", "premium", "enterprise"}, "max_null_frac": 0.1},
    "churned": {"allowed": {0, 1, 0.0, 1.0}, "max_null_frac": 0.05},
}


def validate(df: pd.DataFrame) -> list[str]:
    failures = []
    for col, rules in SCHEMA.items():
        if col not in df:
            failures.append(f"{col}: missing column")
            continue
        s = df[col]
        if rules.get("dtype") == "int" and not pd.api.types.is_integer_dtype(s):
            failures.append(f"{col}: expected integers, got {s.dtype}")
        if rules.get("dtype") == "float" and not pd.api.types.is_numeric_dtype(s):
            failures.append(f"{col}: expected numbers, got {s.dtype}")
        null_frac = s.isna().mean()
        if not rules.get("nullable", True) and null_frac > 0:
            failures.append(f"{col}: {s.isna().sum()} nulls but column must be complete")
        if null_frac > rules.get("max_null_frac", 1.0):
            failures.append(f"{col}: {null_frac:.1%} nulls > allowed {rules['max_null_frac']:.0%}")
        if rules.get("unique") and s.duplicated().any():
            failures.append(f"{col}: {s.duplicated().sum()} duplicated values")
        if "min" in rules and (s.dropna() < rules["min"]).any():
            failures.append(f"{col}: {(s < rules['min']).sum()} values below {rules['min']}")
        if "max" in rules and (s.dropna() > rules["max"]).any():
            failures.append(f"{col}: {(s > rules['max']).sum()} values above {rules['max']}")
        if "allowed" in rules:
            bad = set(s.dropna().unique()) - rules["allowed"]
            if bad:
                failures.append(f"{col}: unexpected values {sorted(map(str, bad))[:5]}")
    return failures


def psi(reference, current, bins=10):
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(reference, edges)[0] / len(reference) + 1e-6
    q = np.histogram(current, edges)[0] / len(current) + 1e-6
    return float(np.sum((p - q) * np.log(p / q)))


clean = pd.read_csv(DATA_DIR / "clean_customers.csv")
print("clean data:", validate(clean) or "all checks passed ✓")

broken = clean.copy()
broken.loc[:20, "age"] = broken.loc[:20, "age"] * 12            # someone sent ages in months
broken.loc[5, "customer_id"] = broken.loc[6, "customer_id"]      # duplicated id
broken.loc[30:40, "plan"] = "Platinum"                          # new category upstream
print("\ncorrupted batch:")
for f in validate(broken):
    print("  ✗", f)

rng = np.random.default_rng(0)
ref = clean["annual_income"].dropna().to_numpy()
same = rng.choice(ref, 400)
shifted = rng.choice(ref, 400) * 1.35                            # incomes 35% higher (inflation? new market?)
print(f"\nPSI income, new sample from same distribution: {psi(ref, same):.3f}  (stable < 0.1)")
print(f"PSI income, shifted by +35%:                    {psi(ref, shifted):.3f}  (large shift > 0.25)")
