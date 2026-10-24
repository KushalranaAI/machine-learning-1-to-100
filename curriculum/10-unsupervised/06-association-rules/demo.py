"""Apriori from scratch on simulated grocery baskets: frequent itemsets, rules, and the confidence-vs-lift trap."""

import numpy as np

from mlscratch.association import apriori, association_rules

rng = np.random.default_rng(0)
items = ["bread", "milk", "eggs", "butter", "beer", "diapers", "chips", "salsa", "coffee", "sugar"]
baskets = []
for _ in range(2000):
    b = {i for i in items if rng.random() < 0.12}
    if rng.random() < 0.85:
        b.add("bread")                                   # almost everyone buys bread
    if rng.random() < 0.25:
        b |= {"chips", "salsa"}                          # a strong pair
    if rng.random() < 0.15:
        b |= {"beer", "diapers"}                         # the famous (urban-legend) pair
    if "coffee" in b and rng.random() < 0.7:
        b.add("sugar")
    baskets.append(b)

for s in (0.2, 0.1, 0.05, 0.02):
    print(f"min_support {s:<5} → {len(apriori(baskets, s)):>4} frequent itemsets")

freq = apriori(baskets, 0.05)
rules = association_rules(freq, min_confidence=0.5)
print(f"\ntop rules by lift (from {len(rules)} rules with confidence ≥ 0.5):")
for r in rules[:6]:
    print(f"  {sorted(r['antecedent'])} → {sorted(r['consequent'])}   support {r['support']:.3f}  confidence {r['confidence']:.2f}  lift {r['lift']:.2f}")

bread = [r for r in rules if r["consequent"] == {"bread"}]
if bread:
    r = max(bread, key=lambda r: r["confidence"])
    print(f"\nconfidence trap: {sorted(r['antecedent'])} → ['bread'] has confidence {r['confidence']:.2f} but lift {r['lift']:.2f} "
          "- bread is in most baskets anyway, so this rule tells us nothing.")
