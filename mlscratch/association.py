"""Association rule mining with Apriori (market-basket analysis).

support(A) = P(A),  confidence(A→B) = P(B|A) = support(A∪B)/support(A),  lift = confidence / support(B)
Apriori principle: every subset of a frequent itemset is frequent → prune candidates level by level.
"""

from __future__ import annotations

from itertools import combinations


def apriori(transactions: list[set], min_support: float = 0.1) -> dict[frozenset, float]:
    n = len(transactions)
    transactions = [frozenset(t) for t in transactions]
    items = {frozenset([i]) for t in transactions for i in t}

    def support(itemsets):
        return {s: sum(1 for t in transactions if s <= t) / n for s in itemsets}

    frequent, level = {}, {s: v for s, v in support(items).items() if v >= min_support}
    k = 1
    while level:
        frequent.update(level)
        k += 1
        keys = list(level)
        candidates = {a | b for a in keys for b in keys if len(a | b) == k}
        # prune: all (k-1)-subsets must already be frequent
        candidates = {c for c in candidates if all(frozenset(s) in level for s in combinations(c, k - 1))}
        level = {s: v for s, v in support(candidates).items() if v >= min_support}
    return frequent


def association_rules(frequent: dict[frozenset, float], min_confidence: float = 0.5) -> list[dict]:
    rules = []
    for itemset, sup in frequent.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for antecedent in map(frozenset, combinations(itemset, r)):
                consequent = itemset - antecedent
                conf = sup / frequent[antecedent]
                if conf >= min_confidence:
                    rules.append({"antecedent": set(antecedent), "consequent": set(consequent), "support": sup,
                                  "confidence": conf, "lift": conf / frequent[consequent]})
    return sorted(rules, key=lambda r: (-r["lift"], -r["confidence"]))
