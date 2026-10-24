---
title: Association Rules (Apriori)
order: 6
summary: Market-basket analysis - support, confidence and lift, the Apriori principle for pruning, generating rules, and interpreting them.
level: beginner
minutes: 30
why: "\"Customers who buy X also buy Y\" drives product placement, recommendations and bundling. Association rules find such patterns in transaction data without any labels."
background: Agrawal, Imieliński and Swami introduced association rule mining (1993) and Agrawal & Srikant the Apriori algorithm (1994), analysing supermarket baskets. FP-Growth (Han et al., 2000) avoids candidate generation.
use_cases:
  - Market-basket analysis and store layout
  - "\"Frequently bought together\" recommendations"
  - Web usage mining, medical co-occurrence of symptoms
where_used:
  - 03-customer-segmentation
code:
  - mlscratch/association.py:apriori
  - mlscratch/association.py:association_rules
demo: demo.py
cons:
  - Number of candidate itemsets explodes with low support thresholds
  - Rules show co-occurrence, not causation
workflow: |
  flowchart LR
    A[transactions] --> B[frequent 1-itemsets ≥ min support]
    B --> C[join → k-itemset candidates]
    C --> D[prune: every k−1 subset frequent]
    D --> E[count support, keep frequent]
    E --> C
    E --> F[rules A→B with confidence and lift]
---

## Measures
For itemsets $A$ and $B$ over $N$ transactions:
$$
\text{support}(A) = \frac{\#\{t : A\subseteq t\}}{N}, \quad \text{confidence}(A\to B) = \frac{\text{support}(A\cup B)}{\text{support}(A)} = P(B\mid A), \quad \text{lift}(A\to B) = \frac{\text{confidence}(A\to B)}{\text{support}(B)}
$$
- **Lift > 1**: A and B occur together more than if they were independent. Lift = 1: independent. < 1: substitutes.
- High confidence alone can mislead: if 90% of baskets contain bread, "anything → bread" has 90% confidence and lift ≈ 1.

## The Apriori principle
If an itemset is frequent, **all its subsets are frequent**. Equivalently, if $\{a, b\}$ is infrequent, no superset
containing $\{a, b\}$ can be frequent. So we grow itemsets level by level and only generate size-$k$ candidates whose
every $(k-1)$-subset was frequent - pruning most of the exponential search space.

## From itemsets to rules
For each frequent itemset $S$ and each non-empty proper subset $A\subset S$, form $A\to S\setminus A$ and keep it if its
confidence ≥ the threshold. Sort by lift.

## Exercises
1. Run the demo and interpret the top three rules by lift.
2. Lower min_support - how quickly does the number of frequent itemsets grow?
3. Find a rule with high confidence but lift < 1 and explain why it's useless.
