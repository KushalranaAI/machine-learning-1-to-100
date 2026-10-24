---
title: Decision Trees (CART)
order: 6
summary: Recursive binary splitting - Gini impurity, entropy and information gain, regression trees with MSE, the O(n log n) split search, stopping rules, pruning, and feature importance.
level: intermediate
minutes: 60
why: Trees mirror human decision rules, handle mixed feature types without scaling, capture interactions automatically, and are the building block of random forests and gradient boosting - the most successful algorithms on tabular data.
background: CART (Breiman, Friedman, Olshen & Stone, 1984) and ID3/C4.5 (Quinlan, 1986/1993) are the classic tree algorithms. Our implementation follows CART - binary splits, Gini or entropy, MSE for regression.
use_cases:
  - Credit and insurance rules that must be explained
  - Medical triage and diagnostic flowcharts
  - Base learners for random forests and boosting
  - Quick insight into which features matter
where_used:
  - 02-titanic-survival
  - 01-house-prices
prerequisites:
  - 00-math-foundations/03-probability-statistics
code:
  - mlscratch/tree.py:gini_from_counts
  - mlscratch/tree.py:DecisionTreeClassifier._best_split
  - mlscratch/tree.py:DecisionTreeRegressor._best_split
  - mlscratch/tree.py:_BaseTree._grow
  - cpp/include/mlcpp/decision_tree.hpp
demo: demo.py
widgets:
  - tree-split
  - decision-tree
pros:
  - Interpretable, no scaling needed, handles non-linearity and interactions
  - Fast prediction (depth comparisons)
cons:
  - High variance - small data changes give different trees
  - Greedy, axis-aligned splits; overfits without depth / leaf limits
workflow: |
  flowchart TD
    A[node with samples] --> B{stop? depth, min samples, pure}
    B -->|yes| L[leaf: majority class / mean]
    B -->|no| C[for each feature: sort, sweep thresholds]
    C --> D[gain = I parent − weighted I children]
    D --> E[best feature + threshold]
    E --> F[left child x ≤ t] --> A
    E --> G[right child x > t] --> A
---

## 1. Impurity: how mixed is a node?
With class proportions $p_k$ in a node:

| Criterion | Formula | Pure node | 50/50 binary node |
|-----------|---------|-----------|-------------------|
| **Gini** | $1 - \sum_k p_k^2$ | 0 | 0.5 |
| **Entropy** | $-\sum_k p_k\log_2 p_k$ | 0 | 1 |
| Misclassification | $1 - \max_k p_k$ | 0 | 0.5 |
| **MSE** (regression) | $\frac1n\sum(y_i - \bar y)^2$ | 0 | - |

Gini = probability that two random samples from the node have different labels. Entropy = average bits needed to
encode a label. They almost always choose the same splits; Gini is slightly cheaper (no log).

## 2. The best split
A split of node $P$ into $L$ (where $x_f \le t$) and $R$ is scored by the impurity decrease
$$
\Delta I = I(P) - \frac{n_L}{n_P}I(L) - \frac{n_R}{n_P}I(R)
$$
With entropy, $\Delta I$ is the **information gain**. The tree greedily picks the feature $f$ and threshold $t$ with the largest $\Delta I$.

## 3. Finding it in O(n log n) per feature
Trying every threshold naively re-counts classes each time: $O(n^2)$. Instead (see `_best_split`):
1. sort the node's samples by $x_f$,
2. sweep left to right keeping **running class counts** (cumulative sums of one-hot labels) for the left side; the
   right side is total minus left,
3. evaluate the gain of all $n-1$ candidate thresholds in one vectorised step, skipping equal adjacent values,
4. threshold = midpoint between the two neighbouring values.

For regression the same trick uses running sums of $y$ and $y^2$, since $\text{SSE} = \sum y^2 - (\sum y)^2/n$.

## 4. Growing and stopping
Recurse until a stopping rule fires: `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease`,
or the node is pure. A leaf predicts the majority class (and class frequencies as probabilities), or the mean for regression.

## 5. Overfitting and pruning
A fully grown tree memorises the training set (100% training accuracy, poor test accuracy). Control it with the stopping rules
(**pre-pruning**) - tune `max_depth` / `min_samples_leaf` by CV - or grow fully and cut back with **cost-complexity
pruning** (minimise $\text{error} + \alpha\cdot|\text{leaves}|$).

## 6. Feature importance
Sum of $n_{node}\cdot\Delta I$ over all splits on a feature, normalised to 1. Matches scikit-learn exactly (see the tests).
Beware: biased towards continuous / high-cardinality features - use permutation importance for a fairer view.

## Exercises
1. In the split explorer, find the best split by hand, then press "Jump to best split".
2. Run the demo; compare train/test accuracy as max_depth grows. Where does it start to overfit?
3. Print the tree with `export_text` and translate the top three levels into plain-English rules.
