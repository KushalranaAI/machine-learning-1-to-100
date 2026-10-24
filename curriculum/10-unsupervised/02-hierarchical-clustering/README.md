---
title: Hierarchical Clustering
order: 2
summary: Agglomerative clustering - linkage criteria (single, complete, average, Ward), the Lance-Williams update, dendrograms and cutting the tree.
level: intermediate
minutes: 40
why: Hierarchical clustering doesn't need k up front and produces a full tree (dendrogram) of nested groups - perfect when structure exists at several scales, like taxonomies, product categories or gene families.
background: Agglomerative methods date to numerical taxonomy in the 1950s-60s (Sokal & Sneath). Ward's minimum-variance method (1963) and the Lance-Williams recurrence (1967) remain the standard.
use_cases:
  - Taxonomies and phylogenetic trees
  - Gene expression analysis
  - Grouping documents or products at several granularities
  - Small/medium datasets where a dendrogram aids interpretation
where_used:
  - 03-customer-segmentation
code:
  - mlscratch/cluster.py:AgglomerativeClustering
demo: demo.py
pros:
  - No need to fix k in advance; dendrogram shows every level
  - Works with any distance / linkage
cons:
  - O(n²) memory and O(n²)-O(n³) time - limited to ~10k points
  - Greedy merges can't be undone
workflow: |
  flowchart LR
    A[n singleton clusters] --> B[distance matrix]
    B --> C[merge the closest pair]
    C --> D[update distances with the linkage rule]
    D --> E{one cluster left?}
    E -->|no| C
    E -->|yes| F[dendrogram → cut at k or a height]
---

## Agglomerative algorithm
1. Start with every point as its own cluster.
2. Find the two **closest** clusters and merge them; record the merge height.
3. Update the distances from the new cluster to all others.
4. Repeat until one cluster remains. Cutting the resulting tree at height $h$ (or at $k$ clusters) gives a flat clustering.

## Linkage: what "closest clusters" means
| Linkage | $d(A, B)$ | Behaviour |
|---------|-----------|-----------|
| single | $\min_{a,b}d(a,b)$ | chains along thin structures; sensitive to noise |
| complete | $\max_{a,b}d(a,b)$ | compact, similar-diameter clusters |
| average | mean of all pairwise distances | in between |
| **Ward** | increase in total within-cluster SSE caused by the merge | compact, k-means-like clusters |

Ward's merge cost for clusters $A, B$: $\;\Delta(A, B) = \frac{|A||B|}{|A|+|B|}\|\mu_A - \mu_B\|^2$.

## Lance-Williams update
After merging $i$ and $j$, the distance to any cluster $k$ can be computed from the old distances alone:
$$
d(i\cup j, k) = \alpha_i d(i,k) + \alpha_j d(j,k) + \beta\,d(i,j) + \gamma\,|d(i,k) - d(j,k)|
$$
- single: $\alpha_i = \alpha_j = \tfrac12, \gamma = -\tfrac12$ → $\min$
- complete: $\gamma = +\tfrac12$ → $\max$
- average: $\alpha_i = \frac{n_i}{n_i+n_j}$
- Ward (on squared distances): $\alpha_i = \frac{n_i+n_k}{n_i+n_j+n_k}$, $\beta = \frac{-n_k}{n_i+n_j+n_k}$

`AgglomerativeClustering.fit` implements exactly these updates; the tests confirm identical clusterings to scikit-learn for all four linkages.

## Reading a dendrogram
Long vertical lines = merges that happened at a large distance → clusters that are well separated. Cut where a
horizontal line crosses the longest vertical gaps.

## Exercises
1. Run the demo; compare single and Ward linkage on the moons.
2. Draw the dendrogram for 30 points and pick k by the largest gap.
3. Why does single linkage handle the moons but fail when you add a few noise points between them?
