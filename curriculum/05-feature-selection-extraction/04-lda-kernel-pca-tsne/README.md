---
title: LDA, Kernel PCA & t-SNE
order: 4
summary: Supervised projection with Fisher's LDA, non-linear PCA with kernels, and t-SNE for visualising clusters - what each optimises and when to use it.
level: advanced
minutes: 50
why: PCA only finds linear directions of high variance. LDA uses the labels to find the directions that best separate classes; kernel PCA and t-SNE unfold curved structure that linear methods cannot.
background: Fisher introduced linear discriminants in 1936 on the iris data. Kernel PCA (Schölkopf, Smola & Müller, 1998) brought the kernel trick to PCA. t-SNE (van der Maaten & Hinton, 2008) became the standard tool for visualising embeddings.
use_cases:
  - LDA - dimensionality reduction before classification, face recognition (Fisherfaces)
  - Kernel PCA - non-linear denoising and feature extraction
  - t-SNE - visualising word embeddings, single-cell data, neural-network features
where_used:
  - 05-digit-recognizer
  - 03-customer-segmentation
code:
  - mlscratch/decomposition.py:LinearDiscriminantAnalysis
  - mlscratch/decomposition.py:KernelPCA
  - mlscratch/decomposition.py:TSNE
demo: demo.py
cons:
  - LDA assumes Gaussian classes with a shared covariance; at most K-1 components
  - Kernel PCA needs an n×n kernel matrix (memory O(n²))
  - t-SNE is for visualisation only - distances between clusters and cluster sizes are not meaningful
workflow: |
  flowchart TD
    A{What do you need?} -->|labels available, separate classes| L[LDA: max S_B / S_W]
    A -->|non-linear structure, features for a model| K[Kernel PCA: eigenvectors of centred K]
    A -->|2-D picture of clusters| T[t-SNE: match neighbour probabilities]
---

## Fisher's Linear Discriminant Analysis
Find the direction $w$ that pushes class means apart while keeping each class tight:
$$
J(w) = \frac{w^\top S_B w}{w^\top S_W w}, \qquad S_W = \sum_k\sum_{i\in k}(x_i-\mu_k)(x_i-\mu_k)^\top,\quad S_B = \sum_k n_k(\mu_k-\mu)(\mu_k-\mu)^\top
$$
Setting the derivative to zero gives the generalised eigenproblem $S_W^{-1}S_B\,w = \lambda w$. $S_B$ has rank at most
$K-1$, so LDA gives at most $K-1$ useful directions (2 for iris). As a classifier, LDA assumes Gaussian classes with a
shared covariance $\Sigma$ and predicts with the linear discriminant
$\delta_k(x) = x^\top\Sigma^{-1}\mu_k - \tfrac12\mu_k^\top\Sigma^{-1}\mu_k + \log\pi_k$.

## Kernel PCA
Map the data through a feature map $\phi(x)$ (possibly infinite-dimensional) and run PCA there. We never compute
$\phi$: everything is expressed through the kernel $K_{ij} = k(x_i, x_j) = \phi(x_i)^\top\phi(x_j)$ (the **kernel trick**).
1. Centre the kernel matrix: $\tilde K = K - \mathbf{1}_nK - K\mathbf{1}_n + \mathbf{1}_nK\mathbf{1}_n$.
2. Eigen-decompose $\tilde K = A\Lambda A^\top$.
3. Projections of the training points: $z = \tilde K\,\alpha_k/\sqrt{\lambda_k}$.

With an RBF kernel $k(x,x') = e^{-\gamma\|x-x'\|^2}$, concentric circles become linearly separable - but look at *which*
component does it: for points spread evenly around rings, the top two kernel components form a sine/cosine-like pair that
encodes the **angle**, and the **third** encodes the **radius**. The demo shows a linear classifier scoring ~50% on each of
the first two and 100% on the third alone. Lesson: "largest eigenvalue" means most variance in feature space, not
"most useful for your task" - always check several components.

## t-SNE
1. In high-D, define neighbour probabilities with a Gaussian whose width $\sigma_i$ is tuned so each point has an
   effective number of neighbours = **perplexity**: $p_{j|i} \propto e^{-\|x_i-x_j\|^2/2\sigma_i^2}$, symmetrised $p_{ij}$.
2. In 2-D, use a heavy-tailed Student-t: $q_{ij} \propto (1+\|y_i-y_j\|^2)^{-1}$ - it lets moderately distant points spread out (solves the "crowding problem").
3. Minimise $\mathrm{KL}(P\|Q) = \sum p_{ij}\log\frac{p_{ij}}{q_{ij}}$ by gradient descent:
$$
\frac{\partial\,\mathrm{KL}}{\partial y_i} = 4\sum_j (p_{ij}-q_{ij})(y_i-y_j)(1+\|y_i-y_j\|^2)^{-1}
$$
Read t-SNE plots carefully: **local** neighbourhoods are meaningful; cluster sizes, gaps between clusters and global
layout are not. Different perplexities and seeds give different pictures. (UMAP is a faster modern alternative.)

## Exercises
1. Run the demo: compare PCA vs LDA projections of iris - which separates versicolor and virginica better?
2. Try kernel PCA with γ = 0.1, 1, 10 on the circles. What happens for very large γ?
3. Run t-SNE with perplexity 5 and 50 on the digits. How do the pictures differ?
