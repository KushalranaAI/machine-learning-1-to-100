---
title: Principal Component Analysis (PCA)
order: 3
summary: Find the orthogonal directions of maximum variance via the covariance eigen-decomposition / SVD - derivation, explained variance, reconstruction, whitening and pitfalls.
level: intermediate
minutes: 50
why: PCA compresses correlated features into a few uncorrelated components, speeds up models, removes noise, and lets you visualise high-dimensional data in 2-D. It is also the clearest example of how linear algebra turns into an ML algorithm.
background: Karl Pearson (1901) and Harold Hotelling (1933) invented PCA. It is the same mathematics as the SVD and the Karhunen-Loève transform in signal processing.
use_cases:
  - Visualising high-dimensional data (digits, customers, genes) in 2-D
  - Compressing images or sensor data
  - Removing multicollinearity before regression (principal component regression)
  - Denoising (keep top components, drop the rest)
where_used:
  - 03-customer-segmentation
  - 05-digit-recognizer
prerequisites:
  - 00-math-foundations/01-linear-algebra
  - 04-feature-engineering/01-feature-scaling
code:
  - mlscratch/decomposition.py:PCA
demo: demo.py
widgets:
  - pca
pros:
  - Fast, deterministic, no hyper-parameters except k
  - Components are uncorrelated
cons:
  - Linear only - curved structure needs kernel PCA / t-SNE
  - Sensitive to feature scale - standardise first
  - Maximises variance, not class separation (that's LDA)
workflow: |
  flowchart LR
    A[X n×d] --> B[standardise]
    B --> C[centre: X - μ]
    C --> D[SVD: X = U S Vᵀ]
    D --> E[components = rows of Vᵀ]
    D --> F[explained variance = S²/n-1]
    E --> G[Z = X Vₖ  n×k]
    G --> H[reconstruct X̂ = Z Vₖᵀ + μ]
---

## The goal
Find a unit vector $v$ such that projecting the centred data onto it, $z = X_c v$, has **maximum variance**.

## Derivation
The variance of the projection is
$$
\frac{1}{n-1}\|X_c v\|^2 = v^\top \underbrace{\frac{X_c^\top X_c}{n-1}}_{\Sigma}\, v
$$
Maximise $v^\top\Sigma v$ subject to $v^\top v = 1$ with a Lagrange multiplier:
$$
\mathcal{L} = v^\top\Sigma v - \lambda(v^\top v - 1) \;\Rightarrow\; \nabla_v\mathcal L = 2\Sigma v - 2\lambda v = 0 \;\Rightarrow\; \Sigma v = \lambda v
$$
So $v$ must be an **eigenvector** of the covariance matrix, and the variance captured is $v^\top\Sigma v = \lambda$.
The best direction is the eigenvector with the **largest eigenvalue** (PC1); the next is the best direction
orthogonal to it (PC2), and so on.

## Computing it with the SVD (what the code does)
Decompose the centred data $X_c = U S V^\top$. Then
$$
\Sigma = \frac{X_c^\top X_c}{n-1} = V\,\frac{S^2}{n-1}\,V^\top
$$
so the principal axes are the columns of $V$ (rows of $V^\top$) and the variances are $s_i^2/(n-1)$. The SVD avoids
forming $X^\top X$, which squares the condition number - it is the numerically stable route.

## Choosing k
**Explained variance ratio** $\frac{\lambda_i}{\sum_j \lambda_j}$. Keep enough components for 90-99% of the variance
(`PCA(0.95)`), look for an "elbow" in the scree plot, or tune k by cross-validating the downstream model.

## Reconstruction
$\hat X = Z V_k^\top + \mu$. The reconstruction error equals the variance in the dropped components: $\sum_{i>k}\lambda_i$.
That's why PCA is the optimal linear compressor for squared error (Eckart-Young theorem).

## Whitening
Divide each component by $\sqrt{\lambda_i}$: the transformed data has identity covariance. Useful before some algorithms, but it amplifies noise in small components.

## Pitfalls
- **Scale first** - otherwise the feature with the largest units dominates PC1.
- Fit PCA on the **training** set only.
- Components are linear combinations - interpret loadings carefully.
- Maximum variance ≠ most predictive (a low-variance direction may separate the classes).

## Exercises
1. In the widget, make both spreads equal. Why does PC1 become arbitrary?
2. Run the demo; how many components keep 95% of the digit images' variance? Look at the reconstructions.
3. Fit PCA without scaling on the house data with prices in dollars - what does PC1 become?
