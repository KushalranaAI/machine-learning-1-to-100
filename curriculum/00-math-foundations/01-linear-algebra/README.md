---
title: Linear Algebra for ML
order: 1
summary: Vectors, matrices, dot products, norms, matrix multiplication, inverses, eigenvalues and the SVD - the language every dataset and model is written in.
level: beginner
minutes: 50
why: A dataset is a matrix, a linear model is a matrix-vector product, PCA is an eigen-decomposition and a neural network is a stack of matrix multiplications. If you can read $Xw$, $X^\top X$ and $U\Sigma V^\top$, you can read almost every ML paper.
background: Matrices were formalised by Cayley and Sylvester in the 1850s. Least squares (Gauss, Legendre, ~1805) was the first great application to data; modern ML runs on the same operations, now accelerated by GPUs.
use_cases:
  - Representing tabular data (rows = samples, columns = features)
  - Solving linear regression in closed form
  - Dimensionality reduction (PCA, SVD, LSA for text)
  - Recommendation systems (matrix factorisation)
  - Every layer of a neural network
where_used:
  - 01-house-prices
  - 03-customer-segmentation
  - 05-digit-recognizer
code:
  - mlscratch/linear_model.py:LinearRegression
  - mlscratch/decomposition.py:PCA
  - mlscratch/neighbors.py:pairwise_distances
demo: demo.py
widgets:
  - matrix-transform
workflow: |
  flowchart LR
    A[Raw table] --> B[Matrix X n×d]
    B --> C[Linear map Xw]
    B --> D[Covariance XᵀX/n]
    D --> E[Eigenvectors → PCA]
    C --> F[Predictions ŷ]
---

## Vectors: one sample, one point

A sample with $d$ features is a vector $x \in \mathbb{R}^d$, e.g. a house $x = [\text{size}, \text{rooms}, \text{age}]$.
A dataset of $n$ samples stacks them as rows of a **design matrix**:

$$
X = \begin{bmatrix} x_1^\top \\ \vdots \\ x_n^\top \end{bmatrix} \in \mathbb{R}^{n \times d}
$$

### Dot product, length and angle
$$
a \cdot b = a^\top b = \sum_i a_i b_i = \|a\|\,\|b\|\cos\theta
$$
- $\|a\|_2 = \sqrt{a^\top a}$ (Euclidean length), $\|a\|_1 = \sum|a_i|$ (Manhattan).
- $a^\top b = 0$ means **orthogonal** (unrelated directions).
- **Cosine similarity** $\frac{a^\top b}{\|a\|\|b\|}$ compares directions - used for text and embeddings.

These norms reappear as **regularisers**: Ridge penalises $\|w\|_2^2$, Lasso penalises $\|w\|_1$.

## Matrices: transformations

$Ax$ is a linear transformation: it rotates, stretches, shears or projects space (try the interactive tab).
The columns of $A$ are where the basis vectors $e_1, e_2$ land.

**Matrix product** $(AB)_{ij} = \sum_k A_{ik}B_{kj}$ - composing transformations. Two facts worth memorising:
- $(AB)^\top = B^\top A^\top$
- In general $AB \ne BA$.

**A linear model in one line:** $\hat y = Xw + b$ computes all $n$ predictions at once - a matrix-vector product.

## Determinant, inverse, rank
- $\det(A)$ = how much $A$ scales area/volume. $\det(A) = 0$ ⇒ space is squashed into a lower dimension ⇒ **no inverse**.
- $A^{-1}$ undoes $A$: $A^{-1}A = I$.
- **Rank** = number of independent columns. If features are perfectly collinear (e.g. price in USD *and* in EUR),
  $X^\top X$ is singular and the normal equation breaks - one reason we use `lstsq`/SVD or Ridge.

## Solving linear systems and least squares
Linear regression asks for $w$ minimising $\|Xw - y\|^2$. Setting the gradient to zero:

$$
\nabla_w \|Xw-y\|^2 = 2X^\top(Xw - y) = 0 \;\Rightarrow\; X^\top X\, w = X^\top y \quad\text{(normal equation)}
$$

Geometrically, $X\hat w$ is the **orthogonal projection** of $y$ onto the column space of $X$ - the residual
$y - X\hat w$ is perpendicular to every feature column.

> In code, never compute `inv(X.T @ X)`. Use `np.linalg.solve` or `np.linalg.lstsq` - faster and numerically stable.

## Eigenvalues and eigenvectors
$$
A v = \lambda v
$$
$v$ is a direction that $A$ only stretches (by $\lambda$). For a **symmetric** matrix (like a covariance matrix):
- all eigenvalues are real, eigenvectors are orthogonal,
- $A = Q \Lambda Q^\top$ (spectral decomposition).

**PCA** (Part 5) = eigenvectors of the covariance matrix $\Sigma = \frac{1}{n-1}X_c^\top X_c$: the directions of greatest variance.

## Singular value decomposition (SVD)
Every matrix (any shape) factors as
$$
X = U \Sigma V^\top
$$
with orthonormal $U, V$ and non-negative singular values on the diagonal of $\Sigma$. Uses in this course:
- PCA: principal axes = rows of $V^\top$, variances = $\sigma_i^2/(n-1)$
- Least squares and pseudo-inverse: $X^+ = V\Sigma^{+}U^\top$
- Truncated SVD / LSA for text, low-rank compression, recommender systems

## Vectorisation: think in matrices
Loops over samples in Python are slow; one matrix operation is fast (it runs in optimised C/BLAS). Example from
`mlscratch/neighbors.py` - all pairwise distances with one matrix product:

$$
\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2\,a^\top b
$$

```python
sq = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
```

## Exercises
1. In the interactive tab, find a matrix with determinant 0. What happens to the circle?
2. Show that if $Av = \lambda v$ then $A^2 v = \lambda^2 v$. What does that mean for repeated transformations?
3. Run the demo and verify that the residual of least squares is orthogonal to every column of $X$.
