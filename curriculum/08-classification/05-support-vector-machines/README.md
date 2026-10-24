---
title: Support Vector Machines
order: 5
summary: The maximum-margin classifier - geometric margin, hard and soft margins, hinge loss, the Lagrangian dual, the kernel trick, and the SMO algorithm that solves it.
level: advanced
minutes: 70
why: SVMs give strong, well-regularised classifiers from a clean optimisation principle - maximise the margin - and the kernel trick lets a linear method learn non-linear boundaries. They dominated ML before deep learning and remain excellent on small and medium datasets.
background: Vapnik and Chervonenkis developed the theory (1960s-70s); Boser, Guyon and Vapnik introduced kernels (1992); Cortes and Vapnik added the soft margin (1995); Platt's SMO (1998) made training practical.
use_cases:
  - Text and image classification on small/medium data
  - Bioinformatics (protein classification)
  - Handwriting recognition (historically)
  - Any problem where a clean margin-based model with kernels fits
where_used:
  - 05-digit-recognizer
prerequisites:
  - 08-classification/01-logistic-regression
  - 00-math-foundations/02-calculus-gradients
code:
  - mlscratch/svm.py:LinearSVC
  - mlscratch/svm.py:kernel_matrix
  - mlscratch/svm.py:SVC._smo
demo: demo.py
widgets:
  - svm
pros:
  - Maximum margin → good generalisation, convex problem with a unique solution
  - Kernels give non-linear boundaries; only support vectors matter
cons:
  - Kernel SVM training is O(n²)-O(n³) - slow beyond ~50k samples
  - Needs feature scaling and tuning of C and γ; no native probabilities
workflow: |
  flowchart LR
    A[scaled X, y ∈ {-1,+1}] --> B{linear?}
    B -->|yes| C[primal: hinge loss + ½w² by subgradient descent]
    B -->|no| D[kernel K xᵢ,xⱼ]
    D --> E[dual QP solved by SMO]
    E --> F[support vectors αᵢ > 0]
    C & F --> G[f x = Σαᵢyᵢk xᵢ,x + b]
    G --> H[sign f x]
---

## 1. The margin
For labels $y_i\in\{-1, +1\}$ and a hyperplane $w^\top x + b = 0$, the distance of point $x_i$ to the plane is
$\frac{|w^\top x_i + b|}{\|w\|}$. Scale $w, b$ so the closest points satisfy $y_i(w^\top x_i + b) = 1$; the **margin**
(gap between the classes) is then $\frac{2}{\|w\|}$. Maximising it = minimising $\|w\|$.

## 2. Hard margin (separable data)
$$
\min_{w,b}\ \tfrac12\|w\|^2 \quad\text{s.t.}\quad y_i(w^\top x_i + b) \ge 1\ \ \forall i
$$
## 3. Soft margin (real data)
Allow violations with slack $\xi_i \ge 0$ and pay $C$ per unit:
$$
\min_{w,b,\xi}\ \tfrac12\|w\|^2 + C\sum_i \xi_i \quad\text{s.t.}\quad y_i(w^\top x_i + b) \ge 1 - \xi_i
$$
At the optimum $\xi_i = \max(0, 1 - y_if(x_i))$, giving the unconstrained **hinge-loss** form
$$
\min_w\ \tfrac12\|w\|^2 + C\sum_i \max\big(0,\ 1 - y_i(w^\top x_i + b)\big)
$$
Points beyond the margin cost nothing - only points on or inside the margin influence the solution. Small $C$ → wide
margin, more violations (high bias); large $C$ → narrow margin, fewer violations (high variance).
`LinearSVC` minimises this with subgradient descent: for violators the subgradient is $w - C\,y_ix_i$.

Compare losses as a function of the margin $m = y\,f(x)$: hinge $\max(0, 1-m)$, logistic $\log(1+e^{-m})$, 0-1 loss $[m < 0]$.
Both hinge and logistic are convex surrogates for the 0-1 loss; hinge is exactly 0 for confident correct points.

## 4. The dual
Introduce multipliers $\alpha_i \ge 0$ for the constraints, set the derivatives of the Lagrangian to zero
($w = \sum_i \alpha_i y_i x_i$, $\sum_i\alpha_iy_i = 0$) and substitute back:
$$
\max_\alpha\ \sum_i\alpha_i - \tfrac12\sum_{i,j}\alpha_i\alpha_jy_iy_j\,x_i^\top x_j \quad\text{s.t.}\quad 0\le\alpha_i\le C,\ \ \sum_i\alpha_iy_i = 0
$$
KKT conditions: $\alpha_i = 0$ for points outside the margin, $0<\alpha_i<C$ on the margin, $\alpha_i = C$ inside it.
Only the **support vectors** ($\alpha_i > 0$) appear in $f(x) = \sum_i\alpha_iy_i\,x_i^\top x + b$.

## 5. The kernel trick
The data appears in the dual **only through dot products** $x_i^\top x_j$. Replace them by a kernel
$k(x_i, x_j) = \phi(x_i)^\top\phi(x_j)$ - an inner product in a high-dimensional feature space we never compute:

| Kernel | $k(x, x')$ | Feature space |
|--------|------------|---------------|
| linear | $x^\top x'$ | original |
| polynomial | $(\gamma x^\top x' + r)^d$ | all monomials up to degree d |
| **RBF / Gaussian** | $\exp(-\gamma\|x - x'\|^2)$ | infinite-dimensional |

RBF's γ sets the reach of each support vector: large γ → very local, wiggly boundary (overfit); small γ → smooth.

## 6. SMO: solving the dual two variables at a time
The constraint $\sum\alpha_iy_i = 0$ means you can't change one $\alpha$ alone. SMO picks a pair $(\alpha_i, \alpha_j)$
that violates the KKT conditions, and solves the 2-variable problem **analytically**:
1. errors $E_k = f(x_k) - y_k$; curvature $\eta = 2K_{ij} - K_{ii} - K_{jj}$
2. unconstrained optimum $\alpha_j^{new} = \alpha_j - y_j(E_i - E_j)/\eta$, clipped to the box $[L, H]$ implied by $0\le\alpha\le C$ and the equality constraint
3. $\alpha_i^{new} = \alpha_i + y_iy_j(\alpha_j - \alpha_j^{new})$; update the bias $b$ from whichever multiplier is strictly inside $(0, C)$

Repeat until no pair changes. `SVC._smo` is this algorithm in ~40 lines; multi-class uses one-vs-rest.

## Exercises
1. In the widget (degree 1), vary C on overlapping blobs and watch the boundary tilt.
2. Run the demo; how many support vectors does the RBF SVM use, and how does that change with C and γ?
3. Verify the KKT conditions on the trained model: $\alpha_i = 0 \Rightarrow y_if(x_i) \ge 1$.
