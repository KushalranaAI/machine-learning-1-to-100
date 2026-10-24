"""Linear algebra you will use constantly: products, solving least squares, eigen-decomposition and SVD."""

import time

import numpy as np

rng = np.random.default_rng(0)

# --- a dataset is a matrix ----------------------------------------------------------------
X = rng.normal(size=(100, 3))                 # 100 samples, 3 features
w_true = np.array([2.0, -1.0, 0.5])
y = X @ w_true + 0.1 * rng.normal(size=100)   # all 100 predictions in one matrix-vector product
print("X shape:", X.shape, " y shape:", y.shape)

# --- least squares three ways ----------------------------------------------------------------
w_normal = np.linalg.solve(X.T @ X, X.T @ y)        # normal equation XᵀX w = Xᵀy
w_lstsq, *_ = np.linalg.lstsq(X, y, rcond=None)      # SVD-based (preferred)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
w_svd = Vt.T @ np.diag(1 / S) @ U.T @ y              # pseudo-inverse X⁺ = V Σ⁻¹ Uᵀ
print("\nrecovered weights (true = [2, -1, 0.5]):")
for name, w in [("normal equation", w_normal), ("lstsq", w_lstsq), ("SVD pseudo-inverse", w_svd)]:
    print(f"  {name:<20} {np.round(w, 4)}")

residual = y - X @ w_lstsq
print("\nXᵀ·residual (should be ~0: residual ⟂ every feature column):", np.round(X.T @ residual, 10))

# --- eigen-decomposition of a covariance matrix -------------------------------------------
Xc = X - X.mean(0)
cov = Xc.T @ Xc / (len(X) - 1)
vals, vecs = np.linalg.eigh(cov)
print("\ncovariance eigenvalues:", np.round(vals, 4))
print("eigenvectors orthonormal? QᵀQ = I:", np.allclose(vecs.T @ vecs, np.eye(3)))
print("reconstruct Σ = QΛQᵀ:", np.allclose(vecs @ np.diag(vals) @ vecs.T, cov))
print("SVD link: σ²/(n-1) =", np.round(np.sort(S**2 / (len(X) - 1)), 4), "(≈ eigenvalues of uncentred XᵀX/(n-1))")

# --- collinearity breaks the normal equation ----------------------------------------------
X_bad = np.column_stack([X[:, 0], 2 * X[:, 0]])     # second column = 2 × first
print("\nrank of collinear matrix:", np.linalg.matrix_rank(X_bad), "of", X_bad.shape[1])
print("condition number of XᵀX:", f"{np.linalg.cond(X_bad.T @ X_bad):.2e}  (huge → unstable inverse)")

# --- vectorisation ---------------------------------------------------------------------------
A, B = rng.normal(size=(500, 20)), rng.normal(size=(400, 20))
t = time.perf_counter()
loop = np.array([[np.sqrt(((a - b) ** 2).sum()) for b in B] for a in A])
t_loop = time.perf_counter() - t
t = time.perf_counter()
vec = np.sqrt(np.maximum((A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T, 0))
t_vec = time.perf_counter() - t
print(f"\npairwise distances: loops {t_loop:.3f}s vs vectorised {t_vec:.4f}s "
      f"({t_loop / t_vec:.0f}× faster), same result: {np.allclose(loop, vec)}")
