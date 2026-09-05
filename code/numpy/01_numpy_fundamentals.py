# ============================================================
# NUMPY FUNDAMENTALS FOR ML
# The subset of NumPy that appears in real ML code daily.
# Run: python 01_numpy_fundamentals.py
# Requires: numpy
# ============================================================
import numpy as np

# ------------------------------------------------------------
# 1. Creation + shape/dtype
# ------------------------------------------------------------
X = np.random.default_rng(0).normal(size=(100, 4))   # 100 samples x 4 features
print("Shape:", X.shape, "dtype:", X.dtype, "ndim:", X.ndim)

# ------------------------------------------------------------
# 2. Broadcasting — align shapes by stretching trailing dims
# ------------------------------------------------------------
mean = X.mean(axis=0)              # shape (4,)
Xc = X - mean                      # (100,4) - (4,) -> broadcasts
print("Column means after centering ~0:", np.allclose(Xc.mean(axis=0), 0))

# ------------------------------------------------------------
# 3. Vectorization — never loop in Python over arrays
# ------------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))   # clip prevents overflow


z = np.linspace(-10, 10, 1_000_000)
t0 = np.random.default_rng(1)
# Python loop vs vectorized (comparison only — do NOT loop in real code)
import time
zs = np.linspace(-10, 10, 100_000)
t = time.perf_counter()
slow = [1 / (1 + np.exp(-v)) for v in zs]             # list comprehension
t_list = time.perf_counter() - t
t = time.perf_counter()
fast = sigmoid(zs)                                     # vectorized
t_vec = time.perf_counter() - t
print(f"sigmoid: list={t_list*1000:.1f}ms  numpy={t_vec*1000:.2f}ms  "
      f"({t_list/t_vec:.0f}x faster)")

# ------------------------------------------------------------
# 4. Aggregations along axes
# ------------------------------------------------------------
print("Mean per feature:", np.round(X.mean(axis=0), 3))
print("Std  per feature:", np.round(X.std(axis=0), 3))

# ------------------------------------------------------------
# 5. Boolean masking / fancy indexing
# ------------------------------------------------------------
mask = X[:, 0] > 0                 # keep rows where feature 0 positive
print("Rows kept by mask:", mask.sum(), "/", len(X))
idx = np.argsort(X[:, 1])          # sort rows by feature 1
print("First row index after sort-by-f1:", idx[0])

# ------------------------------------------------------------
# 6. Linear algebra: y = X w + noise, solve with normal equations
# ------------------------------------------------------------
w_true = np.array([1.5, -2.0, 0.5, 3.0])
y = X @ w_true + np.random.default_rng(2).normal(0, 0.1, len(X))
w_hat = np.linalg.solve(X.T @ X, X.T @ y)            # (X^T X)^{-1} X^T y
print("Recovered weights:", np.round(w_hat, 3))

# ------------------------------------------------------------
# 7. Where NumPy math is used
#    matrix products (np.matmul / @), norms, eigendecomposition
#    and SVD — e.g., PCA in one line:
# ------------------------------------------------------------
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
X_pca2 = Xc @ Vt[:2].T
print("PCA to 2 dims:", X_pca2.shape)

# ------------------------------------------------------------
# 8. Random API (recommended Generator interface)
# ------------------------------------------------------------
rng = np.random.default_rng(42)    # seed once, reuse for reproducibility
sample = rng.choice(len(X), size=32, replace=False)
print("Train batch of indices:", sample[:5], "...")

# ------------------------------------------------------------
# Golden rules for ML code:
#  1. Never write a Python for-loop over array elements.
#  2. Use axis= arguments, not nested loops.
#  3. Use float32/16 for big arrays when precision allows.
#  4. Seed one Generator and pass it around.
# ============================================================
