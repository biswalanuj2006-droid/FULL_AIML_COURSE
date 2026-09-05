"""
Module 02: NumPy — Code Examples
==================================
Run these examples to see NumPy in action.

Usage:
    python numpy_examples.py
"""

import numpy as np

print("=" * 60)
print("NUMPY CODE EXAMPLES FOR ML")
print("=" * 60)

# =============================================================================
# 1. ARRAY CREATION
# =============================================================================

print("\n1. ARRAY CREATION")
print("-" * 40)

# From lists
a = np.array([1, 2, 3, 4, 5])
print(f"1D array: {a}")
print(f"Shape: {a.shape}, Dtype: {a.dtype}")

# 2D array
b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(f"\n2D array:\n{b}")
print(f"Shape: {b.shape}")

# Factory functions
print(f"\nZeros (2x3):\n{np.zeros((2, 3))}")
print(f"\nOnes (2x3):\n{np.ones((2, 3))}")
print(f"\nEye (4):\n{np.eye(4)}")
print(f"\nArange: {np.arange(0, 10, 2)}")
print(f"Linspace: {np.linspace(0, 1, 5)}")

# Random
rng = np.random.default_rng(42)
print(f"\nRandom (3x3):\n{rng.random((3, 3)).round(3)}")
print(f"Normal (5): {rng.standard_normal(5).round(3)}")
print(f"Integers: {rng.integers(0, 10, size=5)}")

# =============================================================================
# 2. INDEXING AND SLICING
# =============================================================================

print("\n\n2. INDEXING AND SLICING")
print("-" * 40)

a = np.array([10, 20, 30, 40, 50])
print(f"Array: {a}")
print(f"a[0] = {a[0]}")
print(f"a[-1] = {a[-1]}")
print(f"a[1:4] = {a[1:4]}")
print(f"a[::2] = {a[::2]}")

# 2D
b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print(f"\n2D Array:\n{b}")
print(f"b[0, 0] = {b[0, 0]}")
print(f"b[:, 1] = {b[:, 1]}")      # All rows, column 1
print(f"b[0:2, :] =\n{b[0:2, :]}")

# Boolean masking
data = np.array([1, 5, 3, 8, 2, 9, 4])
mask = data > 4
print(f"\nData: {data}")
print(f"Mask (data > 4): {mask}")
print(f"Filtered: {data[mask]}")
print(f"Count > 4: {(data > 4).sum()}")

# Combined masks
filtered = data[(data > 2) & (data < 8)]
print(f"2 < data < 8: {filtered}")

# =============================================================================
# 3. BROADCASTING
# =============================================================================

print("\n\n3. BROADCASTING")
print("-" * 40)

# Feature standardization
X = np.array([[1, 100, 0.5],
              [2, 200, 0.8],
              [3, 300, 0.3],
              [4, 400, 0.6]])

mean = X.mean(axis=0)
std = X.std(axis=0)
X_std = (X - mean) / std

print("Original data:")
print(X)
print(f"\nMeans: {mean.round(2)}")
print(f"Stds: {std.round(2)}")
print(f"\nStandardized:\n{X_std.round(3)}")

# Add bias
bias = np.array([10, 20, 30])
result = X[:, :3] + bias    # Broadcasting
print(f"\nX[:, :3] + bias:\n{result}")

# =============================================================================
# 4. LINEAR ALGEBRA
# =============================================================================

print("\n\n4. LINEAR ALGEBRA")
print("-" * 40)

# Dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"Dot product: {a @ b}")

# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B
print(f"\nA @ B =\n{C}")

# Transpose
print(f"\nA.T =\n{A.T}")

# Inverse
A_inv = np.linalg.inv(A)
print(f"\nA_inv =\n{A_inv.round(4)}")
print(f"A @ A_inv ≈ I:\n{(A @ A_inv).round(10)}")

# Determinant
print(f"\ndet(A) = {np.linalg.det(A):.1f}")

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\nEigenvalues: {eigenvalues.round(4)}")
print(f"Eigenvectors:\n{eigenvectors.round(4)}")

# SVD
U, S, Vt = np.linalg.svd(A)
print(f"\nSVD of A:")
print(f"U:\n{U.round(4)}")
print(f"Singular values: {S.round(4)}")
print(f"Vt:\n{Vt.round(4)}")

# Solve linear system: Ax = b
b = np.array([5, 7])
x = np.linalg.solve(A, b)
print(f"\nSolve Ax = b: x = {x}")
print(f"Verify: A @ x = {A @ x}")

# =============================================================================
# 5. RANDOM NUMBERS
# =============================================================================

print("\n\n5. RANDOM NUMBERS")
print("-" * 40)

rng = np.random.default_rng(42)

# Reproducibility
print("Same seed → same results:")
rng1 = np.random.default_rng(42)
rng2 = np.random.default_rng(42)
print(f"  rng1: {rng1.random(3).round(3)}")
print(f"  rng2: {rng2.random(3).round(3)}")

# Distributions
print(f"\nUniform [0,1): {rng.uniform(0, 1, 5).round(3)}")
print(f"Normal (0,1): {rng.normal(0, 1, 5).round(3)}")
print(f"Binomial: {rng.binomial(1, 0.5, 10)}")

# Random sampling
indices = rng.choice(100, size=10, replace=False)
print(f"\nRandom 10 from 0-99: {sorted(indices)}")

# Simple train/test split
n = 100
X = rng.standard_normal((n, 5))
y = rng.integers(0, 2, n)
indices = rng.permutation(n)
split = int(n * 0.8)
X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# =============================================================================
# 6. STATISTICS
# =============================================================================

print("\n\n6. STATISTICS")
print("-" * 40)

data = rng.standard_normal(1000)
print(f"Mean: {data.mean():.4f}")
print(f"Median: {np.median(data):.4f}")
print(f"Std: {data.std():.4f}")
print(f"Min: {data.min():.4f}, Max: {data.max():.4f}")
print(f"Q1: {np.percentile(data, 25):.4f}")
print(f"Q3: {np.percentile(data, 75):.4f}")
print(f"IQR: {np.percentile(data, 75) - np.percentile(data, 25):.4f}")

# Axis-based statistics
X = rng.standard_normal((100, 5))
print(f"\nColumn means: {X.mean(axis=0).round(3)}")
print(f"Row means: {X.mean(axis=1)[:5].round(3)}...")

# Correlation
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)
corr = np.corrcoef(x, y)[0, 1]
print(f"\nCorrelation between x and y: {corr:.4f}")

# =============================================================================
# 7. PRACTICAL ML EXAMPLES
# =============================================================================

print("\n\n7. PRACTICAL ML EXAMPLES")
print("-" * 40)

# Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

point_a = np.array([1, 2, 3])
point_b = np.array([4, 5, 6])
print(f"Distance: {euclidean_distance(point_a, point_b):.4f}")

# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"Cosine similarity: {cosine_similarity(point_a, point_b):.4f}")

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.array([-2, -1, 0, 1, 2])
print(f"Sigmoid: {sigmoid(z).round(3)}")

# Softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))    # Subtract max for numerical stability
    return e_x / e_x.sum()

logits = np.array([2.0, 1.0, 0.5])
print(f"Softmax: {softmax(logits).round(3)}")

# MSE loss
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])
print(f"MSE: {mse(y_true, y_pred):.4f}")

# MAE loss
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

print(f"MAE: {mae(y_true, y_pred):.4f}")

# Binary cross-entropy
def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])
print(f"BCE: {binary_cross_entropy(y_true, y_pred):.4f}")

# One-hot encoding
def one_hot(labels, num_classes):
    n = len(labels)
    result = np.zeros((n, num_classes))
    result[np.arange(n), labels] = 1
    return result

labels = np.array([0, 1, 2, 1, 0])
print(f"\nLabels: {labels}")
print(f"One-hot:\n{one_hot(labels, 3)}")

print("\n" + "=" * 60)
print("ALL NUMPY EXAMPLES COMPLETED!")
print("=" * 60)
