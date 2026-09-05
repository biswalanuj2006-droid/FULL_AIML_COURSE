"""
Module 05: Mathematics for ML — Code Examples
================================================
Demonstrates math concepts with Python/NumPy implementations.

Usage:
    python math_examples.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

print("=" * 60)
print("MATHEMATICS FOR ML — CODE EXAMPLES")
print("=" * 60)

# =============================================================================
# 1. ALGEBRA — Linear Regression from Scratch
# =============================================================================

print("\n1. ALGEBRA: Linear Regression")
print("-" * 40)

# Data: y = 2x + 1 + noise
np.random.seed(42)
X = np.linspace(0, 10, 50)
y = 2 * X + 1 + np.random.randn(50) * 1.5

# Method 1: Normal equation (algebra)
X_b = np.c_[np.ones(len(X)), X]  # Add bias column
w_normal = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(f"Normal equation: y = {w_normal[1]:.3f}x + {w_normal[0]:.3f}")

# Method 2: Gradient descent (optimization)
w_gd = np.random.randn(2) * 0.1
bias = 0.01
lr = 0.01
for epoch in range(1000):
    y_pred = X_b @ w_gd
    loss = np.mean((y_pred - y) ** 2)
    grad = (2 / len(X)) * X_b.T @ (y_pred - y)
    w_gd -= lr * grad

print(f"Gradient descent: y = {w_gd[1]:.3f}x + {w_gd[0]:.3f}")
print(f"True:             y = 2.000x + 1.000")

# =============================================================================
# 2. LINEAR ALGEBRA — Dot Product and Matrix Operations
# =============================================================================

print("\n\n2. LINEAR ALGEBRA")
print("-" * 40)

# Dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"Dot product: {np.dot(a, b)}")

# Matrix multiplication
W = np.array([[0.5, 0.3, 0.2],
              [0.1, 0.7, 0.2]])
x = np.array([100, 50, 80])
z = W @ x
print(f"Linear combination (W @ x): {z}")

# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([1, 1, 0])
print(f"cos([1,0,0], [0,1,0]) = {cosine_similarity(v1, v2):.3f}")
print(f"cos([1,0,0], [1,1,0]) = {cosine_similarity(v1, v3):.3f}")

# Eigendecomposition
A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")

# SVD
U, S, Vt = np.linalg.svd(np.random.randn(3, 3))
print(f"Singular values: {S.round(3)}")

# =============================================================================
# 3. CALCULUS — Derivatives and Gradients
# =============================================================================

print("\n\n3. CALCULUS")
print("-" * 40)

# Numerical derivative
def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

# Example functions
f1 = lambda x: x**2
f2 = lambda x: np.sin(x)
f3 = lambda x: 1 / (1 + np.exp(-x))  # sigmoid

print(f"d/dx(x²) at x=3: {numerical_derivative(f1, 3):.4f} (exact: 6)")
print(f"d/dx(sin(x)) at x=π/4: {numerical_derivative(f2, np.pi/4):.4f} (exact: 0.7071)")
print(f"d/dx(sigmoid) at x=0: {numerical_derivative(f3, 0):.4f} (exact: 0.25)")

# Gradient of loss function
def compute_gradient(X, y, w):
    """Gradient of MSE loss."""
    n = len(y)
    predictions = X @ w
    return (2/n) * X.T @ (predictions - y)

# Gradient descent visualization
np.random.seed(42)
X_gd = np.random.randn(100, 2)
true_w = np.array([3.0, -2.0])
y_gd = X_gd @ true_w + np.random.randn(100) * 0.5

w = np.zeros(2)
lr = 0.01
trajectory = [w.copy()]

for _ in range(100):
    grad = compute_gradient(X_gd, y_gd, w)
    w -= lr * grad
    trajectory.append(w.copy())

trajectory = np.array(trajectory)
print(f"\nGradient descent convergence:")
print(f"  True weights: {true_w}")
print(f"  Learned weights: {w.round(4)}")
print(f"  Error: {np.abs(w - true_w).max():.6f}")

# =============================================================================
# 4. PROBABILITY — Bayes Theorem
# =============================================================================

print("\n\n4. PROBABILITY")
print("-" * 40)

# Bayes theorem: Medical test
p_disease = 0.01           # Prior
p_pos_given_disease = 0.95  # Sensitivity
p_pos_given_no_disease = 0.05  # False positive rate

# Total probability of positive
p_pos = p_pos_given_disease * p_disease + \
        p_pos_given_no_disease * (1 - p_disease)

# Posterior
p_disease_given_pos = (p_pos_given_disease * p_disease) / p_pos

print(f"P(disease) = {p_disease}")
print(f"P(positive|disease) = {p_pos_given_disease}")
print(f"P(positive|no disease) = {p_pos_given_no_disease}")
print(f"P(disease|positive) = {p_disease_given_pos:.4f}")

# Gaussian distribution
from scipy import stats

x = np.linspace(-4, 4, 100)
gaussian = stats.norm.pdf(x, loc=0, scale=1)
print(f"\nGaussian: mean=0, std=1")
print(f"P(-1 < X < 1) = {stats.norm.cdf(1) - stats.norm.cdf(-1):.4f}")
print(f"P(-2 < X < 2) = {stats.norm.cdf(2) - stats.norm.cdf(-2):.4f}")

# Maximum Likelihood Estimation
data = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1])  # Bernoulli
p_mle = data.mean()
print(f"\nMLE for coin bias: {p_mle} (true: 0.7)")

# =============================================================================
# 5. STATISTICS — Descriptive and Inferential
# =============================================================================

print("\n\n5. STATISTICS")
print("-" * 40)

data = np.array([14, 18, 20, 22, 25, 30, 35, 40, 50, 65])

print(f"Mean: {np.mean(data):.2f}")
print(f"Median: {np.median(data):.2f}")
print(f"Std: {np.std(data, ddof=1):.2f}")
print(f"Q1: {np.percentile(data, 25):.2f}")
print(f"Q3: {np.percentile(data, 75):.2f}")

# Correlation
X_corr = np.random.randn(200, 3)
X_corr[:, 1] = 0.8 * X_corr[:, 0] + 0.2 * np.random.randn(200)
corr = np.corrcoef(X_corr.T)
print(f"\nCorrelation matrix:\n{corr.round(3)}")

# Confidence interval
sample_mean = np.mean(data)
sample_std = np.std(data, ddof=1)
n = len(data)
se = sample_std / np.sqrt(n)
ci_lower = sample_mean - 1.96 * se
ci_upper = sample_mean + 1.96 * se
print(f"\n95% CI for mean: [{ci_lower:.2f}, {ci_upper:.2f}]")

# Outlier detection (IQR)
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
outliers = data[(data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)]
print(f"Outliers (IQR method): {outliers}")

# =============================================================================
# 6. OPTIMIZATION — Gradient Descent Variants
# =============================================================================

print("\n\n6. OPTIMIZATION")
print("-" * 40)

def rosenbrock(x, y):
    """Non-convex test function."""
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(x, y):
    dx = -2*(1-x) - 400*x*(y - x**2)
    dy = 200*(y - x**2)
    return np.array([dx, dy])

# Batch Gradient Descent
w = np.array([-1.0, 1.0])  # Start point
lr = 0.001
history_bgd = [w.copy()]

for _ in range(5000):
    grad = rosenbrock_grad(w[0], w[1])
    w -= lr * grad
    history_bgd.append(w.copy())

print(f"BGD final: {w.round(4)} (optimal: [1, 1])")

# SGD with noise
w = np.array([-1.0, 1.0])
lr = 0.001
history_sgd = [w.copy()]

for _ in range(5000):
    grad = rosenbrock_grad(w[0], w[1])
    noise = np.random.randn(2) * 0.1
    w -= lr * (grad + noise)
    history_sgd.append(w.copy())

print(f"SGD final: {w.round(4)}")

# Adam optimizer
def adam_optimizer(grad_fn, start, lr=0.001, beta1=0.9, beta2=0.999,
                   eps=1e-8, steps=5000):
    w = start.copy()
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    history = [w.copy()]

    for t in range(1, steps + 1):
        g = grad_fn(w[0], w[1])
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        w -= lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(w.copy())

    return w, history

w_adam, _ = adam_optimizer(rosenbrock_grad, np.array([-1.0, 1.0]))
print(f"Adam final: {w_adam.round(4)}")

# Learning rate experiment
print("\nLearning rate experiment on L(w) = (w-3)²:")
for lr in [0.01, 0.1, 0.5, 1.1]:
    w = 0.0
    for step in range(20):
        grad = 2 * (w - 3)
        w -= lr * grad
    status = "converged" if abs(w - 3) < 0.1 else "diverged"
    print(f"  lr={lr}: w={w:.4f} ({status})")

# =============================================================================
# 7. SIGMOID AND SOFTMAX
# =============================================================================

print("\n\n7. ACTIVATION FUNCTIONS")
print("-" * 40)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    e_z = np.exp(z - np.max(z))  # Numerical stability
    return e_z / e_z.sum()

z = np.array([-2, -1, 0, 1, 2])
print(f"Input: {z}")
print(f"Sigmoid: {sigmoid(z).round(3)}")
print(f"Sum of sigmoid: {sigmoid(z).sum():.3f}")

logits = np.array([2.0, 1.0, 0.5])
probs = softmax(logits)
print(f"\nLogits: {logits}")
print(f"Softmax: {probs.round(3)}")
print(f"Sum: {probs.sum():.3f}")

# =============================================================================
# 8. LOSS FUNCTIONS
# =============================================================================

print("\n\n8. LOSS FUNCTIONS")
print("-" * 40)

y_true = np.array([1, 0, 1, 1, 0])
y_prob = np.array([0.9, 0.1, 0.8, 0.7, 0.2])

# MSE
mse = np.mean((y_true - y_prob)**2)
print(f"MSE: {mse:.4f}")

# Binary Cross-Entropy
eps = 1e-15
bce = -np.mean(y_true * np.log(y_prob + eps) +
               (1 - y_true) * np.log(1 - y_prob + eps))
print(f"BCE: {bce:.4f}")

# Hinge loss
y_signed = 2 * y_true - 1  # Convert 0/1 to -1/+1
hinge = np.mean(np.maximum(0, 1 - y_signed * y_prob))
print(f"Hinge: {hinge:.4f}")

print("\n" + "=" * 60)
print("ALL MATH EXAMPLES COMPLETED!")
print("=" * 60)
