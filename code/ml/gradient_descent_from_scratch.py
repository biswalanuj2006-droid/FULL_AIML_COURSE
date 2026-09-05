# ============================================================
# GRADIENT DESCENT FROM SCRATCH
# Implements batch GD, SGD (one pass per epoch over shuffled
# samples), mini-batch GD, and momentum for linear regression,
# then compares them with the closed-form solution.
#
# Run: python gradient_descent_from_scratch.py
# Requires: numpy, matplotlib
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Data: y = 3.0*x + 1.5 + noise
# ------------------------------------------------------------
rng = np.random.default_rng(42)
n = 500
X = rng.uniform(-3, 3, n)
y = 3.0 * X + 1.5 + rng.normal(0, 0.5, n)

Xb = np.column_stack([np.ones(n), X])   # add bias column: [1, x]


# ------------------------------------------------------------
# 2. Loss and gradient (closed forms for linear regression)
#    L(w)   = (1/2m) ||Xw - y||^2   (we drop the 1/2, it cancels)
#    grad L = (1/m) X^T (Xw - y)
# ------------------------------------------------------------
def mse(w, X, y):
    return float(np.mean((X @ w - y) ** 2))


def gradient(w, X, y):
    m = len(y)
    return (1.0 / m) * X.T @ (X @ w - y)


# ------------------------------------------------------------
# 3. Optimizer
#    - "batch": 1 full-data update per epoch
#    - "sgd":   one update per sample, shuffled each epoch (textbook SGD)
#    - "mini":  shuffled mini-batches per epoch
#    - "momentum": batch updates with a velocity term
# ------------------------------------------------------------
def gradient_descent(X, y, variant="batch", lr=0.1, epochs=200,
                     batch_size=32, momentum=0.9, patience=20):
    m = len(y)
    w = np.zeros(X.shape[1])
    v = np.zeros_like(w)
    history = []
    best_w, best_loss, stall = w.copy(), float("inf"), 0

    for epoch in range(epochs):
        if variant == "batch":
            w = w - lr * gradient(w, X, y)
        elif variant == "momentum":
            v = momentum * v + lr * gradient(w, X, y)
            w = w - v
        elif variant == "sgd":
            for i in rng.permutation(m):
                w = w - lr * gradient(w, X[i:i + 1], y[i:i + 1])
        elif variant == "mini":
            idx = rng.permutation(m)
            for start in range(0, m, batch_size):
                batch = idx[start:start + batch_size]
                w = w - lr * gradient(w, X[batch], y[batch])

        loss = mse(w, X, y)
        history.append(loss)
        if loss < best_loss:
            best_loss, best_w, stall = loss, w.copy(), 0
        else:
            stall += 1
            if stall >= patience:
                break
    return best_w, history


# ------------------------------------------------------------
# 4. Reference: closed form  w = (X^T X)^{-1} X^T y
# ------------------------------------------------------------
w_closed = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)
print(f"Closed form:        w0={w_closed[0]:.4f} w1={w_closed[1]:.4f}  "
      f"MSE={mse(w_closed, Xb, y):.4f}")

# Each variant needs an appropriate learning-rate scale:
# momentum accumulates velocity, so it uses a smaller lr.
variant_lr = {"batch": 0.1, "sgd": 0.05, "mini": 0.1, "momentum": 0.05}

results = {}
for name in ["batch", "sgd", "mini", "momentum"]:
    w, hist = gradient_descent(Xb, y, variant=name,
                               lr=variant_lr[name], epochs=300)
    results[name] = (w, hist)
    print(f"{name:<9}: w0={w[0]:.4f} w1={w[1]:.4f}  "
          f"MSE={mse(w, Xb, y):.4f}  ran {len(hist)} epochs")

# ------------------------------------------------------------
# 5. Visualization: loss curves + fitted lines
# ------------------------------------------------------------
plt.figure(figsize=(12, 4.5))

plt.subplot(1, 2, 1)
for name, (w, hist) in results.items():
    plt.plot(hist, label=name)
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.title("Convergence of GD variants")
plt.legend()
plt.yscale("log")

plt.subplot(1, 2, 2)
plt.scatter(X, y, s=4, alpha=0.4, label="data")
xs = np.linspace(X.min(), X.max(), 100)
for name, (w, hist) in results.items():
    plt.plot(xs, w[0] + w[1] * xs, label=name, linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Fitted lines")
plt.legend()

plt.tight_layout()
plt.savefig("gradient_descent_from_scratch.png", dpi=110)
print("\nSaved figure: gradient_descent_from_scratch.png")
print("(move it into diagrams/ or delete it after inspecting the curves)")

# ------------------------------------------------------------
# Takeaways
#  - Batch GD takes the exact steepest step: smooth but each epoch
#    is expensive for large datasets.
#  - SGD makes m cheap noisy steps per epoch: it reaches the
#    minimum after far fewer full passes over the data.
#  - Mini-batch is the practical middle ground (the default in DL).
#  - Momentum accumulates velocity, damps zig-zag, and can escape
#    shallow plateaus — with a smaller lr it matches the others here.
#  - Linear regression is convex, so all variants land at the same
#    optimum the closed form finds (up to early-stopping tolerance).
# ------------------------------------------------------------
