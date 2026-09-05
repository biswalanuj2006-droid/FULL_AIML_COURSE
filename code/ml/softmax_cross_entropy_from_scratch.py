# ============================================================
# SOFTMAX + CROSS-ENTROPY FROM SCRATCH
#
# Softmax turns logits z into a probability distribution:
#     p_i = exp(z_i) / sum_j exp(z_j)
# Cross-entropy loss for one-hot target y:
#     L = -sum_i y_i * log(p_i)
#
# Numerically stable trick: subtract max(z) before exp — the
# softmax is invariant to adding a constant to all logits.
#
# The key result (worth memorizing): with softmax + cross-entropy
# the gradient of the loss w.r.t. logits simplifies to
#     dL/dz = p - y
# which is why classification layers are so easy to backprop.
#
# Run: python softmax_cross_entropy_from_scratch.py
# Requires: numpy
# ============================================================
import numpy as np


# ------------------------------------------------------------
# 1. Softmax
# ------------------------------------------------------------
def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)   # numerical stability
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


# ------------------------------------------------------------
# 2. Cross-entropy (works with one-hot OR integer labels)
# ------------------------------------------------------------
def cross_entropy(z, y_int, eps=1e-12):
    """z: logits (n_samples, n_classes), y_int: integer labels (n,)."""
    p = softmax(z)
    n = len(y_int)
    # log-likelihood of the true class per sample
    log_lik = np.log(p[np.arange(n), y_int] + eps)
    return float(-np.mean(log_lik))


# ------------------------------------------------------------
# 3. Gradient: dL/dz = (p - y_onehot) / n   [per-sample (p - y)]
# ------------------------------------------------------------
def gradient_logits(z, y_int):
    p = softmax(z)
    g = p.copy()
    g[np.arange(len(y_int)), y_int] -= 1.0   # p - y
    return g / len(y_int)


# ------------------------------------------------------------
# 4. Sanity check vs finite differences on a random logit row
# ------------------------------------------------------------
def finite_diff_check(z_row, y, eps=1e-6):
    """Compare analytic gradient with numeric gradient for one sample."""
    analytic = gradient_logits(z_row[None, :], np.array([y]))[0]

    numeric = np.zeros_like(z_row)
    for i in range(len(z_row)):
        zp, zm = z_row.copy(), z_row.copy()
        zp[i] += eps
        zm[i] -= eps
        numeric[i] = (cross_entropy(zp[None, :], np.array([y])) -
                      cross_entropy(zm[None, :], np.array([y]))) / (2 * eps)
    return analytic, numeric


# ------------------------------------------------------------
# 5. Small training loop: linear layer + softmax on synthetic
#    3-class data, update with gradient descent.
# ------------------------------------------------------------
def demo_training():
    rng = np.random.default_rng(0)
    # 3 separable-ish blobs
    X = np.vstack([
        rng.normal([-2, -2], 0.6, (200, 2)),
        rng.normal([2, -2], 0.6, (200, 2)),
        rng.normal([0, 2], 0.6, (200, 2)),
    ])
    y = np.repeat([0, 1, 2], 200)

    W = np.random.default_rng(1).normal(0, 0.1, (3, 2))
    b = np.zeros(3)

    for epoch in range(200):
        z = X @ W.T + b
        loss = cross_entropy(z, y)
        grad_z = gradient_logits(z, y)            # (n, 3)
        # chain rule: dL/dW = grad_z^T X, dL/db = sum(grad_z)
        dW = grad_z.T @ X
        db = grad_z.sum(axis=0)
        lr = 0.5
        W -= lr * dW
        b -= lr * db
        if epoch % 40 == 0:
            acc = np.mean(np.argmax(z, axis=1) == y)
            print(f"epoch {epoch:3d}  loss={loss:.4f}  accuracy={acc:.2f}")


# ------------------------------------------------------------
# TESTS
# ------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(5, 4))
    labels = rng.integers(0, 4, 5)

    p = softmax(logits)
    print("Softmax rows sum to 1:", np.allclose(p.sum(axis=1), 1))
    print(f"Cross-entropy loss: {cross_entropy(logits, labels):.4f}")

    # finite-difference gradient check
    z_row = rng.normal(size=4)
    y_val = 2
    analytic, numeric = finite_diff_check(z_row, y_val)
    print("\nGradient check (dL/dz = p - y):")
    print("  analytic:", np.round(analytic, 6))
    print("  numeric: ", np.round(numeric, 6))
    print("  match:", np.allclose(analytic, numeric, atol=1e-5))

    print("\nTraining a tiny 3-class linear classifier (no hidden layer):")
    demo_training()

    # numeric example from theory: logits [2.0, 1.0, 0.1]
    ex = np.array([2.0, 1.0, 0.1])
    ex_p = softmax(ex)
    print("\nExample: softmax([2.0, 1.0, 0.1]) =",
          np.round(ex_p, 4), "-> sums to", round(ex_p.sum(), 6))
