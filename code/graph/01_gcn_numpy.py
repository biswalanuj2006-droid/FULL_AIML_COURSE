"""
2-layer GCN in pure NumPy - runnable graph example.
Implements the GCN update from 60_GRAPH_MACHINE_LEARNING:
    H^(l+1) = ReLU( D^{-1/2} A D^{-1/2} H^(l) W^(l) )
on a synthetic 2-community graph (planted partition), then trains the
weights with gradient descent to classify nodes from a few labels.
Purpose: show that message passing beats node-features-only when the
label signal lives in the GRAPH structure.

Run:  python code/graph/01_gcn_numpy.py
Expected: accuracy clearly above the majority-class baseline.
"""
import numpy as np


def relu(x):
    return np.maximum(x, 0)


def make_community_graph(n_per=60, p_in=0.25, p_out=0.02, seed=0):
    rng = np.random.default_rng(seed)
    n = 2 * n_per
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if (i < n_per) == (j < n_per) else p_out
            if rng.random() < p:
                A[i, j] = A[j, i] = 1.0
    labels = np.array([0] * n_per + [1] * n_per)
    # features: NO community signal - just noise + degree (structure-only)
    X = rng.normal(0, 1, (n, 5))
    return A, X, labels


def normalize_adjacency(A):
    """Symmetric normalization: D^-1/2 (A + I) D^-1/2 (see lesson)."""
    A = A + np.eye(A.shape[0])
    deg = A.sum(1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    return d_inv_sqrt @ A @ d_inv_sqrt


def forward(X, Ahat, W1, W2):
    H1 = relu(Ahat @ X @ W1)
    logits = Ahat @ H1 @ W2
    return H1, logits


def softmax_ce_grad(logits, y_onehot):
    """Returns loss and dLoss/dlogits (pred - onehot)."""
    exps = np.exp(logits - logits.max(1, keepdims=True))
    p = exps / exps.sum(1, keepdims=True)
    loss = -np.mean(np.sum(y_onehot * np.log(p + 1e-12), 1))
    return loss, (p - y_onehot) / logits.shape[0]


def full_loss_grad(X, Ahat, W1, W2, y_onehot, mask):
    """Vectorized forward + full backward pass. logits = Ahat H1 W2, so
    BOTH layers need their Ahat (message-passing) factor on the way back:
      dH1 = Ahat^T (dlogits W2^T)   and   dW2 = (Ahat H1)^T dlogits
    (missing these was the classic bug in the first version of this file)."""
    H1 = relu(Ahat @ X @ W1)
    logits = Ahat @ H1 @ W2
    exps = np.exp(logits - logits.max(1, keepdims=True))
    p = exps / exps.sum(1, keepdims=True)
    loss = -np.mean(np.sum(y_onehot * np.log(p + 1e-12), 1))
    dlogits = (p - y_onehot) * mask[:, None]   # labeled nodes only
    dW2 = (Ahat @ H1).T @ dlogits
    dH1 = Ahat.T @ (dlogits @ W2.T)
    dZ1 = dH1 * (H1 > 0)
    dW1 = X.T @ (Ahat.T @ dZ1)
    return loss, dW1, dW2


def main():
    A, X, y = make_community_graph()
    n, d = X.shape
    Ahat = normalize_adjacency(A)
    hidden, n_class = 16, 2

    rng = np.random.default_rng(0)
    W1 = rng.normal(0, 0.1, (d, hidden))
    W2 = rng.normal(0, 0.1, (hidden, n_class))

    # few labeled nodes per class (semi-supervised setting)
    labeled = np.concatenate([rng.choice(np.where(y == 0)[0], 5),
                              rng.choice(np.where(y == 1)[0], 5)])
    y_onehot = np.eye(n_class)[y]
    mask = np.zeros(n, bool)
    mask[labeled] = True

    lr = 1.0
    for epoch in range(400):
        loss, dW1, dW2 = full_loss_grad(X, Ahat, W1, W2, y_onehot, mask)
        W1 -= lr * dW1
        W2 -= lr * dW2
        if epoch % 50 == 0:
            print(f"  epoch {epoch:3d}  labeled loss {loss:.4f}")

    _, logits = forward(X, Ahat, W1, W2)
    pred = logits.argmax(1)
    acc = (pred == y).mean()
    maj = max((y == c).mean() for c in range(n_class))
    print(f"\nmajority-class baseline accuracy : {maj:.3f}")
    print(f"2-layer GCN (5 labels/class)    : {acc:.3f}")
    print("=> PASS: structure beats features-only baseline"
          if acc > maj + 0.1 else "=> check graph/features split")


if __name__ == "__main__":
    main()
