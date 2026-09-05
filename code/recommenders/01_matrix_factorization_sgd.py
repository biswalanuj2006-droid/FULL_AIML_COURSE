"""
Matrix Factorization (Funk SVD) with SGD - runnable recommender example.
Verifies the core math of 59_RECOMMENDER_SYSTEMS: predict ratings as
user-item dot products, trained by SGD on observed ratings only.

Run:  python code/recommenders/01_matrix_factorization_sgd.py
Expected: test RMSE clearly below the mean-imputation baseline.
"""
import numpy as np


def mf_sgd(ratings, k=6, lr=0.03, reg=0.3, epochs=40, seed=0):
    """ratings: list of (user_idx, item_idx, rating). Returns P, Q.
    k is a bias-variance knob: true latent dims here are 3; k=10 with a
    sparse matrix overfits (test RMSE explodes) - the lesson's point."""
    rng = np.random.default_rng(seed)
    n_users = max(r[0] for r in ratings) + 1
    n_items = max(r[1] for r in ratings) + 1
    # small random init: scale matters for dot products
    P = rng.normal(0, 0.1, (n_users, k))
    Q = rng.normal(0, 0.1, (n_items, k))
    train = ratings[:int(len(ratings) * 0.8)]
    test = ratings[int(len(ratings) * 0.8):]
    for epoch in range(epochs):
        rng.shuffle(train)
        for u, i, r in train:
            err = r - P[u] @ Q[i]
            # SGD update with L2 regularization (derive it!)
            P[u] += lr * (err * Q[i] - reg * P[u])
            Q[i] += lr * (err * P[u] - reg * Q[i])
        if (epoch + 1) % 10 == 0:
            rmse = _rmse(P, Q, train)
            print(f"  epoch {epoch + 1:3d}  train RMSE {rmse:.4f}")
    return P, Q, train, test


def _rmse(P, Q, pairs):
    errs = [P[u] @ Q[i] - r for u, i, r in pairs]
    return float(np.sqrt(np.mean(np.square(errs))))


def main():
    rng = np.random.default_rng(42)
    # synthetic: 3 latent "tastes". Scale the dot product so ratings are
    # spread across 1-5, NOT collapsed against the clip bounds (the first
    # version of this demo used raw N(0,1) factors -> ~75% of ratings hit
    # the floor at exactly 1.0 and there was almost no signal to learn).
    n_u, n_i, k_true = 200, 150, 3
    U = rng.normal(0, 1, (n_u, k_true))
    V = rng.normal(0, 1, (n_i, k_true))
    pairs = []
    for u in range(n_u):
        # each user rates ~15% of items
        for i in rng.choice(n_i, int(n_i * 0.15), replace=False):
            r = 3.0 + 0.5 * (U[u] @ V[i]) + rng.normal(0, 0.3)  # ~N(3, 0.9)
            pairs.append((u, int(i), float(np.clip(r, 1, 5))))
    rng.shuffle(pairs)   # RANDOM-pair split below - slicing an ordered list
    # would give the test set users/items with little training data
    print(f"ratings: {len(pairs)} | users {n_u} | items {n_i} "
          f"| spread: {min(r for *_, r in pairs):.1f}-{max(r for *_, r in pairs):.1f}")

    k = 6
    P, Q, train, test = mf_sgd(pairs, k=k, lr=0.05, reg=0.15, epochs=60)
    # baselines
    mean_global = np.mean([r for _, _, r in train])
    base = float(np.sqrt(np.mean([(mean_global - r) ** 2 for _, _, r in test])))
    mf = _rmse(P, Q, test)
    print(f"\nbaseline (predict global mean): test RMSE {base:.4f}")
    print(f"matrix factorization (k={k})    : test RMSE {mf:.4f}")
    print("=> PASS" if mf < base * 0.85 else "=> check hyperparameters")


if __name__ == "__main__":
    main()
