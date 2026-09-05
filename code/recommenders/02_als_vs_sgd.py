"""
ALS vs SGD matrix factorization - runnable comparison.
Both fit r_hat = b_u + b_i + p_u . q_i (bias terms included) on the same
synthetic ratings, sharing the 59_RECOMMENDER_SYSTEMS data generator.
SGD: one example at a time (the Funk update). ALS: alternate fixing one
factor side and solving the other with the closed-form ridge solution -
the version production systems use because it parallelizes per user/item.

Run:  python code/recommenders/02_als_vs_sgd.py
Expected: both beat the global-mean baseline; ALS reaches comparable
RMSE in a handful of alternating passes (no learning-rate tuning),
SGD needs many epochs and an LR.
"""
import time
import numpy as np


def make_ratings(n_u=200, n_i=150, k_true=3, seed=42):
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 1, (n_u, k_true))
    V = rng.normal(0, 1, (n_i, k_true))
    pairs = []
    for u in range(n_u):
        for i in rng.choice(n_i, int(n_i * 0.15), replace=False):
            r = 3.0 + 0.5 * (U[u] @ V[i]) + rng.normal(0, 0.3)
            pairs.append((u, int(i), float(np.clip(r, 1, 5))))
    rng.shuffle(pairs)
    cut = int(len(pairs) * 0.8)
    return pairs[:cut], pairs[cut:]


def rmse(P, Q, bu, bi, mu, pairs):
    errs = [mu + bu[u] + bi[i] + P[u] @ Q[i] - r for u, i, r in pairs]
    return float(np.sqrt(np.mean(np.square(errs))))


def sgd(train, test, k=6, lr=0.05, reg=0.1, epochs=40, seed=0):
    rng = np.random.default_rng(seed)
    n_u = max(r[0] for r in train) + 1
    n_i = max(r[1] for r in train) + 1
    mu = float(np.mean([r for *_, r in train]))
    P = rng.normal(0, 0.1, (n_u, k))
    Q = rng.normal(0, 0.1, (n_i, k))
    bu = np.zeros(n_u); bi = np.zeros(n_i)
    t0 = time.time()
    train_rmse, test_rmse = None, None
    for ep in range(epochs):
        rng.shuffle(train)
        for u, i, r in train:
            pred = mu + bu[u] + bi[i] + P[u] @ Q[i]
            e = r - pred
            bu[u] += lr * (e - reg * bu[u])
            bi[i] += lr * (e - reg * bi[i])
            P[u] += lr * (e * Q[i] - reg * P[u])
            Q[i] += lr * (e * P[u] - reg * Q[i])
        if ep % 10 == 9:
            train_rmse = rmse(P, Q, bu, bi, mu, train)
    test_rmse = rmse(P, Q, bu, bi, mu, test)
    return time.time() - t0, train_rmse, test_rmse


def als(train, test, k=6, reg=1.0, passes=25, seed=0):
    """Alternating least squares with biases. Per user/item solve:
    (G^T G + reg I) x = G^T r   with G = [1, biases, factors] columns."""
    n_u = max(r[0] for r in train) + 1
    n_i = max(r[1] for r in train) + 1
    mu = float(np.mean([r for *_, r in train]))
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 0.1, (n_u, k))
    Q = rng.normal(0, 0.1, (n_i, k))
    bu = np.zeros(n_u); bi = np.zeros(n_i)
    # user -> list of (item, rating); item -> list of (user, rating)
    by_u = {}
    by_i = {}
    for u, i, r in train:
        by_u.setdefault(u, []).append((i, r))
        by_i.setdefault(i, []).append((u, r))

    t0 = time.time()
    train_rmse = None
    for it in range(passes):
        # ALS half-step 1: fix Q, solve each user's p_u + intercept
        for u, pairs in by_u.items():
            P[u] = _solve_entity(pairs, Q, bi, bu, u, reg, mu)
        # ALS half-step 2: fix P, solve each item's q_i + intercept
        for i, pairs in by_i.items():
            Q[i] = _solve_entity(pairs, P, bu, bi, i, reg, mu)
        if (it + 1) % 5 == 0:
            train_rmse = rmse(P, Q, bu, bi, mu, train)
    test_rmse = rmse(P, Q, bu, bi, mu, test)
    return time.time() - t0, train_rmse, test_rmse


def _solve_entity(pairs, factors, other_bias, own_bias, idx, reg, mu):
    """Ridge solve for one entity given its rated partners' factors."""
    F = np.array([factors[j] for j, _ in pairs])
    G = np.hstack([np.ones((F.shape[0], 1)), F])
    y = np.array([r - mu - other_bias[j] for j, r in pairs])
    w = np.linalg.solve(G.T @ G + reg * np.eye(G.shape[1]), G.T @ y)
    own_bias[idx] = w[0]
    return w[1:]


def main():
    train, test = make_ratings()
    mu = float(np.mean([r for *_, r in train]))
    base = float(np.sqrt(np.mean([(mu - r) ** 2 for *_, r in test])))
    print(f"ratings: train {len(train)} | test {len(test)} | "
          f"baseline (global mean) test RMSE: {base:.4f}\n")

    t_sgd, tr_sgd, te_sgd = sgd(train, test, k=6, lr=0.05, reg=0.1,
                                epochs=40)
    t_als, tr_als, te_als = als(train, test, k=6, reg=1.0, passes=25)
    print(f"{'method':<8} {'wall time':>10} {'train RMSE':>11} "
          f"{'test RMSE':>11}")
    print(f"{'SGD':<8} {t_sgd:>8.2f}s {tr_sgd:>11.4f} {te_sgd:>11.4f}")
    print(f"{'ALS':<8} {t_als:>8.2f}s {tr_als:>11.4f} {te_als:>11.4f}")
    ok = te_sgd < base * 0.85 and te_als < base * 0.85
    print("\n=> PASS: both beat the baseline"
          if ok else "=> check k/reg/epochs (both should beat baseline)")
    print("Takeaway: ALS needs no learning rate, reaches the same or better")
    print("RMSE in ~25 alternating passes, and parallelizes per user/item -")
    print("why Spark ALS dominates at scale.")


if __name__ == "__main__":
    main()
