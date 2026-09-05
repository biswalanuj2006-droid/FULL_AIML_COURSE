"""
Link prediction - runnable graph example (pure NumPy).
Holds out real edges as positives, samples non-edges as negatives, and
scores candidate edges with three methods from 60_GRAPH_MACHINE_LEARNING:
  (a) common neighbors, (b) Adamic-Adar, (c) spectral embeddings
      (z = U sqrt(S) from the SVD of the adjacency, score = dot product).
Reports AUC for each - the honest evaluation protocol from the module
(negatives = random non-edges; split edges, never a flattened table).

Run:  python code/graph/02_link_prediction.py
Expected: all methods AUC > 0.5 on the planted-community graph;
embedding score >= common-neighbors (structure learned, not counted).
"""
import numpy as np


def auc_from_scores(pos_scores, neg_scores):
    """Rank-based AUC (Mann-Whitney) - no sklearn needed."""
    all_scores = np.concatenate([pos_scores, neg_scores])
    order = np.argsort(all_scores)  # ascending; ties get fractional rank
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(all_scores, return_inverse=True, return_counts=True)
    if cnt.max() > 1:
        for val in np.unique(all_scores):
            ranks[all_scores == val] = ranks[all_scores == val].mean()
    n_pos = len(pos_scores)
    sum_ranks_pos = ranks[:n_pos].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * len(neg_scores))


def make_graph(n_per=70, p_in=0.22, p_out=0.015, seed=0):
    """Two planted communities (structure carries the link signal)."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per
    comm = np.array([0] * n_per + [1] * n_per)
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if comm[i] == comm[j] else p_out
            if rng.random() < p:
                edges.add((i, j))
    return n, comm, edges


def split_edges(n, edges, frac=0.15, seed=0):
    """Hold out edges as positives; sample equal non-edges as negatives."""
    rng = np.random.default_rng(seed)
    edge_list = sorted(edges)
    n_hold = max(1, int(len(edge_list) * frac))
    test_pos = set(rng.choice(len(edge_list), n_hold, replace=False))
    pos_test = [edge_list[i] for i in sorted(test_pos)]
    pos_train = {e for i, e in enumerate(edge_list) if i not in test_pos}
    non_edges = set()
    while len(non_edges) < n_hold:
        i, j = rng.integers(0, n), rng.integers(0, n)
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) not in edges and (a, b) not in non_edges:
            non_edges.add((a, b))
    return pos_train, pos_test, sorted(non_edges)


def common_neighbor_score(a, b, adj):
    return float(np.dot(adj[a], adj[b]))


def adamic_adar_score(a, b, adj, deg):
    nbrs = np.nonzero(adj[a])[0]
    nbrs_b = set(np.nonzero(adj[b])[0])
    return float(sum(1.0 / np.log(deg[u]) for u in nbrs if u in nbrs_b
                     and deg[u] > 1))


def spectral_embeddings(adj, dim=16):
    """z = U sqrt(S): SVD of the adjacency gives node embeddings whose
    dot product approximates connection probability (low-rank A ~ Z Z^T)."""
    U, S, _ = np.linalg.svd(adj)
    return U[:, :dim] * np.sqrt(S[:dim])[None, :]


def main():
    n, comm, edges = make_graph()
    pos_train, pos_test, neg_test = split_edges(n, edges)
    print(f"graph: {n} nodes | {len(edges)} edges | train {len(pos_train)} "
          f"| test pos {len(pos_test)} | test neg {len(neg_test)}")

    A = np.zeros((n, n))
    for (i, j) in pos_train:
        A[i, j] = A[j, i] = 1.0
    deg = A.sum(1).astype(int)

    pairs = pos_test + neg_test
    cn = np.array([common_neighbor_score(a, b, A) for a, b in pairs])
    aa = np.array([adamic_adar_score(a, b, A, deg) for a, b in pairs])
    Z = spectral_embeddings(A)
    emb = np.array([float(np.dot(Z[a], Z[b])) for a, b in pairs])

    n_pos = len(pos_test)
    results = [
        ("common neighbors", auc_from_scores(cn[:n_pos], cn[n_pos:])),
        ("Adamic-Adar      ", auc_from_scores(aa[:n_pos], aa[n_pos:])),
        ("spectral emb. dot", auc_from_scores(emb[:n_pos], emb[n_pos:])),
    ]
    print(f"\nmethod              AUC")
    for name, auc in results:
        print(f"  {name}   {auc:.3f}")
    best = max(auc for _, auc in results)
    print("\n=> PASS: structure-based link prediction beats random (0.5)"
          if best > 0.65 else "=> check graph density/holdout fraction")


if __name__ == "__main__":
    main()
