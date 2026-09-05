"""
Generate graphical representations for modules 59-61 (recommenders, graph
ML, RL) plus the new runnable examples. Data panels use the ACTUAL
numbers produced by the example scripts (imported here so they cannot
drift); schematic panels are hand-drawn concept figures.

Run:  python diagrams/generate_module_diagrams.py
Output (PNG, Agg backend):
  diagrams/recommenders/als_vs_sgd_rmse.png   (real benchmark numbers)
  diagrams/recommenders/two_stage_recsys.png  (architecture schematic)
  diagrams/graph/link_prediction_auc.png      (real AUC numbers)
  diagrams/graph/message_passing.png          (GCN step schematic)
  diagrams/rl/qlearning_vs_optimal.png        (real return numbers)
  diagrams/rl/agent_env_loop.png              (RL loop schematic)
"""
import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "diagrams")


def load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save(fig, area, fname):
    d = os.path.join(OUT, area)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, fname)
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


def box(ax, x, y, w, h, text, fc="#eef2fb", ec="#4a6fa5", fs=9, style="round,pad=0.02,rounding_size=0.02"):
    b = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=1.4)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None, color="#333"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.03, text, ha="center",
                fontsize=7.5, color="#555")


# ----------------------------------------------------------------------
# 1. Recommenders: ALS vs SGD (REAL numbers from 02_als_vs_sgd.py)
# ----------------------------------------------------------------------
def panel_als_vs_sgd():
    import numpy as np
    m = load("rec", "code/recommenders/02_als_vs_sgd.py")
    train, test = m.make_ratings()
    mu = float(np.mean([r for *_, r in train]))
    base = float(np.sqrt(np.mean((mu - np.array([r for *_, r in test])) ** 2)))
    _, _, te_sgd = m.sgd(train, test, k=6, lr=0.05, reg=0.1, epochs=40)
    _, tr_als, te_als = m.als(train, test, k=6, reg=1.0, passes=25)

    fig, ax = plt.subplots(figsize=(6.4, 4))
    names = ["global-mean\nbaseline", "SGD\n(40 epochs)", "ALS\n(25 passes)"]
    vals = [base, te_sgd, te_als]
    colors = ["#b0bec5", "#4a6fa5", "#2e7d32"]
    bars = ax.bar(names, vals, color=colors, alpha=0.9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("test RMSE (lower is better)")
    ax.set_title("Matrix factorization: ALS vs SGD on the same ratings",
                 fontsize=11)
    ax.set_ylim(0, max(vals) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "recommenders", "als_vs_sgd_rmse.png")


# ----------------------------------------------------------------------
# 2. Recommenders: two-stage architecture (schematic)
# ----------------------------------------------------------------------
def panel_two_stage():
    fig, ax = plt.subplots(figsize=(9.6, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    box(ax, 4.3, 5.1, 1.4, 0.7, "User U\n(history,\ncontext)", fs=8)
    box(ax, 0.2, 2.6, 2.2, 1.0, "STAGE 1: CANDIDATES\nMF / two-tower x ANN\n(100s of items)", fc="#e8f5e9", ec="#2e7d32")
    box(ax, 3.9, 2.6, 2.2, 1.0, "STAGE 2: RANKING\nfeatures + model\n(user, item, context)", fc="#fff3e0", ec="#ef6c00")
    box(ax, 7.6, 2.6, 2.2, 1.0, "POST-PROCESS\ndedup, diversity,\nbusiness rules", fc="#fce4ec", ec="#c62828")
    box(ax, 7.6, 0.3, 2.2, 1.2, "SERVE TOP-k\n(log + eval\nharness)", fs=8)
    arrow(ax, 5.0, 5.1, 2.4, 3.6)
    arrow(ax, 2.4, 3.1, 3.9, 3.1)
    arrow(ax, 6.1, 3.1, 7.6, 3.1)
    arrow(ax, 8.7, 2.6, 8.7, 1.5)
    arrow(ax, 5.0, 5.05, 8.5, 2.4, text="candidate + features")
    ax.set_title("Two-stage recommender: retrieve hundreds, rank a few\n"
                 "(59_RECOMMENDER_SYSTEMS)", fontsize=12)
    save(fig, "recommenders", "two_stage_recsys.png")


# ----------------------------------------------------------------------
# 3. Graph: link prediction AUC (REAL numbers from 02_link_prediction.py)
# ----------------------------------------------------------------------
def panel_link_auc():
    m = load("lp", "code/graph/02_link_prediction.py")
    n, comm, edges = m.make_graph()
    pos_train, pos_test, neg_test = m.split_edges(n, edges)
    A = __import__("numpy").zeros((n, n))
    import numpy as np
    for (i, j) in pos_train:
        A[i, j] = A[j, i] = 1.0
    deg = A.sum(1).astype(int)
    pairs = pos_test + neg_test
    cn = np.array([m.common_neighbor_score(a, b, A) for a, b in pairs])
    aa = np.array([m.adamic_adar_score(a, b, A, deg) for a, b in pairs])
    Z = m.spectral_embeddings(A)
    emb = np.array([float(np.dot(Z[a], Z[b])) for a, b in pairs])
    n_pos = len(pos_test)
    aucs = {
        "common\nneighbors": m.auc_from_scores(cn[:n_pos], cn[n_pos:]),
        "Adamic-Adar": m.auc_from_scores(aa[:n_pos], aa[n_pos:]),
        "spectral\nembeddings": m.auc_from_scores(emb[:n_pos], emb[n_pos:]),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4))
    bars = ax.bar(list(aucs), list(aucs.values()), color=["#4a6fa5", "#5b8cb8", "#2e7d32"], alpha=0.9)
    for b, v in zip(bars, aucs.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0.5, ls="--", color="#888", lw=1)
    ax.text(0.5, 0.52, "random guess = 0.5", color="#555", fontsize=8)
    ax.set_ylabel("AUC on held-out edges")
    ax.set_title("Link prediction: structure-based scoring beats random\n"
                 "(60_GRAPH_MACHINE_LEARNING)", fontsize=11)
    ax.set_ylim(0, 1.0); ax.grid(axis="y", alpha=0.3)
    save(fig, "graph", "link_prediction_auc.png")


# ----------------------------------------------------------------------
# 4. Graph: message passing (schematic)
# ----------------------------------------------------------------------
def panel_message_passing():
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    cx, cy, r = 5.6, 3.5, 0.55
    nbrs = [(2.0, 5.6), (1.4, 2.4), (8.8, 5.6), (9.6, 2.2), (3.2, 1.4)]
    for (x, y) in nbrs:
        ax.add_patch(plt.Circle((x, y), r - 0.28, fc="#e3f2fd", ec="#1565c0", lw=1.2))
    ax.add_patch(plt.Circle((cx, cy), r, fc="#ffebee", ec="#c62828", lw=2))
    for (x, y) in nbrs:
        arrow(ax, x, y, cx, cy)
    ax.text(cx, cy, "v", ha="center", va="center", fontsize=13, fontweight="bold", color="#c62828")
    ax.text(8.4, 0.2, "h'_v = sigma( W . AGG({ h_u : u in N(v) }) )",
            fontsize=10, family="monospace")
    ax.text(0.2, 6.6, "Each neighbor u sends its representation h_u;\nthe node aggregates (mean/sum/attention)\nand transforms once - L layers = L-hop view.",
            fontsize=9, color="#333")
    ax.set_title("Message passing: one GCN/GAT/GraphSAGE layer",
                 fontsize=12)
    save(fig, "graph", "message_passing.png")


# ----------------------------------------------------------------------
# 5. RL: Q-learning vs optimal (REAL numbers from 01_qlearning_gridworld)
# ----------------------------------------------------------------------
def panel_rl_returns():
    m = load("rl", "code/rl/01_qlearning_gridworld.py")
    env = m.GridWorld()
    V_star = m.value_iteration(env)
    Q = m.q_learning(env)
    g_star = m.policy_return(env, m.optimal_policy(env, V_star))
    g_ql = m.policy_return(env, m.greedy_policy(Q))

    fig, ax = plt.subplots(figsize=(6.4, 4))
    bars = ax.bar(["optimal policy\n(value iteration)", "Q-learning\n(6000 episodes)"],
                  [g_star, g_ql], color=["#4a6fa5", "#2e7d32"], alpha=0.9)
    for b, v in zip(bars, [g_star, g_ql]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("mean discounted return")
    ax.set_title("Gridworld under 15% slip noise: Q-learning vs exact\n"
                 "(61_REINFORCEMENT_LEARNING)", fontsize=11)
    ax.set_ylim(0, max(g_star, g_ql) * 1.2); ax.grid(axis="y", alpha=0.3)
    save(fig, "rl", "qlearning_vs_optimal.png")


# ----------------------------------------------------------------------
# 6. RL: agent-environment loop (schematic)
# ----------------------------------------------------------------------
def panel_rl_loop():
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    box(ax, 0.3, 2.0, 2.4, 1.4, "AGENT\npolicy pi(a|s)\nQ / actor-critic", fc="#e8f5e9", ec="#2e7d32", fs=9)
    box(ax, 9.2, 2.0, 2.4, 1.4, "ENVIRONMENT\np(s'|s,a), r\nstates, rewards", fc="#fff3e0", ec="#ef6c00", fs=9)
    arrow(ax, 2.7, 3.2, 9.2, 3.2, "action a_t")
    arrow(ax, 9.2, 2.2, 2.7, 2.2, "state s_{t+1}, reward r_{t+1}")
    ax.text(6.0, 0.4, "TD target:  r + gamma max_a' Q(s', a')   ->   "
            "Q(s,a) <- Q(s,a) + lr [ target - Q(s,a) ]",
            ha="center", fontsize=10, family="monospace")
    ax.set_title("The agent-environment loop and the TD update it learns from",
                 fontsize=12)
    save(fig, "rl", "agent_env_loop.png")


if __name__ == "__main__":
    panel_als_vs_sgd()
    panel_two_stage()
    panel_link_auc()
    panel_message_passing()
    panel_rl_returns()
    panel_rl_loop()
    print("diagram generation complete")
