"""
generate_rag_media.py - RAG DIAGRAMS + VIDEOS + IMAGES
=======================================================
Three jobs, one script:

  A) RAG diagram panels  -> diagrams/rag/*.png   (matches the style of
     generate_course_diagrams.py: Agg backend, box/arrow/save helpers)
  B) Animated videos     -> Videos/*.mp4         (matplotlib + ffmpeg, the
     available ffmpeg build is used via FFMpegWriter; each clip is short,
     ~4 s at 12 fps, 960x540)
  C) Images folder       -> Images/<category>/*  (copies of every diagram,
     organized by category, so the root Images/ folder is populated)

Run:  python diagrams/generate_rag_media.py
"""
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAGRAMS = os.path.join(ROOT, "diagrams")
RAG = os.path.join(DIAGRAMS, "rag")
VIDEOS = os.path.join(ROOT, "Videos")
IMAGES = os.path.join(ROOT, "Images")

NPIX = 140


def save(fig, area, fname):
    d = os.path.join(DIAGRAMS, area)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, fname)
    fig.savefig(p, dpi=NPIX, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


def box(ax, x, y, w, h, text, fc="#eef2fb", ec="#4a6fa5", fs=8.5,
        style="round,pad=0.02,rounding_size=0.02"):
    b = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=1.3)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None, color="#333", lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.035, text, ha="center",
                fontsize=7, color="#555")


# =============================================================================
# A) RAG DIAGRAM PANELS -> diagrams/rag/
# =============================================================================

def rag_lifecycle():
    """One figure: ingestion column + query column meeting at the LLM."""
    fig, ax = plt.subplots(figsize=(10.5, 7))
    ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Complete RAG lifecycle\n"
                 "ingestion pipeline (left) + query pipeline (right) -> grounded answer",
                 fontsize=12)
    # ingestion column
    ingest = ["DOCUMENTS", "LOAD + PARSE", "CLEAN + NORMALIZE",
              "METADATA", "CHUNK", "EMBED", "INDEX (vector store)"]
    y = 9.3
    for i, t in enumerate(ingest):
        box(ax, 0.4, y - i * 1.15, 3.3, 0.8, t, fc="#eaf6ec", ec="#2e7d32")
        if i < len(ingest) - 1:
            arrow(ax, 2.05, y - i * 1.15 - 0.02, 2.05, y - (i + 1) * 1.15 - 0.72)
    # query column
    query = ["USER QUERY", "QUERY PROCESSING", "QUERY TRANSFORM (rewrite / HyDE)",
             "EMBED", "HYBRID RETRIEVAL (dense + BM25)", "FILTER + RERANK",
             "CONTEXT + PROMPT"]
    y = 9.3
    for i, t in enumerate(query):
        box(ax, 10.3, y - i * 1.15, 3.3, 0.8, t, fc="#fff7e6", ec="#b26a00")
        if i < len(query) - 1:
            arrow(ax, 11.95, y - i * 1.15 - 0.02, 11.95, y - (i + 1) * 1.15 - 0.72)
    # join at the LLM
    box(ax, 5.4, 4.3, 3.2, 1.0, "LLM (grounded generation)", fc="#fde8e8",
        ec="#b02a37", fs=9.5)
    arrow(ax, 3.7, 6.0, 5.4, 4.95, "candidates")
    arrow(ax, 10.3, 6.0, 8.6, 4.95, "prompt")
    box(ax, 5.4, 2.4, 3.2, 0.9, "CITATIONS [1] [2]", fc="#ede7f6",
        ec="#5e35b1")
    box(ax, 5.4, 0.9, 3.2, 0.9, "FINAL GROUNDED ANSWER", fc="#e3f2fd",
        ec="#1565c0")
    arrow(ax, 7.0, 4.3, 7.0, 3.3)
    arrow(ax, 7.0, 2.4, 7.0, 1.8)
    save(fig, "rag", "rag_lifecycle.png")


def rerank_funnel():
    """Retrieve 50 -> cross-encoder rerank -> top 5 -> LLM."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.6); ax.axis("off")
    ax.set_title("Two-stage retrieval: the rerank funnel\n"
                 "bi-encoders are fast but coarse; cross-encoders are slow but precise",
                 fontsize=11.5)
    box(ax, 0.5, 4.6, 2.6, 1.1, "QUERY", fc="#fff7e6", ec="#b26a00")
    box(ax, 4.3, 4.6, 3.4, 1.1, "Bi-encoder ANN search\n(top-50 candidates)",
        fc="#e3f2fd", ec="#1565c0", fs=8.5)
    arrow(ax, 3.1, 5.15, 4.3, 5.15)
    box(ax, 8.6, 4.6, 2.9, 1.1, "Cross-encoder\nRERANKER", fc="#ede7f6",
        ec="#5e35b1", fs=8.5)
    arrow(ax, 7.7, 5.15, 8.6, 5.15, "score all 50")
    box(ax, 8.6, 2.7, 2.9, 1.0, "TOP 5 CONTEXT", fc="#eaf6ec",
        ec="#2e7d32")
    arrow(ax, 10.05, 4.6, 10.05, 3.7, "keep 5")
    box(ax, 4.3, 0.6, 3.4, 1.0, "LLM (grounded answer)", fc="#fde8e8",
        ec="#b02a37")
    arrow(ax, 8.6, 2.7, 7.7, 1.25, "context")
    # latency annotation
    ax.text(4.3, 0.05, "latency: ANN ~ms  |  reranker ~tens of ms  |  LLM ~s",
            fontsize=8, color="#555")
    save(fig, "rag", "rerank_funnel.png")


def hybrid_rrf():
    """Dense + sparse -> RRF fusion worked example."""
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("Hybrid retrieval with Reciprocal Rank Fusion (RRF)\n"
                 "score = sum over lists of 1/(k + rank), k = 60", fontsize=11.5)
    dense = [("d2", 1), ("d1", 2), ("d5", 3), ("d3", 4), ("d7", 5)]
    sparse = [("d7", 1), ("d4", 2), ("d2", 3), ("d9", 4), ("d1", 5)]
    box(ax, 0.4, 5.5, 2.6, 1.0, "DENSE (semantic)", fc="#e3f2fd",
        ec="#1565c0")
    box(ax, 0.4, 1.2, 2.6, 1.0, "SPARSE (BM25)", fc="#fff7e6", ec="#b26a00")
    for i, (c, r) in enumerate(dense):
        ax.text(0.6, 4.9 - i * 0.8, f"rank {r}: {c}", fontsize=8.5)
    for i, (c, r) in enumerate(sparse):
        ax.text(0.6, 0.55 - i * 0.8, f"rank {r}: {c}", fontsize=8.5)
    # contribution bars
    ax.text(4.2, 6.5, "RRF contribution 1/(60+rank)", fontsize=8.5, color="#555")
    items = {}
    for lst, col in ((dense, "#1565c0"), (sparse, "#b26a00")):
        for rank, (c, _) in enumerate(lst, start=1):
            items.setdefault(c, []).append(1.0 / (60 + rank))
    order = sorted(items, key=lambda c: -sum(items[c]))
    for i, c in enumerate(order):
        x = 4.4
        total = sum(items[c])
        ax.barh(-i, total, height=0.6, left=x, color="#2e7d32")
        for part in items[c]:
            ax.barh(-i, part, height=0.6, left=x, color="#2e7d32",
                    alpha=0.45)
            x += part
        ax.text(4.4, -i + 0.32, c, fontsize=9, fontweight="bold")
        ax.text(4.4 + total + 0.15, -i + 0.28, f"{total:.4f}", fontsize=8)
    ax.text(4.4, 0.9, "FUSED RANKING (d2 > d7 > d1 > ...)", fontsize=9.5,
            color="#2e7d32", fontweight="bold")
    ax.set_ylim(-5.4, 7)
    ax.text(0.4, -5.1, "dense finds paraphrases, sparse finds identifiers;\n"
            "RRF merges the two rankings without score normalization",
            fontsize=8, color="#555")
    save(fig, "rag", "hybrid_rrf.png")


def parent_child():
    """Child chunks embedded; retrieve child -> return parent."""
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.6); ax.axis("off")
    ax.set_title("Parent-child (small-to-big) retrieval\n"
                 "precision of small chunks + context of large parents",
                 fontsize=11.5)
    box(ax, 0.4, 5.2, 2.8, 1.0, "QUERY", fc="#fff7e6", ec="#b26a00")
    box(ax, 4.4, 4.4, 3.0, 1.0, "CHILD chunks\n(256 tokens, embedded)",
        fc="#e3f2fd", ec="#1565c0")
    arrow(ax, 3.2, 5.7, 4.4, 5.0)
    box(ax, 8.6, 4.4, 2.8, 1.0, "Retrieve child #3\n(semantically precise)",
        fc="#ede7f6", ec="#5e35b1", fs=8)
    arrow(ax, 7.4, 4.9, 8.6, 4.9)
    box(ax, 4.4, 1.0, 3.0, 1.2, "PARENT document\n(full section, passed to LLM)",
        fc="#eaf6ec", ec="#2e7d32")
    box(ax, 8.6, 1.0, 2.8, 1.2, "LLM gets the full\ncontext, cites parent",
        fc="#fde8e8", ec="#b02a37", fs=8)
    arrow(ax, 10.0, 4.4, 8.8, 2.3, "jump to parent")
    arrow(ax, 8.6, 1.6, 7.4, 1.6)
    ax.text(4.4, 0.35, "why: small chunks embed precisely (high recall @ small K); "
            "parents restore the surrounding context the LLM needs",
            fontsize=8, color="#555")
    save(fig, "rag", "parent_child.png")


def agentic_rag():
    """Planner -> tools -> verifier loop."""
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Agentic RAG: plan -> retrieve -> verify -> answer\n"
                 "(traditional RAG is the single pass inside the retriever box)",
                 fontsize=11.5)
    box(ax, 0.4, 6.6, 2.8, 1.1, "USER", fc="#fff7e6", ec="#b26a00")
    box(ax, 4.4, 6.6, 2.8, 1.1, "PLANNER\n(decompose, route)", fc="#ede7f6",
        ec="#5e35b1")
    arrow(ax, 3.2, 7.15, 4.4, 7.15)
    tools = ["RETRIEVER\n(vector + BM25)", "WEB SEARCH", "SQL / DB",
             "CALCULATOR", "GRAPH TRAVERSAL"]
    for i, t in enumerate(tools):
        box(ax, 8.8, 7.0 - i * 1.35, 2.7, 1.0, t, fc="#e3f2fd", ec="#1565c0")
        arrow(ax, 7.2, 6.9 - i * 1.35 + 0.45, 8.8, 6.9 - i * 1.35 + 0.45)
    box(ax, 4.4, 1.2, 2.8, 1.1, "VERIFIER\n(grounded? cite?)", fc="#eaf6ec",
        ec="#2e7d32")
    arrow(ax, 10.15, 5.0, 6.5, 2.0, "evidence + observations")
    box(ax, 0.4, 1.2, 2.8, 1.1, "FINAL ANSWER\nwith citations", fc="#fde8e8",
        ec="#b02a37")
    arrow(ax, 4.4, 1.75, 3.2, 1.75)
    # loop annotation
    ax.annotate("", xy=(3.0, 5.2), xytext=(4.4, 6.6),
                arrowprops=dict(arrowstyle="-|>", color="#b02a37", lw=1.5))
    ax.text(3.4, 5.75, "iterate /\nreflect", fontsize=7.5, color="#b02a37")
    ax.text(0.4, 0.35, "stopping conditions: verifier passes, max steps reached, "
            "or cost/latency budget spent", fontsize=8, color="#555")
    save(fig, "rag", "agentic_rag.png")


def graph_vs_vector():
    """Vector RAG vs GraphRAG side by side."""
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6.6); ax.axis("off")
    ax.set_title("Vector RAG vs GraphRAG\n"
                 "chunks in a vector space vs entities in a knowledge graph",
                 fontsize=11.5)
    # vector side
    box(ax, 0.4, 5.4, 3.0, 1.0, "VECTOR RAG", fc="#e3f2fd", ec="#1565c0",
        fs=9.5)
    box(ax, 0.4, 3.4, 3.0, 0.9, "chunk -> embed -> ANN\nsemantic similarity",
        fc="#eef2fb", ec="#4a6fa5")
    box(ax, 0.4, 1.4, 3.0, 0.9, "answers = nearest chunks", fc="#eef2fb",
        ec="#4a6fa5")
    ax.text(0.4, 0.7, "good for: exact facts, prose,\nparaphrase questions",
            fontsize=7.5, color="#555")
    arrow(ax, 1.9, 5.4, 1.9, 4.3)
    arrow(ax, 1.9, 3.4, 1.9, 2.3)
    # graph side
    box(ax, 10.3, 5.4, 3.2, 1.0, "GRAPH RAG", fc="#eaf6ec", ec="#2e7d32",
        fs=9.5)
    box(ax, 10.3, 3.4, 3.2, 0.9, "entity + relation extraction ->\nknowledge graph",
        fc="#eef6ee", ec="#2e7d32", fs=7.5)
    box(ax, 10.3, 1.4, 3.2, 0.9, "community summaries +\ngraph traversal",
        fc="#eef6ee", ec="#2e7d32", fs=7.5)
    ax.text(10.3, 0.7, "good for: multi-hop questions,\naggregation, relationships",
            fontsize=7.5, color="#555")
    arrow(ax, 11.9, 5.4, 11.9, 4.3)
    arrow(ax, 11.9, 3.4, 11.9, 2.3)
    # middle: hybrid
    box(ax, 5.2, 3.3, 3.6, 1.0, "HYBRID: vector + graph\n(recall + connectivity)",
        fc="#fff7e6", ec="#b26a00", fs=8.5)
    ax.text(5.2, 2.6, "vector finds candidate facts;\ngraph connects them across docs",
            fontsize=7.5, color="#555")
    save(fig, "rag", "graph_vs_vector_rag.png")


def rag_metrics():
    """Retrieval + generation metric families."""
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("RAG evaluation: measure retrieval AND generation separately",
                 fontsize=11.5)
    box(ax, 0.4, 5.6, 6.0, 1.0, "RETRIEVAL METRICS (rank quality)",
        fc="#e3f2fd", ec="#1565c0", fs=9.5)
    for i, t in enumerate(["Hit@K: gold chunk in top-K?",
                           "Recall@K / Precision@K",
                           "MRR: 1 / rank of first relevant",
                           "NDCG: graded ranking quality"]):
        ax.text(0.7, 4.9 - i * 0.85, t, fontsize=8.5)
    box(ax, 7.6, 5.6, 6.0, 1.0, "GENERATION METRICS (answer quality)",
        fc="#eaf6ec", ec="#2e7d32", fs=9.5)
    for i, t in enumerate(["Faithfulness: claims in evidence?",
                           "Answer relevance: addresses the query?",
                           "Context precision/recall",
                           "F1 / BLEU / ROUGE / BERTScore"]):
        ax.text(7.9, 4.9 - i * 0.85, t, fontsize=8.5)
    box(ax, 0.4, 1.2, 13.2, 1.0,
        "golden set: {question, ground-truth answer, relevant chunks, hard negatives}\n"
        "A RAG system can score 100% retrieval and still fail generation - and vice versa.",
        fc="#fde8e8", ec="#b02a37", fs=8.5)
    save(fig, "rag", "rag_metrics.png")


def temperature_softmax():
    """Generation parameter: P(token) = softmax(logits / T)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    logits = np.array([2.0, 1.0, 0.1])
    labels = ["token A", "token B", "token C"]
    temps = [0.1, 0.5, 1.0, 2.0]
    width = 0.45
    x = np.arange(3)
    for i, t in enumerate(temps):
        p = np.exp(logits / t)
        p = p / p.sum()
        ax.bar(x + i * 1.35, p, width=width, label=f"T = {t}")
        for j, v in enumerate(p):
            ax.text(x[j] + i * 1.35, v + 0.02, f"{v:.2f}", ha="center",
                    fontsize=7)
    ax.set_xticks(x + 1.35 * 1.5)
    ax.set_xticklabels(labels)
    ax.set_ylabel("P(token)")
    ax.set_title("Temperature reshapes the output distribution\n"
                 "P(token) = softmax(logits / T): low T -> sharp, high T -> flat",
                 fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "rag", "temperature_softmax.png")


def rag_diagrams():
    rag_lifecycle()
    rerank_funnel()
    hybrid_rrf()
    parent_child()
    agentic_rag()
    graph_vs_vector()
    rag_metrics()
    temperature_softmax()


# =============================================================================
# B) VIDEOS -> Videos/*.mp4
# =============================================================================

N_FRAMES = 48


def _writer():
    return FFMpegWriter(fps=12, codec="libx264", bitrate=600)


def _frame_mark(fig, f, n):
    fig.text(0.5, 0.97, f"frame {f + 1}/{n}", fontsize=8, ha="center",
             color="#667085")


def video_topk():
    """Query moves through semantic space; top-3 highlight shifts."""
    rng = np.random.default_rng(7)
    pts = rng.normal(size=(10, 2))
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.6, 4.6))
    fig.suptitle("Top-K retrieval: nearest neighbors move as the query changes",
                 fontsize=11)
    n = N_FRAMES
    qx = np.linspace(-2.2, 2.2, n)
    qy = np.linspace(2.2, -2.2, n)
    w = _writer()
    with w.saving(fig, os.path.join(VIDEOS, "topk_retrieval.mp4"), dpi=90):
        for f in range(n):
            q = np.array([qx[f], qy[f]])
            d = np.linalg.norm(pts - q, axis=1)
            order = np.argsort(d)
            axl.clear()
            axl.set_title("query + 10 chunks")
            axl.scatter(pts[:, 0], pts[:, 1], s=90, c="#9aa7b8")
            axl.scatter(pts[order[:3], 0], pts[order[:3], 1], s=110,
                        c="#2e7d32")
            axl.scatter(*q, marker="*", s=320, c="#b02a37", zorder=5)
            axl.set_xlim(-3.2, 3.2); axl.set_ylim(-3.2, 3.2)
            axl.grid(alpha=0.3)
            axr.clear()
            axr.set_title("distance -> top-3 green")
            axr.barh(np.arange(10)[::-1], d, color="#9aa7b8")
            axr.barh(9 - order[:3], d[order[:3]], color="#2e7d32")
            axr.set_yticks([])
            axr.set_xlim(0, 7.5)
            fig.tight_layout()
            _frame_mark(fig, f, n)
            fig.canvas.draw()
            w.grab_frame()


def video_bm25():
    """BM25 score vs term frequency: k1 saturation and b length norm."""
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    tf = np.linspace(0, 12, 200)
    n = 48
    k1s = np.linspace(0.5, 3.0, n)
    w = _writer()
    with w.saving(fig, os.path.join(VIDEOS, "bm25_parameters.mp4"), dpi=90):
        for f in range(n):
            k1 = k1s[f]
            ax.clear()
            for b, col, lab in ((0.0, "#1565c0", "b=0 (no length norm)"),
                                (0.75, "#b26a00", "b=0.75 (length norm)")):
                s = tf * (k1 + 1) / (tf + k1 * (1 - b + b * 0.8))
                ax.plot(tf, s, color=col, lw=2, label=lab)
            ax.set_title(f"BM25 term-frequency saturation, k1 = {k1:.2f}\n"
                         "higher k1 -> term frequency counts more before saturating",
                         fontsize=10.5)
            ax.set_xlabel("term frequency (tf)")
            ax.set_ylabel("BM25 score")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 4.6)
            fig.tight_layout()
            ax.text(0.5, 0.97, f"frame {f + 1}/{n}", transform=fig.transFigure,
                    fontsize=8, ha="center", color="#667085")
            fig.canvas.draw()
            w.grab_frame()


def video_rrf():
    """Two rank lists fuse into one via 1/(k+rank)."""
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    dense = [("d2", 1), ("d1", 2), ("d5", 3), ("d3", 4), ("d7", 5)]
    sparse = [("d7", 1), ("d4", 2), ("d2", 3), ("d9", 4), ("d1", 5)]
    n = N_FRAMES
    w = _writer()
    with w.saving(fig, os.path.join(VIDEOS, "rrf_fusion.mp4"), dpi=90):
        for f in range(n):
            k = 60 - int(40 * f / (n - 1))  # animate the k constant 60 -> 20
            ax.clear()
            acc = {}
            for lst, col in ((dense, "#1565c0"), (sparse, "#b26a00")):
                for rank, (c, _) in enumerate(lst, start=1):
                    acc[c] = acc.get(c, 0) + 1.0 / (k + rank)
            order = sorted(acc, key=lambda c: -acc[c])
            ys = np.arange(len(order))[::-1]
            ax.barh(ys, [acc[c] for c in order], color="#2e7d32", alpha=0.85)
            for y, c in zip(ys, order):
                ax.text(acc[c] + 0.002, y, f"{c}  {acc[c]:.4f}", va="center",
                        fontsize=8.5)
            ax.set_yticks([])
            ax.set_title(f"RRF fusion of dense + BM25 rankings (k = {k})\n"
                         "each list contributes 1/(k+rank) - no score normalization",
                         fontsize=10.5)
            ax.set_xlabel("fused RRF score")
            ax.grid(axis="x", alpha=0.3)
            ax.text(0.5, 0.97, f"frame {f + 1}/{n}", transform=fig.transFigure,
                    fontsize=8, ha="center", color="#667085")
            fig.tight_layout()
            fig.canvas.draw()
            w.grab_frame()


def video_temperature():
    """Softmax flattening as temperature sweeps up."""
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    logits = np.array([2.0, 1.0, 0.1])
    labels = ["token A", "token B", "token C"]
    n = N_FRAMES
    temps = np.linspace(0.15, 2.6, n)
    w = _writer()
    with w.saving(fig, os.path.join(VIDEOS, "temperature_sampling.mp4"),
                  dpi=90):
        for f in range(n):
            t = temps[f]
            p = np.exp(logits / t)
            p = p / p.sum()
            ax.clear()
            cols = ["#b02a37", "#1565c0", "#b26a00"]
            bars = ax.bar(labels, p, color=cols, alpha=0.9)
            for b, v in zip(bars, p):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                        ha="center", fontsize=9)
            top = int(np.argmax(p))
            ax.text(0.5, 0.90, f"most likely: {labels[top]}",
                    transform=ax.transAxes, ha="center", fontsize=10,
                    color=cols[top], fontweight="bold")
            ax.set_title(f"P(token) = softmax(logits / T),  T = {t:.2f}\n"
                         "low T -> greedy & deterministic; high T -> uniform",
                         fontsize=10.5)
            ax.set_ylim(0, 1.15)
            ax.grid(axis="y", alpha=0.3)
            ax.text(0.5, 0.97, f"frame {f + 1}/{n}", transform=fig.transFigure,
                    fontsize=8, ha="center", color="#667085")
            fig.tight_layout()
            fig.canvas.draw()
            w.grab_frame()


def video_chunking():
    """Chunk-size tradeoff: precision vs context."""
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    sizes = np.linspace(100, 1200, 200)
    precision = 1.0 - 0.45 * np.log1p(sizes / 100) / np.log1p(12)
    context = 0.15 + 0.85 * np.log1p(sizes / 100) / np.log1p(12)
    n = N_FRAMES
    w = _writer()
    with w.saving(fig, os.path.join(VIDEOS, "chunk_size_tradeoff.mp4"),
                  dpi=90):
        for f in range(n):
            i = int(f / (n - 1) * (len(sizes) - 1))
            ax.clear()
            ax.plot(sizes, precision, color="#1565c0", lw=2,
                    label="retrieval precision")
            ax.plot(sizes, context, color="#b26a00", lw=2,
                    label="context completeness")
            ax.axvline(sizes[i], color="#b02a37", ls="--", lw=1.5)
            ax.scatter([sizes[i]], [precision[i]], color="#1565c0", zorder=5)
            ax.scatter([sizes[i]], [context[i]], color="#b26a00", zorder=5)
            ax.text(sizes[i] + 15, 0.5,
                    f"{int(sizes[i])} tokens", fontsize=9, color="#b02a37")
            ax.set_title("Chunk size tradeoff: small = precise, "
                         "large = complete context\nno single size wins - "
                         "measure on your corpus", fontsize=10.5)
            ax.set_xlabel("chunk size (tokens)")
            ax.set_ylabel("score")
            ax.set_ylim(0, 1.15)
            ax.legend(fontsize=8, loc="center right")
            ax.grid(alpha=0.3)
            ax.text(0.5, 0.97, f"frame {f + 1}/{n}", transform=fig.transFigure,
                    fontsize=8, ha="center", color="#667085")
            fig.tight_layout()
            fig.canvas.draw()
            w.grab_frame()


def video_ann():
    """ANN recall vs search effort (probes): the recall-latency frontier."""
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    probes = np.arange(1, 65)
    recall = 1 - np.exp(-probes / 14)
    n = N_FRAMES
    w = _writer()
    with w.saving(fig, os.path.join(VIDEOS, "ann_recall_latency.mp4"), dpi=90):
        for f in range(n):
            i = int(f / (n - 1) * (len(probes) - 1))
            ax.clear()
            ax.plot(probes[:i + 1], recall[:i + 1], color="#2e7d32", lw=2.2)
            ax.scatter([probes[i]], [recall[i]], color="#b02a37", zorder=5)
            ax.axvline(14, color="#667085", ls=":", lw=1.2)
            ax.text(15, 0.1, "knee: most recall gained by probe ~14\n"
                    "(every extra probe = latency)", fontsize=8, color="#555")
            ax.set_title(f"ANN recall vs search effort (probe {int(probes[i])})\n"
                         "brute force = recall 1.0 but slow; ANN trades "
                         "recall for latency", fontsize=10.5)
            ax.set_xlabel("search effort (probes / efSearch)")
            ax.set_ylabel("recall@10")
            ax.set_ylim(0, 1.1)
            ax.grid(alpha=0.3)
            ax.text(0.5, 0.97, f"frame {f + 1}/{n}", transform=fig.transFigure,
                    fontsize=8, ha="center", color="#667085")
            fig.tight_layout()
            fig.canvas.draw()
            w.grab_frame()


def rag_videos():
    os.makedirs(VIDEOS, exist_ok=True)
    video_topk()
    video_bm25()
    video_rrf()
    video_temperature()
    video_chunking()
    video_ann()
    # README
    with open(os.path.join(VIDEOS, "README.txt"), "w", encoding="utf-8") as f:
        f.write("Videos generated by diagrams/generate_rag_media.py "
                "(matplotlib FFMpegWriter + ffmpeg).\n"
                "Each clip is ~4 s at 12 fps, 960x540.\n"
                "Regenerate: python diagrams/generate_rag_media.py\n")


# =============================================================================
# C) IMAGES FOLDER - populate Images/ with every diagram, by category
# =============================================================================

def populate_images():
    os.makedirs(IMAGES, exist_ok=True)
    total = 0
    for cat in sorted(os.listdir(DIAGRAMS)):
        src = os.path.join(DIAGRAMS, cat)
        if not os.path.isdir(src) or cat in ("gallery", "__pycache__"):
            continue
        dst = os.path.join(IMAGES, cat)
        os.makedirs(dst, exist_ok=True)
        for fn in sorted(os.listdir(src)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))
                total += 1
    with open(os.path.join(IMAGES, "README.txt"), "w", encoding="utf-8") as f:
        f.write(f"Images folder populated from diagrams/ by "
                f"generate_rag_media.py ({total} images, organized by "
                f"category).\nRegenerate: python diagrams/generate_rag_media.py\n")
    print(f"Images/: {total} images copied, organized by category")


if __name__ == "__main__":
    rag_diagrams()
    rag_videos()
    populate_images()
    print("done")