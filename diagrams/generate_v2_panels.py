"""
generate_v2_panels.py - SECOND WAVE VISUAL PANELS
=================================================
Fills the highest-value gaps flagged by VISUAL_CONTENT_MASTER_PLAN.txt
(sections 0 + 8) using the repo's established matplotlib style
(FancyBboxPatch boxes, colored arrows, save-to-diagrams/<cat>):

  transformers/attention_mask.png   causal self-attention mask (row i -> col <= i)
  transformers/mqa_gqa.png          MQA / GQA / MHA head + KV sharing comparison
  transformers/ffn_glu.png          FFN vs SwiGLU: gated activation path
  ml/overfit_gap.png                train vs val curves with bias/variance regions
  math/broadcasting.png             NumPy broadcasting (3,1)+(1,4)->(3,4)
  nlp/tokenization.png              raw text -> tokens -> ids -> embedding rows

Run:  python diagrams/generate_v2_panels.py
Then: python diagrams/generate_rag_media.py   (refresh Images/ copies)
      python diagrams/generate_gallery.py     (refresh gallery)
      python diagrams/verify_diagrams.py      (validity check)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAGRAMS = os.path.join(ROOT, "diagrams")
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


# -----------------------------------------------------------------------------
# transformers/attention_mask.png - causal masking
# -----------------------------------------------------------------------------
def attention_mask():
    n = 6
    rng = np.random.default_rng(3)
    S = rng.uniform(0.25, 1.0, (n, n))
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    S[mask] = 0.0
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    # attention grid
    gx0, gy0, gw = 0.12, 0.52, 0.42
    ax_im = fig.add_axes([gx0, gy0, gw, gw])
    ax_im.imshow(S, cmap="Blues", vmin=0, vmax=1)
    ax_im.set_xticks(range(n)); ax_im.set_yticks(range(n))
    ax_im.set_xticklabels([f"k{i}" for i in range(1, n + 1)], fontsize=8)
    ax_im.set_yticklabels([f"q{i}" for i in range(1, n + 1)], fontsize=8)
    ax_im.tick_params(length=0)
    ax_im.set_xlabel("KEY position (attended to)", fontsize=8.5)
    ax_im.set_ylabel("QUERY position", fontsize=8.5)
    ax_im.set_title("Causal mask: row q attends only to k <= q\n"
                    "(upper triangle forced to -inf -> softmax 0)",
                    fontsize=9)
    for i in range(n):
        for j in range(i, n):
            if j > i:
                ax_im.text(j, i, "0", ha="center", va="center", fontsize=7,
                           color="#b02a37")
            else:
                ax_im.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center",
                           fontsize=6.5, color="black")
    # token flow on the right
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(0.66, 0.98, "why it matters", transform=ax.transAxes,
            fontsize=9, fontweight="bold", color="#b02a37")
    ax.text(6.1, 8.6, "decoder-only LM predicts token t+1\n"
                      "from tokens 1..t ONLY - the future must\n"
                      "stay invisible or training leaks answers",
            fontsize=7.5, va="top")
    arrow(ax, 6.1, 7.3, 5.3, 6.2, text="q4 sees k1..k4")
    # softmax annotation
    ax.text(6.1, 4.6, "row softmax over allowed keys:\n"
                      "P(k | q) proportional to exp(q.k/sqrt d)",
            fontsize=7.5, va="top")
    ax.text(6.1, 2.6, "same matrix powers the KV cache:\n"
                      "row t is the ONLY new row at decode step t",
            fontsize=7.5, va="top")
    save(fig, "transformers", "attention_mask.png")


# -----------------------------------------------------------------------------
# transformers/mqa_gqa.png - KV sharing across heads
# -----------------------------------------------------------------------------
def mqa_gqa():
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.6); ax.axis("off")
    ax.set_title("MHA vs MQA vs GQA: how many K/V copies does the KV cache keep?\n"
                 "(Q heads are always per-head; the cache stores K and V only)",
                 fontsize=11.5)
    def panel(x0, title, nq, nkv, label, fc="#eef2fb"):
        box(ax, x0, 6.2, 3.8, 0.9, title, fc=fc, ec="#4a6fa5", fs=9.5)
        # query heads
        for i in range(nq):
            b = FancyBboxPatch((x0 + 0.2 + i * 0.85, 4.2), 0.6, 0.9,
                               boxstyle="round,pad=0.01", fc="#e3f2fd",
                               ec="#1565c0", lw=1.2)
            ax.add_patch(b)
        ax.text(x0 + 0.2, 5.35, "Q heads" if nq <= 4 else "Q heads x8 (two shown)",
                fontsize=6.5, color="#1565c0")
        # kv heads
        for i in range(nkv):
            b = FancyBboxPatch((x0 + 0.2 + i * 1.7, 1.6), 0.6, 0.9,
                               boxstyle="round,pad=0.01", fc="#eaf6ec",
                               ec="#2e7d32", lw=1.2)
            ax.add_patch(b)
        ax.text(x0 + 0.2, 0.8, "shared K/V heads", fontsize=6.5,
                color="#2e7d32")
        # cache bar
        ax.barh(0.0, nkv, height=0.35, left=x0 + 0.2, color="#c8e6c9")
        ax.text(x0 + 0.2 + nkv + 0.1, 0.0, f"cache writes\nx{nkv}",
                fontsize=6.5, va="center", color="#2e7d32")
        # per-head attention wiring
        for qi in range(nq):
            kv = qi % nkv
            ax.plot([x0 + 0.5 + qi * 0.85, x0 + 0.5 + kv * 1.7],
                    [4.2, 2.5], color="#9aa5b1", lw=0.7,
                    alpha=0.7)
        ax.text(x0, 6.9, label, fontsize=7.5, color="#555")
    panel(0.3, "MULTI-HEAD ATTENTION", 4, 4, "each Q head has its own K/V\n"
          "-> cache x4 (x8/64/128 heads)\nbest quality, most memory")
    panel(5.1, "MULTI-QUERY ATTENTION", 4, 1, "ALL Q heads share ONE K/V\n"
          "-> cache x1, biggest saving\nquality can drop at scale")
    panel(9.9, "GROUPED-QUERY ATTENTION", 4, 2, "heads grouped: each group\n"
          "shares one K/V -> cache x2\n(8:1, 4:1, 2:1 configs)")
    ax.text(0.3, -0.55,
            "KV cache size = 2 x n_layers x n_kv_heads x T x d_head x bytes "
            "(GQA cuts the n_kv_heads factor vs MHA)",
            fontsize=7.5, color="#555")
    save(fig, "transformers", "mqa_gqa.png")


# -----------------------------------------------------------------------------
# ml/overfit_gap.png - train vs val, bias/variance regimes
# -----------------------------------------------------------------------------
def overfit_gap():
    rng = np.random.default_rng(11)
    x = np.linspace(0.05, 0.95, 60)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + rng.normal(0, 0.25, size=60)
    # model capacity sweep via polynomial degrees
    degs = [1, 3, 9, 18]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    colors = ["#1565c0", "#2e7d32", "#b26a00", "#b02a37"]
    for deg, c in zip(degs, colors):
        coeffs = np.polyfit(x, y, deg)
        xs = np.linspace(0, 1, 300)
        ax.plot(xs, np.polyval(coeffs, xs), color=c, lw=2,
                label=f"poly deg {deg}")
    ax.scatter(x, y, s=8, color="#9aa5b1", alpha=0.6, label="data (sin + noise)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Overfitting: more capacity fits the noise\n"
                 "deg 1 underfits (bias), deg 18 chases every point (variance)",
                 fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(alpha=0.25)
    # regimes inset
    ax.text(0.98, 0.02,
            "train err:   decr as deg grows\n"
            "val err:     U-shape -> pick the elbow\n"
            "high bias  = both errs high\n"
            "high variance = train << val",
            transform=ax.transAxes, fontsize=7.8, va="bottom", ha="right",
            bbox=dict(boxstyle="round", fc="#fff7e6", ec="#b26a00", alpha=0.9))
    save(fig, "ml", "overfit_gap.png")


# -----------------------------------------------------------------------------
# math/broadcasting.png - NumPy (3,1)+(1,4)
# -----------------------------------------------------------------------------
def broadcasting():
    a = np.arange(3).reshape(3, 1)
    b = np.arange(4).reshape(1, 4)
    c = a + b
    fig = plt.figure(figsize=(10, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.2, 1.6], wspace=0.35)
    axa = fig.add_subplot(gs[0])
    axa.imshow(a, cmap="Reds")
    for i in range(3):
        axa.text(0, i, f"{a[i,0]}", ha="center", va="center", color="white",
                 fontsize=11, fontweight="bold")
    axa.set_title("A  (3,1)", fontsize=10)
    axa.set_xticks([]); axa.set_yticks([])
    axb = fig.add_subplot(gs[1])
    axb.imshow(b, cmap="Blues")
    for j in range(4):
        axb.text(j, 0, f"{b[0,j]}", ha="center", va="center", color="white",
                 fontsize=11, fontweight="bold")
    axb.set_title("B  (1,4)", fontsize=10)
    axb.set_xticks([]); axb.set_yticks([])
    axc = fig.add_subplot(gs[2])
    axc.imshow(c, cmap="Purples")
    for i in range(3):
        for j in range(4):
            axc.text(j, i, f"{c[i,j]}", ha="center", va="center",
                     color="white", fontsize=9)
    axc.set_title("A + B  (3,4)", fontsize=10)
    axc.set_xticks([]); axc.set_yticks([])
    fig.suptitle("NumPy broadcasting: stretch the size-1 axes, then add elementwise\n"
                 "no copies are materialized - the stride repeats the row/column",
                 fontsize=11.5)
    save(fig, "math", "broadcasting.png")


# -----------------------------------------------------------------------------
# nlp/tokenization.png - raw text -> tokens -> ids -> embeddings
# -----------------------------------------------------------------------------
def tokenization():
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6.8); ax.axis("off")
    ax.set_title("Tokenization: raw text -> token ids -> embedding rows\n"
                 "(BPE/WordPiece split frequent words, keep rare words whole)",
                 fontsize=11.5)
    box(ax, 0.4, 4.9, 2.6, 1.2, "RAW TEXT\n\"the cat sat on the mat\"",
        fc="#fff7e6", ec="#b26a00")
    toks = ["the", "cat", "sat", "on", "the", "mat"]
    ids = [971, 5070, 7252, 336, 971, 4419]
    # token boxes
    for i, (t, tid) in enumerate(zip(toks, ids)):
        x0 = 3.5 + i * 1.45
        box(ax, x0, 4.9, 1.3, 0.55, t, fc="#e3f2fd", ec="#1565c0", fs=9)
        box(ax, x0, 4.15, 1.3, 0.5, str(tid), fc="#eef2fb", ec="#9aa5b1",
            fs=7.5)
    arrow(ax, 3.0, 5.5, 3.5, 5.4, "tokenize")
    # embedding table
    box(ax, 0.4, 1.3, 13.2, 1.7,
        "EMBEDDING TABLE  vocab_size x d_model  (each row = learned vector)\n"
        "row[971] = vector('the')   row[5070] = vector('cat')  ...\n"
        "'the' appears twice -> SAME row -> SAME vector -> model sees position via "
        "positional encoding",
        fc="#eaf6ec", ec="#2e7d32", fs=8.5)
    arrow(ax, 3.5 + 1 * 1.45, 4.15, 6.3, 3.0, "lookup")
    # vocab slice
    box(ax, 9.6, 0.2, 4.0, 0.7, "vocab: 50k tokens (BPE merge rules)",
        fc="#ede7f6", ec="#5e35b1", fs=7.5)
    ax.text(0.4, 0.05, "why tokenization matters: context cost, chunk limits,\n"
            "vocabulary size, and every token = 1 step of generation",
            fontsize=7.5, color="#555")
    save(fig, "nlp", "tokenization.png")


# -----------------------------------------------------------------------------
def main():
    attention_mask()
    mqa_gqa()
    overfit_gap()
    broadcasting()
    tokenization()
    print("v2 panels done -> now run generate_rag_media.py (Images/) and "
          "generate_gallery.py (gallery)")


if __name__ == "__main__":
    main()
