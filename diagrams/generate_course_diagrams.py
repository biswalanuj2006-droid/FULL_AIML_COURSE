"""
Generate diagrams + visualizations for the three course tracks:

  llm_course/            -> diagrams/llm/     (LLM internals, training, inference)
  genai_agents_course/   -> diagrams/agents/  (RAG agents, multi-agent, embedding bench)
  ml_course/             -> diagrams/ml/      (ML lifecycle + model diagnostics)

Data panels use the ACTUAL numbers produced by the course labs (scale_sweep_lab,
embedding_rag_lab, embedding_hf_bench, kv_decode_sweep_lab, mini_gpt_lab) so the
figures agree with the verified results in COURSE_INDEX.md / COURSE_AUDIT.txt.
Schematic panels are hand-drawn concept figures.

Run:  python diagrams/generate_course_diagrams.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "diagrams")


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


def arrow(ax, x1, y1, x2, y2, text=None, color="#333", lw=1.6):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.04, text, ha="center",
                fontsize=7.5, color="#555")


# =============================================================================
# LLM COURSE  (diagrams/llm/)
# =============================================================================

def llm_architecture():
    """Decoder-only LLM stack: embed + pos -> N transformer blocks -> LM head."""
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")
    ax.set_title("Decoder-only LLM architecture (GPT-style)\n"
                 "token stream in -> next-token distribution out", fontsize=12)
    cx = 5
    y = 10.6
    box(ax, cx - 2.2, y, 4.4, 1.0, "TOKEN EMBEDDINGS\nlookup table [V, d]", fc="#e3f2fd", ec="#1565c0")
    arrow(ax, cx, y, cx, y - 0.55)
    y -= 1.1
    box(ax, cx - 2.2, y, 4.4, 1.0, "POSITION INFO  (RoPE)\norder is injected, attention is order-free", fc="#e3f2fd", ec="#1565c0")
    arrow(ax, cx, y, cx, y - 0.55)
    y -= 1.1
    # N blocks
    block = FancyBboxPatch((cx - 3.0, y - 4.6), 6.0, 4.6, boxstyle="round,pad=0.04",
                           fc="#f5f5f5", ec="#333", lw=1.6)
    ax.add_patch(block)
    ax.text(cx, y + 0.15, "N transformer blocks (residual stream)", fontsize=9.5,
            ha="center", fontweight="bold", color="#333")
    by = y - 0.6
    box(ax, cx - 2.2, by, 4.4, 0.95, "x + MHA(LN(x))\nmulti-head self-attention (causal)", fc="#e8f5e9", ec="#2e7d32")
    arrow(ax, cx, by + 0.95, cx, by + 1.45)
    by += 1.5
    box(ax, cx - 2.2, by, 4.4, 0.95, "x + FFN(LN(x))\nLinear -> GELU/SwiGLU -> Linear", fc="#e8f5e9", ec="#2e7d32")
    arrow(ax, cx, by + 0.95, cx, by + 1.45)
    y -= 5.5
    box(ax, cx - 2.2, y, 4.4, 1.0, "FINAL NORM  (RMSNorm)\nstabilizes the grown residual stream", fc="#fff3e0", ec="#ef6c00")
    arrow(ax, cx, y, cx, y - 0.55)
    y -= 1.1
    box(ax, cx - 2.2, y, 4.4, 1.0, "LM HEAD  (linear to vocab)\nhidden [B,T,d] -> logits [B,T,V]", fc="#fce4ec", ec="#c62828")
    arrow(ax, cx, y, cx, y - 0.55)
    y -= 1.1
    box(ax, cx - 2.2, y, 4.4, 1.0, "SOFTMAX + SAMPLING\ntemperature / top-k / top-p -> next token", fc="#fce4ec", ec="#c62828")
    ax.text(cx, 0.4, "next token appended -> repeat (autoregressive)", fontsize=9,
            ha="center", style="italic")
    save(fig, "llm", "llm_architecture.png")


def llm_pretraining_pipeline():
    """Raw data -> ... -> checkpoint flow (llm_course Part 20)."""
    fig, ax = plt.subplots(figsize=(16, 4.6))
    ax.set_xlim(0, 16); ax.set_ylim(0, 4.6); ax.axis("off")
    ax.set_title("LLM pretraining pipeline (Part 20)", fontsize=13, fontweight="bold")
    steps = [
        (0.8, "RAW DATA\nweb / books / code", "#e3f2fd", "#1565c0"),
        (2.3, "FILTER\nlanguage, quality,\nPII", "#e3f2fd", "#1565c0"),
        (3.8, "CLEAN\nboilerplate,\nnormalize", "#e3f2fd", "#1565c0"),
        (5.3, "DEDUPLICATE\nexact + MinHash", "#e3f2fd", "#1565c0"),
        (6.8, "TOKENIZE\nsubword vocab", "#fff3e0", "#ef6c00"),
        (8.3, "PACK\nconcat to seq len", "#fff3e0", "#ef6c00"),
        (9.8, "SHUFFLE\nbreak correlation", "#fff3e0", "#ef6c00"),
        (11.3, "BATCH\n~0.5-4M tokens", "#fff3e0", "#ef6c00"),
        (12.8, "TRAIN\nnext-token CE\nAdamW + warmup", "#e8f5e9", "#2e7d32"),
        (14.3, "VALIDATE +\nCHECKPOINT\nholdout ppl", "#fce4ec", "#c62828"),
    ]
    for i, (x, t, fc, ec) in enumerate(steps):
        box(ax, x - 0.62, 1.5, 1.24, 1.6, t, fc=fc, ec=ec, fs=7.5)
        if i < len(steps) - 1:
            arrow(ax, x + 0.62, 2.3, steps[i + 1][0] - 0.62, 2.3)
    ax.text(8.0, 0.5, "each stage shapes the final distribution; dedup + mixture dominate quality",
            fontsize=9, ha="center", style="italic")
    save(fig, "llm", "pretraining_pipeline.png")


def llm_kv_cache():
    """KV cache: linear memory vs quadratic recompute (real measured speedups)."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    T = np.linspace(1, 1024, 200)
    recompute = T ** 2 / 1024.0          # quadratic work, normalized
    cached = T                           # linear work, normalized
    ax.plot(T, recompute, "r-", lw=2.5, label="no cache: recompute all past K,V (O(T^2))")
    ax.plot(T, cached, "g-", lw=2.5, label="KV cache: append one token (O(T))")
    # measured points from kv_decode_sweep_lab probes
    meas = [(128, 2.4), (256, 2.8), (512, 6.2), (768, 11.0)]
    for t, sp in meas:
        ax.plot(t, t, "ko", ms=4)
        ax.annotate(f"measured {sp:.1f}x", xy=(t, t), xytext=(t * 0.72, t * 1.7),
                    fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.8, color="#555"))
    ax.set_xlabel("generated sequence length T")
    ax.set_ylabel("relative work (log)")
    ax.set_yscale("log")
    ax.set_title("Why the KV cache exists: recompute is quadratic,\ncached decode is linear (llm_course Part 41)", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(True, alpha=0.3)
    save(fig, "llm", "kv_cache.png")


def llm_kv_memory():
    """KV cache memory growth (LLaMA-7B-like: 0.5 MiB/token fp16)."""
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ctx = np.array([1, 1024, 4096, 16384, 32768, 131072])
    gb = ctx * 0.5 / 1024.0  # 0.5 MiB/token -> GiB
    bars = ax.bar([f"{c//1024}k" if c >= 1024 else "1" for c in ctx], gb,
                  color=["#4a6fa5"] * 4 + ["#ef6c00", "#c62828"], alpha=0.9)
    for b, v in zip(bars, gb):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.1f} GiB",
                ha="center", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("KV cache size (GiB)")
    ax.set_title("KV cache memory: 0.5 MiB/token (L=32, H=32, d_h=128, fp16)\n"
                 "linear in context, which is why long context is expensive", fontsize=11)
    ax.set_ylim(0, max(gb) * 1.12)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "llm", "kv_cache_memory.png")


def llm_lora():
    """LoRA: W' = W + (alpha/r) B A schematic with dimensions."""
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.2); ax.axis("off")
    ax.set_title("LoRA: learn a low-rank update instead of the full matrix\n"
                 "W' = W + (alpha/r) B A   (llm_course Part 37)", fontsize=12)
    # W
    box(ax, 0.4, 1.7, 2.6, 2.2, "W  (frozen)\n[d_out, d_in]\nfull rank", fc="#eceff1", ec="#333")
    ax.text(1.7, 1.25, "e.g. 768x768\n= 589,824 params", fontsize=7.5, ha="center", color="#555")
    # + 
    ax.text(3.4, 2.8, "+", fontsize=18, ha="center", fontweight="bold")
    # B A
    box(ax, 3.9, 2.9, 2.4, 1.0, "B  [d_out, r]\ntrainable", fc="#e8f5e9", ec="#2e7d32")
    box(ax, 3.9, 1.6, 2.4, 1.0, "A  [r, d_in]\ntrainable", fc="#e8f5e9", ec="#2e7d32")
    ax.text(5.1, 1.15, "r=8: 768*8 + 8*768 = 12,288\n= 48x fewer trainable params",
            fontsize=7.5, ha="center", color="#2e7d32")
    # =
    ax.text(6.8, 2.8, "=", fontsize=18, ha="center", fontweight="bold")
    # W'
    box(ax, 7.2, 1.7, 3.4, 2.2, "W'  (used in forward)\n[W + (alpha/r) B A]\nA~N(0,sigma), B=0 at init\n-> step 0: W' = W exactly", fc="#fff3e0", ec="#ef6c00", fs=8.5)
    ax.text(5.5, 0.3, "apply to W_q, W_k, W_v, W_o and FFN matrices; only A, B get optimizer updates",
            fontsize=8.5, ha="center", style="italic")
    save(fig, "llm", "lora.png")


def llm_sampling():
    """Temperature / top-k / top-p: verified numbers from the course text."""
    fig = plt.figure(figsize=(11.5, 4.4))
    logits = np.array([2.0, 1.0, 0.1])
    labels = ["token A", "token B", "token C"]
    ax1 = fig.add_subplot(1, 3, 1)
    temps = [(1.0, "#4a6fa5"), (0.5, "#c62828"), (2.0, "#2e7d32")]
    width = 0.25
    for i, (t, c) in enumerate(temps):
        p = np.exp(logits / t); p = p / p.sum()
        ax1.bar([j + i * width - width for j in range(3)], p, width=width,
                color=c, label=f"T={t:g}", alpha=0.9)
        for j, v in enumerate(p):
            ax1.text(j + i * width - width, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    ax1.set_title("Temperature (logits [2, 1, 0.1])\nT<1 sharpens, T>1 flattens", fontsize=9.5)
    ax1.set_xticks(range(3)); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylim(0, 1.0); ax1.legend(fontsize=7.5); ax1.grid(axis="y", alpha=0.3)
    # top-k
    ax2 = fig.add_subplot(1, 3, 2)
    z = np.array([3.0, 2.5, 2.0, 0.5, 0.1, -0.5, -1.0, -2.0])
    p = np.exp(z); p = p / p.sum()
    keep = np.zeros_like(p)
    keep[np.argsort(z)[-3:]] = p[np.argsort(z)[-3:]]
    keep = keep / keep.sum()
    ax2.bar(range(8), p, color="#b0bec5", alpha=0.7, label="full softmax")
    ax2.bar(range(8), keep, color="#ef6c00", alpha=0.85, label="top-k kept (renorm)")
    ax2.set_title("top-k: keep the k highest\nlogits, zero + renormalize", fontsize=9.5)
    ax2.set_xticks(range(8)); ax2.tick_params(axis="x", labelsize=7)
    ax2.legend(fontsize=7.5); ax2.grid(axis="y", alpha=0.3)
    # top-p
    ax3 = fig.add_subplot(1, 3, 3)
    order = np.argsort(z)[::-1]
    cum = np.cumsum(p[order])
    p_keep = np.zeros_like(p)
    p_keep[order[cum <= 0.9]] = p[order[cum <= 0.9]]
    p_keep = p_keep / p_keep.sum()
    ax3.bar(range(8), p, color="#b0bec5", alpha=0.7, label="full softmax")
    ax3.bar(range(8), p_keep, color="#2e7d32", alpha=0.85, label="nucleus kept")
    ax3.set_title("top-p (nucleus): smallest set with\ncumulative prob >= p", fontsize=9.5)
    ax3.set_xticks(range(8)); ax3.tick_params(axis="x", labelsize=7)
    ax3.legend(fontsize=7.5); ax3.grid(axis="y", alpha=0.3)
    fig.suptitle("Text-generation sampling knobs (llm_course Part 16)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "llm", "sampling.png")


def llm_scaling_laws():
    """REAL numbers from scale_sweep_lab (450 steps, lr 1e-3)."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    sizes = ["S\n119k params", "M\n253k params", "L\n435k params"]
    losses = [2.723, 2.617, 2.421]
    ppl = [15.2, 13.7, 11.3]
    colors = ["#90caf9", "#4a6fa5", "#1a237e"]
    bars = ax.bar(sizes, losses, color=colors, alpha=0.9, edgecolor="black", lw=0.8)
    for b, v, p in zip(bars, losses, ppl):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}\n(ppl {p})",
                ha="center", fontsize=9, fontweight="bold")
    ax.axhline(2.763, ls="--", color="#c62828", lw=1.6)
    ax.text(2.42, 2.775, "bigram baseline 2.763 (ppl 15.9)", color="#c62828", fontsize=8.5)
    ax.set_ylabel("validation loss (nats, lower is better)")
    ax.set_title("Scaling laws on one corpus: bigger model -> lower loss\n"
                 "(real mini-GPT sweep, 450 steps, same seed/corpus)", fontsize=11)
    ax.set_ylim(2.3, 2.9)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "llm", "scaling_laws.png")


def llm_prefill_decode():
    """Prefill (compute-bound) vs decode (bandwidth-bound)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("PREFILL: process the whole prompt at once", fontsize=10.5, fontweight="bold")
    box(ax, 0.3, 3.2, 2.6, 1.6, "prompt tokens\n[B, T, d]\nall at once", fc="#e3f2fd", ec="#1565c0")
    box(ax, 4.0, 3.2, 2.6, 1.6, "one big matmul\n(compute-bound)\nhigh FLOP util", fc="#e8f5e9", ec="#2e7d32")
    box(ax, 7.7, 3.2, 2.0, 1.6, "KV cache\nfor the prompt", fc="#fff3e0", ec="#ef6c00")
    arrow(ax, 2.9, 4.0, 4.0, 4.0); arrow(ax, 6.6, 4.0, 7.7, 4.0)
    ax.text(5.0, 1.0, "time ~ prompt length; output = K,V for every prompt token\nlatency here = time-to-first-token", fontsize=8.5, ha="center")
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("DECODE: one token at a time", fontsize=10.5, fontweight="bold")
    box(ax, 0.3, 3.2, 2.6, 1.6, "one new token\n[B, 1, d]", fc="#e3f2fd", ec="#1565c0")
    box(ax, 4.0, 3.2, 2.6, 1.6, "matrix-vector\n(bandwidth-bound)\nreads all weights", fc="#fce4ec", ec="#c62828")
    box(ax, 7.7, 3.2, 2.0, 1.6, "append K,V\ncache grows", fc="#fff3e0", ec="#ef6c00")
    arrow(ax, 2.9, 4.0, 4.0, 4.0); arrow(ax, 6.6, 4.0, 7.7, 4.0)
    ax.text(5.0, 1.0, "per-token time ~ model size / memory bandwidth\n(7B at ~1 TB/s HBM: ~7 ms/token minimum)\nbatch streams amortize the weight reads", fontsize=8.5, ha="center")
    fig.suptitle("Two inference phases (llm_course Parts 40-42)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "llm", "prefill_decode.png")


def llm_quantization():
    """7B weights in fp32/fp16/int8/int4 (Part 39 memory math)."""
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    fmt = ["FP32\n4 B/param", "FP16/BF16\n2 B/param", "INT8\n1 B/param", "INT4\n0.5 B/param"]
    gb = [28, 14, 7, 3.5]
    colors = ["#b0bec5", "#4a6fa5", "#ef6c00", "#2e7d32"]
    bars = ax.bar(fmt, gb, color=colors, alpha=0.9, edgecolor="black", lw=0.8)
    for b, v in zip(bars, gb):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:g} GB",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("7B model weights (GB)")
    ax.set_title("Quantization halves memory per step (llm_course Part 39)\n"
                 "INT4 + LoRA = QLoRA: fine-tune a 7B on one consumer GPU", fontsize=11)
    ax.set_ylim(0, 32)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "llm", "quantization.png")


def llm_speculative():
    """Speculative decoding: draft k tokens, verify in one pass (Part 43)."""
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("Speculative decoding: identical output, ~2-3x faster decode\n"
                 "(llm_course Part 43 + speculative_decoding_lab)", fontsize=12)
    box(ax, 0.3, 2.6, 2.2, 1.4, "DRAFT model\n(small, fast)\nbigram / small GPT", fc="#e3f2fd", ec="#1565c0")
    box(ax, 3.3, 2.6, 2.2, 1.4, "propose gamma\ntokens\nk1 k2 k3 k4", fc="#e8f5e9", ec="#2e7d32")
    box(ax, 6.3, 2.6, 2.2, 1.4, "TARGET model\nverifies ALL in\nONE forward pass", fc="#fff3e0", ec="#ef6c00")
    box(ax, 9.3, 2.6, 2.4, 1.4, "accept prefix\nreject first wrong\n-> resample there", fc="#fce4ec", ec="#c62828")
    arrow(ax, 2.5, 3.3, 3.3, 3.3); arrow(ax, 5.5, 3.3, 6.3, 3.3); arrow(ax, 8.5, 3.3, 9.3, 3.3)
    ax.text(6.0, 1.2, "verification keeps the EXACT target distribution; the draft only saves time\n"
                      "target tokens accepted: k1 k2 k3  |  k4 wrong -> sample anew at position 4",
            fontsize=9, ha="center", family="monospace")
    save(fig, "llm", "speculative_decoding.png")


# =============================================================================
# GENAI + AGENTS COURSE  (diagrams/agents/)
# =============================================================================

def agent_loop():
    """Observe -> reason -> act -> observe loop (genai Part 26+)."""
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("The agent loop: an LLM alone is not an agent\n"
                 "(no loop, no tools, no effects - the loop makes it one)", fontsize=12)
    box(ax, 4.9, 5.4, 2.2, 1.3, "USER GOAL\n+ memory\n+ state", fc="#e3f2fd", ec="#1565c0")
    box(ax, 4.9, 3.3, 2.2, 1.3, "LLM BRAIN\nreason + plan\n+ reflection", fc="#f3e5f5", ec="#7b1fa2")
    box(ax, 8.6, 3.3, 2.2, 1.3, "TOOL CALL\nsearch / code /\nDB / calculator", fc="#e8f5e9", ec="#2e7d32")
    box(ax, 8.6, 0.7, 2.2, 1.3, "OBSERVATION\nresult text\n(verify it)", fc="#fff3e0", ec="#ef6c00")
    box(ax, 4.9, 0.7, 2.2, 1.3, "FINAL ANSWER\nor continue", fc="#fce4ec", ec="#c62828")
    arrow(ax, 6.0, 5.4, 6.0, 4.6)
    arrow(ax, 7.1, 4.0, 8.6, 4.0, text="structured tool_calls")
    arrow(ax, 9.7, 3.3, 9.7, 2.0, text="execute")
    arrow(ax, 8.6, 0.7, 7.1, 0.7, text="result")
    arrow(ax, 6.0, 0.7, 6.0, 3.3, text="assess: done?")
    arrow(ax, 6.0, 3.3, 6.0, 4.6, text="loop (max steps)")
    ax.text(5.5, 6.2, "memory", fontsize=8, color="#1565c0", ha="center")
    save(fig, "agents", "agent_loop.png")


def multi_agent():
    """Supervisor pattern from the genai course multi-agent section."""
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("Multi-agent: supervisor delegates, specialists execute\n"
                 "(supervisor / router patterns)", fontsize=12)
    box(ax, 4.9, 5.2, 2.2, 1.4, "SUPERVISOR\nroutes + plans\nshared state", fc="#f3e5f5", ec="#7b1fa2", fs=9)
    subs = [
        (0.4, 2.6, "RESEARCH\nAGENT\nsearch + RAG", "#e3f2fd", "#1565c0"),
        (3.3, 2.6, "CODING\nAGENT\nsandbox + tests", "#e8f5e9", "#2e7d32"),
        (6.2, 2.6, "DATA\nAGENT\nSQL + stats", "#fff3e0", "#ef6c00"),
        (9.1, 2.6, "WRITING\nAGENT\nsynthesis", "#fce4ec", "#c62828"),
    ]
    for x, y, t, fc, ec in subs:
        box(ax, x, y, 2.5, 1.5, t, fc=fc, ec=ec, fs=8.5)
        arrow(ax, 5.5, 5.2, x + 1.25, y + 1.5, text="delegate", lw=1.2)
        arrow(ax, x + 1.25, y + 1.5, 5.5, 5.2, text="result", lw=1.2)
    ax.text(6.0, 0.6, "each agent: its own loop + tools + memory; supervisor owns termination\n"
                      "failures: handoffs, shared-state races, agent isolation (sandbox per agent)",
            fontsize=8.5, ha="center", style="italic")
    save(fig, "agents", "multi_agent.png")


def rag_agent():
    """RAG agent: retrieval + grounded generation + tools (embedding_rag_lab shape)."""
    fig, ax = plt.subplots(figsize=(12.5, 6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.5); ax.axis("off")
    ax.set_title("RAG agent: answer grounded in retrieved documents\n"
                 "(genai_agents_course: embedding_rag_lab + rag_agent_server)", fontsize=12)
    box(ax, 0.3, 4.6, 2.2, 1.2, "USER QUERY\nnatural language", fc="#e3f2fd", ec="#1565c0")
    box(ax, 3.2, 4.6, 2.0, 1.2, "EMBED QUERY\n(query -> vector)", fc="#e8f5e9", ec="#2e7d32")
    box(ax, 5.9, 4.6, 2.2, 1.2, "RETRIEVE top-k\ncosine / ANN", fc="#fff3e0", ec="#ef6c00")
    box(ax, 8.8, 4.6, 2.0, 1.2, "RERANK\n(cross-encoder)", fc="#fff3e0", ec="#ef6c00")
    box(ax, 11.3, 4.6, 1.5, 1.2, "CONTEXT\n+ query", fc="#f3e5f5", ec="#7b1fa2", fs=8)
    arrow(ax, 2.5, 5.2, 3.2, 5.2); arrow(ax, 5.2, 5.2, 5.9, 5.2)
    arrow(ax, 8.1, 5.2, 8.8, 5.2); arrow(ax, 10.8, 5.2, 11.3, 5.2)
    # ingestion lane
    box(ax, 0.3, 2.2, 2.2, 1.2, "DOCS\nparse + chunk\n(overlap)", fc="#e3f2fd", ec="#1565c0")
    box(ax, 3.2, 2.2, 2.0, 1.2, "EMBED\nchunks", fc="#e8f5e9", ec="#2e7d32")
    box(ax, 5.9, 2.2, 2.2, 1.2, "VECTOR DB\nindex + metadata\nfilter", fc="#fff3e0", ec="#ef6c00")
    arrow(ax, 2.5, 2.8, 3.2, 2.8); arrow(ax, 5.2, 2.8, 5.9, 2.8)
    arrow(ax, 7.0, 3.4, 7.0, 4.6, text="query vectors", lw=1.2)
    # generation
    box(ax, 8.8, 1.4, 3.9, 1.4, "GROUNDED GENERATION\n\"answer only from context; cite;\\nif absent say I don't know\"", fc="#fce4ec", ec="#c62828", fs=8.5)
    arrow(ax, 12.05, 4.6, 12.05, 2.8)
    box(ax, 8.8, 0.1, 3.9, 1.0, "ANSWER + SOURCES\n(optionally tool-verified)", fc="#fce4ec", ec="#c62828", fs=8.5)
    arrow(ax, 10.75, 1.4, 10.75, 1.1)
    ax.text(6.2, 0.4, "eval retrieval AND generation separately: hit@1, MRR, faithfulness",
            fontsize=8.5, ha="center", style="italic")
    save(fig, "agents", "rag_agent.png")


def prod_rag_server():
    """Production RAG server layers (rag_agent_server_prod: auth, quota, cache, SQL)."""
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("Production RAG agent server (rag_agent_server_prod.py)\n"
                 "fastapi: roles, quotas, Redis-style cache, SQL request log", fontsize=11.5)
    box(ax, 5.4, 5.8, 2.2, 1.0, "CLIENT\nuser + API key", fc="#e3f2fd", ec="#1565c0")
    box(ax, 5.4, 4.2, 2.2, 1.0, "AUTH / RBAC\n401 / 403\nroles: admin,user,trial", fc="#f3e5f5", ec="#7b1fa2", fs=8.5)
    box(ax, 2.4, 4.2, 2.2, 1.0, "QUOTA / RATE LIMIT\n402 trial quota\n429 burst limit", fc="#fce4ec", ec="#c62828", fs=8.5)
    box(ax, 8.4, 4.2, 2.2, 1.0, "CACHE (Redis-style)\nTTL + LRU\nhit / miss / expiry", fc="#fff3e0", ec="#ef6c00", fs=8.5)
    box(ax, 5.4, 2.5, 2.2, 1.0, "RAG BRAIN\nembed + retrieve\n+ grounded answer", fc="#e8f5e9", ec="#2e7d32", fs=8.5)
    box(ax, 2.4, 2.5, 2.2, 1.0, "SQL REQUEST LOG\n(PostgreSQL-style)\nusage GROUP BY report", fc="#e3f2fd", ec="#1565c0", fs=8.5)
    box(ax, 8.4, 2.5, 2.2, 1.0, "PROMPT-INJECTION\nGUARD + validation", fc="#fce4ec", ec="#c62828", fs=8.5)
    box(ax, 5.4, 0.7, 2.2, 1.0, "METRICS + RESPONSE\nlatency, tokens,\nstreaming", fc="#e8f5e9", ec="#2e7d32", fs=8.5)
    arrow(ax, 6.5, 5.8, 6.5, 5.2)
    arrow(ax, 6.5, 4.2, 4.6, 4.7, text="check")
    arrow(ax, 6.5, 4.2, 8.4, 4.7, text="check")
    arrow(ax, 6.5, 4.2, 6.5, 3.5)
    arrow(ax, 4.6, 4.2, 4.6, 3.5)
    arrow(ax, 9.5, 4.2, 9.5, 3.5)
    arrow(ax, 6.5, 2.5, 6.5, 1.7)
    save(fig, "agents", "prod_rag_server.png")


def embedding_bench():
    """REAL numbers from embedding_rag_lab + embedding_hf_bench."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    groups = ["direct\nqueries", "paraphrase\nqueries"]
    lex = [6 / 9, 4 / 6]
    dense = [9 / 9, 6 / 6]
    x = np.arange(2)
    w = 0.3
    ax.bar(x - w / 2, lex, w, color="#b0bec5", label="lexical (keyword overlap)", edgecolor="black", lw=0.6)
    ax.bar(x + w / 2, dense, w, color="#2e7d32", label="dense embeddings (SVD / HF)", edgecolor="black", lw=0.6)
    for xi, v in zip(x - w / 2, lex):
        ax.text(xi, v + 0.02, f"{v:.0%}", ha="center", fontsize=9, fontweight="bold")
    for xi, v in zip(x + w / 2, dense):
        ax.text(xi, v + 0.02, f"{v:.0%}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.12); ax.set_ylabel("hit@1")
    ax.set_title("Retrieval hit@1 (real lab results)\n9 direct + 6 paraphrase queries", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    m = ["lexical", "SVD\n(PPMI + PCA)", "HF MiniLM\n(all-MiniLM-L6-v2)"]
    v = [6 / 9, 9 / 9, 9 / 9]
    colors = ["#b0bec5", "#4a6fa5", "#1a237e"]
    bars = ax.bar(m, v, color=colors, alpha=0.9, edgecolor="black", lw=0.7)
    for b, val in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.02, f"{val:.0%}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.12); ax.set_ylabel("hit@1")
    ax.set_title("Local SVD embeddings match a real\ntransformer encoder (offline bench)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Embeddings beat keyword overlap, especially on paraphrases", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "agents", "embedding_bench.png")


# =============================================================================
# ML COURSE  (diagrams/ml/)
# =============================================================================

def ml_lifecycle():
    """Full ML lifecycle (ml_course Module 0 / 46)."""
    fig, ax = plt.subplots(figsize=(16, 4.8))
    ax.set_xlim(0, 16); ax.set_ylim(0, 4.8); ax.axis("off")
    ax.set_title("The ML lifecycle: it does not end at training (ml_course Module 0)", fontsize=13, fontweight="bold")
    steps = [
        (0.75, "DATA\nCOLLECTION", "#e3f2fd", "#1565c0"),
        (2.15, "DATA\nCLEANING", "#e3f2fd", "#1565c0"),
        (3.55, "EDA", "#e3f2fd", "#1565c0"),
        (4.95, "FEATURE\nENGINEERING", "#e3f2fd", "#1565c0"),
        (6.35, "TRAIN / VAL /\nTEST SPLIT", "#fff3e0", "#ef6c00"),
        (7.75, "BASELINE\n(dummy)", "#fff3e0", "#ef6c00"),
        (9.15, "MODEL\nTRAINING", "#fff3e0", "#ef6c00"),
        (10.55, "HYPERPARAMETER\nOPTIMIZATION", "#fff3e0", "#ef6c00"),
        (11.95, "EVALUATION +\nERROR ANALYSIS", "#e8f5e9", "#2e7d32"),
        (13.35, "DEPLOY\n(API / batch)", "#e8f5e9", "#2e7d32"),
        (14.75, "MONITOR +\nRETRAIN", "#fce4ec", "#c62828"),
    ]
    for i, (x, t, fc, ec) in enumerate(steps):
        box(ax, x - 0.6, 1.6, 1.2, 1.5, t, fc=fc, ec=ec, fs=7.5)
        if i < len(steps) - 1:
            arrow(ax, x + 0.6, 2.35, steps[i + 1][0] - 0.6, 2.35)
    ax.annotate("", xy=(14.75, 1.15), xytext=(0.75, 1.15),
                arrowprops=dict(arrowstyle="-|>", color="#c62828", lw=1.4, connectionstyle="arc3,rad=-0.25"))
    ax.text(7.7, 0.7, "drift detection -> retraining decision -> loop (monitoring closes the circle)",
            fontsize=8.5, ha="center", color="#c62828", style="italic")
    save(fig, "ml", "ml_lifecycle.png")


def ml_learning_curves():
    """Train vs validation error across training-set size."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    size = np.linspace(200, 2000, 100)
    train = 0.25 + 0.45 * np.exp(-size / 350)
    val = 0.55 + 0.5 * np.exp(-size / 700)
    ax.plot(size, train, "b-", lw=2.5, label="training error")
    ax.plot(size, val, "r-", lw=2.5, label="validation error")
    ax.fill_between(size, 0, 0.6, alpha=0.08, color="red")
    ax.text(300, 0.3, "overfitting zone:\nbig gap", fontsize=9, color="#c62828")
    ax.axvspan(300, 700, alpha=0.06, color="green")
    ax.text(1750, 0.58, "gap closes as data grows", fontsize=9, color="#2e7d32", ha="center")
    ax.set_xlabel("training-set size")
    ax.set_ylabel("error")
    ax.set_title("Learning curves diagnose under/overfitting\n(ml_course Module 26 / error analysis)", fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    save(fig, "ml", "learning_curves.png")


def ml_kmeans():
    """K-means clusters + elbow curve (Module 20)."""
    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_subplot(1, 2, 1)
    rng = np.random.default_rng(0)
    centers = [(2, 2), (6, 2), (4, 6)]
    X = np.vstack([rng.normal(c, 0.55, (60, 2)) for c in centers])
    # k-means (2 iterations of assignment/update for display)
    k = 3
    idx = rng.choice(len(X), k, replace=False)
    cents = X[idx]
    for _ in range(8):
        d = ((X[:, None, :] - cents[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        cents = np.array([X[lab == j].mean(0) for j in range(k)])
    cols = ["#4a6fa5", "#2e7d32", "#ef6c00"]
    for j in range(k):
        ax.scatter(X[lab == j, 0], X[lab == j, 1], c=cols[j], alpha=0.6, s=18,
                   edgecolors="black", linewidths=0.3)
    ax.scatter(cents[:, 0], cents[:, 1], c="black", marker="X", s=130, zorder=5)
    ax.set_title("K-means: Lloyd's algorithm\n(assign -> update centroids)", fontsize=10)
    ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax = fig.add_subplot(1, 2, 2)
    ks = np.arange(1, 9)
    inert = [float(((X - X.mean(0)) ** 2).sum())]
    for kk in ks[1:]:
        idx2 = rng.choice(len(X), kk, replace=False)
        cc = X[idx2]
        for _ in range(8):
            dd = ((X[:, None, :] - cc[None, :, :]) ** 2).sum(-1)
            ll = dd.argmin(1)
            cc = np.array([X[ll == j].mean(0) for j in range(kk)])
        inert.append(float(((X - cc[ll]) ** 2).sum()))
    ax.plot(ks, inert, "o-", color="#4a6fa5", lw=2)
    ax.axvline(3, ls="--", color="#c62828", lw=1.5)
    ax.text(3.05, inert[-1], "elbow: k=3", fontsize=9, color="#c62828")
    ax.set_xlabel("k"); ax.set_ylabel("inertia (within-cluster SSE)")
    ax.set_title("Elbow method: choose k at the knee", fontsize=10)
    ax.set_xticks(ks); ax.grid(True, alpha=0.3)
    fig.suptitle("K-means clustering (ml_course Module 20)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "ml", "kmeans.png")


def ml_pca():
    """PCA: projection + explained variance (Module 21)."""
    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_subplot(1, 2, 1)
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (250, 2)) @ np.array([[3.0, 1.0], [0.5, 1.2]])
    X -= X.mean(0)
    cov = np.cov(X.T)
    w, v = np.linalg.eigh(cov)
    pc1 = v[:, -1]
    proj = (X @ pc1)[:, None] * pc1[None, :]
    ax.scatter(X[:, 0], X[:, 1], c="#b0bec5", alpha=0.6, s=16, label="original")
    ax.scatter(proj[:, 0], proj[:, 1], c="#c62828", alpha=0.7, s=16, label="projected (PC1)")
    ax.axline((0, 0), pc1, color="#1a237e", lw=2.2)
    ax.set_title("PCA: project onto the direction of\nmaximum variance (Module 21)", fontsize=10)
    ax.set_aspect("equal"); ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)
    ax = fig.add_subplot(1, 2, 2)
    ev = np.linalg.eigvalsh(np.cov(X.T))[::-1]
    pv = ev / ev.sum()
    ax.bar(["PC1", "PC2"], pv, color=["#1a237e", "#b0bec5"], alpha=0.9, edgecolor="black", lw=0.6)
    for b, v in zip(ax.patches, pv):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("explained variance")
    ax.set_title("Explained variance: PC1 keeps most\ninformation (dimension reduction)", fontsize=10)
    ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Dimensionality reduction with PCA", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "ml", "pca.png")


def ml_pr_curve():
    """Precision-recall curve (Module 14 metrics)."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    rng = np.random.default_rng(2)
    score = rng.normal(0, 1, 2000)
    y = (rng.random(2000) < 0.2).astype(float)
    y = np.where(score + rng.normal(0, 1.3, 2000) > 0.5, 1.0, y)
    thresh = np.linspace(score.max() + 0.01, score.min() - 0.01, 200)
    prec, rec = [], []
    for t in thresh:
        tp = ((score >= t) & (y == 1)).sum()
        fp = ((score >= t) & (y == 0)).sum()
        fn = ((score < t) & (y == 1)).sum()
        prec.append(tp / max(tp + fp, 1))
        rec.append(tp / max(tp + fn, 1))
    auc = np.trapezoid(prec, rec)
    ax.plot(rec, prec, "b-", lw=2.5, label=f"PR curve (AUC-PR={auc:.3f})")
    ax.fill_between(rec, prec, alpha=0.15, color="blue")
    base = 0.2
    ax.axhline(base, ls="--", color="#c62828", lw=1.4)
    ax.text(0.02, base + 0.02, f"positive-class baseline = {base:.0%}", color="#c62828", fontsize=8.5)
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Precision-recall: the right curve for rare classes\n(fraud, defects, churn - Module 14)", fontsize=10.5)
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    save(fig, "ml", "pr_curve.png")


def ml_imbalance():
    """Class imbalance + SMOTE concept (Module 23)."""
    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_subplot(1, 2, 1)
    ax.bar(["majority", "minority"], [900, 100], color=["#4a6fa5", "#c62828"], alpha=0.9, edgecolor="black", lw=0.7)
    for b, v in zip(ax.patches, [900, 100]):
        ax.text(b.get_x() + b.get_width() / 2, v + 10, f"{v}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("samples")
    ax.set_title("Imbalanced data: 90/10\n(accuracy can hide a useless model)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax = fig.add_subplot(1, 2, 2)
    rng = np.random.default_rng(3)
    maj = rng.normal((2, 2), 0.8, (60, 2))
    mino = rng.normal((5, 5), 0.6, (12, 2))
    ax.scatter(maj[:, 0], maj[:, 1], c="#4a6fa5", alpha=0.6, s=20, label="majority")
    ax.scatter(mino[:, 0], mino[:, 1], c="#c62828", alpha=0.85, s=30, label="minority")
    synth = []
    for _ in range(24):
        a, b = mino[rng.integers(0, len(mino), 2)]
        synth.append(a + rng.random() * (b - a))
    synth = np.array(synth)
    ax.scatter(synth[:, 0], synth[:, 1], c="#ef6c00", marker="^", s=34, alpha=0.9,
               label="SMOTE synthetic")
    ax.set_title("SMOTE: synthesize minority points on\nedges between real neighbors (Module 23)", fontsize=10)
    ax.set_aspect("equal"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle("Dealing with class imbalance", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "ml", "imbalance.png")


def ml_feature_importance():
    """Permutation/SHAP-style feature importance (Module 28)."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    feats = ["annual income", "credit history", "loan amount", "debt ratio", "age",
             "employment years", "dependents", "region"]
    imp = [0.31, 0.24, 0.17, 0.11, 0.08, 0.05, 0.03, 0.01]
    bars = ax.barh(feats[::-1], imp, color="#4a6fa5", alpha=0.9, edgecolor="black", lw=0.6)
    for b, v in zip(bars, imp[::-1]):
        ax.text(v + 0.004, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=8.5)
    ax.set_xlabel("permutation importance (drop in score when shuffled)")
    ax.set_title("Feature importance: which inputs actually drive predictions\n"
                 "(permutation / SHAP - Module 28)", fontsize=10.5)
    ax.set_xlim(0, 0.36)
    ax.grid(axis="x", alpha=0.3)
    save(fig, "ml", "feature_importance.png")


def ml_time_series():
    """Time-series forecast: train/test + prediction (Module 25)."""
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    rng = np.random.default_rng(4)
    t = np.arange(120)
    seasonal = 3 * np.sin(2 * np.pi * t / 12)
    trend = 0.06 * t
    y = seasonal + trend + rng.normal(0, 0.8, 120)
    train_t, train_y = t[:90], y[:90]
    # naive seasonal forecast
    fc = [y[i - 12] for i in range(90, 120)]  # lag-12 seasonal persistence
    ax.plot(t, y, color="#b0bec5", lw=1.4, label="actual (with seasonality + trend)")
    ax.plot(train_t, train_y, color="#4a6fa5", lw=1.4)
    ax.axvline(90, ls="--", color="#c62828", lw=1.4)
    ax.text(91.5, 3.5, "train | test", fontsize=9, color="#c62828")
    ax.plot(t[90:], fc, "g--", lw=2, label="forecast (lag-12 seasonal)")
    ax.set_xlabel("time step")
    ax.set_ylabel("value")
    ax.set_title("Time-series ML: never shuffle, use temporal validation\n"
                 "(Module 25 - lag/rolling features + TimeSeriesSplit)", fontsize=10.5)
    ax.legend(fontsize=8.5); ax.grid(True, alpha=0.3)
    save(fig, "ml", "time_series.png")


if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING COURSE TRACK DIAGRAMS (llm / agents / ml)")
    print("=" * 60)
    print("\n--- llm_course (diagrams/llm/) ---")
    llm_architecture(); llm_pretraining_pipeline(); llm_kv_cache()
    llm_kv_memory(); llm_lora(); llm_sampling(); llm_scaling_laws()
    llm_prefill_decode(); llm_quantization(); llm_speculative()
    print("\n--- genai_agents_course (diagrams/agents/) ---")
    agent_loop(); multi_agent(); rag_agent(); prod_rag_server(); embedding_bench()
    print("\n--- ml_course (diagrams/ml/) ---")
    ml_lifecycle(); ml_learning_curves(); ml_kmeans(); ml_pca()
    ml_pr_curve(); ml_imbalance(); ml_feature_importance(); ml_time_series()
    count = 0
    for root, _dirs, files in os.walk(OUT):
        for f in files:
            if f.endswith((".jpg", ".png")):
                count += 1
    print("\n" + "=" * 60)
    print(f"done - total images in diagrams/: {count}")
    print("=" * 60)