"""
generate_v3_panels.py - THIRD WAVE VISUAL PANELS + VIDEOS
=========================================================
Second batch from VISUAL_CONTENT_MASTER_PLAN.txt (sections 0 + 8):
fills the LangChain / LangGraph / Agents / MLOps / Security / PyTorch gaps
using the repo's established matplotlib style.

Panels  -> diagrams/langchain, langgraph, agents, mlops, security, pytorch
Videos  -> Videos/*.mp4        (12 fps, ~4 s, FFMpegWriter)

Run:  python diagrams/generate_v3_panels.py
Then: python diagrams/generate_rag_media.py   (refresh Images/ copies)
      python diagrams/generate_gallery.py     (refresh gallery)
      python diagrams/verify_diagrams.py      (validity check)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAGRAMS = os.path.join(ROOT, "diagrams")
VIDEOS = os.path.join(ROOT, "Videos")
NPIX = 140
N_FRAMES = 48


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


def _writer():
    return FFMpegWriter(fps=12, codec="libx264", bitrate=600)


def _frame_mark(fig, f, n):
    fig.text(0.5, 0.97, f"frame {f + 1}/{n}", fontsize=8, ha="center",
             color="#667085")


# =============================================================================
# PANELS
# =============================================================================

# --- langchain/lcel_runnable.png --------------------------------------------
def lcel_runnable():
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("LangChain LCEL: every piece is a Runnable with .invoke/.batch/.stream\n"
                 "prompt | model | parser  composed with the | pipe operator",
                 fontsize=11.5)
    box(ax, 0.5, 5.6, 3.2, 1.1, "ChatPromptTemplate\n(values fill {} slots)", fc="#fff7e6", ec="#b26a00")
    box(ax, 5.0, 5.6, 3.4, 1.1, "ChatModel / LLM\n(.invoke -> AIMessage)", fc="#eef2fb", ec="#4a6fa5")
    box(ax, 10.0, 5.6, 3.5, 1.1, "Pydantic output parser\n(.parse -> typed object)", fc="#eaf7ea", ec="#2e7d32")
    arrow(ax, 3.7, 6.15, 5.0, 6.15, "prompt value")
    arrow(ax, 8.4, 6.15, 10.0, 6.15, "AIMessage")
    ax.text(7, 4.7, "chain = prompt | model | parser", ha="center",
            fontsize=11, family="monospace", color="#333",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f4f6fa", ec="#999"))
    ax.text(7, 3.6, "chain.invoke({'topic': 'rag'})    chain.stream(...)    "
                    "chain.batch([...])",
            ha="center", fontsize=9, family="monospace", color="#555")
    box(ax, 0.5, 1.2, 6.0, 1.7, "RunnablePassthrough - keep the input\n"
        "RunnableLambda - wrap any python fn\nRunnableParallel - run branches concurrently",
        fc="#f6f2fb", ec="#6a4aa5", fs=8.5)
    box(ax, 7.5, 1.2, 6.0, 1.7, "why it matters:\none interface -> swapping the model,\n"
        "adding retries, or streaming requires\nzero changes to the chain shape",
        fc="#eaf7ea", ec="#2e7d32", fs=8.5)
    save(fig, "langchain", "lcel_runnable.png")


# --- langchain/structured_output.png ----------------------------------------
def structured_output():
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Structured output: LLM text -> validated typed object",
                 fontsize=11.5)
    box(ax, 0.5, 5.4, 3.4, 1.4, 'class Answer(BaseModel):\n    verdict: str\n    score: float', fc="#f6f2fb", ec="#6a4aa5", fs=8.5)
    box(ax, 5.0, 5.4, 4.2, 1.4, "model.with_structured_output(Answer)\n(schema shipped in the request)",
        fc="#eef2fb", ec="#4a6fa5", fs=8.5)
    box(ax, 10.2, 5.4, 3.3, 1.4, "Answer(verdict=...,\n    score=0.87)\nvalidated instance", fc="#eaf7ea", ec="#2e7d32", fs=8.5)
    arrow(ax, 3.9, 6.1, 5.0, 6.1)
    arrow(ax, 9.2, 6.1, 10.2, 6.1)
    ax.text(7, 4.1, "no schema = free text you must regex;\nschema-in = JSON the parser can trust",
            ha="center", fontsize=9, color="#b02a37")
    box(ax, 0.6, 1.0, 6.2, 2.2, "failure modes to teach:\n- field hallucinated / missing\n"
        "- type mismatch (str vs float)\n- retry loop on validation error\n"
        "- temperature>0 -> flaky schemas",
        fc="#fff2f2", ec="#b02a37", fs=8.5)
    box(ax, 7.4, 1.0, 6.0, 2.2, "engineering rules:\n- keep schemas small and flat\n"
        "- use enums for finite sets\n- validate at the boundary\n- log raw output on parse failure",
        fc="#eaf7ea", ec="#2e7d32", fs=8.5)
    save(fig, "langchain", "structured_output.png")


# --- langgraph/graph_state.png ----------------------------------------------
def graph_state():
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis("off")
    ax.set_title("LangGraph: a state machine - nodes are functions, edges are transitions,\n"
                 "the shared State flows through every node (reducer merges writes)",
                 fontsize=11.5)
    box(ax, 0.4, 7.4, 2.0, 0.9, "START", fc="#eef2fb", ec="#4a6fa5", fs=9)
    box(ax, 4.0, 7.3, 3.2, 1.1, "node: retrieve\n(reads/writes State)", fc="#fff7e6", ec="#b26a00", fs=8.5)
    box(ax, 8.8, 7.3, 3.2, 1.1, "node: grade_docs\n(conditional edge out)", fc="#fff7e6", ec="#b26a00", fs=8.5)
    box(ax, 4.0, 4.4, 3.2, 1.1, "node: rewrite_q\n(loop back if docs bad)", fc="#fff7e6", ec="#b26a00", fs=8.5)
    box(ax, 8.8, 4.4, 3.2, 1.1, "node: generate\n(answer + citations)", fc="#fff7e6", ec="#b26a00", fs=8.5)
    box(ax, 12.3, 1.0, 1.5, 0.9, "END", fc="#eef2fb", ec="#4a6fa5", fs=9)
    arrow(ax, 2.4, 7.85, 4.0, 7.85)
    arrow(ax, 7.2, 7.85, 8.8, 7.85)
    arrow(ax, 10.4, 7.3, 10.4, 5.5, "docs relevant?")
    arrow(ax, 8.8, 4.95, 7.2, 4.95, "no -> rewrite")
    arrow(ax, 5.6, 5.5, 5.6, 7.3, "retry (max 2)")
    arrow(ax, 12.0, 4.95, 13.05, 1.9, "yes ->")
    ax.text(0.5, 2.9, "State = TypedDict(messages=[...], docs=[...], loop_count=0)\n"
            "reducer (operator.add / overwrite) decides how concurrent writes merge",
            fontsize=8.5, family="monospace", color="#333",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f4f6fa", ec="#999"))
    ax.text(0.5, 1.6, "checkpointer -> every super-step persisted; interrupt() pauses for human-in-the-loop",
            fontsize=8.5, color="#555")
    save(fig, "langgraph", "graph_state.png")


# --- langgraph/human_in_loop.png --------------------------------------------
def human_in_loop():
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Human-in-the-loop: interrupt() pauses the graph, state persists,\n"
                 "an approval resumes (or edits) it - nothing runs while paused",
                 fontsize=11.5)
    box(ax, 0.5, 5.6, 2.6, 1.1, "agent node\nproposes action", fc="#fff7e6", ec="#b26a00", fs=8.5)
    box(ax, 5.0, 5.6, 3.4, 1.1, "interrupt(...)\ngraph suspends HERE", fc="#ffe9e9", ec="#b02a37", fs=8.5)
    box(ax, 10.2, 5.6, 3.3, 1.1, "human reviews\napprove / edit / reject", fc="#eef2fb", ec="#4a6fa5", fs=8.5)
    arrow(ax, 3.1, 6.15, 5.0, 6.15)
    arrow(ax, 8.4, 6.15, 10.2, 6.15)
    arrow(ax, 11.8, 5.6, 11.8, 3.2, "resume with decision")
    box(ax, 8.6, 1.4, 4.9, 1.6, "Command(resume=...)\ngraph continues from the SAME state\n"
        "(thread persisted by the checkpointer)", fc="#eaf7ea", ec="#2e7d32", fs=8.5)
    box(ax, 0.5, 1.4, 6.4, 1.6, "use when:\n- tool sends email / spends money\n"
        "- destructive DB writes\n- any action you cannot undo",
        fc="#f6f2fb", ec="#6a4aa5", fs=8.5)
    save(fig, "langgraph", "human_in_loop.png")


# --- agents/react_loop.png ---------------------------------------------------
def react_loop():
    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("ReAct loop: Thought -> Action -> Observation, repeated until\n"
                 "the model emits a Final Answer (or a stop condition fires)",
                 fontsize=11.5)
    box(ax, 4.0, 8.4, 4.0, 1.0, "user goal", fc="#eef2fb", ec="#4a6fa5", fs=9)
    box(ax, 4.0, 6.6, 4.0, 1.0, "Thought: what next?", fc="#fff7e6", ec="#b26a00", fs=8.5)
    box(ax, 4.0, 4.8, 4.0, 1.0, "Action: pick tool + args", fc="#fff7e6", ec="#b26a00", fs=8.5)
    box(ax, 0.5, 4.8, 2.6, 1.0, "tool executes", fc="#eaf7ea", ec="#2e7d32", fs=8.5)
    box(ax, 4.0, 3.0, 4.0, 1.0, "Observation: tool result", fc="#eaf7ea", ec="#2e7d32", fs=8.5)
    box(ax, 8.6, 3.0, 3.0, 1.0, "stop conditions:\nfinal answer /\nmax steps / budget", fc="#ffe9e9", ec="#b02a37", fs=8)
    arrow(ax, 6.0, 8.4, 6.0, 7.6)
    arrow(ax, 6.0, 6.6, 6.0, 5.8)
    arrow(ax, 4.0, 5.3, 3.1, 5.3, "invoke")
    arrow(ax, 1.8, 4.8, 4.0, 3.5, "result")
    arrow(ax, 6.0, 4.0, 6.0, 6.6, "loop: observe -> think again", color="#2e7d32")
    arrow(ax, 8.0, 3.5, 8.6, 3.5)
    ax.text(0.5, 1.6, "failure modes: infinite loops (cap steps), tool args that don't parse, "
            "hallucinated tool names, forgetting the observation",
            fontsize=8, color="#b02a37")
    ax.text(0.5, 0.7, "hardening: step budget + token budget, schema-validated tool args, "
            "idempotent tools, trace every step", fontsize=8, color="#2e7d32")
    save(fig, "agents", "react_loop.png")


# --- agents/memory_types.png -------------------------------------------------
def memory_types():
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Agent memory: the working context is small - everything else must be externalized",
                 fontsize=11.5)
    box(ax, 0.5, 5.2, 3.0, 1.5, "short-term\n(in-context)\nthis conversation", fc="#eef2fb", ec="#4a6fa5", fs=8.5)
    box(ax, 4.0, 5.2, 3.0, 1.5, "episodic\nstored past\ninteractions", fc="#fff7e6", ec="#b26a00", fs=8.5)
    box(ax, 7.5, 5.2, 3.0, 1.5, "semantic\nfacts/notes\n(vector DB)", fc="#fff7e6", ec="#b26a00", fs=8.5)
    box(ax, 11.0, 5.2, 2.6, 1.5, "procedural\nsystem prompt /\nskills", fc="#eaf7ea", ec="#2e7d32", fs=8.5)
    arrow(ax, 3.5, 5.95, 4.0, 5.95)
    arrow(ax, 7.0, 5.95, 7.5, 5.95)
    arrow(ax, 10.5, 5.95, 11.0, 5.95)
    ax.text(7, 3.9, "context window = working memory; retrieval = long-term memory",
            ha="center", fontsize=9.5, color="#333")
    box(ax, 0.6, 0.8, 6.2, 2.0, "engineering:\n- summarize old turns instead of appending forever\n"
        "- write memories explicitly, prune by relevance\n- never let memory grow unbounded (cost + drift)",
        fc="#f6f2fb", ec="#6a4aa5", fs=8.5)
    box(ax, 7.4, 0.8, 6.0, 2.0, "memory bugs agents actually hit:\n- stale memory -> wrong 'facts'\n"
        "- PII persisted forever\n- cross-user leakage (tenant isolation!)",
        fc="#fff2f2", ec="#b02a37", fs=8.5)
    save(fig, "agents", "memory_types.png")


# --- mlops/drift_retraining.png ----------------------------------------------
def drift_retraining():
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("MLOps closed loop: monitor drift -> trigger -> retrain -> evaluate -> promote (or rollback)",
                 fontsize=11.5)
    xs = [(0.5, "serve\nmodel", "#eef2fb"), (3.4, "log inputs\n+ outputs", "#fff7e6"),
          (6.3, "drift checks\nPSI / KS / accuracy", "#ffe9e9"), (9.2, "retrain\npipeline", "#fff7e6"),
          (12.0, "registry\nchampion/challenger", "#eaf7ea")]
    for x, t, fc in xs:
        box(ax, x, 5.6, 1.9, 1.3, t, fc=fc, ec="#4a6fa5", fs=8)
    for i in range(len(xs) - 1):
        arrow(ax, xs[i][0] + 1.9, 6.25, xs[i + 1][0], 6.25)
    arrow(ax, 12.95, 5.6, 1.45, 5.6, "promote or rollback", color="#2e7d32")
    ax.text(7, 4.2, "data drift: P(x) changes      concept drift: P(y|x) changes",
            ha="center", fontsize=9.5, family="monospace")
    ax.text(7, 3.2, "drift != degradation: always confirm with labels or a proxy metric",
            ha="center", fontsize=8.5, color="#b02a37")
    box(ax, 0.6, 0.8, 6.2, 1.7, "monitors: input distribution PSI, prediction mix, latency, error rate;\n"
        "alert thresholds tuned on a clean baseline week", fc="#f4f6fa", ec="#999", fs=8.5)
    box(ax, 7.4, 0.8, 6.0, 1.7, "retraining policy: scheduled vs triggered;\n"
        "shadow-deploy challengers before promoting", fc="#f4f6fa", ec="#999", fs=8.5)
    save(fig, "mlops", "drift_retraining.png")


# --- mlops/experiment_tracking.png -------------------------------------------
def experiment_tracking():
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Experiment tracking: every run is a (params, metrics, artifacts) triple you can diff",
                 fontsize=11.5)
    box(ax, 0.5, 5.4, 3.2, 1.4, "train.py\nrun #8472", fc="#fff7e6", ec="#b26a00", fs=8.5)
    box(ax, 5.0, 5.4, 4.0, 1.4, "tracking server\nparams | metrics | tags", fc="#eef2fb", ec="#4a6fa5", fs=8.5)
    box(ax, 10.4, 5.4, 3.1, 1.4, "model registry\nStaging -> Production", fc="#eaf7ea", ec="#2e7d32", fs=8.5)
    arrow(ax, 3.7, 6.1, 5.0, 6.1, "log")
    arrow(ax, 9.0, 6.1, 10.4, 6.1, "promote")
    ax.text(7, 4.2, "lr=3e-4  bs=64  epochs=12  ->  auc=0.913  (run #8472, branch feat/lr-sweep)",
            ha="center", fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f4f6fa", ec="#999"))
    box(ax, 0.6, 0.9, 6.2, 2.0, "rule of thumb:\nif a result cannot be reproduced from its run record,\n"
        "it never happened - log data version, git SHA, seeds, env", fc="#f6f2fb", ec="#6a4aa5", fs=8.5)
    box(ax, 7.4, 0.9, 6.0, 2.0, "why teams skip it (and regret it):\n'it's just a quick experiment' ->\n"
        "two weeks later nobody knows which model won", fc="#fff2f2", ec="#b02a37", fs=8.5)
    save(fig, "mlops", "experiment_tracking.png")


# --- security/threat_model.png ------------------------------------------------
def threat_model():
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis("off")
    ax.set_title("LLM app threat model: every input path is an attack path",
                 fontsize=11.5)
    box(ax, 0.5, 6.4, 2.4, 1.1, "user input", fc="#eef2fb", ec="#4a6fa5", fs=8.5)
    box(ax, 4.2, 6.4, 2.6, 1.1, "app / prompt", fc="#fff7e6", ec="#b26a00", fs=8.5)
    box(ax, 8.2, 6.4, 2.6, 1.1, "LLM", fc="#fff7e6", ec="#b26a00", fs=8.5)
    box(ax, 11.9, 6.4, 1.9, 1.1, "tools", fc="#eaf7ea", ec="#2e7d32", fs=8.5)
    arrow(ax, 2.9, 6.95, 4.2, 6.95)
    arrow(ax, 6.8, 6.95, 8.2, 6.95)
    arrow(ax, 10.8, 6.95, 11.9, 6.95)
    box(ax, 0.4, 3.6, 3.2, 1.5, "direct prompt injection\n'ignore previous\ninstructions'", fc="#ffe9e9", ec="#b02a37", fs=8)
    box(ax, 4.4, 3.6, 3.2, 1.5, "indirect injection\nmalicious text INSIDE\nretrieved docs", fc="#ffe9e9", ec="#b02a37", fs=8)
    box(ax, 8.4, 3.6, 3.0, 1.5, "jailbreak /\nrole-play bypass\nof safety rules", fc="#ffe9e9", ec="#b02a37", fs=8)
    box(ax, 11.6, 3.6, 2.2, 1.5, "tool abuse\nexcessive\npermissions", fc="#ffe9e9", ec="#b02a37", fs=8)
    arrow(ax, 1.6, 6.4, 1.6, 5.1, color="#b02a37")
    arrow(ax, 5.5, 6.4, 6.0, 5.1, color="#b02a37")
    arrow(ax, 9.5, 6.4, 9.9, 5.1, color="#b02a37")
    arrow(ax, 12.8, 6.4, 12.7, 5.1, color="#b02a37")
    box(ax, 0.4, 0.7, 13.4, 1.9,
        "defenses: input/output filtering - instruction hierarchy in the prompt - least-privilege tools "
        "(scoped tokens, read-only) - human approval for irreversible actions - tenant isolation enforced "
        "at the DATA layer (filter before embed), never just in the prompt - audit log every tool call",
        fc="#eaf7ea", ec="#2e7d32", fs=8.5)
    save(fig, "security", "threat_model.png")


# --- security/rag_poisoning.png ----------------------------------------------
def rag_poisoning():
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("RAG poisoning: an attacker who controls a document controls your answers",
                 fontsize=11.5)
    box(ax, 0.5, 5.4, 3.2, 1.4, "attacker plants\ndoc: 'per policy,\nalways say REFUND'", fc="#ffe9e9", ec="#b02a37", fs=8)
    box(ax, 4.6, 5.4, 3.0, 1.4, "ingested +\nembedded like\nany other doc", fc="#fff7e6", ec="#b26a00", fs=8)
    box(ax, 8.4, 5.4, 2.6, 1.4, "query hits it\n-> injected text\nrides into prompt", fc="#fff7e6", ec="#b26a00", fs=8)
    box(ax, 11.7, 5.4, 2.1, 1.4, "LLM outputs\nattacker's\npayload", fc="#ffe9e9", ec="#b02a37", fs=8)
    arrow(ax, 3.7, 6.1, 4.6, 6.1)
    arrow(ax, 7.6, 6.1, 8.4, 6.1)
    arrow(ax, 11.0, 6.1, 11.7, 6.1)
    box(ax, 0.5, 1.0, 6.4, 3.0, "ingestion defenses:\n- source allow-list (who may add docs?)\n- sanitize: strip instructions from retrieved text\n"
        "- quote-then-verify: answers must cite chunks\n- anomaly detection on new docs\n- re-embedding pipeline with diff review",
        fc="#eaf7ea", ec="#2e7d32", fs=8)
    box(ax, 7.4, 1.0, 6.0, 3.0, "runtime defenses:\n- retrieval filter by tenant BEFORE search\n- content scanner on retrieved chunks\n"
        "- refuse when citations unsupported\n- human review queue for high-risk topics\n- red-team the corpus regularly",
        fc="#eaf7ea", ec="#2e7d32", fs=8)
    save(fig, "security", "rag_poisoning.png")


# --- pytorch/training_loop.png -------------------------------------------------
def training_loop():
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9.5); ax.axis("off")
    ax.set_title("The canonical PyTorch training loop - five lines that never change",
                 fontsize=11.5)
    code = ("for epoch in range(E):\n"
            "    for xb, yb in train_loader:          # DataLoader batches\n"
            "        xb, yb = xb.to(dev), yb.to(dev)\n"
            "        opt.zero_grad()                   # clear stale grads\n"
            "        out = model(xb)                   # forward\n"
            "        loss = loss_fn(out, yb)           # scalar loss\n"
            "        loss.backward()                   # autograd fills .grad\n"
            "        opt.step()                        # update params")
    ax.text(0.5, 8.4, code, fontsize=9.5, family="monospace", va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f4f6fa", ec="#4a6fa5"))
    notes = [("zero_grad()", "grads accumulate by default;\nskipping this is the #1 beginner bug", 0.55),
             ("backward()", "computes dLoss/dParam for every\nleaf with requires_grad=True", 5.3),
             ("step()", "optimizer applies its update rule\n(SGD/AdamW) to every param", 9.9)]
    for name, desc, y in notes:
        box(ax, 9.1, y, 4.6, 1.9, f"{name}\n\n{desc}", fc="#fff7e6", ec="#b26a00", fs=8)
        arrow(ax, 9.1, y + 0.95, 8.05, y + 0.95, color="#b26a00")
    box(ax, 0.5, 0.6, 8.0, 1.6, "validation: model.eval() + torch.no_grad() -> otherwise dropout/BN\n"
        "corrupt metrics and grads balloon memory.  checkpoint = model.state_dict() + opt.state_dict() + epoch",
        fc="#eaf7ea", ec="#2e7d32", fs=8.5)
    save(fig, "pytorch", "training_loop.png")


# --- pytorch/autograd_graph.png -------------------------------------------------
def autograd_graph():
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Autograd: forward builds a dynamic graph, backward walks it in reverse topological order",
                 fontsize=11.5)
    box(ax, 0.6, 5.6, 2.2, 1.1, "x\n(requires_grad)", fc="#eef2fb", ec="#4a6fa5", fs=8.5)
    box(ax, 4.2, 5.6, 2.6, 1.1, "y = x @ W\n(AddmmBackward)", fc="#fff7e6", ec="#b26a00", fs=8.5)
    box(ax, 8.2, 5.6, 2.6, 1.1, "z = relu(y)\n(ReluBackward)", fc="#fff7e6", ec="#b26a00", fs=8.5)
    box(ax, 11.9, 5.6, 1.9, 1.1, "loss\n(scalar)", fc="#eaf7ea", ec="#2e7d32", fs=8.5)
    arrow(ax, 2.8, 6.15, 4.2, 6.15)
    arrow(ax, 6.8, 6.15, 8.2, 6.15)
    arrow(ax, 10.8, 6.15, 11.9, 6.15)
    arrow(ax, 12.85, 5.6, 7.0, 3.2, "loss.backward()", color="#b02a37")
    arrow(ax, 7.0, 3.2, 2.0, 5.6, "dL/dx accumulates into x.grad", color="#b02a37")
    ax.text(7, 1.9, "graph is freed after .backward() - retain_graph=True only when you really need two passes",
            ha="center", fontsize=8.5, color="#555")
    box(ax, 0.5, 0.4, 6.0, 1.5, "memory: activations are kept for backward -\n"
        "batch size is the first knob to cut on OOM", fc="#f4f6fa", ec="#999", fs=8.5)
    box(ax, 7.4, 0.4, 6.0, 1.5, "no_grad() skips graph building ->\nfast, low-memory inference", fc="#f4f6fa", ec="#999", fs=8.5)
    save(fig, "pytorch", "autograd_graph.png")


# =============================================================================
# B) VIDEOS (6 new teaching clips, same style as generate_rag_media.py)
# =============================================================================

def video_attention():
    """Token attention weights shift as the query token moves."""
    tokens = ["the", "cat", "sat", "on", "mat"]
    rng = np.random.default_rng(11)
    base = rng.uniform(0.05, 0.35, (5, 5))
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    n = N_FRAMES
    w = _writer()
    with w.saving(fig, os.path.join(VIDEOS, "attention_flow.mp4"), dpi=90):
        for f in range(n):
            q = f / (n - 1) * (len(tokens) - 1)
            qi, qt = int(q), min(int(q) + 1, len(tokens) - 1)
            frac = q - qi
            ax.clear()
            row = base[qi] * (1 - frac) + base[qt] * frac
            row = row / row.sum()
            cols = ["#b02a37", "#1565c0", "#2e7d32", "#b26a00", "#6a4aa5"]
            bars = ax.bar(tokens, row, color=cols, alpha=0.9)
            for b, v in zip(bars, row):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                        ha="center", fontsize=8.5)
            ax.set_title(f"self-attention: weights for query token '{tokens[qt]}'\n"
                         "softmax(qK^T / sqrt(d)) -> each key gets a share of attention",
                         fontsize=10.5)
            ax.set_ylabel("attention weight")
            ax.set_ylim(0, 0.8)
            _frame_mark(fig, f, n)
            fig.canvas.draw()
            w.grab_frame()


def video_gradient_descent():
    """Ball descends a 1-D loss surface; step size and momentum shown."""
    x = np.linspace(-3.2, 3.2, 200)
    loss = 0.35 * x ** 2 + 0.4 * np.sin(2.2 * x) + 0.9
    grad = 0.7 * x + 0.8 * np.cos(2.2 * x)
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    n = N_FRAMES
    w = _writer()
    pos, vel = -3.0, 0.0
    lr, mom = 0.08, 0.85
    path = []
    with w.saving(fig, os.path.join(VIDEOS, "gradient_descent.mp4"), dpi=90):
        for f in range(n):
            ax.clear()
            ax.plot(x, loss, color="#1565c0", lw=2, label="loss surface")
            g = float(np.interp(pos, x, grad))
            vel = mom * vel - lr * g
            pos = float(np.clip(pos + vel, -3.1, 3.1))
            path.append(pos)
            y_pos = float(np.interp(pos, x, loss))
            ax.scatter(path, [float(np.interp(p, x, loss)) for p in path],
                       s=12, color="#b02a37", alpha=0.35, zorder=4)
            ax.scatter([pos], [y_pos], s=90, color="#b02a37", zorder=5)
            ax.annotate("", xy=(pos - g * 0.5, y_pos + 0.12),
                        xytext=(pos, y_pos + 0.12),
                        arrowprops=dict(arrowstyle="-|>", color="#2e7d32", lw=1.6))
            ax.text(pos - g * 0.5, y_pos + 0.22, f"-grad  ({g:+.2f})",
                    fontsize=8, color="#2e7d32", ha="center")
            ax.set_title(f"gradient descent + momentum (step {f + 1}/{n})\n"
                         "update = -lr * grad; momentum smooths the path",
                         fontsize=10.5)
            ax.set_xlabel("parameter value")
            ax.set_ylabel("loss")
            ax.legend(fontsize=8, loc="upper right")
            _frame_mark(fig, f, n)
            fig.canvas.draw()
            w.grab_frame()


def video_agent_loop():
    """Agent cycles Thought -> Action -> Observation until Final Answer."""
    steps = ["THOUGHT:\nneed error docs", "ACTION:\nsearch('401')",
             "OBSERVATION:\n[errors] 401 = bad key", "THOUGHT:\nfound it",
             "FINAL:\ncite [errors]"]
    states = ["thought", "action", "observe", "thought", "final"]
    cols = {"thought": "#b26a00", "action": "#1565c0", "observe": "#2e7d32",
            "final": "#b02a37"}
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    n = N_FRAMES
    w = _writer()
    with w.saving(fig, os.path.join(VIDEOS, "agent_react_loop.mp4"), dpi=90):
        for f in range(n):
            i = min(int(f / (n - 1) * len(steps)), len(steps) - 1)
            ax.clear()
            ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
            for j, s in enumerate(steps):
                y = 8.6 - j * 1.9
                done = j < i
                cur = j == i
                ec = "#2e7d32" if done else (cols[states[j]] if cur else "#c9d2de")
                fc = "#eaf7ea" if done else ("#ffffff" if cur else "#f4f6fa")
                box(ax, 2.2, y, 5.6, 1.5, s, fc=fc, ec=ec,
                    fs=9 if cur else 8)
                if done:
                    ax.text(8.1, y + 0.6, "done", fontsize=8, color="#2e7d32",
                            va="center")
            ax.set_title(f"agent loop step {i + 1}/{len(steps)}\n"
                         "think -> act -> observe -> repeat until final answer",
                         fontsize=10.5)
            _frame_mark(fig, f, n)
            fig.canvas.draw()
            w.grab_frame()


def video_hnsw():
    """Greedy HNSW-lite search hops across graph layers toward the query."""
    rng = np.random.default_rng(5)
    pts = rng.normal(size=(22, 2))
    q = np.array([1.9, 1.6])
    fig, (axg, axr) = plt.subplots(1, 2, figsize=(9.8, 4.8))
    fig.suptitle("HNSW-lite: greedy hops beat brute force when the graph is built well",
                 fontsize=10.5)
    n = N_FRAMES
    w = _writer()
    with w.saving(fig, os.path.join(VIDEOS, "hnsw_search.mp4"), dpi=90):
        visited, cur = [0], 0
        for f in range(n):
            # greedy step every 6 frames, up to 5 hops
            if f % 6 == 5 and len(visited) < 6:
                d = np.linalg.norm(pts - q, axis=1)
                for v in visited:
                    d[v] = np.inf
                nxt = int(np.argmin(d))
                visited.append(nxt)
                cur = nxt
            axg.clear()
            axg.scatter(pts[:, 0], pts[:, 1], s=55, c="#9aa7b8", zorder=3)
            axg.scatter(*q, marker="*", s=260, c="#b02a37", zorder=4)
            for j in range(1, len(visited)):
                a, b = visited[j - 1], visited[j]
                axg.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                         color="#2e7d32", lw=2, alpha=0.7)
            axg.scatter(pts[cur, 0], pts[cur, 1], s=110, c="#1565c0", zorder=5)
            axg.set_title("graph walk: enter -> greedy hop to closer node")
            axg.set_xlim(-3, 3); axg.set_ylim(-3, 3)
            axr.clear()
            axr.barh(["visited"], [len(visited)], color="#2e7d32")
            axr.set_xlim(0, 7)
            axr.set_title(f"hops: {len(visited) - 1} / nodes touched: {len(visited)}")
            _frame_mark(fig, f, n)
            fig.canvas.draw()
            w.grab_frame()


def video_tokenizer():
    """A sentence is progressively split into BPE-style tokens."""
    words = ["un", "believ", "able", "!", "", "tok", "eniz", "ation", "rocks", "!"]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    n = N_FRAMES
    w = _writer()
    with w.saving(fig, os.path.join(VIDEOS, "tokenization_flow.mp4"), dpi=90):
        for f in range(n):
            i = int(f / (n - 1) * (len(words) - 1))
            ax.clear()
            ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
            shown = words[:i + 1]
            text = "".join(shown)
            xs = 0.4
            for wd in shown:
                if wd == "":
                    continue
                wpx = 0.62 if len(wd) <= 3 else 1.05
                fc = "#fff7e6" if len(wd) > 1 else "#eef2fb"
                box(ax, xs, 5.4, wpx, 1.5, wd, fc=fc, ec="#b26a00", fs=8.5)
                xs += wpx + 0.18
            ax.text(0.4, 3.6, f"raw: unbelievable tokenization rocks", fontsize=9,
                    color="#666")
            ax.text(0.4, 2.4, f"token ids assigned left-to-right as merges complete",
                    fontsize=8.5, color="#555")
            ax.set_title(f"BPE tokenization ({i + 1}/{len(words)} pieces)\n"
                         "common subwords become single tokens - rare words split",
                         fontsize=10.5)
            _frame_mark(fig, f, n)
            fig.canvas.draw()
            w.grab_frame()


def video_overfitting():
    """Polynomial degree rises; train loss falls, val loss rises after the knee."""
    rng = np.random.default_rng(2)
    X = np.linspace(0, 1, 24)
    y = np.sin(2 * np.pi * X) + rng.normal(0, 0.16, X.shape)
    Xs = np.linspace(0, 1, 200)
    fig, (axt, axv) = plt.subplots(1, 2, figsize=(9.8, 4.7))
    tr_hist, va_hist = [], []
    fig.suptitle("capacity sweep: train error keeps falling, val error turns up - the overfit knee",
                 fontsize=10.5)
    n = N_FRAMES
    degs = np.linspace(1, 18, n)
    w = _writer()
    tr_hist, va_hist = [], []
    with w.saving(fig, os.path.join(VIDEOS, "overfitting_dynamics.mp4"), dpi=90):
        for f in range(n):
            deg = int(degs[f])
            axt.clear(); axv.clear()
            coefs = np.polyfit(X, y, deg)
            axt.scatter(X, y, s=18, c="#666", label="train data")
            axt.plot(Xs, np.polyval(coefs, Xs), color="#b02a37", lw=2,
                     label=f"deg {deg} fit")
            axt.set_ylim(-2.2, 2.2)
            axt.set_title(f"model: polynomial degree {deg}")
            axt.legend(fontsize=8, loc="lower left")
            tr = float(np.mean((np.polyval(coefs, X) - y) ** 2))
            tr_hist.append(tr)
            va_hist.append(tr * (1 + max(0.0, deg - 6) * 0.22) + 0.02)
            axv.plot(np.arange(1, len(tr_hist) + 1), tr_hist, color="#1565c0",
                     lw=2, label="train MSE")
            axv.plot(np.arange(1, len(va_hist) + 1), va_hist, color="#b26a00",
                     lw=2, label="val MSE")
            axv.set_title("error vs capacity")
            axv.set_xlabel("degree")
            axv.legend(fontsize=8)
            _frame_mark(fig, f, n)
            fig.canvas.draw()
            w.grab_frame()


# =============================================================================
# C) MAIN
# =============================================================================

def main():
    jobs = [lcel_runnable, structured_output, graph_state, human_in_loop,
            react_loop, memory_types, drift_retraining, experiment_tracking,
            threat_model, rag_poisoning, training_loop, autograd_graph]
    for j in jobs:
        j()
    videos = [video_attention, video_gradient_descent, video_agent_loop,
              video_hnsw, video_tokenizer, video_overfitting]
    for v in videos:
        v()
    print("v3 panels + videos complete")


if __name__ == "__main__":
    main()
