"""
generate_gallery.py - DIAGRAM GALLERY INDEX
===========================================
Creates a browsable index of every generated diagram:

  diagrams/gallery/thumbs/<cat>/  - small thumbnails (max 340px wide)
  diagrams/gallery/index.html     - open in any browser: grouped cards,
                                    thumbnail, caption, file, dimensions
  diagrams/DIAGRAM_GALLERY.md     - markdown index (renders on GitHub)

Run:   python diagrams/generate_gallery.py
"""
import os
from html import escape

from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
GALLERY = os.path.join(ROOT, "gallery")
THUMBS = os.path.join(GALLERY, "thumbs")

CATEGORY_TITLES = {
    "llm": "LLM Course — internals, training, inference (diagrams/llm/)",
    "agents": "GenAI + Agents Course — RAG agents, multi-agent, bench (diagrams/agents/)",
    "ml": "ML Course — lifecycle + model diagnostics (diagrams/ml/)",
    "math": "Mathematics (diagrams/math/)",
    "dl": "Deep Learning (diagrams/dl/)",
    "backend": "Backend / APIs (diagrams/backend/)",
    "nlp": "NLP (diagrams/nlp/)",
    "rag": "RAG (diagrams/rag/)",
    "transformers": "Transformers (diagrams/transformers/)",
    "graphs": "Training-run graphs (diagrams/graphs/)",
    "graph": "Graph ML (diagrams/graph/)",
    "recommenders": "Recommenders (diagrams/recommenders/)",
    "rl": "Reinforcement Learning (diagrams/rl/)",
}

CAPTIONS = {
    "llm": {
        "llm_architecture": "Decoder-only GPT stack: token embeddings -> RoPE -> N blocks (RMSNorm, MHA, SwiGLU FFN) -> final norm -> LM head -> sampling",
        "pretraining_pipeline": "Raw data -> filter -> clean -> dedup -> tokenize -> pack -> shuffle -> batch -> train -> validate -> checkpoint",
        "kv_cache": "O(T^2) full recompute vs O(T) cached decoding, annotated with measured speedups (2.4x @ T=128 ... 11x @ T=768)",
        "kv_cache_memory": "KV cache memory growth: ~0.5 MiB/token => 2 GiB @ 4k, 64 GiB @ 128k context",
        "lora": "W' = W + (alpha/r) B A: adapter low-rank decomposition with the verified 48x trainable-parameter reduction",
        "sampling": "Temperature reshaping of logits [2, 1, 0.1] (T=0.1 sharp, T=1.0, T=2.0 flat), top-k and top-p truncation",
        "scaling_laws": "REAL lab sweep: val loss 2.723 (S) -> 2.617 (M) -> 2.421 (L) vs bigram 2.763; ppl 15.2/13.7/11.3",
        "prefill_decode": "Prefill phase (parallel, compute-bound) vs decode phase (one token at a time, memory-bandwidth-bound)",
        "quantization": "7B model memory: FP32 28 GB -> FP16/BF16 14 GB -> INT8 7 GB -> INT4 3.5 GB",
        "speculative_decoding": "Draft (gamma tokens) -> verify in parallel -> accept/reject -> resample: ~2.6x fewer forwards",
    },
    "agents": {
        "agent_loop": "Observe -> reason -> act (tool call) -> observe result -> repeat until final answer; tool registry + safety guards",
        "multi_agent": "Supervisor pattern: supervisor routes to writer / reviewer / researcher specialists, results return to supervisor",
        "rag_agent": "RAG agent: query -> retrieve -> ground -> generate with citations; tool calls for calculator/retrieval; injection guard",
        "prod_rag_server": "Production RAG server: auth/RBAC -> per-key quota -> Redis-style cache -> RAG brain -> SQL request log",
        "embedding_bench": "REAL lab numbers: dense 9/9 vs lexical 6/9 hit@1; paraphrases 6/6 vs 4/6; HF MiniLM matches local SVD",
    },
    "ml": {
        "ml_lifecycle": "Full ML lifecycle: data -> EDA -> features -> train/val/test -> baseline -> model -> tune -> deploy -> monitor -> drift -> retrain (closed loop)",
        "learning_curves": "Train vs validation error as training set grows: high-variance (gap) vs high-bias (plateau) diagnosis",
        "kmeans": "K-means iterations: centroids moving, assignments updating, convergence on the objective",
        "pca": "PCA projection: data cloud, first principal component direction, explained-variance scree",
        "pr_curve": "Precision-recall curve with baseline and trade-off regions (for imbalanced classes)",
        "imbalance": "Class imbalance: skewed distribution + SMOTE oversampling of the minority class",
        "feature_importance": "Permutation / tree-based feature importance ranking with drop-off",
        "time_series": "Trend + seasonality decomposition with forecast over the historical series",
        "confusion_matrix": "TP/TN/FP/FN layout with derived metrics (accuracy, precision, recall, F1)",
        "cross_validation": "K-fold CV: fold rotation, train/validation blocks per fold",
        "decision_boundary": "Classifier decision boundary vs data points (linear and non-linear)",
        "ml_workflow": "End-to-end ML workflow diagram (data -> modeling -> evaluation -> deployment)",
        "model_comparison": "Model comparison: accuracy/latency/size trade-offs",
        "overfitting": "Overfitting vs underfitting vs good fit curves",
        "roc_curve": "ROC curve with AUC annotation",
        "test": "Train / validation / test split with leakage warning",
    },
    "math": {
        "bias_variance": "Bias-variance decomposition: low/high bias x low/high variance quadrants",
        "entropy_gini": "Entropy and Gini impurity as functions of class probability",
        "gradient_descent": "Gradient descent: cost surface, steps to minimum, learning-rate effect",
        "sigmoid": "Sigmoid curve with its derivative",
    },
    "dl": {
        "activation_functions": "Sigmoid / tanh / ReLU / GELU / SiLU curves with derivative behavior",
        "backpropagation": "Backpropagation flow: forward pass, loss, gradients flowing backward through the graph",
        "cnn_architecture": "CNN stack: conv -> pool -> conv -> pool -> flatten -> dense",
        "gradient_descent": "Neural-net gradient descent: loss landscape and optimization path",
        "lstm_gates": "LSTM cell: forget / input / output gates and cell-state highway",
        "neural_network": "MLP: input -> hidden layers -> output with weights and activations",
    },
    "backend": {
        "api_flow": "API request flow: client -> route -> validation -> service -> response",
        "backend_architecture": "Backend architecture: FastAPI, DB, cache, workers",
    },
    "nlp": {
        "nlp_pipeline": "NLP pipeline: raw text -> clean -> tokenize -> features -> model",
        "sentiment_analysis": "Sentiment classification flow with example scores",
        "word_embeddings": "Word embeddings: high-dim one-hot to dense vector space with similar words near each other",
    },
    "rag": {
        "rag_pipeline": "RAG pipeline: documents -> chunk -> embed -> vector DB -> retrieve -> LLM -> grounded answer",
        "rag_vs_finetuning": "RAG vs fine-tuning comparison: knowledge updates, cost, hallucination",
    },
    "transformers": {
        "self_attention": "Scaled dot-product attention: Q, K, V, scores, softmax, weighted sum",
        "transformer_architecture": "Transformer block: attention + add&norm + FFN + add&norm",
    },
    "graphs": {
        "gradient_descent_from_scratch": "From-scratch gradient descent run: loss over iterations",
        "lstm_sine_forecast": "LSTM sine-wave forecast vs ground truth",
        "nn_training_loss": "Neural-network training loss curves",
        "pca_demo": "PCA demo projection",
        "vis_01_loss_curve": "Training loss curve (matplotlib basics)",
        "vis_02_regression": "Regression fit example",
        "vis_03_confusion": "Confusion matrix example",
        "vis_04_roc": "ROC curve example",
    },
    "graph": {
        "link_prediction_auc": "GCN link prediction: real AUC benchmark numbers",
        "message_passing": "Graph message passing: node -> aggregate neighbors -> update (GCN step)",
    },
    "recommenders": {
        "als_vs_sgd_rmse": "ALS vs SGD matrix factorization: real RMSE benchmark numbers",
        "two_stage_recsys": "Two-stage recommender: candidate generation -> ranking",
    },
    "rl": {
        "agent_env_loop": "RL loop: agent <-> environment, action/state/reward cycle",
        "qlearning_vs_optimal": "Q-learning returns vs optimal: real benchmark numbers",
    },
}

EXTENSIONS = (".png", ".jpg", ".jpeg")


def build_index():
    categories = [c for c in CATEGORY_TITLES if os.path.isdir(os.path.join(ROOT, c))]
    entries = {}          # cat -> list of (file, title, caption, w, h)
    for cat in categories:
        folder = os.path.join(ROOT, cat)
        caps = CAPTIONS.get(cat, {})
        cat_entries = []
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith(EXTENSIONS):
                continue
            path = os.path.join(folder, fn)
            title = fn.rsplit(".", 1)[0]
            with Image.open(path) as im:
                w, h = im.size
            cat_entries.append((fn, title, caps.get(title, ""), w, h))
        entries[cat] = cat_entries
    return entries


def make_thumbnails(entries):
    made = 0
    for cat, cat_entries in entries.items():
        out_dir = os.path.join(THUMBS, cat)
        os.makedirs(out_dir, exist_ok=True)
        for fn, _, _, w, h in cat_entries:
            src = os.path.join(ROOT, cat, fn)
            dst = os.path.join(out_dir, fn)
            if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
                continue
            with Image.open(src) as im:
                im = im.convert("RGB")
                scale = min(1.0, 340.0 / max(w, h))
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                               Image.LANCZOS)
                im.save(dst, optimize=True)
            made += 1
    return made


def write_html(entries):
    cards = []
    total = 0
    for cat, cat_entries in entries.items():
        total += len(cat_entries)
        blocks = []
        for fn, title, caption, w, h in cat_entries:
            thumb = f"thumbs/{cat}/{fn}"
            full = f"../{cat}/{fn}"
            cap = escape(caption) if caption else "&nbsp;"
            blocks.append(f"""
      <div class="card">
        <a href="{full}" target="_blank"><img loading="lazy" src="{thumb}" alt="{escape(title)}"></a>
        <div class="meta">
          <div class="title">{escape(title)}</div>
          <div class="file">{cat}/{fn} &middot; {w}x{h}</div>
          <div class="cap">{cap}</div>
        </div>
      </div>""")
        cards.append(f"""
    <section>
      <h2>{escape(CATEGORY_TITLES[cat])}</h2>
      <div class="grid">{"".join(blocks)}
      </div>
    </section>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI/ML Engineering — Diagram Gallery ({total} images)</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
         margin: 0; background: #f4f5f7; color: #1d2129; }}
  header {{ background: #101828; color: #fff; padding: 22px 28px; }}
  header h1 {{ margin: 0 0 4px; font-size: 22px; }}
  header p {{ margin: 0; color: #aab4c4; font-size: 14px; }}
  main {{ max-width: 1280px; margin: 0 auto; padding: 18px 28px 60px; }}
  section {{ margin-top: 26px; }}
  h2 {{ font-size: 16px; border-bottom: 2px solid #cbd2dc; padding-bottom: 6px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
          gap: 16px; margin-top: 12px; }}
  .card {{ background: #fff; border: 1px solid #dfe3ea; border-radius: 8px;
          overflow: hidden; box-shadow: 0 1px 2px rgba(16,24,40,.06); }}
  .card img {{ width: 100%; height: auto; display: block; background: #fff;
              border-bottom: 1px solid #eef0f4; }}
  .meta {{ padding: 10px 12px 12px; }}
  .title {{ font-weight: 600; font-size: 14px; }}
  .file {{ color: #667085; font-size: 12px; margin-top: 2px; font-family: ui-monospace, monospace; }}
  .cap {{ color: #475467; font-size: 12.5px; margin-top: 6px; line-height: 1.45; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #0f1218; color: #e4e7ec; }}
    .card {{ background: #161b24; border-color: #242b37; }}
    .card img {{ background: #161b24; border-color: #1c2230; }}
    .file {{ color: #98a2b3; }} .cap {{ color: #c0c6d0; }}
    h2 {{ border-color: #2a3240; }}
  }}
</style>
</head>
<body>
<header>
  <h1>AI/ML Engineering — Diagram Gallery</h1>
  <p>{total} diagrams &middot; llm_course / genai_agents_course / ml_course + core-curriculum visuals
     &middot; regenerate: python diagrams/generate_gallery.py &middot; review: python diagrams/verify_diagrams.py</p>
</header>
<main>{"".join(cards)}
  <footer style="margin-top:34px; color:#667085; font-size:12.5px;">
    Full-size images live in <code>diagrams/&lt;category&gt;/</code>. Thumbnails regenerate only when the
    source is newer. See <code>diagrams/DIAGRAM_GALLERY.md</code> for the markdown index and
    <code>diagrams/gallery/REVIEW.txt</code> for the integrity review.
  </footer>
</main>
</body>
</html>
"""
    with open(os.path.join(GALLERY, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    return total


def write_md(entries):
    lines = []
    lines.append("# AI/ML Engineering — Diagram Gallery Index")
    lines.append("")
    lines.append(f"{sum(len(v) for v in entries.values())} diagrams, grouped by course / topic. "
                 "Regenerate with `python diagrams/generate_gallery.py`; integrity review: "
                 "`python diagrams/verify_diagrams.py`.")
    lines.append("")
    for cat, cat_entries in entries.items():
        lines.append(f"## {CATEGORY_TITLES[cat]}")
        lines.append("")
        lines.append("| Diagram | Description |")
        lines.append("|---------|-------------|")
        for fn, title, caption, w, h in cat_entries:
            desc = caption.replace("|", "\\|") if caption else "—"
            lines.append(f"| ![{title}]({cat}/{fn})<br>`{fn}` {w}x{h} | {desc} |")
        lines.append("")
    with open(os.path.join(ROOT, "DIAGRAM_GALLERY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    entries = build_index()
    thumbs = make_thumbnails(entries)
    total = write_html(entries)
    write_md(entries)
    print(f"gallery built: {total} diagrams, {thumbs} thumbnails generated")
    print("  HTML: diagrams/gallery/index.html")
    print("  MD  : diagrams/DIAGRAM_GALLERY.md")


if __name__ == "__main__":
    main()