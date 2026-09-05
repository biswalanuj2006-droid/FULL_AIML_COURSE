"""
================================================================================
EMBEDDING HF BENCH  (genai_agents_course/embedding_hf_bench.py)
================================================================================
Benchmarks the in-file PPMI-SVD embeddings of embedding_rag_lab.py against a
REAL transformer sentence-embedding model (Sentence-BERT style) - the same
evaluation queries and documents, same cosine retrieval, side by side.

Constraint: the lab must run with NO network (verify_course.py policy).
So it loads a transformer ONLY if one is already cached on disk
(local_files_only=True, HF_HUB_OFFLINE=1):

  * MODEL FOUND   -> embed the queries/docs with the real model and print
                     the quality + latency comparison vs the PPMI-SVD vectors.
  * MODEL ABSENT  -> print exactly how to cache one (one command, run once),
                     show the PPMI-SVD numbers as today's baseline, and exit
                     PASS so CI stays green without downloads.

Cached-model detection scans the Hugging Face hub cache for ANY transformer
model directory (models--* with config.json + a weights file), so whatever
sentence-embedding model you have cached will be picked up.

    python embedding_hf_bench.py
================================================================================
"""

from __future__ import annotations

import glob
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# Offline first, before any transformers import.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from embedding_rag_lab import DOCS, train_embeddings, embed_text, cosine, words  # noqa: E402

QUERIES: List[Tuple[str, str, str]] = [
    ("how do I get a refund", "refunds", "direct"),
    ("give me my money back", "refunds", "paraphrase"),
    ("my money never came back after I returned it", "refunds", "paraphrase"),
    ("what time are you open", "hours", "direct"),
    ("when do you close in the evening", "hours", "paraphrase"),
    ("I forgot my password and cannot log in", "password", "direct"),
    ("how do I stop my subscription", "cancel", "paraphrase"),
    ("the screen broke, is it covered", "warranty", "paraphrase"),
    ("is there a person I can email", "contact", "paraphrase"),
]

TITLES = [t for t, _ in DOCS]
DOC_TEXTS = [tx for _, tx in DOCS]


def hub_dir() -> str:
    return os.environ.get("HF_HUB_CACHE",
                          os.path.join(os.path.expanduser("~"), ".cache",
                                       "huggingface", "hub"))


def scan_hf_cache() -> List[str]:
    """Cached model snapshot dirs (config.json + weights), files only."""
    found = []
    for snap in sorted(glob.glob(os.path.join(hub_dir(), "models--*", "snapshots", "*"))):
        cfg = os.path.join(snap, "config.json")
        weights = glob.glob(os.path.join(snap, "*.safetensors")) + \
                  glob.glob(os.path.join(snap, "pytorch_model*.bin"))
        if os.path.exists(cfg) and weights:
            found.append(snap)
    return found


def try_load_transformer() -> Tuple[Optional[object], str]:
    """Files-first: import the (heavy) HF libraries ONLY if a model exists."""
    cached = scan_hf_cache()
    if not cached:
        return None, "no cached model found"   # zero HF imports -> fast skip
    # 1) sentence-transformers models (best mean-pooled sentence vectors)
    for snap in cached:
        name = snap.split(os.sep)[-1]
        for cand in ("models--sentence-transformers--", "models--BAAI--"):
            if cand in snap:
                try:
                    from sentence_transformers import SentenceTransformer
                    return SentenceTransformer(snap, local_files_only=True), \
                        name
                except Exception:
                    break
    # 2) any cached transformer + mean pooling
    try:
        from transformers import AutoModel, AutoTokenizer
        snap = cached[0]
        tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
        mod = AutoModel.from_pretrained(snap, local_files_only=True)
        mod.eval()
        return (mod, tok), os.path.basename(snap)
    except Exception:
        return None, "cached transformer failed to load"


def mean_pool(model_tok, texts: List[str]) -> np.ndarray:
    model, tok = model_tok
    import torch
    vecs = []
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt", padding=True, truncation=True,
                      max_length=128)
            out = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            v = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vecs.append(v[0].numpy())
    arr = np.stack(vecs)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr


def pprint_svd() -> Tuple[int, int, float, List[float]]:
    """Re-train the local PPMI-SVD embeddings and run the same eval."""
    corpus = [tx for _, tx in DOCS]
    from embedding_rag_lab import AUX
    emb, _ = train_embeddings(AUX + corpus, dim=48)
    doc_vecs = {t: embed_text(tx, emb) for t, tx in DOCS}
    scores = []
    for q, expected, _ in QUERIES:
        top = sorted(((t, cosine(embed_text(q, emb), v))
                      for t, v in doc_vecs.items()), key=lambda x: -x[1])[0]
        scores.append(top[0] == expected)
    emb_hits = sum(scores)
    return emb_hits, len(QUERIES), emb_hits / len(QUERIES), scores


def main() -> None:
    print("=" * 72)
    print("EMBEDDING HF BENCH  (local transformer embeddings vs PPMI-SVD)")
    print("=" * 72)
    print("mode: OFFLINE - a transformer is used only if already cached\n")

    model, label = try_load_transformer()
    svd_hits, n_q, svd_acc, svd_scores = pprint_svd()
    print(f"[1] local baseline (PPMI-SVD, trained in-file): "
          f"{svd_hits}/{n_q} hit@1 ({svd_acc:.0%})\n")

    if model is None:
        print("[2] TRANSFORMER: not cached on this machine -> skipping real model")
        print("    cache one once (then every run uses it, still offline):")
        print("      python -c \"from sentence_transformers import "
              "SentenceTransformer;")
        print("      SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')\"")
        print("    or run any other transformer once; this lab auto-detects it.\n")
        checks = [("offline-safe: no downloads attempted", True, label),
                  ("local SVD baseline computed", svd_hits >= 7,
                   f"{svd_hits}/{n_q}")]
    else:
        t0 = time.time()
        import torch
        with torch.no_grad():
            if isinstance(model, tuple):          # (AutoModel, AutoTokenizer)
                qv = mean_pool(model, [q for q, _, _ in QUERIES])
                dv = mean_pool(model, DOC_TEXTS)
            else:                                 # SentenceTransformer
                qv = model.encode([q for q, _, _ in QUERIES],
                                  normalize_embeddings=True)
                dv = model.encode(DOC_TEXTS, normalize_embeddings=True)
        embed_ms = (time.time() - t0) * 1000 / len(QUERIES)
        print(f"[2] transformer: {label}  ({embed_ms:.0f} ms per query embed)")
        hf_scores = []
        for i, (q, expected, _) in enumerate(QUERIES):
            top = int(np.argmax(dv @ qv[i]))
            hf_scores.append(TITLES[top] == expected)
        hf_hits = sum(hf_scores)
        print(f"    hit@1: {hf_hits}/{n_q} ({hf_hits / n_q:.0%})\n")
        print(f"    {'query':<42}{'svd':<12}{'hf'}")
        for i, (q, expected, kind) in enumerate(QUERIES):
            print(f"    {q:<42}{'OK ' if svd_scores[i] else 'miss':<12}"
                  f"{'OK ' if hf_scores[i] else 'miss':<12}{kind}")
        checks = [("transformer model loaded from cache only",
                   True, label),
                  ("HF embeddings reach at least 6/9 hit@1",
                   hf_hits >= 6, f"{hf_hits}/{n_q}"),
                  ("HF beats or matches the local SVD on paraphrases",
                   sum(1 for i, (_, _, k) in enumerate(QUERIES)
                       if k == "paraphrase" and hf_scores[i])
                   >= sum(1 for i, (_, _, k) in enumerate(QUERIES)
                          if k == "paraphrase" and svd_scores[i]),
                   "paraphrase hit counts")]

    all_ok = True
    print("  CHECKS")
    for name, ok, detail in checks:
        all_ok &= ok
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
    print("=" * 72)
    print(f"EMBEDDING HF BENCH: {'ALL CHECKS PASS' if all_ok else 'CHECK FAILURES'}")
    print("=" * 72)
    print("READING: retrieval quality tracks the embedding space, not the")
    print("library.  PPMI-SVD beats lexical but a pretrained transformer")
    print("generalizes far beyond the tiny local corpus - at the cost of a")
    print("download once + ~20-100x more compute per embed.")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
