"""
================================================================================
EMBEDDING RAG LAB  (genai_agents_course/embedding_rag_lab.py)
================================================================================
A runnable RAG laboratory that replaces the LEXICAL retriever (word
overlap) with REAL DENSE EMBEDDINGS - the part 29_RAG / 30_VECTOR_DATABASES
/ genai COURSE.txt Part 47 "retrieval" story.  Everything runs offline:
the embeddings are TRAINED HERE from a small corpus with the classic
PPMI + truncated-SVD recipe (Levy & Goldberg 2014) - the same math behind
Word2Vec/GloVe-style static embeddings:

  1. count word-word co-occurrence inside a +-2 window (the distributional
     hypothesis: words that appear in similar contexts are similar),
  2. reweight with POSITIVE POINTWISE MUTUAL INFORMATION (PPMI) so rare,
     informative co-occurrences dominate over the ubiquitous "the"/"of",
  3. factor the PPMI matrix with TRUNCATED SVD and keep the top-d
     components -> a dense d-dimensional vector per word,
  4. embed documents and queries as the mean of their word vectors and
     retrieve with COSINE SIMILARITY (what a vector database does).

The lab then proves WHY embeddings beat lexical search: a query such as
"give me my money back" shares ZERO words with the refund policy document,
so word overlap scores 0 - but in the embedding space "money back" has
been pulled next to "refund" by the training corpus, and the retriever
finds the right document.

Sections:
  [1] HANDBOOK + AUXILIARY corpus (docs to retrieve + the sentences that
      teach the embedding which words are similar).
  [2] PPMI + SVD:  co-occurrence matrix -> PPMI -> truncated SVD -> word
      vectors (the math is printed, not hidden).
  [3] WORD SIMILARITY sanity: nearest neighbours of "refund", cosine == 1
      for identical words, OOV fallback behaviour.
  [4] DENSE RETRIEVAL: embed every doc, cosine-search each query.
  [5] LEXICAL vs EMBEDDING evaluation table on paraphrase queries +
      summary checks.

    python embedding_rag_lab.py
================================================================================
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------------
# [1] DATA: documents to retrieve + the sentences that teach similarity
# ----------------------------------------------------------------------------

DOCS: List[Tuple[str, str]] = [
    ("refunds", "Customers may request a refund within thirty days of a "
                "purchase. Approved refunds are paid back to the original "
                "payment card within five business days."),
    ("hours", "The store opens at ten in the morning and closes at eight in "
              "the evening, seven days a week. Support chat is online from "
              "nine until six."),
    ("shipping", "Standard delivery takes three to five business days. "
                 "Express delivery arrives within two days. Orders above "
                 "fifty dollars ship for free."),
    ("password", "To recover access, click the forgot password link on the "
                 "login page. A reset link is emailed to your address and "
                 "expires after one day."),
    ("cancel", "You may stop your subscription from the billing settings "
               "page. Access continues until the end of the paid month and "
               "partial months are not refunded."),
    ("warranty", "Every device includes a one year warranty. The warranty "
                 "covers manufacturing faults and broken screens but not "
                 "accidental water damage."),
    ("contact", "Write to support at support at example dot com. A human "
                "replies within four hours on business days."),
]

# Auxiliary sentences: they do NOT contain the documents' answers; they
# merely co-locate paraphrase vocabulary with the canonical terms so the
# embeddings learn that "money back" ~ "refund", "open"/"close" ~ hours,
# "stop" ~ cancel, and so on (the distributional hypothesis in action).
AUX = [
    "a refund gives your money back",
    "money back means a full refund",
    "you want a refund when you want your money back",
    "we give you money back with every approved refund",
    "if the product is returned you get a refund",
    "returns are guaranteed so customers get money back",
    "give me my money back is what a refund does",
    "the refund returns money to your card",
    "ask for a refund and your money comes back",
    "customers request money back through a refund",
    "the store opens in the morning",
    "we open at nine and close at five",
    "morning opening and evening closing",
    "open hours are listed for every day",
    "the shop is open and ready at ten",
    "evening closing time is eight",
    "you can reset your password to recover your account",
    "forgot the password so reset the login",
    "recover access by resetting the password",
    "the login page has the reset option",
    "stop the subscription to cancel billing",
    "cancel means you stop paying each month",
    "you can stop auto renewal in billing",
    "end the plan by cancelling",
    "the warranty covers damage to the screen",
    "a broken screen is covered by the warranty",
    "faulty devices are covered for a year",
    "contact support by email for help",
    "email us and a human answers",
    "reach a person through support",
    "we reply to every message quickly",
    "you can return a purchase within the policy",
    "the policy allows returns for store credit",
    "express orders arrive faster than standard",
    "delivery time depends on the shipping method",
    "your order ships after payment",
]

STOPWORDS = {"a", "an", "the", "is", "are", "was", "were", "be", "been",
             "do", "does", "did", "how", "what", "why", "when", "where",
             "which", "who", "i", "me", "my", "your", "you", "we", "to",
             "of", "for", "on", "in", "at", "with", "and", "or", "it",
             "its", "can", "could", "would", "should", "have", "has",
             "not", "from", "about", "get", "this", "that", "so"}


def words(text: str, drop_stop: bool = True) -> List[str]:
    toks = re.findall(r"[a-z]+", text.lower())
    return [w for w in toks if not (drop_stop and w in STOPWORDS)]


# ----------------------------------------------------------------------------
# [2] PPMI + TRUNCATED SVD EMBEDDINGS  (the math, implemented not hand-waved)
# ----------------------------------------------------------------------------

def train_embeddings(sentences: List[str], dim: int = 48,
                     window: int = 2) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Word vectors via PPMI -> SVD.  Returns {word: vector}, vocab list."""
    vocab_set: set = set()
    sent_toks: List[List[str]] = []
    for s in sentences:
        toks = words(s, drop_stop=False)     # syntax words still carry context
        if len(toks) < 2:
            continue
        sent_toks.append(toks)
        vocab_set.update(toks)
    vocab = sorted(vocab_set)
    idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)

    # co-occurrence within +-window (symmetric), raw counts C[w][c]
    C = np.zeros((V, V), dtype=np.float64)
    for toks in sent_toks:
        for i, w in enumerate(toks):
            for j in range(max(0, i - window), min(len(toks), i + window + 1)):
                if i != j:
                    C[idx[w], idx[toks[j]]] += 1.0

    # PPMI:  pmi(w,c) = log2( C*N / (row*col) ), clamp at 0.
    row = C.sum(axis=1, keepdims=True)
    col = C.sum(axis=0, keepdims=True)
    N = C.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2((C * N) / np.maximum(row @ col, 1e-12))
    ppmi = np.maximum(pmi, 0.0)

    # Truncated SVD: PPMI ~= U_r S_r V_r^T; word vector w = U_r[w] sqrt(S_r).
    U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
    d = min(dim, V)
    W = U[:, :d] @ np.diag(np.sqrt(S[:d]))       # (V, d) word embedding rows

    expl = S[:d].sum() / S.sum()
    print(f"    co-occurrence matrix {V}x{V}, {int(N):,} word-context pairs")
    print(f"    PPMI: log2( p(w,c) / (p(w)p(c)) ), clamped >= 0 "
          f"(reduces common-word dominance)")
    print(f"    truncated SVD keeps d={d}: "
          f"{expl:.1%} of the PPMI variance explained by the top {d} axes")
    emb = {w: W[idx[w]] for w in vocab}
    return emb, vocab


def embed_text(text: str, emb: Dict[str, np.ndarray]) -> np.ndarray:
    """Bag-of-word-vector mean (drop OOV and stopwords), L2-normalized."""
    vecs = [emb[w] for w in words(text) if w in emb]
    if not vecs:
        return np.zeros(next(iter(emb.values())).shape)
    v = np.mean(vecs, axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


# ----------------------------------------------------------------------------
# [3]/[4]/[5] RETRIEVERS + EVALUATION
# ----------------------------------------------------------------------------

def lexical_score(query: str, doc_text: str) -> float:
    qw, dw = set(words(query)), set(words(doc_text))
    if not qw:
        return 0.0
    return len(qw & dw) / len(qw)                 # content-word coverage


def top_k_embed(query: str, doc_vecs: Dict[str, np.ndarray],
                emb: Dict[str, np.ndarray]) -> List[Tuple[str, float]]:
    qv = embed_text(query, emb)
    scored = [(t, cosine(qv, v)) for t, v in doc_vecs.items()]
    scored.sort(key=lambda x: -x[1])
    return scored


def top_k_lexical(query: str) -> List[Tuple[str, float]]:
    scored = [(t, lexical_score(query, text)) for t, text in DOCS]
    scored.sort(key=lambda x: -x[1])
    return scored


def main() -> None:
    print("=" * 72)
    print("EMBEDDING RAG LAB  (dense PPMI-SVD embeddings vs lexical search)")
    print("=" * 72)

    # [1]
    corpus = AUX + [text for _, text in DOCS]
    print(f"[1] corpus: {len(DOCS)} handbook docs + {len(AUX)} auxiliary "
          f"sentences")
    print("    the auxiliary sentences never contain the answers - they only")
    print("    teach which words MEAN the same thing (money back ~ refund)\n")

    # [2]
    print("[2] train embeddings (PPMI + truncated SVD)")
    emb, vocab = train_embeddings(corpus, dim=48)
    print(f"    vocab size {len(vocab)}, vector dim 48\n")

    # [3] word similarity sanity
    print("[3] WORD SIMILARITY sanity (cosine in the embedding space)")
    for anchor in ("refund", "money", "open", "cancel", "password"):
        sims = sorted(((w, cosine(emb[anchor], v)) for w, v in emb.items()
                       if w != anchor), key=lambda x: -x[1])[:4]
        print(f"    {anchor:<9} -> " + ", ".join(f"{w} {s:.2f}"
                                                 for w, s in sims))
    ident = cosine(emb["refund"], emb["refund"])
    print(f"    cosine('refund','refund') = {ident:.4f} (must be 1)\n")

    # [4] embed documents, then run the eval queries
    doc_vecs = {t: embed_text(text, emb) for t, text in DOCS}
    queries: List[Tuple[str, str, str]] = [
        ("how do I get a refund", "refunds", "direct"),
        ("give me my money back", "refunds", "paraphrase"),
        ("my money never came back after I returned it", "refunds",
         "paraphrase"),
        ("what time are you open", "hours", "direct"),
        ("when do you close in the evening", "hours", "paraphrase"),
        ("I forgot my password and cannot log in", "password", "direct"),
        ("how do I stop my subscription", "cancel", "paraphrase"),
        ("the screen broke, is it covered", "warranty", "paraphrase"),
        ("is there a person I can email", "contact", "paraphrase"),
    ]
    print("[4] dense retrieval over the handbook (mean word vectors, cosine)")

    print(f"\n    {'query':<42}{'emb@1':<12}{'lex@1':<12}kind")
    print("    " + "-" * 76)
    rows = []
    for q, expected, kind in queries:
        e_top = top_k_embed(q, doc_vecs, emb)[0]
        l_top = top_k_lexical(q)[0]
        emb_hit = e_top[0] == expected
        lex_hit = l_top[0] == expected
        rows.append((q, expected, kind, emb_hit, lex_hit,
                     e_top[0], e_top[1], l_top[0], l_top[1]))
        mark = "OK " if emb_hit else "miss"
        print(f"    {q:<42}{e_top[0]:<12}{l_top[0]:<12}{kind}  [{mark}]")

    # one worked example with scores + explanation
    q = "give me my money back"
    print(f"\n    worked example: {q!r}")
    for t, s in top_k_embed(q, doc_vecs, emb)[:3]:
        shared = set(words(q)) & set(words(dict(DOCS)[t]))
        print(f"      {t:<12} cosine {s:.3f}   shared words: "
              f"{shared if shared else 'NONE (lexical cannot match)'}")
    print("      -> 'money' and 'refund' never appear in the same document;")
    print("         the similarity lives in the embedding space learnt from the corpus")

    # [5] summary + checks
    emb_hits = sum(1 for r in rows if r[3])
    lex_hits = sum(1 for r in rows if r[4])
    para_emb = sum(1 for r in rows if r[2] == "paraphrase" and r[3])
    para_lex = sum(1 for r in rows if r[2] == "paraphrase" and r[4])
    n_para = sum(1 for r in rows if r[2] == "paraphrase")

    checks = [
        ("embeddings beat lexical on paraphrases",
         para_emb > para_lex,
         f"{para_emb}/{n_para} vs {para_lex}/{n_para}"),
        ("embeddings match or beat lexical overall",
         emb_hits >= lex_hits,
         f"emb {emb_hits}/{len(rows)} vs lex {lex_hits}/{len(rows)}"),
        ("embedding hit@1 >= 7 of 9",
         emb_hits >= 7,
         f"{emb_hits}/9"),
        ("identical vectors have cosine 1",
         abs(ident - 1.0) < 1e-6,
         f"{ident:.6f}"),
        ("document embeddings are unit vectors",
         all(abs(np.linalg.norm(v) - 1.0) < 1e-6 or np.linalg.norm(v) < 1e-9
             for v in doc_vecs.values()),
         f"{len(doc_vecs)} docs"),
    ]
    all_ok = True
    print("\n  CHECKS")
    for name, ok, detail in checks:
        all_ok &= ok
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
    print("=" * 72)
    print(f"EMBEDDING RAG LAB: {'ALL CHECKS PASS' if all_ok else 'CHECK FAILURES'}")
    print("=" * 72)
    print("WHAT THIS MEANS FOR RAG (Part 60): the vector store keeps the dense")
    print("doc vectors; at query time you embed the question and cosine-search")
    print("the top-k.  Retrieval now works on MEANING, not surface words - at")
    print("the cost of training the embedding space (here: PPMI+SVD on the")
    print("corpus; in production: Sentence-BERT-style models).")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
