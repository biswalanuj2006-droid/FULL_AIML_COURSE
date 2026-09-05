"""
RAG FAIL -> FIX DEMO (runnable without an LLM or network)
==========================================================
Reproduces the classic "RAG gives wrong answers" failure from
29_RAG/real_problem.txt and shows the fix, measured with the same
diagnostic harness described there.

The trick: we use a GENERATOR STUB that answers ONLY from the retrieved
chunk (like a perfectly grounded LLM). Therefore any wrong answer in
this demo is PROVABLY a retrieval failure - exactly the 80% case in
real life. In the naive pipeline retrieval fails on hard questions; in
the fixed pipeline it does not. A real LLM can then be swapped in for
the stub (see PROPER_GENERATOR at the bottom).

Run:  python code/rag/02_rag_fail_fix_demo.py
"""

import math
import re
from collections import Counter

# ----------------------------------------------------------------------
# 0. CORPUS: facts spread across documents (numbers are easy to verify)
# ----------------------------------------------------------------------
DOCS = {
    "vacation_policy": (
        "Our vacation policy allows 25 paid days per year after one year of "
        "employment. Employees in their first year receive 15 paid days. "
        "Vacation days cannot be carried over beyond March 31 of the next "
        "calendar year. Approval requires the manager's sign-off at least "
        "two weeks in advance."
    ),
    "reimbursement_policy": (
        "The travel reimbursement limit is 120 dollars per night for "
        "hotels and 60 dollars per day for meals. Reimbursement claims must "
        "be submitted within 30 days of the trip. Receipts are mandatory "
        "for amounts above 25 dollars. International trips require "
        "pre-approval from finance."
    ),
    "remote_work": (
        "Remote work is allowed up to three days per week. The company "
        "provides a one-time 500 dollar home-office allowance for remote "
        "employees. Core collaboration hours are 10am to 3pm in the "
        "employee's local timezone. Equipment must be returned if the "
        "employee leaves within six months of receiving it."
    ),
    "onboarding": (
        "New employees receive a company laptop on their first day. "
        "The onboarding checklist includes security training, which must be "
        "completed within the first week. Access to the financial systems "
        "is granted after the manager approves the access request, usually "
        "within three business days."
    ),
}

# ----------------------------------------------------------------------
# 1. CHUNKING
# ----------------------------------------------------------------------
def chunk_text(text: str, title: str, size: int = 40,
               overlap: int = 8) -> list:
    """Split into word windows with overlap; each chunk carries its source
    title (the FIX for 'model cannot tell two contradictory chunks apart')."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        piece = " ".join(words[i:i + size])
        chunks.append(f"[Source: {title}] {piece}")
        i += size - overlap
    return chunks


def build_corpus():
    chunks = []
    for title, text in DOCS.items():
        chunks += chunk_text(text, title)
    return chunks


# ----------------------------------------------------------------------
# 2. "EMBEDDINGS": TF-IDF vectors used as dense embeddings + BM25 hybrid
#    (lexical stand-in for a real embedder - the demo logic is identical)
# ----------------------------------------------------------------------
def tokenize(s: str):
    return re.findall(r"[a-z0-9]+", s.lower())


class ToyEmbedder:
    """Bag-of-words vectors with idf weighting; cosine = 'embedding'."""

    def __init__(self, corpus):
        df = Counter()
        self.docs = []
        for c in corpus:
            toks = tokenize(c)
            df.update(set(toks))
            self.docs.append(toks)
        self.n = len(corpus)
        self.idf = {t: math.log((self.n + 1) / (df[t] + 1)) + 1
                    for t in df}

    def idf_of(self, t):
        """Unseen query words get the maximum idf (they are rare and
        therefore informative) instead of a KeyError."""
        return self.idf.get(t, max(self.idf.values()))

    def vector(self, text):
        v = Counter(tokenize(text))
        norm = math.sqrt(sum((self.idf_of(t) * c) ** 2 for t, c in v.items()))
        if norm == 0:
            return {}
        return {t: self.idf_of(t) * c / norm for t, c in v.items()}

    def cosine(self, text, doc_idx):
        q = self.vector(text)
        if not q:
            return 0.0
        d = Counter(self.docs[doc_idx])
        norm = math.sqrt(sum((self.idf_of(t) * c) ** 2 for t, c in d.items()))
        if norm == 0:
            return 0.0
        return sum(q.get(t, 0) * self.idf_of(t) * c for t, c in d.items()) / norm


def bm25_like(q_toks, doc_toks, df, n, k1=1.5, b=0.75):
    """Standard BM25 score for one query vs one document."""
    dl = len(doc_toks)
    avgdl = sum(len(d) for d in df["_lens_"]) / n if n else 1
    tf = Counter(doc_toks)
    score = 0.0
    for t in set(q_toks):
        if t not in tf:
            continue
        f = tf[t]
        idf = math.log((n - df[t] + 0.5) / (df[t] + 0.5) + 1)
        score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
    return score


# ----------------------------------------------------------------------
# 3. THE TWO RETRIEVERS (naive vs fixed)
# ----------------------------------------------------------------------
def naive_retrieve(query, chunks, embed, top_k=2):
    """Pure cosine, top_k only. No titles matched, no hybrid, no rerank."""
    scored = sorted(((embed.cosine(query, i), i) for i in range(len(chunks))),
                    reverse=True)
    return [chunks[i] for _, i in scored[:top_k]]


def fixed_retrieve(query, chunks, embed, top_k=3):
    """Hybrid: dense cosine candidates (3x) reranked by BM25 lexical
    score. Catches both 'semantically close but wrong word' and
    'exact keyword but low cosine' failures."""
    n = len(chunks)
    df = Counter()
    all_docs = [tokenize(c) for c in chunks]
    for d in all_docs:
        df.update(set(d))
    df["_lens_"] = all_docs  # stash for avgdl
    q_toks = tokenize(query)
    dense_cands = sorted(((embed.cosine(query, i), i)
                          for i in range(n)), reverse=True)[:top_k * 3]
    sparse_cands = sorted(
        ((bm25_like(q_toks, all_docs[i], df, n), i)
         for _, i in dense_cands), reverse=True)
    return [chunks[i] for _, i in sparse_cands[:top_k]]


# ----------------------------------------------------------------------
# 4. GENERATOR STUB: answers ONLY from the retrieved chunks (grounded).
#    Like a real LLM it reads the whole retrieved context and may answer
#    "I don't know" when the fact is absent - so a WRONG answer here is
#    provably a RETRIEVAL failure (the fact never reached the generator).
#    With a real LLM, add the 'NOT IN CONTEXT' prompt + faithfulness gate
#    from 29_RAG/real_problem.txt on top of this same shape.
# ----------------------------------------------------------------------
def generate_grounded(query, chunks):
    context = "\n\n".join(chunks)
    if not context.strip():
        return "NOT IN CONTEXT"
    return context  # grounded answer = retrieved evidence


def answer_has_fact(answer, fact):
    return fact.lower() in answer.lower()


# ----------------------------------------------------------------------
# 5. THE EVAL HARNESS (the diagnostic from real_problem.txt STEP 3)
# ----------------------------------------------------------------------
GOLDEN = [
    ("How many vacation days in the first year?", "15 paid days"),
    ("What is the hotel reimbursement per night?", "120 dollars"),
    ("How many remote days per week are allowed?", "three days"),
    ("What is the home office allowance?", "500 dollar"),
    ("When must security training be completed?", "first week"),
    ("Meal reimbursement per day?", "60 dollars"),
    ("How many business days for financial access?", "three business"),
    ("What is the deadline to submit a claim?", "30 days"),
]


def evaluate(retrieve, label, chunks, embed):
    retr_hits, ans_hits = 0, 0
    print(f"\n--- {label} ---")
    for q, fact in GOLDEN:
        got = retrieve(q, chunks, embed)
        in_topk = any(answer_has_fact(c, fact) for c in got)
        ans = generate_grounded(q, got)
        correct = answer_has_fact(ans, fact)
        print(f"  {'OK ' if correct else 'WRONG'}  {q}")
        print(f"        retrieval hit: {in_topk} | answer right: {correct}")
        retr_hits += in_topk
        ans_hits += correct
    print(f"  ==> retrieval hit-rate {retr_hits}/{len(GOLDEN)} | "
          f"correct answers {ans_hits}/{len(GOLDEN)}")
    return retr_hits, ans_hits


def main():
    chunks = build_corpus()
    embed = ToyEmbedder(chunks)

    print("=" * 72)
    print("RAG WRONG-ANSWER DEMO: retrieve -> grounded-generate")
    print("=" * 72)

    # The failing pipeline (small chunks + pure cosine, no rerank)
    r1, a1 = evaluate(naive_retrieve, "NAIVE (small chunks, cosine only)",
                      chunks, embed)
    # The fixed pipeline (hybrid + rerank + source titles)
    r2, a2 = evaluate(fixed_retrieve, "FIXED (hybrid BM25+cosine, rerank)",
                      chunks, embed)

    print("\n" + "=" * 72)
    print(f"RESULT: naive {a1}/{len(GOLDEN)} correct answers  ->  "
          f"fixed {a2}/{len(GOLDEN)} correct answers")
    print("If answers are wrong while retrieval hits are high, the failure")
    print("is in GENERATION (see real_problem.txt stages 2/4); if retrieval")
    print("hits are low, it is in RETRIEVAL (chunking/embeddings) - as here.")
    print("=" * 72)


# ----------------------------------------------------------------------
# SWAP-IN FOR A REAL LLM (optional; needs an API key). Replace the stub's
# body with this shape plus the faithfulness gate from real_problem.txt.
# ----------------------------------------------------------------------
# def PROPER_GENERATOR(client, query, chunks):
#     context = "\n\n".join(chunks)
#     answer = client.chat.completions.create(
#         model="your-model",
#         temperature=0.0,
#         messages=[
#             {"role": "system",
#              "content": "Answer ONLY from the context. If the context "
#                          "lacks the answer reply exactly: NOT IN CONTEXT."},
#             {"role": "user",
#              "content": f"Context:\n{context}\n\nQuestion: {query}"},
#         ]).choices[0].message.content
#     return answer

if __name__ == "__main__":
    main()
