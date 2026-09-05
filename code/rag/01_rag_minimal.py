# ============================================================
# MINIMAL RAG — RETRIEVAL-AUGMENTED GENERATION, STEP BY STEP
# The whole idea of RAG in one script:
#   documents -> chunk -> embed -> index -> retrieve -> prompt
#
# This version is dependency-light on purpose: it uses TF-IDF
# vectors as stand-in embeddings so you can run it anywhere.
# Swap the vectorizer for sentence-transformers / OpenAI
# embeddings to get real semantic search (code comment below).
#
# Run: python 01_rag_minimal.py
# Requires: scikit-learn (only)
# ============================================================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# 1. Knowledge base (pretend these are chunks of a manual)
# ------------------------------------------------------------
chunks = [
    "RAG retrieves relevant document chunks and inserts them into the prompt.",
    "Embeddings map text to vectors; nearby vectors mean similar meaning.",
    "Vector databases like FAISS, Qdrant and pgvector index embeddings for fast search.",
    "Chunking splits long documents so each chunk is focused and retrievable.",
    "A reranker scores retrieved chunks and keeps only the most relevant few.",
    "Citations in RAG answers should point back to the exact source chunk.",
    "Hallucination risk drops when the model must answer from retrieved context.",
    "Hybrid search combines dense embeddings with sparse keyword (BM25) matching.",
]

# ------------------------------------------------------------
# 2. Embed + index
# ------------------------------------------------------------
# REAL embeddings: replace TfidfVectorizer with
#   from sentence_transformers import SentenceTransformer
#   embedder = SentenceTransformer("all-MiniLM-L6-v2")
#   X = embedder.encode(chunks)
# TF-IDF is a sparse bag-of-words vector: zero semantics, cheap demo.
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(chunks)

# ------------------------------------------------------------
# 3. Retrieve (top-k by cosine similarity)
# ------------------------------------------------------------
def retrieve(query, k=3):
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, X)[0]
    top = sims.argsort()[::-1][:k]
    return [(sims[i], chunks[i]) for i in top]


query = "how do vector databases speed up similarity search?"
hits = retrieve(query)
print(f"Query: {query}\n")
for score, chunk in hits:
    print(f"  [{score:.2f}] {chunk}")

# ------------------------------------------------------------
# 4. Build the prompt (what actually goes to the LLM)
# ------------------------------------------------------------
context = "\n".join(f"- {chunk}" for _, chunk in hits)
prompt = f"""Answer the question using ONLY the context below.
If the context does not contain the answer, say so.

Context:
{context}

Question: {query}
Answer:"""
print("\n--- PROMPT SENT TO THE LLM ---\n")
print(prompt)

# ------------------------------------------------------------
# The real production stack (module 29_RAG):
#   document loaders (pdf/docx) -> chunkers -> real embeddings
#   -> vector DB (pgvector/Qdrant) -> hybrid retrieval + reranker
#   -> LLM call with citations -> evaluation (faithfulness, MRR)
# This file is the skeleton every production RAG builds on.
# ============================================================
