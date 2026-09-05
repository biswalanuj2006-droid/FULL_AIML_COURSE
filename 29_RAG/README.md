# Module 29: RAG (Retrieval-Augmented Generation)

Giving LLMs access to your documents: chunk → embed → retrieve → generate,
plus the engineering that makes it reliable.

## What You Will Learn

- The RAG pipeline end to end (diagram in `diagrams/rag/`)
- Document loading, parsing, and cleaning (PDFs, HTML, code)
- Chunking strategies: size, overlap, structure-aware
- Embeddings and similarity search
- Vector retrieval: dense, sparse, hybrid; reranking
- Context construction and prompt assembly
- Evaluation: retrieval quality, answer quality, hallucination checks
- Failure modes: missed context, stale data, injection
- Minimal RAG with FAISS/Chroma → full stack with LangChain/LlamaIndex
- Production concerns: citations, metadata filters, cost, latency

## Module Files

| File | Topic |
|------|-------|
| rag_complete.txt | Full theory → code progression |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

Runnable code: `code/rag/01_rag_minimal.py` (verified).
Related modules: 30_VECTOR_DATABASES, 34_MODEL_DEPLOYMENT.

## Prerequisites

- 28_LLM_FUNDAMENTALS, 24_WORD_EMBEDDINGS

## Exit Criteria

- [ ] You can explain every box of the RAG pipeline
- [ ] You built a PDF Q&A system with citations
- [ ] You can evaluate retrieval and answer quality quantitatively
