# Module 30: Vector Databases

Storing and searching embeddings at scale — the retrieval layer under RAG,
semantic search, and recommendation.

## What You Will Learn

- Why regular databases fail at similarity search
- Embeddings as points; similarity measures (cosine, L2, inner product)
- ANN search: HNSW, IVF, PQ — speed/recall trade-offs
- FAISS: index types, training, search, GPU notes
- Chroma (embedded, easy) vs Qdrant/Weaviate (server) vs pgvector
- Metadata filtering combined with vector search
- Insertion, deletion, and index rebuild semantics
- Choosing a vector DB: scale, cost, ops, filtering needs
- Realistic benchmarks and common failure modes

## Module Files

| File | Topic |
|------|-------|
| vector_databases_complete.txt | Full theory → hands-on |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

## Prerequisites

- 24_WORD_EMBEDDINGS / sentence embeddings
- 29_RAG motivation

## Exit Criteria

- [ ] You can compare FAISS vs Chroma vs Qdrant vs pgvector on real criteria
- [ ] You can build a search index with metadata filtering
- [ ] You know recall vs latency trade-offs in index choice
