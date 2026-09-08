"""Course catalog: 14 courses with lessons wired to repo assets."""

# (slug, title, description, order,
#  lessons: [(title, summary, video, diagram)])

COURSES = [
    ("python", "Python for AI",
     "Syntax, OOP, iterators, generators, decorators, async and typing - the engineer's toolkit.", 1, [
        ("Execution model", "Interpreter, bytecode, the PVM", "Videos/gradient_descent.mp4", ""),
        ("Comprehensions & generators", "Lazy evaluation and memory wins", "", ""),
        ("Decorators & closures", "Function factories, the #1 interview topic", "", ""),
        ("Async & concurrency", "asyncio, threads vs processes, the GIL", "", ""),
    ]),
    ("numpy", "NumPy",
     "ndarrays, broadcasting, vectorization, memory layout - the substrate of ML.", 2, [
        ("Shapes, axes, dtypes", "The mental model that everything else builds on", "", "Images/math/broadcasting.png"),
        ("Broadcasting rules", "(3,1)+(1,4)->(3,4): stretch, never copy", "", "Images/math/broadcasting.png"),
        ("Views vs copies", "Why slices are free and copies are not", "", ""),
        ("Vectorization", "Replacing loops with ufuncs; numerical stability", "", ""),
    ]),
    ("pandas", "Pandas",
     "DataFrames end to end: indexing, groupby, merges, reshaping, datetime handling.", 3, [
        ("Series & DataFrame basics", "Indexing: loc, iloc, boolean masks", "", ""),
        ("groupby & aggregation", "Split-apply-combine, the workhorse", "", ""),
        ("Merges & joins", "SQL-style joins, validation, indicator", "", ""),
        ("Missing data & dtypes", "NaN semantics, nullable types, categoricals", "", ""),
    ]),
    ("mathematics", "Mathematics for AI",
     "Linear algebra, calculus, probability - with geometric intuition for each.", 4, [
        ("Vectors & dot products", "Projection, similarity, norms", "", "Images/math/sigmoid.jpg"),
        ("Eigenvalues & SVD", "What PCA actually computes", "", ""),
        ("Gradients & the chain rule", "Backprop is the chain rule, organized", "", "Images/math/gradient_descent.jpg"),
        ("Probability & Bayes", "Distributions, expectation, conditional probability", "", ""),
    ]),
    ("machine-learning", "Machine Learning",
     "The full supervised/unsupervised toolkit with honest failure analysis.", 5, [
        ("Linear & logistic regression", "Loss surfaces, regularization L1/L2", "", "Images/ml/decision_boundary.jpg"),
        ("Trees, forests, boosting", "XGBoost vs LightGBM vs CatBoost", "", "Images/ml/overfitting.jpg"),
        ("Clustering & PCA", "K-Means, DBSCAN, when PCA helps", "", "Images/ml/confusion_matrix.jpg"),
        ("Overfitting & validation", "Bias-variance, CV, leakage", "", "Images/ml/overfitting.jpg"),
    ]),
    ("deep-learning", "Deep Learning",
     "From the perceptron to residual networks and normalization strategies.", 6, [
        ("Forward & backward passes", "Computational graphs, autograd", "", "Images/dl/backpropagation.jpg"),
        ("Activations & losses", "ReLU family, softmax+CE, numerical stability", "", "Images/dl/activation_functions.jpg"),
        ("Normalization & regularization", "BatchNorm, LayerNorm, dropout, init", "", ""),
        ("Optimizers", "SGD, momentum, AdamW, schedules", "", "Images/dl/gradient_descent.jpg"),
    ]),
    ("nlp", "NLP",
     "Text processing, classic representations and neural sequence models.", 7, [
        ("Tokenization", "BPE, WordPiece, SentencePiece", "", "Images/nlp/tokenization.png"),
        ("TF-IDF & embeddings", "Sparse vs dense representations", "", ""),
        ("Sequence tasks", "NER, sentiment, classification heads", "", ""),
    ]),
    ("transformers", "Transformers",
     "Self-attention to GQA: the architecture that ate the field.", 8, [
        ("Scaled dot-product attention", "softmax(QK^T/sqrt(d))V - shapes and intuition", "", "Images/transformers/self_attention.jpg"),
        ("Causal masking", "Why decoders can't see the future", "", "Images/transformers/attention_mask.png"),
        ("MHA vs MQA vs GQA", "KV-cache memory: the production lever", "", "Images/transformers/mqa_gqa.png"),
        ("Modern stack", "RMSNorm, SwiGLU, RoPE", "", "Images/transformers/transformer_architecture.jpg"),
    ]),
    ("llm-engineering", "LLM Engineering",
     "Serving LLMs: sampling, context, cost, caching, fine-tuning.", 9, [
        ("Autoregressive generation", "Logits -> temperature/top-k/top-p", "Videos/temperature_sampling.mp4", "Images/llm/llm_architecture.png"),
        ("KV cache & inference", "Prefill vs decode, memory math", "Videos/ann_recall_latency.mp4", "Images/llm/kv_cache.png"),
        ("LoRA & fine-tuning", "W' = W + (a/r)BA - 48x fewer trainable params", "", "Images/llm/lora.png"),
        ("Structured outputs & tools", "JSON schemas, function calling", "", ""),
    ]),
    ("rag", "RAG",
     "The complete retrieval-augmented generation pipeline, ingestion to citations.", 10, [
        ("Ingestion & chunking", "Parse, clean, chunk size tradeoffs", "", "Images/rag/rag_lifecycle.png"),
        ("Embeddings & vector DBs", "Cosine similarity, ANN indexes", "Videos/topk_retrieval.mp4", "Images/rag/rag_pipeline.jpg"),
        ("Hybrid retrieval & reranking", "BM25 + dense + RRF, cross-encoders", "", "Images/rag/hybrid_rrf.png"),
        ("Grounded generation", "Abstention, citations, evaluation", "", "Images/rag/rerank_funnel.png"),
    ]),
    ("langchain-langgraph", "LangChain & LangGraph",
     "Composable LLM apps and stateful agent graphs.", 11, [
        ("LCEL runnables", "prompt | model | parser, streaming for free", "", "Images/langchain/lcel_runnable.png"),
        ("Structured output", "Pydantic schemas, validation loops", "", "Images/langchain/structured_output.png"),
        ("LangGraph state machines", "Nodes, edges, reducers, checkpointers", "", "Images/langgraph/graph_state.png"),
        ("Human-in-the-loop", "interrupt() and Command(resume=...)", "", "Images/langgraph/human_in_loop.png"),
    ]),
    ("ai-agents", "AI Agents",
     "ReAct loops, tool calling, memory and multi-agent systems.", 12, [
        ("The ReAct loop", "Thought -> Action -> Observation -> repeat", "", "Images/agents/react_loop.png"),
        ("Tools & function calling", "Schemas, validation, safe execution", "", ""),
        ("Memory types", "Short-term, episodic, semantic, procedural", "", "Images/agents/memory_types.png"),
        ("Multi-agent patterns", "Supervisor, planner-executor, map-reduce", "Videos/agent_react_loop.mp4", "Images/agents/multi_agent.png"),
    ]),
    ("mlops", "MLOps",
     "Tracking, registries, deployment, monitoring, drift and retraining.", 13, [
        ("Experiment tracking", "Params/metrics/artifacts you can diff", "", "Images/mlops/experiment_tracking.png"),
        ("Deployment patterns", "Batch vs online, shadow, canary", "", "Images/backend/api_flow.jpg"),
        ("Monitoring & drift", "PSI/KS, concept drift, retraining triggers", "", "Images/mlops/drift_retraining.png"),
    ]),
    ("ai-security", "AI Security",
     "Threat models for LLM apps and the defenses that actually work.", 14, [
        ("Prompt injection", "Direct vs indirect, instruction hierarchy", "", "Images/security/threat_model.png"),
        ("RAG poisoning", "Attacker-controlled documents, ingestion defenses", "", "Images/security/rag_poisoning.png"),
        ("Tool safety & isolation", "Least privilege, tenant isolation, audit", "", ""),
    ]),
]
