# Python Libraries for AI/ML — Complete Library Map

## Visual Library Map

```
PYTHON ECOSYSTEM FOR AI/ML
│
├── NUMERICAL COMPUTING
│   └── NumPy                    ★★★ CORE — Foundation of everything
│
├── DATA MANIPULATION
│   ├── Pandas                   ★★★ CORE — Data analysis standard
│   ├── Polars                   ★★☆ IMPORTANT — Fast alternative to Pandas
│   └── PyArrow                  ★☆☆ OPTIONAL — Columnar memory format
│
├── SCIENTIFIC COMPUTING
│   ├── SciPy                    ★★☆ IMPORTANT — Scientific algorithms
│   └── Statsmodels              ★☆☆ OPTIONAL — Statistical modeling
│
├── VISUALIZATION
│   ├── Matplotlib               ★★★ CORE — Foundation plotting
│   ├── Seaborn                  ★★☆ IMPORTANT — Statistical plots
│   └── Plotly                   ★★☆ IMPORTANT — Interactive charts
│
├── MACHINE LEARNING
│   ├── Scikit-learn             ★★★ CORE — ML standard library
│   ├── XGBoost                  ★★★ CORE — Gradient boosting king
│   ├── LightGBM                 ★★☆ IMPORTANT — Fast gradient boosting
│   └── CatBoost                 ★★☆ IMPORTANT — Best categorical support
│
├── DEEP LEARNING
│   ├── PyTorch                  ★★★ CORE — Research & production standard
│   └── TensorFlow/Keras         ★★☆ IMPORTANT — Production & deployment
│
├── COMPUTER VISION
│   ├── OpenCV                   ★★★ CORE — CV standard library
│   ├── Pillow                   ★★☆ IMPORTANT — Image I/O & basic ops
│   ├── torchvision              ★★☆ IMPORTANT — PyTorch vision tools
│   └── YOLO ecosystem           ★★☆ IMPORTANT — Object detection
│
├── NLP
│   ├── NLTK                     ★★☆ IMPORTANT — Classical NLP
│   ├── spaCy                    ★★☆ IMPORTANT — Production NLP
│   ├── Gensim                   ★☆☆ OPTIONAL — Topic modeling
│   └── Transformers (HF)        ★★★ CORE — Modern NLP standard
│
├── GENERATIVE AI / LLMs
│   ├── Transformers (HF)        ★★★ CORE — Model hub & inference
│   ├── Datasets (HF)            ★★☆ IMPORTANT — Dataset loading
│   ├── Tokenizers (HF)          ★★☆ IMPORTANT — Tokenization
│   ├── Sentence Transformers    ★★★ CORE — Embeddings & semantic search
│   ├── LangChain                ★★☆ IMPORTANT — LLM orchestration
│   └── LlamaIndex               ★★☆ IMPORTANT — RAG framework
│
├── VECTOR SEARCH
│   ├── FAISS                    ★★★ CORE — Facebook's vector search
│   ├── Chroma                   ★★☆ IMPORTANT — Easy vector DB
│   ├── Qdrant                   ★★☆ IMPORTANT — Production vector DB
│   └── pgvector                 ★★☆ IMPORTANT — PostgreSQL vectors
│
├── API / BACKEND
│   ├── FastAPI                  ★★★ CORE — Modern Python API framework
│   ├── Flask                    ★★☆ IMPORTANT — Lightweight API framework
│   ├── Django                   ★★☆ IMPORTANT — Full-stack web framework
│   ├── Pydantic                 ★★★ CORE — Data validation (FastAPI dep)
│   ├── Requests                 ★★★ CORE — HTTP client
│   └── HTTPX                    ★☆☆ OPTIONAL — Async HTTP client
│
├── DATABASES
│   ├── SQLAlchemy               ★★★ CORE — Python SQL toolkit
│   ├── psycopg2/psycopg         ★★☆ IMPORTANT — PostgreSQL driver
│   ├── Alembic                  ★★☆ IMPORTANT — Database migrations
│   └── Redis (redis-py)         ★★☆ IMPORTANT — Caching & sessions
│
├── STREAMING
│   └── kafka-python              ★☆☆ OPTIONAL — Kafka client
│
├── MLOps
│   ├── MLflow                   ★★☆ IMPORTANT — Experiment tracking
│   ├── DVC                      ★☆☆ OPTIONAL — Data versioning
│   └── Optuna                   ★★☆ IMPORTANT — Hyperparameter tuning
│
├── EXPLAINABILITY
│   ├── SHAP                     ★★☆ IMPORTANT — Model explanations
│   └── LIME                     ★☆☆ OPTIONAL — Local explanations
│
├── TESTING
│   ├── pytest                   ★★★ CORE — Testing framework
│   └── unittest                 ★☆☆ OPTIONAL — Built-in testing
│
├── APPS / DASHBOARDS
│   ├── Streamlit                ★★☆ IMPORTANT — Rapid AI apps
│   └── Gradio                   ★★☆ IMPORTANT — ML demos
│
└── PERFORMANCE
    ├── Numba                    ★☆☆ OPTIONAL — JIT compilation
    └── Cython                   ★☆☆ OPTIONAL — C extensions
```

## Priority Levels

| Level | Meaning | Action |
|-------|---------|--------|
| ★★★ CORE | Must learn. Used in every ML project. | Learn deeply |
| ★★☆ IMPORTANT | Should learn. Used in many projects. | Learn practically |
| ★☆☆ OPTIONAL | Good to know. Used in specific scenarios. | Learn basics |

---

## Library Comparison Tables

### Data Manipulation

| Feature | Pandas | Polars | PyArrow |
|---------|--------|--------|---------|
| **Speed** | Moderate | Fast (Rust backend) | Fast |
| **Memory** | Higher | Lower (lazy eval) | Lower (columnar) |
| **Ease of use** | Excellent | Good (different API) | Low-level |
| **Ecosystem** | Huge | Growing | Foundation for others |
| **Lazy evaluation** | No | Yes | Yes |
| **Multi-threading** | Limited (GIL) | Yes | Yes |
| **Best for** | General analysis | Large datasets, speed | Interoperability |
| **When to use** | Default choice | >1GB datasets | Arrow ecosystem |

### Machine Learning

| Feature | Scikit-learn | XGBoost | LightGBM | CatBoost |
|---------|-------------|---------|----------|----------|
| **Type** | General ML | Gradient Boosting | Gradient Boosting | Gradient Boosting |
| **Speed** | Moderate | Fast | Fastest | Fast |
| **Categorical** | Manual encoding | Basic | Basic | Native |
| **Accuracy** | Good | Excellent | Excellent | Excellent |
| **Ease of use** | Excellent | Good | Good | Excellent |
| **GPU support** | No | Yes | Yes | Yes |
| **Best for** | Prototyping | Tabular data | Large datasets | Categorical data |
| **When to use** | Baselines, pipelines | Default booster | Speed-critical | Many categoricals |

### Deep Learning

| Feature | PyTorch | TensorFlow/Keras |
|---------|---------|-----------------|
| **Type** | Dynamic graph | Static graph (TF2 eager) |
| **Ease of learning** | Moderate | Easy (Keras) |
| **Research** | Dominant | Declining |
| **Production** | Growing (TorchServe) | Strong (TF Serving) |
| **Mobile** | PyTorch Mobile | TF Lite (mature) |
| **Community** | Largest (research) | Large (industry) |
| **Hugging Face** | Primary support | Supported |
| **Best for** | Research, custom models | Production, deployment |
| **When to use** | Default for new projects | Existing TF infrastructure |

### Visualization

| Feature | Matplotlib | Seaborn | Plotly |
|---------|-----------|---------|--------|
| **Type** | Base library | Statistical (built on MPL) | Interactive |
| **Static output** | Excellent | Excellent | Good |
| **Interactive** | No | No | Yes |
| **Learning curve** | Moderate | Easy | Easy |
| **Customization** | Full control | Limited | Moderate |
| **Web-ready** | No | No | Yes |
| **Best for** | Publication plots | Statistical EDA | Dashboards, web |
| **When to use** | Default, full control | Quick statistical plots | Interactive apps |

### NLP Libraries

| Feature | NLTK | spaCy | Transformers (HF) |
|---------|------|-------|-------------------|
| **Type** | Classical NLP | Production NLP | Modern NLP (DL) |
| **Speed** | Slow | Fast | Slow (GPU helps) |
| **Accuracy** | Moderate | Good | State-of-the-art |
| **Models** | Rule-based | Small neural | Large pretrained |
| **GPU** | No | Optional | Yes |
| **Best for** | Learning, education | Production text processing | SOTA NLP tasks |
| **When to use** | Teaching concepts | Text preprocessing | Classification, NER, generation |

### Vector Databases

| Feature | FAISS | Chroma | Qdrant | pgvector |
|---------|-------|--------|--------|----------|
| **Type** | Library | Database | Database | PostgreSQL ext |
| **Setup** | pip install | pip install | Docker | Extension |
| **Persistence** | File-based | File-based | Server | PostgreSQL |
| **Metadata** | Limited | Yes | Yes | Yes |
| **Scale** | Millions | Thousands-Millions | Millions+ | Millions |
| **Ease of use** | Moderate | Easy | Easy | SQL knowledge |
| **Best for** | Research, speed | Prototyping | Production | Existing PG setup |
| **When to use** | Large-scale search | Quick start | Production RAG | Already using PostgreSQL |

### Backend Frameworks

| Feature | FastAPI | Flask | Django |
|---------|---------|-------|--------|
| **Type** | Async API | Micro framework | Full-stack |
| **Speed** | Fastest | Moderate | Moderate |
| **Learning curve** | Easy | Very easy | Steep |
| **API docs** | Auto (OpenAPI) | Manual | DRF extension |
| **ORM** | SQLAlchemy | SQLAlchemy | Built-in ORM |
| **Auth** | Manual/JWT | Manual | Built-in |
| **Best for** | ML APIs | Simple APIs | Full web apps |
| **When to use** | Default for AI APIs | Quick prototypes | Complex web apps |

### HTTP Clients

| Feature | Requests | HTTPX |
|---------|----------|-------|
| **Async** | No | Yes |
| **Speed** | Good | Good |
| **Ease of use** | Excellent | Excellent (similar API) |
| **Features** | Mature, stable | Modern, async |
| **Best for** | Synchronous calls | Async applications |
| **When to use** | Default choice | Need async |

---

## From-Scratch → Library Progression

For every major algorithm, implement at multiple levels:

```
Level 1: MATHEMATICAL FORMULA
    y = wx + b

Level 2: PURE PYTHON
    def linear_regression(X, y):
        n = len(X)
        sum_xy = sum(x*y for x, y in zip(X, y))
        sum_x = sum(X)
        sum_y = sum(y)
        sum_x2 = sum(x*x for x in X)
        w = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x**2)
        b = (sum_y - w*sum_x) / n
        return w, b

Level 3: NUMPY
    def linear_regression_np(X, y):
        X_b = np.c_[np.ones(len(X)), X]
        w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        return w[1], w[0]

Level 4: SCIKIT-LEARN
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)
    print(model.coef_[0], model.intercept_)

Level 5: PRODUCTION
    # Full pipeline with preprocessing, evaluation, API, deployment
```

---

## Integration Map

Shows which libraries work together:

```
DATA PIPELINE
    Pandas/Polars → Scikit-learn Pipeline → Joblib → FastAPI

ML PIPELINE
    NumPy → Scikit-learn → XGBoost → MLflow → Docker

DEEP LEARNING PIPELINE
    PyTorch → DataLoader → Training Loop → Checkpoint → TorchServe

NLP PIPELINE
    spaCy (preprocessing) → Transformers (model) → Sentence Transformers (embeddings)

RAG PIPELINE
    Document Loaders → Text Splitters → Embeddings → Vector DB → LLM → FastAPI

CV PIPELINE
    OpenCV (preprocessing) → PyTorch (model) → ONNX (export) → FastAPI

PRODUCTION STACK
    FastAPI + SQLAlchemy + PostgreSQL + Redis + Docker + MLflow
```
