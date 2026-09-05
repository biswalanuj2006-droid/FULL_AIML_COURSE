# code/projects — Full Project Implementations

Full project **specifications** live in:

- `47_PROJECTS/` — Level 1/2/3 project briefs (problem, dataset, steps, deliverables)
- `48_CAPSTONE/` — end-to-end capstone briefs

This folder is where you **implement** those briefs. Suggested layout:

```
code/projects/
├── student_predictor/          # Level 1  (spec: 47_PROJECTS/LEVEL_1)
│   ├── data/                   # downloaded or generated data
│   ├── notebook.ipynb          # exploration + experiments
│   ├── train.py                # clean training script
│   ├── evaluate.py
│   ├── requirements.txt
│   └── README.md               # what it does + how to run it
├── fraud_detection/            # Level 2 (spec: 47_PROJECTS/LEVEL_2)
│   ├── src/
│   ├── tests/
│   └── api/                    # FastAPI wrapper around the model
├── rag_platform/               # Level 3 (spec: 47_PROJECTS/LEVEL_3)
│   ├── backend/                # FastAPI + PostgreSQL + vector DB
│   ├── frontend/               # Streamlit/React client
│   ├── Dockerfile
│   └── docker-compose.yml
└── capstones/                  # module 48_CAPSTONE
```

## Reference implementations you already have

| Concern | File |
|---|---|
| From-scratch ML | `code/01_*_from_scratch.py`, `code/ml/*_from_scratch.py` |
| NumPy/Pandas data prep | `code/numpy/`, `code/pandas/` |
| NLP baseline | `code/nlp/01_nlp_classic_pipeline.py` |
| Deep learning | `code/dl/`, `code/cnn/`, `code/rnn/` |
| RAG skeleton | `code/rag/01_rag_minimal.py` |
| ML API | `code/fastapi/01_fastapi_ml_api.py` |
| Containerization | `code/docker/` |
| Experiment tracking | `code/mlops/01_mlflow_example.py` |

## How a real project should be structured

Every finished project here should include (scaled to its level):

1. `README.md` — problem, architecture, dataset source, how to run
2. `requirements.txt` — pinned dependencies
3. `data/` — dataset + `data_dictionary.md`
4. training + evaluation scripts (reproducible, seeded)
5. tests for the critical functions (`pytest`)
6. a prediction API (FastAPI) once the model exists
7. `Dockerfile`/`docker-compose.yml` for Level 3
8. results + error analysis in a short report

**Rule: a project is done when a stranger can clone the repo, run one
command, and reproduce your numbers.**
