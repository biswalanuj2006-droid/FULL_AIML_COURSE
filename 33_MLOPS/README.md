# Module 33: MLOps

The engineering layer around ML: tracking experiments, versioning data and
models, and automating the path from notebook to production.

## What You Will Learn

- The ML lifecycle and where things break (notebook-to-production gap)
- Experiment tracking: MLflow runs, metrics, params, artifacts
- Model registry and model versioning
- Data versioning (DVC concepts) for reproducibility
- Reproducibility: seeds, environments, dependency pinning
- CI/CD for ML: tests, training pipelines, deployment gates
- Model serving options recap (cross-ref Module 34)
- Monitoring: drift, metrics, retraining triggers
- Git/GitHub + Docker integration in ML workflows

## Module Files

| File | Topic |
|------|-------|
| mlops_libraries.txt | MLflow/DVC/W&B tool course |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

Runnable code: `code/mlops/01_mlflow_example.py`.

## Prerequisites

- 09/10/14 (real models to track)
- 41_DOCKER, 42_GIT_GITHUB basics recommended

## Exit Criteria

- [ ] You can log an experiment and compare runs in MLflow
- [ ] You can version a dataset + model and reproduce an old result
- [ ] You can sketch a CI/CD pipeline for a model
