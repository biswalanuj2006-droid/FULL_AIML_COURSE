# Module 34: Model Deployment

Getting trained models in front of real traffic: serialization, serving,
scaling, monitoring, and safe rollouts.

## What You Will Learn

- Batch vs online vs edge vs streaming deployment
- Serialization: pickle, joblib, ONNX, TorchScript, SavedModel
- Inference servers: TF Serving, TorchServe, Triton, ONNX Runtime
- Serving patterns: in-process vs microservice vs serverless
- REST/GRPC model APIs with validation and versioning
- Dockerizing model services; memory and worker sizing
- Scaling: caching, batching, quantization, horizontal scaling
- Blue/green, canary, shadow, and rollback
- Monitoring: latency, errors, data/concept drift
- Deploying DL/NLP/RAG systems; edge deployment concepts

## Module Files

| File | Topic |
|------|-------|
| model_deployment_complete.txt | Full deployment course |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

Runnable code: `code/fastapi/`, `code/flask/`, `code/docker/`, `code/mlops/`.

## Prerequisites

- 09/10/19 (a real model worth deploying)
- 41_DOCKER, 35_FASTAPI recommended

## Exit Criteria

- [ ] You deployed a model behind a versioned API with a health check
- [ ] You can do a canary rollout and rollback
- [ ] You can monitor drift and latency on your deployed model
