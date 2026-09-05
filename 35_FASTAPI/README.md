# Module 35: FastAPI

Modern Python API framework — typed, async, auto-documented. The default
choice for serving ML models.

## What You Will Learn

- Routing, path/query params, and typed request/response models
- Pydantic validation and why it prevents model bugs
- Dependencies and dependency injection
- Async endpoints and background tasks
- Middleware, CORS, error handling
- JWT authentication integration (cross-ref Module 39)
- File uploads and serving static outputs
- Auto-generated OpenAPI docs (/docs)
- Testing with FastAPI's TestClient
- Building and deploying an ML inference API

## Module Files

| File | Topic |
|------|-------|
| fastapi_deep_dive.txt | Full FastAPI course |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

Runnable code: `code/fastapi/01_fastapi_ml_api.py`, plus the full
`code/docker/` setup.

## Prerequisites

- 38_REST_APIS (HTTP concepts)
- Any trained model from Modules 09-16

## Exit Criteria

- [ ] You can build a validated, documented prediction API
- [ ] You can write API tests with TestClient
- [ ] Your API runs in Docker with a health endpoint
