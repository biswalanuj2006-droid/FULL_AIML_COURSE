# Module 41: Docker

Packaging your AI app so it runs anywhere: images, containers, Compose,
and multi-service stacks for ML systems.

## What You Will Learn

- Images vs containers; the container model
- Writing Dockerfiles: layers, caching, best practices
- Building, running, tagging, and pushing images
- Ports, volumes, environment variables, and secrets
- Networking between containers
- Docker Compose for multi-service stacks
- Containerizing ML apps: model files, deps, memory limits
- GPU containers (nvidia-container-toolkit concept)
- Multi-stage builds and image size optimization
- Debugging containers: logs, exec, health checks

## Module Files

| File | Topic |
|------|-------|
| docker_complete.txt | Full Docker course |
| practice.txt | Exercises |
| project.txt | Level 1-3 projects |

Runnable code: `code/docker/` (FastAPI app + Dockerfile + compose).

## Prerequisites

- 35_FASTAPI or any runnable service to containerize

## Exit Criteria

- [ ] You can write a Dockerfile and explain each line
- [ ] You can run a multi-service compose stack
- [ ] Your ML API is containerized with a health check
