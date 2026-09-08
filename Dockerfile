# --- Stage 1: build the React SPA ---
FROM node:20-slim AS frontend
WORKDIR /fe
COPY platform/frontend/package.json platform/frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY platform/frontend ./
RUN npm run build

# --- Stage 2: python runtime serving API + SPA + course assets ---
FROM python:3.12-slim

WORKDIR /app

COPY platform/backend/requirements.txt platform/backend/requirements.txt
RUN pip install --no-cache-dir -r platform/backend/requirements.txt

COPY platform/backend platform/backend
COPY --from=frontend /fe/dist platform/frontend/dist
COPY Images Images
COPY Videos Videos
COPY diagrams diagrams
COPY 29_RAG 29_RAG

ENV ASSETS_DIR=/app \
    FRONTEND_DIST=/app/platform/frontend/dist \
    DATABASE_URL=sqlite:////data/platform.db \
    JWT_SECRET=change-me-in-production

VOLUME ["/data"]
EXPOSE 8000

CMD ["sh", "-c", "cd platform/backend && python seed_run.py && python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
