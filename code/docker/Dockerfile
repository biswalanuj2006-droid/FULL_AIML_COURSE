# ============================================================
# Dockerfile — ML inference service
#
# Build & run (from this folder):
#   docker build -t iris-api .
#   docker run -p 8000:8000 iris-api
# Then POST to http://127.0.0.1:8000/predict
#
# Multi-stage optimization ideas (module 41_DOCKER):
#   - pin a tag you control (python:3.11-slim), not latest
#   - copy requirements.txt first, install, THEN copy code
#     so dependency layers are cached across rebuilds
#   - consider a non-root user (USER appuser)
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
