# Dockerfile — толық түзетілген нұсқа
# ---- Stage 1: builder ----
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.api.txt .
RUN pip install --no-cache-dir -r requirements.api.txt

# ---- Stage 2: runtime ----
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY api/ ./api/
COPY src/ ./src/
COPY models_store/onnx/ ./models_store/onnx/
COPY data/splits/test.csv ./data/splits/test.csv

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["sh", "-c", "/opt/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]