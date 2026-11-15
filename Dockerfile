# =============================================================
# Stage 1: Build Environment & Artifact Generation
# =============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/install/bin:${PATH}" \
    PYTHONPATH="/install/lib/python3.11/site-packages:${PYTHONPATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

COPY data/ /build/data/
COPY training/ /build/training/

# Pre-train and serialize model using the authentic repository dataset
RUN python training/train.py --n-estimators 50 --max-depth 4 --data-path data/raw_data.csv

# =============================================================
# Stage 2: Minimal Production Runtime
# =============================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_VERSION=1.0.0 \
    ENVIRONMENT=production \
    MODEL_PATH=/app/training/model.joblib \
    PATH="/install/bin:${PATH}" \
    PYTHONPATH="/install/lib/python3.11/site-packages:${PYTHONPATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /install /install
COPY --from=builder /build/training/model.joblib /app/training/model.joblib

COPY api/ /app/api/
COPY training/ /app/training/

EXPOSE 8000

# Security: run as non-root user
RUN useradd -m -u 1000 mlopsuser && chown -R mlopsuser:mlopsuser /app
USER mlopsuser

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
