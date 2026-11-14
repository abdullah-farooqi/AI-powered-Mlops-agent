FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_VERSION=1.0.0 \
    ENVIRONMENT=production \
    MODEL_PATH=/app/training/model.joblib

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ /app/api/
COPY training/ /app/training/
COPY data/ /app/data/

# Expose FastAPI port
EXPOSE 8000

# Non-root user for production security
RUN useradd -m -u 1000 mlopsuser && chown -R mlopsuser:mlopsuser /app
USER mlopsuser

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
