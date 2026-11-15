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

# Generate model artifact during image build if not pre-baked
RUN python -c "
import os
if not os.path.exists('training/model.joblib'):
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    iris = load_iris()
    clf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    clf.fit(iris.data, iris.target)
    os.makedirs('training', exist_ok=True)
    joblib.dump(clf, 'training/model.joblib')
    print('Baked production model artifact into container image.')
"

EXPOSE 8000

RUN useradd -m -u 1000 mlopsuser && chown -R mlopsuser:mlopsuser /app
USER mlopsuser

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
