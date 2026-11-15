import os
import joblib
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional

MODEL_PATH = os.getenv("MODEL_PATH", "training/model.joblib")
MODEL = None
APP_READY = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan context manager for startup and shutdown."""
    global MODEL, APP_READY
    if os.path.exists(MODEL_PATH):
        try:
            MODEL = joblib.load(MODEL_PATH)
            APP_READY = True
            print(f"Loaded trained model successfully from {MODEL_PATH}")
        except Exception as e:
            MODEL = None
            APP_READY = False
            print(f"Failed to load model from {MODEL_PATH}: {e}")
    else:
        MODEL = None
        APP_READY = False
        print(f"Warning: Model file '{MODEL_PATH}' not found. Application will serve 503 until model is available.")
    yield
    MODEL = None
    APP_READY = False

app = FastAPI(
    title="MLOps Model Inference Service",
    version=os.getenv("APP_VERSION", "1.0.0"),
    description="Containerized prediction API with readiness/liveness health probes",
    lifespan=lifespan
)

class PredictionRequest(BaseModel):
    features: List[float] = Field(..., example=[5.1, 3.5, 1.4, 0.2], description="Feature vector for inference")

class PredictionResponse(BaseModel):
    prediction: int
    probabilities: Optional[List[float]] = None
    model_version: str
    app_version: str

@app.get("/health", status_code=status.HTTP_200_OK, tags=["Probes"])
def liveness_probe():
    """Liveness probe: returns 200 if container process is running."""
    return {"status": "alive", "service": "mlops-prediction-api"}

@app.get("/ready", status_code=status.HTTP_200_OK, tags=["Probes"])
def readiness_probe():
    """Readiness probe: returns 200 ONLY if model artifact is actually loaded and ready."""
    if not APP_READY or MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Application initialization in progress or model artifact not loaded"
        )
    return {"status": "ready", "model_loaded": True}

@app.get("/version", tags=["Metadata"])
def get_version():
    """Returns application and model metadata."""
    return {
        "app_version": os.getenv("APP_VERSION", "1.0.0"),
        "model_name": os.getenv("MODEL_NAME", "iris_classifier"),
        "model_version": os.getenv("MODEL_VERSION", "v1.0.0"),
        "environment": os.getenv("ENVIRONMENT", "production")
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(request: PredictionRequest):
    """Generates inference predictions from feature input."""
    if not APP_READY or MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready to serve predictions. Ensure model.joblib is trained and mounted."
        )
    
    features = np.array(request.features).reshape(1, -1)
    
    try:
        pred = int(MODEL.predict(features)[0])
        probs = MODEL.predict_proba(features)[0].tolist() if hasattr(MODEL, "predict_proba") else None
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {str(e)}")

    return PredictionResponse(
        prediction=pred,
        probabilities=probs,
        model_version=os.getenv("MODEL_VERSION", "v1.0.0"),
        app_version=os.getenv("APP_VERSION", "1.0.0")
    )
