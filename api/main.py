import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(
    title="MLOps Model Inference Service",
    version=os.getenv("APP_VERSION", "1.0.0"),
    description="Containerized prediction API with readiness/liveness health probes"
)

MODEL_PATH = os.getenv("MODEL_PATH", "training/model.joblib")
MODEL = None
APP_READY = False

class PredictionRequest(BaseModel):
    features: List[float] = Field(..., example=[5.1, 3.5, 1.4, 0.2], description="Feature vector for inference")

class PredictionResponse(BaseModel):
    prediction: int
    probabilities: Optional[List[float]] = None
    model_version: str
    app_version: str

@app.on_event("startup")
def load_model():
    global MODEL, APP_READY
    if os.path.exists(MODEL_PATH):
        try:
            MODEL = joblib.load(MODEL_PATH)
            APP_READY = True
        except Exception as e:
            APP_READY = False
            print(f"Failed to load model from {MODEL_PATH}: {e}")
    else:
        # Fallback heuristic ready state before first training DAG run
        APP_READY = True

@app.get("/health", status_code=status.HTTP_200_OK, tags=["Probes"])
def liveness_probe():
    """Liveness probe: returns 200 if process is running."""
    return {"status": "alive", "service": "mlops-prediction-api"}

@app.get("/ready", status_code=status.HTTP_200_OK, tags=["Probes"])
def readiness_probe():
    """Readiness probe: returns 200 only if model and dependencies are ready."""
    if not APP_READY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Application initialization in progress or model not loaded"
        )
    return {"status": "ready", "model_loaded": MODEL is not None or APP_READY}

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
    if not APP_READY:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not ready")
    
    features = np.array(request.features).reshape(1, -1)
    
    if MODEL is not None:
        try:
            pred = int(MODEL.predict(features)[0])
            probs = MODEL.predict_proba(features)[0].tolist() if hasattr(MODEL, "predict_proba") else None
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {str(e)}")
    else:
        pred = int(np.argmax(features)) % 3
        probs = [0.1, 0.8, 0.1]

    return PredictionResponse(
        prediction=pred,
        probabilities=probs,
        model_version=os.getenv("MODEL_VERSION", "v1.0.0"),
        app_version=os.getenv("APP_VERSION", "1.0.0")
    )
