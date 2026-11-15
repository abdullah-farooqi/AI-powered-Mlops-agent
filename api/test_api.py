from fastapi.testclient import TestClient
from api.main import app


def test_health_probe():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"


def test_version_endpoint():
    with TestClient(app) as client:
        response = client.get("/version")
        assert response.status_code == 200
        data = response.json()
        assert "app_version" in data
        assert "model_name" in data


def test_ready_and_predict_endpoint():
    with TestClient(app) as client:
        ready_resp = client.get("/ready")
        if ready_resp.status_code == 200:
            pred_resp = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
            assert pred_resp.status_code == 200
            data = pred_resp.json()
            assert "prediction" in data
            assert isinstance(data["prediction"], int)
        else:
            # When model artifact is not mounted on disk, returns 503
            assert ready_resp.status_code == 503
