import pytest
import os
import tempfile


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Sets environment variables for testing."""
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    os.environ["AIRFLOW_BASE_URL"] = "http://localhost:8080/api/v1"
    os.environ["DATA_PATH"] = "data/raw_data.csv"
    os.environ["ALLOWED_NAMESPACE"] = "mlops-app"


@pytest.fixture
def sample_csv():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("sepal_length,sepal_width,petal_length,petal_width,target\n5.1,3.5,1.4,0.2,0\n")
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)
