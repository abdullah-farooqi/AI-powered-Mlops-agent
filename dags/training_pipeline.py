from datetime import datetime, timedelta
import os
import sys
import logging
from airflow import DAG

# Ensure project root is in sys.path
for p in ["/opt/airflow", "."]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Compatible with both Airflow 2 and Airflow 3 operator paths
try:
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:
    from airflow.operators.python import PythonOperator

logger = logging.getLogger("airflow.task")

default_args = {
    "owner": "mlops_agent",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(seconds=10),
}


def validate_data(**kwargs):
    """Stage 1: Validate dataset schema, non-emptiness, and format."""
    data_path = os.getenv("DATA_PATH", "data/raw_data.csv")
    logger.info(f"Validating dataset at: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Validation failure: Dataset '{data_path}' not found.")

    with open(data_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) <= 1:
        raise ValueError("Validation failure: Dataset is empty or contains header only.")

    header = lines[0].split(",")
    expected_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
    if header != expected_cols:
        raise ValueError(f"Validation failure: Schema mismatch. Expected {expected_cols}, got {header}")

    logger.info(f"Data validation passed: {len(lines) - 1} records verified.")
    return {"status": "validated", "record_count": len(lines) - 1}


def preprocess_data(**kwargs):
    """Stage 2: Feature verification and sanitization."""
    logger.info("Executing preprocessing: Checking feature distributions and missing values.")
    data_path = os.getenv("DATA_PATH", "data/raw_data.csv")
    import pandas as pd

    df = pd.read_csv(data_path)
    if df.isnull().sum().sum() > 0:
        raise ValueError("Preprocessing failure: Missing values detected.")
    logger.info(f"Preprocessing completed. Shape: {df.shape}")
    return {"status": "preprocessed", "shape": list(df.shape)}


def train_model(**kwargs):
    """Stage 3: Execute model training and push MLflow run_id via XCom."""
    from training.train import train_and_log

    logger.info("Executing model training with provenance logging...")
    run_id = train_and_log(
        n_estimators=50,
        max_depth=4,
        data_path="data/raw_data.csv",
        run_name="airflow-automated-run",
        register_model=True,
    )
    logger.info(f"Training completed successfully. Run ID: {run_id}")
    ti = kwargs.get("ti")
    if ti:
        ti.xcom_push(key="mlflow_run_id", value=run_id)
    return run_id


def evaluate_model(**kwargs):
    """Stage 4: Evaluate run metrics against operational thresholds."""
    ti = kwargs.get("ti")
    run_id = ti.xcom_pull(key="mlflow_run_id", task_ids="model_training") if ti else None
    if not run_id:
        raise ValueError("Model evaluation failure: No MLflow run ID found from upstream model training stage.")

    logger.info(f"Evaluating model metrics for MLflow Run ID: {run_id}")

    from mlflow.tracking import MlflowClient

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        run = client.get_run(run_id)
        metrics = run.data.metrics
    except Exception as e:
        raise RuntimeError(f"Failed retrieving MLflow metrics for run '{run_id}': {e}")

    f1 = float(metrics.get("f1_score", 0.0))
    acc = float(metrics.get("accuracy", 0.0))

    if not metrics:
        raise ValueError(f"Run '{run_id}' contains no logged metrics.")

    logger.info(f"Retrieved authentic metrics: F1={f1:.4f}, Accuracy={acc:.4f}")
    if ti:
        ti.xcom_push(key="eval_f1_score", value=f1)
        ti.xcom_push(key="eval_accuracy", value=acc)
    return {"f1_score": f1, "accuracy": acc}


def quality_gate(**kwargs):
    """
    Stage 5: Quality Gate.
    Enforces minimum F1 threshold (0.75). A failure stops execution and blocks promotion!
    """
    ti = kwargs.get("ti")
    if not ti:
        raise ValueError("Quality Gate failure: Task instance context missing.")

    f1 = ti.xcom_pull(key="eval_f1_score", task_ids="model_evaluation")
    if f1 is None:
        raise ValueError("Quality Gate failure: No F1 score received from upstream model evaluation stage.")

    min_threshold = 0.75
    logger.info(f"Enforcing Quality Gate: F1={f1:.4f} vs threshold={min_threshold}")
    if f1 < min_threshold:
        raise ValueError(
            f"Quality Gate Failed! Model F1 score ({f1:.4f}) is below minimum threshold ({min_threshold}). "
            "Downstream model promotion is strictly blocked."
        )

    logger.info(f"Quality Gate PASSED: Model F1 ({f1:.4f}) >= {min_threshold}")
    return {"quality_gate": "passed", "promoted": True}


def promote_model(**kwargs):
    """Stage 6: Model Promotion. Runs ONLY after quality gate passes!"""
    ti = kwargs.get("ti")
    if not ti:
        raise ValueError("Promotion failure: Task instance context missing.")

    run_id = ti.xcom_pull(key="mlflow_run_id", task_ids="model_training")
    if not run_id:
        raise ValueError("Promotion failure: No run_id available from model training stage.")

    logger.info(f"Promoting model from Run ID '{run_id}' in MLflow Model Registry.")

    from mlflow.tracking import MlflowClient

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    client = MlflowClient(tracking_uri=tracking_uri)
    model_name = "iris_classifier"

    try:
        client.set_registered_model_tag(model_name, "promoted_run_id", str(run_id))
        client.set_registered_model_tag(model_name, "deployment_stage", "Staging")
        client.set_registered_model_tag(model_name, "promoted_at", datetime.utcnow().isoformat())
        logger.info(f"Model '{model_name}' successfully tagged for Staging promotion.")
    except Exception as e:
        logger.warning(f"Notice: Model registry promotion tag update: {e}")

    return {"promoted_to_staging": True, "run_id": run_id}


# Modern Airflow 3 compatible DAG definition (schedule=None replaces deprecated schedule_interval)
with DAG(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    description="Production MLOps training DAG: validation -> preprocessing -> training -> evaluation -> quality_gate",
    schedule=None,
    catchup=False,
    tags=["mlops", "agent", "training", "airflow3"],
) as dag:

    t1_validation = PythonOperator(
        task_id="data_validation",
        python_callable=validate_data,
    )

    t2_preprocessing = PythonOperator(
        task_id="data_preprocessing",
        python_callable=preprocess_data,
    )

    t3_training = PythonOperator(
        task_id="model_training",
        python_callable=train_model,
    )

    t4_evaluation = PythonOperator(
        task_id="model_evaluation",
        python_callable=evaluate_model,
    )

    t5_quality_gate = PythonOperator(
        task_id="quality_gate",
        python_callable=quality_gate,
    )

    t6_promotion = PythonOperator(
        task_id="model_promotion",
        python_callable=promote_model,
    )

    # Pipeline stages dependency graph
    t1_validation >> t2_preprocessing >> t3_training >> t4_evaluation >> t5_quality_gate >> t6_promotion
