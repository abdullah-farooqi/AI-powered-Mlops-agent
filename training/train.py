import os
import hashlib
import subprocess
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def get_git_commit_hash() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "git_commit_unavailable"

def get_data_hash(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "data_file_not_found"
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def resolve_mlflow_tracking_uri() -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if uri.startswith("http://") or uri.startswith("https://"):
        try:
            # Check if MLflow server is responding
            urllib.request.urlopen(f"{uri}/health", timeout=1)
            return uri
        except Exception:
            abs_path = os.path.abspath("mlruns")
            quoted = urllib.parse.quote(abs_path)
            fallback_uri = f"file://{quoted}"
            print(f"[MLflow Notice] Tracking server at '{uri}' is offline. Logging run locally to: {fallback_uri}")
            return fallback_uri
    return uri

def main():
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops-model-training")
    tracking_uri = resolve_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    data_path = os.getenv("DATA_PATH", "data/raw_data.csv")
    git_commit = get_git_commit_hash()
    data_hash = get_data_hash(data_path)
    dvc_version = os.getenv("DVC_DATA_VERSION", "v1.0")

    print(f"Loading data from {data_path} (MD5: {data_hash})...")
    df = pd.read_csv(data_path)

    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    n_estimators = int(os.getenv("N_ESTIMATORS", "50"))
    max_depth = int(os.getenv("MAX_DEPTH", "4"))

    with mlflow.start_run(run_name="agent-rf-run") as run:
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_test, preds, average="weighted", zero_division=0)

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("data_path", data_path)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        # Log DVC provenance tags for the AI Agent
        mlflow.set_tag("dvc_data_version", dvc_version)
        mlflow.set_tag("dvc_data_hash", data_hash)
        mlflow.set_tag("git_commit", git_commit)
        mlflow.set_tag("pipeline_trigger", "airflow_or_manual")

        # Save model locally and to MLflow
        os.makedirs("training", exist_ok=True)
        joblib.dump(clf, "training/model.joblib")
        mlflow.sklearn.log_model(clf, artifact_path="model")

        print(f"Run completed successfully! ID={run.info.run_id}")
        print(f"Metrics: Accuracy={acc:.4f}, F1={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")
        print(f"DVC Version Tag: {dvc_version}, Git Commit: {git_commit}")

if __name__ == "__main__":
    main()
