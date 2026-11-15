import os
import hashlib
import subprocess
import argparse
import urllib.request
import urllib.parse
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def get_git_provenance(dvc_file_path: str = "data/raw_data.csv.dvc") -> dict:
    """Retrieve authentic Git commit and tag provenance associated with the tracked dataset."""
    commit_hash = "unknown_commit"
    tag_name = "v1.0"
    try:
        # Get latest commit that touched the DVC pointer file
        res = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", dvc_file_path], capture_output=True, text=True, check=False
        )
        if res.returncode == 0 and res.stdout.strip():
            commit_hash = res.stdout.strip()
        else:
            res_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
            if res_head.returncode == 0:
                commit_hash = res_head.stdout.strip()

        # Get git tag (e.g. data-v1, data-v2)
        res_tag = subprocess.run(["git", "describe", "--tags", "--always"], capture_output=True, text=True, check=False)
        if res_tag.returncode == 0 and res_tag.stdout.strip():
            tag_name = res_tag.stdout.strip()
    except Exception as e:
        print(f"Git inspection warning: {e}")

    return {"git_commit": commit_hash, "git_tag": tag_name}


def get_dvc_file_md5(dvc_file_path: str = "data/raw_data.csv.dvc") -> str:
    """Extract authentic MD5 checksum recorded inside the .dvc pointer file."""
    if os.path.exists(dvc_file_path):
        try:
            with open(dvc_file_path, "r") as f:
                for line in f:
                    if "md5:" in line:
                        return line.split("md5:")[-1].strip()
        except Exception:
            pass
    return "untracked_in_dvc"


def compute_file_md5(filepath: str) -> str:
    """Calculate actual md5 hash of the data file on disk."""
    if not os.path.exists(filepath):
        return "file_not_found"
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_mlflow_tracking_uri() -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if uri.startswith("http://") or uri.startswith("https://"):
        try:
            urllib.request.urlopen(f"{uri}/health", timeout=1)
            return uri
        except Exception:
            # Fallback to local mlruns with properly escaped path
            abs_path = os.path.abspath("mlruns")
            quoted = urllib.parse.quote(abs_path)
            fallback_uri = f"file://{quoted}"
            print(f"[MLflow Notice] Tracking server at '{uri}' is offline. Logging run locally to: {fallback_uri}")
            return fallback_uri
    return uri


def train_and_log(
    n_estimators: int = 50,
    max_depth: int = 4,
    data_path: str = "data/raw_data.csv",
    run_name: str = "rf-classifier-run",
    register_model: bool = True,
):
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops-model-training")
    tracking_uri = resolve_mlflow_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    # Authentic Provenance Collection
    dvc_pointer = f"{data_path}.dvc"
    provenance = get_git_provenance(dvc_pointer)
    recorded_dvc_md5 = get_dvc_file_md5(dvc_pointer)
    actual_data_md5 = compute_file_md5(data_path)

    df = pd.read_csv(data_path)
    row_count = len(df)
    print(f"Loaded dataset from {data_path}: {row_count} rows (MD5: {actual_data_md5})")

    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    with mlflow.start_run(run_name=run_name) as run:
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, average="weighted", zero_division=0))
        prec = float(precision_score(y_test, preds, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, preds, average="weighted", zero_division=0))

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("row_count", row_count)

        # Log evaluation metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        # Authentic DVC & Git Provenance Tags for AI Agent
        dvc_tag = os.getenv("DVC_DATA_VERSION", provenance["git_tag"])
        mlflow.set_tag("dvc_data_version", dvc_tag)
        mlflow.set_tag("dvc_data_hash", recorded_dvc_md5)
        mlflow.set_tag("actual_data_hash", actual_data_md5)
        mlflow.set_tag("git_commit", provenance["git_commit"])
        mlflow.set_tag("dataset_rows", str(row_count))
        mlflow.set_tag("pipeline_trigger", os.getenv("PIPELINE_TRIGGER", "manual_or_airflow"))

        # Save model locally
        os.makedirs("training", exist_ok=True)
        joblib.dump(clf, "training/model.joblib")

        # Register to MLflow Model Registry when requested and supported
        model_reg_name = "iris_classifier" if (register_model and not tracking_uri.startswith("file://")) else None
        try:
            mlflow.sklearn.log_model(sk_model=clf, artifact_path="model", registered_model_name=model_reg_name)
        except Exception as e:
            print(f"Warning: MLflow model logging/registration fallback: {e}")

        print(f"Run completed: ID={run.info.run_id}")
        print(f"Metrics: Accuracy={acc:.4f}, F1={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")
        print(
            f"Provenance: DVC Version={dvc_tag}, Git Commit={provenance['git_commit'][:8]}, DVC MD5={recorded_dvc_md5}"
        )
        return run.info.run_id


def main():
    parser = argparse.ArgumentParser(description="MLOps Training & Provenance Pipeline")
    parser.add_argument("--n-estimators", type=int, default=50, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=4, help="Max tree depth")
    parser.add_argument("--data-path", type=str, default="data/raw_data.csv", help="Path to CSV dataset")
    parser.add_argument("--run-name", type=str, default="agent-rf-run", help="MLflow run name")
    parser.add_argument("--register", action="store_true", default=True, help="Register model in MLflow Model Registry")
    args = parser.parse_args()

    train_and_log(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        data_path=args.data_path,
        run_name=args.run_name,
        register_model=args.register,
    )


if __name__ == "__main__":
    main()
