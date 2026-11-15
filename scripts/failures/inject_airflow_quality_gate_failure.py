#!/usr/bin/env python3
"""
Failure Scenario 2: Inject Airflow Pipeline Quality Gate Rejection.
Trains an intentionally degraded model (max_depth=1, n_estimators=1 on noisy targets)
resulting in F1 < 0.75, causing the Airflow Quality Gate stage to reject and fail.
"""

import os
import pandas as pd
import numpy as np
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def inject_quality_gate_failure():
    print("[INJECTION] Generating degraded training run to trigger Quality Gate rejection...")
    df = pd.read_csv("data/raw_data.csv")
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    # Corrupt targets with random noise to force low performance
    np.random.seed(42)
    y_corrupted = np.random.choice([0, 1, 2], size=len(df))

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops-model-training"))

    with mlflow.start_run(run_name="degraded-quality-gate-test-run") as run:
        clf = RandomForestClassifier(n_estimators=1, max_depth=1, random_state=42)
        clf.fit(X, y_corrupted)
        preds = clf.predict(X)

        acc = float(accuracy_score(y_corrupted, preds))
        f1 = float(f1_score(y_corrupted, preds, average="weighted", zero_division=0))

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.set_tag("pipeline_trigger", "quality_gate_failure_test")

        print(f"[INJECTION] Degraded Run Logged: Run ID={run.info.run_id}, F1={f1:.4f}, Accuracy={acc:.4f}")
        print(f"[INJECTION] F1 ({f1:.4f}) is strictly below threshold 0.75 -> Triggers Quality Gate failure.")
        return run.info.run_id


if __name__ == "__main__":
    inject_quality_gate_failure()
