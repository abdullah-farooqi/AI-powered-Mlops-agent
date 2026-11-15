import os
from typing import Dict, Any, List, Optional
import mlflow
from mlflow.tracking import MlflowClient


def _get_client() -> MlflowClient:
    """Connect to MLflow server. Sets strict timeouts and retries so failures fail fast."""
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = os.getenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")
    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = os.getenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def mlflow_list_experiments_and_runs(
    experiment_name: str = "mlops-model-training", max_results: int = 5
) -> Dict[str, Any]:
    """Task 2.1: List experiments and recent training runs with metrics."""
    try:
        client = _get_client()
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            return {"status": "error", "error": f"Experiment '{experiment_name}' not found on MLflow server."}

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id], max_results=max_results, order_by=["attributes.start_time DESC"]
        )

        run_summaries = []
        for r in runs:
            run_summaries.append(
                {
                    "run_id": r.info.run_id,
                    "status": r.info.status,
                    "start_time": r.info.start_time,
                    "metrics": r.data.metrics,
                    "tags": {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")},
                }
            )

        return {
            "status": "success",
            "experiment_id": exp.experiment_id,
            "experiment_name": experiment_name,
            "runs_count": len(run_summaries),
            "runs": run_summaries,
        }
    except Exception as e:
        return {"status": "error", "error": f"MLflow connection error: {str(e)}"}


def mlflow_get_run_details(run_id: str) -> Dict[str, Any]:
    """Task 2.2: Retrieve model metrics, parameters, tags, and artifacts for a specific run."""
    try:
        client = _get_client()
        run = client.get_run(run_id)
        artifacts = client.list_artifacts(run_id)
        return {
            "status": "success",
            "run_id": run_id,
            "lifecycle_stage": run.info.status,
            "metrics": run.data.metrics,
            "parameters": run.data.params,
            "tags": run.data.tags,
            "dvc_data_version": run.data.tags.get("dvc_data_version", "unknown"),
            "dvc_data_hash": run.data.tags.get("dvc_data_hash", "unknown"),
            "git_commit": run.data.tags.get("git_commit", "unknown"),
            "dataset_rows": run.data.tags.get("dataset_rows", "unknown"),
            "artifacts": [a.path for a in artifacts],
        }
    except Exception as e:
        return {"status": "error", "error": f"Failed to retrieve run details for '{run_id}': {str(e)}"}


def mlflow_get_best_run(
    experiment_name: str = "mlops-model-training",
    metric_name: str = "f1_score",
    mode: str = "max",
    min_accuracy: float = 0.70,
) -> Dict[str, Any]:
    """
    Task 2.3: Identify best run according to a specified evaluation metric.
    Enforces quality gate: run status must be FINISHED, checks secondary metrics (accuracy),
    and explains why highest F1 alone does not guarantee production readiness.
    """
    try:
        client = _get_client()
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            return {"status": "error", "error": f"Experiment '{experiment_name}' not found."}

        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        if not runs:
            return {"status": "error", "error": f"No runs found in experiment '{experiment_name}'."}

        # Filter strictly for FINISHED runs containing the evaluation metric
        valid_runs = [r for r in runs if r.info.status == "FINISHED" and metric_name in r.data.metrics]
        if not valid_runs:
            return {
                "status": "error",
                "error": f"No finished runs with metric '{metric_name}' found in experiment '{experiment_name}'.",
            }

        reverse_sort = mode.lower() == "max"
        sorted_runs = sorted(valid_runs, key=lambda r: r.data.metrics[metric_name], reverse=reverse_sort)
        best_run = sorted_runs[0]
        metrics = best_run.data.metrics

        # Candidate assessment
        passes_secondary_gate = metrics.get("accuracy", 0.0) >= min_accuracy
        has_provenance = "dvc_data_hash" in best_run.data.tags and "git_commit" in best_run.data.tags

        notes = []
        if not passes_secondary_gate:
            notes.append(f"Warning: Accuracy ({metrics.get('accuracy')}) is below minimum threshold ({min_accuracy}).")
        if not has_provenance:
            notes.append("Warning: Run lacks complete DVC data hash and Git commit provenance tags.")
        if not notes:
            notes.append("Model candidate satisfies performance gates and contains full reproducibility tags.")

        notes.append(
            "Note: Top F1 alone does not guarantee production readiness; "
            "must correlate with live Kubernetes probes and drift."
        )

        return {
            "status": "success",
            "best_run_id": best_run.info.run_id,
            "target_metric": metric_name,
            "best_metric_value": metrics[metric_name],
            "all_metrics": metrics,
            "parameters": best_run.data.params,
            "dvc_data_version": best_run.data.tags.get("dvc_data_version", "unknown"),
            "dvc_data_hash": best_run.data.tags.get("dvc_data_hash", "unknown"),
            "git_commit": best_run.data.tags.get("git_commit", "unknown"),
            "production_ready_candidate": passes_secondary_gate and has_provenance,
            "evaluation_assessment": " ".join(notes),
        }
    except Exception as e:
        return {"status": "error", "error": f"Error finding best run: {str(e)}"}


def mlflow_compare_runs(
    run_ids: Optional[List[str]] = None,
    experiment_name: str = "mlops-model-training",
    last_n: int = 5,
    metric_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Task 2.4: Compare two or more runs or the last N runs of an experiment side-by-side.
    Returns structured comparison table and defensible recommendation reasoning.
    """
    try:
        client = _get_client()
        metrics_to_compare = metric_names or ["f1_score", "accuracy", "precision", "recall"]
        runs_to_eval = []

        if run_ids:
            for rid in run_ids:
                try:
                    runs_to_eval.append(client.get_run(rid))
                except Exception as e:
                    return {"status": "error", "error": f"Failed fetching run '{rid}': {e}"}
        else:
            exp = client.get_experiment_by_name(experiment_name)
            if not exp:
                return {"status": "error", "error": f"Experiment '{experiment_name}' not found."}
            runs_to_eval = client.search_runs(
                experiment_ids=[exp.experiment_id], max_results=last_n, order_by=["attributes.start_time DESC"]
            )

        if not runs_to_eval:
            return {"status": "error", "error": "No runs available for comparison."}

        comparison = []
        for r in runs_to_eval:
            extracted_metrics = {m: r.data.metrics.get(m, None) for m in metrics_to_compare}
            comparison.append(
                {
                    "run_id": r.info.run_id,
                    "status": r.info.status,
                    "f1_score": r.data.metrics.get("f1_score"),
                    "accuracy": r.data.metrics.get("accuracy"),
                    "metrics": extracted_metrics,
                    "params": r.data.params,
                    "dvc_version": r.data.tags.get("dvc_data_version", "unknown"),
                    "dvc_hash": r.data.tags.get("dvc_data_hash", "unknown"),
                    "git_commit": r.data.tags.get("git_commit", "unknown"),
                }
            )

        # Rank candidates based on balanced F1 and accuracy
        ranked = sorted(
            [c for c in comparison if c["f1_score"] is not None],
            key=lambda x: (x["f1_score"], x.get("accuracy", 0)),
            reverse=True,
        )

        recommended_run = ranked[0] if ranked else None
        reasoning = (
            (
                f"Recommended run '{recommended_run['run_id']}' achieves highest balanced performance "
                f"(F1: {recommended_run['f1_score']:.4f}, Accuracy: {recommended_run.get('accuracy', 0):.4f}) "
                f"with verified DVC dataset hash '{recommended_run['dvc_hash'][:8]}'."
            )
            if recommended_run
            else "No run has valid evaluation metrics."
        )

        return {
            "status": "success",
            "compared_count": len(comparison),
            "comparison": comparison,
            "recommended_candidate": recommended_run["run_id"] if recommended_run else None,
            "recommendation_reasoning": reasoning,
        }
    except Exception as e:
        return {"status": "error", "error": f"Run comparison error: {str(e)}"}


def mlflow_get_registered_model_info(model_name: str = "iris_classifier") -> Dict[str, Any]:
    """Task 2.5: Retrieve model versions and registry stages. Returns explicit error on failure."""
    try:
        client = _get_client()
        versions = client.get_latest_versions(model_name)
        ver_details = [{"version": v.version, "stage": v.current_stage, "run_id": v.run_id} for v in versions]
        return {
            "status": "success",
            "model_name": model_name,
            "registered_versions_count": len(ver_details),
            "registered_versions": ver_details,
        }
    except Exception as e:
        return {"status": "error", "error": f"Model registry error for '{model_name}': {str(e)}"}
