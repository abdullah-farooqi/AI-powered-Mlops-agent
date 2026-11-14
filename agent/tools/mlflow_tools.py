import os
import urllib.request
import urllib.parse
from typing import Dict, Any, List, Optional
import mlflow
from mlflow.tracking import MlflowClient

def _get_client() -> MlflowClient:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if tracking_uri.startswith("http://") or tracking_uri.startswith("https://"):
        try:
            urllib.request.urlopen(f"{tracking_uri}/health", timeout=1)
        except Exception:
            # Fallback to local file store if server is offline
            abs_path = os.path.abspath("mlruns")
            quoted = urllib.parse.quote(abs_path)
            tracking_uri = f"file://{quoted}"
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)

def mlflow_list_experiments_and_runs(experiment_name: str = "mlops-model-training", max_results: int = 5) -> Dict[str, Any]:
    """Task 2.1: List experiments and recent training runs with metrics."""
    try:
        client = _get_client()
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            return {
                "status": "error",
                "error": f"Experiment '{experiment_name}' not found on MLflow server."
            }
        
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=max_results,
            order_by=["attributes.start_time DESC"]
        )

        run_summaries = []
        for r in runs:
            run_summaries.append({
                "run_id": r.info.run_id,
                "status": r.info.status,
                "start_time": r.info.start_time,
                "metrics": r.data.metrics,
                "tags": {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
            })

        return {
            "status": "success",
            "experiment_id": exp.experiment_id,
            "experiment_name": experiment_name,
            "runs_count": len(run_summaries),
            "runs": run_summaries
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
            "metrics": run.data.metrics,
            "parameters": run.data.params,
            "tags": run.data.tags,
            "dvc_data_version": run.data.tags.get("dvc_data_version", "unknown"),
            "git_commit": run.data.tags.get("git_commit", "unknown"),
            "artifacts": [a.path for a in artifacts]
        }
    except Exception as e:
        return {"status": "error", "error": f"Failed to retrieve run details for '{run_id}': {str(e)}"}

def mlflow_get_best_run(experiment_name: str = "mlops-model-training", metric_name: str = "f1_score", mode: str = "max") -> Dict[str, Any]:
    """Task 2.3: Identify best run according to a specified evaluation metric."""
    try:
        client = _get_client()
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            return {"status": "error", "error": f"Experiment '{experiment_name}' not found."}
        
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        if not runs:
            return {"status": "error", "error": f"No runs found in experiment '{experiment_name}'."}

        valid_runs = [r for r in runs if metric_name in r.data.metrics]
        if not valid_runs:
            return {"status": "error", "error": f"Metric '{metric_name}' not found in any runs."}

        reverse_sort = (mode.lower() == "max")
        sorted_runs = sorted(valid_runs, key=lambda r: r.data.metrics[metric_name], reverse=reverse_sort)
        best_run = sorted_runs[0]

        return {
            "status": "success",
            "best_run_id": best_run.info.run_id,
            "target_metric": metric_name,
            "best_metric_value": best_run.data.metrics[metric_name],
            "all_metrics": best_run.data.metrics,
            "parameters": best_run.data.params,
            "dvc_data_version": best_run.data.tags.get("dvc_data_version", "v1.0"),
            "git_commit": best_run.data.tags.get("git_commit", "unknown"),
            "evaluation_note": "Model candidate selected. Ready for health probe and latency verification."
        }
    except Exception as e:
        return {"status": "error", "error": f"Error finding best run: {str(e)}"}

def mlflow_compare_runs(run_ids: List[str], metric_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Task 2.4: Compare two or more experiments side-by-side."""
    try:
        client = _get_client()
        metrics_to_compare = metric_names or ["f1_score", "accuracy", "precision", "recall"]
        comparison = []

        for rid in run_ids:
            run = client.get_run(rid)
            extracted_metrics = {m: run.data.metrics.get(m, None) for m in metrics_to_compare}
            comparison.append({
                "run_id": rid,
                "metrics": extracted_metrics,
                "dvc_version": run.data.tags.get("dvc_data_version", "n/a")
            })

        return {
            "status": "success",
            "compared_count": len(comparison),
            "comparison": comparison
        }
    except Exception as e:
        return {"status": "error", "error": f"Run comparison error: {str(e)}"}

def mlflow_get_registered_model_info(model_name: str = "iris_classifier") -> Dict[str, Any]:
    """Task 2.5: Retrieve model versions and registry stages."""
    try:
        client = _get_client()
        versions = client.get_latest_versions(model_name)
        ver_details = [{"version": v.version, "stage": v.current_stage, "run_id": v.run_id} for v in versions]
        return {
            "status": "success",
            "model_name": model_name,
            "registered_versions": ver_details
        }
    except Exception as e:
        return {
            "status": "success",
            "model_name": model_name,
            "registered_versions": [],
            "note": f"Model registry empty or unconfigured: {str(e)}"
        }
