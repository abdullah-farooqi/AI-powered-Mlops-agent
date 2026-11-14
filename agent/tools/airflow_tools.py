import os
import requests
from typing import Dict, Any, List

AIRFLOW_BASE_URL = os.getenv("AIRFLOW_BASE_URL", "http://localhost:8080/api/v1")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "admin")

APPROVED_DAGS = {"ml_training_pipeline"}

def _auth():
    return (AIRFLOW_USER, AIRFLOW_PASSWORD)

def airflow_list_dags() -> Dict[str, Any]:
    """Task 3.1: List available Airflow DAGs."""
    try:
        resp = requests.get(f"{AIRFLOW_BASE_URL}/dags", auth=_auth(), timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            dags = [{"dag_id": d["dag_id"], "is_paused": d["is_paused"]} for d in data.get("dags", [])]
            return {"status": "success", "dags": dags}
        return {"status": "error", "error": f"Airflow returned status {resp.status_code}: {resp.text}"}
    except Exception as e:
        return {
            "status": "error",
            "error": f"Airflow service unavailable: {str(e)}",
            "simulated_available_dags": ["ml_training_pipeline"]
        }

def airflow_get_dag_run_status(dag_id: str = "ml_training_pipeline", run_id: str = None) -> Dict[str, Any]:
    """Task 3.2: Retrieve latest DAG run status."""
    try:
        url = f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns"
        resp = requests.get(url, auth=_auth(), timeout=5)
        if resp.status_code == 200:
            runs = resp.json().get("dag_runs", [])
            if not runs:
                return {"status": "success", "dag_id": dag_id, "latest_run": None, "state": "no_runs_yet"}
            latest = runs[-1]
            return {
                "status": "success",
                "dag_id": dag_id,
                "run_id": latest["dag_run_id"],
                "state": latest["state"],
                "execution_date": latest["execution_date"]
            }
        return {"status": "error", "error": resp.text}
    except Exception as e:
        return {"status": "error", "error": f"Airflow unreachable: {str(e)}"}

def airflow_trigger_dag(dag_id: str = "ml_training_pipeline", conf: Dict[str, Any] = None) -> Dict[str, Any]:
    """Task 3.3: Trigger an approved training DAG (rejects unauthorized DAGs)."""
    if dag_id not in APPROVED_DAGS:
        return {
            "status": "denied",
            "error": f"Authorization failure: DAG '{dag_id}' is not in the approved training DAG whitelist {APPROVED_DAGS}."
        }
    try:
        url = f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns"
        payload = {"conf": conf or {}}
        resp = requests.post(url, json=payload, auth=_auth(), timeout=5)
        if resp.status_code in [200, 201]:
            data = resp.json()
            return {
                "status": "success",
                "dag_id": dag_id,
                "run_id": data.get("dag_run_id"),
                "state": data.get("state"),
                "message": f"Successfully triggered training pipeline DAG '{dag_id}'."
            }
        return {"status": "error", "error": resp.text}
    except Exception as e:
        return {"status": "error", "error": f"Failed to trigger DAG: {str(e)}"}

def airflow_get_failed_tasks(dag_id: str = "ml_training_pipeline", run_id: str = None) -> Dict[str, Any]:
    """Task 3.4: Identify failed tasks for diagnostic root-cause analysis."""
    try:
        if not run_id:
            status_res = airflow_get_dag_run_status(dag_id)
            run_id = status_res.get("run_id")
        if not run_id:
            return {"status": "success", "failed_tasks": [], "note": "No prior runs found."}

        url = f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns/{run_id}/taskInstances"
        resp = requests.get(url, auth=_auth(), timeout=5)
        if resp.status_code == 200:
            tasks = resp.json().get("task_instances", [])
            failed = [t["task_id"] for t in tasks if t.get("state") == "failed"]
            return {
                "status": "success",
                "dag_id": dag_id,
                "run_id": run_id,
                "failed_tasks": failed,
                "failed_count": len(failed)
            }
        return {"status": "error", "error": resp.text}
    except Exception as e:
        return {"status": "error", "error": f"Airflow error: {str(e)}"}

def airflow_get_task_logs(dag_id: str, run_id: str, task_id: str) -> Dict[str, Any]:
    """Task 3.5: Retrieve task-execution information and failure diagnostics."""
    try:
        url = f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/logs/1"
        resp = requests.get(url, auth=_auth(), timeout=5)
        if resp.status_code == 200:
            return {
                "status": "success",
                "dag_id": dag_id,
                "task_id": task_id,
                "logs": resp.text[-2000:]
            }
        return {"status": "error", "error": resp.text}
    except Exception as e:
        return {"status": "error", "error": f"Could not fetch logs: {str(e)}"}
