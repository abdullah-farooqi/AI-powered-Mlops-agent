import os
import requests
from typing import Dict, Any, Optional

AIRFLOW_BASE_URL = os.getenv("AIRFLOW_BASE_URL", "http://localhost:8080/api/v1")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "admin")
AIRFLOW_JWT_TOKEN = os.getenv("AIRFLOW_JWT_TOKEN", None)

APPROVED_DAGS = {"ml_training_pipeline"}

def _get_auth_headers():
    if AIRFLOW_JWT_TOKEN:
        return {"Authorization": f"Bearer {AIRFLOW_JWT_TOKEN}"}
    return None

def _get_auth():
    if not AIRFLOW_JWT_TOKEN:
        return (AIRFLOW_USER, AIRFLOW_PASSWORD)
    return None

def airflow_list_dags() -> Dict[str, Any]:
    """Task 3.1: List available Airflow DAGs via REST API. Never fabricates results if offline."""
    try:
        url = f"{AIRFLOW_BASE_URL}/dags"
        resp = requests.get(
            url,
            auth=_get_auth(),
            headers=_get_auth_headers(),
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            dags = [
                {
                    "dag_id": d["dag_id"],
                    "is_paused": d.get("is_paused", False),
                    "is_active": d.get("is_active", True)
                }
                for d in data.get("dags", [])
            ]
            return {"status": "success", "dags_count": len(dags), "dags": dags}
        return {"status": "error", "error": f"Airflow API error {resp.status_code}: {resp.text}"}
    except Exception as e:
        return {"status": "error", "error": f"Airflow service unavailable at '{AIRFLOW_BASE_URL}': {str(e)}"}

def airflow_get_dag_run_status(dag_id: str = "ml_training_pipeline", run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Task 3.2: Retrieve latest DAG run status sorted by execution date.
    Extracts run state, start date, end date, and execution metadata.
    """
    try:
        url = f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns"
        resp = requests.get(
            url,
            auth=_get_auth(),
            headers=_get_auth_headers(),
            timeout=5
        )
        if resp.status_code == 200:
            runs = resp.json().get("dag_runs", [])
            if not runs:
                return {
                    "status": "success",
                    "dag_id": dag_id,
                    "state": "no_runs_recorded",
                    "message": "No execution runs recorded for this DAG."
                }

            # Sort descending by start_date or logical_date/execution_date
            sorted_runs = sorted(
                runs,
                key=lambda r: r.get("start_date") or r.get("logical_date") or r.get("execution_date") or "",
                reverse=True
            )
            latest = sorted_runs[0]
            return {
                "status": "success",
                "dag_id": dag_id,
                "run_id": latest.get("dag_run_id"),
                "state": latest.get("state"),
                "start_date": latest.get("start_date"),
                "end_date": latest.get("end_date"),
                "logical_date": latest.get("logical_date") or latest.get("execution_date")
            }
        return {"status": "error", "error": f"Failed fetching DAG runs ({resp.status_code}): {resp.text}"}
    except Exception as e:
        return {"status": "error", "error": f"Airflow API unreachable: {str(e)}"}

def airflow_trigger_dag(
    dag_id: str = "ml_training_pipeline",
    conf: Optional[Dict[str, Any]] = None,
    confirmed: bool = False
) -> Dict[str, Any]:
    """
    Task 3.3: Trigger an approved training DAG.
    Safety Guard: Checks whitelist and requires confirmation before executing state-changing trigger.
    """
    if dag_id not in APPROVED_DAGS:
        return {
            "status": "denied",
            "error": f"Authorization failure: DAG '{dag_id}' is not in approved whitelist {APPROVED_DAGS}."
        }

    if not confirmed:
        return {
            "status": "needs_confirmation",
            "dag_id": dag_id,
            "message": f"Confirmation required to trigger state-changing pipeline DAG '{dag_id}'. Please confirm execution."
        }

    try:
        url = f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns"
        payload = {"conf": conf or {}}
        resp = requests.post(
            url,
            json=payload,
            auth=_get_auth(),
            headers=_get_auth_headers(),
            timeout=5
        )
        if resp.status_code in [200, 201]:
            data = resp.json()
            return {
                "status": "success",
                "dag_id": dag_id,
                "run_id": data.get("dag_run_id"),
                "state": data.get("state", "queued"),
                "message": f"Authorized DAG '{dag_id}' triggered successfully."
            }
        return {"status": "error", "error": f"Airflow returned {resp.status_code}: {resp.text}"}
    except Exception as e:
        return {"status": "error", "error": f"Failed to trigger DAG: {str(e)}"}

def airflow_get_failed_tasks(dag_id: str = "ml_training_pipeline", run_id: Optional[str] = None) -> Dict[str, Any]:
    """Task 3.4: Identify failed task instances with detailed execution try counts."""
    try:
        if not run_id:
            latest_run = airflow_get_dag_run_status(dag_id)
            if latest_run.get("status") != "success":
                return latest_run
            run_id = latest_run.get("run_id")

        if not run_id or run_id == "no_runs_recorded":
            return {"status": "success", "dag_id": dag_id, "failed_tasks": [], "note": "No active runs found."}

        url = f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns/{run_id}/taskInstances"
        resp = requests.get(url, auth=_get_auth(), headers=_get_auth_headers(), timeout=5)
        if resp.status_code == 200:
            tasks = resp.json().get("task_instances", [])
            failed_list = []
            for t in tasks:
                if t.get("state") == "failed":
                    failed_list.append({
                        "task_id": t.get("task_id"),
                        "state": t.get("state"),
                        "try_number": t.get("try_number"),
                        "start_date": t.get("start_date"),
                        "end_date": t.get("end_date")
                    })
            return {
                "status": "success",
                "dag_id": dag_id,
                "run_id": run_id,
                "failed_count": len(failed_list),
                "failed_tasks": failed_list
            }
        return {"status": "error", "error": f"Failed reading task instances: {resp.text}"}
    except Exception as e:
        return {"status": "error", "error": f"Airflow error: {str(e)}"}

def airflow_get_task_logs(dag_id: str, run_id: str, task_id: str, try_number: int = 1) -> Dict[str, Any]:
    """Task 3.5: Retrieve task execution logs for failure diagnostics."""
    try:
        url = f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/logs/{try_number}"
        resp = requests.get(url, auth=_get_auth(), headers=_get_auth_headers(), timeout=5)
        if resp.status_code == 200:
            return {
                "status": "success",
                "dag_id": dag_id,
                "task_id": task_id,
                "try_number": try_number,
                "logs_tail": resp.text[-2000:]
            }
        return {"status": "error", "error": f"Failed fetching logs ({resp.status_code}): {resp.text}"}
    except Exception as e:
        return {"status": "error", "error": f"Could not fetch logs: {str(e)}"}
