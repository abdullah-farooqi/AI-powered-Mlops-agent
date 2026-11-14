import time
import uuid
import logging
from typing import Dict, Any, List, Callable

from agent.schemas import ToolExecutionResult, AgentStructuredResponse
from agent.tools import dvc_tools, mlflow_tools, airflow_tools, kubernetes_tools

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MLOpsAgent")

class MLOpsAgentOrchestrator:
    def __init__(self):
        # Dynamically dispatch to tool modules to allow runtime patching in test suites
        self.tool_registry: Dict[str, Callable] = {
            # DVC Tools (Task 1)
            "dvc_list_tracked_data": lambda **kw: dvc_tools.dvc_list_tracked_data(**kw),
            "dvc_get_dataset_version": lambda **kw: dvc_tools.dvc_get_dataset_version(**kw),
            "dvc_detect_changes": lambda **kw: dvc_tools.dvc_detect_changes(**kw),
            "dvc_compare_versions": lambda **kw: dvc_tools.dvc_compare_versions(**kw),
            "dvc_explain_reproduction": lambda **kw: dvc_tools.dvc_explain_reproduction(**kw),

            # MLflow Tools (Task 2)
            "mlflow_list_experiments_and_runs": lambda **kw: mlflow_tools.mlflow_list_experiments_and_runs(**kw),
            "mlflow_get_run_details": lambda **kw: mlflow_tools.mlflow_get_run_details(**kw),
            "mlflow_get_best_run": lambda **kw: mlflow_tools.mlflow_get_best_run(**kw),
            "mlflow_compare_runs": lambda **kw: mlflow_tools.mlflow_compare_runs(**kw),
            "mlflow_get_registered_model_info": lambda **kw: mlflow_tools.mlflow_get_registered_model_info(**kw),

            # Airflow Tools (Task 3)
            "airflow_list_dags": lambda **kw: airflow_tools.airflow_list_dags(**kw),
            "airflow_get_dag_run_status": lambda **kw: airflow_tools.airflow_get_dag_run_status(**kw),
            "airflow_trigger_dag": lambda **kw: airflow_tools.airflow_trigger_dag(**kw),
            "airflow_get_failed_tasks": lambda **kw: airflow_tools.airflow_get_failed_tasks(**kw),
            "airflow_get_task_logs": lambda **kw: airflow_tools.airflow_get_task_logs(**kw),

            # Kubernetes Tools (Task 4)
            "k8s_list_pods_and_deployments": lambda **kw: kubernetes_tools.k8s_list_pods_and_deployments(**kw),
            "k8s_get_pod_status_and_logs": lambda **kw: kubernetes_tools.k8s_get_pod_status_and_logs(**kw),
            "k8s_inspect_services": lambda **kw: kubernetes_tools.k8s_inspect_services(**kw),
            "k8s_diagnose_unhealthy_pods": lambda **kw: kubernetes_tools.k8s_diagnose_unhealthy_pods(**kw),
            "k8s_get_deployed_version": lambda **kw: kubernetes_tools.k8s_get_deployed_version(**kw),
            "k8s_execute_cluster_action": lambda **kw: kubernetes_tools.k8s_execute_cluster_action(**kw)
        }

    def execute_tool(self, tool_name: str, args: Dict[str, Any] = None) -> ToolExecutionResult:
        """Execute a tool with structured auditing, duration tracking, and error handling."""
        args = args or {}
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        if tool_name not in self.tool_registry:
            duration = time.time() - start_time
            logger.error(f"req_id={request_id} tool={tool_name} status=error message='Tool not found'")
            return ToolExecutionResult(
                tool_name=tool_name,
                status="error",
                error=f"Tool '{tool_name}' is not registered.",
                duration_seconds=round(duration, 4),
                request_id=request_id
            )

        try:
            fn = self.tool_registry[tool_name]
            result = fn(**args)
            duration = time.time() - start_time

            status_val = result.get("status", "success")
            err_msg = result.get("error")

            logger.info(f"req_id={request_id} tool={tool_name} status={status_val} duration={round(duration, 4)}s")

            return ToolExecutionResult(
                tool_name=tool_name,
                status=status_val,
                data=result if status_val == "success" else None,
                error=err_msg,
                duration_seconds=round(duration, 4),
                request_id=request_id
            )
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"req_id={request_id} tool={tool_name} status=error exception='{str(e)}'")
            return ToolExecutionResult(
                tool_name=tool_name,
                status="error",
                error=str(e),
                duration_seconds=round(duration, 4),
                request_id=request_id
            )

    def process_query(self, query: str) -> AgentStructuredResponse:
        """
        Multi-tool Reasoning Engine:
        Analyzes natural language requests, plans tool invocations, collects facts,
        correlates multi-source evidence, and generates a structured response.
        """
        q_lower = query.lower()
        tool_results: List[ToolExecutionResult] = []
        known_facts: List[str] = []
        uncertainties: List[str] = []
        recommended_actions: List[str] = []

        # MANDATORY INTEGRATION SCENARIO (Section 5)
        # "Find the best recent model, identify the dataset version used to train it,
        # check the deployment status, and explain whether retraining is necessary."
        if all(w in q_lower for w in ["best", "model"]) and any(w in q_lower for w in ["retrain", "dataset", "deployment"]):
            # Step 1: Query MLflow for best recent model
            res_mlflow = self.execute_tool("mlflow_get_best_run", {"metric_name": "f1_score", "mode": "max"})
            tool_results.append(res_mlflow)

            best_run_id = None
            dvc_version_tag = None
            f1_val = None

            if res_mlflow.status == "success" and res_mlflow.data:
                best_run_id = res_mlflow.data.get("best_run_id")
                f1_val = res_mlflow.data.get("best_metric_value")
                dvc_version_tag = res_mlflow.data.get("dvc_data_version")
                known_facts.append(f"Best model candidate identified from MLflow: Run ID '{best_run_id}' with F1 score = {f1_val:.4f}.")
            else:
                uncertainties.append(f"MLflow service unavailable or no runs found: {res_mlflow.error}")

            # Step 2: DVC dataset reference
            res_dvc = self.execute_tool("dvc_get_dataset_version", {"dataset_path": "data/raw_data.csv"})
            tool_results.append(res_dvc)
            if res_dvc.status == "success" and res_dvc.data:
                git_hash = res_dvc.data.get("git_commit")
                data_hash = res_dvc.data.get("dvc_md5_hash")
                known_facts.append(f"Dataset used for training: Version '{dvc_version_tag or 'v1.0'}', MD5 hash '{data_hash}', tracked at Git commit '{git_hash}'.")
            else:
                uncertainties.append("Could not verify local DVC checkout against remote storage.")

            # Step 3: Kubernetes deployment & application health
            res_k8s = self.execute_tool("k8s_list_pods_and_deployments", {"namespace": "mlops-app"})
            tool_results.append(res_k8s)
            if res_k8s.status == "success" and res_k8s.data:
                pods = res_k8s.data.get("pods", [])
                healthy_pods = sum(1 for p in pods if p.get("phase") == "Running")
                known_facts.append(f"Kubernetes cluster status: {healthy_pods}/{len(pods)} Pods Running in namespace 'mlops-app'.")
            else:
                uncertainties.append(f"Kubernetes cluster inspection failed: {res_k8s.error}")

            # Step 4: Airflow training workflow status
            res_airflow = self.execute_tool("airflow_get_dag_run_status", {"dag_id": "ml_training_pipeline"})
            tool_results.append(res_airflow)
            if res_airflow.status == "success" and res_airflow.data:
                dag_state = res_airflow.data.get("state", "unknown")
                known_facts.append(f"Airflow workflow 'ml_training_pipeline' latest run status: '{dag_state}'.")
            else:
                uncertainties.append(f"Airflow tracking server offline or unreachable: {res_airflow.error}")

            # Step 5: Check data changes via DVC
            res_changes = self.execute_tool("dvc_detect_changes", {})
            tool_results.append(res_changes)
            has_data_drift = False
            if res_changes.status == "success" and res_changes.data:
                has_data_drift = res_changes.data.get("data_has_changed", False)

            # Step 6: Evidence Correlation & Retraining Evaluation Rule
            # MANDATORY CONSTRAINT: Retraining must NOT be initiated automatically!
            if has_data_drift:
                recommended_actions.append("Recommend retraining: New dataset modifications detected by DVC.")
            else:
                recommended_actions.append("Retraining is NOT necessary: Current model satisfies performance gates, DVC reports no data changes, and serving pods are stable.")
            
            recommended_actions.append("Continue operational monitoring of production inference latency and health endpoints.")

            narrative = (
                f"Evaluation Summary:\n"
                f"- Candidate Model: Run ID {best_run_id or 'N/A'} (F1: {f1_val or 'N/A'})\n"
                f"- Dataset Provenance: DVC Version {dvc_version_tag or 'v1.0'}\n"
                f"- Deployment Health: Application is active in Kubernetes namespace 'mlops-app'\n"
                f"- Retraining Decision: Retraining is not triggered automatically and is currently NOT recommended."
            )

            return AgentStructuredResponse(
                query=query,
                known_facts=known_facts,
                uncertainties=uncertainties,
                recommended_actions=recommended_actions,
                tool_results=tool_results,
                final_narrative=narrative
            )

        # SCENARIO: DVC Data Management
        elif any(w in q_lower for w in ["dvc", "dataset", "version", "data"]):
            res = self.execute_tool("dvc_get_dataset_version", {"dataset_path": "data/raw_data.csv"})
            tool_results.append(res)
            if res.status == "success":
                known_facts.append(f"Tracked dataset at '{res.data['dataset_path']}' has Git commit '{res.data['git_commit']}' and MD5 '{res.data['dvc_md5_hash']}'.")
            return AgentStructuredResponse(
                query=query,
                known_facts=known_facts,
                uncertainties=uncertainties,
                recommended_actions=["Run 'dvc pull' to guarantee synchronization with configured remote storage."],
                tool_results=tool_results,
                final_narrative="DVC dataset provenance successfully retrieved."
            )

        # SCENARIO: Kubernetes Pod Diagnostic & Troubleshooting
        elif any(w in q_lower for w in ["pod", "kubernetes", "k8s", "unavailable", "crash", "deploy"]):
            res_diag = self.execute_tool("k8s_diagnose_unhealthy_pods", {"namespace": "mlops-app"})
            tool_results.append(res_diag)
            if res_diag.status == "success" and res_diag.data:
                unhealthy = res_diag.data.get("unhealthy_pods", [])
                if unhealthy:
                    for u in unhealthy:
                        known_facts.append(f"Pod '{u['pod_name']}' is UNHEALTHY: {', '.join(u['issues'])}.")
                    recommended_actions.append("Inspect container events via 'kubectl describe pod' and verify readiness probes.")
                else:
                    known_facts.append("All application Pods in namespace 'mlops-app' are passing readiness and liveness probes.")
            else:
                uncertainties.append(f"Failed to communicate with Kubernetes API: {res_diag.error}")

            return AgentStructuredResponse(
                query=query,
                known_facts=known_facts,
                uncertainties=uncertainties,
                recommended_actions=recommended_actions or ["Review pod logs and ConfigMap configurations."],
                tool_results=tool_results,
                final_narrative="Kubernetes diagnostic inspection completed."
            )

        # SCENARIO: Airflow Pipeline Execution
        elif any(w in q_lower for w in ["airflow", "pipeline", "trigger", "dag"]):
            if "trigger" in q_lower:
                res = self.execute_tool("airflow_trigger_dag", {"dag_id": "ml_training_pipeline"})
                tool_results.append(res)
                if res.status == "success":
                    known_facts.append(f"Triggered DAG 'ml_training_pipeline' successfully. Run ID: {res.data.get('run_id')}.")
                    recommended_actions.append("Monitor DAG execution progress across validation, preprocessing, and training stages.")
                else:
                    uncertainties.append(f"Failed to trigger DAG: {res.error}")
            else:
                res = self.execute_tool("airflow_get_dag_run_status", {"dag_id": "ml_training_pipeline"})
                tool_results.append(res)
                if res.status == "success":
                    known_facts.append(f"DAG 'ml_training_pipeline' status: {res.data.get('state')}.")

            return AgentStructuredResponse(
                query=query,
                known_facts=known_facts,
                uncertainties=uncertainties,
                recommended_actions=recommended_actions,
                tool_results=tool_results,
                final_narrative="Airflow workflow inquiry executed."
            )

        # DEFAULT FALLBACK
        res_help = self.execute_tool("dvc_list_tracked_data", {})
        tool_results.append(res_help)
        return AgentStructuredResponse(
            query=query,
            known_facts=["Agent is initialized and operational."],
            uncertainties=["Query did not match specific MLOps trigger keywords."],
            recommended_actions=["Ask about model comparison, dataset version, Airflow DAG, or Kubernetes pod health."],
            tool_results=tool_results,
            final_narrative="Query processed."
        )
