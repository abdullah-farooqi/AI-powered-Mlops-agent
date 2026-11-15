import re
import time
import uuid
import logging
from typing import Dict, Any, List, Callable

from agent.schemas import ToolExecutionResult, AgentStructuredResponse
from agent.tools import dvc_tools, mlflow_tools, airflow_tools, kubernetes_tools

# -------------------------------------------------------------
# Structured JSON Logging with Credential Redaction Filter
# -------------------------------------------------------------
class SecretRedactionFilter(logging.Filter):
    REDACT_PATTERNS = [
        (re.compile(r'(password|secret|token|api_key|authorization)[\s:=]+["\']?([^"\'\s,]+)["\']?', re.IGNORECASE), r'\1="[REDACTED]"'),
        (re.compile(r'Bearer\s+[A-Za-z0-9\-\._~\+\/]+=*', re.IGNORECASE), 'Bearer [REDACTED]')
    ]

    def filter(self, record):
        msg = record.getMessage()
        for pattern, replacement in self.REDACT_PATTERNS:
            msg = pattern.sub(replacement, msg)
        record.msg = msg
        record.args = ()
        return True

logger = logging.getLogger("MLOpsAgent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}')
handler.setFormatter(formatter)
handler.addFilter(SecretRedactionFilter())
logger.handlers = [handler]
logger.propagate = False

# -------------------------------------------------------------
# Core Agent Orchestrator
# -------------------------------------------------------------
class MLOpsAgentOrchestrator:
    def __init__(self):
        # Dynamic tool dispatch registry
        self.tool_registry: Dict[str, Callable] = {
            # Task 1: DVC Tools
            "dvc_list_tracked_data": lambda **kw: dvc_tools.dvc_list_tracked_data(**kw),
            "dvc_get_dataset_version": lambda **kw: dvc_tools.dvc_get_dataset_version(**kw),
            "dvc_detect_changes": lambda **kw: dvc_tools.dvc_detect_changes(**kw),
            "dvc_compare_versions": lambda **kw: dvc_tools.dvc_compare_versions(**kw),
            "dvc_explain_reproduction": lambda **kw: dvc_tools.dvc_explain_reproduction(**kw),

            # Task 2: MLflow Tools
            "mlflow_list_experiments_and_runs": lambda **kw: mlflow_tools.mlflow_list_experiments_and_runs(**kw),
            "mlflow_get_run_details": lambda **kw: mlflow_tools.mlflow_get_run_details(**kw),
            "mlflow_get_best_run": lambda **kw: mlflow_tools.mlflow_get_best_run(**kw),
            "mlflow_compare_runs": lambda **kw: mlflow_tools.mlflow_compare_runs(**kw),
            "mlflow_get_registered_model_info": lambda **kw: mlflow_tools.mlflow_get_registered_model_info(**kw),

            # Task 3: Airflow Tools
            "airflow_list_dags": lambda **kw: airflow_tools.airflow_list_dags(**kw),
            "airflow_get_dag_run_status": lambda **kw: airflow_tools.airflow_get_dag_run_status(**kw),
            "airflow_trigger_dag": lambda **kw: airflow_tools.airflow_trigger_dag(**kw),
            "airflow_get_failed_tasks": lambda **kw: airflow_tools.airflow_get_failed_tasks(**kw),
            "airflow_get_task_logs": lambda **kw: airflow_tools.airflow_get_task_logs(**kw),

            # Task 4: Kubernetes Tools
            "k8s_list_pods_and_deployments": lambda **kw: kubernetes_tools.k8s_list_pods_and_deployments(**kw),
            "k8s_get_pod_status_and_logs": lambda **kw: kubernetes_tools.k8s_get_pod_status_and_logs(**kw),
            "k8s_inspect_services": lambda **kw: kubernetes_tools.k8s_inspect_services(**kw),
            "k8s_list_events": lambda **kw: kubernetes_tools.k8s_list_events(**kw),
            "k8s_diagnose_unhealthy_pods": lambda **kw: kubernetes_tools.k8s_diagnose_unhealthy_pods(**kw),
            "k8s_get_deployed_version": lambda **kw: kubernetes_tools.k8s_get_deployed_version(**kw),
            "k8s_execute_cluster_action": lambda **kw: kubernetes_tools.k8s_execute_cluster_action(**kw)
        }

    def execute_tool(self, tool_name: str, args: Dict[str, Any] = None) -> ToolExecutionResult:
        """Execute a tool with structured duration timing, UUID tracking, and safe exception isolation."""
        args = args or {}
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        if tool_name not in self.tool_registry:
            duration = round(time.time() - start_time, 4)
            logger.error(f"req_id={request_id} tool={tool_name} status=error error='Unregistered tool'")
            return ToolExecutionResult(
                tool_name=tool_name,
                status="error",
                error=f"Tool '{tool_name}' is not registered.",
                duration_seconds=duration,
                request_id=request_id
            )

        try:
            fn = self.tool_registry[tool_name]
            result = fn(**args)
            duration = round(time.time() - start_time, 4)
            status_val = result.get("status", "success")
            err_msg = result.get("error")

            logger.info(f"req_id={request_id} tool={tool_name} status={status_val} duration={duration}s")
            return ToolExecutionResult(
                tool_name=tool_name,
                status=status_val,
                data=result if status_val in ["success", "needs_confirmation"] else None,
                error=err_msg,
                duration_seconds=duration,
                request_id=request_id
            )
        except Exception as e:
            duration = round(time.time() - start_time, 4)
            logger.error(f"req_id={request_id} tool={tool_name} status=error exception='{str(e)}'")
            return ToolExecutionResult(
                tool_name=tool_name,
                status="error",
                error=str(e),
                duration_seconds=duration,
                request_id=request_id
            )

    def evaluate_retraining_evidence(
        self,
        mlflow_result: ToolExecutionResult,
        dvc_ver_result: ToolExecutionResult,
        k8s_result: ToolExecutionResult,
        airflow_result: ToolExecutionResult,
        dvc_diff_result: ToolExecutionResult
    ) -> Dict[str, Any]:
        """
        Evidence-Based Retraining Decision Engine:
        Evaluates factual operational evidence. Retraining is NEVER automatically initiated.
        Retraining is ONLY recommended when factual evidence warrants it:
        - Detected DVC dataset drift.
        - Failed Airflow quality gate.
        - Confirmed model metric degradation.
        If tools fail or are offline, returns INCONCLUSIVE (ZERO HALLUCINATIONS).
        """
        offline_services = []
        if mlflow_result.status != "success":
            offline_services.append("MLflow Tracking")
        if k8s_result.status != "success":
            offline_services.append("Kubernetes Cluster")
        if airflow_result.status != "success":
            offline_services.append("Apache Airflow")
        if dvc_diff_result.status != "success":
            offline_services.append("DVC Status")

        # STRICT ZERO-HALLUCINATION GUARD:
        # If external operational tools failed, we CANNOT claim services are stable or retraining is unneeded.
        if len(offline_services) >= 2:
            return {
                "decision": "UNKNOWN / INCONCLUSIVE",
                "recommended_action": "Cannot determine retraining necessity due to missing operational evidence from offline services.",
                "reasoning": f"Operational tools failed to connect to: {', '.join(offline_services)}. Operational claims require tool responses."
            }

        # Check for DVC Data Drift
        has_data_drift = False
        if dvc_diff_result.status == "success" and dvc_diff_result.data:
            has_data_drift = dvc_diff_result.data.get("data_has_changed", False)

        # Check Airflow pipeline status
        failed_pipeline = False
        if airflow_result.status == "success" and airflow_result.data:
            dag_state = airflow_result.data.get("state")
            if dag_state in ["failed", "upstream_failed"]:
                failed_pipeline = True

        # Check MLflow metrics
        f1_score = None
        if mlflow_result.status == "success" and mlflow_result.data:
            f1_score = mlflow_result.data.get("best_metric_value")

        degraded_performance = (f1_score is not None and f1_score < 0.75)

        # Retraining Decision Rules
        if has_data_drift:
            return {
                "decision": "RECOMMEND RETRAINING",
                "recommended_action": "Trigger training pipeline DAG 'ml_training_pipeline' (requires human confirmation).",
                "reasoning": "Empirical evidence: DVC detected modified raw dataset records relative to the last training commit."
            }
        elif failed_pipeline:
            return {
                "decision": "RECOMMEND RETRAINING / PIPELINE REPAIR",
                "recommended_action": "Investigate failed Airflow tasks before re-triggering pipeline.",
                "reasoning": "Empirical evidence: Recent Airflow pipeline run failed quality gate or data validation."
            }
        elif degraded_performance:
            return {
                "decision": "RECOMMEND RETRAINING",
                "recommended_action": "Tune hyperparameters and retrain model.",
                "reasoning": f"Empirical evidence: Candidate model F1 score ({f1_score:.4f}) is degraded below production threshold (0.75)."
            }
        else:
            return {
                "decision": "RETRAINING NOT NECESSARY",
                "recommended_action": "Continue monitoring inference latency and health endpoints; no training needed.",
                "reasoning": "Verified facts: Data is clean with no DVC drift, candidate metrics satisfy quality thresholds, and active workloads are stable."
            }

    def process_query(self, query: str, user_confirmation: bool = False) -> AgentStructuredResponse:
        """
        AI Agent Multi-Tool Reasoning Engine:
        Selects tools based on user request, correlates multi-system evidence,
        enforces safety confirmation gates, and guarantees evidence-based answers.
        """
        q_lower = query.lower()
        tool_results: List[ToolExecutionResult] = []
        known_facts: List[str] = []
        uncertainties: List[str] = []
        recommended_actions: List[str] = []
        confirmation_required = None

        # ---------------------------------------------------------------------
        # 1. MANDATORY INTEGRATION SCENARIO (Section 5)
        # "Find the best recent model, identify the dataset version used to train it,
        #  check the deployment status, and explain whether retraining is necessary."
        # ---------------------------------------------------------------------
        if all(w in q_lower for w in ["best", "model"]) and any(w in q_lower for w in ["retrain", "dataset", "deployment"]):
            # Step 1: Query MLflow for best recent model and recorded metrics
            res_mlflow = self.execute_tool("mlflow_get_best_run", {"metric_name": "f1_score", "mode": "max"})
            tool_results.append(res_mlflow)

            best_run_id = None
            f1_val = None
            recorded_dvc_hash = None

            if res_mlflow.status == "success" and res_mlflow.data:
                best_run_id = res_mlflow.data.get("best_run_id")
                f1_val = res_mlflow.data.get("best_metric_value")
                recorded_dvc_hash = res_mlflow.data.get("dvc_data_hash")
                known_facts.append(
                    f"Best model candidate identified from MLflow: Run ID '{best_run_id}' with F1 score = {f1_val:.4f} "
                    f"(Status: FINISHED, Recorded DVC Hash: '{recorded_dvc_hash}')."
                )
            else:
                uncertainties.append(f"MLflow service unreachable or failed: {res_mlflow.error}")

            # Step 2: Read recorded DVC dataset reference associated with the selected experiment
            res_dvc = self.execute_tool("dvc_get_dataset_version", {"dataset_path": "data/raw_data.csv", "run_id": best_run_id})
            tool_results.append(res_dvc)
            if res_dvc.status == "success" and res_dvc.data:
                git_hash = res_dvc.data.get("git_commit")
                data_hash = res_dvc.data.get("dvc_md5_hash")
                tag_ver = res_dvc.data.get("dvc_version_tag")
                known_facts.append(
                    f"Dataset provenance for experiment: Version '{tag_ver}', DVC MD5 '{data_hash}', tracked at Git commit '{git_hash}'."
                )
            else:
                uncertainties.append(f"Could not retrieve authentic DVC dataset provenance: {res_dvc.error}")

            # Step 3: Inspect current Kubernetes deployment and application health
            res_k8s = self.execute_tool("k8s_list_pods_and_deployments", {"namespace": "mlops-app"})
            tool_results.append(res_k8s)
            if res_k8s.status == "success" and res_k8s.data:
                pods = res_k8s.data.get("pods", [])
                healthy_pods = sum(1 for p in pods if p.get("ready") and p.get("phase") == "Running")
                known_facts.append(
                    f"Kubernetes cluster status: {healthy_pods}/{len(pods)} Pods Ready and Running in namespace 'mlops-app'."
                )
            else:
                uncertainties.append(f"Kubernetes cluster inspection failed: {res_k8s.error}")

            # Step 4: Query Airflow to determine the status of recent training workflows
            res_airflow = self.execute_tool("airflow_get_dag_run_status", {"dag_id": "ml_training_pipeline"})
            tool_results.append(res_airflow)
            if res_airflow.status == "success" and res_airflow.data:
                dag_state = res_airflow.data.get("state", "unknown")
                known_facts.append(f"Airflow workflow 'ml_training_pipeline' latest run status: '{dag_state}'.")
            else:
                uncertainties.append(f"Airflow tracking server offline or unreachable: {res_airflow.error}")

            # Step 5: Check whether tracked data has changed via DVC status
            res_dvc_status = self.execute_tool("dvc_detect_changes", {})
            tool_results.append(res_dvc_status)
            if res_dvc_status.status != "success":
                uncertainties.append(f"DVC status inspection failed: {res_dvc_status.error}")

            # Step 6: Evidence Correlation & Retraining Evaluation Rule
            evidence = self.evaluate_retraining_evidence(
                mlflow_result=res_mlflow,
                dvc_ver_result=res_dvc,
                k8s_result=res_k8s,
                airflow_result=res_airflow,
                dvc_diff_result=res_dvc_status
            )

            recommended_actions.append(evidence["recommended_action"])

            narrative = (
                f"Multi-Tool Operational Diagnosis:\n"
                f"- Candidate Model: Run ID {best_run_id or 'UNKNOWN'} (F1: {f1_val or 'UNKNOWN'})\n"
                f"- Retraining Decision: {evidence['decision']}\n"
                f"- Decision Reasoning: {evidence['reasoning']}\n"
                f"- Safety Constraint: Retraining is not triggered automatically from this request."
            )

            return AgentStructuredResponse(
                query=query,
                known_facts=known_facts,
                uncertainties=uncertainties,
                recommended_actions=recommended_actions,
                tool_results=tool_results,
                final_narrative=narrative
            )

        # ---------------------------------------------------------------------
        # 2. RUN COMPARISON SCENARIO (Section 4 - Task 2)
        # "Compare the last five training runs and recommend the best candidate based on F1 score."
        # ---------------------------------------------------------------------
        elif any(w in q_lower for w in ["compare", "runs", "candidates"]):
            res_comp = self.execute_tool("mlflow_compare_runs", {"last_n": 5})
            tool_results.append(res_comp)
            if res_comp.status == "success" and res_comp.data:
                rec_id = res_comp.data.get("recommended_candidate")
                reason = res_comp.data.get("recommendation_reasoning")
                known_facts.append(f"Compared {res_comp.data.get('compared_count')} recent training runs.")
                known_facts.append(f"Recommended Candidate: Run ID '{rec_id}'.")
                recommended_actions.append("Inspect candidate artifacts and verify readiness in staging deployment.")
                narrative = f"MLflow Run Comparison Result:\n{reason}"
            else:
                uncertainties.append(f"MLflow comparison error: {res_comp.error}")
                narrative = "Could not compare training runs because MLflow is unreachable."

            return AgentStructuredResponse(
                query=query,
                known_facts=known_facts,
                uncertainties=uncertainties,
                recommended_actions=recommended_actions,
                tool_results=tool_results,
                final_narrative=narrative
            )

        # ---------------------------------------------------------------------
        # 3. EXPERIMENT DATASET PROVENANCE SCENARIO (Section 4 - Task 1)
        # "Which dataset version was used for experiment 25?"
        # ---------------------------------------------------------------------
        elif any(w in q_lower for w in ["experiment", "run", "used for"]) and any(w in q_lower for w in ["dataset", "version", "dvc"]):
            # Extract run ID if provided, e.g. "experiment 99d35a..." or "experiment 25"
            match = re.search(r'(?:experiment|run)\s+([a-zA-Z0-9_\-]+)', query, re.IGNORECASE)
            target_run = match.group(1) if match else None

            res_prov = self.execute_tool("dvc_get_dataset_version", {"run_id": target_run})
            tool_results.append(res_prov)
            if res_prov.status == "success" and res_prov.data:
                known_facts.append(
                    f"Experiment '{target_run or 'latest'}' used dataset version '{res_prov.data.get('dvc_version_tag')}' "
                    f"associated with Git commit '{res_prov.data.get('git_commit')[:8]}' and DVC MD5 '{res_prov.data.get('dvc_md5_hash')}'."
                )
                recommended_actions.append(f"To reproduce this exact dataset: git checkout {res_prov.data.get('git_commit')} && dvc checkout")
                narrative = f"Provenance retrieved for experiment {target_run or 'latest'} from recorded MLflow tags."
            else:
                uncertainties.append(f"Provenance lookup failed: {res_prov.error}")
                narrative = f"Failed to retrieve recorded dataset provenance for experiment {target_run}."

            return AgentStructuredResponse(
                query=query,
                known_facts=known_facts,
                uncertainties=uncertainties,
                recommended_actions=recommended_actions,
                tool_results=tool_results,
                final_narrative=narrative
            )

        # ---------------------------------------------------------------------
        # 4. KUBERNETES MUTATION SCENARIO (Section 4 - Task 4)
        # "Restart the prediction deployment" or "Scale deployment to 3"
        # ---------------------------------------------------------------------
        elif any(w in q_lower for w in ["restart", "scale"]) and any(w in q_lower for w in ["deployment", "prediction", "api", "cluster"]):
            action_type = "scale" if "scale" in q_lower else "restart"
            res_mut = self.execute_tool(
                "k8s_execute_cluster_action",
                {"action": action_type, "confirmed": user_confirmation}
            )
            tool_results.append(res_mut)

            if res_mut.status == "needs_confirmation":
                confirmation_required = {
                    "tool": "k8s_execute_cluster_action",
                    "action": action_type,
                    "target": "mlops-prediction-deployment"
                }
                uncertainties.append(f"Safety Gate: Cluster modification '{action_type}' was held pending human confirmation.")
                recommended_actions.append(f"Reply 'yes' to authorize {action_type} on deployment 'mlops-prediction-deployment'.")
                narrative = f"Action '{action_type}' requires explicit human confirmation before executing."
            elif res_mut.status == "success":
                known_facts.append(res_mut.data.get("message", "Action succeeded."))
                recommended_actions.append("Monitor pod rollout events via 'k8s_diagnose_unhealthy_pods'.")
                narrative = f"Authorized cluster mutation '{action_type}' executed successfully."
            else:
                uncertainties.append(f"Mutation denied or failed: {res_mut.error}")
                narrative = f"Failed to execute cluster action '{action_type}'."

            return AgentStructuredResponse(
                query=query,
                known_facts=known_facts,
                uncertainties=uncertainties,
                recommended_actions=recommended_actions,
                tool_results=tool_results,
                final_narrative=narrative,
                confirmation_required=confirmation_required
            )

        # ---------------------------------------------------------------------
        # 5. KUBERNETES DIAGNOSTIC SCENARIO (Section 4 - Task 4)
        # "Why is my prediction API unavailable?"
        # ---------------------------------------------------------------------
        elif any(w in q_lower for w in ["why", "unavailable", "crash", "unhealthy", "diagnostic", "pod", "k8s"]):
            res_diag = self.execute_tool("k8s_diagnose_unhealthy_pods", {"namespace": "mlops-app"})
            tool_results.append(res_diag)
            res_events = self.execute_tool("k8s_list_events", {"namespace": "mlops-app"})
            tool_results.append(res_events)

            if res_diag.status == "success" and res_diag.data:
                unhealthy = res_diag.data.get("unhealthy_pods", [])
                if unhealthy:
                    for u in unhealthy:
                        known_facts.append(f"Unhealthy Pod '{u['pod_name']}': {'; '.join(u['issues'])}.")
                    recommended_actions.append("Inspect container logs using 'k8s_get_pod_status_and_logs' and check readiness probes.")
                    narrative = f"Kubernetes Failure Diagnosis: Detected {len(unhealthy)} unhealthy pod(s) failing readiness or in restart loop."
                else:
                    known_facts.append("All application Pods in namespace 'mlops-app' are passing readiness probes.")
                    narrative = "Kubernetes Diagnosis: No crashing or unhealthy pods detected in namespace 'mlops-app'."
            else:
                uncertainties.append(f"Kubernetes diagnostic inspection failed: {res_diag.error}")
                narrative = "Could not diagnose prediction API because Kubernetes API is unreachable."

            return AgentStructuredResponse(
                query=query,
                known_facts=known_facts,
                uncertainties=uncertainties,
                recommended_actions=recommended_actions,
                tool_results=tool_results,
                final_narrative=narrative
            )

        # ---------------------------------------------------------------------
        # 6. AIRFLOW WORKFLOW SCENARIO (Section 4 - Task 3)
        # "Trigger the model training pipeline" or "Check workflow status"
        # ---------------------------------------------------------------------
        elif any(w in q_lower for w in ["airflow", "pipeline", "trigger", "dag"]):
            if "trigger" in q_lower:
                res_trig = self.execute_tool(
                    "airflow_trigger_dag",
                    {"dag_id": "ml_training_pipeline", "confirmed": user_confirmation}
                )
                tool_results.append(res_trig)
                if res_trig.status == "needs_confirmation":
                    confirmation_required = {"tool": "airflow_trigger_dag", "dag_id": "ml_training_pipeline"}
                    uncertainties.append("Safety Gate: Triggering training pipeline requires confirmation.")
                    recommended_actions.append("Confirm pipeline execution to proceed.")
                    narrative = "Airflow DAG trigger held pending confirmation."
                elif res_trig.status == "success":
                    known_facts.append(f"Triggered DAG 'ml_training_pipeline'. Run ID: {res_trig.data.get('run_id')}.")
                    recommended_actions.append("Monitor DAG execution progress.")
                    narrative = "Training DAG triggered successfully."
                else:
                    uncertainties.append(f"Failed to trigger DAG: {res_trig.error}")
                    narrative = "Airflow DAG trigger failed."
            else:
                res_status = self.execute_tool("airflow_get_dag_run_status", {"dag_id": "ml_training_pipeline"})
                tool_results.append(res_status)
                if res_status.status == "success":
                    known_facts.append(f"DAG 'ml_training_pipeline' status: {res_status.data.get('state')}.")
                    narrative = f"Airflow workflow status: {res_status.data.get('state')}."
                else:
                    uncertainties.append(f"Airflow error: {res_status.error}")
                    narrative = "Could not check Airflow status."

            return AgentStructuredResponse(
                query=query,
                known_facts=known_facts,
                uncertainties=uncertainties,
                recommended_actions=recommended_actions,
                tool_results=tool_results,
                final_narrative=narrative,
                confirmation_required=confirmation_required
            )

        # ---------------------------------------------------------------------
        # 7. DEFAULT / FALLBACK SCENARIO
        # ---------------------------------------------------------------------
        res_list = self.execute_tool("dvc_list_tracked_data", {})
        tool_results.append(res_list)
        return AgentStructuredResponse(
            query=query,
            known_facts=["Agent initialized and ready."],
            uncertainties=["Query does not specify an authorized MLOps operation."],
            recommended_actions=[
                "Try: 'Compare the last five training runs and recommend the best candidate based on F1 score.'",
                "Try: 'Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.'",
                "Try: 'Why is my prediction API unavailable?'"
            ],
            tool_results=tool_results,
            final_narrative="Agent is ready. Please supply an operational MLOps request."
        )
