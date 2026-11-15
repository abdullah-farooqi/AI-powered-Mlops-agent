import logging
from agent.orchestrator import MLOpsAgentOrchestrator, SecretRedactionFilter

def test_zero_hallucination_when_tools_fail(mocker):
    """
    CRITICAL SAFETY TEST: Proves that the agent does NOT invent operational facts
    when external services are offline.
    """
    mocker.patch("agent.tools.mlflow_tools.mlflow_get_best_run", return_value={
        "status": "error",
        "error": "MLflow tracking server unreachable at localhost:5000"
    })
    mocker.patch("agent.tools.kubernetes_tools.k8s_list_pods_and_deployments", return_value={
        "status": "error",
        "error": "Kubernetes cluster connection timed out"
    })
    mocker.patch("agent.tools.airflow_tools.airflow_get_dag_run_status", return_value={
        "status": "error",
        "error": "Airflow service unavailable"
    })

    agent = MLOpsAgentOrchestrator()
    query = "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
    resp = agent.process_query(query)

    # Must list errors under uncertainties
    assert len(resp.uncertainties) >= 2
    assert any("MLflow" in u for u in resp.uncertainties)
    assert any("Kubernetes" in u for u in resp.uncertainties)

    # ZERO HALLUCINATION VERIFICATION:
    # Must NOT claim that retraining is not necessary or that pods are healthy!
    assert "RETRAINING NOT NECESSARY" not in resp.final_narrative
    assert "Current model satisfies performance gates" not in resp.final_narrative
    assert "UNKNOWN / INCONCLUSIVE" in resp.final_narrative

def test_tool_timeout_handling(mocker):
    """Verifies that slow external subprocess calls or API queries do not hang the orchestrator."""
    mocker.patch("subprocess.run", side_effect=Exception("Subprocess timed out after 10s"))

    agent = MLOpsAgentOrchestrator()
    res = agent.execute_tool("dvc_detect_changes", {})

    assert res.status == "error"
    assert "timed out" in res.error.lower() or "exception" in res.error.lower()

def test_structured_logging_secret_redaction():
    """Verifies that secrets, bearer tokens, and passwords are redacted from log messages."""
    filter_obj = SecretRedactionFilter()
    record = logging.LogRecord(
        name="TestLogger",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Connecting to service with password=my_super_secret_password and Bearer eyJhbGciOiJIUzI1NiJ9",
        args=(),
        exc_info=None
    )
    filter_obj.filter(record)
    assert "my_super_secret_password" not in record.msg
    assert "Bearer [REDACTED]" in record.msg
    assert 'password="[REDACTED]"' in record.msg
