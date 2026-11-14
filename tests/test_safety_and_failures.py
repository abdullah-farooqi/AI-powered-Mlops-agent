from agent.orchestrator import MLOpsAgentOrchestrator

def test_no_hallucination_when_tools_fail(mocker):
    mocker.patch("agent.tools.mlflow_tools.mlflow_get_best_run", return_value={
        "status": "error",
        "error": "MLflow tracking server unreachable at 5000"
    })
    mocker.patch("agent.tools.kubernetes_tools.k8s_list_pods_and_deployments", return_value={
        "status": "error",
        "error": "Kubernetes cluster connection timed out"
    })

    agent = MLOpsAgentOrchestrator()
    query = "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
    resp = agent.process_query(query)

    assert len(resp.uncertainties) >= 1
    assert any("MLflow" in u for u in resp.uncertainties)
    assert any("Kubernetes" in u for u in resp.uncertainties)

def test_structured_audit_logging(caplog):
    import logging
    caplog.set_level(logging.INFO)
    agent = MLOpsAgentOrchestrator()
    agent.execute_tool("dvc_list_tracked_data", {})
    
    assert any("req_id=" in record.message for record in caplog.records)
    assert any("tool=dvc_list_tracked_data" in record.message for record in caplog.records)
