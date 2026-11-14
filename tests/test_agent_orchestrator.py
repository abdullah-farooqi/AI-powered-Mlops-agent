from agent.orchestrator import MLOpsAgentOrchestrator

def test_mandatory_multi_tool_integration_scenario(mocker):
    mocker.patch("agent.tools.mlflow_tools.mlflow_get_best_run", return_value={
        "status": "success",
        "best_run_id": "exp_run_99",
        "best_metric_value": 0.95,
        "dvc_data_version": "v2.0"
    })
    mocker.patch("agent.tools.dvc_tools.dvc_get_dataset_version", return_value={
        "status": "success",
        "dataset_path": "data/raw_data.csv",
        "git_commit": "c0ffee1",
        "dvc_md5_hash": "e99a18c428cb38d5f260853678922e03"
    })
    mocker.patch("agent.tools.kubernetes_tools.k8s_list_pods_and_deployments", return_value={
        "status": "success",
        "pods": [{"phase": "Running"}],
        "deployments": [{"name": "mlops-prediction-deployment"}]
    })
    mocker.patch("agent.tools.airflow_tools.airflow_get_dag_run_status", return_value={
        "status": "success",
        "state": "success"
    })
    mocker.patch("agent.tools.dvc_tools.dvc_detect_changes", return_value={
        "status": "success",
        "data_has_changed": False
    })

    agent = MLOpsAgentOrchestrator()
    query = "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
    resp = agent.process_query(query)

    assert len(resp.tool_results) >= 4
    assert any("exp_run_99" in fact for fact in resp.known_facts)
    assert any("v2.0" in fact for fact in resp.known_facts)
    assert any("Retraining is NOT necessary" in action for action in resp.recommended_actions)
