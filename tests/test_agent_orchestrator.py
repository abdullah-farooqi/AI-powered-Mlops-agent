from agent.orchestrator import MLOpsAgentOrchestrator

def test_mandatory_multi_tool_integration_scenario(mocker):
    mocker.patch("agent.tools.mlflow_tools.mlflow_get_best_run", return_value={
        "status": "success",
        "best_run_id": "exp_run_99",
        "best_metric_value": 0.95,
        "dvc_data_version": "data-v2",
        "dvc_data_hash": "c0ffee123"
    })
    mocker.patch("agent.tools.dvc_tools.dvc_get_dataset_version", return_value={
        "status": "success",
        "dataset_path": "data/raw_data.csv",
        "dvc_version_tag": "data-v2",
        "git_commit": "c0ffee1",
        "dvc_md5_hash": "c0ffee123"
    })
    mocker.patch("agent.tools.kubernetes_tools.k8s_list_pods_and_deployments", return_value={
        "status": "success",
        "pods": [{"phase": "Running", "ready": True}],
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
    assert any("data-v2" in fact for fact in resp.known_facts)
    assert any("RETRAINING NOT NECESSARY" in resp.final_narrative for _ in [1])
    assert "Retraining is not triggered automatically" in resp.final_narrative

def test_tool_selection_run_comparison(mocker):
    mocker.patch("agent.tools.mlflow_tools.mlflow_compare_runs", return_value={
        "status": "success",
        "compared_count": 5,
        "recommended_candidate": "run_top_5",
        "recommendation_reasoning": "Run top_5 achieved highest balanced F1 score 0.93."
    })

    agent = MLOpsAgentOrchestrator()
    resp = agent.process_query("Compare the last five training runs and recommend the best candidate based on F1 score.")

    assert any(t.tool_name == "mlflow_compare_runs" for t in resp.tool_results)
    assert any("run_top_5" in fact for fact in resp.known_facts)

def test_tool_selection_experiment_provenance(mocker):
    mocker.patch("agent.tools.dvc_tools.dvc_get_dataset_version", return_value={
        "status": "success",
        "dvc_version_tag": "data-v1",
        "git_commit": "a4c92e1",
        "dvc_md5_hash": "deadbeef99"
    })

    agent = MLOpsAgentOrchestrator()
    resp = agent.process_query("Which dataset version was used for experiment 25?")

    assert any(t.tool_name == "dvc_get_dataset_version" for t in resp.tool_results)
    assert any("a4c92e1" in fact for fact in resp.known_facts)

def test_tool_selection_k8s_restart_confirmation():
    agent = MLOpsAgentOrchestrator()
    resp = agent.process_query("Restart the prediction deployment", user_confirmation=False)

    assert any(t.tool_name == "k8s_execute_cluster_action" for t in resp.tool_results)
    assert resp.confirmation_required is not None
    assert resp.confirmation_required["action"] == "restart"
