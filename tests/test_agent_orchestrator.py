from agent.orchestrator import MLOpsAgentOrchestrator


def test_mandatory_multi_tool_integration_scenario(mocker):
    mocker.patch(
        "agent.tools.mlflow_tools.mlflow_get_best_run",
        return_value={
            "status": "success",
            "best_run_id": "exp_run_99",
            "best_metric_value": 0.95,
            "dvc_data_version": "data-v2",
            "dvc_data_hash": "c0ffee123",
        },
    )
    mocker.patch(
        "agent.tools.dvc_tools.dvc_get_dataset_version",
        return_value={
            "status": "success",
            "dataset_path": "data/raw_data.csv",
            "dvc_version_tag": "data-v2",
            "git_commit": "c0ffee1",
            "dvc_md5_hash": "c0ffee123",
        },
    )
    mocker.patch(
        "agent.tools.kubernetes_tools.k8s_list_pods_and_deployments",
        return_value={
            "status": "success",
            "pods": [{"phase": "Running", "ready": True}],
            "deployments": [{"name": "mlops-prediction-deployment"}],
        },
    )
    mocker.patch(
        "agent.tools.airflow_tools.airflow_get_dag_run_status", return_value={"status": "success", "state": "success"}
    )
    mocker.patch(
        "agent.tools.dvc_tools.dvc_detect_changes", return_value={"status": "success", "data_has_changed": False}
    )

    agent = MLOpsAgentOrchestrator()
    query = (
        "Find the best recent model, identify the dataset version used to train it, "
        "check the deployment status, and explain whether retraining is necessary."
    )
    resp = agent.process_query(query)

    assert len(resp.tool_results) >= 4
    assert any("exp_run_99" in fact for fact in resp.known_facts)
    assert any("data-v2" in fact for fact in resp.known_facts)
    assert "NO RETRAINING NECESSARY" in resp.final_narrative
    assert "Retraining is not triggered automatically" in resp.final_narrative


def test_tool_selection_run_comparison(mocker):
    mocker.patch(
        "agent.tools.mlflow_tools.mlflow_compare_runs",
        return_value={
            "status": "success",
            "compared_count": 5,
            "defensible_recommendation": "Run top_5 achieved highest balanced F1 score 0.93.",
        },
    )

    agent = MLOpsAgentOrchestrator()
    resp = agent.process_query(
        "Compare the last five training runs and recommend the best candidate based on F1 score."
    )

    assert any(t.tool_name == "mlflow_compare_runs" for t in resp.tool_results)
    assert any("Run top_5" in fact for fact in resp.known_facts)


def test_tool_selection_experiment_provenance(mocker):
    mocker.patch(
        "agent.tools.dvc_tools.dvc_get_dataset_version",
        return_value={
            "status": "success",
            "dvc_version_tag": "data-v1",
            "git_commit": "a4c92e1",
            "dvc_md5_hash": "deadbeef99",
        },
    )

    agent = MLOpsAgentOrchestrator()
    resp = agent.process_query("What is the dataset version used for this model?")

    assert any(t.tool_name == "dvc_get_dataset_version" for t in resp.tool_results)
    assert any("deadbeef99" in fact for fact in resp.known_facts)


def test_tool_selection_k8s_restart_confirmation():
    agent = MLOpsAgentOrchestrator()
    resp = agent.process_query("Restart the prediction deployment", confirmed=False)

    assert any(t.tool_name == "k8s_execute_cluster_action" for t in resp.tool_results)
    assert resp.confirmation_required is not None
    assert resp.confirmation_required["tool"] == "k8s_execute_cluster_action"


def test_tool_selection_airflow_failed_tasks_and_logs(mocker):
    mocker.patch(
        "agent.tools.airflow_tools.airflow_get_failed_tasks",
        return_value={
            "status": "success",
            "failed_count": 1,
            "failed_tasks": [{"task_id": "quality_gate", "state": "failed"}],
        },
    )
    mocker.patch(
        "agent.tools.airflow_tools.airflow_get_task_logs",
        return_value={"status": "success", "logs_tail": "Quality Gate Failed! F1 score below threshold."},
    )

    agent = MLOpsAgentOrchestrator()
    resp = agent.process_query("Show me the failed Airflow tasks and their logs")

    tool_names = [t.tool_name for t in resp.tool_results]
    assert "airflow_get_failed_tasks" in tool_names
    assert "airflow_get_task_logs" in tool_names
    assert any("Airflow failed task count: 1" in f for f in resp.known_facts)


def test_tool_selection_k8s_services_and_pod_logs(mocker):
    mocker.patch(
        "agent.tools.kubernetes_tools.k8s_get_pod_status_and_logs",
        return_value={"status": "success", "logs": "Uvicorn running on http://0.0.0.0:8000"},
    )
    mocker.patch(
        "agent.tools.kubernetes_tools.k8s_inspect_services",
        return_value={"status": "success", "services_count": 1, "services": [{"name": "mlops-prediction-service"}]},
    )

    agent = MLOpsAgentOrchestrator()
    resp = agent.process_query("Show the Kubernetes services and pod logs")

    tool_names = [t.tool_name for t in resp.tool_results]
    assert "k8s_get_pod_status_and_logs" in tool_names
    assert "k8s_inspect_services" in tool_names


def test_tool_selection_dvc_compare_and_reproduce(mocker):
    mocker.patch(
        "agent.tools.dvc_tools.dvc_compare_versions",
        return_value={"status": "success", "summary": "Dataset modified: hash changed from c9156e0d to 4224576f."},
    )

    agent = MLOpsAgentOrchestrator()
    resp = agent.process_query("Compare dataset versions data-v1 and data-v2")

    assert any(t.tool_name == "dvc_compare_versions" for t in resp.tool_results)
    assert any("Dataset modified" in f for f in resp.known_facts)
