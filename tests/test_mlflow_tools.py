from unittest.mock import MagicMock
from agent.tools.mlflow_tools import mlflow_get_best_run, mlflow_compare_runs, mlflow_get_registered_model_info


def test_mlflow_get_best_run_enforces_finished_status(mocker):
    mock_run = MagicMock()
    mock_run.info.run_id = "run_finished_1"
    mock_run.info.status = "FINISHED"
    mock_run.data.metrics = {"f1_score": 0.92, "accuracy": 0.94}
    mock_run.data.params = {"n_estimators": "50"}
    mock_run.data.tags = {"dvc_data_version": "data-v1", "dvc_data_hash": "hash123", "git_commit": "c0ffee"}

    mock_client = MagicMock()
    mock_exp = MagicMock()
    mock_exp.experiment_id = "exp_01"
    mock_client.get_experiment_by_name.return_value = mock_exp
    mock_client.search_runs.return_value = [mock_run]

    mocker.patch("agent.tools.mlflow_tools._get_client", return_value=mock_client)

    res = mlflow_get_best_run("mlops-model-training", metric_name="f1_score")
    assert res["status"] == "success"
    assert res["best_run_id"] == "run_finished_1"
    assert res["best_metric_value"] == 0.92
    assert res["production_ready_candidate"] is True


def test_mlflow_compare_runs_ranking(mocker):
    r1 = MagicMock()
    r1.info.run_id = "run_1"
    r1.info.status = "FINISHED"
    r1.data.metrics = {"f1_score": 0.70, "accuracy": 0.75}
    r1.data.params = {"n_estimators": "10"}
    r1.data.tags = {"dvc_data_hash": "hash_a", "dvc_data_version": "v1.0", "git_commit": "c1"}

    r2 = MagicMock()
    r2.info.run_id = "run_2"
    r2.info.status = "FINISHED"
    r2.data.metrics = {"f1_score": 0.91, "accuracy": 0.92}
    r2.data.params = {"n_estimators": "100"}
    r2.data.tags = {"dvc_data_hash": "hash_b", "dvc_data_version": "v2.0", "git_commit": "c2"}

    mock_client = MagicMock()
    mock_exp = MagicMock()
    mock_client.get_experiment_by_name.return_value = mock_exp
    mock_client.search_runs.return_value = [r1, r2]

    mocker.patch("agent.tools.mlflow_tools._get_client", return_value=mock_client)

    res = mlflow_compare_runs(last_n=2)
    assert res["status"] == "success"
    assert res["recommended_candidate"] == "run_2"
    assert "run_2" in res["recommendation_reasoning"]


def test_mlflow_get_registered_model_info_error_on_failure(mocker):
    mock_client = MagicMock()
    mock_client.get_latest_versions.side_effect = Exception("Registry storage unavailable")
    mocker.patch("agent.tools.mlflow_tools._get_client", return_value=mock_client)

    res = mlflow_get_registered_model_info("iris_classifier")
    assert res["status"] == "error"
    assert "Registry storage unavailable" in res["error"]
