from unittest.mock import MagicMock
from agent.tools.mlflow_tools import (
    mlflow_list_experiments_and_runs,
    mlflow_get_run_details,
    mlflow_get_best_run,
    mlflow_compare_runs
)

def test_mlflow_get_best_run_mocked(mocker):
    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_123"
    mock_run.data.metrics = {"f1_score": 0.94, "accuracy": 0.96}
    mock_run.data.params = {"n_estimators": "50"}
    mock_run.data.tags = {"dvc_data_version": "v1.2", "git_commit": "a4c92e1"}

    mock_client = MagicMock()
    mock_exp = MagicMock()
    mock_exp.experiment_id = "exp_01"
    mock_client.get_experiment_by_name.return_value = mock_exp
    mock_client.search_runs.return_value = [mock_run]

    mocker.patch("agent.tools.mlflow_tools._get_client", return_value=mock_client)

    res = mlflow_get_best_run("mlops-model-training", metric_name="f1_score")
    assert res["status"] == "success"
    assert res["best_run_id"] == "test_run_123"
    assert res["best_metric_value"] == 0.94
    assert res["dvc_data_version"] == "v1.2"

def test_mlflow_unavailable_returns_error(mocker):
    mocker.patch("agent.tools.mlflow_tools._get_client", side_effect=Exception("Connection refused to 5000"))
    res = mlflow_list_experiments_and_runs("mlops-model-training")
    assert res["status"] == "error"
    assert "Connection refused" in res["error"]
