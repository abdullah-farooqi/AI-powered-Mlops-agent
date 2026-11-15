from unittest.mock import MagicMock
from agent.tools.airflow_tools import airflow_trigger_dag, airflow_list_dags


def test_airflow_trigger_unauthorized_dag_rejected():
    res = airflow_trigger_dag("malicious_unauthorized_dag")
    assert res["status"] == "denied"
    assert "Authorization failure" in res["error"]


def test_airflow_trigger_requires_confirmation():
    res = airflow_trigger_dag("ml_training_pipeline", confirmed=False)
    assert res["status"] == "needs_confirmation"
    assert "Confirmation required" in res["message"]


def test_airflow_trigger_authorized_with_confirmation(mocker):
    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = {"dag_run_id": "manual__2026", "state": "queued"}
    mocker.patch("requests.post", return_value=mock_resp)

    res = airflow_trigger_dag("ml_training_pipeline", confirmed=True)
    assert res["status"] == "success"
    assert res["run_id"] == "manual__2026"


def test_airflow_unreachable_returns_honest_error(mocker):
    mocker.patch("requests.get", side_effect=Exception("Connection refused on 8080"))
    res = airflow_list_dags()
    assert res["status"] == "error"
    assert "Connection refused" in res["error"]
    # Ensure no fabricated simulated DAGs are returned
    assert "simulated_available_dags" not in res
