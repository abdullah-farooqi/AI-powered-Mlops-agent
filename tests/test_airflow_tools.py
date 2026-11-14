from agent.tools.airflow_tools import airflow_trigger_dag, airflow_list_dags

def test_airflow_trigger_unauthorized_dag_rejected():
    res = airflow_trigger_dag("unauthorized_malicious_dag")
    assert res["status"] == "denied"
    assert "Authorization failure" in res["error"]

def test_airflow_unreachable_returns_structured_error(mocker):
    mocker.patch("requests.get", side_effect=Exception("Airflow offline"))
    res = airflow_list_dags()
    assert res["status"] == "error"
    assert "Airflow service unavailable" in res["error"]
