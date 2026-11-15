from agent.tools.dvc_tools import (
    dvc_list_tracked_data,
    dvc_get_dataset_version,
    dvc_compare_versions,
    dvc_explain_reproduction,
    _run_safe_command
)

def test_dvc_list_tracked_data():
    res = dvc_list_tracked_data()
    assert res["status"] == "success"
    assert "tracked_files_count" in res
    assert "tracked_datasets" in res

def test_dvc_get_dataset_version_local():
    res = dvc_get_dataset_version("data/raw_data.csv")
    assert res["status"] == "success"
    assert "git_commit" in res
    assert "dvc_md5_hash" in res

def test_dvc_get_dataset_version_from_run(mocker):
    mocker.patch("agent.tools.mlflow_tools.mlflow_get_run_details", return_value={
        "status": "success",
        "tags": {
            "dvc_data_version": "data-v1",
            "dvc_data_hash": "a4c92e1deadbeef",
            "git_commit": "3b4575c3"
        }
    })
    res = dvc_get_dataset_version("data/raw_data.csv", run_id="exp_run_1")
    assert res["status"] == "success"
    assert res["dvc_version_tag"] == "data-v1"
    assert res["dvc_md5_hash"] == "a4c92e1deadbeef"

def test_dvc_compare_versions_mock(mocker):
    mocker.patch("agent.tools.dvc_tools._run_safe_command", side_effect=[
        {"status": "success", "stdout": "outs:\n- md5: hash_v1\n  size: 1000\n"},
        {"status": "success", "stdout": "outs:\n- md5: hash_v2\n  size: 1500\n"}
    ])
    res = dvc_compare_versions("data-v1", "data-v2", "data/raw_data.csv")
    assert res["status"] == "success"
    assert res["data_identical"] is False
    assert res["meta_a"]["md5"] == "hash_v1"
    assert res["meta_b"]["md5"] == "hash_v2"

def test_dvc_explain_reproduction():
    res = dvc_explain_reproduction("data-v1")
    assert res["status"] == "success"
    assert "safe_reproduction_command" in res
    assert "git checkout data-v1 && dvc checkout" in res["safe_reproduction_command"]

def test_dvc_command_injection_prevention():
    res = _run_safe_command("rm -rf /")
    assert res["status"] == "error"
    assert "Security violation" in res["error"]

def test_dvc_semicolon_injection_prevention():
    res = _run_safe_command("git status; cat /etc/passwd")
    assert res["status"] == "error"
    assert "Security violation" in res["error"]
