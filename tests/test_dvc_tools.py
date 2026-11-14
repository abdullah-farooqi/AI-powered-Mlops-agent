from agent.tools.dvc_tools import (
    dvc_list_tracked_data,
    dvc_get_dataset_version,
    dvc_detect_changes,
    dvc_explain_reproduction,
    _run_command
)

def test_dvc_list_tracked_data():
    res = dvc_list_tracked_data()
    assert res["status"] == "success"
    assert "tracked_files_count" in res
    assert "has_dvc_yaml" in res

def test_dvc_get_dataset_version():
    res = dvc_get_dataset_version("data/raw_data.csv")
    assert res["status"] == "success"
    assert "git_commit" in res
    assert "dvc_md5_hash" in res

def test_dvc_detect_changes():
    res = dvc_detect_changes()
    assert res["status"] == "success"
    assert "data_has_changed" in res

def test_dvc_explain_reproduction():
    res = dvc_explain_reproduction("v1.0")
    assert res["status"] == "success"
    assert "safe_checkout_command" in res
    assert "git checkout v1.0 && dvc pull" in res["safe_checkout_command"]

def test_dvc_command_injection_prevention():
    res = _run_command("rm -rf /")
    assert res["status"] == "error"
    assert "Security violation" in res["error"]
