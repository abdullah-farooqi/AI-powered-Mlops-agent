from unittest.mock import MagicMock
from agent.tools.kubernetes_tools import (
    k8s_diagnose_unhealthy_pods,
    k8s_execute_cluster_action
)

def test_k8s_diagnose_unhealthy_pods_detects_crashloop(mocker):
    mock_pod = MagicMock()
    mock_pod.metadata.name = "mlops-prediction-crashloop"
    mock_pod.status.phase = "Running"

    mock_cs = MagicMock()
    mock_cs.restart_count = 5
    mock_cs.ready = False
    mock_cs.state.waiting.reason = "CrashLoopBackOff"
    mock_pod.status.container_statuses = [mock_cs]

    mock_core = MagicMock()
    mock_core.list_namespaced_pod.return_value.items = [mock_pod]

    mocker.patch("agent.tools.kubernetes_tools._get_k8s_clients", return_value=(mock_core, MagicMock()))

    res = k8s_diagnose_unhealthy_pods("mlops-app")
    assert res["status"] == "success"
    assert res["healthy"] is False
    assert len(res["unhealthy_pods"]) == 1
    assert "CrashLoopBackOff" in str(res["unhealthy_pods"][0]["issues"])

def test_k8s_cluster_mutation_requires_confirmation():
    res = k8s_execute_cluster_action("restart", "mlops-app", "mlops-prediction-deployment", confirmed=False)
    assert res["status"] == "denied"
    assert "Safety Confirmation Required" in res["error"]

def test_k8s_cluster_mutation_rejects_unauthorized_namespace():
    res = k8s_execute_cluster_action("restart", "kube-system", "core-dns", confirmed=True)
    assert res["status"] == "denied"
    assert "Security Policy Violation" in res["error"]
