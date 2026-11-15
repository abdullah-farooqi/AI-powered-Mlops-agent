from unittest.mock import MagicMock
from agent.tools.kubernetes_tools import (
    k8s_diagnose_unhealthy_pods,
    k8s_execute_cluster_action,
    k8s_list_pods_and_deployments,
)


def test_k8s_diagnose_unhealthy_pods_detects_crashloop_and_probe_failure(mocker):
    mock_pod = MagicMock()
    mock_pod.metadata.name = "mlops-prediction-crashloop"
    mock_pod.status.phase = "Running"

    mock_cs = MagicMock()
    mock_cs.restart_count = 5
    mock_cs.ready = False
    mock_cs.state.running = True
    mock_cs.state.waiting.reason = "CrashLoopBackOff"
    mock_cs.state.waiting.message = "Back-off 5m0s restarting failed container"
    mock_cs.last_state.terminated.reason = "OOMKilled"
    mock_cs.last_state.terminated.exit_code = 137
    mock_pod.status.container_statuses = [mock_cs]

    mock_core = MagicMock()
    mock_core.list_namespaced_pod.return_value.items = [mock_pod]
    mock_core.list_namespaced_event.return_value.items = []

    mocker.patch("agent.tools.kubernetes_tools._get_k8s_clients", return_value=(mock_core, MagicMock()))

    res = k8s_diagnose_unhealthy_pods("mlops-app")
    assert res["status"] == "success"
    assert res["healthy"] is False
    assert len(res["unhealthy_pods"]) == 1
    issues_str = str(res["unhealthy_pods"][0]["issues"])
    assert "CrashLoopBackOff" in issues_str
    assert "OOMKilled" in issues_str


def test_k8s_mutation_requires_confirmation():
    res = k8s_execute_cluster_action("restart", "mlops-app", confirmed=False)
    assert res["status"] == "needs_confirmation"
    assert "requires explicit human confirmation" in res["message"]


def test_k8s_mutation_executes_real_restart_when_confirmed(mocker):
    mock_apps = MagicMock()
    mocker.patch("agent.tools.kubernetes_tools._get_k8s_clients", return_value=(MagicMock(), mock_apps))

    res = k8s_execute_cluster_action("restart", "mlops-app", confirmed=True)
    assert res["status"] == "success"
    assert res["action"] == "restart"
    assert mock_apps.patch_namespaced_deployment.called


def test_k8s_mutation_rejects_unauthorized_action():
    res = k8s_execute_cluster_action("delete_deployment", "mlops-app", confirmed=True)
    assert res["status"] == "denied"
    assert "Unauthorized action" in res["error"]


def test_k8s_read_tool_rejects_unauthorized_namespace():
    res = k8s_list_pods_and_deployments(namespace="kube-system")
    assert res["status"] == "denied"
    assert "Security Policy Violation" in res["error"]
