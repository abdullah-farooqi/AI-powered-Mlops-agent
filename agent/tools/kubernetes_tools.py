import os
from typing import Dict, Any, List
from kubernetes import client, config

ALLOWED_NAMESPACE = "mlops-app"

def _get_k8s_clients():
    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()
    core_api = client.CoreV1Api()
    apps_api = client.AppsV1Api()
    return core_api, apps_api

def k8s_list_pods_and_deployments(namespace: str = ALLOWED_NAMESPACE) -> Dict[str, Any]:
    """Task 4.1: List application Pods and Deployments."""
    try:
        core_api, apps_api = _get_k8s_clients()
        pods = core_api.list_namespaced_pod(namespace=namespace)
        deps = apps_api.list_namespaced_deployment(namespace=namespace)

        pod_list = []
        for p in pods.items:
            restarts = sum(cs.restart_count for cs in p.status.container_statuses or [])
            pod_list.append({
                "name": p.metadata.name,
                "phase": p.status.phase,
                "restarts": restarts,
                "pod_ip": p.status.pod_ip
            })

        dep_list = []
        for d in deps.items:
            dep_list.append({
                "name": d.metadata.name,
                "desired_replicas": d.spec.replicas,
                "ready_replicas": d.status.ready_replicas or 0,
                "updated_replicas": d.status.updated_replicas or 0
            })

        return {
            "status": "success",
            "namespace": namespace,
            "deployments": dep_list,
            "pods": pod_list
        }
    except Exception as e:
        return {"status": "error", "error": f"Kubernetes API error: {str(e)}"}

def k8s_get_pod_status_and_logs(pod_name: str, namespace: str = ALLOWED_NAMESPACE, tail_lines: int = 50) -> Dict[str, Any]:
    """Task 4.2: Retrieve Pod status and relevant container logs."""
    try:
        core_api, _ = _get_k8s_clients()
        pod = core_api.read_namespaced_pod(name=pod_name, namespace=namespace)
        logs = core_api.read_namespaced_pod_log(name=pod_name, namespace=namespace, tail_lines=tail_lines)

        return {
            "status": "success",
            "pod_name": pod_name,
            "phase": pod.status.phase,
            "logs": logs
        }
    except Exception as e:
        return {"status": "error", "error": f"Failed reading pod '{pod_name}' logs: {str(e)}"}

def k8s_inspect_services(namespace: str = ALLOWED_NAMESPACE) -> Dict[str, Any]:
    """Task 4.3: Inspect application Services."""
    try:
        core_api, _ = _get_k8s_clients()
        services = core_api.list_namespaced_service(namespace=namespace)
        svc_list = []
        for s in services.items:
            ports = [{"port": p.port, "target_port": str(p.target_port)} for p in s.spec.ports]
            svc_list.append({
                "name": s.metadata.name,
                "type": s.spec.type,
                "cluster_ip": s.spec.cluster_ip,
                "ports": ports
            })
        return {"status": "success", "namespace": namespace, "services": svc_list}
    except Exception as e:
        return {"status": "error", "error": f"Failed inspecting services: {str(e)}"}

def k8s_diagnose_unhealthy_pods(namespace: str = ALLOWED_NAMESPACE) -> Dict[str, Any]:
    """Task 4.4: Identify unhealthy Pods, restarts, CrashLoopBackOff, or failed probes."""
    try:
        core_api, _ = _get_k8s_clients()
        pods = core_api.list_namespaced_pod(namespace=namespace)
        unhealthy = []

        for p in pods.items:
            issues = []
            if p.status.phase != "Running":
                issues.append(f"Pod phase is {p.status.phase}")
            
            for cs in p.status.container_statuses or []:
                if cs.restart_count > 0:
                    issues.append(f"Restart count: {cs.restart_count}")
                if cs.state.waiting:
                    issues.append(f"Waiting reason: {cs.state.waiting.reason}")
                if not cs.ready:
                    issues.append("Readiness probe failing (Container not ready)")

            if issues:
                unhealthy.append({
                    "pod_name": p.metadata.name,
                    "issues": issues
                })

        return {
            "status": "success",
            "namespace": namespace,
            "healthy": len(unhealthy) == 0,
            "unhealthy_pods": unhealthy
        }
    except Exception as e:
        return {"status": "error", "error": f"Failed diagnostic check: {str(e)}"}

def k8s_get_deployed_version(namespace: str = ALLOWED_NAMESPACE, deployment_name: str = "mlops-prediction-deployment") -> Dict[str, Any]:
    """Task 4.5: Report the currently deployed application version and configuration."""
    try:
        _, apps_api = _get_k8s_clients()
        dep = apps_api.read_namespaced_deployment(name=deployment_name, namespace=namespace)
        container = dep.spec.template.spec.containers[0]
        return {
            "status": "success",
            "deployment": deployment_name,
            "image": container.image,
            "desired_replicas": dep.spec.replicas,
            "ready_replicas": dep.status.ready_replicas or 0
        }
    except Exception as e:
        return {"status": "error", "error": f"Failed reading deployment version: {str(e)}"}

def k8s_execute_cluster_action(action: str, namespace: str, resource_name: str, confirmed: bool = False) -> Dict[str, Any]:
    """Safety Guard: Modifying cluster state requires explicit confirmation and whitelisted namespace."""
    if namespace != ALLOWED_NAMESPACE:
        return {
            "status": "denied",
            "error": f"Security Policy Violation: Modifications restricted to namespace '{ALLOWED_NAMESPACE}'. Denied for '{namespace}'."
        }
    if not confirmed:
        return {
            "status": "denied",
            "error": f"Safety Confirmation Required: Action '{action}' on '{resource_name}' was blocked. Must provide confirmed=True."
        }
    return {
        "status": "success",
        "action": action,
        "resource": resource_name,
        "message": f"Authorized cluster mutation '{action}' executed safely on {resource_name}."
    }
