import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from kubernetes import client, config

ALLOWED_NAMESPACE = os.getenv("ALLOWED_NAMESPACE", "mlops-app")
ALLOWED_ACTIONS = {"restart", "scale"}


def _get_k8s_clients():
    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()
    core_api = client.CoreV1Api()
    apps_api = client.AppsV1Api()
    return core_api, apps_api


def _check_namespace(namespace: str) -> Optional[Dict[str, Any]]:
    """Enforce approved namespace constraint on all operations per assignment specification."""
    if namespace != ALLOWED_NAMESPACE:
        return {
            "status": "denied",
            "error": (
                f"Security Policy Violation: Operations are strictly restricted to "
                f"approved namespace '{ALLOWED_NAMESPACE}'. Denied for namespace '{namespace}'."
            ),
        }
    return None


def k8s_list_pods_and_deployments(namespace: str = ALLOWED_NAMESPACE) -> Dict[str, Any]:
    """Task 4.1: List application Pods and Deployments in the approved namespace."""
    ns_error = _check_namespace(namespace)
    if ns_error:
        return ns_error

    try:
        core_api, apps_api = _get_k8s_clients()
        pods = core_api.list_namespaced_pod(namespace=namespace, _request_timeout=5)
        deps = apps_api.list_namespaced_deployment(namespace=namespace, _request_timeout=5)

        pod_list = []
        for p in pods.items:
            restarts = sum(cs.restart_count for cs in p.status.container_statuses or [])
            ready = all(cs.ready for cs in p.status.container_statuses or []) if p.status.container_statuses else False
            pod_list.append(
                {
                    "name": p.metadata.name,
                    "phase": p.status.phase,
                    "ready": ready,
                    "restarts": restarts,
                    "pod_ip": p.status.pod_ip,
                }
            )

        dep_list = []
        for d in deps.items:
            container = d.spec.template.spec.containers[0]
            dep_list.append(
                {
                    "name": d.metadata.name,
                    "image": container.image,
                    "desired_replicas": d.spec.replicas,
                    "ready_replicas": d.status.ready_replicas or 0,
                    "updated_replicas": d.status.updated_replicas or 0,
                }
            )

        return {
            "status": "success",
            "namespace": namespace,
            "deployments_count": len(dep_list),
            "deployments": dep_list,
            "pods_count": len(pod_list),
            "pods": pod_list,
        }
    except Exception as e:
        return {"status": "error", "error": f"Kubernetes API error: {str(e)}"}


def k8s_get_pod_status_and_logs(
    pod_name: str, namespace: str = ALLOWED_NAMESPACE, tail_lines: int = 50
) -> Dict[str, Any]:
    """Task 4.2: Retrieve detailed Pod status and recent container stdout/stderr logs."""
    ns_error = _check_namespace(namespace)
    if ns_error:
        return ns_error

    try:
        core_api, _ = _get_k8s_clients()
        pod = core_api.read_namespaced_pod(name=pod_name, namespace=namespace, _request_timeout=5)
        logs = core_api.read_namespaced_pod_log(
            name=pod_name, namespace=namespace, tail_lines=tail_lines, _request_timeout=5
        )

        return {
            "status": "success",
            "pod_name": pod_name,
            "phase": pod.status.phase,
            "recent_logs": logs if logs else "No logs recorded for this container.",
        }
    except Exception as e:
        return {"status": "error", "error": f"Failed reading pod '{pod_name}' logs: {str(e)}"}


def k8s_inspect_services(namespace: str = ALLOWED_NAMESPACE) -> Dict[str, Any]:
    """Task 4.3: Inspect application Services, ClusterIP, ports, and endpoints."""
    ns_error = _check_namespace(namespace)
    if ns_error:
        return ns_error

    try:
        core_api, _ = _get_k8s_clients()
        services = core_api.list_namespaced_service(namespace=namespace, _request_timeout=5)
        svc_list = []
        for s in services.items:
            ports = [{"port": p.port, "target_port": str(p.target_port), "protocol": p.protocol} for p in s.spec.ports]
            svc_list.append(
                {
                    "name": s.metadata.name,
                    "type": s.spec.type,
                    "cluster_ip": s.spec.cluster_ip,
                    "ports": ports,
                    "selector": s.spec.selector,
                }
            )
        return {"status": "success", "namespace": namespace, "services": svc_list}
    except Exception as e:
        return {"status": "error", "error": f"Failed inspecting services: {str(e)}"}


def k8s_list_events(namespace: str = ALLOWED_NAMESPACE, warning_only: bool = True) -> Dict[str, Any]:
    """Task 4.4a: Retrieve Kubernetes cluster events for diagnosis."""
    ns_error = _check_namespace(namespace)
    if ns_error:
        return ns_error

    try:
        core_api, _ = _get_k8s_clients()
        events = core_api.list_namespaced_event(namespace=namespace, _request_timeout=5)
        extracted = []
        for e in events.items:
            if warning_only and e.type != "Warning":
                continue
            extracted.append(
                {
                    "type": e.type,
                    "reason": e.reason,
                    "message": e.message,
                    "object": f"{e.involved_object.kind}/{e.involved_object.name}",
                    "count": e.count,
                    "last_timestamp": str(e.last_timestamp),
                }
            )
        return {
            "status": "success",
            "namespace": namespace,
            "warning_events_count": len(extracted),
            "events": extracted,
        }
    except Exception as e:
        return {"status": "error", "error": f"Failed listing cluster events: {str(e)}"}


def k8s_diagnose_unhealthy_pods(namespace: str = ALLOWED_NAMESPACE) -> Dict[str, Any]:
    """
    Task 4.4b: Deep diagnostic scan of application Pods.
    Inspects CrashLoopBackOff, OOMKilled, exit codes, failed probes, and recent restart events.
    """
    ns_error = _check_namespace(namespace)
    if ns_error:
        return ns_error

    try:
        core_api, _ = _get_k8s_clients()
        pods = core_api.list_namespaced_pod(namespace=namespace, _request_timeout=5)
        unhealthy = []

        for p in pods.items:
            issues = []
            if p.status.phase != "Running":
                issues.append(f"Pod phase is '{p.status.phase}'")

            for cs in p.status.container_statuses or []:
                if cs.restart_count > 0:
                    issues.append(f"Restart count: {cs.restart_count}")

                # Check waiting state (CrashLoopBackOff, ImagePullBackOff, ErrImagePull)
                if cs.state.waiting:
                    issues.append(f"Waiting reason: '{cs.state.waiting.reason}' (message: {cs.state.waiting.message})")

                # Check last termination state (e.g. OOMKilled, error exit code)
                if cs.last_state.terminated:
                    term = cs.last_state.terminated
                    issues.append(f"Last terminated with reason='{term.reason}', exit_code={term.exit_code}")

                # Probe failure
                if not cs.ready and cs.state.running:
                    issues.append("Readiness probe failing (Container process is running but not serving traffic)")

            if issues:
                unhealthy.append({"pod_name": p.metadata.name, "issues": issues})

        # Also pull warning events for context
        events_res = k8s_list_events(namespace=namespace, warning_only=True)
        warning_events = events_res.get("events", []) if events_res.get("status") == "success" else []

        return {
            "status": "success",
            "namespace": namespace,
            "healthy": len(unhealthy) == 0,
            "unhealthy_pods_count": len(unhealthy),
            "unhealthy_pods": unhealthy,
            "relevant_warning_events": warning_events[:5],
        }
    except Exception as e:
        return {"status": "error", "error": f"Failed diagnostic check: {str(e)}"}


def k8s_get_deployed_version(
    namespace: str = ALLOWED_NAMESPACE, deployment_name: str = "mlops-prediction-deployment"
) -> Dict[str, Any]:
    """Task 4.5: Report currently deployed application version, image tag, and ConfigMap configuration."""
    ns_error = _check_namespace(namespace)
    if ns_error:
        return ns_error

    try:
        core_api, apps_api = _get_k8s_clients()
        dep = apps_api.read_namespaced_deployment(name=deployment_name, namespace=namespace, _request_timeout=5)
        container = dep.spec.template.spec.containers[0]

        # Read configmap for APP_VERSION & MODEL_VERSION
        config_data = {}
        try:
            cm = core_api.read_namespaced_config_map(
                name="mlops-prediction-config", namespace=namespace, _request_timeout=5
            )
            config_data = cm.data or {}
        except Exception:
            pass

        return {
            "status": "success",
            "deployment": deployment_name,
            "image": container.image,
            "image_tag": container.image.split(":")[-1] if ":" in container.image else "latest",
            "desired_replicas": dep.spec.replicas,
            "ready_replicas": dep.status.ready_replicas or 0,
            "configmap_app_version": config_data.get("APP_VERSION", "unknown"),
            "configmap_model_version": config_data.get("MODEL_VERSION", "unknown"),
            "environment": config_data.get("ENVIRONMENT", "production"),
        }
    except Exception as e:
        return {"status": "error", "error": f"Failed reading deployment version: {str(e)}"}


def k8s_execute_cluster_action(
    action: str,
    namespace: str = ALLOWED_NAMESPACE,
    deployment_name: str = "mlops-prediction-deployment",
    replicas: Optional[int] = None,
    confirmed: bool = False,
) -> Dict[str, Any]:
    """
    Safety Guarded Cluster Mutation Tool:
    Performs real cluster modifications (rollout restart or replica scaling) via Kubernetes API.
    STRICT SAFETY CONTROLS:
    1. Restricted strictly to approved namespace 'mlops-app'.
    2. Restricted to approved action list {'restart', 'scale'}.
    3. Requires explicit confirmation (confirmed=True) before executing mutation.
    """
    ns_error = _check_namespace(namespace)
    if ns_error:
        return ns_error

    action_lower = action.lower()
    if action_lower not in ALLOWED_ACTIONS:
        return {
            "status": "denied",
            "error": f"Unauthorized action '{action}'. Permitted cluster actions: {ALLOWED_ACTIONS}",
        }

    if not confirmed:
        return {
            "status": "needs_confirmation",
            "action": action,
            "deployment": deployment_name,
            "namespace": namespace,
            "message": (
                f"Cluster mutation '{action}' on '{deployment_name}' requires "
                "explicit human confirmation. Confirm to execute."
            ),
        }

    try:
        _, apps_api = _get_k8s_clients()

        if action_lower == "restart":
            # Real rollout restart via patching deployment annotations with restart timestamp
            now_iso = datetime.now(timezone.utc).isoformat()
            patch_body = {
                "spec": {"template": {"metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": now_iso}}}}
            }
            apps_api.patch_namespaced_deployment(
                name=deployment_name, namespace=namespace, body=patch_body, _request_timeout=10
            )
            return {
                "status": "success",
                "action": "restart",
                "deployment": deployment_name,
                "restarted_at": now_iso,
                "message": f"Successfully initiated rolling restart for deployment '{deployment_name}'.",
            }

        elif action_lower == "scale":
            target_replicas = replicas if replicas is not None else 2
            patch_body = {"spec": {"replicas": target_replicas}}
            apps_api.patch_namespaced_deployment(
                name=deployment_name, namespace=namespace, body=patch_body, _request_timeout=10
            )
            return {
                "status": "success",
                "action": "scale",
                "deployment": deployment_name,
                "replicas": target_replicas,
                "message": f"Successfully scaled deployment '{deployment_name}' to {target_replicas} replicas.",
            }

    except Exception as e:
        return {"status": "error", "error": f"Kubernetes mutation execution failed: {str(e)}"}
