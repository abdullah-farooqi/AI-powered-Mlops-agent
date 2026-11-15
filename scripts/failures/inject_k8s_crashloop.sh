#!/bin/bash
# -------------------------------------------------------------
# Failure Scenario 1: Inject Kubernetes CrashLoopBackOff Failure
# -------------------------------------------------------------
set -e

echo "[INJECTION] Deploying faulty container into namespace 'mlops-app'..."
kubectl apply -f kubernetes/failure_pod_crash.yaml

echo "[INJECTION] Waiting for pod failure state to trigger..."
sleep 5
kubectl get pods -n mlops-app
echo "[INJECTION] Pod is entering CrashLoopBackOff. Run agent diagnosis now:"
echo "python -m agent.main --query 'Diagnose any unhealthy pods or crash loop issues in the cluster.'"
