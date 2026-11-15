#!/bin/bash
# -------------------------------------------------------------
# Fix Scenario 1: Restore Healthy Kubernetes Deployment
# -------------------------------------------------------------
set -e

echo "[RECOVERY] Re-applying production Deployment with health probes..."
kubectl apply -f kubernetes/deployment.yaml

echo "[RECOVERY] Waiting for healthy rollout in namespace 'mlops-app'..."
kubectl rollout status deployment/mlops-prediction-deployment -n mlops-app --timeout=60s
kubectl get pods -n mlops-app
echo "[RECOVERY] Deployment restored successfully."
