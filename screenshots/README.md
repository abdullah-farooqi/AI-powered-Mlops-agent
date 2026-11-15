# Demonstration Evidence & Screenshots Guide

This guide details the exact screenshots, visual proof, and terminal logs required for the project report and defense submission.

---

## 📸 Screenshots Master Table

| Filename | Scenario / Stage | Component | What to Capture (Visual Evidence) | Generation Command |
|---|---|---|---|---|
| `01_dvc_provenance.png` | **Scenario 1** | **DVC & Git** | Terminal showing `dvc status` (clean), `git tag` (`data-v1`, `data-v2`), and `git log -1 -- data/raw_data.csv.dvc` showing MD5 `4224576f` and commit hash. | `git tag && git log -1 -- data/raw_data.csv.dvc && dvc status` |
| `02_mlflow_runs.png` | **Scenario 2** | **MLflow UI** | Browser view of MLflow Tracking UI (`http://localhost:5000`) showing multiple runs, hyperparameter columns (`n_estimators`, `max_depth`), metrics (`accuracy`, `f1_score`), and registered model `iris_classifier`. | `python training/train.py --n-estimators 50 --max-depth 4 --register` |
| `03_airflow_dag_grid.png` | **Scenario 3** | **Airflow UI** | Browser grid view of Airflow (`http://localhost:8080/dags/ml_training_pipeline/grid`) showing the 6 stages (`data_validation` $\rightarrow$ `data_preprocessing` $\rightarrow$ `model_training` $\rightarrow$ `model_evaluation` $\rightarrow$ `quality_gate` $\rightarrow$ `model_promotion`) all green. | Trigger from Agent CLI or Airflow UI |
| `04_k8s_pods_services.png` | **Scenario 4** | **Kubernetes** | Terminal showing running pods, cluster service, deployment with 2/2 ready replicas in namespace `mlops-app`. | `kubectl get pods,svc,deploy -n mlops-app -o wide` |
| `05_agent_cli_mandatory.png` | **Scenario 5** | **Agent CLI** | Terminal screenshot of Rich-formatted table and reasoning for the mandatory multi-tool inquiry. | `python -m agent.main --query "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."` |
| `06_k8s_failure_crashloop.png` | **Failure 1** | **K8s Diagnosis** | Terminal showing agent diagnosing `CrashLoopBackOff` and failed probes when the bad manifest is applied. | `bash scripts/failures/inject_k8s_crashloop.sh && python -m agent.main --query "Diagnose any unhealthy pods or crash loop issues in the cluster."` |
| `07_airflow_failure_gate.png` | **Failure 2** | **Airflow Quality Gate** | Airflow UI task log or Agent CLI showing task `quality_gate` failed with `ValueError: Quality Gate Failed! Model F1 score (0.42) is below threshold (0.75)`. | `python scripts/failures/inject_airflow_quality_gate_failure.py` |

---

## 🛠️ Step-by-Step Instructions to Generate Screenshots

### Step 1: Start Services
```bash
# Start MLflow and Airflow
docker compose up -d

# Start local Kubernetes (Minikube or Kind)
minikube start

# Deploy Kubernetes manifests
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

### Step 2: Capture DVC & Git Provenance (`01_dvc_provenance.png`)
```bash
git tag
git show data-v1:data/raw_data.csv.dvc
git show data-v2:data/raw_data.csv.dvc
dvc status
```
*Screenshot the terminal displaying the two tagged versions, their MD5 checksums (`c9156e0d` vs `4224576f`), and clean DVC status.*

### Step 3: Capture MLflow Experiments (`02_mlflow_runs.png`)
```bash
# Generate 3 comparison runs with different parameters
python training/train.py --n-estimators 20 --max-depth 2
python training/train.py --n-estimators 50 --max-depth 4
python training/train.py --n-estimators 100 --max-depth 5
```
*Open `http://localhost:5000` in your web browser. Take a screenshot showing the experiments table and run metrics comparison.*

### Step 4: Capture Airflow DAG Execution (`03_airflow_dag_grid.png`)
*Open `http://localhost:8080` in your web browser (login: admin/admin). Navigate to `ml_training_pipeline` DAG grid view. Take a screenshot showing all 6 stages passing green.*

### Step 5: Capture Kubernetes Pods & Services (`04_k8s_pods_services.png`)
```bash
kubectl get all -n mlops-app
```
*Screenshot the terminal showing the pods with status `Running (Ready: 1/1)`, Service ClusterIP port 80 -> 8000, and deployment `2/2 ready`.*

### Step 6: Capture Mandatory Agent Inquiry (`05_agent_cli_mandatory.png`)
```bash
python -m agent.main --query "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
```
*Screenshot the Rich table showing all 5 executed tools, durations, facts, and retraining evaluation.*

### Step 7: Capture Injected Failure 1 (`06_k8s_failure_crashloop.png`)
```bash
# Inject failure
bash scripts/failures/inject_k8s_crashloop.sh

# Run agent diagnosis
python -m agent.main --query "Diagnose any unhealthy pods or crash loop issues in the cluster."

# Restore healthy state
bash scripts/failures/fix_k8s_crashloop.sh
```
*Screenshot the agent output highlighting `CrashLoopBackOff`, termination exit code, and failed liveness probes.*

### Step 8: Capture Injected Failure 2 (`07_airflow_failure_gate.png`)
```bash
# Inject degraded run
python scripts/failures/inject_airflow_quality_gate_failure.py

# Check failed tasks
python -m agent.main --query "Show me the failed Airflow tasks and their logs"

# Restore high-performing run
python scripts/failures/fix_airflow_quality_gate.py
```
*Screenshot the terminal showing task `quality_gate` failed and promotion safely blocked.*
