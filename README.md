# AI-powered MLOps Agent

A Production-Ready AI Agent for MLOps and Machine Learning Infrastructure Automation integrating DVC, MLflow, Apache Airflow, and Kubernetes.

---

## 📌 Overview

This repository implements an intelligent MLOps AI Agent that integrates four foundational pillars of modern machine learning operations:
1. **DVC (Data Version Control)**: Dataset versioning, lineage tracking, and Git-backed commit provenance.
2. **MLflow**: Experiment tracking, candidate model ranking, parameter/metric logging, and model registry governance.
3. **Apache Airflow**: Directed Acyclic Graph (DAG) pipeline orchestration with multi-stage evaluation and automated quality gates.
4. **Kubernetes**: Containerized FastAPI prediction service deployment (`mlops-app` namespace), health/readiness probe diagnostics, crash loop analysis, and human-in-the-loop safe cluster mutation.

---

## 📂 Project Structure

```
assignment-03-mlops-agent/
├── .github/workflows/ci.yml # GitHub Actions automated CI workflow (lint + test)
├── .env.example            # Environment variable configuration template
├── .flake8                 # Flake8 linting rules (max-line-length = 120)
├── agent/                  # AI Agent core & structured tools
│   ├── tools/
│   │   ├── airflow_tools.py    # Airflow REST API integration & DAG execution
│   │   ├── dvc_tools.py        # DVC dataset versioning & provenance tools
│   │   ├── kubernetes_tools.py # K8s diagnostics, pod logs, events & mutations
│   │   └── mlflow_tools.py     # MLflow run comparisons, best run & registry tools
│   ├── main.py             # Interactive Rich CLI interface
│   ├── orchestrator.py     # Multi-tool reasoning & evidence correlation engine
│   └── schemas.py          # Pydantic schemas and LLM function-calling declarations
├── api/                    # Production FastAPI serving service
│   ├── main.py             # REST API endpoints (/health, /ready, /predict)
│   └── test_api.py         # Test client for FastAPI service
├── dags/
│   └── training_pipeline.py# Airflow 3 compatible DAG with quality gate
├── data/
│   ├── raw_data.csv        # Iris dataset (150 samples)
│   └── raw_data.csv.dvc    # DVC tracking descriptor
├── docker-compose.yaml     # Spin up MLflow tracking server and Airflow webserver
├── Dockerfile              # Multi-stage production container build for API
├── kubernetes/             # Kubernetes manifests
│   ├── namespace.yaml      # Dedicated 'mlops-app' namespace
│   ├── configmap.yaml      # Application environment configuration
│   ├── deployment.yaml     # 2-replica deployment with liveness & readiness probes
│   └── service.yaml        # ClusterIP service routing port 80 -> 8000
├── screenshots/            # Demonstration screenshots checklist and defense evidence
├── tests/                  # 26 automated unit & safety tests (100% green)
│   ├── test_agent_orchestrator.py
│   ├── test_airflow_tools.py
│   ├── test_dvc_tools.py
│   ├── test_kubernetes_tools.py
│   ├── test_mlflow_tools.py
│   └── test_safety_and_failures.py
├── training/
│   └── train.py            # Iris classifier training script with MLflow logging
├── architecture.md         # Comprehensive technical architecture & report
└── requirements.txt        # Pinned production and development dependencies
```

---

## ⚙️ Prerequisites & Setup

### 1. Python Environment Setup
Create and activate a Python 3.10+ virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy the template configuration file:
```bash
cp .env.example .env
```
Default parameters in `.env.example`:
- `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`
- `AIRFLOW_BASE_URL=http://127.0.0.1:8080`
- `AIRFLOW_USERNAME=admin`
- `AIRFLOW_PASSWORD=admin`
- `K8S_NAMESPACE=mlops-app`

---

## 🚀 Service Orchestration

### Option A: Spin Up Infrastructure via Docker Compose
To launch MLflow and Apache Airflow simultaneously:
```bash
docker compose up -d
```
- **MLflow Tracking UI**: [http://127.0.0.1:5000](http://127.0.0.1:5000)
- **Airflow Webserver**: [http://127.0.0.1:8080](http://127.0.0.1:8080)

### Option B: Standalone MLflow Server
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

---

## 📊 Data Management & Model Training

### 1. DVC Data Tracking
Track data changes with DVC:
```bash
dvc add data/raw_data.csv
git add data/raw_data.csv.dvc .gitignore
git commit -m "chore(data): track updated raw dataset with DVC"
```

### 2. Model Training with MLflow & DVC Lineage
Run the Iris classification training script:
```bash
python training/train.py --n-estimators 150 --max-depth 4 --register
```
This logs:
- Hyperparameters (`n_estimators`, `max_depth`, `criterion`)
- Metrics (`accuracy`, `f1_score`, `precision`, `recall`)
- Provenance tags: `dataset_version_md5` and `git_commit`
- Artifacts: serialized model registered under `iris_classifier`

---

## 🚢 Kubernetes Deployment

### 1. Build Prediction API Container Image
```bash
docker build -t mlops-prediction-api:1.0.0 .
```

### 2. Deploy to Minikube / Local Cluster
Ensure your Kubernetes cluster is running (`minikube status` or `kubectl cluster-info`):
```bash
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

Verify deployment status:
```bash
kubectl get pods -n mlops-app
kubectl get svc -n mlops-app
```

---

## 🤖 Running the AI Agent

### 1. Interactive CLI Mode
Launch the rich interactive CLI:
```bash
python -m agent.main
```

### 2. Single Query Mode
Run standard multi-tool diagnostic queries directly:
```bash
# General operational query (DVC + MLflow + K8s + Airflow correlation)
python -m agent.main --query "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."

# Model comparison query
python -m agent.main --query "Compare the last 3 training runs and rank them by validation accuracy."

# Cluster diagnostics query
python -m agent.main --query "Diagnose any unhealthy pods or crash loop issues in the cluster."
```

### 3. Human-in-the-Loop Safe Execution
Mutating actions (such as restarting pods or triggering Airflow DAGs) require confirmation. When running queries that propose actions:
```bash
# Dry-run / recommendation:
python -m agent.main --query "Restart the prediction service deployment."

# Explicitly authorized execution:
python -m agent.main --query "Restart the prediction service deployment." --confirm
```

---

## 🧪 Testing & Code Quality

### 1. Code Quality & Linting
Enforce PEP8 compliance with zero warnings:
```bash
flake8 agent/ api/ training/ tests/
```

### 2. Pytest Automated Test Suite
Run the comprehensive 26-test suite:
```bash
pytest -v tests/
```

With coverage report:
```bash
pytest -v --cov=agent --cov=api --cov=training tests/
```

---

## 🛡️ Safety & Reliability Principles

1. **Zero Hallucination Guard**: The agent never fabricates deployment statuses, metric values, or DAG outcomes. If external infrastructure is offline, the agent explicitly marks the state as `UNKNOWN / INCONCLUSIVE` and lists the unavailable dependencies.
2. **Retraining Guardrails**: Retraining is never initiated automatically from casual user queries. It requires empirical justification (data drift, pipeline quality gate failure, or metric degradation) plus explicit human confirmation (`confirmed=True`).
3. **Namespace Isolation**: All Kubernetes inspection and mutation tools strictly validate operations against the authorized `mlops-app` namespace, blocking unauthorized cluster-wide actions.
4. **Structured Redaction**: Sensitive tokens, passwords, and private connection strings are scrubbed before reaching agent logs.
