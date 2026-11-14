# AI-powered MLOps Agent

A Production-Ready AI Agent for MLOps and Machine Learning Infrastructure Automation.

## Overview
This repository implements an intelligent MLOps Agent that integrates with:
- **DVC**: Versioned dataset management & data provenance.
- **MLflow**: Experiment tracking, metrics, parameters, and candidate model selection.
- **Apache Airflow**: Training pipeline orchestration (validation, preprocessing, training, evaluation, quality gate).
- **Kubernetes**: Containerized FastAPI prediction service deployment, health diagnostics, and safe cluster management.

## Project Structure
```
assignment-03-mlops-agent/
├── agent/                  # AI Agent core & structured tools
│   ├── tools/              # DVC, MLflow, Airflow, and K8s tool implementations
│   ├── schemas.py          # Pydantic models for tools, state & responses
│   ├── orchestrator.py     # Multi-tool reasoning & evidence correlation engine
│   └── main.py             # Interactive CLI interface
├── api/                    # FastAPI inference service with /health & /ready probes
├── dags/                   # Apache Airflow training pipeline DAG
├── data/                   # Tracked dataset directory
├── training/               # Model training script with MLflow & DVC logging
├── kubernetes/             # ConfigMap, Deployment, and Service manifests
├── tests/                  # Pytest test suite (safety, mocking, diagnostics)
├── Dockerfile              # Container definition for prediction service
└── dvc.yaml                # DVC pipeline stages
```

## Setup & Quickstart

### 1. Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Automated Pytest Suite
```bash
pytest -v --cov=agent --cov=api tests/
```

### 3. Run Agent CLI
```bash
python -m agent.main
```
