# Demonstration Evidence & Screenshots Directory

This directory stores visual evidence and exported terminal logs for the 5 demonstration scenarios and 2 failure investigations required by the assignment defense rubric.

## Required Screenshots Checklist

| Scenario # | Title | Expected Visual Evidence | CLI / System Command |
|---|---|---|---|
| **Scenario 1** | **Dataset Provenance** | DVC hash, Git commit, and tracked file pointers | `dvc status` & `git log -1 -- data/raw_data.csv.dvc` |
| **Scenario 2** | **Model Selection** | MLflow UI showing multiple runs, parameters, and F1 metrics | Open `http://localhost:5000` (Runs comparison view) |
| **Scenario 3** | **Pipeline Execution** | Airflow UI showing DAG stages and task tree (all green) | Open `http://localhost:8080/dags/ml_training_pipeline/grid` |
| **Scenario 4** | **Deployment Diagnosis** | `kubectl get pods,svc,deploy -n mlops-app` and pod events | `kubectl get all -n mlops-app && kubectl describe pod -n mlops-app` |
| **Scenario 5** | **Multi-Tool Integration** | Full Agent CLI output for the mandatory inquiry | `python -m agent.main --query "Find the best recent model..."` |
| **Failure 1** | **K8s CrashLoopBackOff** | Agent detecting CrashLoopBackOff and failed probes | Injected deployment failure + `python -m agent.main` |
| **Failure 2** | **Airflow Gate Rejection** | Task `quality_gate` failing, blocking downstream promotion | Injected low F1 threshold + Airflow task failure log |
