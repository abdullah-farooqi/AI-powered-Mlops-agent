# MLOps AI Agent — System Architecture & Technical Report
**Course:** MLOps & Production AI Engineering | Fall 2026  
**Assignment:** Assignment No. 3 — Building a Production-Ready MLOps AI Agent  
**Student:** Abdullah Ahmad (i23-3006)  

---

## 1. Executive Summary & System Architecture

The **MLOps AI Agent** is a production-grade system designed to automate the operational duties of a machine learning engineer across four foundational layers:
1. **Dataset Management & Provenance Layer (DVC & Git)**: Guarantees reproducible dataset versioning, MD5 checksum integrity, and historical provenance tracking linked to Git commits.
2. **Experiment Intelligence Layer (MLflow)**: Queries the MLflow Tracking Server and Model Registry via the official Python `MlflowClient` API to evaluate metrics, compare hyperparameter runs, and recommend production candidates.
3. **Pipeline Orchestration Layer (Apache Airflow)**: Manages a modern multi-stage training pipeline (validation $\rightarrow$ preprocessing $\rightarrow$ training $\rightarrow$ evaluation $\rightarrow$ quality gate $\rightarrow$ promotion) with retry logic, XCom metric passing, and promotion blocking.
4. **Cloud-Native Serving & Infrastructure Layer (Kubernetes)**: Observes containerized FastAPI inference microservices in the `mlops-app` namespace, checks liveness/readiness probes, inspects warning events, and executes authorized cluster actions (restart, scale).

```mermaid
flowchart TD
    User([User Request / CLI]) --> Orchestrator[MLOps Agent Orchestrator]

    subgraph Security_Gate [Security & Safety Enforcement]
        Orchestrator --> AuthCheck{Authorized Namespace & Action?}
        AuthCheck -->|Approved & Confirmed| ExecLayer[Execution Engine]
        AuthCheck -->|Mutating Action Unconfirmed| ConfirmGate[Human-in-the-Loop Confirmation Gate]
        ConfirmGate -->|User Confirms [yes]| ExecLayer
        ConfirmGate -->|User Rejects [no]| Abort[Abort Mutation Safely]
    end

    subgraph Tools_Layer [Structured Tools Layer]
        ExecLayer --> DVC_Tools[DVC & Git Tools]
        ExecLayer --> MLflow_Tools[MLflow Tracking Tools]
        ExecLayer --> Airflow_Tools[Airflow REST API Tools]
        ExecLayer --> K8s_Tools[Kubernetes Operations Tools]
    end

    subgraph Infrastructure [Target Operational Systems]
        DVC_Tools --> DVC_Store[(DVC Remote Storage & .dvc Pointers)]
        MLflow_Tools --> MLflow_Server[(MLflow Server & Model Registry)]
        Airflow_Tools --> Airflow_Cluster[(Airflow Scheduler & Workers)]
        K8s_Tools --> K8s_Cluster[(Kubernetes Cluster: Namespace mlops-app)]
    end

    subgraph Synthesis [Evidence-Based Correlation & Zero Hallucination]
        ExecLayer --> Collector[Evidence Aggregator]
        Collector --> Facts[Known Facts - Empirical Proof]
        Collector --> Warnings[Uncertainties & Offline Service Alerts]
        Collector --> RetrainingRules[Deterministic Retraining Evaluator]
        RetrainingRules --> Output[Structured JSON / CLI Presentation]
    end
```

---

## 2. Tool Definitions, Permissions & Security Whitelists

Every tool operates under the **Principle of Least Privilege**:

| Module | Callable Tool Name | Method / Implementation | Security & Permission Scope |
|---|---|---|---|
| **DVC** | `dvc_list_tracked_data` | Directory AST inspection | **Read-Only**: Scans repository for `.dvc` files and `dvc.yaml`. |
| **DVC** | `dvc_get_dataset_version` | Git log / MLflow tag reader | **Read-Only**: Retrieves authentic Git commit and recorded DVC MD5. |
| **DVC** | `dvc_detect_changes` | Subprocess `dvc status` | **Read-Only**: Detects modified data. Strict argument whitelist (`shlex`). |
| **DVC** | `dvc_compare_versions` | `git show <ref>:<file>.dvc` | **Read-Only**: Compares MD5 checksums across two Git refs/tags. |
| **DVC** | `dvc_explain_reproduction` | Guidance generator | **Read-Only**: Outputs exact checkout syntax (`git checkout <tag> && dvc checkout`). |
| **MLflow** | `mlflow_list_experiments_and_runs` | `MlflowClient.search_runs` | **Read-Only**: Queries active runs and parameters. |
| **MLflow** | `mlflow_get_run_details` | `MlflowClient.get_run` | **Read-Only**: Pulls metrics, artifacts, and tags (`dvc_data_hash`). |
| **MLflow** | `mlflow_get_best_run` | `MlflowClient` search & sort | **Read-Only**: Filters `FINISHED` runs, enforces secondary accuracy gates. |
| **MLflow** | `mlflow_compare_runs` | `MlflowClient` multi-run scan | **Read-Only**: Ranks last $N$ runs and generates defensible recommendation. |
| **MLflow** | `mlflow_get_registered_model_info` | `MlflowClient.get_latest_versions` | **Read-Only**: Inspects model stages in Model Registry. |
| **Airflow**| `airflow_list_dags` | Airflow REST API (`GET /dags`) | **Read-Only**: Lists active/paused workflows. |
| **Airflow**| `airflow_get_dag_run_status` | Airflow REST API (`GET /dagRuns`) | **Read-Only**: Inspects latest run state, logical date, and timestamps. |
| **Airflow**| `airflow_trigger_dag` | Airflow REST API (`POST /dagRuns`) | **Mutating**: Whitelist restricted (`ml_training_pipeline`). Requires `confirmed=True`. |
| **Airflow**| `airflow_get_failed_tasks` | Airflow REST API (`GET /taskInstances`) | **Read-Only**: Isolates failed stages with retry try counts. |
| **Airflow**| `airflow_get_task_logs` | Airflow REST API (`GET /logs`) | **Read-Only**: Extracts stdout/stderr logs of failed tasks. |
| **Kubernetes** | `k8s_list_pods_and_deployments` | `CoreV1Api` & `AppsV1Api` | **Read-Only**: Restricted to namespace `mlops-app`. |
| **Kubernetes** | `k8s_get_pod_status_and_logs` | `CoreV1Api.read_namespaced_pod_log` | **Read-Only**: Reads container logs (tail lines 50). |
| **Kubernetes** | `k8s_inspect_services` | `CoreV1Api.list_namespaced_service` | **Read-Only**: Lists ClusterIP and port mappings in `mlops-app`. |
| **Kubernetes** | `k8s_list_events` | `CoreV1Api.list_namespaced_event` | **Read-Only**: Filters cluster Warning events (`Unhealthy`, `CrashLoopBackOff`). |
| **Kubernetes** | `k8s_diagnose_unhealthy_pods` | `CoreV1Api` container statuses | **Read-Only**: Inspects exit codes, `OOMKilled`, and failed readiness probes. |
| **Kubernetes** | `k8s_get_deployed_version` | `AppsV1Api` deployment reader | **Read-Only**: Inspects container image tag and ConfigMap `APP_VERSION`. |
| **Kubernetes** | `k8s_execute_cluster_action` | `AppsV1Api.patch_namespaced_deployment` | **Mutating**: Restricted to namespace `mlops-app` and actions `restart`/`scale`. Requires `confirmed=True`. |

---

## 3. Five Tested Agent Conversations with Actual Tool Evidence

### Conversation 1: Dataset Provenance Inspection (Task 1)
**User Request:**
> *"Which dataset version was used for experiment 99d35a4b?"*

**Agent Execution Plan:**
1. Calls `dvc_get_dataset_version(dataset_path="data/raw_data.csv", run_id="99d35a4bc73b4d2183bdad0e1f3b9afe")`
2. Reads authentic MLflow run tags `dvc_data_version`, `dvc_data_hash`, and `git_commit`.

**Tool Output:**
```json
{
  "status": "success",
  "provenance_source": "MLflow Run ID '99d35a4bc73b4d2183bdad0e1f3b9afe'",
  "dataset_path": "data/raw_data.csv",
  "dvc_version_tag": "v1.0",
  "dvc_md5_hash": "c9156e0de8d16bf37a14bbe75be4d2d5",
  "git_commit": "3b4575c3e72ac3885c2bb459e3b6579ce0462656",
  "dataset_rows": "150"
}
```

**Agent Response:**
> **Known Facts:**
> - Experiment `99d35a4b` was trained using dataset version `v1.0` (150 rows).
> - Tracked DVC MD5 Checksum: `c9156e0de8d16bf37a14bbe75be4d2d5`.
> - Associated Git Commit: `3b4575c3e72ac3885c2bb459e3b6579ce0462656`.
>
> **Recommended Action:**
> To reproduce this exact dataset locally, execute:  
> `git checkout 3b4575c3e72ac3885c2bb459e3b6579ce0462656 && dvc checkout`

---

### Conversation 2: Experiment Intelligence & Candidate Selection (Task 2)
**User Request:**
> *"Compare the last five training runs and recommend the best candidate based on F1 score."*

**Agent Execution Plan:**
1. Calls `mlflow_compare_runs(last_n=5)`
2. Pulls metrics: F1 score, Accuracy, Precision, Recall, and recorded DVC dataset hashes.
3. Ranks candidates and evaluates against secondary criteria.

**Tool Output:**
```json
{
  "status": "success",
  "compared_count": 5,
  "recommended_candidate": "99d35a4bc73b4d2183bdad0e1f3b9afe",
  "recommendation_reasoning": "Recommended run '99d35a4b' achieves highest balanced performance (F1: 0.7778, Accuracy: 0.8333) with verified DVC dataset hash 'c9156e0d'."
}
```

**Agent Response:**
> **Synthesis:**  
> Recommended candidate run `99d35a4b` achieved the highest balanced test performance (F1: `0.7778`, Accuracy: `0.8333`). Run status is `FINISHED`, and model artifact is registered in the Model Registry as `iris_classifier`. Note: A high F1 score alone does not guarantee production readiness; verify probe latency and container memory limits before promoting to active traffic.

---

### Conversation 3: Airflow Workflow Execution & Human Confirmation (Task 3)
**User Request:**
> *"Trigger the model training pipeline."*

**Agent Execution Plan:**
1. Calls `airflow_trigger_dag(dag_id="ml_training_pipeline", confirmed=False)`
2. Detects mutation; intercepts execution with `needs_confirmation`.
3. Prompts user: `[CONFIRMATION GATE] Authorize action 'airflow_trigger_dag'? (yes/no): `
4. Upon user confirmation (`yes`), re-invokes with `confirmed=True`.

**Tool Output:**
```json
{
  "status": "success",
  "dag_id": "ml_training_pipeline",
  "run_id": "manual__2026-11-15T11:45:00",
  "state": "queued",
  "message": "Authorized DAG 'ml_training_pipeline' triggered successfully."
}
```

**Agent Response:**
> **Known Facts:**  
> Pipeline run `manual__2026-11-15T11:45:00` has been queued on the Airflow scheduler.  
> **Recommended Action:**  
> Monitor progression through the 6 stages: `data_validation` $\rightarrow$ `data_preprocessing` $\rightarrow$ `model_training` $\rightarrow$ `model_evaluation` $\rightarrow$ `quality_gate` $\rightarrow$ `model_promotion`.

---

### Conversation 4: Kubernetes Operations & Cluster Restart (Task 4)
**User Request:**
> *"Restart the prediction deployment."*

**Agent Execution Plan:**
1. Calls `k8s_execute_cluster_action(action="restart", namespace="mlops-app", confirmed=False)`
2. Safety gate triggers `needs_confirmation`.
3. Interactively confirms authorization from user.
4. Calls `AppsV1Api.patch_namespaced_deployment` setting `kubectl.kubernetes.io/restartedAt`.

**Tool Output:**
```json
{
  "status": "success",
  "action": "restart",
  "deployment": "mlops-prediction-deployment",
  "restarted_at": "2026-11-15T11:48:00+00:00",
  "message": "Successfully initiated rolling restart for deployment 'mlops-prediction-deployment'."
}
```

**Agent Response:**
> **Known Facts:**  
> Rolling rollout initiated on `mlops-prediction-deployment` in namespace `mlops-app`.  
> Replicas will be sequentially replaced as readiness probes pass on new containers.

---

### Conversation 5: Mandatory Integration Scenario (Multi-Tool Correlation)
**User Request:**
> *"Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."*

**Agent Execution Sequence:**
1. `mlflow_get_best_run` $\rightarrow$ finds Run ID `99d35a4b` (F1 = `0.7778`, Status = `FINISHED`).
2. `dvc_get_dataset_version` $\rightarrow$ extracts DVC MD5 `c9156e0d` and Git commit `3b4575c3`.
3. `k8s_list_pods_and_deployments` $\rightarrow$ inspects namespace `mlops-app` (2/2 Pods Ready and Running).
4. `airflow_get_dag_run_status` $\rightarrow$ inspects latest pipeline state (`success`).
5. `dvc_detect_changes` $\rightarrow$ checks if local raw data has drifted relative to DVC index.
6. `evaluate_retraining_evidence` $\rightarrow$ evaluates empirical criteria:
   - Has data drifted? No.
   - Did quality gate fail? No.
   - Is candidate F1 degraded? No ($0.7778 \ge 0.75$).
   - Are live pods healthy? Yes.

**Agent Response:**
> **Evaluation Summary:**
> - **Candidate Model:** Run ID `99d35a4b` (F1: `0.7778`, Accuracy: `0.8333`)
> - **Dataset Provenance:** DVC Version `v1.0` (MD5: `c9156e0de8d16bf37a14bbe75be4d2d5`)
> - **Deployment Health:** 2/2 Pods Running and Ready in namespace `mlops-app`
> - **Retraining Decision:** **RETRAINING NOT NECESSARY**
> - **Reasoning:** Verified facts: Data is clean with no DVC drift, candidate metrics satisfy quality thresholds, and active workloads are stable.
> - **Safety Enforcement:** Retraining is never triggered automatically from an inquiry.

---

## 4. Two Injected Failure Investigations

### Injected Failure 1: Kubernetes Pod CrashLoopBackOff & Failed Probes
- **Root Cause Injected:** In `kubernetes/deployment.yaml`, the container command was altered to an invalid binary `/bin/invalid_service` and memory limits were set to an unachievable `4Mi` constraint.
- **Observed Symptom:** Application prediction endpoint returned `HTTP 502 / Connection Refused`.
- **Agent Diagnosis Walkthrough:**
  1. User asks: *"Why is my prediction API unavailable?"*
  2. Agent invokes `k8s_diagnose_unhealthy_pods(namespace="mlops-app")`.
  3. Tool extracts:
     - Container state: `waiting.reason = "CrashLoopBackOff"`
     - Restart count: `5`
     - Last termination state: `reason="OOMKilled", exit_code=137`
  4. Agent queries `k8s_list_events(namespace="mlops-app")` and uncovers:
     - Warning event: `Back-off 5m0s restarting failed container`
     - Warning event: `Liveness probe failed: HTTP probe failed with status 503`
- **Resolution:**
  - Restored container entrypoint to `uvicorn api.main:app`.
  - Scaled memory request to `128Mi` and limit to `256Mi`.
  - Applied fix: `kubectl apply -f kubernetes/deployment.yaml`.
  - Re-queried agent: Confirmed healthy status (`2/2 Pods Ready and Running`).

---

### Injected Failure 2: Airflow Data Quality Gate Rejection & Promotion Blocking
- **Root Cause Injected:** Injected corrupt data records into `data/raw_data.csv` causing severe feature distortion and dropped classification F1 score to `0.42` (below the required `0.75` threshold).
- **Observed Symptom:** Training workflow stopped abruptly before promoting the model to production.
- **Agent Diagnosis Walkthrough:**
  1. User asks: *"Check the training pipeline status and diagnose why execution stopped."*
  2. Agent calls `airflow_get_dag_run_status` $\rightarrow$ reports state `failed`.
  3. Agent calls `airflow_get_failed_tasks` $\rightarrow$ pinpoints failed task ID: `quality_gate`.
  4. Agent calls `airflow_get_task_logs(task_id="quality_gate")` $\rightarrow$ reveals exact error log:  
     `ValueError: Quality Gate Failed! Model F1 score (0.4211) is below threshold (0.7500). Downstream model promotion is strictly blocked.`
  5. Crucial Safety Behavior: Downstream task `model_promotion` was never executed. The flawed model was successfully prevented from polluting the staging or production registry!
- **Resolution:**
  - Reverted corrupted records using `dvc checkout data/raw_data.csv`.
  - Validated clean dataset schema.
  - Re-ran pipeline: All stages (`data_validation` through `model_promotion`) passed green.

---

## 5. Known Limitations & Proposed Improvements

1. **Local vs Cloud Multi-Tenancy**:
   - *Current State*: Configured for local development clusters (Minikube / Kind) and local DVC remote storage.
   - *Proposed Improvement*: Transition DVC storage to AWS S3 / Google Cloud Storage with IAM role-based access tokens (OIDC).
2. **Dynamic Semantic Routing vs Multi-Agent Swarm**:
   - *Current State*: Centralized Orchestrator handling structured routing and confirmation gates.
   - *Proposed Improvement*: Introduce specialized subagents (e.g. `KubernetesDiagnosticAgent`, `MLflowIntelligenceAgent`) running as an autonomous agent swarm.
3. **Automated Canary Rollout**:
   - *Current State*: Rolling restart and scaling via Kubernetes Deployment API.
   - *Proposed Improvement*: Integrate Argo Rollouts or Istio Service Mesh for traffic-splitting canary deployments based on real-time inference error rates.
