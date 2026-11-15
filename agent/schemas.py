from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

class ToolExecutionResult(BaseModel):
    tool_name: str
    status: str  # "success", "error", "denied", "needs_confirmation"
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_seconds: float
    request_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class RetrainingEvidence(BaseModel):
    has_data_drift: bool = False
    quality_gate_passed: bool = False
    model_degraded: bool = False
    active_pods_healthy: bool = False
    service_unreachable: bool = False
    recommendation: str
    decision_reasoning: str

class AgentStructuredResponse(BaseModel):
    query: str
    known_facts: List[str]
    uncertainties: List[str]
    recommended_actions: List[str]
    tool_results: List[ToolExecutionResult]
    final_narrative: str
    confirmation_required: Optional[Dict[str, Any]] = None

# Comprehensive Tool Schemas for LLM Function Calling (OpenAI / Anthropic / Gemini format)
TOOL_SCHEMAS = [
    {
        "name": "dvc_list_tracked_data",
        "description": "Lists all datasets, .dvc pointer files, and dvc.yaml stages in the repository.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "dvc_get_dataset_version",
        "description": "Retrieves the dataset version, MD5 checksum, and Git commit hash. Can look up provenance for a specific MLflow run_id or inspect current git commit.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_path": {"type": "string", "default": "data/raw_data.csv", "description": "Relative path to tracked dataset"},
                "run_id": {"type": "string", "description": "Optional MLflow run ID to look up historical training data provenance"}
            }
        }
    },
    {
        "name": "dvc_detect_changes",
        "description": "Runs DVC status check to verify whether dataset has been modified or drifted since the last commit.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "dvc_compare_versions",
        "description": "Compares dataset versions between two Git commits or tags, analyzing changes in DVC pointer files and MD5 checksums.",
        "parameters": {
            "type": "object",
            "properties": {
                "commit_a": {"type": "string", "description": "Baseline Git commit or tag (e.g. data-v1)"},
                "commit_b": {"type": "string", "description": "Target Git commit or tag (e.g. data-v2)"},
                "target_path": {"type": "string", "default": "data/raw_data.csv"}
            },
            "required": ["commit_a", "commit_b"]
        }
    },
    {
        "name": "dvc_explain_reproduction",
        "description": "Explains exact reproducible commands to checkout a specific historical dataset version.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_version": {"type": "string", "description": "Target Git commit or tag"}
            },
            "required": ["target_version"]
        }
    },
    {
        "name": "mlflow_list_experiments_and_runs",
        "description": "Lists recent training runs and their metrics in the MLflow tracking server.",
        "parameters": {
            "type": "object",
            "properties": {
                "experiment_name": {"type": "string", "default": "mlops-model-training"},
                "max_results": {"type": "integer", "default": 5}
            }
        }
    },
    {
        "name": "mlflow_get_run_details",
        "description": "Retrieves full evaluation metrics, hyperparameters, artifacts, and DVC tags for a specific run ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "MLflow run ID"}
            },
            "required": ["run_id"]
        }
    },
    {
        "name": "mlflow_get_best_run",
        "description": "Selects the best trained model run by evaluation metric (e.g. f1_score), enforcing status FINISHED and secondary performance gates.",
        "parameters": {
            "type": "object",
            "properties": {
                "experiment_name": {"type": "string", "default": "mlops-model-training"},
                "metric_name": {"type": "string", "default": "f1_score"},
                "mode": {"type": "string", "default": "max"}
            }
        }
    },
    {
        "name": "mlflow_compare_runs",
        "description": "Compares recent training runs side-by-side on metrics and provides a defensible candidate recommendation.",
        "parameters": {
            "type": "object",
            "properties": {
                "run_ids": {"type": "array", "items": {"type": "string"}, "description": "Specific run IDs to compare"},
                "last_n": {"type": "integer", "default": 5, "description": "Number of recent runs to compare"}
            }
        }
    },
    {
        "name": "mlflow_get_registered_model_info",
        "description": "Queries the MLflow Model Registry for registered versions and deployment stages (Staging, Production).",
        "parameters": {
            "type": "object",
            "properties": {
                "model_name": {"type": "string", "default": "iris_classifier"}
            }
        }
    },
    {
        "name": "airflow_list_dags",
        "description": "Lists all available DAGs and their paused/active status via Airflow REST API.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "airflow_get_dag_run_status",
        "description": "Retrieves the latest execution status and timestamps for a DAG.",
        "parameters": {
            "type": "object",
            "properties": {
                "dag_id": {"type": "string", "default": "ml_training_pipeline"}
            }
        }
    },
    {
        "name": "airflow_trigger_dag",
        "description": "Triggers an authorized training pipeline DAG. Requires human confirmation.",
        "parameters": {
            "type": "object",
            "properties": {
                "dag_id": {"type": "string", "default": "ml_training_pipeline"},
                "confirmed": {"type": "boolean", "default": False}
            }
        }
    },
    {
        "name": "airflow_get_failed_tasks",
        "description": "Queries task instance states of a DAG run to identify any failed stages (e.g. data_validation or quality_gate).",
        "parameters": {
            "type": "object",
            "properties": {
                "dag_id": {"type": "string", "default": "ml_training_pipeline"},
                "run_id": {"type": "string", "description": "Optional specific DAG run ID"}
            }
        }
    },
    {
        "name": "airflow_get_task_logs",
        "description": "Fetches task execution logs from Airflow for root-cause diagnosis of failure.",
        "parameters": {
            "type": "object",
            "properties": {
                "dag_id": {"type": "string"},
                "run_id": {"type": "string"},
                "task_id": {"type": "string"}
            },
            "required": ["dag_id", "run_id", "task_id"]
        }
    },
    {
        "name": "k8s_list_pods_and_deployments",
        "description": "Lists Pods, phases, readiness, and Deployments in the approved Kubernetes namespace.",
        "parameters": {
            "type": "object",
            "properties": {
                "namespace": {"type": "string", "default": "mlops-app"}
            }
        }
    },
    {
        "name": "k8s_get_pod_status_and_logs",
        "description": "Retrieves container phase, readiness, and recent stdout/stderr logs for a specific pod.",
        "parameters": {
            "type": "object",
            "properties": {
                "pod_name": {"type": "string"},
                "namespace": {"type": "string", "default": "mlops-app"}
            },
            "required": ["pod_name"]
        }
    },
    {
        "name": "k8s_inspect_services",
        "description": "Inspects Kubernetes ClusterIP services, ports, and selectors in namespace.",
        "parameters": {
            "type": "object",
            "properties": {
                "namespace": {"type": "string", "default": "mlops-app"}
            }
        }
    },
    {
        "name": "k8s_list_events",
        "description": "Inspects Kubernetes cluster Warning events (e.g. Unhealthy, BackOff, FailedScheduling).",
        "parameters": {
            "type": "object",
            "properties": {
                "namespace": {"type": "string", "default": "mlops-app"}
            }
        }
    },
    {
        "name": "k8s_diagnose_unhealthy_pods",
        "description": "Deep diagnostic inspection scanning for CrashLoopBackOff, OOMKilled, failed readiness probes, and restart events.",
        "parameters": {
            "type": "object",
            "properties": {
                "namespace": {"type": "string", "default": "mlops-app"}
            }
        }
    },
    {
        "name": "k8s_get_deployed_version",
        "description": "Reports deployed container image tag, ConfigMap APP_VERSION, and replica counts.",
        "parameters": {
            "type": "object",
            "properties": {
                "namespace": {"type": "string", "default": "mlops-app"},
                "deployment_name": {"type": "string", "default": "mlops-prediction-deployment"}
            }
        }
    },
    {
        "name": "k8s_execute_cluster_action",
        "description": "Executes rolling restart or replica scaling on the Kubernetes deployment. Requires explicit human confirmation.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["restart", "scale"]},
                "deployment_name": {"type": "string", "default": "mlops-prediction-deployment"},
                "replicas": {"type": "integer", "description": "Target replica count for scale action"},
                "confirmed": {"type": "boolean", "default": False}
            },
            "required": ["action"]
        }
    }
]
