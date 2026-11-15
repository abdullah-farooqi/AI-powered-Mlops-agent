import os
import subprocess
import shlex
import re
from typing import Dict, Any, Optional

APPROVED_COMMAND_PREFIXES = {
    "dvc": {"status", "diff", "version"},
    "git": {"status", "log", "rev-parse", "describe", "show"},
}


def _run_safe_command(cmd_str: str) -> Dict[str, Any]:
    """Execute whitelisted subprocess commands with strict argument parsing and security checks."""
    try:
        parts = shlex.split(cmd_str)
    except Exception as e:
        return {"status": "error", "error": f"Command parsing error: {e}"}

    if not parts:
        return {"status": "error", "error": "Empty command provided."}

    binary = parts[0]
    if binary not in APPROVED_COMMAND_PREFIXES:
        return {"status": "error", "error": f"Security violation: Binary '{binary}' is not permitted."}

    # Find the first non-option argument (subcommand)
    subcommand = next((p for p in parts[1:] if not p.startswith("-")), None)
    if subcommand not in APPROVED_COMMAND_PREFIXES[binary]:
        return {
            "status": "error",
            "error": (
                f"Security violation: Subcommand '{subcommand}' for '{binary}' "
                f"is not in whitelist {APPROVED_COMMAND_PREFIXES[binary]}."
            ),
        }

    # Block dangerous redirection, output flags, or arbitrary execution
    for p in parts:
        if any(bad in p for bad in [";", "&", "|", "`", "$", ">", "<"]):
            return {"status": "error", "error": "Security violation: Shell operators are strictly forbidden."}

    try:
        res = subprocess.run(parts, capture_output=True, text=True, timeout=10)
        return {
            "status": "success" if res.returncode == 0 else "error",
            "returncode": res.returncode,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip(),
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": f"Command '{cmd_str}' timed out after 10 seconds."}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def dvc_list_tracked_data() -> Dict[str, Any]:
    """Task 1.1: Identify all datasets and files tracked by DVC in the workspace."""
    tracked_items = []
    for root, _, files in os.walk("."):
        if any(excluded in root for excluded in [".git", ".dvc", "venv", ".venv"]):
            continue
        for file in files:
            if file.endswith(".dvc"):
                data_target = os.path.join(root, file[:-4])
                tracked_items.append(
                    {
                        "dvc_pointer": os.path.join(root, file),
                        "data_file": data_target,
                        "exists_locally": os.path.exists(data_target),
                        "file_size_bytes": os.path.getsize(data_target) if os.path.exists(data_target) else 0,
                    }
                )

    has_dvc_yaml = os.path.exists("dvc.yaml")
    return {
        "status": "success",
        "tracked_files_count": len(tracked_items),
        "tracked_datasets": tracked_items,
        "dvc_yaml_present": has_dvc_yaml,
    }


def dvc_get_dataset_version(dataset_path: str = "data/raw_data.csv", run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Task 1.2: Retrieve dataset version, DVC md5, and Git commit.
    If run_id is supplied, retrieves the authentic recorded provenance from MLflow tags.
    Otherwise, inspects the current repository commit touching the .dvc file.
    """
    # 1. If run_id is provided, look up recorded provenance from MLflow run
    if run_id:
        try:
            from agent.tools.mlflow_tools import mlflow_get_run_details

            run_res = mlflow_get_run_details(run_id)
            if run_res.get("status") == "success":
                tags = run_res.get("tags", {})
                return {
                    "status": "success",
                    "provenance_source": f"MLflow Run ID '{run_id}'",
                    "dataset_path": dataset_path,
                    "dvc_version_tag": tags.get("dvc_data_version", "unknown"),
                    "dvc_md5_hash": tags.get("dvc_data_hash", "unknown"),
                    "actual_data_hash": tags.get("actual_data_hash", "unknown"),
                    "git_commit": tags.get("git_commit", "unknown"),
                    "dataset_rows": tags.get("dataset_rows", "unknown"),
                }
        except Exception as e:
            return {"status": "error", "error": f"Failed retrieving provenance for run '{run_id}': {e}"}

    # 2. Inspect local repository commit and .dvc pointer file
    dvc_pointer = f"{dataset_path}.dvc"
    git_commit = "uncommitted"
    git_tag = "untagged"

    res_git = _run_safe_command(f"git log -1 --format=%H -- {dvc_pointer}")
    if res_git.get("status") == "success" and res_git.get("stdout"):
        git_commit = res_git["stdout"]
    else:
        res_head = _run_safe_command("git rev-parse HEAD")
        if res_head.get("status") == "success":
            git_commit = res_head["stdout"]

    res_tag = _run_safe_command("git describe --tags --always")
    if res_tag.get("status") == "success" and res_tag.get("stdout"):
        git_tag = res_tag["stdout"]

    dvc_md5 = "untracked"
    if os.path.exists(dvc_pointer):
        try:
            with open(dvc_pointer, "r") as f:
                for line in f:
                    if "md5:" in line:
                        dvc_md5 = line.split("md5:")[-1].strip()
                        break
        except Exception as e:
            return {"status": "error", "error": f"Failed reading {dvc_pointer}: {e}"}

    return {
        "status": "success",
        "provenance_source": "Repository Git/DVC Inspection",
        "dataset_path": dataset_path,
        "dvc_pointer": dvc_pointer,
        "git_commit": git_commit,
        "git_tag": git_tag,
        "dvc_md5_hash": dvc_md5,
        "data_exists": os.path.exists(dataset_path),
    }


def dvc_detect_changes(target_file: str = "data/raw_data.csv.dvc") -> Dict[str, Any]:
    """Task 1.3: Detect whether tracked data has changed. Returns explicit error if DVC fails."""
    cmd = f"dvc status {target_file}" if os.path.exists(target_file) else "dvc status"
    res = _run_safe_command(cmd)
    if res.get("status") != "success":
        return {"status": "error", "error": f"DVC status inspection failed: {res.get('stderr') or res.get('error')}"}
    output = res.get("stdout", "")
    has_changes = "modified" in output.lower() or "not in cache" in output.lower() or "deleted" in output.lower()
    return {
        "status": "success",
        "target_file": target_file,
        "data_has_changed": has_changes,
        "dvc_status_output": output if output else "Data is clean and matches DVC index.",
    }


def dvc_compare_versions(commit_a: str, commit_b: str, target_path: str = "data/raw_data.csv") -> Dict[str, Any]:
    """
    Task 1.4: Compare dataset versions between two Git commits by inspecting .dvc pointer changes.
    Returns authentic MD5 hashes from each commit and detected differences.
    """
    dvc_file = f"{target_path}.dvc"
    cmd_a = f"git show {commit_a}:{dvc_file}"
    cmd_b = f"git show {commit_b}:{dvc_file}"

    res_a = _run_safe_command(cmd_a)
    res_b = _run_safe_command(cmd_b)

    if res_a.get("status") != "success":
        return {
            "status": "error",
            "error": f"Could not retrieve {dvc_file} at commit '{commit_a}': {res_a.get('stderr')}",
        }
    if res_b.get("status") != "success":
        return {
            "status": "error",
            "error": f"Could not retrieve {dvc_file} at commit '{commit_b}': {res_b.get('stderr')}",
        }

    def _extract_md5_and_size(content: str):
        md5_match = re.search(r"md5:\s*(\S+)", content)
        size_match = re.search(r"size:\s*(\d+)", content)
        return {
            "md5": md5_match.group(1) if md5_match else "unknown",
            "size": int(size_match.group(1)) if size_match else 0,
        }

    meta_a = _extract_md5_and_size(res_a.get("stdout", ""))
    meta_b = _extract_md5_and_size(res_b.get("stdout", ""))

    identical = meta_a["md5"] == meta_b["md5"]

    return {
        "status": "success",
        "target_dataset": target_path,
        "commit_a": commit_a,
        "commit_b": commit_b,
        "meta_a": meta_a,
        "meta_b": meta_b,
        "data_identical": identical,
        "summary": (
            "Datasets are identical."
            if identical
            else f"Dataset modified: hash changed from {meta_a['md5'][:8]} to {meta_b['md5'][:8]}."
        ),
    }


def dvc_explain_reproduction(target_version: str) -> Dict[str, Any]:
    """Task 1.5: Explain exact reproducible commands to checkout a previous dataset version."""
    return {
        "status": "success",
        "target_version": target_version,
        "instructions": [
            f"1. Checkout target Git commit/tag: git checkout {target_version}",
            "2. Pull versioned data binaries from DVC remote: dvc checkout (or dvc pull)",
            "3. Verify data checksum matches: dvc status",
        ],
        "safe_reproduction_command": f"git checkout {target_version} && dvc checkout",
    }
