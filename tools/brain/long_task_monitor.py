from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.brain.project_profiles import load_project_profile


DEFAULT_TIMEOUT_SECONDS = 7200
DEFAULT_STALE_AFTER_SECONDS = 3600
ARTIFACT_SCAN_FILE_LIMIT = 512
DEFAULT_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEASE_ROOT = "brain/output/resource_leases"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {"_load_error": str(exc)}
    return payload if isinstance(payload, dict) else {"_load_error": "progress payload is not a JSON object"}


def _load_lease_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        return {"_load_error": "lease file does not exist"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {"_load_error": str(exc)}
    return payload if isinstance(payload, dict) else {"_load_error": "lease payload is not a JSON object"}


def _number(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _tail(path: Path | None, line_count: int = 12) -> list[str]:
    if path is None or not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except Exception as exc:
        return [f"<tail_error: {exc}>"]
    return lines[-line_count:]


def _artifact_files(path: Path | None, *, limit: int = ARTIFACT_SCAN_FILE_LIMIT) -> tuple[list[Path], bool]:
    if path is None or not path.exists():
        return [], False
    if path.is_file():
        return [path], False
    candidates: list[Path] = []
    truncated = False
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        candidates.append(item)
        if len(candidates) >= max(1, int(limit)):
            truncated = True
            break
    return candidates, truncated


def _artifact_mtime(path: Path | None) -> str:
    candidates, _truncated = _artifact_files(path)
    if not candidates:
        return ""
    latest = max(item.stat().st_mtime for item in candidates)
    return datetime.fromtimestamp(latest, tz=timezone.utc).isoformat()


def _artifact_scan_truncated(path: Path | None) -> bool:
    _candidates, truncated = _artifact_files(path)
    return truncated


def _artifact_summary(path: Path | None, *, limit: int = 8) -> list[dict[str, Any]]:
    candidates, _truncated = _artifact_files(path)
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    out: list[dict[str, Any]] = []
    for item in candidates[:limit]:
        stat = item.stat()
        out.append(
            {
                "path": str(item),
                "name": item.name,
                "bytes": int(stat.st_size),
                "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return out


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"$p = Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue; if ($p) {{ 'alive' }}",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return "alive" in (result.stdout or "")


def _parse_child_pids(value: str | list[int] | None) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[int] = []
        for item in value:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out
    out = []
    for part in str(value or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def _normalize_rel_path(path: str | Path, *, workspace_root: Path) -> str:
    candidate = Path(path)
    try:
        rel = candidate.resolve().relative_to(workspace_root.resolve())
        return rel.as_posix()
    except Exception:
        return str(candidate).replace("\\", "/")


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _target_projects_from_paths(paths: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for path in paths:
        parts = str(path or "").replace("\\", "/").split("/")
        if len(parts) >= 3 and parts[0] == "brain" and parts[1] == "output" and parts[2] == "agent_runs":
            if "workspace" not in seen:
                seen.add("workspace")
                out.append("workspace")
            continue
        if len(parts) >= 3 and parts[1] == "output" and parts[2] == "agent_runs":
            project = parts[0].strip()
            if project and project not in seen:
                seen.add(project)
                out.append(project)
    return out


def _lease_path(
    lease_id: str,
    *,
    workspace_root: Path,
    lease_root: str | Path = DEFAULT_LEASE_ROOT,
) -> Path:
    lease = str(lease_id or "").strip()
    if not lease or "/" in lease or "\\" in lease or ".." in lease:
        return Path("")
    root = Path(str(lease_root))
    if not root.is_absolute():
        root = workspace_root / root
    return root / f"{lease}.json"


def _project_run_namespace_root(project_id: str, run_id: str) -> str:
    project = str(project_id or "").strip()
    run = str(run_id or "").strip().replace("\\", "/").strip("/")
    if not project or not run:
        return ""
    try:
        profile = load_project_profile(project)
    except Exception:
        profile = {}
    namespace = profile.get("process_namespace") if isinstance(profile.get("process_namespace"), dict) else {}
    template = str(namespace.get("run_path_template", "") or "").strip().replace("\\", "/")
    if template:
        return template.replace("<run_id>", run).strip("/")
    return f"{project}/output/agent_runs/{run}"


def validate_resource_lease(
    *,
    lease_id: str,
    project_id: str,
    target_project_ids: list[str],
    workspace_root: str | Path | None = None,
    lease_root: str | Path = DEFAULT_LEASE_ROOT,
) -> dict[str, Any]:
    requester = str(project_id or "").strip().replace("\\", "/").strip("/")
    targets = [item for item in _as_string_list(target_project_ids) if item]
    root = Path(workspace_root) if workspace_root is not None else DEFAULT_WORKSPACE_ROOT
    lease_text = str(lease_id or "").strip()
    if not lease_text or "/" in lease_text or "\\" in lease_text or ".." in lease_text:
        return {
            "schema_version": 1,
            "status": "blocked",
            "reason": "invalid_lease_id",
            "lease_id": str(lease_id or ""),
            "project_id": requester,
            "target_project_ids": targets,
            "path": "",
        }
    path = _lease_path(str(lease_id or ""), workspace_root=root, lease_root=lease_root)
    payload = _load_lease_json(path)
    if payload.get("_load_error"):
        return {
            "schema_version": 1,
            "status": "blocked",
            "reason": "lease_not_found_or_unreadable",
            "lease_id": str(lease_id or ""),
            "project_id": requester,
            "target_project_ids": targets,
            "path": _normalize_rel_path(path, workspace_root=root),
            "error": payload.get("_load_error"),
        }
    status = str(payload.get("status", "") or "").strip().lower()
    if status not in {"active", "approved", "granted"}:
        return {
            "schema_version": 1,
            "status": "blocked",
            "reason": "lease_not_active",
            "lease_id": str(lease_id or ""),
            "project_id": requester,
            "target_project_ids": targets,
            "path": _normalize_rel_path(path, workspace_root=root),
            "lease_status": status,
        }
    expires_at = _parse_datetime(payload.get("expires_at") or payload.get("valid_until"))
    if expires_at is not None and expires_at <= _utc_now():
        return {
            "schema_version": 1,
            "status": "blocked",
            "reason": "lease_expired",
            "lease_id": str(lease_id or ""),
            "project_id": requester,
            "target_project_ids": targets,
            "path": _normalize_rel_path(path, workspace_root=root),
            "expires_at": expires_at.isoformat(),
        }

    allowed_requesters = set(
        _as_string_list(payload.get("requester_project_id"))
        + _as_string_list(payload.get("requester_project_ids"))
        + _as_string_list(payload.get("granted_to_project_id"))
        + _as_string_list(payload.get("granted_to_project_ids"))
        + _as_string_list(payload.get("project_id"))
        + _as_string_list(payload.get("project_ids"))
        + _as_string_list(payload.get("allowed_project_id"))
        + _as_string_list(payload.get("allowed_project_ids"))
    )
    allowed_targets = set(
        _as_string_list(payload.get("target_project_id"))
        + _as_string_list(payload.get("target_project_ids"))
        + _as_string_list(payload.get("resource_project_id"))
        + _as_string_list(payload.get("resource_project_ids"))
        + _as_string_list(payload.get("allowed_target_project_id"))
        + _as_string_list(payload.get("allowed_target_project_ids"))
    )
    requester_ok = "*" in allowed_requesters or requester in allowed_requesters
    target_ok = "*" in allowed_targets or not targets or all(target in allowed_targets for target in targets)
    if not requester_ok:
        return {
            "schema_version": 1,
            "status": "blocked",
            "reason": "lease_requester_not_authorized",
            "lease_id": str(lease_id or ""),
            "project_id": requester,
            "target_project_ids": targets,
            "path": _normalize_rel_path(path, workspace_root=root),
            "allowed_requesters": sorted(allowed_requesters),
        }
    if not target_ok:
        return {
            "schema_version": 1,
            "status": "blocked",
            "reason": "lease_target_not_authorized",
            "lease_id": str(lease_id or ""),
            "project_id": requester,
            "target_project_ids": targets,
            "path": _normalize_rel_path(path, workspace_root=root),
            "allowed_targets": sorted(allowed_targets),
        }
    return {
        "schema_version": 1,
        "status": "ok",
        "lease_id": str(lease_id or ""),
        "project_id": requester,
        "target_project_ids": targets,
        "path": _normalize_rel_path(path, workspace_root=root),
        "expires_at": expires_at.isoformat() if expires_at is not None else "",
    }


def validate_project_namespace(
    *,
    project_id: str,
    run_id: str,
    paths: list[str | Path],
    workspace_root: str | Path | None = None,
    allow_cross_project: bool = False,
    lease_id: str = "",
    lease_root: str | Path = DEFAULT_LEASE_ROOT,
) -> dict[str, Any]:
    project = str(project_id or "").strip().replace("\\", "/").strip("/")
    run = str(run_id or "").strip().replace("\\", "/").strip("/")
    root = Path(workspace_root) if workspace_root is not None else DEFAULT_WORKSPACE_ROOT
    namespace_root = _project_run_namespace_root(project, run)
    rel_paths = [_normalize_rel_path(path, workspace_root=root) for path in paths if str(path or "").strip()]
    if not project or not run:
        return {
            "schema_version": 1,
            "status": "blocked",
            "reason": "project_run_identity_required",
            "project_id": project,
            "run_id": run,
            "namespace_root": namespace_root,
            "paths": rel_paths,
            "violating_paths": rel_paths,
            "allow_cross_project": bool(allow_cross_project),
        }
    if allow_cross_project:
        cross_paths = [path for path in rel_paths if path and not (path == namespace_root or path.startswith(f"{namespace_root}/"))]
        if not cross_paths:
            return {
                "schema_version": 1,
                "status": "ok",
                "project_id": project,
                "run_id": run,
                "namespace_root": namespace_root,
                "paths": rel_paths,
                "violating_paths": [],
                "allow_cross_project": True,
                "lease": {"schema_version": 1, "status": "not_required", "reason": "own_namespace_only"},
            }
        targets = _target_projects_from_paths(cross_paths)
        unknown_cross_paths = [
            path
            for path in cross_paths
            if path
            and not _target_projects_from_paths([path])
        ]
        if unknown_cross_paths:
            return {
                "schema_version": 1,
                "status": "blocked",
                "reason": "cross_project_agent_run_path_required",
                "project_id": project,
                "run_id": run,
                "namespace_root": namespace_root,
                "paths": rel_paths,
                "violating_paths": unknown_cross_paths,
                "allow_cross_project": True,
            }
        lease = validate_resource_lease(
            lease_id=lease_id,
            project_id=project,
            target_project_ids=targets,
            workspace_root=root,
            lease_root=lease_root,
        )
        if lease.get("status") != "ok":
            return {
                "schema_version": 1,
                "status": "blocked",
                "reason": "cross_project_lease_required",
                "project_id": project,
                "run_id": run,
                "namespace_root": namespace_root,
                "paths": rel_paths,
                "violating_paths": rel_paths,
                "allow_cross_project": True,
                "lease": lease,
            }
        return {
            "schema_version": 1,
            "status": "ok",
            "project_id": project,
            "run_id": run,
            "namespace_root": namespace_root,
            "paths": rel_paths,
            "violating_paths": [],
            "allow_cross_project": True,
            "lease": lease,
        }
    violating = [path for path in rel_paths if path and not (path == namespace_root or path.startswith(f"{namespace_root}/"))]
    payload = {
        "schema_version": 1,
        "status": "blocked" if violating else "ok",
        "project_id": project,
        "run_id": run,
        "namespace_root": namespace_root,
        "paths": rel_paths,
        "violating_paths": violating,
        "allow_cross_project": bool(allow_cross_project),
    }
    if violating:
        payload["reason"] = "project_namespace_violation"
    return payload


def _progress_fraction(progress: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    current = _number(progress, "current_step", "completed_steps", "step", "global_step")
    total = _number(progress, "total_steps", "step_count", "max_steps")
    if current is None or total is None:
        current = _number(progress, "current_epoch", "epoch", "completed_epochs")
        total = _number(progress, "total_epochs", "epochs", "max_epochs")
    if current is None or total is None or total <= 0:
        percent = _number(progress, "progress_percent")
        if percent is None:
            return None, None, None
        return percent / 100.0, None, None
    fraction = max(0.0, min(1.0, current / total))
    return fraction, current, total


def _started_at(progress: dict[str, Any], progress_path: Path | None, now: datetime) -> datetime | None:
    parsed = _parse_datetime(progress.get("started_at") or progress.get("start_time"))
    if parsed is not None:
        return parsed
    if progress_path is not None and progress_path.exists():
        return datetime.fromtimestamp(progress_path.stat().st_ctime, tz=timezone.utc)
    return None


def _updated_at(progress: dict[str, Any], progress_path: Path | None) -> datetime | None:
    parsed = _parse_datetime(progress.get("updated_at") or progress.get("last_update_at") or progress.get("timestamp"))
    if parsed is not None:
        return parsed
    if progress_path is not None and progress_path.exists():
        return datetime.fromtimestamp(progress_path.stat().st_mtime, tz=timezone.utc)
    return None


def build_template(
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    timeout = int(timeout_seconds)
    return {
        "schema_version": 1,
        "poll_window_seconds": timeout,
        "monitoring_mode": "foreground_wait_process_after_gpu_start",
        "gpu_active_wait_rule": (
            "After a GPU task is confirmed active, use a foreground Wait-Process window unless the host crashes."
        ),
        "eta_required": True,
        "required_wait_command": f"Wait-Process -Id <pid> -Timeout {timeout}",
        "powershell_template": "\n".join(
            [
                "$proc = Start-Process -FilePath <command> -ArgumentList <args> -PassThru -NoNewWindow",
                "$taskPid = $proc.Id",
                f"Wait-Process -Id $taskPid -Timeout {timeout}",
                "C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.long_task_monitor status --project-id <project_id> --run-id <run_id> --pid $taskPid --progress <progress.json> --stdout <stdout.log> --stderr <stderr.log> --artifact-dir <artifact_dir> --json",
                "C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.long_task_monitor trace-poll --trace-json <trace.json> --pid $taskPid --poll-window-seconds "
                f"{timeout} --project-id <project_id> --run-id <run_id> --progress <progress.json> --stdout <stdout.log> --stderr <stderr.log> --artifact-dir <artifact_dir> --json",
            ]
        ),
    }


def build_status(
    *,
    pid: int | None = None,
    project_id: str = "",
    run_id: str = "",
    progress_path: str | Path | None = None,
    stdout_path: str | Path | None = None,
    stderr_path: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
    workspace_root: str | Path | None = None,
    allow_cross_project: bool = False,
    lease_id: str = "",
) -> dict[str, Any]:
    progress_file = Path(progress_path) if progress_path else None
    stdout_file = Path(stdout_path) if stdout_path else None
    stderr_file = Path(stderr_path) if stderr_path else None
    artifact_path = Path(artifact_dir) if artifact_dir else None
    namespace = validate_project_namespace(
        project_id=project_id,
        run_id=run_id,
        paths=[path for path in (progress_file, stdout_file, stderr_file, artifact_path) if path is not None],
        workspace_root=workspace_root,
        allow_cross_project=allow_cross_project,
        lease_id=lease_id,
    )
    if namespace.get("status") == "blocked":
        return {
            "schema_version": 1,
            "status": "blocked",
            "reason": namespace.get("reason") or "project_namespace_violation",
            "project_id": str(project_id or ""),
            "run_id": str(run_id or ""),
            "namespace": namespace,
            "pid": pid,
            "pid_alive": False,
            "progress_path": str(progress_file or ""),
            "elapsed_seconds": None,
            "estimated_remaining_seconds": None,
            "eta_at": "",
            "eta_status": "namespace_blocked",
            "progress_percent": None,
            "current_step": None,
            "total_steps": None,
            "stage": "",
            "latest_metric": {},
            "last_log_lines": [],
            "last_error_lines": [],
            "artifact_mtime": "",
            "artifact_scan_truncated": False,
            "updated_at": "",
            "decision": str(namespace.get("reason") or "project_namespace_violation"),
        }
    now = _utc_now()
    progress = _load_json(progress_file)
    started = _started_at(progress, progress_file, now)
    updated = _updated_at(progress, progress_file)
    elapsed = max(0.0, (now - started).total_seconds()) if started is not None else None
    fraction, current, total = _progress_fraction(progress)
    progress_percent = round(fraction * 100.0, 3) if fraction is not None else None

    estimated_remaining: float | None = None
    eta_at = ""
    eta_status = "progress_unavailable"
    if elapsed is not None and fraction is not None:
        stale_seconds = (now - updated).total_seconds() if updated is not None else 0.0
        if stale_seconds >= int(stale_after_seconds):
            eta_status = "stalled_or_waiting"
        elif current is not None and current <= 0:
            eta_status = "warming_up"
        elif fraction <= 0:
            eta_status = "warming_up"
        elif fraction >= 1.0:
            eta_status = "completed"
            estimated_remaining = 0.0
            eta_at = now.isoformat()
        else:
            estimated_remaining = max(0.0, elapsed * (1.0 - fraction) / fraction)
            eta_at = datetime.fromtimestamp(time.time() + estimated_remaining, tz=timezone.utc).isoformat()
            eta_status = "estimated"

    if eta_status == "stalled_or_waiting":
        decision = "inspect_logs_or_resources"
    elif _pid_alive(pid):
        decision = "continue_short_polling"
    elif eta_status == "completed":
        decision = "verify_artifacts"
    else:
        decision = "inspect_exit_or_artifacts"

    return {
        "schema_version": 1,
        "project_id": str(project_id or ""),
        "run_id": str(run_id or ""),
        "namespace": namespace,
        "pid": pid,
        "pid_alive": _pid_alive(pid),
        "progress_path": str(progress_file or ""),
        "elapsed_seconds": round(elapsed, 3) if elapsed is not None else None,
        "estimated_remaining_seconds": round(estimated_remaining, 3) if estimated_remaining is not None else None,
        "eta_at": eta_at,
        "eta_status": eta_status,
        "progress_percent": progress_percent,
        "current_step": current,
        "total_steps": total,
        "stage": progress.get("stage", ""),
        "latest_metric": progress.get("latest_metric", {}),
        "last_log_lines": _tail(stdout_file),
        "last_error_lines": _tail(stderr_file),
        "artifact_mtime": _artifact_mtime(artifact_path),
        "artifact_scan_truncated": _artifact_scan_truncated(artifact_path),
        "updated_at": updated.isoformat() if updated is not None else "",
        "decision": decision,
    }


def build_trace_event(
    *,
    task: str = "",
    project_id: str = "",
    run_id: str = "",
    step_id: str = "",
    run_tag: str = "",
    pid: int | None = None,
    child_pids: list[int] | None = None,
    poll_window_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    progress_path: str | Path | None = None,
    stdout_path: str | Path | None = None,
    stderr_path: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
    final_verification: str = "",
    workspace_root: str | Path | None = None,
    allow_cross_project: bool = False,
    lease_id: str = "",
) -> dict[str, Any]:
    progress_file = Path(progress_path) if progress_path else None
    stdout_file = Path(stdout_path) if stdout_path else None
    stderr_file = Path(stderr_path) if stderr_path else None
    artifact_path = Path(artifact_dir) if artifact_dir else None
    status = build_status(
        pid=pid,
        project_id=project_id,
        run_id=run_id,
        progress_path=progress_file,
        stdout_path=stdout_file,
        stderr_path=stderr_file,
        artifact_dir=artifact_path,
        stale_after_seconds=stale_after_seconds,
        workspace_root=workspace_root,
        allow_cross_project=allow_cross_project,
        lease_id=lease_id,
    )
    eta_no_eta_reason = ""
    if status.get("eta_status") not in {"estimated", "completed"}:
        eta_no_eta_reason = str(status.get("eta_status") or "progress_unavailable")
    event = {
        "type": "long_task_poll",
        "project_id": str(project_id or ""),
        "run_id": str(run_id or ""),
        "namespace": dict(status.get("namespace", {}) or {}),
        "step_id": str(step_id or ""),
        "summary": f"long task poll for {run_tag or task or 'unnamed task'}",
        "evidence": (
            f"pid_alive={status.get('pid_alive')} progress_percent={status.get('progress_percent')} "
            f"eta_status={status.get('eta_status')} decision={status.get('decision')}"
        ),
        "task": str(task or ""),
        "run_tag": str(run_tag or ""),
        "pid": int(pid or 0),
        "pid_alive": bool(status.get("pid_alive")),
        "child_pids": list(child_pids or []),
        "poll_window_seconds": int(poll_window_seconds),
        "progress_path": str(progress_file or ""),
        "stdout_path": str(stdout_file or ""),
        "stderr_path": str(stderr_file or ""),
        "artifact_dir": str(artifact_path or ""),
        "artifact_mtime": str(status.get("artifact_mtime") or ""),
        "artifact_scan_truncated": bool(status.get("artifact_scan_truncated")),
        "artifact_summary": _artifact_summary(artifact_path),
        "elapsed_seconds": status.get("elapsed_seconds"),
        "estimated_remaining_seconds": status.get("estimated_remaining_seconds"),
        "eta_at": str(status.get("eta_at") or ""),
        "eta_status": str(status.get("eta_status") or ""),
        "eta_no_eta_reason": eta_no_eta_reason,
        "progress_percent": status.get("progress_percent"),
        "current_step": status.get("current_step"),
        "total_steps": status.get("total_steps"),
        "updated_at": str(status.get("updated_at") or ""),
        "decision": str(status.get("decision") or ""),
        "stdout_tail": list(status.get("last_log_lines") or []),
        "stderr_tail": list(status.get("last_error_lines") or []),
        "final_verification": str(final_verification or ""),
        "observed_at": _utc_now().isoformat(),
    }
    return event


def append_trace_event(trace_path: str | Path, event: dict[str, Any], *, task: str = "") -> dict[str, Any]:
    path = Path(trace_path)
    payload = _load_json(path)
    if not payload or "_load_error" in payload:
        payload = {
            "schema_version": 1,
            "task": task,
            "planned_steps": [],
            "events": [],
            "final_state": {
                "completed": False,
                "skipped_steps": [],
                "unresolved_blockers": [],
                "user_nudges": [],
                "verification": [],
            },
        }
    if task and not payload.get("task"):
        payload["task"] = task
    events = payload.get("events")
    if not isinstance(events, list):
        events = []
        payload["events"] = events
    events.append(event)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _print_payload(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if "powershell_template" in payload:
        print(payload["powershell_template"])
        return
    if payload.get("status") == "blocked":
        print(f"status=blocked reason={payload.get('reason', '')} decision={payload.get('decision', '')}")
        return
    print(
        "pid={pid} alive={pid_alive} elapsed={elapsed_seconds}s progress={progress_percent}% "
        "eta={estimated_remaining_seconds}s eta_status={eta_status} decision={decision}".format(**payload)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Brain long-task wait/status helper.")
    sub = parser.add_subparsers(dest="command", required=True)

    template = sub.add_parser("template", help="Print the required PowerShell long-task wait template.")
    template.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    template.add_argument("--json", action="store_true")

    status = sub.add_parser("status", help="Read PID, logs, progress, artifacts, and ETA.")
    status.add_argument("--pid", type=int, default=0)
    status.add_argument("--project-id", default="")
    status.add_argument("--run-id", default="")
    status.add_argument("--progress", default="")
    status.add_argument("--stdout", default="")
    status.add_argument("--stderr", default="")
    status.add_argument("--artifact-dir", default="")
    status.add_argument("--stale-after-seconds", type=int, default=DEFAULT_STALE_AFTER_SECONDS)
    status.add_argument("--allow-cross-project", action="store_true")
    status.add_argument("--lease-id", default="")
    status.add_argument("--workspace-root", default="")
    status.add_argument("--json", action="store_true")

    wait_once = sub.add_parser("wait-once", help="Run one Wait-Process window, then report status.")
    wait_once.add_argument("--pid", type=int, required=True)
    wait_once.add_argument("--project-id", default="")
    wait_once.add_argument("--run-id", default="")
    wait_once.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    wait_once.add_argument("--progress", default="")
    wait_once.add_argument("--stdout", default="")
    wait_once.add_argument("--stderr", default="")
    wait_once.add_argument("--artifact-dir", default="")
    wait_once.add_argument("--stale-after-seconds", type=int, default=DEFAULT_STALE_AFTER_SECONDS)
    wait_once.add_argument("--allow-cross-project", action="store_true")
    wait_once.add_argument("--lease-id", default="")
    wait_once.add_argument("--workspace-root", default="")
    wait_once.add_argument("--json", action="store_true")

    trace_poll = sub.add_parser("trace-poll", help="Build and optionally append a structured long-task poll trace event.")
    trace_poll.add_argument("--trace-json", default="")
    trace_poll.add_argument("--task", default="")
    trace_poll.add_argument("--project-id", default="")
    trace_poll.add_argument("--run-id", default="")
    trace_poll.add_argument("--step-id", default="")
    trace_poll.add_argument("--run-tag", default="")
    trace_poll.add_argument("--pid", type=int, default=0)
    trace_poll.add_argument("--child-pids", default="")
    trace_poll.add_argument("--poll-window-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    trace_poll.add_argument("--progress", default="")
    trace_poll.add_argument("--stdout", default="")
    trace_poll.add_argument("--stderr", default="")
    trace_poll.add_argument("--artifact-dir", default="")
    trace_poll.add_argument("--stale-after-seconds", type=int, default=DEFAULT_STALE_AFTER_SECONDS)
    trace_poll.add_argument("--final-verification", default="")
    trace_poll.add_argument("--allow-cross-project", action="store_true")
    trace_poll.add_argument("--lease-id", default="")
    trace_poll.add_argument("--workspace-root", default="")
    trace_poll.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "template":
        _print_payload(build_template(timeout_seconds=args.timeout), as_json=bool(args.json))
        return 0
    if args.command == "wait-once":
        namespace = validate_project_namespace(
            project_id=str(getattr(args, "project_id", "") or ""),
            run_id=str(getattr(args, "run_id", "") or ""),
            paths=[
                path
                for path in (
                    str(getattr(args, "progress", "") or ""),
                    str(getattr(args, "stdout", "") or ""),
                    str(getattr(args, "stderr", "") or ""),
                    str(getattr(args, "artifact_dir", "") or ""),
                )
                if path
            ],
            allow_cross_project=bool(getattr(args, "allow_cross_project", False)),
            lease_id=str(getattr(args, "lease_id", "") or ""),
            workspace_root=str(getattr(args, "workspace_root", "") or "") or None,
        )
        if namespace.get("status") == "blocked":
            payload = {
                "schema_version": 1,
                "status": "blocked",
                "reason": namespace.get("reason") or "project_namespace_violation",
                "project_id": str(getattr(args, "project_id", "") or ""),
                "run_id": str(getattr(args, "run_id", "") or ""),
                "namespace": namespace,
                "pid": int(getattr(args, "pid", 0) or 0),
            }
            _print_payload(payload, as_json=bool(getattr(args, "json", False)))
            return 2
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"Wait-Process -Id {int(args.pid)} -Timeout {int(args.timeout)} -ErrorAction SilentlyContinue",
            ],
            check=False,
        )
    if args.command == "trace-poll":
        trace_json = str(getattr(args, "trace_json", "") or "")
        trace_namespace = validate_project_namespace(
            project_id=str(getattr(args, "project_id", "") or ""),
            run_id=str(getattr(args, "run_id", "") or ""),
            paths=[trace_json] if trace_json else [],
            allow_cross_project=bool(getattr(args, "allow_cross_project", False)),
            lease_id=str(getattr(args, "lease_id", "") or ""),
            workspace_root=str(getattr(args, "workspace_root", "") or "") or None,
        )
        if trace_namespace.get("status") == "blocked":
            payload = {
                "schema_version": 1,
                "status": "blocked",
                "reason": trace_namespace.get("reason") or "project_namespace_violation",
                "project_id": str(getattr(args, "project_id", "") or ""),
                "run_id": str(getattr(args, "run_id", "") or ""),
                "namespace": trace_namespace,
                "trace_json": trace_json,
            }
            _print_payload(payload, as_json=bool(getattr(args, "json", False)))
            return 2
        event = build_trace_event(
            task=str(getattr(args, "task", "") or ""),
            project_id=str(getattr(args, "project_id", "") or ""),
            run_id=str(getattr(args, "run_id", "") or ""),
            step_id=str(getattr(args, "step_id", "") or ""),
            run_tag=str(getattr(args, "run_tag", "") or ""),
            pid=int(getattr(args, "pid", 0) or 0),
            child_pids=_parse_child_pids(str(getattr(args, "child_pids", "") or "")),
            poll_window_seconds=int(getattr(args, "poll_window_seconds", DEFAULT_TIMEOUT_SECONDS)),
            progress_path=str(getattr(args, "progress", "") or "") or None,
            stdout_path=str(getattr(args, "stdout", "") or "") or None,
            stderr_path=str(getattr(args, "stderr", "") or "") or None,
            artifact_dir=str(getattr(args, "artifact_dir", "") or "") or None,
            stale_after_seconds=int(getattr(args, "stale_after_seconds", DEFAULT_STALE_AFTER_SECONDS)),
            final_verification=str(getattr(args, "final_verification", "") or ""),
            allow_cross_project=bool(getattr(args, "allow_cross_project", False)),
            lease_id=str(getattr(args, "lease_id", "") or ""),
            workspace_root=str(getattr(args, "workspace_root", "") or "") or None,
        )
        if trace_json:
            append_trace_event(trace_json, event, task=str(getattr(args, "task", "") or ""))
        _print_payload({"status": "ok", "event": event, "trace_json": trace_json}, as_json=bool(getattr(args, "json", False)))
        return 0
    payload = build_status(
        pid=int(getattr(args, "pid", 0) or 0),
        project_id=str(getattr(args, "project_id", "") or ""),
        run_id=str(getattr(args, "run_id", "") or ""),
        progress_path=str(getattr(args, "progress", "") or "") or None,
        stdout_path=str(getattr(args, "stdout", "") or "") or None,
        stderr_path=str(getattr(args, "stderr", "") or "") or None,
        artifact_dir=str(getattr(args, "artifact_dir", "") or "") or None,
        stale_after_seconds=int(getattr(args, "stale_after_seconds", DEFAULT_STALE_AFTER_SECONDS)),
        allow_cross_project=bool(getattr(args, "allow_cross_project", False)),
        lease_id=str(getattr(args, "lease_id", "") or ""),
        workspace_root=str(getattr(args, "workspace_root", "") or "") or None,
    )
    _print_payload(payload, as_json=bool(getattr(args, "json", False)))
    return 2 if payload.get("status") == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
