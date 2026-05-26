from __future__ import annotations

import json
import os
import tempfile
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
RUNTIME_ROOT = PROJECT_ROOT / "output" / "execution_app"
JOBS_ROOT = RUNTIME_ROOT / "jobs"
STATE_PATH = RUNTIME_ROOT / "runtime_state.json"
EVENTS_PATH = RUNTIME_ROOT / "events.jsonl"
LOCK_PATH = RUNTIME_ROOT / "execution_app.lock"
SCHEMA_VERSION = 1
RECENT_JOB_LIMIT = 24


class ExecutionAppLockError(RuntimeError):
    pass


@dataclass(frozen=True)
class JobPaths:
    job_id: str
    job_root: Path
    metadata_path: Path
    stdout_path: Path
    stderr_path: Path


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_runtime_layout() -> None:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)


def _default_state() -> dict[str, Any]:
    return {
        "app_name": "daily_research_execution_app",
        "schema_version": SCHEMA_VERSION,
        "updated_at": "",
        "lock": {},
        "current_job": {},
        "recent_jobs": [],
        "last_success_by_task": {},
        "last_failure_by_task": {},
        "scheduler": {
            "enabled": True,
            "post_close_time": "15:30",
            "timezone": "Asia/Shanghai",
            "last_auto_refresh": {},
            "last_tick": {},
        },
    }


def _write_text_atomic(path: Path, text: str) -> None:
    ensure_runtime_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        temp_path.replace(path)
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


def read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2))


def load_runtime_state() -> dict[str, Any]:
    state = _default_state()
    loaded = read_json_file(STATE_PATH)
    if loaded:
        state.update(loaded)
    state["schema_version"] = SCHEMA_VERSION
    return state


def write_runtime_state(state: dict[str, Any]) -> None:
    merged = _default_state()
    merged.update(state if isinstance(state, dict) else {})
    merged["updated_at"] = now_iso()
    write_json_file(STATE_PATH, merged)


def append_event(event_type: str, **payload: Any) -> None:
    ensure_runtime_layout()
    event = {"timestamp": now_iso(), "event_type": str(event_type)}
    event.update(payload)
    with EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def new_job_id(task_name: str) -> str:
    prefix = str(task_name or "job").strip().replace(" ", "_").replace("/", "_")
    return f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}"


def build_job_paths(job_id: str) -> JobPaths:
    ensure_runtime_layout()
    job_root = JOBS_ROOT / job_id
    job_root.mkdir(parents=True, exist_ok=True)
    return JobPaths(
        job_id=job_id,
        job_root=job_root,
        metadata_path=job_root / "metadata.json",
        stdout_path=job_root / "stdout.log",
        stderr_path=job_root / "stderr.log",
    )


def load_job_metadata(job_id: str) -> dict[str, Any]:
    return read_json_file(JOBS_ROOT / job_id / "metadata.json")


def _trim_recent_jobs(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [entry for entry in entries if isinstance(entry, dict) and str(entry.get("job_id", "")).strip()]
    return normalized[:RECENT_JOB_LIMIT]


def _update_recent_jobs(state: dict[str, Any], summary: dict[str, Any]) -> None:
    recent = [entry for entry in state.get("recent_jobs", []) if isinstance(entry, dict)]
    recent = [entry for entry in recent if str(entry.get("job_id", "")) != str(summary.get("job_id", ""))]
    recent.insert(0, summary)
    state["recent_jobs"] = _trim_recent_jobs(recent)


def create_job_record(
    *,
    task_name: str,
    command: list[str],
    cwd: Path,
    python_executable: str,
    passthrough_args: list[str],
    job_label: str = "",
    resumed_from_job_id: str = "",
) -> JobPaths:
    job_id = new_job_id(task_name)
    counter = 1
    while (JOBS_ROOT / job_id).exists():
        job_id = f"{new_job_id(task_name)}_{counter:02d}"
        counter += 1
    job_paths = build_job_paths(job_id)
    metadata = {
        "job_id": job_paths.job_id,
        "task_name": str(task_name),
        "job_label": str(job_label or ""),
        "status": "queued",
        "created_at": now_iso(),
        "started_at": "",
        "completed_at": "",
        "cwd": str(Path(cwd).resolve()),
        "python_executable": str(python_executable),
        "command_argv": [str(item) for item in command],
        "command_display": " ".join(str(item) for item in command),
        "passthrough_args": [str(item) for item in passthrough_args],
        "resumed_from_job_id": str(resumed_from_job_id or ""),
        "stdout_log": str(job_paths.stdout_path.resolve()),
        "stderr_log": str(job_paths.stderr_path.resolve()),
        "exit_code": None,
        "business_status": "",
        "runner_status": "",
        "artifact_status": "",
        "artifact_paths": {},
        "evidence_paths": {},
        "runner_warnings": [],
        "last_heartbeat_at": "",
        "last_output_stream": "",
        "last_output_line": "",
    }
    write_json_file(job_paths.metadata_path, metadata)
    append_event("job_created", job_id=job_paths.job_id, task_name=task_name, job_label=job_label)
    state = load_runtime_state()
    _update_recent_jobs(
        state,
        {
            "job_id": job_paths.job_id,
            "task_name": task_name,
            "job_label": job_label,
            "status": "queued",
            "created_at": metadata["created_at"],
        },
    )
    write_runtime_state(state)
    return job_paths


def update_job_metadata(job_paths: JobPaths, **patch: Any) -> dict[str, Any]:
    metadata = read_json_file(job_paths.metadata_path)
    metadata.update(patch)
    write_json_file(job_paths.metadata_path, metadata)
    return metadata


def mark_job_started(job_paths: JobPaths, *, lock_payload: dict[str, Any]) -> None:
    started_at = now_iso()
    metadata = update_job_metadata(
        job_paths,
        status="running",
        started_at=started_at,
        last_heartbeat_at=started_at,
    )
    state = load_runtime_state()
    state["lock"] = dict(lock_payload)
    state["current_job"] = {
        "job_id": job_paths.job_id,
        "task_name": str(metadata.get("task_name", "")),
        "job_label": str(metadata.get("job_label", "")),
        "status": "running",
        "started_at": started_at,
    }
    _update_recent_jobs(
        state,
        {
            "job_id": job_paths.job_id,
            "task_name": str(metadata.get("task_name", "")),
            "job_label": str(metadata.get("job_label", "")),
            "status": "running",
            "started_at": started_at,
        },
    )
    write_runtime_state(state)
    append_event("job_started", job_id=job_paths.job_id, task_name=str(metadata.get("task_name", "")))


def mark_job_heartbeat(job_paths: JobPaths, *, stream_name: str, line: str) -> None:
    clean_line = str(line or "").strip()
    payload = {
        "last_heartbeat_at": now_iso(),
        "last_output_stream": str(stream_name or ""),
        "last_output_line": clean_line[:500],
    }
    update_job_metadata(job_paths, **payload)
    state = load_runtime_state()
    current_job = dict(state.get("current_job", {})) if isinstance(state.get("current_job"), dict) else {}
    if str(current_job.get("job_id", "")) == job_paths.job_id:
        current_job.update(payload)
        current_job["status"] = "running"
        state["current_job"] = current_job
        write_runtime_state(state)


def mark_job_finished(
    job_paths: JobPaths,
    *,
    status: str,
    exit_code: int,
    summary_note: str = "",
    business_status: str = "",
    runner_status: str = "",
    artifact_status: str = "",
    artifact_paths: dict[str, Any] | None = None,
    evidence_paths: dict[str, Any] | None = None,
    runner_warnings: list[str] | None = None,
) -> dict[str, Any]:
    completed_at = now_iso()
    metadata = update_job_metadata(
        job_paths,
        status=str(status),
        exit_code=int(exit_code),
        completed_at=completed_at,
        summary_note=str(summary_note or ""),
        business_status=str(business_status or ""),
        runner_status=str(runner_status or ""),
        artifact_status=str(artifact_status or ""),
        artifact_paths=artifact_paths or {},
        evidence_paths=evidence_paths or artifact_paths or {},
        runner_warnings=runner_warnings or [],
        last_heartbeat_at=completed_at,
    )
    state = load_runtime_state()
    if str(state.get("current_job", {}).get("job_id", "")) == job_paths.job_id:
        state["current_job"] = {}
    summary = {
        "job_id": job_paths.job_id,
        "task_name": str(metadata.get("task_name", "")),
        "job_label": str(metadata.get("job_label", "")),
        "status": str(status),
        "started_at": str(metadata.get("started_at", "")),
        "completed_at": completed_at,
        "exit_code": int(exit_code),
        "business_status": str(metadata.get("business_status", "")),
        "runner_status": str(metadata.get("runner_status", "")),
        "artifact_status": str(metadata.get("artifact_status", "")),
    }
    _update_recent_jobs(state, summary)
    task_name = str(metadata.get("task_name", ""))
    if status == "succeeded":
        state.setdefault("last_success_by_task", {})[task_name] = summary
    else:
        state.setdefault("last_failure_by_task", {})[task_name] = summary
    write_runtime_state(state)
    append_event(
        "job_finished",
        job_id=job_paths.job_id,
        task_name=task_name,
        status=status,
        exit_code=exit_code,
    )
    return metadata


def list_recent_job_metadata(limit: int = 10) -> list[dict[str, Any]]:
    ensure_runtime_layout()
    job_dirs = [path for path in JOBS_ROOT.iterdir() if path.is_dir()]
    ordered = sorted(job_dirs, key=lambda item: item.stat().st_mtime, reverse=True)
    payloads: list[dict[str, Any]] = []
    for job_dir in ordered[: max(int(limit), 0)]:
        metadata = read_json_file(job_dir / "metadata.json")
        if metadata:
            payloads.append(metadata)
    return payloads


def tail_file(path: Path, *, lines: int = 40) -> list[str]:
    if not path.exists():
        return []
    limit = max(int(lines), 0)
    if limit == 0:
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        content = handle.readlines()
    return [line.rstrip("\n") for line in content[-limit:]]


def clear_lock_file() -> None:
    if LOCK_PATH.exists():
        LOCK_PATH.unlink()
    state = load_runtime_state()
    state["lock"] = {}
    state["current_job"] = {}
    write_runtime_state(state)
    append_event("lock_cleared")


def _lock_payload_is_stale(payload: dict[str, Any]) -> bool:
    job_id = str(payload.get("job_id", "") or "").strip() if isinstance(payload, dict) else ""
    if not job_id:
        return False
    metadata = read_json_file(JOBS_ROOT / job_id / "metadata.json")
    status = str(metadata.get("status", "") or "").strip().lower()
    return status in {"succeeded", "failed", "blocked", "abandoned", "skipped"}


def _lock_payload_is_launch_pending_for_job(payload: dict[str, Any], *, job_id: str) -> bool:
    if not isinstance(payload, dict):
        return False
    return (
        str(payload.get("status", "") or "").strip().lower() == "launch_pending"
        and str(payload.get("job_id", "") or "").strip() == str(job_id or "").strip()
    )


class ExecutionAppLock(AbstractContextManager["ExecutionAppLock"]):
    def __init__(self, *, job_id: str, task_name: str, force: bool = False) -> None:
        self.job_id = str(job_id)
        self.task_name = str(task_name)
        self.force = bool(force)
        self.payload: dict[str, Any] = {}

    def __enter__(self) -> "ExecutionAppLock":
        ensure_runtime_layout()
        if LOCK_PATH.exists():
            existing = read_json_file(LOCK_PATH)
            launch_pending_for_this_job = _lock_payload_is_launch_pending_for_job(existing, job_id=self.job_id)
            if not self.force and not launch_pending_for_this_job and not _lock_payload_is_stale(existing):
                raise ExecutionAppLockError(
                    "Execution app lock is already held. "
                    f"job_id={existing.get('job_id', '')} task_name={existing.get('task_name', '')} "
                    f"acquired_at={existing.get('acquired_at', '')}"
                )
            clear_lock_file()
        self.payload = {
            "job_id": self.job_id,
            "task_name": self.task_name,
            "pid": os.getpid(),
            "acquired_at": now_iso(),
        }
        try:
            write_json_file(LOCK_PATH, self.payload)
            state = load_runtime_state()
            state["lock"] = dict(self.payload)
            write_runtime_state(state)
            append_event("lock_acquired", **self.payload)
            return self
        except Exception:
            if LOCK_PATH.exists():
                current = read_json_file(LOCK_PATH)
                if str(current.get("job_id", "")) == self.job_id:
                    try:
                        LOCK_PATH.unlink()
                    except OSError:
                        pass
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        if LOCK_PATH.exists():
            current = read_json_file(LOCK_PATH)
            if str(current.get("job_id", "")) == self.job_id:
                LOCK_PATH.unlink()
        state = load_runtime_state()
        if str(state.get("lock", {}).get("job_id", "")) == self.job_id:
            state["lock"] = {}
            write_runtime_state(state)
        append_event("lock_released", job_id=self.job_id, task_name=self.task_name)
        return None
