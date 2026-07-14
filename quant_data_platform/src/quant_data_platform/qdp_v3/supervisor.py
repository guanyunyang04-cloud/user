from __future__ import annotations

import os
import threading
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from quant_data_platform.core.json_io import read_json
from quant_data_platform.qdp_v3.manifest import atomic_write_json, utc_now
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout, qdp_v3_paths


HEARTBEAT_INTERVAL_SECONDS = 30.0
STALE_AFTER_SECONDS = 300.0


def _safe_job_id(job_id: str) -> str:
    value = "".join(
        character if character.isalnum() or character in "._=-" else "_"
        for character in str(job_id or "")
    )
    if not value or value in {".", ".."}:
        raise ValueError(f"invalid_supervised_job_id:{job_id!r}")
    return value


def _lock_file(handle: Any, *, nonblocking: bool) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        mode = msvcrt.LK_NBLCK if nonblocking else msvcrt.LK_LOCK
        msvcrt.locking(handle.fileno(), mode, 1)
        return
    import fcntl  # pragma: no cover - Windows is the production platform.

    mode = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
    fcntl.flock(handle.fileno(), mode)


def _unlock_file(handle: Any) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl  # pragma: no cover - Windows is the production platform.

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class JobSupervisor(AbstractContextManager["JobSupervisor"]):
    """Hold a process-owned job lease and publish an independent heartbeat."""

    def __init__(
        self,
        job_id: str,
        *,
        workspace_root: str | Path | None = None,
        heartbeat_interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        self.job_id = _safe_job_id(job_id)
        self.workspace_root = workspace_root
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self._handle: Any | None = None
        self._started_at = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def lock_path(self) -> Path:
        return qdp_v3_paths(self.workspace_root).jobs / "locks" / f"{self.job_id}.lock"

    @property
    def heartbeat_path(self) -> Path:
        return qdp_v3_paths(self.workspace_root).jobs / "heartbeats" / f"{self.job_id}.json"

    def __enter__(self) -> "JobSupervisor":
        ensure_qdp_v3_layout(self.workspace_root)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+b")
        if self.lock_path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        try:
            _lock_file(handle, nonblocking=True)
        except OSError as exc:
            handle.close()
            raise RuntimeError(f"qdp_v3_job_lock_held:{self.job_id}") from exc
        self._handle = handle
        self._started_at = utc_now()
        self._write_heartbeat(active=True)
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"qdp-v3-heartbeat-{self.job_id[:32]}",
            daemon=True,
        )
        self._thread.start()
        return self

    def _write_heartbeat(self, *, active: bool, stopped_at: str = "") -> None:
        payload = {
            "job_id": self.job_id,
            "pid": os.getpid(),
            "active": bool(active),
            "started_at": self._started_at,
            "heartbeat_at": utc_now(),
            "interval_seconds": self.heartbeat_interval_seconds,
            "lock_path": str(self.lock_path.resolve()),
        }
        if stopped_at:
            payload["stopped_at"] = stopped_at
        atomic_write_json(self.heartbeat_path, payload)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_interval_seconds):
            self._write_heartbeat(active=True)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.heartbeat_interval_seconds))
        self._write_heartbeat(active=False, stopped_at=utc_now())
        if self._handle is not None:
            try:
                _unlock_file(self._handle)
            finally:
                self._handle.close()
                self._handle = None


def supervise_job(
    job_id: str,
    *,
    workspace_root: str | Path | None = None,
    heartbeat_interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
) -> JobSupervisor:
    return JobSupervisor(
        job_id,
        workspace_root=workspace_root,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _job_last_seen(job: Mapping[str, Any], heartbeat: Mapping[str, Any]) -> datetime | None:
    candidates = (
        heartbeat.get("heartbeat_at") if heartbeat.get("active") else "",
        job.get("updated_at"),
        job.get("started_at"),
    )
    parsed = [_parse_timestamp(value) for value in candidates]
    return max((value for value in parsed if value is not None), default=None)


def _try_recovery_lock(path: Path) -> Any | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    try:
        _lock_file(handle, nonblocking=True)
    except OSError:
        handle.close()
        return None
    return handle


def requeue_stale_jobs(
    *,
    workspace_root: str | Path | None = None,
    stale_after_seconds: float = STALE_AFTER_SECONDS,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Recover stale JSON jobs only after proving that no process owns the lock."""

    paths = ensure_qdp_v3_layout(workspace_root)
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    recovered: list[dict[str, Any]] = []
    locked: list[str] = []
    fresh: list[str] = []
    for job_path in sorted(paths.jobs.glob("*.json")):
        job = read_json(job_path)
        if str(job.get("status", "") or "") != "running":
            continue
        job_id = _safe_job_id(str(job.get("job_id", "") or job.get("run_id", "") or job_path.stem))
        heartbeat_path = paths.jobs / "heartbeats" / f"{job_id}.json"
        heartbeat = read_json(heartbeat_path)
        last_seen = _job_last_seen(job, heartbeat)
        age_seconds = float("inf") if last_seen is None else max(0.0, (current - last_seen).total_seconds())
        if age_seconds <= float(stale_after_seconds):
            fresh.append(job_id)
            continue
        lock_path = paths.jobs / "locks" / f"{job_id}.lock"
        handle = _try_recovery_lock(lock_path)
        if handle is None:
            locked.append(job_id)
            continue
        try:
            latest = read_json(job_path)
            latest_heartbeat = read_json(heartbeat_path)
            latest_seen = _job_last_seen(latest, latest_heartbeat)
            latest_age = float("inf") if latest_seen is None else max(0.0, (current - latest_seen).total_seconds())
            if str(latest.get("status", "") or "") != "running" or latest_age <= float(stale_after_seconds):
                fresh.append(job_id)
                continue
            tasks = dict(latest.get("tasks", {}) or {})
            reset_count = 0
            for task_id, raw_state in tasks.items():
                if not isinstance(raw_state, Mapping):
                    continue
                task_state = dict(raw_state)
                if str(task_state.get("status", "") or "") == "running":
                    task_state["status"] = "pending"
                    task_state["error_code"] = "stale_worker_requeued"
                    task_state["error"] = "worker heartbeat exceeded 300 seconds and no OS lock was held"
                    task_state["updated_at"] = utc_now()
                    tasks[str(task_id)] = task_state
                    reset_count += 1
            latest["tasks"] = tasks
            latest["status"] = "interrupted_recoverable"
            latest["interrupted_at"] = utc_now()
            latest["interruption_reason"] = "stale_heartbeat_without_os_lock"
            latest["updated_at"] = utc_now()
            atomic_write_json(job_path, latest)
            atomic_write_json(
                heartbeat_path,
                {
                    "job_id": job_id,
                    "active": False,
                    "heartbeat_at": utc_now(),
                    "recovered_at": utc_now(),
                    "recovery_reason": "stale_heartbeat_without_os_lock",
                },
            )
            recovered.append({"job_id": job_id, "task_count_requeued": reset_count})
        finally:
            try:
                _unlock_file(handle)
            finally:
                handle.close()
    return {
        "status": "completed",
        "stale_after_seconds": float(stale_after_seconds),
        "recovered_count": len(recovered),
        "recovered": recovered,
        "locked_count": len(locked),
        "locked_jobs": locked,
        "fresh_running_count": len(set(fresh)),
        "fresh_running_jobs": sorted(set(fresh)),
    }
