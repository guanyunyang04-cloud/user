from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from daily_research.continuous_policy.runtime import now_iso, read_json, write_json


def _cuda_memory_allocated_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated() / (1024.0 * 1024.0))
    except Exception:
        return 0.0
    return 0.0


class JsonlProgressSink:
    def __init__(self, path: str | Path | None, *, run_tag: str = "", stage: str = "") -> None:
        self.path = Path(path) if path else None
        self.run_tag = str(run_tag or "")
        self.stage = str(stage or "")
        self._started_monotonic = time.monotonic()

    @property
    def latest_path(self) -> Path | None:
        if self.path is None:
            return None
        return self.path.with_name("protocol_progress.json")

    def emit(self, event: str, **payload: Any) -> dict[str, Any]:
        if self.path is None:
            return {}
        progress_event: dict[str, Any] = {
            "run_tag": self.run_tag,
            "stage": self.stage,
            "event": str(event),
            "updated_at": now_iso(),
            "elapsed_seconds": round(time.monotonic() - self._started_monotonic, 3),
            "pid": os.getpid(),
            "epoch": 0,
            "batch_index": 0,
            "completed_epochs": 0,
            "amp_enabled": False,
            "device": "",
            "cuda_memory_allocated_mb": _cuda_memory_allocated_mb(),
            "train_seconds": 0.0,
            "validation_seconds": 0.0,
        }
        progress_event.update(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(progress_event, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
        latest_path = self.latest_path
        if latest_path is not None:
            write_json(latest_path, progress_event)
        return progress_event

    def latest(self) -> dict[str, Any]:
        latest_path = self.latest_path
        if latest_path is None:
            return {}
        return read_json(latest_path)


def summarize_progress_log(path: str | Path) -> dict[str, Any]:
    progress_path = Path(path)
    if not progress_path.exists():
        return {
            "exists": False,
            "event_count": 0,
            "last_event": {},
            "last_epoch": 0,
            "completed_epochs": 0,
            "stale_seconds": 0.0,
        }
    events: list[dict[str, Any]] = []
    for line in progress_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    last_event = events[-1] if events else {}
    stale_seconds = max(0.0, time.time() - progress_path.stat().st_mtime)
    return {
        "exists": True,
        "path": str(progress_path.resolve()),
        "event_count": int(len(events)),
        "last_event": last_event,
        "last_epoch": int(last_event.get("epoch", 0) or 0),
        "completed_epochs": int(last_event.get("completed_epochs", 0) or 0),
        "last_updated_at": str(last_event.get("updated_at", "") or ""),
        "stale_seconds": float(stale_seconds),
    }
