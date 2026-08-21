"""Historical Intraday External Archive Repair: context responsibilities."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quantlab.data.core.paths import qdp_paths
from quantlab.data.qdp_v2.manifest import (
    atomic_write_json,
    utc_now,
)
from quantlab.data.qdp_v2.research_event_update import (
    _assert_credential_free,
)

from .config import (
    BASELINE_ID,
    REPAIR_ID,
    ExternalArchiveRepairError,
)


def _workspace(value: str | Path | None) -> Path:
    return Path(value or Path.cwd()).resolve()


def _runtime(workspace: Path) -> Path:
    path = qdp_paths(workspace).data_dir / "qdp_runtime" / REPAIR_ID
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _state_path(workspace: Path) -> Path:
    return _runtime(workspace) / "state.json"


def _read_state(workspace: Path) -> dict[str, Any]:
    path = _state_path(workspace)
    if not path.is_file():
        return {"repair_id": REPAIR_ID, "status": "pending"}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_state(workspace: Path, state: Mapping[str, Any]) -> None:
    payload = {**dict(state), "updated_at": utc_now()}
    _assert_credential_free(payload)
    last_error: PermissionError | None = None
    for attempt in range(6):
        try:
            atomic_write_json(_state_path(workspace), payload)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.10 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _baseline_manifest(workspace: Path) -> dict[str, Any]:
    path = qdp_paths(workspace).data_dir / "qdp_runtime" / BASELINE_ID / "manifest.json"
    if not path.is_file():
        raise ExternalArchiveRepairError(f"repair_baseline_missing:{path}")
    payload = dict(json.loads(path.read_text(encoding="utf-8")))
    if payload.get("status") != "frozen":
        raise ExternalArchiveRepairError("repair_baseline_not_frozen")
    if int(payload.get("read_2026_rows", -1)) != 0:
        raise ExternalArchiveRepairError("repair_baseline_read_2026_nonzero")
    return payload
