"""Runtime Archive: ledger responsibilities."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from quantlab.core.io import sha256_file as _sha256
from quantlab.data.qdp_v2.manifest import atomic_write_json
from quantlab.data.qdp_v2.research_event_update.context import _assert_credential_free

from .config import (
    LEDGER_COLUMNS,
    SENSITIVE_KEYS,
    ArchiveUnit,
    RuntimeArchiveError,
)
from .discovery import (
    _archive_root,
    _runtime_root,
    _safe_name,
    _within,
)


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): ("<redacted>" if str(key).lower() in SENSITIVE_KEYS else _redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _json_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    params = payload.get("params", payload.get("request_params", {}))
    return {
        "request_status": str(payload.get("status", "") or ""),
        "request_offset": payload.get("offset", payload.get("request_offset")),
        "request_params_json": json.dumps(_redact(params), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if params
        else "",
        "response_sha256": str(payload.get("response_sha256", payload.get("response_hash", "")) or ""),
        "error_type": str(payload.get("error_type", "") or ""),
        "error_message": str(payload.get("last_error", payload.get("error", payload.get("message", ""))) or "")[:1000],
        "attempt_count": payload.get("attempt_count", payload.get("attempts")),
        "empty_confirmation_count": payload.get("empty_confirmation_count", payload.get("empty_confirmations")),
        "completed_at": str(payload.get("completed_at", "") or ""),
    }


def _ledger_record(unit: ArchiveUnit, path: Path) -> dict[str, Any]:
    stat = path.stat()
    row_count: int | None = None
    schema_hash = ""
    if path.suffix.lower() == ".parquet":
        parquet = pq.ParquetFile(path)
        row_count = int(parquet.metadata.num_rows)
        schema_hash = hashlib.sha256(str(parquet.schema_arrow).encode("utf-8")).hexdigest()
    json_meta = _json_metadata(path) if path.suffix.lower() == ".json" else {}
    return {
        "workflow": unit.workflow,
        "domain": unit.domain,
        "year": unit.year,
        "relative_path": path.relative_to(unit.workflow_root).as_posix(),
        "file_size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256(path),
        "suffix": path.suffix.lower(),
        "row_count": row_count,
        "schema_sha256": schema_hash,
        "request_status": str(json_meta.get("request_status", "")),
        "request_offset": json_meta.get("request_offset"),
        "request_params_json": str(json_meta.get("request_params_json", "")),
        "response_sha256": str(json_meta.get("response_sha256", "")),
        "error_type": str(json_meta.get("error_type", "")),
        "error_message": str(json_meta.get("error_message", "")),
        "attempt_count": json_meta.get("attempt_count"),
        "empty_confirmation_count": json_meta.get("empty_confirmation_count"),
        "completed_at": str(json_meta.get("completed_at", "")),
    }


def build_ledger(unit: ArchiveUnit) -> pd.DataFrame:
    frame = pd.DataFrame([_ledger_record(unit, path) for path in unit.files])
    if frame.empty or tuple(frame.columns) != LEDGER_COLUMNS:
        raise RuntimeArchiveError(f"runtime_archive_ledger_invalid:{unit.key}")
    if frame["relative_path"].duplicated().any():
        raise RuntimeArchiveError(f"runtime_archive_duplicate_path:{unit.key}")
    return frame.sort_values("relative_path", kind="stable").reset_index(drop=True)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _state_snapshot(workspace: Path, workflow: str) -> dict[str, Any]:
    source = _within(_runtime_root(workspace) / workflow / "state.json", _runtime_root(workspace))
    if not source.is_file():
        return {"status": "not_available"}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise RuntimeArchiveError(f"runtime_archive_state_unreadable:{workflow}:{type(exc).__name__}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeArchiveError(f"runtime_archive_state_invalid:{workflow}")
    redacted = _redact(payload)
    _assert_credential_free(redacted)
    canonical = json.dumps(redacted, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    target = _archive_root(workspace) / _safe_name(workflow) / f"state.snapshot.{digest[:16]}.json"
    if not target.is_file():
        atomic_write_json(target, redacted)
    try:
        stored = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise RuntimeArchiveError(f"runtime_archive_state_snapshot_unreadable:{workflow}") from exc
    if stored != redacted:
        raise RuntimeArchiveError(f"runtime_archive_state_snapshot_hash:{workflow}")
    return {
        "status": "preserved",
        "path": str(target),
        "sha256": _sha256(target),
        "source_path": str(source),
        "redacted": True,
    }
