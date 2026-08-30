"""Margin Eligibility Update: context responsibilities."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from quantlab.data.core.paths import qdp_paths
from quantlab.data.domains.contracts.schema import DataDomain
from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    utc_now,
)
from quantlab.data.qdp_v2.research_event_update.context import _assert_credential_free
from quantlab.data.qdp_v2.status import active_dataset_map

from .config import (
    END_DATE,
    OFFICIAL_COLUMNS,
    START_DATE,
    UPDATE_ID,
    MarginEligibilityUpdateError,
)


def _workspace(value: str | Path | None) -> Path:
    return Path(value or Path.cwd()).resolve()


def _runtime(workspace: Path) -> Path:
    path = qdp_paths(workspace).data_dir / "qdp_runtime" / UPDATE_ID
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _state_path(workspace: Path) -> Path:
    return _runtime(workspace) / "state.json"


def _read_state(workspace: Path) -> dict[str, Any]:
    path = _state_path(workspace)
    if not path.is_file():
        return {
            "update_id": UPDATE_ID,
            "status": "pending",
            "start_date": START_DATE,
            "end_date": END_DATE,
        }
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


def _raw_day_path(workspace: Path, trade_date: str) -> Path:
    return _runtime(workspace) / "raw" / f"year={trade_date[:4]}" / f"date={trade_date}.parquet"


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _number(series: pd.Series, *, multiplier: float = 1.0) -> pd.Series:
    values = pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")
    return values.astype("float64") * float(multiplier)


def _empty_official() -> pd.DataFrame:
    return pd.DataFrame(columns=OFFICIAL_COLUMNS)


def _dataset_paths(workspace: Path, domain: str) -> tuple[DatasetManifest, list[Path]]:
    root = qdp_v2_root(workspace)
    dataset_id = active_dataset_map(read_active_manifest(root)).get(domain)
    if not dataset_id:
        raise MarginEligibilityUpdateError(f"active_dataset_missing:{domain}")
    manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
    if manifest_path is None:
        raise MarginEligibilityUpdateError(f"dataset_manifest_missing:{domain}")
    manifest = read_dataset_manifest(manifest_path)
    paths = [resolve_manifest_path(item.path, root=root) for item in manifest.shards]
    return manifest, paths


def _sql_paths(paths: Sequence[Path]) -> str:
    return ",".join(f"'{path.resolve().as_posix().replace(chr(39), chr(39) * 2)}'" for path in paths)


def _trade_dates(workspace: Path) -> tuple[str, ...]:
    _, paths = _dataset_paths(workspace, DataDomain.TRADING_CALENDAR)
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            "SELECT DISTINCT trade_date FROM read_parquet(?, union_by_name=true) "
            "WHERE is_open AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
            [[str(path) for path in paths], START_DATE, END_DATE],
        ).fetchall()
    finally:
        connection.close()
    dates = tuple(str(row[0]) for row in rows)
    if len(dates) != 3_644 or any(value.startswith("2026-") for value in dates):
        raise MarginEligibilityUpdateError(f"unexpected_margin_trade_dates:{len(dates)}")
    return dates
