"""Tushare Extended Backfill: context responsibilities."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from quantlab.core.io import sha256_file
from quantlab.data.core.json_io import json_safe
from quantlab.data.core.paths import qdp_paths
from quantlab.data.domains.contracts import DataDomain
from quantlab.data.qdp_v2.manifest import (
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    utc_now,
)
from quantlab.data.qdp_v2.provider_credentials import (
    tushare_credential_values,
)
from quantlab.data.qdp_v2.status import active_dataset_map

from .config import (
    BURN_IN_START,
    END_DATE,
    EXPECTED_FACTOR_FIELD_COUNT,
    EXPECTED_FACTOR_SCHEMA_HASH,
    FORBIDDEN_YEAR,
    RESEARCH_START,
    UPDATE_ID,
    EndpointSpec,
    TushareExtendedBackfillError,
)

_sha256 = sha256_file


def _workspace(value: str | Path | None) -> Path:
    return Path(value or Path.cwd()).resolve()


def _runtime(workspace: Path) -> Path:
    path = (qdp_paths(workspace).data_dir / "qdp_runtime" / UPDATE_ID).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _state_path(workspace: Path) -> Path:
    return _runtime(workspace) / "state.json"


def _credential_values() -> tuple[str, ...]:
    return tushare_credential_values()


def _assert_credential_free(payload: Any) -> None:
    encoded = json.dumps(json_safe(payload), ensure_ascii=False, default=str)
    for secret in _credential_values():
        if secret and secret in encoded:
            raise TushareExtendedBackfillError("credential_persistence_blocked")


def _read_state(workspace: Path) -> dict[str, Any]:
    path = _state_path(workspace)
    if not path.is_file():
        return {
            "update_id": UPDATE_ID,
            "status": "pending",
            "burn_in_start": BURN_IN_START,
            "research_start": RESEARCH_START,
            "end_date": END_DATE,
            "training_performed": False,
        }
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_state(workspace: Path, state: Mapping[str, Any]) -> None:
    payload = {**dict(state), "updated_at": utc_now()}
    _assert_credential_free(payload)
    atomic_write_json(_state_path(workspace), payload)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(json_safe(payload), sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _provider_cache_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.select_dtypes(include=["object"]).columns:
        output[column] = output[column].map(
            lambda value: (
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (list, tuple, dict))
                else ""
                if value is None
                else str(value)
            )
        )
    return output


def _active_paths(workspace: Path, domain: str) -> list[Path]:
    root = qdp_v2_root(workspace)
    datasets = active_dataset_map(read_active_manifest(root))
    dataset_id = datasets.get(domain, "")
    manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
    if manifest_path is None:
        raise TushareExtendedBackfillError(f"active_domain_missing:{domain}")
    manifest = read_dataset_manifest(manifest_path)
    paths = [resolve_manifest_path(item.path, root=root) for item in manifest.shards]
    if not paths or not all(path.is_file() for path in paths):
        raise TushareExtendedBackfillError(f"active_domain_shards_missing:{domain}")
    return paths


def _scan(paths: Sequence[Path]) -> str:
    literals = ",".join("'" + str(path).replace("'", "''") + "'" for path in paths)
    return f"read_parquet([{literals}], union_by_name=true)"


def _open_dates(workspace: Path) -> list[str]:
    paths = _active_paths(workspace, DataDomain.TRADING_CALENDAR)
    with duckdb.connect() as connection:
        rows = connection.execute(
            f"""
            SELECT DISTINCT cast(trade_date AS VARCHAR)
            FROM {_scan(paths)}
            WHERE is_open=true
              AND cast(trade_date AS VARCHAR) BETWEEN '{BURN_IN_START}' AND '{END_DATE}'
            ORDER BY 1
            """
        ).fetchall()
    dates = [str(row[0])[:10] for row in rows]
    if not dates or dates[0] < BURN_IN_START or dates[-1] > END_DATE:
        raise TushareExtendedBackfillError("open_date_inventory_invalid")
    if any(date.startswith(str(FORBIDDEN_YEAR)) for date in dates):
        raise TushareExtendedBackfillError("forbidden_2026_date_planned")
    return dates


def _schema_hash(columns: Sequence[str]) -> str:
    return hashlib.sha256(",".join(str(item) for item in columns).encode()).hexdigest()


def _validate_schema(spec: EndpointSpec, columns: Sequence[str]) -> str:
    actual = tuple(str(item) for item in columns)
    digest = _schema_hash(actual)
    if spec.name == "stk-factor-pro":
        if len(actual) != EXPECTED_FACTOR_FIELD_COUNT or digest != EXPECTED_FACTOR_SCHEMA_HASH:
            raise TushareExtendedBackfillError(f"factor_schema_changed:count={len(actual)}:hash={digest}")
    elif actual != spec.fields:
        raise TushareExtendedBackfillError(f"endpoint_schema_changed:{spec.name}:actual={actual}")
    return digest
