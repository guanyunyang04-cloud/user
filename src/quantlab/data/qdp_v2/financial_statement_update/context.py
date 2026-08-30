"""Financial Statement Update: context responsibilities."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data.core.paths import qdp_paths
from quantlab.data.qdp_v2.manifest import (
    atomic_write_json,
    utc_now,
)
from quantlab.data.qdp_v2.research_event_update.context import _assert_credential_free

from .config import (
    END_DATE,
    MAX_REPORT_PERIOD,
    START_DATE,
    UPDATE_ID,
    StatementSpec,
)


def _workspace(value: str | Path | None) -> Path:
    return Path(value or Path.cwd()).resolve()


def _runtime(workspace: Path) -> Path:
    path = (qdp_paths(workspace).data_dir / "qdp_runtime" / UPDATE_ID).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def _report_periods() -> list[str]:
    return [item.strftime("%Y%m%d") for item in pd.date_range("2010-03-31", MAX_REPORT_PERIOD, freq="QE-DEC")]


def _raw_path(workspace: Path, spec: StatementSpec, period: str) -> Path:
    return _runtime(workspace) / "raw" / spec.name / f"period={period}.parquet"


def _normalized_path(workspace: Path, spec: StatementSpec, period: str) -> Path:
    return _runtime(workspace) / "normalized" / spec.name / f"period={period}.parquet"


def _provider_publish_date(frame: pd.DataFrame) -> pd.Series:
    actual = frame.get("f_ann_date", pd.Series("", index=frame.index))
    announced = frame.get("ann_date", pd.Series("", index=frame.index))
    actual_text = actual.fillna("").astype(str).str.strip()
    return pd.to_datetime(actual.where(actual_text.ne(""), announced), errors="coerce")


def _in_scope_provider_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    publish = _provider_publish_date(frame)
    future = int(publish.gt(pd.Timestamp(END_DATE)).sum())
    valid = publish.between(START_DATE, END_DATE)
    return frame.loc[valid].reset_index(drop=True), future
