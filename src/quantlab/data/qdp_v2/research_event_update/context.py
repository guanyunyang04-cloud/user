"""Research Event Update: context responsibilities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import unicodedata
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

from quantlab.core.io import json_safe
from quantlab.core.io import sha256_file as _sha256
from quantlab.data.core.paths import qdp_paths
from quantlab.data.domains.contracts.schema import DataDomain
from quantlab.data.qdp_v2.manifest import (
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    utc_now,
)
from quantlab.data.qdp_v2.provider_credentials import tushare_credential_values
from quantlab.data.qdp_v2.status import active_dataset_map

from .config import (
    END_DATE,
    LEGACY_UPDATE_ID,
    REPORT_RC_PAGE_SIZE,
    START_DATE,
    UPDATE_ID,
    ResearchEventUpdateError,
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _workspace(value: str | Path | None) -> Path:
    return Path(value or Path.cwd()).resolve()


def _runtime(workspace: Path) -> Path:
    path = (qdp_paths(workspace).data_dir / "qdp_runtime" / UPDATE_ID).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _legacy_runtime(workspace: Path) -> Path:
    return (qdp_paths(workspace).data_dir / "qdp_runtime" / LEGACY_UPDATE_ID).resolve()


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


def _credential_values() -> tuple[str, ...]:
    return tushare_credential_values()


def _assert_credential_free(payload: Any) -> None:
    encoded = json.dumps(json_safe(payload), ensure_ascii=False, default=str)
    for secret in _credential_values():
        if secret and secret in encoded:
            raise ResearchEventUpdateError("credential_persistence_blocked")


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


def _stable_hash(*parts: Any) -> str:
    encoded = "\x1f".join(str(part or "") for part in parts).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _parquet_safe_provider_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Make loosely typed provider JSON deterministic before raw caching."""
    output = frame.copy()

    def encoded(value: Any) -> str:
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        return str(value)

    for column in output.select_dtypes(include=["object"]).columns:
        output[column] = output[column].map(encoded)
    return output


def _request_json(
    method: str,
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    data: Mapping[str, Any] | None = None,
    retries: int = 4,
    timeout: int = 45,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    last = ""
    headers = {"User-Agent": "Mozilla/5.0 qdp-research-event-update"}
    for attempt in range(max(1, int(retries))):
        try:
            request = session.request if session is not None else requests.request
            response = request(
                method,
                url,
                params=dict(params or {}),
                data=dict(data or {}),
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise TypeError("response payload is not an object")
            return payload
        except (requests.RequestException, TypeError, ValueError) as exc:
            last = f"{type(exc).__name__}:{str(exc)[:240]}"
            if attempt + 1 >= max(1, int(retries)):
                break
            time.sleep(min(2**attempt, 8))
    raise ResearchEventUpdateError(f"provider_request_failed:{url}:{last}")


def _active_paths(workspace: Path, domain: str) -> list[Path]:
    root = qdp_v2_root(workspace)
    active = read_active_manifest(root)
    dataset_id = active_dataset_map(active).get(domain)
    if not dataset_id:
        raise ResearchEventUpdateError(f"active_domain_missing:{domain}")
    manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
    if manifest_path is None:
        raise ResearchEventUpdateError(f"active_manifest_missing:{domain}")
    manifest = read_dataset_manifest(manifest_path)
    return [resolve_manifest_path(item.path, root=root) for item in manifest.shards]


def _identity_symbols(workspace: Path) -> list[str]:
    paths = _active_paths(workspace, DataDomain.SECURITY_IDENTITY)
    frame = pd.concat(
        [pd.read_parquet(path, columns=["current_symbol"]) for path in paths],
        ignore_index=True,
    )
    values = {
        str(value).strip().upper()
        for value in frame["current_symbol"]
        if str(value).strip().upper().endswith((".SH", ".SZ"))
    }
    return sorted(values)


def _open_dates(workspace: Path) -> np.ndarray:
    frames = [
        pd.read_parquet(path, columns=["trade_date", "is_open"])
        for path in _active_paths(workspace, DataDomain.TRADING_CALENDAR)
    ]
    frame = pd.concat(frames, ignore_index=True)
    dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    mask = frame["is_open"].astype(bool) & dates.between(START_DATE, END_DATE)
    return np.sort(dates.loc[mask].dropna().dt.normalize().unique())


def _next_open_date(values: pd.Series, open_dates: np.ndarray) -> pd.Series:
    source = pd.to_datetime(values, errors="coerce").to_numpy(dtype="datetime64[ns]")
    positions = np.searchsorted(open_dates, source, side="right")
    result = np.full(len(source), "", dtype=object)
    valid = (~pd.isna(source)) & (positions < len(open_dates))
    result[valid] = pd.to_datetime(open_dates[positions[valid]]).strftime("%Y-%m-%d")
    return pd.Series(result, index=values.index, dtype="object")


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _normalized_text(value: Any) -> str:
    raw = unicodedata.normalize("NFKC", _text(value)).lower()
    raw = re.sub(r"<[^>]+>", "", raw)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", raw)


def _normalized_institution(value: Any) -> str:
    raw = _normalized_text(value)
    suffixes = (
        "证券股份有限公司",
        "证券有限责任公司",
        "股份有限公司",
        "有限责任公司",
        "有限公司",
        "证券研究院",
        "证券研究所",
        "研究院",
        "研究所",
        "证券",
    )
    changed = True
    while changed:
        before = raw
        for suffix in suffixes:
            raw = raw.removesuffix(suffix)
        changed = raw != before
    return raw


def _json_list(values: Iterable[Any]) -> str:
    unique = sorted({_text(value) for value in values if _text(value)})
    return json.dumps(unique, ensure_ascii=False, separators=(",", ":"))


def _first(values: Iterable[Any]) -> Any:
    for value in values:
        if _text(value):
            return value
    return ""


def _numeric(value: Any) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else math.nan


def _report_source_key(
    symbol: Any,
    report_date: Any,
    title: Any,
    institution: Any,
) -> str:
    return _stable_hash(
        _text(symbol).upper(),
        _text(report_date)[:10],
        _normalized_text(title),
        _normalized_institution(institution),
    )


def _report_id(source_report_key: str) -> str:
    return f"rr_{source_report_key[:24]}"


def _report_rc_page_path(workspace: Path, report_date: str, offset: int) -> Path:
    normalized = str(report_date)[:10]
    year = int(normalized[:4])
    return (
        _runtime(workspace)
        / "raw"
        / "tushare_report_rc"
        / f"year={year}"
        / f"date={normalized}"
        / f"offset={int(offset):09d}.parquet"
    )


def _next_report_offset(row_count: int, offset: int) -> int | None:
    return int(offset) + REPORT_RC_PAGE_SIZE if int(row_count) == REPORT_RC_PAGE_SIZE else None


def _report_request_dates() -> tuple[str, ...]:
    return tuple(pd.date_range(START_DATE, END_DATE, freq="D").strftime("%Y-%m-%d"))


def _frame_schema_hash(frame: pd.DataFrame) -> str:
    payload = [{"name": str(column), "dtype": str(frame[column].dtype)} for column in frame.columns]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _report_year_source_coverage(
    year: int,
    statistics: Mapping[str, Any],
    task_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    earliest = _text(statistics.get("earliest_report_date"))
    latest = _text(statistics.get("latest_report_date"))
    raw_rows = int(statistics.get("raw_row_count", 0) or 0)
    coverage = dict(task_coverage or {})
    if coverage:
        requested = int(coverage.get("requested_date_count", 0) or 0)
        terminal = int(coverage.get("terminal_date_count", 0) or 0)
        failed = int(coverage.get("failed_date_count", 0) or 0)
        pending = int(coverage.get("pending_date_count", 0) or 0)
        if requested and terminal == requested and not failed and not pending:
            status = "complete_daily_task_ledger"
        else:
            status = "incomplete_daily_task_ledger"
    elif raw_rows == 0:
        status = "source_unavailable"
    elif earliest > f"{int(year)}-01-01" or latest < f"{int(year)}-12-31":
        status = "partial_year_span"
    else:
        status = "observed_full_year_span"
    return {
        "tushare_source_coverage_status": status,
        "tushare_source_coverage_start": earliest,
        "tushare_source_coverage_end": latest,
        "tushare_source_unavailable_is_not_zero_reports": status
        not in {"observed_full_year_span", "complete_daily_task_ledger"},
        **coverage,
    }


def _cached_report_day(
    workspace: Path,
    report_date: str,
) -> tuple[int, bool, dict[str, Any], str]:
    root = _report_rc_page_path(workspace, report_date, 0).parent
    offset = 0
    pages: dict[str, Any] = {}
    while True:
        path = root / f"offset={offset:09d}.parquet"
        if not path.is_file():
            return offset, False, pages, "pending"
        parquet = pq.ParquetFile(path)
        rows = int(parquet.metadata.num_rows)
        pages[str(offset)] = {
            "path": str(path),
            "row_count": rows,
            "sha256": _sha256(path),
            "schema_hash": hashlib.sha256(str(parquet.schema_arrow).encode("utf-8")).hexdigest(),
        }
        following = _next_report_offset(rows, offset)
        if following is None:
            status = "confirmed_empty" if not sum(int(item["row_count"]) for item in pages.values()) else "observed"
            return offset, True, pages, status
        offset = following
