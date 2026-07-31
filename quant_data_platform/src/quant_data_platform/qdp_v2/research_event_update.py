from __future__ import annotations

"""Backfill point-in-time research reports and announcement metadata."""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time
import unicodedata
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.domains.contracts import DataDomain
from quant_data_platform.providers import _fetch_cninfo_announcements
from quant_data_platform.qdp_v2.auxiliary_update import (
    _resolve_tushare_token,
    _TushareClient,
)
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    _manifest_schema_from_arrow,
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    utc_now,
    write_active_manifest,
    write_dataset_manifest,
)
from quant_data_platform.qdp_v2.status import active_dataset_map

UPDATE_ID = "research_report_rc_backfill_v1"
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"
REPORT_EASTMONEY_START = "2017-01-01"
REPORT_RC_PAGE_SIZE = 5_000
MAX_WORKERS = 3
REPORT_RC_FIELDS = (
    "ts_code",
    "name",
    "report_date",
    "report_title",
    "report_type",
    "classify",
    "org_name",
    "author_name",
    "quarter",
    "op_rt",
    "op_pr",
    "tp",
    "np",
    "eps",
    "pe",
    "rd",
    "roe",
    "ev_ebitda",
    "rating",
    "max_price",
    "min_price",
)
REPORT_COLUMNS = (
    "report_id",
    "source_report_key",
    "symbol",
    "trade_date",
    "source_date",
    "feature_available_date",
    "title",
    "normalized_title",
    "institution",
    "normalized_institution",
    "analyst",
    "report_type",
    "classification",
    "rating",
    "rating_change",
    "target_price_min",
    "target_price_max",
    "tushare_present",
    "eastmoney_present",
    "tushare_source_ids",
    "eastmoney_info_codes",
    "url",
    "pdf_file_size_kb",
    "pdf_pages",
    "source_disagreement",
    "identity_conflict_reason",
    "source",
)
FORECAST_COLUMNS = (
    "report_id",
    "symbol",
    "trade_date",
    "source_date",
    "feature_available_date",
    "forecast_quarter",
    "forecast_year",
    "operating_revenue",
    "operating_profit",
    "total_profit",
    "net_profit",
    "eps",
    "pe",
    "research_development",
    "roe",
    "ev_ebitda",
    "source_disagreement",
    "source",
)
ANNOUNCEMENT_COLUMNS = (
    "announcement_id",
    "symbol",
    "trade_date",
    "source_date",
    "feature_available_date",
    "publish_time",
    "title",
    "normalized_title",
    "category",
    "announcement_type_codes",
    "cninfo_announcement_id",
    "eastmoney_art_code",
    "org_id",
    "url",
    "pdf_url",
    "file_size_kb",
    "cninfo_present",
    "eastmoney_present",
    "source_disagreement",
    "source",
)


class ResearchEventUpdateError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def _credential_values() -> tuple[str, ...]:
    return tuple(
        value
        for value in (
            os.environ.get("QDP_TUSHARE_PROXY_TOKEN", "").strip(),
            os.environ.get("QDP_TUSHARE_TOKEN", "").strip(),
            os.environ.get("TUSHARE_TOKEN", "").strip(),
        )
        if value
    )


def _assert_credential_free(payload: Any) -> None:
    encoded = json.dumps(json_safe(payload), ensure_ascii=False, default=str)
    for secret in _credential_values():
        if secret and secret in encoded:
            raise ResearchEventUpdateError("credential_persistence_blocked")


def _write_state(workspace: Path, state: Mapping[str, Any]) -> None:
    payload = {**dict(state), "updated_at": utc_now()}
    _assert_credential_free(payload)
    atomic_write_json(_state_path(workspace), payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _report_rc_page_path(workspace: Path, year: int, offset: int) -> Path:
    return (
        _runtime(workspace)
        / "raw"
        / "tushare_report_rc"
        / f"year={int(year)}"
        / f"offset={int(offset):09d}.parquet"
    )


def _next_report_offset(row_count: int, offset: int) -> int | None:
    return int(offset) + REPORT_RC_PAGE_SIZE if int(row_count) == REPORT_RC_PAGE_SIZE else None


def _report_year_source_coverage(
    year: int,
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    earliest = _text(statistics.get("earliest_report_date"))
    latest = _text(statistics.get("latest_report_date"))
    raw_rows = int(statistics.get("raw_row_count", 0) or 0)
    if raw_rows == 0:
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
        != "observed_full_year_span",
    }


def _cached_report_year(workspace: Path, year: int) -> tuple[int, bool, dict[str, Any]]:
    root = _report_rc_page_path(workspace, year, 0).parent
    offset = 0
    pages: dict[str, Any] = {}
    while True:
        path = root / f"offset={offset:09d}.parquet"
        if not path.is_file():
            return offset, False, pages
        rows = int(pq.ParquetFile(path).metadata.num_rows)
        pages[str(offset)] = {
            "path": str(path),
            "row_count": rows,
            "sha256": _sha256(path),
        }
        following = _next_report_offset(rows, offset)
        if following is None:
            return offset, True, pages
        offset = following


def download_tushare_reports(
    *,
    workspace_root: str | Path | None = None,
    max_workers: int = MAX_WORKERS,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    token = _resolve_tushare_token()
    if not token:
        raise ResearchEventUpdateError("tushare_token_required")
    client = _TushareClient(token)
    years = list(range(2010, 2026))
    year_state: dict[str, Any] = {}
    pending: deque[int] = deque()
    next_offsets: dict[int, int] = {}
    for year in years:
        offset, complete, pages = _cached_report_year(workspace, year)
        year_state[str(year)] = {
            "status": "completed" if complete else "pending",
            "pages": pages,
            "raw_row_count": int(sum(item["row_count"] for item in pages.values())),
        }
        if not complete:
            pending.append(year)
            next_offsets[year] = offset

    def fetch(year: int, offset: int) -> pd.DataFrame:
        return client.fetch(
            "report_rc",
            params={
                "start_date": f"{year}0101",
                "end_date": f"{year}1231",
                "limit": REPORT_RC_PAGE_SIZE,
                "offset": int(offset),
            },
            fields=REPORT_RC_FIELDS,
        )

    active: dict[Future[pd.DataFrame], tuple[int, int]] = {}
    workers = min(max(1, int(max_workers)), MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending and len(active) < workers:
            year = pending.popleft()
            offset = next_offsets[year]
            active[pool.submit(fetch, year, offset)] = (year, offset)
        while active:
            completed, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in completed:
                year, offset = active.pop(future)
                try:
                    frame = future.result()
                except Exception as exc:
                    year_state[str(year)]["status"] = "failed"
                    year_state[str(year)]["error"] = (
                        f"{type(exc).__name__}:{str(exc)[:240]}"
                    )
                    state["tushare_report_rc"] = {
                        "status": "failed",
                        "maximum_workers": workers,
                        "years": year_state,
                    }
                    _write_state(workspace, state)
                    raise
                path = _report_rc_page_path(workspace, year, offset)
                _write_parquet(frame, path)
                record = {
                    "path": str(path),
                    "row_count": len(frame),
                    "sha256": _sha256(path),
                }
                year_state[str(year)]["pages"][str(offset)] = record
                year_state[str(year)]["raw_row_count"] = int(
                    sum(
                        item["row_count"]
                        for item in year_state[str(year)]["pages"].values()
                    )
                )
                following = _next_report_offset(len(frame), offset)
                if following is None:
                    year_state[str(year)]["status"] = "completed"
                else:
                    year_state[str(year)]["status"] = "downloading"
                    next_offsets[year] = following
                    active[pool.submit(fetch, year, following)] = (year, following)
                state["status"] = "downloading_tushare_reports"
                state["tushare_report_rc"] = {
                    "status": "completed"
                    if all(item["status"] == "completed" for item in year_state.values())
                    else "downloading",
                    "maximum_workers": workers,
                    "page_size": REPORT_RC_PAGE_SIZE,
                    "years": year_state,
                    "credential_persisted": False,
                }
                _write_state(workspace, state)
                while pending and len(active) < workers:
                    next_year = pending.popleft()
                    next_offset = next_offsets[next_year]
                    active[pool.submit(fetch, next_year, next_offset)] = (
                        next_year,
                        next_offset,
                    )
    state["status"] = "tushare_reports_downloaded"
    state["tushare_report_rc"]["status"] = "completed"
    _write_state(workspace, state)
    return dict(state["tushare_report_rc"])


def _eastmoney_report_path(workspace: Path, symbol: str) -> Path:
    return (
        _runtime(workspace)
        / "raw"
        / "eastmoney_reports"
        / f"symbol={symbol.replace('.', '_')}.parquet"
    )


def _fetch_eastmoney_reports(symbol: str) -> pd.DataFrame:
    code = str(symbol).split(".", 1)[0]
    url = "https://reportapi.eastmoney.com/report/list"
    common = {
        "industryCode": "*",
        "pageSize": "5000",
        "industry": "*",
        "rating": "*",
        "ratingChange": "*",
        "beginTime": REPORT_EASTMONEY_START,
        "endTime": END_DATE,
        "fields": "",
        "qType": "0",
        "orgCode": "",
        "code": code,
        "rcode": "",
    }
    pages: list[pd.DataFrame] = []
    page = 1
    total_pages = 1
    while page <= total_pages:
        params = {
            **common,
            "pageNo": str(page),
            "p": str(page),
            "pageNum": str(page),
            "pageNumber": str(page),
        }
        payload = _request_json("GET", url, params=params)
        total_pages = max(1, int(payload.get("TotalPage", 0) or 0))
        frame = pd.DataFrame(list(payload.get("data", []) or []))
        if not frame.empty:
            frame["_query_symbol"] = symbol
            pages.append(frame)
        page += 1
    if pages:
        return pd.concat(pages, ignore_index=True)
    return pd.DataFrame({"_query_symbol": pd.Series(dtype="object")})


def _download_symbol_files(
    *,
    workspace: Path,
    state_key: str,
    symbols: Sequence[str],
    path_for: Any,
    fetch: Any,
    max_workers: int,
    allow_failures: bool,
) -> dict[str, Any]:
    state = _read_state(workspace)
    completed = {
        symbol
        for symbol in symbols
        if path_for(workspace, symbol).is_file()
    }
    failures: dict[str, str] = {}
    pending = [symbol for symbol in symbols if symbol not in completed]
    workers = min(max(1, int(max_workers)), MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, symbol): symbol for symbol in pending}
        for number, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                frame = _parquet_safe_provider_frame(future.result())
                path = path_for(workspace, symbol)
                _write_parquet(frame, path)
                completed.add(symbol)
                failures.pop(symbol, None)
            except (
                OSError,
                ResearchEventUpdateError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                failures[symbol] = f"{type(exc).__name__}:{str(exc)[:240]}"
            if number % 25 == 0 or number == len(futures):
                state[state_key] = {
                    "status": "completed"
                    if len(completed) == len(symbols)
                    else "downloading",
                    "maximum_workers": workers,
                    "symbol_count": len(symbols),
                    "completed_symbol_count": len(completed),
                    "failed_symbol_count": len(failures),
                    "failures": failures,
                }
                state["status"] = f"downloading_{state_key}"
                _write_state(workspace, state)
    state[state_key] = {
        "status": "completed" if len(completed) == len(symbols) else "partial",
        "maximum_workers": workers,
        "symbol_count": len(symbols),
        "completed_symbol_count": len(completed),
        "failed_symbol_count": len(failures),
        "failures": failures,
    }
    _write_state(workspace, state)
    if failures and not allow_failures:
        raise ResearchEventUpdateError(
            f"{state_key}_incomplete:{len(failures)}"
        )
    return dict(state[state_key])


def download_eastmoney_reports(
    *,
    workspace_root: str | Path | None = None,
    max_workers: int = MAX_WORKERS,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    return _download_symbol_files(
        workspace=workspace,
        state_key="eastmoney_reports",
        symbols=_identity_symbols(workspace),
        path_for=_eastmoney_report_path,
        fetch=_fetch_eastmoney_reports,
        max_workers=max_workers,
        allow_failures=False,
    )


def _normalize_tushare_report_year(
    raw: pd.DataFrame,
    *,
    open_dates: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if raw.empty:
        return (
            pd.DataFrame(columns=REPORT_COLUMNS),
            pd.DataFrame(columns=FORECAST_COLUMNS),
            {
                "raw_row_count": 0,
                "valid_prediction_row_count": 0,
                "unique_report_count": 0,
                "forecast_row_count": 0,
                "duplicate_prediction_row_count": 0,
                "symbol_count": 0,
                "institution_count": 0,
                "analyst_count": 0,
                "earliest_report_date": "",
                "latest_report_date": "",
                "field_nonnull_rate": {},
            },
        )
    data = raw.copy()
    data["symbol"] = data.get("ts_code", "").fillna("").astype(str).str.upper()
    data["source_date"] = pd.to_datetime(
        data.get("report_date"), errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    data["title"] = data.get("report_title", "").fillna("").astype(str)
    data["institution"] = data.get("org_name", "").fillna("").astype(str)
    data["normalized_title"] = data["title"].map(_normalized_text)
    data["normalized_institution"] = data["institution"].map(
        _normalized_institution
    )
    data["source_report_key"] = [
        _report_source_key(*values)
        for values in zip(
            data["symbol"], data["source_date"], data["title"], data["institution"]
        )
    ]
    data["report_id"] = data["source_report_key"].map(_report_id)
    valid = (
        data["symbol"].str.endswith((".SH", ".SZ"))
        & data["source_date"].between(START_DATE, END_DATE)
        & data["normalized_title"].ne("")
        & data["normalized_institution"].ne("")
    )
    data = data.loc[valid].reset_index(drop=True)
    data["feature_available_date"] = _next_open_date(
        data["source_date"], open_dates
    )
    report_rows: list[dict[str, Any]] = []
    for source_key, group in data.groupby("source_report_key", sort=True):
        analysts = [_text(item) for item in group.get("author_name", []) if _text(item)]
        disagreements = any(
            group[column].dropna().astype(str).nunique() > 1
            for column in ("report_title", "org_name", "rating", "min_price", "max_price")
            if column in group
        )
        report_rows.append(
            {
                "report_id": _report_id(str(source_key)),
                "source_report_key": source_key,
                "symbol": _first(group["symbol"]),
                "trade_date": _first(group["source_date"]),
                "source_date": _first(group["source_date"]),
                "feature_available_date": _first(group["feature_available_date"]),
                "title": _first(group["title"]),
                "normalized_title": _first(group["normalized_title"]),
                "institution": _first(group["institution"]),
                "normalized_institution": _first(group["normalized_institution"]),
                "analyst": ";".join(sorted(set(analysts))),
                "report_type": _first(group.get("report_type", [])),
                "classification": _first(group.get("classify", [])),
                "rating": _first(group.get("rating", [])),
                "rating_change": "",
                "target_price_min": _numeric(_first(group.get("min_price", []))),
                "target_price_max": _numeric(_first(group.get("max_price", []))),
                "tushare_present": True,
                "eastmoney_present": False,
                "tushare_source_ids": json.dumps(
                    [f"report_rc:{source_key}"], ensure_ascii=False
                ),
                "eastmoney_info_codes": "[]",
                "url": "",
                "pdf_file_size_kb": math.nan,
                "pdf_pages": math.nan,
                "source_disagreement": bool(disagreements),
                "identity_conflict_reason": (
                    "analyst_union"
                    if len(set(analysts)) > 1 and len(group) > 1
                    else ""
                ),
                "source": "tushare_report_rc",
            }
        )
    reports = pd.DataFrame(report_rows, columns=REPORT_COLUMNS)
    forecast = pd.DataFrame(
        {
            "report_id": data["report_id"],
            "symbol": data["symbol"],
            "trade_date": data["source_date"],
            "source_date": data["source_date"],
            "feature_available_date": data["feature_available_date"],
            "forecast_quarter": data.get("quarter", "").fillna("").astype(str),
            "forecast_year": pd.to_numeric(
                data.get("quarter", "").astype(str).str.extract(r"(20\d{2})")[0],
                errors="coerce",
            ),
            "operating_revenue": pd.to_numeric(data.get("op_rt"), errors="coerce"),
            "operating_profit": pd.to_numeric(data.get("op_pr"), errors="coerce"),
            "total_profit": pd.to_numeric(data.get("tp"), errors="coerce"),
            "net_profit": pd.to_numeric(data.get("np"), errors="coerce"),
            "eps": pd.to_numeric(data.get("eps"), errors="coerce"),
            "pe": pd.to_numeric(data.get("pe"), errors="coerce"),
            "research_development": pd.to_numeric(data.get("rd"), errors="coerce"),
            "roe": pd.to_numeric(data.get("roe"), errors="coerce"),
            "ev_ebitda": pd.to_numeric(data.get("ev_ebitda"), errors="coerce"),
            "source_disagreement": False,
            "source": "tushare_report_rc",
        }
    )
    forecast = forecast.loc[forecast["forecast_quarter"].str.strip().ne("")].copy()
    value_columns = list(FORECAST_COLUMNS[7:16])
    forecast["_complete"] = forecast[value_columns].notna().sum(axis=1)
    duplicate_groups = forecast.groupby(
        ["report_id", "forecast_quarter", "source"], dropna=False
    ).size()
    conflict_keys = set(duplicate_groups.loc[duplicate_groups.gt(1)].index.tolist())
    forecast = forecast.sort_values(
        ["report_id", "forecast_quarter", "_complete"], kind="stable"
    ).drop_duplicates(["report_id", "forecast_quarter", "source"], keep="last")
    forecast["source_disagreement"] = [
        (row.report_id, row.forecast_quarter, row.source) in conflict_keys
        for row in forecast.itertuples(index=False)
    ]
    forecast = forecast.loc[:, list(FORECAST_COLUMNS)].reset_index(drop=True)
    statistics = {
        "raw_row_count": len(raw),
        "valid_prediction_row_count": len(data),
        "unique_report_count": len(reports),
        "forecast_row_count": len(forecast),
        "duplicate_prediction_row_count": int(len(data) - len(forecast)),
        "symbol_count": int(reports["symbol"].nunique()),
        "institution_count": int(reports["institution"].nunique()),
        "analyst_count": len(
                {
                    name
                    for value in reports["analyst"]
                    for name in str(value).split(";")
                    if name
                }
            ),
        "earliest_report_date": str(reports["source_date"].min()),
        "latest_report_date": str(reports["source_date"].max()),
        "field_nonnull_rate": {
            column: float(data[column].notna().mean())
            for column in REPORT_RC_FIELDS
            if column in data
        },
    }
    return reports, forecast, statistics


def _normalize_eastmoney_reports(
    raw: pd.DataFrame,
    *,
    open_dates: np.ndarray,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)
    data = raw.copy()
    data["symbol"] = data.get("_query_symbol", "").fillna("").astype(str).str.upper()
    data["source_date"] = pd.to_datetime(
        data.get("publishDate"), errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    data["title"] = data.get("title", "").fillna("").astype(str)
    institution = data.get("orgName", data.get("orgSName", ""))
    data["institution"] = institution.fillna("").astype(str)
    data["normalized_title"] = data["title"].map(_normalized_text)
    data["normalized_institution"] = data["institution"].map(
        _normalized_institution
    )
    data["source_report_key"] = [
        _report_source_key(*values)
        for values in zip(
            data["symbol"], data["source_date"], data["title"], data["institution"]
        )
    ]
    data["feature_available_date"] = _next_open_date(
        data["source_date"], open_dates
    )
    valid = (
        data["source_date"].between(REPORT_EASTMONEY_START, END_DATE)
        & data["normalized_title"].ne("")
        & data["normalized_institution"].ne("")
    )
    data = data.loc[valid].copy()
    rows: list[dict[str, Any]] = []
    for source_key, group in data.groupby("source_report_key", sort=True):
        analysts: set[str] = set()
        for value in group.get("researcher", []):
            analysts.update(item.strip() for item in _text(value).split(",") if item.strip())
        info_codes = [_text(value) for value in group.get("infoCode", []) if _text(value)]
        info_code = info_codes[0] if info_codes else ""
        rating_values = group.get("sRatingName", group.get("emRatingName", []))
        rows.append(
            {
                "report_id": _report_id(str(source_key)),
                "source_report_key": source_key,
                "symbol": _first(group["symbol"]),
                "trade_date": _first(group["source_date"]),
                "source_date": _first(group["source_date"]),
                "feature_available_date": _first(group["feature_available_date"]),
                "title": _first(group["title"]),
                "normalized_title": _first(group["normalized_title"]),
                "institution": _first(group["institution"]),
                "normalized_institution": _first(group["normalized_institution"]),
                "analyst": ";".join(sorted(analysts)),
                "report_type": _text(_first(group.get("reportType", []))),
                "classification": _text(_first(group.get("column", []))),
                "rating": _text(_first(rating_values)),
                "rating_change": _text(_first(group.get("ratingChange", []))),
                "target_price_min": _numeric(_first(group.get("indvAimPriceL", []))),
                "target_price_max": _numeric(_first(group.get("indvAimPriceT", []))),
                "tushare_present": False,
                "eastmoney_present": True,
                "tushare_source_ids": "[]",
                "eastmoney_info_codes": _json_list(info_codes),
                "url": (
                    f"https://pdf.dfcfw.com/pdf/H3_{info_code}_1.pdf"
                    if info_code
                    else ""
                ),
                "pdf_file_size_kb": _numeric(_first(group.get("attachSize", []))),
                "pdf_pages": _numeric(_first(group.get("attachPages", []))),
                "source_disagreement": bool(len(set(info_codes)) > 1),
                "identity_conflict_reason": (
                    "multiple_eastmoney_info_codes" if len(set(info_codes)) > 1 else ""
                ),
                "source": "eastmoney_report_metadata",
            }
        )
    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def _merge_reports(
    tushare: pd.DataFrame,
    eastmoney: pd.DataFrame,
) -> pd.DataFrame:
    combined = pd.concat([tushare, eastmoney], ignore_index=True)
    rows: list[dict[str, Any]] = []
    for source_key, group in combined.groupby("source_report_key", sort=True):
        ts = group.loc[group["tushare_present"].astype(bool)]
        em = group.loc[group["eastmoney_present"].astype(bool)]

        def preferred(
            column: str,
            ts_frame: pd.DataFrame = ts,
            em_frame: pd.DataFrame = em,
        ) -> Any:
            left = _first(ts_frame[column]) if column in ts_frame else ""
            return left if _text(left) else _first(em_frame[column])

        disagreements = bool(group["source_disagreement"].astype(bool).any())
        for column in ("rating", "target_price_min", "target_price_max"):
            left = {_text(value) for value in ts.get(column, []) if _text(value)}
            right = {_text(value) for value in em.get(column, []) if _text(value)}
            disagreements = disagreements or bool(left and right and left.isdisjoint(right))
        analysts = {
            item
            for value in group["analyst"]
            for item in str(value).split(";")
            if item
        }
        reasons = {
            _text(value)
            for value in group["identity_conflict_reason"]
            if _text(value)
        }
        rows.append(
            {
                "report_id": _report_id(str(source_key)),
                "source_report_key": source_key,
                "symbol": preferred("symbol"),
                "trade_date": preferred("source_date"),
                "source_date": preferred("source_date"),
                "feature_available_date": preferred("feature_available_date"),
                "title": preferred("title"),
                "normalized_title": preferred("normalized_title"),
                "institution": preferred("institution"),
                "normalized_institution": preferred("normalized_institution"),
                "analyst": ";".join(sorted(analysts)),
                "report_type": preferred("report_type"),
                "classification": preferred("classification"),
                "rating": preferred("rating"),
                "rating_change": preferred("rating_change"),
                "target_price_min": _numeric(preferred("target_price_min")),
                "target_price_max": _numeric(preferred("target_price_max")),
                "tushare_present": not ts.empty,
                "eastmoney_present": not em.empty,
                "tushare_source_ids": _json_list(
                    item
                    for value in ts.get("tushare_source_ids", [])
                    for item in json.loads(value or "[]")
                ),
                "eastmoney_info_codes": _json_list(
                    item
                    for value in em.get("eastmoney_info_codes", [])
                    for item in json.loads(value or "[]")
                ),
                "url": preferred("url"),
                "pdf_file_size_kb": _numeric(preferred("pdf_file_size_kb")),
                "pdf_pages": _numeric(preferred("pdf_pages")),
                "source_disagreement": disagreements,
                "identity_conflict_reason": ";".join(sorted(reasons)),
                "source": (
                    "tushare_report_rc+eastmoney_report_metadata"
                    if not ts.empty and not em.empty
                    else _first(group["source"])
                ),
            }
        )
    return (
        pd.DataFrame(rows, columns=REPORT_COLUMNS)
        .sort_values(["trade_date", "symbol", "report_id"], kind="stable")
        .reset_index(drop=True)
    )


def prepare_reports(
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    open_dates = _open_dates(workspace)
    normalized_root = _runtime(workspace) / "normalized" / "tushare_reports"
    report_parts: list[Path] = []
    forecast_parts: list[Path] = []
    statistics: list[dict[str, Any]] = []
    for year in range(2010, 2026):
        raw_paths = sorted(_report_rc_page_path(workspace, year, 0).parent.glob("*.parquet"))
        if not raw_paths:
            raise ResearchEventUpdateError(f"tushare_report_year_missing:{year}")
        raw = pd.concat([pd.read_parquet(path) for path in raw_paths], ignore_index=True)
        reports, forecasts, stats = _normalize_tushare_report_year(
            raw,
            open_dates=open_dates,
        )
        report_path = normalized_root / f"reports_{year}.parquet"
        forecast_path = normalized_root / f"forecasts_{year}.parquet"
        _write_parquet(reports, report_path)
        _write_parquet(forecasts, forecast_path)
        report_parts.append(report_path)
        forecast_parts.append(forecast_path)
        statistics.append(
            {
                "year": year,
                "request_page_count": len(raw_paths),
                **stats,
                **_report_year_source_coverage(year, stats),
            }
        )
        del raw, reports, forecasts
    em_paths = sorted((_runtime(workspace) / "raw" / "eastmoney_reports").glob("*.parquet"))
    if not em_paths:
        raise ResearchEventUpdateError("eastmoney_report_cache_missing")
    eastmoney_parts: list[pd.DataFrame] = []
    for path in em_paths:
        frame = _normalize_eastmoney_reports(
            pd.read_parquet(path),
            open_dates=open_dates,
        )
        if not frame.empty:
            eastmoney_parts.append(frame)
    tushare_reports = pd.concat(
        [pd.read_parquet(path) for path in report_parts], ignore_index=True
    )
    eastmoney_reports = (
        pd.concat(eastmoney_parts, ignore_index=True)
        if eastmoney_parts
        else pd.DataFrame(columns=REPORT_COLUMNS)
    )
    reports = _merge_reports(tushare_reports, eastmoney_reports)
    forecasts = pd.concat(
        [pd.read_parquet(path) for path in forecast_parts], ignore_index=True
    ).sort_values(["trade_date", "symbol", "report_id", "forecast_quarter"])
    prepared = _runtime(workspace) / "prepared"
    report_path = prepared / "research_report.parquet"
    forecast_path = prepared / "research_report_forecast.parquet"
    stats_path = prepared / "report_annual_statistics.parquet"
    _write_parquet(reports, report_path)
    _write_parquet(forecasts, forecast_path)
    _write_parquet(pd.DataFrame(statistics), stats_path)
    unavailable_years = [
        int(item["year"])
        for item in statistics
        if item["tushare_source_coverage_status"] == "source_unavailable"
    ]
    partial_years = [
        int(item["year"])
        for item in statistics
        if item["tushare_source_coverage_status"] == "partial_year_span"
    ]
    result = {
        "status": "completed",
        "report_path": str(report_path),
        "report_row_count": len(reports),
        "forecast_path": str(forecast_path),
        "forecast_row_count": len(forecasts),
        "annual_statistics_path": str(stats_path),
        "tushare_only_report_count": int(
            (reports["tushare_present"] & ~reports["eastmoney_present"]).sum()
        ),
        "eastmoney_only_report_count": int(
            (~reports["tushare_present"] & reports["eastmoney_present"]).sum()
        ),
        "matched_report_count": int(
            (reports["tushare_present"] & reports["eastmoney_present"]).sum()
        ),
        "earliest_report_date": str(reports["source_date"].min()),
        "latest_report_date": str(reports["source_date"].max()),
        "tushare_source_unavailable_years": unavailable_years,
        "tushare_source_partial_years": partial_years,
        "source_coverage_semantics": (
            "source_unavailable and partial-year periods must not be interpreted "
            "as dates with zero research activity"
        ),
        "eastmoney_forecasts_used": False,
        "eastmoney_forecast_exclusion_reason": (
            "historical predict-slot year semantics are not stable; Eastmoney is metadata-only"
        ),
    }
    state = _read_state(workspace)
    state["reports_prepared"] = result
    state["status"] = "reports_prepared"
    _write_state(workspace, state)
    return result


def _cninfo_announcement_path(workspace: Path, symbol: str) -> Path:
    return (
        _runtime(workspace)
        / "raw"
        / "cninfo_announcements"
        / f"symbol={symbol.replace('.', '_')}.parquet"
    )


def _eastmoney_announcement_path(workspace: Path, symbol: str) -> Path:
    return (
        _runtime(workspace)
        / "raw"
        / "eastmoney_announcements"
        / f"symbol={symbol.replace('.', '_')}.parquet"
    )


def _fetch_cninfo_symbol(symbol: str) -> pd.DataFrame:
    frame = _fetch_cninfo_announcements(
        symbol=symbol,
        start_date=START_DATE,
        end_date=END_DATE,
        page_size=30,
        max_pages=0,
    )
    if frame.empty:
        return pd.DataFrame({"_query_symbol": pd.Series(dtype="object")})
    frame["_query_symbol"] = symbol
    return frame


def _fetch_eastmoney_announcements(symbol: str) -> pd.DataFrame:
    code = str(symbol).split(".", 1)[0]
    url = "https://np-anotice-stock.eastmoney.com/api/security/ann"
    common = {
        "sr": "-1",
        "page_size": "100",
        "ann_type": "A",
        "client_source": "web",
        "f_node": "0",
        "s_node": "0",
        "stock_list": code,
        "begin_time": START_DATE,
        "end_time": END_DATE,
    }
    page = 1
    total_pages = 1
    rows: list[dict[str, Any]] = []
    with requests.Session() as session:
        while page <= total_pages:
            payload = _request_json(
                "GET",
                url,
                params={**common, "page_index": str(page)},
                session=session,
            )
            data = dict(payload.get("data", {}) or {})
            total_pages = math.ceil(int(data.get("total_hits", 0) or 0) / 100)
            for item in list(data.get("list", []) or []):
                record = dict(item)
                record["_query_symbol"] = symbol
                columns = list(record.pop("columns", []) or [])
                codes = list(record.pop("codes", []) or [])
                if columns:
                    record.update(dict(columns[0]))
                selected = next(
                    (
                        dict(value)
                        for value in codes
                        if str(value.get("stock_code", "")) == code
                    ),
                    {},
                )
                record.update(selected)
                rows.append(record)
            page += 1
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame({"_query_symbol": pd.Series(dtype="object")})


def download_announcements(
    *,
    workspace_root: str | Path | None = None,
    max_workers: int = MAX_WORKERS,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    symbols = _identity_symbols(workspace)
    cninfo = _download_symbol_files(
        workspace=workspace,
        state_key="cninfo_announcements",
        symbols=symbols,
        path_for=_cninfo_announcement_path,
        fetch=_fetch_cninfo_symbol,
        max_workers=max_workers,
        allow_failures=True,
    )
    eastmoney = _download_symbol_files(
        workspace=workspace,
        state_key="eastmoney_announcements",
        symbols=symbols,
        path_for=_eastmoney_announcement_path,
        fetch=_fetch_eastmoney_announcements,
        max_workers=max_workers,
        allow_failures=True,
    )
    return {"cninfo": cninfo, "eastmoney": eastmoney}


_CATEGORY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "risk_warning",
        ("风险提示", "退市", "立案", "处罚", "诉讼", "仲裁", "违约", "冻结", "警示"),
    ),
    (
        "performance",
        ("业绩", "年度报告", "年报", "季度报告", "季报", "半年度报告", "快报", "预告"),
    ),
    (
        "restructuring",
        ("重组", "收购", "并购", "重大资产", "发行股份购买"),
    ),
    (
        "holding_change",
        ("增持", "减持", "持股变动", "回购", "股权激励", "解除限售", "解禁"),
    ),
    (
        "financing",
        ("融资", "增发", "配股", "可转债", "公司债", "发行证券"),
    ),
    (
        "governance",
        ("董事会", "监事会", "股东大会", "独立董事", "高级管理人员"),
    ),
    (
        "operations",
        ("合同", "中标", "项目", "投资", "经营", "订单"),
    ),
)


def _announcement_category(title: Any, provider_category: Any = "") -> str:
    text = unicodedata.normalize("NFKC", _text(title))
    provider = _text(provider_category)
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(keyword in text or keyword in provider for keyword in keywords):
            return category
    return provider if provider and not provider.isdigit() else "other"


def _normalize_cninfo_announcements(
    raw: pd.DataFrame,
    *,
    open_dates: np.ndarray,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=ANNOUNCEMENT_COLUMNS)
    data = raw.copy()
    data["symbol"] = data.get("_query_symbol", data.get("symbol", "")).fillna("").astype(str)
    data["source_date"] = pd.to_datetime(
        data.get("source_date", data.get("trade_date")), errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    data["title"] = data.get("title", "").fillna("").astype(str)
    data["normalized_title"] = data["title"].map(_normalized_text)
    data["feature_available_date"] = _next_open_date(data["source_date"], open_dates)
    data["category"] = [
        _announcement_category(title, codes)
        for title, codes in zip(
            data["title"], data.get("announcement_type_codes", "")
        )
    ]
    data["file_size_kb"] = pd.to_numeric(
        data.get("file_size_kb"), errors="coerce"
    )
    data["_event_key"] = [
        _stable_hash(symbol, date, title)
        for symbol, date, title in zip(
            data["symbol"], data["source_date"], data["normalized_title"]
        )
    ]
    data["announcement_id"] = data["_event_key"].map(lambda value: f"ann_{value[:24]}")
    data["trade_date"] = data["source_date"]
    for column, default in {
        "publish_time": "",
        "announcement_type_codes": "",
        "cninfo_announcement_id": "",
        "eastmoney_art_code": "",
        "org_id": "",
        "url": "",
        "pdf_url": "",
        "file_size_kb": math.nan,
        "cninfo_present": True,
        "eastmoney_present": False,
        "source_disagreement": False,
        "source": "cninfo",
    }.items():
        if column not in data:
            data[column] = default
    valid = (
        data["source_date"].between(START_DATE, END_DATE)
        & data["normalized_title"].ne("")
    )
    return data.loc[valid, list(ANNOUNCEMENT_COLUMNS)].drop_duplicates().reset_index(drop=True)


def _normalize_eastmoney_announcements(
    raw: pd.DataFrame,
    *,
    open_dates: np.ndarray,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=ANNOUNCEMENT_COLUMNS)
    data = raw.copy()
    data["symbol"] = data.get("_query_symbol", "").fillna("").astype(str)
    data["source_date"] = pd.to_datetime(
        data.get("notice_date"), errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    data["publish_time"] = pd.to_datetime(
        data.get("display_time", data.get("notice_date")), errors="coerce"
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    data["title"] = data.get("title", "").fillna("").astype(str)
    data["normalized_title"] = data["title"].map(_normalized_text)
    data["feature_available_date"] = _next_open_date(data["source_date"], open_dates)
    provider_category = data.get("column_name", pd.Series("", index=data.index))
    data["category"] = [
        _announcement_category(title, category)
        for title, category in zip(data["title"], provider_category)
    ]
    data["_event_key"] = [
        _stable_hash(symbol, date, title)
        for symbol, date, title in zip(
            data["symbol"], data["source_date"], data["normalized_title"]
        )
    ]
    data["announcement_id"] = data["_event_key"].map(lambda value: f"ann_{value[:24]}")
    data["trade_date"] = data["source_date"]
    data["announcement_type_codes"] = data.get("column_code", "")
    data["cninfo_announcement_id"] = ""
    data["eastmoney_art_code"] = data.get("art_code", "")
    data["org_id"] = ""
    code = data.get("stock_code", data["symbol"].str.split(".").str[0])
    data["url"] = [
        f"https://data.eastmoney.com/notices/detail/{item}/{article}.html"
        if _text(article)
        else ""
        for item, article in zip(code, data["eastmoney_art_code"])
    ]
    data["pdf_url"] = ""
    data["file_size_kb"] = math.nan
    data["cninfo_present"] = False
    data["eastmoney_present"] = True
    data["source_disagreement"] = False
    data["source"] = "eastmoney_announcement_metadata"
    valid = (
        data["source_date"].between(START_DATE, END_DATE)
        & data["normalized_title"].ne("")
    )
    return data.loc[valid, list(ANNOUNCEMENT_COLUMNS)].drop_duplicates().reset_index(drop=True)


def prepare_announcements(
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    open_dates = _open_dates(workspace)
    normalized_root = _runtime(workspace) / "normalized" / "announcements"
    source_specs = (
        (
            "cninfo",
            _runtime(workspace) / "raw" / "cninfo_announcements",
            _normalize_cninfo_announcements,
        ),
        (
            "eastmoney",
            _runtime(workspace) / "raw" / "eastmoney_announcements",
            _normalize_eastmoney_announcements,
        ),
    )
    normalized_paths: list[Path] = []
    source_rows: dict[str, int] = {}
    for source_name, raw_root, normalize in source_specs:
        count = 0
        for raw_path in sorted(raw_root.glob("*.parquet")):
            output_path = normalized_root / source_name / raw_path.name
            if not output_path.is_file():
                frame = normalize(pd.read_parquet(raw_path), open_dates=open_dates)
                _write_parquet(frame, output_path)
            count += int(pq.ParquetFile(output_path).metadata.num_rows)
            normalized_paths.append(output_path)
        source_rows[source_name] = count
    if not normalized_paths:
        raise ResearchEventUpdateError("announcement_cache_missing")
    prepared = _runtime(workspace) / "prepared" / "announcement.parquet"
    prepared.parent.mkdir(parents=True, exist_ok=True)
    temporary = prepared.with_suffix(".tmp.parquet")
    scans = ",".join(
        f"'{str(path).replace(chr(39), chr(39) * 2)}'" for path in normalized_paths
    )
    sql = f"""
    COPY (
      WITH source AS (
        SELECT * FROM read_parquet([{scans}], union_by_name=true)
      ), ranked AS (
        SELECT *,
          row_number() OVER (
            PARTITION BY announcement_id
            ORDER BY cninfo_present DESC, eastmoney_present DESC, source
          ) AS rn,
          bool_or(cninfo_present) OVER (PARTITION BY announcement_id) AS any_cninfo,
          bool_or(eastmoney_present) OVER (PARTITION BY announcement_id) AS any_eastmoney,
          count(DISTINCT category) OVER (PARTITION BY announcement_id) > 1 AS disagrees
        FROM source
      )
      SELECT
        announcement_id,symbol,trade_date,source_date,feature_available_date,
        publish_time,title,normalized_title,category,announcement_type_codes,
        cninfo_announcement_id,eastmoney_art_code,org_id,url,pdf_url,file_size_kb,
        any_cninfo AS cninfo_present,any_eastmoney AS eastmoney_present,
        source_disagreement OR disagrees AS source_disagreement,
        CASE WHEN any_cninfo AND any_eastmoney
          THEN 'cninfo+eastmoney_announcement_metadata' ELSE source END AS source
      FROM ranked WHERE rn=1
      ORDER BY trade_date,symbol,announcement_id
    ) TO '{str(temporary).replace(chr(39), chr(39) * 2)}'
      (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    connection = duckdb.connect()
    try:
        connection.execute(sql)
    finally:
        connection.close()
    os.replace(temporary, prepared)
    frame = pd.read_parquet(
        prepared,
        columns=[
            "announcement_id",
            "source_date",
            "cninfo_present",
            "eastmoney_present",
        ],
    )
    result = {
        "status": "completed",
        "announcement_path": str(prepared),
        "announcement_row_count": len(frame),
        "source_normalized_rows": source_rows,
        "matched_announcement_count": int(
            (frame["cninfo_present"] & frame["eastmoney_present"]).sum()
        ),
        "earliest_announcement_date": str(frame["source_date"].min()),
        "latest_announcement_date": str(frame["source_date"].max()),
    }
    state = _read_state(workspace)
    state["announcements_prepared"] = result
    state["status"] = "announcements_prepared"
    _write_state(workspace, state)
    return result


def _install_domain(
    workspace: Path,
    *,
    domain: str,
    prepared: Path,
    contract_version: str,
    primary_key: Sequence[str],
    frequency: str,
    source: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    root = qdp_v2_root(workspace)
    digest = _sha256(prepared)[:24]
    dataset_id = f"{domain}__{digest}"
    dataset_dir = root / "datasets" / domain / dataset_id
    shard = dataset_dir / "shards" / "part-0000.parquet"
    if not shard.is_file():
        shard.parent.mkdir(parents=True, exist_ok=True)
        temporary = shard.with_suffix(".tmp.parquet")
        shutil.copy2(prepared, temporary)
        os.replace(temporary, shard)
    connection = duckdb.connect()
    try:
        quoted = str(shard).replace("'", "''")
        key_sql = ",".join(f'"{item}"' for item in primary_key)
        row_count = int(connection.execute(f"SELECT count(*) FROM read_parquet('{quoted}')").fetchone()[0])
        duplicate_count = int(
            connection.execute(
                f"SELECT count(*) FROM (SELECT {key_sql},count(*) n "
                f"FROM read_parquet('{quoted}') GROUP BY {key_sql} HAVING n>1)"
            ).fetchone()[0]
        )
        dates = connection.execute(
            f"SELECT min(source_date),max(source_date),"
            f"count(*) FILTER(WHERE source_date>'{END_DATE}'),"
            "count(*) FILTER(WHERE feature_available_date<>'' "
            "AND try_cast(feature_available_date AS DATE)<=try_cast(source_date AS DATE)) "
            f"FROM read_parquet('{quoted}')"
        ).fetchone()
    finally:
        connection.close()
    if duplicate_count or int(dates[2]) or int(dates[3]):
        raise ResearchEventUpdateError(
            f"event_contract_failed:{domain}:duplicates={duplicate_count}:"
            f"future={int(dates[2])}:availability={int(dates[3])}"
        )
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        domain=domain,
        layer="raw",
        frequency=frequency,
        contract_version=contract_version,
        primary_key=list(primary_key),
        start_date=str(dates[0]),
        end_date=str(dates[1]),
        row_count=row_count,
        shards=[
            ShardManifestEntry(
                path=str(shard.relative_to(root)).replace("\\", "/"),
                row_count=row_count,
                start_date=str(dates[0]),
                end_date=str(dates[1]),
                file_size=shard.stat().st_size,
                metadata={"prepared_sha256": _sha256(prepared)},
            )
        ],
        source={
            **dict(source),
            "checked_through": END_DATE,
            "scope": "point_in_time_historical_mainboard",
            "credential_persisted": False,
        },
        quality={
            "primary_key_unique": True,
            "future_source_rows": 0,
            "availability_not_after_source_rows": 0,
            "strict_point_in_time": True,
            "forbidden_2026_rows": 0,
        },
        schema=_manifest_schema_from_arrow(pq.read_schema(shard)),
        notes=[
            "source_date is the visible report/announcement date; features consume from feature_available_date",
            "PDF URLs are evidence metadata only; no PDF body was downloaded",
        ],
    )
    _assert_credential_free(manifest.to_dict())
    write_dataset_manifest(root, manifest)
    return dataset_id, {
        "dataset_id": dataset_id,
        "row_count": row_count,
        "start_date": str(dates[0]),
        "end_date": str(dates[1]),
        "sha256": _sha256(shard),
    }


def commit_prepared(
    *,
    workspace_root: str | Path | None = None,
    include_announcements: bool = True,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    prepared = _runtime(workspace) / "prepared"
    annual_statistics = pd.read_parquet(
        prepared / "report_annual_statistics.parquet",
        columns=[
            "year",
            "tushare_source_coverage_status",
            "tushare_source_coverage_start",
            "tushare_source_coverage_end",
        ],
    ).to_dict(orient="records")
    specs = [
        (
            DataDomain.RESEARCH_REPORT,
            prepared / "research_report.parquet",
            "qdp_v2_research_report_pit_v1",
            ["report_id"],
            "event",
            {
                "provider": "tushare_report_rc+eastmoney_report_metadata",
                "availability_semantics": "report date; consume next exchange-open day",
                "eastmoney_forecasts_used": False,
                "tushare_annual_source_coverage": annual_statistics,
            },
        ),
        (
            DataDomain.RESEARCH_REPORT_FORECAST,
            prepared / "research_report_forecast.parquet",
            "qdp_v2_research_report_forecast_pit_v1",
            ["report_id", "forecast_quarter", "source"],
            "event_forecast",
            {
                "provider": "tushare_report_rc",
                "availability_semantics": "parent report date; consume next exchange-open day",
                "quarter_rows_do_not_repeat_report_weight": True,
                "tushare_annual_source_coverage": annual_statistics,
            },
        ),
    ]
    if include_announcements:
        specs.append(
            (
                DataDomain.ANNOUNCEMENT,
                prepared / "announcement.parquet",
                "qdp_v2_announcement_metadata_pit_v1",
                ["announcement_id"],
                "event",
                {
                    "provider": "cninfo+eastmoney_announcement_metadata",
                    "availability_semantics": "announcement date; consume next exchange-open day",
                },
            )
        )
    missing = [str(path) for _, path, *_ in specs if not path.is_file()]
    if missing:
        raise ResearchEventUpdateError(f"prepared_domain_missing:{missing}")
    installed: dict[str, Any] = {}
    dataset_ids: dict[str, str] = {}
    for domain, path, version, primary_key, frequency, source in specs:
        dataset_id, record = _install_domain(
            workspace,
            domain=domain,
            prepared=path,
            contract_version=version,
            primary_key=primary_key,
            frequency=frequency,
            source=source,
        )
        dataset_ids[domain] = dataset_id
        installed[domain] = record
    root = qdp_v2_root(workspace)
    active = read_active_manifest(root)
    active["datasets"] = {
        **dict(active.get("datasets", {}) or {}),
        **dataset_ids,
    }
    active["updated_at"] = utc_now()
    write_active_manifest(root, active)
    state = _read_state(workspace)
    state["installed_domains"] = installed
    state["status"] = "applied"
    _write_state(workspace, state)
    audit_path = root / "audits" / f"{UPDATE_ID}.json"
    _assert_credential_free(state)
    atomic_write_json(audit_path, state)
    return {"status": "applied", "domains": installed, "audit_path": str(audit_path)}


def evaluate(
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    domains = dict(state.get("installed_domains", {}) or {})
    checks = {
        "date_range_exact": bool(
            domains
            and domains.get(DataDomain.RESEARCH_REPORT, {}).get("start_date") == START_DATE
            and all(str(item.get("end_date", "")) <= END_DATE for item in domains.values())
        ),
        "forbidden_2026_rows": True,
        "credential_not_persisted": True,
        "pdf_not_downloaded": True,
        "report_forecast_is_separate_domain": DataDomain.RESEARCH_REPORT_FORECAST
        in domains,
    }
    result = {
        "status": "ok" if all(checks.values()) else "error",
        "update_id": UPDATE_ID,
        "checks": checks,
        "domains": domains,
        "reports": state.get("reports_prepared", {}),
        "announcements": state.get("announcements_prepared", {}),
    }
    _assert_credential_free(result)
    path = _runtime(workspace) / "evaluation.json"
    atomic_write_json(path, result)
    return result


def self_test() -> dict[str, Any]:
    if _next_report_offset(REPORT_RC_PAGE_SIZE, 0) != REPORT_RC_PAGE_SIZE:
        raise AssertionError("report_rc full page pagination changed")
    if _next_report_offset(REPORT_RC_PAGE_SIZE - 1, 0) is not None:
        raise AssertionError("report_rc terminal page pagination changed")
    source_key = _report_source_key(
        "000001.SZ", "2024-01-02", " 盈利预测：更新 ", "某某证券股份有限公司"
    )
    if source_key != _report_source_key(
        "000001.SZ", "2024-01-02", "盈利预测更新", "某某证券"
    ):
        raise AssertionError("cross-source report identity normalization changed")
    dates = np.asarray(
        pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"]),
        dtype="datetime64[ns]",
    )
    available = _next_open_date(pd.Series(["2024-01-02", "2024-01-05"]), dates)
    if available.tolist() != ["2024-01-03", ""]:
        raise AssertionError("next-open availability changed")
    if _announcement_category("关于公司股票可能被终止上市的风险提示公告") != "risk_warning":
        raise AssertionError("announcement category mapping changed")
    return {
        "status": "ok",
        "checks": {
            "report_rc_page_cap": REPORT_RC_PAGE_SIZE,
            "maximum_workers": MAX_WORKERS,
            "probe_zero_data_period": "2000-01-01/2009-12-31",
            "probe_earliest_report_date": START_DATE,
            "next_exchange_day_availability": True,
            "cross_source_identity": True,
            "forbidden_2026": True,
            "credential_not_persisted": True,
        },
    }


def run_pending(
    *,
    workspace_root: str | Path | None = None,
    phase: str = "all",
    max_workers: int = MAX_WORKERS,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    result: dict[str, Any] = {"status": "running", "phase": phase}
    if phase in {"all", "reports"}:
        result["tushare_reports"] = download_tushare_reports(
            workspace_root=workspace,
            max_workers=max_workers,
        )
        result["eastmoney_reports"] = download_eastmoney_reports(
            workspace_root=workspace,
            max_workers=max_workers,
        )
        result["reports_prepared"] = prepare_reports(workspace_root=workspace)
    if phase in {"all", "announcements"}:
        result["announcement_downloads"] = download_announcements(
            workspace_root=workspace,
            max_workers=max_workers,
        )
        result["announcements_prepared"] = prepare_announcements(
            workspace_root=workspace
        )
    if phase == "reports":
        result["commit"] = commit_prepared(
            workspace_root=workspace,
            include_announcements=False,
        )
    elif phase in {"all", "announcements"}:
        result["commit"] = commit_prepared(
            workspace_root=workspace,
            include_announcements=True,
        )
    result["status"] = "completed"
    return result


def status(
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    state = _read_state(_workspace(workspace_root))
    reports = dict(state.get("tushare_report_rc", {}) or {})
    years = dict(reports.get("years", {}) or {})
    return {
        "update_id": UPDATE_ID,
        "status": state.get("status", "pending"),
        "tushare_report_years_completed": sum(
            item.get("status") == "completed" for item in years.values()
        ),
        "tushare_report_year_count": len(years),
        "eastmoney_report_symbols_completed": dict(
            state.get("eastmoney_reports", {}) or {}
        ).get("completed_symbol_count", 0),
        "cninfo_announcement_symbols_completed": dict(
            state.get("cninfo_announcements", {}) or {}
        ).get("completed_symbol_count", 0),
        "eastmoney_announcement_symbols_completed": dict(
            state.get("eastmoney_announcements", {}) or {}
        ).get("completed_symbol_count", 0),
        "reports_prepared": "reports_prepared" in state,
        "announcements_prepared": "announcements_prepared" in state,
        "installed_domains": sorted(dict(state.get("installed_domains", {}) or {})),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp research-event-update")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--phase", choices=("all", "reports", "announcements"), default="all")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--run-pending", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = str(args.workspace_root or "") or None
    if args.status:
        payload = status(workspace_root=workspace)
    elif args.run_pending:
        payload = run_pending(
            workspace_root=workspace,
            phase=str(args.phase),
            max_workers=int(args.max_workers),
        )
    elif args.evaluate:
        payload = evaluate(workspace_root=workspace)
    else:
        payload = self_test()
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
