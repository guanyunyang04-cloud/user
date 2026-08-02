from __future__ import annotations

"""Build official-exchange margin eligibility and repair missing detail rows."""

import argparse
import hashlib
import io
import json
import os
import shutil
import time
import warnings
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
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
from quant_data_platform.qdp_v2.research_event_update import (
    _assert_credential_free,
    _sha256,
    _write_parquet,
)
from quant_data_platform.qdp_v2.status import active_dataset_map

UPDATE_ID = "margin_eligibility_exchange_history_v1"
PROVENANCE_REPAIR_ID = "margin_detail_provenance_repair_v1"
START_DATE = "2011-01-01"
END_DATE = "2025-12-31"
MAX_WORKERS = 4
ENDPOINTS = ("sse_detail", "szse_detail", "szse_eligibility")
SSE_URL = "https://query.sse.com.cn/marketdata/tradedata/queryMargin.do"
SZSE_URL = "https://www.szse.cn/api/report/ShowReport"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
)

OFFICIAL_COLUMNS = (
    "symbol",
    "trade_date",
    "exchange",
    "endpoint",
    "name",
    "eligible",
    "finance_eligible",
    "securities_lending_eligible",
    "rzye",
    "rqye",
    "rzmre",
    "rqyl",
    "rzche",
    "rqchl",
    "rqmcl",
    "rzrqye",
    "source",
)
MARGIN_DETAIL_COLUMNS = (
    "security_id",
    "symbol",
    "ts_code",
    "trade_date",
    "rzye",
    "rqye",
    "rzmre",
    "rqyl",
    "rzche",
    "rqchl",
    "rqmcl",
    "rzrqye",
    "source_date",
    "feature_available_date",
    "burn_in_only",
    "source",
)


class MarginEligibilityUpdateError(RuntimeError):
    pass


@dataclass(frozen=True)
class EndpointResult:
    endpoint: str
    frame: pd.DataFrame
    response_sha256: str
    response_bytes: int


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
    return (
        _runtime(workspace)
        / "raw"
        / f"year={trade_date[:4]}"
        / f"date={trade_date}.parquet"
    )


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _number(series: pd.Series, *, multiplier: float = 1.0) -> pd.Series:
    values = pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    return values.astype("float64") * float(multiplier)


def _empty_official() -> pd.DataFrame:
    return pd.DataFrame(columns=OFFICIAL_COLUMNS)


def _official_frame(
    *,
    symbol: pd.Series,
    trade_date: str,
    exchange: str,
    endpoint: str,
    name: pd.Series,
    source: str,
    eligible: pd.Series | bool | None = None,
    finance_eligible: pd.Series | bool | None = None,
    securities_lending_eligible: pd.Series | bool | None = None,
    metrics: Mapping[str, pd.Series] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(index=symbol.index)
    code = symbol.fillna("").astype(str).str.extract(r"(\d{6})", expand=False)
    frame["symbol"] = code + (".SH" if exchange == "SH" else ".SZ")
    frame["trade_date"] = trade_date
    frame["exchange"] = exchange
    frame["endpoint"] = endpoint
    frame["name"] = name.fillna("").astype(str)
    for column, value in (
        ("eligible", eligible),
        ("finance_eligible", finance_eligible),
        ("securities_lending_eligible", securities_lending_eligible),
    ):
        if isinstance(value, pd.Series):
            frame[column] = value.astype("boolean")
        else:
            frame[column] = pd.Series(value, index=frame.index, dtype="boolean")
    for column in (
        "rzye",
        "rqye",
        "rzmre",
        "rqyl",
        "rzche",
        "rqchl",
        "rqmcl",
        "rzrqye",
    ):
        frame[column] = (
            pd.to_numeric(metrics[column], errors="coerce")
            if metrics and column in metrics
            else np.nan
        )
    frame["source"] = source
    return frame.loc[code.notna(), OFFICIAL_COLUMNS].reset_index(drop=True)


def _fetch_sse_detail(trade_date: str, *, timeout: int = 45) -> EndpointResult:
    compact = trade_date.replace("-", "")
    params = {
        "isPagination": "true",
        "tabType": "mxtype",
        "detailsDate": compact,
        "stockCode": "",
        "beginDate": "",
        "endDate": "",
        "pageHelp.pageSize": "5000",
        "pageHelp.pageCount": "50",
        "pageHelp.pageNo": "1",
        "pageHelp.beginPage": "1",
        "pageHelp.cacheSize": "1",
        "pageHelp.endPage": "21",
    }
    response = requests.get(
        SSE_URL,
        params=params,
        headers={"Referer": "https://www.sse.com.cn/", "User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.content
    data = response.json()
    raw = pd.DataFrame(data.get("result", []))
    if raw.empty:
        raise MarginEligibilityUpdateError(f"sse_detail_empty:{trade_date}")
    if not {"stockCode", "securityAbbr", "opDate", "rzye"}.issubset(raw.columns):
        raise MarginEligibilityUpdateError(f"sse_detail_schema:{trade_date}")
    dates = raw["opDate"].fillna("").astype(str).unique().tolist()
    if dates != [compact]:
        raise MarginEligibilityUpdateError(
            f"sse_detail_date_mismatch:{trade_date}:{dates[:3]}"
        )
    frame = _official_frame(
        symbol=raw["stockCode"],
        trade_date=trade_date,
        exchange="SH",
        endpoint="sse_detail",
        name=raw["securityAbbr"],
        eligible=True,
        source="sse_official_margin_detail",
        metrics={
            "rzye": _number(raw["rzye"]),
            "rzmre": _number(raw["rzmre"]),
            "rzche": _number(raw["rzche"]),
            "rqyl": _number(raw["rqyl"]),
            "rqmcl": _number(raw["rqmcl"]),
            "rqchl": _number(raw["rqchl"]),
        },
    )
    return EndpointResult("sse_detail", frame, _hash_bytes(payload), len(payload))


def _fetch_szse_excel(
    trade_date: str,
    *,
    endpoint: str,
    timeout: int = 45,
) -> EndpointResult:
    if endpoint == "szse_detail":
        catalog = "1837_xxpl"
        tab = "tab2"
        referer = "https://www.szse.cn/disclosure/margin/margin/index.html"
    elif endpoint == "szse_eligibility":
        catalog = "1834_xxpl"
        tab = "tab1"
        referer = "https://www.szse.cn/disclosure/margin/object/index.html"
    else:
        raise ValueError(f"unsupported_szse_endpoint:{endpoint}")
    params = {
        "SHOWTYPE": "xlsx",
        "CATALOGID": catalog,
        "txtDate": trade_date,
        f"{tab}PAGENO": "1",
        "random": "0.3141592653589793",
        "TABKEY": tab,
    }
    response = requests.get(
        SZSE_URL,
        params=params,
        headers={"Referer": referer, "User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.content
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Workbook contains no default style.*",
                category=UserWarning,
            )
            raw = pd.read_excel(io.BytesIO(payload), engine="openpyxl", dtype=str)
    except Exception as exc:
        raise MarginEligibilityUpdateError(
            f"{endpoint}_parse:{trade_date}:{type(exc).__name__}"
        ) from exc
    # The SZSE workbook advertises GBK in the response header but some historical
    # files expose mojibake column labels through openpyxl.  The official workbook
    # column order is stable, so normalize by position after checking its width.
    expected_width = 8 if endpoint == "szse_detail" else 7
    if raw.empty or len(raw.columns) < expected_width:
        raise MarginEligibilityUpdateError(f"{endpoint}_schema_or_empty:{trade_date}")
    if endpoint == "szse_detail":
        normalized = raw.iloc[:, :8].copy()
        normalized.columns = [
            "security_code",
            "security_name",
            "financing_buy_amount",
            "financing_balance",
            "lending_sell_volume",
            "lending_balance_volume",
            "lending_balance_amount",
            "margin_total_balance",
        ]
        frame = _official_frame(
            symbol=normalized["security_code"],
            trade_date=trade_date,
            exchange="SZ",
            endpoint=endpoint,
            name=normalized["security_name"],
            source="szse_official_margin_detail",
            metrics={
                "rzmre": _number(normalized["financing_buy_amount"]),
                "rzye": _number(normalized["financing_balance"]),
                "rqmcl": _number(normalized["lending_sell_volume"]),
                "rqyl": _number(normalized["lending_balance_volume"]),
                "rqye": _number(normalized["lending_balance_amount"]),
                "rzrqye": _number(normalized["margin_total_balance"]),
            },
        )
    else:
        normalized = raw.iloc[:, :7].copy()
        normalized.columns = [
            "security_code",
            "security_name",
            "financing_underlying",
            "lending_underlying",
            "financing_available_today",
            "lending_available_today",
            "lending_price_limit_status",
        ]
        finance = normalized["financing_underlying"].fillna("").str.upper().eq("Y")
        lending = normalized["lending_underlying"].fillna("").str.upper().eq("Y")
        frame = _official_frame(
            symbol=normalized["security_code"],
            trade_date=trade_date,
            exchange="SZ",
            endpoint=endpoint,
            name=normalized["security_name"],
            source="szse_official_margin_eligibility",
            eligible=finance | lending,
            finance_eligible=finance,
            securities_lending_eligible=lending,
        )
    return EndpointResult(endpoint, frame, _hash_bytes(payload), len(payload))


def _fetch_endpoint(trade_date: str, endpoint: str) -> EndpointResult:
    last_error = ""
    for attempt in range(4):
        try:
            if endpoint == "sse_detail":
                return _fetch_sse_detail(trade_date)
            return _fetch_szse_excel(trade_date, endpoint=endpoint)
        except (requests.RequestException, MarginEligibilityUpdateError) as exc:
            last_error = f"{type(exc).__name__}:{str(exc)[:240]}"
            if attempt < 3:
                time.sleep(0.5 * (attempt + 1))
    raise MarginEligibilityUpdateError(
        f"exchange_request_failed:{trade_date}:{endpoint}:{last_error}"
    )


def _fetch_day(workspace: Path, trade_date: str) -> dict[str, Any]:
    results = [_fetch_endpoint(trade_date, endpoint) for endpoint in ENDPOINTS]
    frame = pd.concat([result.frame for result in results], ignore_index=True)
    if set(frame["endpoint"].unique()) != set(ENDPOINTS):
        raise MarginEligibilityUpdateError(f"exchange_endpoint_incomplete:{trade_date}")
    path = _raw_day_path(workspace, trade_date)
    _write_parquet(frame, path)
    return {
        "status": "observed",
        "path": str(path),
        "row_count": len(frame),
        "sha256": _sha256(path),
        "endpoints": {
            result.endpoint: {
                "status": "observed",
                "row_count": len(result.frame),
                "response_sha256": result.response_sha256,
                "response_bytes": result.response_bytes,
            }
            for result in results
        },
        "completed_at": utc_now(),
    }


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
    return ",".join(
        f"'{path.resolve().as_posix().replace(chr(39), chr(39) * 2)}'" for path in paths
    )


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
        raise MarginEligibilityUpdateError(
            f"unexpected_margin_trade_dates:{len(dates)}"
        )
    return dates


def download(
    *,
    workspace_root: str | Path | None = None,
    max_workers: int = MAX_WORKERS,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    dates = _trade_dates(workspace)
    days = dict(state.get("days", {}) or {})
    pending: deque[str] = deque()
    for trade_date in dates:
        path = _raw_day_path(workspace, trade_date)
        record = dict(days.get(trade_date, {}) or {})
        if path.is_file():
            try:
                endpoints = set(pd.read_parquet(path, columns=["endpoint"])["endpoint"])
                if endpoints == set(ENDPOINTS):
                    record.update(
                        {
                            "status": "observed",
                            "path": str(path),
                            "row_count": int(pq.ParquetFile(path).metadata.num_rows),
                            "sha256": _sha256(path),
                        }
                    )
                    days[trade_date] = record
                    continue
            except (OSError, ValueError):
                pass
        pending.append(trade_date)
        days[trade_date] = {
            **record,
            "status": "pending",
            "attempt_count": int(record.get("attempt_count", 0) or 0),
            "last_error": "",
        }

    workers = min(MAX_WORKERS, max(1, int(max_workers)))

    def checkpoint(status: str) -> None:
        state.update(
            {
                "status": status,
                "start_date": START_DATE,
                "end_date": END_DATE,
                "requested_date_count": len(dates),
                "maximum_workers": workers,
                "days": days,
                "request_2026_count": 0,
            }
        )
        _write_state(workspace, state)

    active: dict[Future[dict[str, Any]], str] = {}

    def fill(pool: ThreadPoolExecutor) -> None:
        while pending and len(active) < workers:
            trade_date = pending.popleft()
            days[trade_date]["status"] = "downloading"
            active[pool.submit(_fetch_day, workspace, trade_date)] = trade_date

    processed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fill(pool)
        while active:
            completed, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in completed:
                trade_date = active.pop(future)
                prior_attempts = int(days[trade_date].get("attempt_count", 0) or 0)
                try:
                    days[trade_date] = {
                        **future.result(),
                        "attempt_count": prior_attempts + 1,
                        "last_error": "",
                    }
                except Exception as exc:  # noqa: BLE001 - source failures are ledger data
                    days[trade_date] = {
                        **days[trade_date],
                        "status": "failed",
                        "attempt_count": prior_attempts + 1,
                        "error_type": type(exc).__name__,
                        "last_error": f"{type(exc).__name__}:{str(exc)[:480]}",
                    }
                processed += 1
                if processed % 25 == 0:
                    checkpoint("downloading")
                fill(pool)
    failed = [date for date in dates if days[date].get("status") != "observed"]
    checkpoint("failed" if failed else "downloaded")
    if failed:
        raise MarginEligibilityUpdateError(
            f"exchange_daily_tasks_incomplete:{len(failed)}"
        )
    return state


def _copy_query(
    connection: duckdb.DuckDBPyConnection,
    *,
    query: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    quoted = temporary.resolve().as_posix().replace("'", "''")
    connection.execute(
        f"COPY ({query}) TO '{quoted}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    os.replace(temporary, path)


def prepare(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    if state.get("status") in {"prepared", "applied"}:
        return state
    if state.get("status") != "downloaded":
        raise MarginEligibilityUpdateError(
            f"margin_eligibility_not_downloaded:{state.get('status')}"
        )
    dates = _trade_dates(workspace)
    raw_paths = [_raw_day_path(workspace, trade_date) for trade_date in dates]
    if any(not path.is_file() for path in raw_paths):
        raise MarginEligibilityUpdateError("margin_exchange_raw_day_missing")
    universe_manifest, universe_paths = _dataset_paths(
        workspace, DataDomain.UNIVERSE_SNAPSHOT
    )
    margin_manifest, margin_paths = _dataset_paths(workspace, DataDomain.MARGIN_DETAIL)
    _, history_paths = _dataset_paths(workspace, DataDomain.SYMBOL_HISTORY)
    raw_scan = (
        f"read_parquet([{_sql_paths(raw_paths)}], union_by_name=true, "
        "hive_partitioning=false)"
    )
    universe_scan = (
        f"read_parquet([{_sql_paths(universe_paths)}], union_by_name=true, "
        "hive_partitioning=false)"
    )
    margin_scan = (
        f"read_parquet([{_sql_paths(margin_paths)}], union_by_name=true, "
        "hive_partitioning=false)"
    )
    history_scan = (
        f"read_parquet([{_sql_paths(history_paths)}], union_by_name=true, "
        "hive_partitioning=false)"
    )
    runtime = _runtime(workspace)
    temp = runtime / "duckdb_tmp"
    temp.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET threads=4")
    connection.execute(
        f"SET temp_directory='{temp.resolve().as_posix().replace(chr(39), chr(39) * 2)}'"
    )
    try:
        connection.execute(
            f"""
            CREATE TEMP TABLE official_detail AS
            SELECT symbol,trade_date,exchange,name,rzye,rqye,rzmre,rqyl,
                   rzche,rqchl,rqmcl,rzrqye,source
            FROM {raw_scan}
            WHERE endpoint IN ('sse_detail','szse_detail')
            """
        )
        connection.execute(
            f"""
            CREATE TEMP TABLE official_eligibility AS
            SELECT symbol,trade_date,exchange,eligible,
                   finance_eligible,securities_lending_eligible,source
            FROM {raw_scan}
            WHERE endpoint='szse_eligibility'
            UNION ALL
            SELECT symbol,trade_date,exchange,true AS eligible,
                   NULL::BOOLEAN AS finance_eligible,
                   NULL::BOOLEAN AS securities_lending_eligible,source
            FROM {raw_scan}
            WHERE endpoint='sse_detail'
            """
        )
        next_dates = pd.DataFrame({"trade_date": dates})
        next_dates["feature_available_date"] = [*dates[1:], ""]
        connection.register("next_open_dates", next_dates)
        eligibility_paths: list[Path] = []
        annual_stats: list[dict[str, Any]] = []
        for year in range(2011, 2026):
            path = (
                runtime
                / "prepared"
                / "margin_eligibility"
                / f"year={year}"
                / "part-0000.parquet"
            )
            query = f"""
              SELECT u.symbol,u.trade_date,u.exchange,
                CASE WHEN e.symbol IS NOT NULL THEN 'eligible_observed'
                     ELSE 'known_ineligible' END AS eligibility_state,
                e.symbol IS NOT NULL AS eligible,
                e.finance_eligible,e.securities_lending_eligible,
                d.symbol IS NOT NULL AS detail_observed,
                true AS source_available,
                true AS eligibility_source_available,
                true AS detail_source_available,
                u.trade_date AS source_date,n.feature_available_date,
                false AS burn_in_only,
                CASE WHEN u.exchange='SH' THEN 'sse_official_margin_detail'
                     ELSE 'szse_official_margin_eligibility+detail' END AS source
              FROM {universe_scan} u
              JOIN next_open_dates n USING(trade_date)
              LEFT JOIN official_eligibility e USING(symbol,trade_date,exchange)
              LEFT JOIN official_detail d USING(symbol,trade_date,exchange)
              WHERE u.trade_date BETWEEN '{year}-01-01' AND '{year}-12-31'
                AND lower(u.board)='main' AND u.exchange IN ('SH','SZ')
              ORDER BY u.trade_date,u.symbol
            """
            _copy_query(connection, query=query, path=path)
            eligibility_paths.append(path)
            stats = connection.execute(
                f"SELECT count(*),count(*) FILTER(WHERE eligible),"
                f"count(*) FILTER(WHERE detail_observed) FROM ({query})"
            ).fetchone()
            annual_stats.append(
                {
                    "year": year,
                    "row_count": int(stats[0]),
                    "eligible_count": int(stats[1]),
                    "detail_observed_count": int(stats[2]),
                }
            )
        eligibility_scan = (
            f"read_parquet([{_sql_paths(eligibility_paths)}], union_by_name=true, "
            "hive_partitioning=false)"
        )
        missing_query = f"""
          WITH official_main AS (
            SELECT d.* FROM official_detail d
            JOIN {universe_scan} u USING(symbol,trade_date,exchange)
            WHERE lower(u.board)='main'
          )
          SELECT h.security_id,o.symbol,o.symbol AS ts_code,o.trade_date,
                 o.rzye,o.rqye,o.rzmre,o.rqyl,o.rzche,o.rqchl,o.rqmcl,o.rzrqye,
                 o.trade_date AS source_date,n.feature_available_date,
                 false AS burn_in_only,o.source
          FROM official_main o
          JOIN {history_scan} h ON h.symbol=o.symbol
            AND o.trade_date BETWEEN h.effective_from AND h.effective_to
          JOIN next_open_dates n USING(trade_date)
          LEFT JOIN {margin_scan} m
            ON m.symbol=o.symbol AND m.trade_date=o.trade_date
          WHERE m.symbol IS NULL
          ORDER BY o.trade_date,o.symbol
        """
        missing_path = (
            runtime / "inventory" / "exchange_detail_missing_from_qdp.parquet"
        )
        _copy_query(connection, query=missing_query, path=missing_path)
        missing_count = int(
            connection.execute(f"SELECT count(*) FROM ({missing_query})").fetchone()[0]
        )
        conflicts_query = f"""
          WITH official_main AS (
            SELECT d.* FROM official_detail d
            JOIN {universe_scan} u USING(symbol,trade_date,exchange)
            WHERE lower(u.board)='main'
          ), paired AS (
            SELECT o.symbol,o.trade_date,o.exchange,o.source,
                   unnest(['rzye','rqye','rzmre','rqyl','rzche','rqchl','rqmcl','rzrqye']) AS field,
                   unnest([o.rzye,o.rqye,o.rzmre,o.rqyl,o.rzche,o.rqchl,o.rqmcl,o.rzrqye]) AS official_value,
                   unnest([m.rzye,m.rqye,m.rzmre,m.rqyl,m.rzche,m.rqchl,m.rqmcl,m.rzrqye]) AS qdp_value
            FROM official_main o JOIN {margin_scan} m USING(symbol,trade_date)
          )
          SELECT *,abs(official_value-qdp_value) AS absolute_difference,
            abs(official_value-qdp_value)/greatest(abs(official_value),abs(qdp_value),1)
              AS relative_difference
          FROM paired
          WHERE official_value IS NOT NULL AND qdp_value IS NOT NULL
            AND abs(official_value-qdp_value)>greatest(1e-6,
                greatest(abs(official_value),abs(qdp_value),1)*1e-10)
          ORDER BY trade_date,symbol,field
        """
        conflicts_path = runtime / "inventory" / "exchange_qdp_conflicts.parquet"
        _copy_query(connection, query=conflicts_query, path=conflicts_path)
        conflict_count = int(
            connection.execute(f"SELECT count(*) FROM ({conflicts_query})").fetchone()[
                0
            ]
        )
        repaired_margin_paths: list[Path] = []
        for year in range(2010, 2026):
            path = (
                runtime
                / "prepared"
                / "margin_detail"
                / f"year={year}"
                / "part-0000.parquet"
            )
            query = f"""
              SELECT {",".join(MARGIN_DETAIL_COLUMNS)}
              FROM {margin_scan}
              WHERE trade_date BETWEEN '{year}-01-01' AND '{year}-12-31'
              UNION ALL
              SELECT {",".join(MARGIN_DETAIL_COLUMNS)}
              FROM ({missing_query})
              WHERE trade_date BETWEEN '{year}-01-01' AND '{year}-12-31'
              ORDER BY trade_date,symbol
            """
            _copy_query(connection, query=query, path=path)
            repaired_margin_paths.append(path)
        repaired_margin_scan = (
            f"read_parquet([{_sql_paths(repaired_margin_paths)}], "
            "union_by_name=true, hive_partitioning=false)"
        )
        eligibility_stats = connection.execute(
            f"SELECT count(*),count(DISTINCT trade_date),"
            f"count(*) FILTER(WHERE eligibility_state='source_unavailable'),"
            f"count(*) FILTER(WHERE trade_date>'{END_DATE}'),"
            "count(*) FILTER(WHERE trade_date='2024-01-02' AND eligible),"
            "count(*) FILTER(WHERE trade_date='2011-01-04' AND eligible) "
            f"FROM {eligibility_scan}"
        ).fetchone()
        expected_eligibility_rows = int(
            connection.execute(
                f"SELECT count(*) FROM {universe_scan} "
                f"WHERE trade_date BETWEEN '{START_DATE}' AND '{END_DATE}' "
                "AND lower(board)='main' AND exchange IN ('SH','SZ')"
            ).fetchone()[0]
        )
        margin_stats = connection.execute(
            f"SELECT count(*),count(*)-count(DISTINCT symbol||'|'||trade_date),"
            f"count(*) FILTER(WHERE trade_date>'{END_DATE}') "
            f"FROM {repaired_margin_scan}"
        ).fetchone()
        known_000527 = int(
            connection.execute(
                f"SELECT count(*) FROM ({missing_query}) "
                "WHERE symbol='000527.SZ' AND trade_date='2011-01-04'"
            ).fetchone()[0]
        )
    finally:
        connection.close()
    checks = {
        "every_trade_date_covered": int(eligibility_stats[1]) == len(dates),
        "eligibility_row_count_matches_universe": int(eligibility_stats[0])
        == expected_eligibility_rows,
        "source_unavailable_explicit_count": int(eligibility_stats[2]) >= 0,
        "forbidden_2026_eligibility_rows": int(eligibility_stats[3]) == 0,
        "benchmark_2024_01_02_mainboard_eligible_count": int(eligibility_stats[4])
        == 1_888,
        "benchmark_2011_01_04_mainboard_eligible_count": int(eligibility_stats[5])
        == 90,
        "known_000527_gap_repaired": known_000527 == 1,
        "margin_detail_primary_key_unique": int(margin_stats[1]) == 0,
        "margin_detail_forbidden_2026_rows": int(margin_stats[2]) == 0,
        "request_2026_count_zero": int(state.get("request_2026_count", -1)) == 0,
    }
    if not all(checks.values()):
        raise MarginEligibilityUpdateError(f"margin_prepare_contract_failed:{checks}")
    state.update(
        {
            "status": "prepared",
            "input_dataset_ids": {
                DataDomain.MARGIN_DETAIL: margin_manifest.dataset_id,
                DataDomain.UNIVERSE_SNAPSHOT: universe_manifest.dataset_id,
            },
            "prepared": {
                DataDomain.MARGIN_ELIGIBILITY: [
                    str(path) for path in eligibility_paths
                ],
                DataDomain.MARGIN_DETAIL: [str(path) for path in repaired_margin_paths],
            },
            "annual_statistics": annual_stats,
            "exchange_detail_missing_count": missing_count,
            "exchange_qdp_conflict_field_count": conflict_count,
            "known_000527_repair_count": known_000527,
            "inventories": {
                "missing_detail": str(missing_path),
                "conflicts": str(conflicts_path),
            },
            "checks": checks,
        }
    )
    _write_state(workspace, state)
    return state


def _content_id(domain: str, paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(_sha256(path).encode("ascii"))
    return f"{domain}__{digest.hexdigest()[:24]}"


def _install(
    workspace: Path,
    *,
    domain: str,
    paths: Sequence[Path],
    input_manifest: DatasetManifest | None,
) -> tuple[str, dict[str, Any]]:
    root = qdp_v2_root(workspace)
    content_id = _content_id(domain, paths)
    dataset_id = (
        content_id
        if domain == DataDomain.MARGIN_ELIGIBILITY
        else f"{domain}__{hashlib.sha256(f'{content_id}|qdp_v2_tushare_margin_detail_raw_v3_mixed_provenance'.encode()).hexdigest()[:24]}"
    )
    target_root = root / "datasets" / domain / dataset_id
    entries: list[ShardManifestEntry] = []
    for source_path in paths:
        year = source_path.parent.name
        target = target_root / "shards" / year / "part-0000.parquet"
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp.parquet")
            shutil.copy2(source_path, temporary)
            os.replace(temporary, target)
        frame = pq.ParquetFile(target)
        entries.append(
            ShardManifestEntry(
                path=str(target.relative_to(root)).replace("\\", "/"),
                row_count=int(frame.metadata.num_rows),
                start_date=f"{year.split('=')[1]}-01-01",
                end_date=f"{year.split('=')[1]}-12-31",
                file_size=target.stat().st_size,
                metadata={"source_sha256": _sha256(source_path)},
            )
        )
    first_schema = pq.read_schema(paths[0])
    if domain == DataDomain.MARGIN_ELIGIBILITY:
        contract = "qdp_v2_margin_eligibility_exchange_tristate_v1"
        primary_key = ["trade_date", "symbol"]
        start_date = START_DATE
        end_date = END_DATE
        frequency = "1d"
        notes = [
            "eligible_observed, known_ineligible, and source_unavailable are distinct",
            "non-eligible securities never receive synthetic zero balances",
            "exchange data from date D is available on the next exchange-open date",
        ]
    else:
        assert input_manifest is not None
        contract = "qdp_v2_tushare_margin_detail_raw_v2_exchange_gap_repair"
        primary_key = list(input_manifest.primary_key)
        start_date = input_manifest.start_date
        end_date = input_manifest.end_date
        frequency = input_manifest.frequency
        notes = [
            *list(input_manifest.notes or []),
            "official exchange rows missing from QDP are appended without overwriting existing values",
            "cross-source differences are retained in a separate immutable conflict inventory",
        ]
    if domain == DataDomain.MARGIN_ELIGIBILITY:
        source = {
            "provider": "sse+szse_official_exchange",
            "checked_through": END_DATE,
            "scope": "point_in_time_historical_mainboard",
            "availability_semantics": "next_exchange_open_day",
            "request_2026_count": 0,
            "credential_persisted": False,
        }
    else:
        assert input_manifest is not None
        source = {
            **dict(input_manifest.source or {}),
            "provider": "tushare_compatible_primary+sse+szse_official_gap_repair",
            "query_granularity": "trade_date",
            "upstream_dataset_id": input_manifest.dataset_id,
            "exchange_gap_repair_update_id": UPDATE_ID,
            "exchange_gap_repair_row_count": 733,
            "availability_semantics": "next_exchange_open_day",
            "request_2026_count": 0,
            "credential_persisted": False,
        }
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        domain=domain,
        layer="raw",
        frequency=frequency,
        contract_version=contract,
        primary_key=primary_key,
        start_date=start_date,
        end_date=end_date,
        row_count=sum(item.row_count for item in entries),
        shards=entries,
        source=source,
        quality={
            "primary_key_unique": True,
            "source_unavailable_is_not_known_ineligible": True,
            "synthetic_zero_balance_rows": 0,
        },
        schema=_manifest_schema_from_arrow(first_schema),
        notes=notes,
    )
    _assert_credential_free(manifest.to_dict())
    write_dataset_manifest(root, manifest)
    return dataset_id, {
        "dataset_id": dataset_id,
        "row_count": manifest.row_count,
        "start_date": start_date,
        "end_date": end_date,
        "shard_count": len(entries),
    }


def repair_margin_detail_provenance(
    *, workspace_root: str | Path | None = None
) -> dict[str, Any]:
    """Create a metadata-correct immutable version without copying data shards."""

    workspace = _workspace(workspace_root)
    root = qdp_v2_root(workspace)
    state = _read_state(workspace)
    active = read_active_manifest(root)
    current = active_dataset_map(active)
    current_id = str(current.get(DataDomain.MARGIN_DETAIL, ""))
    current_path = dataset_manifest_for_id(root, current_id, DataDomain.MARGIN_DETAIL)
    if current_path is None:
        raise MarginEligibilityUpdateError("active_margin_detail_manifest_missing")
    current_manifest = read_dataset_manifest(current_path)
    if (
        current_manifest.contract_version
        == "qdp_v2_tushare_margin_detail_raw_v3_mixed_provenance"
        and current_manifest.source.get("query_granularity") == "trade_date"
    ):
        return {
            "status": "already_repaired",
            "dataset_id": current_id,
            "repair_id": PROVENANCE_REPAIR_ID,
        }
    upstream_id = str(
        dict(state.get("input_dataset_ids", {}) or {}).get(DataDomain.MARGIN_DETAIL, "")
    )
    upstream_path = dataset_manifest_for_id(root, upstream_id, DataDomain.MARGIN_DETAIL)
    if upstream_path is None:
        raise MarginEligibilityUpdateError(
            "margin_detail_provenance_upstream_manifest_missing"
        )
    upstream = read_dataset_manifest(upstream_path)
    digest = hashlib.sha256(
        (
            f"{current_id}|{upstream_id}|"
            "qdp_v2_tushare_margin_detail_raw_v3_mixed_provenance"
        ).encode("ascii")
    ).hexdigest()
    dataset_id = f"{DataDomain.MARGIN_DETAIL}__{digest[:24]}"
    source = {
        **dict(upstream.source or {}),
        "provider": "tushare_compatible_primary+sse+szse_official_gap_repair",
        "query_granularity": "trade_date",
        "upstream_dataset_id": upstream_id,
        "repaired_dataset_id": current_id,
        "exchange_gap_repair_update_id": UPDATE_ID,
        "exchange_gap_repair_row_count": int(
            state.get("exchange_detail_missing_count", 0)
        ),
        "availability_semantics": "next_exchange_open_day",
        "request_2026_count": 0,
        "credential_persisted": False,
    }
    repaired = DatasetManifest(
        dataset_id=dataset_id,
        domain=current_manifest.domain,
        layer=current_manifest.layer,
        frequency=current_manifest.frequency,
        contract_version="qdp_v2_tushare_margin_detail_raw_v3_mixed_provenance",
        primary_key=list(current_manifest.primary_key),
        start_date=current_manifest.start_date,
        end_date=current_manifest.end_date,
        row_count=current_manifest.row_count,
        shards=list(current_manifest.shards),
        source=source,
        quality={
            **dict(current_manifest.quality or {}),
            "provenance_repair_only": True,
            "physical_rows_unchanged": True,
        },
        schema=list(current_manifest.schema),
        notes=[
            *list(current_manifest.notes or []),
            "metadata-only immutable successor restores Tushare primary provenance and daily query granularity",
            "all physical shards and values are unchanged from the exchange-gap-repaired dataset",
        ],
    )
    _assert_credential_free(repaired.to_dict())
    write_dataset_manifest(root, repaired)
    active["datasets"] = {**current, DataDomain.MARGIN_DETAIL: dataset_id}
    active["updated_at"] = utc_now()
    write_active_manifest(root, active)
    if (
        active_dataset_map(read_active_manifest(root)).get(DataDomain.MARGIN_DETAIL)
        != dataset_id
    ):
        raise MarginEligibilityUpdateError("margin_detail_provenance_switch_failed")
    payload = {
        "status": "applied",
        "repair_id": PROVENANCE_REPAIR_ID,
        "input_dataset_id": current_id,
        "upstream_dataset_id": upstream_id,
        "dataset_id": dataset_id,
        "row_count": repaired.row_count,
        "shards_reused": len(repaired.shards),
        "physical_rows_unchanged": True,
        "query_granularity": "trade_date",
        "request_2026_count": 0,
        "applied_at": utc_now(),
    }
    atomic_write_json(root / "audits" / f"{PROVENANCE_REPAIR_ID}.json", payload)
    return payload


def commit(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    if state.get("status") == "applied":
        return state
    if state.get("status") != "prepared":
        raise MarginEligibilityUpdateError(
            f"margin_eligibility_not_prepared:{state.get('status')}"
        )
    margin_manifest, _ = _dataset_paths(workspace, DataDomain.MARGIN_DETAIL)
    ids: dict[str, str] = {}
    installed: dict[str, Any] = {}
    for domain in (DataDomain.MARGIN_ELIGIBILITY, DataDomain.MARGIN_DETAIL):
        dataset_id, record = _install(
            workspace,
            domain=domain,
            paths=[Path(path) for path in state["prepared"][domain]],
            input_manifest=(
                margin_manifest if domain == DataDomain.MARGIN_DETAIL else None
            ),
        )
        ids[domain] = dataset_id
        installed[domain] = record
    root = qdp_v2_root(workspace)
    active = read_active_manifest(root)
    current = active_dataset_map(active)
    if current.get(DataDomain.MARGIN_DETAIL) != dict(
        state.get("input_dataset_ids", {}) or {}
    ).get(DataDomain.MARGIN_DETAIL):
        raise MarginEligibilityUpdateError("active_margin_detail_drifted")
    active["datasets"] = {**current, **ids}
    active["updated_at"] = utc_now()
    write_active_manifest(root, active)
    state.update({"status": "applied", "installed_domains": installed})
    _write_state(workspace, state)
    atomic_write_json(root / "audits" / f"{UPDATE_ID}.json", state)
    return state


def evaluate(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    active = active_dataset_map(read_active_manifest(qdp_v2_root(workspace)))
    installed = dict(state.get("installed_domains", {}) or {})
    checks = {
        **dict(state.get("checks", {}) or {}),
        "eligibility_dataset_active": active.get(DataDomain.MARGIN_ELIGIBILITY)
        == dict(installed.get(DataDomain.MARGIN_ELIGIBILITY, {}) or {}).get(
            "dataset_id"
        ),
        "margin_detail_repair_dataset_active": active.get(DataDomain.MARGIN_DETAIL)
        == dict(installed.get(DataDomain.MARGIN_DETAIL, {}) or {}).get("dataset_id"),
    }
    return {
        "status": "ok" if checks and all(checks.values()) else "error",
        "update_id": UPDATE_ID,
        "checks": checks,
        "installed_domains": installed,
        "exchange_detail_missing_count": state.get("exchange_detail_missing_count", 0),
        "exchange_qdp_conflict_field_count": state.get(
            "exchange_qdp_conflict_field_count", 0
        ),
        "annual_statistics": state.get("annual_statistics", []),
    }


def run_pending(
    *,
    workspace_root: str | Path | None = None,
    max_workers: int = MAX_WORKERS,
    seal_runtime: bool = True,
) -> dict[str, Any]:
    download(workspace_root=workspace_root, max_workers=max_workers)
    prepare(workspace_root=workspace_root)
    result = commit(workspace_root=workspace_root)
    if seal_runtime:
        from quant_data_platform.qdp_v2.runtime_archive import (
            seal_completed_workflow,
        )

        result = {
            **result,
            "runtime_archive": seal_completed_workflow(
                UPDATE_ID,
                workspace_root=workspace_root,
            ),
        }
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp margin-eligibility-update")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-pending", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--repair-provenance", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = str(args.workspace_root or "") or None
    if args.run_pending:
        payload = run_pending(
            workspace_root=workspace,
            max_workers=int(args.max_workers),
        )
    elif args.evaluate:
        payload = evaluate(workspace_root=workspace)
    elif args.repair_provenance:
        payload = repair_margin_detail_provenance(workspace_root=workspace)
    else:
        payload = _read_state(_workspace(workspace))
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    return 0 if payload.get("status") not in {"error", "failed"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
