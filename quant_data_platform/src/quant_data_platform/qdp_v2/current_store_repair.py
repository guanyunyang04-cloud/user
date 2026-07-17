from __future__ import annotations

"""Targeted repair of the single mutable QDP store.

The module deliberately avoids generations or shadow datasets.  It prepares one
replacement per active shard under the repository runtime directory, validates
the replacements through the existing manifest transaction, switches the
current table in place, and removes the superseded shard.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.qdp_v2.duckdb_resources import open_guarded_duckdb
from quant_data_platform.qdp_v2.manifest import (
    atomic_write_json,
    qdp_v2_root,
    read_active_manifest,
    write_active_manifest,
)
from quant_data_platform.qdp_v2.repair import (
    ActiveDomain,
    mutate_active_shards_from_parquet,
    resolve_active_domain,
)


REPAIR_VERSION = 1
APPLY_REASON = (
    "restrict QDP to current listed main-board survivors and apply "
    "targeted 2026-07-17 quality corrections"
)
STATUS_SEMANTICS_REASON = (
    "normalize security_status suspension flags from daily market presence"
)
TUSHARE_API_URL = "https://ts.gyzcloud.top/api"
TUSHARE_MAX_RPM = 140
EXPECTED_BAR_TIMES = tuple(
    f"{hour:02d}{minute:02d}00000"
    for hour, minute in (
        *[(9, minute) for minute in range(35, 60, 5)],
        *[(10, minute) for minute in range(0, 60, 5)],
        *[(11, minute) for minute in range(0, 31, 5)],
        *[(13, minute) for minute in range(5, 60, 5)],
        *[(14, minute) for minute in range(0, 60, 5)],
        (15, 0),
    )
)
MARKET_COLUMNS = (
    "symbol",
    "trade_date",
    "bar_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "source",
    "adjusted_flag",
)
SYMBOL_DOMAINS = (
    "adjust_factor",
    "corporate_actions",
    "index_constituents",
    "industry_concept",
    "market_daily_raw",
    "market_intraday_5m",
    "name_change",
    "security_status",
    "share_capital",
    "universe_snapshot",
    "valuation",
)


class CurrentStoreRepairError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScopeInventory:
    as_of_date: str
    current_symbols: pd.DataFrame
    current_ids: pd.DataFrame
    input_symbols: pd.DataFrame
    aliases: pd.DataFrame
    targets: pd.DataFrame


class _SharedRateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self._interval = 60.0 / max(1, int(requests_per_minute))
        self._next_at = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + self._interval
        if wait:
            time.sleep(wait)


def _workspace(value: str | Path | None) -> Path:
    return Path(value or Path.cwd()).resolve()


def _runtime_root(workspace: Path) -> Path:
    root = (
        qdp_paths(workspace).data_dir
        / "qdp_runtime"
        / "current_store_repair_20260717"
    ).resolve()
    if workspace not in root.parents:
        raise CurrentStoreRepairError(f"repair_runtime_outside_workspace:{root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _corrections_path(workspace: Path) -> Path:
    return workspace / "quant_data_platform" / "configs" / "qdp_current_corrections.json"


def _load_corrections(workspace: Path) -> dict[str, Any]:
    path = _corrections_path(workspace)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != REPAIR_VERSION:
        raise CurrentStoreRepairError("current_corrections_version_mismatch")
    return payload


def _path_texts(context: ActiveDomain) -> list[str]:
    return [str(item) for item in context.shard_paths]


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_inventory(
    *, workspace_root: str | Path | None = None, persist: bool = True
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    runtime = _runtime_root(workspace)
    corrections = _load_corrections(workspace)
    active = read_active_manifest(qdp_v2_root(workspace))
    as_of_date = str(active.get("active_as_of_date", ""))
    if not as_of_date:
        raise CurrentStoreRepairError("active_as_of_date_missing")
    universe = resolve_active_domain("universe_snapshot", workspace_root=workspace)
    history = resolve_active_domain("symbol_history", workspace_root=workspace)
    daily = resolve_active_domain("market_daily_raw", workspace_root=workspace)
    intraday = resolve_active_domain("market_intraday_5m", workspace_root=workspace)

    spill = runtime / "inventory_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=4) as con:
        current_symbols = con.execute(
            """
            SELECT DISTINCT cast(symbol AS VARCHAR) AS current_symbol
            FROM read_parquet(?, union_by_name=true)
            WHERE cast(trade_date AS VARCHAR)=?
              AND lower(cast(board AS VARCHAR))='main'
              AND upper(cast(list_status AS VARCHAR))='L'
            ORDER BY 1
            """,
            [_path_texts(universe), as_of_date],
        ).fetchdf()
        if current_symbols.empty:
            raise CurrentStoreRepairError("current_survivor_scope_empty")
        con.register("qdp_current_symbols", current_symbols)
        current_ids = con.execute(
            """
            SELECT c.current_symbol,cast(h.security_id AS VARCHAR) AS security_id
            FROM qdp_current_symbols c
            JOIN read_parquet(?, union_by_name=true) h
              ON cast(h.symbol AS VARCHAR)=c.current_symbol
             AND cast(h.effective_from AS VARCHAR)<=?
             AND (coalesce(cast(h.effective_to AS VARCHAR),'')='' OR cast(h.effective_to AS VARCHAR)>=?)
            ORDER BY c.current_symbol
            """,
            [_path_texts(history), as_of_date, as_of_date],
        ).fetchdf()
        if len(current_ids) != len(current_symbols):
            raise CurrentStoreRepairError(
                f"current_identity_mapping_incomplete:{len(current_ids)}:{len(current_symbols)}"
            )
        if current_ids["current_symbol"].duplicated().any() or current_ids["security_id"].duplicated().any():
            raise CurrentStoreRepairError("current_identity_mapping_not_one_to_one")
        con.register("qdp_current_ids", current_ids)
        input_symbols = con.execute(
            """
            SELECT DISTINCT cast(h.symbol AS VARCHAR) AS source_symbol,
                            c.current_symbol,
                            c.security_id
            FROM read_parquet(?, union_by_name=true) h
            JOIN qdp_current_ids c
              ON cast(h.security_id AS VARCHAR)=c.security_id
            ORDER BY 1
            """,
            [_path_texts(history)],
        ).fetchdf()
        if input_symbols["source_symbol"].duplicated().any():
            raise CurrentStoreRepairError("historical_symbol_maps_to_multiple_current_ids")
        aliases = input_symbols.loc[
            input_symbols["source_symbol"].ne(input_symbols["current_symbol"])
        ].reset_index(drop=True)
        con.register("qdp_input_symbols", input_symbols)
        daily_fixes = pd.DataFrame(corrections.get("daily_from_5m", []))
        if daily_fixes.empty:
            daily_fixes = pd.DataFrame(columns=["symbol", "trade_date"])
        con.register("qdp_daily_fixes", daily_fixes.loc[:, ["symbol", "trade_date"]])
        target_frames: list[pd.DataFrame] = []
        daily_entries = list(zip(daily.shard_paths, daily.manifest.shards, strict=True))
        for intraday_path, entry in zip(
            intraday.shard_paths, intraday.manifest.shards, strict=True
        ):
            window_start = str(entry.start_date or "")
            window_end = str(entry.end_date or "")
            if not window_start or not window_end:
                raise CurrentStoreRepairError("intraday_inventory_shard_range_missing")
            daily_paths = [
                str(path)
                for path, daily_entry in daily_entries
                if not daily_entry.start_date
                or not daily_entry.end_date
                or (
                    str(daily_entry.start_date) <= window_end
                    and str(daily_entry.end_date) >= window_start
                )
            ]
            common = [[str(intraday_path)], window_start, window_end, daily_paths, window_start, window_end]
            severe_price = con.execute(
                """
                WITH five AS (
                  SELECT x.symbol,x.trade_date,arg_min(x.open,x.bar_time) AS iopen,
                         max(x.high) AS ihigh,min(x.low) AS ilow,arg_max(x.close,x.bar_time) AS iclose
                  FROM read_parquet(?, union_by_name=true) x
                  JOIN qdp_input_symbols s ON cast(x.symbol AS VARCHAR)=s.source_symbol
                  WHERE cast(x.trade_date AS VARCHAR) BETWEEN ? AND ?
                  GROUP BY x.symbol,x.trade_date
                ), joined AS (
                  SELECT d.symbol,d.trade_date,s.current_symbol,
                         greatest(abs(f.iopen/d.open-1),abs(f.ihigh/d.high-1),
                                  abs(f.ilow/d.low-1),abs(f.iclose/d.close-1)) AS price_rel_diff
                  FROM read_parquet(?, union_by_name=true) d
                  JOIN qdp_input_symbols s ON cast(d.symbol AS VARCHAR)=s.source_symbol
                  JOIN five f USING(symbol,trade_date)
                  WHERE cast(d.trade_date AS VARCHAR) BETWEEN ? AND ?
                    AND d.open>0 AND d.high>0 AND d.low>0 AND d.close>0
                )
                SELECT symbol AS source_symbol,current_symbol,trade_date,
                       'severe_price_mismatch' AS reason
                FROM joined j WHERE price_rel_diff>0.5
                  AND NOT EXISTS (SELECT 1 FROM qdp_daily_fixes f
                    WHERE f.symbol=j.symbol AND f.trade_date=j.trade_date)
                """,
                common,
            ).fetchdf()
            material_flow = con.execute(
                """
                WITH five AS (
                  SELECT x.symbol,x.trade_date,sum(x.volume) AS ivolume,sum(x.amount) AS iamount,
                         sum(CASE WHEN x.volume=0 AND x.amount=0 THEN 1 ELSE 0 END) AS zero_bars
                  FROM read_parquet(?, union_by_name=true) x
                  JOIN qdp_input_symbols s ON cast(x.symbol AS VARCHAR)=s.source_symbol
                  WHERE cast(x.trade_date AS VARCHAR) BETWEEN ? AND ?
                  GROUP BY x.symbol,x.trade_date
                ), joined AS (
                  SELECT d.symbol,d.trade_date,s.current_symbol,f.zero_bars,
                         abs(f.ivolume-d.volume)/greatest(abs(f.ivolume),abs(d.volume),1) AS volume_rel_diff,
                         abs(f.iamount-d.amount)/greatest(abs(f.iamount),abs(d.amount),1) AS amount_rel_diff
                  FROM read_parquet(?, union_by_name=true) d
                  JOIN qdp_input_symbols s ON cast(d.symbol AS VARCHAR)=s.source_symbol
                  JOIN five f USING(symbol,trade_date)
                  WHERE cast(d.trade_date AS VARCHAR) BETWEEN ? AND ?
                )
                SELECT symbol AS source_symbol,current_symbol,trade_date,
                       CASE WHEN zero_bars>=12 AND volume_rel_diff>0.05
                            THEN 'pseudo_complete_flow_mismatch'
                            ELSE 'material_flow_mismatch' END AS reason
                FROM joined j
                WHERE ((zero_bars>=12 AND volume_rel_diff>0.05)
                       OR (volume_rel_diff>0.05 AND amount_rel_diff>0.05))
                  AND NOT EXISTS (SELECT 1 FROM qdp_daily_fixes f
                    WHERE f.symbol=j.symbol AND f.trade_date=j.trade_date)
                """,
                common,
            ).fetchdf()
            missing = con.execute(
                """
                WITH eligible AS (
                  SELECT DISTINCT d.symbol AS source_symbol,s.current_symbol,d.trade_date
                  FROM read_parquet(?, union_by_name=true) d
                  JOIN qdp_input_symbols s ON cast(d.symbol AS VARCHAR)=s.source_symbol
                  WHERE cast(d.trade_date AS VARCHAR) BETWEEN ? AND ? AND d.volume>0
                ), complete AS (
                  SELECT x.symbol AS source_symbol,x.trade_date
                  FROM read_parquet(?, union_by_name=true) x
                  JOIN qdp_input_symbols s ON cast(x.symbol AS VARCHAR)=s.source_symbol
                  WHERE cast(x.trade_date AS VARCHAR) BETWEEN ? AND ?
                  GROUP BY 1,2 HAVING count(*)=48 AND count(DISTINCT x.bar_time)=48
                )
                SELECT e.source_symbol,e.current_symbol,e.trade_date,'missing_complete_day' AS reason
                FROM eligible e LEFT JOIN complete c USING(source_symbol,trade_date)
                WHERE c.source_symbol IS NULL
                """,
                [daily_paths, window_start, window_end, [str(intraday_path)], window_start, window_end],
            ).fetchdf()
            year_targets = pd.concat(
                [severe_price, material_flow, missing], ignore_index=True
            )
            target_frames.append(year_targets)
            print(json.dumps({"stage": "inventory_year", "year": window_start[:4], "target_days": int(len(year_targets))}, ensure_ascii=False), flush=True)
        targets = pd.concat(target_frames, ignore_index=True)

    forced = pd.DataFrame(corrections.get("force_refetch_5m", []))
    if not forced.empty:
        forced = forced.rename(columns={"symbol": "source_symbol"})
        forced = forced.merge(
            input_symbols.loc[:, ["source_symbol", "current_symbol"]],
            on="source_symbol",
            how="inner",
        )
        forced = forced.loc[:, ["source_symbol", "current_symbol", "trade_date", "reason"]]
        targets = pd.concat([targets, forced], ignore_index=True)
    targets = (
        targets.drop_duplicates(["source_symbol", "trade_date"], keep="last")
        .sort_values(["trade_date", "source_symbol"], kind="stable")
        .reset_index(drop=True)
    )
    inventory = ScopeInventory(
        as_of_date=as_of_date,
        current_symbols=current_symbols,
        current_ids=current_ids,
        input_symbols=input_symbols,
        aliases=aliases,
        targets=targets,
    )
    result = {
        "status": "planned",
        "as_of_date": as_of_date,
        "current_symbol_count": int(len(current_symbols)),
        "current_security_id_count": int(len(current_ids)),
        "historical_alias_count": int(len(aliases)),
        "target_5m_day_count": int(len(targets)),
        "target_reason_counts": {
            str(key): int(value)
            for key, value in targets["reason"].value_counts().sort_index().items()
        },
    }
    if persist:
        _write_frame(runtime / "current_symbols.parquet", inventory.current_symbols)
        _write_frame(runtime / "current_ids.parquet", inventory.current_ids)
        _write_frame(runtime / "input_symbols.parquet", inventory.input_symbols)
        _write_frame(runtime / "aliases.parquet", inventory.aliases)
        _write_frame(runtime / "target_5m_days.parquet", inventory.targets)
        atomic_write_json(runtime / "inventory.json", result)
    return result


def _load_inventory(workspace: Path) -> ScopeInventory:
    runtime = _runtime_root(workspace)
    metadata = json.loads((runtime / "inventory.json").read_text(encoding="utf-8"))
    return ScopeInventory(
        as_of_date=str(metadata["as_of_date"]),
        current_symbols=pd.read_parquet(runtime / "current_symbols.parquet"),
        current_ids=pd.read_parquet(runtime / "current_ids.parquet"),
        input_symbols=pd.read_parquet(runtime / "input_symbols.parquet"),
        aliases=pd.read_parquet(runtime / "aliases.parquet"),
        targets=pd.read_parquet(runtime / "target_5m_days.parquet"),
    )


def _fetch_tushare_day(
    row: Mapping[str, Any],
    *,
    token: str,
    limiter: _SharedRateLimiter,
) -> tuple[pd.DataFrame | None, str]:
    symbol = str(row["source_symbol"])
    current_symbol = str(row["current_symbol"])
    trade_date = str(row["trade_date"])
    payload = {
        "api_name": "stk_mins",
        "token": token,
        "params": {
            "ts_code": symbol,
            "freq": "5min",
            "start_date": f"{trade_date} 00:00:00",
            "end_date": f"{trade_date} 23:59:59",
            "limit": 8000,
        },
        "fields": "ts_code,trade_time,open,high,low,close,vol,amount",
    }
    last_error = "request_failed"
    for attempt in range(1, 4):
        limiter.acquire()
        try:
            response = requests.post(
                TUSHARE_API_URL,
                json=payload,
                headers={"Accept-Encoding": "gzip"},
                timeout=(10, 45),
            )
            if response.status_code >= 500:
                last_error = f"http_{response.status_code}"
                continue
            response.raise_for_status()
            body = response.json()
            if int(body.get("code", -1)) != 0:
                last_error = f"api_code_{body.get('code', 'unknown')}"
                continue
            data = body.get("data") or {}
            fields = list(data.get("fields") or [])
            items = list(data.get("items") or [])
            if not fields or any(len(item) != len(fields) for item in items):
                return None, "protocol_shape_invalid"
            frame = pd.DataFrame(items, columns=fields)
            if frame.empty:
                return None, "empty_response"
            timestamps = pd.to_datetime(frame["trade_time"], errors="coerce")
            if timestamps.isna().any():
                return None, "timestamp_invalid"
            frame["trade_date"] = timestamps.dt.strftime("%Y-%m-%d")
            frame["bar_time"] = timestamps.dt.strftime("%H%M00000")
            if set(frame["trade_date"].astype(str)) != {trade_date}:
                return None, "response_outside_day"
            if len(frame) != 48 or set(frame["bar_time"].astype(str)) != set(EXPECTED_BAR_TIMES):
                return None, "not_complete_48_bar_day"
            if frame["bar_time"].duplicated().any():
                return None, "duplicate_bar_time"
            frame = frame.rename(columns={"vol": "volume"})
            for column in ("open", "high", "low", "close", "volume", "amount"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            numeric = frame[["open", "high", "low", "close", "volume", "amount"]]
            if not np.isfinite(numeric.to_numpy(dtype=float)).all():
                return None, "numeric_nonfinite"
            if numeric[["open", "high", "low", "close"]].le(0).any().any():
                return None, "price_nonpositive"
            if numeric[["volume", "amount"]].lt(0).any().any():
                return None, "flow_negative"
            if (numeric["high"] < numeric[["open", "low", "close"]].max(axis=1)).any():
                return None, "high_invalid"
            if (numeric["low"] > numeric[["open", "high", "close"]].min(axis=1)).any():
                return None, "low_invalid"
            frame["symbol"] = current_symbol
            frame["source"] = "tushare_proxy_5m_targeted_repair_20260717"
            frame["adjusted_flag"] = "none"
            return (
                frame.loc[:, MARKET_COLUMNS]
                .sort_values("bar_time", kind="stable")
                .reset_index(drop=True),
                "",
            )
        except (requests.RequestException, ValueError, TypeError):
            last_error = f"request_or_json_error_attempt_{attempt}"
            time.sleep(float(attempt))
    return None, last_error


def _daily_reference(workspace: Path, inventory: ScopeInventory) -> pd.DataFrame:
    daily = resolve_active_domain("market_daily_raw", workspace_root=workspace)
    targets = inventory.targets.loc[:, ["source_symbol", "trade_date"]].copy()
    with open_guarded_duckdb(
        temp_directory=_runtime_root(workspace) / "download_validation_spill",
        threads=2,
    ) as con:
        con.register("qdp_targets", targets)
        return con.execute(
            """
            SELECT d.symbol AS source_symbol,d.trade_date,d.open,d.high,d.low,d.close,d.volume,d.amount
            FROM read_parquet(?, union_by_name=true) d
            JOIN qdp_targets t
              ON cast(d.symbol AS VARCHAR)=t.source_symbol
             AND cast(d.trade_date AS VARCHAR)=t.trade_date
            """,
            [_path_texts(daily)],
        ).fetchdf()


def _validate_download_against_daily(
    frame: pd.DataFrame, reference: Mapping[str, Any]
) -> str:
    ordered = frame.sort_values("bar_time", kind="stable")
    aggregate = {
        "open": float(ordered.iloc[0]["open"]),
        "high": float(ordered["high"].max()),
        "low": float(ordered["low"].min()),
        "close": float(ordered.iloc[-1]["close"]),
        "volume": float(ordered["volume"].sum()),
        "amount": float(ordered["amount"].sum()),
    }
    prices = ("open", "high", "low", "close")
    if any(
        abs(aggregate[key] / float(reference[key]) - 1.0) > 0.25
        for key in prices
        if float(reference[key]) > 0
    ):
        return "daily_price_sanity_failed"
    for key in ("volume", "amount"):
        expected = float(reference[key])
        actual = aggregate[key]
        if expected > 0 and not 0.5 <= actual / expected <= 2.0:
            return f"daily_{key}_unit_sanity_failed"
    return ""


def _resolve_tushare_token(token: str = "") -> str:
    secret = str(
        token
        or os.environ.get("QDP_TUSHARE_PROXY_TOKEN")
        or os.environ.get("QDP_TUSHARE_TOKEN")
        or os.environ.get("TUSHARE_TOKEN")
        or ""
    ).strip()
    if not secret:
        raise CurrentStoreRepairError("tushare_token_missing")
    return secret


def download_targeted_5m(
    *,
    workspace_root: str | Path | None = None,
    token: str = "",
    workers: int = 3,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    runtime = _runtime_root(workspace)
    inventory = _load_inventory(workspace)
    secret = _resolve_tushare_token(token)
    references = _daily_reference(workspace, inventory)
    reference_by_key = {
        (str(row.source_symbol), str(row.trade_date)): row._asdict()
        for row in references.itertuples(index=False)
    }
    limiter = _SharedRateLimiter(TUSHARE_MAX_RPM)
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    rows = inventory.targets.to_dict("records")
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, min(3, int(workers)))) as pool:
        future_map = {
            pool.submit(_fetch_tushare_day, row, token=secret, limiter=limiter): row
            for row in rows
        }
        completed = 0
        for future in as_completed(future_map):
            row = future_map[future]
            frame, error = future.result()
            key = (str(row["source_symbol"]), str(row["trade_date"]))
            if frame is not None:
                reference = reference_by_key.get(key)
                if reference is None:
                    error = "daily_reference_missing"
                else:
                    error = _validate_download_against_daily(frame, reference)
            if frame is None or error:
                failures.append(
                    {
                        "source_symbol": key[0],
                        "trade_date": key[1],
                        "error": str(error or "download_failed"),
                    }
                )
            else:
                frames.append(frame)
            completed += 1
            if completed % 50 == 0 or completed == len(rows):
                progress = {
                    "stage": "tushare_targeted_5m",
                    "completed": completed,
                    "total": len(rows),
                    "accepted": len(frames),
                    "failed": len(failures),
                }
                atomic_write_json(runtime / "download_progress.json", progress)
                print(
                    json.dumps(
                        progress,
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    downloaded = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=MARKET_COLUMNS)
    )
    if not downloaded.empty and downloaded.duplicated(
        ["symbol", "trade_date", "bar_time"]
    ).any():
        raise CurrentStoreRepairError("downloaded_5m_duplicate_output_keys")
    _write_frame(runtime / "downloaded_5m.parquet", downloaded)
    failure_frame = pd.DataFrame(
        failures, columns=["source_symbol", "trade_date", "error"]
    )
    _write_frame(runtime / "download_failures.parquet", failure_frame)
    result = {
        "status": "downloaded" if not failures else "downloaded_with_gaps",
        "target_day_count": int(len(rows)),
        "accepted_day_count": int(downloaded[["symbol", "trade_date"]].drop_duplicates().shape[0]),
        "accepted_row_count": int(len(downloaded)),
        "failed_day_count": int(len(failures)),
        "request_count": int(len(rows)),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    atomic_write_json(runtime / "download.json", result)
    return result


def retry_failed_5m(
    *,
    workspace_root: str | Path | None = None,
    token: str = "",
    workers: int = 3,
) -> dict[str, Any]:
    """Retry only failed target days and merge accepted bars atomically."""

    workspace = _workspace(workspace_root)
    runtime = _runtime_root(workspace)
    inventory = _load_inventory(workspace)
    secret = _resolve_tushare_token(token)

    failure_path = runtime / "download_failures.parquet"
    downloaded_path = runtime / "downloaded_5m.parquet"
    if not failure_path.is_file() or not downloaded_path.is_file():
        raise CurrentStoreRepairError("targeted_5m_download_missing")
    prior_failures = pd.read_parquet(failure_path)
    if prior_failures.empty:
        result = {
            "status": "nothing_to_retry",
            "retried_day_count": 0,
            "accepted_day_count": 0,
            "remaining_failed_day_count": 0,
        }
        atomic_write_json(runtime / "retry_download.json", result)
        return result

    failure_keys = prior_failures.loc[:, ["source_symbol", "trade_date"]].drop_duplicates()
    retry_targets = inventory.targets.merge(
        failure_keys,
        on=["source_symbol", "trade_date"],
        how="inner",
        validate="one_to_one",
    )
    if len(retry_targets) != len(failure_keys):
        raise CurrentStoreRepairError("retry_target_inventory_mismatch")

    references = _daily_reference(workspace, inventory)
    reference_by_key = {
        (str(row.source_symbol), str(row.trade_date)): row._asdict()
        for row in references.itertuples(index=False)
    }
    limiter = _SharedRateLimiter(TUSHARE_MAX_RPM)
    accepted: list[pd.DataFrame] = []
    rejected: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    rows = retry_targets.to_dict("records")
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, min(3, int(workers)))) as pool:
        future_map = {
            pool.submit(_fetch_tushare_day, row, token=secret, limiter=limiter): row
            for row in rows
        }
        completed = 0
        for future in as_completed(future_map):
            row = future_map[future]
            frame, error = future.result()
            key = (str(row["source_symbol"]), str(row["trade_date"]))
            if frame is not None:
                reference = reference_by_key.get(key)
                error = (
                    "daily_reference_missing"
                    if reference is None
                    else _validate_download_against_daily(frame, reference)
                )
            if frame is None or error:
                failures.append(
                    {
                        "source_symbol": key[0],
                        "trade_date": key[1],
                        "error": str(error or "download_failed"),
                    }
                )
                if frame is not None:
                    diagnostic = frame.copy()
                    diagnostic["source_symbol"] = key[0]
                    diagnostic["validation_error"] = str(error)
                    rejected.append(diagnostic)
            else:
                accepted.append(frame)
            completed += 1
            if completed % 25 == 0 or completed == len(rows):
                progress = {
                    "stage": "tushare_targeted_5m_retry",
                    "completed": completed,
                    "total": len(rows),
                    "accepted": len(accepted),
                    "failed": len(failures),
                }
                atomic_write_json(runtime / "retry_download_progress.json", progress)
                print(json.dumps(progress, ensure_ascii=False), flush=True)

    existing = pd.read_parquet(downloaded_path).loc[:, MARKET_COLUMNS]
    newly_accepted = (
        pd.concat(accepted, ignore_index=True)
        if accepted
        else pd.DataFrame(columns=MARKET_COLUMNS)
    )
    merge_frames = [existing]
    if not newly_accepted.empty:
        merge_frames.append(newly_accepted)
    merged = (
        pd.concat(merge_frames, ignore_index=True)
        .drop_duplicates(["symbol", "trade_date", "bar_time"], keep="last")
        .sort_values(["trade_date", "symbol", "bar_time"], kind="stable")
        .reset_index(drop=True)
    )
    day_counts = merged.groupby(["symbol", "trade_date"], sort=False).size()
    if not day_counts.eq(48).all():
        raise CurrentStoreRepairError("retried_5m_incomplete_merged_day")
    _write_frame(downloaded_path, merged)
    _write_frame(
        failure_path,
        pd.DataFrame(failures, columns=["source_symbol", "trade_date", "error"]),
    )
    rejected_frame = (
        pd.concat(rejected, ignore_index=True)
        if rejected
        else pd.DataFrame(columns=[*MARKET_COLUMNS, "source_symbol", "validation_error"])
    )
    _write_frame(runtime / "rejected_5m_diagnostics.parquet", rejected_frame)
    result = {
        "status": "retried" if not failures else "retried_with_gaps",
        "retried_day_count": int(len(rows)),
        "accepted_day_count": int(len(newly_accepted) // 48),
        "remaining_failed_day_count": int(len(failures)),
        "total_downloaded_day_count": int(len(day_counts)),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    atomic_write_json(runtime / "retry_download.json", result)
    return result


def _reconcile_rejected_day(
    frame: pd.DataFrame,
    reference: Mapping[str, Any],
    *,
    source: str = "tushare_proxy_5m_reconciled_to_baostock_daily_20260717",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = frame.loc[:, MARKET_COLUMNS].sort_values("bar_time", kind="stable").copy()
    if len(data) != 48 or set(data["bar_time"].astype(str)) != set(EXPECTED_BAR_TIMES):
        raise CurrentStoreRepairError("rejected_5m_day_contract_invalid")

    daily = {key: float(reference[key]) for key in ("open", "high", "low", "close", "volume", "amount")}
    if not all(math.isfinite(value) and value > 0 for value in daily.values()):
        raise CurrentStoreRepairError("rejected_5m_daily_reference_invalid")

    changed_fields: set[str] = set()
    row_floor = data[["open", "low", "close"]].max(axis=1)
    divided_high = data["high"] / 10.0
    decimal_high = (
        data["high"].gt(daily["high"] * 2.0)
        & divided_high.ge(row_floor * 0.999)
        & divided_high.le(daily["high"] * 1.02)
    )
    if decimal_high.any():
        data.loc[decimal_high, "high"] = divided_high.loc[decimal_high]
        changed_fields.add("high_decimal_shift")

    first = data.index[0]
    last = data.index[-1]
    if not math.isclose(float(data.at[first, "open"]), daily["open"], rel_tol=0.0, abs_tol=1.0e-9):
        data.at[first, "open"] = daily["open"]
        changed_fields.add("open")
    if not math.isclose(float(data.at[last, "close"]), daily["close"], rel_tol=0.0, abs_tol=1.0e-9):
        data.at[last, "close"] = daily["close"]
        changed_fields.add("close")

    for column in ("open", "close"):
        clipped = data[column].clip(lower=daily["low"], upper=daily["high"])
        if not clipped.equals(data[column]):
            changed_fields.add(f"{column}_range")
            data[column] = clipped
    implausible_low = data["low"].lt(daily["low"] * 0.5)
    if implausible_low.any():
        data.loc[implausible_low, "low"] = data.loc[
            implausible_low, ["open", "close", "high"]
        ].min(axis=1)
        changed_fields.add("low_decimal_shift")

    data["high"] = data[["open", "high", "low", "close"]].max(axis=1).clip(upper=daily["high"])
    data["low"] = data[["open", "high", "low", "close"]].min(axis=1).clip(lower=daily["low"])
    high_index = data["high"].idxmax()
    low_index = data["low"].idxmin()
    data.at[high_index, "high"] = daily["high"]
    data.at[low_index, "low"] = daily["low"]
    data["high"] = data[["open", "high", "low", "close"]].max(axis=1)
    data["low"] = data[["open", "high", "low", "close"]].min(axis=1)

    raw_volume = data["volume"].to_numpy(dtype=float)
    raw_amount = data["amount"].to_numpy(dtype=float)
    volume_total = float(raw_volume.sum())
    amount_total = float(raw_amount.sum())
    if volume_total <= 0 or amount_total <= 0:
        raise CurrentStoreRepairError("rejected_5m_flow_total_invalid")
    volume_scale = daily["volume"] / volume_total
    amount_scale = daily["amount"] / amount_total

    scaled_volume = raw_volume * volume_scale
    volume_floor = np.floor(scaled_volume)
    residual = int(round(daily["volume"] - float(volume_floor.sum())))
    if residual < 0 or residual > len(volume_floor):
        raise CurrentStoreRepairError("rejected_5m_volume_allocation_invalid")
    if residual:
        order = np.argsort(-(scaled_volume - volume_floor), kind="stable")
        volume_floor[order[:residual]] += 1.0
    data["volume"] = volume_floor

    scaled_amount = np.round(raw_amount * amount_scale, 2)
    amount_index = int(np.argmax(scaled_amount))
    scaled_amount[amount_index] += daily["amount"] - float(scaled_amount.sum())
    data["amount"] = scaled_amount
    data["source"] = source
    data["adjusted_flag"] = "none"

    error = _validate_download_against_daily(data, reference)
    if error:
        raise CurrentStoreRepairError(f"reconciled_5m_validation_failed:{error}")
    if (
        data[["open", "high", "low", "close"]].le(0).any().any()
        or data[["volume", "amount"]].lt(0).any().any()
        or (data["high"] < data[["open", "low", "close"]].max(axis=1)).any()
        or (data["low"] > data[["open", "high", "close"]].min(axis=1)).any()
    ):
        raise CurrentStoreRepairError("reconciled_5m_row_contract_invalid")
    return data.loc[:, MARKET_COLUMNS], {
        "price_repairs": ",".join(sorted(changed_fields)),
        "volume_scale": volume_scale,
        "amount_scale": amount_scale,
    }


def reconcile_rejected_5m(
    *, workspace_root: str | Path | None = None
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    runtime = _runtime_root(workspace)
    rejected_path = runtime / "rejected_5m_diagnostics.parquet"
    baostock_path = runtime / "baostock_daily_reference.parquet"
    downloaded_path = runtime / "downloaded_5m.parquet"
    failure_path = runtime / "download_failures.parquet"
    required = (rejected_path, baostock_path, downloaded_path, failure_path)
    if not all(path.is_file() for path in required):
        raise CurrentStoreRepairError("rejected_5m_reconciliation_input_missing")

    rejected = pd.read_parquet(rejected_path)
    baostock = pd.read_parquet(baostock_path)
    failures = pd.read_parquet(failure_path)
    if rejected.empty:
        result = {"status": "nothing_to_reconcile", "reconciled_day_count": 0}
        atomic_write_json(runtime / "reconcile_rejected.json", result)
        return result

    inventory = _load_inventory(workspace)
    current_daily = _daily_reference(workspace, inventory)
    keys = rejected.loc[:, ["source_symbol", "trade_date"]].drop_duplicates()
    evidence = (
        keys.merge(baostock, on=["source_symbol", "trade_date"], how="left", validate="one_to_one")
        .merge(
            current_daily,
            on=["source_symbol", "trade_date"],
            how="left",
            suffixes=("_baostock", "_qdp"),
            validate="one_to_one",
        )
    )
    fields = ("open", "high", "low", "close", "volume", "amount")
    if evidence[[f"{field}_baostock" for field in fields] + [f"{field}_qdp" for field in fields]].isna().any().any():
        raise CurrentStoreRepairError("rejected_5m_daily_evidence_missing")
    for field in fields:
        left = evidence[f"{field}_baostock"].to_numpy(dtype=float)
        right = evidence[f"{field}_qdp"].to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=1.0e-8, atol=1.0e-6):
            raise CurrentStoreRepairError(f"baostock_qdp_daily_mismatch:{field}")

    reference_by_key = {
        (str(row.source_symbol), str(row.trade_date)): {
            field: getattr(row, f"{field}_qdp") for field in fields
        }
        for row in evidence.itertuples(index=False)
    }
    repaired: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for (source_symbol, trade_date), day in rejected.groupby(
        ["source_symbol", "trade_date"], sort=True
    ):
        key = (str(source_symbol), str(trade_date))
        reference = reference_by_key[key]
        fixed, audit = _reconcile_rejected_day(day, reference)
        repaired.append(fixed)
        audit_rows.append(
            {
                "source_symbol": key[0],
                "symbol": str(fixed.iloc[0]["symbol"]),
                "trade_date": key[1],
                "validation_error": str(day.iloc[0]["validation_error"]),
                **audit,
            }
        )

    repaired_frame = pd.concat(repaired, ignore_index=True)
    existing = pd.read_parquet(downloaded_path).loc[:, MARKET_COLUMNS]
    merged = (
        pd.concat([existing, repaired_frame], ignore_index=True)
        .drop_duplicates(["symbol", "trade_date", "bar_time"], keep="last")
        .sort_values(["trade_date", "symbol", "bar_time"], kind="stable")
        .reset_index(drop=True)
    )
    day_counts = merged.groupby(["symbol", "trade_date"], sort=False).size()
    if not day_counts.eq(48).all():
        raise CurrentStoreRepairError("reconciled_5m_incomplete_merged_day")

    reconciled_keys = set(reference_by_key)
    remaining = failures.loc[
        ~failures.apply(
            lambda row: (str(row["source_symbol"]), str(row["trade_date"])) in reconciled_keys,
            axis=1,
        )
    ].reset_index(drop=True)
    _write_frame(downloaded_path, merged)
    _write_frame(failure_path, remaining)
    audit_path = runtime / "reconciled_5m_audit.parquet"
    new_audit = pd.DataFrame(audit_rows)
    if audit_path.is_file():
        new_audit = (
            pd.concat([pd.read_parquet(audit_path), new_audit], ignore_index=True)
            .drop_duplicates(["source_symbol", "trade_date"], keep="last")
            .sort_values(["trade_date", "source_symbol"], kind="stable")
            .reset_index(drop=True)
        )
    _write_frame(audit_path, new_audit)
    result = {
        "status": "reconciled",
        "reconciled_day_count": int(len(audit_rows)),
        "total_reconciled_audit_day_count": int(len(new_audit)),
        "remaining_failed_day_count": int(len(remaining)),
        "total_downloaded_day_count": int(len(day_counts)),
    }
    atomic_write_json(runtime / "reconcile_rejected.json", result)
    return result


def _aggregate_local_1m_gap(
    raw: pd.DataFrame,
    *,
    current_symbol: str,
    trade_date: str,
) -> pd.DataFrame:
    date_column = "\u65e5\u671f"
    columns = {
        "open": "\u5f00\u76d8",
        "high": "\u6700\u9ad8",
        "low": "\u6700\u4f4e",
        "close": "\u6536\u76d8",
        "volume": "\u6210\u4ea4\u91cf(\u80a1)",
        "amount": "\u6210\u4ea4\u989d(\u5143)",
    }
    required = {date_column, *columns.values()}
    if not required.issubset(raw.columns):
        raise CurrentStoreRepairError("local_1m_gap_schema_invalid")
    timestamps = pd.to_datetime(raw[date_column], errors="coerce")
    day = raw.loc[timestamps.dt.strftime("%Y-%m-%d").eq(trade_date)].copy()
    day["timestamp"] = timestamps.loc[day.index]
    minutes = day["timestamp"].dt.hour * 60 + day["timestamp"].dt.minute
    session = ((minutes > 570) & (minutes <= 690)) | (
        (minutes > 780) & (minutes <= 900)
    )
    day = day.loc[session].copy()
    if len(day) != 240 or day["timestamp"].duplicated().any():
        raise CurrentStoreRepairError(
            f"local_1m_gap_minute_contract_invalid:{current_symbol}:{trade_date}:{len(day)}"
        )
    minutes = day["timestamp"].dt.hour * 60 + day["timestamp"].dt.minute
    starts = np.where(minutes <= 690, 570, 780)
    labels = starts + (((minutes - starts - 1) // 5) + 1) * 5
    day["bar_time"] = [
        f"{int(value // 60):02d}{int(value % 60):02d}00000" for value in labels
    ]
    for canonical, source_column in columns.items():
        day[canonical] = pd.to_numeric(day[source_column], errors="coerce")
    values = day[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise CurrentStoreRepairError("local_1m_gap_numeric_invalid")
    bars = (
        day.groupby("bar_time", as_index=False, sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            amount=("amount", "sum"),
        )
    )
    if len(bars) != 48 or set(bars["bar_time"].astype(str)) != set(EXPECTED_BAR_TIMES):
        raise CurrentStoreRepairError("local_1m_gap_48_bar_contract_invalid")
    bars["symbol"] = current_symbol
    bars["trade_date"] = trade_date
    bars["source"] = "baidu_local_mootdx_1m_to_5m_gap_fill_20260717"
    bars["adjusted_flag"] = "none"
    return bars.loc[:, MARKET_COLUMNS]


def fill_local_1m_gaps(
    *, workspace_root: str | Path | None = None
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    runtime = _runtime_root(workspace)
    corrections = _load_corrections(workspace)
    entries = list(corrections.get("local_1m_gap_fills", []))
    source_root = Path(str(corrections.get("local_minute_source_root", "")))
    downloaded_path = runtime / "downloaded_5m.parquet"
    failure_path = runtime / "download_failures.parquet"
    if not entries or not source_root.is_dir():
        raise CurrentStoreRepairError("local_1m_gap_configuration_invalid")
    if not downloaded_path.is_file() or not failure_path.is_file():
        raise CurrentStoreRepairError("targeted_5m_download_missing")

    inventory = _load_inventory(workspace)
    current_daily = _daily_reference(workspace, inventory)
    reference_by_key = {
        (str(row.source_symbol), str(row.trade_date)): row._asdict()
        for row in current_daily.itertuples(index=False)
    }
    repaired: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    configured_keys: set[tuple[str, str]] = set()
    for entry in entries:
        source_symbol = str(entry["source_symbol"])
        current_symbol = str(entry["current_symbol"])
        trade_date = str(entry["trade_date"])
        key = (source_symbol, trade_date)
        configured_keys.add(key)
        zip_path = (source_root / str(entry["zip"])).resolve()
        if source_root.resolve() not in zip_path.parents or not zip_path.is_file():
            raise CurrentStoreRepairError(f"local_1m_gap_zip_invalid:{zip_path}")
        with zipfile.ZipFile(zip_path) as archive:
            member = str(entry["member"])
            if member not in archive.namelist():
                raise CurrentStoreRepairError(f"local_1m_gap_member_missing:{member}")
            with archive.open(member) as handle:
                raw = pd.read_csv(handle, encoding="utf-8-sig")
        bars = _aggregate_local_1m_gap(
            raw,
            current_symbol=current_symbol,
            trade_date=trade_date,
        )
        reference = reference_by_key.get(key)
        if reference is None:
            raise CurrentStoreRepairError(f"local_1m_gap_daily_missing:{source_symbol}:{trade_date}")
        fixed, audit = _reconcile_rejected_day(
            bars,
            reference,
            source="baidu_local_mootdx_1m_to_5m_reconciled_20260717",
        )
        repaired.append(fixed)
        audit_rows.append(
            {
                "source_symbol": source_symbol,
                "symbol": current_symbol,
                "trade_date": trade_date,
                "zip": str(entry["zip"]),
                "member": str(entry["member"]),
                **audit,
            }
        )

    repaired_frame = pd.concat(repaired, ignore_index=True)
    existing = pd.read_parquet(downloaded_path).loc[:, MARKET_COLUMNS]
    merged = (
        pd.concat([existing, repaired_frame], ignore_index=True)
        .drop_duplicates(["symbol", "trade_date", "bar_time"], keep="last")
        .sort_values(["trade_date", "symbol", "bar_time"], kind="stable")
        .reset_index(drop=True)
    )
    day_counts = merged.groupby(["symbol", "trade_date"], sort=False).size()
    if not day_counts.eq(48).all():
        raise CurrentStoreRepairError("local_1m_gap_merged_day_invalid")
    failures = pd.read_parquet(failure_path)
    remaining = failures.loc[
        ~failures.apply(
            lambda row: (str(row["source_symbol"]), str(row["trade_date"]))
            in configured_keys,
            axis=1,
        )
    ].reset_index(drop=True)
    _write_frame(downloaded_path, merged)
    _write_frame(failure_path, remaining)
    _write_frame(runtime / "local_1m_gap_fill_audit.parquet", pd.DataFrame(audit_rows))
    result = {
        "status": "filled",
        "filled_day_count": int(len(audit_rows)),
        "remaining_failed_day_count": int(len(remaining)),
        "total_downloaded_day_count": int(len(day_counts)),
    }
    atomic_write_json(runtime / "fill_local_1m_gaps.json", result)
    return result


def reconcile_all_targeted_5m(
    *, workspace_root: str | Path | None = None
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    runtime = _runtime_root(workspace)
    downloaded_path = runtime / "downloaded_5m.parquet"
    failure_path = runtime / "download_failures.parquet"
    if not downloaded_path.is_file() or not failure_path.is_file():
        raise CurrentStoreRepairError("targeted_5m_download_missing")

    inventory = _load_inventory(workspace)
    current_daily = _daily_reference(workspace, inventory)
    symbols = inventory.input_symbols.loc[:, ["source_symbol", "current_symbol"]]
    evidence = current_daily.merge(
        symbols,
        on="source_symbol",
        how="left",
        validate="many_to_one",
    )
    reference_by_key = {
        (str(row.current_symbol), str(row.trade_date)): row._asdict()
        for row in evidence.itertuples(index=False)
    }
    downloaded = pd.read_parquet(downloaded_path).loc[:, MARKET_COLUMNS]
    repaired: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    suspended_rows: list[dict[str, str]] = []
    for (symbol, trade_date), day in downloaded.groupby(
        ["symbol", "trade_date"], sort=True
    ):
        key = (str(symbol), str(trade_date))
        reference = reference_by_key.get(key)
        if reference is None:
            raise CurrentStoreRepairError(f"targeted_5m_daily_missing:{symbol}:{trade_date}")
        if float(reference["volume"]) <= 0 or float(reference["amount"]) <= 0:
            suspended_rows.append(
                {
                    "source_symbol": str(reference["source_symbol"]),
                    "trade_date": str(trade_date),
                    "error": "daily_suspended_nonzero_5m_rejected",
                }
            )
            continue
        original_source = str(day.iloc[0]["source"])
        fixed, audit = _reconcile_rejected_day(
            day,
            reference,
            source="qdp_targeted_5m_reconciled_to_daily_20260717",
        )
        repaired.append(fixed)
        audit_rows.append(
            {
                "source_symbol": str(reference["source_symbol"]),
                "symbol": str(symbol),
                "trade_date": str(trade_date),
                "original_source": original_source,
                **audit,
            }
        )

    repaired_frame = pd.concat(repaired, ignore_index=True)
    day_counts = repaired_frame.groupby(["symbol", "trade_date"], sort=False).size()
    if not day_counts.eq(48).all():
        raise CurrentStoreRepairError("targeted_5m_final_day_contract_invalid")
    failures = pd.concat(
        [
            pd.read_parquet(failure_path),
            pd.DataFrame(
                suspended_rows,
                columns=["source_symbol", "trade_date", "error"],
            ),
        ],
        ignore_index=True,
    ).drop_duplicates(["source_symbol", "trade_date"], keep="last")
    _write_frame(downloaded_path, repaired_frame)
    _write_frame(failure_path, failures)
    _write_frame(
        runtime / "targeted_5m_final_reconciliation_audit.parquet",
        pd.DataFrame(audit_rows),
    )
    result = {
        "status": "reconciled",
        "reconciled_day_count": int(len(audit_rows)),
        "suspended_day_rejected_count": int(len(suspended_rows)),
        "remaining_failed_day_count": int(len(failures)),
        "total_downloaded_day_count": int(len(day_counts)),
    }
    atomic_write_json(runtime / "reconcile_all_targeted_5m.json", result)
    return result


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _normalized_primary_key_sql(context: ActiveDomain) -> str:
    if not context.manifest.primary_key:
        raise CurrentStoreRepairError(f"domain_primary_key_missing:{context.domain}")
    return ",".join(
        "s.current_symbol" if key == "symbol" else f"d.{_sql_identifier(key)}"
        for key in context.manifest.primary_key
    )


def _domain_repair_already_applied(context: ActiveDomain) -> bool:
    return bool(context.manifest.shards) and all(
        str(entry.metadata.get("repair_reason", "")) == APPLY_REASON
        for entry in context.manifest.shards
    )


def _copy_query(con: Any, query: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        con.execute(
            f"COPY ({query}) TO {_sql_literal(temporary)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)"
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _prevalidate_intraday_replacements(
    context: ActiveDomain,
    replacements: Sequence[tuple[Path, Path]],
    *,
    workspace: Path,
) -> str:
    if not replacements:
        raise CurrentStoreRepairError("intraday_prevalidation_replacements_missing")
    entry_by_path = {
        str(path.resolve()): entry
        for path, entry in zip(context.shard_paths, context.manifest.shards, strict=True)
    }
    replacement_old_paths = {str(old_path.resolve()) for old_path, _ in replacements}
    if len(replacement_old_paths) != len(replacements) or not replacement_old_paths.issubset(entry_by_path):
        raise CurrentStoreRepairError("intraday_prevalidation_old_shards_invalid")
    ranges: list[tuple[str, str]] = []
    with open_guarded_duckdb(
        temp_directory=_runtime_root(workspace) / "intraday_key_prevalidation_spill",
        threads=1,
    ) as con:
        for index, (_, prepared_path) in enumerate(replacements):
            row = con.execute(
                f"""
                WITH ordered AS (
                  SELECT cast(symbol AS VARCHAR) AS symbol,
                         cast(trade_date AS VARCHAR) AS trade_date,
                         cast(bar_time AS VARCHAR) AS bar_time,
                         lag(cast(symbol AS VARCHAR)) OVER () AS prior_symbol,
                         lag(cast(trade_date AS VARCHAR)) OVER () AS prior_date,
                         lag(cast(bar_time AS VARCHAR)) OVER () AS prior_time
                  FROM read_parquet({_sql_literal(prepared_path)}, union_by_name=false)
                )
                SELECT min(trade_date),max(trade_date),
                       max(CASE
                         WHEN symbol IS NULL OR trade_date IS NULL OR bar_time IS NULL THEN 1
                         WHEN prior_symbol IS NULL THEN 0
                         WHEN symbol < prior_symbol THEN 1
                         WHEN symbol = prior_symbol AND trade_date < prior_date THEN 1
                         WHEN symbol = prior_symbol AND trade_date = prior_date
                              AND bar_time <= prior_time THEN 1
                         ELSE 0 END) AS invalid_order_or_duplicate
                FROM ordered
                """
            ).fetchone()
            start_date, end_date, invalid = str(row[0] or ""), str(row[1] or ""), int(row[2] or 0)
            if not start_date or not end_date or invalid:
                raise CurrentStoreRepairError(
                    f"intraday_prevalidated_primary_key_invalid:{index}"
                )
            old_entry = entry_by_path[str(replacements[index][0].resolve())]
            if (
                (old_entry.start_date and start_date < str(old_entry.start_date))
                or (old_entry.end_date and end_date > str(old_entry.end_date))
            ):
                raise CurrentStoreRepairError(
                    f"intraday_prevalidated_range_outside_replaced_shard:{index}"
                )
            ranges.append((start_date, end_date))
            print(
                json.dumps(
                    {
                        "stage": "prevalidate_intraday_shard",
                        "completed": index + 1,
                        "total": len(replacements),
                        "start_date": start_date,
                        "end_date": end_date,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    ordered_ranges = sorted(ranges)
    if any(left[1] >= right[0] for left, right in zip(ordered_ranges, ordered_ranges[1:])):
        raise CurrentStoreRepairError("intraday_prevalidated_date_ranges_overlap")
    retained_ranges = [
        (str(entry.start_date), str(entry.end_date))
        for path, entry in zip(context.shard_paths, context.manifest.shards, strict=True)
        if str(path.resolve()) not in replacement_old_paths
        and str(entry.start_date)
        and str(entry.end_date)
    ]
    if any(
        start <= retained_end and retained_start <= end
        for start, end in ordered_ranges
        for retained_start, retained_end in retained_ranges
    ):
        raise CurrentStoreRepairError("intraday_prevalidated_retained_range_overlap")

    root = context.root.resolve()
    mutation_replacements = sorted(
        (
            os.path.normcase(os.path.abspath(str(old_path.resolve()))),
            _sha256_file(prepared_path),
        )
        for old_path, prepared_path in replacements
    )
    payload = {
        "version": 1,
        "domain": context.domain,
        "dataset_id": context.dataset_id,
        "replacements": mutation_replacements,
        "removals": [],
        "appends": [],
    }
    if root != context.root.resolve():
        raise CurrentStoreRepairError("intraday_prevalidation_root_changed")
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"shard-mutation-v1:{hashlib.sha256(encoded).hexdigest()}"


def _factor_pending_corrections(
    workspace: Path, corrections: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    factor = resolve_active_domain("adjust_factor", workspace_root=workspace)
    with open_guarded_duckdb(
        temp_directory=_runtime_root(workspace) / "factor_guard_spill", threads=2
    ) as con:
        rows = con.execute(
            """
            WITH s AS (
              SELECT symbol,trade_date,adjust_factor,
                     lag(adjust_factor) OVER(PARTITION BY symbol ORDER BY trade_date) AS prior_factor
              FROM read_parquet(?, union_by_name=true)
            )
            SELECT symbol,trade_date,adjust_factor/prior_factor AS ratio
            FROM s WHERE prior_factor>0
            """,
            [_path_texts(factor)],
        ).fetchall()
    ratios = {(str(symbol), str(date)): float(ratio) for symbol, date, ratio in rows}
    pending_shifts: list[dict[str, Any]] = []
    for item in corrections.get("factor_shifts", []):
        expected = float(item["expected_ratio"])
        old_ratio = ratios.get((str(item["symbol"]), str(item["from_date"])))
        new_ratio = ratios.get((str(item["symbol"]), str(item["to_date"])))
        if old_ratio is not None and math.isclose(old_ratio, expected, rel_tol=1e-8, abs_tol=1e-10):
            pending_shifts.append(dict(item))
        elif old_ratio is not None and new_ratio is not None and math.isclose(old_ratio, 1.0, rel_tol=1e-8, abs_tol=1e-10) and math.isclose(new_ratio, expected, rel_tol=1e-8, abs_tol=1e-10):
            continue
        else:
            raise CurrentStoreRepairError(
                f"factor_shift_old_value_mismatch:{item['symbol']}:{item['from_date']}:{old_ratio}"
            )
    pending_neutral: list[dict[str, Any]] = []
    for item in corrections.get("factor_neutralizations", []):
        expected = float(item["expected_ratio"])
        old_ratio = ratios.get((str(item["symbol"]), str(item["from_date"])))
        if old_ratio is not None and math.isclose(old_ratio, expected, rel_tol=1e-8, abs_tol=1e-10):
            pending_neutral.append(dict(item))
        elif old_ratio is not None and math.isclose(old_ratio, 1.0, rel_tol=1e-8, abs_tol=1e-10):
            continue
        else:
            raise CurrentStoreRepairError(
                f"factor_neutralization_old_value_mismatch:{item['symbol']}:{item['from_date']}:{old_ratio}"
            )
    return pending_shifts, pending_neutral


def _factor_multiplier_sql(
    shifts: Sequence[Mapping[str, Any]], neutral: Sequence[Mapping[str, Any]]
) -> str:
    clauses: list[str] = []
    for item in shifts:
        clauses.append(
            "WHEN d.symbol={symbol} AND d.trade_date>={to_date} AND d.trade_date<{from_date} THEN {ratio}".format(
                symbol=_sql_literal(str(item["symbol"])),
                to_date=_sql_literal(str(item["to_date"])),
                from_date=_sql_literal(str(item["from_date"])),
                ratio=repr(float(item["expected_ratio"])),
            )
        )
    for item in neutral:
        clauses.append(
            "WHEN d.symbol={symbol} AND d.trade_date>={from_date} THEN {ratio}".format(
                symbol=_sql_literal(str(item["symbol"])),
                from_date=_sql_literal(str(item["from_date"])),
                ratio=repr(1.0 / float(item["expected_ratio"])),
            )
        )
    return "CASE " + " ".join(clauses) + " ELSE 1.0 END"


def _derive_daily_fixes(
    workspace: Path, corrections: Mapping[str, Any]
) -> pd.DataFrame:
    requested = pd.DataFrame(corrections.get("daily_from_5m", []))
    if requested.empty:
        return pd.DataFrame()
    intraday = resolve_active_domain("market_intraday_5m", workspace_root=workspace)
    with open_guarded_duckdb(
        temp_directory=_runtime_root(workspace) / "daily_fix_spill", threads=2
    ) as con:
        con.register("qdp_daily_fix_keys", requested)
        frame = con.execute(
            """
            SELECT x.symbol,x.trade_date,
                   first(x.open ORDER BY x.bar_time) AS open,
                   max(x.high) AS high,min(x.low) AS low,
                   last(x.close ORDER BY x.bar_time) AS close,
                   sum(x.volume) AS volume,sum(x.amount) AS amount,
                   count(*) AS bars,count(DISTINCT x.bar_time) AS times
            FROM read_parquet(?, union_by_name=true) x
            JOIN qdp_daily_fix_keys k USING(symbol,trade_date)
            GROUP BY x.symbol,x.trade_date
            """,
            [_path_texts(intraday)],
        ).fetchdf()
    if len(frame) != len(requested) or not frame["bars"].eq(48).all() or not frame["times"].eq(48).all():
        raise CurrentStoreRepairError("daily_fix_intraday_reference_incomplete")
    frame["source"] = "mootdx_5m_daily_repair_20260717"
    return frame.drop(columns=["bars", "times"])


def _register_scope_tables(con: Any, inventory: ScopeInventory) -> None:
    con.register("qdp_current_symbols", inventory.current_symbols)
    con.register("qdp_current_ids", inventory.current_ids)
    con.register("qdp_input_symbols", inventory.input_symbols)
    con.register("qdp_aliases", inventory.aliases)


def _prepared_replacements(
    context: ActiveDomain,
    *,
    workspace: Path,
    inventory: ScopeInventory,
    downloaded: pd.DataFrame,
    daily_fixes: pd.DataFrame,
    pending_shifts: Sequence[Mapping[str, Any]],
    pending_neutral: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[Path, Path]], list[Path]]:
    runtime = _runtime_root(workspace)
    prepared_dir = runtime / "prepared" / context.domain
    prepared_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = {
        str(path.resolve()): entry
        for path, entry in zip(context.shard_paths, context.manifest.shards, strict=True)
    }
    replacements: list[tuple[Path, Path]] = []
    removals: list[Path] = []
    with open_guarded_duckdb(
        temp_directory=runtime / "rewrite_spill" / context.domain,
        threads=4,
    ) as con:
        _register_scope_tables(con, inventory)
        con.register("qdp_targets", inventory.targets)
        con.register("qdp_downloaded", downloaded)
        con.register("qdp_daily_fix_rows", daily_fixes)
        factor_multiplier = _factor_multiplier_sql(pending_shifts, pending_neutral)
        normalized_key = (
            _normalized_primary_key_sql(context)
            if context.domain in SYMBOL_DOMAINS
            else ""
        )
        source_preference = """
            CASE WHEN lower(coalesce(cast(d.source AS VARCHAR),'')) LIKE 'baostock%'
                 THEN 0 ELSE 1 END,
            cast(d.source AS VARCHAR)
        """
        alias_preferred_join = ""
        if context.domain == "market_daily_raw":
            con.execute(
                """
                CREATE TEMP TABLE qdp_preferred_daily_keys AS
                SELECT DISTINCT s.current_symbol AS symbol,
                       cast(d.trade_date AS VARCHAR) AS trade_date
                FROM read_parquet(?, union_by_name=true) d
                JOIN qdp_input_symbols s
                  ON cast(d.symbol AS VARCHAR)=s.source_symbol
                WHERE lower(coalesce(cast(d.source AS VARCHAR),'')) LIKE 'baostock%'
                """,
                [_path_texts(context)],
            )
        elif context.domain == "adjust_factor":
            con.execute(
                """
                CREATE TEMP TABLE qdp_preferred_factor_keys AS
                SELECT DISTINCT s.current_symbol AS symbol,
                       cast(d.trade_date AS VARCHAR) AS trade_date
                FROM read_parquet(?, union_by_name=true) d
                JOIN qdp_input_symbols s
                  ON cast(d.symbol AS VARCHAR)=s.source_symbol
                WHERE lower(coalesce(cast(d.factor_provider AS VARCHAR),''))
                      LIKE 'tushare%'
                """,
                [_path_texts(context)],
            )
        elif context.domain == "universe_snapshot":
            con.execute(
                """
                CREATE TEMP TABLE qdp_preferred_universe_keys AS
                SELECT symbol,trade_date,filename AS preferred_filename
                FROM (
                  SELECT s.current_symbol AS symbol,
                         cast(d.trade_date AS VARCHAR) AS trade_date,
                         filename,
                         row_number() OVER(
                           PARTITION BY s.current_symbol,cast(d.trade_date AS VARCHAR)
                           ORDER BY
                             CASE WHEN coalesce(cast(d.list_date AS VARCHAR),'')<>''
                                  THEN 0 ELSE 1 END,
                             CASE WHEN coalesce(cast(d.delist_date AS VARCHAR),'')<>''
                                  THEN 0 ELSE 1 END,
                             CASE WHEN lower(coalesce(cast(d.source AS VARCHAR),''))
                                       LIKE 'baostock%'
                                  THEN 0 ELSE 1 END,
                             filename DESC
                         ) AS preference,
                         count(*) OVER(
                           PARTITION BY s.current_symbol,cast(d.trade_date AS VARCHAR)
                         ) AS candidates
                  FROM read_parquet(?, union_by_name=true, filename=true) d
                  JOIN qdp_input_symbols s
                    ON cast(d.symbol AS VARCHAR)=s.source_symbol
                  JOIN qdp_aliases a
                    ON s.current_symbol=a.current_symbol
                )
                WHERE preference=1 AND candidates>1
                """,
                [_path_texts(context)],
            )
        elif context.domain == "security_status":
            con.execute(
                """
                CREATE TEMP TABLE qdp_preferred_status_keys AS
                SELECT symbol,trade_date,filename AS preferred_filename
                FROM (
                  SELECT s.current_symbol AS symbol,
                         cast(d.trade_date AS VARCHAR) AS trade_date,
                         filename,
                         row_number() OVER(
                           PARTITION BY s.current_symbol,cast(d.trade_date AS VARCHAR)
                           ORDER BY
                             CASE WHEN coalesce(cast(d.status_reason AS VARCHAR),'')<>''
                                  THEN 0 ELSE 1 END,
                             CASE WHEN lower(coalesce(cast(d.source AS VARCHAR),''))
                                       LIKE 'tushare%'
                                  THEN 0 ELSE 1 END,
                             filename
                         ) AS preference,
                         count(*) OVER(
                           PARTITION BY s.current_symbol,cast(d.trade_date AS VARCHAR)
                         ) AS candidates
                  FROM read_parquet(?, union_by_name=true, filename=true) d
                  JOIN qdp_input_symbols s
                    ON cast(d.symbol AS VARCHAR)=s.source_symbol
                  JOIN qdp_aliases a
                    ON s.current_symbol=a.current_symbol
                )
                WHERE preference=1 AND candidates>1
                """,
                [_path_texts(context)],
            )
        for index, old_path in enumerate(context.shard_paths):
            old = _sql_literal(old_path)
            entry = manifest_entries[str(old_path.resolve())]
            if context.domain == "security_identity":
                query = (
                    f"SELECT d.* FROM read_parquet({old}, union_by_name=true) d "
                    "JOIN qdp_current_ids c ON cast(d.security_id AS VARCHAR)=c.security_id"
                )
            elif context.domain == "symbol_history":
                query = (
                    f"SELECT d.* FROM read_parquet({old}, union_by_name=true) d "
                    "JOIN qdp_current_ids c ON cast(d.security_id AS VARCHAR)=c.security_id"
                )
            elif context.domain == "market_intraday_5m":
                year = str(entry.start_date or entry.end_date)[:4]
                if not year:
                    raise CurrentStoreRepairError("intraday_shard_year_missing")
                query = f"""
                    SELECT d.* REPLACE(s.current_symbol AS symbol)
                    FROM read_parquet({old}, union_by_name=true) d
                    JOIN qdp_input_symbols s ON cast(d.symbol AS VARCHAR)=s.source_symbol
                    LEFT JOIN qdp_targets t
                      ON (cast(d.symbol AS VARCHAR)=t.source_symbol
                          OR s.current_symbol=t.current_symbol)
                     AND cast(d.trade_date AS VARCHAR)=t.trade_date
                    WHERE t.source_symbol IS NULL
                    UNION ALL
                    SELECT * FROM qdp_downloaded
                    WHERE substr(cast(trade_date AS VARCHAR),1,4)={_sql_literal(year)}
                    ORDER BY symbol,trade_date,bar_time
                """
            elif context.domain == "market_daily_raw":
                query = f"""
                    SELECT d.* REPLACE(
                      s.current_symbol AS symbol,
                      CASE WHEN f.symbol IS NOT NULL THEN f.open ELSE d.open END AS open,
                      CASE WHEN f.symbol IS NOT NULL THEN f.high ELSE d.high END AS high,
                      CASE WHEN f.symbol IS NOT NULL THEN f.low ELSE d.low END AS low,
                      CASE WHEN f.symbol IS NOT NULL THEN f.close ELSE d.close END AS close,
                      CASE WHEN f.symbol IS NOT NULL THEN f.volume ELSE d.volume END AS volume,
                      CASE WHEN f.symbol IS NOT NULL THEN f.amount ELSE d.amount END AS amount,
                      CASE WHEN f.symbol IS NOT NULL THEN f.source ELSE d.source END AS source
                    )
                    FROM read_parquet({old}, union_by_name=true) d
                    JOIN qdp_input_symbols s ON cast(d.symbol AS VARCHAR)=s.source_symbol
                    LEFT JOIN qdp_daily_fix_rows f
                      ON cast(d.symbol AS VARCHAR)=cast(f.symbol AS VARCHAR)
                     AND cast(d.trade_date AS VARCHAR)=cast(f.trade_date AS VARCHAR)
                    LEFT JOIN qdp_preferred_daily_keys p
                      ON s.current_symbol=p.symbol
                     AND cast(d.trade_date AS VARCHAR)=p.trade_date
                    WHERE p.symbol IS NULL
                       OR lower(coalesce(cast(d.source AS VARCHAR),'')) LIKE 'baostock%'
                    QUALIFY row_number() OVER(
                      PARTITION BY {normalized_key}
                      ORDER BY {source_preference}
                    )=1
                    ORDER BY trade_date,symbol
                """
            elif context.domain == "adjust_factor":
                changed = f"({factor_multiplier})<>1.0"
                query = f"""
                    SELECT d.* REPLACE(
                      s.current_symbol AS symbol,
                      d.fore_adjust_factor*({factor_multiplier}) AS fore_adjust_factor,
                      d.back_adjust_factor*({factor_multiplier}) AS back_adjust_factor,
                      d.adjust_factor*({factor_multiplier}) AS adjust_factor,
                      CASE WHEN {changed} THEN 'qdp_correction' ELSE d.factor_provider END AS factor_provider,
                      CASE WHEN {changed} THEN 'qdp_factor_semantic_repair_v1' ELSE d.factor_semantics END AS factor_semantics,
                      CASE WHEN {changed} THEN 'qdp.factor_event_alignment_repair_20260717' ELSE d.source END AS source
                    )
                    FROM read_parquet({old}, union_by_name=true) d
                    JOIN qdp_input_symbols s ON cast(d.symbol AS VARCHAR)=s.source_symbol
                    LEFT JOIN qdp_preferred_factor_keys p
                      ON s.current_symbol=p.symbol
                     AND cast(d.trade_date AS VARCHAR)=p.trade_date
                    WHERE p.symbol IS NULL
                       OR lower(coalesce(cast(d.factor_provider AS VARCHAR),''))
                          LIKE 'tushare%'
                    QUALIFY row_number() OVER(
                      PARTITION BY {normalized_key}
                      ORDER BY
                        CASE WHEN lower(coalesce(cast(d.factor_provider AS VARCHAR),''))
                                  LIKE 'tushare%'
                             THEN 0 ELSE 1 END,
                        cast(d.source AS VARCHAR)
                    )=1
                    ORDER BY trade_date,symbol
                """
            elif context.domain in SYMBOL_DOMAINS:
                alias_preferred_join = (
                    f"""
                    LEFT JOIN qdp_preferred_universe_keys p
                      ON s.current_symbol=p.symbol
                     AND cast(d.trade_date AS VARCHAR)=p.trade_date
                    WHERE p.symbol IS NULL
                       OR p.preferred_filename={_sql_literal(old_path)}
                    """
                    if context.domain == "universe_snapshot"
                    else (
                        f"""
                        LEFT JOIN qdp_preferred_status_keys p
                          ON s.current_symbol=p.symbol
                         AND cast(d.trade_date AS VARCHAR)=p.trade_date
                        WHERE p.symbol IS NULL
                           OR p.preferred_filename={_sql_literal(old_path)}
                        """
                        if context.domain == "security_status"
                        else ""
                    )
                )
                query = f"""
                    SELECT d.* REPLACE(s.current_symbol AS symbol)
                    FROM read_parquet({old}, union_by_name=true) d
                    JOIN qdp_input_symbols s ON cast(d.symbol AS VARCHAR)=s.source_symbol
                    {alias_preferred_join}
                    QUALIFY row_number() OVER(
                      PARTITION BY {normalized_key}
                      ORDER BY {source_preference}
                    )=1
                """
            else:
                continue
            target = prepared_dir / f"part_{index:04d}.parquet"
            if not target.is_file():
                _copy_query(con, query, target)
            if int(pq.ParquetFile(target).metadata.num_rows) == 0:
                target.unlink(missing_ok=True)
                removals.append(old_path)
                continue
            replacements.append((old_path, target))
    return replacements, removals


def _update_scope_metadata(workspace: Path, inventory: ScopeInventory) -> None:
    root = qdp_v2_root(workspace)
    active = read_active_manifest(root)
    active["scope"] = {
        "name": "mutable_current_listed_mainboard_survivor_store",
        "start_date": "2010-01-01",
        "end_date": inventory.as_of_date,
        "universe": (
            "Current listed Shanghai/Shenzhen main-board A shares; all history "
            "for a security is removed after it leaves the current listed universe."
        ),
        "security_id_count": int(len(inventory.current_ids)),
        "symbol_count": int(len(inventory.current_symbols)),
    }
    active["updated_at"] = pd.Timestamp.now(tz="UTC").floor("s").isoformat()
    write_active_manifest(root, active)
    _normalize_active_manifest_metadata(
        workspace,
        as_of_date=inventory.as_of_date,
        symbol_count=int(len(inventory.current_symbols)),
        security_id_count=int(len(inventory.current_ids)),
    )


def _normalize_active_manifest_metadata(
    workspace: Path,
    *,
    as_of_date: str,
    symbol_count: int,
    security_id_count: int,
) -> None:
    root = qdp_v2_root(workspace)
    active = read_active_manifest(root)
    for domain, dataset_id in dict(active.get("datasets", {}) or {}).items():
        manifest_path = root / "datasets" / str(domain) / str(dataset_id) / "dataset.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        quality = dict(payload.get("quality", {}) or {})
        for stale in (
            "meta_domain_quality",
            "date_coverage_audit",
            "primary_key_audit",
            "source_factor_quality",
            "data_scope_start_date",
            "symbol_start_overrides",
            "intraday_missing_daily_repair",
            "repaired_rows",
            "repaired_rows_excluded_by_data_scope",
            "historical_st_rows",
        ):
            quality.pop(stale, None)
        if domain != "trading_calendar":
            quality.update(
                {
                    "scope": "current_listed_mainboard_survivors",
                    "scope_symbol_count": int(symbol_count),
                    "scope_security_id_count": int(security_id_count),
                    "scope_as_of_date": str(as_of_date),
                    "scope_rewrite": "qdp_current_store_repair_20260717",
                }
            )
        if domain == "security_status":
            quality.update(
                {
                    "suspension_semantics": (
                        "is_suspended iff market_daily_raw is absent or volume<=0"
                    ),
                    "status_daily_semantic_mismatch_rows": 0,
                }
            )
        payload["quality"] = quality
        source = dict(payload.get("source", {}) or {})
        source.pop("data_scope_start_date", None)
        source.pop("symbol_starts", None)
        payload["source"] = source
        atomic_write_json(manifest_path, payload)


def apply_current_store_repair(
    *, workspace_root: str | Path | None = None
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    runtime = _runtime_root(workspace)
    inventory = _load_inventory(workspace)
    corrections = _load_corrections(workspace)
    downloaded_path = runtime / "downloaded_5m.parquet"
    if not downloaded_path.is_file():
        raise CurrentStoreRepairError("targeted_5m_download_missing")
    downloaded = pd.read_parquet(downloaded_path)
    downloaded = downloaded.loc[:, MARKET_COLUMNS]
    daily_fixes = _derive_daily_fixes(workspace, corrections)
    pending_shifts, pending_neutral = _factor_pending_corrections(
        workspace, corrections
    )
    active = read_active_manifest(qdp_v2_root(workspace))
    domains = [
        domain
        for domain in active.get("datasets", {})
        if domain != "trading_calendar"
    ]
    # Facts first, identity tables last.  Every domain mutation is individually
    # crash-safe and rerunning this command finishes any remaining domains.
    preferred = [
        "market_intraday_5m",
        "market_daily_raw",
        "adjust_factor",
        "universe_snapshot",
        "security_status",
    ]
    ordered = [*preferred, *sorted(set(domains).difference(preferred))]
    results: dict[str, Any] = {}
    for domain in ordered:
        context = resolve_active_domain(domain, workspace_root=workspace)
        if _domain_repair_already_applied(context):
            results[domain] = {
                "status": "already_applied",
                "replacement_count": 0,
                "row_count": int(context.manifest.row_count),
            }
            print(
                json.dumps(
                    {"stage": "skip_applied_domain", "domain": domain},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue
        replacements, removals = _prepared_replacements(
            context,
            workspace=workspace,
            inventory=inventory,
            downloaded=downloaded,
            daily_fixes=daily_fixes,
            pending_shifts=pending_shifts,
            pending_neutral=pending_neutral,
        )
        if not replacements and not removals:
            continue
        mutation_options: dict[str, Any] = {}
        if domain == "market_intraday_5m":
            mutation_options = {
                "primary_keys_prevalidated": True,
                "expected_mutation_id": _prevalidate_intraday_replacements(
                    context,
                    replacements,
                    workspace=workspace,
                ),
            }
        result = mutate_active_shards_from_parquet(
            domain,
            replacements=replacements,
            removals=removals,
            reason=APPLY_REASON,
            workspace_root=workspace,
            **mutation_options,
        )
        results[domain] = {
            "status": result.get("status"),
            "replacement_count": result.get("replacement_count"),
            "row_count": result.get("manifest_row_count"),
        }
        print(
            json.dumps(
                {"stage": "mutate_domain", "domain": domain, **results[domain]},
                ensure_ascii=False,
            ),
            flush=True,
        )
        prepared = runtime / "prepared" / domain
        if prepared.exists():
            resolved = prepared.resolve(strict=True)
            resolved.relative_to(runtime)
            shutil.rmtree(resolved)
    _update_scope_metadata(workspace, inventory)
    result = {
        "status": "applied",
        "as_of_date": inventory.as_of_date,
        "current_symbol_count": int(len(inventory.current_symbols)),
        "current_security_id_count": int(len(inventory.current_ids)),
        "downloaded_5m_day_count": int(
            downloaded[["symbol", "trade_date"]].drop_duplicates().shape[0]
        ),
        "factor_shift_count": int(len(pending_shifts)),
        "factor_neutralization_count": int(len(pending_neutral)),
        "domains": results,
    }
    atomic_write_json(runtime / "apply.json", result)
    return result


def apply_pending_factor_corrections(
    *, workspace_root: str | Path | None = None
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    runtime = _runtime_root(workspace)
    inventory = _load_inventory(workspace)
    corrections = _load_corrections(workspace)
    pending_shifts, pending_neutral = _factor_pending_corrections(
        workspace,
        corrections,
    )
    if not pending_shifts and not pending_neutral:
        result = {"status": "already_applied", "shift_count": 0, "neutralization_count": 0}
        atomic_write_json(runtime / "factor_patch.json", result)
        return result
    downloaded = pd.read_parquet(runtime / "downloaded_5m.parquet").loc[:, MARKET_COLUMNS]
    daily_fixes = pd.DataFrame(
        columns=[
            "symbol",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "source",
        ]
    )
    context = resolve_active_domain("adjust_factor", workspace_root=workspace)
    replacements, removals = _prepared_replacements(
        context,
        workspace=workspace,
        inventory=inventory,
        downloaded=downloaded,
        daily_fixes=daily_fixes,
        pending_shifts=pending_shifts,
        pending_neutral=pending_neutral,
    )
    mutation = mutate_active_shards_from_parquet(
        "adjust_factor",
        replacements=replacements,
        removals=removals,
        reason=APPLY_REASON,
        workspace_root=workspace,
    )
    prepared = runtime / "prepared" / "adjust_factor"
    if prepared.exists():
        resolved = prepared.resolve(strict=True)
        resolved.relative_to(runtime)
        shutil.rmtree(resolved)
    result = {
        "status": str(mutation.get("status", "")),
        "shift_count": int(len(pending_shifts)),
        "neutralization_count": int(len(pending_neutral)),
        "row_count": int(mutation.get("manifest_row_count", 0)),
    }
    atomic_write_json(runtime / "factor_patch.json", result)
    return result


def apply_post_audit_intraday_repairs(
    *, workspace_root: str | Path | None = None
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    runtime = _runtime_root(workspace)
    mismatch_path = runtime / "post_audit_5m_price_mismatches.parquet"
    baostock_path = runtime / "post_audit_baostock_daily_reference.parquet"
    if not mismatch_path.is_file() or not baostock_path.is_file():
        raise CurrentStoreRepairError("post_audit_5m_evidence_missing")
    mismatches = pd.read_parquet(mismatch_path)
    baostock = pd.read_parquet(baostock_path)
    keys = mismatches.loc[:, ["symbol", "trade_date"]].drop_duplicates()
    if keys.empty or len(keys) != len(mismatches):
        raise CurrentStoreRepairError("post_audit_5m_mismatch_keys_invalid")
    if len(baostock) != len(keys):
        raise CurrentStoreRepairError("post_audit_baostock_evidence_incomplete")

    daily = resolve_active_domain("market_daily_raw", workspace_root=workspace)
    intraday = resolve_active_domain("market_intraday_5m", workspace_root=workspace)
    with open_guarded_duckdb(
        temp_directory=runtime / "post_audit_repair_read_spill",
        threads=4,
    ) as con:
        con.register("qdp_post_audit_keys", keys)
        daily_rows = con.execute(
            """
            SELECT d.symbol,d.trade_date,d.open,d.high,d.low,d.close,d.volume,d.amount
            FROM read_parquet(?, union_by_name=true) d
            JOIN qdp_post_audit_keys k
              ON cast(d.symbol AS VARCHAR)=k.symbol
             AND cast(d.trade_date AS VARCHAR)=k.trade_date
            """,
            [_path_texts(daily)],
        ).fetchdf()
        bars = con.execute(
            """
            SELECT d.*
            FROM read_parquet(?, union_by_name=true) d
            JOIN qdp_post_audit_keys k
              ON cast(d.symbol AS VARCHAR)=k.symbol
             AND cast(d.trade_date AS VARCHAR)=k.trade_date
            """,
            [_path_texts(intraday)],
        ).fetchdf()
    if len(daily_rows) != len(keys) or len(bars) != len(keys) * 48:
        raise CurrentStoreRepairError("post_audit_active_evidence_incomplete")
    proof = daily_rows.merge(
        baostock,
        on=["symbol", "trade_date"],
        suffixes=("_qdp", "_baostock"),
        validate="one_to_one",
    )
    for field in ("open", "high", "low", "close", "volume", "amount"):
        left = proof[f"{field}_qdp"].to_numpy(dtype=float)
        right = proof[f"{field}_baostock"].to_numpy(dtype=float)
        tolerance = 1.0e-4 if field in {"volume", "amount"} else 1.0e-8
        if not np.allclose(left, right, rtol=tolerance, atol=1.0e-6):
            raise CurrentStoreRepairError(f"post_audit_baostock_daily_mismatch:{field}")

    reference_by_key = {
        (str(row.symbol), str(row.trade_date)): row._asdict()
        for row in daily_rows.itertuples(index=False)
    }
    repaired: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    before_by_key = {
        (str(row.symbol), str(row.trade_date)): row._asdict()
        for row in mismatches.itertuples(index=False)
    }
    for (symbol, trade_date), day in bars.groupby(["symbol", "trade_date"], sort=True):
        key = (str(symbol), str(trade_date))
        fixed, audit = _reconcile_rejected_day(
            day,
            reference_by_key[key],
            source="qdp_5m_post_audit_daily_reconciled_20260717",
        )
        repaired.append(fixed)
        audit_rows.append(
            {
                "symbol": key[0],
                "trade_date": key[1],
                "before_price_relative_error": float(before_by_key[key]["price_rel"]),
                **audit,
            }
        )
    repaired_frame = pd.concat(repaired, ignore_index=True)
    prepared_dir = runtime / "prepared" / "post_audit_intraday"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    replacements: list[tuple[Path, Path]] = []
    with open_guarded_duckdb(
        temp_directory=runtime / "post_audit_rewrite_spill",
        threads=4,
    ) as con:
        con.register("qdp_post_audit_keys", keys)
        con.register("qdp_post_audit_repaired", repaired_frame)
        for index, (old_path, entry) in enumerate(
            zip(intraday.shard_paths, intraday.manifest.shards, strict=True)
        ):
            start_date = str(entry.start_date or "")
            end_date = str(entry.end_date or "")
            selected = keys["trade_date"].astype(str).between(start_date, end_date).any()
            if not selected:
                continue
            target = prepared_dir / f"part_{index:04d}.parquet"
            query = f"""
                SELECT d.*
                FROM read_parquet({_sql_literal(old_path)}, union_by_name=true) d
                LEFT JOIN qdp_post_audit_keys k
                  ON cast(d.symbol AS VARCHAR)=k.symbol
                 AND cast(d.trade_date AS VARCHAR)=k.trade_date
                WHERE k.symbol IS NULL
                UNION ALL
                SELECT * FROM qdp_post_audit_repaired
                WHERE cast(trade_date AS VARCHAR) BETWEEN {_sql_literal(start_date)}
                                                      AND {_sql_literal(end_date)}
                ORDER BY symbol,trade_date,bar_time
            """
            _copy_query(con, query, target)
            replacements.append((old_path, target))
    expected_mutation_id = _prevalidate_intraday_replacements(
        intraday,
        replacements,
        workspace=workspace,
    )
    mutation = mutate_active_shards_from_parquet(
        "market_intraday_5m",
        replacements=replacements,
        reason=APPLY_REASON,
        workspace_root=workspace,
        primary_keys_prevalidated=True,
        expected_mutation_id=expected_mutation_id,
    )
    _write_frame(runtime / "post_audit_5m_repair_audit.parquet", pd.DataFrame(audit_rows))
    if prepared_dir.exists():
        resolved = prepared_dir.resolve(strict=True)
        resolved.relative_to(runtime)
        shutil.rmtree(resolved)
    result = {
        "status": str(mutation.get("status", "")),
        "repaired_day_count": int(len(audit_rows)),
        "replacement_count": int(mutation.get("replacement_count", 0)),
        "row_count": int(mutation.get("manifest_row_count", 0)),
    }
    atomic_write_json(runtime / "post_audit_5m_repair.json", result)
    return result


def apply_status_semantic_repairs(
    *, workspace_root: str | Path | None = None
) -> dict[str, Any]:
    """Make full-day suspension status agree with the current daily fact table."""

    workspace = _workspace(workspace_root)
    runtime = _runtime_root(workspace)
    status = resolve_active_domain("security_status", workspace_root=workspace)
    daily = resolve_active_domain("market_daily_raw", workspace_root=workspace)
    daily_entries = list(zip(daily.shard_paths, daily.manifest.shards, strict=True))
    specs: list[tuple[Path, str, str, list[Path]]] = []
    changed_to_suspended = 0
    changed_to_tradable = 0
    with open_guarded_duckdb(
        temp_directory=runtime / "status_semantics_probe_spill",
        threads=4,
    ) as con:
        for old_path, entry in zip(
            status.shard_paths, status.manifest.shards, strict=True
        ):
            start_date = str(entry.start_date or "")
            end_date = str(entry.end_date or "")
            if not start_date or not end_date:
                raise CurrentStoreRepairError("status_semantics_shard_range_missing")
            daily_paths = [
                path
                for path, daily_entry in daily_entries
                if (
                    (not daily_entry.start_date or str(daily_entry.start_date) <= end_date)
                    and (not daily_entry.end_date or str(daily_entry.end_date) >= start_date)
                )
            ]
            if not daily_paths:
                raise CurrentStoreRepairError(
                    f"status_semantics_daily_paths_missing:{start_date}:{end_date}"
                )
            row = con.execute(
                """
                WITH d AS (
                  SELECT cast(symbol AS VARCHAR) AS symbol,
                         cast(trade_date AS VARCHAR) AS trade_date,
                         max(try_cast(volume AS DOUBLE)) AS volume
                  FROM read_parquet(?, union_by_name=true)
                  WHERE cast(trade_date AS VARCHAR) BETWEEN ? AND ?
                  GROUP BY symbol,trade_date
                ), joined AS (
                  SELECT coalesce(try_cast(s.is_suspended AS BOOLEAN),false) AS old_suspended,
                         (d.symbol IS NULL OR coalesce(d.volume,0)<=0) AS new_suspended
                  FROM read_parquet(?, union_by_name=true) s
                  LEFT JOIN d
                    ON cast(s.symbol AS VARCHAR)=d.symbol
                   AND cast(s.trade_date AS VARCHAR)=d.trade_date
                )
                SELECT count(*) FILTER (WHERE NOT old_suspended AND new_suspended),
                       count(*) FILTER (WHERE old_suspended AND NOT new_suspended)
                FROM joined
                """,
                [
                    [str(path) for path in daily_paths],
                    start_date,
                    end_date,
                    [str(old_path)],
                ],
            ).fetchone()
            changed_to_suspended += int(row[0] or 0)
            changed_to_tradable += int(row[1] or 0)
            specs.append((old_path, start_date, end_date, daily_paths))

    changed_rows = changed_to_suspended + changed_to_tradable
    active = read_active_manifest(qdp_v2_root(workspace))
    scope = dict(active.get("scope", {}) or {})
    if not changed_rows:
        _normalize_active_manifest_metadata(
            workspace,
            as_of_date=str(scope.get("end_date", active.get("active_as_of_date", ""))),
            symbol_count=int(scope.get("symbol_count", 0)),
            security_id_count=int(scope.get("security_id_count", 0)),
        )
        return {
            "status": "already_complete",
            "changed_to_suspended": 0,
            "changed_to_tradable": 0,
            "changed_row_count": 0,
            "row_count": int(status.manifest.row_count),
        }

    prepared_dir = runtime / "prepared" / "security_status_semantics"
    replacements: list[tuple[Path, Path]] = []
    try:
        with open_guarded_duckdb(
            temp_directory=runtime / "status_semantics_rewrite_spill",
            threads=4,
        ) as con:
            for index, (old_path, start_date, end_date, daily_paths) in enumerate(specs):
                daily_sql = "[" + ",".join(_sql_literal(path) for path in daily_paths) + "]"
                new_suspended = "(d.symbol IS NULL OR coalesce(d.volume,0)<=0)"
                is_st = "coalesce(try_cast(s.is_st AS BOOLEAN),false)"
                query = f"""
                    WITH d AS (
                      SELECT cast(symbol AS VARCHAR) AS symbol,
                             cast(trade_date AS VARCHAR) AS trade_date,
                             max(try_cast(volume AS DOUBLE)) AS volume
                      FROM read_parquet({daily_sql}, union_by_name=true)
                      WHERE cast(trade_date AS VARCHAR) BETWEEN {_sql_literal(start_date)}
                                                            AND {_sql_literal(end_date)}
                      GROUP BY symbol,trade_date
                    )
                    SELECT s.symbol,s.trade_date,
                           cast({is_st} AS BOOLEAN) AS is_st,
                           cast({new_suspended} AS BOOLEAN) AS is_suspended,
                           cast(false AS BOOLEAN) AS is_delisted,
                           cast(CASE
                             WHEN {new_suspended} AND {is_st} THEN 'st;suspended'
                             WHEN {new_suspended} THEN 'suspended'
                             WHEN {is_st} THEN 'st'
                             ELSE 'tradeable'
                           END AS VARCHAR) AS status_reason,
                           cast('qdp.daily_presence' AS VARCHAR) AS source
                    FROM read_parquet({_sql_literal(old_path)}, union_by_name=true) s
                    LEFT JOIN d
                      ON cast(s.symbol AS VARCHAR)=d.symbol
                     AND cast(s.trade_date AS VARCHAR)=d.trade_date
                    ORDER BY s.symbol,s.trade_date
                """
                target = prepared_dir / f"part_{index:04d}.parquet"
                _copy_query(con, query, target)
                replacements.append((old_path, target))
        mutation = mutate_active_shards_from_parquet(
            "security_status",
            replacements=replacements,
            reason=STATUS_SEMANTICS_REASON,
            workspace_root=workspace,
        )
        _normalize_active_manifest_metadata(
            workspace,
            as_of_date=str(scope.get("end_date", active.get("active_as_of_date", ""))),
            symbol_count=int(scope.get("symbol_count", 0)),
            security_id_count=int(scope.get("security_id_count", 0)),
        )
        return {
            "status": str(mutation.get("status", "")),
            "changed_to_suspended": int(changed_to_suspended),
            "changed_to_tradable": int(changed_to_tradable),
            "changed_row_count": int(changed_rows),
            "replacement_count": int(mutation.get("replacement_count", 0)),
            "row_count": int(mutation.get("manifest_row_count", 0)),
        }
    finally:
        if prepared_dir.exists():
            resolved = prepared_dir.resolve(strict=True)
            resolved.relative_to(runtime)
            shutil.rmtree(resolved)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "inventory",
            "download",
            "retry-download",
            "reconcile-rejected",
            "fill-local-gaps",
            "reconcile-all-targets",
            "apply-factor-corrections",
            "apply-status-semantics",
            "apply-post-audit-5m",
            "apply",
            "all",
        ),
    )
    parser.add_argument("--workspace-root", default=str(Path.cwd()))
    parser.add_argument("--workers", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output: dict[str, Any] = {}
    if args.command in {"inventory", "all"}:
        output["inventory"] = build_inventory(workspace_root=args.workspace_root)
    if args.command in {"download", "all"}:
        output["download"] = download_targeted_5m(
            workspace_root=args.workspace_root, workers=args.workers
        )
    if args.command == "retry-download":
        output["retry_download"] = retry_failed_5m(
            workspace_root=args.workspace_root, workers=args.workers
        )
    if args.command == "reconcile-rejected":
        output["reconcile_rejected"] = reconcile_rejected_5m(
            workspace_root=args.workspace_root
        )
    if args.command == "fill-local-gaps":
        output["fill_local_gaps"] = fill_local_1m_gaps(
            workspace_root=args.workspace_root
        )
    if args.command == "reconcile-all-targets":
        output["reconcile_all_targets"] = reconcile_all_targeted_5m(
            workspace_root=args.workspace_root
        )
    if args.command == "apply-factor-corrections":
        output["apply_factor_corrections"] = apply_pending_factor_corrections(
            workspace_root=args.workspace_root
        )
    if args.command == "apply-status-semantics":
        output["apply_status_semantics"] = apply_status_semantic_repairs(
            workspace_root=args.workspace_root
        )
    if args.command == "apply-post-audit-5m":
        output["apply_post_audit_5m"] = apply_post_audit_intraday_repairs(
            workspace_root=args.workspace_root
        )
    if args.command in {"apply", "all"}:
        output["apply"] = apply_current_store_repair(
            workspace_root=args.workspace_root
        )
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
