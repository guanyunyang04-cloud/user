from __future__ import annotations

"""Lean in-place repair for recent daily and 5-minute market data.

The downloader writes only final Parquet bundles.  There is no raw-file fanout,
SQLite payload store, ZIP conversion, candidate generation, or active-pointer
rewrite.  Work is resumable at a stable security bucket, and the existing v2
repair API remains the single manifest commit point.
"""

import argparse
import hashlib
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.domains.contracts import DataDomain, DomainFetchRequest
from quant_data_platform.providers import MootdxOnlineProvider
from quant_data_platform.qdp_v2.duckdb_resources import open_guarded_duckdb
from quant_data_platform.qdp_v2.manifest import (
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    stable_hash,
)
from quant_data_platform.qdp_v2.repair import bulk_append_active_shards_from_parquet


REPAIR_VERSION = 1
BUCKET_COUNT = 16
DEFAULT_WORKERS = 3
DEFAULT_START_DATE = "2026-06-29"
DEFAULT_END_DATE = "2026-07-13"
MEMORY_FLOOR_BYTES = int(0.5 * 1024**3)
LOW_MEMORY_SECONDS = 5.0
POLL_SECONDS = 0.25

DAILY_DOMAIN = "market_daily_raw"
INTRADAY_DOMAIN = "market_intraday_5m"
DAILY_COLUMNS = (
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "source",
    "adjusted_flag",
)
INTRADAY_COLUMNS = (
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
NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume", "amount")
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
assert len(EXPECTED_BAR_TIMES) == 48

DAILY_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string()),
        pa.field("trade_date", pa.string()),
        *[pa.field(item, pa.float64()) for item in NUMERIC_COLUMNS],
        pa.field("source", pa.string()),
        pa.field("adjusted_flag", pa.string()),
    ]
)
INTRADAY_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string()),
        pa.field("trade_date", pa.string()),
        pa.field("bar_time", pa.string()),
        *[pa.field(item, pa.float64()) for item in NUMERIC_COLUMNS],
        pa.field("source", pa.string()),
        pa.field("adjusted_flag", pa.string()),
    ]
)


class RecentMarketRepairError(RuntimeError):
    pass


class RecentMarketMemoryError(RecentMarketRepairError):
    pass


class _SystemMemoryGuard:
    """Continuously observe the user-selected 0.5 GiB / five-second floor."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._triggered = threading.Event()
        self._thread: threading.Thread | None = None
        self._low_since: float | None = None
        self.minimum_available_bytes: int | None = None

    def __enter__(self) -> "_SystemMemoryGuard":
        self._thread = threading.Thread(
            target=self._run,
            name="qdp-recent-repair-memory-guard",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def check(self, label: str) -> None:
        if self._triggered.is_set():
            raise RecentMarketMemoryError(
                f"available_memory_below_0.5_gib_for_5_seconds:{label}"
            )

    def _run(self) -> None:
        try:
            import psutil  # type: ignore
        except ImportError:
            self._triggered.set()
            return
        while not self._stop.wait(POLL_SECONDS):
            available = int(psutil.virtual_memory().available)
            if self.minimum_available_bytes is None or available < self.minimum_available_bytes:
                self.minimum_available_bytes = available
            if available >= MEMORY_FLOOR_BYTES:
                self._low_since = None
                continue
            now = time.monotonic()
            if self._low_since is None:
                self._low_since = now
            elif now - self._low_since >= LOW_MEMORY_SECONDS:
                self._triggered.set()
                return


@dataclass(frozen=True)
class _DomainInput:
    domain: str
    dataset_id: str
    manifest_path: Path
    manifest_sha256: str
    paths: tuple[Path, ...]


def run_recent_daily_repair(
    *,
    trade_date: str = DEFAULT_END_DATE,
    workspace_root: str | Path | None = None,
    workers: int = DEFAULT_WORKERS,
    resume: bool = True,
) -> dict[str, Any]:
    date_text = _date_text(trade_date)
    workspace = Path(workspace_root or Path.cwd()).resolve()
    _configure_workspace_runtime(workspace)
    inputs = _active_inputs(
        workspace,
        (DAILY_DOMAIN, "universe_snapshot", "security_status"),
    )
    active_sha = _sha256_file(qdp_v2_root(workspace) / "active" / "active.json")
    symbols = _missing_daily_symbols(inputs, trade_date=date_text, workspace=workspace)
    fingerprint = stable_hash(
        {
            "version": REPAIR_VERSION,
            "kind": "daily",
            "trade_date": date_text,
            "symbols": symbols,
            "active_sha256": active_sha,
            "manifest_sha256": {
                key: value.manifest_sha256 for key, value in inputs.items()
            },
        },
        length=32,
    )
    runtime = _runtime_root(workspace) / f"mootdx_daily__{fingerprint[:24]}"
    state = _load_or_initialize_state(
        runtime,
        fingerprint=fingerprint,
        kind="daily",
        start_date=date_text,
        end_date=date_text,
        task_count=len(symbols),
        resume=resume,
    )
    if state.get("status") == "applied":
        return state
    wanted = {symbol: (date_text,) for symbol in symbols}
    with _SystemMemoryGuard() as guard:
        state = _download_buckets(
            runtime=runtime,
            state=state,
            wanted=wanted,
            provider_domain=DataDomain.MARKET_DAILY,
            output_domain=DAILY_DOMAIN,
            start_date=date_text,
            end_date=date_text,
            workers=workers,
            guard=guard,
        )
        guard.check("daily_before_commit")
        state = _commit_bundles(
            state,
            runtime=runtime,
            domain=DAILY_DOMAIN,
            reason=f"fill missing {date_text} daily rows from healthy mootdx",
            workspace=workspace,
        )
        state["minimum_available_bytes"] = guard.minimum_available_bytes
        atomic_write_json(runtime / "state.json", state)
    return state


def run_recent_intraday_repair(
    *,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    workspace_root: str | Path | None = None,
    workers: int = DEFAULT_WORKERS,
    resume: bool = True,
) -> dict[str, Any]:
    start = _date_text(start_date)
    end = _date_text(end_date)
    if start > end:
        raise ValueError(f"recent_repair_invalid_range:{start}>{end}")
    workspace = Path(workspace_root or Path.cwd()).resolve()
    _configure_workspace_runtime(workspace)
    domains = (
        DAILY_DOMAIN,
        INTRADAY_DOMAIN,
        "universe_snapshot",
        "security_status",
    )
    inputs = _active_inputs(workspace, domains, start_date=start, end_date=end)
    active_sha = _sha256_file(qdp_v2_root(workspace) / "active" / "active.json")
    pairs = _missing_intraday_pairs(
        inputs,
        start_date=start,
        end_date=end,
        workspace=workspace,
    )
    wanted: dict[str, tuple[str, ...]] = {}
    for symbol, trade_date in pairs:
        wanted.setdefault(symbol, tuple())
        wanted[symbol] = (*wanted[symbol], trade_date)
    fingerprint = stable_hash(
        {
            "version": REPAIR_VERSION,
            "kind": "intraday_5m",
            "start_date": start,
            "end_date": end,
            "pairs": pairs,
            "active_sha256": active_sha,
            "manifest_sha256": {
                key: value.manifest_sha256 for key, value in inputs.items()
            },
        },
        length=32,
    )
    runtime = _runtime_root(workspace) / f"mootdx_5m__{fingerprint[:24]}"
    state = _load_or_initialize_state(
        runtime,
        fingerprint=fingerprint,
        kind="intraday_5m",
        start_date=start,
        end_date=end,
        task_count=len(pairs),
        resume=resume,
    )
    if state.get("status") == "applied":
        return state
    with _SystemMemoryGuard() as guard:
        state = _download_buckets(
            runtime=runtime,
            state=state,
            wanted=wanted,
            provider_domain=DataDomain.MARKET_INTRADAY_5M,
            output_domain=INTRADAY_DOMAIN,
            start_date=start,
            end_date=end,
            workers=workers,
            guard=guard,
        )
        guard.check("intraday_before_commit")
        state = _commit_bundles(
            state,
            runtime=runtime,
            domain=INTRADAY_DOMAIN,
            reason=f"fill complete recent 5m stock-days from healthy mootdx {start}..{end}",
            workspace=workspace,
        )
        state["minimum_available_bytes"] = guard.minimum_available_bytes
        atomic_write_json(runtime / "state.json", state)
    return state


def run_tushare_intraday_fallback(
    *,
    mootdx_state_path: str | Path,
    workspace_root: str | Path | None = None,
    workers: int = DEFAULT_WORKERS,
    resume: bool = True,
) -> dict[str, Any]:
    """Fill only mootdx-unresolved stock-days, primarily delisted securities."""

    source_state_path = Path(mootdx_state_path).resolve()
    source_state = json.loads(source_state_path.read_text(encoding="utf-8"))
    unresolved_pairs = _unresolved_pairs_from_state(source_state)
    if not unresolved_pairs:
        return {"status": "nothing_to_fill", "task_count": 0}
    workspace = Path(workspace_root or Path.cwd()).resolve()
    _configure_workspace_runtime(workspace)
    active = _active_inputs(
        workspace,
        (INTRADAY_DOMAIN,),
        start_date=min(item[1] for item in unresolved_pairs),
        end_date=max(item[1] for item in unresolved_pairs),
    )[INTRADAY_DOMAIN]
    fingerprint = stable_hash(
        {
            "version": REPAIR_VERSION,
            "kind": "tushare_intraday_fallback",
            "source_state_sha256": _sha256_file(source_state_path),
            "pairs": unresolved_pairs,
            "active_manifest_sha256": active.manifest_sha256,
        },
        length=32,
    )
    runtime = _runtime_root(workspace) / f"tushare_5m_fallback__{fingerprint[:24]}"
    state = _load_or_initialize_state(
        runtime,
        fingerprint=fingerprint,
        kind="tushare_intraday_fallback",
        start_date=min(item[1] for item in unresolved_pairs),
        end_date=max(item[1] for item in unresolved_pairs),
        task_count=len(unresolved_pairs),
        resume=resume,
    )
    if state.get("status") == "applied":
        return state
    bundle_path = runtime / "bundles" / "market_intraday_5m_tushare_fallback.parquet"
    if state.get("status") == "downloaded":
        _validate_reused_bucket(
            {
                "bundle_path": str(bundle_path),
                "bundle_sha256": state.get("bundle_sha256", ""),
                "row_count": state.get("row_count", 0),
            },
            runtime=runtime,
        )
    else:
        from quant_data_platform.qdp_v2.tushare_gap_repair import (
            GapTask,
            _fetch_gap_task,
        )
        from quant_data_platform.tushare_proxy import TushareProxyClient

        wanted = _unresolved_mapping(
            {pair: "mootdx_unresolved" for pair in unresolved_pairs}
        )
        tasks = [
            GapTask(
                symbol=symbol,
                start_date=min(dates),
                end_date=max(dates),
                scope="recent_tail",
            )
            for symbol, dates in sorted(wanted.items())
        ]
        frames: list[pd.DataFrame] = []
        errors: list[dict[str, str]] = []
        client = TushareProxyClient()
        with _SystemMemoryGuard() as guard:
            with ThreadPoolExecutor(max_workers=min(max(1, int(workers)), len(tasks))) as executor:
                futures = {
                    executor.submit(
                        _fetch_gap_task,
                        task,
                        client,
                        guard,
                        "tushare_proxy_5m_direct_gap_repair",
                        200 * 1024**3,
                        workspace,
                    ): task
                    for task in tasks
                }
                for future in as_completed(futures):
                    task = futures[future]
                    guard.check(f"tushare_fallback_result:{task.symbol}")
                    try:
                        capture = future.result()
                    except BaseException as exc:
                        errors.append(
                            {
                                "symbol": task.symbol,
                                "error_type": type(exc).__name__,
                                "message": str(exc)[:500],
                            }
                        )
                        continue
                    frame = pq.read_table(pa.BufferReader(capture.payload)).to_pandas()
                    dates = set(wanted[task.symbol])
                    frame = frame.loc[frame["trade_date"].astype(str).isin(dates)].copy()
                    if not frame.empty:
                        frames.append(frame)
            try:
                client.close_thread_session()
            except AttributeError:
                pass
            combined = (
                pd.concat(frames, ignore_index=True, sort=False)
                if frames
                else pd.DataFrame(columns=INTRADAY_COLUMNS)
            )
            accepted_pairs = {
                (str(symbol), str(trade_date))
                for (symbol, trade_date), day in combined.groupby(
                    ["symbol", "trade_date"], sort=False
                )
                if len(day) == 48 and day["bar_time"].nunique() == 48
            }
            requested_pairs = set(unresolved_pairs)
            combined = combined.loc[
                pd.MultiIndex.from_frame(combined[["symbol", "trade_date"]]).isin(
                    pd.MultiIndex.from_tuples(
                        sorted(accepted_pairs), names=("symbol", "trade_date")
                    )
                )
            ].copy() if accepted_pairs else combined.iloc[:0].copy()
            if not combined.empty:
                _write_final_parquet(combined, bundle_path, domain=INTRADAY_DOMAIN)
            state.update(
                {
                    "status": "downloaded",
                    "row_count": int(len(combined)),
                    "accepted_day_count": len(accepted_pairs),
                    "unresolved": [
                        {"symbol": symbol, "trade_date": trade_date}
                        for symbol, trade_date in sorted(requested_pairs - accepted_pairs)
                    ],
                    "unresolved_day_count": len(requested_pairs - accepted_pairs),
                    "provider_error_count": len(errors),
                    "provider_error_sample": errors[:10],
                    "bundle_path": str(bundle_path) if not combined.empty else "",
                    "bundle_sha256": _sha256_file(bundle_path) if not combined.empty else "",
                    "minimum_available_bytes": guard.minimum_available_bytes,
                    "updated_at": _utc_now(),
                }
            )
            atomic_write_json(runtime / "state.json", state)
    bundle_paths = [bundle_path] if bundle_path.is_file() else []
    if bundle_paths:
        result = bulk_append_active_shards_from_parquet(
            INTRADAY_DOMAIN,
            bundle_paths,
            "fill exact mootdx recent gaps and delisted securities from Tushare",
            workspace_root=workspace,
        )
    else:
        result = {"status": "nothing_to_append", "row_count": 0}
    state.update({"status": "applied", "commit": result, "updated_at": _utc_now()})
    atomic_write_json(runtime / "state.json", state)
    return state


def _download_buckets(
    *,
    runtime: Path,
    state: dict[str, Any],
    wanted: Mapping[str, Sequence[str]],
    provider_domain: str,
    output_domain: str,
    start_date: str,
    end_date: str,
    workers: int,
    guard: _SystemMemoryGuard,
) -> dict[str, Any]:
    bundles = runtime / "bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    completed = dict(state.get("buckets", {}) or {})
    for bucket in range(BUCKET_COUNT):
        key = f"{bucket:02d}"
        guard.check(f"bucket_{key}_start")
        bucket_wanted = {
            symbol: tuple(sorted(set(str(item) for item in dates)))
            for symbol, dates in wanted.items()
            if _stable_bucket(symbol) == bucket
        }
        previous = completed.get(key)
        if isinstance(previous, dict) and previous.get("status") == "completed":
            _validate_reused_bucket(previous, runtime=runtime)
            continue
        frame, unresolved, errors = _fetch_bucket(
            bucket_wanted,
            provider_domain=provider_domain,
            output_domain=output_domain,
            start_date=start_date,
            end_date=end_date,
            workers=workers,
            guard=guard,
        )
        bundle_path: Path | None = None
        if not frame.empty:
            bundle_path = bundles / f"{output_domain}_bucket_{key}.parquet"
            _write_final_parquet(frame, bundle_path, domain=output_domain)
        completed[key] = {
            "status": "completed",
            "symbol_count": len(bucket_wanted),
            "requested_day_count": sum(len(item) for item in bucket_wanted.values()),
            "accepted_day_count": int(
                len(frame) if output_domain == DAILY_DOMAIN else len(frame) // 48
            ),
            "row_count": int(len(frame)),
            "bundle_path": str(bundle_path) if bundle_path else "",
            "bundle_sha256": _sha256_file(bundle_path) if bundle_path else "",
            "unresolved": [
                {"symbol": symbol, "trade_date": trade_date, "reason": reason}
                for (symbol, trade_date), reason in sorted(unresolved.items())
            ],
            "provider_error_count": len(errors),
            "provider_error_sample": errors[:10],
        }
        state.update(
            {
                "status": "downloading",
                "buckets": completed,
                "completed_bucket_count": sum(
                    item.get("status") == "completed" for item in completed.values()
                ),
                "updated_at": _utc_now(),
            }
        )
        atomic_write_json(runtime / "state.json", state)
        print(
            json.dumps(
                {
                    "bucket": key,
                    "symbols": len(bucket_wanted),
                    "rows": len(frame),
                    "unresolved_days": len(unresolved),
                    "provider_errors": len(errors),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    state["status"] = "downloaded"
    state["buckets"] = completed
    state["row_count"] = sum(int(item.get("row_count", 0)) for item in completed.values())
    state["accepted_day_count"] = sum(
        int(item.get("accepted_day_count", 0)) for item in completed.values()
    )
    state["unresolved_day_count"] = sum(
        len(item.get("unresolved", []) or []) for item in completed.values()
    )
    state["updated_at"] = _utc_now()
    atomic_write_json(runtime / "state.json", state)
    return state


def _fetch_bucket(
    wanted: Mapping[str, Sequence[str]],
    *,
    provider_domain: str,
    output_domain: str,
    start_date: str,
    end_date: str,
    workers: int,
    guard: _SystemMemoryGuard,
) -> tuple[pd.DataFrame, dict[tuple[str, str], str], list[dict[str, Any]]]:
    pending = {symbol: tuple(dates) for symbol, dates in wanted.items()}
    accepted: list[pd.DataFrame] = []
    all_errors: list[dict[str, Any]] = []
    last_reasons: dict[tuple[str, str], str] = {
        (symbol, trade_date): "not_requested"
        for symbol, dates in pending.items()
        for trade_date in dates
    }
    for attempt in range(1, 3):
        guard.check(f"provider_round_{attempt}")
        symbols = tuple(sorted(pending))
        if not symbols:
            break
        chunks = _split_symbols(symbols, max(1, int(workers)))
        frames: list[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = {
                executor.submit(
                    _fetch_mootdx_chunk,
                    chunk,
                    domain=provider_domain,
                    start_date=start_date,
                    end_date=end_date,
                ): chunk
                for chunk in chunks
            }
            for future in as_completed(futures):
                guard.check(f"provider_round_{attempt}_result")
                try:
                    frame, errors = future.result()
                except BaseException as exc:
                    all_errors.append(
                        {
                            "code": "chunk_fetch_error",
                            "error_type": type(exc).__name__,
                            "message": str(exc)[:500],
                            "symbol_count": len(futures[future]),
                        }
                    )
                    continue
                if not frame.empty:
                    frames.append(frame)
                all_errors.extend(dict(item) for item in errors)
        downloaded = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        if output_domain == DAILY_DOMAIN:
            valid, unresolved = _validate_daily_frame(downloaded, pending)
        else:
            valid, unresolved = _validate_intraday_frame(downloaded, pending)
        if not valid.empty:
            accepted.append(valid)
        pending = _unresolved_mapping(unresolved)
        last_reasons = unresolved
    if accepted:
        combined = pd.concat(accepted, ignore_index=True, sort=False)
        key_columns = ["symbol", "trade_date"] + (
            ["bar_time"] if output_domain == INTRADAY_DOMAIN else []
        )
        if combined.duplicated(key_columns).any():
            raise RecentMarketRepairError("recent_repair_duplicate_accepted_primary_key")
        combined = combined.sort_values(key_columns, kind="stable").reset_index(drop=True)
    else:
        columns = INTRADAY_COLUMNS if output_domain == INTRADAY_DOMAIN else DAILY_COLUMNS
        combined = pd.DataFrame(columns=columns)
    return combined, last_reasons, all_errors


def _fetch_mootdx_chunk(
    symbols: Sequence[str],
    *,
    domain: str,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    provider = MootdxOnlineProvider(page_size=800, max_pages=3)
    try:
        result = provider.fetch_domain(
            DomainFetchRequest(
                domain=domain,
                symbols=tuple(symbols),
                start_date=start_date,
                end_date=end_date,
                adjusted_flag="none",
            )
        )
        return result.data, list(result.error_report or [])
    finally:
        provider.close()


def _validate_daily_frame(
    frame: pd.DataFrame,
    wanted: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, dict[tuple[str, str], str]]:
    wanted_pairs = {(symbol, str(date)) for symbol, dates in wanted.items() for date in dates}
    if frame is None or frame.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS), {
            item: "missing_provider_day" for item in wanted_pairs
        }
    data = frame.copy()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    data["trade_date"] = data["trade_date"].astype(str).str.slice(0, 10)
    pair_index = pd.MultiIndex.from_frame(data[["symbol", "trade_date"]])
    data = data.loc[pair_index.isin(pd.MultiIndex.from_tuples(sorted(wanted_pairs)))].copy()
    if data.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS), {
            item: "missing_provider_day" for item in wanted_pairs
        }
    numeric, row_valid = _numeric_validity(data)
    data.loc[:, list(NUMERIC_COLUMNS)] = numeric
    data["_row_valid"] = row_valid
    stats = data.groupby(["symbol", "trade_date"], sort=False).agg(
        rows=("symbol", "size"),
        valid=("_row_valid", "all"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
    )
    valid_pairs = {
        (str(symbol), str(trade_date))
        for (symbol, trade_date), row in stats.iterrows()
        if int(row["rows"]) == 1
        and bool(row["valid"])
        and (float(row["volume"]) != 0.0 or float(row["amount"]) != 0.0)
    }
    selected_index = pd.MultiIndex.from_frame(data[["symbol", "trade_date"]])
    valid_index = pd.MultiIndex.from_tuples(
        sorted(valid_pairs), names=("symbol", "trade_date")
    ) if valid_pairs else selected_index[:0]
    out = data.loc[selected_index.isin(valid_index)].copy()
    out["source"] = "mootdx_online"
    out["adjusted_flag"] = "none"
    out = out.loc[:, DAILY_COLUMNS].reset_index(drop=True)
    unresolved = {
        pair: "missing_or_invalid_provider_day" for pair in wanted_pairs - valid_pairs
    }
    return out, unresolved


def _validate_intraday_frame(
    frame: pd.DataFrame,
    wanted: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, dict[tuple[str, str], str]]:
    wanted_pairs = {(symbol, str(date)) for symbol, dates in wanted.items() for date in dates}
    if frame is None or frame.empty:
        return pd.DataFrame(columns=INTRADAY_COLUMNS), {
            item: "missing_provider_day" for item in wanted_pairs
        }
    data = frame.copy()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    data["trade_date"] = data["trade_date"].astype(str).str.slice(0, 10)
    data["bar_time"] = (
        data["bar_time"].astype(str).str.replace(":", "", regex=False).str.slice(0, 4)
        + "00000"
    )
    pair_index = pd.MultiIndex.from_frame(data[["symbol", "trade_date"]])
    data = data.loc[pair_index.isin(pd.MultiIndex.from_tuples(sorted(wanted_pairs)))].copy()
    if data.empty:
        return pd.DataFrame(columns=INTRADAY_COLUMNS), {
            item: "missing_provider_day" for item in wanted_pairs
        }
    numeric, row_valid = _numeric_validity(data)
    data.loc[:, list(NUMERIC_COLUMNS)] = numeric
    data["_row_valid"] = row_valid
    data["_expected_time"] = data["bar_time"].isin(EXPECTED_BAR_TIMES)
    stats = data.groupby(["symbol", "trade_date"], sort=False).agg(
        rows=("bar_time", "size"),
        times=("bar_time", "nunique"),
        expected=("_expected_time", "all"),
        valid=("_row_valid", "all"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
    )
    valid_pairs = {
        (str(symbol), str(trade_date))
        for (symbol, trade_date), row in stats.iterrows()
        if int(row["rows"]) == 48
        and int(row["times"]) == 48
        and bool(row["expected"])
        and bool(row["valid"])
        and (float(row["volume"]) != 0.0 or float(row["amount"]) != 0.0)
    }
    selected_index = pd.MultiIndex.from_frame(data[["symbol", "trade_date"]])
    valid_index = pd.MultiIndex.from_tuples(
        sorted(valid_pairs), names=("symbol", "trade_date")
    ) if valid_pairs else selected_index[:0]
    out = data.loc[selected_index.isin(valid_index)].copy()
    out["source"] = "mootdx_online"
    out["adjusted_flag"] = "none"
    out = out.loc[:, INTRADAY_COLUMNS].reset_index(drop=True)
    unresolved = {
        pair: "missing_or_invalid_complete_48_bar_day"
        for pair in wanted_pairs - valid_pairs
    }
    return out, unresolved


def _numeric_validity(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    numeric = data.loc[:, list(NUMERIC_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype="float64", na_value=np.nan, copy=True)
    finite = np.isfinite(numeric).all(axis=1)
    positive_prices = (numeric[:, :4] > 0).all(axis=1)
    nonnegative_turnover = (numeric[:, 4:6] >= 0).all(axis=1)
    valid_ohlc = (numeric[:, 1] >= numeric[:, [0, 2, 3]].max(axis=1)) & (
        numeric[:, 2] <= numeric[:, [0, 1, 3]].min(axis=1)
    )
    return numeric, finite & positive_prices & nonnegative_turnover & valid_ohlc


def _missing_daily_symbols(
    inputs: Mapping[str, _DomainInput],
    *,
    trade_date: str,
    workspace: Path,
) -> tuple[str, ...]:
    with open_guarded_duckdb(
        temp_directory=_runtime_root(workspace) / "inventory_spill",
        threads=4,
    ) as connection:
        rows = connection.execute(
            """
            WITH eligible AS (
              SELECT DISTINCT cast(u.symbol AS VARCHAR) AS symbol
              FROM read_parquet(?, union_by_name=true) AS u
              LEFT JOIN read_parquet(?, union_by_name=true) AS s
                ON cast(s.symbol AS VARCHAR)=cast(u.symbol AS VARCHAR)
               AND cast(s.trade_date AS VARCHAR)=cast(u.trade_date AS VARCHAR)
              WHERE cast(u.trade_date AS VARCHAR)=?
                AND lower(cast(u.board AS VARCHAR))='main'
                AND upper(cast(u.list_status AS VARCHAR))='L'
                AND NOT coalesce(try_cast(s.is_suspended AS BOOLEAN), false)
                AND NOT coalesce(try_cast(s.is_delisted AS BOOLEAN), false)
            ), existing AS (
              SELECT DISTINCT cast(symbol AS VARCHAR) AS symbol
              FROM read_parquet(?, union_by_name=true)
              WHERE cast(trade_date AS VARCHAR)=?
            )
            SELECT e.symbol FROM eligible e
            LEFT JOIN existing x USING(symbol)
            WHERE x.symbol IS NULL
            ORDER BY e.symbol
            """,
            [
                _path_texts(inputs["universe_snapshot"]),
                _path_texts(inputs["security_status"]),
                trade_date,
                _path_texts(inputs[DAILY_DOMAIN]),
                trade_date,
            ],
        ).fetchall()
    return tuple(str(row[0]) for row in rows)


def _missing_intraday_pairs(
    inputs: Mapping[str, _DomainInput],
    *,
    start_date: str,
    end_date: str,
    workspace: Path,
) -> tuple[tuple[str, str], ...]:
    with open_guarded_duckdb(
        temp_directory=_runtime_root(workspace) / "inventory_spill",
        threads=4,
    ) as connection:
        rows = connection.execute(
            """
            WITH eligible AS (
              SELECT DISTINCT cast(d.symbol AS VARCHAR) AS symbol,
                              cast(d.trade_date AS VARCHAR) AS trade_date
              FROM read_parquet(?, union_by_name=true) AS d
              INNER JOIN read_parquet(?, union_by_name=true) AS u
                ON cast(u.symbol AS VARCHAR)=cast(d.symbol AS VARCHAR)
               AND cast(u.trade_date AS VARCHAR)=cast(d.trade_date AS VARCHAR)
              LEFT JOIN read_parquet(?, union_by_name=true) AS s
                ON cast(s.symbol AS VARCHAR)=cast(d.symbol AS VARCHAR)
               AND cast(s.trade_date AS VARCHAR)=cast(d.trade_date AS VARCHAR)
              WHERE cast(d.trade_date AS VARCHAR) BETWEEN ? AND ?
                AND coalesce(try_cast(d.volume AS DOUBLE),0)>0
                AND lower(cast(u.board AS VARCHAR))='main'
                AND upper(cast(u.list_status AS VARCHAR))='L'
                AND NOT coalesce(try_cast(s.is_suspended AS BOOLEAN), false)
                AND NOT coalesce(try_cast(s.is_delisted AS BOOLEAN), false)
            ), complete AS (
              SELECT cast(symbol AS VARCHAR) AS symbol,
                     cast(trade_date AS VARCHAR) AS trade_date
              FROM read_parquet(?, union_by_name=true)
              WHERE cast(trade_date AS VARCHAR) BETWEEN ? AND ?
              GROUP BY 1,2
              HAVING count(*)=48 AND count(DISTINCT cast(bar_time AS VARCHAR))=48
            )
            SELECT e.symbol,e.trade_date FROM eligible e
            LEFT JOIN complete c USING(symbol,trade_date)
            WHERE c.symbol IS NULL
            ORDER BY e.symbol,e.trade_date
            """,
            [
                _path_texts(inputs[DAILY_DOMAIN]),
                _path_texts(inputs["universe_snapshot"]),
                _path_texts(inputs["security_status"]),
                start_date,
                end_date,
                _path_texts(inputs[INTRADAY_DOMAIN]),
                start_date,
                end_date,
            ],
        ).fetchall()
    return tuple((str(symbol), str(trade_date)) for symbol, trade_date in rows)


def _active_inputs(
    workspace: Path,
    domains: Sequence[str],
    *,
    start_date: str = "",
    end_date: str = "",
) -> dict[str, _DomainInput]:
    root = qdp_v2_root(workspace).resolve()
    active = read_active_manifest(root)
    inputs: dict[str, _DomainInput] = {}
    for domain in domains:
        dataset_id = str(active.get("datasets", {}).get(domain, ""))
        manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
        if manifest_path is None:
            raise RecentMarketRepairError(f"recent_repair_active_domain_missing:{domain}")
        manifest = read_dataset_manifest(manifest_path)
        entries = manifest.shards
        if start_date and end_date:
            overlapping = [
                item
                for item in entries
                if not item.start_date
                or not item.end_date
                or (str(item.start_date) <= end_date and str(item.end_date) >= start_date)
            ]
            if overlapping:
                entries = overlapping
        paths = tuple(resolve_manifest_path(item.path, root=root) for item in entries)
        if not paths or any(not item.is_file() for item in paths):
            raise RecentMarketRepairError(f"recent_repair_active_shards_missing:{domain}")
        inputs[domain] = _DomainInput(
            domain=domain,
            dataset_id=dataset_id,
            manifest_path=manifest_path,
            manifest_sha256=_sha256_file(manifest_path),
            paths=paths,
        )
    return inputs


def _commit_bundles(
    state: dict[str, Any],
    *,
    runtime: Path,
    domain: str,
    reason: str,
    workspace: Path,
) -> dict[str, Any]:
    bundle_paths = [
        Path(item["bundle_path"])
        for item in state.get("buckets", {}).values()
        if str(item.get("bundle_path", ""))
    ]
    if bundle_paths:
        result = bulk_append_active_shards_from_parquet(
            domain,
            bundle_paths,
            reason,
            workspace_root=workspace,
        )
    else:
        result = {"status": "nothing_to_append", "row_count": 0}
    state.update(
        {
            "status": "applied",
            "commit": result,
            "updated_at": _utc_now(),
        }
    )
    atomic_write_json(runtime / "state.json", state)
    return state


def _write_final_parquet(frame: pd.DataFrame, path: Path, *, domain: str) -> None:
    columns = DAILY_COLUMNS if domain == DAILY_DOMAIN else INTRADAY_COLUMNS
    schema = DAILY_SCHEMA if domain == DAILY_DOMAIN else INTRADAY_SCHEMA
    ordered = frame.loc[:, columns].copy()
    table = pa.Table.from_pandas(ordered, schema=schema, preserve_index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=True,
            row_group_size=100_000,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_or_initialize_state(
    runtime: Path,
    *,
    fingerprint: str,
    kind: str,
    start_date: str,
    end_date: str,
    task_count: int,
    resume: bool,
) -> dict[str, Any]:
    runtime.mkdir(parents=True, exist_ok=True)
    state_path = runtime / "state.json"
    if state_path.is_file():
        if not resume:
            raise RecentMarketRepairError("recent_repair_existing_job_requires_resume")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if str(state.get("fingerprint", "")) != fingerprint:
            raise RecentMarketRepairError("recent_repair_state_fingerprint_mismatch")
        return state
    state = {
        "contract": "qdp_v2_recent_market_direct_repair_v1",
        "fingerprint": fingerprint,
        "kind": kind,
        "status": "pending",
        "start_date": start_date,
        "end_date": end_date,
        "task_count": int(task_count),
        "bucket_count": BUCKET_COUNT,
        "workers": DEFAULT_WORKERS,
        "buckets": {},
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
    }
    atomic_write_json(state_path, state)
    return state


def _validate_reused_bucket(record: Mapping[str, Any], *, runtime: Path) -> None:
    text = str(record.get("bundle_path", ""))
    if not text:
        if int(record.get("row_count", 0)) != 0:
            raise RecentMarketRepairError("recent_repair_empty_bundle_row_count_mismatch")
        return
    path = Path(text).resolve()
    if not path.is_file() or runtime.resolve() not in path.parents:
        raise RecentMarketRepairError(f"recent_repair_reused_bundle_missing:{path}")
    if _sha256_file(path) != str(record.get("bundle_sha256", "")):
        raise RecentMarketRepairError(f"recent_repair_reused_bundle_hash_mismatch:{path}")
    if pq.ParquetFile(path).metadata.num_rows != int(record.get("row_count", 0)):
        raise RecentMarketRepairError(f"recent_repair_reused_bundle_rows_mismatch:{path}")


def _unresolved_mapping(
    unresolved: Mapping[tuple[str, str], str],
) -> dict[str, tuple[str, ...]]:
    result: dict[str, list[str]] = {}
    for symbol, trade_date in unresolved:
        result.setdefault(symbol, []).append(trade_date)
    return {symbol: tuple(sorted(set(dates))) for symbol, dates in result.items()}


def _unresolved_pairs_from_state(state: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    pairs: set[tuple[str, str]] = set()
    for record in dict(state.get("buckets", {}) or {}).values():
        for item in list(record.get("unresolved", []) or []):
            symbol = str(item.get("symbol", "")).strip().upper()
            trade_date = str(item.get("trade_date", ""))[:10]
            if symbol and trade_date:
                pairs.add((symbol, trade_date))
    return tuple(sorted(pairs))


def _split_symbols(symbols: Sequence[str], workers: int) -> tuple[tuple[str, ...], ...]:
    count = min(max(1, int(workers)), len(symbols))
    return tuple(tuple(symbols[index::count]) for index in range(count))


def _stable_bucket(symbol: str) -> int:
    return int(hashlib.sha256(str(symbol).encode("utf-8")).hexdigest()[:8], 16) % BUCKET_COUNT


def _path_texts(value: _DomainInput) -> list[str]:
    return [str(item) for item in value.paths]


def _runtime_root(workspace: Path) -> Path:
    root = (qdp_paths(workspace).data_dir / "qdp_runtime" / "recent_market_repair").resolve()
    if workspace.resolve() not in root.parents:
        raise RecentMarketRepairError(f"recent_repair_runtime_outside_workspace:{root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _configure_workspace_runtime(workspace: Path) -> None:
    paths = qdp_paths(workspace)
    os.environ["QDP_DATA_ROOT"] = str(paths.data_dir)
    runtime = (paths.data_dir / "qdp_runtime").resolve()
    os.environ["QDP_RUNTIME_ROOT"] = str(runtime)
    mootdx_state = runtime / "mootdx" / "last_good_5m.json"
    os.environ.setdefault("QDP_MOOTDX_LAST_GOOD_PATH", str(mootdx_state))


def _date_text(value: str) -> str:
    return pd.Timestamp(str(value)).strftime("%Y-%m-%d")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").floor("s").isoformat()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("daily", "intraday", "tushare-fallback", "all")
    )
    parser.add_argument("--workspace-root", default=str(Path.cwd()))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--mootdx-state-path", default="")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    resume = not bool(args.no_resume)
    results: dict[str, Any] = {}
    if args.command in {"daily", "all"}:
        results["daily"] = run_recent_daily_repair(
            trade_date=args.end_date,
            workspace_root=args.workspace_root,
            workers=args.workers,
            resume=resume,
        )
    if args.command in {"intraday", "all"}:
        results["intraday"] = run_recent_intraday_repair(
            start_date=args.start_date,
            end_date=args.end_date,
            workspace_root=args.workspace_root,
            workers=args.workers,
            resume=resume,
        )
    if args.command == "tushare-fallback":
        if not str(args.mootdx_state_path or "").strip():
            raise ValueError("--mootdx-state-path is required for tushare-fallback")
        results["tushare_fallback"] = run_tushare_intraday_fallback(
            mootdx_state_path=args.mootdx_state_path,
            workspace_root=args.workspace_root,
            workers=args.workers,
            resume=resume,
        )
    print(json.dumps(results, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
