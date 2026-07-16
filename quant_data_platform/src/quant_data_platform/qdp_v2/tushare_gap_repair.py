from __future__ import annotations

"""Download only unresolved v2 5-minute gaps into one compact staging file."""

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import time
import uuid
import zipfile
from concurrent.futures import (
    FIRST_COMPLETED,
    CancelledError,
    Future,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from quant_data_platform.core.json_io import read_json
from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.domains.contracts import HistoryPageFetchRequest
from quant_data_platform.qdp_v2.duckdb_resources import open_guarded_duckdb
from quant_data_platform.qdp_v2.intraday_repair import (
    EXPECTED_5M_BAR_ENDS,
    PATCH_COLUMNS,
    IntradayRepairResourceGuardError,
    _MemoryGuard,
    _acquire_job_process_lock,
    _normalize_complete_archive_day,
    _release_job_process_lock,
)
from quant_data_platform.qdp_v2.manifest import (
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
)
from quant_data_platform.qdp_v2.status import active_dataset_map
from quant_data_platform.qdp_v3.intraday import normalize_tushare_proxy_5m
from quant_data_platform.tushare_proxy import (
    TUSHARE_PROXY_HISTORY_PAGE_SIZE,
    TushareProxyClient,
    TushareProxyProtocolError,
    TushareProxyQuotaError,
    redact_secrets,
)


GAP_DOWNLOAD_VERSION = 2
DEFAULT_RECENT_START = "2026-06-29"
DEFAULT_CUTOFF = "2026-07-13"
DEFAULT_SOURCE_NAME = "tushare_proxy_5m_direct_gap_repair"
DEFAULT_WORKERS = 3
DEFAULT_MINIMUM_FREE_BYTES = 200 * 1024**3
TASK_SCOPES = ("full_unresolved", "recent_tail")
_NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume", "amount")


class TushareGapRepairError(RuntimeError):
    pass


class TushareGapResourceGuardError(TushareGapRepairError):
    pass


@dataclass(frozen=True)
class GapTask:
    symbol: str
    start_date: str
    end_date: str
    scope: str

    def to_dict(self) -> dict[str, str]:
        return {
            "symbol": self.symbol,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "scope": self.scope,
        }


@dataclass(frozen=True)
class GapCapture:
    symbol: str
    page_count: int
    captured_rows: int
    valid_days: int
    rejected_days: int
    payload: bytes
    payload_sha256: str
    page_receipts_json: str
    reject_sample_json: str


def build_gap_tasks(
    *,
    patch_manifest_path: str | Path,
    planner_database_path: str | Path,
    workspace_root: str | Path | None = None,
    recent_start: str = DEFAULT_RECENT_START,
    cutoff: str = DEFAULT_CUTOFF,
    lifecycle_ranges: Mapping[str, tuple[str, str]] | None = None,
) -> tuple[list[GapTask], dict[str, Any]]:
    manifest_path = Path(patch_manifest_path).resolve()
    payload = read_json(manifest_path)
    selected = {str(item) for item in payload.get("selected_symbols", ()) or ()}
    archive_missing = {
        str(item) for item in payload.get("eligible_missing_archive", ()) or ()
    }
    if not selected:
        raise TushareGapRepairError("tushare_gap_patch_selected_symbols_empty")
    rejected = _rejected_symbols(Path(planner_database_path).resolve())
    target = sorted(selected | archive_missing)
    full_symbols = archive_missing | rejected
    recent = _date_text(recent_start)
    end = _date_text(cutoff)
    if recent > end:
        raise ValueError(f"tushare_gap_recent_range_invalid:{recent}>{end}")
    lifecycles = dict(lifecycle_ranges or _active_daily_lifecycle_ranges(
        target,
        workspace_root=workspace_root,
    ))
    tasks: list[GapTask] = []
    excluded_outside_lifecycle = 0
    for symbol in target:
        listed, last_trade = lifecycles.get(symbol, ("2010-01-01", end))
        requested_start = "2010-01-01" if symbol in full_symbols else recent
        effective_start = max(requested_start, str(listed or requested_start)[:10])
        effective_end = min(end, str(last_trade or end)[:10])
        if effective_start > effective_end:
            excluded_outside_lifecycle += 1
            continue
        tasks.append(
            GapTask(
                symbol=symbol,
                start_date=effective_start,
                end_date=effective_end,
                scope="full_unresolved" if symbol in full_symbols else "recent_tail",
            )
        )
    tasks.sort(key=lambda item: (item.scope != "full_unresolved", item.symbol))
    inventory = {
        "target_symbol_count": len(target),
        "task_count": len(tasks),
        "full_unresolved_symbol_count": sum(
            item.scope == "full_unresolved" for item in tasks
        ),
        "recent_tail_symbol_count": sum(item.scope == "recent_tail" for item in tasks),
        "archive_missing_symbol_count": len(archive_missing),
        "archive_rejected_symbol_count": len(rejected),
        "outside_lifecycle_count": excluded_outside_lifecycle,
        "recent_start": recent,
        "cutoff": end,
    }
    return tasks, inventory


def run_tushare_gap_download(
    *,
    patch_manifest_path: str | Path,
    planner_database_path: str | Path,
    workspace_root: str | Path | None = None,
    recent_start: str = DEFAULT_RECENT_START,
    cutoff: str = DEFAULT_CUTOFF,
    lifecycle_ranges: Mapping[str, tuple[str, str]] | None = None,
    client: TushareProxyClient | None = None,
    max_workers: int = DEFAULT_WORKERS,
    minimum_free_bytes: int = DEFAULT_MINIMUM_FREE_BYTES,
    resume: bool = True,
    source_name: str = DEFAULT_SOURCE_NAME,
    task_scopes: Sequence[str] = TASK_SCOPES,
) -> dict[str, Any]:
    tasks, inventory = build_gap_tasks(
        patch_manifest_path=patch_manifest_path,
        planner_database_path=planner_database_path,
        workspace_root=workspace_root,
        recent_start=recent_start,
        cutoff=cutoff,
        lifecycle_ranges=lifecycle_ranges,
    )
    normalized_source = str(source_name or "").strip()
    if not normalized_source:
        raise ValueError("tushare_gap_source_name_required")
    selected_scopes = _normalize_task_scopes(task_scopes)
    task_fingerprint = _stable_hash(
        {
            "version": GAP_DOWNLOAD_VERSION,
            "tasks": [item.to_dict() for item in tasks],
            "source_name": normalized_source,
        }
    )
    job_id = f"tushare_5m_gap__{task_fingerprint[:24]}"
    workspace = Path(workspace_root or Path.cwd()).resolve()
    runtime_root = (
        qdp_paths(workspace_root).data_dir
        / "qdp_runtime"
        / "tushare_gap_repair"
        / job_id
    ).resolve()
    if not _is_within(runtime_root, workspace):
        raise TushareGapRepairError(f"tushare_gap_runtime_outside_workspace:{runtime_root}")
    runtime_root.mkdir(parents=True, exist_ok=True)
    database_path = runtime_root / "gap.sqlite3"
    state_path = runtime_root / "state.json"
    archive_path = runtime_root / "tushare_gap_5m.zip"
    if not resume and database_path.exists():
        raise TushareGapRepairError("tushare_gap_existing_job_requires_resume")

    lock = _acquire_job_process_lock(
        runtime_root.parent / ".locks" / f"{job_id}.lock"
    )
    source: Any | None = None
    guard = _MemoryGuard(0.5, 5.0)
    connection: sqlite3.Connection | None = None
    try:
        source = client or TushareProxyClient()
        connection = sqlite3.connect(database_path)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=FILE")
        _initialize_database(connection)
        _bind_inventory(
            connection,
            tasks=tasks,
            task_fingerprint=task_fingerprint,
            source_name=normalized_source,
        )
        connection.execute(
            "UPDATE tasks SET status='pending', error='' "
            "WHERE status IN ('running', 'failed')"
        )
        connection.commit()
        _write_state(
            state_path,
            connection,
            status="running",
            job_id=job_id,
            inventory=inventory,
            source=source,
            archive_path=archive_path,
            task_scopes=selected_scopes,
        )

        placeholders = ",".join("?" for _ in selected_scopes)
        pending = [
            GapTask(str(row[0]), str(row[1]), str(row[2]), str(row[3]))
            for row in connection.execute(
                "SELECT symbol, start_date, end_date, scope FROM tasks "
                f"WHERE status != 'completed' AND scope IN ({placeholders}) "
                "ORDER BY CASE scope WHEN 'full_unresolved' THEN 0 ELSE 1 END, symbol",
                list(selected_scopes),
            )
        ]
        workers = max(1, min(DEFAULT_WORKERS, int(max_workers)))
        stop_reason = ""
        next_index = 0
        live: dict[Future[GapCapture], GapTask] = {}
        protocol_errors = 0
        last_state_write = time.monotonic()

        def submit(executor: ThreadPoolExecutor) -> None:
            nonlocal next_index, stop_reason
            while (
                not stop_reason
                and len(live) < workers
                and next_index < len(pending)
            ):
                task = pending[next_index]
                next_index += 1
                try:
                    guard.check(f"tushare_gap_submit:{task.symbol}")
                except IntradayRepairResourceGuardError:
                    next_index -= 1
                    stop_reason = "paused_low_memory"
                    return
                connection.execute(
                    "UPDATE tasks SET status='running', updated_at=? WHERE symbol=?",
                    [_utc_now(), task.symbol],
                )
                connection.commit()
                live[executor.submit(
                    _fetch_gap_task,
                    task,
                    source,
                    guard,
                    normalized_source,
                    int(minimum_free_bytes),
                    workspace,
                )] = task

        with ThreadPoolExecutor(max_workers=workers) as executor:
            submit(executor)
            while live:
                done, _ = wait(tuple(live), return_when=FIRST_COMPLETED)
                for future in done:
                    task = live.pop(future)
                    try:
                        if future.cancelled():
                            raise CancelledError()
                        capture = future.result()
                        _store_capture(connection, capture)
                    except CancelledError:
                        _store_task_error(
                            connection,
                            task.symbol,
                            status="pending",
                            error="",
                        )
                    except IntradayRepairResourceGuardError as exc:
                        stop_reason = "paused_low_memory"
                        _store_task_error(
                            connection,
                            task.symbol,
                            status="pending",
                            error=_safe_error(exc, source),
                        )
                    except TushareGapResourceGuardError as exc:
                        stop_reason = "paused_resource_guard"
                        _store_task_error(
                            connection,
                            task.symbol,
                            status="pending",
                            error=_safe_error(exc, source),
                        )
                    except TushareProxyQuotaError as exc:
                        stop_reason = "paused_quota"
                        _store_task_error(
                            connection,
                            task.symbol,
                            status="pending",
                            error=_safe_error(exc, source),
                        )
                    except TushareProxyProtocolError as exc:
                        protocol_errors += 1
                        _store_task_error(
                            connection,
                            task.symbol,
                            status="failed",
                            error=_safe_error(exc, source),
                        )
                        if protocol_errors >= 5:
                            stop_reason = "circuit_open_protocol"
                    except BaseException as exc:
                        _store_task_error(
                            connection,
                            task.symbol,
                            status="failed",
                            error=_safe_error(exc, source),
                        )
                    finally:
                        connection.commit()
                if stop_reason:
                    for future in live:
                        future.cancel()
                else:
                    submit(executor)
                completed = int(
                    connection.execute(
                        "SELECT count(*) FROM tasks WHERE status='completed'"
                    ).fetchone()[0]
                )
                if (
                    completed % 10 == 0
                    or time.monotonic() - last_state_write >= 30.0
                    or stop_reason
                ):
                    _write_state(
                        state_path,
                        connection,
                        status=stop_reason or "running",
                        job_id=job_id,
                        inventory=inventory,
                        source=source,
                        archive_path=archive_path,
                        task_scopes=selected_scopes,
                    )
                    last_state_write = time.monotonic()

        counts = _task_status_counts(connection, task_scopes=selected_scopes)
        if stop_reason:
            status = stop_reason
        elif counts.get("pending", 0) or counts.get("running", 0):
            status = "pending"
        elif counts.get("failed", 0):
            status = "completed_with_failures"
        else:
            status = "completed"
        archive_info: dict[str, Any] = {}
        if status in {"completed", "completed_with_failures"}:
            archive_info = _export_gap_archive(
                connection,
                archive_path=archive_path,
                guard=guard,
                task_scopes=selected_scopes,
            )
        _write_state(
            state_path,
            connection,
            status=status,
            job_id=job_id,
            inventory=inventory,
            source=source,
            archive_path=archive_path,
            archive_info=archive_info,
            task_scopes=selected_scopes,
        )
        return _result(
            connection,
            status=status,
            job_id=job_id,
            runtime_root=runtime_root,
            state_path=state_path,
            database_path=database_path,
            archive_path=archive_path,
            inventory=inventory,
            source=source,
            archive_info=archive_info,
        )
    finally:
        if connection is not None:
            connection.close()
        if source is not None:
            try:
                source.close_thread_session()
            except AttributeError:
                pass
        _release_job_process_lock(lock)


def _fetch_gap_task(
    task: GapTask,
    client: TushareProxyClient,
    guard: _MemoryGuard,
    source_name: str,
    minimum_free_bytes: int,
    workspace: Path,
) -> GapCapture:
    frames: list[pd.DataFrame] = []
    receipts: list[dict[str, Any]] = []
    cursor = f"{task.end_date} 23:59:59"
    start_at = f"{task.start_date} 00:00:00"
    previous_min = ""
    try:
        while True:
            guard.check(f"tushare_gap_page:{task.symbol}")
            free_bytes = int(shutil.disk_usage(workspace).free)
            if free_bytes < int(minimum_free_bytes):
                raise TushareGapResourceGuardError(
                    f"tushare_gap_disk_floor:free={free_bytes}:required={minimum_free_bytes}"
                )
            result = client.fetch_history_page(
                HistoryPageFetchRequest(
                    provider_symbol=task.symbol,
                    start_at=start_at,
                    end_at=cursor,
                    page_size=TUSHARE_PROXY_HISTORY_PAGE_SIZE,
                )
            )
            if (
                previous_min
                and result.max_timestamp
                and pd.Timestamp(result.max_timestamp) >= pd.Timestamp(previous_min)
            ):
                raise TushareProxyProtocolError(
                    f"tushare_gap_cross_page_overlap:{task.symbol}"
                )
            if not result.raw_data.empty:
                frames.append(result.raw_data.copy())
            receipts.append(
                {
                    "page": len(receipts) + 1,
                    "row_count": int(result.row_count),
                    "min_timestamp": str(result.min_timestamp),
                    "max_timestamp": str(result.max_timestamp),
                    "response_sha256": str(result.response_sha256),
                }
            )
            previous_min = str(result.min_timestamp or previous_min)
            if result.is_complete:
                break
            if not result.next_end_at:
                raise TushareProxyProtocolError(
                    f"tushare_gap_missing_next_cursor:{task.symbol}"
                )
            cursor = str(result.next_end_at)

        captured_rows = sum(len(item) for item in frames)
        guard.check(f"tushare_gap_normalize:{task.symbol}")
        if frames:
            raw = pd.concat(frames, ignore_index=True, sort=False)
            normalized = normalize_tushare_proxy_5m(
                raw,
                provider_symbol=task.symbol,
            )
            normalized = normalized.sort_values(
                ["trade_date", "bar_end"], kind="stable"
            ).reset_index(drop=True)
            if normalized.duplicated(["trade_date", "bar_end"]).any():
                raise TushareProxyProtocolError(
                    f"tushare_gap_duplicate_bar:{task.symbol}"
                )
        else:
            normalized = pd.DataFrame()
        frame, valid_days, rejected_days, reject_sample = _normalize_gap_days(
            normalized,
            canonical_symbol=task.symbol,
            source_name=source_name,
            guard=guard,
        )
        payload = _parquet_bytes(frame)
        return GapCapture(
            symbol=task.symbol,
            page_count=len(receipts),
            captured_rows=int(captured_rows),
            valid_days=int(valid_days),
            rejected_days=int(rejected_days),
            payload=payload,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            page_receipts_json=json.dumps(
                receipts, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
            reject_sample_json=json.dumps(
                reject_sample,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    finally:
        try:
            client.close_thread_session()
        except AttributeError:
            pass


def _normalize_gap_days(
    normalized: pd.DataFrame,
    *,
    canonical_symbol: str,
    source_name: str,
    guard: _MemoryGuard,
) -> tuple[pd.DataFrame, int, int, list[dict[str, str]]]:
    """Vectorize ordinary 48-bar days; fall back only for exceptional days."""

    if normalized.empty:
        return pd.DataFrame(columns=PATCH_COLUMNS), 0, 0, []
    guard.check(f"tushare_gap_vector_gate:{canonical_symbol}")
    data = normalized.copy()
    data["trade_date"] = data["trade_date"].astype(str)
    data["bar_end"] = data["bar_end"].astype(str)
    try:
        numeric = data.loc[:, list(_NUMERIC_COLUMNS)].to_numpy(
            dtype="float64", na_value=np.nan, copy=True
        )
    except (TypeError, ValueError):
        numeric = data.loc[:, list(_NUMERIC_COLUMNS)].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype="float64", na_value=np.nan, copy=True)
    finite = np.isfinite(numeric).all(axis=1)
    positive_prices = (numeric[:, :4] > 0).all(axis=1)
    nonnegative_turnover = (numeric[:, 4:6] >= 0).all(axis=1)
    valid_ohlc = (
        numeric[:, 1] >= numeric[:, [0, 2, 3]].max(axis=1)
    ) & (
        numeric[:, 2] <= numeric[:, [0, 1, 3]].min(axis=1)
    )
    flags = pd.DataFrame(
        {
            "trade_date": data["trade_date"].to_numpy(copy=False),
            "bar_end": data["bar_end"].to_numpy(copy=False),
            "expected_bar": data["bar_end"].isin(EXPECTED_5M_BAR_ENDS).to_numpy(),
            "valid_numeric": finite
            & positive_prices
            & nonnegative_turnover
            & valid_ohlc,
            "volume": numeric[:, 4],
            "amount": numeric[:, 5],
        }
    )
    stats = flags.groupby("trade_date", sort=True).agg(
        row_count=("bar_end", "size"),
        distinct_bar_count=("bar_end", "nunique"),
        expected_bars=("expected_bar", "all"),
        valid_numeric=("valid_numeric", "all"),
        volume_sum=("volume", "sum"),
        amount_sum=("amount", "sum"),
    )
    fast_mask = (
        stats["row_count"].eq(48)
        & stats["distinct_bar_count"].eq(48)
        & stats["expected_bars"]
        & stats["valid_numeric"]
        & (stats["volume_sum"].ne(0.0) | stats["amount_sum"].ne(0.0))
    )
    fast_dates = {str(item) for item in stats.index[fast_mask]}
    fast_row_mask = data["trade_date"].isin(fast_dates)
    valid_frames: list[pd.DataFrame] = []
    if fast_dates:
        fast = data.loc[fast_row_mask].copy()
        fast.loc[:, list(_NUMERIC_COLUMNS)] = numeric[fast_row_mask.to_numpy()]
        fast = fast.assign(
            symbol=canonical_symbol,
            bar_time=fast["bar_end"].str.replace(":", "", regex=False) + "00000",
            source=source_name,
            adjusted_flag="none",
        )
        valid_frames.append(fast.loc[:, PATCH_COLUMNS])

    rejected_days = 0
    fallback_valid_days = 0
    reject_sample: list[dict[str, str]] = []
    fallback = data.loc[~fast_row_mask]
    if not fallback.empty:
        for trade_date, day in fallback.groupby("trade_date", sort=True):
            guard.check(f"tushare_gap_fallback_day:{canonical_symbol}:{trade_date}")
            patch, reason = _normalize_complete_archive_day(
                day,
                canonical_symbol=canonical_symbol,
                trade_date=str(trade_date),
                source_name=source_name,
            )
            if reason:
                rejected_days += 1
                if len(reject_sample) < 20:
                    reject_sample.append(
                        {"trade_date": str(trade_date), "reason": reason}
                    )
            else:
                valid_frames.append(patch)
                fallback_valid_days += 1

    if valid_frames:
        frame = pd.concat(valid_frames, ignore_index=True, sort=False)
        frame = frame.loc[:, PATCH_COLUMNS].sort_values(
            ["trade_date", "bar_time"], kind="stable"
        ).reset_index(drop=True)
    else:
        frame = pd.DataFrame(columns=PATCH_COLUMNS)
    valid_days = len(fast_dates) + fallback_valid_days
    return frame, int(valid_days), int(rejected_days), reject_sample


def _active_daily_lifecycle_ranges(
    symbols: Sequence[str],
    *,
    workspace_root: str | Path | None,
) -> dict[str, tuple[str, str]]:
    root = qdp_v2_root(workspace_root).resolve()
    active = read_active_manifest(root)
    dataset_id = active_dataset_map(active).get("market_daily_raw", "")
    path = dataset_manifest_for_id(root, dataset_id, "market_daily_raw")
    if path is None:
        raise TushareGapRepairError("tushare_gap_active_daily_manifest_missing")
    manifest = read_dataset_manifest(path)
    paths = [str(resolve_manifest_path(item.path, root=root)) for item in manifest.shards]
    target = pd.DataFrame({"symbol": list(symbols)})
    spill = (
        qdp_paths(workspace_root).data_dir
        / "qdp_runtime"
        / "tushare_gap_repair"
        / "lifecycle_spill"
    )
    with open_guarded_duckdb(temp_directory=spill, threads=4) as connection:
        connection.register("gap_target_symbols", target)
        rows = connection.execute(
            """
            SELECT CAST(d.symbol AS VARCHAR),
                   strftime(min(try_cast(d.trade_date AS DATE)), '%Y-%m-%d'),
                   strftime(max(try_cast(d.trade_date AS DATE)), '%Y-%m-%d')
            FROM read_parquet(?, union_by_name=true) AS d
            INNER JOIN gap_target_symbols AS t
              ON CAST(d.symbol AS VARCHAR) = t.symbol
            GROUP BY d.symbol
            """,
            [paths],
        ).fetchall()
    return {str(symbol): (str(first), str(last)) for symbol, first, last in rows}


def _rejected_symbols(database_path: Path) -> set[str]:
    if not database_path.is_file():
        raise FileNotFoundError(f"tushare_gap_planner_database_missing:{database_path}")
    connection = sqlite3.connect(f"file:{database_path.as_posix()}?mode=ro", uri=True)
    try:
        return {
            str(row[0])
            for row in connection.execute(
                "SELECT symbol FROM symbol_receipts WHERE rejected_days > 0"
            )
        }
    finally:
        connection.close()


def _initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tasks(
            symbol TEXT PRIMARY KEY,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            scope TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            page_count INTEGER NOT NULL DEFAULT 0,
            captured_rows INTEGER NOT NULL DEFAULT 0,
            valid_days INTEGER NOT NULL DEFAULT 0,
            rejected_days INTEGER NOT NULL DEFAULT 0,
            payload_sha256 TEXT NOT NULL DEFAULT '',
            payload BLOB,
            page_receipts_json TEXT NOT NULL DEFAULT '[]',
            reject_sample_json TEXT NOT NULL DEFAULT '[]',
            error TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT ''
        );
        """
    )
    connection.commit()


def _bind_inventory(
    connection: sqlite3.Connection,
    *,
    tasks: Sequence[GapTask],
    task_fingerprint: str,
    source_name: str,
) -> None:
    expected = {
        "version": str(GAP_DOWNLOAD_VERSION),
        "task_fingerprint": task_fingerprint,
        "source_name": source_name,
        "task_count": str(len(tasks)),
    }
    for key, value in expected.items():
        row = connection.execute(
            "SELECT value FROM metadata WHERE key=?", [key]
        ).fetchone()
        if row and str(row[0]) != value:
            raise TushareGapRepairError(
                f"tushare_gap_runtime_fingerprint_mismatch:{key}"
            )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
            [key, value],
        )
    for task in tasks:
        row = connection.execute(
            "SELECT start_date,end_date,scope FROM tasks WHERE symbol=?",
            [task.symbol],
        ).fetchone()
        expected_task = (task.start_date, task.end_date, task.scope)
        if row and tuple(str(item) for item in row) != expected_task:
            raise TushareGapRepairError(
                f"tushare_gap_task_fingerprint_mismatch:{task.symbol}"
            )
        connection.execute(
            "INSERT OR IGNORE INTO tasks(symbol,start_date,end_date,scope) "
            "VALUES (?,?,?,?)",
            [task.symbol, task.start_date, task.end_date, task.scope],
        )
    connection.commit()


def _store_capture(connection: sqlite3.Connection, capture: GapCapture) -> None:
    connection.execute(
        """
        UPDATE tasks
        SET status='completed', page_count=?, captured_rows=?, valid_days=?,
            rejected_days=?, payload_sha256=?, payload=?, page_receipts_json=?,
            reject_sample_json=?, error='', updated_at=?
        WHERE symbol=?
        """,
        [
            capture.page_count,
            capture.captured_rows,
            capture.valid_days,
            capture.rejected_days,
            capture.payload_sha256,
            sqlite3.Binary(capture.payload),
            capture.page_receipts_json,
            capture.reject_sample_json,
            _utc_now(),
            capture.symbol,
        ],
    )


def _store_task_error(
    connection: sqlite3.Connection,
    symbol: str,
    *,
    status: str,
    error: str,
) -> None:
    connection.execute(
        "UPDATE tasks SET status=?, error=?, updated_at=? WHERE symbol=?",
        [str(status), str(error)[:1000], _utc_now(), str(symbol)],
    )


def _export_gap_archive(
    connection: sqlite3.Connection,
    *,
    archive_path: Path,
    guard: _MemoryGuard,
    task_scopes: Sequence[str],
) -> dict[str, Any]:
    temporary = archive_path.with_name(
        f".{archive_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    member_count = 0
    row_count = 0
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            placeholders = ",".join("?" for _ in task_scopes)
            cursor = connection.execute(
                "SELECT symbol,payload,payload_sha256,valid_days FROM tasks "
                f"WHERE status='completed' AND valid_days>0 AND scope IN ({placeholders}) "
                "ORDER BY symbol",
                list(task_scopes),
            )
            for symbol, blob, expected_sha, valid_days in cursor:
                guard.check(f"tushare_gap_export:{symbol}")
                payload = bytes(blob or b"")
                if hashlib.sha256(payload).hexdigest() != str(expected_sha):
                    raise TushareGapRepairError(
                        f"tushare_gap_payload_hash_mismatch:{symbol}"
                    )
                frame = pq.read_table(pa.BufferReader(payload)).to_pandas()
                if len(frame) != int(valid_days) * 48:
                    raise TushareGapRepairError(
                        f"tushare_gap_payload_row_count_mismatch:{symbol}"
                    )
                time_text = frame["bar_time"].astype(str).str.slice(0, 4)
                exported = pd.DataFrame(
                    {
                        "datetime": frame["trade_date"].astype(str)
                        + " "
                        + time_text.str.slice(0, 2)
                        + ":"
                        + time_text.str.slice(2, 4),
                        "open": frame["open"],
                        "high": frame["high"],
                        "low": frame["low"],
                        "close": frame["close"],
                        "volume": frame["volume"],
                        "amount": frame["amount"],
                    }
                )
                code, exchange = str(symbol).split(".", 1)
                info = zipfile.ZipInfo(
                    filename=f"{exchange.lower()}{code}.csv",
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(
                    info,
                    exported.to_csv(index=False, lineterminator="\n").encode("utf-8-sig"),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=6,
                )
                member_count += 1
                row_count += len(frame)
        os.replace(temporary, archive_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "archive_path": str(archive_path),
        "archive_sha256": _sha256_file(archive_path),
        "archive_size": int(archive_path.stat().st_size),
        "archive_member_count": member_count,
        "archive_row_count": row_count,
    }


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    ordered = frame.loc[:, list(PATCH_COLUMNS)].copy()
    table = pa.Table.from_pandas(ordered, preserve_index=False)
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        use_dictionary=True,
        row_group_size=100_000,
    )
    return sink.getvalue().to_pybytes()


def _task_status_counts(
    connection: sqlite3.Connection,
    *,
    task_scopes: Sequence[str] | None = None,
) -> dict[str, int]:
    if task_scopes:
        placeholders = ",".join("?" for _ in task_scopes)
        rows = connection.execute(
            f"SELECT status,count(*) FROM tasks WHERE scope IN ({placeholders}) GROUP BY status",
            list(task_scopes),
        )
    else:
        rows = connection.execute(
            "SELECT status,count(*) FROM tasks GROUP BY status"
        )
    return {
        str(status): int(count)
        for status, count in rows
    }


def _write_state(
    state_path: Path,
    connection: sqlite3.Connection,
    *,
    status: str,
    job_id: str,
    inventory: Mapping[str, Any],
    source: Any,
    archive_path: Path,
    archive_info: Mapping[str, Any] | None = None,
    task_scopes: Sequence[str] = TASK_SCOPES,
) -> None:
    selected_scopes = _normalize_task_scopes(task_scopes)
    counts = _task_status_counts(connection, task_scopes=selected_scopes)
    global_counts = _task_status_counts(connection)
    placeholders = ",".join("?" for _ in selected_scopes)
    row = connection.execute(
        "SELECT coalesce(sum(page_count),0),coalesce(sum(captured_rows),0),"
        "coalesce(sum(valid_days),0),coalesce(sum(rejected_days),0) FROM tasks "
        f"WHERE scope IN ({placeholders})",
        list(selected_scopes),
    ).fetchone()
    try:
        metrics = dict(source.operational_metrics())
    except AttributeError:
        metrics = {}
    payload = {
        "version": GAP_DOWNLOAD_VERSION,
        "status": str(status),
        "job_id": job_id,
        **dict(inventory),
        "selected_scopes": list(selected_scopes),
        "selected_task_count": sum(counts.values()),
        "completed": counts.get("completed", 0),
        "pending": counts.get("pending", 0) + counts.get("running", 0),
        "failed": counts.get("failed", 0),
        "global_status_counts": global_counts,
        "page_count": int(row[0] or 0),
        "captured_rows": int(row[1] or 0),
        "valid_days": int(row[2] or 0),
        "rejected_days": int(row[3] or 0),
        "metrics": metrics,
        "archive_path": str(archive_path),
        "archive": dict(archive_info or {}),
        "heartbeat_at": _utc_now(),
    }
    atomic_write_json(state_path, payload)


def _result(
    connection: sqlite3.Connection,
    *,
    status: str,
    job_id: str,
    runtime_root: Path,
    state_path: Path,
    database_path: Path,
    archive_path: Path,
    inventory: Mapping[str, Any],
    source: Any,
    archive_info: Mapping[str, Any],
) -> dict[str, Any]:
    state = read_json(state_path)
    return {
        **state,
        "status": status,
        "job_id": job_id,
        "runtime_root": str(runtime_root),
        "state_path": str(state_path),
        "database_path": str(database_path),
        "archive_path": str(archive_path),
        "archive": dict(archive_info),
    }


def _safe_error(error: BaseException, client: Any) -> str:
    token = str(getattr(getattr(client, "config", None), "token", "") or "")
    return str(redact_secrets(error, secrets=(token,) if token else ()))[:1000]


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_task_scopes(values: Sequence[str]) -> tuple[str, ...]:
    scopes = tuple(dict.fromkeys(str(item or "").strip() for item in values))
    if not scopes or any(not item for item in scopes):
        raise ValueError("tushare_gap_task_scope_required")
    unknown = sorted(set(scopes).difference(TASK_SCOPES))
    if unknown:
        raise ValueError(f"tushare_gap_task_scope_invalid:{unknown}")
    return tuple(item for item in TASK_SCOPES if item in scopes)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _date_text(value: Any) -> str:
    parsed = pd.to_datetime(str(value or ""), errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"tushare_gap_date_invalid:{value}")
    return parsed.strftime("%Y-%m-%d")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m quant_data_platform.qdp_v2.tushare_gap_repair",
        description="Download only unresolved/recent Tushare 5m ranges into one compact resumable job.",
    )
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--patch-manifest", required=True)
    parser.add_argument("--planner-database", required=True)
    parser.add_argument("--recent-start", default=DEFAULT_RECENT_START)
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--scope",
        choices=("all", *TASK_SCOPES),
        default="all",
        help="Restrict execution without changing or duplicating the resumable job.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = str(args.workspace_root or "") or None
    selected_scopes = TASK_SCOPES if args.scope == "all" else (str(args.scope),)
    if args.plan_only:
        tasks, inventory = build_gap_tasks(
            patch_manifest_path=args.patch_manifest,
            planner_database_path=args.planner_database,
            workspace_root=workspace,
            recent_start=args.recent_start,
            cutoff=args.cutoff,
        )
        selected = [item for item in tasks if item.scope in selected_scopes]
        result = {
            **inventory,
            "status": "planned",
            "selected_scopes": list(selected_scopes),
            "selected_task_count": len(selected),
            "tasks": [item.to_dict() for item in selected],
        }
    else:
        result = run_tushare_gap_download(
            patch_manifest_path=args.patch_manifest,
            planner_database_path=args.planner_database,
            workspace_root=workspace,
            recent_start=args.recent_start,
            cutoff=args.cutoff,
            max_workers=int(args.workers),
            resume=not bool(args.no_resume),
            task_scopes=selected_scopes,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"planned", "completed", "completed_with_failures"} else 1


__all__ = [
    "DEFAULT_CUTOFF",
    "DEFAULT_RECENT_START",
    "DEFAULT_SOURCE_NAME",
    "GAP_DOWNLOAD_VERSION",
    "GapCapture",
    "GapTask",
    "TASK_SCOPES",
    "TushareGapRepairError",
    "TushareGapResourceGuardError",
    "build_gap_tasks",
    "run_tushare_gap_download",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
