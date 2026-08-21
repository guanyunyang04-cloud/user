"""Recent Market Repair: baostock responsibilities."""

from __future__ import annotations

import json
import multiprocessing
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data.qdp_v2.duckdb_resources import (
    open_guarded_duckdb,
)
from quantlab.data.qdp_v2.manifest import (
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
)
from quantlab.data.qdp_v2.status import active_dataset_map

from .commit import (
    _write_final_parquet,
)
from .config import (
    BUCKET_COUNT,
    DAILY_DOMAIN,
    INTRADAY_COLUMNS,
    INTRADAY_DOMAIN,
    RecentMarketRepairError,
    _BaostockTask,
    _DomainInput,
    _SystemMemoryGuard,
)
from .session import (
    _baostock_process_session,
    _DirectBaostockSession,
)
from .state import (
    _file_sha256,
    _pairs_sha256,
    _path_texts,
    _requested_pairs,
    _reused_bucket_is_valid,
    _stable_bucket,
    _utc_now,
)
from .validation import (
    _validate_intraday_against_daily,
    _validate_intraday_frame,
)


def _active_intraday_quality(workspace: Path) -> dict[str, Any]:
    """Read current intraday counters before appending a validated bundle."""
    root = qdp_v2_root(workspace)
    active = read_active_manifest(root)
    datasets = active_dataset_map(active)
    dataset_id = datasets.get(INTRADAY_DOMAIN)
    if not dataset_id:
        raise ValueError("active intraday dataset is missing")
    manifest_path = dataset_manifest_for_id(root, dataset_id, INTRADAY_DOMAIN)
    if manifest_path is None:
        raise ValueError("active intraday manifest is missing")
    return dict(read_dataset_manifest(manifest_path).quality)


def _build_baostock_daily_reference(
    inputs: Mapping[str, _DomainInput],
    *,
    wanted: Mapping[str, Sequence[str]],
    runtime: Path,
) -> dict[str, Path]:
    """Materialize one daily-reference file per stable BaoStock bucket."""
    runtime.mkdir(parents=True, exist_ok=True)
    pairs = runtime / "requested_pairs.parquet"
    pair_frame = pd.DataFrame(
        [(symbol, date) for symbol, dates in wanted.items() for date in dates],
        columns=["symbol", "trade_date"],
    )
    pair_frame.to_parquet(pairs, index=False, compression="zstd")
    pairs.unlink(missing_ok=True)
    pair_frame["bucket"] = pair_frame["symbol"].map(_stable_bucket)
    references: dict[str, Path] = {}
    for bucket in range(BUCKET_COUNT):
        key = f"{bucket:02d}"
        target = runtime / f"daily_reference_bucket_{key}.parquet"
        temporary = target.with_suffix(".tmp.parquet")
        bucket_pairs = runtime / f"requested_pairs_bucket_{key}.parquet"
        pair_frame.loc[pair_frame["bucket"] == bucket, ["symbol", "trade_date"]].to_parquet(
            bucket_pairs, index=False, compression="zstd"
        )
        with open_guarded_duckdb(temp_directory=runtime / f"reference_spill_{key}", threads=2) as con:
            con.execute(
                f"""
                COPY (
                  SELECT upper(cast(d.symbol AS VARCHAR)) AS symbol,
                         cast(d.trade_date AS VARCHAR) AS trade_date,
                         try_cast(d.open AS DOUBLE) AS open,
                         try_cast(d.high AS DOUBLE) AS high,
                         try_cast(d.low AS DOUBLE) AS low,
                         try_cast(d.close AS DOUBLE) AS close,
                         try_cast(d.volume AS DOUBLE) AS volume,
                         try_cast(d.amount AS DOUBLE) AS amount
                  FROM read_parquet(?, union_by_name=true) d
                  JOIN read_parquet(?) p
                    ON upper(cast(d.symbol AS VARCHAR))=p.symbol
                   AND cast(d.trade_date AS VARCHAR)=p.trade_date
                  ORDER BY symbol, trade_date
                ) TO '{str(temporary).replace("'", "''")}'
                  (FORMAT PARQUET, COMPRESSION ZSTD)
                """,
                [_path_texts(inputs[DAILY_DOMAIN]), str(bucket_pairs)],
            )
        temporary.replace(target)
        bucket_pairs.unlink(missing_ok=True)
        references[key] = target
    return references


def _pending_baostock_buckets(
    *,
    runtime: Path,
    wanted: Mapping[str, Sequence[str]],
    completed: Mapping[str, Any],
    process_workers: int,
) -> list[tuple[str, dict[str, tuple[str, ...]], Path, float]]:
    pending: list[tuple[str, dict[str, tuple[str, ...]], Path, float]] = []
    for bucket in range(BUCKET_COUNT):
        key = f"{bucket:02d}"
        bucket_wanted = {
            symbol: tuple(sorted({str(item) for item in dates}))
            for symbol, dates in wanted.items()
            if _stable_bucket(symbol) == bucket
        }
        previous = completed.get(key)
        if (
            isinstance(previous, dict)
            and previous.get("status") == "completed"
            and _reused_bucket_is_valid(previous, runtime=runtime, wanted=bucket_wanted)
        ):
            continue
        pending.append(
            (
                key,
                bucket_wanted,
                runtime / "bundles" / f"{INTRADAY_DOMAIN}_bucket_{key}.parquet",
                2.0 * (len(pending) % process_workers),
            )
        )
    return pending


def _collect_baostock_buckets(
    *,
    runtime: Path,
    state: dict[str, Any],
    completed: dict[str, Any],
    pending: list[tuple[str, dict[str, tuple[str, ...]], Path, float]],
    daily_reference_paths: Mapping[str, Path],
    process_workers: int,
    guard: _SystemMemoryGuard,
) -> None:
    if not pending:
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(process_workers, len(pending)),
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(
                _run_baostock_bucket_process,
                bucket_wanted,
                progress_label=key,
                bundle_path_text=str(bundle_path),
                daily_reference_path=str(daily_reference_paths[key]),
                startup_delay_seconds=startup_delay,
            ): key
            for key, bucket_wanted, bundle_path, startup_delay in pending
        }
        for future in as_completed(futures):
            key = futures[future]
            guard.check(f"baostock_bucket_{key}_result")
            record = future.result()
            reference_path = daily_reference_paths[key]
            record["daily_reference_sha256"] = _file_sha256(reference_path)
            record["requested_pairs_sha256"] = _pairs_sha256(
                tuple((str(item[0]), str(item[1])) for item in list(record.get("requested_pairs", []) or []))
            )
            completed[key] = record
            state.update(
                {
                    "status": "downloading",
                    "buckets": completed,
                    "completed_bucket_count": sum(item.get("status") == "completed" for item in completed.values()),
                    "updated_at": _utc_now(),
                }
            )
            atomic_write_json(runtime / "state.json", state)
            print(
                json.dumps(
                    {
                        "provider": "baostock",
                        "bucket": key,
                        "symbols": record["symbol_count"],
                        "rows": record["row_count"],
                        "unresolved_days": len(record["unresolved"]),
                        "provider_errors": record["provider_error_count"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


def _summarize_baostock_buckets(
    state: dict[str, Any],
    *,
    completed: dict[str, Any],
    daily_reference_paths: Mapping[str, Path],
) -> None:
    rejection_totals: dict[str, int] = {}
    for item in completed.values():
        for reason, count in dict(item.get("rejection_counts", {}) or {}).items():
            rejection_totals[str(reason)] = rejection_totals.get(str(reason), 0) + int(count)
    state.update(
        {
            "status": "downloaded",
            "buckets": completed,
            "row_count": sum(int(item.get("row_count", 0)) for item in completed.values()),
            "accepted_day_count": sum(int(item.get("accepted_day_count", 0)) for item in completed.values()),
            "unresolved_day_count": sum(len(item.get("unresolved", []) or []) for item in completed.values()),
            "provider_error_count": sum(int(item.get("provider_error_count", 0)) for item in completed.values()),
            "rejection_counts": rejection_totals,
            "daily_reference_paths": {key: str(path) for key, path in daily_reference_paths.items()},
            "updated_at": _utc_now(),
        }
    )


def _download_baostock_buckets(
    *,
    runtime: Path,
    state: dict[str, Any],
    wanted: Mapping[str, Sequence[str]],
    daily_reference_paths: Mapping[str, Path],
    workers: int,
    guard: _SystemMemoryGuard,
) -> dict[str, Any]:
    bundles = runtime / "bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    completed = dict(state.get("buckets", {}) or {})
    state["requested_workers"] = int(workers)
    process_workers = min(max(1, int(workers)), 4)
    state["workers"] = process_workers
    atomic_write_json(runtime / "state.json", state)
    pending = _pending_baostock_buckets(
        runtime=runtime,
        wanted=wanted,
        completed=completed,
        process_workers=process_workers,
    )
    _collect_baostock_buckets(
        runtime=runtime,
        state=state,
        completed=completed,
        pending=pending,
        daily_reference_paths=daily_reference_paths,
        process_workers=process_workers,
        guard=guard,
    )
    _summarize_baostock_buckets(state, completed=completed, daily_reference_paths=daily_reference_paths)
    atomic_write_json(runtime / "state.json", state)
    return state


def _run_baostock_bucket_process(
    wanted: Mapping[str, Sequence[str]],
    *,
    progress_label: str,
    bundle_path_text: str,
    daily_reference_path: str,
    startup_delay_seconds: float = 0.0,
) -> dict[str, Any]:
    provider = _baostock_process_session(
        startup_delay_seconds=startup_delay_seconds,
    )
    with _SystemMemoryGuard() as guard:
        reference = pd.read_parquet(daily_reference_path)
        reference = reference.loc[reference["symbol"].astype(str).map(_stable_bucket).eq(int(progress_label))].copy()
        frame, unresolved, errors, rejection_counts = _fetch_baostock_bucket(
            wanted,
            provider=provider,
            guard=guard,
            progress_label=progress_label,
            daily_reference=reference,
        )
        bundle_path: Path | None = None
        if not frame.empty:
            bundle_path = Path(bundle_path_text).resolve()
            _write_final_parquet(frame, bundle_path, domain=INTRADAY_DOMAIN)
        return {
            "status": "completed",
            "symbol_count": len(wanted),
            "requested_day_count": sum(len(item) for item in wanted.values()),
            "accepted_day_count": int(len(frame) // 48),
            "row_count": len(frame),
            "bundle_path": str(bundle_path) if bundle_path else "",
            "file_size": bundle_path.stat().st_size if bundle_path else 0,
            "requested_pairs": _requested_pairs(wanted),
            "unresolved": [
                {"symbol": symbol, "trade_date": trade_date, "reason": reason}
                for (symbol, trade_date), reason in sorted(unresolved.items())
            ],
            "provider_error_count": len(errors),
            "provider_error_sample": errors[:10],
            "rejection_counts": dict(rejection_counts),
            "minimum_available_bytes": guard.minimum_available_bytes,
        }


def _fetch_baostock_bucket(
    wanted: Mapping[str, Sequence[str]],
    *,
    provider: _DirectBaostockSession,
    guard: _SystemMemoryGuard,
    progress_label: str,
    daily_reference: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, str], str], list[dict[str, Any]], dict[str, int]]:
    tasks = _baostock_tasks(wanted)
    if not tasks:
        return (
            pd.DataFrame(columns=INTRADAY_COLUMNS),
            {},
            [],
            {},
        )
    accepted: list[pd.DataFrame] = []
    unresolved: dict[tuple[str, str], str] = {}
    errors: list[dict[str, Any]] = []
    for completed, task in enumerate(tasks, start=1):
        guard.check(f"baostock_task_start:{task.symbol}:{task.start_date}")
        if completed == 1 or completed % 10 == 0 or completed == len(tasks):
            print(
                json.dumps(
                    {
                        "provider": "baostock",
                        "bucket": progress_label,
                        "phase": "start_task",
                        "task_index": completed,
                        "task_count": len(tasks),
                        "symbol": task.symbol,
                        "start_date": task.start_date,
                        "end_date": task.end_date,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        try:
            frame, task_unresolved, task_errors = _fetch_baostock_task(
                task,
                provider=provider,
            )
        except BaseException as exc:  # noqa: BLE001 - preserve provider failure state
            frame = pd.DataFrame(columns=INTRADAY_COLUMNS)
            task_unresolved = {(task.symbol, trade_date): "provider_error" for trade_date in task.dates}
            task_errors = [
                {
                    "symbol": task.symbol,
                    "start_date": task.start_date,
                    "end_date": task.end_date,
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:500],
                }
            ]
        if not frame.empty:
            accepted.append(frame)
        unresolved.update(task_unresolved)
        errors.extend(task_errors)
        if completed == 1 or completed % 10 == 0 or completed == len(tasks):
            print(
                json.dumps(
                    {
                        "provider": "baostock",
                        "bucket": progress_label,
                        "completed_tasks": completed,
                        "task_count": len(tasks),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    rejection_counts: dict[str, int] = {}
    if accepted:
        combined = pd.concat(accepted, ignore_index=True, sort=False)
        keys = ["symbol", "trade_date", "bar_time"]
        if combined.duplicated(keys).any():
            raise RecentMarketRepairError("baostock_repair_duplicate_accepted_primary_key")
        combined = combined.sort_values(keys, kind="stable").reset_index(drop=True)
        if daily_reference is not None and not combined.empty:
            combined, daily_unresolved, _diagnostics = _validate_intraday_against_daily(
                combined,
                daily_reference,
                source_name="baostock_history_daily_validated",
            )
            unresolved.update(daily_unresolved)
    else:
        combined = pd.DataFrame(columns=INTRADAY_COLUMNS)
    for reason in unresolved.values():
        rejection_counts[str(reason)] = rejection_counts.get(str(reason), 0) + 1
    return combined, unresolved, errors, rejection_counts


def _baostock_tasks(wanted: Mapping[str, Sequence[str]]) -> tuple[_BaostockTask, ...]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for symbol, dates in wanted.items():
        for trade_date in dates:
            date_text = str(trade_date)
            grouped.setdefault((str(symbol), date_text[:4]), []).append(date_text)
    return tuple(
        _BaostockTask(symbol=symbol, dates=tuple(sorted(set(dates))))
        for (symbol, _year), dates in sorted(grouped.items())
    )


def _fetch_baostock_task(
    task: _BaostockTask,
    *,
    provider: _DirectBaostockSession | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, str], str], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    result_data = pd.DataFrame()
    owned_provider = provider is None
    client = provider or _DirectBaostockSession()
    for attempt in range(1, 3):
        try:
            result_data = client.fetch(task)
            break
        except BaseException as exc:  # noqa: BLE001 - retry/provider boundary
            errors.append(
                {
                    "symbol": task.symbol,
                    "start_date": task.start_date,
                    "end_date": task.end_date,
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:500],
                }
            )
            client.reset()
    if owned_provider:
        client.close()
    frame, unresolved = _validate_intraday_frame(
        result_data,
        {task.symbol: task.dates},
        source_name="baostock",
        detailed_reasons=True,
    )
    if not frame.empty or not result_data.empty:
        return frame, unresolved, errors
    reason = "provider_error" if errors else "provider_empty"
    unresolved = {(task.symbol, trade_date): reason for trade_date in task.dates}
    return frame, unresolved, errors
