"""Resumable, month-at-a-time minute-v2 dataset construction."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from quantlab.core.io import read_json, sha256_file, write_json
from quantlab.data.qdp_v2.duckdb_resources import GIB, MIB, open_guarded_duckdb

from .contracts import EXPECTED_DECISION_BARS, MinuteV2Config, MinuteV2Error
from .features import feature_query
from .labels import calendar_query, label_query, label_stock_day_query
from .sampling import event_query
from .source import (
    SourceSnapshot,
    materialize_stock_days,
    month_bounds,
    register_source_views,
    resolve_source_snapshot,
)

FORBIDDEN_FEATURE_PREFIXES = ("label_", "actual_exit", "planned_exit", "entry_")
CHECKPOINT_SCHEMA = "quantlab.minute_v2_day_parts/1"


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _artifact(path: Path) -> dict[str, Any]:
    metadata = pq.ParquetFile(path).metadata
    return {
        "path": str(path),
        "size": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "row_count": int(metadata.num_rows),
        "row_group_count": int(metadata.num_row_groups),
    }


def _copy_query(connection: Any, query: str, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    connection.execute(
        f"COPY ({query}) TO {_sql_literal(partial)} "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)"
    )
    if not partial.is_file():
        raise MinuteV2Error(f"minute_v2_copy_missing:{partial}")
    os.replace(partial, target)
    return _artifact(target)


def _parquet_scan(paths: list[Path]) -> str:
    if not paths:
        raise MinuteV2Error("minute_v2_parquet_parts_empty")
    values = ",".join(_sql_literal(path) for path in paths)
    return f"read_parquet([{values}], union_by_name=true)"


def _calendar_extension(
    snapshot: SourceSnapshot,
    *,
    end_date: str,
    future_open_days: int,
) -> str:
    frames = [pd.read_parquet(path, columns=["trade_date", "is_open"]) for path in snapshot.shard_paths["trading_calendar"]]
    calendar = pd.concat(frames, ignore_index=True)
    values = sorted(
        calendar.loc[
            calendar["is_open"].fillna(False)
            & calendar["trade_date"].astype(str).gt(str(end_date)),
            "trade_date",
        ]
        .astype(str)
        .unique()
        .tolist()
    )
    if len(values) < int(future_open_days):
        raise MinuteV2Error(
            f"minute_v2_future_calendar_incomplete:{end_date}:{len(values)}:{future_open_days}"
        )
    return str(values[int(future_open_days) - 1])


def _month_directory(output_root: Path, year: int, month: int) -> Path:
    return output_root / "months" / f"year={int(year):04d}" / f"month={int(month):02d}"


def _manifest_is_complete(path: Path, *, keep_base: bool) -> bool:
    if not path.is_file():
        return False
    value = read_json(path)
    if value.get("status") != "ok":
        return False
    required = ["events", "labels"] + (["base"] if keep_base else [])
    artifacts = dict(value.get("artifacts", {}))
    for name in required:
        record = dict(artifacts.get(name, {}))
        target = Path(str(record.get("path", "")))
        if not target.is_file() or int(record.get("size", -1)) != target.stat().st_size:
            return False
    return True


def _safe_clean_generated(directory: Path, month_directory: Path, *, expected_name: str) -> None:
    if not directory.exists():
        return
    resolved = directory.resolve()
    parent = month_directory.resolve()
    if resolved.parent != parent or resolved.name != expected_name:
        raise MinuteV2Error(f"minute_v2_generated_cleanup_refused:{resolved}")
    shutil.rmtree(resolved)


def _checkpoint_spec(
    *,
    snapshot: SourceSnapshot,
    config: MinuteV2Config,
    year: int,
    month: int,
    keep_base: bool,
    extended_end: str,
) -> dict[str, Any]:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "source_dataset_ids": snapshot.dataset_ids,
        "config": config.as_dict(),
        "year": int(year),
        "month": int(month),
        "keep_base": bool(keep_base),
        "extended_end": str(extended_end),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "signature": hashlib.sha256(encoded).hexdigest()}


def _prepare_checkpoint(
    directory: Path,
    *,
    spec: dict[str, Any],
    force: bool,
) -> tuple[Path, bool]:
    checkpoint = directory / "_parts"
    spec_path = checkpoint / "checkpoint.json"
    resumed = False
    if checkpoint.exists():
        current: dict[str, Any] | None = None
        if spec_path.is_file():
            try:
                current = read_json(spec_path)
            except (OSError, ValueError, TypeError):
                current = None
        if force or current != spec:
            _safe_clean_generated(checkpoint, directory, expected_name="_parts")
        else:
            resumed = True
    checkpoint.mkdir(parents=True, exist_ok=True)
    for name in ("base", "events", "labels"):
        (checkpoint / name).mkdir(parents=True, exist_ok=True)
    write_json(spec_path, spec)
    return checkpoint, resumed


def _valid_part(path: Path, *, expected_rows: int | None = None) -> bool:
    if not path.is_file():
        return False
    try:
        rows = int(pq.ParquetFile(path).metadata.num_rows)
    except (OSError, ValueError):
        return False
    return expected_rows is None or rows == int(expected_rows)


def _date_text(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _chunks(values: list[str], size: int) -> list[list[str]]:
    width = max(1, int(size))
    return [values[index : index + width] for index in range(0, len(values), width)]


def _evenly_spaced_dates(values: list[str], maximum: int) -> list[str]:
    limit = int(maximum)
    if limit <= 0 or len(values) <= limit:
        return list(values)
    indices = [((position + 1) * len(values)) // (limit + 1) for position in range(limit)]
    selected = [values[index] for index in dict.fromkeys(indices)]
    if len(selected) < limit:
        selected_set = set(selected)
        selected.extend(value for value in values if value not in selected_set)
    return sorted(selected[:limit])


def _create_feature_view(connection: Any) -> None:
    connection.execute("CREATE OR REPLACE TEMP VIEW minute_features AS " + feature_query())


def _materialize_label_support(
    connection: Any,
    *,
    start_date: str,
    extended_end_date: str,
) -> None:
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE label_stock_days AS "
        + label_stock_day_query(start_date=start_date, end_date=extended_end_date)
    )
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE calendar_dates AS "
        + calendar_query(start_date=start_date, end_date=extended_end_date)
    )


def _verify_month_artifacts(
    connection: Any,
    *,
    base: Path | None,
    events: Path,
    labels: Path,
    expected_decision_rows: int,
) -> dict[str, Any]:
    event_schema = {field.name for field in pq.ParquetFile(events).schema_arrow}
    forbidden = sorted(
        name
        for name in event_schema
        if any(name.startswith(prefix) for prefix in FORBIDDEN_FEATURE_PREFIXES)
    )
    if forbidden:
        raise MinuteV2Error(f"minute_v2_future_columns_in_features:{','.join(forbidden)}")
    event_rows = int(pq.ParquetFile(events).metadata.num_rows)
    label_rows = int(pq.ParquetFile(labels).metadata.num_rows)
    if event_rows != label_rows:
        raise MinuteV2Error(f"minute_v2_event_label_row_mismatch:{event_rows}:{label_rows}")
    if event_rows > int(expected_decision_rows):
        raise MinuteV2Error(
            f"minute_v2_event_rows_exceed_decisions:{event_rows}:{expected_decision_rows}"
        )
    event_scan = _parquet_scan([events])
    label_scan = _parquet_scan([labels])
    duplicate_events = int(
        connection.execute(
            "SELECT count(*) FROM (SELECT symbol,trade_date,bar_time,count(*) c "
            f"FROM {event_scan} GROUP BY ALL HAVING c>1)"
        ).fetchone()[0]
    )
    duplicate_labels = int(
        connection.execute(
            "SELECT count(*) FROM (SELECT symbol,trade_date,bar_time,count(*) c "
            f"FROM {label_scan} GROUP BY ALL HAVING c>1)"
        ).fetchone()[0]
    )
    if duplicate_events or duplicate_labels:
        raise MinuteV2Error(
            f"minute_v2_artifact_duplicate_keys:{duplicate_events}:{duplicate_labels}"
        )
    label_stats = connection.execute(
        "SELECT count(*) FILTER(WHERE label_observed),"
        "count(*) FILTER(WHERE entry_executable),min(label_net_return),max(label_net_return) "
        f"FROM {label_scan}"
    ).fetchone()
    base_rows = int(pq.ParquetFile(base).metadata.num_rows) if base is not None else None
    if base_rows is not None and base_rows != int(expected_decision_rows):
        raise MinuteV2Error(f"minute_v2_base_row_mismatch:{base_rows}:{expected_decision_rows}")
    return {
        "expected_decision_rows": int(expected_decision_rows),
        "base_rows": base_rows,
        "event_rows": event_rows,
        "label_rows": label_rows,
        "duplicate_event_keys": duplicate_events,
        "duplicate_label_keys": duplicate_labels,
        "observed_label_rows": int(label_stats[0] or 0),
        "executable_entry_rows": int(label_stats[1] or 0),
        "minimum_net_label": float(label_stats[2]) if label_stats[2] is not None else None,
        "maximum_net_label": float(label_stats[3]) if label_stats[3] is not None else None,
        "feature_future_column_count": 0,
    }


def build_month(
    workspace_root: str | Path,
    *,
    year: int,
    month: int,
    output_root: str | Path | None = None,
    config: MinuteV2Config | None = None,
    keep_base: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    current = config or MinuteV2Config()
    current.validate()
    workspace = Path(workspace_root).resolve()
    output = (
        Path(output_root).resolve()
        if output_root is not None
        else (workspace / "data" / "research" / "minute_v2").resolve()
    )
    directory = _month_directory(output, year, month)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if not force and _manifest_is_complete(manifest_path, keep_base=keep_base):
        return read_json(manifest_path)
    start_date, end_date = month_bounds(year, month)
    snapshot = resolve_source_snapshot(workspace)
    extended_end = _calendar_extension(
        snapshot,
        end_date=end_date,
        future_open_days=current.maximum_delayed_exit_days + 2,
    )
    paths = {
        "events": directory / "events.parquet",
        "labels": directory / "labels.parquet",
    }
    if keep_base:
        paths["base"] = directory / "base.parquet"
    runtime = directory / "_runtime"
    _safe_clean_generated(runtime, directory, expected_name="_runtime")
    runtime.mkdir(parents=True, exist_ok=True)
    checkpoint_spec = _checkpoint_spec(
        snapshot=snapshot,
        config=current,
        year=year,
        month=month,
        keep_base=keep_base,
        extended_end=extended_end,
    )
    checkpoint, resumed_checkpoint = _prepare_checkpoint(
        directory,
        spec=checkpoint_spec,
        force=force,
    )
    connection = open_guarded_duckdb(
        ":memory:",
        temp_directory=runtime,
        threads=current.duckdb_threads,
        floor_bytes=int(current.memory_floor_gib * GIB),
        minimum_limit_bytes=256 * MIB,
    )
    completed = False
    try:
        register_source_views(
            connection,
            snapshot,
            start_date=start_date,
            end_date=end_date,
            minute_end_date=extended_end,
        )
        stock_day_stats = materialize_stock_days(
            connection,
            start_date=start_date,
            end_date=end_date,
            config=current,
        )
        _materialize_label_support(
            connection,
            start_date=start_date,
            extended_end_date=extended_end,
        )
        available_dates = [
            _date_text(row[0])
            for row in connection.execute(
                "SELECT DISTINCT trade_date FROM stock_days ORDER BY trade_date"
            ).fetchall()
        ]
        if not available_dates:
            raise MinuteV2Error(f"minute_v2_trading_dates_empty:{start_date}:{end_date}")
        dates = _evenly_spaced_dates(
            available_dates,
            current.maximum_trading_days_per_month,
        )
        complete_sessions = 0
        expected_decision_rows = 0
        reused_parts = 0
        built_parts = 0
        processing_chunks = _chunks(
            dates,
            1 if keep_base else current.processing_days_per_chunk,
        )
        for chunk_dates in processing_chunks:
            date_values = ",".join(_sql_literal(value) for value in chunk_dates)
            chunk_name = (
                chunk_dates[0].replace("-", "")
                if len(chunk_dates) == 1
                else chunk_dates[0].replace("-", "") + "_" + chunk_dates[-1].replace("-", "")
            )
            connection.execute(
                "CREATE OR REPLACE TEMP VIEW day_stock_days AS "
                f"SELECT * FROM stock_days WHERE trade_date IN ({date_values})"
            )
            connection.execute(
                "CREATE OR REPLACE TEMP VIEW day_minute_bars AS "
                f"SELECT * FROM minute_bars WHERE trade_date IN ({date_values})"
            )
            day_sessions = int(
                connection.execute(
                    "SELECT count(*) FROM ("
                    "SELECT b.symbol,b.trade_date,count(*) bars FROM day_minute_bars b "
                    "JOIN day_stock_days s USING(symbol,trade_date) "
                    "GROUP BY b.symbol,b.trade_date HAVING bars=240)"
                ).fetchone()[0]
            )
            day_decisions = day_sessions * EXPECTED_DECISION_BARS
            complete_sessions += day_sessions
            expected_decision_rows += day_decisions
            stem = chunk_name + ".parquet"
            base_part = checkpoint / "base" / stem
            event_part = checkpoint / "events" / stem
            label_part = checkpoint / "labels" / stem
            need_base = keep_base and not _valid_part(base_part, expected_rows=day_decisions)
            need_events = not _valid_part(event_part)
            need_labels = not _valid_part(label_part)
            if not (need_base or need_events or need_labels):
                reused_parts += 3 if keep_base else 2
                continue
            if need_base:
                _copy_query(
                    connection,
                    feature_query(
                        bars_view="day_minute_bars",
                        stock_days_view="day_stock_days",
                    ),
                    base_part,
                )
                if not _valid_part(base_part, expected_rows=day_decisions):
                    raise MinuteV2Error(
                        f"minute_v2_chunk_base_row_mismatch:{chunk_name}:{day_decisions}"
                    )
                built_parts += 1
            elif keep_base:
                reused_parts += 1
            if need_events:
                if keep_base:
                    feature_source = _parquet_scan([base_part])
                else:
                    feature_source = "(" + feature_query(
                        bars_view="day_minute_bars",
                        stock_days_view="day_stock_days",
                    ) + ")"
                connection.execute(
                    "CREATE OR REPLACE TEMP VIEW day_minute_features AS SELECT * FROM "
                    + feature_source
                )
                _copy_query(
                    connection,
                    event_query(feature_view="day_minute_features", config=current),
                    event_part,
                )
                built_parts += 1
            else:
                reused_parts += 1
            if need_labels:
                connection.execute(
                    "CREATE OR REPLACE TEMP VIEW day_minute_events AS SELECT * FROM "
                    + _parquet_scan([event_part])
                )
                _copy_query(
                    connection,
                    label_query(
                        event_view="day_minute_events",
                        target_bars_view="day_minute_bars",
                        extended_bars_view="minute_bars_extended",
                        stock_days_view="label_stock_days",
                        calendar_view="calendar_dates",
                        config=current,
                    ),
                    label_part,
                )
                built_parts += 1
            else:
                reused_parts += 1
            event_rows = int(pq.ParquetFile(event_part).metadata.num_rows)
            label_rows = int(pq.ParquetFile(label_part).metadata.num_rows)
            if event_rows != label_rows:
                raise MinuteV2Error(
                    f"minute_v2_chunk_event_label_mismatch:{chunk_name}:{event_rows}:{label_rows}"
                )
        event_parts = sorted((checkpoint / "events").glob("*.parquet"))
        label_parts = sorted((checkpoint / "labels").glob("*.parquet"))
        if len(event_parts) != len(processing_chunks) or len(label_parts) != len(processing_chunks):
            raise MinuteV2Error(
                "minute_v2_chunk_parts_incomplete:"
                f"{len(processing_chunks)}:{len(event_parts)}:{len(label_parts)}"
            )
        artifacts: dict[str, dict[str, Any]] = {}
        if keep_base:
            base_parts = sorted((checkpoint / "base").glob("*.parquet"))
            if len(base_parts) != len(processing_chunks):
                raise MinuteV2Error(
                    "minute_v2_chunk_base_parts_incomplete:"
                    f"{len(processing_chunks)}:{len(base_parts)}"
                )
            artifacts["base"] = _copy_query(
                connection,
                f"SELECT * FROM {_parquet_scan(base_parts)}",
                paths["base"],
            )
        artifacts["events"] = _copy_query(
            connection,
            f"SELECT * FROM {_parquet_scan(event_parts)}",
            paths["events"],
        )
        artifacts["labels"] = _copy_query(
            connection,
            f"SELECT * FROM {_parquet_scan(label_parts)}",
            paths["labels"],
        )
        verification = _verify_month_artifacts(
            connection,
            base=paths.get("base"),
            events=paths["events"],
            labels=paths["labels"],
            expected_decision_rows=expected_decision_rows,
        )
        result = {
            "schema": "quantlab.minute_v2_month/1",
            "status": "ok",
            "year": int(year),
            "month": int(month),
            "start_date": start_date,
            "end_date": end_date,
            "extended_label_end_date": extended_end,
            "config": current.as_dict(),
            "source": snapshot.as_dict(),
            "stock_days": stock_day_stats,
            "complete_session_stock_days": complete_sessions,
            "date_selection": {
                "available_trading_days": len(available_dates),
                "selected_trading_days": len(dates),
                "selected_dates": dates,
                "policy": (
                    "all trading days"
                    if len(dates) == len(available_dates)
                    else "evenly spaced whole trading days for development; full market retained"
                ),
            },
            "storage_policy": (
                "complete causal base retained for audit"
                if keep_base
                else "complete causal base computed transiently; deterministic event rows retained"
            ),
            "resource_settings": asdict(connection.settings),
            "day_part_build": {
                "trading_days": len(dates),
                "processing_chunks": len(processing_chunks),
                "days_per_chunk": 1 if keep_base else current.processing_days_per_chunk,
                "resumed_checkpoint": resumed_checkpoint,
                "reused_parts": reused_parts,
                "built_parts": built_parts,
            },
            "artifacts": artifacts,
            "verification": verification,
        }
        write_json(manifest_path, result)
        completed = True
    finally:
        connection.close()
        _safe_clean_generated(runtime, directory, expected_name="_runtime")
        if completed:
            _safe_clean_generated(checkpoint, directory, expected_name="_parts")
    return result


def _month_sequence(start_year: int, end_year: int) -> list[tuple[int, int]]:
    if int(end_year) < int(start_year):
        raise MinuteV2Error("minute_v2_year_range_invalid")
    return [(year, month) for year in range(int(start_year), int(end_year) + 1) for month in range(1, 13)]


def build_range(
    workspace_root: str | Path,
    *,
    start_year: int,
    end_year: int,
    output_root: str | Path | None = None,
    config: MinuteV2Config | None = None,
    force: bool = False,
) -> dict[str, Any]:
    current = config or MinuteV2Config()
    workspace = Path(workspace_root).resolve()
    output = (
        Path(output_root).resolve()
        if output_root is not None
        else (workspace / "data" / "research" / "minute_v2").resolve()
    )
    results: list[dict[str, Any]] = []
    for year, month in _month_sequence(start_year, end_year):
        results.append(
            build_month(
                workspace,
                year=year,
                month=month,
                output_root=output,
                config=current,
                keep_base=False,
                force=force,
            )
        )
    def total(key: str) -> int:
        return sum(int(result["verification"][key] or 0) for result in results)

    aggregate = {
        "schema": "quantlab.minute_v2_dataset/1",
        "status": "ok",
        "start_year": int(start_year),
        "end_year": int(end_year),
        "month_count": int(len(results)),
        "config": current.as_dict(),
        "expected_decision_rows": total("expected_decision_rows"),
        "event_rows": total("event_rows"),
        "label_rows": total("label_rows"),
        "observed_label_rows": total("observed_label_rows"),
        "months": [
            {
                "year": result["year"],
                "month": result["month"],
                "manifest": str(
                    _month_directory(output, int(result["year"]), int(result["month"])) / "manifest.json"
                ),
            }
            for result in results
        ],
    }
    write_json(output / "manifest.json", aggregate)
    return aggregate


def verify_month(manifest_path: str | Path) -> dict[str, Any]:
    manifest = read_json(Path(manifest_path))
    if manifest.get("status") != "ok":
        raise MinuteV2Error("minute_v2_manifest_not_ok")
    checked: dict[str, Any] = {}
    for name, record_value in dict(manifest.get("artifacts", {})).items():
        record = dict(record_value)
        path = Path(str(record.get("path", "")))
        if not path.is_file():
            raise MinuteV2Error(f"minute_v2_verify_artifact_missing:{name}:{path}")
        current = _artifact(path)
        if current["size"] != int(record.get("size", -1)) or current["sha256"] != record.get("sha256"):
            raise MinuteV2Error(f"minute_v2_verify_artifact_changed:{name}:{path}")
        checked[name] = current
    return {"status": "ok", "manifest": str(Path(manifest_path).resolve()), "artifacts": checked}


def verify_dataset(manifest_path: str | Path) -> dict[str, Any]:
    dataset_manifest_path = Path(manifest_path).resolve()
    manifest = read_json(dataset_manifest_path)
    if manifest.get("status") != "ok" or manifest.get("schema") != "quantlab.minute_v2_dataset/1":
        raise MinuteV2Error("minute_v2_dataset_manifest_not_ok")
    months = list(manifest.get("months", []))
    if len(months) != int(manifest.get("month_count", -1)):
        raise MinuteV2Error("minute_v2_dataset_month_count_mismatch")
    totals = {
        "expected_decision_rows": 0,
        "event_rows": 0,
        "label_rows": 0,
        "observed_label_rows": 0,
    }
    artifact_count = 0
    artifact_bytes = 0
    selected_dates = 0
    for record in months:
        month_manifest_path = Path(str(record["manifest"])).resolve()
        month_manifest = read_json(month_manifest_path)
        checked = verify_month(month_manifest_path)
        artifact_count += len(checked["artifacts"])
        artifact_bytes += sum(int(value["size"]) for value in checked["artifacts"].values())
        verification = dict(month_manifest["verification"])
        for name in totals:
            totals[name] += int(verification[name] or 0)
        selected_dates += int(dict(month_manifest.get("date_selection", {})).get("selected_trading_days", 0))
    for name, value in totals.items():
        if value != int(manifest.get(name, -1)):
            raise MinuteV2Error(f"minute_v2_dataset_total_mismatch:{name}:{value}")
    return {
        "schema": "quantlab.minute_v2_dataset_verification/1",
        "status": "ok",
        "manifest": str(dataset_manifest_path),
        "month_count": len(months),
        "selected_trading_days": selected_dates,
        "artifact_count": artifact_count,
        "artifact_bytes": artifact_bytes,
        **totals,
    }


__all__ = ["build_month", "build_range", "verify_dataset", "verify_month"]
