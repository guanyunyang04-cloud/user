"""Resumable, month-at-a-time minute-v2 dataset construction."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from quantlab.core.io import read_json, sha256_file, write_json
from quantlab.data.qdp_v2.duckdb_resources import GIB, MIB, open_guarded_duckdb

from .contracts import (
    EXPECTED_DECISION_BARS,
    MAXIMUM_DAILY_LABEL_HORIZON,
    MODEL_FEATURE_COLUMNS,
    MinuteV2Config,
    MinuteV2Error,
)
from .features import feature_query, sixty_state_query
from .labels import calendar_query, label_query, label_stock_day_query
from .sampling import (
    CANDIDATE_COLUMNS,
    EVENT_CONTEXT_COLUMNS,
    EVENT_FLAG_COLUMNS,
    event_query,
)
from .source import (
    SourceSnapshot,
    month_bounds,
    prior_open_date,
    register_source_views,
    resolve_source_snapshot,
    stock_day_query,
)

FORBIDDEN_FEATURE_PREFIXES = ("label_", "actual_exit", "planned_exit", "entry_")
CHECKPOINT_SCHEMA = "quantlab.minute_v2_day_parts/4"
LABEL_CACHE_SCHEMA = "quantlab.minute_v2_all_event_labels/1"
MONTH_SCHEMA = "quantlab.minute_v2_month/2"
SUPPORT_CACHE_SCHEMA = "quantlab.minute_v2_support_cache/1"
LABEL_BUCKET_COUNT = 16
RUNTIME_ONLY_CONFIG_FIELDS = frozenset(
    {
        "duckdb_threads",
        "memory_floor_gib",
        "duckdb_memory_limit_gib",
    }
)


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


def _parquet_uncompressed_bytes(path: Path) -> int:
    metadata = pq.ParquetFile(path).metadata
    return sum(
        int(metadata.row_group(group).column(column).total_uncompressed_size)
        for group in range(metadata.num_row_groups)
        for column in range(metadata.num_columns)
    )


def _copy_query(
    connection: Any,
    query: str,
    target: Path,
    *,
    compression: str = "ZSTD",
) -> dict[str, Any]:
    codec = str(compression).upper()
    if codec not in {"SNAPPY", "ZSTD"}:
        raise MinuteV2Error(f"minute_v2_parquet_compression_invalid:{codec}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    completed = False
    try:
        connection.execute(
            f"COPY ({query}) TO {_sql_literal(partial)} "
            f"(FORMAT PARQUET, COMPRESSION {codec}, ROW_GROUP_SIZE 50000)"
        )
        if not partial.is_file():
            raise MinuteV2Error(f"minute_v2_copy_missing:{partial}")
        completed = True
    finally:
        if not completed and partial.exists():
            partial.unlink()
    os.replace(partial, target)
    return _artifact(target)


def _combine_parts(connection: Any, parts: list[Path], target: Path) -> dict[str, Any]:
    if not parts:
        raise MinuteV2Error("minute_v2_parquet_parts_empty")
    # Intermediate chunks use Snappy for faster writes. Re-encode the final
    # artifact consistently so one-day development builds follow the same
    # storage contract as multi-day production builds.
    return _copy_query(connection, f"SELECT * FROM {_parquet_scan(parts)}", target)


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
    if value.get("status") != "ok" or value.get("schema") != MONTH_SCHEMA:
        return False
    required = ["events", "labels"] + (["base"] if keep_base else [])
    artifacts = dict(value.get("artifacts", {}))
    for name in required:
        record = dict(artifacts.get(name, {}))
        target = Path(str(record.get("path", "")))
        if not target.is_file() or int(record.get("size", -1)) != target.stat().st_size:
            return False
    return True


def _protect_existing_manifest(path: Path, *, force: bool) -> None:
    """Refuse to overwrite a completed artifact from an older contract."""

    if not path.is_file() or force:
        return
    try:
        value = read_json(path)
    except (OSError, ValueError, TypeError) as exc:
        raise MinuteV2Error(f"minute_v2_existing_manifest_unreadable:{path}") from exc
    schema = value.get("schema")
    if schema != MONTH_SCHEMA:
        raise MinuteV2Error(
            "minute_v2_existing_manifest_schema_requires_force:"
            f"{path}:{schema!r}:{MONTH_SCHEMA}"
        )


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
    content_payload = dict(payload)
    content_config = dict(content_payload["config"])
    for name in RUNTIME_ONLY_CONFIG_FIELDS:
        content_config.pop(name, None)
    content_payload["config"] = content_config
    content_encoded = json.dumps(
        content_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        **payload,
        "content_signature": hashlib.sha256(content_encoded).hexdigest(),
        "signature": hashlib.sha256(encoded).hexdigest(),
    }


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
        if force or not _checkpoint_specs_compatible(current, spec):
            _safe_clean_generated(checkpoint, directory, expected_name="_parts")
        else:
            resumed = True
    checkpoint.mkdir(parents=True, exist_ok=True)
    for name in ("base", "events", "labels", "support"):
        (checkpoint / name).mkdir(parents=True, exist_ok=True)
    write_json(spec_path, spec)
    return checkpoint, resumed


def _checkpoint_specs_compatible(
    current: dict[str, Any] | None,
    requested: dict[str, Any],
) -> bool:
    """Allow resource tuning without invalidating deterministic day parts."""

    if current is None:
        return False
    if current == requested:
        return True
    for name in (
        "schema",
        "source_dataset_ids",
        "year",
        "month",
        "keep_base",
        "extended_end",
    ):
        if current.get(name) != requested.get(name):
            return False
    current_config = dict(current.get("config", {}))
    requested_config = dict(requested.get("config", {}))
    for name in RUNTIME_ONLY_CONFIG_FIELDS:
        current_config.pop(name, None)
        requested_config.pop(name, None)
    return current_config == requested_config


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


def _materialize_cached_view(
    connection: Any,
    *,
    view_name: str,
    path: Path,
    query: str,
    metadata_path: Path | None = None,
    cache_spec: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    expected_metadata: dict[str, Any] | None = None
    if metadata_path is not None:
        expected_metadata = {
            "schema": SUPPORT_CACHE_SCHEMA,
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "spec": dict(cache_spec or {}),
        }
    reused = _valid_part(path)
    if reused and expected_metadata is not None:
        try:
            reused = metadata_path.is_file() and read_json(metadata_path) == expected_metadata
        except (OSError, ValueError, TypeError):
            reused = False
    if not reused:
        _copy_query(connection, query, path, compression="SNAPPY")
        if metadata_path is not None and expected_metadata is not None:
            write_json(metadata_path, expected_metadata)
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW {view_name} AS SELECT * FROM "
        + _parquet_scan([path])
    )
    return _artifact(path), reused


def _support_cache_signature(
    snapshot: SourceSnapshot,
    config: MinuteV2Config,
) -> str:
    """Return a source/config key shared by all month builds.

    Quality exclusions live outside QDP dataset manifests, so include their
    file metadata as part of the key.  A changed quality file consequently
    creates a new cache namespace instead of reusing stale support rows.
    """

    quality_files: list[dict[str, Any]] = []
    if snapshot.minute_quality_root.is_dir():
        for path in sorted(snapshot.minute_quality_root.rglob("*.parquet")):
            stat = path.stat()
            quality_files.append(
                {
                    "path": str(path.relative_to(snapshot.minute_quality_root)),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
    config_values = config.as_dict()
    for name in RUNTIME_ONLY_CONFIG_FIELDS:
        config_values.pop(name, None)
    payload = {
        "schema": SUPPORT_CACHE_SCHEMA,
        "source_dataset_ids": snapshot.dataset_ids,
        "config": config_values,
        "quality_files": quality_files,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _materialize_label_support(
    connection: Any,
    *,
    start_date: str,
    extended_end_date: str,
    support_directory: Path,
    support_cache_context: dict[str, Any],
) -> dict[str, Any]:
    stock_day_path = support_directory / (
        f"label_stock_days__{start_date}__{extended_end_date}.parquet"
    )
    stock_day_artifact, stock_day_reused = _materialize_cached_view(
        connection,
        view_name="label_stock_days",
        path=stock_day_path,
        metadata_path=stock_day_path.with_suffix(".json"),
        cache_spec={
            **support_cache_context,
            "view": "label_stock_days",
            "start_date": start_date,
            "end_date": extended_end_date,
        },
        query=label_stock_day_query(
            start_date=start_date,
            end_date=extended_end_date,
        ),
    )
    calendar_path = support_directory / (
        f"calendar_dates__{start_date}__{extended_end_date}.parquet"
    )
    calendar_artifact, calendar_reused = _materialize_cached_view(
        connection,
        view_name="calendar_dates",
        path=calendar_path,
        metadata_path=calendar_path.with_suffix(".json"),
        cache_spec={
            **support_cache_context,
            "view": "calendar_dates",
            "start_date": start_date,
            "end_date": extended_end_date,
        },
        query=calendar_query(start_date=start_date, end_date=extended_end_date),
    )
    return {
        "label_stock_days": {**stock_day_artifact, "reused": stock_day_reused},
        "calendar_dates": {**calendar_artifact, "reused": calendar_reused},
    }


def _label_cache_is_valid(
    path: Path,
    metadata_path: Path,
    *,
    expected_rows: int,
    expected_spec: dict[str, Any],
) -> bool:
    if not _valid_part(path, expected_rows=expected_rows) or not metadata_path.is_file():
        return False
    try:
        return read_json(metadata_path) == expected_spec
    except (OSError, ValueError, TypeError):
        return False


def _assert_label_cache_matches_existing_parts(
    connection: Any,
    *,
    cache_paths: list[Path],
    existing_parts: list[Path],
) -> dict[str, int]:
    if not existing_parts:
        return {"existing_rows": 0, "cached_rows": 0, "different_rows": 0}
    cache_scan = _parquet_scan(cache_paths)
    existing_rows = 0
    cached_rows = 0
    different_rows = 0
    for existing_part in existing_parts:
        existing_scan = _parquet_scan([existing_part])
        dates = [
            str(row[0])
            for row in connection.execute(
                f"SELECT DISTINCT trade_date FROM {existing_scan} ORDER BY trade_date"
            ).fetchall()
        ]
        for trade_date in dates:
            date_sql = _sql_literal(trade_date)
            row = connection.execute(
                "WITH old_labels AS ("
                f"SELECT * FROM {existing_scan} WHERE trade_date={date_sql}),"
                "selected_cache AS ("
                f"SELECT * FROM {cache_scan} WHERE trade_date={date_sql}),"
                "differences AS ("
                "(SELECT * FROM old_labels EXCEPT ALL SELECT * FROM selected_cache) "
                "UNION ALL "
                "(SELECT * FROM selected_cache EXCEPT ALL SELECT * FROM old_labels)) "
                "SELECT "
                "(SELECT count(*) FROM old_labels),"
                "(SELECT count(*) FROM selected_cache),"
                "(SELECT count(*) FROM differences)"
            ).fetchone()
            if not row:
                raise MinuteV2Error("minute_v2_label_cache_comparison_empty")
            existing_rows += int(row[0])
            cached_rows += int(row[1])
            different_rows += int(row[2])
    result = {
        "existing_rows": existing_rows,
        "cached_rows": cached_rows,
        "different_rows": different_rows,
    }
    if result["existing_rows"] != result["cached_rows"] or result["different_rows"]:
        raise MinuteV2Error(
            "minute_v2_label_cache_not_equivalent:"
            f"{result['existing_rows']}:{result['cached_rows']}:"
            f"{result['different_rows']}"
        )
    return result


def _verify_month_artifacts(
    connection: Any,
    *,
    base: Path | None,
    events: Path,
    labels: Path,
    expected_decision_rows: int,
    expected_decision_groups: int,
    base_reference_parts: list[Path],
) -> dict[str, Any]:
    event_schema = {field.name for field in pq.ParquetFile(events).schema_arrow}
    required_event_columns = {
        "symbol",
        "trade_date",
        "bar_time",
        *EVENT_CONTEXT_COLUMNS,
        *EVENT_FLAG_COLUMNS,
        *CANDIDATE_COLUMNS,
        "candidate_reason_mask",
        "event_mask",
    }
    missing_event_columns = sorted(required_event_columns.difference(event_schema))
    if missing_event_columns:
        raise MinuteV2Error(
            "minute_v2_event_columns_missing:" + ",".join(missing_event_columns)
        )
    # A narrow event artifact must not silently grow back into a second copy of
    # the model feature matrix.  The full matrix remains in `base`.
    duplicated_model_features = sorted(
        (set(event_schema) - set(EVENT_CONTEXT_COLUMNS)).intersection(MODEL_FEATURE_COLUMNS)
    )
    if duplicated_model_features:
        raise MinuteV2Error(
            "minute_v2_event_wide_columns_present:" + ",".join(duplicated_model_features)
        )
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
    base_reference_scan = _parquet_scan(base_reference_parts)
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
    decision_groups = int(
        connection.execute(
            f"SELECT count(*) FROM (SELECT DISTINCT trade_date,bar_time FROM {event_scan})"
        ).fetchone()[0]
    )
    if decision_groups != int(expected_decision_groups):
        raise MinuteV2Error(
            f"minute_v2_candidate_group_count_mismatch:{decision_groups}:"
            f"{expected_decision_groups}"
        )
    missing_groups = int(
        connection.execute(
            "WITH expected AS (SELECT trade_date,bar_time,count(*) AS rows "
            f"FROM {base_reference_scan} GROUP BY trade_date,bar_time),"
            "actual AS (SELECT trade_date,bar_time,count(*) AS rows "
            f"FROM {event_scan} GROUP BY trade_date,bar_time) "
            "SELECT count(*) FROM expected LEFT JOIN actual USING(trade_date,bar_time) "
            "WHERE actual.rows IS NULL"
        ).fetchone()[0]
    )
    oversized_groups = int(
        connection.execute(
            "WITH expected AS (SELECT trade_date,bar_time,count(*) AS rows "
            f"FROM {base_reference_scan} GROUP BY trade_date,bar_time),"
            "actual AS (SELECT trade_date,bar_time,count(*) AS rows "
            f"FROM {event_scan} GROUP BY trade_date,bar_time) "
            "SELECT count(*) FROM actual JOIN expected USING(trade_date,bar_time) "
            "WHERE actual.rows > expected.rows"
        ).fetchone()[0]
    )
    if missing_groups or oversized_groups:
        raise MinuteV2Error(
            f"minute_v2_candidate_group_contract_invalid:{missing_groups}:{oversized_groups}"
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
        "decision_group_count": decision_groups,
        "missing_decision_group_count": missing_groups,
        "oversized_candidate_group_count": oversized_groups,
        "candidate_fraction_of_decision_rows": event_rows / int(expected_decision_rows),
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
    keep_base: bool = True,
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
    _protect_existing_manifest(manifest_path, force=force)
    if not force and _manifest_is_complete(manifest_path, keep_base=keep_base):
        return read_json(manifest_path)
    start_date, end_date = month_bounds(year, month)
    snapshot = resolve_source_snapshot(workspace)
    minute_history_start = prior_open_date(
        snapshot,
        start_date=start_date,
        open_days=current.minute_history_lookback_open_days,
    )
    extended_end = _calendar_extension(
        snapshot,
        end_date=end_date,
        future_open_days=max(
            MAXIMUM_DAILY_LABEL_HORIZON,
            current.maximum_delayed_exit_days + 1,
        ),
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
    effective_memory_limit_bytes = min(
        int(connection.settings.memory_limit_bytes),
        int(current.duckdb_memory_limit_gib * GIB),
    )
    connection.execute(f"SET memory_limit='{effective_memory_limit_bytes}B'")
    completed = False
    build_started = time.perf_counter()
    try:
        register_source_views(
            connection,
            snapshot,
            start_date=start_date,
            end_date=end_date,
            minute_history_start_date=minute_history_start,
            minute_end_date=extended_end,
        )
        support_signature = _support_cache_signature(snapshot, current)
        support_directory = output / "_support_cache" / support_signature
        support_directory.mkdir(parents=True, exist_ok=True)
        support_cache_context = {
            "support_cache_schema": SUPPORT_CACHE_SCHEMA,
            "support_signature": support_signature,
            "source_dataset_ids": snapshot.dataset_ids,
            "config": {
                name: value
                for name, value in current.as_dict().items()
                if name not in RUNTIME_ONLY_CONFIG_FIELDS
            },
        }
        stock_day_support_path = support_directory / (
            f"stock_days__{start_date}__{end_date}.parquet"
        )
        stock_day_artifact, stock_day_reused = _materialize_cached_view(
            connection,
            view_name="stock_days",
            path=stock_day_support_path,
            metadata_path=stock_day_support_path.with_suffix(".json"),
            cache_spec={
                **support_cache_context,
                "view": "stock_days",
                "start_date": start_date,
                "end_date": end_date,
            },
            query=stock_day_query(
                start_date=start_date,
                end_date=end_date,
                config=current,
            ),
        )
        stock_day_row = connection.execute(
            "SELECT count(*),count(DISTINCT symbol),count(DISTINCT trade_date) "
            "FROM stock_days"
        ).fetchone()
        if not stock_day_row or int(stock_day_row[0]) == 0:
            raise MinuteV2Error(f"minute_v2_stock_days_empty:{start_date}:{end_date}")
        stock_day_stats = {
            "stock_days": int(stock_day_row[0]),
            "symbols": int(stock_day_row[1]),
            "dates": int(stock_day_row[2]),
        }
        sixty_state_started = time.perf_counter()
        sixty_state_path = support_directory / (
            "sixty_minute_states__"
            f"{minute_history_start}__{end_date}__{start_date}__{end_date}.parquet"
        )
        sixty_state_artifact, sixty_state_reused = _materialize_cached_view(
            connection,
            view_name="sixty_minute_states",
            path=sixty_state_path,
            metadata_path=sixty_state_path.with_suffix(".json"),
            cache_spec={
                **support_cache_context,
                "view": "sixty_minute_states",
                "minute_history_start": minute_history_start,
                "start_date": start_date,
                "end_date": end_date,
                "stock_days_sha256": stock_day_artifact["sha256"],
            },
            query=sixty_state_query(
                bars_view="minute_bars_history",
                stock_days_view="stock_days",
            ),
        )
        sixty_state_seconds = time.perf_counter() - sixty_state_started
        label_support_artifacts = _materialize_label_support(
            connection,
            start_date=start_date,
            extended_end_date=extended_end,
            support_directory=support_directory,
            support_cache_context=support_cache_context,
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
        processing_chunks = _chunks(dates, current.processing_days_per_chunk)
        feature_phase_started = time.perf_counter()
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
            connection.execute(
                "CREATE OR REPLACE TEMP VIEW day_minute_history AS "
                "WITH target_symbols AS (SELECT DISTINCT symbol FROM day_stock_days),"
                "complete_prior_days AS ("
                "SELECT b.symbol,b.trade_date FROM minute_bars_history b "
                "JOIN target_symbols t ON t.symbol=b.symbol "
                f"WHERE b.trade_date < {_sql_literal(chunk_dates[0])} "
                "GROUP BY b.symbol,b.trade_date HAVING count(*)=240),"
                "prior_bars AS ("
                "SELECT b.* FROM minute_bars_history b JOIN complete_prior_days p "
                "ON p.symbol=b.symbol AND p.trade_date=b.trade_date "
                f"WHERE b.trade_date < {_sql_literal(chunk_dates[0])} "
                "QUALIFY row_number() OVER (PARTITION BY b.symbol "
                "ORDER BY b.trade_date DESC,b.bar_time DESC)<=240),"
                "target_bars AS (SELECT b.* FROM minute_bars b JOIN target_symbols t "
                f"ON t.symbol=b.symbol WHERE b.trade_date IN ({date_values})) "
                "SELECT * FROM prior_bars UNION ALL BY NAME SELECT * FROM target_bars"
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
            need_base = not _valid_part(base_part, expected_rows=day_decisions)
            if need_base:
                _copy_query(
                    connection,
                    feature_query(
                        bars_view="day_minute_history",
                        stock_days_view="day_stock_days",
                        sixty_state_view="sixty_minute_states",
                    ),
                    base_part,
                    compression="SNAPPY",
                )
                if not _valid_part(base_part, expected_rows=day_decisions):
                    raise MinuteV2Error(
                        f"minute_v2_chunk_base_row_mismatch:{chunk_name}:{day_decisions}"
                    )
                built_parts += 1
            else:
                reused_parts += 1
        base_parts = sorted((checkpoint / "base").glob("*.parquet"))
        if len(base_parts) != len(processing_chunks):
            raise MinuteV2Error(
                f"minute_v2_chunk_base_parts_incomplete:{len(processing_chunks)}:{len(base_parts)}"
            )
        feature_phase_seconds = time.perf_counter() - feature_phase_started
        base_rows = sum(int(pq.ParquetFile(path).metadata.num_rows) for path in base_parts)
        if base_rows != expected_decision_rows:
            raise MinuteV2Error(
                f"minute_v2_base_parts_row_mismatch:{base_rows}:{expected_decision_rows}"
            )
        base_scan = _parquet_scan(base_parts)
        group_stats = connection.execute(
            "SELECT sum(groups_per_day) AS groups,count(*) AS dates,min(groups_per_day),"
            "max(groups_per_day) "
            "FROM (SELECT trade_date,count(DISTINCT bar_time) AS groups_per_day "
            f"FROM {base_scan} GROUP BY trade_date)"
        ).fetchone()
        if not group_stats or min(int(value or 0) for value in group_stats) <= 0:
            raise MinuteV2Error("minute_v2_base_cross_sections_empty")
        if int(group_stats[2]) != EXPECTED_DECISION_BARS or int(group_stats[3]) != EXPECTED_DECISION_BARS:
            raise MinuteV2Error(
                f"minute_v2_decision_grid_incomplete:{group_stats[2]}:{group_stats[3]}"
            )
        uncompressed_base_bytes = sum(_parquet_uncompressed_bytes(path) for path in base_parts)
        compressed_base_bytes = sum(path.stat().st_size for path in base_parts)
        expected_decision_groups = int(group_stats[0])
        candidate_phase_started = time.perf_counter()
        # First materialize every candidate event.  The expensive future-window
        # calculation is performed once per symbol bucket below, rather than
        # once per trading-day part.
        for chunk_dates, base_part in zip(processing_chunks, base_parts, strict=True):
            chunk_name = (
                chunk_dates[0].replace("-", "")
                if len(chunk_dates) == 1
                else chunk_dates[0].replace("-", "") + "_" + chunk_dates[-1].replace("-", "")
            )
            stem = chunk_name + ".parquet"
            event_part = checkpoint / "events" / stem
            need_events = not _valid_part(event_part)
            if not need_events:
                reused_parts += 1
                continue
            connection.execute(
                "CREATE OR REPLACE TEMP VIEW day_minute_features AS SELECT * FROM "
                + _parquet_scan([base_part])
            )
            _copy_query(
                connection,
                event_query(
                    feature_view="day_minute_features",
                    config=current,
                ),
                event_part,
                compression="SNAPPY",
            )
            built_parts += 1

        event_parts = sorted((checkpoint / "events").glob("*.parquet"))
        if len(event_parts) != len(processing_chunks):
            raise MinuteV2Error(
                f"minute_v2_event_parts_incomplete:{len(processing_chunks)}:{len(event_parts)}"
            )
        total_event_rows = sum(
            int(pq.ParquetFile(path).metadata.num_rows) for path in event_parts
        )
        event_scan = _parquet_scan(event_parts)
        # Buckets are tied to this month's event keys.  Keep the persistent
        # cache namespace shared across months, but isolate each date range so
        # one month's labels cannot overwrite another month's buckets.
        label_bucket_directory = support_directory / (
            f"label_buckets__{start_date}__{extended_end}"
        )
        label_bucket_directory.mkdir(parents=True, exist_ok=True)
        label_bucket_paths: list[Path] = []
        label_bucket_specs: list[dict[str, Any]] = []
        label_cache_started = time.perf_counter()
        for bucket in range(LABEL_BUCKET_COUNT):
            bucket_name = f"labels_bucket_{bucket:02d}_of_{LABEL_BUCKET_COUNT:02d}.parquet"
            bucket_path = label_bucket_directory / bucket_name
            bucket_metadata = bucket_path.with_suffix(".json")
            bucket_event_count = int(
                connection.execute(
                    f"SELECT count(*) FROM {event_scan} "
                    f"WHERE hash(symbol)%{LABEL_BUCKET_COUNT}={bucket}"
                ).fetchone()[0]
            )
            bucket_spec = {
                "schema": LABEL_CACHE_SCHEMA,
                "content_signature": checkpoint_spec["content_signature"],
                "bucket": bucket,
                "bucket_count": LABEL_BUCKET_COUNT,
                "event_rows": bucket_event_count,
                "event_parts": [
                    {
                        "name": path.name,
                        "size": int(path.stat().st_size),
                        "sha256": sha256_file(path),
                    }
                    for path in event_parts
                ],
            }
            bucket_reused = _label_cache_is_valid(
                bucket_path,
                bucket_metadata,
                expected_rows=bucket_event_count,
                expected_spec=bucket_spec,
            )
            if not bucket_reused:
                connection.execute(
                    "CREATE OR REPLACE TEMP VIEW bucket_events AS SELECT * FROM "
                    f"{event_scan} WHERE hash(symbol)%{LABEL_BUCKET_COUNT}={bucket}"
                )
                connection.execute(
                    "CREATE OR REPLACE TEMP VIEW bucket_bars AS SELECT * FROM "
                    "minute_bars_extended "
                    f"WHERE hash(symbol)%{LABEL_BUCKET_COUNT}={bucket}"
                )
                _copy_query(
                    connection,
                    label_query(
                        event_view="bucket_events",
                        target_bars_view="minute_bars",
                        extended_bars_view="bucket_bars",
                        stock_days_view="label_stock_days",
                        calendar_view="calendar_dates",
                        config=current,
                    ),
                    bucket_path,
                    compression="SNAPPY",
                )
                if not _valid_part(bucket_path, expected_rows=bucket_event_count):
                    raise MinuteV2Error(
                        "minute_v2_label_bucket_row_mismatch:"
                        f"{bucket}:{bucket_event_count}"
                    )
                write_json(bucket_metadata, bucket_spec)
                built_parts += 1
            else:
                reused_parts += 1
            label_bucket_paths.append(bucket_path)
            label_bucket_specs.append(
                {
                    "path": str(bucket_path),
                    "rows": bucket_event_count,
                    "reused": bucket_reused,
                }
            )
        label_cache_seconds = time.perf_counter() - label_cache_started
        if sum(item["rows"] for item in label_bucket_specs) != total_event_rows:
            raise MinuteV2Error("minute_v2_label_bucket_rows_incomplete")
        label_cache_rows = sum(item["rows"] for item in label_bucket_specs)
        label_cache_bytes = sum(path.stat().st_size for path in label_bucket_paths)
        existing_label_parts = sorted((checkpoint / "labels").glob("*.parquet"))
        label_cache_comparison = _assert_label_cache_matches_existing_parts(
            connection,
            cache_paths=label_bucket_paths,
            existing_parts=existing_label_parts,
        )
        label_cache_reused = all(item["reused"] for item in label_bucket_specs)
        label_bucket_scan = _parquet_scan(label_bucket_paths)

        for chunk_dates, event_part in zip(processing_chunks, event_parts, strict=True):
            date_values = ",".join(_sql_literal(value) for value in chunk_dates)
            chunk_name = (
                chunk_dates[0].replace("-", "")
                if len(chunk_dates) == 1
                else chunk_dates[0].replace("-", "") + "_" + chunk_dates[-1].replace("-", "")
            )
            label_part = checkpoint / "labels" / f"{chunk_name}.parquet"
            if _valid_part(label_part):
                reused_parts += 1
            else:
                _copy_query(
                    connection,
                    "SELECT * FROM "
                    + label_bucket_scan
                    + f" WHERE trade_date IN ({date_values}) "
                    "ORDER BY trade_date,bar_time,symbol",
                    label_part,
                    compression="SNAPPY",
                )
                built_parts += 1
            event_rows = int(pq.ParquetFile(event_part).metadata.num_rows)
            label_rows = int(pq.ParquetFile(label_part).metadata.num_rows)
            if event_rows != label_rows:
                raise MinuteV2Error(
                    f"minute_v2_chunk_event_label_mismatch:{chunk_name}:{event_rows}:{label_rows}"
                )
        candidate_phase_seconds = time.perf_counter() - candidate_phase_started
        label_parts = sorted((checkpoint / "labels").glob("*.parquet"))
        if len(event_parts) != len(processing_chunks) or len(label_parts) != len(processing_chunks):
            raise MinuteV2Error(
                "minute_v2_chunk_parts_incomplete:"
                f"{len(processing_chunks)}:{len(event_parts)}:{len(label_parts)}"
            )
        artifacts: dict[str, dict[str, Any]] = {}
        if keep_base:
            artifacts["base"] = _combine_parts(connection, base_parts, paths["base"])
        artifacts["events"] = _combine_parts(connection, event_parts, paths["events"])
        artifacts["labels"] = _combine_parts(connection, label_parts, paths["labels"])
        verification = _verify_month_artifacts(
            connection,
            base=paths.get("base"),
            events=paths["events"],
            labels=paths["labels"],
            expected_decision_rows=expected_decision_rows,
            expected_decision_groups=expected_decision_groups,
            base_reference_parts=base_parts,
        )
        total_seconds = time.perf_counter() - build_started
        benchmark = {
            "feature_construction_seconds": feature_phase_seconds,
            "sixty_minute_state_seconds": sixty_state_seconds,
            "sixty_minute_state_rows": sixty_state_artifact["row_count"],
            "sixty_minute_state_bytes": sixty_state_artifact["size"],
            "sixty_minute_state_reused": sixty_state_reused,
            "candidate_and_label_seconds": candidate_phase_seconds,
            "all_event_label_cache_seconds": label_cache_seconds,
            "all_event_label_cache_rows": label_cache_rows,
            "all_event_label_cache_bytes": label_cache_bytes,
            "all_event_label_cache_reused": label_cache_reused,
            "all_event_label_cache_buckets": label_bucket_specs,
            "existing_label_equivalence": label_cache_comparison,
            "total_seconds": total_seconds,
            "feature_rows_per_second": (
                base_rows / feature_phase_seconds if feature_phase_seconds > 0 else None
            ),
            "base_uncompressed_bytes": uncompressed_base_bytes,
            "base_compressed_bytes": compressed_base_bytes,
            "base_parquet_compression_ratio": (
                compressed_base_bytes / uncompressed_base_bytes
                if uncompressed_base_bytes > 0
                else None
            ),
            "decision_time_policy": "all 234 causal decision minutes per complete trading day",
            "intermediate_compression": "SNAPPY for new day/support parts; final artifacts ZSTD",
        }
        result = {
            "schema": MONTH_SCHEMA,
            "status": "ok",
            "year": int(year),
            "month": int(month),
            "start_date": start_date,
            "end_date": end_date,
            "extended_label_end_date": extended_end,
            "minute_history_start_date": minute_history_start,
            "config": current.as_dict(),
            "source": snapshot.as_dict(),
            "stock_days": stock_day_stats,
            "shared_support": {
                "stock_days": {
                    "rows": stock_day_artifact["row_count"],
                    "bytes": stock_day_artifact["size"],
                    "reused": stock_day_reused,
                },
                **{
                    name: {
                        "rows": value["row_count"],
                        "bytes": value["size"],
                        "reused": value["reused"],
                    }
                    for name, value in label_support_artifacts.items()
                },
            },
            "support_cache": {
                "schema": SUPPORT_CACHE_SCHEMA,
                "signature": support_signature,
                "directory": str(support_directory),
                "persistent": True,
            },
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
                else "full-market causal base computed transiently; causal stock candidates retained"
            ),
            "benchmark": benchmark,
            "resource_settings": asdict(connection.settings),
            "effective_duckdb_memory_limit_bytes": effective_memory_limit_bytes,
            "day_part_build": {
                "trading_days": len(dates),
                "processing_chunks": len(processing_chunks),
                "days_per_chunk": current.processing_days_per_chunk,
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
    keep_base: bool = True,
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
                keep_base=keep_base,
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
