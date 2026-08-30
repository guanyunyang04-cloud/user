"""Resumable, month-at-a-time minute-v2 dataset construction."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from quantlab.core.io import DataContractError, read_json, sha256_file, write_json
from quantlab.data.qdp_v2.duckdb_resources import GIB, MIB, open_guarded_duckdb

from .artifacts import artifact_matches as _artifact_matches
from .artifacts import artifact_record as _artifact
from .contracts import (
    CORE_STORAGE_COLUMNS,
    EXPECTED_DECISION_BARS,
    MAXIMUM_DAILY_LABEL_HORIZON,
    MODEL_FEATURE_COLUMNS,
    OPTIONAL_STORAGE_COLUMNS,
    MinuteV2Config,
    MinuteV2Error,
)
from .features import feature_query, sixty_state_query
from .labels import LABEL_COLUMNS, calendar_query, label_query, label_stock_day_query
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
BUILD_SPEC_SCHEMA = "quantlab.minute_v2_build_spec/1"
# Bump this when a change can alter a generated month even if the Parquet
# column contract remains compatible.  It prevents a valid-looking old
# manifest from silently surviving a logic change.
BUILD_IMPLEMENTATION_REVISION = "2026-08-29-1"
LABEL_BUCKET_COUNT = 16
PERIOD_SUPPORT_BUCKET_COUNT = 16
RUNTIME_ONLY_CONFIG_FIELDS = frozenset(
    {
        "duckdb_threads",
        "memory_floor_gib",
        "duckdb_memory_limit_gib",
        "temp_directory",
        "query_profile_path",
    }
)
_DIRECTORY_REMOVE_RETRY_DELAYS = (0.05, 0.10, 0.20, 0.40, 0.80)

def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _unfinished_build_files(manifest_path: Path) -> list[Path]:
    """Return transient files that prove a month build did not finish cleanly."""

    month_directory = manifest_path.parent
    candidates: list[Path] = []
    candidates.extend(path for path in month_directory.rglob("*.partial") if path.is_file())
    for directory_name in ("_parts", "_runtime"):
        directory = month_directory / directory_name
        if directory.is_dir():
            candidates.extend(path for path in directory.rglob("*") if path.is_file())
    return sorted(set(candidates))


def _remove_stale_base_artifact(month_directory: Path) -> bool:
    """Remove the exact month-level base artifact when ``keep_base`` is off."""

    directory = Path(month_directory).resolve()
    target = directory / "base.parquet"
    # ``exists`` is false for a broken symlink, but unlinking that exact path
    # is still safe and prevents it from being mistaken for a clean build.
    if not target.exists() and not target.is_symlink():
        return False
    if target.is_dir():
        raise MinuteV2Error(f"minute_v2_stale_base_not_file:{target}")
    target.unlink()
    return True


def _remove_exact_file(path: Path) -> bool:
    """Remove one generated file, refusing directories and broad paths."""

    target = Path(path).resolve()
    if not target.exists() and not target.is_symlink():
        return False
    if target.is_dir():
        raise MinuteV2Error(f"minute_v2_generated_file_is_directory:{target}")
    target.unlink()
    return True


def _remove_generated_tree(path: Path) -> None:
    """Remove one authorized generated tree with bounded transient retries.

    Windows and exFAT can briefly report an emptied directory as non-empty
    while a closed Parquet handle or filesystem metadata update is settling.
    The caller remains responsible for checking the exact owned path before
    invoking this helper.
    """

    target = Path(path)
    last_error: OSError | None = None
    attempts = len(_DIRECTORY_REMOVE_RETRY_DELAYS) + 1
    for attempt in range(attempts):
        if not target.exists():
            return
        try:
            shutil.rmtree(target)
        except OSError as exc:
            if not target.exists():
                return
            transient = getattr(exc, "winerror", None) in {32, 33, 145} or getattr(
                exc, "errno", None
            ) in {16, 39}
            if not transient:
                raise
            last_error = exc
        if not target.exists():
            return
        if attempt < len(_DIRECTORY_REMOVE_RETRY_DELAYS):
            time.sleep(_DIRECTORY_REMOVE_RETRY_DELAYS[attempt])
    if last_error is not None:
        raise last_error
    raise MinuteV2Error(f"minute_v2_generated_cleanup_incomplete:{target}")


def _remove_stale_optional_artifact(month_directory: Path) -> bool:
    """Remove the exact split-feature sidecar when the requested build omits it."""

    return _remove_exact_file(Path(month_directory).resolve() / "optional_features.parquet")


def _runtime_directory(
    month_directory: Path,
    config: MinuteV2Config,
    *,
    year: int,
    month: int,
) -> tuple[Path, bool]:
    """Return an isolated spill directory and whether it is externally rooted.

    An explicit temporary root is useful when the output volume is slow or
    nearly full (for example, putting DuckDB spill on the NVMe C: volume).
    Only a uniquely named child carrying our marker is ever cleaned up.
    """

    if not config.temp_directory:
        runtime = month_directory / "_runtime"
        _safe_clean_generated(runtime, month_directory, expected_name="_runtime")
        runtime.mkdir(parents=True, exist_ok=True)
        return runtime, False
    root = Path(config.temp_directory).expanduser().resolve()
    runtime = root / "quantlab_minute_v2" / f"year={int(year):04d}" / f"month={int(month):02d}"
    marker = runtime / ".quantlab_runtime_marker"
    if runtime.exists():
        if not marker.is_file():
            raise MinuteV2Error(f"minute_v2_external_runtime_not_owned:{runtime}")
        _remove_generated_tree(runtime)
    runtime.mkdir(parents=True, exist_ok=True)
    marker.write_text("quantlab.minute_v2.runtime/1\n", encoding="utf-8")
    return runtime, True


def _clean_runtime_directory(runtime: Path, *, external: bool, month_directory: Path) -> None:
    if external:
        marker = runtime / ".quantlab_runtime_marker"
        if marker.is_file():
            _remove_generated_tree(runtime)
    else:
        _safe_clean_generated(runtime, month_directory, expected_name="_runtime")


def _effective_memory_limit_bytes(connection: Any, config: MinuteV2Config) -> int:
    """Apply an optional user cap on top of the live available-RAM limit."""

    dynamic = int(connection.settings.memory_limit_bytes)
    requested = config.duckdb_memory_limit_gib
    if isinstance(requested, str) and requested.strip().lower() == "auto":
        return dynamic
    return min(dynamic, int(float(requested) * GIB))


def _profile_path(directory: Path, config: MinuteV2Config) -> Path | None:
    value = config.query_profile_path
    if not value:
        return None
    if str(value).strip().lower() == "auto":
        return directory / "query_profile.json"
    return Path(value).expanduser().resolve()


def _quarter_support_bounds(
    snapshot: SourceSnapshot,
    *,
    year: int,
    month: int,
    config: MinuteV2Config,
) -> dict[str, str]:
    """Return fixed quarter boundaries shared by the three monthly builds."""

    quarter = (int(month) - 1) // 3
    quarter_start = date(int(year), quarter * 3 + 1, 1).isoformat()
    if quarter == 3:
        next_start = date(int(year) + 1, 1, 1)
    else:
        next_start = date(int(year), quarter * 3 + 4, 1)
    quarter_end = (next_start - timedelta(days=1)).isoformat()
    history_start = prior_open_date(
        snapshot,
        start_date=quarter_start,
        open_days=config.minute_history_lookback_open_days,
    )
    extended_end = _calendar_extension(
        snapshot,
        end_date=quarter_end,
        future_open_days=max(
            MAXIMUM_DAILY_LABEL_HORIZON,
            config.maximum_delayed_exit_days + 1,
        ),
    )
    return {
        "key": f"{int(year):04d}Q{quarter + 1}",
        "start_date": quarter_start,
        "end_date": quarter_end,
        "history_start_date": history_start,
        "extended_end_date": extended_end,
    }


def _state_support_bounds(
    snapshot: SourceSnapshot,
    *,
    period: dict[str, str],
    year: int,
    month: int,
    config: MinuteV2Config,
) -> dict[str, str]:
    """Choose the state-query interval without changing the raw cache key.

    A sampled development build only needs state rows through its selected
    month.  Keeping the raw quarter cache while narrowing this window avoids a
    large warm-up query during one-day validation; full builds use the entire
    shared period.
    """

    if int(config.maximum_trading_days_per_month) <= 0:
        return {
            "start_date": period["start_date"],
            "end_date": period["end_date"],
            "history_start_date": period["history_start_date"],
        }
    start_date, end_date = month_bounds(year, month)
    return {
        "start_date": start_date,
        "end_date": end_date,
        "history_start_date": prior_open_date(
            snapshot,
            start_date=start_date,
            open_days=config.minute_history_lookback_open_days,
        ),
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
        # The final path is changed only after the write is complete.  Keep
        # cleanup active until the atomic replacement itself succeeds.
        os.replace(partial, target)
        completed = True
    finally:
        if not completed and partial.exists():
            partial.unlink()
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


def _manifest_is_complete(
    path: Path,
    *,
    keep_base: bool,
    expected_build_spec: dict[str, Any] | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        value = read_json(path)
    except (DataContractError, OSError, ValueError, TypeError):
        return False
    if not isinstance(value, dict):
        return False
    if value.get("status") != "ok" or value.get("schema") != MONTH_SCHEMA:
        return False
    if expected_build_spec is not None and value.get("build_spec") != expected_build_spec:
        return False
    if _unfinished_build_files(path):
        return False
    feature_storage = "full"
    if isinstance(expected_build_spec, dict):
        feature_storage = str(expected_build_spec.get("feature_storage", "full"))
    elif isinstance(value.get("build_spec"), dict):
        feature_storage = str(value["build_spec"].get("feature_storage", "full"))
    required = ["events", "labels"]
    if keep_base:
        required.append("base")
        if feature_storage == "split":
            required.append("optional_features")
    raw_artifacts = value.get("artifacts")
    if not isinstance(raw_artifacts, dict):
        return False
    artifacts = raw_artifacts
    if set(artifacts) != set(required):
        return False
    for name in required:
        raw_record = artifacts.get(name)
        if not isinstance(raw_record, dict):
            return False
        record = raw_record
        target = Path(str(record.get("path", "")))
        try:
            if not target.is_absolute():
                target = (path.parent / target).resolve()
            expected_target = (path.parent / f"{name}.parquet").resolve()
            if target != expected_target or not target.is_file():
                return False
        except Exception:
            return False
        if not _artifact_matches(target, record, base_directory=path.parent):
            return False
    return True


def _protect_existing_manifest(path: Path, *, force: bool) -> None:
    """Refuse to overwrite a completed artifact from an older contract."""

    if not path.is_file() or force:
        return
    try:
        value = read_json(path)
    except (DataContractError, OSError, ValueError, TypeError) as exc:
        raise MinuteV2Error(f"minute_v2_existing_manifest_unreadable:{path}") from exc
    if not isinstance(value, dict):
        raise MinuteV2Error(f"minute_v2_existing_manifest_not_object:{path}")
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
    _remove_generated_tree(resolved)


def _checkpoint_spec(
    *,
    snapshot: SourceSnapshot,
    config: MinuteV2Config,
    year: int,
    month: int,
    keep_base: bool,
    extended_end: str,
    support_signature: str | None = None,
    source_manifest_sha256: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "build_implementation_revision": BUILD_IMPLEMENTATION_REVISION,
        "source_dataset_ids": snapshot.dataset_ids,
        "source_manifest_sha256": dict(source_manifest_sha256 or {}),
        "config": config.as_dict(),
        "year": int(year),
        "month": int(month),
        "keep_base": bool(keep_base),
        "extended_end": str(extended_end),
        "support_signature": support_signature,
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
        "build_implementation_revision",
        "source_dataset_ids",
        "source_manifest_sha256",
        "year",
        "month",
        "keep_base",
        "extended_end",
        "support_signature",
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
    except Exception:
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
            metadata = read_json(metadata_path)
            reused = (
                metadata.get("schema") == expected_metadata["schema"]
                and metadata.get("query_sha256") == expected_metadata["query_sha256"]
                and metadata.get("spec") == expected_metadata["spec"]
                and _artifact_matches(
                    path,
                    metadata.get("artifact"),
                    base_directory=path.parent,
                )
            )
        except (DataContractError, OSError, ValueError, TypeError):
            reused = False
    if not reused:
        artifact = _copy_query(connection, query, path, compression="SNAPPY")
        if metadata_path is not None and expected_metadata is not None:
            write_json(metadata_path, {**expected_metadata, "artifact": artifact})
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW {view_name} AS SELECT * FROM "
        + _parquet_scan([path])
    )
    return _artifact(path), reused


def _bucketed_artifact(paths: list[Path]) -> dict[str, Any]:
    """Summarise a deterministic set of Parquet parts as one cache artifact."""

    if not paths:
        raise MinuteV2Error("minute_v2_bucketed_cache_parts_empty")
    records = [_artifact(path) for path in paths]
    digest_payload = json.dumps(
        [
            {
                "path": record["path"],
                "size": record["size"],
                "sha256": record["sha256"],
                "row_count": record["row_count"],
                "row_group_count": record["row_group_count"],
            }
            for record in records
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        # ``path`` is retained for human-facing compatibility; consumers that
        # need to scan the cache must use ``paths`` because this is a part set.
        "path": str(paths[0]),
        "paths": [str(path) for path in paths],
        "size": sum(int(record["size"]) for record in records),
        "sha256": hashlib.sha256(digest_payload).hexdigest(),
        "row_count": sum(int(record["row_count"]) for record in records),
        "row_group_count": sum(int(record["row_group_count"]) for record in records),
        "parts": records,
    }


def _materialize_bucketed_view(
    connection: Any,
    *,
    view_name: str,
    directory: Path,
    prefix: str,
    bucket_count: int,
    query_factory: Any,
    metadata_spec: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Materialize a view in hash buckets so window queries stay bounded.

    A single quarter can contain several times the rows of the original
    one-month pilot.  Running the 60-minute window query over all symbols at
    once can exhaust RAM even when the final result is modest.  Hash buckets
    cap each window execution while retaining one reusable, logically unified
    view for downstream joins.
    """

    count = int(bucket_count)
    if count <= 0:
        raise MinuteV2Error("minute_v2_bucketed_cache_bucket_count_invalid")
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{prefix}_{bucket:02d}_of_{count:02d}.parquet" for bucket in range(count)]
    metadata_path = directory / f"{prefix}.json"
    queries = [str(query_factory(bucket)) for bucket in range(count)]
    expected_metadata = {
        "schema": SUPPORT_CACHE_SCHEMA,
        "spec": dict(metadata_spec),
        "bucket_count": count,
        "query_sha256": [
            hashlib.sha256(query.encode("utf-8")).hexdigest() for query in queries
        ],
    }
    reused = True
    metadata: Any = None
    try:
        metadata = read_json(metadata_path)
        reused = metadata == {
            **expected_metadata,
            "artifact": metadata.get("artifact"),
        }
        if reused:
            artifact = metadata.get("artifact")
            if not isinstance(artifact, dict):
                reused = False
            else:
                raw_parts = artifact.get("parts")
                if not isinstance(raw_parts, list) or len(raw_parts) != count:
                    reused = False
                else:
                    for path, record in zip(paths, raw_parts, strict=True):
                        if not _artifact_matches(path, record, base_directory=directory):
                            reused = False
                            break
    except (DataContractError, OSError, ValueError, TypeError):
        reused = False

    if not reused:
        for path in paths:
            _remove_exact_file(path)
        for bucket, query in enumerate(queries):
            part = paths[bucket]
            _copy_query(connection, query, part, compression="SNAPPY")
        artifact = _bucketed_artifact(paths)
        write_json(metadata_path, {**expected_metadata, "artifact": artifact})
    else:
        artifact = _bucketed_artifact(paths)
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW {view_name} AS SELECT * FROM "
        + _parquet_scan(paths)
    )
    return artifact, reused


def _materialize_period_raw_bars(
    connection: Any,
    *,
    period: dict[str, str],
    directory: Path,
    bucket_count: int,
    metadata_spec: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Cache the quarter's raw bars in one-pass, hash-partitioned Parquet.

    The source QDP minute domain is spread over many shards.  Reading those
    shards once and partitioning the result is substantially cheaper than
    making every state/label bucket scan all source shards independently.
    DuckDB's partition writer drops the partition expression from the stored
    schema, so downstream views retain the original nine raw-bar columns.
    """

    count = int(bucket_count)
    if count <= 0:
        raise MinuteV2Error("minute_v2_period_raw_bucket_count_invalid")
    directory.mkdir(parents=True, exist_ok=True)
    prefix = "period_raw_minute_bars"
    paths = [directory / f"{prefix}_{bucket:02d}_of_{count:02d}.parquet" for bucket in range(count)]
    metadata_path = directory / f"{prefix}.json"
    raw_query = (
        "SELECT symbol,trade_date,bar_time,open,high,low,close,volume,amount "
        "FROM minute_bars_extended "
        f"WHERE trade_date BETWEEN {_sql_literal(period['history_start_date'])} "
        f"AND {_sql_literal(period['extended_end_date'])}"
    )
    expected_metadata = {
        "schema": SUPPORT_CACHE_SCHEMA,
        "spec": dict(metadata_spec),
        "bucket_count": count,
        "query_sha256": hashlib.sha256(raw_query.encode("utf-8")).hexdigest(),
    }
    reused = False
    artifact: dict[str, Any] | None = None
    try:
        metadata = read_json(metadata_path)
        if isinstance(metadata, dict) and all(
            metadata.get(name) == value for name, value in expected_metadata.items()
        ):
            candidate = metadata.get("artifact")
            if isinstance(candidate, dict) and isinstance(candidate.get("parts"), list):
                raw_parts = candidate["parts"]
                if len(raw_parts) == count and all(
                    _artifact_matches(path, record, base_directory=directory)
                    for path, record in zip(paths, raw_parts, strict=True)
                ):
                    reused = True
                    artifact = _bucketed_artifact(paths)
    except (DataContractError, OSError, ValueError, TypeError):
        reused = False

    if not reused:
        for path in paths:
            _remove_exact_file(path)
        staging = directory / f".{prefix}_staging"
        if staging.exists():
            resolved = staging.resolve()
            if resolved.parent != directory.resolve() or resolved.name != staging.name:
                raise MinuteV2Error(f"minute_v2_period_raw_cleanup_refused:{resolved}")
            _remove_generated_tree(resolved)
        staging.mkdir(parents=True, exist_ok=True)
        completed = False
        try:
            partition_query = (
                "SELECT *, hash(symbol)%"
                f"{count} AS __minute_v2_bucket FROM ({raw_query}) raw_bars"
            )
            connection.execute(
                f"COPY ({partition_query}) TO {_sql_literal(staging)} "
                "(FORMAT PARQUET, PARTITION_BY (__minute_v2_bucket), "
                "COMPRESSION SNAPPY, PER_THREAD_OUTPUT FALSE)"
            )
            for bucket, path in enumerate(paths):
                candidates = sorted(
                    (staging / f"__minute_v2_bucket={bucket}").glob("*.parquet")
                )
                if len(candidates) == 1:
                    os.replace(candidates[0], path)
                elif not candidates:
                    # Keep a schema-bearing empty part so every bucket has a
                    # stable path and the union view remains deterministic.
                    _copy_query(
                        connection,
                        raw_query + " AND FALSE",
                        path,
                        compression="SNAPPY",
                    )
                else:
                    raise MinuteV2Error(
                        f"minute_v2_period_raw_partition_count:{bucket}:{len(candidates)}"
                    )
            completed = True
        finally:
            if staging.exists():
                _remove_generated_tree(staging)
        if not completed:
            for path in paths:
                _remove_exact_file(path)
        artifact = _bucketed_artifact(paths)
        write_json(metadata_path, {**expected_metadata, "artifact": artifact})
    if artifact is None:
        raise MinuteV2Error("minute_v2_period_raw_artifact_missing")
    connection.execute(
        "CREATE OR REPLACE TEMP VIEW period_raw_minute_bars AS SELECT * FROM "
        + _parquet_scan(paths)
    )
    return artifact, reused


def _materialize_period_support(
    connection: Any,
    *,
    snapshot: SourceSnapshot,
    year: int,
    month: int,
    period: dict[str, str],
    support_directory: Path,
    support_cache_context: dict[str, Any],
    config: MinuteV2Config,
) -> dict[str, Any]:
    """Materialize support once per calendar quarter and expose stable views.

    Monthly builds overlap roughly two months of warm-up and future bars.  The
    old cache key included each month's dates, so every month decompressed the
    same overlap again.  A fixed quarter namespace makes the first month pay
    that cost once; later months simply scan the already validated Parquet
    files.  The files are support inputs, not model outputs, and therefore do
    not alter the month key contract.
    """

    period_directory = support_directory / f"period={period['key']}"
    period_directory.mkdir(parents=True, exist_ok=True)
    state_bounds = _state_support_bounds(
        snapshot,
        period=period,
        year=year,
        month=month,
        config=config,
    )
    common = {
        **support_cache_context,
        "period": dict(period),
        "cache_grain": "calendar_quarter",
        "state_bounds": dict(state_bounds),
    }

    stock_path = period_directory / "period_stock_days.parquet"
    stock_artifact, stock_reused = _materialize_cached_view(
        connection,
        view_name="period_stock_days",
        path=stock_path,
        metadata_path=stock_path.with_suffix(".json"),
        cache_spec={**common, "view": "period_stock_days"},
        query=stock_day_query(
            start_date=period["history_start_date"],
            end_date=period["extended_end_date"],
            config=config,
        ),
    )
    period_label_path = period_directory / "period_label_stock_days.parquet"
    period_label_artifact, period_label_reused = _materialize_cached_view(
        connection,
        view_name="period_label_stock_days",
        path=period_label_path,
        metadata_path=period_label_path.with_suffix(".json"),
        cache_spec={**common, "view": "period_label_stock_days"},
        query=label_stock_day_query(
            start_date=period["start_date"],
            end_date=period["extended_end_date"],
        ),
    )
    calendar_path = period_directory / "period_calendar_dates.parquet"
    calendar_artifact, calendar_reused = _materialize_cached_view(
        connection,
        view_name="period_calendar_dates",
        path=calendar_path,
        metadata_path=calendar_path.with_suffix(".json"),
        cache_spec={**common, "view": "period_calendar_dates"},
        query=calendar_query(
            start_date=period["history_start_date"],
            end_date=period["extended_end_date"],
        ),
    )
    raw_spec = {
        **support_cache_context,
        "period": dict(period),
        "cache_grain": "calendar_quarter",
        "view": "period_raw_minute_bars",
        "bucket_count": LABEL_BUCKET_COUNT,
    }
    raw_artifact, raw_reused = _materialize_period_raw_bars(
        connection,
        period=period,
        directory=period_directory,
        bucket_count=LABEL_BUCKET_COUNT,
        metadata_spec=raw_spec,
    )
    raw_paths = [Path(str(value)).resolve() for value in raw_artifact["paths"]]
    for bucket, path in enumerate(raw_paths):
        connection.execute(
            f"CREATE OR REPLACE TEMP VIEW period_raw_minute_bars_{bucket:02d} AS SELECT * FROM "
            + _parquet_scan([path])
        )
    # A full-quarter window query can exceed the available RAM.  Restrict all
    # state inputs to one symbol bucket per execution and retain the parts for
    # later month builds.  Auxiliary views are bucketed as well so DuckDB does
    # not have to keep unrelated symbols in the join hash tables.
    state_bucket_count = (
        4 if int(config.maximum_trading_days_per_month) > 0 else PERIOD_SUPPORT_BUCKET_COUNT
    )
    state_spec = {
        **common,
        "view": "period_sixty_minute_states",
        "stock_days_sha256": stock_artifact["sha256"],
        "bucket_count": state_bucket_count,
    }
    state_start_sql = _sql_literal(state_bounds["history_start_date"])
    state_end_sql = _sql_literal(state_bounds["end_date"])

    def state_query_for_bucket(bucket: int) -> str:
        suffix = f"_{bucket:02d}"
        state_raw_view = f"period_raw_minute_bars_state{suffix}"
        if state_bucket_count == LABEL_BUCKET_COUNT:
            connection.execute(
                f"CREATE OR REPLACE TEMP VIEW {state_raw_view} AS SELECT * "
                f"FROM period_raw_minute_bars_{bucket:02d} "
                f"WHERE trade_date BETWEEN {state_start_sql} AND {state_end_sql}"
            )
        else:
            connection.execute(
                f"CREATE OR REPLACE TEMP VIEW {state_raw_view} AS SELECT * FROM "
                + _parquet_scan(
                    [
                        path
                        for index, path in enumerate(raw_paths)
                        if index % state_bucket_count == bucket
                    ]
                )
                + f" WHERE trade_date BETWEEN {state_start_sql} AND {state_end_sql}"
            )
        for source_name, bucket_name in (
            ("period_stock_days", f"period_stock_days{suffix}"),
            ("adjust_factor", f"adjust_factor{suffix}"),
            ("minute_feature_exclusions", f"minute_feature_exclusions{suffix}"),
            ("opening_auction", f"opening_auction{suffix}"),
        ):
            connection.execute(
                f"CREATE OR REPLACE TEMP VIEW {bucket_name} AS SELECT * FROM {source_name} "
                f"WHERE hash(symbol)%{state_bucket_count}={bucket} "
                f"AND trade_date BETWEEN {state_start_sql} AND {state_end_sql}"
            )
        return sixty_state_query(
            bars_view=state_raw_view,
            stock_days_view=f"period_stock_days{suffix}",
            factor_view=f"adjust_factor{suffix}",
            quality_view=f"minute_feature_exclusions{suffix}",
            auction_view=f"opening_auction{suffix}",
            ordered=False,
        )

    state_artifact, state_reused = _materialize_bucketed_view(
        connection,
        view_name="period_sixty_minute_states",
        directory=period_directory,
        prefix="period_sixty_minute_states",
        bucket_count=state_bucket_count,
        query_factory=state_query_for_bucket,
        metadata_spec=state_spec,
    )

    # The raw period cache is also the future-label cache.  Its partitions are
    # already bounded to the exact history/future interval and can be scanned
    # directly by each label bucket without a second copy on disk.
    future_artifact = {**raw_artifact, "reused": raw_reused}
    connection.execute(
        "CREATE OR REPLACE TEMP VIEW period_stock_days AS SELECT * FROM "
        + _parquet_scan([stock_path])
    )
    connection.execute(
        "CREATE OR REPLACE TEMP VIEW period_label_stock_days AS SELECT * FROM "
        + _parquet_scan([period_label_path])
    )
    connection.execute(
        "CREATE OR REPLACE TEMP VIEW period_calendar_dates AS SELECT * FROM "
        + _parquet_scan([calendar_path])
    )
    # ``_materialize_bucketed_view`` already exposes the two unified views;
    # keep these explicit replacements for clarity if its implementation is
    # later changed to defer view registration.
    connection.execute(
        "CREATE OR REPLACE TEMP VIEW period_sixty_minute_states AS SELECT * FROM "
        + _parquet_scan([Path(value) for value in state_artifact["paths"]])
    )
    connection.execute(
        "CREATE OR REPLACE TEMP VIEW future_label_bars AS SELECT * FROM "
        + _parquet_scan(raw_paths)
    )
    return {
        "period": dict(period),
        "directory": str(period_directory),
        "stock_days": {**stock_artifact, "reused": stock_reused},
        "label_stock_days": {**period_label_artifact, "reused": period_label_reused},
        "calendar_dates": {**calendar_artifact, "reused": calendar_reused},
        "raw_minute_bars": {**raw_artifact, "reused": raw_reused},
        "sixty_minute_states": {**state_artifact, "reused": state_reused},
        "future_label_bars": {**future_artifact, "reused": raw_reused},
    }


def _support_cache_signature(
    snapshot: SourceSnapshot,
    config: MinuteV2Config,
    *,
    source_manifest_sha256: dict[str, str] | None = None,
) -> str:
    """Return a source/config key shared by all month builds.

    Quality exclusions live outside QDP dataset manifests, so include their
    content hashes as part of the key.  A changed quality file consequently
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
                    "sha256": sha256_file(path),
                }
            )
    config_values = config.as_dict()
    for name in RUNTIME_ONLY_CONFIG_FIELDS:
        config_values.pop(name, None)
    payload = {
        "schema": SUPPORT_CACHE_SCHEMA,
        "source_dataset_ids": snapshot.dataset_ids,
        "source_manifest_sha256": dict(
            source_manifest_sha256 or _source_manifest_sha256(snapshot)
        ),
        "config": config_values,
        "quality_files": quality_files,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _source_manifest_sha256(snapshot: SourceSnapshot) -> dict[str, str]:
    """Hash active source manifests so mutable dataset IDs cannot hide changes."""

    return {
        name: sha256_file(Path(path))
        for name, path in sorted(snapshot.manifests.items())
    }


def _manifest_build_spec(
    *,
    checkpoint_spec: dict[str, Any],
    start_date: str,
    end_date: str,
    minute_history_start: str,
    support_signature: str,
) -> dict[str, Any]:
    """Return the stable input identity stored in a completed month manifest."""

    config = dict(checkpoint_spec.get("config", {}))
    for name in RUNTIME_ONLY_CONFIG_FIELDS:
        config.pop(name, None)
    return {
        "schema": BUILD_SPEC_SCHEMA,
        "implementation_revision": BUILD_IMPLEMENTATION_REVISION,
        "checkpoint_schema": checkpoint_spec["schema"],
        "content_signature": checkpoint_spec["content_signature"],
        "source_dataset_ids": dict(checkpoint_spec["source_dataset_ids"]),
        "source_manifest_sha256": dict(checkpoint_spec.get("source_manifest_sha256", {})),
        "config": config,
        "year": int(checkpoint_spec["year"]),
        "month": int(checkpoint_spec["month"]),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "extended_label_end_date": str(checkpoint_spec["extended_end"]),
        "minute_history_start_date": str(minute_history_start),
        "keep_base": bool(checkpoint_spec["keep_base"]),
        "feature_storage": str(config.get("feature_storage", "full")),
        "support_signature": str(support_signature),
    }


def _materialize_label_support(
    connection: Any,
    *,
    start_date: str,
    extended_end_date: str,
    support_directory: Path,
    support_cache_context: dict[str, Any],
    stock_days_view: str = "stock_days",
    calendar_view: str = "trading_calendar",
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
        query=(
            f"SELECT * FROM {stock_days_view} "
            f"WHERE trade_date BETWEEN {_sql_literal(start_date)} "
            f"AND {_sql_literal(extended_end_date)}"
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
        query=(
            f"SELECT * FROM {calendar_view} "
            f"WHERE trade_date BETWEEN {_sql_literal(start_date)} "
            f"AND {_sql_literal(extended_end_date)}"
        ),
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
        metadata = read_json(metadata_path)
        if not isinstance(metadata, dict):
            return False
        if any(metadata.get(name) != value for name, value in expected_spec.items()):
            return False
        return _artifact_matches(
            path,
            metadata.get("artifact"),
            base_directory=path.parent,
        )
    except (DataContractError, OSError, ValueError, TypeError):
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
    optional: Path | None = None,
    events: Path,
    labels: Path,
    expected_decision_rows: int,
    expected_decision_groups: int | None,
    base_reference_parts: list[Path] | None,
    feature_storage: str = "full",
) -> dict[str, Any]:
    event_parquet = pq.ParquetFile(events)
    label_parquet = pq.ParquetFile(labels)
    event_schema = {field.name for field in event_parquet.schema_arrow}
    label_schema = {field.name for field in label_parquet.schema_arrow}
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
    missing_label_columns = sorted(set(LABEL_COLUMNS).difference(label_schema))
    if missing_label_columns:
        raise MinuteV2Error(
            "minute_v2_label_columns_missing:" + ",".join(missing_label_columns)
        )
    if base is not None:
        base_schema = {field.name for field in pq.ParquetFile(base).schema_arrow}
        required_base = (
            set(MODEL_FEATURE_COLUMNS)
            if feature_storage == "full"
            else set(CORE_STORAGE_COLUMNS)
        )
        missing_base_columns = sorted(required_base.difference(base_schema))
        if missing_base_columns:
            raise MinuteV2Error(
                "minute_v2_base_columns_missing:" + ",".join(missing_base_columns)
            )
        if feature_storage == "split":
            if optional is None or not optional.is_file():
                raise MinuteV2Error("minute_v2_optional_feature_artifact_missing")
            optional_schema = set(pq.ParquetFile(optional).schema_arrow.names)
            missing_optional = sorted(set(OPTIONAL_STORAGE_COLUMNS).difference(optional_schema))
            if missing_optional:
                raise MinuteV2Error(
                    "minute_v2_optional_feature_columns_missing:" + ",".join(missing_optional)
                )
            optional_rows = int(pq.ParquetFile(optional).metadata.num_rows)
            if optional_rows != int(expected_decision_rows):
                raise MinuteV2Error(
                    f"minute_v2_optional_feature_row_mismatch:{optional_rows}:{expected_decision_rows}"
                )
            base_scan_for_optional = _parquet_scan([base])
            optional_scan = _parquet_scan([optional])
            optional_key_mismatch = int(
                connection.execute(
                    "SELECT count(*) FROM ("
                    f"(SELECT symbol,trade_date,bar_time FROM {base_scan_for_optional} "
                    f"EXCEPT ALL SELECT symbol,trade_date,bar_time FROM {optional_scan}) UNION ALL "
                    f"(SELECT symbol,trade_date,bar_time FROM {optional_scan} "
                    f"EXCEPT ALL SELECT symbol,trade_date,bar_time FROM {base_scan_for_optional})"
                    ")"
                ).fetchone()[0]
            )
            if optional_key_mismatch:
                raise MinuteV2Error(
                    f"minute_v2_optional_feature_key_mismatch:{optional_key_mismatch}"
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
    event_rows = int(event_parquet.metadata.num_rows)
    label_rows = int(label_parquet.metadata.num_rows)
    if int(expected_decision_rows) <= 0:
        raise MinuteV2Error("minute_v2_expected_decision_rows_invalid")
    if event_rows != label_rows:
        raise MinuteV2Error(f"minute_v2_event_label_row_mismatch:{event_rows}:{label_rows}")
    if event_rows > int(expected_decision_rows):
        raise MinuteV2Error(
            f"minute_v2_event_rows_exceed_decisions:{event_rows}:{expected_decision_rows}"
        )
    base_reference_scan = (
        _parquet_scan(base_reference_parts) if base_reference_parts else None
    )
    event_scan = _parquet_scan([events])
    label_scan = _parquet_scan([labels])

    def count(query: str) -> int:
        value = connection.execute(query).fetchone()
        return int(value[0]) if value and value[0] is not None else 0

    null_events = count(
        f"SELECT count(*) FROM {event_scan} "
        "WHERE symbol IS NULL OR trade_date IS NULL OR bar_time IS NULL"
    )
    null_labels = count(
        f"SELECT count(*) FROM {label_scan} "
        "WHERE symbol IS NULL OR trade_date IS NULL OR bar_time IS NULL"
    )
    duplicate_events = int(
        count(
            "SELECT count(*) FROM (SELECT symbol,trade_date,bar_time,count(*) c "
            f"FROM {event_scan} GROUP BY ALL HAVING c>1)"
        )
    )
    duplicate_labels = int(
        count(
            "SELECT count(*) FROM (SELECT symbol,trade_date,bar_time,count(*) c "
            f"FROM {label_scan} GROUP BY ALL HAVING c>1)"
        )
    )
    duplicate_base = 0
    if base_reference_scan is not None:
        duplicate_base = count(
            "SELECT count(*) FROM (SELECT symbol,trade_date,bar_time,count(*) c "
            f"FROM {base_reference_scan} GROUP BY ALL HAVING c>1)"
        )
    if null_events or null_labels:
        raise MinuteV2Error(f"minute_v2_artifact_null_keys:{null_events}:{null_labels}")
    if duplicate_events or duplicate_labels or duplicate_base:
        raise MinuteV2Error(
            "minute_v2_artifact_duplicate_keys:"
            f"{duplicate_events}:{duplicate_labels}:{duplicate_base}"
        )

    event_label_missing = count(
        f"SELECT count(*) FROM {event_scan} e LEFT JOIN {label_scan} l "
        f"USING(symbol,trade_date,bar_time) WHERE l.symbol IS NULL"
    )
    label_event_extra = count(
        f"SELECT count(*) FROM {label_scan} l LEFT JOIN {event_scan} e "
        f"USING(symbol,trade_date,bar_time) WHERE e.symbol IS NULL"
    )
    if event_label_missing or label_event_extra:
        raise MinuteV2Error(
            "minute_v2_event_label_key_contract_invalid:"
            f"{event_label_missing}:{label_event_extra}"
        )

    event_outside_base = 0
    if base_reference_scan is not None:
        event_outside_base = count(
            f"SELECT count(*) FROM {event_scan} e LEFT JOIN {base_reference_scan} b "
            f"USING(symbol,trade_date,bar_time) WHERE b.symbol IS NULL"
        )
        if event_outside_base:
            raise MinuteV2Error(
                f"minute_v2_event_keys_outside_base:{event_outside_base}"
            )
    decision_groups = int(
        connection.execute(
            f"SELECT count(*) FROM (SELECT DISTINCT trade_date,bar_time FROM {event_scan})"
        ).fetchone()[0]
    )
    if expected_decision_groups is not None and decision_groups != int(expected_decision_groups):
        raise MinuteV2Error(
            f"minute_v2_candidate_group_count_mismatch:{decision_groups}:"
            f"{expected_decision_groups}"
        )
    missing_groups = 0
    oversized_groups = 0
    if base_reference_scan is not None:
        missing_groups = count(
            "WITH expected AS (SELECT trade_date,bar_time,count(*) AS rows "
            f"FROM {base_reference_scan} GROUP BY trade_date,bar_time),"
            "actual AS (SELECT trade_date,bar_time,count(*) AS rows "
            f"FROM {event_scan} GROUP BY trade_date,bar_time) "
            "SELECT count(*) FROM expected LEFT JOIN actual USING(trade_date,bar_time) "
            "WHERE actual.rows IS NULL"
        )
        oversized_groups = count(
            "WITH expected AS (SELECT trade_date,bar_time,count(*) AS rows "
            f"FROM {base_reference_scan} GROUP BY trade_date,bar_time),"
            "actual AS (SELECT trade_date,bar_time,count(*) AS rows "
            f"FROM {event_scan} GROUP BY trade_date,bar_time) "
            "SELECT count(*) FROM actual JOIN expected USING(trade_date,bar_time) "
            "WHERE actual.rows > expected.rows"
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
        "null_event_key_rows": null_events,
        "null_label_key_rows": null_labels,
        "duplicate_event_keys": duplicate_events,
        "duplicate_label_keys": duplicate_labels,
        "duplicate_base_keys": duplicate_base,
        "event_label_missing_keys": event_label_missing,
        "label_event_extra_keys": label_event_extra,
        "event_keys_outside_base": event_outside_base,
        "missing_label_column_count": 0,
        "observed_label_rows": int(label_stats[0] or 0),
        "executable_entry_rows": int(label_stats[1] or 0),
        "minimum_net_label": float(label_stats[2]) if label_stats[2] is not None else None,
        "maximum_net_label": float(label_stats[3]) if label_stats[3] is not None else None,
        "feature_future_column_count": 0,
    }


def _chunk_stem(chunk_dates: list[str]) -> str:
    """Return the stable filename stem for one processing chunk."""

    if len(chunk_dates) == 1:
        return chunk_dates[0].replace("-", "")
    return f"{chunk_dates[0].replace('-', '')}_{chunk_dates[-1].replace('-', '')}"


@dataclass(frozen=True)
class _FeatureMaterialization:
    base_parts: list[Path]
    complete_sessions: int
    expected_decision_rows: int
    feature_phase_seconds: float
    reused_parts: int
    built_parts: int
    base_rows: int
    expected_decision_groups: int
    uncompressed_base_bytes: int
    compressed_base_bytes: int


def _materialize_feature_parts(
    connection: Any,
    *,
    checkpoint: Path,
    current: MinuteV2Config,
    processing_chunks: list[list[str]],
) -> _FeatureMaterialization:
    """Build and validate causal feature parts for the selected dates."""

    complete_sessions = 0
    expected_decision_rows = 0
    reused_parts = 0
    built_parts = 0
    feature_phase_started = time.perf_counter()
    for chunk_dates in processing_chunks:
        date_values = ",".join(_sql_literal(value) for value in chunk_dates)
        chunk_name = _chunk_stem(chunk_dates)
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
        base_part = checkpoint / "base" / f"{chunk_name}.parquet"
        if not _valid_part(base_part, expected_rows=day_decisions):
            _copy_query(
                connection,
                feature_query(
                    bars_view="day_minute_history",
                    stock_days_view="day_stock_days",
                    auction_view="opening_auction",
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
    return _FeatureMaterialization(
        base_parts=base_parts,
        complete_sessions=complete_sessions,
        expected_decision_rows=expected_decision_rows,
        feature_phase_seconds=feature_phase_seconds,
        reused_parts=reused_parts,
        built_parts=built_parts,
        base_rows=base_rows,
        expected_decision_groups=int(group_stats[0]),
        uncompressed_base_bytes=uncompressed_base_bytes,
        compressed_base_bytes=compressed_base_bytes,
    )


@dataclass(frozen=True)
class _EventLabelMaterialization:
    event_parts: list[Path]
    label_parts: list[Path]
    candidate_phase_seconds: float
    label_cache_seconds: float
    label_cache_rows: int
    label_cache_bytes: int
    label_cache_reused: bool
    label_bucket_specs: list[dict[str, Any]]
    label_cache_comparison: dict[str, Any]
    reused_parts: int
    built_parts: int


def _materialize_event_and_label_parts(
    connection: Any,
    *,
    checkpoint: Path,
    current: MinuteV2Config,
    processing_chunks: list[list[str]],
    base_parts: list[Path],
    support_directory: Path,
    start_date: str,
    extended_end: str,
    period_support: dict[str, Any],
    checkpoint_spec: dict[str, Any],
) -> _EventLabelMaterialization:
    """Materialize candidate events and their bucketed future labels."""

    reused_parts = 0
    built_parts = 0
    candidate_phase_started = time.perf_counter()
    for chunk_dates, base_part in zip(processing_chunks, base_parts, strict=True):
        chunk_name = _chunk_stem(chunk_dates)
        event_part = checkpoint / "events" / f"{chunk_name}.parquet"
        if _valid_part(event_part):
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
    total_event_rows = sum(int(pq.ParquetFile(path).metadata.num_rows) for path in event_parts)
    event_scan = _parquet_scan(event_parts)
    label_bucket_directory = support_directory / f"label_buckets__{start_date}__{extended_end}"
    label_bucket_directory.mkdir(parents=True, exist_ok=True)
    label_bucket_paths: list[Path] = []
    label_bucket_specs: list[dict[str, Any]] = []
    label_cache_started = time.perf_counter()
    future_label_paths = [Path(str(value)).resolve() for value in period_support["future_label_bars"]["paths"]]
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
                + _parquet_scan([future_label_paths[bucket]])
            )
            bucket_artifact = _copy_query(
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
            write_json(bucket_metadata, {**bucket_spec, "artifact": bucket_artifact})
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
        chunk_name = _chunk_stem(chunk_dates)
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
    label_parts = sorted((checkpoint / "labels").glob("*.parquet"))
    if len(event_parts) != len(processing_chunks) or len(label_parts) != len(processing_chunks):
        raise MinuteV2Error(
            "minute_v2_chunk_parts_incomplete:"
            f"{len(processing_chunks)}:{len(event_parts)}:{len(label_parts)}"
        )
    return _EventLabelMaterialization(
        event_parts=event_parts,
        label_parts=label_parts,
        candidate_phase_seconds=time.perf_counter() - candidate_phase_started,
        label_cache_seconds=label_cache_seconds,
        label_cache_rows=label_cache_rows,
        label_cache_bytes=label_cache_bytes,
        label_cache_reused=label_cache_reused,
        label_bucket_specs=label_bucket_specs,
        label_cache_comparison=label_cache_comparison,
        reused_parts=reused_parts,
        built_parts=built_parts,
    )


@dataclass(frozen=True)
class _MonthPreparation:
    current: MinuteV2Config
    year: int
    month: int
    workspace: Path
    output: Path
    directory: Path
    manifest_path: Path
    snapshot: SourceSnapshot
    start_date: str
    end_date: str
    minute_history_start: str
    extended_end: str
    period: dict[str, str]
    source_manifest_sha256: dict[str, str]
    support_signature: str
    checkpoint_spec: dict[str, Any]
    manifest_build_spec: dict[str, Any]
    paths: dict[str, Path]
    cached_result: dict[str, Any] | None = None


def _prepare_month(
    workspace_root: str | Path,
    *,
    year: int,
    month: int,
    output_root: str | Path | None,
    config: MinuteV2Config | None,
    keep_base: bool,
    force: bool,
) -> _MonthPreparation:
    """Resolve source identity, build signatures, and output paths."""

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
    period = _quarter_support_bounds(snapshot, year=year, month=month, config=current)
    source_manifest_sha256 = _source_manifest_sha256(snapshot)
    support_signature = _support_cache_signature(
        snapshot,
        current,
        source_manifest_sha256=source_manifest_sha256,
    )
    checkpoint_spec = _checkpoint_spec(
        snapshot=snapshot,
        config=current,
        year=year,
        month=month,
        keep_base=keep_base,
        extended_end=extended_end,
        support_signature=support_signature,
        source_manifest_sha256=source_manifest_sha256,
    )
    manifest_build_spec = _manifest_build_spec(
        checkpoint_spec=checkpoint_spec,
        start_date=start_date,
        end_date=end_date,
        minute_history_start=minute_history_start,
        support_signature=support_signature,
    )
    if not force and _manifest_is_complete(
        manifest_path,
        keep_base=keep_base,
        expected_build_spec=manifest_build_spec,
    ):
        if not keep_base:
            _remove_stale_base_artifact(directory)
        if not keep_base or current.feature_storage != "split":
            _remove_stale_optional_artifact(directory)
        cached_result = read_json(manifest_path)
    else:
        cached_result = None
        if not keep_base or current.feature_storage != "split":
            _remove_stale_optional_artifact(directory)
    paths: dict[str, Path] = {
        "events": directory / "events.parquet",
        "labels": directory / "labels.parquet",
    }
    if keep_base:
        paths["base"] = directory / "base.parquet"
        if current.feature_storage == "split":
            paths["optional_features"] = directory / "optional_features.parquet"
    return _MonthPreparation(
        current=current,
        year=int(year),
        month=int(month),
        workspace=workspace,
        output=output,
        directory=directory,
        manifest_path=manifest_path,
        snapshot=snapshot,
        start_date=start_date,
        end_date=end_date,
        minute_history_start=minute_history_start,
        extended_end=extended_end,
        period=period,
        source_manifest_sha256=source_manifest_sha256,
        support_signature=support_signature,
        checkpoint_spec=checkpoint_spec,
        manifest_build_spec=manifest_build_spec,
        paths=paths,
        cached_result=cached_result,
    )


@dataclass(frozen=True)
class _SupportMaterialization:
    support_directory: Path
    support_cache_context: dict[str, Any]
    period_support: dict[str, Any]
    stock_day_artifact: dict[str, Any]
    stock_day_reused: bool
    stock_day_stats: dict[str, int]
    sixty_state_artifact: dict[str, Any]
    sixty_state_reused: bool
    sixty_state_seconds: float
    label_support_artifacts: dict[str, Any]
    available_dates: list[str]


def _materialize_support_views(
    connection: Any,
    *,
    preparation: _MonthPreparation,
) -> _SupportMaterialization:
    """Register source data and materialize shared month support views."""

    register_source_views(
        connection,
        preparation.snapshot,
        start_date=preparation.start_date,
        end_date=preparation.end_date,
        minute_history_start_date=min(
            preparation.minute_history_start,
            preparation.period["history_start_date"],
        ),
        minute_history_end_date=preparation.period["end_date"],
        minute_end_date=preparation.period["extended_end_date"],
        minute_extended_start_date=preparation.period["history_start_date"],
    )
    support_directory = preparation.output / "_support_cache" / preparation.support_signature
    support_directory.mkdir(parents=True, exist_ok=True)
    support_cache_context = {
        "support_cache_schema": SUPPORT_CACHE_SCHEMA,
        "support_signature": preparation.support_signature,
        "source_dataset_ids": preparation.snapshot.dataset_ids,
        "config": {
            name: value
            for name, value in preparation.current.as_dict().items()
            if name not in RUNTIME_ONLY_CONFIG_FIELDS
        },
    }
    period_support = _materialize_period_support(
        connection,
        snapshot=preparation.snapshot,
        year=preparation.year,
        month=preparation.month,
        period=preparation.period,
        support_directory=support_directory,
        support_cache_context=support_cache_context,
        config=preparation.current,
    )
    connection.execute(
        "CREATE OR REPLACE TEMP VIEW stock_days AS SELECT * FROM period_stock_days "
        f"WHERE trade_date BETWEEN {_sql_literal(preparation.start_date)} "
        f"AND {_sql_literal(preparation.end_date)}"
    )
    connection.execute(
        "CREATE OR REPLACE TEMP VIEW calendar_dates AS SELECT * FROM period_calendar_dates "
        f"WHERE trade_date BETWEEN {_sql_literal(preparation.start_date)} "
        f"AND {_sql_literal(preparation.extended_end)}"
    )
    connection.execute(
        "CREATE OR REPLACE TEMP VIEW sixty_minute_states AS SELECT * FROM period_sixty_minute_states"
    )
    stock_day_support_path = support_directory / (
        f"stock_days__{preparation.start_date}__{preparation.end_date}.parquet"
    )
    stock_day_artifact, stock_day_reused = _materialize_cached_view(
        connection,
        view_name="stock_days",
        path=stock_day_support_path,
        metadata_path=stock_day_support_path.with_suffix(".json"),
        cache_spec={
            **support_cache_context,
            "view": "stock_days",
            "start_date": preparation.start_date,
            "end_date": preparation.end_date,
        },
        query=(
            "SELECT * FROM period_stock_days "
            f"WHERE trade_date BETWEEN {_sql_literal(preparation.start_date)} "
            f"AND {_sql_literal(preparation.end_date)}"
        ),
    )
    stock_day_row = connection.execute(
        "SELECT count(*),count(DISTINCT symbol),count(DISTINCT trade_date) FROM stock_days"
    ).fetchone()
    if not stock_day_row or int(stock_day_row[0]) == 0:
        raise MinuteV2Error(
            f"minute_v2_stock_days_empty:{preparation.start_date}:{preparation.end_date}"
        )
    stock_day_stats = {
        "stock_days": int(stock_day_row[0]),
        "symbols": int(stock_day_row[1]),
        "dates": int(stock_day_row[2]),
    }
    sixty_state_started = time.perf_counter()
    sixty_state_artifact = period_support["sixty_minute_states"]
    sixty_state_reused = bool(sixty_state_artifact.get("reused", False))
    sixty_state_paths = [Path(str(value)).resolve() for value in sixty_state_artifact["paths"]]
    connection.execute(
        "CREATE OR REPLACE TEMP VIEW sixty_minute_states AS SELECT * FROM "
        + _parquet_scan(sixty_state_paths)
    )
    sixty_state_seconds = time.perf_counter() - sixty_state_started
    label_support_artifacts = _materialize_label_support(
        connection,
        start_date=preparation.start_date,
        extended_end_date=preparation.extended_end,
        support_directory=support_directory,
        support_cache_context=support_cache_context,
        stock_days_view="period_label_stock_days",
        calendar_view="period_calendar_dates",
    )
    available_dates = [
        _date_text(row[0])
        for row in connection.execute(
            "SELECT DISTINCT trade_date FROM stock_days ORDER BY trade_date"
        ).fetchall()
    ]
    if not available_dates:
        raise MinuteV2Error(
            f"minute_v2_trading_dates_empty:{preparation.start_date}:{preparation.end_date}"
        )
    return _SupportMaterialization(
        support_directory=support_directory,
        support_cache_context=support_cache_context,
        period_support=period_support,
        stock_day_artifact=stock_day_artifact,
        stock_day_reused=stock_day_reused,
        stock_day_stats=stock_day_stats,
        sixty_state_artifact=sixty_state_artifact,
        sixty_state_reused=sixty_state_reused,
        sixty_state_seconds=sixty_state_seconds,
        label_support_artifacts=label_support_artifacts,
        available_dates=available_dates,
    )


@dataclass(frozen=True)
class _MonthParts:
    dates: list[str]
    processing_chunks: list[list[str]]
    base_parts: list[Path]
    event_parts: list[Path]
    label_parts: list[Path]
    complete_sessions: int
    expected_decision_rows: int
    expected_decision_groups: int
    feature_phase_seconds: float
    candidate_phase_seconds: float
    reused_parts: int
    built_parts: int
    base_rows: int
    uncompressed_base_bytes: int
    compressed_base_bytes: int
    label_cache_seconds: float
    label_cache_rows: int
    label_cache_bytes: int
    label_cache_reused: bool
    label_bucket_specs: list[dict[str, Any]]
    label_cache_comparison: dict[str, Any]


def _materialize_month_parts(
    connection: Any,
    *,
    preparation: _MonthPreparation,
    support: _SupportMaterialization,
    checkpoint: Path,
) -> _MonthParts:
    """Select dates and materialize feature, event, and label parts."""

    dates = _evenly_spaced_dates(
        support.available_dates,
        preparation.current.maximum_trading_days_per_month,
    )
    processing_chunks = _chunks(dates, preparation.current.processing_days_per_chunk)
    feature_result = _materialize_feature_parts(
        connection,
        checkpoint=checkpoint,
        current=preparation.current,
        processing_chunks=processing_chunks,
    )
    event_label_result = _materialize_event_and_label_parts(
        connection,
        checkpoint=checkpoint,
        current=preparation.current,
        processing_chunks=processing_chunks,
        base_parts=feature_result.base_parts,
        support_directory=support.support_directory,
        start_date=preparation.start_date,
        extended_end=preparation.extended_end,
        period_support=support.period_support,
        checkpoint_spec=preparation.checkpoint_spec,
    )
    return _MonthParts(
        dates=dates,
        processing_chunks=processing_chunks,
        base_parts=feature_result.base_parts,
        event_parts=event_label_result.event_parts,
        label_parts=event_label_result.label_parts,
        complete_sessions=feature_result.complete_sessions,
        expected_decision_rows=feature_result.expected_decision_rows,
        expected_decision_groups=feature_result.expected_decision_groups,
        feature_phase_seconds=feature_result.feature_phase_seconds,
        candidate_phase_seconds=event_label_result.candidate_phase_seconds,
        reused_parts=feature_result.reused_parts + event_label_result.reused_parts,
        built_parts=feature_result.built_parts + event_label_result.built_parts,
        base_rows=feature_result.base_rows,
        uncompressed_base_bytes=feature_result.uncompressed_base_bytes,
        compressed_base_bytes=feature_result.compressed_base_bytes,
        label_cache_seconds=event_label_result.label_cache_seconds,
        label_cache_rows=event_label_result.label_cache_rows,
        label_cache_bytes=event_label_result.label_cache_bytes,
        label_cache_reused=event_label_result.label_cache_reused,
        label_bucket_specs=event_label_result.label_bucket_specs,
        label_cache_comparison=event_label_result.label_cache_comparison,
    )


def _finalize_month(
    connection: Any,
    *,
    preparation: _MonthPreparation,
    support: _SupportMaterialization,
    parts: _MonthParts,
    keep_base: bool,
    resumed_checkpoint: bool,
    effective_memory_limit_bytes: int,
    build_started: float,
) -> dict[str, Any]:
    """Combine temporary parts, verify the month contract, and build its manifest."""

    current = preparation.current
    paths = preparation.paths
    artifacts: dict[str, dict[str, Any]] = {}
    if keep_base:
        if current.feature_storage == "full":
            artifacts["base"] = _combine_parts(connection, parts.base_parts, paths["base"])
        else:
            artifacts["base"] = _copy_query(
                connection,
                "SELECT "
                + ",".join(CORE_STORAGE_COLUMNS)
                + " FROM "
                + _parquet_scan(parts.base_parts)
                + " ORDER BY trade_date,bar_time,symbol",
                paths["base"],
            )
            if current.feature_storage == "split":
                artifacts["optional_features"] = _copy_query(
                    connection,
                    "SELECT "
                    + ",".join(OPTIONAL_STORAGE_COLUMNS)
                    + " FROM "
                    + _parquet_scan(parts.base_parts)
                    + " ORDER BY trade_date,bar_time,symbol",
                    paths["optional_features"],
                )
    artifacts["events"] = _combine_parts(connection, parts.event_parts, paths["events"])
    artifacts["labels"] = _combine_parts(connection, parts.label_parts, paths["labels"])
    verification = _verify_month_artifacts(
        connection,
        base=paths.get("base"),
        optional=paths.get("optional_features"),
        events=paths["events"],
        labels=paths["labels"],
        expected_decision_rows=parts.expected_decision_rows,
        expected_decision_groups=parts.expected_decision_groups,
        base_reference_parts=parts.base_parts,
        feature_storage=current.feature_storage,
    )
    if not keep_base:
        _remove_stale_base_artifact(preparation.directory)
        _remove_stale_optional_artifact(preparation.directory)
    total_seconds = time.perf_counter() - build_started
    feature_phase_seconds = parts.feature_phase_seconds
    benchmark = {
        "feature_construction_seconds": feature_phase_seconds,
        "sixty_minute_state_seconds": support.sixty_state_seconds,
        "sixty_minute_state_rows": support.sixty_state_artifact["row_count"],
        "sixty_minute_state_bytes": support.sixty_state_artifact["size"],
        "sixty_minute_state_reused": support.sixty_state_reused,
        "candidate_and_label_seconds": parts.candidate_phase_seconds,
        "all_event_label_cache_seconds": parts.label_cache_seconds,
        "all_event_label_cache_rows": parts.label_cache_rows,
        "all_event_label_cache_bytes": parts.label_cache_bytes,
        "all_event_label_cache_reused": parts.label_cache_reused,
        "all_event_label_cache_buckets": parts.label_bucket_specs,
        "existing_label_equivalence": parts.label_cache_comparison,
        "total_seconds": total_seconds,
        "feature_rows_per_second": (
            parts.base_rows / feature_phase_seconds if feature_phase_seconds > 0 else None
        ),
        "base_uncompressed_bytes": parts.uncompressed_base_bytes,
        "base_compressed_bytes": parts.compressed_base_bytes,
        "base_parquet_compression_ratio": (
            parts.compressed_base_bytes / parts.uncompressed_base_bytes
            if parts.uncompressed_base_bytes > 0
            else None
        ),
        "decision_time_policy": "all 234 causal decision minutes per complete trading day",
        "intermediate_compression": "SNAPPY for new day/support parts; final artifacts ZSTD",
        "feature_storage": current.feature_storage,
        "stored_core_columns": len(CORE_STORAGE_COLUMNS),
        "stored_optional_columns": (
            len(OPTIONAL_STORAGE_COLUMNS) if current.feature_storage == "split" else 0
        ),
    }
    result = {
        "schema": MONTH_SCHEMA,
        "status": "ok",
        "year": preparation.year,
        "month": preparation.month,
        "start_date": preparation.start_date,
        "end_date": preparation.end_date,
        "extended_label_end_date": preparation.extended_end,
        "minute_history_start_date": preparation.minute_history_start,
        "config": current.as_dict(),
        "source": preparation.snapshot.as_dict(),
        "build_spec": preparation.manifest_build_spec,
        "stock_days": support.stock_day_stats,
        "shared_support": {
            "stock_days": {
                "rows": support.stock_day_artifact["row_count"],
                "bytes": support.stock_day_artifact["size"],
                "reused": support.stock_day_reused,
            },
            **{
                name: {
                    "rows": value["row_count"],
                    "bytes": value["size"],
                    "reused": value["reused"],
                }
                for name, value in support.label_support_artifacts.items()
            },
            "period_support": {
                "period": preparation.period,
                "directory": support.period_support["directory"],
                "reused": all(
                    bool(value.get("reused", False))
                    for name, value in support.period_support.items()
                    if isinstance(value, dict) and name != "period"
                ),
                "artifacts": {
                    name: {
                        "rows": value["row_count"],
                        "bytes": value["size"],
                        "reused": value.get("reused", False),
                    }
                    for name, value in support.period_support.items()
                    if isinstance(value, dict) and "row_count" in value
                },
            },
        },
        "support_cache": {
            "schema": SUPPORT_CACHE_SCHEMA,
            "signature": preparation.support_signature,
            "directory": str(support.support_directory),
            "persistent": True,
            "period": preparation.period,
        },
        "complete_session_stock_days": parts.complete_sessions,
        "date_selection": {
            "available_trading_days": len(support.available_dates),
            "selected_trading_days": len(parts.dates),
            "selected_dates": parts.dates,
            "policy": (
                "all trading days"
                if len(parts.dates) == len(support.available_dates)
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
            "trading_days": len(parts.dates),
            "processing_chunks": len(parts.processing_chunks),
            "days_per_chunk": current.processing_days_per_chunk,
            "resumed_checkpoint": resumed_checkpoint,
            "reused_parts": parts.reused_parts,
            "built_parts": parts.built_parts,
        },
        "artifacts": artifacts,
        "verification": verification,
    }
    write_json(preparation.manifest_path, result)
    return result


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
    preparation = _prepare_month(
        workspace_root,
        year=year,
        month=month,
        output_root=output_root,
        config=config,
        keep_base=keep_base,
        force=force,
    )
    if preparation.cached_result is not None:
        return preparation.cached_result

    runtime, external_runtime = _runtime_directory(
        preparation.directory,
        preparation.current,
        year=preparation.year,
        month=preparation.month,
    )
    checkpoint, resumed_checkpoint = _prepare_checkpoint(
        preparation.directory,
        spec=preparation.checkpoint_spec,
        force=force,
    )
    connection = open_guarded_duckdb(
        ":memory:",
        temp_directory=runtime,
        threads=preparation.current.duckdb_threads,
        profiling_path=_profile_path(preparation.directory, preparation.current),
        floor_bytes=int(preparation.current.memory_floor_gib * GIB),
        minimum_limit_bytes=256 * MIB,
    )
    effective_memory_limit_bytes = _effective_memory_limit_bytes(
        connection,
        preparation.current,
    )
    connection.execute(f"SET memory_limit='{effective_memory_limit_bytes}B'")
    completed = False
    build_started = time.perf_counter()
    try:
        support = _materialize_support_views(connection, preparation=preparation)
        parts = _materialize_month_parts(
            connection,
            preparation=preparation,
            support=support,
            checkpoint=checkpoint,
        )
        result = _finalize_month(
            connection,
            preparation=preparation,
            support=support,
            parts=parts,
            keep_base=keep_base,
            resumed_checkpoint=resumed_checkpoint,
            effective_memory_limit_bytes=effective_memory_limit_bytes,
            build_started=build_started,
        )
        completed = True
        return result
    finally:
        connection.close()
        _clean_runtime_directory(
            runtime,
            external=external_runtime,
            month_directory=preparation.directory,
        )
        if completed:
            _safe_clean_generated(
                checkpoint,
                preparation.directory,
                expected_name="_parts",
            )

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
    manifest_file = Path(manifest_path).resolve()
    try:
        manifest = read_json(manifest_file)
    except (DataContractError, OSError, ValueError, TypeError) as exc:
        raise MinuteV2Error(f"minute_v2_manifest_unreadable:{manifest_file}") from exc
    if not isinstance(manifest, dict):
        raise MinuteV2Error("minute_v2_manifest_not_object")
    if manifest.get("status") != "ok" or manifest.get("schema") != MONTH_SCHEMA:
        raise MinuteV2Error("minute_v2_manifest_not_ok")
    unfinished = _unfinished_build_files(manifest_file)
    if unfinished:
        raise MinuteV2Error(f"minute_v2_manifest_build_incomplete:{unfinished[0]}")
    build_spec = manifest.get("build_spec")
    if (
        not isinstance(build_spec, dict)
        or build_spec.get("schema") != BUILD_SPEC_SCHEMA
        or not isinstance(build_spec.get("keep_base"), bool)
        or not isinstance(build_spec.get("implementation_revision"), str)
        or not isinstance(build_spec.get("content_signature"), str)
    ):
        raise MinuteV2Error("minute_v2_manifest_build_spec_missing")
    if build_spec["implementation_revision"] != BUILD_IMPLEMENTATION_REVISION:
        raise MinuteV2Error(
            "minute_v2_manifest_build_revision_mismatch:"
            f"{build_spec['implementation_revision']}:{BUILD_IMPLEMENTATION_REVISION}"
        )
    feature_storage = str(build_spec.get("feature_storage", "full"))
    if feature_storage not in {"full", "split", "core"}:
        raise MinuteV2Error("minute_v2_manifest_feature_storage_invalid")
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, dict) or not raw_artifacts:
        raise MinuteV2Error("minute_v2_manifest_artifacts_missing")
    artifacts = raw_artifacts
    keep_base = build_spec["keep_base"]
    stale_base = manifest_file.parent / "base.parquet"
    if not keep_base and (stale_base.exists() or stale_base.is_symlink()):
        raise MinuteV2Error(f"minute_v2_verify_unexpected_base_artifact:{stale_base}")
    expected_names = {"events", "labels"} | ({"base"} if keep_base else set())
    if keep_base and feature_storage == "split":
        expected_names.add("optional_features")
    if set(artifacts) != expected_names:
        raise MinuteV2Error(
            "minute_v2_manifest_artifact_set_invalid:"
            f"{sorted(artifacts)}:{sorted(expected_names)}"
        )
    checked: dict[str, Any] = {}
    for name, record_value in artifacts.items():
        if not isinstance(record_value, dict):
            raise MinuteV2Error(f"minute_v2_verify_artifact_record_invalid:{name}")
        record = record_value
        path = Path(str(record.get("path", "")))
        try:
            if not path.is_absolute():
                path = (manifest_file.parent / path).resolve()
            expected_path = (manifest_file.parent / f"{name}.parquet").resolve()
        except (OSError, ValueError, TypeError) as exc:
            raise MinuteV2Error(f"minute_v2_verify_artifact_path_invalid:{name}") from exc
        if path != expected_path:
            raise MinuteV2Error(f"minute_v2_verify_artifact_path_invalid:{name}:{path}")
        if not path.is_file():
            raise MinuteV2Error(f"minute_v2_verify_artifact_missing:{name}:{path}")
        try:
            current = _artifact(path)
        except Exception as exc:
            raise MinuteV2Error(f"minute_v2_verify_artifact_unreadable:{name}:{path}") from exc
        if any(record.get(field) != current[field] for field in ("size", "sha256", "row_count", "row_group_count")):
            raise MinuteV2Error(f"minute_v2_verify_artifact_changed:{name}:{path}")
        checked[name] = current
    raw_verification = manifest.get("verification")
    verification = raw_verification if isinstance(raw_verification, dict) else {}
    expected_rows_value = verification.get("expected_decision_rows")
    if expected_rows_value is None:
        expected_rows_value = checked.get("base", checked["events"]).get("row_count")
    try:
        expected_rows = int(expected_rows_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MinuteV2Error("minute_v2_manifest_expected_decision_rows_invalid") from exc
    expected_groups_value = verification.get("decision_group_count")
    try:
        expected_groups = (
            int(expected_groups_value) if expected_groups_value is not None else None
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise MinuteV2Error("minute_v2_manifest_decision_group_count_invalid") from exc
    if not keep_base and expected_groups is None:
        raise MinuteV2Error("minute_v2_manifest_verification_missing")
    base_path = Path(checked["base"]["path"]) if "base" in checked else None
    optional_path = (
        Path(checked["optional_features"]["path"])
        if "optional_features" in checked
        else None
    )
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET threads=2")
        connection.execute("SET preserve_insertion_order=false")
        contract = _verify_month_artifacts(
            connection,
            base=base_path,
            optional=optional_path,
            events=Path(checked["events"]["path"]),
            labels=Path(checked["labels"]["path"]),
            expected_decision_rows=expected_rows,
            expected_decision_groups=expected_groups,
            base_reference_parts=[base_path] if base_path is not None else None,
            feature_storage=feature_storage,
        )
    except duckdb.Error as exc:
        raise MinuteV2Error(f"minute_v2_verify_contract_unreadable:{manifest_file}") from exc
    finally:
        connection.close()
    return {
        "status": "ok",
        "manifest": str(manifest_file),
        "artifacts": checked,
        "contract": contract,
    }


def verify_dataset(manifest_path: str | Path) -> dict[str, Any]:
    dataset_manifest_path = Path(manifest_path).resolve()
    try:
        manifest = read_json(dataset_manifest_path)
    except (DataContractError, OSError, ValueError, TypeError) as exc:
        raise MinuteV2Error(
            f"minute_v2_dataset_manifest_unreadable:{dataset_manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise MinuteV2Error("minute_v2_dataset_manifest_not_object")
    if manifest.get("status") != "ok" or manifest.get("schema") != "quantlab.minute_v2_dataset/1":
        raise MinuteV2Error("minute_v2_dataset_manifest_not_ok")

    def required_int(name: str, *, minimum: int = 0) -> int:
        value = manifest.get(name)
        if type(value) is not int or value < minimum:
            raise MinuteV2Error(f"minute_v2_dataset_{name}_invalid")
        return int(value)

    start_year = required_int("start_year", minimum=1900)
    end_year = required_int("end_year", minimum=start_year)
    expected_pairs = {
        (year, month)
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
    }
    month_count = required_int("month_count")
    raw_months = manifest.get("months")
    if not isinstance(raw_months, list):
        raise MinuteV2Error("minute_v2_dataset_months_invalid")
    months = raw_months
    if len(months) != month_count or month_count != len(expected_pairs):
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
    seen_pairs: set[tuple[int, int]] = set()
    for record in months:
        if not isinstance(record, dict):
            raise MinuteV2Error("minute_v2_dataset_month_record_invalid")
        year = record.get("year")
        month = record.get("month")
        manifest_value = record.get("manifest")
        if (
            type(year) is not int
            or type(month) is not int
            or not isinstance(manifest_value, str)
            or not 1 <= month <= 12
        ):
            raise MinuteV2Error("minute_v2_dataset_month_record_invalid")
        pair = (int(year), int(month))
        if pair not in expected_pairs or pair in seen_pairs:
            raise MinuteV2Error(f"minute_v2_dataset_month_set_invalid:{pair[0]}:{pair[1]}")
        seen_pairs.add(pair)
        try:
            month_manifest_path = Path(manifest_value)
            if not month_manifest_path.is_absolute():
                month_manifest_path = dataset_manifest_path.parent / month_manifest_path
            month_manifest_path = month_manifest_path.resolve()
            expected_month_path = (
                dataset_manifest_path.parent
                / "months"
                / f"year={pair[0]:04d}"
                / f"month={pair[1]:02d}"
                / "manifest.json"
            ).resolve()
        except (OSError, ValueError, TypeError) as exc:
            raise MinuteV2Error("minute_v2_dataset_month_manifest_path_invalid") from exc
        if month_manifest_path != expected_month_path:
            raise MinuteV2Error(
                "minute_v2_dataset_month_manifest_path_invalid:"
                f"{month_manifest_path}"
            )
        try:
            month_manifest = read_json(month_manifest_path)
        except (DataContractError, OSError, ValueError, TypeError) as exc:
            raise MinuteV2Error(
                f"minute_v2_dataset_month_manifest_unreadable:{month_manifest_path}"
            ) from exc
        if (
            not isinstance(month_manifest, dict)
            or month_manifest.get("year") != pair[0]
            or month_manifest.get("month") != pair[1]
        ):
            raise MinuteV2Error(
                f"minute_v2_dataset_month_identity_invalid:{pair[0]:04d}-{pair[1]:02d}"
            )
        checked = verify_month(month_manifest_path)
        artifact_count += len(checked["artifacts"])
        artifact_bytes += sum(int(value["size"]) for value in checked["artifacts"].values())
        verification = checked["contract"]
        for name in totals:
            value = verification.get(name)
            if type(value) is not int or value < 0:
                raise MinuteV2Error(
                    f"minute_v2_dataset_month_total_invalid:{pair[0]:04d}-{pair[1]:02d}:{name}"
                )
            totals[name] += int(value)
        selection = month_manifest.get("date_selection", {})
        if not isinstance(selection, dict):
            raise MinuteV2Error("minute_v2_dataset_date_selection_invalid")
        selected = selection.get("selected_trading_days", 0)
        if type(selected) is not int or selected < 0:
            raise MinuteV2Error("minute_v2_dataset_date_selection_invalid")
        selected_dates += int(selected)
    if seen_pairs != expected_pairs:
        raise MinuteV2Error("minute_v2_dataset_month_set_invalid")
    for name, value in totals.items():
        declared = manifest.get(name)
        if type(declared) is not int or declared < 0 or value != declared:
            raise MinuteV2Error(f"minute_v2_dataset_total_mismatch:{name}:{value}")
    return {
        "schema": "quantlab.minute_v2_dataset_verification/1",
        "status": "ok",
        "manifest": str(dataset_manifest_path),
        "start_year": start_year,
        "end_year": end_year,
        "month_count": len(months),
        "selected_trading_days": selected_dates,
        "artifact_count": artifact_count,
        "artifact_bytes": artifact_bytes,
        **totals,
    }


__all__ = ["build_month", "build_range", "verify_dataset", "verify_month"]
