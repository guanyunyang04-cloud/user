"""Single-split, bounded-memory baseline training for minute-v2 events."""

from __future__ import annotations

import gc
import os
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from quantlab.core.io import DataContractError, read_json, write_json

from .artifacts import artifact_matches as _training_artifact_matches
from .builder import BUILD_IMPLEMENTATION_REVISION, BUILD_SPEC_SCHEMA, MONTH_SCHEMA
from .contracts import (
    CORE_STORAGE_COLUMNS,
    KEY_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    OPTIONAL_STORAGE_COLUMNS,
    MinuteV2Error,
    model_feature_columns_for_storage,
)
from .models import (
    evaluate_scores,
    fit_lightgbm_ranker,
    fit_ridge_chunks,
    rule_score,
)
from .replay import EventReplayConfig, replay_events

DEVELOPMENT_YEARS = (2022, 2023, 2024)
FINAL_VALIDATION_YEAR = 2025


@dataclass(frozen=True)
class DevelopmentValidationSpec:
    development_start_year: int
    development_end_year: int
    validation_start_year: int
    validation_end_year: int

    @property
    def development_label_end_exclusive(self) -> str:
        """Keep every outcome touching validation outside development fitting."""

        return f"{self.validation_start_year:04d}-01-01"

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


DEVELOPMENT_VALIDATION_SPEC = DevelopmentValidationSpec(
    development_start_year=DEVELOPMENT_YEARS[0],
    development_end_year=DEVELOPMENT_YEARS[-1],
    validation_start_year=FINAL_VALIDATION_YEAR,
    validation_end_year=FINAL_VALIDATION_YEAR,
)

EXECUTION_LABEL_COLUMNS = (
    "planned_exit_date",
    "entry_bar_time",
    "entry_price",
    "entry_amount",
    "entry_executable",
    "actual_exit_date",
    "exit_amount",
    "label_observed",
)

DEFAULT_BATCH_ROWS = 50_000
DEFAULT_LGBM_ROWS_PER_GROUP = 16
DEFAULT_LGBM_MAX_ROWS = 600_000
DEFAULT_MEDIAN_SAMPLE_ROWS = 8_192
MIN_TRAINING_AVAILABLE_GIB = 2.0


def _literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _scan(paths: list[Path]) -> str:
    if not paths:
        raise MinuteV2Error("minute_v2_training_paths_empty")
    values = ",".join(_literal(path) for path in paths)
    return f"read_parquet([{values}], union_by_name=true)"


@dataclass(frozen=True)
class PeriodPart:
    year: int
    month: int
    events: Path
    labels: Path
    base: Path
    optional: Path | None = None
    feature_storage: str = "full"


def _strict_positive_integer(value: Any, error: str) -> int:
    """Accept integer counts without silently truncating floats or booleans."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise MinuteV2Error(error)
    result = int(value)
    if result <= 0:
        raise MinuteV2Error(error)
    return result


def _has_unfinished_month_build(directory: Path) -> bool:
    if any(path.is_file() for path in directory.rglob("*.partial")):
        return True
    return any((directory / name).is_dir() for name in ("_parts", "_runtime"))


def _period_parts(dataset_root: Path, start_year: int, end_year: int) -> list[PeriodPart]:
    dataset_root = Path(dataset_root).resolve()
    parts: list[PeriodPart] = []
    for year in range(int(start_year), int(end_year) + 1):
        for month in range(1, 13):
            directory = dataset_root / "months" / f"year={year:04d}" / f"month={month:02d}"
            manifest_path = directory / "manifest.json"
            if not manifest_path.is_file():
                raise MinuteV2Error(f"minute_v2_training_manifest_missing:{year:04d}-{month:02d}")
            try:
                manifest = read_json(manifest_path)
            except (DataContractError, OSError, ValueError, TypeError) as exc:
                raise MinuteV2Error(
                    f"minute_v2_training_manifest_unreadable:{manifest_path}"
                ) from exc
            if (
                not isinstance(manifest, dict)
                or manifest.get("status") != "ok"
                or manifest.get("schema") != MONTH_SCHEMA
            ):
                raise MinuteV2Error(f"minute_v2_training_manifest_not_ok:{manifest_path}")
            if _has_unfinished_month_build(directory):
                raise MinuteV2Error(
                    f"minute_v2_training_manifest_build_incomplete:{manifest_path}"
                )
            if manifest.get("year") != year or manifest.get("month") != month:
                raise MinuteV2Error(
                    f"minute_v2_training_manifest_identity_invalid:{manifest_path}"
                )
            build_spec = manifest.get("build_spec")
            if not isinstance(build_spec, dict):
                raise MinuteV2Error(
                    f"minute_v2_training_build_spec_invalid:{manifest_path}"
                )
            if build_spec.get("schema") != BUILD_SPEC_SCHEMA:
                raise MinuteV2Error(
                    f"minute_v2_training_build_spec_schema_invalid:{manifest_path}"
                )
            if build_spec.get("implementation_revision") != BUILD_IMPLEMENTATION_REVISION:
                raise MinuteV2Error(
                    f"minute_v2_training_build_revision_mismatch:{manifest_path}"
                )
            if build_spec.get("keep_base") is not True:
                raise MinuteV2Error(
                    f"minute_v2_training_base_required:{manifest_path}"
                )
            feature_storage = str(build_spec.get("feature_storage", "full"))
            if feature_storage not in {"full", "split", "core"}:
                raise MinuteV2Error(
                    f"minute_v2_training_feature_storage_invalid:{manifest_path}"
                )
            raw_artifacts = manifest.get("artifacts")
            expected_artifacts = {"base", "events", "labels"}
            if feature_storage == "split":
                expected_artifacts.add("optional_features")
            if not isinstance(raw_artifacts, dict) or set(raw_artifacts) != expected_artifacts:
                raise MinuteV2Error(f"minute_v2_training_artifacts_invalid:{manifest_path}")

            def artifact_path(
                name: str,
                *,
                records=raw_artifacts,
                month_directory=directory,
                manifest_file=manifest_path,
            ) -> Path:
                record = records.get(name)
                if not isinstance(record, dict):
                    raise MinuteV2Error(
                        f"minute_v2_training_artifact_record_invalid:{manifest_file}:{name}"
                    )
                value = Path(str(record.get("path", "")))
                target = value if value.is_absolute() else month_directory / value
                target = target.resolve()
                expected = (month_directory / f"{name}.parquet").resolve()
                if target != expected:
                    raise MinuteV2Error(
                        f"minute_v2_training_artifact_path_invalid:{manifest_file}:{name}"
                    )
                return target

            artifact_paths = {
                name: artifact_path(name) for name in expected_artifacts
            }
            for name, path in artifact_paths.items():
                if not path.is_file():
                    raise MinuteV2Error(
                        f"minute_v2_training_artifact_missing:{year:04d}-{month:02d}:{name}"
                    )
                if not _training_artifact_matches(
                    path,
                    raw_artifacts[name],
                    base_directory=path.parent,
                ):
                    raise MinuteV2Error(
                        f"minute_v2_training_artifact_changed:{manifest_path}:{name}"
                    )

            schemas = {
                name: set(pq.ParquetFile(path).schema_arrow.names)
                for name, path in artifact_paths.items()
            }
            required_base_columns = (
                set(CORE_STORAGE_COLUMNS)
                if feature_storage in {"split", "core"}
                else set(KEY_COLUMNS) | set(MODEL_FEATURE_COLUMNS)
            )
            missing_base = sorted(required_base_columns.difference(schemas["base"]))
            if missing_base:
                raise MinuteV2Error(
                    "minute_v2_training_base_columns_missing:"
                    f"{manifest_path}:{','.join(missing_base)}"
                )
            if feature_storage == "split":
                missing_optional = sorted(
                    set(OPTIONAL_STORAGE_COLUMNS).difference(schemas["optional_features"])
                )
                if missing_optional:
                    raise MinuteV2Error(
                        "minute_v2_training_optional_columns_missing:"
                        f"{manifest_path}:{','.join(missing_optional)}"
                    )
            missing_events = sorted(set(KEY_COLUMNS).difference(schemas["events"]))
            if missing_events:
                raise MinuteV2Error(
                    "minute_v2_training_event_columns_missing:"
                    f"{manifest_path}:{','.join(missing_events)}"
                )
            missing_labels = sorted(
                {
                    "label_net_return",
                    "label_observed",
                    *EXECUTION_LABEL_COLUMNS,
                }.difference(schemas["labels"])
            )
            if missing_labels:
                raise MinuteV2Error(
                    "minute_v2_training_label_columns_missing:"
                    f"{manifest_path}:{','.join(missing_labels)}"
                )
            row_counts = {
                name: int(pq.ParquetFile(path).metadata.num_rows)
                for name, path in artifact_paths.items()
            }
            if row_counts["base"] <= 0 or row_counts["events"] <= 0:
                raise MinuteV2Error(
                    f"minute_v2_training_artifact_empty:{manifest_path}"
                )
            if row_counts["events"] != row_counts["labels"]:
                raise MinuteV2Error(
                    f"minute_v2_training_event_label_rows_mismatch:{manifest_path}"
                )
            if feature_storage == "split" and row_counts["optional_features"] != row_counts["base"]:
                raise MinuteV2Error(
                    f"minute_v2_training_optional_rows_mismatch:{manifest_path}"
                )
            raw_verification = manifest.get("verification")
            if not isinstance(raw_verification, dict):
                raise MinuteV2Error(
                    f"minute_v2_training_verification_invalid:{manifest_path}"
                )
            for name, count in (
                ("expected_decision_rows", row_counts["base"]),
                ("event_rows", row_counts["events"]),
                ("label_rows", row_counts["labels"]),
            ):
                declared = raw_verification.get(name)
                if type(declared) is not int or declared != count:
                    raise MinuteV2Error(
                        f"minute_v2_training_verification_mismatch:{manifest_path}:{name}"
                    )
            event_path = artifact_paths["events"]
            label_path = artifact_paths["labels"]
            base_path = artifact_paths["base"]
            parts.append(
                PeriodPart(
                    year=int(year),
                    month=int(month),
                    events=event_path.resolve(),
                    labels=label_path.resolve(),
                    base=base_path.resolve(),
                    optional=(
                        artifact_paths["optional_features"].resolve()
                        if feature_storage == "split"
                        else None
                    ),
                    feature_storage=feature_storage,
                )
            )
    storage_modes = sorted({part.feature_storage for part in parts})
    if len(storage_modes) > 1:
        raise MinuteV2Error(
            "minute_v2_training_mixed_feature_storage:" + ",".join(storage_modes)
        )
    return parts


def _period_paths(
    dataset_root: Path, start_year: int, end_year: int
) -> tuple[list[Path], list[Path], list[Path]]:
    parts = _period_parts(dataset_root, start_year, end_year)
    return (
        [part.events for part in parts],
        [part.labels for part in parts],
        [part.base for part in parts],
    )


def _period_query(
    part: PeriodPart,
    *,
    include_execution: bool,
    label_end_exclusive: str | None,
    ordered: bool,
) -> str:
    event_scan = _scan([part.events])
    label_scan = _scan([part.labels])
    base_scan = _scan([part.base])
    optional_scan = None
    if part.feature_storage == "split":
        if part.optional is None:
            raise MinuteV2Error("minute_v2_training_optional_features_missing")
        optional_scan = _scan([part.optional])
    optional_names = set(OPTIONAL_STORAGE_COLUMNS)
    feature_names = model_feature_columns_for_storage(part.feature_storage)
    feature_projection = ",".join(
        (
            f"o.{name}"
            if optional_scan is not None and name in optional_names
            else f"b.{name}"
        )
        for name in feature_names
    )
    execution_projection = (
        "," + ",".join(f"l.{name}" for name in EXECUTION_LABEL_COLUMNS)
        if include_execution
        else ""
    )
    boundary = ""
    if label_end_exclusive is not None:
        cutoff = _literal(str(label_end_exclusive))
        boundary = (
            "AND l.label_10d_observed "
            f"AND CAST(l.label_end_date_10d AS DATE) < CAST({cutoff} AS DATE)"
        )
    order = "ORDER BY e.trade_date,e.bar_time,e.symbol" if ordered else ""
    optional_join = (
        f"JOIN {optional_scan} o USING(symbol,trade_date,bar_time)"
        if optional_scan is not None
        else ""
    )
    return f"""
        SELECT
            CAST(e.symbol AS VARCHAR) AS symbol,
            CAST(e.trade_date AS VARCHAR) AS trade_date,
            CAST(e.bar_time AS VARCHAR) AS bar_time,
            {feature_projection},
            CAST(l.label_net_return AS DOUBLE) AS label_net_return
            {execution_projection}
        FROM {event_scan} e
        JOIN {base_scan} b USING(symbol,trade_date,bar_time)
        {optional_join}
        JOIN {label_scan} l USING(symbol,trade_date,bar_time)
        WHERE l.label_observed {boundary}
        {order}
    """


def _normalise_training_frame(
    frame: pd.DataFrame,
    *,
    feature_names: tuple[str, ...] = MODEL_FEATURE_COLUMNS,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    for name in KEY_COLUMNS:
        frame[name] = frame[name].astype(str)
    for name in feature_names:
        frame[name] = pd.to_numeric(frame[name], errors="coerce").astype("float32")
    frame["label_net_return"] = pd.to_numeric(
        frame["label_net_return"], errors="coerce"
    ).astype("float64")
    return frame


def iter_period_frames(
    dataset_root: str | Path,
    *,
    start_year: int,
    end_year: int,
    include_execution: bool,
    temp_directory: str | Path,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    label_end_exclusive: str | None = None,
    ordered: bool = False,
) -> Iterator[pd.DataFrame]:
    """Yield bounded joined feature/label chunks, releasing DuckDB each month."""

    rows_per_batch = int(batch_rows)
    if rows_per_batch <= 0:
        raise MinuteV2Error("minute_v2_training_batch_rows_invalid")
    dataset = Path(dataset_root).resolve()
    temporary = Path(temp_directory).resolve()
    temporary.mkdir(parents=True, exist_ok=True)
    for part in _period_parts(dataset, start_year, end_year):
        feature_names = model_feature_columns_for_storage(part.feature_storage)
        month_temp = temporary / f"year={part.year:04d}" / f"month={part.month:02d}"
        month_temp.mkdir(parents=True, exist_ok=True)
        connection = duckdb.connect(":memory:")
        try:
            connection.execute("SET threads=2")
            connection.execute("SET memory_limit='2GB'")
            connection.execute("SET preserve_insertion_order=false")
            connection.execute(f"SET temp_directory={_literal(month_temp)}")
            query = _period_query(
                part,
                include_execution=include_execution,
                label_end_exclusive=label_end_exclusive,
                ordered=ordered,
            )
            reader = connection.execute(query).fetch_record_batch(rows_per_batch)
            for batch in reader:
                frame = _normalise_training_frame(
                    batch.to_pandas(),
                    feature_names=feature_names,
                )
                if not frame.empty:
                    yield frame
        finally:
            connection.close()


def _load_period(
    dataset_root: Path,
    *,
    start_year: int,
    end_year: int,
    include_execution: bool,
    temp_directory: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = list(
        iter_period_frames(
            dataset_root,
            start_year=start_year,
            end_year=end_year,
            include_execution=include_execution,
            temp_directory=temp_directory,
            ordered=True,
        )
    )
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if frame.empty:
        raise MinuteV2Error(
            f"minute_v2_training_period_empty:{start_year}:{end_year}"
        )
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise MinuteV2Error("minute_v2_training_duplicate_keys")
    event_paths, label_paths, base_paths = _period_paths(dataset_root, start_year, end_year)
    metadata = {
        "start_year": int(start_year),
        "end_year": int(end_year),
        "sampling_policy": "all causal candidate rows at all decision minutes",
        "row_count": int(len(frame)),
        "group_count": int(frame.groupby(["trade_date", "bar_time"]).ngroups),
        "event_file_count": len(event_paths),
        "label_file_count": len(label_paths),
        "base_file_count": len(base_paths),
        "feature_source": "base_joined_by_symbol_trade_date_bar_time",
    }
    return frame, metadata


def _sample_cross_sections(
    frame: pd.DataFrame,
    *,
    rows_per_group: int = DEFAULT_LGBM_ROWS_PER_GROUP,
) -> pd.DataFrame:
    """Keep a deterministic bounded slice from each decision cross-section."""

    limit = int(rows_per_group)
    if limit <= 0:
        raise MinuteV2Error("minute_v2_training_group_sample_invalid")
    if frame.empty:
        return frame.copy()
    working = frame.copy()
    keys = working.loc[:, list(KEY_COLUMNS)].astype(str)
    working["_stable_sample_hash"] = pd.util.hash_pandas_object(
        keys, index=False
    ).astype("uint64")
    working.sort_values(
        ["trade_date", "bar_time", "_stable_sample_hash", "symbol"],
        kind="stable",
        inplace=True,
    )
    selected = (
        working.groupby(["trade_date", "bar_time"], sort=False, group_keys=False)
        .head(limit)
        .copy()
    )
    return selected.drop(columns=["_stable_sample_hash"])


def _collect_cross_section_sample(
    chunk_factory: Callable[[], Iterable[pd.DataFrame]],
    *,
    rows_per_group: int = DEFAULT_LGBM_ROWS_PER_GROUP,
    maximum_rows: int = DEFAULT_LGBM_MAX_ROWS,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Collect a deterministic LightGBM sample while bounding its size."""

    maximum = int(maximum_rows)
    if maximum <= 0:
        raise MinuteV2Error("minute_v2_training_sample_limit_invalid")
    pieces: list[pd.DataFrame] = []
    source_rows = 0
    source_chunks = 0
    for frame in chunk_factory():
        source_chunks += 1
        source_rows += int(len(frame))
        sampled = _sample_cross_sections(frame, rows_per_group=rows_per_group)
        if not sampled.empty:
            pieces.append(sampled)
    if not pieces:
        raise MinuteV2Error("minute_v2_training_sample_empty")
    sample = pd.concat(pieces, ignore_index=True)
    sample.sort_values(list(KEY_COLUMNS), kind="stable", inplace=True)
    sample = sample.drop_duplicates(list(KEY_COLUMNS), keep="first")
    # A decision cross-section can straddle two record batches. Reapply the
    # cap after merging so the bound is global, not merely per batch.
    sample = _sample_cross_sections(sample, rows_per_group=rows_per_group)
    # The per-group cap normally keeps this well below the global limit. If a
    # future universe grows beyond it, retain a stable hash prefix rather than
    # silently depending on filesystem/read order.
    if len(sample) > maximum:
        keys = sample.loc[:, list(KEY_COLUMNS)].astype(str)
        stable_hash = pd.util.hash_pandas_object(keys, index=False).astype("uint64")
        keep_indices = stable_hash.nsmallest(maximum).index
        sample = sample.loc[keep_indices].copy()
    sample.sort_values(["trade_date", "bar_time", "symbol"], kind="stable", inplace=True)
    sample.reset_index(drop=True, inplace=True)
    return sample, {
        "source_rows": int(source_rows),
        "source_chunks": int(source_chunks),
        "sample_rows": int(len(sample)),
        "rows_per_group": int(rows_per_group),
        "maximum_rows": int(maximum),
        "sampling_policy": "deterministic smallest hash per decision cross-section",
    }


SCORED_COLUMNS = (
    *KEY_COLUMNS,
    "score",
    "label_net_return",
    *EXECUTION_LABEL_COLUMNS,
)

SCORED_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string()),
        pa.field("trade_date", pa.string()),
        pa.field("bar_time", pa.string()),
        pa.field("score", pa.float64()),
        pa.field("label_net_return", pa.float64()),
        pa.field("planned_exit_date", pa.string()),
        pa.field("entry_bar_time", pa.string()),
        pa.field("entry_price", pa.float64()),
        pa.field("entry_amount", pa.float64()),
        pa.field("entry_executable", pa.bool_()),
        pa.field("actual_exit_date", pa.string()),
        pa.field("exit_amount", pa.float64()),
        pa.field("label_observed", pa.bool_()),
    ]
)


def _write_scored_period(
    chunk_factory: Callable[[], Iterable[pd.DataFrame]],
    scorer: Callable[[pd.DataFrame], np.ndarray | pd.Series],
    path: Path,
) -> dict[str, int | str]:
    """Score chunks into one temporary, narrow Parquet file."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    writer: pq.ParquetWriter | None = None
    row_count = 0
    chunk_count = 0
    completed = False
    try:
        for frame in chunk_factory():
            if frame.empty:
                continue
            scores = np.asarray(scorer(frame), dtype=np.float64)
            if len(scores) != len(frame):
                raise MinuteV2Error("minute_v2_stream_score_length_mismatch")
            table = _scored_table(frame, scores)
            if writer is None:
                writer = pq.ParquetWriter(partial, SCORED_SCHEMA, compression="SNAPPY")
            writer.write_table(table)
            row_count += int(len(frame))
            chunk_count += 1
        if writer is None:
            raise MinuteV2Error("minute_v2_stream_score_empty")
        # Keep the target untouched until the writer is closed and the atomic
        # replacement succeeds.  A failed replacement must not leave a
        # misleading .partial artifact behind.
        writer.close()
        writer = None
        os.replace(partial, target)
        completed = True
    finally:
        if writer is not None:
            writer.close()
        if not completed and partial.exists():
            partial.unlink()
    return {
        "path": str(target),
        "row_count": int(row_count),
        "chunk_count": int(chunk_count),
        "size": int(target.stat().st_size),
    }


def _scored_table(
    frame: pd.DataFrame,
    scores: np.ndarray | pd.Series,
) -> pa.Table:
    values = np.asarray(scores, dtype=np.float64)
    if len(values) != len(frame):
        raise MinuteV2Error("minute_v2_stream_score_length_mismatch")
    selected = frame.loc[:, [name for name in SCORED_COLUMNS if name in frame]].copy()
    selected["score"] = values
    for name in SCORED_COLUMNS:
        if name not in selected:
            selected[name] = None
    selected = selected.loc[:, list(SCORED_COLUMNS)]
    for name in (
        "symbol",
        "trade_date",
        "bar_time",
        "planned_exit_date",
        "entry_bar_time",
        "actual_exit_date",
    ):
        selected[name] = selected[name].astype("string")
    for name in ("entry_executable", "label_observed"):
        selected[name] = selected[name].astype("boolean")
    return pa.Table.from_pandas(
        selected,
        schema=SCORED_SCHEMA,
        preserve_index=False,
        safe=False,
    )


def _commit_staged_score_files(
    paths: dict[str, Path],
    partial_paths: dict[str, Path],
) -> None:
    """Commit a group of score files with rollback on a partial failure."""

    backups: dict[Path, Path] = {}
    committed: list[Path] = []
    token = uuid.uuid4().hex
    try:
        for name, path in paths.items():
            partial = partial_paths[name]
            backup = path.with_suffix(path.suffix + f".rollback-{token}")
            if path.exists():
                os.replace(path, backup)
                backups[path] = backup
            os.replace(partial, path)
            committed.append(path)
    except Exception:
        # Remove any newly committed files, then restore every target that had
        # an older version.  Paths are exact generated outputs, never globs.
        for path in reversed(committed):
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass
        for path, backup in backups.items():
            if backup.exists() and not path.exists():
                try:
                    os.replace(backup, path)
                except OSError:
                    pass
        raise
    else:
        for backup in backups.values():
            if backup.exists():
                backup.unlink()
    finally:
        for partial in partial_paths.values():
            if partial.exists():
                partial.unlink()
        for backup in backups.values():
            if backup.exists():
                backup.unlink()


def _write_multiple_scored_periods(
    chunk_factory: Callable[[], Iterable[pd.DataFrame]],
    scorers: dict[str, Callable[[pd.DataFrame], np.ndarray | pd.Series]],
    directory: Path,
) -> dict[str, dict[str, int | str]]:
    """Score several models in one source pass and write narrow files."""

    if not scorers:
        raise MinuteV2Error("minute_v2_stream_scorers_empty")
    directory.mkdir(parents=True, exist_ok=True)
    writers: dict[str, pq.ParquetWriter] = {}
    paths: dict[str, Path] = {}
    partial_paths: dict[str, Path] = {}
    row_counts = {name: 0 for name in scorers}
    chunk_counts = {name: 0 for name in scorers}
    completed = False
    try:
        for name, _scorer in scorers.items():
            path = directory / f"{name}_test_scores.parquet"
            partial = path.with_suffix(path.suffix + ".partial")
            if partial.exists():
                partial.unlink()
            paths[name] = path
            partial_paths[name] = partial
            writers[name] = pq.ParquetWriter(partial, SCORED_SCHEMA, compression="SNAPPY")
        for frame in chunk_factory():
            if frame.empty:
                continue
            for name, scorer in scorers.items():
                table = _scored_table(frame, scorer(frame))
                writers[name].write_table(table)
                row_counts[name] += int(len(frame))
                chunk_counts[name] += 1
        if any(row_counts[name] <= 0 for name in scorers):
            empty_name = next(name for name in scorers if row_counts[name] <= 0)
            raise MinuteV2Error(f"minute_v2_stream_score_empty:{empty_name}")
        for writer in writers.values():
            writer.close()
        writers.clear()
        _commit_staged_score_files(paths, partial_paths)
        completed = True
    finally:
        for writer in writers.values():
            writer.close()
        if not completed:
            for partial in partial_paths.values():
                if partial.exists():
                    partial.unlink()
    result: dict[str, dict[str, int | str]] = {}
    for name, path in paths.items():
        result[name] = {
            "path": str(path),
            "row_count": int(row_counts[name]),
            "chunk_count": int(chunk_counts[name]),
            "size": int(path.stat().st_size),
        }
    return result


def _score_file_scan(path: Path) -> str:
    return _scan([Path(path).resolve()])


def _normalise_score_top_k(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise MinuteV2Error("minute_v2_score_top_k_invalid")
    normalised = int(value)
    if normalised <= 0:
        raise MinuteV2Error("minute_v2_score_top_k_invalid")
    return normalised


def _score_typed_projection(alias: str = "") -> str:
    """Project the narrow score contract with coercion matching pandas."""

    prefix = f"{alias}." if alias else ""
    expressions = {
        "symbol": f"CAST({prefix}symbol AS VARCHAR)",
        "trade_date": f"CAST({prefix}trade_date AS VARCHAR)",
        "bar_time": f"CAST({prefix}bar_time AS VARCHAR)",
        "score": f"TRY_CAST({prefix}score AS DOUBLE)",
        "label_net_return": f"TRY_CAST({prefix}label_net_return AS DOUBLE)",
        "planned_exit_date": f"CAST({prefix}planned_exit_date AS VARCHAR)",
        "entry_bar_time": f"CAST({prefix}entry_bar_time AS VARCHAR)",
        "entry_price": f"TRY_CAST({prefix}entry_price AS DOUBLE)",
        "entry_amount": f"TRY_CAST({prefix}entry_amount AS DOUBLE)",
        "entry_executable": f"TRY_CAST({prefix}entry_executable AS BOOLEAN)",
        "actual_exit_date": f"CAST({prefix}actual_exit_date AS VARCHAR)",
        "exit_amount": f"TRY_CAST({prefix}exit_amount AS DOUBLE)",
        "label_observed": f"TRY_CAST({prefix}label_observed AS BOOLEAN)",
    }
    return ",".join(f"{expression} AS {name}" for name, expression in expressions.items())


def _evaluate_score_file(path: Path, *, top_k: int = 3) -> dict[str, Any]:
    """Compute score metrics in DuckDB without loading the scored universe."""

    k = _normalise_score_top_k(top_k)
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET threads=2")
        connection.execute("SET memory_limit='2GB'")
        connection.execute("SET preserve_insertion_order=false")
        scan = _score_file_scan(path)
        typed_projection = _score_typed_projection()
        row = connection.execute(
            f"""
            WITH typed AS (
                SELECT {typed_projection}
                FROM {scan}
            ), current AS (
                SELECT symbol,trade_date,bar_time,score,label_net_return AS target
                FROM typed
                WHERE isfinite(score) AND isfinite(label_net_return)
            ), ranked_base AS (
                SELECT *,
                       COUNT(*) OVER w AS group_size,
                       RANK() OVER (w ORDER BY score) AS score_rank_min,
                       RANK() OVER (w ORDER BY target) AS target_rank_min,
                       COUNT(*) OVER (
                           PARTITION BY trade_date,bar_time,score
                       ) AS score_tie_count,
                       COUNT(*) OVER (
                           PARTITION BY trade_date,bar_time,target
                       ) AS target_tie_count,
                       ROW_NUMBER() OVER (
                           PARTITION BY trade_date,bar_time
                           ORDER BY score DESC,symbol
                       ) AS score_position
                FROM current
                WINDOW w AS (PARTITION BY trade_date,bar_time)
            ), ranked AS (
                SELECT *,
                       score_rank_min + (score_tie_count - 1) / 2.0 AS score_rank,
                       target_rank_min + (target_tie_count - 1) / 2.0 AS target_rank
                FROM ranked_base
            ),
            groups AS (
                SELECT trade_date,bar_time,MAX(group_size) AS group_size,
                       AVG(target) AS group_mean,
                       corr(CAST(score_rank AS DOUBLE),CAST(target_rank AS DOUBLE)) AS rank_ic
                FROM ranked GROUP BY trade_date,bar_time
            ),
            top_rows AS (
                SELECT r.*,g.group_mean
                FROM ranked r JOIN groups g USING(trade_date,bar_time)
                WHERE r.score_position <= {k}
            ),
            top_group_excess AS (
                SELECT trade_date,bar_time,
                       AVG(target - group_mean) AS top_excess
                FROM top_rows
                GROUP BY trade_date,bar_time
            ),
            daily AS (
                SELECT trade_date,AVG(target) AS day_top_mean
                FROM top_rows GROUP BY trade_date
            )
            SELECT
                (SELECT count(*) FROM current),
                (SELECT count(*) FROM groups),
                (SELECT AVG(rank_ic) FROM groups WHERE group_size >= 5),
                (SELECT AVG(CASE WHEN rank_ic > 0 THEN 1.0 ELSE 0.0 END)
                   FROM groups WHERE group_size >= 5 AND rank_ic IS NOT NULL),
                (SELECT AVG(target) FROM current),
                (SELECT AVG(group_mean) FROM groups),
                (SELECT AVG(target) FROM top_rows),
                (SELECT AVG(target - group_mean) FROM top_rows),
                (SELECT AVG(CASE WHEN top_excess > 0 THEN 1.0 ELSE 0.0 END)
                   FROM top_group_excess),
                (SELECT AVG(day_top_mean) FROM daily),
                (SELECT AVG(CASE WHEN day_top_mean > 0 THEN 1.0 ELSE 0.0 END)
                   FROM daily)
            """
        ).fetchone()
        if not row or row[0] is None or int(row[0]) == 0:
            raise MinuteV2Error("minute_v2_score_evaluation_empty")
        values = list(row)
        return {
            "row_count": int(values[0]),
            "group_count": int(values[1]),
            "rank_ic_mean": float(values[2]) if values[2] is not None else None,
            "rank_ic_positive_fraction": float(values[3]) if values[3] is not None else None,
            "universe_row_mean_net_return": float(values[4]) if values[4] is not None else None,
            "universe_group_mean_net_return": float(values[5]) if values[5] is not None else None,
            "top_k": k,
            "top_k_mean_net_return": float(values[6]) if values[6] is not None else None,
            "top_k_mean_excess_over_group_mean": float(values[7]) if values[7] is not None else None,
            "top_k_positive_excess_group_fraction": float(values[8]) if values[8] is not None else None,
            "daily_top_k_mean_net_return": float(values[9]) if values[9] is not None else None,
            "positive_day_fraction": float(values[10]) if values[10] is not None else None,
            "evaluation_storage": "DuckDB external scan of temporary scored Parquet",
        }
    finally:
        connection.close()


def _load_top_scored(path: Path, *, top_k: int = 3) -> pd.DataFrame:
    k = _normalise_score_top_k(top_k)
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET threads=2")
        connection.execute("SET memory_limit='2GB'")
        connection.execute("SET preserve_insertion_order=false")
        scan = _score_file_scan(path)
        columns = ",".join(SCORED_COLUMNS)
        typed_projection = _score_typed_projection()
        frame = connection.execute(
            f"""
            SELECT {columns}
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY trade_date,bar_time ORDER BY score DESC,symbol
                ) AS score_position
                FROM (
                    SELECT {typed_projection}
                    FROM {scan}
                ) typed
                WHERE isfinite(score) AND isfinite(label_net_return)
            ) ranked
            WHERE score_position <= {k}
            ORDER BY trade_date,bar_time,score DESC,symbol
            """
        ).fetchdf()
    finally:
        connection.close()
    for name in KEY_COLUMNS:
        frame[name] = frame[name].astype(str)
    for name in ("planned_exit_date", "actual_exit_date"):
        if name in frame:
            frame[name] = frame[name].astype("string")
    for name in ("entry_executable", "label_observed"):
        if name in frame:
            frame[name] = frame[name].astype("boolean")
    for name in ("score", "label_net_return", "entry_price", "entry_amount", "exit_amount"):
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame


def _stream_evaluate_and_replay(
    chunk_factory: Callable[[], Iterable[pd.DataFrame]],
    scorer: Callable[[pd.DataFrame], np.ndarray | pd.Series],
    *,
    model_name: str,
    output_directory: Path,
    temp_directory: Path,
) -> dict[str, Any]:
    scored_path = temp_directory / f"{model_name}_test_scores.parquet"
    score_meta = _write_scored_period(chunk_factory, scorer, scored_path)
    try:
        return _finish_scored_evaluation(
            scored_path,
            score_meta=score_meta,
            model_name=model_name,
            output_directory=output_directory,
        )
    finally:
        if scored_path.exists():
            scored_path.unlink()


def _finish_scored_evaluation(
    scored_path: Path,
    *,
    score_meta: dict[str, int | str],
    model_name: str,
    output_directory: Path,
) -> dict[str, Any]:
    metrics = _evaluate_score_file(scored_path)
    top = _load_top_scored(scored_path, top_k=3)
    replay_result, daily, trades = replay_events(
        top,
        replay_config=EventReplayConfig(score_threshold=-1.0e30),
    )
    daily.to_parquet(output_directory / f"{model_name}_replay_daily.parquet", index=False)
    trades.to_parquet(output_directory / f"{model_name}_replay_trades.parquet", index=False)
    return {
        "score_metrics": metrics,
        "sampled_event_replay": replay_result,
        "streaming_score": {
            **score_meta,
            "top_rows_loaded_for_replay": int(len(top)),
            "temporary_score_file_removed": True,
        },
    }


def _finish_multiple_scored_evaluations(
    score_meta: dict[str, dict[str, int | str]],
    *,
    output_directory: Path,
) -> dict[str, dict[str, Any]]:
    evaluations: dict[str, dict[str, Any]] = {}
    for name, metadata in score_meta.items():
        path = Path(str(metadata["path"]))
        try:
            evaluations[name] = _finish_scored_evaluation(
                path,
                score_meta=metadata,
                model_name=name,
                output_directory=output_directory,
            )
        finally:
            if path.exists():
                path.unlink()
    return evaluations


def _evaluate_and_replay(
    test: pd.DataFrame,
    scores: np.ndarray | pd.Series,
    *,
    model_name: str,
    output_directory: Path,
) -> dict[str, Any]:
    scored = test.copy()
    scored["score"] = np.asarray(scores, dtype=float)
    metrics = evaluate_scores(scored)
    replay_result, daily, trades = replay_events(
        scored,
        replay_config=EventReplayConfig(score_threshold=-1.0e30),
    )
    daily.to_parquet(output_directory / f"{model_name}_replay_daily.parquet", index=False)
    trades.to_parquet(output_directory / f"{model_name}_replay_trades.parquet", index=False)
    return {"score_metrics": metrics, "sampled_event_replay": replay_result}


def _run_development_validation(
    dataset_root: Path,
    output_root: Path,
    spec: DevelopmentValidationSpec,
) -> dict[str, Any]:
    period_parts = _period_parts(
        dataset_root,
        spec.development_start_year,
        spec.validation_end_year,
    )
    if not period_parts:
        raise MinuteV2Error(
            "minute_v2_training_period_empty:"
            f"{spec.development_start_year}:{spec.validation_end_year}"
        )
    feature_storage = period_parts[0].feature_storage
    feature_names = model_feature_columns_for_storage(feature_storage)
    with TemporaryDirectory(
        prefix="development_validation_",
        dir=output_root,
    ) as temporary:
        temp = Path(temporary)

        def factory(
            *,
            start_year: int,
            end_year: int,
            include_execution: bool,
            label_end_exclusive: str | None = None,
            ordered: bool = False,
        ) -> Callable[[], Iterable[pd.DataFrame]]:
            return lambda: iter_period_frames(
                dataset_root,
                start_year=start_year,
                end_year=end_year,
                include_execution=include_execution,
                temp_directory=temp / "scan",
                batch_rows=DEFAULT_BATCH_ROWS,
                label_end_exclusive=label_end_exclusive,
                ordered=ordered,
            )

        development_factory = factory(
            start_year=spec.development_start_year,
            end_year=spec.development_end_year,
            include_execution=False,
            label_end_exclusive=spec.development_label_end_exclusive,
        )
        validation_factory = factory(
            start_year=spec.validation_start_year,
            end_year=spec.validation_end_year,
            include_execution=True,
        )
        development_sample, development_sample_meta = _collect_cross_section_sample(
            development_factory
        )
        ridge, ridge_meta = fit_ridge_chunks(
            development_factory,
            feature_names=feature_names,
            alpha=10.0,
            median_sample_size=DEFAULT_MEDIAN_SAMPLE_ROWS,
        )
        ridge.save(output_root / "ridge.json")
        lightgbm_model, lightgbm_meta = fit_lightgbm_ranker(
            development_sample,
            validation=None,
            feature_names=feature_names,
        )
        lightgbm_meta["sampling"] = {
            "development": development_sample_meta,
            "validation": {
                "status": "not_used",
                "reason": "2025 is held out from fitting and early stopping",
            },
        }
        lightgbm_meta["early_stopping_used"] = False
        lightgbm_model.booster_.save_model(str(output_root / "lightgbm.txt"))
        score_meta = _write_multiple_scored_periods(
            validation_factory,
            {
                "rule": rule_score,
                "ridge": ridge.predict,
                "lightgbm": lambda frame, model=lightgbm_model: model.predict(
                    frame.loc[:, list(feature_names)]
                ),
            },
            temp / "scores",
        )
        evaluations = _finish_multiple_scored_evaluations(
            score_meta,
            output_directory=output_root,
        )
        validation_rows = int(next(iter(score_meta.values()))["row_count"])
        result = {
            "schema": "quantlab.minute_v2_development_validation/1",
            "status": "ok",
            "dataset_root": str(dataset_root.resolve()),
            "year_policy": {
                **spec.as_dict(),
                "development_years": list(DEVELOPMENT_YEARS),
                "final_validation_year": FINAL_VALIDATION_YEAR,
                "independent_test_set": False,
                "validation_used_for_fitting_or_early_stopping": False,
            },
            "period_samples": {
                "development": {
                    **development_sample_meta,
                    "effective_label_rows": int(ridge_meta["row_count"]),
                    "streaming_ridge": ridge_meta,
                    "label_end_exclusive": spec.development_label_end_exclusive,
                },
                "validation": {
                    "row_count": validation_rows,
                    "sampling_policy": (
                        "all causal candidate rows; streamed score evaluation"
                    ),
                    "feature_source": (
                        "base_joined_by_symbol_trade_date_bar_time"
                    ),
                    "held_out_from_fitting": True,
                    "used_for_early_stopping": False,
                },
            },
            "feature_storage": feature_storage,
            "feature_names": list(feature_names),
            "lightgbm": lightgbm_meta,
            "evaluations": evaluations,
            "available_memory_gib_after_validation": (
                psutil.virtual_memory().available / 1024**3
            ),
        }
        write_json(output_root / "result.json", result)
    del development_sample, ridge, lightgbm_model
    gc.collect()
    return result


def run_development_validation_baselines(
    dataset_root: str | Path,
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    dataset = Path(dataset_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if psutil.virtual_memory().available < MIN_TRAINING_AVAILABLE_GIB * 1024**3:
        raise MinuteV2Error("minute_v2_training_requires_two_gib_available_memory")
    return _run_development_validation(
        dataset,
        output,
        DEVELOPMENT_VALIDATION_SPEC,
    )


__all__ = [
    "DEVELOPMENT_VALIDATION_SPEC",
    "DEVELOPMENT_YEARS",
    "FINAL_VALIDATION_YEAR",
    "DevelopmentValidationSpec",
    "run_development_validation_baselines",
]
