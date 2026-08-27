"""Two-fold, bounded-memory baseline training for minute-v2 events."""

from __future__ import annotations

import gc
import os
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from quantlab.core.io import read_json, write_json

from .contracts import KEY_COLUMNS, MODEL_FEATURE_COLUMNS, MinuteV2Error
from .mining import mine_formula_features, write_mining_result
from .models import (
    evaluate_scores,
    fit_lightgbm_ranker,
    fit_ridge_chunks,
    rule_score,
)
from .replay import EventReplayConfig, replay_events


@dataclass(frozen=True)
class FoldSpec:
    fold: int
    train_start_year: int
    train_end_year: int
    validation_year: int
    test_start_year: int
    test_end_year: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


TWO_FOLDS = (
    FoldSpec(1, 2012, 2017, 2018, 2019, 2020),
    FoldSpec(2, 2012, 2019, 2020, 2021, 2022),
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

MONTH_SCHEMA = "quantlab.minute_v2_month/2"
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


def _period_parts(dataset_root: Path, start_year: int, end_year: int) -> list[PeriodPart]:
    parts: list[PeriodPart] = []
    for year in range(int(start_year), int(end_year) + 1):
        for month in range(1, 13):
            directory = dataset_root / "months" / f"year={year:04d}" / f"month={month:02d}"
            manifest_path = directory / "manifest.json"
            if not manifest_path.is_file():
                raise MinuteV2Error(f"minute_v2_training_manifest_missing:{year:04d}-{month:02d}")
            manifest = read_json(manifest_path)
            if manifest.get("status") != "ok" or manifest.get("schema") != MONTH_SCHEMA:
                raise MinuteV2Error(f"minute_v2_training_manifest_not_ok:{manifest_path}")
            artifacts = dict(manifest.get("artifacts", {}))
            def artifact_path(name: str) -> Path:
                value = Path(str(dict(artifacts.get(name, {})).get("path", "")))
                return value if value.is_absolute() else (directory / value).resolve()

            event_path = artifact_path("events")
            label_path = artifact_path("labels")
            if not event_path.is_file() or not label_path.is_file():
                raise MinuteV2Error(
                    f"minute_v2_training_artifact_missing:{year:04d}-{month:02d}"
                )
            base_path = artifact_path("base")
            if not base_path.is_file():
                raise MinuteV2Error(
                    f"minute_v2_training_base_missing:{year:04d}-{month:02d}"
                )
            parts.append(
                PeriodPart(
                    year=int(year),
                    month=int(month),
                    events=event_path.resolve(),
                    labels=label_path.resolve(),
                    base=base_path.resolve(),
                )
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
    feature_projection = ",".join(f"b.{name}" for name in MODEL_FEATURE_COLUMNS)
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
        JOIN {label_scan} l USING(symbol,trade_date,bar_time)
        WHERE l.label_observed {boundary}
        {order}
    """


def _normalise_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    for name in KEY_COLUMNS:
        frame[name] = frame[name].astype(str)
    for name in MODEL_FEATURE_COLUMNS:
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
                frame = _normalise_training_frame(batch.to_pandas())
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
        completed = True
    finally:
        if writer is not None:
            writer.close()
        if not completed and partial.exists():
            partial.unlink()
    os.replace(partial, target)
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
        for name, scorer in scorers.items():
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
        if row_counts[name] <= 0:
            raise MinuteV2Error(f"minute_v2_stream_score_empty:{name}")
        partial = path.with_suffix(path.suffix + ".partial")
        os.replace(partial, path)
        result[name] = {
            "path": str(path),
            "row_count": int(row_counts[name]),
            "chunk_count": int(chunk_counts[name]),
            "size": int(path.stat().st_size),
        }
    return result


def _score_file_scan(path: Path) -> str:
    return _scan([Path(path).resolve()])


def _evaluate_score_file(path: Path, *, top_k: int = 3) -> dict[str, Any]:
    """Compute score metrics in DuckDB without loading the scored universe."""

    k = int(top_k)
    if k <= 0:
        raise MinuteV2Error("minute_v2_score_top_k_invalid")
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET threads=2")
        connection.execute("SET memory_limit='2GB'")
        connection.execute("SET preserve_insertion_order=false")
        scan = _score_file_scan(path)
        row = connection.execute(
            f"""
            WITH current AS (
                SELECT symbol,trade_date,bar_time,
                       CAST(score AS DOUBLE) AS score,
                       CAST(label_net_return AS DOUBLE) AS target
                FROM {scan}
                WHERE isfinite(CAST(score AS DOUBLE))
                  AND isfinite(CAST(label_net_return AS DOUBLE))
            ),
            ranked AS (
                SELECT *,
                       COUNT(*) OVER w AS group_size,
                       RANK() OVER (w ORDER BY score) AS score_rank,
                       RANK() OVER (w ORDER BY target) AS target_rank,
                       ROW_NUMBER() OVER (
                           PARTITION BY trade_date,bar_time
                           ORDER BY score DESC,symbol
                       ) AS score_position
                FROM current
                WINDOW w AS (PARTITION BY trade_date,bar_time)
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
                (SELECT AVG(CASE WHEN target - group_mean > 0 THEN 1.0 ELSE 0.0 END)
                   FROM top_rows),
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
    k = int(top_k)
    if k <= 0:
        raise MinuteV2Error("minute_v2_score_top_k_invalid")
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET threads=2")
        connection.execute("SET memory_limit='2GB'")
        connection.execute("SET preserve_insertion_order=false")
        scan = _score_file_scan(path)
        columns = ",".join(SCORED_COLUMNS)
        frame = connection.execute(
            f"""
            SELECT {columns}
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY trade_date,bar_time ORDER BY score DESC,symbol
                ) AS score_position
                FROM {scan}
                WHERE isfinite(CAST(score AS DOUBLE))
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


def _run_fold(
    dataset_root: Path,
    output_root: Path,
    spec: FoldSpec,
    *,
    mining_result: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    directory = output_root / f"fold_{spec.fold}"
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f"fold_{spec.fold}_", dir=output_root) as temporary:
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

        train_factory = factory(
            start_year=spec.train_start_year,
            end_year=spec.train_end_year,
            include_execution=False,
            label_end_exclusive=f"{spec.validation_year:04d}-01-01",
        )
        validation_factory = factory(
            start_year=spec.validation_year,
            end_year=spec.validation_year,
            include_execution=False,
            label_end_exclusive=f"{spec.test_start_year:04d}-01-01",
        )
        test_factory = factory(
            start_year=spec.test_start_year,
            end_year=spec.test_end_year,
            include_execution=True,
            ordered=False,
        )
        train_sample, train_sample_meta = _collect_cross_section_sample(train_factory)
        validation_sample, validation_sample_meta = _collect_cross_section_sample(
            validation_factory
        )
        ridge, ridge_meta = fit_ridge_chunks(
            train_factory,
            alpha=10.0,
            median_sample_size=DEFAULT_MEDIAN_SAMPLE_ROWS,
        )
        if spec.fold == 1 and mining_result is None:
            mining_result = mine_formula_features(
                train_sample,
                validation_sample,
                maximum_candidates=120,
                maximum_selected=12,
                minimum_coverage=0.85,
            )
            mining_result["train_rows"] = int(ridge_meta["row_count"])
            mining_result["validation_rows"] = int(validation_sample_meta["sample_rows"])
            mining_result["sampling"] = {
                "train": train_sample_meta,
                "validation": validation_sample_meta,
            }
            write_mining_result(mining_result, output_root / "formula_mining_fold_1.json")
        ridge.save(directory / "ridge.json")
        lightgbm, lightgbm_meta = fit_lightgbm_ranker(
            train_sample,
            validation=validation_sample,
        )
        lightgbm_meta["sampling"] = {
            "train": train_sample_meta,
            "validation": validation_sample_meta,
        }
        lightgbm.booster_.save_model(str(directory / "lightgbm.txt"))
        score_meta = _write_multiple_scored_periods(
            test_factory,
            {
                "rule": rule_score,
                "ridge": ridge.predict,
                "lightgbm": lambda frame: lightgbm.predict(
                    frame.loc[:, list(MODEL_FEATURE_COLUMNS)]
                ),
            },
            temp / "scores",
        )
        evaluations = _finish_multiple_scored_evaluations(
            score_meta,
            output_directory=directory,
        )
        test_rows = int(next(iter(score_meta.values()))["row_count"])
        result = {
            "schema": "quantlab.minute_v2_fold/2",
            "status": "ok",
            "fold": spec.as_dict(),
            "period_samples": {
                "train": {
                    **train_sample_meta,
                    "effective_label_rows": int(ridge_meta["row_count"]),
                    "streaming_ridge": ridge_meta,
                },
                "validation": validation_sample_meta,
                "test": {
                    "row_count": test_rows,
                    "sampling_policy": "all causal candidate rows; streamed score evaluation",
                    "feature_source": "base_joined_by_symbol_trade_date_bar_time",
                },
            },
            "feature_names": list(MODEL_FEATURE_COLUMNS),
            "lightgbm": lightgbm_meta,
            "evaluations": evaluations,
            "available_memory_gib_after_fold": psutil.virtual_memory().available / 1024**3,
        }
        write_json(directory / "result.json", result)
    del train_sample, validation_sample, ridge, lightgbm
    gc.collect()
    return result, mining_result


def run_two_fold_baselines(
    dataset_root: str | Path,
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    dataset = Path(dataset_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if psutil.virtual_memory().available < MIN_TRAINING_AVAILABLE_GIB * 1024**3:
        raise MinuteV2Error(
            "minute_v2_training_requires_two_gib_available_memory"
        )
    results: list[dict[str, Any]] = []
    mining_result: dict[str, Any] | None = None
    for spec in TWO_FOLDS:
        result, mining_result = _run_fold(
            dataset,
            output,
            spec,
            mining_result=mining_result,
        )
        results.append(result)
    aggregate = {
        "schema": "quantlab.minute_v2_two_fold/1",
        "status": "ok",
        "dataset_root": str(dataset),
        "fold_count": len(results),
        "sampling_policy": "all causal candidate rows at all decision minutes",
        "formula_mining": {
            "path": str(output / "formula_mining_fold_1.json"),
            "selected_count": int((mining_result or {}).get("selected_count", 0)),
            "used_final_fold": False,
        },
        "fold_results": [str(output / f"fold_{spec.fold}" / "result.json") for spec in TWO_FOLDS],
    }
    write_json(output / "result.json", aggregate)
    return aggregate


__all__ = ["FoldSpec", "TWO_FOLDS", "run_two_fold_baselines"]
