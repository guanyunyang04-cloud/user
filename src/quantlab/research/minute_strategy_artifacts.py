"""Persistent narrow artifacts for the causal minute-strategy study.

The historical minute-strategy runner writes a convenient wide outcome row for
each strategy signal.  That shape is useful for backwards compatibility but
repeats the same forward price path whenever several rules refer to the same
symbol and minute.  This module splits a date partition into three narrow,
stable tables:

``events``
    One row per logical strategy/control signal, containing causal fields and
    execution information but no forward-return columns.
``paths``
    One row per unique symbol/date/time execution path, containing the shared
    forward outcomes.
``references``
    The mapping from a logical signal to its shared path, including the
    cross-sectional control reference when present.

The wide ``outcomes.parquet`` files remain the compatibility source.  The
normalized tables are deliberately additive so old records and callers can be
read without migration.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.research.minute_ma import normalize_bar_time
from quantlab.research.minute_ma_strategies import SIGNAL_COLUMNS


class MinuteStrategyArtifactError(ValueError):
    """Raised when a normalized minute-strategy artifact is invalid."""


PATH_KEY_COLUMNS = (
    "symbol",
    "signal_date",
    "signal_time",
    "signal_executable",
)

ENTRY_COLUMNS = (
    "entry_date",
    "entry_time",
    "entry_price",
    "entry_adjusted_price",
    "entry_observed",
    "entry_executable",
    "entry_reason",
    "entry_bar_index",
)

PATH_RUNTIME_COLUMNS = (
    *ENTRY_COLUMNS,
    "mfe_same_day",
    "mae_same_day",
    "same_day_observed",
    "t1_exit_date",
    "t1_exit_time",
    "t1_exit_adjusted_price",
    "t1_exit_observed",
    "t1_exit_reason",
    "t1_gross_return",
    "t1_net_return",
)

REFERENCE_COLUMNS = (
    "signal_id",
    "path_id",
    "strategy_id",
    "symbol",
    "signal_date",
    "signal_time",
    "reference_signal_id",
    "reference_symbol",
    "liquidity_match_ratio",
)

NORMALIZED_SCHEMA = "quantlab.minute_strategy_normalized_artifacts/1"

EVENT_SIGNAL_COLUMNS = tuple(
    name for name in SIGNAL_COLUMNS if name not in set(ENTRY_COLUMNS)
)


def _metric_columns(columns: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        name
        for name in columns
        if str(name).startswith(("gross_return_", "net_return_"))
    )


def _normalise_key_series(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        raise MinuteStrategyArtifactError(f"minute_strategy_artifact_column_missing:{name}")
    values = frame[name].astype("string").fillna("")
    if name == "signal_time":
        values = values.map(normalize_bar_time).astype("string")
    return values


def _path_ids(frame: pd.DataFrame) -> pd.Series:
    """Return collision-free IDs for the fixed A-share path key."""

    for name in PATH_KEY_COLUMNS:
        if name not in frame.columns:
            raise MinuteStrategyArtifactError(
                f"minute_strategy_artifact_column_missing:{name}"
            )
    symbol = _normalise_key_series(frame, "symbol")
    if symbol.str.contains("|", regex=False).any():
        raise MinuteStrategyArtifactError("minute_strategy_artifact_symbol_separator")
    date_values = _normalise_key_series(frame, "signal_date")
    time_values = _normalise_key_series(frame, "signal_time")
    if date_values.str.contains("|", regex=False).any() or time_values.str.contains(
        "|", regex=False
    ).any():
        raise MinuteStrategyArtifactError("minute_strategy_artifact_key_separator")
    executable = (
        frame["signal_executable"].fillna(False).astype(bool).astype("int8").astype("string")
    )
    return symbol + "|" + date_values + "|" + time_values + "|" + executable


def _ordered_columns(frame: pd.DataFrame, preferred: Sequence[str]) -> list[str]:
    available = set(frame.columns)
    return list(dict.fromkeys(name for name in preferred if name in available))


def split_outcome_frame(
    outcomes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split one wide outcome frame into events, paths and references.

    The function is pure and intended for small contract tests as well as
    callers that already hold one date in memory.  The production conversion
    path uses :func:`write_normalized_partition`, which writes each narrow
    table sequentially to keep the peak lower.
    """

    if not isinstance(outcomes, pd.DataFrame):
        raise MinuteStrategyArtifactError("minute_strategy_artifact_frame_invalid")
    required = {"signal_id", "strategy_id", *PATH_KEY_COLUMNS}
    missing = sorted(required.difference(outcomes.columns))
    if missing:
        raise MinuteStrategyArtifactError(
            f"minute_strategy_artifact_columns_missing:{','.join(missing)}"
        )
    if outcomes["signal_id"].duplicated().any():
        raise MinuteStrategyArtifactError("minute_strategy_artifact_signal_id_duplicate")

    working = outcomes.copy()
    working["path_id"] = _path_ids(working)

    metric_columns = _metric_columns(working.columns)
    event_preferred = ("path_id", *EVENT_SIGNAL_COLUMNS)
    path_preferred = (
        "path_id",
        *PATH_KEY_COLUMNS,
        "canonical_signal_id",
        *PATH_RUNTIME_COLUMNS,
        *metric_columns,
    )
    reference_preferred = REFERENCE_COLUMNS

    events = working.loc[:, _ordered_columns(working, event_preferred)].copy()
    # ``path_id`` is built from the canonical key representation (notably the
    # normalized minute encoding), so deduplicate on it rather than on raw
    # source spellings that may represent the same path.
    paths = working.loc[~working["path_id"].duplicated(keep="first")].copy()
    paths["canonical_signal_id"] = paths["signal_id"].astype("string")
    paths = paths.loc[:, _ordered_columns(paths, path_preferred)].copy()
    references = working.loc[:, _ordered_columns(working, reference_preferred)].copy()

    events.sort_values(["signal_date", "signal_time", "strategy_id", "symbol", "signal_id"], kind="stable", inplace=True)
    paths.sort_values(list(PATH_KEY_COLUMNS), kind="stable", inplace=True)
    references.sort_values(["signal_date", "signal_time", "strategy_id", "signal_id"], kind="stable", inplace=True)
    events.reset_index(drop=True, inplace=True)
    paths.reset_index(drop=True, inplace=True)
    references.reset_index(drop=True, inplace=True)

    if not paths["path_id"].is_unique:
        raise MinuteStrategyArtifactError("minute_strategy_artifact_path_id_duplicate")
    if not references["signal_id"].is_unique:
        raise MinuteStrategyArtifactError("minute_strategy_artifact_reference_signal_duplicate")
    return events, paths, references


def _write_parquet_atomic(frame: pd.DataFrame, path: Path, *, compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(
        partial,
        index=False,
        compression=None if compression == "none" else compression,
    )
    os.replace(partial, path)


def write_normalized_partition(
    outcomes: pd.DataFrame,
    output_root: str | Path,
    trade_date: str,
    *,
    compression: str = "zstd",
) -> dict[str, int | str]:
    """Write one date's three normalized tables sequentially."""

    date_text = str(trade_date).strip()
    if not date_text:
        raise MinuteStrategyArtifactError("minute_strategy_artifact_trade_date_invalid")
    try:
        if date.fromisoformat(date_text).isoformat() != date_text:
            raise ValueError
    except ValueError as exc:
        raise MinuteStrategyArtifactError(
            "minute_strategy_artifact_trade_date_invalid"
        ) from exc
    root = Path(output_root).resolve()
    # Build only one narrow table at a time.  This avoids retaining three
    # copies of a several-hundred-thousand-row date in the production runner.
    if not isinstance(outcomes, pd.DataFrame):
        raise MinuteStrategyArtifactError("minute_strategy_artifact_frame_invalid")
    required = {"signal_id", "strategy_id", *PATH_KEY_COLUMNS}
    missing = sorted(required.difference(outcomes.columns))
    if missing:
        raise MinuteStrategyArtifactError(
            f"minute_strategy_artifact_columns_missing:{','.join(missing)}"
        )
    if outcomes["signal_id"].duplicated().any():
        raise MinuteStrategyArtifactError("minute_strategy_artifact_signal_id_duplicate")
    source_dates = outcomes["signal_date"].astype("string").str.strip().str.slice(0, 10)
    if source_dates.isna().any() or source_dates.ne(date_text).any():
        raise MinuteStrategyArtifactError(
            f"minute_strategy_artifact_trade_date_mismatch:{date_text}"
        )
    path_ids = _path_ids(outcomes)
    metric_columns = _metric_columns(outcomes.columns)
    destinations = {
        "events": root / "events" / f"date={date_text}" / "events.parquet",
        "paths": root / "paths" / f"date={date_text}" / "paths.parquet",
        "references": root / "references" / f"date={date_text}" / "references.parquet",
    }
    event_columns = _ordered_columns(outcomes, EVENT_SIGNAL_COLUMNS)
    events = outcomes.loc[:, event_columns].copy()
    events.insert(0, "path_id", path_ids.to_numpy())
    events.sort_values(
        ["signal_date", "signal_time", "strategy_id", "symbol", "signal_id"],
        kind="stable",
        inplace=True,
    )
    _write_parquet_atomic(events, destinations["events"], compression=compression)
    event_rows = int(len(events))
    del events

    unique_mask = ~path_ids.duplicated(keep="first")
    path_columns = _ordered_columns(
        outcomes,
        (*PATH_KEY_COLUMNS, *PATH_RUNTIME_COLUMNS, *metric_columns),
    )
    paths = outcomes.loc[unique_mask, path_columns].copy()
    paths.insert(0, "path_id", path_ids.loc[unique_mask].to_numpy())
    paths.insert(1, "canonical_signal_id", outcomes.loc[unique_mask, "signal_id"].astype("string").to_numpy())
    paths.sort_values(list(PATH_KEY_COLUMNS), kind="stable", inplace=True)
    _write_parquet_atomic(paths, destinations["paths"], compression=compression)
    path_rows = int(len(paths))
    del paths, unique_mask

    reference_columns = _ordered_columns(outcomes, ("signal_id", *REFERENCE_COLUMNS[2:]))
    references = outcomes.loc[:, reference_columns].copy()
    references.insert(1, "path_id", path_ids.to_numpy())
    # ``signal_id`` is already the first source column; inserting at index 1
    # keeps the public mapping order stable without another wide copy.
    references = references.loc[:, _ordered_columns(references, REFERENCE_COLUMNS)].copy()
    references.sort_values(
        ["signal_date", "signal_time", "strategy_id", "signal_id"],
        kind="stable",
        inplace=True,
    )
    _write_parquet_atomic(references, destinations["references"], compression=compression)
    reference_rows = int(len(references))
    return {
        "trade_date": date_text,
        "event_rows": event_rows,
        "path_rows": path_rows,
        "reference_rows": reference_rows,
    }


def normalized_partition_paths(
    output_root: str | Path,
    kind: str,
) -> list[Path]:
    """List normalized date partitions for ``kind`` in deterministic order."""

    if kind not in {"events", "paths", "references"}:
        raise MinuteStrategyArtifactError("minute_strategy_artifact_kind_invalid")
    root = Path(output_root).resolve()
    return sorted((root / kind).glob("date=*/" + kind + ".parquet"))


def normalized_partition_complete(output_root: str | Path, trade_date: str) -> bool:
    """Return whether all three normalized tables exist for one date."""

    root = Path(output_root).resolve()
    date_text = str(trade_date)
    return all(
        (root / kind / f"date={date_text}" / f"{kind}.parquet").is_file()
        for kind in ("events", "paths", "references")
    )


def _source_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return [str(value) for value in pq.ParquetFile(path).schema.names]
    except (ImportError, OSError, ValueError) as exc:
        raise MinuteStrategyArtifactError(
            f"minute_strategy_artifact_schema_unreadable:{path}"
        ) from exc


def _memory_check(memory_floor_gib: float) -> None:
    try:
        import psutil

        available = int(psutil.virtual_memory().available)
    except ImportError as exc:  # pragma: no cover - dependency is project-wide
        raise MinuteStrategyArtifactError("minute_strategy_artifact_psutil_required") from exc
    floor = int(float(memory_floor_gib) * (1024**3))
    if available < floor:
        raise MinuteStrategyArtifactError(
            f"minute_strategy_artifact_memory_floor_breached:available={available}:floor={floor}"
        )


def _read_columns_for_normalization(path: Path) -> list[str]:
    source_columns = _source_columns(path)
    available = set(source_columns)
    required = {"signal_id", "strategy_id", *PATH_KEY_COLUMNS}
    missing = sorted(required.difference(available))
    if missing:
        raise MinuteStrategyArtifactError(
            f"minute_strategy_artifact_columns_missing:{','.join(missing)}:{path}"
        )
    # Retain all signal fields and all outcome metrics/runtime fields that are
    # actually present.  The projection is much narrower than the source's
    # auxiliary columns when older outputs contain extra diagnostics.
    preferred = (
        *SIGNAL_COLUMNS,
        *ENTRY_COLUMNS,
        "mfe_same_day",
        "mae_same_day",
        "same_day_observed",
        "t1_exit_date",
        "t1_exit_time",
        "t1_exit_adjusted_price",
        "t1_exit_observed",
        "t1_exit_reason",
        "t1_gross_return",
        "t1_net_return",
    )
    # Preserve the source schema order for any horizon columns not listed
    # explicitly above; iterating a set would make migration output unstable.
    preferred += tuple(
        name
        for name in source_columns
        if name.startswith(("gross_return_", "net_return_"))
    )
    return [name for name in dict.fromkeys(preferred) if name in available]


def materialize_normalized_artifacts(
    source_root: str | Path,
    output_root: str | Path | None = None,
    *,
    force: bool = False,
    memory_floor_gib: float = 0.5,
    compression: str = "zstd",
) -> dict[str, Any]:
    """Convert existing wide date outputs into normalized partitions.

    Conversion is date-at-a-time and never loads the complete month into
    memory.  Existing output files are untouched.  This function is also the
    migration path for historical runs created before normalized artifacts
    were enabled in the development runner.
    """

    source = Path(source_root).resolve()
    target = (Path(output_root).resolve() if output_root is not None else source / "normalized")
    source_paths = sorted((source / "outcomes").glob("date=*/outcomes.parquet"))
    if not source_paths:
        raise MinuteStrategyArtifactError("minute_strategy_artifact_source_empty")
    target.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    partitions: list[dict[str, int | str]] = []
    skipped = 0
    for path in source_paths:
        _memory_check(memory_floor_gib)
        date_text = path.parent.name.split("=", 1)[-1]
        destination_probes = [
            target / kind / f"date={date_text}" / f"{kind}.parquet"
            for kind in ("events", "paths", "references")
        ]
        if all(probe.is_file() for probe in destination_probes) and not force:
            skipped += 1
            continue
        columns = _read_columns_for_normalization(path)
        outcomes = pd.read_parquet(path, columns=columns)
        stats = write_normalized_partition(
            outcomes,
            target,
            date_text,
            compression=compression,
        )
        partitions.append(stats)
        del outcomes
        _memory_check(memory_floor_gib)

    counts = {
        "event_rows": int(sum(int(item["event_rows"]) for item in partitions)),
        "path_rows": int(sum(int(item["path_rows"]) for item in partitions)),
        "reference_rows": int(sum(int(item["reference_rows"]) for item in partitions)),
    }
    manifest = {
        "schema": NORMALIZED_SCHEMA,
        "status": "ok",
        "source_root": str(source),
        "output_root": str(target),
        "source_partition_count": len(source_paths),
        "written_partition_count": len(partitions),
        "skipped_partition_count": int(skipped),
        "path_key_columns": list(PATH_KEY_COLUMNS),
        "counts": counts,
        "partitions": partitions,
        "compression": compression,
        "memory_floor_gib": float(memory_floor_gib),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    partial = target / "normalized_manifest.json.partial"
    partial.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, target / "normalized_manifest.json")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantlab minute-strategy-normalize")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--memory-floor-gib", type=float, default=0.5)
    parser.add_argument("--compression", choices=("snappy", "zstd", "gzip", "brotli", "lz4", "none"), default="zstd")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    result = materialize_normalized_artifacts(
        args.source_root,
        args.output_root,
        force=bool(args.force),
        memory_floor_gib=float(args.memory_floor_gib),
        compression=args.compression,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ENTRY_COLUMNS",
    "EVENT_SIGNAL_COLUMNS",
    "MinuteStrategyArtifactError",
    "NORMALIZED_SCHEMA",
    "PATH_KEY_COLUMNS",
    "PATH_RUNTIME_COLUMNS",
    "REFERENCE_COLUMNS",
    "materialize_normalized_artifacts",
    "normalized_partition_complete",
    "normalized_partition_paths",
    "split_outcome_frame",
    "write_normalized_partition",
]
