"""Chunked development runner for the hand-designed minute-MA strategies."""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from quantlab.data.qdp_v2.duckdb_resources import (
    DEFAULT_MEMORY_FLOOR_BYTES,
    open_guarded_duckdb,
)
from quantlab.research.minute_ma import MA_PERIODS, MinuteMAConfig
from quantlab.research.minute_ma_event_study import (
    OUTCOME_COLUMNS,
    EventStudyConfig,
    compute_event_outcomes,
)
from quantlab.research.minute_ma_strategies import (
    build_liquidity_matched_controls_many,
    build_strategy_signals_vectorized,
    strategy_catalog,
)

from .minute_strategy_data import (
    StrategyDataConfig,
    build_enriched_minute_ma_states,
    build_live_ma_alignment,
    build_minute_market_context,
    load_daily_context,
    load_hourly_history,
    load_point_in_time_universe,
    load_target_bars,
    load_trading_calendar,
)


class MinuteStrategyRunError(RuntimeError):
    """Raised when a chunked strategy run cannot satisfy its output contract."""


GIB = 1024**3


def _require_memory_floor(config: DevelopmentStudyConfig, stage: str) -> int:
    """Stop before another allocation when the requested RAM reserve is gone."""

    available = int(psutil.virtual_memory().available)
    floor = int(float(config.memory_floor_gib) * GIB)
    if available < floor:
        raise MinuteStrategyRunError(
            f"strategy_run_memory_floor_breached:{stage}:"
            f"available={available}:floor={floor}"
        )
    return available


@dataclass(frozen=True)
class DevelopmentStudyConfig:
    """Execution settings for one reproducible development run."""

    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"
    output_root: str = "runs/minute_ma_development"
    max_months: int = 0
    max_symbols: int = 0
    force: bool = False
    history_open_days: int = 66
    outcome_open_days: int = 5
    strategy_periods: tuple[int, ...] = MA_PERIODS
    signal_symbol_chunk_size: int = 300
    duckdb_threads: int | str = 2
    temp_directory: str | None = None
    memory_floor_gib: float = 0.5
    duckdb_memory_floor_gib: float = 2.0

    def validate(self) -> None:
        start = date.fromisoformat(str(self.start_date))
        end = date.fromisoformat(str(self.end_date))
        if start > end:
            raise MinuteStrategyRunError("strategy_run_date_range_invalid")
        for name, value, minimum in (
            ("max_months", self.max_months, 0),
            ("max_symbols", self.max_symbols, 0),
            ("history_open_days", self.history_open_days, 1),
            ("outcome_open_days", self.outcome_open_days, 1),
            ("signal_symbol_chunk_size", self.signal_symbol_chunk_size, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
                raise MinuteStrategyRunError(f"strategy_run_{name}_invalid")
        if isinstance(self.duckdb_threads, str):
            if self.duckdb_threads.strip().lower() != "auto":
                raise MinuteStrategyRunError("strategy_run_duckdb_threads_invalid")
        elif isinstance(self.duckdb_threads, bool) or not isinstance(self.duckdb_threads, (int, np.integer)) or int(self.duckdb_threads) < 1:
            raise MinuteStrategyRunError("strategy_run_duckdb_threads_invalid")
        if not np.isfinite(float(self.memory_floor_gib)) or float(self.memory_floor_gib) < 0.5:
            raise MinuteStrategyRunError("strategy_run_memory_floor_invalid")
        if (
            not np.isfinite(float(self.duckdb_memory_floor_gib))
            or float(self.duckdb_memory_floor_gib) < float(self.memory_floor_gib)
        ):
            raise MinuteStrategyRunError("strategy_run_duckdb_memory_floor_invalid")
        if not self.strategy_periods or tuple(sorted(set(self.strategy_periods))) != tuple(self.strategy_periods):
            raise MinuteStrategyRunError("strategy_run_periods_invalid")
        if any(int(value) not in MA_PERIODS for value in self.strategy_periods):
            raise MinuteStrategyRunError("strategy_run_period_unsupported")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "start_date": str(self.start_date),
            "end_date": str(self.end_date),
            "output_root": str(self.output_root),
            "max_months": int(self.max_months),
            "max_symbols": int(self.max_symbols),
            "force": bool(self.force),
            "history_open_days": int(self.history_open_days),
            "outcome_open_days": int(self.outcome_open_days),
            "strategy_periods": [int(value) for value in self.strategy_periods],
            "signal_symbol_chunk_size": int(self.signal_symbol_chunk_size),
            "duckdb_threads": self.duckdb_threads,
            "temp_directory": self.temp_directory,
            "memory_floor_gib": float(self.memory_floor_gib),
            "duckdb_memory_floor_gib": float(self.duckdb_memory_floor_gib),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return _jsonable(value.to_dict("records"))
    if isinstance(value, pd.Series):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _scan(paths: Sequence[Path]) -> str:
    if not paths:
        raise MinuteStrategyRunError("strategy_run_outcome_paths_empty")
    values = ",".join("'" + str(path).replace("'", "''") + "'" for path in paths)
    return f"read_parquet([{values}], union_by_name=true)"


def _month_ranges(calendar: Sequence[str], start_date: str, end_date: str) -> list[tuple[str, str, list[str]]]:
    selected = [str(value) for value in calendar if str(start_date) <= str(value) <= str(end_date)]
    grouped: list[tuple[str, str, list[str]]] = []
    for month, group in pd.Series(selected).groupby(pd.Series(selected).str[:7], sort=True):
        dates = group.astype(str).tolist()
        grouped.append((f"{month}-01", f"{month}-31", dates))
    return grouped


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def _month_state_path(output_root: Path, start_date: str) -> Path:
    return output_root / "months" / f"year={start_date[:4]}" / f"month={start_date[5:7]}"


def _completed_dates(checkpoint: Path) -> set[str]:
    if not checkpoint.is_file():
        return set()
    try:
        value = json.loads(checkpoint.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return set()
    dates = value.get("completed_dates", []) if isinstance(value, dict) else []
    return {str(item) for item in dates}


def _attach_alignment(states: pd.DataFrame, alignment: pd.DataFrame) -> pd.DataFrame:
    if alignment.empty:
        return states
    keys = ["symbol", "trade_date", "bar_time"]
    current = states.drop(columns=["ma_alignment_score", "bullish_ma_stack", "bearish_ma_stack"], errors="ignore")
    return current.merge(alignment, on=keys, how="left", validate="many_to_one")


def _decision_signal_filter(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals
    # A signal at the final 15:00 bar has no next-minute open.  Retain it in
    # the audit only when the caller explicitly asks for diagnostics; normal
    # strategy comparisons use bars with a possible next fill.
    return signals.loc[signals["signal_time"].astype(str) < "150000000"].copy()


def _build_day_signals(
    day_bars: pd.DataFrame,
    hourly_history: pd.DataFrame,
    *,
    daily_context: pd.DataFrame,
    ma_config: MinuteMAConfig,
    strategy_periods: Sequence[int],
    include_diagnostic: bool,
    liquidity_band: float,
    signal_symbol_chunk_size: int,
    random_seed: int = 7,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if day_bars.empty:
        return pd.DataFrame(), {"state_rows": 0, "signal_rows": 0, "control_reference_count": 0, "control_matched_count": 0}
    market_context = build_minute_market_context(day_bars, daily_context=daily_context)
    # Alignment is the only cross-period state.  Compute it once, then process
    # each MA period independently so the six expanded state tables are never
    # resident at the same time.
    alignment = build_live_ma_alignment(day_bars, hourly_history, config=ma_config)
    parts: list[pd.DataFrame] = []
    control_parts: list[pd.DataFrame] = []
    state_rows = 0
    control_reference_count = 0
    control_matched_count = 0
    control_unmatched_reasons: dict[str, int] = {}
    specs = [
        spec
        for spec in strategy_catalog(include_unavailable=True)
        if spec.implemented
        and not spec.reference_control
        and (include_diagnostic or spec.executable)
        and spec.signal_rule != "posthoc_catchup"
    ]
    for period in strategy_periods:
        period = int(period)
        if period not in ma_config.periods:
            continue
        period_config = MinuteMAConfig(
            periods=(period,),
            near_touch_bps=ma_config.near_touch_bps,
            posthoc_catchup_bps=ma_config.posthoc_catchup_bps,
            prior_touch_window_hours=ma_config.prior_touch_window_hours,
        )
        states = build_enriched_minute_ma_states(
            day_bars,
            hourly_history,
            daily_context=daily_context,
            context=market_context,
            config=period_config,
        )
        if states.empty:
            continue
        if not alignment.empty:
            states = _attach_alignment(states, alignment)
        state_rows += int(len(states))
        state_symbols = states["symbol"].astype(str).drop_duplicates().tolist()
        signal_parts: list[pd.DataFrame] = []
        for start in range(0, len(state_symbols), int(signal_symbol_chunk_size)):
            chunk_symbols = set(state_symbols[start : start + int(signal_symbol_chunk_size)])
            chunk_states = states.loc[states["symbol"].astype(str).isin(chunk_symbols)]
            chunk_signals = build_strategy_signals_vectorized(
                chunk_states,
                strategy_ids=[spec.strategy_id for spec in specs],
                include_diagnostic=include_diagnostic,
                random_seed=random_seed,
            )
            if not chunk_signals.empty:
                signal_parts.append(chunk_signals)
        signal = pd.concat(signal_parts, ignore_index=True) if signal_parts else pd.DataFrame()
        filtered = _decision_signal_filter(signal) if not signal.empty else signal
        if not filtered.empty:
            parts.append(filtered)
            reference_frame = filtered.loc[
                ~filtered["strategy_id"].isin(["s0_random_matched", "s0_liquidity_matched"])
                & filtered["signal_executable"].fillna(False)
            ].copy()
            if not reference_frame.empty:
                controls = build_liquidity_matched_controls_many(
                    states,
                    reference_frame,
                    liquidity_band_ratio=liquidity_band,
                    random_seed=random_seed,
                )
                control_reference_count += int(controls.attrs.get("reference_count", len(reference_frame)))
                control_matched_count += int(controls.attrs.get("matched_count", len(controls)))
                for name, value in controls.attrs.get("unmatched_reasons", {}).items():
                    control_unmatched_reasons[str(name)] = control_unmatched_reasons.get(str(name), 0) + int(value)
                if not controls.empty:
                    filtered_controls = _decision_signal_filter(controls)
                    if not filtered_controls.empty:
                        control_parts.append(filtered_controls)
        del states, signal, signal_parts
        gc.collect()
    if not parts:
        return pd.DataFrame(), {"state_rows": state_rows, "signal_rows": 0, "control_reference_count": control_reference_count, "control_matched_count": control_matched_count, "control_unmatched_reasons": control_unmatched_reasons}
    signals = pd.concat([*parts, *control_parts], ignore_index=True)
    return signals, {
        "state_rows": int(state_rows),
        "signal_rows": int(len(signals)),
        "control_reference_count": int(control_reference_count),
        "control_matched_count": int(control_matched_count),
        "control_unmatched_reasons": control_unmatched_reasons,
    }


def _lean_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    keep = [name for name in OUTCOME_COLUMNS if name in outcomes.columns]
    metric_names = [name for name in outcomes.columns if name.startswith(("gross_return_", "net_return_"))]
    keep.extend(name for name in ("mfe_same_day", "mae_same_day", "t1_gross_return", "t1_net_return") if name in outcomes.columns)
    keep.extend(name for name in metric_names if name not in keep)
    return outcomes.loc[:, list(dict.fromkeys(keep))].copy()


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _outcome_metric_columns(connection: Any, paths: Sequence[Path]) -> list[str]:
    schema = connection.execute(f"DESCRIBE SELECT * FROM {_scan(paths[:1])}").df()
    names = schema["column_name"].astype(str).tolist()
    return [
        name
        for name in names
        if name.startswith(("gross_return_", "net_return_"))
        or name in {"mfe_same_day", "mae_same_day", "t1_gross_return", "t1_net_return"}
    ]


def _stream_event_summary(
    connection: Any,
    paths: Sequence[Path],
    *,
    group_by: Sequence[str],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    """Aggregate outcome files in DuckDB without loading the whole panel."""

    if not paths:
        return []
    keys = tuple(str(value) for value in group_by)
    if not keys:
        raise MinuteStrategyRunError("strategy_summary_group_empty")
    source_keys = set(keys)
    expressions = [
        _quote_identifier("strategy_id"),
        _quote_identifier("ma_period"),
        _quote_identifier("signal_date"),
        _quote_identifier("diagnostic_only"),
        _quote_identifier("entry_observed"),
        _quote_identifier("entry_executable"),
    ]
    if "market_regime" in source_keys:
        expressions.append(_quote_identifier("market_regime"))
    if "year" in source_keys:
        expressions.append(
            "TRY_CAST(substr(CAST(\"signal_date\" AS VARCHAR), 1, 4) AS INTEGER) AS \"year\""
        )
    for name in metrics:
        expressions.append(_quote_identifier(name))
    group_sql = ", ".join(_quote_identifier(name) for name in keys)
    metric_sql = ", ".join(_quote_identifier(name) for name in metrics)
    observed_expr = "COALESCE(\"entry_observed\", FALSE)"
    executable_expr = "COALESCE(\"entry_executable\", FALSE)"
    query = f"""
        WITH source AS (
            SELECT {", ".join(expressions)}
            FROM {_scan(paths)}
            WHERE NOT COALESCE(\"diagnostic_only\", FALSE)
        ),
        counts AS (
            SELECT {group_sql},
                   COUNT(*) AS signal_count,
                   COUNT(*) FILTER (WHERE {observed_expr}) AS entry_observed_count,
                   COUNT(*) FILTER (WHERE {executable_expr}) AS entry_executable_count
            FROM source
            GROUP BY {group_sql}
        ),
        long_values AS (
            SELECT {group_sql}, metric, CAST(value AS DOUBLE) AS value
            FROM source
            UNPIVOT (value FOR metric IN ({metric_sql}))
        ),
        valid AS (
            SELECT * FROM long_values WHERE isfinite(value)
        ),
        stats AS (
            SELECT {group_sql}, metric,
                   COUNT(*) AS observed_count,
                   AVG(value) AS mean,
                   MEDIAN(value) AS median,
                   SUM(value) FILTER (WHERE value > 0) AS positive_sum,
                   SUM(-value) FILTER (WHERE value < 0) AS negative_abs,
                   COUNT(*) FILTER (WHERE value > 0) AS positive_count
            FROM valid
            GROUP BY {group_sql}, metric
        ),
        ranked AS (
            SELECT {group_sql}, metric, value,
                   CASE WHEN value > 0 THEN ROW_NUMBER() OVER (
                       PARTITION BY {group_sql}, metric ORDER BY value DESC
                   ) END AS positive_rank,
                   COUNT(*) FILTER (WHERE value > 0) OVER (
                       PARTITION BY {group_sql}, metric
                   ) AS positive_total
            FROM valid
        ),
        trimmed AS (
            SELECT {group_sql}, metric,
                   AVG(value) FILTER (
                       WHERE value <= 0 OR positive_rank > CEIL(positive_total * 0.01)
                   ) AS trimmed_mean
            FROM ranked
            GROUP BY {group_sql}, metric
        )
        SELECT counts.*, stats.metric, stats.observed_count, stats.mean,
               stats.median, stats.positive_sum, stats.negative_abs,
               stats.positive_count, trimmed.trimmed_mean
        FROM counts
        LEFT JOIN stats USING ({group_sql})
        LEFT JOIN trimmed USING ({group_sql}, metric)
        ORDER BY {group_sql}, stats.metric
    """
    result = connection.execute(query).fetchdf()
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in result.to_dict("records"):
        key = tuple(record.get(name) for name in keys)
        row = grouped.setdefault(
            key,
            {
                name: record.get(name)
                for name in keys
            }
            | {
                "signal_count": int(record.get("signal_count") or 0),
                "entry_observed_count": int(record.get("entry_observed_count") or 0),
                "entry_executable_count": int(record.get("entry_executable_count") or 0),
                "diagnostic_excluded": True,
            },
        )
        metric = record.get("metric")
        if metric is None:
            continue
        observed = int(record.get("observed_count") or 0)
        positive_count = int(record.get("positive_count") or 0)
        negative_abs = record.get("negative_abs")
        positive_sum = record.get("positive_sum")
        row[f"{metric}_observed_count"] = observed
        row[f"{metric}_mean"] = record.get("mean")
        row[f"{metric}_median"] = record.get("median")
        row[f"{metric}_win_rate"] = positive_count / observed if observed else None
        row[f"{metric}_profit_factor"] = (
            float(positive_sum) / float(negative_abs)
            if positive_sum is not None and negative_abs is not None and float(negative_abs) > 0
            else None
        )
        row[f"{metric}_trimmed_mean_top1pct_removed"] = record.get("trimmed_mean")
    return list(grouped.values())


def _stream_control_summaries(
    connection: Any,
    paths: Sequence[Path],
    strategy_ids: Sequence[str],
    *,
    metric: str = "net_return_60m",
) -> list[dict[str, Any]]:
    """Compute both control comparisons in one bounded DuckDB query."""

    if not paths or not strategy_ids:
        return []
    query = f"""
        WITH source AS (
            SELECT strategy_id, signal_id, symbol, signal_date,
                   signal_time, sixty_minute_bucket, ma_period,
                   reference_signal_id, TRY_CAST({_quote_identifier(metric)} AS DOUBLE) AS metric
            FROM {_scan(paths)}
        ),
        regular AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY strategy_id, symbol, signal_date, sixty_minute_bucket, ma_period
                ORDER BY signal_id
            ) AS rn
            FROM source
            WHERE strategy_id NOT IN ('s0_random_matched', 's0_liquidity_matched')
        ),
        random_control AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY symbol, signal_date, sixty_minute_bucket, ma_period
                ORDER BY signal_id
            ) AS rn
            FROM source WHERE strategy_id = 's0_random_matched'
        ),
        liquidity_control AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY reference_signal_id ORDER BY signal_id
            ) AS rn
            FROM source
            WHERE strategy_id = 's0_liquidity_matched'
              AND reference_signal_id IS NOT NULL
        ),
        same_pairs AS (
            SELECT r.strategy_id, 's0_random_matched' AS control_id,
                   r.metric - c.metric AS difference
            FROM regular r
            JOIN random_control c USING(symbol, signal_date, sixty_minute_bucket, ma_period)
            WHERE r.rn = 1 AND c.rn = 1
        ),
        liquidity_pairs AS (
            SELECT r.strategy_id, 's0_liquidity_matched' AS control_id,
                   r.metric - c.metric AS difference
            FROM regular r
            JOIN liquidity_control c ON c.reference_signal_id = r.signal_id
            WHERE r.rn = 1 AND c.rn = 1
        ),
        pairs AS (
            SELECT * FROM same_pairs
            UNION ALL
            SELECT * FROM liquidity_pairs
        )
        SELECT strategy_id, control_id,
               COUNT(*) AS matched_count,
               COUNT(*) FILTER (WHERE isfinite(difference)) AS observed_count,
               AVG(difference) AS mean_difference,
               MEDIAN(difference) AS median_difference,
               AVG(CASE WHEN isfinite(difference) THEN
                   CASE WHEN difference > 0 THEN 1.0 ELSE 0.0 END END) AS positive_fraction
        FROM pairs
        GROUP BY strategy_id, control_id
        ORDER BY strategy_id, control_id
    """
    result = connection.execute(query).fetchdf()
    by_key = {
        (str(row["strategy_id"]), str(row["control_id"])): row
        for row in result.to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    for strategy_id in strategy_ids:
        for control_id in ("s0_random_matched", "s0_liquidity_matched"):
            row = by_key.get((str(strategy_id), control_id))
            rows.append(
                {
                    "strategy_id": str(strategy_id),
                    "control_id": control_id,
                    "summary": {
                        "matched_count": int(row["matched_count"]) if row is not None else 0,
                        "observed_count": int(row["observed_count"]) if row is not None else 0,
                        "mean_difference": row.get("mean_difference") if row is not None else None,
                        "median_difference": row.get("median_difference") if row is not None else None,
                        "positive_fraction": row.get("positive_fraction") if row is not None else None,
                    },
                }
            )
    return rows


def summarize_development_outputs(output_root: str | Path) -> dict[str, Any]:
    """Read compact day outputs and produce the global development summary."""

    root = Path(output_root).resolve()
    paths = sorted((root / "outcomes").glob("date=*/outcomes.parquet"))
    if not paths:
        raise MinuteStrategyRunError("strategy_run_outcomes_missing")
    connection = open_guarded_duckdb(
        ":memory:",
        temp_directory=root / ".tmp" / "minute_strategy_summary_duckdb",
        threads="auto",
        floor_bytes=DEFAULT_MEMORY_FLOOR_BYTES,
        minimum_limit_bytes=128 * 1024**2,
    )
    try:
        metrics = _outcome_metric_columns(connection, paths)
        outcome_rows = int(
            connection.execute(f"SELECT COUNT(*) FROM {_scan(paths)}").fetchone()[0]
        )
        if not outcome_rows:
            raise MinuteStrategyRunError("strategy_run_outcomes_empty")
        summary: dict[str, Any] = {
            "output_root": str(root),
            "outcome_file_count": len(paths),
            "outcome_rows": outcome_rows,
            "summary_overall": _stream_event_summary(
                connection, paths, group_by=("strategy_id", "ma_period"), metrics=metrics
            ),
            "summary_overall_pooled": _stream_event_summary(
                connection, paths, group_by=("strategy_id",), metrics=metrics
            ),
            "summary_by_year": _stream_event_summary(
                connection, paths, group_by=("year", "strategy_id"), metrics=metrics
            ),
            "summary_by_market_regime": _stream_event_summary(
                connection, paths, group_by=("market_regime", "strategy_id"), metrics=metrics
            ),
        }
        strategy_ids = [
            str(row["strategy_id"])
            for row in summary["summary_overall_pooled"]
            if str(row["strategy_id"]) not in {"s0_random_matched", "s0_liquidity_matched"}
        ]
        summary["control_comparisons"] = _stream_control_summaries(
            connection, paths, sorted(set(strategy_ids)), metric="net_return_60m"
        )
    finally:
        connection.close()
    _write_json(root / "development_summary.json", summary)
    return _jsonable(summary)


def run_development_study(
    workspace_root: str | Path,
    *,
    config: DevelopmentStudyConfig | None = None,
    data_config: StrategyDataConfig | None = None,
    ma_config: MinuteMAConfig | None = None,
    event_config: EventStudyConfig | None = None,
) -> dict[str, Any]:
    """Run the finite strategy family over declared development dates.

    Each target date is processed independently after one month-level hourly
    history aggregation.  A checkpoint is written after every date, making a
    multi-year run resumable without pretending that a partial run is complete.
    """

    selected = config or DevelopmentStudyConfig()
    selected.validate()
    selected_data = data_config or StrategyDataConfig(
        history_open_days=selected.history_open_days,
        outcome_open_days=selected.outcome_open_days,
        duckdb_threads=selected.duckdb_threads,
        temp_directory=selected.temp_directory,
        memory_floor_gib=selected.memory_floor_gib,
        duckdb_memory_floor_gib=selected.duckdb_memory_floor_gib,
    )
    selected_data.validate()
    selected_ma = ma_config or MinuteMAConfig(periods=tuple(selected.strategy_periods))
    selected_ma.validate()
    selected_event = event_config or EventStudyConfig()
    selected_event.validate()
    root = Path(workspace_root).resolve()
    output_root = (root / selected.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _require_memory_floor(selected, "run_start")
    # Keep the development date range separate from the outcome look-ahead:
    # the latter needs a few open dates after the final target date (and may
    # cross a calendar year).  The extra calendar rows are never used to
    # generate a signal; they only make forward outcomes observable.
    calendar_end = (
        date.fromisoformat(str(selected.end_date))
        + timedelta(days=max(14, int(selected.outcome_open_days) * 3))
    ).isoformat()
    calendar = load_trading_calendar(
        root,
        start_date="2010-01-01",
        end_date=calendar_end,
        config=selected_data,
    )
    _require_memory_floor(selected, "calendar_loaded")
    month_ranges = _month_ranges(calendar["trade_date"].astype(str).tolist(), selected.start_date, selected.end_date)
    if selected.max_months:
        month_ranges = month_ranges[: int(selected.max_months)]
    aggregate = {
        "schema": "quantlab.minute_ma_development_run/1",
        "status": "running",
        "config": selected.as_dict(),
        "data_config": selected_data.as_dict(),
        "minute_ma_config": selected_ma.as_dict(),
        "event_study_config": selected_event.as_dict(),
        "months": [],
    }
    for month_start, month_end, target_dates in month_ranges:
        month_dir = _month_state_path(output_root, month_start)
        month_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = month_dir / "checkpoint.json"
        completed = set() if selected.force else _completed_dates(checkpoint)
        # Date-level outputs are the resumable unit and are never overwritten
        # unless --force is explicitly supplied.
        target_dates = [value for value in target_dates if value not in completed]
        target_month_dates = [
            value
            for value in calendar["trade_date"].astype(str).tolist()
            if month_start <= value <= month_end and selected.start_date <= value <= selected.end_date
        ]
        if not target_month_dates:
            continue
        target_universe = load_point_in_time_universe(root, trade_dates=target_month_dates, config=selected_data)
        all_open = calendar["trade_date"].astype(str).tolist()
        start_index = all_open.index(target_month_dates[0])
        support_start = all_open[max(0, start_index - int(selected.history_open_days))]
        outcome_end_index = min(len(all_open) - 1, all_open.index(target_month_dates[-1]) + int(selected.outcome_open_days))
        context_end = all_open[outcome_end_index]
        all_symbols = sorted(target_universe["symbol"].astype(str).unique().tolist())
        symbols = all_symbols
        if selected.max_symbols and int(selected.max_symbols) < len(all_symbols):
            # A bounded run is still point-in-time: cap the pool by turnover
            # known on the first target date, then apply daily membership and
            # status again below.  The uncapped run retains every declared
            # main-board symbol.
            ranking_context = load_daily_context(
                root,
                symbols=all_symbols,
                start_date=support_start,
                end_date=target_month_dates[0],
                config=selected_data,
            )
            first_day_symbols = set(
                target_universe.loc[
                    target_universe["trade_date"].astype(str).eq(target_month_dates[0]), "symbol"
                ].astype(str)
            )
            ranked = ranking_context.loc[
                ranking_context["trade_date"].astype(str).eq(target_month_dates[0])
                & ranking_context["symbol"].astype(str).isin(first_day_symbols),
                ["symbol", "prior_20d_median_amount"],
            ].copy()
            ranked["prior_20d_median_amount"] = pd.to_numeric(
                ranked["prior_20d_median_amount"], errors="coerce"
            )
            ranked.sort_values(
                ["prior_20d_median_amount", "symbol"],
                ascending=[False, True],
                na_position="last",
                inplace=True,
                kind="stable",
            )
            symbols = ranked["symbol"].astype(str).drop_duplicates().tolist()[: int(selected.max_symbols)]
            if len(symbols) < int(selected.max_symbols):
                symbols.extend(
                    value for value in all_symbols if value not in set(symbols)
                )
                symbols = symbols[: int(selected.max_symbols)]
            target_universe = target_universe.loc[target_universe["symbol"].isin(symbols)].copy()
        started_month = time.perf_counter()
        _require_memory_floor(selected, f"month_start:{month_start[:7]}")
        daily_context = load_daily_context(
            root,
            symbols=symbols,
            start_date=support_start,
            end_date=context_end,
            config=selected_data,
        )
        _require_memory_floor(selected, f"daily_context_loaded:{month_start[:7]}")
        # ``month_end`` is a lexical partition label (e.g. YYYY-MM-31), not
        # necessarily a real date.  The last actual target date is the
        # correct inclusive bound for the historical hourly aggregation.
        hourly_history = load_hourly_history(
            root,
            symbols=symbols,
            start_date=support_start,
            end_date=target_month_dates[-1],
            config=selected_data,
        )
        _require_memory_floor(selected, f"hourly_history_loaded:{month_start[:7]}")
        month_outcome_paths: list[str] = []
        month_stats = {
            "month": month_start[:7],
            "target_date_count": len(target_month_dates),
            "symbol_count": len(symbols),
            "support_start": support_start,
            "context_end": context_end,
            "universe_selection": (
                "all_point_in_time_main_board"
                if not selected.max_symbols
                else "top_prior_20d_median_amount_at_month_start"
            ),
            "completed_dates_before_run": len(completed),
            "dates": [],
        }
        for trade_date in target_month_dates:
            date_dir = output_root / "outcomes" / f"date={trade_date}"
            outcome_path = date_dir / "outcomes.parquet"
            if trade_date in completed and outcome_path.is_file() and not selected.force:
                month_outcome_paths.append(str(outcome_path))
                continue
            date_started = time.perf_counter()
            _require_memory_floor(selected, f"date_start:{trade_date}")
            day_universe = target_universe.loc[target_universe["trade_date"].astype(str).eq(trade_date)].copy()
            day_symbols = day_universe["symbol"].astype(str).tolist()
            if not day_symbols:
                continue
            day_bars = load_target_bars(
                root,
                symbols=day_symbols,
                trade_dates=[trade_date],
                daily_context=daily_context,
                require_complete_session=True,
                config=selected_data,
            )
            signals, signal_stats = _build_day_signals(
                day_bars,
                hourly_history,
                daily_context=daily_context,
                ma_config=selected_ma,
                strategy_periods=selected.strategy_periods,
                include_diagnostic=False,
                liquidity_band=selected_data.liquidity_match_band,
                signal_symbol_chunk_size=selected.signal_symbol_chunk_size,
            )
            _require_memory_floor(selected, f"signals_built:{trade_date}")
            if signals.empty:
                continue
            target_index = all_open.index(trade_date)
            future_dates = all_open[target_index : min(len(all_open), target_index + int(selected_data.outcome_open_days) + 1)]
            outcome_symbols = sorted(signals["symbol"].astype(str).unique().tolist())
            outcome_bars = load_target_bars(
                root,
                symbols=outcome_symbols,
                trade_dates=future_dates,
                daily_context=daily_context,
                require_complete_session=False,
                config=selected_data,
            )
            _require_memory_floor(selected, f"outcome_bars_loaded:{trade_date}")
            outcomes = compute_event_outcomes(
                outcome_bars,
                signals,
                config=selected_event,
                trading_dates=all_open,
            )
            outcomes = _lean_outcomes(outcomes)
            date_dir.mkdir(parents=True, exist_ok=True)
            outcomes.to_parquet(outcome_path, index=False)
            month_outcome_paths.append(str(outcome_path))
            completed.add(trade_date)
            day_record = {
                "trade_date": trade_date,
                "symbol_count": len(day_symbols),
                "bar_rows": int(len(day_bars)),
                "signal_rows": int(len(signals)),
                "outcome_rows": int(len(outcomes)),
                "entry_observed": int(outcomes["entry_observed"].fillna(False).sum()),
                "entry_executable": int(outcomes["entry_executable"].fillna(False).sum()),
                "signal_stats": signal_stats,
                "elapsed_seconds": round(time.perf_counter() - date_started, 3),
                "outcome_path": str(outcome_path),
            }
            month_stats["dates"].append(day_record)
            _write_json(
                checkpoint,
                {
                    "schema": "quantlab.minute_ma_development_checkpoint/1",
                    "month": month_start[:7],
                    "completed_dates": sorted(completed),
                    "outcome_paths": sorted(month_outcome_paths),
                    "month_stats": month_stats,
                },
            )
            del day_bars, signals, outcome_bars, outcomes
            gc.collect()
            _require_memory_floor(selected, f"date_complete:{trade_date}")
        month_stats["completed_date_count"] = len(completed)
        month_stats["outcome_paths"] = sorted(set(month_outcome_paths))
        month_stats["elapsed_seconds"] = round(time.perf_counter() - started_month, 3)
        aggregate["months"].append(month_stats)
        _write_json(
            checkpoint,
            {
                "schema": "quantlab.minute_ma_development_checkpoint/1",
                "month": month_start[:7],
                "completed_dates": sorted(completed),
                "outcome_paths": sorted(set(month_outcome_paths)),
                "month_stats": month_stats,
            },
        )
        del daily_context, hourly_history
        gc.collect()
        _require_memory_floor(selected, f"month_complete:{month_start[:7]}")
    aggregate["status"] = "ok" if len(aggregate["months"]) == len(month_ranges) else "partial"
    _write_json(output_root / "run_manifest.json", aggregate)
    if aggregate["status"] == "ok":
        try:
            summary = summarize_development_outputs(output_root)
            aggregate["summary"] = summary
            _write_json(output_root / "run_manifest.json", aggregate)
        except MinuteStrategyRunError:
            aggregate["status"] = "partial"
    return _jsonable(aggregate)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantlab minute-strategy-study")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--output-root", default="runs/minute_ma_development")
    parser.add_argument("--max-months", type=int, default=0)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--history-open-days", type=int, default=66)
    parser.add_argument("--outcome-open-days", type=int, default=5)
    parser.add_argument("--duckdb-threads", default="2")
    parser.add_argument("--signal-symbol-chunk-size", type=int, default=300)
    parser.add_argument("--memory-floor-gib", type=float, default=0.5)
    parser.add_argument("--duckdb-memory-floor-gib", type=float, default=2.0)
    parser.add_argument("--temp-directory", default="")
    parser.add_argument("--period", type=int, action="append", dest="periods")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    duckdb_threads: int | str = (
        args.duckdb_threads
        if str(args.duckdb_threads).strip().lower() == "auto"
        else int(args.duckdb_threads)
    )
    config = DevelopmentStudyConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        output_root=args.output_root,
        max_months=args.max_months,
        max_symbols=args.max_symbols,
        force=bool(args.force),
        history_open_days=args.history_open_days,
        outcome_open_days=args.outcome_open_days,
        strategy_periods=tuple(args.periods or MA_PERIODS),
        signal_symbol_chunk_size=args.signal_symbol_chunk_size,
        duckdb_threads=duckdb_threads,
        temp_directory=args.temp_directory or None,
        memory_floor_gib=args.memory_floor_gib,
        duckdb_memory_floor_gib=args.duckdb_memory_floor_gib,
    )
    result = run_development_study(args.workspace_root, config=config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"ok", "partial"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DevelopmentStudyConfig",
    "MinuteStrategyRunError",
    "run_development_study",
    "summarize_development_outputs",
]
