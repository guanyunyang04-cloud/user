"""Chunked development runner for the hand-designed minute-MA strategies."""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from quantlab.data.qdp_v2.duckdb_resources import (
    DEFAULT_MEMORY_FLOOR_BYTES,
    DuckDbMemoryFloorError,
    open_guarded_duckdb,
)
from quantlab.research.minute_ma import (
    MA_PERIODS,
    MinuteMAConfig,
    build_target_hourly_ma_inputs,
    session_minute_ordinal,
)
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

from .minute_strategy_artifacts import (
    normalized_partition_complete,
    write_normalized_partition,
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

# The state builder naturally carries many diagnostic columns from the raw
# minute frame.  Signals and cross-sectional controls need only this compact
# causal subset.  Keeping the broad frame out of the per-symbol work table is
# the main protection against pandas' temporary merge allocations.
_COMPACT_STATE_COLUMNS = (
    "symbol",
    "trade_date",
    "bar_time",
    "sixty_minute_bucket",
    "ma_period",
    "hour_sequence",
    "session_minute_ordinal",
    "adjusted_close",
    "adjusted_high",
    "adjusted_low",
    "amount",
    "prior_20d_median_amount",
    "causal_intersection_adjusted",
    "close_to_intersection_bps",
    "range_distance_to_intersection_bps",
    "touched_now",
    "close_below_intersection",
    "previous_hour_close_adjusted",
    "previous_hour_high_adjusted",
    "prior_ma_slope_bps",
    "ma_alignment_score",
    "prior_true_touch_count_window",
    "up_down_amount_ratio",
    "bullish_ma_stack",
    "recent_high_breakout",
    "prior_acceleration",
    "breakout_recent",
    "daily_trend_positive",
    "market_supportive",
    "market_regime",
    "sector_strength_rank",
    "sector_breadth",
    "leader_relative_return",
    "sector_strong",
    "leader_sync",
    "auction_confirmed",
    "volume_normal",
    "not_repeated_cross",
    "vwap_supportive",
    "amount_acceleration_positive",
    "auction_gap",
    "volume_acceleration_5_20",
    "amount_curve_surprise",
    "vwap_deviation",
)

_COMPACT_CONTEXT_COLUMNS = (
    "symbol",
    "trade_date",
    "bar_time",
    "recent_high_breakout",
    "prior_acceleration",
    "breakout_recent",
    "daily_trend_positive",
    "market_regime",
    "market_supportive",
    "sector_strength_rank",
    "sector_breadth",
    "leader_relative_return",
    "sector_strong",
    "leader_sync",
    "auction_confirmed",
    "volume_normal",
    "vwap_supportive",
    "amount_acceleration_positive",
    "volume_acceleration_5_20",
    "amount_curve_surprise",
    "vwap_deviation",
)


def _compact_state_frame(states: pd.DataFrame) -> pd.DataFrame:
    """Retain only columns consumed by rules and matched controls."""

    columns = [name for name in _COMPACT_STATE_COLUMNS if name in states.columns]
    return states.loc[:, columns].copy()


def _compact_context_frame(context: pd.DataFrame) -> pd.DataFrame:
    columns = [name for name in _COMPACT_CONTEXT_COLUMNS if name in context.columns]
    return context.loc[:, columns].copy()


def _chunk_size_for_memory(config: DevelopmentStudyConfig, *, requested: int | None = None) -> int:
    """Choose a conservative symbol chunk before a large pandas allocation.

    The cap is intentionally based on *available* RAM, not process RSS.  RSS
    misses DuckDB buffers and other processes, while the user's safety margin
    is a machine-wide requirement.  A smaller chunk is preferable to letting a
    transient merge consume the last few hundred MiB.
    """

    available = _require_memory_floor(config, "chunk_size_probe", soft=True)
    available_gib = available / GIB
    if available_gib < 2.0:
        cap = 24
    elif available_gib < 3.0:
        cap = 64
    elif available_gib < 4.0:
        cap = 128
    elif available_gib < 5.0:
        cap = 256
    elif available_gib < 6.0:
        cap = 384
    else:
        cap = 512
    return max(1, min(int(requested or config.signal_symbol_chunk_size), cap))


def _require_memory_floor(
    config: DevelopmentStudyConfig,
    stage: str,
    *,
    soft: bool = False,
) -> int:
    """Stop before another allocation when the requested RAM reserve is gone.

    ``memory_floor_gib`` is the non-negotiable reserve.  The optional soft
    floor is checked immediately before chunked pandas work so a transient
    allocation has room to complete without crossing the hard floor.
    """

    available = int(psutil.virtual_memory().available)
    floor_gib = config.soft_memory_floor_gib if soft else config.memory_floor_gib
    floor = int(float(floor_gib) * GIB)
    if available < floor:
        raise MinuteStrategyRunError(
            f"strategy_run_memory_floor_breached:{stage}:"
            f"available={available}:floor={floor}:soft={bool(soft)}"
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
    signal_symbol_chunk_size: int = 512
    duckdb_threads: int | str = 2
    temp_directory: str | None = None
    memory_floor_gib: float = 0.5
    soft_memory_floor_gib: float = 1.0
    duckdb_memory_floor_gib: float = 2.0
    write_normalized_artifacts: bool = True
    normalized_subdirectory: str = "normalized"

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
            not np.isfinite(float(self.soft_memory_floor_gib))
            or float(self.soft_memory_floor_gib) < float(self.memory_floor_gib)
        ):
            raise MinuteStrategyRunError("strategy_run_soft_memory_floor_invalid")
        if (
            not np.isfinite(float(self.duckdb_memory_floor_gib))
            or float(self.duckdb_memory_floor_gib) < float(self.memory_floor_gib)
        ):
            raise MinuteStrategyRunError("strategy_run_duckdb_memory_floor_invalid")
        if not isinstance(self.write_normalized_artifacts, (bool, np.bool_)):
            raise MinuteStrategyRunError("strategy_run_write_normalized_artifacts_invalid")
        if (
            not isinstance(self.normalized_subdirectory, str)
            or not self.normalized_subdirectory.strip()
            or Path(self.normalized_subdirectory).is_absolute()
            or any(part in {".", ".."} for part in Path(self.normalized_subdirectory).parts)
        ):
            raise MinuteStrategyRunError("strategy_run_normalized_subdirectory_invalid")
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
            "soft_memory_floor_gib": float(self.soft_memory_floor_gib),
            "duckdb_memory_floor_gib": float(self.duckdb_memory_floor_gib),
            "write_normalized_artifacts": bool(self.write_normalized_artifacts),
            "normalized_subdirectory": str(self.normalized_subdirectory),
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
    memory_config: DevelopmentStudyConfig | None = None,
    ma_inputs: Mapping[int, tuple[pd.DataFrame, pd.DataFrame]] | None = None,
    random_seed: int = 7,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if day_bars.empty:
        return pd.DataFrame(), {"state_rows": 0, "signal_rows": 0, "control_reference_count": 0, "control_matched_count": 0}
    # Build cross-sectional context once (it must see the full universe), then
    # immediately discard the diagnostic/raw columns.  Per-symbol chunks below
    # retain the market/sector fields while keeping transient MA tables small.
    market_context = _compact_context_frame(
        build_minute_market_context(day_bars, daily_context=daily_context)
    )
    # Alignment is the only cross-period state.  Compute it once, then process
    # each MA period independently so the six expanded state tables are never
    # resident at the same time.
    alignment_history = None
    if ma_inputs:
        day_dates = set(day_bars["trade_date"].astype(str).unique())
        alignment_parts = [
            history.loc[history["trade_date"].astype(str).isin(day_dates)].copy()
            for history, _touches in ma_inputs.values()
            if not history.empty
        ]
        if alignment_parts:
            alignment_history = pd.concat(alignment_parts, ignore_index=True)
    alignment = build_live_ma_alignment(
        day_bars,
        hourly_history,
        config=ma_config,
        hourly_ma_history=alignment_history,
    )
    chunk_config = memory_config or DevelopmentStudyConfig(
        signal_symbol_chunk_size=int(signal_symbol_chunk_size)
    )
    chunk_size = _chunk_size_for_memory(chunk_config, requested=signal_symbol_chunk_size)
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
    day_symbol_values = day_bars["symbol"].astype(str)
    context_symbol_values = market_context["symbol"].astype(str)
    hourly_symbol_values = (
        hourly_history["symbol"].astype(str)
        if hourly_history is not None and not hourly_history.empty
        else None
    )
    all_symbols = day_symbol_values.drop_duplicates().tolist()
    alignment_symbol_values = (
        alignment["symbol"].astype(str) if not alignment.empty else pd.Series(dtype="string")
    )
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
        # Keep one compact state chunk per symbol group for the control match.
        # No broad all-universe state frame is needed by the signal builder.
        period_state_parts: list[pd.DataFrame] = []
        period_signal_parts: list[pd.DataFrame] = []
        period_reference_parts: list[pd.DataFrame] = []
        for start in range(0, len(all_symbols), chunk_size):
            _require_memory_floor(
                memory_config or DevelopmentStudyConfig(signal_symbol_chunk_size=chunk_size),
                f"signal_chunk_start:{period}:{start}",
                soft=True,
            )
            chunk_symbols = set(all_symbols[start : start + chunk_size])
            bars_mask = day_symbol_values.isin(chunk_symbols).to_numpy(dtype=bool)
            context_mask = context_symbol_values.isin(chunk_symbols).to_numpy(dtype=bool)
            hourly_mask = (
                hourly_symbol_values.isin(chunk_symbols).to_numpy(dtype=bool)
                if hourly_symbol_values is not None
                else None
            )
            chunk_bars = day_bars.loc[bars_mask].copy()
            chunk_context = market_context.loc[context_mask].copy()
            chunk_hourly = (
                hourly_history.loc[hourly_mask].copy()
                if hourly_history is not None and hourly_mask is not None
                else None
            )
            if ma_inputs and period in ma_inputs:
                cached_history, cached_touches = ma_inputs[period]
                target_dates = set(chunk_bars["trade_date"].astype(str).unique())
                cached_history_chunk = cached_history.loc[
                    cached_history["symbol"].astype(str).isin(chunk_symbols)
                    & cached_history["trade_date"].astype(str).isin(target_dates)
                ].copy()
                cached_touches_chunk = cached_touches.loc[
                    cached_touches["symbol"].astype(str).isin(chunk_symbols)
                    & cached_touches["trade_date"].astype(str).isin(target_dates)
                ].copy()
                states = build_enriched_minute_ma_states(
                    chunk_bars,
                    None,
                    # Context already contains the daily and cross-sectional
                    # fields; avoiding a second daily merge saves a large copy.
                    daily_context=None,
                    context=chunk_context,
                    config=period_config,
                    hourly_ma_history=cached_history_chunk,
                    prior_touch_counts=cached_touches_chunk,
                )
                del cached_history_chunk, cached_touches_chunk
            else:
                states = build_enriched_minute_ma_states(
                    chunk_bars,
                    chunk_hourly,
                    # Context already contains the daily and cross-sectional
                    # fields; avoiding a second daily merge saves a large copy.
                    daily_context=None,
                    context=chunk_context,
                    config=period_config,
                )
            if states.empty:
                del chunk_bars, chunk_context, chunk_hourly, states
                gc.collect()
                continue
            if not alignment.empty:
                chunk_alignment = alignment.loc[alignment_symbol_values.isin(chunk_symbols)].copy()
                states = _attach_alignment(states, chunk_alignment)
                del chunk_alignment
            state_rows += int(len(states))
            compact_states = _compact_state_frame(states)
            period_state_parts.append(compact_states)
            chunk_signals = build_strategy_signals_vectorized(
                compact_states,
                strategy_ids=[spec.strategy_id for spec in specs],
                include_diagnostic=include_diagnostic,
                random_seed=random_seed,
            )
            if not chunk_signals.empty:
                filtered = _decision_signal_filter(chunk_signals)
                if not filtered.empty:
                    period_signal_parts.append(filtered)
                    reference_frame = filtered.loc[
                        ~filtered["strategy_id"].isin(["s0_random_matched", "s0_liquidity_matched"])
                        & filtered["signal_executable"].fillna(False)
                    ].copy()
                    if not reference_frame.empty:
                        period_reference_parts.append(reference_frame)
            del chunk_bars, chunk_context, chunk_hourly, states, compact_states, chunk_signals
            gc.collect()
            _require_memory_floor(
                memory_config or DevelopmentStudyConfig(signal_symbol_chunk_size=chunk_size),
                f"signal_chunk_complete:{period}:{start}",
            )
        if period_signal_parts:
            parts.append(pd.concat(period_signal_parts, ignore_index=True))
        if period_reference_parts and period_state_parts:
            reference_frame = pd.concat(period_reference_parts, ignore_index=True)
            control_states = pd.concat(period_state_parts, ignore_index=True)
            controls = build_liquidity_matched_controls_many(
                control_states,
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
            del reference_frame, control_states, controls
        del period_state_parts, period_signal_parts, period_reference_parts
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


def _empty_outcome_frame(config: EventStudyConfig) -> pd.DataFrame:
    metric_names = (
        [f"gross_return_{horizon}m" for horizon in config.minute_horizons]
        + [f"net_return_{horizon}m" for horizon in config.minute_horizons]
        + [f"gross_return_{horizon}d" for horizon in config.day_horizons]
        + [f"net_return_{horizon}d" for horizon in config.day_horizons]
        + ["mfe_same_day", "mae_same_day", "t1_gross_return", "t1_net_return"]
    )
    return pd.DataFrame(
        {
            name: pd.Series(dtype="object")
            for name in (*OUTCOME_COLUMNS, *metric_names)
        }
    )


_OUTCOME_RUNTIME_COLUMNS = {
    "entry_date",
    "entry_time",
    "entry_price",
    "entry_adjusted_price",
    "entry_observed",
    "entry_executable",
    "entry_reason",
    "entry_bar_index",
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
}


def _broadcast_unique_event_outcomes(
    signals: pd.DataFrame,
    unique_outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Map one computed price path back to every strategy using that event."""

    key_columns = ["symbol", "signal_date", "signal_time", "signal_executable"]
    runtime_columns = [
        name
        for name in unique_outcomes.columns
        if name in _OUTCOME_RUNTIME_COLUMNS
        or name.startswith(("gross_return_", "net_return_"))
    ]
    lookup = unique_outcomes.loc[:, [*key_columns, *runtime_columns]].drop_duplicates(
        key_columns
    )
    base_columns = [
        name
        for name in OUTCOME_COLUMNS
        if name in signals.columns and name not in _OUTCOME_RUNTIME_COLUMNS
    ]
    base = signals.loc[:, base_columns].copy()
    base["_row_order"] = np.arange(len(base), dtype=np.int64)
    result = base.merge(
        lookup,
        on=key_columns,
        how="left",
        validate="many_to_one",
        sort=False,
    )
    result.sort_values("_row_order", inplace=True, kind="stable")
    result.drop(columns="_row_order", inplace=True)
    for name in OUTCOME_COLUMNS:
        if name not in result.columns:
            result[name] = None
    return _lean_outcomes(result)


def _compute_outcomes_chunked(
    workspace_root: Path,
    signals: pd.DataFrame,
    *,
    future_dates: Sequence[str],
    trading_dates: Sequence[str],
    data_config: StrategyDataConfig,
    event_config: EventStudyConfig,
    requested_chunk_size: int,
    memory_config: DevelopmentStudyConfig,
) -> pd.DataFrame:
    """Score one date in symbol chunks instead of loading all forward minutes.

    A five-session outcome window is several times larger than the target-day
    signal frame.  Keeping only one symbol slice of those minutes at a time
    prevents the forward lookup table from becoming the run's memory peak.
    """

    if signals.empty:
        return pd.DataFrame()
    chunk_size = _chunk_size_for_memory(memory_config, requested=requested_chunk_size)
    symbol_values = signals["symbol"].astype(str)
    symbols = symbol_values.drop_duplicates().tolist()
    parts: list[pd.DataFrame] = []
    for start in range(0, len(symbols), chunk_size):
        _require_memory_floor(
            memory_config,
            f"outcome_chunk_start:{start}",
            soft=True,
        )
        chunk_symbols = set(symbols[start : start + chunk_size])
        signal_mask = symbol_values.isin(chunk_symbols).to_numpy(dtype=bool)
        chunk_signals = signals.loc[signal_mask].copy()
        # Outcomes only need OHLC, execution status and session ordinals.  The
        # daily/context columns are deliberately not merged into this frame.
        chunk_bars = load_target_bars(
            workspace_root,
            symbols=sorted(chunk_symbols),
            trade_dates=future_dates,
            daily_context=None,
            require_complete_session=False,
            config=data_config,
        )
        event_keys = ["symbol", "signal_date", "signal_time", "signal_executable"]
        unique_signals = chunk_signals.drop_duplicates(event_keys, keep="first").copy()
        unique_outcomes = compute_event_outcomes(
            chunk_bars,
            unique_signals,
            config=event_config,
            trading_dates=trading_dates,
        )
        parts.append(_broadcast_unique_event_outcomes(chunk_signals, unique_outcomes))
        del chunk_signals, chunk_bars, unique_signals, unique_outcomes
        gc.collect()
        _require_memory_floor(memory_config, f"outcome_chunk_complete:{start}")
    if not parts:
        return pd.DataFrame()
    result = pd.concat(parts, ignore_index=True)
    # ``compute_event_outcomes`` normally exposes a batch-local row number.
    # Once the bars are partitioned by symbol that number would depend on the
    # chunk boundary.  Replace it with a deterministic key within this
    # outcome window so reruns and resumed chunks remain comparable.
    if "entry_bar_index" in result.columns and {"entry_date", "entry_time"}.issubset(result.columns):
        symbol_codes = {value: index for index, value in enumerate(sorted(symbols))}
        date_codes = {str(value): index for index, value in enumerate(future_dates)}
        entry_symbols = result["symbol"].astype(str).map(symbol_codes)
        entry_dates = result["entry_date"].astype("string").map(date_codes)
        entry_ordinals = result["entry_time"].map(session_minute_ordinal)
        valid = entry_symbols.notna() & entry_dates.notna() & entry_ordinals.notna()
        if valid.any():
            date_stride = max(1, len(date_codes)) * 240
            stable = (
                entry_symbols.loc[valid].astype("int64") * date_stride
                + entry_dates.loc[valid].astype("int64") * 240
                + entry_ordinals.loc[valid].astype("int64")
            )
            result.loc[valid, "entry_bar_index"] = stable.to_numpy(dtype=np.int64)
    result.sort_values(
        ["signal_date", "signal_time", "strategy_id", "symbol", "ma_period"],
        kind="stable",
        inplace=True,
        ignore_index=True,
    )
    return result


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


def _stream_event_summary_legacy(
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


def _source_select_fields(keys: Sequence[str], metrics: Sequence[str]) -> tuple[list[str], str]:
    """Build a narrow, typed source projection for summary queries."""

    del keys  # Group keys are selected explicitly below; this keeps the helper stable.
    fields = [
        f"CAST({_quote_identifier('strategy_id')} AS VARCHAR) AS {_quote_identifier('strategy_id')}",
        f"TRY_CAST({_quote_identifier('ma_period')} AS INTEGER) AS {_quote_identifier('ma_period')}",
        f"CAST({_quote_identifier('signal_date')} AS VARCHAR) AS {_quote_identifier('signal_date')}",
        f"COALESCE(TRY_CAST({_quote_identifier('diagnostic_only')} AS BOOLEAN), FALSE) AS {_quote_identifier('__diagnostic_only')}",
        f"COALESCE(TRY_CAST({_quote_identifier('entry_observed')} AS BOOLEAN), FALSE) AS {_quote_identifier('__entry_observed')}",
        f"COALESCE(TRY_CAST({_quote_identifier('entry_executable')} AS BOOLEAN), FALSE) AS {_quote_identifier('__entry_executable')}",
        f"CAST({_quote_identifier('market_regime')} AS VARCHAR) AS {_quote_identifier('market_regime')}",
        f"TRY_CAST(substr(CAST({_quote_identifier('signal_date')} AS VARCHAR), 1, 4) AS INTEGER) AS {_quote_identifier('year')}",
    ]
    aliases: list[str] = []
    for name in metrics:
        alias = f"__metric_{name}"
        aliases.append(alias)
        fields.append(
            f"TRY_CAST({_quote_identifier(name)} AS DOUBLE) AS {_quote_identifier(alias)}"
        )
    return fields, ", ".join(aliases)


def _summary_stats_query(
    paths: Sequence[Path],
    *,
    keys: Sequence[str],
    metrics: Sequence[str],
) -> str:
    fields, _ = _source_select_fields(keys, metrics)
    group_sql = ", ".join(_quote_identifier(name) for name in keys)
    expressions = [
        "COUNT(*) AS __signal_count",
        'COUNT(*) FILTER (WHERE "__entry_observed") AS __entry_observed_count',
        'COUNT(*) FILTER (WHERE "__entry_executable") AS __entry_executable_count',
    ]
    for name in metrics:
        alias = f"__metric_{name}"
        expressions.extend(
            [
                f'COUNT(*) FILTER (WHERE isfinite("{alias}")) AS "{alias}__observed"',
                f'AVG("{alias}") FILTER (WHERE isfinite("{alias}")) AS "{alias}__mean"',
                f'MEDIAN("{alias}") FILTER (WHERE isfinite("{alias}")) AS "{alias}__median"',
                f'SUM("{alias}") FILTER (WHERE isfinite("{alias}")) AS "{alias}__sum"',
                f'SUM(-"{alias}") FILTER (WHERE isfinite("{alias}") AND "{alias}" < 0) AS "{alias}__negative_abs"',
                f'COUNT(*) FILTER (WHERE isfinite("{alias}") AND "{alias}" > 0) AS "{alias}__positive"',
                f'SUM("{alias}") FILTER (WHERE isfinite("{alias}") AND "{alias}" > 0) AS "{alias}__positive_sum"',
            ]
        )
    return f"""
        WITH source AS (
            SELECT {", ".join(fields)}
            FROM {_scan(paths)}
            WHERE NOT "__diagnostic_only"
        )
        SELECT {group_sql}, {", ".join(expressions)}
        FROM source
        GROUP BY {group_sql}
        ORDER BY {group_sql}
    """


def _summary_trim_query(
    paths: Sequence[Path],
    *,
    keys: Sequence[str],
    metrics: Sequence[str],
) -> str:
    fields, _ = _source_select_fields(keys, metrics)
    group_sql = ", ".join(_quote_identifier(name) for name in keys)
    aggregate: list[str] = []
    output: list[str] = []
    for name in metrics:
        alias = f"__metric_{name}"
        aggregate.extend(
            [
                f'COUNT(*) FILTER (WHERE isfinite("{alias}")) AS "{alias}__observed"',
                f'SUM("{alias}") FILTER (WHERE isfinite("{alias}")) AS "{alias}__sum"',
                f'COUNT(*) FILTER (WHERE isfinite("{alias}") AND "{alias}" > 0) AS "{alias}__positive"',
                f'list_sort(list("{alias}") FILTER (WHERE isfinite("{alias}") AND "{alias}" > 0), \'DESC\') AS "{alias}__list"',
            ]
        )
        output.append(
            f'''CASE
                WHEN "{alias}__observed" = 0 THEN NULL
                WHEN "{alias}__positive" = 0 THEN "{alias}__sum" / NULLIF("{alias}__observed", 0)
                ELSE ("{alias}__sum" - COALESCE(
                    list_sum(list_slice("{alias}__list", 1, CAST(CEIL("{alias}__positive" * 0.01) AS BIGINT))), 0
                )) / NULLIF(
                    "{alias}__observed" - CAST(CEIL("{alias}__positive" * 0.01) AS BIGINT), 0
                )
            END AS "{alias}__trimmed"'''
        )
    return f"""
        WITH source AS (
            SELECT {", ".join(fields)}
            FROM {_scan(paths)}
            WHERE NOT "__diagnostic_only"
        ), aggregate AS (
            SELECT {group_sql}, {", ".join(aggregate)}
            FROM source
            GROUP BY {group_sql}
        )
        SELECT {group_sql}, {", ".join(output)}
        FROM aggregate
        ORDER BY {group_sql}
    """


def _summary_trim_approx_query(
    paths: Sequence[Path],
    *,
    keys: Sequence[str],
    metrics: Sequence[str],
) -> str:
    fields, _ = _source_select_fields(keys, metrics)
    group_sql = ", ".join(_quote_identifier(name) for name in keys)
    quantiles: list[str] = []
    outputs: list[str] = []
    for name in metrics:
        alias = f"__metric_{name}"
        quantiles.append(
            f'approx_quantile("{alias}", 0.99) FILTER (WHERE isfinite("{alias}") AND "{alias}" > 0) AS "{alias}__q99"'
        )
        outputs.append(
            f'AVG(CASE WHEN isfinite("{alias}") AND ("{alias}" <= 0 OR "{alias}" <= "{alias}__q99") THEN "{alias}" END) AS "{alias}__trimmed"'
        )
    return f"""
        WITH source AS (
            SELECT {", ".join(fields)}
            FROM {_scan(paths)}
            WHERE NOT "__diagnostic_only"
        ), quantiles AS (
            SELECT {group_sql}, {", ".join(quantiles)}
            FROM source
            GROUP BY {group_sql}
        ), joined AS (
            SELECT source.*, quantiles.* EXCLUDE ({group_sql})
            FROM source
            JOIN quantiles USING ({group_sql})
        )
        SELECT {group_sql}, {", ".join(outputs)}
        FROM joined
        GROUP BY {group_sql}
        ORDER BY {group_sql}
    """


def _stream_event_summary(
    connection: Any,
    paths: Sequence[Path],
    *,
    group_by: Sequence[str],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    """Aggregate wide outcome metrics without a row-multiplying UNPIVOT."""

    if not paths:
        return []
    keys = tuple(str(value) for value in group_by)
    if not keys:
        raise MinuteStrategyRunError("strategy_summary_group_empty")
    allowed = {"strategy_id", "ma_period", "signal_date", "market_regime", "year"}
    unknown = sorted(set(keys).difference(allowed))
    if unknown:
        raise MinuteStrategyRunError(f"strategy_summary_group_columns_invalid:{','.join(unknown)}")
    metric_names = tuple(dict.fromkeys(str(value) for value in metrics))
    if not metric_names:
        return []
    stats = connection.execute(
        _summary_stats_query(paths, keys=keys, metrics=metric_names)
    ).fetchdf()
    if stats.empty:
        return []
    rows_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    stats_records = stats.to_dict("records")
    stats_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in stats_records:
        key = tuple(record.get(name) for name in keys)
        stats_by_key[key] = record
        rows_by_key[key] = {
            **{name: record.get(name) for name in keys},
            "signal_count": int(record.get("__signal_count") or 0),
            "entry_observed_count": int(record.get("__entry_observed_count") or 0),
            "entry_executable_count": int(record.get("__entry_executable_count") or 0),
            "diagnostic_excluded": True,
        }
    max_group = int(pd.to_numeric(stats["__signal_count"], errors="coerce").max())
    total_rows = int(pd.to_numeric(stats["__signal_count"], errors="coerce").sum())
    # Lists give the historical top-1% definition exactly, but only while a
    # group is small enough to keep its positive values bounded.  Large pooled
    # runs use a 99th-positive-quantile approximation instead of risking the
    # machine-wide memory reserve.
    exact_trim = max_group <= 3_000_000 and total_rows <= 30_000_000
    if exact_trim:
        batch_size = 8 if max_group <= 1_000_000 else (4 if max_group <= 2_000_000 else 2)
        trim_frames: list[pd.DataFrame] = []
        for start in range(0, len(metric_names), batch_size):
            trim_frames.append(
                connection.execute(
                    _summary_trim_query(
                        paths,
                        keys=keys,
                        metrics=metric_names[start : start + batch_size],
                    )
                ).fetchdf()
            )
        trim = trim_frames[0]
        for frame in trim_frames[1:]:
            trim = trim.merge(frame, on=list(keys), how="outer", validate="one_to_one")
    else:
        trim = connection.execute(
            _summary_trim_approx_query(paths, keys=keys, metrics=metric_names)
        ).fetchdf()
    for record in trim.to_dict("records"):
        key = tuple(record.get(name) for name in keys)
        row = rows_by_key.get(key)
        stat_record = stats_by_key.get(key)
        if row is None or stat_record is None:
            continue
        for metric in metric_names:
            alias = f"__metric_{metric}"
            observed = int(stat_record.get(f"{alias}__observed") or 0)
            positive_count = int(stat_record.get(f"{alias}__positive") or 0)
            positive_sum = stat_record.get(f"{alias}__positive_sum")
            negative_abs = stat_record.get(f"{alias}__negative_abs")
            row[f"{metric}_observed_count"] = observed
            row[f"{metric}_mean"] = stat_record.get(f"{alias}__mean")
            row[f"{metric}_median"] = stat_record.get(f"{alias}__median")
            row[f"{metric}_win_rate"] = positive_count / observed if observed else None
            row[f"{metric}_profit_factor"] = (
                float(positive_sum) / float(negative_abs)
                if positive_sum is not None and negative_abs is not None and float(negative_abs) > 0
                else None
            )
            row[f"{metric}_trimmed_mean_top1pct_removed"] = record.get(f"{alias}__trimmed")
    # A group with no finite value for a metric is still represented in the
    # summary, matching the previous event-study behaviour.
    for row in rows_by_key.values():
        for metric in metric_names:
            alias = f"__metric_{metric}"
            row.setdefault(f"{metric}_observed_count", 0)
            row.setdefault(f"{metric}_mean", None)
            row.setdefault(f"{metric}_median", None)
            row.setdefault(f"{metric}_win_rate", None)
            row.setdefault(f"{metric}_profit_factor", None)
            row.setdefault(f"{metric}_trimmed_mean_top1pct_removed", None)
    return list(rows_by_key.values())


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
        # Summary queries are window-heavy (median and winner trimming).  A
        # single DuckDB worker is deliberately used here: parallel hash/window
        # buffers can consume the entire machine-wide reserve even though the
        # underlying outcome files are already compact.
        threads=1,
        floor_bytes=DEFAULT_MEMORY_FLOOR_BYTES,
        minimum_limit_bytes=64 * 1024**2,
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
    normalized_root = output_root / selected.normalized_subdirectory
    _require_memory_floor(selected, "run_start")
    # Keep the development date range separate from the outcome look-ahead:
    # the latter needs a few open dates after the final target date (and may
    # cross a calendar year).  The extra calendar rows are never used to
    # generate a signal; they only make forward outcomes observable.
    calendar_end = (
        date.fromisoformat(str(selected.end_date))
        + timedelta(days=max(14, int(selected.outcome_open_days) * 3))
    ).isoformat()
    _require_memory_floor(selected, "calendar_load_start", soft=True)
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
        "normalized_artifact_root": str(normalized_root)
        if selected.write_normalized_artifacts
        else None,
        "months": [],
    }
    for month_start, month_end, target_dates in month_ranges:
        month_dir = _month_state_path(output_root, month_start)
        month_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = month_dir / "checkpoint.json"
        completed = set() if selected.force else _completed_dates(checkpoint)
        completed = {
            value
            for value in completed
            if (output_root / "outcomes" / f"date={value}" / "outcomes.parquet").is_file()
        }
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
        if not target_dates:
            # Everything in this month already has a durable date output.
            # Avoid reloading the large support window merely to update a
            # checkpoint during a resume, while preserving an ``ok`` run
            # status when every month is already complete.
            aggregate["months"].append(
                {
                    "month": month_start[:7],
                    "target_date_count": len(target_month_dates),
                    "completed_date_count": len(completed),
                    "status": "already_complete",
                }
            )
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
        _require_memory_floor(selected, f"daily_context_load_start:{month_start[:7]}", soft=True)
        daily_context = load_daily_context(
            root,
            symbols=symbols,
            start_date=support_start,
            end_date=context_end,
            config=selected_data,
        )
        _require_memory_floor(selected, f"daily_context_loaded:{month_start[:7]}")
        # Use the same adaptive symbol block for historical hourly aggregation
        # and minute-state construction.  The SQL group-by is the largest
        # transient allocation in a full-universe month; chunking it prevents
        # a four-thread query from exhausting DuckDB's memory budget.
        cache_chunk_size = _chunk_size_for_memory(
            selected, requested=selected.signal_symbol_chunk_size
        )
        _require_memory_floor(selected, f"hourly_history_load_start:{month_start[:7]}", soft=True)
        # ``month_end`` is a lexical partition label (e.g. YYYY-MM-31), not
        # necessarily a real date.  The last actual target date is the
        # correct inclusive bound for the historical hourly aggregation.
        hourly_history = load_hourly_history(
            root,
            symbols=symbols,
            start_date=support_start,
            end_date=target_month_dates[-1],
            config=selected_data,
            symbol_chunk_size=cache_chunk_size,
        )
        _require_memory_floor(selected, f"hourly_history_loaded:{month_start[:7]}")
        # Rolling MA history is computed once per month, retained only for the
        # target dates, and split by period.  This removes the old repeated
        # 66-session rolling calculation from every target day.
        _require_memory_floor(
            selected, f"ma_cache_start:{month_start[:7]}", soft=True
        )
        cached_history, cached_touches = build_target_hourly_ma_inputs(
            hourly_history,
            target_dates,
            config=selected_ma,
            symbol_chunk_size=cache_chunk_size,
        )
        ma_inputs = {
            int(period): (
                cached_history.loc[cached_history["ma_period"].eq(int(period))].copy(),
                cached_touches.loc[cached_touches["ma_period"].eq(int(period))].copy(),
            )
            for period in selected.strategy_periods
        }
        del cached_history, cached_touches
        # The cache contains everything the signal path needs; raw hourly bars
        # are no longer kept beside the target-day minute frame.
        del hourly_history
        hourly_history = None
        gc.collect()
        _require_memory_floor(selected, f"ma_cache_ready:{month_start[:7]}")
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

        def finish_date(
            trade_date: str,
            *,
            symbol_count: int,
            bar_rows: int,
            signal_rows: int,
            outcomes: pd.DataFrame,
            signal_stats: dict[str, Any],
            started: float,
            _month_outcome_paths: list[str] = month_outcome_paths,
            _completed: set[str] = completed,
            _month_stats: dict[str, Any] = month_stats,
            _checkpoint: Path = checkpoint,
            _month_label: str = month_start[:7],
        ) -> None:
            """Persist one date, including an explicit empty-date result."""

            date_dir = output_root / "outcomes" / f"date={trade_date}"
            outcome_path = date_dir / "outcomes.parquet"
            date_dir.mkdir(parents=True, exist_ok=True)
            outcomes.to_parquet(outcome_path, index=False)
            if selected.write_normalized_artifacts:
                _require_memory_floor(selected, f"normalized_artifacts_start:{trade_date}", soft=True)
                write_normalized_partition(
                    outcomes,
                    normalized_root,
                    trade_date,
                )
                _require_memory_floor(selected, f"normalized_artifacts_complete:{trade_date}")
            _month_outcome_paths.append(str(outcome_path))
            _completed.add(trade_date)
            observed = int(
                outcomes["entry_observed"].fillna(False).sum()
                if "entry_observed" in outcomes
                else 0
            )
            executable = int(
                outcomes["entry_executable"].fillna(False).sum()
                if "entry_executable" in outcomes
                else 0
            )
            _month_stats["dates"].append(
                {
                    "trade_date": trade_date,
                    "symbol_count": int(symbol_count),
                    "bar_rows": int(bar_rows),
                    "signal_rows": int(signal_rows),
                    "outcome_rows": int(len(outcomes)),
                    "entry_observed": observed,
                    "entry_executable": executable,
                    "signal_stats": signal_stats,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "outcome_path": str(outcome_path),
                    "normalized_root": str(normalized_root)
                    if selected.write_normalized_artifacts
                    else None,
                }
            )
            _write_json(
                _checkpoint,
                {
                    "schema": "quantlab.minute_ma_development_checkpoint/1",
                    "month": _month_label,
                    "completed_dates": sorted(_completed),
                    "outcome_paths": sorted(_month_outcome_paths),
                    "month_stats": _month_stats,
                },
            )

        for trade_date in target_month_dates:
            date_dir = output_root / "outcomes" / f"date={trade_date}"
            outcome_path = date_dir / "outcomes.parquet"
            if trade_date in completed and outcome_path.is_file() and not selected.force:
                if selected.write_normalized_artifacts and not normalized_partition_complete(
                    normalized_root, trade_date
                ):
                    _require_memory_floor(
                        selected,
                        f"normalized_resume_start:{trade_date}",
                        soft=True,
                    )
                    existing = pd.read_parquet(outcome_path)
                    write_normalized_partition(existing, normalized_root, trade_date)
                    del existing
                    gc.collect()
                    _require_memory_floor(selected, f"normalized_resume_complete:{trade_date}")
                month_outcome_paths.append(str(outcome_path))
                continue
            date_started = time.perf_counter()
            _require_memory_floor(selected, f"date_start:{trade_date}")
            _require_memory_floor(selected, f"target_bars_load_start:{trade_date}", soft=True)
            day_universe = target_universe.loc[target_universe["trade_date"].astype(str).eq(trade_date)].copy()
            day_symbols = day_universe["symbol"].astype(str).tolist()
            if not day_symbols:
                finish_date(
                    trade_date,
                    symbol_count=0,
                    bar_rows=0,
                    signal_rows=0,
                    outcomes=_empty_outcome_frame(selected_event),
                    signal_stats={
                        "state_rows": 0,
                        "signal_rows": 0,
                        "control_reference_count": 0,
                        "control_matched_count": 0,
                    },
                    started=date_started,
                )
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
                memory_config=selected,
                ma_inputs=ma_inputs,
            )
            _require_memory_floor(selected, f"signals_built:{trade_date}")
            if signals.empty:
                finish_date(
                    trade_date,
                    symbol_count=len(day_symbols),
                    bar_rows=len(day_bars),
                    signal_rows=0,
                    outcomes=_empty_outcome_frame(selected_event),
                    signal_stats=signal_stats,
                    started=date_started,
                )
                del day_bars, signals
                gc.collect()
                continue
            target_index = all_open.index(trade_date)
            future_dates = all_open[target_index : min(len(all_open), target_index + int(selected_data.outcome_open_days) + 1)]
            outcomes = _compute_outcomes_chunked(
                root,
                signals,
                future_dates=future_dates,
                trading_dates=all_open,
                data_config=selected_data,
                event_config=selected_event,
                requested_chunk_size=selected.signal_symbol_chunk_size,
                memory_config=selected,
            )
            finish_date(
                trade_date,
                symbol_count=len(day_symbols),
                bar_rows=len(day_bars),
                signal_rows=len(signals),
                outcomes=outcomes,
                signal_stats=signal_stats,
                started=date_started,
            )
            del day_bars, signals, outcomes
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
        del daily_context, hourly_history, ma_inputs
        gc.collect()
        _require_memory_floor(selected, f"month_complete:{month_start[:7]}")
    aggregate["status"] = "ok" if len(aggregate["months"]) == len(month_ranges) else "partial"
    _write_json(output_root / "run_manifest.json", aggregate)
    if aggregate["status"] == "ok":
        try:
            summary = summarize_development_outputs(output_root)
            aggregate["summary"] = summary
            _write_json(output_root / "run_manifest.json", aggregate)
        except (MinuteStrategyRunError, DuckDbMemoryFloorError) as exc:
            aggregate["status"] = "partial"
            aggregate["summary_status"] = "deferred_memory_floor"
            aggregate["summary_error"] = f"{type(exc).__name__}:{str(exc)[:300]}"
            _write_json(output_root / "run_manifest.json", aggregate)
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
    parser.add_argument("--signal-symbol-chunk-size", type=int, default=512)
    parser.add_argument("--memory-floor-gib", type=float, default=0.5)
    parser.add_argument("--soft-memory-floor-gib", type=float, default=1.0)
    parser.add_argument("--duckdb-memory-floor-gib", type=float, default=2.0)
    parser.add_argument("--temp-directory", default="")
    parser.add_argument(
        "--no-normalized-artifacts",
        action="store_true",
        help="Keep only legacy wide outcomes; skip normalized event/path artifacts.",
    )
    parser.add_argument(
        "--normalized-subdirectory",
        default="normalized",
        help="Relative subdirectory under the output root for normalized artifacts.",
    )
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
        soft_memory_floor_gib=args.soft_memory_floor_gib,
        duckdb_memory_floor_gib=args.duckdb_memory_floor_gib,
        write_normalized_artifacts=not bool(args.no_normalized_artifacts),
        normalized_subdirectory=args.normalized_subdirectory,
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
