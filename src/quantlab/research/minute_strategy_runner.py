"""Chunked development runner for the hand-designed minute-MA strategies."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from quantlab.research.minute_ma import MA_PERIODS, MinuteMAConfig
from quantlab.research.minute_ma_event_study import (
    OUTCOME_COLUMNS,
    EventStudyConfig,
    compare_to_control,
    compare_to_reference_control,
    compute_event_outcomes,
    summarize_control_comparison,
    summarize_event_study,
)
from quantlab.research.minute_ma_strategies import (
    build_liquidity_matched_controls_many,
    build_strategy_signals_vectorized,
    strategy_catalog,
)

from .minute_strategy_data import (
    StrategyDataConfig,
    build_enriched_minute_ma_states,
    build_minute_market_context,
    load_daily_context,
    load_hourly_history,
    load_point_in_time_universe,
    load_target_bars,
    load_trading_calendar,
)


class MinuteStrategyRunError(RuntimeError):
    """Raised when a chunked strategy run cannot satisfy its output contract."""


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
    duckdb_threads: int = 8
    temp_directory: str | None = None

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
            ("duckdb_threads", self.duckdb_threads, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
                raise MinuteStrategyRunError(f"strategy_run_{name}_invalid")
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
            "duckdb_threads": int(self.duckdb_threads),
            "temp_directory": self.temp_directory,
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
    parts: list[pd.DataFrame] = []
    control_parts: list[pd.DataFrame] = []
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
    states = build_enriched_minute_ma_states(
        day_bars,
        hourly_history,
        daily_context=daily_context,
        context=market_context,
        config=ma_config,
    )
    if states.empty:
        return pd.DataFrame(), {"state_rows": 0, "signal_rows": 0, "control_reference_count": 0, "control_matched_count": 0}
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
    if not signal.empty:
        filtered = _decision_signal_filter(signal)
        if not filtered.empty:
            parts.append(filtered)
    references = [part for part in parts if not part.empty]
    reference_frame = pd.concat(references, ignore_index=True) if references else pd.DataFrame()
    if not reference_frame.empty:
        reference_frame = reference_frame.loc[
            ~reference_frame["strategy_id"].isin(["s0_random_matched", "s0_liquidity_matched"])
            & reference_frame["signal_executable"].fillna(False)
        ].copy()
    if not reference_frame.empty:
        controls = build_liquidity_matched_controls_many(
            states,
            reference_frame,
            liquidity_band_ratio=liquidity_band,
            random_seed=random_seed,
        )
        control_reference_count = int(controls.attrs.get("reference_count", len(reference_frame)))
        control_matched_count = int(controls.attrs.get("matched_count", len(controls)))
        control_unmatched_reasons = dict(controls.attrs.get("unmatched_reasons", {}))
        if not controls.empty:
            filtered_controls = _decision_signal_filter(controls)
            if not filtered_controls.empty:
                control_parts.append(filtered_controls)
    if not parts:
        return pd.DataFrame(), {"state_rows": 0, "signal_rows": 0, "control_reference_count": 0, "control_matched_count": 0}
    signals = pd.concat([*parts, *control_parts], ignore_index=True)
    return signals, {
        "state_rows": int(len(states)),
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


def _load_outcome_files(paths: Sequence[Path], columns: Sequence[str] | None = None) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    con = duckdb.connect()
    try:
        projection = "*" if columns is None else ",".join('"' + str(name).replace('"', '""') + '"' for name in columns)
        return con.execute(f"SELECT {projection} FROM {_scan(paths)}").df()
    finally:
        con.close()


def summarize_development_outputs(output_root: str | Path) -> dict[str, Any]:
    """Read compact day outputs and produce the global development summary."""

    root = Path(output_root).resolve()
    paths = sorted((root / "outcomes").glob("date=*/outcomes.parquet"))
    if not paths:
        raise MinuteStrategyRunError("strategy_run_outcomes_missing")
    frame = _load_outcome_files(paths)
    if frame.empty:
        raise MinuteStrategyRunError("strategy_run_outcomes_empty")
    summary = {
        "output_root": str(root),
        "outcome_file_count": len(paths),
        "outcome_rows": int(len(frame)),
        "summary_overall": summarize_event_study(frame, group_by=("strategy_id", "ma_period")),
        "summary_overall_pooled": summarize_event_study(frame, group_by=("strategy_id",)),
        "summary_by_year": summarize_event_study(frame, group_by=("year", "strategy_id")) if "year" in frame else summarize_event_study(frame.assign(year=frame["signal_date"].astype(str).str[:4]), group_by=("year", "strategy_id")),
    }
    if "market_regime" in frame.columns:
        summary["summary_by_market_regime"] = summarize_event_study(frame, group_by=("market_regime", "strategy_id"))
    control_rows: list[dict[str, Any]] = []
    for strategy_id in sorted(set(frame["strategy_id"].astype(str)) - {"s0_random_matched", "s0_liquidity_matched"}):
        paired = compare_to_control(frame, strategy_id=strategy_id, control_id="s0_random_matched", metric="net_return_60m")
        control_rows.append({"strategy_id": strategy_id, "control_id": "s0_random_matched", "summary": summarize_control_comparison(paired)})
        paired_cross = compare_to_reference_control(
            frame,
            strategy_id=strategy_id,
            control_id="s0_liquidity_matched",
            metric="net_return_60m",
        )
        control_rows.append({"strategy_id": strategy_id, "control_id": "s0_liquidity_matched", "summary": summarize_control_comparison(paired_cross)})
    summary["control_comparisons"] = control_rows
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
    )
    selected_data.validate()
    selected_ma = ma_config or MinuteMAConfig(periods=tuple(selected.strategy_periods))
    selected_ma.validate()
    selected_event = event_config or EventStudyConfig()
    selected_event.validate()
    root = Path(workspace_root).resolve()
    output_root = (root / selected.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    # Keep the development date range separate from the outcome look-ahead:
    # the latter needs a few open dates after the final target date (and may
    # cross a calendar year).  The extra calendar rows are never used to
    # generate a signal; they only make forward outcomes observable.
    calendar_end = (
        date.fromisoformat(str(selected.end_date))
        + timedelta(days=max(14, int(selected.outcome_open_days) * 3))
    ).isoformat()
    calendar = load_trading_calendar(root, start_date="2010-01-01", end_date=calendar_end)
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
        daily_context = load_daily_context(root, symbols=symbols, start_date=support_start, end_date=context_end, config=selected_data)
        # ``month_end`` is a lexical partition label (e.g. YYYY-MM-31), not
        # necessarily a real date.  The last actual target date is the
        # correct inclusive bound for the historical hourly aggregation.
        hourly_history = load_hourly_history(
            root,
            symbols=symbols,
            start_date=support_start,
            end_date=target_month_dates[-1],
        )
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
            )
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
    parser.add_argument("--duckdb-threads", type=int, default=8)
    parser.add_argument("--signal-symbol-chunk-size", type=int, default=300)
    parser.add_argument("--temp-directory", default="")
    parser.add_argument("--period", type=int, action="append", dest="periods")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
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
        duckdb_threads=args.duckdb_threads,
        temp_directory=args.temp_directory or None,
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
