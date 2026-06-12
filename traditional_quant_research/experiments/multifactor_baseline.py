"""Run the phase-2 multi-factor ranking baseline."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.backtest_protocol import top_n_rebalance_backtest
from traditional_quant_research.dataset_v2 import load_pit_manifest, load_quality_report, load_tradeable_panel
from traditional_quant_research.diagnostics import quantile_returns, summarize_factor_ic
from traditional_quant_research.horizon_backtest import (
    horizon_aligned_top_n_backtest,
    period_horizon_summary,
    summarize_horizon_returns,
    yearly_horizon_summary,
)
from traditional_quant_research.multifactor import (
    add_equal_rank_score,
    add_ic_weighted_rank_score,
    add_rank_score_from_weight_table,
    factor_coverage,
    mean_daily_factor_correlation,
    neutralize_factors_by_date,
    rolling_ic_weights_by_date,
    select_low_correlation_factors,
)
from traditional_quant_research.research_panel import (
    DEFAULT_FACTOR_SET,
    FACTOR_SETS,
    add_baseline_score,
    add_cross_sectional_excess_return_labels,
    add_cross_sectional_zscores,
    build_factor_label_panel,
    factor_columns_for_set,
    normalize_factor_set,
    panel_summary,
)


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/multifactor_baseline")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/brain/references/research_log/2026-06-02_multifactor_baseline.md")
DEFAULT_HORIZON = 5
DEFAULT_TOP_N_VALUES = (50, 100, 200)
DEFAULT_FEE_BPS_VALUES = (0.0, 10.0, 20.0)
DEFAULT_REBALANCE_FREQUENCIES = ("daily", "weekly", "monthly")
LABEL_MODES = ("raw", "xsec-excess")
DEFAULT_ROLLING_WINDOW = 252
DEFAULT_ROLLING_MIN_PERIODS = 60
DEFAULT_MAX_FACTOR_CORR = 0.75
DEFAULT_NEUTRALIZE_BY = ("log_amount_mean_20d_z",)
DEFAULT_HORIZON_BUFFER_MULTIPLIERS = (1.0,)


def run_multifactor_baseline(
    *,
    root: str | None = None,
    history_start_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    horizon: int = DEFAULT_HORIZON,
    label_mode: str = "raw",
    factor_set: str | None = DEFAULT_FACTOR_SET,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    rolling_min_periods: int = DEFAULT_ROLLING_MIN_PERIODS,
    max_factor_corr: float = DEFAULT_MAX_FACTOR_CORR,
    neutralize_by: Sequence[str] = DEFAULT_NEUTRALIZE_BY,
    selected_signals: Sequence[str] | None = None,
    selection_filters: Sequence[Mapping[str, Any]] | None = None,
    top_n_values: Sequence[int] = DEFAULT_TOP_N_VALUES,
    fee_bps_values: Sequence[float] = DEFAULT_FEE_BPS_VALUES,
    rebalance_frequencies: Sequence[str] = DEFAULT_REBALANCE_FREQUENCIES,
    include_horizon_backtest: bool = False,
    horizon_buffer_multipliers: Sequence[float] = DEFAULT_HORIZON_BUFFER_MULTIPLIERS,
    execution_constraints: bool = False,
    limit_threshold: float = 0.095,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    """Run a reproducible multi-factor baseline on the v2 PIT tradeable panel."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if label_mode not in LABEL_MODES:
        raise ValueError(f"unsupported label_mode: {label_mode}")
    selected_factor_set = normalize_factor_set(factor_set)
    if rolling_window <= 0:
        raise ValueError("rolling_window must be positive")
    if rolling_min_periods <= 0:
        raise ValueError("rolling_min_periods must be positive")
    if max_factor_corr < 0 or max_factor_corr > 1:
        raise ValueError("max_factor_corr must be between 0 and 1")
    if any(top_n <= 0 for top_n in top_n_values):
        raise ValueError("top_n_values must be positive")
    if any(value < 1.0 for value in horizon_buffer_multipliers):
        raise ValueError("horizon_buffer_multipliers must be at least 1.0")
    if limit_threshold <= 0:
        raise ValueError("limit_threshold must be positive")

    manifest = load_pit_manifest(root)
    quality = load_quality_report(root)
    dataset = manifest.get("dataset", {})
    effective_start = start_date or dataset.get("date_min")
    effective_end = end_date or dataset.get("date_max")
    effective_history_start = history_start_date or effective_start
    raw_label_col = f"fwd_ret_{horizon}d"
    label_col = raw_label_col if label_mode == "raw" else f"xsec_excess_ret_{horizon}d"
    run_id = f"multifactor_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_panel = load_tradeable_panel(root, start_date=effective_history_start, end_date=effective_end)
    factor_panel = build_factor_label_panel(raw_panel, horizons=tuple(sorted({1, 5, 20, horizon})), factor_set=selected_factor_set)
    factor_panel = add_cross_sectional_excess_return_labels(factor_panel, horizons=(horizon,))
    raw_factor_columns = factor_columns_for_set(selected_factor_set)
    factor_panel = add_cross_sectional_zscores(factor_panel, raw_factor_columns)
    signal_columns = [f"{column}_z" for column in raw_factor_columns]
    factor_panel = add_baseline_score(factor_panel, score_columns=signal_columns)
    evaluation_panel = filter_panel_dates(factor_panel, start_date=effective_start, end_date=effective_end)

    if label_col not in evaluation_panel.columns:
        raise ValueError(f"label not generated: {label_col}")

    single_factor_ic = summarize_factor_ic(evaluation_panel, signal_columns, label_col)
    factor_directions = directions_from_ic(single_factor_ic, signal_columns)
    factor_panel = add_equal_rank_score(
        factor_panel,
        signal_columns,
        directions=factor_directions,
        score_col="multifactor_equal_rank_score",
        min_factors=3,
    )
    factor_panel = add_ic_weighted_rank_score(
        factor_panel,
        signal_columns,
        single_factor_ic,
        directions=factor_directions,
        score_col="multifactor_ic_weighted_score",
        min_factors=3,
    )
    rolling_weights = rolling_ic_weights_by_date(
        factor_panel,
        signal_columns,
        label_col,
        window=rolling_window,
        min_periods=rolling_min_periods,
    )
    factor_panel = add_rank_score_from_weight_table(
        factor_panel,
        signal_columns,
        rolling_weights,
        score_col="multifactor_rolling_ic_weighted_score",
        min_factors=3,
    )
    evaluation_panel = filter_panel_dates(factor_panel, start_date=effective_start, end_date=effective_end)
    coverage = factor_coverage(evaluation_panel, signal_columns)
    correlation = mean_daily_factor_correlation(evaluation_panel, signal_columns)
    low_corr_factor_columns = select_low_correlation_factors(
        correlation,
        signal_columns,
        ic_summary=single_factor_ic,
        max_abs_corr=max_factor_corr,
    )
    if low_corr_factor_columns:
        factor_panel = add_equal_rank_score(
            factor_panel,
            low_corr_factor_columns,
            directions=factor_directions,
            score_col="multifactor_low_corr_rank_score",
            min_factors=min(3, len(low_corr_factor_columns)),
        )
    neutralizer_columns = [column for column in neutralize_by if column]
    neutralized_source_columns = [column for column in signal_columns if column not in neutralizer_columns]
    neutralized_factor_columns: list[str] = []
    if neutralizer_columns and neutralized_source_columns:
        missing_neutralizers = [column for column in neutralizer_columns if column not in factor_panel.columns]
        if missing_neutralizers:
            raise ValueError(f"missing neutralizer columns: {missing_neutralizers}")
        factor_panel = neutralize_factors_by_date(
            factor_panel,
            neutralized_source_columns,
            neutralizer_columns,
        )
        neutralized_factor_columns = [f"{column}_neutral" for column in neutralized_source_columns]
        neutralized_directions = {
            f"{column}_neutral": factor_directions.get(column, 1)
            for column in neutralized_source_columns
        }
        factor_panel = add_equal_rank_score(
            factor_panel,
            neutralized_factor_columns,
            directions=neutralized_directions,
            score_col="multifactor_neutral_rank_score",
            min_factors=min(3, len(neutralized_factor_columns)),
        )
    multifactor_signals = [
        "baseline_score",
        "multifactor_equal_rank_score",
        "multifactor_ic_weighted_score",
        "multifactor_rolling_ic_weighted_score",
    ]
    if low_corr_factor_columns:
        multifactor_signals.append("multifactor_low_corr_rank_score")
    if neutralized_factor_columns:
        multifactor_signals.append("multifactor_neutral_rank_score")
    if selected_signals is not None:
        selected = [signal for signal in selected_signals if signal]
        if not selected:
            raise ValueError("selected_signals must not be empty")
        unknown_signals = sorted(set(selected) - set(multifactor_signals))
        if unknown_signals:
            raise ValueError(f"unknown selected signals: {unknown_signals}")
        multifactor_signals = selected

    evaluation_panel = filter_panel_dates(factor_panel, start_date=effective_start, end_date=effective_end)
    multifactor_ic = summarize_factor_ic(evaluation_panel, multifactor_signals, label_col)
    quantile_frames = []
    for signal in multifactor_signals:
        quantile = quantile_returns(evaluation_panel, signal, label_col, quantiles=5)
        if not quantile.empty:
            quantile.insert(0, "signal", signal)
            quantile_frames.append(quantile)
    quantile_summary = pd.concat(quantile_frames, ignore_index=True) if quantile_frames else pd.DataFrame()
    quantile_spread = summarize_quantile_spread(quantile_summary)
    portfolio_panel, selection_filter_report = apply_selection_filters(evaluation_panel, selection_filters)
    horizon_panel = apply_selection_filters_to_signals(evaluation_panel, multifactor_signals, selection_filters)
    backtest_summary = run_backtest_grid(
        portfolio_panel,
        multifactor_signals,
        label_col,
        top_n_values=top_n_values,
        fee_bps_values=fee_bps_values,
        rebalance_frequencies=rebalance_frequencies,
    )
    rolling_weight_audit = rolling_ic_weight_audit(rolling_weights)
    horizon_backtest_summary, horizon_backtest_yearly_summary, horizon_backtest_period_summary, basket_exposure_summary = (
        run_horizon_backtest_tables(
            horizon_panel,
            multifactor_signals,
            horizon=horizon,
            top_n_values=top_n_values,
            fee_bps_values=fee_bps_values,
            rebalance_frequencies=rebalance_frequencies,
            buffer_multipliers=horizon_buffer_multipliers,
            exposure_columns=signal_columns,
            execution_constraints=execution_constraints,
            limit_threshold=limit_threshold,
        )
        if include_horizon_backtest
        else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    )

    result = {
        "run_id": run_id,
        "snapshot_id": manifest.get("snapshot_id"),
        "snapshot_path": manifest.get("snapshot_path"),
        "history_start_date": effective_history_start,
        "start_date": effective_start,
        "end_date": effective_end,
        "horizon": int(horizon),
        "label_mode": label_mode,
        "factor_set": selected_factor_set,
        "raw_label": raw_label_col,
        "label": label_col,
        "label_mode_note": label_mode_note(label_mode),
        "backtest_return_note": backtest_return_note(horizon),
        "include_horizon_backtest": bool(include_horizon_backtest),
        "horizon_backtest_note": horizon_backtest_note(horizon, execution_constraints=execution_constraints, limit_threshold=limit_threshold),
        "horizon_buffer_multipliers": [float(value) for value in horizon_buffer_multipliers],
        "execution_constraints": bool(execution_constraints),
        "limit_threshold": float(limit_threshold),
        "rolling_window": int(rolling_window),
        "rolling_min_periods": int(rolling_min_periods),
        "rolling_fallback_rate": rolling_fallback_rate(rolling_weights),
        "max_factor_corr": float(max_factor_corr),
        "low_corr_factor_columns": low_corr_factor_columns,
        "neutralize_by": neutralizer_columns,
        "neutralized_factor_columns": neutralized_factor_columns,
        "selected_signals": list(selected_signals) if selected_signals is not None else [],
        "selection_filters": [dict(rule) for rule in selection_filters] if selection_filters else [],
        "selection_filter_report": selection_filter_report,
        "top_n_values": [int(value) for value in top_n_values],
        "fee_bps_values": [float(value) for value in fee_bps_values],
        "rebalance_frequencies": list(rebalance_frequencies),
        "raw_factor_columns": raw_factor_columns,
        "signal_columns": signal_columns,
        "multifactor_signals": multifactor_signals,
        "factor_directions": factor_directions,
        "history_panel": panel_summary(factor_panel),
        "panel": panel_summary(evaluation_panel),
        "portfolio_panel": panel_summary(portfolio_panel),
        "quality": {
            "failure_count": quality.get("failure_count"),
            "missing_bar_rows": quality.get("missing_bar_rows"),
            "st_rows": quality.get("st_rows"),
            "suspended_like_rows": quality.get("suspended_like_rows"),
        },
        "best_rank_ic": best_rank_ic_rows(multifactor_ic),
        "best_quantile_spread": best_quantile_spread_rows(quantile_spread),
        "best_backtests": best_backtest_configs(backtest_summary),
        "best_horizon_backtests": best_backtest_configs(horizon_backtest_summary),
        "horizon_backtest_years": sorted(horizon_backtest_yearly_summary["year"].dropna().astype(int).unique().tolist()) if not horizon_backtest_yearly_summary.empty else [],
        "horizon_backtest_period_types": sorted(horizon_backtest_period_summary["period_type"].dropna().unique().tolist()) if not horizon_backtest_period_summary.empty else [],
        "horizon_backtest_period_count": int(horizon_backtest_period_summary["period"].dropna().nunique()) if not horizon_backtest_period_summary.empty else 0,
        "basket_exposure_period_types": sorted(basket_exposure_summary["period_type"].dropna().unique().tolist()) if not basket_exposure_summary.empty else [],
        "basket_exposure_factor_count": int(basket_exposure_summary["factor"].dropna().nunique()) if not basket_exposure_summary.empty else 0,
        "rolling_weight_audit_period_types": sorted(rolling_weight_audit["period_type"].dropna().unique().tolist()) if not rolling_weight_audit.empty else [],
        "output_dir": str(run_dir),
    }

    coverage.to_csv(run_dir / "factor_coverage.csv", index=False, encoding="utf-8-sig")
    correlation.to_csv(run_dir / "factor_correlation.csv", encoding="utf-8-sig")
    single_factor_ic.to_csv(run_dir / "single_factor_ic.csv", index=False, encoding="utf-8-sig")
    multifactor_ic.to_csv(run_dir / "multifactor_ic.csv", index=False, encoding="utf-8-sig")
    quantile_summary.to_csv(run_dir / "quantile_returns.csv", index=False, encoding="utf-8-sig")
    quantile_spread.to_csv(run_dir / "quantile_spread.csv", index=False, encoding="utf-8-sig")
    rolling_weights.to_csv(run_dir / "rolling_ic_weights.csv", index=False, encoding="utf-8-sig")
    rolling_weight_audit.to_csv(run_dir / "rolling_ic_weight_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"factor": low_corr_factor_columns}).to_csv(run_dir / "low_corr_factors.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"factor": neutralized_factor_columns}).to_csv(run_dir / "neutralized_factors.csv", index=False, encoding="utf-8-sig")
    backtest_summary.to_csv(run_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")
    if include_horizon_backtest:
        horizon_backtest_summary.to_csv(run_dir / "horizon_backtest_summary.csv", index=False, encoding="utf-8-sig")
        horizon_backtest_yearly_summary.to_csv(run_dir / "horizon_backtest_yearly_summary.csv", index=False, encoding="utf-8-sig")
        horizon_backtest_period_summary.to_csv(run_dir / "horizon_backtest_period_summary.csv", index=False, encoding="utf-8-sig")
        basket_exposure_summary.to_csv(run_dir / "basket_factor_exposure.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_multifactor_markdown(
        result,
        coverage,
        single_factor_ic,
        multifactor_ic,
        quantile_summary,
        quantile_spread,
        backtest_summary,
        horizon_backtest_summary,
        horizon_backtest_yearly_summary,
        horizon_backtest_period_summary,
        basket_exposure_summary,
        rolling_weight_audit,
    )
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        research_log_path.write_text(markdown, encoding="utf-8")
    return result


def directions_from_ic(ic_summary: pd.DataFrame, signal_columns: Sequence[str]) -> dict[str, int]:
    """Use in-sample diagnostic RankIC sign as a simple factor direction rule."""

    if ic_summary.empty:
        return {signal: 1 for signal in signal_columns}
    lookup = ic_summary.drop_duplicates("signal", keep="first").set_index("signal")["mean_rank_ic"]
    directions: dict[str, int] = {}
    for signal in signal_columns:
        value = pd.to_numeric(lookup.get(signal, np.nan), errors="coerce")
        directions[signal] = -1 if pd.notna(value) and float(value) < 0 else 1
    return directions


def rolling_fallback_rate(rolling_weights: pd.DataFrame) -> float:
    if rolling_weights.empty or "is_fallback" not in rolling_weights.columns:
        return np.nan
    by_date = rolling_weights.groupby("date", sort=True)["is_fallback"].max()
    return float(pd.to_numeric(by_date, errors="coerce").mean()) if not by_date.empty else np.nan


def filter_panel_dates(
    frame: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
    date_col: str = "date",
) -> pd.DataFrame:
    """Return a date-bounded evaluation view without mutating the source panel."""

    if frame.empty or date_col not in frame.columns:
        return frame.copy()
    output = frame.copy()
    dates = pd.to_datetime(output[date_col])
    if start_date is not None:
        output = output.loc[dates >= pd.Timestamp(start_date)]
        dates = pd.to_datetime(output[date_col])
    if end_date is not None:
        output = output.loc[dates <= pd.Timestamp(end_date)]
    return output.reset_index(drop=True)


def parse_selection_filters(spec: str) -> list[dict[str, Any]]:
    """Parse comma-separated portfolio selection filter rules."""

    if not spec or not spec.strip():
        return []
    rules: list[dict[str, Any]] = []
    pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|>|<|==)\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*$")
    for raw_rule in spec.split(","):
        if not raw_rule.strip():
            continue
        match = pattern.match(raw_rule)
        if not match:
            raise ValueError(f"invalid selection filter: {raw_rule}")
        column, operator, threshold = match.groups()
        rules.append({"column": column, "operator": operator, "threshold": float(threshold)})
    return rules


def apply_selection_filters(
    frame: pd.DataFrame,
    rules: Sequence[Mapping[str, Any]] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Filter candidate portfolio rows while leaving diagnostic inputs unchanged."""

    if not rules:
        return frame.copy(), {
            "enabled": False,
            "rules": [],
            "rows_before": int(len(frame)),
            "rows_after": int(len(frame)),
            "rows_removed": 0,
            "kept_rate": 1.0 if len(frame) else np.nan,
            "date_count_before": int(frame["date"].nunique()) if "date" in frame.columns else 0,
            "date_count_after": int(frame["date"].nunique()) if "date" in frame.columns else 0,
        }
    output = frame.copy()
    mask, normalized_rules = selection_filter_mask(output, rules)
    filtered = output.loc[mask].reset_index(drop=True)
    rows_before = int(len(output))
    rows_after = int(len(filtered))
    return filtered, {
        "enabled": True,
        "rules": normalized_rules,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_removed": rows_before - rows_after,
        "kept_rate": float(rows_after / rows_before) if rows_before else np.nan,
        "date_count_before": int(output["date"].nunique()) if "date" in output.columns else 0,
        "date_count_after": int(filtered["date"].nunique()) if "date" in filtered.columns else 0,
    }


def selection_filter_mask(
    frame: pd.DataFrame,
    rules: Sequence[Mapping[str, Any]],
) -> tuple[pd.Series, list[dict[str, Any]]]:
    """Build a boolean mask for portfolio selection filter rules."""

    mask = pd.Series(True, index=frame.index)
    normalized_rules: list[dict[str, Any]] = []
    for rule in rules:
        column = str(rule.get("column", ""))
        operator = str(rule.get("operator", ""))
        threshold = float(rule.get("threshold"))
        if column not in frame.columns:
            raise ValueError(f"selection filter column not found: {column}")
        values = pd.to_numeric(frame[column], errors="coerce")
        if operator == ">=":
            rule_mask = values >= threshold
        elif operator == "<=":
            rule_mask = values <= threshold
        elif operator == ">":
            rule_mask = values > threshold
        elif operator == "<":
            rule_mask = values < threshold
        elif operator == "==":
            rule_mask = values == threshold
        else:
            raise ValueError(f"unsupported selection filter operator: {operator}")
        mask &= rule_mask.fillna(False)
        normalized_rules.append({"column": column, "operator": operator, "threshold": threshold})
    return mask, normalized_rules


def apply_selection_filters_to_signals(
    frame: pd.DataFrame,
    signal_columns: Sequence[str],
    rules: Sequence[Mapping[str, Any]] | None,
) -> pd.DataFrame:
    """Keep full price path but make ineligible signal-date rows unselectable."""

    output = frame.copy()
    if not rules:
        return output
    eligible_mask, _ = selection_filter_mask(output, rules)
    ineligible_mask = ~eligible_mask
    for signal in signal_columns:
        if signal not in output.columns:
            raise ValueError(f"signal column not found: {signal}")
        output.loc[ineligible_mask, signal] = np.nan
    return output


def rolling_ic_weight_audit(rolling_weights: pd.DataFrame, *, periods: Sequence[str] = ("M", "Q")) -> pd.DataFrame:
    """Summarize rolling IC weights by calendar month and quarter."""

    columns = [
        "period",
        "period_type",
        "factor",
        "mean_weight",
        "median_weight",
        "max_weight",
        "mean_rank_ic",
        "positive_direction_rate",
        "fallback_rate",
        "mean_history_days",
        "date_count",
    ]
    required = {"date", "factor", "weight", "mean_rank_ic", "direction", "history_days", "is_fallback"}
    if rolling_weights.empty:
        return pd.DataFrame(columns=columns)
    missing = sorted(required - set(rolling_weights.columns))
    if missing:
        raise ValueError(f"rolling_weights missing required columns: {missing}")

    working = rolling_weights.copy()
    working["date"] = pd.to_datetime(working["date"])
    for column in ["weight", "mean_rank_ic", "direction", "history_days"]:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working["is_fallback"] = working["is_fallback"].astype(bool)
    period_labels = {"M": "monthly", "Q": "quarterly"}
    rows: list[dict[str, Any]] = []
    for period in periods:
        if period not in period_labels:
            raise ValueError(f"unsupported period: {period}")
        period_work = working.copy()
        period_work["_period"] = period_work["date"].dt.to_period(period)
        for (period_value, factor), group in period_work.groupby(["_period", "factor"], sort=True):
            rows.append(
                {
                    "period": str(period_value),
                    "period_type": period_labels[period],
                    "factor": factor,
                    "mean_weight": float(group["weight"].mean()),
                    "median_weight": float(group["weight"].median()),
                    "max_weight": float(group["weight"].max()),
                    "mean_rank_ic": float(group["mean_rank_ic"].mean()),
                    "positive_direction_rate": float((group["direction"] > 0).mean()),
                    "fallback_rate": float(group["is_fallback"].mean()),
                    "mean_history_days": float(group["history_days"].mean()),
                    "date_count": int(group["date"].nunique()),
                }
            )
    return pd.DataFrame(rows).loc[:, columns]


def selected_basket_factor_exposure(
    frame: pd.DataFrame,
    trades: pd.DataFrame,
    exposure_columns: Sequence[str],
    *,
    signal_col: str,
    horizon: int,
    rebalance_frequency: str,
    top_n: int,
    buffer_multiplier: float,
    periods: Sequence[str] = ("M", "Q"),
    date_col: str = "date",
    code_col: str = "code",
) -> pd.DataFrame:
    """Summarize selected Top-N basket factor exposures by exit month/quarter."""

    columns = [
        "signal",
        "period",
        "period_type",
        "factor",
        "horizon",
        "rebalance_frequency",
        "top_n",
        "buffer_multiplier",
        "trade_count",
        "selected_count",
        "selected_mean",
        "selected_median",
        "selected_std",
        "universe_mean",
        "active_exposure",
    ]
    if frame.empty or trades.empty or not exposure_columns:
        return pd.DataFrame(columns=columns)
    required_frame = {date_col, code_col, *exposure_columns}
    required_trades = {"signal_date", "exit_date", "codes"}
    missing_frame = sorted(required_frame - set(frame.columns))
    missing_trades = sorted(required_trades - set(trades.columns))
    if missing_frame:
        raise ValueError(f"frame missing required columns: {missing_frame}")
    if missing_trades:
        raise ValueError(f"trades missing required columns: {missing_trades}")

    panel = frame.loc[:, [date_col, code_col, *exposure_columns]].copy()
    panel[date_col] = pd.to_datetime(panel[date_col])
    panel[code_col] = panel[code_col].astype(str)
    for column in exposure_columns:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")

    trade_rows: list[dict[str, Any]] = []
    trades_work = trades.copy().reset_index(drop=True)
    trades_work["signal_date"] = pd.to_datetime(trades_work["signal_date"])
    trades_work["exit_date"] = pd.to_datetime(trades_work["exit_date"])
    for trade_id, trade in trades_work.iterrows():
        codes = [code.strip() for code in str(trade["codes"]).split(",") if code.strip()]
        for code in codes:
            trade_rows.append(
                {
                    "trade_id": int(trade_id),
                    "signal_date": pd.Timestamp(trade["signal_date"]),
                    "exit_date": pd.Timestamp(trade["exit_date"]),
                    code_col: code,
                }
            )
    if not trade_rows:
        return pd.DataFrame(columns=columns)

    selected = pd.DataFrame(trade_rows)
    selected = selected.merge(
        panel.rename(columns={date_col: "signal_date"}),
        on=["signal_date", code_col],
        how="left",
    )
    universe_by_date = panel.groupby(date_col, sort=True)[list(exposure_columns)].mean(numeric_only=True)
    per_trade_rows: list[dict[str, Any]] = []
    for trade_id, group in selected.groupby("trade_id", sort=True):
        signal_date = pd.Timestamp(group["signal_date"].iloc[0])
        exit_date = pd.Timestamp(group["exit_date"].iloc[0])
        universe_row = universe_by_date.loc[signal_date] if signal_date in universe_by_date.index else pd.Series(dtype=float)
        for factor in exposure_columns:
            values = pd.to_numeric(group[factor], errors="coerce").dropna()
            if values.empty:
                continue
            universe_mean = pd.to_numeric(universe_row.get(factor, np.nan), errors="coerce")
            selected_mean = float(values.mean())
            per_trade_rows.append(
                {
                    "trade_id": int(trade_id),
                    "signal_date": signal_date,
                    "exit_date": exit_date,
                    "factor": factor,
                    "selected_mean": selected_mean,
                    "selected_median": float(values.median()),
                    "selected_std": float(values.std(ddof=0)) if len(values) > 1 else 0.0,
                    "universe_mean": float(universe_mean) if pd.notna(universe_mean) else np.nan,
                    "selected_count": int(len(values)),
                }
            )
    if not per_trade_rows:
        return pd.DataFrame(columns=columns)

    per_trade = pd.DataFrame(per_trade_rows)
    period_labels = {"M": "monthly", "Q": "quarterly"}
    rows: list[dict[str, Any]] = []
    for period in periods:
        if period not in period_labels:
            raise ValueError(f"unsupported period: {period}")
        work = per_trade.copy()
        work["_period"] = work["exit_date"].dt.to_period(period)
        for (period_value, factor), group in work.groupby(["_period", "factor"], sort=True):
            selected_mean = float(group["selected_mean"].mean())
            universe_mean = float(group["universe_mean"].mean())
            rows.append(
                {
                    "signal": signal_col,
                    "period": str(period_value),
                    "period_type": period_labels[period],
                    "factor": factor,
                    "horizon": int(horizon),
                    "rebalance_frequency": rebalance_frequency,
                    "top_n": int(top_n),
                    "buffer_multiplier": float(buffer_multiplier),
                    "trade_count": int(group["trade_id"].nunique()),
                    "selected_count": int(group["selected_count"].sum()),
                    "selected_mean": selected_mean,
                    "selected_median": float(group["selected_median"].mean()),
                    "selected_std": float(group["selected_std"].mean()),
                    "universe_mean": universe_mean,
                    "active_exposure": selected_mean - universe_mean,
                }
            )
    return pd.DataFrame(rows).loc[:, columns]


def run_backtest_grid(
    frame: pd.DataFrame,
    signal_columns: Sequence[str],
    label_col: str,
    *,
    top_n_values: Sequence[int],
    fee_bps_values: Sequence[float],
    rebalance_frequencies: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for signal_col in signal_columns:
        for frequency in rebalance_frequencies:
            for top_n in top_n_values:
                for fee_bps in fee_bps_values:
                    backtest = top_n_rebalance_backtest(
                        frame,
                        signal_col,
                        label_col,
                        top_n=int(top_n),
                        fee_bps=float(fee_bps),
                        rebalance_frequency=frequency,
                    )
                    row = {"signal": signal_col}
                    row.update(backtest.summary)
                    rows.append(row)
    return pd.DataFrame(rows)


def apply_horizon_fee(trades: pd.DataFrame, *, fee_bps: float) -> pd.DataFrame:
    """Apply a fee rate to an existing horizon trade path without changing holdings."""

    output = trades.copy()
    if output.empty:
        return output
    fee = float(fee_bps) / 10000.0
    output["cost"] = pd.to_numeric(output["turnover"], errors="coerce") * fee
    output["net_return"] = pd.to_numeric(output["gross_return"], errors="coerce") - output["cost"]
    return output


def summarize_horizon_trade_table(
    trades: pd.DataFrame,
    *,
    horizon: int,
    rebalance_frequency: str,
    top_n: int,
    fee_bps: float,
    buffer_multiplier: float,
    execution_constraints: bool,
    limit_threshold: float,
    capital_col: str | None = None,
) -> dict[str, Any]:
    """Summarize a horizon trade table after fee application."""

    summary = summarize_horizon_returns(
        trades["net_return"].tolist() if not trades.empty and "net_return" in trades.columns else [],
        horizon=horizon,
        rebalance_frequency=rebalance_frequency,
        top_n=top_n,
        fee_bps=fee_bps,
        non_overlapping=True,
        buffer_multiplier=buffer_multiplier,
        execution_constraints=execution_constraints,
        limit_threshold=limit_threshold,
        capital_col=capital_col,
    )
    summary.update(
        {
            "periods": int(len(trades)),
            "mean_turnover": float(trades["turnover"].mean()) if not trades.empty and "turnover" in trades.columns else np.nan,
            "mean_gross_return": float(trades["gross_return"].mean()) if not trades.empty and "gross_return" in trades.columns else np.nan,
            "mean_net_return": float(trades["net_return"].mean()) if not trades.empty and "net_return" in trades.columns else np.nan,
            "mean_holdings": float(trades["holdings"].mean()) if not trades.empty and "holdings" in trades.columns else np.nan,
            "mean_requested_holdings": float(trades["requested_holdings"].mean()) if not trades.empty and "requested_holdings" in trades.columns else np.nan,
            "mean_capital_scale": float(trades["capital_scale"].mean()) if not trades.empty and "capital_scale" in trades.columns else np.nan,
            "blocked_entry_count": int(trades["blocked_entry_count"].sum()) if not trades.empty and "blocked_entry_count" in trades.columns else 0,
            "entry_limit_up_count": int(trades["entry_limit_up_count"].sum()) if not trades.empty and "entry_limit_up_count" in trades.columns else 0,
            "exit_delayed_count": int(trades["exit_delayed_count"].sum()) if not trades.empty and "exit_delayed_count" in trades.columns else 0,
            "exit_limit_down_count": int(trades["exit_limit_down_count"].sum()) if not trades.empty and "exit_limit_down_count" in trades.columns else 0,
        }
    )
    return summary


def run_horizon_backtest_grid(
    frame: pd.DataFrame,
    signal_columns: Sequence[str],
    *,
    horizon: int,
    top_n_values: Sequence[int],
    fee_bps_values: Sequence[float],
    rebalance_frequencies: Sequence[str],
    buffer_multipliers: Sequence[float] = DEFAULT_HORIZON_BUFFER_MULTIPLIERS,
    execution_constraints: bool = False,
    limit_threshold: float = 0.095,
    capital_col: str | None = None,
) -> pd.DataFrame:
    summary, _, _, _ = run_horizon_backtest_tables(
        frame,
        signal_columns,
        horizon=horizon,
        top_n_values=top_n_values,
        fee_bps_values=fee_bps_values,
        rebalance_frequencies=rebalance_frequencies,
        buffer_multipliers=buffer_multipliers,
        execution_constraints=execution_constraints,
        limit_threshold=limit_threshold,
        capital_col=capital_col,
    )
    return summary


def run_horizon_backtest_tables(
    frame: pd.DataFrame,
    signal_columns: Sequence[str],
    *,
    horizon: int,
    top_n_values: Sequence[int],
    fee_bps_values: Sequence[float],
    rebalance_frequencies: Sequence[str],
    buffer_multipliers: Sequence[float] = DEFAULT_HORIZON_BUFFER_MULTIPLIERS,
    exposure_columns: Sequence[str] = (),
    execution_constraints: bool = False,
    limit_threshold: float = 0.095,
    capital_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []
    period_frames: list[pd.DataFrame] = []
    exposure_frames: list[pd.DataFrame] = []
    first_fee_bps = float(fee_bps_values[0]) if fee_bps_values else np.nan
    for signal_col in signal_columns:
        for frequency in rebalance_frequencies:
            for top_n in top_n_values:
                for buffer_multiplier in buffer_multipliers:
                    base_backtest = horizon_aligned_top_n_backtest(
                        frame,
                        signal_col,
                        horizon=horizon,
                        top_n=int(top_n),
                        fee_bps=0.0,
                        rebalance_frequency=frequency,
                        buffer_multiplier=float(buffer_multiplier),
                        execution_constraints=execution_constraints,
                        limit_threshold=limit_threshold,
                        capital_col=capital_col,
                    )
                    for fee_bps in fee_bps_values:
                        trades = apply_horizon_fee(base_backtest.trades, fee_bps=float(fee_bps))
                        row = {"signal": signal_col}
                        row.update(
                            summarize_horizon_trade_table(
                                trades,
                                horizon=horizon,
                                rebalance_frequency=frequency,
                                top_n=int(top_n),
                                fee_bps=float(fee_bps),
                                buffer_multiplier=float(buffer_multiplier),
                                execution_constraints=execution_constraints,
                                limit_threshold=limit_threshold,
                                capital_col=capital_col,
                            )
                        )
                        rows.append(row)
                        yearly = yearly_horizon_summary(
                            trades,
                            horizon=horizon,
                            rebalance_frequency=frequency,
                            top_n=int(top_n),
                            fee_bps=float(fee_bps),
                            non_overlapping=True,
                            buffer_multiplier=float(buffer_multiplier),
                            execution_constraints=execution_constraints,
                            limit_threshold=limit_threshold,
                            signal_col=signal_col,
                        )
                        if not yearly.empty:
                            yearly_frames.append(yearly)
                        for period in ("M", "Q"):
                            period_summary = period_horizon_summary(
                                trades,
                                period=period,
                                horizon=horizon,
                                rebalance_frequency=frequency,
                                top_n=int(top_n),
                                fee_bps=float(fee_bps),
                                non_overlapping=True,
                                buffer_multiplier=float(buffer_multiplier),
                                execution_constraints=execution_constraints,
                                limit_threshold=limit_threshold,
                                signal_col=signal_col,
                            )
                            if not period_summary.empty:
                                period_frames.append(period_summary)
                        if exposure_columns and float(fee_bps) == first_fee_bps:
                            exposure = selected_basket_factor_exposure(
                                frame,
                                trades,
                                exposure_columns,
                                signal_col=signal_col,
                                horizon=horizon,
                                rebalance_frequency=frequency,
                                top_n=int(top_n),
                                buffer_multiplier=float(buffer_multiplier),
                            )
                            if not exposure.empty:
                                exposure_frames.append(exposure)
    summary = pd.DataFrame(rows)
    yearly_summary = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    period_summary = pd.concat(period_frames, ignore_index=True) if period_frames else pd.DataFrame()
    exposure_summary = pd.concat(exposure_frames, ignore_index=True) if exposure_frames else pd.DataFrame()
    return summary, yearly_summary, period_summary, exposure_summary


def run_horizon_backtest_yearly_grid(
    frame: pd.DataFrame,
    signal_columns: Sequence[str],
    *,
    horizon: int,
    top_n_values: Sequence[int],
    fee_bps_values: Sequence[float],
    rebalance_frequencies: Sequence[str],
    buffer_multipliers: Sequence[float] = DEFAULT_HORIZON_BUFFER_MULTIPLIERS,
    execution_constraints: bool = False,
    limit_threshold: float = 0.095,
) -> pd.DataFrame:
    _, yearly_summary, _, _ = run_horizon_backtest_tables(
        frame,
        signal_columns,
        horizon=horizon,
        top_n_values=top_n_values,
        fee_bps_values=fee_bps_values,
        rebalance_frequencies=rebalance_frequencies,
        buffer_multipliers=buffer_multipliers,
        execution_constraints=execution_constraints,
        limit_threshold=limit_threshold,
    )
    return yearly_summary


def summarize_quantile_spread(quantile_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize high-minus-low quantile return spreads per signal."""

    if quantile_summary.empty:
        return pd.DataFrame(columns=["signal", "low_quantile", "high_quantile", "low_mean_return", "high_mean_return", "q_high_minus_low"])
    rows: list[dict[str, Any]] = []
    for signal, group in quantile_summary.groupby("signal", sort=True):
        low_quantile = int(group["quantile"].min())
        high_quantile = int(group["quantile"].max())
        low = group.loc[group["quantile"] == low_quantile, "mean_return"].iloc[0]
        high = group.loc[group["quantile"] == high_quantile, "mean_return"].iloc[0]
        rows.append(
            {
                "signal": signal,
                "low_quantile": low_quantile,
                "high_quantile": high_quantile,
                "low_mean_return": float(low),
                "high_mean_return": float(high),
                "q_high_minus_low": float(high - low),
            }
        )
    return pd.DataFrame(rows)


def best_rank_ic_rows(ic_summary: pd.DataFrame, *, top: int = 5) -> list[dict[str, Any]]:
    if ic_summary.empty:
        return []
    return ic_summary.sort_values("mean_rank_ic", ascending=False).head(top).to_dict("records")


def best_quantile_spread_rows(spread: pd.DataFrame, *, top: int = 5) -> list[dict[str, Any]]:
    if spread.empty:
        return []
    return spread.sort_values("q_high_minus_low", ascending=False).head(top).to_dict("records")


def best_backtest_configs(summary_table: pd.DataFrame, *, top: int = 10) -> list[dict[str, Any]]:
    if summary_table.empty:
        return []
    return summary_table.sort_values(["sharpe", "annualized_return"], ascending=[False, False]).head(top).to_dict("records")


def label_mode_note(label_mode: str) -> str:
    if label_mode == "xsec-excess":
        return "`xsec-excess` subtracts the same-date tradeable-universe mean forward return, so Top-N metrics are alpha-style ranking diagnostics rather than standalone investable portfolio returns."
    return "`raw` keeps the same-date market move in the label, so IC and quantile returns are closer to absolute return diagnostics but still need execution-aware portfolio validation."


def backtest_return_note(horizon: int) -> str:
    if horizon <= 1:
        return "Top-N backtests use the generated 1-day forward return as the rebalance-period return proxy; they remain diagnostic until execution constraints are added."
    return (
        f"Top-N backtests use the generated {horizon}-day forward return as each selected rebalance-date return. "
        "Daily or otherwise overlapping rebalance schedules can overstate portfolio PnL; treat these rows as holding-period ranking diagnostics until a horizon-aligned portfolio simulator is used."
    )


def horizon_backtest_note(horizon: int, *, execution_constraints: bool = False, limit_threshold: float = 0.095) -> str:
    base = (
        f"Horizon-aligned Top-N backtests use explicit next-open entry and {horizon}-tradeable-day close exit with non-overlapping baskets. "
        "They are stricter than label-column diagnostic Top-N rows. Buffer multipliers above 1.0 keep prior holdings that remain inside the expanded rank buffer to reduce turnover. "
    )
    if execution_constraints:
        return (
            base
            + f"Execution constraints are enabled with limit_threshold `{limit_threshold}`: next-session limit-up or non-tradeable entries are not filled, unfilled capital stays in cash, and target exits blocked by limit-down/non-tradeable status are delayed to the next sellable close. "
            "This is still an approximation and does not model queue priority, partial fills, slippage, or market impact."
        )
    return base + "These rows still need limit-up/down, suspension holding, slippage, and impact-cost constraints before strategy-candidate promotion."


def render_multifactor_markdown(
    result: dict[str, Any],
    coverage: pd.DataFrame,
    single_factor_ic: pd.DataFrame,
    multifactor_ic: pd.DataFrame,
    quantile_summary: pd.DataFrame,
    quantile_spread: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    horizon_backtest_summary: pd.DataFrame,
    horizon_backtest_yearly_summary: pd.DataFrame,
    horizon_backtest_period_summary: pd.DataFrame,
    basket_exposure_summary: pd.DataFrame,
    rolling_weight_audit: pd.DataFrame,
) -> str:
    best_rank_ic = result.get("best_rank_ic") or [{}]
    best_quantile_spread = result.get("best_quantile_spread") or [{}]
    best_backtests = result.get("best_backtests") or [{}]
    best_signal = best_rank_ic[0].get("signal")
    best_spread_signal = best_quantile_spread[0].get("signal")
    best_backtest_signal = best_backtests[0].get("signal")
    conflict_note = "一致" if best_signal == best_spread_signal == best_backtest_signal else "不一致"
    lines = [
        "# 2026-06-02 Multifactor Baseline",
        "",
        "- Hypothesis: 方向校正后的传统价量因子 rank 合成，应该比第一阶段等权 z-score `baseline_score` 更适合作为第二阶段多因子诊断基线。",
        f"- Data Scope: snapshot `{result.get('snapshot_id')}`, `{result.get('start_date')}` to `{result.get('end_date')}`.",
        f"- History Scope: rolling weights and factor construction loaded from `{result.get('history_start_date')}`; diagnostics/backtests are evaluated from `{result.get('start_date')}`.",
        f"- Panel: `{result.get('panel', {}).get('rows')}` rows, `{result.get('panel', {}).get('date_count')}` dates, `{result.get('panel', {}).get('security_count')}` securities.",
        f"- Portfolio Panel: `{result.get('portfolio_panel', {}).get('rows')}` rows after selection filters `{result.get('selection_filters')}`; filter report `{result.get('selection_filter_report')}`.",
        f"- Label: `{result.get('label')}`; label mode: `{result.get('label_mode')}`; raw label: `{result.get('raw_label')}`.",
        f"- Label Mode Note: {result.get('label_mode_note')}",
        f"- Backtest Return Note: {result.get('backtest_return_note')}",
        f"- Horizon Backtest Note: {result.get('horizon_backtest_note')}",
        f"- Rolling IC: window `{result.get('rolling_window')}`, min periods `{result.get('rolling_min_periods')}`, fallback date rate `{_fmt(result.get('rolling_fallback_rate'))}`.",
        f"- Low-Correlation Factors: max abs corr `{result.get('max_factor_corr')}`, selected `{result.get('low_corr_factor_columns')}`.",
        f"- Neutralization: proxy columns `{result.get('neutralize_by')}`, neutralized factors `{result.get('neutralized_factor_columns')}`.",
        f"- Signals: `{result.get('multifactor_signals')}`.",
        f"- Factor Directions: `{result.get('factor_directions')}`.",
        f"- Result: best IC signal `{best_signal}`; best quantile-spread signal `{best_spread_signal}`; best backtest signal `{best_backtest_signal}` under the tested Top-N/cost/frequency grid.",
        f"- Consistency Check: IC、分位 spread 和 Top-N 最优信号 `{conflict_note}`；若不一致，本实验只作为诊断证据，不能升级为策略结论。",
        "- Assessment: this is a phase-2 diagnostic baseline; it is not yet a traditional ML model or production strategy candidate.",
        "- Next Step: inspect 2026 monthly/quarterly horizon slices and rolling IC weight drift, then add limit-up/down and suspension execution constraints.",
        "",
        "## Factor Coverage",
        "",
        _markdown_table(coverage),
        "",
        "## Single Factor IC",
        "",
        _markdown_table(single_factor_ic),
        "",
        "## Multifactor IC",
        "",
        _markdown_table(multifactor_ic),
        "",
        "## Quantile Returns",
        "",
        _markdown_table(quantile_summary),
        "",
        "## Quantile Spread",
        "",
        _markdown_table(quantile_spread),
        "",
        "## Backtest Summary",
        "",
        _markdown_table(backtest_summary),
        "",
        "## Rolling IC Weight Audit",
        "",
        _markdown_table(rolling_weight_audit),
        "",
    ]
    if result.get("include_horizon_backtest"):
        lines.extend(
            [
                "## Horizon-Aligned Backtest Summary",
                "",
                _markdown_table(horizon_backtest_summary),
                "",
                "## Horizon-Aligned Yearly Summary",
                "",
                _markdown_table(horizon_backtest_yearly_summary),
                "",
                "## Horizon-Aligned Monthly/Quarterly Summary",
                "",
                _markdown_table(horizon_backtest_period_summary),
                "",
                "## Selected Basket Factor Exposure",
                "",
                _markdown_table(basket_exposure_summary),
                "",
            ]
        )
    lines.extend(
        [
        f"Artifacts: `{result.get('output_dir')}`",
        "",
        ]
    )
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(_fmt)
    return view.to_markdown(index=False)


def _fmt(value: Any) -> str:
    if value is None:
        return "nan"
    try:
        if pd.isna(value):
            return "nan"
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="Optional v2 snapshot root.")
    parser.add_argument("--history-start-date", default=None, help="Optional earlier start date used for rolling-weight warm-up; evaluation still starts at --start-date.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="raw")
    parser.add_argument("--factor-set", choices=FACTOR_SETS, default=DEFAULT_FACTOR_SET, help="Factor family to build; core preserves historical comparability, expanded adds Baostock-only price/volume/turnover factors.")
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW)
    parser.add_argument("--rolling-min-periods", type=int, default=DEFAULT_ROLLING_MIN_PERIODS)
    parser.add_argument("--max-factor-corr", type=float, default=DEFAULT_MAX_FACTOR_CORR)
    parser.add_argument("--neutralize-by", default=",".join(DEFAULT_NEUTRALIZE_BY), help="Comma-separated neutralizer columns; pass empty string to disable.")
    parser.add_argument("--signals", default="", help="Optional comma-separated multifactor signals to evaluate; empty means all generated signals.")
    parser.add_argument("--selection-filters", default="", help="Optional comma-separated portfolio filters such as log_amount_mean_20d_z>=-0.8,momentum_20d_z>=-0.8.")
    parser.add_argument("--top-n", default="50,100,200", help="Comma-separated Top-N portfolio sizes.")
    parser.add_argument("--fee-bps", default="0,10,20", help="Comma-separated fee bps values.")
    parser.add_argument("--frequencies", default="daily,weekly,monthly", help="Comma-separated rebalance frequencies.")
    parser.add_argument("--horizon-backtest", action="store_true", help="Also run non-overlapping horizon-aligned Top-N backtests.")
    parser.add_argument("--horizon-buffer-multipliers", default="1.0", help="Comma-separated buffer multipliers for horizon backtests; values above 1.0 retain prior holdings within an expanded rank buffer.")
    parser.add_argument("--execution-constraints", action="store_true", help="Apply approximate limit-up entry blocks, limit-down exit delays, and non-tradeable entry/exit handling in horizon backtests.")
    parser.add_argument("--limit-threshold", type=float, default=0.095, help="Approximate daily limit threshold used by --execution-constraints.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top_n_values = tuple(int(value.strip()) for value in args.top_n.split(",") if value.strip())
    fee_bps_values = tuple(float(value.strip()) for value in args.fee_bps.split(",") if value.strip())
    frequencies = tuple(value.strip() for value in args.frequencies.split(",") if value.strip())
    neutralize_by = tuple(value.strip() for value in args.neutralize_by.split(",") if value.strip())
    selected_signals = tuple(value.strip() for value in args.signals.split(",") if value.strip()) if args.signals else None
    selection_filters = parse_selection_filters(args.selection_filters)
    horizon_buffer_multipliers = tuple(float(value.strip()) for value in args.horizon_buffer_multipliers.split(",") if value.strip())
    result = run_multifactor_baseline(
        root=args.root,
        history_start_date=args.history_start_date,
        start_date=args.start_date,
        end_date=args.end_date,
        horizon=args.horizon,
        label_mode=args.label_mode,
        factor_set=args.factor_set,
        rolling_window=args.rolling_window,
        rolling_min_periods=args.rolling_min_periods,
        max_factor_corr=args.max_factor_corr,
        neutralize_by=neutralize_by,
        selected_signals=selected_signals,
        selection_filters=selection_filters,
        top_n_values=top_n_values,
        fee_bps_values=fee_bps_values,
        rebalance_frequencies=frequencies,
        include_horizon_backtest=args.horizon_backtest,
        horizon_buffer_multipliers=horizon_buffer_multipliers,
        execution_constraints=args.execution_constraints,
        limit_threshold=args.limit_threshold,
        output_dir=args.output_dir,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
