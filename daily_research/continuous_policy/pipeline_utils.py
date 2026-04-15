from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.continuous_policy.label_builder import (
    FuturePathMetrics,
    build_action_labels_for_date,
    resolve_label_config,
    build_teacher_global_targets,
    build_teacher_policy_frame,
)
from daily_research.continuous_policy.model import predict_policy
from daily_research.continuous_policy.portfolio_simulator import PortfolioState
from daily_research.continuous_policy.state_builder import (
    PreparedPolicyInputs,
    build_cross_section_state,
    build_daily_state_features,
)


SAMPLE_LABEL_COLUMNS = {
    "action_label",
    "target_delta_hint",
    "teacher_priority",
    "teacher_edge",
    "teacher_opportunity",
    "teacher_downside",
    "teacher_signal",
    "teacher_urgency",
    "exit_urgency",
    "entry_quality",
    "hold_quality",
    "add_quality",
    "reduce_quality",
    "reentry_readiness",
    "planned_holding_days",
    "label_preset",
}
NON_FEATURE_COLUMNS = {"date", "stock", *SAMPLE_LABEL_COLUMNS}
DAILY_TARGET_COLUMNS = {
    "date",
    "gross_exposure_target",
    "candidate_budget",
    "turnover_budget",
    "max_position_weight_target",
    "hold_bias_target",
}
DEFAULT_OUTCOME_HORIZONS = (1, 3, 5, 10, 20)


def signal_dates_between(prepared: PreparedPolicyInputs, *, start_date: str, end_date: str = "", max_forward_horizon: int = 10) -> list[pd.Timestamp]:
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize() if str(end_date or "").strip() else prepared.close.index.max()
    dates = [dt for dt in prepared.close.index if dt >= start_ts and dt <= end_ts]
    if max_forward_horizon > 0 and len(dates) > max_forward_horizon:
        return dates[:-max_forward_horizon]
    return dates


def select_feature_columns(sample_frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in sample_frame.columns
        if column not in NON_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(sample_frame[column])
        and sample_frame[column].replace([np.inf, -np.inf], np.nan).notna().any()
    ]


def select_daily_feature_columns(daily_frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in daily_frame.columns
        if column not in DAILY_TARGET_COLUMNS
        and pd.api.types.is_numeric_dtype(daily_frame[column])
        and daily_frame[column].replace([np.inf, -np.inf], np.nan).notna().any()
    ]


def estimate_trading_cost(
    *,
    diagnostics: dict[str, Any],
    transaction_cost_bps: float,
    slippage_bps: float,
    sell_tax_bps: float,
) -> float:
    buy_turnover = float(diagnostics.get("buy_turnover", 0.0) or 0.0)
    sell_turnover = float(diagnostics.get("sell_turnover", 0.0) or 0.0)
    buy_cost = buy_turnover * (float(transaction_cost_bps) + float(slippage_bps)) / 10_000.0
    sell_cost = sell_turnover * (float(transaction_cost_bps) + float(slippage_bps) + float(sell_tax_bps)) / 10_000.0
    return float(buy_cost + sell_cost)


def compute_curve_metrics(daily_returns: pd.Series) -> dict[str, float]:
    clean = pd.Series(daily_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "daily_return_mean": 0.0,
            "daily_return_std": 0.0,
        }
    equity = (1.0 + clean).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    annual_return = float(equity.iloc[-1] ** (252.0 / max(len(clean), 1)) - 1.0)
    annual_volatility = float(clean.std(ddof=0) * np.sqrt(252.0))
    sharpe = float(clean.mean() / clean.std(ddof=0) * np.sqrt(252.0)) if float(clean.std(ddof=0)) > 0 else 0.0
    max_drawdown = float((equity / equity.cummax() - 1.0).min())
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "daily_return_mean": float(clean.mean()),
        "daily_return_std": float(clean.std(ddof=0)),
    }


def _build_forward_excess_frame(prepared: PreparedPolicyInputs, horizon: int) -> pd.DataFrame:
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError(f"Forward horizon must be positive, got {horizon}.")
    stock_return = prepared.close.shift(-horizon).div(prepared.close).sub(1.0)
    benchmark_return = prepared.benchmark_close.shift(-horizon).div(prepared.benchmark_close).sub(1.0)
    return stock_return.sub(benchmark_return, axis=0)


def _build_forward_benchmark_return(prepared: PreparedPolicyInputs, horizon: int) -> pd.Series:
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError(f"Forward horizon must be positive, got {horizon}.")
    return prepared.benchmark_close.shift(-horizon).div(prepared.benchmark_close).sub(1.0)


def _future_window_extreme(close: pd.DataFrame, window: int, *, mode: str) -> pd.DataFrame:
    shifted = close.shift(-1)
    reversed_frame = shifted.iloc[::-1]
    if mode == "max":
        rolled = reversed_frame.rolling(window, min_periods=1).max()
    elif mode == "min":
        rolled = reversed_frame.rolling(window, min_periods=1).min()
    else:
        raise ValueError(f"Unsupported future extreme mode: {mode}")
    return rolled.iloc[::-1]


def _build_future_max_up_frame(prepared: PreparedPolicyInputs, horizon: int) -> pd.DataFrame:
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError(f"Future max-up horizon must be positive, got {horizon}.")
    future_max = _future_window_extreme(prepared.close, horizon, mode="max")
    return future_max.div(prepared.close).sub(1.0)


def _build_future_min_down_frame(prepared: PreparedPolicyInputs, horizon: int) -> pd.DataFrame:
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError(f"Future min-down horizon must be positive, got {horizon}.")
    future_min = _future_window_extreme(prepared.close, horizon, mode="min")
    return future_min.div(prepared.close).sub(1.0)


def build_action_outcome_frame(
    *,
    prepared: PreparedPolicyInputs,
    action_panel: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_OUTCOME_HORIZONS,
) -> pd.DataFrame:
    if action_panel.empty:
        return action_panel.copy()

    working = action_panel.copy()
    working["date"] = pd.to_datetime(working["date"]).dt.normalize()
    working["stock"] = working["stock"].astype(str).str.upper()

    date_lookup = set(prepared.close.index)
    column_lookup = set(prepared.close.columns.astype(str))
    for horizon in horizons:
        excess_frame = _build_forward_excess_frame(prepared, int(horizon))
        benchmark_return = _build_forward_benchmark_return(prepared, int(horizon))
        max_up_frame = _build_future_max_up_frame(prepared, int(horizon))
        min_down_frame = _build_future_min_down_frame(prepared, int(horizon))
        working[f"forward_excess_{int(horizon)}d"] = [
            float(excess_frame.at[date_value, stock_value])
            if date_value in date_lookup and stock_value in column_lookup
            else np.nan
            for date_value, stock_value in zip(working["date"], working["stock"])
        ]
        working[f"forward_benchmark_return_{int(horizon)}d"] = [
            float(benchmark_return.loc[date_value]) if date_value in date_lookup else np.nan
            for date_value in working["date"]
        ]
        working[f"future_max_up_{int(horizon)}d"] = [
            float(max_up_frame.at[date_value, stock_value])
            if date_value in date_lookup and stock_value in column_lookup
            else np.nan
            for date_value, stock_value in zip(working["date"], working["stock"])
        ]
        working[f"future_min_down_{int(horizon)}d"] = [
            float(min_down_frame.at[date_value, stock_value])
            if date_value in date_lookup and stock_value in column_lookup
            else np.nan
            for date_value, stock_value in zip(working["date"], working["stock"])
        ]
    working["date"] = working["date"].dt.strftime("%Y-%m-%d")
    return working


def compute_continuity_metrics(
    *,
    prepared: PreparedPolicyInputs,
    action_panel: pd.DataFrame,
    turnover_frame: pd.DataFrame,
    returns_series: pd.Series,
) -> tuple[dict[str, float], pd.DataFrame]:
    action_outcomes = build_action_outcome_frame(prepared=prepared, action_panel=action_panel)
    metrics: dict[str, float] = {
        "action_rows": float(len(action_outcomes)),
        "days_evaluated": float(len(returns_series)),
    }

    if not action_outcomes.empty:
        action_lookup = action_outcomes["execution_action"].astype(str)
        date_positions = {
            pd.Timestamp(date_value).normalize().strftime("%Y-%m-%d"): position
            for position, date_value in enumerate(prepared.close.index)
        }
        for action_name in ("open", "add", "reduce", "exit", "hold"):
            metrics[f"{action_name}_count"] = float((action_lookup == action_name).sum())

        held_rows = action_outcomes.loc[action_outcomes["hold_days_before"].fillna(0.0) > 0]
        exit_rows = action_outcomes.loc[action_lookup == "exit"]
        add_rows = action_outcomes.loc[action_lookup == "add"]
        hold_rows = action_outcomes.loc[action_lookup == "hold"]
        open_rows = action_outcomes.loc[action_lookup == "open"]
        reduce_rows = action_outcomes.loc[action_lookup == "reduce"]
        add_forward_5d = add_rows["forward_excess_5d"].dropna()
        hold_forward_5d = hold_rows["forward_excess_5d"].dropna()
        open_forward_5d = open_rows["forward_excess_5d"].dropna()
        reduce_forward_5d = reduce_rows["forward_excess_5d"].dropna()
        exit_forward_5d = exit_rows["forward_excess_5d"].dropna()
        open_add_rows = action_outcomes.loc[action_lookup.isin({"open", "add"})]
        position_rows = action_outcomes.loc[action_lookup.isin({"open", "add", "hold"})]
        sell_rows = action_outcomes.loc[action_lookup.isin({"reduce", "exit"})]
        main_leg_threshold_10d = 0.06
        entry_max_up_10d = open_add_rows["future_max_up_10d"].dropna()
        hold_max_up_10d = hold_rows["future_max_up_10d"].dropna()
        sell_leg_frame = sell_rows.loc[:, ["future_max_up_10d", "future_min_down_10d"]].dropna()
        position_max_up_10d = position_rows["future_max_up_10d"].dropna()

        metrics["avg_hold_days_before_action"] = float(held_rows["hold_days_before"].mean()) if not held_rows.empty else 0.0
        metrics["median_hold_days_before_action"] = float(held_rows["hold_days_before"].median()) if not held_rows.empty else 0.0
        metrics["holding_duration_p25"] = float(held_rows["hold_days_before"].quantile(0.25)) if not held_rows.empty else 0.0
        metrics["holding_duration_p75"] = float(held_rows["hold_days_before"].quantile(0.75)) if not held_rows.empty else 0.0
        metrics["exit_hold_days_mean"] = float(exit_rows["hold_days_before"].mean()) if not exit_rows.empty else 0.0
        metrics["exit_hold_days_median"] = float(exit_rows["hold_days_before"].median()) if not exit_rows.empty else 0.0
        metrics["add_win_rate_5d"] = float((add_forward_5d > 0).mean()) if not add_forward_5d.empty else 0.0
        metrics["open_win_rate_5d"] = float((open_forward_5d > 0).mean()) if not open_forward_5d.empty else 0.0
        metrics["hold_positive_rate_5d"] = float((hold_forward_5d > 0).mean()) if not hold_forward_5d.empty else 0.0
        metrics["hold_retention_quality_5d"] = float(hold_forward_5d.mean()) if not hold_forward_5d.empty else 0.0
        metrics["reduce_success_rate_5d"] = float((reduce_forward_5d <= 0).mean()) if not reduce_forward_5d.empty else 0.0
        metrics["wrong_side_reduce_share"] = float((reduce_forward_5d > 0).mean()) if not reduce_forward_5d.empty else 0.0
        metrics["profit_take_too_early_share"] = (
            float(
                (
                    (reduce_rows["unrealized_pnl_before"].fillna(0.0) > 0.05)
                    & (reduce_rows["forward_excess_5d"].fillna(0.0) > 0.0)
                ).mean()
            )
            if not reduce_rows.empty
            else 0.0
        )
        metrics["reduce_preservation_quality_5d"] = (
            float((-reduce_forward_5d).clip(lower=0.0).mean()) if not reduce_forward_5d.empty else 0.0
        )
        metrics["exit_timeliness_rate_5d"] = float((exit_forward_5d <= 0).mean()) if not exit_forward_5d.empty else 0.0
        metrics["exit_avoided_loss_5d"] = (
            float((-exit_forward_5d).clip(lower=0.0).mean()) if not exit_forward_5d.empty else 0.0
        )
        metrics["exit_missed_upside_5d"] = (
            float(exit_forward_5d.clip(lower=0.0).mean()) if not exit_forward_5d.empty else 0.0
        )
        metrics["exit_then_rebound_cost"] = metrics["exit_missed_upside_5d"]
        metrics["avg_unrealized_pnl_before_action"] = (
            float(held_rows["unrealized_pnl_before"].mean()) if not held_rows.empty else 0.0
        )
        metrics["avg_drawdown_before_action"] = (
            float(held_rows["drawdown_from_peak_before"].mean()) if not held_rows.empty else 0.0
        )
        metrics["hold_share"] = float(metrics["hold_count"] / max(metrics["action_rows"], 1.0))
        metrics["churn_ratio"] = float(action_lookup.isin({"open", "add", "reduce", "exit"}).mean())
        metrics["early_reduce_share"] = (
            float((reduce_rows["hold_days_before"].fillna(999.0) <= 3.0).mean()) if not reduce_rows.empty else 0.0
        )
        metrics["early_exit_share"] = (
            float((exit_rows["hold_days_before"].fillna(999.0) <= 3.0).mean()) if not exit_rows.empty else 0.0
        )
        metrics["profitable_reduce_share"] = (
            float((reduce_rows["unrealized_pnl_before"].fillna(0.0) > 0.05).mean()) if not reduce_rows.empty else 0.0
        )
        metrics["main_leg_threshold_10d"] = float(main_leg_threshold_10d)
        metrics["trend_capture_rate_10d"] = (
            float((position_max_up_10d >= main_leg_threshold_10d).mean()) if not position_max_up_10d.empty else 0.0
        )
        metrics["entry_trend_capture_rate_10d"] = (
            float((entry_max_up_10d >= main_leg_threshold_10d).mean()) if not entry_max_up_10d.empty else 0.0
        )
        metrics["entry_trend_capture_quality_10d"] = float(entry_max_up_10d.mean()) if not entry_max_up_10d.empty else 0.0
        metrics["hold_trend_capture_quality_10d"] = float(hold_max_up_10d.mean()) if not hold_max_up_10d.empty else 0.0
        metrics["missed_main_leg_rate_10d"] = (
            float((sell_leg_frame["future_max_up_10d"] >= main_leg_threshold_10d).mean()) if not sell_leg_frame.empty else 0.0
        )
        metrics["premature_sell_share_10d"] = (
            float(
                (
                    (sell_leg_frame["future_max_up_10d"] >= main_leg_threshold_10d)
                    & (sell_leg_frame["future_min_down_10d"] > -0.04)
                ).mean()
            )
            if not sell_leg_frame.empty
            else 0.0
        )

        if not open_rows.empty:
            open_positions = open_rows["date"].map(date_positions).dropna().astype(int)
            evaluation_start_pos = int(action_outcomes["date"].map(date_positions).dropna().astype(int).min())
            first_open_pos = int(open_positions.min())
            metrics["days_to_first_open"] = float(max(first_open_pos - evaluation_start_pos, 0))
            metrics["cold_start_open_rate"] = float((open_positions <= evaluation_start_pos + 2).mean())
        else:
            metrics["days_to_first_open"] = float(len(date_positions))
            metrics["cold_start_open_rate"] = 0.0

        action_events = action_outcomes.loc[:, ["date", "stock", "execution_action", "forward_excess_5d"]].copy()
        action_events["signal_pos"] = action_events["date"].map(date_positions)
        action_events = action_events.dropna(subset=["signal_pos"]).sort_values(["stock", "signal_pos"]).reset_index(drop=True)
        reversal_hits = 0
        reversal_base = 0
        reentry_hits = 0
        reduce_reversal_hits = 0
        reduce_reversal_base = 0
        exit_reversal_hits = 0
        exit_reversal_base = 0
        reentry_quality_rows: list[pd.Series] = []
        for _, stock_rows in action_events.groupby("stock", sort=False):
            stock_rows = stock_rows.reset_index(drop=True)
            for row_idx in range(len(stock_rows)):
                current_event = stock_rows.iloc[row_idx]
                current_action = str(current_event["execution_action"])
                current_group = 1 if current_action in {"open", "add"} else -1 if current_action in {"reduce", "exit"} else 0
                if current_group != 0:
                    reversal_base += 1
                if current_action == "exit":
                    later_open = stock_rows.loc[
                        (stock_rows.index > row_idx)
                        & (stock_rows["execution_action"].astype(str).isin({"open", "add"}))
                        & (stock_rows["signal_pos"] <= float(current_event["signal_pos"]) + 10)
                    ]
                    if not later_open.empty:
                        reentry_hits += 1
                        reentry_quality_rows.append(later_open.iloc[0])
                if current_group == 0:
                    continue
                later_rows = stock_rows.loc[
                    (stock_rows.index > row_idx)
                    & (stock_rows["signal_pos"] <= float(current_event["signal_pos"]) + 3)
                ]
                if later_rows.empty:
                    continue
                next_event = later_rows.iloc[0]
                next_action = str(next_event["execution_action"])
                next_group = 1 if next_action in {"open", "add"} else -1 if next_action in {"reduce", "exit"} else 0
                if next_group != 0 and next_group != current_group:
                    reversal_hits += 1
                if current_action == "reduce":
                    reduce_reversal_base += 1
                    if next_action in {"open", "add"}:
                        reduce_reversal_hits += 1
                if current_action == "exit":
                    exit_reversal_base += 1
                    if next_action in {"open", "add"}:
                        exit_reversal_hits += 1

        if not exit_rows.empty:
            exit_events = exit_rows.loc[:, ["date", "stock"]].copy()
            exit_events["signal_pos"] = exit_events["date"].map(date_positions)
            open_events = open_rows.loc[:, ["date", "stock"]].copy()
            open_events["signal_pos"] = open_events["date"].map(date_positions)
            for exit_event in exit_events.itertuples(index=False):
                later_open = open_events.loc[
                    (open_events["stock"] == exit_event.stock)
                    & (open_events["signal_pos"] > exit_event.signal_pos)
                    & (open_events["signal_pos"] <= exit_event.signal_pos + 10)
                ]
                if not later_open.empty:
                    continue
            metrics["reentry_within_10d_rate"] = float(reentry_hits / len(exit_events)) if len(exit_events) else 0.0
            metrics["reentry_after_exit_3d_rate"] = (
                float(
                    sum(
                        1
                        for exit_event in exit_events.itertuples(index=False)
                        if not open_events.loc[
                            (open_events["stock"] == exit_event.stock)
                            & (open_events["signal_pos"] > exit_event.signal_pos)
                            & (open_events["signal_pos"] <= exit_event.signal_pos + 3)
                        ].empty
                    )
                    / len(exit_events)
                )
                if len(exit_events)
                else 0.0
            )
        else:
            metrics["reentry_within_10d_rate"] = 0.0
            metrics["reentry_after_exit_3d_rate"] = 0.0
        metrics["reentry_quality_10d"] = (
            float(pd.DataFrame(reentry_quality_rows)["forward_excess_5d"].gt(0).mean()) if reentry_quality_rows else 0.0
        )
        metrics["immediate_reversal_rate_3d"] = float(reversal_hits / reversal_base) if reversal_base else 0.0
        metrics["reversal_after_reduce_3d_rate"] = (
            float(reduce_reversal_hits / reduce_reversal_base) if reduce_reversal_base else 0.0
        )
        metrics["reversal_after_exit_3d_rate"] = (
            float(exit_reversal_hits / exit_reversal_base) if exit_reversal_base else 0.0
        )
    else:
        for key in (
            "open_count",
            "add_count",
            "reduce_count",
            "exit_count",
            "hold_count",
            "avg_hold_days_before_action",
            "median_hold_days_before_action",
            "holding_duration_p25",
            "holding_duration_p75",
            "exit_hold_days_mean",
            "exit_hold_days_median",
            "add_win_rate_5d",
            "open_win_rate_5d",
            "hold_positive_rate_5d",
            "hold_retention_quality_5d",
            "reduce_success_rate_5d",
            "wrong_side_reduce_share",
            "profit_take_too_early_share",
            "reduce_preservation_quality_5d",
            "exit_timeliness_rate_5d",
            "exit_avoided_loss_5d",
            "exit_missed_upside_5d",
            "exit_then_rebound_cost",
            "avg_unrealized_pnl_before_action",
            "avg_drawdown_before_action",
            "reentry_within_10d_rate",
            "reentry_after_exit_3d_rate",
            "reentry_quality_10d",
            "hold_share",
            "churn_ratio",
            "early_reduce_share",
            "early_exit_share",
            "profitable_reduce_share",
            "main_leg_threshold_10d",
            "trend_capture_rate_10d",
            "entry_trend_capture_rate_10d",
            "entry_trend_capture_quality_10d",
            "hold_trend_capture_quality_10d",
            "missed_main_leg_rate_10d",
            "premature_sell_share_10d",
            "days_to_first_open",
            "cold_start_open_rate",
            "immediate_reversal_rate_3d",
            "reversal_after_reduce_3d_rate",
            "reversal_after_exit_3d_rate",
        ):
            metrics[key] = 0.0

    if not turnover_frame.empty:
        turnover_working = turnover_frame.copy()
        turnover_working["date"] = pd.to_datetime(turnover_working["date"]).dt.normalize()
        benchmark_1d = _build_forward_benchmark_return(prepared, 1).rename("benchmark_forward_return_1d")
        aligned = turnover_working.merge(
            benchmark_1d.rename_axis("date").reset_index(),
            how="left",
            on="date",
        )
        cash_series = aligned["cash_weight"].astype(float)
        benchmark_series = aligned["benchmark_forward_return_1d"].astype(float)
        valid = cash_series.notna() & benchmark_series.notna()
        if int(valid.sum()) >= 2:
            corr = float(np.corrcoef(cash_series.loc[valid], benchmark_series.loc[valid])[0, 1])
            metrics["cash_timing_quality_1d"] = float(-corr) if np.isfinite(corr) else 0.0
        else:
            metrics["cash_timing_quality_1d"] = 0.0
        high_cash_threshold = float(cash_series.quantile(0.75)) if len(cash_series) else 0.0
        high_cash_mask = valid & (cash_series >= high_cash_threshold)
        metrics["high_cash_share"] = float(high_cash_mask.mean()) if len(cash_series) else 0.0
        metrics["high_cash_down_market_hit_rate"] = (
            float((benchmark_series.loc[high_cash_mask] < 0).mean()) if bool(high_cash_mask.any()) else 0.0
        )
        metrics["risk_off_cash_hit_rate"] = metrics["high_cash_down_market_hit_rate"]
    else:
        metrics["cash_timing_quality_1d"] = 0.0
        metrics["high_cash_share"] = 0.0
        metrics["high_cash_down_market_hit_rate"] = 0.0
        metrics["risk_off_cash_hit_rate"] = 0.0
    if not turnover_frame.empty:
        metrics["avg_position_cap_target"] = float(turnover_frame["max_position_weight_target"].mean())
        metrics["avg_hold_bias_target"] = float(turnover_frame["hold_bias_target"].mean())
        metrics["avg_reduce_bias_target"] = float(turnover_frame["reduce_bias_target"].mean()) if "reduce_bias_target" in turnover_frame.columns else 0.0
        metrics["avg_exit_patience_target"] = float(turnover_frame["exit_patience_target"].mean()) if "exit_patience_target" in turnover_frame.columns else 0.0
        metrics["avg_reentry_guard_target"] = float(turnover_frame["reentry_guard_target"].mean()) if "reentry_guard_target" in turnover_frame.columns else 0.0
    else:
        metrics["avg_position_cap_target"] = 0.0
        metrics["avg_hold_bias_target"] = 0.0
        metrics["avg_reduce_bias_target"] = 0.0
        metrics["avg_exit_patience_target"] = 0.0
        metrics["avg_reentry_guard_target"] = 0.0

    return metrics, action_outcomes


def build_training_matrices(
    *,
    prepared: PreparedPolicyInputs,
    future_metrics: FuturePathMetrics,
    start_date: str,
    end_date: str = "",
    transaction_cost_bps: float = 3.0,
    slippage_bps: float = 7.0,
    sell_tax_bps: float = 10.0,
    random_seed: int = 7,
    skip_multiplier: float = 2.0,
    label_preset: str = "balanced_v2",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    resolved_label_config = resolve_label_config(label_preset)
    dates = signal_dates_between(
        prepared,
        start_date=start_date,
        end_date=end_date,
        max_forward_horizon=max(int(item) for item in future_metrics.horizons),
    )
    if len(dates) < 25:
        raise ValueError("Continuous-policy training window is too short after forward-horizon trimming.")

    rng = np.random.default_rng(int(random_seed))
    portfolio = PortfolioState()
    sample_frames: list[pd.DataFrame] = []
    daily_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    teacher_returns: list[float] = []

    for idx, signal_dt in enumerate(dates):
        state_frame = build_cross_section_state(prepared, date=signal_dt, portfolio_state=portfolio)
        label_frame = build_action_labels_for_date(
            date=signal_dt,
            state_frame=state_frame,
            future_metrics=future_metrics,
            label_config=resolved_label_config,
        )
        daily_features = build_daily_state_features(state_frame)
        global_targets = build_teacher_global_targets(label_frame, label_config=resolved_label_config)
        teacher_policy = build_teacher_policy_frame(label_frame)
        step_result = portfolio.step(
            date=signal_dt,
            prices=prepared.close.loc[signal_dt],
            policy_frame=teacher_policy,
            global_targets=global_targets,
            source_label="teacher",
        )

        sample_frames.append(label_frame)
        daily_rows.append({"date": signal_dt.strftime("%Y-%m-%d"), **daily_features, **global_targets})
        action_rows.extend(step_result.actions)

        if idx + 1 < len(dates):
            next_dt = dates[idx + 1]
            next_returns = prepared.close.loc[next_dt].div(prepared.close.loc[signal_dt]).sub(1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            gross_return = float(step_result.weights.reindex(next_returns.index).fillna(0.0).mul(next_returns).sum())
            trading_cost = estimate_trading_cost(
                diagnostics=step_result.diagnostics,
                transaction_cost_bps=transaction_cost_bps,
                slippage_bps=slippage_bps,
                sell_tax_bps=sell_tax_bps,
            )
            net_return = gross_return - trading_cost
            teacher_returns.append(net_return)
            portfolio.record_realized_return(net_return)

    full_sample_frame = pd.concat(sample_frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    daily_frame = pd.DataFrame(daily_rows).replace([np.inf, -np.inf], np.nan)
    interesting = full_sample_frame.loc[full_sample_frame["action_label"].astype(str) != "skip"]
    skip_frame = full_sample_frame.loc[full_sample_frame["action_label"].astype(str) == "skip"]
    skip_cap = min(len(skip_frame), max(int(len(interesting) * float(skip_multiplier)), 4_000))
    if len(skip_frame) > skip_cap > 0:
        skip_frame = skip_frame.sample(n=skip_cap, random_state=int(rng.integers(0, 2**31 - 1)))
    train_frame = (
        pd.concat([interesting, skip_frame], ignore_index=True)
        .sort_values(["date", "stock"], ascending=[True, True])
        .reset_index(drop=True)
    )
    action_panel = pd.DataFrame(action_rows)
    teacher_metrics = compute_curve_metrics(pd.Series(teacher_returns, index=[dt.strftime("%Y-%m-%d") for dt in dates[1:]], dtype=float))
    summary = {
        "label_preset": resolved_label_config.name,
        "teacher_action_distribution": {
            str(key): int(value)
            for key, value in full_sample_frame["action_label"].astype(str).value_counts().sort_index().items()
        },
        "teacher_rollout_metrics": teacher_metrics,
        "full_sample_rows": int(len(full_sample_frame)),
        "train_sample_rows": int(len(train_frame)),
        "daily_rows": int(len(daily_frame)),
        "action_rows": int(len(action_panel)),
        "teacher_recent_turnover_mean": float(action_panel["delta_weight"].abs().mean()) if not action_panel.empty else 0.0,
    }
    return train_frame, daily_frame, summary


def run_policy_rollout(
    *,
    prepared: PreparedPolicyInputs,
    artifact: Any | None,
    start_date: str,
    end_date: str = "",
    initial_portfolio: PortfolioState | None = None,
    future_metrics: FuturePathMetrics | None = None,
    transaction_cost_bps: float = 3.0,
    slippage_bps: float = 7.0,
    sell_tax_bps: float = 10.0,
    source_label: str = "model",
    label_preset: str = "balanced_v2",
) -> dict[str, Any]:
    dates = signal_dates_between(prepared, start_date=start_date, end_date=end_date, max_forward_horizon=0)
    if len(dates) < 2:
        raise ValueError("Continuous-policy rollout requires at least two signal dates.")

    portfolio = initial_portfolio.clone() if initial_portfolio is not None else PortfolioState()
    action_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    daily_returns: list[float] = []
    turnover_rows: list[dict[str, Any]] = []

    for idx, signal_dt in enumerate(dates):
        state_frame = build_cross_section_state(prepared, date=signal_dt, portfolio_state=portfolio)
        daily_features = build_daily_state_features(state_frame)
        if artifact is None:
            if future_metrics is None:
                raise ValueError("future_metrics is required when artifact is None.")
            label_frame = build_action_labels_for_date(
                date=signal_dt,
                state_frame=state_frame,
                future_metrics=future_metrics,
                label_config=label_preset,
            )
            policy_frame = build_teacher_policy_frame(label_frame)
            global_targets = build_teacher_global_targets(label_frame, label_config=label_preset)
        else:
            policy_frame, global_targets = predict_policy(artifact, state_frame=state_frame, daily_features=daily_features)

        step_result = portfolio.step(
            date=signal_dt,
            prices=prepared.close.loc[signal_dt],
            policy_frame=policy_frame,
            global_targets=global_targets,
            source_label=source_label,
        )
        action_rows.extend(step_result.actions)
        turnover_rows.append({"date": step_result.date, **step_result.diagnostics})
        position_rows.append({"date": step_result.date, **portfolio.snapshot()})

        if idx + 1 < len(dates):
            next_dt = dates[idx + 1]
            next_returns = prepared.close.loc[next_dt].div(prepared.close.loc[signal_dt]).sub(1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            gross_return = float(step_result.weights.reindex(next_returns.index).fillna(0.0).mul(next_returns).sum())
            trading_cost = estimate_trading_cost(
                diagnostics=step_result.diagnostics,
                transaction_cost_bps=transaction_cost_bps,
                slippage_bps=slippage_bps,
                sell_tax_bps=sell_tax_bps,
            )
            net_return = gross_return - trading_cost
            daily_returns.append(net_return)
            portfolio.record_realized_return(net_return)

    action_panel = pd.DataFrame(action_rows)
    turnover_frame = pd.DataFrame(turnover_rows)
    returns_index = [dt.strftime("%Y-%m-%d") for dt in dates[1:]]
    returns_series = pd.Series(daily_returns, index=returns_index, dtype=float)
    metrics = compute_curve_metrics(returns_series)
    if not turnover_frame.empty:
        metrics["avg_turnover"] = float(turnover_frame["realized_turnover"].mean())
        metrics["avg_buy_turnover"] = float(turnover_frame["buy_turnover"].mean())
        metrics["avg_sell_turnover"] = float(turnover_frame["sell_turnover"].mean())
        metrics["avg_gross_exposure"] = float(1.0 - turnover_frame["cash_weight"].mean())
        metrics["avg_holding_count"] = float(turnover_frame["holding_count"].mean())
    else:
        metrics["avg_turnover"] = 0.0
        metrics["avg_buy_turnover"] = 0.0
        metrics["avg_sell_turnover"] = 0.0
        metrics["avg_gross_exposure"] = 0.0
        metrics["avg_holding_count"] = 0.0
    metrics["action_counts"] = {
        str(key): int(value)
        for key, value in action_panel["execution_action"].astype(str).value_counts().sort_index().items()
    } if not action_panel.empty else {}
    continuity_metrics, action_outcomes = compute_continuity_metrics(
        prepared=prepared,
        action_panel=action_panel,
        turnover_frame=turnover_frame,
        returns_series=returns_series,
    )
    return {
        "dates": [dt.strftime("%Y-%m-%d") for dt in dates],
        "returns": returns_series,
        "metrics": metrics,
        "continuity_metrics": continuity_metrics,
        "action_panel": action_panel,
        "action_outcomes": action_outcomes,
        "turnover_frame": turnover_frame,
        "position_history": pd.DataFrame(position_rows),
        "final_portfolio": portfolio,
    }


def load_target_weight_panel(panel_path: str | Path) -> pd.DataFrame:
    path = Path(panel_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Target-weight panel file not found: {path}")
    panel = pd.read_csv(path)
    required_columns = {"date", "stock", "target_weight"}
    if not required_columns.issubset(panel.columns):
        raise ValueError(f"Target-weight panel is missing columns: {sorted(required_columns - set(panel.columns))}")
    panel["date"] = pd.to_datetime(panel["date"])
    panel["stock"] = panel["stock"].astype(str).str.upper()
    pivot = panel.pivot_table(index="date", columns="stock", values="target_weight", aggfunc="last").sort_index().fillna(0.0)
    return pivot


def evaluate_reference_panel(
    *,
    prepared: PreparedPolicyInputs,
    panel_frame: pd.DataFrame,
    start_date: str,
    end_date: str = "",
    transaction_cost_bps: float = 3.0,
    slippage_bps: float = 7.0,
    sell_tax_bps: float = 10.0,
) -> dict[str, Any]:
    dates = signal_dates_between(prepared, start_date=start_date, end_date=end_date, max_forward_horizon=0)
    if len(dates) < 2:
        return {"metrics": compute_curve_metrics(pd.Series(dtype=float)), "returns": pd.Series(dtype=float)}
    weights = panel_frame.reindex(index=dates, columns=prepared.close.columns, fill_value=0.0).fillna(0.0)
    previous_weights = pd.Series(0.0, index=prepared.close.columns, dtype=float)
    returns: list[float] = []
    turnovers: list[float] = []
    gross_exposures: list[float] = []
    holding_counts: list[float] = []
    for idx, signal_dt in enumerate(dates[:-1]):
        current_weights = weights.loc[signal_dt].astype(float).clip(lower=0.0)
        total_weight = float(current_weights.sum())
        if total_weight > 0.999:
            current_weights = current_weights / total_weight
        next_dt = dates[idx + 1]
        next_returns = prepared.close.loc[next_dt].div(prepared.close.loc[signal_dt]).sub(1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        delta = current_weights - previous_weights
        diagnostics = {
            "buy_turnover": float(delta.clip(lower=0.0).sum()),
            "sell_turnover": float((-delta.clip(upper=0.0)).sum()),
        }
        diagnostics["realized_turnover"] = diagnostics["buy_turnover"] + diagnostics["sell_turnover"]
        gross_return = float(current_weights.mul(next_returns).sum())
        trading_cost = estimate_trading_cost(
            diagnostics=diagnostics,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            sell_tax_bps=sell_tax_bps,
        )
        returns.append(gross_return - trading_cost)
        turnovers.append(float(diagnostics["realized_turnover"]))
        gross_exposures.append(float(current_weights.sum()))
        holding_counts.append(float((current_weights > 1e-8).sum()))
        previous_weights = current_weights
    returns_series = pd.Series(returns, index=[dt.strftime("%Y-%m-%d") for dt in dates[1:]], dtype=float)
    metrics = compute_curve_metrics(returns_series)
    metrics["avg_turnover"] = float(np.mean(turnovers)) if turnovers else 0.0
    metrics["avg_gross_exposure"] = float(np.mean(gross_exposures)) if gross_exposures else 0.0
    metrics["avg_holding_count"] = float(np.mean(holding_counts)) if holding_counts else 0.0
    metrics["avg_cash_weight"] = float(np.mean([1.0 - item for item in gross_exposures])) if gross_exposures else 0.0
    return {"metrics": metrics, "returns": returns_series}


def translate_target_weights_to_share_actions(
    *,
    account_snapshot: dict[str, Any],
    target_weights: pd.Series,
    price_row: pd.Series,
) -> pd.DataFrame:
    positions = account_snapshot.get("positions", []) if isinstance(account_snapshot, dict) else []
    cash_value = float(account_snapshot.get("available_cash", 0.0) or 0.0)
    current_shares = {
        str((item or {}).get("stock", "") or "").strip().upper(): int(float((item or {}).get("shares", 0.0) or 0.0))
        for item in positions
        if str((item or {}).get("stock", "") or "").strip()
    }
    tracked_stocks = sorted(set(target_weights.index.astype(str)) | set(current_shares))
    market_values = {stock: current_shares.get(stock, 0) * float(price_row.get(stock, np.nan) or 0.0) for stock in tracked_stocks}
    total_equity = cash_value + float(sum(max(value, 0.0) for value in market_values.values()))

    rows: list[dict[str, Any]] = []
    for stock in tracked_stocks:
        price = float(price_row.get(stock, np.nan))
        current_qty = int(current_shares.get(stock, 0))
        current_weight = float(market_values.get(stock, 0.0) / total_equity) if total_equity > 0 and np.isfinite(price) and price > 0 else 0.0
        target_weight = float(target_weights.get(stock, 0.0) or 0.0)
        if not np.isfinite(price) or price <= 0:
            target_qty = current_qty
        else:
            raw_target_qty = int((target_weight * total_equity) / price) if total_equity > 0 else 0
            target_qty = max(0, raw_target_qty // 100 * 100)
        delta_qty = target_qty - current_qty
        if current_qty <= 0 and target_qty <= 0:
            continue
        if current_qty <= 0 and target_qty > 0:
            action = "买入"
        elif current_qty > 0 and target_qty <= 0:
            action = "卖出"
        elif delta_qty > 0:
            action = "加仓"
        elif delta_qty < 0:
            action = "减仓"
        else:
            action = "持有"
        rows.append(
            {
                "stock": stock,
                "price": price,
                "current_weight": current_weight,
                "target_weight": target_weight,
                "current_shares": current_qty,
                "target_shares": target_qty,
                "delta_shares": delta_qty,
                "share_action": action,
            }
        )
    return pd.DataFrame(rows)


def build_reason_summary(action_row: pd.Series) -> str:
    execution_action = str(action_row.get("execution_action", "") or "")
    model_action = str(action_row.get("model_action", "") or "")
    exit_urgency = float(action_row.get("exit_urgency", 0.0) or 0.0)
    delta_weight = float(action_row.get("delta_weight", 0.0) or 0.0)
    if execution_action == "exit":
        return f"模型判定 `{model_action}`，退出紧迫度 {exit_urgency:.3f}，目标直接降至 0。"
    if execution_action == "reduce":
        return f"模型判定 `{model_action}`，降低风险暴露，目标权重变化 {delta_weight:.4f}。"
    if execution_action == "add":
        return f"模型判定 `{model_action}`，延续持仓并提高暴露，目标权重变化 {delta_weight:.4f}。"
    if execution_action == "open":
        return f"模型判定 `{model_action}`，满足开仓条件，建立新持仓。"
    return f"模型判定 `{model_action}`，维持当前持仓。"
