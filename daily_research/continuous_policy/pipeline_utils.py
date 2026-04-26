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
from daily_research.continuous_policy.portfolio_simulator import (
    DEFAULT_BUDGET_CALIBRATION,
    DEFAULT_BUDGET_SEMANTICS,
    DEFAULT_EXECUTION_SEMANTICS,
    PortfolioState,
)
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
    "reduce_fraction_target",
    "exit_hazard_target",
    "sell_attribution_score",
    "sell_rank_score",
    "lifecycle_sell_gate",
    "large_upside_1d_target",
    "alpha_opportunity_value",
    "hold_continuation_value",
    "sell_release_value",
    "cash_defense_value",
    "deployment_opportunity_cost",
    "risk_adjusted_action_value",
    "multi_horizon_forward_value",
    "multi_horizon_forward_risk",
    "multi_horizon_path_value",
    "open_action_value",
    "add_action_value",
    "hold_action_value",
    "reduce_action_value",
    "exit_action_value",
    "relative_opportunity_value",
    "action_value_consistency_target",
    "value_arbitration_target",
    "deploy_value_target",
    "release_value_target",
    "defense_value_target",
    "deploy_gate_target",
    "release_gate_target",
    "defense_gate_target",
    "deploy_executability_target",
    "clipped_intent_risk",
    "holding_flag_target",
    "forward_benchmark_return_1d",
    "forward_benchmark_return_3d",
    "forward_benchmark_return_5d",
    "forward_benchmark_return_10d",
    "forward_benchmark_return_20d",
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
    "reduce_bias_target",
    "exit_patience_target",
    "reentry_guard_target",
    "budget_risk_signal_target",
    "budget_deploy_signal_target",
    "budget_cash_timing_signal_target",
    "budget_alpha_focus_signal_target",
}
DEFAULT_BUDGET_OBJECTIVE = "teacher_imitation"
BUDGET_OBJECTIVE_RESULT_VALUE_V1 = "result_value_v1"
BUDGET_OBJECTIVE_RESULT_VALUE_V2 = "result_value_v2"
BUDGET_OBJECTIVE_RESULT_VALUE_V3 = "result_value_v3"
BUDGET_OBJECTIVE_RESULT_VALUE_V4 = "result_value_v4"
BUDGET_OBJECTIVE_RESULT_VALUE_V4B = "result_value_v4b"
BUDGET_OBJECTIVE_RESULT_VALUE_V5 = "result_value_v5"
BUDGET_OBJECTIVE_RESULT_VALUE_V6 = "result_value_v6"
BUDGET_OBJECTIVE_RESULT_VALUE_V7 = "result_value_v7"
BUDGET_OBJECTIVE_RESULT_VALUE_V8 = "result_value_v8"
BUDGET_OBJECTIVE_RESULT_VALUE_V9 = "result_value_v9"
BUDGET_OBJECTIVE_RESULT_VALUE_V10 = "result_value_v10"
BUDGET_OBJECTIVE_CHOICES: tuple[str, ...] = (
    DEFAULT_BUDGET_OBJECTIVE,
    BUDGET_OBJECTIVE_RESULT_VALUE_V1,
    BUDGET_OBJECTIVE_RESULT_VALUE_V2,
    BUDGET_OBJECTIVE_RESULT_VALUE_V3,
    BUDGET_OBJECTIVE_RESULT_VALUE_V4,
    BUDGET_OBJECTIVE_RESULT_VALUE_V4B,
    BUDGET_OBJECTIVE_RESULT_VALUE_V5,
    BUDGET_OBJECTIVE_RESULT_VALUE_V6,
    BUDGET_OBJECTIVE_RESULT_VALUE_V7,
    BUDGET_OBJECTIVE_RESULT_VALUE_V8,
    BUDGET_OBJECTIVE_RESULT_VALUE_V9,
    BUDGET_OBJECTIVE_RESULT_VALUE_V10,
)
DEFAULT_OUTCOME_HORIZONS = (1, 3, 5, 10, 20)
MONTHLY_RETURN_COLUMNS = (
    "month",
    "start_date",
    "end_date",
    "trading_day_count",
    "monthly_return",
    "monthly_equity",
    "intramonth_max_drawdown",
)


def resolve_budget_objective(budget_objective: str | None) -> str:
    objective = str(budget_objective or DEFAULT_BUDGET_OBJECTIVE).strip() or DEFAULT_BUDGET_OBJECTIVE
    if objective in {"teacher", "imitation", "none", "default"}:
        objective = DEFAULT_BUDGET_OBJECTIVE
    if objective not in BUDGET_OBJECTIVE_CHOICES:
        raise ValueError(
            f"Unsupported budget objective: {objective}. Available: {', '.join(BUDGET_OBJECTIVE_CHOICES)}"
        )
    return objective


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


def _max_consecutive_negative(values: pd.Series) -> int:
    longest = 0
    current = 0
    for value in pd.Series(values, dtype=float).fillna(0.0):
        if float(value) < 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _empty_monthly_return_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(MONTHLY_RETURN_COLUMNS))


def build_monthly_return_frame(daily_returns: pd.Series) -> pd.DataFrame:
    clean = pd.Series(daily_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return _empty_monthly_return_frame()

    parsed_index = pd.to_datetime(clean.index, errors="coerce")
    rows: list[dict[str, Any]] = []
    if pd.Series(parsed_index).notna().all():
        working = pd.DataFrame({"date": parsed_index, "daily_return": clean.to_numpy(dtype=float)})
        working = working.sort_values("date").reset_index(drop=True)
        working["month"] = working["date"].dt.to_period("M").astype(str)
        grouped = working.groupby("month", sort=True)
    else:
        working = pd.DataFrame(
            {
                "date": pd.RangeIndex(start=0, stop=len(clean), step=1),
                "daily_return": clean.to_numpy(dtype=float),
            }
        )
        working["month"] = [f"block_{int(idx // 21) + 1:03d}" for idx in range(len(working))]
        grouped = working.groupby("month", sort=True)

    cumulative_equity = 1.0
    for month, group in grouped:
        returns = pd.Series(group["daily_return"], dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        monthly_return = float((1.0 + returns).prod() - 1.0)
        cumulative_equity *= 1.0 + monthly_return
        local_equity = pd.concat(
            [pd.Series([1.0], dtype=float), (1.0 + returns).cumprod().reset_index(drop=True)],
            ignore_index=True,
        )
        intramonth_drawdown = float((local_equity / local_equity.cummax() - 1.0).min()) if not local_equity.empty else 0.0
        first_date = group["date"].iloc[0]
        last_date = group["date"].iloc[-1]
        rows.append(
            {
                "month": str(month),
                "start_date": first_date.strftime("%Y-%m-%d") if hasattr(first_date, "strftime") else str(first_date),
                "end_date": last_date.strftime("%Y-%m-%d") if hasattr(last_date, "strftime") else str(last_date),
                "trading_day_count": int(len(group)),
                "monthly_return": monthly_return,
                "monthly_equity": float(cumulative_equity),
                "intramonth_max_drawdown": intramonth_drawdown,
            }
        )
    return pd.DataFrame(rows, columns=list(MONTHLY_RETURN_COLUMNS))


def _monthly_curve_metrics(daily_returns: pd.Series) -> dict[str, float]:
    monthly_frame = build_monthly_return_frame(daily_returns)
    if monthly_frame.empty:
        return {
            "monthly_count": 0.0,
            "monthly_return_mean": 0.0,
            "monthly_return_median": 0.0,
            "monthly_return_std": 0.0,
            "monthly_sharpe": 0.0,
            "monthly_win_rate": 0.0,
            "monthly_positive_count": 0.0,
            "monthly_negative_count": 0.0,
            "monthly_best_return": 0.0,
            "monthly_worst_return": 0.0,
            "monthly_gain_loss_ratio": 0.0,
            "monthly_max_consecutive_loss_months": 0.0,
            "monthly_intramonth_max_drawdown": 0.0,
            "monthly_consistency_score": 0.0,
        }
    monthly_returns = pd.to_numeric(monthly_frame["monthly_return"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if monthly_returns.empty:
        return _monthly_curve_metrics(pd.Series(dtype=float))
    positive = monthly_returns[monthly_returns > 0.0]
    negative = monthly_returns[monthly_returns < 0.0]
    monthly_std = float(monthly_returns.std(ddof=0))
    monthly_win_rate = float((monthly_returns > 0.0).mean())
    worst_month = float(monthly_returns.min())
    best_month = float(monthly_returns.max())
    gain_loss_ratio = (
        float(positive.mean() / abs(float(negative.mean())))
        if len(positive) and len(negative) and abs(float(negative.mean())) > 1.0e-12
        else (float("inf") if len(positive) and not len(negative) else 0.0)
    )
    gain_loss_ratio = float(min(gain_loss_ratio, 9.99)) if np.isfinite(gain_loss_ratio) else 9.99
    intramonth_drawdown = pd.to_numeric(monthly_frame["intramonth_max_drawdown"], errors="coerce").fillna(0.0)
    monthly_consistency_score = float(
        np.clip(
            monthly_win_rate * 0.48
            + np.clip((float(monthly_returns.mean()) + 0.04) / 0.10, 0.0, 1.0) * 0.24
            + (1.0 - np.clip(abs(min(worst_month, 0.0)) / 0.18, 0.0, 1.0)) * 0.18
            + (1.0 - np.clip(float(_max_consecutive_negative(monthly_returns)) / 4.0, 0.0, 1.0)) * 0.10,
            0.0,
            1.0,
        )
    )
    return {
        "monthly_count": float(len(monthly_returns)),
        "monthly_return_mean": float(monthly_returns.mean()),
        "monthly_return_median": float(monthly_returns.median()),
        "monthly_return_std": monthly_std,
        "monthly_sharpe": float(monthly_returns.mean() / monthly_std * np.sqrt(12.0)) if monthly_std > 0.0 else 0.0,
        "monthly_win_rate": monthly_win_rate,
        "monthly_positive_count": float((monthly_returns > 0.0).sum()),
        "monthly_negative_count": float((monthly_returns < 0.0).sum()),
        "monthly_best_return": best_month,
        "monthly_worst_return": worst_month,
        "monthly_gain_loss_ratio": gain_loss_ratio,
        "monthly_max_consecutive_loss_months": float(_max_consecutive_negative(monthly_returns)),
        "monthly_intramonth_max_drawdown": float(intramonth_drawdown.min()) if len(intramonth_drawdown) else 0.0,
        "monthly_consistency_score": monthly_consistency_score,
    }


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
            **_monthly_curve_metrics(clean),
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
        **_monthly_curve_metrics(clean),
    }


def _safe_corrcoef(left: pd.Series, right: pd.Series) -> float:
    aligned = pd.concat(
        [
            pd.to_numeric(left, errors="coerce").replace([np.inf, -np.inf], np.nan),
            pd.to_numeric(right, errors="coerce").replace([np.inf, -np.inf], np.nan),
        ],
        axis=1,
    ).dropna()
    if len(aligned) < 2:
        return 0.0
    left_std = float(aligned.iloc[:, 0].std(ddof=0))
    right_std = float(aligned.iloc[:, 1].std(ddof=0))
    if left_std <= 1.0e-12 or right_std <= 1.0e-12:
        return 0.0
    corr = float(np.corrcoef(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1])
    return corr if np.isfinite(corr) else 0.0


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


def _annotate_label_frame_with_execution_feedback(
    label_frame: pd.DataFrame,
    step_actions: list[dict[str, Any]],
) -> pd.DataFrame:
    if label_frame.empty:
        return label_frame
    working = label_frame.copy()
    if not step_actions:
        working["clipped_intent_risk"] = 0.0
        return working
    feedback = pd.DataFrame(step_actions)
    if feedback.empty or "stock" not in feedback.columns:
        working["clipped_intent_risk"] = 0.0
        return working
    feedback["stock"] = feedback["stock"].astype(str).str.upper()
    active_intent = feedback.get("model_action", pd.Series("", index=feedback.index)).astype(str).str.lower().isin(
        {"open", "add", "reduce", "exit"}
    )
    weight_change = feedback.get("weight_change_action", pd.Series("", index=feedback.index)).astype(str).str.lower()
    model_action = feedback.get("model_action", pd.Series("", index=feedback.index)).astype(str).str.lower()
    order_drift = active_intent & (model_action != weight_change)
    drift_sources = pd.DataFrame(index=feedback.index)
    for column in (
        "budget_dropped",
        "semantic_delta_guarded",
        "budget_split_bound_guarded",
        "forced_zero",
    ):
        drift_sources[column] = feedback.get(column, pd.Series(False, index=feedback.index)).astype(bool)
    clipped_risk = (
        order_drift.astype(float) * 0.60
        + drift_sources["budget_dropped"].astype(float) * 0.55
        + drift_sources["semantic_delta_guarded"].astype(float) * 0.35
        + drift_sources["budget_split_bound_guarded"].astype(float) * 0.35
        + drift_sources["forced_zero"].astype(float) * 0.20
    ).clip(0.0, 1.0)
    feedback = feedback.assign(
        stock_key=feedback["stock"].astype(str).str.upper(),
        clipped_intent_risk_feedback=clipped_risk.to_numpy(dtype=float),
    )
    working["_feedback_stock_key"] = working["stock"].astype(str).str.upper()
    merged = working.merge(
        feedback.loc[:, ["stock_key", "clipped_intent_risk_feedback"]],
        how="left",
        left_on="_feedback_stock_key",
        right_on="stock_key",
        suffixes=("", "_feedback"),
    )
    merged.index = working.index
    merged = merged.drop(columns=[column for column in ("_feedback_stock_key", "stock_key") if column in merged.columns])
    feedback_risk = (
        merged.pop("clipped_intent_risk_feedback")
        if "clipped_intent_risk_feedback" in merged.columns
        else pd.Series(0.0, index=merged.index, dtype=float)
    )
    merged["clipped_intent_risk"] = (
        pd.to_numeric(feedback_risk, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
    )
    return merged


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
        for optional_column in (
            "sell_rank_score",
            "lifecycle_sell_gate",
            "large_upside_1d_target",
            "alpha_opportunity_value",
            "hold_continuation_value",
            "sell_release_value",
            "cash_defense_value",
            "deployment_opportunity_cost",
            "risk_adjusted_action_value",
            "multi_horizon_forward_value",
            "multi_horizon_forward_risk",
            "multi_horizon_path_value",
            "open_action_value",
            "add_action_value",
            "hold_action_value",
            "reduce_action_value",
            "exit_action_value",
            "relative_opportunity_value",
            "action_value_consistency_target",
            "direct_action_value_applied",
            "direct_action_value_selected",
            "direct_action_value_gap",
            "direct_action_utility_skip",
            "direct_action_utility_open",
            "direct_action_utility_hold",
            "direct_action_utility_add",
            "direct_action_utility_reduce",
            "direct_action_utility_exit",
            "direct_action_keep_utility",
            "direct_action_release_utility",
            "direct_action_deploy_utility",
            "direct_action_release_advantage",
            "direct_action_deploy_advantage",
            "direct_action_pair_opportunity_spread",
            "direct_action_pair_source_opportunity_cost",
            "direct_action_pair_source_release_score",
            "portfolio_daily_receiver_score",
            "portfolio_daily_source_gap",
            "portfolio_daily_source_score",
            "portfolio_daily_cash_score",
            "value_arbitration_target",
            "deploy_value_target",
            "release_value_target",
            "defense_value_target",
            "deploy_gate_target",
            "release_gate_target",
            "defense_gate_target",
            "deploy_executability_target",
            "clipped_intent_risk",
        ):
            if optional_column not in action_outcomes.columns:
                action_outcomes[optional_column] = 0.0
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
        held_decision_rows = action_outcomes.loc[
            action_lookup.isin({"hold", "add", "reduce", "exit"})
            & (action_outcomes["hold_days_before"].fillna(0.0) > 0.0)
        ].copy()
        sell_selection_spreads: list[float] = []
        sell_selection_hits = 0

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
        if not held_decision_rows.empty:
            for _, day_rows in held_decision_rows.groupby("date", sort=False):
                sold_rows = day_rows.loc[day_rows["execution_action"].astype(str).isin({"reduce", "exit"})]
                kept_rows = day_rows.loc[day_rows["execution_action"].astype(str).isin({"hold", "add"})]
                if sold_rows.empty or kept_rows.empty:
                    continue
                sold_mean = float(pd.to_numeric(sold_rows["forward_excess_5d"], errors="coerce").dropna().mean())
                kept_mean = float(pd.to_numeric(kept_rows["forward_excess_5d"], errors="coerce").dropna().mean())
                if not (np.isfinite(sold_mean) and np.isfinite(kept_mean)):
                    continue
                spread = kept_mean - sold_mean
                sell_selection_spreads.append(float(spread))
                if spread > 0.0:
                    sell_selection_hits += 1
        metrics["sell_selection_quality_5d"] = (
            float(np.mean(sell_selection_spreads)) if sell_selection_spreads else 0.0
        )
        metrics["sell_selection_hit_rate_5d"] = (
            float(sell_selection_hits / len(sell_selection_spreads)) if sell_selection_spreads else 0.0
        )
        arbitration_forward_5d = pd.to_numeric(action_outcomes["forward_excess_5d"], errors="coerce")
        alpha_signal_all = pd.to_numeric(action_outcomes["alpha_opportunity_value"], errors="coerce")
        value_signal_all = pd.to_numeric(action_outcomes["value_arbitration_target"], errors="coerce")
        deployment_signal_all = pd.to_numeric(action_outcomes["deployment_opportunity_cost"], errors="coerce")
        large_upside_signal_all = pd.to_numeric(action_outcomes["large_upside_1d_target"], errors="coerce")
        risk_action_signal_all = pd.to_numeric(action_outcomes["risk_adjusted_action_value"], errors="coerce")
        multi_horizon_forward_value_all = pd.to_numeric(action_outcomes["multi_horizon_forward_value"], errors="coerce")
        multi_horizon_forward_risk_all = pd.to_numeric(action_outcomes["multi_horizon_forward_risk"], errors="coerce")
        multi_horizon_path_value_all = pd.to_numeric(action_outcomes["multi_horizon_path_value"], errors="coerce")
        open_action_value_all = pd.to_numeric(action_outcomes["open_action_value"], errors="coerce")
        add_action_value_all = pd.to_numeric(action_outcomes["add_action_value"], errors="coerce")
        hold_action_value_all = pd.to_numeric(action_outcomes["hold_action_value"], errors="coerce")
        reduce_action_value_all = pd.to_numeric(action_outcomes["reduce_action_value"], errors="coerce")
        exit_action_value_all = pd.to_numeric(action_outcomes["exit_action_value"], errors="coerce")
        relative_opportunity_value_all = pd.to_numeric(action_outcomes["relative_opportunity_value"], errors="coerce")
        action_value_consistency_all = pd.to_numeric(action_outcomes["action_value_consistency_target"], errors="coerce")
        deploy_value_signal_all = pd.to_numeric(action_outcomes["deploy_value_target"], errors="coerce")
        deploy_gate_signal_all = pd.to_numeric(action_outcomes["deploy_gate_target"], errors="coerce")
        deploy_executability_signal_all = pd.to_numeric(action_outcomes["deploy_executability_target"], errors="coerce")
        for signal_name, signal_values, metric_name in (
            ("alpha_opportunity_value", alpha_signal_all, "alpha_opportunity_forward_alignment_5d"),
            ("value_arbitration_target", value_signal_all, "value_arbitration_forward_alignment_5d"),
            ("deployment_opportunity_cost", deployment_signal_all, "deployment_opportunity_forward_alignment_5d"),
            ("large_upside_1d_target", large_upside_signal_all, "large_upside_forward_alignment_5d"),
            ("risk_adjusted_action_value", risk_action_signal_all, "risk_adjusted_action_forward_alignment_5d"),
            ("multi_horizon_forward_value", multi_horizon_forward_value_all, "multi_horizon_forward_value_alignment_5d"),
            ("multi_horizon_path_value", multi_horizon_path_value_all, "multi_horizon_path_value_alignment_5d"),
            ("open_action_value", open_action_value_all, "open_action_value_forward_alignment_5d"),
            ("add_action_value", add_action_value_all, "add_action_value_forward_alignment_5d"),
            ("hold_action_value", hold_action_value_all, "hold_action_value_forward_alignment_5d"),
            ("relative_opportunity_value", relative_opportunity_value_all, "relative_opportunity_forward_alignment_5d"),
            ("deploy_value_target", deploy_value_signal_all, "deploy_value_forward_alignment_5d"),
            ("deploy_gate_target", deploy_gate_signal_all, "deploy_gate_forward_alignment_5d"),
            ("deploy_executability_target", deploy_executability_signal_all, "deploy_executability_forward_alignment_5d"),
        ):
            valid_signal = signal_values.notna() & arbitration_forward_5d.notna()
            metrics[metric_name] = (
                float(_safe_corrcoef(signal_values.loc[valid_signal], arbitration_forward_5d.loc[valid_signal]))
                if int(valid_signal.sum()) >= 2
                else 0.0
            )
        for signal_name, signal_values, metric_name in (
            ("reduce_action_value", reduce_action_value_all, "reduce_action_value_forward_avoidance_5d"),
            ("exit_action_value", exit_action_value_all, "exit_action_value_forward_avoidance_5d"),
            ("multi_horizon_forward_risk", multi_horizon_forward_risk_all, "multi_horizon_forward_risk_avoidance_5d"),
        ):
            valid_signal = signal_values.notna() & arbitration_forward_5d.notna()
            metrics[metric_name] = (
                float(-_safe_corrcoef(signal_values.loc[valid_signal], arbitration_forward_5d.loc[valid_signal]))
                if int(valid_signal.sum()) >= 2
                else 0.0
            )
        cash_defense_signal_all = pd.to_numeric(action_outcomes["cash_defense_value"], errors="coerce")
        defense_value_signal_all = pd.to_numeric(action_outcomes["defense_value_target"], errors="coerce")
        defense_gate_signal_all = pd.to_numeric(action_outcomes["defense_gate_target"], errors="coerce")
        benchmark_1d_all = pd.to_numeric(action_outcomes["forward_benchmark_return_1d"], errors="coerce")
        cash_valid = cash_defense_signal_all.notna() & benchmark_1d_all.notna()
        metrics["cash_defense_timing_quality_1d"] = (
            float(-_safe_corrcoef(cash_defense_signal_all.loc[cash_valid], benchmark_1d_all.loc[cash_valid]))
            if int(cash_valid.sum()) >= 2
            else 0.0
        )
        defense_value_valid = defense_value_signal_all.notna() & benchmark_1d_all.notna()
        metrics["defense_value_timing_quality_1d"] = (
            float(-_safe_corrcoef(defense_value_signal_all.loc[defense_value_valid], benchmark_1d_all.loc[defense_value_valid]))
            if int(defense_value_valid.sum()) >= 2
            else 0.0
        )
        defense_gate_valid = defense_gate_signal_all.notna() & benchmark_1d_all.notna()
        metrics["defense_gate_timing_quality_1d"] = (
            float(-_safe_corrcoef(defense_gate_signal_all.loc[defense_gate_valid], benchmark_1d_all.loc[defense_gate_valid]))
            if int(defense_gate_valid.sum()) >= 2
            else 0.0
        )
        metrics["avg_alpha_opportunity_value"] = float(alpha_signal_all.fillna(0.0).mean())
        metrics["avg_hold_continuation_value"] = float(pd.to_numeric(action_outcomes["hold_continuation_value"], errors="coerce").fillna(0.0).mean())
        metrics["avg_sell_release_value"] = float(pd.to_numeric(action_outcomes["sell_release_value"], errors="coerce").fillna(0.0).mean())
        metrics["avg_cash_defense_value"] = float(cash_defense_signal_all.fillna(0.0).mean())
        metrics["avg_deployment_opportunity_cost"] = float(deployment_signal_all.fillna(0.0).mean())
        metrics["avg_value_arbitration_target"] = float(value_signal_all.fillna(0.0).mean())
        metrics["avg_multi_horizon_forward_value"] = float(multi_horizon_forward_value_all.fillna(0.0).mean())
        metrics["avg_multi_horizon_forward_risk"] = float(multi_horizon_forward_risk_all.fillna(0.0).mean())
        metrics["avg_multi_horizon_path_value"] = float(multi_horizon_path_value_all.fillna(0.0).mean())
        metrics["avg_open_action_value"] = float(open_action_value_all.fillna(0.0).mean())
        metrics["avg_add_action_value"] = float(add_action_value_all.fillna(0.0).mean())
        metrics["avg_hold_action_value"] = float(hold_action_value_all.fillna(0.0).mean())
        metrics["avg_reduce_action_value"] = float(reduce_action_value_all.fillna(0.0).mean())
        metrics["avg_exit_action_value"] = float(exit_action_value_all.fillna(0.0).mean())
        metrics["avg_relative_opportunity_value"] = float(relative_opportunity_value_all.fillna(0.0).mean())
        metrics["avg_action_value_consistency_target"] = float(action_value_consistency_all.fillna(0.5).mean())
        metrics["avg_deploy_value_target"] = float(deploy_value_signal_all.fillna(0.0).mean())
        metrics["avg_release_value_target"] = float(pd.to_numeric(action_outcomes["release_value_target"], errors="coerce").fillna(0.0).mean())
        metrics["avg_defense_value_target"] = float(defense_value_signal_all.fillna(0.0).mean())
        metrics["avg_deploy_gate_target"] = float(deploy_gate_signal_all.fillna(0.0).mean())
        metrics["avg_release_gate_target"] = float(pd.to_numeric(action_outcomes["release_gate_target"], errors="coerce").fillna(0.0).mean())
        metrics["avg_defense_gate_target"] = float(defense_gate_signal_all.fillna(0.0).mean())
        metrics["avg_deploy_executability_target"] = float(deploy_executability_signal_all.fillna(0.0).mean())
        model_action_lookup = action_outcomes.get(
            "model_action",
            pd.Series("", index=action_outcomes.index),
        ).astype(str).str.lower()
        weight_change_lookup = action_outcomes.get(
            "weight_change_action",
            action_outcomes.get("execution_action", pd.Series("", index=action_outcomes.index)),
        ).astype(str).str.lower()
        delta_weight_lookup = pd.to_numeric(
            action_outcomes.get("delta_weight", pd.Series(0.0, index=action_outcomes.index)),
            errors="coerce",
        ).fillna(0.0)
        deploy_intent_mask = model_action_lookup.isin({"open", "add"})
        add_intent_mask = model_action_lookup == "add"
        deploy_realized_mask = weight_change_lookup.isin({"open", "add"})
        deploy_positive_delta_mask = delta_weight_lookup > 1.0e-8
        add_to_hold_mask = add_intent_mask & (weight_change_lookup == "hold")
        deploy_hold_mask = deploy_intent_mask & (weight_change_lookup == "hold")
        deploy_intent_count = int(deploy_intent_mask.sum())
        add_intent_count = int(add_intent_mask.sum())
        action_value_table = pd.DataFrame(
            {
                "open": open_action_value_all.fillna(0.0),
                "add": add_action_value_all.fillna(0.0),
                "hold": hold_action_value_all.fillna(0.0),
                "reduce": reduce_action_value_all.fillna(0.0),
                "exit": exit_action_value_all.fillna(0.0),
            },
            index=action_outcomes.index,
        )
        best_action_value = action_value_table.max(axis=1)
        best_action_name = action_value_table.idxmax(axis=1)
        chosen_action_value = pd.Series(0.0, index=action_outcomes.index, dtype=float)
        for action_name in ("open", "add", "hold", "reduce", "exit"):
            chosen_action_value = chosen_action_value.where(
                model_action_lookup != action_name,
                action_value_table[action_name],
            )
        action_value_conflict_mask = (
            model_action_lookup.isin({"open", "add", "hold", "reduce", "exit"})
            & (best_action_value > chosen_action_value + 0.08)
        )
        held_value_rows = action_outcomes["hold_days_before"].fillna(0.0) > 0.0
        keep_action_value = pd.concat([add_action_value_all.fillna(0.0), hold_action_value_all.fillna(0.0)], axis=1).max(axis=1)
        release_action_value = pd.concat([reduce_action_value_all.fillna(0.0), exit_action_value_all.fillna(0.0)], axis=1).max(axis=1)
        sell_against_keep_value_mask = (
            held_value_rows
            & model_action_lookup.isin({"reduce", "exit"})
            & (keep_action_value > release_action_value + 0.08)
        )
        keep_against_release_value_mask = (
            held_value_rows
            & model_action_lookup.isin({"add", "hold"})
            & (release_action_value > keep_action_value + 0.08)
        )
        open_low_value_mask = (
            (model_action_lookup == "open")
            & (open_action_value_all.fillna(0.0) < 0.30)
        )
        metrics["action_value_conflict_share"] = (
            float(action_value_conflict_mask.mean()) if len(action_value_conflict_mask) else 0.0
        )
        metrics["action_value_selected_gap"] = (
            float((best_action_value - chosen_action_value).where(action_value_conflict_mask, 0.0).mean())
            if len(best_action_value)
            else 0.0
        )
        metrics["held_keep_release_value_gap"] = (
            float((keep_action_value - release_action_value).where(held_value_rows, 0.0).sum() / max(float(held_value_rows.sum()), 1.0))
            if len(keep_action_value)
            else 0.0
        )
        metrics["sell_against_keep_value_share"] = (
            float(sell_against_keep_value_mask.sum() / max(float(model_action_lookup.isin({"reduce", "exit"}).sum()), 1.0))
            if len(sell_against_keep_value_mask)
            else 0.0
        )
        metrics["keep_against_release_value_share"] = (
            float(keep_against_release_value_mask.sum() / max(float(model_action_lookup.isin({"add", "hold"}).sum()), 1.0))
            if len(keep_against_release_value_mask)
            else 0.0
        )
        metrics["open_low_action_value_share"] = (
            float(open_low_value_mask.sum() / max(float((model_action_lookup == "open").sum()), 1.0))
            if len(open_low_value_mask)
            else 0.0
        )
        metrics["action_value_consistency_score"] = float(
            np.clip(
                1.0
                - metrics["action_value_conflict_share"] * 0.46
                - metrics["sell_against_keep_value_share"] * 0.28
                - metrics["keep_against_release_value_share"] * 0.18
                - metrics["open_low_action_value_share"] * 0.08,
                0.0,
                1.0,
            )
        )
        policy_decision_mode_lookup = action_outcomes.get(
            "policy_decision_mode",
            pd.Series("", index=action_outcomes.index),
        ).astype(str).str.lower()
        direct_action_label_lookup = action_outcomes.get(
            "direct_action_value_label",
            pd.Series("", index=action_outcomes.index),
        ).astype(str).str.lower()
        direct_action_applied = pd.to_numeric(
            action_outcomes.get("direct_action_value_applied", pd.Series(0.0, index=action_outcomes.index)),
            errors="coerce",
        ).fillna(0.0)
        direct_selected_value = pd.to_numeric(
            action_outcomes.get("direct_action_value_selected", pd.Series(0.0, index=action_outcomes.index)),
            errors="coerce",
        ).fillna(0.0)
        direct_value_gap = pd.to_numeric(
            action_outcomes.get("direct_action_value_gap", pd.Series(0.0, index=action_outcomes.index)),
            errors="coerce",
        ).fillna(0.0)
        direct_mode_mask = policy_decision_mode_lookup.eq("direct_action_value_v1") | (direct_action_applied > 0.5)
        metrics["direct_action_value_mode_share"] = (
            float(direct_mode_mask.mean()) if len(direct_mode_mask) else 0.0
        )
        metrics["direct_action_value_label_match_share"] = (
            float((model_action_lookup.loc[direct_mode_mask] == direct_action_label_lookup.loc[direct_mode_mask]).mean())
            if bool(direct_mode_mask.any())
            else 0.0
        )
        metrics["direct_action_value_selected_mean"] = (
            float(direct_selected_value.loc[direct_mode_mask].mean()) if bool(direct_mode_mask.any()) else 0.0
        )
        metrics["direct_action_value_gap_mean"] = (
            float(direct_value_gap.loc[direct_mode_mask].mean()) if bool(direct_mode_mask.any()) else 0.0
        )
        metrics["direct_action_value_low_margin_share"] = (
            float((direct_value_gap.loc[direct_mode_mask] < 0.05).mean()) if bool(direct_mode_mask.any()) else 0.0
        )
        metrics["direct_action_order_translation_conflict_rate"] = (
            float((model_action_lookup.loc[direct_mode_mask] != weight_change_lookup.loc[direct_mode_mask]).mean())
            if bool(direct_mode_mask.any())
            else 0.0
        )
        direct_funding_authorized = action_outcomes.get(
            "direct_action_funding_release_authorized",
            pd.Series(False, index=action_outcomes.index),
        ).astype(bool)
        direct_funding_protected = action_outcomes.get(
            "direct_action_funding_protected",
            pd.Series(False, index=action_outcomes.index),
        ).astype(bool)
        sell_origin_lookup = action_outcomes.get(
            "sell_execution_origin",
            pd.Series("", index=action_outcomes.index),
        ).astype(str).str.lower()
        direct_release_advantage = pd.to_numeric(
            action_outcomes.get("direct_action_release_advantage", pd.Series(0.0, index=action_outcomes.index)),
            errors="coerce",
        ).fillna(0.0)
        direct_deploy_advantage = pd.to_numeric(
            action_outcomes.get("direct_action_deploy_advantage", pd.Series(0.0, index=action_outcomes.index)),
            errors="coerce",
        ).fillna(0.0)
        direct_deploy_signal = action_outcomes.get(
            "direct_action_deploy_signal",
            pd.Series(False, index=action_outcomes.index),
        ).astype(bool)
        direct_core_deploy_target = action_outcomes.get(
            "direct_action_core_deploy_target",
            pd.Series(False, index=action_outcomes.index),
        ).astype(bool)
        direct_add_authorized = action_outcomes.get(
            "direct_action_add_authorized",
            pd.Series(False, index=action_outcomes.index),
        ).astype(bool)
        direct_deploy_authorized = action_outcomes.get(
            "direct_action_deploy_authorized",
            pd.Series(False, index=action_outcomes.index),
        ).astype(bool)
        direct_reallocation_source = action_outcomes.get(
            "direct_action_reallocation_source",
            pd.Series(False, index=action_outcomes.index),
        ).astype(bool)
        direct_pair_reallocation_source = action_outcomes.get(
            "direct_action_pair_reallocation_source",
            pd.Series(False, index=action_outcomes.index),
        ).astype(bool)
        direct_pair_cost_guard_pass = action_outcomes.get(
            "direct_action_pair_cost_guard_pass",
            pd.Series(False, index=action_outcomes.index),
        ).astype(bool)
        direct_pair_cost_guard_blocked = action_outcomes.get(
            "direct_action_pair_cost_guard_blocked",
            pd.Series(False, index=action_outcomes.index),
        ).astype(bool)
        direct_pair_opportunity_spread = pd.to_numeric(
            action_outcomes.get("direct_action_pair_opportunity_spread", pd.Series(0.0, index=action_outcomes.index)),
            errors="coerce",
        ).fillna(0.0)
        direct_pair_source_cost = pd.to_numeric(
            action_outcomes.get(
                "direct_action_pair_source_opportunity_cost",
                pd.Series(0.0, index=action_outcomes.index),
            ),
            errors="coerce",
        ).fillna(0.0)
        portfolio_receiver_target = action_outcomes.get(
            "portfolio_daily_receiver_target",
            pd.Series(False, index=action_outcomes.index),
        ).astype(bool)
        portfolio_receiver_exec_guarded = action_outcomes.get(
            "portfolio_daily_receiver_exec_guarded",
            pd.Series(False, index=action_outcomes.index),
        ).astype(bool)
        portfolio_source_candidate = action_outcomes.get(
            "portfolio_daily_source_candidate",
            pd.Series(False, index=action_outcomes.index),
        ).astype(bool)
        portfolio_source_target = action_outcomes.get(
            "portfolio_daily_source_target",
            pd.Series(False, index=action_outcomes.index),
        ).astype(bool)
        portfolio_cash_reserve_signal = action_outcomes.get(
            "portfolio_daily_cash_reserve_signal",
            pd.Series(False, index=action_outcomes.index),
        ).astype(bool)
        portfolio_receiver_score = pd.to_numeric(
            action_outcomes.get("portfolio_daily_receiver_score", pd.Series(0.0, index=action_outcomes.index)),
            errors="coerce",
        ).fillna(0.0)
        portfolio_receiver_add_headroom = pd.to_numeric(
            action_outcomes.get(
                "portfolio_daily_receiver_add_headroom",
                pd.Series(0.0, index=action_outcomes.index),
            ),
            errors="coerce",
        ).fillna(0.0)
        portfolio_receiver_min_add_delta = pd.to_numeric(
            action_outcomes.get(
                "portfolio_daily_receiver_min_add_delta",
                pd.Series(0.0, index=action_outcomes.index),
            ),
            errors="coerce",
        ).fillna(0.0)
        portfolio_source_gap = pd.to_numeric(
            action_outcomes.get("portfolio_daily_source_gap", pd.Series(0.0, index=action_outcomes.index)),
            errors="coerce",
        ).fillna(0.0)
        portfolio_source_score = pd.to_numeric(
            action_outcomes.get("portfolio_daily_source_score", pd.Series(0.0, index=action_outcomes.index)),
            errors="coerce",
        ).fillna(0.0)
        portfolio_cash_score = pd.to_numeric(
            action_outcomes.get("portfolio_daily_cash_score", pd.Series(0.0, index=action_outcomes.index)),
            errors="coerce",
        ).fillna(0.0)
        direct_funding_sell_mask = weight_change_lookup.isin({"reduce", "exit"}) & sell_origin_lookup.eq(
            "deploy_funding_rebalance"
        )
        portfolio_daily_source_sell_mask = weight_change_lookup.isin({"reduce", "exit"}) & sell_origin_lookup.eq(
            "portfolio_daily_ranking_source"
        )
        metrics["direct_action_intent_preserved_share"] = (
            float((direct_action_label_lookup.loc[direct_mode_mask] == weight_change_lookup.loc[direct_mode_mask]).mean())
            if bool(direct_mode_mask.any())
            else 0.0
        )
        metrics["direct_action_funding_authorized_sell_share"] = (
            float(direct_funding_authorized.loc[direct_funding_sell_mask].mean())
            if bool(direct_funding_sell_mask.any())
            else 0.0
        )
        metrics["direct_action_funding_protected_sell_share"] = (
            float(direct_funding_protected.loc[direct_funding_sell_mask].mean())
            if bool(direct_funding_sell_mask.any())
            else 0.0
        )
        metrics["direct_action_release_advantage_mean"] = (
            float(direct_release_advantage.loc[direct_mode_mask].mean()) if bool(direct_mode_mask.any()) else 0.0
        )
        metrics["direct_action_deploy_advantage_mean"] = (
            float(direct_deploy_advantage.loc[direct_mode_mask].mean()) if bool(direct_mode_mask.any()) else 0.0
        )
        metrics["direct_action_deploy_signal_count"] = float(direct_deploy_signal.sum())
        metrics["direct_action_core_deploy_target_count"] = float(direct_core_deploy_target.sum())
        metrics["direct_action_core_deploy_target_realized_rate"] = (
            float(
                (direct_core_deploy_target & weight_change_lookup.isin({"open", "add"})).sum()
                / direct_core_deploy_target.sum()
            )
            if bool(direct_core_deploy_target.any())
            else 0.0
        )
        metrics["direct_action_add_authorized_count"] = float(direct_add_authorized.sum())
        metrics["direct_action_add_authorized_realized_rate"] = (
            float((direct_add_authorized & weight_change_lookup.eq("add")).sum() / direct_add_authorized.sum())
            if bool(direct_add_authorized.any())
            else 0.0
        )
        metrics["direct_action_deploy_authorized_count"] = float(direct_deploy_authorized.sum())
        metrics["direct_action_deploy_authorized_realized_rate"] = (
            float(
                (direct_deploy_authorized & weight_change_lookup.isin({"open", "add"})).sum()
                / direct_deploy_authorized.sum()
            )
            if bool(direct_deploy_authorized.any())
            else 0.0
        )
        metrics["direct_action_reallocation_source_count"] = float(direct_reallocation_source.sum())
        metrics["direct_action_pair_reallocation_source_count"] = float(direct_pair_reallocation_source.sum())
        direct_pair_guard_observed_count = float(direct_pair_cost_guard_pass.sum() + direct_pair_cost_guard_blocked.sum())
        metrics["direct_action_pair_cost_guard_pass_count"] = float(direct_pair_cost_guard_pass.sum())
        metrics["direct_action_pair_cost_guard_blocked_count"] = float(direct_pair_cost_guard_blocked.sum())
        metrics["direct_action_pair_cost_guard_pass_rate"] = (
            float(direct_pair_cost_guard_pass.sum() / direct_pair_guard_observed_count)
            if direct_pair_guard_observed_count > 0.0
            else 0.0
        )
        metrics["direct_action_pair_source_spread_mean"] = (
            float(direct_pair_opportunity_spread.loc[direct_pair_reallocation_source].mean())
            if bool(direct_pair_reallocation_source.any())
            else 0.0
        )
        metrics["direct_action_pair_source_cost_mean"] = (
            float(direct_pair_source_cost.loc[direct_pair_reallocation_source].mean())
            if bool(direct_pair_reallocation_source.any())
            else 0.0
        )
        metrics["direct_action_pair_source_forward_excess_5d"] = (
            float(arbitration_forward_5d.loc[direct_pair_reallocation_source].mean())
            if bool(direct_pair_reallocation_source.any())
            else 0.0
        )
        metrics["direct_action_core_target_forward_excess_5d"] = (
            float(arbitration_forward_5d.loc[direct_core_deploy_target].mean())
            if bool(direct_core_deploy_target.any())
            else 0.0
        )
        metrics["direct_action_core_minus_pair_forward_excess_5d"] = (
            metrics["direct_action_core_target_forward_excess_5d"]
            - metrics["direct_action_pair_source_forward_excess_5d"]
            if bool(direct_core_deploy_target.any()) and bool(direct_pair_reallocation_source.any())
            else 0.0
        )
        portfolio_receiver_realized_count = float(
            (portfolio_receiver_target & weight_change_lookup.isin({"open", "add"})).sum()
        )
        metrics["portfolio_daily_receiver_exec_guard_count"] = float(portfolio_receiver_exec_guarded.sum())
        metrics["portfolio_daily_receiver_add_headroom_mean"] = (
            float(portfolio_receiver_add_headroom.loc[portfolio_receiver_exec_guarded].mean())
            if bool(portfolio_receiver_exec_guarded.any())
            else 0.0
        )
        metrics["portfolio_daily_receiver_min_add_delta_mean"] = (
            float(portfolio_receiver_min_add_delta.loc[portfolio_receiver_exec_guarded].mean())
            if bool(portfolio_receiver_exec_guarded.any())
            else 0.0
        )
        metrics["portfolio_daily_receiver_target_count"] = float(portfolio_receiver_target.sum())
        metrics["portfolio_daily_receiver_realized_deploy_rate"] = (
            float(portfolio_receiver_realized_count / portfolio_receiver_target.sum())
            if bool(portfolio_receiver_target.any())
            else 0.0
        )
        metrics["portfolio_daily_receiver_unrealized_deploy_count"] = float(
            (portfolio_receiver_target & (~weight_change_lookup.isin({"open", "add"}))).sum()
        )
        metrics["portfolio_daily_receiver_unrealized_deploy_share"] = (
            float(metrics["portfolio_daily_receiver_unrealized_deploy_count"] / portfolio_receiver_target.sum())
            if bool(portfolio_receiver_target.any())
            else 0.0
        )
        metrics["portfolio_daily_source_candidate_count"] = float(portfolio_source_candidate.sum())
        metrics["portfolio_daily_source_target_count"] = float(portfolio_source_target.sum())
        metrics["portfolio_daily_source_sell_count"] = float(portfolio_daily_source_sell_mask.sum())
        metrics["portfolio_daily_source_realized_sell_rate"] = (
            float((portfolio_source_target & weight_change_lookup.isin({"reduce", "exit"})).sum() / portfolio_source_target.sum())
            if bool(portfolio_source_target.any())
            else 0.0
        )
        metrics["portfolio_daily_cash_score_mean"] = float(portfolio_cash_score.mean()) if len(portfolio_cash_score) else 0.0
        metrics["portfolio_daily_cash_reserve_rate"] = (
            float(portfolio_cash_reserve_signal.mean()) if len(portfolio_cash_reserve_signal) else 0.0
        )
        metrics["portfolio_daily_receiver_score_mean"] = (
            float(portfolio_receiver_score.loc[portfolio_receiver_target].mean())
            if bool(portfolio_receiver_target.any())
            else 0.0
        )
        metrics["portfolio_daily_source_score_mean"] = (
            float(portfolio_source_score.loc[portfolio_source_target].mean())
            if bool(portfolio_source_target.any())
            else 0.0
        )
        metrics["portfolio_daily_source_gap_mean"] = (
            float(portfolio_source_gap.loc[portfolio_source_target].mean())
            if bool(portfolio_source_target.any())
            else 0.0
        )
        metrics["portfolio_daily_receiver_forward_excess_5d"] = (
            float(arbitration_forward_5d.loc[portfolio_receiver_target].mean())
            if bool(portfolio_receiver_target.any())
            else 0.0
        )
        metrics["portfolio_daily_source_forward_excess_5d"] = (
            float(arbitration_forward_5d.loc[portfolio_source_target].mean())
            if bool(portfolio_source_target.any())
            else 0.0
        )
        metrics["portfolio_daily_receiver_minus_source_forward_excess_5d"] = (
            metrics["portfolio_daily_receiver_forward_excess_5d"]
            - metrics["portfolio_daily_source_forward_excess_5d"]
            if bool(portfolio_receiver_target.any()) and bool(portfolio_source_target.any())
            else 0.0
        )
        metrics["deploy_intent_action_count"] = float(deploy_intent_count)
        metrics["deploy_intent_realized_count"] = float((deploy_intent_mask & deploy_realized_mask).sum())
        metrics["deploy_intent_realized_rate"] = (
            float((deploy_intent_mask & deploy_realized_mask).sum() / deploy_intent_count) if deploy_intent_count else 0.0
        )
        metrics["open_add_positive_weight_change_rate"] = (
            float((deploy_intent_mask & deploy_positive_delta_mask).sum() / deploy_intent_count) if deploy_intent_count else 0.0
        )
        metrics["add_to_hold_conflict_count"] = float(add_to_hold_mask.sum())
        metrics["add_to_hold_conflict_share"] = (
            float(add_to_hold_mask.sum() / add_intent_count) if add_intent_count else 0.0
        )
        metrics["deploy_intent_hold_conflict_share"] = (
            float(deploy_hold_mask.sum() / deploy_intent_count) if deploy_intent_count else 0.0
        )
        held_rank_rows = held_decision_rows.copy()
        if not held_rank_rows.empty:
            rank_signal = pd.to_numeric(held_rank_rows["sell_rank_score"], errors="coerce")
            lifecycle_signal = pd.to_numeric(held_rank_rows["lifecycle_sell_gate"], errors="coerce")
            sell_release_signal = pd.to_numeric(held_rank_rows["sell_release_value"], errors="coerce")
            release_value_signal = pd.to_numeric(held_rank_rows["release_value_target"], errors="coerce")
            release_gate_signal = pd.to_numeric(held_rank_rows["release_gate_target"], errors="coerce")
            hold_value_signal = pd.to_numeric(held_rank_rows["hold_continuation_value"], errors="coerce")
            clip_signal = pd.to_numeric(held_rank_rows["clipped_intent_risk"], errors="coerce")
            forward_5d = pd.to_numeric(held_rank_rows["forward_excess_5d"], errors="coerce")
            valid_rank = rank_signal.notna() & forward_5d.notna()
            metrics["sell_rank_forward_alignment_5d"] = (
                float(-_safe_corrcoef(rank_signal.loc[valid_rank], forward_5d.loc[valid_rank]))
                if int(valid_rank.sum()) >= 2
                else 0.0
            )
            valid_gate = lifecycle_signal.notna() & forward_5d.notna()
            metrics["lifecycle_sell_gate_forward_alignment_5d"] = (
                float(-_safe_corrcoef(lifecycle_signal.loc[valid_gate], forward_5d.loc[valid_gate]))
                if int(valid_gate.sum()) >= 2
                else 0.0
            )
            valid_sell_release = sell_release_signal.notna() & forward_5d.notna()
            metrics["sell_release_forward_alignment_5d"] = (
                float(-_safe_corrcoef(sell_release_signal.loc[valid_sell_release], forward_5d.loc[valid_sell_release]))
                if int(valid_sell_release.sum()) >= 2
                else 0.0
            )
            valid_release_value = release_value_signal.notna() & forward_5d.notna()
            metrics["release_value_forward_alignment_5d"] = (
                float(-_safe_corrcoef(release_value_signal.loc[valid_release_value], forward_5d.loc[valid_release_value]))
                if int(valid_release_value.sum()) >= 2
                else 0.0
            )
            valid_release_gate = release_gate_signal.notna() & forward_5d.notna()
            metrics["release_gate_forward_alignment_5d"] = (
                float(-_safe_corrcoef(release_gate_signal.loc[valid_release_gate], forward_5d.loc[valid_release_gate]))
                if int(valid_release_gate.sum()) >= 2
                else 0.0
            )
            valid_hold_value = hold_value_signal.notna() & forward_5d.notna()
            metrics["hold_continuation_forward_alignment_5d"] = (
                float(_safe_corrcoef(hold_value_signal.loc[valid_hold_value], forward_5d.loc[valid_hold_value]))
                if int(valid_hold_value.sum()) >= 2
                else 0.0
            )
            metrics["avg_sell_rank_score"] = float(rank_signal.fillna(0.0).mean())
            metrics["avg_lifecycle_sell_gate"] = float(lifecycle_signal.fillna(0.0).mean())
            metrics["avg_clipped_intent_risk"] = float(clip_signal.fillna(0.0).mean())
            order_conflict = held_rank_rows.get("weight_change_action", held_rank_rows.get("execution_action", pd.Series("", index=held_rank_rows.index))).astype(str) != held_rank_rows.get("model_action", pd.Series("", index=held_rank_rows.index)).astype(str)
            if bool(order_conflict.any()) and bool((~order_conflict).any()):
                metrics["clipped_intent_risk_conflict_gap"] = float(
                    clip_signal.loc[order_conflict].fillna(0.0).mean()
                    - clip_signal.loc[~order_conflict].fillna(0.0).mean()
                )
            else:
                metrics["clipped_intent_risk_conflict_gap"] = 0.0
        else:
            metrics["sell_rank_forward_alignment_5d"] = 0.0
            metrics["lifecycle_sell_gate_forward_alignment_5d"] = 0.0
            metrics["sell_release_forward_alignment_5d"] = 0.0
            metrics["release_value_forward_alignment_5d"] = 0.0
            metrics["release_gate_forward_alignment_5d"] = 0.0
            metrics["hold_continuation_forward_alignment_5d"] = 0.0
            metrics["avg_sell_rank_score"] = 0.0
            metrics["avg_lifecycle_sell_gate"] = 0.0
            metrics["avg_clipped_intent_risk"] = 0.0
            metrics["clipped_intent_risk_conflict_gap"] = 0.0

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
            "sell_selection_quality_5d",
            "sell_selection_hit_rate_5d",
            "alpha_opportunity_forward_alignment_5d",
            "value_arbitration_forward_alignment_5d",
            "deployment_opportunity_forward_alignment_5d",
            "large_upside_forward_alignment_5d",
            "risk_adjusted_action_forward_alignment_5d",
            "deploy_value_forward_alignment_5d",
            "deploy_gate_forward_alignment_5d",
            "deploy_executability_forward_alignment_5d",
            "cash_defense_timing_quality_1d",
            "defense_value_timing_quality_1d",
            "defense_gate_timing_quality_1d",
            "sell_rank_forward_alignment_5d",
            "lifecycle_sell_gate_forward_alignment_5d",
            "sell_release_forward_alignment_5d",
            "release_value_forward_alignment_5d",
            "release_gate_forward_alignment_5d",
            "hold_continuation_forward_alignment_5d",
            "avg_sell_rank_score",
            "avg_lifecycle_sell_gate",
            "avg_alpha_opportunity_value",
            "avg_hold_continuation_value",
            "avg_sell_release_value",
            "avg_cash_defense_value",
            "avg_deployment_opportunity_cost",
            "avg_value_arbitration_target",
            "avg_deploy_value_target",
            "avg_release_value_target",
            "avg_defense_value_target",
            "avg_deploy_gate_target",
            "avg_release_gate_target",
            "avg_defense_gate_target",
            "avg_deploy_executability_target",
            "action_value_conflict_share",
            "action_value_selected_gap",
            "held_keep_release_value_gap",
            "sell_against_keep_value_share",
            "keep_against_release_value_share",
            "open_low_action_value_share",
            "action_value_consistency_score",
            "direct_action_value_mode_share",
            "direct_action_value_label_match_share",
            "direct_action_value_selected_mean",
            "direct_action_value_gap_mean",
            "direct_action_value_low_margin_share",
            "direct_action_order_translation_conflict_rate",
            "direct_action_intent_preserved_share",
            "direct_action_funding_authorized_sell_share",
            "direct_action_funding_protected_sell_share",
            "direct_action_release_advantage_mean",
            "direct_action_deploy_advantage_mean",
            "direct_action_deploy_signal_count",
            "direct_action_core_deploy_target_count",
            "direct_action_core_deploy_target_realized_rate",
            "direct_action_add_authorized_count",
            "direct_action_add_authorized_realized_rate",
            "direct_action_deploy_authorized_count",
            "direct_action_deploy_authorized_realized_rate",
            "direct_action_reallocation_source_count",
            "direct_action_pair_reallocation_source_count",
            "direct_action_pair_cost_guard_pass_count",
            "direct_action_pair_cost_guard_blocked_count",
            "direct_action_pair_cost_guard_pass_rate",
            "direct_action_pair_source_spread_mean",
            "direct_action_pair_source_cost_mean",
            "direct_action_pair_source_forward_excess_5d",
            "direct_action_core_target_forward_excess_5d",
            "direct_action_core_minus_pair_forward_excess_5d",
            "portfolio_daily_receiver_exec_guard_count",
            "portfolio_daily_receiver_add_headroom_mean",
            "portfolio_daily_receiver_min_add_delta_mean",
            "portfolio_daily_receiver_target_count",
            "portfolio_daily_receiver_realized_deploy_rate",
            "portfolio_daily_receiver_unrealized_deploy_count",
            "portfolio_daily_receiver_unrealized_deploy_share",
            "portfolio_daily_source_candidate_count",
            "portfolio_daily_source_target_count",
            "portfolio_daily_source_sell_count",
            "portfolio_daily_source_realized_sell_rate",
            "portfolio_daily_source_exec_guard_count",
            "portfolio_daily_source_exec_cap_guard_count",
            "portfolio_daily_source_target_not_sold_count",
            "portfolio_daily_source_target_not_sold_share",
            "portfolio_daily_source_realized_reduction_weight",
            "portfolio_daily_receiver_realized_deploy_count",
            "portfolio_daily_effective_capital_transfer_count",
            "portfolio_daily_cash_score_mean",
            "portfolio_daily_cash_reserve_rate",
            "portfolio_daily_receiver_score_mean",
            "portfolio_daily_source_score_mean",
            "portfolio_daily_source_gap_mean",
            "portfolio_daily_receiver_forward_excess_5d",
            "portfolio_daily_source_forward_excess_5d",
            "portfolio_daily_receiver_minus_source_forward_excess_5d",
            "deploy_intent_action_count",
            "deploy_intent_realized_count",
            "deploy_intent_realized_rate",
            "open_add_positive_weight_change_rate",
            "add_to_hold_conflict_count",
            "add_to_hold_conflict_share",
            "deploy_intent_hold_conflict_share",
            "avg_clipped_intent_risk",
            "clipped_intent_risk_conflict_gap",
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
            corr = _safe_corrcoef(cash_series.loc[valid], benchmark_series.loc[valid])
            metrics["cash_timing_quality_1d"] = float(-corr) if np.isfinite(corr) else 0.0
        else:
            metrics["cash_timing_quality_1d"] = 0.0
        for signal_column, metric_name, preferred_sign in (
            ("budget_model_risk_signal", "budget_model_risk_timing_quality_1d", -1.0),
            ("budget_model_deploy_signal", "budget_model_deploy_timing_quality_1d", 1.0),
            ("budget_model_cash_timing_signal", "budget_model_cash_timing_quality_1d", -1.0),
            ("budget_model_risk_deploy_gap", "budget_model_risk_deploy_gap_quality_1d", -1.0),
        ):
            if signal_column in aligned.columns:
                signal_series = aligned[signal_column].astype(float)
                signal_valid = signal_series.notna() & benchmark_series.notna()
                if int(signal_valid.sum()) >= 2:
                    corr = _safe_corrcoef(signal_series.loc[signal_valid], benchmark_series.loc[signal_valid])
                    metrics[metric_name] = float(corr * preferred_sign) if np.isfinite(corr) else 0.0
                else:
                    metrics[metric_name] = 0.0
            else:
                metrics[metric_name] = 0.0
        high_cash_threshold = float(cash_series.quantile(0.75)) if len(cash_series) else 0.0
        high_cash_mask = valid & (cash_series >= high_cash_threshold)
        metrics["high_cash_share"] = float(high_cash_mask.mean()) if len(cash_series) else 0.0
        metrics["high_cash_down_market_hit_rate"] = (
            float((benchmark_series.loc[high_cash_mask] < 0).mean()) if bool(high_cash_mask.any()) else 0.0
        )
        metrics["risk_off_cash_hit_rate"] = metrics["high_cash_down_market_hit_rate"]
    else:
        metrics["cash_timing_quality_1d"] = 0.0
        metrics["budget_model_risk_timing_quality_1d"] = 0.0
        metrics["budget_model_deploy_timing_quality_1d"] = 0.0
        metrics["budget_model_cash_timing_quality_1d"] = 0.0
        metrics["budget_model_risk_deploy_gap_quality_1d"] = 0.0
        metrics["high_cash_share"] = 0.0
        metrics["high_cash_down_market_hit_rate"] = 0.0
        metrics["risk_off_cash_hit_rate"] = 0.0
    if not turnover_frame.empty:
        metrics["avg_position_cap_target"] = float(turnover_frame["max_position_weight_target"].mean())
        metrics["avg_hold_bias_target"] = float(turnover_frame["hold_bias_target"].mean())
        metrics["avg_reduce_bias_target"] = float(turnover_frame["reduce_bias_target"].mean()) if "reduce_bias_target" in turnover_frame.columns else 0.0
        metrics["avg_exit_patience_target"] = float(turnover_frame["exit_patience_target"].mean()) if "exit_patience_target" in turnover_frame.columns else 0.0
        metrics["avg_reentry_guard_target"] = float(turnover_frame["reentry_guard_target"].mean()) if "reentry_guard_target" in turnover_frame.columns else 0.0
        metrics["avg_budget_risk_off_score"] = float(turnover_frame["budget_risk_off_score"].mean()) if "budget_risk_off_score" in turnover_frame.columns else 0.0
        metrics["avg_budget_deploy_score"] = float(turnover_frame["budget_deploy_score"].mean()) if "budget_deploy_score" in turnover_frame.columns else 0.0
        metrics["avg_budget_entry_candidate_count"] = float(turnover_frame["budget_entry_candidate_count"].mean()) if "budget_entry_candidate_count" in turnover_frame.columns else 0.0
        metrics["avg_budget_entry_keep_count"] = float(turnover_frame["budget_entry_keep_count"].mean()) if "budget_entry_keep_count" in turnover_frame.columns else 0.0
        metrics["avg_budget_held_protected_count"] = float(turnover_frame["budget_held_protected_count"].mean()) if "budget_held_protected_count" in turnover_frame.columns else 0.0
        metrics["avg_budget_split_bound_guard_count"] = float(turnover_frame["budget_split_bound_guard_count"].mean()) if "budget_split_bound_guard_count" in turnover_frame.columns else 0.0
        metrics["avg_budget_translation_floor_guard_count"] = float(turnover_frame["budget_translation_floor_guard_count"].mean()) if "budget_translation_floor_guard_count" in turnover_frame.columns else 0.0
        metrics["avg_budget_translation_cap_guard_count"] = float(turnover_frame["budget_translation_cap_guard_count"].mean()) if "budget_translation_cap_guard_count" in turnover_frame.columns else 0.0
        metrics["avg_budget_sell_priority_guard_count"] = float(turnover_frame["budget_sell_priority_guard_count"].mean()) if "budget_sell_priority_guard_count" in turnover_frame.columns else 0.0
        metrics["avg_budget_model_risk_signal"] = float(turnover_frame["budget_model_risk_signal"].mean()) if "budget_model_risk_signal" in turnover_frame.columns else 0.0
        metrics["avg_budget_model_deploy_signal"] = float(turnover_frame["budget_model_deploy_signal"].mean()) if "budget_model_deploy_signal" in turnover_frame.columns else 0.0
        metrics["avg_budget_model_cash_timing_signal"] = float(turnover_frame["budget_model_cash_timing_signal"].mean()) if "budget_model_cash_timing_signal" in turnover_frame.columns else 0.0
        metrics["avg_budget_model_alpha_focus_signal"] = float(turnover_frame["budget_model_alpha_focus_signal"].mean()) if "budget_model_alpha_focus_signal" in turnover_frame.columns else 0.0
        metrics["avg_budget_model_value_arbitration_signal"] = float(turnover_frame["budget_model_value_arbitration_signal"].mean()) if "budget_model_value_arbitration_signal" in turnover_frame.columns else 0.0
        metrics["avg_budget_model_alpha_opportunity_signal"] = float(turnover_frame["budget_model_alpha_opportunity_signal"].mean()) if "budget_model_alpha_opportunity_signal" in turnover_frame.columns else 0.0
        metrics["avg_budget_model_cash_defense_signal"] = float(turnover_frame["budget_model_cash_defense_signal"].mean()) if "budget_model_cash_defense_signal" in turnover_frame.columns else 0.0
        metrics["avg_budget_model_deploy_value_signal"] = float(turnover_frame["budget_model_deploy_value_signal"].mean()) if "budget_model_deploy_value_signal" in turnover_frame.columns else 0.0
        metrics["avg_budget_model_release_value_signal"] = float(turnover_frame["budget_model_release_value_signal"].mean()) if "budget_model_release_value_signal" in turnover_frame.columns else 0.0
        metrics["avg_budget_model_defense_value_signal"] = float(turnover_frame["budget_model_defense_value_signal"].mean()) if "budget_model_defense_value_signal" in turnover_frame.columns else 0.0
        metrics["avg_budget_model_deploy_gate_signal"] = float(turnover_frame["budget_model_deploy_gate_signal"].mean()) if "budget_model_deploy_gate_signal" in turnover_frame.columns else 0.0
        metrics["avg_budget_model_release_gate_signal"] = float(turnover_frame["budget_model_release_gate_signal"].mean()) if "budget_model_release_gate_signal" in turnover_frame.columns else 0.0
        metrics["avg_budget_model_defense_gate_signal"] = float(turnover_frame["budget_model_defense_gate_signal"].mean()) if "budget_model_defense_gate_signal" in turnover_frame.columns else 0.0
        metrics["avg_turnover_value_arbitration_target"] = float(turnover_frame["avg_value_arbitration_target"].mean()) if "avg_value_arbitration_target" in turnover_frame.columns else 0.0
        metrics["avg_turnover_alpha_opportunity_value"] = float(turnover_frame["avg_alpha_opportunity_value"].mean()) if "avg_alpha_opportunity_value" in turnover_frame.columns else 0.0
        metrics["avg_turnover_cash_defense_value"] = float(turnover_frame["avg_cash_defense_value"].mean()) if "avg_cash_defense_value" in turnover_frame.columns else 0.0
        metrics["avg_turnover_deploy_gate_target"] = float(turnover_frame["avg_deploy_gate_target"].mean()) if "avg_deploy_gate_target" in turnover_frame.columns else 0.0
        metrics["avg_turnover_release_gate_target"] = float(turnover_frame["avg_release_gate_target"].mean()) if "avg_release_gate_target" in turnover_frame.columns else 0.0
        metrics["avg_turnover_defense_gate_target"] = float(turnover_frame["avg_defense_gate_target"].mean()) if "avg_defense_gate_target" in turnover_frame.columns else 0.0
    else:
        metrics["avg_position_cap_target"] = 0.0
        metrics["avg_hold_bias_target"] = 0.0
        metrics["avg_reduce_bias_target"] = 0.0
        metrics["avg_exit_patience_target"] = 0.0
        metrics["avg_reentry_guard_target"] = 0.0
        metrics["avg_budget_risk_off_score"] = 0.0
        metrics["avg_budget_deploy_score"] = 0.0
        metrics["avg_budget_entry_candidate_count"] = 0.0
        metrics["avg_budget_entry_keep_count"] = 0.0
        metrics["avg_budget_held_protected_count"] = 0.0
        metrics["avg_budget_split_bound_guard_count"] = 0.0
        metrics["avg_budget_translation_floor_guard_count"] = 0.0
        metrics["avg_budget_translation_cap_guard_count"] = 0.0
        metrics["avg_budget_sell_priority_guard_count"] = 0.0
        metrics["avg_budget_model_risk_signal"] = 0.0
        metrics["avg_budget_model_deploy_signal"] = 0.0
        metrics["avg_budget_model_cash_timing_signal"] = 0.0
        metrics["avg_budget_model_alpha_focus_signal"] = 0.0
        metrics["avg_budget_model_value_arbitration_signal"] = 0.0
        metrics["avg_budget_model_alpha_opportunity_signal"] = 0.0
        metrics["avg_budget_model_cash_defense_signal"] = 0.0
        metrics["avg_budget_model_deploy_value_signal"] = 0.0
        metrics["avg_budget_model_release_value_signal"] = 0.0
        metrics["avg_budget_model_defense_value_signal"] = 0.0
        metrics["avg_budget_model_deploy_gate_signal"] = 0.0
        metrics["avg_budget_model_release_gate_signal"] = 0.0
        metrics["avg_budget_model_defense_gate_signal"] = 0.0
        metrics["avg_turnover_value_arbitration_target"] = 0.0
        metrics["avg_turnover_alpha_opportunity_value"] = 0.0
        metrics["avg_turnover_cash_defense_value"] = 0.0
        metrics["avg_turnover_deploy_gate_target"] = 0.0
        metrics["avg_turnover_release_gate_target"] = 0.0
        metrics["avg_turnover_defense_gate_target"] = 0.0

    return metrics, action_outcomes


def _safe_label_column(label_frame: pd.DataFrame, column: str, *, default: float = 0.0) -> pd.Series:
    if column not in label_frame.columns:
        return pd.Series(float(default), index=label_frame.index, dtype=float)
    return pd.to_numeric(label_frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(float(default)).astype(float)


def _first_label_scalar(label_frame: pd.DataFrame, column: str, *, default: float = 0.0) -> float:
    series = _safe_label_column(label_frame, column, default=default)
    if series.empty:
        return float(default)
    value = float(series.iloc[0])
    return value if np.isfinite(value) else float(default)


def _result_value_budget_signals(label_frame: pd.DataFrame) -> dict[str, float]:
    if label_frame.empty:
        return {
            "deploy_score": 0.0,
            "risk_score": 0.0,
            "alpha_alignment": 0.0,
            "edge_top_mean": 0.0,
            "downside_top_mean": 0.0,
            "sell_pressure": 0.0,
            "candidate_count_hint": 0.0,
            "cash_timing_score": 0.0,
            "reentry_guard_score": 0.0,
            "opportunity_concentration": 0.0,
            "risk_deploy_gap": 0.0,
            "market_downside": 0.0,
            "cash_regime": 0.0,
            "reversal_rate": 0.0,
            "forward_benchmark_downside": 0.0,
            "forward_benchmark_upside": 0.0,
            "held_sell_pressure": 0.0,
            "sell_selection_pressure": 0.0,
            "opportunity_cost_pressure": 0.0,
            "deployment_floor_pressure": 0.0,
            "alpha_opportunity_value": 0.0,
            "hold_continuation_value": 0.0,
            "sell_release_value": 0.0,
            "cash_defense_value": 0.0,
            "deployment_opportunity_cost": 0.0,
            "risk_adjusted_action_value": 0.0,
            "multi_horizon_forward_value": 0.0,
            "multi_horizon_forward_risk": 0.0,
            "multi_horizon_path_value": 0.0,
            "open_action_value": 0.0,
            "add_action_value": 0.0,
            "hold_action_value": 0.0,
            "reduce_action_value": 0.0,
            "exit_action_value": 0.0,
            "relative_opportunity_value": 0.0,
            "action_value_consistency_target": 0.5,
            "value_arbitration_target": 0.0,
            "deploy_value_target": 0.0,
            "release_value_target": 0.0,
            "defense_value_target": 0.0,
            "deploy_gate_target": 0.0,
            "release_gate_target": 0.0,
            "defense_gate_target": 0.0,
            "deploy_executability_target": 0.0,
            "large_upside_1d_target": 0.0,
            "arbitration_deploy_pressure": 0.0,
            "arbitration_sell_pressure": 0.0,
            "arbitration_cash_pressure": 0.0,
            "deploy_executability_pressure": 0.0,
        }
    priority = _safe_label_column(label_frame, "teacher_priority")
    edge = _safe_label_column(label_frame, "teacher_edge")
    opportunity = _safe_label_column(label_frame, "teacher_opportunity")
    downside = _safe_label_column(label_frame, "teacher_downside")
    alpha_score = _safe_label_column(label_frame, "alpha_prior_score_z")
    alpha_rank = _safe_label_column(label_frame, "alpha_prior_rank_pct")
    alpha_weight = _safe_label_column(label_frame, "alpha_prior_target_weight")
    alpha_selected = _safe_label_column(label_frame, "alpha_prior_selected")
    current_weight = _safe_label_column(label_frame, "current_weight")
    holding_flag = _safe_label_column(label_frame, "holding_flag")
    sell_attribution = _safe_label_column(label_frame, "sell_attribution_score")
    alpha_value = _safe_label_column(label_frame, "alpha_opportunity_value")
    hold_value = _safe_label_column(label_frame, "hold_continuation_value")
    sell_release_value = _safe_label_column(label_frame, "sell_release_value")
    cash_defense_value = _safe_label_column(label_frame, "cash_defense_value")
    deployment_cost_value = _safe_label_column(label_frame, "deployment_opportunity_cost")
    risk_adjusted_action_value = _safe_label_column(label_frame, "risk_adjusted_action_value")
    multi_horizon_forward_value = _safe_label_column(label_frame, "multi_horizon_forward_value")
    multi_horizon_forward_risk = _safe_label_column(label_frame, "multi_horizon_forward_risk")
    multi_horizon_path_value = _safe_label_column(label_frame, "multi_horizon_path_value")
    open_action_value = _safe_label_column(label_frame, "open_action_value")
    add_action_value = _safe_label_column(label_frame, "add_action_value")
    hold_action_value = _safe_label_column(label_frame, "hold_action_value")
    reduce_action_value = _safe_label_column(label_frame, "reduce_action_value")
    exit_action_value = _safe_label_column(label_frame, "exit_action_value")
    relative_opportunity_value = _safe_label_column(label_frame, "relative_opportunity_value")
    action_value_consistency_target = _safe_label_column(label_frame, "action_value_consistency_target", default=0.5)
    value_arbitration_target = _safe_label_column(label_frame, "value_arbitration_target", default=0.5)
    fallback_deploy_value = pd.Series(
        np.maximum(
            np.clip(0.56 * deployment_cost_value + 0.34 * alpha_value + 0.10 * _safe_label_column(label_frame, "large_upside_1d_target"), 0.0, 1.0),
            np.clip(0.62 * hold_value + 0.24 * alpha_value + 0.14 * _safe_label_column(label_frame, "large_upside_1d_target"), 0.0, 1.0),
        ),
        index=label_frame.index,
    )
    fallback_release_value = pd.Series(
        np.clip(
            0.62 * sell_release_value + 0.24 * cash_defense_value + 0.14 * _safe_label_column(label_frame, "lifecycle_sell_gate"),
            0.0,
            1.0,
        ),
        index=label_frame.index,
    )
    fallback_defense_value = pd.Series(
        np.clip(
            0.68 * cash_defense_value + 0.20 * (1.0 - alpha_value.clip(0.0, 1.0)) + 0.12 * _safe_label_column(label_frame, "clipped_intent_risk"),
            0.0,
            1.0,
        ),
        index=label_frame.index,
    )
    deploy_value = (
        _safe_label_column(label_frame, "deploy_value_target", default=0.0).clip(0.0, 1.0)
        if "deploy_value_target" in label_frame.columns
        else fallback_deploy_value.clip(0.0, 1.0)
    )
    release_value = (
        _safe_label_column(label_frame, "release_value_target", default=0.0).clip(0.0, 1.0)
        if "release_value_target" in label_frame.columns
        else fallback_release_value.clip(0.0, 1.0)
    )
    defense_value = (
        _safe_label_column(label_frame, "defense_value_target", default=0.0).clip(0.0, 1.0)
        if "defense_value_target" in label_frame.columns
        else fallback_defense_value.clip(0.0, 1.0)
    )
    gate_denominator = deploy_value + release_value + defense_value + 1.0e-6
    deploy_gate = (
        _safe_label_column(label_frame, "deploy_gate_target", default=0.0).clip(0.0, 1.0)
        if "deploy_gate_target" in label_frame.columns
        else (deploy_value / gate_denominator).clip(0.0, 1.0)
    )
    deploy_executability = (
        _safe_label_column(label_frame, "deploy_executability_target", default=0.0).clip(0.0, 1.0)
        if "deploy_executability_target" in label_frame.columns
        else pd.Series(0.0, index=label_frame.index)
    )
    release_gate = (
        _safe_label_column(label_frame, "release_gate_target", default=0.0).clip(0.0, 1.0)
        if "release_gate_target" in label_frame.columns
        else (release_value / gate_denominator).clip(0.0, 1.0)
    )
    defense_gate = (
        _safe_label_column(label_frame, "defense_gate_target", default=0.0).clip(0.0, 1.0)
        if "defense_gate_target" in label_frame.columns
        else (defense_value / gate_denominator).clip(0.0, 1.0)
    )
    large_upside_target = _safe_label_column(label_frame, "large_upside_1d_target")
    actions = label_frame.get("action_label", pd.Series("", index=label_frame.index)).astype(str)

    positive_mask = actions.isin({"open", "add", "hold"})
    sell_mask = actions.isin({"reduce", "exit"})
    top_count = max(1, min(8, int(positive_mask.sum()) or len(label_frame)))
    top_index = priority.nlargest(top_count).index if len(priority) else label_frame.index[:0]
    edge_top_mean = float(edge.reindex(top_index).mean()) if len(top_index) else 0.0
    opportunity_top_mean = float(opportunity.reindex(top_index).mean()) if len(top_index) else 0.0
    downside_top_mean = float(downside.reindex(top_index).mean()) if len(top_index) else 0.0
    alpha_selected_top = float(alpha_selected.reindex(top_index).clip(0.0, 1.0).mean()) if len(top_index) else 0.0
    alpha_rank_top = float(alpha_rank.reindex(top_index).clip(0.0, 1.0).mean()) if len(top_index) else 0.0
    alpha_score_top = float(np.clip(alpha_score.reindex(top_index).mean() / 2.0, -1.0, 1.0)) if len(top_index) else 0.0
    alpha_weight_sum = float(alpha_weight.clip(lower=0.0).sum())
    alpha_alignment = float(
        np.clip(
            0.42 * alpha_selected_top
            + 0.28 * alpha_rank_top
            + 0.20 * max(alpha_score_top, 0.0)
            + 0.10 * np.clip(alpha_weight_sum / 0.45, 0.0, 1.0),
            0.0,
            1.0,
        )
    )

    market_downside = _first_label_scalar(label_frame, "market_downside_pressure")
    cash_regime = _first_label_scalar(label_frame, "cash_regime_pressure")
    portfolio_cash_pressure = _first_label_scalar(label_frame, "portfolio_cash_pressure")
    reversal_rate = _first_label_scalar(label_frame, "recent_reversal_rate_20d")
    forward_benchmark_1d = _first_label_scalar(label_frame, "forward_benchmark_return_1d")
    forward_benchmark_3d = _first_label_scalar(label_frame, "forward_benchmark_return_3d")
    edge_score = float(np.clip(edge_top_mean / 0.055, -1.0, 1.0))
    opportunity_score = float(np.clip(opportunity_top_mean / 0.10, 0.0, 1.0))
    downside_score = float(np.clip(downside_top_mean / 0.08, 0.0, 1.0))
    forward_benchmark_downside = float(
        np.clip(
            max(-forward_benchmark_1d, 0.0) / 0.012 * 0.44
            + max(-forward_benchmark_3d, 0.0) / 0.028 * 0.56,
            0.0,
            1.0,
        )
    )
    forward_benchmark_upside = float(
        np.clip(
            max(forward_benchmark_1d, 0.0) / 0.012 * 0.40
            + max(forward_benchmark_3d, 0.0) / 0.028 * 0.44
            + max(edge_score, 0.0) * 0.16,
            0.0,
            1.0,
        )
    )
    held_mask = (holding_flag > 0.5) | (current_weight > 1.0e-8)
    held_weight = current_weight.where(held_mask, 0.0).clip(lower=0.0)
    if float(held_weight.sum()) <= 1.0e-8 and bool(held_mask.any()):
        held_weight = held_mask.astype(float)
    held_weight_total = float(held_weight.sum())
    if held_weight_total > 1.0e-8:
        held_weight = held_weight / held_weight_total
        held_sell_pressure = float((sell_attribution.where(held_mask, 0.0).clip(0.0, 1.0) * held_weight).sum())
        held_edge_penalty = float((np.clip(-edge, 0.0, None).where(held_mask, 0.0) * held_weight).sum())
        held_alpha_support = float((alpha_selected.where(held_mask, 0.0).clip(0.0, 1.0) * held_weight).sum())
        held_opportunity_support = float((opportunity.where(held_mask, 0.0).clip(lower=0.0) * held_weight).sum())
        sell_selection_pressure = float(
            np.clip(
                held_sell_pressure * 0.46
                + np.clip(held_edge_penalty / 0.045, 0.0, 1.0) * 0.30
                + forward_benchmark_downside * 0.16
                + market_downside * 0.10
                - held_alpha_support * 0.08
                - np.clip(held_opportunity_support / 0.08, 0.0, 1.0) * 0.08,
                0.0,
                1.0,
            )
        )
    else:
        held_sell_pressure = 0.0
        sell_selection_pressure = 0.0
    risk_score = float(
        np.clip(
            0.34 * np.clip(market_downside / 0.24, 0.0, 1.0)
            + 0.24 * np.clip(cash_regime / 0.24, 0.0, 1.0)
            + 0.18 * np.clip(portfolio_cash_pressure / 0.22, 0.0, 1.0)
            + 0.14 * downside_score
            + 0.10 * np.clip(reversal_rate / 0.28, 0.0, 1.0)
            + 0.08 * forward_benchmark_downside,
            0.0,
            1.0,
        )
    )
    risk_score = float(
        np.clip(
            risk_score
            + held_sell_pressure * 0.08
            + sell_selection_pressure * 0.10,
            0.0,
            1.0,
        )
    )
    deploy_score = float(
        np.clip(
            0.22
            + 0.30 * max(edge_score, 0.0)
            + 0.22 * opportunity_score
            + 0.24 * alpha_alignment
            - 0.34 * risk_score
            + 0.08 * max(edge_score, -0.4),
            0.0,
            1.0,
        )
    )
    sell_pressure = float(
        np.clip(
            sell_mask.mean() * 0.32
            + downside_score * 0.28
            + risk_score * 0.26
            - max(edge_score, 0.0) * 0.16
            - alpha_alignment * 0.10,
            0.0,
            1.0,
        )
    )
    candidate_count_hint = float(
        np.clip(
            positive_mask.sum() * 0.45
            + (alpha_selected.clip(0.0, 1.0).sum() * 0.40)
            + deploy_score * 4.0
            - risk_score * 3.0,
            2.0,
            14.0,
        )
    )
    opportunity_concentration = float(
        np.clip(
            max(edge_top_mean, 0.0) / 0.06 * 0.42
            + max(opportunity_top_mean, 0.0) / 0.10 * 0.34
            + alpha_alignment * 0.24,
            0.0,
            1.0,
        )
    )
    opportunity_cost_pressure = float(
        np.clip(
            forward_benchmark_upside * 0.30
            + opportunity_concentration * 0.28
            + alpha_alignment * 0.24
            + max(edge_score, 0.0) * 0.16
            - forward_benchmark_downside * 0.18
            - risk_score * 0.18
            - sell_selection_pressure * 0.10,
            0.0,
            1.0,
        )
    )
    deployment_floor_pressure = float(
        np.clip(
            0.16
            + opportunity_cost_pressure * 0.34
            + alpha_alignment * 0.22
            + opportunity_score * 0.18
            + max(edge_score, 0.0) * 0.12
            - risk_score * 0.20
            - forward_benchmark_downside * 0.12,
            0.0,
            1.0,
        )
    )
    alpha_value_top = float(alpha_value.reindex(top_index).clip(0.0, 1.0).mean()) if len(top_index) else 0.0
    deployment_cost_top = float(deployment_cost_value.reindex(top_index).clip(0.0, 1.0).mean()) if len(top_index) else 0.0
    large_upside_top = float(large_upside_target.reindex(top_index).clip(0.0, 1.0).mean()) if len(top_index) else 0.0
    deploy_value_top = float(deploy_value.reindex(top_index).clip(0.0, 1.0).mean()) if len(top_index) else 0.0
    deploy_gate_top = float(deploy_gate.reindex(top_index).clip(0.0, 1.0).mean()) if len(top_index) else 0.0
    deploy_executability_top = (
        float(deploy_executability.reindex(top_index).clip(0.0, 1.0).mean()) if len(top_index) else 0.0
    )
    open_action_value_top = float(open_action_value.reindex(top_index).clip(0.0, 1.0).mean()) if len(top_index) else 0.0
    multi_horizon_forward_value_top = (
        float(multi_horizon_forward_value.reindex(top_index).clip(0.0, 1.0).mean()) if len(top_index) else 0.0
    )
    multi_horizon_path_value_top = (
        float(multi_horizon_path_value.reindex(top_index).clip(0.0, 1.0).mean()) if len(top_index) else 0.0
    )
    multi_horizon_forward_risk_mean = (
        float(multi_horizon_forward_risk.clip(0.0, 1.0).mean()) if len(multi_horizon_forward_risk) else 0.0
    )
    value_arbitration_mean = float(value_arbitration_target.clip(0.0, 1.0).mean()) if len(value_arbitration_target) else 0.0
    risk_adjusted_action_mean = float(risk_adjusted_action_value.clip(0.0, 1.0).mean()) if len(risk_adjusted_action_value) else 0.0
    cash_defense_mean = float(cash_defense_value.clip(0.0, 1.0).mean()) if len(cash_defense_value) else 0.0
    defense_value_mean = float(defense_value.clip(0.0, 1.0).mean()) if len(defense_value) else 0.0
    defense_gate_mean = float(defense_gate.clip(0.0, 1.0).mean()) if len(defense_gate) else 0.0
    hold_value_mean = float(hold_value.clip(0.0, 1.0).where(held_mask, 0.0).sum() / max(float(held_mask.sum()), 1.0))
    add_action_value_mean = float(add_action_value.clip(0.0, 1.0).where(held_mask, 0.0).sum() / max(float(held_mask.sum()), 1.0))
    hold_action_value_mean = float(hold_action_value.clip(0.0, 1.0).where(held_mask, 0.0).sum() / max(float(held_mask.sum()), 1.0))
    reduce_action_value_mean = float(reduce_action_value.clip(0.0, 1.0).where(held_mask, 0.0).sum() / max(float(held_mask.sum()), 1.0))
    exit_action_value_mean = float(exit_action_value.clip(0.0, 1.0).where(held_mask, 0.0).sum() / max(float(held_mask.sum()), 1.0))
    action_value_consistency_mean = (
        float(action_value_consistency_target.clip(0.0, 1.0).mean()) if len(action_value_consistency_target) else 0.5
    )
    if held_weight_total > 1.0e-8:
        sell_release_mean = float((sell_release_value.clip(0.0, 1.0).where(held_mask, 0.0) * held_weight).sum())
        release_value_mean = float((release_value.clip(0.0, 1.0).where(held_mask, 0.0) * held_weight).sum())
        release_gate_mean = float((release_gate.clip(0.0, 1.0).where(held_mask, 0.0) * held_weight).sum())
    else:
        sell_release_mean = float(sell_release_value.clip(0.0, 1.0).mean()) if len(sell_release_value) else 0.0
        release_value_mean = float(release_value.clip(0.0, 1.0).mean()) if len(release_value) else 0.0
        release_gate_mean = float(release_gate.clip(0.0, 1.0).mean()) if len(release_gate) else 0.0
    arbitration_deploy_pressure = float(
        np.clip(
            0.28 * deploy_value_top
            + 0.22 * deploy_gate_top
            + 0.20 * alpha_value_top
            + 0.16 * deployment_cost_top
            + 0.08 * large_upside_top
            + 0.08 * opportunity_cost_pressure
            - 0.18 * defense_gate_mean
            - 0.10 * release_gate_mean,
            0.0,
            1.0,
        )
    )
    arbitration_sell_pressure = float(
        np.clip(
            0.34 * release_value_mean
            + 0.24 * release_gate_mean
            + 0.16 * sell_release_mean
            + 0.12 * sell_selection_pressure
            + 0.12 * held_sell_pressure
            + 0.06 * defense_gate_mean
            - 0.14 * deploy_gate_top
            - 0.08 * hold_value_mean,
            0.0,
            1.0,
        )
    )
    arbitration_cash_pressure = float(
        np.clip(
            0.34 * defense_value_mean
            + 0.26 * defense_gate_mean
            + 0.16 * cash_defense_mean
            + 0.14 * forward_benchmark_downside
            + 0.08 * release_gate_mean
            + 0.08 * max(risk_score - deploy_score, 0.0)
            - 0.22 * deploy_gate_top
            - 0.10 * deploy_value_top,
            0.0,
            1.0,
        )
    )
    deploy_executability_pressure = float(
        np.clip(
            0.34 * deploy_executability_top
            + 0.22 * deploy_gate_top
            + 0.18 * deploy_value_top
            + 0.14 * deployment_cost_top
            + 0.08 * large_upside_top
            + 0.06 * alpha_alignment
            - 0.16 * release_gate_mean
            - 0.14 * defense_gate_mean,
            0.0,
            1.0,
        )
    )
    cash_timing_score = float(
        np.clip(
            risk_score * 0.54
            + max(risk_score - deploy_score, 0.0) * 0.18
            + downside_score * 0.18
            + np.clip(cash_regime / 0.26, 0.0, 1.0) * 0.16
            + np.clip(reversal_rate / 0.28, 0.0, 1.0) * 0.10
            + forward_benchmark_downside * 0.12
            + held_sell_pressure * 0.10
            + sell_selection_pressure * 0.12
            - deploy_score * 0.24
            - alpha_alignment * 0.08
            - opportunity_score * 0.06,
            0.0,
            1.0,
        )
    )
    reentry_guard_score = float(
        np.clip(
            np.clip(reversal_rate / 0.28, 0.0, 1.0) * 0.42
            + np.clip(cash_regime / 0.24, 0.0, 1.0) * 0.22
            + sell_pressure * 0.18
            + cash_timing_score * 0.18
            - alpha_alignment * 0.10,
            0.0,
            1.0,
        )
    )
    risk_deploy_gap = float(np.clip(risk_score - deploy_score, -1.0, 1.0))
    return {
        "deploy_score": deploy_score,
        "risk_score": risk_score,
        "alpha_alignment": alpha_alignment,
        "edge_top_mean": edge_top_mean,
        "downside_top_mean": downside_top_mean,
        "sell_pressure": sell_pressure,
        "candidate_count_hint": candidate_count_hint,
        "cash_timing_score": cash_timing_score,
        "reentry_guard_score": reentry_guard_score,
        "opportunity_concentration": opportunity_concentration,
        "risk_deploy_gap": risk_deploy_gap,
        "market_downside": market_downside,
        "cash_regime": cash_regime,
        "reversal_rate": reversal_rate,
        "forward_benchmark_downside": forward_benchmark_downside,
        "forward_benchmark_upside": forward_benchmark_upside,
        "held_sell_pressure": held_sell_pressure,
        "sell_selection_pressure": sell_selection_pressure,
        "opportunity_cost_pressure": opportunity_cost_pressure,
        "deployment_floor_pressure": deployment_floor_pressure,
        "alpha_opportunity_value": alpha_value_top,
        "hold_continuation_value": hold_value_mean,
        "sell_release_value": sell_release_mean,
        "cash_defense_value": cash_defense_mean,
        "deployment_opportunity_cost": deployment_cost_top,
        "risk_adjusted_action_value": risk_adjusted_action_mean,
        "multi_horizon_forward_value": multi_horizon_forward_value_top,
        "multi_horizon_forward_risk": multi_horizon_forward_risk_mean,
        "multi_horizon_path_value": multi_horizon_path_value_top,
        "open_action_value": open_action_value_top,
        "add_action_value": add_action_value_mean,
        "hold_action_value": hold_action_value_mean,
        "reduce_action_value": reduce_action_value_mean,
        "exit_action_value": exit_action_value_mean,
        "relative_opportunity_value": (
            float(relative_opportunity_value.clip(0.0, 1.0).mean()) if len(relative_opportunity_value) else 0.0
        ),
        "action_value_consistency_target": action_value_consistency_mean,
        "value_arbitration_target": value_arbitration_mean,
        "deploy_value_target": deploy_value_top,
        "release_value_target": release_value_mean,
        "defense_value_target": defense_value_mean,
        "deploy_gate_target": deploy_gate_top,
        "release_gate_target": release_gate_mean,
        "defense_gate_target": defense_gate_mean,
        "deploy_executability_target": deploy_executability_top,
        "large_upside_1d_target": large_upside_top,
        "arbitration_deploy_pressure": arbitration_deploy_pressure,
        "arbitration_sell_pressure": arbitration_sell_pressure,
        "arbitration_cash_pressure": arbitration_cash_pressure,
        "deploy_executability_pressure": deploy_executability_pressure,
    }


def _apply_budget_objective_targets(
    *,
    label_frame: pd.DataFrame,
    global_targets: dict[str, float],
    budget_objective: str,
) -> tuple[dict[str, float], dict[str, float]]:
    objective = resolve_budget_objective(budget_objective)
    signals = _result_value_budget_signals(label_frame)
    diagnostics = {f"result_value_{key}": float(value) for key, value in signals.items()}
    diagnostics["budget_objective_is_result_value"] = 1.0 if objective != DEFAULT_BUDGET_OBJECTIVE else 0.0
    diagnostics["budget_objective_is_result_value_v2"] = 1.0 if objective == BUDGET_OBJECTIVE_RESULT_VALUE_V2 else 0.0
    diagnostics["budget_objective_is_result_value_v3"] = 1.0 if objective == BUDGET_OBJECTIVE_RESULT_VALUE_V3 else 0.0
    diagnostics["budget_objective_is_result_value_v4"] = 1.0 if objective == BUDGET_OBJECTIVE_RESULT_VALUE_V4 else 0.0
    diagnostics["budget_objective_is_result_value_v4b"] = 1.0 if objective == BUDGET_OBJECTIVE_RESULT_VALUE_V4B else 0.0
    diagnostics["budget_objective_is_result_value_v5"] = 1.0 if objective == BUDGET_OBJECTIVE_RESULT_VALUE_V5 else 0.0
    diagnostics["budget_objective_is_result_value_v6"] = 1.0 if objective == BUDGET_OBJECTIVE_RESULT_VALUE_V6 else 0.0
    diagnostics["budget_objective_is_result_value_v7"] = 1.0 if objective == BUDGET_OBJECTIVE_RESULT_VALUE_V7 else 0.0
    diagnostics["budget_objective_is_result_value_v8"] = 1.0 if objective == BUDGET_OBJECTIVE_RESULT_VALUE_V8 else 0.0
    diagnostics["budget_objective_is_result_value_v9"] = 1.0 if objective == BUDGET_OBJECTIVE_RESULT_VALUE_V9 else 0.0
    diagnostics["budget_objective_is_result_value_v10"] = 1.0 if objective == BUDGET_OBJECTIVE_RESULT_VALUE_V10 else 0.0

    adjusted = dict(global_targets)
    adjusted["budget_risk_signal_target"] = float(np.clip(signals["risk_score"], 0.0, 1.0))
    adjusted["budget_deploy_signal_target"] = float(np.clip(signals["deploy_score"], 0.0, 1.0))
    adjusted["budget_cash_timing_signal_target"] = float(np.clip(signals["cash_timing_score"], 0.0, 1.0))
    adjusted["budget_alpha_focus_signal_target"] = float(np.clip(signals["alpha_alignment"], 0.0, 1.0))
    if objective == DEFAULT_BUDGET_OBJECTIVE:
        return adjusted, diagnostics

    deploy = float(signals["deploy_score"])
    risk = float(signals["risk_score"])
    alpha_alignment = float(signals["alpha_alignment"])
    sell_pressure = float(signals["sell_pressure"])
    candidate_hint = float(signals["candidate_count_hint"])
    edge_top = float(signals["edge_top_mean"])
    base_gross = float(adjusted.get("gross_exposure_target", 0.0) or 0.0)
    if objective == BUDGET_OBJECTIVE_RESULT_VALUE_V1:
        adjusted["gross_exposure_target"] = float(
            np.clip(
                base_gross
                + 0.16 * (deploy - 0.42)
                + 0.05 * alpha_alignment
                - 0.18 * max(risk - deploy, 0.0)
                - 0.04 * sell_pressure,
                0.12,
                0.94,
            )
        )
        adjusted["candidate_budget"] = float(
            int(
                np.clip(
                    round(
                        0.55 * float(adjusted.get("candidate_budget", 2.0) or 2.0)
                        + 0.45 * candidate_hint
                        + 1.5 * max(deploy - risk, 0.0)
                        - 1.0 * max(risk - deploy, 0.0)
                    ),
                    2,
                    14,
                )
            )
        )
        adjusted["turnover_budget"] = float(
            np.clip(
                float(adjusted.get("turnover_budget", 0.10) or 0.10)
                + 0.10 * sell_pressure
                + 0.04 * max(deploy - 0.45, 0.0)
                + 0.03 * max(risk - deploy, 0.0)
                - 0.03 * alpha_alignment * max(deploy - risk, 0.0),
                0.08,
                0.82,
            )
        )
        adjusted["max_position_weight_target"] = float(
            np.clip(
                float(adjusted.get("max_position_weight_target", 0.10) or 0.10)
                + 0.024 * alpha_alignment
                + 0.014 * max(deploy - risk, 0.0)
                - 0.018 * sell_pressure,
                0.07,
                0.28,
            )
        )
        adjusted["hold_bias_target"] = float(
            np.clip(
                float(adjusted.get("hold_bias_target", 0.18) or 0.18)
                + 0.16 * max(edge_top / 0.055, 0.0)
                + 0.12 * alpha_alignment
                - 0.20 * sell_pressure
                - 0.08 * max(risk - deploy, 0.0),
                0.10,
                0.95,
            )
        )
        adjusted["reduce_bias_target"] = float(
            np.clip(
                float(adjusted.get("reduce_bias_target", 0.10) or 0.10)
                + 0.12 * sell_pressure
                + 0.08 * max(risk - deploy, 0.0)
                - 0.04 * alpha_alignment,
                0.0,
                0.65,
            )
        )
        adjusted["exit_patience_target"] = float(
            np.clip(
                float(adjusted.get("exit_patience_target", 0.20) or 0.20)
                + 0.10 * alpha_alignment
                + 0.08 * max(edge_top / 0.055, 0.0)
                - 0.18 * sell_pressure
                - 0.08 * max(risk - deploy, 0.0),
                0.05,
                0.95,
            )
        )
        adjusted["reentry_guard_target"] = float(
            np.clip(
                float(adjusted.get("reentry_guard_target", 0.0) or 0.0)
                + 0.08 * max(risk - deploy, 0.0)
                + 0.12 * signals["reentry_guard_score"]
                - 0.04 * alpha_alignment,
                0.0,
                0.45,
            )
        )
        return adjusted, diagnostics

    cash_timing = float(signals["cash_timing_score"])
    reentry_guard = float(signals["reentry_guard_score"])
    opportunity_concentration = float(signals["opportunity_concentration"])
    risk_deploy_gap = float(signals["risk_deploy_gap"])
    sell_selection_pressure = float(signals["sell_selection_pressure"])
    held_sell_pressure = float(signals["held_sell_pressure"])
    forward_benchmark_downside = float(signals["forward_benchmark_downside"])
    opportunity_cost = float(signals["opportunity_cost_pressure"])
    deployment_floor = float(signals["deployment_floor_pressure"])
    alpha_opportunity_value = float(signals["alpha_opportunity_value"])
    hold_continuation_value = float(signals["hold_continuation_value"])
    sell_release_value = float(signals["sell_release_value"])
    cash_defense_value = float(signals["cash_defense_value"])
    deployment_opportunity_cost = float(signals["deployment_opportunity_cost"])
    value_arbitration_target = float(signals["value_arbitration_target"])
    deploy_value_target = float(signals["deploy_value_target"])
    release_value_target = float(signals["release_value_target"])
    defense_value_target = float(signals["defense_value_target"])
    deploy_gate_target = float(signals["deploy_gate_target"])
    release_gate_target = float(signals["release_gate_target"])
    defense_gate_target = float(signals["defense_gate_target"])
    deploy_executability_target = float(signals["deploy_executability_target"])
    deploy_executability_pressure = float(signals["deploy_executability_pressure"])
    large_upside_1d_target = float(signals["large_upside_1d_target"])
    multi_horizon_path_value = float(signals["multi_horizon_path_value"])
    multi_horizon_forward_risk = float(signals["multi_horizon_forward_risk"])
    open_action_value = float(signals["open_action_value"])
    add_action_value = float(signals["add_action_value"])
    hold_action_value = float(signals["hold_action_value"])
    reduce_action_value = float(signals["reduce_action_value"])
    exit_action_value = float(signals["exit_action_value"])
    action_value_consistency_target = float(signals["action_value_consistency_target"])
    arbitration_deploy_pressure = float(signals["arbitration_deploy_pressure"])
    arbitration_sell_pressure = float(signals["arbitration_sell_pressure"])
    arbitration_cash_pressure = float(signals["arbitration_cash_pressure"])
    if objective == BUDGET_OBJECTIVE_RESULT_VALUE_V10:
        release_dominance = max(release_gate_target - deploy_gate_target, 0.0)
        deploy_dominance = max(deploy_gate_target - release_gate_target, 0.0)
        deploy_release_spread = deploy_value_target - release_value_target
        executable_deploy = float(
            np.clip(
                0.40 * deploy_executability_pressure
                + 0.22 * deploy_executability_target
                + 0.18 * deploy_gate_target
                + 0.14 * deploy_value_target
                + 0.10 * alpha_opportunity_value
                + 0.08 * max(open_action_value, add_action_value)
                + 0.06 * multi_horizon_path_value
                - 0.14 * release_gate_target
                - 0.10 * defense_gate_target,
                0.0,
                1.0,
            )
        )
        protected_hold_pressure = float(
            np.clip(
                0.34 * hold_continuation_value
                + 0.18 * hold_action_value
                + 0.18 * alpha_opportunity_value
                + 0.14 * deploy_value_target
                + 0.12 * deploy_gate_target
                + 0.12 * deploy_executability_target
                + 0.10 * executable_deploy
                - 0.20 * release_value_target
                - 0.14 * max(reduce_action_value, exit_action_value)
                - 0.20 * release_gate_target
                - 0.14 * sell_release_value
                - 0.08 * cash_defense_value,
                0.0,
                1.0,
            )
        )
        funding_release_pressure = float(
            np.clip(
                0.32 * release_value_target
                + 0.24 * release_gate_target
                + 0.18 * max(reduce_action_value, exit_action_value)
                + 0.14 * held_sell_pressure
                + 0.10 * sell_selection_pressure
                + 0.08 * forward_benchmark_downside
                + 0.06 * multi_horizon_forward_risk
                + 0.06 * cash_defense_value
                - 0.22 * hold_continuation_value
                - 0.14 * max(add_action_value, hold_action_value)
                - 0.16 * deploy_executability_target
                - 0.12 * alpha_opportunity_value
                - 0.10 * deploy_gate_target,
                0.0,
                1.0,
            )
        )
        hierarchical_defense_pressure = float(
            np.clip(
                0.34 * cash_timing
                + 0.20 * cash_defense_value
                + 0.14 * risk
                + 0.10 * forward_benchmark_downside
                + 0.08 * max(risk_deploy_gap, 0.0)
                + 0.08 * defense_value_target
                + 0.06 * funding_release_pressure
                - 0.16 * executable_deploy
                - 0.10 * deploy_value_target,
                0.0,
                1.0,
            )
        )
        disciplined_funding_need = float(
            np.clip(
                0.40 * executable_deploy
                + 0.20 * deployment_floor
                + 0.14 * deploy_dominance
                + 0.10 * deploy_value_target
                + 0.08 * alpha_alignment
                - 0.22 * protected_hold_pressure
                - 0.10 * hierarchical_defense_pressure,
                0.0,
                1.0,
            )
        )
        portfolio_release_pressure = float(
            np.clip(
                0.38 * funding_release_pressure
                + 0.22 * max(release_dominance - protected_hold_pressure * 0.50, 0.0)
                + 0.14 * held_sell_pressure
                + 0.10 * sell_selection_pressure
                + 0.08 * forward_benchmark_downside
                - 0.18 * protected_hold_pressure
                - 0.06 * executable_deploy,
                0.0,
                1.0,
            )
        )
        sharpened_risk = float(
            np.clip(
                0.30 * risk
                + 0.28 * hierarchical_defense_pressure
                + 0.14 * portfolio_release_pressure
                + 0.10 * forward_benchmark_downside
                + 0.08 * max(risk_deploy_gap, 0.0)
                - 0.12 * disciplined_funding_need
                - 0.08 * deploy_value_target,
                0.0,
                1.0,
            )
        )
        sharpened_deploy = float(
            np.clip(
                0.30 * deploy
                + 0.26 * deploy_value_target
                + 0.20 * deploy_gate_target
                + 0.18 * executable_deploy
                + 0.10 * alpha_opportunity_value
                + 0.08 * disciplined_funding_need
                + 0.06 * deployment_opportunity_cost
                + 0.04 * large_upside_1d_target
                - 0.10 * hierarchical_defense_pressure
                - 0.10 * portfolio_release_pressure,
                0.0,
                1.0,
            )
        )
        sharpened_cash = float(
            np.clip(
                0.44 * cash_timing
                + 0.28 * hierarchical_defense_pressure
                + 0.10 * cash_defense_value
                + 0.08 * forward_benchmark_downside
                + 0.06 * max(risk_deploy_gap, 0.0)
                - 0.14 * disciplined_funding_need
                - 0.08 * deploy_value_target,
                0.0,
                1.0,
            )
        )
        min_gross = float(
            np.clip(
                0.22
                + 0.28 * deployment_floor
                + 0.18 * disciplined_funding_need
                + 0.14 * executable_deploy
                + 0.10 * deploy_dominance
                - 0.18 * hierarchical_defense_pressure
                - 0.08 * portfolio_release_pressure,
                0.18,
                0.78,
            )
        )
        min_candidate = int(
            np.clip(
                round(
                    2.0
                    + disciplined_funding_need * 4.4
                    + executable_deploy * 2.2
                    + deploy_gate_target * 1.8
                    + large_upside_1d_target * 1.2
                    - hierarchical_defense_pressure * 1.4
                ),
                2,
                10,
            )
        )
        diagnostics["result_value_hierarchical_defense_pressure"] = hierarchical_defense_pressure
        diagnostics["result_value_portfolio_release_pressure"] = portfolio_release_pressure
        diagnostics["result_value_executable_deploy_pressure"] = executable_deploy
        diagnostics["result_value_protected_hold_pressure"] = protected_hold_pressure
        diagnostics["result_value_funding_release_pressure"] = funding_release_pressure
        diagnostics["result_value_disciplined_funding_need"] = disciplined_funding_need
        diagnostics["result_value_action_value_consistency_target"] = action_value_consistency_target
        diagnostics["result_value_sell_source_contract_mode"] = 1.0
        adjusted["budget_risk_signal_target"] = sharpened_risk
        adjusted["budget_deploy_signal_target"] = sharpened_deploy
        adjusted["budget_cash_timing_signal_target"] = sharpened_cash
        adjusted["budget_alpha_focus_signal_target"] = float(
            np.clip(
                0.30 * alpha_alignment
                + 0.24 * deploy_value_target
                + 0.20 * deploy_gate_target
                + 0.14 * executable_deploy
                + 0.08 * disciplined_funding_need
                + 0.04 * large_upside_1d_target,
                0.0,
                1.0,
            )
        )
        adjusted["gross_exposure_target"] = float(
            np.clip(
                base_gross
                + 0.14 * (sharpened_deploy - 0.30)
                + 0.10 * disciplined_funding_need
                + 0.06 * deploy_dominance
                + 0.04 * max(deploy_release_spread, 0.0)
                - 0.18 * hierarchical_defense_pressure
                - 0.08 * portfolio_release_pressure,
                min_gross,
                0.96,
            )
        )
        adjusted["candidate_budget"] = float(
            int(
                np.clip(
                    round(
                        0.32 * float(adjusted.get("candidate_budget", 2.0) or 2.0)
                        + 0.24 * candidate_hint
                        + 2.8 * disciplined_funding_need
                        + 1.8 * executable_deploy
                        + 0.8 * alpha_opportunity_value
                        - 0.8 * hierarchical_defense_pressure
                        - 0.4 * portfolio_release_pressure
                    ),
                    min_candidate,
                    14,
                )
            )
        )
        adjusted["turnover_budget"] = float(
            np.clip(
                float(adjusted.get("turnover_budget", 0.10) or 0.10)
                + 0.08 * max(portfolio_release_pressure - protected_hold_pressure, 0.0)
                + 0.05 * max(hierarchical_defense_pressure - executable_deploy, 0.0)
                + 0.05 * max(disciplined_funding_need - protected_hold_pressure, 0.0),
                0.08,
                0.82,
            )
        )
        adjusted["max_position_weight_target"] = float(
            np.clip(
                float(adjusted.get("max_position_weight_target", 0.10) or 0.10)
                + 0.018 * deploy_gate_target
                + 0.016 * executable_deploy
                + 0.010 * disciplined_funding_need
                - 0.008 * hierarchical_defense_pressure
                - 0.006 * portfolio_release_pressure,
                0.07,
                0.28,
            )
        )
        adjusted["hold_bias_target"] = float(
            np.clip(
                float(adjusted.get("hold_bias_target", 0.18) or 0.18)
                + 0.18 * protected_hold_pressure
                + 0.12 * executable_deploy
                + 0.10 * hold_continuation_value
                + 0.06 * deploy_dominance
                - 0.10 * funding_release_pressure
                - 0.08 * release_dominance,
                0.10,
                0.96,
            )
        )
        adjusted["reduce_bias_target"] = float(
            np.clip(
                float(adjusted.get("reduce_bias_target", 0.10) or 0.10)
                + 0.16 * release_gate_target
                + 0.14 * funding_release_pressure
                + 0.08 * portfolio_release_pressure
                - 0.14 * protected_hold_pressure
                - 0.08 * executable_deploy
                - 0.04 * hold_continuation_value,
                0.0,
                0.62,
            )
        )
        adjusted["exit_patience_target"] = float(
            np.clip(
                float(adjusted.get("exit_patience_target", 0.20) or 0.20)
                + 0.16 * protected_hold_pressure
                + 0.10 * executable_deploy
                + 0.08 * hold_continuation_value
                - 0.12 * funding_release_pressure
                - 0.08 * release_gate_target,
                0.05,
                0.95,
            )
        )
        adjusted["reentry_guard_target"] = float(
            np.clip(
                float(adjusted.get("reentry_guard_target", 0.0) or 0.0)
                + 0.14 * hierarchical_defense_pressure
                + 0.08 * funding_release_pressure
                + 0.04 * reentry_guard
                - 0.10 * disciplined_funding_need,
                0.0,
                0.52,
            )
        )
        diagnostics["result_value_constraint_only_mode"] = 1.0
        return adjusted, diagnostics
    if objective == BUDGET_OBJECTIVE_RESULT_VALUE_V9:
        release_dominance = max(release_gate_target - deploy_gate_target, 0.0)
        deploy_dominance = max(deploy_gate_target - release_gate_target, 0.0)
        deploy_release_spread = deploy_value_target - release_value_target
        executable_deploy = float(
            np.clip(
                0.42 * deploy_executability_pressure
                + 0.24 * deploy_executability_target
                + 0.18 * deploy_gate_target
                + 0.16 * deploy_value_target
                - 0.16 * release_gate_target
                - 0.12 * defense_gate_target,
                0.0,
                1.0,
            )
        )
        portfolio_release_pressure = float(
            np.clip(
                0.40 * release_value_target
                + 0.24 * release_gate_target
                + 0.14 * held_sell_pressure
                + 0.10 * sell_selection_pressure
                + 0.08 * forward_benchmark_downside
                - 0.08 * executable_deploy,
                0.0,
                1.0,
            )
        )
        hierarchical_defense_pressure = float(
            np.clip(
                0.36 * cash_timing
                + 0.22 * cash_defense_value
                + 0.16 * risk
                + 0.10 * forward_benchmark_downside
                + 0.08 * max(risk_deploy_gap, 0.0)
                + 0.08 * defense_value_target
                - 0.14 * executable_deploy
                - 0.08 * deploy_value_target,
                0.0,
                1.0,
            )
        )
        sharpened_risk = float(
            np.clip(
                0.32 * risk
                + 0.26 * hierarchical_defense_pressure
                + 0.16 * portfolio_release_pressure
                + 0.10 * forward_benchmark_downside
                + 0.08 * max(risk_deploy_gap, 0.0)
                - 0.12 * executable_deploy
                - 0.08 * deploy_value_target,
                0.0,
                1.0,
            )
        )
        sharpened_deploy = float(
            np.clip(
                0.30 * deploy
                + 0.28 * deploy_value_target
                + 0.20 * deploy_gate_target
                + 0.20 * executable_deploy
                + 0.12 * alpha_opportunity_value
                + 0.08 * deployment_opportunity_cost
                + 0.06 * large_upside_1d_target
                - 0.12 * hierarchical_defense_pressure
                - 0.08 * portfolio_release_pressure,
                0.0,
                1.0,
            )
        )
        sharpened_cash = float(
            np.clip(
                0.42 * cash_timing
                + 0.30 * hierarchical_defense_pressure
                + 0.12 * cash_defense_value
                + 0.10 * forward_benchmark_downside
                - 0.16 * executable_deploy
                - 0.08 * deploy_value_target,
                0.0,
                1.0,
            )
        )
        min_gross = float(
            np.clip(
                0.24
                + 0.28 * deployment_floor
                + 0.22 * executable_deploy
                + 0.12 * deploy_dominance
                - 0.18 * hierarchical_defense_pressure
                - 0.08 * portfolio_release_pressure,
                0.20,
                0.80,
            )
        )
        min_candidate = int(
            np.clip(
                round(
                    2.0
                    + executable_deploy * 5.0
                    + deploy_gate_target * 2.2
                    + deploy_dominance * 1.8
                    + large_upside_1d_target * 1.4
                    - hierarchical_defense_pressure * 1.6
                ),
                2,
                10,
            )
        )
        diagnostics["result_value_hierarchical_defense_pressure"] = hierarchical_defense_pressure
        diagnostics["result_value_portfolio_release_pressure"] = portfolio_release_pressure
        diagnostics["result_value_executable_deploy_pressure"] = executable_deploy
        diagnostics["result_value_deploy_executability_mode"] = 1.0
        adjusted["budget_risk_signal_target"] = sharpened_risk
        adjusted["budget_deploy_signal_target"] = sharpened_deploy
        adjusted["budget_cash_timing_signal_target"] = sharpened_cash
        adjusted["budget_alpha_focus_signal_target"] = float(
            np.clip(
                0.32 * alpha_alignment
                + 0.24 * deploy_value_target
                + 0.20 * deploy_gate_target
                + 0.18 * executable_deploy
                + 0.06 * large_upside_1d_target,
                0.0,
                1.0,
            )
        )
        adjusted["gross_exposure_target"] = float(
            np.clip(
                base_gross
                + 0.16 * (sharpened_deploy - 0.32)
                + 0.10 * executable_deploy
                + 0.06 * deploy_dominance
                + 0.05 * max(deploy_release_spread, 0.0)
                - 0.20 * hierarchical_defense_pressure
                - 0.08 * portfolio_release_pressure,
                min_gross,
                0.97,
            )
        )
        adjusted["candidate_budget"] = float(
            int(
                np.clip(
                    round(
                        0.30 * float(adjusted.get("candidate_budget", 2.0) or 2.0)
                        + 0.26 * candidate_hint
                        + 3.2 * executable_deploy
                        + 1.8 * deploy_gate_target
                        + 0.8 * alpha_opportunity_value
                        - 1.0 * hierarchical_defense_pressure
                        - 0.5 * portfolio_release_pressure
                    ),
                    min_candidate,
                    15,
                )
            )
        )
        adjusted["turnover_budget"] = float(
            np.clip(
                float(adjusted.get("turnover_budget", 0.10) or 0.10)
                + 0.09 * portfolio_release_pressure
                + 0.05 * max(hierarchical_defense_pressure - executable_deploy, 0.0)
                + 0.07 * max(executable_deploy - hierarchical_defense_pressure, 0.0),
                0.08,
                0.88,
            )
        )
        adjusted["max_position_weight_target"] = float(
            np.clip(
                float(adjusted.get("max_position_weight_target", 0.10) or 0.10)
                + 0.020 * deploy_gate_target
                + 0.018 * executable_deploy
                + 0.010 * deploy_value_target
                - 0.010 * hierarchical_defense_pressure
                - 0.006 * portfolio_release_pressure,
                0.07,
                0.30,
            )
        )
        adjusted["hold_bias_target"] = float(
            np.clip(
                float(adjusted.get("hold_bias_target", 0.18) or 0.18)
                + 0.16 * hold_continuation_value
                + 0.14 * executable_deploy
                + 0.10 * deploy_dominance
                + 0.08 * deploy_gate_target
                - 0.09 * release_dominance
                - 0.07 * release_gate_target,
                0.10,
                0.96,
            )
        )
        adjusted["reduce_bias_target"] = float(
            np.clip(
                float(adjusted.get("reduce_bias_target", 0.10) or 0.10)
                + 0.18 * release_gate_target
                + 0.14 * release_value_target
                + 0.08 * portfolio_release_pressure
                - 0.10 * executable_deploy
                - 0.04 * hold_continuation_value,
                0.0,
                0.68,
            )
        )
        adjusted["exit_patience_target"] = float(
            np.clip(
                float(adjusted.get("exit_patience_target", 0.20) or 0.20)
                + 0.14 * executable_deploy
                + 0.10 * deploy_gate_target
                + 0.08 * hold_continuation_value
                - 0.12 * release_gate_target
                - 0.10 * portfolio_release_pressure,
                0.05,
                0.95,
            )
        )
        adjusted["reentry_guard_target"] = float(
            np.clip(
                float(adjusted.get("reentry_guard_target", 0.0) or 0.0)
                + 0.16 * hierarchical_defense_pressure
                + 0.08 * portfolio_release_pressure
                - 0.10 * executable_deploy,
                0.0,
                0.55,
            )
        )
        diagnostics["result_value_constraint_only_mode"] = 1.0
        return adjusted, diagnostics
    if objective == BUDGET_OBJECTIVE_RESULT_VALUE_V8:
        release_dominance = max(release_gate_target - deploy_gate_target, 0.0)
        deploy_dominance = max(deploy_gate_target - release_gate_target, 0.0)
        deploy_release_spread = deploy_value_target - release_value_target
        portfolio_release_pressure = float(
            np.clip(
                0.40 * release_value_target
                + 0.26 * release_gate_target
                + 0.16 * held_sell_pressure
                + 0.10 * sell_selection_pressure
                + 0.08 * forward_benchmark_downside,
                0.0,
                1.0,
            )
        )
        hierarchical_defense_pressure = float(
            np.clip(
                0.38 * cash_timing
                + 0.24 * cash_defense_value
                + 0.16 * risk
                + 0.10 * forward_benchmark_downside
                + 0.08 * max(risk_deploy_gap, 0.0)
                + 0.08 * defense_value_target
                - 0.12 * deploy_gate_target
                - 0.10 * deploy_value_target,
                0.0,
                1.0,
            )
        )
        sharpened_risk = float(
            np.clip(
                0.34 * risk
                + 0.28 * hierarchical_defense_pressure
                + 0.18 * portfolio_release_pressure
                + 0.10 * forward_benchmark_downside
                + 0.08 * max(risk_deploy_gap, 0.0)
                - 0.10 * deploy_gate_target
                - 0.08 * deploy_value_target,
                0.0,
                1.0,
            )
        )
        sharpened_deploy = float(
            np.clip(
                0.34 * deploy
                + 0.30 * deploy_value_target
                + 0.22 * deploy_gate_target
                + 0.14 * alpha_opportunity_value
                + 0.10 * deployment_opportunity_cost
                + 0.08 * large_upside_1d_target
                - 0.14 * hierarchical_defense_pressure
                - 0.10 * portfolio_release_pressure,
                0.0,
                1.0,
            )
        )
        sharpened_cash = float(
            np.clip(
                0.44 * cash_timing
                + 0.32 * hierarchical_defense_pressure
                + 0.12 * cash_defense_value
                + 0.10 * forward_benchmark_downside
                - 0.14 * deploy_gate_target
                - 0.10 * deploy_value_target,
                0.0,
                1.0,
            )
        )
        min_gross = float(
            np.clip(
                0.22
                + 0.30 * deployment_floor
                + 0.18 * deploy_gate_target
                + 0.12 * deploy_dominance
                - 0.20 * hierarchical_defense_pressure
                - 0.08 * portfolio_release_pressure,
                0.18,
                0.76,
            )
        )
        min_candidate = int(
            np.clip(
                round(
                    2.0
                    + deploy_gate_target * 4.8
                    + deploy_dominance * 2.2
                    + large_upside_1d_target * 1.8
                    - hierarchical_defense_pressure * 1.8
                ),
                2,
                9,
            )
        )
        diagnostics["result_value_hierarchical_defense_pressure"] = hierarchical_defense_pressure
        diagnostics["result_value_portfolio_release_pressure"] = portfolio_release_pressure
        adjusted["budget_risk_signal_target"] = sharpened_risk
        adjusted["budget_deploy_signal_target"] = sharpened_deploy
        adjusted["budget_cash_timing_signal_target"] = sharpened_cash
        adjusted["budget_alpha_focus_signal_target"] = float(
            np.clip(
                0.34 * alpha_alignment
                + 0.28 * deploy_value_target
                + 0.22 * deploy_gate_target
                + 0.10 * alpha_opportunity_value
                + 0.06 * large_upside_1d_target,
                0.0,
                1.0,
            )
        )
        adjusted["gross_exposure_target"] = float(
            np.clip(
                base_gross
                + 0.18 * (sharpened_deploy - 0.34)
                + 0.08 * deploy_dominance
                + 0.06 * max(deploy_release_spread, 0.0)
                - 0.22 * hierarchical_defense_pressure
                - 0.08 * portfolio_release_pressure,
                min_gross,
                0.96,
            )
        )
        adjusted["candidate_budget"] = float(
            int(
                np.clip(
                    round(
                        0.34 * float(adjusted.get("candidate_budget", 2.0) or 2.0)
                        + 0.30 * candidate_hint
                        + 2.8 * deploy_gate_target
                        + 1.2 * deploy_dominance
                        + 0.8 * alpha_opportunity_value
                        - 1.2 * hierarchical_defense_pressure
                        - 0.6 * portfolio_release_pressure
                    ),
                    min_candidate,
                    14,
                )
            )
        )
        adjusted["turnover_budget"] = float(
            np.clip(
                float(adjusted.get("turnover_budget", 0.10) or 0.10)
                + 0.10 * portfolio_release_pressure
                + 0.06 * max(hierarchical_defense_pressure - deploy_dominance, 0.0)
                + 0.04 * max(deploy_dominance - hierarchical_defense_pressure, 0.0),
                0.08,
                0.86,
            )
        )
        adjusted["max_position_weight_target"] = float(
            np.clip(
                float(adjusted.get("max_position_weight_target", 0.10) or 0.10)
                + 0.022 * deploy_gate_target
                + 0.014 * deploy_value_target
                - 0.012 * hierarchical_defense_pressure
                - 0.006 * portfolio_release_pressure,
                0.07,
                0.29,
            )
        )
        adjusted["hold_bias_target"] = float(
            np.clip(
                float(adjusted.get("hold_bias_target", 0.18) or 0.18)
                + 0.18 * hold_continuation_value
                + 0.12 * deploy_dominance
                + 0.10 * deploy_gate_target
                + 0.06 * deploy_value_target
                - 0.10 * release_dominance
                - 0.08 * release_gate_target,
                0.10,
                0.96,
            )
        )
        adjusted["reduce_bias_target"] = float(
            np.clip(
                float(adjusted.get("reduce_bias_target", 0.10) or 0.10)
                + 0.20 * release_gate_target
                + 0.16 * release_value_target
                + 0.08 * portfolio_release_pressure
                - 0.08 * deploy_dominance
                - 0.04 * hold_continuation_value,
                0.0,
                0.70,
            )
        )
        adjusted["exit_patience_target"] = float(
            np.clip(
                float(adjusted.get("exit_patience_target", 0.20) or 0.20)
                + 0.14 * deploy_gate_target
                + 0.10 * deploy_value_target
                + 0.08 * hold_continuation_value
                - 0.14 * release_gate_target
                - 0.12 * portfolio_release_pressure,
                0.05,
                0.95,
            )
        )
        adjusted["reentry_guard_target"] = float(
            np.clip(
                float(adjusted.get("reentry_guard_target", 0.0) or 0.0)
                + 0.18 * hierarchical_defense_pressure
                + 0.08 * portfolio_release_pressure
                - 0.08 * deploy_gate_target,
                0.0,
                0.55,
            )
        )
        diagnostics["result_value_constraint_only_mode"] = 1.0
        return adjusted, diagnostics
    if objective == BUDGET_OBJECTIVE_RESULT_VALUE_V7:
        release_dominance = max(release_gate_target - deploy_gate_target, 0.0)
        deploy_dominance = max(deploy_gate_target - release_gate_target, 0.0)
        deploy_release_spread = deploy_value_target - release_value_target
        hierarchical_defense_pressure = float(
            np.clip(
                0.34 * cash_timing
                + 0.22 * cash_defense_value
                + 0.18 * risk
                + 0.12 * forward_benchmark_downside
                + 0.10 * sell_selection_pressure
                + 0.08 * max(risk_deploy_gap, 0.0)
                + 0.06 * defense_value_target
                - 0.14 * deploy_gate_target
                - 0.10 * deploy_value_target,
                0.0,
                1.0,
            )
        )
        sharpened_risk = float(
            np.clip(
                0.38 * risk
                + 0.24 * hierarchical_defense_pressure
                + 0.18 * release_value_target
                + 0.12 * release_gate_target
                + 0.08 * forward_benchmark_downside
                - 0.12 * deploy_gate_target
                - 0.08 * deploy_value_target,
                0.0,
                1.0,
            )
        )
        sharpened_deploy = float(
            np.clip(
                0.36 * deploy
                + 0.28 * deploy_value_target
                + 0.20 * deploy_gate_target
                + 0.14 * alpha_opportunity_value
                + 0.12 * deployment_opportunity_cost
                + 0.08 * large_upside_1d_target
                - 0.12 * hierarchical_defense_pressure
                - 0.10 * release_gate_target
                - 0.06 * release_value_target,
                0.0,
                1.0,
            )
        )
        sharpened_cash = float(
            np.clip(
                0.42 * cash_timing
                + 0.30 * hierarchical_defense_pressure
                + 0.12 * cash_defense_value
                + 0.10 * forward_benchmark_downside
                + 0.08 * max(risk_deploy_gap, 0.0)
                - 0.18 * deploy_gate_target
                - 0.10 * deploy_value_target,
                0.0,
                1.0,
            )
        )
        min_gross = float(
            np.clip(
                0.22
                + 0.34 * deployment_floor
                + 0.22 * deploy_gate_target
                + 0.14 * deploy_dominance
                - 0.24 * hierarchical_defense_pressure
                - 0.08 * release_dominance,
                0.18,
                0.74,
            )
        )
        min_candidate = int(
            np.clip(
                round(
                    2.0
                    + deploy_gate_target * 5.2
                    + deploy_dominance * 2.6
                    + large_upside_1d_target * 2.0
                    - hierarchical_defense_pressure * 2.0
                ),
                2,
                9,
            )
        )
        diagnostics["result_value_hierarchical_defense_pressure"] = hierarchical_defense_pressure
        adjusted["budget_risk_signal_target"] = sharpened_risk
        adjusted["budget_deploy_signal_target"] = sharpened_deploy
        adjusted["budget_cash_timing_signal_target"] = sharpened_cash
        adjusted["budget_alpha_focus_signal_target"] = float(
            np.clip(
                0.34 * alpha_alignment
                + 0.26 * deploy_value_target
                + 0.20 * deploy_gate_target
                + 0.12 * alpha_opportunity_value
                + 0.08 * large_upside_1d_target,
                0.0,
                1.0,
            )
        )
        adjusted["gross_exposure_target"] = float(
            np.clip(
                base_gross
                + 0.18 * (sharpened_deploy - 0.34)
                + 0.10 * deploy_dominance
                + 0.08 * max(deploy_release_spread, 0.0)
                + 0.05 * alpha_opportunity_value
                - 0.18 * hierarchical_defense_pressure
                - 0.10 * release_dominance,
                min_gross,
                0.96,
            )
        )
        adjusted["candidate_budget"] = float(
            int(
                np.clip(
                    round(
                        0.34 * float(adjusted.get("candidate_budget", 2.0) or 2.0)
                        + 0.28 * candidate_hint
                        + 2.8 * deploy_gate_target
                        + 1.4 * deploy_dominance
                        + 1.0 * alpha_opportunity_value
                        - 1.4 * hierarchical_defense_pressure
                        - 0.8 * release_gate_target
                    ),
                    min_candidate,
                    14,
                )
            )
        )
        adjusted["turnover_budget"] = float(
            np.clip(
                float(adjusted.get("turnover_budget", 0.10) or 0.10)
                + 0.10 * release_gate_target
                + 0.08 * release_value_target
                + 0.05 * max(deploy_release_spread, 0.0)
                + 0.03 * hierarchical_defense_pressure
                - 0.03 * deploy_dominance,
                0.08,
                0.86,
            )
        )
        adjusted["max_position_weight_target"] = float(
            np.clip(
                float(adjusted.get("max_position_weight_target", 0.10) or 0.10)
                + 0.022 * deploy_gate_target
                + 0.016 * deploy_value_target
                + 0.010 * alpha_opportunity_value
                - 0.014 * hierarchical_defense_pressure
                - 0.006 * release_gate_target,
                0.07,
                0.29,
            )
        )
        adjusted["hold_bias_target"] = float(
            np.clip(
                float(adjusted.get("hold_bias_target", 0.18) or 0.18)
                + 0.16 * hold_continuation_value
                + 0.14 * deploy_gate_target
                + 0.10 * deploy_value_target
                + 0.06 * alpha_opportunity_value
                - 0.14 * release_gate_target
                - 0.12 * hierarchical_defense_pressure,
                0.10,
                0.96,
            )
        )
        adjusted["reduce_bias_target"] = float(
            np.clip(
                float(adjusted.get("reduce_bias_target", 0.10) or 0.10)
                + 0.18 * release_gate_target
                + 0.14 * release_value_target
                + 0.08 * hierarchical_defense_pressure
                - 0.10 * deploy_gate_target
                - 0.06 * deploy_value_target,
                0.0,
                0.66,
            )
        )
        adjusted["exit_patience_target"] = float(
            np.clip(
                float(adjusted.get("exit_patience_target", 0.20) or 0.20)
                + 0.12 * deploy_gate_target
                + 0.10 * deploy_value_target
                + 0.08 * hold_continuation_value
                - 0.16 * release_gate_target
                - 0.14 * hierarchical_defense_pressure,
                0.05,
                0.95,
            )
        )
        adjusted["reentry_guard_target"] = float(
            np.clip(
                float(adjusted.get("reentry_guard_target", 0.0) or 0.0)
                + 0.10 * reentry_guard
                + 0.14 * hierarchical_defense_pressure
                + 0.06 * release_gate_target
                - 0.10 * deploy_gate_target,
                0.0,
                0.45,
            )
        )
        diagnostics["result_value_constraint_only_mode"] = 0.0
        return adjusted, diagnostics
    if objective == BUDGET_OBJECTIVE_RESULT_VALUE_V6:
        release_dominance = max(release_gate_target - deploy_gate_target, 0.0)
        defense_dominance = max(defense_gate_target - deploy_gate_target, 0.0)
        deploy_dominance = max(deploy_gate_target - max(release_gate_target, defense_gate_target), 0.0)
        deploy_release_spread = deploy_value_target - release_value_target
        deploy_defense_spread = deploy_value_target - defense_value_target
        sharpened_risk = float(
            np.clip(
                0.42 * risk
                + 0.22 * release_value_target
                + 0.18 * defense_value_target
                + 0.12 * release_gate_target
                + 0.12 * defense_gate_target
                + 0.08 * forward_benchmark_downside
                - 0.14 * deploy_gate_target
                - 0.10 * deploy_value_target,
                0.0,
                1.0,
            )
        )
        sharpened_deploy = float(
            np.clip(
                0.34 * deploy
                + 0.24 * deploy_value_target
                + 0.18 * deploy_gate_target
                + 0.14 * alpha_opportunity_value
                + 0.10 * deployment_opportunity_cost
                + 0.08 * large_upside_1d_target
                - 0.14 * defense_gate_target
                - 0.10 * release_gate_target,
                0.0,
                1.0,
            )
        )
        sharpened_cash = float(
            np.clip(
                0.36 * cash_timing
                + 0.24 * defense_value_target
                + 0.20 * defense_gate_target
                + 0.10 * max(risk_deploy_gap, 0.0)
                + 0.08 * release_gate_target
                - 0.22 * deploy_gate_target
                - 0.12 * deploy_value_target,
                0.0,
                1.0,
            )
        )
        min_gross = float(
            np.clip(
                0.24
                + 0.30 * deployment_floor
                + 0.22 * deploy_gate_target
                + 0.12 * deploy_dominance
                - 0.18 * defense_gate_target
                - 0.10 * release_dominance,
                0.20,
                0.72,
            )
        )
        min_candidate = int(
            np.clip(
                round(2.0 + deploy_gate_target * 5.0 + deploy_dominance * 3.0 + large_upside_1d_target * 2.0 - defense_gate_target * 1.4),
                2,
                9,
            )
        )
        adjusted["budget_risk_signal_target"] = sharpened_risk
        adjusted["budget_deploy_signal_target"] = sharpened_deploy
        adjusted["budget_cash_timing_signal_target"] = sharpened_cash
        adjusted["budget_alpha_focus_signal_target"] = float(
            np.clip(
                0.34 * alpha_alignment
                + 0.24 * deploy_value_target
                + 0.20 * deploy_gate_target
                + 0.14 * alpha_opportunity_value
                + 0.08 * large_upside_1d_target,
                0.0,
                1.0,
            )
        )
        adjusted["gross_exposure_target"] = float(
            np.clip(
                base_gross
                + 0.16 * (sharpened_deploy - 0.36)
                + 0.10 * deploy_dominance
                + 0.08 * max(deploy_release_spread, 0.0)
                + 0.05 * alpha_opportunity_value
                - 0.14 * sharpened_cash
                - 0.10 * defense_dominance
                - 0.08 * release_dominance,
                min_gross,
                0.96,
            )
        )
        adjusted["candidate_budget"] = float(
            int(
                np.clip(
                    round(
                        0.36 * float(adjusted.get("candidate_budget", 2.0) or 2.0)
                        + 0.26 * candidate_hint
                        + 2.6 * deploy_gate_target
                        + 1.5 * deploy_dominance
                        + 1.0 * alpha_opportunity_value
                        - 1.3 * defense_gate_target
                        - 0.7 * release_gate_target
                    ),
                    min_candidate,
                    14,
                )
            )
        )
        adjusted["turnover_budget"] = float(
            np.clip(
                float(adjusted.get("turnover_budget", 0.10) or 0.10)
                + 0.10 * release_gate_target
                + 0.08 * release_value_target
                + 0.06 * max(deploy_release_spread, 0.0)
                + 0.04 * deploy_gate_target
                + 0.04 * sharpened_cash
                - 0.04 * defense_dominance,
                0.08,
                0.86,
            )
        )
        adjusted["max_position_weight_target"] = float(
            np.clip(
                float(adjusted.get("max_position_weight_target", 0.10) or 0.10)
                + 0.020 * deploy_gate_target
                + 0.016 * deploy_value_target
                + 0.010 * alpha_opportunity_value
                - 0.012 * defense_gate_target
                - 0.008 * release_gate_target,
                0.07,
                0.29,
            )
        )
        adjusted["hold_bias_target"] = float(
            np.clip(
                float(adjusted.get("hold_bias_target", 0.18) or 0.18)
                + 0.14 * hold_continuation_value
                + 0.14 * deploy_gate_target
                + 0.10 * deploy_value_target
                + 0.06 * alpha_opportunity_value
                - 0.14 * release_gate_target
                - 0.08 * defense_gate_target,
                0.10,
                0.96,
            )
        )
        adjusted["reduce_bias_target"] = float(
            np.clip(
                float(adjusted.get("reduce_bias_target", 0.10) or 0.10)
                + 0.18 * release_gate_target
                + 0.14 * release_value_target
                + 0.06 * defense_gate_target
                - 0.08 * deploy_gate_target
                - 0.06 * deploy_value_target,
                0.0,
                0.66,
            )
        )
        adjusted["exit_patience_target"] = float(
            np.clip(
                float(adjusted.get("exit_patience_target", 0.20) or 0.20)
                + 0.12 * deploy_gate_target
                + 0.10 * deploy_value_target
                + 0.08 * hold_continuation_value
                - 0.16 * release_gate_target
                - 0.10 * defense_gate_target,
                0.05,
                0.95,
            )
        )
        adjusted["reentry_guard_target"] = float(
            np.clip(
                float(adjusted.get("reentry_guard_target", 0.0) or 0.0)
                + 0.10 * reentry_guard
                + 0.10 * defense_gate_target
                + 0.08 * max(-deploy_defense_spread, 0.0)
                + 0.04 * release_gate_target
                - 0.10 * deploy_gate_target,
                0.0,
                0.45,
            )
        )
        return adjusted, diagnostics
    if objective == BUDGET_OBJECTIVE_RESULT_VALUE_V5:
        positive_gap = max(risk_deploy_gap, 0.0)
        opportunity_gap = max(arbitration_deploy_pressure - arbitration_cash_pressure, 0.0)
        value_capital_bias = max(value_arbitration_target - 0.50, 0.0) * 2.0
        value_defense_bias = max(0.50 - value_arbitration_target, 0.0) * 2.0
        sharpened_risk = float(
            np.clip(
                0.56 * risk
                + 0.24 * cash_defense_value
                + 0.16 * arbitration_cash_pressure
                + 0.16 * arbitration_sell_pressure
                + 0.10 * forward_benchmark_downside
                - 0.12 * arbitration_deploy_pressure
                - 0.08 * alpha_opportunity_value,
                0.0,
                1.0,
            )
        )
        sharpened_deploy = float(
            np.clip(
                0.42 * deploy
                + 0.24 * arbitration_deploy_pressure
                + 0.18 * alpha_opportunity_value
                + 0.12 * deployment_opportunity_cost
                + 0.08 * large_upside_1d_target
                + 0.06 * opportunity_cost
                - 0.16 * arbitration_cash_pressure
                - 0.10 * cash_defense_value,
                0.0,
                1.0,
            )
        )
        sharpened_cash = float(
            np.clip(
                0.42 * cash_timing
                + 0.26 * arbitration_cash_pressure
                + 0.16 * cash_defense_value
                + 0.10 * positive_gap
                + 0.08 * arbitration_sell_pressure
                + 0.06 * value_defense_bias
                - 0.24 * arbitration_deploy_pressure
                - 0.14 * alpha_opportunity_value,
                0.0,
                1.0,
            )
        )
        min_gross = float(
            np.clip(
                0.20
                + 0.28 * deployment_floor
                + 0.18 * arbitration_deploy_pressure
                + 0.08 * value_capital_bias
                - 0.16 * sharpened_cash,
                0.18,
                0.64,
            )
        )
        min_candidate = int(
            np.clip(
                round(2.0 + arbitration_deploy_pressure * 5.0 + large_upside_1d_target * 2.0 - sharpened_cash * 1.2),
                2,
                8,
            )
        )
        adjusted["budget_risk_signal_target"] = sharpened_risk
        adjusted["budget_deploy_signal_target"] = sharpened_deploy
        adjusted["budget_cash_timing_signal_target"] = sharpened_cash
        adjusted["budget_alpha_focus_signal_target"] = float(
            np.clip(
                0.50 * alpha_alignment
                + 0.24 * alpha_opportunity_value
                + 0.14 * deployment_opportunity_cost
                + 0.12 * large_upside_1d_target,
                0.0,
                1.0,
            )
        )
        adjusted["gross_exposure_target"] = float(
            np.clip(
                base_gross
                + 0.14 * (sharpened_deploy - 0.38)
                + 0.07 * arbitration_deploy_pressure
                + 0.05 * alpha_opportunity_value
                + 0.04 * value_capital_bias
                - 0.16 * sharpened_cash
                - 0.10 * arbitration_sell_pressure
                - 0.08 * max(sharpened_risk - sharpened_deploy, 0.0),
                min_gross,
                0.95,
            )
        )
        adjusted["candidate_budget"] = float(
            int(
                np.clip(
                    round(
                        0.40 * float(adjusted.get("candidate_budget", 2.0) or 2.0)
                        + 0.30 * candidate_hint
                        + 2.1 * arbitration_deploy_pressure
                        + 1.2 * alpha_opportunity_value
                        + 0.9 * large_upside_1d_target
                        - 1.4 * sharpened_cash
                        - 0.8 * arbitration_sell_pressure
                    ),
                    min_candidate,
                    14,
                )
            )
        )
        adjusted["turnover_budget"] = float(
            np.clip(
                float(adjusted.get("turnover_budget", 0.10) or 0.10)
                + 0.11 * arbitration_sell_pressure
                + 0.08 * sell_release_value
                + 0.06 * sharpened_cash
                + 0.05 * opportunity_gap
                + 0.03 * large_upside_1d_target
                - 0.04 * value_capital_bias
                - 0.04 * alpha_opportunity_value * max(sharpened_deploy - sharpened_risk, 0.0),
                0.08,
                0.86,
            )
        )
        adjusted["max_position_weight_target"] = float(
            np.clip(
                float(adjusted.get("max_position_weight_target", 0.10) or 0.10)
                + 0.022 * alpha_opportunity_value
                + 0.018 * arbitration_deploy_pressure
                + 0.010 * large_upside_1d_target
                - 0.014 * sharpened_cash
                - 0.010 * arbitration_sell_pressure,
                0.07,
                0.28,
            )
        )
        adjusted["hold_bias_target"] = float(
            np.clip(
                float(adjusted.get("hold_bias_target", 0.18) or 0.18)
                + 0.16 * hold_continuation_value
                + 0.12 * alpha_opportunity_value
                + 0.08 * value_capital_bias
                + 0.05 * large_upside_1d_target
                - 0.18 * arbitration_sell_pressure
                - 0.08 * sharpened_cash,
                0.10,
                0.95,
            )
        )
        adjusted["reduce_bias_target"] = float(
            np.clip(
                float(adjusted.get("reduce_bias_target", 0.10) or 0.10)
                + 0.20 * arbitration_sell_pressure
                + 0.12 * sell_release_value
                + 0.06 * sharpened_cash
                - 0.05 * hold_continuation_value
                - 0.05 * alpha_opportunity_value,
                0.0,
                0.66,
            )
        )
        adjusted["exit_patience_target"] = float(
            np.clip(
                float(adjusted.get("exit_patience_target", 0.20) or 0.20)
                + 0.12 * hold_continuation_value
                + 0.10 * alpha_opportunity_value
                + 0.06 * value_capital_bias
                - 0.18 * arbitration_sell_pressure
                - 0.10 * sharpened_cash
                - 0.08 * value_defense_bias,
                0.05,
                0.95,
            )
        )
        adjusted["reentry_guard_target"] = float(
            np.clip(
                float(adjusted.get("reentry_guard_target", 0.0) or 0.0)
                + 0.12 * reentry_guard
                + 0.08 * sharpened_cash
                + 0.08 * arbitration_cash_pressure
                + 0.06 * sell_release_value
                - 0.08 * arbitration_deploy_pressure
                - 0.05 * alpha_opportunity_value,
                0.0,
                0.45,
            )
        )
        return adjusted, diagnostics
    if objective == BUDGET_OBJECTIVE_RESULT_VALUE_V3:
        sharpened_risk = float(np.clip(risk + max(risk_deploy_gap, 0.0) * 0.22 + sell_pressure * 0.08, 0.0, 1.0))
        sharpened_deploy = float(np.clip(deploy + alpha_alignment * 0.08 + opportunity_concentration * 0.06 - cash_timing * 0.12, 0.0, 1.0))
        sharpened_cash = float(
            np.clip(
                cash_timing * 1.20
                + max(risk_deploy_gap, 0.0) * 0.22
                + sell_pressure * 0.10
                - max(sharpened_deploy - sharpened_risk, 0.0) * 0.08,
                0.0,
                1.0,
            )
        )
        adjusted["budget_risk_signal_target"] = sharpened_risk
        adjusted["budget_deploy_signal_target"] = sharpened_deploy
        adjusted["budget_cash_timing_signal_target"] = sharpened_cash
        adjusted["budget_alpha_focus_signal_target"] = float(np.clip(alpha_alignment * 0.88 + opportunity_concentration * 0.12, 0.0, 1.0))
        adjusted["gross_exposure_target"] = float(
            np.clip(
                base_gross
                + 0.12 * (sharpened_deploy - 0.36)
                + 0.05 * alpha_alignment
                + 0.04 * opportunity_concentration
                - 0.30 * sharpened_cash
                - 0.18 * max(risk_deploy_gap, 0.0)
                - 0.05 * sell_pressure,
                0.10,
                0.92,
            )
        )
        adjusted["candidate_budget"] = float(
            int(
                np.clip(
                    round(
                        0.40 * float(adjusted.get("candidate_budget", 2.0) or 2.0)
                        + 0.36 * candidate_hint
                        + 1.8 * max(sharpened_deploy - sharpened_risk, 0.0)
                        + 1.0 * alpha_alignment
                        - 2.8 * sharpened_cash
                        - 1.0 * max(risk_deploy_gap, 0.0)
                    ),
                    1,
                    12,
                )
            )
        )
        adjusted["turnover_budget"] = float(
            np.clip(
                float(adjusted.get("turnover_budget", 0.10) or 0.10)
                + 0.14 * sell_pressure
                + 0.10 * sharpened_cash
                + 0.04 * reentry_guard
                + 0.02 * max(sharpened_deploy - 0.42, 0.0)
                - 0.02 * opportunity_concentration
                - 0.02 * alpha_alignment * max(sharpened_deploy - sharpened_risk, 0.0),
                0.08,
                0.94,
            )
        )
        adjusted["max_position_weight_target"] = float(
            np.clip(
                float(adjusted.get("max_position_weight_target", 0.10) or 0.10)
                + 0.026 * alpha_alignment
                + 0.018 * opportunity_concentration
                + 0.010 * max(sharpened_deploy - sharpened_risk, 0.0)
                - 0.036 * sharpened_cash
                - 0.016 * sell_pressure,
                0.06,
                0.26,
            )
        )
        adjusted["hold_bias_target"] = float(
            np.clip(
                float(adjusted.get("hold_bias_target", 0.18) or 0.18)
                + 0.12 * max(edge_top / 0.055, 0.0)
                + 0.10 * alpha_alignment
                + 0.05 * opportunity_concentration
                - 0.20 * sell_pressure
                - 0.18 * sharpened_cash
                - 0.10 * max(risk_deploy_gap, 0.0),
                0.10,
                0.92,
            )
        )
        adjusted["reduce_bias_target"] = float(
            np.clip(
                float(adjusted.get("reduce_bias_target", 0.10) or 0.10)
                + 0.22 * sell_pressure
                + 0.22 * sharpened_cash
                + 0.12 * max(risk_deploy_gap, 0.0)
                - 0.04 * alpha_alignment
                - 0.03 * max(sharpened_deploy - sharpened_risk, 0.0),
                0.0,
                0.68,
            )
        )
        adjusted["exit_patience_target"] = float(
            np.clip(
                float(adjusted.get("exit_patience_target", 0.20) or 0.20)
                + 0.10 * alpha_alignment
                + 0.07 * opportunity_concentration
                + 0.05 * max(edge_top / 0.055, 0.0)
                - 0.22 * sell_pressure
                - 0.20 * sharpened_cash
                - 0.08 * max(risk_deploy_gap, 0.0),
                0.05,
                0.92,
            )
        )
        adjusted["reentry_guard_target"] = float(
            np.clip(
                float(adjusted.get("reentry_guard_target", 0.0) or 0.0)
                + 0.20 * reentry_guard
                + 0.14 * sharpened_cash
                + 0.08 * max(risk_deploy_gap, 0.0)
                - 0.05 * alpha_alignment
                - 0.02 * max(sharpened_deploy - sharpened_risk, 0.0),
                0.0,
                0.45,
            )
        )
        return adjusted, diagnostics

    if objective == BUDGET_OBJECTIVE_RESULT_VALUE_V4B:
        positive_gap = max(risk_deploy_gap, 0.0)
        opportunity_gap = max(deploy - risk, 0.0)
        sharpened_risk = float(
            np.clip(
                risk
                + positive_gap * 0.14
                + sell_selection_pressure * 0.16
                + held_sell_pressure * 0.08
                + forward_benchmark_downside * 0.12
                - opportunity_cost * 0.08,
                0.0,
                1.0,
            )
        )
        sharpened_deploy = float(
            np.clip(
                deploy
                + alpha_alignment * 0.12
                + opportunity_concentration * 0.10
                + opportunity_cost * 0.16
                + deployment_floor * 0.08
                - cash_timing * 0.08
                - sell_selection_pressure * 0.05,
                0.0,
                1.0,
            )
        )
        sharpened_cash = float(
            np.clip(
                cash_timing * 0.96
                + positive_gap * 0.16
                + sell_selection_pressure * 0.16
                + held_sell_pressure * 0.08
                + forward_benchmark_downside * 0.12
                - opportunity_cost * 0.30
                - alpha_alignment * 0.08
                - opportunity_gap * 0.08,
                0.0,
                1.0,
            )
        )
        min_gross = float(np.clip(0.18 + deployment_floor * 0.34 - sharpened_cash * 0.12, 0.18, 0.56))
        min_candidate = int(np.clip(round(2.0 + deployment_floor * 4.0 - sharpened_cash * 1.0), 2, 7))
        adjusted["budget_risk_signal_target"] = sharpened_risk
        adjusted["budget_deploy_signal_target"] = sharpened_deploy
        adjusted["budget_cash_timing_signal_target"] = sharpened_cash
        adjusted["budget_alpha_focus_signal_target"] = float(
            np.clip(alpha_alignment * 0.78 + opportunity_concentration * 0.14 + opportunity_cost * 0.08, 0.0, 1.0)
        )
        adjusted["gross_exposure_target"] = float(
            np.clip(
                base_gross
                + 0.12 * (sharpened_deploy - 0.38)
                + 0.06 * alpha_alignment
                + 0.06 * opportunity_cost
                + 0.04 * deployment_floor
                - 0.18 * sharpened_cash
                - 0.12 * positive_gap
                - 0.06 * sell_selection_pressure,
                min_gross,
                0.94,
            )
        )
        adjusted["candidate_budget"] = float(
            int(
                np.clip(
                    round(
                        0.44 * float(adjusted.get("candidate_budget", 2.0) or 2.0)
                        + 0.34 * candidate_hint
                        + 1.7 * max(sharpened_deploy - sharpened_risk, 0.0)
                        + 1.1 * alpha_alignment
                        + 1.2 * opportunity_cost
                        - 1.6 * sharpened_cash
                        - 0.8 * sell_selection_pressure
                    ),
                    min_candidate,
                    13,
                )
            )
        )
        adjusted["turnover_budget"] = float(
            np.clip(
                float(adjusted.get("turnover_budget", 0.10) or 0.10)
                + 0.10 * sell_pressure
                + 0.12 * sell_selection_pressure
                + 0.06 * sharpened_cash
                + 0.04 * positive_gap
                - 0.05 * opportunity_cost
                - 0.03 * alpha_alignment * max(sharpened_deploy - sharpened_risk, 0.0),
                0.08,
                0.82,
            )
        )
        adjusted["max_position_weight_target"] = float(
            np.clip(
                float(adjusted.get("max_position_weight_target", 0.10) or 0.10)
                + 0.024 * alpha_alignment
                + 0.014 * opportunity_cost
                + 0.010 * max(sharpened_deploy - sharpened_risk, 0.0)
                - 0.014 * sharpened_cash
                - 0.012 * sell_selection_pressure,
                0.07,
                0.27,
            )
        )
        adjusted["hold_bias_target"] = float(
            np.clip(
                float(adjusted.get("hold_bias_target", 0.18) or 0.18)
                + 0.14 * max(edge_top / 0.055, 0.0)
                + 0.12 * alpha_alignment
                + 0.08 * opportunity_cost
                + 0.04 * deployment_floor
                - 0.20 * sell_selection_pressure
                - 0.08 * sharpened_cash
                - 0.06 * max(sharpened_risk - sharpened_deploy, 0.0),
                0.10,
                0.94,
            )
        )
        adjusted["reduce_bias_target"] = float(
            np.clip(
                float(adjusted.get("reduce_bias_target", 0.10) or 0.10)
                + 0.20 * sell_selection_pressure
                + 0.10 * held_sell_pressure
                + 0.07 * sell_pressure
                + 0.05 * max(sharpened_risk - sharpened_deploy, 0.0)
                - 0.05 * opportunity_cost
                - 0.04 * alpha_alignment,
                0.0,
                0.66,
            )
        )
        adjusted["exit_patience_target"] = float(
            np.clip(
                float(adjusted.get("exit_patience_target", 0.20) or 0.20)
                + 0.10 * alpha_alignment
                + 0.08 * opportunity_cost
                + 0.05 * max(edge_top / 0.055, 0.0)
                - 0.18 * sell_selection_pressure
                - 0.10 * sharpened_cash
                - 0.07 * max(sharpened_risk - sharpened_deploy, 0.0),
                0.05,
                0.94,
            )
        )
        adjusted["reentry_guard_target"] = float(
            np.clip(
                float(adjusted.get("reentry_guard_target", 0.0) or 0.0)
                + 0.15 * reentry_guard
                + 0.08 * sharpened_cash
                + 0.08 * sell_selection_pressure
                + 0.04 * positive_gap
                - 0.06 * opportunity_cost
                - 0.05 * alpha_alignment,
                0.0,
                0.45,
            )
        )
        return adjusted, diagnostics

    if objective == BUDGET_OBJECTIVE_RESULT_VALUE_V4:
        sharpened_risk = float(
            np.clip(
                risk
                + max(risk_deploy_gap, 0.0) * 0.18
                + sell_selection_pressure * 0.18
                + held_sell_pressure * 0.10
                + forward_benchmark_downside * 0.14,
                0.0,
                1.0,
            )
        )
        sharpened_deploy = float(
            np.clip(
                deploy
                + alpha_alignment * 0.10
                + opportunity_concentration * 0.08
                - cash_timing * 0.10
                - sell_selection_pressure * 0.08,
                0.0,
                1.0,
            )
        )
        sharpened_cash = float(
            np.clip(
                cash_timing * 1.12
                + max(risk_deploy_gap, 0.0) * 0.18
                + sell_selection_pressure * 0.18
                + held_sell_pressure * 0.10
                + forward_benchmark_downside * 0.14
                - max(sharpened_deploy - sharpened_risk, 0.0) * 0.10,
                0.0,
                1.0,
            )
        )
        adjusted["budget_risk_signal_target"] = sharpened_risk
        adjusted["budget_deploy_signal_target"] = sharpened_deploy
        adjusted["budget_cash_timing_signal_target"] = sharpened_cash
        adjusted["budget_alpha_focus_signal_target"] = float(
            np.clip(alpha_alignment * 0.84 + opportunity_concentration * 0.16, 0.0, 1.0)
        )
        adjusted["gross_exposure_target"] = float(
            np.clip(
                base_gross
                + 0.11 * (sharpened_deploy - 0.38)
                + 0.05 * alpha_alignment
                + 0.04 * opportunity_concentration
                - 0.24 * sharpened_cash
                - 0.14 * max(risk_deploy_gap, 0.0)
                - 0.08 * sell_selection_pressure
                - 0.04 * sell_pressure,
                0.10,
                0.92,
            )
        )
        adjusted["candidate_budget"] = float(
            int(
                np.clip(
                    round(
                        0.42 * float(adjusted.get("candidate_budget", 2.0) or 2.0)
                        + 0.34 * candidate_hint
                        + 1.6 * max(sharpened_deploy - sharpened_risk, 0.0)
                        + 1.0 * alpha_alignment
                        - 2.2 * sharpened_cash
                        - 1.4 * sell_selection_pressure
                    ),
                    2,
                    12,
                )
            )
        )
        adjusted["turnover_budget"] = float(
            np.clip(
                float(adjusted.get("turnover_budget", 0.10) or 0.10)
                + 0.12 * sell_pressure
                + 0.10 * sell_selection_pressure
                + 0.08 * sharpened_cash
                + 0.04 * max(sharpened_risk - sharpened_deploy, 0.0)
                - 0.04 * alpha_alignment * max(sharpened_deploy - sharpened_risk, 0.0),
                0.08,
                0.74,
            )
        )
        adjusted["max_position_weight_target"] = float(
            np.clip(
                float(adjusted.get("max_position_weight_target", 0.10) or 0.10)
                + 0.020 * alpha_alignment
                + 0.012 * max(sharpened_deploy - sharpened_risk, 0.0)
                - 0.018 * sharpened_cash
                - 0.014 * sell_selection_pressure,
                0.07,
                0.26,
            )
        )
        adjusted["hold_bias_target"] = float(
            np.clip(
                float(adjusted.get("hold_bias_target", 0.18) or 0.18)
                + 0.14 * max(edge_top / 0.055, 0.0)
                + 0.10 * alpha_alignment
                - 0.22 * sell_selection_pressure
                - 0.12 * sharpened_cash
                - 0.08 * max(sharpened_risk - sharpened_deploy, 0.0),
                0.10,
                0.92,
            )
        )
        adjusted["reduce_bias_target"] = float(
            np.clip(
                float(adjusted.get("reduce_bias_target", 0.10) or 0.10)
                + 0.14 * sell_selection_pressure
                + 0.08 * held_sell_pressure
                + 0.08 * sell_pressure
                + 0.06 * max(sharpened_risk - sharpened_deploy, 0.0)
                - 0.04 * alpha_alignment,
                0.0,
                0.65,
            )
        )
        adjusted["exit_patience_target"] = float(
            np.clip(
                float(adjusted.get("exit_patience_target", 0.20) or 0.20)
                + 0.08 * alpha_alignment
                + 0.06 * max(edge_top / 0.055, 0.0)
                - 0.18 * sell_selection_pressure
                - 0.14 * sharpened_cash
                - 0.08 * max(sharpened_risk - sharpened_deploy, 0.0),
                0.05,
                0.92,
            )
        )
        adjusted["reentry_guard_target"] = float(
            np.clip(
                float(adjusted.get("reentry_guard_target", 0.0) or 0.0)
                + 0.16 * reentry_guard
                + 0.10 * sharpened_cash
                + 0.08 * sell_selection_pressure
                + 0.04 * max(risk_deploy_gap, 0.0)
                - 0.05 * alpha_alignment,
                0.0,
                0.45,
            )
        )
        return adjusted, diagnostics

    adjusted["gross_exposure_target"] = float(
        np.clip(
            base_gross
            + 0.14 * (deploy - 0.40)
            + 0.06 * alpha_alignment
            + 0.04 * opportunity_concentration
            - 0.22 * cash_timing
            - 0.12 * max(risk_deploy_gap, 0.0)
            - 0.04 * sell_pressure,
            0.12,
            0.94,
        )
    )
    adjusted["candidate_budget"] = float(
        int(
            np.clip(
                round(
                    0.45 * float(adjusted.get("candidate_budget", 2.0) or 2.0)
                    + 0.40 * candidate_hint
                    + 2.0 * max(deploy - risk, 0.0)
                    + 1.2 * alpha_alignment
                    - 2.2 * cash_timing
                    - 0.8 * max(risk_deploy_gap, 0.0)
                ),
                2,
                14,
            )
        )
    )
    adjusted["turnover_budget"] = float(
        np.clip(
            float(adjusted.get("turnover_budget", 0.10) or 0.10)
            + 0.12 * sell_pressure
            + 0.08 * cash_timing
            + 0.03 * reentry_guard
            + 0.02 * max(deploy - 0.45, 0.0)
            - 0.03 * opportunity_concentration
            - 0.03 * alpha_alignment * max(deploy - risk, 0.0),
            0.08,
            0.90,
        )
    )
    adjusted["max_position_weight_target"] = float(
        np.clip(
            float(adjusted.get("max_position_weight_target", 0.10) or 0.10)
            + 0.032 * alpha_alignment
            + 0.020 * opportunity_concentration
            + 0.010 * max(deploy - risk, 0.0)
            - 0.026 * cash_timing
            - 0.014 * sell_pressure,
            0.07,
            0.28,
        )
    )
    adjusted["hold_bias_target"] = float(
        np.clip(
            float(adjusted.get("hold_bias_target", 0.18) or 0.18)
            + 0.14 * max(edge_top / 0.055, 0.0)
            + 0.12 * alpha_alignment
            + 0.06 * opportunity_concentration
            - 0.18 * sell_pressure
            - 0.12 * cash_timing
            - 0.08 * max(risk_deploy_gap, 0.0),
            0.10,
            0.95,
        )
    )
    adjusted["reduce_bias_target"] = float(
        np.clip(
            float(adjusted.get("reduce_bias_target", 0.10) or 0.10)
            + 0.18 * sell_pressure
            + 0.16 * cash_timing
            + 0.10 * max(risk_deploy_gap, 0.0)
            - 0.04 * alpha_alignment
            - 0.03 * max(deploy - risk, 0.0),
            0.0,
            0.65,
        )
    )
    adjusted["exit_patience_target"] = float(
        np.clip(
            float(adjusted.get("exit_patience_target", 0.20) or 0.20)
            + 0.12 * alpha_alignment
            + 0.08 * opportunity_concentration
            + 0.05 * max(edge_top / 0.055, 0.0)
            - 0.18 * sell_pressure
            - 0.16 * cash_timing
            - 0.06 * max(risk_deploy_gap, 0.0),
            0.05,
            0.95,
        )
    )
    adjusted["reentry_guard_target"] = float(
        np.clip(
            float(adjusted.get("reentry_guard_target", 0.0) or 0.0)
            + 0.18 * reentry_guard
            + 0.10 * cash_timing
            + 0.06 * max(risk_deploy_gap, 0.0)
            - 0.06 * alpha_alignment
            - 0.03 * max(deploy - risk, 0.0),
            0.0,
            0.45,
        )
    )
    return adjusted, diagnostics


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
    execution_semantics: str = DEFAULT_EXECUTION_SEMANTICS,
    budget_semantics: str = DEFAULT_BUDGET_SEMANTICS,
    budget_calibration: str = DEFAULT_BUDGET_CALIBRATION,
    budget_objective: str = DEFAULT_BUDGET_OBJECTIVE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    resolved_label_config = resolve_label_config(label_preset)
    resolved_budget_objective = resolve_budget_objective(budget_objective)
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
    budget_diagnostic_rows: list[dict[str, float]] = []

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
        global_targets, budget_diagnostics = _apply_budget_objective_targets(
            label_frame=label_frame,
            global_targets=global_targets,
            budget_objective=resolved_budget_objective,
        )
        budget_diagnostic_rows.append(budget_diagnostics)
        teacher_policy = build_teacher_policy_frame(label_frame)
        step_result = portfolio.step(
            date=signal_dt,
            prices=prepared.close.loc[signal_dt],
            policy_frame=teacher_policy,
            global_targets=global_targets,
            source_label="teacher",
            execution_semantics=execution_semantics,
            budget_semantics=budget_semantics,
            budget_calibration=budget_calibration,
        )
        label_frame = _annotate_label_frame_with_execution_feedback(label_frame, step_result.actions)

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
        "execution_semantics": str(execution_semantics),
        "budget_semantics": str(budget_semantics),
        "budget_calibration": str(budget_calibration),
        "budget_objective": str(resolved_budget_objective),
        "budget_objective_diagnostics_mean": {
            str(column): float(pd.DataFrame(budget_diagnostic_rows)[column].mean())
            for column in (pd.DataFrame(budget_diagnostic_rows).columns if budget_diagnostic_rows else [])
        },
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
    execution_semantics: str = DEFAULT_EXECUTION_SEMANTICS,
    budget_semantics: str = DEFAULT_BUDGET_SEMANTICS,
    budget_calibration: str = DEFAULT_BUDGET_CALIBRATION,
    budget_objective: str = DEFAULT_BUDGET_OBJECTIVE,
) -> dict[str, Any]:
    resolved_budget_objective = resolve_budget_objective(budget_objective)
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
            global_targets, _ = _apply_budget_objective_targets(
                label_frame=label_frame,
                global_targets=global_targets,
                budget_objective=resolved_budget_objective,
            )
        else:
            policy_frame, global_targets = predict_policy(artifact, state_frame=state_frame, daily_features=daily_features)

        step_result = portfolio.step(
            date=signal_dt,
            prices=prepared.close.loc[signal_dt],
            policy_frame=policy_frame,
            global_targets=global_targets,
            source_label=source_label,
            execution_semantics=execution_semantics,
            budget_semantics=budget_semantics,
            budget_calibration=budget_calibration,
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
    monthly_returns = build_monthly_return_frame(returns_series)
    metrics = compute_curve_metrics(returns_series)
    if not turnover_frame.empty:
        metrics["avg_turnover"] = float(turnover_frame["realized_turnover"].mean())
        metrics["avg_buy_turnover"] = float(turnover_frame["buy_turnover"].mean())
        metrics["avg_sell_turnover"] = float(turnover_frame["sell_turnover"].mean())
        metrics["avg_gross_exposure"] = float(1.0 - turnover_frame["cash_weight"].mean())
        metrics["avg_holding_count"] = float(turnover_frame["holding_count"].mean())
        metrics["avg_semantic_conflict_rate"] = float(turnover_frame["semantic_conflict_rate"].mean()) if "semantic_conflict_rate" in turnover_frame.columns else 0.0
        metrics["avg_order_translation_conflict_rate"] = float(turnover_frame["order_translation_conflict_rate"].mean()) if "order_translation_conflict_rate" in turnover_frame.columns else 0.0
        metrics["avg_budget_risk_off_score"] = float(turnover_frame["budget_risk_off_score"].mean()) if "budget_risk_off_score" in turnover_frame.columns else 0.0
        metrics["avg_budget_deploy_score"] = float(turnover_frame["budget_deploy_score"].mean()) if "budget_deploy_score" in turnover_frame.columns else 0.0
        metrics["avg_budget_entry_candidate_count"] = float(turnover_frame["budget_entry_candidate_count"].mean()) if "budget_entry_candidate_count" in turnover_frame.columns else 0.0
        metrics["avg_budget_entry_keep_count"] = float(turnover_frame["budget_entry_keep_count"].mean()) if "budget_entry_keep_count" in turnover_frame.columns else 0.0
        metrics["avg_budget_held_protected_count"] = float(turnover_frame["budget_held_protected_count"].mean()) if "budget_held_protected_count" in turnover_frame.columns else 0.0
        metrics["avg_budget_split_bound_guard_count"] = float(turnover_frame["budget_split_bound_guard_count"].mean()) if "budget_split_bound_guard_count" in turnover_frame.columns else 0.0
        metrics["avg_budget_model_value_arbitration_signal"] = float(turnover_frame["budget_model_value_arbitration_signal"].mean()) if "budget_model_value_arbitration_signal" in turnover_frame.columns else 0.0
        metrics["avg_budget_model_alpha_opportunity_signal"] = float(turnover_frame["budget_model_alpha_opportunity_signal"].mean()) if "budget_model_alpha_opportunity_signal" in turnover_frame.columns else 0.0
        metrics["avg_budget_model_cash_defense_signal"] = float(turnover_frame["budget_model_cash_defense_signal"].mean()) if "budget_model_cash_defense_signal" in turnover_frame.columns else 0.0
        metrics["avg_budget_model_deploy_value_signal"] = float(turnover_frame["budget_model_deploy_value_signal"].mean()) if "budget_model_deploy_value_signal" in turnover_frame.columns else 0.0
        metrics["avg_budget_model_release_value_signal"] = float(turnover_frame["budget_model_release_value_signal"].mean()) if "budget_model_release_value_signal" in turnover_frame.columns else 0.0
        metrics["avg_budget_model_defense_value_signal"] = float(turnover_frame["budget_model_defense_value_signal"].mean()) if "budget_model_defense_value_signal" in turnover_frame.columns else 0.0
        metrics["avg_budget_model_deploy_gate_signal"] = float(turnover_frame["budget_model_deploy_gate_signal"].mean()) if "budget_model_deploy_gate_signal" in turnover_frame.columns else 0.0
        metrics["avg_budget_model_release_gate_signal"] = float(turnover_frame["budget_model_release_gate_signal"].mean()) if "budget_model_release_gate_signal" in turnover_frame.columns else 0.0
        metrics["avg_budget_model_defense_gate_signal"] = float(turnover_frame["budget_model_defense_gate_signal"].mean()) if "budget_model_defense_gate_signal" in turnover_frame.columns else 0.0
    else:
        metrics["avg_turnover"] = 0.0
        metrics["avg_buy_turnover"] = 0.0
        metrics["avg_sell_turnover"] = 0.0
        metrics["avg_gross_exposure"] = 0.0
        metrics["avg_holding_count"] = 0.0
        metrics["avg_semantic_conflict_rate"] = 0.0
        metrics["avg_order_translation_conflict_rate"] = 0.0
        metrics["avg_budget_risk_off_score"] = 0.0
        metrics["avg_budget_deploy_score"] = 0.0
        metrics["avg_budget_entry_candidate_count"] = 0.0
        metrics["avg_budget_entry_keep_count"] = 0.0
        metrics["avg_budget_held_protected_count"] = 0.0
        metrics["avg_budget_split_bound_guard_count"] = 0.0
        metrics["avg_budget_model_value_arbitration_signal"] = 0.0
        metrics["avg_budget_model_alpha_opportunity_signal"] = 0.0
        metrics["avg_budget_model_cash_defense_signal"] = 0.0
        metrics["avg_budget_model_deploy_value_signal"] = 0.0
        metrics["avg_budget_model_release_value_signal"] = 0.0
        metrics["avg_budget_model_defense_value_signal"] = 0.0
        metrics["avg_budget_model_deploy_gate_signal"] = 0.0
        metrics["avg_budget_model_release_gate_signal"] = 0.0
        metrics["avg_budget_model_defense_gate_signal"] = 0.0
    metrics["action_counts"] = {
        str(key): int(value)
        for key, value in action_panel["execution_action"].astype(str).value_counts().sort_index().items()
    } if not action_panel.empty else {}
    metrics["weight_change_action_counts"] = {
        str(key): int(value)
        for key, value in action_panel.get("weight_change_action", action_panel.get("execution_action", pd.Series(dtype=str))).astype(str).value_counts().sort_index().items()
    } if not action_panel.empty else {}
    continuity_metrics, action_outcomes = compute_continuity_metrics(
        prepared=prepared,
        action_panel=action_panel,
        turnover_frame=turnover_frame,
        returns_series=returns_series,
    )
    return {
        "dates": [dt.strftime("%Y-%m-%d") for dt in dates],
        "execution_semantics": str(execution_semantics),
        "budget_semantics": str(budget_semantics),
        "budget_calibration": str(budget_calibration),
        "budget_objective": str(resolved_budget_objective),
        "returns": returns_series,
        "monthly_returns": monthly_returns,
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
    columns = [
        "stock",
        "price",
        "current_weight",
        "target_weight",
        "current_shares",
        "target_shares",
        "delta_shares",
        "share_action",
    ]
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
    return pd.DataFrame(rows, columns=columns)


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
