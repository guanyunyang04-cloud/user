from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from daily_research.continuous_policy.pipeline_utils import compute_curve_metrics, estimate_trading_cost, signal_dates_between
from daily_research.continuous_policy.portfolio_simulator import (
    BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
    BUDGET_SEMANTICS_ALLOCATION_LAYER,
    DEFAULT_EXECUTION_SEMANTICS,
    PortfolioState,
)
from daily_research.continuous_policy.state_builder import PreparedPolicyInputs, build_cross_section_state
from daily_research.path_policy.adapter import build_path_policy_frame
from daily_research.path_policy.labels import (
    PATH20_HORIZON,
    Path20LabelBundle,
    build_path20_labels,
    path20_label_frame_for_date,
)


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(float(default)).astype(float)


def build_oracle_target_weights(
    label_frame: pd.DataFrame,
    *,
    current_weight: pd.Series,
    max_position_weight: float = 0.20,
    max_gross_exposure: float = 0.95,
    max_positions: int = 50,
    score_column: str = "future_cum_excess_return_20d",
    downside_column: str = "future_path_max_drawdown_20d",
    risk_penalty: float = 0.35,
) -> pd.Series:
    working = label_frame.copy()
    if "stock" in working.columns:
        working.index = working["stock"].astype(str)
    score = _numeric(working, score_column, 0.0)
    downside = _numeric(working, downside_column, 0.0)
    adjusted = score + float(risk_penalty) * downside.clip(upper=0.0)
    adjusted = adjusted.where(adjusted > 0.0, 0.0)
    if int((adjusted > 0.0).sum()) == 0:
        return pd.Series(0.0, index=working.index, dtype=float)
    if max_positions > 0:
        keep = adjusted.sort_values(ascending=False).head(int(max_positions)).index
        adjusted = adjusted.where(adjusted.index.isin(keep), 0.0)
    weights = adjusted / max(float(adjusted.sum()), 1.0e-12) * float(max_gross_exposure)
    weights = weights.clip(lower=0.0, upper=float(max_position_weight))
    total = float(weights.sum())
    if total > float(max_gross_exposure) and total > 1.0e-12:
        weights = weights / total * float(max_gross_exposure)
    weights = weights.where(weights > 1.0e-12, 0.0)
    return weights.reindex(current_weight.index).fillna(0.0).astype(float)


def build_oracle_policy_for_date(
    *,
    prepared: PreparedPolicyInputs,
    labels: Path20LabelBundle,
    date: pd.Timestamp,
    portfolio: PortfolioState,
    max_position_weight: float = 0.20,
    max_gross_exposure: float = 0.95,
    max_positions: int = 50,
    turnover_budget: float = 1.00,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    state_frame = build_cross_section_state(prepared, date=date, portfolio_state=portfolio)
    state_frame = state_frame.copy()
    state_frame.index = state_frame["stock"].astype(str)
    label_frame = path20_label_frame_for_date(labels, date)
    label_frame.index = label_frame["stock"].astype(str)
    current = _numeric(state_frame, "current_weight", 0.0)
    target_weight = build_oracle_target_weights(
        label_frame,
        current_weight=current,
        max_position_weight=max_position_weight,
        max_gross_exposure=max_gross_exposure,
        max_positions=max_positions,
    )
    membership = prepared.membership_frame.reindex(index=[date], columns=prepared.close.columns).iloc[0].fillna(False).astype(bool)
    policy, global_targets = build_path_policy_frame(
        state_frame,
        raw_target_weight=target_weight,
        current_weight=current,
        tradable_mask=membership,
        max_position_weight=max_position_weight,
        max_gross_exposure=max_gross_exposure,
        max_positions=max_positions,
        turnover_budget=turnover_budget,
        source_label="oracle_path20_future",
    )
    for column in label_frame.columns:
        if column != "stock":
            policy[f"oracle_path20_{column}"] = label_frame[column].reindex(policy.index)
    return policy, global_targets


def run_oracle_path20_rollout(
    *,
    prepared: PreparedPolicyInputs,
    start_date: str,
    end_date: str = "",
    execution_mode: str = "next_open",
    transaction_cost_bps: float = 3.0,
    slippage_bps: float = 7.0,
    sell_tax_bps: float = 10.0,
    max_position_weight: float = 0.20,
    max_gross_exposure: float = 0.95,
    max_positions: int = 50,
    turnover_budget: float = 1.00,
) -> dict[str, Any]:
    labels = build_path20_labels(prepared, execution_mode=execution_mode)
    dates = signal_dates_between(prepared, start_date=start_date, end_date=end_date, max_forward_horizon=PATH20_HORIZON)
    if len(dates) < 2:
        raise ValueError("oracle path20 rollout requires at least two signal dates.")
    portfolio = PortfolioState(max_positions=int(max_positions), max_position_weight=float(max_position_weight))
    action_rows: list[dict[str, Any]] = []
    turnover_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    returns: list[float] = []
    oracle_rows: list[dict[str, Any]] = []
    for idx, signal_dt in enumerate(dates):
        policy_frame, global_targets = build_oracle_policy_for_date(
            prepared=prepared,
            labels=labels,
            date=signal_dt,
            portfolio=portfolio,
            max_position_weight=max_position_weight,
            max_gross_exposure=max_gross_exposure,
            max_positions=max_positions,
            turnover_budget=turnover_budget,
        )
        execution_price_frame = (
            prepared.open_.shift(-1)
            if str(execution_mode or "next_open").strip().lower() == "next_open"
            else prepared.close
        )
        step_result = portfolio.step(
            date=signal_dt,
            prices=execution_price_frame.loc[signal_dt],
            policy_frame=policy_frame,
            global_targets=global_targets,
            source_label="oracle_path20_future",
            execution_semantics=DEFAULT_EXECUTION_SEMANTICS,
            budget_semantics=BUDGET_SEMANTICS_ALLOCATION_LAYER,
            budget_calibration=BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
        )
        action_rows.extend(step_result.actions)
        turnover_rows.append({"date": step_result.date, **step_result.diagnostics, **global_targets})
        position_rows.append({"date": step_result.date, **portfolio.snapshot()})
        target_weight = _numeric(policy_frame, "portfolio_daily_target_weight", 0.0)
        oracle_rows.append(
            {
                "date": signal_dt.strftime("%Y-%m-%d"),
                "oracle_target_count": int((target_weight > 1.0e-8).sum()),
                "oracle_target_weight_sum": float(target_weight.sum()),
                "oracle_top20_mean_excess": float(_numeric(policy_frame, "oracle_path20_future_cum_excess_return_20d", 0.0).loc[target_weight > 1.0e-8].mean())
                if bool((target_weight > 1.0e-8).any())
                else 0.0,
            }
        )
        if idx + 1 < len(dates):
            next_dt = dates[idx + 1]
            next_returns = (
                execution_price_frame.loc[next_dt]
                .div(execution_price_frame.loc[signal_dt])
                .sub(1.0)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
            gross_return = float(step_result.weights.reindex(next_returns.index).fillna(0.0).mul(next_returns).sum())
            net_return = gross_return - estimate_trading_cost(
                diagnostics=step_result.diagnostics,
                transaction_cost_bps=transaction_cost_bps,
                slippage_bps=slippage_bps,
                sell_tax_bps=sell_tax_bps,
            )
            returns.append(net_return)
            portfolio.record_realized_return(net_return)
    returns_series = pd.Series(returns, index=[dt.strftime("%Y-%m-%d") for dt in dates[1:]], dtype=float)
    turnover_frame = pd.DataFrame(turnover_rows)
    metrics = compute_curve_metrics(returns_series)
    metrics["return_count"] = float(len(returns_series))
    metrics["signal_date_count"] = float(len(dates))
    if not turnover_frame.empty:
        metrics["avg_turnover"] = float(pd.to_numeric(turnover_frame["realized_turnover"], errors="coerce").fillna(0.0).mean())
        metrics["avg_gross_exposure"] = float(1.0 - pd.to_numeric(turnover_frame["cash_weight"], errors="coerce").fillna(1.0).mean())
        metrics["avg_holding_count"] = float(pd.to_numeric(turnover_frame["holding_count"], errors="coerce").fillna(0.0).mean())
        native_target_valid = (
            turnover_frame["native_target_valid"]
            if "native_target_valid" in turnover_frame.columns
            else pd.Series(0.0, index=turnover_frame.index, dtype=float)
        )
        allocation_layer_native_target_used = (
            turnover_frame["allocation_layer_native_target_used"]
            if "allocation_layer_native_target_used" in turnover_frame.columns
            else pd.Series(0.0, index=turnover_frame.index, dtype=float)
        )
        metrics["native_target_valid_rate"] = float(pd.to_numeric(native_target_valid, errors="coerce").fillna(0.0).mean())
        metrics["allocation_layer_native_target_used_rate"] = float(
            pd.to_numeric(allocation_layer_native_target_used, errors="coerce").fillna(0.0).mean()
        )
    return {
        "dates": [dt.strftime("%Y-%m-%d") for dt in dates],
        "returns": returns_series,
        "metrics": metrics,
        "action_panel": pd.DataFrame(action_rows),
        "turnover_frame": turnover_frame,
        "position_history": pd.DataFrame(position_rows),
        "oracle_daily": pd.DataFrame(oracle_rows),
    }
