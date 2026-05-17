from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from daily_research.continuous_policy.pipeline_utils import compute_curve_metrics, estimate_trading_cost
from daily_research.continuous_policy.portfolio_simulator import (
    BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
    BUDGET_SEMANTICS_ALLOCATION_LAYER,
    DEFAULT_EXECUTION_SEMANTICS,
    PortfolioState,
)
from daily_research.path_policy.adapter import build_path_policy_frame
from daily_research.path_policy.rl_dataset import Path20TrajectoryDataset


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(float(default)).astype(float)


def _diagnose_projection(raw_target: pd.Series, projected_target: pd.Series, current_weight: pd.Series) -> dict[str, float]:
    aligned_raw = raw_target.reindex(projected_target.index).fillna(0.0).astype(float)
    aligned_current = current_weight.reindex(projected_target.index).fillna(0.0).astype(float)
    return {
        "raw_gross_exposure": float(aligned_raw.clip(lower=0.0).sum()),
        "projected_gross_exposure": float(projected_target.clip(lower=0.0).sum()),
        "raw_turnover": float((aligned_raw - aligned_current).abs().sum()),
        "projected_turnover": float((projected_target - aligned_current).abs().sum()),
        "projection_l1_distance": float((projected_target - aligned_raw).abs().sum()),
    }


def run_sequence_policy_replay(
    *,
    trajectory: Path20TrajectoryDataset,
    target_weight_by_date: dict[str, pd.Series],
    execution_price_frame: pd.DataFrame,
    transaction_cost_bps: float = 3.0,
    slippage_bps: float = 7.0,
    sell_tax_bps: float = 10.0,
    max_position_weight: float = 0.10,
    max_gross_exposure: float = 0.95,
    max_positions: int = 30,
    turnover_budget: float = 1.0,
    source_label: str = "alpha_path20_sequence_policy_v1",
) -> dict[str, Any]:
    if trajectory.empty:
        return {
            "dates": [],
            "returns": pd.Series(dtype=float),
            "metrics": compute_curve_metrics(pd.Series(dtype=float)),
            "action_panel": pd.DataFrame(),
            "turnover_frame": pd.DataFrame(),
            "position_history": pd.DataFrame(),
            "projection_diagnostics": pd.DataFrame(),
            "attribution_frame": pd.DataFrame(),
        }
    portfolio = PortfolioState(max_positions=int(max_positions), max_position_weight=float(max_position_weight))
    action_rows: list[dict[str, Any]] = []
    turnover_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    returns: list[float] = []
    return_dates: list[str] = []
    dates = trajectory.dates
    for idx, dt in enumerate(dates):
        date_text = pd.Timestamp(dt).strftime("%Y-%m-%d")
        daily = trajectory.daily_frames[idx].copy()
        daily.index = daily["stock"].astype(str)
        current = portfolio.current_weights(daily.index)
        daily["current_weight"] = current.reindex(daily.index).fillna(0.0).astype(float)
        raw_target = target_weight_by_date.get(date_text)
        if raw_target is None:
            raw_target = pd.Series(0.0, index=daily.index, dtype=float)
        else:
            raw_target = raw_target.copy()
            raw_target.index = raw_target.index.map(str)
            raw_target = raw_target.reindex(daily.index).fillna(0.0).astype(float)
        tradable = _numeric(daily, "tradable_mask", 0.0) > 0.5
        policy_frame, global_targets = build_path_policy_frame(
            daily,
            raw_target_weight=raw_target,
            current_weight=current,
            tradable_mask=tradable,
            max_position_weight=max_position_weight,
            max_gross_exposure=max_gross_exposure,
            max_positions=max_positions,
            turnover_budget=turnover_budget,
            source_label=source_label,
        )
        target = _numeric(policy_frame, "portfolio_daily_target_weight", 0.0)
        projection_diagnostics = {
            "date": date_text,
            **_diagnose_projection(raw_target, target, current),
            **{key: value for key, value in global_targets.items() if str(key).startswith("path_policy_")},
        }
        projection_rows.append(projection_diagnostics)
        step_result = portfolio.step(
            date=dt,
            prices=execution_price_frame.loc[dt],
            policy_frame=policy_frame,
            global_targets=global_targets,
            source_label=source_label,
            execution_semantics=DEFAULT_EXECUTION_SEMANTICS,
            budget_semantics=BUDGET_SEMANTICS_ALLOCATION_LAYER,
            budget_calibration=BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
        )
        action_rows.extend(step_result.actions)
        turnover_rows.append({"date": step_result.date, **step_result.diagnostics, **global_targets})
        position_rows.append({"date": step_result.date, **portfolio.snapshot()})
        if idx + 1 < len(dates):
            next_dt = dates[idx + 1]
            next_returns = execution_price_frame.loc[next_dt].div(execution_price_frame.loc[dt]).sub(1.0)
            next_returns = next_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            gross_return = float(step_result.weights.reindex(next_returns.index).fillna(0.0).mul(next_returns).sum())
            estimated_cost = estimate_trading_cost(
                diagnostics=step_result.diagnostics,
                transaction_cost_bps=transaction_cost_bps,
                slippage_bps=slippage_bps,
                sell_tax_bps=sell_tax_bps,
            )
            net_return = gross_return - estimated_cost
            benchmark_return = float(_numeric(daily, "benchmark_return", 0.0).iloc[0]) if len(daily) else 0.0
            attribution_rows.append(
                {
                    "signal_date": date_text,
                    "return_date": pd.Timestamp(next_dt).strftime("%Y-%m-%d"),
                    "gross_return": float(gross_return),
                    "estimated_cost": float(estimated_cost),
                    "net_return": float(net_return),
                    "benchmark_return": float(benchmark_return),
                    "excess_return": float(net_return - benchmark_return),
                    "cash_weight": float(projection_diagnostics.get("path_policy_cash_weight", max(0.0, 1.0 - float(projection_diagnostics.get("projected_gross_exposure", 0.0))))),
                    "raw_gross_exposure": float(projection_diagnostics.get("raw_gross_exposure", 0.0)),
                    "projected_gross_exposure": float(projection_diagnostics.get("projected_gross_exposure", 0.0)),
                    "raw_turnover": float(projection_diagnostics.get("raw_turnover", 0.0)),
                    "projected_turnover": float(projection_diagnostics.get("projected_turnover", 0.0)),
                    "projection_l1_distance": float(projection_diagnostics.get("projection_l1_distance", 0.0)),
                    "target_count": float(projection_diagnostics.get("path_policy_target_count", 0.0)),
                }
            )
            returns.append(net_return)
            return_dates.append(pd.Timestamp(next_dt).strftime("%Y-%m-%d"))
            portfolio.record_realized_return(net_return)
    returns_series = pd.Series(returns, index=return_dates, dtype=float)
    metrics = compute_curve_metrics(returns_series)
    metrics["return_count"] = float(len(returns_series))
    metrics["signal_date_count"] = float(len(dates))
    projection_frame = pd.DataFrame(projection_rows)
    if not projection_frame.empty:
        metrics["avg_projection_l1_distance"] = float(pd.to_numeric(projection_frame["projection_l1_distance"], errors="coerce").fillna(0.0).mean())
        metrics["avg_projected_gross_exposure"] = float(pd.to_numeric(projection_frame["projected_gross_exposure"], errors="coerce").fillna(0.0).mean())
        metrics["avg_projected_turnover"] = float(pd.to_numeric(projection_frame["projected_turnover"], errors="coerce").fillna(0.0).mean())
        metrics["avg_raw_turnover"] = float(pd.to_numeric(projection_frame["raw_turnover"], errors="coerce").fillna(0.0).mean())
    turnover_frame = pd.DataFrame(turnover_rows)
    if not turnover_frame.empty and "realized_turnover" in turnover_frame.columns:
        metrics["avg_turnover"] = float(pd.to_numeric(turnover_frame["realized_turnover"], errors="coerce").fillna(0.0).mean())
    elif not projection_frame.empty and "projected_turnover" in projection_frame.columns:
        metrics["avg_turnover"] = float(pd.to_numeric(projection_frame["projected_turnover"], errors="coerce").fillna(0.0).mean())
    attribution_frame = pd.DataFrame(attribution_rows)
    if not attribution_frame.empty:
        metrics["gross_return_sum"] = float(pd.to_numeric(attribution_frame["gross_return"], errors="coerce").fillna(0.0).sum())
        metrics["estimated_cost_sum"] = float(pd.to_numeric(attribution_frame["estimated_cost"], errors="coerce").fillna(0.0).sum())
        metrics["excess_return_sum"] = float(pd.to_numeric(attribution_frame["excess_return"], errors="coerce").fillna(0.0).sum())
        metrics["avg_cash_weight"] = float(pd.to_numeric(attribution_frame["cash_weight"], errors="coerce").fillna(0.0).mean())
        metrics["avg_estimated_cost"] = float(pd.to_numeric(attribution_frame["estimated_cost"], errors="coerce").fillna(0.0).mean())
        metrics["avg_benchmark_return"] = float(pd.to_numeric(attribution_frame["benchmark_return"], errors="coerce").fillna(0.0).mean())
    return {
        "dates": [pd.Timestamp(dt).strftime("%Y-%m-%d") for dt in dates],
        "returns": returns_series,
        "metrics": metrics,
        "action_panel": pd.DataFrame(action_rows),
        "turnover_frame": turnover_frame,
        "position_history": pd.DataFrame(position_rows),
        "projection_diagnostics": projection_frame,
        "attribution_frame": attribution_frame,
    }
