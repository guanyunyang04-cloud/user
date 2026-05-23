from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from daily_research.continuous_policy.decision_core_v6 import (
    DECISION_CORE_V6_DEADBAND,
    DECISION_CORE_V6_PROFILE,
    DECISION_CORE_V6_STRICT_GOLD_DATASET_ID,
    DECISION_CORE_V6_VERSION,
    build_decision_core_v6_targets,
)
from daily_research.continuous_policy.label_builder import (
    LABEL_CONFIGS,
    build_action_labels_for_date,
    build_future_path_metrics,
    build_teacher_global_targets,
)
from daily_research.continuous_policy.model import load_artifact, predict_policy
from daily_research.continuous_policy.pipeline_utils import (
    BUDGET_OBJECTIVE_CHOICES,
    DEFAULT_BUDGET_OBJECTIVE,
    compute_continuity_metrics,
    compute_curve_metrics,
    estimate_trading_cost,
    run_policy_rollout,
    signal_dates_between,
)
from daily_research.continuous_policy.portfolio_cashflow_decision import PORTFOLIO_CASHFLOW_DECISION_MODE_COLUMN
from daily_research.continuous_policy.portfolio_simulator import (
    BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
    BUDGET_CALIBRATION_CHOICES,
    BUDGET_SEMANTICS_ALLOCATION_LAYER,
    BUDGET_SEMANTICS_CHOICES,
    DEFAULT_EXECUTION_SEMANTICS,
    EXECUTION_SEMANTICS_CHOICES,
    PortfolioState,
)
from daily_research.continuous_policy.runtime import STUDIES_ROOT, now_iso, safe_print_json, timestamp_tag, write_json
from daily_research.continuous_policy.state_builder import DEFAULT_ALPHA_PRIOR_SOURCE, build_cross_section_state, build_daily_state_features, prepare_policy_inputs, resolve_active_policy_defaults
from daily_research.data_lake import DEFAULT_POLICY_INPUT_LAKE_DATASET_ID


DIAGNOSTIC_KEYS: tuple[str, ...] = (
    "total_return",
    "annual_return",
    "sharpe",
    "max_drawdown",
    "avg_turnover",
    "avg_gross_exposure",
    "avg_holding_count",
    "cashflow_decision_contract_valid_rate",
    "intent_translation_conflict_rate",
    "decision_oracle_constraint_violation_mean",
    "feature_contract_blocker_rate",
    "feature_contract_degraded_rate",
    "portfolio_daily_source_target_count",
    "portfolio_daily_receiver_target_count",
    "portfolio_daily_source_sell_count",
    "portfolio_daily_source_positive_forward_sell_share",
    "portfolio_daily_receiver_forward_excess_5d",
    "portfolio_daily_source_forward_excess_5d",
    "portfolio_daily_receiver_minus_source_forward_excess_5d",
    "immediate_reversal_rate_3d",
    "cash_timing_quality_1d",
    "native_target_valid",
    "native_target_constraint_violations",
    "allocation_layer_native_target_used",
    "allocation_layer_native_fallback_used",
    "allocation_layer_primary_mode",
    "portfolio_cashflow_decision_v1_mode_used",
    "cashflow_decision_used",
    "cashflow_decision_failed_closed",
)

V6_EXECUTION_PATH_MEAN_FLOOR = 0.999
V6_EXECUTION_PARTIAL_PATH_MEAN_FLOOR = 0.90
TARGET_PROFIT_TOTAL_RETURN_FLOOR = 0.0
TARGET_PROFIT_SHARPE_FLOOR = 0.0
MODEL_RETURN_GAP_LARGE = -0.02
MODEL_ACTION_MATCH_WEAK_FLOOR = 0.85


def build_parser() -> argparse.ArgumentParser:
    defaults = resolve_active_policy_defaults()
    parser = argparse.ArgumentParser(description="Diagnose v6 target upper-bound, model imitation gap, and simulator/execution gap.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--pool-name", default=defaults["pool_name"] or "liquid500")
    parser.add_argument("--start-date", default=defaults["start_date"] or "20250318")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--benchmark", default=defaults["benchmark"] or "000300.SH")
    parser.add_argument("--data-source", default="lake", choices=("lake",))
    parser.add_argument("--csv-folder", default="")
    parser.add_argument("--lake-dataset-id", default=DEFAULT_POLICY_INPUT_LAKE_DATASET_ID)
    parser.add_argument("--data-lake-root", default="")
    parser.add_argument("--training-dataset-id", default=DECISION_CORE_V6_STRICT_GOLD_DATASET_ID)
    parser.add_argument("--pool-rebalance-days", type=int, default=21)
    parser.add_argument("--pool-adv-window", type=int, default=20)
    parser.add_argument("--max-universe-size", type=int, default=0)
    parser.add_argument("--transaction-cost-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=7.0)
    parser.add_argument("--sell-tax-bps", type=float, default=10.0)
    parser.add_argument("--execution-semantics", default=DEFAULT_EXECUTION_SEMANTICS, choices=EXECUTION_SEMANTICS_CHOICES)
    parser.add_argument("--budget-semantics", default=BUDGET_SEMANTICS_ALLOCATION_LAYER, choices=BUDGET_SEMANTICS_CHOICES)
    parser.add_argument("--budget-calibration", default=BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER, choices=BUDGET_CALIBRATION_CHOICES)
    parser.add_argument("--budget-objective", default=DEFAULT_BUDGET_OBJECTIVE, choices=BUDGET_OBJECTIVE_CHOICES)
    parser.add_argument("--label-preset", default="balanced_v2", choices=tuple(sorted(LABEL_CONFIGS)))
    parser.add_argument("--alpha-prior-source", default=DEFAULT_ALPHA_PRIOR_SOURCE)
    parser.add_argument("--alpha-prior-score-panel", default="")
    parser.add_argument("--alpha-prior-target-weight-panel", default="")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--allow-non-v6-execution-path",
        action="store_true",
        help="Allow diagnostics to run without the v6 allocation-layer execution path. Intended only for explicit execution-gap counterfactuals.",
    )
    parser.add_argument("--tag", default="")
    return parser


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def _series_numeric(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(float(default)).astype(float)


def _v6_target_policy_frame(state_frame: pd.DataFrame, label_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    sample_frame = state_frame.merge(
        label_frame.drop(columns=[column for column in ("current_weight", "holding_flag") if column in label_frame.columns]),
        how="left",
        on=["date", "stock"],
        suffixes=("", "_label"),
    )
    targets = build_decision_core_v6_targets(sample_frame)
    working = sample_frame.copy()
    for column in targets.columns:
        working[column] = targets[column].to_numpy()
    working = working.set_index("stock", drop=False)
    target_weight = _series_numeric(working, "target_weight", 0.0)
    target_delta = _series_numeric(working, "target_delta", 0.0)
    source_supply = _series_numeric(working, "source_supply", 0.0)
    receiver_demand = _series_numeric(working, "receiver_demand", 0.0)
    cash_reserve = _series_numeric(working, "cash_reserve", 0.0)
    current = _series_numeric(working, "current_weight", 0.0)
    source_release_intent = (source_supply / current.clip(lower=DECISION_CORE_V6_DEADBAND)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
    policy = working.copy()
    policy["decision_core_version"] = DECISION_CORE_V6_VERSION
    policy["decision_core_v6_source_supply"] = source_supply.astype(float)
    policy["decision_core_v6_receiver_demand"] = receiver_demand.astype(float)
    policy["decision_core_v6_cash_reserve"] = cash_reserve.astype(float)
    policy["decision_core_v6_target_weight"] = target_weight.astype(float)
    policy["decision_core_v6_target_delta"] = target_delta.astype(float)
    policy["decision_core_v6_intent"] = policy.get("intent", pd.Series("hold", index=policy.index)).astype(str)
    policy["decision_core_v6_reason_code"] = policy.get("reason_code", pd.Series("neutral", index=policy.index)).astype(str)
    policy["decision_core_v6_contract_status"] = policy.get("contract_status", pd.Series("ok", index=policy.index)).astype(str)
    policy["decision_core_v6_contract_blocker_reason"] = policy.get("contract_blocker_reason", pd.Series("none", index=policy.index)).astype(str)
    policy["decision_core_v6_contract_valid"] = (policy["decision_core_v6_contract_status"] != "blocker").astype(float)
    policy["decision_core_v6_oracle_constraint_violation"] = _series_numeric(policy, "constraint_violation", 0.0)
    policy["decision_core_v6_feature_contract_blocker_count"] = (policy["decision_core_v6_contract_status"] == "blocker").astype(float)
    policy["decision_core_v6_feature_contract_degraded_count"] = (policy["decision_core_v6_contract_status"] == "degraded").astype(float)
    policy["decision_core_v6_feature_contract_neutral_fallback_count"] = 0.0
    policy["decision_core_v6_source_weakness"] = _series_numeric(policy, "source_weakness", 0.0)
    policy["decision_core_v6_source_keep_strength"] = _series_numeric(policy, "source_keep_strength", 0.0)
    policy["decision_core_v6_receiver_quality"] = _series_numeric(policy, "receiver_quality", 0.0)
    policy["decision_core_v6_deploy_value"] = _series_numeric(policy, "deploy_value", 0.0)
    policy["decision_core_v6_release_value"] = _series_numeric(policy, "release_value", 0.0)
    policy["decision_core_v6_defense_value"] = _series_numeric(policy, "defense_value", 0.0)
    policy["decision_core_v6_cash_timing_value"] = _series_numeric(policy, "cash_timing_value", 0.0)
    policy["decision_core_v6_source_opportunity_cost"] = _series_numeric(policy, "source_opportunity_cost", 0.0)
    policy["decision_core_v6_reversal_risk_penalty"] = _series_numeric(policy, "reversal_risk_penalty", 0.0)
    policy["decision_core_v6_source_wrong_side_sell_penalty"] = _series_numeric(policy, "source_wrong_side_sell_penalty", 0.0)
    policy["decision_core_v6_receiver_source_spread_value"] = _series_numeric(policy, "receiver_source_spread_value", 0.0)
    policy["decision_core_v6_release_utility"] = _series_numeric(policy, "release_utility", 0.0)
    policy["decision_core_v6_release_gate"] = _series_numeric(policy, "release_gate", 0.0)
    policy["decision_core_v6_source_release_evidence"] = _series_numeric(policy, "source_release_evidence", 0.0)
    policy["decision_core_v6_source_false_sell_penalty"] = _series_numeric(policy, "source_false_sell_penalty", 0.0)
    policy["portfolio_daily_target_weight_intent"] = target_weight.astype(float)
    policy["portfolio_daily_target_delta_intent"] = target_delta.astype(float)
    policy["portfolio_daily_target_weight"] = target_weight.astype(float)
    policy["target_delta_hint"] = target_delta.astype(float)
    policy["portfolio_set_v5_source_supply"] = source_supply.astype(float)
    policy["portfolio_set_v5_receiver_demand"] = receiver_demand.astype(float)
    policy["portfolio_set_v5_source_supply_score"] = _series_numeric(policy, "source_supply_score", 0.0)
    policy["portfolio_set_v5_receiver_demand_score"] = _series_numeric(policy, "receiver_demand_score", 0.0)
    policy["portfolio_set_v5_cash_buffer_score"] = cash_reserve.astype(float)
    policy["portfolio_set_v5_oracle_feasible"] = (_series_numeric(policy, "constraint_violation", 0.0) <= 1.0e-6).astype(float)
    policy["portfolio_set_v5_turnover_budget"] = _series_numeric(policy, "turnover_budget", 0.08)
    policy[PORTFOLIO_CASHFLOW_DECISION_MODE_COLUMN] = 1.0
    policy["allocation_intent_v2_mode"] = 1.0
    policy["release_first_allocation_v3_mode"] = 1.0
    policy["portfolio_daily_release_first_intent"] = source_release_intent.astype(float)
    policy["release_first_intent_score"] = source_release_intent.astype(float)
    policy["release_first_intent_delta"] = (-source_supply).where(target_delta < -DECISION_CORE_V6_DEADBAND, 0.0)
    policy["release_first_action_hint"] = "hold"
    policy.loc[policy["decision_core_v6_intent"].isin(["reduce", "exit"]), "release_first_action_hint"] = policy.loc[
        policy["decision_core_v6_intent"].isin(["reduce", "exit"]),
        "decision_core_v6_intent",
    ].astype(str)
    policy["release_first_block_reason"] = "none"
    policy.loc[current <= DECISION_CORE_V6_DEADBAND, "release_first_block_reason"] = "not_held"
    policy["action_label"] = policy["decision_core_v6_intent"].astype(str)
    policy["action_strength"] = (source_supply + receiver_demand).clip(0.0, 1.0)
    policy["portfolio_daily_source_score"] = _series_numeric(policy, "source_supply_score", 0.0)
    policy["portfolio_daily_unified_source_score"] = policy["portfolio_daily_source_score"]
    policy["portfolio_daily_receiver_score"] = _series_numeric(policy, "receiver_demand_score", 0.0)
    policy["portfolio_daily_unified_receiver_score"] = policy["portfolio_daily_receiver_score"]
    policy["portfolio_daily_cash_score"] = cash_reserve.astype(float)
    policy["portfolio_daily_unified_cash_score"] = cash_reserve.astype(float)
    policy["portfolio_daily_source_release_quality"] = source_release_intent.astype(float)
    policy["portfolio_daily_source_release_capacity"] = source_release_intent.astype(float)
    policy["portfolio_daily_source_release_preference"] = source_release_intent.astype(float)
    policy["portfolio_daily_source_economic_release_score"] = source_release_intent.astype(float)
    policy["portfolio_daily_source_executability"] = (source_supply > DECISION_CORE_V6_DEADBAND).astype(float)
    policy["portfolio_daily_receiver_executability"] = (receiver_demand > DECISION_CORE_V6_DEADBAND).astype(float)
    policy["portfolio_daily_receiver_add_headroom"] = (0.24 - current).clip(lower=0.0)
    policy["portfolio_daily_receiver_add_capacity"] = policy["portfolio_daily_receiver_add_headroom"]
    global_targets = build_teacher_global_targets(label_frame, label_config="balanced_v2")
    global_targets.update(
        {
            "decision_core_version": DECISION_CORE_V6_VERSION,
            "decision_core_v6_mode": 1.0,
            "release_first_allocation_v3_mode": 1.0,
            "allocation_intent_v2_mode": 1.0,
            PORTFOLIO_CASHFLOW_DECISION_MODE_COLUMN: 1.0,
            "turnover_budget": float(np.clip(_series_numeric(policy, "turnover_budget", 0.08).median(), 0.04, 0.08)),
            "max_position_weight_target": 0.24,
        }
    )
    return policy, global_targets


def _run_v6_target_replay(
    *,
    prepared: Any,
    start_date: str,
    end_date: str,
    label_preset: str,
    execution_semantics: str,
    budget_semantics: str,
    budget_calibration: str,
    transaction_cost_bps: float,
    slippage_bps: float,
    sell_tax_bps: float,
) -> dict[str, Any]:
    future_metrics = build_future_path_metrics(prepared)
    dates = signal_dates_between(prepared, start_date=start_date, end_date=end_date, max_forward_horizon=0)
    if len(dates) < 2:
        raise ValueError("v6 target replay requires at least two signal dates.")
    portfolio = PortfolioState()
    action_rows: list[dict[str, Any]] = []
    turnover_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    returns: list[float] = []
    target_rows: list[dict[str, Any]] = []
    for idx, signal_dt in enumerate(dates):
        state_frame = build_cross_section_state(prepared, date=signal_dt, portfolio_state=portfolio)
        label_frame = build_action_labels_for_date(
            date=signal_dt,
            state_frame=state_frame,
            future_metrics=future_metrics,
            label_config=label_preset,
        )
        policy_frame, global_targets = _v6_target_policy_frame(state_frame, label_frame)
        target_rows.append(
            {
                "date": signal_dt.strftime("%Y-%m-%d"),
                "target_source_count": float((_series_numeric(policy_frame, "portfolio_set_v5_source_supply", 0.0) > DECISION_CORE_V6_DEADBAND).sum()),
                "target_receiver_count": float((_series_numeric(policy_frame, "portfolio_set_v5_receiver_demand", 0.0) > DECISION_CORE_V6_DEADBAND).sum()),
                "target_turnover": float((_series_numeric(policy_frame, "portfolio_daily_target_delta_intent", 0.0)).abs().sum()),
                "target_gross": float(_series_numeric(policy_frame, "portfolio_daily_target_weight_intent", 0.0).sum()),
                "target_constraint_violation": float(_series_numeric(policy_frame, "constraint_violation", 0.0).max()),
                "target_feature_blocker_count": float((policy_frame["decision_core_v6_contract_status"].astype(str) == "blocker").sum()),
                "target_feature_degraded_count": float((policy_frame["decision_core_v6_contract_status"].astype(str) == "degraded").sum()),
            }
        )
        step_result = portfolio.step(
            date=signal_dt,
            prices=prepared.close.loc[signal_dt],
            policy_frame=policy_frame,
            global_targets=global_targets,
            source_label="decision_core_v6_target_replay",
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
            net_return = gross_return - estimate_trading_cost(
                diagnostics=step_result.diagnostics,
                transaction_cost_bps=transaction_cost_bps,
                slippage_bps=slippage_bps,
                sell_tax_bps=sell_tax_bps,
            )
            returns.append(net_return)
            portfolio.record_realized_return(net_return)
    action_panel = pd.DataFrame(action_rows)
    turnover_frame = pd.DataFrame(turnover_rows)
    returns_series = pd.Series(returns, index=[dt.strftime("%Y-%m-%d") for dt in dates[1:]], dtype=float)
    metrics = compute_curve_metrics(returns_series)
    if not turnover_frame.empty:
        metrics["avg_turnover"] = float(pd.to_numeric(turnover_frame["realized_turnover"], errors="coerce").fillna(0.0).mean())
        metrics["avg_gross_exposure"] = float(1.0 - pd.to_numeric(turnover_frame["cash_weight"], errors="coerce").fillna(1.0).mean())
        metrics["avg_holding_count"] = float(pd.to_numeric(turnover_frame["holding_count"], errors="coerce").fillna(0.0).mean())
        for column in ("native_target_valid", "native_target_constraint_violations", "allocation_layer_native_target_used", "allocation_layer_native_fallback_used"):
            if column in turnover_frame.columns:
                metrics[column] = float(pd.to_numeric(turnover_frame[column], errors="coerce").fillna(0.0).mean())
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
        "target_daily": pd.DataFrame(target_rows),
    }


def _summarize_rollout(name: str, rollout: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(rollout.get("metrics", {}) or {})
    continuity = dict(rollout.get("continuity_metrics", {}) or {})
    combined = {**metrics, **continuity}
    summary = {key: combined.get(key, 0.0) for key in DIAGNOSTIC_KEYS if key in combined}
    summary["layer"] = name
    summary["signal_date_count"] = float(len(rollout.get("dates", []) or []))
    summary["action_rows"] = float(len(rollout.get("action_panel", pd.DataFrame())))
    summary["outcome_rows"] = float(len(rollout.get("action_outcomes", pd.DataFrame())))
    return _json_safe(summary)


def _mean_column(frame: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if frame.empty or column not in frame.columns:
        return float(default)
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.mean()) if len(values) else float(default)


def _execution_path_summary(rollout: dict[str, Any]) -> dict[str, Any]:
    turnover_frame = rollout.get("turnover_frame", pd.DataFrame())
    action_outcomes = rollout.get("action_outcomes", pd.DataFrame())
    if not isinstance(turnover_frame, pd.DataFrame):
        turnover_frame = pd.DataFrame()
    if not isinstance(action_outcomes, pd.DataFrame):
        action_outcomes = pd.DataFrame()
    summary: dict[str, Any] = {
        "allocation_layer_primary_mode_mean": _mean_column(turnover_frame, "allocation_layer_primary_mode"),
        "portfolio_cashflow_decision_v1_mode_used_mean": _mean_column(
            turnover_frame,
            "portfolio_cashflow_decision_v1_mode_used",
        ),
        "cashflow_decision_used_mean": _mean_column(turnover_frame, "cashflow_decision_used"),
        "cashflow_decision_failed_closed_mean": _mean_column(turnover_frame, "cashflow_decision_failed_closed"),
        "native_target_valid_mean": _mean_column(turnover_frame, "native_target_valid"),
        "allocation_layer_native_target_used_mean": _mean_column(turnover_frame, "allocation_layer_native_target_used"),
        "cashflow_decision_contract_valid_rate": float(
            (rollout.get("continuity_metrics", {}) or {}).get("cashflow_decision_contract_valid_rate", 0.0) or 0.0
        ),
    }
    if "cashflow_decision_invalid_reason" in turnover_frame.columns:
        reasons = turnover_frame["cashflow_decision_invalid_reason"].fillna("").astype(str)
        reasons = reasons.loc[reasons.ne("") & reasons.ne("none")]
        summary["cashflow_decision_invalid_reason_counts"] = {
            str(key): int(value) for key, value in reasons.value_counts().sort_index().items()
        }
    if "portfolio_cashflow_decision_v1_invalid_reason" in action_outcomes.columns:
        reasons = action_outcomes["portfolio_cashflow_decision_v1_invalid_reason"].fillna("").astype(str)
        reasons = reasons.loc[reasons.ne("") & reasons.ne("none")]
        summary["action_cashflow_invalid_reason_counts"] = {
            str(key): int(value) for key, value in reasons.value_counts().sort_index().items()
        }
    summary["v6_execution_path_valid"] = bool(
        summary["allocation_layer_primary_mode_mean"] >= V6_EXECUTION_PATH_MEAN_FLOOR
        and summary["portfolio_cashflow_decision_v1_mode_used_mean"] >= V6_EXECUTION_PATH_MEAN_FLOOR
        and summary["cashflow_decision_used_mean"] >= V6_EXECUTION_PATH_MEAN_FLOOR
        and summary["cashflow_decision_failed_closed_mean"] <= (1.0 - V6_EXECUTION_PATH_MEAN_FLOOR)
        and summary["native_target_valid_mean"] >= V6_EXECUTION_PATH_MEAN_FLOOR
        and summary["allocation_layer_native_target_used_mean"] >= V6_EXECUTION_PATH_MEAN_FLOOR
        and summary["cashflow_decision_contract_valid_rate"] >= V6_EXECUTION_PATH_MEAN_FLOOR
    )
    summary["v6_execution_path_partially_valid"] = bool(
        summary["allocation_layer_primary_mode_mean"] >= V6_EXECUTION_PATH_MEAN_FLOOR
        and summary["portfolio_cashflow_decision_v1_mode_used_mean"] >= V6_EXECUTION_PATH_MEAN_FLOOR
        and summary["cashflow_decision_used_mean"] >= V6_EXECUTION_PATH_MEAN_FLOOR
        and summary["native_target_valid_mean"] >= V6_EXECUTION_PARTIAL_PATH_MEAN_FLOOR
        and summary["allocation_layer_native_target_used_mean"] >= V6_EXECUTION_PATH_MEAN_FLOOR
        and summary["cashflow_decision_contract_valid_rate"] >= V6_EXECUTION_PARTIAL_PATH_MEAN_FLOOR
    )
    return _json_safe(summary)


def _compare_action_outcomes(left: pd.DataFrame, right: pd.DataFrame, *, left_name: str, right_name: str) -> dict[str, Any]:
    columns = ["date", "stock"]
    if left.empty or right.empty or not set(columns).issubset(left.columns) or not set(columns).issubset(right.columns):
        return {"status": "empty", "left": left_name, "right": right_name}
    left_work = left.copy()
    right_work = right.copy()
    left_work["date"] = pd.to_datetime(left_work["date"]).dt.strftime("%Y-%m-%d")
    right_work["date"] = pd.to_datetime(right_work["date"]).dt.strftime("%Y-%m-%d")
    merged = left_work.merge(
        right_work,
        how="inner",
        on=columns,
        suffixes=(f"_{left_name}", f"_{right_name}"),
    )
    if merged.empty:
        return {"status": "no_overlap", "left": left_name, "right": right_name}
    action_left = merged.get(f"execution_action_{left_name}", pd.Series("", index=merged.index)).astype(str)
    action_right = merged.get(f"execution_action_{right_name}", pd.Series("", index=merged.index)).astype(str)
    source_left = pd.to_numeric(merged.get(f"portfolio_daily_source_target_{left_name}", pd.Series(False, index=merged.index)), errors="coerce").fillna(0.0) > 0.5
    source_right = pd.to_numeric(merged.get(f"portfolio_daily_source_target_{right_name}", pd.Series(False, index=merged.index)), errors="coerce").fillna(0.0) > 0.5
    receiver_left = pd.to_numeric(merged.get(f"portfolio_daily_receiver_target_{left_name}", pd.Series(False, index=merged.index)), errors="coerce").fillna(0.0) > 0.5
    receiver_right = pd.to_numeric(merged.get(f"portfolio_daily_receiver_target_{right_name}", pd.Series(False, index=merged.index)), errors="coerce").fillna(0.0) > 0.5
    return {
        "status": "ok",
        "left": left_name,
        "right": right_name,
        "overlap_rows": int(len(merged)),
        "execution_action_match_rate": float((action_left == action_right).mean()),
        "source_target_match_rate": float((source_left == source_right).mean()),
        "receiver_target_match_rate": float((receiver_left == receiver_right).mean()),
        "left_source_count": int(source_left.sum()),
        "right_source_count": int(source_right.sum()),
        "left_receiver_count": int(receiver_left.sum()),
        "right_receiver_count": int(receiver_right.sum()),
    }


def _write_rollout_tables(root: Path, prefix: str, rollout: dict[str, Any]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for key, filename in (
        ("action_panel", "daily_action_panel.csv"),
        ("action_outcomes", "daily_action_outcomes.csv"),
        ("turnover_frame", "daily_turnover.csv"),
        ("position_history", "daily_position_history.csv"),
        ("target_daily", "daily_target_summary.csv"),
    ):
        value = rollout.get(key)
        if isinstance(value, pd.DataFrame) and not value.empty:
            path = root / f"{prefix}_{filename}"
            value.to_csv(path, index=False, encoding="utf-8-sig")
            paths[key] = str(path.resolve())
    returns = rollout.get("returns")
    if isinstance(returns, pd.Series):
        path = root / f"{prefix}_daily_returns.csv"
        returns.rename("daily_return").to_csv(path, encoding="utf-8-sig")
        paths["returns"] = str(path.resolve())
    return paths


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if str(args.training_dataset_id or "").strip() != DECISION_CORE_V6_STRICT_GOLD_DATASET_ID:
        raise ValueError(f"v6 diagnostics require strict Gold dataset id {DECISION_CORE_V6_STRICT_GOLD_DATASET_ID}.")
    if not bool(args.allow_non_v6_execution_path):
        if str(args.budget_semantics) != BUDGET_SEMANTICS_ALLOCATION_LAYER:
            raise ValueError(
                "v6 layer diagnostics require --budget-semantics allocation_layer_v1. "
                "Use --allow-non-v6-execution-path only for an explicit execution-gap counterfactual."
            )
        if str(args.budget_calibration) != BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER:
            raise ValueError(
                "v6 layer diagnostics require --budget-calibration end_to_end_allocation_layer_v1. "
                "Use --allow-non-v6-execution-path only for an explicit execution-gap counterfactual."
            )
    artifact_path = Path(args.model_path).expanduser().resolve()
    artifact = load_artifact(artifact_path)
    alpha_prior_source = str(args.alpha_prior_source or getattr(artifact, "train_summary", {}).get("alpha_prior_source") or DEFAULT_ALPHA_PRIOR_SOURCE)
    alpha_prior_score_panel = str(args.alpha_prior_score_panel or getattr(artifact, "train_summary", {}).get("alpha_prior_score_panel") or "")
    alpha_prior_target_weight_panel = str(args.alpha_prior_target_weight_panel or getattr(artifact, "train_summary", {}).get("alpha_prior_target_weight_panel") or "")
    tag = str(args.tag or timestamp_tag("decision_core_v6_layer_diagnostic"))
    root = STUDIES_ROOT / tag
    root.mkdir(parents=True, exist_ok=True)
    prepared = prepare_policy_inputs(
        pool_name=args.pool_name,
        start_date=args.start_date,
        end_date=args.end_date,
        benchmark=args.benchmark,
        data_source=args.data_source,
        csv_folder=args.csv_folder,
        lake_dataset_id=args.lake_dataset_id,
        data_lake_root=args.data_lake_root,
        max_universe_size=args.max_universe_size,
        pool_rebalance_days=args.pool_rebalance_days,
        pool_adv_window=args.pool_adv_window,
        alpha_prior_source=alpha_prior_source,
        alpha_prior_score_panel=alpha_prior_score_panel,
        alpha_prior_target_weight_panel=alpha_prior_target_weight_panel,
        refresh_cache=args.refresh_cache,
        progress_desc="decision-core v6 layer diagnostic",
    )
    common = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "execution_semantics": args.execution_semantics,
        "budget_semantics": args.budget_semantics,
        "budget_calibration": args.budget_calibration,
        "transaction_cost_bps": args.transaction_cost_bps,
        "slippage_bps": args.slippage_bps,
        "sell_tax_bps": args.sell_tax_bps,
    }
    target_rollout = _run_v6_target_replay(
        prepared=prepared,
        label_preset=args.label_preset,
        **common,
    )
    model_rollout = run_policy_rollout(
        prepared=prepared,
        artifact=artifact,
        source_label="decision_core_v6_model",
        budget_objective=args.budget_objective,
        **common,
    )
    target_paths = _write_rollout_tables(root, "target_upper_bound", target_rollout)
    model_paths = _write_rollout_tables(root, "model_rollout", model_rollout)
    model_vs_target = _compare_action_outcomes(
        target_rollout.get("action_outcomes", pd.DataFrame()),
        model_rollout.get("action_outcomes", pd.DataFrame()),
        left_name="target",
        right_name="model",
    )
    target_summary = _summarize_rollout("target_upper_bound", target_rollout)
    model_summary = _summarize_rollout("model_rollout", model_rollout)
    target_execution_path = _execution_path_summary(target_rollout)
    model_execution_path = _execution_path_summary(model_rollout)
    target_return_profitable = float(target_summary.get("total_return", 0.0) or 0.0) > TARGET_PROFIT_TOTAL_RETURN_FLOOR
    target_sharpe_profitable = float(target_summary.get("sharpe", 0.0) or 0.0) > TARGET_PROFIT_SHARPE_FLOOR
    target_profitable = bool(target_return_profitable and target_sharpe_profitable)
    target_contract_complete = bool(target_execution_path.get("v6_execution_path_valid", False))
    target_contract_partial = bool(target_execution_path.get("v6_execution_path_partially_valid", False))
    model_return_gap = float(model_summary.get("total_return", 0.0) or 0.0) - float(target_summary.get("total_return", 0.0) or 0.0)
    model_action_match = float(model_vs_target.get("execution_action_match_rate", 0.0) or 0.0) if model_vs_target.get("status") == "ok" else 0.0
    model_gap_large = bool(
        target_profitable
        and (model_return_gap <= MODEL_RETURN_GAP_LARGE or model_action_match < MODEL_ACTION_MATCH_WEAK_FLOOR)
    )
    execution_gap_inspect = bool(
        not target_contract_complete
        or float(target_summary.get("native_target_constraint_violations", 0.0) or 0.0) > 0.0
        or float(target_summary.get("decision_oracle_constraint_violation_mean", 0.0) or 0.0) > 1.0e-6
    )
    diagnosis = {
        "target_upper_bound": (
            "profitable_contract_clean"
            if target_profitable and target_contract_complete
            else (
                "profitable_contract_partial"
                if target_profitable and target_contract_partial
                else ("profitable_contract_broken" if target_profitable else "not_profitable")
            )
        ),
        "target_behavior_gate": {
            "source_positive_forward_sell_share_le_0_10": float(
                target_summary.get("portfolio_daily_source_positive_forward_sell_share", 1.0) or 0.0
            )
            <= 0.10,
            "immediate_reversal_rate_3d_le_0_005": float(target_summary.get("immediate_reversal_rate_3d", 1.0) or 0.0)
            <= 0.005,
            "receiver_minus_source_forward_excess_5d_gt_0": float(
                target_summary.get("portfolio_daily_receiver_minus_source_forward_excess_5d", -1.0) or 0.0
            )
            > 0.0,
        },
        "model_imitation_gap": "large" if model_gap_large else ("not_primary" if target_profitable else "blocked_by_target_return"),
        "simulator_execution_gap": "inspect" if execution_gap_inspect else "not_primary",
        "model_return_gap_vs_target": model_return_gap,
        "model_execution_action_match_rate": model_action_match,
        "recommended_next": (
            "fix_v6_target_oracle_cashflow_contract"
            if target_profitable and execution_gap_inspect
            else "fix_v6_target_oracle_cashflow_contract_and_objective_labels"
            if (not target_profitable) and execution_gap_inspect
            else "fix_v6_execution_adapter_or_diagnostic_path"
            if execution_gap_inspect
            else (
                "fix_v6_objective_or_labels"
                if not target_profitable
                else (
                    "investigate_model_capacity_loss_training"
                    if model_gap_large
                    else "run_longer_multi_segment_layer_diagnostic"
                )
            )
        ),
    }
    summary = {
        "run_tag": tag,
        "generated_at": now_iso(),
        "status": "completed",
        "decision_core": "v6",
        "decision_core_profile": DECISION_CORE_V6_PROFILE,
        "shadow_only": True,
        "model_path": str(artifact_path),
        "training_dataset_id": str(args.training_dataset_id),
        "lake_dataset_id": str(args.lake_dataset_id),
        "window": {"start_date": args.start_date, "end_date": args.end_date},
        "target_upper_bound": target_summary,
        "model_rollout": model_summary,
        "execution_path": {
            "target_upper_bound": target_execution_path,
            "model_rollout": model_execution_path,
        },
        "model_vs_target": _json_safe(model_vs_target),
        "diagnosis": diagnosis,
        "artifacts": {
            "target_upper_bound": target_paths,
            "model_rollout": model_paths,
        },
    }
    write_json(root / "layer_diagnostic_summary.json", _json_safe(summary))
    safe_print_json(_json_safe(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
