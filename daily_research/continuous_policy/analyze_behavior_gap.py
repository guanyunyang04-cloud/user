from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.continuous_policy.label_builder import build_future_path_metrics
from daily_research.continuous_policy.pipeline_utils import run_policy_rollout
from daily_research.continuous_policy.runtime import (
    CONTINUOUS_POLICY_ROOT,
    LATEST_EVALUATION_SUMMARY_PATH,
    now_iso,
    read_json,
    timestamp_tag,
    update_latest_summary,
    write_json,
)
from daily_research.continuous_policy.state_builder import prepare_policy_inputs


ANALYSIS_ROOT = CONTINUOUS_POLICY_ROOT / "analysis" / "behavior_audits"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze teacher-vs-model behavior gaps for continuous_policy.")
    parser.add_argument("--evaluation-summary", default=str(LATEST_EVALUATION_SUMMARY_PATH))
    parser.add_argument("--tag", default="")
    return parser


def _safe_float(mapping: dict[str, Any], key: str) -> float:
    return float(mapping.get(key, 0.0) or 0.0)


def _read_csv(path_value: str) -> pd.DataFrame:
    path = Path(str(path_value or "")).expanduser()
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _gap_row(metric: str, *, teacher: float, model: float, preferred_direction: str) -> dict[str, Any]:
    gap = float(teacher - model) if preferred_direction == "higher_better" else float(model - teacher)
    return {
        "metric": metric,
        "teacher": float(teacher),
        "model": float(model),
        "gap_vs_teacher": gap,
        "preferred_direction": preferred_direction,
    }


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y"}


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    working = frame.copy()
    for column in columns:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    return working


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    value = float(series.mean())
    return value if np.isfinite(value) else 0.0


def _action_pair_rows(
    action_outcomes: pd.DataFrame,
    *,
    actual_column: str = "execution_action",
    actual_key: str = "execution_action",
    limit: int = 8,
) -> list[dict[str, Any]]:
    if action_outcomes.empty:
        return []
    if actual_column not in action_outcomes.columns:
        actual_column = "execution_action"
    grouped = (
        action_outcomes.groupby(["model_action", actual_column], dropna=False)
        .agg(
            row_count=("stock", "count"),
            avg_abs_delta_weight=("abs_delta_weight", "mean"),
            avg_execution_deadband=("execution_deadband", "mean"),
            avg_forward_excess_5d=("forward_excess_5d", "mean"),
            avg_hold_days_before=("hold_days_before", "mean"),
        )
        .reset_index()
        .sort_values(["row_count", "avg_abs_delta_weight"], ascending=[False, False])
    )
    rows: list[dict[str, Any]] = []
    for item in grouped.head(limit).to_dict(orient="records"):
        rows.append(
            {
                "model_action": str(item.get("model_action", "") or ""),
                actual_key: str(item.get(actual_column, "") or ""),
                "row_count": int(item.get("row_count", 0) or 0),
                "avg_abs_delta_weight": float(item.get("avg_abs_delta_weight", 0.0) or 0.0),
                "avg_execution_deadband": float(item.get("avg_execution_deadband", 0.0) or 0.0),
                "avg_forward_excess_5d": float(item.get("avg_forward_excess_5d", 0.0) or 0.0),
                "avg_hold_days_before": float(item.get("avg_hold_days_before", 0.0) or 0.0),
            }
        )
    return rows


def _build_semantic_conflicts(
    *,
    action_outcomes: pd.DataFrame,
    turnover_frame: pd.DataFrame,
) -> dict[str, Any]:
    if action_outcomes.empty:
        return {
            "action_rows": 0,
            "semantic_conflict_rate": 0.0,
            "micro_rebalance_conflict_rate": 0.0,
            "deadband_conflict_rate": 0.0,
            "small_delta_conflict_rate": 0.0,
            "budget_clipped_day_share": 0.0,
            "budget_clipped_conflict_rate": 0.0,
            "unclipped_conflict_rate": 0.0,
            "budget_semantics_counts": {},
            "budget_calibration_counts": {},
            "avg_budget_risk_off_score": 0.0,
            "avg_budget_deploy_score": 0.0,
            "avg_budget_entry_candidate_count": 0.0,
            "avg_budget_entry_keep_count": 0.0,
            "avg_budget_held_protected_count": 0.0,
            "avg_budget_split_bound_guard_count": 0.0,
            "avg_budget_sell_priority_guard_count": 0.0,
            "high_cash_up_market_share": 0.0,
            "high_cash_down_market_share": 0.0,
            "top_action_pairs": [],
            "top_conflict_pairs": [],
            "diagnoses": [],
        }

    working = _coerce_numeric(
        action_outcomes,
        [
            "current_weight",
            "target_weight",
            "delta_weight",
            "hold_days_before",
            "execution_deadband",
            "forward_excess_5d",
            "forward_benchmark_return_1d",
        ],
    ).copy()
    working["model_action"] = working.get("model_action", pd.Series("", index=working.index)).astype(str)
    working["execution_action"] = working.get("execution_action", pd.Series("", index=working.index)).astype(str)
    working["weight_change_action"] = working.get("weight_change_action", working["execution_action"]).astype(str)
    working["is_semantic_conflict"] = working["model_action"] != working["execution_action"]
    working["is_order_translation_conflict"] = working["model_action"] != working["weight_change_action"]
    working["is_conflict"] = working["is_semantic_conflict"]
    working["abs_delta_weight"] = working["delta_weight"].abs()
    working["contradictory_micro_rebalance"] = working.get(
        "contradictory_micro_rebalance",
        pd.Series(False, index=working.index),
    ).map(_safe_bool)
    working["deadband_active"] = working.get(
        "execution_deadband",
        pd.Series(0.0, index=working.index),
    ).fillna(0.0) > 1e-12
    working["small_delta_conflict"] = working["is_semantic_conflict"] & (
        working["abs_delta_weight"] <= working.get("execution_deadband", pd.Series(0.0, index=working.index)).fillna(0.0)
    )
    working["small_delta_order_translation_conflict"] = working["is_order_translation_conflict"] & (
        working["abs_delta_weight"] <= working.get("execution_deadband", pd.Series(0.0, index=working.index)).fillna(0.0)
    )

    by_date = (
        working.groupby("date", dropna=False)
        .agg(
            action_rows=("stock", "count"),
            semantic_conflict_rate=("is_semantic_conflict", "mean"),
            order_translation_conflict_rate=("is_order_translation_conflict", "mean"),
            micro_rebalance_conflict_rate=("contradictory_micro_rebalance", "mean"),
            avg_abs_delta_weight=("abs_delta_weight", "mean"),
            benchmark_forward_return_1d=("forward_benchmark_return_1d", "mean"),
        )
        .reset_index()
    )
    turnover_working = _coerce_numeric(
        turnover_frame,
        [
            "raw_turnover",
            "realized_turnover",
            "budget_drop_count",
            "cash_weight",
            "gross_exposure_target",
            "gross_exposure_target_raw",
            "candidate_budget",
            "candidate_budget_raw",
            "turnover_budget",
            "turnover_budget_raw",
            "budget_risk_off_score",
            "budget_deploy_score",
            "budget_entry_candidate_count",
            "budget_entry_keep_count",
            "budget_held_protected_count",
            "budget_split_bound_guard_count",
            "budget_sell_priority_guard_count",
        ],
    )
    if turnover_working.empty:
        day_merge = by_date.copy()
        day_merge["budget_clipped"] = False
        day_merge["budget_drop_count"] = 0.0
        day_merge["cash_weight"] = np.nan
    else:
        turnover_working = turnover_working.copy()
        turnover_working["date"] = turnover_working["date"].astype(str)
        turnover_working["budget_clipped"] = (
            turnover_working["raw_turnover"].fillna(0.0) > turnover_working["realized_turnover"].fillna(0.0) + 1e-8
        )
        for optional_column in (
            "gross_exposure_target_raw",
            "candidate_budget",
            "candidate_budget_raw",
            "turnover_budget",
            "turnover_budget_raw",
            "budget_risk_off_score",
            "budget_deploy_score",
            "budget_entry_candidate_count",
            "budget_entry_keep_count",
            "budget_held_protected_count",
            "budget_split_bound_guard_count",
            "budget_sell_priority_guard_count",
        ):
            if optional_column not in turnover_working.columns:
                turnover_working[optional_column] = 0.0
        day_merge = by_date.merge(
            turnover_working[
                [
                    "date",
                    "budget_clipped",
                    "budget_drop_count",
                    "cash_weight",
                    "raw_turnover",
                    "realized_turnover",
                    "gross_exposure_target",
                    "gross_exposure_target_raw",
                    "candidate_budget",
                    "candidate_budget_raw",
                    "turnover_budget",
                    "turnover_budget_raw",
                    "budget_risk_off_score",
                    "budget_deploy_score",
                    "budget_entry_candidate_count",
                    "budget_entry_keep_count",
                    "budget_held_protected_count",
                    "budget_split_bound_guard_count",
                    "budget_sell_priority_guard_count",
                ]
            ],
            how="left",
            on="date",
        )
    clipped_days = day_merge.loc[day_merge["budget_clipped"].fillna(False)]
    unclipped_days = day_merge.loc[~day_merge["budget_clipped"].fillna(False)]
    cash_series = day_merge["cash_weight"].dropna()
    if cash_series.empty:
        high_cash_mask = pd.Series(False, index=day_merge.index)
    else:
        high_cash_threshold = float(cash_series.quantile(0.75))
        high_cash_mask = day_merge["cash_weight"].fillna(0.0) >= high_cash_threshold

    high_cash_up_market_share = _safe_mean(
        ((day_merge["benchmark_forward_return_1d"].fillna(0.0) > 0) & high_cash_mask).astype(float)
    )
    high_cash_down_market_share = _safe_mean(
        ((day_merge["benchmark_forward_return_1d"].fillna(0.0) < 0) & high_cash_mask).astype(float)
    )

    diagnoses: list[str] = []
    semantic_conflict_rate = _safe_mean(working["is_conflict"].astype(float))
    order_translation_conflict_rate = _safe_mean(working["is_order_translation_conflict"].astype(float))
    micro_rebalance_conflict_rate = _safe_mean(working["contradictory_micro_rebalance"].astype(float))
    deadband_conflict_rate = _safe_mean((working["is_semantic_conflict"] & working["deadband_active"]).astype(float))
    deadband_order_translation_conflict_rate = _safe_mean(
        (working["is_order_translation_conflict"] & working["deadband_active"]).astype(float)
    )
    small_delta_conflict_rate = _safe_mean(working["small_delta_conflict"].astype(float))
    small_delta_order_translation_conflict_rate = _safe_mean(
        working["small_delta_order_translation_conflict"].astype(float)
    )
    budget_clipped_conflict_rate = _safe_mean(clipped_days["semantic_conflict_rate"]) if not clipped_days.empty else 0.0
    unclipped_conflict_rate = _safe_mean(unclipped_days["semantic_conflict_rate"]) if not unclipped_days.empty else 0.0
    budget_clipped_order_translation_conflict_rate = (
        _safe_mean(clipped_days["order_translation_conflict_rate"]) if not clipped_days.empty else 0.0
    )
    unclipped_order_translation_conflict_rate = (
        _safe_mean(unclipped_days["order_translation_conflict_rate"]) if not unclipped_days.empty else 0.0
    )
    if semantic_conflict_rate >= 0.05:
        diagnoses.append("执行层仍在非小概率地重写模型动作语义，策略学习闭环还不干净。")
    if order_translation_conflict_rate >= 0.05:
        diagnoses.append("真实权重变化仍与模型动作意图存在偏离，预算/持仓约束还需要继续和动作头分层校准。")
    if micro_rebalance_conflict_rate >= 0.02 or small_delta_order_translation_conflict_rate >= 0.02:
        diagnoses.append("微幅再平衡仍会造成订单层反向变化，需继续隔离 hold/add/reduce 的生命周期语义。")
    if budget_clipped_order_translation_conflict_rate > unclipped_order_translation_conflict_rate + 0.03:
        diagnoses.append("预算/换手约束会显著放大订单层动作偏离，预算头与动作头仍未真正解耦。")
    if high_cash_up_market_share > high_cash_down_market_share + 0.05:
        diagnoses.append("高现金日更多出现在次日上涨前，当前现金部署仍带有顺周期保守偏差。")

    pair_rows = _action_pair_rows(working, actual_column="execution_action", actual_key="execution_action", limit=12)
    conflict_pair_rows = [row for row in pair_rows if row["model_action"] != row["execution_action"]][:8]
    order_pair_rows = _action_pair_rows(
        working,
        actual_column="weight_change_action",
        actual_key="weight_change_action",
        limit=12,
    )
    order_conflict_pair_rows = [
        row for row in order_pair_rows if row["model_action"] != row["weight_change_action"]
    ][:8]
    budget_semantics_counts = (
        {
            str(key): int(value)
            for key, value in turnover_frame.get("budget_semantics", pd.Series(dtype=str)).astype(str).value_counts().sort_index().items()
        }
        if not turnover_frame.empty
        else {}
    )
    budget_calibration_counts = (
        {
            str(key): int(value)
            for key, value in turnover_frame.get("budget_calibration", pd.Series(dtype=str)).astype(str).value_counts().sort_index().items()
        }
        if not turnover_frame.empty
        else {}
    )
    return {
        "action_rows": int(len(working)),
        "semantic_conflict_rate": semantic_conflict_rate,
        "order_translation_conflict_rate": order_translation_conflict_rate,
        "weight_change_conflict_rate": order_translation_conflict_rate,
        "micro_rebalance_conflict_rate": micro_rebalance_conflict_rate,
        "deadband_conflict_rate": deadband_conflict_rate,
        "deadband_order_translation_conflict_rate": deadband_order_translation_conflict_rate,
        "small_delta_conflict_rate": small_delta_conflict_rate,
        "small_delta_order_translation_conflict_rate": small_delta_order_translation_conflict_rate,
        "budget_clipped_day_share": _safe_mean(day_merge["budget_clipped"].fillna(False).astype(float)),
        "budget_clipped_conflict_rate": budget_clipped_conflict_rate,
        "unclipped_conflict_rate": unclipped_conflict_rate,
        "budget_clipped_order_translation_conflict_rate": budget_clipped_order_translation_conflict_rate,
        "unclipped_order_translation_conflict_rate": unclipped_order_translation_conflict_rate,
        "avg_budget_drop_count": _safe_mean(day_merge["budget_drop_count"].fillna(0.0)),
        "avg_cash_weight": _safe_mean(day_merge["cash_weight"].fillna(0.0)),
        "budget_semantics_counts": budget_semantics_counts,
        "budget_calibration_counts": budget_calibration_counts,
        "avg_budget_risk_off_score": _safe_mean(day_merge.get("budget_risk_off_score", pd.Series(0.0, index=day_merge.index)).fillna(0.0)),
        "avg_budget_deploy_score": _safe_mean(day_merge.get("budget_deploy_score", pd.Series(0.0, index=day_merge.index)).fillna(0.0)),
        "avg_budget_entry_candidate_count": _safe_mean(day_merge.get("budget_entry_candidate_count", pd.Series(0.0, index=day_merge.index)).fillna(0.0)),
        "avg_budget_entry_keep_count": _safe_mean(day_merge.get("budget_entry_keep_count", pd.Series(0.0, index=day_merge.index)).fillna(0.0)),
        "avg_budget_held_protected_count": _safe_mean(day_merge.get("budget_held_protected_count", pd.Series(0.0, index=day_merge.index)).fillna(0.0)),
        "avg_budget_split_bound_guard_count": _safe_mean(day_merge.get("budget_split_bound_guard_count", pd.Series(0.0, index=day_merge.index)).fillna(0.0)),
        "avg_budget_sell_priority_guard_count": _safe_mean(day_merge.get("budget_sell_priority_guard_count", pd.Series(0.0, index=day_merge.index)).fillna(0.0)),
        "high_cash_up_market_share": high_cash_up_market_share,
        "high_cash_down_market_share": high_cash_down_market_share,
        "top_action_pairs": pair_rows,
        "top_conflict_pairs": conflict_pair_rows,
        "top_weight_change_pairs": order_pair_rows,
        "top_order_translation_conflict_pairs": order_conflict_pair_rows,
        "diagnoses": diagnoses,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluation_summary_path = Path(str(args.evaluation_summary or "")).expanduser().resolve()
    evaluation_summary = read_json(evaluation_summary_path)
    if not evaluation_summary:
        raise FileNotFoundError(f"Evaluation summary not found or unreadable: {evaluation_summary_path}")

    prepared = prepare_policy_inputs(
        pool_name=str(evaluation_summary.get("pool_name", "") or "liquid500"),
        start_date=str(evaluation_summary.get("start_date", "") or "20250318"),
        end_date=str(evaluation_summary.get("end_date", "") or ""),
        benchmark=str(evaluation_summary.get("benchmark", "") or "000300.SH"),
        progress_desc="continuous policy behavior audit",
    )
    future_metrics = build_future_path_metrics(prepared)
    teacher_rollout = run_policy_rollout(
        prepared=prepared,
        artifact=None,
        future_metrics=future_metrics,
        start_date=str(evaluation_summary.get("start_date", "") or ""),
        end_date=str(evaluation_summary.get("end_date", "") or ""),
        source_label="teacher_oracle",
        label_preset=str(evaluation_summary.get("label_preset", "") or "balanced_v2"),
        execution_semantics=str(evaluation_summary.get("execution_semantics", "") or "semantic_preserving_v1"),
        budget_semantics=str(evaluation_summary.get("budget_semantics", "") or "legacy_total_candidate"),
        budget_calibration=str(evaluation_summary.get("budget_calibration", "") or "none"),
    )

    model_continuity = dict(evaluation_summary.get("continuity_metrics", {}) or {})
    teacher_continuity = dict(teacher_rollout.get("continuity_metrics", {}) or {})
    model_metrics = dict(evaluation_summary.get("continuous_policy_metrics", {}) or {})
    teacher_metrics = dict(teacher_rollout.get("metrics", {}) or {})
    action_outcomes = _read_csv(str(evaluation_summary.get("action_outcomes_csv", "") or ""))
    turnover_frame = _read_csv(str(evaluation_summary.get("turnover_csv", "") or ""))
    semantic_conflicts = _build_semantic_conflicts(
        action_outcomes=action_outcomes,
        turnover_frame=turnover_frame,
    )

    early_reduce_share = _safe_float(model_continuity, "early_reduce_share")
    early_exit_share = _safe_float(model_continuity, "early_exit_share")
    profitable_reduce_share = _safe_float(model_continuity, "profitable_reduce_share")
    wrong_side_reduce_share = _safe_float(model_continuity, "wrong_side_reduce_share")
    profit_take_too_early_share = _safe_float(model_continuity, "profit_take_too_early_share")
    reentry_after_exit_3d_rate = _safe_float(model_continuity, "reentry_after_exit_3d_rate")
    reversal_after_reduce_3d_rate = _safe_float(model_continuity, "reversal_after_reduce_3d_rate")
    reversal_after_exit_3d_rate = _safe_float(model_continuity, "reversal_after_exit_3d_rate")
    exit_then_rebound_cost = _safe_float(model_continuity, "exit_then_rebound_cost")
    risk_off_cash_hit_rate = _safe_float(model_continuity, "risk_off_cash_hit_rate")
    cold_start_open_rate = _safe_float(model_continuity, "cold_start_open_rate")
    days_to_first_open = _safe_float(model_continuity, "days_to_first_open")
    if not action_outcomes.empty:
        reduce_rows = action_outcomes.loc[action_outcomes["execution_action"].astype(str) == "reduce"].copy()
        exit_rows = action_outcomes.loc[action_outcomes["execution_action"].astype(str) == "exit"].copy()
        if early_reduce_share <= 0.0 and not reduce_rows.empty:
            early_reduce_share = float((reduce_rows["hold_days_before"].fillna(999.0) <= 3.0).mean())
        if early_exit_share <= 0.0 and not exit_rows.empty:
            early_exit_share = float((exit_rows["hold_days_before"].fillna(999.0) <= 3.0).mean())
        if profitable_reduce_share <= 0.0 and not reduce_rows.empty:
            profitable_reduce_share = float((reduce_rows["unrealized_pnl_before"].fillna(0.0) > 0.05).mean())

    gap_rows = [
        _gap_row(
            "hold_share",
            teacher=_safe_float(teacher_continuity, "hold_share"),
            model=_safe_float(model_continuity, "hold_share"),
            preferred_direction="higher_better",
        ),
        _gap_row(
            "reduce_success_rate_5d",
            teacher=_safe_float(teacher_continuity, "reduce_success_rate_5d"),
            model=_safe_float(model_continuity, "reduce_success_rate_5d"),
            preferred_direction="higher_better",
        ),
        _gap_row(
            "cash_timing_quality_1d",
            teacher=_safe_float(teacher_continuity, "cash_timing_quality_1d"),
            model=_safe_float(model_continuity, "cash_timing_quality_1d"),
            preferred_direction="higher_better",
        ),
        _gap_row(
            "hold_retention_quality_5d",
            teacher=_safe_float(teacher_continuity, "hold_retention_quality_5d"),
            model=_safe_float(model_continuity, "hold_retention_quality_5d"),
            preferred_direction="higher_better",
        ),
        _gap_row(
            "sell_selection_quality_5d",
            teacher=_safe_float(teacher_continuity, "sell_selection_quality_5d"),
            model=_safe_float(model_continuity, "sell_selection_quality_5d"),
            preferred_direction="higher_better",
        ),
        _gap_row(
            "avg_turnover",
            teacher=_safe_float(teacher_metrics, "avg_turnover"),
            model=_safe_float(model_metrics, "avg_turnover"),
            preferred_direction="lower_better",
        ),
        _gap_row(
            "churn_ratio",
            teacher=_safe_float(teacher_continuity, "churn_ratio"),
            model=_safe_float(model_continuity, "churn_ratio"),
            preferred_direction="lower_better",
        ),
        _gap_row(
            "immediate_reversal_rate_3d",
            teacher=_safe_float(teacher_continuity, "immediate_reversal_rate_3d"),
            model=_safe_float(model_continuity, "immediate_reversal_rate_3d"),
            preferred_direction="lower_better",
        ),
    ]

    bottlenecks: list[dict[str, Any]] = []
    model_hold_share = _safe_float(model_continuity, "hold_share")
    teacher_hold_share = _safe_float(teacher_continuity, "hold_share")
    if model_hold_share < max(0.12, teacher_hold_share * 0.70):
        bottlenecks.append(
            {
                "name": "hold_share_collapse",
                "severity": "high",
                "diagnosis": "模型的持有占比仍然过低，说明连续持有逻辑还没有稳定学出来。",
                "evidence": {
                    "model_hold_share": model_hold_share,
                    "teacher_hold_share": teacher_hold_share,
                },
            }
        )
    if _safe_float(model_continuity, "reduce_success_rate_5d") < 0.45:
        bottlenecks.append(
            {
                "name": "reduce_too_early_or_wrong_side",
                "severity": "high",
                "diagnosis": "减仓后 5 日成功率仍偏低，说明模型仍在错误地对强势票过早减仓，或者把减仓下在了错误的股票上。",
                "evidence": {
                    "model_reduce_success_rate_5d": _safe_float(model_continuity, "reduce_success_rate_5d"),
                    "teacher_reduce_success_rate_5d": _safe_float(teacher_continuity, "reduce_success_rate_5d"),
                    "early_reduce_share": early_reduce_share,
                    "profitable_reduce_share": profitable_reduce_share,
                    "wrong_side_reduce_share": wrong_side_reduce_share,
                    "profit_take_too_early_share": profit_take_too_early_share,
                    "reversal_after_reduce_3d_rate": reversal_after_reduce_3d_rate,
                },
            }
        )
    if _safe_float(model_continuity, "sell_selection_quality_5d") < 0.01:
        bottlenecks.append(
            {
                "name": "sell_selection_not_learned",
                "severity": "high",
                "diagnosis": "组合需要收缩时，模型仍没有稳定选对该减仓/退出的持仓，卖出对象选择质量不足。",
                "evidence": {
                    "model_sell_selection_quality_5d": _safe_float(model_continuity, "sell_selection_quality_5d"),
                    "teacher_sell_selection_quality_5d": _safe_float(teacher_continuity, "sell_selection_quality_5d"),
                    "model_sell_selection_hit_rate_5d": _safe_float(model_continuity, "sell_selection_hit_rate_5d"),
                    "teacher_sell_selection_hit_rate_5d": _safe_float(teacher_continuity, "sell_selection_hit_rate_5d"),
                    "wrong_side_reduce_share": wrong_side_reduce_share,
                },
            }
        )
    if _safe_float(model_continuity, "cash_timing_quality_1d") < 0.02:
        bottlenecks.append(
            {
                "name": "cash_timing_not_learned",
                "severity": "high",
                "diagnosis": "现金时机质量仍接近 0 或为负，说明组合预算头还没有学会在市场走弱时主动降暴露。",
                "evidence": {
                    "model_cash_timing_quality_1d": _safe_float(model_continuity, "cash_timing_quality_1d"),
                    "teacher_cash_timing_quality_1d": _safe_float(teacher_continuity, "cash_timing_quality_1d"),
                    "risk_off_cash_hit_rate": risk_off_cash_hit_rate,
                    "cold_start_open_rate": cold_start_open_rate,
                    "days_to_first_open": days_to_first_open,
                },
            }
        )
    if _safe_float(model_metrics, "avg_turnover") > max(0.25, _safe_float(teacher_metrics, "avg_turnover") * 1.4):
        bottlenecks.append(
            {
                "name": "turnover_decoder_churn",
                "severity": "medium",
                "diagnosis": "模型换手仍偏高，说明 decoder 还在把局部动作信号翻译成碎片化调仓。",
                "evidence": {
                    "model_avg_turnover": _safe_float(model_metrics, "avg_turnover"),
                    "teacher_avg_turnover": _safe_float(teacher_metrics, "avg_turnover"),
                    "model_churn_ratio": _safe_float(model_continuity, "churn_ratio"),
                    "teacher_churn_ratio": _safe_float(teacher_continuity, "churn_ratio"),
                },
            }
        )
    if early_exit_share > 0.20:
        bottlenecks.append(
            {
                "name": "early_exit_bias",
                "severity": "medium",
                "diagnosis": "过早退出占比偏高，说明模型对持仓生命周期的容忍度还不够。",
                "evidence": {
                    "early_exit_share": early_exit_share,
                    "exit_then_rebound_cost": exit_then_rebound_cost,
                    "reentry_after_exit_3d_rate": reentry_after_exit_3d_rate,
                    "reversal_after_exit_3d_rate": reversal_after_exit_3d_rate,
                },
            }
        )
    if _safe_float(model_continuity, "immediate_reversal_rate_3d") > 0.35:
        bottlenecks.append(
            {
                "name": "shadow_reversal_still_high",
                "severity": "medium",
                "diagnosis": "短期反手率仍偏高，说明模型在连续持有与再次进出之间还存在过度摆动。",
                "evidence": {
                    "model_immediate_reversal_rate_3d": _safe_float(model_continuity, "immediate_reversal_rate_3d"),
                    "reversal_after_reduce_3d_rate": reversal_after_reduce_3d_rate,
                    "reversal_after_exit_3d_rate": reversal_after_exit_3d_rate,
                },
            }
        )
    if float(semantic_conflicts.get("semantic_conflict_rate", 0.0) or 0.0) >= 0.05:
        bottlenecks.append(
            {
                "name": "execution_semantics_rewrite",
                "severity": "high",
                "diagnosis": "执行层仍会把模型动作改写成别的动作，训练与回测看到的策略语义不是同一个对象。",
                "evidence": {
                    "semantic_conflict_rate": float(semantic_conflicts.get("semantic_conflict_rate", 0.0) or 0.0),
                    "micro_rebalance_conflict_rate": float(semantic_conflicts.get("micro_rebalance_conflict_rate", 0.0) or 0.0),
                    "top_conflict_pairs": list(semantic_conflicts.get("top_conflict_pairs", []) or []),
                },
            }
        )
    if float(semantic_conflicts.get("order_translation_conflict_rate", 0.0) or 0.0) >= 0.08:
        bottlenecks.append(
            {
                "name": "order_translation_drift",
                "severity": "medium",
                "diagnosis": "模型生命周期语义与真实权重变化仍存在偏离；这不再等同于语义改写，但说明预算/权重翻译还需校准。",
                "evidence": {
                    "order_translation_conflict_rate": float(
                        semantic_conflicts.get("order_translation_conflict_rate", 0.0) or 0.0
                    ),
                    "small_delta_order_translation_conflict_rate": float(
                        semantic_conflicts.get("small_delta_order_translation_conflict_rate", 0.0) or 0.0
                    ),
                    "top_order_translation_conflict_pairs": list(
                        semantic_conflicts.get("top_order_translation_conflict_pairs", []) or []
                    ),
                },
            }
        )
    if float(semantic_conflicts.get("budget_clipped_order_translation_conflict_rate", 0.0) or 0.0) > float(
        semantic_conflicts.get("unclipped_order_translation_conflict_rate", 0.0) or 0.0
    ) + 0.03:
        bottlenecks.append(
            {
                "name": "budget_action_entanglement",
                "severity": "high",
                "diagnosis": "预算/换手约束显著放大订单层动作偏离，说明组合预算头与个股动作头仍在互相干扰。",
                "evidence": {
                    "budget_clipped_day_share": float(semantic_conflicts.get("budget_clipped_day_share", 0.0) or 0.0),
                    "budget_clipped_order_translation_conflict_rate": float(
                        semantic_conflicts.get("budget_clipped_order_translation_conflict_rate", 0.0) or 0.0
                    ),
                    "unclipped_order_translation_conflict_rate": float(
                        semantic_conflicts.get("unclipped_order_translation_conflict_rate", 0.0) or 0.0
                    ),
                },
            }
        )

    recommended_focus: list[str] = []
    if model_hold_share < 0.12:
        recommended_focus.append("先把 hold_share 拉过最低连续持有阈值，再去追求更复杂的收益最优行为。")
    else:
        recommended_focus.append("保住当前持有连续性，不要在修 reduce / cash / reversal 时把 hold_share 再打回塌缩。")
    if _safe_float(model_continuity, "reduce_success_rate_5d") < 0.45:
        recommended_focus.append("优先把 reduce 聚焦到真正的信号衰减与防守切换，而不是盈利后的机械落袋。")
    if _safe_float(model_continuity, "sell_selection_quality_5d") < 0.01:
        recommended_focus.append("把预算收缩映射到明确的卖出归因上，确保需要降风险时优先减的是低后续收益、高卖出压力的持仓。")
    if _safe_float(model_continuity, "cash_timing_quality_1d") < 0.02:
        recommended_focus.append("继续强化组合层 cash / gross / turnover 预算头，让市场转弱时能主动降暴露。")
    if early_exit_share > 0.20:
        recommended_focus.append("延缓过早退出，把生命周期未完成阶段和真正失效阶段分开。")
    if _safe_float(model_continuity, "immediate_reversal_rate_3d") > 0.35:
        recommended_focus.append("优先压低 shadow reversal，避免刚减仓或退出后又被过快追回。")
    if float(semantic_conflicts.get("semantic_conflict_rate", 0.0) or 0.0) >= 0.05:
        recommended_focus.append("先把 execution layer 退回“翻译器”角色，停止改写模型动作语义。")
    if float(semantic_conflicts.get("order_translation_conflict_rate", 0.0) or 0.0) >= 0.08:
        recommended_focus.append("继续校准权重翻译层，保留模型生命周期语义，同时减少真实订单方向与模型意图的偏离。")
    if float(semantic_conflicts.get("budget_clipped_order_translation_conflict_rate", 0.0) or 0.0) > float(
        semantic_conflicts.get("unclipped_order_translation_conflict_rate", 0.0) or 0.0
    ) + 0.03:
        recommended_focus.append("把预算头与个股动作头分层，让 cash / gross / turnover 只做组合部署，不再吞掉 reduce / exit 语义。")
    if float(semantic_conflicts.get("high_cash_up_market_share", 0.0) or 0.0) > float(
        semantic_conflicts.get("high_cash_down_market_share", 0.0) or 0.0
    ) + 0.05:
        recommended_focus.append("cash timing 不能继续主要依赖 teacher 模仿，而应更多由结果驱动目标直接校准。")
    if not recommended_focus:
        recommended_focus.append("当前主要行为瓶颈已明显缓解，可以把重点转向更高阶的收益优化与更强模型复核。")

    payload = {
        "run_tag": str(args.tag or timestamp_tag("behavior_audit")),
        "generated_at": now_iso(),
        "evaluation_summary_path": str(evaluation_summary_path),
        "teacher_recomputed": {
            "metrics": teacher_metrics,
            "continuity_metrics": teacher_continuity,
        },
        "model_metrics": model_metrics,
        "model_continuity_metrics": model_continuity,
        "gap_table": gap_rows,
        "micro_signals": {
            "early_reduce_share": early_reduce_share,
            "early_exit_share": early_exit_share,
            "profitable_reduce_share": profitable_reduce_share,
            "wrong_side_reduce_share": wrong_side_reduce_share,
            "profit_take_too_early_share": profit_take_too_early_share,
            "reentry_after_exit_3d_rate": reentry_after_exit_3d_rate,
            "reversal_after_reduce_3d_rate": reversal_after_reduce_3d_rate,
            "reversal_after_exit_3d_rate": reversal_after_exit_3d_rate,
            "exit_then_rebound_cost": exit_then_rebound_cost,
            "risk_off_cash_hit_rate": risk_off_cash_hit_rate,
            "cold_start_open_rate": cold_start_open_rate,
            "days_to_first_open": days_to_first_open,
        },
        "semantic_conflicts": semantic_conflicts,
        "bottlenecks": bottlenecks,
        "recommended_focus": recommended_focus,
    }
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = ANALYSIS_ROOT / f"{payload['run_tag']}.json"
    write_json(output_path, payload)
    payload["output_path"] = str(output_path.resolve())
    update_latest_summary("behavior_audit", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
