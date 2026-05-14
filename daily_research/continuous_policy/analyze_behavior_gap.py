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
from daily_research.continuous_policy.allocation_closure_diagnostics import (
    summarize_allocation_closure_from_turnover,
)
from daily_research.continuous_policy.pipeline_utils import run_policy_rollout
from daily_research.continuous_policy.runtime import (
    CONTINUOUS_POLICY_ROOT,
    LATEST_EVALUATION_SUMMARY_PATH,
    now_iso,
    read_json,
    safe_print_json,
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
    parser.add_argument("--export-held-side-details", action="store_true")
    parser.add_argument("--held-side-detail-limit", type=int, default=40)
    return parser


def _coerce_positive_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)


def _resolve_audit_prepare_kwargs(evaluation_summary: dict[str, Any]) -> dict[str, Any]:
    prepared_summary = dict(evaluation_summary.get("prepared_summary", {}) or {})
    alpha_summary = dict(prepared_summary.get("alpha_prior_summary", {}) or {})
    max_universe_size = _coerce_positive_int(evaluation_summary.get("max_universe_size"), 0)
    if max_universe_size <= 0:
        max_universe_size = _coerce_positive_int(prepared_summary.get("max_universe_size"), 0)
    if max_universe_size <= 0:
        max_universe_size = _coerce_positive_int(prepared_summary.get("universe_size"), 0)
    return {
        "pool_name": str(evaluation_summary.get("pool_name", "") or prepared_summary.get("pool_name", "") or "liquid500"),
        "start_date": str(evaluation_summary.get("start_date", "") or prepared_summary.get("requested_start_date", "") or "20250318"),
        "end_date": str(evaluation_summary.get("end_date", "") or prepared_summary.get("end_date", "") or ""),
        "benchmark": str(evaluation_summary.get("benchmark", "") or prepared_summary.get("benchmark", "") or "000300.SH"),
        "data_source": str(evaluation_summary.get("data_source", "") or prepared_summary.get("data_source", "") or "tq"),
        "csv_folder": str(evaluation_summary.get("csv_folder", "") or prepared_summary.get("csv_folder", "") or ""),
        "max_universe_size": int(max_universe_size),
        "alpha_prior_source": str(evaluation_summary.get("alpha_prior_source", "") or alpha_summary.get("source", "") or ""),
        "alpha_prior_score_panel": str(evaluation_summary.get("alpha_prior_score_panel", "") or alpha_summary.get("score_panel_csv", "") or ""),
        "alpha_prior_target_weight_panel": str(
            evaluation_summary.get("alpha_prior_target_weight_panel", "") or alpha_summary.get("target_weight_panel_csv", "") or ""
        ),
        "progress_desc": "continuous policy behavior audit",
    }


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


def _weight_change_action(previous_weight: float, new_weight: float) -> str:
    previous = float(previous_weight)
    new = float(new_weight)
    if previous <= 1.0e-8 and new <= 1.0e-8:
        return "skip"
    if previous <= 1.0e-8 and new > 1.0e-8:
        return "open"
    if previous > 1.0e-8 and new <= 1.0e-8:
        return "exit"
    if new > previous + 1.0e-8:
        return "add"
    if new < previous - 1.0e-8:
        return "reduce"
    return "hold"


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    working = frame.copy()
    for column in columns:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    return working


def _compute_exposure_utilization_from_turnover(turnover_frame: pd.DataFrame) -> tuple[float, float]:
    """Return gross-exposure target and realized utilization for audit panels."""
    if turnover_frame.empty:
        return 0.0, 0.0
    working = turnover_frame.copy()
    index = working.index
    cash_weight = (
        pd.to_numeric(working["cash_weight"], errors="coerce")
        if "cash_weight" in working.columns
        else pd.Series(np.nan, index=index, dtype=float)
    )
    if "gross_exposure" in working.columns:
        gross_exposure = pd.to_numeric(working["gross_exposure"], errors="coerce")
        usable_gross = gross_exposure.dropna()
        if usable_gross.empty or float(usable_gross.abs().sum()) <= 1.0e-12:
            gross_exposure = 1.0 - cash_weight
    elif "cash_weight" in working.columns:
        gross_exposure = 1.0 - cash_weight
    else:
        gross_exposure = pd.Series(0.0, index=index, dtype=float)
    avg_gross_exposure = _safe_mean(gross_exposure.replace([np.inf, -np.inf], np.nan).fillna(0.0))
    target_series = (
        pd.to_numeric(working["gross_exposure_target"], errors="coerce")
        if "gross_exposure_target" in working.columns
        else pd.Series(0.0, index=index, dtype=float)
    )
    avg_gross_exposure_target = _safe_mean(target_series.replace([np.inf, -np.inf], np.nan).fillna(0.0))
    utilization = (
        float(avg_gross_exposure / max(avg_gross_exposure_target, 1.0e-8))
        if avg_gross_exposure_target > 0.0
        else 0.0
    )
    return float(avg_gross_exposure_target), float(utilization)


def _compute_held_side_support_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    working = frame.copy()
    hold_value_series = working.get("hold_continuation_value", pd.Series(0.0, index=working.index)).fillna(0.0)
    alpha_value_series = working.get("alpha_opportunity_value", pd.Series(0.0, index=working.index)).fillna(0.0)
    sell_release_value_series = working.get("sell_release_value", pd.Series(0.0, index=working.index)).fillna(0.0)
    cash_defense_value_series = working.get("cash_defense_value", pd.Series(0.0, index=working.index)).fillna(0.0)
    deploy_value_series = working.get("deploy_value_target", pd.Series(0.0, index=working.index)).fillna(0.0)
    release_value_series = working.get("release_value_target", pd.Series(0.0, index=working.index)).fillna(0.0)
    deploy_gate_series = working.get("deploy_gate_target", pd.Series(0.0, index=working.index)).fillna(0.0)
    release_gate_series = working.get("release_gate_target", pd.Series(0.0, index=working.index)).fillna(0.0)
    deploy_executability_series = working.get(
        "deploy_executability_target",
        pd.Series(0.0, index=working.index),
    ).fillna(0.0)
    disciplined_funding_need = (
        0.48 * deploy_executability_series
        + 0.24 * deploy_gate_series
        + 0.16 * deploy_value_series
        + 0.06 * alpha_value_series
        - 0.16 * hold_value_series
        - 0.08 * release_value_series
    ).clip(lower=0.0, upper=1.0)
    protected_hold_support = (
        0.34 * hold_value_series
        + 0.18 * alpha_value_series
        + 0.12 * deploy_value_series
        + 0.08 * deploy_gate_series
        + 0.06 * deploy_executability_series
        - 0.20 * release_value_series
        - 0.18 * release_gate_series
        - 0.14 * sell_release_value_series
        - 0.08 * cash_defense_value_series
    ).clip(lower=0.0, upper=1.0)
    funding_release_support = (
        0.28 * release_value_series
        + 0.22 * release_gate_series
        + 0.18 * sell_release_value_series
        + 0.10 * cash_defense_value_series
        + 0.36 * disciplined_funding_need
        - 0.18 * hold_value_series
        - 0.08 * alpha_value_series
    ).clip(lower=0.0, upper=1.0)
    support_margin = 0.10
    working["disciplined_funding_need"] = disciplined_funding_need
    working["protected_hold_support"] = protected_hold_support
    working["funding_release_support"] = funding_release_support
    working["held_side_support_gap"] = funding_release_support - protected_hold_support
    working["held_side_release_consistent"] = (
        funding_release_support > protected_hold_support + support_margin
    ).astype(float)
    working["held_side_against_protected_hold"] = (
        protected_hold_support > funding_release_support + support_margin
    ).astype(float)
    return working


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    value = float(series.mean())
    return value if np.isfinite(value) else 0.0


def _bounded_unit(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    clipped = min(max(float(value), low), high)
    return (clipped - low) / float(high - low)


def _weighted_unit_score(components: list[tuple[float, float]]) -> float:
    total_weight = sum(weight for _, weight in components if weight > 0.0)
    if total_weight <= 0.0:
        return 0.0
    return float(sum(value * weight for value, weight in components if weight > 0.0) / total_weight)


def _release_translation_deploy_health(
    *,
    deploy_intent_action_count: float,
    deploy_intent_realized_rate: float,
    order_translation_conflict_rate: float,
    add_to_hold_conflict_share: float,
    sell_intent_suppressed_share: float,
    budget_origin_sell_share: float,
    deploy_funding_rebalance_sell_count: float,
    deploy_funding_rebalance_sell_share: float,
    deploy_funding_rebalance_forward_excess_5d: float,
    deploy_funding_against_protected_hold_share: float,
    deploy_funding_release_consistent_share: float,
    model_release_signal_sell_count: float,
    model_release_signal_forward_excess_5d: float,
    model_release_against_protected_hold_share: float,
    model_release_release_consistent_share: float,
) -> dict[str, Any]:
    deploy_observed = float(deploy_intent_action_count) >= 3.0
    funding_observed = float(deploy_funding_rebalance_sell_count) >= 5.0
    release_observed = float(model_release_signal_sell_count) >= 5.0

    deploy_score = _bounded_unit(deploy_intent_realized_rate, 0.35, 0.85) if deploy_observed else 0.0
    release_score = (
        _bounded_unit(deploy_funding_release_consistent_share, 0.05, 0.65)
        if funding_observed
        else 0.0
    )
    translation_score = 1.0 - _bounded_unit(order_translation_conflict_rate, 0.06, 0.35)
    intent_integrity_score = 1.0 - _bounded_unit(
        max(add_to_hold_conflict_share, sell_intent_suppressed_share),
        0.05,
        0.35,
    )
    funding_forward_score = 1.0 - _bounded_unit(max(0.0, deploy_funding_rebalance_forward_excess_5d), 0.0, 0.04)
    protected_hold_score = 1.0 - _bounded_unit(deploy_funding_against_protected_hold_share, 0.10, 0.45)
    funding_sell_share_score = 1.0 - _bounded_unit(
        max(0.0, deploy_funding_rebalance_sell_share - 0.35),
        0.0,
        0.45,
    )
    budget_origin_score = 1.0 - _bounded_unit(budget_origin_sell_share, 0.0, 0.35)
    funding_score = (
        _weighted_unit_score(
            [
                (funding_forward_score, 0.30),
                (protected_hold_score, 0.30),
                (funding_sell_share_score, 0.20),
                (budget_origin_score, 0.20),
            ]
        )
        if funding_observed
        else 0.0
    )
    model_release_score = (
        _weighted_unit_score(
            [
                (1.0 - _bounded_unit(max(0.0, model_release_signal_forward_excess_5d), 0.0, 0.04), 0.35),
                (1.0 - _bounded_unit(model_release_against_protected_hold_share, 0.10, 0.45), 0.30),
                (_bounded_unit(model_release_release_consistent_share, 0.05, 0.65), 0.35),
            ]
        )
        if release_observed
        else release_score
    )
    health_score = _weighted_unit_score(
        [
            (deploy_score, 0.26),
            (release_score, 0.28),
            (translation_score, 0.20),
            (funding_score, 0.20),
            (model_release_score, 0.06),
        ]
    )

    if not deploy_observed:
        failure_mode = "deploy_intent_not_observed"
    elif deploy_score < 0.40:
        failure_mode = "deploy_not_realized"
    elif translation_score < 0.45 or intent_integrity_score < 0.45:
        failure_mode = "order_translation_drift"
    elif not funding_observed:
        failure_mode = "funding_release_not_observed"
    elif release_score < 0.30:
        failure_mode = "release_not_learned_despite_deploy"
    elif funding_score < 0.45:
        failure_mode = "funding_source_pollution"
    elif release_observed and model_release_score < 0.45:
        failure_mode = "release_signal_not_selective"
    elif health_score >= 0.70:
        failure_mode = "healthy"
    else:
        failure_mode = "mixed_or_improving"

    components = {
        "deploy_score": deploy_score,
        "release_score": release_score,
        "translation_score": translation_score,
        "intent_integrity_score": intent_integrity_score,
        "funding_score": funding_score,
        "model_release_score": model_release_score,
        "funding_forward_score": funding_forward_score if funding_observed else 0.0,
        "protected_hold_score": protected_hold_score if funding_observed else 0.0,
        "funding_sell_share_score": funding_sell_share_score if funding_observed else 0.0,
        "budget_origin_score": budget_origin_score,
    }
    return {
        "release_translation_deploy_health_score": health_score,
        "release_translation_deploy_failure_mode": failure_mode,
        "release_translation_deploy_components": components,
        "release_translation_deploy_deploy_score": deploy_score,
        "release_translation_deploy_release_score": release_score,
        "release_translation_deploy_translation_score": translation_score,
        "release_translation_deploy_intent_integrity_score": intent_integrity_score,
        "release_translation_deploy_funding_score": funding_score,
        "release_translation_deploy_model_release_score": model_release_score,
    }


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


def _build_held_side_detail_payload(
    *,
    action_outcomes: pd.DataFrame,
    run_tag: str,
    detail_limit: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if action_outcomes.empty:
        return pd.DataFrame(), {
            "event_count": 0,
            "csv_path": "",
            "summary_json": "",
            "origin_counts": {},
            "top_symbols_by_event_count": [],
            "top_symbols_by_positive_forward_excess": [],
            "top_conflict_rows": [],
            "top_release_consistent_rows": [],
        }
    working = _coerce_numeric(
        action_outcomes,
        [
            "current_weight",
            "target_weight",
            "delta_weight",
            "hold_days_before",
            "forward_excess_5d",
            "sell_rank_score",
            "lifecycle_sell_gate",
            "alpha_opportunity_value",
            "hold_continuation_value",
            "sell_release_value",
            "cash_defense_value",
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
            "direct_action_deploy_rank_score",
            "deploy_value_target",
            "release_value_target",
            "deploy_gate_target",
            "release_gate_target",
            "deploy_executability_target",
        ],
    )
    working = _compute_held_side_support_columns(working)
    realized_sell_mask = working.get("weight_change_action", pd.Series("", index=working.index)).astype(str).str.lower().isin(
        {"reduce", "exit"}
    )
    held_side_origin_mask = working.get("sell_execution_origin", pd.Series("", index=working.index)).isin(
        {"deploy_funding_rebalance", "model_release_signal", "direct_action_pair_reallocation"}
    )
    detail_frame = working.loc[realized_sell_mask & held_side_origin_mask].copy()
    if detail_frame.empty:
        return pd.DataFrame(), {
            "event_count": 0,
            "csv_path": "",
            "summary_json": "",
            "origin_counts": {},
            "top_symbols_by_event_count": [],
            "top_symbols_by_positive_forward_excess": [],
            "top_conflict_rows": [],
            "top_release_consistent_rows": [],
        }
    detail_frame["support_alignment"] = np.where(
        detail_frame["held_side_release_consistent"] > 0.5,
        "release_consistent",
        np.where(
            detail_frame["held_side_against_protected_hold"] > 0.5,
            "protected_hold_conflict",
            "ambiguous",
        ),
    )
    detail_frame["abs_delta_weight"] = detail_frame["delta_weight"].fillna(0.0).abs()
    detail_frame = detail_frame.sort_values(
        ["date", "sell_execution_origin", "held_side_support_gap", "forward_excess_5d"],
        ascending=[True, True, True, False],
    )
    detail_columns = [
        "date",
        "stock",
        "model_action",
        "weight_change_action",
        "sell_execution_origin",
        "support_alignment",
        "current_weight",
        "target_weight",
        "delta_weight",
        "abs_delta_weight",
        "hold_days_before",
        "forward_excess_5d",
        "disciplined_funding_need",
        "protected_hold_support",
        "funding_release_support",
        "held_side_support_gap",
        "held_side_against_protected_hold",
        "held_side_release_consistent",
        "sell_rank_score",
        "lifecycle_sell_gate",
        "alpha_opportunity_value",
        "hold_continuation_value",
        "sell_release_value",
        "cash_defense_value",
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
        "deploy_value_target",
        "release_value_target",
        "deploy_gate_target",
        "release_gate_target",
        "deploy_executability_target",
    ]
    detail_frame = detail_frame[[column for column in detail_columns if column in detail_frame.columns]].copy()
    top_symbols = (
        detail_frame.groupby("stock", dropna=False)
        .agg(
            event_count=("stock", "count"),
            avg_forward_excess_5d=("forward_excess_5d", "mean"),
            avg_support_gap=("held_side_support_gap", "mean"),
            avg_disciplined_funding_need=("disciplined_funding_need", "mean"),
        )
        .reset_index()
        .sort_values(["event_count", "avg_forward_excess_5d"], ascending=[False, False])
    )
    positive_forward_symbols = (
        detail_frame.groupby("stock", dropna=False)
        .agg(
            event_count=("stock", "count"),
            total_positive_forward_excess_5d=("forward_excess_5d", lambda values: float(values.clip(lower=0.0).sum())),
            avg_forward_excess_5d=("forward_excess_5d", "mean"),
            avg_support_gap=("held_side_support_gap", "mean"),
        )
        .reset_index()
        .sort_values(["total_positive_forward_excess_5d", "event_count"], ascending=[False, False])
    )

    def _detail_rows(frame: pd.DataFrame, *, ascending: bool) -> list[dict[str, Any]]:
        if frame.empty:
            return []
        trimmed = frame.head(max(int(detail_limit), 1)).copy()
        rows: list[dict[str, Any]] = []
        for item in trimmed.to_dict(orient="records"):
            rows.append(
                {
                    "date": str(item.get("date", "") or ""),
                    "stock": str(item.get("stock", "") or ""),
                    "sell_execution_origin": str(item.get("sell_execution_origin", "") or ""),
                    "support_alignment": str(item.get("support_alignment", "") or ""),
                    "current_weight": float(item.get("current_weight", 0.0) or 0.0),
                    "target_weight": float(item.get("target_weight", 0.0) or 0.0),
                    "delta_weight": float(item.get("delta_weight", 0.0) or 0.0),
                    "hold_days_before": float(item.get("hold_days_before", 0.0) or 0.0),
                    "forward_excess_5d": float(item.get("forward_excess_5d", 0.0) or 0.0),
                    "disciplined_funding_need": float(item.get("disciplined_funding_need", 0.0) or 0.0),
                    "protected_hold_support": float(item.get("protected_hold_support", 0.0) or 0.0),
                    "funding_release_support": float(item.get("funding_release_support", 0.0) or 0.0),
                    "held_side_support_gap": float(item.get("held_side_support_gap", 0.0) or 0.0),
                }
            )
        return rows

    conflict_rows = detail_frame.sort_values(
        ["held_side_support_gap", "forward_excess_5d"],
        ascending=[True, False],
    )
    release_rows = detail_frame.sort_values(
        ["held_side_support_gap", "forward_excess_5d"],
        ascending=[False, False],
    )
    summary_payload = {
        "run_tag": run_tag,
        "event_count": int(len(detail_frame)),
        "origin_counts": {
            str(key): int(value)
            for key, value in detail_frame["sell_execution_origin"].astype(str).value_counts().sort_index().items()
        },
        "alignment_counts": {
            str(key): int(value)
            for key, value in detail_frame["support_alignment"].astype(str).value_counts().sort_index().items()
        },
        "top_symbols_by_event_count": [
            {
                "stock": str(item.get("stock", "") or ""),
                "event_count": int(item.get("event_count", 0) or 0),
                "avg_forward_excess_5d": float(item.get("avg_forward_excess_5d", 0.0) or 0.0),
                "avg_support_gap": float(item.get("avg_support_gap", 0.0) or 0.0),
                "avg_disciplined_funding_need": float(item.get("avg_disciplined_funding_need", 0.0) or 0.0),
            }
            for item in top_symbols.head(max(int(detail_limit), 1)).to_dict(orient="records")
        ],
        "top_symbols_by_positive_forward_excess": [
            {
                "stock": str(item.get("stock", "") or ""),
                "event_count": int(item.get("event_count", 0) or 0),
                "total_positive_forward_excess_5d": float(item.get("total_positive_forward_excess_5d", 0.0) or 0.0),
                "avg_forward_excess_5d": float(item.get("avg_forward_excess_5d", 0.0) or 0.0),
                "avg_support_gap": float(item.get("avg_support_gap", 0.0) or 0.0),
            }
            for item in positive_forward_symbols.head(max(int(detail_limit), 1)).to_dict(orient="records")
        ],
        "top_conflict_rows": _detail_rows(conflict_rows, ascending=True),
        "top_release_consistent_rows": _detail_rows(release_rows, ascending=False),
    }
    return detail_frame, summary_payload


def _build_semantic_conflicts(
    *,
    action_outcomes: pd.DataFrame,
    turnover_frame: pd.DataFrame,
) -> dict[str, Any]:
    if action_outcomes.empty:
        return {
            "action_rows": 0,
            "semantic_conflict_rate": 0.0,
            "order_translation_conflict_rate": 0.0,
            "weight_change_conflict_rate": 0.0,
            "micro_rebalance_conflict_rate": 0.0,
            "deadband_conflict_rate": 0.0,
            "deadband_order_translation_conflict_rate": 0.0,
            "small_delta_conflict_rate": 0.0,
            "small_delta_order_translation_conflict_rate": 0.0,
            "budget_clipped_day_share": 0.0,
            "budget_clipped_conflict_rate": 0.0,
            "unclipped_conflict_rate": 0.0,
            "budget_clipped_order_translation_conflict_rate": 0.0,
            "unclipped_order_translation_conflict_rate": 0.0,
            "budget_semantics_counts": {},
            "budget_calibration_counts": {},
            "avg_budget_risk_off_score": 0.0,
            "avg_budget_deploy_score": 0.0,
            "avg_budget_entry_candidate_count": 0.0,
            "avg_budget_entry_keep_count": 0.0,
            "avg_budget_held_protected_count": 0.0,
            "avg_budget_reclaimable_held_count": 0.0,
            "avg_budget_released_held_count": 0.0,
            "avg_model_release_signal_count": 0.0,
            "avg_deploy_funding_rebalance_signal_count": 0.0,
            "avg_sell_authorized_held_count": 0.0,
            "avg_sell_source_floor_guard_count": 0.0,
            "avg_budget_split_bound_guard_count": 0.0,
            "avg_budget_sell_priority_guard_count": 0.0,
            "avg_sell_rank_score": 0.0,
            "avg_lifecycle_sell_gate": 0.0,
            "avg_alpha_opportunity_value": 0.0,
            "avg_hold_continuation_value": 0.0,
            "avg_sell_release_value": 0.0,
            "avg_cash_defense_value": 0.0,
            "avg_value_arbitration_target": 0.0,
            "avg_deploy_value_target": 0.0,
            "avg_release_value_target": 0.0,
            "avg_defense_value_target": 0.0,
            "avg_deploy_gate_target": 0.0,
            "avg_release_gate_target": 0.0,
            "avg_defense_gate_target": 0.0,
            "avg_deploy_executability_target": 0.0,
            "direct_action_value_mode_share": 0.0,
            "direct_action_value_label_match_share": 0.0,
            "direct_action_value_selected_mean": 0.0,
            "direct_action_value_gap_mean": 0.0,
            "direct_action_value_low_margin_share": 0.0,
            "direct_action_order_translation_conflict_rate": 0.0,
            "direct_action_intent_preserved_share": 0.0,
            "allocation_intent_v2_mode_share": 0.0,
            "intent_translation_conflict_rate": 0.0,
            "lifecycle_hint_conflict_rate": 0.0,
            "direct_action_funding_authorized_sell_share": 0.0,
            "direct_action_funding_protected_sell_share": 0.0,
            "direct_action_release_advantage_mean": 0.0,
            "direct_action_deploy_advantage_mean": 0.0,
            "direct_action_deploy_signal_count": 0,
            "direct_action_core_deploy_target_count": 0,
            "direct_action_core_deploy_target_realized_rate": 0.0,
            "direct_action_add_authorized_count": 0,
            "direct_action_add_authorized_realized_rate": 0.0,
            "authorized_add_no_weight_change_count": 0,
            "authorized_add_no_weight_change_share": 0.0,
            "direct_action_deploy_authorized_count": 0,
            "direct_action_deploy_authorized_realized_rate": 0.0,
            "direct_action_authorization_subset_violation_count": 0,
            "direct_action_reallocation_source_count": 0,
            "direct_action_pair_reallocation_source_count": 0,
            "direct_action_pair_cost_guard_pass_count": 0,
            "direct_action_pair_cost_guard_blocked_count": 0,
            "direct_action_pair_cost_guard_pass_rate": 0.0,
            "direct_action_pair_source_spread_mean": 0.0,
            "direct_action_pair_source_cost_mean": 0.0,
            "direct_action_pair_source_forward_excess_5d": 0.0,
            "direct_action_core_target_forward_excess_5d": 0.0,
            "direct_action_core_minus_pair_forward_excess_5d": 0.0,
            "portfolio_daily_receiver_candidate_count": 0,
            "portfolio_daily_receiver_exec_guard_count": 0,
            "portfolio_daily_receiver_semantic_no_headroom_count": 0,
            "portfolio_daily_receiver_open_breadth_candidate_count": 0,
            "portfolio_daily_receiver_add_headroom_mean": 0.0,
            "portfolio_daily_receiver_min_add_delta_mean": 0.0,
            "portfolio_daily_receiver_target_count": 0,
            "portfolio_daily_receiver_realized_deploy_rate": 0.0,
            "portfolio_daily_receiver_unrealized_deploy_count": 0,
            "portfolio_daily_receiver_unrealized_deploy_share": 0.0,
            "portfolio_daily_source_candidate_count": 0,
            "portfolio_daily_source_target_count": 0,
            "portfolio_daily_source_sell_count": 0,
            "portfolio_daily_source_sell_share": 0.0,
            "portfolio_daily_source_realized_sell_rate": 0.0,
            "portfolio_daily_source_exec_guard_count": 0,
            "portfolio_daily_source_exec_cap_guard_count": 0,
            "portfolio_daily_source_target_not_sold_count": 0,
            "portfolio_daily_source_target_not_sold_share": 0.0,
            "portfolio_daily_source_realized_reduction_weight": 0.0,
            "portfolio_daily_receiver_realized_deploy_count": 0,
            "portfolio_daily_effective_capital_transfer_count": 0,
            "portfolio_daily_cash_score_mean": 0.0,
            "portfolio_daily_cash_reserve_rate": 0.0,
            "portfolio_daily_budget_closed": 0.0,
            "portfolio_daily_deployment_required": 0.0,
            "portfolio_daily_risk_reduction_required": 0.0,
            "portfolio_daily_receiver_activity_required": 0.0,
            "portfolio_daily_receiver_score_mean": 0.0,
            "portfolio_daily_source_score_mean": 0.0,
            "portfolio_daily_source_gap_mean": 0.0,
            "portfolio_daily_source_forward_proxy_keep_risk_mean": 0.0,
            "portfolio_daily_source_release_conviction_mean": 0.0,
            "portfolio_daily_receiver_forward_excess_5d": 0.0,
            "portfolio_daily_source_forward_excess_5d": 0.0,
            "portfolio_daily_source_positive_forward_sell_share": 0.0,
            "portfolio_daily_source_strong_positive_forward_sell_count": 0,
            "portfolio_daily_source_max_forward_excess_5d": 0.0,
            "portfolio_daily_source_p75_forward_excess_5d": 0.0,
            "portfolio_daily_receiver_minus_source_forward_excess_5d": 0.0,
            "avg_gross_exposure_target": 0.0,
            "portfolio_daily_exposure_utilization": 0.0,
            "deploy_intent_action_count": 0,
            "deploy_intent_realized_count": 0,
            "deploy_intent_realized_rate": 0.0,
            "deploy_intent_unrealized_count": 0,
            "deploy_intent_unrealized_share": 0.0,
            "open_add_positive_weight_change_rate": 0.0,
            "add_to_hold_conflict_count": 0,
            "add_to_hold_conflict_share": 0.0,
            "deploy_intent_hold_conflict_share": 0.0,
            "deploy_intent_dropped_count": 0,
            "deploy_intent_dropped_share": 0.0,
            "avg_deploy_intent_candidate_count": 0.0,
            "avg_deploy_intent_candidate_realized_rate": 0.0,
            "avg_deploy_intent_candidate_budget_drop_share": 0.0,
            "sell_intent_action_count": 0,
            "sell_intent_realized_count": 0,
            "sell_intent_realized_rate": 0.0,
            "sell_intent_hold_conflict_count": 0,
            "sell_intent_hold_conflict_share": 0.0,
            "sell_intent_suppressed_count": 0,
            "sell_intent_suppressed_share": 0.0,
            "realized_sell_action_count": 0,
            "budget_origin_sell_count": 0,
            "budget_origin_sell_share": 0.0,
            "model_release_signal_sell_count": 0,
            "model_release_signal_sell_share": 0.0,
            "deploy_funding_rebalance_sell_count": 0,
            "deploy_funding_rebalance_sell_share": 0.0,
            "direct_action_pair_reallocation_sell_count": 0,
            "direct_action_pair_reallocation_sell_share": 0.0,
            "budget_slot_reclaim_sell_count": 0,
            "budget_slot_reclaim_sell_share": 0.0,
            "sell_priority_guard_sell_count": 0,
            "sell_priority_guard_sell_share": 0.0,
            "turnover_trim_sell_count": 0,
            "turnover_trim_sell_share": 0.0,
            "forced_zero_sell_count": 0,
            "forced_zero_sell_share": 0.0,
            "model_origin_sell_forward_excess_5d": 0.0,
            "model_authorized_sell_forward_excess_5d": 0.0,
            "model_release_signal_forward_excess_5d": 0.0,
            "deploy_funding_rebalance_forward_excess_5d": 0.0,
            "budget_origin_sell_forward_excess_5d": 0.0,
            "avg_disciplined_funding_need": 0.0,
            "avg_protected_hold_support": 0.0,
            "avg_funding_release_support": 0.0,
            "model_release_signal_disciplined_funding_need": 0.0,
            "model_release_signal_keep_support": 0.0,
            "model_release_signal_release_support": 0.0,
            "deploy_funding_disciplined_funding_need": 0.0,
            "deploy_funding_keep_support": 0.0,
            "deploy_funding_release_support": 0.0,
            "model_release_against_protected_hold_share": 0.0,
            "model_release_release_consistent_share": 0.0,
            "deploy_funding_against_protected_hold_share": 0.0,
            "deploy_funding_release_consistent_share": 0.0,
            "avg_clipped_intent_risk": 0.0,
            "clipped_intent_risk_conflict_gap": 0.0,
            "high_cash_up_market_share": 0.0,
            "high_cash_down_market_share": 0.0,
            "high_cash_sell_action_count": 0,
            "high_cash_budget_origin_sell_share": 0.0,
            "sell_execution_origin_counts": {},
            "sell_suppression_origin_counts": {},
            "sell_source_floor_guard_count": 0,
            "sell_source_floor_guard_share": 0.0,
            **_release_translation_deploy_health(
                deploy_intent_action_count=0.0,
                deploy_intent_realized_rate=0.0,
                order_translation_conflict_rate=0.0,
                add_to_hold_conflict_share=0.0,
                sell_intent_suppressed_share=0.0,
                budget_origin_sell_share=0.0,
                deploy_funding_rebalance_sell_count=0.0,
                deploy_funding_rebalance_sell_share=0.0,
                deploy_funding_rebalance_forward_excess_5d=0.0,
                deploy_funding_against_protected_hold_share=0.0,
                deploy_funding_release_consistent_share=0.0,
                model_release_signal_sell_count=0.0,
                model_release_signal_forward_excess_5d=0.0,
                model_release_against_protected_hold_share=0.0,
                model_release_release_consistent_share=0.0,
            ),
            "top_action_pairs": [],
            "top_conflict_pairs": [],
            "top_weight_change_pairs": [],
            "top_order_translation_conflict_pairs": [],
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
            "sell_rank_score",
            "lifecycle_sell_gate",
            "alpha_opportunity_value",
            "hold_continuation_value",
            "sell_release_value",
            "cash_defense_value",
            "value_arbitration_target",
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
            "deploy_value_target",
            "release_value_target",
            "defense_value_target",
            "deploy_gate_target",
            "release_gate_target",
            "defense_gate_target",
            "deploy_executability_target",
            "clipped_intent_risk",
            "sell_authorization_score",
            "direct_action_value_applied",
            "direct_action_value_selected",
            "direct_action_value_gap",
            "direct_action_release_advantage",
            "direct_action_deploy_advantage",
            "direct_action_deploy_rank_score",
            "direct_action_pair_opportunity_spread",
            "direct_action_pair_source_opportunity_cost",
            "direct_action_pair_source_release_score",
            "portfolio_daily_receiver_score",
            "portfolio_daily_receiver_add_headroom",
            "portfolio_daily_receiver_min_add_delta",
            "portfolio_daily_source_gap",
            "portfolio_daily_source_score",
            "portfolio_daily_cash_score",
            "portfolio_daily_target_weight_intent",
            "portfolio_daily_target_delta_intent",
        ],
    ).copy()
    working["model_action"] = working.get("model_action", pd.Series("", index=working.index)).astype(str)
    working["effective_model_action"] = working.get(
        "portfolio_daily_effective_model_action",
        working["model_action"],
    ).astype(str)
    working["execution_action"] = working.get("execution_action", pd.Series("", index=working.index)).astype(str)
    working["weight_change_action"] = working.get("weight_change_action", working["execution_action"]).astype(str)
    working["is_semantic_conflict"] = working["effective_model_action"] != working["execution_action"]
    working["is_order_translation_conflict"] = working["effective_model_action"] != working["weight_change_action"]
    working["is_conflict"] = working["is_semantic_conflict"]
    working["abs_delta_weight"] = working["delta_weight"].abs()
    working["contradictory_micro_rebalance"] = working.get(
        "contradictory_micro_rebalance",
        pd.Series(False, index=working.index),
    ).map(_safe_bool)
    for bool_column in (
        "budget_dropped",
        "budget_released_from_hold",
        "forced_zero",
        "semantic_delta_guarded",
        "budget_split_bound_guarded",
        "translation_floor_guarded",
        "translation_cap_guarded",
        "translation_soft_lift_guarded",
        "sell_priority_guarded",
        "turnover_intent_guarded",
        "sell_source_floor_guarded",
        "model_release_signal",
        "deploy_funding_rebalance_signal",
        "sell_authorized_by_model",
        "direct_action_funding_release_authorized",
        "direct_action_funding_protected",
        "direct_action_add_authorized",
        "direct_action_open_authorized",
        "direct_action_deploy_authorized",
        "direct_action_deploy_signal",
        "direct_action_core_deploy_target",
        "direct_action_reallocation_source",
        "direct_action_pair_reallocation_source",
        "direct_action_pair_cost_guard_pass",
        "direct_action_pair_cost_guard_blocked",
        "portfolio_daily_receiver_candidate",
        "portfolio_daily_receiver_target",
        "portfolio_daily_receiver_exec_guarded",
        "portfolio_daily_receiver_semantic_no_headroom",
        "portfolio_daily_receiver_open_breadth_candidate",
        "portfolio_daily_source_candidate",
        "portfolio_daily_source_target",
        "portfolio_daily_cash_reserve_signal",
        "allocation_intent_v2_mode",
    ):
        working[bool_column] = working.get(
            bool_column,
            pd.Series(False, index=working.index),
        ).map(_safe_bool)
    working["sell_execution_origin"] = working.get(
        "sell_execution_origin",
        pd.Series("none", index=working.index),
    ).astype(str)
    working["sell_suppression_origin"] = working.get(
        "sell_suppression_origin",
        pd.Series("none", index=working.index),
    ).astype(str)
    working["deadband_active"] = working.get(
        "execution_deadband",
        pd.Series(0.0, index=working.index),
    ).fillna(0.0) > 1e-12
    working["small_delta_conflict"] = working["is_semantic_conflict"] & (
        working["abs_delta_weight"] <= working.get("execution_deadband", pd.Series(0.0, index=working.index)).fillna(0.0)
    )
    working = _compute_held_side_support_columns(working)
    working["small_delta_order_translation_conflict"] = working["is_order_translation_conflict"] & (
        working["abs_delta_weight"] <= working.get("execution_deadband", pd.Series(0.0, index=working.index)).fillna(0.0)
    )
    intent_delta_abs = working.get(
        "portfolio_daily_target_delta_intent",
        pd.Series(0.0, index=working.index),
    ).abs()
    intent_deadband = working.get("execution_deadband", pd.Series(0.0, index=working.index)).fillna(0.0).clip(
        lower=1.0e-8
    )
    def _deadbanded_weight_change_action(previous_weight: float, new_weight: float, deadband: float) -> str:
        if abs(float(new_weight) - float(previous_weight)) <= float(deadband):
            return "hold" if float(previous_weight) > 1.0e-8 else "skip"
        return _weight_change_action(float(previous_weight), float(new_weight))

    working["intent_weight_change_action"] = [
        _deadbanded_weight_change_action(
            float(previous_weight),
            float(intent_weight),
            float(deadband),
        )
        for previous_weight, intent_weight, deadband in zip(
            working["current_weight"].fillna(0.0),
            working.get("portfolio_daily_target_weight_intent", pd.Series(0.0, index=working.index)).fillna(0.0),
            intent_deadband,
        )
    ]
    working["final_weight_change_action_deadbanded"] = [
        _deadbanded_weight_change_action(
            float(previous_weight),
            float(target_weight),
            float(deadband),
        )
        for previous_weight, target_weight, deadband in zip(
            working["current_weight"].fillna(0.0),
            working.get("target_weight", pd.Series(0.0, index=working.index)).fillna(0.0),
            intent_deadband,
        )
    ]
    working["intent_translation_active"] = working["allocation_intent_v2_mode"] & (
        (intent_delta_abs > intent_deadband) | (working["abs_delta_weight"] > intent_deadband)
    )
    working["intent_translation_conflict"] = working["intent_translation_active"] & (
        (working["intent_weight_change_action"] != working["final_weight_change_action_deadbanded"])
        & (intent_delta_abs > intent_deadband)
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
            "gross_exposure",
            "gross_exposure_target",
            "gross_exposure_target_raw",
            "allocation_layer_cash_after",
            "allocation_layer_available_cash_to_deploy",
            "allocation_layer_stock_budget",
            "allocation_layer_target_weight_sum",
            "allocation_layer_target_sum_gap",
            "allocation_layer_cash_funded_deploy_amount",
            "allocation_layer_source_funded_deploy_amount",
            "allocation_layer_unused_receiver_headroom",
            "allocation_layer_receiver_headroom_utilization",
            "allocation_layer_source_release_required",
            "release_first_allocation_v3_mode_used",
            "release_first_allocation_v3_used",
            "release_first_intent_target_count",
            "release_first_source_intent_count",
            "release_first_source_realized_count",
            "release_first_rotation_amount",
            "release_first_cash_buffer_amount",
            "allocation_layer_budget_closed",
            "allocation_layer_deployment_required",
            "allocation_layer_risk_reduction_required",
            "allocation_layer_receiver_activity_required",
            "allocation_intent_v2_mode_used",
            "intent_translation_conflict_count",
            "intent_translation_active_count",
            "intent_translation_conflict_rate",
            "lifecycle_hint_conflict_rate",
            "allocation_layer_objective_value",
            "allocation_layer_receiver_executable_candidate_count",
            "allocation_layer_receiver_target_count",
            "allocation_layer_source_target_count",
            "allocation_layer_native_fallback_used",
            "candidate_budget",
            "candidate_budget_raw",
            "turnover_budget",
            "turnover_budget_raw",
            "budget_risk_off_score",
            "budget_deploy_score",
            "budget_entry_candidate_count",
            "budget_entry_keep_count",
            "budget_held_protected_count",
            "budget_reclaimable_held_count",
            "budget_released_held_count",
            "model_release_signal_count",
            "deploy_funding_rebalance_signal_count",
            "sell_authorized_held_count",
            "budget_split_bound_guard_count",
            "budget_sell_priority_guard_count",
            "sell_source_floor_guard_count",
            "authorized_add_no_weight_change_count",
            "authorized_add_no_weight_change_share",
            "deploy_intent_unrealized_count",
            "deploy_intent_unrealized_share",
            "portfolio_daily_receiver_semantic_no_headroom_count",
            "portfolio_daily_receiver_open_breadth_candidate_count",
            "portfolio_daily_cash_reserve_signal",
            "portfolio_daily_receiver_candidate_count",
            "portfolio_daily_receiver_target_count",
            "portfolio_daily_source_target_count",
            "deploy_intent_candidate_count",
            "deploy_intent_candidate_realized_count",
            "deploy_intent_candidate_realized_rate",
            "deploy_intent_candidate_budget_drop_count",
            "deploy_intent_candidate_budget_drop_share",
            "sell_intent_action_count",
            "sell_intent_realized_count",
            "sell_intent_realized_rate",
            "sell_intent_hold_conflict_count",
            "sell_intent_hold_conflict_share",
            "sell_intent_suppressed_count",
            "sell_intent_suppressed_share",
            "realized_sell_action_count",
            "budget_origin_sell_count",
            "budget_origin_sell_share",
            "model_release_signal_sell_count",
            "model_release_signal_sell_share",
            "deploy_funding_rebalance_sell_count",
            "deploy_funding_rebalance_sell_share",
            "direct_action_pair_reallocation_sell_count",
            "direct_action_pair_reallocation_sell_share",
            "budget_slot_reclaim_sell_count",
            "budget_slot_reclaim_sell_share",
            "sell_priority_guard_sell_count",
            "sell_priority_guard_sell_share",
            "turnover_trim_sell_count",
            "turnover_trim_sell_share",
            "forced_zero_sell_count",
            "forced_zero_sell_share",
        ],
    )
    if turnover_working.empty:
        day_merge = by_date.copy()
        day_merge["budget_clipped"] = False
        day_merge["budget_drop_count"] = 0.0
        day_merge["cash_weight"] = np.nan
        allocation_closure_frame = day_merge
    else:
        turnover_working = turnover_working.copy()
        turnover_working["date"] = turnover_working["date"].astype(str)
        turnover_working["budget_clipped"] = (
            turnover_working["raw_turnover"].fillna(0.0) > turnover_working["realized_turnover"].fillna(0.0) + 1e-8
        )
        for optional_column in (
            "gross_exposure",
            "gross_exposure_target_raw",
            "allocation_layer_cash_after",
            "allocation_layer_available_cash_to_deploy",
            "allocation_layer_stock_budget",
            "allocation_layer_target_weight_sum",
            "allocation_layer_target_sum_gap",
            "allocation_layer_cash_funded_deploy_amount",
            "allocation_layer_source_funded_deploy_amount",
            "allocation_layer_unused_receiver_headroom",
            "allocation_layer_receiver_headroom_utilization",
            "allocation_layer_source_release_required",
            "release_first_allocation_v3_mode_used",
            "release_first_allocation_v3_used",
            "release_first_intent_target_count",
            "release_first_source_intent_count",
            "release_first_source_realized_count",
            "release_first_rotation_amount",
            "release_first_cash_buffer_amount",
            "allocation_layer_budget_closed",
            "allocation_layer_deployment_required",
            "allocation_layer_risk_reduction_required",
            "allocation_layer_receiver_activity_required",
            "allocation_intent_v2_mode_used",
            "intent_translation_conflict_count",
            "intent_translation_active_count",
            "intent_translation_conflict_rate",
            "lifecycle_hint_conflict_rate",
            "allocation_layer_objective_value",
            "allocation_layer_receiver_executable_candidate_count",
            "allocation_layer_receiver_target_count",
            "allocation_layer_source_target_count",
            "allocation_layer_native_fallback_used",
            "candidate_budget",
            "candidate_budget_raw",
            "turnover_budget",
            "turnover_budget_raw",
            "budget_risk_off_score",
            "budget_deploy_score",
            "budget_entry_candidate_count",
            "budget_entry_keep_count",
            "budget_held_protected_count",
            "budget_reclaimable_held_count",
            "budget_released_held_count",
            "model_release_signal_count",
            "deploy_funding_rebalance_signal_count",
            "sell_authorized_held_count",
            "budget_split_bound_guard_count",
            "budget_sell_priority_guard_count",
            "sell_source_floor_guard_count",
            "authorized_add_no_weight_change_count",
            "authorized_add_no_weight_change_share",
            "deploy_intent_unrealized_count",
            "deploy_intent_unrealized_share",
            "portfolio_daily_receiver_semantic_no_headroom_count",
            "portfolio_daily_receiver_open_breadth_candidate_count",
            "portfolio_daily_cash_reserve_signal",
            "portfolio_daily_receiver_candidate_count",
            "portfolio_daily_receiver_target_count",
            "portfolio_daily_source_target_count",
            "deploy_intent_candidate_count",
            "deploy_intent_candidate_realized_count",
            "deploy_intent_candidate_realized_rate",
            "deploy_intent_candidate_budget_drop_count",
            "deploy_intent_candidate_budget_drop_share",
            "sell_intent_action_count",
            "sell_intent_realized_count",
            "sell_intent_realized_rate",
            "sell_intent_hold_conflict_count",
            "sell_intent_hold_conflict_share",
            "sell_intent_suppressed_count",
            "sell_intent_suppressed_share",
            "realized_sell_action_count",
            "budget_origin_sell_count",
            "budget_origin_sell_share",
            "model_release_signal_sell_count",
            "model_release_signal_sell_share",
            "deploy_funding_rebalance_sell_count",
            "deploy_funding_rebalance_sell_share",
            "direct_action_pair_reallocation_sell_count",
            "direct_action_pair_reallocation_sell_share",
            "budget_slot_reclaim_sell_count",
            "budget_slot_reclaim_sell_share",
            "sell_priority_guard_sell_count",
            "sell_priority_guard_sell_share",
            "turnover_trim_sell_count",
            "turnover_trim_sell_share",
            "forced_zero_sell_count",
            "forced_zero_sell_share",
        ):
            if optional_column not in turnover_working.columns:
                turnover_working[optional_column] = 0.0
        allocation_closure_frame = turnover_working
        day_merge = by_date.merge(
            turnover_working[
                [
                    "date",
                    "budget_clipped",
                    "budget_drop_count",
                    "cash_weight",
                    "gross_exposure",
                    "raw_turnover",
                    "realized_turnover",
                    "gross_exposure_target",
                    "gross_exposure_target_raw",
                    "allocation_layer_target_sum_gap",
                    "allocation_layer_budget_closed",
                    "allocation_layer_deployment_required",
                    "allocation_layer_risk_reduction_required",
                    "allocation_layer_receiver_activity_required",
                    "allocation_intent_v2_mode_used",
                    "intent_translation_conflict_count",
                    "intent_translation_active_count",
                    "intent_translation_conflict_rate",
                    "lifecycle_hint_conflict_rate",
                    "candidate_budget",
                    "candidate_budget_raw",
                    "turnover_budget",
                    "turnover_budget_raw",
                    "budget_risk_off_score",
                    "budget_deploy_score",
                    "budget_entry_candidate_count",
                    "budget_entry_keep_count",
                    "budget_held_protected_count",
                    "budget_reclaimable_held_count",
                    "budget_released_held_count",
                    "model_release_signal_count",
                    "deploy_funding_rebalance_signal_count",
                    "sell_authorized_held_count",
                    "budget_split_bound_guard_count",
                    "budget_sell_priority_guard_count",
                    "sell_source_floor_guard_count",
                    "authorized_add_no_weight_change_count",
                    "authorized_add_no_weight_change_share",
                    "deploy_intent_unrealized_count",
                    "deploy_intent_unrealized_share",
                    "portfolio_daily_receiver_semantic_no_headroom_count",
                    "portfolio_daily_receiver_open_breadth_candidate_count",
                    "deploy_intent_candidate_count",
                    "deploy_intent_candidate_realized_count",
                    "deploy_intent_candidate_realized_rate",
                    "deploy_intent_candidate_budget_drop_count",
                    "deploy_intent_candidate_budget_drop_share",
                    "sell_intent_action_count",
                    "sell_intent_realized_count",
                    "sell_intent_realized_rate",
                    "sell_intent_hold_conflict_count",
                    "sell_intent_hold_conflict_share",
                    "sell_intent_suppressed_count",
                    "sell_intent_suppressed_share",
                    "realized_sell_action_count",
                    "budget_origin_sell_count",
                    "budget_origin_sell_share",
                    "model_release_signal_sell_count",
                    "model_release_signal_sell_share",
                    "deploy_funding_rebalance_sell_count",
                    "deploy_funding_rebalance_sell_share",
                    "direct_action_pair_reallocation_sell_count",
                    "direct_action_pair_reallocation_sell_share",
                    "budget_slot_reclaim_sell_count",
                    "budget_slot_reclaim_sell_share",
                    "sell_priority_guard_sell_count",
                    "sell_priority_guard_sell_share",
                    "turnover_trim_sell_count",
                    "turnover_trim_sell_share",
                    "forced_zero_sell_count",
                    "forced_zero_sell_share",
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
    day_merge["high_cash_day"] = high_cash_mask.astype(bool)
    high_cash_lookup = {
        str(key): bool(value)
        for key, value in day_merge.set_index("date")["high_cash_day"].to_dict().items()
    }
    working["high_cash_day"] = working["date"].astype(str).map(high_cash_lookup).fillna(False)

    diagnoses: list[str] = []
    semantic_conflict_rate = _safe_mean(working["is_conflict"].astype(float))
    order_translation_conflict_rate = _safe_mean(working["is_order_translation_conflict"].astype(float))
    allocation_intent_v2_mode_share = _safe_mean(working["allocation_intent_v2_mode"].astype(float))
    if bool(working["intent_translation_active"].any()):
        intent_active_rows = working.loc[working["intent_translation_active"]]
        intent_translation_conflict_rate = _safe_mean(intent_active_rows["intent_translation_conflict"].astype(float))
    else:
        intent_translation_conflict_rate = _safe_mean(
            day_merge.get("intent_translation_conflict_rate", pd.Series(0.0, index=day_merge.index)).fillna(0.0)
        )
    lifecycle_hint_conflict_rate = semantic_conflict_rate if allocation_intent_v2_mode_share >= 0.5 else 0.0
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
    avg_sell_rank_score = _safe_mean(working.get("sell_rank_score", pd.Series(0.0, index=working.index)).fillna(0.0))
    avg_lifecycle_sell_gate = _safe_mean(
        working.get("lifecycle_sell_gate", pd.Series(0.0, index=working.index)).fillna(0.0)
    )
    avg_alpha_opportunity_value = _safe_mean(
        working.get("alpha_opportunity_value", pd.Series(0.0, index=working.index)).fillna(0.0)
    )
    avg_hold_continuation_value = _safe_mean(
        working.get("hold_continuation_value", pd.Series(0.0, index=working.index)).fillna(0.0)
    )
    avg_sell_release_value = _safe_mean(
        working.get("sell_release_value", pd.Series(0.0, index=working.index)).fillna(0.0)
    )
    avg_cash_defense_value = _safe_mean(
        working.get("cash_defense_value", pd.Series(0.0, index=working.index)).fillna(0.0)
    )
    avg_multi_horizon_forward_value = _safe_mean(
        working.get("multi_horizon_forward_value", pd.Series(0.0, index=working.index)).fillna(0.0)
    )
    avg_multi_horizon_forward_risk = _safe_mean(
        working.get("multi_horizon_forward_risk", pd.Series(0.0, index=working.index)).fillna(0.0)
    )
    avg_multi_horizon_path_value = _safe_mean(
        working.get("multi_horizon_path_value", pd.Series(0.0, index=working.index)).fillna(0.0)
    )
    avg_open_action_value = _safe_mean(
        working.get("open_action_value", pd.Series(0.0, index=working.index)).fillna(0.0)
    )
    avg_add_action_value = _safe_mean(
        working.get("add_action_value", pd.Series(0.0, index=working.index)).fillna(0.0)
    )
    avg_hold_action_value = _safe_mean(
        working.get("hold_action_value", pd.Series(0.0, index=working.index)).fillna(0.0)
    )
    avg_reduce_action_value = _safe_mean(
        working.get("reduce_action_value", pd.Series(0.0, index=working.index)).fillna(0.0)
    )
    avg_exit_action_value = _safe_mean(
        working.get("exit_action_value", pd.Series(0.0, index=working.index)).fillna(0.0)
    )
    avg_action_value_consistency_target = _safe_mean(
        working.get("action_value_consistency_target", pd.Series(0.5, index=working.index)).fillna(0.5)
    )
    avg_value_arbitration_target = _safe_mean(
        working.get("value_arbitration_target", pd.Series(0.5, index=working.index)).fillna(0.5)
    )
    avg_deploy_value_target = _safe_mean(
        working.get("deploy_value_target", pd.Series(0.0, index=working.index)).fillna(0.0)
    )
    avg_release_value_target = _safe_mean(
        working.get("release_value_target", pd.Series(0.0, index=working.index)).fillna(0.0)
    )
    avg_defense_value_target = _safe_mean(
        working.get("defense_value_target", pd.Series(0.0, index=working.index)).fillna(0.0)
    )
    avg_deploy_gate_target = _safe_mean(
        working.get("deploy_gate_target", pd.Series(0.0, index=working.index)).fillna(0.0)
    )
    avg_release_gate_target = _safe_mean(
        working.get("release_gate_target", pd.Series(0.0, index=working.index)).fillna(0.0)
    )
    avg_defense_gate_target = _safe_mean(
        working.get("defense_gate_target", pd.Series(0.0, index=working.index)).fillna(0.0)
    )
    avg_deploy_executability_target = _safe_mean(
        working.get("deploy_executability_target", pd.Series(0.0, index=working.index)).fillna(0.0)
    )
    disciplined_funding_need = working.get("disciplined_funding_need", pd.Series(0.0, index=working.index)).fillna(0.0)
    protected_hold_support = working.get("protected_hold_support", pd.Series(0.0, index=working.index)).fillna(0.0)
    funding_release_support = working.get("funding_release_support", pd.Series(0.0, index=working.index)).fillna(0.0)
    support_margin = 0.10
    avg_disciplined_funding_need = _safe_mean(disciplined_funding_need)
    avg_protected_hold_support = _safe_mean(protected_hold_support)
    avg_funding_release_support = _safe_mean(funding_release_support)
    model_action_lookup = working["effective_model_action"].astype(str).str.lower()
    weight_change_lookup = working["weight_change_action"].astype(str).str.lower()
    deploy_intent_mask = model_action_lookup.isin({"open", "add"})
    add_intent_mask = model_action_lookup == "add"
    sell_intent_mask = model_action_lookup.isin({"reduce", "exit"})
    realized_sell_mask = weight_change_lookup.isin({"reduce", "exit"})
    deploy_realized_mask = weight_change_lookup.isin({"open", "add"})
    positive_weight_change_mask = working["delta_weight"].fillna(0.0) > 1.0e-8
    add_to_hold_mask = add_intent_mask & (weight_change_lookup == "hold")
    deploy_hold_mask = deploy_intent_mask & (weight_change_lookup == "hold")
    sell_intent_hold_mask = sell_intent_mask & (weight_change_lookup == "hold")
    sell_intent_realized_mask = sell_intent_mask & realized_sell_mask
    sell_intent_suppressed_mask = sell_intent_mask & (~realized_sell_mask)
    model_authorized_sell_origin_mask = working["sell_execution_origin"].isin(
        {
            "model_sell_intent",
            "model_release_signal",
            "deploy_funding_rebalance",
            "direct_action_pair_reallocation",
            "portfolio_daily_ranking_source",
        }
    )
    action_value_table = pd.DataFrame(
        {
            "open": working.get("open_action_value", pd.Series(0.0, index=working.index)).fillna(0.0),
            "add": working.get("add_action_value", pd.Series(0.0, index=working.index)).fillna(0.0),
            "hold": working.get("hold_action_value", pd.Series(0.0, index=working.index)).fillna(0.0),
            "reduce": working.get("reduce_action_value", pd.Series(0.0, index=working.index)).fillna(0.0),
            "exit": working.get("exit_action_value", pd.Series(0.0, index=working.index)).fillna(0.0),
        },
        index=working.index,
    )
    best_action_value = action_value_table.max(axis=1)
    chosen_action_value = pd.Series(0.0, index=working.index, dtype=float)
    for action_name in ("open", "add", "hold", "reduce", "exit"):
        chosen_action_value = chosen_action_value.where(model_action_lookup != action_name, action_value_table[action_name])
    action_value_conflict_mask = (
        model_action_lookup.isin({"open", "add", "hold", "reduce", "exit"})
        & (best_action_value > chosen_action_value + 0.08)
    )
    keep_action_value = pd.concat(
        [
            action_value_table["add"],
            action_value_table["hold"],
        ],
        axis=1,
    ).max(axis=1)
    release_action_value = pd.concat(
        [
            action_value_table["reduce"],
            action_value_table["exit"],
        ],
        axis=1,
    ).max(axis=1)
    held_value_mask = working["hold_days_before"].fillna(0.0) > 0.0
    sell_against_keep_value_mask = (
        held_value_mask
        & sell_intent_mask
        & (keep_action_value > release_action_value + 0.08)
    )
    keep_against_release_value_mask = (
        held_value_mask
        & model_action_lookup.isin({"add", "hold"})
        & (release_action_value > keep_action_value + 0.08)
    )
    open_low_action_value_mask = (model_action_lookup == "open") & (action_value_table["open"] < 0.30)
    action_value_conflict_share = _safe_mean(action_value_conflict_mask.astype(float))
    sell_against_keep_value_share = float(
        sell_against_keep_value_mask.sum() / max(float(sell_intent_mask.sum()), 1.0)
    )
    keep_against_release_value_share = float(
        keep_against_release_value_mask.sum() / max(float(model_action_lookup.isin({"add", "hold"}).sum()), 1.0)
    )
    open_low_action_value_share = float(open_low_action_value_mask.sum() / max(float((model_action_lookup == "open").sum()), 1.0))
    action_value_consistency_score = float(
        np.clip(
            1.0
            - action_value_conflict_share * 0.46
            - sell_against_keep_value_share * 0.28
            - keep_against_release_value_share * 0.18
            - open_low_action_value_share * 0.08,
            0.0,
            1.0,
        )
    )
    policy_decision_mode_lookup = working.get(
        "policy_decision_mode",
        pd.Series("", index=working.index),
    ).astype(str).str.lower()
    direct_action_label_lookup = working.get(
        "direct_action_value_label",
        pd.Series("", index=working.index),
    ).astype(str).str.lower()
    direct_action_applied = working.get(
        "direct_action_value_applied",
        pd.Series(0.0, index=working.index),
    ).fillna(0.0)
    direct_selected_value = working.get(
        "direct_action_value_selected",
        pd.Series(0.0, index=working.index),
    ).fillna(0.0)
    direct_value_gap = working.get(
        "direct_action_value_gap",
        pd.Series(0.0, index=working.index),
    ).fillna(0.0)
    direct_mode_mask = policy_decision_mode_lookup.eq("direct_action_value_v1") | (direct_action_applied > 0.5)
    direct_action_value_mode_share = _safe_mean(direct_mode_mask.astype(float))
    direct_action_value_label_match_share = (
        _safe_mean((model_action_lookup.loc[direct_mode_mask] == direct_action_label_lookup.loc[direct_mode_mask]).astype(float))
        if bool(direct_mode_mask.any())
        else 0.0
    )
    direct_action_value_selected_mean = (
        _safe_mean(direct_selected_value.loc[direct_mode_mask]) if bool(direct_mode_mask.any()) else 0.0
    )
    direct_action_value_gap_mean = (
        _safe_mean(direct_value_gap.loc[direct_mode_mask]) if bool(direct_mode_mask.any()) else 0.0
    )
    direct_action_value_low_margin_share = (
        _safe_mean((direct_value_gap.loc[direct_mode_mask] < 0.05).astype(float))
        if bool(direct_mode_mask.any())
        else 0.0
    )
    direct_action_order_translation_conflict_rate = (
        _safe_mean((model_action_lookup.loc[direct_mode_mask] != weight_change_lookup.loc[direct_mode_mask]).astype(float))
        if bool(direct_mode_mask.any())
        else 0.0
    )
    direct_funding_authorized = working.get(
        "direct_action_funding_release_authorized",
        pd.Series(False, index=working.index),
    ).astype(bool)
    direct_funding_protected = working.get(
        "direct_action_funding_protected",
        pd.Series(False, index=working.index),
    ).astype(bool)
    direct_release_advantage = working.get(
        "direct_action_release_advantage",
        pd.Series(0.0, index=working.index),
    ).fillna(0.0)
    direct_deploy_advantage = working.get(
        "direct_action_deploy_advantage",
        pd.Series(0.0, index=working.index),
    ).fillna(0.0)
    direct_deploy_signal = working.get(
        "direct_action_deploy_signal",
        pd.Series(False, index=working.index),
    ).astype(bool)
    direct_core_deploy_target = working.get(
        "direct_action_core_deploy_target",
        pd.Series(False, index=working.index),
    ).astype(bool)
    direct_add_authorized = working.get(
        "direct_action_add_authorized",
        pd.Series(False, index=working.index),
    ).astype(bool)
    direct_open_authorized = working.get(
        "direct_action_open_authorized",
        pd.Series(False, index=working.index),
    ).astype(bool)
    direct_deploy_authorized = working.get(
        "direct_action_deploy_authorized",
        pd.Series(False, index=working.index),
    ).astype(bool)
    direct_reallocation_source = working.get(
        "direct_action_reallocation_source",
        pd.Series(False, index=working.index),
    ).astype(bool)
    direct_pair_reallocation_source = working.get(
        "direct_action_pair_reallocation_source",
        pd.Series(False, index=working.index),
    ).astype(bool)
    direct_pair_cost_guard_pass = working.get(
        "direct_action_pair_cost_guard_pass",
        pd.Series(False, index=working.index),
    ).astype(bool)
    direct_pair_cost_guard_blocked = working.get(
        "direct_action_pair_cost_guard_blocked",
        pd.Series(False, index=working.index),
    ).astype(bool)
    direct_pair_opportunity_spread = working.get(
        "direct_action_pair_opportunity_spread",
        pd.Series(0.0, index=working.index),
    ).fillna(0.0)
    direct_pair_source_cost = working.get(
        "direct_action_pair_source_opportunity_cost",
        pd.Series(0.0, index=working.index),
    ).fillna(0.0)
    portfolio_receiver_target = working.get(
        "portfolio_daily_receiver_target",
        pd.Series(False, index=working.index),
    ).astype(bool)
    portfolio_receiver_candidate = working.get(
        "portfolio_daily_receiver_candidate",
        pd.Series(False, index=working.index),
    ).astype(bool)
    portfolio_receiver_exec_guarded = working.get(
        "portfolio_daily_receiver_exec_guarded",
        pd.Series(False, index=working.index),
    ).astype(bool)
    portfolio_receiver_semantic_no_headroom = working.get(
        "portfolio_daily_receiver_semantic_no_headroom",
        pd.Series(False, index=working.index),
    ).astype(bool)
    portfolio_receiver_open_breadth_candidate = working.get(
        "portfolio_daily_receiver_open_breadth_candidate",
        pd.Series(False, index=working.index),
    ).astype(bool)
    portfolio_receiver_add_headroom = working.get(
        "portfolio_daily_receiver_add_headroom",
        pd.Series(0.0, index=working.index),
    ).fillna(0.0)
    portfolio_receiver_min_add_delta = working.get(
        "portfolio_daily_receiver_min_add_delta",
        pd.Series(0.0, index=working.index),
    ).fillna(0.0)
    portfolio_source_candidate = working.get(
        "portfolio_daily_source_candidate",
        pd.Series(False, index=working.index),
    ).astype(bool)
    portfolio_source_target = working.get(
        "portfolio_daily_source_target",
        pd.Series(False, index=working.index),
    ).astype(bool)
    portfolio_source_exec_guard = working.get(
        "portfolio_daily_source_exec_guard",
        pd.Series(False, index=working.index),
    ).astype(bool)
    portfolio_source_exec_cap_guarded = working.get(
        "portfolio_daily_source_exec_cap_guarded",
        pd.Series(False, index=working.index),
    ).astype(bool)
    portfolio_source_realized_reduction_weight = working.get(
        "portfolio_daily_source_realized_reduction_weight",
        pd.Series(0.0, index=working.index),
    ).fillna(0.0)
    portfolio_cash_reserve_signal = working.get(
        "portfolio_daily_cash_reserve_signal",
        pd.Series(False, index=working.index),
    ).astype(bool)
    portfolio_receiver_score = working.get(
        "portfolio_daily_receiver_score",
        pd.Series(0.0, index=working.index),
    ).fillna(0.0)
    portfolio_source_gap = working.get(
        "portfolio_daily_source_gap",
        pd.Series(0.0, index=working.index),
    ).fillna(0.0)
    portfolio_source_score = working.get(
        "portfolio_daily_source_score",
        pd.Series(0.0, index=working.index),
    ).fillna(0.0)
    portfolio_source_forward_proxy_keep_risk = working.get(
        "portfolio_daily_source_forward_proxy_keep_risk",
        pd.Series(0.0, index=working.index),
    ).fillna(0.0)
    portfolio_source_release_conviction = working.get(
        "portfolio_daily_source_release_conviction",
        pd.Series(0.0, index=working.index),
    ).fillna(0.0)
    portfolio_cash_score = working.get(
        "portfolio_daily_cash_score",
        pd.Series(0.0, index=working.index),
    ).fillna(0.0)
    budget_origin_sell_mask = realized_sell_mask & (~model_authorized_sell_origin_mask)
    model_release_signal_sell_mask = realized_sell_mask & working["sell_execution_origin"].eq("model_release_signal")
    deploy_funding_rebalance_sell_mask = (
        realized_sell_mask & working["sell_execution_origin"].eq("deploy_funding_rebalance")
    )
    direct_action_pair_reallocation_sell_mask = (
        realized_sell_mask & working["sell_execution_origin"].eq("direct_action_pair_reallocation")
    )
    portfolio_daily_source_sell_mask = (
        realized_sell_mask & working["sell_execution_origin"].eq("portfolio_daily_ranking_source")
    )
    direct_action_intent_preserved_share = (
        _safe_mean((direct_action_label_lookup.loc[direct_mode_mask] == weight_change_lookup.loc[direct_mode_mask]).astype(float))
        if bool(direct_mode_mask.any())
        else 0.0
    )
    direct_action_funding_authorized_sell_share = (
        _safe_mean(direct_funding_authorized.loc[deploy_funding_rebalance_sell_mask].astype(float))
        if bool(deploy_funding_rebalance_sell_mask.any())
        else 0.0
    )
    direct_action_funding_protected_sell_share = (
        _safe_mean(direct_funding_protected.loc[deploy_funding_rebalance_sell_mask].astype(float))
        if bool(deploy_funding_rebalance_sell_mask.any())
        else 0.0
    )
    direct_action_release_advantage_mean = (
        _safe_mean(direct_release_advantage.loc[direct_mode_mask]) if bool(direct_mode_mask.any()) else 0.0
    )
    direct_action_deploy_advantage_mean = (
        _safe_mean(direct_deploy_advantage.loc[direct_mode_mask]) if bool(direct_mode_mask.any()) else 0.0
    )
    direct_action_deploy_signal_count = int(direct_deploy_signal.sum())
    direct_action_core_deploy_target_count = int(direct_core_deploy_target.sum())
    direct_action_core_deploy_target_realized_rate = (
        _safe_mean(weight_change_lookup.loc[direct_core_deploy_target].isin({"open", "add"}).astype(float))
        if bool(direct_core_deploy_target.any())
        else 0.0
    )
    direct_action_add_authorized_count = int(direct_add_authorized.sum())
    direct_action_add_authorized_realized_rate = (
        _safe_mean(weight_change_lookup.loc[direct_add_authorized].eq("add").astype(float))
        if bool(direct_add_authorized.any())
        else 0.0
    )
    authorized_add_no_weight_change_mask = direct_add_authorized & (
        (~weight_change_lookup.eq("add")) | (working["delta_weight"].fillna(0.0) <= 1.0e-8)
    )
    authorized_add_no_weight_change_count = int(authorized_add_no_weight_change_mask.sum())
    authorized_add_no_weight_change_share = (
        float(authorized_add_no_weight_change_count / direct_action_add_authorized_count)
        if direct_action_add_authorized_count
        else 0.0
    )
    direct_action_deploy_authorized_count = int(direct_deploy_authorized.sum())
    direct_action_deploy_authorized_realized_rate = (
        _safe_mean(weight_change_lookup.loc[direct_deploy_authorized].isin({"open", "add"}).astype(float))
        if bool(direct_deploy_authorized.any())
        else 0.0
    )
    direct_action_authorization_subset_violation_count = int(
        (
            (direct_add_authorized | direct_open_authorized)
            & ((~portfolio_receiver_target) | (~portfolio_receiver_candidate))
        ).sum()
    )
    direct_action_reallocation_source_count = int(direct_reallocation_source.sum())
    direct_action_pair_reallocation_source_count = int(direct_pair_reallocation_source.sum())
    direct_action_pair_cost_guard_pass_count = int(direct_pair_cost_guard_pass.sum())
    direct_action_pair_cost_guard_blocked_count = int(direct_pair_cost_guard_blocked.sum())
    direct_action_pair_cost_guard_observed_count = (
        direct_action_pair_cost_guard_pass_count + direct_action_pair_cost_guard_blocked_count
    )
    direct_action_pair_cost_guard_pass_rate = (
        float(direct_action_pair_cost_guard_pass_count / direct_action_pair_cost_guard_observed_count)
        if direct_action_pair_cost_guard_observed_count
        else 0.0
    )
    direct_action_pair_source_spread_mean = _safe_mean(
        direct_pair_opportunity_spread.loc[direct_pair_reallocation_source]
        if direct_pair_reallocation_source.any()
        else pd.Series(dtype=float)
    )
    direct_action_pair_source_cost_mean = _safe_mean(
        direct_pair_source_cost.loc[direct_pair_reallocation_source]
        if direct_pair_reallocation_source.any()
        else pd.Series(dtype=float)
    )
    budget_slot_reclaim_sell_mask = realized_sell_mask & working["sell_execution_origin"].eq("budget_slot_reclaim")
    sell_priority_guard_sell_mask = realized_sell_mask & working["sell_execution_origin"].eq("budget_sell_priority")
    turnover_trim_sell_mask = realized_sell_mask & working["sell_execution_origin"].eq("turnover_budget_trim")
    forced_zero_sell_mask = realized_sell_mask & working["sell_execution_origin"].eq("forced_zero")
    deploy_dropped_mask = deploy_intent_mask & (
        working["budget_dropped"]
        | working["forced_zero"]
        | (working["target_weight"].fillna(0.0) <= 1.0e-8)
    )
    deploy_intent_count = int(deploy_intent_mask.sum())
    add_intent_count = int(add_intent_mask.sum())
    deploy_intent_realized_count = int((deploy_intent_mask & deploy_realized_mask).sum())
    open_add_positive_weight_change_count = int((deploy_intent_mask & positive_weight_change_mask).sum())
    add_to_hold_conflict_count = int(add_to_hold_mask.sum())
    deploy_intent_dropped_count = int(deploy_dropped_mask.sum())
    deploy_intent_realized_rate = (
        float(deploy_intent_realized_count / deploy_intent_count) if deploy_intent_count else 0.0
    )
    open_add_positive_weight_change_rate = (
        float(open_add_positive_weight_change_count / deploy_intent_count) if deploy_intent_count else 0.0
    )
    add_to_hold_conflict_share = (
        float(add_to_hold_conflict_count / add_intent_count) if add_intent_count else 0.0
    )
    deploy_intent_hold_conflict_share = (
        float(deploy_hold_mask.sum() / deploy_intent_count) if deploy_intent_count else 0.0
    )
    deploy_intent_dropped_share = (
        float(deploy_intent_dropped_count / deploy_intent_count) if deploy_intent_count else 0.0
    )
    deploy_intent_unrealized_count = max(0, deploy_intent_count - open_add_positive_weight_change_count)
    deploy_intent_unrealized_share = (
        float(deploy_intent_unrealized_count / deploy_intent_count) if deploy_intent_count else 0.0
    )
    avg_deploy_intent_candidate_count = _safe_mean(
        day_merge.get("deploy_intent_candidate_count", pd.Series(0.0, index=day_merge.index)).fillna(0.0)
    )
    avg_deploy_intent_candidate_realized_rate = _safe_mean(
        day_merge.get("deploy_intent_candidate_realized_rate", pd.Series(0.0, index=day_merge.index)).fillna(0.0)
    )
    avg_deploy_intent_candidate_budget_drop_share = _safe_mean(
        day_merge.get("deploy_intent_candidate_budget_drop_share", pd.Series(0.0, index=day_merge.index)).fillna(0.0)
    )
    sell_intent_count = int(sell_intent_mask.sum())
    realized_sell_count = int(realized_sell_mask.sum())
    sell_intent_realized_count = int(sell_intent_realized_mask.sum())
    sell_intent_hold_conflict_count = int(sell_intent_hold_mask.sum())
    sell_intent_suppressed_count = int(sell_intent_suppressed_mask.sum())
    budget_origin_sell_count = int(budget_origin_sell_mask.sum())
    model_release_signal_sell_count = int(model_release_signal_sell_mask.sum())
    deploy_funding_rebalance_sell_count = int(deploy_funding_rebalance_sell_mask.sum())
    direct_action_pair_reallocation_sell_count = int(direct_action_pair_reallocation_sell_mask.sum())
    portfolio_daily_source_sell_count = int(portfolio_daily_source_sell_mask.sum())
    budget_slot_reclaim_sell_count = int(budget_slot_reclaim_sell_mask.sum())
    sell_priority_guard_sell_count = int(sell_priority_guard_sell_mask.sum())
    turnover_trim_sell_count = int(turnover_trim_sell_mask.sum())
    forced_zero_sell_count = int(forced_zero_sell_mask.sum())
    sell_intent_realized_rate = float(sell_intent_realized_count / sell_intent_count) if sell_intent_count else 0.0
    sell_intent_hold_conflict_share = float(sell_intent_hold_conflict_count / sell_intent_count) if sell_intent_count else 0.0
    sell_intent_suppressed_share = float(sell_intent_suppressed_count / sell_intent_count) if sell_intent_count else 0.0
    budget_origin_sell_share = float(budget_origin_sell_count / realized_sell_count) if realized_sell_count else 0.0
    model_release_signal_sell_share = (
        float(model_release_signal_sell_count / realized_sell_count) if realized_sell_count else 0.0
    )
    deploy_funding_rebalance_sell_share = (
        float(deploy_funding_rebalance_sell_count / realized_sell_count) if realized_sell_count else 0.0
    )
    direct_action_pair_reallocation_sell_share = (
        float(direct_action_pair_reallocation_sell_count / realized_sell_count) if realized_sell_count else 0.0
    )
    portfolio_daily_source_sell_share = (
        float(portfolio_daily_source_sell_count / realized_sell_count) if realized_sell_count else 0.0
    )
    budget_slot_reclaim_sell_share = (
        float(budget_slot_reclaim_sell_count / realized_sell_count) if realized_sell_count else 0.0
    )
    sell_priority_guard_sell_share = (
        float(sell_priority_guard_sell_count / realized_sell_count) if realized_sell_count else 0.0
    )
    turnover_trim_sell_share = float(turnover_trim_sell_count / realized_sell_count) if realized_sell_count else 0.0
    forced_zero_sell_share = float(forced_zero_sell_count / realized_sell_count) if realized_sell_count else 0.0
    model_origin_sell_forward_excess_5d = _safe_mean(
        working.loc[sell_intent_realized_mask, "forward_excess_5d"]
        if sell_intent_realized_mask.any()
        else pd.Series(dtype=float)
    )
    model_authorized_sell_mask = realized_sell_mask & model_authorized_sell_origin_mask
    model_authorized_sell_forward_excess_5d = _safe_mean(
        working.loc[model_authorized_sell_mask, "forward_excess_5d"]
        if model_authorized_sell_mask.any()
        else pd.Series(dtype=float)
    )
    model_release_signal_forward_excess_5d = _safe_mean(
        working.loc[model_release_signal_sell_mask, "forward_excess_5d"]
        if model_release_signal_sell_mask.any()
        else pd.Series(dtype=float)
    )
    deploy_funding_rebalance_forward_excess_5d = _safe_mean(
        working.loc[deploy_funding_rebalance_sell_mask, "forward_excess_5d"]
        if deploy_funding_rebalance_sell_mask.any()
        else pd.Series(dtype=float)
    )
    budget_origin_sell_forward_excess_5d = _safe_mean(
        working.loc[budget_origin_sell_mask, "forward_excess_5d"]
        if budget_origin_sell_mask.any()
        else pd.Series(dtype=float)
    )
    direct_action_pair_source_forward_excess_5d = _safe_mean(
        working.loc[direct_pair_reallocation_source, "forward_excess_5d"]
        if direct_pair_reallocation_source.any()
        else pd.Series(dtype=float)
    )
    direct_action_core_target_forward_excess_5d = _safe_mean(
        working.loc[direct_core_deploy_target, "forward_excess_5d"]
        if direct_core_deploy_target.any()
        else pd.Series(dtype=float)
    )
    direct_action_core_minus_pair_forward_excess_5d = (
        direct_action_core_target_forward_excess_5d - direct_action_pair_source_forward_excess_5d
        if direct_core_deploy_target.any() and direct_pair_reallocation_source.any()
        else 0.0
    )
    portfolio_daily_receiver_candidate_count = int(portfolio_receiver_candidate.sum())
    portfolio_daily_receiver_target_count = int(portfolio_receiver_target.sum())
    portfolio_daily_receiver_exec_guard_count = int(portfolio_receiver_exec_guarded.sum())
    portfolio_daily_receiver_semantic_no_headroom_count = int(portfolio_receiver_semantic_no_headroom.sum())
    portfolio_daily_receiver_open_breadth_candidate_count = int(portfolio_receiver_open_breadth_candidate.sum())
    portfolio_daily_source_candidate_count = int(portfolio_source_candidate.sum())
    portfolio_daily_source_target_count = int(portfolio_source_target.sum())
    portfolio_daily_source_realized_sell_rate = (
        _safe_mean(weight_change_lookup.loc[portfolio_source_target].isin({"reduce", "exit"}).astype(float))
        if portfolio_source_target.any()
        else 0.0
    )
    portfolio_daily_source_target_not_sold_mask = portfolio_source_target & (
        ~weight_change_lookup.isin({"reduce", "exit"})
    )
    portfolio_daily_source_target_not_sold_count = int(portfolio_daily_source_target_not_sold_mask.sum())
    portfolio_daily_source_target_not_sold_share = (
        float(portfolio_daily_source_target_not_sold_count / portfolio_daily_source_target_count)
        if portfolio_daily_source_target_count
        else 0.0
    )
    portfolio_daily_source_exec_guard_count = int((portfolio_source_target & portfolio_source_exec_guard).sum())
    portfolio_daily_source_exec_cap_guard_count = int(
        (portfolio_source_target & portfolio_source_exec_cap_guarded).sum()
    )
    portfolio_daily_source_realized_reduction_weight = (
        float(portfolio_source_realized_reduction_weight.loc[portfolio_source_target].sum())
        if portfolio_source_target.any()
        else 0.0
    )
    portfolio_daily_receiver_realized_deploy_count = int(
        (portfolio_receiver_target & weight_change_lookup.isin({"open", "add"})).sum()
    )
    portfolio_daily_receiver_unrealized_deploy_count = int(
        (portfolio_receiver_target & (~weight_change_lookup.isin({"open", "add"}))).sum()
    )
    portfolio_daily_receiver_realized_deploy_rate = (
        float(portfolio_daily_receiver_realized_deploy_count / portfolio_daily_receiver_target_count)
        if portfolio_daily_receiver_target_count
        else 0.0
    )
    portfolio_daily_receiver_unrealized_deploy_share = (
        float(portfolio_daily_receiver_unrealized_deploy_count / portfolio_daily_receiver_target_count)
        if portfolio_daily_receiver_target_count
        else 0.0
    )
    portfolio_daily_effective_capital_transfer_count = min(
        int((portfolio_source_target & weight_change_lookup.isin({"reduce", "exit"})).sum()),
        portfolio_daily_receiver_realized_deploy_count,
    )
    portfolio_daily_cash_score_mean = _safe_mean(portfolio_cash_score)
    portfolio_daily_cash_reserve_rate = _safe_mean(portfolio_cash_reserve_signal.astype(float))
    portfolio_daily_receiver_score_mean = (
        _safe_mean(portfolio_receiver_score.loc[portfolio_receiver_target])
        if portfolio_receiver_target.any()
        else 0.0
    )
    portfolio_daily_source_score_mean = (
        _safe_mean(portfolio_source_score.loc[portfolio_source_target])
        if portfolio_source_target.any()
        else 0.0
    )
    portfolio_daily_source_gap_mean = (
        _safe_mean(portfolio_source_gap.loc[portfolio_source_target])
        if portfolio_source_target.any()
        else 0.0
    )
    portfolio_daily_source_forward_proxy_keep_risk_mean = (
        _safe_mean(portfolio_source_forward_proxy_keep_risk.loc[portfolio_source_target])
        if portfolio_source_target.any()
        else 0.0
    )
    portfolio_daily_source_release_conviction_mean = (
        _safe_mean(portfolio_source_release_conviction.loc[portfolio_source_target])
        if portfolio_source_target.any()
        else 0.0
    )
    portfolio_daily_receiver_forward_excess_5d = (
        _safe_mean(working.loc[portfolio_receiver_target, "forward_excess_5d"])
        if portfolio_receiver_target.any()
        else 0.0
    )
    portfolio_daily_source_forward_excess_5d = (
        _safe_mean(working.loc[portfolio_source_target, "forward_excess_5d"])
        if portfolio_source_target.any()
        else 0.0
    )
    portfolio_source_forward_values = working.loc[portfolio_source_target, "forward_excess_5d"].dropna()
    portfolio_daily_source_positive_forward_sell_share = (
        float((portfolio_source_forward_values > 0.0).mean())
        if len(portfolio_source_forward_values)
        else 0.0
    )
    portfolio_daily_source_strong_positive_forward_sell_count = (
        int((portfolio_source_forward_values > 0.055).sum())
        if len(portfolio_source_forward_values)
        else 0
    )
    portfolio_daily_source_max_forward_excess_5d = (
        float(portfolio_source_forward_values.max()) if len(portfolio_source_forward_values) else 0.0
    )
    portfolio_daily_source_p75_forward_excess_5d = (
        float(portfolio_source_forward_values.quantile(0.75)) if len(portfolio_source_forward_values) else 0.0
    )
    portfolio_daily_receiver_minus_source_forward_excess_5d = (
        portfolio_daily_receiver_forward_excess_5d - portfolio_daily_source_forward_excess_5d
        if portfolio_receiver_target.any() and portfolio_source_target.any()
        else 0.0
    )
    avg_gross_exposure_target, portfolio_daily_exposure_utilization = _compute_exposure_utilization_from_turnover(
        day_merge
    )
    allocation_closure = summarize_allocation_closure_from_turnover(allocation_closure_frame)
    high_cash_sell_mask = working["high_cash_day"].astype(bool) & realized_sell_mask
    high_cash_sell_action_count = int(high_cash_sell_mask.sum())
    high_cash_budget_origin_sell_share = (
        float((budget_origin_sell_mask & working["high_cash_day"].astype(bool)).sum() / high_cash_sell_action_count)
        if high_cash_sell_action_count
        else 0.0
    )
    clipped_intent_series = working.get("clipped_intent_risk", pd.Series(0.0, index=working.index)).fillna(0.0)
    avg_clipped_intent_risk = _safe_mean(clipped_intent_series)
    if bool(working["is_order_translation_conflict"].any()) and bool((~working["is_order_translation_conflict"]).any()):
        clipped_intent_risk_conflict_gap = float(
            clipped_intent_series.loc[working["is_order_translation_conflict"]].mean()
            - clipped_intent_series.loc[~working["is_order_translation_conflict"]].mean()
        )
    else:
        clipped_intent_risk_conflict_gap = 0.0
    model_release_signal_keep_support = _safe_mean(
        protected_hold_support.loc[model_release_signal_sell_mask]
        if model_release_signal_sell_mask.any()
        else pd.Series(dtype=float)
    )
    model_release_signal_disciplined_funding_need = _safe_mean(
        disciplined_funding_need.loc[model_release_signal_sell_mask]
        if model_release_signal_sell_mask.any()
        else pd.Series(dtype=float)
    )
    model_release_signal_release_support = _safe_mean(
        funding_release_support.loc[model_release_signal_sell_mask]
        if model_release_signal_sell_mask.any()
        else pd.Series(dtype=float)
    )
    deploy_funding_disciplined_funding_need = _safe_mean(
        disciplined_funding_need.loc[deploy_funding_rebalance_sell_mask]
        if deploy_funding_rebalance_sell_mask.any()
        else pd.Series(dtype=float)
    )
    deploy_funding_keep_support = _safe_mean(
        protected_hold_support.loc[deploy_funding_rebalance_sell_mask]
        if deploy_funding_rebalance_sell_mask.any()
        else pd.Series(dtype=float)
    )
    deploy_funding_release_support = _safe_mean(
        funding_release_support.loc[deploy_funding_rebalance_sell_mask]
        if deploy_funding_rebalance_sell_mask.any()
        else pd.Series(dtype=float)
    )
    model_release_against_protected_hold_share = _safe_mean(
        (
            protected_hold_support.loc[model_release_signal_sell_mask]
            > funding_release_support.loc[model_release_signal_sell_mask] + support_margin
        ).astype(float)
        if model_release_signal_sell_mask.any()
        else pd.Series(dtype=float)
    )
    model_release_release_consistent_share = _safe_mean(
        (
            funding_release_support.loc[model_release_signal_sell_mask]
            > protected_hold_support.loc[model_release_signal_sell_mask] + support_margin
        ).astype(float)
        if model_release_signal_sell_mask.any()
        else pd.Series(dtype=float)
    )
    deploy_funding_against_protected_hold_share = _safe_mean(
        (
            protected_hold_support.loc[deploy_funding_rebalance_sell_mask]
            > funding_release_support.loc[deploy_funding_rebalance_sell_mask] + support_margin
        ).astype(float)
        if deploy_funding_rebalance_sell_mask.any()
        else pd.Series(dtype=float)
    )
    deploy_funding_release_consistent_share = _safe_mean(
        (
            funding_release_support.loc[deploy_funding_rebalance_sell_mask]
            > protected_hold_support.loc[deploy_funding_rebalance_sell_mask] + support_margin
        ).astype(float)
        if deploy_funding_rebalance_sell_mask.any()
        else pd.Series(dtype=float)
    )
    if semantic_conflict_rate >= 0.05:
        diagnoses.append("执行层仍在非小概率地重写模型动作语义，策略学习闭环还不干净。")
    if order_translation_conflict_rate >= 0.05:
        diagnoses.append("真实权重变化仍与模型动作意图存在偏离，预算/持仓约束还需要继续和动作头分层校准。")
    if direct_action_value_mode_share >= 0.50 and direct_action_order_translation_conflict_rate >= 0.08:
        diagnoses.append("直接动作值已进入决策主路径，但订单/预算翻译仍在改写 direct action intent，需要优先做 direct-action-preserving calibration。")
    if direct_action_value_mode_share >= 0.50 and direct_action_value_low_margin_share >= 0.60:
        diagnoses.append("直接动作值边际偏低，低置信动作应更多保留为 hold/cash，而不是被预算层强行转成交易。")
    if deploy_funding_rebalance_sell_count >= 5 and direct_action_funding_authorized_sell_share < 0.50:
        diagnoses.append("deploy funding sell 中直接动作释放授权不足，说明 funding 责任仍被组合层吸走。")
    if micro_rebalance_conflict_rate >= 0.02 or small_delta_order_translation_conflict_rate >= 0.02:
        diagnoses.append("微幅再平衡仍会造成订单层反向变化，需继续隔离 hold/add/reduce 的生命周期语义。")
    if budget_clipped_order_translation_conflict_rate > unclipped_order_translation_conflict_rate + 0.03:
        diagnoses.append("预算/换手约束会显著放大订单层动作偏离，预算头与动作头仍未真正解耦。")
    if deploy_intent_count >= 5 and deploy_intent_realized_rate < 0.55:
        diagnoses.append("模型给出的 open/add 部署意图未被足量转化为真实加仓，部署可执行性仍是主瓶颈。")
    if authorized_add_no_weight_change_share >= 0.05 or direct_action_authorization_subset_violation_count > 0:
        diagnoses.append("receiver 授权仍存在语义旁路：部分 add/open 授权没有落到可执行 receiver 子集或没有形成真实权重变化。")
    if portfolio_daily_receiver_semantic_no_headroom_count >= 3:
        diagnoses.append("无 headroom 的 held add 已被语义层降级，需要继续把 receiver 容量作为模型评分信号，而不是交给后置订单翻译层处理。")
    if deploy_intent_count >= 5 and deploy_intent_unrealized_share >= 0.25:
        diagnoses.append("部署意图仍有较高比例未兑现，需要继续提高 cash/receiver/source 的组合日资金闭环能力。")
    if add_intent_count >= 3 and add_to_hold_conflict_share >= 0.20:
        diagnoses.append("add -> hold 冲突占比偏高，持仓加仓意图在预算/换手翻译层被过度钝化。")
    if direct_action_deploy_authorized_count >= 5 and direct_action_deploy_authorized_realized_rate < 0.55:
        diagnoses.append("direct action 授权部署仍未被稳定兑现，说明缺口集中在预算再分配和订单翻译闭环。")
    if (
        (deploy_intent_count >= 5 and deploy_intent_dropped_share >= 0.20)
        or avg_deploy_intent_candidate_budget_drop_share >= 0.20
    ):
        diagnoses.append("open/add 意图被预算丢弃或归零的比例偏高，需要优先检查候选预算与部署地板。")
    if realized_sell_count >= 5 and budget_origin_sell_share >= 0.30:
        diagnoses.append("真实卖出中相当一部分并非来自模型 reduce/exit 主意图，而是由预算/翻译层被动创造，卖出责任边界仍不干净。")
    if sell_intent_count >= 5 and sell_intent_suppressed_share >= 0.20:
        diagnoses.append("模型自身给出的 reduce/exit 意图仍有较高比例被翻译层压回 hold，卖出学习与组合约束还没有真正分责。")
    if high_cash_sell_action_count >= 3 and high_cash_budget_origin_sell_share >= 0.50:
        diagnoses.append("高现金日的卖出更像预算挤压而不是主动卖出决策，当前 cash timing 仍偏被动。")
    if high_cash_up_market_share > high_cash_down_market_share + 0.05:
        diagnoses.append("高现金日更多出现在次日上涨前，当前现金部署仍带有顺周期保守偏差。")

    if (
        deploy_funding_rebalance_sell_count >= 5
        and deploy_funding_rebalance_sell_share >= 0.45
        and (
            deploy_funding_rebalance_forward_excess_5d > 0.0
            or deploy_funding_against_protected_hold_share >= 0.35
            or deploy_funding_release_consistent_share <= 0.35
        )
    ):
        diagnoses.append("为了给新部署腾挪资金，系统仍在过多卖出原本更值得保留的旧仓，held-side funding/release 仲裁还没有学稳。")
    if (
        model_release_signal_sell_count >= 5
        and (
            model_release_signal_forward_excess_5d > 0.0
            or model_release_against_protected_hold_share >= 0.35
            or model_release_release_consistent_share <= 0.35
        )
    ):
        diagnoses.append("模型给出的 release 信号仍不够选择性，被释放的持仓里仍有相当比例属于应当继续保留的强持仓。")

    if (
        direct_action_pair_reallocation_source_count >= 5
        and direct_action_core_minus_pair_forward_excess_5d <= 0.0
    ):
        diagnoses.append("direct-action pair reallocation 的资金来源没有稳定弱于核心部署目标，说明组合内相对机会成本守门仍需收紧。")
    if (
        direct_action_pair_reallocation_source_count >= 5
        and direct_action_pair_source_cost_mean >= 0.78
    ):
        diagnoses.append("direct-action pair reallocation 正在释放高持有价值来源，资金来源成本偏高会吞掉核心部署收益。")

    if (
        portfolio_daily_receiver_target_count >= 3
        and portfolio_daily_source_target_count >= 3
        and portfolio_daily_receiver_minus_source_forward_excess_5d <= 0.0
    ):
        diagnoses.append("portfolio daily ranking has not separated capital receivers from funding sources; the listwise allocation target still needs tighter relative credit assignment.")
    if portfolio_daily_source_target_count >= 3 and portfolio_daily_source_forward_excess_5d > 0.006:
        diagnoses.append("portfolio daily ranking is releasing sources that still have positive forward excess return, suggesting sell/opportunity-cost attribution remains too weak.")
    if (
        portfolio_daily_source_target_count >= 3
        and (
            portfolio_daily_source_positive_forward_sell_share > 0.45
            or portfolio_daily_source_strong_positive_forward_sell_count >= 1
        )
    ):
        diagnoses.append("source 选择的均值改善不足以说明卖得对：正 forward source 占比或强势误卖仍偏高，需把分布尾部纳入 source opportunity cost。")
    if portfolio_daily_source_target_count >= 3 and portfolio_daily_source_forward_proxy_keep_risk_mean > 0.280:
        diagnoses.append("source 目标的 forward proxy keep risk 偏高，说明卖出候选仍混入了继续持有价值较强的标的，需要继续抬高 source opportunity cost 或收紧 release label。")
    if portfolio_daily_source_target_count >= 3 and portfolio_daily_source_release_conviction_mean < 0.300:
        diagnoses.append("source release conviction 均值偏低，说明 source 目标虽被执行为卖出，但释放资金的综合证据仍不足，需要继续把 opportunity cost、forward proxy keep risk 和 release label 对齐。")
    if portfolio_daily_receiver_target_count >= 5 and portfolio_daily_source_target_count == 0:
        diagnoses.append("portfolio daily ranking selects capital receivers but finds no explicit funding source, so cash/source coordination remains incomplete.")
    if portfolio_daily_source_target_count >= 5 and portfolio_daily_source_target_not_sold_share > 0.65:
        diagnoses.append("portfolio daily ranking selects funding sources, but too many source targets are not translated into real reduce/exit actions; source execution coupling remains the primary bottleneck.")
    if (
        portfolio_daily_receiver_target_count >= 5
        and portfolio_daily_source_target_count >= 5
        and portfolio_daily_effective_capital_transfer_count < min(
            portfolio_daily_receiver_target_count,
            portfolio_daily_source_target_count,
        )
        * 0.30
    ):
        diagnoses.append("portfolio daily receiver/source matching is sparse after execution, so the learned ranking is not yet becoming an effective capital transfer graph.")

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
    sell_execution_origin_counts = {
        str(key): int(value)
        for key, value in working.loc[realized_sell_mask, "sell_execution_origin"].astype(str).value_counts().sort_index().items()
    }
    sell_suppression_origin_counts = {
        str(key): int(value)
        for key, value in working.loc[sell_intent_suppressed_mask, "sell_suppression_origin"].astype(str).value_counts().sort_index().items()
    }
    release_translation_deploy_health = _release_translation_deploy_health(
        deploy_intent_action_count=float(deploy_intent_count),
        deploy_intent_realized_rate=deploy_intent_realized_rate,
        order_translation_conflict_rate=order_translation_conflict_rate,
        add_to_hold_conflict_share=add_to_hold_conflict_share,
        sell_intent_suppressed_share=sell_intent_suppressed_share,
        budget_origin_sell_share=budget_origin_sell_share,
        deploy_funding_rebalance_sell_count=float(deploy_funding_rebalance_sell_count),
        deploy_funding_rebalance_sell_share=deploy_funding_rebalance_sell_share,
        deploy_funding_rebalance_forward_excess_5d=deploy_funding_rebalance_forward_excess_5d,
        deploy_funding_against_protected_hold_share=deploy_funding_against_protected_hold_share,
        deploy_funding_release_consistent_share=deploy_funding_release_consistent_share,
        model_release_signal_sell_count=float(model_release_signal_sell_count),
        model_release_signal_forward_excess_5d=model_release_signal_forward_excess_5d,
        model_release_against_protected_hold_share=model_release_against_protected_hold_share,
        model_release_release_consistent_share=model_release_release_consistent_share,
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
        "avg_budget_reclaimable_held_count": _safe_mean(day_merge.get("budget_reclaimable_held_count", pd.Series(0.0, index=day_merge.index)).fillna(0.0)),
        "avg_budget_released_held_count": _safe_mean(day_merge.get("budget_released_held_count", pd.Series(0.0, index=day_merge.index)).fillna(0.0)),
        "avg_model_release_signal_count": _safe_mean(day_merge.get("model_release_signal_count", pd.Series(0.0, index=day_merge.index)).fillna(0.0)),
        "avg_deploy_funding_rebalance_signal_count": _safe_mean(day_merge.get("deploy_funding_rebalance_signal_count", pd.Series(0.0, index=day_merge.index)).fillna(0.0)),
        "avg_sell_authorized_held_count": _safe_mean(day_merge.get("sell_authorized_held_count", pd.Series(0.0, index=day_merge.index)).fillna(0.0)),
        "avg_sell_source_floor_guard_count": _safe_mean(day_merge.get("sell_source_floor_guard_count", pd.Series(0.0, index=day_merge.index)).fillna(0.0)),
        "avg_budget_split_bound_guard_count": _safe_mean(day_merge.get("budget_split_bound_guard_count", pd.Series(0.0, index=day_merge.index)).fillna(0.0)),
        "avg_budget_sell_priority_guard_count": _safe_mean(day_merge.get("budget_sell_priority_guard_count", pd.Series(0.0, index=day_merge.index)).fillna(0.0)),
        "avg_sell_rank_score": avg_sell_rank_score,
        "avg_lifecycle_sell_gate": avg_lifecycle_sell_gate,
        "avg_alpha_opportunity_value": avg_alpha_opportunity_value,
        "avg_hold_continuation_value": avg_hold_continuation_value,
        "avg_sell_release_value": avg_sell_release_value,
        "avg_cash_defense_value": avg_cash_defense_value,
        "avg_multi_horizon_forward_value": avg_multi_horizon_forward_value,
        "avg_multi_horizon_forward_risk": avg_multi_horizon_forward_risk,
        "avg_multi_horizon_path_value": avg_multi_horizon_path_value,
        "avg_open_action_value": avg_open_action_value,
        "avg_add_action_value": avg_add_action_value,
        "avg_hold_action_value": avg_hold_action_value,
        "avg_reduce_action_value": avg_reduce_action_value,
        "avg_exit_action_value": avg_exit_action_value,
        "avg_action_value_consistency_target": avg_action_value_consistency_target,
        "action_value_conflict_share": action_value_conflict_share,
        "action_value_selected_gap": _safe_mean((best_action_value - chosen_action_value).where(action_value_conflict_mask, 0.0)),
        "held_keep_release_value_gap": _safe_mean((keep_action_value - release_action_value).where(held_value_mask, 0.0)),
        "sell_against_keep_value_share": sell_against_keep_value_share,
        "keep_against_release_value_share": keep_against_release_value_share,
        "open_low_action_value_share": open_low_action_value_share,
        "action_value_consistency_score": action_value_consistency_score,
        "direct_action_value_mode_share": direct_action_value_mode_share,
        "direct_action_value_label_match_share": direct_action_value_label_match_share,
        "direct_action_value_selected_mean": direct_action_value_selected_mean,
        "direct_action_value_gap_mean": direct_action_value_gap_mean,
        "direct_action_value_low_margin_share": direct_action_value_low_margin_share,
        "direct_action_order_translation_conflict_rate": direct_action_order_translation_conflict_rate,
        "direct_action_intent_preserved_share": direct_action_intent_preserved_share,
        "allocation_intent_v2_mode_share": allocation_intent_v2_mode_share,
        "intent_translation_conflict_rate": intent_translation_conflict_rate,
        "lifecycle_hint_conflict_rate": lifecycle_hint_conflict_rate,
        "direct_action_funding_authorized_sell_share": direct_action_funding_authorized_sell_share,
        "direct_action_funding_protected_sell_share": direct_action_funding_protected_sell_share,
        "direct_action_release_advantage_mean": direct_action_release_advantage_mean,
        "direct_action_deploy_advantage_mean": direct_action_deploy_advantage_mean,
        "direct_action_deploy_signal_count": int(direct_action_deploy_signal_count),
        "direct_action_core_deploy_target_count": int(direct_action_core_deploy_target_count),
        "direct_action_core_deploy_target_realized_rate": direct_action_core_deploy_target_realized_rate,
        "direct_action_add_authorized_count": int(direct_action_add_authorized_count),
        "direct_action_add_authorized_realized_rate": direct_action_add_authorized_realized_rate,
        "authorized_add_no_weight_change_count": int(authorized_add_no_weight_change_count),
        "authorized_add_no_weight_change_share": authorized_add_no_weight_change_share,
        "direct_action_deploy_authorized_count": int(direct_action_deploy_authorized_count),
        "direct_action_deploy_authorized_realized_rate": direct_action_deploy_authorized_realized_rate,
        "direct_action_authorization_subset_violation_count": int(
            direct_action_authorization_subset_violation_count
        ),
        "direct_action_reallocation_source_count": int(direct_action_reallocation_source_count),
        "direct_action_pair_reallocation_source_count": int(direct_action_pair_reallocation_source_count),
        "direct_action_pair_cost_guard_pass_count": int(direct_action_pair_cost_guard_pass_count),
        "direct_action_pair_cost_guard_blocked_count": int(direct_action_pair_cost_guard_blocked_count),
        "direct_action_pair_cost_guard_pass_rate": direct_action_pair_cost_guard_pass_rate,
        "direct_action_pair_source_spread_mean": direct_action_pair_source_spread_mean,
        "direct_action_pair_source_cost_mean": direct_action_pair_source_cost_mean,
        "direct_action_pair_source_forward_excess_5d": direct_action_pair_source_forward_excess_5d,
        "direct_action_core_target_forward_excess_5d": direct_action_core_target_forward_excess_5d,
        "direct_action_core_minus_pair_forward_excess_5d": direct_action_core_minus_pair_forward_excess_5d,
        "portfolio_daily_receiver_candidate_count": int(portfolio_daily_receiver_candidate_count),
        "portfolio_daily_receiver_exec_guard_count": int(portfolio_daily_receiver_exec_guard_count),
        "portfolio_daily_receiver_semantic_no_headroom_count": int(
            portfolio_daily_receiver_semantic_no_headroom_count
        ),
        "portfolio_daily_receiver_open_breadth_candidate_count": int(
            portfolio_daily_receiver_open_breadth_candidate_count
        ),
        "portfolio_daily_receiver_add_headroom_mean": (
            _safe_mean(portfolio_receiver_add_headroom.loc[portfolio_receiver_exec_guarded])
            if portfolio_receiver_exec_guarded.any()
            else 0.0
        ),
        "portfolio_daily_receiver_min_add_delta_mean": (
            _safe_mean(portfolio_receiver_min_add_delta.loc[portfolio_receiver_exec_guarded])
            if portfolio_receiver_exec_guarded.any()
            else 0.0
        ),
        "portfolio_daily_receiver_target_count": int(portfolio_daily_receiver_target_count),
        "portfolio_daily_receiver_realized_deploy_rate": portfolio_daily_receiver_realized_deploy_rate,
        "portfolio_daily_receiver_unrealized_deploy_count": int(portfolio_daily_receiver_unrealized_deploy_count),
        "portfolio_daily_receiver_unrealized_deploy_share": portfolio_daily_receiver_unrealized_deploy_share,
        "portfolio_daily_source_candidate_count": int(portfolio_daily_source_candidate_count),
        "portfolio_daily_source_target_count": int(portfolio_daily_source_target_count),
        "portfolio_daily_source_sell_count": int(portfolio_daily_source_sell_count),
        "portfolio_daily_source_sell_share": portfolio_daily_source_sell_share,
        "portfolio_daily_source_realized_sell_rate": portfolio_daily_source_realized_sell_rate,
        "portfolio_daily_source_exec_guard_count": int(portfolio_daily_source_exec_guard_count),
        "portfolio_daily_source_exec_cap_guard_count": int(portfolio_daily_source_exec_cap_guard_count),
        "portfolio_daily_source_target_not_sold_count": int(portfolio_daily_source_target_not_sold_count),
        "portfolio_daily_source_target_not_sold_share": portfolio_daily_source_target_not_sold_share,
        "portfolio_daily_source_realized_reduction_weight": portfolio_daily_source_realized_reduction_weight,
        "portfolio_daily_receiver_realized_deploy_count": int(portfolio_daily_receiver_realized_deploy_count),
        "portfolio_daily_effective_capital_transfer_count": int(portfolio_daily_effective_capital_transfer_count),
        "portfolio_daily_cash_score_mean": portfolio_daily_cash_score_mean,
        "portfolio_daily_cash_reserve_rate": portfolio_daily_cash_reserve_rate,
        "portfolio_daily_budget_closed": float(bool(allocation_closure.get("budget_closed", False))),
        "portfolio_daily_deployment_required": float(bool(allocation_closure.get("deployment_required", False))),
        "portfolio_daily_risk_reduction_required": float(
            bool(allocation_closure.get("risk_reduction_required", False))
        ),
        "portfolio_daily_receiver_activity_required": float(
            bool(allocation_closure.get("receiver_activity_required", False))
        ),
        "portfolio_daily_receiver_score_mean": portfolio_daily_receiver_score_mean,
        "portfolio_daily_source_score_mean": portfolio_daily_source_score_mean,
        "portfolio_daily_source_gap_mean": portfolio_daily_source_gap_mean,
        "portfolio_daily_source_forward_proxy_keep_risk_mean": (
            portfolio_daily_source_forward_proxy_keep_risk_mean
        ),
        "portfolio_daily_source_release_conviction_mean": (
            portfolio_daily_source_release_conviction_mean
        ),
        "portfolio_daily_receiver_forward_excess_5d": portfolio_daily_receiver_forward_excess_5d,
        "portfolio_daily_source_forward_excess_5d": portfolio_daily_source_forward_excess_5d,
        "portfolio_daily_source_positive_forward_sell_share": portfolio_daily_source_positive_forward_sell_share,
        "portfolio_daily_source_strong_positive_forward_sell_count": int(
            portfolio_daily_source_strong_positive_forward_sell_count
        ),
        "portfolio_daily_source_max_forward_excess_5d": portfolio_daily_source_max_forward_excess_5d,
        "portfolio_daily_source_p75_forward_excess_5d": portfolio_daily_source_p75_forward_excess_5d,
        "portfolio_daily_receiver_minus_source_forward_excess_5d": portfolio_daily_receiver_minus_source_forward_excess_5d,
        "avg_gross_exposure_target": avg_gross_exposure_target,
        "portfolio_daily_exposure_utilization": portfolio_daily_exposure_utilization,
        "portfolio_daily_actual_cash_weight_mean": float(
            allocation_closure.get("actual_cash_weight_mean", 0.0) or 0.0
        ),
        "portfolio_daily_actual_gross_exposure_mean": float(
            allocation_closure.get("actual_gross_exposure_mean", 0.0) or 0.0
        ),
        "portfolio_daily_deployable_idle_cash_mean": float(
            allocation_closure.get("deployable_idle_cash_mean", 0.0) or 0.0
        ),
        "portfolio_daily_cash_semantics_mismatch": float(
            bool(allocation_closure.get("cash_semantics_mismatch", False))
        ),
        "portfolio_daily_receiver_candidate_without_target_day_share": float(
            allocation_closure.get("receiver_candidate_without_target_day_share", 0.0) or 0.0
        ),
        "portfolio_daily_receiver_candidate_without_target_required_day_share": float(
            allocation_closure.get("receiver_candidate_without_target_required_day_share", 0.0) or 0.0
        ),
        "portfolio_daily_source_dead_day_share": float(
            allocation_closure.get("source_dead_day_share", 0.0) or 0.0
        ),
        "portfolio_daily_allocation_objective_mean": float(
            allocation_closure.get("allocation_objective_mean", 0.0) or 0.0
        ),
        "portfolio_daily_native_fallback_mean": float(
            allocation_closure.get("native_fallback_mean", 0.0) or 0.0
        ),
        "portfolio_daily_target_sum_gap": float(
            allocation_closure.get("target_sum_gap", 0.0) or 0.0
        ),
        "portfolio_daily_cash_funded_deploy_amount_mean": float(
            allocation_closure.get("cash_funded_deploy_amount_mean", 0.0) or 0.0
        ),
        "portfolio_daily_source_funded_deploy_amount_mean": float(
            allocation_closure.get("source_funded_deploy_amount_mean", 0.0) or 0.0
        ),
        "portfolio_daily_unused_receiver_headroom_mean": float(
            allocation_closure.get("unused_receiver_headroom_mean", 0.0) or 0.0
        ),
        "portfolio_daily_receiver_headroom_utilization_mean": float(
            allocation_closure.get("receiver_headroom_utilization_mean", 0.0) or 0.0
        ),
        "portfolio_daily_source_release_required": float(
            bool(allocation_closure.get("source_release_required", False))
        ),
        "portfolio_daily_underdeployment_reason": str(
            allocation_closure.get("underdeployment_reason", "") or ""
        ),
        "release_first_source_intent_count": float(
            allocation_closure.get("release_first_source_intent_count", 0.0) or 0.0
        ),
        "release_first_source_realized_count": float(
            allocation_closure.get("release_first_source_realized_count", 0.0) or 0.0
        ),
        "release_first_rotation_amount_mean": float(
            allocation_closure.get("release_first_rotation_amount_mean", 0.0) or 0.0
        ),
        "release_first_cash_buffer_amount_mean": float(
            allocation_closure.get("release_first_cash_buffer_amount_mean", 0.0) or 0.0
        ),
        "release_first_block_reason": str(allocation_closure.get("release_first_block_reason", "") or ""),
        "avg_value_arbitration_target": avg_value_arbitration_target,
        "avg_deploy_value_target": avg_deploy_value_target,
        "avg_release_value_target": avg_release_value_target,
        "avg_defense_value_target": avg_defense_value_target,
        "avg_deploy_gate_target": avg_deploy_gate_target,
        "avg_release_gate_target": avg_release_gate_target,
        "avg_defense_gate_target": avg_defense_gate_target,
        "avg_deploy_executability_target": avg_deploy_executability_target,
        "deploy_intent_action_count": int(deploy_intent_count),
        "deploy_intent_realized_count": int(deploy_intent_realized_count),
        "deploy_intent_realized_rate": deploy_intent_realized_rate,
        "deploy_intent_unrealized_count": int(deploy_intent_unrealized_count),
        "deploy_intent_unrealized_share": deploy_intent_unrealized_share,
        "open_add_positive_weight_change_rate": open_add_positive_weight_change_rate,
        "add_to_hold_conflict_count": int(add_to_hold_conflict_count),
        "add_to_hold_conflict_share": add_to_hold_conflict_share,
        "deploy_intent_hold_conflict_share": deploy_intent_hold_conflict_share,
        "deploy_intent_dropped_count": int(deploy_intent_dropped_count),
        "deploy_intent_dropped_share": deploy_intent_dropped_share,
        "avg_deploy_intent_candidate_count": avg_deploy_intent_candidate_count,
        "avg_deploy_intent_candidate_realized_rate": avg_deploy_intent_candidate_realized_rate,
        "avg_deploy_intent_candidate_budget_drop_share": avg_deploy_intent_candidate_budget_drop_share,
        "sell_intent_action_count": int(sell_intent_count),
        "sell_intent_realized_count": int(sell_intent_realized_count),
        "sell_intent_realized_rate": sell_intent_realized_rate,
        "sell_intent_hold_conflict_count": int(sell_intent_hold_conflict_count),
        "sell_intent_hold_conflict_share": sell_intent_hold_conflict_share,
        "sell_intent_suppressed_count": int(sell_intent_suppressed_count),
        "sell_intent_suppressed_share": sell_intent_suppressed_share,
        "realized_sell_action_count": int(realized_sell_count),
        "budget_origin_sell_count": int(budget_origin_sell_count),
        "budget_origin_sell_share": budget_origin_sell_share,
        "model_release_signal_sell_count": int(model_release_signal_sell_count),
        "model_release_signal_sell_share": model_release_signal_sell_share,
        "deploy_funding_rebalance_sell_count": int(deploy_funding_rebalance_sell_count),
        "deploy_funding_rebalance_sell_share": deploy_funding_rebalance_sell_share,
        "direct_action_pair_reallocation_sell_count": int(direct_action_pair_reallocation_sell_count),
        "direct_action_pair_reallocation_sell_share": direct_action_pair_reallocation_sell_share,
        "budget_slot_reclaim_sell_count": int(budget_slot_reclaim_sell_count),
        "budget_slot_reclaim_sell_share": budget_slot_reclaim_sell_share,
        "sell_priority_guard_sell_count": int(sell_priority_guard_sell_count),
        "sell_priority_guard_sell_share": sell_priority_guard_sell_share,
        "turnover_trim_sell_count": int(turnover_trim_sell_count),
        "turnover_trim_sell_share": turnover_trim_sell_share,
        "forced_zero_sell_count": int(forced_zero_sell_count),
        "forced_zero_sell_share": forced_zero_sell_share,
        "model_origin_sell_forward_excess_5d": model_origin_sell_forward_excess_5d,
        "model_authorized_sell_forward_excess_5d": model_authorized_sell_forward_excess_5d,
        "model_release_signal_forward_excess_5d": model_release_signal_forward_excess_5d,
        "deploy_funding_rebalance_forward_excess_5d": deploy_funding_rebalance_forward_excess_5d,
        "budget_origin_sell_forward_excess_5d": budget_origin_sell_forward_excess_5d,
        "avg_disciplined_funding_need": avg_disciplined_funding_need,
        "avg_protected_hold_support": avg_protected_hold_support,
        "avg_funding_release_support": avg_funding_release_support,
        "model_release_signal_disciplined_funding_need": model_release_signal_disciplined_funding_need,
        "model_release_signal_keep_support": model_release_signal_keep_support,
        "model_release_signal_release_support": model_release_signal_release_support,
        "deploy_funding_disciplined_funding_need": deploy_funding_disciplined_funding_need,
        "deploy_funding_keep_support": deploy_funding_keep_support,
        "deploy_funding_release_support": deploy_funding_release_support,
        "model_release_against_protected_hold_share": model_release_against_protected_hold_share,
        "model_release_release_consistent_share": model_release_release_consistent_share,
        "deploy_funding_against_protected_hold_share": deploy_funding_against_protected_hold_share,
        "deploy_funding_release_consistent_share": deploy_funding_release_consistent_share,
        "avg_clipped_intent_risk": avg_clipped_intent_risk,
        "clipped_intent_risk_conflict_gap": clipped_intent_risk_conflict_gap,
        "high_cash_up_market_share": high_cash_up_market_share,
        "high_cash_down_market_share": high_cash_down_market_share,
        "high_cash_sell_action_count": int(high_cash_sell_action_count),
        "high_cash_budget_origin_sell_share": high_cash_budget_origin_sell_share,
        "sell_execution_origin_counts": sell_execution_origin_counts,
        "sell_suppression_origin_counts": sell_suppression_origin_counts,
        "sell_source_floor_guard_count": int(working["sell_source_floor_guarded"].sum()),
        "sell_source_floor_guard_share": float(working["sell_source_floor_guarded"].mean()) if len(working) else 0.0,
        **release_translation_deploy_health,
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

    prepared = prepare_policy_inputs(**_resolve_audit_prepare_kwargs(evaluation_summary))
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
    recommended_focus: list[str] = []
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
    if (
        _safe_float(model_continuity, "sell_rank_forward_alignment_5d") < 0.01
        or _safe_float(model_continuity, "lifecycle_sell_gate_forward_alignment_5d") < 0.01
    ):
        bottlenecks.append(
            {
                "name": "lifecycle_sell_arbitration_not_aligned",
                "severity": "high",
                "diagnosis": "新增的卖出排序/生命周期卖出门控尚未和后续收益形成稳定负相关，说明个股生命周期仲裁仍没有真正学会“该卖哪只”。",
                "evidence": {
                    "model_sell_rank_forward_alignment_5d": _safe_float(model_continuity, "sell_rank_forward_alignment_5d"),
                    "teacher_sell_rank_forward_alignment_5d": _safe_float(teacher_continuity, "sell_rank_forward_alignment_5d"),
                    "model_lifecycle_sell_gate_forward_alignment_5d": _safe_float(model_continuity, "lifecycle_sell_gate_forward_alignment_5d"),
                    "teacher_lifecycle_sell_gate_forward_alignment_5d": _safe_float(teacher_continuity, "lifecycle_sell_gate_forward_alignment_5d"),
                    "avg_clipped_intent_risk": _safe_float(model_continuity, "avg_clipped_intent_risk"),
                    "clipped_intent_risk_conflict_gap": _safe_float(model_continuity, "clipped_intent_risk_conflict_gap"),
                },
            }
        )
    if (
        _safe_float(model_continuity, "sell_rank_forward_alignment_5d") < 0.01
        or _safe_float(model_continuity, "lifecycle_sell_gate_forward_alignment_5d") < 0.01
    ):
        recommended_focus.append("优先训练并验证生命周期卖出仲裁：卖出排序、卖出门控和预算剪裁风险必须共同解释真实 reduce / exit，而不是只靠动作分类。")
    if (
        _safe_float(model_continuity, "value_arbitration_forward_alignment_5d") < 0.02
        or _safe_float(model_continuity, "alpha_opportunity_forward_alignment_5d") < 0.02
        or _safe_float(model_continuity, "cash_defense_timing_quality_1d") < 0.0
    ):
        bottlenecks.append(
            {
                "name": "unified_value_arbitration_not_aligned",
                "severity": "high",
                "diagnosis": "r6 的核心价值仲裁还没有稳定对齐后验收益：机会、持有、卖出释放与现金防御仍可能在互相抵消，而不是形成统一资金去留判断。",
                "evidence": {
                    "model_value_arbitration_forward_alignment_5d": _safe_float(model_continuity, "value_arbitration_forward_alignment_5d"),
                    "teacher_value_arbitration_forward_alignment_5d": _safe_float(teacher_continuity, "value_arbitration_forward_alignment_5d"),
                    "model_alpha_opportunity_forward_alignment_5d": _safe_float(model_continuity, "alpha_opportunity_forward_alignment_5d"),
                    "model_cash_defense_timing_quality_1d": _safe_float(model_continuity, "cash_defense_timing_quality_1d"),
                    "avg_alpha_opportunity_value": float(semantic_conflicts.get("avg_alpha_opportunity_value", 0.0) or 0.0),
                    "avg_sell_release_value": float(semantic_conflicts.get("avg_sell_release_value", 0.0) or 0.0),
                    "avg_cash_defense_value": float(semantic_conflicts.get("avg_cash_defense_value", 0.0) or 0.0),
                    "avg_value_arbitration_target": float(semantic_conflicts.get("avg_value_arbitration_target", 0.0) or 0.0),
                },
            }
        )
        recommended_focus.append("优先检查 r6 价值仲裁是否真正区分“值得继续部署资金”和“应该释放资金/留现金”，避免 alpha 机会、卖出释放、现金防御被平均成一个折中信号。")
    if (
        _safe_float(model_continuity, "deploy_gate_forward_alignment_5d") < 0.02
        or _safe_float(model_continuity, "release_gate_forward_alignment_5d") < 0.02
        or _safe_float(model_continuity, "defense_gate_timing_quality_1d") < 0.0
    ):
        bottlenecks.append(
            {
                "name": "three_value_gate_not_aligned",
                "severity": "high",
                "diagnosis": "r6b 三值门控尚未稳定形成资金边际价值竞争：deploy 应对齐正向收益，release 应对齐持仓后续弱势，defense 应对齐市场下行。",
                "evidence": {
                    "model_deploy_gate_forward_alignment_5d": _safe_float(model_continuity, "deploy_gate_forward_alignment_5d"),
                    "model_release_gate_forward_alignment_5d": _safe_float(model_continuity, "release_gate_forward_alignment_5d"),
                    "model_defense_gate_timing_quality_1d": _safe_float(model_continuity, "defense_gate_timing_quality_1d"),
                    "avg_deploy_gate_target": float(semantic_conflicts.get("avg_deploy_gate_target", 0.0) or 0.0),
                    "avg_release_gate_target": float(semantic_conflicts.get("avg_release_gate_target", 0.0) or 0.0),
                    "avg_defense_gate_target": float(semantic_conflicts.get("avg_defense_gate_target", 0.0) or 0.0),
                },
            }
        )
        recommended_focus.append("优先验证 r6b 三值门控是否形成 deploy/release/defense 的真实竞争；若 deploy gate 仍不对齐正收益，不要继续扩大模型。")
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
    if (
        float(semantic_conflicts.get("direct_action_value_mode_share", 0.0) or 0.0) >= 0.50
        and (
            float(semantic_conflicts.get("direct_action_order_translation_conflict_rate", 0.0) or 0.0) >= 0.08
            or float(semantic_conflicts.get("direct_action_value_low_margin_share", 0.0) or 0.0) >= 0.60
            or (
                float(semantic_conflicts.get("direct_action_deploy_authorized_count", 0.0) or 0.0) >= 5.0
                and float(semantic_conflicts.get("direct_action_deploy_authorized_realized_rate", 0.0) or 0.0) < 0.55
            )
        )
    ):
        bottlenecks.append(
            {
                "name": "direct_action_intent_not_preserved",
                "severity": "high",
                "diagnosis": "direct action 已经成为模型决策入口，但低边际动作和订单翻译仍让 open/add/hold/reduce/exit 的最终权重变化不稳定。",
                "evidence": {
                    "direct_action_value_mode_share": float(
                        semantic_conflicts.get("direct_action_value_mode_share", 0.0) or 0.0
                    ),
                    "direct_action_value_gap_mean": float(
                        semantic_conflicts.get("direct_action_value_gap_mean", 0.0) or 0.0
                    ),
                    "direct_action_value_low_margin_share": float(
                        semantic_conflicts.get("direct_action_value_low_margin_share", 0.0) or 0.0
                    ),
                    "direct_action_order_translation_conflict_rate": float(
                        semantic_conflicts.get("direct_action_order_translation_conflict_rate", 0.0) or 0.0
                    ),
                    "direct_action_intent_preserved_share": float(
                        semantic_conflicts.get("direct_action_intent_preserved_share", 0.0) or 0.0
                    ),
                    "direct_action_funding_authorized_sell_share": float(
                        semantic_conflicts.get("direct_action_funding_authorized_sell_share", 0.0) or 0.0
                    ),
                    "direct_action_deploy_advantage_mean": float(
                        semantic_conflicts.get("direct_action_deploy_advantage_mean", 0.0) or 0.0
                    ),
                    "direct_action_deploy_signal_count": float(
                        semantic_conflicts.get("direct_action_deploy_signal_count", 0.0) or 0.0
                    ),
                    "direct_action_core_deploy_target_count": float(
                        semantic_conflicts.get("direct_action_core_deploy_target_count", 0.0) or 0.0
                    ),
                    "direct_action_core_deploy_target_realized_rate": float(
                        semantic_conflicts.get("direct_action_core_deploy_target_realized_rate", 0.0) or 0.0
                    ),
                    "direct_action_add_authorized_count": float(
                        semantic_conflicts.get("direct_action_add_authorized_count", 0.0) or 0.0
                    ),
                    "direct_action_add_authorized_realized_rate": float(
                        semantic_conflicts.get("direct_action_add_authorized_realized_rate", 0.0) or 0.0
                    ),
                    "direct_action_deploy_authorized_count": float(
                        semantic_conflicts.get("direct_action_deploy_authorized_count", 0.0) or 0.0
                    ),
                    "direct_action_deploy_authorized_realized_rate": float(
                        semantic_conflicts.get("direct_action_deploy_authorized_realized_rate", 0.0) or 0.0
                    ),
                    "direct_action_reallocation_source_count": float(
                        semantic_conflicts.get("direct_action_reallocation_source_count", 0.0) or 0.0
                    ),
                    "direct_action_pair_reallocation_source_count": float(
                        semantic_conflicts.get("direct_action_pair_reallocation_source_count", 0.0) or 0.0
                    ),
                    "direct_action_pair_cost_guard_pass_rate": float(
                        semantic_conflicts.get("direct_action_pair_cost_guard_pass_rate", 0.0) or 0.0
                    ),
                    "direct_action_pair_source_spread_mean": float(
                        semantic_conflicts.get("direct_action_pair_source_spread_mean", 0.0) or 0.0
                    ),
                    "direct_action_pair_source_cost_mean": float(
                        semantic_conflicts.get("direct_action_pair_source_cost_mean", 0.0) or 0.0
                    ),
                    "direct_action_core_minus_pair_forward_excess_5d": float(
                        semantic_conflicts.get("direct_action_core_minus_pair_forward_excess_5d", 0.0) or 0.0
                    ),
                },
            }
        )
    if (
        float(semantic_conflicts.get("action_value_consistency_score", 1.0) or 1.0) < 0.78
        or float(semantic_conflicts.get("sell_against_keep_value_share", 0.0) or 0.0) >= 0.18
        or float(semantic_conflicts.get("keep_against_release_value_share", 0.0) or 0.0) >= 0.18
    ):
        bottlenecks.append(
            {
                "name": "action_value_contract_not_unified",
                "severity": "high",
                "diagnosis": "buy/add/hold/reduce/exit are not yet ordered by one shared multi-horizon action-value contract.",
                "evidence": {
                    "action_value_consistency_score": float(
                        semantic_conflicts.get("action_value_consistency_score", 0.0) or 0.0
                    ),
                    "action_value_conflict_share": float(
                        semantic_conflicts.get("action_value_conflict_share", 0.0) or 0.0
                    ),
                    "sell_against_keep_value_share": float(
                        semantic_conflicts.get("sell_against_keep_value_share", 0.0) or 0.0
                    ),
                    "keep_against_release_value_share": float(
                        semantic_conflicts.get("keep_against_release_value_share", 0.0) or 0.0
                    ),
                    "open_low_action_value_share": float(
                        semantic_conflicts.get("open_low_action_value_share", 0.0) or 0.0
                    ),
                },
            }
        )
    if (
        float(semantic_conflicts.get("deploy_intent_realized_rate", 0.0) or 0.0) < 0.55
        and float(semantic_conflicts.get("deploy_intent_action_count", 0.0) or 0.0) >= 5
    ) or float(semantic_conflicts.get("add_to_hold_conflict_share", 0.0) or 0.0) >= 0.20:
        bottlenecks.append(
            {
                "name": "deploy_intent_not_executable",
                "severity": "high",
                "diagnosis": "open/add 部署意图未被稳定翻译为真实加仓，尤其需要跟踪 add -> hold 是否仍是主冲突。",
                "evidence": {
                    "deploy_intent_action_count": float(semantic_conflicts.get("deploy_intent_action_count", 0.0) or 0.0),
                    "deploy_intent_realized_rate": float(semantic_conflicts.get("deploy_intent_realized_rate", 0.0) or 0.0),
                    "open_add_positive_weight_change_rate": float(
                        semantic_conflicts.get("open_add_positive_weight_change_rate", 0.0) or 0.0
                    ),
                    "add_to_hold_conflict_share": float(semantic_conflicts.get("add_to_hold_conflict_share", 0.0) or 0.0),
                    "deploy_intent_dropped_share": float(semantic_conflicts.get("deploy_intent_dropped_share", 0.0) or 0.0),
                    "avg_deploy_intent_candidate_budget_drop_share": float(
                        semantic_conflicts.get("avg_deploy_intent_candidate_budget_drop_share", 0.0) or 0.0
                    ),
                    "avg_deploy_executability_target": float(
                        semantic_conflicts.get("avg_deploy_executability_target", 0.0) or 0.0
                    ),
                    "direct_action_deploy_authorized_count": float(
                        semantic_conflicts.get("direct_action_deploy_authorized_count", 0.0) or 0.0
                    ),
                    "direct_action_deploy_authorized_realized_rate": float(
                        semantic_conflicts.get("direct_action_deploy_authorized_realized_rate", 0.0) or 0.0
                    ),
                    "direct_action_core_deploy_target_realized_rate": float(
                        semantic_conflicts.get("direct_action_core_deploy_target_realized_rate", 0.0) or 0.0
                    ),
                    "direct_action_reallocation_source_count": float(
                        semantic_conflicts.get("direct_action_reallocation_source_count", 0.0) or 0.0
                    ),
                    "direct_action_pair_reallocation_source_count": float(
                        semantic_conflicts.get("direct_action_pair_reallocation_source_count", 0.0) or 0.0
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
    if (
        float(semantic_conflicts.get("budget_origin_sell_share", 0.0) or 0.0) >= 0.30
        or float(semantic_conflicts.get("sell_intent_suppressed_share", 0.0) or 0.0) >= 0.20
    ):
        bottlenecks.append(
            {
                "name": "sell_execution_source_entangled",
                "severity": "high",
                "diagnosis": "真实卖出中仍有相当比例来自预算/翻译层被动创造，或模型自身的 reduce/exit 意图被压回 hold，卖出责任边界还没有真正稳定下来。",
                "evidence": {
                    "sell_intent_action_count": float(semantic_conflicts.get("sell_intent_action_count", 0.0) or 0.0),
                    "sell_intent_realized_rate": float(semantic_conflicts.get("sell_intent_realized_rate", 0.0) or 0.0),
                    "sell_intent_hold_conflict_share": float(
                        semantic_conflicts.get("sell_intent_hold_conflict_share", 0.0) or 0.0
                    ),
                    "sell_intent_suppressed_share": float(
                        semantic_conflicts.get("sell_intent_suppressed_share", 0.0) or 0.0
                    ),
                    "budget_origin_sell_share": float(semantic_conflicts.get("budget_origin_sell_share", 0.0) or 0.0),
                    "sell_execution_origin_counts": dict(semantic_conflicts.get("sell_execution_origin_counts", {}) or {}),
                    "sell_suppression_origin_counts": dict(
                        semantic_conflicts.get("sell_suppression_origin_counts", {}) or {}
                    ),
                },
            }
        )
    if (
        float(semantic_conflicts.get("deploy_funding_rebalance_sell_share", 0.0) or 0.0) >= 0.45
        and (
            float(semantic_conflicts.get("deploy_funding_rebalance_forward_excess_5d", 0.0) or 0.0) > 0.0
            or float(semantic_conflicts.get("deploy_funding_against_protected_hold_share", 0.0) or 0.0) >= 0.35
            or float(semantic_conflicts.get("deploy_funding_release_consistent_share", 0.0) or 0.0) <= 0.35
        )
    ):
        bottlenecks.append(
            {
                "name": "held_funding_release_arbitration_not_learned",
                "severity": "high",
                "diagnosis": "显式 funding rebalance 已替代了隐藏 budget-origin sell，但系统仍会卖出本该保护的旧仓来资助新部署，说明 held-side release/funding 仲裁还没有真正学会。",
                "evidence": {
                    "deploy_funding_rebalance_sell_share": float(
                        semantic_conflicts.get("deploy_funding_rebalance_sell_share", 0.0) or 0.0
                    ),
                    "deploy_funding_rebalance_forward_excess_5d": float(
                        semantic_conflicts.get("deploy_funding_rebalance_forward_excess_5d", 0.0) or 0.0
                    ),
                    "deploy_funding_disciplined_funding_need": float(
                        semantic_conflicts.get("deploy_funding_disciplined_funding_need", 0.0) or 0.0
                    ),
                    "deploy_funding_keep_support": float(
                        semantic_conflicts.get("deploy_funding_keep_support", 0.0) or 0.0
                    ),
                    "deploy_funding_release_support": float(
                        semantic_conflicts.get("deploy_funding_release_support", 0.0) or 0.0
                    ),
                    "deploy_funding_against_protected_hold_share": float(
                        semantic_conflicts.get("deploy_funding_against_protected_hold_share", 0.0) or 0.0
                    ),
                    "deploy_funding_release_consistent_share": float(
                        semantic_conflicts.get("deploy_funding_release_consistent_share", 0.0) or 0.0
                    ),
                    "release_gate_forward_alignment_5d": _safe_float(model_continuity, "release_gate_forward_alignment_5d"),
                    "value_arbitration_forward_alignment_5d": _safe_float(
                        model_continuity, "value_arbitration_forward_alignment_5d"
                    ),
                },
            }
        )
    if (
        float(semantic_conflicts.get("model_release_signal_sell_count", 0.0) or 0.0) >= 5.0
        and (
            float(semantic_conflicts.get("model_release_signal_forward_excess_5d", 0.0) or 0.0) > 0.0
            or float(semantic_conflicts.get("model_release_against_protected_hold_share", 0.0) or 0.0) >= 0.35
            or float(semantic_conflicts.get("model_release_release_consistent_share", 0.0) or 0.0) <= 0.35
        )
    ):
        bottlenecks.append(
            {
                "name": "release_signal_not_selective",
                "severity": "medium",
                "diagnosis": "模型给出的 release 信号仍不够选择性，被释放的持仓里仍混有较多应继续保护的旧仓，release head 还没有稳定学会“该放掉谁”。",
                "evidence": {
                    "model_release_signal_sell_count": float(
                        semantic_conflicts.get("model_release_signal_sell_count", 0.0) or 0.0
                    ),
                    "model_release_signal_forward_excess_5d": float(
                        semantic_conflicts.get("model_release_signal_forward_excess_5d", 0.0) or 0.0
                    ),
                    "model_release_signal_disciplined_funding_need": float(
                        semantic_conflicts.get("model_release_signal_disciplined_funding_need", 0.0) or 0.0
                    ),
                    "model_release_signal_keep_support": float(
                        semantic_conflicts.get("model_release_signal_keep_support", 0.0) or 0.0
                    ),
                    "model_release_signal_release_support": float(
                        semantic_conflicts.get("model_release_signal_release_support", 0.0) or 0.0
                    ),
                    "model_release_against_protected_hold_share": float(
                        semantic_conflicts.get("model_release_against_protected_hold_share", 0.0) or 0.0
                    ),
                    "model_release_release_consistent_share": float(
                        semantic_conflicts.get("model_release_release_consistent_share", 0.0) or 0.0
                    ),
                },
            }
        )
    if float(semantic_conflicts.get("action_value_consistency_score", 1.0) or 1.0) < 0.78:
        recommended_focus.append(
            "Next continuous_policy round should optimize one shared multi-horizon action-value contract for open/add/hold/reduce/exit before tuning release loss again."
        )
    if (
        float(semantic_conflicts.get("action_rows", 0.0) or 0.0) > 0.0
        and float(semantic_conflicts.get("release_translation_deploy_health_score", 0.0) or 0.0) < 0.50
    ):
        bottlenecks.append(
            {
                "name": "release_translation_deploy_triad_not_closed",
                "severity": "high",
                "diagnosis": "部署实现、旧仓释放与订单翻译尚未形成同向闭环；单独优化 deploy、release 或 sell-source 都可能把压力转移到另一个环节。",
                "evidence": {
                    "release_translation_deploy_health_score": float(
                        semantic_conflicts.get("release_translation_deploy_health_score", 0.0) or 0.0
                    ),
                    "release_translation_deploy_failure_mode": str(
                        semantic_conflicts.get("release_translation_deploy_failure_mode", "") or ""
                    ),
                    "release_translation_deploy_components": dict(
                        semantic_conflicts.get("release_translation_deploy_components", {}) or {}
                    ),
                    "deploy_intent_realized_rate": float(
                        semantic_conflicts.get("deploy_intent_realized_rate", 0.0) or 0.0
                    ),
                    "deploy_funding_release_consistent_share": float(
                        semantic_conflicts.get("deploy_funding_release_consistent_share", 0.0) or 0.0
                    ),
                    "order_translation_conflict_rate": float(
                        semantic_conflicts.get("order_translation_conflict_rate", 0.0) or 0.0
                    ),
                },
            }
        )
    if (
        _safe_float(model_continuity, "cash_timing_quality_1d") < 0.02
        and float(semantic_conflicts.get("high_cash_budget_origin_sell_share", 0.0) or 0.0) >= 0.50
        and float(semantic_conflicts.get("high_cash_sell_action_count", 0.0) or 0.0) >= 3.0
    ):
        bottlenecks.append(
            {
                "name": "cash_timing_still_passive",
                "severity": "high",
                "diagnosis": "高现金日的卖出更多来自预算层被动挤压，而不是模型主动给出的 release/exit，说明 cash timing 仍更像结果后的被动收缩。",
                "evidence": {
                    "cash_timing_quality_1d": _safe_float(model_continuity, "cash_timing_quality_1d"),
                    "high_cash_sell_action_count": float(
                        semantic_conflicts.get("high_cash_sell_action_count", 0.0) or 0.0
                    ),
                    "high_cash_budget_origin_sell_share": float(
                        semantic_conflicts.get("high_cash_budget_origin_sell_share", 0.0) or 0.0
                    ),
                    "budget_origin_sell_share": float(semantic_conflicts.get("budget_origin_sell_share", 0.0) or 0.0),
                },
            }
        )

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
    if float(semantic_conflicts.get("direct_action_order_translation_conflict_rate", 0.0) or 0.0) >= 0.08:
        recommended_focus.append("下一轮优先使用 direct-action-preserving translation，让预算层只做现金/风控约束，不再吞掉 direct action intent。")
    if (
        float(semantic_conflicts.get("direct_action_deploy_authorized_count", 0.0) or 0.0) >= 5.0
        and float(semantic_conflicts.get("direct_action_deploy_authorized_realized_rate", 0.0) or 0.0) < 0.55
    ):
        recommended_focus.append("对高置信 direct add/open 建立显式可成交预算再分配，优先从未受保护的弱持仓释放小额资金。")
    if float(semantic_conflicts.get("direct_action_value_low_margin_share", 0.0) or 0.0) >= 0.60:
        recommended_focus.append("对低边际 direct action 增加 hold/cash abstain 纪律，避免弱优势动作被强行交易。")
    if (
        float(semantic_conflicts.get("action_rows", 0.0) or 0.0) > 0.0
        and float(semantic_conflicts.get("release_translation_deploy_health_score", 0.0) or 0.0) < 0.50
    ):
        recommended_focus.append("下一轮 continuous_policy 应使用 release/translation/deploy 联合目标审计，避免只把 release loss、deploy realized 或 sell-source cleanliness 单点做高。")
    if float(semantic_conflicts.get("budget_clipped_order_translation_conflict_rate", 0.0) or 0.0) > float(
        semantic_conflicts.get("unclipped_order_translation_conflict_rate", 0.0) or 0.0
    ) + 0.03:
        recommended_focus.append("把预算头与个股动作头分层，让 cash / gross / turnover 只做组合部署，不再吞掉 reduce / exit 语义。")
    if float(semantic_conflicts.get("budget_origin_sell_share", 0.0) or 0.0) >= 0.30:
        recommended_focus.append("把真实卖出拆成模型主动卖与预算被动卖两条链，优先减少预算层主动创造 reduce / exit。")
    if float(semantic_conflicts.get("sell_intent_suppressed_share", 0.0) or 0.0) >= 0.20:
        recommended_focus.append("优先检查 reduce / exit 意图被谁压回 hold：semantic guard、translation floor 还是 turnover trim，不要再把所有卖出失败都归咎给模型本身。")
    if float(semantic_conflicts.get("deploy_funding_against_protected_hold_share", 0.0) or 0.0) >= 0.35:
        recommended_focus.append("把 held-side funding 卖出单独拆出来复盘，优先减少“为了给新机会腾资金而卖掉本该继续保护的旧仓”这类伪 release。")
    if (
        float(semantic_conflicts.get("model_release_signal_sell_count", 0.0) or 0.0) >= 5.0
        and float(semantic_conflicts.get("model_release_against_protected_hold_share", 0.0) or 0.0) >= 0.35
    ):
        recommended_focus.append("把 release head 从泛化的卖出开关收紧成真正的旧仓释放头，要求 release 信号优先落在低 continuation / 高 release support 的持仓上。")
    if (
        _safe_float(model_continuity, "cash_timing_quality_1d") < 0.02
        and float(semantic_conflicts.get("high_cash_budget_origin_sell_share", 0.0) or 0.0) >= 0.50
    ):
        recommended_focus.append("cash timing 需要从“预算挤压后留下更多现金”升级为“模型主动决定何时释放风险暴露”，避免高现金主要靠被动卖出形成。")
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
    if args.export_held_side_details:
        detail_frame, detail_payload = _build_held_side_detail_payload(
            action_outcomes=action_outcomes,
            run_tag=str(payload["run_tag"]),
            detail_limit=int(args.held_side_detail_limit),
        )
        if not detail_frame.empty:
            ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
            detail_csv_path = ANALYSIS_ROOT / f"{payload['run_tag']}__held_side_details.csv"
            detail_summary_path = ANALYSIS_ROOT / f"{payload['run_tag']}__held_side_details.json"
            detail_frame.to_csv(detail_csv_path, index=False, encoding="utf-8-sig")
            detail_payload["csv_path"] = str(detail_csv_path.resolve())
            detail_payload["summary_json"] = str(detail_summary_path.resolve())
            write_json(detail_summary_path, detail_payload)
        payload["held_side_details"] = detail_payload
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = ANALYSIS_ROOT / f"{payload['run_tag']}.json"
    write_json(output_path, payload)
    payload["output_path"] = str(output_path.resolve())
    update_latest_summary("behavior_audit", payload)
    safe_print_json(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
