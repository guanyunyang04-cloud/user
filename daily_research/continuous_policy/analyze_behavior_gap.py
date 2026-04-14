from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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
    )

    model_continuity = dict(evaluation_summary.get("continuity_metrics", {}) or {})
    teacher_continuity = dict(teacher_rollout.get("continuity_metrics", {}) or {})
    model_metrics = dict(evaluation_summary.get("continuous_policy_metrics", {}) or {})
    teacher_metrics = dict(teacher_rollout.get("metrics", {}) or {})
    action_outcomes = _read_csv(str(evaluation_summary.get("action_outcomes_csv", "") or ""))

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

    recommended_focus: list[str] = []
    if model_hold_share < 0.12:
        recommended_focus.append("先把 hold_share 拉过最低连续持有阈值，再去追求更复杂的收益最优行为。")
    else:
        recommended_focus.append("保住当前持有连续性，不要在修 reduce / cash / reversal 时把 hold_share 再打回塌缩。")
    if _safe_float(model_continuity, "reduce_success_rate_5d") < 0.45:
        recommended_focus.append("优先把 reduce 聚焦到真正的信号衰减与防守切换，而不是盈利后的机械落袋。")
    if _safe_float(model_continuity, "cash_timing_quality_1d") < 0.02:
        recommended_focus.append("继续强化组合层 cash / gross / turnover 预算头，让市场转弱时能主动降暴露。")
    if early_exit_share > 0.20:
        recommended_focus.append("延缓过早退出，把生命周期未完成阶段和真正失效阶段分开。")
    if _safe_float(model_continuity, "immediate_reversal_rate_3d") > 0.35:
        recommended_focus.append("优先压低 shadow reversal，避免刚减仓或退出后又被过快追回。")
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
