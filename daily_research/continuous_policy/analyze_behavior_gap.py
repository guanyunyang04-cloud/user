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
    return pd.read_csv(path)


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
    teacher_metrics = dict(evaluation_summary.get("teacher_oracle_metrics", {}) or {})
    action_outcomes = _read_csv(str(evaluation_summary.get("action_outcomes_csv", "") or ""))

    early_reduce_share = 0.0
    early_exit_share = 0.0
    profitable_reduce_share = 0.0
    if not action_outcomes.empty:
        reduce_rows = action_outcomes.loc[action_outcomes["execution_action"].astype(str) == "reduce"].copy()
        exit_rows = action_outcomes.loc[action_outcomes["execution_action"].astype(str) == "exit"].copy()
        early_reduce_share = float((reduce_rows["hold_days_before"].fillna(999.0) <= 3.0).mean()) if not reduce_rows.empty else 0.0
        early_exit_share = float((exit_rows["hold_days_before"].fillna(999.0) <= 3.0).mean()) if not exit_rows.empty else 0.0
        profitable_reduce_share = (
            float((reduce_rows["unrealized_pnl_before"].fillna(0.0) > 0.05).mean()) if not reduce_rows.empty else 0.0
        )

    gap_rows = [
        _gap_row("hold_share", teacher=_safe_float(teacher_continuity, "hold_share"), model=_safe_float(model_continuity, "hold_share"), preferred_direction="higher_better"),
        _gap_row("reduce_success_rate_5d", teacher=_safe_float(teacher_continuity, "reduce_success_rate_5d"), model=_safe_float(model_continuity, "reduce_success_rate_5d"), preferred_direction="higher_better"),
        _gap_row("cash_timing_quality_1d", teacher=_safe_float(teacher_continuity, "cash_timing_quality_1d"), model=_safe_float(model_continuity, "cash_timing_quality_1d"), preferred_direction="higher_better"),
        _gap_row("hold_retention_quality_5d", teacher=_safe_float(teacher_continuity, "hold_retention_quality_5d"), model=_safe_float(model_continuity, "hold_retention_quality_5d"), preferred_direction="higher_better"),
        _gap_row("avg_turnover", teacher=_safe_float(teacher_metrics, "avg_turnover"), model=_safe_float(model_metrics, "avg_turnover"), preferred_direction="lower_better"),
        _gap_row("churn_ratio", teacher=_safe_float(teacher_continuity, "churn_ratio"), model=_safe_float(model_continuity, "churn_ratio"), preferred_direction="lower_better"),
    ]

    bottlenecks: list[dict[str, Any]] = []
    if _safe_float(model_continuity, "hold_share") < 0.12:
        bottlenecks.append(
            {
                "name": "hold_share_collapse",
                "severity": "high",
                "diagnosis": "模型仍明显不会持有，持有占比还没有达到日频连续代理应有的最低水平。",
                "evidence": {
                    "model_hold_share": _safe_float(model_continuity, "hold_share"),
                    "teacher_hold_share": _safe_float(teacher_continuity, "hold_share"),
                },
            }
        )
    if _safe_float(model_continuity, "reduce_success_rate_5d") < 0.45:
        bottlenecks.append(
            {
                "name": "reduce_too_early_or_wrong_side",
                "severity": "high",
                "diagnosis": "减仓后 5 日成功率过低，说明模型仍在错误地对强势票过早减仓，或没有把减仓聚焦到真正衰减的票。",
                "evidence": {
                    "model_reduce_success_rate_5d": _safe_float(model_continuity, "reduce_success_rate_5d"),
                    "teacher_reduce_success_rate_5d": _safe_float(teacher_continuity, "reduce_success_rate_5d"),
                    "early_reduce_share": early_reduce_share,
                    "profitable_reduce_share": profitable_reduce_share,
                },
            }
        )
    if _safe_float(model_continuity, "cash_timing_quality_1d") < 0.02:
        bottlenecks.append(
            {
                "name": "cash_timing_not_learned",
                "severity": "high",
                "diagnosis": "现金时机质量仍然接近 0 或为负，说明组合层预算头还没学会在市场走弱时主动降暴露。",
                "evidence": {
                    "model_cash_timing_quality_1d": _safe_float(model_continuity, "cash_timing_quality_1d"),
                    "teacher_cash_timing_quality_1d": _safe_float(teacher_continuity, "cash_timing_quality_1d"),
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
                "diagnosis": "过早退出占比偏高，说明模型对持仓生命周期的未完成阶段容忍度仍不够。",
                "evidence": {"early_exit_share": early_exit_share},
            }
        )

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
        },
        "bottlenecks": bottlenecks,
        "recommended_focus": [
            "先提升 hold_share，再去追求更复杂的收益最优行为。",
            "优先把 reduce 聚焦到信号衰减和防守切换，而不是盈利后机械落袋。",
            "继续加强组合层 cash / gross / turnover 预算头的防守学习。",
        ],
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
