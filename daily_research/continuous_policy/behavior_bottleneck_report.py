from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _section(summary: dict[str, Any], *keys: str) -> dict[str, Any]:
    value: Any = summary
    for key in keys:
        value = (value or {}).get(key, {})
    return dict(value or {}) if isinstance(value, dict) else {}


def _float(mapping: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(mapping.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def _continuity_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    merged.update(_section(summary, "shadow", "continuity_metrics"))
    merged.update(_section(summary, "evaluation", "continuity_metrics"))
    merged.update(_section(summary, "continuity_metrics"))
    return merged


def build_behavior_bottleneck_report(protocol_summary: dict[str, Any]) -> dict[str, Any]:
    evaluation_metrics = _section(protocol_summary, "evaluation", "continuous_policy_metrics")
    continuity = _continuity_metrics(protocol_summary)
    training = _section(protocol_summary, "training_evidence")
    promotion = _section(protocol_summary, "promotion_gate")
    failed_checks = list(promotion.get("failed_checks", []) or [])

    cash_timing = _float(continuity, "cash_timing_quality_1d")
    reduce_success = _float(continuity, "reduce_success_rate_5d")
    exit_timeliness = _float(continuity, "exit_timeliness_rate_5d")
    source_count = _float(continuity, "portfolio_daily_source_target_count")
    source_sell_rate = _float(continuity, "portfolio_daily_source_realized_sell_rate")
    exposure_utilization = _float(continuity, "portfolio_daily_exposure_utilization")
    target_sum_gap = _float(continuity, "portfolio_daily_target_sum_gap")
    intent_conflict = _float(continuity, "intent_translation_conflict_rate")
    intent_conflict_present = "intent_translation_conflict_rate" in continuity
    best_epoch = int(training.get("best_epoch", 0) or 0)
    completed_epochs = int(training.get("completed_epochs", 0) or 0)

    blockers: list[str] = []
    if cash_timing < -0.05 or "cash_timing_quality_1d" in failed_checks:
        blockers.append("cash_timing_negative")
    if source_count < 1.0 or source_sell_rate < 0.20:
        blockers.append("sell_intent_dead")
    if reduce_success < 0.45 or "reduce_success_rate_5d" in failed_checks:
        blockers.append("reduce_quality_weak")
    if exit_timeliness < 0.45 or "exit_timeliness_rate_5d" in failed_checks:
        blockers.append("exit_timeliness_weak")
    if str(training.get("status", "")) != "sufficient":
        blockers.append("training_evidence_insufficient")
    if completed_epochs > 0 and best_epoch > max(completed_epochs - 2, 0):
        blockers.append("best_epoch_at_edge")

    allocation_closure_score = max(0.0, min(1.0, exposure_utilization)) * max(
        0.0,
        1.0 - min(target_sum_gap, 1.0),
    )
    if intent_conflict == 0.0:
        if intent_conflict_present:
            allocation_closure_score = max(allocation_closure_score, 0.90)

    return {
        "run_tag": str(protocol_summary.get("run_tag", "")),
        "primary_blocker": blockers[0] if blockers else "none",
        "blockers": blockers,
        "metrics": {
            "annual_return": _float(evaluation_metrics, "annual_return"),
            "monthly_return_mean": _float(evaluation_metrics, "monthly_return_mean"),
            "max_drawdown": _float(evaluation_metrics, "max_drawdown"),
            "cash_timing_quality_1d": cash_timing,
            "reduce_success_rate_5d": reduce_success,
            "exit_timeliness_rate_5d": exit_timeliness,
            "portfolio_daily_source_target_count": source_count,
            "portfolio_daily_source_realized_sell_rate": source_sell_rate,
            "portfolio_daily_exposure_utilization": exposure_utilization,
            "portfolio_daily_target_sum_gap": target_sum_gap,
            "intent_translation_conflict_rate": intent_conflict,
        },
        "progress_assets": {
            "allocation_closure_score": allocation_closure_score,
            "intent_translation_clean": bool(intent_conflict_present and intent_conflict <= 0.01),
            "exposure_closed": exposure_utilization >= 0.60 and target_sum_gap <= 0.05,
        },
        "next_focus": [
            "cash_timing_direction",
            "source_release_intent",
            "reduce_exit_realization",
            "strict_resume_training_evidence",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a continuous-policy behavior bottleneck report.")
    parser.add_argument("--protocol-summary-json", required=True)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args(argv)

    protocol_path = Path(args.protocol_summary_json)
    summary = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = build_behavior_bottleneck_report(summary)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
