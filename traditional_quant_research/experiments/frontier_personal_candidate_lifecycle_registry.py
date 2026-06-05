"""Consolidate personal candidate lifecycle state across bootstrap, plan, and review artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from traditional_quant_research.experiments.frontier_personal_candidate_gate import (
    PERSONAL_BACKTEST_ONLY_LEVEL,
    PERSONAL_BACKTEST_PROMOTION_LEVEL,
)
from traditional_quant_research.experiments.frontier_personal_paper_tracking_bootstrap import (
    DEFAULT_OUTPUT_DIR as DEFAULT_BOOTSTRAP_OUTPUT_ROOT,
    PAPER_TRACKING_CANCELLED_MESSAGE,
    PERSONAL_PAPER_CANDIDATE_LEVEL,
)
from traditional_quant_research.experiments.frontier_personal_paper_tracking_plan import (
    DEFAULT_OUTPUT_DIR as DEFAULT_PLAN_OUTPUT_ROOT,
    FUTURE_RECORD_STATUS,
)
from traditional_quant_research.experiments.frontier_personal_paper_tracking_review import (
    DEFAULT_OUTPUT_DIR as DEFAULT_REVIEW_OUTPUT_ROOT,
)
from traditional_quant_research.experiments.frontier_promotion_gate import latest_run_dir


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/frontier_personal_candidate_lifecycle_registry")
DEFAULT_RESEARCH_LOG = Path(
    "traditional_quant_research/research_log/2026-06-04_frontier_personal_candidate_lifecycle_registry.md"
)


def run_frontier_personal_candidate_lifecycle_registry(
    *,
    bootstrap_run_dir: str | Path | None = None,
    plan_run_dir: str | Path | None = None,
    review_run_dir: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    """Write a current lifecycle registry for all bootstrapped personal candidates."""

    raise RuntimeError(PAPER_TRACKING_CANCELLED_MESSAGE)

    run_id = f"frontier_personal_candidate_lifecycle_registry_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    bootstrap_dir = Path(bootstrap_run_dir) if bootstrap_run_dir is not None else latest_run_dir(DEFAULT_BOOTSTRAP_OUTPUT_ROOT)
    candidates, protocol, bootstrap_summary = read_bootstrap_artifacts(bootstrap_dir)
    bootstrap_run_id = str(bootstrap_summary.get("run_id", ""))

    selected_plan_dir = _select_optional_run_dir(plan_run_dir, DEFAULT_PLAN_OUTPUT_ROOT)
    plan_calendar, plan_summary, plan_context_status = read_matching_plan_artifacts(selected_plan_dir, bootstrap_run_id)

    selected_review_dir = _select_optional_run_dir(review_run_dir, DEFAULT_REVIEW_OUTPUT_ROOT)
    review, review_summary, review_context_status = read_matching_review_artifacts(selected_review_dir, bootstrap_run_id)

    registry = build_lifecycle_registry(
        candidates,
        protocol,
        plan_calendar,
        review,
        bootstrap_summary=bootstrap_summary,
        plan_summary=plan_summary,
        review_summary=review_summary,
        plan_context_status=plan_context_status,
        review_context_status=review_context_status,
    )
    stage_counts = build_lifecycle_stage_counts(registry)
    summary = summarize_lifecycle_registry(
        registry,
        stage_counts,
        run_id=run_id,
        bootstrap_run_dir=bootstrap_dir,
        plan_run_dir=selected_plan_dir,
        review_run_dir=selected_review_dir,
        bootstrap_summary=bootstrap_summary,
        plan_summary=plan_summary,
        review_summary=review_summary,
        plan_context_status=plan_context_status,
        review_context_status=review_context_status,
    )
    markdown = render_lifecycle_registry_markdown(summary, registry, stage_counts)

    registry.to_csv(run_dir / "personal_candidate_lifecycle_registry.csv", index=False, encoding="utf-8-sig")
    stage_counts.to_csv(run_dir / "personal_candidate_lifecycle_stage_counts.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        path = Path(research_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    return {**summary, "run_dir": str(run_dir), "research_log": str(research_log_path) if write_research_log else None}


def read_bootstrap_artifacts(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    candidates_path = run_dir / "paper_tracking_candidates.csv"
    protocol_path = run_dir / "paper_tracking_protocol.csv"
    summary_path = run_dir / "summary.json"
    if not candidates_path.exists():
        raise FileNotFoundError(f"paper tracking candidates not found: {candidates_path}")
    if not protocol_path.exists():
        raise FileNotFoundError(f"paper tracking protocol not found: {protocol_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"paper tracking bootstrap summary not found: {summary_path}")
    return (
        pd.read_csv(candidates_path),
        pd.read_csv(protocol_path),
        json.loads(summary_path.read_text(encoding="utf-8-sig")),
    )


def read_matching_plan_artifacts(run_dir: Path | None, bootstrap_run_id: str) -> tuple[pd.DataFrame, dict[str, Any], str]:
    if run_dir is None:
        return pd.DataFrame(), {}, "plan_missing"
    summary_path = run_dir / "summary.json"
    calendar_path = run_dir / "paper_tracking_plan_calendar.csv"
    if not summary_path.exists() or not calendar_path.exists():
        return pd.DataFrame(), {}, "plan_artifacts_missing"
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if str(summary.get("bootstrap_run_id", "")) != bootstrap_run_id:
        return pd.DataFrame(), summary, "plan_bootstrap_mismatch"
    return pd.read_csv(calendar_path), summary, "plan_matched"


def read_matching_review_artifacts(run_dir: Path | None, bootstrap_run_id: str) -> tuple[pd.DataFrame, dict[str, Any], str]:
    if run_dir is None:
        return pd.DataFrame(), {}, "review_missing"
    summary_path = run_dir / "summary.json"
    review_path = run_dir / "paper_tracking_review_summary.csv"
    if not summary_path.exists() or not review_path.exists():
        return pd.DataFrame(), {}, "review_artifacts_missing"
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if str(summary.get("bootstrap_run_id", "")) != bootstrap_run_id:
        return pd.DataFrame(), summary, "review_bootstrap_mismatch"
    return pd.read_csv(review_path), summary, "review_matched"


def build_lifecycle_registry(
    candidates: pd.DataFrame,
    protocol: pd.DataFrame,
    plan_calendar: pd.DataFrame,
    review: pd.DataFrame,
    *,
    bootstrap_summary: Mapping[str, Any] | None = None,
    plan_summary: Mapping[str, Any] | None = None,
    review_summary: Mapping[str, Any] | None = None,
    plan_context_status: str = "plan_missing",
    review_context_status: str = "review_missing",
) -> pd.DataFrame:
    raise RuntimeError(PAPER_TRACKING_CANCELLED_MESSAGE)

    columns = _registry_columns()
    if candidates.empty:
        return pd.DataFrame(columns=columns)

    bootstrap_summary = bootstrap_summary or {}
    plan_summary = plan_summary or {}
    review_summary = review_summary or {}
    protocol_by_candidate = _records_by_candidate(protocol)
    plan_by_candidate = _group_records_by_candidate(plan_calendar)
    review_by_candidate = _records_by_candidate(review)

    rows: list[dict[str, Any]] = []
    for candidate in candidates.to_dict("records"):
        candidate_id = str(candidate.get("candidate_id", ""))
        protocol_row = protocol_by_candidate.get(candidate_id, {})
        plan_rows = plan_by_candidate.get(candidate_id, [])
        review_row = review_by_candidate.get(candidate_id, {})
        lifecycle = _derive_lifecycle(candidate, plan_rows, review_row, plan_context_status, review_context_status)
        next_expected_signal = _next_expected_signal_period_end(plan_rows)
        rows.append(
            {
                "candidate_id": candidate_id,
                "lifecycle_level": lifecycle["lifecycle_level"],
                "lifecycle_status": lifecycle["lifecycle_status"],
                "next_action": lifecycle["next_action"],
                "current_level": str(candidate.get("current_level", PERSONAL_BACKTEST_PROMOTION_LEVEL)),
                "target_next_level": str(candidate.get("target_next_level", PERSONAL_PAPER_CANDIDATE_LEVEL)),
                "tracking_status": str(candidate.get("tracking_status", "")),
                "plan_context_status": plan_context_status,
                "review_context_status": review_context_status,
                "bootstrap_run_id": str(bootstrap_summary.get("run_id", "")),
                "plan_run_id": str(plan_summary.get("run_id", "")),
                "review_run_id": str(review_summary.get("run_id", "")),
                "snapshot_id": str(bootstrap_summary.get("snapshot_id", "")),
                "signal": str(candidate.get("signal", "")),
                "constraint_variant": str(candidate.get("constraint_variant", "")),
                "exposure_penalty_strength": _optional_float(candidate.get("exposure_penalty_strength")),
                "rebalance_frequency": str(protocol_row.get("rebalance_frequency", candidate.get("rebalance_frequency", ""))),
                "horizon": _optional_int(protocol_row.get("horizon", candidate.get("horizon"))),
                "top_n": _optional_int(protocol_row.get("top_n", candidate.get("top_n"))),
                "fee_bps": _optional_float(protocol_row.get("fee_bps", candidate.get("fee_bps"))),
                "impact_bps_per_1pct": _optional_float(
                    protocol_row.get("impact_bps_per_1pct", candidate.get("impact_bps_per_1pct"))
                ),
                "personal_capital_amount": _optional_float(
                    protocol_row.get("personal_capital_amount", candidate.get("personal_capital_amount"))
                ),
                "mean_annualized_return": _optional_float(candidate.get("mean_annualized_return")),
                "min_annualized_return": _optional_float(candidate.get("min_annualized_return")),
                "positive_year_rate": _optional_float(candidate.get("positive_year_rate")),
                "worst_max_drawdown": _optional_float(candidate.get("worst_max_drawdown")),
                "total_periods": _optional_int(candidate.get("total_periods")),
                "planned_observation_count": int(len(plan_rows)),
                "future_observation_pending_count": _future_pending_count(plan_rows),
                "next_expected_signal_period_end": next_expected_signal,
                "completed_periods": _optional_int(review_row.get("completed_periods")) or 0,
                "tracking_days": _optional_int(review_row.get("tracking_days")) or 0,
                "cumulative_paper_return": _optional_float(review_row.get("cumulative_paper_return")),
                "cumulative_excess_return": _optional_float(review_row.get("cumulative_excess_return")),
                "worst_drawdown": _optional_float(review_row.get("worst_drawdown")),
                "failed_gates": str(review_row.get("failed_gates", "")),
                "review_recommendation": str(review_row.get("recommendation", "")),
                "personal_paper_candidate": lifecycle["lifecycle_level"] == PERSONAL_PAPER_CANDIDATE_LEVEL,
                "strategy_candidate": False,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_lifecycle_stage_counts(registry: pd.DataFrame) -> pd.DataFrame:
    columns = ["lifecycle_level", "lifecycle_status", "next_action", "candidate_count"]
    if registry.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        registry.groupby(["lifecycle_level", "lifecycle_status", "next_action"], dropna=False)
        .size()
        .reset_index(name="candidate_count")
        .sort_values(["lifecycle_level", "lifecycle_status", "next_action"])
        .reset_index(drop=True)
    )
    return grouped.reindex(columns=columns)


def summarize_lifecycle_registry(
    registry: pd.DataFrame,
    stage_counts: pd.DataFrame,
    *,
    run_id: str,
    bootstrap_run_dir: Path,
    plan_run_dir: Path | None,
    review_run_dir: Path | None,
    bootstrap_summary: Mapping[str, Any],
    plan_summary: Mapping[str, Any],
    review_summary: Mapping[str, Any],
    plan_context_status: str,
    review_context_status: str,
) -> dict[str, Any]:
    paper_count = int(registry["lifecycle_level"].astype(str).eq(PERSONAL_PAPER_CANDIDATE_LEVEL).sum()) if not registry.empty else 0
    backtest_count = (
        int(registry["lifecycle_level"].astype(str).eq(PERSONAL_BACKTEST_PROMOTION_LEVEL).sum()) if not registry.empty else 0
    )
    research_only_count = (
        int(registry["lifecycle_level"].astype(str).eq(PERSONAL_BACKTEST_ONLY_LEVEL).sum()) if not registry.empty else 0
    )
    action_counts = registry["next_action"].astype(str).value_counts().to_dict() if not registry.empty else {}
    if paper_count:
        decision = "personal_paper_candidate_review_ready"
    elif action_counts.get("continue_paper_tracking", 0):
        decision = "continue_paper_tracking"
    elif action_counts.get("fill_next_paper_observation", 0):
        decision = "fill_paper_tracking_records"
    elif action_counts.get("create_paper_tracking_plan", 0):
        decision = "create_paper_tracking_plan"
    elif research_only_count:
        decision = "rebuild_or_retire_candidates"
    else:
        decision = "no_personal_candidate"
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "north_star": "Baostock-only personal quant strategy research",
        "bootstrap_run_dir": str(bootstrap_run_dir),
        "bootstrap_run_id": str(bootstrap_summary.get("run_id", "")),
        "plan_run_dir": str(plan_run_dir) if plan_run_dir else "",
        "plan_run_id": str(plan_summary.get("run_id", "")),
        "plan_context_status": plan_context_status,
        "review_run_dir": str(review_run_dir) if review_run_dir else "",
        "review_run_id": str(review_summary.get("run_id", "")),
        "review_context_status": review_context_status,
        "candidate_count": int(len(registry)),
        "personal_backtest_candidate_count": backtest_count,
        "personal_paper_candidate_count": paper_count,
        "personal_research_only_count": research_only_count,
        "strategy_candidate_count": 0,
        "stage_count_rows": int(len(stage_counts)),
        "action_counts": {str(key): int(value) for key, value in action_counts.items()},
        "decision": decision,
        "limitations": [
            "This registry consolidates existing artifacts and does not generate signals, rerun backtests, or infer paper returns.",
            "Plan rows and empty live log starters remain incomplete observations, not paper evidence.",
            "A personal paper candidate remains separate from institutional strategy_candidate promotion.",
        ],
    }


def render_lifecycle_registry_markdown(
    summary: Mapping[str, Any],
    registry: pd.DataFrame,
    stage_counts: pd.DataFrame,
) -> str:
    lines = [
        "# Frontier Personal Candidate Lifecycle Registry",
        "",
        f"- run_id: `{summary.get('run_id', '')}`",
        f"- decision: `{summary.get('decision', '')}`",
        f"- candidate_count: `{summary.get('candidate_count', 0)}`",
        f"- personal_backtest_candidate_count: `{summary.get('personal_backtest_candidate_count', 0)}`",
        f"- personal_paper_candidate_count: `{summary.get('personal_paper_candidate_count', 0)}`",
        f"- personal_research_only_count: `{summary.get('personal_research_only_count', 0)}`",
        f"- strategy_candidate_count: `{summary.get('strategy_candidate_count', 0)}`",
        f"- plan_context_status: `{summary.get('plan_context_status', '')}`",
        f"- review_context_status: `{summary.get('review_context_status', '')}`",
        "",
        "## Stage Counts",
        "",
        _markdown_table(stage_counts),
        "",
        "## Registry",
        "",
        _markdown_table(registry),
        "",
        "## Interpretation",
        "",
        "This registry is the current personal-candidate lifecycle view. It only consolidates evidence already produced by "
        "bootstrap, plan, and review artifacts; it deliberately keeps `strategy_candidate_count=0`.",
        "",
    ]
    return "\n".join(lines)


def _derive_lifecycle(
    candidate: Mapping[str, Any],
    plan_rows: list[dict[str, Any]],
    review_row: Mapping[str, Any],
    plan_context_status: str,
    review_context_status: str,
) -> dict[str, str]:
    review_recommendation = str(review_row.get("recommendation", ""))
    review_level = str(review_row.get("promotion_level", ""))
    if review_context_status == "review_matched" and review_level == PERSONAL_PAPER_CANDIDATE_LEVEL:
        return {
            "lifecycle_level": PERSONAL_PAPER_CANDIDATE_LEVEL,
            "lifecycle_status": "paper_evidence_passed",
            "next_action": "prepare_personal_trading_review",
        }
    if review_context_status == "review_matched" and review_recommendation == "downgrade_to_personal_research":
        return {
            "lifecycle_level": PERSONAL_BACKTEST_ONLY_LEVEL,
            "lifecycle_status": "paper_evidence_failed",
            "next_action": "rebuild_or_retire_candidate",
        }
    if review_context_status == "review_matched" and review_recommendation == "continue_paper_tracking":
        return {
            "lifecycle_level": PERSONAL_BACKTEST_PROMOTION_LEVEL,
            "lifecycle_status": "paper_tracking_incomplete",
            "next_action": "continue_paper_tracking",
        }
    if plan_context_status == "plan_matched" and plan_rows:
        return {
            "lifecycle_level": str(candidate.get("current_level", PERSONAL_BACKTEST_PROMOTION_LEVEL)),
            "lifecycle_status": "tracking_plan_ready",
            "next_action": "fill_next_paper_observation",
        }
    return {
        "lifecycle_level": str(candidate.get("current_level", PERSONAL_BACKTEST_PROMOTION_LEVEL)),
        "lifecycle_status": "tracking_bootstrapped",
        "next_action": "create_paper_tracking_plan",
    }


def _select_optional_run_dir(user_value: str | Path | None, default_root: Path) -> Path | None:
    if user_value is not None:
        path = Path(user_value)
        return path if path.exists() else None
    try:
        return latest_run_dir(default_root)
    except FileNotFoundError:
        return None


def _records_by_candidate(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty or "candidate_id" not in frame.columns:
        return {}
    records = {}
    for row in frame.to_dict("records"):
        candidate_id = str(row.get("candidate_id", ""))
        if candidate_id:
            records[candidate_id] = row
    return records


def _group_records_by_candidate(frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    if frame.empty or "candidate_id" not in frame.columns:
        return {}
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in frame.to_dict("records"):
        candidate_id = str(row.get("candidate_id", ""))
        if candidate_id:
            groups.setdefault(candidate_id, []).append(row)
    return groups


def _future_pending_count(plan_rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in plan_rows if str(row.get("record_status", "")) == FUTURE_RECORD_STATUS)


def _next_expected_signal_period_end(plan_rows: list[dict[str, Any]]) -> str:
    dates = pd.to_datetime(
        [row.get("expected_signal_period_end", "") for row in plan_rows if _blank(row.get("actual_signal_date", ""))],
        errors="coerce",
    )
    clean = pd.Series(dates).dropna()
    if clean.empty:
        return ""
    return pd.Timestamp(clean.min()).date().isoformat()


def _blank(value: Any) -> bool:
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    return not str(value).strip()


def _registry_columns() -> list[str]:
    return [
        "candidate_id",
        "lifecycle_level",
        "lifecycle_status",
        "next_action",
        "current_level",
        "target_next_level",
        "tracking_status",
        "plan_context_status",
        "review_context_status",
        "bootstrap_run_id",
        "plan_run_id",
        "review_run_id",
        "snapshot_id",
        "signal",
        "constraint_variant",
        "exposure_penalty_strength",
        "rebalance_frequency",
        "horizon",
        "top_n",
        "fee_bps",
        "impact_bps_per_1pct",
        "personal_capital_amount",
        "mean_annualized_return",
        "min_annualized_return",
        "positive_year_rate",
        "worst_max_drawdown",
        "total_periods",
        "planned_observation_count",
        "future_observation_pending_count",
        "next_expected_signal_period_end",
        "completed_periods",
        "tracking_days",
        "cumulative_paper_return",
        "cumulative_excess_return",
        "worst_drawdown",
        "failed_gates",
        "review_recommendation",
        "personal_paper_candidate",
        "strategy_candidate",
    ]


def _optional_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _optional_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{float(value):.6f}" if pd.notna(value) else "nan")
    return view.to_markdown(index=False)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-run-dir", default=None)
    parser.add_argument("--plan-run-dir", default=None)
    parser.add_argument("--review-run-dir", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_frontier_personal_candidate_lifecycle_registry(
        bootstrap_run_dir=args.bootstrap_run_dir,
        plan_run_dir=args.plan_run_dir,
        review_run_dir=args.review_run_dir,
        output_dir=args.output_dir,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
