"""Bootstrap paper-tracking artifacts from personal frontier candidate evidence."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from traditional_quant_research.experiments.frontier_personal_candidate_gate import (
    PERSONAL_BACKTEST_PROMOTION_LEVEL,
)
from traditional_quant_research.experiments.frontier_promotion_gate import latest_run_dir


DEFAULT_PERSONAL_GATE_OUTPUT_ROOT = Path("traditional_quant_research/output/experiments/frontier_personal_candidate_gate")
DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/frontier_personal_paper_tracking_bootstrap")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-04_frontier_personal_paper_tracking_bootstrap.md")

PERSONAL_PAPER_CANDIDATE_LEVEL = "personal_paper_candidate"
BOOTSTRAPPED_TRACKING_STATUS = "paper_tracking_bootstrapped"

DEFAULT_MIN_TRACKING_PERIODS = 6
DEFAULT_MIN_TRACKING_DAYS = 120
DEFAULT_MAX_PAPER_DRAWDOWN = -0.20
DEFAULT_MAX_SINGLE_PERIOD_LOSS = -0.12
DEFAULT_MIN_PAPER_EXCESS_RETURN = -0.02
PAPER_TRACKING_CANCELLED_MESSAGE = (
    "frontier personal paper tracking is cancelled for traditional_quant_research: "
    "the agent's responsibility ends at selecting strong model and strategy candidates; "
    "post-selection risk, recordkeeping, paper/live tracking, and execution decisions are user discretion."
)


def run_frontier_personal_paper_tracking_bootstrap(
    *,
    personal_gate_run_dir: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    min_tracking_periods: int = DEFAULT_MIN_TRACKING_PERIODS,
    min_tracking_days: int = DEFAULT_MIN_TRACKING_DAYS,
    max_paper_drawdown: float = DEFAULT_MAX_PAPER_DRAWDOWN,
    max_single_period_loss: float = DEFAULT_MAX_SINGLE_PERIOD_LOSS,
    min_paper_excess_return: float = DEFAULT_MIN_PAPER_EXCESS_RETURN,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    """Create a reproducible paper-tracking package for personal candidates."""

    raise RuntimeError(PAPER_TRACKING_CANCELLED_MESSAGE)

    if min_tracking_periods <= 0:
        raise ValueError("min_tracking_periods must be positive")
    if min_tracking_days <= 0:
        raise ValueError("min_tracking_days must be positive")
    if max_paper_drawdown >= 0:
        raise ValueError("max_paper_drawdown should be a negative return threshold")
    if max_single_period_loss >= 0:
        raise ValueError("max_single_period_loss should be a negative return threshold")

    run_id = f"frontier_personal_paper_tracking_bootstrap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    gate_dir = Path(personal_gate_run_dir) if personal_gate_run_dir is not None else latest_run_dir(DEFAULT_PERSONAL_GATE_OUTPUT_ROOT)
    gate, gate_summary = read_personal_gate_evidence(gate_dir)
    combined_summary = read_combined_summary_from_gate(gate_summary)
    candidates = build_paper_tracking_candidates(gate, gate_summary, combined_summary)
    protocol = build_paper_tracking_protocol(
        candidates,
        combined_summary,
        min_tracking_periods=min_tracking_periods,
        min_tracking_days=min_tracking_days,
        max_paper_drawdown=max_paper_drawdown,
        max_single_period_loss=max_single_period_loss,
        min_paper_excess_return=min_paper_excess_return,
    )
    log_template = build_paper_tracking_log_template(candidates, combined_summary)
    review_rules = build_paper_tracking_review_rules(
        min_tracking_periods=min_tracking_periods,
        min_tracking_days=min_tracking_days,
        max_paper_drawdown=max_paper_drawdown,
        max_single_period_loss=max_single_period_loss,
        min_paper_excess_return=min_paper_excess_return,
    )
    summary = summarize_paper_tracking_bootstrap(
        candidates,
        protocol,
        review_rules,
        run_id=run_id,
        personal_gate_run_dir=gate_dir,
        gate_summary=gate_summary,
        combined_summary=combined_summary,
        min_tracking_periods=min_tracking_periods,
        min_tracking_days=min_tracking_days,
        max_paper_drawdown=max_paper_drawdown,
        max_single_period_loss=max_single_period_loss,
        min_paper_excess_return=min_paper_excess_return,
    )
    markdown = render_paper_tracking_bootstrap_markdown(summary, candidates, protocol, review_rules)

    candidates.to_csv(run_dir / "paper_tracking_candidates.csv", index=False, encoding="utf-8-sig")
    protocol.to_csv(run_dir / "paper_tracking_protocol.csv", index=False, encoding="utf-8-sig")
    log_template.to_csv(run_dir / "paper_tracking_log_template.csv", index=False, encoding="utf-8-sig")
    review_rules.to_csv(run_dir / "paper_tracking_review_rules.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        path = Path(research_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    return {**summary, "run_dir": str(run_dir), "research_log": str(research_log_path) if write_research_log else None}


def read_personal_gate_evidence(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    gate_path = run_dir / "personal_candidate_gate_summary.csv"
    summary_path = run_dir / "summary.json"
    if not gate_path.exists():
        raise FileNotFoundError(f"personal candidate gate summary not found: {gate_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"personal candidate gate run summary not found: {summary_path}")
    return pd.read_csv(gate_path), json.loads(summary_path.read_text(encoding="utf-8-sig"))


def read_combined_summary_from_gate(gate_summary: Mapping[str, Any]) -> dict[str, Any]:
    combined_dir = str(gate_summary.get("combined_run_dir", "")).strip()
    if not combined_dir:
        return {}
    path = Path(combined_dir) / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_paper_tracking_candidates(
    gate: pd.DataFrame,
    gate_summary: Mapping[str, Any],
    combined_summary: Mapping[str, Any],
) -> pd.DataFrame:
    raise RuntimeError(PAPER_TRACKING_CANCELLED_MESSAGE)

    columns = _candidate_columns()
    if gate.empty:
        return pd.DataFrame(columns=columns)
    work = gate.copy()
    if "promotion_level" not in work.columns:
        return pd.DataFrame(columns=columns)
    if "paper_tracking_recommendation" not in work.columns:
        work["paper_tracking_recommendation"] = ""
    if "promoted" not in work.columns:
        work["promoted"] = False
    candidates = work.loc[
        work["promotion_level"].astype(str).eq(PERSONAL_BACKTEST_PROMOTION_LEVEL)
        & work["paper_tracking_recommendation"].astype(str).eq("start_paper_tracking")
        & work["promoted"].map(_truthy)
    ].copy()
    if candidates.empty:
        return pd.DataFrame(columns=columns)

    for column in [
        "fee_bps",
        "capital_amount",
        "impact_bps_per_1pct",
        "personal_capital_amount",
        "exposure_penalty_strength",
        "mean_annualized_return",
        "min_annualized_return",
        "positive_year_rate",
        "worst_max_drawdown",
        "total_periods",
        "eval_year_count",
        "max_proxy_mean_abs_active_exposure",
        "top_n",
    ]:
        if column in candidates.columns:
            candidates[column] = pd.to_numeric(candidates[column], errors="coerce")

    rows: list[dict[str, Any]] = []
    for row in candidates.to_dict("records"):
        signal = str(row.get("signal", ""))
        variant = str(row.get("constraint_variant", "baseline") or "baseline")
        strength = float(row.get("exposure_penalty_strength", 0.0) or 0.0)
        top_n = _optional_int(row.get("top_n", combined_summary.get("top_n")))
        candidate_id = _candidate_id(signal=signal, constraint_variant=variant, exposure_penalty_strength=strength, top_n=top_n)
        rows.append(
            {
                "candidate_id": candidate_id,
                "tracking_status": BOOTSTRAPPED_TRACKING_STATUS,
                "current_level": PERSONAL_BACKTEST_PROMOTION_LEVEL,
                "target_next_level": PERSONAL_PAPER_CANDIDATE_LEVEL,
                "paper_tracking_recommendation": "start_paper_tracking",
                "north_star": str(gate_summary.get("north_star", "Baostock-only personal quant strategy research")),
                "snapshot_id": str(gate_summary.get("snapshot_id", combined_summary.get("snapshot_id", ""))),
                "required_year_window": str(gate_summary.get("required_year_window", "")),
                "signal": signal,
                "constraint_variant": variant,
                "exposure_penalty_strength": strength,
                "fee_bps": float(row.get("fee_bps", np.nan)),
                "impact_bps_per_1pct": float(row.get("impact_bps_per_1pct", np.nan)),
                "backtest_capital_stress_amount": float(row.get("capital_amount", np.nan)),
                "personal_capital_amount": float(row.get("personal_capital_amount", np.nan)),
                "horizon": _optional_int(combined_summary.get("horizon")),
                "rebalance_frequency": str(combined_summary.get("rebalance_frequency", "")),
                "top_n": top_n,
                "buffer_multiplier": _optional_float(combined_summary.get("buffer_multiplier")),
                "mean_annualized_return": float(row.get("mean_annualized_return", np.nan)),
                "min_annualized_return": float(row.get("min_annualized_return", np.nan)),
                "positive_year_rate": float(row.get("positive_year_rate", np.nan)),
                "worst_max_drawdown": float(row.get("worst_max_drawdown", np.nan)),
                "total_periods": _optional_int(row.get("total_periods")),
                "eval_year_count": _optional_int(row.get("eval_year_count")),
                "max_proxy_mean_abs_active_exposure": float(row.get("max_proxy_mean_abs_active_exposure", np.nan)),
                "paper_tracking_started": False,
                "personal_paper_candidate": False,
                "strategy_candidate": False,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_paper_tracking_protocol(
    candidates: pd.DataFrame,
    combined_summary: Mapping[str, Any],
    *,
    min_tracking_periods: int,
    min_tracking_days: int,
    max_paper_drawdown: float,
    max_single_period_loss: float,
    min_paper_excess_return: float,
) -> pd.DataFrame:
    raise RuntimeError(PAPER_TRACKING_CANCELLED_MESSAGE)

    columns = _protocol_columns()
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for row in candidates.to_dict("records"):
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "tracking_status": row["tracking_status"],
                "current_level": row["current_level"],
                "target_next_level": row["target_next_level"],
                "data_source": "baostock_only",
                "signal_generation_source": "combined_constraint_artifacts",
                "execution_mode": "paper_tracking_only",
                "rebalance_frequency": row.get("rebalance_frequency", combined_summary.get("rebalance_frequency", "")),
                "horizon": row.get("horizon", combined_summary.get("horizon")),
                "top_n": row.get("top_n", combined_summary.get("top_n")),
                "buffer_multiplier": row.get("buffer_multiplier", combined_summary.get("buffer_multiplier")),
                "fee_bps": row.get("fee_bps"),
                "impact_bps_per_1pct": row.get("impact_bps_per_1pct"),
                "personal_capital_amount": row.get("personal_capital_amount"),
                "min_tracking_periods": int(min_tracking_periods),
                "min_tracking_days": int(min_tracking_days),
                "max_paper_drawdown": float(max_paper_drawdown),
                "max_single_period_loss": float(max_single_period_loss),
                "min_paper_excess_return": float(min_paper_excess_return),
                "promotion_rule": "promote only after complete paper evidence meets all review rules",
                "downgrade_rule": "downgrade to personal_research/backtest_only on execution drift or drawdown breach",
                "true_size_required": False,
                "institutional_promotion_allowed": False,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_paper_tracking_log_template(candidates: pd.DataFrame, combined_summary: Mapping[str, Any]) -> pd.DataFrame:
    raise RuntimeError(PAPER_TRACKING_CANCELLED_MESSAGE)

    columns = _log_template_columns()
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for row in candidates.to_dict("records"):
        rows.append(
            {
                "record_date": "",
                "candidate_id": row["candidate_id"],
                "signal": row["signal"],
                "constraint_variant": row["constraint_variant"],
                "signal_date": "",
                "rebalance_date": "",
                "planned_entry_date": "",
                "planned_exit_date": "",
                "target_top_n": row.get("top_n", combined_summary.get("top_n", "")),
                "selected_count": "",
                "blocked_entry_count": "",
                "entry_limit_up_count": "",
                "exit_delayed_count": "",
                "exit_limit_down_count": "",
                "gross_return": "",
                "net_return": "",
                "paper_account_return": "",
                "benchmark_return": "",
                "excess_return": "",
                "realized_turnover": "",
                "realized_fee_bps": "",
                "realized_impact_note": "",
                "max_drawdown_to_date": "",
                "execution_notes": "",
                "decision_notes": "",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_paper_tracking_review_rules(
    *,
    min_tracking_periods: int,
    min_tracking_days: int,
    max_paper_drawdown: float,
    max_single_period_loss: float,
    min_paper_excess_return: float,
) -> pd.DataFrame:
    raise RuntimeError(PAPER_TRACKING_CANCELLED_MESSAGE)

    rows = [
        {
            "dimension": "minimum_tracking_sample",
            "pass_condition": f">={min_tracking_periods} completed rebalance records and >={min_tracking_days} calendar days",
            "remain_or_downgrade_condition": "remain personal_backtest_candidate until the paper sample is complete",
        },
        {
            "dimension": "execution_integrity",
            "pass_condition": "record selected_count, blocked entries, limit-up/down delays, turnover, fees, and notes for every period",
            "remain_or_downgrade_condition": "downgrade if execution records are missing or materially inconsistent with the backtest protocol",
        },
        {
            "dimension": "paper_return_sanity",
            "pass_condition": f"paper excess return >= {min_paper_excess_return:.4f} after costs over the completed tracking window",
            "remain_or_downgrade_condition": "remain research-only if paper return is too weak even when execution is clean",
        },
        {
            "dimension": "drawdown_control",
            "pass_condition": f"max paper drawdown stays above {max_paper_drawdown:.4f}",
            "remain_or_downgrade_condition": "downgrade if drawdown breaches the paper limit",
        },
        {
            "dimension": "single_period_damage",
            "pass_condition": f"no single completed tracking period is below {max_single_period_loss:.4f}",
            "remain_or_downgrade_condition": "trigger review if a single period breaches the damage threshold",
        },
        {
            "dimension": "promotion_boundary",
            "pass_condition": "only personal_paper_candidate is allowed after paper evidence passes",
            "remain_or_downgrade_condition": "never call the row strategy_candidate without true-size and institutional gates",
        },
    ]
    return pd.DataFrame(rows, columns=["dimension", "pass_condition", "remain_or_downgrade_condition"])


def summarize_paper_tracking_bootstrap(
    candidates: pd.DataFrame,
    protocol: pd.DataFrame,
    review_rules: pd.DataFrame,
    *,
    run_id: str,
    personal_gate_run_dir: Path,
    gate_summary: Mapping[str, Any],
    combined_summary: Mapping[str, Any],
    min_tracking_periods: int,
    min_tracking_days: int,
    max_paper_drawdown: float,
    max_single_period_loss: float,
    min_paper_excess_return: float,
) -> dict[str, Any]:
    candidate_count = int(len(candidates))
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "north_star": "Baostock-only personal quant strategy research",
        "personal_gate_run_dir": str(personal_gate_run_dir),
        "personal_gate_run_id": str(gate_summary.get("run_id", "")),
        "combined_run_dir": str(gate_summary.get("combined_run_dir", "")),
        "combined_run_id": str(combined_summary.get("run_id", "")),
        "snapshot_id": str(gate_summary.get("snapshot_id", combined_summary.get("snapshot_id", ""))),
        "required_year_window": str(gate_summary.get("required_year_window", "")),
        "candidate_count": candidate_count,
        "personal_backtest_candidate_count": candidate_count,
        "personal_paper_candidate_count": 0,
        "strategy_candidate_count": 0,
        "decision": "paper_tracking_bootstrap_ready" if candidate_count else "no_personal_backtest_candidate_to_track",
        "tracking_status": BOOTSTRAPPED_TRACKING_STATUS if candidate_count else "not_started",
        "min_tracking_periods": int(min_tracking_periods),
        "min_tracking_days": int(min_tracking_days),
        "max_paper_drawdown": float(max_paper_drawdown),
        "max_single_period_loss": float(max_single_period_loss),
        "min_paper_excess_return": float(min_paper_excess_return),
        "candidate_ids": candidates["candidate_id"].astype(str).tolist() if not candidates.empty else [],
        "protocol_rows": int(len(protocol)),
        "review_rule_count": int(len(review_rules)),
        "limitations": [
            "This bootstrap reads existing personal gate evidence and does not rerun a backtest.",
            "It creates paper-tracking artifacts but does not upgrade any row to personal_paper_candidate.",
            "Paper tracking is personal research evidence, not institutional strategy promotion.",
        ],
    }


def render_paper_tracking_bootstrap_markdown(
    summary: Mapping[str, Any],
    candidates: pd.DataFrame,
    protocol: pd.DataFrame,
    review_rules: pd.DataFrame,
) -> str:
    lines = [
        "# Frontier Personal Paper Tracking Bootstrap",
        "",
        f"- run_id: `{summary.get('run_id', '')}`",
        f"- decision: `{summary.get('decision', '')}`",
        f"- north_star: `{summary.get('north_star', '')}`",
        f"- personal_backtest_candidate_count: `{summary.get('personal_backtest_candidate_count', 0)}`",
        f"- personal_paper_candidate_count: `{summary.get('personal_paper_candidate_count', 0)}`",
        f"- strategy_candidate_count: `{summary.get('strategy_candidate_count', 0)}`",
        f"- tracking_status: `{summary.get('tracking_status', '')}`",
        f"- personal_gate_run_id: `{summary.get('personal_gate_run_id', '')}`",
        f"- combined_run_id: `{summary.get('combined_run_id', '')}`",
        "",
        "## Candidates",
        "",
        _markdown_table(candidates),
        "",
        "## Protocol",
        "",
        _markdown_table(protocol),
        "",
        "## Review Rules",
        "",
        _markdown_table(review_rules),
        "",
        "## Interpretation",
        "",
        "This package is the first operational step after a personal backtest candidate passes. "
        "It defines what to track, how to record execution, and what evidence is required before any future "
        "`personal_paper_candidate` decision. It deliberately keeps `strategy_candidate_count=0`.",
        "",
    ]
    return "\n".join(lines)


def _candidate_columns() -> list[str]:
    return [
        "candidate_id",
        "tracking_status",
        "current_level",
        "target_next_level",
        "paper_tracking_recommendation",
        "north_star",
        "snapshot_id",
        "required_year_window",
        "signal",
        "constraint_variant",
        "exposure_penalty_strength",
        "fee_bps",
        "impact_bps_per_1pct",
        "backtest_capital_stress_amount",
        "personal_capital_amount",
        "horizon",
        "rebalance_frequency",
        "top_n",
        "buffer_multiplier",
        "mean_annualized_return",
        "min_annualized_return",
        "positive_year_rate",
        "worst_max_drawdown",
        "total_periods",
        "eval_year_count",
        "max_proxy_mean_abs_active_exposure",
        "paper_tracking_started",
        "personal_paper_candidate",
        "strategy_candidate",
    ]


def _protocol_columns() -> list[str]:
    return [
        "candidate_id",
        "tracking_status",
        "current_level",
        "target_next_level",
        "data_source",
        "signal_generation_source",
        "execution_mode",
        "rebalance_frequency",
        "horizon",
        "top_n",
        "buffer_multiplier",
        "fee_bps",
        "impact_bps_per_1pct",
        "personal_capital_amount",
        "min_tracking_periods",
        "min_tracking_days",
        "max_paper_drawdown",
        "max_single_period_loss",
        "min_paper_excess_return",
        "promotion_rule",
        "downgrade_rule",
        "true_size_required",
        "institutional_promotion_allowed",
    ]


def _log_template_columns() -> list[str]:
    return [
        "record_date",
        "candidate_id",
        "signal",
        "constraint_variant",
        "signal_date",
        "rebalance_date",
        "planned_entry_date",
        "planned_exit_date",
        "target_top_n",
        "selected_count",
        "blocked_entry_count",
        "entry_limit_up_count",
        "exit_delayed_count",
        "exit_limit_down_count",
        "gross_return",
        "net_return",
        "paper_account_return",
        "benchmark_return",
        "excess_return",
        "realized_turnover",
        "realized_fee_bps",
        "realized_impact_note",
        "max_drawdown_to_date",
        "execution_notes",
        "decision_notes",
    ]


def _candidate_id(*, signal: str, constraint_variant: str, exposure_penalty_strength: float, top_n: int | None = None) -> str:
    top_n_part = f"__top{top_n}" if top_n is not None else ""
    raw = f"{signal}__{constraint_variant}__penalty_{exposure_penalty_strength:g}{top_n_part}"
    return _safe_token(raw)


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    token = re.sub(r"_+", "_", token).strip("_").lower()
    return token or "candidate"


def _optional_float(value: Any) -> float:
    try:
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


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            if pd.isna(value):
                return False
        except TypeError:
            pass
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "passed"}


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(_fmt)
    return view.to_markdown(index=False)


def _fmt(value: Any) -> str:
    if value is None:
        return "nan"
    try:
        if pd.isna(value):
            return "nan"
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


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
    parser.add_argument("--personal-gate-run-dir", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-tracking-periods", type=int, default=DEFAULT_MIN_TRACKING_PERIODS)
    parser.add_argument("--min-tracking-days", type=int, default=DEFAULT_MIN_TRACKING_DAYS)
    parser.add_argument("--max-paper-drawdown", type=float, default=DEFAULT_MAX_PAPER_DRAWDOWN)
    parser.add_argument("--max-single-period-loss", type=float, default=DEFAULT_MAX_SINGLE_PERIOD_LOSS)
    parser.add_argument("--min-paper-excess-return", type=float, default=DEFAULT_MIN_PAPER_EXCESS_RETURN)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_frontier_personal_paper_tracking_bootstrap(
        personal_gate_run_dir=args.personal_gate_run_dir,
        output_dir=args.output_dir,
        min_tracking_periods=args.min_tracking_periods,
        min_tracking_days=args.min_tracking_days,
        max_paper_drawdown=args.max_paper_drawdown,
        max_single_period_loss=args.max_single_period_loss,
        min_paper_excess_return=args.min_paper_excess_return,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
