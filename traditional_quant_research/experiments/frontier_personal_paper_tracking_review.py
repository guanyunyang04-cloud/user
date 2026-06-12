"""Review paper-tracking logs for personal frontier candidates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from traditional_quant_research.experiments.frontier_personal_candidate_gate import (
    PERSONAL_BACKTEST_PROMOTION_LEVEL,
    PERSONAL_BACKTEST_ONLY_LEVEL,
)
from traditional_quant_research.experiments.frontier_personal_paper_tracking_bootstrap import (
    DEFAULT_OUTPUT_DIR as DEFAULT_BOOTSTRAP_OUTPUT_ROOT,
    PERSONAL_PAPER_CANDIDATE_LEVEL,
)
from traditional_quant_research.experiments.frontier_promotion_gate import latest_run_dir


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/frontier_personal_paper_tracking_review")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/brain/references/research_log/2026-06-04_frontier_personal_paper_tracking_review.md")
PAPER_TRACKING_CANCELLED_MESSAGE = (
    "frontier personal paper tracking review is cancelled for traditional_quant_research: "
    "the agent's responsibility ends at selecting strong model and strategy candidates; "
    "post-selection risk, recordkeeping, paper/live tracking, and execution decisions are user discretion."
)


def run_frontier_personal_paper_tracking_review(
    *,
    bootstrap_run_dir: str | Path | None = None,
    tracking_log_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    """Review a filled paper-tracking log against the bootstrap protocol."""

    raise RuntimeError(PAPER_TRACKING_CANCELLED_MESSAGE)

    run_id = f"frontier_personal_paper_tracking_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    bootstrap_dir = Path(bootstrap_run_dir) if bootstrap_run_dir is not None else latest_run_dir(DEFAULT_BOOTSTRAP_OUTPUT_ROOT)
    candidates, protocol, bootstrap_summary = read_bootstrap_artifacts(bootstrap_dir)
    log_path = Path(tracking_log_path) if tracking_log_path is not None else bootstrap_dir / "paper_tracking_log_template.csv"
    tracking_log = read_tracking_log(log_path)
    review = evaluate_paper_tracking_log(candidates, protocol, tracking_log)
    summary = summarize_paper_tracking_review(
        review,
        run_id=run_id,
        bootstrap_run_dir=bootstrap_dir,
        tracking_log_path=log_path,
        bootstrap_summary=bootstrap_summary,
    )
    markdown = render_paper_tracking_review_markdown(summary, review)

    review.to_csv(run_dir / "paper_tracking_review_summary.csv", index=False, encoding="utf-8-sig")
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


def read_tracking_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"paper tracking log not found: {path}")
    return pd.read_csv(path)


def evaluate_paper_tracking_log(candidates: pd.DataFrame, protocol: pd.DataFrame, tracking_log: pd.DataFrame) -> pd.DataFrame:
    raise RuntimeError(PAPER_TRACKING_CANCELLED_MESSAGE)

    columns = _review_columns()
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    protocol_by_candidate = {
        str(row.get("candidate_id", "")): row
        for row in protocol.to_dict("records")
        if str(row.get("candidate_id", "")).strip()
    }
    log = _normalize_tracking_log(tracking_log)
    rows: list[dict[str, Any]] = []
    for candidate in candidates.to_dict("records"):
        candidate_id = str(candidate.get("candidate_id", ""))
        candidate_log = log.loc[log["candidate_id"].astype(str).eq(candidate_id)].copy() if not log.empty else pd.DataFrame()
        protocol_row = protocol_by_candidate.get(candidate_id, {})
        review_row = _evaluate_single_candidate(candidate, protocol_row, candidate_log)
        rows.append(review_row)
    return pd.DataFrame(rows, columns=columns)


def summarize_paper_tracking_review(
    review: pd.DataFrame,
    *,
    run_id: str,
    bootstrap_run_dir: Path,
    tracking_log_path: Path,
    bootstrap_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if review.empty:
        paper_count = 0
        downgrade_count = 0
        incomplete_count = 0
    else:
        paper_count = int(review["promotion_level"].astype(str).eq(PERSONAL_PAPER_CANDIDATE_LEVEL).sum())
        downgrade_count = int(review["recommendation"].astype(str).eq("downgrade_to_personal_research").sum())
        incomplete_count = int(review["recommendation"].astype(str).eq("continue_paper_tracking").sum())
    if paper_count:
        decision = "personal_paper_candidate_ready"
    elif downgrade_count:
        decision = "downgrade_or_research_review_required"
    elif incomplete_count:
        decision = "continue_paper_tracking"
    else:
        decision = "no_paper_tracking_candidate"
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "north_star": "Baostock-only personal quant strategy research",
        "bootstrap_run_dir": str(bootstrap_run_dir),
        "bootstrap_run_id": str(bootstrap_summary.get("run_id", "")),
        "tracking_log_path": str(tracking_log_path),
        "candidate_count": int(len(review)),
        "personal_paper_candidate_count": paper_count,
        "downgrade_count": downgrade_count,
        "incomplete_count": incomplete_count,
        "strategy_candidate_count": 0,
        "decision": decision,
        "limitations": [
            "This review reads paper-tracking logs and does not generate signals or rerun backtests.",
            "A personal paper candidate is still not a strategy candidate or institutional promotion.",
            "Missing or incomplete paper logs keep the candidate below personal_paper_candidate.",
        ],
    }


def render_paper_tracking_review_markdown(summary: Mapping[str, Any], review: pd.DataFrame) -> str:
    lines = [
        "# Frontier Personal Paper Tracking Review",
        "",
        f"- run_id: `{summary.get('run_id', '')}`",
        f"- decision: `{summary.get('decision', '')}`",
        f"- personal_paper_candidate_count: `{summary.get('personal_paper_candidate_count', 0)}`",
        f"- downgrade_count: `{summary.get('downgrade_count', 0)}`",
        f"- incomplete_count: `{summary.get('incomplete_count', 0)}`",
        f"- strategy_candidate_count: `{summary.get('strategy_candidate_count', 0)}`",
        f"- bootstrap_run_id: `{summary.get('bootstrap_run_id', '')}`",
        f"- tracking_log_path: `{summary.get('tracking_log_path', '')}`",
        "",
        "## Review Summary",
        "",
        _markdown_table(review),
        "",
        "## Interpretation",
        "",
        "This review only upgrades a row to `personal_paper_candidate` after complete paper evidence passes the bootstrap rules. "
        "It deliberately keeps `strategy_candidate_count=0` and treats incomplete tracking as continued paper tracking rather than success.",
        "",
    ]
    return "\n".join(lines)


def _evaluate_single_candidate(candidate: Mapping[str, Any], protocol: Mapping[str, Any], log: pd.DataFrame) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id", ""))
    completed = _completed_tracking_rows(log)
    completed_count = int(len(completed))
    first_date, last_date, tracking_days = _tracking_window(completed)
    min_periods = _int_from(protocol.get("min_tracking_periods"), default=6)
    min_days = _int_from(protocol.get("min_tracking_days"), default=120)
    max_drawdown_limit = _float_from(protocol.get("max_paper_drawdown"), default=-0.20)
    max_single_period_loss = _float_from(protocol.get("max_single_period_loss"), default=-0.12)
    min_excess_return = _float_from(protocol.get("min_paper_excess_return"), default=-0.02)

    net_returns = pd.to_numeric(completed.get("net_return", pd.Series(dtype=float)), errors="coerce")
    paper_returns = pd.to_numeric(completed.get("paper_account_return", pd.Series(dtype=float)), errors="coerce")
    excess_returns = pd.to_numeric(completed.get("excess_return", pd.Series(dtype=float)), errors="coerce")
    max_drawdown_to_date = pd.to_numeric(completed.get("max_drawdown_to_date", pd.Series(dtype=float)), errors="coerce")

    cumulative_net_return = _compound_return(net_returns)
    cumulative_paper_return = _compound_return(paper_returns)
    cumulative_excess_return = _compound_return(excess_returns)
    worst_period_net_return = _safe_min(net_returns)
    worst_period_paper_return = _safe_min(paper_returns)
    worst_period_return = _safe_min(pd.concat([net_returns, paper_returns], ignore_index=True))
    worst_drawdown = _safe_min(max_drawdown_to_date)
    if np.isnan(worst_drawdown) and not paper_returns.dropna().empty:
        worst_drawdown = _max_drawdown_from_returns(paper_returns)

    missing_required_fields = _missing_required_fields(completed)
    execution_integrity_gate = not missing_required_fields
    sample_gate = completed_count >= min_periods and tracking_days >= min_days
    paper_return_gate = bool(not np.isnan(cumulative_excess_return) and cumulative_excess_return >= min_excess_return)
    drawdown_gate = bool(not np.isnan(worst_drawdown) and worst_drawdown >= max_drawdown_limit)
    single_period_gate = bool(not np.isnan(worst_period_return) and worst_period_return >= max_single_period_loss)
    gates = {
        "sample_gate": sample_gate,
        "execution_integrity_gate": execution_integrity_gate,
        "paper_return_gate": paper_return_gate,
        "drawdown_gate": drawdown_gate,
        "single_period_gate": single_period_gate,
    }
    failed = [name for name, passed in gates.items() if not passed]
    damage_failures = {"drawdown_gate", "single_period_gate"}
    if not completed_count:
        recommendation = "continue_paper_tracking"
        promotion_level = PERSONAL_BACKTEST_PROMOTION_LEVEL
    elif any(name in failed for name in damage_failures) or ("execution_integrity_gate" in failed and sample_gate):
        recommendation = "downgrade_to_personal_research"
        promotion_level = PERSONAL_BACKTEST_ONLY_LEVEL
    elif failed:
        recommendation = "continue_paper_tracking"
        promotion_level = PERSONAL_BACKTEST_PROMOTION_LEVEL
    else:
        recommendation = "promote_personal_paper_candidate"
        promotion_level = PERSONAL_PAPER_CANDIDATE_LEVEL

    return {
        "candidate_id": candidate_id,
        "current_level": str(candidate.get("current_level", PERSONAL_BACKTEST_PROMOTION_LEVEL)),
        "promotion_level": promotion_level,
        "recommendation": recommendation,
        "completed_periods": completed_count,
        "tracking_days": tracking_days,
        "first_record_date": first_date,
        "last_record_date": last_date,
        "min_tracking_periods": min_periods,
        "min_tracking_days": min_days,
        "cumulative_net_return": cumulative_net_return,
        "cumulative_paper_return": cumulative_paper_return,
        "cumulative_excess_return": cumulative_excess_return,
        "worst_period_net_return": worst_period_net_return,
        "worst_period_paper_return": worst_period_paper_return,
        "worst_drawdown": worst_drawdown,
        "sample_gate": sample_gate,
        "execution_integrity_gate": execution_integrity_gate,
        "paper_return_gate": paper_return_gate,
        "drawdown_gate": drawdown_gate,
        "single_period_gate": single_period_gate,
        "failed_gates": ",".join(failed),
        "missing_required_fields": ",".join(missing_required_fields),
        "strategy_candidate": False,
    }


def _normalize_tracking_log(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    output = frame.copy()
    if "candidate_id" not in output.columns:
        output["candidate_id"] = ""
    for column in [
        "gross_return",
        "net_return",
        "paper_account_return",
        "benchmark_return",
        "excess_return",
        "realized_turnover",
        "realized_fee_bps",
        "max_drawdown_to_date",
        "selected_count",
        "blocked_entry_count",
        "entry_limit_up_count",
        "exit_delayed_count",
        "exit_limit_down_count",
    ]:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _completed_tracking_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    required = ["record_date", "net_return", "paper_account_return", "excess_return"]
    missing_cols = [column for column in required if column not in frame.columns]
    if missing_cols:
        return frame.iloc[0:0].copy()
    output = frame.copy()
    output["record_date"] = pd.to_datetime(output["record_date"], errors="coerce")
    mask = output["record_date"].notna()
    for column in ["net_return", "paper_account_return", "excess_return"]:
        mask &= pd.to_numeric(output[column], errors="coerce").notna()
    return output.loc[mask].sort_values("record_date").reset_index(drop=True)


def _tracking_window(frame: pd.DataFrame) -> tuple[str, str, int]:
    if frame.empty or "record_date" not in frame.columns:
        return "", "", 0
    dates = pd.to_datetime(frame["record_date"], errors="coerce").dropna()
    if dates.empty:
        return "", "", 0
    first = dates.min()
    last = dates.max()
    return first.date().isoformat(), last.date().isoformat(), int((last - first).days) + 1


def _missing_required_fields(frame: pd.DataFrame) -> list[str]:
    required = [
        "selected_count",
        "blocked_entry_count",
        "entry_limit_up_count",
        "exit_delayed_count",
        "exit_limit_down_count",
        "net_return",
        "paper_account_return",
        "benchmark_return",
        "excess_return",
        "realized_turnover",
        "realized_fee_bps",
        "max_drawdown_to_date",
    ]
    if frame.empty:
        return required
    missing: list[str] = []
    for column in required:
        if column not in frame.columns or frame[column].isna().any():
            missing.append(column)
    return missing


def _compound_return(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float((1.0 + clean).prod() - 1.0)


def _max_drawdown_from_returns(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    equity = (1.0 + clean).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def _safe_min(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.min()) if not clean.empty else np.nan


def _int_from(value: Any, *, default: int) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _float_from(value: Any, *, default: float) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _review_columns() -> list[str]:
    return [
        "candidate_id",
        "current_level",
        "promotion_level",
        "recommendation",
        "completed_periods",
        "tracking_days",
        "first_record_date",
        "last_record_date",
        "min_tracking_periods",
        "min_tracking_days",
        "cumulative_net_return",
        "cumulative_paper_return",
        "cumulative_excess_return",
        "worst_period_net_return",
        "worst_period_paper_return",
        "worst_drawdown",
        "sample_gate",
        "execution_integrity_gate",
        "paper_return_gate",
        "drawdown_gate",
        "single_period_gate",
        "failed_gates",
        "missing_required_fields",
        "strategy_candidate",
    ]


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
    parser.add_argument("--bootstrap-run-dir", default=None)
    parser.add_argument("--tracking-log-path", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_frontier_personal_paper_tracking_review(
        bootstrap_run_dir=args.bootstrap_run_dir,
        tracking_log_path=args.tracking_log_path,
        output_dir=args.output_dir,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
