"""Summarize personal candidate-selection evidence without paper tracking."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.experiments.frontier_personal_candidate_gate import (
    FORMAL_PERSONAL_GATE_SCOPE,
    PERSONAL_BACKTEST_PROMOTION_LEVEL,
)
from traditional_quant_research.experiments.frontier_promotion_gate import latest_run_dir


DEFAULT_PERSONAL_PROTOCOL_GRID_ROOT = Path("traditional_quant_research/output/experiments/frontier_personal_protocol_grid")
DEFAULT_PERSONAL_GATE_ROOT = Path("traditional_quant_research/output/experiments/frontier_personal_candidate_gate")
DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/frontier_personal_candidate_selection_report")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/brain/references/research_log/2026-06-05_frontier_personal_candidate_selection_report.md")

SELECTION_CANDIDATE = "personal_backtest_candidate"
SELECTION_BACKTEST_ONLY = "personal_research/backtest_only"
SELECTION_DIAGNOSTIC = "diagnostic"


def run_frontier_personal_candidate_selection_report(
    *,
    personal_protocol_grid_run_dirs: Sequence[str | Path] | str | None = None,
    personal_gate_run_dirs: Sequence[str | Path] | str | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    """Write one candidate-selection report from existing formal gate artifacts."""

    protocol_dirs = _resolve_run_dirs(personal_protocol_grid_run_dirs, DEFAULT_PERSONAL_PROTOCOL_GRID_ROOT)
    gate_dirs = _resolve_run_dirs(personal_gate_run_dirs, DEFAULT_PERSONAL_GATE_ROOT)
    run_id = f"frontier_personal_candidate_selection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    evidence = read_selection_evidence(protocol_dirs=protocol_dirs, gate_dirs=gate_dirs)
    selection = build_candidate_selection_summary(evidence)
    rejected = build_rejected_diagnostic_summary(selection)
    summary = summarize_candidate_selection_report(
        selection,
        rejected,
        run_id=run_id,
        run_dir=run_dir,
        protocol_dirs=protocol_dirs,
        gate_dirs=gate_dirs,
    )
    markdown = render_candidate_selection_markdown(summary, selection, rejected)

    selection.to_csv(run_dir / "candidate_selection_summary.csv", index=False, encoding="utf-8-sig")
    rejected.to_csv(run_dir / "candidate_selection_rejected_diagnostics.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        path = Path(research_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    return {**summary, "run_dir": str(run_dir), "research_log": str(research_log_path) if write_research_log else None}


def read_selection_evidence(*, protocol_dirs: Sequence[Path], gate_dirs: Sequence[Path]) -> pd.DataFrame:
    """Read personal protocol-grid and standalone personal-gate rows."""

    frames: list[pd.DataFrame] = []
    for run_dir in protocol_dirs:
        path = run_dir / "personal_protocol_grid_ledger.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frames.append(_with_source(frame, source_type="personal_protocol_grid", run_dir=run_dir))
    for run_dir in gate_dirs:
        path = run_dir / "personal_candidate_gate_summary.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frames.append(_with_source(frame, source_type="personal_candidate_gate", run_dir=run_dir))
    if not frames:
        return pd.DataFrame(columns=_evidence_columns())
    return pd.concat(frames, ignore_index=True, sort=False)


def _with_source(frame: pd.DataFrame, *, source_type: str, run_dir: Path) -> pd.DataFrame:
    output = frame.copy()
    output["source_type"] = source_type
    output["source_run_dir"] = str(run_dir)
    leading = ["source_type", "source_run_dir"]
    return output[leading + [column for column in output.columns if column not in leading]]


def build_candidate_selection_summary(evidence: pd.DataFrame) -> pd.DataFrame:
    """Normalize gate/protocol evidence into current agent-side selection semantics."""

    columns = _selection_columns()
    if evidence.empty:
        return pd.DataFrame(columns=columns)
    work = evidence.copy()
    for column in [
        "top_n",
        "fee_bps",
        "capital_amount",
        "impact_bps_per_1pct",
        "exposure_penalty_strength",
        "mean_annualized_return",
        "min_annualized_return",
        "positive_year_rate",
        "worst_max_drawdown",
        "total_periods",
        "eval_year_count",
    ]:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    for column, default in {
        "constraint_variant": "baseline",
        "portfolio_constraint_mode": "penalty_top_n",
        "signal": "",
        "promotion_level": "",
        "evidence_scope": "",
        "failed_gates": "",
        "exposure_failures": "",
        "gate_profile_detail": "",
        "source_type": "",
        "source_run_dir": "",
    }.items():
        if column not in work.columns:
            work[column] = default
    if "formal_profile_gate" not in work.columns:
        work["formal_profile_gate"] = False
    work["formal_profile_gate"] = _truthy(work["formal_profile_gate"])

    rows: list[dict[str, Any]] = []
    for item in work.to_dict("records"):
        signal = _text_or_default(item.get("signal"), "")
        variant = _text_or_default(item.get("constraint_variant"), "baseline")
        portfolio_mode = _text_or_default(item.get("portfolio_constraint_mode"), "penalty_top_n")
        top_n = _optional_int(item.get("top_n")) or 0
        strength = _optional_float(item.get("exposure_penalty_strength")) or 0.0
        formal = bool(item.get("formal_profile_gate", False)) and str(item.get("evidence_scope", "")) == FORMAL_PERSONAL_GATE_SCOPE
        promoted = str(item.get("promotion_level", "")) == PERSONAL_BACKTEST_PROMOTION_LEVEL
        if promoted and formal:
            selection_status = SELECTION_CANDIDATE
        elif formal:
            selection_status = SELECTION_BACKTEST_ONLY
        else:
            selection_status = SELECTION_DIAGNOSTIC
        rows.append(
            {
                "candidate_id": _candidate_id(
                    signal=signal,
                    variant=variant,
                    portfolio_constraint_mode=portfolio_mode,
                    top_n=top_n,
                    strength=strength,
                ),
                "selection_status": selection_status,
                "source_type": str(item.get("source_type", "")),
                "source_run_dir": str(item.get("source_run_dir", "")),
                "constraint_variant": variant,
                "portfolio_constraint_mode": portfolio_mode,
                "top_n": top_n,
                "signal": signal,
                "fee_bps": _optional_float(item.get("fee_bps")),
                "capital_amount": _optional_float(item.get("capital_amount")),
                "impact_bps_per_1pct": _optional_float(item.get("impact_bps_per_1pct")),
                "exposure_penalty_strength": strength,
                "mean_annualized_return": _optional_float(item.get("mean_annualized_return")),
                "min_annualized_return": _optional_float(item.get("min_annualized_return")),
                "positive_year_rate": _optional_float(item.get("positive_year_rate")),
                "worst_max_drawdown": _optional_float(item.get("worst_max_drawdown")),
                "total_periods": _optional_int(item.get("total_periods")),
                "eval_year_count": _optional_int(item.get("eval_year_count")),
                "promotion_level": str(item.get("promotion_level", "")),
                "evidence_scope": str(item.get("evidence_scope", "")),
                "formal_profile_gate": formal,
                "failed_gates": str(item.get("failed_gates", "")),
                "exposure_failures": str(item.get("exposure_failures", "")),
                "post_selection_recommendation": "user_discretion" if selection_status == SELECTION_CANDIDATE else "none",
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    frame = frame.drop_duplicates(
        subset=[
            "candidate_id",
            "selection_status",
            "constraint_variant",
            "portfolio_constraint_mode",
            "top_n",
            "signal",
            "exposure_penalty_strength",
            "mean_annualized_return",
            "min_annualized_return",
            "positive_year_rate",
            "worst_max_drawdown",
            "total_periods",
            "eval_year_count",
            "promotion_level",
            "evidence_scope",
            "formal_profile_gate",
            "failed_gates",
        ],
        keep="first",
    )
    frame["_status_rank"] = frame["selection_status"].map(
        {SELECTION_CANDIDATE: 0, SELECTION_BACKTEST_ONLY: 1, SELECTION_DIAGNOSTIC: 2}
    ).fillna(9)
    frame = frame.sort_values(
        ["_status_rank", "mean_annualized_return", "min_annualized_return", "positive_year_rate"],
        ascending=[True, False, False, False],
        na_position="last",
    ).drop(columns=["_status_rank"])
    return frame.reset_index(drop=True)


def build_rejected_diagnostic_summary(selection: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "selection_status",
        "row_count",
        "top_failed_gates",
        "best_rejected_candidate_id",
        "best_rejected_mean_annualized_return",
    ]
    if selection.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    rejected = selection.loc[~selection["selection_status"].eq(SELECTION_CANDIDATE)].copy()
    for status, group in rejected.groupby("selection_status", sort=True):
        counter: Counter[str] = Counter()
        for gates in group["failed_gates"].dropna().astype(str):
            counter.update(part for part in gates.split(",") if part)
        best = group.sort_values("mean_annualized_return", ascending=False, na_position="last").iloc[0]
        rows.append(
            {
                "selection_status": str(status),
                "row_count": int(len(group)),
                "top_failed_gates": json.dumps(dict(counter.most_common()), ensure_ascii=False),
                "best_rejected_candidate_id": str(best.get("candidate_id", "")),
                "best_rejected_mean_annualized_return": _optional_float(best.get("mean_annualized_return")),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_candidate_selection_report(
    selection: pd.DataFrame,
    rejected: pd.DataFrame,
    *,
    run_id: str,
    run_dir: Path,
    protocol_dirs: Sequence[Path],
    gate_dirs: Sequence[Path],
) -> dict[str, Any]:
    candidates = selection.loc[selection["selection_status"].eq(SELECTION_CANDIDATE)] if not selection.empty else pd.DataFrame()
    backtest_only = selection.loc[selection["selection_status"].eq(SELECTION_BACKTEST_ONLY)] if not selection.empty else pd.DataFrame()
    diagnostic = selection.loc[selection["selection_status"].eq(SELECTION_DIAGNOSTIC)] if not selection.empty else pd.DataFrame()
    best = candidates.iloc[0].to_dict() if not candidates.empty else (selection.iloc[0].to_dict() if not selection.empty else {})
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "north_star": "Baostock-only personal quant strategy research",
        "objective": "summarize agent-side model and strategy candidate selections only",
        "protocol_run_dirs": [str(path) for path in protocol_dirs],
        "personal_gate_run_dirs": [str(path) for path in gate_dirs],
        "evaluated_rows": int(len(selection)),
        "personal_backtest_candidate_count": int(len(candidates)),
        "personal_research_backtest_only_count": int(len(backtest_only)),
        "diagnostic_count": int(len(diagnostic)),
        "strategy_candidate_count": 0,
        "personal_paper_candidate_count": 0,
        "personal_trading_candidate_count": 0,
        "best_candidate_id": str(best.get("candidate_id", "")),
        "best_selection_status": str(best.get("selection_status", "")),
        "best_signal": str(best.get("signal", "")),
        "best_constraint_variant": str(best.get("constraint_variant", "")),
        "best_portfolio_constraint_mode": str(best.get("portfolio_constraint_mode", "")),
        "best_top_n": _optional_int(best.get("top_n")),
        "best_mean_annualized_return": _optional_float(best.get("mean_annualized_return")),
        "rejected_summary_rows": int(len(rejected)),
        "decision": "personal_backtest_candidates_selected" if not candidates.empty else "keep_personal_research_backtest_only",
        "post_selection_boundary": "agent_selects_models_and_strategies_only; user_handles_risk_recording_and_live_decisions",
        "allowed_selection_statuses": [SELECTION_CANDIDATE, SELECTION_BACKTEST_ONLY, SELECTION_DIAGNOSTIC],
        "disallowed_outputs": ["personal_paper_candidate", "personal_trading_candidate", "paper_or_live_tracking_plan"],
        "output_dir": str(run_dir),
    }


def render_candidate_selection_markdown(
    summary: Mapping[str, Any],
    selection: pd.DataFrame,
    rejected: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# Frontier Personal Candidate Selection Report",
            "",
            f"- run_id: `{summary.get('run_id', '')}`",
            f"- decision: `{summary.get('decision', '')}`",
            f"- personal_backtest_candidate_count: `{summary.get('personal_backtest_candidate_count', 0)}`",
            f"- personal_research_backtest_only_count: `{summary.get('personal_research_backtest_only_count', 0)}`",
            f"- diagnostic_count: `{summary.get('diagnostic_count', 0)}`",
            f"- strategy_candidate_count: `{summary.get('strategy_candidate_count', 0)}`",
            f"- post_selection_boundary: `{summary.get('post_selection_boundary', '')}`",
            f"- best_candidate_id: `{summary.get('best_candidate_id', '')}`",
            "",
            "## Selection Summary",
            "",
            _markdown_table(selection),
            "",
            "## Rejected Diagnostics",
            "",
            _markdown_table(rejected),
            "",
            "## Interpretation",
            "",
            "This report only summarizes selected models and strategies. Downstream risk, recordkeeping, and execution decisions remain user discretion.",
            "",
        ]
    )


def _resolve_run_dirs(value: Sequence[str | Path] | str | None, default_root: Path) -> tuple[Path, ...]:
    if value is None:
        try:
            return (latest_run_dir(default_root),)
        except FileNotFoundError:
            return tuple()
    raw = value.split(",") if isinstance(value, str) else list(value)
    return tuple(Path(item) for item in raw if str(item).strip())


def _candidate_id(
    *,
    signal: str,
    variant: str,
    portfolio_constraint_mode: str,
    top_n: int,
    strength: float,
) -> str:
    return (
        f"{_safe_token(signal)}_"
        f"{_safe_token(variant)}_"
        f"{_safe_token(portfolio_constraint_mode)}_"
        f"top{int(top_n)}_"
        f"penalty_{_strength_token(strength)}"
    )


def _safe_token(value: Any) -> str:
    text = "".join(ch if ch.isalnum() else "_" for ch in str(value).lower()).strip("_")
    while "__" in text:
        text = text.replace("__", "_")
    return text or "na"


def _text_or_default(value: Any, default: str) -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return default
    return text


def _strength_token(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "na"
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "neg_").replace(".", "_")


def _optional_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _evidence_columns() -> list[str]:
    return ["source_type", "source_run_dir", *_selection_columns()]


def _selection_columns() -> list[str]:
    return [
        "candidate_id",
        "selection_status",
        "source_type",
        "source_run_dir",
        "constraint_variant",
        "portfolio_constraint_mode",
        "top_n",
        "signal",
        "fee_bps",
        "capital_amount",
        "impact_bps_per_1pct",
        "exposure_penalty_strength",
        "mean_annualized_return",
        "min_annualized_return",
        "positive_year_rate",
        "worst_max_drawdown",
        "total_periods",
        "eval_year_count",
        "promotion_level",
        "evidence_scope",
        "formal_profile_gate",
        "failed_gates",
        "exposure_failures",
        "post_selection_recommendation",
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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--personal-protocol-grid-run-dirs", default=None)
    parser.add_argument("--personal-gate-run-dirs", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_frontier_personal_candidate_selection_report(
        personal_protocol_grid_run_dirs=args.personal_protocol_grid_run_dirs,
        personal_gate_run_dirs=args.personal_gate_run_dirs,
        output_dir=args.output_dir,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
