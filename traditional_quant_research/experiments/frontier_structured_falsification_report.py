"""Structured falsification report for frontier promotion evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from traditional_quant_research.experiments.frontier_promotion_gate import latest_run_dir


DEFAULT_PROMOTION_OUTPUT_ROOT = Path("traditional_quant_research/output/experiments/frontier_promotion_gate")
DEFAULT_TRIAL_LEDGER_ROOT = Path("traditional_quant_research/output/experiments/frontier_trial_ledger")
DEFAULT_SIZE_AUDIT_ROOT = Path("traditional_quant_research/output/experiments/v2_daily_size_audit")
DEFAULT_FAILURE_ATTRIBUTION_ROOT = Path("traditional_quant_research/output/experiments/frontier_failure_attribution")
DEFAULT_WEAK_YEAR_REBUILD_ROOT = Path("traditional_quant_research/output/experiments/frontier_weak_year_rebuild")
DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/frontier_structured_falsification_report")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-03_frontier_structured_falsification_report.md")


def run_frontier_structured_falsification_report(
    *,
    promotion_run_dir: str | Path | None = None,
    trial_ledger_run_dir: str | Path | None = None,
    size_audit_run_dir: str | Path | None = None,
    failure_run_dir: str | Path | None = None,
    weak_year_rebuild_run_dir: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    """Write one promotion-or-falsification decision artifact from existing evidence."""

    promotion_dir = Path(promotion_run_dir) if promotion_run_dir is not None else latest_run_dir(DEFAULT_PROMOTION_OUTPUT_ROOT)
    ledger_dir = Path(trial_ledger_run_dir) if trial_ledger_run_dir is not None else latest_run_dir(DEFAULT_TRIAL_LEDGER_ROOT)
    size_dir = Path(size_audit_run_dir) if size_audit_run_dir is not None else latest_run_dir(DEFAULT_SIZE_AUDIT_ROOT)
    failure_dir = Path(failure_run_dir) if failure_run_dir is not None else latest_run_dir(DEFAULT_FAILURE_ATTRIBUTION_ROOT)
    weak_dir = (
        Path(weak_year_rebuild_run_dir)
        if weak_year_rebuild_run_dir is not None
        else latest_run_dir(DEFAULT_WEAK_YEAR_REBUILD_ROOT)
    )

    run_id = f"frontier_structured_falsification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    promotion_summary = _read_json(promotion_dir / "summary.json")
    size_summary = _read_json(size_dir / "summary.json")
    ledger_summary = _read_json(ledger_dir / "summary.json")
    failure_summary = _read_json(failure_dir / "summary.json")
    weak_summary = _read_json(weak_dir / "summary.json")
    promotion_gate = pd.read_csv(promotion_dir / "promotion_gate_summary.csv")
    selection_bias = _read_optional_csv(ledger_dir / "selection_bias_report.csv")
    maturity = _read_optional_csv(ledger_dir / "research_maturity_report.csv")
    weak_backlog = _read_optional_csv(weak_dir / "rebuild_backlog.csv")

    gate_status = build_gate_status(promotion_gate)
    failure_matrix = build_failure_matrix(
        promotion_gate,
        size_summary=size_summary,
        failure_summary=failure_summary,
        weak_summary=weak_summary,
    )
    next_actions = build_next_minimum_actions(failure_matrix, weak_backlog)
    evidence_manifest = build_evidence_manifest(
        promotion_dir=promotion_dir,
        ledger_dir=ledger_dir,
        size_dir=size_dir,
        failure_dir=failure_dir,
        weak_dir=weak_dir,
        promotion_summary=promotion_summary,
        ledger_summary=ledger_summary,
        size_summary=size_summary,
        failure_summary=failure_summary,
        weak_summary=weak_summary,
    )
    summary = summarize_structured_falsification(
        gate_status=gate_status,
        failure_matrix=failure_matrix,
        next_actions=next_actions,
        evidence_manifest=evidence_manifest,
        promotion_summary=promotion_summary,
        ledger_summary=ledger_summary,
        size_summary=size_summary,
        failure_summary=failure_summary,
        weak_summary=weak_summary,
        selection_bias=selection_bias,
        maturity=maturity,
        run_id=run_id,
        promotion_run_dir=promotion_dir,
        trial_ledger_run_dir=ledger_dir,
        size_audit_run_dir=size_dir,
        failure_run_dir=failure_dir,
        weak_year_rebuild_run_dir=weak_dir,
    )
    markdown = render_structured_falsification_markdown(
        summary,
        gate_status=gate_status,
        failure_matrix=failure_matrix,
        next_actions=next_actions,
        evidence_manifest=evidence_manifest,
        selection_bias=selection_bias,
        maturity=maturity,
    )

    gate_status.to_csv(run_dir / "gate_status.csv", index=False, encoding="utf-8-sig")
    failure_matrix.to_csv(run_dir / "structured_failure_matrix.csv", index=False, encoding="utf-8-sig")
    next_actions.to_csv(run_dir / "next_minimum_actions.csv", index=False, encoding="utf-8-sig")
    evidence_manifest.to_csv(run_dir / "evidence_manifest.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        path = Path(research_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    return {**summary, "run_dir": str(run_dir), "research_log": str(research_log_path) if write_research_log else None}


def build_gate_status(promotion_gate: pd.DataFrame) -> pd.DataFrame:
    columns = ["gate", "passed_rows", "failed_rows", "status"]
    gate_columns = ["size_gate", "return_gate", "year_gate", "sample_gate", "drawdown_gate", "style_exposure_gate"]
    if promotion_gate.empty:
        return pd.DataFrame(
            [{"gate": gate, "passed_rows": 0, "failed_rows": 0, "status": "missing_promotion_rows"} for gate in gate_columns],
            columns=columns,
        )
    rows: list[dict[str, Any]] = []
    for gate in gate_columns:
        values = _truthy(promotion_gate[gate]) if gate in promotion_gate.columns else pd.Series(False, index=promotion_gate.index)
        passed = int(values.sum())
        failed = int((~values).sum())
        rows.append({"gate": gate, "passed_rows": passed, "failed_rows": failed, "status": "passed" if failed == 0 else "failed"})
    return pd.DataFrame(rows, columns=columns)


def build_failure_matrix(
    promotion_gate: pd.DataFrame,
    *,
    size_summary: Mapping[str, Any],
    failure_summary: Mapping[str, Any],
    weak_summary: Mapping[str, Any],
) -> pd.DataFrame:
    columns = [
        "signal",
        "failed_gate",
        "failure_type",
        "evidence_grade",
        "evidence_detail",
        "blocking_artifact",
        "minimum_next_action",
    ]
    if promotion_gate.empty:
        return pd.DataFrame(
            [
                {
                    "signal": "",
                    "failed_gate": "promotion_rows",
                    "failure_type": "promotion_evidence_missing",
                    "evidence_grade": "missing",
                    "evidence_detail": "promotion_gate_summary.csv has no rows.",
                    "blocking_artifact": "",
                    "minimum_next_action": "Rerun frontier_promotion_gate against current combined constraint and size audit artifacts.",
                }
            ],
            columns=columns,
        )

    rows: list[dict[str, Any]] = []
    for item in promotion_gate.to_dict("records"):
        signal = str(item.get("signal", ""))
        failed_gates = [part for part in str(item.get("failed_gates", "")).split(",") if part]
        for gate in failed_gates:
            rows.append(
                _failure_row(
                    signal=signal,
                    gate=gate,
                    promotion_row=item,
                    size_summary=size_summary,
                    failure_summary=failure_summary,
                    weak_summary=weak_summary,
                )
            )
    return pd.DataFrame(rows, columns=columns)


def build_next_minimum_actions(failure_matrix: pd.DataFrame, weak_backlog: pd.DataFrame | None = None) -> pd.DataFrame:
    columns = ["priority", "failure_type", "blocking_gate", "minimum_next_action", "success_evidence"]
    if failure_matrix.empty:
        return pd.DataFrame(
            [
                {
                    "priority": 1,
                    "failure_type": "none",
                    "blocking_gate": "",
                    "minimum_next_action": "Prepare independent review package before any production discussion.",
                    "success_evidence": "All promotion rows pass and at least one row reaches out_of_sample_supported review readiness.",
                }
            ],
            columns=columns,
        )
    priority = {
        "data_size_gate": 1,
        "weak_year_return_year_gate": 2,
        "style_exposure_gate": 3,
        "sample_gate": 4,
        "drawdown_gate": 5,
        "promotion_evidence_missing": 6,
    }
    rows: list[dict[str, Any]] = []
    for failure_type in sorted(failure_matrix["failure_type"].dropna().astype(str).unique(), key=lambda item: priority.get(item, 99)):
        failed = failure_matrix.loc[failure_matrix["failure_type"].astype(str) == failure_type]
        gates = ",".join(sorted(failed["failed_gate"].dropna().astype(str).unique()))
        rows.append(
            {
                "priority": priority.get(failure_type, 99),
                "failure_type": failure_type,
                "blocking_gate": gates,
                "minimum_next_action": _minimum_action_for_failure_type(failure_type),
                "success_evidence": _success_evidence_for_failure_type(failure_type),
            }
        )
    if weak_backlog is not None and not weak_backlog.empty:
        for backlog in weak_backlog.to_dict("records"):
            if str(backlog.get("status", "")) in {"blocked", "planned"}:
                rows.append(
                    {
                        "priority": 10 + len(rows),
                        "failure_type": str(backlog.get("path", "")),
                        "blocking_gate": str(backlog.get("blocking_gate", "")),
                        "minimum_next_action": str(backlog.get("next_action", "")),
                        "success_evidence": "Backlog item is implemented, rerun through combined constraint and promotion gate, and ledger records evidence grade.",
                    }
                )
    return pd.DataFrame(rows, columns=columns).drop_duplicates(["failure_type", "blocking_gate"]).reset_index(drop=True)


def build_evidence_manifest(
    *,
    promotion_dir: Path,
    ledger_dir: Path,
    size_dir: Path,
    failure_dir: Path,
    weak_dir: Path,
    promotion_summary: Mapping[str, Any],
    ledger_summary: Mapping[str, Any],
    size_summary: Mapping[str, Any],
    failure_summary: Mapping[str, Any],
    weak_summary: Mapping[str, Any],
) -> pd.DataFrame:
    rows = [
        _manifest_row(
            "frontier_promotion_gate",
            promotion_dir,
            promotion_summary,
            evidence_grade="candidate-frontier/backtest_only" if int(promotion_summary.get("candidate_count", 0) or 0) == 0 else "promotion_review_ready",
            research_log=Path("traditional_quant_research/research_log/2026-06-03_frontier_promotion_gate.md"),
        ),
        _manifest_row(
            "frontier_trial_ledger",
            ledger_dir,
            ledger_summary,
            evidence_grade="governance_backtest_only",
            research_log=Path("traditional_quant_research/research_log/2026-06-03_frontier_trial_ledger.md"),
        ),
        _manifest_row(
            "v2_daily_size_audit",
            size_dir,
            size_summary,
            evidence_grade="diagnostic" if not bool(size_summary.get("daily_size_ready_for_research", False)) else "size_ready_for_research",
            research_log=Path("traditional_quant_research/research_log/2026-06-03_v2_daily_size_audit.md"),
        ),
        _manifest_row(
            "frontier_failure_attribution",
            failure_dir,
            failure_summary,
            evidence_grade="diagnostic",
            research_log=Path("traditional_quant_research/research_log/2026-06-03_frontier_failure_attribution.md"),
        ),
        _manifest_row(
            "frontier_weak_year_rebuild",
            weak_dir,
            weak_summary,
            evidence_grade="diagnostic_not_backtest",
            research_log=Path("traditional_quant_research/research_log/2026-06-03_frontier_weak_year_rebuild.md"),
        ),
    ]
    cross_path = Path(str(size_summary.get("current_cross_check_path", "")))
    if cross_path.exists():
        cross_summary = _read_json(cross_path.parent / "summary.json") if (cross_path.parent / "summary.json").exists() else {}
        rows.append(
            _manifest_row(
                "v2_free_size_current_cross_check",
                cross_path.parent,
                cross_summary,
                evidence_grade="diagnostic",
                research_log=Path("traditional_quant_research/research_log/2026-06-03_v2_free_size_current_cross_check.md"),
            )
        )
    return pd.DataFrame(rows)


def summarize_structured_falsification(
    *,
    gate_status: pd.DataFrame,
    failure_matrix: pd.DataFrame,
    next_actions: pd.DataFrame,
    evidence_manifest: pd.DataFrame,
    promotion_summary: Mapping[str, Any],
    ledger_summary: Mapping[str, Any],
    size_summary: Mapping[str, Any],
    failure_summary: Mapping[str, Any],
    weak_summary: Mapping[str, Any],
    selection_bias: pd.DataFrame,
    maturity: pd.DataFrame,
    run_id: str,
    promotion_run_dir: Path,
    trial_ledger_run_dir: Path,
    size_audit_run_dir: Path,
    failure_run_dir: Path,
    weak_year_rebuild_run_dir: Path,
) -> dict[str, Any]:
    candidate_count = int(promotion_summary.get("candidate_count", 0) or 0)
    structured_falsification = bool(candidate_count == 0 and not failure_matrix.empty)
    missing_logs = evidence_manifest.loc[~evidence_manifest["research_log_exists"].astype(bool), "artifact"].astype(str).tolist()
    blocking_categories = sorted(failure_matrix["failure_type"].dropna().astype(str).unique().tolist()) if not failure_matrix.empty else []
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "promotion_run_dir": str(promotion_run_dir),
        "trial_ledger_run_dir": str(trial_ledger_run_dir),
        "size_audit_run_dir": str(size_audit_run_dir),
        "failure_run_dir": str(failure_run_dir),
        "weak_year_rebuild_run_dir": str(weak_year_rebuild_run_dir),
        "candidate_count": candidate_count,
        "out_of_sample_supported_count": 0 if candidate_count == 0 else candidate_count,
        "current_evidence_grade": "candidate-frontier/backtest_only" if candidate_count == 0 else "promotion_review_ready",
        "decision": "structured_falsification_keep_candidate_frontier_backtest_only" if structured_falsification else "promotion_review_ready",
        "structured_falsification_complete": structured_falsification,
        "daily_size_status": str(size_summary.get("status", "")),
        "daily_size_ready_for_research": bool(size_summary.get("daily_size_ready_for_research", False)),
        "weak_year_rebuild_decision": str(weak_summary.get("decision", "")),
        "failure_attribution_decision": str(failure_summary.get("decision", "")),
        "trial_count": int(ledger_summary.get("trial_count", 0) or 0),
        "selection_bias_report_count": int(len(selection_bias)),
        "maturity_dimension_count": int(len(maturity)),
        "blocking_categories": blocking_categories,
        "gate_status": gate_status.to_dict("records"),
        "next_minimum_action_count": int(len(next_actions)),
        "missing_research_logs": missing_logs,
        "formal_evidence_count": int(len(evidence_manifest)),
        "limitations": [
            "This report reads existing audit artifacts and does not rerun backtests.",
            "A structured falsification is not a strategy candidate; it is the auditable stop condition for the current frontier.",
            "Any future upgrade must rerun daily_size audit, frontier promotion gate, and trial ledger after the blocking evidence changes.",
        ],
    }


def render_structured_falsification_markdown(
    summary: Mapping[str, Any],
    *,
    gate_status: pd.DataFrame,
    failure_matrix: pd.DataFrame,
    next_actions: pd.DataFrame,
    evidence_manifest: pd.DataFrame,
    selection_bias: pd.DataFrame,
    maturity: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# Frontier Structured Falsification Report",
            "",
            f"- run_id: `{summary.get('run_id', '')}`",
            f"- decision: `{summary.get('decision', '')}`",
            f"- current_evidence_grade: `{summary.get('current_evidence_grade', '')}`",
            f"- candidate_count: `{summary.get('candidate_count', 0)}`",
            f"- out_of_sample_supported_count: `{summary.get('out_of_sample_supported_count', 0)}`",
            f"- daily_size_status: `{summary.get('daily_size_status', '')}`",
            f"- daily_size_ready_for_research: `{summary.get('daily_size_ready_for_research')}`",
            f"- structured_falsification_complete: `{summary.get('structured_falsification_complete')}`",
            "",
            "## Gate Status",
            "",
            _markdown_table(gate_status),
            "",
            "## Structured Failure Matrix",
            "",
            _markdown_table(failure_matrix),
            "",
            "## Next Minimum Actions",
            "",
            _markdown_table(next_actions),
            "",
            "## Selection Bias",
            "",
            _markdown_table(selection_bias),
            "",
            "## Research Maturity",
            "",
            _markdown_table(maturity),
            "",
            "## Evidence Manifest",
            "",
            _markdown_table(evidence_manifest),
            "",
            "## Interpretation",
            "",
            "The current frontier is structurally falsified for promotion review: no row reaches the required evidence grade, "
            "and the blocking gates are explicit. The project remains `candidate-frontier/backtest_only` until the listed "
            "minimum actions produce new audited evidence and the promotion gate is rerun.",
            "",
        ]
    )


def _failure_row(
    *,
    signal: str,
    gate: str,
    promotion_row: Mapping[str, Any],
    size_summary: Mapping[str, Any],
    failure_summary: Mapping[str, Any],
    weak_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if gate == "size_gate":
        return {
            "signal": signal,
            "failed_gate": gate,
            "failure_type": "data_size_gate",
            "evidence_grade": "diagnostic",
            "evidence_detail": (
                f"daily_size_status={size_summary.get('status', '')}; "
                f"size_rows={size_summary.get('size_rows', 0)}; "
                f"current_cross_check_ok={size_summary.get('current_cross_check_ok', False)}"
            ),
            "blocking_artifact": str(size_summary.get("current_cross_check_path", "")),
            "minimum_next_action": _minimum_action_for_failure_type("data_size_gate"),
        }
    if gate in {"return_gate", "year_gate"}:
        return {
            "signal": signal,
            "failed_gate": gate,
            "failure_type": "weak_year_return_year_gate",
            "evidence_grade": "backtest_only",
            "evidence_detail": (
                f"min_annualized_return={_fmt(promotion_row.get('min_annualized_return'))}; "
                f"positive_year_rate={_fmt(promotion_row.get('positive_year_rate'))}; "
                f"total_weak_signal_years={failure_summary.get('total_weak_signal_years', '')}; "
                f"weak_rebuild_decision={weak_summary.get('decision', '')}"
            ),
            "blocking_artifact": str(failure_summary.get("combined_run_dir", "")),
            "minimum_next_action": _minimum_action_for_failure_type("weak_year_return_year_gate"),
        }
    if gate == "style_exposure_gate":
        return {
            "signal": signal,
            "failed_gate": gate,
            "failure_type": "style_exposure_gate",
            "evidence_grade": "backtest_only",
            "evidence_detail": (
                f"max_monthly_mean_abs_active_exposure={_fmt(promotion_row.get('max_monthly_mean_abs_active_exposure'))}; "
                f"exposure_failures={promotion_row.get('exposure_failures', '')}"
            ),
            "blocking_artifact": str(failure_summary.get("combined_run_dir", "")),
            "minimum_next_action": _minimum_action_for_failure_type("style_exposure_gate"),
        }
    return {
        "signal": signal,
        "failed_gate": gate,
        "failure_type": gate,
        "evidence_grade": "backtest_only",
        "evidence_detail": f"failed_gates={promotion_row.get('failed_gates', '')}",
        "blocking_artifact": str(failure_summary.get("combined_run_dir", "")),
        "minimum_next_action": _minimum_action_for_failure_type(gate),
    }


def _minimum_action_for_failure_type(failure_type: str) -> str:
    mapping = {
        "data_size_gate": "Resolve PIT size evidence first: either make free current cross-check available and then build audited formal daily_size cache, or switch to an authenticated PIT size source before assemble --include-size.",
        "weak_year_return_year_gate": "Wire prior-fit weak-year regime or rebuilt factor rules into the 2017-2026 combined constraint run; do not select thresholds from eval-year returns.",
        "style_exposure_gate": "Upgrade portfolio construction from heuristic penalty to explicit exposure constraints or optimizer, then rerun basket exposure and promotion gate.",
        "sample_gate": "Increase independent evaluation periods without changing the promotion threshold, then rerun combined constraint.",
        "drawdown_gate": "Reduce drawdown through prior-specified risk controls and rerun combined constraint.",
        "promotion_evidence_missing": "Rerun frontier_promotion_gate and frontier_trial_ledger with current artifacts.",
    }
    return mapping.get(failure_type, "Rerun the responsible audit and record a research_log entry with evidence grade.")


def _success_evidence_for_failure_type(failure_type: str) -> str:
    mapping = {
        "data_size_gate": "v2_daily_size_audit reports daily_size_ready_for_research=True and promotion size_gate passes.",
        "weak_year_return_year_gate": "2017-2026 combined constraint has mean/min annualized return >= 0 and positive_year_rate=1.0 at required costs.",
        "style_exposure_gate": "promotion_gate_summary shows style_exposure_gate=True and monthly mean abs active exposure <= threshold.",
        "sample_gate": "promotion_gate_summary shows sample_gate=True at the default total-period threshold.",
        "drawdown_gate": "promotion_gate_summary shows drawdown_gate=True at the default drawdown threshold.",
        "promotion_evidence_missing": "promotion_gate_summary.csv exists with evaluated frontier rows and trial ledger links it.",
    }
    return mapping.get(failure_type, "The responsible gate passes in frontier_promotion_gate and is recorded in trial ledger.")


def _manifest_row(
    artifact: str,
    run_dir: Path,
    summary: Mapping[str, Any],
    *,
    evidence_grade: str,
    research_log: Path,
) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "run_id": str(summary.get("run_id", run_dir.name)),
        "run_dir": str(run_dir),
        "summary_exists": bool((run_dir / "summary.json").exists()),
        "research_log": str(research_log),
        "research_log_exists": bool(research_log.exists()),
        "evidence_grade": evidence_grade,
        "candidate_count": int(summary.get("candidate_count", 0) or 0),
        "decision": str(summary.get("decision", summary.get("status", ""))),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"summary not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{float(value):.6f}" if pd.notna(value) else "nan")
    return view.to_markdown(index=False)


def _fmt(value: Any) -> str:
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
    parser.add_argument("--promotion-run-dir", default=None)
    parser.add_argument("--trial-ledger-run-dir", default=None)
    parser.add_argument("--size-audit-run-dir", default=None)
    parser.add_argument("--failure-run-dir", default=None)
    parser.add_argument("--weak-year-rebuild-run-dir", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", default=str(DEFAULT_RESEARCH_LOG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_frontier_structured_falsification_report(
        promotion_run_dir=args.promotion_run_dir,
        trial_ledger_run_dir=args.trial_ledger_run_dir,
        size_audit_run_dir=args.size_audit_run_dir,
        failure_run_dir=args.failure_run_dir,
        weak_year_rebuild_run_dir=args.weak_year_rebuild_run_dir,
        output_dir=args.output_dir,
        write_research_log=bool(args.write_research_log),
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
