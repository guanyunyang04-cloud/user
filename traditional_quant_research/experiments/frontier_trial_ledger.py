"""Build a trial ledger and selection-bias report for frontier evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from traditional_quant_research.experiments.frontier_promotion_gate import (
    BAOSTOCK_ONLY_PROMOTION_LEVEL,
    STRATEGY_PROMOTION_LEVEL,
    latest_run_dir,
)
from traditional_quant_research.experiments.frontier_personal_candidate_gate import (
    PERSONAL_BACKTEST_PROMOTION_LEVEL,
)


DEFAULT_COMBINED_OUTPUT_ROOT = Path(
    "traditional_quant_research/output/experiments/low_corr_frontier_combined_constraint_audit"
)
DEFAULT_PROMOTION_OUTPUT_ROOT = Path("traditional_quant_research/output/experiments/frontier_promotion_gate")
DEFAULT_FAILURE_OUTPUT_ROOT = Path("traditional_quant_research/output/experiments/frontier_failure_attribution")
DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/frontier_trial_ledger")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-03_frontier_trial_ledger.md")


def run_frontier_trial_ledger(
    *,
    combined_run_dir: str | Path | None = None,
    promotion_run_dir: str | Path | None = None,
    failure_run_dir: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    """Write trial ledger, selection-bias report, and maturity report."""

    combined_dir = Path(combined_run_dir) if combined_run_dir is not None else latest_run_dir(DEFAULT_COMBINED_OUTPUT_ROOT)
    promotion_dir = Path(promotion_run_dir) if promotion_run_dir is not None and Path(promotion_run_dir).exists() else _latest_or_none(DEFAULT_PROMOTION_OUTPUT_ROOT)
    failure_dir = Path(failure_run_dir) if failure_run_dir is not None and Path(failure_run_dir).exists() else _latest_or_none(DEFAULT_FAILURE_OUTPUT_ROOT)

    run_id = f"frontier_trial_ledger_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    aggregate = pd.read_csv(combined_dir / "combined_constraint_aggregate.csv")
    promotion = _read_optional_csv(promotion_dir / "promotion_gate_summary.csv" if promotion_dir else None)
    failure = _read_optional_csv(failure_dir / "signal_failure_summary.csv" if failure_dir else None)
    ledger = build_trial_ledger(aggregate, promotion)
    selection_bias = build_selection_bias_report(ledger, failure)
    maturity = build_research_maturity_report(ledger, selection_bias, promotion)
    summary = summarize_trial_ledger(
        ledger,
        selection_bias,
        maturity,
        run_id=run_id,
        combined_run_dir=combined_dir,
        promotion_run_dir=promotion_dir,
        failure_run_dir=failure_dir,
    )
    markdown = render_trial_ledger_markdown(summary, selection_bias, maturity)

    ledger.to_csv(run_dir / "trial_ledger.csv", index=False, encoding="utf-8-sig")
    selection_bias.to_csv(run_dir / "selection_bias_report.csv", index=False, encoding="utf-8-sig")
    maturity.to_csv(run_dir / "research_maturity_report.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        path = Path(research_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    return {**summary, "run_dir": str(run_dir), "research_log": str(research_log_path) if write_research_log else None}


def build_trial_ledger(aggregate: pd.DataFrame, promotion: pd.DataFrame | None = None) -> pd.DataFrame:
    """Normalize combined-constraint rows into an auditable trial ledger."""

    columns = [
        "trial_id",
        "experiment_family",
        "research_mode",
        "constraint_variant",
        "signal",
        "fee_bps",
        "impact_bps_per_1pct",
        "capital_amount",
        "exposure_penalty_strength",
        "constraint_fallback_count",
        "constraint_fallback_rate",
        "mean_annualized_return",
        "min_annualized_return",
        "positive_year_rate",
        "worst_max_drawdown",
        "total_periods",
        "promotion_level",
        "failed_gates",
        "evidence_grade",
        "used_for_selection",
    ]
    if aggregate.empty:
        return pd.DataFrame(columns=columns)
    frame = aggregate.copy()
    for column in [
        "fee_bps",
        "impact_bps_per_1pct",
        "capital_amount",
        "exposure_penalty_strength",
        "constraint_fallback_count",
        "constraint_fallback_rate",
        "mean_annualized_return",
        "min_annualized_return",
        "positive_year_rate",
        "worst_max_drawdown",
        "total_periods",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "constraint_variant" not in frame.columns:
        frame["constraint_variant"] = "baseline"
    if "research_mode" not in frame.columns:
        frame["research_mode"] = ""
    if "constraint_fallback_count" not in frame.columns:
        frame["constraint_fallback_count"] = 0
    if "constraint_fallback_rate" not in frame.columns:
        frame["constraint_fallback_rate"] = 0.0
    if "evidence_grade" not in frame.columns:
        frame["evidence_grade"] = "backtest_only"
    frame["trial_id"] = [
        f"trial_{idx:04d}_{_safe_signal(row.get('constraint_variant', 'baseline'))}_{_safe_signal(row.get('signal', ''))}"
        for idx, row in enumerate(frame.to_dict("records"), start=1)
    ]
    frame["experiment_family"] = "frontier_combined_constraint"
    frame["promotion_level"] = "candidate-frontier/backtest_only"
    frame["failed_gates"] = ""
    if promotion is not None and not promotion.empty:
        promo = promotion.copy()
        if "constraint_variant" not in promo.columns:
            promo["constraint_variant"] = "baseline"
        if "research_mode" not in promo.columns:
            promo["research_mode"] = ""
        promo = promo[
            [
                "research_mode",
                "constraint_variant",
                "signal",
                "exposure_penalty_strength",
                "promotion_level",
                "failed_gates",
            ]
        ].copy()
        promo["exposure_penalty_strength"] = pd.to_numeric(promo["exposure_penalty_strength"], errors="coerce")
        frame = frame.merge(
            promo,
            on=["constraint_variant", "signal", "exposure_penalty_strength"],
            how="left",
            suffixes=("", "_gate"),
        )
        frame["research_mode"] = frame["research_mode_gate"].fillna(frame["research_mode"])
        frame["promotion_level"] = frame["promotion_level_gate"].fillna(frame["promotion_level"])
        frame["failed_gates"] = frame["failed_gates_gate"].fillna("")
    selected_idx = frame["mean_annualized_return"].astype(float).idxmax()
    frame["used_for_selection"] = False
    if pd.notna(selected_idx):
        frame.loc[selected_idx, "used_for_selection"] = True
    promoted_mask = frame["promotion_level"].astype(str).eq(STRATEGY_PROMOTION_LEVEL)
    frame.loc[promoted_mask, "evidence_grade"] = "strategy_candidate"
    baostock_mask = frame["promotion_level"].astype(str).eq(BAOSTOCK_ONLY_PROMOTION_LEVEL)
    frame.loc[baostock_mask, "evidence_grade"] = "baostock_only_research_candidate"
    personal_mask = frame["promotion_level"].astype(str).eq(PERSONAL_BACKTEST_PROMOTION_LEVEL)
    frame.loc[personal_mask, "evidence_grade"] = PERSONAL_BACKTEST_PROMOTION_LEVEL
    return frame.reindex(columns=columns).reset_index(drop=True)


def build_selection_bias_report(ledger: pd.DataFrame, failure: pd.DataFrame | None = None) -> pd.DataFrame:
    """Summarize trial count, best-vs-median return, and weak-year evidence."""

    columns = [
        "experiment_family",
        "trial_count",
        "signal_count",
        "best_signal",
        "best_mean_annualized_return",
        "median_mean_annualized_return",
        "best_minus_median_return",
        "selected_trial_id",
        "weak_year_count",
        "positive_year_rate_min",
        "selection_bias_risk",
    ]
    if ledger.empty:
        return pd.DataFrame(columns=columns)
    frame = ledger.copy()
    failure_lookup: dict[str, Any] = {}
    if failure is not None and not failure.empty:
        for row in failure.to_dict("records"):
            failure_lookup[str(row.get("signal", ""))] = row
    rows: list[dict[str, Any]] = []
    for family, group in frame.groupby("experiment_family", sort=True):
        ranked = group.sort_values("mean_annualized_return", ascending=False)
        best = ranked.iloc[0]
        failure_row = failure_lookup.get(str(best.get("signal", "")), {})
        best_ret = float(best.get("mean_annualized_return", np.nan))
        median_ret = float(pd.to_numeric(group["mean_annualized_return"], errors="coerce").median())
        trial_count = int(len(group))
        rows.append(
            {
                "experiment_family": family,
                "trial_count": trial_count,
                "signal_count": int(group["signal"].nunique()),
                "best_signal": str(best.get("signal", "")),
                "best_mean_annualized_return": best_ret,
                "median_mean_annualized_return": median_ret,
                "best_minus_median_return": best_ret - median_ret,
                "selected_trial_id": str(best.get("trial_id", "")),
                "weak_year_count": int(float(failure_row.get("weak_year_count", 0) or 0)),
                "positive_year_rate_min": float(pd.to_numeric(group["positive_year_rate"], errors="coerce").min()),
                "selection_bias_risk": "high" if trial_count >= 9 or best_ret - median_ret > 0.10 else "moderate",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_research_maturity_report(
    ledger: pd.DataFrame,
    selection_bias: pd.DataFrame,
    promotion: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score the current research program against common researcher tiers."""

    true_size_gate = bool(
        promotion is not None
        and not promotion.empty
        and _truthy(promotion.get("true_size_gate", pd.Series(dtype=object))).any()
    )
    strategy_candidate = bool(
        promotion is not None
        and not promotion.empty
        and promotion.get("promotion_level", pd.Series(dtype=object)).astype(str).eq(STRATEGY_PROMOTION_LEVEL).any()
    )
    baostock_only_candidate = bool(
        promotion is not None
        and not promotion.empty
        and promotion.get("promotion_level", pd.Series(dtype=object)).astype(str).eq(BAOSTOCK_ONLY_PROMOTION_LEVEL).any()
    )
    personal_candidate = bool(
        promotion is not None
        and not promotion.empty
        and promotion.get("promotion_level", pd.Series(dtype=object)).astype(str).eq(PERSONAL_BACKTEST_PROMOTION_LEVEL).any()
    )
    trial_ledger_ready = bool(not ledger.empty)
    selection_bias_visible = bool(not selection_bias.empty)
    rows = [
        {
            "dimension": "retail_research_baseline",
            "status": "ahead",
            "evidence": "PIT universe, execution constraints, impact stress, promotion gate and failure attribution are already explicit.",
        },
        {
            "dimension": "academic_factor_standard",
            "status": "partial",
            "evidence": "Trial ledger and selection-bias report are now explicit, but cross-market replication and formal significance correction remain absent.",
        },
        {
            "dimension": "professional_quant_standard",
            "status": "partial" if true_size_gate else "blocked",
            "evidence": "True size/float-size, optimizer-grade risk controls and paper/live shadow evidence are not yet complete.",
        },
        {
            "dimension": "baostock_only_research_readiness",
            "status": "passed" if baostock_only_candidate or personal_candidate or strategy_candidate else "blocked",
            "evidence": "Baostock-only research can advance when return, year, sample, drawdown and proxy-style exposure gates pass; true size neutrality remains unproven.",
        },
        {
            "dimension": "personal_backtest_candidate_readiness",
            "status": "passed" if personal_candidate else "blocked",
            "evidence": "Personal candidate readiness requires Baostock-only source, walk-forward evidence, cost/execution stress, drawdown/weak-year controls and clear user-discretion boundary after selection.",
        },
        {
            "dimension": "strategy_candidate_readiness",
            "status": "passed" if strategy_candidate else "blocked",
            "evidence": "Strategy candidate readiness requires true_size mode, daily_size_ready_for_research, and all return/year/sample/drawdown/style gates.",
        },
        {
            "dimension": "governance_visibility",
            "status": "passed" if trial_ledger_ready and selection_bias_visible else "blocked",
            "evidence": "Trial ledger and selection-bias report artifacts are required before promotion review.",
        },
    ]
    return pd.DataFrame(rows)


def summarize_trial_ledger(
    ledger: pd.DataFrame,
    selection_bias: pd.DataFrame,
    maturity: pd.DataFrame,
    *,
    run_id: str,
    combined_run_dir: Path,
    promotion_run_dir: Path | None,
    failure_run_dir: Path | None,
) -> dict[str, Any]:
    strategy_candidate_count = int(ledger["promotion_level"].astype(str).eq(STRATEGY_PROMOTION_LEVEL).sum()) if not ledger.empty else 0
    baostock_only_candidate_count = int(ledger["promotion_level"].astype(str).eq(BAOSTOCK_ONLY_PROMOTION_LEVEL).sum()) if not ledger.empty else 0
    personal_candidate_count = int(ledger["promotion_level"].astype(str).eq(PERSONAL_BACKTEST_PROMOTION_LEVEL).sum()) if not ledger.empty else 0
    candidate_count = strategy_candidate_count + baostock_only_candidate_count + personal_candidate_count
    if strategy_candidate_count:
        decision = "promotion_review_ready"
    elif personal_candidate_count:
        decision = "personal_strategy_candidates_selected"
    elif baostock_only_candidate_count:
        decision = "baostock_only_research_review_ready"
    else:
        decision = "keep_candidate_frontier_backtest_only"
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "combined_run_dir": str(combined_run_dir),
        "promotion_run_dir": str(promotion_run_dir) if promotion_run_dir else "",
        "failure_run_dir": str(failure_run_dir) if failure_run_dir else "",
        "trial_count": int(len(ledger)),
        "selection_bias_report_count": int(len(selection_bias)),
        "maturity_dimension_count": int(len(maturity)),
        "candidate_count": candidate_count,
        "strategy_candidate_count": strategy_candidate_count,
        "baostock_only_candidate_count": baostock_only_candidate_count,
        "personal_backtest_candidate_count": personal_candidate_count,
        "decision": decision,
    }


def render_trial_ledger_markdown(
    summary: Mapping[str, Any],
    selection_bias: pd.DataFrame,
    maturity: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# Frontier Trial Ledger",
            "",
            f"- run_id: `{summary.get('run_id', '')}`",
            f"- decision: `{summary.get('decision', '')}`",
            f"- trial_count: `{summary.get('trial_count', 0)}`",
            f"- candidate_count: `{summary.get('candidate_count', 0)}`",
            f"- strategy_candidate_count: `{summary.get('strategy_candidate_count', 0)}`",
            f"- baostock_only_candidate_count: `{summary.get('baostock_only_candidate_count', 0)}`",
            f"- personal_backtest_candidate_count: `{summary.get('personal_backtest_candidate_count', 0)}`",
            "",
            "## Selection Bias Report",
            "",
            _markdown_table(selection_bias),
            "",
            "## Research Maturity Report",
            "",
            _markdown_table(maturity),
            "",
            "## Interpretation",
            "",
            "This ledger is a governance artifact. It does not rerun backtests and does not promote candidates by itself.",
            "",
        ]
    )


def _latest_or_none(root: Path) -> Path | None:
    try:
        return latest_run_dir(root)
    except FileNotFoundError:
        return None


def _read_optional_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_signal(value: Any) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value))[:40]


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
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-run-dir", default=None)
    parser.add_argument("--promotion-run-dir", default=None)
    parser.add_argument("--failure-run-dir", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", default=str(DEFAULT_RESEARCH_LOG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_frontier_trial_ledger(
        combined_run_dir=args.combined_run_dir,
        promotion_run_dir=args.promotion_run_dir,
        failure_run_dir=args.failure_run_dir,
        output_dir=args.output_dir,
        write_research_log=bool(args.write_research_log),
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
