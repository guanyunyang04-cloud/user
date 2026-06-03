"""Fit/eval separated weak-year rebuild diagnostics for current frontier."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.experiments.frontier_promotion_gate import latest_run_dir


DEFAULT_FAILURE_ATTRIBUTION_ROOT = Path("traditional_quant_research/output/experiments/frontier_failure_attribution")
DEFAULT_REGIME_ATTRIBUTION_ROOT = Path("traditional_quant_research/output/experiments/frontier_weak_year_regime_attribution")
DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/frontier_weak_year_rebuild")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-03_frontier_weak_year_rebuild.md")
DEFAULT_REGIME_METRICS = ("breadth_20d_positive_rate", "market_ret_20d_mean", "breadth_5d_positive_rate")
DEFAULT_THRESHOLDS = {
    "breadth_20d_positive_rate": (0.40, 0.45, 0.50, 0.55),
    "breadth_5d_positive_rate": (0.40, 0.45, 0.50),
    "market_ret_20d_mean": (-0.02, 0.00, 0.02),
}


def run_frontier_weak_year_rebuild(
    *,
    failure_run_dir: str | Path | None = None,
    regime_run_dir: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    """Write fit/eval-separated regime rebuild candidates and backlog."""

    failure_dir = Path(failure_run_dir) if failure_run_dir is not None else latest_run_dir(DEFAULT_FAILURE_ATTRIBUTION_ROOT)
    regime_dir = Path(regime_run_dir) if regime_run_dir is not None else latest_run_dir(DEFAULT_REGIME_ATTRIBUTION_ROOT)
    run_id = f"frontier_weak_year_rebuild_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    yearly_failure = pd.read_csv(failure_dir / "yearly_failure_attribution.csv")
    yearly_regime = pd.read_csv(regime_dir / "yearly_market_regime.csv")
    signal_year_regime = build_signal_year_regime(yearly_failure, yearly_regime)
    fit_eval = build_fit_eval_regime_candidates(signal_year_regime)
    backlog = build_rebuild_backlog()
    summary = summarize_weak_year_rebuild(
        fit_eval,
        backlog,
        run_id=run_id,
        failure_run_dir=failure_dir,
        regime_run_dir=regime_dir,
    )
    markdown = render_weak_year_rebuild_markdown(summary, fit_eval, backlog)

    signal_year_regime.to_csv(run_dir / "signal_year_regime.csv", index=False, encoding="utf-8-sig")
    fit_eval.to_csv(run_dir / "fit_eval_regime_candidates.csv", index=False, encoding="utf-8-sig")
    backlog.to_csv(run_dir / "rebuild_backlog.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        path = Path(research_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    return {**summary, "run_dir": str(run_dir), "research_log": str(research_log_path) if write_research_log else None}


def build_signal_year_regime(yearly_failure: pd.DataFrame, yearly_regime: pd.DataFrame) -> pd.DataFrame:
    required_failure = {"eval_year", "signal", "annualized_return", "weak_year"}
    required_regime = {"eval_year", *DEFAULT_REGIME_METRICS}
    if missing := sorted(required_failure - set(yearly_failure.columns)):
        raise ValueError(f"yearly_failure missing required columns: {missing}")
    if missing := sorted(required_regime - set(yearly_regime.columns)):
        raise ValueError(f"yearly_regime missing required columns: {missing}")
    failure = yearly_failure.copy()
    failure["eval_year"] = pd.to_numeric(failure["eval_year"], errors="coerce").astype("Int64")
    failure["annualized_return"] = pd.to_numeric(failure["annualized_return"], errors="coerce")
    failure["weak_year"] = _to_bool(failure["weak_year"])
    if "exposure_penalty_strength" not in failure.columns:
        failure["exposure_penalty_strength"] = 0.0
    failure["exposure_penalty_strength"] = pd.to_numeric(failure["exposure_penalty_strength"], errors="coerce").fillna(0.0)
    regime = yearly_regime.copy()
    regime["eval_year"] = pd.to_numeric(regime["eval_year"], errors="coerce").astype("Int64")
    for metric in DEFAULT_REGIME_METRICS:
        regime[metric] = pd.to_numeric(regime[metric], errors="coerce")
    return failure.merge(regime[["eval_year", *DEFAULT_REGIME_METRICS]], on="eval_year", how="left").dropna(subset=["eval_year"]).reset_index(drop=True)


def build_fit_eval_regime_candidates(
    signal_year_regime: pd.DataFrame,
    *,
    metrics: Sequence[str] = DEFAULT_REGIME_METRICS,
    thresholds: Mapping[str, Sequence[float]] = DEFAULT_THRESHOLDS,
    max_prior_years: int = 3,
) -> pd.DataFrame:
    """Choose thresholds from prior years only and evaluate the current year descriptively."""

    columns = [
        "eval_year",
        "signal",
        "exposure_penalty_strength",
        "fit_years",
        "fit_row_count",
        "metric",
        "threshold",
        "fit_mean_return_when_allowed",
        "fit_positive_year_rate_when_allowed",
        "eval_metric_value",
        "eval_allowed_by_rule",
        "eval_annualized_return",
        "eval_weak_year",
        "fit_uses_eval_year",
        "evidence_grade",
    ]
    if signal_year_regime.empty:
        return pd.DataFrame(columns=columns)
    frame = signal_year_regime.copy()
    rows: list[dict[str, Any]] = []
    for (signal, strength), group in frame.groupby(["signal", "exposure_penalty_strength"], sort=True):
        group = group.sort_values("eval_year")
        years = [int(year) for year in group["eval_year"].dropna().astype(int).unique()]
        for year in years:
            prior_years = [candidate for candidate in years if candidate < year][-max_prior_years:]
            fit_rows = group.loc[group["eval_year"].astype(int).isin(prior_years)].copy()
            eval_row = group.loc[group["eval_year"].astype(int) == year].iloc[0]
            if fit_rows.empty:
                continue
            best = _choose_prior_threshold(fit_rows, metrics=metrics, thresholds=thresholds)
            rows.append(
                {
                    "eval_year": year,
                    "signal": signal,
                    "exposure_penalty_strength": float(strength),
                    "fit_years": ",".join(str(item) for item in prior_years),
                    "fit_row_count": int(len(fit_rows)),
                    "metric": best["metric"],
                    "threshold": best["threshold"],
                    "fit_mean_return_when_allowed": best["fit_mean_return_when_allowed"],
                    "fit_positive_year_rate_when_allowed": best["fit_positive_year_rate_when_allowed"],
                    "eval_metric_value": float(eval_row.get(best["metric"], np.nan)),
                    "eval_allowed_by_rule": bool(float(eval_row.get(best["metric"], np.nan)) >= float(best["threshold"])),
                    "eval_annualized_return": float(eval_row.get("annualized_return", np.nan)),
                    "eval_weak_year": bool(eval_row.get("weak_year", False)),
                    "fit_uses_eval_year": False,
                    "evidence_grade": "diagnostic_not_backtest",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_rebuild_backlog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "path": "regime_prior_fit",
                "status": "implemented_diagnostic",
                "blocking_gate": "",
                "next_action": "Use fit_eval_regime_candidates as pre-backtest rule candidates; rerun combined constraint before any promotion.",
            },
            {
                "path": "valuation_lagged_factor_family",
                "status": "blocked",
                "blocking_gate": "valuation_pit_timing_ready",
                "next_action": "Verify peTTM/pbMRQ/psTTM/pcfNcfTTM official timing and revision behavior before adding as candidate signals.",
            },
            {
                "path": "portfolio_constraint_optimizer",
                "status": "planned",
                "blocking_gate": "optimizer_backtest_extension",
                "next_action": "Extend horizon backtest selection from heuristic top-N penalty to explicit exposure/turnover/cost constrained optimizer.",
            },
        ]
    )


def summarize_weak_year_rebuild(
    fit_eval: pd.DataFrame,
    backlog: pd.DataFrame,
    *,
    run_id: str,
    failure_run_dir: Path,
    regime_run_dir: Path,
) -> dict[str, Any]:
    rule_rows = int(len(fit_eval))
    eval_weak_allowed = (
        int(fit_eval.loc[fit_eval["eval_weak_year"].eq(True) & fit_eval["eval_allowed_by_rule"].eq(True)].shape[0])
        if not fit_eval.empty
        else 0
    )
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "failure_run_dir": str(failure_run_dir),
        "regime_run_dir": str(regime_run_dir),
        "fit_eval_candidate_rows": rule_rows,
        "weak_years_allowed_by_prior_rule": eval_weak_allowed,
        "fit_uses_eval_year_count": int(fit_eval["fit_uses_eval_year"].sum()) if not fit_eval.empty else 0,
        "backlog_count": int(len(backlog)),
        "candidate_count": 0,
        "decision": "diagnostic_rebuild_rules_ready_for_backtest" if rule_rows else "insufficient_prior_rows_for_rebuild",
        "limitations": [
            "This experiment chooses regime thresholds from prior yearly evidence only, but it does not rerun trades.",
            "Rows remain diagnostic until the selected rule is wired into combined constraint backtests.",
            "Valuation and optimizer paths are explicit backlog items, not completed candidate signals.",
        ],
    }


def render_weak_year_rebuild_markdown(summary: Mapping[str, Any], fit_eval: pd.DataFrame, backlog: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Frontier Weak-Year Rebuild",
            "",
            f"- run_id: `{summary.get('run_id', '')}`",
            f"- decision: `{summary.get('decision', '')}`",
            f"- fit_eval_candidate_rows: `{summary.get('fit_eval_candidate_rows', 0)}`",
            f"- weak_years_allowed_by_prior_rule: `{summary.get('weak_years_allowed_by_prior_rule', 0)}`",
            f"- fit_uses_eval_year_count: `{summary.get('fit_uses_eval_year_count', 0)}`",
            f"- candidate_count: `{summary.get('candidate_count', 0)}`",
            "",
            "## Fit/Eval Regime Candidates",
            "",
            _markdown_table(fit_eval),
            "",
            "## Rebuild Backlog",
            "",
            _markdown_table(backlog),
            "",
            "## Interpretation",
            "",
            "This is a weak-year rebuild planning artifact. It prevents eval-year leakage in regime threshold selection, but it does not upgrade any frontier strategy.",
            "",
        ]
    )


def _choose_prior_threshold(
    fit_rows: pd.DataFrame,
    *,
    metrics: Sequence[str],
    thresholds: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    best: dict[str, Any] = {
        "metric": "",
        "threshold": np.nan,
        "fit_mean_return_when_allowed": -np.inf,
        "fit_positive_year_rate_when_allowed": 0.0,
    }
    for metric in metrics:
        for threshold in thresholds.get(metric, ()):
            allowed = fit_rows.loc[pd.to_numeric(fit_rows[metric], errors="coerce") >= float(threshold)]
            if allowed.empty:
                continue
            returns = pd.to_numeric(allowed["annualized_return"], errors="coerce")
            candidate = {
                "metric": metric,
                "threshold": float(threshold),
                "fit_mean_return_when_allowed": float(returns.mean()),
                "fit_positive_year_rate_when_allowed": float((returns >= 0).mean()),
            }
            if _better_threshold(candidate, best):
                best = candidate
    if not best["metric"]:
        metric = metrics[0]
        best = {
            "metric": metric,
            "threshold": float(thresholds[metric][0]),
            "fit_mean_return_when_allowed": float(pd.to_numeric(fit_rows["annualized_return"], errors="coerce").mean()),
            "fit_positive_year_rate_when_allowed": float((pd.to_numeric(fit_rows["annualized_return"], errors="coerce") >= 0).mean()),
        }
    return best


def _better_threshold(candidate: Mapping[str, Any], incumbent: Mapping[str, Any]) -> bool:
    return (
        float(candidate["fit_positive_year_rate_when_allowed"]),
        float(candidate["fit_mean_return_when_allowed"]),
        -float(candidate["threshold"]),
    ) > (
        float(incumbent["fit_positive_year_rate_when_allowed"]),
        float(incumbent["fit_mean_return_when_allowed"]),
        -float(incumbent["threshold"]) if not pd.isna(incumbent["threshold"]) else -np.inf,
    )


def _to_bool(series: pd.Series) -> pd.Series:
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
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-run-dir", default=None)
    parser.add_argument("--regime-run-dir", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", default=str(DEFAULT_RESEARCH_LOG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_frontier_weak_year_rebuild(
        failure_run_dir=args.failure_run_dir,
        regime_run_dir=args.regime_run_dir,
        output_dir=args.output_dir,
        write_research_log=bool(args.write_research_log),
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
