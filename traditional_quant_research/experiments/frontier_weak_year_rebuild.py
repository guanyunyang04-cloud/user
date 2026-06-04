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
DEFAULT_WEAK_YEARS = (2017, 2018, 2022, 2023)
GENERIC_REGIME_SIGNAL = "__generic_market_regime__"
SIGNAL_SPECIFIC_RULE_SCOPE = "signal_specific"
GENERIC_REGIME_RULE_SCOPE = "generic_market_regime"
DEFAULT_THRESHOLDS = {
    "breadth_20d_positive_rate": (0.40, 0.45, 0.50, 0.55),
    "breadth_5d_positive_rate": (0.40, 0.45, 0.50),
    "market_ret_20d_mean": (-0.02, 0.00, 0.02),
}


def run_frontier_weak_year_rebuild(
    *,
    failure_run_dir: str | Path | None = None,
    regime_run_dir: str | Path | None = None,
    factor_family_run_dirs: Mapping[str, str | Path] | Sequence[str] | str | None = None,
    include_generic_regime: bool = False,
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
    fit_eval = build_fit_eval_regime_candidates(signal_year_regime, include_generic_regime=include_generic_regime)
    factor_family_evidence = read_factor_family_evidence(factor_family_run_dirs)
    factor_family_fit_eval = build_fit_eval_factor_family_candidates(factor_family_evidence)
    backlog = build_rebuild_backlog()
    summary = summarize_weak_year_rebuild(
        fit_eval,
        factor_family_fit_eval,
        backlog,
        run_id=run_id,
        failure_run_dir=failure_dir,
        regime_run_dir=regime_dir,
        include_generic_regime=include_generic_regime,
    )
    markdown = render_weak_year_rebuild_markdown(summary, fit_eval, factor_family_fit_eval, backlog)

    signal_year_regime.to_csv(run_dir / "signal_year_regime.csv", index=False, encoding="utf-8-sig")
    fit_eval.to_csv(run_dir / "fit_eval_regime_candidates.csv", index=False, encoding="utf-8-sig")
    factor_family_evidence.to_csv(run_dir / "factor_family_evidence.csv", index=False, encoding="utf-8-sig")
    factor_family_fit_eval.to_csv(run_dir / "fit_eval_factor_family_candidates.csv", index=False, encoding="utf-8-sig")
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
    include_generic_regime: bool = False,
) -> pd.DataFrame:
    """Choose thresholds from prior years only and evaluate the current year descriptively."""

    columns = [
        "eval_year",
        "signal",
        "exposure_penalty_strength",
        "rule_scope",
        "source_signal_count",
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
                    "rule_scope": SIGNAL_SPECIFIC_RULE_SCOPE,
                    "source_signal_count": 1,
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
    if include_generic_regime:
        rows.extend(
            _build_generic_regime_candidate_rows(
                frame,
                metrics=metrics,
                thresholds=thresholds,
                max_prior_years=max_prior_years,
            )
        )
    return pd.DataFrame(rows, columns=columns)


def _build_generic_regime_candidate_rows(
    signal_year_regime: pd.DataFrame,
    *,
    metrics: Sequence[str],
    thresholds: Mapping[str, Sequence[float]],
    max_prior_years: int,
) -> list[dict[str, Any]]:
    """Build reusable market-regime rules without binding them to one signal."""

    if signal_year_regime.empty:
        return []
    rows: list[dict[str, Any]] = []
    frame = signal_year_regime.copy()
    frame["eval_year"] = pd.to_numeric(frame["eval_year"], errors="coerce").astype("Int64")
    frame["exposure_penalty_strength"] = pd.to_numeric(frame["exposure_penalty_strength"], errors="coerce").fillna(0.0)
    for strength, group in frame.groupby("exposure_penalty_strength", sort=True):
        group = group.sort_values(["eval_year", "signal"])
        years = [int(year) for year in group["eval_year"].dropna().astype(int).unique()]
        for year in years:
            prior_years = [candidate for candidate in years if candidate < year][-max_prior_years:]
            fit_rows = group.loc[group["eval_year"].astype(int).isin(prior_years)].copy()
            eval_rows = group.loc[group["eval_year"].astype(int).eq(year)].copy()
            if fit_rows.empty or eval_rows.empty:
                continue
            best = _choose_prior_threshold(fit_rows, metrics=metrics, thresholds=thresholds)
            metric_value = float(pd.to_numeric(eval_rows[best["metric"]], errors="coerce").mean())
            eval_returns = pd.to_numeric(eval_rows["annualized_return"], errors="coerce")
            rows.append(
                {
                    "eval_year": year,
                    "signal": GENERIC_REGIME_SIGNAL,
                    "exposure_penalty_strength": float(strength),
                    "rule_scope": GENERIC_REGIME_RULE_SCOPE,
                    "source_signal_count": int(fit_rows["signal"].dropna().astype(str).nunique()),
                    "fit_years": ",".join(str(item) for item in prior_years),
                    "fit_row_count": int(len(fit_rows)),
                    "metric": best["metric"],
                    "threshold": best["threshold"],
                    "fit_mean_return_when_allowed": best["fit_mean_return_when_allowed"],
                    "fit_positive_year_rate_when_allowed": best["fit_positive_year_rate_when_allowed"],
                    "eval_metric_value": metric_value,
                    "eval_allowed_by_rule": bool(metric_value >= float(best["threshold"])),
                    "eval_annualized_return": float(eval_returns.mean()) if not eval_returns.dropna().empty else np.nan,
                    "eval_weak_year": bool(_to_bool(eval_rows["weak_year"]).any()),
                    "fit_uses_eval_year": False,
                    "evidence_grade": "diagnostic_not_backtest",
                }
            )
    return rows


def read_factor_family_evidence(
    factor_family_run_dirs: Mapping[str, str | Path] | Sequence[str] | str | None,
) -> pd.DataFrame:
    """Read combined-constraint yearly evidence for named factor families."""

    columns = _factor_family_evidence_columns()
    specs = _normalize_factor_family_run_dirs(factor_family_run_dirs)
    if not specs:
        return pd.DataFrame(columns=columns)
    frames: list[pd.DataFrame] = []
    for family, run_dir in specs.items():
        path = Path(run_dir)
        summary_path = path / "combined_constraint_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"factor-family combined summary not found: {summary_path}")
        summary = pd.read_csv(summary_path)
        if summary.empty:
            continue
        metadata = _read_factor_family_metadata(path)
        factor_family = family or str(metadata.get("factor_set", "") or path.name)
        frame = summary.copy()
        frame.insert(0, "factor_family", str(factor_family))
        frame.insert(1, "factor_family_run_dir", str(path))
        for column, default in (
            ("top_n", np.nan),
            ("constraint_variant", "baseline"),
            ("exposure_penalty_strength", 0.0),
            ("fee_bps", np.nan),
            ("capital_amount", np.nan),
            ("impact_bps_per_1pct", np.nan),
            ("max_drawdown", np.nan),
            ("periods", 0),
        ):
            if column not in frame.columns:
                frame[column] = default
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=columns)
    output = pd.concat(frames, ignore_index=True)
    required = {"factor_family", "eval_year", "signal", "annualized_return"}
    if missing := sorted(required - set(output.columns)):
        raise ValueError(f"factor-family evidence missing required columns: {missing}")
    output["factor_family"] = output["factor_family"].astype(str)
    output["eval_year"] = pd.to_numeric(output["eval_year"], errors="coerce").astype("Int64")
    output["signal"] = output["signal"].astype(str)
    output["top_n"] = pd.to_numeric(output["top_n"], errors="coerce").astype("Int64")
    output["constraint_variant"] = output["constraint_variant"].fillna("baseline").astype(str)
    for column in [
        "exposure_penalty_strength",
        "fee_bps",
        "capital_amount",
        "impact_bps_per_1pct",
        "annualized_return",
        "max_drawdown",
        "periods",
    ]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    for column in columns:
        if column not in output.columns:
            output[column] = np.nan
    return output.loc[:, columns].dropna(subset=["eval_year", "signal", "annualized_return"]).reset_index(drop=True)


def build_fit_eval_factor_family_candidates(
    factor_family_evidence: pd.DataFrame,
    *,
    max_prior_years: int = 3,
) -> pd.DataFrame:
    """Select factor families from prior years only and evaluate the current year descriptively."""

    columns = [
        "eval_year",
        "signal",
        "top_n",
        "constraint_variant",
        "exposure_penalty_strength",
        "fee_bps",
        "capital_amount",
        "impact_bps_per_1pct",
        "fit_years",
        "fit_row_count",
        "candidate_factor_families",
        "selected_factor_family",
        "fit_mean_return_selected_family",
        "fit_positive_year_rate_selected_family",
        "fit_worst_drawdown_selected_family",
        "eval_selected_family_annualized_return",
        "eval_selected_family_max_drawdown",
        "eval_selected_family_periods",
        "eval_best_available_family",
        "eval_best_available_annualized_return",
        "fit_uses_eval_year",
        "evidence_grade",
    ]
    if factor_family_evidence.empty:
        return pd.DataFrame(columns=columns)
    required = {
        "factor_family",
        "eval_year",
        "signal",
        "top_n",
        "constraint_variant",
        "exposure_penalty_strength",
        "fee_bps",
        "capital_amount",
        "impact_bps_per_1pct",
        "annualized_return",
    }
    if missing := sorted(required - set(factor_family_evidence.columns)):
        raise ValueError(f"factor-family evidence missing required columns: {missing}")

    frame = factor_family_evidence.copy()
    frame["eval_year"] = pd.to_numeric(frame["eval_year"], errors="coerce").astype("Int64")
    numeric_cols = [
        "top_n",
        "exposure_penalty_strength",
        "fee_bps",
        "capital_amount",
        "impact_bps_per_1pct",
        "annualized_return",
        "max_drawdown",
        "periods",
    ]
    for column in numeric_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["constraint_variant"] = frame["constraint_variant"].fillna("baseline").astype(str)
    group_cols = [
        "signal",
        "top_n",
        "constraint_variant",
        "exposure_penalty_strength",
        "fee_bps",
        "capital_amount",
        "impact_bps_per_1pct",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, sort=True, dropna=False):
        signal, top_n, variant, strength, fee_bps, capital, impact = keys
        group = group.sort_values(["eval_year", "factor_family"])
        years = [int(year) for year in group["eval_year"].dropna().astype(int).unique()]
        for year in years:
            prior_years = [candidate for candidate in years if candidate < year][-max_prior_years:]
            fit_rows = group.loc[group["eval_year"].astype(int).isin(prior_years)].copy()
            eval_rows = group.loc[group["eval_year"].astype(int).eq(year)].copy()
            if fit_rows.empty or eval_rows.empty:
                continue
            family_choice = _choose_prior_factor_family(fit_rows)
            selected_family = str(family_choice["selected_factor_family"])
            eval_selected = eval_rows.loc[eval_rows["factor_family"].astype(str).eq(selected_family)].copy()
            eval_best = eval_rows.sort_values("annualized_return", ascending=False).iloc[0]
            selected_row = eval_selected.iloc[0] if not eval_selected.empty else pd.Series(dtype=object)
            rows.append(
                {
                    "eval_year": year,
                    "signal": signal,
                    "top_n": int(top_n) if pd.notna(top_n) else np.nan,
                    "constraint_variant": variant,
                    "exposure_penalty_strength": float(strength) if pd.notna(strength) else np.nan,
                    "fee_bps": float(fee_bps) if pd.notna(fee_bps) else np.nan,
                    "capital_amount": float(capital) if pd.notna(capital) else np.nan,
                    "impact_bps_per_1pct": float(impact) if pd.notna(impact) else np.nan,
                    "fit_years": ",".join(str(item) for item in prior_years),
                    "fit_row_count": int(len(fit_rows)),
                    "candidate_factor_families": ",".join(sorted(fit_rows["factor_family"].astype(str).unique())),
                    "selected_factor_family": selected_family,
                    "fit_mean_return_selected_family": family_choice["fit_mean_return_selected_family"],
                    "fit_positive_year_rate_selected_family": family_choice["fit_positive_year_rate_selected_family"],
                    "fit_worst_drawdown_selected_family": family_choice["fit_worst_drawdown_selected_family"],
                    "eval_selected_family_annualized_return": float(selected_row.get("annualized_return", np.nan)),
                    "eval_selected_family_max_drawdown": float(selected_row.get("max_drawdown", np.nan)),
                    "eval_selected_family_periods": int(float(selected_row.get("periods", 0) or 0)) if not selected_row.empty else 0,
                    "eval_best_available_family": str(eval_best.get("factor_family", "")),
                    "eval_best_available_annualized_return": float(eval_best.get("annualized_return", np.nan)),
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
                "path": "generic_regime_prior_fit",
                "status": "implemented_diagnostic",
                "blocking_gate": "",
                "next_action": "Use explicitly tagged generic market-regime rows as fallback rules for new signals without signal-specific weak-year history; rerun formal gates before any promotion.",
            },
            {
                "path": "factor_family_prior_fit",
                "status": "implemented_diagnostic",
                "blocking_gate": "",
                "next_action": "Use fit_eval_factor_family_candidates to select core/expanded/blended factor-family candidates, then rerun combined constraint before any promotion.",
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
    factor_family_fit_eval: pd.DataFrame,
    backlog: pd.DataFrame,
    *,
    run_id: str,
    failure_run_dir: Path,
    regime_run_dir: Path,
    include_generic_regime: bool = False,
) -> dict[str, Any]:
    rule_rows = int(len(fit_eval))
    rule_scope = fit_eval.get("rule_scope", pd.Series(dtype=str)).fillna("").astype(str) if not fit_eval.empty else pd.Series(dtype=str)
    generic_rows = int(rule_scope.eq(GENERIC_REGIME_RULE_SCOPE).sum()) if not rule_scope.empty else 0
    eval_weak_allowed = (
        int(fit_eval.loc[fit_eval["eval_weak_year"].eq(True) & fit_eval["eval_allowed_by_rule"].eq(True)].shape[0])
        if not fit_eval.empty
        else 0
    )
    family_rows = int(len(factor_family_fit_eval))
    family_eval_positive = (
        int(pd.to_numeric(factor_family_fit_eval["eval_selected_family_annualized_return"], errors="coerce").ge(0).sum())
        if not factor_family_fit_eval.empty
        else 0
    )
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "failure_run_dir": str(failure_run_dir),
        "regime_run_dir": str(regime_run_dir),
        "include_generic_regime": bool(include_generic_regime),
        "fit_eval_candidate_rows": rule_rows,
        "generic_regime_candidate_rows": generic_rows,
        "weak_years_allowed_by_prior_rule": eval_weak_allowed,
        "fit_uses_eval_year_count": int(fit_eval["fit_uses_eval_year"].sum()) if not fit_eval.empty else 0,
        "fit_eval_factor_family_candidate_rows": family_rows,
        "factor_family_eval_positive_count": family_eval_positive,
        "factor_family_fit_uses_eval_year_count": int(factor_family_fit_eval["fit_uses_eval_year"].sum()) if not factor_family_fit_eval.empty else 0,
        "backlog_count": int(len(backlog)),
        "candidate_count": 0,
        "decision": "diagnostic_rebuild_rules_ready_for_backtest" if rule_rows or family_rows else "insufficient_prior_rows_for_rebuild",
        "limitations": [
            "This experiment chooses regime thresholds and factor-family candidates from prior yearly evidence only, but it does not rerun trades.",
            "Rows remain diagnostic until the selected rule is wired into combined constraint backtests.",
            "Valuation and optimizer paths are explicit backlog items, not completed candidate signals.",
        ],
    }


def render_weak_year_rebuild_markdown(
    summary: Mapping[str, Any],
    fit_eval: pd.DataFrame,
    factor_family_fit_eval: pd.DataFrame,
    backlog: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# Frontier Weak-Year Rebuild",
            "",
            f"- run_id: `{summary.get('run_id', '')}`",
            f"- decision: `{summary.get('decision', '')}`",
            f"- fit_eval_candidate_rows: `{summary.get('fit_eval_candidate_rows', 0)}`",
            f"- generic_regime_candidate_rows: `{summary.get('generic_regime_candidate_rows', 0)}`",
            f"- weak_years_allowed_by_prior_rule: `{summary.get('weak_years_allowed_by_prior_rule', 0)}`",
            f"- fit_uses_eval_year_count: `{summary.get('fit_uses_eval_year_count', 0)}`",
            f"- fit_eval_factor_family_candidate_rows: `{summary.get('fit_eval_factor_family_candidate_rows', 0)}`",
            f"- factor_family_eval_positive_count: `{summary.get('factor_family_eval_positive_count', 0)}`",
            f"- factor_family_fit_uses_eval_year_count: `{summary.get('factor_family_fit_uses_eval_year_count', 0)}`",
            f"- candidate_count: `{summary.get('candidate_count', 0)}`",
            "",
            "## Research Interpretation",
            "",
            *_research_interpretation_lines(summary, fit_eval, factor_family_fit_eval),
            "",
            "## Fit/Eval Regime Candidates",
            "",
            _markdown_table(fit_eval),
            "",
            "## Fit/Eval Factor-Family Candidates",
            "",
            _markdown_table(factor_family_fit_eval),
            "",
            "## Rebuild Backlog",
            "",
            _markdown_table(backlog),
            "",
            "## Interpretation",
            "",
            "This is a weak-year rebuild planning artifact. It prevents eval-year leakage in regime threshold and factor-family selection, but it does not upgrade any frontier strategy.",
            "",
        ]
    )


def _research_interpretation_lines(
    summary: Mapping[str, Any],
    fit_eval: pd.DataFrame,
    factor_family_fit_eval: pd.DataFrame,
) -> list[str]:
    lines = [
        "- This artifact is diagnostic only: it selects regime thresholds and factor families from prior yearly evidence, then records current-year descriptive outcomes without rerunning trades.",
    ]
    if fit_eval.empty:
        lines.append("- Regime prior-fit evidence is empty.")
    else:
        lines.append(
            f"- Regime prior-fit produced `{len(fit_eval)}` rows; weak-year rows allowed by the prior rule: `{summary.get('weak_years_allowed_by_prior_rule', 0)}`."
        )
        generic_rows = int(summary.get("generic_regime_candidate_rows", 0) or 0)
        if generic_rows:
            lines.append(
                f"- Generic market-regime fallback rows: `{generic_rows}`. These rows can support new signals without signal-specific weak-year history, but they must remain explicitly tagged as `{GENERIC_REGIME_RULE_SCOPE}`."
            )
    if factor_family_fit_eval.empty:
        lines.append("- No factor-family evidence was supplied.")
        return lines

    lines.append(
        f"- Factor-family prior-fit produced `{len(factor_family_fit_eval)}` rows; eval-year leakage count: `{summary.get('factor_family_fit_uses_eval_year_count', 0)}`."
    )
    comparable = factor_family_fit_eval.loc[
        factor_family_fit_eval["candidate_factor_families"].fillna("").astype(str).str.contains(",", regex=False)
    ].copy()
    if comparable.empty:
        lines.append("- Factor-family evidence has no rows where more than one family is available under the same protocol; use it as single-family diagnostics only.")
        return lines

    returns = pd.to_numeric(comparable["eval_selected_family_annualized_return"], errors="coerce")
    lines.append(
        "- Comparable factor-family rows: "
        f"`{len(comparable)}`; selected-family mean return `{_fmt_float(returns.mean())}`, "
        f"worst selected return `{_fmt_float(returns.min())}`, positive rate `{_fmt_float((returns >= 0).mean())}`."
    )
    for signal, group in comparable.groupby("signal", sort=True):
        signal_returns = pd.to_numeric(group["eval_selected_family_annualized_return"], errors="coerce")
        selected_counts = ",".join(
            f"{family}:{count}" for family, count in group["selected_factor_family"].astype(str).value_counts().sort_index().items()
        )
        lines.append(
            f"- Comparable `{signal}`: years `{len(group)}`, mean `{_fmt_float(signal_returns.mean())}`, "
            f"worst `{_fmt_float(signal_returns.min())}`, positive rate `{_fmt_float((signal_returns >= 0).mean())}`, selected counts `{selected_counts}`."
        )
    weak = comparable.loc[comparable["eval_year"].isin(DEFAULT_WEAK_YEARS)].copy()
    if not weak.empty:
        weak_parts = []
        for year, group in weak.groupby("eval_year", sort=True):
            year_returns = pd.to_numeric(group["eval_selected_family_annualized_return"], errors="coerce")
            weak_parts.append(f"{int(year)} mean={_fmt_float(year_returns.mean())}, worst={_fmt_float(year_returns.min())}")
        lines.append("- Weak-year comparable rows remain the main damage channel: " + "; ".join(weak_parts) + ".")
    lines.append("- `eval_best_available_family` is an oracle diagnostic column; only `selected_factor_family` is leakage-free enough to wire into a future backtest.")
    return lines


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


def _factor_family_evidence_columns() -> list[str]:
    return [
        "factor_family",
        "factor_family_run_dir",
        "eval_year",
        "signal",
        "top_n",
        "constraint_variant",
        "exposure_penalty_strength",
        "fee_bps",
        "capital_amount",
        "impact_bps_per_1pct",
        "annualized_return",
        "max_drawdown",
        "periods",
    ]


def _normalize_factor_family_run_dirs(
    value: Mapping[str, str | Path] | Sequence[str] | str | None,
) -> dict[str, Path]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key).strip(): Path(path) for key, path in value.items() if str(key).strip()}
    parts = value.split(",") if isinstance(value, str) else list(value)
    output: dict[str, Path] = {}
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        if "=" in text:
            family, path = text.split("=", 1)
            output[family.strip()] = Path(path.strip())
        else:
            path = Path(text)
            output[path.name] = path
    return output


def _read_factor_family_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def _choose_prior_factor_family(fit_rows: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for family, group in fit_rows.groupby("factor_family", sort=True):
        returns = pd.to_numeric(group["annualized_return"], errors="coerce")
        drawdowns = pd.to_numeric(group.get("max_drawdown", pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "selected_factor_family": str(family),
                "fit_mean_return_selected_family": float(returns.mean()),
                "fit_positive_year_rate_selected_family": float((returns >= 0).mean()),
                "fit_worst_drawdown_selected_family": float(drawdowns.min()) if not drawdowns.dropna().empty else np.nan,
                "fit_row_count": int(len(group)),
            }
        )
    if not rows:
        return {
            "selected_factor_family": "",
            "fit_mean_return_selected_family": np.nan,
            "fit_positive_year_rate_selected_family": np.nan,
            "fit_worst_drawdown_selected_family": np.nan,
            "fit_row_count": 0,
        }
    ranked = sorted(
        rows,
        key=lambda item: (
            item["fit_positive_year_rate_selected_family"],
            item["fit_mean_return_selected_family"],
            item["fit_worst_drawdown_selected_family"] if not pd.isna(item["fit_worst_drawdown_selected_family"]) else -np.inf,
            _factor_family_tie_rank(item["selected_factor_family"]),
        ),
        reverse=True,
    )
    return ranked[0]


def _factor_family_tie_rank(family: str) -> int:
    return {"core": 2, "expanded": 1}.get(str(family), 0)


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


def _fmt_float(value: Any) -> str:
    return "nan" if pd.isna(value) else f"{float(value):.6f}"


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
    parser.add_argument("--factor-family-run-dirs", default=None, help="Comma-separated factor family specs such as core=<dir>,expanded=<dir>.")
    parser.add_argument("--include-generic-regime", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", default=str(DEFAULT_RESEARCH_LOG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_frontier_weak_year_rebuild(
        failure_run_dir=args.failure_run_dir,
        regime_run_dir=args.regime_run_dir,
        factor_family_run_dirs=args.factor_family_run_dirs,
        include_generic_regime=bool(args.include_generic_regime),
        output_dir=args.output_dir,
        write_research_log=bool(args.write_research_log),
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
