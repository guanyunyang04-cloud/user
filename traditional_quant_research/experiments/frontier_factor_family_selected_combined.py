"""Assemble prior-fit factor-family selected combined evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.experiments.frontier_promotion_gate import latest_run_dir
from traditional_quant_research.experiments.low_corr_frontier_combined_constraint_audit import (
    best_combined_rows,
    summarize_combined_basket_exposure,
    summarize_combined_constraint_audit,
    summarize_combined_industry_exposure,
)


DEFAULT_WEAK_YEAR_REBUILD_ROOT = Path("traditional_quant_research/output/experiments/frontier_weak_year_rebuild")
DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/frontier_factor_family_selected_combined")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-04_frontier_factor_family_selected_combined.md")
DEFAULT_OUTPUT_CONSTRAINT_VARIANT = "factor_family_prior_fit"
DEFAULT_COLD_START_FAMILY = "core"

SELECTION_META_COLUMNS = [
    "selected_factor_family",
    "factor_family_selection_reason",
    "factor_family_candidate_families",
    "factor_family_fit_years",
    "factor_family_fit_row_count",
    "factor_family_selection_grade",
    "eval_best_available_family",
    "eval_best_available_annualized_return",
    "source_constraint_variant",
    "source_factor_family_run_dir",
]


def run_frontier_factor_family_selected_combined(
    *,
    weak_year_rebuild_run_dir: str | Path | None = None,
    factor_family_run_dirs: Mapping[str, str | Path] | Sequence[str] | str | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_n_values: Sequence[int] | str | None = None,
    impact_bps_per_1pct_values: Sequence[float] | str | None = None,
    cold_start_family: str = DEFAULT_COLD_START_FAMILY,
    output_constraint_variant: str = DEFAULT_OUTPUT_CONSTRAINT_VARIANT,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    weak_dir = Path(weak_year_rebuild_run_dir) if weak_year_rebuild_run_dir is not None else latest_run_dir(DEFAULT_WEAK_YEAR_REBUILD_ROOT)
    rules_path = weak_dir / "fit_eval_factor_family_candidates.csv"
    if not rules_path.exists():
        raise FileNotFoundError(f"factor-family candidates not found: {rules_path}")
    factor_evidence_path = weak_dir / "factor_family_evidence.csv"
    if not factor_evidence_path.exists():
        raise FileNotFoundError(f"factor-family evidence not found: {factor_evidence_path}")

    rules = pd.read_csv(rules_path)
    factor_evidence = pd.read_csv(factor_evidence_path)
    selected_top_n = _parse_optional_int_tuple(top_n_values)
    selected_impacts = _parse_optional_float_tuple(impact_bps_per_1pct_values)
    plan = build_selected_factor_family_plan(
        rules,
        factor_family_evidence=factor_evidence,
        cold_start_family=cold_start_family,
        top_n_values=selected_top_n,
        impact_bps_per_1pct_values=selected_impacts,
        output_constraint_variant=output_constraint_variant,
    )
    if plan.empty:
        raise ValueError("selected factor-family plan is empty")

    specs = _normalize_factor_family_run_dirs(factor_family_run_dirs)
    if not specs:
        specs = _factor_family_run_dirs_from_evidence(factor_evidence)
    family_runs = {family: _read_combined_run(path) for family, path in specs.items()}
    missing_families = sorted(set(plan["selected_factor_family"].astype(str)) - set(family_runs))
    if missing_families:
        raise ValueError(f"missing combined run dirs for selected factor families: {missing_families}")

    run_id = f"frontier_factor_family_selected_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    selected = _assemble_selected_frames(plan, family_runs, output_constraint_variant=output_constraint_variant)
    summary = selected["combined_constraint_summary"]
    if summary.empty:
        raise ValueError("no selected combined summary rows")
    aggregate = summarize_combined_constraint_audit(summary)
    exposure = selected["combined_constraint_basket_exposure"]
    exposure_summary = summarize_combined_basket_exposure(exposure)
    industry_exposure = selected["combined_constraint_industry_exposure"]
    industry_summary = summarize_combined_industry_exposure(industry_exposure)
    liquidity_summary = selected["combined_constraint_liquidity_summary"]

    first_summary = next(iter(family_runs.values()))["summary_json"]
    result = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "weak_year_rebuild_run_dir": str(weak_dir),
        "snapshot_id": str(first_summary.get("snapshot_id", "")),
        "selection_method": "prior_fit_selected_factor_family",
        "cold_start_family": str(cold_start_family),
        "output_constraint_variant": str(output_constraint_variant),
        "selected_years": sorted(pd.to_numeric(plan["eval_year"], errors="coerce").dropna().astype(int).unique().tolist()),
        "top_n_values": sorted(pd.to_numeric(plan["top_n"], errors="coerce").dropna().astype(int).unique().tolist()),
        "impact_bps_per_1pct_values": sorted(pd.to_numeric(plan["impact_bps_per_1pct"], errors="coerce").dropna().astype(float).unique().tolist()),
        "selected_plan_rows": int(len(plan)),
        "selected_summary_rows": int(len(summary)),
        "selected_factor_family_counts": plan["selected_factor_family"].astype(str).value_counts().sort_index().to_dict(),
        "factor_family_run_dirs": {family: str(run["run_dir"]) for family, run in family_runs.items()},
        "horizon": first_summary.get("horizon"),
        "rebalance_frequency": first_summary.get("rebalance_frequency"),
        "buffer_multiplier": first_summary.get("buffer_multiplier"),
        "execution_constraints": bool(first_summary.get("execution_constraints", False)),
        "limit_threshold": first_summary.get("limit_threshold"),
        "candidate_count": 0,
        "assessment": "prior-fit factor-family selected combined evidence only; rerun the formal personal gate before candidate selection",
        "best_30bps_100m_rows": best_combined_rows(aggregate, fee_bps=30.0, capital_amount=100_000_000.0),
        "output_dir": str(run_dir),
    }

    summary.to_csv(run_dir / "combined_constraint_summary.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(run_dir / "combined_constraint_aggregate.csv", index=False, encoding="utf-8-sig")
    selected["combined_constraint_trades"].to_csv(run_dir / "combined_constraint_trades.csv", index=False, encoding="utf-8-sig")
    selected["combined_constraint_liquidity"].to_csv(run_dir / "combined_constraint_liquidity.csv", index=False, encoding="utf-8-sig")
    liquidity_summary.to_csv(run_dir / "combined_constraint_liquidity_summary.csv", index=False, encoding="utf-8-sig")
    exposure.to_csv(run_dir / "combined_constraint_basket_exposure.csv", index=False, encoding="utf-8-sig")
    exposure_summary.to_csv(run_dir / "combined_constraint_basket_exposure_summary.csv", index=False, encoding="utf-8-sig")
    industry_exposure.to_csv(run_dir / "combined_constraint_industry_exposure.csv", index=False, encoding="utf-8-sig")
    industry_summary.to_csv(run_dir / "combined_constraint_industry_summary.csv", index=False, encoding="utf-8-sig")
    selected["combined_constraint_meta"].to_csv(run_dir / "combined_constraint_meta.csv", index=False, encoding="utf-8-sig")
    plan.to_csv(run_dir / "factor_family_selection_plan.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_factor_family_selected_markdown(result, plan, aggregate, exposure_summary, industry_summary, liquidity_summary)
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        path = Path(research_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    return result


def build_selected_factor_family_plan(
    fit_eval_factor_family_candidates: pd.DataFrame,
    *,
    factor_family_evidence: pd.DataFrame,
    cold_start_family: str = DEFAULT_COLD_START_FAMILY,
    top_n_values: Sequence[int] | None = None,
    impact_bps_per_1pct_values: Sequence[float] | None = None,
    output_constraint_variant: str = DEFAULT_OUTPUT_CONSTRAINT_VARIANT,
) -> pd.DataFrame:
    columns = [
        "eval_year",
        "signal",
        "top_n",
        "source_constraint_variant",
        "constraint_variant",
        "exposure_penalty_strength",
        "fee_bps",
        "capital_amount",
        "impact_bps_per_1pct",
        "selected_factor_family",
        "factor_family_selection_reason",
        "factor_family_candidate_families",
        "factor_family_fit_years",
        "factor_family_fit_row_count",
        "factor_family_selection_grade",
        "eval_best_available_family",
        "eval_best_available_annualized_return",
        "selected_evidence_available",
    ]
    if factor_family_evidence.empty:
        return pd.DataFrame(columns=columns)
    evidence = _normalize_factor_evidence(factor_family_evidence)
    if top_n_values is not None:
        evidence = evidence.loc[evidence["top_n"].isin({int(value) for value in top_n_values})].copy()
    if impact_bps_per_1pct_values is not None:
        impacts = [float(value) for value in impact_bps_per_1pct_values]
        evidence = evidence.loc[evidence["impact_bps_per_1pct"].map(lambda value: any(np.isclose(value, item) for item in impacts))].copy()
    rules = _normalize_factor_rules(fit_eval_factor_family_candidates)
    if not rules.empty and rules["fit_uses_eval_year"].any():
        raise ValueError("fit_uses_eval_year must be false for factor-family selection")

    rows: list[dict[str, Any]] = []
    group_cols = [
        "signal",
        "top_n",
        "constraint_variant",
        "exposure_penalty_strength",
        "fee_bps",
        "capital_amount",
        "impact_bps_per_1pct",
    ]
    for keys, group in evidence.groupby(group_cols, sort=True, dropna=False):
        signal, top_n, variant, strength, fee_bps, capital, impact = keys
        years = sorted(group["eval_year"].dropna().astype(int).unique().tolist())
        for year in years:
            available = sorted(group.loc[group["eval_year"].eq(year), "factor_family"].astype(str).unique().tolist())
            rule = _matching_rule(
                rules,
                eval_year=int(year),
                signal=str(signal),
                top_n=int(top_n),
                constraint_variant=str(variant),
                exposure_penalty_strength=float(strength),
                fee_bps=float(fee_bps),
                capital_amount=float(capital),
                impact_bps_per_1pct=float(impact),
            )
            if rule is not None:
                selected_family = str(rule.get("selected_factor_family", ""))
                reason = "prior_fit_selected"
                fit_years = str(rule.get("fit_years", ""))
                fit_row_count = int(float(rule.get("fit_row_count", 0) or 0))
                candidate_families = str(rule.get("candidate_factor_families", ",".join(available)))
                selection_grade = str(rule.get("evidence_grade", "diagnostic_not_backtest"))
                eval_best_family = str(rule.get("eval_best_available_family", ""))
                eval_best_return = float(rule.get("eval_best_available_annualized_return", np.nan))
            elif len(available) == 1:
                selected_family = available[0]
                reason = "single_available_family"
                fit_years = ""
                fit_row_count = 0
                candidate_families = ",".join(available)
                selection_grade = "diagnostic_not_backtest"
                eval_best_family = available[0]
                eval_best_return = float(
                    group.loc[group["eval_year"].eq(year) & group["factor_family"].astype(str).eq(available[0]), "annualized_return"].iloc[0]
                )
            else:
                selected_family = str(cold_start_family)
                reason = "cold_start_family"
                fit_years = ""
                fit_row_count = 0
                candidate_families = ",".join(available)
                selection_grade = "diagnostic_not_backtest"
                best = group.loc[group["eval_year"].eq(year)].sort_values("annualized_return", ascending=False).iloc[0]
                eval_best_family = str(best.get("factor_family", ""))
                eval_best_return = float(best.get("annualized_return", np.nan))
            rows.append(
                {
                    "eval_year": int(year),
                    "signal": str(signal),
                    "top_n": int(top_n),
                    "source_constraint_variant": str(variant),
                    "constraint_variant": str(output_constraint_variant),
                    "exposure_penalty_strength": float(strength),
                    "fee_bps": float(fee_bps),
                    "capital_amount": float(capital),
                    "impact_bps_per_1pct": float(impact),
                    "selected_factor_family": selected_family,
                    "factor_family_selection_reason": reason,
                    "factor_family_candidate_families": candidate_families,
                    "factor_family_fit_years": fit_years,
                    "factor_family_fit_row_count": fit_row_count,
                    "factor_family_selection_grade": selection_grade,
                    "eval_best_available_family": eval_best_family,
                    "eval_best_available_annualized_return": eval_best_return,
                    "selected_evidence_available": selected_family in available,
                }
            )
    output = pd.DataFrame(rows, columns=columns)
    if not output.empty:
        output["constraint_variant"] = str(output_constraint_variant)
    return output


def render_factor_family_selected_markdown(
    result: Mapping[str, Any],
    plan: pd.DataFrame,
    aggregate: pd.DataFrame,
    exposure_summary: pd.DataFrame,
    industry_summary: pd.DataFrame,
    liquidity_summary: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# Frontier Factor-Family Selected Combined Evidence",
            "",
            f"- run_id: `{result.get('run_id', '')}`",
            f"- selection_method: `{result.get('selection_method', '')}`",
            f"- output_constraint_variant: `{result.get('output_constraint_variant', '')}`",
            f"- selected_factor_family_counts: `{result.get('selected_factor_family_counts', {})}`",
            f"- selected_summary_rows: `{result.get('selected_summary_rows', 0)}`",
            "- candidate_count: `0`",
            "",
            "## Interpretation",
            "",
            "This experiment assembles already-backtested family evidence using only prior-fit `selected_factor_family` rules. "
            "`eval_best_available_family` remains an oracle diagnostic and is not used for selection.",
            "",
            "## Selection Plan",
            "",
            _markdown_table(plan),
            "",
            "## Aggregate",
            "",
            _markdown_table(aggregate),
            "",
            "## Basket Exposure Summary",
            "",
            _markdown_table(exposure_summary),
            "",
            "## Industry Summary",
            "",
            _markdown_table(industry_summary),
            "",
            "## Liquidity Summary",
            "",
            _markdown_table(liquidity_summary),
            "",
        ]
    )


def _assemble_selected_frames(
    plan: pd.DataFrame,
    family_runs: Mapping[str, Mapping[str, Any]],
    *,
    output_constraint_variant: str,
) -> dict[str, pd.DataFrame]:
    output: dict[str, list[pd.DataFrame]] = {
        "combined_constraint_summary": [],
        "combined_constraint_trades": [],
        "combined_constraint_liquidity": [],
        "combined_constraint_liquidity_summary": [],
        "combined_constraint_basket_exposure": [],
        "combined_constraint_industry_exposure": [],
        "combined_constraint_meta": [],
    }
    for item in plan.to_dict("records"):
        if not bool(item.get("selected_evidence_available", False)):
            raise ValueError(
                f"selected family evidence missing for year={item.get('eval_year')} signal={item.get('signal')} "
                f"family={item.get('selected_factor_family')}"
            )
        family = str(item["selected_factor_family"])
        run = family_runs[family]
        for name in output:
            frame = run.get(name, pd.DataFrame())
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                continue
            subset = _select_frame_rows(frame, item, name=name, default_top_n=run["default_top_n"])
            if subset.empty:
                if name == "combined_constraint_summary":
                    raise ValueError(f"missing selected summary row for {item}")
                continue
            subset = _attach_selection_meta(subset, item, run_dir=run["run_dir"], output_constraint_variant=output_constraint_variant)
            output[name].append(subset)
    return {name: _dedupe_selected_frames(frames) for name, frames in output.items()}


def _select_frame_rows(frame: pd.DataFrame, item: Mapping[str, Any], *, name: str, default_top_n: int) -> pd.DataFrame:
    work = _ensure_selection_columns(frame, default_top_n=default_top_n)
    mask = pd.to_numeric(work["eval_year"], errors="coerce").eq(int(item["eval_year"]))
    if name == "combined_constraint_meta":
        return work.loc[mask].copy()
    signal = str(item["signal"])
    if "signal" in work.columns:
        mask &= work["signal"].astype(str).eq(signal)
    elif "signal_config" in work.columns:
        mask &= work["signal_config"].astype(str).eq(signal)
    if "top_n" in work.columns:
        mask &= pd.to_numeric(work["top_n"], errors="coerce").eq(int(item["top_n"]))
    if "constraint_variant" in work.columns:
        mask &= work["constraint_variant"].astype(str).eq(str(item["source_constraint_variant"]))
    if "exposure_penalty_strength" in work.columns:
        mask &= np.isclose(pd.to_numeric(work["exposure_penalty_strength"], errors="coerce"), float(item["exposure_penalty_strength"]))
    if name in {"combined_constraint_summary", "combined_constraint_trades"}:
        mask &= np.isclose(pd.to_numeric(work["fee_bps"], errors="coerce"), float(item["fee_bps"]))
        mask &= np.isclose(pd.to_numeric(work["capital_amount"], errors="coerce"), float(item["capital_amount"]))
        mask &= np.isclose(pd.to_numeric(work["impact_bps_per_1pct"], errors="coerce"), float(item["impact_bps_per_1pct"]))
    return work.loc[mask].copy()


def _ensure_selection_columns(frame: pd.DataFrame, *, default_top_n: int) -> pd.DataFrame:
    work = frame.copy()
    if "top_n" not in work.columns:
        work["top_n"] = int(default_top_n)
    if "constraint_variant" not in work.columns:
        work["constraint_variant"] = "baseline"
    if "evidence_grade" not in work.columns:
        work["evidence_grade"] = "backtest_only"
    if "constraint_fallback_count" not in work.columns:
        work["constraint_fallback_count"] = 0
    if "constraint_fallback_rate" not in work.columns:
        work["constraint_fallback_rate"] = 0.0
    return work


def _attach_selection_meta(
    frame: pd.DataFrame,
    item: Mapping[str, Any],
    *,
    run_dir: Path,
    output_constraint_variant: str,
) -> pd.DataFrame:
    output = frame.copy()
    source_variant = str(item["source_constraint_variant"])
    output["source_constraint_variant"] = source_variant
    output["constraint_variant"] = str(output_constraint_variant)
    output["selected_factor_family"] = str(item["selected_factor_family"])
    output["factor_family_selection_reason"] = str(item["factor_family_selection_reason"])
    output["factor_family_candidate_families"] = str(item["factor_family_candidate_families"])
    output["factor_family_fit_years"] = str(item["factor_family_fit_years"])
    output["factor_family_fit_row_count"] = int(float(item.get("factor_family_fit_row_count", 0) or 0))
    output["factor_family_selection_grade"] = str(item["factor_family_selection_grade"])
    output["eval_best_available_family"] = str(item["eval_best_available_family"])
    output["eval_best_available_annualized_return"] = float(item.get("eval_best_available_annualized_return", np.nan))
    output["source_factor_family_run_dir"] = str(run_dir)
    return output


def _dedupe_selected_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    return frame.drop_duplicates().reset_index(drop=True)


def _read_combined_run(path: Path) -> dict[str, Any]:
    summary_json_path = path / "summary.json"
    summary_json = json.loads(summary_json_path.read_text(encoding="utf-8-sig")) if summary_json_path.exists() else {}
    default_top_n = int(summary_json.get("top_n", 200) or 200)
    return {
        "run_dir": path,
        "summary_json": summary_json,
        "default_top_n": default_top_n,
        "combined_constraint_summary": _read_csv(path / "combined_constraint_summary.csv"),
        "combined_constraint_trades": _read_csv(path / "combined_constraint_trades.csv"),
        "combined_constraint_liquidity": _read_csv(path / "combined_constraint_liquidity.csv"),
        "combined_constraint_liquidity_summary": _read_csv(path / "combined_constraint_liquidity_summary.csv"),
        "combined_constraint_basket_exposure": _read_csv(path / "combined_constraint_basket_exposure.csv"),
        "combined_constraint_industry_exposure": _read_csv(path / "combined_constraint_industry_exposure.csv"),
        "combined_constraint_meta": _read_csv(path / "combined_constraint_meta.csv"),
    }


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _normalize_factor_evidence(frame: pd.DataFrame) -> pd.DataFrame:
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
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"factor_family_evidence missing required columns: {missing}")
    output = frame.copy()
    output["factor_family"] = output["factor_family"].astype(str)
    output["eval_year"] = pd.to_numeric(output["eval_year"], errors="coerce").astype(int)
    output["signal"] = output["signal"].astype(str)
    output["top_n"] = pd.to_numeric(output["top_n"], errors="coerce").astype(int)
    output["constraint_variant"] = output["constraint_variant"].fillna("baseline").astype(str)
    for column in ["exposure_penalty_strength", "fee_bps", "capital_amount", "impact_bps_per_1pct", "annualized_return"]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _normalize_factor_rules(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    required = {
        "eval_year",
        "signal",
        "top_n",
        "constraint_variant",
        "exposure_penalty_strength",
        "fee_bps",
        "capital_amount",
        "impact_bps_per_1pct",
        "selected_factor_family",
        "fit_uses_eval_year",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"fit_eval_factor_family_candidates missing required columns: {missing}")
    output = frame.copy()
    output["eval_year"] = pd.to_numeric(output["eval_year"], errors="coerce").astype(int)
    output["signal"] = output["signal"].astype(str)
    output["top_n"] = pd.to_numeric(output["top_n"], errors="coerce").astype(int)
    output["constraint_variant"] = output["constraint_variant"].fillna("baseline").astype(str)
    output["selected_factor_family"] = output["selected_factor_family"].astype(str)
    output["fit_uses_eval_year"] = _truthy(output["fit_uses_eval_year"])
    for column in ["exposure_penalty_strength", "fee_bps", "capital_amount", "impact_bps_per_1pct"]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _matching_rule(
    rules: pd.DataFrame,
    *,
    eval_year: int,
    signal: str,
    top_n: int,
    constraint_variant: str,
    exposure_penalty_strength: float,
    fee_bps: float,
    capital_amount: float,
    impact_bps_per_1pct: float,
) -> dict[str, Any] | None:
    if rules.empty:
        return None
    subset = rules.loc[
        rules["eval_year"].eq(int(eval_year))
        & rules["signal"].astype(str).eq(str(signal))
        & rules["top_n"].eq(int(top_n))
        & rules["constraint_variant"].astype(str).eq(str(constraint_variant))
        & np.isclose(rules["exposure_penalty_strength"], float(exposure_penalty_strength))
        & np.isclose(rules["fee_bps"], float(fee_bps))
        & np.isclose(rules["capital_amount"], float(capital_amount))
        & np.isclose(rules["impact_bps_per_1pct"], float(impact_bps_per_1pct))
    ].copy()
    if subset.empty:
        return None
    subset["fit_row_count"] = pd.to_numeric(subset.get("fit_row_count", 0), errors="coerce").fillna(0)
    return subset.sort_values(["fit_row_count", "selected_factor_family"], ascending=[False, True]).iloc[0].to_dict()


def _normalize_factor_family_run_dirs(value: Mapping[str, str | Path] | Sequence[str] | str | None) -> dict[str, Path]:
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


def _factor_family_run_dirs_from_evidence(factor_family_evidence: pd.DataFrame) -> dict[str, Path]:
    if "factor_family_run_dir" not in factor_family_evidence.columns:
        return {}
    frame = factor_family_evidence.dropna(subset=["factor_family", "factor_family_run_dir"]).copy()
    output: dict[str, Path] = {}
    for row in frame.to_dict("records"):
        output.setdefault(str(row["factor_family"]), Path(str(row["factor_family_run_dir"])))
    return output


def _parse_optional_int_tuple(values: Sequence[int] | str | None) -> tuple[int, ...] | None:
    if values is None:
        return None
    raw = values.split(",") if isinstance(values, str) else values
    output = tuple(int(value) for value in raw if str(value).strip())
    return output or None


def _parse_optional_float_tuple(values: Sequence[float] | str | None) -> tuple[float, ...] | None:
    if values is None:
        return None
    raw = values.split(",") if isinstance(values, str) else values
    output = tuple(float(value) for value in raw if str(value).strip())
    return output or None


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
    parser.add_argument("--weak-year-rebuild-run-dir", default=None)
    parser.add_argument("--factor-family-run-dirs", default=None, help="Comma-separated specs such as core=<dir>,expanded=<dir>.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n-values", default=None)
    parser.add_argument("--impact-bps-per-1pct-values", default=None)
    parser.add_argument("--cold-start-family", default=DEFAULT_COLD_START_FAMILY)
    parser.add_argument("--output-constraint-variant", default=DEFAULT_OUTPUT_CONSTRAINT_VARIANT)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_frontier_factor_family_selected_combined(
        weak_year_rebuild_run_dir=args.weak_year_rebuild_run_dir,
        factor_family_run_dirs=args.factor_family_run_dirs,
        output_dir=args.output_dir,
        top_n_values=args.top_n_values,
        impact_bps_per_1pct_values=args.impact_bps_per_1pct_values,
        cold_start_family=args.cold_start_family,
        output_constraint_variant=args.output_constraint_variant,
        write_research_log=bool(args.write_research_log),
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
