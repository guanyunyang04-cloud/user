"""Personal small-capital candidate gate for Baostock-only frontier evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.experiments.frontier_promotion_gate import latest_run_dir


DEFAULT_COMBINED_OUTPUT_ROOT = Path(
    "traditional_quant_research/output/experiments/low_corr_frontier_combined_constraint_audit"
)
DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/frontier_personal_candidate_gate")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-04_frontier_personal_candidate_gate.md")

PERSONAL_BACKTEST_PROMOTION_LEVEL = "personal_backtest_candidate"
PERSONAL_BACKTEST_ONLY_LEVEL = "personal_research/backtest_only"
FORMAL_PERSONAL_GATE_SCOPE = "formal_personal_backtest_candidate_gate"
DIAGNOSTIC_PERSONAL_GATE_SCOPE = "diagnostic_relaxed_personal_gate"

DEFAULT_REQUIRED_FEE_BPS = 30.0
DEFAULT_REQUIRED_IMPACT_BPS_PER_1PCT = 10.0
DEFAULT_PERSONAL_CAPITAL_AMOUNT = 1_000_000.0
DEFAULT_MIN_EVAL_YEAR_COUNT = 10
DEFAULT_REQUIRED_START_YEAR = 2017
DEFAULT_REQUIRED_END_YEAR = 2026
DEFAULT_MIN_TOTAL_PERIODS = 50
DEFAULT_MIN_MEAN_ANNUALIZED_RETURN = 0.05
DEFAULT_MIN_POSITIVE_YEAR_RATE = 0.60
DEFAULT_MIN_WEAKEST_YEAR_ANNUALIZED_RETURN = -0.35
DEFAULT_MAX_WORST_DRAWDOWN = -0.25
DEFAULT_MAX_PROXY_MEAN_ABS_ACTIVE_EXPOSURE = 1.25
DEFAULT_EXPOSURE_FIELDS = (
    "log_amount_mean_20d_z",
    "neg_volatility_20d_z",
    "momentum_20d_z",
    "turn_xsec_z",
)


def run_frontier_personal_candidate_gate(
    *,
    combined_run_dir: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    required_fee_bps: float = DEFAULT_REQUIRED_FEE_BPS,
    required_impact_bps_per_1pct: float = DEFAULT_REQUIRED_IMPACT_BPS_PER_1PCT,
    personal_capital_amount: float = DEFAULT_PERSONAL_CAPITAL_AMOUNT,
    min_eval_year_count: int = DEFAULT_MIN_EVAL_YEAR_COUNT,
    required_start_year: int = DEFAULT_REQUIRED_START_YEAR,
    required_end_year: int = DEFAULT_REQUIRED_END_YEAR,
    min_total_periods: int = DEFAULT_MIN_TOTAL_PERIODS,
    min_mean_annualized_return: float = DEFAULT_MIN_MEAN_ANNUALIZED_RETURN,
    min_positive_year_rate: float = DEFAULT_MIN_POSITIVE_YEAR_RATE,
    min_weakest_year_annualized_return: float = DEFAULT_MIN_WEAKEST_YEAR_ANNUALIZED_RETURN,
    max_worst_drawdown: float = DEFAULT_MAX_WORST_DRAWDOWN,
    max_proxy_mean_abs_active_exposure: float = DEFAULT_MAX_PROXY_MEAN_ABS_ACTIVE_EXPOSURE,
    exposure_fields: Sequence[str] = DEFAULT_EXPOSURE_FIELDS,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    """Evaluate existing combined-constraint evidence under personal-trading standards."""

    run_id = f"frontier_personal_candidate_gate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    combined_dir = Path(combined_run_dir) if combined_run_dir is not None else latest_run_dir(DEFAULT_COMBINED_OUTPUT_ROOT)
    aggregate, exposure_summary, meta, combined_summary = read_combined_constraint_evidence(combined_dir)
    evidence_scope, formal_gate_profile, gate_profile_detail = infer_personal_gate_profile(
        required_fee_bps=required_fee_bps,
        required_impact_bps_per_1pct=required_impact_bps_per_1pct,
        personal_capital_amount=personal_capital_amount,
        min_eval_year_count=min_eval_year_count,
        required_start_year=required_start_year,
        required_end_year=required_end_year,
        min_total_periods=min_total_periods,
        min_mean_annualized_return=min_mean_annualized_return,
        min_positive_year_rate=min_positive_year_rate,
        min_weakest_year_annualized_return=min_weakest_year_annualized_return,
        max_worst_drawdown=max_worst_drawdown,
        max_proxy_mean_abs_active_exposure=max_proxy_mean_abs_active_exposure,
    )
    gate = evaluate_personal_candidate_gates(
        aggregate,
        exposure_summary,
        meta,
        combined_summary=combined_summary,
        evidence_scope=evidence_scope,
        formal_gate_profile=formal_gate_profile,
        gate_profile_detail=gate_profile_detail,
        required_fee_bps=required_fee_bps,
        required_impact_bps_per_1pct=required_impact_bps_per_1pct,
        personal_capital_amount=personal_capital_amount,
        min_eval_year_count=min_eval_year_count,
        required_start_year=required_start_year,
        required_end_year=required_end_year,
        min_total_periods=min_total_periods,
        min_mean_annualized_return=min_mean_annualized_return,
        min_positive_year_rate=min_positive_year_rate,
        min_weakest_year_annualized_return=min_weakest_year_annualized_return,
        max_worst_drawdown=max_worst_drawdown,
        max_proxy_mean_abs_active_exposure=max_proxy_mean_abs_active_exposure,
        exposure_fields=exposure_fields,
    )
    summary = summarize_personal_candidate_gate(
        gate,
        run_id=run_id,
        combined_run_dir=combined_dir,
        combined_summary=combined_summary,
        required_fee_bps=required_fee_bps,
        required_impact_bps_per_1pct=required_impact_bps_per_1pct,
        personal_capital_amount=personal_capital_amount,
        min_eval_year_count=min_eval_year_count,
        required_start_year=required_start_year,
        required_end_year=required_end_year,
        min_total_periods=min_total_periods,
        min_mean_annualized_return=min_mean_annualized_return,
        min_positive_year_rate=min_positive_year_rate,
        min_weakest_year_annualized_return=min_weakest_year_annualized_return,
        max_worst_drawdown=max_worst_drawdown,
        max_proxy_mean_abs_active_exposure=max_proxy_mean_abs_active_exposure,
        evidence_scope=evidence_scope,
        formal_gate_profile=formal_gate_profile,
        gate_profile_detail=gate_profile_detail,
    )
    markdown = render_personal_candidate_gate_markdown(summary, gate)

    gate.to_csv(run_dir / "personal_candidate_gate_summary.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        path = Path(research_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    return {**summary, "run_dir": str(run_dir), "research_log": str(research_log_path) if write_research_log else None}


def evaluate_personal_candidate_gates(
    aggregate: pd.DataFrame,
    exposure_summary: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    combined_summary: Mapping[str, Any],
    evidence_scope: str | None = None,
    formal_gate_profile: bool | None = None,
    gate_profile_detail: str | None = None,
    required_fee_bps: float,
    required_impact_bps_per_1pct: float,
    personal_capital_amount: float,
    min_eval_year_count: int,
    required_start_year: int,
    required_end_year: int,
    min_total_periods: int,
    min_mean_annualized_return: float,
    min_positive_year_rate: float,
    min_weakest_year_annualized_return: float,
    max_worst_drawdown: float,
    max_proxy_mean_abs_active_exposure: float,
    exposure_fields: Sequence[str],
) -> pd.DataFrame:
    if aggregate.empty:
        return pd.DataFrame(columns=_gate_columns())
    inferred_scope, inferred_formal_profile, inferred_profile_detail = infer_personal_gate_profile(
        required_fee_bps=required_fee_bps,
        required_impact_bps_per_1pct=required_impact_bps_per_1pct,
        personal_capital_amount=personal_capital_amount,
        min_eval_year_count=min_eval_year_count,
        required_start_year=required_start_year,
        required_end_year=required_end_year,
        min_total_periods=min_total_periods,
        min_mean_annualized_return=min_mean_annualized_return,
        min_positive_year_rate=min_positive_year_rate,
        min_weakest_year_annualized_return=min_weakest_year_annualized_return,
        max_worst_drawdown=max_worst_drawdown,
        max_proxy_mean_abs_active_exposure=max_proxy_mean_abs_active_exposure,
    )
    evidence_scope = evidence_scope or inferred_scope
    formal_gate_profile = inferred_formal_profile if formal_gate_profile is None else bool(formal_gate_profile)
    gate_profile_detail = gate_profile_detail or inferred_profile_detail
    frame = aggregate.copy()
    for column in [
        "fee_bps",
        "impact_bps_per_1pct",
        "capital_amount",
        "top_n",
        "eval_year_count",
        "mean_annualized_return",
        "min_annualized_return",
        "positive_year_rate",
        "worst_max_drawdown",
        "total_periods",
        "exposure_penalty_strength",
        "constraint_fallback_count",
        "constraint_fallback_rate",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "constraint_variant" not in frame.columns:
        frame["constraint_variant"] = "baseline"
    if "top_n" not in frame.columns:
        frame["top_n"] = int(combined_summary.get("top_n", 0) or 0)
    if "constraint_fallback_count" not in frame.columns:
        frame["constraint_fallback_count"] = 0
    if "constraint_fallback_rate" not in frame.columns:
        frame["constraint_fallback_rate"] = 0.0
    rows_to_score = frame.loc[
        np.isclose(frame.get("fee_bps", np.nan), required_fee_bps)
        & np.isclose(frame.get("impact_bps_per_1pct", np.nan), required_impact_bps_per_1pct)
    ].copy()

    baostock_gate = _baostock_only_source_gate(combined_summary)
    walk_forward_gate, walk_forward_detail = _walk_forward_gate(
        meta,
        required_start_year=required_start_year,
        required_end_year=required_end_year,
    )
    execution_gate = bool(combined_summary.get("execution_constraints", False))

    rows: list[dict[str, Any]] = []
    for row in rows_to_score.to_dict("records"):
        signal = str(row.get("signal", ""))
        constraint_variant = str(row.get("constraint_variant", "baseline") or "baseline")
        top_n = int(float(row.get("top_n", combined_summary.get("top_n", 0)) or 0))
        strength = float(row.get("exposure_penalty_strength", 0.0) or 0.0)
        fallback_count = int(float(row.get("constraint_fallback_count", 0) or 0))
        fallback_rate = float(row.get("constraint_fallback_rate", 0.0) or 0.0)
        max_proxy_exposure, exposure_failures = _proxy_exposure_evidence(
            exposure_summary,
            signal=signal,
            constraint_variant=constraint_variant,
            top_n=top_n,
            exposure_penalty_strength=strength,
            exposure_fields=exposure_fields,
            max_proxy_mean_abs_active_exposure=max_proxy_mean_abs_active_exposure,
        )
        sample_gate = int(float(row.get("eval_year_count", 0) or 0)) >= min_eval_year_count and int(
            float(row.get("total_periods", 0) or 0)
        ) >= min_total_periods
        formal_profile_failures: list[str] = []
        if not bool(formal_gate_profile) or evidence_scope != FORMAL_PERSONAL_GATE_SCOPE:
            formal_profile_failures.append("threshold_profile")
        if not walk_forward_gate:
            formal_profile_failures.append("walk_forward_gate")
        if not sample_gate:
            formal_profile_failures.append("sample_gate")
        row_formal_profile_gate = len(formal_profile_failures) == 0
        row_evidence_scope = FORMAL_PERSONAL_GATE_SCOPE if row_formal_profile_gate else DIAGNOSTIC_PERSONAL_GATE_SCOPE
        row_gate_profile_detail = (
            gate_profile_detail
            if row_formal_profile_gate
            else f"{gate_profile_detail};nonformal_evidence=" + ",".join(formal_profile_failures)
        )
        checks = {
            "formal_profile_gate": row_formal_profile_gate,
            "baostock_source_gate": baostock_gate,
            "walk_forward_gate": walk_forward_gate,
            "execution_gate": execution_gate,
            "capital_stress_gate": float(row.get("capital_amount", 0.0) or 0.0) >= personal_capital_amount,
            "return_gate": float(row.get("mean_annualized_return", np.nan)) >= min_mean_annualized_return,
            "weak_year_damage_gate": float(row.get("min_annualized_return", np.nan)) >= min_weakest_year_annualized_return
            and float(row.get("positive_year_rate", 0.0) or 0.0) >= min_positive_year_rate,
            "sample_gate": sample_gate,
            "drawdown_gate": float(row.get("worst_max_drawdown", np.nan)) >= max_worst_drawdown,
            "proxy_exposure_sanity_gate": len(exposure_failures) == 0,
            "optimizer_fallback_gate": fallback_count == 0 and fallback_rate == 0.0,
            "explainability_gate": _explainability_gate(row),
        }
        failed = [name for name, passed in checks.items() if not passed]
        rows.append(
            {
                "constraint_variant": constraint_variant,
                "top_n": top_n,
                "signal": signal,
                "fee_bps": float(row.get("fee_bps", np.nan)),
                "capital_amount": float(row.get("capital_amount", np.nan)),
                "impact_bps_per_1pct": float(row.get("impact_bps_per_1pct", np.nan)),
                "personal_capital_amount": float(personal_capital_amount),
                "exposure_penalty_strength": strength,
                "constraint_fallback_count": fallback_count,
                "constraint_fallback_rate": fallback_rate,
                "mean_annualized_return": float(row.get("mean_annualized_return", np.nan)),
                "min_annualized_return": float(row.get("min_annualized_return", np.nan)),
                "positive_year_rate": float(row.get("positive_year_rate", np.nan)),
                "worst_max_drawdown": float(row.get("worst_max_drawdown", np.nan)),
                "total_periods": int(float(row.get("total_periods", 0) or 0)),
                "eval_year_count": int(float(row.get("eval_year_count", 0) or 0)),
                "max_proxy_mean_abs_active_exposure": max_proxy_exposure,
                "formal_profile_gate": checks["formal_profile_gate"],
                "evidence_scope": row_evidence_scope,
                "gate_profile_detail": row_gate_profile_detail,
                "baostock_source_gate": checks["baostock_source_gate"],
                "walk_forward_gate": checks["walk_forward_gate"],
                "execution_gate": checks["execution_gate"],
                "capital_stress_gate": checks["capital_stress_gate"],
                "return_gate": checks["return_gate"],
                "weak_year_damage_gate": checks["weak_year_damage_gate"],
                "sample_gate": checks["sample_gate"],
                "drawdown_gate": checks["drawdown_gate"],
                "proxy_exposure_sanity_gate": checks["proxy_exposure_sanity_gate"],
                "optimizer_fallback_gate": checks["optimizer_fallback_gate"],
                "explainability_gate": checks["explainability_gate"],
                "promoted": not failed,
                "failed_gates": ",".join(failed),
                "exposure_failures": ",".join(exposure_failures),
                "walk_forward_detail": walk_forward_detail,
                "promotion_level": PERSONAL_BACKTEST_PROMOTION_LEVEL if not failed else PERSONAL_BACKTEST_ONLY_LEVEL,
                "paper_tracking_recommendation": "start_paper_tracking" if not failed else "continue_research",
            }
        )
    return pd.DataFrame(rows, columns=_gate_columns())


def summarize_personal_candidate_gate(
    gate: pd.DataFrame,
    *,
    run_id: str,
    combined_run_dir: Path,
    combined_summary: Mapping[str, Any],
    required_fee_bps: float,
    required_impact_bps_per_1pct: float,
    personal_capital_amount: float,
    min_eval_year_count: int,
    required_start_year: int,
    required_end_year: int,
    min_total_periods: int,
    min_mean_annualized_return: float,
    min_positive_year_rate: float,
    min_weakest_year_annualized_return: float,
    max_worst_drawdown: float,
    max_proxy_mean_abs_active_exposure: float,
    evidence_scope: str = FORMAL_PERSONAL_GATE_SCOPE,
    formal_gate_profile: bool = True,
    gate_profile_detail: str = "formal_defaults_or_stricter",
) -> dict[str, Any]:
    candidates = gate.loc[gate["promotion_level"].astype(str).eq(PERSONAL_BACKTEST_PROMOTION_LEVEL)] if not gate.empty else pd.DataFrame()
    fail_counter: Counter[str] = Counter()
    if not gate.empty:
        for item in gate["failed_gates"].dropna().astype(str):
            fail_counter.update(part for part in item.split(",") if part)
    best_row = {}
    if not gate.empty:
        best_row = gate.sort_values(["promoted", "mean_annualized_return"], ascending=[False, False]).iloc[0].to_dict()
        evidence_scopes = sorted(gate["evidence_scope"].dropna().astype(str).unique().tolist())
        evidence_scope = evidence_scopes[0] if len(evidence_scopes) == 1 else "mixed:" + ",".join(evidence_scopes)
        formal_gate_profile = bool(gate["formal_profile_gate"].fillna(False).astype(bool).all())
        details = sorted(gate["gate_profile_detail"].dropna().astype(str).unique().tolist())
        gate_profile_detail = details[0] if len(details) == 1 else "mixed:" + " | ".join(details)
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "north_star": "Baostock-only personal quant strategy research",
        "combined_run_dir": str(combined_run_dir),
        "snapshot_id": str(combined_summary.get("snapshot_id", "")),
        "required_year_window": f"{required_start_year}-{required_end_year}",
        "required_fee_bps": required_fee_bps,
        "required_impact_bps_per_1pct": required_impact_bps_per_1pct,
        "personal_capital_amount": personal_capital_amount,
        "evidence_scope": evidence_scope,
        "formal_gate_profile": bool(formal_gate_profile),
        "gate_profile_detail": gate_profile_detail,
        "min_eval_year_count": min_eval_year_count,
        "min_total_periods": min_total_periods,
        "min_mean_annualized_return": min_mean_annualized_return,
        "min_positive_year_rate": min_positive_year_rate,
        "min_weakest_year_annualized_return": min_weakest_year_annualized_return,
        "max_worst_drawdown": max_worst_drawdown,
        "max_proxy_mean_abs_active_exposure": max_proxy_mean_abs_active_exposure,
        "evaluated_rows": int(len(gate)),
        "personal_backtest_candidate_count": int(len(candidates)),
        "strategy_candidate_count": 0,
        "decision": "personal_paper_tracking_ready" if len(candidates) else "keep_personal_research_backtest_only",
        "best_signal_by_personal_gate": str(best_row.get("signal", "")),
        "best_top_n_by_personal_gate": int(float(best_row.get("top_n", 0) or 0)) if best_row else None,
        "best_signal_mean_annualized_return": float(best_row.get("mean_annualized_return", np.nan)) if best_row else None,
        "top_failed_gates": dict(fail_counter.most_common()),
        "limitations": [
            "This gate reads existing combined-constraint artifacts and does not rerun backtests.",
            "It is a personal small-capital research gate, not an institutional promotion gate.",
            "Relaxed smoke or threshold-override runs are diagnostic and cannot create personal_backtest_candidate rows.",
            "Baostock-only evidence can justify paper tracking, but not true market-cap neutrality or production deployment.",
        ],
    }


def render_personal_candidate_gate_markdown(summary: Mapping[str, Any], gate: pd.DataFrame) -> str:
    lines = [
        "# Frontier Personal Candidate Gate",
        "",
        f"- run_id: `{summary.get('run_id', '')}`",
        f"- north_star: `{summary.get('north_star', '')}`",
        f"- decision: `{summary.get('decision', '')}`",
        f"- personal_backtest_candidate_count: `{summary.get('personal_backtest_candidate_count', 0)}`",
        f"- strategy_candidate_count: `{summary.get('strategy_candidate_count', 0)}`",
        f"- required_year_window: `{summary.get('required_year_window', '')}`",
        f"- required_fee_bps: `{summary.get('required_fee_bps')}`",
        f"- required_impact_bps_per_1pct: `{summary.get('required_impact_bps_per_1pct')}`",
        f"- personal_capital_amount: `{summary.get('personal_capital_amount')}`",
        f"- evidence_scope: `{summary.get('evidence_scope', '')}`",
        f"- formal_gate_profile: `{summary.get('formal_gate_profile', '')}`",
        f"- gate_profile_detail: `{summary.get('gate_profile_detail', '')}`",
        f"- best_signal_by_personal_gate: `{summary.get('best_signal_by_personal_gate', '')}`",
        f"- best_top_n_by_personal_gate: `{summary.get('best_top_n_by_personal_gate', '')}`",
        f"- best_signal_mean_annualized_return: `{_fmt(summary.get('best_signal_mean_annualized_return'))}`",
        "",
        "## Failed Gates",
        "",
    ]
    failed = summary.get("top_failed_gates", {})
    if failed:
        for key, value in failed.items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Gate Summary", "", _markdown_table(gate), "", "## Interpretation", ""])
    lines.append(
        "This is a pragmatic personal-trading gate. It keeps the checks that prevent fake profitability "
        "(walk-forward evidence, execution constraints, cost stress, drawdown, weak-year damage, and proxy exposure disclosure) "
        "while explicitly not requiring true market-cap or float-cap data before paper tracking."
    )
    lines.append("")
    return "\n".join(lines)


def read_combined_constraint_evidence(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    aggregate_path = run_dir / "combined_constraint_aggregate.csv"
    if not aggregate_path.exists():
        raise FileNotFoundError(f"aggregate not found: {aggregate_path}")
    exposure_path = run_dir / "combined_constraint_basket_exposure_summary.csv"
    meta_path = run_dir / "combined_constraint_meta.csv"
    summary_path = run_dir / "summary.json"
    aggregate = pd.read_csv(aggregate_path)
    exposure = pd.read_csv(exposure_path) if exposure_path.exists() else pd.DataFrame()
    meta = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig")) if summary_path.exists() else {}
    return aggregate, exposure, meta, summary


def infer_personal_gate_profile(
    *,
    required_fee_bps: float,
    required_impact_bps_per_1pct: float,
    personal_capital_amount: float,
    min_eval_year_count: int,
    required_start_year: int,
    required_end_year: int,
    min_total_periods: int,
    min_mean_annualized_return: float,
    min_positive_year_rate: float,
    min_weakest_year_annualized_return: float,
    max_worst_drawdown: float,
    max_proxy_mean_abs_active_exposure: float,
) -> tuple[str, bool, str]:
    """Classify whether this gate run uses the formal personal promotion profile."""

    checks = {
        "required_fee_bps": float(required_fee_bps) >= DEFAULT_REQUIRED_FEE_BPS,
        "required_impact_bps_per_1pct": float(required_impact_bps_per_1pct) >= DEFAULT_REQUIRED_IMPACT_BPS_PER_1PCT,
        "personal_capital_amount": float(personal_capital_amount) >= DEFAULT_PERSONAL_CAPITAL_AMOUNT,
        "min_eval_year_count": int(min_eval_year_count) >= DEFAULT_MIN_EVAL_YEAR_COUNT,
        "required_start_year": int(required_start_year) <= DEFAULT_REQUIRED_START_YEAR,
        "required_end_year": int(required_end_year) >= DEFAULT_REQUIRED_END_YEAR,
        "min_total_periods": int(min_total_periods) >= DEFAULT_MIN_TOTAL_PERIODS,
        "min_mean_annualized_return": float(min_mean_annualized_return) >= DEFAULT_MIN_MEAN_ANNUALIZED_RETURN,
        "min_positive_year_rate": float(min_positive_year_rate) >= DEFAULT_MIN_POSITIVE_YEAR_RATE,
        "min_weakest_year_annualized_return": float(min_weakest_year_annualized_return)
        >= DEFAULT_MIN_WEAKEST_YEAR_ANNUALIZED_RETURN,
        "max_worst_drawdown": float(max_worst_drawdown) >= DEFAULT_MAX_WORST_DRAWDOWN,
        "max_proxy_mean_abs_active_exposure": float(max_proxy_mean_abs_active_exposure)
        <= DEFAULT_MAX_PROXY_MEAN_ABS_ACTIVE_EXPOSURE,
    }
    relaxed = [name for name, passed in checks.items() if not passed]
    if relaxed:
        return DIAGNOSTIC_PERSONAL_GATE_SCOPE, False, "relaxed_or_nonformal=" + ",".join(relaxed)
    return FORMAL_PERSONAL_GATE_SCOPE, True, "formal_defaults_or_stricter"


def _baostock_only_source_gate(summary: Mapping[str, Any]) -> bool:
    snapshot_id = str(summary.get("snapshot_id", "")).lower()
    return "baostock" in snapshot_id


def _walk_forward_gate(meta: pd.DataFrame, *, required_start_year: int, required_end_year: int) -> tuple[bool, str]:
    if meta.empty:
        return False, "missing_meta"
    work = meta.copy()
    if "eval_year" not in work.columns or "fit_end_date" not in work.columns or "start_date" not in work.columns:
        return False, "missing_walk_forward_columns"
    years = set(pd.to_numeric(work["eval_year"], errors="coerce").dropna().astype(int).tolist())
    required_years = set(range(int(required_start_year), int(required_end_year) + 1))
    missing_years = sorted(required_years - years)
    fit_end = pd.to_datetime(work["fit_end_date"], errors="coerce")
    start = pd.to_datetime(work["start_date"], errors="coerce")
    bad_fit = bool((fit_end >= start).fillna(True).any())
    if missing_years:
        return False, "missing_eval_years=" + ",".join(str(year) for year in missing_years)
    if bad_fit:
        return False, "fit_window_not_prior_to_eval"
    return True, f"covered_years={min(years)}-{max(years)}"


def _proxy_exposure_evidence(
    exposure_summary: pd.DataFrame,
    *,
    signal: str,
    constraint_variant: str,
    top_n: int,
    exposure_penalty_strength: float,
    exposure_fields: Sequence[str],
    max_proxy_mean_abs_active_exposure: float,
) -> tuple[float, list[str]]:
    if exposure_summary.empty:
        return np.nan, list(exposure_fields)
    frame = exposure_summary.copy()
    if "constraint_variant" not in frame.columns:
        frame["constraint_variant"] = "baseline"
    if "top_n" not in frame.columns:
        frame["top_n"] = int(top_n)
    for column in ["exposure_penalty_strength", "mean_abs_active_exposure"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    subset = frame.loc[
        (frame["signal"].astype(str) == signal)
        & (frame["constraint_variant"].astype(str) == constraint_variant)
        & (pd.to_numeric(frame["top_n"], errors="coerce") == int(top_n))
        & np.isclose(frame["exposure_penalty_strength"], exposure_penalty_strength)
        & (frame["period_type"].astype(str) == "monthly")
        & (frame["factor"].astype(str).isin(set(exposure_fields)))
    ].copy()
    if subset.empty:
        return np.nan, list(exposure_fields)
    max_abs = float(subset["mean_abs_active_exposure"].max())
    failures = (
        subset.loc[subset["mean_abs_active_exposure"] > max_proxy_mean_abs_active_exposure, "factor"]
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )
    return max_abs, failures


def _explainability_gate(row: Mapping[str, Any]) -> bool:
    signal = str(row.get("signal", ""))
    return bool(signal) and (
        signal.startswith("multifactor_")
        or bool(str(row.get("exposure_penalty_cols", "")).strip())
        or bool(str(row.get("group_col", "")).strip())
    )


def _gate_columns() -> list[str]:
    return [
        "constraint_variant",
        "top_n",
        "signal",
        "fee_bps",
        "capital_amount",
        "impact_bps_per_1pct",
        "personal_capital_amount",
        "exposure_penalty_strength",
        "constraint_fallback_count",
        "constraint_fallback_rate",
        "mean_annualized_return",
        "min_annualized_return",
        "positive_year_rate",
        "worst_max_drawdown",
        "total_periods",
        "eval_year_count",
        "max_proxy_mean_abs_active_exposure",
        "formal_profile_gate",
        "evidence_scope",
        "gate_profile_detail",
        "baostock_source_gate",
        "walk_forward_gate",
        "execution_gate",
        "capital_stress_gate",
        "return_gate",
        "weak_year_damage_gate",
        "sample_gate",
        "drawdown_gate",
        "proxy_exposure_sanity_gate",
        "optimizer_fallback_gate",
        "explainability_gate",
        "promoted",
        "failed_gates",
        "exposure_failures",
        "walk_forward_detail",
        "promotion_level",
        "paper_tracking_recommendation",
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
    parser.add_argument("--combined-run-dir", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--required-fee-bps", type=float, default=DEFAULT_REQUIRED_FEE_BPS)
    parser.add_argument("--required-impact-bps-per-1pct", type=float, default=DEFAULT_REQUIRED_IMPACT_BPS_PER_1PCT)
    parser.add_argument("--personal-capital-amount", type=float, default=DEFAULT_PERSONAL_CAPITAL_AMOUNT)
    parser.add_argument("--min-eval-year-count", type=int, default=DEFAULT_MIN_EVAL_YEAR_COUNT)
    parser.add_argument("--required-start-year", type=int, default=DEFAULT_REQUIRED_START_YEAR)
    parser.add_argument("--required-end-year", type=int, default=DEFAULT_REQUIRED_END_YEAR)
    parser.add_argument("--min-total-periods", type=int, default=DEFAULT_MIN_TOTAL_PERIODS)
    parser.add_argument("--min-mean-annualized-return", type=float, default=DEFAULT_MIN_MEAN_ANNUALIZED_RETURN)
    parser.add_argument("--min-positive-year-rate", type=float, default=DEFAULT_MIN_POSITIVE_YEAR_RATE)
    parser.add_argument("--min-weakest-year-annualized-return", type=float, default=DEFAULT_MIN_WEAKEST_YEAR_ANNUALIZED_RETURN)
    parser.add_argument("--max-worst-drawdown", type=float, default=DEFAULT_MAX_WORST_DRAWDOWN)
    parser.add_argument("--max-proxy-mean-abs-active-exposure", type=float, default=DEFAULT_MAX_PROXY_MEAN_ABS_ACTIVE_EXPOSURE)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_frontier_personal_candidate_gate(
        combined_run_dir=args.combined_run_dir,
        output_dir=args.output_dir,
        required_fee_bps=args.required_fee_bps,
        required_impact_bps_per_1pct=args.required_impact_bps_per_1pct,
        personal_capital_amount=args.personal_capital_amount,
        min_eval_year_count=args.min_eval_year_count,
        required_start_year=args.required_start_year,
        required_end_year=args.required_end_year,
        min_total_periods=args.min_total_periods,
        min_mean_annualized_return=args.min_mean_annualized_return,
        min_positive_year_rate=args.min_positive_year_rate,
        min_weakest_year_annualized_return=args.min_weakest_year_annualized_return,
        max_worst_drawdown=args.max_worst_drawdown,
        max_proxy_mean_abs_active_exposure=args.max_proxy_mean_abs_active_exposure,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
