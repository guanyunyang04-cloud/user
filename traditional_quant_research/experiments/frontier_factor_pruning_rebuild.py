"""Build prior-fit factor pruning plans from yearly factor IC evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.experiments.frontier_promotion_gate import latest_run_dir
from traditional_quant_research.experiments.frontier_weak_year_rebuild import DEFAULT_WEAK_YEARS
from traditional_quant_research.experiments.full_cycle_factor_diagnostics import DEFAULT_OUTPUT_DIR as DEFAULT_FACTOR_DIAGNOSTICS_ROOT


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/frontier_factor_pruning_rebuild")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/brain/references/research_log/2026-06-04_frontier_factor_pruning_rebuild.md")
DEFAULT_HORIZON = 20
DEFAULT_EXCLUDE_SIGNALS = ("baseline_score",)


def run_frontier_factor_pruning_rebuild(
    *,
    factor_diagnostics_run_dir: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    horizon: int = DEFAULT_HORIZON,
    max_prior_years: int = 3,
    min_prior_years: int = 1,
    min_abs_mean_rank_ic: float = 0.005,
    min_sign_consistency: float = 0.67,
    min_keep_factors: int = 3,
    max_keep_factors: int = 4,
    exclude_signals: Sequence[str] = DEFAULT_EXCLUDE_SIGNALS,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    """Write leakage-free factor pruning plans for later backtest wiring."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if max_prior_years <= 0:
        raise ValueError("max_prior_years must be positive")
    if min_prior_years <= 0:
        raise ValueError("min_prior_years must be positive")
    if min_keep_factors <= 0:
        raise ValueError("min_keep_factors must be positive")
    if max_keep_factors < min_keep_factors:
        raise ValueError("max_keep_factors must be >= min_keep_factors")

    diagnostics_dir = (
        Path(factor_diagnostics_run_dir) if factor_diagnostics_run_dir is not None else latest_run_dir(DEFAULT_FACTOR_DIAGNOSTICS_ROOT)
    )
    yearly_ic = pd.read_csv(diagnostics_dir / "yearly_ic_summary.csv")

    run_id = f"frontier_factor_pruning_rebuild_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    detail, plan = build_prior_fit_factor_pruning_plan(
        yearly_ic,
        horizon=horizon,
        max_prior_years=max_prior_years,
        min_prior_years=min_prior_years,
        min_abs_mean_rank_ic=min_abs_mean_rank_ic,
        min_sign_consistency=min_sign_consistency,
        min_keep_factors=min_keep_factors,
        max_keep_factors=max_keep_factors,
        exclude_signals=exclude_signals,
    )
    summary = summarize_factor_pruning_rebuild(
        detail,
        plan,
        run_id=run_id,
        factor_diagnostics_run_dir=diagnostics_dir,
        horizon=horizon,
        max_prior_years=max_prior_years,
        min_prior_years=min_prior_years,
        min_abs_mean_rank_ic=min_abs_mean_rank_ic,
        min_sign_consistency=min_sign_consistency,
        min_keep_factors=min_keep_factors,
        max_keep_factors=max_keep_factors,
        exclude_signals=exclude_signals,
    )
    markdown = render_factor_pruning_rebuild_markdown(summary, plan, detail)

    detail.to_csv(run_dir / "factor_pruning_detail.csv", index=False, encoding="utf-8-sig")
    plan.to_csv(run_dir / "factor_pruning_plan.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        path = Path(research_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    return {**summary, "run_dir": str(run_dir), "research_log": str(research_log_path) if write_research_log else None}


def build_prior_fit_factor_pruning_plan(
    yearly_ic: pd.DataFrame,
    *,
    horizon: int = DEFAULT_HORIZON,
    max_prior_years: int = 3,
    min_prior_years: int = 1,
    min_abs_mean_rank_ic: float = 0.005,
    min_sign_consistency: float = 0.67,
    min_keep_factors: int = 3,
    max_keep_factors: int = 4,
    exclude_signals: Sequence[str] = DEFAULT_EXCLUDE_SIGNALS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select factors from prior yearly IC only, returning detail rows and plan rows."""

    detail_columns = _detail_columns()
    plan_columns = _plan_columns()
    if yearly_ic.empty:
        return pd.DataFrame(columns=detail_columns), pd.DataFrame(columns=plan_columns)
    required = {"horizon", "year", "signal", "mean_rank_ic"}
    if missing := sorted(required - set(yearly_ic.columns)):
        raise ValueError(f"yearly_ic missing required columns: {missing}")

    frame = yearly_ic.copy()
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce").astype("Int64")
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame["signal"] = frame["signal"].astype(str)
    frame["mean_rank_ic"] = pd.to_numeric(frame["mean_rank_ic"], errors="coerce")
    frame["rank_icir"] = pd.to_numeric(frame.get("rank_icir", np.nan), errors="coerce")
    exclude = {str(value) for value in exclude_signals}
    frame = frame.loc[frame["horizon"].eq(int(horizon)) & ~frame["signal"].isin(exclude)].dropna(subset=["year", "signal", "mean_rank_ic"])
    if frame.empty:
        return pd.DataFrame(columns=detail_columns), pd.DataFrame(columns=plan_columns)

    years = sorted(frame["year"].dropna().astype(int).unique().tolist())
    all_signals = sorted(frame["signal"].dropna().astype(str).unique().tolist())
    detail_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    for eval_year in years:
        prior_years = [year for year in years if year < eval_year][-max_prior_years:]
        prior = frame.loc[frame["year"].astype(int).isin(prior_years)].copy()
        if prior.empty:
            continue
        scored = _score_prior_factors(
            prior,
            all_signals=all_signals,
            eval_year=eval_year,
            horizon=horizon,
            prior_years=prior_years,
            min_prior_years=min_prior_years,
            min_abs_mean_rank_ic=min_abs_mean_rank_ic,
            min_sign_consistency=min_sign_consistency,
        )
        if scored.empty:
            continue
        selected = _select_pruned_factors(scored, min_keep_factors=min_keep_factors, max_keep_factors=max_keep_factors)
        selected_signals = selected["signal"].astype(str).tolist()
        selected_set = set(selected_signals)
        scored["selected"] = scored["signal"].astype(str).isin(selected_set)
        scored["selection_reason"] = np.where(
            scored["selected"] & scored["keep_by_rule"],
            "prior_rule",
            np.where(scored["selected"], "fallback_min_keep", "dropped"),
        )
        detail_rows.extend(scored.to_dict("records"))
        direction_map = {row["signal"]: int(row["direction"]) for row in selected.to_dict("records")}
        plan_rows.append(
            {
                "eval_year": int(eval_year),
                "horizon": int(horizon),
                "fit_years": ",".join(str(year) for year in prior_years),
                "fit_row_count": int(len(prior)),
                "candidate_signal_name": f"factor_pruned_rank_score_h{horizon}_prior_fit",
                "selected_factors": ",".join(selected_signals),
                "selected_factor_directions": json.dumps(direction_map, ensure_ascii=False, sort_keys=True),
                "selected_factor_count": int(len(selected_signals)),
                "dropped_factors": ",".join(signal for signal in all_signals if signal not in selected_set),
                "dropped_factor_count": int(len([signal for signal in all_signals if signal not in selected_set])),
                "rule_kept_factor_count": int(scored["keep_by_rule"].sum()),
                "fallback_selected_count": int((scored["selected"] & ~scored["keep_by_rule"]).sum()),
                "min_abs_mean_rank_ic": float(min_abs_mean_rank_ic),
                "min_sign_consistency": float(min_sign_consistency),
                "min_keep_factors": int(min_keep_factors),
                "max_keep_factors": int(max_keep_factors),
                "eval_is_fixed_weak_year": int(eval_year) in set(DEFAULT_WEAK_YEARS),
                "fit_uses_eval_year": False,
                "evidence_grade": "diagnostic_not_backtest",
            }
        )
    detail = pd.DataFrame(detail_rows, columns=detail_columns)
    plan = pd.DataFrame(plan_rows, columns=plan_columns)
    return detail, plan


def summarize_factor_pruning_rebuild(
    detail: pd.DataFrame,
    plan: pd.DataFrame,
    *,
    run_id: str,
    factor_diagnostics_run_dir: Path,
    horizon: int,
    max_prior_years: int,
    min_prior_years: int,
    min_abs_mean_rank_ic: float,
    min_sign_consistency: float,
    min_keep_factors: int,
    max_keep_factors: int,
    exclude_signals: Sequence[str],
) -> dict[str, Any]:
    selected_counts = pd.to_numeric(plan.get("selected_factor_count", pd.Series(dtype=float)), errors="coerce")
    fallback_counts = pd.to_numeric(plan.get("fallback_selected_count", pd.Series(dtype=float)), errors="coerce")
    weak_plan = plan.loc[plan.get("eval_is_fixed_weak_year", pd.Series(dtype=bool)).eq(True)] if not plan.empty else pd.DataFrame()
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "factor_diagnostics_run_dir": str(factor_diagnostics_run_dir),
        "horizon": int(horizon),
        "max_prior_years": int(max_prior_years),
        "min_prior_years": int(min_prior_years),
        "min_abs_mean_rank_ic": float(min_abs_mean_rank_ic),
        "min_sign_consistency": float(min_sign_consistency),
        "min_keep_factors": int(min_keep_factors),
        "max_keep_factors": int(max_keep_factors),
        "exclude_signals": list(exclude_signals),
        "detail_rows": int(len(detail)),
        "plan_rows": int(len(plan)),
        "eval_years": sorted(plan["eval_year"].dropna().astype(int).unique().tolist()) if not plan.empty else [],
        "weak_year_plan_rows": int(len(weak_plan)),
        "mean_selected_factor_count": float(selected_counts.mean()) if not selected_counts.empty else np.nan,
        "max_fallback_selected_count": int(fallback_counts.max()) if not fallback_counts.empty else 0,
        "fit_uses_eval_year_count": int(plan["fit_uses_eval_year"].sum()) if not plan.empty else 0,
        "candidate_count": 0,
        "decision": "diagnostic_factor_pruning_plan_ready" if not plan.empty else "insufficient_prior_factor_ic",
        "limitations": [
            "This experiment uses prior yearly IC only and does not rerun multifactor scores or horizon backtests.",
            "Selected factors and directions are a pruning plan, not a promoted strategy signal.",
            "Any pruned signal must be wired into multifactor scoring and rerun through the 2017-2026 formal personal gate before candidate selection.",
        ],
    }


def render_factor_pruning_rebuild_markdown(
    summary: Mapping[str, Any],
    plan: pd.DataFrame,
    detail: pd.DataFrame,
) -> str:
    lines = [
        "# Frontier Factor Pruning Rebuild",
        "",
        f"- run_id: `{summary.get('run_id', '')}`",
        f"- decision: `{summary.get('decision', '')}`",
        f"- horizon: `{summary.get('horizon')}`",
        f"- plan_rows: `{summary.get('plan_rows', 0)}`",
        f"- weak_year_plan_rows: `{summary.get('weak_year_plan_rows', 0)}`",
        f"- mean_selected_factor_count: `{_fmt_float(summary.get('mean_selected_factor_count'))}`",
        f"- max_fallback_selected_count: `{summary.get('max_fallback_selected_count', 0)}`",
        f"- fit_uses_eval_year_count: `{summary.get('fit_uses_eval_year_count', 0)}`",
        f"- candidate_count: `{summary.get('candidate_count', 0)}`",
        "",
        "## Research Interpretation",
        "",
        *_interpretation_lines(summary, plan, detail),
        "",
        "## Factor Pruning Plan",
        "",
        _markdown_table(plan),
        "",
        "## Factor Detail",
        "",
        _markdown_table(detail),
        "",
        "## Interpretation",
        "",
        "This is a model-rebuild planning artifact. It can propose a pruned Baostock-only factor set and prior-fit directions, but it does not upgrade any candidate until the pruned signal is backtested through the formal personal gate.",
        "",
    ]
    return "\n".join(lines)


def _score_prior_factors(
    prior: pd.DataFrame,
    *,
    all_signals: Sequence[str],
    eval_year: int,
    horizon: int,
    prior_years: Sequence[int],
    min_prior_years: int,
    min_abs_mean_rank_ic: float,
    min_sign_consistency: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for signal in all_signals:
        group = prior.loc[prior["signal"].astype(str).eq(str(signal))].copy()
        values = pd.to_numeric(group["mean_rank_ic"], errors="coerce").dropna()
        icir = pd.to_numeric(group.get("rank_icir", pd.Series(dtype=float)), errors="coerce").dropna()
        prior_count = int(group["year"].dropna().nunique())
        mean_rank_ic = float(values.mean()) if not values.empty else np.nan
        abs_mean_rank_ic = abs(mean_rank_ic) if pd.notna(mean_rank_ic) else np.nan
        positive_rate = float((values > 0).mean()) if not values.empty else np.nan
        negative_rate = float((values < 0).mean()) if not values.empty else np.nan
        sign_consistency = float(max(positive_rate, negative_rate)) if pd.notna(positive_rate) and pd.notna(negative_rate) else np.nan
        direction = 1 if pd.isna(mean_rank_ic) or mean_rank_ic >= 0 else -1
        mean_abs_icir = float(icir.abs().mean()) if not icir.empty else np.nan
        keep_by_rule = bool(
            prior_count >= min_prior_years
            and pd.notna(abs_mean_rank_ic)
            and abs_mean_rank_ic >= min_abs_mean_rank_ic
            and pd.notna(sign_consistency)
            and sign_consistency >= min_sign_consistency
        )
        rows.append(
            {
                "eval_year": int(eval_year),
                "horizon": int(horizon),
                "signal": str(signal),
                "fit_years": ",".join(str(year) for year in prior_years),
                "prior_year_count": prior_count,
                "mean_rank_ic": mean_rank_ic,
                "abs_mean_rank_ic": abs_mean_rank_ic,
                "positive_year_rate": positive_rate,
                "negative_year_rate": negative_rate,
                "sign_consistency": sign_consistency,
                "direction": int(direction),
                "mean_abs_rank_icir": mean_abs_icir,
                "keep_by_rule": keep_by_rule,
                "selected": False,
                "selection_reason": "dropped",
                "fit_uses_eval_year": False,
                "evidence_grade": "diagnostic_not_backtest",
            }
        )
    return pd.DataFrame(rows, columns=_detail_columns())


def _select_pruned_factors(scored: pd.DataFrame, *, min_keep_factors: int, max_keep_factors: int) -> pd.DataFrame:
    ranked = scored.sort_values(
        ["keep_by_rule", "sign_consistency", "abs_mean_rank_ic", "mean_abs_rank_icir", "signal"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    selected = ranked.loc[ranked["keep_by_rule"].eq(True)].head(max_keep_factors).copy()
    if len(selected) < min_keep_factors:
        fill_count = min_keep_factors - len(selected)
        fill = ranked.loc[~ranked["signal"].isin(set(selected["signal"]))].head(fill_count)
        selected = pd.concat([selected, fill], ignore_index=True)
    return selected.head(max_keep_factors).copy()


def _interpretation_lines(summary: Mapping[str, Any], plan: pd.DataFrame, detail: pd.DataFrame) -> list[str]:
    if plan.empty:
        return ["- No prior-fit pruning rows were generated."]
    lines = [
        "- The plan uses only years before each eval year to choose factor retention and direction.",
        f"- Generated `{len(plan)}` eval-year plans; fixed weak-year rows: `{summary.get('weak_year_plan_rows', 0)}`.",
    ]
    selected_counts = pd.to_numeric(plan["selected_factor_count"], errors="coerce")
    lines.append(
        f"- Selected factor count ranges from `{int(selected_counts.min())}` to `{int(selected_counts.max())}`, mean `{_fmt_float(selected_counts.mean())}`."
    )
    if not detail.empty:
        selected_detail = detail.loc[detail["selected"].eq(True)].copy()
        counts = selected_detail["signal"].astype(str).value_counts().sort_index()
        lines.append("- Most selected factors: " + ", ".join(f"{signal}:{count}" for signal, count in counts.items()) + ".")
    lines.append("- `fit_uses_eval_year_count` must remain zero; otherwise the pruning plan is invalid.")
    lines.append("- Next required step is to wire `selected_factors` and `selected_factor_directions` into a pruned rank score and rerun formal 2017-2026 personal gates.")
    return lines


def _detail_columns() -> list[str]:
    return [
        "eval_year",
        "horizon",
        "signal",
        "fit_years",
        "prior_year_count",
        "mean_rank_ic",
        "abs_mean_rank_ic",
        "positive_year_rate",
        "negative_year_rate",
        "sign_consistency",
        "direction",
        "mean_abs_rank_icir",
        "keep_by_rule",
        "selected",
        "selection_reason",
        "fit_uses_eval_year",
        "evidence_grade",
    ]


def _plan_columns() -> list[str]:
    return [
        "eval_year",
        "horizon",
        "fit_years",
        "fit_row_count",
        "candidate_signal_name",
        "selected_factors",
        "selected_factor_directions",
        "selected_factor_count",
        "dropped_factors",
        "dropped_factor_count",
        "rule_kept_factor_count",
        "fallback_selected_count",
        "min_abs_mean_rank_ic",
        "min_sign_consistency",
        "min_keep_factors",
        "max_keep_factors",
        "eval_is_fixed_weak_year",
        "fit_uses_eval_year",
        "evidence_grade",
    ]


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(_fmt_float)
    return view.to_markdown(index=False)


def _fmt_float(value: Any) -> str:
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
    parser.add_argument("--factor-diagnostics-run-dir", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--max-prior-years", type=int, default=3)
    parser.add_argument("--min-prior-years", type=int, default=1)
    parser.add_argument("--min-abs-mean-rank-ic", type=float, default=0.005)
    parser.add_argument("--min-sign-consistency", type=float, default=0.67)
    parser.add_argument("--min-keep-factors", type=int, default=3)
    parser.add_argument("--max-keep-factors", type=int, default=4)
    parser.add_argument("--exclude-signals", default=",".join(DEFAULT_EXCLUDE_SIGNALS))
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exclude_signals = tuple(value.strip() for value in args.exclude_signals.split(",") if value.strip())
    result = run_frontier_factor_pruning_rebuild(
        factor_diagnostics_run_dir=args.factor_diagnostics_run_dir,
        output_dir=args.output_dir,
        horizon=args.horizon,
        max_prior_years=args.max_prior_years,
        min_prior_years=args.min_prior_years,
        min_abs_mean_rank_ic=args.min_abs_mean_rank_ic,
        min_sign_consistency=args.min_sign_consistency,
        min_keep_factors=args.min_keep_factors,
        max_keep_factors=args.max_keep_factors,
        exclude_signals=exclude_signals,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
