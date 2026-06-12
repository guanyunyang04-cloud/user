"""Attribute current frontier failures from combined constraint artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.experiments.frontier_promotion_gate import latest_run_dir


DEFAULT_COMBINED_OUTPUT_ROOT = Path(
    "traditional_quant_research/output/experiments/low_corr_frontier_combined_constraint_audit"
)
DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/frontier_failure_attribution")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/brain/references/research_log/2026-06-03_frontier_failure_attribution.md")

DEFAULT_REQUIRED_IMPACT_BPS = 10.0
DEFAULT_REQUIRED_FEE_BPS = 30.0
DEFAULT_REQUIRED_CAPITAL_AMOUNT = 100_000_000.0
DEFAULT_PERIOD_TYPE = "monthly"
DEFAULT_MIN_YEAR_RETURN = 0.0
DEFAULT_SEVERE_YEAR_RETURN = -0.10
DEFAULT_EXPOSURE_FACTORS = (
    "log_amount_mean_20d_z",
    "neg_volatility_20d_z",
    "momentum_20d_z",
    "turn_xsec_z",
)


def run_frontier_failure_attribution(
    *,
    combined_run_dir: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    required_impact_bps: float = DEFAULT_REQUIRED_IMPACT_BPS,
    required_fee_bps: float = DEFAULT_REQUIRED_FEE_BPS,
    required_capital_amount: float = DEFAULT_REQUIRED_CAPITAL_AMOUNT,
    exposure_factors: Sequence[str] = DEFAULT_EXPOSURE_FACTORS,
    period_type: str = DEFAULT_PERIOD_TYPE,
    min_year_return: float = DEFAULT_MIN_YEAR_RETURN,
    severe_year_return: float = DEFAULT_SEVERE_YEAR_RETURN,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    """Write an attribution report from the latest combined constraint run."""

    combined_dir = Path(combined_run_dir) if combined_run_dir is not None else latest_run_dir(DEFAULT_COMBINED_OUTPUT_ROOT)
    run_id = f"frontier_failure_attribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    frames = read_combined_constraint_outputs(combined_dir)
    yearly = build_yearly_failure_attribution(
        frames["summary"],
        frames["basket_exposure"],
        frames["liquidity_summary"],
        required_impact_bps=required_impact_bps,
        required_fee_bps=required_fee_bps,
        required_capital_amount=required_capital_amount,
        exposure_factors=exposure_factors,
        period_type=period_type,
        min_year_return=min_year_return,
        severe_year_return=severe_year_return,
    )
    signal_summary = summarize_signal_failure_profile(yearly, exposure_factors=exposure_factors)
    exposure_summary = summarize_exposure_failure_profile(yearly, exposure_factors=exposure_factors)
    return_pivot = annualized_return_pivot(yearly)
    result = summarize_failure_attribution(
        yearly,
        signal_summary,
        run_id=run_id,
        combined_run_dir=combined_dir,
        required_impact_bps=required_impact_bps,
        required_fee_bps=required_fee_bps,
        required_capital_amount=required_capital_amount,
        period_type=period_type,
    )
    markdown = render_failure_attribution_markdown(result, signal_summary, yearly, exposure_summary, return_pivot)

    yearly.to_csv(run_dir / "yearly_failure_attribution.csv", index=False, encoding="utf-8-sig")
    signal_summary.to_csv(run_dir / "signal_failure_summary.csv", index=False, encoding="utf-8-sig")
    exposure_summary.to_csv(run_dir / "exposure_failure_summary.csv", index=False, encoding="utf-8-sig")
    return_pivot.to_csv(run_dir / "annualized_return_pivot.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        path = Path(research_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    return {**result, "run_dir": str(run_dir), "research_log": str(research_log_path) if write_research_log else None}


def read_combined_constraint_outputs(combined_run_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read the artifact tables needed for failure attribution."""

    root = Path(combined_run_dir)
    paths = {
        "summary": root / "combined_constraint_summary.csv",
        "basket_exposure": root / "combined_constraint_basket_exposure.csv",
        "liquidity_summary": root / "combined_constraint_liquidity_summary.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing combined constraint artifacts: {missing}")
    return {name: pd.read_csv(path) for name, path in paths.items()}


def build_yearly_failure_attribution(
    summary: pd.DataFrame,
    basket_exposure: pd.DataFrame,
    liquidity_summary: pd.DataFrame,
    *,
    required_impact_bps: float,
    required_fee_bps: float,
    required_capital_amount: float,
    exposure_factors: Sequence[str],
    period_type: str,
    min_year_return: float,
    severe_year_return: float,
) -> pd.DataFrame:
    """Return per-year rows enriched with exposure and liquidity diagnostics."""

    required = {
        "eval_year",
        "signal",
        "exposure_penalty_strength",
        "fee_bps",
        "capital_amount",
        "impact_bps_per_1pct",
        "annualized_return",
        "max_drawdown",
        "periods",
        "mean_turnover",
        "mean_gross_return",
        "mean_net_return",
        "mean_fee_cost",
        "mean_impact_cost",
        "mean_total_cost",
    }
    if missing := sorted(required - set(summary.columns)):
        raise ValueError(f"summary missing required columns: {missing}")
    if not exposure_factors:
        raise ValueError("exposure_factors must not be empty")
    numeric_summary_columns = (required - {"signal"}) | {"blocked_entry_count", "exit_delayed_count"}
    work = _numeric(summary.copy(), numeric_summary_columns)
    stress = work.loc[
        np.isclose(work["fee_bps"], float(required_fee_bps))
        & np.isclose(work["capital_amount"], float(required_capital_amount))
        & np.isclose(work["impact_bps_per_1pct"], float(required_impact_bps))
    ].copy()
    if stress.empty:
        return pd.DataFrame(columns=_yearly_columns(exposure_factors))

    no_impact = work.loc[
        np.isclose(work["fee_bps"], float(required_fee_bps))
        & np.isclose(work["capital_amount"], float(required_capital_amount))
        & np.isclose(work["impact_bps_per_1pct"], 0.0),
        ["eval_year", "signal", "exposure_penalty_strength", "annualized_return"],
    ].rename(columns={"annualized_return": "annualized_return_no_impact"})
    stress = stress.merge(no_impact, on=["eval_year", "signal", "exposure_penalty_strength"], how="left")
    stress["impact_drag_vs_no_impact"] = stress["annualized_return"] - stress["annualized_return_no_impact"]
    stress["weak_year"] = stress["annualized_return"] < float(min_year_return)
    stress["severe_weak_year"] = stress["annualized_return"] <= float(severe_year_return)
    stress["return_rank_within_year"] = stress.groupby("eval_year")["annualized_return"].rank(
        method="first",
        ascending=False,
    )
    stress["cost_drag"] = stress["mean_gross_return"] - stress["mean_net_return"]

    exposures = summarize_yearly_basket_exposure(
        basket_exposure,
        exposure_factors=exposure_factors,
        period_type=period_type,
    )
    if not exposures.empty:
        stress = stress.merge(
            exposures,
            on=["eval_year", "signal", "exposure_penalty_strength"],
            how="left",
        )
    liquidity = prepare_liquidity_summary(liquidity_summary)
    if not liquidity.empty:
        stress = stress.merge(
            liquidity,
            on=["eval_year", "signal", "exposure_penalty_strength"],
            how="left",
        )

    abs_cols = [f"abs_active_{_safe_name(factor)}" for factor in exposure_factors if f"abs_active_{_safe_name(factor)}" in stress.columns]
    if abs_cols:
        stress["mean_abs_style_exposure"] = stress[abs_cols].mean(axis=1)
        stress["max_abs_style_exposure"] = stress[abs_cols].max(axis=1)
    else:
        stress["mean_abs_style_exposure"] = np.nan
        stress["max_abs_style_exposure"] = np.nan

    return stress.reindex(columns=_yearly_columns(exposure_factors)).sort_values(["eval_year", "signal"]).reset_index(drop=True)


def summarize_yearly_basket_exposure(
    basket_exposure: pd.DataFrame,
    *,
    exposure_factors: Sequence[str],
    period_type: str,
) -> pd.DataFrame:
    """Pivot basket active exposures to one row per eval_year/signal."""

    required = {"eval_year", "signal_config", "exposure_penalty_strength", "factor", "period_type", "active_exposure"}
    if basket_exposure.empty:
        return pd.DataFrame()
    if missing := sorted(required - set(basket_exposure.columns)):
        raise ValueError(f"basket_exposure missing required columns: {missing}")
    frame = basket_exposure.copy()
    frame["active_exposure"] = pd.to_numeric(frame["active_exposure"], errors="coerce")
    subset = frame.loc[
        frame["factor"].astype(str).isin(set(exposure_factors))
        & (frame["period_type"].astype(str) == str(period_type))
    ].copy()
    if subset.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in subset.groupby(["eval_year", "signal_config", "exposure_penalty_strength"], sort=True):
        eval_year, signal, strength = keys
        row: dict[str, Any] = {
            "eval_year": int(eval_year),
            "signal": signal,
            "exposure_penalty_strength": float(strength),
        }
        for factor in exposure_factors:
            values = pd.to_numeric(group.loc[group["factor"].astype(str) == factor, "active_exposure"], errors="coerce").dropna()
            safe = _safe_name(factor)
            row[f"active_{safe}"] = float(values.mean()) if not values.empty else np.nan
            row[f"abs_active_{safe}"] = float(values.abs().mean()) if not values.empty else np.nan
            row[f"max_abs_active_{safe}"] = float(values.abs().max()) if not values.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_liquidity_summary(liquidity_summary: pd.DataFrame) -> pd.DataFrame:
    """Normalize liquidity fields for attribution joins."""

    if liquidity_summary.empty:
        return pd.DataFrame()
    required = {"eval_year", "signal", "exposure_penalty_strength"}
    if missing := sorted(required - set(liquidity_summary.columns)):
        raise ValueError(f"liquidity_summary missing required columns: {missing}")
    keep = [
        "eval_year",
        "signal",
        "exposure_penalty_strength",
        "trade_count",
        "mean_selected_count",
        "mean_amount_mean",
        "min_amount_p10",
        "min_amount_min",
        "mean_log_amount_z",
        "mean_momentum_z",
        "mean_participation_p95_100m",
        "worst_participation_max_100m",
    ]
    frame = liquidity_summary.loc[:, [column for column in keep if column in liquidity_summary.columns]].copy()
    return _numeric(frame, set(frame.columns) - {"signal"}).drop_duplicates(["eval_year", "signal", "exposure_penalty_strength"])


def summarize_signal_failure_profile(yearly: pd.DataFrame, *, exposure_factors: Sequence[str]) -> pd.DataFrame:
    """Aggregate yearly attribution into one row per signal."""

    columns = [
        "signal",
        "exposure_penalty_strength",
        "eval_year_count",
        "weak_year_count",
        "severe_weak_year_count",
        "positive_year_rate",
        "mean_annualized_return",
        "min_annualized_return",
        "max_annualized_return",
        "worst_year",
        "best_year",
        "weak_years",
        "total_periods",
        "mean_impact_drag_vs_no_impact",
        "mean_total_cost",
        "mean_turnover",
        "mean_abs_style_exposure",
        "max_abs_style_exposure",
        "first_window_years",
        "first_window_mean_annualized_return",
        "last_window_years",
        "last_window_mean_annualized_return",
        "last_minus_first_window_return",
        "diagnosis",
    ]
    if yearly.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for keys, group in yearly.groupby(["signal", "exposure_penalty_strength"], sort=True):
        signal, strength = keys
        ordered = group.sort_values("eval_year")
        weak = ordered.loc[ordered["weak_year"].fillna(False)]
        severe = ordered.loc[ordered["severe_weak_year"].fillna(False)]
        worst = ordered.loc[pd.to_numeric(ordered["annualized_return"], errors="coerce").idxmin()]
        best = ordered.loc[pd.to_numeric(ordered["annualized_return"], errors="coerce").idxmax()]
        first = ordered.head(3)
        last = ordered.tail(3)
        first_mean = float(pd.to_numeric(first["annualized_return"], errors="coerce").mean()) if not first.empty else np.nan
        last_mean = float(pd.to_numeric(last["annualized_return"], errors="coerce").mean()) if not last.empty else np.nan
        rows.append(
            {
                "signal": signal,
                "exposure_penalty_strength": float(strength),
                "eval_year_count": int(ordered["eval_year"].nunique()),
                "weak_year_count": int(len(weak)),
                "severe_weak_year_count": int(len(severe)),
                "positive_year_rate": float((pd.to_numeric(ordered["annualized_return"], errors="coerce") >= 0).mean()),
                "mean_annualized_return": float(pd.to_numeric(ordered["annualized_return"], errors="coerce").mean()),
                "min_annualized_return": float(pd.to_numeric(ordered["annualized_return"], errors="coerce").min()),
                "max_annualized_return": float(pd.to_numeric(ordered["annualized_return"], errors="coerce").max()),
                "worst_year": int(worst["eval_year"]),
                "best_year": int(best["eval_year"]),
                "weak_years": ",".join(weak["eval_year"].astype(int).astype(str).tolist()),
                "total_periods": int(pd.to_numeric(ordered["periods"], errors="coerce").sum()),
                "mean_impact_drag_vs_no_impact": float(pd.to_numeric(ordered["impact_drag_vs_no_impact"], errors="coerce").mean()),
                "mean_total_cost": float(pd.to_numeric(ordered["mean_total_cost"], errors="coerce").mean()),
                "mean_turnover": float(pd.to_numeric(ordered["mean_turnover"], errors="coerce").mean()),
                "mean_abs_style_exposure": float(pd.to_numeric(ordered["mean_abs_style_exposure"], errors="coerce").mean()),
                "max_abs_style_exposure": float(pd.to_numeric(ordered["max_abs_style_exposure"], errors="coerce").max()),
                "first_window_years": ",".join(first["eval_year"].astype(int).astype(str).tolist()),
                "first_window_mean_annualized_return": first_mean,
                "last_window_years": ",".join(last["eval_year"].astype(int).astype(str).tolist()),
                "last_window_mean_annualized_return": last_mean,
                "last_minus_first_window_return": last_mean - first_mean,
                "diagnosis": _signal_diagnosis(ordered),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("mean_annualized_return", ascending=False).reset_index(drop=True)


def summarize_exposure_failure_profile(yearly: pd.DataFrame, *, exposure_factors: Sequence[str]) -> pd.DataFrame:
    """Compare exposure levels in weak years versus non-weak years."""

    columns = [
        "signal",
        "exposure_penalty_strength",
        "factor",
        "weak_year_count",
        "positive_year_count",
        "weak_mean_active_exposure",
        "positive_mean_active_exposure",
        "active_delta_weak_minus_positive",
        "weak_mean_abs_active_exposure",
        "positive_mean_abs_active_exposure",
        "abs_delta_weak_minus_positive",
    ]
    if yearly.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for keys, group in yearly.groupby(["signal", "exposure_penalty_strength"], sort=True):
        signal, strength = keys
        weak = group.loc[group["weak_year"].fillna(False)]
        positive = group.loc[~group["weak_year"].fillna(False)]
        for factor in exposure_factors:
            safe = _safe_name(factor)
            active_col = f"active_{safe}"
            abs_col = f"abs_active_{safe}"
            if active_col not in group.columns or abs_col not in group.columns:
                continue
            weak_active = _mean_or_nan(weak, active_col)
            positive_active = _mean_or_nan(positive, active_col)
            weak_abs = _mean_or_nan(weak, abs_col)
            positive_abs = _mean_or_nan(positive, abs_col)
            rows.append(
                {
                    "signal": signal,
                    "exposure_penalty_strength": float(strength),
                    "factor": factor,
                    "weak_year_count": int(len(weak)),
                    "positive_year_count": int(len(positive)),
                    "weak_mean_active_exposure": weak_active,
                    "positive_mean_active_exposure": positive_active,
                    "active_delta_weak_minus_positive": weak_active - positive_active,
                    "weak_mean_abs_active_exposure": weak_abs,
                    "positive_mean_abs_active_exposure": positive_abs,
                    "abs_delta_weak_minus_positive": weak_abs - positive_abs,
                }
            )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["signal", "abs_delta_weak_minus_positive"],
        ascending=[True, False],
    ).reset_index(drop=True)


def annualized_return_pivot(yearly: pd.DataFrame) -> pd.DataFrame:
    if yearly.empty:
        return pd.DataFrame()
    pivot = yearly.pivot_table(index="eval_year", columns="signal", values="annualized_return", aggfunc="mean")
    return pivot.reset_index().rename_axis(None, axis=1)


def summarize_failure_attribution(
    yearly: pd.DataFrame,
    signal_summary: pd.DataFrame,
    *,
    run_id: str,
    combined_run_dir: Path,
    required_impact_bps: float,
    required_fee_bps: float,
    required_capital_amount: float,
    period_type: str,
) -> dict[str, Any]:
    if yearly.empty:
        return {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "combined_run_dir": str(combined_run_dir),
            "status": "empty",
            "candidate_count": 0,
            "decision": "no_rows_to_attribute",
        }
    best = signal_summary.sort_values("mean_annualized_return", ascending=False).iloc[0].to_dict()
    worst = yearly.sort_values("annualized_return", ascending=True).iloc[0].to_dict()
    total_weak = int(pd.to_numeric(signal_summary["weak_year_count"], errors="coerce").sum())
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "combined_run_dir": str(combined_run_dir),
        "status": "ok",
        "required_impact_bps": float(required_impact_bps),
        "required_fee_bps": float(required_fee_bps),
        "required_capital_amount": float(required_capital_amount),
        "period_type": period_type,
        "signal_count": int(yearly["signal"].nunique()),
        "eval_year_count": int(yearly["eval_year"].nunique()),
        "total_weak_signal_years": total_weak,
        "best_signal_by_mean_return": str(best.get("signal", "")),
        "best_signal_mean_annualized_return": float(best.get("mean_annualized_return", np.nan)),
        "best_signal_positive_year_rate": float(best.get("positive_year_rate", np.nan)),
        "worst_signal_year": str(worst.get("signal", "")),
        "worst_eval_year": int(worst.get("eval_year", 0)),
        "worst_annualized_return": float(worst.get("annualized_return", np.nan)),
        "candidate_count": 0,
        "decision": "keep_candidate_frontier_backtest_only_rebuild_required",
        "limitations": [
            "This attribution reads existing combined constraint artifacts and does not rerun backtests.",
            "Exposure deltas are descriptive and do not prove causality.",
            "True size/float-size is still absent from the latest PIT snapshot.",
        ],
    }


def render_failure_attribution_markdown(
    result: Mapping[str, Any],
    signal_summary: pd.DataFrame,
    yearly: pd.DataFrame,
    exposure_summary: pd.DataFrame,
    return_pivot: pd.DataFrame,
) -> str:
    worst = yearly.sort_values("annualized_return", ascending=True).head(12) if not yearly.empty else pd.DataFrame()
    lines = [
        "# Frontier Failure Attribution",
        "",
        f"- run_id: `{result.get('run_id', '')}`",
        f"- decision: `{result.get('decision', '')}`",
        f"- combined_run_dir: `{result.get('combined_run_dir', '')}`",
        f"- stress: `{result.get('required_fee_bps')} bps / {result.get('required_capital_amount')} capital / {result.get('required_impact_bps')} bps per 1 pct participation`",
        f"- candidate_count: `{result.get('candidate_count', 0)}`",
        f"- best_signal_by_mean_return: `{result.get('best_signal_by_mean_return', '')}`",
        f"- best_signal_mean_annualized_return: `{_fmt(result.get('best_signal_mean_annualized_return'))}`",
        f"- total_weak_signal_years: `{result.get('total_weak_signal_years', 0)}`",
        "",
        "## Signal Summary",
        "",
        _markdown_table(signal_summary),
        "",
        "## Worst Signal Years",
        "",
        _markdown_table(worst),
        "",
        "## Return Pivot",
        "",
        _markdown_table(return_pivot),
        "",
        "## Weak Vs Positive Exposure",
        "",
        _markdown_table(exposure_summary),
        "",
        "## Interpretation",
        "",
        "The extended frontier no longer fails because the non-overlapping sample is too small; it fails because weak years remain, the positive-year rate is below a strategy-candidate threshold, and style exposure remains high. This report is an attribution layer, not a promotion layer.",
        "",
    ]
    return "\n".join(lines)


def _yearly_columns(exposure_factors: Sequence[str]) -> list[str]:
    base = [
        "eval_year",
        "signal",
        "exposure_penalty_strength",
        "fee_bps",
        "capital_amount",
        "impact_bps_per_1pct",
        "annualized_return",
        "annualized_return_no_impact",
        "impact_drag_vs_no_impact",
        "weak_year",
        "severe_weak_year",
        "return_rank_within_year",
        "max_drawdown",
        "periods",
        "mean_turnover",
        "mean_gross_return",
        "mean_net_return",
        "cost_drag",
        "mean_fee_cost",
        "mean_impact_cost",
        "mean_total_cost",
        "blocked_entry_count",
        "exit_delayed_count",
        "trade_count",
        "mean_selected_count",
        "mean_amount_mean",
        "min_amount_p10",
        "min_amount_min",
        "mean_log_amount_z",
        "mean_momentum_z",
        "mean_participation_p95_100m",
        "worst_participation_max_100m",
        "mean_abs_style_exposure",
        "max_abs_style_exposure",
    ]
    exposure_columns: list[str] = []
    for factor in exposure_factors:
        safe = _safe_name(factor)
        exposure_columns.extend([f"active_{safe}", f"abs_active_{safe}", f"max_abs_active_{safe}"])
    return base + exposure_columns


def _signal_diagnosis(group: pd.DataFrame) -> str:
    weak_count = int(group["weak_year"].fillna(False).sum())
    positive_rate = float((pd.to_numeric(group["annualized_return"], errors="coerce") >= 0).mean())
    max_abs_style = float(pd.to_numeric(group["max_abs_style_exposure"], errors="coerce").max())
    if weak_count and max_abs_style > 0.5:
        return "weak_years_and_style_exposure"
    if weak_count:
        return "weak_years"
    if positive_rate < 1.0:
        return "year_stability"
    return "no_failure_in_attribution_window"


def _numeric(frame: pd.DataFrame, columns: set[str]) -> pd.DataFrame:
    for column in sorted(columns):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _mean_or_nan(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char == "_" else "_" for char in str(value))


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


def _parse_csv_values(spec: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in spec.split(",") if value.strip())
    if not values:
        raise ValueError("expected at least one value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-run-dir", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--required-impact-bps", type=float, default=DEFAULT_REQUIRED_IMPACT_BPS)
    parser.add_argument("--required-fee-bps", type=float, default=DEFAULT_REQUIRED_FEE_BPS)
    parser.add_argument("--required-capital-amount", type=float, default=DEFAULT_REQUIRED_CAPITAL_AMOUNT)
    parser.add_argument("--exposure-factors", default=",".join(DEFAULT_EXPOSURE_FACTORS))
    parser.add_argument("--period-type", default=DEFAULT_PERIOD_TYPE)
    parser.add_argument("--min-year-return", type=float, default=DEFAULT_MIN_YEAR_RETURN)
    parser.add_argument("--severe-year-return", type=float, default=DEFAULT_SEVERE_YEAR_RETURN)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_frontier_failure_attribution(
        combined_run_dir=args.combined_run_dir,
        output_dir=args.output_dir,
        required_impact_bps=args.required_impact_bps,
        required_fee_bps=args.required_fee_bps,
        required_capital_amount=args.required_capital_amount,
        exposure_factors=_parse_csv_values(args.exposure_factors),
        period_type=args.period_type,
        min_year_return=args.min_year_return,
        severe_year_return=args.severe_year_return,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
