"""Run Baostock-only personal small-capital protocol grids for frontier signals."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.experiments.frontier_personal_candidate_gate import (
    DEFAULT_MAX_PROXY_MEAN_ABS_ACTIVE_EXPOSURE,
    DEFAULT_MAX_WORST_DRAWDOWN,
    DEFAULT_MIN_EVAL_YEAR_COUNT,
    DEFAULT_MIN_MEAN_ANNUALIZED_RETURN,
    DEFAULT_MIN_POSITIVE_YEAR_RATE,
    DEFAULT_MIN_TOTAL_PERIODS,
    DEFAULT_MIN_WEAKEST_YEAR_ANNUALIZED_RETURN,
    DEFAULT_PERSONAL_CAPITAL_AMOUNT,
    DEFAULT_REQUIRED_END_YEAR,
    DEFAULT_REQUIRED_FEE_BPS,
    DEFAULT_REQUIRED_IMPACT_BPS_PER_1PCT,
    DEFAULT_REQUIRED_START_YEAR,
    PERSONAL_BACKTEST_PROMOTION_LEVEL,
    run_frontier_personal_candidate_gate,
)
from traditional_quant_research.experiments.frontier_promotion_gate import latest_run_dir
from traditional_quant_research.experiments.frontier_weak_year_rebuild import DEFAULT_OUTPUT_DIR as DEFAULT_WEAK_YEAR_REBUILD_OUTPUT_DIR
from traditional_quant_research.experiments.low_corr_frontier_combined_constraint_audit import (
    DEFAULT_BUFFER_MULTIPLIER,
    DEFAULT_EXPOSURE_COLUMNS,
    DEFAULT_FINAL_END_DATE,
    DEFAULT_FRONTIER_SIGNALS,
    DEFAULT_GROUP_COL,
    DEFAULT_HORIZON,
    DEFAULT_MAX_FACTOR_CORR,
    DEFAULT_MAX_GROUP_WEIGHT,
    DEFAULT_PENALTY_COLS,
    DEFAULT_REBALANCE_FREQUENCY,
    DEFAULT_ROLLING_MIN_PERIODS,
    DEFAULT_ROLLING_WINDOW,
    DEFAULT_SIGNAL_PENALTY_STRENGTHS,
    LABEL_MODES,
    best_combined_rows,
    render_combined_constraint_markdown,
    run_low_corr_frontier_combined_constraint_audit,
    summarize_combined_basket_exposure,
    summarize_combined_constraint_audit,
    summarize_combined_industry_exposure,
)
from traditional_quant_research.research_panel import DEFAULT_FACTOR_SET, FACTOR_SETS, normalize_factor_set


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/frontier_personal_protocol_grid")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-04_frontier_personal_protocol_grid.md")
DEFAULT_PERSONAL_PROTOCOL_YEARS = tuple(range(2017, 2027))
DEFAULT_TOP_N_VALUES = (20, 50, 100)
DEFAULT_FEE_BPS_VALUES = (DEFAULT_REQUIRED_FEE_BPS,)
DEFAULT_CAPITAL_AMOUNTS = (100_000_000.0,)
DEFAULT_IMPACT_BPS_PER_1PCT = (DEFAULT_REQUIRED_IMPACT_BPS_PER_1PCT,)


def run_frontier_personal_protocol_grid(
    *,
    root: str | None = None,
    years: Sequence[int] = DEFAULT_PERSONAL_PROTOCOL_YEARS,
    final_end_date: str | None = DEFAULT_FINAL_END_DATE,
    horizon: int = DEFAULT_HORIZON,
    label_mode: str = "raw",
    factor_set: str | None = DEFAULT_FACTOR_SET,
    max_factor_corr: float = DEFAULT_MAX_FACTOR_CORR,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    rolling_min_periods: int = DEFAULT_ROLLING_MIN_PERIODS,
    signals: Sequence[str] = DEFAULT_FRONTIER_SIGNALS,
    signal_penalty_strengths: Mapping[str, float] | Sequence[str] | str = DEFAULT_SIGNAL_PENALTY_STRENGTHS,
    top_n_values: Sequence[int] = DEFAULT_TOP_N_VALUES,
    rebalance_frequency: str = DEFAULT_REBALANCE_FREQUENCY,
    buffer_multiplier: float = DEFAULT_BUFFER_MULTIPLIER,
    fee_bps_values: Sequence[float] = DEFAULT_FEE_BPS_VALUES,
    capital_amounts: Sequence[float] = DEFAULT_CAPITAL_AMOUNTS,
    impact_bps_per_1pct_values: Sequence[float] = DEFAULT_IMPACT_BPS_PER_1PCT,
    exposure_penalty_cols: Sequence[str] = DEFAULT_PENALTY_COLS,
    exposure_columns: Sequence[str] = DEFAULT_EXPOSURE_COLUMNS,
    exposure_constraint_cols: Sequence[str] | str | None = None,
    max_abs_exposure: float | None = None,
    group_col: str | None = DEFAULT_GROUP_COL,
    max_group_weight: float | None = DEFAULT_MAX_GROUP_WEIGHT,
    execution_constraints: bool = True,
    limit_threshold: float = 0.095,
    include_metrics: bool = True,
    include_industry: bool = True,
    weak_year_rebuild_run_dir: str | Path | None = None,
    factor_pruning_run_dir: str | Path | None = None,
    ml_signal_run_dir: str | Path | None = None,
    constraint_variants: Sequence[str] | str | None = None,
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
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    resume_run_dir: str | Path | None = None,
    write_research_log: bool = False,
    research_log_path: str | Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    """Run a Top-N protocol grid and score every row with the personal gate."""

    selected_years = _parse_int_values(years, name="years")
    selected_top_n = _parse_int_values(top_n_values, name="top_n_values")
    if not selected_years:
        raise ValueError("years must not be empty")
    if not selected_top_n:
        raise ValueError("top_n_values must not be empty")
    if any(value <= 0 for value in selected_top_n):
        raise ValueError("top_n_values must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if label_mode not in LABEL_MODES:
        raise ValueError(f"unsupported label_mode: {label_mode}")
    selected_factor_set = normalize_factor_set(factor_set)

    selected_signals = _parse_str_values(signals, name="signals")
    fee_values = _parse_float_values(fee_bps_values, name="fee_bps_values")
    capital_values = _parse_float_values(capital_amounts, name="capital_amounts")
    impact_values = _parse_float_values(impact_bps_per_1pct_values, name="impact_bps_per_1pct_values")
    penalty_cols = _parse_str_values(exposure_penalty_cols, name="exposure_penalty_cols")
    exposure_cols = _parse_str_values(exposure_columns, name="exposure_columns")
    constraint_cols = _parse_optional_str_values(exposure_constraint_cols)
    selected_variants = _parse_optional_str_values(constraint_variants) if constraint_variants is not None else None
    resolved_weak_year_rebuild_run_dir = _resolve_ml_weak_year_rebuild_run_dir(
        weak_year_rebuild_run_dir,
        ml_signal_run_dir=ml_signal_run_dir,
        constraint_variants=selected_variants,
    )

    if resume_run_dir is not None:
        run_dir = Path(resume_run_dir)
        run_id = run_dir.name
    else:
        run_id = f"frontier_personal_protocol_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    yearly_results = run_yearly_combined_constraint_grid(
        root=root,
        years=selected_years,
        final_end_date=final_end_date,
        horizon=horizon,
        label_mode=label_mode,
        factor_set=selected_factor_set,
        max_factor_corr=max_factor_corr,
        rolling_window=rolling_window,
        rolling_min_periods=rolling_min_periods,
        signals=selected_signals,
        signal_penalty_strengths=signal_penalty_strengths,
        top_n_values=selected_top_n,
        rebalance_frequency=rebalance_frequency,
        buffer_multiplier=buffer_multiplier,
        fee_bps_values=fee_values,
        capital_amounts=capital_values,
        impact_bps_per_1pct_values=impact_values,
        exposure_penalty_cols=penalty_cols,
        exposure_columns=exposure_cols,
        exposure_constraint_cols=constraint_cols,
        max_abs_exposure=max_abs_exposure,
        group_col=group_col,
        max_group_weight=max_group_weight,
        execution_constraints=execution_constraints,
        limit_threshold=limit_threshold,
        include_metrics=include_metrics,
        include_industry=include_industry,
        weak_year_rebuild_run_dir=resolved_weak_year_rebuild_run_dir,
        factor_pruning_run_dir=factor_pruning_run_dir,
        ml_signal_run_dir=ml_signal_run_dir,
        constraint_variants=selected_variants,
        output_dir=run_dir / "yearly_combined_constraint",
        progress_path=run_dir / "personal_protocol_grid_progress.csv",
        resume=resume_run_dir is not None,
    )
    combined_result = merge_yearly_combined_constraint_runs(
        yearly_results,
        output_dir=run_dir / "combined_constraint_merged",
        run_id=f"{run_id}_combined_merged",
        years=selected_years,
        final_end_date=final_end_date,
        horizon=horizon,
        label_mode=label_mode,
        factor_set=selected_factor_set,
        rolling_window=rolling_window,
        rolling_min_periods=rolling_min_periods,
        signals=selected_signals,
        top_n_values=selected_top_n,
        rebalance_frequency=rebalance_frequency,
        buffer_multiplier=buffer_multiplier,
        fee_bps_values=fee_values,
        capital_amounts=capital_values,
        impact_bps_per_1pct_values=impact_values,
        execution_constraints=execution_constraints,
        limit_threshold=limit_threshold,
        include_metrics=include_metrics,
        include_industry=include_industry,
        group_col=group_col,
        max_group_weight=max_group_weight,
        exposure_penalty_cols=penalty_cols,
        exposure_constraint_cols=constraint_cols,
        max_abs_exposure=max_abs_exposure,
        constraint_variants=selected_variants,
        weak_year_rebuild_run_dir=resolved_weak_year_rebuild_run_dir,
        factor_pruning_run_dir=factor_pruning_run_dir,
        ml_signal_run_dir=ml_signal_run_dir,
    )
    combined_run_dir = Path(str(combined_result["output_dir"]))
    personal_gate_result = run_frontier_personal_candidate_gate(
        combined_run_dir=combined_run_dir,
        output_dir=run_dir / "personal_candidate_gate",
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
        write_research_log=False,
    )
    personal_gate_run_dir = Path(str(personal_gate_result["run_dir"]))
    gate_path = personal_gate_run_dir / "personal_candidate_gate_summary.csv"
    gate = pd.read_csv(gate_path) if gate_path.exists() else pd.DataFrame()
    protocol_runs: list[dict[str, Any]] = [
        {
            "combined_result": combined_result,
            "personal_gate_result": personal_gate_result,
            "gate": gate,
        }
    ]

    ledger = rank_protocol_ledger(build_personal_protocol_ledger(protocol_runs))
    top_n_summary = build_top_n_summary(ledger)
    summary = summarize_personal_protocol_grid(
        ledger,
        top_n_summary,
        run_id=run_id,
        run_dir=run_dir,
        years=selected_years,
        final_end_date=final_end_date,
        horizon=horizon,
        factor_set=selected_factor_set,
        rebalance_frequency=rebalance_frequency,
        buffer_multiplier=buffer_multiplier,
        top_n_values=selected_top_n,
        required_fee_bps=required_fee_bps,
        required_impact_bps_per_1pct=required_impact_bps_per_1pct,
        personal_capital_amount=personal_capital_amount,
        ml_signal_run_dir=ml_signal_run_dir,
    )
    markdown = render_personal_protocol_grid_markdown(summary, top_n_summary, ledger)

    ledger.to_csv(run_dir / "personal_protocol_grid_ledger.csv", index=False, encoding="utf-8-sig")
    top_n_summary.to_csv(run_dir / "personal_protocol_grid_top_n_summary.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        path = Path(research_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
    return {**summary, "run_dir": str(run_dir), "research_log": str(research_log_path) if write_research_log else None}


def run_yearly_combined_constraint_grid(
    *,
    root: str | None,
    years: Sequence[int],
    final_end_date: str | None,
    horizon: int,
    label_mode: str,
    factor_set: str = DEFAULT_FACTOR_SET,
    max_factor_corr: float,
    rolling_window: int,
    rolling_min_periods: int,
    signals: Sequence[str],
    signal_penalty_strengths: Mapping[str, float] | Sequence[str] | str,
    top_n_values: Sequence[int],
    rebalance_frequency: str,
    buffer_multiplier: float,
    fee_bps_values: Sequence[float],
    capital_amounts: Sequence[float],
    impact_bps_per_1pct_values: Sequence[float],
    exposure_penalty_cols: Sequence[str],
    exposure_columns: Sequence[str],
    exposure_constraint_cols: Sequence[str],
    max_abs_exposure: float | None,
    group_col: str | None,
    max_group_weight: float | None,
    execution_constraints: bool,
    limit_threshold: float,
    include_metrics: bool,
    include_industry: bool,
    weak_year_rebuild_run_dir: str | Path | None,
    constraint_variants: Sequence[str] | None,
    output_dir: str | Path,
    progress_path: str | Path,
    factor_pruning_run_dir: str | Path | None = None,
    ml_signal_run_dir: str | Path | None = None,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Run each eval year separately so long protocol grids leave resumable evidence."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    progress = Path(progress_path)
    rows: list[dict[str, Any]] = _load_progress_rows(progress) if resume else []
    results: list[dict[str, Any]] = []
    for year in years:
        existing_index = _progress_row_index(rows, int(year))
        if resume and existing_index is not None:
            existing = rows[existing_index]
            combined_run_dir = str(existing.get("combined_run_dir", "")).strip()
            if str(existing.get("status", "")).strip() == "completed" and _combined_run_complete(combined_run_dir):
                results.append(_load_combined_result(combined_run_dir, eval_year=int(year)))
                continue
        started_at = datetime.now().isoformat(timespec="seconds")
        row = {
            "eval_year": int(year),
            "status": "running",
            "started_at": started_at,
            "finished_at": "",
            "combined_run_dir": "",
            "error": "",
        }
        if existing_index is None:
            rows.append(row)
            active_index = len(rows) - 1
        else:
            rows[existing_index].update(row)
            active_index = existing_index
        pd.DataFrame(rows).to_csv(progress, index=False, encoding="utf-8-sig")
        try:
            result = run_low_corr_frontier_combined_constraint_audit(
                root=root,
                years=(int(year),),
                final_end_date=final_end_date,
                horizon=horizon,
                label_mode=label_mode,
                factor_set=factor_set,
                max_factor_corr=max_factor_corr,
                rolling_window=rolling_window,
                rolling_min_periods=rolling_min_periods,
                signals=signals,
                signal_penalty_strengths=signal_penalty_strengths,
                top_n=int(top_n_values[0]),
                top_n_values=top_n_values,
                rebalance_frequency=rebalance_frequency,
                buffer_multiplier=buffer_multiplier,
                fee_bps_values=fee_bps_values,
                capital_amounts=capital_amounts,
                impact_bps_per_1pct_values=impact_bps_per_1pct_values,
                exposure_penalty_cols=exposure_penalty_cols,
                exposure_columns=exposure_columns,
                exposure_constraint_cols=exposure_constraint_cols,
                max_abs_exposure=max_abs_exposure,
                group_col=group_col,
                max_group_weight=max_group_weight,
                execution_constraints=execution_constraints,
                limit_threshold=limit_threshold,
                include_metrics=include_metrics,
                include_industry=include_industry,
                weak_year_rebuild_run_dir=weak_year_rebuild_run_dir,
                factor_pruning_run_dir=factor_pruning_run_dir,
                ml_signal_run_dir=ml_signal_run_dir,
                constraint_variants=constraint_variants,
                output_dir=output_root / f"year_{int(year)}",
                write_research_log=False,
            )
            result = {**result, "eval_year": int(year)}
            results.append(result)
            rows[active_index].update(
                {
                    "status": "completed",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "combined_run_dir": result.get("output_dir", ""),
                }
            )
            pd.DataFrame(rows).to_csv(progress, index=False, encoding="utf-8-sig")
        except Exception as exc:
            rows[active_index].update(
                {
                    "status": "failed",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "error": str(exc),
                }
            )
            pd.DataFrame(rows).to_csv(progress, index=False, encoding="utf-8-sig")
            raise
    return results


def merge_yearly_combined_constraint_runs(
    yearly_results: Sequence[Mapping[str, Any]],
    *,
    output_dir: str | Path,
    run_id: str,
    years: Sequence[int],
    final_end_date: str | None,
    horizon: int,
    label_mode: str,
    factor_set: str = DEFAULT_FACTOR_SET,
    rolling_window: int,
    rolling_min_periods: int,
    signals: Sequence[str],
    top_n_values: Sequence[int],
    rebalance_frequency: str,
    buffer_multiplier: float,
    fee_bps_values: Sequence[float],
    capital_amounts: Sequence[float],
    impact_bps_per_1pct_values: Sequence[float],
    execution_constraints: bool,
    limit_threshold: float,
    include_metrics: bool,
    include_industry: bool,
    group_col: str | None,
    max_group_weight: float | None,
    exposure_penalty_cols: Sequence[str],
    exposure_constraint_cols: Sequence[str],
    max_abs_exposure: float | None,
    constraint_variants: Sequence[str] | None,
    weak_year_rebuild_run_dir: str | Path | None,
    factor_pruning_run_dir: str | Path | None = None,
    ml_signal_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Merge yearly combined-constraint runs into one standard combined evidence directory."""

    if not yearly_results:
        raise ValueError("yearly_results must not be empty")
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    yearly_dirs = [Path(str(result["output_dir"])) for result in yearly_results]
    summary = _concat_csv(yearly_dirs, "combined_constraint_summary.csv")
    if summary.empty:
        raise ValueError("no yearly combined summary rows to merge")
    aggregate = summarize_combined_constraint_audit(summary)
    trades = _concat_csv(yearly_dirs, "combined_constraint_trades.csv")
    liquidity = _concat_csv(yearly_dirs, "combined_constraint_liquidity.csv")
    liquidity_summary = _concat_csv(yearly_dirs, "combined_constraint_liquidity_summary.csv")
    exposure = _concat_csv(yearly_dirs, "combined_constraint_basket_exposure.csv")
    exposure_summary = summarize_combined_basket_exposure(exposure)
    industry_exposure = _concat_csv(yearly_dirs, "combined_constraint_industry_exposure.csv")
    industry_summary = summarize_combined_industry_exposure(industry_exposure)
    metadata = _concat_csv(yearly_dirs, "combined_constraint_meta.csv")

    first = dict(yearly_results[0])
    result = {
        "run_id": run_id,
        "snapshot_id": first.get("snapshot_id"),
        "years": [int(year) for year in years],
        "final_end_date": final_end_date,
        "horizon": int(horizon),
        "label_mode": label_mode,
        "factor_set": factor_set,
        "rolling_window": int(rolling_window),
        "rolling_min_periods": int(rolling_min_periods),
        "signals": list(signals),
        "rebalance_frequency": rebalance_frequency,
        "top_n": int(top_n_values[0]),
        "top_n_values": [int(value) for value in top_n_values],
        "buffer_multiplier": float(buffer_multiplier),
        "fee_bps_values": [float(value) for value in fee_bps_values],
        "capital_amounts": [float(value) for value in capital_amounts],
        "impact_bps_per_1pct_values": [float(value) for value in impact_bps_per_1pct_values],
        "execution_constraints": bool(execution_constraints),
        "limit_threshold": float(limit_threshold),
        "include_metrics": bool(include_metrics),
        "include_industry": bool(include_industry),
        "group_col": group_col or "",
        "max_group_weight": float(max_group_weight) if max_group_weight is not None else None,
        "exposure_penalty_cols": list(exposure_penalty_cols),
        "exposure_constraint_cols": list(exposure_constraint_cols),
        "max_abs_exposure": float(max_abs_exposure) if max_abs_exposure is not None else None,
        "constraint_variants": list(constraint_variants or ["baseline"]),
        "weak_year_rebuild_run_dir": str(weak_year_rebuild_run_dir or ""),
        "factor_pruning_run_dir": str(factor_pruning_run_dir or ""),
        "ml_signal_run_dir": str(ml_signal_run_dir or ""),
        "yearly_run_dirs": [str(path) for path in yearly_dirs],
        "best_30bps_100m_rows": best_combined_rows(aggregate, fee_bps=30.0, capital_amount=100_000_000.0),
        "candidate_count": 0,
        "assessment": "merged yearly combined constraint gate evidence only; no strategy candidate is promoted by this experiment alone",
        "output_dir": str(run_dir),
    }

    summary.to_csv(run_dir / "combined_constraint_summary.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(run_dir / "combined_constraint_aggregate.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(run_dir / "combined_constraint_trades.csv", index=False, encoding="utf-8-sig")
    liquidity.to_csv(run_dir / "combined_constraint_liquidity.csv", index=False, encoding="utf-8-sig")
    liquidity_summary.to_csv(run_dir / "combined_constraint_liquidity_summary.csv", index=False, encoding="utf-8-sig")
    exposure.to_csv(run_dir / "combined_constraint_basket_exposure.csv", index=False, encoding="utf-8-sig")
    exposure_summary.to_csv(run_dir / "combined_constraint_basket_exposure_summary.csv", index=False, encoding="utf-8-sig")
    industry_exposure.to_csv(run_dir / "combined_constraint_industry_exposure.csv", index=False, encoding="utf-8-sig")
    industry_summary.to_csv(run_dir / "combined_constraint_industry_summary.csv", index=False, encoding="utf-8-sig")
    metadata.to_csv(run_dir / "combined_constraint_meta.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_combined_constraint_markdown(result, aggregate, exposure_summary, industry_summary, liquidity_summary)
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    return result


def _concat_csv(run_dirs: Sequence[Path], filename: str) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        path = run_dir / filename
        if path.exists():
            frame = pd.read_csv(path)
            if not frame.empty:
                frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_progress_rows(progress_path: Path) -> list[dict[str, Any]]:
    if not progress_path.exists():
        return []
    frame = pd.read_csv(progress_path)
    return frame.to_dict("records") if not frame.empty else []


def _progress_row_index(rows: Sequence[Mapping[str, Any]], year: int) -> int | None:
    for index, row in enumerate(rows):
        try:
            if int(float(row.get("eval_year", -1))) == int(year):
                return index
        except (TypeError, ValueError):
            continue
    return None


def _combined_run_complete(combined_run_dir: str | Path) -> bool:
    path = Path(str(combined_run_dir))
    return (
        path.exists()
        and (path / "summary.json").exists()
        and (path / "combined_constraint_summary.csv").exists()
        and (path / "combined_constraint_meta.csv").exists()
    )


def _load_combined_result(combined_run_dir: str | Path, *, eval_year: int) -> dict[str, Any]:
    path = Path(str(combined_run_dir))
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8-sig"))
    return {**summary, "output_dir": str(path), "eval_year": int(eval_year)}


def build_personal_protocol_ledger(protocol_runs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Flatten per-Top-N personal gate outputs into one auditable protocol ledger."""

    columns = _ledger_columns()
    rows: list[dict[str, Any]] = []
    for item in protocol_runs:
        combined_result = item.get("combined_result", {}) or {}
        gate_result = item.get("personal_gate_result", {}) or {}
        gate = item.get("gate", pd.DataFrame())
        if not isinstance(gate, pd.DataFrame) or gate.empty:
            continue
        horizon = _optional_int(combined_result.get("horizon"))
        frequency = str(combined_result.get("rebalance_frequency", ""))
        buffer = _optional_float(combined_result.get("buffer_multiplier"))
        combined_run_dir = str(combined_result.get("output_dir", ""))
        personal_gate_run_dir = str(gate_result.get("run_dir", ""))
        for raw in gate.to_dict("records"):
            signal = str(raw.get("signal", ""))
            variant = str(raw.get("constraint_variant", "baseline") or "baseline")
            top_n = _optional_int(raw.get("top_n", combined_result.get("top_n"))) or 0
            strength = _optional_float(raw.get("exposure_penalty_strength"))
            promoted = _truthy(raw.get("promoted", False))
            promotion_level = str(raw.get("promotion_level", ""))
            rows.append(
                {
                    "protocol_id": _protocol_id(
                        top_n=top_n,
                        signal=signal,
                        constraint_variant=variant,
                        exposure_penalty_strength=strength,
                    ),
                    "research_track": "baostock_only_personal_protocol_grid",
                    "top_n": top_n,
                    "horizon": horizon,
                    "rebalance_frequency": frequency,
                    "buffer_multiplier": buffer,
                    "constraint_variant": variant,
                    "signal": signal,
                    "exposure_penalty_strength": strength,
                    "fee_bps": _optional_float(raw.get("fee_bps")),
                    "capital_amount": _optional_float(raw.get("capital_amount")),
                    "impact_bps_per_1pct": _optional_float(raw.get("impact_bps_per_1pct")),
                    "personal_capital_amount": _optional_float(raw.get("personal_capital_amount")),
                    "mean_annualized_return": _optional_float(raw.get("mean_annualized_return")),
                    "min_annualized_return": _optional_float(raw.get("min_annualized_return")),
                    "positive_year_rate": _optional_float(raw.get("positive_year_rate")),
                    "worst_max_drawdown": _optional_float(raw.get("worst_max_drawdown")),
                    "total_periods": _optional_int(raw.get("total_periods")),
                    "eval_year_count": _optional_int(raw.get("eval_year_count")),
                    "max_proxy_mean_abs_active_exposure": _optional_float(raw.get("max_proxy_mean_abs_active_exposure")),
                    "formal_profile_gate": _truthy(
                        raw.get("formal_profile_gate", promotion_level == PERSONAL_BACKTEST_PROMOTION_LEVEL)
                    ),
                    "evidence_scope": str(raw.get("evidence_scope", "")),
                    "gate_profile_detail": str(raw.get("gate_profile_detail", "")),
                    "promoted": promoted,
                    "promotion_level": promotion_level,
                    "paper_tracking_recommendation": str(raw.get("paper_tracking_recommendation", "")),
                    "failed_gates": str(raw.get("failed_gates", "")),
                    "evidence_grade": PERSONAL_BACKTEST_PROMOTION_LEVEL if promotion_level == PERSONAL_BACKTEST_PROMOTION_LEVEL else "personal_research/backtest_only",
                    "strategy_candidate": False,
                    "combined_run_dir": combined_run_dir,
                    "personal_gate_run_dir": personal_gate_run_dir,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def rank_protocol_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame(columns=[*_ledger_columns(), "rank_overall", "rank_within_top_n"])
    frame = ledger.copy()
    for column in [
        "top_n",
        "mean_annualized_return",
        "min_annualized_return",
        "positive_year_rate",
        "worst_max_drawdown",
        "total_periods",
        "eval_year_count",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["promoted"] = frame["promoted"].map(_truthy)
    frame = frame.sort_values(
        [
            "promoted",
            "mean_annualized_return",
            "positive_year_rate",
            "min_annualized_return",
            "worst_max_drawdown",
            "total_periods",
            "top_n",
        ],
        ascending=[False, False, False, False, False, False, True],
    ).reset_index(drop=True)
    frame["rank_overall"] = range(1, len(frame) + 1)
    frame["rank_within_top_n"] = (
        frame.groupby("top_n", sort=False).cumcount() + 1
        if "top_n" in frame.columns
        else range(1, len(frame) + 1)
    )
    return frame


def build_top_n_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "top_n",
        "evaluated_rows",
        "personal_backtest_candidate_count",
        "best_protocol_id",
        "best_signal",
        "best_promotion_level",
        "best_mean_annualized_return",
        "best_min_annualized_return",
        "best_positive_year_rate",
        "best_worst_max_drawdown",
        "decision",
    ]
    if ledger.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for top_n, group in ledger.groupby("top_n", sort=True):
        ranked = rank_protocol_ledger(group)
        best = ranked.iloc[0].to_dict() if not ranked.empty else {}
        candidate_count = int(group["promotion_level"].astype(str).eq(PERSONAL_BACKTEST_PROMOTION_LEVEL).sum())
        rows.append(
            {
                "top_n": int(top_n),
                "evaluated_rows": int(len(group)),
                "personal_backtest_candidate_count": candidate_count,
                "best_protocol_id": str(best.get("protocol_id", "")),
                "best_signal": str(best.get("signal", "")),
                "best_promotion_level": str(best.get("promotion_level", "")),
                "best_mean_annualized_return": _optional_float(best.get("mean_annualized_return")),
                "best_min_annualized_return": _optional_float(best.get("min_annualized_return")),
                "best_positive_year_rate": _optional_float(best.get("positive_year_rate")),
                "best_worst_max_drawdown": _optional_float(best.get("worst_max_drawdown")),
                "decision": "candidate_selected" if candidate_count else "continue_research",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_personal_protocol_grid(
    ledger: pd.DataFrame,
    top_n_summary: pd.DataFrame,
    *,
    run_id: str,
    run_dir: Path,
    years: Sequence[int],
    final_end_date: str | None,
    horizon: int,
    factor_set: str = DEFAULT_FACTOR_SET,
    rebalance_frequency: str,
    buffer_multiplier: float,
    top_n_values: Sequence[int],
    required_fee_bps: float,
    required_impact_bps_per_1pct: float,
    personal_capital_amount: float,
    ml_signal_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    candidate_count = int(ledger["promotion_level"].astype(str).eq(PERSONAL_BACKTEST_PROMOTION_LEVEL).sum()) if not ledger.empty else 0
    best_row = ledger.iloc[0].to_dict() if not ledger.empty else {}
    evidence_scopes = (
        sorted(ledger["evidence_scope"].dropna().astype(str).unique().tolist())
        if "evidence_scope" in ledger.columns and not ledger.empty
        else []
    )
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "north_star": "Baostock-only personal quant strategy research",
        "objective": "compare personal small-capital Top-N protocols under the same walk-forward, cost, and execution gates",
        "years": [int(value) for value in years],
        "final_end_date": final_end_date,
        "horizon": int(horizon),
        "factor_set": factor_set,
        "rebalance_frequency": rebalance_frequency,
        "buffer_multiplier": float(buffer_multiplier),
        "top_n_values": [int(value) for value in top_n_values],
        "required_fee_bps": float(required_fee_bps),
        "required_impact_bps_per_1pct": float(required_impact_bps_per_1pct),
        "personal_capital_amount": float(personal_capital_amount),
        "ml_signal_run_dir": str(ml_signal_run_dir or ""),
        "evaluated_rows": int(len(ledger)),
        "top_n_summary_count": int(len(top_n_summary)),
        "evidence_scopes": evidence_scopes,
        "personal_backtest_candidate_count": candidate_count,
        "personal_paper_candidate_count": 0,
        "strategy_candidate_count": 0,
        "decision": "personal_protocol_candidates_selected" if candidate_count else "keep_personal_research_backtest_only",
        "post_selection_boundary": "agent_selects_models_and_strategies_only; user_handles_risk_recording_and_live_decisions",
        "best_protocol_id": str(best_row.get("protocol_id", "")),
        "best_top_n": _optional_int(best_row.get("top_n")),
        "best_signal": str(best_row.get("signal", "")),
        "best_mean_annualized_return": _optional_float(best_row.get("mean_annualized_return")),
        "output_dir": str(run_dir),
        "limitations": [
            "This grid promotes only to personal_backtest_candidate and never to strategy_candidate.",
            "Relaxed smoke or threshold-override runs remain diagnostic and cannot create selected candidates.",
            "Every row remains Baostock-only evidence; true market-cap and institutional capacity are future enhancement lines.",
            "Passing rows are selected model or strategy candidates for user discretion; agent-side paper tracking, risk recording, and live-decision workflows are out of scope.",
        ],
    }


def render_personal_protocol_grid_markdown(
    summary: Mapping[str, Any],
    top_n_summary: pd.DataFrame,
    ledger: pd.DataFrame,
) -> str:
    lines = [
        "# Frontier Personal Protocol Grid",
        "",
        f"- run_id: `{summary.get('run_id', '')}`",
        f"- north_star: `{summary.get('north_star', '')}`",
        f"- decision: `{summary.get('decision', '')}`",
        f"- top_n_values: `{summary.get('top_n_values')}`",
        f"- protocol: `horizon={summary.get('horizon')}`, `{summary.get('rebalance_frequency')}`, `buffer={summary.get('buffer_multiplier')}`",
        f"- required stress: `{summary.get('required_fee_bps')}` fee bps, `{summary.get('required_impact_bps_per_1pct')}` impact bps per 1 pct, personal capital `{summary.get('personal_capital_amount')}`",
        f"- personal_backtest_candidate_count: `{summary.get('personal_backtest_candidate_count', 0)}`",
        f"- personal_paper_candidate_count: `{summary.get('personal_paper_candidate_count', 0)}`",
        f"- strategy_candidate_count: `{summary.get('strategy_candidate_count', 0)}`",
        f"- post_selection_boundary: `{summary.get('post_selection_boundary', '')}`",
        f"- evidence_scopes: `{summary.get('evidence_scopes', [])}`",
        f"- best_protocol_id: `{summary.get('best_protocol_id', '')}`",
        f"- Artifacts: `{summary.get('output_dir', '')}`",
        "",
        "## Top-N Summary",
        "",
        _markdown_table(top_n_summary),
        "",
        "## Protocol Ledger",
        "",
        _markdown_table(ledger),
        "",
        "## Interpretation",
        "",
        "This experiment compares the current frontier under personal small-capital assumptions. Passing rows are selected candidates for the user to judge after delivery; they are not institutional strategy candidates.",
        "",
    ]
    return "\n".join(lines)


def _ledger_columns() -> list[str]:
    return [
        "protocol_id",
        "research_track",
        "top_n",
        "horizon",
        "rebalance_frequency",
        "buffer_multiplier",
        "constraint_variant",
        "signal",
        "exposure_penalty_strength",
        "fee_bps",
        "capital_amount",
        "impact_bps_per_1pct",
        "personal_capital_amount",
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
        "promoted",
        "promotion_level",
        "paper_tracking_recommendation",
        "failed_gates",
        "evidence_grade",
        "strategy_candidate",
        "combined_run_dir",
        "personal_gate_run_dir",
    ]


def _protocol_id(
    *,
    top_n: int,
    signal: str,
    constraint_variant: str,
    exposure_penalty_strength: float | None,
) -> str:
    return (
        f"top{int(top_n)}_"
        f"{_safe_token(signal)}_"
        f"{_safe_token(constraint_variant)}_"
        f"penalty_{_strength_token(exposure_penalty_strength)}"
    )


def _safe_token(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value).lower()).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "na"


def _strength_token(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "na"
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "neg_").replace(".", "_")


def _parse_str_values(values: Sequence[str] | str, *, name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        parts = values.split(",")
    else:
        parts = list(values)
    parsed = tuple(str(value).strip() for value in parts if str(value).strip())
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    return parsed


def _parse_optional_str_values(values: Sequence[str] | str | None) -> tuple[str, ...]:
    if values is None:
        return tuple()
    if isinstance(values, str):
        parts = values.split(",")
    else:
        parts = list(values)
    return tuple(str(value).strip() for value in parts if str(value).strip())


def _resolve_ml_weak_year_rebuild_run_dir(
    weak_year_rebuild_run_dir: str | Path | None,
    *,
    ml_signal_run_dir: str | Path | None,
    constraint_variants: Sequence[str] | None,
) -> str | Path | None:
    if weak_year_rebuild_run_dir is not None or ml_signal_run_dir is None:
        return weak_year_rebuild_run_dir
    variants = set(constraint_variants or ())
    if variants & {"regime_gated", "capital_scaled", "factor_blend"}:
        return latest_run_dir(DEFAULT_WEAK_YEAR_REBUILD_OUTPUT_DIR)
    return weak_year_rebuild_run_dir


def _parse_int_values(values: Sequence[int] | str, *, name: str) -> tuple[int, ...]:
    if isinstance(values, str):
        parts = values.split(",")
    else:
        parts = list(values)
    parsed = tuple(int(str(value).strip()) for value in parts if str(value).strip())
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    return parsed


def _parse_float_values(values: Sequence[float] | str, *, name: str) -> tuple[float, ...]:
    if isinstance(values, str):
        parts = values.split(",")
    else:
        parts = list(values)
    parsed = tuple(float(str(value).strip()) for value in parts if str(value).strip())
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    return parsed


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


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "passed"}


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 40) -> str:
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
    parser.add_argument("--root", default=None)
    parser.add_argument("--years", default=",".join(str(year) for year in DEFAULT_PERSONAL_PROTOCOL_YEARS))
    parser.add_argument("--final-end-date", default=DEFAULT_FINAL_END_DATE)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="raw")
    parser.add_argument("--factor-set", choices=FACTOR_SETS, default=DEFAULT_FACTOR_SET)
    parser.add_argument("--max-factor-corr", type=float, default=DEFAULT_MAX_FACTOR_CORR)
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW)
    parser.add_argument("--rolling-min-periods", type=int, default=DEFAULT_ROLLING_MIN_PERIODS)
    parser.add_argument("--signals", default=",".join(DEFAULT_FRONTIER_SIGNALS))
    parser.add_argument("--signal-penalty-strengths", default=",".join(DEFAULT_SIGNAL_PENALTY_STRENGTHS))
    parser.add_argument("--top-n-values", default=",".join(str(value) for value in DEFAULT_TOP_N_VALUES))
    parser.add_argument("--frequency", default=DEFAULT_REBALANCE_FREQUENCY)
    parser.add_argument("--buffer-multiplier", type=float, default=DEFAULT_BUFFER_MULTIPLIER)
    parser.add_argument("--fee-bps", default=",".join(str(value) for value in DEFAULT_FEE_BPS_VALUES))
    parser.add_argument("--capital-amounts", default=",".join(str(value) for value in DEFAULT_CAPITAL_AMOUNTS))
    parser.add_argument("--impact-bps-per-1pct", default=",".join(str(value) for value in DEFAULT_IMPACT_BPS_PER_1PCT))
    parser.add_argument("--exposure-penalty-cols", default=",".join(DEFAULT_PENALTY_COLS))
    parser.add_argument("--exposure-columns", default=",".join(DEFAULT_EXPOSURE_COLUMNS))
    parser.add_argument("--exposure-constraint-cols", default="")
    parser.add_argument("--max-abs-exposure", type=float, default=None)
    parser.add_argument("--weak-year-rebuild-run-dir", type=Path, default=None)
    parser.add_argument("--constraint-variants", default=None)
    parser.add_argument("--group-col", default=DEFAULT_GROUP_COL)
    parser.add_argument("--max-group-weight", type=float, default=DEFAULT_MAX_GROUP_WEIGHT)
    parser.add_argument("--execution-constraints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-threshold", type=float, default=0.095)
    parser.add_argument("--include-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-industry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--required-fee-bps", type=float, default=DEFAULT_REQUIRED_FEE_BPS)
    parser.add_argument("--factor-pruning-run-dir", type=Path, default=None)
    parser.add_argument("--ml-signal-run-dir", type=Path, default=None)
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_frontier_personal_protocol_grid(
        root=args.root,
        years=_parse_int_values(args.years, name="years"),
        final_end_date=args.final_end_date,
        horizon=args.horizon,
        label_mode=args.label_mode,
        factor_set=args.factor_set,
        max_factor_corr=args.max_factor_corr,
        rolling_window=args.rolling_window,
        rolling_min_periods=args.rolling_min_periods,
        signals=_parse_str_values(args.signals, name="signals"),
        signal_penalty_strengths=args.signal_penalty_strengths,
        top_n_values=_parse_int_values(args.top_n_values, name="top_n_values"),
        rebalance_frequency=args.frequency,
        buffer_multiplier=args.buffer_multiplier,
        fee_bps_values=_parse_float_values(args.fee_bps, name="fee_bps_values"),
        capital_amounts=_parse_float_values(args.capital_amounts, name="capital_amounts"),
        impact_bps_per_1pct_values=_parse_float_values(args.impact_bps_per_1pct, name="impact_bps_per_1pct_values"),
        exposure_penalty_cols=_parse_str_values(args.exposure_penalty_cols, name="exposure_penalty_cols"),
        exposure_columns=_parse_str_values(args.exposure_columns, name="exposure_columns"),
        exposure_constraint_cols=_parse_optional_str_values(args.exposure_constraint_cols),
        max_abs_exposure=args.max_abs_exposure,
        group_col=args.group_col.strip() or None,
        max_group_weight=args.max_group_weight,
        execution_constraints=args.execution_constraints,
        limit_threshold=args.limit_threshold,
        include_metrics=args.include_metrics,
        include_industry=args.include_industry,
        weak_year_rebuild_run_dir=args.weak_year_rebuild_run_dir,
        factor_pruning_run_dir=args.factor_pruning_run_dir,
        ml_signal_run_dir=args.ml_signal_run_dir,
        constraint_variants=_parse_optional_str_values(args.constraint_variants) if args.constraint_variants is not None else None,
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
        output_dir=args.output_dir,
        resume_run_dir=args.resume_run_dir,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
