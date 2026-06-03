"""Audit frontier signals under combined portfolio construction constraints."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.experiments.low_corr_candidate_frontier_audit import (
    DEFAULT_BUFFER_MULTIPLIER,
    DEFAULT_FINAL_END_DATE,
    DEFAULT_HORIZON,
    DEFAULT_MAX_FACTOR_CORR,
    DEFAULT_REBALANCE_FREQUENCY,
    DEFAULT_TOP_N,
    DEFAULT_YEARS,
    LABEL_MODES,
    expand_selected_holdings,
    summarize_trade_liquidity,
)
from traditional_quant_research.experiments.low_corr_candidate_signal_comparison import (
    IC_WEIGHTED_SIGNAL,
    LOW_CORR_SIGNAL,
    ROLLING_IC_SIGNAL,
    build_candidate_protocol_signal_panel,
)
from traditional_quant_research.experiments.low_corr_frontier_impact_stress import (
    apply_fee_and_participation_impact,
)
from traditional_quant_research.experiments.low_corr_frontier_neutralization_audit import (
    DEFAULT_INDUSTRY_COLUMN,
    selected_basket_industry_exposure,
)
from traditional_quant_research.experiments.low_corr_regime_yearly_validation import year_windows
from traditional_quant_research.experiments.multifactor_baseline import (
    selected_basket_factor_exposure,
    summarize_horizon_trade_table,
)
from traditional_quant_research.horizon_backtest import horizon_aligned_top_n_backtest


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/low_corr_frontier_combined_constraint_audit")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-03_low_corr_frontier_combined_constraint_audit.md")
DEFAULT_FRONTIER_SIGNALS = (
    ROLLING_IC_SIGNAL,
    IC_WEIGHTED_SIGNAL,
    LOW_CORR_SIGNAL,
)
DEFAULT_SIGNAL_PENALTY_STRENGTHS = (
    f"{ROLLING_IC_SIGNAL}=0.25",
    f"{IC_WEIGHTED_SIGNAL}=0.0",
    f"{LOW_CORR_SIGNAL}=1.0",
)
DEFAULT_FEE_BPS_VALUES = (30.0,)
DEFAULT_CAPITAL_AMOUNTS = (100_000_000.0,)
DEFAULT_IMPACT_BPS_PER_1PCT = (0.0, 10.0)
DEFAULT_ROLLING_WINDOW = 252
DEFAULT_ROLLING_MIN_PERIODS = 60
DEFAULT_GROUP_COL = DEFAULT_INDUSTRY_COLUMN
DEFAULT_MAX_GROUP_WEIGHT = 0.10
DEFAULT_PENALTY_COLS = (
    "log_amount_mean_20d_z",
    "neg_volatility_20d_z",
    "momentum_20d_z",
    "turn_xsec_z",
)
DEFAULT_EXPOSURE_COLUMNS = (
    "log_amount_mean_20d_z",
    "neg_volatility_20d_z",
    "momentum_20d_z",
    "turn_xsec_z",
    "pctChg_xsec_z",
)
DEFAULT_CONSTRAINT_VARIANTS = ("baseline", "regime_gated", "capital_scaled", "factor_blend")
DEFAULT_BASELINE_VARIANTS = ("baseline",)


def run_low_corr_frontier_combined_constraint_audit(
    *,
    root: str | None = None,
    years: Sequence[int] = DEFAULT_YEARS,
    final_end_date: str | None = DEFAULT_FINAL_END_DATE,
    horizon: int = DEFAULT_HORIZON,
    label_mode: str = "raw",
    max_factor_corr: float = DEFAULT_MAX_FACTOR_CORR,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    rolling_min_periods: int = DEFAULT_ROLLING_MIN_PERIODS,
    signals: Sequence[str] = DEFAULT_FRONTIER_SIGNALS,
    signal_penalty_strengths: Mapping[str, float] | Sequence[str] | str = DEFAULT_SIGNAL_PENALTY_STRENGTHS,
    top_n: int = DEFAULT_TOP_N,
    top_n_values: Sequence[int] | str | None = None,
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
    constraint_variants: Sequence[str] | str | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    """Run frontier signals through exposure penalty, industry cap, and impact stress."""

    if not years:
        raise ValueError("years must not be empty")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if label_mode not in LABEL_MODES:
        raise ValueError(f"unsupported label_mode: {label_mode}")
    selected_top_n = _normalize_int_tuple(top_n_values if top_n_values is not None else (top_n,), name="top_n_values")
    if any(value <= 0 for value in selected_top_n):
        raise ValueError("top_n_values must be positive")
    if buffer_multiplier < 1.0:
        raise ValueError("buffer_multiplier must be at least 1.0")
    if max_group_weight is not None and (max_group_weight <= 0 or max_group_weight > 1):
        raise ValueError("max_group_weight must be between 0 and 1")
    if group_col is None and max_group_weight is not None:
        raise ValueError("group_col is required when max_group_weight is set")
    selected_signals = _normalize_tuple(signals, name="signals")
    fee_values = _normalize_float_tuple(fee_bps_values, name="fee_bps_values")
    capital_values = _normalize_float_tuple(capital_amounts, name="capital_amounts")
    impact_values = _normalize_float_tuple(impact_bps_per_1pct_values, name="impact_bps_per_1pct_values")
    penalty_cols = _normalize_tuple(exposure_penalty_cols, name="exposure_penalty_cols")
    exposure_cols = _normalize_tuple(exposure_columns, name="exposure_columns")
    constraint_cols = _normalize_optional_tuple(exposure_constraint_cols)
    if max_abs_exposure is not None and max_abs_exposure <= 0:
        raise ValueError("max_abs_exposure must be positive")
    if constraint_cols and max_abs_exposure is None:
        raise ValueError("max_abs_exposure is required when exposure_constraint_cols is set")
    if not constraint_cols and max_abs_exposure is not None:
        raise ValueError("exposure_constraint_cols is required when max_abs_exposure is set")
    weak_rules = _read_weak_year_rules(weak_year_rebuild_run_dir)
    selected_variants = _normalize_constraint_variants(
        constraint_variants
        if constraint_variants is not None
        else DEFAULT_CONSTRAINT_VARIANTS
        if weak_year_rebuild_run_dir is not None
        else DEFAULT_BASELINE_VARIANTS,
    )
    strength_by_signal = normalize_signal_penalty_strengths(signal_penalty_strengths, selected_signals)

    run_id = f"low_corr_frontier_combined_constraint_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    liquidity_frames: list[pd.DataFrame] = []
    liquidity_summary_frames: list[pd.DataFrame] = []
    exposure_frames: list[pd.DataFrame] = []
    industry_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    manifest: Mapping[str, Any] = {}
    quality: Mapping[str, Any] = {}

    for year in years:
        windows = year_windows(int(year), final_end_date=final_end_date)
        built = build_candidate_protocol_signal_panel(
            root=root,
            history_start_date=windows["history_start_date"],
            fit_start_date=windows["fit_start_date"],
            fit_end_date=windows["fit_end_date"],
            start_date=windows["start_date"],
            end_date=windows["end_date"],
            horizon=horizon,
            label_mode=label_mode,
            max_factor_corr=max_factor_corr,
            rolling_window=rolling_window,
            rolling_min_periods=rolling_min_periods,
            include_industry=include_industry or group_col is not None,
            include_metrics=include_metrics,
        )
        manifest = built["manifest"]
        quality = built["quality"]
        evaluation_panel = built["evaluation_panel"]
        missing_signals = sorted(set(selected_signals) - set(built["available_signals"]))
        if missing_signals:
            raise ValueError(f"signals not available in panel: {missing_signals}")
        missing_penalty_cols = sorted(set(penalty_cols) - set(evaluation_panel.columns))
        if missing_penalty_cols:
            raise ValueError(f"exposure penalty columns not available in panel: {missing_penalty_cols}")
        missing_constraint_cols = sorted(set(constraint_cols) - set(evaluation_panel.columns))
        if missing_constraint_cols:
            raise ValueError(f"exposure constraint columns not available in panel: {missing_constraint_cols}")
        if group_col is not None and group_col not in evaluation_panel.columns:
            raise ValueError(f"group column not available in panel: {group_col}")
        available_exposures = [column for column in _unique_columns([*exposure_cols, *penalty_cols, *constraint_cols]) if column in evaluation_panel.columns]

        for signal in selected_signals:
            strength = float(strength_by_signal[signal])
            rule = _weak_year_rule_for(weak_rules, eval_year=int(year), signal=signal, exposure_penalty_strength=strength)
            for variant in selected_variants:
                variant_panel, variant_signal, capital_col, variant_meta = _prepare_constraint_variant(
                    evaluation_panel,
                    signal=signal,
                    variant=variant,
                    rule=rule,
                    penalty_cols=penalty_cols,
                )
                for current_top_n in selected_top_n:
                    base_backtest = horizon_aligned_top_n_backtest(
                        variant_panel,
                        variant_signal,
                        horizon=horizon,
                        top_n=int(current_top_n),
                        fee_bps=0.0,
                        rebalance_frequency=rebalance_frequency,
                        buffer_multiplier=buffer_multiplier,
                        execution_constraints=execution_constraints,
                        limit_threshold=limit_threshold,
                        capital_col=capital_col,
                        group_col=group_col,
                        max_group_weight=max_group_weight,
                        exposure_penalty_cols=penalty_cols,
                        exposure_penalty_strength=strength,
                        exposure_constraint_cols=constraint_cols,
                        max_abs_exposure=max_abs_exposure,
                    )
                    base_trades = base_backtest.trades.copy().reset_index(drop=True)
                    if base_trades.empty:
                        continue
                    base_trades.insert(0, "trade_id", range(len(base_trades)))

                    selected_holdings = expand_selected_holdings(
                        variant_panel,
                        base_trades,
                        extra_columns=_holding_extra_columns(variant_panel, variant_signal),
                    )
                    if selected_holdings.empty:
                        per_trade_liquidity = _zero_liquidity_for_trades(base_trades, capital_amounts=capital_values)
                        yearly_liquidity = pd.DataFrame()
                    else:
                        per_trade_liquidity, yearly_liquidity = summarize_trade_liquidity(
                            selected_holdings,
                            capital_amounts=capital_values,
                        )
                    if not per_trade_liquidity.empty:
                        per_trade_liquidity.insert(0, "top_n", int(current_top_n))
                        per_trade_liquidity.insert(0, "constraint_variant", variant)
                        per_trade_liquidity.insert(0, "exposure_penalty_strength", strength)
                        per_trade_liquidity.insert(0, "signal", signal)
                        per_trade_liquidity.insert(0, "eval_year", int(year))
                        liquidity_frames.append(per_trade_liquidity)
                    if not yearly_liquidity.empty:
                        yearly_liquidity.insert(0, "top_n", int(current_top_n))
                        yearly_liquidity.insert(0, "constraint_variant", variant)
                        yearly_liquidity.insert(0, "exposure_penalty_strength", strength)
                        yearly_liquidity.insert(0, "signal", signal)
                        yearly_liquidity.insert(0, "eval_year", int(year))
                        liquidity_summary_frames.append(yearly_liquidity)

                    if available_exposures:
                        exposure = selected_basket_factor_exposure(
                            variant_panel,
                            base_trades,
                            available_exposures,
                            signal_col=variant_signal,
                            horizon=horizon,
                            rebalance_frequency=rebalance_frequency,
                            top_n=int(current_top_n),
                            buffer_multiplier=buffer_multiplier,
                        )
                        if not exposure.empty:
                            _set_or_insert(exposure, "top_n", int(current_top_n))
                            exposure.insert(0, "constraint_variant", variant)
                            exposure.insert(0, "exposure_penalty_strength", strength)
                            exposure.insert(0, "signal_config", signal)
                            exposure.insert(0, "eval_year", int(year))
                            exposure_frames.append(exposure)
                    if group_col is not None:
                        industry_exposure = selected_basket_industry_exposure(
                            variant_panel,
                            base_trades,
                            industry_col=group_col,
                            signal_col=variant_signal,
                            horizon=horizon,
                            rebalance_frequency=rebalance_frequency,
                            top_n=int(current_top_n),
                            buffer_multiplier=buffer_multiplier,
                        )
                        if not industry_exposure.empty:
                            _set_or_insert(industry_exposure, "top_n", int(current_top_n))
                            industry_exposure.insert(0, "constraint_variant", variant)
                            industry_exposure.insert(0, "exposure_penalty_strength", strength)
                            industry_exposure.insert(0, "signal_config", signal)
                            industry_exposure.insert(0, "eval_year", int(year))
                            industry_frames.append(industry_exposure)

                    for fee_bps in fee_values:
                        for capital in capital_values:
                            for impact_bps in impact_values:
                                trades = apply_fee_and_participation_impact(
                                    base_trades,
                                    per_trade_liquidity,
                                    fee_bps=float(fee_bps),
                                    capital_amount=float(capital),
                                    impact_bps_per_1pct=float(impact_bps),
                                )
                                row = {
                                    "eval_year": int(year),
                                    "top_n": int(current_top_n),
                                    "signal": signal,
                                    "constraint_variant": variant,
                                    "evidence_grade": variant_meta["evidence_grade"],
                                    "exposure_penalty_cols": ",".join(penalty_cols),
                                    "exposure_penalty_strength": strength,
                                    "exposure_constraint_cols": ",".join(constraint_cols),
                                    "max_abs_exposure": float(max_abs_exposure) if max_abs_exposure is not None else np.nan,
                                    "group_col": group_col or "",
                                    "max_group_weight": float(max_group_weight) if max_group_weight is not None else np.nan,
                                    "capital_amount": float(capital),
                                    "impact_bps_per_1pct": float(impact_bps),
                                    **variant_meta,
                                }
                                row.update(
                                    summarize_horizon_trade_table(
                                        trades,
                                        horizon=horizon,
                                        rebalance_frequency=rebalance_frequency,
                                        top_n=int(current_top_n),
                                        fee_bps=float(fee_bps),
                                        buffer_multiplier=buffer_multiplier,
                                        execution_constraints=execution_constraints,
                                        limit_threshold=limit_threshold,
                                        capital_col=capital_col,
                                    )
                                )
                                row["top_n"] = int(current_top_n)
                                row["constraint_variant"] = variant
                                row["evidence_grade"] = variant_meta["evidence_grade"]
                                row["exposure_penalty_cols"] = ",".join(penalty_cols)
                                row["exposure_penalty_strength"] = strength
                                row["exposure_constraint_cols"] = ",".join(constraint_cols)
                                row["max_abs_exposure"] = float(max_abs_exposure) if max_abs_exposure is not None else np.nan
                                row["group_col"] = group_col or ""
                                row["max_group_weight"] = float(max_group_weight) if max_group_weight is not None else np.nan
                                row["capital_amount"] = float(capital)
                                row["impact_bps_per_1pct"] = float(impact_bps)
                                row["constraint_fallback_count"] = _sum_bool_column(trades, "constraint_fallback")
                                row["constraint_fallback_rate"] = (
                                    float(trades["constraint_fallback"].fillna(False).astype(bool).mean())
                                    if "constraint_fallback" in trades.columns and not trades.empty
                                    else 0.0
                                )
                                row["mean_fee_cost"] = float(trades["fee_cost"].mean())
                                row["mean_impact_cost"] = float(trades["impact_cost"].mean())
                                row["mean_total_cost"] = float(trades["cost"].mean())
                                row["mean_impact_rate"] = float(trades["impact_rate"].mean())
                                summary_rows.append(row)
                                trades_out = trades.copy()
                                for meta_key, meta_value in reversed(tuple(variant_meta.items())):
                                    trades_out.insert(0, meta_key, meta_value)
                                trades_out.insert(0, "top_n", int(current_top_n))
                                trades_out.insert(0, "constraint_variant", variant)
                                trades_out.insert(0, "exposure_penalty_strength", strength)
                                trades_out.insert(0, "signal", signal)
                                trades_out.insert(0, "eval_year", int(year))
                                trade_frames.append(trades_out)
        metadata_rows.append(
            {
                "eval_year": int(year),
                "history_start_date": windows["history_start_date"],
                "fit_start_date": windows["fit_start_date"],
                "fit_end_date": windows["fit_end_date"],
                "start_date": windows["start_date"],
                "end_date": windows["end_date"],
                "available_signals": json.dumps(list(selected_signals), ensure_ascii=False),
                "top_n_values": json.dumps(list(selected_top_n), ensure_ascii=False),
                "constraint_variants": ",".join(selected_variants),
                "signal_penalty_strengths": json.dumps(strength_by_signal, ensure_ascii=False),
                "exposure_penalty_cols": json.dumps(list(penalty_cols), ensure_ascii=False),
                "exposure_constraint_cols": json.dumps(list(constraint_cols), ensure_ascii=False),
                "max_abs_exposure": float(max_abs_exposure) if max_abs_exposure is not None else np.nan,
                "weak_year_rebuild_run_dir": str(weak_year_rebuild_run_dir or ""),
                "group_col": group_col or "",
                "max_group_weight": float(max_group_weight) if max_group_weight is not None else np.nan,
                "rolling_fallback_rate": float(built["rolling_fallback_rate"]),
                "evaluation_rows": int(len(evaluation_panel)),
                "evaluation_dates": int(evaluation_panel["date"].nunique()) if "date" in evaluation_panel.columns else 0,
                "evaluation_securities": int(evaluation_panel["code"].nunique()) if "code" in evaluation_panel.columns else 0,
            }
        )

    summary = pd.DataFrame(summary_rows)
    aggregate = summarize_combined_constraint_audit(summary)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    liquidity = pd.concat(liquidity_frames, ignore_index=True) if liquidity_frames else pd.DataFrame()
    liquidity_summary = pd.concat(liquidity_summary_frames, ignore_index=True) if liquidity_summary_frames else pd.DataFrame()
    exposure = pd.concat(exposure_frames, ignore_index=True) if exposure_frames else pd.DataFrame()
    exposure_summary = summarize_combined_basket_exposure(exposure)
    industry_exposure = pd.concat(industry_frames, ignore_index=True) if industry_frames else pd.DataFrame()
    industry_summary = summarize_combined_industry_exposure(industry_exposure)
    metadata = pd.DataFrame(metadata_rows)

    result = {
        "run_id": run_id,
        "snapshot_id": manifest.get("snapshot_id"),
        "years": [int(year) for year in years],
        "final_end_date": final_end_date,
        "horizon": int(horizon),
        "label_mode": label_mode,
        "rolling_window": int(rolling_window),
        "rolling_min_periods": int(rolling_min_periods),
        "signals": list(selected_signals),
        "signal_penalty_strengths": strength_by_signal,
        "rebalance_frequency": rebalance_frequency,
        "top_n": int(selected_top_n[0]),
        "top_n_values": [int(value) for value in selected_top_n],
        "buffer_multiplier": float(buffer_multiplier),
        "fee_bps_values": [float(value) for value in fee_values],
        "capital_amounts": [float(value) for value in capital_values],
        "impact_bps_per_1pct_values": [float(value) for value in impact_values],
        "execution_constraints": bool(execution_constraints),
        "limit_threshold": float(limit_threshold),
        "include_metrics": bool(include_metrics),
        "include_industry": bool(include_industry),
        "group_col": group_col or "",
        "max_group_weight": float(max_group_weight) if max_group_weight is not None else None,
        "exposure_penalty_cols": list(penalty_cols),
        "exposure_constraint_cols": list(constraint_cols),
        "max_abs_exposure": float(max_abs_exposure) if max_abs_exposure is not None else None,
        "constraint_variants": list(selected_variants),
        "weak_year_rebuild_run_dir": str(weak_year_rebuild_run_dir or ""),
        "quality": {
            "failure_count": quality.get("failure_count"),
            "missing_bar_rows": quality.get("missing_bar_rows"),
            "st_rows": quality.get("st_rows"),
            "suspended_like_rows": quality.get("suspended_like_rows"),
        },
        "best_30bps_100m_rows": best_combined_rows(aggregate, fee_bps=30.0, capital_amount=100_000_000.0),
        "candidate_count": 0,
        "assessment": "combined constraint gate evidence only; no strategy candidate is promoted by this experiment alone",
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
    if write_research_log:
        research_log_path.write_text(markdown, encoding="utf-8")
    return result


def normalize_signal_penalty_strengths(
    values: Mapping[str, float] | Sequence[str] | str,
    signals: Sequence[str],
) -> dict[str, float]:
    """Return a complete signal->strength mapping."""

    selected_signals = _normalize_tuple(signals, name="signals")
    if isinstance(values, Mapping):
        raw = {str(key).strip(): float(value) for key, value in values.items() if str(key).strip()}
    else:
        parts = values.split(",") if isinstance(values, str) else values
        raw: dict[str, float] = {}
        for part in parts:
            text = str(part).strip()
            if not text:
                continue
            if "=" not in text:
                raise ValueError(f"invalid signal penalty strength spec: {text}")
            signal, strength = text.split("=", 1)
            raw[signal.strip()] = float(strength.strip())
    missing = sorted(set(selected_signals) - set(raw))
    if missing:
        raise ValueError(f"missing penalty strengths for signals: {missing}")
    unknown = sorted(set(raw) - set(selected_signals))
    if unknown:
        raise ValueError(f"penalty strengths contain unknown signals: {unknown}")
    if any(value < 0 for value in raw.values()):
        raise ValueError("penalty strengths must be non-negative")
    return {signal: float(raw[signal]) for signal in selected_signals}


def _read_weak_year_rules(run_dir: str | Path | None) -> pd.DataFrame:
    columns = [
        "eval_year",
        "signal",
        "exposure_penalty_strength",
        "fit_years",
        "fit_row_count",
        "metric",
        "threshold",
        "eval_metric_value",
        "eval_allowed_by_rule",
        "fit_uses_eval_year",
        "evidence_grade",
    ]
    if run_dir is None:
        return pd.DataFrame(columns=columns)
    path = Path(run_dir) / "fit_eval_regime_candidates.csv"
    if not path.exists():
        raise FileNotFoundError(f"weak-year rebuild rules not found: {path}")
    frame = pd.read_csv(path)
    required = {"eval_year", "signal", "exposure_penalty_strength", "fit_uses_eval_year"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"weak-year rules missing required columns: {missing}")
    output = frame.copy()
    output["eval_year"] = pd.to_numeric(output["eval_year"], errors="coerce").astype("Int64")
    output["signal"] = output["signal"].astype(str)
    output["exposure_penalty_strength"] = pd.to_numeric(output["exposure_penalty_strength"], errors="coerce").fillna(0.0)
    output["fit_uses_eval_year"] = _truthy(output["fit_uses_eval_year"])
    if output["fit_uses_eval_year"].any():
        raise ValueError("weak-year rules must be prior-fit; fit_uses_eval_year must be false")
    for column in columns:
        if column not in output.columns:
            output[column] = np.nan if column not in {"signal", "fit_years", "metric", "evidence_grade"} else ""
    return output.loc[:, columns]


def _weak_year_rule_for(
    rules: pd.DataFrame,
    *,
    eval_year: int,
    signal: str,
    exposure_penalty_strength: float,
) -> dict[str, Any] | None:
    if rules.empty:
        return None
    work = rules.copy()
    strength = pd.to_numeric(work["exposure_penalty_strength"], errors="coerce")
    subset = work.loc[
        pd.to_numeric(work["eval_year"], errors="coerce").eq(int(eval_year))
        & work["signal"].astype(str).eq(str(signal))
        & np.isclose(strength, float(exposure_penalty_strength))
    ].copy()
    if subset.empty:
        return None
    subset["fit_row_count"] = pd.to_numeric(subset.get("fit_row_count", 0), errors="coerce").fillna(0)
    subset = subset.sort_values(["fit_row_count", "metric"], ascending=[False, True])
    row = subset.iloc[0].to_dict()
    row["eval_allowed_by_rule"] = bool(_truthy(pd.Series([row.get("eval_allowed_by_rule", False)])).iloc[0])
    row["fit_uses_eval_year"] = bool(_truthy(pd.Series([row.get("fit_uses_eval_year", False)])).iloc[0])
    return row


def _prepare_constraint_variant(
    panel: pd.DataFrame,
    *,
    signal: str,
    variant: str,
    rule: Mapping[str, Any] | None,
    penalty_cols: Sequence[str],
) -> tuple[pd.DataFrame, str, str | None, dict[str, Any]]:
    output = panel.copy()
    allowed = bool(rule.get("eval_allowed_by_rule", True)) if rule is not None else True
    has_rule = rule is not None
    meta = {
        "weak_year_rule_metric": str(rule.get("metric", "")) if rule is not None else "",
        "weak_year_rule_threshold": float(rule.get("threshold", np.nan)) if rule is not None else np.nan,
        "weak_year_rule_allowed": allowed,
        "weak_year_rule_fit_years": str(rule.get("fit_years", "")) if rule is not None else "",
        "weak_year_rule_fit_row_count": int(float(rule.get("fit_row_count", 0) or 0)) if rule is not None else 0,
        "weak_year_rule_eval_metric_value": float(rule.get("eval_metric_value", np.nan)) if rule is not None else np.nan,
        "weak_year_rule_fit_uses_eval_year": bool(rule.get("fit_uses_eval_year", False)) if rule is not None else False,
        "weak_year_capital_scale": 1.0,
        "evidence_grade": "out_of_sample_supported" if has_rule and variant != "baseline" else "backtest_only",
    }
    if variant == "baseline":
        return output, signal, None, meta
    if variant == "regime_gated":
        capital_col = "__frontier_regime_gate_capital"
        output[capital_col] = 1.0 if allowed else 0.0
        meta["weak_year_capital_scale"] = float(output[capital_col].iloc[0]) if not output.empty else 1.0
        return output, signal, capital_col, meta
    if variant == "capital_scaled":
        capital_col = "__frontier_capital_scale"
        output[capital_col] = 1.0 if allowed else 0.5
        meta["weak_year_capital_scale"] = float(output[capital_col].iloc[0]) if not output.empty else 1.0
        return output, signal, capital_col, meta
    if variant == "factor_blend":
        blend_col = f"__frontier_factor_blend_{_safe_signal(signal)}"
        available_factors = [column for column in penalty_cols if column in output.columns]
        output[blend_col] = _factor_blend_signal(output, signal=signal, factor_cols=available_factors)
        return output, blend_col, None, meta
    raise ValueError(f"unsupported constraint variant: {variant}")


def _factor_blend_signal(frame: pd.DataFrame, *, signal: str, factor_cols: Sequence[str]) -> pd.Series:
    work = frame.copy()
    signal_rank = work.groupby("date", sort=False)[signal].rank(pct=True, method="average")
    if not factor_cols:
        return signal_rank
    factor_ranks = []
    for column in factor_cols:
        numeric = pd.to_numeric(work[column], errors="coerce")
        factor_ranks.append(numeric.groupby(work["date"], sort=False).rank(pct=True, method="average"))
    factor_mean = pd.concat(factor_ranks, axis=1).mean(axis=1)
    return (signal_rank + factor_mean) / 2.0


def _zero_liquidity_for_trades(trades: pd.DataFrame, *, capital_amounts: Sequence[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade_id, row in trades.reset_index(drop=True).iterrows():
        output = {
            "trade_id": int(row.get("trade_id", trade_id)),
            "signal_date": row.get("signal_date"),
            "entry_date": row.get("entry_date"),
            "exit_date": row.get("exit_date"),
            "selected_count": 0,
            "amount_mean": np.nan,
            "amount_median": np.nan,
            "amount_p10": np.nan,
            "amount_min": np.nan,
            "volume_mean": np.nan,
            "log_amount_mean_20d_z_mean": np.nan,
            "momentum_20d_z_mean": np.nan,
            "low_corr_score_mean": np.nan,
        }
        for capital in capital_amounts:
            token = _capital_token(float(capital))
            output[f"participation_mean_{token}"] = 0.0
            output[f"participation_p95_{token}"] = 0.0
            output[f"participation_max_{token}"] = 0.0
        rows.append(output)
    return pd.DataFrame(rows)


def summarize_combined_constraint_audit(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate combined gate rows by signal and stress setting."""

    if summary.empty:
        return pd.DataFrame()
    required = {
        "eval_year",
        "signal",
        "fee_bps",
        "capital_amount",
        "impact_bps_per_1pct",
        "exposure_penalty_strength",
        "annualized_return",
        "sharpe",
        "max_drawdown",
        "mean_turnover",
        "mean_impact_cost",
        "periods",
    }
    if missing := sorted(required - set(summary.columns)):
        raise ValueError(f"summary missing required columns: {missing}")
    work = summary.copy()
    if "constraint_variant" not in work.columns:
        work["constraint_variant"] = "baseline"
    if "top_n" not in work.columns:
        work["top_n"] = DEFAULT_TOP_N
    if "constraint_fallback_count" not in work.columns:
        work["constraint_fallback_count"] = 0
    if "constraint_fallback_rate" not in work.columns:
        work["constraint_fallback_rate"] = 0.0
    if "evidence_grade" not in work.columns:
        work["evidence_grade"] = "backtest_only"
    no_impact = work.loc[
        pd.to_numeric(work["impact_bps_per_1pct"], errors="coerce").eq(0.0),
        ["eval_year", "top_n", "signal", "constraint_variant", "fee_bps", "capital_amount", "annualized_return"],
    ].rename(columns={"annualized_return": "annualized_return_no_impact"})
    work = work.merge(no_impact, on=["eval_year", "top_n", "signal", "constraint_variant", "fee_bps", "capital_amount"], how="left")
    low_corr = work.loc[
        work["signal"] == LOW_CORR_SIGNAL,
        ["eval_year", "top_n", "constraint_variant", "fee_bps", "capital_amount", "impact_bps_per_1pct", "annualized_return"],
    ].rename(columns={"annualized_return": "low_corr_annualized_return"})
    work = work.merge(low_corr, on=["eval_year", "top_n", "constraint_variant", "fee_bps", "capital_amount", "impact_bps_per_1pct"], how="left")
    work["impact_drag_vs_no_impact"] = pd.to_numeric(work["annualized_return"], errors="coerce") - pd.to_numeric(
        work["annualized_return_no_impact"],
        errors="coerce",
    )
    work["delta_annualized_return_vs_low_corr"] = pd.to_numeric(work["annualized_return"], errors="coerce") - pd.to_numeric(
        work["low_corr_annualized_return"],
        errors="coerce",
    )

    rows: list[dict[str, Any]] = []
    group_cols = [
        "top_n",
        "constraint_variant",
        "signal",
        "exposure_penalty_cols",
        "exposure_penalty_strength",
        "group_col",
        "max_group_weight",
        "fee_bps",
        "capital_amount",
        "impact_bps_per_1pct",
    ]
    for keys, group in work.groupby(group_cols, sort=True, dropna=False):
        (
            top_n,
            constraint_variant,
            signal,
            penalty_cols,
            strength,
            group_col,
            max_group_weight,
            fee_bps,
            capital,
            impact_bps,
        ) = keys
        returns = pd.to_numeric(group["annualized_return"], errors="coerce")
        deltas = pd.to_numeric(group["delta_annualized_return_vs_low_corr"], errors="coerce")
        evidence_grades = sorted(set(group["evidence_grade"].dropna().astype(str)))
        rows.append(
            {
                "constraint_variant": constraint_variant,
                "top_n": int(top_n),
                "signal": signal,
                "evidence_grade": ",".join(evidence_grades),
                "exposure_penalty_cols": penalty_cols,
                "exposure_penalty_strength": float(strength),
                "group_col": group_col,
                "max_group_weight": float(max_group_weight) if pd.notna(max_group_weight) else np.nan,
                "fee_bps": float(fee_bps),
                "capital_amount": float(capital),
                "impact_bps_per_1pct": float(impact_bps),
                "eval_year_count": int(group["eval_year"].nunique()),
                "mean_annualized_return": float(returns.mean()),
                "min_annualized_return": float(returns.min()),
                "positive_year_rate": float((returns > 0).mean()),
                "mean_sharpe": float(pd.to_numeric(group["sharpe"], errors="coerce").mean()),
                "worst_max_drawdown": float(pd.to_numeric(group["max_drawdown"], errors="coerce").min()),
                "mean_turnover": float(pd.to_numeric(group["mean_turnover"], errors="coerce").mean()),
                "mean_impact_cost": float(pd.to_numeric(group["mean_impact_cost"], errors="coerce").mean()),
                "mean_total_cost": float(pd.to_numeric(group["mean_total_cost"], errors="coerce").mean()),
                "mean_impact_drag_vs_no_impact": float(pd.to_numeric(group["impact_drag_vs_no_impact"], errors="coerce").mean()),
                "mean_delta_annualized_return_vs_low_corr": float(deltas.mean()) if not deltas.dropna().empty else np.nan,
                "positive_delta_year_rate_vs_low_corr": float((deltas > 0).mean()) if not deltas.dropna().empty else np.nan,
                "total_periods": int(pd.to_numeric(group["periods"], errors="coerce").sum()),
                "constraint_fallback_count": int(pd.to_numeric(group["constraint_fallback_count"], errors="coerce").fillna(0).sum()),
                "constraint_fallback_rate": float(pd.to_numeric(group["constraint_fallback_rate"], errors="coerce").fillna(0.0).mean()),
            }
        )
    output = pd.DataFrame(rows).sort_values(
        ["fee_bps", "capital_amount", "impact_bps_per_1pct", "constraint_variant", "mean_annualized_return"],
        ascending=[True, True, True, True, False],
    )
    if output.empty:
        return output
    output["rank_within_stress"] = output.groupby(["top_n", "fee_bps", "capital_amount", "impact_bps_per_1pct"])["mean_annualized_return"].rank(
        method="first",
        ascending=False,
    ).astype(int)
    return output.reset_index(drop=True)


def summarize_combined_basket_exposure(exposure: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "constraint_variant",
        "top_n",
        "signal",
        "signal_config",
        "exposure_penalty_strength",
        "factor",
        "period_type",
        "row_count",
        "mean_active_exposure",
        "mean_abs_active_exposure",
        "max_abs_active_exposure",
    ]
    if exposure.empty:
        return pd.DataFrame(columns=columns)
    required = {"signal", "signal_config", "exposure_penalty_strength", "factor", "period_type", "active_exposure"}
    if missing := sorted(required - set(exposure.columns)):
        raise ValueError(f"exposure missing required columns: {missing}")
    work = exposure.copy()
    if "constraint_variant" not in work.columns:
        work["constraint_variant"] = "baseline"
    if "top_n" not in work.columns:
        work["top_n"] = DEFAULT_TOP_N
    work["active_exposure"] = pd.to_numeric(work["active_exposure"], errors="coerce")
    work["abs_active_exposure"] = work["active_exposure"].abs()
    rows: list[dict[str, Any]] = []
    for keys, group in work.groupby(["constraint_variant", "top_n", "signal", "signal_config", "exposure_penalty_strength", "factor", "period_type"], sort=True):
        constraint_variant, top_n, signal, signal_config, strength, factor, period_type = keys
        rows.append(
            {
                "constraint_variant": constraint_variant,
                "top_n": int(top_n),
                "signal": signal,
                "signal_config": signal_config,
                "exposure_penalty_strength": float(strength),
                "factor": factor,
                "period_type": period_type,
                "row_count": int(len(group)),
                "mean_active_exposure": float(pd.to_numeric(group["active_exposure"], errors="coerce").mean()),
                "mean_abs_active_exposure": float(group["abs_active_exposure"].mean()),
                "max_abs_active_exposure": float(group["abs_active_exposure"].max()),
            }
        )
    return pd.DataFrame(rows).loc[:, columns]


def summarize_combined_industry_exposure(industry_exposure: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "constraint_variant",
        "top_n",
        "signal",
        "signal_config",
        "exposure_penalty_strength",
        "period_type",
        "row_count",
        "mean_abs_active_weight",
        "max_abs_active_weight",
    ]
    if industry_exposure.empty:
        return pd.DataFrame(columns=columns)
    required = {"signal", "signal_config", "exposure_penalty_strength", "period_type", "abs_active_weight"}
    if missing := sorted(required - set(industry_exposure.columns)):
        raise ValueError(f"industry exposure missing required columns: {missing}")
    work = industry_exposure.copy()
    if "constraint_variant" not in work.columns:
        work["constraint_variant"] = "baseline"
    if "top_n" not in work.columns:
        work["top_n"] = DEFAULT_TOP_N
    work["abs_active_weight"] = pd.to_numeric(work["abs_active_weight"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for keys, group in work.groupby(["constraint_variant", "top_n", "signal", "signal_config", "exposure_penalty_strength", "period_type"], sort=True):
        constraint_variant, top_n, signal, signal_config, strength, period_type = keys
        rows.append(
            {
                "constraint_variant": constraint_variant,
                "top_n": int(top_n),
                "signal": signal,
                "signal_config": signal_config,
                "exposure_penalty_strength": float(strength),
                "period_type": period_type,
                "row_count": int(len(group)),
                "mean_abs_active_weight": float(group["abs_active_weight"].mean()),
                "max_abs_active_weight": float(group["abs_active_weight"].max()),
            }
        )
    return pd.DataFrame(rows).loc[:, columns]


def best_combined_rows(
    aggregate: pd.DataFrame,
    *,
    fee_bps: float,
    capital_amount: float,
    top: int = 12,
) -> list[dict[str, Any]]:
    if aggregate.empty:
        return []
    rows = aggregate.loc[
        pd.to_numeric(aggregate["fee_bps"], errors="coerce").eq(float(fee_bps))
        & pd.to_numeric(aggregate["capital_amount"], errors="coerce").eq(float(capital_amount))
    ]
    if rows.empty:
        rows = aggregate
    return rows.sort_values(
        ["impact_bps_per_1pct", "mean_annualized_return", "positive_year_rate", "worst_max_drawdown"],
        ascending=[True, False, False, False],
    ).head(top).to_dict("records")


def render_combined_constraint_markdown(
    result: Mapping[str, Any],
    aggregate: pd.DataFrame,
    exposure_summary: pd.DataFrame,
    industry_summary: pd.DataFrame,
    liquidity_summary: pd.DataFrame,
) -> str:
    best_rows = pd.DataFrame(result.get("best_30bps_100m_rows") or [])
    lines = [
        "# Low-Corr Frontier Combined Constraint Audit",
        "",
        "- Hypothesis: frontier signals should survive a combined portfolio-construction gate, not isolated single controls.",
        f"- Protocol: `horizon={result.get('horizon')}`, `{result.get('rebalance_frequency')}`, `top_n_values={result.get('top_n_values', [result.get('top_n')])}`, `buffer={result.get('buffer_multiplier')}`.",
        f"- Signals: `{result.get('signals')}`.",
        f"- Signal penalty strengths: `{result.get('signal_penalty_strengths')}`.",
        f"- Group cap: `{result.get('group_col') or 'none'}` max weight `{result.get('max_group_weight')}`.",
        f"- Capital: `{result.get('capital_amounts')}`; impact bps per 1 pct participation `{result.get('impact_bps_per_1pct_values')}`.",
        "- Assessment: combined constraint gate evidence only; candidate count remains `0` until all promotion gates pass.",
        f"- Artifacts: `{result.get('output_dir')}`",
        "",
        "## 30 bps / 100m Rows",
        "",
        _markdown_table(best_rows),
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
    return "\n".join(lines)


def _holding_extra_columns(frame: pd.DataFrame, signal: str) -> list[str]:
    columns = [
        "amount",
        "volume",
        signal,
        "log_amount_mean_20d_z",
        "momentum_20d_z",
        "reversal_5d_z",
        "neg_volatility_20d_z",
        "neg_amplitude_20d_z",
        "turn_xsec_z",
    ]
    output: list[str] = []
    for column in columns:
        if column in frame.columns and column not in output:
            output.append(column)
    return output


def _normalize_tuple(values: Sequence[str] | str, *, name: str) -> tuple[str, ...]:
    raw_values = values.split(",") if isinstance(values, str) else values
    output = tuple(str(value).strip() for value in raw_values if str(value).strip())
    if not output:
        raise ValueError(f"{name} must not be empty")
    return output


def _normalize_optional_tuple(values: Sequence[str] | str | None) -> tuple[str, ...]:
    if values is None:
        return ()
    raw_values = values.split(",") if isinstance(values, str) else values
    output: list[str] = []
    for value in raw_values:
        text = str(value).strip()
        if text and text not in output:
            output.append(text)
    return tuple(output)


def _normalize_constraint_variants(values: Sequence[str] | str) -> tuple[str, ...]:
    output = _normalize_tuple(values, name="constraint_variants")
    allowed = set(DEFAULT_CONSTRAINT_VARIANTS)
    unknown = sorted(set(output) - allowed)
    if unknown:
        raise ValueError(f"unsupported constraint variants: {unknown}")
    return output


def _normalize_float_tuple(values: Sequence[float] | str, *, name: str) -> tuple[float, ...]:
    raw_values = values.split(",") if isinstance(values, str) else values
    output = tuple(float(value) for value in raw_values if str(value).strip())
    if not output:
        raise ValueError(f"{name} must not be empty")
    return output


def _normalize_int_tuple(values: Sequence[int] | str, *, name: str) -> tuple[int, ...]:
    raw_values = values.split(",") if isinstance(values, str) else values
    output = tuple(int(str(value).strip()) for value in raw_values if str(value).strip())
    if not output:
        raise ValueError(f"{name} must not be empty")
    return output


def _unique_columns(columns: Sequence[str]) -> list[str]:
    output: list[str] = []
    for column in columns:
        if column and column not in output:
            output.append(column)
    return output


def _sum_bool_column(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int(_truthy(frame[column]).sum())


def _set_or_insert(frame: pd.DataFrame, column: str, value: Any) -> None:
    if column in frame.columns:
        frame[column] = value
    else:
        frame.insert(0, column, value)


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _safe_signal(value: Any) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value))[:48]


def _capital_token(value: float) -> str:
    if value >= 1_000_000:
        return f"{int(round(value / 1_000_000))}m"
    return f"{int(round(value))}"


def _parse_years(spec: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in spec.split(",") if value.strip())
    if not values:
        raise ValueError("expected at least one year")
    return values


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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--years", default=",".join(str(year) for year in DEFAULT_YEARS))
    parser.add_argument("--final-end-date", default=DEFAULT_FINAL_END_DATE)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="raw")
    parser.add_argument("--max-factor-corr", type=float, default=DEFAULT_MAX_FACTOR_CORR)
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW)
    parser.add_argument("--rolling-min-periods", type=int, default=DEFAULT_ROLLING_MIN_PERIODS)
    parser.add_argument("--signals", default=",".join(DEFAULT_FRONTIER_SIGNALS))
    parser.add_argument("--signal-penalty-strengths", default=",".join(DEFAULT_SIGNAL_PENALTY_STRENGTHS))
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--top-n-values", default=None)
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_low_corr_frontier_combined_constraint_audit(
        root=args.root,
        years=_parse_years(args.years),
        final_end_date=args.final_end_date,
        horizon=args.horizon,
        label_mode=args.label_mode,
        max_factor_corr=args.max_factor_corr,
        rolling_window=args.rolling_window,
        rolling_min_periods=args.rolling_min_periods,
        signals=_normalize_tuple(args.signals, name="signals"),
        signal_penalty_strengths=args.signal_penalty_strengths,
        top_n=args.top_n,
        top_n_values=_normalize_int_tuple(args.top_n_values, name="top_n_values") if args.top_n_values is not None else None,
        rebalance_frequency=args.frequency,
        buffer_multiplier=args.buffer_multiplier,
        fee_bps_values=_normalize_float_tuple(args.fee_bps, name="fee_bps_values"),
        capital_amounts=_normalize_float_tuple(args.capital_amounts, name="capital_amounts"),
        impact_bps_per_1pct_values=_normalize_float_tuple(args.impact_bps_per_1pct, name="impact_bps_per_1pct"),
        exposure_penalty_cols=_normalize_tuple(args.exposure_penalty_cols, name="exposure_penalty_cols"),
        exposure_columns=_normalize_tuple(args.exposure_columns, name="exposure_columns"),
        exposure_constraint_cols=_normalize_optional_tuple(args.exposure_constraint_cols),
        max_abs_exposure=args.max_abs_exposure,
        group_col=args.group_col.strip() or None,
        max_group_weight=args.max_group_weight,
        execution_constraints=args.execution_constraints,
        limit_threshold=args.limit_threshold,
        include_metrics=args.include_metrics,
        include_industry=args.include_industry,
        weak_year_rebuild_run_dir=args.weak_year_rebuild_run_dir,
        constraint_variants=_normalize_optional_tuple(args.constraint_variants) if args.constraint_variants is not None else None,
        output_dir=args.output_dir,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
