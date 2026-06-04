"""Compare candidate-frontier protocol performance across signal variants."""

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
    DEFAULT_FEE_BPS_VALUES,
    DEFAULT_FINAL_END_DATE,
    DEFAULT_HORIZON,
    DEFAULT_MAX_FACTOR_CORR,
    DEFAULT_REBALANCE_FREQUENCY,
    DEFAULT_TOP_N,
    DEFAULT_YEARS,
    LABEL_MODES,
    build_low_corr_signal_panel,
)
from traditional_quant_research.experiments.low_corr_exposure_grid import LOW_CORR_SIGNAL
from traditional_quant_research.experiments.low_corr_regime_yearly_validation import year_windows
from traditional_quant_research.experiments.v2_metrics_exposure_diagnostics import add_metric_exposure_fields
from traditional_quant_research.experiments.multifactor_baseline import (
    apply_horizon_fee,
    filter_panel_dates,
    rolling_fallback_rate,
    rolling_ic_weight_audit,
    selected_basket_factor_exposure,
    summarize_horizon_trade_table,
)
from traditional_quant_research.experiments.frontier_ml_signal_rebuild import ML_SIGNAL_NAME_TEMPLATE
from traditional_quant_research.horizon_backtest import (
    horizon_aligned_top_n_backtest,
    period_horizon_summary,
    yearly_horizon_summary,
)
from traditional_quant_research.multifactor import (
    add_equal_rank_score,
    add_ic_weighted_rank_score,
    add_prior_fit_pruned_rank_score,
    add_rank_score_from_weight_table,
    factor_coverage,
    rolling_ic_weights_by_date,
)
from traditional_quant_research.research_panel import add_baseline_score
from traditional_quant_research.research_panel import DEFAULT_FACTOR_SET, FACTOR_SETS, normalize_factor_set


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/low_corr_candidate_signal_comparison")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-02_low_corr_candidate_signal_comparison.md")
DEFAULT_ROLLING_WINDOW = 252
DEFAULT_ROLLING_MIN_PERIODS = 60
BASELINE_SIGNAL = "baseline_score"
EQUAL_SIGNAL = "multifactor_equal_rank_score"
IC_WEIGHTED_SIGNAL = "multifactor_ic_weighted_score"
ROLLING_IC_SIGNAL = "multifactor_rolling_ic_weighted_score"
PRUNED_SIGNAL_TEMPLATE = "factor_pruned_rank_score_h{horizon}_prior_fit"
DEFAULT_SIGNALS = (
    BASELINE_SIGNAL,
    EQUAL_SIGNAL,
    IC_WEIGHTED_SIGNAL,
    ROLLING_IC_SIGNAL,
    LOW_CORR_SIGNAL,
)


def build_candidate_protocol_signal_panel(
    *,
    root: str | None,
    history_start_date: str,
    fit_start_date: str,
    fit_end_date: str,
    start_date: str,
    end_date: str,
    horizon: int,
    label_mode: str,
    factor_set: str | None = DEFAULT_FACTOR_SET,
    max_factor_corr: float,
    rolling_window: int,
    rolling_min_periods: int,
    include_industry: bool = False,
    include_metrics: bool = False,
    factor_pruning_run_dir: str | Path | None = None,
    ml_signal_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build all signal variants using the same prior-year fit contract."""

    if rolling_window <= 0:
        raise ValueError("rolling_window must be positive")
    if rolling_min_periods <= 0:
        raise ValueError("rolling_min_periods must be positive")
    selected_factor_set = normalize_factor_set(factor_set)
    built = build_low_corr_signal_panel(
        root=root,
        history_start_date=history_start_date,
        fit_start_date=fit_start_date,
        fit_end_date=fit_end_date,
        start_date=start_date,
        end_date=end_date,
        horizon=horizon,
        label_mode=label_mode,
        factor_set=selected_factor_set,
        max_factor_corr=max_factor_corr,
        include_industry=include_industry,
        include_metrics=include_metrics,
    )
    signal_columns = list(built["signal_columns"])
    factor_panel = add_baseline_score(built["factor_panel"], score_columns=signal_columns)
    factor_directions = built["factor_directions"]
    label_col = built["label"]
    factor_panel = add_equal_rank_score(
        factor_panel,
        signal_columns,
        directions=factor_directions,
        score_col=EQUAL_SIGNAL,
        min_factors=3,
    )
    factor_panel = add_ic_weighted_rank_score(
        factor_panel,
        signal_columns,
        built["single_factor_ic"],
        directions=factor_directions,
        score_col=IC_WEIGHTED_SIGNAL,
        min_factors=3,
    )
    rolling_weights = rolling_ic_weights_by_date(
        factor_panel,
        signal_columns,
        label_col,
        window=rolling_window,
        min_periods=rolling_min_periods,
    )
    factor_panel = add_rank_score_from_weight_table(
        factor_panel,
        signal_columns,
        rolling_weights,
        score_col=ROLLING_IC_SIGNAL,
        min_factors=3,
    )
    pruned_signal = ""
    pruned_plan = pd.DataFrame()
    if factor_pruning_run_dir is not None:
        pruned_plan = read_factor_pruning_plan(factor_pruning_run_dir, horizon=horizon)
        pruned_signal = factor_pruning_signal_name(pruned_plan, horizon=horizon)
        factor_panel = add_prior_fit_pruned_rank_score(
            factor_panel,
            pruned_plan,
            score_col=pruned_signal,
            min_factors=3,
        )
    ml_signal = ""
    ml_predictions = pd.DataFrame()
    if ml_signal_run_dir is not None:
        ml_predictions = read_ml_signal_predictions(ml_signal_run_dir, horizon=horizon)
        ml_signal = ml_signal_name(ml_predictions, horizon=horizon)
        merge = ml_predictions.loc[:, ["date", "code", "score"]].rename(columns={"score": ml_signal})
        factor_panel = factor_panel.copy()
        factor_panel["date"] = pd.to_datetime(factor_panel["date"])
        factor_panel["code"] = factor_panel["code"].astype(str)
        factor_panel = factor_panel.merge(merge, on=["date", "code"], how="left")
    metric_exposure_columns: list[str] = []
    if include_metrics:
        factor_panel, metric_exposure_columns = add_metric_exposure_fields(factor_panel)
    evaluation_panel = filter_panel_dates(factor_panel, start_date=start_date, end_date=end_date)
    candidate_signals = [*DEFAULT_SIGNALS, *([pruned_signal] if pruned_signal else []), *([ml_signal] if ml_signal else [])]
    available_signals = [signal for signal in candidate_signals if signal in evaluation_panel.columns]
    return {
        **built,
        "factor_set": selected_factor_set,
        "factor_panel": factor_panel,
        "evaluation_panel": evaluation_panel,
        "available_signals": available_signals,
        "rolling_weights": rolling_weights,
        "rolling_weight_audit": rolling_ic_weight_audit(rolling_weights),
        "rolling_fallback_rate": rolling_fallback_rate(rolling_weights),
        "signal_coverage": factor_coverage(evaluation_panel, available_signals),
        "metric_exposure_columns": metric_exposure_columns,
        "factor_pruning_run_dir": str(factor_pruning_run_dir or ""),
        "factor_pruning_signal": pruned_signal,
        "factor_pruning_plan_rows": int(len(pruned_plan)),
        "ml_signal_run_dir": str(ml_signal_run_dir or ""),
        "ml_signal": ml_signal,
        "ml_prediction_rows": int(len(ml_predictions)),
    }


def read_factor_pruning_plan(run_dir: str | Path, *, horizon: int) -> pd.DataFrame:
    """Read the prior-fit factor pruning plan for a specific horizon."""

    path = Path(run_dir) / "factor_pruning_plan.csv"
    if not path.exists():
        raise FileNotFoundError(f"factor pruning plan not found: {path}")
    frame = pd.read_csv(path)
    required = {"eval_year", "horizon", "candidate_signal_name", "selected_factors", "selected_factor_directions", "fit_uses_eval_year"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"factor pruning plan missing required columns: {missing}")
    output = frame.copy()
    output["horizon"] = pd.to_numeric(output["horizon"], errors="coerce").astype("Int64")
    output = output.loc[output["horizon"].eq(int(horizon))].copy()
    if output.empty:
        raise ValueError(f"factor pruning plan has no rows for horizon {horizon}")
    if _truthy(output["fit_uses_eval_year"]).any():
        raise ValueError("factor pruning plan must be prior-fit; fit_uses_eval_year must be false")
    return output.reset_index(drop=True)


def factor_pruning_signal_name(plan: pd.DataFrame, *, horizon: int) -> str:
    if plan.empty:
        return PRUNED_SIGNAL_TEMPLATE.format(horizon=int(horizon))
    names = sorted(set(plan["candidate_signal_name"].dropna().astype(str)))
    return names[0] if len(names) == 1 and names[0] else PRUNED_SIGNAL_TEMPLATE.format(horizon=int(horizon))


def read_ml_signal_predictions(run_dir: str | Path, *, horizon: int) -> pd.DataFrame:
    """Read prior-fit ML predictions for a specific horizon."""

    path = Path(run_dir) / "ml_signal_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"ML signal predictions not found: {path}")
    frame = pd.read_csv(path)
    required = {"eval_year", "date", "code", "horizon", "ml_signal_name", "score", "fit_uses_eval_year"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"ML signal predictions missing required columns: {missing}")
    output = frame.copy()
    output["horizon"] = pd.to_numeric(output["horizon"], errors="coerce").astype("Int64")
    output = output.loc[output["horizon"].eq(int(horizon))].copy()
    if output.empty:
        raise ValueError(f"ML signal predictions have no rows for horizon {horizon}")
    if _truthy(output["fit_uses_eval_year"]).any():
        raise ValueError("ML signal predictions must be prior-fit; fit_uses_eval_year must be false")
    output["date"] = pd.to_datetime(output["date"])
    output["code"] = output["code"].astype(str)
    output["score"] = pd.to_numeric(output["score"], errors="coerce")
    output = output.dropna(subset=["date", "code", "score"]).copy()
    if output.empty:
        raise ValueError(f"ML signal predictions have no finite score rows for horizon {horizon}")
    if output.duplicated(["date", "code"]).any():
        raise ValueError("ML signal predictions must be unique by date,code after horizon filtering")
    expected_name = ML_SIGNAL_NAME_TEMPLATE.format(horizon=int(horizon))
    names = sorted(set(output["ml_signal_name"].dropna().astype(str)))
    if names != [expected_name]:
        raise ValueError(f"ML signal predictions must use {expected_name}; got {names}")
    return output.reset_index(drop=True)


def ml_signal_name(predictions: pd.DataFrame, *, horizon: int) -> str:
    if predictions.empty:
        return ML_SIGNAL_NAME_TEMPLATE.format(horizon=int(horizon))
    names = sorted(set(predictions["ml_signal_name"].dropna().astype(str)))
    return names[0] if len(names) == 1 else ML_SIGNAL_NAME_TEMPLATE.format(horizon=int(horizon))


def run_low_corr_candidate_signal_comparison(
    *,
    root: str | None = None,
    years: Sequence[int] = DEFAULT_YEARS,
    final_end_date: str | None = DEFAULT_FINAL_END_DATE,
    horizon: int = DEFAULT_HORIZON,
    label_mode: str = "raw",
    factor_set: str | None = DEFAULT_FACTOR_SET,
    max_factor_corr: float = DEFAULT_MAX_FACTOR_CORR,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    rolling_min_periods: int = DEFAULT_ROLLING_MIN_PERIODS,
    signals: Sequence[str] = DEFAULT_SIGNALS,
    top_n: int = DEFAULT_TOP_N,
    rebalance_frequency: str = DEFAULT_REBALANCE_FREQUENCY,
    buffer_multiplier: float = DEFAULT_BUFFER_MULTIPLIER,
    fee_bps_values: Sequence[float] = DEFAULT_FEE_BPS_VALUES,
    execution_constraints: bool = True,
    limit_threshold: float = 0.095,
    factor_pruning_run_dir: str | Path | None = None,
    ml_signal_run_dir: str | Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    """Run the same candidate protocol for several signal variants."""

    if not years:
        raise ValueError("years must not be empty")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if label_mode not in LABEL_MODES:
        raise ValueError(f"unsupported label_mode: {label_mode}")
    selected_factor_set = normalize_factor_set(factor_set)
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if buffer_multiplier < 1.0:
        raise ValueError("buffer_multiplier must be at least 1.0")
    if not fee_bps_values:
        raise ValueError("fee_bps_values must not be empty")
    selected_signals = _normalize_signals(signals)

    run_id = f"low_corr_candidate_signal_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    yearly_frames: list[pd.DataFrame] = []
    period_frames: list[pd.DataFrame] = []
    exposure_frames: list[pd.DataFrame] = []
    signal_coverage_frames: list[pd.DataFrame] = []
    rolling_audit_frames: list[pd.DataFrame] = []
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
            factor_set=selected_factor_set,
            max_factor_corr=max_factor_corr,
            rolling_window=rolling_window,
            rolling_min_periods=rolling_min_periods,
            factor_pruning_run_dir=factor_pruning_run_dir,
            ml_signal_run_dir=ml_signal_run_dir,
        )
        manifest = built["manifest"]
        quality = built["quality"]
        evaluation_panel = built["evaluation_panel"]
        year_signals = _validate_available_signals(selected_signals, built["available_signals"])

        coverage = built["signal_coverage"].copy()
        if not coverage.empty:
            coverage.insert(0, "eval_year", int(year))
            signal_coverage_frames.append(coverage)
        rolling_audit = built["rolling_weight_audit"].copy()
        if not rolling_audit.empty:
            rolling_audit.insert(0, "eval_year", int(year))
            rolling_audit_frames.append(rolling_audit)

        exposure_columns = _unique_columns([*built["signal_columns"], *year_signals])
        for signal in year_signals:
            base_backtest = horizon_aligned_top_n_backtest(
                evaluation_panel,
                signal,
                horizon=horizon,
                top_n=top_n,
                fee_bps=0.0,
                rebalance_frequency=rebalance_frequency,
                buffer_multiplier=buffer_multiplier,
                execution_constraints=execution_constraints,
                limit_threshold=limit_threshold,
            )
            base_trades = base_backtest.trades.copy()
            if not base_trades.empty:
                base_trades.insert(0, "signal", signal)
                base_trades.insert(0, "eval_year", int(year))
                trade_frames.append(base_trades)
            exposure = selected_basket_factor_exposure(
                evaluation_panel,
                base_backtest.trades,
                exposure_columns,
                signal_col=signal,
                horizon=horizon,
                rebalance_frequency=rebalance_frequency,
                top_n=top_n,
                buffer_multiplier=buffer_multiplier,
            )
            if not exposure.empty:
                exposure.insert(0, "eval_year", int(year))
                exposure_frames.append(exposure)
            for fee_bps in fee_bps_values:
                trades = apply_horizon_fee(base_backtest.trades, fee_bps=float(fee_bps))
                row = {"eval_year": int(year), "signal": signal}
                row.update(
                    summarize_horizon_trade_table(
                        trades,
                        horizon=horizon,
                        rebalance_frequency=rebalance_frequency,
                        top_n=top_n,
                        fee_bps=float(fee_bps),
                        buffer_multiplier=buffer_multiplier,
                        execution_constraints=execution_constraints,
                        limit_threshold=limit_threshold,
                    )
                )
                summary_rows.append(row)
                yearly = yearly_horizon_summary(
                    trades,
                    horizon=horizon,
                    rebalance_frequency=rebalance_frequency,
                    top_n=top_n,
                    fee_bps=float(fee_bps),
                    non_overlapping=True,
                    buffer_multiplier=buffer_multiplier,
                    execution_constraints=execution_constraints,
                    limit_threshold=limit_threshold,
                    signal_col=signal,
                )
                if not yearly.empty:
                    yearly.insert(0, "eval_year", int(year))
                    yearly_frames.append(yearly)
                for period in ("M", "Q"):
                    period_summary = period_horizon_summary(
                        trades,
                        period=period,
                        horizon=horizon,
                        rebalance_frequency=rebalance_frequency,
                        top_n=top_n,
                        fee_bps=float(fee_bps),
                        non_overlapping=True,
                        buffer_multiplier=buffer_multiplier,
                        execution_constraints=execution_constraints,
                        limit_threshold=limit_threshold,
                        signal_col=signal,
                    )
                    if not period_summary.empty:
                        period_summary.insert(0, "eval_year", int(year))
                        period_frames.append(period_summary)
        metadata_rows.append(
            {
                "eval_year": int(year),
                "history_start_date": windows["history_start_date"],
                "fit_start_date": windows["fit_start_date"],
                "fit_end_date": windows["fit_end_date"],
                "start_date": windows["start_date"],
                "end_date": windows["end_date"],
                "factor_set": selected_factor_set,
                "available_signals": json.dumps(year_signals, ensure_ascii=False),
                "low_corr_factor_columns": json.dumps(built["low_corr_factor_columns"], ensure_ascii=False),
                "factor_pruning_run_dir": str(factor_pruning_run_dir or ""),
                "factor_pruning_signal": str(built.get("factor_pruning_signal", "")),
                "factor_pruning_plan_rows": int(built.get("factor_pruning_plan_rows", 0)),
                "ml_signal_run_dir": str(ml_signal_run_dir or ""),
                "ml_signal": str(built.get("ml_signal", "")),
                "ml_prediction_rows": int(built.get("ml_prediction_rows", 0)),
                "rolling_fallback_rate": float(built["rolling_fallback_rate"]),
                "evaluation_rows": int(len(evaluation_panel)),
                "evaluation_dates": int(evaluation_panel["date"].nunique()) if "date" in evaluation_panel.columns else 0,
                "evaluation_securities": int(evaluation_panel["code"].nunique()) if "code" in evaluation_panel.columns else 0,
            }
        )

    summary = pd.DataFrame(summary_rows)
    aggregate = summarize_signal_comparison(summary)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    yearly_summary = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    period_summary = pd.concat(period_frames, ignore_index=True) if period_frames else pd.DataFrame()
    exposure_summary = pd.concat(exposure_frames, ignore_index=True) if exposure_frames else pd.DataFrame()
    signal_coverage = pd.concat(signal_coverage_frames, ignore_index=True) if signal_coverage_frames else pd.DataFrame()
    rolling_audit = pd.concat(rolling_audit_frames, ignore_index=True) if rolling_audit_frames else pd.DataFrame()
    metadata = pd.DataFrame(metadata_rows)

    result = {
        "run_id": run_id,
        "snapshot_id": manifest.get("snapshot_id"),
        "years": [int(year) for year in years],
        "final_end_date": final_end_date,
        "horizon": int(horizon),
        "label_mode": label_mode,
        "factor_set": selected_factor_set,
        "rolling_window": int(rolling_window),
        "rolling_min_periods": int(rolling_min_periods),
        "signals": list(selected_signals),
        "rebalance_frequency": rebalance_frequency,
        "top_n": int(top_n),
        "buffer_multiplier": float(buffer_multiplier),
        "fee_bps_values": [float(value) for value in fee_bps_values],
        "execution_constraints": bool(execution_constraints),
        "limit_threshold": float(limit_threshold),
        "factor_pruning_run_dir": str(factor_pruning_run_dir or ""),
        "ml_signal_run_dir": str(ml_signal_run_dir or ""),
        "quality": {
            "failure_count": quality.get("failure_count"),
            "missing_bar_rows": quality.get("missing_bar_rows"),
            "st_rows": quality.get("st_rows"),
            "suspended_like_rows": quality.get("suspended_like_rows"),
        },
        "best_30bps_rows": best_signal_rows(aggregate, fee_bps=30.0),
        "candidate_count": 0,
        "assessment": "same-protocol signal comparison; no strategy candidate is promoted by this experiment alone",
        "output_dir": str(run_dir),
    }

    summary.to_csv(run_dir / "signal_protocol_summary.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(run_dir / "signal_protocol_aggregate.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(run_dir / "signal_protocol_trades.csv", index=False, encoding="utf-8-sig")
    yearly_summary.to_csv(run_dir / "signal_protocol_yearly_summary.csv", index=False, encoding="utf-8-sig")
    period_summary.to_csv(run_dir / "signal_protocol_period_summary.csv", index=False, encoding="utf-8-sig")
    exposure_summary.to_csv(run_dir / "signal_basket_exposure.csv", index=False, encoding="utf-8-sig")
    signal_coverage.to_csv(run_dir / "signal_coverage.csv", index=False, encoding="utf-8-sig")
    rolling_audit.to_csv(run_dir / "rolling_ic_weight_audit.csv", index=False, encoding="utf-8-sig")
    metadata.to_csv(run_dir / "signal_comparison_meta.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_signal_comparison_markdown(result, aggregate, summary, metadata)
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        research_log_path.write_text(markdown, encoding="utf-8")
    return result


def summarize_signal_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate same-protocol rows by signal and fee."""

    if summary.empty:
        return pd.DataFrame()
    required = {"eval_year", "signal", "fee_bps", "annualized_return", "sharpe", "max_drawdown", "mean_turnover"}
    if missing := sorted(required - set(summary.columns)):
        raise ValueError(f"summary missing required columns: {missing}")
    work = summary.copy()
    zero_fee = work.loc[
        pd.to_numeric(work["fee_bps"], errors="coerce") == 0.0,
        ["eval_year", "signal", "annualized_return"],
    ].rename(columns={"annualized_return": "annualized_return_0bps"})
    work = work.merge(zero_fee, on=["eval_year", "signal"], how="left")
    low_corr = work.loc[
        work["signal"] == LOW_CORR_SIGNAL,
        ["eval_year", "fee_bps", "annualized_return"],
    ].rename(columns={"annualized_return": "low_corr_annualized_return"})
    work = work.merge(low_corr, on=["eval_year", "fee_bps"], how="left")
    work["cost_drag_vs_0bps"] = pd.to_numeric(work["annualized_return"], errors="coerce") - pd.to_numeric(
        work["annualized_return_0bps"],
        errors="coerce",
    )
    work["delta_annualized_return_vs_low_corr"] = pd.to_numeric(work["annualized_return"], errors="coerce") - pd.to_numeric(
        work["low_corr_annualized_return"],
        errors="coerce",
    )

    rows: list[dict[str, Any]] = []
    for (signal, fee_bps), group in work.groupby(["signal", "fee_bps"], sort=True):
        returns = pd.to_numeric(group["annualized_return"], errors="coerce")
        deltas = pd.to_numeric(group["delta_annualized_return_vs_low_corr"], errors="coerce")
        rows.append(
            {
                "signal": signal,
                "fee_bps": float(fee_bps),
                "eval_year_count": int(group["eval_year"].nunique()),
                "mean_annualized_return": float(returns.mean()),
                "min_annualized_return": float(returns.min()),
                "positive_year_rate": float((returns > 0).mean()),
                "mean_sharpe": float(pd.to_numeric(group["sharpe"], errors="coerce").mean()),
                "worst_max_drawdown": float(pd.to_numeric(group["max_drawdown"], errors="coerce").min()),
                "mean_turnover": float(pd.to_numeric(group["mean_turnover"], errors="coerce").mean()),
                "mean_cost_drag_vs_0bps": float(pd.to_numeric(group["cost_drag_vs_0bps"], errors="coerce").mean()),
                "mean_delta_annualized_return_vs_low_corr": float(deltas.mean()) if not deltas.dropna().empty else np.nan,
                "positive_delta_year_rate_vs_low_corr": float((deltas > 0).mean()) if not deltas.dropna().empty else np.nan,
                "total_periods": int(pd.to_numeric(group["periods"], errors="coerce").sum()) if "periods" in group.columns else 0,
            }
        )
    output = pd.DataFrame(rows).sort_values(
        ["fee_bps", "mean_annualized_return", "positive_year_rate", "worst_max_drawdown"],
        ascending=[True, False, False, False],
    )
    if output.empty:
        return output
    output["rank_within_fee"] = output.groupby("fee_bps")["mean_annualized_return"].rank(method="first", ascending=False).astype(int)
    return output.reset_index(drop=True)


def best_signal_rows(aggregate: pd.DataFrame, *, fee_bps: float = 30.0, top: int = 10) -> list[dict[str, Any]]:
    if aggregate.empty:
        return []
    rows = aggregate.loc[pd.to_numeric(aggregate["fee_bps"], errors="coerce") == float(fee_bps)]
    if rows.empty:
        rows = aggregate
    return rows.sort_values(
        ["mean_annualized_return", "positive_year_rate", "worst_max_drawdown"],
        ascending=[False, False, False],
    ).head(top).to_dict("records")


def render_signal_comparison_markdown(
    result: Mapping[str, Any],
    aggregate: pd.DataFrame,
    summary: pd.DataFrame,
    metadata: pd.DataFrame,
) -> str:
    best_rows = pd.DataFrame(result.get("best_30bps_rows") or [])
    lines = [
        "# Low-Corr Candidate Signal Comparison",
        "",
        "- Hypothesis: the candidate-frontier protocol should be compared against other rank signal variants under the exact same execution protocol.",
        f"- Protocol: `horizon={result.get('horizon')}`, `{result.get('rebalance_frequency')}`, `top_n={result.get('top_n')}`, `buffer={result.get('buffer_multiplier')}`.",
        f"- Years: `{result.get('years')}`; final end date `{result.get('final_end_date')}`.",
        f"- Signals: `{result.get('signals')}`.",
        f"- Fees: `{result.get('fee_bps_values')}` bps.",
        "- Assessment: signal comparison evidence only; candidate count remains `0` until promotion gates pass.",
        f"- Artifacts: `{result.get('output_dir')}`",
        "",
        "## Best 30 bps Rows",
        "",
        _markdown_table(best_rows),
        "",
        "## Aggregate",
        "",
        _markdown_table(aggregate),
        "",
        "## Year Rows",
        "",
        _markdown_table(summary),
        "",
        "## Metadata",
        "",
        _markdown_table(metadata),
        "",
    ]
    return "\n".join(lines)


def _normalize_signals(signals: Sequence[str]) -> tuple[str, ...]:
    output = tuple(signal.strip() for signal in signals if signal and signal.strip())
    if not output:
        raise ValueError("signals must not be empty")
    return output


def _validate_available_signals(selected: Sequence[str], available: Sequence[str]) -> tuple[str, ...]:
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"signals not available in panel: {missing}")
    return tuple(selected)


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _unique_columns(columns: Sequence[str]) -> list[str]:
    output: list[str] = []
    for column in columns:
        if column and column not in output:
            output.append(column)
    return output


def _parse_years(spec: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in spec.split(",") if value.strip())
    if not values:
        raise ValueError("expected at least one year")
    return values


def _parse_float_tuple(spec: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in spec.split(",") if value.strip())
    if not values:
        raise ValueError("expected at least one numeric value")
    return values


def _parse_signals(spec: str) -> tuple[str, ...]:
    return _normalize_signals(tuple(value.strip() for value in spec.split(",") if value.strip()))


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
    parser.add_argument("--factor-set", choices=FACTOR_SETS, default=DEFAULT_FACTOR_SET)
    parser.add_argument("--max-factor-corr", type=float, default=DEFAULT_MAX_FACTOR_CORR)
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW)
    parser.add_argument("--rolling-min-periods", type=int, default=DEFAULT_ROLLING_MIN_PERIODS)
    parser.add_argument("--signals", default=",".join(DEFAULT_SIGNALS))
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--frequency", default=DEFAULT_REBALANCE_FREQUENCY)
    parser.add_argument("--buffer-multiplier", type=float, default=DEFAULT_BUFFER_MULTIPLIER)
    parser.add_argument("--fee-bps", default=",".join(str(value) for value in DEFAULT_FEE_BPS_VALUES))
    parser.add_argument("--execution-constraints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-threshold", type=float, default=0.095)
    parser.add_argument("--factor-pruning-run-dir", type=Path, default=None)
    parser.add_argument("--ml-signal-run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_low_corr_candidate_signal_comparison(
        root=args.root,
        years=_parse_years(args.years),
        final_end_date=args.final_end_date,
        horizon=args.horizon,
        label_mode=args.label_mode,
        factor_set=args.factor_set,
        max_factor_corr=args.max_factor_corr,
        rolling_window=args.rolling_window,
        rolling_min_periods=args.rolling_min_periods,
        signals=_parse_signals(args.signals),
        top_n=args.top_n,
        rebalance_frequency=args.frequency,
        buffer_multiplier=args.buffer_multiplier,
        fee_bps_values=_parse_float_tuple(args.fee_bps),
        execution_constraints=args.execution_constraints,
        limit_threshold=args.limit_threshold,
        factor_pruning_run_dir=args.factor_pruning_run_dir,
        ml_signal_run_dir=args.ml_signal_run_dir,
        output_dir=args.output_dir,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
