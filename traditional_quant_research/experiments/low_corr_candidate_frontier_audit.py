"""Audit the current low-corr candidate-frontier protocol."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.dataset_v2 import load_pit_manifest, load_quality_report, load_tradeable_panel
from traditional_quant_research.diagnostics import summarize_factor_ic
from traditional_quant_research.experiments.low_corr_exposure_grid import LOW_CORR_SIGNAL
from traditional_quant_research.experiments.low_corr_regime_yearly_validation import year_windows
from traditional_quant_research.experiments.multifactor_baseline import (
    apply_horizon_fee,
    directions_from_ic,
    filter_panel_dates,
    selected_basket_factor_exposure,
    summarize_horizon_trade_table,
)
from traditional_quant_research.horizon_backtest import horizon_aligned_top_n_backtest
from traditional_quant_research.multifactor import (
    add_equal_rank_score,
    factor_coverage,
    mean_daily_factor_correlation,
    select_low_correlation_factors,
)
from traditional_quant_research.research_panel import (
    add_cross_sectional_excess_return_labels,
    add_cross_sectional_zscores,
    build_factor_label_panel,
    default_factor_columns,
    panel_summary,
)


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/low_corr_candidate_frontier_audit")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-02_low_corr_candidate_frontier_audit.md")
DEFAULT_YEARS = (2024, 2025, 2026)
DEFAULT_FINAL_END_DATE = "2026-06-01"
DEFAULT_HORIZON = 20
DEFAULT_TOP_N = 200
DEFAULT_REBALANCE_FREQUENCY = "monthly"
DEFAULT_BUFFER_MULTIPLIER = 3.0
DEFAULT_FEE_BPS_VALUES = (0.0, 30.0, 60.0, 100.0)
DEFAULT_CAPITAL_AMOUNTS = (10_000_000.0, 50_000_000.0, 100_000_000.0)
DEFAULT_MAX_FACTOR_CORR = 0.75
LABEL_MODES = ("raw", "xsec-excess")


def build_low_corr_signal_panel(
    *,
    root: str | None,
    history_start_date: str,
    fit_start_date: str,
    fit_end_date: str,
    start_date: str,
    end_date: str,
    horizon: int,
    label_mode: str,
    max_factor_corr: float,
    include_industry: bool = False,
    include_metrics: bool = False,
) -> dict[str, Any]:
    """Build an evaluation panel with the low-corr score using a prior fit window."""

    if label_mode not in LABEL_MODES:
        raise ValueError(f"unsupported label_mode: {label_mode}")
    manifest = load_pit_manifest(root)
    quality = load_quality_report(root)
    raw_panel = load_tradeable_panel(
        root,
        start_date=history_start_date,
        end_date=end_date,
        include_industry=include_industry,
        include_metrics=include_metrics,
    )
    factor_panel = build_factor_label_panel(raw_panel, horizons=tuple(sorted({1, 5, 20, horizon})))
    factor_panel = add_cross_sectional_excess_return_labels(factor_panel, horizons=(horizon,))
    raw_factor_columns = default_factor_columns()
    factor_panel = add_cross_sectional_zscores(factor_panel, raw_factor_columns)
    signal_columns = [f"{column}_z" for column in raw_factor_columns]
    label_col = f"fwd_ret_{horizon}d" if label_mode == "raw" else f"xsec_excess_ret_{horizon}d"
    fit_panel = filter_panel_dates(factor_panel, start_date=fit_start_date, end_date=fit_end_date)
    evaluation_panel = filter_panel_dates(factor_panel, start_date=start_date, end_date=end_date)
    if fit_panel.empty:
        raise ValueError("fit panel is empty")
    if evaluation_panel.empty:
        raise ValueError("evaluation panel is empty")
    if label_col not in fit_panel.columns or label_col not in evaluation_panel.columns:
        raise ValueError(f"label not generated: {label_col}")

    single_factor_ic = summarize_factor_ic(fit_panel, signal_columns, label_col)
    factor_directions = directions_from_ic(single_factor_ic, signal_columns)
    factor_correlation = mean_daily_factor_correlation(fit_panel, signal_columns)
    low_corr_factor_columns = select_low_correlation_factors(
        factor_correlation,
        signal_columns,
        ic_summary=single_factor_ic,
        max_abs_corr=max_factor_corr,
    )
    if not low_corr_factor_columns:
        raise ValueError("low-correlation factor selection produced no factors")
    factor_panel = add_equal_rank_score(
        factor_panel,
        low_corr_factor_columns,
        directions=factor_directions,
        score_col=LOW_CORR_SIGNAL,
        min_factors=min(3, len(low_corr_factor_columns)),
    )
    evaluation_panel = filter_panel_dates(factor_panel, start_date=start_date, end_date=end_date)
    return {
        "manifest": manifest,
        "quality": quality,
        "factor_panel": factor_panel,
        "fit_panel": fit_panel,
        "evaluation_panel": evaluation_panel,
        "raw_factor_columns": raw_factor_columns,
        "signal_columns": signal_columns,
        "label": label_col,
        "single_factor_ic": single_factor_ic,
        "factor_correlation": factor_correlation,
        "factor_directions": factor_directions,
        "low_corr_factor_columns": low_corr_factor_columns,
        "coverage": factor_coverage(evaluation_panel, signal_columns),
    }


def expand_selected_holdings(
    frame: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    extra_columns: Sequence[str],
    date_col: str = "date",
    code_col: str = "code",
) -> pd.DataFrame:
    """Expand trade-level code strings into selected holding rows."""

    output_columns = [
        "trade_id",
        "signal_date",
        "entry_date",
        "exit_date",
        code_col,
        *extra_columns,
    ]
    if frame.empty or trades.empty:
        return pd.DataFrame(columns=output_columns)
    required_frame = {date_col, code_col, *extra_columns}
    required_trades = {"signal_date", "entry_date", "exit_date", "codes"}
    missing_frame = sorted(required_frame - set(frame.columns))
    missing_trades = sorted(required_trades - set(trades.columns))
    if missing_frame:
        raise ValueError(f"frame missing required columns: {missing_frame}")
    if missing_trades:
        raise ValueError(f"trades missing required columns: {missing_trades}")

    rows: list[dict[str, Any]] = []
    trade_work = trades.copy().reset_index(drop=True)
    trade_work["signal_date"] = pd.to_datetime(trade_work["signal_date"])
    trade_work["entry_date"] = pd.to_datetime(trade_work["entry_date"])
    trade_work["exit_date"] = pd.to_datetime(trade_work["exit_date"])
    for trade_id, trade in trade_work.iterrows():
        codes = [code.strip() for code in str(trade["codes"]).split(",") if code.strip()]
        for code in codes:
            rows.append(
                {
                    "trade_id": int(trade_id),
                    "signal_date": pd.Timestamp(trade["signal_date"]),
                    "entry_date": pd.Timestamp(trade["entry_date"]),
                    "exit_date": pd.Timestamp(trade["exit_date"]),
                    code_col: code,
                }
            )
    if not rows:
        return pd.DataFrame(columns=output_columns)
    selected = pd.DataFrame(rows)
    panel = frame.loc[:, [date_col, code_col, *extra_columns]].copy()
    panel[date_col] = pd.to_datetime(panel[date_col])
    panel[code_col] = panel[code_col].astype(str)
    return selected.merge(
        panel.rename(columns={date_col: "signal_date"}),
        on=["signal_date", code_col],
        how="left",
    )


def summarize_trade_liquidity(
    selected_holdings: pd.DataFrame,
    *,
    capital_amounts: Sequence[float] = DEFAULT_CAPITAL_AMOUNTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize per-trade liquidity and participation proxies."""

    trade_columns = _liquidity_trade_columns(capital_amounts)
    if selected_holdings.empty:
        return pd.DataFrame(columns=trade_columns), pd.DataFrame(columns=_liquidity_year_columns(capital_amounts))
    required = {"trade_id", "signal_date", "entry_date", "exit_date", "amount", "volume"}
    if missing := sorted(required - set(selected_holdings.columns)):
        raise ValueError(f"selected_holdings missing required columns: {missing}")

    rows: list[dict[str, Any]] = []
    for trade_id, group in selected_holdings.groupby("trade_id", sort=True):
        amount = pd.to_numeric(group["amount"], errors="coerce")
        volume = pd.to_numeric(group["volume"], errors="coerce")
        selected_count = int(len(group))
        row = {
            "trade_id": int(trade_id),
            "signal_date": pd.Timestamp(group["signal_date"].iloc[0]),
            "entry_date": pd.Timestamp(group["entry_date"].iloc[0]),
            "exit_date": pd.Timestamp(group["exit_date"].iloc[0]),
            "selected_count": selected_count,
            "amount_mean": float(amount.mean()),
            "amount_median": float(amount.median()),
            "amount_p10": float(amount.quantile(0.10)),
            "amount_min": float(amount.min()),
            "volume_mean": float(volume.mean()),
            "log_amount_mean_20d_z_mean": _optional_mean(group, "log_amount_mean_20d_z"),
            "momentum_20d_z_mean": _optional_mean(group, "momentum_20d_z"),
            "low_corr_score_mean": _optional_mean(group, LOW_CORR_SIGNAL),
        }
        for capital in capital_amounts:
            token = _capital_token(capital)
            per_name = float(capital) / max(selected_count, 1)
            participation = per_name / amount.replace(0, np.nan)
            row[f"participation_mean_{token}"] = float(participation.mean())
            row[f"participation_p95_{token}"] = float(participation.quantile(0.95))
            row[f"participation_max_{token}"] = float(participation.max())
        rows.append(row)
    per_trade = pd.DataFrame(rows)
    summary = summarize_liquidity_by_year(per_trade, capital_amounts=capital_amounts)
    return per_trade, summary


def summarize_liquidity_by_year(
    per_trade: pd.DataFrame,
    *,
    capital_amounts: Sequence[float] = DEFAULT_CAPITAL_AMOUNTS,
) -> pd.DataFrame:
    """Aggregate per-trade liquidity diagnostics by exit year."""

    if per_trade.empty:
        return pd.DataFrame(columns=_liquidity_year_columns(capital_amounts))
    work = per_trade.copy()
    work["exit_date"] = pd.to_datetime(work["exit_date"])
    work["year"] = work["exit_date"].dt.year
    rows: list[dict[str, Any]] = []
    for year, group in work.groupby("year", sort=True):
        row = {
            "year": int(year),
            "trade_count": int(len(group)),
            "mean_selected_count": float(pd.to_numeric(group["selected_count"], errors="coerce").mean()),
            "mean_amount_mean": float(pd.to_numeric(group["amount_mean"], errors="coerce").mean()),
            "min_amount_p10": float(pd.to_numeric(group["amount_p10"], errors="coerce").min()),
            "min_amount_min": float(pd.to_numeric(group["amount_min"], errors="coerce").min()),
            "mean_log_amount_z": float(pd.to_numeric(group["log_amount_mean_20d_z_mean"], errors="coerce").mean()),
            "mean_momentum_z": float(pd.to_numeric(group["momentum_20d_z_mean"], errors="coerce").mean()),
        }
        for capital in capital_amounts:
            token = _capital_token(capital)
            row[f"mean_participation_p95_{token}"] = float(pd.to_numeric(group[f"participation_p95_{token}"], errors="coerce").mean())
            row[f"worst_participation_max_{token}"] = float(pd.to_numeric(group[f"participation_max_{token}"], errors="coerce").max())
        rows.append(row)
    return pd.DataFrame(rows)


def run_low_corr_candidate_frontier_audit(
    *,
    root: str | None = None,
    years: Sequence[int] = DEFAULT_YEARS,
    final_end_date: str | None = DEFAULT_FINAL_END_DATE,
    horizon: int = DEFAULT_HORIZON,
    label_mode: str = "raw",
    max_factor_corr: float = DEFAULT_MAX_FACTOR_CORR,
    top_n: int = DEFAULT_TOP_N,
    rebalance_frequency: str = DEFAULT_REBALANCE_FREQUENCY,
    buffer_multiplier: float = DEFAULT_BUFFER_MULTIPLIER,
    fee_bps_values: Sequence[float] = DEFAULT_FEE_BPS_VALUES,
    execution_constraints: bool = True,
    limit_threshold: float = 0.095,
    capital_amounts: Sequence[float] = DEFAULT_CAPITAL_AMOUNTS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    """Run candidate-frontier protocol audit across evaluation years."""

    if not years:
        raise ValueError("years must not be empty")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if buffer_multiplier < 1.0:
        raise ValueError("buffer_multiplier must be at least 1.0")
    if not fee_bps_values:
        raise ValueError("fee_bps_values must not be empty")

    run_id = f"low_corr_candidate_frontier_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    exposure_frames: list[pd.DataFrame] = []
    holdings_frames: list[pd.DataFrame] = []
    liquidity_frames: list[pd.DataFrame] = []
    liquidity_summary_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []
    low_corr_rows: list[dict[str, Any]] = []
    manifest: Mapping[str, Any] = {}
    quality: Mapping[str, Any] = {}

    for year in years:
        windows = year_windows(int(year), final_end_date=final_end_date)
        built = build_low_corr_signal_panel(
            root=root,
            history_start_date=windows["history_start_date"],
            fit_start_date=windows["fit_start_date"],
            fit_end_date=windows["fit_end_date"],
            start_date=windows["start_date"],
            end_date=windows["end_date"],
            horizon=horizon,
            label_mode=label_mode,
            max_factor_corr=max_factor_corr,
        )
        manifest = built["manifest"]
        quality = built["quality"]
        evaluation_panel = built["evaluation_panel"]
        base_backtest = horizon_aligned_top_n_backtest(
            evaluation_panel,
            LOW_CORR_SIGNAL,
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
            base_trades.insert(0, "eval_year", int(year))
            trade_frames.append(base_trades)
        for fee_bps in fee_bps_values:
            trades = apply_horizon_fee(base_backtest.trades, fee_bps=float(fee_bps))
            row = {"eval_year": int(year), "signal": LOW_CORR_SIGNAL}
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
        exposure_columns = _unique_columns([*built["signal_columns"], *built["low_corr_factor_columns"], LOW_CORR_SIGNAL])
        exposure = selected_basket_factor_exposure(
            evaluation_panel,
            base_backtest.trades,
            exposure_columns,
            signal_col=LOW_CORR_SIGNAL,
            horizon=horizon,
            rebalance_frequency=rebalance_frequency,
            top_n=top_n,
            buffer_multiplier=buffer_multiplier,
        )
        if not exposure.empty:
            exposure.insert(0, "eval_year", int(year))
            exposure_frames.append(exposure)
        holding_columns = _unique_columns(
            [
                "amount",
                "volume",
                LOW_CORR_SIGNAL,
                "log_amount_mean_20d",
                "log_amount_mean_20d_z",
                "momentum_20d_z",
                "reversal_5d_z",
                "neg_volatility_20d_z",
                "neg_amplitude_20d_z",
            ]
        )
        selected_holdings = expand_selected_holdings(evaluation_panel, base_backtest.trades, extra_columns=[column for column in holding_columns if column in evaluation_panel.columns])
        if not selected_holdings.empty:
            selected_holdings.insert(0, "eval_year", int(year))
            holdings_frames.append(selected_holdings)
            per_trade_liquidity, yearly_liquidity = summarize_trade_liquidity(selected_holdings, capital_amounts=capital_amounts)
            if not per_trade_liquidity.empty:
                per_trade_liquidity.insert(0, "eval_year", int(year))
                liquidity_frames.append(per_trade_liquidity)
            if not yearly_liquidity.empty:
                yearly_liquidity.insert(0, "eval_year", int(year))
                liquidity_summary_frames.append(yearly_liquidity)
        coverage = built["coverage"].copy()
        if not coverage.empty:
            coverage.insert(0, "eval_year", int(year))
            coverage_frames.append(coverage)
        low_corr_rows.append(
            {
                "eval_year": int(year),
                "history_start_date": windows["history_start_date"],
                "fit_start_date": windows["fit_start_date"],
                "fit_end_date": windows["fit_end_date"],
                "start_date": windows["start_date"],
                "end_date": windows["end_date"],
                "low_corr_factor_columns": json.dumps(built["low_corr_factor_columns"], ensure_ascii=False),
                "evaluation_rows": int(len(evaluation_panel)),
                "evaluation_dates": int(evaluation_panel["date"].nunique()) if "date" in evaluation_panel.columns else 0,
                "evaluation_securities": int(evaluation_panel["code"].nunique()) if "code" in evaluation_panel.columns else 0,
            }
        )

    summary = pd.DataFrame(summary_rows)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    exposure_summary = pd.concat(exposure_frames, ignore_index=True) if exposure_frames else pd.DataFrame()
    selected_holdings = pd.concat(holdings_frames, ignore_index=True) if holdings_frames else pd.DataFrame()
    trade_liquidity = pd.concat(liquidity_frames, ignore_index=True) if liquidity_frames else pd.DataFrame()
    liquidity_summary = pd.concat(liquidity_summary_frames, ignore_index=True) if liquidity_summary_frames else pd.DataFrame()
    coverage = pd.concat(coverage_frames, ignore_index=True) if coverage_frames else pd.DataFrame()
    low_corr_meta = pd.DataFrame(low_corr_rows)
    aggregate = summarize_candidate_audit(summary)

    result = {
        "run_id": run_id,
        "snapshot_id": manifest.get("snapshot_id"),
        "years": [int(year) for year in years],
        "final_end_date": final_end_date,
        "horizon": int(horizon),
        "rebalance_frequency": rebalance_frequency,
        "top_n": int(top_n),
        "buffer_multiplier": float(buffer_multiplier),
        "fee_bps_values": [float(value) for value in fee_bps_values],
        "execution_constraints": bool(execution_constraints),
        "limit_threshold": float(limit_threshold),
        "capital_amounts": [float(value) for value in capital_amounts],
        "quality": {
            "failure_count": quality.get("failure_count"),
            "missing_bar_rows": quality.get("missing_bar_rows"),
            "st_rows": quality.get("st_rows"),
            "suspended_like_rows": quality.get("suspended_like_rows"),
        },
        "best_fee_rows": best_fee_rows(aggregate),
        "candidate_count": 0,
        "assessment": "candidate-frontier audit; no strategy candidate is promoted until capacity, exposure, and stress gates pass",
        "output_dir": str(run_dir),
    }

    summary.to_csv(run_dir / "candidate_protocol_summary.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(run_dir / "candidate_protocol_aggregate.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(run_dir / "candidate_protocol_trades.csv", index=False, encoding="utf-8-sig")
    exposure_summary.to_csv(run_dir / "candidate_basket_exposure.csv", index=False, encoding="utf-8-sig")
    selected_holdings.to_csv(run_dir / "candidate_selected_holdings.csv", index=False, encoding="utf-8-sig")
    trade_liquidity.to_csv(run_dir / "candidate_trade_liquidity.csv", index=False, encoding="utf-8-sig")
    liquidity_summary.to_csv(run_dir / "candidate_liquidity_summary.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(run_dir / "candidate_factor_coverage.csv", index=False, encoding="utf-8-sig")
    low_corr_meta.to_csv(run_dir / "candidate_low_corr_meta.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_candidate_audit_markdown(result, aggregate, liquidity_summary, exposure_summary)
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        research_log_path.write_text(markdown, encoding="utf-8")
    return result


def summarize_candidate_audit(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate candidate protocol rows by fee level."""

    if summary.empty:
        return pd.DataFrame()
    required = {"fee_bps", "annualized_return", "sharpe", "max_drawdown", "mean_turnover", "eval_year"}
    if missing := sorted(required - set(summary.columns)):
        raise ValueError(f"summary missing required columns: {missing}")
    work = summary.copy()
    zero_fee = work.loc[pd.to_numeric(work["fee_bps"], errors="coerce") == 0.0, ["eval_year", "annualized_return"]].rename(
        columns={"annualized_return": "annualized_return_0bps"}
    )
    work = work.merge(zero_fee, on="eval_year", how="left")
    work["cost_drag_vs_0bps"] = pd.to_numeric(work["annualized_return"], errors="coerce") - pd.to_numeric(
        work["annualized_return_0bps"],
        errors="coerce",
    )

    rows: list[dict[str, Any]] = []
    for fee_bps, group in work.groupby("fee_bps", sort=True):
        returns = pd.to_numeric(group["annualized_return"], errors="coerce")
        rows.append(
            {
                "fee_bps": float(fee_bps),
                "eval_year_count": int(group["eval_year"].nunique()),
                "mean_annualized_return": float(returns.mean()),
                "min_annualized_return": float(returns.min()),
                "positive_year_rate": float((returns > 0).mean()),
                "mean_sharpe": float(pd.to_numeric(group["sharpe"], errors="coerce").mean()),
                "worst_max_drawdown": float(pd.to_numeric(group["max_drawdown"], errors="coerce").min()),
                "mean_turnover": float(pd.to_numeric(group["mean_turnover"], errors="coerce").mean()),
                "mean_cost_drag_vs_0bps": float(pd.to_numeric(group["cost_drag_vs_0bps"], errors="coerce").mean()),
                "total_periods": int(pd.to_numeric(group["periods"], errors="coerce").sum()) if "periods" in group.columns else 0,
            }
        )
    return pd.DataFrame(rows)


def best_fee_rows(aggregate: pd.DataFrame) -> list[dict[str, Any]]:
    if aggregate.empty:
        return []
    return aggregate.sort_values("fee_bps").to_dict("records")


def render_candidate_audit_markdown(
    result: Mapping[str, Any],
    aggregate: pd.DataFrame,
    liquidity_summary: pd.DataFrame,
    exposure_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Low-Corr Candidate-Frontier Audit",
        "",
        "- Hypothesis: the strongest current protocol needs exposure, liquidity, trade-count, and fee-stress auditing before promotion.",
        f"- Protocol: `horizon={result.get('horizon')}`, `{result.get('rebalance_frequency')}`, `top_n={result.get('top_n')}`, `buffer={result.get('buffer_multiplier')}`.",
        f"- Years: `{result.get('years')}`; final end date `{result.get('final_end_date')}`.",
        f"- Fees: `{result.get('fee_bps_values')}` bps.",
        "- Assessment: candidate-frontier only; candidate count remains `0` until all promotion gates pass.",
        f"- Artifacts: `{result.get('output_dir')}`",
        "",
        "## Fee Stress",
        "",
        _markdown_table(aggregate),
        "",
        "## Liquidity Summary",
        "",
        _markdown_table(liquidity_summary),
        "",
        "## Basket Exposure",
        "",
        _markdown_table(exposure_summary),
        "",
    ]
    return "\n".join(lines)


def _optional_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.mean()) if not values.dropna().empty else np.nan


def _capital_token(value: float) -> str:
    if value >= 1_000_000:
        return f"{int(round(value / 1_000_000))}m"
    return f"{int(round(value))}"


def _liquidity_trade_columns(capital_amounts: Sequence[float]) -> list[str]:
    columns = [
        "trade_id",
        "signal_date",
        "entry_date",
        "exit_date",
        "selected_count",
        "amount_mean",
        "amount_median",
        "amount_p10",
        "amount_min",
        "volume_mean",
        "log_amount_mean_20d_z_mean",
        "momentum_20d_z_mean",
        "low_corr_score_mean",
    ]
    for capital in capital_amounts:
        token = _capital_token(capital)
        columns.extend(
            [
                f"participation_mean_{token}",
                f"participation_p95_{token}",
                f"participation_max_{token}",
            ]
        )
    return columns


def _liquidity_year_columns(capital_amounts: Sequence[float]) -> list[str]:
    columns = [
        "year",
        "trade_count",
        "mean_selected_count",
        "mean_amount_mean",
        "min_amount_p10",
        "min_amount_min",
        "mean_log_amount_z",
        "mean_momentum_z",
    ]
    for capital in capital_amounts:
        token = _capital_token(capital)
        columns.extend(
            [
                f"mean_participation_p95_{token}",
                f"worst_participation_max_{token}",
            ]
        )
    return columns


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
    parser.add_argument("--max-factor-corr", type=float, default=DEFAULT_MAX_FACTOR_CORR)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--frequency", default=DEFAULT_REBALANCE_FREQUENCY)
    parser.add_argument("--buffer-multiplier", type=float, default=DEFAULT_BUFFER_MULTIPLIER)
    parser.add_argument("--fee-bps", default=",".join(str(value) for value in DEFAULT_FEE_BPS_VALUES))
    parser.add_argument("--execution-constraints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-threshold", type=float, default=0.095)
    parser.add_argument("--capital-amounts", default=",".join(str(value) for value in DEFAULT_CAPITAL_AMOUNTS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_low_corr_candidate_frontier_audit(
        root=args.root,
        years=_parse_years(args.years),
        final_end_date=args.final_end_date,
        horizon=args.horizon,
        label_mode=args.label_mode,
        max_factor_corr=args.max_factor_corr,
        top_n=args.top_n,
        rebalance_frequency=args.frequency,
        buffer_multiplier=args.buffer_multiplier,
        fee_bps_values=_parse_float_tuple(args.fee_bps),
        execution_constraints=args.execution_constraints,
        limit_threshold=args.limit_threshold,
        capital_amounts=_parse_float_tuple(args.capital_amounts),
        output_dir=args.output_dir,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
