"""Audit frontier signals after same-date proxy neutralization."""

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
)
from traditional_quant_research.experiments.low_corr_candidate_signal_comparison import (
    IC_WEIGHTED_SIGNAL,
    LOW_CORR_SIGNAL,
    ROLLING_IC_SIGNAL,
    build_candidate_protocol_signal_panel,
)
from traditional_quant_research.experiments.low_corr_regime_yearly_validation import year_windows
from traditional_quant_research.experiments.multifactor_baseline import (
    apply_horizon_fee,
    selected_basket_factor_exposure,
    summarize_horizon_trade_table,
)
from traditional_quant_research.horizon_backtest import horizon_aligned_top_n_backtest
from traditional_quant_research.multifactor import factor_coverage, neutralize_factors_by_date


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/low_corr_frontier_neutralization_audit")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-02_low_corr_frontier_neutralization_audit.md")
DEFAULT_FRONTIER_SIGNALS = (
    ROLLING_IC_SIGNAL,
    IC_WEIGHTED_SIGNAL,
    LOW_CORR_SIGNAL,
)
DEFAULT_NEUTRALIZE_BY = ("log_amount_mean_20d_z",)
DEFAULT_EXPOSURE_COLUMNS = (
    "log_amount_mean_20d_z",
    "momentum_20d_z",
    "reversal_5d_z",
    "neg_volatility_20d_z",
    "neg_amplitude_20d_z",
)
DEFAULT_FEE_BPS_VALUES = (30.0,)
DEFAULT_ROLLING_WINDOW = 252
DEFAULT_ROLLING_MIN_PERIODS = 60
NEUTRAL_SUFFIX = "_proxy_neutral"


def add_proxy_neutralized_signals(
    frame: pd.DataFrame,
    signals: Sequence[str],
    neutralizer_cols: Sequence[str],
    *,
    suffix: str = NEUTRAL_SUFFIX,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Add same-date residualized signal columns and return original->neutral map."""

    selected_signals = _normalize_tuple(signals, name="signals")
    neutralizers = _normalize_tuple(neutralizer_cols, name="neutralizer_cols")
    output = neutralize_factors_by_date(
        frame,
        selected_signals,
        neutralizers,
        suffix=suffix,
    )
    mapping = {signal: f"{signal}{suffix}" for signal in selected_signals}
    return output, mapping


def run_low_corr_frontier_neutralization_audit(
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
    neutralize_by: Sequence[str] = DEFAULT_NEUTRALIZE_BY,
    exposure_columns: Sequence[str] = DEFAULT_EXPOSURE_COLUMNS,
    top_n: int = DEFAULT_TOP_N,
    rebalance_frequency: str = DEFAULT_REBALANCE_FREQUENCY,
    buffer_multiplier: float = DEFAULT_BUFFER_MULTIPLIER,
    fee_bps_values: Sequence[float] = DEFAULT_FEE_BPS_VALUES,
    execution_constraints: bool = True,
    limit_threshold: float = 0.095,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    """Run frontier signal comparison before/after proxy neutralization."""

    if not years:
        raise ValueError("years must not be empty")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if label_mode not in LABEL_MODES:
        raise ValueError(f"unsupported label_mode: {label_mode}")
    if rolling_window <= 0:
        raise ValueError("rolling_window must be positive")
    if rolling_min_periods <= 0:
        raise ValueError("rolling_min_periods must be positive")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if buffer_multiplier < 1.0:
        raise ValueError("buffer_multiplier must be at least 1.0")
    selected_signals = _normalize_tuple(signals, name="signals")
    neutralizers = _normalize_tuple(neutralize_by, name="neutralize_by")
    fee_values = _normalize_float_tuple(fee_bps_values, name="fee_bps_values")

    run_id = f"low_corr_frontier_neutralization_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    exposure_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []
    correlation_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    manifest: Mapping[str, Any] = {}
    quality: Mapping[str, Any] = {}
    neutralizer_spec = ",".join(neutralizers)

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
        )
        manifest = built["manifest"]
        quality = built["quality"]
        evaluation_panel = built["evaluation_panel"].copy()
        missing_signals = sorted(set(selected_signals) - set(built["available_signals"]))
        if missing_signals:
            raise ValueError(f"signals not available in panel: {missing_signals}")
        missing_neutralizers = sorted(set(neutralizers) - set(evaluation_panel.columns))
        if missing_neutralizers:
            raise ValueError(f"neutralizers not available in panel: {missing_neutralizers}")

        neutral_panel, neutral_signal_map = add_proxy_neutralized_signals(
            evaluation_panel,
            selected_signals,
            neutralizers,
        )
        variant_signals = [*selected_signals, *neutral_signal_map.values()]
        coverage = factor_coverage(neutral_panel, variant_signals)
        if not coverage.empty:
            coverage.insert(0, "eval_year", int(year))
            coverage_frames.append(coverage)
        correlation = signal_neutralizer_correlation(neutral_panel, variant_signals, neutralizers)
        if not correlation.empty:
            correlation.insert(0, "eval_year", int(year))
            correlation_frames.append(correlation)

        available_exposures = _unique_columns([*(exposure_columns or ()), *neutralizers])
        available_exposures = [column for column in available_exposures if column in neutral_panel.columns]
        for base_signal in selected_signals:
            variants = [
                {
                    "variant_signal": base_signal,
                    "neutralized": False,
                    "neutralizer_spec": "none",
                },
                {
                    "variant_signal": neutral_signal_map[base_signal],
                    "neutralized": True,
                    "neutralizer_spec": neutralizer_spec,
                },
            ]
            for variant in variants:
                variant_signal = str(variant["variant_signal"])
                base_backtest = horizon_aligned_top_n_backtest(
                    neutral_panel,
                    variant_signal,
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
                    trades_out = base_trades.copy()
                    trades_out.insert(0, "neutralizer_spec", str(variant["neutralizer_spec"]))
                    trades_out.insert(0, "neutralized", bool(variant["neutralized"]))
                    trades_out.insert(0, "variant_signal", variant_signal)
                    trades_out.insert(0, "base_signal", base_signal)
                    trades_out.insert(0, "eval_year", int(year))
                    trade_frames.append(trades_out)
                if available_exposures:
                    exposure = selected_basket_factor_exposure(
                        neutral_panel,
                        base_trades,
                        available_exposures,
                        signal_col=variant_signal,
                        horizon=horizon,
                        rebalance_frequency=rebalance_frequency,
                        top_n=top_n,
                        buffer_multiplier=buffer_multiplier,
                    )
                    if not exposure.empty:
                        exposure.insert(0, "neutralizer_spec", str(variant["neutralizer_spec"]))
                        exposure.insert(0, "neutralized", bool(variant["neutralized"]))
                        exposure.insert(0, "variant_signal", variant_signal)
                        exposure.insert(0, "base_signal", base_signal)
                        exposure.insert(0, "eval_year", int(year))
                        exposure_frames.append(exposure)
                for fee_bps in fee_values:
                    trades = apply_horizon_fee(base_trades, fee_bps=float(fee_bps))
                    row = {
                        "eval_year": int(year),
                        "base_signal": base_signal,
                        "variant_signal": variant_signal,
                        "neutralized": bool(variant["neutralized"]),
                        "neutralizer_spec": str(variant["neutralizer_spec"]),
                    }
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

        metadata_rows.append(
            {
                "eval_year": int(year),
                "history_start_date": windows["history_start_date"],
                "fit_start_date": windows["fit_start_date"],
                "fit_end_date": windows["fit_end_date"],
                "start_date": windows["start_date"],
                "end_date": windows["end_date"],
                "available_signals": json.dumps(list(selected_signals), ensure_ascii=False),
                "neutralize_by": json.dumps(list(neutralizers), ensure_ascii=False),
                "rolling_fallback_rate": float(built["rolling_fallback_rate"]),
                "evaluation_rows": int(len(neutral_panel)),
                "evaluation_dates": int(neutral_panel["date"].nunique()) if "date" in neutral_panel.columns else 0,
                "evaluation_securities": int(neutral_panel["code"].nunique()) if "code" in neutral_panel.columns else 0,
            }
        )

    summary = pd.DataFrame(summary_rows)
    aggregate = summarize_neutralization_audit(summary)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    exposure_summary = pd.concat(exposure_frames, ignore_index=True) if exposure_frames else pd.DataFrame()
    signal_coverage = pd.concat(coverage_frames, ignore_index=True) if coverage_frames else pd.DataFrame()
    signal_neutralizer_corr = pd.concat(correlation_frames, ignore_index=True) if correlation_frames else pd.DataFrame()
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
        "neutralize_by": list(neutralizers),
        "rebalance_frequency": rebalance_frequency,
        "top_n": int(top_n),
        "buffer_multiplier": float(buffer_multiplier),
        "fee_bps_values": [float(value) for value in fee_values],
        "execution_constraints": bool(execution_constraints),
        "limit_threshold": float(limit_threshold),
        "quality": {
            "failure_count": quality.get("failure_count"),
            "missing_bar_rows": quality.get("missing_bar_rows"),
            "st_rows": quality.get("st_rows"),
            "suspended_like_rows": quality.get("suspended_like_rows"),
        },
        "best_30bps_rows": best_neutralization_rows(aggregate, fee_bps=30.0),
        "candidate_count": 0,
        "assessment": "proxy-neutralization evidence only; no strategy candidate is promoted by this experiment alone",
        "output_dir": str(run_dir),
    }

    summary.to_csv(run_dir / "neutralization_summary.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(run_dir / "neutralization_aggregate.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(run_dir / "neutralization_trades.csv", index=False, encoding="utf-8-sig")
    exposure_summary.to_csv(run_dir / "neutralization_basket_exposure.csv", index=False, encoding="utf-8-sig")
    signal_coverage.to_csv(run_dir / "neutralization_signal_coverage.csv", index=False, encoding="utf-8-sig")
    signal_neutralizer_corr.to_csv(run_dir / "neutralization_signal_neutralizer_correlation.csv", index=False, encoding="utf-8-sig")
    metadata.to_csv(run_dir / "neutralization_meta.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_neutralization_markdown(result, aggregate, summary, exposure_summary, signal_neutralizer_corr)
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        research_log_path.write_text(markdown, encoding="utf-8")
    return result


def summarize_neutralization_audit(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate original and proxy-neutralized rows by signal and fee."""

    if summary.empty:
        return pd.DataFrame()
    required = {
        "eval_year",
        "base_signal",
        "variant_signal",
        "neutralized",
        "neutralizer_spec",
        "fee_bps",
        "annualized_return",
        "sharpe",
        "max_drawdown",
        "mean_turnover",
    }
    if missing := sorted(required - set(summary.columns)):
        raise ValueError(f"summary missing required columns: {missing}")

    work = summary.copy()
    work["neutralized"] = work["neutralized"].astype(bool)
    original = work.loc[
        ~work["neutralized"],
        ["eval_year", "base_signal", "fee_bps", "annualized_return"],
    ].rename(columns={"annualized_return": "original_annualized_return"})
    work = work.merge(original, on=["eval_year", "base_signal", "fee_bps"], how="left")
    work["delta_annualized_return_vs_original"] = pd.to_numeric(work["annualized_return"], errors="coerce") - pd.to_numeric(
        work["original_annualized_return"],
        errors="coerce",
    )

    rows: list[dict[str, Any]] = []
    group_cols = ["base_signal", "variant_signal", "neutralized", "neutralizer_spec", "fee_bps"]
    for keys, group in work.groupby(group_cols, sort=True):
        base_signal, variant_signal, neutralized, neutralizer_spec, fee_bps = keys
        returns = pd.to_numeric(group["annualized_return"], errors="coerce")
        deltas = pd.to_numeric(group["delta_annualized_return_vs_original"], errors="coerce")
        rows.append(
            {
                "base_signal": base_signal,
                "variant_signal": variant_signal,
                "neutralized": bool(neutralized),
                "neutralizer_spec": neutralizer_spec,
                "fee_bps": float(fee_bps),
                "eval_year_count": int(group["eval_year"].nunique()),
                "mean_annualized_return": float(returns.mean()),
                "min_annualized_return": float(returns.min()),
                "positive_year_rate": float((returns > 0).mean()),
                "mean_sharpe": float(pd.to_numeric(group["sharpe"], errors="coerce").mean()),
                "worst_max_drawdown": float(pd.to_numeric(group["max_drawdown"], errors="coerce").min()),
                "mean_turnover": float(pd.to_numeric(group["mean_turnover"], errors="coerce").mean()),
                "mean_delta_annualized_return_vs_original": float(deltas.mean()) if not deltas.dropna().empty else np.nan,
                "positive_delta_year_rate_vs_original": float((deltas > 0).mean()) if not deltas.dropna().empty else np.nan,
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


def signal_neutralizer_correlation(
    frame: pd.DataFrame,
    signal_cols: Sequence[str],
    neutralizer_cols: Sequence[str],
    *,
    date_col: str = "date",
    method: str = "pearson",
) -> pd.DataFrame:
    """Summarize daily cross-sectional correlations between signals and neutralizers."""

    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")
    signals = _normalize_tuple(signal_cols, name="signal_cols")
    neutralizers = _normalize_tuple(neutralizer_cols, name="neutralizer_cols")
    missing = sorted({date_col, *signals, *neutralizers} - set(frame.columns))
    if missing:
        raise ValueError(f"frame missing required columns: {missing}")
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "signal",
                "neutralizer",
                "method",
                "daily_count",
                "mean_daily_corr",
                "mean_abs_daily_corr",
                "max_abs_daily_corr",
            ]
        )

    work = frame.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    rows: list[dict[str, Any]] = []
    for signal in signals:
        for neutralizer in neutralizers:
            daily_corrs: list[float] = []
            for _, group in work.groupby(date_col, sort=True):
                clean = group.loc[:, [signal, neutralizer]].replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce").dropna()
                if len(clean) < 2 or clean[signal].nunique() < 2 or clean[neutralizer].nunique() < 2:
                    continue
                if clean[signal].std(ddof=0) <= 1e-12 or clean[neutralizer].std(ddof=0) <= 1e-12:
                    continue
                corr = clean[signal].corr(clean[neutralizer], method=method)
                if pd.notna(corr):
                    daily_corrs.append(float(corr))
            values = pd.Series(daily_corrs, dtype=float)
            rows.append(
                {
                    "signal": signal,
                    "neutralizer": neutralizer,
                    "method": method,
                    "daily_count": int(len(values)),
                    "mean_daily_corr": float(values.mean()) if not values.empty else np.nan,
                    "mean_abs_daily_corr": float(values.abs().mean()) if not values.empty else np.nan,
                    "max_abs_daily_corr": float(values.abs().max()) if not values.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def best_neutralization_rows(aggregate: pd.DataFrame, *, fee_bps: float = 30.0, top: int = 12) -> list[dict[str, Any]]:
    if aggregate.empty:
        return []
    rows = aggregate.loc[pd.to_numeric(aggregate["fee_bps"], errors="coerce") == float(fee_bps)]
    if rows.empty:
        rows = aggregate
    return rows.sort_values(
        ["mean_annualized_return", "positive_year_rate", "worst_max_drawdown"],
        ascending=[False, False, False],
    ).head(top).to_dict("records")


def render_neutralization_markdown(
    result: Mapping[str, Any],
    aggregate: pd.DataFrame,
    summary: pd.DataFrame,
    exposure_summary: pd.DataFrame,
    signal_neutralizer_corr: pd.DataFrame,
) -> str:
    best_rows = pd.DataFrame(result.get("best_30bps_rows") or [])
    lines = [
        "# Low-Corr Frontier Neutralization Audit",
        "",
        "- Hypothesis: frontier signals should not rely entirely on same-date size/liquidity proxy exposure.",
        f"- Protocol: `horizon={result.get('horizon')}`, `{result.get('rebalance_frequency')}`, `top_n={result.get('top_n')}`, `buffer={result.get('buffer_multiplier')}`.",
        f"- Signals: `{result.get('signals')}`.",
        f"- Neutralizers: `{result.get('neutralize_by')}`.",
        "- Assessment: proxy neutralization evidence only; candidate count remains `0` until all promotion gates pass.",
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
        "## Basket Exposure",
        "",
        _markdown_table(exposure_summary),
        "",
        "## Signal-Neutralizer Correlation",
        "",
        _markdown_table(signal_neutralizer_corr),
        "",
    ]
    return "\n".join(lines)


def _normalize_tuple(values: Sequence[str] | str, *, name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raw_values = values.split(",")
    else:
        raw_values = values
    output = tuple(value.strip() for value in raw_values if value and value.strip())
    if not output:
        raise ValueError(f"{name} must not be empty")
    return output


def _normalize_float_tuple(values: Sequence[float] | str, *, name: str) -> tuple[float, ...]:
    if isinstance(values, str):
        raw_values = values.split(",")
    else:
        raw_values = values
    output = tuple(float(value) for value in raw_values if str(value).strip())
    if not output:
        raise ValueError(f"{name} must not be empty")
    return output


def _parse_years(spec: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in spec.split(",") if value.strip())
    if not values:
        raise ValueError("expected at least one year")
    return values


def _unique_columns(columns: Sequence[str]) -> list[str]:
    output: list[str] = []
    for column in columns:
        if column and column not in output:
            output.append(column)
    return output


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
    parser.add_argument("--neutralize-by", default=",".join(DEFAULT_NEUTRALIZE_BY))
    parser.add_argument("--exposure-columns", default=",".join(DEFAULT_EXPOSURE_COLUMNS))
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--frequency", default=DEFAULT_REBALANCE_FREQUENCY)
    parser.add_argument("--buffer-multiplier", type=float, default=DEFAULT_BUFFER_MULTIPLIER)
    parser.add_argument("--fee-bps", default=",".join(str(value) for value in DEFAULT_FEE_BPS_VALUES))
    parser.add_argument("--execution-constraints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-threshold", type=float, default=0.095)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_low_corr_frontier_neutralization_audit(
        root=args.root,
        years=_parse_years(args.years),
        final_end_date=args.final_end_date,
        horizon=args.horizon,
        label_mode=args.label_mode,
        max_factor_corr=args.max_factor_corr,
        rolling_window=args.rolling_window,
        rolling_min_periods=args.rolling_min_periods,
        signals=_normalize_tuple(args.signals, name="signals"),
        neutralize_by=_normalize_tuple(args.neutralize_by, name="neutralize_by"),
        exposure_columns=_normalize_tuple(args.exposure_columns, name="exposure_columns"),
        top_n=args.top_n,
        rebalance_frequency=args.frequency,
        buffer_multiplier=args.buffer_multiplier,
        fee_bps_values=_normalize_float_tuple(args.fee_bps, name="fee_bps_values"),
        execution_constraints=args.execution_constraints,
        limit_threshold=args.limit_threshold,
        output_dir=args.output_dir,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
