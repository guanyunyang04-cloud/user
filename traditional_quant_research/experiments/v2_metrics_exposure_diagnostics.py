"""Diagnose frontier signal exposure to v2.1 daily metrics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.dataset_v2 import load_pit_manifest, load_tradeable_panel
from traditional_quant_research.diagnostics import summarize_factor_ic
from traditional_quant_research.experiments.low_corr_regime_yearly_validation import year_windows
from traditional_quant_research.experiments.multifactor_baseline import (
    directions_from_ic,
    filter_panel_dates,
    selected_basket_factor_exposure,
)
from traditional_quant_research.horizon_backtest import horizon_aligned_top_n_backtest
from traditional_quant_research.multifactor import (
    add_equal_rank_score,
    add_ic_weighted_rank_score,
    add_rank_score_from_weight_table,
    mean_daily_factor_correlation,
    rolling_ic_weights_by_date,
    select_low_correlation_factors,
)
from traditional_quant_research.research_panel import (
    add_cross_sectional_excess_return_labels,
    add_cross_sectional_zscores,
    build_factor_label_panel,
    default_factor_columns,
)


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/v2_metrics_exposure_diagnostics")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/brain/references/research_log/2026-06-03_v2_metrics_exposure_diagnostics.md")
DEFAULT_YEARS = (2024, 2025, 2026)
DEFAULT_FINAL_END_DATE = "2026-06-01"
DEFAULT_HORIZON = 20
DEFAULT_TOP_N = 200
DEFAULT_REBALANCE_FREQUENCY = "monthly"
DEFAULT_BUFFER_MULTIPLIER = 3.0
DEFAULT_MAX_FACTOR_CORR = 0.75
DEFAULT_ROLLING_WINDOW = 252
DEFAULT_ROLLING_MIN_PERIODS = 60
FRONTIER_SIGNALS = (
    "multifactor_ic_weighted_score",
    "multifactor_rolling_ic_weighted_score",
    "multifactor_low_corr_rank_score",
)
VALUATION_FIELDS = ("peTTM", "pbMRQ", "psTTM", "pcfNcfTTM")
BASE_EXPOSURE_FIELDS = ("turn", "pctChg")


def run_v2_metrics_exposure_diagnostics(
    *,
    root: str | None = None,
    years: Sequence[int] = DEFAULT_YEARS,
    final_end_date: str | None = DEFAULT_FINAL_END_DATE,
    horizon: int = DEFAULT_HORIZON,
    top_n: int = DEFAULT_TOP_N,
    rebalance_frequency: str = DEFAULT_REBALANCE_FREQUENCY,
    buffer_multiplier: float = DEFAULT_BUFFER_MULTIPLIER,
    max_factor_corr: float = DEFAULT_MAX_FACTOR_CORR,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    rolling_min_periods: int = DEFAULT_ROLLING_MIN_PERIODS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    if not years:
        raise ValueError("years must not be empty")
    run_id = f"v2_metrics_exposure_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_pit_manifest(root)

    daily_frames: list[pd.DataFrame] = []
    yearly_frames: list[pd.DataFrame] = []
    basket_frames: list[pd.DataFrame] = []
    meta_rows: list[dict[str, Any]] = []
    for year in years:
        windows = year_windows(int(year), final_end_date=final_end_date)
        built = build_frontier_signal_panel(
            root=root,
            history_start_date=windows["history_start_date"],
            fit_start_date=windows["fit_start_date"],
            fit_end_date=windows["fit_end_date"],
            start_date=windows["start_date"],
            end_date=windows["end_date"],
            horizon=horizon,
            max_factor_corr=max_factor_corr,
            rolling_window=rolling_window,
            rolling_min_periods=rolling_min_periods,
        )
        panel = built["evaluation_panel"]
        exposure_columns = built["exposure_z_columns"]
        signal_columns = [signal for signal in FRONTIER_SIGNALS if signal in panel.columns]
        daily_corr = daily_signal_exposure_correlation(panel, signal_columns, exposure_columns)
        if not daily_corr.empty:
            daily_corr.insert(0, "eval_year", int(year))
            daily_frames.append(daily_corr)
            yearly_frames.append(summarize_signal_exposure_correlation(daily_corr, eval_year=int(year)))
        for signal in signal_columns:
            backtest = horizon_aligned_top_n_backtest(
                panel,
                signal,
                horizon=horizon,
                top_n=top_n,
                fee_bps=0.0,
                rebalance_frequency=rebalance_frequency,
                buffer_multiplier=buffer_multiplier,
                execution_constraints=True,
            )
            basket = selected_basket_factor_exposure(
                panel,
                backtest.trades,
                exposure_columns,
                signal_col=signal,
                horizon=horizon,
                rebalance_frequency=rebalance_frequency,
                top_n=top_n,
                buffer_multiplier=buffer_multiplier,
            )
            if not basket.empty:
                basket.insert(0, "eval_year", int(year))
                basket_frames.append(basket)
        meta_rows.append(
            {
                "eval_year": int(year),
                **windows,
                "rows": int(len(panel)),
                "dates": int(panel["date"].nunique()) if "date" in panel.columns else 0,
                "codes": int(panel["code"].nunique()) if "code" in panel.columns else 0,
                "low_corr_factor_columns": json.dumps(built["low_corr_factor_columns"], ensure_ascii=False),
                "exposure_z_columns": json.dumps(exposure_columns, ensure_ascii=False),
            }
        )

    daily_correlation = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    yearly_correlation = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    basket_exposure = pd.concat(basket_frames, ignore_index=True) if basket_frames else pd.DataFrame()
    basket_summary = summarize_basket_metric_exposure(basket_exposure)
    meta = pd.DataFrame(meta_rows)
    result = summarize_run(
        run_id=run_id,
        manifest=manifest,
        years=years,
        final_end_date=final_end_date,
        yearly_correlation=yearly_correlation,
        basket_summary=basket_summary,
        run_dir=run_dir,
    )

    daily_correlation.to_csv(run_dir / "daily_signal_metric_correlation.csv", index=False, encoding="utf-8-sig")
    yearly_correlation.to_csv(run_dir / "yearly_signal_metric_correlation.csv", index=False, encoding="utf-8-sig")
    basket_exposure.to_csv(run_dir / "basket_metric_exposure.csv", index=False, encoding="utf-8-sig")
    basket_summary.to_csv(run_dir / "basket_metric_exposure_summary.csv", index=False, encoding="utf-8-sig")
    meta.to_csv(run_dir / "year_meta.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_metrics_exposure_markdown(result, yearly_correlation, basket_summary)
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        research_log_path.parent.mkdir(parents=True, exist_ok=True)
        research_log_path.write_text(markdown, encoding="utf-8")
    return result


def build_frontier_signal_panel(
    *,
    root: str | None,
    history_start_date: str,
    fit_start_date: str,
    fit_end_date: str,
    start_date: str,
    end_date: str,
    horizon: int,
    max_factor_corr: float,
    rolling_window: int,
    rolling_min_periods: int,
) -> dict[str, Any]:
    raw_panel = load_tradeable_panel(root, start_date=history_start_date, end_date=end_date, include_metrics=True)
    factor_panel = build_factor_label_panel(raw_panel, horizons=tuple(sorted({1, 5, 20, horizon})))
    factor_panel = add_cross_sectional_excess_return_labels(factor_panel, horizons=(horizon,))
    raw_factor_columns = default_factor_columns()
    factor_panel = add_cross_sectional_zscores(factor_panel, raw_factor_columns)
    signal_columns = [f"{column}_z" for column in raw_factor_columns]
    label_col = f"fwd_ret_{horizon}d"
    fit_panel = filter_panel_dates(factor_panel, start_date=fit_start_date, end_date=fit_end_date)
    single_factor_ic = summarize_factor_ic(fit_panel, signal_columns, label_col)
    factor_directions = directions_from_ic(single_factor_ic, signal_columns)
    factor_panel = add_ic_weighted_rank_score(
        factor_panel,
        signal_columns,
        single_factor_ic,
        directions=factor_directions,
        score_col="multifactor_ic_weighted_score",
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
        score_col="multifactor_rolling_ic_weighted_score",
        min_factors=3,
    )
    factor_correlation = mean_daily_factor_correlation(fit_panel, signal_columns)
    low_corr_factor_columns = select_low_correlation_factors(
        factor_correlation,
        signal_columns,
        ic_summary=single_factor_ic,
        max_abs_corr=max_factor_corr,
    )
    if low_corr_factor_columns:
        factor_panel = add_equal_rank_score(
            factor_panel,
            low_corr_factor_columns,
            directions=factor_directions,
            score_col="multifactor_low_corr_rank_score",
            min_factors=min(3, len(low_corr_factor_columns)),
        )
    factor_panel, exposure_z_columns = add_metric_exposure_fields(factor_panel)
    evaluation_panel = filter_panel_dates(factor_panel, start_date=start_date, end_date=end_date)
    return {
        "evaluation_panel": evaluation_panel,
        "low_corr_factor_columns": low_corr_factor_columns,
        "exposure_z_columns": exposure_z_columns,
    }


def add_metric_exposure_fields(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    output = frame.copy()
    if output.empty:
        return output, []
    output["date"] = pd.to_datetime(output["date"])
    output = output.sort_values(["code", "date"]).reset_index(drop=True)
    for field in (*BASE_EXPOSURE_FIELDS, *VALUATION_FIELDS):
        if field in output.columns:
            output[field] = pd.to_numeric(output[field], errors="coerce")
    for field in VALUATION_FIELDS:
        if field in output.columns:
            output[f"{field}_lag1"] = output.groupby("code", sort=False)[field].shift(1)
    raw_exposures = [
        column
        for column in [
            "turn",
            "pctChg",
            "log_amount_mean_20d_z",
            "momentum_20d_z",
            "neg_volatility_20d_z",
            "peTTM_lag1",
            "pbMRQ_lag1",
            "psTTM_lag1",
            "pcfNcfTTM_lag1",
        ]
        if column in output.columns
    ]
    exposure_z_columns: list[str] = []
    for column in raw_exposures:
        z_col = f"{column}_xsec_z"
        output[z_col] = output.groupby("date", sort=False)[column].transform(_zscore)
        exposure_z_columns.append(z_col)
    return output.sort_values(["date", "code"]).reset_index(drop=True), exposure_z_columns


def daily_signal_exposure_correlation(
    frame: pd.DataFrame,
    signal_columns: Sequence[str],
    exposure_columns: Sequence[str],
    *,
    min_pairs: int = 50,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame(columns=["date", "signal", "exposure", "pairs", "pearson_corr"])
    for date, group in frame.groupby("date", sort=True):
        for signal in signal_columns:
            for exposure in exposure_columns:
                if signal not in group.columns or exposure not in group.columns:
                    continue
                values = group[[signal, exposure]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if len(values) < min_pairs:
                    continue
                corr = values[signal].corr(values[exposure])
                rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "signal": signal,
                        "exposure": exposure,
                        "pairs": int(len(values)),
                        "pearson_corr": float(corr) if pd.notna(corr) else np.nan,
                    }
                )
    output = pd.DataFrame(rows)
    if not output.empty:
        output["date"] = pd.to_datetime(output["date"]).dt.strftime("%Y-%m-%d")
    return output


def summarize_signal_exposure_correlation(daily: pd.DataFrame, *, eval_year: int | None = None) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["eval_year", "signal", "exposure", "date_count", "mean_corr", "mean_abs_corr", "p95_abs_corr", "max_abs_corr"])
    work = daily.copy()
    work["abs_corr"] = pd.to_numeric(work["pearson_corr"], errors="coerce").abs()
    rows: list[dict[str, Any]] = []
    for (signal, exposure), group in work.groupby(["signal", "exposure"], sort=True):
        rows.append(
            {
                "eval_year": int(eval_year) if eval_year is not None else np.nan,
                "signal": signal,
                "exposure": exposure,
                "date_count": int(group["date"].nunique()),
                "mean_corr": float(pd.to_numeric(group["pearson_corr"], errors="coerce").mean()),
                "mean_abs_corr": float(group["abs_corr"].mean()),
                "p95_abs_corr": float(group["abs_corr"].quantile(0.95)),
                "max_abs_corr": float(group["abs_corr"].max()),
            }
        )
    return pd.DataFrame(rows)


def summarize_basket_metric_exposure(basket: pd.DataFrame) -> pd.DataFrame:
    if basket.empty:
        return pd.DataFrame(columns=["eval_year", "signal", "factor", "period_type", "period_count", "mean_active_exposure", "mean_abs_active_exposure", "max_abs_active_exposure"])
    work = basket.copy()
    work["abs_active_exposure"] = pd.to_numeric(work["active_exposure"], errors="coerce").abs()
    rows: list[dict[str, Any]] = []
    for (eval_year, signal, factor, period_type), group in work.groupby(["eval_year", "signal", "factor", "period_type"], sort=True):
        rows.append(
            {
                "eval_year": int(eval_year),
                "signal": signal,
                "factor": factor,
                "period_type": period_type,
                "period_count": int(group["period"].nunique()),
                "mean_active_exposure": float(pd.to_numeric(group["active_exposure"], errors="coerce").mean()),
                "mean_abs_active_exposure": float(group["abs_active_exposure"].mean()),
                "max_abs_active_exposure": float(group["abs_active_exposure"].max()),
            }
        )
    return pd.DataFrame(rows)


def summarize_run(
    *,
    run_id: str,
    manifest: Mapping[str, Any],
    years: Sequence[int],
    final_end_date: str | None,
    yearly_correlation: pd.DataFrame,
    basket_summary: pd.DataFrame,
    run_dir: Path,
) -> dict[str, Any]:
    max_mean_abs_corr = _max_or_nan(yearly_correlation, "mean_abs_corr")
    max_basket_abs_active = _max_or_nan(basket_summary, "mean_abs_active_exposure")
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "snapshot_id": manifest.get("snapshot_id", ""),
        "years": [int(year) for year in years],
        "final_end_date": final_end_date,
        "max_mean_abs_signal_metric_corr": max_mean_abs_corr,
        "max_mean_abs_basket_metric_active_exposure": max_basket_abs_active,
        "candidate_count": 0,
        "status": "metrics_exposure_diagnostics",
        "assessment": "diagnostic only; no strategy candidate is promoted",
        "output_dir": str(run_dir),
    }


def render_metrics_exposure_markdown(
    result: Mapping[str, Any],
    yearly_correlation: pd.DataFrame,
    basket_summary: pd.DataFrame,
) -> str:
    lines = [
        "# v2.1 Metrics Exposure Diagnostics",
        "",
        f"- run_id: `{result.get('run_id', '')}`",
        f"- snapshot_id: `{result.get('snapshot_id', '')}`",
        f"- years: `{result.get('years', [])}`",
        f"- status: `{result.get('status', '')}`",
        f"- max_mean_abs_signal_metric_corr: `{_fmt(result.get('max_mean_abs_signal_metric_corr'))}`",
        f"- max_mean_abs_basket_metric_active_exposure: `{_fmt(result.get('max_mean_abs_basket_metric_active_exposure'))}`",
        f"- candidate_count: `{result.get('candidate_count', 0)}`",
        "",
        "## Strongest Signal-Metric Correlations",
        "",
        _markdown_table(_top_abs(yearly_correlation, "mean_abs_corr", top=30)),
        "",
        "## Strongest Basket Metric Active Exposures",
        "",
        _markdown_table(_top_abs(basket_summary, "mean_abs_active_exposure", top=30)),
        "",
        "## Interpretation",
        "",
        "This diagnostic uses `turn`, `pctChg`, liquidity/volatility proxies, and one-day-lagged valuation fields. "
        "Lagged valuation fields are exposure diagnostics only because valuation publication timing is not proven. "
        "Large signal-metric correlations or basket active exposures identify promotion gates, not strategy candidates.",
        "",
        f"Artifacts: `{result.get('output_dir')}`",
        "",
    ]
    return "\n".join(lines)


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    scale = values.std(ddof=0)
    if pd.isna(scale) or scale == 0:
        return pd.Series(np.where(values.notna(), 0.0, np.nan), index=series.index)
    return (values - values.mean()) / scale


def _max_or_nan(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.max()) if values.notna().any() else np.nan


def _top_abs(frame: pd.DataFrame, column: str, *, top: int) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return frame
    return frame.sort_values(column, ascending=False).head(top).reset_index(drop=True)


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


def _parse_years(spec: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in spec.split(",") if value.strip())
    if not values:
        raise ValueError("expected at least one year")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--years", default=",".join(str(year) for year in DEFAULT_YEARS))
    parser.add_argument("--final-end-date", default=DEFAULT_FINAL_END_DATE)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--frequency", default=DEFAULT_REBALANCE_FREQUENCY)
    parser.add_argument("--buffer-multiplier", type=float, default=DEFAULT_BUFFER_MULTIPLIER)
    parser.add_argument("--max-factor-corr", type=float, default=DEFAULT_MAX_FACTOR_CORR)
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW)
    parser.add_argument("--rolling-min-periods", type=int, default=DEFAULT_ROLLING_MIN_PERIODS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_v2_metrics_exposure_diagnostics(
        root=args.root,
        years=_parse_years(args.years),
        final_end_date=args.final_end_date,
        horizon=args.horizon,
        top_n=args.top_n,
        rebalance_frequency=args.frequency,
        buffer_multiplier=args.buffer_multiplier,
        max_factor_corr=args.max_factor_corr,
        rolling_window=args.rolling_window,
        rolling_min_periods=args.rolling_min_periods,
        output_dir=args.output_dir,
        write_research_log=bool(args.write_research_log),
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
