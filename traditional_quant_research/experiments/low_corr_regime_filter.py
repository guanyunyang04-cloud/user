"""Run regime-filter experiments for the low-correlation rank signal."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.backtest_protocol import select_rebalance_dates
from traditional_quant_research.dataset_v2 import load_pit_manifest, load_quality_report, load_tradeable_panel
from traditional_quant_research.diagnostics import summarize_factor_ic
from traditional_quant_research.experiments.low_corr_exposure_grid import (
    LOW_CORR_SIGNAL,
    add_baseline_delta_columns,
)
from traditional_quant_research.experiments.multifactor_baseline import (
    apply_selection_filters,
    apply_selection_filters_to_signals,
    best_backtest_configs,
    directions_from_ic,
    filter_panel_dates,
    parse_selection_filters,
    run_horizon_backtest_tables,
    selection_filter_mask,
)
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


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/low_corr_regime_filter")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-02_low_corr_regime_filter.md")
DEFAULT_HORIZON = 5
DEFAULT_TOP_N_VALUES = (100,)
DEFAULT_FEE_BPS_VALUES = (0.0, 30.0)
DEFAULT_REBALANCE_FREQUENCIES = ("weekly",)
DEFAULT_BUFFER_MULTIPLIERS = (2.0,)
DEFAULT_MAX_FACTOR_CORR = 0.75
DEFAULT_REGIME_FILTER_SPECS = (
    "market_ret_20d_mean>=0",
    "breadth_20d_positive_rate>=0.5",
    "market_ret_20d_mean>=0,breadth_20d_positive_rate>=0.5",
    "market_ret_5d_mean>=0,breadth_5d_positive_rate>=0.5",
    "market_ret_20d_mean>=-0.02,breadth_20d_positive_rate>=0.45",
)
LABEL_MODES = ("raw", "xsec-excess")


def build_market_regime_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize same-day market state from the tradeable PIT panel."""

    required = {"date", "code", "ret_1d", "ret_5d", "ret_20d", "volatility_20d", "amplitude_20d"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"frame missing required columns for market regime: {missing}")
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "security_count",
                "market_ret_1d_mean",
                "market_ret_5d_mean",
                "market_ret_20d_mean",
                "breadth_1d_positive_rate",
                "breadth_5d_positive_rate",
                "breadth_20d_positive_rate",
                "market_volatility_20d_mean",
                "market_amplitude_20d_mean",
            ]
        )

    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"])
    for column in ["ret_1d", "ret_5d", "ret_20d", "volatility_20d", "amplitude_20d"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    rows: list[dict[str, Any]] = []
    for date, group in work.groupby("date", sort=True):
        rows.append(
            {
                "date": pd.Timestamp(date),
                "security_count": int(group["code"].nunique()),
                "market_ret_1d_mean": _nanmean(group["ret_1d"]),
                "market_ret_5d_mean": _nanmean(group["ret_5d"]),
                "market_ret_20d_mean": _nanmean(group["ret_20d"]),
                "breadth_1d_positive_rate": _positive_rate(group["ret_1d"]),
                "breadth_5d_positive_rate": _positive_rate(group["ret_5d"]),
                "breadth_20d_positive_rate": _positive_rate(group["ret_20d"]),
                "market_volatility_20d_mean": _nanmean(group["volatility_20d"]),
                "market_amplitude_20d_mean": _nanmean(group["amplitude_20d"]),
            }
        )
    return pd.DataFrame(rows)


def parse_regime_filter_specs(specs: str, *, include_baseline: bool = True) -> list[dict[str, Any]]:
    """Parse semicolon-separated regime filter specs."""

    configs: list[dict[str, Any]] = []
    if include_baseline:
        configs.append({"config_id": "baseline_no_regime", "regime_filter_spec": "", "regime_filters": []})
    for index, raw_spec in enumerate(specs.split(";"), start=1):
        regime_filter_spec = raw_spec.strip()
        if not regime_filter_spec:
            continue
        configs.append(
            {
                "config_id": f"regime_{index:03d}",
                "regime_filter_spec": regime_filter_spec,
                "regime_filters": parse_selection_filters(regime_filter_spec),
            }
        )
    if not configs:
        raise ValueError("regime filter specs produced no configurations")
    return configs


def default_regime_filter_configs(*, include_baseline: bool = True) -> list[dict[str, Any]]:
    return parse_regime_filter_specs(";".join(DEFAULT_REGIME_FILTER_SPECS), include_baseline=include_baseline)


def apply_fixed_rebalance_regime_to_signals(
    frame: pd.DataFrame,
    regime_frame: pd.DataFrame,
    signal_columns: Sequence[str],
    rules: Sequence[Mapping[str, Any]] | None,
    *,
    rebalance_frequency: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep full price path but only allow signals on scheduled, regime-approved dates."""

    if frame.empty:
        return frame.copy(), _empty_regime_report(rules, rebalance_frequency=rebalance_frequency)
    missing_signals = [signal for signal in signal_columns if signal not in frame.columns]
    if missing_signals:
        raise ValueError(f"signal columns not found: {missing_signals}")
    if "date" not in frame.columns:
        raise ValueError("frame missing required column: date")
    if "date" not in regime_frame.columns:
        raise ValueError("regime_frame missing required column: date")

    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"])
    regime = regime_frame.copy()
    regime["date"] = pd.to_datetime(regime["date"])
    scheduled_dates = pd.DatetimeIndex(select_rebalance_dates(output["date"], rebalance_frequency))
    scheduled_regime = pd.DataFrame({"date": scheduled_dates}).merge(regime, on="date", how="left")

    normalized_rules: list[dict[str, Any]] = []
    if rules:
        pass_mask, normalized_rules = selection_filter_mask(scheduled_regime, rules)
        allowed_dates = set(pd.to_datetime(scheduled_regime.loc[pass_mask, "date"]))
    else:
        allowed_dates = set(pd.to_datetime(scheduled_dates))
    scheduled_date_set = set(pd.to_datetime(scheduled_dates))
    allowed_schedule_mask = output["date"].isin(allowed_dates)
    scheduled_mask = output["date"].isin(scheduled_date_set)
    before_signal_rows = int(output.loc[scheduled_mask, list(signal_columns)].notna().any(axis=1).sum())

    for signal in signal_columns:
        output.loc[~allowed_schedule_mask, signal] = np.nan
    after_signal_rows = int(output.loc[allowed_schedule_mask, list(signal_columns)].notna().any(axis=1).sum())

    blocked_dates = [pd.Timestamp(date).strftime("%Y-%m-%d") for date in scheduled_dates if pd.Timestamp(date) not in allowed_dates]
    allowed_date_strings = [pd.Timestamp(date).strftime("%Y-%m-%d") for date in scheduled_dates if pd.Timestamp(date) in allowed_dates]
    report = {
        "enabled": bool(rules),
        "rebalance_frequency": rebalance_frequency,
        "rules": normalized_rules,
        "scheduled_rebalance_count": int(len(scheduled_dates)),
        "allowed_rebalance_count": int(len(allowed_dates)),
        "blocked_rebalance_count": int(len(scheduled_dates) - len(allowed_dates)),
        "allowed_rate": float(len(allowed_dates) / len(scheduled_dates)) if len(scheduled_dates) else np.nan,
        "scheduled_signal_rows_before": before_signal_rows,
        "scheduled_signal_rows_after": after_signal_rows,
        "allowed_dates": allowed_date_strings,
        "blocked_dates": blocked_dates,
    }
    return output, report


def run_low_corr_regime_filter(
    *,
    root: str | None = None,
    history_start_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    fit_start_date: str | None = None,
    fit_end_date: str | None = None,
    horizon: int = DEFAULT_HORIZON,
    label_mode: str = "raw",
    max_factor_corr: float = DEFAULT_MAX_FACTOR_CORR,
    selection_filters: Sequence[Mapping[str, Any]] | None = None,
    regime_filter_configs: Sequence[Mapping[str, Any]] | None = None,
    include_baseline: bool = True,
    top_n_values: Sequence[int] = DEFAULT_TOP_N_VALUES,
    fee_bps_values: Sequence[float] = DEFAULT_FEE_BPS_VALUES,
    rebalance_frequencies: Sequence[str] = DEFAULT_REBALANCE_FREQUENCIES,
    buffer_multipliers: Sequence[float] = DEFAULT_BUFFER_MULTIPLIERS,
    execution_constraints: bool = True,
    limit_threshold: float = 0.095,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    """Run low-corr regime filters under fixed rebalance-date semantics."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if label_mode not in LABEL_MODES:
        raise ValueError(f"unsupported label_mode: {label_mode}")
    if max_factor_corr < 0 or max_factor_corr > 1:
        raise ValueError("max_factor_corr must be between 0 and 1")
    if any(top_n <= 0 for top_n in top_n_values):
        raise ValueError("top_n_values must be positive")
    if any(value < 1.0 for value in buffer_multipliers):
        raise ValueError("buffer_multipliers must be at least 1.0")
    if limit_threshold <= 0:
        raise ValueError("limit_threshold must be positive")

    manifest = load_pit_manifest(root)
    quality = load_quality_report(root)
    dataset = manifest.get("dataset", {})
    effective_start = start_date or dataset.get("date_min")
    effective_end = end_date or dataset.get("date_max")
    effective_history_start = history_start_date or fit_start_date or effective_start
    effective_fit_start = fit_start_date or effective_history_start
    effective_fit_end = fit_end_date or effective_start
    raw_label_col = f"fwd_ret_{horizon}d"
    label_col = raw_label_col if label_mode == "raw" else f"xsec_excess_ret_{horizon}d"
    run_id = f"low_corr_regime_filter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    configs = list(regime_filter_configs) if regime_filter_configs is not None else default_regime_filter_configs(include_baseline=include_baseline)
    if not configs:
        raise ValueError("regime_filter_configs must not be empty")

    raw_panel = load_tradeable_panel(root, start_date=effective_history_start, end_date=effective_end)
    factor_panel = build_factor_label_panel(raw_panel, horizons=tuple(sorted({1, 5, 20, horizon})))
    factor_panel = add_cross_sectional_excess_return_labels(factor_panel, horizons=(horizon,))
    raw_factor_columns = default_factor_columns()
    factor_panel = add_cross_sectional_zscores(factor_panel, raw_factor_columns)
    signal_columns = [f"{column}_z" for column in raw_factor_columns]
    fit_panel = filter_panel_dates(factor_panel, start_date=effective_fit_start, end_date=effective_fit_end)
    evaluation_panel = filter_panel_dates(factor_panel, start_date=effective_start, end_date=effective_end)

    if label_col not in fit_panel.columns or label_col not in evaluation_panel.columns:
        raise ValueError(f"label not generated: {label_col}")
    if fit_panel.empty:
        raise ValueError("fit panel is empty")
    if evaluation_panel.empty:
        raise ValueError("evaluation panel is empty")

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
    evaluation_panel = filter_panel_dates(factor_panel, start_date=effective_start, end_date=effective_end)
    market_regime = build_market_regime_frame(evaluation_panel)
    coverage = factor_coverage(evaluation_panel, signal_columns)
    selection_filtered_panel, selection_filter_report = apply_selection_filters(evaluation_panel, selection_filters)
    signal_panel = apply_selection_filters_to_signals(evaluation_panel, [LOW_CORR_SIGNAL], selection_filters)

    grid_summary_frames: list[pd.DataFrame] = []
    yearly_frames: list[pd.DataFrame] = []
    period_frames: list[pd.DataFrame] = []
    exposure_frames: list[pd.DataFrame] = []
    regime_reports: list[dict[str, Any]] = []

    for config in configs:
        config_id = str(config.get("config_id") or f"regime_{len(regime_reports) + 1:03d}")
        regime_filter_spec = str(config.get("regime_filter_spec") or "")
        rules = list(config.get("regime_filters") or parse_selection_filters(regime_filter_spec))
        for frequency in rebalance_frequencies:
            regime_panel, regime_report = apply_fixed_rebalance_regime_to_signals(
                signal_panel,
                market_regime,
                [LOW_CORR_SIGNAL],
                rules,
                rebalance_frequency=frequency,
            )
            regime_reports.append(
                {
                    "config_id": config_id,
                    "regime_filter_spec": regime_filter_spec,
                    **{key: _table_scalar(value) for key, value in regime_report.items()},
                }
            )
            horizon_summary, yearly_summary, period_summary, exposure_summary = run_horizon_backtest_tables(
                regime_panel,
                [LOW_CORR_SIGNAL],
                horizon=horizon,
                top_n_values=top_n_values,
                fee_bps_values=fee_bps_values,
                rebalance_frequencies=(frequency,),
                buffer_multipliers=buffer_multipliers,
                exposure_columns=low_corr_factor_columns,
                execution_constraints=execution_constraints,
                limit_threshold=limit_threshold,
            )
            for frame in (horizon_summary, yearly_summary, period_summary, exposure_summary):
                _insert_config_columns(frame, config_id=config_id, regime_filter_spec=regime_filter_spec)
            if not horizon_summary.empty:
                for key, value in regime_report.items():
                    horizon_summary[f"regime_{key}"] = _table_scalar(value)
                for key, value in selection_filter_report.items():
                    horizon_summary[f"selection_{key}"] = _table_scalar(value)
                grid_summary_frames.append(horizon_summary)
            if not yearly_summary.empty:
                yearly_frames.append(yearly_summary)
            if not period_summary.empty:
                period_frames.append(period_summary)
            if not exposure_summary.empty:
                exposure_frames.append(exposure_summary)

    grid_summary = pd.concat(grid_summary_frames, ignore_index=True) if grid_summary_frames else pd.DataFrame()
    grid_summary = add_baseline_delta_columns(_rename_baseline_for_delta(grid_summary))
    grid_summary["config_id"] = grid_summary["config_id"].replace({"baseline_no_filter": "baseline_no_regime"})
    yearly_summary = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    period_summary = pd.concat(period_frames, ignore_index=True) if period_frames else pd.DataFrame()
    exposure_summary = pd.concat(exposure_frames, ignore_index=True) if exposure_frames else pd.DataFrame()
    regime_report_frame = pd.DataFrame(regime_reports)

    best_configs = best_backtest_configs(grid_summary)
    result = {
        "run_id": run_id,
        "snapshot_id": manifest.get("snapshot_id"),
        "snapshot_path": manifest.get("snapshot_path"),
        "history_start_date": effective_history_start,
        "fit_start_date": effective_fit_start,
        "fit_end_date": effective_fit_end,
        "start_date": effective_start,
        "end_date": effective_end,
        "horizon": int(horizon),
        "label_mode": label_mode,
        "label": label_col,
        "max_factor_corr": float(max_factor_corr),
        "low_corr_signal": LOW_CORR_SIGNAL,
        "low_corr_factor_columns": low_corr_factor_columns,
        "factor_directions": factor_directions,
        "config_count": int(len(configs)),
        "selection_filters": [dict(rule) for rule in selection_filters] if selection_filters else [],
        "selection_filter_report": selection_filter_report,
        "top_n_values": [int(value) for value in top_n_values],
        "fee_bps_values": [float(value) for value in fee_bps_values],
        "rebalance_frequencies": list(rebalance_frequencies),
        "buffer_multipliers": [float(value) for value in buffer_multipliers],
        "execution_constraints": bool(execution_constraints),
        "limit_threshold": float(limit_threshold),
        "history_panel": panel_summary(factor_panel),
        "fit_panel": panel_summary(fit_panel),
        "panel": panel_summary(evaluation_panel),
        "selection_panel": panel_summary(selection_filtered_panel),
        "market_regime": {
            "rows": int(len(market_regime)),
            "date_min": str(market_regime["date"].min().date()) if not market_regime.empty else None,
            "date_max": str(market_regime["date"].max().date()) if not market_regime.empty else None,
        },
        "quality": {
            "failure_count": quality.get("failure_count"),
            "missing_bar_rows": quality.get("missing_bar_rows"),
            "st_rows": quality.get("st_rows"),
            "suspended_like_rows": quality.get("suspended_like_rows"),
        },
        "best_configs": best_configs,
        "candidate_count": 0,
        "assessment": "backtest_only regime evidence; no strategy candidate is promoted by this experiment alone",
        "output_dir": str(run_dir),
    }

    coverage.to_csv(run_dir / "factor_coverage.csv", index=False, encoding="utf-8-sig")
    single_factor_ic.to_csv(run_dir / "fit_single_factor_ic.csv", index=False, encoding="utf-8-sig")
    factor_correlation.to_csv(run_dir / "fit_factor_correlation.csv", encoding="utf-8-sig")
    market_regime.to_csv(run_dir / "market_regime.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"factor": low_corr_factor_columns}).to_csv(run_dir / "low_corr_factors.csv", index=False, encoding="utf-8-sig")
    regime_report_frame.to_csv(run_dir / "regime_filter_reports.csv", index=False, encoding="utf-8-sig")
    grid_summary.to_csv(run_dir / "regime_summary.csv", index=False, encoding="utf-8-sig")
    yearly_summary.to_csv(run_dir / "regime_yearly_summary.csv", index=False, encoding="utf-8-sig")
    period_summary.to_csv(run_dir / "regime_period_summary.csv", index=False, encoding="utf-8-sig")
    exposure_summary.to_csv(run_dir / "regime_basket_exposure.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_regime_filter_markdown(result, grid_summary, regime_report_frame, period_summary, exposure_summary)
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        research_log_path.write_text(markdown, encoding="utf-8")
    return result


def render_regime_filter_markdown(
    result: Mapping[str, Any],
    grid_summary: pd.DataFrame,
    regime_reports: pd.DataFrame,
    period_summary: pd.DataFrame,
    exposure_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Low-Corr Regime Filter",
        "",
        "- Hypothesis: low-corr exposure-controlled baskets may improve if weak market regimes are skipped at scheduled rebalance dates.",
        f"- Data Scope: snapshot `{result.get('snapshot_id')}`, `{result.get('start_date')}` to `{result.get('end_date')}`.",
        f"- Fit Scope: `{result.get('fit_start_date')}` to `{result.get('fit_end_date')}`; label `{result.get('label')}`.",
        f"- Signal: `{result.get('low_corr_signal')}` from factors `{result.get('low_corr_factor_columns')}`.",
        f"- Stock Selection Filters: `{result.get('selection_filters')}`; report `{result.get('selection_filter_report')}`.",
        f"- Grid: `{result.get('config_count')}` regime configs; Top-N `{result.get('top_n_values')}`, fees `{result.get('fee_bps_values')}`, frequencies `{result.get('rebalance_frequencies')}`, buffers `{result.get('buffer_multipliers')}`.",
        "- Regime Semantics: rebalance dates are fixed from the full evaluation calendar; disallowed regimes skip the scheduled rebalance instead of shifting it earlier within the same week or month.",
        f"- Execution Constraints: `{result.get('execution_constraints')}` with limit threshold `{result.get('limit_threshold')}`.",
        "- Assessment: this remains `backtest_only` evidence; candidate count is `0` until annual stability, cost robustness, execution feasibility, and exposure diagnostics all pass.",
        f"- Artifacts: `{result.get('output_dir')}`",
        "",
        "## Best Regime Rows",
        "",
        _markdown_table(grid_summary.sort_values(["sharpe", "annualized_return"], ascending=[False, False]) if not grid_summary.empty else grid_summary),
        "",
        "## Regime Reports",
        "",
        _markdown_table(regime_reports),
        "",
        "## Monthly/Quarterly Slices",
        "",
        _markdown_table(period_summary),
        "",
        "## Basket Exposure",
        "",
        _markdown_table(exposure_summary),
        "",
    ]
    return "\n".join(lines)


def _rename_baseline_for_delta(summary: pd.DataFrame) -> pd.DataFrame:
    output = summary.copy()
    if "config_id" in output.columns:
        output["config_id"] = output["config_id"].replace({"baseline_no_regime": "baseline_no_filter"})
    return output


def _insert_config_columns(frame: pd.DataFrame, *, config_id: str, regime_filter_spec: str) -> None:
    if frame.empty:
        return
    frame.insert(0, "regime_filter_spec", regime_filter_spec)
    frame.insert(0, "config_id", config_id)


def _empty_regime_report(
    rules: Sequence[Mapping[str, Any]] | None,
    *,
    rebalance_frequency: str,
) -> dict[str, Any]:
    return {
        "enabled": bool(rules),
        "rebalance_frequency": rebalance_frequency,
        "rules": [],
        "scheduled_rebalance_count": 0,
        "allowed_rebalance_count": 0,
        "blocked_rebalance_count": 0,
        "allowed_rate": np.nan,
        "scheduled_signal_rows_before": 0,
        "scheduled_signal_rows_after": 0,
        "allowed_dates": [],
        "blocked_dates": [],
    }


def _nanmean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.mean()) if not clean.empty else np.nan


def _positive_rate(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float((clean > 0).mean()) if not clean.empty else np.nan


def _table_scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_json_ready(value), ensure_ascii=False)
    return value


def _parse_float_tuple(spec: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in spec.split(",") if value.strip())
    if not values:
        raise ValueError("expected at least one numeric value")
    return values


def _parse_int_tuple(spec: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in spec.split(",") if value.strip())
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def _parse_str_tuple(spec: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in spec.split(",") if value.strip())
    if not values:
        raise ValueError("expected at least one value")
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
    parser.add_argument("--root", default=None, help="Optional v2 snapshot root.")
    parser.add_argument("--history-start-date", default=None, help="Earlier panel start used to fit low-corr factors.")
    parser.add_argument("--fit-start-date", default=None, help="Start date for factor direction and low-corr selection.")
    parser.add_argument("--fit-end-date", default=None, help="End date for factor direction and low-corr selection.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="raw")
    parser.add_argument("--max-factor-corr", type=float, default=DEFAULT_MAX_FACTOR_CORR)
    parser.add_argument("--selection-filters", default="", help="Stock-level filters applied before regime filters.")
    parser.add_argument("--regime-filter-specs", default="", help="Semicolon-separated regime filters. Empty uses the default regime grid.")
    parser.add_argument("--include-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-n", default=",".join(str(value) for value in DEFAULT_TOP_N_VALUES))
    parser.add_argument("--fee-bps", default=",".join(str(value) for value in DEFAULT_FEE_BPS_VALUES))
    parser.add_argument("--frequencies", default=",".join(DEFAULT_REBALANCE_FREQUENCIES))
    parser.add_argument("--buffer-multipliers", default=",".join(str(value) for value in DEFAULT_BUFFER_MULTIPLIERS))
    parser.add_argument("--execution-constraints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-threshold", type=float, default=0.095)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    regime_configs = (
        parse_regime_filter_specs(args.regime_filter_specs, include_baseline=args.include_baseline)
        if args.regime_filter_specs.strip()
        else None
    )
    result = run_low_corr_regime_filter(
        root=args.root,
        history_start_date=args.history_start_date,
        fit_start_date=args.fit_start_date,
        fit_end_date=args.fit_end_date,
        start_date=args.start_date,
        end_date=args.end_date,
        horizon=args.horizon,
        label_mode=args.label_mode,
        max_factor_corr=args.max_factor_corr,
        selection_filters=parse_selection_filters(args.selection_filters),
        regime_filter_configs=regime_configs,
        include_baseline=args.include_baseline,
        top_n_values=_parse_int_tuple(args.top_n),
        fee_bps_values=_parse_float_tuple(args.fee_bps),
        rebalance_frequencies=_parse_str_tuple(args.frequencies),
        buffer_multipliers=_parse_float_tuple(args.buffer_multipliers),
        execution_constraints=args.execution_constraints,
        limit_threshold=args.limit_threshold,
        output_dir=args.output_dir,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
