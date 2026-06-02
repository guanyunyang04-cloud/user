"""Test dynamic regime-based capital scaling for the low-corr signal."""

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
from traditional_quant_research.experiments.low_corr_exposure_grid import LOW_CORR_SIGNAL, add_baseline_delta_columns
from traditional_quant_research.experiments.low_corr_regime_filter import build_market_regime_frame
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


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/low_corr_regime_capital_scaling")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-02_low_corr_regime_capital_scaling.md")
DEFAULT_HORIZON = 5
DEFAULT_TOP_N_VALUES = (100,)
DEFAULT_FEE_BPS_VALUES = (0.0, 30.0)
DEFAULT_REBALANCE_FREQUENCIES = ("weekly",)
DEFAULT_BUFFER_MULTIPLIERS = (2.0,)
DEFAULT_MAX_FACTOR_CORR = 0.75
DEFAULT_SELECTION_FILTERS = "log_amount_mean_20d_z>=-0.8,momentum_20d_z>=-0.8"
DEFAULT_CAPITAL_PLAN_SPECS = (
    "breadth_soft=breadth_20d_positive_rate>=0.50@1.0|breadth_20d_positive_rate>=0.45@0.6|default@0.3;"
    "breadth_floor60=breadth_20d_positive_rate>=0.50@1.0|breadth_20d_positive_rate>=0.45@0.8|default@0.6;"
    "ret_breadth_soft=market_ret_20d_mean>=-0.02,breadth_20d_positive_rate>=0.45@1.0|breadth_20d_positive_rate>=0.40@0.6|default@0.3"
)
LABEL_MODES = ("raw", "xsec-excess")
CAPITAL_COL = "capital_scale"


def parse_capital_plan_specs(specs: str, *, include_baseline: bool = True) -> list[dict[str, Any]]:
    """Parse semicolon-separated capital scale plans.

    Plan format:
    ``name=condition@scale|condition@scale|default@scale``.
    Conditions use the same syntax as selection filters.
    """

    plans: list[dict[str, Any]] = []
    if include_baseline:
        plans.append({"config_id": "baseline_full_capital", "plan_spec": "default@1.0", "levels": [{"rules": [], "scale": 1.0, "label": "default"}]})
    for raw_plan in specs.split(";"):
        raw_plan = raw_plan.strip()
        if not raw_plan:
            continue
        if "=" not in raw_plan:
            raise ValueError(f"invalid capital plan: {raw_plan}")
        config_id, raw_levels = raw_plan.split("=", 1)
        config_id = config_id.strip()
        if not config_id:
            raise ValueError(f"invalid capital plan id: {raw_plan}")
        levels = []
        for raw_level in raw_levels.split("|"):
            raw_level = raw_level.strip()
            if not raw_level:
                continue
            if "@" not in raw_level:
                raise ValueError(f"invalid capital plan level: {raw_level}")
            raw_rule, raw_scale = raw_level.rsplit("@", 1)
            scale = float(raw_scale)
            if scale < 0 or scale > 1:
                raise ValueError(f"capital scale must be between 0 and 1: {raw_level}")
            rule_text = raw_rule.strip()
            rules = [] if rule_text == "default" else parse_selection_filters(rule_text)
            levels.append({"rules": rules, "scale": scale, "label": rule_text})
        if not levels:
            raise ValueError(f"capital plan has no levels: {raw_plan}")
        plans.append({"config_id": config_id, "plan_spec": raw_levels, "levels": levels})
    if not plans:
        raise ValueError("capital plan specs produced no plans")
    return plans


def capital_scale_by_date(
    regime_frame: pd.DataFrame,
    levels: Sequence[Mapping[str, Any]],
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    """Apply ordered capital plan levels to a market regime frame."""

    if regime_frame.empty:
        return pd.DataFrame(columns=[date_col, CAPITAL_COL, "capital_rule"])
    if date_col not in regime_frame.columns:
        raise ValueError(f"regime_frame missing required column: {date_col}")
    if not levels:
        raise ValueError("levels must not be empty")

    output = regime_frame.copy()
    output[date_col] = pd.to_datetime(output[date_col])
    output[CAPITAL_COL] = np.nan
    output["capital_rule"] = ""
    remaining = pd.Series(True, index=output.index)
    for level in levels:
        scale = float(level.get("scale"))
        rules = list(level.get("rules") or [])
        label = str(level.get("label") or "default")
        if rules:
            level_mask, _ = selection_filter_mask(output, rules)
        else:
            level_mask = pd.Series(True, index=output.index)
        apply_mask = remaining & level_mask
        output.loc[apply_mask, CAPITAL_COL] = scale
        output.loc[apply_mask, "capital_rule"] = label
        remaining &= ~apply_mask
    if output[CAPITAL_COL].isna().any():
        raise ValueError("capital plan did not assign all dates; add a default level")
    return output.loc[:, [date_col, CAPITAL_COL, "capital_rule"]]


def attach_capital_scale(frame: pd.DataFrame, capital_by_date: pd.DataFrame) -> pd.DataFrame:
    """Attach same-date capital scale to every security row."""

    if "date" not in frame.columns or "date" not in capital_by_date.columns:
        raise ValueError("frame and capital_by_date must both contain date")
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"])
    capital = capital_by_date.copy()
    capital["date"] = pd.to_datetime(capital["date"])
    output = output.merge(capital, on="date", how="left")
    if output[CAPITAL_COL].isna().any():
        raise ValueError("capital scale missing for some frame dates")
    return output


def capital_plan_report(
    frame: pd.DataFrame,
    *,
    config_id: str,
    plan_spec: str,
    rebalance_frequency: str,
) -> dict[str, Any]:
    """Summarize capital scale usage on scheduled rebalance dates."""

    if frame.empty:
        return {
            "config_id": config_id,
            "plan_spec": plan_spec,
            "rebalance_frequency": rebalance_frequency,
            "scheduled_rebalance_count": 0,
            "mean_scheduled_capital_scale": np.nan,
            "min_scheduled_capital_scale": np.nan,
            "max_scheduled_capital_scale": np.nan,
            "full_capital_date_count": 0,
            "reduced_capital_date_count": 0,
        }
    dates = pd.DatetimeIndex(select_rebalance_dates(frame["date"], rebalance_frequency))
    schedule = frame.loc[pd.to_datetime(frame["date"]).isin(dates), ["date", CAPITAL_COL]].drop_duplicates("date")
    scales = pd.to_numeric(schedule[CAPITAL_COL], errors="coerce")
    return {
        "config_id": config_id,
        "plan_spec": plan_spec,
        "rebalance_frequency": rebalance_frequency,
        "scheduled_rebalance_count": int(len(schedule)),
        "mean_scheduled_capital_scale": float(scales.mean()) if not scales.empty else np.nan,
        "min_scheduled_capital_scale": float(scales.min()) if not scales.empty else np.nan,
        "max_scheduled_capital_scale": float(scales.max()) if not scales.empty else np.nan,
        "full_capital_date_count": int((scales == 1.0).sum()),
        "reduced_capital_date_count": int((scales < 1.0).sum()),
    }


def run_low_corr_regime_capital_scaling(
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
    capital_plan_configs: Sequence[Mapping[str, Any]] | None = None,
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
    """Run low-corr dynamic capital scaling on one evaluation window."""

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
    run_id = f"low_corr_regime_capital_scaling_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    plans = list(capital_plan_configs) if capital_plan_configs is not None else parse_capital_plan_specs(DEFAULT_CAPITAL_PLAN_SPECS, include_baseline=include_baseline)
    if not plans:
        raise ValueError("capital_plan_configs must not be empty")
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

    summary_frames: list[pd.DataFrame] = []
    yearly_frames: list[pd.DataFrame] = []
    period_frames: list[pd.DataFrame] = []
    exposure_frames: list[pd.DataFrame] = []
    report_rows: list[dict[str, Any]] = []

    for plan in plans:
        config_id = str(plan["config_id"])
        plan_spec = str(plan["plan_spec"])
        capital_by_date = capital_scale_by_date(market_regime, plan["levels"])
        capital_panel = attach_capital_scale(signal_panel, capital_by_date)
        for frequency in rebalance_frequencies:
            report_rows.append(capital_plan_report(capital_panel, config_id=config_id, plan_spec=plan_spec, rebalance_frequency=frequency))
            summary, yearly, period, exposure = run_horizon_backtest_tables(
                capital_panel,
                [LOW_CORR_SIGNAL],
                horizon=horizon,
                top_n_values=top_n_values,
                fee_bps_values=fee_bps_values,
                rebalance_frequencies=(frequency,),
                buffer_multipliers=buffer_multipliers,
                exposure_columns=low_corr_factor_columns,
                execution_constraints=execution_constraints,
                limit_threshold=limit_threshold,
                capital_col=CAPITAL_COL,
            )
            for frame in (summary, yearly, period, exposure):
                _insert_config_columns(frame, config_id=config_id, plan_spec=plan_spec)
            if not summary.empty:
                for key, value in selection_filter_report.items():
                    summary[f"selection_{key}"] = _table_scalar(value)
                summary_frames.append(summary)
            if not yearly.empty:
                yearly_frames.append(yearly)
            if not period.empty:
                period_frames.append(period)
            if not exposure.empty:
                exposure_frames.append(exposure)

    capital_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    capital_summary = _add_full_capital_delta_columns(capital_summary)
    yearly_summary = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    period_summary = pd.concat(period_frames, ignore_index=True) if period_frames else pd.DataFrame()
    exposure_summary = pd.concat(exposure_frames, ignore_index=True) if exposure_frames else pd.DataFrame()
    report_frame = pd.DataFrame(report_rows)

    result = {
        "run_id": run_id,
        "snapshot_id": manifest.get("snapshot_id"),
        "history_start_date": effective_history_start,
        "fit_start_date": effective_fit_start,
        "fit_end_date": effective_fit_end,
        "start_date": effective_start,
        "end_date": effective_end,
        "horizon": int(horizon),
        "label_mode": label_mode,
        "label": label_col,
        "low_corr_signal": LOW_CORR_SIGNAL,
        "low_corr_factor_columns": low_corr_factor_columns,
        "capital_plan_count": int(len(plans)),
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
        "quality": {
            "failure_count": quality.get("failure_count"),
            "missing_bar_rows": quality.get("missing_bar_rows"),
            "st_rows": quality.get("st_rows"),
            "suspended_like_rows": quality.get("suspended_like_rows"),
        },
        "best_configs": best_backtest_configs(capital_summary),
        "candidate_count": 0,
        "assessment": "backtest_only dynamic capital evidence; no strategy candidate is promoted by this experiment alone",
        "output_dir": str(run_dir),
    }
    coverage.to_csv(run_dir / "factor_coverage.csv", index=False, encoding="utf-8-sig")
    single_factor_ic.to_csv(run_dir / "fit_single_factor_ic.csv", index=False, encoding="utf-8-sig")
    factor_correlation.to_csv(run_dir / "fit_factor_correlation.csv", encoding="utf-8-sig")
    market_regime.to_csv(run_dir / "market_regime.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"factor": low_corr_factor_columns}).to_csv(run_dir / "low_corr_factors.csv", index=False, encoding="utf-8-sig")
    report_frame.to_csv(run_dir / "capital_plan_reports.csv", index=False, encoding="utf-8-sig")
    capital_summary.to_csv(run_dir / "capital_scaling_summary.csv", index=False, encoding="utf-8-sig")
    yearly_summary.to_csv(run_dir / "capital_scaling_yearly_summary.csv", index=False, encoding="utf-8-sig")
    period_summary.to_csv(run_dir / "capital_scaling_period_summary.csv", index=False, encoding="utf-8-sig")
    exposure_summary.to_csv(run_dir / "capital_scaling_basket_exposure.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_capital_scaling_markdown(result, capital_summary, report_frame, period_summary)
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        research_log_path.write_text(markdown, encoding="utf-8")
    return result


def render_capital_scaling_markdown(
    result: Mapping[str, Any],
    summary: pd.DataFrame,
    reports: pd.DataFrame,
    period_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Low-Corr Regime Capital Scaling",
        "",
        "- Hypothesis: reducing capital in weak regimes may retain strong-year upside better than all-or-nothing regime skips.",
        f"- Data Scope: `{result.get('start_date')}` to `{result.get('end_date')}`; fit `{result.get('fit_start_date')}` to `{result.get('fit_end_date')}`.",
        f"- Stock Selection Filters: `{result.get('selection_filters')}`.",
        f"- Capital Plans: `{result.get('capital_plan_count')}`.",
        "- Assessment: this remains `backtest_only` evidence; candidate count is `0` until multi-year cost robustness passes.",
        f"- Artifacts: `{result.get('output_dir')}`",
        "",
        "## Summary",
        "",
        _markdown_table(summary.sort_values(["sharpe", "annualized_return"], ascending=[False, False]) if not summary.empty else summary),
        "",
        "## Capital Plan Reports",
        "",
        _markdown_table(reports),
        "",
        "## Monthly/Quarterly Slices",
        "",
        _markdown_table(period_summary),
        "",
    ]
    return "\n".join(lines)


def _add_full_capital_delta_columns(summary: pd.DataFrame) -> pd.DataFrame:
    renamed = summary.copy()
    if "config_id" in renamed.columns:
        renamed["config_id"] = renamed["config_id"].replace({"baseline_full_capital": "baseline_no_filter"})
    output = add_baseline_delta_columns(renamed)
    if "config_id" in output.columns:
        output["config_id"] = output["config_id"].replace({"baseline_no_filter": "baseline_full_capital"})
    return output


def _insert_config_columns(frame: pd.DataFrame, *, config_id: str, plan_spec: str) -> None:
    if frame.empty:
        return
    frame.insert(0, "capital_plan_spec", plan_spec)
    frame.insert(0, "config_id", config_id)


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
    parser.add_argument("--root", default=None)
    parser.add_argument("--history-start-date", default=None)
    parser.add_argument("--fit-start-date", default=None)
    parser.add_argument("--fit-end-date", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="raw")
    parser.add_argument("--max-factor-corr", type=float, default=DEFAULT_MAX_FACTOR_CORR)
    parser.add_argument("--selection-filters", default=DEFAULT_SELECTION_FILTERS)
    parser.add_argument("--capital-plans", default=DEFAULT_CAPITAL_PLAN_SPECS)
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
    result = run_low_corr_regime_capital_scaling(
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
        capital_plan_configs=parse_capital_plan_specs(args.capital_plans, include_baseline=args.include_baseline),
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
