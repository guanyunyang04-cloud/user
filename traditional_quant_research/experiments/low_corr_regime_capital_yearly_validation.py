"""Validate low-corr regime capital scaling across multiple evaluation years."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.experiments.low_corr_regime_capital_scaling import (
    DEFAULT_BUFFER_MULTIPLIERS,
    DEFAULT_CAPITAL_PLAN_SPECS,
    DEFAULT_FEE_BPS_VALUES,
    DEFAULT_HORIZON,
    DEFAULT_MAX_FACTOR_CORR,
    DEFAULT_REBALANCE_FREQUENCIES,
    DEFAULT_SELECTION_FILTERS,
    DEFAULT_TOP_N_VALUES,
    parse_capital_plan_specs,
    run_low_corr_regime_capital_scaling,
)
from traditional_quant_research.experiments.low_corr_regime_yearly_validation import year_windows
from traditional_quant_research.experiments.multifactor_baseline import parse_selection_filters


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/low_corr_regime_capital_yearly_validation")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/brain/references/research_log/2026-06-02_low_corr_regime_capital_yearly_validation.md")
DEFAULT_YEARS = (2024, 2025, 2026)
DEFAULT_FINAL_END_DATE = "2026-06-01"


def run_low_corr_regime_capital_yearly_validation(
    *,
    root: str | None = None,
    years: Sequence[int] = DEFAULT_YEARS,
    final_end_date: str | None = DEFAULT_FINAL_END_DATE,
    horizon: int = DEFAULT_HORIZON,
    label_mode: str = "raw",
    max_factor_corr: float = DEFAULT_MAX_FACTOR_CORR,
    selection_filter_spec: str = DEFAULT_SELECTION_FILTERS,
    capital_plan_specs: str = DEFAULT_CAPITAL_PLAN_SPECS,
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
    """Run prior-year-fit dynamic capital validations over multiple years."""

    if not years:
        raise ValueError("years must not be empty")
    run_id = f"low_corr_regime_capital_yearly_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    child_output_dir = run_dir / "year_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    capital_plan_configs = parse_capital_plan_specs(capital_plan_specs, include_baseline=include_baseline)
    selection_filters = parse_selection_filters(selection_filter_spec)

    child_results: list[dict[str, Any]] = []
    summary_frames: list[pd.DataFrame] = []
    report_frames: list[pd.DataFrame] = []
    period_frames: list[pd.DataFrame] = []

    for year in years:
        windows = year_windows(int(year), final_end_date=final_end_date)
        child_result = run_low_corr_regime_capital_scaling(
            root=root,
            history_start_date=windows["history_start_date"],
            fit_start_date=windows["fit_start_date"],
            fit_end_date=windows["fit_end_date"],
            start_date=windows["start_date"],
            end_date=windows["end_date"],
            horizon=horizon,
            label_mode=label_mode,
            max_factor_corr=max_factor_corr,
            selection_filters=selection_filters,
            capital_plan_configs=capital_plan_configs,
            include_baseline=include_baseline,
            top_n_values=top_n_values,
            fee_bps_values=fee_bps_values,
            rebalance_frequencies=rebalance_frequencies,
            buffer_multipliers=buffer_multipliers,
            execution_constraints=execution_constraints,
            limit_threshold=limit_threshold,
            output_dir=child_output_dir,
        )
        child_result["eval_year"] = int(year)
        child_results.append(child_result)
        child_dir = Path(child_result["output_dir"])
        _append_child_frame(summary_frames, child_dir / "capital_scaling_summary.csv", eval_year=int(year), run_id=child_result["run_id"])
        _append_child_frame(report_frames, child_dir / "capital_plan_reports.csv", eval_year=int(year), run_id=child_result["run_id"])
        _append_child_frame(period_frames, child_dir / "capital_scaling_period_summary.csv", eval_year=int(year), run_id=child_result["run_id"])

    yearly_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    capital_reports = pd.concat(report_frames, ignore_index=True) if report_frames else pd.DataFrame()
    period_summary = pd.concat(period_frames, ignore_index=True) if period_frames else pd.DataFrame()
    aggregate = summarize_capital_yearly_validation(yearly_summary)

    result = {
        "run_id": run_id,
        "years": [int(year) for year in years],
        "final_end_date": final_end_date,
        "horizon": int(horizon),
        "label_mode": label_mode,
        "selection_filter_spec": selection_filter_spec,
        "capital_plan_specs": capital_plan_specs,
        "config_count": int(len(capital_plan_configs)),
        "top_n_values": [int(value) for value in top_n_values],
        "fee_bps_values": [float(value) for value in fee_bps_values],
        "rebalance_frequencies": list(rebalance_frequencies),
        "buffer_multipliers": [float(value) for value in buffer_multipliers],
        "execution_constraints": bool(execution_constraints),
        "limit_threshold": float(limit_threshold),
        "child_runs": [
            {
                "eval_year": item["eval_year"],
                "run_id": item["run_id"],
                "output_dir": item["output_dir"],
                "start_date": item["start_date"],
                "end_date": item["end_date"],
                "fit_start_date": item["fit_start_date"],
                "fit_end_date": item["fit_end_date"],
            }
            for item in child_results
        ],
        "best_aggregate_rows": _best_aggregate_rows(aggregate),
        "candidate_count": 0,
        "assessment": "multi-year backtest_only dynamic capital evidence; no strategy candidate is promoted by this experiment alone",
        "output_dir": str(run_dir),
    }

    yearly_summary.to_csv(run_dir / "capital_yearly_validation_summary.csv", index=False, encoding="utf-8-sig")
    capital_reports.to_csv(run_dir / "capital_yearly_validation_reports.csv", index=False, encoding="utf-8-sig")
    period_summary.to_csv(run_dir / "capital_yearly_validation_period_summary.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(run_dir / "capital_yearly_validation_aggregate.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_capital_yearly_validation_markdown(result, aggregate, yearly_summary)
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        research_log_path.write_text(markdown, encoding="utf-8")
    return result


def summarize_capital_yearly_validation(yearly_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-year dynamic-capital rows by plan and protocol."""

    if yearly_summary.empty:
        return pd.DataFrame()
    required = {
        "config_id",
        "capital_plan_spec",
        "fee_bps",
        "annualized_return",
        "sharpe",
        "max_drawdown",
        "mean_capital_scale",
        "delta_annualized_return_vs_baseline",
        "eval_year",
    }
    if missing := sorted(required - set(yearly_summary.columns)):
        raise ValueError(f"yearly_summary missing required columns: {missing}")
    group_cols = [
        "config_id",
        "capital_plan_spec",
        "signal",
        "horizon",
        "rebalance_frequency",
        "top_n",
        "fee_bps",
        "non_overlapping",
        "buffer_multiplier",
        "execution_constraints",
        "limit_threshold",
        "capital_col",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in yearly_summary.groupby(group_cols, dropna=False, sort=True):
        row = dict(zip(group_cols, keys))
        returns = pd.to_numeric(group["annualized_return"], errors="coerce")
        deltas = pd.to_numeric(group["delta_annualized_return_vs_baseline"], errors="coerce")
        drawdowns = pd.to_numeric(group["max_drawdown"], errors="coerce")
        sharpes = pd.to_numeric(group["sharpe"], errors="coerce")
        capital_scales = pd.to_numeric(group["mean_capital_scale"], errors="coerce")
        row.update(
            {
                "eval_year_count": int(group["eval_year"].nunique()),
                "mean_annualized_return": float(returns.mean()),
                "min_annualized_return": float(returns.min()),
                "positive_year_rate": float((returns > 0).mean()),
                "mean_delta_annualized_return_vs_baseline": float(deltas.mean()),
                "positive_delta_year_rate": float((deltas > 0).mean()),
                "mean_sharpe": float(sharpes.mean()),
                "worst_max_drawdown": float(drawdowns.min()),
                "mean_capital_scale": float(capital_scales.mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["fee_bps", "mean_annualized_return", "mean_delta_annualized_return_vs_baseline"],
        ascending=[True, False, False],
    )


def render_capital_yearly_validation_markdown(
    result: Mapping[str, Any],
    aggregate: pd.DataFrame,
    yearly_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Low-Corr Regime Capital Yearly Validation",
        "",
        "- Hypothesis: dynamic capital scaling can reduce weak-regime damage without the full opportunity cost of static regime skips.",
        f"- Years: `{result.get('years')}`; final end date `{result.get('final_end_date')}`.",
        f"- Stock Selection Filter: `{result.get('selection_filter_spec')}`.",
        f"- Capital Plan Specs: `{result.get('capital_plan_specs')}`.",
        f"- Grid: `{result.get('config_count')}` configs; Top-N `{result.get('top_n_values')}`, fees `{result.get('fee_bps_values')}`, frequencies `{result.get('rebalance_frequencies')}`, buffers `{result.get('buffer_multipliers')}`.",
        "- Assessment: this remains `backtest_only` evidence; candidate count is `0` until multi-year cost robustness and execution gates pass.",
        f"- Artifacts: `{result.get('output_dir')}`",
        "",
        "## Aggregate",
        "",
        _markdown_table(aggregate),
        "",
        "## Yearly Rows",
        "",
        _markdown_table(yearly_summary),
        "",
    ]
    return "\n".join(lines)


def _append_child_frame(frames: list[pd.DataFrame], path: Path, *, eval_year: int, run_id: str) -> None:
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if frame.empty:
        return
    frame.insert(0, "child_run_id", run_id)
    frame.insert(0, "eval_year", int(eval_year))
    frames.append(frame)


def _best_aggregate_rows(aggregate: pd.DataFrame, *, top: int = 10) -> list[dict[str, Any]]:
    if aggregate.empty:
        return []
    return aggregate.sort_values(["mean_annualized_return", "mean_delta_annualized_return_vs_baseline"], ascending=[False, False]).head(top).to_dict("records")


def _parse_years(spec: str) -> tuple[int, ...]:
    years = tuple(int(value.strip()) for value in spec.split(",") if value.strip())
    if not years:
        raise ValueError("expected at least one year")
    return years


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
    parser.add_argument("--years", default=",".join(str(year) for year in DEFAULT_YEARS))
    parser.add_argument("--final-end-date", default=DEFAULT_FINAL_END_DATE)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--label-mode", choices=("raw", "xsec-excess"), default="raw")
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
    result = run_low_corr_regime_capital_yearly_validation(
        root=args.root,
        years=_parse_years(args.years),
        final_end_date=args.final_end_date,
        horizon=args.horizon,
        label_mode=args.label_mode,
        max_factor_corr=args.max_factor_corr,
        selection_filter_spec=args.selection_filters,
        capital_plan_specs=args.capital_plans,
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
