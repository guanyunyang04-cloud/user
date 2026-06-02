"""Validate low-corr horizon/cost grid settings across evaluation years."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.experiments.low_corr_horizon_cost_grid import (
    DEFAULT_BUFFER_MULTIPLIERS,
    DEFAULT_FEE_BPS_VALUES,
    DEFAULT_OUTPUT_DIR as COST_GRID_OUTPUT_DIR,
    DEFAULT_REBALANCE_FREQUENCIES,
    DEFAULT_SELECTION_FILTERS,
    DEFAULT_TOP_N_VALUES,
    run_low_corr_horizon_cost_grid,
)
from traditional_quant_research.experiments.low_corr_regime_yearly_validation import year_windows


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/low_corr_horizon_cost_yearly_validation")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-02_low_corr_horizon_cost_yearly_validation.md")
DEFAULT_YEARS = (2024, 2025, 2026)
DEFAULT_FINAL_END_DATE = "2026-06-01"
DEFAULT_HORIZONS = (20,)


def run_low_corr_horizon_cost_yearly_validation(
    *,
    root: str | None = None,
    years: Sequence[int] = DEFAULT_YEARS,
    final_end_date: str | None = DEFAULT_FINAL_END_DATE,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    label_mode: str = "raw",
    filter_specs: str = DEFAULT_SELECTION_FILTERS,
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
    """Run prior-year-fit low-corr horizon/cost validations over years."""

    if not years:
        raise ValueError("years must not be empty")
    if not horizons:
        raise ValueError("horizons must not be empty")

    run_id = f"low_corr_horizon_cost_yearly_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    child_output_dir = run_dir / "year_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    child_results: list[dict[str, Any]] = []
    summary_frames: list[pd.DataFrame] = []
    yearly_frames: list[pd.DataFrame] = []
    period_frames: list[pd.DataFrame] = []

    for year in years:
        windows = year_windows(int(year), final_end_date=final_end_date)
        child_result = run_low_corr_horizon_cost_grid(
            root=root,
            history_start_date=windows["history_start_date"],
            fit_start_date=windows["fit_start_date"],
            fit_end_date=windows["fit_end_date"],
            start_date=windows["start_date"],
            end_date=windows["end_date"],
            horizons=horizons,
            label_mode=label_mode,
            filter_specs=filter_specs,
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
        _append_child_frame(summary_frames, child_dir / "horizon_cost_summary.csv", eval_year=int(year), run_id=child_result["run_id"])
        _append_child_frame(yearly_frames, child_dir / "horizon_cost_yearly_summary.csv", eval_year=int(year), run_id=child_result["run_id"])
        _append_child_frame(period_frames, child_dir / "horizon_cost_period_summary.csv", eval_year=int(year), run_id=child_result["run_id"])

    validation_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    child_yearly_summary = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    period_summary = pd.concat(period_frames, ignore_index=True) if period_frames else pd.DataFrame()
    aggregate = summarize_horizon_cost_yearly_validation(validation_summary)

    result = {
        "run_id": run_id,
        "years": [int(year) for year in years],
        "final_end_date": final_end_date,
        "horizons": [int(horizon) for horizon in horizons],
        "label_mode": label_mode,
        "filter_specs": filter_specs,
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
                "horizons": item["horizons"],
                "start_date": item["child_runs"][0]["start_date"] if item.get("child_runs") else None,
                "end_date": item["child_runs"][0]["end_date"] if item.get("child_runs") else None,
            }
            for item in child_results
        ],
        "best_30bps_rows": _best_aggregate_rows(aggregate, fee_bps=30.0),
        "candidate_count": 0,
        "assessment": "multi-year backtest_only horizon/cost evidence; no strategy candidate is promoted by this experiment alone",
        "output_dir": str(run_dir),
        "cost_grid_output_dir": str(COST_GRID_OUTPUT_DIR),
    }

    validation_summary.to_csv(run_dir / "horizon_cost_yearly_validation_summary.csv", index=False, encoding="utf-8-sig")
    child_yearly_summary.to_csv(run_dir / "horizon_cost_yearly_validation_child_yearly.csv", index=False, encoding="utf-8-sig")
    period_summary.to_csv(run_dir / "horizon_cost_yearly_validation_period_summary.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(run_dir / "horizon_cost_yearly_validation_aggregate.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_horizon_cost_yearly_validation_markdown(result, aggregate, validation_summary)
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        research_log_path.write_text(markdown, encoding="utf-8")
    return result


def summarize_horizon_cost_yearly_validation(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-evaluation-year cost grid rows by protocol."""

    if summary.empty:
        return pd.DataFrame()
    required = {
        "eval_year",
        "config_id",
        "filter_spec",
        "signal",
        "horizon",
        "rebalance_frequency",
        "top_n",
        "fee_bps",
        "non_overlapping",
        "buffer_multiplier",
        "execution_constraints",
        "limit_threshold",
        "annualized_return",
        "sharpe",
        "max_drawdown",
        "mean_turnover",
        "cost_drag_vs_0bps",
    }
    if missing := sorted(required - set(summary.columns)):
        raise ValueError(f"summary missing required columns: {missing}")
    group_cols = [
        "config_id",
        "filter_spec",
        "signal",
        "horizon",
        "rebalance_frequency",
        "top_n",
        "fee_bps",
        "non_overlapping",
        "buffer_multiplier",
        "execution_constraints",
        "limit_threshold",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in summary.groupby(group_cols, dropna=False, sort=True):
        row = dict(zip(group_cols, keys))
        returns = pd.to_numeric(group["annualized_return"], errors="coerce")
        sharpes = pd.to_numeric(group["sharpe"], errors="coerce")
        drawdowns = pd.to_numeric(group["max_drawdown"], errors="coerce")
        turnover = pd.to_numeric(group["mean_turnover"], errors="coerce")
        cost_drag = pd.to_numeric(group["cost_drag_vs_0bps"], errors="coerce")
        row.update(
            {
                "eval_year_count": int(group["eval_year"].nunique()),
                "mean_annualized_return": float(returns.mean()),
                "min_annualized_return": float(returns.min()),
                "positive_year_rate": float((returns > 0).mean()),
                "mean_sharpe": float(sharpes.mean()),
                "worst_max_drawdown": float(drawdowns.min()),
                "mean_turnover": float(turnover.mean()),
                "mean_cost_drag_vs_0bps": float(cost_drag.mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["fee_bps", "mean_annualized_return", "positive_year_rate", "worst_max_drawdown"],
        ascending=[True, False, False, False],
    )


def render_horizon_cost_yearly_validation_markdown(
    result: Mapping[str, Any],
    aggregate: pd.DataFrame,
    validation_summary: pd.DataFrame,
) -> str:
    best_rows = pd.DataFrame(result.get("best_30bps_rows") or [])
    lines = [
        "# Low-Corr Horizon Cost Yearly Validation",
        "",
        "- Hypothesis: 20-day or lower-turnover protocols that looked promising in 2026 should also survive prior-year-fit validation across multiple years.",
        f"- Years: `{result.get('years')}`; final end date `{result.get('final_end_date')}`.",
        f"- Horizons: `{result.get('horizons')}`.",
        f"- Filter Specs: `{result.get('filter_specs')}`.",
        f"- Protocol Grid: Top-N `{result.get('top_n_values')}`, fees `{result.get('fee_bps_values')}`, frequencies `{result.get('rebalance_frequencies')}`, buffers `{result.get('buffer_multipliers')}`.",
        "- Assessment: this remains `backtest_only` evidence; candidate count is `0` until all promotion gates pass.",
        f"- Artifacts: `{result.get('output_dir')}`",
        "",
        "## Best 30 bps Aggregate Rows",
        "",
        _markdown_table(best_rows),
        "",
        "## Aggregate",
        "",
        _markdown_table(aggregate),
        "",
        "## Year Rows",
        "",
        _markdown_table(validation_summary),
        "",
    ]
    return "\n".join(lines)


def _append_child_frame(frames: list[pd.DataFrame], path: Path, *, eval_year: int, run_id: str) -> None:
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if frame.empty:
        return
    if "child_run_id" not in frame.columns:
        frame.insert(0, "child_run_id", run_id)
    if "eval_year" not in frame.columns:
        frame.insert(0, "eval_year", int(eval_year))
    frames.append(frame)


def _best_aggregate_rows(aggregate: pd.DataFrame, *, fee_bps: float, top: int = 10) -> list[dict[str, Any]]:
    if aggregate.empty:
        return []
    work = aggregate.loc[pd.to_numeric(aggregate["fee_bps"], errors="coerce") == float(fee_bps)]
    if work.empty:
        work = aggregate
    return work.sort_values(["mean_annualized_return", "positive_year_rate", "worst_max_drawdown"], ascending=[False, False, False]).head(top).to_dict("records")


def _parse_years(spec: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in spec.split(",") if value.strip())
    if not values:
        raise ValueError("expected at least one year")
    return values


def _parse_int_tuple(spec: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in spec.split(",") if value.strip())
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _parse_float_tuple(spec: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in spec.split(",") if value.strip())
    if not values:
        raise ValueError("expected at least one numeric value")
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
    parser.add_argument("--years", default=",".join(str(year) for year in DEFAULT_YEARS))
    parser.add_argument("--final-end-date", default=DEFAULT_FINAL_END_DATE)
    parser.add_argument("--horizons", default=",".join(str(value) for value in DEFAULT_HORIZONS))
    parser.add_argument("--label-mode", choices=("raw", "xsec-excess"), default="raw")
    parser.add_argument("--filter-specs", default=DEFAULT_SELECTION_FILTERS)
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
    result = run_low_corr_horizon_cost_yearly_validation(
        root=args.root,
        years=_parse_years(args.years),
        final_end_date=args.final_end_date,
        horizons=_parse_int_tuple(args.horizons),
        label_mode=args.label_mode,
        filter_specs=args.filter_specs,
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
