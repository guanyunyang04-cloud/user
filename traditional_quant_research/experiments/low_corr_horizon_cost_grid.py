"""Run a cost-sensitive horizon/rebalance grid for the low-corr signal."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.experiments.low_corr_exposure_grid import (
    DEFAULT_MAX_FACTOR_CORR,
    parse_filter_specs,
    run_low_corr_exposure_grid,
)


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/low_corr_horizon_cost_grid")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-02_low_corr_horizon_cost_grid.md")
DEFAULT_SELECTION_FILTERS = "log_amount_mean_20d_z>=-0.8,momentum_20d_z>=-0.8"
DEFAULT_HORIZONS = (5, 10, 20)
DEFAULT_TOP_N_VALUES = (100, 200)
DEFAULT_FEE_BPS_VALUES = (0.0, 30.0)
DEFAULT_REBALANCE_FREQUENCIES = ("weekly", "monthly")
DEFAULT_BUFFER_MULTIPLIERS = (2.0, 3.0)


def run_low_corr_horizon_cost_grid(
    *,
    root: str | None = None,
    history_start_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    fit_start_date: str | None = None,
    fit_end_date: str | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    label_mode: str = "raw",
    max_factor_corr: float = DEFAULT_MAX_FACTOR_CORR,
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
    """Run the existing low-corr exposure grid across multiple horizons."""

    if not horizons:
        raise ValueError("horizons must not be empty")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must be positive")
    if any(top_n <= 0 for top_n in top_n_values):
        raise ValueError("top_n_values must be positive")
    if any(value < 1.0 for value in buffer_multipliers):
        raise ValueError("buffer_multipliers must be at least 1.0")
    if limit_threshold <= 0:
        raise ValueError("limit_threshold must be positive")

    run_id = f"low_corr_horizon_cost_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    child_output_dir = run_dir / "horizon_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    filter_configs = parse_filter_specs(filter_specs, include_baseline=include_baseline)
    child_results: list[dict[str, Any]] = []
    summary_frames: list[pd.DataFrame] = []
    yearly_frames: list[pd.DataFrame] = []
    period_frames: list[pd.DataFrame] = []
    selection_report_frames: list[pd.DataFrame] = []

    for horizon in horizons:
        child_result = run_low_corr_exposure_grid(
            root=root,
            history_start_date=history_start_date,
            fit_start_date=fit_start_date,
            fit_end_date=fit_end_date,
            start_date=start_date,
            end_date=end_date,
            horizon=int(horizon),
            label_mode=label_mode,
            max_factor_corr=max_factor_corr,
            filter_configs=filter_configs,
            include_baseline=include_baseline,
            top_n_values=top_n_values,
            fee_bps_values=fee_bps_values,
            rebalance_frequencies=rebalance_frequencies,
            buffer_multipliers=buffer_multipliers,
            execution_constraints=execution_constraints,
            limit_threshold=limit_threshold,
            output_dir=child_output_dir,
        )
        child_result["grid_horizon"] = int(horizon)
        child_results.append(child_result)
        child_dir = Path(child_result["output_dir"])
        _append_child_frame(summary_frames, child_dir / "grid_summary.csv", grid_horizon=int(horizon), run_id=child_result["run_id"])
        _append_child_frame(yearly_frames, child_dir / "grid_yearly_summary.csv", grid_horizon=int(horizon), run_id=child_result["run_id"])
        _append_child_frame(period_frames, child_dir / "grid_period_summary.csv", grid_horizon=int(horizon), run_id=child_result["run_id"])
        _append_child_frame(selection_report_frames, child_dir / "grid_selection_reports.csv", grid_horizon=int(horizon), run_id=child_result["run_id"])

    raw_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    yearly_summary = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    period_summary = pd.concat(period_frames, ignore_index=True) if period_frames else pd.DataFrame()
    selection_reports = pd.concat(selection_report_frames, ignore_index=True) if selection_report_frames else pd.DataFrame()
    cost_summary = summarize_horizon_cost_grid(raw_summary)

    result = {
        "run_id": run_id,
        "horizons": [int(horizon) for horizon in horizons],
        "label_mode": label_mode,
        "filter_specs": filter_specs,
        "config_count": int(len(filter_configs)),
        "top_n_values": [int(value) for value in top_n_values],
        "fee_bps_values": [float(value) for value in fee_bps_values],
        "rebalance_frequencies": list(rebalance_frequencies),
        "buffer_multipliers": [float(value) for value in buffer_multipliers],
        "execution_constraints": bool(execution_constraints),
        "limit_threshold": float(limit_threshold),
        "child_runs": [
            {
                "grid_horizon": item["grid_horizon"],
                "run_id": item["run_id"],
                "output_dir": item["output_dir"],
                "start_date": item["start_date"],
                "end_date": item["end_date"],
                "fit_start_date": item["fit_start_date"],
                "fit_end_date": item["fit_end_date"],
                "low_corr_factor_columns": item.get("low_corr_factor_columns", []),
            }
            for item in child_results
        ],
        "best_30bps_rows": best_cost_grid_rows(cost_summary, fee_bps=30.0),
        "candidate_count": 0,
        "assessment": "backtest_only cost/horizon grid evidence; no strategy candidate is promoted by this experiment alone",
        "output_dir": str(run_dir),
    }

    raw_summary.to_csv(run_dir / "horizon_cost_raw_summary.csv", index=False, encoding="utf-8-sig")
    cost_summary.to_csv(run_dir / "horizon_cost_summary.csv", index=False, encoding="utf-8-sig")
    yearly_summary.to_csv(run_dir / "horizon_cost_yearly_summary.csv", index=False, encoding="utf-8-sig")
    period_summary.to_csv(run_dir / "horizon_cost_period_summary.csv", index=False, encoding="utf-8-sig")
    selection_reports.to_csv(run_dir / "horizon_cost_selection_reports.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_horizon_cost_grid_markdown(result, cost_summary, period_summary)
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        research_log_path.write_text(markdown, encoding="utf-8")
    return result


def summarize_horizon_cost_grid(summary: pd.DataFrame) -> pd.DataFrame:
    """Attach same-protocol 0 bps rows and cost-drag diagnostics."""

    if summary.empty:
        return pd.DataFrame()
    required = {
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
    }
    if missing := sorted(required - set(summary.columns)):
        raise ValueError(f"summary missing required columns: {missing}")
    output = summary.copy()
    protocol_cols = [
        "config_id",
        "filter_spec",
        "signal",
        "horizon",
        "rebalance_frequency",
        "top_n",
        "non_overlapping",
        "buffer_multiplier",
        "execution_constraints",
        "limit_threshold",
    ]
    metrics = ["annualized_return", "sharpe", "max_drawdown", "mean_turnover"]
    fee0 = output.loc[pd.to_numeric(output["fee_bps"], errors="coerce") == 0.0, protocol_cols + metrics].copy()
    fee0 = fee0.rename(columns={metric: f"{metric}_0bps" for metric in metrics})
    output = output.merge(fee0, on=protocol_cols, how="left")
    output["cost_drag_vs_0bps"] = pd.to_numeric(output["annualized_return"], errors="coerce") - pd.to_numeric(output["annualized_return_0bps"], errors="coerce")
    output["turnover_adjusted_return"] = pd.to_numeric(output["annualized_return"], errors="coerce") / pd.to_numeric(output["mean_turnover"], errors="coerce").replace(0, np.nan)
    return output.sort_values(
        ["fee_bps", "annualized_return", "sharpe", "mean_turnover"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def best_cost_grid_rows(summary: pd.DataFrame, *, fee_bps: float = 30.0, top: int = 10) -> list[dict[str, Any]]:
    if summary.empty:
        return []
    fee_rows = summary.loc[pd.to_numeric(summary["fee_bps"], errors="coerce") == float(fee_bps)]
    if fee_rows.empty:
        fee_rows = summary
    return fee_rows.sort_values(["annualized_return", "sharpe", "max_drawdown"], ascending=[False, False, False]).head(top).to_dict("records")


def render_horizon_cost_grid_markdown(
    result: Mapping[str, Any],
    cost_summary: pd.DataFrame,
    period_summary: pd.DataFrame,
) -> str:
    best_rows = pd.DataFrame(result.get("best_30bps_rows") or [])
    lines = [
        "# Low-Corr Horizon Cost Grid",
        "",
        "- Hypothesis: lower-turnover holding-period and rebalance choices may improve the low-corr watchlist under 30 bps costs.",
        f"- Horizons: `{result.get('horizons')}`.",
        f"- Filter Specs: `{result.get('filter_specs')}`.",
        f"- Grid: `{result.get('config_count')}` filter configs; Top-N `{result.get('top_n_values')}`, fees `{result.get('fee_bps_values')}`, frequencies `{result.get('rebalance_frequencies')}`, buffers `{result.get('buffer_multipliers')}`.",
        f"- Execution Constraints: `{result.get('execution_constraints')}` with limit threshold `{result.get('limit_threshold')}`.",
        "- Assessment: this remains `backtest_only` evidence; candidate count is `0` until multi-year cost robustness and execution gates pass.",
        f"- Artifacts: `{result.get('output_dir')}`",
        "",
        "## Best 30 bps Rows",
        "",
        _markdown_table(best_rows),
        "",
        "## Cost Summary",
        "",
        _markdown_table(cost_summary),
        "",
        "## Monthly/Quarterly Slices",
        "",
        _markdown_table(period_summary),
        "",
    ]
    return "\n".join(lines)


def _append_child_frame(frames: list[pd.DataFrame], path: Path, *, grid_horizon: int, run_id: str) -> None:
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if frame.empty:
        return
    frame.insert(0, "child_run_id", run_id)
    frame.insert(0, "grid_horizon", int(grid_horizon))
    frames.append(frame)


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
    parser.add_argument("--history-start-date", default=None)
    parser.add_argument("--fit-start-date", default=None)
    parser.add_argument("--fit-end-date", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--horizons", default=",".join(str(value) for value in DEFAULT_HORIZONS))
    parser.add_argument("--label-mode", choices=("raw", "xsec-excess"), default="raw")
    parser.add_argument("--max-factor-corr", type=float, default=DEFAULT_MAX_FACTOR_CORR)
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
    result = run_low_corr_horizon_cost_grid(
        root=args.root,
        history_start_date=args.history_start_date,
        fit_start_date=args.fit_start_date,
        fit_end_date=args.fit_end_date,
        start_date=args.start_date,
        end_date=args.end_date,
        horizons=_parse_int_tuple(args.horizons),
        label_mode=args.label_mode,
        max_factor_corr=args.max_factor_corr,
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
