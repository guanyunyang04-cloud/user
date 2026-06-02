"""Run full-cycle single-factor diagnostics across horizons and years."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.dataset_v2 import load_pit_manifest, load_tradeable_panel
from traditional_quant_research.diagnostics import (
    quantile_returns_by_group,
    summarize_factor_ic,
    top_n_backtest_by_group,
)
from traditional_quant_research.research_panel import (
    add_baseline_score,
    add_cross_sectional_zscores,
    build_factor_label_panel,
    default_factor_columns,
    panel_summary,
)


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/full_cycle_factor_diagnostics")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-02_full_cycle_factor_diagnostics.md")
DEFAULT_HORIZONS = (1, 5, 20)
DEFAULT_TOP_N_VALUES = (50, 100, 200)


def run_full_cycle_factor_diagnostics(
    *,
    root: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    top_n_values: Sequence[int] = DEFAULT_TOP_N_VALUES,
    fee_bps: float = 10.0,
    top_n_signals: Sequence[str] = ("baseline_score",),
    quantile_signals: Sequence[str] = ("baseline_score",),
    include_top_n: bool = True,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must be positive")
    if any(top_n <= 0 for top_n in top_n_values):
        raise ValueError("top_n_values must be positive")

    manifest = load_pit_manifest(root)
    dataset = manifest.get("dataset", {})
    effective_start = start_date or dataset.get("date_min")
    effective_end = end_date or dataset.get("date_max")
    run_id = f"full_cycle_factor_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    factor_columns = default_factor_columns()
    signal_columns = [f"{column}_z" for column in factor_columns] + ["baseline_score"]

    all_quantile_rows: list[pd.DataFrame] = []
    all_yearly_top_n_rows: list[pd.DataFrame] = []
    raw_panel = load_tradeable_panel(root, start_date=effective_start, end_date=effective_end)
    factor_panel = build_factor_label_panel(raw_panel, horizons=tuple(sorted(set(horizons))))
    factor_panel = add_cross_sectional_zscores(factor_panel, factor_columns)
    factor_panel = add_baseline_score(factor_panel)
    factor_panel["date"] = pd.to_datetime(factor_panel["date"])
    factor_panel["year"] = factor_panel["date"].dt.year

    yearly_ic_frames: list[pd.DataFrame] = []
    ic_frames: list[pd.DataFrame] = []
    for horizon in horizons:
        label_col = f"fwd_ret_{horizon}d"
        ic_summary = summarize_factor_ic(factor_panel, signal_columns, label_col)
        if not ic_summary.empty:
            ic_summary.insert(0, "horizon", int(horizon))
            ic_frames.append(ic_summary)

        yearly_ic = summarize_factor_ic_by_group_fast(factor_panel, signal_columns, label_col, group_col="year")
        if not yearly_ic.empty:
            yearly_ic.insert(0, "horizon", int(horizon))
            yearly_ic_frames.append(yearly_ic)

        for signal_col in quantile_signals:
            quantile = quantile_returns_by_group(factor_panel, signal_col, label_col, group_col="year", quantiles=5)
            if not quantile.empty:
                quantile.insert(0, "horizon", int(horizon))
                all_quantile_rows.append(quantile)

        if include_top_n:
            for top_n in top_n_values:
                yearly_top_n = top_n_backtest_by_group(
                    factor_panel,
                    top_n_signals,
                    label_col,
                    group_col="year",
                    top_n=int(top_n),
                    fee_bps=fee_bps,
                )
                if not yearly_top_n.empty:
                    yearly_top_n.insert(0, "horizon", int(horizon))
                    all_yearly_top_n_rows.append(yearly_top_n)

    ic_summary_all = _concat_or_empty(ic_frames)
    yearly_ic_all = _concat_or_empty(yearly_ic_frames)
    quantile_all = _concat_or_empty(all_quantile_rows)
    yearly_top_n_all = _concat_or_empty(all_yearly_top_n_rows)
    top_n_all = summarize_top_n_from_yearly(yearly_top_n_all)
    stability = summarize_factor_stability(yearly_ic_all)
    panel = panel_summary(factor_panel)

    result = {
        "run_id": run_id,
        "snapshot_id": manifest.get("snapshot_id"),
        "snapshot_path": manifest.get("snapshot_path"),
        "start_date": effective_start,
        "end_date": effective_end,
        "years": years_in_range(effective_start, effective_end),
        "horizons": [int(value) for value in horizons],
        "top_n_values": [int(value) for value in top_n_values],
        "top_n_signals": list(top_n_signals),
        "quantile_signals": list(quantile_signals),
        "include_top_n": bool(include_top_n),
        "fee_bps": float(fee_bps),
        "panel": panel,
        "best_rank_ic": best_rank_ic_rows(ic_summary_all),
        "baseline_stability": baseline_stability_rows(stability),
        "output_dir": str(run_dir),
    }

    ic_summary_all.to_csv(run_dir / "ic_summary.csv", index=False, encoding="utf-8-sig")
    yearly_ic_all.to_csv(run_dir / "yearly_ic_summary.csv", index=False, encoding="utf-8-sig")
    quantile_all.to_csv(run_dir / "yearly_quantile_returns.csv", index=False, encoding="utf-8-sig")
    top_n_all.to_csv(run_dir / "top_n_summary.csv", index=False, encoding="utf-8-sig")
    yearly_top_n_all.to_csv(run_dir / "yearly_top_n_summary.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(run_dir / "factor_stability.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")

    markdown = render_full_cycle_markdown(result, ic_summary_all, stability, top_n_all)
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        research_log_path.write_text(markdown, encoding="utf-8")
    return result


def summarize_factor_stability(yearly_ic: pd.DataFrame) -> pd.DataFrame:
    if yearly_ic.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (horizon, signal), group in yearly_ic.groupby(["horizon", "signal"], sort=True):
        values = pd.to_numeric(group["mean_rank_ic"], errors="coerce").dropna()
        if values.empty:
            positive_rate = np.nan
            mean_rank_ic = np.nan
            min_rank_ic = np.nan
            max_rank_ic = np.nan
        else:
            positive_rate = float((values > 0).mean())
            mean_rank_ic = float(values.mean())
            min_rank_ic = float(values.min())
            max_rank_ic = float(values.max())
        rows.append(
            {
                "horizon": int(horizon),
                "signal": signal,
                "years": int(group["year"].nunique()),
                "positive_year_rate": positive_rate,
                "mean_yearly_rank_ic": mean_rank_ic,
                "min_yearly_rank_ic": min_rank_ic,
                "max_yearly_rank_ic": max_rank_ic,
            }
        )
    return pd.DataFrame(rows)


def summarize_factor_ic_by_group_fast(
    frame: pd.DataFrame,
    signal_cols: Sequence[str],
    label_col: str,
    *,
    group_col: str,
) -> pd.DataFrame:
    """Summarize IC by group without rebuilding panels per group."""

    if group_col not in frame.columns:
        raise ValueError(f"missing required column: {group_col}")
    rows: list[dict[str, Any]] = []
    for group_name, group in frame.groupby(group_col, sort=True):
        summary = summarize_factor_ic(group, signal_cols, label_col)
        if summary.empty:
            continue
        summary.insert(0, group_col, group_name)
        rows.extend(summary.to_dict("records"))
    return pd.DataFrame(rows)


def summarize_top_n_from_yearly(yearly_top_n: pd.DataFrame) -> pd.DataFrame:
    if yearly_top_n.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (horizon, signal, top_n), group in yearly_top_n.groupby(["horizon", "signal", "top_n"], sort=True):
        weights = pd.to_numeric(group["periods"], errors="coerce").fillna(0)
        row = {
            "horizon": int(horizon),
            "signal": signal,
            "top_n": int(top_n),
            "periods": int(weights.sum()),
            "fee_bps": weighted_mean(group["fee_bps"], weights),
            "annualized_return": weighted_mean(group["annualized_return"], weights),
            "volatility": weighted_mean(group["volatility"], weights),
            "sharpe": weighted_mean(group["sharpe"], weights),
            "max_drawdown": float(pd.to_numeric(group["max_drawdown"], errors="coerce").min()),
            "mean_turnover": weighted_mean(group["mean_turnover"], weights),
            "mean_gross_return": weighted_mean(group["mean_gross_return"], weights),
            "mean_net_return": weighted_mean(group["mean_net_return"], weights),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    mask = numeric.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float((numeric[mask] * weights[mask]).sum() / weights[mask].sum())


def years_in_range(start_date: str, end_date: str) -> list[int]:
    start = pd.Timestamp(start_date).year
    end = pd.Timestamp(end_date).year
    return list(range(start, end + 1))


def best_rank_ic_rows(ic_summary: pd.DataFrame, *, top: int = 5) -> list[dict[str, Any]]:
    if ic_summary.empty:
        return []
    frame = ic_summary.copy()
    frame["abs_mean_rank_ic"] = pd.to_numeric(frame["mean_rank_ic"], errors="coerce").abs()
    return frame.sort_values(["horizon", "abs_mean_rank_ic"], ascending=[True, False]).groupby("horizon", sort=True).head(top).drop(columns=["abs_mean_rank_ic"]).to_dict("records")


def baseline_stability_rows(stability: pd.DataFrame) -> list[dict[str, Any]]:
    if stability.empty:
        return []
    return stability.loc[stability["signal"] == "baseline_score"].to_dict("records")


def render_full_cycle_markdown(
    result: dict[str, Any],
    ic_summary: pd.DataFrame,
    stability: pd.DataFrame,
    top_n_summary: pd.DataFrame,
) -> str:
    lines = [
        "# 2026-06-02 Full Cycle Factor Diagnostics",
        "",
        f"- Data Scope: snapshot `{result.get('snapshot_id')}`, `{result.get('start_date')}` to `{result.get('end_date')}`.",
        f"- Panel: `{result.get('panel', {}).get('rows')}` rows, `{result.get('panel', {}).get('date_count')}` dates, `{result.get('panel', {}).get('security_count')}` securities.",
        f"- Horizons: `{result.get('horizons')}`; Top-N values: `{result.get('top_n_values')}`; include Top-N: `{result.get('include_top_n')}`; fee bps: `{result.get('fee_bps')}`.",
        f"- Method: baseline price/volume factors are cross-sectional z-scored per date; `baseline_score` is an equal-weight factor score; diagnostics cover IC, yearly IC stability, yearly quantiles, and Top-N baselines for `{result.get('top_n_signals')}`.",
        "- Assessment: this is still diagnostic evidence; extreme long-horizon labels from the v2 audit require follow-up before upgrading conclusions.",
        "- Next Step: add return-adjusted labels/excess returns, cost sensitivity, and rebalancing-frequency comparison before entering multi-factor or traditional ML.",
        "",
        "## IC Summary",
        "",
        _markdown_table(ic_summary),
        "",
        "## Stability",
        "",
        _markdown_table(stability),
        "",
        "## Top-N Summary",
        "",
        _markdown_table(top_n_summary),
        "",
        f"Artifacts: `{result.get('output_dir')}`",
        "",
    ]
    return "\n".join(lines)


def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--horizons", default="1,5,20", help="Comma-separated forward label horizons.")
    parser.add_argument("--top-n", default="50,100,200", help="Comma-separated Top-N portfolio sizes.")
    parser.add_argument("--top-n-signals", default="baseline_score", help="Comma-separated signals for Top-N backtests.")
    parser.add_argument("--skip-top-n", action="store_true", help="Skip Top-N backtests for faster IC/stability diagnostics.")
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = tuple(int(value.strip()) for value in args.horizons.split(",") if value.strip())
    top_n_values = tuple(int(value.strip()) for value in args.top_n.split(",") if value.strip())
    top_n_signals = tuple(value.strip() for value in args.top_n_signals.split(",") if value.strip())
    result = run_full_cycle_factor_diagnostics(
        root=args.root,
        start_date=args.start_date,
        end_date=args.end_date,
        horizons=horizons,
        top_n_values=top_n_values,
        fee_bps=args.fee_bps,
        top_n_signals=top_n_signals,
        include_top_n=not args.skip_top_n,
        output_dir=args.output_dir,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
