"""Run the first PIT-data-to-backtest traditional quant baseline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from traditional_quant_research.dataset_v2 import load_pit_manifest, load_quality_report, load_tradeable_panel
from traditional_quant_research.diagnostics import (
    assign_time_split,
    quantile_returns,
    single_factor_diagnostics,
    summarize_factor_ic,
    top_n_backtest,
)
from traditional_quant_research.research_panel import (
    add_baseline_score,
    add_cross_sectional_zscores,
    build_factor_label_panel,
    default_factor_columns,
    panel_summary,
)


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/baseline_first_loop")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/brain/references/research_log/2026-06-02_first_loop_baseline.md")


def run_baseline_first_loop(
    *,
    root: str | None = None,
    start_date: str = "2024-01-01",
    end_date: str | None = None,
    horizon: int = 1,
    top_n: int = 100,
    fee_bps: float = 10.0,
    split_date: str | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    manifest = load_pit_manifest(root)
    quality = load_quality_report(root)
    label_col = f"fwd_ret_{horizon}d"
    run_id = f"baseline_first_loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_panel = load_tradeable_panel(root, start_date=start_date, end_date=end_date)
    factor_panel = build_factor_label_panel(raw_panel, horizons=sorted({1, 5, 20, horizon}))
    factor_columns = default_factor_columns()
    factor_panel = add_cross_sectional_zscores(factor_panel, factor_columns)
    factor_panel = add_baseline_score(factor_panel)

    if label_col not in factor_panel.columns:
        raise ValueError(f"label not generated: {label_col}")

    signal_columns = [f"{column}_z" for column in factor_columns] + ["baseline_score"]
    factor_panel = assign_time_split(factor_panel, split_date=split_date)
    ic_summary = summarize_factor_ic(factor_panel, signal_columns, label_col)
    quantile_summary = quantile_returns(factor_panel, "baseline_score", label_col, quantiles=5)
    backtest = top_n_backtest(factor_panel, "baseline_score", label_col, top_n=top_n, fee_bps=fee_bps)
    split_diagnostics = single_factor_diagnostics(
        factor_panel,
        signal_columns,
        label_col,
        split_date=split_date,
        top_n=top_n,
        fee_bps=fee_bps,
        quantiles=5,
    )

    result: dict[str, Any] = {
        "run_id": run_id,
        "snapshot_id": manifest.get("snapshot_id"),
        "snapshot_path": manifest.get("snapshot_path"),
        "start_date": start_date,
        "end_date": end_date,
        "horizon": horizon,
        "top_n": top_n,
        "fee_bps": fee_bps,
        "split_date": split_date,
        "raw_panel": panel_summary(raw_panel),
        "factor_panel": panel_summary(factor_panel),
        "label": label_col,
        "quality": {
            "failure_count": quality.get("failure_count"),
            "missing_bar_rows": quality.get("missing_bar_rows"),
            "st_rows": quality.get("st_rows"),
            "suspended_like_rows": quality.get("suspended_like_rows"),
        },
        "best_available_rank_ic": _row_for_signal(ic_summary, "baseline_score"),
        "out_of_sample_rank_ic": _row_for_signal_and_split(split_diagnostics["ic"], "baseline_score", "out_of_sample"),
        "top_n_backtest": backtest.summary,
        "out_of_sample_top_n_backtest": _row_for_signal_and_split(split_diagnostics["top_n"], "baseline_score", "out_of_sample"),
        "output_dir": str(run_dir),
    }

    ic_summary.to_csv(run_dir / "ic_summary.csv", index=False, encoding="utf-8-sig")
    quantile_summary.to_csv(run_dir / "quantile_returns.csv", index=False, encoding="utf-8-sig")
    backtest.daily_returns.to_csv(run_dir / "top_n_daily_returns.csv", index=False, encoding="utf-8-sig")
    split_diagnostics["ic"].to_csv(run_dir / "ic_by_split.csv", index=False, encoding="utf-8-sig")
    split_diagnostics["quantile"].to_csv(run_dir / "quantile_by_split.csv", index=False, encoding="utf-8-sig")
    split_diagnostics["top_n"].to_csv(run_dir / "top_n_by_split.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(
        json.dumps(_json_ready(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_markdown = render_summary_markdown(result, ic_summary, quantile_summary, split_diagnostics["ic"], split_diagnostics["top_n"])
    (run_dir / "summary.md").write_text(summary_markdown, encoding="utf-8")
    if write_research_log:
        research_log_path.write_text(summary_markdown, encoding="utf-8")
    return result


def render_summary_markdown(
    result: dict[str, Any],
    ic_summary: pd.DataFrame,
    quantile_summary: pd.DataFrame,
    split_ic_summary: pd.DataFrame | None = None,
    split_top_n_summary: pd.DataFrame | None = None,
) -> str:
    top_n = result["top_n_backtest"]
    baseline_ic = result.get("best_available_rank_ic") or {}
    oos_ic = result.get("out_of_sample_rank_ic") or {}
    oos_top_n = result.get("out_of_sample_top_n_backtest") or {}
    lines = [
        "# 2026-06-02 First Loop Baseline",
        "",
        "- Hypothesis: 基础传统价量因子在沪深主板 PIT 可交易股票池上具有可诊断的横截面排序信息。",
        f"- Data Scope: snapshot `{result.get('snapshot_id')}`, `{result.get('start_date')}` to `{result.get('factor_panel', {}).get('date_max')}`.",
        f"- Method: z-score baseline factors, average into `baseline_score`, evaluate `{result.get('label')}` with IC, quantile returns, Top-{result.get('top_n')} equal-weight baseline, and IS/OOS split diagnostics.",
        f"- Cost/Risk Assumptions: daily rebalance, next-open-to-horizon-close label, `{result.get('fee_bps')}` bps per one-way turnover unit, no industry/size neutralization yet.",
        f"- Result: RankIC mean `{_fmt(baseline_ic.get('mean_rank_ic'))}`, RankICIR `{_fmt(baseline_ic.get('rank_icir'))}`, Top-N annualized return `{_fmt(top_n.get('annualized_return'))}`, Sharpe `{_fmt(top_n.get('sharpe'))}`, max drawdown `{_fmt(top_n.get('max_drawdown'))}`.",
        f"- OOS Result: split date `{result.get('split_date')}`, OOS RankIC mean `{_fmt(oos_ic.get('mean_rank_ic'))}`, OOS RankICIR `{_fmt(oos_ic.get('rank_icir'))}`, OOS Top-N annualized return `{_fmt(oos_top_n.get('annualized_return'))}`, OOS Sharpe `{_fmt(oos_top_n.get('sharpe'))}`.",
        "- Failure Modes: 因子未做行业/市值中性化，Top-N 回测是研究基线而非完整生产执行模型，涨跌停成交约束尚未建模。",
        "- Next Step: 增加行业/市值数据或替代暴露 proxy，扩展单因子报告，并加入样本内/样本外切分。",
        "",
        "## Panel",
        "",
        f"- Rows: `{result.get('factor_panel', {}).get('rows')}`",
        f"- Dates: `{result.get('factor_panel', {}).get('date_count')}`",
        f"- Securities: `{result.get('factor_panel', {}).get('security_count')}`",
        "",
        "## IC Summary",
        "",
        _markdown_table(ic_summary),
        "",
        "## IC By Split",
        "",
        _markdown_table(split_ic_summary if split_ic_summary is not None else pd.DataFrame()),
        "",
        "## Quantile Returns",
        "",
        _markdown_table(quantile_summary),
        "",
        "## Top-N By Split",
        "",
        _markdown_table(split_top_n_summary if split_top_n_summary is not None else pd.DataFrame()),
        "",
        f"Artifacts: `{result.get('output_dir')}`",
        "",
    ]
    return "\n".join(lines)


def _row_for_signal(frame: pd.DataFrame, signal: str) -> dict[str, Any]:
    if frame.empty or "signal" not in frame.columns:
        return {}
    rows = frame.loc[frame["signal"] == signal]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _row_for_signal_and_split(frame: pd.DataFrame, signal: str, sample_split: str) -> dict[str, Any]:
    if frame.empty or "signal" not in frame.columns or "sample_split" not in frame.columns:
        return {}
    rows = frame.loc[(frame["signal"] == signal) & (frame["sample_split"] == sample_split)]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 20) -> str:
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
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--split-date", default=None, help="Date boundary for in_sample/out_of_sample diagnostics.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_baseline_first_loop(
        root=args.root,
        start_date=args.start_date,
        end_date=args.end_date,
        horizon=args.horizon,
        top_n=args.top_n,
        fee_bps=args.fee_bps,
        split_date=args.split_date,
        output_dir=args.output_dir,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
