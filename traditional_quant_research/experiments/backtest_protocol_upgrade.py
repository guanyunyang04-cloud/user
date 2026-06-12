"""Run baseline Top-N, cost, and rebalance-frequency protocol checks."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from traditional_quant_research.backtest_protocol import top_n_rebalance_backtest
from traditional_quant_research.dataset_v2 import load_pit_manifest, load_tradeable_panel
from traditional_quant_research.research_panel import (
    add_baseline_score,
    add_cross_sectional_zscores,
    build_factor_label_panel,
    default_factor_columns,
    panel_summary,
)


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/backtest_protocol_upgrade")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/brain/references/research_log/2026-06-02_backtest_protocol_upgrade.md")


def run_backtest_protocol_upgrade(
    *,
    root: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    horizon: int = 1,
    top_n_values: Sequence[int] = (50, 100, 200),
    fee_bps_values: Sequence[float] = (0.0, 10.0, 20.0),
    rebalance_frequencies: Sequence[str] = ("daily", "weekly", "monthly"),
    signal_col: str = "baseline_score",
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    manifest = load_pit_manifest(root)
    dataset = manifest.get("dataset", {})
    effective_start = start_date or dataset.get("date_min")
    effective_end = end_date or dataset.get("date_max")
    run_id = f"backtest_protocol_upgrade_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_panel = load_tradeable_panel(root, start_date=effective_start, end_date=effective_end)
    factor_panel = build_factor_label_panel(raw_panel, horizons=(horizon,))
    factor_columns = default_factor_columns()
    factor_panel = add_cross_sectional_zscores(factor_panel, factor_columns)
    factor_panel = add_baseline_score(factor_panel)
    label_col = f"fwd_ret_{horizon}d"

    rows: list[dict[str, Any]] = []
    for frequency in rebalance_frequencies:
        for top_n in top_n_values:
            for fee_bps in fee_bps_values:
                backtest = top_n_rebalance_backtest(
                    factor_panel,
                    signal_col,
                    label_col,
                    top_n=int(top_n),
                    fee_bps=float(fee_bps),
                    rebalance_frequency=frequency,
                )
                row = {"horizon": int(horizon), "signal": signal_col}
                row.update(backtest.summary)
                rows.append(row)

    summary_table = pd.DataFrame(rows)
    result = {
        "run_id": run_id,
        "snapshot_id": manifest.get("snapshot_id"),
        "snapshot_path": manifest.get("snapshot_path"),
        "start_date": effective_start,
        "end_date": effective_end,
        "horizon": int(horizon),
        "signal": signal_col,
        "top_n_values": [int(value) for value in top_n_values],
        "fee_bps_values": [float(value) for value in fee_bps_values],
        "rebalance_frequencies": list(rebalance_frequencies),
        "panel": panel_summary(factor_panel),
        "best_configs": best_backtest_configs(summary_table),
        "output_dir": str(run_dir),
    }

    summary_table.to_csv(run_dir / "backtest_protocol_summary.csv", index=False, encoding="utf-8-sig")
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_backtest_protocol_markdown(result, summary_table)
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        research_log_path.write_text(markdown, encoding="utf-8")
    return result


def best_backtest_configs(summary_table: pd.DataFrame, *, top: int = 10) -> list[dict[str, Any]]:
    if summary_table.empty:
        return []
    return summary_table.sort_values(["sharpe", "annualized_return"], ascending=[False, False]).head(top).to_dict("records")


def render_backtest_protocol_markdown(result: dict[str, Any], summary_table: pd.DataFrame) -> str:
    lines = [
        "# 2026-06-02 Backtest Protocol Upgrade",
        "",
        f"- Data Scope: snapshot `{result.get('snapshot_id')}`, `{result.get('start_date')}` to `{result.get('end_date')}`.",
        f"- Panel: `{result.get('panel', {}).get('rows')}` rows, `{result.get('panel', {}).get('date_count')}` dates, `{result.get('panel', {}).get('security_count')}` securities.",
        f"- Signal: `{result.get('signal')}`; horizon: `{result.get('horizon')}`.",
        f"- Grid: Top-N `{result.get('top_n_values')}`, fees `{result.get('fee_bps_values')}`, frequencies `{result.get('rebalance_frequencies')}`.",
        "- Method: signal date ranks form equal-weight Top-N portfolios; forward label is next-open-to-horizon-close; turnover cost is charged on rebalance weight changes.",
        "- Assessment: protocol comparison is still a baseline; it does not yet model limit-up/down execution or overlapping multi-day holdings.",
        "- Next Step: combine this with the factor stability report and decide whether to enter multi-factor/ML after excess-return labels and execution constraints are added.",
        "",
        "## Summary",
        "",
        _markdown_table(summary_table),
        "",
        f"Artifacts: `{result.get('output_dir')}`",
        "",
    ]
    return "\n".join(lines)


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
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--top-n", default="50,100,200", help="Comma-separated Top-N portfolio sizes.")
    parser.add_argument("--fee-bps", default="0,10,20", help="Comma-separated fee bps values.")
    parser.add_argument("--frequencies", default="daily,weekly,monthly", help="Comma-separated rebalance frequencies.")
    parser.add_argument("--signal-col", default="baseline_score")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top_n_values = tuple(int(value.strip()) for value in args.top_n.split(",") if value.strip())
    fee_bps_values = tuple(float(value.strip()) for value in args.fee_bps.split(",") if value.strip())
    frequencies = tuple(value.strip() for value in args.frequencies.split(",") if value.strip())
    result = run_backtest_protocol_upgrade(
        root=args.root,
        start_date=args.start_date,
        end_date=args.end_date,
        horizon=args.horizon,
        top_n_values=top_n_values,
        fee_bps_values=fee_bps_values,
        rebalance_frequencies=frequencies,
        signal_col=args.signal_col,
        output_dir=args.output_dir,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
