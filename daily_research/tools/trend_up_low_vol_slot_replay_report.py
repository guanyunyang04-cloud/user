from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether trend_up_low_vol slot signatures replay across quarters."
    )
    parser.add_argument("--ma47-run", required=True, help="Path to ma47 baseline run")
    parser.add_argument("--ma48-run", required=True, help="Path to ma48 baseline run")
    parser.add_argument("--ma50-run", required=True, help="Path to ma50 baseline run")
    parser.add_argument("--focus-quarter", default="2025Q3", help="Quarter to diagnose, e.g. 2025Q3")
    parser.add_argument("--focus-quadrant", default="trend_up_low_vol")
    parser.add_argument("--top-n", type=int, default=5, help="Top additions to treat as slot signature")
    parser.add_argument(
        "--min-positive-delta",
        type=float,
        default=0.001,
        help="Minimum average weight delta to count as repeated positive slot",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output dir. Defaults to daily_research/output/trend_up_low_vol_slot_replay_<timestamp>",
    )
    return parser.parse_args()


def _load_run(run_dir: Path, label: str) -> dict[str, Any]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    equity = pd.read_csv(run_dir / "equity_curve.csv", parse_dates=["date"])
    regime = pd.read_csv(run_dir / "regime_state.csv", parse_dates=["Date"]).rename(columns={"Date": "date"})
    target_weights = pd.read_csv(run_dir / "target_weights.csv", parse_dates=["Date"]).rename(columns={"Date": "date"})
    equity = equity.merge(regime[["date", "quadrant", "regime_on"]], on="date", how="left", suffixes=("", "_reg"))
    regime_on_col = "regime_on_reg" if "regime_on_reg" in equity.columns else "regime_on"
    equity["regime_on_state"] = equity[regime_on_col]
    equity["quarter"] = equity["date"].dt.to_period("Q").astype(str)
    return {
        "label": label,
        "metrics": metrics,
        "equity": equity,
        "target_weights": target_weights,
    }


def _compound_return(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return float("nan")
    return float((1.0 + valid).prod() - 1.0)


def _safe_num(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{float(value):.3f}"


def _safe_pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{float(value):.2%}"


def _pair_quarter_table(
    candidate: dict[str, Any],
    other: dict[str, Any],
    focus_quadrant: str,
    top_n: int,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    cand_equity = candidate["equity"].copy()
    other_equity = other["equity"].copy()
    merged = cand_equity[["date", "quarter", "quadrant", "regime_on_state", "excess_return"]].merge(
        other_equity[["date", "excess_return"]],
        on="date",
        how="inner",
        suffixes=("_candidate", "_other"),
    )
    merged = merged.loc[(merged["quadrant"] == focus_quadrant) & (merged["regime_on_state"] == True)].copy()
    merged["edge"] = merged["excess_return_candidate"] - merged["excess_return_other"]

    cand_weights = candidate["target_weights"].set_index("date").sort_index()
    other_weights = other["target_weights"].set_index("date").sort_index()

    quarter_rows: list[dict[str, Any]] = []
    quarter_deltas: dict[str, pd.Series] = {}
    quarter_top_sets: dict[str, set[str]] = {}
    for quarter, g in merged.groupby("quarter", sort=True):
        dates = g["date"]
        cand_avg = cand_weights.loc[cand_weights.index.isin(dates)].astype(float).mean(axis=0)
        other_avg = other_weights.loc[other_weights.index.isin(dates)].astype(float).mean(axis=0)
        delta = (cand_avg - other_avg).sort_values(ascending=False)
        quarter_deltas[quarter] = delta
        top_additions = [stock for stock, value in delta.head(top_n).items() if value > 1e-9]
        top_reductions = [stock for stock, value in delta.sort_values(ascending=True).head(top_n).items() if value < -1e-9]
        quarter_top_sets[quarter] = set(top_additions)
        quarter_rows.append(
            {
                "quarter": quarter,
                "days": int(len(g)),
                "edge_sum": float(g["edge"].sum()),
                "edge_compound_gap": _compound_return(g["excess_return_candidate"]) - _compound_return(g["excess_return_other"]),
                "positive_day_count": int((g["edge"] > 0).sum()),
                "negative_day_count": int((g["edge"] < 0).sum()),
                "top_additions": ", ".join(top_additions),
                "top_reductions": ", ".join(top_reductions),
            }
        )

    table = pd.DataFrame(quarter_rows).sort_values("quarter").reset_index(drop=True)
    return table, quarter_deltas


def _signature_replay(
    quarter_table: pd.DataFrame,
    quarter_deltas: dict[str, pd.Series],
    focus_quarter: str,
    min_positive_delta: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    focus_delta = quarter_deltas[focus_quarter]
    focus_signature = [stock for stock, value in focus_delta.head(5).items() if value > 1e-9]
    focus_signature_set = set(focus_signature)

    replay_rows: list[dict[str, Any]] = []
    for _, row in quarter_table.iterrows():
        quarter = row["quarter"]
        top_set = {stock.strip() for stock in row["top_additions"].split(",") if stock.strip()}
        union = focus_signature_set | top_set
        replay_rows.append(
            {
                **row.to_dict(),
                "focus_signature_overlap_count": int(len(focus_signature_set & top_set)),
                "focus_signature_jaccard": float(len(focus_signature_set & top_set) / len(union)) if union else 1.0,
            }
        )

    replay_df = pd.DataFrame(replay_rows)
    stock_rows: list[dict[str, Any]] = []
    for stock in focus_signature:
        positive_quarters: list[str] = []
        positive_edge_quarters: list[str] = []
        for _, row in replay_df.iterrows():
            quarter = row["quarter"]
            if quarter == focus_quarter:
                continue
            delta = float(quarter_deltas[quarter].get(stock, 0.0))
            if delta > min_positive_delta:
                positive_quarters.append(quarter)
                if float(row["edge_sum"]) > 0:
                    positive_edge_quarters.append(quarter)
        stock_rows.append(
            {
                "stock": stock,
                "focus_quarter_avg_weight_delta": float(focus_delta.get(stock, 0.0)),
                "repeat_positive_delta_quarters": ", ".join(positive_quarters),
                "repeat_positive_delta_quarter_count": int(len(positive_quarters)),
                "repeat_positive_delta_and_positive_edge_quarters": ", ".join(positive_edge_quarters),
                "repeat_positive_delta_and_positive_edge_count": int(len(positive_edge_quarters)),
            }
        )

    stock_df = pd.DataFrame(stock_rows)
    other_quarters = replay_df.loc[replay_df["quarter"] != focus_quarter]
    summary = {
        "focus_quarter": focus_quarter,
        "focus_signature": focus_signature,
        "other_quarters_with_any_signature_overlap": int((other_quarters["focus_signature_overlap_count"] > 0).sum()) if not other_quarters.empty else 0,
        "other_quarters_with_full_signature_replay": int((other_quarters["focus_signature_overlap_count"] == len(focus_signature_set)).sum()) if not other_quarters.empty else 0,
        "best_other_quarter_overlap": float(other_quarters["focus_signature_overlap_count"].max()) if not other_quarters.empty else 0.0,
        "best_other_quarter_jaccard": float(other_quarters["focus_signature_jaccard"].max()) if not other_quarters.empty else 0.0,
        "positive_other_quarters_with_any_signature_overlap": int(
            ((other_quarters["edge_sum"] > 0) & (other_quarters["focus_signature_overlap_count"] > 0)).sum()
        ) if not other_quarters.empty else 0,
    }
    return replay_df, stock_df, summary


def _write_markdown(
    output_path: Path,
    ma48_vs_ma47_summary: dict[str, Any],
    ma48_vs_ma50_summary: dict[str, Any],
    ma48_vs_ma47_stock_df: pd.DataFrame,
    ma48_vs_ma50_stock_df: pd.DataFrame,
) -> None:
    lines = [
        "# trend_up_low_vol 关键槽位复现诊断",
        "",
        "## 核心判断",
        (
            f"- `ma48` 相对 `ma47` 的 `{ma48_vs_ma47_summary['focus_quarter']}` top5 槽位签名为 "
            f"{', '.join(ma48_vs_ma47_summary['focus_signature'])}"
        ),
        (
            f"- `ma48` 相对 `ma50` 的 `{ma48_vs_ma50_summary['focus_quarter']}` top5 槽位签名为 "
            f"{', '.join(ma48_vs_ma50_summary['focus_signature'])}"
        ),
        (
            f"- 这两组签名在其他季度里的完整复现次数都为 "
            f"{ma48_vs_ma47_summary['other_quarters_with_full_signature_replay']} / {ma48_vs_ma50_summary['other_quarters_with_full_signature_replay']}"
        ),
        (
            f"- 即使放宽到“任意重叠”，相对 `ma47` 也只有 {ma48_vs_ma47_summary['other_quarters_with_any_signature_overlap']} 个其他季度出现过，"
            f"相对 `ma50` 只有 {ma48_vs_ma50_summary['other_quarters_with_any_signature_overlap']} 个其他季度出现过"
        ),
        "",
        "## 解释",
        "- 这说明：`2025Q3` 的关键槽位更像季度特定命中，而不是已经跨季度稳定复放的固定签名。",
        "- 如果后续还要继续推进 `ma47/48` 左侧带，重点应从“继续扫边界”切到“把这些槽位替换抽象成更稳定的信号逻辑”。",
        "",
        "## Q3 槽位名录",
        "- 相对 `ma47`：",
    ]
    for row in ma48_vs_ma47_stock_df.itertuples():
        lines.append(
            f"  - {row.stock}: Q3 平均权重差 {_safe_pct(row.focus_quarter_avg_weight_delta)}"
            f"，其他季度重复出现 {row.repeat_positive_delta_quarter_count} 次"
        )
    lines.append("- 相对 `ma50`：")
    for row in ma48_vs_ma50_stock_df.itertuples():
        lines.append(
            f"  - {row.stock}: Q3 平均权重差 {_safe_pct(row.focus_quarter_avg_weight_delta)}"
            f"，其他季度重复出现 {row.repeat_positive_delta_quarter_count} 次"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path("daily_research/output") / (
        f"trend_up_low_vol_slot_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ma47 = _load_run(Path(args.ma47_run), "ma47_baseline")
    ma48 = _load_run(Path(args.ma48_run), "ma48_baseline")
    ma50 = _load_run(Path(args.ma50_run), "ma50_baseline")

    ma48_vs_ma47_quarterly, ma48_vs_ma47_deltas = _pair_quarter_table(ma48, ma47, args.focus_quadrant, args.top_n)
    ma48_vs_ma50_quarterly, ma48_vs_ma50_deltas = _pair_quarter_table(ma48, ma50, args.focus_quadrant, args.top_n)

    ma48_vs_ma47_replay, ma48_vs_ma47_stock, ma48_vs_ma47_summary = _signature_replay(
        ma48_vs_ma47_quarterly,
        ma48_vs_ma47_deltas,
        args.focus_quarter,
        args.min_positive_delta,
    )
    ma48_vs_ma50_replay, ma48_vs_ma50_stock, ma48_vs_ma50_summary = _signature_replay(
        ma48_vs_ma50_quarterly,
        ma48_vs_ma50_deltas,
        args.focus_quarter,
        args.min_positive_delta,
    )

    ma48_vs_ma47_replay.to_csv(output_dir / "ma48_vs_ma47_slot_replay.csv", index=False, encoding="utf-8-sig")
    ma48_vs_ma50_replay.to_csv(output_dir / "ma48_vs_ma50_slot_replay.csv", index=False, encoding="utf-8-sig")
    ma48_vs_ma47_stock.to_csv(output_dir / "ma48_vs_ma47_focus_signature_stocks.csv", index=False, encoding="utf-8-sig")
    ma48_vs_ma50_stock.to_csv(output_dir / "ma48_vs_ma50_focus_signature_stocks.csv", index=False, encoding="utf-8-sig")

    report = {
        "focus_quarter": args.focus_quarter,
        "focus_quadrant": args.focus_quadrant,
        "ma48_vs_ma47": ma48_vs_ma47_summary,
        "ma48_vs_ma50": ma48_vs_ma50_summary,
    }
    (output_dir / "slot_replay_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(
        output_dir / "slot_replay_report.md",
        ma48_vs_ma47_summary,
        ma48_vs_ma50_summary,
        ma48_vs_ma47_stock,
        ma48_vs_ma50_stock,
    )

    print(f"output: {output_dir}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
