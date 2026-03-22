from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two advanced ML run outputs and attribute performance differences.")
    parser.add_argument("--base-run", required=True, help="Path to the baseline run output directory")
    parser.add_argument("--candidate-run", required=True, help="Path to the candidate run output directory")
    parser.add_argument("--output-dir", default="", help="Optional output directory for attribution results")
    parser.add_argument("--base-label", default="base")
    parser.add_argument("--candidate-label", default="candidate")
    return parser.parse_args()


def _load_run(run_dir: Path) -> Dict[str, object]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    equity = pd.read_csv(run_dir / "equity_curve.csv", parse_dates=["date"])
    regime = pd.read_csv(run_dir / "regime_state.csv", parse_dates=["Date"])
    actions = pd.read_csv(run_dir / "actions.csv", parse_dates=["date"])
    return {
        "metrics": metrics,
        "equity": equity,
        "regime": regime,
        "actions": actions,
    }


def _compound_return(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return float((1.0 + valid).prod() - 1.0)


def _build_quarterly(equity: pd.DataFrame, regime: pd.DataFrame, label: str) -> pd.DataFrame:
    df = equity.copy()
    df["date"] = pd.to_datetime(df["date"])
    regime_df = regime.rename(columns={"Date": "date"}).copy()
    df = df.merge(regime_df[["date", "quadrant", "regime_on"]], on="date", how="left", suffixes=("", "_regime"))
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    rows = []
    for quarter, g in df.groupby("quarter", sort=True):
        rows.append(
            {
                "quarter": quarter,
                f"{label}_portfolio_return": _compound_return(g["portfolio_return"]),
                f"{label}_benchmark_return": _compound_return(g["benchmark_return"]),
                f"{label}_excess_return": _compound_return(g["excess_return"]),
                f"{label}_avg_holding_count": float(g["holding_count"].mean()),
                f"{label}_avg_turnover": float(g["turnover"].mean()),
                f"{label}_regime_active_ratio": float(g["regime_on"].mean()),
                f"{label}_days": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def _build_quadrant(equity: pd.DataFrame, regime: pd.DataFrame, label: str) -> pd.DataFrame:
    df = equity.copy()
    df["date"] = pd.to_datetime(df["date"])
    regime_df = regime.rename(columns={"Date": "date"}).copy()
    df = df.merge(regime_df[["date", "quadrant", "regime_on"]], on="date", how="left", suffixes=("", "_regime"))
    rows = []
    for quadrant, g in df.groupby("quadrant", sort=True):
        rows.append(
            {
                "quadrant": quadrant,
                f"{label}_portfolio_return": _compound_return(g["portfolio_return"]),
                f"{label}_benchmark_return": _compound_return(g["benchmark_return"]),
                f"{label}_excess_return": _compound_return(g["excess_return"]),
                f"{label}_avg_holding_count": float(g["holding_count"].mean()),
                f"{label}_avg_turnover": float(g["turnover"].mean()),
                f"{label}_regime_active_ratio": float(g["regime_on"].mean()),
                f"{label}_days": int(len(g)),
            }
        )
    return pd.DataFrame(rows).sort_values("quadrant").reset_index(drop=True)


def _build_action_summary(actions: pd.DataFrame, label: str) -> pd.DataFrame:
    if actions.empty:
        return pd.DataFrame(columns=["group", "name", f"{label}_count", f"{label}_avg_score"])

    action_summary = (
        actions.groupby("action")
        .agg(count=("action", "size"), avg_score=("score", "mean"))
        .reset_index()
        .rename(columns={"action": "name", "count": f"{label}_count", "avg_score": f"{label}_avg_score"})
    )
    action_summary.insert(0, "group", "action")

    reason_summary = (
        actions.groupby("reason")
        .agg(count=("reason", "size"), avg_score=("score", "mean"))
        .reset_index()
        .rename(columns={"reason": "name", "count": f"{label}_count", "avg_score": f"{label}_avg_score"})
    )
    reason_summary.insert(0, "group", "reason")
    return pd.concat([action_summary, reason_summary], axis=0, ignore_index=True)


def _build_metric_compare(base_metrics: dict, cand_metrics: dict, base_label: str, cand_label: str) -> pd.DataFrame:
    keys = [
        "total_return",
        "annual_return",
        "sharpe",
        "max_drawdown",
        "excess_total_return",
        "excess_annual_return",
        "excess_sharpe",
        "excess_max_drawdown",
        "avg_holding_count",
        "avg_turnover",
        "hit_rate",
        "regime_active_ratio",
    ]
    rows = []
    for key in keys:
        base_val = base_metrics.get(key, np.nan)
        cand_val = cand_metrics.get(key, np.nan)
        rows.append(
            {
                "metric": key,
                base_label: base_val,
                cand_label: cand_val,
                "candidate_minus_base": float(cand_val - base_val) if pd.notna(base_val) and pd.notna(cand_val) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _build_summary_text(
    base_metrics: dict,
    cand_metrics: dict,
    quarterly_compare: pd.DataFrame,
    quadrant_compare: pd.DataFrame,
    base_label: str,
    cand_label: str,
) -> str:
    def pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    def num(value: float) -> str:
        return f"{value:.3f}"

    lines = []
    lines.append("Advanced ML 全A归因分析")
    lines.append("=" * 32)
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"对比对象: {base_label} vs {cand_label}")
    lines.append("")
    lines.append("一、总览")
    lines.append(f"- {base_label}: 累计收益 {pct(base_metrics['total_return'])} | 超额收益 {pct(base_metrics['excess_total_return'])} | 超额Sharpe {num(base_metrics['excess_sharpe'])}")
    lines.append(f"- {cand_label}: 累计收益 {pct(cand_metrics['total_return'])} | 超额收益 {pct(cand_metrics['excess_total_return'])} | 超额Sharpe {num(cand_metrics['excess_sharpe'])}")
    lines.append("")

    if not quarterly_compare.empty:
        q = quarterly_compare.copy()
        q["candidate_win"] = q[f"{cand_label}_excess_return"] > q[f"{base_label}_excess_return"]
        win_count = int(q["candidate_win"].sum())
        lines.append("二、季度对比")
        lines.append(f"- {cand_label} 超额胜出的季度数: {win_count} / {len(q)}")
        best_row = q.sort_values("candidate_minus_base_excess", ascending=False).iloc[0]
        worst_row = q.sort_values("candidate_minus_base_excess", ascending=True).iloc[0]
        lines.append(f"- {cand_label} 相对 {base_label} 最强季度: {best_row['quarter']} | 超额差 {pct(best_row['candidate_minus_base_excess'])}")
        lines.append(f"- {cand_label} 相对 {base_label} 最弱季度: {worst_row['quarter']} | 超额差 {pct(worst_row['candidate_minus_base_excess'])}")
        lines.append("")

    if not quadrant_compare.empty:
        qd = quadrant_compare.copy()
        lines.append("三、市场状态对比")
        for _, row in qd.iterrows():
            lines.append(
                f"- {row['quadrant']}: {base_label}超额 {pct(row[f'{base_label}_excess_return'])} | {cand_label}超额 {pct(row[f'{cand_label}_excess_return'])} | 差值 {pct(row['candidate_minus_base_excess'])}"
            )
        lines.append("")

    lines.append("四、初步判断")
    if cand_metrics["excess_sharpe"] > base_metrics["excess_sharpe"]:
        lines.append(f"- {cand_label} 的全A超额Sharpe更高，值得继续作为主线候选。")
    else:
        lines.append(f"- {base_label} 的全A超额Sharpe更高，当前仍应作为默认主线。")
    if cand_metrics["avg_turnover"] > base_metrics["avg_turnover"]:
        lines.append(f"- {cand_label} 的平均换手更高，说明它更依赖频繁调仓。")
    else:
        lines.append(f"- {cand_label} 的平均换手不高于 {base_label}，差异更多来自选股质量。")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    base_dir = Path(args.base_run)
    cand_dir = Path(args.candidate_run)

    base = _load_run(base_dir)
    cand = _load_run(cand_dir)

    metric_compare = _build_metric_compare(base["metrics"], cand["metrics"], args.base_label, args.candidate_label)

    quarterly_base = _build_quarterly(base["equity"], base["regime"], args.base_label)
    quarterly_cand = _build_quarterly(cand["equity"], cand["regime"], args.candidate_label)
    quarterly_compare = quarterly_base.merge(quarterly_cand, on="quarter", how="outer")
    quarterly_compare["candidate_minus_base_excess"] = (
        quarterly_compare[f"{args.candidate_label}_excess_return"] - quarterly_compare[f"{args.base_label}_excess_return"]
    )

    quadrant_base = _build_quadrant(base["equity"], base["regime"], args.base_label)
    quadrant_cand = _build_quadrant(cand["equity"], cand["regime"], args.candidate_label)
    quadrant_compare = quadrant_base.merge(quadrant_cand, on="quadrant", how="outer")
    quadrant_compare["candidate_minus_base_excess"] = (
        quadrant_compare[f"{args.candidate_label}_excess_return"] - quadrant_compare[f"{args.base_label}_excess_return"]
    )

    action_base = _build_action_summary(base["actions"], args.base_label)
    action_cand = _build_action_summary(cand["actions"], args.candidate_label)
    action_compare = action_base.merge(action_cand, on=["group", "name"], how="outer")
    action_compare[f"{args.base_label}_count"] = action_compare[f"{args.base_label}_count"].fillna(0).astype(int)
    action_compare[f"{args.candidate_label}_count"] = action_compare[f"{args.candidate_label}_count"].fillna(0).astype(int)
    action_compare["candidate_minus_base_count"] = (
        action_compare[f"{args.candidate_label}_count"] - action_compare[f"{args.base_label}_count"]
    )

    output_dir = Path(args.output_dir) if args.output_dir else Path("daily_research/output") / f"advanced_ml_attr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_compare.to_csv(output_dir / "metric_compare.csv", index=False, encoding="utf-8-sig")
    quarterly_compare.to_csv(output_dir / "quarterly_compare.csv", index=False, encoding="utf-8-sig")
    quadrant_compare.to_csv(output_dir / "quadrant_compare.csv", index=False, encoding="utf-8-sig")
    action_compare.to_csv(output_dir / "action_compare.csv", index=False, encoding="utf-8-sig")

    summary_text = _build_summary_text(
        base["metrics"],
        cand["metrics"],
        quarterly_compare,
        quadrant_compare,
        args.base_label,
        args.candidate_label,
    )
    (output_dir / "summary.txt").write_text(summary_text, encoding="utf-8")

    print(f"输出目录: {output_dir}")
    print(summary_text)


if __name__ == "__main__":
    main()
