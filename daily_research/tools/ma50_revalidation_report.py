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
        description="Build a focused revalidation report for ma50 execution-repair candidates."
    )
    parser.add_argument("--current-baseline-run", required=True, help="Run dir for current execution baseline")
    parser.add_argument("--backup-run", required=True, help="Run dir for second-tier backup candidate")
    parser.add_argument("--candidate-run", required=True, help="Run dir for ma50 baseline candidate")
    parser.add_argument("--candidate-label", default="ma50_baseline")
    parser.add_argument("--current-label", default="ma60_baseline")
    parser.add_argument("--backup-label", default="ma60_up_low_ml55_none25_v220")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional report output dir. Defaults to daily_research/output/ma50_revalidation_<timestamp>",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compound_return(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return float("nan")
    return float((1.0 + valid).prod() - 1.0)


def _safe_pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{float(value):.2%}"


def _safe_num(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{float(value):.3f}"


def _load_run(run_dir: Path, label: str) -> dict[str, Any]:
    metrics = _load_json(run_dir / "metrics.json")
    recent_full = _load_json(run_dir / "metrics_recent_full.json")
    latest_weak = _load_json(run_dir / "metrics_latest_weak.json")
    equity = pd.read_csv(run_dir / "equity_curve.csv", parse_dates=["date"])
    regime = pd.read_csv(run_dir / "regime_state.csv", parse_dates=["Date"])
    return {
        "label": label,
        "run_dir": run_dir,
        "metrics": metrics,
        "recent_full": recent_full,
        "latest_weak": latest_weak,
        "equity": equity,
        "regime": regime,
    }


def _build_summary_rows(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        metrics = run["metrics"]
        recent_full = run["recent_full"]
        latest_weak = run["latest_weak"]
        rows.append(
            {
                "label": run["label"],
                "regime_ma_window": metrics.get("regime_ma_window"),
                "full_excess_total_return": metrics.get("excess_total_return"),
                "full_excess_sharpe": metrics.get("excess_sharpe"),
                "full_excess_max_drawdown": metrics.get("excess_max_drawdown"),
                "full_avg_turnover": metrics.get("avg_turnover"),
                "full_avg_holding_count": metrics.get("avg_holding_count"),
                "full_regime_active_ratio": metrics.get("regime_active_ratio"),
                "recent_full_excess_total_return": recent_full.get("excess_total_return"),
                "recent_full_excess_sharpe": recent_full.get("excess_sharpe"),
                "recent_full_excess_max_drawdown": recent_full.get("excess_max_drawdown"),
                "recent_full_avg_turnover": recent_full.get("avg_turnover"),
                "recent_full_avg_holding_count": recent_full.get("avg_holding_count"),
                "recent_full_regime_active_ratio": recent_full.get("regime_active_ratio"),
                "latest_weak_excess_total_return": latest_weak.get("excess_total_return"),
                "latest_weak_excess_sharpe": latest_weak.get("excess_sharpe"),
                "latest_weak_excess_max_drawdown": latest_weak.get("excess_max_drawdown"),
                "latest_weak_avg_turnover": latest_weak.get("avg_turnover"),
                "latest_weak_avg_holding_count": latest_weak.get("avg_holding_count"),
                "latest_weak_regime_active_ratio": latest_weak.get("regime_active_ratio"),
            }
        )
    return pd.DataFrame(rows)


def _build_quarterly(run: dict[str, Any]) -> pd.DataFrame:
    equity = run["equity"].copy()
    regime = run["regime"].rename(columns={"Date": "date"}).copy()
    merged = equity.merge(regime[["date", "quadrant", "regime_on"]], on="date", how="left", suffixes=("", "_regime"))
    quadrant_col = "quadrant" if "quadrant" in merged.columns else "quadrant_regime"
    regime_on_col = "regime_on" if "regime_on" in merged.columns else "regime_on_regime"
    merged["quarter"] = merged["date"].dt.to_period("Q").astype(str)
    rows = []
    for quarter, g in merged.groupby("quarter", sort=True):
        rows.append(
            {
                "label": run["label"],
                "quarter": quarter,
                "excess_return": _compound_return(g["excess_return"]),
                "avg_turnover": float(g["turnover"].mean()),
                "avg_holding_count": float(g["holding_count"].mean()),
                "regime_active_ratio": float(g[regime_on_col].mean()),
                "dominant_quadrant": str(g[quadrant_col].mode().iloc[0]) if quadrant_col in g and not g[quadrant_col].dropna().empty else "",
                "days": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def _build_pairwise_quarter_compare(candidate: pd.DataFrame, other: pd.DataFrame, other_label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = candidate.merge(other, on="quarter", how="outer", suffixes=("_candidate", "_other")).sort_values("quarter").reset_index(drop=True)
    merged["candidate_minus_other_excess_return"] = merged["excess_return_candidate"] - merged["excess_return_other"]
    valid = merged["candidate_minus_other_excess_return"].dropna()
    positive = valid[valid > 0]
    negative = valid[valid < 0]
    best_row = merged.loc[merged["candidate_minus_other_excess_return"].idxmax()] if not merged.empty else None
    worst_row = merged.loc[merged["candidate_minus_other_excess_return"].idxmin()] if not merged.empty else None
    positive_sum = float(positive.sum()) if not positive.empty else 0.0
    best_share = float(best_row["candidate_minus_other_excess_return"] / positive_sum) if best_row is not None and positive_sum > 0 else np.nan
    summary = {
        "other_label": other_label,
        "quarters_total": int(valid.shape[0]),
        "quarters_candidate_better": int((valid > 0).sum()),
        "quarters_candidate_worse": int((valid < 0).sum()),
        "candidate_minus_other_total_quarter_edge": float(valid.sum()) if not valid.empty else np.nan,
        "best_quarter": None if best_row is None else str(best_row["quarter"]),
        "best_quarter_edge": None if best_row is None else float(best_row["candidate_minus_other_excess_return"]),
        "best_quarter_positive_edge_share": None if np.isnan(best_share) else float(best_share),
        "worst_quarter": None if worst_row is None else str(worst_row["quarter"]),
        "worst_quarter_edge": None if worst_row is None else float(worst_row["candidate_minus_other_excess_return"]),
    }
    return merged, summary


def _write_markdown(
    output_path: Path,
    summary_df: pd.DataFrame,
    candidate_label: str,
    current_label: str,
    backup_label: str,
    compare_current: dict[str, Any],
    compare_backup: dict[str, Any],
) -> None:
    rows = {row["label"]: row for row in summary_df.to_dict(orient="records")}
    candidate = rows[candidate_label]
    current = rows[current_label]
    backup = rows[backup_label]

    lines = [
        "# ma50 baseline 第三轮专项复验",
        "",
        "## 候选结论",
        f"- 当前执行默认值仍保持冻结：`advanced_ml + liquid500 + next_open`",
        f"- 头号候选：`{candidate_label}`",
        f"- 次一级备选：`{backup_label}`",
        "",
        "## 核心比较",
        (
            f"- `{candidate_label}` 相对 `{current_label}`："
            f"全样本超额 Sharpe {_safe_num(candidate['full_excess_sharpe'])} vs {_safe_num(current['full_excess_sharpe'])}，"
            f"最新弱窗口超额 Sharpe {_safe_num(candidate['latest_weak_excess_sharpe'])} vs {_safe_num(current['latest_weak_excess_sharpe'])}"
        ),
        (
            f"- `{candidate_label}` 相对 `{backup_label}`："
            f"全样本超额 Sharpe {_safe_num(candidate['full_excess_sharpe'])} vs {_safe_num(backup['full_excess_sharpe'])}，"
            f"最新弱窗口超额 Sharpe {_safe_num(candidate['latest_weak_excess_sharpe'])} vs {_safe_num(backup['latest_weak_excess_sharpe'])}"
        ),
        (
            f"- `{candidate_label}` 的全样本超额回撤 {_safe_pct(candidate['full_excess_max_drawdown'])}，"
            f"当前执行主线 {_safe_pct(current['full_excess_max_drawdown'])}，次一级备选 {_safe_pct(backup['full_excess_max_drawdown'])}"
        ),
        (
            f"- `{candidate_label}` 的全样本活跃比例 / 换手 / 持仓数："
            f"{_safe_pct(candidate['full_regime_active_ratio'])} / {_safe_num(candidate['full_avg_turnover'])} / {_safe_num(candidate['full_avg_holding_count'])}"
        ),
        "",
        "## 季度稳定性",
        (
            f"- 相对 `{current_label}`：{compare_current['quarters_candidate_better']}/{compare_current['quarters_total']} 个季度更强，"
            f"最强季度 `{compare_current['best_quarter']}`，超额差 {_safe_pct(compare_current['best_quarter_edge'])}，"
            f"若只看正向季度贡献，最强季度占比 {_safe_pct(compare_current['best_quarter_positive_edge_share'])}"
        ),
        (
            f"- 相对 `{backup_label}`：{compare_backup['quarters_candidate_better']}/{compare_backup['quarters_total']} 个季度更强，"
            f"最强季度 `{compare_backup['best_quarter']}`，超额差 {_safe_pct(compare_backup['best_quarter_edge'])}，"
            f"若只看正向季度贡献，最强季度占比 {_safe_pct(compare_backup['best_quarter_positive_edge_share'])}"
        ),
        "",
        "## 复验口径",
        "- 不只看全样本，还同时检查最近完整窗口、最新弱窗口、季度分解。",
        "- 只要出现“修弱窗口但破坏强窗口或整体表现”的情况，就不晋级执行端。",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path("daily_research/output") / f"ma50_revalidation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        _load_run(Path(args.current_baseline_run), args.current_label),
        _load_run(Path(args.backup_run), args.backup_label),
        _load_run(Path(args.candidate_run), args.candidate_label),
    ]

    summary_df = _build_summary_rows(runs)
    summary_df.to_csv(output_dir / "revalidation_summary.csv", index=False, encoding="utf-8-sig")

    quarterly_frames = [_build_quarterly(run) for run in runs]
    quarterly_df = pd.concat(quarterly_frames, axis=0, ignore_index=True)
    quarterly_df.to_csv(output_dir / "quarterly_metrics.csv", index=False, encoding="utf-8-sig")

    candidate_quarterly = quarterly_df.loc[quarterly_df["label"] == args.candidate_label, ["quarter", "excess_return", "avg_turnover", "avg_holding_count", "regime_active_ratio"]]
    current_quarterly = quarterly_df.loc[quarterly_df["label"] == args.current_label, ["quarter", "excess_return", "avg_turnover", "avg_holding_count", "regime_active_ratio"]]
    backup_quarterly = quarterly_df.loc[quarterly_df["label"] == args.backup_label, ["quarter", "excess_return", "avg_turnover", "avg_holding_count", "regime_active_ratio"]]

    compare_current_df, compare_current = _build_pairwise_quarter_compare(candidate_quarterly, current_quarterly, args.current_label)
    compare_backup_df, compare_backup = _build_pairwise_quarter_compare(candidate_quarterly, backup_quarterly, args.backup_label)
    compare_current_df.to_csv(output_dir / "quarterly_compare_vs_current.csv", index=False, encoding="utf-8-sig")
    compare_backup_df.to_csv(output_dir / "quarterly_compare_vs_backup.csv", index=False, encoding="utf-8-sig")

    report = {
        "current_label": args.current_label,
        "backup_label": args.backup_label,
        "candidate_label": args.candidate_label,
        "summary_rows": summary_df.to_dict(orient="records"),
        "candidate_vs_current": compare_current,
        "candidate_vs_backup": compare_backup,
    }
    (output_dir / "revalidation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(
        output_dir / "revalidation_report.md",
        summary_df,
        args.candidate_label,
        args.current_label,
        args.backup_label,
        compare_current,
        compare_backup,
    )

    print(f"output: {output_dir}")
    print(summary_df.to_string(index=False))
    print(json.dumps({"candidate_vs_current": compare_current, "candidate_vs_backup": compare_backup}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
