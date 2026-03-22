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
        description="Build a dedicated ma48 stability and quarterly concentration report."
    )
    parser.add_argument("--anchor-run", required=True, help="Reference stable run, usually ma50 baseline")
    parser.add_argument("--candidate-run", required=True, help="Primary ma48 candidate run")
    parser.add_argument("--variant-run", required=True, help="Optional ma48 light-risk variant run")
    parser.add_argument("--left-neighbor-run", required=True, help="Left boundary neighbor, usually ma47 baseline")
    parser.add_argument("--right-neighbor-run", required=True, help="Right boundary neighbor, usually ma49 baseline")
    parser.add_argument("--anchor-label", default="ma50_baseline")
    parser.add_argument("--candidate-label", default="ma48_baseline")
    parser.add_argument("--variant-label", default="ma48_take20")
    parser.add_argument("--left-neighbor-label", default="ma47_baseline")
    parser.add_argument("--right-neighbor-label", default="ma49_baseline")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output dir. Defaults to daily_research/output/ma48_stability_<timestamp>",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_num(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{float(value):.3f}"


def _safe_pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{float(value):.2%}"


def _compound_return(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return float("nan")
    return float((1.0 + valid).prod() - 1.0)


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


def _summary_rows(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        metrics = run["metrics"]
        recent_full = run["recent_full"]
        latest_weak = run["latest_weak"]
        rows.append(
            {
                "label": run["label"],
                "regime_ma_window": metrics.get("regime_ma_window"),
                "stop_loss": metrics.get("stop_loss"),
                "take_profit": metrics.get("take_profit"),
                "full_excess_total_return": metrics.get("excess_total_return"),
                "full_excess_sharpe": metrics.get("excess_sharpe"),
                "full_excess_max_drawdown": metrics.get("excess_max_drawdown"),
                "full_avg_turnover": metrics.get("avg_turnover"),
                "full_avg_holding_count": metrics.get("avg_holding_count"),
                "recent_full_excess_total_return": recent_full.get("excess_total_return"),
                "recent_full_excess_sharpe": recent_full.get("excess_sharpe"),
                "recent_full_excess_max_drawdown": recent_full.get("excess_max_drawdown"),
                "latest_weak_excess_total_return": latest_weak.get("excess_total_return"),
                "latest_weak_excess_sharpe": latest_weak.get("excess_sharpe"),
                "latest_weak_excess_max_drawdown": latest_weak.get("excess_max_drawdown"),
            }
        )
    return pd.DataFrame(rows)


def _quarterly_frame(run: dict[str, Any]) -> pd.DataFrame:
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


def _concentration_summary(candidate: pd.DataFrame, other: pd.DataFrame, other_label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = candidate.merge(other, on="quarter", how="outer", suffixes=("_candidate", "_other")).sort_values("quarter").reset_index(drop=True)
    merged["candidate_minus_other_excess_return"] = merged["excess_return_candidate"] - merged["excess_return_other"]
    valid = merged["candidate_minus_other_excess_return"].dropna()
    pos = valid[valid > 0].sort_values(ascending=False)
    neg = valid[valid < 0]
    positive_sum = float(pos.sum()) if not pos.empty else 0.0
    top1 = float(pos.iloc[:1].sum()) if len(pos) >= 1 else np.nan
    top2 = float(pos.iloc[:2].sum()) if len(pos) >= 2 else top1
    top3 = float(pos.iloc[:3].sum()) if len(pos) >= 3 else top2
    shares = pos / positive_sum if positive_sum > 0 else pd.Series(dtype=float)
    hhi = float((shares ** 2).sum()) if not shares.empty else np.nan
    best_row = merged.loc[merged["candidate_minus_other_excess_return"].idxmax()] if not merged.empty else None
    worst_row = merged.loc[merged["candidate_minus_other_excess_return"].idxmin()] if not merged.empty else None
    summary = {
        "other_label": other_label,
        "quarters_total": int(valid.shape[0]),
        "quarters_candidate_better": int((valid > 0).sum()),
        "quarters_candidate_worse": int((valid < 0).sum()),
        "candidate_minus_other_total_quarter_edge": float(valid.sum()) if not valid.empty else np.nan,
        "positive_quarter_edge_sum": positive_sum if positive_sum > 0 else np.nan,
        "best_quarter": None if best_row is None else str(best_row["quarter"]),
        "best_quarter_edge": None if best_row is None else float(best_row["candidate_minus_other_excess_return"]),
        "best_quarter_positive_share": float(top1 / positive_sum) if positive_sum > 0 and np.isfinite(top1) else np.nan,
        "top2_positive_share": float(top2 / positive_sum) if positive_sum > 0 and np.isfinite(top2) else np.nan,
        "top3_positive_share": float(top3 / positive_sum) if positive_sum > 0 and np.isfinite(top3) else np.nan,
        "positive_quarter_hhi": hhi,
        "worst_quarter": None if worst_row is None else str(worst_row["quarter"]),
        "worst_quarter_edge": None if worst_row is None else float(worst_row["candidate_minus_other_excess_return"]),
        "quarters_with_positive_edge": int(pos.shape[0]),
        "quarters_with_negative_edge": int(neg.shape[0]),
    }
    return merged, summary


def _write_markdown(
    output_path: Path,
    summary_df: pd.DataFrame,
    candidate_vs_anchor: dict[str, Any],
    candidate_vs_left: dict[str, Any],
    candidate_vs_right: dict[str, Any],
    variant_vs_candidate: dict[str, Any],
) -> None:
    rows = {row["label"]: row for row in summary_df.to_dict(orient="records")}
    candidate = rows["ma48_baseline"] if "ma48_baseline" in rows else rows[next(iter(rows))]
    anchor = rows.get("ma50_baseline")
    variant = rows.get("ma48_take20")
    left = rows.get("ma47_baseline")
    right = rows.get("ma49_baseline")

    lines = [
        "# ma48 稳定性复验与季度集中度诊断",
        "",
        "## 核心判断",
        (
            f"- `ma48_baseline` 全样本超额 Sharpe {_safe_num(candidate['full_excess_sharpe'])}"
            f"，相对 `ma50_baseline` {_safe_num(anchor['full_excess_sharpe']) if anchor else 'nan'}"
            f"，最近弱窗口 {_safe_pct(candidate['latest_weak_excess_total_return'])} / {_safe_num(candidate['latest_weak_excess_sharpe'])}"
        ),
        (
            f"- `ma48_take20` 全样本超额 Sharpe {_safe_num(variant['full_excess_sharpe']) if variant else 'nan'}"
            f"，相对 `ma48_baseline` 的附加增益目前仍偏集中"
        ),
        "",
        "## 邻域稳定性",
        (
            f"- 左邻 `ma47_baseline`: 全样本超额 Sharpe {_safe_num(left['full_excess_sharpe']) if left else 'nan'}"
            f"，最新弱窗口 {_safe_pct(left['latest_weak_excess_total_return']) if left else 'nan'} / {_safe_num(left['latest_weak_excess_sharpe']) if left else 'nan'}"
        ),
        (
            f"- 右邻 `ma49_baseline`: 全样本超额 Sharpe {_safe_num(right['full_excess_sharpe']) if right else 'nan'}"
            f"，最新弱窗口 {_safe_pct(right['latest_weak_excess_total_return']) if right else 'nan'} / {_safe_num(right['latest_weak_excess_sharpe']) if right else 'nan'}"
        ),
        "",
        "## 季度集中度",
        (
            f"- `ma48_baseline` 相对 `ma50_baseline`: {candidate_vs_anchor['quarters_candidate_better']}/{candidate_vs_anchor['quarters_total']} 个季度更强，"
            f"最佳季度占正向季度总优势 {_safe_pct(candidate_vs_anchor['best_quarter_positive_share'])}，"
            f"Top3 季度占比 {_safe_pct(candidate_vs_anchor['top3_positive_share'])}，"
            f"HHI {_safe_num(candidate_vs_anchor['positive_quarter_hhi'])}"
        ),
        (
            f"- `ma48_baseline` 相对 `ma47_baseline`: {candidate_vs_left['quarters_candidate_better']}/{candidate_vs_left['quarters_total']} 个季度更强，"
            f"最佳季度占比 {_safe_pct(candidate_vs_left['best_quarter_positive_share'])}，"
            f"Top3 季度占比 {_safe_pct(candidate_vs_left['top3_positive_share'])}"
        ),
        (
            f"- `ma48_baseline` 相对 `ma49_baseline`: {candidate_vs_right['quarters_candidate_better']}/{candidate_vs_right['quarters_total']} 个季度更强，"
            f"最佳季度占比 {_safe_pct(candidate_vs_right['best_quarter_positive_share'])}，"
            f"Top3 季度占比 {_safe_pct(candidate_vs_right['top3_positive_share'])}"
        ),
        (
            f"- `ma48_take20` 相对 `ma48_baseline`: {variant_vs_candidate['quarters_candidate_better']}/{variant_vs_candidate['quarters_total']} 个季度更强，"
            f"最佳季度占比 {_safe_pct(variant_vs_candidate['best_quarter_positive_share'])}，"
            f"Top3 季度占比 {_safe_pct(variant_vs_candidate['top3_positive_share'])}"
        ),
        "",
        "## 解释口径",
        "- 若最佳季度占比和 Top3 季度占比过高，说明优势更像是少数季度驱动，而不是平滑稳定增益。",
        "- 若 ma48 相对 ma47 和 ma49 都更强，说明它至少不是纯随机噪音点；但若季度集中度很高，仍不能直接晋级执行端。",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path("daily_research/output") / f"ma48_stability_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        _load_run(Path(args.anchor_run), args.anchor_label),
        _load_run(Path(args.candidate_run), args.candidate_label),
        _load_run(Path(args.variant_run), args.variant_label),
        _load_run(Path(args.left_neighbor_run), args.left_neighbor_label),
        _load_run(Path(args.right_neighbor_run), args.right_neighbor_label),
    ]

    summary_df = _summary_rows(runs)
    summary_df.to_csv(output_dir / "stability_summary.csv", index=False, encoding="utf-8-sig")

    quarterly_frames = [_quarterly_frame(run) for run in runs]
    quarterly_df = pd.concat(quarterly_frames, axis=0, ignore_index=True)
    quarterly_df.to_csv(output_dir / "quarterly_metrics.csv", index=False, encoding="utf-8-sig")

    def pick(label: str) -> pd.DataFrame:
        return quarterly_df.loc[quarterly_df["label"] == label, ["quarter", "excess_return", "avg_turnover", "avg_holding_count", "regime_active_ratio", "dominant_quadrant"]]

    candidate_vs_anchor_df, candidate_vs_anchor = _concentration_summary(pick(args.candidate_label), pick(args.anchor_label), args.anchor_label)
    candidate_vs_left_df, candidate_vs_left = _concentration_summary(pick(args.candidate_label), pick(args.left_neighbor_label), args.left_neighbor_label)
    candidate_vs_right_df, candidate_vs_right = _concentration_summary(pick(args.candidate_label), pick(args.right_neighbor_label), args.right_neighbor_label)
    variant_vs_candidate_df, variant_vs_candidate = _concentration_summary(pick(args.variant_label), pick(args.candidate_label), args.candidate_label)

    candidate_vs_anchor_df.to_csv(output_dir / "candidate_vs_anchor_quarterly.csv", index=False, encoding="utf-8-sig")
    candidate_vs_left_df.to_csv(output_dir / "candidate_vs_left_neighbor_quarterly.csv", index=False, encoding="utf-8-sig")
    candidate_vs_right_df.to_csv(output_dir / "candidate_vs_right_neighbor_quarterly.csv", index=False, encoding="utf-8-sig")
    variant_vs_candidate_df.to_csv(output_dir / "variant_vs_candidate_quarterly.csv", index=False, encoding="utf-8-sig")

    report = {
        "anchor_label": args.anchor_label,
        "candidate_label": args.candidate_label,
        "variant_label": args.variant_label,
        "left_neighbor_label": args.left_neighbor_label,
        "right_neighbor_label": args.right_neighbor_label,
        "summary_rows": summary_df.to_dict(orient="records"),
        "candidate_vs_anchor": candidate_vs_anchor,
        "candidate_vs_left_neighbor": candidate_vs_left,
        "candidate_vs_right_neighbor": candidate_vs_right,
        "variant_vs_candidate": variant_vs_candidate,
    }
    (output_dir / "stability_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(
        output_dir / "stability_report.md",
        summary_df,
        candidate_vs_anchor,
        candidate_vs_left,
        candidate_vs_right,
        variant_vs_candidate,
    )

    print(f"output: {output_dir}")
    print(summary_df.to_string(index=False))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
