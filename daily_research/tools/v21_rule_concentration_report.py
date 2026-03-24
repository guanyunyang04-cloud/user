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
        description="Diagnose quarterly concentration and focus-quarter attribution for a v2.1 rule candidate."
    )
    parser.add_argument("--baseline-run", required=True, help="Path to baseline v2 run directory")
    parser.add_argument("--candidate-run", required=True, help="Path to v2.1 candidate run directory")
    parser.add_argument("--baseline-label", default="v2")
    parser.add_argument("--candidate-label", default="v21_volume_contraction_015")
    parser.add_argument("--focus-quarter", default="auto", help="Quarter to diagnose, or 'auto' to use the best positive-edge quarter")
    parser.add_argument("--top-days", type=int, default=10)
    parser.add_argument("--top-weight-delta", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output dir. Defaults to daily_research/output/v21_rule_concentration_<timestamp>",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run(run_dir: Path, label: str) -> dict[str, Any]:
    metrics = _load_json(run_dir / "metrics.json")
    recent_full = _load_json(run_dir / "metrics_recent_full.json")
    latest_weak = _load_json(run_dir / "metrics_latest_weak.json")
    equity = pd.read_csv(run_dir / "equity_curve.csv", parse_dates=["date"])
    regime = pd.read_csv(run_dir / "regime_state.csv", parse_dates=["Date"]).rename(columns={"Date": "date"})
    target_weights = pd.read_csv(run_dir / "target_weights.csv", parse_dates=["Date"]).rename(columns={"Date": "date"})
    rankic_quarterly_path = run_dir / "up_low_rankic_quarterly.csv"
    rankic_quarterly = pd.read_csv(rankic_quarterly_path) if rankic_quarterly_path.exists() else pd.DataFrame()
    equity = equity.merge(regime[["date", "quadrant", "regime_on"]], on="date", how="left", suffixes=("", "_regime"))
    if "regime_on" not in equity.columns and "regime_on_regime" in equity.columns:
        equity["regime_on"] = equity["regime_on_regime"]
    if "quadrant" not in equity.columns and "quadrant_regime" in equity.columns:
        equity["quadrant"] = equity["quadrant_regime"]
    return {
        "label": label,
        "run_dir": run_dir,
        "metrics": metrics,
        "recent_full": recent_full,
        "latest_weak": latest_weak,
        "equity": equity,
        "target_weights": target_weights,
        "rankic_quarterly": rankic_quarterly,
    }


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


def _summary_rows(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        metrics = run["metrics"]
        recent_full = run["recent_full"]
        latest_weak = run["latest_weak"]
        rows.append(
            {
                "label": run["label"],
                "full_excess_total_return": metrics.get("excess_total_return"),
                "full_excess_sharpe": metrics.get("excess_sharpe"),
                "full_excess_max_drawdown": metrics.get("excess_max_drawdown"),
                "full_avg_turnover": metrics.get("avg_turnover"),
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
    equity["quarter"] = equity["date"].dt.to_period("Q").astype(str)
    rows: list[dict[str, Any]] = []
    for quarter, g in equity.groupby("quarter", sort=True):
        quadrant_mode = g["quadrant"].mode()
        rows.append(
            {
                "label": run["label"],
                "quarter": quarter,
                "excess_return": _compound_return(g["excess_return"]),
                "avg_turnover": float(g["turnover"].mean()),
                "avg_holding_count": float(g["holding_count"].mean()),
                "regime_active_ratio": float(g["regime_on"].mean()),
                "dominant_quadrant": str(quadrant_mode.iloc[0]) if not quadrant_mode.empty else "",
                "days": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def _concentration_summary(candidate: pd.DataFrame, baseline: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = candidate.merge(baseline, on="quarter", how="outer", suffixes=("_candidate", "_baseline"))
    merged = merged.sort_values("quarter").reset_index(drop=True)
    merged["candidate_minus_baseline_excess_return"] = (
        merged["excess_return_candidate"] - merged["excess_return_baseline"]
    )
    valid = merged["candidate_minus_baseline_excess_return"].dropna()
    pos = valid[valid > 0].sort_values(ascending=False)
    neg = valid[valid < 0]
    positive_sum = float(pos.sum()) if not pos.empty else 0.0
    top1 = float(pos.iloc[:1].sum()) if len(pos) >= 1 else np.nan
    top2 = float(pos.iloc[:2].sum()) if len(pos) >= 2 else top1
    top3 = float(pos.iloc[:3].sum()) if len(pos) >= 3 else top2
    shares = pos / positive_sum if positive_sum > 0 else pd.Series(dtype=float)
    hhi = float((shares**2).sum()) if not shares.empty else np.nan
    best_row = merged.loc[merged["candidate_minus_baseline_excess_return"].idxmax()] if not merged.empty else None
    worst_row = merged.loc[merged["candidate_minus_baseline_excess_return"].idxmin()] if not merged.empty else None
    summary = {
        "quarters_total": int(valid.shape[0]),
        "quarters_candidate_better": int((valid > 0).sum()),
        "quarters_candidate_worse": int((valid < 0).sum()),
        "positive_quarter_edge_sum": positive_sum if positive_sum > 0 else np.nan,
        "best_quarter": None if best_row is None else str(best_row["quarter"]),
        "best_quarter_edge": None if best_row is None else float(best_row["candidate_minus_baseline_excess_return"]),
        "best_quarter_positive_share": float(top1 / positive_sum) if positive_sum > 0 and np.isfinite(top1) else np.nan,
        "top2_positive_share": float(top2 / positive_sum) if positive_sum > 0 and np.isfinite(top2) else np.nan,
        "top3_positive_share": float(top3 / positive_sum) if positive_sum > 0 and np.isfinite(top3) else np.nan,
        "positive_quarter_hhi": hhi,
        "worst_quarter": None if worst_row is None else str(worst_row["quarter"]),
        "worst_quarter_edge": None if worst_row is None else float(worst_row["candidate_minus_baseline_excess_return"]),
        "quarters_with_positive_edge": int(pos.shape[0]),
        "quarters_with_negative_edge": int(neg.shape[0]),
    }
    return merged, summary


def _resolve_focus_quarter(explicit_quarter: str, concentration: dict[str, Any]) -> str:
    quarter = str(explicit_quarter or "").strip()
    if quarter and quarter.lower() != "auto":
        return quarter
    resolved = str(concentration.get("best_quarter") or "").strip()
    if not resolved:
        raise ValueError("Unable to resolve focus quarter from concentration summary.")
    return resolved


def _focus_compare(candidate: dict[str, Any], baseline: dict[str, Any], focus_quarter: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    cand = candidate["equity"].copy()
    base = baseline["equity"].copy()
    for df in (cand, base):
        df["quarter"] = df["date"].dt.to_period("Q").astype(str)
        df["month"] = df["date"].dt.to_period("M").astype(str)
    cand = cand.loc[
        cand["quarter"] == focus_quarter,
        ["date", "month", "quadrant", "regime_on", "excess_return", "turnover", "holding_count"],
    ]
    base = base.loc[
        base["quarter"] == focus_quarter,
        ["date", "excess_return", "turnover", "holding_count"],
    ]
    merged = cand.merge(base, on="date", how="inner", suffixes=("_candidate", "_baseline"))
    merged["edge"] = merged["excess_return_candidate"] - merged["excess_return_baseline"]
    merged["positive_edge"] = merged["edge"].clip(lower=0.0)
    positive_sum = float(merged["positive_edge"].sum())
    positive_sorted = merged.loc[merged["edge"] > 0, "edge"].sort_values(ascending=False)
    summary = {
        "focus_quarter": focus_quarter,
        "focus_quarter_candidate_compound_excess_return": _compound_return(merged["excess_return_candidate"]),
        "focus_quarter_baseline_compound_excess_return": _compound_return(merged["excess_return_baseline"]),
        "focus_quarter_edge_compound_gap": _compound_return(merged["excess_return_candidate"])
        - _compound_return(merged["excess_return_baseline"]),
        "focus_quarter_edge_sum_gap": float(merged["edge"].sum()),
        "focus_quarter_positive_day_count": int((merged["edge"] > 0).sum()),
        "focus_quarter_negative_day_count": int((merged["edge"] < 0).sum()),
        "focus_quarter_top1_positive_day_share": float(positive_sorted.iloc[:1].sum() / positive_sum)
        if positive_sum > 0
        else np.nan,
        "focus_quarter_top3_positive_day_share": float(positive_sorted.iloc[:3].sum() / positive_sum)
        if positive_sum > 0
        else np.nan,
        "focus_quarter_top5_positive_day_share": float(positive_sorted.iloc[:5].sum() / positive_sum)
        if positive_sum > 0
        else np.nan,
        "focus_quarter_top10_positive_day_share": float(positive_sorted.iloc[:10].sum() / positive_sum)
        if positive_sum > 0
        else np.nan,
        "focus_quarter_only_quadrant": str(merged["quadrant"].mode().iloc[0]) if not merged["quadrant"].dropna().empty else "",
        "focus_quarter_regime_active_ratio": float(merged["regime_on"].mean()) if not merged.empty else np.nan,
    }
    return merged, summary


def _monthly_edge(compare_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month, g in compare_df.groupby("month", sort=True):
        positive = g.loc[g["edge"] > 0, "edge"].sort_values(ascending=False)
        positive_sum = float(positive.sum())
        rows.append(
            {
                "month": month,
                "candidate_compound_excess_return": _compound_return(g["excess_return_candidate"]),
                "baseline_compound_excess_return": _compound_return(g["excess_return_baseline"]),
                "compound_edge_gap": _compound_return(g["excess_return_candidate"])
                - _compound_return(g["excess_return_baseline"]),
                "sum_edge_gap": float(g["edge"].sum()),
                "positive_day_count": int((g["edge"] > 0).sum()),
                "negative_day_count": int((g["edge"] < 0).sum()),
                "top3_positive_day_share": float(positive.iloc[:3].sum() / positive_sum) if positive_sum > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _top_day_swaps(candidate: dict[str, Any], baseline: dict[str, Any], compare_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    candidate_weights = candidate["target_weights"].set_index("date").sort_index()
    baseline_weights = baseline["target_weights"].set_index("date").sort_index()
    top_days = compare_df.nlargest(top_n, "edge").copy()
    rows: list[dict[str, Any]] = []
    for _, row in top_days.iterrows():
        date = row["date"]
        if date not in candidate_weights.index or date not in baseline_weights.index:
            continue
        cand_row = candidate_weights.loc[date].astype(float)
        base_row = baseline_weights.loc[date].astype(float)
        delta = (cand_row - base_row).sort_values(ascending=False)
        positive = delta[delta > 1e-9].head(5)
        negative = delta[delta < -1e-9].head(5)
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "month": row["month"],
                "quadrant": row["quadrant"],
                "edge": float(row["edge"]),
                "candidate_excess_return": float(row["excess_return_candidate"]),
                "baseline_excess_return": float(row["excess_return_baseline"]),
                "candidate_top_additions": "; ".join(f"{stock}:{value:.2%}" for stock, value in positive.items()),
                "candidate_top_reductions": "; ".join(f"{stock}:{value:.2%}" for stock, value in negative.items()),
            }
        )
    return pd.DataFrame(rows)


def _average_weight_delta(candidate: dict[str, Any], baseline: dict[str, Any], focus_quarter: str, top_n: int) -> pd.DataFrame:
    cand = candidate["target_weights"].copy()
    base = baseline["target_weights"].copy()
    cand["quarter"] = cand["date"].dt.to_period("Q").astype(str)
    base["quarter"] = base["date"].dt.to_period("Q").astype(str)
    cand = cand.loc[cand["quarter"] == focus_quarter].drop(columns=["date", "quarter"])
    base = base.loc[base["quarter"] == focus_quarter].drop(columns=["date", "quarter"])
    if cand.empty or base.empty:
        return pd.DataFrame(columns=["stock", "candidate_avg_weight", "baseline_avg_weight", "avg_weight_delta"])
    common_cols = [col for col in cand.columns if col in base.columns]
    delta = cand[common_cols].mean(axis=0) - base[common_cols].mean(axis=0)
    out = pd.DataFrame(
        {
            "stock": common_cols,
            "candidate_avg_weight": cand[common_cols].mean(axis=0).values,
            "baseline_avg_weight": base[common_cols].mean(axis=0).values,
            "avg_weight_delta": delta.values,
        }
    )
    return out.sort_values("avg_weight_delta", ascending=False, na_position="last").head(top_n).reset_index(drop=True)


def _holding_overlap(candidate: dict[str, Any], baseline: dict[str, Any], focus_quarter: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    cand = candidate["target_weights"].copy()
    base = baseline["target_weights"].copy()
    cand = cand.loc[cand["date"].dt.to_period("Q").astype(str) == focus_quarter].set_index("date").sort_index()
    base = base.loc[base["date"].dt.to_period("Q").astype(str) == focus_quarter].set_index("date").sort_index()
    common_dates = cand.index.intersection(base.index)
    rows: list[dict[str, Any]] = []
    for date in common_dates:
        cand_set = set(cand.columns[cand.loc[date].fillna(0.0) > 1e-9])
        base_set = set(base.columns[base.loc[date].fillna(0.0) > 1e-9])
        inter = cand_set & base_set
        union = cand_set | base_set
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "candidate_count": int(len(cand_set)),
                "baseline_count": int(len(base_set)),
                "intersection_count": int(len(inter)),
                "union_count": int(len(union)),
                "jaccard": float(len(inter) / len(union)) if union else np.nan,
            }
        )
    overlap_df = pd.DataFrame(rows)
    summary = {
        "focus_quarter": focus_quarter,
        "avg_jaccard": float(overlap_df["jaccard"].mean()) if not overlap_df.empty else np.nan,
        "median_jaccard": float(overlap_df["jaccard"].median()) if not overlap_df.empty else np.nan,
        "days_with_at_least_4_overlap": int((overlap_df["intersection_count"] >= 4).sum()) if not overlap_df.empty else 0,
        "days_total": int(len(overlap_df)),
        "at_least_4_overlap_ratio": float((overlap_df["intersection_count"] >= 4).mean()) if not overlap_df.empty else np.nan,
    }
    return overlap_df, summary


def _rankic_quarter_compare(candidate: dict[str, Any], baseline: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    cand = candidate["rankic_quarterly"].copy()
    base = baseline["rankic_quarterly"].copy()
    if cand.empty or base.empty:
        return pd.DataFrame(columns=["quarter"]), {}
    cand = cand[["quarter", "rank_ic_mean", "sample_days"]].rename(
        columns={"rank_ic_mean": "candidate_rank_ic_mean", "sample_days": "candidate_sample_days"}
    )
    base = base[["quarter", "rank_ic_mean", "sample_days"]].rename(
        columns={"rank_ic_mean": "baseline_rank_ic_mean", "sample_days": "baseline_sample_days"}
    )
    merged = cand.merge(base, on="quarter", how="outer").sort_values("quarter").reset_index(drop=True)
    merged["candidate_minus_baseline_rank_ic"] = merged["candidate_rank_ic_mean"] - merged["baseline_rank_ic_mean"]
    valid = merged["candidate_minus_baseline_rank_ic"].dropna()
    best_row = merged.loc[merged["candidate_minus_baseline_rank_ic"].idxmax()] if not merged.empty else None
    worst_row = merged.loc[merged["candidate_minus_baseline_rank_ic"].idxmin()] if not merged.empty else None
    summary = {
        "quarters_total": int(valid.shape[0]),
        "quarters_candidate_better": int((valid > 0).sum()),
        "quarters_candidate_worse": int((valid < 0).sum()),
        "best_quarter": None if best_row is None else str(best_row["quarter"]),
        "best_quarter_rank_ic_edge": None if best_row is None else float(best_row["candidate_minus_baseline_rank_ic"]),
        "worst_quarter": None if worst_row is None else str(worst_row["quarter"]),
        "worst_quarter_rank_ic_edge": None if worst_row is None else float(worst_row["candidate_minus_baseline_rank_ic"]),
    }
    return merged, summary


def _write_markdown(output_path: Path, summary_df: pd.DataFrame, results: dict[str, Any]) -> None:
    concentration = results["concentration"]
    focus_summary = results["focus_summary"]
    overlap_summary = results["overlap_summary"]
    rankic_summary = results["rankic_summary"]
    lines = [
        "# v2.1 Concentration Report",
        "",
        "## Overall",
        (
            f"- `{results['candidate_label']}` 相对 `{results['baseline_label']}`："
            f"{concentration['quarters_candidate_better']}/{concentration['quarters_total']} 个季度更强，"
            f"最佳季度 `{concentration['best_quarter']}` 占正向季度总优势 {_safe_pct(concentration['best_quarter_positive_share'])}，"
            f"Top3 季度占比 {_safe_pct(concentration['top3_positive_share'])}，"
            f"HHI {_safe_num(concentration['positive_quarter_hhi'])}"
        ),
        (
            f"- 焦点季度 `{focus_summary['focus_quarter']}`：compound 超额边际"
            f" {_safe_pct(focus_summary['focus_quarter_edge_compound_gap'])}，"
            f"Top5 正向日占比 {_safe_pct(focus_summary['focus_quarter_top5_positive_day_share'])}，"
            f"平均持仓重叠 Jaccard {_safe_num(overlap_summary['avg_jaccard'])}"
        ),
        (
            f"- `trend_up_low_vol` 的季度 RankIC 对照："
            f"{int(rankic_summary.get('quarters_candidate_better', 0))}/{int(rankic_summary.get('quarters_total', 0))} 个季度更强，"
            f"最佳季度 `{rankic_summary.get('best_quarter')}`，边际 {_safe_num(rankic_summary.get('best_quarter_rank_ic_edge'))}"
        ),
        "",
        "## Summary Rows",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- `{row['label']}`: full_excess_sharpe={_safe_num(row['full_excess_sharpe'])}, "
            f"recent_full_excess_return={_safe_pct(row['recent_full_excess_total_return'])}, "
            f"latest_weak_excess_return={_safe_pct(row['latest_weak_excess_total_return'])}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- 若最佳季度占比、Top3 季度占比、Top5 正向日占比都偏高，说明这条修复更像少数季度或少数交易日放大，而不是平滑稳定增益。",
            "- 若持仓重叠 Jaccard 很高，说明增量更可能来自少数槽位替换，而不是整套组合重写。",
            "- 若季度 RankIC 没有同步改善，则说明收益提升更可能来自交易路径或样本结构，而不是排序质量稳定升级。",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    baseline_run = _load_run(Path(args.baseline_run), args.baseline_label)
    candidate_run = _load_run(Path(args.candidate_run), args.candidate_label)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("daily_research/output") / f"v21_rule_concentration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = _summary_rows([baseline_run, candidate_run])
    summary_df.to_csv(output_dir / "summary_rows.csv", index=False, encoding="utf-8-sig")

    baseline_quarterly = _quarterly_frame(baseline_run)
    candidate_quarterly = _quarterly_frame(candidate_run)
    quarterly_df = pd.concat([baseline_quarterly, candidate_quarterly], axis=0, ignore_index=True)
    quarterly_df.to_csv(output_dir / "quarterly_metrics.csv", index=False, encoding="utf-8-sig")

    quarter_compare_df, concentration = _concentration_summary(candidate_quarterly, baseline_quarterly)
    focus_quarter = _resolve_focus_quarter(args.focus_quarter, concentration)
    quarter_compare_df.to_csv(output_dir / "candidate_vs_baseline_quarterly_compare.csv", index=False, encoding="utf-8-sig")

    focus_df, focus_summary = _focus_compare(candidate_run, baseline_run, focus_quarter)
    focus_df.to_csv(output_dir / "focus_quarter_daily.csv", index=False, encoding="utf-8-sig")

    monthly_df = _monthly_edge(focus_df)
    monthly_df.to_csv(output_dir / "focus_quarter_monthly.csv", index=False, encoding="utf-8-sig")

    top_days_df = _top_day_swaps(candidate_run, baseline_run, focus_df, args.top_days)
    top_days_df.to_csv(output_dir / "focus_quarter_top_days.csv", index=False, encoding="utf-8-sig")

    weight_delta_df = _average_weight_delta(candidate_run, baseline_run, focus_quarter, args.top_weight_delta)
    weight_delta_df.to_csv(output_dir / "focus_quarter_weight_delta.csv", index=False, encoding="utf-8-sig")

    overlap_df, overlap_summary = _holding_overlap(candidate_run, baseline_run, focus_quarter)
    overlap_df.to_csv(output_dir / "focus_quarter_overlap.csv", index=False, encoding="utf-8-sig")

    rankic_compare_df, rankic_summary = _rankic_quarter_compare(candidate_run, baseline_run)
    rankic_compare_df.to_csv(output_dir / "quarterly_rankic_compare.csv", index=False, encoding="utf-8-sig")

    results = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "focus_quarter": focus_quarter,
        "concentration": concentration,
        "focus_summary": focus_summary,
        "overlap_summary": overlap_summary,
        "rankic_summary": rankic_summary,
    }
    (output_dir / "report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(output_dir / "report.md", summary_df, results)

    print(f"Output: {output_dir}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
