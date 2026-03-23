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
        description="Diagnose quarterly concentration and focus-quarter attribution for ma50 ensemble candidates."
    )
    parser.add_argument("--baseline-run", required=True, help="Path to ma50 baseline run")
    parser.add_argument("--repair-run", required=True, help="Path to weak-window repair candidate run")
    parser.add_argument("--balance-run", required=True, help="Path to relative balance candidate run")
    parser.add_argument("--baseline-label", default="ma50_baseline")
    parser.add_argument("--repair-label", default="up_low_ml62_none23_v215")
    parser.add_argument("--balance-label", default="up_low_ml61_none24_v215")
    parser.add_argument("--focus-quarter", default="2026Q1")
    parser.add_argument("--top-days", type=int, default=10)
    parser.add_argument("--top-weight-delta", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output dir. Defaults to daily_research/output/ma50_ensemble_q1_<timestamp>",
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
    target_weights_path = run_dir / "target_weights.csv"
    target_weights = (
        pd.read_csv(target_weights_path, parse_dates=["Date"]).rename(columns={"Date": "date"})
        if target_weights_path.exists()
        else pd.DataFrame(columns=["date"])
    )
    equity = equity.merge(regime[["date", "quadrant", "regime_on"]], on="date", how="left", suffixes=("", "_regime"))
    return {
        "label": label,
        "run_dir": run_dir,
        "metrics": metrics,
        "recent_full": recent_full,
        "latest_weak": latest_weak,
        "equity": equity,
        "target_weights": target_weights,
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
    if candidate["target_weights"].empty or baseline["target_weights"].empty:
        return pd.DataFrame(
            columns=[
                "date",
                "month",
                "quadrant",
                "edge",
                "candidate_excess_return",
                "baseline_excess_return",
                "candidate_top_additions",
                "candidate_top_reductions",
            ]
        )
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
                "candidate_top_additions": "; ".join(f"{stock}:{weight:.3f}" for stock, weight in positive.items()),
                "candidate_top_reductions": "; ".join(f"{stock}:{weight:.3f}" for stock, weight in negative.items()),
            }
        )
    return pd.DataFrame(rows)


def _average_weight_delta(candidate: dict[str, Any], baseline: dict[str, Any], focus_quarter: str, top_n: int) -> pd.DataFrame:
    if candidate["target_weights"].empty or baseline["target_weights"].empty:
        return pd.DataFrame(columns=["stock", "candidate_avg_weight", "baseline_avg_weight", "delta", "abs_delta"])
    cand = candidate["target_weights"].copy()
    base = baseline["target_weights"].copy()
    cand = cand.loc[cand["date"].dt.to_period("Q").astype(str) == focus_quarter].drop(columns=["date"])
    base = base.loc[base["date"].dt.to_period("Q").astype(str) == focus_quarter].drop(columns=["date"])
    cand_avg = cand.astype(float).mean(axis=0).rename("candidate_avg_weight")
    base_avg = base.astype(float).mean(axis=0).rename("baseline_avg_weight")
    merged = pd.concat([cand_avg, base_avg], axis=1).fillna(0.0)
    merged["delta"] = merged["candidate_avg_weight"] - merged["baseline_avg_weight"]
    merged["abs_delta"] = merged["delta"].abs()
    merged = merged.sort_values("abs_delta", ascending=False).head(top_n).reset_index().rename(columns={"index": "stock"})
    return merged


def _holding_overlap(candidate: dict[str, Any], baseline: dict[str, Any], focus_quarter: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if candidate["target_weights"].empty or baseline["target_weights"].empty:
        empty = pd.DataFrame(columns=["date", "candidate_count", "baseline_count", "intersection", "union", "jaccard"])
        return empty, {"avg_jaccard": np.nan, "median_jaccard": np.nan, "share_overlap_ge4": np.nan, "days": 0}
    cand = candidate["target_weights"].copy()
    base = baseline["target_weights"].copy()
    cand = cand.loc[cand["date"].dt.to_period("Q").astype(str) == focus_quarter].set_index("date").sort_index()
    base = base.loc[base["date"].dt.to_period("Q").astype(str) == focus_quarter].set_index("date").sort_index()
    rows: list[dict[str, Any]] = []
    for date in cand.index.intersection(base.index):
        cand_row = cand.loc[date].astype(float)
        base_row = base.loc[date].astype(float)
        cand_names = set(cand_row.index[cand_row > 1e-9])
        base_names = set(base_row.index[base_row > 1e-9])
        intersection = cand_names & base_names
        union = cand_names | base_names
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "candidate_count": int(len(cand_names)),
                "baseline_count": int(len(base_names)),
                "intersection": int(len(intersection)),
                "union": int(len(union)),
                "jaccard": float(len(intersection) / len(union)) if union else np.nan,
            }
        )
    overlap_df = pd.DataFrame(rows)
    summary = {
        "avg_jaccard": float(overlap_df["jaccard"].mean()) if not overlap_df.empty else np.nan,
        "median_jaccard": float(overlap_df["jaccard"].median()) if not overlap_df.empty else np.nan,
        "share_overlap_ge4": float((overlap_df["intersection"] >= 4).mean()) if not overlap_df.empty else np.nan,
        "days": int(len(overlap_df)),
    }
    return overlap_df, summary


def _write_markdown(output_path: Path, summary_df: pd.DataFrame, results: dict[str, dict[str, Any]], focus_quarter: str) -> None:
    rows = {row["label"]: row for row in summary_df.to_dict(orient="records")}
    baseline = rows[next(label for label in rows if "baseline" in label)]
    repair = results["repair"]
    balance = results["balance"]
    lines = [
        "# ma50 ensemble 候选季度集中度与 2026Q1 归因诊断",
        "",
        "## 核心判断",
        (
            f"- `up_low_ml62_none23_v215`：全样本超额 Sharpe {_safe_num(rows[repair['label']]['full_excess_sharpe'])}，"
            f"最新弱窗口 {_safe_pct(rows[repair['label']]['latest_weak_excess_total_return'])} / {_safe_num(rows[repair['label']]['latest_weak_excess_sharpe'])}"
        ),
        (
            f"- `up_low_ml61_none24_v215`：全样本超额 Sharpe {_safe_num(rows[balance['label']]['full_excess_sharpe'])}，"
            f"最新弱窗口 {_safe_pct(rows[balance['label']]['latest_weak_excess_total_return'])} / {_safe_num(rows[balance['label']]['latest_weak_excess_sharpe'])}"
        ),
        (
            f"- baseline：全样本超额 Sharpe {_safe_num(baseline['full_excess_sharpe'])}，"
            f"最新弱窗口 {_safe_pct(baseline['latest_weak_excess_total_return'])} / {_safe_num(baseline['latest_weak_excess_sharpe'])}"
        ),
        "",
        "## 季度集中度",
        (
            f"- `up_low_ml62_none23_v215` 相对 baseline：{repair['concentration']['quarters_candidate_better']}/{repair['concentration']['quarters_total']} 个季度更强，"
            f"最佳季度 `{repair['concentration']['best_quarter']}` 占正向季度总优势 {_safe_pct(repair['concentration']['best_quarter_positive_share'])}，"
            f"Top3 季度占比 {_safe_pct(repair['concentration']['top3_positive_share'])}，"
            f"HHI {_safe_num(repair['concentration']['positive_quarter_hhi'])}"
        ),
        (
            f"- `up_low_ml61_none24_v215` 相对 baseline：{balance['concentration']['quarters_candidate_better']}/{balance['concentration']['quarters_total']} 个季度更强，"
            f"最佳季度 `{balance['concentration']['best_quarter']}` 占正向季度总优势 {_safe_pct(balance['concentration']['best_quarter_positive_share'])}，"
            f"Top3 季度占比 {_safe_pct(balance['concentration']['top3_positive_share'])}，"
            f"HHI {_safe_num(balance['concentration']['positive_quarter_hhi'])}"
        ),
        "",
        f"## 焦点季度 `{focus_quarter}`",
        (
            f"- `up_low_ml62_none23_v215`：compound 超额边际 {_safe_pct(repair['focus_summary']['focus_quarter_edge_compound_gap'])}，"
            f"Top5 正向日占比 {_safe_pct(repair['focus_summary']['focus_quarter_top5_positive_day_share'])}，"
            f"平均持仓重叠 Jaccard {_safe_num(repair['overlap_summary']['avg_jaccard'])}"
        ),
        (
            f"- `up_low_ml61_none24_v215`：compound 超额边际 {_safe_pct(balance['focus_summary']['focus_quarter_edge_compound_gap'])}，"
            f"Top5 正向日占比 {_safe_pct(balance['focus_summary']['focus_quarter_top5_positive_day_share'])}，"
            f"平均持仓重叠 Jaccard {_safe_num(balance['overlap_summary']['avg_jaccard'])}"
        ),
        "",
        "## 解释口径",
        "- 若最佳季度占比与 Top3 季度占比偏高，说明修复更像少数季度放大，而不是平滑稳定增益。",
        "- 若焦点季度的平均持仓重叠仍较高，则说明差异主要来自少数槽位替换，而不是整套组合重写。",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("daily_research/output") / f"ma50_ensemble_q1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_run = _load_run(Path(args.baseline_run), args.baseline_label)
    repair_run = _load_run(Path(args.repair_run), args.repair_label)
    balance_run = _load_run(Path(args.balance_run), args.balance_label)
    runs = [baseline_run, repair_run, balance_run]
    summary_df = _summary_rows(runs)
    summary_df.to_csv(output_dir / "summary_rows.csv", index=False, encoding="utf-8-sig")

    baseline_quarterly = _quarterly_frame(baseline_run)
    repair_quarterly = _quarterly_frame(repair_run)
    balance_quarterly = _quarterly_frame(balance_run)

    repair_q_compare, repair_concentration = _concentration_summary(repair_quarterly, baseline_quarterly)
    balance_q_compare, balance_concentration = _concentration_summary(balance_quarterly, baseline_quarterly)
    repair_q_compare.to_csv(output_dir / "repair_vs_baseline_quarterly_compare.csv", index=False, encoding="utf-8-sig")
    balance_q_compare.to_csv(output_dir / "balance_vs_baseline_quarterly_compare.csv", index=False, encoding="utf-8-sig")

    repair_focus_df, repair_focus_summary = _focus_compare(repair_run, baseline_run, args.focus_quarter)
    balance_focus_df, balance_focus_summary = _focus_compare(balance_run, baseline_run, args.focus_quarter)
    repair_focus_df.to_csv(output_dir / "repair_focus_quarter_daily.csv", index=False, encoding="utf-8-sig")
    balance_focus_df.to_csv(output_dir / "balance_focus_quarter_daily.csv", index=False, encoding="utf-8-sig")

    repair_monthly = _monthly_edge(repair_focus_df)
    balance_monthly = _monthly_edge(balance_focus_df)
    repair_monthly.to_csv(output_dir / "repair_focus_quarter_monthly.csv", index=False, encoding="utf-8-sig")
    balance_monthly.to_csv(output_dir / "balance_focus_quarter_monthly.csv", index=False, encoding="utf-8-sig")

    repair_top_days = _top_day_swaps(repair_run, baseline_run, repair_focus_df, args.top_days)
    balance_top_days = _top_day_swaps(balance_run, baseline_run, balance_focus_df, args.top_days)
    repair_top_days.to_csv(output_dir / "repair_focus_quarter_top_days.csv", index=False, encoding="utf-8-sig")
    balance_top_days.to_csv(output_dir / "balance_focus_quarter_top_days.csv", index=False, encoding="utf-8-sig")

    repair_weight_delta = _average_weight_delta(repair_run, baseline_run, args.focus_quarter, args.top_weight_delta)
    balance_weight_delta = _average_weight_delta(balance_run, baseline_run, args.focus_quarter, args.top_weight_delta)
    repair_weight_delta.to_csv(output_dir / "repair_focus_quarter_weight_delta.csv", index=False, encoding="utf-8-sig")
    balance_weight_delta.to_csv(output_dir / "balance_focus_quarter_weight_delta.csv", index=False, encoding="utf-8-sig")

    repair_overlap_df, repair_overlap_summary = _holding_overlap(repair_run, baseline_run, args.focus_quarter)
    balance_overlap_df, balance_overlap_summary = _holding_overlap(balance_run, baseline_run, args.focus_quarter)
    repair_overlap_df.to_csv(output_dir / "repair_focus_quarter_overlap.csv", index=False, encoding="utf-8-sig")
    balance_overlap_df.to_csv(output_dir / "balance_focus_quarter_overlap.csv", index=False, encoding="utf-8-sig")

    results = {
        "focus_quarter": args.focus_quarter,
        "repair": {
            "label": args.repair_label,
            "concentration": repair_concentration,
            "focus_summary": repair_focus_summary,
            "overlap_summary": repair_overlap_summary,
        },
        "balance": {
            "label": args.balance_label,
            "concentration": balance_concentration,
            "focus_summary": balance_focus_summary,
            "overlap_summary": balance_overlap_summary,
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "summary_rows": summary_df.to_dict(orient="records"),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_markdown(output_dir / "report.md", summary_df, results, args.focus_quarter)
    print(f"output: {output_dir}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
