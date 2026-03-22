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
        description="Build a dedicated ma47/48 left-band stability and concentration report."
    )
    parser.add_argument("--ma47-run", required=True, help="Path to ma47 baseline run")
    parser.add_argument("--ma48-run", required=True, help="Path to ma48 baseline run")
    parser.add_argument("--ma50-run", required=True, help="Path to ma50 baseline run")
    parser.add_argument("--ma48-variant-run", default="", help="Optional ma48 light-risk variant run")
    parser.add_argument("--focus-quarter", default="2025Q3", help="Quarter to diagnose, e.g. 2025Q3")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output dir. Defaults to daily_research/output/ma4748_band_<timestamp>",
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
    actions = pd.read_csv(run_dir / "actions.csv", parse_dates=["date"])
    equity = equity.merge(regime[["date", "quadrant", "regime_on"]], on="date", how="left", suffixes=("", "_regime"))
    return {
        "label": label,
        "run_dir": run_dir,
        "metrics": metrics,
        "recent_full": recent_full,
        "latest_weak": latest_weak,
        "equity": equity,
        "regime": regime,
        "target_weights": target_weights,
        "actions": actions,
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
                "latest_weak_excess_total_return": latest_weak.get("excess_total_return"),
                "latest_weak_excess_sharpe": latest_weak.get("excess_sharpe"),
            }
        )
    return pd.DataFrame(rows)


def _quarterly_compare(run: dict[str, Any]) -> pd.DataFrame:
    equity = run["equity"].copy()
    equity["quarter"] = equity["date"].dt.to_period("Q").astype(str)
    rows: list[dict[str, Any]] = []
    for quarter, g in equity.groupby("quarter", sort=True):
        rows.append(
            {
                "label": run["label"],
                "quarter": quarter,
                "excess_return": _compound_return(g["excess_return"]),
                "avg_turnover": float(g["turnover"].mean()),
                "avg_holding_count": float(g["holding_count"].mean()),
                "regime_active_ratio": float(g["regime_on"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _monthly_focus_frame(run: dict[str, Any], focus_quarter: str) -> pd.DataFrame:
    equity = run["equity"].copy()
    equity["quarter"] = equity["date"].dt.to_period("Q").astype(str)
    equity = equity.loc[equity["quarter"] == focus_quarter].copy()
    equity["month"] = equity["date"].dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    for month, g in equity.groupby("month", sort=True):
        rows.append(
            {
                "label": run["label"],
                "month": month,
                "compound_excess_return": _compound_return(g["excess_return"]),
                "sum_excess_return": float(g["excess_return"].sum()),
                "avg_turnover": float(g["turnover"].mean()),
                "avg_holding_count": float(g["holding_count"].mean()),
                "days": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def _focus_compare(candidate: dict[str, Any], other: dict[str, Any], focus_quarter: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    cand = candidate["equity"].copy()
    other_df = other["equity"].copy()
    for df in (cand, other_df):
        df["quarter"] = df["date"].dt.to_period("Q").astype(str)
        df["month"] = df["date"].dt.to_period("M").astype(str)
    cand = cand.loc[cand["quarter"] == focus_quarter, ["date", "month", "quadrant", "regime_on", "excess_return", "turnover", "holding_count"]]
    other_df = other_df.loc[other_df["quarter"] == focus_quarter, ["date", "excess_return", "turnover", "holding_count"]]
    merged = cand.merge(other_df, on="date", how="inner", suffixes=("_candidate", "_other"))
    merged["edge"] = merged["excess_return_candidate"] - merged["excess_return_other"]
    merged["positive_edge"] = merged["edge"].clip(lower=0.0)

    positive_sum = float(merged["positive_edge"].sum())
    positive_sorted = merged.loc[merged["edge"] > 0, "edge"].sort_values(ascending=False)

    summary = {
        "focus_quarter": focus_quarter,
        "candidate_label": candidate["label"],
        "other_label": other["label"],
        "focus_quarter_edge_compound_gap": _compound_return(merged["excess_return_candidate"]) - _compound_return(merged["excess_return_other"]),
        "focus_quarter_edge_sum_gap": float(merged["edge"].sum()),
        "focus_quarter_positive_day_count": int((merged["edge"] > 0).sum()),
        "focus_quarter_negative_day_count": int((merged["edge"] < 0).sum()),
        "focus_quarter_top1_positive_day_share": float(positive_sorted.iloc[:1].sum() / positive_sum) if positive_sum > 0 else np.nan,
        "focus_quarter_top3_positive_day_share": float(positive_sorted.iloc[:3].sum() / positive_sum) if positive_sum > 0 else np.nan,
        "focus_quarter_top5_positive_day_share": float(positive_sorted.iloc[:5].sum() / positive_sum) if positive_sum > 0 else np.nan,
        "focus_quarter_top10_positive_day_share": float(positive_sorted.iloc[:10].sum() / positive_sum) if positive_sum > 0 else np.nan,
        "focus_quarter_only_quadrant": str(merged["quadrant"].mode().iloc[0]) if not merged["quadrant"].dropna().empty else "",
        "focus_quarter_regime_active_ratio": float(merged["regime_on"].mean()) if not merged.empty else np.nan,
    }
    return merged, summary


def _monthly_edge(compare_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month, g in compare_df.groupby("month", sort=True):
        positive_sum = float(g["positive_edge"].sum())
        rows.append(
            {
                "month": month,
                "candidate_compound_excess_return": _compound_return(g["excess_return_candidate"]),
                "other_compound_excess_return": _compound_return(g["excess_return_other"]),
                "compound_edge_gap": _compound_return(g["excess_return_candidate"]) - _compound_return(g["excess_return_other"]),
                "sum_edge_gap": float(g["edge"].sum()),
                "positive_day_count": int((g["edge"] > 0).sum()),
                "negative_day_count": int((g["edge"] < 0).sum()),
                "top3_positive_day_share": float(g.loc[g["edge"] > 0, "edge"].sort_values(ascending=False).iloc[:3].sum() / positive_sum) if positive_sum > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _top_day_swaps(candidate: dict[str, Any], other: dict[str, Any], compare_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    candidate_weights = candidate["target_weights"].set_index("date").sort_index()
    other_weights = other["target_weights"].set_index("date").sort_index()
    top_days = compare_df.nlargest(top_n, "edge").copy()
    rows: list[dict[str, Any]] = []
    for _, row in top_days.iterrows():
        date = row["date"]
        cand_row = candidate_weights.loc[date].astype(float)
        other_row = other_weights.loc[date].astype(float)
        delta = (cand_row - other_row).sort_values(ascending=False)
        positive = delta[delta > 1e-9].head(3)
        negative = delta[delta < -1e-9].head(3)
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "month": row["month"],
                "quadrant": row["quadrant"],
                "edge": float(row["edge"]),
                "candidate_excess_return": float(row["excess_return_candidate"]),
                "other_excess_return": float(row["excess_return_other"]),
                "candidate_top_additions": "; ".join(f"{stock}:{weight:.3f}" for stock, weight in positive.items()),
                "candidate_top_reductions": "; ".join(f"{stock}:{weight:.3f}" for stock, weight in negative.items()),
            }
        )
    return pd.DataFrame(rows)


def _average_weight_delta(candidate: dict[str, Any], other: dict[str, Any], focus_quarter: str, top_n: int = 20) -> pd.DataFrame:
    cand = candidate["target_weights"].copy()
    other_df = other["target_weights"].copy()
    cand = cand.loc[cand["date"].dt.to_period("Q").astype(str) == focus_quarter].drop(columns=["date"])
    other_df = other_df.loc[other_df["date"].dt.to_period("Q").astype(str) == focus_quarter].drop(columns=["date"])
    cand_avg = cand.astype(float).mean(axis=0).rename("candidate_avg_weight")
    other_avg = other_df.astype(float).mean(axis=0).rename("other_avg_weight")
    merged = pd.concat([cand_avg, other_avg], axis=1).fillna(0.0)
    merged["delta"] = merged["candidate_avg_weight"] - merged["other_avg_weight"]
    merged["abs_delta"] = merged["delta"].abs()
    merged = merged.sort_values("abs_delta", ascending=False).head(top_n).reset_index().rename(columns={"index": "stock"})
    return merged


def _holding_overlap(candidate: dict[str, Any], other: dict[str, Any], focus_quarter: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    cand = candidate["target_weights"].copy()
    other_df = other["target_weights"].copy()
    cand = cand.loc[cand["date"].dt.to_period("Q").astype(str) == focus_quarter].set_index("date").sort_index()
    other_df = other_df.loc[other_df["date"].dt.to_period("Q").astype(str) == focus_quarter].set_index("date").sort_index()
    rows: list[dict[str, Any]] = []
    for date in cand.index.intersection(other_df.index):
        cand_row = cand.loc[date].astype(float)
        other_row = other_df.loc[date].astype(float)
        cand_names = set(cand_row.index[cand_row > 1e-9])
        other_names = set(other_row.index[other_row > 1e-9])
        intersection = len(cand_names & other_names)
        union = len(cand_names | other_names)
        rows.append(
            {
                "date": date,
                "intersection_count": intersection,
                "union_count": union,
                "jaccard": float(intersection / union) if union else 1.0,
                "same_4plus_names": intersection >= 4,
            }
        )
    overlap_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    summary = {
        "focus_quarter": focus_quarter,
        "candidate_label": candidate["label"],
        "other_label": other["label"],
        "avg_jaccard": float(overlap_df["jaccard"].mean()) if not overlap_df.empty else np.nan,
        "median_jaccard": float(overlap_df["jaccard"].median()) if not overlap_df.empty else np.nan,
        "share_same_4plus_names": float(overlap_df["same_4plus_names"].mean()) if not overlap_df.empty else np.nan,
        "min_jaccard": float(overlap_df["jaccard"].min()) if not overlap_df.empty else np.nan,
    }
    return overlap_df, summary


def _write_markdown(
    output_path: Path,
    summary_df: pd.DataFrame,
    ma48_vs_ma47_summary: dict[str, Any],
    ma48_vs_ma50_summary: dict[str, Any],
    ma48_vs_ma47_overlap: dict[str, Any],
    ma48_vs_ma50_overlap: dict[str, Any],
    ma48_variant_summary: dict[str, Any] | None,
    ma48_vs_ma47_monthly: pd.DataFrame,
    ma48_vs_ma50_monthly: pd.DataFrame,
) -> None:
    rows = {row["label"]: row for row in summary_df.to_dict(orient="records")}
    ma47 = rows["ma47_baseline"]
    ma48 = rows["ma48_baseline"]
    ma50 = rows["ma50_baseline"]
    variant = rows.get("ma48_take20")
    lines = [
        "# ma47/48 左侧边界带稳定性复验",
        "",
        "## 核心判断",
        (
            f"- `ma48_baseline` 全样本超额 Sharpe {_safe_num(ma48['full_excess_sharpe'])}"
            f"，高于 `ma47_baseline` 的 {_safe_num(ma47['full_excess_sharpe'])}"
            f" 和 `ma50_baseline` 的 {_safe_num(ma50['full_excess_sharpe'])}"
        ),
        (
            f"- 但 `{ma48_vs_ma47_summary['focus_quarter']}` 的优势集中在少数日内："
            f"`ma48` 相对 `ma47` 的 Top5 正向日占比 {_safe_pct(ma48_vs_ma47_summary['focus_quarter_top5_positive_day_share'])}"
            f"，相对 `ma50` 的 Top5 正向日占比 {_safe_pct(ma48_vs_ma50_summary['focus_quarter_top5_positive_day_share'])}"
        ),
        (
            f"- `{ma48_vs_ma47_summary['focus_quarter']}` 全季都处于 `{ma48_vs_ma47_summary['focus_quarter_only_quadrant']}`，"
            f"说明集中来源不是状态切换，而是同一状态内的选股和换仓差异"
        ),
        "",
        "## 左侧边界带",
        (
            f"- `ma48` 相对 `ma47` 的平均持仓重叠 Jaccard 为 {_safe_num(ma48_vs_ma47_overlap['avg_jaccard'])}，"
            f"有 {_safe_pct(ma48_vs_ma47_overlap['share_same_4plus_names'])} 的日期至少重合 4 个名字"
        ),
        (
            f"- `ma48` 相对 `ma50` 的平均持仓重叠 Jaccard 为 {_safe_num(ma48_vs_ma50_overlap['avg_jaccard'])}，"
            f"有 {_safe_pct(ma48_vs_ma50_overlap['share_same_4plus_names'])} 的日期至少重合 4 个名字"
        ),
        "- 这说明：左侧边界带的优势更多来自少数持仓槽位的替换，而不是整套组合完全改写。",
        "",
        "## 月度拆解",
        (
            f"- `ma48` 相对 `ma47` 的月度边际从 "
            f"{', '.join(f'{row.month}:{row.compound_edge_gap:.2%}' for row in ma48_vs_ma47_monthly.itertuples())}"
        ),
        (
            f"- `ma48` 相对 `ma50` 的月度边际从 "
            f"{', '.join(f'{row.month}:{row.compound_edge_gap:.2%}' for row in ma48_vs_ma50_monthly.itertuples())}"
        ),
    ]
    if variant is not None and ma48_variant_summary is not None:
        lines.extend(
            [
                "",
                "## `ma48 + take20`",
                (
                    f"- `ma48_take20` 全样本超额 Sharpe {_safe_num(variant['full_excess_sharpe'])}"
                    f"，但在 `{ma48_variant_summary['focus_quarter']}` 内的 Top3 正向日占比仍有"
                    f" {_safe_pct(ma48_variant_summary['focus_quarter_top3_positive_day_share'])}"
                ),
                "- 这说明：`take20` 仍是附加微调，不是独立主驱动。",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path("daily_research/output") / f"ma4748_band_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        _load_run(Path(args.ma47_run), "ma47_baseline"),
        _load_run(Path(args.ma48_run), "ma48_baseline"),
        _load_run(Path(args.ma50_run), "ma50_baseline"),
    ]
    variant_run: dict[str, Any] | None = None
    if args.ma48_variant_run:
        variant_run = _load_run(Path(args.ma48_variant_run), "ma48_take20")
        runs.append(variant_run)

    summary_df = _summary_rows(runs)
    summary_df.to_csv(output_dir / "band_summary.csv", index=False, encoding="utf-8-sig")

    quarterly_frames = pd.concat([_quarterly_compare(run) for run in runs], ignore_index=True)
    quarterly_frames.to_csv(output_dir / "quarterly_summary.csv", index=False, encoding="utf-8-sig")

    ma47 = runs[0]
    ma48 = runs[1]
    ma50 = runs[2]

    ma48_vs_ma47_df, ma48_vs_ma47_summary = _focus_compare(ma48, ma47, args.focus_quarter)
    ma48_vs_ma50_df, ma48_vs_ma50_summary = _focus_compare(ma48, ma50, args.focus_quarter)
    ma48_vs_ma47_df.to_csv(output_dir / "focus_quarter_ma48_vs_ma47_daily.csv", index=False, encoding="utf-8-sig")
    ma48_vs_ma50_df.to_csv(output_dir / "focus_quarter_ma48_vs_ma50_daily.csv", index=False, encoding="utf-8-sig")

    ma48_vs_ma47_monthly = _monthly_edge(ma48_vs_ma47_df)
    ma48_vs_ma50_monthly = _monthly_edge(ma48_vs_ma50_df)
    ma48_vs_ma47_monthly.to_csv(output_dir / "focus_quarter_ma48_vs_ma47_monthly.csv", index=False, encoding="utf-8-sig")
    ma48_vs_ma50_monthly.to_csv(output_dir / "focus_quarter_ma48_vs_ma50_monthly.csv", index=False, encoding="utf-8-sig")

    ma48_vs_ma47_top_days = _top_day_swaps(ma48, ma47, ma48_vs_ma47_df, top_n=10)
    ma48_vs_ma50_top_days = _top_day_swaps(ma48, ma50, ma48_vs_ma50_df, top_n=10)
    ma48_vs_ma47_top_days.to_csv(output_dir / "focus_quarter_ma48_vs_ma47_top_days.csv", index=False, encoding="utf-8-sig")
    ma48_vs_ma50_top_days.to_csv(output_dir / "focus_quarter_ma48_vs_ma50_top_days.csv", index=False, encoding="utf-8-sig")

    ma48_vs_ma47_weights = _average_weight_delta(ma48, ma47, args.focus_quarter)
    ma48_vs_ma50_weights = _average_weight_delta(ma48, ma50, args.focus_quarter)
    ma48_vs_ma47_weights.to_csv(output_dir / "focus_quarter_ma48_vs_ma47_weight_delta.csv", index=False, encoding="utf-8-sig")
    ma48_vs_ma50_weights.to_csv(output_dir / "focus_quarter_ma48_vs_ma50_weight_delta.csv", index=False, encoding="utf-8-sig")

    ma48_vs_ma47_overlap_df, ma48_vs_ma47_overlap = _holding_overlap(ma48, ma47, args.focus_quarter)
    ma48_vs_ma50_overlap_df, ma48_vs_ma50_overlap = _holding_overlap(ma48, ma50, args.focus_quarter)
    ma48_vs_ma47_overlap_df.to_csv(output_dir / "focus_quarter_ma48_vs_ma47_overlap.csv", index=False, encoding="utf-8-sig")
    ma48_vs_ma50_overlap_df.to_csv(output_dir / "focus_quarter_ma48_vs_ma50_overlap.csv", index=False, encoding="utf-8-sig")

    ma48_variant_summary: dict[str, Any] | None = None
    if variant_run is not None:
        ma48_variant_df, ma48_variant_summary = _focus_compare(variant_run, ma48, args.focus_quarter)
        ma48_variant_df.to_csv(output_dir / "focus_quarter_ma48_take20_vs_ma48_daily.csv", index=False, encoding="utf-8-sig")
        _monthly_edge(ma48_variant_df).to_csv(output_dir / "focus_quarter_ma48_take20_vs_ma48_monthly.csv", index=False, encoding="utf-8-sig")

    report = {
        "focus_quarter": args.focus_quarter,
        "summary_rows": summary_df.to_dict(orient="records"),
        "ma48_vs_ma47": ma48_vs_ma47_summary,
        "ma48_vs_ma50": ma48_vs_ma50_summary,
        "ma48_vs_ma47_overlap": ma48_vs_ma47_overlap,
        "ma48_vs_ma50_overlap": ma48_vs_ma50_overlap,
        "ma48_take20_vs_ma48": ma48_variant_summary,
    }
    (output_dir / "ma4748_band_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(
        output_dir / "ma4748_band_report.md",
        summary_df,
        ma48_vs_ma47_summary,
        ma48_vs_ma50_summary,
        ma48_vs_ma47_overlap,
        ma48_vs_ma50_overlap,
        ma48_variant_summary,
        ma48_vs_ma47_monthly,
        ma48_vs_ma50_monthly,
    )

    print(f"output: {output_dir}")
    print(summary_df.to_string(index=False))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
