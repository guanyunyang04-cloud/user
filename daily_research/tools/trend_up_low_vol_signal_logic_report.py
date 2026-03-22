from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.baseline.alpha import combine_scores_by_state
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import load_daily_from_tq, split_benchmark_from_universe
from daily_research.baseline.features import compute_factors
from daily_research.baseline.ml_alpha import build_ml_target
from daily_research.baseline.regime import compute_market_regime_state
from daily_research.baseline.state_profiles import build_state_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Abstract Q3 slot swaps into reusable trend_up_low_vol signal logic and test replay."
    )
    parser.add_argument("--ma47-run", required=True, help="Path to ma47 baseline run")
    parser.add_argument("--ma48-run", required=True, help="Path to ma48 baseline run")
    parser.add_argument("--ma50-run", required=True, help="Path to ma50 baseline run")
    parser.add_argument("--start-date", default="20220101")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--focus-quarter", default="2025Q3")
    parser.add_argument("--focus-quadrant", default="trend_up_low_vol")
    parser.add_argument("--top-days", type=int, default=10, help="Top Q3 edge days used to infer candidate factors")
    parser.add_argument("--slot-top-n", type=int, default=5, help="Top additions and reductions kept per quarter")
    parser.add_argument("--factor-top-k", type=int, default=6, help="Maximum factor count for abstracted union signal")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output dir. Defaults to daily_research/output/trend_up_low_vol_signal_logic_<timestamp>",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{float(value):.2%}"


def _rank_corr(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 2:
        return np.nan
    if pair.iloc[:, 0].nunique() <= 1 or pair.iloc[:, 1].nunique() <= 1:
        return np.nan
    xr = pair.iloc[:, 0].rank(method="average")
    yr = pair.iloc[:, 1].rank(method="average")
    return float(xr.corr(yr))


def _load_run(run_dir: Path, label: str) -> Dict[str, Any]:
    return {
        "label": label,
        "run_dir": run_dir,
        "equity": pd.read_csv(run_dir / "equity_curve.csv", parse_dates=["date"]),
        "target_weights": pd.read_csv(run_dir / "target_weights.csv", parse_dates=["Date"]).rename(columns={"Date": "date"}),
        "metrics": _load_json(run_dir / "metrics.json"),
    }


def _collect_union_stocks(runs: Sequence[Dict[str, Any]]) -> List[str]:
    stocks: set[str] = set()
    for run in runs:
        stocks.update(col for col in run["target_weights"].columns if col != "date")
    return sorted(stocks)


def _resolve_end_date(raw_end_date: str, runs: Sequence[Dict[str, Any]]) -> str:
    if str(raw_end_date).strip():
        return str(raw_end_date).strip()
    latest = max(pd.Timestamp(run["equity"]["date"].max()) for run in runs)
    return latest.strftime("%Y%m%d")


def _build_base_config(benchmark: str) -> ResearchConfig:
    return ResearchConfig(
        start_date="20220101",
        benchmark=benchmark,
        execution_mode="next_open",
        universe_scope="all_a",
        weighting_method="score",
        rebalance_freq="5d",
        enable_market_regime_filter=True,
        regime_ma_window=50,
        regime_vol_window=20,
        regime_max_annual_vol=0.32,
        regime_allowed_quadrants=["trend_up_low_vol", "trend_up_high_vol"],
        enable_style_cap=True,
        max_style_weight=0.50,
    )


def _prepare_signal_frames(stocks: List[str], benchmark: str, start_date: str, end_date: str) -> Dict[str, Any]:
    cfg = _build_base_config(benchmark)
    raw_df_dict = load_daily_from_tq(stocks, start_date, end_date, benchmark=benchmark)
    df_dict, benchmark_close = split_benchmark_from_universe(raw_df_dict, benchmark)
    benchmark_open = raw_df_dict["Open"][benchmark].copy()
    factor_bundle = compute_factors(df_dict)
    regime_state = compute_market_regime_state(benchmark_close, cfg)

    score_none, _, filter_mask = combine_scores_by_state(
        factor_bundle=factor_bundle,
        config=cfg,
        quadrant_series=regime_state["quadrant"],
        state_configs=build_state_configs(cfg, "none"),
    )
    score_v2, _, _ = combine_scores_by_state(
        factor_bundle=factor_bundle,
        config=cfg,
        quadrant_series=regime_state["quadrant"],
        state_configs=build_state_configs(cfg, "up_low_breakout_v2"),
    )
    score_v3, _, _ = combine_scores_by_state(
        factor_bundle=factor_bundle,
        config=cfg,
        quadrant_series=regime_state["quadrant"],
        state_configs=build_state_configs(cfg, "up_low_breakout_v3"),
    )
    forward_excess = build_ml_target(
        close=factor_bundle["raw_inputs"]["Close"],
        benchmark_close=benchmark_close,
        horizon=20,
        execution_mode="next_open",
        open_df=factor_bundle["raw_inputs"]["Open"],
        benchmark_open=benchmark_open,
    )
    return {
        "config": cfg,
        "factor_bundle": factor_bundle,
        "regime_state": regime_state,
        "filter_mask": filter_mask,
        "score_none": score_none,
        "score_v2": score_v2,
        "score_v3": score_v3,
        "forward_excess": forward_excess,
    }


def _focus_compare(
    candidate: Dict[str, Any],
    other: Dict[str, Any],
    regime_state: pd.DataFrame,
    focus_quarter: str,
    focus_quadrant: str,
) -> pd.DataFrame:
    regime_slice = regime_state[["quadrant", "regime_on"]].reset_index()
    regime_slice = regime_slice.rename(columns={regime_slice.columns[0]: "date"})
    merged = candidate["equity"][["date", "excess_return"]].merge(
        other["equity"][["date", "excess_return"]],
        on="date",
        how="inner",
        suffixes=("_candidate", "_other"),
    )
    merged = merged.merge(regime_slice, on="date", how="left")
    merged["quarter"] = merged["date"].dt.to_period("Q").astype(str)
    merged["month"] = merged["date"].dt.to_period("M").astype(str)
    merged = merged.loc[
        (merged["quarter"] == focus_quarter)
        & merged["quadrant"].eq(focus_quadrant)
        & merged["regime_on"].fillna(False)
    ].copy()
    merged["edge"] = merged["excess_return_candidate"] - merged["excess_return_other"]
    return merged.sort_values("edge", ascending=False).reset_index(drop=True)


def _top_day_factor_diff(
    compare_df: pd.DataFrame,
    candidate_weights: pd.DataFrame,
    other_weights: pd.DataFrame,
    zscore_factors: Dict[str, pd.DataFrame],
    top_days: int,
    slot_top_n: int,
) -> pd.DataFrame:
    candidate_weights = candidate_weights.set_index("date").sort_index()
    other_weights = other_weights.set_index("date").sort_index()
    rows: List[Dict[str, Any]] = []
    for row in compare_df.head(top_days).itertuples():
        date = pd.Timestamp(row.date)
        delta = (candidate_weights.loc[date].astype(float) - other_weights.loc[date].astype(float)).sort_values(ascending=False)
        additions = [stock for stock, value in delta.items() if value > 1e-9][:slot_top_n]
        reductions = [stock for stock, value in delta.sort_values(ascending=True).items() if value < -1e-9][:slot_top_n]
        if not additions or not reductions:
            continue
        for factor_name, frame in zscore_factors.items():
            add_value = frame.loc[date, additions].astype(float).dropna()
            reduce_value = frame.loc[date, reductions].astype(float).dropna()
            if add_value.empty or reduce_value.empty:
                continue
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "factor": factor_name,
                    "add_mean": float(add_value.mean()),
                    "reduce_mean": float(reduce_value.mean()),
                    "diff": float(add_value.mean() - reduce_value.mean()),
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(columns=["factor", "mean_diff", "median_diff", "positive_day_count", "sample_days"])
    return (
        detail.groupby("factor", as_index=False)
        .agg(
            mean_diff=("diff", "mean"),
            median_diff=("diff", "median"),
            positive_day_count=("diff", lambda s: int((pd.Series(s) > 0).sum())),
            sample_days=("diff", "count"),
        )
        .sort_values(["mean_diff", "positive_day_count"], ascending=[False, False])
        .reset_index(drop=True)
    )


def _select_slot_logic_factors(
    ma48_vs_ma47_diff: pd.DataFrame,
    ma48_vs_ma50_diff: pd.DataFrame,
    factor_top_k: int,
    shared_only: bool = False,
) -> pd.DataFrame:
    merged = ma48_vs_ma47_diff[["factor", "mean_diff"]].rename(columns={"mean_diff": "mean_diff_vs_ma47"}).merge(
        ma48_vs_ma50_diff[["factor", "mean_diff"]].rename(columns={"mean_diff": "mean_diff_vs_ma50"}),
        on="factor",
        how="outer",
    )
    merged = merged.fillna(0.0)
    merged["shared_positive"] = (merged["mean_diff_vs_ma47"] > 0.0) & (merged["mean_diff_vs_ma50"] > 0.0)
    merged["positive_support"] = merged["mean_diff_vs_ma47"].clip(lower=0.0) + merged["mean_diff_vs_ma50"].clip(lower=0.0)
    merged = merged.sort_values(
        ["shared_positive", "positive_support", "mean_diff_vs_ma47", "mean_diff_vs_ma50"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    selected = merged.loc[merged["positive_support"] > 0.0].copy()
    if shared_only:
        selected = selected.loc[selected["shared_positive"]].copy()
    selected = selected.head(max(int(factor_top_k), 1)).copy()
    if selected.empty:
        return selected
    selected["weight"] = selected["positive_support"] / float(selected["positive_support"].sum())
    return selected.reset_index(drop=True)


def _build_slot_logic_score(
    zscore_factors: Dict[str, pd.DataFrame],
    selected_factors: pd.DataFrame,
    filter_mask: pd.DataFrame,
    score_clip: float,
) -> pd.DataFrame:
    if selected_factors.empty:
        return pd.DataFrame(np.nan, index=filter_mask.index, columns=filter_mask.columns)
    score = pd.DataFrame(0.0, index=filter_mask.index, columns=filter_mask.columns, dtype=float)
    for row in selected_factors.itertuples():
        frame = zscore_factors[str(row.factor)].clip(lower=-abs(float(score_clip)), upper=abs(float(score_clip)))
        score = score.add(frame * float(row.weight), fill_value=0.0)
    return score.where(filter_mask)


def _quarter_rank_ic_table(
    score_map: Dict[str, pd.DataFrame],
    forward_excess: pd.DataFrame,
    regime_state: pd.DataFrame,
    focus_quadrant: str,
) -> pd.DataFrame:
    mask = regime_state["quadrant"].eq(focus_quadrant) & regime_state["regime_on"].fillna(False)
    dates = regime_state.index[mask]
    quarters = sorted({str(dt.to_period("Q")) for dt in dates})
    rows: List[Dict[str, Any]] = []
    for quarter in quarters:
        quarter_dates = [dt for dt in dates if str(dt.to_period("Q")) == quarter]
        row: Dict[str, Any] = {"quarter": quarter, "days": int(len(quarter_dates))}
        for score_name, score_frame in score_map.items():
            values = [_rank_corr(score_frame.loc[dt], forward_excess.loc[dt]) for dt in quarter_dates]
            series = pd.Series(values).dropna()
            row[f"{score_name}_rank_ic"] = float(series.mean()) if not series.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("quarter").reset_index(drop=True)


def _quarter_top_sets(
    candidate_weights: pd.DataFrame,
    other_weights: pd.DataFrame,
    regime_state: pd.DataFrame,
    focus_quadrant: str,
    slot_top_n: int,
) -> Dict[str, Dict[str, Any]]:
    candidate_weights = candidate_weights.set_index("date").sort_index()
    other_weights = other_weights.set_index("date").sort_index()
    regime_subset = regime_state.loc[
        regime_state["quadrant"].eq(focus_quadrant) & regime_state["regime_on"].fillna(False)
    ].copy()
    regime_subset["quarter"] = regime_subset.index.to_period("Q").astype(str)
    quarter_map: Dict[str, Dict[str, Any]] = {}
    for quarter, g in regime_subset.groupby("quarter", sort=True):
        dates = list(g.index)
        cand_avg = candidate_weights.loc[candidate_weights.index.isin(dates)].astype(float).mean(axis=0)
        other_avg = other_weights.loc[other_weights.index.isin(dates)].astype(float).mean(axis=0)
        delta = (cand_avg - other_avg).sort_values(ascending=False)
        additions = [stock for stock, value in delta.items() if value > 1e-9][:slot_top_n]
        reductions = [stock for stock, value in delta.sort_values(ascending=True).items() if value < -1e-9][:slot_top_n]
        quarter_map[quarter] = {
            "dates": dates,
            "additions": additions,
            "reductions": reductions,
        }
    return quarter_map


def _score_percentile_map(score_map: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    return {score_name: score_frame.rank(axis=1, pct=True, method="average") for score_name, score_frame in score_map.items()}


def _quarter_edge_summaries(candidate: Dict[str, Any], other: Dict[str, Any]) -> pd.DataFrame:
    merged = candidate["equity"][["date", "excess_return"]].merge(
        other["equity"][["date", "excess_return"]],
        on="date",
        how="inner",
        suffixes=("_candidate", "_other"),
    )
    merged["quarter"] = merged["date"].dt.to_period("Q").astype(str)
    merged["edge"] = merged["excess_return_candidate"] - merged["excess_return_other"]
    rows: List[Dict[str, Any]] = []
    for quarter, g in merged.groupby("quarter", sort=True):
        rows.append(
            {
                "quarter": quarter,
                "edge_sum": float(g["edge"].sum()),
                "positive_day_count": int((g["edge"] > 0).sum()),
                "negative_day_count": int((g["edge"] < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def _quarter_slot_edge_table(
    comparison_name: str,
    candidate: Dict[str, Any],
    other: Dict[str, Any],
    score_percentiles: Dict[str, pd.DataFrame],
    regime_state: pd.DataFrame,
    focus_quadrant: str,
    slot_top_n: int,
) -> pd.DataFrame:
    quarter_sets = _quarter_top_sets(candidate["target_weights"], other["target_weights"], regime_state, focus_quadrant, slot_top_n)
    edge_summary = _quarter_edge_summaries(candidate, other).set_index("quarter")
    rows: List[Dict[str, Any]] = []
    for quarter, payload in quarter_sets.items():
        additions = list(payload["additions"])
        reductions = list(payload["reductions"])
        if not additions or not reductions:
            continue
        row: Dict[str, Any] = {
            "comparison": comparison_name,
            "quarter": quarter,
            "additions": ", ".join(additions),
            "reductions": ", ".join(reductions),
            "ma48_edge_sum": float(edge_summary["edge_sum"].get(quarter, np.nan)),
        }
        for score_name, pct_frame in score_percentiles.items():
            per_date_values: List[float] = []
            for dt in payload["dates"]:
                add_value = pct_frame.loc[dt, additions].astype(float).dropna()
                reduce_value = pct_frame.loc[dt, reductions].astype(float).dropna()
                if add_value.empty or reduce_value.empty:
                    continue
                per_date_values.append(float(add_value.mean() - reduce_value.mean()))
            series = pd.Series(per_date_values).dropna()
            row[f"{score_name}_slot_edge"] = float(series.mean()) if not series.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("quarter").reset_index(drop=True)


def _rank_ic_summary(quarter_rank_ic: pd.DataFrame, focus_quarter: str, score_names: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for score_name in score_names:
        current = quarter_rank_ic[f"{score_name}_rank_ic"]
        row: Dict[str, Any] = {
            "score_name": score_name,
            "mean_rank_ic": float(current.mean()) if not current.dropna().empty else np.nan,
        }
        for base_name in ("none", "v2"):
            diff = current - quarter_rank_ic[f"{base_name}_rank_ic"]
            positive = diff.clip(lower=0.0)
            other = diff.loc[quarter_rank_ic["quarter"] != focus_quarter]
            other_positive = other.clip(lower=0.0)
            row[f"beats_{base_name}_quarters"] = int((diff > 0).sum())
            row[f"beats_{base_name}_other_quarters"] = int((other > 0).sum())
            row[f"top_positive_vs_{base_name}_share"] = float(positive.max() / positive.sum()) if float(positive.sum()) > 0 else np.nan
            row[f"top_other_positive_vs_{base_name}_share"] = (
                float(other_positive.max() / other_positive.sum()) if float(other_positive.sum()) > 0 else np.nan
            )
            focus_mask = quarter_rank_ic["quarter"] == focus_quarter
            row[f"focus_quarter_vs_{base_name}"] = float(diff.loc[focus_mask].iloc[0]) if focus_mask.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _slot_edge_summary(slot_edge_df: pd.DataFrame, focus_quarter: str, score_names: Iterable[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for comparison in sorted(slot_edge_df["comparison"].unique()):
        comp_df = slot_edge_df.loc[slot_edge_df["comparison"] == comparison].copy()
        for score_name in score_names:
            edge_col = f"{score_name}_slot_edge"
            positive = comp_df[edge_col].clip(lower=0.0)
            other_mask = comp_df["quarter"] != focus_quarter
            rows.append(
                {
                    "comparison": comparison,
                    "score_name": score_name,
                    "focus_quarter_slot_edge": float(comp_df.loc[comp_df["quarter"] == focus_quarter, edge_col].iloc[0])
                    if (comp_df["quarter"] == focus_quarter).any()
                    else np.nan,
                    "positive_slot_edge_quarters": int((comp_df[edge_col] > 0).sum()),
                    "positive_slot_edge_other_quarters": int((comp_df.loc[other_mask, edge_col] > 0).sum()),
                    "positive_slot_edge_on_positive_ma48_edge_other_quarters": int(
                        ((comp_df.loc[other_mask, edge_col] > 0) & (comp_df.loc[other_mask, "ma48_edge_sum"] > 0)).sum()
                    ),
                    "top_positive_slot_edge_share": float(positive.max() / positive.sum()) if float(positive.sum()) > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _summary_row(df: pd.DataFrame, score_name: str) -> Dict[str, Any]:
    row = df.loc[df["score_name"] == score_name]
    return row.iloc[0].to_dict() if not row.empty else {}


def _summary_comp_row(df: pd.DataFrame, comparison: str, score_name: str) -> Dict[str, Any]:
    row = df.loc[(df["comparison"] == comparison) & (df["score_name"] == score_name)]
    return row.iloc[0].to_dict() if not row.empty else {}


def _write_markdown(
    output_path: Path,
    selected_union_factors: pd.DataFrame,
    selected_shared_factors: pd.DataFrame,
    rank_ic_summary: pd.DataFrame,
    slot_edge_summary: pd.DataFrame,
    focus_quarter: str,
) -> None:
    union_row = _summary_row(rank_ic_summary, "slot_logic")
    shared_row = _summary_row(rank_ic_summary, "slot_logic_shared")
    v3_row = _summary_row(rank_ic_summary, "v3")
    ma47_union = _summary_comp_row(slot_edge_summary, "ma48_vs_ma47", "slot_logic")
    ma50_union = _summary_comp_row(slot_edge_summary, "ma48_vs_ma50", "slot_logic")
    ma47_shared = _summary_comp_row(slot_edge_summary, "ma48_vs_ma47", "slot_logic_shared")
    ma50_shared = _summary_comp_row(slot_edge_summary, "ma48_vs_ma50", "slot_logic_shared")

    lines = [
        "# trend_up_low_vol 信号逻辑抽象诊断",
        "",
        "## 核心结论",
        (
            f"- 本轮把 `2025Q3` 的 `ma48` 槽位替换抽象成两种候选信号："
            f"`slot_logic_shared` 只保留双对照共同支持的因子，"
            f"`slot_logic` 则保留更宽的 union 候选。"
        ),
        (
            f"- `slot_logic_shared` 因子为："
            f"{', '.join(f'{row.factor}({float(row.weight):.1%})' for row in selected_shared_factors.itertuples()) if not selected_shared_factors.empty else '无'}"
        ),
        (
            f"- `slot_logic` 因子为："
            f"{', '.join(f'{row.factor}({float(row.weight):.1%})' for row in selected_union_factors.itertuples()) if not selected_union_factors.empty else '无'}"
        ),
        (
            f"- 更克制的 `slot_logic_shared` 相对 `v2` 在非 `{focus_quarter}` 季度只在 "
            f"`{int(shared_row.get('beats_v2_other_quarters', 0))}` 个季度更强；"
            f"对 `ma48_vs_ma47 / ma48_vs_ma50` 的非焦点季度正向槽位复放分别只有 "
            f"`{int(ma47_shared.get('positive_slot_edge_other_quarters', 0))}` / "
            f"`{int(ma50_shared.get('positive_slot_edge_other_quarters', 0))}` 个季度。"
        ),
        (
            f"- 更宽的 `slot_logic` 也没有形成稳定跨季度排序优势：相对 `v2` 仅在 "
            f"`{int(union_row.get('beats_v2_other_quarters', 0))}` 个非 `{focus_quarter}` 季度更强，"
            f"而且这些改善的集中度约 `{_safe_pct(union_row.get('top_other_positive_vs_v2_share'))}`。"
        ),
        (
            f"- 现有 `v3` 也没有形成稳定跨季度优势：相对 `v2` 仅在 "
            f"`{int(v3_row.get('beats_v2_other_quarters', 0))}` 个非焦点季度更强。"
        ),
        (
            f"- 对 `ma48_vs_ma47` / `ma48_vs_ma50` 两组槽位复放，宽口径 `slot_logic` 在非 `{focus_quarter}` 季度里"
            f"分别只在 `{int(ma47_union.get('positive_slot_edge_other_quarters', 0))}` / "
            f"`{int(ma50_union.get('positive_slot_edge_other_quarters', 0))}` 个季度仍保持正向槽位边际。"
        ),
        "",
        "## 判断",
        "- 如果一个信号逻辑想支撑 `ma47/48` 分支晋级，它至少要同时满足两件事：",
        "1. 对未来超额排序不是只在 `2025Q3` 有效。",
        "2. 在别的季度里也能继续解释 `ma48` 的加减仓槽位。",
        "- 当前 `shared` 和 `union` 两种抽象都没有把这两条同时做出来，所以这条分支暂时不该继续晋级执行端。",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path("daily_research/output") / (
        f"trend_up_low_vol_signal_logic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ma47 = _load_run(Path(args.ma47_run), "ma47_baseline")
    ma48 = _load_run(Path(args.ma48_run), "ma48_baseline")
    ma50 = _load_run(Path(args.ma50_run), "ma50_baseline")
    runs = [ma47, ma48, ma50]

    stocks = _collect_union_stocks(runs)
    end_date = _resolve_end_date(args.end_date, runs)
    prepared = _prepare_signal_frames(stocks, args.benchmark, args.start_date, end_date)
    zscore_factors = prepared["factor_bundle"]["zscore_factors"]
    regime_state = prepared["regime_state"]

    ma48_vs_ma47_compare = _focus_compare(ma48, ma47, regime_state, args.focus_quarter, args.focus_quadrant)
    ma48_vs_ma50_compare = _focus_compare(ma48, ma50, regime_state, args.focus_quarter, args.focus_quadrant)
    ma48_vs_ma47_compare.to_csv(output_dir / "focus_quarter_ma48_vs_ma47_daily.csv", index=False, encoding="utf-8-sig")
    ma48_vs_ma50_compare.to_csv(output_dir / "focus_quarter_ma48_vs_ma50_daily.csv", index=False, encoding="utf-8-sig")

    ma48_vs_ma47_factor_diff = _top_day_factor_diff(
        compare_df=ma48_vs_ma47_compare,
        candidate_weights=ma48["target_weights"],
        other_weights=ma47["target_weights"],
        zscore_factors=zscore_factors,
        top_days=args.top_days,
        slot_top_n=args.slot_top_n,
    )
    ma48_vs_ma50_factor_diff = _top_day_factor_diff(
        compare_df=ma48_vs_ma50_compare,
        candidate_weights=ma48["target_weights"],
        other_weights=ma50["target_weights"],
        zscore_factors=zscore_factors,
        top_days=args.top_days,
        slot_top_n=args.slot_top_n,
    )
    ma48_vs_ma47_factor_diff.to_csv(output_dir / "ma48_vs_ma47_topday_factor_diff.csv", index=False, encoding="utf-8-sig")
    ma48_vs_ma50_factor_diff.to_csv(output_dir / "ma48_vs_ma50_topday_factor_diff.csv", index=False, encoding="utf-8-sig")

    selected_union_factors = _select_slot_logic_factors(
        ma48_vs_ma47_diff=ma48_vs_ma47_factor_diff,
        ma48_vs_ma50_diff=ma48_vs_ma50_factor_diff,
        factor_top_k=args.factor_top_k,
        shared_only=False,
    )
    selected_shared_factors = _select_slot_logic_factors(
        ma48_vs_ma47_diff=ma48_vs_ma47_factor_diff,
        ma48_vs_ma50_diff=ma48_vs_ma50_factor_diff,
        factor_top_k=args.factor_top_k,
        shared_only=True,
    )
    selected_union_factors.to_csv(output_dir / "selected_slot_logic_factors.csv", index=False, encoding="utf-8-sig")
    selected_shared_factors.to_csv(output_dir / "selected_slot_logic_shared_factors.csv", index=False, encoding="utf-8-sig")

    slot_logic = _build_slot_logic_score(
        zscore_factors=zscore_factors,
        selected_factors=selected_union_factors,
        filter_mask=prepared["filter_mask"],
        score_clip=float(prepared["config"].score_clip),
    )
    slot_logic_shared = _build_slot_logic_score(
        zscore_factors=zscore_factors,
        selected_factors=selected_shared_factors,
        filter_mask=prepared["filter_mask"],
        score_clip=float(prepared["config"].score_clip),
    )

    score_map = {
        "none": prepared["score_none"],
        "v2": prepared["score_v2"],
        "v3": prepared["score_v3"],
        "slot_logic_shared": slot_logic_shared,
        "slot_logic": slot_logic,
    }
    quarter_rank_ic = _quarter_rank_ic_table(score_map, prepared["forward_excess"], regime_state, args.focus_quadrant)
    quarter_rank_ic.to_csv(output_dir / "quarter_rank_ic.csv", index=False, encoding="utf-8-sig")

    score_percentiles = _score_percentile_map(score_map)
    slot_edge_df = pd.concat(
        [
            _quarter_slot_edge_table(
                comparison_name="ma48_vs_ma47",
                candidate=ma48,
                other=ma47,
                score_percentiles=score_percentiles,
                regime_state=regime_state,
                focus_quadrant=args.focus_quadrant,
                slot_top_n=args.slot_top_n,
            ),
            _quarter_slot_edge_table(
                comparison_name="ma48_vs_ma50",
                candidate=ma48,
                other=ma50,
                score_percentiles=score_percentiles,
                regime_state=regime_state,
                focus_quadrant=args.focus_quadrant,
                slot_top_n=args.slot_top_n,
            ),
        ],
        ignore_index=True,
    )
    slot_edge_df.to_csv(output_dir / "quarter_slot_edge.csv", index=False, encoding="utf-8-sig")

    rank_ic_summary = _rank_ic_summary(
        quarter_rank_ic,
        args.focus_quarter,
        score_names=["v2", "v3", "slot_logic_shared", "slot_logic"],
    )
    slot_edge_summary = _slot_edge_summary(
        slot_edge_df,
        args.focus_quarter,
        score_names=["none", "v2", "v3", "slot_logic_shared", "slot_logic"],
    )
    rank_ic_summary.to_csv(output_dir / "rank_ic_summary.csv", index=False, encoding="utf-8-sig")
    slot_edge_summary.to_csv(output_dir / "slot_edge_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "focus_quarter": args.focus_quarter,
        "focus_quadrant": args.focus_quadrant,
        "benchmark": args.benchmark,
        "base_regime_ma_window": 50,
        "base_regime_vol_window": 20,
        "base_regime_max_annual_vol": 0.32,
        "selected_slot_logic_factors": selected_union_factors.to_dict(orient="records"),
        "selected_slot_logic_shared_factors": selected_shared_factors.to_dict(orient="records"),
        "rank_ic_summary": rank_ic_summary.to_dict(orient="records"),
        "slot_edge_summary": slot_edge_summary.to_dict(orient="records"),
    }
    (output_dir / "signal_logic_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(
        output_dir / "signal_logic_report.md",
        selected_union_factors=selected_union_factors,
        selected_shared_factors=selected_shared_factors,
        rank_ic_summary=rank_ic_summary,
        slot_edge_summary=slot_edge_summary,
        focus_quarter=args.focus_quarter,
    )

    print(f"output: {output_dir}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
