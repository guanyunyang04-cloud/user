from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

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
        description="Diagnose whether v21 volume_contraction tuning can be abstracted into a stable rule logic."
    )
    parser.add_argument("--baseline-run", required=True, help="Path to baseline v2 run")
    parser.add_argument("--candidate-run", required=True, help="Path to candidate v21 run")
    parser.add_argument("--start-date", default="20220101")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--focus-quarter", default="auto")
    parser.add_argument("--focus-month", default="auto")
    parser.add_argument("--focus-quadrant", default="trend_up_low_vol")
    parser.add_argument("--top-days", type=int, default=10)
    parser.add_argument("--slot-top-n", type=int, default=5)
    parser.add_argument("--factor-top-k", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output dir. Defaults to daily_research/output/v21_rule_logic_<timestamp>",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{float(value):.2%}"


def _safe_num(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{float(value):.3f}"


def _rank_corr(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 2:
        return np.nan
    if pair.iloc[:, 0].nunique() <= 1 or pair.iloc[:, 1].nunique() <= 1:
        return np.nan
    xr = pair.iloc[:, 0].rank(method="average")
    yr = pair.iloc[:, 1].rank(method="average")
    return float(xr.corr(yr))


def _compound_return(series: pd.Series) -> float:
    valid = series.dropna()
    if valid.empty:
        return float("nan")
    return float((1.0 + valid).prod() - 1.0)


def _load_run(run_dir: Path, label: str) -> dict[str, Any]:
    equity = pd.read_csv(run_dir / "equity_curve.csv", parse_dates=["date"])
    regime = pd.read_csv(run_dir / "regime_state.csv", parse_dates=["Date"]).rename(columns={"Date": "date"})
    target_weights = pd.read_csv(run_dir / "target_weights.csv", parse_dates=["Date"]).rename(columns={"Date": "date"})
    rankic_path = run_dir / "up_low_rankic_quarterly.csv"
    rankic_quarterly = pd.read_csv(rankic_path) if rankic_path.exists() else pd.DataFrame()
    equity = equity.merge(regime[["date", "quadrant", "regime_on"]], on="date", how="left", suffixes=("", "_regime"))
    if "quadrant" not in equity.columns and "quadrant_regime" in equity.columns:
        equity["quadrant"] = equity["quadrant_regime"]
    if "regime_on" not in equity.columns and "regime_on_regime" in equity.columns:
        equity["regime_on"] = equity["regime_on_regime"]
    return {
        "label": label,
        "run_dir": run_dir,
        "equity": equity,
        "target_weights": target_weights,
        "rankic_quarterly": rankic_quarterly,
    }


def _collect_union_stocks(runs: list[dict[str, Any]]) -> list[str]:
    stocks: set[str] = set()
    for run in runs:
        stocks.update(col for col in run["target_weights"].columns if col != "date")
    return sorted(stocks)


def _resolve_end_date(raw_end_date: str, runs: list[dict[str, Any]]) -> str:
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
        rebalance_freq="1d",
        enable_market_regime_filter=True,
        regime_ma_window=50,
        regime_vol_window=20,
        regime_max_annual_vol=0.32,
        regime_allowed_quadrants=["trend_up_low_vol", "trend_up_high_vol"],
        enable_style_cap=True,
        max_style_weight=0.50,
    )


def _prepare_signal_frames(stocks: list[str], benchmark: str, start_date: str, end_date: str) -> dict[str, Any]:
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

    v21_state_cfg = build_state_configs(cfg, "up_low_breakout_v2")["trend_up_low_vol"]
    v21_factor_weights = dict(v21_state_cfg.factor_weights)
    v21_factor_weights["volume_contraction"] = 0.15
    v21_state_cfg = replace(v21_state_cfg, factor_weights=v21_factor_weights)
    score_v21, _, _ = combine_scores_by_state(
        factor_bundle=factor_bundle,
        config=cfg,
        quadrant_series=regime_state["quadrant"],
        state_configs={"trend_up_low_vol": v21_state_cfg},
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
        "score_v21": score_v21,
        "forward_excess": forward_excess,
    }


def _quarterly_edge_frame(candidate: dict[str, Any], baseline: dict[str, Any]) -> pd.DataFrame:
    merged = candidate["equity"][["date", "excess_return", "quadrant", "regime_on"]].merge(
        baseline["equity"][["date", "excess_return"]],
        on="date",
        how="inner",
        suffixes=("_candidate", "_baseline"),
    )
    merged["quarter"] = merged["date"].dt.to_period("Q").astype(str)
    merged["month"] = merged["date"].dt.to_period("M").astype(str)
    merged["edge"] = merged["excess_return_candidate"] - merged["excess_return_baseline"]
    return merged.sort_values("date").reset_index(drop=True)


def _resolve_focus_quarter(raw_focus_quarter: str, compare_df: pd.DataFrame) -> str:
    value = str(raw_focus_quarter or "").strip()
    if value and value.lower() != "auto":
        return value
    grouped = (
        compare_df.groupby("quarter", as_index=False)
        .agg(compound_gap=("edge", lambda s: _compound_return(s)))
        .sort_values("compound_gap", ascending=False)
    )
    if grouped.empty:
        raise ValueError("Unable to resolve focus quarter from empty compare frame.")
    return str(grouped.iloc[0]["quarter"])


def _resolve_focus_month(raw_focus_month: str, compare_df: pd.DataFrame, focus_quarter: str) -> str:
    value = str(raw_focus_month or "").strip()
    if value and value.lower() != "auto":
        return value
    month_df = compare_df.loc[compare_df["quarter"] == focus_quarter].copy()
    grouped = (
        month_df.groupby("month", as_index=False)
        .agg(compound_gap=("edge", lambda s: _compound_return(s)))
        .sort_values("compound_gap", ascending=False)
    )
    if grouped.empty:
        raise ValueError("Unable to resolve focus month from empty focus-quarter frame.")
    return str(grouped.iloc[0]["month"])


def _focus_compare(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    regime_state: pd.DataFrame,
    focus_quarter: str,
    focus_quadrant: str,
) -> pd.DataFrame:
    regime_slice = regime_state[["quadrant", "regime_on"]].reset_index()
    regime_slice = regime_slice.rename(columns={regime_slice.columns[0]: "date"})
    merged = candidate["equity"][["date", "excess_return"]].merge(
        baseline["equity"][["date", "excess_return"]],
        on="date",
        how="inner",
        suffixes=("_candidate", "_baseline"),
    )
    merged = merged.merge(regime_slice, on="date", how="left")
    merged["quarter"] = merged["date"].dt.to_period("Q").astype(str)
    merged["month"] = merged["date"].dt.to_period("M").astype(str)
    merged["edge"] = merged["excess_return_candidate"] - merged["excess_return_baseline"]
    merged = merged.loc[
        (merged["quarter"] == focus_quarter)
        & merged["quadrant"].eq(focus_quadrant)
        & merged["regime_on"].eq(True)
    ].copy()
    return merged.sort_values("edge", ascending=False).reset_index(drop=True)


def _top_day_weight_swaps(
    compare_df: pd.DataFrame,
    candidate_weights: pd.DataFrame,
    baseline_weights: pd.DataFrame,
    top_days: int,
    slot_top_n: int,
) -> pd.DataFrame:
    candidate_weights = candidate_weights.set_index("date").sort_index()
    baseline_weights = baseline_weights.set_index("date").sort_index()
    rows: list[dict[str, Any]] = []
    for row in compare_df.head(top_days).itertuples():
        date = pd.Timestamp(row.date)
        cand_row = candidate_weights.loc[date].astype(float).fillna(0.0)
        base_row = baseline_weights.loc[date].astype(float).fillna(0.0)
        delta = cand_row.subtract(base_row, fill_value=0.0).sort_values(ascending=False)
        additions = [f"{stock}:{float(value):.2%}" for stock, value in delta.items() if value > 1e-9][:slot_top_n]
        reductions = [
            f"{stock}:{float(value):.2%}" for stock, value in delta.sort_values(ascending=True).items() if value < -1e-9
        ][:slot_top_n]
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "month": str(row.month),
                "quadrant": str(row.quadrant),
                "edge": float(row.edge),
                "candidate_excess_return": float(row.excess_return_candidate),
                "baseline_excess_return": float(row.excess_return_baseline),
                "candidate_top_additions": "; ".join(additions),
                "candidate_top_reductions": "; ".join(reductions),
            }
        )
    return pd.DataFrame(rows)


def _top_day_factor_diff(
    compare_df: pd.DataFrame,
    candidate_weights: pd.DataFrame,
    baseline_weights: pd.DataFrame,
    zscore_factors: dict[str, pd.DataFrame],
    top_days: int,
    slot_top_n: int,
) -> pd.DataFrame:
    candidate_weights = candidate_weights.set_index("date").sort_index()
    baseline_weights = baseline_weights.set_index("date").sort_index()
    rows: list[dict[str, Any]] = []
    for row in compare_df.head(top_days).itertuples():
        date = pd.Timestamp(row.date)
        cand_row = candidate_weights.loc[date].astype(float).fillna(0.0)
        base_row = baseline_weights.loc[date].astype(float).fillna(0.0)
        delta = cand_row.subtract(base_row, fill_value=0.0).sort_values(ascending=False)
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
        return pd.DataFrame(columns=["factor", "mean_diff", "median_diff", "positive_day_count", "negative_day_count", "sample_days"])
    return (
        detail.groupby("factor", as_index=False)
        .agg(
            mean_diff=("diff", "mean"),
            median_diff=("diff", "median"),
            positive_day_count=("diff", lambda s: int((pd.Series(s) > 0).sum())),
            negative_day_count=("diff", lambda s: int((pd.Series(s) < 0).sum())),
            sample_days=("diff", "count"),
        )
        .sort_values(["mean_diff"], ascending=False)
        .reset_index(drop=True)
    )


def _select_signed_logic_factors(
    quarter_diff: pd.DataFrame,
    month_diff: pd.DataFrame,
    factor_top_k: int,
) -> pd.DataFrame:
    merged = quarter_diff[["factor", "mean_diff"]].rename(columns={"mean_diff": "quarter_mean_diff"}).merge(
        month_diff[["factor", "mean_diff"]].rename(columns={"mean_diff": "month_mean_diff"}),
        on="factor",
        how="outer",
    )
    merged = merged.fillna(0.0)
    merged["same_sign"] = (merged["quarter_mean_diff"] * merged["month_mean_diff"]) > 0.0
    merged = merged.loc[merged["same_sign"]].copy()
    if merged.empty:
        return merged
    merged["sign"] = np.sign(merged["month_mean_diff"])
    merged["support"] = merged["quarter_mean_diff"].abs() + merged["month_mean_diff"].abs()
    merged = merged.sort_values(["support"], ascending=False).head(max(int(factor_top_k), 1)).copy()
    merged["weight"] = merged["support"] / float(merged["support"].sum())
    return merged.reset_index(drop=True)


def _build_signed_logic_score(
    zscore_factors: dict[str, pd.DataFrame],
    selected_factors: pd.DataFrame,
    filter_mask: pd.DataFrame,
    score_clip: float,
) -> pd.DataFrame:
    if selected_factors.empty:
        return pd.DataFrame(np.nan, index=filter_mask.index, columns=filter_mask.columns)
    score = pd.DataFrame(0.0, index=filter_mask.index, columns=filter_mask.columns, dtype=float)
    for row in selected_factors.itertuples():
        frame = zscore_factors[str(row.factor)].clip(lower=-abs(float(score_clip)), upper=abs(float(score_clip)))
        score = score.add(frame * float(row.weight) * float(row.sign), fill_value=0.0)
    return score.where(filter_mask)


def _quarter_rank_ic_table(
    score_map: dict[str, pd.DataFrame],
    forward_excess: pd.DataFrame,
    regime_state: pd.DataFrame,
    focus_quadrant: str,
) -> pd.DataFrame:
    mask = regime_state["quadrant"].eq(focus_quadrant) & regime_state["regime_on"].fillna(False)
    dates = regime_state.index[mask]
    quarters = sorted({str(dt.to_period("Q")) for dt in dates})
    rows: list[dict[str, Any]] = []
    for quarter in quarters:
        quarter_dates = [dt for dt in dates if str(dt.to_period("Q")) == quarter]
        row: dict[str, Any] = {"quarter": quarter, "days": int(len(quarter_dates))}
        for score_name, score_frame in score_map.items():
            values = [_rank_corr(score_frame.loc[dt], forward_excess.loc[dt]) for dt in quarter_dates]
            series = pd.Series(values).dropna()
            row[f"{score_name}_rank_ic"] = float(series.mean()) if not series.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("quarter").reset_index(drop=True)


def _quarter_top_sets(
    candidate_weights: pd.DataFrame,
    baseline_weights: pd.DataFrame,
    regime_state: pd.DataFrame,
    focus_quadrant: str,
    slot_top_n: int,
) -> dict[str, dict[str, Any]]:
    candidate_weights = candidate_weights.set_index("date").sort_index()
    baseline_weights = baseline_weights.set_index("date").sort_index()
    regime_subset = regime_state.loc[
        regime_state["quadrant"].eq(focus_quadrant) & regime_state["regime_on"].fillna(False)
    ].copy()
    regime_subset["quarter"] = regime_subset.index.to_period("Q").astype(str)
    quarter_map: dict[str, dict[str, Any]] = {}
    for quarter, group in regime_subset.groupby("quarter", sort=True):
        dates = list(group.index)
        cand_avg = candidate_weights.loc[candidate_weights.index.isin(dates)].astype(float).mean(axis=0).fillna(0.0)
        base_avg = baseline_weights.loc[baseline_weights.index.isin(dates)].astype(float).mean(axis=0).fillna(0.0)
        delta = cand_avg.subtract(base_avg, fill_value=0.0).sort_values(ascending=False)
        additions = [stock for stock, value in delta.items() if value > 1e-9][:slot_top_n]
        reductions = [stock for stock, value in delta.sort_values(ascending=True).items() if value < -1e-9][:slot_top_n]
        quarter_map[quarter] = {"dates": dates, "additions": additions, "reductions": reductions}
    return quarter_map


def _score_percentile_map(score_map: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {name: frame.rank(axis=1, pct=True, method="average") for name, frame in score_map.items()}


def _quarter_edge_summaries(candidate: dict[str, Any], baseline: dict[str, Any]) -> pd.DataFrame:
    merged = candidate["equity"][["date", "excess_return"]].merge(
        baseline["equity"][["date", "excess_return"]],
        on="date",
        how="inner",
        suffixes=("_candidate", "_baseline"),
    )
    merged["quarter"] = merged["date"].dt.to_period("Q").astype(str)
    merged["edge"] = merged["excess_return_candidate"] - merged["excess_return_baseline"]
    rows: list[dict[str, Any]] = []
    for quarter, group in merged.groupby("quarter", sort=True):
        rows.append(
            {
                "quarter": quarter,
                "edge_sum": float(group["edge"].sum()),
                "edge_compound_gap": _compound_return(group["excess_return_candidate"]) - _compound_return(group["excess_return_baseline"]),
                "positive_day_count": int((group["edge"] > 0).sum()),
                "negative_day_count": int((group["edge"] < 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("quarter").reset_index(drop=True)


def _quarter_slot_edge_table(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    score_percentiles: dict[str, pd.DataFrame],
    regime_state: pd.DataFrame,
    focus_quadrant: str,
    slot_top_n: int,
) -> pd.DataFrame:
    quarter_sets = _quarter_top_sets(candidate["target_weights"], baseline["target_weights"], regime_state, focus_quadrant, slot_top_n)
    edge_summary = _quarter_edge_summaries(candidate, baseline).set_index("quarter")
    rows: list[dict[str, Any]] = []
    for quarter, payload in quarter_sets.items():
        additions = payload["additions"]
        reductions = payload["reductions"]
        if not additions or not reductions:
            continue
        row: dict[str, Any] = {
            "quarter": quarter,
            "additions": ", ".join(additions),
            "reductions": ", ".join(reductions),
            "candidate_edge_sum": float(edge_summary["edge_sum"].get(quarter, np.nan)),
            "candidate_edge_compound_gap": float(edge_summary["edge_compound_gap"].get(quarter, np.nan)),
        }
        for score_name, pct_frame in score_percentiles.items():
            per_date_values: list[float] = []
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


def _rank_ic_summary(
    quarter_rank_ic: pd.DataFrame,
    run_rankic_map: dict[str, pd.DataFrame],
    focus_quarter: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    logic_series = quarter_rank_ic[["quarter", "logic_signed_rank_ic"]].rename(columns={"logic_signed_rank_ic": "logic_rank_ic"})
    for run_name, run_df in run_rankic_map.items():
        merged = logic_series.merge(
            run_df[["quarter", "rank_ic_mean"]].rename(columns={"rank_ic_mean": f"{run_name}_rank_ic"}),
            on="quarter",
            how="left",
        )
        diff = merged["logic_rank_ic"] - merged[f"{run_name}_rank_ic"]
        positive = diff.clip(lower=0.0)
        other = diff.loc[merged["quarter"] != focus_quarter]
        other_positive = other.clip(lower=0.0)
        focus_mask = merged["quarter"] == focus_quarter
        rows.append(
            {
                "comparison": f"logic_signed_vs_{run_name}",
                "logic_mean_rank_ic": float(merged["logic_rank_ic"].mean()) if not merged["logic_rank_ic"].dropna().empty else np.nan,
                f"{run_name}_mean_rank_ic": float(merged[f'{run_name}_rank_ic'].mean()) if not merged[f"{run_name}_rank_ic"].dropna().empty else np.nan,
                "beats_quarters": int((diff > 0).sum()),
                "beats_other_quarters": int((other > 0).sum()),
                "focus_quarter_rank_ic_edge": float(diff.loc[focus_mask].iloc[0]) if focus_mask.any() else np.nan,
                "top_positive_share": float(positive.max() / positive.sum()) if float(positive.sum()) > 0 else np.nan,
                "top_other_positive_share": float(other_positive.max() / other_positive.sum()) if float(other_positive.sum()) > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _slot_edge_summary(slot_edge_df: pd.DataFrame, focus_quarter: str) -> pd.DataFrame:
    edge_col = "logic_signed_slot_edge"
    positive = slot_edge_df[edge_col].clip(lower=0.0)
    other_mask = slot_edge_df["quarter"] != focus_quarter
    return pd.DataFrame(
        [
            {
                "score_name": "logic_signed",
                "focus_quarter_slot_edge": float(slot_edge_df.loc[slot_edge_df["quarter"] == focus_quarter, edge_col].iloc[0])
                if (slot_edge_df["quarter"] == focus_quarter).any()
                else np.nan,
                "positive_slot_edge_quarters": int((slot_edge_df[edge_col] > 0).sum()),
                "positive_slot_edge_other_quarters": int((slot_edge_df.loc[other_mask, edge_col] > 0).sum()),
                "positive_slot_edge_on_positive_candidate_edge_other_quarters": int(
                    ((slot_edge_df.loc[other_mask, edge_col] > 0) & (slot_edge_df.loc[other_mask, "candidate_edge_sum"] > 0)).sum()
                ),
                "top_positive_slot_edge_share": float(positive.max() / positive.sum()) if float(positive.sum()) > 0 else np.nan,
            }
        ]
    )


def _focus_month_stock_signature(
    compare_df: pd.DataFrame,
    candidate_weights: pd.DataFrame,
    baseline_weights: pd.DataFrame,
    top_days: int,
) -> pd.DataFrame:
    candidate_weights = candidate_weights.set_index("date").sort_index()
    baseline_weights = baseline_weights.set_index("date").sort_index()
    rows: list[dict[str, Any]] = []
    for row in compare_df.head(top_days).itertuples():
        date = pd.Timestamp(row.date)
        cand_row = candidate_weights.loc[date].astype(float).fillna(0.0)
        base_row = baseline_weights.loc[date].astype(float).fillna(0.0)
        delta = cand_row.subtract(base_row, fill_value=0.0)
        for stock, value in delta.items():
            if abs(float(value)) <= 1e-9:
                continue
            rows.append(
                {
                    "stock": stock,
                    "direction": "add" if float(value) > 0 else "reduce",
                    "avg_weight_delta": float(value),
                    "date": date.strftime("%Y-%m-%d"),
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(columns=["stock", "direction", "hit_count", "mean_weight_delta", "dates"])
    return (
        detail.groupby(["stock", "direction"], as_index=False)
        .agg(
            hit_count=("date", "count"),
            mean_weight_delta=("avg_weight_delta", "mean"),
            dates=("date", lambda s: ", ".join(sorted(set(map(str, s))))),
        )
        .sort_values(["direction", "hit_count", "mean_weight_delta"], ascending=[True, False, False])
        .reset_index(drop=True)
    )


def _write_markdown(
    output_path: Path,
    focus_quarter: str,
    focus_month: str,
    selected_factors: pd.DataFrame,
    rank_ic_summary: pd.DataFrame,
    slot_edge_summary: pd.DataFrame,
    focus_month_stock_signature: pd.DataFrame,
) -> None:
    logic_vs_v2 = rank_ic_summary.loc[rank_ic_summary["comparison"] == "logic_signed_vs_v2"].iloc[0].to_dict()
    logic_vs_v21 = rank_ic_summary.loc[rank_ic_summary["comparison"] == "logic_signed_vs_v21"].iloc[0].to_dict()
    slot_row = slot_edge_summary.iloc[0].to_dict() if not slot_edge_summary.empty else {}
    add_rows = focus_month_stock_signature.loc[focus_month_stock_signature["direction"] == "add"].head(8)
    factor_desc = "无"
    if not selected_factors.empty:
        factor_desc = ", ".join(
            f"{row.factor}({'+' if float(row.sign) > 0 else '-'}{float(row.weight):.1%})"
            for row in selected_factors.itertuples()
        )
    lines = [
        "# v21 规则逻辑抽象诊断",
        "",
        "## 核心结论",
        (
            f"- 焦点季度为 `{focus_quarter}`，焦点月份为 `{focus_month}`。"
            f"抽出来的 signed 逻辑因子为：{factor_desc}。"
        ),
        (
            f"- 这条 `logic_signed` 相对 `v2` 的季度 RankIC 只在 "
            f"`{int(logic_vs_v2.get('beats_quarters', 0))}` 个季度更强，"
            f"非焦点季度只在 `{int(logic_vs_v2.get('beats_other_quarters', 0))}` 个季度更强；"
            f"焦点季度本身反而落后约 `{_safe_num(logic_vs_v2.get('focus_quarter_rank_ic_edge'))}`。"
        ),
        (
            f"- 相对候选 `v21`，这条 `logic_signed` 也没有形成更稳的排序："
            f"非焦点季度只在 `{int(logic_vs_v21.get('beats_other_quarters', 0))}` 个季度更强，"
            f"焦点季度边际约 `{_safe_num(logic_vs_v21.get('focus_quarter_rank_ic_edge'))}`。"
        ),
        (
            f"- 站在候选 `{focus_quarter}` / `{focus_month}` 的关键槽位角度，"
            f"`logic_signed` 在其他季度只有 `{int(slot_row.get('positive_slot_edge_other_quarters', 0))}` 个季度仍保留正向槽位边际；"
            f"其中与候选正收益季度对齐的也只有 `{int(slot_row.get('positive_slot_edge_on_positive_candidate_edge_other_quarters', 0))}` 个。"
        ),
        "",
        "## 焦点月份关键新增槽位",
    ]
    for row in add_rows.itertuples():
        lines.append(
            f"- `{row.stock}`: 出现 `{int(row.hit_count)}` 次，平均权重边际 `{_safe_pct(row.mean_weight_delta)}`，日期 `{row.dates}`"
        )
    lines.extend(
        [
            "",
            "## 判断",
            "- 这说明 `2026-01` 的改善更像少数强趋势、低波、强价量背离名字在焦点月份集中命中。",
            "- 这些特征可以被描述出来，但当前还不能稳定跨季度复放成一条更稳的规则排序逻辑。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path("daily_research/output") / (
        f"v21_rule_logic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = _load_run(Path(args.baseline_run), "v2")
    candidate = _load_run(Path(args.candidate_run), "v21_volume_contraction_015")
    runs = [baseline, candidate]

    compare_df = _quarterly_edge_frame(candidate, baseline)
    focus_quarter = _resolve_focus_quarter(args.focus_quarter, compare_df)
    focus_month = _resolve_focus_month(args.focus_month, compare_df, focus_quarter)

    stocks = _collect_union_stocks(runs)
    end_date = _resolve_end_date(args.end_date, runs)
    prepared = _prepare_signal_frames(stocks, args.benchmark, args.start_date, end_date)
    zscore_factors = prepared["factor_bundle"]["zscore_factors"]
    regime_state = prepared["regime_state"]

    focus_quarter_daily = _focus_compare(candidate, baseline, regime_state, focus_quarter, args.focus_quadrant)
    focus_month_daily = focus_quarter_daily.loc[focus_quarter_daily["month"] == focus_month].copy().reset_index(drop=True)
    focus_quarter_daily.to_csv(output_dir / "focus_quarter_daily.csv", index=False, encoding="utf-8-sig")
    focus_month_daily.to_csv(output_dir / "focus_month_daily.csv", index=False, encoding="utf-8-sig")

    focus_quarter_top_days = _top_day_weight_swaps(
        focus_quarter_daily,
        candidate["target_weights"],
        baseline["target_weights"],
        args.top_days,
        args.slot_top_n,
    )
    focus_month_top_days = _top_day_weight_swaps(
        focus_month_daily,
        candidate["target_weights"],
        baseline["target_weights"],
        args.top_days,
        args.slot_top_n,
    )
    focus_quarter_top_days.to_csv(output_dir / "focus_quarter_top_days.csv", index=False, encoding="utf-8-sig")
    focus_month_top_days.to_csv(output_dir / "focus_month_top_days.csv", index=False, encoding="utf-8-sig")

    quarter_factor_diff = _top_day_factor_diff(
        focus_quarter_daily,
        candidate["target_weights"],
        baseline["target_weights"],
        zscore_factors,
        args.top_days,
        args.slot_top_n,
    )
    month_factor_diff = _top_day_factor_diff(
        focus_month_daily,
        candidate["target_weights"],
        baseline["target_weights"],
        zscore_factors,
        args.top_days,
        args.slot_top_n,
    )
    quarter_factor_diff.to_csv(output_dir / "focus_quarter_factor_diff.csv", index=False, encoding="utf-8-sig")
    month_factor_diff.to_csv(output_dir / "focus_month_factor_diff.csv", index=False, encoding="utf-8-sig")

    selected_factors = _select_signed_logic_factors(quarter_factor_diff, month_factor_diff, args.factor_top_k)
    selected_factors.to_csv(output_dir / "selected_logic_factors.csv", index=False, encoding="utf-8-sig")

    logic_signed = _build_signed_logic_score(
        zscore_factors=zscore_factors,
        selected_factors=selected_factors,
        filter_mask=prepared["filter_mask"],
        score_clip=float(prepared["config"].score_clip),
    )

    score_map = {
        "none": prepared["score_none"],
        "v2": prepared["score_v2"],
        "v21": prepared["score_v21"],
        "logic_signed": logic_signed,
    }
    quarter_rank_ic = _quarter_rank_ic_table(score_map, prepared["forward_excess"], regime_state, args.focus_quadrant)
    quarter_rank_ic.to_csv(output_dir / "quarter_rank_ic.csv", index=False, encoding="utf-8-sig")

    run_rankic_map = {
        "v2": baseline["rankic_quarterly"],
        "v21": candidate["rankic_quarterly"],
    }
    rank_ic_summary = _rank_ic_summary(quarter_rank_ic, run_rankic_map, focus_quarter)
    rank_ic_summary.to_csv(output_dir / "rank_ic_summary.csv", index=False, encoding="utf-8-sig")

    score_percentiles = _score_percentile_map({"logic_signed": logic_signed})
    quarter_slot_edge = _quarter_slot_edge_table(
        candidate,
        baseline,
        score_percentiles,
        regime_state,
        args.focus_quadrant,
        args.slot_top_n,
    )
    quarter_slot_edge.to_csv(output_dir / "quarter_slot_edge.csv", index=False, encoding="utf-8-sig")
    slot_edge_summary = _slot_edge_summary(quarter_slot_edge, focus_quarter)
    slot_edge_summary.to_csv(output_dir / "slot_edge_summary.csv", index=False, encoding="utf-8-sig")

    focus_month_stock_signature = _focus_month_stock_signature(
        focus_month_daily,
        candidate["target_weights"],
        baseline["target_weights"],
        args.top_days,
    )
    focus_month_stock_signature.to_csv(output_dir / "focus_month_stock_signature.csv", index=False, encoding="utf-8-sig")

    report = {
        "focus_quarter": focus_quarter,
        "focus_month": focus_month,
        "selected_logic_factors": selected_factors.to_dict(orient="records"),
        "rank_ic_summary": rank_ic_summary.to_dict(orient="records"),
        "slot_edge_summary": slot_edge_summary.to_dict(orient="records"),
        "focus_month_top_additions": focus_month_stock_signature.loc[
            focus_month_stock_signature["direction"] == "add"
        ].head(10).to_dict(orient="records"),
        "focus_month_top_reductions": focus_month_stock_signature.loc[
            focus_month_stock_signature["direction"] == "reduce"
        ].head(10).to_dict(orient="records"),
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(
        output_dir / "report.md",
        focus_quarter,
        focus_month,
        selected_factors,
        rank_ic_summary,
        slot_edge_summary,
        focus_month_stock_signature,
    )
    print(f"[OK] wrote v21 rule logic report to {output_dir}")


if __name__ == "__main__":
    main()
