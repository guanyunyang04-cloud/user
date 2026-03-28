from __future__ import annotations

import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json

import numpy as np
import pandas as pd

from daily_research.baseline.advanced_ml_runtime import (
    build_prepared_bundle_with_cache,
    history_window_to_dict,
    load_raw_data_with_cache,
    resolve_history_window,
)
from daily_research.baseline.cli_utils import (
    load_stock_list_from_file,
    parse_csv_list,
    parse_horizon_weights,
    parse_int_tuple,
    parse_named_windows,
    parse_stock_list,
)
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import load_industry_map_from_tq, load_style_map_from_tq, load_universe_from_tq
from daily_research.baseline.diagnose_advanced_ml_ensemble import (
    DEFAULT_FOCUS_STATE,
    _load_pickle,
    _ml_score_cache_path,
    _run_candidate,
    _save_pickle,
)
from daily_research.baseline.ml_alpha import MLAplhaConfig, build_ml_feature_bundle, rolling_ml_scores_multi_detail
from daily_research.baseline.scan_advanced_ml_attack_defense_controller import (
    _build_static_candidates,
    _resolve_weak_window_name,
    _safe_num,
    _safe_pct,
    _subset_raw_df_dict_to_stocks,
)
from daily_research.execution.liquidity_universe import build_rolling_liquidity_membership


DEFAULT_THRESHOLDS = (0.20, 0.35, 0.50, 0.65, 0.80)
FEATURE_COLUMNS = (
    "trend_gap",
    "annual_vol",
    "benchmark_ret_5d",
    "benchmark_ret_10d",
    "benchmark_ret_20d",
    "focus_streak",
)
SUMMARY_FEATURE_COLUMNS = (
    "trend_gap",
    "annual_vol",
    "benchmark_ret_5d",
    "benchmark_ret_10d",
    "focus_streak",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose when advanced_ml shortlist offense vs defense should win within the focus state."
    )
    parser.add_argument("--data-source", choices=["tq", "csv"], default="tq")
    parser.add_argument("--csv-folder", default=None)
    parser.add_argument("--stocks", default=None)
    parser.add_argument("--stocks-file", default=None)
    parser.add_argument("--start-date", default="20190101")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--universe-scope", default="all_a")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument(
        "--rolling-liquidity-pool",
        choices=["liquid300", "liquid500", "liquid800"],
        default="liquid500",
    )
    parser.add_argument("--pool-rebalance-days", type=int, default=21)
    parser.add_argument("--pool-adv-window", type=int, default=20)
    parser.add_argument("--holding-count", type=int, default=5)
    parser.add_argument("--rebalance-freq", default="1d")
    parser.add_argument("--enhanced-profile", default="up_low_breakout_v2")
    parser.add_argument("--focus-state", default=DEFAULT_FOCUS_STATE)
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--output-dir", default="")

    parser.add_argument("--min-adv20", type=float, default=50_000.0)
    parser.add_argument("--min-price", type=float, default=2.0)
    parser.add_argument("--max-price", type=float, default=300.0)
    parser.add_argument("--max-weight", type=float, default=0.25)
    parser.add_argument("--score-threshold", type=float, default=0.0)

    parser.add_argument("--no-market-regime-filter", action="store_true")
    parser.add_argument("--regime-ma-window", type=int, default=50)
    parser.add_argument("--regime-vol-window", type=int, default=20)
    parser.add_argument("--regime-max-annual-vol", type=float, default=0.32)
    parser.add_argument("--regime-quadrants", default="trend_up_low_vol,trend_up_high_vol")

    parser.add_argument("--no-style-cap", action="store_true")
    parser.add_argument("--max-style-weight", type=float, default=0.50)
    parser.add_argument("--industry-cap", action="store_true")
    parser.add_argument("--max-industry-weight", type=float, default=0.40)

    parser.add_argument("--ml-target-horizon", type=int, default=20)
    parser.add_argument("--ml-target-horizons", default="5,10,20")
    parser.add_argument("--ml-horizon-weights", default="5:0.2,10:0.3,20:0.5")
    parser.add_argument("--ml-train-window-days", type=int, default=504)
    parser.add_argument("--ml-retrain-every-days", type=int, default=21)
    parser.add_argument("--ml-min-train-dates", type=int, default=120)
    parser.add_argument("--ml-max-samples-per-day", type=int, default=600)
    parser.add_argument("--ml-max-train-rows", type=int, default=200000)
    parser.add_argument("--ml-random-seed", type=int, default=7)
    parser.add_argument("--ml-model-family", choices=["lgbm"], default="lgbm")
    parser.add_argument("--lgbm-n-estimators", type=int, default=520)

    parser.add_argument("--ensemble-ml-weight", type=float, default=0.70)
    parser.add_argument("--ensemble-none-weight", type=float, default=0.20)
    parser.add_argument("--ensemble-v2-weight", type=float, default=0.10)
    parser.add_argument("--offense-state-weights", default="ml:0.25,none:0.20,v2:0.55")
    parser.add_argument("--defense-state-weights", default="ml:0.25,none:0.25,v2:0.50")
    parser.add_argument(
        "--windows",
        default="recent_full:20250307:20260327,weak_window_20250905_20260319:20250905:20260319",
    )
    parser.add_argument("--auto-trim-history", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def _parse_weight_triplet(raw: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for chunk in str(raw).split(","):
        item = chunk.strip()
        if not item:
            continue
        name_raw, value_raw = item.split(":", 1)
        weights[name_raw.strip().lower()] = float(value_raw.strip())
    required = {"ml", "none", "v2"}
    if set(weights) != required:
        raise ValueError(f"weight triplet must contain exactly {sorted(required)}, got {weights}")
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("weight triplet must sum to a positive number")
    return {key: float(weights[key] / total) for key in ("ml", "none", "v2")}


def _resolve_stock_universe(args: argparse.Namespace) -> list[str]:
    explicit: list[str] = []
    explicit.extend(parse_stock_list(args.stocks))
    explicit.extend(load_stock_list_from_file(args.stocks_file))
    if explicit:
        return list(dict.fromkeys(explicit))
    return list(load_universe_from_tq(args.universe_scope))


def _load_or_build_shared_scores(
    *,
    prepared_bundle: dict[str, Any],
    prepared_cache_key: str,
    ml_cfg: MLAplhaConfig,
    current_membership_mask: pd.DataFrame | None,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    cache_path = _ml_score_cache_path(prepared_cache_key, ml_cfg)
    cached = _load_pickle(cache_path)
    if isinstance(cached, dict) and "per_horizon_scores" in cached:
        training_log = cached.get("training_log")
        return cached["per_horizon_scores"], training_log if isinstance(training_log, pd.DataFrame) else pd.DataFrame()
    if isinstance(cached, dict):
        return cached, pd.DataFrame()

    filter_mask = prepared_bundle["filter_mask"]
    if current_membership_mask is not None:
        filter_mask = filter_mask & current_membership_mask
    _, training_log_df, per_horizon_scores = rolling_ml_scores_multi_detail(
        feature_frames=prepared_bundle["feature_frames"],
        market_features=prepared_bundle["market_features"],
        close=prepared_bundle["factor_bundle"]["raw_inputs"]["Close"],
        benchmark_close=prepared_bundle["benchmark_close"],
        filter_mask=filter_mask,
        regime_state=prepared_bundle["regime_state"],
        config=ml_cfg,
        open_df=prepared_bundle["factor_bundle"]["raw_inputs"]["Open"],
        benchmark_open=prepared_bundle["benchmark_open"],
    )
    _save_pickle(
        cache_path,
        {
            "training_log": training_log_df,
            "per_horizon_scores": per_horizon_scores,
        },
    )
    return per_horizon_scores, training_log_df


def _build_signal_feature_frame(prepared_bundle: dict[str, Any], focus_state: str) -> pd.DataFrame:
    regime_state = prepared_bundle["regime_state"]
    benchmark_close = prepared_bundle["benchmark_close"].sort_index()

    feature_df = pd.DataFrame(index=benchmark_close.index)
    feature_df["quadrant"] = regime_state["quadrant"].reindex(feature_df.index)
    feature_df["market_state"] = regime_state["market_state"].reindex(feature_df.index)
    feature_df["trend_bucket"] = regime_state["trend_bucket"].reindex(feature_df.index)
    feature_df["vol_bucket"] = regime_state["vol_bucket"].reindex(feature_df.index)
    feature_df["trend_gap"] = regime_state["benchmark_trend_gap"].reindex(feature_df.index)
    feature_df["annual_vol"] = regime_state["benchmark_annual_vol"].reindex(feature_df.index)
    feature_df["benchmark_vol_ratio"] = regime_state["benchmark_vol_ratio"].reindex(feature_df.index)
    feature_df["benchmark_ret_5d"] = benchmark_close.pct_change(5)
    feature_df["benchmark_ret_10d"] = benchmark_close.pct_change(10)
    feature_df["benchmark_ret_20d"] = benchmark_close.pct_change(20)

    focus_flag = feature_df["quadrant"].eq(focus_state).fillna(False)
    streak_values: list[int] = []
    streak = 0
    for is_focus in focus_flag.tolist():
        if is_focus:
            streak += 1
        else:
            streak = 0
        streak_values.append(int(streak))
    feature_df["focus_streak"] = streak_values
    feature_df.index.name = "signal_date"
    return feature_df.reset_index()


def _build_daily_comparison(
    *,
    offense_equity_df: pd.DataFrame,
    defense_equity_df: pd.DataFrame,
    signal_feature_df: pd.DataFrame,
    focus_state: str,
    weak_window_name: str,
    windows: list[tuple[str, str, str]],
) -> pd.DataFrame:
    offense = offense_equity_df.reset_index()
    defense = defense_equity_df.reset_index()
    if "date" in offense.columns:
        offense = offense.rename(columns={"date": "row_date"})
    if "date" in defense.columns:
        defense = defense.rename(columns={"date": "row_date"})

    daily_df = offense[
        [
            "signal_date",
            "execution_date",
            "portfolio_return",
            "excess_return",
            "turnover",
            "holding_count",
        ]
    ].rename(
        columns={
            "portfolio_return": "offense_portfolio_return",
            "excess_return": "offense_excess_return",
            "turnover": "offense_turnover",
            "holding_count": "offense_holding_count",
        }
    ).merge(
        defense[
            [
                "signal_date",
                "execution_date",
                "portfolio_return",
                "excess_return",
                "turnover",
                "holding_count",
            ]
        ].rename(
            columns={
                "portfolio_return": "defense_portfolio_return",
                "excess_return": "defense_excess_return",
                "turnover": "defense_turnover",
                "holding_count": "defense_holding_count",
            }
        ),
        on=["execution_date", "signal_date"],
        how="inner",
    )
    daily_df["signal_date"] = pd.to_datetime(daily_df["signal_date"])
    daily_df = daily_df.loc[daily_df["signal_date"].notna()].copy()
    daily_df = daily_df.merge(signal_feature_df, on="signal_date", how="left")
    daily_df = daily_df.loc[daily_df["quadrant"].eq(focus_state)].copy()
    daily_df["offense_minus_defense_excess_return"] = (
        daily_df["offense_excess_return"] - daily_df["defense_excess_return"]
    )
    daily_df["offense_minus_defense_portfolio_return"] = (
        daily_df["offense_portfolio_return"] - daily_df["defense_portfolio_return"]
    )
    daily_df["offense_win"] = daily_df["offense_minus_defense_excess_return"] > 0
    daily_df["in_weak_window"] = False
    for window_name, start, end in windows:
        flag_col = f"in_{window_name}"
        mask = (daily_df["signal_date"] >= pd.Timestamp(start)) & (daily_df["signal_date"] <= pd.Timestamp(end))
        daily_df[flag_col] = mask
        if window_name == weak_window_name:
            daily_df["in_weak_window"] = mask
    return daily_df.sort_values(["signal_date", "execution_date"]).reset_index(drop=True)


def _corr_rows(daily_df: pd.DataFrame, feature_names: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_name in feature_names:
        valid = daily_df[[feature_name, "offense_minus_defense_excess_return"]].dropna()
        corr = valid[feature_name].corr(valid["offense_minus_defense_excess_return"]) if len(valid) >= 5 else np.nan
        rows.append(
            {
                "feature": feature_name,
                "corr_with_offense_minus_defense_excess_return": float(corr) if pd.notna(corr) else np.nan,
                "valid_days": int(len(valid)),
            }
        )
    return pd.DataFrame(rows)


def _build_section_masks(daily_df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    return [
        ("overall", pd.Series(True, index=daily_df.index)),
        ("weak_window", daily_df["in_weak_window"].fillna(False)),
        ("outside_weak_window", ~daily_df["in_weak_window"].fillna(False)),
    ]


def _summarize_feature_bins(daily_df: pd.DataFrame, feature_names: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for section_name, section_mask in _build_section_masks(daily_df):
        section_df = daily_df.loc[section_mask].copy()
        if section_df.empty:
            continue
        for feature_name in feature_names:
            valid = section_df[[feature_name, "offense_minus_defense_excess_return", "offense_win"]].dropna().copy()
            if len(valid) < 12:
                continue
            try:
                valid["bin"] = pd.qcut(valid[feature_name], 4, duplicates="drop")
            except ValueError:
                continue
            grouped = (
                valid.groupby("bin", observed=False)
                .agg(
                    days=("offense_minus_defense_excess_return", "count"),
                    mean_diff=("offense_minus_defense_excess_return", "mean"),
                    median_diff=("offense_minus_defense_excess_return", "median"),
                    win_rate=("offense_win", "mean"),
                )
                .reset_index()
            )
            for _, row in grouped.iterrows():
                rows.append(
                    {
                        "section": section_name,
                        "feature": feature_name,
                        "bin": str(row["bin"]),
                        "days": int(row["days"]),
                        "mean_diff": float(row["mean_diff"]),
                        "median_diff": float(row["median_diff"]),
                        "win_rate": float(row["win_rate"]),
                    }
                )
    return pd.DataFrame(rows)


def _threshold_grid(series: pd.Series) -> list[float]:
    valid = series.dropna()
    if valid.empty:
        return []
    return sorted(
        {
            float(value)
            for value in valid.quantile(list(DEFAULT_THRESHOLDS)).tolist()
            if pd.notna(value)
        }
    )


def _summarize_rule(
    *,
    daily_df: pd.DataFrame,
    label: str,
    mask: pd.Series,
) -> dict[str, Any] | None:
    applied = daily_df.loc[mask.fillna(False)].copy()
    if applied.empty:
        return None
    weak_df = applied.loc[applied["in_weak_window"]].copy()
    outside_weak_df = applied.loc[~applied["in_weak_window"]].copy()
    return {
        "rule": label,
        "days": int(len(applied)),
        "weak_days": int(len(weak_df)),
        "mean_diff": float(applied["offense_minus_defense_excess_return"].mean()),
        "median_diff": float(applied["offense_minus_defense_excess_return"].median()),
        "win_rate": float(applied["offense_win"].mean()),
        "weak_mean_diff": float(weak_df["offense_minus_defense_excess_return"].mean()) if not weak_df.empty else np.nan,
        "outside_weak_mean_diff": (
            float(outside_weak_df["offense_minus_defense_excess_return"].mean()) if not outside_weak_df.empty else np.nan
        ),
    }


def _scan_single_rules(daily_df: pd.DataFrame) -> pd.DataFrame:
    rules: list[dict[str, Any]] = []
    trend_gap_grid = _threshold_grid(daily_df["trend_gap"])
    annual_vol_grid = _threshold_grid(daily_df["annual_vol"])
    ret10_grid = _threshold_grid(daily_df["benchmark_ret_10d"])
    streak_grid = _threshold_grid(daily_df["focus_streak"])

    for threshold in trend_gap_grid:
        row = _summarize_rule(
            daily_df=daily_df,
            label=f"trend_gap>={threshold:.6f}",
            mask=daily_df["trend_gap"] >= float(threshold),
        )
        if row is not None:
            rules.append(row)
    for threshold in annual_vol_grid:
        row = _summarize_rule(
            daily_df=daily_df,
            label=f"annual_vol<={threshold:.6f}",
            mask=daily_df["annual_vol"] <= float(threshold),
        )
        if row is not None:
            rules.append(row)
    for threshold in ret10_grid:
        row = _summarize_rule(
            daily_df=daily_df,
            label=f"benchmark_ret_10d>={threshold:.6f}",
            mask=daily_df["benchmark_ret_10d"] >= float(threshold),
        )
        if row is not None:
            rules.append(row)
    for threshold in streak_grid:
        row = _summarize_rule(
            daily_df=daily_df,
            label=f"focus_streak>={threshold:.0f}",
            mask=daily_df["focus_streak"] >= float(threshold),
        )
        if row is not None:
            rules.append(row)

    if not rules:
        return pd.DataFrame()
    return pd.DataFrame(rules).sort_values(
        ["mean_diff", "weak_mean_diff", "win_rate", "days"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def _scan_combo_rules(daily_df: pd.DataFrame) -> pd.DataFrame:
    combo_rows: list[dict[str, Any]] = []
    trend_gap_grid = _threshold_grid(daily_df["trend_gap"])
    annual_vol_grid = _threshold_grid(daily_df["annual_vol"])
    ret10_grid = _threshold_grid(daily_df["benchmark_ret_10d"])

    for trend_gap_min in trend_gap_grid:
        for annual_vol_max in annual_vol_grid:
            for ret10_min in ret10_grid:
                mask = (
                    (daily_df["trend_gap"] >= float(trend_gap_min))
                    & (daily_df["annual_vol"] <= float(annual_vol_max))
                    & (daily_df["benchmark_ret_10d"] >= float(ret10_min))
                )
                if int(mask.sum()) < 20:
                    continue
                row = _summarize_rule(
                    daily_df=daily_df,
                    label=(
                        f"trend_gap>={trend_gap_min:.6f} & "
                        f"annual_vol<={annual_vol_max:.6f} & "
                        f"benchmark_ret_10d>={ret10_min:.6f}"
                    ),
                    mask=mask,
                )
                if row is None:
                    continue
                row["trend_gap_min"] = float(trend_gap_min)
                row["annual_vol_max"] = float(annual_vol_max)
                row["benchmark_ret_10d_min"] = float(ret10_min)
                combo_rows.append(row)
    if not combo_rows:
        return pd.DataFrame()
    return pd.DataFrame(combo_rows).sort_values(
        ["mean_diff", "weak_mean_diff", "win_rate", "days"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def _render_summary(
    *,
    latest_data_date: str,
    history_window: dict[str, Any],
    static_metrics_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    feature_bin_df: pd.DataFrame,
    single_rule_df: pd.DataFrame,
    combo_rule_df: pd.DataFrame,
    weak_window_name: str,
) -> str:
    lines: list[str] = []
    lines.append("# Advanced ML Attack / Defense Diagnosis")
    lines.append("")
    lines.append(f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- latest_data_date: {latest_data_date}")
    lines.append(
        f"- history_window: {history_window['effective_start_date']} -> {history_window['end_date']} (mode={history_window['mode']})"
    )
    lines.append("")

    if not static_metrics_df.empty:
        lines.append("## Static Controls")
        for _, row in static_metrics_df.iterrows():
            lines.append(
                f"- {row['label']}: full excess Sharpe {_safe_num(row['full_excess_sharpe'])}, "
                f"{weak_window_name} excess Sharpe {_safe_num(row[f'{weak_window_name}_excess_sharpe'])}, "
                f"{weak_window_name} focus excess Sharpe {_safe_num(row[f'trend_up_low_vol_{weak_window_name}_excess_sharpe'])}"
            )
        lines.append("")

    lines.append("## Daily Differential")
    lines.append(f"- focus_state_days: {int(len(daily_df))}")
    lines.append(
        f"- offense_minus_defense_mean_excess_return: {_safe_num(daily_df['offense_minus_defense_excess_return'].mean())}"
    )
    lines.append(f"- offense_win_rate: {_safe_num(daily_df['offense_win'].mean())}")
    weak_df = daily_df.loc[daily_df["in_weak_window"]]
    outside_weak_df = daily_df.loc[~daily_df["in_weak_window"]]
    lines.append(
        f"- weak_window_mean_diff: {_safe_num(weak_df['offense_minus_defense_excess_return'].mean())}"
    )
    lines.append(
        f"- outside_weak_window_mean_diff: {_safe_num(outside_weak_df['offense_minus_defense_excess_return'].mean())}"
    )
    lines.append("")

    if not corr_df.empty:
        lines.append("## Feature Correlation")
        for _, row in corr_df.sort_values(
            "corr_with_offense_minus_defense_excess_return",
            ascending=False,
        ).iterrows():
            lines.append(
                f"- {row['feature']}: corr {_safe_num(row['corr_with_offense_minus_defense_excess_return'])} "
                f"(days={int(row['valid_days'])})"
            )
        lines.append("")

    if not feature_bin_df.empty:
        lines.append("## Bin Highlights")
        overall_bins = feature_bin_df.loc[feature_bin_df["section"].eq("overall")].copy()
        for feature_name in SUMMARY_FEATURE_COLUMNS:
            feature_rows = overall_bins.loc[overall_bins["feature"].eq(feature_name)].copy()
            if feature_rows.empty:
                continue
            best_row = feature_rows.sort_values(["mean_diff", "win_rate", "days"], ascending=[False, False, False]).iloc[0]
            worst_row = feature_rows.sort_values(["mean_diff", "win_rate", "days"], ascending=[True, True, False]).iloc[0]
            lines.append(
                f"- {feature_name}: best bin {best_row['bin']} -> mean diff {_safe_num(best_row['mean_diff'])}, "
                f"win rate {_safe_pct(best_row['win_rate'])}; "
                f"worst bin {worst_row['bin']} -> mean diff {_safe_num(worst_row['mean_diff'])}, "
                f"win rate {_safe_pct(worst_row['win_rate'])}"
            )
        lines.append("")

    if not single_rule_df.empty:
        lines.append("## Single Rules")
        for _, row in single_rule_df.head(8).iterrows():
            lines.append(
                f"- {row['rule']}: days {int(row['days'])}, mean diff {_safe_num(row['mean_diff'])}, "
                f"weak mean diff {_safe_num(row['weak_mean_diff'])}, win rate {_safe_pct(row['win_rate'])}"
            )
        lines.append("")

    if not combo_rule_df.empty:
        lines.append("## Combo Rules")
        for _, row in combo_rule_df.head(8).iterrows():
            lines.append(
                f"- {row['rule']}: days {int(row['days'])}, mean diff {_safe_num(row['mean_diff'])}, "
                f"weak mean diff {_safe_num(row['weak_mean_diff'])}, win rate {_safe_pct(row['win_rate'])}"
            )
        lines.append("")

    lines.append("## Note")
    lines.append(
        "- These rule tables are diagnostic approximations based on static offense-vs-defense daily differentials."
    )
    lines.append(
        "- Any candidate promoted from here still needs a separate formal controller backtest before it can change the execution default."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    stocks = parse_stock_list(args.stocks)
    file_stocks = load_stock_list_from_file(args.stocks_file)
    if stocks or file_stocks:
        stocks = list(dict.fromkeys(stocks + file_stocks))
    if args.rolling_liquidity_pool and stocks:
        raise ValueError("Use either fixed stocks or --rolling-liquidity-pool, not both.")

    cfg = ResearchConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        universe_scope=args.universe_scope,
        benchmark=args.benchmark,
        execution_mode="next_open",
        rebalance_freq=args.rebalance_freq,
        holding_count=args.holding_count,
        weighting_method="score",
        min_adv20=args.min_adv20,
        min_price=args.min_price,
        max_price=args.max_price,
        max_weight=args.max_weight,
        score_threshold=args.score_threshold,
        enable_market_regime_filter=not args.no_market_regime_filter,
        regime_ma_window=args.regime_ma_window,
        regime_vol_window=args.regime_vol_window,
        regime_max_annual_vol=args.regime_max_annual_vol,
        regime_allowed_quadrants=parse_csv_list(args.regime_quadrants),
        enable_style_cap=not args.no_style_cap,
        max_style_weight=args.max_style_weight,
        enable_industry_cap=args.industry_cap,
        max_industry_weight=args.max_industry_weight,
    )
    if stocks:
        cfg.universe = stocks
    base_weights = {
        "ml": float(args.ensemble_ml_weight),
        "none": float(args.ensemble_none_weight),
        "v2": float(args.ensemble_v2_weight),
    }
    offense_weights = _parse_weight_triplet(args.offense_state_weights)
    defense_weights = _parse_weight_triplet(args.defense_state_weights)
    windows = parse_named_windows(args.windows)
    weak_window_name = _resolve_weak_window_name(windows)
    focus_state = str(args.focus_state).strip().lower() or DEFAULT_FOCUS_STATE

    ml_cfg = MLAplhaConfig(
        target_horizon=args.ml_target_horizon,
        target_horizons=parse_int_tuple(args.ml_target_horizons, args.ml_target_horizon),
        target_horizon_weights=parse_horizon_weights(args.ml_horizon_weights),
        enhanced_profile=args.enhanced_profile,
        train_window_days=args.ml_train_window_days,
        retrain_every_days=args.ml_retrain_every_days,
        min_train_dates=args.ml_min_train_dates,
        max_samples_per_day=args.ml_max_samples_per_day,
        max_train_rows=args.ml_max_train_rows,
        random_seed=args.ml_random_seed,
        model_family=args.ml_model_family,
        lgbm_n_estimators=args.lgbm_n_estimators,
        ensemble_ml_weight=base_weights["ml"],
        ensemble_none_weight=base_weights["none"],
        ensemble_v2_weight=base_weights["v2"],
        train_regime_only=cfg.enable_market_regime_filter,
        execution_mode=cfg.execution_mode,
    )
    static_candidates = _build_static_candidates(
        focus_state=focus_state,
        base_weights=base_weights,
        offense_weights=offense_weights,
        defense_weights=defense_weights,
    )

    if args.data_source == "tq" and not cfg.universe:
        cfg.universe = load_universe_from_tq(cfg.universe_scope)
    if not cfg.universe:
        raise ValueError("Universe is empty after loading.")

    history_window = resolve_history_window(
        cfg,
        ml_cfg,
        requested_start_date=args.start_date,
        end_date=args.end_date,
        mode="train",
        auto_trim_history=args.auto_trim_history,
    )
    raw_df_dict, raw_meta = load_raw_data_with_cache(
        data_source=args.data_source,
        csv_folder=args.csv_folder,
        universe=cfg.universe,
        benchmark=cfg.benchmark,
        history_window=history_window,
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
        progress_desc="attack/defense diagnosis raw",
        progress_position=0,
    )
    rolling_pool_artifact = None
    rolling_membership_mask = None
    prepared_raw_df_dict = raw_df_dict
    if args.rolling_liquidity_pool:
        raw_universe_df_dict = {
            key: frame.drop(columns=[cfg.benchmark], errors="ignore")
            for key, frame in raw_df_dict.items()
        }
        rolling_pool_artifact = build_rolling_liquidity_membership(
            close_frame=raw_universe_df_dict["Close"],
            amount_frame=raw_universe_df_dict["Amount"],
            pool_name=args.rolling_liquidity_pool,
            signal_start_date=cfg.start_date,
            signal_end_date=args.end_date,
            rebalance_every_days=args.pool_rebalance_days,
            adv_window=args.pool_adv_window,
            min_price=cfg.min_price,
            max_price=cfg.max_price,
        )
        rolling_membership_mask = rolling_pool_artifact.membership_frame
        rolling_union = rolling_membership_mask.columns[rolling_membership_mask.any(axis=0)].tolist()
        if not rolling_union:
            raise RuntimeError(f"Rolling {args.rolling_liquidity_pool} pool is empty for the requested window.")
        prepared_raw_df_dict = _subset_raw_df_dict_to_stocks(raw_df_dict, cfg.benchmark, rolling_union)

    prepared_raw_cache_key = (
        raw_meta["cache_key"]
        if rolling_pool_artifact is None
        else f"{raw_meta['cache_key']}|{args.rolling_liquidity_pool}|{args.pool_rebalance_days}|{args.pool_adv_window}"
    )
    prepared_bundle, prepared_meta = build_prepared_bundle_with_cache(
        raw_df_dict=prepared_raw_df_dict,
        raw_cache_key=prepared_raw_cache_key,
        cfg=cfg,
        enhanced_profile=args.enhanced_profile,
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
    )
    current_membership_mask = None
    if rolling_membership_mask is not None:
        current_membership_mask = rolling_membership_mask.reindex(
            index=prepared_bundle["df_dict"]["Close"].index,
            columns=prepared_bundle["df_dict"]["Close"].columns,
        ).fillna(False)
        prepared_bundle["filter_mask"] = prepared_bundle["filter_mask"] & current_membership_mask
        prepared_bundle["score_none"] = prepared_bundle["score_none"].where(current_membership_mask)
        prepared_bundle["score_v2"] = prepared_bundle["score_v2"].where(current_membership_mask)
        feature_frames, market_features = build_ml_feature_bundle(
            prepared_bundle["factor_bundle"],
            prepared_bundle["regime_state"],
            prepared_bundle["score_none"],
            prepared_bundle["score_v2"],
        )
        prepared_bundle["feature_frames"] = feature_frames
        prepared_bundle["market_features"] = market_features

    candidate_columns = [col for col in prepared_bundle["df_dict"]["Close"].columns if col != cfg.benchmark]
    industry_map = None
    style_map = None
    if cfg.enable_industry_cap and args.data_source == "tq":
        industry_map = load_industry_map_from_tq(candidate_columns)
    if cfg.enable_style_cap and args.data_source == "tq":
        style_map = load_style_map_from_tq(candidate_columns)
    shared_per_horizon_scores, training_log_df = _load_or_build_shared_scores(
        prepared_bundle=prepared_bundle,
        prepared_cache_key=prepared_meta["cache_key"],
        ml_cfg=ml_cfg,
        current_membership_mask=current_membership_mask,
    )

    latest_data_date = pd.Timestamp(
        prepared_bundle["factor_bundle"]["raw_inputs"]["Close"].dropna(how="all").index.max()
    ).strftime("%Y-%m-%d")

    static_rows: list[dict[str, Any]] = []
    offense_equity_df: pd.DataFrame | None = None
    defense_equity_df: pd.DataFrame | None = None
    for candidate in static_candidates:
        row, equity_df = _run_candidate(
            candidate=candidate,
            cfg=cfg,
            ml_cfg=ml_cfg,
            shared_per_horizon_scores=shared_per_horizon_scores,
            prepared_bundle=prepared_bundle,
            current_membership_mask=current_membership_mask,
            industry_map=industry_map,
            style_map=style_map,
            windows=windows,
            focus_state=focus_state,
        )
        static_rows.append(row)
        if str(candidate["profile"]) == "offense":
            offense_equity_df = equity_df
        elif str(candidate["profile"]) == "defense":
            defense_equity_df = equity_df

    if offense_equity_df is None or defense_equity_df is None:
        raise RuntimeError("Failed to build offense/defense comparison equity frames.")

    signal_feature_df = _build_signal_feature_frame(prepared_bundle, focus_state)
    daily_df = _build_daily_comparison(
        offense_equity_df=offense_equity_df,
        defense_equity_df=defense_equity_df,
        signal_feature_df=signal_feature_df,
        focus_state=focus_state,
        weak_window_name=weak_window_name,
        windows=windows,
    )
    corr_df = _corr_rows(daily_df, FEATURE_COLUMNS)
    feature_bin_df = _summarize_feature_bins(daily_df, SUMMARY_FEATURE_COLUMNS)
    single_rule_df = _scan_single_rules(daily_df)
    combo_rule_df = _scan_combo_rules(daily_df)
    static_metrics_df = pd.DataFrame(static_rows).sort_values("label").reset_index(drop=True)

    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        tag = str(args.experiment_tag).strip() or datetime.now().strftime(
            "advanced_ml_attack_defense_diagnosis_%Y%m%d_%H%M%S"
        )
        output_root = Path("daily_research/output") / tag
    output_root.mkdir(parents=True, exist_ok=True)

    run_config = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latest_data_date": latest_data_date,
        "history_window": history_window_to_dict(history_window),
        "cfg": asdict(cfg),
        "ml_cfg": asdict(ml_cfg),
        "focus_state": focus_state,
        "weak_window_name": weak_window_name,
        "static_candidates": static_candidates,
        "raw_cache_meta": raw_meta,
        "prepared_cache_meta": prepared_meta,
    }
    (output_root / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    static_metrics_df.to_csv(output_root / "static_control_metrics.csv", index=False, encoding="utf-8-sig")
    daily_df.to_csv(output_root / "daily_comparison.csv", index=False, encoding="utf-8-sig")
    corr_df.to_csv(output_root / "feature_correlations.csv", index=False, encoding="utf-8-sig")
    feature_bin_df.to_csv(output_root / "feature_bin_summary.csv", index=False, encoding="utf-8-sig")
    single_rule_df.to_csv(output_root / "single_rule_scan.csv", index=False, encoding="utf-8-sig")
    combo_rule_df.to_csv(output_root / "combo_rule_scan.csv", index=False, encoding="utf-8-sig")
    if not training_log_df.empty:
        training_dir = output_root / "training_logs"
        training_dir.mkdir(parents=True, exist_ok=True)
        training_log_df.to_csv(training_dir / "shared_training_log.csv", index=False, encoding="utf-8-sig")

    summary_text = _render_summary(
        latest_data_date=latest_data_date,
        history_window=history_window_to_dict(history_window),
        static_metrics_df=static_metrics_df,
        daily_df=daily_df,
        corr_df=corr_df,
        feature_bin_df=feature_bin_df,
        single_rule_df=single_rule_df,
        combo_rule_df=combo_rule_df,
        weak_window_name=weak_window_name,
    )
    (output_root / "summary.md").write_text(summary_text, encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_root),
                "focus_state_days": int(len(daily_df)),
                "single_rule_count": int(len(single_rule_df)),
                "combo_rule_count": int(len(combo_rule_df)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
