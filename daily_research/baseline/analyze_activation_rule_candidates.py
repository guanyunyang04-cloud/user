from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from daily_research.baseline.alpha import combine_scores_by_state
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import load_daily_from_tq, load_universe_from_tq, split_benchmark_from_universe
from daily_research.baseline.evaluation import compute_forward_returns
from daily_research.baseline.features import compute_factors
from daily_research.baseline.quadrant_ic_activation_validation import _build_test_windows, _parse_csv_list
from daily_research.baseline.regime import compute_market_regime_state
from daily_research.baseline.state_profiles import build_state_configs


BREAKOUT_FACTORS = ["breakout_20", "drawdown_20", "range_position_20", "up_day_ratio_10"]
DEFENSIVE_FACTORS = [
    "volatility_20",
    "atr_14_pct",
    "volatility_contraction",
    "volume_contraction",
    "price_volume_divergence",
    "close_strength",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Explain why state profile activation switches on or off")
    parser.add_argument("--start-date", default="20210101")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--activation-frequency", default="quarter", choices=["year", "halfyear", "quarter"])
    parser.add_argument("--regime-max-annual-vol", type=float, default=0.32)
    parser.add_argument("--regime-quadrants", default="trend_up_low_vol,trend_up_high_vol")
    parser.add_argument("--ic-horizon", type=int, default=20)
    parser.add_argument("--focus-quadrant", default="trend_up_low_vol")
    parser.add_argument("--profiles", default="none,up_low_breakout_v2")
    parser.add_argument("--experiment-tag", default="")
    return parser.parse_args()


def _rank_corr(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 2:
        return np.nan
    if pair.iloc[:, 0].nunique() <= 1 or pair.iloc[:, 1].nunique() <= 1:
        return np.nan
    xr = pair.iloc[:, 0].rank(method="average")
    yr = pair.iloc[:, 1].rank(method="average")
    return float(xr.corr(yr))


def _average_rank_ic(frame: pd.DataFrame, forward_ret: pd.DataFrame, dates: pd.DatetimeIndex) -> float:
    values = []
    for dt in dates:
        if dt not in frame.index or dt not in forward_ret.index:
            continue
        values.append(_rank_corr(frame.loc[dt], forward_ret.loc[dt]))
    series = pd.Series(values).dropna()
    return float(series.mean()) if not series.empty else np.nan


def _group_average_rank_ic(
    raw_factors: Dict[str, pd.DataFrame],
    factor_names: List[str],
    forward_ret: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> float:
    scores = []
    for factor_name in factor_names:
        if factor_name not in raw_factors:
            continue
        rank_ic = _average_rank_ic(raw_factors[factor_name], forward_ret, dates)
        if not np.isnan(rank_ic):
            scores.append(rank_ic)
    return float(np.mean(scores)) if scores else np.nan


def main():
    args = parse_args()
    profiles = _parse_csv_list(args.profiles)
    allowed_quadrants = _parse_csv_list(args.regime_quadrants)

    cfg = ResearchConfig(
        start_date=args.start_date,
        benchmark=args.benchmark,
        universe_scope="all_a",
        weighting_method="score",
        rebalance_freq="5d",
        enable_market_regime_filter=True,
        regime_max_annual_vol=args.regime_max_annual_vol,
        regime_allowed_quadrants=allowed_quadrants,
        enable_style_cap=True,
        max_style_weight=0.50,
    )

    print("[1/5] 加载全A股票池...")
    universe = load_universe_from_tq("all_a")
    print(f"[2/5] 拉取日线数据，股票数: {len(universe)}")
    raw_df_dict = load_daily_from_tq(universe, args.start_date, benchmark=args.benchmark)
    df_dict, benchmark_close = split_benchmark_from_universe(raw_df_dict, args.benchmark)

    print("[3/5] 计算因子与前瞻收益...")
    factor_bundle = compute_factors(df_dict)
    close = factor_bundle["raw_inputs"]["Close"]
    forward_ret = compute_forward_returns(close, horizons=[args.ic_horizon])[f"fwd_{args.ic_horizon}d"]

    print("[4/5] 计算市场状态与 profile 分数...")
    regime_state = compute_market_regime_state(benchmark_close, cfg)
    quadrant_series = regime_state["quadrant"]
    score_map: Dict[str, pd.DataFrame] = {}
    for profile_name in profiles:
        state_configs = build_state_configs(cfg, profile_name)
        score_map[profile_name], _, _ = combine_scores_by_state(
            factor_bundle=factor_bundle,
            config=cfg,
            quadrant_series=quadrant_series,
            state_configs=state_configs,
        )

    print("[5/5] 逐窗口分析激活依据...")
    rows = []
    windows = _build_test_windows(close.index, train_years=2, train_months=args.train_months, activation_frequency=args.activation_frequency)
    for window_label, train_start, train_end, test_start, test_end, train_years in windows:
        train_mask = (quadrant_series.index >= train_start) & (quadrant_series.index <= train_end) & (quadrant_series == args.focus_quadrant)
        train_dates = quadrant_series.index[train_mask]
        if len(train_dates) == 0:
            continue

        profile_rankics = {
            profile_name: _average_rank_ic(score_map[profile_name], forward_ret, train_dates)
            for profile_name in profiles
        }
        selected_profile = max(profile_rankics.items(), key=lambda kv: (-np.inf if np.isnan(kv[1]) else kv[1]))[0]
        breakout_rank_ic = _group_average_rank_ic(
            factor_bundle["raw_factors"], BREAKOUT_FACTORS, forward_ret, train_dates
        )
        defensive_rank_ic = _group_average_rank_ic(
            factor_bundle["raw_factors"], DEFENSIVE_FACTORS, forward_ret, train_dates
        )
        benchmark_slice = benchmark_close.loc[train_start:train_end]
        vol_slice = regime_state["benchmark_annual_vol"].loc[train_start:train_end]
        rows.append(
            {
                "window_label": window_label,
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "train_years": ",".join(map(str, train_years)),
                "quadrant_days": int(len(train_dates)),
                "benchmark_return_60d_like": float(benchmark_slice.iloc[-1] / benchmark_slice.iloc[max(0, len(benchmark_slice) - 60)] - 1.0)
                if len(benchmark_slice) > 1
                else np.nan,
                "benchmark_mean_annual_vol": float(vol_slice.mean()) if not vol_slice.dropna().empty else np.nan,
                "none_rank_ic": float(profile_rankics.get("none", np.nan)),
                "up_low_breakout_v2_rank_ic": float(profile_rankics.get("up_low_breakout_v2", np.nan)),
                "v2_minus_none_rank_ic": float(profile_rankics.get("up_low_breakout_v2", np.nan) - profile_rankics.get("none", np.nan))
                if "none" in profile_rankics and "up_low_breakout_v2" in profile_rankics
                else np.nan,
                "breakout_pack_rank_ic": breakout_rank_ic,
                "defensive_pack_rank_ic": defensive_rank_ic,
                "breakout_minus_defensive": float(breakout_rank_ic - defensive_rank_ic)
                if not np.isnan(breakout_rank_ic) and not np.isnan(defensive_rank_ic)
                else np.nan,
                "selected_profile": selected_profile,
            }
        )

    result_df = pd.DataFrame(rows)
    out_root = Path(__file__).resolve().parents[1] / "output"
    run_name = args.experiment_tag.strip() or datetime.now().strftime("activation_rule_candidates_%Y%m%d_%H%M%S")
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(out_dir / "activation_rule_candidates.csv", index=False, encoding="utf-8-sig")

    selected_df = result_df[result_df["selected_profile"] == "up_low_breakout_v2"].copy()
    selected_df.to_csv(out_dir / "selected_v2_windows.csv", index=False, encoding="utf-8-sig")

    print(f"输出目录: {out_dir}")
    if not result_df.empty:
        display_cols = [
            "window_label",
            "quadrant_days",
            "none_rank_ic",
            "up_low_breakout_v2_rank_ic",
            "v2_minus_none_rank_ic",
            "breakout_pack_rank_ic",
            "defensive_pack_rank_ic",
            "breakout_minus_defensive",
            "selected_profile",
        ]
        print(result_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
