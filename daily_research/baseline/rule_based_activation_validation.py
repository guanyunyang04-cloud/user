from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from daily_research.baseline.alpha import combine_scores_by_state
from daily_research.baseline.backtest import backtest
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import load_daily_from_tq, load_style_map_from_tq, load_universe_from_tq, split_benchmark_from_universe
from daily_research.baseline.evaluation import compute_forward_returns
from daily_research.baseline.features import compute_factors
from daily_research.baseline.portfolio import build_target_weights
from daily_research.baseline.quadrant_ic_activation_validation import _apply_rebalance_frequency, _build_test_windows
from daily_research.baseline.regime import apply_market_regime_filter, compute_market_regime_state
from daily_research.baseline.state_profiles import build_state_configs


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a rule-based activation schedule for state profiles")
    parser.add_argument("--start-date", default="20210101")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--activation-frequency", default="quarter", choices=["year", "halfyear", "quarter"])
    parser.add_argument("--rebalance-freq", default="5d")
    parser.add_argument("--regime-max-annual-vol", type=float, default=0.32)
    parser.add_argument("--regime-quadrants", default="trend_up_low_vol,trend_up_high_vol")
    parser.add_argument("--style-cap", action="store_true")
    parser.add_argument("--max-style-weight", type=float, default=0.50)
    parser.add_argument("--focus-quadrant", default="trend_up_low_vol")
    parser.add_argument("--baseline-profile", default="none")
    parser.add_argument("--enhanced-profile", default="up_low_breakout_v2")
    parser.add_argument("--ic-horizon", type=int, default=20)
    parser.add_argument("--breakout-rankic-min", type=float, default=0.0)
    parser.add_argument("--drawdown-rankic-min", type=float, default=0.0)
    parser.add_argument("--range-position-rankic-min", type=float, default=-0.06)
    parser.add_argument("--experiment-tag", default="")
    return parser.parse_args()


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip().lower() for item in str(raw).split(",") if item.strip()]


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


def _build_profile_artifacts(
    factor_bundle: Dict[str, Dict[str, pd.DataFrame]],
    benchmark_close: pd.Series,
    cfg: ResearchConfig,
    profile_name: str,
    style_map: pd.DataFrame | None,
):
    regime_state = compute_market_regime_state(benchmark_close, cfg)
    state_configs = build_state_configs(cfg, profile_name)
    score, _, _ = combine_scores_by_state(
        factor_bundle,
        cfg,
        quadrant_series=regime_state["quadrant"],
        state_configs=state_configs,
    )
    target_weights = build_target_weights(score, cfg, style_map=style_map)
    target_weights = _apply_rebalance_frequency(target_weights, cfg.rebalance_freq)
    score_bt = _apply_rebalance_frequency(score.fillna(0.0), cfg.rebalance_freq)
    if cfg.enable_market_regime_filter:
        target_weights, score_bt = apply_market_regime_filter(target_weights, score_bt, regime_state)
    return {
        "score": score,
        "target_weights": target_weights,
        "target_scores": score_bt,
        "regime_state": regime_state,
    }


def _stitch_selected_profile_outputs(
    profile_artifacts: Dict[str, dict],
    quadrant_selection_rows: pd.DataFrame,
    index: pd.DatetimeIndex,
    quadrant_series: pd.Series,
    baseline_profile: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first = next(iter(profile_artifacts.values()))
    combined_weights = pd.DataFrame(0.0, index=index, columns=first["target_weights"].columns)
    combined_scores = pd.DataFrame(0.0, index=index, columns=first["target_scores"].columns)

    selection_map = {
        str(row["window_label"]): {
            "focus_quadrant": str(row["focus_quadrant"]),
            "selected_profile": str(row["selected_profile"]),
        }
        for _, row in quadrant_selection_rows.iterrows()
    }

    period_index = pd.Series(index=index, dtype="object")
    for dt in index:
        if dt.month <= 3:
            period_index.loc[dt] = f"{dt.year}Q1"
        elif dt.month <= 6:
            period_index.loc[dt] = f"{dt.year}Q2"
        elif dt.month <= 9:
            period_index.loc[dt] = f"{dt.year}Q3"
        else:
            period_index.loc[dt] = f"{dt.year}Q4"

    for dt in index:
        window_label = period_index.loc[dt]
        quadrant = quadrant_series.get(dt)
        selected_profile = baseline_profile
        selection_info = selection_map.get(window_label)
        if selection_info and quadrant == selection_info["focus_quadrant"]:
            selected_profile = selection_info["selected_profile"]
        artifacts = profile_artifacts[selected_profile]
        combined_weights.loc[dt] = artifacts["target_weights"].loc[dt]
        combined_scores.loc[dt] = artifacts["target_scores"].loc[dt]

    return combined_weights, combined_scores


def main():
    args = parse_args()
    allowed_quadrants = _parse_csv_list(args.regime_quadrants)
    cfg = ResearchConfig(
        start_date=args.start_date,
        benchmark=args.benchmark,
        universe_scope="all_a",
        weighting_method="score",
        rebalance_freq=args.rebalance_freq,
        enable_market_regime_filter=True,
        regime_max_annual_vol=args.regime_max_annual_vol,
        regime_allowed_quadrants=allowed_quadrants,
        enable_style_cap=args.style_cap,
        max_style_weight=args.max_style_weight,
    )

    print("[1/6] 加载全A股票池...")
    universe = load_universe_from_tq("all_a")
    print(f"[2/6] 拉取日线数据，股票数: {len(universe)}")
    raw_df_dict = load_daily_from_tq(universe, args.start_date, benchmark=args.benchmark)
    df_dict, benchmark_close = split_benchmark_from_universe(raw_df_dict, args.benchmark)

    print("[3/6] 计算因子与前瞻收益...")
    factor_bundle = compute_factors(df_dict)
    close = factor_bundle["raw_inputs"]["Close"]
    forward_ret = compute_forward_returns(close, horizons=[args.ic_horizon])[f"fwd_{args.ic_horizon}d"]

    style_map = None
    if cfg.enable_style_cap:
        print("[4/6] 加载风格映射...")
        style_map = load_style_map_from_tq(list(close.columns))

    print("[5/6] 预计算 baseline / enhanced profile...")
    profile_artifacts = {
        args.baseline_profile: _build_profile_artifacts(factor_bundle, benchmark_close, cfg, args.baseline_profile, style_map),
        args.enhanced_profile: _build_profile_artifacts(factor_bundle, benchmark_close, cfg, args.enhanced_profile, style_map),
    }
    base_regime_state = profile_artifacts[args.baseline_profile]["regime_state"]
    quadrant_series = base_regime_state["quadrant"]
    regime_on = base_regime_state["regime_on"]

    print("[6/6] 用可解释规则生成激活计划并回测...")
    rows = []
    windows = _build_test_windows(close.index, train_years=2, train_months=args.train_months, activation_frequency=args.activation_frequency)
    for window_label, train_start, train_end, test_start, test_end, train_years in windows:
        train_dates = quadrant_series.index[
            (quadrant_series.index >= train_start)
            & (quadrant_series.index <= train_end)
            & (quadrant_series == args.focus_quadrant)
        ]
        if len(train_dates) == 0:
            continue

        breakout_rank_ic = _average_rank_ic(factor_bundle["raw_factors"]["breakout_20"], forward_ret, train_dates)
        drawdown_rank_ic = _average_rank_ic(factor_bundle["raw_factors"]["drawdown_20"], forward_ret, train_dates)
        range_rank_ic = _average_rank_ic(factor_bundle["raw_factors"]["range_position_20"], forward_ret, train_dates)

        use_enhanced = (
            (not np.isnan(breakout_rank_ic) and breakout_rank_ic >= args.breakout_rankic_min)
            and (not np.isnan(drawdown_rank_ic) and drawdown_rank_ic >= args.drawdown_rankic_min)
            and (not np.isnan(range_rank_ic) and range_rank_ic >= args.range_position_rankic_min)
        )
        selected_profile = args.enhanced_profile if use_enhanced else args.baseline_profile
        rows.append(
            {
                "window_label": window_label,
                "focus_quadrant": args.focus_quadrant,
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "train_years": ",".join(map(str, train_years)),
                "breakout_rank_ic": breakout_rank_ic,
                "drawdown_rank_ic": drawdown_rank_ic,
                "range_position_rank_ic": range_rank_ic,
                "selected_profile": selected_profile,
            }
        )

    selection_df = pd.DataFrame(rows)
    active_index = close.loc[selection_df["test_start"].min():].index if not selection_df.empty else close.index
    selected_weights, selected_scores = _stitch_selected_profile_outputs(
        profile_artifacts=profile_artifacts,
        quadrant_selection_rows=selection_df,
        index=active_index,
        quadrant_series=quadrant_series,
        baseline_profile=args.baseline_profile,
    )

    equity_df, actions_df, metrics = backtest(
        close=close.loc[active_index],
        benchmark_close=benchmark_close.loc[active_index],
        target_weights=selected_weights.loc[active_index],
        target_scores=selected_scores.loc[active_index],
        config=cfg,
        regime_on=regime_on.loc[active_index],
    )

    metrics.update(
        {
            "activation_frequency": args.activation_frequency,
            "train_months": int(args.train_months),
            "ic_horizon": int(args.ic_horizon),
            "baseline_profile": args.baseline_profile,
            "enhanced_profile": args.enhanced_profile,
            "focus_quadrant": args.focus_quadrant,
            "breakout_rankic_min": float(args.breakout_rankic_min),
            "drawdown_rankic_min": float(args.drawdown_rankic_min),
            "range_position_rankic_min": float(args.range_position_rankic_min),
        }
    )

    out_root = Path(__file__).resolve().parents[1] / "output"
    run_name = args.experiment_tag.strip() or datetime.now().strftime("rule_based_activation_%Y%m%d_%H%M%S")
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    selection_df.to_csv(out_dir / "rule_activation_windows.csv", index=False, encoding="utf-8-sig")
    equity_df.to_csv(out_dir / "rule_activation_equity_curve.csv", encoding="utf-8-sig")
    if not actions_df.empty:
        actions_df.to_csv(out_dir / "rule_activation_actions.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "rule_activation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"输出目录: {out_dir}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if not selection_df.empty:
        print(selection_df.to_string(index=False))


if __name__ == "__main__":
    main()
