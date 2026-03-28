from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from daily_research.baseline.alpha import combine_scores_by_state
from daily_research.baseline.backtest import backtest
from daily_research.baseline.cli_utils import parse_csv_list
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import load_daily_from_tq, load_style_map_from_tq, load_universe_from_tq, split_benchmark_from_universe
from daily_research.baseline.features import compute_factors
from daily_research.baseline.portfolio import build_target_weights
from daily_research.baseline.regime import apply_market_regime_filter, compute_market_regime_state, resolve_regime_label_series
from daily_research.baseline.state_profiles import build_state_configs, validate_state_profile_selector


def parse_args():
    parser = argparse.ArgumentParser(description="Frozen-date validation for state alpha profiles")
    parser.add_argument("--start-date", default="20210101")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--train-end", default="20231231")
    parser.add_argument("--test-start", default="20240101")
    parser.add_argument("--test-end", default="")
    parser.add_argument("--rebalance-freq", default="5d")
    parser.add_argument("--regime-ma-window", type=int, default=60)
    parser.add_argument("--regime-vol-window", type=int, default=20)
    parser.add_argument("--regime-max-annual-vol", type=float, default=0.32)
    parser.add_argument("--regime-trend-flat-band", type=float, default=0.01)
    parser.add_argument("--regime-vol-transition-band", type=float, default=0.10)
    parser.add_argument(
        "--regime-state-selector",
        choices=["quadrant", "market_state", "trend_bucket", "vol_bucket"],
        default="quadrant",
        help="状态标签来源；当前非 none profile 仍只支持 quadrant。",
    )
    parser.add_argument("--regime-quadrants", default="trend_up_low_vol,trend_up_high_vol")
    parser.add_argument("--style-cap", action="store_true")
    parser.add_argument("--max-style-weight", type=float, default=0.50)
    parser.add_argument("--profiles", default="none,up_low_breakout_v1,up_dual_v1")
    parser.add_argument("--experiment-tag", default="")
    return parser.parse_args()


def _apply_rebalance_frequency(frame: pd.DataFrame, rebalance_freq: str) -> pd.DataFrame:
    rebalance_freq = str(rebalance_freq or "1d").lower().strip()
    if rebalance_freq in {"1d", "d", "daily"}:
        return frame.copy()
    step = int(rebalance_freq[:-1])
    return frame.loc[frame.index[::step]].reindex(frame.index).ffill().fillna(0.0)


def _build_profile_artifacts(
    factor_bundle: Dict[str, Dict[str, pd.DataFrame]],
    benchmark_close: pd.Series,
    cfg: ResearchConfig,
    profile_name: str,
    style_map: pd.DataFrame | None,
):
    regime_state = compute_market_regime_state(benchmark_close, cfg)
    state_label_series = resolve_regime_label_series(regime_state, cfg.regime_state_selector)
    state_configs = build_state_configs(cfg, profile_name)
    score, _, _ = combine_scores_by_state(
        factor_bundle,
        cfg,
        quadrant_series=state_label_series,
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


def _slice_backtest(
    close: pd.DataFrame,
    benchmark_close: pd.Series,
    target_weights: pd.DataFrame,
    target_scores: pd.DataFrame,
    regime_on: pd.Series,
    cfg: ResearchConfig,
    start_dt: str,
    end_dt: str,
):
    close_slice = close.loc[start_dt:end_dt]
    benchmark_slice = benchmark_close.loc[start_dt:end_dt]
    weight_slice = target_weights.loc[start_dt:end_dt]
    score_slice = target_scores.loc[start_dt:end_dt]
    regime_slice = regime_on.loc[start_dt:end_dt]
    return backtest(
        close=close_slice,
        benchmark_close=benchmark_slice,
        target_weights=weight_slice,
        target_scores=score_slice,
        config=cfg,
        regime_on=regime_slice,
    )


def main():
    args = parse_args()
    profiles = parse_csv_list(args.profiles)
    for profile_name in profiles:
        validate_state_profile_selector(profile_name, args.regime_state_selector)
    cfg = ResearchConfig(
        start_date=args.start_date,
        end_date=args.test_end,
        benchmark=args.benchmark,
        universe_scope="all_a",
        weighting_method="score",
        rebalance_freq=args.rebalance_freq,
        enable_market_regime_filter=True,
        regime_ma_window=args.regime_ma_window,
        regime_vol_window=args.regime_vol_window,
        regime_max_annual_vol=args.regime_max_annual_vol,
        regime_trend_flat_band=args.regime_trend_flat_band,
        regime_vol_transition_band=args.regime_vol_transition_band,
        regime_state_selector=args.regime_state_selector,
        regime_allowed_quadrants=parse_csv_list(args.regime_quadrants),
        enable_style_cap=args.style_cap,
        max_style_weight=args.max_style_weight,
    )

    print("[1/6] 加载全A股票池...")
    universe = load_universe_from_tq("all_a")
    print(f"[2/6] 拉取日线数据，股票数: {len(universe)}")
    raw_df_dict = load_daily_from_tq(universe, args.start_date, args.test_end, benchmark=args.benchmark)
    df_dict, benchmark_close = split_benchmark_from_universe(raw_df_dict, args.benchmark)

    print("[3/6] 计算因子...")
    factor_bundle = compute_factors(df_dict)
    close = factor_bundle["raw_inputs"]["Close"]

    style_map = None
    if cfg.enable_style_cap:
        print("[4/6] 加载风格映射...")
        style_map = load_style_map_from_tq(list(close.columns))

    print("[5/6] 预计算各 profile 的目标组合...")
    profile_artifacts = {
        profile: _build_profile_artifacts(factor_bundle, benchmark_close, cfg, profile, style_map)
        for profile in profiles
    }

    print("[6/6] 执行冻结时点验证...")
    train_rows = []
    test_rows = []
    selected_profile = None
    selected_train_score = None
    selected_test_equity = None
    selected_test_actions = None
    selected_test_metrics = None

    for profile_name, artifacts in profile_artifacts.items():
        _, _, train_metrics = _slice_backtest(
            close=close,
            benchmark_close=benchmark_close,
            target_weights=artifacts["target_weights"],
            target_scores=artifacts["target_scores"],
            regime_on=artifacts["regime_state"]["regime_on"],
            cfg=cfg,
            start_dt=args.start_date,
            end_dt=args.train_end,
        )
        train_rows.append({"profile": profile_name, **train_metrics})

        test_equity, test_actions, test_metrics = _slice_backtest(
            close=close,
            benchmark_close=benchmark_close,
            target_weights=artifacts["target_weights"],
            target_scores=artifacts["target_scores"],
            regime_on=artifacts["regime_state"]["regime_on"],
            cfg=cfg,
            start_dt=args.test_start,
            end_dt=args.test_end or str(close.index.max().date()),
        )
        test_rows.append({"profile": profile_name, **test_metrics})

        rank_value = float(train_metrics.get("excess_sharpe", 0.0))
        if selected_train_score is None or rank_value > selected_train_score:
            selected_train_score = rank_value
            selected_profile = profile_name
            selected_test_equity = test_equity
            selected_test_actions = test_actions
            selected_test_metrics = test_metrics

    out_root = Path(__file__).resolve().parents[1] / "output"
    run_name = args.experiment_tag.strip() or datetime.now().strftime("frozen_validation_%Y%m%d_%H%M%S")
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(train_rows).sort_values("excess_sharpe", ascending=False)
    test_df = pd.DataFrame(test_rows)
    train_df.to_csv(out_dir / "train_profile_metrics.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(out_dir / "test_profile_metrics.csv", index=False, encoding="utf-8-sig")
    if selected_test_equity is not None:
        selected_test_equity.to_csv(out_dir / "selected_profile_equity_curve.csv", encoding="utf-8-sig")
    if selected_test_actions is not None and not selected_test_actions.empty:
        selected_test_actions.to_csv(out_dir / "selected_profile_actions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "train_period": [args.start_date, args.train_end],
        "test_period": [args.test_start, args.test_end or str(close.index.max().date())],
        "selected_profile": selected_profile,
        "selected_train_excess_sharpe": selected_train_score,
        "profiles": profiles,
        "regime_state_selector": cfg.regime_state_selector,
        "selected_test_metrics": selected_test_metrics or {},
    }
    with open(out_dir / "frozen_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"输出目录: {out_dir}")
    print("训练期排序：")
    print(train_df[["profile", "excess_total_return", "excess_annual_return", "excess_sharpe", "excess_max_drawdown"]].to_string(index=False))
    print("测试期结果：")
    print(test_df[["profile", "total_return", "excess_total_return", "excess_annual_return", "excess_sharpe", "excess_max_drawdown"]].to_string(index=False))
    print(f"最终选中 profile: {selected_profile}")


if __name__ == "__main__":
    main()
