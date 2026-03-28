import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from datetime import datetime
from pathlib import Path

from daily_research.baseline.alpha import combine_scores_by_state
from daily_research.baseline.backtest import backtest
from daily_research.baseline.cli_utils import parse_csv_list
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import (
    load_daily_from_tq,
    load_style_map_from_tq,
    load_universe_from_tq,
    split_benchmark_from_universe,
)
from daily_research.baseline.features import compute_factors
from daily_research.baseline.portfolio import build_target_weights
from daily_research.baseline.regime import apply_market_regime_filter, compute_market_regime_state, resolve_regime_label_series
from daily_research.baseline.regime_analysis import classify_market_quadrants, summarize_strategy_by_state, summarize_year_states
from daily_research.baseline.state_profiles import build_state_configs, validate_state_profile_selector


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze strategy performance by market states")
    parser.add_argument("--start-date", default="20210101")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--weighting-method", choices=["equal", "score"], default="score")
    parser.add_argument("--rebalance-freq", default="5d")
    parser.add_argument("--market-regime-filter", action="store_true")
    parser.add_argument("--regime-ma-window", type=int, default=60)
    parser.add_argument("--regime-vol-window", type=int, default=20)
    parser.add_argument("--regime-max-annual-vol", type=float, default=0.32)
    parser.add_argument("--regime-trend-flat-band", type=float, default=0.01)
    parser.add_argument("--regime-vol-transition-band", type=float, default=0.10)
    parser.add_argument(
        "--regime-state-selector",
        choices=["quadrant", "market_state", "trend_bucket", "vol_bucket"],
        default="quadrant",
        help="按哪一列市场状态标签做分组分析；默认 quadrant。",
    )
    parser.add_argument(
        "--regime-quadrants",
        default="trend_up_low_vol,trend_up_high_vol",
        help="允许持仓的市场状态筛选器，逗号分隔。",
    )
    parser.add_argument("--style-cap", action="store_true")
    parser.add_argument("--max-style-weight", type=float, default=0.50)
    parser.add_argument("--state-alpha-profile", default="none")
    return parser.parse_args()


def _apply_rebalance_frequency(frame, rebalance_freq: str):
    rebalance_freq = str(rebalance_freq or "1d").lower().strip()
    if rebalance_freq in {"1d", "d", "daily"}:
        return frame.copy()
    step = int(rebalance_freq[:-1])
    return frame.loc[frame.index[::step]].reindex(frame.index).ffill().fillna(0.0)


def main():
    args = parse_args()
    validate_state_profile_selector(args.state_alpha_profile, args.regime_state_selector)
    cfg = ResearchConfig(
        start_date=args.start_date,
        benchmark=args.benchmark,
        universe_scope="all_a",
        weighting_method=args.weighting_method,
        rebalance_freq=args.rebalance_freq,
        enable_market_regime_filter=args.market_regime_filter,
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
    raw_df_dict = load_daily_from_tq(universe, args.start_date, benchmark=args.benchmark)
    df_dict, benchmark_close = split_benchmark_from_universe(raw_df_dict, args.benchmark)

    print("[3/6] 计算因子和分数...")
    factor_bundle = compute_factors(df_dict)
    regime_state = compute_market_regime_state(benchmark_close, cfg)
    state_label_series = resolve_regime_label_series(regime_state, cfg.regime_state_selector)
    state_configs = build_state_configs(cfg, args.state_alpha_profile)
    score, _, _ = combine_scores_by_state(
        factor_bundle,
        cfg,
        quadrant_series=state_label_series,
        state_configs=state_configs,
    )

    style_map = None
    if cfg.enable_style_cap:
        print("[4/6] 加载风格映射...")
        style_map = load_style_map_from_tq(list(df_dict["Close"].columns))

    print("[5/6] 构建组合并执行回测...")
    target_weights = build_target_weights(score, cfg, style_map=style_map)
    target_weights = _apply_rebalance_frequency(target_weights, cfg.rebalance_freq)
    score_bt = _apply_rebalance_frequency(score.fillna(0.0), cfg.rebalance_freq)

    if cfg.enable_market_regime_filter:
        target_weights, score_bt = apply_market_regime_filter(target_weights, score_bt, regime_state)

    equity_df, _, metrics = backtest(
        close=factor_bundle["raw_inputs"]["Close"],
        benchmark_close=benchmark_close,
        target_weights=target_weights,
        target_scores=score_bt,
        config=cfg,
        regime_on=regime_state["regime_on"] if cfg.enable_market_regime_filter else None,
    )

    print("[6/6] 生成市场状态分析报表...")
    regime_labels = classify_market_quadrants(
        benchmark_close=benchmark_close.loc[equity_df.index],
        ma_window=cfg.regime_ma_window,
        vol_window=cfg.regime_vol_window,
        vol_threshold=cfg.regime_max_annual_vol,
        trend_flat_band=cfg.regime_trend_flat_band,
        vol_transition_band=cfg.regime_vol_transition_band,
        state_selector=cfg.regime_state_selector,
    )
    state_summary = summarize_strategy_by_state(equity_df, regime_labels, label_column="state_label")
    year_state_summary = summarize_year_states(equity_df, regime_labels, label_column="state_label")
    metrics["regime_state_selector"] = cfg.regime_state_selector

    base_dir = Path(__file__).resolve().parents[1] / "output"
    run_name = args.experiment_tag.strip() or datetime.now().strftime("regime_state_%Y%m%d_%H%M%S")
    out_dir = base_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    equity_df.to_csv(out_dir / "equity_curve.csv", encoding="utf-8-sig")
    regime_labels.to_csv(out_dir / "regime_state.csv", encoding="utf-8-sig")
    state_summary.to_csv(out_dir / "state_summary.csv", index=False, encoding="utf-8-sig")
    year_state_summary.to_csv(out_dir / "year_state_summary.csv", index=False, encoding="utf-8-sig")
    if cfg.regime_state_selector == "quadrant":
        regime_labels.to_csv(out_dir / "quadrant_state.csv", encoding="utf-8-sig")
        state_summary.to_csv(out_dir / "quadrant_summary.csv", index=False, encoding="utf-8-sig")
        year_state_summary.to_csv(out_dir / "year_quadrant_summary.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"输出目录: {out_dir}")
    print(state_summary.to_string(index=False))


if __name__ == "__main__":
    main()
