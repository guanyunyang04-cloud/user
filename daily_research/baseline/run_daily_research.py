from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from datetime import datetime

import pandas as pd

from daily_research.baseline.alpha import combine_scores, combine_scores_by_state
from daily_research.baseline.backtest import backtest
from daily_research.baseline.cli_utils import parse_csv_list, parse_stock_list
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import (
    load_daily_from_csv,
    load_industry_map_from_tq,
    load_daily_from_tq,
    load_style_map_from_tq,
    load_universe_from_tq,
    split_benchmark_from_universe,
)
from daily_research.baseline.evaluation import evaluate_factor_bundle
from daily_research.baseline.features import compute_factors
from daily_research.baseline.portfolio import build_target_weights
from daily_research.baseline.regime import apply_market_regime_filter, compute_market_regime_state
from daily_research.baseline.state_profiles import build_state_configs


def parse_args():
    parser = argparse.ArgumentParser(description="Daily research stage-1 runner")
    parser.add_argument("--data-source", choices=["tq", "csv"], default="tq")
    parser.add_argument("--csv-folder", default=None, help="CSV folder. Each stock should be one file.")
    parser.add_argument("--stocks", default=None, help="Comma-separated stock list, optional override.")
    parser.add_argument("--start-date", default="20180101")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--universe-scope", default="all_a")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--holding-count", type=int, default=8)
    parser.add_argument("--weighting-method", choices=["equal", "score"], default="equal")
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--rebalance-freq", default="1d")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--min-adv20", type=float, default=None, help="20日平均成交额下限，沿用数据源原始单位。")
    parser.add_argument("--min-price", type=float, default=None)
    parser.add_argument("--max-price", type=float, default=None)
    parser.add_argument("--market-regime-filter", action="store_true", help="启用基准趋势/波动市场状态过滤。")
    parser.add_argument("--regime-ma-window", type=int, default=60)
    parser.add_argument("--regime-vol-window", type=int, default=20)
    parser.add_argument("--regime-max-annual-vol", type=float, default=0.28)
    parser.add_argument(
        "--regime-quadrants",
        default="trend_up_low_vol,trend_up_high_vol",
        help="允许持仓的市场状态象限，逗号分隔。例如 trend_up_low_vol,trend_up_high_vol",
    )
    parser.add_argument("--industry-cap", action="store_true", help="启用单行业集中度上限。")
    parser.add_argument("--max-industry-weight", type=float, default=0.40)
    parser.add_argument("--style-cap", action="store_true", help="启用金融/高分红风格偏置约束。")
    parser.add_argument("--max-style-weight", type=float, default=0.60)
    parser.add_argument(
        "--state-alpha-profile",
        default="none",
        help="状态内动态权重方案，例如 up_low_breakout_v1；默认 none",
    )
    return parser.parse_args()


def _apply_rebalance_frequency(frame: pd.DataFrame, rebalance_freq: str) -> pd.DataFrame:
    rebalance_freq = str(rebalance_freq or "1d").lower().strip()
    if rebalance_freq in {"1d", "d", "daily"}:
        return frame.copy()
    if not rebalance_freq.endswith("d"):
        raise ValueError(f"Unsupported rebalance_freq: {rebalance_freq}")
    step = int(rebalance_freq[:-1])
    if step <= 1:
        return frame.copy()

    out = frame.copy()
    rebalance_idx = out.index[::step]
    rebalanced = out.loc[rebalance_idx].reindex(out.index).ffill()
    return rebalanced.fillna(0.0)


def _build_latest_scores(
    score: pd.DataFrame,
    target_weights: pd.DataFrame,
    group_scores: dict,
    filter_mask: pd.DataFrame,
) -> pd.DataFrame:
    latest_dt = score.dropna(how="all").index.max()
    if pd.isna(latest_dt):
        return pd.DataFrame(columns=["stock", "score", "target_weight", "filter_pass"])

    latest = pd.DataFrame({
        "stock": score.columns,
        "score": score.loc[latest_dt].reindex(score.columns).values,
        "target_weight": target_weights.loc[latest_dt].reindex(score.columns).values,
        "filter_pass": filter_mask.loc[latest_dt].reindex(score.columns).fillna(False).astype(bool).values,
    })
    for group_name, group_df in group_scores.items():
        latest[group_name] = group_df.loc[latest_dt].reindex(score.columns).values
    latest.insert(0, "date", latest_dt)
    latest = latest.sort_values(["score", "target_weight"], ascending=[False, False], na_position="last")
    return latest.reset_index(drop=True)


def main():
    args = parse_args()
    cfg = ResearchConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        universe_scope=args.universe_scope,
        benchmark=args.benchmark,
        holding_count=args.holding_count,
        weighting_method=args.weighting_method,
        rebalance_freq=args.rebalance_freq,
        score_threshold=args.score_threshold,
        enable_market_regime_filter=args.market_regime_filter,
        regime_ma_window=args.regime_ma_window,
        regime_vol_window=args.regime_vol_window,
        regime_max_annual_vol=args.regime_max_annual_vol,
        regime_allowed_quadrants=parse_csv_list(args.regime_quadrants),
        enable_industry_cap=args.industry_cap,
        max_industry_weight=args.max_industry_weight,
        enable_style_cap=args.style_cap,
        max_style_weight=args.max_style_weight,
    )
    if args.min_adv20 is not None:
        cfg.min_adv20 = float(args.min_adv20)
    if args.min_price is not None:
        cfg.min_price = float(args.min_price)
    if args.max_price is not None:
        cfg.max_price = float(args.max_price)

    stocks = parse_stock_list(args.stocks)
    if stocks:
        cfg.universe = stocks

    if args.data_source == "tq":
        if not cfg.universe and cfg.universe_scope == "all_a":
            print("[1/6] 正在从 TQ 加载全A股票池...")
            cfg.universe = load_universe_from_tq(cfg.universe_scope)
        elif not cfg.universe:
            raise ValueError("TQ 模式下，未指定 --stocks 时目前仅支持 --universe-scope all_a。")

        print(f"[2/6] 正在拉取日线数据，股票数: {len(cfg.universe)}，基准: {cfg.benchmark}")
        raw_df_dict = load_daily_from_tq(
            stock_list=cfg.universe,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            benchmark=cfg.benchmark,
        )
    else:
        if not args.csv_folder:
            raise ValueError("CSV 模式需要提供 --csv-folder。")
        print(f"[1/6] 正在从 CSV 加载数据目录: {args.csv_folder}")
        raw_df_dict = load_daily_from_csv(args.csv_folder)

    print("[3/6] 正在分离基准并计算因子...")
    df_dict, benchmark_close = split_benchmark_from_universe(raw_df_dict, cfg.benchmark)
    if cfg.universe:
        available = [stock for stock in cfg.universe if stock in df_dict["Close"].columns]
        if available:
            df_dict = {field: df.reindex(columns=available) for field, df in df_dict.items()}

    factor_bundle = compute_factors(df_dict)
    regime_state = compute_market_regime_state(benchmark_close, cfg)
    industry_map = None
    style_map = None
    if cfg.enable_industry_cap and args.data_source == "tq":
        print("[4/7] 正在加载行业映射...")
        industry_map = load_industry_map_from_tq(list(df_dict["Close"].columns))
    if cfg.enable_style_cap and args.data_source == "tq":
        print("[5/8] 正在加载风格映射...")
        style_map = load_style_map_from_tq(list(df_dict["Close"].columns))

    print("[6/8] 正在计算分组分数与目标组合...")
    state_configs = build_state_configs(cfg, args.state_alpha_profile)
    score, group_scores, filter_mask = combine_scores_by_state(
        factor_bundle,
        cfg,
        quadrant_series=regime_state["quadrant"],
        state_configs=state_configs,
    )
    target_weights = build_target_weights(score, cfg, industry_map=industry_map, style_map=style_map)
    target_weights = _apply_rebalance_frequency(target_weights, cfg.rebalance_freq)
    score_for_backtest = _apply_rebalance_frequency(score.fillna(0.0), cfg.rebalance_freq)
    if cfg.enable_market_regime_filter:
        target_weights, score_for_backtest = apply_market_regime_filter(
            target_weights=target_weights,
            target_scores=score_for_backtest,
            regime_state=regime_state,
        )

    print("[7/8] 正在执行回测...")
    equity_df, action_df, metrics = backtest(
        close=factor_bundle["raw_inputs"]["Close"],
        benchmark_close=benchmark_close,
        target_weights=target_weights,
        target_scores=score_for_backtest,
        config=cfg,
        regime_on=regime_state["regime_on"] if regime_state is not None else None,
    )

    print("[8/8] 正在生成因子评估与输出文件...")
    factor_ic_summary, factor_quantile_returns = evaluate_factor_bundle(
        raw_factors=factor_bundle["raw_factors"],
        score=score,
        close=factor_bundle["raw_inputs"]["Close"],
        quantiles=5,
    )
    latest_scores = _build_latest_scores(score, target_weights, group_scores, filter_mask)

    metrics.update(
        {
            "benchmark": cfg.benchmark,
            "universe_scope": cfg.universe_scope,
            "universe_size": int(len(df_dict["Close"].columns)),
            "date_count": int(len(equity_df)),
            "holding_count_target": int(cfg.holding_count),
            "weighting_method": cfg.weighting_method,
            "execution_mode": cfg.execution_mode,
            "rebalance_freq": cfg.rebalance_freq,
            "market_regime_filter": cfg.enable_market_regime_filter,
            "regime_allowed_quadrants": cfg.regime_allowed_quadrants,
            "industry_cap": cfg.enable_industry_cap,
            "max_industry_weight": cfg.max_industry_weight,
            "style_cap": cfg.enable_style_cap,
            "max_style_weight": cfg.max_style_weight,
            "state_alpha_profile": args.state_alpha_profile,
        }
    )

    base_dir = Path(__file__).resolve().parents[1]
    output_root = base_dir / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = args.experiment_tag.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    equity_df.to_csv(output_dir / "equity_curve.csv", encoding="utf-8-sig")
    action_df.to_csv(output_dir / "actions.csv", index=False, encoding="utf-8-sig")
    factor_ic_summary.to_csv(output_dir / "factor_ic_summary.csv", index=False, encoding="utf-8-sig")
    factor_quantile_returns.to_csv(output_dir / "factor_quantile_returns.csv", index=False, encoding="utf-8-sig")
    latest_scores.to_csv(output_dir / "latest_scores.csv", index=False, encoding="utf-8-sig")
    if regime_state is not None:
        regime_state.to_csv(output_dir / "regime_state.csv", encoding="utf-8-sig")
    if industry_map is not None:
        industry_map.rename_axis("stock").reset_index(name="industry").to_csv(
            output_dir / "industry_map.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if style_map is not None:
        style_map.reset_index(names="stock").to_csv(
            output_dir / "style_map.csv",
            index=False,
            encoding="utf-8-sig",
        )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("[完成] Daily research 第一阶段已跑通。")
    print(f"输出目录: {output_dir}")
    print(
        "核心结果: "
        f"累计收益={metrics['total_return']:.2%}, "
        f"超额收益={metrics['excess_total_return']:.2%}, "
        f"年化收益={metrics['annual_return']:.2%}, "
        f"超额夏普={metrics['excess_sharpe']:.3f}, "
        f"平均持仓={metrics['avg_holding_count']:.2f}"
    )


if __name__ == "__main__":
    main()
