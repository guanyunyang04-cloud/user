from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from daily_research.baseline.alpha import combine_scores_by_state
from daily_research.baseline.backtest import backtest
from daily_research.baseline.cli_utils import (
    load_stock_list_from_file,
    parse_csv_list,
    parse_horizon_weights,
    parse_int_tuple,
    parse_state_ensemble_weights,
    parse_state_horizon_profiles,
    parse_stock_list,
)
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
from daily_research.baseline.ml_alpha import MLAplhaConfig, blend_scores, build_ml_feature_bundle, rolling_ml_scores_multi
from daily_research.baseline.portfolio import build_target_weights
from daily_research.baseline.regime import apply_market_regime_filter, compute_market_regime_state
from daily_research.baseline.state_profiles import build_state_configs
from daily_research.execution.liquidity_universe import build_rolling_liquidity_membership
from daily_research.progress import StageProgress


def parse_args():
    parser = argparse.ArgumentParser(description="Advanced daily research runner: regime + ML cross-sectional ranking + portfolio backtest")
    parser.add_argument("--data-source", choices=["tq", "csv"], default="tq")
    parser.add_argument("--csv-folder", default=None)
    parser.add_argument("--stocks", default=None)
    parser.add_argument("--stocks-file", default=None, help="Path to txt/csv file containing stock codes.")
    parser.add_argument(
        "--rolling-liquidity-pool",
        choices=["liquid300", "liquid500", "liquid800"],
        default="",
        help="Use a historical rolling high-liquidity pool for formal research validation.",
    )
    parser.add_argument("--pool-rebalance-days", type=int, default=21, help="Trading-day frequency for rebuilding the historical rolling liquidity pool.")
    parser.add_argument("--pool-adv-window", type=int, default=20, help="ADV lookback window used for rolling liquidity pool construction.")
    parser.add_argument("--start-date", default="20210101")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--universe-scope", default="all_a")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--holding-count", type=int, default=5)
    parser.add_argument("--rebalance-freq", default="1d")
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--enhanced-profile", default="up_low_breakout_v2")

    parser.add_argument("--min-adv20", type=float, default=50_000.0)
    parser.add_argument("--min-price", type=float, default=2.0)
    parser.add_argument("--max-price", type=float, default=300.0)
    parser.add_argument("--max-weight", type=float, default=0.25)
    parser.add_argument("--score-threshold", type=float, default=0.0)

    parser.add_argument("--no-market-regime-filter", action="store_true")
    parser.add_argument("--regime-ma-window", type=int, default=60)
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
    parser.add_argument(
        "--ml-state-horizon-profiles",
        default="",
        help="按市场状态指定周期权重，例如 trend_up_low_vol=5:0.15,10:0.25,20:0.60;trend_up_high_vol=5:0.30,10:0.35,20:0.35",
    )
    parser.add_argument("--ml-train-window-days", type=int, default=504)
    parser.add_argument("--ml-retrain-every-days", type=int, default=21)
    parser.add_argument("--ml-min-train-dates", type=int, default=120)
    parser.add_argument("--ml-max-samples-per-day", type=int, default=600)
    parser.add_argument("--ml-max-train-rows", type=int, default=200000)
    parser.add_argument("--ml-random-seed", type=int, default=7)
    parser.add_argument("--ml-model-family", choices=["histgb", "etr", "lgbm"], default="histgb")
    parser.add_argument("--ensemble-ml-weight", type=float, default=0.70)
    parser.add_argument("--ensemble-none-weight", type=float, default=0.20)
    parser.add_argument("--ensemble-v2-weight", type=float, default=0.10)
    parser.add_argument(
        "--ensemble-state-weights",
        default="",
        help="按市场状态指定集成权重，例如 trend_up_low_vol=ml:0.60,none:0.25,v2:0.15;trend_up_high_vol=ml:0.80,none:0.15,v2:0.05",
    )
    return parser.parse_args()


def _build_latest_scores(
    final_score: pd.DataFrame,
    target_weights: pd.DataFrame,
    score_none: pd.DataFrame,
    score_enhanced: pd.DataFrame,
    ml_score: pd.DataFrame,
) -> pd.DataFrame:
    latest_dt = final_score.dropna(how="all").index.max()
    if pd.isna(latest_dt):
        return pd.DataFrame(columns=["stock", "final_score", "target_weight"])

    out = pd.DataFrame(
        {
            "date": latest_dt,
            "stock": final_score.columns,
            "final_score": final_score.loc[latest_dt].reindex(final_score.columns).values,
            "target_weight": target_weights.loc[latest_dt].reindex(final_score.columns).values,
            "score_none": score_none.loc[latest_dt].reindex(final_score.columns).values,
            "score_enhanced": score_enhanced.loc[latest_dt].reindex(final_score.columns).values,
            "ml_score": ml_score.loc[latest_dt].reindex(final_score.columns).values,
        }
    )
    return out.sort_values(["final_score", "target_weight"], ascending=[False, False], na_position="last").reset_index(drop=True)


def _subset_df_dict_to_stocks(df_dict: dict[str, pd.DataFrame], stocks: list[str]) -> dict[str, pd.DataFrame]:
    keep = [stock for stock in stocks if stock in df_dict["Close"].columns]
    return {field: frame.reindex(columns=keep) for field, frame in df_dict.items()}


def main():
    args = parse_args()
    if args.rolling_liquidity_pool and (args.stocks or args.stocks_file):
        raise ValueError("Use either a fixed --stocks/--stocks-file universe or --rolling-liquidity-pool, not both.")

    cfg = ResearchConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        universe_scope=args.universe_scope,
        benchmark=args.benchmark,
        execution_mode="next_open",
        holding_count=args.holding_count,
        weighting_method="score",
        rebalance_freq=args.rebalance_freq,
        score_threshold=args.score_threshold,
        max_weight=args.max_weight,
        min_adv20=args.min_adv20,
        min_price=args.min_price,
        max_price=args.max_price,
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

    stocks = parse_stock_list(args.stocks)
    file_stocks = load_stock_list_from_file(args.stocks_file)
    if stocks or file_stocks:
        stocks = list(dict.fromkeys(stocks + file_stocks))
    if stocks:
        cfg.universe = stocks

    ml_cfg = MLAplhaConfig(
        target_horizon=args.ml_target_horizon,
        target_horizons=parse_int_tuple(args.ml_target_horizons, args.ml_target_horizon),
        target_horizon_weights=parse_horizon_weights(args.ml_horizon_weights),
        state_horizon_weights=parse_state_horizon_profiles(args.ml_state_horizon_profiles),
        enhanced_profile=args.enhanced_profile,
        train_window_days=args.ml_train_window_days,
        retrain_every_days=args.ml_retrain_every_days,
        min_train_dates=args.ml_min_train_dates,
        max_samples_per_day=args.ml_max_samples_per_day,
        max_train_rows=args.ml_max_train_rows,
        random_seed=args.ml_random_seed,
        model_family=args.ml_model_family,
        ensemble_ml_weight=args.ensemble_ml_weight,
        ensemble_none_weight=args.ensemble_none_weight,
        ensemble_v2_weight=args.ensemble_v2_weight,
        state_ensemble_weights=parse_state_ensemble_weights(args.ensemble_state_weights),
        train_regime_only=cfg.enable_market_regime_filter,
        execution_mode=cfg.execution_mode,
    )

    if args.data_source == "tq":
        if args.rolling_liquidity_pool:
            print(f"[1/9] 正在从 TQ 加载滚动 {args.rolling_liquidity_pool} 研究池的基础宇宙...")
            cfg.universe = load_universe_from_tq(cfg.universe_scope)
        elif not cfg.universe and cfg.universe_scope == "all_a":
            print("[1/9] 正在从 TQ 加载全A股票池...")
            cfg.universe = load_universe_from_tq(cfg.universe_scope)
        elif not cfg.universe:
            raise ValueError("TQ 模式下，未指定 --stocks 时目前仅支持 --universe-scope all_a。")
        print(f"[2/9] 正在拉取日线数据，股票数: {len(cfg.universe)}，基准: {cfg.benchmark}")
        raw_df_dict = load_daily_from_tq(cfg.universe, cfg.start_date, cfg.end_date, benchmark=cfg.benchmark)
    else:
        if not args.csv_folder:
            raise ValueError("CSV 模式需要提供 --csv-folder。")
        print(f"[1/9] 正在从 CSV 加载数据目录: {args.csv_folder}")
        raw_df_dict = load_daily_from_csv(args.csv_folder)

    print("[3/9] 正在分离基准并准备研究输入...")
    benchmark_open = raw_df_dict["Open"][cfg.benchmark].copy()
    df_dict, benchmark_close = split_benchmark_from_universe(raw_df_dict, cfg.benchmark)
    rolling_pool_artifact = None
    rolling_membership_mask = None
    if args.rolling_liquidity_pool:
        print(f"[4/9] 正在构建历史滚动 {args.rolling_liquidity_pool} 研究池（每 {args.pool_rebalance_days} 个交易日重建）...")
        rolling_pool_artifact = build_rolling_liquidity_membership(
            close_frame=df_dict["Close"],
            amount_frame=df_dict["Amount"],
            pool_name=args.rolling_liquidity_pool,
            signal_start_date=cfg.start_date,
            signal_end_date=cfg.end_date,
            rebalance_every_days=args.pool_rebalance_days,
            adv_window=args.pool_adv_window,
            min_price=cfg.min_price,
            max_price=cfg.max_price,
        )
        rolling_membership_mask = rolling_pool_artifact.membership_frame
        rolling_union = rolling_membership_mask.columns[rolling_membership_mask.any(axis=0)].tolist()
        if not rolling_union:
            raise RuntimeError(f"Rolling {args.rolling_liquidity_pool} research pool is empty for the requested window.")
        df_dict = _subset_df_dict_to_stocks(df_dict, rolling_union)
        rolling_membership_mask = rolling_membership_mask.reindex(index=df_dict["Close"].index, columns=df_dict["Close"].columns).fillna(False)
    factor_bundle = compute_factors(df_dict)
    regime_state = compute_market_regime_state(benchmark_close, cfg)

    industry_map = None
    style_map = None
    if cfg.enable_industry_cap and args.data_source == "tq":
        print("[5/9] 正在加载行业映射...")
        industry_map = load_industry_map_from_tq(list(df_dict["Close"].columns))
    if cfg.enable_style_cap and args.data_source == "tq":
        print("[5/9] 正在加载风格映射...")
        style_map = load_style_map_from_tq(list(df_dict["Close"].columns))

    print("[6/9] 正在计算稳健基线分数与增强分数...")
    score_none, _, filter_mask = combine_scores_by_state(
        factor_bundle,
        cfg,
        quadrant_series=regime_state["quadrant"],
        state_configs=build_state_configs(cfg, "none"),
    )
    score_enhanced, _, _ = combine_scores_by_state(
        factor_bundle,
        cfg,
        quadrant_series=regime_state["quadrant"],
        state_configs=build_state_configs(cfg, args.enhanced_profile),
    )
    if rolling_membership_mask is not None:
        filter_mask = filter_mask & rolling_membership_mask
        score_none = score_none.where(rolling_membership_mask)
        score_enhanced = score_enhanced.where(rolling_membership_mask)

    print("[7/9] 正在执行滚动机器学习横截面排序...")
    feature_frames, market_features = build_ml_feature_bundle(factor_bundle, regime_state, score_none, score_enhanced)
    ml_score, training_log = rolling_ml_scores_multi(
        feature_frames=feature_frames,
        market_features=market_features,
        close=factor_bundle["raw_inputs"]["Close"],
        benchmark_close=benchmark_close,
        open_df=factor_bundle["raw_inputs"]["Open"],
        benchmark_open=benchmark_open,
        filter_mask=filter_mask,
        regime_state=regime_state,
        config=ml_cfg,
    )
    final_score = blend_scores(ml_score, score_none, score_enhanced, ml_cfg, quadrant_series=regime_state["quadrant"])
    if rolling_membership_mask is not None:
        final_score = final_score.where(rolling_membership_mask)

    print("[8/9] 正在构建组合并执行回测...")
    target_weights = build_target_weights(final_score, cfg, industry_map=industry_map, style_map=style_map)
    score_for_backtest = final_score.fillna(0.0)
    if cfg.enable_market_regime_filter:
        target_weights, score_for_backtest = apply_market_regime_filter(target_weights, score_for_backtest, regime_state)
    equity_df, action_df, metrics = backtest(
        close=factor_bundle["raw_inputs"]["Close"],
        benchmark_close=benchmark_close,
        open_df=factor_bundle["raw_inputs"]["Open"],
        benchmark_open=benchmark_open,
        target_weights=target_weights,
        target_scores=score_for_backtest,
        config=cfg,
        regime_on=regime_state["regime_on"],
    )

    print("[9/9] 正在整理输出...")
    factor_ic_summary, factor_quantile_returns = evaluate_factor_bundle(
        raw_factors=factor_bundle["raw_factors"],
        score=final_score,
        close=factor_bundle["raw_inputs"]["Close"],
        quantiles=5,
    )
    latest_scores = _build_latest_scores(final_score, target_weights, score_none, score_enhanced, ml_score)

    metrics.update(
        {
            "framework": "advanced_ml_research",
            "benchmark": cfg.benchmark,
            "universe_scope": cfg.universe_scope,
            "universe_size": int(len(df_dict["Close"].columns)),
            "date_count": int(len(equity_df)),
            "execution_mode": cfg.execution_mode,
            "holding_count_target": int(cfg.holding_count),
            "rebalance_freq": cfg.rebalance_freq,
            "ml_target_horizon": int(ml_cfg.target_horizon),
            "ml_target_horizons": list(ml_cfg.target_horizons),
            "ml_horizon_weights": {str(k): float(v) for k, v in (ml_cfg.target_horizon_weights or {}).items()},
            "ml_state_horizon_profiles": {
                str(state): {str(k): float(v) for k, v in weights.items()}
                for state, weights in (ml_cfg.state_horizon_weights or {}).items()
            },
            "ml_train_window_days": int(ml_cfg.train_window_days),
            "ml_retrain_every_days": int(ml_cfg.retrain_every_days),
            "ml_max_train_rows": int(ml_cfg.max_train_rows),
            "ml_model_family": str(ml_cfg.model_family),
            "ensemble_ml_weight": float(ml_cfg.ensemble_ml_weight),
            "ensemble_none_weight": float(ml_cfg.ensemble_none_weight),
            "ensemble_v2_weight": float(ml_cfg.ensemble_v2_weight),
            "state_ensemble_weights": {
                str(state): {str(k): float(v) for k, v in weights.items()}
                for state, weights in (ml_cfg.state_ensemble_weights or {}).items()
            },
            "enhanced_profile": str(args.enhanced_profile),
            "market_regime_filter": cfg.enable_market_regime_filter,
            "regime_allowed_quadrants": cfg.regime_allowed_quadrants,
            "style_cap": cfg.enable_style_cap,
            "max_style_weight": cfg.max_style_weight,
            "industry_cap": cfg.enable_industry_cap,
            "max_industry_weight": cfg.max_industry_weight,
            "rolling_liquidity_pool": args.rolling_liquidity_pool or "",
            "rolling_pool_rebalance_days": int(args.pool_rebalance_days),
            "rolling_pool_adv_window": int(args.pool_adv_window),
            "rolling_pool_union_size": 0 if rolling_pool_artifact is None else int(rolling_membership_mask.columns[rolling_membership_mask.any(axis=0)].size),
            "rolling_pool_rebalance_count": 0 if rolling_pool_artifact is None else int(len(rolling_pool_artifact.schedule_df)),
        }
    )

    base_dir = Path(__file__).resolve().parents[1]
    output_root = base_dir / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = args.experiment_tag.strip() or datetime.now().strftime("advanced_ml_%Y%m%d_%H%M%S")
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    equity_df.to_csv(output_dir / "equity_curve.csv", encoding="utf-8-sig")
    action_df.to_csv(output_dir / "actions.csv", index=False, encoding="utf-8-sig")
    factor_ic_summary.to_csv(output_dir / "factor_ic_summary.csv", index=False, encoding="utf-8-sig")
    factor_quantile_returns.to_csv(output_dir / "factor_quantile_returns.csv", index=False, encoding="utf-8-sig")
    latest_scores.to_csv(output_dir / "latest_scores.csv", index=False, encoding="utf-8-sig")
    training_log.to_csv(output_dir / "training_log.csv", index=False, encoding="utf-8-sig")
    regime_state.to_csv(output_dir / "regime_state.csv", encoding="utf-8-sig")
    if rolling_pool_artifact is not None:
        rolling_pool_artifact.schedule_df.to_csv(output_dir / "rolling_liquidity_schedule.csv", index=False, encoding="utf-8-sig")
        rolling_pool_artifact.summary_df.to_csv(output_dir / "rolling_liquidity_summary.csv", index=False, encoding="utf-8-sig")
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"输出目录: {output_dir}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main_with_progress():
    args = parse_args()
    if args.rolling_liquidity_pool and (args.stocks or args.stocks_file):
        raise ValueError("Use either a fixed --stocks/--stocks-file universe or --rolling-liquidity-pool, not both.")

    cfg = ResearchConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        universe_scope=args.universe_scope,
        benchmark=args.benchmark,
        execution_mode="next_open",
        holding_count=args.holding_count,
        weighting_method="score",
        rebalance_freq=args.rebalance_freq,
        score_threshold=args.score_threshold,
        max_weight=args.max_weight,
        min_adv20=args.min_adv20,
        min_price=args.min_price,
        max_price=args.max_price,
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

    stocks = parse_stock_list(args.stocks)
    file_stocks = load_stock_list_from_file(args.stocks_file)
    if stocks or file_stocks:
        stocks = list(dict.fromkeys(stocks + file_stocks))
    if stocks:
        cfg.universe = stocks

    ml_cfg = MLAplhaConfig(
        target_horizon=args.ml_target_horizon,
        target_horizons=parse_int_tuple(args.ml_target_horizons, args.ml_target_horizon),
        target_horizon_weights=parse_horizon_weights(args.ml_horizon_weights),
        state_horizon_weights=parse_state_horizon_profiles(args.ml_state_horizon_profiles),
        enhanced_profile=args.enhanced_profile,
        train_window_days=args.ml_train_window_days,
        retrain_every_days=args.ml_retrain_every_days,
        min_train_dates=args.ml_min_train_dates,
        max_samples_per_day=args.ml_max_samples_per_day,
        max_train_rows=args.ml_max_train_rows,
        random_seed=args.ml_random_seed,
        model_family=args.ml_model_family,
        ensemble_ml_weight=args.ensemble_ml_weight,
        ensemble_none_weight=args.ensemble_none_weight,
        ensemble_v2_weight=args.ensemble_v2_weight,
        state_ensemble_weights=parse_state_ensemble_weights(args.ensemble_state_weights),
        train_regime_only=cfg.enable_market_regime_filter,
        execution_mode=cfg.execution_mode,
    )

    with StageProgress(total=9, label="高级研究流程") as progress:
        with progress.stage("准备研究宇宙", f"source={args.data_source}"):
            if args.data_source == "tq":
                if args.rolling_liquidity_pool:
                    cfg.universe = load_universe_from_tq(cfg.universe_scope)
                elif not cfg.universe and cfg.universe_scope == "all_a":
                    cfg.universe = load_universe_from_tq(cfg.universe_scope)
                elif not cfg.universe:
                    raise ValueError("TQ mode without --stocks currently requires --universe-scope all_a.")
            else:
                if not args.csv_folder:
                    raise ValueError("CSV mode requires --csv-folder.")
                progress.log(f"csv folder: {args.csv_folder}")

        with progress.stage("读取股票行情", f"stocks={len(cfg.universe)} benchmark={cfg.benchmark}"):
            if args.data_source == "tq":
                raw_df_dict = load_daily_from_tq(
                    cfg.universe,
                    cfg.start_date,
                    cfg.end_date,
                    benchmark=cfg.benchmark,
                    progress_desc="读取股票日线",
                    progress_position=1,
                )
            else:
                raw_df_dict = load_daily_from_csv(
                    args.csv_folder,
                    progress_desc="读取CSV股票数据",
                    progress_position=1,
                )

        with progress.stage("构建研究输入", f"benchmark={cfg.benchmark}"):
            benchmark_open = raw_df_dict["Open"][cfg.benchmark].copy()
            df_dict, benchmark_close = split_benchmark_from_universe(raw_df_dict, cfg.benchmark)

        rolling_pool_artifact = None
        rolling_membership_mask = None
        with progress.stage("处理滚动流动性研究池", args.rolling_liquidity_pool or "skip"):
            if args.rolling_liquidity_pool:
                rolling_pool_artifact = build_rolling_liquidity_membership(
                    close_frame=df_dict["Close"],
                    amount_frame=df_dict["Amount"],
                    pool_name=args.rolling_liquidity_pool,
                    signal_start_date=cfg.start_date,
                    signal_end_date=cfg.end_date,
                    rebalance_every_days=args.pool_rebalance_days,
                    adv_window=args.pool_adv_window,
                    min_price=cfg.min_price,
                    max_price=cfg.max_price,
                )
                rolling_membership_mask = rolling_pool_artifact.membership_frame
                rolling_union = rolling_membership_mask.columns[rolling_membership_mask.any(axis=0)].tolist()
                if not rolling_union:
                    raise RuntimeError(
                        f"Rolling {args.rolling_liquidity_pool} research pool is empty for the requested window."
                    )
                df_dict = _subset_df_dict_to_stocks(df_dict, rolling_union)
                rolling_membership_mask = rolling_membership_mask.reindex(
                    index=df_dict["Close"].index,
                    columns=df_dict["Close"].columns,
                ).fillna(False)

        industry_map = None
        style_map = None
        with progress.stage("加载约束映射", "industry/style"):
            if cfg.enable_industry_cap and args.data_source == "tq":
                industry_map = load_industry_map_from_tq(list(df_dict["Close"].columns))
            if cfg.enable_style_cap and args.data_source == "tq":
                style_map = load_style_map_from_tq(list(df_dict["Close"].columns))

        with progress.stage("计算基础因子分数", args.enhanced_profile):
            factor_bundle = compute_factors(df_dict)
            regime_state = compute_market_regime_state(benchmark_close, cfg)
            score_none, _, filter_mask = combine_scores_by_state(
                factor_bundle,
                cfg,
                quadrant_series=regime_state["quadrant"],
                state_configs=build_state_configs(cfg, "none"),
            )
            score_enhanced, _, _ = combine_scores_by_state(
                factor_bundle,
                cfg,
                quadrant_series=regime_state["quadrant"],
                state_configs=build_state_configs(cfg, args.enhanced_profile),
            )
            if rolling_membership_mask is not None:
                filter_mask = filter_mask & rolling_membership_mask
                score_none = score_none.where(rolling_membership_mask)
                score_enhanced = score_enhanced.where(rolling_membership_mask)

        with progress.stage("滚动训练横截面ML排序", ml_cfg.model_family):
            feature_frames, market_features = build_ml_feature_bundle(
                factor_bundle,
                regime_state,
                score_none,
                score_enhanced,
            )
            ml_score, training_log = rolling_ml_scores_multi(
                feature_frames=feature_frames,
                market_features=market_features,
                close=factor_bundle["raw_inputs"]["Close"],
                benchmark_close=benchmark_close,
                open_df=factor_bundle["raw_inputs"]["Open"],
                benchmark_open=benchmark_open,
                filter_mask=filter_mask,
                regime_state=regime_state,
                config=ml_cfg,
            )
            final_score = blend_scores(
                ml_score,
                score_none,
                score_enhanced,
                ml_cfg,
                quadrant_series=regime_state["quadrant"],
            )
            if rolling_membership_mask is not None:
                final_score = final_score.where(rolling_membership_mask)

        with progress.stage("回测目标组合", cfg.rebalance_freq):
            target_weights = build_target_weights(final_score, cfg, industry_map=industry_map, style_map=style_map)
            score_for_backtest = final_score.fillna(0.0)
            if cfg.enable_market_regime_filter:
                target_weights, score_for_backtest = apply_market_regime_filter(
                    target_weights,
                    score_for_backtest,
                    regime_state,
                )
            equity_df, action_df, metrics = backtest(
                close=factor_bundle["raw_inputs"]["Close"],
                benchmark_close=benchmark_close,
                open_df=factor_bundle["raw_inputs"]["Open"],
                benchmark_open=benchmark_open,
                target_weights=target_weights,
                target_scores=score_for_backtest,
                config=cfg,
                regime_on=regime_state["regime_on"],
            )

        with progress.stage("整理研究输出", "csv/json"):
            factor_ic_summary, factor_quantile_returns = evaluate_factor_bundle(
                raw_factors=factor_bundle["raw_factors"],
                score=final_score,
                close=factor_bundle["raw_inputs"]["Close"],
                quantiles=5,
            )
            latest_scores = _build_latest_scores(final_score, target_weights, score_none, score_enhanced, ml_score)

    metrics.update(
        {
            "framework": "advanced_ml_research",
            "benchmark": cfg.benchmark,
            "universe_scope": cfg.universe_scope,
            "universe_size": int(len(df_dict["Close"].columns)),
            "date_count": int(len(equity_df)),
            "execution_mode": cfg.execution_mode,
            "holding_count_target": int(cfg.holding_count),
            "rebalance_freq": cfg.rebalance_freq,
            "ml_target_horizon": int(ml_cfg.target_horizon),
            "ml_target_horizons": list(ml_cfg.target_horizons),
            "ml_horizon_weights": {str(k): float(v) for k, v in (ml_cfg.target_horizon_weights or {}).items()},
            "ml_state_horizon_profiles": {
                str(state): {str(k): float(v) for k, v in weights.items()}
                for state, weights in (ml_cfg.state_horizon_weights or {}).items()
            },
            "ml_train_window_days": int(ml_cfg.train_window_days),
            "ml_retrain_every_days": int(ml_cfg.retrain_every_days),
            "ml_max_train_rows": int(ml_cfg.max_train_rows),
            "ml_model_family": str(ml_cfg.model_family),
            "ensemble_ml_weight": float(ml_cfg.ensemble_ml_weight),
            "ensemble_none_weight": float(ml_cfg.ensemble_none_weight),
            "ensemble_v2_weight": float(ml_cfg.ensemble_v2_weight),
            "state_ensemble_weights": {
                str(state): {str(k): float(v) for k, v in weights.items()}
                for state, weights in (ml_cfg.state_ensemble_weights or {}).items()
            },
            "enhanced_profile": str(args.enhanced_profile),
            "market_regime_filter": cfg.enable_market_regime_filter,
            "regime_allowed_quadrants": cfg.regime_allowed_quadrants,
            "style_cap": cfg.enable_style_cap,
            "max_style_weight": cfg.max_style_weight,
            "industry_cap": cfg.enable_industry_cap,
            "max_industry_weight": cfg.max_industry_weight,
            "rolling_liquidity_pool": args.rolling_liquidity_pool or "",
            "rolling_pool_rebalance_days": int(args.pool_rebalance_days),
            "rolling_pool_adv_window": int(args.pool_adv_window),
            "rolling_pool_union_size": 0
            if rolling_pool_artifact is None
            else int(rolling_membership_mask.columns[rolling_membership_mask.any(axis=0)].size),
            "rolling_pool_rebalance_count": 0
            if rolling_pool_artifact is None
            else int(len(rolling_pool_artifact.schedule_df)),
        }
    )

    base_dir = Path(__file__).resolve().parents[1]
    output_root = base_dir / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = args.experiment_tag.strip() or datetime.now().strftime("advanced_ml_%Y%m%d_%H%M%S")
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    equity_df.to_csv(output_dir / "equity_curve.csv", encoding="utf-8-sig")
    action_df.to_csv(output_dir / "actions.csv", index=False, encoding="utf-8-sig")
    factor_ic_summary.to_csv(output_dir / "factor_ic_summary.csv", index=False, encoding="utf-8-sig")
    factor_quantile_returns.to_csv(output_dir / "factor_quantile_returns.csv", index=False, encoding="utf-8-sig")
    latest_scores.to_csv(output_dir / "latest_scores.csv", index=False, encoding="utf-8-sig")
    training_log.to_csv(output_dir / "training_log.csv", index=False, encoding="utf-8-sig")
    regime_state.to_csv(output_dir / "regime_state.csv", encoding="utf-8-sig")
    if rolling_pool_artifact is not None:
        rolling_pool_artifact.schedule_df.to_csv(
            output_dir / "rolling_liquidity_schedule.csv",
            index=False,
            encoding="utf-8-sig",
        )
        rolling_pool_artifact.summary_df.to_csv(
            output_dir / "rolling_liquidity_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"输出目录: {output_dir}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


main = main_with_progress


if __name__ == "__main__":
    main()
