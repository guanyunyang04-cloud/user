from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from datetime import datetime

import pandas as pd

from daily_research.baseline.alpha import combine_scores_by_state
from daily_research.baseline.backtest import backtest
from daily_research.baseline.cli_utils import (
    parse_csv_list,
    parse_horizon_weights,
    parse_int_tuple,
    parse_stock_list,
)
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import (
    load_daily_from_csv,
    load_daily_from_tq,
    load_industry_map_from_tq,
    load_style_map_from_tq,
    load_universe_from_tq,
    split_benchmark_from_universe,
)
from daily_research.baseline.features import compute_factors
from daily_research.baseline.ml_alpha import MLAplhaConfig, blend_scores, build_ml_feature_bundle, rolling_ml_scores_multi
from daily_research.baseline.portfolio import build_target_weights
from daily_research.baseline.regime import apply_market_regime_filter, compute_market_regime_state, resolve_regime_label_series
from daily_research.baseline.state_profiles import build_state_configs, validate_state_profile_selector


def parse_args():
    parser = argparse.ArgumentParser(description="Scan state-scoped ensemble weights on one shared advanced ML dataset.")
    parser.add_argument("--data-source", choices=["tq", "csv"], default="tq")
    parser.add_argument("--csv-folder", default=None)
    parser.add_argument("--stocks", default=None)
    parser.add_argument("--start-date", default="20220101")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--universe-scope", default="all_a")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--holding-count", type=int, default=5)
    parser.add_argument("--rebalance-freq", default="5d")
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
    parser.add_argument("--regime-trend-flat-band", type=float, default=0.01)
    parser.add_argument("--regime-vol-transition-band", type=float, default=0.10)
    parser.add_argument(
        "--regime-state-selector",
        choices=["quadrant", "market_state", "trend_bucket", "vol_bucket"],
        default="quadrant",
        help="状态标签来源。当前本脚本的候选 state weights 仍只支持 legacy quadrant。",
    )
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
    return parser.parse_args()


def _candidate_profiles() -> dict[str, dict[str, dict[str, float]]]:
    return {
        "base_global": {},
        "up_low_rule60": {
            "trend_up_low_vol": {"ml": 0.60, "none": 0.25, "v2": 0.15},
        },
        "up_low_rule55": {
            "trend_up_low_vol": {"ml": 0.55, "none": 0.25, "v2": 0.20},
        },
        "up_high_ml80": {
            "trend_up_high_vol": {"ml": 0.80, "none": 0.15, "v2": 0.05},
        },
        "up_split_6015_8015": {
            "trend_up_low_vol": {"ml": 0.60, "none": 0.25, "v2": 0.15},
            "trend_up_high_vol": {"ml": 0.80, "none": 0.15, "v2": 0.05},
        },
        "up_split_5520_8015": {
            "trend_up_low_vol": {"ml": 0.55, "none": 0.25, "v2": 0.20},
            "trend_up_high_vol": {"ml": 0.80, "none": 0.15, "v2": 0.05},
        },
    }


def _latest_scores(final_score: pd.DataFrame, target_weights: pd.DataFrame) -> pd.DataFrame:
    latest_dt = final_score.dropna(how="all").index.max()
    if pd.isna(latest_dt):
        return pd.DataFrame(columns=["stock", "final_score", "target_weight"])
    out = pd.DataFrame(
        {
            "date": latest_dt,
            "stock": final_score.columns,
            "final_score": final_score.loc[latest_dt].reindex(final_score.columns).values,
            "target_weight": target_weights.loc[latest_dt].reindex(final_score.columns).values,
        }
    )
    return out.sort_values(["final_score", "target_weight"], ascending=[False, False], na_position="last").reset_index(drop=True)


def main():
    args = parse_args()
    validate_state_profile_selector(args.enhanced_profile, args.regime_state_selector)
    if args.regime_state_selector != "quadrant":
        raise ValueError("scan_state_ensemble_weights currently compares legacy quadrant-scoped ensemble candidates only.")

    cfg = ResearchConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        universe_scope=args.universe_scope,
        benchmark=args.benchmark,
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
        regime_trend_flat_band=args.regime_trend_flat_band,
        regime_vol_transition_band=args.regime_vol_transition_band,
        regime_state_selector=args.regime_state_selector,
        regime_allowed_quadrants=parse_csv_list(args.regime_quadrants),
        enable_style_cap=not args.no_style_cap,
        max_style_weight=args.max_style_weight,
        enable_industry_cap=args.industry_cap,
        max_industry_weight=args.max_industry_weight,
    )
    stocks = parse_stock_list(args.stocks)
    if stocks:
        cfg.universe = stocks

    ml_cfg_base = MLAplhaConfig(
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
        ensemble_ml_weight=0.70,
        ensemble_none_weight=0.20,
        ensemble_v2_weight=0.10,
        train_regime_only=cfg.enable_market_regime_filter,
    )

    if args.data_source == "tq":
        if not cfg.universe and cfg.universe_scope == "all_a":
            print("[1/8] Loading all-A universe from TQ...")
            cfg.universe = load_universe_from_tq(cfg.universe_scope)
        elif not cfg.universe:
            raise ValueError("TQ mode without --stocks currently requires --universe-scope all_a.")
        print(f"[2/8] Loading daily bars from TQ. universe={len(cfg.universe)} benchmark={cfg.benchmark}")
        raw_df_dict = load_daily_from_tq(cfg.universe, cfg.start_date, cfg.end_date, benchmark=cfg.benchmark)
    else:
        if not args.csv_folder:
            raise ValueError("CSV mode requires --csv-folder.")
        print(f"[1/8] Loading CSV folder: {args.csv_folder}")
        raw_df_dict = load_daily_from_csv(args.csv_folder)

    print("[3/8] Building factor bundle and regime state...")
    df_dict, benchmark_close = split_benchmark_from_universe(raw_df_dict, cfg.benchmark)
    factor_bundle = compute_factors(df_dict)
    regime_state = compute_market_regime_state(benchmark_close, cfg)
    state_label_series = resolve_regime_label_series(regime_state, cfg.regime_state_selector)

    industry_map = None
    style_map = None
    if cfg.enable_industry_cap and args.data_source == "tq":
        print("[4/8] Loading industry map...")
        industry_map = load_industry_map_from_tq(list(df_dict["Close"].columns))
    if cfg.enable_style_cap and args.data_source == "tq":
        print("[5/8] Loading style map...")
        style_map = load_style_map_from_tq(list(df_dict["Close"].columns))

    print("[6/8] Computing none / enhanced scores and rolling ML score once...")
    score_none, _, filter_mask = combine_scores_by_state(
        factor_bundle,
        cfg,
        quadrant_series=state_label_series,
        state_configs=build_state_configs(cfg, "none"),
    )
    score_enhanced, _, _ = combine_scores_by_state(
        factor_bundle,
        cfg,
        quadrant_series=state_label_series,
        state_configs=build_state_configs(cfg, args.enhanced_profile),
    )
    feature_frames, market_features = build_ml_feature_bundle(factor_bundle, regime_state, score_none, score_enhanced)
    ml_score, training_log = rolling_ml_scores_multi(
        feature_frames=feature_frames,
        market_features=market_features,
        close=factor_bundle["raw_inputs"]["Close"],
        benchmark_close=benchmark_close,
        filter_mask=filter_mask,
        regime_state=regime_state,
        config=ml_cfg_base,
        state_label_series=state_label_series,
    )

    output_root = Path("daily_research/output") / (
        args.experiment_tag.strip() or f"state_ensemble_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    candidates = _candidate_profiles()
    print("[7/8] Running candidate backtests...")
    for label, state_weights in candidates.items():
        run_cfg = MLAplhaConfig(**ml_cfg_base.__dict__)
        run_cfg.state_ensemble_weights = state_weights
        final_score = blend_scores(ml_score, score_none, score_enhanced, run_cfg, quadrant_series=state_label_series)
        target_weights = build_target_weights(final_score, cfg, industry_map=industry_map, style_map=style_map)
        score_for_backtest = final_score.fillna(0.0)
        if cfg.enable_market_regime_filter:
            target_weights, score_for_backtest = apply_market_regime_filter(target_weights, score_for_backtest, regime_state)
        equity_df, action_df, metrics = backtest(
            close=factor_bundle["raw_inputs"]["Close"],
            benchmark_close=benchmark_close,
            target_weights=target_weights,
            target_scores=score_for_backtest,
            config=cfg,
            regime_on=regime_state["regime_on"],
        )
        run_dir = output_root / label
        run_dir.mkdir(parents=True, exist_ok=True)
        equity_export = equity_df.reset_index().rename(columns={equity_df.index.name or "index": "date"})
        equity_export.to_csv(run_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
        action_df.to_csv(run_dir / "actions.csv", index=False, encoding="utf-8-sig")
        regime_state.to_csv(run_dir / "regime_state.csv", encoding="utf-8-sig")
        training_log.to_csv(run_dir / "training_log.csv", index=False, encoding="utf-8-sig")
        _latest_scores(final_score, target_weights).to_csv(run_dir / "latest_scores.csv", index=False, encoding="utf-8-sig")
        metrics.update(
            {
                "framework": "advanced_ml_state_ensemble_scan",
                "candidate_label": label,
                "enhanced_profile": args.enhanced_profile,
                "regime_state_selector": cfg.regime_state_selector,
                "state_ensemble_weights": state_weights,
                "ml_horizon_weights": {str(k): float(v) for k, v in (ml_cfg_base.target_horizon_weights or {}).items()},
            }
        )
        (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(
            {
                "label": label,
                "state_ensemble_weights": json.dumps(state_weights, ensure_ascii=False, sort_keys=True),
                "total_return": metrics.get("total_return"),
                "annual_return": metrics.get("annual_return"),
                "sharpe": metrics.get("sharpe"),
                "max_drawdown": metrics.get("max_drawdown"),
                "excess_total_return": metrics.get("excess_total_return"),
                "excess_annual_return": metrics.get("excess_annual_return"),
                "excess_sharpe": metrics.get("excess_sharpe"),
                "excess_max_drawdown": metrics.get("excess_max_drawdown"),
                "avg_holding_count": metrics.get("avg_holding_count"),
                "avg_turnover": metrics.get("avg_turnover"),
            }
        )
        print(
            f"  - {label}: excess_total_return={metrics.get('excess_total_return', float('nan')):.2%}, "
            f"excess_sharpe={metrics.get('excess_sharpe', float('nan')):.3f}"
        )

    summary_df = pd.DataFrame(rows).sort_values(["excess_sharpe", "excess_total_return"], ascending=[False, False]).reset_index(drop=True)
    summary_df.to_csv(output_root / "state_ensemble_scan_summary.csv", index=False, encoding="utf-8-sig")
    print("[8/8] Done.")
    print(f"Output: {output_root}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
