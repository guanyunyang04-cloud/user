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
from daily_research.baseline.regime import apply_market_regime_filter, compute_market_regime_state
from daily_research.baseline.state_profiles import build_state_configs


def parse_args():
    parser = argparse.ArgumentParser(description="Compare advanced ML model families on one shared dataset.")
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
    parser.add_argument("--model-families", default="histgb,lgbm,etr")
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
    parser.add_argument("--ml-train-window-days", type=int, default=504)
    parser.add_argument("--ml-retrain-every-days", type=int, default=21)
    parser.add_argument("--ml-min-train-dates", type=int, default=120)
    parser.add_argument("--ml-max-samples-per-day", type=int, default=600)
    parser.add_argument("--ml-max-train-rows", type=int, default=200000)
    parser.add_argument("--ml-random-seed", type=int, default=7)
    parser.add_argument("--ensemble-ml-weight", type=float, default=0.70)
    parser.add_argument("--ensemble-none-weight", type=float, default=0.20)
    parser.add_argument("--ensemble-v2-weight", type=float, default=0.10)
    return parser.parse_args()


def _parse_stocks(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [stock.strip().upper() for stock in raw.split(",") if stock.strip()]


def _parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _parse_int_tuple(raw: str | None, fallback: int) -> tuple[int, ...]:
    if not raw:
        return (int(fallback),)
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    return values or (int(fallback),)


def _parse_horizon_weights(raw: str | None) -> dict[int, float]:
    if not raw:
        return {}
    out: dict[int, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        horizon_raw, weight_raw = item.split(":", 1)
        out[int(horizon_raw.strip())] = float(weight_raw.strip())
    return out


def _parse_model_families(raw: str | None) -> list[str]:
    if not raw:
        return ["histgb"]
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def main():
    args = parse_args()
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
        regime_allowed_quadrants=_parse_csv_list(args.regime_quadrants),
        enable_style_cap=not args.no_style_cap,
        max_style_weight=args.max_style_weight,
        enable_industry_cap=args.industry_cap,
        max_industry_weight=args.max_industry_weight,
    )
    stocks = _parse_stocks(args.stocks)
    if stocks:
        cfg.universe = stocks

    base_ml_cfg = MLAplhaConfig(
        target_horizon=args.ml_target_horizon,
        target_horizons=_parse_int_tuple(args.ml_target_horizons, args.ml_target_horizon),
        target_horizon_weights=_parse_horizon_weights(args.ml_horizon_weights),
        enhanced_profile=args.enhanced_profile,
        train_window_days=args.ml_train_window_days,
        retrain_every_days=args.ml_retrain_every_days,
        min_train_dates=args.ml_min_train_dates,
        max_samples_per_day=args.ml_max_samples_per_day,
        max_train_rows=args.ml_max_train_rows,
        random_seed=args.ml_random_seed,
        ensemble_ml_weight=args.ensemble_ml_weight,
        ensemble_none_weight=args.ensemble_none_weight,
        ensemble_v2_weight=args.ensemble_v2_weight,
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

    industry_map = None
    style_map = None
    if cfg.enable_industry_cap and args.data_source == "tq":
        print("[4/8] Loading industry map...")
        industry_map = load_industry_map_from_tq(list(df_dict["Close"].columns))
    if cfg.enable_style_cap and args.data_source == "tq":
        print("[5/8] Loading style map...")
        style_map = load_style_map_from_tq(list(df_dict["Close"].columns))

    print("[6/8] Computing shared none / enhanced rule layers...")
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
    feature_frames, market_features = build_ml_feature_bundle(factor_bundle, regime_state, score_none, score_enhanced)

    output_root = Path("daily_research/output") / (
        args.experiment_tag.strip() or f"advanced_ml_model_families_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    print("[7/8] Running model-family A/B...")
    for family in _parse_model_families(args.model_families):
        ml_cfg = MLAplhaConfig(**base_ml_cfg.__dict__)
        ml_cfg.model_family = family
        ml_score, training_log = rolling_ml_scores_multi(
            feature_frames=feature_frames,
            market_features=market_features,
            close=factor_bundle["raw_inputs"]["Close"],
            benchmark_close=benchmark_close,
            filter_mask=filter_mask,
            regime_state=regime_state,
            config=ml_cfg,
        )
        final_score = blend_scores(ml_score, score_none, score_enhanced, ml_cfg, quadrant_series=regime_state["quadrant"])
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
        run_dir = output_root / family
        run_dir.mkdir(parents=True, exist_ok=True)
        equity_export = equity_df.reset_index().rename(columns={equity_df.index.name or "index": "date"})
        equity_export.to_csv(run_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
        action_df.to_csv(run_dir / "actions.csv", index=False, encoding="utf-8-sig")
        regime_state.to_csv(run_dir / "regime_state.csv", encoding="utf-8-sig")
        training_log.to_csv(run_dir / "training_log.csv", index=False, encoding="utf-8-sig")
        metrics.update(
            {
                "framework": "advanced_ml_model_family_compare",
                "model_family": family,
                "enhanced_profile": args.enhanced_profile,
                "ml_horizon_weights": {str(k): float(v) for k, v in (ml_cfg.target_horizon_weights or {}).items()},
            }
        )
        (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(
            {
                "model_family": family,
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
            f"  - {family}: excess_total_return={metrics.get('excess_total_return', float('nan')):.2%}, "
            f"excess_sharpe={metrics.get('excess_sharpe', float('nan')):.3f}"
        )

    summary_df = pd.DataFrame(rows).sort_values(["excess_sharpe", "excess_total_return"], ascending=[False, False]).reset_index(drop=True)
    summary_df.to_csv(output_root / "model_family_compare_summary.csv", index=False, encoding="utf-8-sig")
    print("[8/8] Done.")
    print(f"Output: {output_root}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
