from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from datetime import datetime
from typing import Any

import numpy as np
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
from daily_research.execution.liquidity_universe import build_rolling_liquidity_membership


def parse_args():
    parser = argparse.ArgumentParser(description="Compare advanced ML model families on one shared dataset.")
    parser.add_argument("--data-source", choices=["tq", "csv"], default="tq")
    parser.add_argument("--csv-folder", default=None)
    parser.add_argument("--stocks", default=None)
    parser.add_argument(
        "--rolling-liquidity-pool",
        choices=["liquid300", "liquid500", "liquid800"],
        default="liquid500",
        help="Use the historical rolling high-liquidity pool under current execution settings.",
    )
    parser.add_argument("--pool-rebalance-days", type=int, default=21)
    parser.add_argument("--pool-adv-window", type=int, default=20)
    parser.add_argument("--start-date", default="20220101")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--universe-scope", default="all_a")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--holding-count", type=int, default=5)
    parser.add_argument("--rebalance-freq", default="1d")
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--enhanced-profile", default="up_low_breakout_v2")
    parser.add_argument("--model-families", default="histgb,lgbm,etr")
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
    parser.add_argument("--ensemble-ml-weight", type=float, default=0.70)
    parser.add_argument("--ensemble-none-weight", type=float, default=0.20)
    parser.add_argument("--ensemble-v2-weight", type=float, default=0.10)
    parser.add_argument(
        "--windows",
        default="recent_full:20250307:20260319,latest_weak:20250905:20260319",
        help="Comma-separated named windows in name:YYYYMMDD:YYYYMMDD format.",
    )
    parser.add_argument(
        "--skip-missing-families",
        action="store_true",
        help="Skip model families that cannot be run in the current environment instead of failing the whole comparison.",
    )
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


def _parse_named_windows(raw: str | None) -> list[tuple[str, str, str]]:
    if not raw:
        return []
    windows: list[tuple[str, str, str]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid window spec: {chunk}")
        name, start, end = parts
        windows.append((name.strip(), start.strip(), end.strip()))
    return windows


def _subset_df_dict_to_stocks(df_dict: dict[str, pd.DataFrame], stocks: list[str]) -> dict[str, pd.DataFrame]:
    keep = [stock for stock in stocks if stock in df_dict["Close"].columns]
    return {field: frame.reindex(columns=keep) for field, frame in df_dict.items()}


def _annualized_return(equity: pd.Series) -> float:
    daily_ret = equity.pct_change().dropna()
    if daily_ret.empty:
        return 0.0
    return float(equity.iloc[-1] ** (252 / len(daily_ret)) - 1.0)


def _annualized_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0


def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min()) if len(dd) else 0.0


def _slice_metrics(equity_df: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    window = equity_df.loc[(equity_df.index >= pd.Timestamp(start)) & (equity_df.index <= pd.Timestamp(end))].copy()
    if window.empty:
        return {
            "holdout_start": pd.Timestamp(start).strftime("%Y-%m-%d"),
            "holdout_end": pd.Timestamp(end).strftime("%Y-%m-%d"),
            "total_return": np.nan,
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "sharpe": np.nan,
            "benchmark_total_return": np.nan,
            "benchmark_annual_return": np.nan,
            "excess_total_return": np.nan,
            "excess_annual_return": np.nan,
            "excess_sharpe": np.nan,
            "excess_max_drawdown": np.nan,
            "max_drawdown": np.nan,
            "avg_holding_count": np.nan,
            "avg_turnover": np.nan,
            "hit_rate": np.nan,
            "regime_active_ratio": np.nan,
        }

    portfolio_equity = window["portfolio_equity"] / float(window["portfolio_equity"].iloc[0])
    benchmark_equity = window["benchmark_equity"] / float(window["benchmark_equity"].iloc[0])
    excess_equity = portfolio_equity / benchmark_equity.replace(0.0, np.nan)
    excess_equity = excess_equity.replace([np.inf, -np.inf], np.nan).ffill().dropna()

    portfolio_returns = window["portfolio_return"].dropna()
    excess_returns = window["excess_return"].dropna()
    portfolio_ann_ret = _annualized_return(portfolio_equity)
    benchmark_ann_ret = _annualized_return(benchmark_equity)
    excess_ann_ret = _annualized_return(excess_equity) if not excess_equity.empty else 0.0
    portfolio_ann_vol = _annualized_vol(portfolio_returns)
    excess_ann_vol = _annualized_vol(excess_returns)

    return {
        "holdout_start": pd.Timestamp(start).strftime("%Y-%m-%d"),
        "holdout_end": pd.Timestamp(end).strftime("%Y-%m-%d"),
        "total_return": float(portfolio_equity.iloc[-1] - 1.0),
        "annual_return": float(portfolio_ann_ret),
        "annual_vol": float(portfolio_ann_vol),
        "sharpe": float(portfolio_ann_ret / portfolio_ann_vol) if portfolio_ann_vol > 0 else 0.0,
        "benchmark_total_return": float(benchmark_equity.iloc[-1] - 1.0),
        "benchmark_annual_return": float(benchmark_ann_ret),
        "excess_total_return": float(excess_equity.iloc[-1] - 1.0) if not excess_equity.empty else 0.0,
        "excess_annual_return": float(excess_ann_ret),
        "excess_sharpe": float(excess_ann_ret / excess_ann_vol) if excess_ann_vol > 0 else 0.0,
        "excess_max_drawdown": float(_max_drawdown(excess_equity)) if not excess_equity.empty else 0.0,
        "max_drawdown": float(_max_drawdown(portfolio_equity)),
        "avg_holding_count": float(window["holding_count"].mean()),
        "avg_turnover": float(window["turnover"].mean()),
        "hit_rate": float((portfolio_returns > 0).mean()) if not portfolio_returns.empty else 0.0,
        "regime_active_ratio": float(window["regime_on"].mean()) if "regime_on" in window else np.nan,
    }


def main():
    args = parse_args()
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
    benchmark_open = raw_df_dict["Open"][cfg.benchmark].copy()
    df_dict, benchmark_close = split_benchmark_from_universe(raw_df_dict, cfg.benchmark)
    rolling_pool_artifact = None
    rolling_membership_mask = None
    if args.rolling_liquidity_pool:
        print(
            f"[4/8] Building rolling {args.rolling_liquidity_pool} pool "
            f"(rebalance={args.pool_rebalance_days}d, adv_window={args.pool_adv_window})..."
        )
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
            raise RuntimeError(f"Rolling {args.rolling_liquidity_pool} pool is empty for the requested window.")
        df_dict = _subset_df_dict_to_stocks(df_dict, rolling_union)
        rolling_membership_mask = rolling_membership_mask.reindex(
            index=df_dict["Close"].index,
            columns=df_dict["Close"].columns,
        ).fillna(False)
    factor_bundle = compute_factors(df_dict)
    regime_state = compute_market_regime_state(benchmark_close, cfg)

    industry_map = None
    style_map = None
    if cfg.enable_industry_cap and args.data_source == "tq":
        print("[5/8] Loading industry map...")
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
    if rolling_membership_mask is not None:
        filter_mask = filter_mask & rolling_membership_mask
        score_none = score_none.where(rolling_membership_mask)
        score_enhanced = score_enhanced.where(rolling_membership_mask)
    feature_frames, market_features = build_ml_feature_bundle(factor_bundle, regime_state, score_none, score_enhanced)
    windows = _parse_named_windows(args.windows)

    output_root = Path("daily_research/output") / (
        args.experiment_tag.strip() or f"advanced_ml_model_families_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    print("[7/8] Running model-family A/B...")
    for family in _parse_model_families(args.model_families):
        run_dir = output_root / family
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            ml_cfg = MLAplhaConfig(**base_ml_cfg.__dict__)
            ml_cfg.model_family = family
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
            equity_export = equity_df.reset_index().rename(columns={equity_df.index.name or "index": "date"})
            equity_export.to_csv(run_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
            action_df.to_csv(run_dir / "actions.csv", index=False, encoding="utf-8-sig")
            regime_state.to_csv(run_dir / "regime_state.csv", encoding="utf-8-sig")
            training_log.to_csv(run_dir / "training_log.csv", index=False, encoding="utf-8-sig")
            if rolling_pool_artifact is not None:
                rolling_pool_artifact.schedule_df.to_csv(
                    run_dir / "rolling_liquidity_schedule.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                rolling_pool_artifact.summary_df.to_csv(
                    run_dir / "rolling_liquidity_summary.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

            metrics.update(
                {
                    "framework": "advanced_ml_model_family_compare",
                    "model_family": family,
                    "enhanced_profile": args.enhanced_profile,
                    "execution_mode": cfg.execution_mode,
                    "ml_horizon_weights": {str(k): float(v) for k, v in (ml_cfg.target_horizon_weights or {}).items()},
                    "rolling_liquidity_pool": args.rolling_liquidity_pool or "",
                    "rolling_pool_rebalance_days": int(args.pool_rebalance_days),
                    "rolling_pool_adv_window": int(args.pool_adv_window),
                    "rolling_pool_union_size": 0 if rolling_pool_artifact is None else int(rolling_membership_mask.columns[rolling_membership_mask.any(axis=0)].size),
                    "rolling_pool_rebalance_count": 0 if rolling_pool_artifact is None else int(len(rolling_pool_artifact.schedule_df)),
                }
            )
            (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

            row = {
                "model_family": family,
                "status": "ok",
                "error_message": "",
                "full_excess_total_return": metrics.get("excess_total_return"),
                "full_excess_annual_return": metrics.get("excess_annual_return"),
                "full_excess_sharpe": metrics.get("excess_sharpe"),
                "full_excess_max_drawdown": metrics.get("excess_max_drawdown"),
                "full_avg_turnover": metrics.get("avg_turnover"),
                "full_regime_active_ratio": metrics.get("regime_active_ratio"),
            }
            for window_name, start, end in windows:
                window_metrics = _slice_metrics(equity_df, start, end)
                (run_dir / f"metrics_{window_name}.json").write_text(
                    json.dumps(window_metrics, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                prefix = f"{window_name}_"
                row[f"{prefix}excess_total_return"] = window_metrics.get("excess_total_return")
                row[f"{prefix}excess_sharpe"] = window_metrics.get("excess_sharpe")
                row[f"{prefix}excess_max_drawdown"] = window_metrics.get("excess_max_drawdown")
                row[f"{prefix}avg_turnover"] = window_metrics.get("avg_turnover")
            rows.append(row)
            latest_key = windows[-1][0] if windows else "latest"
            print(
                f"  - {family}: full_excess_sharpe={metrics.get('excess_sharpe', float('nan')):.3f} | "
                f"{latest_key}_excess_sharpe={row.get(f'{latest_key}_excess_sharpe', float('nan')):.3f} | "
                f"{latest_key}_excess_return={row.get(f'{latest_key}_excess_total_return', float('nan')):.2%}"
            )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            (run_dir / "error.txt").write_text(error_message, encoding="utf-8")
            rows.append({"model_family": family, "status": "error", "error_message": error_message})
            print(f"  - {family}: ERROR {error_message}")
            if not args.skip_missing_families:
                raise

    summary_df = pd.DataFrame(rows)
    sort_keys = []
    if windows:
        latest_prefix = f"{windows[-1][0]}_"
        sort_keys.extend([f"{latest_prefix}excess_sharpe", f"{latest_prefix}excess_total_return"])
    sort_keys.extend(["full_excess_sharpe", "full_excess_total_return"])
    existing_sort_keys = [key for key in sort_keys if key in summary_df.columns]
    if existing_sort_keys:
        summary_df = summary_df.sort_values(existing_sort_keys, ascending=[False] * len(existing_sort_keys), na_position="last")
    summary_df = summary_df.reset_index(drop=True)
    summary_df.to_csv(output_root / "model_family_compare_summary.csv", index=False, encoding="utf-8-sig")
    print("[8/8] Done.")
    print(f"Output: {output_root}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
