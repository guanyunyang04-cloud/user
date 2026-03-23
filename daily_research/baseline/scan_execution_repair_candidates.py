from __future__ import annotations

import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

from daily_research.baseline.advanced_ml_runtime import (
    build_prepared_bundle_with_cache,
    history_window_to_dict,
    load_raw_data_with_cache,
    resolve_history_window,
)
from daily_research.baseline.backtest import backtest
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import (
    load_industry_map_from_tq,
    load_style_map_from_tq,
    load_universe_from_tq,
    split_benchmark_from_universe,
)
from daily_research.baseline.ml_alpha import (
    MLAplhaConfig,
    blend_scores,
    build_ml_feature_bundle,
    combine_per_horizon_ml_scores,
    rolling_ml_scores_multi_detail,
)
from daily_research.baseline.portfolio import build_target_weights
from daily_research.baseline.regime import apply_market_regime_filter
from daily_research.execution.liquidity_universe import build_rolling_liquidity_membership


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan formal execution-repair candidates on shared advanced ML data."
    )
    parser.add_argument("--data-source", choices=["tq", "csv"], default="tq")
    parser.add_argument("--csv-folder", default=None)
    parser.add_argument("--stocks", default=None)
    parser.add_argument("--stocks-file", default=None, help="Path to txt/csv file containing stock codes.")
    parser.add_argument(
        "--rolling-liquidity-pool",
        choices=["liquid300", "liquid500", "liquid800"],
        default="liquid500",
    )
    parser.add_argument("--pool-rebalance-days", type=int, default=21)
    parser.add_argument("--pool-adv-window", type=int, default=20)
    parser.add_argument("--start-date", default="20220101")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--universe-scope", default="all_a")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--holding-count", type=int, default=5)
    parser.add_argument("--rebalance-freq", default="5d")
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument(
        "--candidate-set",
        choices=[
            "round1",
            "round2_weights",
            "pair_best",
            "ma50_boundary_risk",
            "ma50_state_horizon_round1",
            "ma50_state_ensemble_round1",
            "ma50_state_ensemble_round2_low_only",
            "ma50_regime_vol_round1",
        ],
        default="round1",
    )
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
    parser.add_argument("--ml-train-window-days", type=int, default=504)
    parser.add_argument("--ml-retrain-every-days", type=int, default=21)
    parser.add_argument("--ml-min-train-dates", type=int, default=120)
    parser.add_argument("--ml-max-samples-per-day", type=int, default=600)
    parser.add_argument("--ml-max-train-rows", type=int, default=200000)
    parser.add_argument("--ml-random-seed", type=int, default=7)
    parser.add_argument("--ml-model-family", choices=["histgb", "etr", "lgbm"], default="histgb")
    parser.add_argument(
        "--ml-state-horizon-profiles",
        default="",
        help="Optional base state horizon weights, e.g. trend_up_high_vol=5:0.3,10:0.4,20:0.3;trend_up_low_vol=5:0.15,10:0.35,20:0.5",
    )
    parser.add_argument("--ensemble-ml-weight", type=float, default=0.70)
    parser.add_argument("--ensemble-none-weight", type=float, default=0.20)
    parser.add_argument("--ensemble-v2-weight", type=float, default=0.10)
    parser.add_argument(
        "--ensemble-state-weights",
        default="",
        help="Optional base state ensemble weights, e.g. trend_up_high_vol=ml:0.8,none:0.15,v2:0.05",
    )
    parser.add_argument(
        "--windows",
        default="recent_full:20250307:20260319,latest_weak:20250905:20260319",
        help="Comma-separated named windows in name:YYYYMMDD:YYYYMMDD format.",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--auto-trim-history",
        action="store_true",
        help="Trim history automatically to the minimum window needed for speed. Off by default for formal scans.",
    )
    return parser.parse_args()


def _parse_stocks(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [stock.strip().upper() for stock in raw.split(",") if stock.strip()]


def _load_stocks_from_file(path: str | None) -> list[str]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"stocks file not found: {path}")
    text = file_path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    tokens: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line:
            tokens.extend(item.strip() for item in line.split(",") if item.strip())
        else:
            tokens.append(line)
    return [token.upper() for token in tokens]


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


def _parse_state_horizon_profiles(raw: str | None) -> dict[str, dict[int, float]]:
    if not raw:
        return {}
    out: dict[str, dict[int, float]] = {}
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        state_raw, weights_raw = chunk.split("=", 1)
        out[state_raw.strip()] = _parse_horizon_weights(weights_raw)
    return out


def _parse_state_ensemble_weights(raw: str | None) -> dict[str, dict[str, float]]:
    if not raw:
        return {}
    out: dict[str, dict[str, float]] = {}
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        state_raw, weights_raw = chunk.split("=", 1)
        weights: dict[str, float] = {}
        for item in weights_raw.split(","):
            item = item.strip()
            if not item:
                continue
            name_raw, value_raw = item.split(":", 1)
            weights[name_raw.strip().lower()] = float(value_raw.strip())
        out[state_raw.strip()] = weights
    return out


def _merge_nested_dict(base: dict[str, dict[Any, Any]] | None, override: dict[str, dict[Any, Any]] | None) -> dict[str, dict[Any, Any]]:
    merged: dict[str, dict[Any, Any]] = {
        str(key): dict(value)
        for key, value in (base or {}).items()
    }
    for key, value in (override or {}).items():
        merged[str(key)] = dict(value)
    return merged


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


def _subset_raw_df_dict_to_stocks(
    raw_df_dict: dict[str, pd.DataFrame],
    benchmark: str,
    stocks: list[str],
) -> dict[str, pd.DataFrame]:
    keep = [benchmark] + [stock for stock in stocks if stock != benchmark]
    keep_set = set(keep)
    out: dict[str, pd.DataFrame] = {}
    for field, frame in raw_df_dict.items():
        cols = [col for col in frame.columns if col in keep_set]
        out[field] = frame.reindex(columns=cols)
    return out


def _candidate_profiles(candidate_set: str) -> list[dict[str, Any]]:
    if candidate_set == "round1":
        return [
            {"label": "baseline"},
            {
                "label": "up_low_ml55_none25_v220",
                "state_ensemble_weights": {
                    "trend_up_low_vol": {"ml": 0.55, "none": 0.25, "v2": 0.20},
                },
            },
            {
                "label": "up_low_ml50_none20_v230",
                "state_ensemble_weights": {
                    "trend_up_low_vol": {"ml": 0.50, "none": 0.20, "v2": 0.30},
                },
            },
            {
                "label": "turnover1_hold3",
                "turnover_limit": 1.00,
                "min_hold_days": 3,
            },
            {
                "label": "up_low_ml55_none25_v220_turnover1_hold3",
                "state_ensemble_weights": {
                    "trend_up_low_vol": {"ml": 0.55, "none": 0.25, "v2": 0.20},
                },
                "turnover_limit": 1.00,
                "min_hold_days": 3,
            },
            {
                "label": "up_low_ml50_none20_v230_turnover1_hold3",
                "state_ensemble_weights": {
                    "trend_up_low_vol": {"ml": 0.50, "none": 0.20, "v2": 0.30},
                },
                "turnover_limit": 1.00,
                "min_hold_days": 3,
            },
        ]

    if candidate_set == "round2_weights":
        return [
            {"label": "baseline"},
            {
                "label": "up_low_ml60_none25_v215",
                "state_ensemble_weights": {
                    "trend_up_low_vol": {"ml": 0.60, "none": 0.25, "v2": 0.15},
                },
            },
            {
                "label": "up_low_ml58_none24_v218",
                "state_ensemble_weights": {
                    "trend_up_low_vol": {"ml": 0.58, "none": 0.24, "v2": 0.18},
                },
            },
            {
                "label": "up_low_ml57_none25_v218",
                "state_ensemble_weights": {
                    "trend_up_low_vol": {"ml": 0.57, "none": 0.25, "v2": 0.18},
                },
            },
            {
                "label": "up_low_ml56_none24_v220",
                "state_ensemble_weights": {
                    "trend_up_low_vol": {"ml": 0.56, "none": 0.24, "v2": 0.20},
                },
            },
            {
                "label": "up_low_ml55_none25_v220",
                "state_ensemble_weights": {
                    "trend_up_low_vol": {"ml": 0.55, "none": 0.25, "v2": 0.20},
                },
            },
        ]

    if candidate_set == "pair_best":
        return [
            {"label": "baseline"},
            {
                "label": "up_low_ml55_none25_v220",
                "state_ensemble_weights": {
                    "trend_up_low_vol": {"ml": 0.55, "none": 0.25, "v2": 0.20},
                },
            },
        ]

    if candidate_set == "ma50_boundary_risk":
        return [
            {"label": "baseline"},
            {
                "label": "stop8",
                "stop_loss": -0.08,
            },
            {
                "label": "take20",
                "take_profit": 0.20,
            },
            {
                "label": "stop8_take20",
                "stop_loss": -0.08,
                "take_profit": 0.20,
            },
        ]

    if candidate_set == "ma50_state_horizon_round1":
        return [
            {"label": "baseline"},
            {
                "label": "up_high_h304030",
                "state_horizon_weights": {
                    "trend_up_high_vol": {5: 0.30, 10: 0.40, 20: 0.30},
                },
            },
            {
                "label": "up_high_h354025",
                "state_horizon_weights": {
                    "trend_up_high_vol": {5: 0.35, 10: 0.40, 20: 0.25},
                },
            },
            {
                "label": "up_high_h403525",
                "state_horizon_weights": {
                    "trend_up_high_vol": {5: 0.40, 10: 0.35, 20: 0.25},
                },
            },
            {
                "label": "up_split_low153550_high304030",
                "state_horizon_weights": {
                    "trend_up_low_vol": {5: 0.15, 10: 0.35, 20: 0.50},
                    "trend_up_high_vol": {5: 0.30, 10: 0.40, 20: 0.30},
                },
            },
            {
                "label": "up_split_low153550_high354025",
                "state_horizon_weights": {
                    "trend_up_low_vol": {5: 0.15, 10: 0.35, 20: 0.50},
                    "trend_up_high_vol": {5: 0.35, 10: 0.40, 20: 0.25},
                },
            },
            {
                "label": "up_split_low103060_high354025",
                "state_horizon_weights": {
                    "trend_up_low_vol": {5: 0.10, 10: 0.30, 20: 0.60},
                    "trend_up_high_vol": {5: 0.35, 10: 0.40, 20: 0.25},
                },
            },
        ]

    if candidate_set == "ma50_state_ensemble_round1":
        return [
            {"label": "baseline"},
            {
                "label": "up_high_ml80_none15_v205",
                "state_ensemble_weights": {
                    "trend_up_high_vol": {"ml": 0.80, "none": 0.15, "v2": 0.05},
                },
            },
            {
                "label": "up_high_ml75_none20_v205",
                "state_ensemble_weights": {
                    "trend_up_high_vol": {"ml": 0.75, "none": 0.20, "v2": 0.05},
                },
            },
            {
                "label": "up_high_ml70_none20_v210",
                "state_ensemble_weights": {
                    "trend_up_high_vol": {"ml": 0.70, "none": 0.20, "v2": 0.10},
                },
            },
            {
                "label": "up_split_low65_none20_v215_high80_none15_v205",
                "state_ensemble_weights": {
                    "trend_up_low_vol": {"ml": 0.65, "none": 0.20, "v2": 0.15},
                    "trend_up_high_vol": {"ml": 0.80, "none": 0.15, "v2": 0.05},
                },
            },
            {
                "label": "up_split_low60_none25_v215_high80_none15_v205",
                "state_ensemble_weights": {
                    "trend_up_low_vol": {"ml": 0.60, "none": 0.25, "v2": 0.15},
                    "trend_up_high_vol": {"ml": 0.80, "none": 0.15, "v2": 0.05},
                },
            },
        ]

    if candidate_set == "ma50_state_ensemble_round2_low_only":
        return [
            {"label": "baseline"},
            {
                "label": "up_low_ml62_none23_v215",
                "state_ensemble_weights": {
                    "trend_up_low_vol": {"ml": 0.62, "none": 0.23, "v2": 0.15},
                },
            },
            {
                "label": "up_low_ml61_none24_v215",
                "state_ensemble_weights": {
                    "trend_up_low_vol": {"ml": 0.61, "none": 0.24, "v2": 0.15},
                },
            },
            {
                "label": "up_low_ml60_none25_v215",
                "state_ensemble_weights": {
                    "trend_up_low_vol": {"ml": 0.60, "none": 0.25, "v2": 0.15},
                },
            },
            {
                "label": "up_low_ml60_none24_v216",
                "state_ensemble_weights": {
                    "trend_up_low_vol": {"ml": 0.60, "none": 0.24, "v2": 0.16},
                },
            },
            {
                "label": "up_low_ml59_none25_v216",
                "state_ensemble_weights": {
                    "trend_up_low_vol": {"ml": 0.59, "none": 0.25, "v2": 0.16},
                },
            },
            {
                "label": "up_low_ml58_none25_v217",
                "state_ensemble_weights": {
                    "trend_up_low_vol": {"ml": 0.58, "none": 0.25, "v2": 0.17},
                },
            },
            {
                "label": "up_low_ml57_none25_v218",
                "state_ensemble_weights": {
                    "trend_up_low_vol": {"ml": 0.57, "none": 0.25, "v2": 0.18},
                },
            },
        ]

    if candidate_set == "ma50_regime_vol_round1":
        return [
            {"label": "vol030", "regime_max_annual_vol": 0.30},
            {"label": "vol031", "regime_max_annual_vol": 0.31},
            {"label": "baseline", "regime_max_annual_vol": 0.32},
            {"label": "vol033", "regime_max_annual_vol": 0.33},
            {"label": "vol034", "regime_max_annual_vol": 0.34},
        ]

    raise ValueError(f"Unsupported candidate_set: {candidate_set}")


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


def _slice_metrics(equity_df: pd.DataFrame, start: str, end: str, include_window_keys: bool = True) -> dict[str, Any]:
    window = equity_df.loc[(equity_df.index >= pd.Timestamp(start)) & (equity_df.index <= pd.Timestamp(end))].copy()
    if window.empty:
        metrics = {
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
    else:
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

        metrics = {
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
    if include_window_keys:
        metrics["holdout_start"] = pd.Timestamp(start).strftime("%Y-%m-%d")
        metrics["holdout_end"] = pd.Timestamp(end).strftime("%Y-%m-%d")
    return metrics


def _build_latest_scores(
    final_score: pd.DataFrame,
    target_weights: pd.DataFrame,
    score_none: pd.DataFrame,
    score_v2: pd.DataFrame,
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
            "score_none": score_none.loc[latest_dt].reindex(score_none.columns).values,
            "score_enhanced": score_v2.loc[latest_dt].reindex(score_v2.columns).values,
            "ml_score": ml_score.loc[latest_dt].reindex(ml_score.columns).values,
        }
    )
    return out.sort_values(["final_score", "target_weight"], ascending=[False, False], na_position="last").reset_index(drop=True)


def _clone_research_config(cfg: ResearchConfig) -> ResearchConfig:
    return replace(cfg)


def _clone_ml_config(ml_cfg: MLAplhaConfig) -> MLAplhaConfig:
    return MLAplhaConfig(**asdict(ml_cfg))


def _candidate_requires_prepared_recompute(candidate: dict[str, Any]) -> bool:
    return any(
        key in candidate
        for key in (
            "regime_ma_window",
            "regime_vol_window",
            "regime_max_annual_vol",
        )
    )


def main():
    args = parse_args()
    candidate_profiles = _candidate_profiles(args.candidate_set)
    full_recompute_mode = any(_candidate_requires_prepared_recompute(candidate) for candidate in candidate_profiles)

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
    file_stocks = _load_stocks_from_file(args.stocks_file)
    if stocks or file_stocks:
        stocks = list(dict.fromkeys(stocks + file_stocks))
    if stocks:
        cfg.universe = stocks

    ml_cfg = MLAplhaConfig(
        target_horizon=args.ml_target_horizon,
        target_horizons=_parse_int_tuple(args.ml_target_horizons, args.ml_target_horizon),
        target_horizon_weights=_parse_horizon_weights(args.ml_horizon_weights),
        state_horizon_weights=_parse_state_horizon_profiles(args.ml_state_horizon_profiles),
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
        state_ensemble_weights=_parse_state_ensemble_weights(args.ensemble_state_weights),
        train_regime_only=cfg.enable_market_regime_filter,
        execution_mode=cfg.execution_mode,
    )

    if args.data_source == "tq":
        if args.rolling_liquidity_pool and not cfg.universe:
            cfg.universe = load_universe_from_tq(cfg.universe_scope)
        elif not cfg.universe and cfg.universe_scope == "all_a":
            cfg.universe = load_universe_from_tq(cfg.universe_scope)
        elif not cfg.universe:
            raise ValueError("TQ mode without explicit stocks currently requires all_a universe scope.")
    elif not args.csv_folder:
        raise ValueError("CSV mode requires --csv-folder.")

    history_window = resolve_history_window(
        cfg=cfg,
        ml_cfg=ml_cfg,
        requested_start_date=args.start_date,
        end_date=args.end_date,
        mode="train",
        auto_trim_history=args.auto_trim_history,
    )
    print(
        f"[1/7] history window: {history_window.effective_start_date} -> "
        f"{history_window.end_date} | required_trading_days={history_window.required_trading_days}"
    )

    raw_df_dict, raw_cache_meta = load_raw_data_with_cache(
        data_source=args.data_source,
        csv_folder=args.csv_folder,
        universe=cfg.universe,
        benchmark=cfg.benchmark,
        history_window=history_window,
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
    )
    print(
        f"[2/7] raw cache: {'hit' if raw_cache_meta['cache_hit'] else 'build'} | "
        f"{raw_cache_meta['cache_path']}"
    )

    rolling_pool_artifact = None
    rolling_membership_mask = None
    prepared_raw_df_dict = raw_df_dict
    if args.rolling_liquidity_pool:
        print(f"[3/7] building rolling {args.rolling_liquidity_pool} membership...")
        raw_universe_df_dict, _ = split_benchmark_from_universe(raw_df_dict, cfg.benchmark)
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
    else:
        print("[3/7] using fixed universe without rolling liquidity pool...")

    prepared_raw_cache_key = (
        raw_cache_meta["cache_key"]
        if rolling_pool_artifact is None
        else (
            f"{raw_cache_meta['cache_key']}|{args.rolling_liquidity_pool}|"
            f"{args.pool_rebalance_days}|{args.pool_adv_window}"
        )
    )

    industry_map = None
    style_map = None
    candidate_columns = [col for col in prepared_raw_df_dict["Close"].columns if col != cfg.benchmark]
    if cfg.enable_industry_cap and args.data_source == "tq":
        industry_map = load_industry_map_from_tq(candidate_columns)
    if cfg.enable_style_cap and args.data_source == "tq":
        style_map = load_style_map_from_tq(candidate_columns)

    output_root = Path("daily_research/output") / (
        args.experiment_tag.strip() or f"advanced_ml_execution_repair_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    shared_prepared_cache_meta = None
    shared_training_log = pd.DataFrame()
    shared_per_horizon_scores: dict[int, pd.DataFrame] = {}
    shared_bundle: dict[str, Any] | None = None
    shared_membership_mask: pd.DataFrame | None = None

    if full_recompute_mode:
        print("[4/7] full recompute mode enabled for candidate-specific regime settings...")
    else:
        prepared_bundle, shared_prepared_cache_meta = build_prepared_bundle_with_cache(
            raw_df_dict=prepared_raw_df_dict,
            raw_cache_key=prepared_raw_cache_key,
            cfg=cfg,
            enhanced_profile=args.enhanced_profile,
            use_cache=not args.no_cache,
            refresh_cache=args.refresh_cache,
        )
        print(
            f"[4/7] factor cache: {'hit' if shared_prepared_cache_meta['cache_hit'] else 'build'} | "
            f"{shared_prepared_cache_meta['cache_path']}"
        )

        shared_bundle = prepared_bundle
        shared_membership_mask = None
        if rolling_membership_mask is not None:
            shared_membership_mask = rolling_membership_mask.reindex(
                index=prepared_bundle["df_dict"]["Close"].index,
                columns=prepared_bundle["df_dict"]["Close"].columns,
            ).fillna(False)
            prepared_bundle["filter_mask"] = prepared_bundle["filter_mask"] & shared_membership_mask
            prepared_bundle["score_none"] = prepared_bundle["score_none"].where(shared_membership_mask)
            prepared_bundle["score_v2"] = prepared_bundle["score_v2"].where(shared_membership_mask)
            feature_frames, market_features = build_ml_feature_bundle(
                prepared_bundle["factor_bundle"],
                prepared_bundle["regime_state"],
                prepared_bundle["score_none"],
                prepared_bundle["score_v2"],
            )
            prepared_bundle["feature_frames"] = feature_frames
            prepared_bundle["market_features"] = market_features

        print("[5/7] computing shared rolling ML score once...")
        _, shared_training_log, shared_per_horizon_scores = rolling_ml_scores_multi_detail(
            feature_frames=prepared_bundle["feature_frames"],
            market_features=prepared_bundle["market_features"],
            close=prepared_bundle["factor_bundle"]["raw_inputs"]["Close"],
            benchmark_close=prepared_bundle["benchmark_close"],
            open_df=prepared_bundle["factor_bundle"]["raw_inputs"]["Open"],
            benchmark_open=prepared_bundle["benchmark_open"],
            filter_mask=prepared_bundle["filter_mask"],
            regime_state=prepared_bundle["regime_state"],
            config=ml_cfg,
        )
        shared_training_log.to_csv(output_root / "shared_training_log.csv", index=False, encoding="utf-8-sig")

    scan_meta = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "history_window": history_window_to_dict(history_window),
        "raw_cache": raw_cache_meta,
        "shared_prepared_cache": shared_prepared_cache_meta,
        "full_recompute_mode": full_recompute_mode,
        "rolling_liquidity_pool": args.rolling_liquidity_pool or "",
        "rolling_pool_rebalance_days": int(args.pool_rebalance_days),
        "rolling_pool_adv_window": int(args.pool_adv_window),
        "rolling_pool_union_size": 0
        if rolling_membership_mask is None
        else int(rolling_membership_mask.columns[rolling_membership_mask.any(axis=0)].size),
        "rolling_pool_rebalance_count": 0
        if rolling_pool_artifact is None
        else int(len(rolling_pool_artifact.schedule_df)),
        "base_config": {
            "research": asdict(cfg),
            "ml": asdict(ml_cfg),
        },
        "candidate_set": args.candidate_set,
        "windows": [
            {"name": name, "start": start, "end": end}
            for name, start, end in _parse_named_windows(args.windows)
        ],
        "candidates": candidate_profiles,
    }
    (output_root / "scan_config.json").write_text(
        json.dumps(scan_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    windows = _parse_named_windows(args.windows)
    summary_rows: list[dict[str, Any]] = []

    print("[6/7] running candidate backtests...")
    for candidate in candidate_profiles:
        label = str(candidate["label"])
        run_cfg = _clone_research_config(cfg)
        run_ml_cfg = _clone_ml_config(ml_cfg)
        run_ml_cfg.state_horizon_weights = _merge_nested_dict(
            ml_cfg.state_horizon_weights,
            candidate.get("state_horizon_weights"),
        )
        run_ml_cfg.state_ensemble_weights = _merge_nested_dict(
            ml_cfg.state_ensemble_weights,
            candidate.get("state_ensemble_weights"),
        )
        if "turnover_limit" in candidate:
            run_cfg.turnover_limit = float(candidate["turnover_limit"])
        if "min_hold_days" in candidate:
            run_cfg.min_hold_days = int(candidate["min_hold_days"])
        if "regime_ma_window" in candidate:
            run_cfg.regime_ma_window = int(candidate["regime_ma_window"])
        if "regime_vol_window" in candidate:
            run_cfg.regime_vol_window = int(candidate["regime_vol_window"])
        if "regime_max_annual_vol" in candidate:
            run_cfg.regime_max_annual_vol = float(candidate["regime_max_annual_vol"])
        if "stop_loss" in candidate:
            run_cfg.stop_loss = float(candidate["stop_loss"])
        if "take_profit" in candidate:
            run_cfg.take_profit = float(candidate["take_profit"])

        run_dir = output_root / label
        run_dir.mkdir(parents=True, exist_ok=True)

        prepared_cache_meta = shared_prepared_cache_meta
        current_membership_mask = shared_membership_mask

        if full_recompute_mode:
            prepared_bundle, prepared_cache_meta = build_prepared_bundle_with_cache(
                raw_df_dict=prepared_raw_df_dict,
                raw_cache_key=prepared_raw_cache_key,
                cfg=run_cfg,
                enhanced_profile=args.enhanced_profile,
                use_cache=not args.no_cache,
                refresh_cache=args.refresh_cache,
            )
            print(
                f"  > {label}: factor cache {'hit' if prepared_cache_meta['cache_hit'] else 'build'} | "
                f"regime_max_annual_vol={run_cfg.regime_max_annual_vol:.2f}"
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

            candidate_ml_score, training_log, _ = rolling_ml_scores_multi_detail(
                feature_frames=prepared_bundle["feature_frames"],
                market_features=prepared_bundle["market_features"],
                close=prepared_bundle["factor_bundle"]["raw_inputs"]["Close"],
                benchmark_close=prepared_bundle["benchmark_close"],
                open_df=prepared_bundle["factor_bundle"]["raw_inputs"]["Open"],
                benchmark_open=prepared_bundle["benchmark_open"],
                filter_mask=prepared_bundle["filter_mask"],
                regime_state=prepared_bundle["regime_state"],
                config=run_ml_cfg,
            )
            training_log.to_csv(run_dir / "training_log.csv", index=False, encoding="utf-8-sig")
        else:
            prepared_bundle = shared_bundle
            if prepared_bundle is None:
                raise RuntimeError("Shared prepared bundle is missing in shared scan mode.")
            candidate_ml_score = combine_per_horizon_ml_scores(
                per_horizon_scores=shared_per_horizon_scores,
                close=prepared_bundle["factor_bundle"]["raw_inputs"]["Close"],
                regime_state=prepared_bundle["regime_state"],
                config=run_ml_cfg,
            )

        final_score = blend_scores(
            candidate_ml_score,
            prepared_bundle["score_none"],
            prepared_bundle["score_v2"],
            run_ml_cfg,
            quadrant_series=prepared_bundle["regime_state"]["quadrant"],
        )
        if current_membership_mask is not None:
            final_score = final_score.where(current_membership_mask)

        target_weights = build_target_weights(
            final_score,
            run_cfg,
            industry_map=industry_map,
            style_map=style_map,
        )
        score_for_backtest = final_score.fillna(0.0)
        if run_cfg.enable_market_regime_filter:
            target_weights, score_for_backtest = apply_market_regime_filter(
                target_weights,
                score_for_backtest,
                prepared_bundle["regime_state"],
            )
        equity_df, action_df, metrics = backtest(
            close=prepared_bundle["factor_bundle"]["raw_inputs"]["Close"],
            benchmark_close=prepared_bundle["benchmark_close"],
            open_df=prepared_bundle["factor_bundle"]["raw_inputs"]["Open"],
            benchmark_open=prepared_bundle["benchmark_open"],
            target_weights=target_weights,
            target_scores=score_for_backtest,
            config=run_cfg,
            regime_on=prepared_bundle["regime_state"]["regime_on"],
        )

        metrics.update(
            {
                "framework": "advanced_ml_execution_repair_scan",
                "candidate_label": label,
                "benchmark": cfg.benchmark,
                "execution_mode": cfg.execution_mode,
                "regime_ma_window": int(run_cfg.regime_ma_window),
                "regime_vol_window": int(run_cfg.regime_vol_window),
                "regime_max_annual_vol": float(run_cfg.regime_max_annual_vol),
                "regime_allowed_quadrants": list(run_cfg.regime_allowed_quadrants),
                "rolling_liquidity_pool": args.rolling_liquidity_pool or "",
                "rolling_pool_rebalance_days": int(args.pool_rebalance_days),
                "rolling_pool_adv_window": int(args.pool_adv_window),
                "rolling_pool_union_size": 0
                if rolling_membership_mask is None
                else int(rolling_membership_mask.columns[rolling_membership_mask.any(axis=0)].size),
                "rolling_pool_rebalance_count": 0
                if rolling_pool_artifact is None
                else int(len(rolling_pool_artifact.schedule_df)),
                "prepared_cache_key": prepared_cache_meta["cache_key"] if prepared_cache_meta else "",
                "prepared_cache_path": prepared_cache_meta["cache_path"] if prepared_cache_meta else "",
                "prepared_cache_hit": bool(prepared_cache_meta["cache_hit"]) if prepared_cache_meta else False,
                "state_horizon_weights": run_ml_cfg.state_horizon_weights or {},
                "state_ensemble_weights": run_ml_cfg.state_ensemble_weights or {},
                "turnover_limit": float(run_cfg.turnover_limit),
                "min_hold_days": int(run_cfg.min_hold_days),
                "stop_loss": float(run_cfg.stop_loss),
                "take_profit": float(run_cfg.take_profit),
            }
        )

        equity_df.to_csv(run_dir / "equity_curve.csv", encoding="utf-8-sig")
        action_df.to_csv(run_dir / "actions.csv", index=False, encoding="utf-8-sig")
        prepared_bundle["regime_state"].to_csv(run_dir / "regime_state.csv", encoding="utf-8-sig")
        target_weights.to_csv(run_dir / "target_weights.csv", encoding="utf-8-sig")
        _build_latest_scores(
            final_score,
            target_weights,
            prepared_bundle["score_none"],
            prepared_bundle["score_v2"],
            candidate_ml_score,
        ).to_csv(
            run_dir / "latest_scores.csv",
            index=False,
            encoding="utf-8-sig",
        )
        (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        row = {
            "label": label,
            "full_excess_total_return": metrics.get("excess_total_return"),
            "full_excess_sharpe": metrics.get("excess_sharpe"),
            "full_excess_max_drawdown": metrics.get("excess_max_drawdown"),
            "full_avg_turnover": metrics.get("avg_turnover"),
            "regime_ma_window": run_cfg.regime_ma_window,
            "regime_vol_window": run_cfg.regime_vol_window,
            "regime_max_annual_vol": run_cfg.regime_max_annual_vol,
            "state_horizon_weights": json.dumps(run_ml_cfg.state_horizon_weights or {}, ensure_ascii=False, sort_keys=True),
            "state_ensemble_weights": json.dumps(run_ml_cfg.state_ensemble_weights or {}, ensure_ascii=False, sort_keys=True),
            "turnover_limit": run_cfg.turnover_limit,
            "min_hold_days": run_cfg.min_hold_days,
            "stop_loss": run_cfg.stop_loss,
            "take_profit": run_cfg.take_profit,
        }
        for window_name, start, end in windows:
            window_metrics = _slice_metrics(equity_df, start, end)
            metrics_name = f"metrics_{window_name}.json"
            (run_dir / metrics_name).write_text(
                json.dumps(window_metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            prefix = f"{window_name}_"
            row[f"{prefix}excess_total_return"] = window_metrics.get("excess_total_return")
            row[f"{prefix}excess_sharpe"] = window_metrics.get("excess_sharpe")
            row[f"{prefix}excess_max_drawdown"] = window_metrics.get("excess_max_drawdown")
            row[f"{prefix}avg_turnover"] = window_metrics.get("avg_turnover")

        summary_rows.append(row)
        latest_key = windows[-1][0] if windows else "latest"
        print(
            f"  - {label}: full_excess_sharpe={metrics.get('excess_sharpe', float('nan')):.3f} | "
            f"{latest_key}_excess_sharpe={row.get(f'{latest_key}_excess_sharpe', float('nan')):.3f} | "
            f"{latest_key}_excess_return={row.get(f'{latest_key}_excess_total_return', float('nan')):.2%}"
        )

    summary_df = pd.DataFrame(summary_rows)
    sort_keys = []
    if windows:
        latest_prefix = f"{windows[-1][0]}_"
        sort_keys.extend([f"{latest_prefix}excess_sharpe", f"{latest_prefix}excess_total_return"])
    sort_keys.extend(["full_excess_sharpe", "full_excess_total_return"])
    summary_df = summary_df.sort_values(sort_keys, ascending=[False] * len(sort_keys)).reset_index(drop=True)
    summary_df.to_csv(output_root / "repair_scan_summary.csv", index=False, encoding="utf-8-sig")

    print("[7/7] done.")
    print(f"output: {output_root}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
