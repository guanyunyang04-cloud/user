from __future__ import annotations

import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json

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
from daily_research.baseline.market_context import compute_continuous_market_context
from daily_research.baseline.ml_alpha import (
    MLAplhaConfig,
    blend_scores,
    build_ml_feature_bundle,
    rolling_ml_scores_multi_detail,
)
from daily_research.baseline.portfolio import build_target_weights
from daily_research.baseline.regime import apply_market_regime_filter
from daily_research.execution.liquidity_universe import build_rolling_liquidity_membership


DEFAULT_WINDOWS = "recent_full:20250307:20260319,latest_weak:20250905:20260319"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Formal round-2 scan for context-score-driven soft execution adjustments."
    )
    parser.add_argument("--data-source", choices=["tq"], default="tq")
    parser.add_argument("--start-date", default="20210101")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--rolling-liquidity-pool", choices=["liquid300", "liquid500", "liquid800"], default="liquid500")
    parser.add_argument("--pool-rebalance-days", type=int, default=21)
    parser.add_argument("--pool-adv-window", type=int, default=20)
    parser.add_argument("--holding-count", type=int, default=5)
    parser.add_argument("--rebalance-freq", default="1d")
    parser.add_argument("--min-adv20", type=float, default=50_000.0)
    parser.add_argument("--min-price", type=float, default=2.0)
    parser.add_argument("--max-price", type=float, default=300.0)
    parser.add_argument("--max-weight", type=float, default=0.25)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--regime-ma-window", type=int, default=50)
    parser.add_argument("--regime-vol-window", type=int, default=20)
    parser.add_argument("--regime-max-annual-vol", type=float, default=0.32)
    parser.add_argument("--regime-quadrants", default="trend_up_low_vol,trend_up_high_vol")
    parser.add_argument("--no-style-cap", action="store_true")
    parser.add_argument("--max-style-weight", type=float, default=0.50)
    parser.add_argument("--industry-cap", action="store_true")
    parser.add_argument("--max-industry-weight", type=float, default=0.40)
    parser.add_argument("--ml-target-horizons", default="5,10,20")
    parser.add_argument("--ml-horizon-weights", default="5:0.2,10:0.3,20:0.5")
    parser.add_argument("--ml-train-window-days", type=int, default=504)
    parser.add_argument("--ml-retrain-every-days", type=int, default=21)
    parser.add_argument("--ml-min-train-dates", type=int, default=120)
    parser.add_argument("--ml-max-samples-per-day", type=int, default=600)
    parser.add_argument("--ml-max-train-rows", type=int, default=200000)
    parser.add_argument("--ml-random-seed", type=int, default=7)
    parser.add_argument("--ml-model-family", choices=["histgb", "etr", "lgbm"], default="lgbm")
    parser.add_argument("--ensemble-ml-weight", type=float, default=0.70)
    parser.add_argument("--ensemble-none-weight", type=float, default=0.20)
    parser.add_argument("--ensemble-v2-weight", type=float, default=0.10)
    parser.add_argument("--enhanced-profile", default="up_low_breakout_v2")
    parser.add_argument("--windows", default=DEFAULT_WINDOWS)
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def _parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _parse_horizon_weights(raw: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        horizon_raw, weight_raw = item.split(":", 1)
        out[int(horizon_raw.strip())] = float(weight_raw.strip())
    return out


def _parse_named_windows(raw: str | None) -> list[tuple[str, str, str]]:
    if not raw:
        return []
    out: list[tuple[str, str, str]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, start, end = chunk.split(":")
        out.append((name.strip(), start.strip(), end.strip()))
    return out


def _subset_raw_df_dict_to_stocks(
    raw_df_dict: dict[str, pd.DataFrame],
    benchmark: str,
    stocks: list[str],
) -> dict[str, pd.DataFrame]:
    keep = [benchmark] + [stock for stock in stocks if stock != benchmark]
    keep_set = set(keep)
    return {
        field: frame.reindex(columns=[col for col in frame.columns if col in keep_set])
        for field, frame in raw_df_dict.items()
    }


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
    return float(dd.min())


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
            "soft_override_active_ratio": np.nan,
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
        "soft_override_active_ratio": float(window["soft_override_active"].mean()) if "soft_override_active" in window else np.nan,
    }


def _build_regime_quantiles(context_score: pd.Series, regime_on: pd.Series, quantiles: int = 5) -> pd.Series:
    out = pd.Series(np.nan, index=context_score.index, dtype=float)
    valid = context_score.loc[regime_on.fillna(False)].dropna()
    if len(valid) < quantiles:
        return out
    ranked = valid.rank(method="first")
    out.loc[valid.index] = pd.qcut(ranked, quantiles, labels=list(range(1, quantiles + 1))).astype(float)
    return out


def _candidate_specs() -> list[dict[str, Any]]:
    return [
        {"label": "baseline"},
        {
            "label": "low_ctx_hold3",
            "target_overrides": {"holding_count": 3},
        },
        {
            "label": "low_ctx_maxw20",
            "target_overrides": {"max_weight": 0.20},
        },
        {
            "label": "low_ctx_turnover1",
            "position_overrides": {"turnover_limit": 1.00},
        },
        {
            "label": "low_ctx_style40",
            "target_overrides": {"max_style_weight": 0.40},
        },
    ]


def _build_override_frames(
    index: pd.Index,
    active_mask: pd.Series,
    candidate: dict[str, Any],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    active_index = index[active_mask.reindex(index).fillna(False)]
    target_df = None
    position_df = None
    if candidate.get("target_overrides"):
        target_df = pd.DataFrame(index=index)
        for key, value in candidate["target_overrides"].items():
            target_df[key] = np.nan
            target_df.loc[active_index, key] = value
    if candidate.get("position_overrides"):
        position_df = pd.DataFrame(index=index)
        for key, value in candidate["position_overrides"].items():
            position_df[key] = np.nan
            position_df.loc[active_index, key] = value
    return target_df, position_df


def _override_summary(
    active_mask: pd.Series,
    latest_weak_window: tuple[str, str],
) -> dict[str, Any]:
    weak_start = pd.Timestamp(latest_weak_window[0])
    weak_end = pd.Timestamp(latest_weak_window[1])
    weak_mask = (active_mask.index >= weak_start) & (active_mask.index <= weak_end)
    return {
        "override_signal_days": int(active_mask.sum()),
        "override_signal_ratio": float(active_mask.mean()),
        "latest_weak_override_days": int(active_mask.loc[weak_mask].sum()),
        "latest_weak_override_ratio": float(active_mask.loc[weak_mask].mean()) if weak_mask.any() else np.nan,
    }


def _candidate_verdict(candidate_row: dict[str, Any], baseline_row: dict[str, Any]) -> str:
    weak_better = (
        float(candidate_row["latest_weak_excess_total_return"]) > float(baseline_row["latest_weak_excess_total_return"])
        and float(candidate_row["latest_weak_excess_sharpe"]) > float(baseline_row["latest_weak_excess_sharpe"])
    )
    strong_hurt = (
        float(candidate_row["full_excess_sharpe"]) < float(baseline_row["full_excess_sharpe"])
        or float(candidate_row["recent_full_excess_sharpe"]) < float(baseline_row["recent_full_excess_sharpe"])
    )
    if weak_better and strong_hurt:
        return "stop_due_to_tradeoff"
    if weak_better and not strong_hurt:
        return "keep_for_next_round"
    return "no_promotion"


def _write_markdown(path: Path, summary_df: pd.DataFrame, report: dict[str, Any]) -> None:
    baseline = summary_df.loc[summary_df["label"] == "baseline"].iloc[0].to_dict()
    best = summary_df.iloc[0].to_dict()
    lines = [
        "# 连续状态软调节第二轮正式回测",
        "",
        "## 本轮目标",
        "- 保留现有四象限门控不动。",
        "- 只在低 `context_score` 的 `regime_on` 日期做轻度去风险。",
        "- 首批单独测试 `holding_count / max_weight / turnover_limit / max_style_weight` 四个旋钮。",
        "",
        "## 核心结果",
        f"- 本轮最佳候选：`{best['label']}`",
        f"- baseline 最新弱窗口：{float(baseline['latest_weak_excess_total_return']):.2%} / {float(baseline['latest_weak_excess_sharpe']):.3f}",
        f"- 最佳候选最新弱窗口：{float(best['latest_weak_excess_total_return']):.2%} / {float(best['latest_weak_excess_sharpe']):.3f}",
        f"- 总结结论：{report['overall_verdict']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cfg = ResearchConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        universe_scope="all_a",
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
        enable_market_regime_filter=True,
        regime_ma_window=args.regime_ma_window,
        regime_vol_window=args.regime_vol_window,
        regime_max_annual_vol=args.regime_max_annual_vol,
        regime_allowed_quadrants=_parse_csv_list(args.regime_quadrants),
        enable_style_cap=not args.no_style_cap,
        max_style_weight=args.max_style_weight,
        enable_industry_cap=args.industry_cap,
        max_industry_weight=args.max_industry_weight,
    )
    ml_cfg = MLAplhaConfig(
        target_horizons=_parse_int_tuple(args.ml_target_horizons),
        target_horizon_weights=_parse_horizon_weights(args.ml_horizon_weights),
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
        train_regime_only=True,
        execution_mode=cfg.execution_mode,
    )
    windows = _parse_named_windows(args.windows)
    latest_weak_window = next((start_end[1:] for start_end in windows if start_end[0] == "latest_weak"), None)
    if latest_weak_window is None:
        raise ValueError("windows must contain latest_weak")

    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path("daily_research/output") / (args.experiment_tag.strip() or f"continuous_market_context_soft_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print("[1/7] loading all-A universe...")
    universe = load_universe_from_tq("all_a")
    history_window = resolve_history_window(
        cfg=cfg,
        ml_cfg=ml_cfg,
        requested_start_date=args.start_date,
        end_date=args.end_date,
        mode="train",
        auto_trim_history=False,
    )
    raw_df_dict, raw_cache_meta = load_raw_data_with_cache(
        data_source=args.data_source,
        csv_folder=None,
        universe=universe,
        benchmark=cfg.benchmark,
        history_window=history_window,
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
    )

    print(f"[2/7] building rolling {args.rolling_liquidity_pool} membership...")
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
        raise RuntimeError("Rolling liquidity pool is empty for the requested window.")
    prepared_raw_df_dict = _subset_raw_df_dict_to_stocks(raw_df_dict, cfg.benchmark, rolling_union)
    prepared_raw_cache_key = (
        f"{raw_cache_meta['cache_key']}|{args.rolling_liquidity_pool}|"
        f"{args.pool_rebalance_days}|{args.pool_adv_window}"
    )

    print("[3/7] building prepared bundle...")
    prepared_bundle, prepared_cache_meta = build_prepared_bundle_with_cache(
        raw_df_dict=prepared_raw_df_dict,
        raw_cache_key=prepared_raw_cache_key,
        cfg=cfg,
        enhanced_profile=args.enhanced_profile,
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
    )
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

    candidate_columns = [col for col in prepared_raw_df_dict["Close"].columns if col != cfg.benchmark]
    industry_map = load_industry_map_from_tq(candidate_columns) if cfg.enable_industry_cap else None
    style_map = load_style_map_from_tq(candidate_columns) if cfg.enable_style_cap else None

    print("[4/7] computing shared ML scores...")
    ml_score, training_log, _ = rolling_ml_scores_multi_detail(
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
    training_log.to_csv(output_root / "shared_training_log.csv", index=False, encoding="utf-8-sig")

    final_score = blend_scores(
        ml_score,
        prepared_bundle["score_none"],
        prepared_bundle["score_v2"],
        ml_cfg,
        quadrant_series=prepared_bundle["regime_state"]["quadrant"],
    ).where(current_membership_mask)

    print("[5/7] computing shared context scores...")
    context_df = compute_continuous_market_context(
        prepared_bundle["df_dict"],
        prepared_bundle["regime_state"],
        membership_mask=current_membership_mask,
    )
    context_quantile = _build_regime_quantiles(
        context_df["context_score"].reindex(final_score.index),
        prepared_bundle["regime_state"]["regime_on"].reindex(final_score.index),
        quantiles=5,
    )
    low_context_mask = context_quantile.eq(1) & prepared_bundle["regime_state"]["regime_on"].reindex(final_score.index).fillna(False)

    context_meta = pd.DataFrame(
        {
            "context_score": context_df["context_score"].reindex(final_score.index),
            "context_quantile": context_quantile,
            "low_context_flag": low_context_mask,
            "quadrant": prepared_bundle["regime_state"]["quadrant"].reindex(final_score.index),
            "regime_on": prepared_bundle["regime_state"]["regime_on"].reindex(final_score.index),
        }
    )
    context_meta.to_csv(output_root / "context_signal_map.csv", encoding="utf-8-sig")

    scan_meta = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "framework": "continuous_market_context_soft_scan",
        "history_window": history_window_to_dict(history_window),
        "raw_cache": raw_cache_meta,
        "prepared_cache": prepared_cache_meta,
        "rolling_liquidity_pool": args.rolling_liquidity_pool,
        "rolling_pool_union_size": int(current_membership_mask.columns[current_membership_mask.any(axis=0)].size),
        "rolling_pool_rebalance_count": int(len(rolling_pool_artifact.schedule_df)),
        "base_config": {"research": asdict(cfg), "ml": asdict(ml_cfg)},
        "low_context_summary": _override_summary(low_context_mask, latest_weak_window),
        "candidates": _candidate_specs(),
    }
    (output_root / "scan_config.json").write_text(json.dumps(scan_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[6/7] running candidates...")
    summary_rows: list[dict[str, Any]] = []
    for candidate in _candidate_specs():
        label = str(candidate["label"])
        run_dir = output_root / label
        run_dir.mkdir(parents=True, exist_ok=True)

        target_override_df, position_override_df = _build_override_frames(final_score.index, low_context_mask, candidate)
        target_weights = build_target_weights(
            final_score,
            cfg,
            industry_map=industry_map,
            style_map=style_map,
            daily_config_overrides=target_override_df,
        )
        score_for_backtest = final_score.fillna(0.0)
        if cfg.enable_market_regime_filter:
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
            config=cfg,
            regime_on=prepared_bundle["regime_state"]["regime_on"],
            daily_position_overrides=position_override_df,
        )

        override_info = _override_summary(low_context_mask, latest_weak_window)
        override_info["target_overrides"] = candidate.get("target_overrides", {})
        override_info["position_overrides"] = candidate.get("position_overrides", {})
        override_info["context_quantile_trigger"] = 1

        metrics.update(
            {
                "framework": "continuous_market_context_soft_scan",
                "candidate_label": label,
                "prepared_cache_key": prepared_cache_meta["cache_key"],
                "prepared_cache_hit": bool(prepared_cache_meta["cache_hit"]),
                "context_override_summary": override_info,
            }
        )

        equity_df.to_csv(run_dir / "equity_curve.csv", encoding="utf-8-sig")
        action_df.to_csv(run_dir / "actions.csv", index=False, encoding="utf-8-sig")
        target_weights.to_csv(run_dir / "target_weights.csv", encoding="utf-8-sig")
        if target_override_df is not None:
            target_override_df.to_csv(run_dir / "daily_target_overrides.csv", encoding="utf-8-sig")
        if position_override_df is not None:
            position_override_df.to_csv(run_dir / "daily_position_overrides.csv", encoding="utf-8-sig")
        (run_dir / "override_summary.json").write_text(json.dumps(override_info, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        row = {
            "label": label,
            "override_signal_days": override_info["override_signal_days"],
            "override_signal_ratio": override_info["override_signal_ratio"],
            "latest_weak_override_ratio": override_info["latest_weak_override_ratio"],
            "full_excess_total_return": metrics.get("excess_total_return"),
            "full_excess_sharpe": metrics.get("excess_sharpe"),
            "full_excess_max_drawdown": metrics.get("excess_max_drawdown"),
            "full_avg_turnover": metrics.get("avg_turnover"),
            "full_avg_holding_count": metrics.get("avg_holding_count"),
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
            row[f"{prefix}avg_holding_count"] = window_metrics.get("avg_holding_count")
            row[f"{prefix}soft_override_active_ratio"] = window_metrics.get("soft_override_active_ratio")
        summary_rows.append(row)
        print(
            f"  - {label}: full_sharpe={metrics.get('excess_sharpe', float('nan')):.3f} | "
            f"latest_weak_sharpe={row.get('latest_weak_excess_sharpe', float('nan')):.3f} | "
            f"latest_weak_return={row.get('latest_weak_excess_total_return', float('nan')):.2%}"
        )

    summary_df = pd.DataFrame(summary_rows)
    baseline_row = summary_df.loc[summary_df["label"] == "baseline"].iloc[0].to_dict()
    summary_df["candidate_verdict"] = summary_df.apply(
        lambda row: "baseline" if row["label"] == "baseline" else _candidate_verdict(row.to_dict(), baseline_row),
        axis=1,
    )
    summary_df = summary_df.sort_values(
        ["latest_weak_excess_sharpe", "latest_weak_excess_total_return", "full_excess_sharpe"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    summary_df.to_csv(output_root / "soft_scan_summary.csv", index=False, encoding="utf-8-sig")

    promote = summary_df.loc[summary_df["candidate_verdict"] == "keep_for_next_round"]
    tradeoff = summary_df.loc[summary_df["candidate_verdict"] == "stop_due_to_tradeoff"]
    overall_verdict = "stop_soft_context_line_for_now"
    if not promote.empty:
        overall_verdict = "keep_best_soft_context_candidate"
    elif not tradeoff.empty:
        overall_verdict = "stop_due_to_weak_window_vs_strong_window_tradeoff"

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_root),
        "overall_verdict": overall_verdict,
        "baseline_label": "baseline",
        "summary_rows": summary_df.to_dict(orient="records"),
        "low_context_summary": scan_meta["low_context_summary"],
    }
    (output_root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(output_root / "report.md", summary_df, report)

    print("[7/7] done.")
    print(f"output: {output_root}")
    print(summary_df.to_string(index=False))
    print(json.dumps({"overall_verdict": overall_verdict}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
