from __future__ import annotations

import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json

import pandas as pd

from daily_research.baseline.advanced_ml_runtime import (
    build_prepared_bundle_with_cache,
    history_window_to_dict,
    load_raw_data_with_cache,
    resolve_history_window,
)
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import load_industry_map_from_tq, load_style_map_from_tq, load_universe_from_tq
from daily_research.baseline.diagnose_advanced_ml_ensemble import (
    DEFAULT_FOCUS_STATE,
    _load_pickle,
    _load_stocks_from_file,
    _ml_score_cache_path,
    _parse_csv_list,
    _parse_horizon_weights,
    _parse_int_tuple,
    _parse_named_windows,
    _parse_stocks,
    _resolve_weak_window_name,
    _run_candidate,
    _save_pickle,
    _subset_raw_df_dict_to_stocks,
)
from daily_research.baseline.ml_alpha import MLAplhaConfig, build_ml_feature_bundle, rolling_ml_scores_multi_detail
from daily_research.execution.liquidity_universe import build_rolling_liquidity_membership
from daily_research.progress import StageProgress, iter_progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Formally compare retrain frequency and LGBM boosting rounds for fixed advanced_ml ensemble candidates."
    )
    parser.add_argument("--data-source", choices=["tq", "csv"], default="tq")
    parser.add_argument("--csv-folder", default=None)
    parser.add_argument("--stocks", default=None)
    parser.add_argument("--stocks-file", default=None)
    parser.add_argument("--start-date", default="20210101")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--universe-scope", default="all_a")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument(
        "--rolling-liquidity-pool",
        choices=["liquid300", "liquid500", "liquid800"],
        default="liquid500",
    )
    parser.add_argument("--pool-rebalance-days", type=int, default=21)
    parser.add_argument("--pool-adv-window", type=int, default=20)
    parser.add_argument("--holding-count", type=int, default=5)
    parser.add_argument("--rebalance-freq", default="1d")
    parser.add_argument("--enhanced-profile", default="up_low_breakout_v2")
    parser.add_argument("--focus-state", default=DEFAULT_FOCUS_STATE)
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--output-dir", default="")

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
    parser.add_argument("--ml-min-train-dates", type=int, default=120)
    parser.add_argument("--ml-max-samples-per-day", type=int, default=600)
    parser.add_argument("--ml-max-train-rows", type=int, default=200000)
    parser.add_argument("--ml-random-seed", type=int, default=7)
    parser.add_argument("--ml-model-family", choices=["lgbm"], default="lgbm")
    parser.add_argument("--retrain-grid", default="5,10,21,42")
    parser.add_argument("--lgbm-estimator-grid", default="130,260,520")

    parser.add_argument("--ensemble-ml-weight", type=float, default=0.70)
    parser.add_argument("--ensemble-none-weight", type=float, default=0.20)
    parser.add_argument("--ensemble-v2-weight", type=float, default=0.10)
    parser.add_argument(
        "--windows",
        default="recent_full:20250307:20260326,weak_window_20250905_20260319:20250905:20260319",
    )
    parser.add_argument("--auto-trim-history", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def _parse_int_grid(raw: str | None, fallback: list[int]) -> list[int]:
    if not raw:
        return list(fallback)
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    deduped = list(dict.fromkeys(values))
    if not deduped:
        return list(fallback)
    return deduped


def _build_strict_compare_candidates(focus_state: str, base_weights: dict[str, float]) -> list[dict[str, Any]]:
    candidates = [
        {
            "label": "base_global",
            "blend_kind": "strict_compare",
            "weights": dict(base_weights),
            "state_ensemble_weights": {},
            "distance_to_default": 0.0,
        },
        {
            "label": f"{focus_state}_ml25_none25_v250",
            "blend_kind": "strict_compare",
            "weights": dict(base_weights),
            "state_ensemble_weights": {focus_state: {"ml": 0.25, "none": 0.25, "v2": 0.50}},
            "distance_to_default": 0.90,
        },
        {
            "label": f"{focus_state}_ml25_none20_v255",
            "blend_kind": "strict_compare",
            "weights": dict(base_weights),
            "state_ensemble_weights": {focus_state: {"ml": 0.25, "none": 0.20, "v2": 0.55}},
            "distance_to_default": 0.90,
        },
    ]
    return candidates


def _safe_num(value: Any) -> str:
    if pd.isna(value):
        return "nan"
    return f"{float(value):.3f}"


def _safe_pct(value: Any) -> str:
    if pd.isna(value):
        return "nan"
    return f"{float(value):.2%}"


def _pick_best_rows(
    df: pd.DataFrame,
    metric: str,
    tie_breaker: str,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    rows: list[pd.Series] = []
    for label, label_df in df.groupby("label", sort=False):
        best = label_df.sort_values([metric, tie_breaker], ascending=[False, False]).iloc[0].copy()
        best["label"] = label
        rows.append(best)
    return pd.DataFrame(rows)


def _build_metric_range_rows(df: pd.DataFrame, weak_window_name: str, focus_state: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    weak_metric = f"{weak_window_name}_excess_sharpe"
    focus_weak_metric = f"{focus_state}_{weak_window_name}_excess_sharpe"
    for label, label_df in df.groupby("label", sort=False):
        rows.append(
            {
                "label": label,
                "full_excess_sharpe_min": float(label_df["full_excess_sharpe"].min()),
                "full_excess_sharpe_max": float(label_df["full_excess_sharpe"].max()),
                "full_excess_sharpe_range": float(label_df["full_excess_sharpe"].max() - label_df["full_excess_sharpe"].min()),
                f"{weak_metric}_min": float(label_df[weak_metric].min()),
                f"{weak_metric}_max": float(label_df[weak_metric].max()),
                f"{weak_metric}_range": float(label_df[weak_metric].max() - label_df[weak_metric].min()),
                f"{focus_weak_metric}_min": float(label_df[focus_weak_metric].min()),
                f"{focus_weak_metric}_max": float(label_df[focus_weak_metric].max()),
                f"{focus_weak_metric}_range": float(label_df[focus_weak_metric].max() - label_df[focus_weak_metric].min()),
            }
        )
    return pd.DataFrame(rows)


def _render_summary(
    *,
    latest_data_date: str,
    history_window: dict[str, Any],
    results_df: pd.DataFrame,
    default_rows: pd.DataFrame,
    best_full_df: pd.DataFrame,
    best_weak_df: pd.DataFrame,
    range_df: pd.DataFrame,
    weak_window_name: str,
    focus_state: str,
    retrain_grid: list[int],
    lgbm_estimator_grid: list[int],
) -> str:
    weak_metric = f"{weak_window_name}_excess_sharpe"
    focus_weak_metric = f"{focus_state}_{weak_window_name}_excess_sharpe"
    lines: list[str] = []
    lines.append("# Advanced ML Retrain / Tree Impact")
    lines.append("")
    lines.append(f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- latest_data_date: {latest_data_date}")
    lines.append(
        f"- history_window: {history_window['effective_start_date']} -> {history_window['end_date']} (mode={history_window['mode']})"
    )
    lines.append(f"- retrain_grid: {retrain_grid}")
    lines.append(f"- lgbm_estimator_grid: {lgbm_estimator_grid}")
    lines.append("")

    lines.append("## Default config rows")
    for _, row in default_rows.iterrows():
        lines.append(
            f"- {row['label']}: retrain={int(row['ml_retrain_every_days'])}, "
            f"lgbm_n_estimators={int(row['lgbm_n_estimators'])}, "
            f"full excess Sharpe {_safe_num(row['full_excess_sharpe'])}, "
            f"{weak_window_name} excess Sharpe {_safe_num(row[weak_metric])}"
        )
    lines.append("")

    lines.append("## Best by full_excess_sharpe")
    for _, row in best_full_df.iterrows():
        lines.append(
            f"- {row['label']}: retrain={int(row['ml_retrain_every_days'])}, "
            f"lgbm_n_estimators={int(row['lgbm_n_estimators'])}, "
            f"full excess Sharpe {_safe_num(row['full_excess_sharpe'])}, "
            f"{weak_window_name} excess Sharpe {_safe_num(row[weak_metric])}, "
            f"{focus_state}_{weak_window_name} excess Sharpe {_safe_num(row[focus_weak_metric])}"
        )
    lines.append("")

    lines.append(f"## Best by {weak_window_name}_excess_sharpe")
    for _, row in best_weak_df.iterrows():
        lines.append(
            f"- {row['label']}: retrain={int(row['ml_retrain_every_days'])}, "
            f"lgbm_n_estimators={int(row['lgbm_n_estimators'])}, "
            f"full excess Sharpe {_safe_num(row['full_excess_sharpe'])}, "
            f"{weak_window_name} excess Sharpe {_safe_num(row[weak_metric])}, "
            f"{focus_state}_{weak_window_name} excess Sharpe {_safe_num(row[focus_weak_metric])}"
        )
    lines.append("")

    lines.append("## Sensitivity ranges")
    for _, row in range_df.iterrows():
        lines.append(
            f"- {row['label']}: "
            f"full range {_safe_num(row['full_excess_sharpe_range'])}, "
            f"{weak_window_name} range {_safe_num(row[f'{weak_metric}_range'])}, "
            f"{focus_state}_{weak_window_name} range {_safe_num(row[f'{focus_weak_metric}_range'])}"
        )
    lines.append("")

    if not results_df.empty:
        overall_best = results_df.sort_values([weak_metric, "full_excess_sharpe"], ascending=[False, False]).iloc[0]
        lines.append("## Direct answer")
        lines.append(
            f"- In the current lgbm execution stack, changing retrain frequency and boosting rounds does affect results: "
            f"best observed row is {overall_best['label']} @ retrain={int(overall_best['ml_retrain_every_days'])}, "
            f"lgbm_n_estimators={int(overall_best['lgbm_n_estimators'])}, "
            f"{weak_window_name} excess Sharpe {_safe_num(overall_best[weak_metric])}, "
            f"full excess Sharpe {_safe_num(overall_best['full_excess_sharpe'])}."
        )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
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
    file_stocks = _load_stocks_from_file(args.stocks_file)
    if stocks or file_stocks:
        stocks = list(dict.fromkeys(stocks + file_stocks))
    if stocks:
        cfg.universe = stocks

    base_ml_cfg = MLAplhaConfig(
        target_horizon=args.ml_target_horizon,
        target_horizons=_parse_int_tuple(args.ml_target_horizons, args.ml_target_horizon),
        target_horizon_weights=_parse_horizon_weights(args.ml_horizon_weights),
        enhanced_profile=args.enhanced_profile,
        train_window_days=args.ml_train_window_days,
        retrain_every_days=21,
        min_train_dates=args.ml_min_train_dates,
        max_samples_per_day=args.ml_max_samples_per_day,
        max_train_rows=args.ml_max_train_rows,
        random_seed=args.ml_random_seed,
        model_family=args.ml_model_family,
        lgbm_n_estimators=260,
        ensemble_ml_weight=args.ensemble_ml_weight,
        ensemble_none_weight=args.ensemble_none_weight,
        ensemble_v2_weight=args.ensemble_v2_weight,
        train_regime_only=cfg.enable_market_regime_filter,
        execution_mode=cfg.execution_mode,
    )

    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path("daily_research/output")
        / (args.experiment_tag.strip() or f"advanced_ml_retrain_tree_impact_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    windows = _parse_named_windows(args.windows)
    weak_window_name = _resolve_weak_window_name(windows)
    if not weak_window_name:
        raise ValueError("windows must include an explicit weak_window_* section.")
    focus_state = str(args.focus_state).strip().lower() or DEFAULT_FOCUS_STATE
    retrain_grid = _parse_int_grid(args.retrain_grid, [5, 10, 21, 42])
    lgbm_estimator_grid = _parse_int_grid(args.lgbm_estimator_grid, [130, 260, 520])
    base_weights = {
        "ml": float(args.ensemble_ml_weight),
        "none": float(args.ensemble_none_weight),
        "v2": float(args.ensemble_v2_weight),
    }
    candidates = _build_strict_compare_candidates(focus_state, base_weights)

    with StageProgress(total=7, label="retrain/tree impact") as progress:
        with progress.stage("prepare universe", f"source={args.data_source}"):
            if args.data_source == "tq":
                if args.rolling_liquidity_pool and not cfg.universe:
                    cfg.universe = load_universe_from_tq(cfg.universe_scope)
                elif not cfg.universe and cfg.universe_scope == "all_a":
                    cfg.universe = load_universe_from_tq(cfg.universe_scope)
                elif not cfg.universe:
                    raise ValueError("TQ mode without explicit stocks currently requires all_a universe scope.")
            elif not args.csv_folder:
                raise ValueError("CSV mode requires --csv-folder.")

        with progress.stage("resolve history window", args.start_date):
            history_window = resolve_history_window(
                cfg=cfg,
                ml_cfg=base_ml_cfg,
                requested_start_date=args.start_date,
                end_date=args.end_date,
                mode="train",
                auto_trim_history=args.auto_trim_history,
            )
            progress.log(
                f"history window: {history_window.effective_start_date} -> {history_window.end_date} "
                f"(required_trading_days={history_window.required_trading_days})"
            )

        with progress.stage("load market data", f"stocks={len(cfg.universe)} benchmark={cfg.benchmark}"):
            raw_df_dict, raw_cache_meta = load_raw_data_with_cache(
                data_source=args.data_source,
                csv_folder=args.csv_folder,
                universe=cfg.universe,
                benchmark=cfg.benchmark,
                history_window=history_window,
                use_cache=not args.no_cache,
                refresh_cache=args.refresh_cache,
                progress_desc="load raw bars",
                progress_position=1,
            )
            progress.log(
                f"raw cache: {'hit' if raw_cache_meta['cache_hit'] else 'build'} | {raw_cache_meta['cache_path']}"
            )

        with progress.stage("build rolling liquidity pool", args.rolling_liquidity_pool or "fixed universe"):
            rolling_pool_artifact = None
            rolling_membership_mask = None
            prepared_raw_df_dict = raw_df_dict
            if args.rolling_liquidity_pool:
                raw_universe_df_dict = {k: v.drop(columns=[cfg.benchmark], errors="ignore") for k, v in raw_df_dict.items()}
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
                progress.log(
                    f"rolling pool union size: {len(rolling_union)} | rebalance count: {len(rolling_pool_artifact.schedule_df)}"
                )

        with progress.stage("build prepared bundle", args.enhanced_profile):
            prepared_raw_cache_key = (
                raw_cache_meta["cache_key"]
                if rolling_pool_artifact is None
                else f"{raw_cache_meta['cache_key']}|{args.rolling_liquidity_pool}|{args.pool_rebalance_days}|{args.pool_adv_window}"
            )
            prepared_bundle, prepared_cache_meta = build_prepared_bundle_with_cache(
                raw_df_dict=prepared_raw_df_dict,
                raw_cache_key=prepared_raw_cache_key,
                cfg=cfg,
                enhanced_profile=args.enhanced_profile,
                use_cache=not args.no_cache,
                refresh_cache=args.refresh_cache,
            )
            progress.log(
                f"factor cache: {'hit' if prepared_cache_meta['cache_hit'] else 'build'} | {prepared_cache_meta['cache_path']}"
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

        with progress.stage("load exposure maps", "style/industry"):
            candidate_columns = [col for col in prepared_bundle["df_dict"]["Close"].columns if col != cfg.benchmark]
            industry_map = None
            style_map = None
            if cfg.enable_industry_cap and args.data_source == "tq":
                industry_map = load_industry_map_from_tq(candidate_columns)
            if cfg.enable_style_cap and args.data_source == "tq":
                style_map = load_style_map_from_tq(candidate_columns)

        with progress.stage("run strict comparisons", f"{len(retrain_grid)}x{len(lgbm_estimator_grid)} configs"):
            results: list[dict[str, Any]] = []
            training_logs_dir = output_root / "training_logs"
            training_logs_dir.mkdir(parents=True, exist_ok=True)
            total_configs = len(retrain_grid) * len(lgbm_estimator_grid)
            config_index = 0
            for retrain_every_days in retrain_grid:
                for lgbm_n_estimators in lgbm_estimator_grid:
                    config_index += 1
                    config_name = f"r{int(retrain_every_days):02d}_n{int(lgbm_n_estimators):03d}"
                    progress.log(f"[{config_index}/{total_configs}] {config_name}")
                    run_ml_cfg = MLAplhaConfig(**asdict(base_ml_cfg))
                    run_ml_cfg.retrain_every_days = int(retrain_every_days)
                    run_ml_cfg.lgbm_n_estimators = int(lgbm_n_estimators)

                    ml_score_cache_path = _ml_score_cache_path(prepared_cache_meta["cache_key"], run_ml_cfg)
                    cached_ml_payload = None if args.refresh_cache else _load_pickle(ml_score_cache_path)
                    if cached_ml_payload is not None:
                        training_log = cached_ml_payload["training_log"]
                        shared_per_horizon_scores = cached_ml_payload["per_horizon_scores"]
                    else:
                        _, training_log, shared_per_horizon_scores = rolling_ml_scores_multi_detail(
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
                        if not args.no_cache:
                            _save_pickle(
                                ml_score_cache_path,
                                {
                                    "training_log": training_log,
                                    "per_horizon_scores": shared_per_horizon_scores,
                                },
                            )

                    training_log.to_csv(training_logs_dir / f"{config_name}.csv", index=False, encoding="utf-8-sig")
                    for candidate in iter_progress(
                        candidates,
                        total=len(candidates),
                        desc=f"compare {config_name}",
                        unit="candidate",
                        position=1,
                    ):
                        row, _ = _run_candidate(
                            candidate=candidate,
                            cfg=cfg,
                            ml_cfg=run_ml_cfg,
                            shared_per_horizon_scores=shared_per_horizon_scores,
                            prepared_bundle=prepared_bundle,
                            current_membership_mask=current_membership_mask,
                            industry_map=industry_map,
                            style_map=style_map,
                            windows=windows,
                            focus_state=focus_state,
                        )
                        row["ml_retrain_every_days"] = int(retrain_every_days)
                        row["lgbm_n_estimators"] = int(lgbm_n_estimators)
                        row["ml_score_cache_path"] = str(ml_score_cache_path)
                        results.append(row)

    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise RuntimeError("No comparison results were produced.")

    weak_metric = f"{weak_window_name}_excess_sharpe"
    best_full_df = _pick_best_rows(results_df, "full_excess_sharpe", weak_metric).sort_values("label").reset_index(drop=True)
    best_weak_df = _pick_best_rows(results_df, weak_metric, "full_excess_sharpe").sort_values("label").reset_index(drop=True)
    default_rows = results_df.loc[
        results_df["ml_retrain_every_days"].eq(21) & results_df["lgbm_n_estimators"].eq(260)
    ].sort_values("label").reset_index(drop=True)
    range_df = _build_metric_range_rows(results_df, weak_window_name, focus_state).sort_values("label").reset_index(drop=True)

    latest_data_date = pd.Timestamp(
        prepared_bundle["factor_bundle"]["raw_inputs"]["Close"].dropna(how="all").index.max()
    ).strftime("%Y-%m-%d")

    results_df.to_csv(output_root / "strict_compare_summary.csv", index=False, encoding="utf-8-sig")
    best_full_df.to_csv(output_root / "best_by_full_excess_sharpe.csv", index=False, encoding="utf-8-sig")
    best_weak_df.to_csv(output_root / f"best_by_{weak_window_name}_excess_sharpe.csv", index=False, encoding="utf-8-sig")
    default_rows.to_csv(output_root / "default_config_rows.csv", index=False, encoding="utf-8-sig")
    range_df.to_csv(output_root / "sensitivity_ranges.csv", index=False, encoding="utf-8-sig")

    run_meta = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latest_data_date": latest_data_date,
        "history_window": history_window_to_dict(history_window),
        "raw_cache": raw_cache_meta,
        "prepared_cache": prepared_cache_meta,
        "focus_state": focus_state,
        "windows": [{"name": name, "start": start, "end": end} for name, start, end in windows],
        "strict_compare_candidates": candidates,
        "retrain_grid": retrain_grid,
        "lgbm_estimator_grid": lgbm_estimator_grid,
        "base_config": {"research": asdict(cfg), "ml": asdict(base_ml_cfg)},
    }
    (output_root / "run_config.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_text = _render_summary(
        latest_data_date=latest_data_date,
        history_window=history_window_to_dict(history_window),
        results_df=results_df,
        default_rows=default_rows,
        best_full_df=best_full_df,
        best_weak_df=best_weak_df,
        range_df=range_df,
        weak_window_name=weak_window_name,
        focus_state=focus_state,
        retrain_grid=retrain_grid,
        lgbm_estimator_grid=lgbm_estimator_grid,
    )
    (output_root / "summary.md").write_text(summary_text, encoding="utf-8")

    print(f"Output: {output_root}")
    print(summary_text)


if __name__ == "__main__":
    main()
