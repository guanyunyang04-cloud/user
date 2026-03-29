from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from daily_research.baseline.advanced_ml_runtime import (
    build_prepared_bundle_with_cache,
    history_window_to_dict,
    load_raw_data_with_cache,
    resolve_history_window,
)
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
    get_next_trading_date,
    load_universe_from_tq,
)
from daily_research.baseline.regime import resolve_regime_label_series
from daily_research.baseline.state_profiles import validate_state_profile_selector
from daily_research.baseline.ml_alpha import (
    MLAplhaConfig,
    build_ml_target,
    resolve_horizon_weights,
    rolling_ml_scores_multi_detail,
    save_ml_artifact,
    train_point_in_time_model_bundle,
)
from daily_research.progress import StageProgress


def parse_args():
    parser = argparse.ArgumentParser(description="Train and export point-in-time trade model artifact")
    parser.add_argument("--data-source", choices=["tq", "csv"], default="tq")
    parser.add_argument("--csv-folder", default=None)
    parser.add_argument("--stocks", default=None)
    parser.add_argument("--stocks-file", default=None, help="Path to txt/csv file containing stock codes.")
    parser.add_argument("--start-date", default="20210101")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--universe-scope", default="all_a")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--enhanced-profile", default="up_low_breakout_v2")
    parser.add_argument("--artifact-path", default="daily_research/execution/models/latest_ml_model.joblib")
    parser.add_argument("--artifact-meta-path", default="")
    parser.add_argument("--min-adv20", type=float, default=50_000.0)
    parser.add_argument("--min-price", type=float, default=2.0)
    parser.add_argument("--max-price", type=float, default=300.0)
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
        help="状态标签来源；当前非 none profile 仍只支持 quadrant。",
    )
    parser.add_argument("--regime-quadrants", default="trend_up_low_vol,trend_up_high_vol")
    parser.add_argument("--ml-target-horizon", type=int, default=20)
    parser.add_argument("--ml-target-horizons", default="5,10,20")
    parser.add_argument("--ml-horizon-weights", default="5:0.2,10:0.3,20:0.5")
    parser.add_argument(
        "--ml-state-horizon-profiles",
        default="",
        help="按市场状态指定周期权重，例如 trend_up_low_vol=5:0.15,10:0.25,20:0.60;trend_up_high_vol=5:0.30,10:0.35,20:0.35",
    )
    parser.add_argument("--ml-train-window-days", type=int, default=504)
    parser.add_argument("--ml-min-train-dates", type=int, default=120)
    parser.add_argument("--ml-max-samples-per-day", type=int, default=600)
    parser.add_argument("--ml-max-train-rows", type=int, default=200000)
    parser.add_argument("--ml-random-seed", type=int, default=7)
    parser.add_argument("--ml-model-family", choices=["histgb", "etr", "lgbm"], default="histgb")
    parser.add_argument("--lgbm-n-estimators", type=int, default=260)
    parser.add_argument("--ensemble-ml-weight", type=float, default=0.70)
    parser.add_argument("--ensemble-none-weight", type=float, default=0.20)
    parser.add_argument("--ensemble-v2-weight", type=float, default=0.10)
    parser.add_argument("--ensemble-state-weights", default="")
    parser.add_argument("--skip-validation-summary", action="store_true")
    parser.add_argument("--validation-retrain-every-days", type=int, default=21)
    parser.add_argument("--validation-min-observations", type=int, default=20)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--no-auto-trim-history", action="store_true")
    return parser.parse_args()


def _compute_rank_ic_frame(
    score_df: pd.DataFrame,
    label_df: pd.DataFrame,
    min_observations: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    common_index = score_df.index.intersection(label_df.index)
    for dt in common_index:
        score_row = score_df.loc[dt]
        label_row = label_df.loc[dt]
        valid = score_row.notna() & label_row.notna()
        sample_count = int(valid.sum())
        rank_ic = np.nan
        if sample_count >= int(min_observations):
            rank_ic = score_row.loc[valid].corr(label_row.loc[valid], method="spearman")
        rows.append(
            {
                "date": pd.Timestamp(dt),
                "rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
                "sample_count": sample_count,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["rank_ic", "sample_count"])
    return out.set_index("date")


def _summarize_rank_ic_frame(ic_frame: pd.DataFrame) -> dict[str, object]:
    if ic_frame.empty:
        return {
            "start_date": "",
            "end_date": "",
            "obs_days": 0,
            "mean_rank_ic": np.nan,
            "rank_ic_std": np.nan,
            "rank_ic_ir": np.nan,
            "positive_ratio": np.nan,
            "avg_sample_count": np.nan,
        }
    valid = ic_frame["rank_ic"].dropna()
    if valid.empty:
        return {
            "start_date": str(ic_frame.index.min().date()),
            "end_date": str(ic_frame.index.max().date()),
            "obs_days": 0,
            "mean_rank_ic": np.nan,
            "rank_ic_std": np.nan,
            "rank_ic_ir": np.nan,
            "positive_ratio": np.nan,
            "avg_sample_count": np.nan,
        }

    std = float(valid.std()) if len(valid) > 1 else np.nan
    mean = float(valid.mean())
    return {
        "start_date": str(valid.index.min().date()),
        "end_date": str(valid.index.max().date()),
        "obs_days": int(len(valid)),
        "mean_rank_ic": mean,
        "rank_ic_std": std,
        "rank_ic_ir": float(mean / std) if std and not np.isnan(std) else np.nan,
        "positive_ratio": float((valid > 0).mean()),
        "avg_sample_count": float(ic_frame.loc[valid.index, "sample_count"].mean()),
    }


def _slice_rank_ic_window(ic_frame: pd.DataFrame, trading_days: int | None) -> pd.DataFrame:
    if trading_days is None or trading_days <= 0:
        return ic_frame
    return ic_frame.tail(int(trading_days))


def _combine_per_horizon_labels(
    label_frames: dict[int, pd.DataFrame],
    close: pd.DataFrame,
    regime_state: pd.DataFrame,
    config: MLAplhaConfig,
) -> pd.DataFrame:
    if not label_frames:
        return pd.DataFrame(index=close.index, columns=close.columns, dtype=float)

    combined = pd.DataFrame(0.0, index=close.index, columns=close.columns, dtype=float)
    valid_mask = None
    quadrant_series = regime_state["quadrant"].reindex(close.index)
    for frame in label_frames.values():
        frame_valid = frame.notna()
        valid_mask = frame_valid if valid_mask is None else (valid_mask | frame_valid)

    for dt in close.index:
        weights_for_date = resolve_horizon_weights(
            config,
            str(quadrant_series.loc[dt]) if dt in quadrant_series.index else None,
        )
        row = pd.Series(0.0, index=close.columns, dtype=float)
        for horizon, weight in weights_for_date.items():
            frame = label_frames.get(int(horizon))
            if frame is None or dt not in frame.index:
                continue
            row = row.add(frame.loc[dt].fillna(0.0) * float(weight), fill_value=0.0)
        combined.loc[dt] = row
    return combined.where(valid_mask)


def _build_validation_summary(
    *,
    feature_frames: dict[str, pd.DataFrame],
    market_features: dict[str, pd.Series],
    close: pd.DataFrame,
    benchmark_close: pd.Series,
    open_df: pd.DataFrame,
    benchmark_open: pd.Series,
    filter_mask: pd.DataFrame,
    regime_state: pd.DataFrame,
    ml_cfg: MLAplhaConfig,
    min_observations: int,
    validation_retrain_every_days: int,
) -> dict[str, object]:
    validation_cfg = MLAplhaConfig(**asdict(ml_cfg))
    validation_cfg.retrain_every_days = int(validation_retrain_every_days)

    combined_score, training_log_df, per_horizon_scores = rolling_ml_scores_multi_detail(
        feature_frames=feature_frames,
        market_features=market_features,
        close=close,
        benchmark_close=benchmark_close,
        open_df=open_df,
        benchmark_open=benchmark_open,
        filter_mask=filter_mask,
        regime_state=regime_state,
        config=validation_cfg,
    )

    label_frames: dict[int, pd.DataFrame] = {}
    for horizon in validation_cfg.target_horizons:
        label_frames[int(horizon)] = build_ml_target(
            close,
            benchmark_close,
            int(horizon),
            execution_mode=validation_cfg.execution_mode,
            open_df=open_df,
            benchmark_open=benchmark_open,
        )

    combined_label = _combine_per_horizon_labels(label_frames, close, regime_state, validation_cfg)
    windows = {
        "full": None,
        "recent_252d": 252,
        "recent_126d": 126,
        "recent_63d": 63,
    }

    combined_ic = _compute_rank_ic_frame(combined_score, combined_label, min_observations=min_observations)
    combined_summary = {
        name: _summarize_rank_ic_frame(_slice_rank_ic_window(combined_ic, trading_days))
        for name, trading_days in windows.items()
    }

    per_horizon_summary: dict[str, dict[str, object]] = {}
    for horizon, score_df in per_horizon_scores.items():
        ic_frame = _compute_rank_ic_frame(score_df, label_frames[int(horizon)], min_observations=min_observations)
        per_horizon_summary[f"h{int(horizon)}"] = {
            name: _summarize_rank_ic_frame(_slice_rank_ic_window(ic_frame, trading_days))
            for name, trading_days in windows.items()
        }

    return {
        "method": "rolling_rank_ic",
        "validation_retrain_every_days": int(validation_cfg.retrain_every_days),
        "min_observations": int(min_observations),
        "training_block_count": int(len(training_log_df)),
        "combined": combined_summary,
        "per_horizon": per_horizon_summary,
    }


def main():
    args = parse_args()

    cfg = ResearchConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        universe_scope=args.universe_scope,
        benchmark=args.benchmark,
        execution_mode="next_open",
        weighting_method="score",
        score_threshold=args.score_threshold,
        min_adv20=args.min_adv20,
        min_price=args.min_price,
        max_price=args.max_price,
        enable_market_regime_filter=not args.no_market_regime_filter,
        regime_ma_window=args.regime_ma_window,
        regime_vol_window=args.regime_vol_window,
        regime_max_annual_vol=args.regime_max_annual_vol,
        regime_allowed_quadrants=parse_csv_list(args.regime_quadrants),
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
        retrain_every_days=1,
        min_train_dates=args.ml_min_train_dates,
        max_samples_per_day=args.ml_max_samples_per_day,
        max_train_rows=args.ml_max_train_rows,
        random_seed=args.ml_random_seed,
        model_family=args.ml_model_family,
        lgbm_n_estimators=args.lgbm_n_estimators,
        ensemble_ml_weight=args.ensemble_ml_weight,
        ensemble_none_weight=args.ensemble_none_weight,
        ensemble_v2_weight=args.ensemble_v2_weight,
        state_ensemble_weights=parse_state_ensemble_weights(args.ensemble_state_weights),
        train_regime_only=cfg.enable_market_regime_filter,
        execution_mode=cfg.execution_mode,
    )

    if args.data_source == "tq":
        if not cfg.universe and cfg.universe_scope == "all_a":
            print("[1/6] 正在从 TQ 加载全A股票池...")
            cfg.universe = load_universe_from_tq(cfg.universe_scope)
        elif not cfg.universe:
            raise ValueError("TQ 模式下，未指定 --stocks 时目前仅支持 --universe-scope all_a。")
    elif not args.csv_folder:
        raise ValueError("CSV 模式需要提供 --csv-folder。")

    history_window = resolve_history_window(
        cfg=cfg,
        ml_cfg=ml_cfg,
        requested_start_date=args.start_date,
        end_date=args.end_date,
        mode="train",
        auto_trim_history=not args.no_auto_trim_history,
    )
    print(
        f"[2/6] 训练历史窗口: {history_window.effective_start_date} -> "
        f"{history_window.end_date or 'latest'} | required_trading_days={history_window.required_trading_days}"
    )
    print(f"[3/6] 正在准备训练数据，股票数: {len(cfg.universe)}，基准: {cfg.benchmark}")
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
        f"[4/6] raw cache: {'hit' if raw_cache_meta['cache_hit'] else 'build'} | "
        f"{raw_cache_meta['cache_path']}"
    )
    prepared_bundle, prepared_cache_meta = build_prepared_bundle_with_cache(
        raw_df_dict=raw_df_dict,
        raw_cache_key=raw_cache_meta["cache_key"],
        cfg=cfg,
        enhanced_profile=args.enhanced_profile,
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
    )
    print(
        f"[5/6] factor cache: {'hit' if prepared_cache_meta['cache_hit'] else 'build'} | "
        f"{prepared_cache_meta['cache_path']}"
    )

    df_dict = prepared_bundle["df_dict"]
    benchmark_close = prepared_bundle["benchmark_close"]
    benchmark_open = prepared_bundle["benchmark_open"]
    factor_bundle = prepared_bundle["factor_bundle"]
    regime_state = prepared_bundle["regime_state"]
    score_none = prepared_bundle["score_none"]
    score_enhanced = prepared_bundle["score_v2"]
    filter_mask = prepared_bundle["filter_mask"]
    feature_frames = prepared_bundle["feature_frames"]
    market_features = prepared_bundle["market_features"]

    print("[6/6] 正在训练点时模型并导出产物...")
    latest_date = factor_bundle["raw_inputs"]["Close"].index.max()
    models, train_summary = train_point_in_time_model_bundle(
        feature_frames=feature_frames,
        market_features=market_features,
        close=factor_bundle["raw_inputs"]["Close"],
        benchmark_close=benchmark_close,
        open_df=factor_bundle["raw_inputs"]["Open"],
        benchmark_open=benchmark_open,
        filter_mask=filter_mask,
        regime_state=regime_state,
        config=ml_cfg,
        as_of_date=latest_date,
    )

    validation_summary: dict[str, object] = {}
    if args.skip_validation_summary:
        print("[6/6] 跳过默认模型验证摘要生成。")
    else:
        print("[6/6] 正在生成默认模型滚动验证摘要...")
        validation_summary = _build_validation_summary(
            feature_frames=feature_frames,
            market_features=market_features,
            close=factor_bundle["raw_inputs"]["Close"],
            benchmark_close=benchmark_close,
            open_df=factor_bundle["raw_inputs"]["Open"],
            benchmark_open=benchmark_open,
            filter_mask=filter_mask,
            regime_state=regime_state,
            ml_cfg=ml_cfg,
            min_observations=args.validation_min_observations,
            validation_retrain_every_days=args.validation_retrain_every_days,
        )
    feature_names = list(feature_frames.keys()) + list(market_features.keys())

    artifact_path = save_ml_artifact(
        path=args.artifact_path,
        models=models,
        feature_names=feature_names,
        ml_config=ml_cfg,
        train_summary=train_summary,
    )

    meta = {
        "artifact_path": str(artifact_path),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latest_data_date": str(latest_date.date()),
        "signal_date": str(latest_date.date()),
        "execution_date": str(get_next_trading_date(latest_date) or ""),
        "benchmark": cfg.benchmark,
        "universe_scope": cfg.universe_scope,
        "universe_size": int(len(df_dict["Close"].columns)),
        "execution_mode": cfg.execution_mode,
        "regime_ma_window": cfg.regime_ma_window,
        "regime_vol_window": cfg.regime_vol_window,
        "regime_max_annual_vol": cfg.regime_max_annual_vol,
        "regime_allowed_quadrants": list(cfg.regime_allowed_quadrants),
        "history_window": history_window_to_dict(history_window),
        "cache": {
            "raw": raw_cache_meta,
            "prepared": prepared_cache_meta,
        },
        "ml_config": {
            "target_horizon": ml_cfg.target_horizon,
            "target_horizons": list(ml_cfg.target_horizons),
            "target_horizon_weights": ml_cfg.target_horizon_weights or {},
            "state_horizon_weights": ml_cfg.state_horizon_weights or {},
            "train_window_days": ml_cfg.train_window_days,
            "min_train_dates": ml_cfg.min_train_dates,
            "max_samples_per_day": ml_cfg.max_samples_per_day,
            "max_train_rows": ml_cfg.max_train_rows,
            "random_seed": ml_cfg.random_seed,
            "model_family": ml_cfg.model_family,
            "lgbm_n_estimators": ml_cfg.lgbm_n_estimators,
            "ensemble_ml_weight": ml_cfg.ensemble_ml_weight,
            "ensemble_none_weight": ml_cfg.ensemble_none_weight,
            "ensemble_v2_weight": ml_cfg.ensemble_v2_weight,
            "state_ensemble_weights": ml_cfg.state_ensemble_weights or {},
            "enhanced_profile": args.enhanced_profile,
        },
        "train_summary": train_summary,
        "validation_summary": validation_summary,
    }
    meta_path = Path(args.artifact_meta_path) if args.artifact_meta_path else artifact_path.with_suffix(".json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("模型产物已写入。")
    print(f"artifact: {artifact_path}")
    print(f"meta: {meta_path}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def main_with_progress():
    args = parse_args()

    cfg = ResearchConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        universe_scope=args.universe_scope,
        benchmark=args.benchmark,
        execution_mode="next_open",
        weighting_method="score",
        score_threshold=args.score_threshold,
        min_adv20=args.min_adv20,
        min_price=args.min_price,
        max_price=args.max_price,
        enable_market_regime_filter=not args.no_market_regime_filter,
        regime_ma_window=args.regime_ma_window,
        regime_vol_window=args.regime_vol_window,
        regime_max_annual_vol=args.regime_max_annual_vol,
        regime_allowed_quadrants=parse_csv_list(args.regime_quadrants),
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
        retrain_every_days=1,
        min_train_dates=args.ml_min_train_dates,
        max_samples_per_day=args.ml_max_samples_per_day,
        max_train_rows=args.ml_max_train_rows,
        random_seed=args.ml_random_seed,
        model_family=args.ml_model_family,
        lgbm_n_estimators=args.lgbm_n_estimators,
        ensemble_ml_weight=args.ensemble_ml_weight,
        ensemble_none_weight=args.ensemble_none_weight,
        ensemble_v2_weight=args.ensemble_v2_weight,
        state_ensemble_weights=parse_state_ensemble_weights(args.ensemble_state_weights),
        train_regime_only=cfg.enable_market_regime_filter,
        execution_mode=cfg.execution_mode,
    )

    with StageProgress(total=6, label="模型训练流程") as progress:
        with progress.stage("准备研究宇宙", f"source={args.data_source}"):
            if args.data_source == "tq":
                if not cfg.universe and cfg.universe_scope == "all_a":
                    cfg.universe = load_universe_from_tq(cfg.universe_scope)
                elif not cfg.universe:
                    raise ValueError("TQ mode without --stocks currently requires --universe-scope all_a.")
            elif not args.csv_folder:
                raise ValueError("CSV mode requires --csv-folder.")

        with progress.stage("解析训练历史窗口", args.start_date):
            history_window = resolve_history_window(
                cfg=cfg,
                ml_cfg=ml_cfg,
                requested_start_date=args.start_date,
                end_date=args.end_date,
                mode="train",
                auto_trim_history=not args.no_auto_trim_history,
            )
            progress.log(
                f"train history: {history_window.effective_start_date} -> "
                f"{history_window.end_date or 'latest'} | required_trading_days={history_window.required_trading_days}"
            )

        with progress.stage("读取训练行情数据", f"stocks={len(cfg.universe)} benchmark={cfg.benchmark}"):
            raw_df_dict, raw_cache_meta = load_raw_data_with_cache(
                data_source=args.data_source,
                csv_folder=args.csv_folder,
                universe=cfg.universe,
                benchmark=cfg.benchmark,
                history_window=history_window,
                use_cache=not args.no_cache,
                refresh_cache=args.refresh_cache,
                progress_desc="读取训练股票数据",
                progress_position=1,
            )
            progress.log(
                f"raw cache: {'hit' if raw_cache_meta['cache_hit'] else 'build'} | {raw_cache_meta['cache_path']}"
            )

        with progress.stage("构建特征与缓存", args.enhanced_profile):
            prepared_bundle, prepared_cache_meta = build_prepared_bundle_with_cache(
                raw_df_dict=raw_df_dict,
                raw_cache_key=raw_cache_meta["cache_key"],
                cfg=cfg,
                enhanced_profile=args.enhanced_profile,
                use_cache=not args.no_cache,
                refresh_cache=args.refresh_cache,
            )
            progress.log(
                f"factor cache: {'hit' if prepared_cache_meta['cache_hit'] else 'build'} | "
                f"{prepared_cache_meta['cache_path']}"
            )

        df_dict = prepared_bundle["df_dict"]
        benchmark_close = prepared_bundle["benchmark_close"]
        benchmark_open = prepared_bundle["benchmark_open"]
        factor_bundle = prepared_bundle["factor_bundle"]
        regime_state = prepared_bundle["regime_state"]
        filter_mask = prepared_bundle["filter_mask"]
        feature_frames = prepared_bundle["feature_frames"]
        market_features = prepared_bundle["market_features"]

        with progress.stage("训练点时模型", ml_cfg.model_family):
            latest_date = factor_bundle["raw_inputs"]["Close"].index.max()
            models, train_summary = train_point_in_time_model_bundle(
                feature_frames=feature_frames,
                market_features=market_features,
                close=factor_bundle["raw_inputs"]["Close"],
                benchmark_close=benchmark_close,
                open_df=factor_bundle["raw_inputs"]["Open"],
                benchmark_open=benchmark_open,
                filter_mask=filter_mask,
                regime_state=regime_state,
                config=ml_cfg,
                as_of_date=latest_date,
            )

        validation_summary: dict[str, object] = {}
        with progress.stage("写出模型产物", "artifact/meta"):
            if args.skip_validation_summary:
                progress.log("skip validation summary")
            else:
                progress.log("building default rolling validation summary...")
                validation_summary = _build_validation_summary(
                    feature_frames=feature_frames,
                    market_features=market_features,
                    close=factor_bundle["raw_inputs"]["Close"],
                    benchmark_close=benchmark_close,
                    open_df=factor_bundle["raw_inputs"]["Open"],
                    benchmark_open=benchmark_open,
                    filter_mask=filter_mask,
                    regime_state=regime_state,
                    ml_cfg=ml_cfg,
                    min_observations=args.validation_min_observations,
                    validation_retrain_every_days=args.validation_retrain_every_days,
                )

            feature_names = list(feature_frames.keys()) + list(market_features.keys())
            artifact_path = save_ml_artifact(
                path=args.artifact_path,
                models=models,
                feature_names=feature_names,
                ml_config=ml_cfg,
                train_summary=train_summary,
            )

            meta = {
                "artifact_path": str(artifact_path),
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "latest_data_date": str(latest_date.date()),
                "signal_date": str(latest_date.date()),
                "execution_date": str(get_next_trading_date(latest_date) or ""),
                "benchmark": cfg.benchmark,
                "universe_scope": cfg.universe_scope,
                "universe_size": int(len(df_dict["Close"].columns)),
                "execution_mode": cfg.execution_mode,
                "regime_ma_window": cfg.regime_ma_window,
                "regime_vol_window": cfg.regime_vol_window,
                "regime_max_annual_vol": cfg.regime_max_annual_vol,
                "regime_allowed_quadrants": list(cfg.regime_allowed_quadrants),
                "history_window": history_window_to_dict(history_window),
                "cache": {
                    "raw": raw_cache_meta,
                    "prepared": prepared_cache_meta,
                },
                "ml_config": {
                    "target_horizon": ml_cfg.target_horizon,
                    "target_horizons": list(ml_cfg.target_horizons),
                    "target_horizon_weights": ml_cfg.target_horizon_weights or {},
                    "state_horizon_weights": ml_cfg.state_horizon_weights or {},
                    "train_window_days": ml_cfg.train_window_days,
                    "min_train_dates": ml_cfg.min_train_dates,
                    "max_samples_per_day": ml_cfg.max_samples_per_day,
                    "max_train_rows": ml_cfg.max_train_rows,
                    "random_seed": ml_cfg.random_seed,
                    "model_family": ml_cfg.model_family,
                    "lgbm_n_estimators": ml_cfg.lgbm_n_estimators,
                    "ensemble_ml_weight": ml_cfg.ensemble_ml_weight,
                    "ensemble_none_weight": ml_cfg.ensemble_none_weight,
                    "ensemble_v2_weight": ml_cfg.ensemble_v2_weight,
                    "state_ensemble_weights": ml_cfg.state_ensemble_weights or {},
                    "enhanced_profile": args.enhanced_profile,
                },
                "train_summary": train_summary,
                "validation_summary": validation_summary,
            }
            meta_path = Path(args.artifact_meta_path) if args.artifact_meta_path else artifact_path.with_suffix(".json")
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("模型产物已写入。")
    print(f"artifact: {artifact_path}")
    print(f"meta: {meta_path}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Train and export point-in-time trade model artifact")
    parser.add_argument("--data-source", choices=["tq", "csv"], default="tq")
    parser.add_argument("--csv-folder", default=None)
    parser.add_argument("--stocks", default=None)
    parser.add_argument("--stocks-file", default=None, help="Path to txt/csv file containing stock codes.")
    parser.add_argument("--start-date", default="20210101")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--universe-scope", default="all_a")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--enhanced-profile", default="up_low_breakout_v2")
    parser.add_argument("--artifact-path", default="daily_research/execution/models/latest_ml_model.joblib")
    parser.add_argument("--artifact-meta-path", default="")
    parser.add_argument("--min-adv20", type=float, default=50_000.0)
    parser.add_argument("--min-price", type=float, default=2.0)
    parser.add_argument("--max-price", type=float, default=300.0)
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
        help="State selector used by state-aware horizon blending. Legacy state profiles currently require quadrant.",
    )
    parser.add_argument("--regime-quadrants", default="trend_up_low_vol,trend_up_high_vol")
    parser.add_argument("--ml-target-horizon", type=int, default=20)
    parser.add_argument("--ml-target-horizons", default="5,10,20")
    parser.add_argument("--ml-horizon-weights", default="5:0.2,10:0.3,20:0.5")
    parser.add_argument(
        "--ml-state-horizon-profiles",
        default="",
        help="Optional per-state horizon weights, e.g. trend_up_low_vol=5:0.15,10:0.25,20:0.60;trend_up_high_vol=5:0.30,10:0.35,20:0.35",
    )
    parser.add_argument("--ml-train-window-days", type=int, default=504)
    parser.add_argument("--ml-min-train-dates", type=int, default=120)
    parser.add_argument("--ml-max-samples-per-day", type=int, default=600)
    parser.add_argument("--ml-max-train-rows", type=int, default=200000)
    parser.add_argument("--ml-random-seed", type=int, default=7)
    parser.add_argument("--ml-model-family", choices=["histgb", "etr", "lgbm"], default="histgb")
    parser.add_argument("--lgbm-n-estimators", type=int, default=260)
    parser.add_argument("--ensemble-ml-weight", type=float, default=0.70)
    parser.add_argument("--ensemble-none-weight", type=float, default=0.20)
    parser.add_argument("--ensemble-v2-weight", type=float, default=0.10)
    parser.add_argument("--ensemble-state-weights", default="")
    parser.add_argument("--skip-validation-summary", action="store_true")
    parser.add_argument("--validation-retrain-every-days", type=int, default=21)
    parser.add_argument("--validation-min-observations", type=int, default=20)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--no-auto-trim-history", action="store_true")
    return parser.parse_args()


def _combine_per_horizon_labels(
    label_frames: dict[int, pd.DataFrame],
    close: pd.DataFrame,
    regime_state: pd.DataFrame,
    config: MLAplhaConfig,
    state_label_series: pd.Series | None = None,
) -> pd.DataFrame:
    if not label_frames:
        return pd.DataFrame(index=close.index, columns=close.columns, dtype=float)

    combined = pd.DataFrame(0.0, index=close.index, columns=close.columns, dtype=float)
    valid_mask = None
    regime_labels = (
        state_label_series.reindex(close.index)
        if state_label_series is not None
        else regime_state["quadrant"].reindex(close.index)
    )
    for frame in label_frames.values():
        frame_valid = frame.notna()
        valid_mask = frame_valid if valid_mask is None else (valid_mask | frame_valid)

    for dt in close.index:
        weights_for_date = resolve_horizon_weights(
            config,
            str(regime_labels.loc[dt]) if dt in regime_labels.index else None,
        )
        row = pd.Series(0.0, index=close.columns, dtype=float)
        for horizon, weight in weights_for_date.items():
            frame = label_frames.get(int(horizon))
            if frame is None or dt not in frame.index:
                continue
            row = row.add(frame.loc[dt].fillna(0.0) * float(weight), fill_value=0.0)
        combined.loc[dt] = row
    return combined.where(valid_mask)


def _build_validation_summary(
    *,
    feature_frames: dict[str, pd.DataFrame],
    market_features: dict[str, pd.Series],
    close: pd.DataFrame,
    benchmark_close: pd.Series,
    open_df: pd.DataFrame,
    benchmark_open: pd.Series,
    filter_mask: pd.DataFrame,
    regime_state: pd.DataFrame,
    ml_cfg: MLAplhaConfig,
    state_label_series: pd.Series | None,
    min_observations: int,
    validation_retrain_every_days: int,
) -> dict[str, object]:
    validation_cfg = MLAplhaConfig(**asdict(ml_cfg))
    validation_cfg.retrain_every_days = int(validation_retrain_every_days)

    combined_score, training_log_df, per_horizon_scores = rolling_ml_scores_multi_detail(
        feature_frames=feature_frames,
        market_features=market_features,
        close=close,
        benchmark_close=benchmark_close,
        open_df=open_df,
        benchmark_open=benchmark_open,
        filter_mask=filter_mask,
        regime_state=regime_state,
        config=validation_cfg,
        state_label_series=state_label_series,
    )

    label_frames: dict[int, pd.DataFrame] = {}
    for horizon in validation_cfg.target_horizons:
        label_frames[int(horizon)] = build_ml_target(
            close,
            benchmark_close,
            int(horizon),
            execution_mode=validation_cfg.execution_mode,
            open_df=open_df,
            benchmark_open=benchmark_open,
        )

    combined_label = _combine_per_horizon_labels(
        label_frames,
        close,
        regime_state,
        validation_cfg,
        state_label_series=state_label_series,
    )
    windows = {
        "full": None,
        "recent_252d": 252,
        "recent_126d": 126,
        "recent_63d": 63,
    }

    combined_ic = _compute_rank_ic_frame(combined_score, combined_label, min_observations=min_observations)
    combined_summary = {
        name: _summarize_rank_ic_frame(_slice_rank_ic_window(combined_ic, trading_days))
        for name, trading_days in windows.items()
    }

    per_horizon_summary: dict[str, dict[str, object]] = {}
    for horizon, score_df in per_horizon_scores.items():
        ic_frame = _compute_rank_ic_frame(score_df, label_frames[int(horizon)], min_observations=min_observations)
        per_horizon_summary[f"h{int(horizon)}"] = {
            name: _summarize_rank_ic_frame(_slice_rank_ic_window(ic_frame, trading_days))
            for name, trading_days in windows.items()
        }

    return {
        "method": "rolling_rank_ic",
        "validation_retrain_every_days": int(validation_cfg.retrain_every_days),
        "min_observations": int(min_observations),
        "training_block_count": int(len(training_log_df)),
        "combined": combined_summary,
        "per_horizon": per_horizon_summary,
    }


def _build_cfg_and_ml_cfg(args: argparse.Namespace) -> tuple[ResearchConfig, MLAplhaConfig]:
    validate_state_profile_selector(args.enhanced_profile, args.regime_state_selector)
    cfg = ResearchConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        universe_scope=args.universe_scope,
        benchmark=args.benchmark,
        execution_mode="next_open",
        weighting_method="score",
        score_threshold=args.score_threshold,
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
        retrain_every_days=1,
        min_train_dates=args.ml_min_train_dates,
        max_samples_per_day=args.ml_max_samples_per_day,
        max_train_rows=args.ml_max_train_rows,
        random_seed=args.ml_random_seed,
        model_family=args.ml_model_family,
        lgbm_n_estimators=args.lgbm_n_estimators,
        ensemble_ml_weight=args.ensemble_ml_weight,
        ensemble_none_weight=args.ensemble_none_weight,
        ensemble_v2_weight=args.ensemble_v2_weight,
        state_ensemble_weights=parse_state_ensemble_weights(args.ensemble_state_weights),
        train_regime_only=cfg.enable_market_regime_filter,
        execution_mode=cfg.execution_mode,
    )
    return cfg, ml_cfg


def _build_meta_payload(
    *,
    artifact_path: Path,
    latest_date: pd.Timestamp,
    cfg: ResearchConfig,
    ml_cfg: MLAplhaConfig,
    df_dict: dict[str, pd.DataFrame],
    history_window,
    raw_cache_meta: dict[str, object],
    prepared_cache_meta: dict[str, object],
    train_summary: dict[str, object],
    validation_summary: dict[str, object],
    enhanced_profile: str,
) -> dict[str, object]:
    return {
        "artifact_path": str(artifact_path),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "latest_data_date": str(latest_date.date()),
        "signal_date": str(latest_date.date()),
        "execution_date": str(get_next_trading_date(latest_date) or ""),
        "benchmark": cfg.benchmark,
        "universe_scope": cfg.universe_scope,
        "universe_size": int(len(df_dict["Close"].columns)),
        "execution_mode": cfg.execution_mode,
        "regime_ma_window": cfg.regime_ma_window,
        "regime_vol_window": cfg.regime_vol_window,
        "regime_max_annual_vol": cfg.regime_max_annual_vol,
        "regime_trend_flat_band": cfg.regime_trend_flat_band,
        "regime_vol_transition_band": cfg.regime_vol_transition_band,
        "regime_state_selector": cfg.regime_state_selector,
        "regime_allowed_quadrants": list(cfg.regime_allowed_quadrants),
        "history_window": history_window_to_dict(history_window),
        "cache": {
            "raw": raw_cache_meta,
            "prepared": prepared_cache_meta,
        },
        "ml_config": {
            "target_horizon": ml_cfg.target_horizon,
            "target_horizons": list(ml_cfg.target_horizons),
            "target_horizon_weights": ml_cfg.target_horizon_weights or {},
            "state_horizon_weights": ml_cfg.state_horizon_weights or {},
            "train_window_days": ml_cfg.train_window_days,
            "min_train_dates": ml_cfg.min_train_dates,
            "max_samples_per_day": ml_cfg.max_samples_per_day,
            "max_train_rows": ml_cfg.max_train_rows,
            "random_seed": ml_cfg.random_seed,
            "model_family": ml_cfg.model_family,
            "lgbm_n_estimators": ml_cfg.lgbm_n_estimators,
            "ensemble_ml_weight": ml_cfg.ensemble_ml_weight,
            "ensemble_none_weight": ml_cfg.ensemble_none_weight,
            "ensemble_v2_weight": ml_cfg.ensemble_v2_weight,
            "state_ensemble_weights": ml_cfg.state_ensemble_weights or {},
            "enhanced_profile": enhanced_profile,
            "regime_state_selector": cfg.regime_state_selector,
        },
        "train_summary": train_summary,
        "validation_summary": validation_summary,
    }


def main():
    args = parse_args()
    cfg, ml_cfg = _build_cfg_and_ml_cfg(args)

    if args.data_source == "tq":
        if not cfg.universe and cfg.universe_scope == "all_a":
            print("[1/6] Loading all-A universe from TQ...")
            cfg.universe = load_universe_from_tq(cfg.universe_scope)
        elif not cfg.universe:
            raise ValueError("TQ mode without --stocks currently requires --universe-scope all_a.")
    elif not args.csv_folder:
        raise ValueError("CSV mode requires --csv-folder.")

    history_window = resolve_history_window(
        cfg=cfg,
        ml_cfg=ml_cfg,
        requested_start_date=args.start_date,
        end_date=args.end_date,
        mode="train",
        auto_trim_history=not args.no_auto_trim_history,
    )
    print(
        f"[2/6] Training history window: {history_window.effective_start_date} -> "
        f"{history_window.end_date or 'latest'} | required_trading_days={history_window.required_trading_days}"
    )
    print(f"[3/6] Preparing training data. stocks={len(cfg.universe)} benchmark={cfg.benchmark}")
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
        f"[4/6] raw cache: {'hit' if raw_cache_meta['cache_hit'] else 'build'} | "
        f"{raw_cache_meta['cache_path']}"
    )
    prepared_bundle, prepared_cache_meta = build_prepared_bundle_with_cache(
        raw_df_dict=raw_df_dict,
        raw_cache_key=raw_cache_meta["cache_key"],
        cfg=cfg,
        enhanced_profile=args.enhanced_profile,
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
    )
    print(
        f"[5/6] factor cache: {'hit' if prepared_cache_meta['cache_hit'] else 'build'} | "
        f"{prepared_cache_meta['cache_path']}"
    )

    df_dict = prepared_bundle["df_dict"]
    benchmark_close = prepared_bundle["benchmark_close"]
    benchmark_open = prepared_bundle["benchmark_open"]
    factor_bundle = prepared_bundle["factor_bundle"]
    regime_state = prepared_bundle["regime_state"]
    state_label_series = resolve_regime_label_series(regime_state, cfg.regime_state_selector)
    filter_mask = prepared_bundle["filter_mask"]
    feature_frames = prepared_bundle["feature_frames"]
    market_features = prepared_bundle["market_features"]

    print("[6/6] Training point-in-time model bundle and exporting artifact...")
    latest_date = factor_bundle["raw_inputs"]["Close"].index.max()
    models, train_summary = train_point_in_time_model_bundle(
        feature_frames=feature_frames,
        market_features=market_features,
        close=factor_bundle["raw_inputs"]["Close"],
        benchmark_close=benchmark_close,
        open_df=factor_bundle["raw_inputs"]["Open"],
        benchmark_open=benchmark_open,
        filter_mask=filter_mask,
        regime_state=regime_state,
        config=ml_cfg,
        as_of_date=latest_date,
    )

    validation_summary: dict[str, object] = {}
    if args.skip_validation_summary:
        print("[6/6] Skipping rolling validation summary.")
    else:
        print("[6/6] Building default rolling validation summary...")
        validation_summary = _build_validation_summary(
            feature_frames=feature_frames,
            market_features=market_features,
            close=factor_bundle["raw_inputs"]["Close"],
            benchmark_close=benchmark_close,
            open_df=factor_bundle["raw_inputs"]["Open"],
            benchmark_open=benchmark_open,
            filter_mask=filter_mask,
            regime_state=regime_state,
            ml_cfg=ml_cfg,
            state_label_series=state_label_series,
            min_observations=args.validation_min_observations,
            validation_retrain_every_days=args.validation_retrain_every_days,
        )
    feature_names = list(feature_frames.keys()) + list(market_features.keys())

    artifact_path = save_ml_artifact(
        path=args.artifact_path,
        models=models,
        feature_names=feature_names,
        ml_config=ml_cfg,
        train_summary=train_summary,
    )

    meta = _build_meta_payload(
        artifact_path=artifact_path,
        latest_date=latest_date,
        cfg=cfg,
        ml_cfg=ml_cfg,
        df_dict=df_dict,
        history_window=history_window,
        raw_cache_meta=raw_cache_meta,
        prepared_cache_meta=prepared_cache_meta,
        train_summary=train_summary,
        validation_summary=validation_summary,
        enhanced_profile=args.enhanced_profile,
    )
    meta_path = Path(args.artifact_meta_path) if args.artifact_meta_path else artifact_path.with_suffix(".json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Model artifact written.")
    print(f"artifact: {artifact_path}")
    print(f"meta: {meta_path}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def main_with_progress():
    args = parse_args()
    cfg, ml_cfg = _build_cfg_and_ml_cfg(args)

    with StageProgress(total=6, label="Model training flow") as progress:
        with progress.stage("Prepare universe", f"source={args.data_source}"):
            if args.data_source == "tq":
                if not cfg.universe and cfg.universe_scope == "all_a":
                    cfg.universe = load_universe_from_tq(cfg.universe_scope)
                elif not cfg.universe:
                    raise ValueError("TQ mode without --stocks currently requires --universe-scope all_a.")
            elif not args.csv_folder:
                raise ValueError("CSV mode requires --csv-folder.")

        with progress.stage("Resolve history window", args.start_date):
            history_window = resolve_history_window(
                cfg=cfg,
                ml_cfg=ml_cfg,
                requested_start_date=args.start_date,
                end_date=args.end_date,
                mode="train",
                auto_trim_history=not args.no_auto_trim_history,
            )
            progress.log(
                f"train history: {history_window.effective_start_date} -> "
                f"{history_window.end_date or 'latest'} | required_trading_days={history_window.required_trading_days}"
            )

        with progress.stage("Load market data", f"stocks={len(cfg.universe)} benchmark={cfg.benchmark}"):
            raw_df_dict, raw_cache_meta = load_raw_data_with_cache(
                data_source=args.data_source,
                csv_folder=args.csv_folder,
                universe=cfg.universe,
                benchmark=cfg.benchmark,
                history_window=history_window,
                use_cache=not args.no_cache,
                refresh_cache=args.refresh_cache,
                progress_desc="Load training market data",
                progress_position=1,
            )
            progress.log(
                f"raw cache: {'hit' if raw_cache_meta['cache_hit'] else 'build'} | {raw_cache_meta['cache_path']}"
            )

        with progress.stage("Build features and cache", args.enhanced_profile):
            prepared_bundle, prepared_cache_meta = build_prepared_bundle_with_cache(
                raw_df_dict=raw_df_dict,
                raw_cache_key=raw_cache_meta["cache_key"],
                cfg=cfg,
                enhanced_profile=args.enhanced_profile,
                use_cache=not args.no_cache,
                refresh_cache=args.refresh_cache,
            )
            progress.log(
                f"factor cache: {'hit' if prepared_cache_meta['cache_hit'] else 'build'} | "
                f"{prepared_cache_meta['cache_path']}"
            )

        df_dict = prepared_bundle["df_dict"]
        benchmark_close = prepared_bundle["benchmark_close"]
        benchmark_open = prepared_bundle["benchmark_open"]
        factor_bundle = prepared_bundle["factor_bundle"]
        regime_state = prepared_bundle["regime_state"]
        state_label_series = resolve_regime_label_series(regime_state, cfg.regime_state_selector)
        filter_mask = prepared_bundle["filter_mask"]
        feature_frames = prepared_bundle["feature_frames"]
        market_features = prepared_bundle["market_features"]

        with progress.stage("Train point-in-time model", ml_cfg.model_family):
            latest_date = factor_bundle["raw_inputs"]["Close"].index.max()
            models, train_summary = train_point_in_time_model_bundle(
                feature_frames=feature_frames,
                market_features=market_features,
                close=factor_bundle["raw_inputs"]["Close"],
                benchmark_close=benchmark_close,
                open_df=factor_bundle["raw_inputs"]["Open"],
                benchmark_open=benchmark_open,
                filter_mask=filter_mask,
                regime_state=regime_state,
                config=ml_cfg,
                as_of_date=latest_date,
            )

        validation_summary: dict[str, object] = {}
        with progress.stage("Write artifacts", "artifact/meta"):
            if args.skip_validation_summary:
                progress.log("skip validation summary")
            else:
                progress.log("building default rolling validation summary...")
                validation_summary = _build_validation_summary(
                    feature_frames=feature_frames,
                    market_features=market_features,
                    close=factor_bundle["raw_inputs"]["Close"],
                    benchmark_close=benchmark_close,
                    open_df=factor_bundle["raw_inputs"]["Open"],
                    benchmark_open=benchmark_open,
                    filter_mask=filter_mask,
                    regime_state=regime_state,
                    ml_cfg=ml_cfg,
                    state_label_series=state_label_series,
                    min_observations=args.validation_min_observations,
                    validation_retrain_every_days=args.validation_retrain_every_days,
                )

            feature_names = list(feature_frames.keys()) + list(market_features.keys())
            artifact_path = save_ml_artifact(
                path=args.artifact_path,
                models=models,
                feature_names=feature_names,
                ml_config=ml_cfg,
                train_summary=train_summary,
            )

            meta = _build_meta_payload(
                artifact_path=artifact_path,
                latest_date=latest_date,
                cfg=cfg,
                ml_cfg=ml_cfg,
                df_dict=df_dict,
                history_window=history_window,
                raw_cache_meta=raw_cache_meta,
                prepared_cache_meta=prepared_cache_meta,
                train_summary=train_summary,
                validation_summary=validation_summary,
                enhanced_profile=args.enhanced_profile,
            )
            meta_path = Path(args.artifact_meta_path) if args.artifact_meta_path else artifact_path.with_suffix(".json")
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Model artifact written.")
    print(f"artifact: {artifact_path}")
    print(f"meta: {meta_path}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


main = main_with_progress


if __name__ == "__main__":
    main()
