from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from datetime import datetime

from daily_research.baseline.advanced_ml_runtime import (
    build_prepared_bundle_with_cache,
    history_window_to_dict,
    load_raw_data_with_cache,
    resolve_history_window,
)
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import (
    get_next_trading_date,
    load_universe_from_tq,
)
from daily_research.baseline.ml_alpha import (
    MLAplhaConfig,
    save_ml_artifact,
    train_point_in_time_model_bundle,
)


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
    parser.add_argument("--ensemble-ml-weight", type=float, default=0.70)
    parser.add_argument("--ensemble-none-weight", type=float, default=0.20)
    parser.add_argument("--ensemble-v2-weight", type=float, default=0.10)
    parser.add_argument("--ensemble-state-weights", default="")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--no-auto-trim-history", action="store_true")
    return parser.parse_args()


def _parse_stocks(raw: str | None):
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


def _parse_csv_list(raw: str | None):
    if not raw:
        return []
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str | None, fallback: int) -> tuple[int, ...]:
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
        regime_allowed_quadrants=_parse_csv_list(args.regime_quadrants),
    )

    stocks = _parse_stocks(args.stocks)
    file_stocks = _load_stocks_from_file(args.stocks_file)
    if stocks or file_stocks:
        stocks = list(dict.fromkeys(stocks + file_stocks))
    if stocks:
        cfg.universe = stocks

    ml_cfg = MLAplhaConfig(
        target_horizon=args.ml_target_horizon,
        target_horizons=_parse_int_list(args.ml_target_horizons, args.ml_target_horizon),
        target_horizon_weights=_parse_horizon_weights(args.ml_horizon_weights),
        state_horizon_weights=_parse_state_horizon_profiles(args.ml_state_horizon_profiles),
        enhanced_profile=args.enhanced_profile,
        train_window_days=args.ml_train_window_days,
        retrain_every_days=1,
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
            "ensemble_ml_weight": ml_cfg.ensemble_ml_weight,
            "ensemble_none_weight": ml_cfg.ensemble_none_weight,
            "ensemble_v2_weight": ml_cfg.ensemble_v2_weight,
            "state_ensemble_weights": ml_cfg.state_ensemble_weights or {},
            "enhanced_profile": args.enhanced_profile,
        },
        "train_summary": train_summary,
    }
    meta_path = Path(args.artifact_meta_path) if args.artifact_meta_path else artifact_path.with_suffix(".json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("模型产物已写入。")
    print(f"artifact: {artifact_path}")
    print(f"meta: {meta_path}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
