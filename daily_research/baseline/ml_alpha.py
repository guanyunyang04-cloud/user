from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - optional dependency
    LGBMRegressor = None


def _zscore_cs(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def _safe_ratio(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a.div(b.replace(0, np.nan))


@dataclass
class MLAplhaConfig:
    target_horizon: int = 20
    target_horizons: tuple[int, ...] = (20,)
    target_horizon_weights: Dict[int, float] | None = None
    state_horizon_weights: Dict[str, Dict[int, float]] | None = None
    enhanced_profile: str = "up_low_breakout_v2"
    train_window_days: int = 504
    retrain_every_days: int = 21
    min_train_dates: int = 120
    max_samples_per_day: int = 600
    max_train_rows: int = 200_000
    random_seed: int = 7
    model_family: str = "histgb"
    ensemble_ml_weight: float = 0.70
    ensemble_none_weight: float = 0.20
    ensemble_v2_weight: float = 0.10
    state_ensemble_weights: Dict[str, Dict[str, float]] | None = None
    train_regime_only: bool = True
    execution_mode: str = "close"


@dataclass
class MLArtifact:
    models: Dict[str, Any]
    feature_names: List[str]
    ml_config: Dict[str, Any]
    train_summary: Dict[str, Any]
    trained_at: str


def _normalize_horizons(config: MLAplhaConfig) -> tuple[int, ...]:
    horizons = tuple(int(h) for h in (config.target_horizons or (config.target_horizon,)) if int(h) > 0)
    if not horizons:
        return (int(config.target_horizon),)
    return tuple(dict.fromkeys(horizons))


def _zscore_series_cs(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    std = float(series.std())
    if std == 0 or np.isnan(std):
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (series - float(series.mean())) / std


def _normalize_horizon_weights(config: MLAplhaConfig) -> Dict[int, float]:
    horizons = _normalize_horizons(config)
    raw = dict(config.target_horizon_weights or {})
    return _normalize_horizon_weight_map(horizons, raw)


def _normalize_horizon_weight_map(horizons: tuple[int, ...], raw: Dict[int, float] | Dict[str, float]) -> Dict[int, float]:
    weights: Dict[int, float] = {}
    for horizon in horizons:
        raw_value = raw.get(horizon, raw.get(str(horizon), 1.0))
        weights[horizon] = float(raw_value)
    positive = {k: v for k, v in weights.items() if v > 0}
    if not positive:
        positive = {h: 1.0 for h in horizons}
    total = float(sum(positive.values()))
    return {k: v / total for k, v in positive.items()}


def resolve_horizon_weights(config: MLAplhaConfig, quadrant: str | None = None) -> Dict[int, float]:
    horizons = _normalize_horizons(config)
    if quadrant and config.state_horizon_weights:
        raw = config.state_horizon_weights.get(str(quadrant), {})
        if raw:
            return _normalize_horizon_weight_map(horizons, raw)
    return _normalize_horizon_weights(config)


def _normalize_ensemble_weight_map(raw: Dict[str, float] | None, fallback: Dict[str, float] | None = None) -> Dict[str, float]:
    weights = {
        "ml": float((raw or {}).get("ml", (fallback or {}).get("ml", 0.0))),
        "none": float((raw or {}).get("none", (fallback or {}).get("none", 0.0))),
        "v2": float((raw or {}).get("v2", (fallback or {}).get("v2", 0.0))),
    }
    positive = {k: v for k, v in weights.items() if v > 0}
    if not positive:
        positive = {"ml": 1.0}
    total = float(sum(positive.values()))
    return {key: positive.get(key, 0.0) / total for key in ("ml", "none", "v2")}


def resolve_ensemble_weights(config: MLAplhaConfig, quadrant: str | None = None) -> Dict[str, float]:
    fallback = _normalize_ensemble_weight_map(
        {
            "ml": float(config.ensemble_ml_weight),
            "none": float(config.ensemble_none_weight),
            "v2": float(config.ensemble_v2_weight),
        }
    )
    if quadrant and config.state_ensemble_weights:
        raw = config.state_ensemble_weights.get(str(quadrant), {})
        if raw:
            return _normalize_ensemble_weight_map(raw, fallback=fallback)
    return fallback


def build_ml_feature_bundle(
    factor_bundle: Dict[str, Dict[str, pd.DataFrame]],
    regime_state: pd.DataFrame,
    score_none: pd.DataFrame,
    score_v2: pd.DataFrame,
) -> tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
    raw_inputs = factor_bundle["raw_inputs"]
    close = raw_inputs["Close"]
    amount = raw_inputs["Amount"]
    volume = raw_inputs["Volume"]

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()
    adv20 = amount.rolling(20).mean()
    vol20 = volume.rolling(20).mean()

    feature_frames: Dict[str, pd.DataFrame] = {}
    excluded_signal_features = {"kama_gap", "kama_slope", "long_regime_flag", "mbuy_flag", "zjtp_flag", "hcw_flag"}
    for name, frame in factor_bundle["zscore_factors"].items():
        if name.endswith("_flag") or name in excluded_signal_features:
            continue
        feature_frames[f"z_{name}"] = frame

    feature_frames["z_score_none"] = _zscore_cs(score_none)
    feature_frames["z_score_v2"] = _zscore_cs(score_v2)
    feature_frames["adv20_rank"] = adv20.rank(axis=1, pct=True)
    feature_frames["price_rank"] = close.rank(axis=1, pct=True)
    feature_frames["ma20_gap"] = close / ema20 - 1.0
    feature_frames["ma60_gap"] = close / ema60 - 1.0
    feature_frames["volume_rank"] = vol20.rank(axis=1, pct=True)

    market_features: Dict[str, pd.Series] = {}
    market_features["benchmark_trend_gap"] = regime_state["benchmark_close"] / regime_state["benchmark_ma"] - 1.0
    market_features["benchmark_annual_vol"] = regime_state["benchmark_annual_vol"]
    market_features["is_up_low"] = regime_state["quadrant"].eq("trend_up_low_vol").astype(float)
    market_features["is_up_high"] = regime_state["quadrant"].eq("trend_up_high_vol").astype(float)
    market_features["is_down_low"] = regime_state["quadrant"].eq("trend_down_low_vol").astype(float)
    market_features["is_down_high"] = regime_state["quadrant"].eq("trend_down_high_vol").astype(float)
    market_features["regime_on"] = regime_state["regime_on"].astype(float)
    return feature_frames, market_features


def build_ml_target(
    close: pd.DataFrame,
    benchmark_close: pd.Series,
    horizon: int,
    execution_mode: str = "close",
    open_df: pd.DataFrame | None = None,
    benchmark_open: pd.Series | None = None,
) -> pd.DataFrame:
    mode = str(execution_mode or "close").lower()
    if mode == "next_open":
        if open_df is None or benchmark_open is None:
            raise ValueError("next_open target requires open_df and benchmark_open.")
        entry = open_df.shift(-1)
        exit_ = open_df.shift(-(horizon + 1))
        benchmark_entry = benchmark_open.shift(-1)
        benchmark_exit = benchmark_open.shift(-(horizon + 1))
        stock_fwd = exit_.div(entry).sub(1.0)
        benchmark_fwd = benchmark_exit.div(benchmark_entry).sub(1.0)
    else:
        stock_fwd = close.shift(-horizon).div(close).sub(1.0)
        benchmark_fwd = benchmark_close.shift(-horizon).div(benchmark_close).sub(1.0)
    return stock_fwd.sub(benchmark_fwd, axis=0)


def _sample_row_index(
    y_row: pd.Series,
    max_samples_per_day: int,
    rng: np.random.Generator,
) -> List[str]:
    valid = y_row.dropna().sort_values()
    if len(valid) <= max_samples_per_day:
        return list(valid.index)

    tail_n = max(1, max_samples_per_day // 3)
    top_idx = list(valid.tail(tail_n).index)
    bottom_idx = list(valid.head(tail_n).index)
    chosen = set(top_idx + bottom_idx)
    remaining = [idx for idx in valid.index if idx not in chosen]
    middle_n = max_samples_per_day - len(chosen)
    if middle_n > 0 and remaining:
        pick_n = min(middle_n, len(remaining))
        sampled = rng.choice(np.array(remaining, dtype=object), size=pick_n, replace=False)
        chosen.update(sampled.tolist())
    return [idx for idx in valid.index if idx in chosen]


def _build_cross_section_frame(
    dt: pd.Timestamp,
    stocks: List[str],
    feature_frames: Dict[str, pd.DataFrame],
    market_features: Dict[str, pd.Series],
) -> pd.DataFrame:
    data = {name: frame.loc[dt, stocks].values for name, frame in feature_frames.items()}
    for name, series in market_features.items():
        data[name] = np.repeat(float(series.loc[dt]), len(stocks))
    return pd.DataFrame(data, index=stocks)


def _fit_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: np.ndarray,
    random_seed: int,
    model_family: str,
) -> Any:
    def _make_histgb() -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=6,
            max_iter=180,
            min_samples_leaf=120,
            l2_regularization=0.10,
            random_state=random_seed,
        )

    if model_family == "histgb":
        model: Any = _make_histgb()
    elif model_family == "etr":
        model = ExtraTreesRegressor(
            n_estimators=120,
            max_depth=10,
            min_samples_leaf=80,
            max_features=0.35,
            n_jobs=-1,
            random_state=random_seed,
        )
    elif model_family == "lgbm":
        if LGBMRegressor is None:
            raise ValueError("model_family=lgbm requires lightgbm to be installed.")
        model = LGBMRegressor(
            objective="regression",
            learning_rate=0.04,
            n_estimators=260,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=120,
            subsample=0.80,
            colsample_bytree=0.60,
            reg_alpha=0.10,
            reg_lambda=0.20,
            n_jobs=-1,
            random_state=random_seed,
            verbosity=-1,
        )
    else:
        raise ValueError(f"Unsupported model_family: {model_family}")
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return model


def train_point_in_time_model(
    feature_frames: Dict[str, pd.DataFrame],
    market_features: Dict[str, pd.Series],
    label_df: pd.DataFrame,
    filter_mask: pd.DataFrame,
    regime_state: pd.DataFrame,
    config: MLAplhaConfig,
    as_of_date: pd.Timestamp | None = None,
) -> tuple[Any, dict]:
    dates = label_df.index
    if len(dates) == 0:
        raise ValueError("No dates available for ML training.")

    if as_of_date is None:
        as_of_date = dates[-1]
    as_of_date = pd.Timestamp(as_of_date)
    if as_of_date not in dates:
        raise ValueError(f"as_of_date {as_of_date} is not in label_df index.")

    as_of_idx = dates.get_loc(as_of_date)
    if as_of_idx <= 0:
        raise ValueError("Not enough history before as_of_date to train model.")

    train_end_idx = as_of_idx - 1
    train_start_idx = max(0, as_of_idx - int(config.train_window_days))
    train_dates = list(dates[train_start_idx : train_end_idx + 1])
    if len(train_dates) < int(config.min_train_dates):
        raise ValueError("Not enough train dates for point-in-time model.")

    rng = np.random.default_rng(config.random_seed)
    x_parts = []
    y_parts = []
    weight_parts = []
    train_rows = 0

    for pos, dt in enumerate(train_dates):
        valid = filter_mask.loc[dt].copy()
        if config.train_regime_only:
            valid &= bool(regime_state["regime_on"].loc[dt])
        valid &= label_df.loc[dt].notna()
        if not bool(valid.any()):
            continue

        sampled_stocks = _sample_row_index(label_df.loc[dt, valid], int(config.max_samples_per_day), rng)
        if not sampled_stocks:
            continue

        x_dt = _build_cross_section_frame(dt, sampled_stocks, feature_frames, market_features).replace([np.inf, -np.inf], np.nan)
        y_dt = label_df.loc[dt, sampled_stocks]
        valid_rows = x_dt.notna().all(axis=1) & y_dt.notna()
        if not bool(valid_rows.any()):
            continue

        x_dt = x_dt.loc[valid_rows]
        y_dt = y_dt.loc[valid_rows]
        recency = (pos + 1) / len(train_dates)
        sample_weight = np.repeat(0.25 + 0.75 * recency, len(x_dt))

        x_parts.append(x_dt)
        y_parts.append(y_dt)
        weight_parts.append(sample_weight)
        train_rows += len(x_dt)
        if train_rows >= int(config.max_train_rows):
            break

    if not x_parts:
        raise ValueError("No valid rows collected for point-in-time ML training.")

    x_train = pd.concat(x_parts, axis=0).iloc[: config.max_train_rows]
    y_train = pd.concat(y_parts, axis=0).iloc[: config.max_train_rows]
    sample_weight = np.concatenate(weight_parts)[: len(x_train)]
    model = _fit_model(x_train, y_train, sample_weight, config.random_seed, config.model_family)

    summary = {
        "train_start": str(train_dates[0].date()),
        "train_end": str(train_dates[-1].date()),
        "predict_date": str(as_of_date.date()),
        "train_rows": int(len(x_train)),
        "feature_count": int(x_train.shape[1]),
        "label_mean": float(y_train.mean()) if len(y_train) else np.nan,
        "label_std": float(y_train.std()) if len(y_train) > 1 else np.nan,
    }
    return model, summary


def train_point_in_time_model_bundle(
    feature_frames: Dict[str, pd.DataFrame],
    market_features: Dict[str, pd.Series],
    close: pd.DataFrame,
    benchmark_close: pd.Series,
    filter_mask: pd.DataFrame,
    regime_state: pd.DataFrame,
    config: MLAplhaConfig,
    as_of_date: pd.Timestamp | None = None,
    open_df: pd.DataFrame | None = None,
    benchmark_open: pd.Series | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    models: Dict[str, Any] = {}
    summaries: Dict[str, Any] = {}
    for horizon in _normalize_horizon_weights(config).keys():
        label_df = build_ml_target(
            close,
            benchmark_close,
            horizon,
            execution_mode=config.execution_mode,
            open_df=open_df,
            benchmark_open=benchmark_open,
        )
        horizon_cfg = MLAplhaConfig(**asdict(config))
        horizon_cfg.target_horizon = horizon
        horizon_cfg.target_horizons = (horizon,)
        model, summary = train_point_in_time_model(
            feature_frames=feature_frames,
            market_features=market_features,
            label_df=label_df,
            filter_mask=filter_mask,
            regime_state=regime_state,
            config=horizon_cfg,
            as_of_date=as_of_date,
        )
        key = f"h{horizon}"
        models[key] = model
        summaries[key] = summary
    return models, summaries


def predict_ml_scores_for_date(
    model: Any,
    feature_frames: Dict[str, pd.DataFrame],
    market_features: Dict[str, pd.Series],
    filter_mask: pd.DataFrame,
    dt: pd.Timestamp,
    feature_names: List[str] | None = None,
) -> pd.Series:
    valid = filter_mask.loc[dt].copy()
    if not bool(valid.any()):
        return pd.Series(dtype=float)

    pred_stocks = list(valid[valid].index)
    x_pred = _build_cross_section_frame(dt, pred_stocks, feature_frames, market_features).replace([np.inf, -np.inf], np.nan)
    if feature_names is not None:
        missing = [name for name in feature_names if name not in x_pred.columns]
        if missing:
            raise ValueError(f"Prediction features missing required columns: {missing}")
        x_pred = x_pred.reindex(columns=feature_names)
    valid_rows = x_pred.notna().all(axis=1)
    x_pred = x_pred.loc[valid_rows]
    if x_pred.empty:
        return pd.Series(dtype=float)
    return pd.Series(model.predict(x_pred), index=x_pred.index, dtype=float)


def predict_ml_scores_for_date_bundle(
    models: Dict[str, Any],
    feature_frames: Dict[str, pd.DataFrame],
    market_features: Dict[str, pd.Series],
    filter_mask: pd.DataFrame,
    dt: pd.Timestamp,
    feature_names: List[str] | None = None,
    horizon_weights: Dict[int, float] | None = None,
) -> tuple[pd.Series, Dict[str, pd.Series]]:
    if horizon_weights is None:
        horizon_weights = _normalize_horizon_weights(
            MLAplhaConfig(target_horizons=tuple(int(name[1:]) for name in models.keys() if name.startswith("h")))
        )
    per_horizon: Dict[str, pd.Series] = {}
    combined_parts = []
    for name, model in models.items():
        pred = predict_ml_scores_for_date(
            model=model,
            feature_frames=feature_frames,
            market_features=market_features,
            filter_mask=filter_mask,
            dt=dt,
            feature_names=feature_names,
        )
        per_horizon[name] = pred
        horizon = int(name[1:]) if name.startswith("h") and name[1:].isdigit() else None
        weight = horizon_weights.get(horizon, 1.0 / max(len(models), 1))
        combined_parts.append(_zscore_series_cs(pred) * weight)

    if not combined_parts:
        return pd.Series(dtype=float), per_horizon

    all_index = combined_parts[0].index
    combined = pd.Series(0.0, index=all_index, dtype=float)
    valid = pd.Series(False, index=all_index, dtype=bool)
    for part in combined_parts:
        aligned = part.reindex(all_index)
        combined = combined.add(aligned.fillna(0.0), fill_value=0.0)
        valid |= aligned.notna().fillna(False)
    combined = combined.where(valid)
    return combined, per_horizon


def save_ml_artifact(
    path: str | Path,
    models: Dict[str, Any],
    feature_names: List[str],
    ml_config: MLAplhaConfig,
    train_summary: Dict[str, Any],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = MLArtifact(
        models=models,
        feature_names=list(feature_names),
        ml_config=asdict(ml_config),
        train_summary=dict(train_summary),
        trained_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    joblib.dump(asdict(artifact), output_path)
    return output_path


def load_ml_artifact(path: str | Path) -> MLArtifact:
    payload = joblib.load(Path(path))
    models = payload.get("models")
    if models is None and "model" in payload:
        models = {"h20": payload["model"]}
    return MLArtifact(
        models=models,
        feature_names=list(payload["feature_names"]),
        ml_config=dict(payload["ml_config"]),
        train_summary=dict(payload["train_summary"]),
        trained_at=str(payload["trained_at"]),
    )


def rolling_ml_scores(
    feature_frames: Dict[str, pd.DataFrame],
    market_features: Dict[str, pd.Series],
    label_df: pd.DataFrame,
    filter_mask: pd.DataFrame,
    regime_state: pd.DataFrame,
    config: MLAplhaConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = label_df.index
    stocks = list(label_df.columns)
    ml_score = pd.DataFrame(np.nan, index=dates, columns=stocks)
    training_logs: List[Dict] = []
    rng = np.random.default_rng(config.random_seed)

    start_idx = max(int(config.train_window_days), int(config.min_train_dates))
    for block_start in range(start_idx, len(dates), int(config.retrain_every_days)):
        train_end_idx = block_start - 1
        train_start_idx = max(0, block_start - int(config.train_window_days))
        train_dates = list(dates[train_start_idx : train_end_idx + 1])
        if len(train_dates) < int(config.min_train_dates):
            continue

        x_parts = []
        y_parts = []
        weight_parts = []
        train_rows = 0

        for pos, dt in enumerate(train_dates):
            valid = filter_mask.loc[dt].copy()
            if config.train_regime_only:
                valid &= bool(regime_state["regime_on"].loc[dt])
            valid &= label_df.loc[dt].notna()
            if not bool(valid.any()):
                continue

            sampled_stocks = _sample_row_index(label_df.loc[dt, valid], int(config.max_samples_per_day), rng)
            if not sampled_stocks:
                continue
            x_dt = _build_cross_section_frame(dt, sampled_stocks, feature_frames, market_features).replace([np.inf, -np.inf], np.nan)
            y_dt = label_df.loc[dt, sampled_stocks]
            valid_rows = x_dt.notna().all(axis=1) & y_dt.notna()
            if not bool(valid_rows.any()):
                continue

            x_dt = x_dt.loc[valid_rows]
            y_dt = y_dt.loc[valid_rows]
            recency = (pos + 1) / len(train_dates)
            sample_weight = np.repeat(0.25 + 0.75 * recency, len(x_dt))

            x_parts.append(x_dt)
            y_parts.append(y_dt)
            weight_parts.append(sample_weight)
            train_rows += len(x_dt)
            if train_rows >= int(config.max_train_rows):
                break

        if not x_parts:
            continue

        x_train = pd.concat(x_parts, axis=0).iloc[: config.max_train_rows]
        y_train = pd.concat(y_parts, axis=0).iloc[: config.max_train_rows]
        sample_weight = np.concatenate(weight_parts)[: len(x_train)]
        model = _fit_model(x_train, y_train, sample_weight, config.random_seed, config.model_family)

        block_end = min(len(dates), block_start + int(config.retrain_every_days))
        predict_dates = list(dates[block_start:block_end])
        predicted_rows = 0
        for dt in predict_dates:
            valid = filter_mask.loc[dt].copy()
            if not bool(valid.any()):
                continue
            pred_stocks = list(valid[valid].index)
            x_pred = _build_cross_section_frame(dt, pred_stocks, feature_frames, market_features).replace([np.inf, -np.inf], np.nan)
            valid_rows = x_pred.notna().all(axis=1)
            x_pred = x_pred.loc[valid_rows]
            if x_pred.empty:
                continue
            pred = pd.Series(model.predict(x_pred), index=x_pred.index)
            ml_score.loc[dt, pred.index] = pred
            predicted_rows += len(pred)

        training_logs.append(
            {
                "train_start": str(train_dates[0].date()),
                "train_end": str(train_dates[-1].date()),
                "predict_start": str(predict_dates[0].date()) if predict_dates else "",
                "predict_end": str(predict_dates[-1].date()) if predict_dates else "",
                "train_rows": int(len(x_train)),
                "predict_rows": int(predicted_rows),
                "feature_count": int(x_train.shape[1]),
                "label_mean": float(y_train.mean()) if len(y_train) else np.nan,
                "label_std": float(y_train.std()) if len(y_train) > 1 else np.nan,
            }
        )

    return ml_score, pd.DataFrame(training_logs)


def rolling_ml_scores_multi(
    feature_frames: Dict[str, pd.DataFrame],
    market_features: Dict[str, pd.Series],
    close: pd.DataFrame,
    benchmark_close: pd.Series,
    filter_mask: pd.DataFrame,
    regime_state: pd.DataFrame,
    config: MLAplhaConfig,
    open_df: pd.DataFrame | None = None,
    benchmark_open: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined, training_log_df, _ = rolling_ml_scores_multi_detail(
        feature_frames=feature_frames,
        market_features=market_features,
        close=close,
        benchmark_close=benchmark_close,
        filter_mask=filter_mask,
        regime_state=regime_state,
        config=config,
        open_df=open_df,
        benchmark_open=benchmark_open,
    )
    return combined, training_log_df


def combine_per_horizon_ml_scores(
    per_horizon_scores: Dict[int, pd.DataFrame],
    close: pd.DataFrame,
    regime_state: pd.DataFrame,
    config: MLAplhaConfig,
) -> pd.DataFrame:
    if not per_horizon_scores:
        return pd.DataFrame(index=close.index, columns=close.columns, dtype=float)

    combined = pd.DataFrame(0.0, index=close.index, columns=close.columns, dtype=float)
    valid_mask = None
    for frame in per_horizon_scores.values():
        frame_valid = frame.notna()
        valid_mask = frame_valid if valid_mask is None else (valid_mask | frame_valid)

    quadrant_series = regime_state["quadrant"].reindex(close.index)
    for dt in close.index:
        weights_for_date = resolve_horizon_weights(config, str(quadrant_series.loc[dt]) if dt in quadrant_series.index else None)
        row = pd.Series(0.0, index=close.columns, dtype=float)
        for horizon, weight in weights_for_date.items():
            frame = per_horizon_scores.get(horizon)
            if frame is None or dt not in frame.index:
                continue
            row = row.add(frame.loc[dt].fillna(0.0) * float(weight), fill_value=0.0)
        combined.loc[dt] = row
    return combined.where(valid_mask)


def rolling_ml_scores_multi_detail(
    feature_frames: Dict[str, pd.DataFrame],
    market_features: Dict[str, pd.Series],
    close: pd.DataFrame,
    benchmark_close: pd.Series,
    filter_mask: pd.DataFrame,
    regime_state: pd.DataFrame,
    config: MLAplhaConfig,
    open_df: pd.DataFrame | None = None,
    benchmark_open: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    horizon_weights = _normalize_horizon_weights(config)
    per_horizon_scores: Dict[int, pd.DataFrame] = {}
    training_logs = []
    for horizon, weight in horizon_weights.items():
        label_df = build_ml_target(
            close,
            benchmark_close,
            horizon,
            execution_mode=config.execution_mode,
            open_df=open_df,
            benchmark_open=benchmark_open,
        )
        horizon_cfg = MLAplhaConfig(**asdict(config))
        horizon_cfg.target_horizon = horizon
        horizon_cfg.target_horizons = (horizon,)
        ml_score, train_log = rolling_ml_scores(
            feature_frames=feature_frames,
            market_features=market_features,
            label_df=label_df,
            filter_mask=filter_mask,
            regime_state=regime_state,
            config=horizon_cfg,
        )
        per_horizon_scores[horizon] = _zscore_cs(ml_score)
        if not train_log.empty:
            train_log = train_log.copy()
            train_log["horizon"] = horizon
            train_log["horizon_weight"] = float(weight)
            training_logs.append(train_log)

    if not per_horizon_scores:
        empty = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
        return empty, pd.DataFrame(), {}

    combined = combine_per_horizon_ml_scores(
        per_horizon_scores=per_horizon_scores,
        close=close,
        regime_state=regime_state,
        config=config,
    )
    training_log_df = pd.concat(training_logs, axis=0, ignore_index=True) if training_logs else pd.DataFrame()
    return combined, training_log_df, per_horizon_scores


def blend_scores(
    ml_score: pd.DataFrame,
    score_none: pd.DataFrame,
    score_v2: pd.DataFrame,
    config: MLAplhaConfig,
    quadrant_series: pd.Series | None = None,
) -> pd.DataFrame:
    ml_z = _zscore_cs(ml_score)
    none_z = _zscore_cs(score_none)
    v2_z = _zscore_cs(score_v2)
    valid_mask = ml_score.notna() | score_none.notna() | score_v2.notna()

    if not config.state_ensemble_weights or quadrant_series is None:
        weights = resolve_ensemble_weights(config)
        final_score = (
            ml_z.fillna(0.0) * float(weights["ml"])
            + none_z.fillna(0.0) * float(weights["none"])
            + v2_z.fillna(0.0) * float(weights["v2"])
        )
        return final_score.where(valid_mask)

    quadrant_series = quadrant_series.reindex(ml_score.index)
    final_score = pd.DataFrame(np.nan, index=ml_score.index, columns=ml_score.columns, dtype=float)
    for dt in ml_score.index:
        weights = resolve_ensemble_weights(config, str(quadrant_series.loc[dt]) if dt in quadrant_series.index else None)
        final_score.loc[dt] = (
            ml_z.loc[dt].fillna(0.0) * float(weights["ml"])
            + none_z.loc[dt].fillna(0.0) * float(weights["none"])
            + v2_z.loc[dt].fillna(0.0) * float(weights["v2"])
        )
    return final_score.where(valid_mask)
