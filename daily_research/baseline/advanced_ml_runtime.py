from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd

from daily_research.baseline.alpha import combine_scores_by_state
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import (
    get_latest_completed_trading_date,
    load_daily_from_csv,
    load_daily_from_tq,
    split_benchmark_from_universe,
)
from daily_research.baseline.features import compute_factors
from daily_research.baseline.ml_alpha import MLAplhaConfig, build_ml_feature_bundle
from daily_research.baseline.regime import compute_market_regime_state
from daily_research.baseline.state_profiles import build_state_configs


@dataclass(frozen=True)
class HistoryWindow:
    mode: str
    requested_start_date: str
    effective_start_date: str
    end_date: str
    required_trading_days: int


def get_cache_root() -> Path:
    root = Path("daily_research/cache/advanced_ml")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_key(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:20]


def _load_pickle(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def _save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _stock_signature(stocks: Iterable[str]) -> str:
    normalized = ",".join(sorted(set(str(stock).upper() for stock in stocks if stock)))
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:20]


def _csv_folder_signature(csv_folder: str | None) -> str:
    if not csv_folder:
        return "none"
    folder = Path(csv_folder)
    if not folder.exists():
        return "missing"
    payload = [
        f"{item.name}:{int(item.stat().st_mtime_ns)}:{item.stat().st_size}"
        for item in sorted(folder.glob("*.csv"))
    ]
    normalized = "|".join(payload)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:20] if normalized else "empty"


def _normalize_horizons(ml_cfg: MLAplhaConfig) -> tuple[int, ...]:
    horizons = tuple(int(h) for h in (ml_cfg.target_horizons or (ml_cfg.target_horizon,)) if int(h) > 0)
    if not horizons:
        return (int(ml_cfg.target_horizon),)
    return tuple(dict.fromkeys(horizons))


def _estimate_feature_warmup_days(cfg: ResearchConfig, max_horizon: int) -> int:
    rolling_lookback = max(60, int(cfg.regime_ma_window), int(cfg.regime_vol_window), int(max_horizon))
    ema_warmup = 120
    safety_buffer = max(40, int(max_horizon))
    return max(252, rolling_lookback + ema_warmup + safety_buffer)


def resolve_history_window(
    cfg: ResearchConfig,
    ml_cfg: MLAplhaConfig,
    requested_start_date: str,
    end_date: str,
    mode: str,
    auto_trim_history: bool = True,
) -> HistoryWindow:
    max_horizon = max(_normalize_horizons(ml_cfg))
    feature_warmup_days = _estimate_feature_warmup_days(cfg, max_horizon)
    if mode == "train":
        required_trading_days = int(ml_cfg.train_window_days) + feature_warmup_days
    elif mode == "infer":
        required_trading_days = feature_warmup_days
    else:
        raise ValueError(f"Unsupported history window mode: {mode}")

    requested_start = str(requested_start_date or "").strip() or "20180101"
    effective_end_date = (
        str(end_date).strip()
        if str(end_date).strip()
        else get_latest_completed_trading_date()
    )

    if not auto_trim_history:
        return HistoryWindow(
            mode=mode,
            requested_start_date=requested_start,
            effective_start_date=requested_start,
            end_date=pd.Timestamp(effective_end_date).strftime("%Y%m%d"),
            required_trading_days=required_trading_days,
        )

    end_ts = pd.Timestamp(effective_end_date).normalize()
    auto_start_ts = end_ts - pd.offsets.BDay(required_trading_days + 40)
    requested_start_ts = pd.Timestamp(requested_start)
    effective_start_ts = max(requested_start_ts, auto_start_ts)

    return HistoryWindow(
        mode=mode,
        requested_start_date=requested_start,
        effective_start_date=effective_start_ts.strftime("%Y%m%d"),
        end_date=end_ts.strftime("%Y%m%d"),
        required_trading_days=required_trading_days,
    )


def slice_data_dict(df_dict: Dict[str, pd.DataFrame], start_date: str, end_date: str = "") -> Dict[str, pd.DataFrame]:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) if str(end_date).strip() else None
    sliced: Dict[str, pd.DataFrame] = {}
    for field, frame in df_dict.items():
        out = frame.copy()
        out.index = pd.to_datetime(out.index)
        out = out.loc[out.index >= start_ts]
        if end_ts is not None:
            out = out.loc[out.index <= end_ts]
        sliced[field] = out
    return sliced


def load_raw_data_with_cache(
    *,
    data_source: str,
    csv_folder: str | None,
    universe: list[str],
    benchmark: str,
    history_window: HistoryWindow,
    use_cache: bool = True,
    refresh_cache: bool = False,
    progress_desc: str = "读取股票日线",
    progress_position: int = 0,
) -> tuple[Dict[str, pd.DataFrame], dict[str, Any]]:
    payload = {
        "kind": "raw_data",
        "data_source": data_source,
        "csv_folder": str(Path(csv_folder).resolve()) if csv_folder else "",
        "csv_folder_signature": _csv_folder_signature(csv_folder),
        "universe_signature": _stock_signature(universe),
        "universe_size": len(universe),
        "benchmark": benchmark,
        "effective_start_date": history_window.effective_start_date,
        "end_date": history_window.end_date,
    }
    cache_id = _cache_key(payload)
    cache_path = get_cache_root() / "raw" / f"{cache_id}.pkl"

    if use_cache and not refresh_cache:
        cached = _load_pickle(cache_path)
        if cached is not None:
            return cached, {"cache_key": cache_id, "cache_path": str(cache_path), "cache_hit": True}

    if data_source == "tq":
        raw_df_dict = load_daily_from_tq(
            universe,
            history_window.effective_start_date,
            history_window.end_date,
            benchmark=benchmark,
            progress_desc=progress_desc,
            progress_position=progress_position,
        )
    elif data_source == "csv":
        if not csv_folder:
            raise ValueError("CSV mode requires --csv-folder.")
        raw_df_dict = load_daily_from_csv(
            csv_folder,
            progress_desc=progress_desc,
            progress_position=progress_position,
        )
        raw_df_dict = slice_data_dict(raw_df_dict, history_window.effective_start_date, history_window.end_date)
    else:
        raise ValueError(f"Unsupported data_source: {data_source}")

    if use_cache:
        _save_pickle(cache_path, raw_df_dict)
    return raw_df_dict, {"cache_key": cache_id, "cache_path": str(cache_path), "cache_hit": False}


def build_prepared_bundle_with_cache(
    *,
    raw_df_dict: Dict[str, pd.DataFrame],
    raw_cache_key: str,
    cfg: ResearchConfig,
    enhanced_profile: str,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "kind": "prepared_bundle",
        "raw_cache_key": raw_cache_key,
        "benchmark": cfg.benchmark,
        "score_clip": float(cfg.score_clip),
        "factor_group_weights": dict(cfg.factor_group_weights),
        "factor_weights": dict(cfg.factor_weights),
        "factor_groups": dict(cfg.factor_groups),
        "regime_ma_window": int(cfg.regime_ma_window),
        "regime_vol_window": int(cfg.regime_vol_window),
        "regime_max_annual_vol": float(cfg.regime_max_annual_vol),
        "regime_allowed_quadrants": list(cfg.regime_allowed_quadrants),
        "min_adv20": float(cfg.min_adv20),
        "min_price": float(cfg.min_price),
        "max_price": float(cfg.max_price),
        "score_threshold": float(cfg.score_threshold),
        "enhanced_profile": str(enhanced_profile),
    }
    cache_id = _cache_key(payload)
    cache_path = get_cache_root() / "prepared" / f"{cache_id}.pkl"

    if use_cache and not refresh_cache:
        cached = _load_pickle(cache_path)
        if cached is not None:
            return cached, {"cache_key": cache_id, "cache_path": str(cache_path), "cache_hit": True}

    df_dict, benchmark_close = split_benchmark_from_universe(raw_df_dict, cfg.benchmark)
    benchmark_open = raw_df_dict["Open"][cfg.benchmark].copy()
    factor_bundle = compute_factors(df_dict)
    regime_state = compute_market_regime_state(benchmark_close, cfg)
    score_none, _, filter_mask = combine_scores_by_state(
        factor_bundle,
        cfg,
        quadrant_series=regime_state["quadrant"],
        state_configs=build_state_configs(cfg, "none"),
    )
    score_v2, _, _ = combine_scores_by_state(
        factor_bundle,
        cfg,
        quadrant_series=regime_state["quadrant"],
        state_configs=build_state_configs(cfg, enhanced_profile),
    )
    feature_frames, market_features = build_ml_feature_bundle(factor_bundle, regime_state, score_none, score_v2)

    bundle = {
        "df_dict": df_dict,
        "benchmark_close": benchmark_close,
        "benchmark_open": benchmark_open,
        "factor_bundle": factor_bundle,
        "regime_state": regime_state,
        "score_none": score_none,
        "score_v2": score_v2,
        "filter_mask": filter_mask,
        "feature_frames": feature_frames,
        "market_features": market_features,
    }
    if use_cache:
        _save_pickle(cache_path, bundle)
    return bundle, {"cache_key": cache_id, "cache_path": str(cache_path), "cache_hit": False}


def history_window_to_dict(history_window: HistoryWindow) -> dict[str, Any]:
    return asdict(history_window)
