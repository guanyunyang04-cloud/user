from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from daily_research.baseline.data_provider import (
    filter_a_share_universe,
    load_cached_stock_name_map,
    load_daily_from_csv,
    load_daily_from_tq,
)
from daily_research.deep_alpha.cache_utils import cache_key, frame_signature, get_cache_root, load_pickle, save_pickle, series_signature
from daily_research.deep_alpha.config import DeepAlphaConfig
from daily_research.deep_alpha.market_state_model import build_state_frame, fit_market_state_model
from daily_research.deep_alpha.sequence_dataset import build_liquidity_bucket_frame
from daily_research.execution.liquidity_universe import build_rolling_liquidity_membership, get_named_pool_file
from daily_research.progress import progress_write


def parse_stocks(raw: str | None) -> list[str]:
    if not raw:
        return []
    stock_name_map = load_cached_stock_name_map()
    stocks = [stock.strip().upper() for stock in raw.split(",") if stock.strip()]
    return filter_a_share_universe(
        stocks,
        universe_scope="all_a",
        stock_name_map=stock_name_map if not stock_name_map.empty else None,
    )


def load_stocks_from_file(path: str | None) -> list[str]:
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
    stock_name_map = load_cached_stock_name_map()
    return filter_a_share_universe(
        [token.upper() for token in tokens],
        universe_scope="all_a",
        stock_name_map=stock_name_map if not stock_name_map.empty else None,
    )


def resolve_stocks_file(args: Any) -> str | None:
    if getattr(args, "rolling_liquidity_pool", ""):
        return None
    if getattr(args, "stocks_file", ""):
        return args.stocks_file
    if getattr(args, "liquidity_pool", ""):
        pool_file = get_named_pool_file(args.liquidity_pool)
        if not pool_file.exists():
            raise FileNotFoundError(
                f"Liquidity pool file not found: {pool_file}. "
                "Run `python daily_research/execution/update_liquid_pool.py` first."
            )
        return str(pool_file)
    return None


def parse_horizons(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def parse_named_weights(raw: str, allowed: set[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    if not raw:
        return weights
    for item in raw.split(","):
        part = item.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid weight item: {part}")
        key, value = part.split(":", 1)
        key = key.strip().lower()
        if key not in allowed:
            raise ValueError(f"Unsupported weight key: {key}")
        weights[key] = float(value.strip())
    return weights


def build_target_loss_weights(horizons: tuple[int, ...], raw: str) -> dict[str, float]:
    defaults = {5: 0.2, 10: 0.3, 20: 0.5}
    allowed = {str(h) for h in horizons} | {"downside"}
    parsed = parse_named_weights(raw, allowed) if raw else {}
    out = {
        f"fwd_excess_{h}": float(parsed.get(str(h), defaults.get(h, 1.0)))
        for h in horizons
    }
    out["risk_downside_20"] = float(parsed.get("downside", 0.35))
    return out


def build_score_horizon_weights(horizons: tuple[int, ...], raw: str) -> dict[int, float]:
    defaults = {5: 0.2, 10: 0.3, 20: 0.5}
    allowed = {str(h) for h in horizons}
    parsed = parse_named_weights(raw, allowed) if raw else {}
    return {h: float(parsed.get(str(h), defaults.get(h, 0.0))) for h in horizons}


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_name_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def resolve_split_dates(
    index: pd.Index,
    train_end_date: str,
    valid_start_date: str,
    valid_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = pd.to_datetime(index)
    if train_end_date and valid_start_date:
        return pd.Timestamp(train_end_date), pd.Timestamp(valid_start_date)
    valid_start = dates[max(0, len(dates) - valid_days)]
    train_end = dates[dates.get_loc(valid_start) - 1]
    return pd.Timestamp(train_end), pd.Timestamp(valid_start)


def resolve_recent_window_start(index: pd.Index, end_date: pd.Timestamp, window_days: int) -> pd.Timestamp | None:
    if int(window_days) <= 0:
        return None
    dates = pd.to_datetime(index)
    eligible = dates[dates <= pd.Timestamp(end_date)]
    if len(eligible) == 0:
        return None
    start_pos = max(0, len(eligible) - int(window_days))
    return pd.Timestamp(eligible[start_pos])


def subset_df_dict_to_stocks(df_dict: dict[str, pd.DataFrame], stocks: list[str]) -> dict[str, pd.DataFrame]:
    keep = [stock for stock in stocks if stock in df_dict["Close"].columns]
    return {field: frame.reindex(columns=keep) for field, frame in df_dict.items()}


def load_raw_market_data(
    cfg: DeepAlphaConfig,
    args: Any,
    universe: list[str],
    *,
    progress_desc: str = "读取股票日线",
    progress_position: int = 0,
) -> tuple[dict[str, pd.DataFrame], str]:
    cache_root = get_cache_root() / "raw"
    raw_meta = {
        "version": 1,
        "data_source": args.data_source,
        "universe_scope": cfg.universe_scope,
        "benchmark": cfg.benchmark,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "stocks_count": len(universe),
        "stocks_hash": cache_key({"stocks": sorted(universe)}),
    }
    raw_key = cache_key(raw_meta)
    raw_path = cache_root / f"{raw_key}.pkl"
    if args.data_source == "tq":
        if args.use_cache and not args.refresh_cache:
            cached = load_pickle(raw_path)
            if cached is not None:
                progress_write(f"raw cache hit: {raw_path.name}")
                return cached, raw_key
        raw_df_dict = load_daily_from_tq(
            universe,
            cfg.start_date,
            cfg.end_date,
            benchmark=cfg.benchmark,
            progress_desc=progress_desc,
            progress_position=progress_position,
        )
        if args.use_cache:
            save_pickle(raw_path, raw_df_dict)
            progress_write(f"saved raw cache: {raw_path}")
        return raw_df_dict, raw_key

    if not args.csv_folder:
        raise ValueError("CSV mode requires --csv-folder.")
    raw_df_dict = load_daily_from_csv(
        args.csv_folder,
        progress_desc=progress_desc,
        progress_position=progress_position,
    )
    return raw_df_dict, raw_key


def load_cached_or_build_rolling_pool(
    *,
    args: Any,
    cfg: DeepAlphaConfig,
    df_dict: dict[str, pd.DataFrame],
    raw_key: str,
):
    if not args.rolling_liquidity_pool:
        return None, ""
    rolling_meta = {
        "version": 1,
        "raw_key": raw_key,
        "pool_name": args.rolling_liquidity_pool,
        "signal_start_date": cfg.start_date,
        "signal_end_date": cfg.end_date,
        "rebalance_every_days": int(args.pool_rebalance_days),
        "adv_window": int(args.pool_adv_window),
        "min_price": float(cfg.min_price),
        "max_price": float(cfg.max_price),
    }
    rolling_key = cache_key(rolling_meta)
    rolling_path = get_cache_root() / "rolling_pools" / f"{rolling_key}.pkl"
    if args.use_cache and not args.refresh_cache:
        cached = load_pickle(rolling_path)
        if cached is not None:
            progress_write(f"rolling pool cache hit: {rolling_path.name}")
            return cached, rolling_key
    progress_write(
        f"building rolling pool: {args.rolling_liquidity_pool} "
        f"(rebalance_every={args.pool_rebalance_days}d)"
    )
    artifact = build_rolling_liquidity_membership(
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
    if args.use_cache:
        save_pickle(rolling_path, artifact)
        progress_write(f"saved rolling pool cache: {rolling_path}")
    return artifact, rolling_key


def load_cached_or_fit_market_state(
    *,
    args: Any,
    cfg: DeepAlphaConfig,
    benchmark_close: pd.Series,
    raw_key: str,
):
    state_meta = {
        "version": 1,
        "raw_key": raw_key,
        "market_state_count": int(cfg.market_state_count),
        "benchmark_signature": series_signature(benchmark_close),
        "random_seed": int(cfg.random_seed),
    }
    state_key = cache_key(state_meta)
    state_path = get_cache_root() / "states" / f"{state_key}.pkl"
    if args.use_cache and not args.refresh_cache:
        cached = load_pickle(state_path)
        if cached is not None:
            progress_write(f"market state cache hit: {state_path.name}")
            return cached["state_frame"], cached["state_name_map"], state_key
    progress_write("fitting market state model...")
    state_artifact = fit_market_state_model(
        benchmark_close,
        n_states=cfg.market_state_count,
        random_seed=cfg.random_seed,
    )
    state_frame = build_state_frame(state_artifact)
    state_name_map = {int(k): str(v) for k, v in state_artifact.state_names.items()}
    if args.use_cache:
        save_pickle(state_path, {"state_frame": state_frame, "state_name_map": state_name_map})
        progress_write(f"saved market state cache: {state_path}")
    return state_frame, state_name_map, state_key


def load_cached_or_build_liquidity_buckets(
    *,
    args: Any,
    cfg: DeepAlphaConfig,
    amount_frame: pd.DataFrame,
    raw_key: str,
):
    bucket_meta = {
        "version": 1,
        "raw_key": raw_key,
        "liquidity_bucket_count": int(cfg.liquidity_bucket_count),
        "amount_signature": frame_signature(amount_frame),
    }
    bucket_key = cache_key(bucket_meta)
    bucket_path = get_cache_root() / "liquidity_buckets" / f"{bucket_key}.pkl"
    if args.use_cache and not args.refresh_cache:
        cached = load_pickle(bucket_path)
        if cached is not None:
            progress_write(f"liquidity bucket cache hit: {bucket_path.name}")
            return cached, bucket_key
    liquidity_bucket_frame = build_liquidity_bucket_frame(
        amount_frame,
        window=20,
        n_buckets=cfg.liquidity_bucket_count,
    )
    if args.use_cache:
        save_pickle(bucket_path, liquidity_bucket_frame)
        progress_write(f"saved liquidity bucket cache: {bucket_path}")
    return liquidity_bucket_frame, bucket_key
