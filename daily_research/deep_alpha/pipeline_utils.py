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


RESEARCH_TIME_UNIT_TRADING_DAYS = "trading_days"
RESEARCH_TIME_UNIT_CALENDAR_MONTHS = "calendar_months"
SUPPORTED_RESEARCH_TIME_UNITS = (
    RESEARCH_TIME_UNIT_TRADING_DAYS,
    RESEARCH_TIME_UNIT_CALENDAR_MONTHS,
)


def normalize_research_time_unit(raw: str | None, *, default: str = RESEARCH_TIME_UNIT_CALENDAR_MONTHS) -> str:
    normalized = str(raw or "").strip().lower()
    if normalized in {"", "default"}:
        return default
    aliases = {
        "day": RESEARCH_TIME_UNIT_TRADING_DAYS,
        "days": RESEARCH_TIME_UNIT_TRADING_DAYS,
        "trading_day": RESEARCH_TIME_UNIT_TRADING_DAYS,
        "trading_days": RESEARCH_TIME_UNIT_TRADING_DAYS,
        "trading-day": RESEARCH_TIME_UNIT_TRADING_DAYS,
        "trading-days": RESEARCH_TIME_UNIT_TRADING_DAYS,
        "month": RESEARCH_TIME_UNIT_CALENDAR_MONTHS,
        "months": RESEARCH_TIME_UNIT_CALENDAR_MONTHS,
        "calendar_month": RESEARCH_TIME_UNIT_CALENDAR_MONTHS,
        "calendar_months": RESEARCH_TIME_UNIT_CALENDAR_MONTHS,
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in SUPPORTED_RESEARCH_TIME_UNITS:
        raise ValueError(
            f"Unsupported research_time_unit: {raw}. "
            f"Expected one of: {', '.join(SUPPORTED_RESEARCH_TIME_UNITS)}."
        )
    return resolved


def _as_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(index))
    if dates.empty:
        raise RuntimeError("Date index is empty.")
    return dates.sort_values().unique()


def _coerce_on_or_after(dates: pd.DatetimeIndex, raw_date: str | pd.Timestamp) -> pd.Timestamp:
    target = pd.Timestamp(raw_date)
    eligible = dates[dates >= target]
    if eligible.empty:
        raise RuntimeError(f"Requested start date {target.date()} is after the available history.")
    return pd.Timestamp(eligible[0])


def _coerce_on_or_before(dates: pd.DatetimeIndex, raw_date: str | pd.Timestamp) -> pd.Timestamp:
    target = pd.Timestamp(raw_date)
    eligible = dates[dates <= target]
    if eligible.empty:
        raise RuntimeError(f"Requested end date {target.date()} is before the available history.")
    return pd.Timestamp(eligible[-1])


def _resolve_day_window_from_start(
    dates: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    window_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_pos = dates.get_loc(pd.Timestamp(start_date))
    end_pos = min(len(dates) - 1, start_pos + max(int(window_days), 1) - 1)
    return pd.Timestamp(dates[start_pos]), pd.Timestamp(dates[end_pos])


def _resolve_calendar_month_window_from_end(
    dates: pd.DatetimeIndex,
    window_months: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if int(window_months) <= 0:
        raise ValueError("window_months must be positive for calendar_months mode.")
    month_periods = dates.to_period("M")
    unique_months = month_periods.unique()
    selected_months = unique_months[-min(int(window_months), len(unique_months)) :]
    mask = month_periods.isin(selected_months)
    selected_dates = dates[mask]
    return pd.Timestamp(selected_dates[0]), pd.Timestamp(selected_dates[-1])


def _resolve_calendar_month_window_from_start(
    dates: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    window_months: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if int(window_months) <= 0:
        raise ValueError("window_months must be positive for calendar_months mode.")
    eligible = dates[dates >= pd.Timestamp(start_date)]
    if eligible.empty:
        raise RuntimeError("No dates remain after the requested start date.")
    month_periods = eligible.to_period("M")
    unique_months = month_periods.unique()
    selected_months = unique_months[: min(int(window_months), len(unique_months))]
    mask = month_periods.isin(selected_months)
    selected_dates = eligible[mask]
    return pd.Timestamp(selected_dates[0]), pd.Timestamp(selected_dates[-1])


def resolve_split_window(
    index: pd.Index,
    train_end_date: str,
    valid_start_date: str,
    valid_days: int,
    *,
    research_time_unit: str = RESEARCH_TIME_UNIT_CALENDAR_MONTHS,
    valid_months: int | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    dates = _as_datetime_index(index)
    time_unit = normalize_research_time_unit(research_time_unit)
    month_window = int(valid_months or 0)

    if train_end_date and valid_start_date:
        valid_start = _coerce_on_or_after(dates, valid_start_date)
        prior_dates = dates[dates < valid_start]
        if prior_dates.empty:
            raise RuntimeError("Not enough history before the requested validation start date.")
        train_end = _coerce_on_or_before(prior_dates, train_end_date)
        if int(valid_days) > 0:
            valid_start, valid_end = _resolve_day_window_from_start(dates, valid_start, int(valid_days))
        elif month_window > 0 and time_unit == RESEARCH_TIME_UNIT_CALENDAR_MONTHS:
            valid_start, valid_end = _resolve_calendar_month_window_from_start(dates, valid_start, month_window)
        else:
            valid_end = pd.Timestamp(dates[-1])
        return pd.Timestamp(train_end), pd.Timestamp(valid_start), pd.Timestamp(valid_end)

    if time_unit == RESEARCH_TIME_UNIT_CALENDAR_MONTHS:
        valid_start, valid_end = _resolve_calendar_month_window_from_end(dates, max(month_window, 1))
    else:
        valid_start = pd.Timestamp(dates[max(0, len(dates) - max(int(valid_days), 1))])
        _valid_start, valid_end = _resolve_day_window_from_start(dates, valid_start, int(valid_days))
        valid_start = _valid_start

    prior_dates = dates[dates < valid_start]
    if prior_dates.empty:
        raise RuntimeError("Not enough history before the derived validation start date.")
    train_end = pd.Timestamp(prior_dates[-1])
    return pd.Timestamp(train_end), pd.Timestamp(valid_start), pd.Timestamp(valid_end)


def resolve_window_dates(
    index: pd.Index,
    end_date: pd.Timestamp,
    window_days: int,
    *,
    research_time_unit: str = RESEARCH_TIME_UNIT_CALENDAR_MONTHS,
    window_months: int | None = None,
) -> pd.DatetimeIndex:
    dates = _as_datetime_index(index)
    eligible = dates[dates <= pd.Timestamp(end_date)]
    if eligible.empty:
        return pd.DatetimeIndex([])
    time_unit = normalize_research_time_unit(research_time_unit)
    if time_unit == RESEARCH_TIME_UNIT_CALENDAR_MONTHS and int(window_months or 0) > 0:
        start_date, resolved_end = _resolve_calendar_month_window_from_end(eligible, int(window_months or 0))
        return eligible[(eligible >= start_date) & (eligible <= resolved_end)]
    if int(window_days) <= 0:
        return eligible
    start_pos = max(0, len(eligible) - int(window_days))
    return eligible[start_pos:]


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
    *,
    research_time_unit: str = RESEARCH_TIME_UNIT_CALENDAR_MONTHS,
    valid_months: int | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    train_end, valid_start, _valid_end = resolve_split_window(
        index,
        train_end_date,
        valid_start_date,
        valid_days,
        research_time_unit=research_time_unit,
        valid_months=valid_months,
    )
    return train_end, valid_start


def resolve_recent_window_start(
    index: pd.Index,
    end_date: pd.Timestamp,
    window_days: int,
    *,
    research_time_unit: str = RESEARCH_TIME_UNIT_CALENDAR_MONTHS,
    window_months: int | None = None,
) -> pd.Timestamp | None:
    window_dates = resolve_window_dates(
        index,
        end_date,
        window_days,
        research_time_unit=research_time_unit,
        window_months=window_months,
    )
    if len(window_dates) == 0:
        return None
    return pd.Timestamp(window_dates[0])


def subset_df_dict_to_stocks(df_dict: dict[str, pd.DataFrame], stocks: list[str]) -> dict[str, pd.DataFrame]:
    keep = [stock for stock in stocks if stock in df_dict["Close"].columns]
    return {field: frame.reindex(columns=keep) for field, frame in df_dict.items()}


def load_raw_market_data(
    cfg: DeepAlphaConfig,
    args: Any,
    universe: list[str],
    *,
    progress_desc: str = "Load daily bars",
    progress_position: int = 0,
) -> tuple[dict[str, pd.DataFrame], str]:
    cache_root = get_cache_root() / "raw"
    force_raw_cache = str(getattr(args, "force_raw_cache_path", "") or "").strip()
    forced_raw_cache_path = Path(force_raw_cache) if force_raw_cache else None
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
    if forced_raw_cache_path is not None:
        forced_cached = load_pickle(forced_raw_cache_path)
        if forced_cached is None:
            raise FileNotFoundError(f"Forced raw cache not found or unreadable: {forced_raw_cache_path}")
        progress_write(f"forced raw cache: {forced_raw_cache_path.name}")
        forced_key = cache_key(
            {
                **raw_meta,
                "forced_raw_cache_path": forced_raw_cache_path.resolve().as_posix(),
                "forced_raw_cache_size": int(forced_raw_cache_path.stat().st_size),
                "forced_raw_cache_mtime_ns": int(forced_raw_cache_path.stat().st_mtime_ns),
            }
        )
        return forced_cached, forced_key
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
