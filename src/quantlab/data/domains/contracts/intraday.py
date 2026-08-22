"""Domain contract intraday definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .base import (
    _date_series,
    _ensure_domain_columns,
    _normalize_symbol,
    _prepare_domain_frame,
    _rename_first,
    _rename_intraday_value_columns,
    _source_series,
    validate_provider_name,
)
from .market import (
    _require_core_columns,
)
from .schema import (
    DOMAIN_STANDARD_COLUMNS,
    NUMERIC_MARKET_COLUMNS,
    DataDomain,
)


def normalize_intraday_5m_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    adjusted_flag: str = "none",
    require_columns: bool = True,
) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(
        frame, domain=DataDomain.MARKET_INTRADAY_5M, source=provider, as_of_date="", require_columns=False
    )
    _rename_first(data, "bar_time", ("time", "bar_time", "minute", "bar_datetime", "时间", "分钟"))
    _rename_intraday_value_columns(data, include_share_fields=False)
    if "bar_time" not in data.columns and "trade_date" in data.columns:
        data["bar_time"] = ""
    if "adjusted_flag" not in data.columns:
        data["adjusted_flag"] = str(adjusted_flag or "none")
    _require_core_columns(
        data, DataDomain.MARKET_INTRADAY_5M, {"symbol", "trade_date", "bar_time"}, require_columns=require_columns
    )
    data = _ensure_domain_columns(data, DataDomain.MARKET_INTRADAY_5M)
    data["symbol"] = data["symbol"].map(_normalize_symbol)
    data = _split_intraday_datetime_column(data)
    data["bar_time"] = data["bar_time"].map(_normalize_bar_time)
    for column in NUMERIC_MARKET_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["source"] = _source_series(data, provider)
    data["adjusted_flag"] = (
        data["adjusted_flag"].fillna(str(adjusted_flag or "none")).astype(str).str.strip().replace("", "none")
    )
    out = data.loc[
        data["symbol"].astype(str).str.len().gt(0)
        & data["trade_date"].astype(str).str.lower().ne("nat")
        & data["bar_time"].astype(str).str.len().gt(0),
        DOMAIN_STANDARD_COLUMNS[DataDomain.MARKET_INTRADAY_5M],
    ]
    return out.sort_values(["trade_date", "symbol", "bar_time", "source"]).reset_index(drop=True)


def normalize_intraday_1m_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    adjusted_flag: str = "none",
    require_columns: bool = True,
) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(
        frame, domain=DataDomain.MARKET_INTRADAY_1M, source=provider, as_of_date="", require_columns=False
    )
    _rename_first(data, "bar_time", ("time", "bar_time", "minute", "bar_datetime", "datetime", "日期", "时间", "分钟"))
    _rename_intraday_value_columns(data, include_share_fields=False)
    if "bar_time" not in data.columns and "trade_date" in data.columns:
        data["bar_time"] = ""
    if "adjusted_flag" not in data.columns:
        data["adjusted_flag"] = str(adjusted_flag or "none")
    _require_core_columns(
        data, DataDomain.MARKET_INTRADAY_1M, {"symbol", "trade_date", "bar_time"}, require_columns=require_columns
    )
    data = _ensure_domain_columns(data, DataDomain.MARKET_INTRADAY_1M)
    data["symbol"] = data["symbol"].map(_normalize_symbol)
    data = _split_intraday_datetime_column(data)
    data["bar_time"] = data["bar_time"].map(_normalize_bar_time)
    for column in NUMERIC_MARKET_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["source"] = _source_series(data, provider)
    data["adjusted_flag"] = (
        data["adjusted_flag"].fillna(str(adjusted_flag or "none")).astype(str).str.strip().replace("", "none")
    )
    out = data.loc[
        data["symbol"].astype(str).str.len().gt(0)
        & data["trade_date"].astype(str).str.lower().ne("nat")
        & data["bar_time"].astype(str).str.len().gt(0),
        DOMAIN_STANDARD_COLUMNS[DataDomain.MARKET_INTRADAY_1M],
    ]
    return (
        out.drop_duplicates(subset=["trade_date", "symbol", "bar_time", "source"])
        .sort_values(["trade_date", "symbol", "bar_time", "source"])
        .reset_index(drop=True)
    )


def aggregate_intraday_1m_to_5m_frame(
    frame: pd.DataFrame,
    *,
    source: str = "external_1m",
    adjusted_flag: str = "none",
) -> pd.DataFrame:
    one_minute = normalize_intraday_1m_frame(frame, source=source, adjusted_flag=adjusted_flag, require_columns=False)
    if one_minute.empty:
        return pd.DataFrame(columns=DOMAIN_STANDARD_COLUMNS[DataDomain.MARKET_INTRADAY_5M])
    data = one_minute.copy()
    clock = data["bar_time"].map(_bar_time_to_clock)
    stamp = pd.to_datetime(data["trade_date"].astype(str) + " " + clock.astype(str), errors="coerce")
    data = data.loc[stamp.notna()].copy()
    stamp = stamp.loc[data.index]
    data["_bar_timestamp"] = stamp
    floored = stamp.dt.floor("5min")
    data["bar_time"] = floored.dt.strftime("%H:%M:%S")
    for column in NUMERIC_MARKET_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.sort_values(["trade_date", "symbol", "_bar_timestamp"])
    grouped = data.groupby(["trade_date", "symbol", "bar_time"], sort=True, as_index=False)
    rows = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
    )
    rows["source"] = f"{source}_agg_5m"
    rows["adjusted_flag"] = str(adjusted_flag or "none")
    return normalize_intraday_5m_frame(
        rows, source=f"{source}_agg_5m", adjusted_flag=adjusted_flag, require_columns=False
    )


@dataclass(frozen=True)
class _IntradayDay:
    day: pd.DataFrame
    open_values: pd.Series
    high_values: pd.Series
    low_values: pd.Series
    close_values: pd.Series
    volume_values: pd.Series
    amount_values: pd.Series
    total_amount: float
    total_volume: float
    first_open: float
    last_close: float
    previous_close: float
    high_max: float
    low_min: float
    close_returns: pd.Series
    clocks: pd.Series
    amount_share: pd.Series
    cumulative_vwap: pd.Series


def _intraday_day(group: pd.DataFrame, previous_close: float) -> _IntradayDay:
    day = group.sort_values("bar_time").reset_index(drop=True).copy()
    open_values = pd.to_numeric(day["open"], errors="coerce")
    high_values = pd.to_numeric(day["high"], errors="coerce")
    low_values = pd.to_numeric(day["low"], errors="coerce")
    close_values = pd.to_numeric(day["close"], errors="coerce")
    volume_values = pd.to_numeric(day["volume"], errors="coerce")
    amount_values = pd.to_numeric(day["amount"], errors="coerce")
    total_amount = _finite_sum(amount_values)
    total_volume = _finite_sum(volume_values)
    cumulative_volume = volume_values.cumsum()
    amount_share = amount_values / total_amount if total_amount > 0 else pd.Series(np.nan, index=amount_values.index)
    return _IntradayDay(
        day=day,
        open_values=open_values,
        high_values=high_values,
        low_values=low_values,
        close_values=close_values,
        volume_values=volume_values,
        amount_values=amount_values,
        total_amount=total_amount,
        total_volume=total_volume,
        first_open=_first_finite(open_values),
        last_close=_last_finite(close_values),
        previous_close=previous_close,
        high_max=float(high_values.max()) if high_values.notna().any() else np.nan,
        low_min=float(low_values.min()) if low_values.notna().any() else np.nan,
        close_returns=close_values.pct_change().replace([np.inf, -np.inf], np.nan),
        clocks=day["bar_time"].map(_bar_clock_int),
        amount_share=amount_share,
        cumulative_vwap=amount_values.cumsum() / cumulative_volume.where(cumulative_volume > 0),
    )


def _opening_features(context: _IntradayDay) -> dict[str, float]:
    opening_amount = _finite_sum(context.amount_values.head(1))
    opening_volume = _finite_sum(context.volume_values.head(1))
    opening_share = opening_amount / context.total_amount if context.total_amount > 0 else np.nan
    opening_return = _head_window_ret(context.day, 1)
    first_30m_return = _head_window_ret(context.day, 6)
    first_30m_share = (
        _window_sum(context.amount_values, 6, head=True) / context.total_amount if context.total_amount > 0 else np.nan
    )
    open_gap = _safe_return(context.first_open, context.previous_close)
    gap_sign = np.sign(open_gap) if pd.notna(open_gap) else np.nan
    return {
        "first_5m_ret": _head_window_ret(context.day, 1),
        "opening_auction_ret": opening_return,
        "opening_auction_amount": opening_amount,
        "opening_auction_volume": opening_volume,
        "opening_auction_amount_share": opening_share,
        "opening_auction_range": _window_range(context.day, 1, head=True),
        "opening_auction_vwap": opening_amount / opening_volume
        if opening_amount > 0 and opening_volume > 0
        else np.nan,
        "opening_auction_pressure": opening_return * opening_share
        if pd.notna(opening_return) and pd.notna(opening_share)
        else np.nan,
        "first_15m_ret": _head_window_ret(context.day, 3),
        "first_30m_ret": first_30m_return,
        "first_30m_amount_share": first_30m_share,
        "open_gap": open_gap,
        "open_gap_first_30m_follow_through": gap_sign * first_30m_return
        if pd.notna(gap_sign) and pd.notna(first_30m_return)
        else np.nan,
        "open_gap_first_30m_reversal": -gap_sign * first_30m_return
        if pd.notna(gap_sign) and pd.notna(first_30m_return)
        else np.nan,
    }


def _closing_features(context: _IntradayDay) -> dict[str, float]:
    closing_amount = _finite_sum(context.amount_values.tail(1))
    closing_volume = _finite_sum(context.volume_values.tail(1))
    closing_share = closing_amount / context.total_amount if context.total_amount > 0 else np.nan
    closing_return = _tail_close_to_previous_close_ret(context.day)
    last_30m_return = _tail_window_ret(context.day, 6)
    last_30m_share = (
        _window_sum(context.amount_values, 6, head=False) / context.total_amount if context.total_amount > 0 else np.nan
    )
    return {
        "last_5m_ret": closing_return,
        "closing_auction_ret": closing_return,
        "closing_auction_amount": closing_amount,
        "closing_auction_volume": closing_volume,
        "closing_auction_amount_share": closing_share,
        "closing_auction_range": _window_range(context.day, 1, head=False),
        "closing_auction_vwap": closing_amount / closing_volume
        if closing_amount > 0 and closing_volume > 0
        else np.nan,
        "closing_auction_pressure": closing_return * closing_share
        if pd.notna(closing_return) and pd.notna(closing_share)
        else np.nan,
        "last_30m_ret": last_30m_return,
        "last_30m_amount_share": last_30m_share,
        "close_pressure_30m": last_30m_return * last_30m_share
        if pd.notna(last_30m_return) and pd.notna(last_30m_share)
        else np.nan,
    }


def _path_features(context: _IntradayDay) -> dict[str, float]:
    bar_count = len(context.day)
    high_position = _first_extreme_position(context.high_values, mode="max")
    low_position = _first_extreme_position(context.low_values, mode="min")
    vwap = (
        context.total_amount / context.total_volume if context.total_amount > 0 and context.total_volume > 0 else np.nan
    )
    close_position = (
        (context.last_close - context.low_min) / (context.high_max - context.low_min)
        if pd.notna(context.last_close)
        and pd.notna(context.high_max)
        and pd.notna(context.low_min)
        and context.high_max > context.low_min
        else np.nan
    )
    price_volume_corr = (
        context.close_returns.corr(context.volume_values)
        if context.close_returns.notna().sum() >= 2 and context.volume_values.notna().sum() >= 2
        else np.nan
    )
    return {
        "intraday_ret": _safe_return(context.last_close, context.first_open),
        "intraday_vwap": vwap,
        "close_to_vwap": _safe_return(context.last_close, vwap),
        "intraday_range": _safe_return(context.high_max, context.low_min),
        "close_position": close_position,
        "intraday_realized_vol": float(context.close_returns.std())
        if context.close_returns.notna().sum() >= 2
        else np.nan,
        "intraday_price_volume_corr": price_volume_corr,
        "bar_count": float(bar_count),
        "high_time_frac": _position_fraction(high_position, bar_count),
        "low_time_frac": _position_fraction(low_position, bar_count),
        "high_before_low": float(high_position < low_position)
        if high_position >= 0 and low_position >= 0 and high_position != low_position
        else (0.5 if high_position >= 0 and low_position >= 0 else np.nan),
        "open_to_high_ret": _safe_return(context.high_max, context.first_open),
        "open_to_low_ret": _safe_return(context.low_min, context.first_open),
        "high_to_close_ret": _safe_return(context.last_close, context.high_max),
        "low_to_close_ret": _safe_return(context.last_close, context.low_min),
        "intraday_max_drawdown": _max_drawdown(context.close_values),
        "intraday_max_runup": _max_runup(context.close_values),
        "price_above_vwap_share": _finite_ratio(context.close_values > vwap) if pd.notna(vwap) else np.nan,
        "cum_vwap_slope": _linear_slope(context.cumulative_vwap),
        "first_5m_amount_share": _window_sum(context.amount_values, 1, head=True) / context.total_amount
        if context.total_amount > 0
        else np.nan,
        "last_5m_amount_share": _window_sum(context.amount_values, 1, head=False) / context.total_amount
        if context.total_amount > 0
        else np.nan,
        "first_30m_range": _window_range(context.day, 6, head=True),
        "last_30m_range": _window_range(context.day, 6, head=False),
        "amount_top_bar_share": float(context.amount_share.max()) if context.amount_share.notna().any() else np.nan,
        "amount_concentration_hhi": float((context.amount_share.dropna() ** 2).sum())
        if context.amount_share.notna().any()
        else np.nan,
    }


def _session_features(context: _IntradayDay) -> dict[str, float]:
    am_mask = context.clocks.le(113000)
    pm_mask = context.clocks.ge(130000)
    if not bool(am_mask.any()) and len(context.day) > 1:
        am_mask = pd.Series(np.arange(len(context.day)) < len(context.day) // 2, index=context.day.index)
    if not bool(pm_mask.any()) and len(context.day) > 1:
        pm_mask = ~am_mask
    morning = context.day.loc[am_mask]
    afternoon = context.day.loc[pm_mask]
    am_return = _session_return(morning)
    pm_return = _session_return(afternoon)
    lunch_gap = (
        _safe_return(afternoon["open"].iloc[0], morning["close"].iloc[-1])
        if not morning.empty and not afternoon.empty
        else np.nan
    )
    am_vol = _session_volatility(morning)
    pm_vol = _session_volatility(afternoon)
    am_amount_share = (
        _finite_sum(pd.to_numeric(morning.get("amount", pd.Series(dtype=float)), errors="coerce"))
        / context.total_amount
        if context.total_amount > 0
        else np.nan
    )
    pm_amount_share = 1.0 - am_amount_share if pd.notna(am_amount_share) else np.nan
    first_30m_return = _head_window_ret(context.day, 6)
    last_30m_return = _tail_window_ret(context.day, 6)
    return {
        "lunch_gap_ret": lunch_gap,
        "am_ret": am_return,
        "pm_ret": pm_return,
        "am_pm_ret_spread": pm_return - am_return,
        "am_pm_vol_spread": pm_vol - am_vol,
        "am_amount_share": am_amount_share,
        "am_pm_amount_spread": am_amount_share - pm_amount_share
        if pd.notna(am_amount_share) and pd.notna(pm_amount_share)
        else np.nan,
        "early_strength_late_weak": first_30m_return - last_30m_return
        if pd.notna(first_30m_return) and pd.notna(last_30m_return)
        else np.nan,
    }


def _session_volatility(frame: pd.DataFrame) -> float:
    if frame.empty:
        return np.nan
    return float(pd.to_numeric(frame["close"], errors="coerce").pct_change().replace([np.inf, -np.inf], np.nan).std())


def build_intraday_daily_feature_frame(
    intraday_frame: pd.DataFrame,
    *,
    source: str = "baostock",
    adjusted_flag: str = "none",
) -> pd.DataFrame:
    bars = normalize_intraday_5m_frame(
        intraday_frame,
        source=source,
        adjusted_flag=adjusted_flag,
        require_columns=False,
    )
    if bars.empty:
        return pd.DataFrame(columns=DOMAIN_STANDARD_COLUMNS[DataDomain.INTRADAY_DAILY_FEATURES])
    rows: list[dict[str, Any]] = []
    previous_close: dict[str, float] = {}
    for (trade_date, symbol), group in bars.groupby(["trade_date", "symbol"], sort=True):
        context = _intraday_day(group, previous_close.get(str(symbol), np.nan))
        if context.day.empty:
            continue
        rows.append(
            {
                "symbol": symbol,
                "trade_date": trade_date,
                **_opening_features(context),
                **_closing_features(context),
                **_path_features(context),
                **_session_features(context),
                "source": source,
                "adjusted_flag": str(adjusted_flag or "none"),
            }
        )
        previous_close[str(symbol)] = context.last_close
    return normalize_intraday_daily_features_frame(
        pd.DataFrame(rows),
        source=source,
        adjusted_flag=adjusted_flag,
        require_columns=False,
    )


def normalize_intraday_daily_features_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    adjusted_flag: str = "none",
    require_columns: bool = True,
) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(
        frame, domain=DataDomain.INTRADAY_DAILY_FEATURES, source=provider, as_of_date="", require_columns=False
    )
    if "adjusted_flag" not in data.columns:
        data["adjusted_flag"] = str(adjusted_flag or "none")
    _require_core_columns(
        data, DataDomain.INTRADAY_DAILY_FEATURES, {"symbol", "trade_date"}, require_columns=require_columns
    )
    data = _ensure_domain_columns(data, DataDomain.INTRADAY_DAILY_FEATURES)
    data["symbol"] = data["symbol"].map(_normalize_symbol)
    data["trade_date"] = _date_series(data["trade_date"])
    for column in DOMAIN_STANDARD_COLUMNS[DataDomain.INTRADAY_DAILY_FEATURES]:
        if column not in {"symbol", "trade_date", "source", "adjusted_flag"}:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data["source"] = _source_series(data, provider)
    data["adjusted_flag"] = (
        data["adjusted_flag"].fillna(str(adjusted_flag or "none")).astype(str).str.strip().replace("", "none")
    )
    out = data.loc[
        data["symbol"].astype(str).str.len().gt(0) & data["trade_date"].astype(str).str.lower().ne("nat"),
        DOMAIN_STANDARD_COLUMNS[DataDomain.INTRADAY_DAILY_FEATURES],
    ]
    return (
        out.drop_duplicates(subset=["trade_date", "symbol", "source"])
        .sort_values(["trade_date", "symbol", "source"])
        .reset_index(drop=True)
    )


def _normalize_bar_time(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 9:
        return digits[-9:]
    if len(digits) >= 6:
        return f"{digits[:6]}000"
    if len(digits) == 4:
        return f"{digits}00000"
    return digits


def _bar_time_to_clock(value: Any) -> str:
    text = _normalize_bar_time(value)
    if len(text) >= 6:
        return f"{text[:2]}:{text[2:4]}:{text[4:6]}"
    return ""


def _split_intraday_datetime_column(data: pd.DataFrame) -> pd.DataFrame:
    trade_ts = pd.to_datetime(data["trade_date"], errors="coerce")
    normalized_trade_date = trade_ts.dt.strftime("%Y-%m-%d")
    data["trade_date"] = normalized_trade_date
    bar_raw = data["bar_time"].fillna("").astype(str).str.strip()
    missing_bar_time = bar_raw.eq("") | bar_raw.str.lower().isin({"nan", "nat", "none"})
    parsed_has_clock = trade_ts.notna() & (trade_ts.dt.hour.ne(0) | trade_ts.dt.minute.ne(0) | trade_ts.dt.second.ne(0))
    fill_from_trade_date = missing_bar_time & parsed_has_clock
    if bool(fill_from_trade_date.any()):
        data.loc[fill_from_trade_date, "bar_time"] = trade_ts.loc[fill_from_trade_date].dt.strftime("%H:%M:%S")
    return data


def _bar_clock_int(value: Any) -> int:
    text = _normalize_bar_time(value)
    try:
        return int(text[:6])
    except (TypeError, ValueError):
        return 0


def _finite_sum(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.sum()) if len(values) else 0.0


def _first_finite(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.iloc[0]) if len(values) else np.nan


def _last_finite(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.iloc[-1]) if len(values) else np.nan


def _window_sum(series: pd.Series, bars: int, *, head: bool) -> float:
    data = pd.to_numeric(series, errors="coerce")
    window = data.head(int(bars)) if head else data.tail(int(bars))
    return _finite_sum(window)


def _window_range(day: pd.DataFrame, bars: int, *, head: bool) -> float:
    if day is None or day.empty or len(day) < int(bars):
        return np.nan
    window = day.head(int(bars)) if head else day.tail(int(bars))
    high = pd.to_numeric(window["high"], errors="coerce")
    low = pd.to_numeric(window["low"], errors="coerce")
    if not high.notna().any() or not low.notna().any():
        return np.nan
    return _safe_return(float(high.max()), float(low.min()))


def _first_extreme_position(series: pd.Series, *, mode: str) -> int:
    data = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if not data.notna().any():
        return -1
    target = data.max() if mode == "max" else data.min()
    matches = data.index[data.eq(target)]
    if len(matches) == 0:
        return -1
    first_match = matches[0]
    if isinstance(first_match, (int, np.integer)):
        return int(first_match)
    return int(data.index.get_loc(first_match))


def _position_fraction(position: int, count: int) -> float:
    if int(position) < 0 or int(count) <= 0:
        return np.nan
    if int(count) == 1:
        return 0.0
    return float(position) / float(int(count) - 1)


def _max_drawdown(series: pd.Series) -> float:
    data = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 2:
        return np.nan
    running_max = data.cummax()
    drawdown = data / running_max - 1.0
    return float(drawdown.min())


def _max_runup(series: pd.Series) -> float:
    data = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 2:
        return np.nan
    running_min = data.cummin()
    runup = data / running_min - 1.0
    return float(runup.max())


def _finite_ratio(mask: pd.Series) -> float:
    data = pd.Series(mask).dropna()
    if len(data) == 0:
        return np.nan
    return float(data.astype(bool).mean())


def _linear_slope(series: pd.Series) -> float:
    data = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 2:
        return np.nan
    x = np.arange(len(data), dtype=float)
    y = data.to_numpy(dtype=float)
    if not np.isfinite(y).all():
        return np.nan
    scale = abs(float(y[0])) if float(y[0]) != 0.0 else 1.0
    return float(np.polyfit(x, y / scale, 1)[0])


def _safe_return(close_value: Any, open_value: Any) -> float:
    close_float = pd.to_numeric(pd.Series([close_value]), errors="coerce").iloc[0]
    open_float = pd.to_numeric(pd.Series([open_value]), errors="coerce").iloc[0]
    if pd.isna(close_float) or pd.isna(open_float) or float(open_float) <= 0.0:
        return np.nan
    return float(close_float) / float(open_float) - 1.0


def _head_window_ret(day: pd.DataFrame, bars: int) -> float:
    if len(day) < int(bars):
        return np.nan
    return _safe_return(day["close"].iloc[int(bars) - 1], day["open"].iloc[0])


def _tail_window_ret(day: pd.DataFrame, bars: int) -> float:
    if len(day) < int(bars):
        return np.nan
    tail = day.tail(int(bars))
    return _safe_return(tail["close"].iloc[-1], tail["open"].iloc[0])


def _tail_close_to_previous_close_ret(day: pd.DataFrame) -> float:
    if len(day) < 2:
        return np.nan
    close_values = pd.to_numeric(day["close"], errors="coerce")
    return _safe_return(close_values.iloc[-1], close_values.iloc[-2])


def _session_return(day: pd.DataFrame) -> float:
    if day is None or day.empty:
        return np.nan
    return _safe_return(day["close"].iloc[-1], day["open"].iloc[0])
