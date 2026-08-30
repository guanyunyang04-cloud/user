"""Mootdx frames responsibilities."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from quantlab.data.domains.contracts.dispatch import normalize_domain_frame
from quantlab.data.domains.contracts.market import normalize_market_frame
from quantlab.data.domains.contracts.schema import DataDomain
from quantlab.data.provider_symbols import (
    is_mootdx_index_symbol as _is_mootdx_index_symbol,
)
from quantlab.data.provider_symbols import (
    mootdx_symbol as _mootdx_symbol,
)


def _mootdx_xdxr_domain_frame(
    raw: pd.DataFrame,
    *,
    domain: str,
    source: str,
    as_of_date: str,
) -> pd.DataFrame:
    """Convert TDX xdxr records while keeping ambiguous fields conservative.

    ``songzhuangu`` is a combined 送转 value in the TDX protocol, so it is
    deliberately not mislabeled as either bonus shares or capital-reserve
    transfers in the generic contract.  The lossless raw record is retained
    by QDP v3 and used by its dedicated corporate-action transform.
    """

    normalized_domain = str(domain)
    if raw is None or raw.empty:
        return normalize_domain_frame(
            pd.DataFrame(),
            domain=normalized_domain,
            source=source,
            as_of_date=as_of_date,
            require_columns=False,
        )
    frame = raw.copy()
    required = {"provider_symbol", "year", "month", "day", "category"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"mootdx_xdxr_missing_fields:{missing}")
    dates = pd.to_datetime(
        {
            "year": pd.to_numeric(frame["year"], errors="coerce"),
            "month": pd.to_numeric(frame["month"], errors="coerce"),
            "day": pd.to_numeric(frame["day"], errors="coerce"),
        },
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    category = pd.to_numeric(frame["category"], errors="coerce")
    if normalized_domain == DataDomain.CORPORATE_ACTIONS:
        mask = category.eq(1)
        selected = frame.loc[mask].copy()
        selected_dates = dates.loc[mask]
        generic = pd.DataFrame(
            {
                "symbol": selected["provider_symbol"],
                "trade_date": selected_dates,
                "ex_date": selected_dates,
                "action_type": selected.get("name", pd.Series(index=selected.index, dtype=str)),
                "cash_dividend_per_10": pd.to_numeric(selected.get("fenhong"), errors="coerce"),
                "bonus_share_per_10": np.nan,
                "transfer_share_per_10": np.nan,
                "description": "TDX xdxr; songzhuangu is combined and remains only in lossless raw evidence",
            },
            index=selected.index,
        )
    elif normalized_domain == DataDomain.SHARE_CAPITAL:
        total = (
            pd.to_numeric(frame.get("houzongguben", pd.Series(index=frame.index, dtype=float)), errors="coerce")
            * 10_000.0
        )
        floating = (
            pd.to_numeric(frame.get("panhouliutong", pd.Series(index=frame.index, dtype=float)), errors="coerce")
            * 10_000.0
        )
        mask = total.notna() | floating.notna()
        selected = frame.loc[mask].copy()
        generic = pd.DataFrame(
            {
                "symbol": selected["provider_symbol"],
                "trade_date": dates.loc[mask],
                "change_reason": selected.get("name", pd.Series(index=selected.index, dtype=str)),
                "total_share": total.loc[mask],
                "float_share": floating.loc[mask],
                "restricted_share": (total - floating).where(total.ge(floating)).loc[mask],
            },
            index=selected.index,
        )
    else:
        raise ValueError(f"mootdx_xdxr_unsupported_domain:{domain}")
    return normalize_domain_frame(
        generic.reset_index(drop=True),
        domain=normalized_domain,
        source=source,
        as_of_date=as_of_date,
        require_columns=False,
    )


def _fetch_mootdx_bars_window(
    *,
    client: Any,
    symbol: str,
    frequency: int,
    start_date: str,
    end_date: str,
    page_size: int,
    max_pages: int,
    source: str,
    adjusted_flag: str,
    volume_factor: float,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    expected_pages = _estimated_mootdx_pages(
        start_date=start_date,
        end_date=end_date,
        frequency=int(frequency),
        page_size=int(page_size),
        configured_max_pages=int(max_pages),
    )
    for page in range(expected_pages):
        offset_start = int(page) * int(page_size)
        payload = _call_mootdx_bars_endpoint(
            client=client,
            symbol=symbol,
            frequency=int(frequency),
            start=offset_start,
            offset=int(page_size),
        )
        raw = _mootdx_payload_frame(payload)
        if raw.empty:
            break
        prepared = _prepare_mootdx_bars_frame(
            raw,
            symbol=symbol,
            source=source,
            adjusted_flag=adjusted_flag,
            volume_factor=float(volume_factor),
        )
        if prepared.empty:
            break
        frames.append(prepared)
        dates = pd.to_datetime(
            prepared["datetime"] if "datetime" in prepared.columns else prepared["trade_date"], errors="coerce"
        ).dropna()
        if (
            len(dates)
            and pd.Timestamp(dates.min()).normalize() <= start_ts.normalize()
            and pd.Timestamp(dates.max()).normalize() >= end_ts.normalize()
        ):
            break
        if len(raw) < int(page_size):
            break
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return _filter_domain_date_window(combined, start_date, end_date)


def _estimated_mootdx_pages(
    *, start_date: str, end_date: str, frequency: int, page_size: int, configured_max_pages: int
) -> int:
    page_size = max(1, int(page_size or 1))
    configured = max(1, int(configured_max_pages or 1))
    if int(frequency) == 8:
        bars_per_day = 240
    elif int(frequency) == 0:
        bars_per_day = 48
    else:
        bars_per_day = 1
    try:
        # mootdx bars are paged backward from the quote server's latest bar, not from
        # the requested end_date. Historical chunks therefore need enough pages to
        # reach start_date from "now", even when the chunk end_date is only a few
        # days after start_date.
        lookback_end = max(pd.Timestamp(end_date).normalize(), pd.Timestamp.today().normalize())
        business_days = max(1, len(pd.bdate_range(pd.Timestamp(start_date), lookback_end)))
    except Exception:
        return configured
    estimated_rows = int(business_days * bars_per_day)
    return min(240, max(configured, int(math.ceil(estimated_rows / page_size)) + 2))


def _call_mootdx_bars_endpoint(*, client: Any, symbol: str, frequency: int, start: int, offset: int) -> Any:
    method = getattr(client, "index_bars", None) if _is_mootdx_index_symbol(symbol) else None
    if not callable(method):
        method = getattr(client, "bars", None)
    if not callable(method):
        raise RuntimeError("mootdx client does not expose bars/index_bars")
    return method(symbol=_mootdx_symbol(symbol), frequency=int(frequency), start=int(start), offset=int(offset))


def _mootdx_payload_frame(payload: Any) -> pd.DataFrame:
    if isinstance(payload, pd.DataFrame):
        frame = payload.copy()
    else:
        frame = pd.DataFrame(payload)
    if frame.empty:
        return frame
    if "datetime" not in frame.columns and isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.copy()
        frame["datetime"] = frame.index
    return frame.reset_index(drop=True)


def _prepare_mootdx_bars_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    source: str,
    adjusted_flag: str,
    volume_factor: float,
) -> pd.DataFrame:
    data = frame.copy()
    data["symbol"] = str(symbol).strip().upper()
    data["source"] = source
    data["adjusted_flag"] = str(adjusted_flag or "none")
    if "trade_date" not in data.columns and "datetime" in data.columns:
        data["trade_date"] = data["datetime"]
    if "datetime" in data.columns:
        parsed_datetime = pd.to_datetime(data["datetime"], errors="coerce")
        if parsed_datetime.notna().any():
            data["bar_time"] = parsed_datetime.dt.strftime("%H:%M:%S")
    volume_source = None
    for candidate in ("volume", "vol", "成交量"):
        if candidate in data.columns:
            volume_source = candidate
            break
    if volume_source is not None:
        data["volume"] = pd.to_numeric(data[volume_source], errors="coerce") * float(volume_factor)
    return data


def _normalize_mootdx_quote_snapshot(
    frame: pd.DataFrame,
    *,
    symbols: tuple[str, ...],
    source: str,
    volume_factor: float,
) -> pd.DataFrame:
    data = frame.copy()
    code_to_symbol = {_mootdx_symbol(symbol): symbol for symbol in symbols}
    if "symbol" not in data.columns:
        if "code" in data.columns:
            data["symbol"] = data["code"].astype(str).str.zfill(6).map(code_to_symbol).fillna(data["code"].astype(str))
        elif len(data) == len(symbols):
            data["symbol"] = list(symbols)
    data["trade_date"] = pd.Timestamp.now().strftime("%Y-%m-%d")
    if "close" not in data.columns and "price" in data.columns:
        data["close"] = data["price"]
    volume_source = None
    for candidate in ("volume", "vol", "成交量"):
        if candidate in data.columns:
            volume_source = candidate
            break
    if volume_source is not None:
        data["volume"] = pd.to_numeric(data[volume_source], errors="coerce") * float(volume_factor)
    data["source"] = source
    data["adjusted_flag"] = "none"
    return normalize_market_frame(data, source=source, adjusted_flag="none", require_columns=False)


def _filter_domain_date_window(data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if data is None or data.empty or "trade_date" not in data.columns:
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame()
    dates = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    mask = dates.ge(pd.Timestamp(start_date).normalize()) & dates.le(pd.Timestamp(end_date).normalize())
    return data.loc[mask].reset_index(drop=True)
