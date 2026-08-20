"""Deterministic transformations used by the PIT history restore flow."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.data.core.security_status import st_status_from_name
from quantlab.data.identifiers import (
    identity_exchange,
    security_id,
    short_exchange,
)

MAINBOARD_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")


def is_mainboard(symbol: object) -> bool:
    text = str(symbol or "").strip().upper()
    code = text.split(".", 1)[0]
    return text.endswith((".SH", ".SZ")) and code.startswith(MAINBOARD_PREFIXES)


def normalize_eastmoney_history(raw: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    columns = [
        "trade_date", "symbol", "open", "high", "low", "close", "volume",
        "amount", "tradestatus", "isST", "turn", "pctChg", "peTTM",
        "pbMRQ", "psTTM", "pcfNcfTTM", "name_on_date", "history_source",
    ]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=columns)
    data = raw.rename(
        columns={
            "日期": "trade_date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turn",
            "涨跌幅": "pctChg",
        }
    ).copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["symbol"] = symbol
    for column in (
        "open", "high", "low", "close", "volume", "amount", "turn", "pctChg"
    ):
        data[column] = pd.to_numeric(data.get(column), errors="coerce")
    # Eastmoney reports A-share daily volume in lots; QDP and BaoStock use shares.
    data["volume"] = data["volume"] * 100.0
    data["tradestatus"] = "1"
    data["isST"] = ""
    for column in ("peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"):
        data[column] = np.nan
    data["name_on_date"] = ""
    data["history_source"] = "akshare_eastmoney_unadjusted_history"
    return (
        data.loc[data["trade_date"].notna(), columns]
        .drop_duplicates("trade_date", keep="last")
        .sort_values("trade_date")
        .reset_index(drop=True)
    )


def normalize_sina_factors(raw: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    columns = [
        "symbol", "trade_date", "fore_adjust_factor", "back_adjust_factor",
        "adjust_factor", "factor_provider", "source",
    ]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=columns)
    data = raw.rename(columns={"date": "trade_date", "hfq_factor": "factor"}).copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["factor"] = pd.to_numeric(data["factor"], errors="coerce")
    data = data.loc[data["trade_date"].notna() & data["factor"].gt(0)].copy()
    data["symbol"] = symbol
    data["fore_adjust_factor"] = data["factor"]
    data["back_adjust_factor"] = data["factor"]
    data["adjust_factor"] = data["factor"]
    data["factor_provider"] = "sina_via_akshare"
    data["source"] = "akshare_sina_hfq_factor_event"
    return data.loc[:, columns].sort_values("trade_date").reset_index(drop=True)


def historical_names(
    dates: pd.Series,
    intervals: pd.DataFrame,
    *,
    fallback: str,
) -> pd.Series:
    result = pd.Series(str(fallback), index=dates.index, dtype="object")
    if intervals.empty:
        return result
    normalized = intervals.copy()
    normalized["start_date"] = pd.to_datetime(normalized["start_date"], errors="coerce")
    normalized["end_date"] = pd.to_datetime(normalized["end_date"], errors="coerce")
    normalized = normalized.dropna(subset=["start_date"]).sort_values("start_date")
    date_values = pd.to_datetime(dates, errors="coerce")
    for row in normalized.itertuples(index=False):
        start = pd.Timestamp(row.start_date)
        end = pd.Timestamp(row.end_date) if pd.notna(row.end_date) else pd.Timestamp.max
        mask = date_values.ge(start) & date_values.le(end)
        result.loc[mask] = str(row.name)
    return result


def name_implies_st(names: pd.Series) -> pd.Series:
    return names.map(st_status_from_name).astype("boolean")


def factor_rows(history: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(history["trade_date"], errors="raise")
    event = events.copy()
    factor_provider = "identity_no_factor_event"
    factor_source = "identity_factor_pit_history_restore"
    if event.empty:
        factor = np.ones(len(history), dtype=np.float64)
        source_dates = dates.copy()
    else:
        event["trade_date"] = pd.to_datetime(event["trade_date"], errors="coerce")
        event["back_adjust_factor"] = pd.to_numeric(
            event["back_adjust_factor"], errors="coerce"
        )
        event = event.dropna(subset=["trade_date", "back_adjust_factor"])
        event = event.loc[event["back_adjust_factor"].gt(0)].sort_values("trade_date")
        if event.empty:
            factor = np.ones(len(history), dtype=np.float64)
            source_dates = dates.copy()
        else:
            factor_provider = str(
                event.get("factor_provider", pd.Series(["sina_via_akshare"])).iloc[0]
            )
            factor_source = str(
                event.get("source", pd.Series(["akshare_sina_hfq_factor_event"])).iloc[0]
            )
            event_dates = event["trade_date"].to_numpy(dtype="datetime64[ns]")
            event_values = event["back_adjust_factor"].to_numpy(dtype=np.float64)
            position = np.searchsorted(
                event_dates,
                dates.to_numpy(dtype="datetime64[ns]"),
                side="right",
            ) - 1
            first_position = int(position[0])
            baseline = float(event_values[first_position]) if first_position >= 0 else 1.0
            current = np.where(
                position >= 0,
                event_values[np.maximum(position, 0)],
                baseline,
            )
            factor = current / baseline
            source_values = np.where(
                position >= 0,
                event_dates[np.maximum(position, 0)],
                dates.to_numpy(dtype="datetime64[ns]"),
            )
            source_dates = pd.Series(
                pd.to_datetime(source_values), index=history.index
            )
    symbol = str(history["symbol"].iloc[0])
    trade_date = history["trade_date"].astype(str)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": trade_date,
            "fore_adjust_factor": factor,
            "back_adjust_factor": factor,
            "adjust_factor": factor,
            "factor_provider": factor_provider,
            "factor_semantics": "sina_hfq_factor_ratio_normalized_to_first_qdp_observation",
            "source": factor_source + "+pit_history_restore",
            "factor_source_date": source_dates.dt.strftime("%Y-%m-%d"),
            "ffill_days": (dates - source_dates).dt.days.astype("int64"),
        }
    )


__all__ = [
    "factor_rows",
    "historical_names",
    "identity_exchange",
    "is_mainboard",
    "name_implies_st",
    "normalize_eastmoney_history",
    "normalize_sina_factors",
    "security_id",
    "short_exchange",
]
