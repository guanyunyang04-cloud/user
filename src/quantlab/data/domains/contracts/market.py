"""Domain contract market definitions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import (
    _column_aliases,
    _date_series,
    _ensure_domain_columns,
    _normalize_symbol,
    _prepare_domain_frame,
    _rename_first,
    _source_series,
    validate_provider_name,
)
from .schema import (
    DOMAIN_STANDARD_COLUMNS,
    NUMERIC_MARKET_COLUMNS,
    PRICE_COLUMNS,
    STANDARD_MARKET_COLUMNS,
    DataDomain,
)


def normalize_market_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    adjusted_flag: str = "none",
    require_columns: bool = True,
) -> pd.DataFrame:
    provider = validate_provider_name(source)
    if frame is None or frame.empty:
        return pd.DataFrame(columns=STANDARD_MARKET_COLUMNS)
    working = frame.copy()
    rename_map: dict[str, str] = {}
    aliases = _column_aliases()
    lower_lookup = {str(column).strip().lower(): column for column in working.columns}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            if candidate.lower() in lower_lookup:
                rename_map[lower_lookup[candidate.lower()]] = canonical
                break
    working = working.rename(columns=rename_map)
    if "source" not in working.columns:
        working["source"] = provider
    if "adjusted_flag" not in working.columns:
        working["adjusted_flag"] = str(adjusted_flag or "none")
    required = {"symbol", "trade_date", *NUMERIC_MARKET_COLUMNS}
    missing = sorted(required - set(working.columns))
    if missing and require_columns:
        raise ValueError(f"provider_frame_schema_error: missing standard market columns {missing}")
    for column in missing:
        working[column] = np.nan
    working["symbol"] = working["symbol"].map(_normalize_symbol)
    working["trade_date"] = pd.to_datetime(working["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in NUMERIC_MARKET_COLUMNS:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working["source"] = working["source"].fillna(provider).astype(str).str.strip().str.lower().replace("", provider)
    working["adjusted_flag"] = (
        working["adjusted_flag"].fillna(str(adjusted_flag or "none")).astype(str).str.strip().replace("", "none")
    )
    out = working.loc[:, [column for column in STANDARD_MARKET_COLUMNS if column in working.columns]].copy()
    for column in STANDARD_MARKET_COLUMNS:
        if column not in out.columns:
            out[column] = "" if column in {"symbol", "trade_date", "source", "adjusted_flag"} else np.nan
    out = out[STANDARD_MARKET_COLUMNS]
    out = out.loc[out["symbol"].astype(str).str.len() > 0]
    out = out.loc[out["trade_date"].astype(str).str.lower() != "nat"]
    return out.sort_values(["trade_date", "symbol", "source"]).reset_index(drop=True)


def normalize_adjust_factor_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    require_columns: bool = True,
) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(
        frame, domain=DataDomain.ADJUST_FACTOR, source=provider, as_of_date="", require_columns=False
    )
    _rename_first(data, "symbol", ("code", "ts_code", "股票代码", "证券代码"))
    _rename_first(data, "trade_date", ("dividOperateDate", "date", "日期", "除权除息日"))
    _rename_first(data, "fore_adjust_factor", ("foreAdjustFactor", "qfq_factor", "前复权因子"))
    _rename_first(data, "back_adjust_factor", ("backAdjustFactor", "hfq_factor", "后复权因子"))
    _rename_first(data, "adjust_factor", ("adjustFactor", "factor", "复权因子"))
    if "factor_provider" not in data.columns:
        data["factor_provider"] = provider
    if "factor_semantics" not in data.columns:
        data["factor_semantics"] = "raw_provider_factor"
    _require_core_columns(data, DataDomain.ADJUST_FACTOR, {"symbol", "trade_date"}, require_columns=require_columns)
    data = _ensure_domain_columns(data, DataDomain.ADJUST_FACTOR)
    data["symbol"] = data["symbol"].map(_normalize_symbol)
    data["trade_date"] = _date_series(data["trade_date"])
    for column in ("fore_adjust_factor", "back_adjust_factor", "adjust_factor"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["factor_provider"] = data["factor_provider"].fillna(provider).astype(str).str.strip().replace("", provider)
    data["factor_semantics"] = (
        data["factor_semantics"]
        .fillna("raw_provider_factor")
        .astype(str)
        .str.strip()
        .replace("", "raw_provider_factor")
    )
    data["source"] = _source_series(data, provider)
    out = data.loc[
        data["symbol"].astype(str).str.len().gt(0) & data["trade_date"].astype(str).str.lower().ne("nat"),
        DOMAIN_STANDARD_COLUMNS[DataDomain.ADJUST_FACTOR],
    ]
    return (
        out.drop_duplicates(subset=["trade_date", "symbol", "factor_provider", "source"])
        .sort_values(["trade_date", "symbol", "factor_provider", "source"])
        .reset_index(drop=True)
    )


def valid_market_rows(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series([], dtype=bool)
    data = normalize_market_frame(
        frame,
        source=str(frame["source"].iloc[0] if "source" in frame.columns and len(frame) else "unknown"),
        require_columns=False,
    )
    price_positive = data[PRICE_COLUMNS].gt(0).all(axis=1)
    range_valid = data["high"].ge(data[["open", "close", "low"]].max(axis=1)) & data["low"].le(
        data[["open", "close", "high"]].min(axis=1)
    )
    activity_valid = data[["volume", "amount"]].ge(0).all(axis=1)
    return (price_positive & range_valid & activity_valid).reindex(data.index, fill_value=False)


def _require_core_columns(data: pd.DataFrame, domain: str, required: set[str], *, require_columns: bool) -> None:
    if not require_columns:
        return
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"provider_frame_schema_error: domain={domain} missing core columns {missing}")
