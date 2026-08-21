"""Domain contract snapshots definitions."""

from __future__ import annotations

import pandas as pd

from .base import (
    _coerce_trade_date,
    _date_series,
    _ensure_domain_columns,
    _infer_exchange,
    _normalize_symbol,
    _prepare_domain_frame,
    _rename_first,
    _require_domain_columns,
    _source_series,
    _tag_value,
    _to_bool,
    normalize_domain,
    validate_provider_name,
)
from .schema import (
    DOMAIN_STANDARD_COLUMNS,
    DataDomain,
)


def normalize_calendar_frame(frame: pd.DataFrame, *, source: str, require_columns: bool = True) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(
        frame, domain=DataDomain.TRADING_CALENDAR, source=provider, as_of_date="", require_columns=False
    )
    if data.empty:
        return data
    if "trade_date" not in data.columns:
        _rename_first(data, "trade_date", ("date", "cal_date", "calendar_date", "交易日期", "日期"))
    if "is_open" not in data.columns:
        _rename_first(data, "is_open", ("is_trading_day", "is_trade", "open", "交易", "是否交易"))
    if "exchange" not in data.columns:
        data["exchange"] = "SSE"
    _require_domain_columns(data, DataDomain.TRADING_CALENDAR, require_columns=require_columns)
    data = _ensure_domain_columns(data, DataDomain.TRADING_CALENDAR)
    data["trade_date"] = _date_series(data["trade_date"])
    data["is_open"] = data["is_open"].map(_to_bool)
    data["exchange"] = data["exchange"].fillna("SSE").astype(str).str.strip().replace("", "SSE")
    data["source"] = _source_series(data, provider)
    return (
        data.loc[
            data["trade_date"].astype(str).str.lower() != "nat", DOMAIN_STANDARD_COLUMNS[DataDomain.TRADING_CALENDAR]
        ]
        .drop_duplicates()
        .sort_values(["trade_date", "exchange"])
        .reset_index(drop=True)
    )


def normalize_universe_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    as_of_date: str,
    require_columns: bool = True,
) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(
        frame, domain=DataDomain.UNIVERSE_SNAPSHOT, source=provider, as_of_date=as_of_date, require_columns=False
    )
    _rename_first(data, "symbol", ("code", "ts_code", "股票代码", "证券代码"))
    _rename_first(data, "name", ("stock_name", "名称", "股票简称", "证券简称"))
    _rename_first(data, "list_status", ("status", "上市状态"))
    _require_domain_columns(data, DataDomain.UNIVERSE_SNAPSHOT, require_columns=require_columns)
    data = _ensure_domain_columns(data, DataDomain.UNIVERSE_SNAPSHOT)
    data["symbol"] = data["symbol"].map(_normalize_symbol)
    data["trade_date"] = _coerce_trade_date(data["trade_date"], as_of_date)
    data["exchange"] = data.apply(lambda row: _infer_exchange(row.get("symbol", ""), row.get("exchange", "")), axis=1)
    for column in ["name", "board", "list_status", "list_date", "delist_date"]:
        data[column] = data[column].fillna("").astype(str).str.strip()
    data["source"] = _source_series(data, provider)
    out = data.loc[data["symbol"].astype(str).str.len() > 0, DOMAIN_STANDARD_COLUMNS[DataDomain.UNIVERSE_SNAPSHOT]]
    return (
        out.drop_duplicates(subset=["trade_date", "symbol", "source"])
        .sort_values(["trade_date", "symbol", "source"])
        .reset_index(drop=True)
    )


def normalize_status_frame(
    frame: pd.DataFrame, *, source: str, as_of_date: str, require_columns: bool = True
) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(
        frame, domain=DataDomain.SECURITY_STATUS, source=provider, as_of_date=as_of_date, require_columns=False
    )
    _rename_first(data, "symbol", ("code", "ts_code", "股票代码", "证券代码"))
    _require_domain_columns(data, DataDomain.SECURITY_STATUS, require_columns=require_columns)
    data = _ensure_domain_columns(data, DataDomain.SECURITY_STATUS)
    data["symbol"] = data["symbol"].map(_normalize_symbol)
    data["trade_date"] = _coerce_trade_date(data["trade_date"], as_of_date)
    for column in ["is_st", "is_suspended", "is_delisted"]:
        data[column] = data[column].map(_to_bool)
    data["status_reason"] = data["status_reason"].fillna("").astype(str)
    data["source"] = _source_series(data, provider)
    return (
        data.loc[data["symbol"].astype(str).str.len() > 0, DOMAIN_STANDARD_COLUMNS[DataDomain.SECURITY_STATUS]]
        .drop_duplicates()
        .sort_values(["trade_date", "symbol"])
        .reset_index(drop=True)
    )


def normalize_limit_frame(
    frame: pd.DataFrame, *, source: str, as_of_date: str, require_columns: bool = True
) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(
        frame, domain=DataDomain.LIMIT_STATUS, source=provider, as_of_date=as_of_date, require_columns=False
    )
    _rename_first(data, "symbol", ("code", "ts_code", "股票代码", "证券代码"))
    _rename_first(data, "up_limit", ("limit_up", "涨停价", "涨停"))
    _rename_first(data, "down_limit", ("limit_down", "跌停价", "跌停"))
    _require_domain_columns(data, DataDomain.LIMIT_STATUS, require_columns=require_columns)
    data = _ensure_domain_columns(data, DataDomain.LIMIT_STATUS)
    data["symbol"] = data["symbol"].map(_normalize_symbol)
    data["trade_date"] = _coerce_trade_date(data["trade_date"], as_of_date)
    for column in ["up_limit", "down_limit"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in ["is_limit_up", "is_limit_down"]:
        data[column] = data[column].map(_to_bool)
    data["source"] = _source_series(data, provider)
    return (
        data.loc[data["symbol"].astype(str).str.len() > 0, DOMAIN_STANDARD_COLUMNS[DataDomain.LIMIT_STATUS]]
        .drop_duplicates()
        .sort_values(["trade_date", "symbol"])
        .reset_index(drop=True)
    )


def normalize_industry_concept_frame(
    frame: pd.DataFrame, *, source: str, as_of_date: str, require_columns: bool = True
) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(
        frame, domain=DataDomain.INDUSTRY_CONCEPT, source=provider, as_of_date=as_of_date, require_columns=False
    )
    _rename_first(data, "symbol", ("code", "ts_code", "股票代码", "证券代码"))
    _rename_first(data, "industry", ("行业", "所属行业"))
    _rename_first(data, "concept_tags", ("concept", "concepts", "概念", "概念标签"))
    _require_domain_columns(data, DataDomain.INDUSTRY_CONCEPT, require_columns=require_columns)
    data = _ensure_domain_columns(data, DataDomain.INDUSTRY_CONCEPT)
    data["symbol"] = data["symbol"].map(_normalize_symbol)
    data["trade_date"] = _coerce_trade_date(data["trade_date"], as_of_date)
    data["industry"] = data["industry"].fillna("").astype(str).str.strip()
    data["concept_tags"] = data["concept_tags"].map(_tag_value)
    data["source"] = _source_series(data, provider)
    return (
        data.loc[data["symbol"].astype(str).str.len() > 0, DOMAIN_STANDARD_COLUMNS[DataDomain.INDUSTRY_CONCEPT]]
        .drop_duplicates()
        .sort_values(["trade_date", "symbol"])
        .reset_index(drop=True)
    )


def normalize_valuation_frame(
    frame: pd.DataFrame, *, source: str, as_of_date: str, require_columns: bool = True
) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(
        frame, domain=DataDomain.VALUATION, source=provider, as_of_date=as_of_date, require_columns=False
    )
    _rename_first(data, "symbol", ("code", "ts_code", "股票代码", "证券代码"))
    _rename_first(data, "total_mv", ("market_cap", "total_market_cap", "总市值", "总市值-元"))
    _rename_first(data, "circ_mv", ("float_market_cap", "circulating_market_cap", "流通市值"))
    _rename_first(data, "pe", ("pe_ttm", "市盈率", "市盈率-动态"))
    _rename_first(data, "pb", ("市净率",))
    _rename_first(data, "turnover_rate", ("turnover", "换手率"))
    _require_domain_columns(data, DataDomain.VALUATION, require_columns=require_columns)
    data = _ensure_domain_columns(data, DataDomain.VALUATION)
    data["symbol"] = data["symbol"].map(_normalize_symbol)
    data["trade_date"] = _coerce_trade_date(data["trade_date"], as_of_date)
    for column in ["total_mv", "circ_mv", "pe", "pb", "turnover_rate"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["source"] = _source_series(data, provider)
    return (
        data.loc[data["symbol"].astype(str).str.len() > 0, DOMAIN_STANDARD_COLUMNS[DataDomain.VALUATION]]
        .drop_duplicates()
        .sort_values(["trade_date", "symbol"])
        .reset_index(drop=True)
    )


def normalize_index_constituents_frame(
    frame: pd.DataFrame, *, source: str, as_of_date: str, require_columns: bool = True
) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(
        frame, domain=DataDomain.INDEX_CONSTITUENTS, source=provider, as_of_date=as_of_date, require_columns=False
    )
    _rename_first(data, "index_symbol", ("index_code", "指数代码", "指数"))
    _rename_first(data, "symbol", ("code", "stock", "stock_code", "证券代码", "股票代码"))
    _rename_first(data, "index_name", ("index", "index_short_name", "指数名称", "指数简称"))
    _require_domain_columns(data, DataDomain.INDEX_CONSTITUENTS, require_columns=require_columns)
    data = _ensure_domain_columns(data, DataDomain.INDEX_CONSTITUENTS)
    data["symbol"] = data["symbol"].map(_normalize_symbol)
    data["index_symbol"] = data["index_symbol"].map(_normalize_symbol)
    data["trade_date"] = _coerce_trade_date(data["trade_date"], as_of_date)
    data["index_name"] = data["index_name"].fillna("").astype(str).str.strip()
    data["source"] = _source_series(data, provider)
    return (
        data.loc[
            data["symbol"].astype(str).str.len().gt(0) & data["index_symbol"].astype(str).str.len().gt(0),
            DOMAIN_STANDARD_COLUMNS[DataDomain.INDEX_CONSTITUENTS],
        ]
        .drop_duplicates(subset=["trade_date", "index_symbol", "symbol", "source"])
        .sort_values(["trade_date", "index_symbol", "symbol"])
        .reset_index(drop=True)
    )


def normalize_money_flow_hotspot_frame(
    frame: pd.DataFrame, *, source: str, as_of_date: str, require_columns: bool = True
) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(
        frame, domain=DataDomain.MONEY_FLOW_HOTSPOT, source=provider, as_of_date=as_of_date, require_columns=False
    )
    _rename_first(data, "symbol", ("code", "ts_code", "股票代码", "证券代码"))
    _rename_first(data, "main_net_inflow", ("net_inflow", "主力净流入", "资金净流入"))
    _rename_first(data, "sector_rank", ("rank", "板块排名"))
    _rename_first(data, "hotspot_tags", ("hotspot", "hotspots", "热点", "题材"))
    _require_domain_columns(data, DataDomain.MONEY_FLOW_HOTSPOT, require_columns=require_columns)
    data = _ensure_domain_columns(data, DataDomain.MONEY_FLOW_HOTSPOT)
    data["symbol"] = data["symbol"].map(_normalize_symbol)
    data["trade_date"] = _coerce_trade_date(data["trade_date"], as_of_date)
    data["main_net_inflow"] = pd.to_numeric(data["main_net_inflow"], errors="coerce")
    data["sector_rank"] = pd.to_numeric(data["sector_rank"], errors="coerce")
    data["hotspot_tags"] = data["hotspot_tags"].map(_tag_value)
    data["source"] = _source_series(data, provider)
    return (
        data.loc[data["symbol"].astype(str).str.len() > 0, DOMAIN_STANDARD_COLUMNS[DataDomain.MONEY_FLOW_HOTSPOT]]
        .drop_duplicates()
        .sort_values(["trade_date", "symbol"])
        .reset_index(drop=True)
    )


def normalize_generic_text_domain_frame(
    frame: pd.DataFrame,
    *,
    domain: str,
    source: str,
    as_of_date: str,
    require_columns: bool = True,
) -> pd.DataFrame:
    normalized_domain = normalize_domain(domain)
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(
        frame, domain=normalized_domain, source=provider, as_of_date=as_of_date, require_columns=False
    )
    if data.empty:
        return data
    if "trade_date" not in data.columns:
        _rename_first(data, "trade_date", ("date", "publish_date", "公告日期", "日期"))
    if "symbol" not in data.columns:
        _rename_first(data, "symbol", ("code", "stock", "ts_code", "证券代码", "股票代码"))
    _require_domain_columns(data, normalized_domain, require_columns=require_columns)
    data = _ensure_domain_columns(data, normalized_domain)
    data["symbol"] = data["symbol"].map(_normalize_symbol)
    data["trade_date"] = _date_series(data["trade_date"]).fillna(pd.Timestamp(as_of_date))
    data["source"] = _source_series(data, provider)
    return (
        data.loc[data["trade_date"].astype(str).str.lower() != "nat", DOMAIN_STANDARD_COLUMNS[normalized_domain]]
        .drop_duplicates()
        .sort_values(["trade_date", "symbol"])
        .reset_index(drop=True)
    )
