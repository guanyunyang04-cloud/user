from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol

import numpy as np
import pandas as pd


STANDARD_MARKET_COLUMNS = [
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "source",
    "adjusted_flag",
]

PRICE_COLUMNS = ["open", "high", "low", "close"]
NUMERIC_MARKET_COLUMNS = [*PRICE_COLUMNS, "volume", "amount"]
TDX_FAMILY_PROVIDER_NAMES = frozenset({"tq", "tqcenter", "tdx", "pytdx", "mootdx"})


class DataDomain:
    MARKET_DAILY = "market_daily"
    TRADING_CALENDAR = "trading_calendar"
    UNIVERSE_SNAPSHOT = "universe_snapshot"
    SECURITY_STATUS = "security_status"
    LIMIT_STATUS = "limit_status"
    INDUSTRY_CONCEPT = "industry_concept"
    VALUATION = "valuation"
    MONEY_FLOW_HOTSPOT = "money_flow_hotspot"


DOMAIN_STANDARD_COLUMNS: dict[str, list[str]] = {
    DataDomain.MARKET_DAILY: STANDARD_MARKET_COLUMNS,
    DataDomain.TRADING_CALENDAR: ["trade_date", "is_open", "exchange", "source"],
    DataDomain.UNIVERSE_SNAPSHOT: [
        "symbol",
        "trade_date",
        "name",
        "exchange",
        "board",
        "list_status",
        "list_date",
        "delist_date",
        "source",
    ],
    DataDomain.SECURITY_STATUS: [
        "symbol",
        "trade_date",
        "is_st",
        "is_suspended",
        "is_delisted",
        "status_reason",
        "source",
    ],
    DataDomain.LIMIT_STATUS: [
        "symbol",
        "trade_date",
        "up_limit",
        "down_limit",
        "is_limit_up",
        "is_limit_down",
        "source",
    ],
    DataDomain.INDUSTRY_CONCEPT: ["symbol", "trade_date", "industry", "concept_tags", "source"],
    DataDomain.VALUATION: ["symbol", "trade_date", "total_mv", "circ_mv", "pe", "pb", "turnover_rate", "source"],
    DataDomain.MONEY_FLOW_HOTSPOT: [
        "symbol",
        "trade_date",
        "main_net_inflow",
        "sector_rank",
        "hotspot_tags",
        "source",
    ],
}


@dataclass(frozen=True)
class FetchRequest:
    symbols: tuple[str, ...]
    start_date: str
    end_date: str
    domain: str = "market_daily"
    adjusted_flag: str = "none"
    fields: tuple[str, ...] = tuple(STANDARD_MARKET_COLUMNS)

    def normalized(self) -> "FetchRequest":
        return FetchRequest(
            symbols=tuple(_normalize_symbol(item) for item in self.symbols if str(item or "").strip()),
            start_date=_normalize_date(self.start_date),
            end_date=_normalize_date(self.end_date),
            domain=normalize_domain(self.domain),
            adjusted_flag=str(self.adjusted_flag or "none"),
            fields=tuple(self.fields or tuple(STANDARD_MARKET_COLUMNS)),
        )


@dataclass(frozen=True)
class DomainFetchRequest:
    domain: str
    symbols: tuple[str, ...] = ()
    start_date: str = ""
    end_date: str = ""
    adjusted_flag: str = "none"
    fields: tuple[str, ...] = ()
    exchange: str = "SSE"

    def normalized(self) -> "DomainFetchRequest":
        domain = normalize_domain(self.domain)
        fields = tuple(self.fields or tuple(DOMAIN_STANDARD_COLUMNS.get(domain, ())))
        return DomainFetchRequest(
            domain=domain,
            symbols=tuple(_normalize_symbol(item) for item in self.symbols if str(item or "").strip()),
            start_date=_normalize_date(self.start_date),
            end_date=_normalize_date(self.end_date),
            adjusted_flag=str(self.adjusted_flag or "none"),
            fields=fields,
            exchange=str(self.exchange or "SSE").strip().upper(),
        )


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    data: pd.DataFrame
    coverage_report: dict[str, Any] = field(default_factory=dict)
    error_report: list[dict[str, Any]] = field(default_factory=list)


DomainProviderResult = ProviderResult


class MarketProvider(Protocol):
    name: str

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        ...


class DomainProvider(Protocol):
    name: str

    def fetch_domain(self, request: DomainFetchRequest) -> DomainProviderResult:
        ...


MarketProviderCallable = Callable[[FetchRequest], ProviderResult]


def normalize_domain(domain: str) -> str:
    normalized = str(domain or DataDomain.MARKET_DAILY).strip().lower()
    aliases = {
        "market": DataDomain.MARKET_DAILY,
        "market_bars": DataDomain.MARKET_DAILY,
        "daily": DataDomain.MARKET_DAILY,
        "calendar": DataDomain.TRADING_CALENDAR,
        "trade_calendar": DataDomain.TRADING_CALENDAR,
        "universe": DataDomain.UNIVERSE_SNAPSHOT,
        "stock_basic": DataDomain.UNIVERSE_SNAPSHOT,
        "status": DataDomain.SECURITY_STATUS,
        "security": DataDomain.SECURITY_STATUS,
        "limit": DataDomain.LIMIT_STATUS,
        "limit_up_down": DataDomain.LIMIT_STATUS,
        "industry": DataDomain.INDUSTRY_CONCEPT,
        "concept": DataDomain.INDUSTRY_CONCEPT,
        "daily_basic": DataDomain.VALUATION,
        "money_flow": DataDomain.MONEY_FLOW_HOTSPOT,
        "hotspot": DataDomain.MONEY_FLOW_HOTSPOT,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in DOMAIN_STANDARD_COLUMNS:
        raise ValueError(f"unsupported data domain: {domain}")
    return normalized


def validate_provider_name(provider_name: str, *, allow_tdx_family: bool = False) -> str:
    normalized = str(provider_name or "").strip().lower()
    if not normalized:
        raise ValueError("provider name cannot be empty")
    if normalized in TDX_FAMILY_PROVIDER_NAMES and not allow_tdx_family:
        raise ValueError(
            "TDX-family provider is disabled by default. "
            f"provider={provider_name}; use daily_research.data_platform.refresh_daily with non-TDX providers."
        )
    return normalized


def ensure_tdx_free_data_source(data_source: str, *, allow_legacy: bool = False) -> str:
    normalized = str(data_source or "lake").strip().lower()
    aliases = {
        "data_lake": "lake",
        "csv_imported_lake": "lake",
        "csv_lake": "lake",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in TDX_FAMILY_PROVIDER_NAMES and not allow_legacy:
        raise ValueError(
            "TDX-family data_source is no longer allowed in formal daily_research paths. "
            "Run `python -m daily_research.data_platform.refresh_daily ...` to update the lake, "
            "then pass `--data-source lake --lake-dataset-id <explicit_id>`."
        )
    return normalized


def market_business_dates(start_date: str, end_date: str) -> list[str]:
    start_ts = pd.Timestamp(_normalize_date(start_date))
    end_ts = pd.Timestamp(_normalize_date(end_date))
    if end_ts < start_ts:
        return []
    return [pd.Timestamp(item).strftime("%Y-%m-%d") for item in pd.bdate_range(start_ts, end_ts)]


def trading_dates_from_calendar(calendar: pd.DataFrame, start_date: str, end_date: str) -> list[str]:
    if calendar is None or calendar.empty:
        return market_business_dates(start_date, end_date)
    data = normalize_calendar_frame(calendar, source=str(calendar["source"].iloc[0] if "source" in calendar.columns and len(calendar) else "calendar"), require_columns=False)
    start_ts = pd.Timestamp(_normalize_date(start_date))
    end_ts = pd.Timestamp(_normalize_date(end_date))
    dates = pd.to_datetime(data.loc[data["is_open"].astype(bool), "trade_date"], errors="coerce").dropna()
    dates = dates.loc[(dates >= start_ts) & (dates <= end_ts)]
    return [pd.Timestamp(item).strftime("%Y-%m-%d") for item in sorted(dates.unique())]


def next_business_date(date_value: str) -> str:
    return (pd.Timestamp(_normalize_date(date_value)) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")


def latest_completed_business_date(
    reference_ts: pd.Timestamp | str | None = None,
    *,
    close_time: str = "15:05",
) -> str:
    now_ts = pd.Timestamp(reference_ts).tz_localize(None) if reference_ts is not None else pd.Timestamp.now().tz_localize(None)
    close_clock = pd.Timestamp(close_time).time()
    today = now_ts.normalize()
    is_business_day = today.weekday() < 5
    include_today = is_business_day and now_ts.time() >= close_clock
    offset = 0 if include_today else 1
    return (today - pd.offsets.BDay(offset)).strftime("%Y-%m-%d")


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
    working["adjusted_flag"] = working["adjusted_flag"].fillna(str(adjusted_flag or "none")).astype(str).str.strip().replace("", "none")
    out = working.loc[:, [column for column in STANDARD_MARKET_COLUMNS if column in working.columns]].copy()
    for column in STANDARD_MARKET_COLUMNS:
        if column not in out.columns:
            out[column] = "" if column in {"symbol", "trade_date", "source", "adjusted_flag"} else np.nan
    out = out[STANDARD_MARKET_COLUMNS]
    out = out.loc[out["symbol"].astype(str).str.len() > 0]
    out = out.loc[out["trade_date"].astype(str).str.lower() != "nat"]
    return out.sort_values(["trade_date", "symbol", "source"]).reset_index(drop=True)


def normalize_domain_frame(
    frame: pd.DataFrame,
    *,
    domain: str,
    source: str,
    as_of_date: str = "",
    adjusted_flag: str = "none",
    require_columns: bool = True,
) -> pd.DataFrame:
    normalized_domain = normalize_domain(domain)
    if normalized_domain == DataDomain.MARKET_DAILY:
        return normalize_market_frame(frame, source=source, adjusted_flag=adjusted_flag, require_columns=require_columns)
    if normalized_domain == DataDomain.TRADING_CALENDAR:
        return normalize_calendar_frame(frame, source=source, require_columns=require_columns)
    if normalized_domain == DataDomain.UNIVERSE_SNAPSHOT:
        return normalize_universe_frame(frame, source=source, as_of_date=as_of_date, require_columns=require_columns)
    if normalized_domain == DataDomain.SECURITY_STATUS:
        return normalize_status_frame(frame, source=source, as_of_date=as_of_date, require_columns=require_columns)
    if normalized_domain == DataDomain.LIMIT_STATUS:
        return normalize_limit_frame(frame, source=source, as_of_date=as_of_date, require_columns=require_columns)
    if normalized_domain == DataDomain.INDUSTRY_CONCEPT:
        return normalize_industry_concept_frame(frame, source=source, as_of_date=as_of_date, require_columns=require_columns)
    if normalized_domain == DataDomain.VALUATION:
        return normalize_valuation_frame(frame, source=source, as_of_date=as_of_date, require_columns=require_columns)
    if normalized_domain == DataDomain.MONEY_FLOW_HOTSPOT:
        return normalize_money_flow_hotspot_frame(frame, source=source, as_of_date=as_of_date, require_columns=require_columns)
    raise ValueError(f"unsupported data domain: {domain}")


def normalize_calendar_frame(frame: pd.DataFrame, *, source: str, require_columns: bool = True) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(frame, domain=DataDomain.TRADING_CALENDAR, source=provider, as_of_date="", require_columns=False)
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
    return data.loc[data["trade_date"].astype(str).str.lower() != "nat", DOMAIN_STANDARD_COLUMNS[DataDomain.TRADING_CALENDAR]].drop_duplicates().sort_values(["trade_date", "exchange"]).reset_index(drop=True)


def normalize_universe_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    as_of_date: str,
    require_columns: bool = True,
) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(frame, domain=DataDomain.UNIVERSE_SNAPSHOT, source=provider, as_of_date=as_of_date, require_columns=False)
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
    return out.drop_duplicates(subset=["trade_date", "symbol", "source"]).sort_values(["trade_date", "symbol", "source"]).reset_index(drop=True)


def normalize_status_frame(frame: pd.DataFrame, *, source: str, as_of_date: str, require_columns: bool = True) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(frame, domain=DataDomain.SECURITY_STATUS, source=provider, as_of_date=as_of_date, require_columns=False)
    _rename_first(data, "symbol", ("code", "ts_code", "股票代码", "证券代码"))
    _require_domain_columns(data, DataDomain.SECURITY_STATUS, require_columns=require_columns)
    data = _ensure_domain_columns(data, DataDomain.SECURITY_STATUS)
    data["symbol"] = data["symbol"].map(_normalize_symbol)
    data["trade_date"] = _coerce_trade_date(data["trade_date"], as_of_date)
    for column in ["is_st", "is_suspended", "is_delisted"]:
        data[column] = data[column].map(_to_bool)
    data["status_reason"] = data["status_reason"].fillna("").astype(str)
    data["source"] = _source_series(data, provider)
    return data.loc[data["symbol"].astype(str).str.len() > 0, DOMAIN_STANDARD_COLUMNS[DataDomain.SECURITY_STATUS]].drop_duplicates().sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def normalize_limit_frame(frame: pd.DataFrame, *, source: str, as_of_date: str, require_columns: bool = True) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(frame, domain=DataDomain.LIMIT_STATUS, source=provider, as_of_date=as_of_date, require_columns=False)
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
    return data.loc[data["symbol"].astype(str).str.len() > 0, DOMAIN_STANDARD_COLUMNS[DataDomain.LIMIT_STATUS]].drop_duplicates().sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def normalize_industry_concept_frame(frame: pd.DataFrame, *, source: str, as_of_date: str, require_columns: bool = True) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(frame, domain=DataDomain.INDUSTRY_CONCEPT, source=provider, as_of_date=as_of_date, require_columns=False)
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
    return data.loc[data["symbol"].astype(str).str.len() > 0, DOMAIN_STANDARD_COLUMNS[DataDomain.INDUSTRY_CONCEPT]].drop_duplicates().sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def normalize_valuation_frame(frame: pd.DataFrame, *, source: str, as_of_date: str, require_columns: bool = True) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(frame, domain=DataDomain.VALUATION, source=provider, as_of_date=as_of_date, require_columns=False)
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
    return data.loc[data["symbol"].astype(str).str.len() > 0, DOMAIN_STANDARD_COLUMNS[DataDomain.VALUATION]].drop_duplicates().sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def normalize_money_flow_hotspot_frame(frame: pd.DataFrame, *, source: str, as_of_date: str, require_columns: bool = True) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(frame, domain=DataDomain.MONEY_FLOW_HOTSPOT, source=provider, as_of_date=as_of_date, require_columns=False)
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
    return data.loc[data["symbol"].astype(str).str.len() > 0, DOMAIN_STANDARD_COLUMNS[DataDomain.MONEY_FLOW_HOTSPOT]].drop_duplicates().sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def valid_market_rows(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series([], dtype=bool)
    data = normalize_market_frame(frame, source=str(frame["source"].iloc[0] if "source" in frame.columns and len(frame) else "unknown"), require_columns=False)
    price_positive = data[PRICE_COLUMNS].gt(0).all(axis=1)
    range_valid = data["high"].ge(data[["open", "close", "low"]].max(axis=1)) & data["low"].le(data[["open", "close", "high"]].min(axis=1))
    activity_valid = data[["volume", "amount"]].ge(0).all(axis=1)
    return (price_positive & range_valid & activity_valid).reindex(data.index, fill_value=False)


def coverage_report_for_frame(frame: pd.DataFrame, request: FetchRequest, *, provider: str) -> dict[str, Any]:
    dates = market_business_dates(request.start_date, request.end_date)
    expected_rows = int(len(request.symbols) * len(dates))
    row_count = int(len(frame))
    unique_symbols = int(frame["symbol"].nunique()) if "symbol" in frame.columns and not frame.empty else 0
    unique_dates = int(frame["trade_date"].nunique()) if "trade_date" in frame.columns and not frame.empty else 0
    valid_rows = int(valid_market_rows(frame).sum()) if not frame.empty else 0
    coverage_ratio = float(row_count / expected_rows) if expected_rows else 0.0
    return {
        "provider": str(provider),
        "expected_rows": expected_rows,
        "row_count": row_count,
        "valid_rows": valid_rows,
        "symbol_count": unique_symbols,
        "trade_date_count": unique_dates,
        "coverage_ratio": coverage_ratio,
        "status": "ok" if row_count > 0 and valid_rows > 0 else "empty",
    }


def coverage_report_for_domain(frame: pd.DataFrame, request: DomainFetchRequest, *, provider: str) -> dict[str, Any]:
    request = request.normalized()
    row_count = int(len(frame))
    unique_symbols = int(frame["symbol"].nunique()) if "symbol" in frame.columns and not frame.empty else 0
    unique_dates = int(frame["trade_date"].nunique()) if "trade_date" in frame.columns and not frame.empty else 0
    if request.domain == DataDomain.TRADING_CALENDAR:
        expected_rows = int(len(market_business_dates(request.start_date, request.end_date)))
    elif request.symbols:
        expected_rows = int(len(request.symbols) * max(1, len(market_business_dates(request.start_date, request.end_date))))
    else:
        expected_rows = row_count
    coverage_ratio = float(row_count / expected_rows) if expected_rows else (1.0 if row_count else 0.0)
    return {
        "provider": str(provider),
        "domain": request.domain,
        "expected_rows": expected_rows,
        "row_count": row_count,
        "symbol_count": unique_symbols,
        "trade_date_count": unique_dates,
        "coverage_ratio": coverage_ratio,
        "status": "ok" if row_count > 0 else "empty",
    }


def _normalize_date(value: Any) -> str:
    if value is None or str(value).strip() == "":
        raise ValueError("date cannot be empty")
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _prepare_domain_frame(
    frame: pd.DataFrame,
    *,
    domain: str,
    source: str,
    as_of_date: str,
    require_columns: bool,
) -> pd.DataFrame:
    provider = validate_provider_name(source)
    columns = DOMAIN_STANDARD_COLUMNS[normalize_domain(domain)]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    data = frame.copy()
    lower_lookup = {str(column).strip().lower(): column for column in data.columns}
    for canonical, candidates in _common_domain_aliases().items():
        if canonical in data.columns:
            continue
        for candidate in candidates:
            if candidate.lower() in lower_lookup:
                data = data.rename(columns={lower_lookup[candidate.lower()]: canonical})
                break
    if "source" not in data.columns:
        data["source"] = provider
    if "trade_date" not in data.columns and as_of_date:
        data["trade_date"] = as_of_date
    _require_domain_columns(data, domain, require_columns=require_columns)
    return data


def _common_domain_aliases() -> dict[str, tuple[str, ...]]:
    return {
        "symbol": ("symbol", "stock", "code", "ts_code", "股票代码", "证券代码"),
        "trade_date": ("trade_date", "date", "cal_date", "datetime", "日期", "交易日期"),
        "source": ("source", "provider", "数据源"),
    }


def _rename_first(data: pd.DataFrame, canonical: str, candidates: Iterable[str]) -> None:
    if canonical in data.columns:
        return
    lower_lookup = {str(column).strip().lower(): column for column in data.columns}
    for candidate in candidates:
        if str(candidate).lower() in lower_lookup:
            data.rename(columns={lower_lookup[str(candidate).lower()]: canonical}, inplace=True)
            return


def _require_domain_columns(data: pd.DataFrame, domain: str, *, require_columns: bool) -> None:
    if not require_columns:
        return
    required = set(DOMAIN_STANDARD_COLUMNS[normalize_domain(domain)])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"provider_frame_schema_error: domain={domain} missing columns {missing}")


def _ensure_domain_columns(data: pd.DataFrame, domain: str) -> pd.DataFrame:
    for column in DOMAIN_STANDARD_COLUMNS[normalize_domain(domain)]:
        if column not in data.columns:
            data[column] = "" if column in {"symbol", "trade_date", "source", "name", "exchange", "board", "list_status", "list_date", "delist_date", "status_reason", "industry", "concept_tags", "hotspot_tags"} else np.nan
    return data


def _date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def _coerce_trade_date(series: pd.Series, as_of_date: str) -> pd.Series:
    data = series.copy()
    if as_of_date:
        data = data.fillna(as_of_date).replace("", as_of_date)
    return _date_series(data)


def _source_series(data: pd.DataFrame, provider: str) -> pd.Series:
    return data["source"].fillna(provider).astype(str).str.strip().str.lower().replace("", provider)


def _infer_exchange(symbol: Any, exchange: Any = "") -> str:
    raw_exchange = str(exchange or "").strip().upper()
    if raw_exchange in {"SH", "SSE"}:
        return "SH"
    if raw_exchange in {"SZ", "SZSE"}:
        return "SZ"
    if raw_exchange in {"BJ", "BSE"}:
        return "BJ"
    raw_symbol = str(symbol or "").upper()
    if raw_symbol.endswith(".SH"):
        return "SH"
    if raw_symbol.endswith(".SZ"):
        return "SZ"
    if raw_symbol.endswith(".BJ"):
        return "BJ"
    return raw_exchange or ""


def _to_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "open", "交易", "正常", "是"}:
        return True
    return False


def _tag_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _normalize_symbol(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return ""
    raw = raw.replace("SH.", "").replace("SZ.", "").replace("BJ.", "")
    if "." in raw:
        base, suffix = raw.rsplit(".", 1)
        suffix = suffix.upper()
        if suffix == "SS":
            suffix = "SH"
        if suffix in {"SH", "SZ", "BJ"}:
            return f"{base.zfill(6) if base.isdigit() and len(base) < 6 else base}.{suffix}"
        return raw
    if raw.startswith(("SH", "SZ", "BJ")) and len(raw) >= 8 and raw[2:].isdigit():
        return f"{raw[2:]}.{raw[:2]}"
    if raw.isdigit() and len(raw) == 6:
        if raw.startswith(("5", "6", "9")):
            return f"{raw}.SH"
        if raw.startswith(("4", "8", "92")):
            return f"{raw}.BJ"
        return f"{raw}.SZ"
    return raw


def _column_aliases() -> Mapping[str, Iterable[str]]:
    return {
        "symbol": ("symbol", "stock", "code", "ts_code", "股票代码", "证券代码"),
        "trade_date": ("trade_date", "date", "datetime", "日期", "交易日期"),
        "open": ("open", "Open", "开盘", "开盘价"),
        "high": ("high", "High", "最高", "最高价"),
        "low": ("low", "Low", "最低", "最低价"),
        "close": ("close", "Close", "收盘", "收盘价", "最新价"),
        "volume": ("volume", "Volume", "vol", "成交量"),
        "amount": ("amount", "Amount", "成交额"),
        "source": ("source", "provider", "数据源"),
        "adjusted_flag": ("adjusted_flag", "adjust", "复权"),
    }
