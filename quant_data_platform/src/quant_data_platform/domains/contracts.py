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
    MARKET_INTRADAY_1M = "market_intraday_1m"
    MARKET_INTRADAY_5M = "market_intraday_5m"
    INTRADAY_DAILY_FEATURES = "intraday_daily_features"
    ADJUST_FACTOR = "adjust_factor"
    ADJUST_FACTOR_EVENT = "adjust_factor_event"
    ADJUST_FACTOR_DAILY = "adjust_factor_daily"
    SECURITY_IDENTITY = "security_identity"
    SYMBOL_HISTORY = "symbol_history"
    ELIGIBLE_SIGNAL_D = "eligible_signal_D"
    TRADABLE_OPEN_D1 = "tradable_open_D1"
    TRADING_CALENDAR = "trading_calendar"
    UNIVERSE_SNAPSHOT = "universe_snapshot"
    SECURITY_STATUS = "security_status"
    LIMIT_STATUS = "limit_status"
    INDUSTRY_CONCEPT = "industry_concept"
    VALUATION = "valuation"
    INDEX_CONSTITUENTS = "index_constituents"
    FINANCIAL_QUARTERLY = "financial_quarterly"
    PERFORMANCE_FORECAST = "performance_forecast"
    PERFORMANCE_EXPRESS = "performance_express"
    CORPORATE_ACTIONS = "corporate_actions"
    SHARE_CAPITAL = "share_capital"
    NAME_CHANGE = "name_change"
    MONEY_FLOW_HOTSPOT = "money_flow_hotspot"
    NEWS_EVENT = "news_event"
    ANNOUNCEMENT = "announcement"
    RESEARCH_REPORT = "research_report"
    IWENCAI_SEMANTIC = "iwencai_semantic"


CANONICAL_START_DATE = "2010-01-01"

CORE_MARKET_DOMAINS = (
    DataDomain.MARKET_DAILY,
    DataDomain.MARKET_INTRADAY_1M,
    DataDomain.MARKET_INTRADAY_5M,
    DataDomain.INTRADAY_DAILY_FEATURES,
    DataDomain.ADJUST_FACTOR,
)

STRUCTURAL_STYLE_DOMAINS = (
    DataDomain.VALUATION,
    DataDomain.INDUSTRY_CONCEPT,
    DataDomain.INDEX_CONSTITUENTS,
)

FILTER_DOMAINS = (
    DataDomain.TRADING_CALENDAR,
    DataDomain.UNIVERSE_SNAPSHOT,
    DataDomain.SECURITY_STATUS,
)

EXCLUDED_V1_DOMAINS = (
    DataDomain.FINANCIAL_QUARTERLY,
    DataDomain.PERFORMANCE_FORECAST,
    DataDomain.PERFORMANCE_EXPRESS,
    DataDomain.NEWS_EVENT,
    DataDomain.RESEARCH_REPORT,
    DataDomain.IWENCAI_SEMANTIC,
)

CANONICAL_BUNDLE_SIDECAR_DOMAINS = (
    DataDomain.MARKET_INTRADAY_1M,
    DataDomain.MARKET_INTRADAY_5M,
    DataDomain.INTRADAY_DAILY_FEATURES,
    DataDomain.ADJUST_FACTOR,
    DataDomain.TRADING_CALENDAR,
    DataDomain.UNIVERSE_SNAPSHOT,
    DataDomain.SECURITY_STATUS,
    DataDomain.VALUATION,
    DataDomain.INDUSTRY_CONCEPT,
    DataDomain.INDEX_CONSTITUENTS,
)

PROFILE_DOMAIN_POLICY = {
    "short_horizon_core_v1": {
        "include": (
            DataDomain.MARKET_DAILY,
            DataDomain.INTRADAY_DAILY_FEATURES,
            DataDomain.ADJUST_FACTOR,
            DataDomain.TRADING_CALENDAR,
            DataDomain.UNIVERSE_SNAPSHOT,
            DataDomain.SECURITY_STATUS,
        ),
        "exclude": STRUCTURAL_STYLE_DOMAINS + EXCLUDED_V1_DOMAINS,
    },
    "style_structural_v1": {
        "include": (
            DataDomain.MARKET_DAILY,
            DataDomain.INTRADAY_DAILY_FEATURES,
            DataDomain.ADJUST_FACTOR,
            *STRUCTURAL_STYLE_DOMAINS,
            *FILTER_DOMAINS,
        ),
        "exclude": EXCLUDED_V1_DOMAINS,
    },
    "style_structural_alpha_v2": {
        "include": (
            DataDomain.MARKET_DAILY,
            DataDomain.INTRADAY_DAILY_FEATURES,
            DataDomain.ADJUST_FACTOR,
            *STRUCTURAL_STYLE_DOMAINS,
            *FILTER_DOMAINS,
        ),
        "exclude": EXCLUDED_V1_DOMAINS,
    },
    "medium_horizon_v1": {
        "include": (
            DataDomain.MARKET_DAILY,
            DataDomain.INTRADAY_DAILY_FEATURES,
            DataDomain.ADJUST_FACTOR,
            *STRUCTURAL_STYLE_DOMAINS,
            *FILTER_DOMAINS,
        ),
        "exclude": EXCLUDED_V1_DOMAINS,
    },
}


DOMAIN_STANDARD_COLUMNS: dict[str, list[str]] = {
    DataDomain.MARKET_DAILY: STANDARD_MARKET_COLUMNS,
    DataDomain.MARKET_INTRADAY_1M: [
        "symbol",
        "trade_date",
        "bar_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover_rate",
        "float_share",
        "total_share",
        "source",
        "adjusted_flag",
    ],
    DataDomain.MARKET_INTRADAY_5M: [
        "symbol",
        "trade_date",
        "bar_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "source",
        "adjusted_flag",
    ],
    DataDomain.INTRADAY_DAILY_FEATURES: [
        "symbol",
        "trade_date",
        "first_5m_ret",
        "opening_auction_ret",
        "opening_auction_amount",
        "opening_auction_volume",
        "opening_auction_amount_share",
        "opening_auction_range",
        "opening_auction_vwap",
        "opening_auction_pressure",
        "first_15m_ret",
        "first_30m_ret",
        "first_30m_amount_share",
        "open_gap",
        "open_gap_first_30m_follow_through",
        "open_gap_first_30m_reversal",
        "last_5m_ret",
        "closing_auction_ret",
        "closing_auction_amount",
        "closing_auction_volume",
        "closing_auction_amount_share",
        "closing_auction_range",
        "closing_auction_vwap",
        "closing_auction_pressure",
        "last_30m_ret",
        "last_30m_amount_share",
        "intraday_ret",
        "intraday_vwap",
        "close_to_vwap",
        "intraday_range",
        "close_position",
        "intraday_realized_vol",
        "intraday_price_volume_corr",
        "bar_count",
        "high_time_frac",
        "low_time_frac",
        "high_before_low",
        "open_to_high_ret",
        "open_to_low_ret",
        "high_to_close_ret",
        "low_to_close_ret",
        "intraday_max_drawdown",
        "intraday_max_runup",
        "price_above_vwap_share",
        "cum_vwap_slope",
        "first_5m_amount_share",
        "last_5m_amount_share",
        "first_30m_range",
        "last_30m_range",
        "amount_top_bar_share",
        "amount_concentration_hhi",
        "lunch_gap_ret",
        "am_ret",
        "pm_ret",
        "am_pm_ret_spread",
        "am_pm_vol_spread",
        "am_amount_share",
        "am_pm_amount_spread",
        "early_strength_late_weak",
        "close_pressure_30m",
        "source",
        "adjusted_flag",
    ],
    DataDomain.ADJUST_FACTOR: [
        "symbol",
        "trade_date",
        "fore_adjust_factor",
        "back_adjust_factor",
        "adjust_factor",
        "factor_provider",
        "factor_semantics",
        "source",
    ],
    DataDomain.ADJUST_FACTOR_EVENT: [
        "security_id",
        "divid_operate_date",
        "symbol_on_date",
        "provider_symbol",
        "fore_adjust_factor",
        "back_adjust_factor",
        "adjust_factor",
        "query_date",
        "source_method",
        "verification_status",
        "source",
    ],
    DataDomain.ADJUST_FACTOR_DAILY: [
        "security_id",
        "trade_date",
        "symbol_on_date",
        "fore_adjust_factor",
        "back_adjust_factor",
        "adjust_factor",
        "factor_event_date",
        "baseline_status",
        "source",
    ],
    DataDomain.SECURITY_IDENTITY: [
        "security_id",
        "official_org_id",
        "issuer_name",
        "exchange",
        "list_date",
        "current_symbol",
        "identity_source",
    ],
    DataDomain.SYMBOL_HISTORY: [
        "security_id",
        "symbol",
        "effective_from",
        "effective_to",
        "name_on_date",
        "board_on_date",
        "evidence_source",
        "official_document_hash",
    ],
    DataDomain.ELIGIBLE_SIGNAL_D: [
        "security_id",
        "trade_date",
        "symbol_on_date",
        "is_eligible_signal",
        "eligibility_reason",
        "source",
    ],
    DataDomain.TRADABLE_OPEN_D1: [
        "security_id",
        "trade_date",
        "symbol_on_date",
        "next_trade_date",
        "next_symbol_on_date",
        "open_d1",
        "tradable_open_d1",
        "tradability_reason",
        "source",
    ],
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
    DataDomain.INDEX_CONSTITUENTS: ["index_symbol", "symbol", "trade_date", "index_name", "source"],
    DataDomain.FINANCIAL_QUARTERLY: [
        "symbol",
        "trade_date",
        "report_date",
        "fiscal_year",
        "fiscal_quarter",
        "publish_date",
        "roe_avg",
        "net_profit_margin",
        "gross_profit_margin",
        "net_profit_yoy",
        "revenue_yoy",
        "eps",
        "net_profit",
        "revenue",
        "asset_turnover",
        "debt_to_asset",
        "current_ratio",
        "cash_flow_ps",
        "lag_policy",
        "source",
    ],
    DataDomain.PERFORMANCE_FORECAST: [
        "symbol",
        "trade_date",
        "report_date",
        "fiscal_year",
        "fiscal_quarter",
        "publish_date",
        "forecast_type",
        "profit_min",
        "profit_max",
        "profit_change_min",
        "profit_change_max",
        "lag_policy",
        "source",
    ],
    DataDomain.PERFORMANCE_EXPRESS: [
        "symbol",
        "trade_date",
        "report_date",
        "fiscal_year",
        "fiscal_quarter",
        "publish_date",
        "eps",
        "roe",
        "net_profit",
        "revenue",
        "total_assets",
        "lag_policy",
        "source",
    ],
    DataDomain.CORPORATE_ACTIONS: [
        "symbol",
        "trade_date",
        "announcement_date",
        "ex_date",
        "record_date",
        "dividend_pay_date",
        "action_type",
        "cash_dividend_per_10",
        "bonus_share_per_10",
        "transfer_share_per_10",
        "description",
        "source",
    ],
    DataDomain.SHARE_CAPITAL: [
        "symbol",
        "trade_date",
        "announcement_date",
        "change_reason",
        "total_share",
        "float_share",
        "restricted_share",
        "source",
    ],
    DataDomain.NAME_CHANGE: [
        "symbol",
        "trade_date",
        "old_name",
        "new_name",
        "change_type",
        "source",
    ],
    DataDomain.MONEY_FLOW_HOTSPOT: [
        "symbol",
        "trade_date",
        "main_net_inflow",
        "sector_rank",
        "hotspot_tags",
        "source",
    ],
    DataDomain.NEWS_EVENT: ["symbol", "trade_date", "title", "url", "summary", "source"],
    DataDomain.ANNOUNCEMENT: ["symbol", "trade_date", "title", "url", "category", "source"],
    DataDomain.RESEARCH_REPORT: ["symbol", "trade_date", "title", "institution", "analyst", "url", "source"],
    DataDomain.IWENCAI_SEMANTIC: ["symbol", "trade_date", "query", "answer", "tags", "source"],
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
class DatePartitionFetchRequest:
    """Request one provider-owned market partition for a single trade date.

    Date-partition endpoints are deliberately separate from symbol/range
    endpoints.  In particular, BaoStock 0.9.3 bulk responses must never pass
    through the ordinary paginated ``ResultData.next()`` iterator.
    """

    domain: str
    trade_date: str
    universe_kind: str = "all_a"
    fetch_mode: str = "date_snapshot"

    def normalized(self) -> "DatePartitionFetchRequest":
        domain = normalize_domain(self.domain)
        universe_kind = str(self.universe_kind or "all_a").strip().lower()
        fetch_mode = str(self.fetch_mode or "date_snapshot").strip().lower()
        if universe_kind not in {"all_a", "etf"}:
            raise ValueError(f"unsupported universe_kind: {self.universe_kind}")
        if fetch_mode not in {"date_snapshot", "date_events"}:
            raise ValueError(f"unsupported fetch_mode: {self.fetch_mode}")
        snapshot_domains = {
            DataDomain.MARKET_DAILY,
            DataDomain.SECURITY_STATUS,
            DataDomain.VALUATION,
        }
        if fetch_mode == "date_snapshot" and domain not in snapshot_domains:
            raise ValueError(f"date_snapshot does not support domain: {domain}")
        if fetch_mode == "date_events" and domain != DataDomain.ADJUST_FACTOR_EVENT:
            raise ValueError(f"date_events does not support domain: {domain}")
        if universe_kind == "etf" and fetch_mode != "date_snapshot":
            raise ValueError("ETF date partition only supports date_snapshot")
        return DatePartitionFetchRequest(
            domain=domain,
            trade_date=_normalize_date(self.trade_date),
            universe_kind=universe_kind,
            fetch_mode=fetch_mode,
        )


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    data: pd.DataFrame
    coverage_report: dict[str, Any] = field(default_factory=dict)
    error_report: list[dict[str, Any]] = field(default_factory=list)


DomainProviderResult = ProviderResult


@dataclass(frozen=True)
class DatePartitionProviderResult:
    provider: str
    request: DatePartitionFetchRequest
    raw_data: pd.DataFrame
    data: pd.DataFrame
    coverage_report: dict[str, Any] = field(default_factory=dict)
    error_report: list[dict[str, Any]] = field(default_factory=list)


class MarketProvider(Protocol):
    name: str

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        ...


class DomainProvider(Protocol):
    name: str

    def fetch_domain(self, request: DomainFetchRequest) -> DomainProviderResult:
        ...


class DatePartitionProvider(Protocol):
    name: str

    def fetch_date_partition(self, request: DatePartitionFetchRequest) -> DatePartitionProviderResult:
        ...


MarketProviderCallable = Callable[[FetchRequest], ProviderResult]


def normalize_domain(domain: str) -> str:
    normalized = str(domain or DataDomain.MARKET_DAILY).strip().lower()
    aliases = {
        "market": DataDomain.MARKET_DAILY,
        "market_bars": DataDomain.MARKET_DAILY,
        "daily": DataDomain.MARKET_DAILY,
        "1m": DataDomain.MARKET_INTRADAY_1M,
        "1min": DataDomain.MARKET_INTRADAY_1M,
        "one_minute": DataDomain.MARKET_INTRADAY_1M,
        "market_1m": DataDomain.MARKET_INTRADAY_1M,
        "intraday_1m": DataDomain.MARKET_INTRADAY_1M,
        "5m": DataDomain.MARKET_INTRADAY_5M,
        "5min": DataDomain.MARKET_INTRADAY_5M,
        "five_minute": DataDomain.MARKET_INTRADAY_5M,
        "market_5m": DataDomain.MARKET_INTRADAY_5M,
        "intraday_5m": DataDomain.MARKET_INTRADAY_5M,
        "intraday_daily": DataDomain.INTRADAY_DAILY_FEATURES,
        "intraday_features": DataDomain.INTRADAY_DAILY_FEATURES,
        "intraday_daily_feature": DataDomain.INTRADAY_DAILY_FEATURES,
        "adjust": DataDomain.ADJUST_FACTOR,
        "adjust_factor": DataDomain.ADJUST_FACTOR,
        "adjustment_factor": DataDomain.ADJUST_FACTOR,
        "复权因子": DataDomain.ADJUST_FACTOR,
        "factor_event": DataDomain.ADJUST_FACTOR_EVENT,
        "adjust_factor_event": DataDomain.ADJUST_FACTOR_EVENT,
        "factor_daily": DataDomain.ADJUST_FACTOR_DAILY,
        "adjust_factor_daily": DataDomain.ADJUST_FACTOR_DAILY,
        "eligible_signal_d": DataDomain.ELIGIBLE_SIGNAL_D,
        "tradable_open_d1": DataDomain.TRADABLE_OPEN_D1,
        "security_identity": DataDomain.SECURITY_IDENTITY,
        "symbol_history": DataDomain.SYMBOL_HISTORY,
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
        "index": DataDomain.INDEX_CONSTITUENTS,
        "index_constituent": DataDomain.INDEX_CONSTITUENTS,
        "index_members": DataDomain.INDEX_CONSTITUENTS,
        "financial": DataDomain.FINANCIAL_QUARTERLY,
        "finance": DataDomain.FINANCIAL_QUARTERLY,
        "quarterly_finance": DataDomain.FINANCIAL_QUARTERLY,
        "financial_report": DataDomain.FINANCIAL_QUARTERLY,
        "forecast_report": DataDomain.PERFORMANCE_FORECAST,
        "performance_forecast": DataDomain.PERFORMANCE_FORECAST,
        "earnings_forecast": DataDomain.PERFORMANCE_FORECAST,
        "express_report": DataDomain.PERFORMANCE_EXPRESS,
        "performance_express": DataDomain.PERFORMANCE_EXPRESS,
        "earnings_express": DataDomain.PERFORMANCE_EXPRESS,
        "corporate_action": DataDomain.CORPORATE_ACTIONS,
        "corporate_actions": DataDomain.CORPORATE_ACTIONS,
        "dividend": DataDomain.CORPORATE_ACTIONS,
        "share_capital": DataDomain.SHARE_CAPITAL,
        "capital_change": DataDomain.SHARE_CAPITAL,
        "name_change": DataDomain.NAME_CHANGE,
        "stock_name_change": DataDomain.NAME_CHANGE,
        "money_flow": DataDomain.MONEY_FLOW_HOTSPOT,
        "hotspot": DataDomain.MONEY_FLOW_HOTSPOT,
        "news": DataDomain.NEWS_EVENT,
        "announcement": DataDomain.ANNOUNCEMENT,
        "research_report": DataDomain.RESEARCH_REPORT,
        "report": DataDomain.RESEARCH_REPORT,
        "iwencai": DataDomain.IWENCAI_SEMANTIC,
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
            f"provider={provider_name}; use quant_data_platform.ingest.refresh_daily with non-TDX providers."
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
            "Run `python -m quant_data_platform.ingest.refresh_daily ...` to update the lake, "
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
    if normalized_domain == DataDomain.MARKET_INTRADAY_5M:
        return normalize_intraday_5m_frame(frame, source=source, adjusted_flag=adjusted_flag, require_columns=require_columns)
    if normalized_domain == DataDomain.MARKET_INTRADAY_1M:
        return normalize_intraday_1m_frame(frame, source=source, adjusted_flag=adjusted_flag, require_columns=require_columns)
    if normalized_domain == DataDomain.INTRADAY_DAILY_FEATURES:
        return normalize_intraday_daily_features_frame(frame, source=source, adjusted_flag=adjusted_flag, require_columns=require_columns)
    if normalized_domain == DataDomain.ADJUST_FACTOR:
        return normalize_adjust_factor_frame(frame, source=source, require_columns=require_columns)
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
    if normalized_domain == DataDomain.INDEX_CONSTITUENTS:
        return normalize_index_constituents_frame(frame, source=source, as_of_date=as_of_date, require_columns=require_columns)
    if normalized_domain == DataDomain.FINANCIAL_QUARTERLY:
        return normalize_financial_quarterly_frame(frame, source=source, require_columns=require_columns)
    if normalized_domain == DataDomain.PERFORMANCE_FORECAST:
        return normalize_performance_forecast_frame(frame, source=source, require_columns=require_columns)
    if normalized_domain == DataDomain.PERFORMANCE_EXPRESS:
        return normalize_performance_express_frame(frame, source=source, require_columns=require_columns)
    if normalized_domain == DataDomain.CORPORATE_ACTIONS:
        return normalize_corporate_actions_frame(frame, source=source, as_of_date=as_of_date, require_columns=require_columns)
    if normalized_domain == DataDomain.SHARE_CAPITAL:
        return normalize_share_capital_frame(frame, source=source, as_of_date=as_of_date, require_columns=require_columns)
    if normalized_domain == DataDomain.NAME_CHANGE:
        return normalize_name_change_frame(frame, source=source, as_of_date=as_of_date, require_columns=require_columns)
    if normalized_domain == DataDomain.MONEY_FLOW_HOTSPOT:
        return normalize_money_flow_hotspot_frame(frame, source=source, as_of_date=as_of_date, require_columns=require_columns)
    if normalized_domain in {
        DataDomain.NEWS_EVENT,
        DataDomain.ANNOUNCEMENT,
        DataDomain.RESEARCH_REPORT,
        DataDomain.IWENCAI_SEMANTIC,
    }:
        return normalize_generic_text_domain_frame(frame, domain=normalized_domain, source=source, as_of_date=as_of_date, require_columns=require_columns)
    raise ValueError(f"unsupported data domain: {domain}")


def normalize_intraday_5m_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    adjusted_flag: str = "none",
    require_columns: bool = True,
) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(frame, domain=DataDomain.MARKET_INTRADAY_5M, source=provider, as_of_date="", require_columns=False)
    _rename_first(data, "bar_time", ("time", "bar_time", "minute", "bar_datetime", "时间", "分钟"))
    _rename_intraday_value_columns(data, include_share_fields=False)
    if "bar_time" not in data.columns and "trade_date" in data.columns:
        data["bar_time"] = ""
    if "adjusted_flag" not in data.columns:
        data["adjusted_flag"] = str(adjusted_flag or "none")
    _require_core_columns(data, DataDomain.MARKET_INTRADAY_5M, {"symbol", "trade_date", "bar_time"}, require_columns=require_columns)
    data = _ensure_domain_columns(data, DataDomain.MARKET_INTRADAY_5M)
    data["symbol"] = data["symbol"].map(_normalize_symbol)
    data = _split_intraday_datetime_column(data)
    data["bar_time"] = data["bar_time"].map(_normalize_bar_time)
    for column in NUMERIC_MARKET_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["source"] = _source_series(data, provider)
    data["adjusted_flag"] = data["adjusted_flag"].fillna(str(adjusted_flag or "none")).astype(str).str.strip().replace("", "none")
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
    data = _prepare_domain_frame(frame, domain=DataDomain.MARKET_INTRADAY_1M, source=provider, as_of_date="", require_columns=False)
    _rename_first(data, "bar_time", ("time", "bar_time", "minute", "bar_datetime", "datetime", "日期", "时间", "分钟"))
    _rename_intraday_value_columns(data, include_share_fields=True)
    if "bar_time" not in data.columns and "trade_date" in data.columns:
        data["bar_time"] = ""
    if "adjusted_flag" not in data.columns:
        data["adjusted_flag"] = str(adjusted_flag or "none")
    _require_core_columns(data, DataDomain.MARKET_INTRADAY_1M, {"symbol", "trade_date", "bar_time"}, require_columns=require_columns)
    data = _ensure_domain_columns(data, DataDomain.MARKET_INTRADAY_1M)
    data["symbol"] = data["symbol"].map(_normalize_symbol)
    data = _split_intraday_datetime_column(data)
    data["bar_time"] = data["bar_time"].map(_normalize_bar_time)
    for column in ("open", "high", "low", "close", "volume", "amount", "turnover_rate", "float_share", "total_share"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["source"] = _source_series(data, provider)
    data["adjusted_flag"] = data["adjusted_flag"].fillna(str(adjusted_flag or "none")).astype(str).str.strip().replace("", "none")
    out = data.loc[
        data["symbol"].astype(str).str.len().gt(0)
        & data["trade_date"].astype(str).str.lower().ne("nat")
        & data["bar_time"].astype(str).str.len().gt(0),
        DOMAIN_STANDARD_COLUMNS[DataDomain.MARKET_INTRADAY_1M],
    ]
    return out.drop_duplicates(subset=["trade_date", "symbol", "bar_time", "source"]).sort_values(["trade_date", "symbol", "bar_time", "source"]).reset_index(drop=True)


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
    return normalize_intraday_5m_frame(rows, source=f"{source}_agg_5m", adjusted_flag=adjusted_flag, require_columns=False)


def normalize_adjust_factor_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    require_columns: bool = True,
) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(frame, domain=DataDomain.ADJUST_FACTOR, source=provider, as_of_date="", require_columns=False)
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
    data["factor_semantics"] = data["factor_semantics"].fillna("raw_provider_factor").astype(str).str.strip().replace("", "raw_provider_factor")
    data["source"] = _source_series(data, provider)
    out = data.loc[
        data["symbol"].astype(str).str.len().gt(0)
        & data["trade_date"].astype(str).str.lower().ne("nat"),
        DOMAIN_STANDARD_COLUMNS[DataDomain.ADJUST_FACTOR],
    ]
    return out.drop_duplicates(subset=["trade_date", "symbol", "factor_provider", "source"]).sort_values(["trade_date", "symbol", "factor_provider", "source"]).reset_index(drop=True)


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
    prev_close_by_symbol: dict[str, float] = {}
    for (trade_date, symbol), group in bars.groupby(["trade_date", "symbol"], sort=True):
        day = group.sort_values("bar_time").reset_index(drop=True).copy()
        if day.empty:
            continue
        open_values = pd.to_numeric(day["open"], errors="coerce")
        high_values = pd.to_numeric(day["high"], errors="coerce")
        low_values = pd.to_numeric(day["low"], errors="coerce")
        close_values = pd.to_numeric(day["close"], errors="coerce")
        volume_values = pd.to_numeric(day["volume"], errors="coerce")
        amount_values = pd.to_numeric(day["amount"], errors="coerce")
        total_amount = _finite_sum(amount_values)
        total_volume = _finite_sum(volume_values)
        first_open = _first_finite(open_values)
        last_close = _last_finite(close_values)
        prev_close = prev_close_by_symbol.get(str(symbol), np.nan)
        high_max = float(high_values.max()) if high_values.notna().any() else np.nan
        low_min = float(low_values.min()) if low_values.notna().any() else np.nan
        first_30m_ret = _head_window_ret(day, 6)
        last_30m_ret = _tail_window_ret(day, 6)
        open_gap = _safe_return(first_open, prev_close)
        amount_share_30 = _window_sum(amount_values, 6, head=True) / total_amount if total_amount > 0 else np.nan
        last_amount_share_30 = _window_sum(amount_values, 6, head=False) / total_amount if total_amount > 0 else np.nan
        opening_amount = _finite_sum(amount_values.head(1))
        opening_volume = _finite_sum(volume_values.head(1))
        closing_amount = _finite_sum(amount_values.tail(1))
        closing_volume = _finite_sum(volume_values.tail(1))
        opening_amount_share = opening_amount / total_amount if total_amount > 0 else np.nan
        closing_amount_share = closing_amount / total_amount if total_amount > 0 else np.nan
        opening_ret = _head_window_ret(day, 1)
        closing_ret = _tail_close_to_previous_close_ret(day)
        vwap = total_amount / total_volume if total_amount > 0 and total_volume > 0 else np.nan
        close_ret = close_values.pct_change().replace([np.inf, -np.inf], np.nan)
        clocks = day["bar_time"].map(_bar_clock_int)
        bar_count = int(len(day))
        high_pos = _first_extreme_position(high_values, mode="max")
        low_pos = _first_extreme_position(low_values, mode="min")
        high_time_frac = _position_fraction(high_pos, bar_count)
        low_time_frac = _position_fraction(low_pos, bar_count)
        high_before_low = (
            float(high_pos < low_pos)
            if high_pos >= 0 and low_pos >= 0 and high_pos != low_pos
            else (0.5 if high_pos >= 0 and low_pos >= 0 else np.nan)
        )
        first_5m_amount_share = _window_sum(amount_values, 1, head=True) / total_amount if total_amount > 0 else np.nan
        last_5m_amount_share = _window_sum(amount_values, 1, head=False) / total_amount if total_amount > 0 else np.nan
        cum_volume = volume_values.cumsum()
        cum_vwap = amount_values.cumsum() / cum_volume.where(cum_volume > 0)
        amount_share = amount_values / total_amount if total_amount > 0 else pd.Series(np.nan, index=amount_values.index)
        am_mask = clocks.le(113000)
        pm_mask = clocks.ge(130000)
        if not bool(am_mask.any()) and len(day) > 1:
            am_mask = pd.Series(np.arange(len(day)) < len(day) // 2, index=day.index)
        if not bool(pm_mask.any()) and len(day) > 1:
            pm_mask = ~am_mask
        am_day = day.loc[am_mask]
        pm_day = day.loc[pm_mask]
        am_ret = _session_return(am_day)
        pm_ret = _session_return(pm_day)
        lunch_gap_ret = _safe_return(pm_day["open"].iloc[0], am_day["close"].iloc[-1]) if not am_day.empty and not pm_day.empty else np.nan
        am_vol = pd.to_numeric(am_day["close"], errors="coerce").pct_change().replace([np.inf, -np.inf], np.nan).std() if not am_day.empty else np.nan
        pm_vol = pd.to_numeric(pm_day["close"], errors="coerce").pct_change().replace([np.inf, -np.inf], np.nan).std() if not pm_day.empty else np.nan
        am_amount_share = _finite_sum(pd.to_numeric(am_day.get("amount", pd.Series(dtype=float)), errors="coerce")) / total_amount if total_amount > 0 else np.nan
        pm_amount_share = 1.0 - am_amount_share if pd.notna(am_amount_share) else np.nan
        close_position = (last_close - low_min) / (high_max - low_min) if pd.notna(last_close) and pd.notna(high_max) and pd.notna(low_min) and high_max > low_min else np.nan
        price_volume_corr = close_ret.corr(volume_values) if close_ret.notna().sum() >= 2 and volume_values.notna().sum() >= 2 else np.nan
        gap_sign = np.sign(open_gap) if pd.notna(open_gap) else np.nan
        rows.append(
            {
                "symbol": symbol,
                "trade_date": trade_date,
                "first_5m_ret": _head_window_ret(day, 1),
                "opening_auction_ret": opening_ret,
                "opening_auction_amount": opening_amount,
                "opening_auction_volume": opening_volume,
                "opening_auction_amount_share": opening_amount_share,
                "opening_auction_range": _window_range(day, 1, head=True),
                "opening_auction_vwap": opening_amount / opening_volume if opening_amount > 0 and opening_volume > 0 else np.nan,
                "opening_auction_pressure": opening_ret * opening_amount_share if pd.notna(opening_ret) and pd.notna(opening_amount_share) else np.nan,
                "first_15m_ret": _head_window_ret(day, 3),
                "first_30m_ret": first_30m_ret,
                "first_30m_amount_share": amount_share_30,
                "open_gap": open_gap,
                "open_gap_first_30m_follow_through": gap_sign * first_30m_ret if pd.notna(gap_sign) and pd.notna(first_30m_ret) else np.nan,
                "open_gap_first_30m_reversal": -gap_sign * first_30m_ret if pd.notna(gap_sign) and pd.notna(first_30m_ret) else np.nan,
                "last_5m_ret": _tail_close_to_previous_close_ret(day),
                "closing_auction_ret": closing_ret,
                "closing_auction_amount": closing_amount,
                "closing_auction_volume": closing_volume,
                "closing_auction_amount_share": closing_amount_share,
                "closing_auction_range": _window_range(day, 1, head=False),
                "closing_auction_vwap": closing_amount / closing_volume if closing_amount > 0 and closing_volume > 0 else np.nan,
                "closing_auction_pressure": closing_ret * closing_amount_share if pd.notna(closing_ret) and pd.notna(closing_amount_share) else np.nan,
                "last_30m_ret": last_30m_ret,
                "last_30m_amount_share": last_amount_share_30,
                "intraday_ret": _safe_return(last_close, first_open),
                "intraday_vwap": vwap,
                "close_to_vwap": _safe_return(last_close, vwap),
                "intraday_range": _safe_return(high_max, low_min),
                "close_position": close_position,
                "intraday_realized_vol": float(close_ret.std()) if close_ret.notna().sum() >= 2 else np.nan,
                "intraday_price_volume_corr": price_volume_corr,
                "bar_count": float(bar_count),
                "high_time_frac": high_time_frac,
                "low_time_frac": low_time_frac,
                "high_before_low": high_before_low,
                "open_to_high_ret": _safe_return(high_max, first_open),
                "open_to_low_ret": _safe_return(low_min, first_open),
                "high_to_close_ret": _safe_return(last_close, high_max),
                "low_to_close_ret": _safe_return(last_close, low_min),
                "intraday_max_drawdown": _max_drawdown(close_values),
                "intraday_max_runup": _max_runup(close_values),
                "price_above_vwap_share": _finite_ratio(close_values > vwap) if pd.notna(vwap) else np.nan,
                "cum_vwap_slope": _linear_slope(cum_vwap),
                "first_5m_amount_share": first_5m_amount_share,
                "last_5m_amount_share": last_5m_amount_share,
                "first_30m_range": _window_range(day, 6, head=True),
                "last_30m_range": _window_range(day, 6, head=False),
                "amount_top_bar_share": float(amount_share.max()) if amount_share.notna().any() else np.nan,
                "amount_concentration_hhi": float((amount_share.dropna() ** 2).sum()) if amount_share.notna().any() else np.nan,
                "lunch_gap_ret": lunch_gap_ret,
                "am_ret": am_ret,
                "pm_ret": pm_ret,
                "am_pm_ret_spread": pm_ret - am_ret,
                "am_pm_vol_spread": pm_vol - am_vol,
                "am_amount_share": am_amount_share,
                "am_pm_amount_spread": am_amount_share - pm_amount_share if pd.notna(am_amount_share) and pd.notna(pm_amount_share) else np.nan,
                "early_strength_late_weak": first_30m_ret - last_30m_ret if pd.notna(first_30m_ret) and pd.notna(last_30m_ret) else np.nan,
                "close_pressure_30m": last_30m_ret * last_amount_share_30 if pd.notna(last_30m_ret) and pd.notna(last_amount_share_30) else np.nan,
                "source": source,
                "adjusted_flag": str(adjusted_flag or "none"),
            }
        )
        prev_close_by_symbol[str(symbol)] = last_close
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
    data = _prepare_domain_frame(frame, domain=DataDomain.INTRADAY_DAILY_FEATURES, source=provider, as_of_date="", require_columns=False)
    if "adjusted_flag" not in data.columns:
        data["adjusted_flag"] = str(adjusted_flag or "none")
    _require_core_columns(data, DataDomain.INTRADAY_DAILY_FEATURES, {"symbol", "trade_date"}, require_columns=require_columns)
    data = _ensure_domain_columns(data, DataDomain.INTRADAY_DAILY_FEATURES)
    data["symbol"] = data["symbol"].map(_normalize_symbol)
    data["trade_date"] = _date_series(data["trade_date"])
    for column in DOMAIN_STANDARD_COLUMNS[DataDomain.INTRADAY_DAILY_FEATURES]:
        if column not in {"symbol", "trade_date", "source", "adjusted_flag"}:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data["source"] = _source_series(data, provider)
    data["adjusted_flag"] = data["adjusted_flag"].fillna(str(adjusted_flag or "none")).astype(str).str.strip().replace("", "none")
    out = data.loc[
        data["symbol"].astype(str).str.len().gt(0)
        & data["trade_date"].astype(str).str.lower().ne("nat"),
        DOMAIN_STANDARD_COLUMNS[DataDomain.INTRADAY_DAILY_FEATURES],
    ]
    return out.drop_duplicates(subset=["trade_date", "symbol", "source"]).sort_values(["trade_date", "symbol", "source"]).reset_index(drop=True)


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


def normalize_index_constituents_frame(frame: pd.DataFrame, *, source: str, as_of_date: str, require_columns: bool = True) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(frame, domain=DataDomain.INDEX_CONSTITUENTS, source=provider, as_of_date=as_of_date, require_columns=False)
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
            data["symbol"].astype(str).str.len().gt(0)
            & data["index_symbol"].astype(str).str.len().gt(0),
            DOMAIN_STANDARD_COLUMNS[DataDomain.INDEX_CONSTITUENTS],
        ]
        .drop_duplicates(subset=["trade_date", "index_symbol", "symbol", "source"])
        .sort_values(["trade_date", "index_symbol", "symbol"])
        .reset_index(drop=True)
    )


def normalize_financial_quarterly_frame(frame: pd.DataFrame, *, source: str, require_columns: bool = True) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(frame, domain=DataDomain.FINANCIAL_QUARTERLY, source=provider, as_of_date="", require_columns=False)
    _rename_financial_common_columns(data)
    _rename_first(data, "roe_avg", ("roeAvg", "roe_avg", "净资产收益率", "平均净资产收益率"))
    _rename_first(data, "net_profit_margin", ("npMargin", "netProfitMargin", "net_profit_margin", "销售净利率"))
    _rename_first(data, "gross_profit_margin", ("gpMargin", "grossProfitMargin", "gross_profit_margin", "销售毛利率"))
    _rename_first(data, "net_profit_yoy", ("YOYPNI", "YOYNI", "netProfitGrowRate", "net_profit_yoy", "净利润同比增长率"))
    _rename_first(data, "revenue_yoy", ("YOYIncome", "YOYRevenue", "revenueGrowRate", "revenue_yoy", "营业收入同比增长率"))
    _rename_first(data, "eps", ("epsTTM", "eps", "每股收益"))
    _rename_first(data, "net_profit", ("netProfit", "net_profit", "归属母公司股东的净利润"))
    _rename_first(data, "revenue", ("totalShare", "revenue", "营业总收入"))
    _rename_first(data, "asset_turnover", ("NRTurnRatio", "asset_turnover", "总资产周转率"))
    _rename_first(data, "debt_to_asset", ("liabilityToAsset", "debt_to_asset", "资产负债率"))
    _rename_first(data, "current_ratio", ("currentRatio", "current_ratio", "流动比率"))
    _rename_first(data, "cash_flow_ps", ("CAToAsset", "cash_flow_ps", "每股经营现金流"))
    _require_core_columns(data, DataDomain.FINANCIAL_QUARTERLY, {"symbol", "report_date"}, require_columns=require_columns)
    data = _ensure_report_domain_columns(data, DataDomain.FINANCIAL_QUARTERLY, provider=provider)
    numeric_columns = [
        "fiscal_year",
        "fiscal_quarter",
        "roe_avg",
        "net_profit_margin",
        "gross_profit_margin",
        "net_profit_yoy",
        "revenue_yoy",
        "eps",
        "net_profit",
        "revenue",
        "asset_turnover",
        "debt_to_asset",
        "current_ratio",
        "cash_flow_ps",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return _finalize_report_domain_frame(data, DataDomain.FINANCIAL_QUARTERLY)


def normalize_performance_forecast_frame(frame: pd.DataFrame, *, source: str, require_columns: bool = True) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(frame, domain=DataDomain.PERFORMANCE_FORECAST, source=provider, as_of_date="", require_columns=False)
    _rename_financial_common_columns(data)
    _rename_first(data, "forecast_type", ("profitForcastType", "forecastType", "type", "业绩预告类型"))
    _rename_first(data, "profit_min", ("profitMin", "profit_min", "预告净利润下限"))
    _rename_first(data, "profit_max", ("profitMax", "profit_max", "预告净利润上限"))
    _rename_first(data, "profit_change_min", ("profitForcastChgPctDwn", "profitForcastChgPctMin", "profit_change_min", "预告净利润变动下限"))
    _rename_first(data, "profit_change_max", ("profitForcastChgPctUp", "profitForcastChgPctMax", "profit_change_max", "预告净利润变动上限"))
    _require_core_columns(data, DataDomain.PERFORMANCE_FORECAST, {"symbol", "report_date", "publish_date"}, require_columns=require_columns)
    data = _ensure_report_domain_columns(data, DataDomain.PERFORMANCE_FORECAST, provider=provider)
    for column in ["fiscal_year", "fiscal_quarter", "profit_min", "profit_max", "profit_change_min", "profit_change_max"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["forecast_type"] = data["forecast_type"].fillna("").astype(str).str.strip()
    return _finalize_report_domain_frame(data, DataDomain.PERFORMANCE_FORECAST)


def normalize_performance_express_frame(frame: pd.DataFrame, *, source: str, require_columns: bool = True) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(frame, domain=DataDomain.PERFORMANCE_EXPRESS, source=provider, as_of_date="", require_columns=False)
    _rename_financial_common_columns(data)
    _rename_first(data, "eps", ("performanceExpressEPSDiluted", "performanceExpressEPSBasic", "eps", "EPS", "每股收益"))
    _rename_first(data, "roe", ("performanceExpressROEWa", "roe", "ROE", "净资产收益率"))
    _rename_first(data, "net_profit", ("performanceExpressNetProfit", "netProfit", "net_profit", "归属母公司股东的净利润"))
    _rename_first(data, "revenue", ("performanceExpressTotalIncome", "totalShare", "revenue", "营业总收入"))
    _rename_first(data, "total_assets", ("performanceExpressTotalAsset", "totalAssets", "total_assets", "总资产"))
    _require_core_columns(data, DataDomain.PERFORMANCE_EXPRESS, {"symbol", "report_date", "publish_date"}, require_columns=require_columns)
    data = _ensure_report_domain_columns(data, DataDomain.PERFORMANCE_EXPRESS, provider=provider)
    for column in ["fiscal_year", "fiscal_quarter", "eps", "roe", "net_profit", "revenue", "total_assets"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return _finalize_report_domain_frame(data, DataDomain.PERFORMANCE_EXPRESS)


def normalize_corporate_actions_frame(frame: pd.DataFrame, *, source: str, as_of_date: str, require_columns: bool = True) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(frame, domain=DataDomain.CORPORATE_ACTIONS, source=provider, as_of_date="", require_columns=False)
    _rename_first(data, "symbol", ("code", "ts_code", "证券代码", "股票代码"))
    _rename_first(data, "trade_date", ("ex_date", "除权日", "除权除息日", "日期"))
    _rename_first(data, "announcement_date", ("实施方案公告日期", "公告日期", "announcement_date"))
    _rename_first(data, "ex_date", ("除权日", "除权除息日", "ex_date"))
    _rename_first(data, "record_date", ("股权登记日", "record_date"))
    _rename_first(data, "dividend_pay_date", ("派息日", "dividend_pay_date"))
    _rename_first(data, "action_type", ("分红类型", "action_type", "event_type"))
    _rename_first(data, "cash_dividend_per_10", ("派息比例", "cash_dividend_per_10", "cash_dividend"))
    _rename_first(data, "bonus_share_per_10", ("送股比例", "bonus_share_per_10", "bonus_share"))
    _rename_first(data, "transfer_share_per_10", ("转增比例", "transfer_share_per_10", "transfer_share"))
    _rename_first(data, "description", ("实施方案分红说明", "description", "方案说明"))
    _require_core_columns(data, DataDomain.CORPORATE_ACTIONS, {"symbol"}, require_columns=require_columns)
    data = _ensure_domain_columns(data, DataDomain.CORPORATE_ACTIONS)
    data["symbol"] = data["symbol"].map(_normalize_symbol)
    for column in ["trade_date", "announcement_date", "ex_date", "record_date", "dividend_pay_date"]:
        data[column] = _date_series(data[column])
    data["trade_date"] = data["trade_date"].replace("NaT", np.nan).fillna(data["announcement_date"]).fillna(str(as_of_date or ""))
    for column in ["cash_dividend_per_10", "bonus_share_per_10", "transfer_share_per_10"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["action_type"] = data["action_type"].fillna("").astype(str).str.strip()
    data["description"] = data["description"].fillna("").astype(str).str.strip()
    data["source"] = _source_series(data, provider)
    return (
        data.loc[
            data["symbol"].astype(str).str.len().gt(0)
            & data["trade_date"].astype(str).str.lower().ne("nat")
            & data["trade_date"].astype(str).str.len().gt(0),
            DOMAIN_STANDARD_COLUMNS[DataDomain.CORPORATE_ACTIONS],
        ]
        .drop_duplicates()
        .sort_values(["trade_date", "symbol", "action_type", "source"])
        .reset_index(drop=True)
    )


def normalize_share_capital_frame(frame: pd.DataFrame, *, source: str, as_of_date: str, require_columns: bool = True) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(frame, domain=DataDomain.SHARE_CAPITAL, source=provider, as_of_date="", require_columns=False)
    _rename_first(data, "symbol", ("code", "ts_code", "证券代码", "股票代码"))
    _rename_first(data, "trade_date", ("变动日期", "change_date", "date", "trade_date"))
    _rename_first(data, "announcement_date", ("公告日期", "announcement_date"))
    _rename_first(data, "change_reason", ("变动原因", "change_reason", "reason"))
    _rename_first(data, "total_share", ("总股本", "total_share", "total_shares"))
    _rename_first(data, "float_share", ("已流通股份", "人民币普通股", "float_share", "float_shares"))
    _rename_first(data, "restricted_share", ("流通受限股份", "restricted_share", "restricted_shares"))
    _require_core_columns(data, DataDomain.SHARE_CAPITAL, {"symbol", "trade_date"}, require_columns=require_columns)
    data = _ensure_domain_columns(data, DataDomain.SHARE_CAPITAL)
    data["symbol"] = data["symbol"].map(_normalize_symbol)
    data["trade_date"] = _coerce_trade_date(data["trade_date"], as_of_date)
    data["announcement_date"] = _date_series(data["announcement_date"])
    data["change_reason"] = data["change_reason"].fillna("").astype(str).str.strip()
    for column in ["total_share", "float_share", "restricted_share"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["source"] = _source_series(data, provider)
    return (
        data.loc[
            data["symbol"].astype(str).str.len().gt(0)
            & data["trade_date"].astype(str).str.lower().ne("nat"),
            DOMAIN_STANDARD_COLUMNS[DataDomain.SHARE_CAPITAL],
        ]
        .drop_duplicates(subset=["trade_date", "symbol", "source"], keep="last")
        .sort_values(["trade_date", "symbol"])
        .reset_index(drop=True)
    )


def normalize_name_change_frame(frame: pd.DataFrame, *, source: str, as_of_date: str, require_columns: bool = True) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(frame, domain=DataDomain.NAME_CHANGE, source=provider, as_of_date="", require_columns=False)
    _rename_first(data, "symbol", ("code", "ts_code", "证券代码", "股票代码"))
    _rename_first(data, "trade_date", ("变更日期", "change_date", "date", "trade_date"))
    _rename_first(data, "old_name", ("变更前简称", "变更前全称", "old_name", "previous_name"))
    _rename_first(data, "new_name", ("变更后简称", "变更后全称", "new_name", "current_name"))
    _rename_first(data, "change_type", ("change_type", "类型", "变更类型"))
    _require_core_columns(data, DataDomain.NAME_CHANGE, {"symbol", "trade_date"}, require_columns=require_columns)
    data = _ensure_domain_columns(data, DataDomain.NAME_CHANGE)
    data["symbol"] = data["symbol"].map(_normalize_symbol)
    data["trade_date"] = _coerce_trade_date(data["trade_date"], as_of_date)
    for column in ["old_name", "new_name", "change_type"]:
        data[column] = data[column].fillna("").astype(str).str.strip()
    data["source"] = _source_series(data, provider)
    return (
        data.loc[
            data["symbol"].astype(str).str.len().gt(0)
            & data["trade_date"].astype(str).str.lower().ne("nat"),
            DOMAIN_STANDARD_COLUMNS[DataDomain.NAME_CHANGE],
        ]
        .drop_duplicates()
        .sort_values(["trade_date", "symbol", "change_type"])
        .reset_index(drop=True)
    )


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
    data = _prepare_domain_frame(frame, domain=normalized_domain, source=provider, as_of_date=as_of_date, require_columns=False)
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
    parsed_has_clock = trade_ts.notna() & (
        trade_ts.dt.hour.ne(0) | trade_ts.dt.minute.ne(0) | trade_ts.dt.second.ne(0)
    )
    fill_from_trade_date = missing_bar_time & parsed_has_clock
    if bool(fill_from_trade_date.any()):
        data.loc[fill_from_trade_date, "bar_time"] = trade_ts.loc[fill_from_trade_date].dt.strftime("%H:%M:%S")
    return data


def _bar_clock_int(value: Any) -> int:
    text = _normalize_bar_time(value)
    try:
        return int(text[:6])
    except Exception:
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
    try:
        return int(matches[0])
    except Exception:
        return int(data.index.get_loc(matches[0]))


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


def _rename_financial_common_columns(data: pd.DataFrame) -> None:
    _rename_first(data, "symbol", ("code", "stock", "ts_code", "股票代码", "证券代码"))
    _rename_first(data, "report_date", ("statDate", "reportDate", "endDate", "performanceExpStatDate", "profitForcastExpStatDate", "报告日期", "统计日期"))
    _rename_first(data, "publish_date", ("pubDate", "publishDate", "performanceExpPubDate", "performanceExpUpdateDate", "profitForcastExpPubDate", "公告日期", "发布日期", "更新日期"))
    _rename_first(data, "fiscal_year", ("year", "fiscalYear", "报告年度"))
    _rename_first(data, "fiscal_quarter", ("quarter", "fiscalQuarter", "报告季度"))


def _ensure_report_domain_columns(data: pd.DataFrame, domain: str, *, provider: str) -> pd.DataFrame:
    data = _ensure_domain_columns(data, domain)
    data["symbol"] = data["symbol"].map(_normalize_symbol)
    report_ts = pd.to_datetime(data["report_date"], errors="coerce")
    publish_ts = pd.to_datetime(data["publish_date"], errors="coerce")
    data["report_date"] = report_ts.dt.strftime("%Y-%m-%d")
    publish = publish_ts.dt.strftime("%Y-%m-%d")
    report_ts = pd.to_datetime(data["report_date"], errors="coerce")
    fiscal_year = pd.to_numeric(data["fiscal_year"], errors="coerce")
    fiscal_quarter = pd.to_numeric(data["fiscal_quarter"], errors="coerce")
    data["fiscal_year"] = fiscal_year.where(fiscal_year.notna(), report_ts.dt.year)
    data["fiscal_quarter"] = fiscal_quarter.where(fiscal_quarter.notna(), report_ts.dt.quarter)
    conservative_publish = (report_ts + pd.offsets.BDay(90)).dt.strftime("%Y-%m-%d")
    has_publish = publish_ts.notna()
    data["publish_date"] = publish.where(has_publish, conservative_publish)
    trade_ts = pd.to_datetime(data["trade_date"], errors="coerce")
    data["trade_date"] = trade_ts.dt.strftime("%Y-%m-%d")
    missing_trade_date = trade_ts.isna()
    data.loc[missing_trade_date, "trade_date"] = data.loc[missing_trade_date, "publish_date"]
    data["lag_policy"] = data["lag_policy"].fillna("").astype(str).str.strip()
    inferred = data["lag_policy"].eq("")
    policy_values = pd.Series(
        np.where(
            has_publish,
            "publish_date_plus_1d_in_features",
            "conservative_report_date_plus_90bd_plus_1d_in_features",
        ),
        index=data.index,
    )
    data.loc[inferred, "lag_policy"] = policy_values.loc[inferred]
    data["source"] = _source_series(data, provider)
    return data


def _finalize_report_domain_frame(data: pd.DataFrame, domain: str) -> pd.DataFrame:
    columns = DOMAIN_STANDARD_COLUMNS[normalize_domain(domain)]
    out = data.loc[
        data["symbol"].astype(str).str.len().gt(0)
        & data["trade_date"].astype(str).str.lower().ne("nat"),
        columns,
    ]
    sort_columns = [column for column in ["trade_date", "symbol", "fiscal_year", "fiscal_quarter", "source"] if column in out.columns]
    return out.drop_duplicates(subset=sort_columns, keep="last").sort_values(sort_columns).reset_index(drop=True)


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


def _rename_intraday_value_columns(data: pd.DataFrame, *, include_share_fields: bool) -> None:
    _rename_first(data, "open", ("open", "开盘", "开盘价"))
    _rename_first(data, "high", ("high", "最高", "最高价"))
    _rename_first(data, "low", ("low", "最低", "最低价"))
    _rename_first(data, "close", ("close", "收盘", "收盘价"))
    _rename_first(data, "volume", ("volume", "vol", "成交量", "成交量(股)", "成交量（股）"))
    _rename_first(data, "amount", ("amount", "成交额", "成交额(元)", "成交额（元）"))
    if include_share_fields:
        _rename_first(data, "turnover_rate", ("turnover_rate", "turnover", "换手率", "换手率(%)", "换手率（%）"))
        _rename_first(data, "float_share", ("float_share", "float_shares", "流通股本", "流通股本(股)", "流通股本（股）"))
        _rename_first(data, "total_share", ("total_share", "total_shares", "总股本", "总股本(股)", "总股本（股）"))


def _require_domain_columns(data: pd.DataFrame, domain: str, *, require_columns: bool) -> None:
    if not require_columns:
        return
    required = set(DOMAIN_STANDARD_COLUMNS[normalize_domain(domain)])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"provider_frame_schema_error: domain={domain} missing columns {missing}")


def _require_core_columns(data: pd.DataFrame, domain: str, required: set[str], *, require_columns: bool) -> None:
    if not require_columns:
        return
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"provider_frame_schema_error: domain={domain} missing core columns {missing}")


def _ensure_domain_columns(data: pd.DataFrame, domain: str) -> pd.DataFrame:
    string_columns = {
        "symbol",
        "trade_date",
        "source",
        "adjusted_flag",
        "bar_time",
        "name",
        "exchange",
        "board",
        "list_status",
        "list_date",
        "delist_date",
        "status_reason",
        "industry",
        "concept_tags",
        "hotspot_tags",
        "index_symbol",
        "index_name",
        "report_date",
        "publish_date",
        "announcement_date",
        "ex_date",
        "record_date",
        "dividend_pay_date",
        "forecast_type",
        "action_type",
        "description",
        "change_reason",
        "old_name",
        "new_name",
        "change_type",
        "lag_policy",
        "factor_provider",
        "factor_semantics",
    }
    for column in DOMAIN_STANDARD_COLUMNS[normalize_domain(domain)]:
        if column not in data.columns:
            data[column] = "" if column in string_columns else np.nan
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
