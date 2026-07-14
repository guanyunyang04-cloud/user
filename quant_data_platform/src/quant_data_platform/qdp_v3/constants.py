from __future__ import annotations

from quant_data_platform.providers import (
    BAOSTOCK_BATCH_VERSION,
    BAOSTOCK_BATCH_WHEEL_SHA256,
)


MANIFEST_VERSION = 3
QDP_V3_CONTRACT_VERSION = "qdp_v3_20260714_bootstrap_1m"
SCHEMA_VERSION = "3.1.0"
SECURITY_IDENTITY_CONTRACT = "qdp_stable_security_id_pit_symbol_v1"
TIMEZONE = "Asia/Shanghai"

MOOTDX_VERSION = "0.11.7"

QUALITY_STRICT = "strict"
QUALITY_PROVISIONAL = "provisional"
QUALITY_QUARANTINED = "quarantined"
QUALITY_TIERS = frozenset({QUALITY_STRICT, QUALITY_PROVISIONAL, QUALITY_QUARANTINED})

RAW_DAILY_ASTOCK = "baostock_daily_astock_raw"
RAW_DAILY_ETF = "baostock_daily_etf_raw"
RAW_ADJUST_FACTOR_EVENT = "baostock_adjust_factor_event_raw"
RAW_ADJUST_FACTOR_SYMBOL_HISTORY = "baostock_adjust_factor_symbol_history_raw"
RAW_ALL_STOCK = "baostock_all_stock_raw"
RAW_TRADING_CALENDAR = "baostock_trading_calendar_raw"
RAW_SECURITY_MASTER = "baostock_security_master_raw"
RAW_INTRADAY_5M_MOOTDX = "mootdx_intraday_5m_raw"
RAW_INTRADAY_5M_BAOSTOCK = "baostock_intraday_5m_raw"
RAW_INTRADAY_5M_SELECTED = "qdp_intraday_5m_selected_raw"
RAW_INTRADAY_1M_MOOTDX = "mootdx_intraday_1m_raw"
RAW_TUSHARE_PROXY_STOCK_BASIC = "tushare_proxy_stock_basic_raw"
RAW_TUSHARE_PROXY_TRADE_CALENDAR = "tushare_proxy_trade_calendar_raw"
RAW_TUSHARE_PROXY_DAILY = "tushare_proxy_daily_raw"
RAW_TUSHARE_PROXY_DAILY_BASIC = "tushare_proxy_daily_basic_raw"
RAW_TUSHARE_PROXY_ADJ_FACTOR = "tushare_proxy_adj_factor_raw"
RAW_TUSHARE_PROXY_NAMECHANGE = "tushare_proxy_namechange_raw"
RAW_TUSHARE_PROXY_SUSPEND = "tushare_proxy_suspend_raw"
RAW_TUSHARE_PROXY_STK_LIMIT = "tushare_proxy_stk_limit_raw"
RAW_TUSHARE_PROXY_DIVIDEND = "tushare_proxy_dividend_raw"
RAW_TUSHARE_PROXY_FINANCIAL = "tushare_proxy_financial_raw"
RAW_TUSHARE_PROXY_INTRADAY_1M = "tushare_proxy_intraday_1m_raw"
RAW_TUSHARE_PROXY_INTRADAY_5M = "tushare_proxy_intraday_5m_raw"
RAW_CORPORATE_ACTION_XDXR = "mootdx_corporate_action_xdxr_raw"
RAW_FINANCIAL_QUARTERLY = "baostock_financial_quarterly_raw"
RAW_PERFORMANCE_FORECAST = "baostock_performance_forecast_raw"
RAW_PERFORMANCE_EXPRESS = "baostock_performance_express_raw"
RAW_INDUSTRY_SNAPSHOT = "baostock_industry_snapshot_raw"
RAW_INDEX_CONSTITUENTS = "baostock_index_constituents_raw"

DOMAIN_MARKET_DAILY_RAW = "market_daily_raw"
DOMAIN_SECURITY_STATUS_DAILY = "security_status_daily"
DOMAIN_VALUATION_DAILY = "valuation_daily"
DOMAIN_TRADING_CALENDAR = "trading_calendar"
DOMAIN_SECURITY_IDENTITY = "security_identity"
DOMAIN_SYMBOL_HISTORY = "symbol_history"
DOMAIN_ELIGIBLE_SIGNAL_D = "eligible_signal_D"
DOMAIN_TRADABLE_OPEN_D1 = "tradable_open_D1"
DOMAIN_ADJUST_FACTOR_EVENT = "adjust_factor_event"
DOMAIN_ADJUST_FACTOR_DAILY = "adjust_factor_daily"
DOMAIN_MARKET_INTRADAY_1M = "market_intraday_1m"
DOMAIN_MARKET_INTRADAY_5M = "market_intraday_5m"
DOMAIN_CORPORATE_ACTIONS = "corporate_actions"
DOMAIN_SHARE_CAPITAL_EVENT = "share_capital_event"
DOMAIN_SHARE_CAPITAL_DAILY = "share_capital_daily"
DOMAIN_FINANCIAL_QUARTERLY = "financial_quarterly"
DOMAIN_PERFORMANCE_FORECAST = "performance_forecast"
DOMAIN_PERFORMANCE_EXPRESS = "performance_express"
DOMAIN_INDUSTRY = "industry_concept"
DOMAIN_INDEX_CONSTITUENTS = "index_constituents"

CORE_CANDIDATE_DOMAINS = (
    DOMAIN_TRADING_CALENDAR,
    DOMAIN_SECURITY_IDENTITY,
    DOMAIN_SYMBOL_HISTORY,
    DOMAIN_MARKET_DAILY_RAW,
    DOMAIN_SECURITY_STATUS_DAILY,
    DOMAIN_VALUATION_DAILY,
    DOMAIN_ADJUST_FACTOR_EVENT,
    DOMAIN_ADJUST_FACTOR_DAILY,
)

STRICT_RELEASE_DOMAINS = (
    DOMAIN_TRADING_CALENDAR,
    DOMAIN_SECURITY_IDENTITY,
    DOMAIN_SYMBOL_HISTORY,
    DOMAIN_MARKET_DAILY_RAW,
    DOMAIN_SECURITY_STATUS_DAILY,
    DOMAIN_VALUATION_DAILY,
    DOMAIN_ADJUST_FACTOR_EVENT,
    DOMAIN_ADJUST_FACTOR_DAILY,
    DOMAIN_ELIGIBLE_SIGNAL_D,
    DOMAIN_TRADABLE_OPEN_D1,
    DOMAIN_MARKET_INTRADAY_1M,
    DOMAIN_MARKET_INTRADAY_5M,
)

BAOSTOCK_DAILY_FIELDS = (
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "adjustflag",
    "turn",
    "tradestatus",
    "pctChg",
    "peTTM",
    "pbMRQ",
    "psTTM",
    "pcfNcfTTM",
    "isST",
)

BAOSTOCK_FACTOR_FIELD_ALIASES = (
    "adjustFacto",
    "adjustFactor",
    "adjust_factor",
)

PACKAGE_LOCK = {
    "baostock": {
        "version": BAOSTOCK_BATCH_VERSION,
        "wheel_sha256": BAOSTOCK_BATCH_WHEEL_SHA256,
    },
    "mootdx": {"version": MOOTDX_VERSION},
    "tushare_proxy": {
        "protocol": "tushare_compatible_http",
        "endpoint": "https://ts.gyzcloud.top/api",
        "upstream_provenance": "not_exposed",
        "production_role": "historical_bootstrap",
    },
}

EXPECTED_1M_BAR_ENDS = tuple(
    [f"{hour:02d}:{minute:02d}" for hour, minute in (
        *[(9, minute) for minute in range(31, 60)],
        *[(10, minute) for minute in range(0, 60)],
        *[(11, minute) for minute in range(0, 31)],
        *[(13, minute) for minute in range(1, 60)],
        *[(14, minute) for minute in range(0, 60)],
        (15, 0),
    )]
)

assert len(EXPECTED_1M_BAR_ENDS) == 240

EXPECTED_5M_BAR_ENDS = tuple(
    [f"{hour:02d}:{minute:02d}" for hour, minute in (
        *[(9, minute) for minute in range(35, 60, 5)],
        *[(10, minute) for minute in range(0, 60, 5)],
        *[(11, minute) for minute in range(0, 31, 5)],
        *[(13, minute) for minute in range(5, 60, 5)],
        *[(14, minute) for minute in range(0, 60, 5)],
        (15, 0),
    )]
)

assert len(EXPECTED_5M_BAR_ENDS) == 48
