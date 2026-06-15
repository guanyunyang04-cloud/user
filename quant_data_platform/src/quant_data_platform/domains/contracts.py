from __future__ import annotations

from daily_research.data_platform.contracts import DataDomain

CANONICAL_START_DATE = "2010-01-01"

CORE_MARKET_DOMAINS = (
    DataDomain.MARKET_DAILY,
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
