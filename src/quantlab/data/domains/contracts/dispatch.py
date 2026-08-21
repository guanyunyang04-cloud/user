"""Domain contract dispatch definitions."""

from __future__ import annotations

import pandas as pd

from .base import (
    normalize_domain,
)
from .fundamentals import (
    normalize_corporate_actions_frame,
    normalize_financial_quarterly_frame,
    normalize_name_change_frame,
    normalize_performance_express_frame,
    normalize_performance_forecast_frame,
    normalize_share_capital_frame,
)
from .intraday import (
    normalize_intraday_1m_frame,
    normalize_intraday_5m_frame,
    normalize_intraday_daily_features_frame,
)
from .market import (
    normalize_adjust_factor_frame,
    normalize_market_frame,
)
from .schema import (
    DataDomain,
)
from .snapshots import (
    normalize_calendar_frame,
    normalize_generic_text_domain_frame,
    normalize_index_constituents_frame,
    normalize_industry_concept_frame,
    normalize_limit_frame,
    normalize_money_flow_hotspot_frame,
    normalize_status_frame,
    normalize_universe_frame,
    normalize_valuation_frame,
)


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
        return normalize_market_frame(
            frame, source=source, adjusted_flag=adjusted_flag, require_columns=require_columns
        )
    if normalized_domain == DataDomain.MARKET_INTRADAY_5M:
        return normalize_intraday_5m_frame(
            frame, source=source, adjusted_flag=adjusted_flag, require_columns=require_columns
        )
    if normalized_domain == DataDomain.MARKET_INTRADAY_1M:
        return normalize_intraday_1m_frame(
            frame, source=source, adjusted_flag=adjusted_flag, require_columns=require_columns
        )
    if normalized_domain == DataDomain.INTRADAY_DAILY_FEATURES:
        return normalize_intraday_daily_features_frame(
            frame, source=source, adjusted_flag=adjusted_flag, require_columns=require_columns
        )
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
        return normalize_industry_concept_frame(
            frame, source=source, as_of_date=as_of_date, require_columns=require_columns
        )
    if normalized_domain == DataDomain.VALUATION:
        return normalize_valuation_frame(frame, source=source, as_of_date=as_of_date, require_columns=require_columns)
    if normalized_domain == DataDomain.INDEX_CONSTITUENTS:
        return normalize_index_constituents_frame(
            frame, source=source, as_of_date=as_of_date, require_columns=require_columns
        )
    if normalized_domain == DataDomain.FINANCIAL_QUARTERLY:
        return normalize_financial_quarterly_frame(frame, source=source, require_columns=require_columns)
    if normalized_domain == DataDomain.PERFORMANCE_FORECAST:
        return normalize_performance_forecast_frame(frame, source=source, require_columns=require_columns)
    if normalized_domain == DataDomain.PERFORMANCE_EXPRESS:
        return normalize_performance_express_frame(frame, source=source, require_columns=require_columns)
    if normalized_domain == DataDomain.CORPORATE_ACTIONS:
        return normalize_corporate_actions_frame(
            frame, source=source, as_of_date=as_of_date, require_columns=require_columns
        )
    if normalized_domain == DataDomain.SHARE_CAPITAL:
        return normalize_share_capital_frame(
            frame, source=source, as_of_date=as_of_date, require_columns=require_columns
        )
    if normalized_domain == DataDomain.NAME_CHANGE:
        return normalize_name_change_frame(frame, source=source, as_of_date=as_of_date, require_columns=require_columns)
    if normalized_domain == DataDomain.MONEY_FLOW_HOTSPOT:
        return normalize_money_flow_hotspot_frame(
            frame, source=source, as_of_date=as_of_date, require_columns=require_columns
        )
    if normalized_domain in {
        DataDomain.NEWS_EVENT,
        DataDomain.ANNOUNCEMENT,
        DataDomain.RESEARCH_REPORT,
        DataDomain.RESEARCH_REPORT_FORECAST,
        DataDomain.IWENCAI_SEMANTIC,
    }:
        return normalize_generic_text_domain_frame(
            frame, domain=normalized_domain, source=source, as_of_date=as_of_date, require_columns=require_columns
        )
    raise ValueError(f"unsupported data domain: {domain}")
