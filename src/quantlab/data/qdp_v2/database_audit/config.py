"""Database audit config checks."""

from __future__ import annotations

from quantlab.data.qdp_v2.audit import VALUATION_REQUIRED_COLUMNS
from quantlab.data.qdp_v2.auxiliary_update import AUXILIARY_DOMAINS

REQUIRED_DOMAINS = {
    "trading_calendar",
    "security_identity",
    "symbol_history",
    "market_daily_raw",
    "security_status",
    "universe_snapshot",
    "adjust_factor",
    "market_intraday_5m",
    *AUXILIARY_DOMAINS,
}


REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "trading_calendar": ("trade_date", "is_open", "exchange", "source"),
    "security_identity": (
        "security_id",
        "exchange",
        "list_date",
        "current_symbol",
    ),
    "symbol_history": (
        "security_id",
        "symbol",
        "effective_from",
        "effective_to",
    ),
    "market_daily_raw": (
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
    ),
    "security_status": (
        "symbol",
        "trade_date",
        "is_st",
        "is_suspended",
        "is_delisted",
        "source",
    ),
    "universe_snapshot": (
        "symbol",
        "trade_date",
        "name",
        "exchange",
        "list_status",
        "source",
    ),
    "adjust_factor": (
        "symbol",
        "trade_date",
        "adjust_factor",
        "factor_provider",
        "factor_semantics",
        "source",
    ),
    "market_intraday_5m": (
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
    ),
    "industry_concept": (
        "symbol",
        "trade_date",
        "industry",
        "source",
        "industry_fill_method",
        "industry_source_date",
        "industry_standard",
    ),
    "share_capital": (
        "symbol",
        "trade_date",
        "total_share",
        "float_share",
        "restricted_share",
        "total_share_source_date",
        "float_share_source_date",
        "restricted_share_source_date",
        "share_fill_method",
        "source",
    ),
    "valuation": tuple(sorted(VALUATION_REQUIRED_COLUMNS)),
    "name_change": (
        "symbol",
        "trade_date",
        "old_name",
        "new_name",
        "change_type",
        "source",
    ),
    "corporate_actions": (
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
        "source",
    ),
    "index_constituents": (
        "index_symbol",
        "symbol",
        "trade_date",
        "index_name",
        "source",
        "source_snapshot_date",
    ),
}


AS_OF_DATE_COLUMNS: dict[str, str] = {
    "security_identity": "list_date",
    "symbol_history": "effective_from",
}
