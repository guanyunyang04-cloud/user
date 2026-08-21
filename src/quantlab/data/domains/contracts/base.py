"""Domain contract base definitions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .schema import (
    DOMAIN_STANDARD_COLUMNS,
    TDX_FAMILY_PROVIDER_NAMES,
    DataDomain,
)


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
        "research_report_forecast": DataDomain.RESEARCH_REPORT_FORECAST,
        "report_forecast": DataDomain.RESEARCH_REPORT_FORECAST,
        "stk_factor_pro": DataDomain.STK_FACTOR_PRO_RAW,
        "technical_factor": DataDomain.STK_FACTOR_PRO_RAW,
        "margin_market": DataDomain.MARGIN_MARKET,
        "margin_detail": DataDomain.MARGIN_DETAIL,
        "margin_eligibility": DataDomain.MARGIN_ELIGIBILITY,
        "margin_secs": DataDomain.MARGIN_SECS,
        "moneyflow": DataDomain.MONEYFLOW_RAW,
        "moneyflow_raw": DataDomain.MONEYFLOW_RAW,
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
            f"provider={provider_name}; use `qdp update` with the current providers."
        )
    return normalized


def market_business_dates(start_date: str, end_date: str) -> list[str]:
    start_ts = pd.Timestamp(_normalize_date(start_date))
    end_ts = pd.Timestamp(_normalize_date(end_date))
    if end_ts < start_ts:
        return []
    return [pd.Timestamp(item).strftime("%Y-%m-%d") for item in pd.bdate_range(start_ts, end_ts)]


def trading_dates_from_calendar(calendar: pd.DataFrame, start_date: str, end_date: str) -> list[str]:
    # Calendar normalization depends on the shared helpers below, so importing
    # it at call time keeps the module dependency one-way.
    from .snapshots import normalize_calendar_frame

    if calendar is None or calendar.empty:
        return market_business_dates(start_date, end_date)
    data = normalize_calendar_frame(
        calendar,
        source=str(calendar["source"].iloc[0] if "source" in calendar.columns and len(calendar) else "calendar"),
        require_columns=False,
    )
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
    now_ts = (
        pd.Timestamp(reference_ts).tz_localize(None)
        if reference_ts is not None
        else pd.Timestamp.now().tz_localize(None)
    )
    close_clock = pd.Timestamp(close_time).time()
    today = now_ts.normalize()
    is_business_day = today.weekday() < 5
    include_today = is_business_day and now_ts.time() >= close_clock
    offset = 0 if include_today else 1
    return (today - pd.offsets.BDay(offset)).strftime("%Y-%m-%d")


def _normalize_date(value: Any) -> str:
    if value is None or str(value).strip() == "":
        raise ValueError("date cannot be empty")
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _normalize_timestamp(value: Any) -> str:
    if value is None or str(value).strip() == "":
        raise ValueError("timestamp cannot be empty")
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Shanghai").tz_localize(None)
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


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
        _rename_first(
            data, "float_share", ("float_share", "float_shares", "流通股本", "流通股本(股)", "流通股本（股）")
        )
        _rename_first(data, "total_share", ("total_share", "total_shares", "总股本", "总股本(股)", "总股本（股）"))


def _require_domain_columns(data: pd.DataFrame, domain: str, *, require_columns: bool) -> None:
    if not require_columns:
        return
    required = set(DOMAIN_STANDARD_COLUMNS[normalize_domain(domain)])
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"provider_frame_schema_error: domain={domain} missing columns {missing}")


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
    return text in {"1", "true", "t", "yes", "y", "open", "交易", "正常", "是"}


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
