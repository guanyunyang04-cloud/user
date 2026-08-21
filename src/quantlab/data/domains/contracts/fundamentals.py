"""Domain contract fundamentals definitions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import (
    _coerce_trade_date,
    _date_series,
    _ensure_domain_columns,
    _normalize_symbol,
    _prepare_domain_frame,
    _rename_first,
    _source_series,
    normalize_domain,
    validate_provider_name,
)
from .market import (
    _require_core_columns,
)
from .schema import (
    DOMAIN_STANDARD_COLUMNS,
    DataDomain,
)


def normalize_financial_quarterly_frame(
    frame: pd.DataFrame, *, source: str, require_columns: bool = True
) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(
        frame, domain=DataDomain.FINANCIAL_QUARTERLY, source=provider, as_of_date="", require_columns=False
    )
    _rename_financial_common_columns(data)
    _rename_first(data, "roe_avg", ("roeAvg", "roe_avg", "净资产收益率", "平均净资产收益率"))
    _rename_first(data, "net_profit_margin", ("npMargin", "netProfitMargin", "net_profit_margin", "销售净利率"))
    _rename_first(data, "gross_profit_margin", ("gpMargin", "grossProfitMargin", "gross_profit_margin", "销售毛利率"))
    _rename_first(
        data, "net_profit_yoy", ("YOYPNI", "YOYNI", "netProfitGrowRate", "net_profit_yoy", "净利润同比增长率")
    )
    _rename_first(
        data, "revenue_yoy", ("YOYIncome", "YOYRevenue", "revenueGrowRate", "revenue_yoy", "营业收入同比增长率")
    )
    _rename_first(data, "eps", ("epsTTM", "eps", "每股收益"))
    _rename_first(data, "net_profit", ("netProfit", "net_profit", "归属母公司股东的净利润"))
    _rename_first(data, "revenue", ("revenue", "totalRevenue", "营业总收入"))
    _rename_first(data, "asset_turnover", ("NRTurnRatio", "asset_turnover", "总资产周转率"))
    _rename_first(data, "debt_to_asset", ("liabilityToAsset", "debt_to_asset", "资产负债率"))
    _rename_first(data, "current_ratio", ("currentRatio", "current_ratio", "流动比率"))
    _rename_first(
        data,
        "cash_flow_ps",
        ("cashFlowPS", "ocfps", "cash_flow_ps", "每股经营现金流"),
    )
    _require_core_columns(
        data, DataDomain.FINANCIAL_QUARTERLY, {"symbol", "report_date"}, require_columns=require_columns
    )
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


def normalize_performance_forecast_frame(
    frame: pd.DataFrame, *, source: str, require_columns: bool = True
) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(
        frame, domain=DataDomain.PERFORMANCE_FORECAST, source=provider, as_of_date="", require_columns=False
    )
    _rename_financial_common_columns(data)
    _rename_first(data, "forecast_type", ("profitForcastType", "forecastType", "type", "业绩预告类型"))
    _rename_first(data, "profit_min", ("profitMin", "profit_min", "预告净利润下限"))
    _rename_first(data, "profit_max", ("profitMax", "profit_max", "预告净利润上限"))
    _rename_first(
        data,
        "profit_change_min",
        ("profitForcastChgPctDwn", "profitForcastChgPctMin", "profit_change_min", "预告净利润变动下限"),
    )
    _rename_first(
        data,
        "profit_change_max",
        ("profitForcastChgPctUp", "profitForcastChgPctMax", "profit_change_max", "预告净利润变动上限"),
    )
    _require_core_columns(
        data,
        DataDomain.PERFORMANCE_FORECAST,
        {"symbol", "report_date", "publish_date"},
        require_columns=require_columns,
    )
    data = _ensure_report_domain_columns(data, DataDomain.PERFORMANCE_FORECAST, provider=provider)
    for column in [
        "fiscal_year",
        "fiscal_quarter",
        "profit_min",
        "profit_max",
        "profit_change_min",
        "profit_change_max",
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["forecast_type"] = data["forecast_type"].fillna("").astype(str).str.strip()
    return _finalize_report_domain_frame(data, DataDomain.PERFORMANCE_FORECAST)


def normalize_performance_express_frame(
    frame: pd.DataFrame, *, source: str, require_columns: bool = True
) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(
        frame, domain=DataDomain.PERFORMANCE_EXPRESS, source=provider, as_of_date="", require_columns=False
    )
    _rename_financial_common_columns(data)
    _rename_first(data, "eps", ("performanceExpressEPSDiluted", "performanceExpressEPSBasic", "eps", "EPS", "每股收益"))
    _rename_first(data, "roe", ("performanceExpressROEWa", "roe", "ROE", "净资产收益率"))
    _rename_first(
        data, "net_profit", ("performanceExpressNetProfit", "netProfit", "net_profit", "归属母公司股东的净利润")
    )
    _rename_first(
        data,
        "revenue",
        (
            "performanceExpressTotalIncome",
            "totalRevenue",
            "revenue",
            "营业总收入",
        ),
    )
    _rename_first(data, "total_assets", ("performanceExpressTotalAsset", "totalAssets", "total_assets", "总资产"))
    _require_core_columns(
        data, DataDomain.PERFORMANCE_EXPRESS, {"symbol", "report_date", "publish_date"}, require_columns=require_columns
    )
    data = _ensure_report_domain_columns(data, DataDomain.PERFORMANCE_EXPRESS, provider=provider)
    for column in ["fiscal_year", "fiscal_quarter", "eps", "roe", "net_profit", "revenue", "total_assets"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return _finalize_report_domain_frame(data, DataDomain.PERFORMANCE_EXPRESS)


def normalize_corporate_actions_frame(
    frame: pd.DataFrame, *, source: str, as_of_date: str, require_columns: bool = True
) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(
        frame, domain=DataDomain.CORPORATE_ACTIONS, source=provider, as_of_date="", require_columns=False
    )
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
    data["trade_date"] = (
        data["trade_date"].replace("NaT", np.nan).fillna(data["announcement_date"]).fillna(str(as_of_date or ""))
    )
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


def normalize_share_capital_frame(
    frame: pd.DataFrame, *, source: str, as_of_date: str, require_columns: bool = True
) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(
        frame, domain=DataDomain.SHARE_CAPITAL, source=provider, as_of_date="", require_columns=False
    )
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
            data["symbol"].astype(str).str.len().gt(0) & data["trade_date"].astype(str).str.lower().ne("nat"),
            DOMAIN_STANDARD_COLUMNS[DataDomain.SHARE_CAPITAL],
        ]
        .drop_duplicates(subset=["trade_date", "symbol", "source"], keep="last")
        .sort_values(["trade_date", "symbol"])
        .reset_index(drop=True)
    )


def normalize_name_change_frame(
    frame: pd.DataFrame, *, source: str, as_of_date: str, require_columns: bool = True
) -> pd.DataFrame:
    provider = validate_provider_name(source)
    data = _prepare_domain_frame(
        frame, domain=DataDomain.NAME_CHANGE, source=provider, as_of_date="", require_columns=False
    )
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
            data["symbol"].astype(str).str.len().gt(0) & data["trade_date"].astype(str).str.lower().ne("nat"),
            DOMAIN_STANDARD_COLUMNS[DataDomain.NAME_CHANGE],
        ]
        .drop_duplicates()
        .sort_values(["trade_date", "symbol", "change_type"])
        .reset_index(drop=True)
    )


def _rename_financial_common_columns(data: pd.DataFrame) -> None:
    _rename_first(data, "symbol", ("code", "stock", "ts_code", "股票代码", "证券代码"))
    _rename_first(
        data,
        "report_date",
        (
            "statDate",
            "reportDate",
            "endDate",
            "performanceExpStatDate",
            "profitForcastExpStatDate",
            "报告日期",
            "统计日期",
        ),
    )
    _rename_first(
        data,
        "publish_date",
        (
            "pubDate",
            "publishDate",
            "performanceExpPubDate",
            "performanceExpUpdateDate",
            "profitForcastExpPubDate",
            "公告日期",
            "发布日期",
            "更新日期",
        ),
    )
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
        data["symbol"].astype(str).str.len().gt(0) & data["trade_date"].astype(str).str.lower().ne("nat"),
        columns,
    ]
    sort_columns = [
        column
        for column in ["trade_date", "symbol", "fiscal_year", "fiscal_quarter", "source"]
        if column in out.columns
    ]
    return out.drop_duplicates(subset=sort_columns, keep="last").sort_values(sort_columns).reset_index(drop=True)
