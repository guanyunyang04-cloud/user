"""Pure payload normalization for auxiliary QDP domains.

The update orchestration module owns network calls, DuckDB mutations, and
retry policy.  These functions only turn provider-shaped frames into the
canonical QDP schemas, which keeps them deterministic and easy to test.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd


def provider_date(value: Any, fallback: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.Timestamp(fallback)
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def from_baostock_code(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("sh.") and text[3:].isdigit():
        return f"{text[3:]}.SH"
    if text.startswith("sz.") and text[3:].isdigit():
        return f"{text[3:]}.SZ"
    return ""


def normalize_comparison_text(value: Any) -> str:
    if value is None or bool(pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return "".join(text.split())


def normalize_industry_comparison(value: Any) -> str:
    return re.sub(r"^[a-z]\d{1,4}", "", normalize_comparison_text(value))


def tushare_date_series(frame: pd.DataFrame, column: str) -> pd.Series:
    raw = frame.get(column, pd.Series(index=frame.index, dtype=object))
    text = raw.fillna("").astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )


CNINFO_INDUSTRY_COLUMNS = (
    "symbol",
    "source_date",
    "industry",
    "industry_standard",
    "source",
)


def normalize_cninfo_industry_history(
    frame: pd.DataFrame,
    *,
    symbol: str,
    target_date: str,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=CNINFO_INDUSTRY_COLUMNS)
    data = frame.copy()
    for column in ("变更日期", "分类标准编码", "分类标准", "行业大类"):
        if column not in data:
            data[column] = np.nan
    code = data["分类标准编码"].fillna("").astype(str).str.strip()
    standard = data["分类标准"].fillna("").astype(str).str.strip()
    csrc = code.isin({"008001", "008009", "008021"}) | standard.str.contains(
        "证监会行业分类", regex=False
    )
    data = data.loc[csrc].copy()
    data["source_date"] = pd.to_datetime(
        data["变更日期"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    data["industry"] = data["行业大类"].fillna("").astype(str).str.strip()
    data["symbol"] = str(symbol).upper()
    data["industry_standard"] = "证监会行业分类"
    data["source"] = "akshare_cninfo_csrc_industry_history"
    data["priority"] = code.map({"008001": 3, "008021": 2, "008009": 1}).fillna(0)
    return (
        data.loc[
            data["source_date"].notna()
            & data["source_date"].le(target_date)
            & data["industry"].ne(""),
            [*CNINFO_INDUSTRY_COLUMNS, "priority"],
        ]
        .sort_values(["source_date", "priority"])
        .drop_duplicates(["symbol", "source_date"], keep="last")
        .loc[:, list(CNINFO_INDUSTRY_COLUMNS)]
        .reset_index(drop=True)
    )


NAME_CHANGE_FIELDS = (
    "ts_code",
    "name",
    "start_date",
    "end_date",
    "ann_date",
    "change_reason",
)
NAME_INTERVAL_COLUMNS = (
    "symbol",
    "name",
    "start_date",
    "end_date",
    "announcement_date",
    "change_reason",
    "source",
)


def normalize_name_intervals(
    frame: pd.DataFrame,
    *,
    target_date: str,
) -> pd.DataFrame:
    data = frame.copy()
    for column in NAME_CHANGE_FIELDS:
        if column not in data.columns:
            data[column] = np.nan
    data["symbol"] = data["ts_code"].fillna("").astype(str).str.upper()
    data["name"] = data["name"].fillna("").astype(str).str.strip()
    data["start_date"] = tushare_date_series(data, "start_date")
    data["end_date"] = tushare_date_series(data, "end_date")
    data["announcement_date"] = tushare_date_series(data, "ann_date")
    data["change_reason"] = data["change_reason"].fillna("").astype(str).str.strip()
    data["source"] = "tushare_namechange_intervals"
    valid = (
        data["symbol"].str.match(r"^\d{6}\.(SH|SZ)$", na=False)
        & data["name"].ne("")
        & data["start_date"].notna()
        & data["start_date"].le(target_date)
    )
    return (
        data.loc[valid, list(NAME_INTERVAL_COLUMNS)]
        .drop_duplicates(["symbol", "start_date"], keep="last")
        .sort_values(["symbol", "start_date"])
        .reset_index(drop=True)
    )


DAILY_BASIC_FIELDS = (
    "ts_code",
    "trade_date",
    "close",
    "turnover_rate",
    "pe_ttm",
    "pb",
    "total_share",
    "float_share",
    "total_mv",
    "circ_mv",
)
DAILY_BASIC_NORMALIZED_COLUMNS = (
    "symbol",
    "trade_date",
    "close",
    "turnover_rate",
    "pe_ttm",
    "pb",
    "total_share",
    "float_share",
    "total_mv",
    "circ_mv",
    "source",
)
CNINFO_SHARE_NORMALIZED_COLUMNS = (
    "symbol",
    "variation_date",
    "source_date",
    "total_share",
    "float_share",
    "source",
)


def normalize_daily_basic(frame: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    data = frame.copy()
    for column in DAILY_BASIC_FIELDS:
        if column not in data.columns:
            data[column] = np.nan
    data["symbol"] = data["ts_code"].fillna("").astype(str).str.upper()
    data["trade_date"] = (
        data["trade_date"]
        .fillna(str(trade_date).replace("-", ""))
        .astype(str)
        .str.replace("-", "", regex=False)
        .str.slice(0, 8)
    )
    data["trade_date"] = pd.to_datetime(
        data["trade_date"], format="%Y%m%d", errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    for column in (
        "close",
        "turnover_rate",
        "pe_ttm",
        "pb",
        "total_share",
        "float_share",
        "total_mv",
        "circ_mv",
    ):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["total_share"] *= 10_000.0
    data["float_share"] *= 10_000.0
    data["total_mv"] *= 10_000.0
    data["circ_mv"] *= 10_000.0
    data["source"] = "tushare_daily_basic_one_time_gap_fill"
    return (
        data.loc[
            data["symbol"].str.match(r"^\d{6}\.(SH|SZ)$", na=False)
            & data["trade_date"].notna(),
            list(DAILY_BASIC_NORMALIZED_COLUMNS),
        ]
        .drop_duplicates(["symbol", "trade_date"], keep="last")
        .reset_index(drop=True)
    )


def normalize_cninfo_share_change(
    frame: pd.DataFrame,
    *,
    symbol: str,
    target_date: str,
    source: str = "akshare_cninfo_share_history",
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(CNINFO_SHARE_NORMALIZED_COLUMNS))
    data = frame.copy()

    def values(column: str) -> pd.Series:
        if column in data.columns:
            return data[column]
        return pd.Series(index=data.index, dtype="object")

    variation = pd.to_datetime(values("变动日期"), errors="coerce")
    announcement = pd.to_datetime(values("公告日期"), errors="coerce")
    visible_date = pd.concat([variation, announcement], axis=1).max(axis=1)
    total = (pd.to_numeric(values("总股本"), errors="coerce") * 10_000.0).round()
    domestic_a_share = pd.to_numeric(values("人民币普通股"), errors="coerce")
    all_circulating = pd.to_numeric(values("已流通股份"), errors="coerce")
    floating = (domestic_a_share.fillna(all_circulating) * 10_000.0).round()
    normalized = pd.DataFrame(
        {
            "symbol": str(symbol).upper(),
            "variation_date": variation.dt.strftime("%Y-%m-%d"),
            "source_date": visible_date.dt.strftime("%Y-%m-%d"),
            "total_share": total,
            "float_share": floating,
            "source": str(source),
        }
    )
    return normalized.loc[
        normalized["source_date"].notna()
        & normalized["source_date"].le(target_date)
        & normalized["total_share"].gt(0)
        & normalized["float_share"].ge(0)
        & normalized["float_share"].le(normalized["total_share"]),
        list(CNINFO_SHARE_NORMALIZED_COLUMNS),
    ].reset_index(drop=True)


DIVIDEND_FIELDS = (
    "ts_code",
    "end_date",
    "ann_date",
    "div_proc",
    "stk_div",
    "stk_bo_rate",
    "stk_co_rate",
    "cash_div_tax",
    "cash_div",
    "record_date",
    "ex_date",
    "pay_date",
    "imp_ann_date",
)
CORPORATE_COLUMNS = (
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
)
MOOTDX_CORPORATE_VALIDATION_COLUMNS = (
    "symbol",
    "trade_date",
    "cash_dividend_per_10",
    "source",
)


def normalize_dividend(frame: pd.DataFrame, *, target_date: str) -> pd.DataFrame:
    data = frame.copy()
    for column in DIVIDEND_FIELDS:
        if column not in data.columns:
            data[column] = np.nan
    data = data.loc[
        data["div_proc"].fillna("").astype(str).str.strip().eq("实施")
    ].copy()
    data["symbol"] = data["ts_code"].fillna("").astype(str).str.upper()
    data["announcement_date"] = tushare_date_series(data, "ann_date").fillna(
        tushare_date_series(data, "imp_ann_date")
    )
    data["ex_date"] = tushare_date_series(data, "ex_date")
    data["trade_date"] = data["ex_date"]
    data["record_date"] = tushare_date_series(data, "record_date")
    data["dividend_pay_date"] = tushare_date_series(data, "pay_date")
    cash = (
        pd.to_numeric(data["cash_div_tax"], errors="coerce")
        .fillna(pd.to_numeric(data["cash_div"], errors="coerce"))
        .fillna(0.0)
        * 10.0
    )
    bonus = pd.to_numeric(data["stk_bo_rate"], errors="coerce").fillna(0.0) * 10.0
    transfer = pd.to_numeric(data["stk_co_rate"], errors="coerce").fillna(0.0) * 10.0
    combined_stock = pd.to_numeric(data["stk_div"], errors="coerce").fillna(0.0) * 10.0
    bonus = bonus.where((bonus + transfer) > 0, combined_stock)
    data["cash_dividend_per_10"] = cash
    data["bonus_share_per_10"] = bonus
    data["transfer_share_per_10"] = transfer
    has_cash = cash > 0
    has_stock = (bonus + transfer) > 0
    data["action_type"] = np.select(
        [has_cash & has_stock, has_stock, has_cash],
        ["cash_stock", "stock", "cash"],
        default="",
    )
    data["description"] = (
        "cash_per_10="
        + cash.round(8).astype(str)
        + ";bonus_per_10="
        + bonus.round(8).astype(str)
        + ";transfer_per_10="
        + transfer.round(8).astype(str)
    )
    data["source"] = "tushare_dividend_implemented"
    return (
        data.loc[
            data["symbol"].str.match(r"^\d{6}\.(SH|SZ)$", na=False)
            & data["trade_date"].between("2010-01-04", target_date)
            & data["action_type"].ne(""),
            list(CORPORATE_COLUMNS),
        ]
        .drop_duplicates(["symbol", "trade_date", "action_type"], keep="last")
        .sort_values(["trade_date", "symbol", "action_type"])
        .reset_index(drop=True)
    )


def normalize_cninfo_dividend(
    frame: pd.DataFrame,
    *,
    symbol: str,
    target_date: str,
) -> pd.DataFrame:
    data = frame.copy()
    for column in (
        "实施方案公告日期",
        "送股比例",
        "转增比例",
        "派息比例",
        "股权登记日",
        "除权日",
        "派息日",
        "实施方案分红说明",
    ):
        if column not in data:
            data[column] = np.nan
    data["symbol"] = str(symbol).upper()
    data["announcement_date"] = pd.to_datetime(
        data["实施方案公告日期"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    data["ex_date"] = pd.to_datetime(data["除权日"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    data["trade_date"] = data["ex_date"]
    data["record_date"] = pd.to_datetime(
        data["股权登记日"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    data["dividend_pay_date"] = pd.to_datetime(
        data["派息日"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    cash = pd.to_numeric(data["派息比例"], errors="coerce").fillna(0.0)
    bonus = pd.to_numeric(data["送股比例"], errors="coerce").fillna(0.0)
    transfer = pd.to_numeric(data["转增比例"], errors="coerce").fillna(0.0)
    data["cash_dividend_per_10"] = cash
    data["bonus_share_per_10"] = bonus
    data["transfer_share_per_10"] = transfer
    has_cash = cash > 0
    has_stock = (bonus + transfer) > 0
    data["action_type"] = np.select(
        [has_cash & has_stock, has_stock, has_cash],
        ["cash_stock", "stock", "cash"],
        default="",
    )
    data["description"] = data["实施方案分红说明"].fillna("").astype(str).str.strip()
    data["source"] = "akshare_cninfo_dividend_validation"
    return (
        data.loc[
            data["trade_date"].between("2010-01-04", target_date)
            & data["action_type"].ne(""),
            list(CORPORATE_COLUMNS),
        ]
        .drop_duplicates(["symbol", "trade_date", "action_type"], keep="last")
        .sort_values(["trade_date", "symbol", "action_type"])
        .reset_index(drop=True)
    )


__all__ = [
    "CORPORATE_COLUMNS",
    "CNINFO_INDUSTRY_COLUMNS",
    "CNINFO_SHARE_NORMALIZED_COLUMNS",
    "DAILY_BASIC_FIELDS",
    "DAILY_BASIC_NORMALIZED_COLUMNS",
    "DIVIDEND_FIELDS",
    "MOOTDX_CORPORATE_VALIDATION_COLUMNS",
    "NAME_CHANGE_FIELDS",
    "NAME_INTERVAL_COLUMNS",
    "from_baostock_code",
    "normalize_cninfo_dividend",
    "normalize_cninfo_industry_history",
    "normalize_cninfo_share_change",
    "normalize_comparison_text",
    "normalize_daily_basic",
    "normalize_dividend",
    "normalize_industry_comparison",
    "normalize_name_intervals",
    "provider_date",
    "tushare_date_series",
]
