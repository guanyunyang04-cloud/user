"""Domain contract coverage definitions."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import (
    market_business_dates,
)
from .market import (
    valid_market_rows,
)
from .requests import (
    DomainFetchRequest,
    FetchRequest,
)
from .schema import (
    DataDomain,
)


def coverage_report_for_frame(frame: pd.DataFrame, request: FetchRequest, *, provider: str) -> dict[str, Any]:
    dates = market_business_dates(request.start_date, request.end_date)
    expected_rows = int(len(request.symbols) * len(dates))
    row_count = len(frame)
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
    row_count = len(frame)
    unique_symbols = int(frame["symbol"].nunique()) if "symbol" in frame.columns and not frame.empty else 0
    unique_dates = int(frame["trade_date"].nunique()) if "trade_date" in frame.columns and not frame.empty else 0
    if request.domain == DataDomain.TRADING_CALENDAR:
        expected_rows = len(market_business_dates(request.start_date, request.end_date))
    elif request.symbols:
        expected_rows = int(
            len(request.symbols) * max(1, len(market_business_dates(request.start_date, request.end_date)))
        )
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
