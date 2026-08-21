"""Domain contract requests definitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

from .base import (
    _normalize_date,
    _normalize_symbol,
    _normalize_timestamp,
    normalize_domain,
)
from .schema import (
    DOMAIN_STANDARD_COLUMNS,
    STANDARD_MARKET_COLUMNS,
    DataDomain,
)


@dataclass(frozen=True)
class FetchRequest:
    symbols: tuple[str, ...]
    start_date: str
    end_date: str
    domain: str = "market_daily"
    adjusted_flag: str = "none"
    fields: tuple[str, ...] = tuple(STANDARD_MARKET_COLUMNS)

    def normalized(self) -> FetchRequest:
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

    def normalized(self) -> DomainFetchRequest:
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

    def normalized(self) -> DatePartitionFetchRequest:
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
class HistoryPageFetchRequest:
    """Request one reverse-chronological 5-minute history page.

    This contract is intentionally separate from both symbol/range and
    date-partition requests.  Providers that cap a response at ``page_size``
    must expose the next cursor explicitly; callers must not interpret a full
    page as end-of-history.
    """

    provider_symbol: str
    start_at: str
    end_at: str
    page_size: int = 8_000

    def normalized(self) -> HistoryPageFetchRequest:
        symbol = _normalize_symbol(self.provider_symbol)
        start_at = _normalize_timestamp(self.start_at)
        end_at = _normalize_timestamp(self.end_at)
        if pd.Timestamp(end_at) < pd.Timestamp(start_at):
            raise ValueError(f"history_page_invalid_range:{start_at}:{end_at}")
        page_size = int(self.page_size)
        if page_size < 1 or page_size > 8_000:
            raise ValueError(f"history_page_size_out_of_range:{page_size}")
        return HistoryPageFetchRequest(
            provider_symbol=symbol,
            start_at=start_at,
            end_at=end_at,
            page_size=page_size,
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


@dataclass(frozen=True)
class HistoryPageResult:
    provider: str
    request: HistoryPageFetchRequest
    raw_data: pd.DataFrame
    fields: tuple[str, ...]
    row_count: int
    min_timestamp: str
    max_timestamp: str
    next_end_at: str
    is_complete: bool
    request_metadata_without_token: dict[str, Any] = field(default_factory=dict)


class MarketProvider(Protocol):
    name: str

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult: ...


class DomainProvider(Protocol):
    name: str

    def fetch_domain(self, request: DomainFetchRequest) -> DomainProviderResult: ...


class DatePartitionProvider(Protocol):
    name: str

    def fetch_date_partition(self, request: DatePartitionFetchRequest) -> DatePartitionProviderResult: ...


class HistoryPageProvider(Protocol):
    name: str

    def fetch_history_page(self, request: HistoryPageFetchRequest) -> HistoryPageResult: ...


MarketProviderCallable = Callable[[FetchRequest], ProviderResult]
