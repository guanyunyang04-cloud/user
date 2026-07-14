from __future__ import annotations

import os
import io
import json
import math
import multiprocessing
import queue as queue_module
import threading
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, distribution as package_distribution, version as package_version
from typing import Any

import numpy as np
import pandas as pd
import requests

from quant_data_platform.domains.contracts import (
    DataDomain,
    DatePartitionFetchRequest,
    DatePartitionProviderResult,
    DomainFetchRequest,
    FetchRequest,
    ProviderResult,
    build_intraday_daily_feature_frame,
    coverage_report_for_domain,
    coverage_report_for_frame,
    normalize_domain_frame,
    normalize_market_frame,
    validate_provider_name,
)
from quant_data_platform.progress import create_progress, progress_write


FORMAL_FREE_V3_REQUIRED_DOMAINS: tuple[str, ...] = (
    DataDomain.MARKET_DAILY,
    DataDomain.TRADING_CALENDAR,
    DataDomain.UNIVERSE_SNAPSHOT,
    DataDomain.SECURITY_STATUS,
    DataDomain.LIMIT_STATUS,
)
FORMAL_FREE_V3_OPTIONAL_DOMAINS: tuple[str, ...] = (
    DataDomain.VALUATION,
    DataDomain.INDUSTRY_CONCEPT,
    DataDomain.INDEX_CONSTITUENTS,
    DataDomain.ADJUST_FACTOR,
    DataDomain.INTRADAY_DAILY_FEATURES,
    DataDomain.FINANCIAL_QUARTERLY,
    DataDomain.PERFORMANCE_FORECAST,
    DataDomain.PERFORMANCE_EXPRESS,
    DataDomain.MONEY_FLOW_HOTSPOT,
)
FORMAL_FREE_V3_RESEARCH_FUTURE_DOMAINS: tuple[str, ...] = (
    DataDomain.NEWS_EVENT,
    DataDomain.ANNOUNCEMENT,
    DataDomain.RESEARCH_REPORT,
    DataDomain.IWENCAI_SEMANTIC,
)
RESEARCH_REBUILD_MINIMAL_REQUIRED_DOMAINS: tuple[str, ...] = (
    DataDomain.MARKET_DAILY,
    DataDomain.TRADING_CALENDAR,
    DataDomain.UNIVERSE_SNAPSHOT,
)
QDP_PRODUCTION_V1_REQUIRED_DOMAINS: tuple[str, ...] = (
    DataDomain.MARKET_DAILY,
    DataDomain.TRADING_CALENDAR,
    DataDomain.UNIVERSE_SNAPSHOT,
    DataDomain.SECURITY_STATUS,
)
QDP_PRODUCTION_V1_OPTIONAL_DOMAINS: tuple[str, ...] = (
    DataDomain.MARKET_INTRADAY_1M,
    DataDomain.MARKET_INTRADAY_5M,
    DataDomain.INTRADAY_DAILY_FEATURES,
    DataDomain.ADJUST_FACTOR,
    DataDomain.VALUATION,
    DataDomain.INDUSTRY_CONCEPT,
    DataDomain.INDEX_CONSTITUENTS,
    DataDomain.FINANCIAL_QUARTERLY,
    DataDomain.PERFORMANCE_FORECAST,
    DataDomain.PERFORMANCE_EXPRESS,
)
QDP_PRODUCTION_V1_RESEARCH_FUTURE_DOMAINS: tuple[str, ...] = (
    DataDomain.ANNOUNCEMENT,
)
QDP_PRODUCTION_V2_REQUIRED_DOMAINS: tuple[str, ...] = (
    DataDomain.MARKET_DAILY,
    DataDomain.TRADING_CALENDAR,
    DataDomain.SECURITY_IDENTITY,
    DataDomain.SYMBOL_HISTORY,
    DataDomain.SECURITY_STATUS,
    DataDomain.ADJUST_FACTOR_EVENT,
)
QDP_PRODUCTION_V2_OPTIONAL_DOMAINS: tuple[str, ...] = (
    DataDomain.MARKET_INTRADAY_5M,
    DataDomain.ADJUST_FACTOR_DAILY,
    DataDomain.VALUATION,
    DataDomain.INDUSTRY_CONCEPT,
    DataDomain.INDEX_CONSTITUENTS,
    DataDomain.FINANCIAL_QUARTERLY,
    DataDomain.PERFORMANCE_FORECAST,
    DataDomain.PERFORMANCE_EXPRESS,
    DataDomain.CORPORATE_ACTIONS,
    DataDomain.SHARE_CAPITAL,
)
QDP_PRODUCTION_V2_RESEARCH_FUTURE_DOMAINS: tuple[str, ...] = (
    DataDomain.ANNOUNCEMENT,
)

BAOSTOCK_BATCH_VERSION = "0.9.3"
BAOSTOCK_BATCH_WHEEL_SHA256 = "acbd19403285bc4e254cee8297cf0e2646ae2276e5af7e549deed3988ab02293"
BAOSTOCK_BULK_PER_PAGE_COUNT = 20_000


def _quiet_baostock_call(operation: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Keep provider banner text out of machine-readable CLI stdout/stderr."""

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return operation(*args, **kwargs)


_PROVIDER_CAPABILITIES: dict[str, dict[str, Any]] = {
    "mootdx_online": {
        "domains": (
            DataDomain.MARKET_DAILY,
            DataDomain.MARKET_INTRADAY_5M,
            DataDomain.MARKET_INTRADAY_1M,
            DataDomain.INTRADAY_DAILY_FEATURES,
            DataDomain.CORPORATE_ACTIONS,
            DataDomain.SHARE_CAPITAL,
        ),
        "requires_token": False,
        "formal_eligible": True,
        "notes": "Fast online market source for recent unadjusted OHLCV plus TDX xdxr corporate-action/share-capital evidence; daily bars use endpoint-specific volume_factor=100",
    },
    "baostock": {
        "domains": (
            DataDomain.MARKET_DAILY,
            DataDomain.TRADING_CALENDAR,
            DataDomain.UNIVERSE_SNAPSHOT,
            DataDomain.SECURITY_STATUS,
            DataDomain.INDUSTRY_CONCEPT,
            DataDomain.VALUATION,
            DataDomain.INDEX_CONSTITUENTS,
            DataDomain.ADJUST_FACTOR,
            DataDomain.ADJUST_FACTOR_EVENT,
            DataDomain.ADJUST_FACTOR_DAILY,
            DataDomain.SECURITY_IDENTITY,
            DataDomain.SYMBOL_HISTORY,
            DataDomain.MARKET_INTRADAY_5M,
            DataDomain.INTRADAY_DAILY_FEATURES,
            DataDomain.FINANCIAL_QUARTERLY,
            DataDomain.PERFORMANCE_FORECAST,
            DataDomain.PERFORMANCE_EXPRESS,
        ),
        "requires_token": False,
        "formal_eligible": True,
        "notes": "BaoStock source for structured history: calendar, universe, status, industry, valuation, index constituents, adjust factors and conservative quarterly/event finance",
    },
    "cninfo": {
        "domains": (
            DataDomain.ANNOUNCEMENT,
        ),
        "requires_token": False,
        "formal_eligible": False,
        "notes": "CNInfo announcement/disclosure source; QDP production v1 keeps it out of model features until PIT audit passes",
    },
    "qdp_production_v1": {
        "domains": (
            *QDP_PRODUCTION_V1_REQUIRED_DOMAINS,
            *QDP_PRODUCTION_V1_OPTIONAL_DOMAINS,
            *QDP_PRODUCTION_V1_RESEARCH_FUTURE_DOMAINS,
        ),
        "requires_token": False,
        "formal_eligible": True,
        "notes": "Router provider: mootdx_online for recent market data, BaoStock for structured history, CNInfo for raw disclosure probes",
    },
    "qdp_production_v2": {
        "domains": (
            *QDP_PRODUCTION_V2_REQUIRED_DOMAINS,
            *QDP_PRODUCTION_V2_OPTIONAL_DOMAINS,
            *QDP_PRODUCTION_V2_RESEARCH_FUTURE_DOMAINS,
        ),
        "requires_token": False,
        "formal_eligible": True,
        "notes": "QDP v3 router: BaoStock 0.9.3 date batches for daily/status/valuation/factor events, mootdx for recent 5m, BaoStock symbol/range endpoints for repair, CNInfo for official disclosure evidence",
    },
    "eastmoney_efinance": {
        "domains": (DataDomain.MARKET_DAILY, DataDomain.UNIVERSE_SNAPSHOT, DataDomain.VALUATION),
        "requires_token": False,
        "formal_eligible": True,
        "notes": "Eastmoney/efinance wrapper for quotes, universe and valuation supplement",
    },
    "akshare_eastmoney": {
        "domains": (
            DataDomain.MARKET_DAILY,
            DataDomain.UNIVERSE_SNAPSHOT,
            DataDomain.VALUATION,
            DataDomain.INDUSTRY_CONCEPT,
            DataDomain.LIMIT_STATUS,
            DataDomain.MONEY_FLOW_HOTSPOT,
        ),
        "requires_token": False,
        "formal_eligible": True,
        "notes": "Akshare Eastmoney endpoints; useful supplement but not the only truth source",
    },
    "tencent_finance": {
        "domains": (DataDomain.VALUATION,),
        "requires_token": False,
        "formal_eligible": True,
        "notes": "Tencent quote snapshot supplement; optional in formal refresh",
    },
    "tonghuashun_hotspot": {
        "domains": (DataDomain.INDUSTRY_CONCEPT, DataDomain.MONEY_FLOW_HOTSPOT),
        "requires_token": False,
        "formal_eligible": True,
        "notes": "Tonghuashun concept/hotspot supplement through public endpoints when available",
    },
    "tushare_http_optional": {
        "domains": (DataDomain.MARKET_DAILY, DataDomain.TRADING_CALENDAR, DataDomain.UNIVERSE_SNAPSHOT, DataDomain.VALUATION, DataDomain.LIMIT_STATUS),
        "requires_token": True,
        "formal_eligible": False,
        "notes": "token-gated optional source; never required by formal_free_v3",
    },
    "sina_tencent_realtime": {
        "domains": (),
        "requires_token": False,
        "formal_eligible": False,
        "notes": "legacy placeholder for future same-day supplement",
    },
    "research_rebuild_minimal_free": {
        "domains": (
            *RESEARCH_REBUILD_MINIMAL_REQUIRED_DOMAINS,
            DataDomain.SECURITY_STATUS,
            DataDomain.INDUSTRY_CONCEPT,
            DataDomain.VALUATION,
            DataDomain.INDEX_CONSTITUENTS,
            DataDomain.ADJUST_FACTOR,
            DataDomain.INTRADAY_DAILY_FEATURES,
            DataDomain.FINANCIAL_QUARTERLY,
            DataDomain.PERFORMANCE_FORECAST,
            DataDomain.PERFORMANCE_EXPRESS,
        ),
        "requires_token": False,
        "formal_eligible": True,
        "notes": "BaoStock-first research rebuild core plan; weak web providers excluded from critical path",
    },
}


def _strip_suffix(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    if "." in raw:
        return raw.split(".", 1)[0]
    if raw.startswith(("SH", "SZ", "BJ")) and raw[2:].isdigit():
        return raw[2:]
    return raw


def _formal_requirement(domain: str) -> str:
    if domain in FORMAL_FREE_V3_REQUIRED_DOMAINS:
        return "required"
    if domain in FORMAL_FREE_V3_OPTIONAL_DOMAINS:
        return "optional"
    if domain in FORMAL_FREE_V3_RESEARCH_FUTURE_DOMAINS:
        return "research_future"
    return "unsupported"


def provider_capability_matrix(provider_plan: str = "formal_free_v3") -> list[dict[str, Any]]:
    plan = str(provider_plan or "formal_free_v3").strip().lower()
    if plan == "formal_free_v3":
        provider_names = ("baostock", "eastmoney_efinance", "akshare_eastmoney", "tencent_finance", "tonghuashun_hotspot", "tushare_http_optional")
        default_domains_by_provider: dict[str, set[str]] = {}
    elif plan == "qdp_production_v1":
        provider_names = ("mootdx_online", "baostock", "cninfo")
        default_domains_by_provider = {
            "mootdx_online": {
                DataDomain.MARKET_DAILY,
                DataDomain.MARKET_INTRADAY_1M,
                DataDomain.MARKET_INTRADAY_5M,
                DataDomain.INTRADAY_DAILY_FEATURES,
            },
            "baostock": {
                DataDomain.TRADING_CALENDAR,
                DataDomain.UNIVERSE_SNAPSHOT,
                DataDomain.SECURITY_STATUS,
                DataDomain.ADJUST_FACTOR,
                DataDomain.VALUATION,
                DataDomain.INDUSTRY_CONCEPT,
                DataDomain.INDEX_CONSTITUENTS,
                DataDomain.FINANCIAL_QUARTERLY,
                DataDomain.PERFORMANCE_FORECAST,
                DataDomain.PERFORMANCE_EXPRESS,
            },
            "cninfo": set(),
        }
    elif plan == "qdp_production_v2":
        provider_names = ("baostock", "mootdx_online", "cninfo")
        default_domains_by_provider = {
            "baostock": {
                DataDomain.MARKET_DAILY,
                DataDomain.TRADING_CALENDAR,
                DataDomain.SECURITY_STATUS,
                DataDomain.VALUATION,
                DataDomain.ADJUST_FACTOR_EVENT,
                DataDomain.ADJUST_FACTOR_DAILY,
                DataDomain.INDUSTRY_CONCEPT,
                DataDomain.INDEX_CONSTITUENTS,
                DataDomain.FINANCIAL_QUARTERLY,
                DataDomain.PERFORMANCE_FORECAST,
                DataDomain.PERFORMANCE_EXPRESS,
            },
            "mootdx_online": {DataDomain.MARKET_INTRADAY_5M, DataDomain.CORPORATE_ACTIONS, DataDomain.SHARE_CAPITAL},
            "cninfo": set(),
        }
    elif plan == "research_rebuild_minimal_free":
        provider_names = ("research_rebuild_minimal_free",)
        default_domains_by_provider = {}
    else:
        provider_names = tuple(str(getattr(provider, "name", "")) for provider in build_default_providers(plan))
        default_domains_by_provider = {}
    rows: list[dict[str, Any]] = []
    all_domains = (
        *FORMAL_FREE_V3_REQUIRED_DOMAINS,
        *FORMAL_FREE_V3_OPTIONAL_DOMAINS,
        *QDP_PRODUCTION_V1_OPTIONAL_DOMAINS,
        *QDP_PRODUCTION_V2_REQUIRED_DOMAINS,
        *QDP_PRODUCTION_V2_OPTIONAL_DOMAINS,
        *FORMAL_FREE_V3_RESEARCH_FUTURE_DOMAINS,
        *QDP_PRODUCTION_V1_RESEARCH_FUTURE_DOMAINS,
        *QDP_PRODUCTION_V2_RESEARCH_FUTURE_DOMAINS,
    )
    for provider_name in provider_names:
        meta = _PROVIDER_CAPABILITIES.get(provider_name, {"domains": (), "requires_token": False, "formal_eligible": False, "notes": ""})
        supported = set(str(item) for item in meta.get("domains", ()))
        for domain in tuple(dict.fromkeys(all_domains)):
            formal_refresh = bool(
                (
                    plan == "formal_free_v3"
                    and meta.get("formal_eligible", False)
                    and _formal_requirement(domain) in {"required", "optional"}
                )
                or (
                    plan == "research_rebuild_minimal_free"
                    and meta.get("formal_eligible", False)
                    and domain in RESEARCH_REBUILD_MINIMAL_REQUIRED_DOMAINS
                    and domain in supported
                )
                or (
                    plan in {"qdp_production_v1", "qdp_production_v2"}
                    and domain in default_domains_by_provider.get(provider_name, set())
                    and domain in supported
                )
            )
            if plan == "qdp_production_v1":
                if domain in QDP_PRODUCTION_V1_REQUIRED_DOMAINS:
                    requirement = "required"
                elif domain in QDP_PRODUCTION_V1_OPTIONAL_DOMAINS:
                    requirement = "optional"
                elif domain in QDP_PRODUCTION_V1_RESEARCH_FUTURE_DOMAINS:
                    requirement = "research_future"
                else:
                    requirement = "unsupported"
            elif plan == "qdp_production_v2":
                if domain in QDP_PRODUCTION_V2_REQUIRED_DOMAINS:
                    requirement = "required"
                elif domain in QDP_PRODUCTION_V2_OPTIONAL_DOMAINS:
                    requirement = "optional"
                elif domain in QDP_PRODUCTION_V2_RESEARCH_FUTURE_DOMAINS:
                    requirement = "research_future"
                else:
                    requirement = "unsupported"
            else:
                requirement = _formal_requirement(domain)
            rows.append(
                {
                    "provider": provider_name,
                    "domain": domain,
                    "supported": domain in supported,
                    "implemented": domain in supported,
                    "requires_token": bool(meta.get("requires_token", False)),
                    "formal_eligible": bool(meta.get("formal_eligible", False)),
                    "formal_default": formal_refresh,
                    "formal_refresh": formal_refresh,
                    "requirement": requirement,
                    "notes": str(meta.get("notes", "")),
                }
            )
    return rows


@dataclass
class EastmoneyEfinanceProvider:
    name: str = "eastmoney_efinance"

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        validate_provider_name(self.name)
        try:
            import efinance as ef  # type: ignore
        except Exception as exc:
            raise RuntimeError("efinance is not installed in the yolos environment") from exc
        request = request.normalized()
        frames: list[pd.DataFrame] = []
        for symbol in request.symbols:
            payload = ef.stock.get_quote_history(
                stock_codes=_strip_suffix(symbol),
                beg=request.start_date.replace("-", ""),
                end=request.end_date.replace("-", ""),
                klt=101,
                fqt=1 if request.adjusted_flag in {"front", "qfq"} else 0,
            )
            if isinstance(payload, dict):
                for value in payload.values():
                    if isinstance(value, pd.DataFrame):
                        frame = value.copy()
                        frame["symbol"] = symbol
                        frames.append(frame)
            elif isinstance(payload, pd.DataFrame):
                frame = payload.copy()
                frame["symbol"] = symbol
                frames.append(frame)
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        data = normalize_market_frame(raw, source=self.name, adjusted_flag=request.adjusted_flag, require_columns=False)
        return ProviderResult(provider=self.name, data=data)

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        request = request.normalized()
        if request.domain == DataDomain.MARKET_DAILY:
            return self.fetch_market_bars(
                FetchRequest(
                    symbols=request.symbols,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    adjusted_flag=request.adjusted_flag,
                )
            )
        if request.domain in {DataDomain.UNIVERSE_SNAPSHOT, DataDomain.VALUATION}:
            try:
                import efinance as ef  # type: ignore
            except Exception as exc:
                raise RuntimeError("efinance is not installed in the yolos environment") from exc
            raw = ef.stock.get_realtime_quotes()
            frame = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame()
            if not frame.empty:
                frame["trade_date"] = request.end_date
            data = normalize_domain_frame(frame, domain=request.domain, source=self.name, as_of_date=request.end_date, require_columns=False)
            return ProviderResult(provider=self.name, data=data)
        raise RuntimeError(f"unsupported_domain: {self.name} does not support {request.domain}")


@dataclass
class AkshareEastmoneyProvider:
    name: str = "akshare_eastmoney"

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        validate_provider_name(self.name)
        try:
            import akshare as ak  # type: ignore
        except Exception as exc:
            raise RuntimeError("akshare is not installed in the yolos environment") from exc
        request = request.normalized()
        frames: list[pd.DataFrame] = []
        for symbol in request.symbols:
            raw = ak.stock_zh_a_hist(
                symbol=_strip_suffix(symbol),
                period="daily",
                start_date=request.start_date.replace("-", ""),
                end_date=request.end_date.replace("-", ""),
                adjust="qfq" if request.adjusted_flag in {"front", "qfq"} else "",
            )
            if isinstance(raw, pd.DataFrame) and not raw.empty:
                raw = raw.copy()
                raw["symbol"] = symbol
                frames.append(raw)
        data = normalize_market_frame(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), source=self.name, adjusted_flag=request.adjusted_flag, require_columns=False)
        return ProviderResult(provider=self.name, data=data)

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        request = request.normalized()
        if request.domain == DataDomain.MARKET_DAILY:
            return self.fetch_market_bars(
                FetchRequest(
                    symbols=request.symbols,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    adjusted_flag=request.adjusted_flag,
                )
            )
        try:
            import akshare as ak  # type: ignore
        except Exception as exc:
            raise RuntimeError("akshare is not installed in the yolos environment") from exc
        frame = pd.DataFrame()
        if request.domain in {DataDomain.UNIVERSE_SNAPSHOT, DataDomain.VALUATION}:
            frame = ak.stock_zh_a_spot_em()
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                frame = frame.copy()
                frame["trade_date"] = request.end_date
        elif request.domain == DataDomain.INDUSTRY_CONCEPT:
            frame = _akshare_industry_members(ak, request)
        elif request.domain == DataDomain.LIMIT_STATUS:
            frame = _akshare_limit_status(ak, request)
        elif request.domain == DataDomain.MONEY_FLOW_HOTSPOT:
            frame = _akshare_money_flow(ak, request)
        else:
            raise RuntimeError(f"unsupported_domain: {self.name} does not support {request.domain}")
        data = normalize_domain_frame(frame, domain=request.domain, source=self.name, as_of_date=request.end_date, require_columns=False)
        return ProviderResult(provider=self.name, data=data)


@dataclass
class MootdxOnlineProvider:
    name: str = "mootdx_online"
    page_size: int = 800
    max_pages: int = 12
    _client_factory: Any = None

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        validate_provider_name(self.name)
        request = request.normalized()
        domain_request = DomainFetchRequest(
            domain=DataDomain.MARKET_DAILY,
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            adjusted_flag=request.adjusted_flag,
        )
        return self._fetch_bars_domain(
            domain_request,
            frequency=9,
            endpoint="bars_frequency_9_daily",
            volume_factor=100.0,
            raw_volume_unit="hands",
            canonical_volume_unit="shares",
        )

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        request = request.normalized()
        if request.domain == DataDomain.MARKET_DAILY:
            return self.fetch_market_bars(
                FetchRequest(
                    symbols=request.symbols,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    adjusted_flag=request.adjusted_flag,
                )
            )
        if request.domain == DataDomain.MARKET_INTRADAY_5M:
            return self._fetch_bars_domain(
                request,
                frequency=0,
                endpoint="bars_frequency_0_5m",
                volume_factor=1.0,
                raw_volume_unit="shares",
                canonical_volume_unit="shares",
            )
        if request.domain == DataDomain.MARKET_INTRADAY_1M:
            return self._fetch_bars_domain(
                request,
                frequency=8,
                endpoint="bars_frequency_8_1m",
                volume_factor=1.0,
                raw_volume_unit="shares",
                canonical_volume_unit="shares",
            )
        if request.domain == DataDomain.INTRADAY_DAILY_FEATURES:
            intraday = self._fetch_bars_domain(
                DomainFetchRequest(
                    domain=DataDomain.MARKET_INTRADAY_5M,
                    symbols=request.symbols,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    adjusted_flag=request.adjusted_flag,
                ),
                frequency=0,
                endpoint="bars_frequency_0_5m",
                volume_factor=1.0,
                raw_volume_unit="shares",
                canonical_volume_unit="shares",
            )
            features = build_intraday_daily_feature_frame(
                intraday.data,
                source=self.name,
                adjusted_flag=request.adjusted_flag,
            )
            coverage = coverage_report_for_domain(features, request, provider=self.name)
            coverage.update(
                {
                    "endpoint": "derived_from_bars_frequency_0_5m",
                    "source_domain": DataDomain.MARKET_INTRADAY_5M,
                    "unit_contract": "derived from canonical 5m OHLCV; no price adjustment applied",
                }
            )
            return ProviderResult(
                provider=self.name,
                data=features,
                coverage_report=coverage,
                error_report=list(intraday.error_report or []),
            )
        if request.domain in {DataDomain.CORPORATE_ACTIONS, DataDomain.SHARE_CAPITAL}:
            raw_result = self.fetch_xdxr_raw(request.symbols)
            data = _mootdx_xdxr_domain_frame(
                raw_result.data,
                domain=request.domain,
                source=self.name,
                as_of_date=request.end_date,
            )
            data = _filter_domain_date_window(data, request.start_date, request.end_date)
            coverage = coverage_report_for_domain(data, request, provider=self.name)
            coverage.update(raw_result.coverage_report)
            coverage.update(
                {
                    "domain": request.domain,
                    "normalized_row_count": int(len(data)),
                    "pit_note": "TDX xdxr is corroborating evidence; official disclosure is required to arbitrate disputed factors.",
                }
            )
            return ProviderResult(provider=self.name, data=data, coverage_report=coverage, error_report=raw_result.error_report)
        raise RuntimeError(f"unsupported_domain: {self.name} does not support {request.domain}")

    def fetch_xdxr_raw(self, symbols: Any) -> ProviderResult:
        """Fetch raw TDX ex-right and share-capital records without semantic loss."""

        normalized_symbols = tuple(
            FetchRequest(symbols=tuple(symbols or ()), start_date="2000-01-01", end_date="2000-01-01").normalized().symbols
        )
        rows: list[pd.DataFrame] = []
        errors: list[dict[str, Any]] = []
        if not normalized_symbols:
            return ProviderResult(provider=self.name, data=pd.DataFrame(), coverage_report={"endpoint": "xdxr", "row_count": 0}, error_report=[])
        client = _open_mootdx_client(self._client_factory)
        try:
            for symbol in normalized_symbols:
                last_error: Exception | None = None
                frame = pd.DataFrame()
                for attempt in range(1, 4):
                    try:
                        payload = client.xdxr(symbol=_mootdx_symbol(symbol))
                        frame = payload.copy() if isinstance(payload, pd.DataFrame) else pd.DataFrame(payload or [])
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt < 3:
                            time.sleep(float((0, 2, 5)[attempt]))
                if last_error is not None:
                    errors.append(
                        {
                            "provider": self.name,
                            "domain": "xdxr_raw",
                            "symbol": symbol,
                            "code": "symbol_fetch_error",
                            "error_type": type(last_error).__name__,
                            "message": str(last_error),
                            "attempts": 3,
                        }
                    )
                    continue
                if not frame.empty:
                    frame["provider_symbol"] = symbol
                    rows.append(frame)
        finally:
            _close_mootdx_client(client)
        data = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
        return ProviderResult(
            provider=self.name,
            data=data,
            coverage_report={
                "provider": self.name,
                "domain": "xdxr_raw",
                "endpoint": "xdxr/get_xdxr_info",
                "row_count": int(len(data)),
                "symbol_count": len(normalized_symbols),
                "successful_symbol_count": len(normalized_symbols) - len(errors),
                "raw_share_unit": "10k_shares",
            },
            error_report=errors,
        )

    def fetch_quote_snapshot(self, symbols: Iterable[str]) -> ProviderResult:
        validate_provider_name(self.name)
        normalized_symbols = tuple(FetchRequest(symbols=tuple(symbols), start_date="2000-01-01", end_date="2000-01-01").normalized().symbols)
        rows: list[pd.DataFrame] = []
        errors: list[dict[str, Any]] = []
        if not normalized_symbols:
            return ProviderResult(provider=self.name, data=pd.DataFrame(), error_report=errors)
        client = _open_mootdx_client(self._client_factory)
        try:
            payload = client.quotes(symbol=[_mootdx_symbol(symbol) for symbol in normalized_symbols])
            frame = payload.copy() if isinstance(payload, pd.DataFrame) else pd.DataFrame(payload)
            if not frame.empty:
                frame = _normalize_mootdx_quote_snapshot(
                    frame,
                    symbols=normalized_symbols,
                    source=self.name,
                    volume_factor=100.0,
                )
                rows.append(frame)
        except Exception as exc:
            errors.append({"provider": self.name, "domain": "quote_snapshot", "code": "provider_exception", "error_type": type(exc).__name__, "message": str(exc)})
        finally:
            _close_mootdx_client(client)
        data = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        return ProviderResult(
            provider=self.name,
            data=data,
            coverage_report={
                "provider": self.name,
                "domain": "quote_snapshot",
                "endpoint": "quotes",
                "row_count": int(len(data)),
                "symbol_count": int(data["symbol"].nunique()) if not data.empty and "symbol" in data.columns else 0,
                "raw_volume_unit": "hands",
                "canonical_volume_unit": "shares",
                "volume_factor": 100.0,
                "status": "ok" if len(data) else "empty",
            },
            error_report=errors,
        )

    def _fetch_bars_domain(
        self,
        request: DomainFetchRequest,
        *,
        frequency: int,
        endpoint: str,
        volume_factor: float,
        raw_volume_unit: str,
        canonical_volume_unit: str,
    ) -> ProviderResult:
        request = request.normalized()
        rows: list[pd.DataFrame] = []
        errors: list[dict[str, Any]] = []
        symbols = tuple(request.symbols)
        if not symbols:
            data = normalize_domain_frame(pd.DataFrame(), domain=request.domain, source=self.name, as_of_date=request.end_date, adjusted_flag=request.adjusted_flag, require_columns=False)
            return ProviderResult(provider=self.name, data=data, error_report=errors)
        try:
            client = _open_mootdx_client(self._client_factory)
        except Exception as exc:
            data = normalize_domain_frame(pd.DataFrame(), domain=request.domain, source=self.name, as_of_date=request.end_date, adjusted_flag=request.adjusted_flag, require_columns=False)
            return ProviderResult(
                provider=self.name,
                data=data,
                error_report=[
                    {
                        "provider": self.name,
                        "domain": request.domain,
                        "code": "client_open_error",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            )
        try:
            for idx, symbol in enumerate(symbols, start=1):
                if idx == 1 or idx % 50 == 0 or idx == len(symbols):
                    progress_write(f"mootdx_online_{request.domain}={idx}/{len(symbols)} symbol={symbol}")
                last_exc: Exception | None = None
                frame = pd.DataFrame()
                for attempt in range(1, 4):
                    try:
                        frame = _fetch_mootdx_bars_window(
                            client=client,
                            symbol=symbol,
                            frequency=frequency,
                            start_date=request.start_date,
                            end_date=request.end_date,
                            page_size=int(self.page_size),
                            max_pages=int(self.max_pages),
                            source=self.name,
                            adjusted_flag=request.adjusted_flag,
                            volume_factor=float(volume_factor),
                        )
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                        _close_mootdx_client(client)
                        time.sleep(min(3.0, 0.5 * attempt))
                        try:
                            client = _open_mootdx_client(self._client_factory)
                        except Exception as open_exc:
                            last_exc = open_exc
                            time.sleep(min(3.0, 0.5 * attempt))
                try:
                    if last_exc is not None:
                        raise last_exc
                except Exception as exc:
                    errors.append({"provider": self.name, "domain": request.domain, "symbol": symbol, "code": "symbol_fetch_error", "error_type": type(exc).__name__, "message": str(exc), "attempts": 3})
                    continue
                if not frame.empty:
                    rows.append(frame)
        finally:
            _close_mootdx_client(client)
        raw = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        data = normalize_domain_frame(
            raw,
            domain=request.domain,
            source=self.name,
            as_of_date=request.end_date,
            adjusted_flag=request.adjusted_flag,
            require_columns=False,
        )
        data = _filter_domain_date_window(data, request.start_date, request.end_date)
        if request.domain == DataDomain.MARKET_DAILY:
            coverage = coverage_report_for_frame(
                data,
                FetchRequest(
                    symbols=request.symbols,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    adjusted_flag=request.adjusted_flag,
                ),
                provider=self.name,
            )
        else:
            coverage = coverage_report_for_domain(data, request, provider=self.name)
        coverage.update(
            {
                "endpoint": endpoint,
                "frequency": int(frequency),
                "raw_volume_unit": raw_volume_unit,
                "canonical_volume_unit": canonical_volume_unit,
                "volume_factor": float(volume_factor),
                "adjustment_semantics": "unadjusted_raw_ohlcv",
                "source_stability_note": "online mootdx quote server; server stability must be monitored by provider-health/provider-eval",
            }
        )
        if request.domain == DataDomain.MARKET_INTRADAY_1M:
            coverage["bar_count_contract"] = "mootdx_1m_240_without_0930"
        elif request.domain == DataDomain.MARKET_INTRADAY_5M:
            coverage["bar_count_contract"] = "mootdx_5m_48_full_trading_day"
        return ProviderResult(provider=self.name, data=data, coverage_report=coverage, error_report=errors)


class _BaostockGlobalLimiter:
    """Process-local limiter shared by every BaoStock endpoint.

    BaoStock sessions are stateful and its public service is sensitive to
    bursts. QDP therefore defaults to one in-flight request and never raises
    that ceiling automatically. A live two-login probe showed that BaoStock
    can invalidate one session when another logs in, even at a zero historical
    error rate.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active = 0
        self._limit = 1
        self._requests = 0
        self._network_errors = 0

    @contextmanager
    def slot(self):
        with self._condition:
            while self._active >= self._limit:
                self._condition.wait()
            self._active += 1
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            with self._condition:
                self._active -= 1
                self._requests += 1
                if failed:
                    self._network_errors += 1
                self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "active": self._active,
                "limit": self._limit,
                "requests": self._requests,
                "network_errors": self._network_errors,
                "network_error_rate": self._network_errors / max(self._requests, 1),
            }

    def configure_limit(self, limit: int) -> dict[str, Any]:
        normalized = int(limit)
        if normalized not in {1, 2}:
            raise ValueError("baostock_global_concurrency_must_be_1_or_2")
        with self._condition:
            self._limit = normalized
            self._condition.notify_all()
        return self.snapshot()


_BAOSTOCK_GLOBAL_LIMITER = _BaostockGlobalLimiter()


def configure_baostock_global_concurrency(max_workers: int) -> dict[str, Any]:
    """Set the process-wide BaoStock request ceiling to one or two slots.

    Production date-partition ingestion always configures one slot. The
    two-slot setting remains an explicit low-level test hook; it is never
    selected from historical error-rate statistics.
    """

    return _BAOSTOCK_GLOBAL_LIMITER.configure_limit(int(max_workers))


def baostock_global_limiter_snapshot() -> dict[str, Any]:
    return _BAOSTOCK_GLOBAL_LIMITER.snapshot()


class _BaostockPersistentSession:
    """One isolated BaoStock login reused across sequential QDP requests."""

    def __init__(self) -> None:
        self._context = multiprocessing.get_context("spawn")
        self._command_queue: Any | None = None
        self._response_queue: Any | None = None
        self._process: Any | None = None
        self._lock = threading.Lock()
        self._sequence = 0

    def _start_locked(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._terminate_locked()
        self._command_queue = self._context.Queue()
        self._response_queue = self._context.Queue()
        self._process = self._context.Process(
            target=_baostock_persistent_session_worker,
            kwargs={"command_queue": self._command_queue, "response_queue": self._response_queue},
        )
        self._process.daemon = True
        self._process.start()

    def _terminate_locked(self) -> None:
        process = self._process
        self._process = None
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join(5)
        for channel in (self._command_queue, self._response_queue):
            if channel is not None:
                try:
                    channel.close()
                except Exception:
                    pass
        self._command_queue = None
        self._response_queue = None

    def request(self, payload: dict[str, Any], *, timeout_seconds: int) -> dict[str, Any]:
        timeout = max(float(timeout_seconds or 0), 1.0)
        with self._lock:
            self._start_locked()
            self._sequence += 1
            request_id = self._sequence
            command = {**payload, "request_id": request_id}
            try:
                with _BAOSTOCK_GLOBAL_LIMITER.slot():
                    self._command_queue.put(command)
                    response = self._response_queue.get(timeout=timeout)
                    if not isinstance(response, dict) or response.get("request_id") != request_id:
                        raise RuntimeError("baostock_persistent_session_response_mismatch")
                    if response.get("status") != "ok":
                        raise RuntimeError(
                            "baostock_persistent_session_error:"
                            f"{response.get('error_type', 'RuntimeError')}: {response.get('error', response)}"
                        )
                return response
            except queue_module.Empty as exc:
                self._terminate_locked()
                raise TimeoutError(f"baostock_persistent_session_timeout:{int(timeout)}") from exc
            except Exception:
                self._terminate_locked()
                raise

    def close(self) -> None:
        with self._lock:
            process = self._process
            if process is not None and process.is_alive() and self._command_queue is not None:
                try:
                    self._command_queue.put({"kind": "close", "request_id": -1})
                    process.join(5)
                except Exception:
                    pass
            self._terminate_locked()


def baostock_runtime_version() -> str:
    try:
        return str(package_version("baostock"))
    except PackageNotFoundError as exc:
        raise RuntimeError("baostock is not installed in the yolos environment") from exc


def baostock_runtime_archive_sha256() -> str:
    """Return pip's recorded source-archive hash when one is available."""

    try:
        direct_url = package_distribution("baostock").read_text("direct_url.json")
    except PackageNotFoundError as exc:
        raise RuntimeError("baostock is not installed in the yolos environment") from exc
    if not direct_url:
        return ""
    try:
        payload = json.loads(direct_url)
    except json.JSONDecodeError as exc:
        raise RuntimeError("baostock_direct_url_metadata_invalid") from exc
    archive = dict(payload.get("archive_info", {}) or {})
    hashes = dict(archive.get("hashes", {}) or {})
    value = str(hashes.get("sha256", "") or "")
    if not value and str(archive.get("hash", "")).startswith("sha256="):
        value = str(archive["hash"]).split("=", 1)[1]
    return value.strip().lower()


def assert_baostock_batch_runtime() -> str:
    installed = baostock_runtime_version()
    if installed != BAOSTOCK_BATCH_VERSION:
        raise RuntimeError(
            "baostock_batch_version_mismatch: "
            f"expected={BAOSTOCK_BATCH_VERSION} installed={installed}; "
            "batch data is forbidden from canonical staging"
        )
    archive_sha = baostock_runtime_archive_sha256()
    if archive_sha and archive_sha != BAOSTOCK_BATCH_WHEEL_SHA256:
        raise RuntimeError(
            "baostock_batch_wheel_hash_mismatch: "
            f"expected={BAOSTOCK_BATCH_WHEEL_SHA256} installed_archive={archive_sha}; "
            "batch data is forbidden from canonical staging"
        )
    return installed


@dataclass
class BaostockProvider:
    name: str = "baostock"
    _market_daily_max_workers: int = 4
    _intraday_max_workers: int = 1
    _bulk_timeout_seconds: int = 120
    _bulk_retry_backoff_seconds: tuple[int, ...] = (2, 5, 15)
    _reuse_date_partition_session: bool = True
    _date_partition_session: Any = field(default=None, init=False, repr=False)
    _date_partition_session_guard: Any = field(default_factory=threading.Lock, init=False, repr=False)

    def _persistent_date_request(self, payload: dict[str, Any], *, timeout_seconds: int) -> tuple[dict[str, Any], int, list[str]]:
        errors: list[str] = []
        attempts = len(tuple(self._bulk_retry_backoff_seconds)) + 1
        for attempt in range(1, attempts + 1):
            try:
                with self._date_partition_session_guard:
                    if self._date_partition_session is None:
                        self._date_partition_session = _BaostockPersistentSession()
                    session = self._date_partition_session
                return session.request(payload, timeout_seconds=timeout_seconds), attempt, errors
            except Exception as exc:
                errors.append(f"attempt={attempt}:{type(exc).__name__}:{exc}")
                self.close_date_partition_session()
                if attempt >= attempts:
                    break
                time.sleep(float(tuple(self._bulk_retry_backoff_seconds)[attempt - 1]))
        raise RuntimeError(
            "baostock_persistent_request_failed:"
            f"kind={payload.get('kind', '')} trade_date={payload.get('trade_date', '')} "
            f"attempts={attempts} errors={' | '.join(errors)}"
        )

    def close_date_partition_session(self) -> None:
        with self._date_partition_session_guard:
            session = self._date_partition_session
            self._date_partition_session = None
        if session is not None:
            session.close()

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        validate_provider_name(self.name)
        request = request.normalized()
        rows: list[pd.DataFrame] = []
        errors: list[dict[str, Any]] = []
        failed_symbols: list[tuple[str, BaseException]] = []
        symbols = tuple(request.symbols)
        max_workers = min(self._market_daily_max_workers, len(symbols)) if symbols else 0
        if not symbols:
            data = normalize_market_frame(pd.DataFrame(), source=self.name, adjusted_flag=request.adjusted_flag, require_columns=False)
            return ProviderResult(provider=self.name, data=data, error_report=errors)
        with create_progress(total=len(symbols), desc="Baostock market_daily", unit="symbol", leave=False) as progress:
            if len(symbols) == 1 or max_workers <= 1:
                for idx, symbol in enumerate(symbols, start=1):
                    if idx == 1 or idx % 50 == 0 or idx == len(symbols):
                        progress_write(f"baostock_market_daily={idx}/{len(symbols)} symbol={symbol}")
                    progress.set_description_str(f"Baostock market_daily {symbol}")
                    progress.update(1)
                    try:
                        frame = _fetch_baostock_history_frame_with_timeout(
                            symbol=symbol,
                            start_date=request.start_date,
                            end_date=request.end_date,
                            adjusted_flag=request.adjusted_flag,
                        )
                    except Exception as exc:
                        failed_symbols.append((symbol, exc))
                        continue
                    if not frame.empty:
                        rows.append(frame)
            else:
                futures: dict[Any, str] = {}
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for idx, symbol in enumerate(symbols, start=1):
                        if idx == 1 or idx % 50 == 0 or idx == len(symbols):
                            progress_write(f"baostock_market_daily={idx}/{len(symbols)} symbol={symbol}")
                        futures[
                            executor.submit(
                                _fetch_baostock_history_frame_with_timeout,
                                symbol=symbol,
                                start_date=request.start_date,
                                end_date=request.end_date,
                                adjusted_flag=request.adjusted_flag,
                            )
                        ] = symbol
                    for future in as_completed(futures):
                        symbol = futures[future]
                        progress.set_description_str(f"Baostock market_daily {symbol}")
                        progress.update(1)
                        try:
                            frame = future.result()
                        except Exception as exc:
                            failed_symbols.append((symbol, exc))
                            continue
                        if not frame.empty:
                            rows.append(frame)
        if failed_symbols:
            progress_write(f"baostock_market_daily_retry={len(failed_symbols)}")
            for retry_idx, (symbol, first_exc) in enumerate(failed_symbols, start=1):
                progress_write(f"baostock_market_daily_retry={retry_idx}/{len(failed_symbols)} symbol={symbol}")
                try:
                    frame = _fetch_baostock_history_frame_with_timeout(
                        symbol=symbol,
                        start_date=request.start_date,
                        end_date=request.end_date,
                        adjusted_flag=request.adjusted_flag,
                    )
                except Exception as exc:
                    errors.append(_baostock_symbol_error(self.name, symbol, exc, first_error=first_exc))
                    continue
                if not frame.empty:
                    rows.append(frame)
        data = normalize_market_frame(pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), source=self.name, adjusted_flag=request.adjusted_flag, require_columns=False)
        return ProviderResult(provider=self.name, data=data, error_report=errors)

    def _fetch_intraday_5m_frames(self, request: DomainFetchRequest, *, progress_label: str) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
        rows: list[pd.DataFrame] = []
        errors: list[dict[str, Any]] = []
        failed_symbols: list[tuple[str, BaseException]] = []
        symbols = tuple(request.symbols)
        max_workers = min(max(1, int(self._intraday_max_workers or 1)), len(symbols)) if symbols else 0
        if not symbols:
            return rows, errors
        if len(symbols) == 1 or max_workers <= 1:
            for idx, symbol in enumerate(symbols, start=1):
                if idx == 1 or idx % 50 == 0 or idx == len(symbols):
                    progress_write(f"{progress_label}={idx}/{len(symbols)} symbol={symbol}")
                try:
                    frame = _fetch_baostock_intraday_5m_frame_with_timeout(
                        symbol=symbol,
                        start_date=request.start_date,
                        end_date=request.end_date,
                        adjusted_flag=request.adjusted_flag,
                    )
                except Exception as exc:
                    failed_symbols.append((symbol, exc))
                    continue
                if not frame.empty:
                    rows.append(frame)
        else:
            futures: dict[Any, str] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for idx, symbol in enumerate(symbols, start=1):
                    if idx == 1 or idx % 50 == 0 or idx == len(symbols):
                        progress_write(f"{progress_label}={idx}/{len(symbols)} symbol={symbol}")
                    futures[
                        executor.submit(
                            _fetch_baostock_intraday_5m_frame_with_timeout,
                            symbol=symbol,
                            start_date=request.start_date,
                            end_date=request.end_date,
                            adjusted_flag=request.adjusted_flag,
                        )
                    ] = symbol
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        frame = future.result()
                    except Exception as exc:
                        failed_symbols.append((symbol, exc))
                        continue
                    if not frame.empty:
                        rows.append(frame)
        if failed_symbols:
            progress_write(f"{progress_label}_retry={len(failed_symbols)}")
            for retry_idx, (symbol, first_exc) in enumerate(failed_symbols, start=1):
                progress_write(f"{progress_label}_retry={retry_idx}/{len(failed_symbols)} symbol={symbol}")
                try:
                    frame = _fetch_baostock_intraday_5m_frame_with_timeout(
                        symbol=symbol,
                        start_date=request.start_date,
                        end_date=request.end_date,
                        adjusted_flag=request.adjusted_flag,
                    )
                except Exception as exc:
                    errors.append(_baostock_symbol_error(self.name, symbol, exc, first_error=first_exc))
                    continue
                if not frame.empty:
                    rows.append(frame)
        return rows, errors

    def fetch_date_partition(self, request: DatePartitionFetchRequest) -> DatePartitionProviderResult:
        """Fetch one BaoStock 0.9.3 bulk response without ordinary paging."""

        request = request.normalized()
        assert_baostock_batch_runtime()
        if request.fetch_mode == "date_events":
            endpoint = "query_daily_adjust_factor"
        elif request.universe_kind == "etf":
            endpoint = "query_daily_history_k_ETF"
        else:
            endpoint = "query_daily_history_k_AStock"
        started = time.perf_counter()
        if self._reuse_date_partition_session:
            payload, attempt_count, retry_errors = self._persistent_date_request(
                {"kind": "bulk", "endpoint": endpoint, "trade_date": request.trade_date},
                timeout_seconds=int(self._bulk_timeout_seconds),
            )
            raw = payload.get("data")
            if not isinstance(raw, pd.DataFrame):
                raise RuntimeError(f"baostock_persistent_bulk_invalid_data:{type(raw).__name__}")
            response_meta = dict(payload.get("meta", {}) or {})
            response_meta.update({"attempt_count": attempt_count, "retry_errors": retry_errors, "session_reused": True})
        else:
            raw, response_meta = _fetch_baostock_bulk_partition_with_retry(
                endpoint=endpoint,
                trade_date=request.trade_date,
                timeout_seconds=int(self._bulk_timeout_seconds),
                backoff_seconds=tuple(self._bulk_retry_backoff_seconds),
            )
        if request.fetch_mode == "date_events":
            data = _baostock_bulk_adjust_factor_event_frame(raw, query_date=request.trade_date)
        else:
            data = _baostock_bulk_daily_domain_frame(raw, domain=request.domain, query_date=request.trade_date)
        elapsed = time.perf_counter() - started
        coverage = {
            **response_meta,
            "provider": self.name,
            "endpoint": endpoint,
            "package_version": BAOSTOCK_BATCH_VERSION,
            "wheel_sha256": BAOSTOCK_BATCH_WHEEL_SHA256,
            "query_date": request.trade_date,
            "universe_kind": request.universe_kind,
            "fetch_mode": request.fetch_mode,
            "row_count": int(len(raw)),
            "elapsed_seconds": round(float(elapsed), 6),
            "limiter": _BAOSTOCK_GLOBAL_LIMITER.snapshot(),
        }
        return DatePartitionProviderResult(
            provider=self.name,
            request=request,
            raw_data=raw,
            data=data,
            coverage_report=coverage,
            error_report=[],
        )

    def fetch_date_partition_with_all_stock(
        self, request: DatePartitionFetchRequest
    ) -> tuple[DatePartitionProviderResult, pd.DataFrame]:
        """Fetch one A-share daily partition and its audit universe atomically."""

        request = request.normalized()
        if request.fetch_mode != "date_snapshot" or request.universe_kind != "all_a":
            raise ValueError("combined_all_stock_fetch_requires_all_a_date_snapshot")
        assert_baostock_batch_runtime()
        endpoint = "query_daily_history_k_AStock"
        started = time.perf_counter()
        payload, attempt_count, retry_errors = self._persistent_date_request(
            {"kind": "bulk_with_all_stock", "endpoint": endpoint, "trade_date": request.trade_date},
            timeout_seconds=int(self._bulk_timeout_seconds),
        )
        raw = payload.get("data")
        universe_frame = payload.get("audit_data")
        if not isinstance(raw, pd.DataFrame) or not isinstance(universe_frame, pd.DataFrame):
            raise RuntimeError(
                "baostock_persistent_combined_invalid_data:"
                f"daily={type(raw).__name__}:all_stock={type(universe_frame).__name__}"
            )
        response_meta = dict(payload.get("meta", {}) or {})
        response_meta.update(
            {
                "attempt_count": attempt_count,
                "retry_errors": retry_errors,
                "session_reused": True,
                "combined_all_stock_audit": True,
            }
        )
        data = _baostock_bulk_daily_domain_frame(raw, domain=request.domain, query_date=request.trade_date)
        elapsed = time.perf_counter() - started
        coverage = {
            **response_meta,
            "provider": self.name,
            "endpoint": endpoint,
            "package_version": BAOSTOCK_BATCH_VERSION,
            "wheel_sha256": BAOSTOCK_BATCH_WHEEL_SHA256,
            "query_date": request.trade_date,
            "universe_kind": request.universe_kind,
            "fetch_mode": request.fetch_mode,
            "row_count": int(len(raw)),
            "elapsed_seconds": round(float(elapsed), 6),
            "limiter": _BAOSTOCK_GLOBAL_LIMITER.snapshot(),
        }
        return (
            DatePartitionProviderResult(
                provider=self.name,
                request=request,
                raw_data=raw,
                data=data,
                coverage_report=coverage,
                error_report=[],
            ),
            universe_frame.copy(),
        )

    def fetch_all_stock_audit_evidence(self, *, trade_date: str) -> pd.DataFrame:
        """Return one lossless ``query_all_stock`` A-share audit snapshot.

        This is separate from ``fetch_domain`` because the standard universe
        contract omits BaoStock's ``tradeStatus`` field.  QDP v3 needs that
        field to classify a missing bulk-daily row as suspended rather than as
        an unexplained provider gap.
        """

        validate_provider_name(self.name)
        if self._reuse_date_partition_session:
            payload, _, _ = self._persistent_date_request(
                {"kind": "all_stock", "domain": DataDomain.UNIVERSE_SNAPSHOT, "trade_date": str(trade_date)},
                timeout_seconds=120,
            )
            frame = payload.get("data")
            if not isinstance(frame, pd.DataFrame):
                raise RuntimeError(f"baostock_persistent_all_stock_invalid_data:{type(frame).__name__}")
            return frame.copy()
        return _fetch_baostock_all_stock_frame_with_timeout(
            domain=DataDomain.UNIVERSE_SNAPSHOT,
            trade_date=str(trade_date),
            timeout_seconds=120,
        )

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        request = request.normalized()
        if request.domain == DataDomain.MARKET_DAILY:
            return self.fetch_market_bars(
                FetchRequest(
                    symbols=request.symbols,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    adjusted_flag=request.adjusted_flag,
                )
            )
        if request.domain == DataDomain.MARKET_INTRADAY_5M:
            rows, errors = self._fetch_intraday_5m_frames(request, progress_label="baostock_intraday_5m")
            frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            data = normalize_domain_frame(frame, domain=request.domain, source=self.name, as_of_date=request.end_date, adjusted_flag=request.adjusted_flag, require_columns=False)
            return ProviderResult(provider=self.name, data=data, error_report=errors)
        if request.domain == DataDomain.INTRADAY_DAILY_FEATURES:
            raw_rows, errors = self._fetch_intraday_5m_frames(request, progress_label="baostock_intraday_daily_features")
            rows = [
                build_intraday_daily_feature_frame(
                    raw_5m,
                    source=self.name,
                    adjusted_flag=request.adjusted_flag,
                )
                for raw_5m in raw_rows
                if not raw_5m.empty
            ]
            frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            data = normalize_domain_frame(frame, domain=request.domain, source=self.name, as_of_date=request.end_date, adjusted_flag=request.adjusted_flag, require_columns=False)
            return ProviderResult(provider=self.name, data=data, error_report=errors)
        if request.domain == DataDomain.TRADING_CALENDAR:
            frame = _fetch_baostock_trade_calendar_frame_with_timeout(
                start_date=request.start_date,
                end_date=request.end_date,
                exchange=request.exchange,
            )
        elif request.domain in {DataDomain.UNIVERSE_SNAPSHOT, DataDomain.SECURITY_STATUS}:
            frame = _fetch_baostock_all_stock_frame_with_timeout(domain=request.domain, trade_date=request.end_date)
        elif request.domain == DataDomain.INDUSTRY_CONCEPT:
            frame = _fetch_baostock_industry_frame_with_timeout(trade_date=request.end_date)
        elif request.domain == DataDomain.INDEX_CONSTITUENTS:
            frame = _fetch_baostock_index_constituents_frame_with_timeout(trade_date=request.end_date)
        elif request.domain == DataDomain.FINANCIAL_QUARTERLY:
            frame = _fetch_baostock_financial_quarterly_frame_with_timeout(
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
            )
        elif request.domain == DataDomain.PERFORMANCE_FORECAST:
            frame = _fetch_baostock_performance_frame_with_timeout(
                domain=request.domain,
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
            )
        elif request.domain == DataDomain.PERFORMANCE_EXPRESS:
            frame = _fetch_baostock_performance_frame_with_timeout(
                domain=request.domain,
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
            )
        elif request.domain == DataDomain.VALUATION:
            frame = _fetch_baostock_valuation_frame_with_timeout(
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
            )
        elif request.domain == DataDomain.ADJUST_FACTOR:
            frame = _fetch_baostock_adjust_factor_frame_with_timeout(
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
            )
        else:
            raise RuntimeError(f"unsupported_domain: {self.name} does not support {request.domain}")
        data = normalize_domain_frame(frame, domain=request.domain, source=self.name, as_of_date=request.end_date, require_columns=False)
        return ProviderResult(provider=self.name, data=data)


@dataclass
class TushareHttpOptionalProvider:
    name: str = "tushare_http_optional"
    token: str = ""

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        validate_provider_name(self.name)
        token = str(self.token or os.environ.get("TUSHARE_TOKEN", "")).strip()
        if not token:
            raise RuntimeError("TUSHARE_TOKEN is not configured; tushare_http_optional is disabled")
        request = request.normalized()
        frames: list[pd.DataFrame] = []
        for symbol in request.symbols:
            response = requests.post(
                "http://api.tushare.pro",
                json={
                    "api_name": "daily",
                    "token": token,
                    "params": {
                        "ts_code": symbol,
                        "start_date": request.start_date.replace("-", ""),
                        "end_date": request.end_date.replace("-", ""),
                    },
                    "fields": "ts_code,trade_date,open,high,low,close,vol,amount",
                },
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") != 0:
                raise RuntimeError(payload.get("msg") or f"tushare error code {payload.get('code')}")
            data = payload.get("data") or {}
            frame = pd.DataFrame(data.get("items") or [], columns=data.get("fields") or [])
            if not frame.empty:
                frames.append(frame.rename(columns={"ts_code": "symbol", "vol": "volume"}))
        data = normalize_market_frame(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), source=self.name, adjusted_flag=request.adjusted_flag, require_columns=False)
        return ProviderResult(provider=self.name, data=data)

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        request = request.normalized()
        if request.domain == DataDomain.MARKET_DAILY:
            return self.fetch_market_bars(
                FetchRequest(
                    symbols=request.symbols,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    adjusted_flag=request.adjusted_flag,
                )
            )
        token = str(self.token or os.environ.get("TUSHARE_TOKEN", "")).strip()
        if not token:
            raise RuntimeError("TUSHARE_TOKEN is not configured; tushare_http_optional is disabled")
        if request.domain == DataDomain.TRADING_CALENDAR:
            frame = self._post_tushare(
                token=token,
                api_name="trade_cal",
                params={"start_date": request.start_date.replace("-", ""), "end_date": request.end_date.replace("-", "")},
                fields="cal_date,is_open,exchange",
            ).rename(columns={"cal_date": "trade_date"})
        elif request.domain == DataDomain.UNIVERSE_SNAPSHOT:
            frame = self._post_tushare(
                token=token,
                api_name="stock_basic",
                params={"list_status": "L"},
                fields="ts_code,name,market,list_status,list_date,delist_date",
            ).rename(columns={"ts_code": "symbol", "market": "board"})
            frame["trade_date"] = request.end_date
        elif request.domain == DataDomain.VALUATION:
            frames = []
            for trade_date in _date_range_strings(request.start_date, request.end_date):
                frames.append(
                    self._post_tushare(
                        token=token,
                        api_name="daily_basic",
                        params={"trade_date": trade_date.replace("-", "")},
                        fields="ts_code,trade_date,total_mv,circ_mv,pe,pb,turnover_rate",
                    ).rename(columns={"ts_code": "symbol"})
                )
            frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        elif request.domain == DataDomain.LIMIT_STATUS:
            frames = []
            for trade_date in _date_range_strings(request.start_date, request.end_date):
                frames.append(
                    self._post_tushare(
                        token=token,
                        api_name="stk_limit",
                        params={"trade_date": trade_date.replace("-", "")},
                        fields="ts_code,trade_date,up_limit,down_limit",
                    ).rename(columns={"ts_code": "symbol"})
                )
            frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        else:
            raise RuntimeError(f"unsupported_domain: {self.name} does not support {request.domain}")
        data = normalize_domain_frame(frame, domain=request.domain, source=self.name, as_of_date=request.end_date, require_columns=False)
        return ProviderResult(provider=self.name, data=data)

    def _post_tushare(self, *, token: str, api_name: str, params: dict[str, Any], fields: str) -> pd.DataFrame:
        response = requests.post(
            "http://api.tushare.pro",
            json={"api_name": api_name, "token": token, "params": params, "fields": fields},
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError(payload.get("msg") or f"tushare error code {payload.get('code')}")
        data = payload.get("data") or {}
        return pd.DataFrame(data.get("items") or [], columns=data.get("fields") or [])


@dataclass
class SinaTencentRealtimeProvider:
    name: str = "sina_tencent_realtime"

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        validate_provider_name(self.name)
        raise RuntimeError("sina_tencent_realtime only supports realtime snapshots; daily refresh uses it as an optional same-day supplement in a later phase")

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        raise RuntimeError("sina_tencent_realtime only supports realtime supplement domains in a later phase")


@dataclass
class TencentFinanceProvider:
    name: str = "tencent_finance"

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        raise RuntimeError("tencent_finance is an optional valuation/realtime supplement; it does not provide formal daily bars")

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        request = request.normalized()
        if request.domain != DataDomain.VALUATION:
            raise RuntimeError(f"unsupported_domain: {self.name} does not support {request.domain}")
        symbols = tuple(request.symbols or ())
        if not symbols:
            return ProviderResult(provider=self.name, data=pd.DataFrame())
        query = ",".join(_to_tencent_simple_code(symbol) for symbol in symbols)
        response = requests.get(f"https://qt.gtimg.cn/q={query}")
        response.encoding = response.encoding or "gbk"
        rows: list[dict[str, Any]] = []
        for line in str(response.text or "").splitlines():
            parts = line.split("~")
            if len(parts) < 4:
                continue
            raw_code = parts[0].split("=", 1)[0].replace("v_s_", "").replace("v_", "").strip()
            symbol = _from_tencent_code(raw_code)
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": request.end_date,
                    "total_mv": float("nan"),
                    "circ_mv": float("nan"),
                    "pe": float("nan"),
                    "pb": float("nan"),
                    "turnover_rate": float("nan"),
                    "source": self.name,
                }
            )
        data = normalize_domain_frame(pd.DataFrame(rows), domain=request.domain, source=self.name, as_of_date=request.end_date, require_columns=False)
        return ProviderResult(provider=self.name, data=data)


@dataclass
class TonghuashunHotspotProvider:
    name: str = "tonghuashun_hotspot"

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        raise RuntimeError("tonghuashun_hotspot only supports optional concept/hotspot domains")

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        request = request.normalized()
        try:
            import akshare as ak  # type: ignore
        except Exception as exc:
            raise RuntimeError("akshare is required for tonghuashun_hotspot optional endpoints") from exc
        if request.domain == DataDomain.INDUSTRY_CONCEPT:
            frame = _ths_concept_frame(ak, request)
        elif request.domain == DataDomain.MONEY_FLOW_HOTSPOT:
            frame = _ths_hotspot_frame(ak, request)
        else:
            raise RuntimeError(f"unsupported_domain: {self.name} does not support {request.domain}")
        data = normalize_domain_frame(frame, domain=request.domain, source=self.name, as_of_date=request.end_date, require_columns=False)
        return ProviderResult(provider=self.name, data=data)


@dataclass
class ResearchRebuildMinimalFreeProvider:
    name: str = "research_rebuild_minimal_free"

    def __post_init__(self) -> None:
        self._baostock = BaostockProvider()
        self._eastmoney = EastmoneyEfinanceProvider()

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        return self._baostock.fetch_market_bars(request)

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        request = request.normalized()
        if request.domain in {
            DataDomain.MARKET_DAILY,
            DataDomain.TRADING_CALENDAR,
            DataDomain.UNIVERSE_SNAPSHOT,
            DataDomain.SECURITY_STATUS,
            DataDomain.INDUSTRY_CONCEPT,
            DataDomain.INDEX_CONSTITUENTS,
            DataDomain.INTRADAY_DAILY_FEATURES,
            DataDomain.FINANCIAL_QUARTERLY,
            DataDomain.PERFORMANCE_FORECAST,
            DataDomain.PERFORMANCE_EXPRESS,
        }:
            return self._baostock.fetch_domain(request)
        if request.domain == DataDomain.VALUATION:
            result = self._baostock.fetch_domain(request)
            if result.data is not None and not result.data.empty:
                return result
            return self._eastmoney.fetch_domain(request)
        raise RuntimeError(f"unsupported_domain: {self.name} does not support {request.domain}")


@dataclass
class CninfoAnnouncementProvider:
    name: str = "cninfo"
    page_size: int = 30
    max_pages: int = 3

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        raise RuntimeError("cninfo only supports announcement/disclosure domains")

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        request = request.normalized()
        if request.domain != DataDomain.ANNOUNCEMENT:
            raise RuntimeError(f"unsupported_domain: {self.name} does not support {request.domain}")
        frames: list[pd.DataFrame] = []
        errors: list[dict[str, Any]] = []
        for symbol in tuple(request.symbols or ()):
            try:
                frame = _fetch_cninfo_announcements(
                    symbol=symbol,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    page_size=int(self.page_size),
                    max_pages=int(self.max_pages),
                )
            except Exception as exc:
                errors.append({"provider": self.name, "domain": request.domain, "symbol": symbol, "code": "symbol_fetch_error", "error_type": type(exc).__name__, "message": str(exc)})
                continue
            if not frame.empty:
                frames.append(frame)
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        data = normalize_domain_frame(raw, domain=request.domain, source=self.name, as_of_date=request.end_date, require_columns=False)
        coverage = coverage_report_for_domain(data, request, provider=self.name)
        coverage.update(
            {
                "endpoint": "hisAnnouncement/query",
                "raw_only_until_pit_audit": True,
                "pit_gate": "blocked_for_model_features_until_disclosure_time_audit",
            }
        )
        return ProviderResult(provider=self.name, data=data, coverage_report=coverage, error_report=errors)


@dataclass
class QdpProductionV1Provider:
    name: str = "qdp_production_v1"

    def __post_init__(self) -> None:
        self._mootdx = MootdxOnlineProvider()
        self._baostock = BaostockProvider()
        self._cninfo = CninfoAnnouncementProvider()

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        return self._mootdx.fetch_market_bars(request)

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        request = request.normalized()
        if request.domain in {
            DataDomain.MARKET_DAILY,
            DataDomain.MARKET_INTRADAY_5M,
            DataDomain.MARKET_INTRADAY_1M,
            DataDomain.INTRADAY_DAILY_FEATURES,
        }:
            return self._mootdx.fetch_domain(request)
        if request.domain in {
            DataDomain.TRADING_CALENDAR,
            DataDomain.UNIVERSE_SNAPSHOT,
            DataDomain.SECURITY_STATUS,
            DataDomain.INDUSTRY_CONCEPT,
            DataDomain.VALUATION,
            DataDomain.INDEX_CONSTITUENTS,
            DataDomain.ADJUST_FACTOR,
            DataDomain.FINANCIAL_QUARTERLY,
            DataDomain.PERFORMANCE_FORECAST,
            DataDomain.PERFORMANCE_EXPRESS,
        }:
            return self._baostock.fetch_domain(request)
        if request.domain == DataDomain.ANNOUNCEMENT:
            return self._cninfo.fetch_domain(request)
        raise RuntimeError(f"unsupported_domain: {self.name} does not support {request.domain}")


@dataclass
class QdpProductionV2Provider:
    """Provider routing contract for the QDP v3 data base."""

    name: str = "qdp_production_v2"

    def __post_init__(self) -> None:
        self._mootdx = MootdxOnlineProvider()
        self._baostock = BaostockProvider()
        self._cninfo = CninfoAnnouncementProvider()

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        # Symbol/range daily queries are repair and compatibility endpoints in
        # v3; the all-market primary path is fetch_date_partition().
        return self._baostock.fetch_market_bars(request)

    def fetch_date_partition(self, request: DatePartitionFetchRequest) -> DatePartitionProviderResult:
        return self._baostock.fetch_date_partition(request)

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        request = request.normalized()
        if request.domain in {DataDomain.MARKET_INTRADAY_5M, DataDomain.CORPORATE_ACTIONS, DataDomain.SHARE_CAPITAL}:
            return self._mootdx.fetch_domain(request)
        if request.domain in {
            DataDomain.MARKET_DAILY,
            DataDomain.TRADING_CALENDAR,
            DataDomain.UNIVERSE_SNAPSHOT,
            DataDomain.SECURITY_STATUS,
            DataDomain.INDUSTRY_CONCEPT,
            DataDomain.VALUATION,
            DataDomain.INDEX_CONSTITUENTS,
            DataDomain.ADJUST_FACTOR,
            DataDomain.FINANCIAL_QUARTERLY,
            DataDomain.PERFORMANCE_FORECAST,
            DataDomain.PERFORMANCE_EXPRESS,
        }:
            return self._baostock.fetch_domain(request)
        if request.domain == DataDomain.ANNOUNCEMENT:
            return self._cninfo.fetch_domain(request)
        raise RuntimeError(f"unsupported_domain: {self.name} does not support {request.domain}")


def build_default_providers(provider_plan: str = "default_free") -> list:
    plan = str(provider_plan or "default_free").strip().lower()
    if plan == "qdp_production_v1":
        return [QdpProductionV1Provider()]
    if plan == "qdp_production_v2":
        return [QdpProductionV2Provider()]
    if plan == "mootdx_online":
        return [MootdxOnlineProvider()]
    if plan == "research_rebuild_minimal_free":
        return [ResearchRebuildMinimalFreeProvider()]
    if plan == "default_free":
        return [EastmoneyEfinanceProvider(), AkshareEastmoneyProvider(), BaostockProvider()]
    if plan == "formal_free_v3":
        return [
            BaostockProvider(),
            EastmoneyEfinanceProvider(),
            AkshareEastmoneyProvider(),
            TencentFinanceProvider(),
            TonghuashunHotspotProvider(),
        ]
    if plan == "default_free_no_realtime":
        return [EastmoneyEfinanceProvider(), AkshareEastmoneyProvider(), BaostockProvider()]
    if plan == "baostock_only":
        return [BaostockProvider()]
    if plan == "default_free_with_realtime":
        return [EastmoneyEfinanceProvider(), AkshareEastmoneyProvider(), BaostockProvider(), SinaTencentRealtimeProvider()]
    if plan == "tushare_optional":
        return [TushareHttpOptionalProvider()]
    raise ValueError(f"Unsupported provider_plan: {provider_plan}")


def _open_mootdx_client(client_factory: Any = None) -> Any:
    if callable(client_factory):
        return client_factory()
    try:
        from mootdx.quotes import Quotes  # type: ignore
    except Exception as exc:
        raise RuntimeError("mootdx is not installed in the yolos environment") from exc
    options = (
        {"multithread": False, "heartbeat": False, "bestip": False, "timeout": 20},
        {"multithread": False, "heartbeat": True, "bestip": False, "timeout": 20},
        {"multithread": True, "heartbeat": True, "bestip": False, "timeout": 20},
    )
    errors: list[str] = []
    server_candidates = _mootdx_hq_server_candidates()
    for attempt in range(2):
        for server in server_candidates:
            for option in options:
                try:
                    return Quotes.factory(market="std", server=server, **option)
                except Exception as exc:
                    errors.append(f"attempt={attempt + 1} server={server} option={option}: {type(exc).__name__}: {exc}")
                    time.sleep(0.1)
        for option in options:
            try:
                return Quotes.factory(market="std", **option)
            except Exception as exc:
                errors.append(f"attempt={attempt + 1} server=default option={option}: {type(exc).__name__}: {exc}")
                time.sleep(0.1)
    raise RuntimeError("mootdx_client_open_failed: " + " | ".join(errors[-4:]))


def _mootdx_hq_server_candidates(limit: int = 12) -> tuple[tuple[str, int], ...]:
    try:
        import mootdx.config as mootdx_config  # type: ignore

        raw_servers = list(mootdx_config.get("SERVER.HQ") or [])
    except Exception:
        raw_servers = []
    out: list[tuple[str, int]] = []
    for item in raw_servers:
        try:
            if len(item) >= 3:
                out.append((str(item[1]), int(item[2])))
            elif len(item) >= 2:
                out.append((str(item[0]), int(item[1])))
        except Exception:
            continue
        if len(out) >= int(limit):
            break
    return tuple(dict.fromkeys(out))


def _close_mootdx_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            return


def _mootdx_symbol(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    if "." in raw:
        raw = raw.split(".", 1)[0]
    if raw.startswith(("SH", "SZ", "BJ")) and raw[2:].isdigit():
        raw = raw[2:]
    return raw[-6:].zfill(6)


def _mootdx_xdxr_domain_frame(
    raw: pd.DataFrame,
    *,
    domain: str,
    source: str,
    as_of_date: str,
) -> pd.DataFrame:
    """Convert TDX xdxr records while keeping ambiguous fields conservative.

    ``songzhuangu`` is a combined 送转 value in the TDX protocol, so it is
    deliberately not mislabeled as either bonus shares or capital-reserve
    transfers in the generic contract.  The lossless raw record is retained
    by QDP v3 and used by its dedicated corporate-action transform.
    """

    normalized_domain = str(domain)
    if raw is None or raw.empty:
        return normalize_domain_frame(
            pd.DataFrame(),
            domain=normalized_domain,
            source=source,
            as_of_date=as_of_date,
            require_columns=False,
        )
    frame = raw.copy()
    required = {"provider_symbol", "year", "month", "day", "category"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"mootdx_xdxr_missing_fields:{missing}")
    dates = pd.to_datetime(
        {
            "year": pd.to_numeric(frame["year"], errors="coerce"),
            "month": pd.to_numeric(frame["month"], errors="coerce"),
            "day": pd.to_numeric(frame["day"], errors="coerce"),
        },
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    category = pd.to_numeric(frame["category"], errors="coerce")
    if normalized_domain == DataDomain.CORPORATE_ACTIONS:
        mask = category.eq(1)
        selected = frame.loc[mask].copy()
        selected_dates = dates.loc[mask]
        generic = pd.DataFrame(
            {
                "symbol": selected["provider_symbol"],
                "trade_date": selected_dates,
                "ex_date": selected_dates,
                "action_type": selected.get("name", pd.Series(index=selected.index, dtype=str)),
                "cash_dividend_per_10": pd.to_numeric(selected.get("fenhong"), errors="coerce"),
                "bonus_share_per_10": np.nan,
                "transfer_share_per_10": np.nan,
                "description": "TDX xdxr; songzhuangu is combined and remains only in lossless raw evidence",
            },
            index=selected.index,
        )
    elif normalized_domain == DataDomain.SHARE_CAPITAL:
        total = pd.to_numeric(frame.get("houzongguben", pd.Series(index=frame.index, dtype=float)), errors="coerce") * 10_000.0
        floating = pd.to_numeric(frame.get("panhouliutong", pd.Series(index=frame.index, dtype=float)), errors="coerce") * 10_000.0
        mask = total.notna() | floating.notna()
        selected = frame.loc[mask].copy()
        generic = pd.DataFrame(
            {
                "symbol": selected["provider_symbol"],
                "trade_date": dates.loc[mask],
                "change_reason": selected.get("name", pd.Series(index=selected.index, dtype=str)),
                "total_share": total.loc[mask],
                "float_share": floating.loc[mask],
                "restricted_share": (total - floating).where(total.ge(floating)).loc[mask],
            },
            index=selected.index,
        )
    else:
        raise ValueError(f"mootdx_xdxr_unsupported_domain:{domain}")
    return normalize_domain_frame(
        generic.reset_index(drop=True),
        domain=normalized_domain,
        source=source,
        as_of_date=as_of_date,
        require_columns=False,
    )


def _is_mootdx_index_symbol(symbol: str) -> bool:
    raw = str(symbol or "").strip().upper()
    code = _mootdx_symbol(raw)
    suffix = raw.rsplit(".", 1)[1] if "." in raw else ""
    if suffix == "SH" and code.startswith(("000", "880", "881", "882", "883", "884", "885", "886", "887", "889")):
        return True
    if suffix == "SZ" and code.startswith("399"):
        return True
    return False


def _fetch_mootdx_bars_window(
    *,
    client: Any,
    symbol: str,
    frequency: int,
    start_date: str,
    end_date: str,
    page_size: int,
    max_pages: int,
    source: str,
    adjusted_flag: str,
    volume_factor: float,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    expected_pages = _estimated_mootdx_pages(
        start_date=start_date,
        end_date=end_date,
        frequency=int(frequency),
        page_size=int(page_size),
        configured_max_pages=int(max_pages),
    )
    for page in range(expected_pages):
        offset_start = int(page) * int(page_size)
        payload = _call_mootdx_bars_endpoint(
            client=client,
            symbol=symbol,
            frequency=int(frequency),
            start=offset_start,
            offset=int(page_size),
        )
        raw = _mootdx_payload_frame(payload)
        if raw.empty:
            break
        prepared = _prepare_mootdx_bars_frame(
            raw,
            symbol=symbol,
            source=source,
            adjusted_flag=adjusted_flag,
            volume_factor=float(volume_factor),
        )
        if prepared.empty:
            break
        frames.append(prepared)
        dates = pd.to_datetime(prepared["datetime"] if "datetime" in prepared.columns else prepared["trade_date"], errors="coerce").dropna()
        if len(dates) and pd.Timestamp(dates.min()).normalize() <= start_ts.normalize() and pd.Timestamp(dates.max()).normalize() >= end_ts.normalize():
            break
        if len(raw) < int(page_size):
            break
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return _filter_domain_date_window(combined, start_date, end_date)


def _estimated_mootdx_pages(*, start_date: str, end_date: str, frequency: int, page_size: int, configured_max_pages: int) -> int:
    page_size = max(1, int(page_size or 1))
    configured = max(1, int(configured_max_pages or 1))
    if int(frequency) == 8:
        bars_per_day = 240
    elif int(frequency) == 0:
        bars_per_day = 48
    else:
        bars_per_day = 1
    try:
        # mootdx bars are paged backward from the quote server's latest bar, not from
        # the requested end_date. Historical chunks therefore need enough pages to
        # reach start_date from "now", even when the chunk end_date is only a few
        # days after start_date.
        lookback_end = max(pd.Timestamp(end_date).normalize(), pd.Timestamp.today().normalize())
        business_days = max(1, len(pd.bdate_range(pd.Timestamp(start_date), lookback_end)))
    except Exception:
        return configured
    estimated_rows = int(business_days * bars_per_day)
    return min(240, max(configured, int(math.ceil(estimated_rows / page_size)) + 2))


def _call_mootdx_bars_endpoint(*, client: Any, symbol: str, frequency: int, start: int, offset: int) -> Any:
    method = getattr(client, "index_bars", None) if _is_mootdx_index_symbol(symbol) else None
    if not callable(method):
        method = getattr(client, "bars", None)
    if not callable(method):
        raise RuntimeError("mootdx client does not expose bars/index_bars")
    return method(symbol=_mootdx_symbol(symbol), frequency=int(frequency), start=int(start), offset=int(offset))


def _mootdx_payload_frame(payload: Any) -> pd.DataFrame:
    if isinstance(payload, pd.DataFrame):
        frame = payload.copy()
    else:
        frame = pd.DataFrame(payload)
    if frame.empty:
        return frame
    if "datetime" not in frame.columns and isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.copy()
        frame["datetime"] = frame.index
    return frame.reset_index(drop=True)


def _prepare_mootdx_bars_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    source: str,
    adjusted_flag: str,
    volume_factor: float,
) -> pd.DataFrame:
    data = frame.copy()
    data["symbol"] = str(symbol).strip().upper()
    data["source"] = source
    data["adjusted_flag"] = str(adjusted_flag or "none")
    if "trade_date" not in data.columns and "datetime" in data.columns:
        data["trade_date"] = data["datetime"]
    if "datetime" in data.columns:
        parsed_datetime = pd.to_datetime(data["datetime"], errors="coerce")
        if parsed_datetime.notna().any():
            data["bar_time"] = parsed_datetime.dt.strftime("%H:%M:%S")
    volume_source = None
    for candidate in ("volume", "vol", "成交量"):
        if candidate in data.columns:
            volume_source = candidate
            break
    if volume_source is not None:
        data["volume"] = pd.to_numeric(data[volume_source], errors="coerce") * float(volume_factor)
    return data


def _normalize_mootdx_quote_snapshot(
    frame: pd.DataFrame,
    *,
    symbols: tuple[str, ...],
    source: str,
    volume_factor: float,
) -> pd.DataFrame:
    data = frame.copy()
    code_to_symbol = {_mootdx_symbol(symbol): symbol for symbol in symbols}
    if "symbol" not in data.columns:
        if "code" in data.columns:
            data["symbol"] = data["code"].astype(str).str.zfill(6).map(code_to_symbol).fillna(data["code"].astype(str))
        elif len(data) == len(symbols):
            data["symbol"] = list(symbols)
    data["trade_date"] = pd.Timestamp.now().strftime("%Y-%m-%d")
    if "close" not in data.columns and "price" in data.columns:
        data["close"] = data["price"]
    volume_source = None
    for candidate in ("volume", "vol", "成交量"):
        if candidate in data.columns:
            volume_source = candidate
            break
    if volume_source is not None:
        data["volume"] = pd.to_numeric(data[volume_source], errors="coerce") * float(volume_factor)
    data["source"] = source
    data["adjusted_flag"] = "none"
    return normalize_market_frame(data, source=source, adjusted_flag="none", require_columns=False)


def _filter_domain_date_window(data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if data is None or data.empty or "trade_date" not in data.columns:
        return data if isinstance(data, pd.DataFrame) else pd.DataFrame()
    dates = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    mask = dates.ge(pd.Timestamp(start_date).normalize()) & dates.le(pd.Timestamp(end_date).normalize())
    return data.loc[mask].reset_index(drop=True)


def _fetch_cninfo_announcements(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    page_size: int,
    max_pages: int,
) -> pd.DataFrame:
    url = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
    headers = {
        "User-Agent": "Mozilla/5.0 qdp-cninfo-provider",
        "Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch",
    }
    rows: list[pd.DataFrame] = []
    for page in range(1, max(int(max_pages), 1) + 1):
        payload = {
            "pageNum": int(page),
            "pageSize": int(page_size),
            "column": "szse" if str(symbol).upper().endswith(".SZ") else "sse",
            "tabName": "fulltext",
            "stock": f"{_strip_suffix(symbol)},",
            "searchkey": "",
            "secid": "",
            "plate": "",
            "category": "",
            "trade": "",
            "seDate": f"{start_date}~{end_date}",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }
        response = requests.post(url, headers=headers, data=payload, timeout=20)
        if response.status_code >= 400:
            raise RuntimeError(f"cninfo_http_{response.status_code}: {response.text[:200]}")
        try:
            body = response.json()
        except Exception as exc:
            raise RuntimeError(f"cninfo_non_json_response: {response.text[:200]}") from exc
        announcements = body.get("announcements", []) if isinstance(body, dict) else []
        frame = pd.DataFrame(announcements)
        if frame.empty:
            break
        frame = frame.rename(columns={"announcementTitle": "title", "announcementTime": "trade_date", "adjunctUrl": "url", "announcementTypeName": "category"})
        if "trade_date" in frame.columns:
            values = pd.to_numeric(frame["trade_date"], errors="coerce")
            parsed_ms = pd.to_datetime(values, unit="ms", errors="coerce")
            parsed_text = pd.to_datetime(frame["trade_date"], errors="coerce")
            frame["trade_date"] = parsed_ms.fillna(parsed_text).dt.strftime("%Y-%m-%d")
        frame["symbol"] = str(symbol).strip().upper()
        frame["source"] = "cninfo"
        rows.append(frame)
        if len(frame) < int(page_size):
            break
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _to_baostock_code(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    if raw.endswith(".SH"):
        return f"sh.{raw[:6]}"
    if raw.endswith(".SZ"):
        return f"sz.{raw[:6]}"
    if raw.endswith(".BJ"):
        return f"bj.{raw[:6]}"
    if raw.startswith(("5", "6", "9")):
        return f"sh.{raw[:6]}"
    if raw.startswith(("4", "8")):
        return f"bj.{raw[:6]}"
    return f"sz.{raw[:6]}"


def _to_tencent_simple_code(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    code = raw.split(".", 1)[0] if "." in raw else raw[-6:]
    exchange = raw.split(".", 1)[1] if "." in raw else ("SH" if code.startswith(("5", "6", "9")) else "SZ")
    prefix = "sh" if exchange == "SH" else "sz"
    return f"s_{prefix}{code}"


def _from_tencent_code(value: Any) -> str:
    raw = str(value or "").strip().lower()
    raw = raw.removeprefix("s_")
    if raw.startswith("sh"):
        return f"{raw[2:8].upper()}.SH"
    if raw.startswith("sz"):
        return f"{raw[2:8].upper()}.SZ"
    return str(value or "").strip().upper()


def _from_baostock_code(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw.startswith("sh."):
        return f"{raw[3:].upper()}.SH"
    if raw.startswith("sz."):
        return f"{raw[3:].upper()}.SZ"
    if raw.startswith("bj."):
        return f"{raw[3:].upper()}.BJ"
    return str(value or "").strip().upper()


def _baostock_bulk_query_to_frame(query: Any, failure_label: str) -> pd.DataFrame:
    """Decode a BaoStock batch response exactly once.

    This function intentionally never calls ``next()`` or ``get_row_data()``.
    BaoStock 0.9.3's ordinary iterator uses 2,000 as a paging heuristic and
    can attempt a bogus page when a batch response contains exactly 2,000
    rows.  Batch endpoints carry one already-decoded ``data`` payload with a
    declared 20,000-row capacity, so direct structural validation is the only
    safe interpretation.
    """

    error_code = str(getattr(query, "error_code", "1"))
    error_msg = str(getattr(query, "error_msg", ""))
    if error_code != "0":
        raise RuntimeError(f"{failure_label}_query_error:{error_code}: {error_msg}")
    raw_per_page = getattr(query, "per_page_count", None)
    try:
        per_page_count = int(raw_per_page)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{failure_label}_invalid_per_page_count:{raw_per_page!r}") from exc
    if per_page_count != BAOSTOCK_BULK_PER_PAGE_COUNT:
        raise RuntimeError(
            f"{failure_label}_unexpected_per_page_count:"
            f"expected={BAOSTOCK_BULK_PER_PAGE_COUNT} actual={per_page_count}"
        )
    fields = [str(item) for item in (getattr(query, "fields", None) or [])]
    if len(fields) != len(set(fields)):
        raise RuntimeError(f"{failure_label}_duplicate_fields:{fields}")
    payload = getattr(query, "data", None)
    if payload is None:
        raise RuntimeError(f"{failure_label}_missing_bulk_payload")
    if not isinstance(payload, (list, tuple)):
        raise RuntimeError(f"{failure_label}_invalid_bulk_payload_type:{type(payload).__name__}")
    rows = list(payload)
    if len(rows) >= BAOSTOCK_BULK_PER_PAGE_COUNT:
        raise RuntimeError(
            f"{failure_label}_potential_truncation:"
            f"row_count={len(rows)} capacity={BAOSTOCK_BULK_PER_PAGE_COUNT}"
        )
    if rows and not fields:
        raise RuntimeError(f"{failure_label}_missing_fields_for_nonempty_payload")
    normalized_rows: list[list[Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)):
            raise RuntimeError(f"{failure_label}_invalid_row_type:index={index} type={type(row).__name__}")
        values = list(row)
        if len(values) != len(fields):
            raise RuntimeError(
                f"{failure_label}_field_width_mismatch:"
                f"index={index} expected={len(fields)} actual={len(values)}"
            )
        normalized_rows.append(values)
    return pd.DataFrame(normalized_rows, columns=fields)


def _baostock_bulk_daily_domain_frame(raw: pd.DataFrame, *, domain: str, query_date: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    required = {"date", "code"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(f"baostock_bulk_daily_schema_error:missing={missing}")
    provider_symbol = raw["code"].map(_from_baostock_code)
    if domain == DataDomain.MARKET_DAILY:
        market_fields = ["open", "high", "low", "close", "preclose", "volume", "amount", "pctChg", "adjustflag"]
        missing = sorted(set(market_fields) - set(raw.columns))
        if missing:
            raise RuntimeError(f"baostock_bulk_daily_market_schema_error:missing={missing}")
        frame = pd.DataFrame(
            {
                "trade_date": raw["date"].astype(str),
                "provider_symbol": provider_symbol,
                "open": pd.to_numeric(raw["open"], errors="coerce"),
                "high": pd.to_numeric(raw["high"], errors="coerce"),
                "low": pd.to_numeric(raw["low"], errors="coerce"),
                "close": pd.to_numeric(raw["close"], errors="coerce"),
                "preclose": pd.to_numeric(raw["preclose"], errors="coerce"),
                "volume": pd.to_numeric(raw["volume"], errors="coerce"),
                "amount": pd.to_numeric(raw["amount"], errors="coerce"),
                "pct_chg": pd.to_numeric(raw["pctChg"], errors="coerce"),
                "adjustflag": raw["adjustflag"].astype(str),
                "source": "baostock",
            }
        )
        return frame.reset_index(drop=True)
    if domain == DataDomain.SECURITY_STATUS:
        status_fields = ["tradestatus", "isST"]
        missing = sorted(set(status_fields) - set(raw.columns))
        if missing:
            raise RuntimeError(f"baostock_bulk_daily_status_schema_error:missing={missing}")
        trade_status = raw["tradestatus"].fillna("").astype(str).str.strip()
        return pd.DataFrame(
            {
                "trade_date": raw["date"].astype(str),
                "provider_symbol": provider_symbol,
                "tradestatus": trade_status,
                "is_st": raw["isST"].fillna("").astype(str).str.strip().eq("1"),
                "is_suspended": trade_status.ne("1"),
                "status_source": "baostock.query_daily_history_k_AStock",
                "source": "baostock",
            }
        ).reset_index(drop=True)
    if domain == DataDomain.VALUATION:
        valuation_fields = ["turn", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]
        missing = sorted(set(valuation_fields) - set(raw.columns))
        if missing:
            raise RuntimeError(f"baostock_bulk_daily_valuation_schema_error:missing={missing}")
        return pd.DataFrame(
            {
                "trade_date": raw["date"].astype(str),
                "provider_symbol": provider_symbol,
                "turnover_rate": pd.to_numeric(raw["turn"], errors="coerce"),
                "pe_ttm": pd.to_numeric(raw["peTTM"], errors="coerce"),
                "pb_mrq": pd.to_numeric(raw["pbMRQ"], errors="coerce"),
                "ps_ttm": pd.to_numeric(raw["psTTM"], errors="coerce"),
                "pcf_ncf_ttm": pd.to_numeric(raw["pcfNcfTTM"], errors="coerce"),
                "source": "baostock",
            }
        ).reset_index(drop=True)
    raise RuntimeError(f"baostock_bulk_daily_unsupported_domain:{domain}")


def _exact_column(frame: pd.DataFrame, candidates: tuple[str, ...], *, label: str) -> str:
    present = [name for name in candidates if name in frame.columns]
    if not present:
        raise RuntimeError(f"{label}_missing_field:accepted={list(candidates)}")
    if len(present) > 1:
        reference = frame[present[0]].astype(str)
        if any(not reference.equals(frame[name].astype(str)) for name in present[1:]):
            raise RuntimeError(f"{label}_conflicting_alias_fields:{present}")
    return present[0]


def _baostock_bulk_adjust_factor_event_frame(raw: pd.DataFrame, *, query_date: str) -> pd.DataFrame:
    columns = [
        "provider_symbol",
        "divid_operate_date",
        "fore_adjust_factor",
        "back_adjust_factor",
        "adjust_factor",
        "query_date",
        "source_method",
        "source",
    ]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=columns)
    code_field = _exact_column(raw, ("code",), label="baostock_bulk_adjust_factor")
    event_date_field = _exact_column(raw, ("dividOperateDate", "divid_operate_date"), label="baostock_bulk_adjust_factor")
    fore_field = _exact_column(raw, ("foreAdjustFactor", "fore_adjust_factor"), label="baostock_bulk_adjust_factor")
    back_field = _exact_column(raw, ("backAdjustFactor", "back_adjust_factor"), label="baostock_bulk_adjust_factor")
    factor_field = _exact_column(raw, ("adjustFacto", "adjustFactor", "adjust_factor"), label="baostock_bulk_adjust_factor")
    event_dates = raw[event_date_field].fillna("").astype(str).str.strip()
    unexpected = sorted(set(event_dates.loc[event_dates.ne(str(query_date))].tolist()))
    if unexpected:
        raise RuntimeError(
            "baostock_bulk_adjust_factor_event_date_mismatch:"
            f"query_date={query_date} returned={unexpected[:10]}"
        )
    return pd.DataFrame(
        {
            "provider_symbol": raw[code_field].map(_from_baostock_code),
            "divid_operate_date": event_dates,
            "fore_adjust_factor": pd.to_numeric(raw[fore_field], errors="coerce"),
            "back_adjust_factor": pd.to_numeric(raw[back_field], errors="coerce"),
            "adjust_factor": pd.to_numeric(raw[factor_field], errors="coerce"),
            "query_date": str(query_date),
            "source_method": "date_batch",
            "source": "baostock",
        }
    ).loc[:, columns].reset_index(drop=True)


def _baostock_query_to_frame(query: Any, failure_label: str) -> pd.DataFrame:
    error_code = str(getattr(query, "error_code", "1"))
    error_msg = str(getattr(query, "error_msg", ""))
    if error_code != "0":
        raise RuntimeError(f"{failure_label}_query_error:{error_code}: {error_msg}")
    fields = [str(item) for item in (getattr(query, "fields", None) or [])]
    rows: list[list[Any]] = []
    while query.next():
        rows.append(query.get_row_data())
    return pd.DataFrame(rows, columns=fields) if fields else pd.DataFrame(rows)


def _is_baostock_a_share_code(value: Any) -> bool:
    raw = str(value or "").strip().lower()
    if raw.startswith("sh."):
        code = raw[3:9]
        return code.startswith(("600", "601", "603", "605", "688"))
    if raw.startswith("sz."):
        code = raw[3:9]
        return code.startswith(("000", "001", "002", "003", "300", "301"))
    if raw.startswith("bj."):
        return raw[3:9].isdigit()
    return False


def _baostock_exchange(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if raw.endswith(".SH"):
        return "SH"
    if raw.endswith(".SZ"):
        return "SZ"
    if raw.endswith(".BJ"):
        return "BJ"
    return ""


def _baostock_board(value: Any) -> str:
    raw = str(value or "").strip().upper()
    code = raw.split(".", 1)[0]
    exchange = raw.split(".", 1)[1] if "." in raw else ""
    if exchange == "BJ":
        return "beijing"
    if exchange == "SH" and code.startswith("688"):
        return "star"
    if exchange == "SZ" and code.startswith(("300", "301")):
        return "chi_next"
    if exchange in {"SH", "SZ"}:
        return "main"
    return "unknown"


def _baostock_name_is_st(value: Any) -> bool:
    name = str(value or "").strip().upper()
    return name.startswith(("ST", "*ST"))


def _baostock_all_stock_raw_frame(query: Any) -> pd.DataFrame:
    frame = _baostock_query_to_frame(query, "baostock_all_stock")
    if frame.empty or "code" not in frame.columns:
        return pd.DataFrame()
    frame = frame.loc[frame["code"].map(_is_baostock_a_share_code)].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["symbol"] = frame["code"].map(_from_baostock_code)
    frame["name"] = frame.get("code_name", "").fillna("").astype(str).str.strip()
    frame["tradeStatus"] = frame.get("tradeStatus", "").fillna("").astype(str).str.strip()
    return frame


def _baostock_all_stock_frame(query: Any, *, trade_date: str) -> pd.DataFrame:
    raw = _baostock_all_stock_raw_frame(query)
    if raw.empty:
        return pd.DataFrame()
    frame = pd.DataFrame(
        {
            "symbol": raw["symbol"],
            "trade_date": str(trade_date),
            "name": raw["name"],
            "exchange": raw["symbol"].map(_baostock_exchange),
            "board": raw["symbol"].map(_baostock_board),
            "list_status": "L",
            "list_date": "",
            "delist_date": "",
            # QDP v3's audit-specific accessor preserves these extension
            # columns.  The normal universe contract still drops them.
            "trade_status": raw["tradeStatus"],
            "is_suspended": raw["tradeStatus"].eq("0"),
            "source": "baostock",
        }
    )
    return frame.reset_index(drop=True)


def _baostock_status_frame_from_all_stock(query: Any, *, trade_date: str) -> pd.DataFrame:
    raw = _baostock_all_stock_raw_frame(query)
    if raw.empty:
        return pd.DataFrame()
    trade_status = raw["tradeStatus"].fillna("").astype(str).str.strip()
    frame = pd.DataFrame(
        {
            "symbol": raw["symbol"],
            "trade_date": str(trade_date),
            "is_st": raw["name"].map(_baostock_name_is_st),
            "is_suspended": trade_status.eq("0"),
            "is_delisted": False,
            "status_reason": "tradeStatus=" + trade_status,
            "source": "baostock",
        }
    )
    return frame.reset_index(drop=True)


def _baostock_industry_frame(query: Any, *, trade_date: str) -> pd.DataFrame:
    frame = _baostock_query_to_frame(query, "baostock_industry")
    if frame.empty or "code" not in frame.columns:
        return pd.DataFrame()
    frame = frame.loc[frame["code"].map(_is_baostock_a_share_code)].copy()
    if frame.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "symbol": frame["code"].map(_from_baostock_code),
            "trade_date": str(trade_date),
            "industry": frame.get("industry", "").fillna("").astype(str).str.strip(),
            "concept_tags": "",
            "source": "baostock",
        }
    ).reset_index(drop=True)


def _baostock_valuation_frame_from_history(bs: Any, request: DomainFetchRequest, *, relogin_retries: int = 2) -> pd.DataFrame:
    request = request.normalized()
    frames: list[pd.DataFrame] = []
    for symbol in request.symbols:
        raw = _baostock_valuation_symbol_frame_with_relogin(
            bs,
            symbol=symbol,
            request=request,
            relogin_retries=relogin_retries,
        )
        if raw.empty:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "symbol": raw["code"].map(_from_baostock_code),
                    "trade_date": raw["date"],
                    "total_mv": float("nan"),
                    "circ_mv": float("nan"),
                    "pe": raw.get("peTTM", ""),
                    "pb": raw.get("pbMRQ", ""),
                    "turnover_rate": raw.get("turn", ""),
                    "source": "baostock",
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _baostock_valuation_symbol_frame_with_relogin(
    bs: Any,
    *,
    symbol: str,
    request: DomainFetchRequest,
    relogin_retries: int,
) -> pd.DataFrame:
    return _baostock_query_to_frame_with_relogin(
        bs,
        lambda: bs.query_history_k_data_plus(
            _to_baostock_code(symbol),
            "date,code,turn,peTTM,pbMRQ",
            start_date=request.start_date,
            end_date=request.end_date,
            frequency="d",
            adjustflag="3",
        ),
        "baostock_valuation",
        relogin_retries=relogin_retries,
        relogin_context="valuation",
    )


def _is_baostock_not_logged_in_error(exc: BaseException) -> bool:
    message = str(exc)
    return "10001001" in message or "用户未登录" in message


def _baostock_query_to_frame_with_relogin(
    bs: Any,
    query_factory: Any,
    failure_label: str,
    *,
    relogin_retries: int = 2,
    relogin_context: str = "query",
) -> pd.DataFrame:
    attempts = max(1, int(relogin_retries or 0) + 1)
    for attempt in range(1, attempts + 1):
        try:
            return _baostock_query_to_frame(query_factory(), failure_label)
        except RuntimeError as exc:
            if attempt >= attempts or not _is_baostock_not_logged_in_error(exc):
                raise
            try:
                _quiet_baostock_call(bs.logout)
            except Exception:
                pass
            time.sleep(min(2.0 * attempt, 5.0))
            login = _quiet_baostock_call(bs.login)
            if getattr(login, "error_code", "1") != "0" and attempt >= attempts - 1:
                raise RuntimeError(f"baostock {relogin_context} relogin failed: {getattr(login, 'error_msg', '')}") from exc
    return pd.DataFrame()


def _baostock_history_frame(query: Any, *, symbol: str) -> pd.DataFrame:
    raw = _baostock_query_to_frame(query, "baostock_history")
    if raw.empty:
        return pd.DataFrame()
    rename_map = {
        "date": "trade_date",
        "code": "symbol",
    }
    frame = raw.rename(columns=rename_map).copy()
    frame["symbol"] = str(symbol).strip().upper()
    expected = ["trade_date", "symbol", "open", "high", "low", "close", "volume", "amount"]
    return frame[[column for column in expected if column in frame.columns]]


def _baostock_intraday_5m_frame(query: Any, *, symbol: str) -> pd.DataFrame:
    raw = _baostock_query_to_frame(query, "baostock_intraday_5m")
    if raw.empty:
        return pd.DataFrame()
    frame = raw.rename(columns={"date": "trade_date", "time": "bar_time", "code": "symbol"}).copy()
    frame["symbol"] = str(symbol).strip().upper()
    expected = ["trade_date", "symbol", "bar_time", "open", "high", "low", "close", "volume", "amount", "adjustflag"]
    return frame[[column for column in expected if column in frame.columns]].rename(columns={"adjustflag": "adjusted_flag"})


def _baostock_index_constituents_frame(bs: Any, *, trade_date: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    specs = [
        ("000016.SH", "SSE 50", bs.query_sz50_stocks),
        ("000300.SH", "CSI 300", bs.query_hs300_stocks),
        ("000905.SH", "CSI 500", bs.query_zz500_stocks),
    ]
    for index_symbol, index_name, query_func in specs:
        try:
            raw = _baostock_query_to_frame(query_func(date=trade_date), f"baostock_{index_symbol}_constituents")
        except TypeError:
            raw = _baostock_query_to_frame(query_func(), f"baostock_{index_symbol}_constituents")
        if raw.empty or "code" not in raw.columns:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "index_symbol": index_symbol,
                    "symbol": raw["code"].map(_from_baostock_code),
                    "trade_date": raw.get("date", trade_date),
                    "index_name": index_name,
                    "source": "baostock",
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _baostock_financial_quarterly_frame_from_bs(bs: Any, request: DomainFetchRequest, *, relogin_retries: int = 2) -> pd.DataFrame:
    request = request.normalized()
    rows: list[dict[str, Any]] = []
    query_specs = [
        ("profit", bs.query_profit_data),
        ("operation", bs.query_operation_data),
        ("growth", bs.query_growth_data),
        ("balance", bs.query_balance_data),
        ("cash_flow", bs.query_cash_flow_data),
    ]
    for symbol in request.symbols:
        code = _to_baostock_code(symbol)
        for year, quarter, report_date in _quarter_points(request.start_date, request.end_date):
            row: dict[str, Any] = {
                "symbol": symbol,
                "fiscal_year": year,
                "fiscal_quarter": quarter,
                "report_date": report_date,
                "publish_date": "",
                "lag_policy": "conservative_report_date_plus_90bd_plus_1d_in_features",
                "source": "baostock",
            }
            has_payload = False
            for _label, query_func in query_specs:
                raw = _baostock_query_to_frame_with_relogin(
                    bs,
                    lambda query_func=query_func, code=code, year=year, quarter=quarter: query_func(code=code, year=year, quarter=quarter),
                    "baostock_financial_quarterly",
                    relogin_retries=relogin_retries,
                    relogin_context="financial_quarterly",
                )
                if raw.empty:
                    continue
                has_payload = True
                payload = raw.iloc[-1].to_dict()
                row.update({str(key): value for key, value in payload.items()})
            if has_payload:
                # Several BaoStock finance tables expose ``pubDate`` while
                # the adapter also predeclares ``publish_date``.  Preserve
                # the provider date explicitly; otherwise the generic
                # normalizer would conservatively infer one from report_date
                # and erase valuable PIT evidence.
                provider_publish = next(
                    (
                        str(row.get(key, "") or "").strip()
                        for key in ("pubDate", "publishDate")
                        if str(row.get(key, "") or "").strip()
                    ),
                    "",
                )
                if provider_publish:
                    row["publish_date"] = provider_publish
                    row["lag_policy"] = "publish_date_plus_1d_in_features"
                rows.append(row)
    return pd.DataFrame(rows)


def _baostock_performance_frame_from_bs(bs: Any, request: DomainFetchRequest, *, relogin_retries: int = 2) -> pd.DataFrame:
    request = request.normalized()
    rows: list[pd.DataFrame] = []
    query_func = bs.query_forecast_report if request.domain == DataDomain.PERFORMANCE_FORECAST else bs.query_performance_express_report
    failure_label = "baostock_performance_forecast" if request.domain == DataDomain.PERFORMANCE_FORECAST else "baostock_performance_express"
    for symbol in request.symbols:
        raw = _baostock_query_to_frame_with_relogin(
            bs,
            lambda query_func=query_func, symbol=symbol: query_func(_to_baostock_code(symbol), start_date=request.start_date, end_date=request.end_date),
            failure_label,
            relogin_retries=relogin_retries,
            relogin_context=request.domain,
        )
        if raw.empty:
            continue
        raw = raw.copy()
        raw["symbol"] = symbol
        raw["source"] = "baostock"
        rows.append(raw)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _baostock_adjust_factor_frame_from_bs(bs: Any, request: DomainFetchRequest, *, relogin_retries: int = 2) -> pd.DataFrame:
    request = request.normalized()
    rows: list[pd.DataFrame] = []
    for symbol in request.symbols:
        raw = _baostock_query_to_frame_with_relogin(
            bs,
            lambda symbol=symbol: bs.query_adjust_factor(
                code=_to_baostock_code(symbol),
                start_date=request.start_date,
                end_date=request.end_date,
            ),
            "baostock_adjust_factor",
            relogin_retries=relogin_retries,
            relogin_context="adjust_factor",
        )
        if raw.empty:
            continue
        frame = raw.copy()
        frame["symbol"] = symbol
        frame["factor_provider"] = "baostock"
        frame["factor_semantics"] = "baostock_adjust_factor"
        frame["source"] = "baostock"
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _quarter_points(start_date: str, end_date: str) -> list[tuple[int, int, str]]:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    start_year = int(start_ts.year)
    end_year = int(end_ts.year)
    points: list[tuple[int, int, str]] = []
    for year in range(start_year, end_year + 1):
        for quarter, month_day in ((1, "03-31"), (2, "06-30"), (3, "09-30"), (4, "12-31")):
            report_date = pd.Timestamp(f"{year}-{month_day}")
            if start_ts <= report_date <= end_ts:
                points.append((year, quarter, report_date.strftime("%Y-%m-%d")))
    return points


def _baostock_symbol_error(provider: str, symbol: str, exc: BaseException, *, first_error: BaseException | None = None) -> dict[str, Any]:
    message = str(exc)
    if first_error is not None and str(first_error) and str(first_error) != message:
        message = f"{message}; first_error={first_error}"
    return {
        "provider": provider,
        "domain": DataDomain.MARKET_DAILY,
        "symbol": str(symbol),
        "code": "symbol_fetch_timeout" if isinstance(exc, TimeoutError) else "symbol_fetch_exception",
        "error_type": type(exc).__name__,
        "message": message,
    }


def _baostock_bulk_partition_worker(queue: Any, endpoint: str, trade_date: str) -> None:
    try:
        import baostock as bs  # type: ignore

        installed = assert_baostock_batch_runtime()
        login = _quiet_baostock_call(bs.login)
        if getattr(login, "error_code", "1") != "0":
            queue.put({"status": "error", "error_type": "RuntimeError", "error": f"baostock login failed: {getattr(login, 'error_msg', '')}"})
            return
        try:
            if endpoint == "query_daily_history_k_AStock":
                query = bs.query_daily_history_k_AStock(date=trade_date)
            elif endpoint == "query_daily_history_k_ETF":
                query = bs.query_daily_history_k_ETF(date=trade_date)
            elif endpoint == "query_daily_adjust_factor":
                query = bs.query_daily_adjust_factor(date=trade_date)
            else:
                raise RuntimeError(f"unsupported_baostock_bulk_endpoint:{endpoint}")
            frame = _baostock_bulk_query_to_frame(query, endpoint)
            meta = {
                "error_code": str(getattr(query, "error_code", "")),
                "error_msg": str(getattr(query, "error_msg", "")),
                "per_page_count": int(getattr(query, "per_page_count", 0) or 0),
                "fields": [str(item) for item in (getattr(query, "fields", None) or [])],
                "package_version": installed,
            }
        finally:
            _quiet_baostock_call(bs.logout)
        queue.put({"status": "ok", "data": frame, "meta": meta})
    except Exception as exc:
        queue.put({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})


def _baostock_persistent_session_worker(command_queue: Any, response_queue: Any) -> None:
    """Serve sequential BaoStock commands under one isolated login."""

    import baostock as bs  # type: ignore

    login = _quiet_baostock_call(bs.login)
    login_error = ""
    if getattr(login, "error_code", "1") != "0":
        login_error = f"baostock login failed: {getattr(login, 'error_msg', '')}"
    try:
        while True:
            command = command_queue.get()
            request_id = int(command.get("request_id", 0) or 0) if isinstance(command, dict) else 0
            if not isinstance(command, dict):
                response_queue.put(
                    {"status": "error", "request_id": request_id, "error_type": "TypeError", "error": "persistent command must be a mapping"}
                )
                continue
            if command.get("kind") == "close":
                break
            if login_error:
                response_queue.put(
                    {"status": "error", "request_id": request_id, "error_type": "RuntimeError", "error": login_error}
                )
                break
            started = time.perf_counter()
            try:
                kind = str(command.get("kind", ""))
                trade_date = str(command.get("trade_date", ""))
                extra_response: dict[str, Any] = {}
                if kind in {"bulk", "bulk_with_all_stock"}:
                    endpoint = str(command.get("endpoint", ""))
                    installed = assert_baostock_batch_runtime()
                    if endpoint == "query_daily_history_k_AStock":
                        query = bs.query_daily_history_k_AStock(date=trade_date)
                    elif endpoint == "query_daily_history_k_ETF":
                        query = bs.query_daily_history_k_ETF(date=trade_date)
                    elif endpoint == "query_daily_adjust_factor":
                        query = bs.query_daily_adjust_factor(date=trade_date)
                    else:
                        raise RuntimeError(f"unsupported_baostock_bulk_endpoint:{endpoint}")
                    frame = _baostock_bulk_query_to_frame(query, endpoint)
                    meta = {
                        "error_code": str(getattr(query, "error_code", "")),
                        "error_msg": str(getattr(query, "error_msg", "")),
                        "per_page_count": int(getattr(query, "per_page_count", 0) or 0),
                        "fields": [str(item) for item in (getattr(query, "fields", None) or [])],
                        "package_version": installed,
                    }
                    if kind == "bulk_with_all_stock":
                        audit_query = bs.query_all_stock(day=trade_date)
                        extra_response["audit_data"] = _baostock_all_stock_frame(
                            audit_query, trade_date=trade_date
                        )
                elif kind == "all_stock":
                    domain = str(command.get("domain", DataDomain.UNIVERSE_SNAPSHOT))
                    query = bs.query_all_stock(day=trade_date)
                    frame = (
                        _baostock_status_frame_from_all_stock(query, trade_date=trade_date)
                        if domain == DataDomain.SECURITY_STATUS
                        else _baostock_all_stock_frame(query, trade_date=trade_date)
                    )
                    meta = {"error_code": "0", "error_msg": "", "package_version": assert_baostock_batch_runtime()}
                else:
                    raise RuntimeError(f"unsupported_baostock_persistent_command:{kind}")
                response_queue.put(
                    {
                        "status": "ok",
                        "request_id": request_id,
                        "data": frame,
                        "meta": {**meta, "worker_elapsed_seconds": round(time.perf_counter() - started, 6)},
                        **extra_response,
                    }
                )
            except Exception as exc:
                response_queue.put(
                    {"status": "error", "request_id": request_id, "error_type": type(exc).__name__, "error": str(exc)}
                )
                break
    finally:
        if not login_error:
            _quiet_baostock_call(bs.logout)


def _baostock_history_worker(
    queue: Any,
    symbol: str,
    start_date: str,
    end_date: str,
    adjusted_flag: str,
) -> None:
    try:
        import baostock as bs  # type: ignore

        login = _quiet_baostock_call(bs.login)
        if getattr(login, "error_code", "1") != "0":
            queue.put({"status": "error", "error_type": "RuntimeError", "error": f"baostock login failed: {getattr(login, 'error_msg', '')}"})
            return
        try:
            query = bs.query_history_k_data_plus(
                _to_baostock_code(symbol),
                "date,code,open,high,low,close,volume,amount",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2" if adjusted_flag in {"front", "qfq"} else "3",
            )
            frame = _baostock_history_frame(query, symbol=symbol)
        finally:
            _quiet_baostock_call(bs.logout)
        queue.put({"status": "ok", "data": frame})
    except Exception as exc:
        queue.put({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})


def _baostock_intraday_5m_worker(
    queue: Any,
    symbol: str,
    start_date: str,
    end_date: str,
    adjusted_flag: str,
) -> None:
    try:
        import baostock as bs  # type: ignore

        login = _quiet_baostock_call(bs.login)
        if getattr(login, "error_code", "1") != "0":
            queue.put({"status": "error", "error_type": "RuntimeError", "error": f"baostock login failed: {getattr(login, 'error_msg', '')}"})
            return
        try:
            query = bs.query_history_k_data_plus(
                _to_baostock_code(symbol),
                "date,time,code,open,high,low,close,volume,amount,adjustflag",
                start_date=start_date,
                end_date=end_date,
                frequency="5",
                adjustflag="2" if adjusted_flag in {"front", "qfq"} else "3",
            )
            frame = _baostock_intraday_5m_frame(query, symbol=symbol)
        finally:
            _quiet_baostock_call(bs.logout)
        queue.put({"status": "ok", "data": frame})
    except Exception as exc:
        queue.put({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})


def _baostock_stock_basic_worker(queue: Any, trade_date: str) -> None:
    try:
        import baostock as bs  # type: ignore

        login = _quiet_baostock_call(bs.login)
        if getattr(login, "error_code", "1") != "0":
            queue.put({"status": "error", "error_type": "RuntimeError", "error": f"baostock login failed: {getattr(login, 'error_msg', '')}"})
            return
        try:
            frame = _baostock_stock_basic_frame(bs.query_stock_basic(), trade_date=trade_date)
        finally:
            _quiet_baostock_call(bs.logout)
        queue.put({"status": "ok", "data": frame})
    except Exception as exc:
        queue.put({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})


def _baostock_all_stock_worker(queue: Any, domain: str, trade_date: str) -> None:
    try:
        import baostock as bs  # type: ignore

        login = _quiet_baostock_call(bs.login)
        if getattr(login, "error_code", "1") != "0":
            queue.put({"status": "error", "error_type": "RuntimeError", "error": f"baostock login failed: {getattr(login, 'error_msg', '')}"})
            return
        try:
            query = bs.query_all_stock(day=trade_date)
            if domain == DataDomain.SECURITY_STATUS:
                frame = _baostock_status_frame_from_all_stock(query, trade_date=trade_date)
            else:
                frame = _baostock_all_stock_frame(query, trade_date=trade_date)
        finally:
            _quiet_baostock_call(bs.logout)
        queue.put({"status": "ok", "data": frame})
    except Exception as exc:
        queue.put({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})


def _baostock_industry_worker(queue: Any, trade_date: str) -> None:
    try:
        import baostock as bs  # type: ignore

        login = _quiet_baostock_call(bs.login)
        if getattr(login, "error_code", "1") != "0":
            queue.put({"status": "error", "error_type": "RuntimeError", "error": f"baostock login failed: {getattr(login, 'error_msg', '')}"})
            return
        try:
            frame = _baostock_industry_frame(bs.query_stock_industry(date=trade_date), trade_date=trade_date)
        finally:
            _quiet_baostock_call(bs.logout)
        queue.put({"status": "ok", "data": frame})
    except Exception as exc:
        queue.put({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})


def _baostock_index_constituents_worker(queue: Any, trade_date: str) -> None:
    try:
        import baostock as bs  # type: ignore

        login = _quiet_baostock_call(bs.login)
        if getattr(login, "error_code", "1") != "0":
            queue.put({"status": "error", "error_type": "RuntimeError", "error": f"baostock login failed: {getattr(login, 'error_msg', '')}"})
            return
        try:
            frame = _baostock_index_constituents_frame(bs, trade_date=trade_date)
        finally:
            _quiet_baostock_call(bs.logout)
        queue.put({"status": "ok", "data": frame})
    except Exception as exc:
        queue.put({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})


def _baostock_financial_quarterly_worker(queue: Any, symbols: tuple[str, ...], start_date: str, end_date: str) -> None:
    try:
        import baostock as bs  # type: ignore

        login = _quiet_baostock_call(bs.login)
        if getattr(login, "error_code", "1") != "0":
            queue.put({"status": "error", "error_type": "RuntimeError", "error": f"baostock login failed: {getattr(login, 'error_msg', '')}"})
            return
        try:
            frame = _baostock_financial_quarterly_frame_from_bs(
                bs,
                DomainFetchRequest(
                    domain=DataDomain.FINANCIAL_QUARTERLY,
                    symbols=tuple(symbols),
                    start_date=start_date,
                    end_date=end_date,
                ),
            )
        finally:
            _quiet_baostock_call(bs.logout)
        queue.put({"status": "ok", "data": frame})
    except Exception as exc:
        queue.put({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})


def _baostock_performance_worker(queue: Any, domain: str, symbols: tuple[str, ...], start_date: str, end_date: str) -> None:
    try:
        import baostock as bs  # type: ignore

        login = _quiet_baostock_call(bs.login)
        if getattr(login, "error_code", "1") != "0":
            queue.put({"status": "error", "error_type": "RuntimeError", "error": f"baostock login failed: {getattr(login, 'error_msg', '')}"})
            return
        try:
            frame = _baostock_performance_frame_from_bs(
                bs,
                DomainFetchRequest(
                    domain=domain,
                    symbols=tuple(symbols),
                    start_date=start_date,
                    end_date=end_date,
                ),
            )
        finally:
            _quiet_baostock_call(bs.logout)
        queue.put({"status": "ok", "data": frame})
    except Exception as exc:
        queue.put({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})


def _baostock_adjust_factor_worker(queue: Any, symbols: tuple[str, ...], start_date: str, end_date: str) -> None:
    try:
        import baostock as bs  # type: ignore

        login = _quiet_baostock_call(bs.login)
        if getattr(login, "error_code", "1") != "0":
            queue.put({"status": "error", "error_type": "RuntimeError", "error": f"baostock login failed: {getattr(login, 'error_msg', '')}"})
            return
        try:
            frame = _baostock_adjust_factor_frame_from_bs(
                bs,
                DomainFetchRequest(
                    domain=DataDomain.ADJUST_FACTOR,
                    symbols=tuple(symbols),
                    start_date=start_date,
                    end_date=end_date,
                ),
            )
        finally:
            _quiet_baostock_call(bs.logout)
        queue.put({"status": "ok", "data": frame})
    except Exception as exc:
        queue.put({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})


def _baostock_valuation_worker(queue: Any, symbols: tuple[str, ...], start_date: str, end_date: str) -> None:
    try:
        import baostock as bs  # type: ignore

        login = _quiet_baostock_call(bs.login)
        if getattr(login, "error_code", "1") != "0":
            queue.put({"status": "error", "error_type": "RuntimeError", "error": f"baostock login failed: {getattr(login, 'error_msg', '')}"})
            return
        try:
            frame = _baostock_valuation_frame_from_history(
                bs,
                DomainFetchRequest(
                    domain=DataDomain.VALUATION,
                    symbols=tuple(symbols),
                    start_date=start_date,
                    end_date=end_date,
                ),
            )
        finally:
            _quiet_baostock_call(bs.logout)
        queue.put({"status": "ok", "data": frame})
    except Exception as exc:
        queue.put({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})


def _baostock_trade_calendar_worker(queue: Any, start_date: str, end_date: str, exchange: str) -> None:
    try:
        import baostock as bs  # type: ignore

        login = _quiet_baostock_call(bs.login)
        if getattr(login, "error_code", "1") != "0":
            queue.put({"status": "error", "error_type": "RuntimeError", "error": f"baostock login failed: {getattr(login, 'error_msg', '')}"})
            return
        try:
            query = bs.query_trade_dates(start_date=start_date, end_date=end_date)
            rows: list[list[Any]] = []
            while getattr(query, "error_code", "1") == "0" and query.next():
                rows.append(query.get_row_data())
            frame = pd.DataFrame(rows, columns=["trade_date", "is_open"])
            frame["exchange"] = exchange
        finally:
            _quiet_baostock_call(bs.logout)
        queue.put({"status": "ok", "data": frame})
    except Exception as exc:
        queue.put({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})


def _fetch_baostock_bulk_partition_once(
    *,
    endpoint: str,
    trade_date: str,
    timeout_seconds: int = 120,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    timeout = max(float(timeout_seconds or 0), 1.0)
    context = multiprocessing.get_context("spawn")
    payload_queue = context.Queue()
    process = context.Process(
        target=_baostock_bulk_partition_worker,
        kwargs={"queue": payload_queue, "endpoint": str(endpoint), "trade_date": str(trade_date)},
    )
    with _BAOSTOCK_GLOBAL_LIMITER.slot():
        process.start()
        try:
            payload = payload_queue.get(timeout=timeout)
        except queue_module.Empty:
            process.join(0)
            if process.is_alive():
                process.terminate()
                process.join(5)
                raise TimeoutError(f"baostock_bulk_timeout:{endpoint}: exceeded {int(timeout)} seconds")
            if process.exitcode not in {0, None}:
                raise RuntimeError(f"baostock_bulk_worker_failed:{endpoint}:exitcode={process.exitcode}")
            raise RuntimeError(f"baostock_bulk_worker_returned_no_payload:{endpoint}")
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
            raise RuntimeError(f"baostock_bulk_worker_did_not_exit:{endpoint}")
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        error_type = str(payload.get("error_type", "RuntimeError")) if isinstance(payload, dict) else "RuntimeError"
        error = str(payload.get("error", payload) if isinstance(payload, dict) else payload)
        raise RuntimeError(f"baostock_bulk_worker_error:{endpoint}:{error_type}: {error}")
    data = payload.get("data")
    if not isinstance(data, pd.DataFrame):
        raise RuntimeError(f"baostock_bulk_worker_invalid_data:{endpoint}:{type(data).__name__}")
    meta = dict(payload.get("meta", {}) or {})
    return data.copy(), meta


def _fetch_baostock_bulk_partition_with_retry(
    *,
    endpoint: str,
    trade_date: str,
    timeout_seconds: int = 120,
    backoff_seconds: tuple[int, ...] = (2, 5, 15),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    errors: list[str] = []
    attempts = len(tuple(backoff_seconds)) + 1
    for attempt in range(1, attempts + 1):
        try:
            frame, meta = _fetch_baostock_bulk_partition_once(
                endpoint=endpoint,
                trade_date=trade_date,
                timeout_seconds=timeout_seconds,
            )
            meta.update({"attempt_count": attempt, "retry_errors": list(errors)})
            return frame, meta
        except Exception as exc:
            errors.append(f"attempt={attempt}:{type(exc).__name__}:{exc}")
            if attempt >= attempts:
                break
            time.sleep(float(tuple(backoff_seconds)[attempt - 1]))
    raise RuntimeError(
        f"baostock_bulk_fetch_failed:endpoint={endpoint} trade_date={trade_date} "
        f"attempts={attempts} errors={' | '.join(errors)}"
    )


def _fetch_baostock_payload_with_timeout(
    *,
    target: Any,
    kwargs: dict[str, Any],
    timeout_seconds: int = 60,
    timeout_label: str,
    failure_label: str,
) -> pd.DataFrame:
    timeout = max(float(timeout_seconds or 0), 1.0)
    context = multiprocessing.get_context("spawn")
    payload_queue = context.Queue()
    process = context.Process(
        target=target,
        kwargs={"queue": payload_queue, **kwargs},
    )
    with _BAOSTOCK_GLOBAL_LIMITER.slot():
        process.start()
        try:
            payload = payload_queue.get(timeout=timeout)
        except queue_module.Empty:
            process.join(0)
            if process.is_alive():
                process.terminate()
                process.join(5)
                raise TimeoutError(f"{timeout_label}: exceeded {int(timeout)} seconds")
            if process.exitcode not in {0, None}:
                raise RuntimeError(f"{failure_label}_worker_failed: exitcode={process.exitcode}")
            raise RuntimeError(f"{failure_label}_worker_returned_no_payload")
        process.join(5)
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        error_type = str(payload.get("error_type", "RuntimeError")) if isinstance(payload, dict) else "RuntimeError"
        error = str(payload.get("error", payload) if isinstance(payload, dict) else payload)
        raise RuntimeError(f"{failure_label}_worker_error:{error_type}: {error}")
    data = payload.get("data")
    return data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame()


def _fetch_baostock_stock_basic_frame_with_timeout(*, trade_date: str, timeout_seconds: int = 60) -> pd.DataFrame:
    return _fetch_baostock_payload_with_timeout(
        target=_baostock_stock_basic_worker,
        kwargs={"trade_date": str(trade_date)},
        timeout_seconds=timeout_seconds,
        timeout_label="baostock_stock_basic_timeout",
        failure_label="baostock_stock_basic",
    )


def _fetch_baostock_history_frame_with_timeout(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    adjusted_flag: str,
    timeout_seconds: int = 60,
) -> pd.DataFrame:
    return _fetch_baostock_payload_with_timeout(
        target=_baostock_history_worker,
        kwargs={
            "symbol": str(symbol),
            "start_date": str(start_date),
            "end_date": str(end_date),
            "adjusted_flag": str(adjusted_flag or "none"),
        },
        timeout_seconds=timeout_seconds,
        timeout_label="baostock_history_timeout",
        failure_label="baostock_history",
    )


def _fetch_baostock_intraday_5m_frame_with_timeout(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    adjusted_flag: str,
    timeout_seconds: int = 90,
) -> pd.DataFrame:
    return _fetch_baostock_payload_with_timeout(
        target=_baostock_intraday_5m_worker,
        kwargs={
            "symbol": str(symbol),
            "start_date": str(start_date),
            "end_date": str(end_date),
            "adjusted_flag": str(adjusted_flag or "none"),
        },
        timeout_seconds=timeout_seconds,
        timeout_label="baostock_intraday_5m_timeout",
        failure_label="baostock_intraday_5m",
    )


def _fetch_baostock_all_stock_frame_with_timeout(*, domain: str, trade_date: str, timeout_seconds: int = 60) -> pd.DataFrame:
    return _fetch_baostock_payload_with_timeout(
        target=_baostock_all_stock_worker,
        kwargs={"domain": str(domain), "trade_date": str(trade_date)},
        timeout_seconds=timeout_seconds,
        timeout_label="baostock_all_stock_timeout",
        failure_label="baostock_all_stock",
    )


def _fetch_baostock_industry_frame_with_timeout(*, trade_date: str, timeout_seconds: int = 300) -> pd.DataFrame:
    return _fetch_baostock_payload_with_timeout(
        target=_baostock_industry_worker,
        kwargs={"trade_date": str(trade_date)},
        timeout_seconds=timeout_seconds,
        timeout_label="baostock_industry_timeout",
        failure_label="baostock_industry",
    )


def _fetch_baostock_index_constituents_frame_with_timeout(*, trade_date: str, timeout_seconds: int = 90) -> pd.DataFrame:
    return _fetch_baostock_payload_with_timeout(
        target=_baostock_index_constituents_worker,
        kwargs={"trade_date": str(trade_date)},
        timeout_seconds=timeout_seconds,
        timeout_label="baostock_index_constituents_timeout",
        failure_label="baostock_index_constituents",
    )


def _fetch_baostock_financial_quarterly_frame_with_timeout(
    *,
    symbols: tuple[str, ...],
    start_date: str,
    end_date: str,
    timeout_seconds: int = 600,
) -> pd.DataFrame:
    return _fetch_baostock_payload_with_timeout(
        target=_baostock_financial_quarterly_worker,
        kwargs={"symbols": tuple(symbols), "start_date": str(start_date), "end_date": str(end_date)},
        timeout_seconds=timeout_seconds,
        timeout_label="baostock_financial_quarterly_timeout",
        failure_label="baostock_financial_quarterly",
    )


def _fetch_baostock_performance_frame_with_timeout(
    *,
    domain: str,
    symbols: tuple[str, ...],
    start_date: str,
    end_date: str,
    timeout_seconds: int = 600,
) -> pd.DataFrame:
    return _fetch_baostock_payload_with_timeout(
        target=_baostock_performance_worker,
        kwargs={"domain": str(domain), "symbols": tuple(symbols), "start_date": str(start_date), "end_date": str(end_date)},
        timeout_seconds=timeout_seconds,
        timeout_label="baostock_performance_timeout",
        failure_label="baostock_performance",
    )


def _fetch_baostock_adjust_factor_frame_with_timeout(
    *,
    symbols: tuple[str, ...],
    start_date: str,
    end_date: str,
    timeout_seconds: int = 900,
) -> pd.DataFrame:
    return _fetch_baostock_payload_with_timeout(
        target=_baostock_adjust_factor_worker,
        kwargs={"symbols": tuple(symbols), "start_date": str(start_date), "end_date": str(end_date)},
        timeout_seconds=timeout_seconds,
        timeout_label="baostock_adjust_factor_timeout",
        failure_label="baostock_adjust_factor",
    )


def _fetch_baostock_valuation_frame_with_timeout(
    *,
    symbols: tuple[str, ...],
    start_date: str,
    end_date: str,
    timeout_seconds: int = 600,
) -> pd.DataFrame:
    return _fetch_baostock_payload_with_timeout(
        target=_baostock_valuation_worker,
        kwargs={"symbols": tuple(symbols), "start_date": str(start_date), "end_date": str(end_date)},
        timeout_seconds=timeout_seconds,
        timeout_label="baostock_valuation_timeout",
        failure_label="baostock_valuation",
    )


def _fetch_baostock_trade_calendar_frame_with_timeout(
    *,
    start_date: str,
    end_date: str,
    exchange: str,
    timeout_seconds: int = 60,
) -> pd.DataFrame:
    return _fetch_baostock_payload_with_timeout(
        target=_baostock_trade_calendar_worker,
        kwargs={"start_date": str(start_date), "end_date": str(end_date), "exchange": str(exchange or "SSE")},
        timeout_seconds=timeout_seconds,
        timeout_label="baostock_trade_calendar_timeout",
        failure_label="baostock_trade_calendar",
    )


def _baostock_stock_basic_frame(query: Any, *, trade_date: str) -> pd.DataFrame:
    frame = _baostock_query_to_frame(query, "baostock_stock_basic")
    if frame.empty:
        return pd.DataFrame()
    rename_map = {
        "code": "symbol",
        "code_name": "name",
        "ipoDate": "list_date",
        "outDate": "delist_date",
    }
    frame = frame.rename(columns=rename_map).copy()
    if "type" in frame.columns:
        frame = frame.loc[frame["type"].astype(str).eq("1")].copy()
    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].map(_from_baostock_code)
    if "status" in frame.columns:
        frame["list_status"] = frame["status"].map(lambda item: "L" if str(item) in {"1", "上市", "L"} else str(item))
    frame["trade_date"] = str(trade_date)
    frame["board"] = frame.get("type", "")
    if "name" in frame.columns:
        name_upper = frame["name"].fillna("").astype(str).str.upper()
        frame["is_st"] = name_upper.str.startswith(("ST", "*ST"))
    else:
        frame["is_st"] = False
    frame["is_suspended"] = False
    frame["is_delisted"] = frame.get("list_status", "").astype(str).str.upper().isin({"D", "DELIST", "0", "退市"})
    frame["status_reason"] = frame.get("status", "").astype(str)
    return frame


def _date_range_strings(start_date: str, end_date: str) -> list[str]:
    return [pd.Timestamp(item).strftime("%Y-%m-%d") for item in pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")]


def _akshare_industry_members(ak: Any, request: DomainFetchRequest) -> pd.DataFrame:
    boards = ak.stock_board_industry_name_em()
    if not isinstance(boards, pd.DataFrame) or boards.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    name_col = "板块名称" if "板块名称" in boards.columns else "名称" if "名称" in boards.columns else boards.columns[0]
    for industry in boards[name_col].dropna().astype(str).head(200):
        try:
            members = ak.stock_board_industry_cons_em(symbol=industry)
        except Exception:
            continue
        if not isinstance(members, pd.DataFrame) or members.empty:
            continue
        code_col = "代码" if "代码" in members.columns else "symbol" if "symbol" in members.columns else members.columns[0]
        for code in members[code_col].dropna().astype(str):
            rows.append({"symbol": code, "trade_date": request.end_date, "industry": industry, "concept_tags": ""})
    return pd.DataFrame(rows)


def _akshare_limit_status(ak: Any, request: DomainFetchRequest) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for trade_date in _date_range_strings(request.start_date, request.end_date):
        try:
            up = ak.stock_zt_pool_em(date=trade_date.replace("-", ""))
        except Exception:
            up = pd.DataFrame()
        if isinstance(up, pd.DataFrame) and not up.empty:
            up = up.copy()
            up["trade_date"] = trade_date
            up["is_limit_up"] = True
            frames.append(up.rename(columns={"代码": "symbol", "最新价": "up_limit"}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _akshare_money_flow(ak: Any, request: DomainFetchRequest) -> pd.DataFrame:
    try:
        frame = ak.stock_sector_fund_flow_rank(indicator="今日")
    except Exception:
        frame = pd.DataFrame()
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        frame = frame.copy()
        frame["trade_date"] = request.end_date
        frame = frame.rename(columns={"名称": "hotspot_tags", "今日主力净流入-净额": "main_net_inflow"})
        frame["sector_rank"] = range(1, len(frame) + 1)
    return frame


def _ths_concept_frame(ak: Any, request: DomainFetchRequest) -> pd.DataFrame:
    boards = ak.stock_board_concept_name_ths()
    if not isinstance(boards, pd.DataFrame) or boards.empty:
        return pd.DataFrame()
    name_col = "概念名称" if "概念名称" in boards.columns else "名称" if "名称" in boards.columns else boards.columns[0]
    rows = [
        {
            "symbol": "HOTSPOT",
            "trade_date": request.end_date,
            "industry": "",
            "concept_tags": str(name),
            "source": "tonghuashun_hotspot",
        }
        for name in boards[name_col].dropna().astype(str).head(300)
    ]
    return pd.DataFrame(rows)


def _ths_hotspot_frame(ak: Any, request: DomainFetchRequest) -> pd.DataFrame:
    boards = ak.stock_board_concept_name_ths()
    if not isinstance(boards, pd.DataFrame) or boards.empty:
        return pd.DataFrame()
    name_col = "概念名称" if "概念名称" in boards.columns else "名称" if "名称" in boards.columns else boards.columns[0]
    rows: list[dict[str, Any]] = []
    for rank, name in enumerate(boards[name_col].dropna().astype(str).head(100), start=1):
        rows.append(
            {
                "symbol": "HOTSPOT",
                "trade_date": request.end_date,
                "main_net_inflow": float("nan"),
                "sector_rank": rank,
                "hotspot_tags": str(name),
                "source": "tonghuashun_hotspot",
            }
        )
    return pd.DataFrame(rows)
