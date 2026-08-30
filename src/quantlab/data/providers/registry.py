"""Provider plans, capability metadata and routing providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quantlab.data.domains.contracts.requests import DomainFetchRequest, FetchRequest, ProviderResult
from quantlab.data.domains.contracts.schema import DataDomain

from .baostock import BaostockProvider
from .cninfo import CninfoAnnouncementProvider
from .mootdx import MootdxOnlineProvider
from .web import (
    AkshareEastmoneyProvider,
    EastmoneyEfinanceProvider,
    SinaTencentRealtimeProvider,
    TencentFinanceProvider,
    TonghuashunHotspotProvider,
)

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


QDP_CURRENT_REQUIRED_DOMAINS: tuple[str, ...] = (
    DataDomain.MARKET_DAILY,
    DataDomain.TRADING_CALENDAR,
    DataDomain.UNIVERSE_SNAPSHOT,
    DataDomain.SECURITY_STATUS,
    DataDomain.ADJUST_FACTOR,
    DataDomain.MARKET_INTRADAY_5M,
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


QDP_PRODUCTION_V1_RESEARCH_FUTURE_DOMAINS: tuple[str, ...] = (DataDomain.ANNOUNCEMENT,)


_PROVIDER_CAPABILITIES: dict[str, dict[str, Any]] = {
    "mootdx_online": {
        "domains": (
            DataDomain.MARKET_DAILY,
            DataDomain.MARKET_INTRADAY_5M,
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
        "domains": (DataDomain.ANNOUNCEMENT,),
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


def _formal_requirement(domain: str) -> str:
    if domain in FORMAL_FREE_V3_REQUIRED_DOMAINS:
        return "required"
    if domain in FORMAL_FREE_V3_OPTIONAL_DOMAINS:
        return "optional"
    if domain in FORMAL_FREE_V3_RESEARCH_FUTURE_DOMAINS:
        return "research_future"
    return "unsupported"


def _capability_plan(plan: str) -> tuple[tuple[str, ...], dict[str, set[str]]]:
    if plan == "qdp_current":
        return ("baostock", "mootdx_online"), {
            "baostock": set(QDP_CURRENT_REQUIRED_DOMAINS),
            "mootdx_online": {DataDomain.MARKET_INTRADAY_5M},
        }
    if plan == "formal_free_v3":
        return (
            "baostock",
            "eastmoney_efinance",
            "akshare_eastmoney",
            "tencent_finance",
            "tonghuashun_hotspot",
        ), {}
    if plan == "qdp_production_v1":
        return ("mootdx_online", "baostock", "cninfo"), {
            "mootdx_online": {
                DataDomain.MARKET_DAILY,
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
    if plan == "research_rebuild_minimal_free":
        return ("research_rebuild_minimal_free",), {}
    names = tuple(str(getattr(provider, "name", "")) for provider in build_default_providers(plan))
    return names, {}


def _plan_requirement(plan: str, domain: str) -> str:
    if plan != "qdp_production_v1":
        return _formal_requirement(domain)
    if domain in QDP_PRODUCTION_V1_REQUIRED_DOMAINS:
        return "required"
    if domain in QDP_PRODUCTION_V1_OPTIONAL_DOMAINS:
        return "optional"
    if domain in QDP_PRODUCTION_V1_RESEARCH_FUTURE_DOMAINS:
        return "research_future"
    return "unsupported"


def _is_formal_refresh(
    *,
    plan: str,
    provider_name: str,
    domain: str,
    supported: set[str],
    meta: dict[str, Any],
    defaults: dict[str, set[str]],
) -> bool:
    if plan == "qdp_current":
        return domain in defaults.get(provider_name, supported)
    if plan == "formal_free_v3":
        return bool(meta.get("formal_eligible", False)) and _formal_requirement(domain) in {"required", "optional"}
    if plan == "research_rebuild_minimal_free":
        return (
            bool(meta.get("formal_eligible", False))
            and domain in RESEARCH_REBUILD_MINIMAL_REQUIRED_DOMAINS
            and domain in supported
        )
    if plan == "qdp_production_v1":
        return domain in defaults.get(provider_name, set()) and domain in supported
    return False


def provider_capability_matrix(provider_plan: str = "formal_free_v3") -> list[dict[str, Any]]:
    plan = str(provider_plan or "formal_free_v3").strip().lower()
    provider_names, default_domains_by_provider = _capability_plan(plan)
    rows: list[dict[str, Any]] = []
    all_domains = (
        *QDP_CURRENT_REQUIRED_DOMAINS,
        *FORMAL_FREE_V3_REQUIRED_DOMAINS,
        *FORMAL_FREE_V3_OPTIONAL_DOMAINS,
        *QDP_PRODUCTION_V1_OPTIONAL_DOMAINS,
        *FORMAL_FREE_V3_RESEARCH_FUTURE_DOMAINS,
        *QDP_PRODUCTION_V1_RESEARCH_FUTURE_DOMAINS,
    )
    for provider_name in provider_names:
        meta = _PROVIDER_CAPABILITIES.get(
            provider_name, {"domains": (), "requires_token": False, "formal_eligible": False, "notes": ""}
        )
        supported = set(str(item) for item in meta.get("domains", ()))
        for domain in tuple(dict.fromkeys(all_domains)):
            formal_refresh = _is_formal_refresh(
                plan=plan,
                provider_name=provider_name,
                domain=domain,
                supported=supported,
                meta=meta,
                defaults=default_domains_by_provider,
            )
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
                    "requirement": _plan_requirement(plan, domain),
                    "notes": str(meta.get("notes", "")),
                }
            )
    return rows


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
class QdpProductionV1Provider:
    name: str = "qdp_production_v1"

    def __post_init__(self) -> None:
        self._mootdx = MootdxOnlineProvider()
        self._baostock = BaostockProvider()
        self._cninfo = CninfoAnnouncementProvider()

    def close(self) -> None:
        self._mootdx.close()
        self._baostock.close()

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        return self._mootdx.fetch_market_bars(request)

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        request = request.normalized()
        if request.domain in {
            DataDomain.MARKET_DAILY,
            DataDomain.MARKET_INTRADAY_5M,
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


def build_default_providers(provider_plan: str = "default_free") -> list:
    plan = str(provider_plan or "default_free").strip().lower()
    if plan == "qdp_current":
        return [BaostockProvider(), MootdxOnlineProvider()]
    if plan == "qdp_production_v1":
        return [QdpProductionV1Provider()]
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
        return [
            EastmoneyEfinanceProvider(),
            AkshareEastmoneyProvider(),
            BaostockProvider(),
            SinaTencentRealtimeProvider(),
        ]
    raise ValueError(f"Unsupported provider_plan: {provider_plan}")
