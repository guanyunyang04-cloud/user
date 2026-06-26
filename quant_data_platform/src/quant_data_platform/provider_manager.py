from __future__ import annotations

from typing import Callable, Iterable

import pandas as pd

from quant_data_platform.domains.contracts import (
    DataDomain,
    DomainFetchRequest,
    FetchRequest,
    MarketProvider,
    ProviderResult,
    coverage_report_for_domain,
    coverage_report_for_frame,
    normalize_domain_frame,
    normalize_market_frame,
    validate_provider_name,
)


class UnsupportedDomainError(RuntimeError):
    pass


class InMemoryMarketProvider:
    def __init__(self, name: str, payload: pd.DataFrame | Callable[[FetchRequest], ProviderResult]) -> None:
        self.name = validate_provider_name(name)
        self._payload = payload
        self.requests: list[FetchRequest] = []

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        request = request.normalized()
        self.requests.append(request)
        if callable(self._payload):
            result = self._payload(request)
            data = normalize_market_frame(result.data, source=result.provider or self.name, adjusted_flag=request.adjusted_flag)
            return ProviderResult(
                provider=self.name,
                data=data,
                coverage_report=result.coverage_report or coverage_report_for_frame(data, request, provider=self.name),
                error_report=list(result.error_report or []),
            )
        data = normalize_market_frame(self._payload, source=self.name, adjusted_flag=request.adjusted_flag)
        if not data.empty:
            start_ts = pd.Timestamp(request.start_date)
            end_ts = pd.Timestamp(request.end_date)
            dates = pd.to_datetime(data["trade_date"])
            data = data.loc[(dates >= start_ts) & (dates <= end_ts)]
            data = data.loc[data["symbol"].isin(set(request.symbols))]
        report = coverage_report_for_frame(data, request, provider=self.name)
        return ProviderResult(provider=self.name, data=data.reset_index(drop=True), coverage_report=report, error_report=[])

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        request = request.normalized()
        if request.domain != DataDomain.MARKET_DAILY:
            raise UnsupportedDomainError(f"{self.name} does not support domain={request.domain}")
        return self.fetch_market_bars(
            FetchRequest(
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
                domain=DataDomain.MARKET_DAILY,
                adjusted_flag=request.adjusted_flag,
            )
        )


class InMemoryDomainProvider:
    def __init__(self, name: str, payloads: dict[str, pd.DataFrame | Callable[[DomainFetchRequest], ProviderResult]]) -> None:
        self.name = validate_provider_name(name)
        self.payloads = dict(payloads)
        self.domain_requests: list[DomainFetchRequest] = []

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        request = request.normalized()
        self.domain_requests.append(request)
        if request.domain not in self.payloads:
            raise UnsupportedDomainError(f"{self.name} does not support domain={request.domain}")
        payload = self.payloads[request.domain]
        if callable(payload):
            result = payload(request)
            data = normalize_domain_frame(
                result.data,
                domain=request.domain,
                source=result.provider or self.name,
                as_of_date=request.end_date,
                adjusted_flag=request.adjusted_flag,
                require_columns=False,
            )
            return ProviderResult(
                provider=self.name,
                data=data,
                coverage_report=result.coverage_report or coverage_report_for_domain(data, request, provider=self.name),
                error_report=list(result.error_report or []),
            )
        data = normalize_domain_frame(
            payload,
            domain=request.domain,
            source=self.name,
            as_of_date=request.end_date,
            adjusted_flag=request.adjusted_flag,
            require_columns=False,
        )
        if not data.empty and "trade_date" in data.columns:
            start_ts = pd.Timestamp(request.start_date)
            end_ts = pd.Timestamp(request.end_date)
            dates = pd.to_datetime(data["trade_date"], errors="coerce")
            data = data.loc[(dates >= start_ts) & (dates <= end_ts)]
        if not data.empty and request.symbols and "symbol" in data.columns:
            data = data.loc[data["symbol"].isin(set(request.symbols))]
        report = coverage_report_for_domain(data, request, provider=self.name)
        return ProviderResult(provider=self.name, data=data.reset_index(drop=True), coverage_report=report, error_report=[])

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        request = request.normalized()
        return self.fetch_domain(
            DomainFetchRequest(
                domain=DataDomain.MARKET_DAILY,
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
                adjusted_flag=request.adjusted_flag,
            )
        )


class ProviderManager:
    def __init__(
        self,
        providers: Iterable[MarketProvider],
        *,
        allow_tdx_family: bool = False,
    ) -> None:
        self.providers = list(providers)
        self.allow_tdx_family = bool(allow_tdx_family)
        for provider in self.providers:
            validate_provider_name(getattr(provider, "name", ""), allow_tdx_family=self.allow_tdx_family)

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        request = request.normalized()
        return self.fetch_domain(
            DomainFetchRequest(
                domain=DataDomain.MARKET_DAILY,
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
                adjusted_flag=request.adjusted_flag,
            )
        )

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        request = request.normalized()
        frames: list[pd.DataFrame] = []
        errors: list[dict] = []
        provider_reports: list[dict] = []
        for provider in self.providers:
            provider_name = validate_provider_name(getattr(provider, "name", ""), allow_tdx_family=self.allow_tdx_family)
            try:
                result = self._call_provider_domain(provider, request)
                data = normalize_domain_frame(
                    result.data,
                    domain=request.domain,
                    source=provider_name,
                    adjusted_flag=request.adjusted_flag,
                    as_of_date=request.end_date,
                    require_columns=False,
                )
                report = result.coverage_report or coverage_report_for_domain(data, request, provider=provider_name)
                provider_reports.append(report)
                errors.extend(list(result.error_report or []))
                if data.empty:
                    errors.append({"provider": provider_name, "domain": request.domain, "code": "empty_return", "message": "provider returned no domain rows"})
                else:
                    frames.append(data)
            except UnsupportedDomainError as exc:
                errors.append(
                    {
                        "provider": provider_name,
                        "domain": request.domain,
                        "code": "unsupported_domain",
                        "message": str(exc),
                    }
                )
                provider_reports.append(
                    {
                        "provider": provider_name,
                        "domain": request.domain,
                        "status": "unsupported",
                        "expected_rows": int(len(request.symbols)),
                        "row_count": 0,
                        "coverage_ratio": 0.0,
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "provider": provider_name,
                        "domain": request.domain,
                        "code": "provider_exception",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                provider_reports.append(
                    {
                        "provider": provider_name,
                        "domain": request.domain,
                        "status": "error",
                        "expected_rows": int(len(request.symbols)),
                        "row_count": 0,
                        "coverage_ratio": 0.0,
                    }
                )
        data = pd.concat(frames, ignore_index=True) if frames else normalize_domain_frame(
            pd.DataFrame(),
            domain=request.domain,
            source="unknown",
            as_of_date=request.end_date,
            require_columns=False,
        )
        row_count = int(len(data))
        expected_rows = int(max((item.get("expected_rows", 0) for item in provider_reports), default=0))
        successful_provider_count = sum(1 for item in provider_reports if int(item.get("row_count", 0) or 0) > 0)
        coverage_ratio = float(row_count / expected_rows) if expected_rows else 0.0
        status = "ok" if row_count > 0 else "no_data"
        return ProviderResult(
            provider="provider_manager",
            data=data,
            coverage_report={
                "status": status,
                "domain": request.domain,
                "provider_count": int(len(self.providers)),
                "successful_provider_count": int(successful_provider_count),
                "row_count": row_count,
                "expected_rows": expected_rows,
                "coverage_ratio": coverage_ratio,
                "provider_reports": provider_reports,
            },
            error_report=errors,
        )

    def _call_provider_domain(self, provider: MarketProvider, request: DomainFetchRequest) -> ProviderResult:
        return self._invoke_provider_domain(provider, request)

    def _invoke_provider_domain(self, provider: MarketProvider, request: DomainFetchRequest) -> ProviderResult:
        fetch_domain = getattr(provider, "fetch_domain", None)
        if callable(fetch_domain):
            return fetch_domain(request)
        if request.domain == DataDomain.MARKET_DAILY:
            return provider.fetch_market_bars(
                FetchRequest(
                    symbols=request.symbols,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    domain=DataDomain.MARKET_DAILY,
                    adjusted_flag=request.adjusted_flag,
                )
            )
        raise UnsupportedDomainError(f"{getattr(provider, 'name', 'provider')} does not support domain={request.domain}")

    def _call_provider(self, provider: MarketProvider, request: FetchRequest) -> ProviderResult:
        return provider.fetch_market_bars(request)
