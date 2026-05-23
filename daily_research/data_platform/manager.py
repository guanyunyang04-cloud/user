from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Callable, Iterable

import pandas as pd

from daily_research.data_platform.contracts import (
    FetchRequest,
    MarketProvider,
    ProviderResult,
    coverage_report_for_frame,
    normalize_market_frame,
    validate_provider_name,
)


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


class ProviderManager:
    def __init__(
        self,
        providers: Iterable[MarketProvider],
        *,
        timeout_seconds: float = 30.0,
        allow_tdx_family: bool = False,
    ) -> None:
        self.providers = list(providers)
        self.timeout_seconds = float(timeout_seconds or 0.0)
        self.allow_tdx_family = bool(allow_tdx_family)
        for provider in self.providers:
            validate_provider_name(getattr(provider, "name", ""), allow_tdx_family=self.allow_tdx_family)

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        request = request.normalized()
        frames: list[pd.DataFrame] = []
        errors: list[dict] = []
        provider_reports: list[dict] = []
        for provider in self.providers:
            provider_name = validate_provider_name(getattr(provider, "name", ""), allow_tdx_family=self.allow_tdx_family)
            try:
                result = self._call_provider(provider, request)
                data = normalize_market_frame(result.data, source=provider_name, adjusted_flag=request.adjusted_flag)
                report = result.coverage_report or coverage_report_for_frame(data, request, provider=provider_name)
                provider_reports.append(report)
                errors.extend(list(result.error_report or []))
                if data.empty:
                    errors.append({"provider": provider_name, "code": "empty_return", "message": "provider returned no market rows"})
                else:
                    frames.append(data)
            except Exception as exc:
                errors.append(
                    {
                        "provider": provider_name,
                        "code": "provider_exception",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                provider_reports.append(
                    {
                        "provider": provider_name,
                        "status": "error",
                        "expected_rows": int(len(request.symbols)),
                        "row_count": 0,
                        "coverage_ratio": 0.0,
                    }
                )
        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=normalize_market_frame(pd.DataFrame(), source="unknown").columns)
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
                "provider_count": int(len(self.providers)),
                "successful_provider_count": int(successful_provider_count),
                "row_count": row_count,
                "expected_rows": expected_rows,
                "coverage_ratio": coverage_ratio,
                "provider_reports": provider_reports,
            },
            error_report=errors,
        )

    def _call_provider(self, provider: MarketProvider, request: FetchRequest) -> ProviderResult:
        if self.timeout_seconds <= 0:
            return provider.fetch_market_bars(request)
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(provider.fetch_market_bars, request)
        try:
            return future.result(timeout=self.timeout_seconds)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"provider timed out after {self.timeout_seconds:g}s") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
