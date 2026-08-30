"""Mootdx provider responsibilities."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quantlab.data.domains.contracts.base import validate_provider_name
from quantlab.data.domains.contracts.coverage import coverage_report_for_domain, coverage_report_for_frame
from quantlab.data.domains.contracts.dispatch import normalize_domain_frame
from quantlab.data.domains.contracts.intraday import build_intraday_daily_feature_frame
from quantlab.data.domains.contracts.requests import DomainFetchRequest, FetchRequest, ProviderResult
from quantlab.data.domains.contracts.schema import DataDomain
from quantlab.data.progress import progress_write
from quantlab.data.provider_symbols import (
    mootdx_symbol as _mootdx_symbol,
)

from .frames import (
    _fetch_mootdx_bars_window,
    _filter_domain_date_window,
    _mootdx_xdxr_domain_frame,
    _normalize_mootdx_quote_snapshot,
)
from .transport import (
    _close_mootdx_client,
    _open_mootdx_client,
)


@dataclass
class MootdxOnlineProvider:
    name: str = "mootdx_online"
    page_size: int = 800
    max_pages: int = 12
    _client_factory: Any = None
    _reuse_client: bool = True
    _client: Any = field(default=None, init=False, repr=False)
    _client_guard: Any = field(default_factory=threading.RLock, init=False, repr=False)
    _failed_servers: set[tuple[str, int]] = field(default_factory=set, init=False, repr=False)
    _circuit_open_until: float = field(default=0.0, init=False, repr=False)
    _circuit_reason: str = field(default="", init=False, repr=False)
    _circuit_cooldown_seconds: float = 300.0
    supports_backward_range_prefetch: bool = field(default=True, init=False)

    def _reset_client_locked(self, *, mark_failed: bool = False) -> None:
        client = self._client
        self._client = None
        if client is None:
            return
        if mark_failed and self._client_factory is None:
            server = getattr(client, "server", None)
            try:
                normalized = (str(server[0]), int(server[1]))
            except Exception:
                normalized = None
            if normalized is not None:
                self._failed_servers.add(normalized)
        _close_mootdx_client(client)

    def _get_client_locked(self) -> Any:
        if time.monotonic() < self._circuit_open_until:
            raise RuntimeError(f"mootdx_circuit_open:{self._circuit_reason}")
        if self._circuit_open_until:
            self._circuit_open_until = 0.0
            self._circuit_reason = ""
            self._failed_servers.clear()
        if self._client is None:
            self._client = _open_mootdx_client(
                self._client_factory,
                excluded_servers=tuple(sorted(self._failed_servers)),
            )
        return self._client

    def _trip_circuit_locked(self, exc: BaseException) -> None:
        self._circuit_reason = f"{type(exc).__name__}: {exc}"
        self._circuit_open_until = time.monotonic() + max(1.0, float(self._circuit_cooldown_seconds))
        self._reset_client_locked(mark_failed=True)

    def _record_client_success_locked(self) -> None:
        self._circuit_open_until = 0.0
        self._circuit_reason = ""

    def close(self) -> None:
        with self._client_guard:
            self._reset_client_locked()
            self._failed_servers.clear()
            self._circuit_open_until = 0.0
            self._circuit_reason = ""

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
            return ProviderResult(
                provider=self.name, data=data, coverage_report=coverage, error_report=raw_result.error_report
            )
        raise RuntimeError(f"unsupported_domain: {self.name} does not support {request.domain}")

    def fetch_xdxr_raw(self, symbols: Any) -> ProviderResult:
        """Fetch raw TDX ex-right and share-capital records without semantic loss."""

        normalized_symbols = tuple(
            FetchRequest(symbols=tuple(symbols or ()), start_date="2000-01-01", end_date="2000-01-01")
            .normalized()
            .symbols
        )
        rows: list[pd.DataFrame] = []
        errors: list[dict[str, Any]] = []
        if not normalized_symbols:
            return ProviderResult(
                provider=self.name,
                data=pd.DataFrame(),
                coverage_report={"endpoint": "xdxr", "row_count": 0},
                error_report=[],
            )
        with self._client_guard:
            for symbol in normalized_symbols:
                if time.monotonic() < self._circuit_open_until:
                    errors.append(
                        {
                            "provider": self.name,
                            "domain": "xdxr_raw",
                            "symbol": symbol,
                            "code": "provider_circuit_open",
                            "error_type": "RuntimeError",
                            "message": self._circuit_reason,
                            "attempts": 0,
                        }
                    )
                    continue
                last_error: Exception | None = None
                frame = pd.DataFrame()
                for attempt in range(1, 4):
                    try:
                        client = self._get_client_locked()
                        payload = client.xdxr(symbol=_mootdx_symbol(symbol))
                        frame = payload.copy() if isinstance(payload, pd.DataFrame) else pd.DataFrame(payload or [])
                        self._record_client_success_locked()
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                        self._reset_client_locked(mark_failed=True)
                        if attempt < 3:
                            time.sleep(float((0, 2, 5)[attempt]))
                if last_error is not None:
                    self._trip_circuit_locked(last_error)
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
            if not self._reuse_client:
                self._reset_client_locked()
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
        normalized_symbols = tuple(
            FetchRequest(symbols=tuple(symbols), start_date="2000-01-01", end_date="2000-01-01").normalized().symbols
        )
        rows: list[pd.DataFrame] = []
        errors: list[dict[str, Any]] = []
        if not normalized_symbols:
            return ProviderResult(provider=self.name, data=pd.DataFrame(), error_report=errors)
        with self._client_guard:
            try:
                client = self._get_client_locked()
                payload = client.quotes(symbol=[_mootdx_symbol(symbol) for symbol in normalized_symbols])
                self._record_client_success_locked()
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
                self._reset_client_locked(mark_failed=True)
                errors.append(
                    {
                        "provider": self.name,
                        "domain": "quote_snapshot",
                        "code": "provider_exception",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
            finally:
                if not self._reuse_client:
                    self._reset_client_locked()
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

    def _empty_bars_result(
        self,
        request: DomainFetchRequest,
        *,
        errors: list[dict[str, Any]],
    ) -> ProviderResult:
        data = normalize_domain_frame(
            pd.DataFrame(),
            domain=request.domain,
            source=self.name,
            as_of_date=request.end_date,
            adjusted_flag=request.adjusted_flag,
            require_columns=False,
        )
        return ProviderResult(provider=self.name, data=data, error_report=errors)

    def _fetch_symbol_bars_locked(
        self,
        request: DomainFetchRequest,
        *,
        symbol: str,
        frequency: int,
        volume_factor: float,
    ) -> tuple[pd.DataFrame, dict[str, Any] | None]:
        if time.monotonic() < self._circuit_open_until:
            return pd.DataFrame(), {
                "provider": self.name,
                "domain": request.domain,
                "symbol": symbol,
                "code": "provider_circuit_open",
                "error_type": "RuntimeError",
                "message": self._circuit_reason,
                "attempts": 0,
            }
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                client = self._get_client_locked()
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
                self._record_client_success_locked()
                return frame, None
            except Exception as exc:
                last_error = exc
                self._reset_client_locked(mark_failed=True)
                time.sleep(min(3.0, 0.5 * attempt))
        assert last_error is not None
        self._trip_circuit_locked(last_error)
        return pd.DataFrame(), {
            "provider": self.name,
            "domain": request.domain,
            "symbol": symbol,
            "code": "symbol_fetch_error",
            "error_type": type(last_error).__name__,
            "message": str(last_error),
            "attempts": 3,
        }

    def _bars_coverage(
        self,
        request: DomainFetchRequest,
        data: pd.DataFrame,
        *,
        frequency: int,
        endpoint: str,
        volume_factor: float,
        raw_volume_unit: str,
        canonical_volume_unit: str,
    ) -> dict[str, Any]:
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
                "source_stability_note": (
                    "online mootdx quote server; server stability must be monitored by provider-health/provider-eval"
                ),
            }
        )
        if request.domain == DataDomain.MARKET_INTRADAY_5M:
            coverage["bar_count_contract"] = "mootdx_5m_48_full_trading_day"
        return coverage

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
            return self._empty_bars_result(request, errors=errors)
        with self._client_guard:
            try:
                self._get_client_locked()
            except Exception as exc:
                return self._empty_bars_result(
                    request,
                    errors=[
                        {
                            "provider": self.name,
                            "domain": request.domain,
                            "code": "client_open_error",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    ],
                )
            for idx, symbol in enumerate(symbols, start=1):
                if idx == 1 or idx % 50 == 0 or idx == len(symbols):
                    progress_write(f"mootdx_online_{request.domain}={idx}/{len(symbols)} symbol={symbol}")
                frame, error = self._fetch_symbol_bars_locked(
                    request,
                    symbol=symbol,
                    frequency=frequency,
                    volume_factor=volume_factor,
                )
                if error is not None:
                    errors.append(error)
                if not frame.empty:
                    rows.append(frame)
            if not self._reuse_client:
                self._reset_client_locked()
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
        coverage = self._bars_coverage(
            request,
            data,
            frequency=frequency,
            endpoint=endpoint,
            volume_factor=volume_factor,
            raw_volume_unit=raw_volume_unit,
            canonical_volume_unit=canonical_volume_unit,
        )
        return ProviderResult(
            provider=self.name,
            data=data,
            coverage_report=coverage,
            error_report=errors,
        )
