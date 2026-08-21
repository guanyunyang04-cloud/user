"""Baostock provider responsibilities."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quantlab.data.domains.contracts import (
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
from quantlab.data.progress import create_progress, progress_write

from .frames import (
    _baostock_bulk_adjust_factor_event_frame,
    _baostock_bulk_daily_domain_frame,
    _baostock_symbol_error,
)
from .runtime import (
    _BAOSTOCK_GLOBAL_LIMITER,
    BAOSTOCK_BATCH_VERSION,
    assert_baostock_batch_runtime,
)
from .transport import (
    _BaostockPersistentSession,
    _fetch_baostock_adjust_factor_frame_with_timeout,
    _fetch_baostock_all_stock_frame_with_timeout,
    _fetch_baostock_bulk_partition_with_retry,
    _fetch_baostock_financial_quarterly_frame_with_timeout,
    _fetch_baostock_history_frame_with_timeout,
    _fetch_baostock_index_constituents_frame_with_timeout,
    _fetch_baostock_industry_frame_with_timeout,
    _fetch_baostock_intraday_5m_frame_with_timeout,
    _fetch_baostock_performance_frame_with_timeout,
    _fetch_baostock_stock_basic_frame_with_timeout,
    _fetch_baostock_trade_calendar_frame_with_timeout,
    _fetch_baostock_valuation_frame_with_timeout,
)


@dataclass
class BaostockProvider:
    name: str = "baostock"
    _market_daily_max_workers: int = 4
    _intraday_max_workers: int = 1
    _bulk_timeout_seconds: int = 120
    _bulk_retry_backoff_seconds: tuple[int, ...] = (2, 5, 15)
    _reuse_date_partition_session: bool = True
    _reuse_symbol_range_session: bool = False
    supports_symbol_range_prefetch: bool = field(default=True, init=False)
    _date_partition_session: Any = field(default=None, init=False, repr=False)
    _date_partition_session_guard: Any = field(default_factory=threading.Lock, init=False, repr=False)

    def _persistent_date_request(
        self, payload: dict[str, Any], *, timeout_seconds: int
    ) -> tuple[dict[str, Any], int, list[str]]:
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

    def close(self) -> None:
        """Close the shared BaoStock login used by every provider endpoint."""

        self.close_date_partition_session()

    def _persistent_frame_request(
        self,
        payload: dict[str, Any],
        *,
        timeout_seconds: int,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
        response, attempt_count, retry_errors = self._persistent_date_request(
            payload,
            timeout_seconds=timeout_seconds,
        )
        session_retry_errors = list(retry_errors)
        frame = response.get("data")
        if not isinstance(frame, pd.DataFrame):
            raise RuntimeError(
                f"baostock_persistent_frame_invalid_data:kind={payload.get('kind', '')}:{type(frame).__name__}"
            )
        error_report = [dict(item) for item in list(response.get("error_report", []) or []) if isinstance(item, dict)]
        meta = dict(response.get("meta", {}) or {})
        retryable_kinds = {
            "history_symbols",
            "security_lifecycle_history_symbols",
            "intraday_5m_symbols",
            "financial_quarterly_symbols",
            "performance_symbols",
            "valuation_symbols",
            "adjust_factor_symbols",
        }
        failed_symbols = tuple(
            dict.fromkeys(str(item.get("symbol", "")) for item in error_report if str(item.get("symbol", "")))
        )
        symbol_retry_count = 0
        if failed_symbols and str(payload.get("kind", "")) in retryable_kinds:
            retry_payload = {**payload, "symbols": failed_symbols}
            symbol_retry_count = len(failed_symbols)
            self.close_date_partition_session()
            try:
                retry_response, retry_attempt_count, retry_session_errors = self._persistent_date_request(
                    retry_payload,
                    timeout_seconds=timeout_seconds,
                )
                retry_frame = retry_response.get("data")
                if not isinstance(retry_frame, pd.DataFrame):
                    raise RuntimeError(
                        "baostock_persistent_retry_invalid_data:"
                        f"kind={payload.get('kind', '')}:{type(retry_frame).__name__}"
                    )
                if not retry_frame.empty:
                    frame = pd.concat([frame, retry_frame], ignore_index=True, sort=False)
                error_report = [
                    dict(item) for item in list(retry_response.get("error_report", []) or []) if isinstance(item, dict)
                ]
                attempt_count += retry_attempt_count
                session_retry_errors.extend(retry_session_errors)
            except Exception as exc:
                for item in error_report:
                    item["retry_error"] = f"{type(exc).__name__}: {exc}"
        meta.update(
            {
                "attempt_count": attempt_count,
                "retry_errors": session_retry_errors,
                "session_reused": True,
                "symbol_retry_count": symbol_retry_count,
            }
        )
        return frame.copy(), error_report, meta

    def _persistent_market_bars(self, request: FetchRequest) -> ProviderResult:
        symbols = tuple(request.symbols)
        frame, errors, meta = self._persistent_frame_request(
            {
                "kind": "history_symbols",
                "symbols": symbols,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "adjusted_flag": request.adjusted_flag,
            },
            timeout_seconds=max(90, 60 * len(symbols)),
        )
        data = normalize_market_frame(
            frame,
            source=self.name,
            adjusted_flag=request.adjusted_flag,
            require_columns=False,
        )
        coverage = coverage_report_for_frame(data, request, provider=self.name)
        coverage.update(meta)
        coverage["download_strategy"] = "single_login_sequential_symbols"
        return ProviderResult(provider=self.name, data=data, coverage_report=coverage, error_report=errors)

    def _fetch_market_symbol_frames(
        self,
        request: FetchRequest,
    ) -> tuple[list[pd.DataFrame], list[tuple[str, BaseException]]]:
        rows: list[pd.DataFrame] = []
        failed_symbols: list[tuple[str, BaseException]] = []
        symbols = tuple(request.symbols)
        max_workers = min(self._market_daily_max_workers, len(symbols))
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
        return rows, failed_symbols

    def _retry_market_symbols(
        self,
        request: FetchRequest,
        failed_symbols: list[tuple[str, BaseException]],
    ) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
        rows: list[pd.DataFrame] = []
        errors: list[dict[str, Any]] = []
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
        return rows, errors

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        validate_provider_name(self.name)
        request = request.normalized()
        symbols = tuple(request.symbols)
        if not symbols:
            data = normalize_market_frame(
                pd.DataFrame(), source=self.name, adjusted_flag=request.adjusted_flag, require_columns=False
            )
            return ProviderResult(provider=self.name, data=data, error_report=[])
        if self._reuse_symbol_range_session:
            return self._persistent_market_bars(request)
        rows, failed_symbols = self._fetch_market_symbol_frames(request)
        retry_rows, errors = self._retry_market_symbols(request, failed_symbols)
        rows.extend(retry_rows)
        data = normalize_market_frame(
            pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
            source=self.name,
            adjusted_flag=request.adjusted_flag,
            require_columns=False,
        )
        return ProviderResult(provider=self.name, data=data, error_report=errors)

    def _fetch_intraday_5m_frames(
        self, request: DomainFetchRequest, *, progress_label: str
    ) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
        rows: list[pd.DataFrame] = []
        errors: list[dict[str, Any]] = []
        failed_symbols: list[tuple[str, BaseException]] = []
        symbols = tuple(request.symbols)
        max_workers = min(max(1, int(self._intraday_max_workers or 1)), len(symbols)) if symbols else 0
        if not symbols:
            return rows, errors
        if self._reuse_symbol_range_session:
            frame, persistent_errors, _meta = self._persistent_frame_request(
                {
                    "kind": "intraday_5m_symbols",
                    "symbols": symbols,
                    "start_date": request.start_date,
                    "end_date": request.end_date,
                    "adjusted_flag": request.adjusted_flag,
                },
                timeout_seconds=max(180, 90 * len(symbols)),
            )
            if not frame.empty and "symbol" in frame.columns:
                rows.extend(group.reset_index(drop=True) for _, group in frame.groupby("symbol", sort=False))
            elif not frame.empty:
                rows.append(frame)
            return rows, persistent_errors
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

    def fetch_stock_basic_snapshot(self, *, trade_date: str) -> pd.DataFrame:
        """Return BaoStock's current stock master under the shared login."""

        if self._reuse_date_partition_session:
            frame, errors, _meta = self._persistent_frame_request(
                {"kind": "stock_basic", "trade_date": str(trade_date)},
                timeout_seconds=120,
            )
            if errors:
                raise RuntimeError(f"baostock_stock_basic_errors:{len(errors)}")
            return frame
        return _fetch_baostock_stock_basic_frame_with_timeout(
            trade_date=str(trade_date),
            timeout_seconds=120,
        )

    def fetch_security_lifecycle_history(
        self,
        *,
        symbols: Sequence[str],
        start_date: str,
        end_date: str,
    ) -> ProviderResult:
        """Fetch raw daily prices, turnover, and same-day status together."""

        requested = tuple(dict.fromkeys(str(item).strip().upper() for item in symbols if str(item).strip()))
        if not requested:
            return ProviderResult(provider=self.name, data=pd.DataFrame())
        frame, errors, meta = self._persistent_frame_request(
            {
                "kind": "security_lifecycle_history_symbols",
                "symbols": requested,
                "start_date": str(start_date),
                "end_date": str(end_date),
                "adjusted_flag": "none",
            },
            timeout_seconds=max(180, 60 * len(requested)),
        )
        meta["download_strategy"] = "single_login_sequential_symbols"
        meta["requested_symbol_count"] = len(requested)
        return ProviderResult(
            provider=self.name,
            data=frame,
            coverage_report=meta,
            error_report=errors,
        )

    def _fetch_intraday_domain(self, request: DomainFetchRequest) -> ProviderResult:
        is_features = request.domain == DataDomain.INTRADAY_DAILY_FEATURES
        label = "baostock_intraday_daily_features" if is_features else "baostock_intraday_5m"
        raw_rows, errors = self._fetch_intraday_5m_frames(request, progress_label=label)
        rows = (
            [
                build_intraday_daily_feature_frame(
                    raw_5m,
                    source=self.name,
                    adjusted_flag=request.adjusted_flag,
                )
                for raw_5m in raw_rows
                if not raw_5m.empty
            ]
            if is_features
            else raw_rows
        )
        frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        data = normalize_domain_frame(
            frame,
            domain=request.domain,
            source=self.name,
            as_of_date=request.end_date,
            adjusted_flag=request.adjusted_flag,
            require_columns=False,
        )
        return ProviderResult(provider=self.name, data=data, error_report=errors)

    def _persistent_domain_payload(
        self,
        request: DomainFetchRequest,
    ) -> tuple[dict[str, Any], int]:
        if request.domain == DataDomain.TRADING_CALENDAR:
            return {
                "kind": "trade_calendar",
                "start_date": request.start_date,
                "end_date": request.end_date,
                "exchange": request.exchange,
            }, 120
        if request.domain in {DataDomain.UNIVERSE_SNAPSHOT, DataDomain.SECURITY_STATUS}:
            return {
                "kind": "all_stock",
                "domain": request.domain,
                "trade_date": request.end_date,
            }, 120
        if request.domain == DataDomain.INDUSTRY_CONCEPT:
            return {"kind": "industry", "trade_date": request.end_date}, 300
        if request.domain == DataDomain.INDEX_CONSTITUENTS:
            return {"kind": "index_constituents", "trade_date": request.end_date}, 180
        if request.domain == DataDomain.FINANCIAL_QUARTERLY:
            return {
                "kind": "financial_quarterly_symbols",
                "symbols": request.symbols,
                "start_date": request.start_date,
                "end_date": request.end_date,
            }, max(600, 180 * len(request.symbols))
        if request.domain in {DataDomain.PERFORMANCE_FORECAST, DataDomain.PERFORMANCE_EXPRESS}:
            return {
                "kind": "performance_symbols",
                "domain": request.domain,
                "symbols": request.symbols,
                "start_date": request.start_date,
                "end_date": request.end_date,
            }, max(600, 90 * len(request.symbols))
        if request.domain == DataDomain.VALUATION:
            return {
                "kind": "valuation_symbols",
                "symbols": request.symbols,
                "start_date": request.start_date,
                "end_date": request.end_date,
            }, max(600, 90 * len(request.symbols))
        if request.domain == DataDomain.ADJUST_FACTOR:
            return {
                "kind": "adjust_factor_symbols",
                "symbols": request.symbols,
                "start_date": request.start_date,
                "end_date": request.end_date,
            }, max(900, 90 * len(request.symbols))
        raise RuntimeError(f"unsupported_domain: {self.name} does not support {request.domain}")

    def _fetch_persistent_domain(self, request: DomainFetchRequest) -> ProviderResult:
        payload, timeout_seconds = self._persistent_domain_payload(request)
        frame, errors, meta = self._persistent_frame_request(
            payload,
            timeout_seconds=timeout_seconds,
        )
        data = normalize_domain_frame(
            frame,
            domain=request.domain,
            source=self.name,
            as_of_date=request.end_date,
            require_columns=False,
        )
        coverage = coverage_report_for_domain(data, request, provider=self.name)
        coverage.update(meta)
        coverage["download_strategy"] = "single_login_sequential_requests"
        return ProviderResult(
            provider=self.name,
            data=data,
            coverage_report=coverage,
            error_report=errors,
        )

    def _fetch_direct_domain_frame(self, request: DomainFetchRequest) -> pd.DataFrame:
        if request.domain == DataDomain.TRADING_CALENDAR:
            return _fetch_baostock_trade_calendar_frame_with_timeout(
                start_date=request.start_date,
                end_date=request.end_date,
                exchange=request.exchange,
            )
        if request.domain in {DataDomain.UNIVERSE_SNAPSHOT, DataDomain.SECURITY_STATUS}:
            return _fetch_baostock_all_stock_frame_with_timeout(
                domain=request.domain,
                trade_date=request.end_date,
            )
        if request.domain == DataDomain.INDUSTRY_CONCEPT:
            return _fetch_baostock_industry_frame_with_timeout(trade_date=request.end_date)
        if request.domain == DataDomain.INDEX_CONSTITUENTS:
            return _fetch_baostock_index_constituents_frame_with_timeout(trade_date=request.end_date)
        if request.domain == DataDomain.FINANCIAL_QUARTERLY:
            return _fetch_baostock_financial_quarterly_frame_with_timeout(
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
            )
        if request.domain in {DataDomain.PERFORMANCE_FORECAST, DataDomain.PERFORMANCE_EXPRESS}:
            return _fetch_baostock_performance_frame_with_timeout(
                domain=request.domain,
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
            )
        if request.domain == DataDomain.VALUATION:
            return _fetch_baostock_valuation_frame_with_timeout(
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
            )
        if request.domain == DataDomain.ADJUST_FACTOR:
            return _fetch_baostock_adjust_factor_frame_with_timeout(
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
            )
        raise RuntimeError(f"unsupported_domain: {self.name} does not support {request.domain}")

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
        if request.domain in {
            DataDomain.MARKET_INTRADAY_5M,
            DataDomain.INTRADAY_DAILY_FEATURES,
        }:
            return self._fetch_intraday_domain(request)
        if self._reuse_symbol_range_session:
            return self._fetch_persistent_domain(request)
        frame = self._fetch_direct_domain_frame(request)
        data = normalize_domain_frame(
            frame, domain=request.domain, source=self.name, as_of_date=request.end_date, require_columns=False
        )
        return ProviderResult(provider=self.name, data=data)
