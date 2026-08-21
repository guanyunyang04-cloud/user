"""Baostock transport responsibilities."""

from __future__ import annotations

import multiprocessing
import queue as queue_module
import threading
import time
from typing import Any

import pandas as pd

from .runtime import (
    _BAOSTOCK_GLOBAL_LIMITER,
)
from .workers import (
    _baostock_adjust_factor_worker,
    _baostock_all_stock_worker,
    _baostock_bulk_partition_worker,
    _baostock_financial_quarterly_worker,
    _baostock_history_worker,
    _baostock_index_constituents_worker,
    _baostock_industry_worker,
    _baostock_intraday_5m_worker,
    _baostock_performance_worker,
    _baostock_persistent_session_worker,
    _baostock_stock_basic_worker,
    _baostock_trade_calendar_worker,
    _baostock_valuation_worker,
)


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
                raise TimeoutError(f"baostock_bulk_timeout:{endpoint}: exceeded {int(timeout)} seconds") from None
            if process.exitcode not in {0, None}:
                raise RuntimeError(f"baostock_bulk_worker_failed:{endpoint}:exitcode={process.exitcode}") from None
            raise RuntimeError(f"baostock_bulk_worker_returned_no_payload:{endpoint}") from None
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
                raise TimeoutError(f"{timeout_label}: exceeded {int(timeout)} seconds") from None
            if process.exitcode not in {0, None}:
                raise RuntimeError(f"{failure_label}_worker_failed: exitcode={process.exitcode}") from None
            raise RuntimeError(f"{failure_label}_worker_returned_no_payload") from None
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


def _fetch_baostock_all_stock_frame_with_timeout(
    *, domain: str, trade_date: str, timeout_seconds: int = 60
) -> pd.DataFrame:
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


def _fetch_baostock_index_constituents_frame_with_timeout(
    *, trade_date: str, timeout_seconds: int = 90
) -> pd.DataFrame:
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
        kwargs={
            "domain": str(domain),
            "symbols": tuple(symbols),
            "start_date": str(start_date),
            "end_date": str(end_date),
        },
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
