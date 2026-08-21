"""Baostock workers responsibilities."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from quantlab.data.domains.contracts import (
    DataDomain,
    DomainFetchRequest,
)
from quantlab.data.provider_symbols import (
    to_baostock_code as _to_baostock_code,
)

from .frames import (
    _baostock_adjust_factor_frame_from_bs,
    _baostock_all_stock_frame,
    _baostock_bulk_query_to_frame,
    _baostock_financial_quarterly_frame_from_bs,
    _baostock_history_frame,
    _baostock_index_constituents_frame,
    _baostock_industry_frame,
    _baostock_intraday_5m_frame,
    _baostock_performance_frame_from_bs,
    _baostock_query_to_frame,
    _baostock_security_lifecycle_history_frame,
    _baostock_status_frame_from_all_stock,
    _baostock_stock_basic_frame,
    _baostock_valuation_frame_from_history,
)
from .runtime import (
    _quiet_baostock_call,
    assert_baostock_batch_runtime,
    baostock_runtime_version,
)


def _baostock_bulk_partition_worker(queue: Any, endpoint: str, trade_date: str) -> None:
    try:
        import baostock as bs  # type: ignore

        installed = assert_baostock_batch_runtime()
        login = _quiet_baostock_call(bs.login)
        if getattr(login, "error_code", "1") != "0":
            queue.put(
                {
                    "status": "error",
                    "error_type": "RuntimeError",
                    "error": f"baostock login failed: {getattr(login, 'error_msg', '')}",
                }
            )
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


def _baostock_persistent_symbol_frames(
    bs: Any,
    *,
    kind: str,
    command: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run a symbol batch without reopening the BaoStock session per symbol."""

    symbols = tuple(str(item) for item in tuple(command.get("symbols", ()) or ()))
    start_date = str(command.get("start_date", ""))
    end_date = str(command.get("end_date", ""))
    adjusted_flag = str(command.get("adjusted_flag", "none") or "none")
    frames: list[pd.DataFrame] = []
    errors: list[dict[str, Any]] = []
    domain_by_kind = {
        "history_symbols": DataDomain.MARKET_DAILY,
        "security_lifecycle_history_symbols": DataDomain.SECURITY_STATUS,
        "intraday_5m_symbols": DataDomain.MARKET_INTRADAY_5M,
        "financial_quarterly_symbols": DataDomain.FINANCIAL_QUARTERLY,
        "performance_symbols": str(command.get("domain", "")),
        "valuation_symbols": DataDomain.VALUATION,
        "adjust_factor_symbols": DataDomain.ADJUST_FACTOR,
    }
    domain = domain_by_kind.get(kind, kind)
    for symbol in symbols:
        try:
            frame = _baostock_persistent_symbol_frame(
                bs,
                kind=kind,
                command=command,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjusted_flag=adjusted_flag,
            )
        except Exception as exc:
            errors.append(
                {
                    "provider": "baostock",
                    "domain": domain,
                    "symbol": symbol,
                    "code": "symbol_fetch_exception",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frames.append(frame)
    return (pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()), errors


def _baostock_persistent_symbol_frame(
    bs: Any,
    *,
    kind: str,
    command: dict[str, Any],
    symbol: str,
    start_date: str,
    end_date: str,
    adjusted_flag: str,
) -> pd.DataFrame:
    if kind == "history_symbols":
        query = bs.query_history_k_data_plus(
            _to_baostock_code(symbol),
            "date,code,open,high,low,close,volume,amount",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2" if adjusted_flag in {"front", "qfq"} else "3",
        )
        return _baostock_history_frame(query, symbol=symbol)
    if kind == "security_lifecycle_history_symbols":
        query = bs.query_history_k_data_plus(
            _to_baostock_code(symbol),
            (
                "date,code,open,high,low,close,preclose,volume,amount,"
                "adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,"
                "psTTM,pcfNcfTTM,isST"
            ),
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3",
        )
        return _baostock_security_lifecycle_history_frame(query, symbol=symbol)
    if kind == "intraday_5m_symbols":
        query = bs.query_history_k_data_plus(
            _to_baostock_code(symbol),
            "date,time,code,open,high,low,close,volume,amount,adjustflag",
            start_date=start_date,
            end_date=end_date,
            frequency="5",
            adjustflag="2" if adjusted_flag in {"front", "qfq"} else "3",
        )
        return _baostock_intraday_5m_frame(query, symbol=symbol)
    domains = {
        "financial_quarterly_symbols": DataDomain.FINANCIAL_QUARTERLY,
        "performance_symbols": str(command.get("domain", "")),
        "valuation_symbols": DataDomain.VALUATION,
        "adjust_factor_symbols": DataDomain.ADJUST_FACTOR,
    }
    domain = domains.get(kind)
    if domain is None:
        raise RuntimeError(f"unsupported_baostock_symbol_command:{kind}")
    request = DomainFetchRequest(
        domain=domain,
        symbols=(symbol,),
        start_date=start_date,
        end_date=end_date,
    )
    if kind == "financial_quarterly_symbols":
        return _baostock_financial_quarterly_frame_from_bs(bs, request)
    if kind == "performance_symbols":
        return _baostock_performance_frame_from_bs(bs, request)
    if kind == "valuation_symbols":
        return _baostock_valuation_frame_from_history(bs, request)
    return _baostock_adjust_factor_frame_from_bs(bs, request)


def _baostock_persistent_command(
    bs: Any,
    command: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    kind = str(command.get("kind", ""))
    trade_date = str(command.get("trade_date", ""))
    extra: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
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
            extra["audit_data"] = _baostock_all_stock_frame(audit_query, trade_date=trade_date)
        return frame, meta, errors, extra
    if kind == "all_stock":
        domain = str(command.get("domain", DataDomain.UNIVERSE_SNAPSHOT))
        query = bs.query_all_stock(day=trade_date)
        frame = (
            _baostock_status_frame_from_all_stock(query, trade_date=trade_date)
            if domain == DataDomain.SECURITY_STATUS
            else _baostock_all_stock_frame(query, trade_date=trade_date)
        )
    elif kind in {
        "history_symbols",
        "security_lifecycle_history_symbols",
        "intraday_5m_symbols",
        "financial_quarterly_symbols",
        "performance_symbols",
        "valuation_symbols",
        "adjust_factor_symbols",
    }:
        frame, errors = _baostock_persistent_symbol_frames(bs, kind=kind, command=command)
    elif kind == "trade_calendar":
        query = bs.query_trade_dates(
            start_date=str(command.get("start_date", "")),
            end_date=str(command.get("end_date", "")),
        )
        raw = _baostock_query_to_frame(query, "baostock_trade_calendar")
        if raw.empty:
            frame = pd.DataFrame(columns=["trade_date", "is_open", "exchange"])
        else:
            date_field = "calendar_date" if "calendar_date" in raw.columns else raw.columns[0]
            open_field = "is_trading_day" if "is_trading_day" in raw.columns else raw.columns[1]
            frame = pd.DataFrame(
                {
                    "trade_date": raw[date_field],
                    "is_open": raw[open_field],
                    "exchange": str(command.get("exchange", "SSE") or "SSE"),
                }
            )
    elif kind == "industry":
        frame = _baostock_industry_frame(
            bs.query_stock_industry(date=trade_date),
            trade_date=trade_date,
        )
    elif kind == "index_constituents":
        frame = _baostock_index_constituents_frame(bs, trade_date=trade_date)
    elif kind == "stock_basic":
        frame = _baostock_stock_basic_frame(bs.query_stock_basic(), trade_date=trade_date)
    else:
        raise RuntimeError(f"unsupported_baostock_persistent_command:{kind}")
    meta = {
        "error_code": "0",
        "error_msg": "",
        "package_version": (assert_baostock_batch_runtime() if kind == "all_stock" else baostock_runtime_version()),
    }
    return frame, meta, errors, extra


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
                    {
                        "status": "error",
                        "request_id": request_id,
                        "error_type": "TypeError",
                        "error": "persistent command must be a mapping",
                    }
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
                frame, meta, error_report, extra_response = _baostock_persistent_command(bs, command)
                response_queue.put(
                    {
                        "status": "ok",
                        "request_id": request_id,
                        "data": frame,
                        "meta": {**meta, "worker_elapsed_seconds": round(time.perf_counter() - started, 6)},
                        "error_report": error_report,
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
            queue.put(
                {
                    "status": "error",
                    "error_type": "RuntimeError",
                    "error": f"baostock login failed: {getattr(login, 'error_msg', '')}",
                }
            )
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
            queue.put(
                {
                    "status": "error",
                    "error_type": "RuntimeError",
                    "error": f"baostock login failed: {getattr(login, 'error_msg', '')}",
                }
            )
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
            queue.put(
                {
                    "status": "error",
                    "error_type": "RuntimeError",
                    "error": f"baostock login failed: {getattr(login, 'error_msg', '')}",
                }
            )
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
            queue.put(
                {
                    "status": "error",
                    "error_type": "RuntimeError",
                    "error": f"baostock login failed: {getattr(login, 'error_msg', '')}",
                }
            )
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
            queue.put(
                {
                    "status": "error",
                    "error_type": "RuntimeError",
                    "error": f"baostock login failed: {getattr(login, 'error_msg', '')}",
                }
            )
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
            queue.put(
                {
                    "status": "error",
                    "error_type": "RuntimeError",
                    "error": f"baostock login failed: {getattr(login, 'error_msg', '')}",
                }
            )
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
            queue.put(
                {
                    "status": "error",
                    "error_type": "RuntimeError",
                    "error": f"baostock login failed: {getattr(login, 'error_msg', '')}",
                }
            )
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


def _baostock_performance_worker(
    queue: Any, domain: str, symbols: tuple[str, ...], start_date: str, end_date: str
) -> None:
    try:
        import baostock as bs  # type: ignore

        login = _quiet_baostock_call(bs.login)
        if getattr(login, "error_code", "1") != "0":
            queue.put(
                {
                    "status": "error",
                    "error_type": "RuntimeError",
                    "error": f"baostock login failed: {getattr(login, 'error_msg', '')}",
                }
            )
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
            queue.put(
                {
                    "status": "error",
                    "error_type": "RuntimeError",
                    "error": f"baostock login failed: {getattr(login, 'error_msg', '')}",
                }
            )
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
            queue.put(
                {
                    "status": "error",
                    "error_type": "RuntimeError",
                    "error": f"baostock login failed: {getattr(login, 'error_msg', '')}",
                }
            )
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
            queue.put(
                {
                    "status": "error",
                    "error_type": "RuntimeError",
                    "error": f"baostock login failed: {getattr(login, 'error_msg', '')}",
                }
            )
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
