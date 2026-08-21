"""Baostock frames responsibilities."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from quantlab.data.domains.contracts import (
    DataDomain,
    DomainFetchRequest,
)
from quantlab.data.provider_symbols import (
    baostock_board as _baostock_board,
)
from quantlab.data.provider_symbols import (
    baostock_exchange as _baostock_exchange,
)
from quantlab.data.provider_symbols import (
    baostock_name_is_st as _baostock_name_is_st,
)
from quantlab.data.provider_symbols import (
    from_baostock_code as _from_baostock_code,
)
from quantlab.data.provider_symbols import (
    is_baostock_a_share_code as _is_baostock_a_share_code,
)
from quantlab.data.provider_symbols import (
    to_baostock_code as _to_baostock_code,
)

from .runtime import (
    BAOSTOCK_BULK_PER_PAGE_COUNT,
    _quiet_baostock_call,
)


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
            f"{failure_label}_unexpected_per_page_count:expected={BAOSTOCK_BULK_PER_PAGE_COUNT} actual={per_page_count}"
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
            f"{failure_label}_potential_truncation:row_count={len(rows)} capacity={BAOSTOCK_BULK_PER_PAGE_COUNT}"
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
                f"{failure_label}_field_width_mismatch:index={index} expected={len(fields)} actual={len(values)}"
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
        raw_is_st = raw["isST"].fillna("").astype(str).str.strip().str.lower()
        is_st = raw_is_st.map({"1": True, "true": True, "0": False, "false": False}).astype("boolean")
        is_suspended = trade_status.map({"0": True, "1": False}).astype("boolean")
        return pd.DataFrame(
            {
                "trade_date": raw["date"].astype(str),
                "provider_symbol": provider_symbol,
                "tradestatus": trade_status,
                "is_st": is_st,
                "is_suspended": is_suspended,
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
    event_date_field = _exact_column(
        raw, ("dividOperateDate", "divid_operate_date"), label="baostock_bulk_adjust_factor"
    )
    fore_field = _exact_column(raw, ("foreAdjustFactor", "fore_adjust_factor"), label="baostock_bulk_adjust_factor")
    back_field = _exact_column(raw, ("backAdjustFactor", "back_adjust_factor"), label="baostock_bulk_adjust_factor")
    factor_field = _exact_column(
        raw, ("adjustFacto", "adjustFactor", "adjust_factor"), label="baostock_bulk_adjust_factor"
    )
    event_dates = raw[event_date_field].fillna("").astype(str).str.strip()
    unexpected = sorted(set(event_dates.loc[event_dates.ne(str(query_date))].tolist()))
    if unexpected:
        raise RuntimeError(
            f"baostock_bulk_adjust_factor_event_date_mismatch:query_date={query_date} returned={unexpected[:10]}"
        )
    return (
        pd.DataFrame(
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
        )
        .loc[:, columns]
        .reset_index(drop=True)
    )


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
    trade_status = raw["tradeStatus"].fillna("").astype(str).str.strip()
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
            "trade_status": trade_status,
            "is_suspended": trade_status.map({"0": True, "1": False}).astype("boolean"),
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
            "is_st": raw["name"].map(_baostock_name_is_st).astype("boolean"),
            "is_suspended": trade_status.map({"0": True, "1": False}).astype("boolean"),
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


def _baostock_valuation_frame_from_history(
    bs: Any, request: DomainFetchRequest, *, relogin_retries: int = 2
) -> pd.DataFrame:
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
                raise RuntimeError(
                    f"baostock {relogin_context} relogin failed: {getattr(login, 'error_msg', '')}"
                ) from exc
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


def _baostock_security_lifecycle_history_frame(
    query: Any,
    *,
    symbol: str,
) -> pd.DataFrame:
    raw = _baostock_query_to_frame(
        query,
        "baostock_security_lifecycle_history",
    )
    if raw.empty:
        return pd.DataFrame()
    frame = raw.rename(columns={"date": "trade_date", "code": "provider_code"}).copy()
    frame["symbol"] = str(symbol).strip().upper()
    expected = [
        "trade_date",
        "symbol",
        "provider_code",
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "adjustflag",
        "turn",
        "tradestatus",
        "pctChg",
        "peTTM",
        "pbMRQ",
        "psTTM",
        "pcfNcfTTM",
        "isST",
    ]
    missing = [item for item in expected if item not in frame.columns]
    if missing:
        raise RuntimeError(f"baostock_security_lifecycle_history_schema_error:symbol={symbol}:missing={missing}")
    return frame.loc[:, expected].reset_index(drop=True)


def _baostock_intraday_5m_frame(query: Any, *, symbol: str) -> pd.DataFrame:
    raw = _baostock_query_to_frame(query, "baostock_intraday_5m")
    if raw.empty:
        return pd.DataFrame()
    frame = raw.rename(columns={"date": "trade_date", "time": "bar_time", "code": "symbol"}).copy()
    frame["symbol"] = str(symbol).strip().upper()
    expected = ["trade_date", "symbol", "bar_time", "open", "high", "low", "close", "volume", "amount", "adjustflag"]
    return frame[[column for column in expected if column in frame.columns]].rename(
        columns={"adjustflag": "adjusted_flag"}
    )


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


def _baostock_financial_quarterly_frame_from_bs(
    bs: Any, request: DomainFetchRequest, *, relogin_retries: int = 2
) -> pd.DataFrame:
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
                    lambda query_func=query_func, code=code, year=year, quarter=quarter: query_func(
                        code=code, year=year, quarter=quarter
                    ),
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


def _baostock_performance_frame_from_bs(
    bs: Any, request: DomainFetchRequest, *, relogin_retries: int = 2
) -> pd.DataFrame:
    request = request.normalized()
    rows: list[pd.DataFrame] = []
    query_func = (
        bs.query_forecast_report
        if request.domain == DataDomain.PERFORMANCE_FORECAST
        else bs.query_performance_express_report
    )
    failure_label = (
        "baostock_performance_forecast"
        if request.domain == DataDomain.PERFORMANCE_FORECAST
        else "baostock_performance_express"
    )
    for symbol in request.symbols:
        raw = _baostock_query_to_frame_with_relogin(
            bs,
            lambda query_func=query_func, symbol=symbol: query_func(
                _to_baostock_code(symbol), start_date=request.start_date, end_date=request.end_date
            ),
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


def _baostock_adjust_factor_frame_from_bs(
    bs: Any, request: DomainFetchRequest, *, relogin_retries: int = 2
) -> pd.DataFrame:
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


def _baostock_symbol_error(
    provider: str, symbol: str, exc: BaseException, *, first_error: BaseException | None = None
) -> dict[str, Any]:
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
