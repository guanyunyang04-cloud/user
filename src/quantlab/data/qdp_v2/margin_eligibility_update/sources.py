"""Margin Eligibility Update: sources responsibilities."""

from __future__ import annotations

import io
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from quantlab.data.qdp_v2.manifest import (
    utc_now,
)
from quantlab.data.qdp_v2.research_event_update import (
    _sha256,
    _write_parquet,
)

from .config import (
    ENDPOINTS,
    OFFICIAL_COLUMNS,
    SSE_URL,
    SZSE_URL,
    USER_AGENT,
    EndpointResult,
    MarginEligibilityUpdateError,
)
from .context import (
    _hash_bytes,
    _number,
    _raw_day_path,
)


def _official_frame(
    *,
    symbol: pd.Series,
    trade_date: str,
    exchange: str,
    endpoint: str,
    name: pd.Series,
    source: str,
    eligible: pd.Series | bool | None = None,
    finance_eligible: pd.Series | bool | None = None,
    securities_lending_eligible: pd.Series | bool | None = None,
    metrics: Mapping[str, pd.Series] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(index=symbol.index)
    code = symbol.fillna("").astype(str).str.extract(r"(\d{6})", expand=False)
    frame["symbol"] = code + (".SH" if exchange == "SH" else ".SZ")
    frame["trade_date"] = trade_date
    frame["exchange"] = exchange
    frame["endpoint"] = endpoint
    frame["name"] = name.fillna("").astype(str)
    for column, value in (
        ("eligible", eligible),
        ("finance_eligible", finance_eligible),
        ("securities_lending_eligible", securities_lending_eligible),
    ):
        if isinstance(value, pd.Series):
            frame[column] = value.astype("boolean")
        else:
            frame[column] = pd.Series(value, index=frame.index, dtype="boolean")
    for column in (
        "rzye",
        "rqye",
        "rzmre",
        "rqyl",
        "rzche",
        "rqchl",
        "rqmcl",
        "rzrqye",
    ):
        frame[column] = pd.to_numeric(metrics[column], errors="coerce") if metrics and column in metrics else np.nan
    frame["source"] = source
    return frame.loc[code.notna(), OFFICIAL_COLUMNS].reset_index(drop=True)


def _fetch_sse_detail(trade_date: str, *, timeout: int = 45) -> EndpointResult:
    compact = trade_date.replace("-", "")
    params = {
        "isPagination": "true",
        "tabType": "mxtype",
        "detailsDate": compact,
        "stockCode": "",
        "beginDate": "",
        "endDate": "",
        "pageHelp.pageSize": "5000",
        "pageHelp.pageCount": "50",
        "pageHelp.pageNo": "1",
        "pageHelp.beginPage": "1",
        "pageHelp.cacheSize": "1",
        "pageHelp.endPage": "21",
    }
    response = requests.get(
        SSE_URL,
        params=params,
        headers={"Referer": "https://www.sse.com.cn/", "User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.content
    data = response.json()
    raw = pd.DataFrame(data.get("result", []))
    if raw.empty:
        raise MarginEligibilityUpdateError(f"sse_detail_empty:{trade_date}")
    if not {"stockCode", "securityAbbr", "opDate", "rzye"}.issubset(raw.columns):
        raise MarginEligibilityUpdateError(f"sse_detail_schema:{trade_date}")
    dates = raw["opDate"].fillna("").astype(str).unique().tolist()
    if dates != [compact]:
        raise MarginEligibilityUpdateError(f"sse_detail_date_mismatch:{trade_date}:{dates[:3]}")
    frame = _official_frame(
        symbol=raw["stockCode"],
        trade_date=trade_date,
        exchange="SH",
        endpoint="sse_detail",
        name=raw["securityAbbr"],
        eligible=True,
        source="sse_official_margin_detail",
        metrics={
            "rzye": _number(raw["rzye"]),
            "rzmre": _number(raw["rzmre"]),
            "rzche": _number(raw["rzche"]),
            "rqyl": _number(raw["rqyl"]),
            "rqmcl": _number(raw["rqmcl"]),
            "rqchl": _number(raw["rqchl"]),
        },
    )
    return EndpointResult("sse_detail", frame, _hash_bytes(payload), len(payload))


def _szse_endpoint_config(endpoint: str) -> tuple[str, str, str, int]:
    if endpoint == "szse_detail":
        return "1837_xxpl", "tab2", "https://www.szse.cn/disclosure/margin/margin/index.html", 8
    if endpoint == "szse_eligibility":
        return "1834_xxpl", "tab1", "https://www.szse.cn/disclosure/margin/object/index.html", 7
    raise ValueError(f"unsupported_szse_endpoint:{endpoint}")


def _read_szse_workbook(payload: bytes, *, endpoint: str, expected_width: int, trade_date: str) -> pd.DataFrame:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Workbook contains no default style.*",
                category=UserWarning,
            )
            raw = pd.read_excel(io.BytesIO(payload), engine="openpyxl", dtype=str)
    except Exception as exc:
        raise MarginEligibilityUpdateError(f"{endpoint}_parse:{trade_date}:{type(exc).__name__}") from exc
    if raw.empty or len(raw.columns) < expected_width:
        raise MarginEligibilityUpdateError(f"{endpoint}_schema_or_empty:{trade_date}")
    return raw


def _normalize_szse_detail(raw: pd.DataFrame, *, trade_date: str, endpoint: str) -> pd.DataFrame:
    normalized = raw.iloc[:, :8].copy()
    normalized.columns = [
        "security_code",
        "security_name",
        "financing_buy_amount",
        "financing_balance",
        "lending_sell_volume",
        "lending_balance_volume",
        "lending_balance_amount",
        "margin_total_balance",
    ]
    return _official_frame(
        symbol=normalized["security_code"],
        trade_date=trade_date,
        exchange="SZ",
        endpoint=endpoint,
        name=normalized["security_name"],
        source="szse_official_margin_detail",
        metrics={
            "rzmre": _number(normalized["financing_buy_amount"]),
            "rzye": _number(normalized["financing_balance"]),
            "rqmcl": _number(normalized["lending_sell_volume"]),
            "rqyl": _number(normalized["lending_balance_volume"]),
            "rqye": _number(normalized["lending_balance_amount"]),
            "rzrqye": _number(normalized["margin_total_balance"]),
        },
    )


def _normalize_szse_eligibility(raw: pd.DataFrame, *, trade_date: str, endpoint: str) -> pd.DataFrame:
    normalized = raw.iloc[:, :7].copy()
    normalized.columns = [
        "security_code",
        "security_name",
        "financing_underlying",
        "lending_underlying",
        "financing_available_today",
        "lending_available_today",
        "lending_price_limit_status",
    ]
    finance = normalized["financing_underlying"].fillna("").str.upper().eq("Y")
    lending = normalized["lending_underlying"].fillna("").str.upper().eq("Y")
    return _official_frame(
        symbol=normalized["security_code"],
        trade_date=trade_date,
        exchange="SZ",
        endpoint=endpoint,
        name=normalized["security_name"],
        source="szse_official_margin_eligibility",
        eligible=finance | lending,
        finance_eligible=finance,
        securities_lending_eligible=lending,
    )


def _fetch_szse_excel(
    trade_date: str,
    *,
    endpoint: str,
    timeout: int = 45,
) -> EndpointResult:
    catalog, tab, referer, expected_width = _szse_endpoint_config(endpoint)
    params = {
        "SHOWTYPE": "xlsx",
        "CATALOGID": catalog,
        "txtDate": trade_date,
        f"{tab}PAGENO": "1",
        "random": "0.3141592653589793",
        "TABKEY": tab,
    }
    response = requests.get(
        SZSE_URL,
        params=params,
        headers={"Referer": referer, "User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.content
    raw = _read_szse_workbook(payload, endpoint=endpoint, expected_width=expected_width, trade_date=trade_date)
    # Official column order is stable even when the workbook's GBK labels are mojibake.
    normalizer = _normalize_szse_detail if endpoint == "szse_detail" else _normalize_szse_eligibility
    frame = normalizer(raw, trade_date=trade_date, endpoint=endpoint)
    return EndpointResult(endpoint, frame, _hash_bytes(payload), len(payload))


def _fetch_endpoint(trade_date: str, endpoint: str) -> EndpointResult:
    last_error = ""
    for attempt in range(4):
        try:
            if endpoint == "sse_detail":
                return _fetch_sse_detail(trade_date)
            return _fetch_szse_excel(trade_date, endpoint=endpoint)
        except (requests.RequestException, MarginEligibilityUpdateError) as exc:
            last_error = f"{type(exc).__name__}:{str(exc)[:240]}"
            if attempt < 3:
                time.sleep(0.5 * (attempt + 1))
    raise MarginEligibilityUpdateError(f"exchange_request_failed:{trade_date}:{endpoint}:{last_error}")


def _fetch_day(workspace: Path, trade_date: str) -> dict[str, Any]:
    results = [_fetch_endpoint(trade_date, endpoint) for endpoint in ENDPOINTS]
    frame = pd.concat([result.frame for result in results], ignore_index=True)
    if set(frame["endpoint"].unique()) != set(ENDPOINTS):
        raise MarginEligibilityUpdateError(f"exchange_endpoint_incomplete:{trade_date}")
    path = _raw_day_path(workspace, trade_date)
    _write_parquet(frame, path)
    return {
        "status": "observed",
        "path": str(path),
        "row_count": len(frame),
        "sha256": _sha256(path),
        "endpoints": {
            result.endpoint: {
                "status": "observed",
                "row_count": len(result.frame),
                "response_sha256": result.response_sha256,
                "response_bytes": result.response_bytes,
            }
            for result in results
        },
        "completed_at": utc_now(),
    }
