"""CNInfo announcement transport and provider implementation."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from quantlab.data.domains.contracts import (
    DataDomain,
    DomainFetchRequest,
    FetchRequest,
    ProviderResult,
    coverage_report_for_domain,
    normalize_domain_frame,
)
from quantlab.data.provider_symbols import (
    strip_suffix as _strip_suffix,
)


@dataclass
class CninfoAnnouncementProvider:
    name: str = "cninfo"
    page_size: int = 30
    max_pages: int = 0

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
                errors.append(
                    {
                        "provider": self.name,
                        "domain": request.domain,
                        "symbol": symbol,
                        "code": "symbol_fetch_error",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                continue
            if not frame.empty:
                frames.append(frame)
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        data = normalize_domain_frame(
            raw, domain=request.domain, source=self.name, as_of_date=request.end_date, require_columns=False
        )
        coverage = coverage_report_for_domain(data, request, provider=self.name)
        coverage.update(
            {
                "endpoint": "hisAnnouncement/query",
                "raw_only_until_pit_audit": True,
                "pit_gate": "blocked_for_model_features_until_disclosure_time_audit",
            }
        )
        return ProviderResult(provider=self.name, data=data, coverage_report=coverage, error_report=errors)


def _cninfo_page_payload(
    *,
    page: int,
    page_size: int,
    symbol: str,
    code: str,
    org_id: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    return {
        "pageNum": page,
        "pageSize": min(max(page_size, 1), 30),
        "column": "szse" if symbol.endswith(".SZ") else "sse",
        "tabName": "fulltext",
        "stock": f"{code},{org_id}",
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


def _cninfo_response_body(response: requests.Response) -> dict[str, Any]:
    if response.status_code >= 400:
        raise RuntimeError(f"cninfo_http_{response.status_code}: {response.text[:200]}")
    try:
        body = response.json()
    except Exception as exc:
        raise RuntimeError(f"cninfo_non_json_response: {response.text[:200]}") from exc
    return body if isinstance(body, dict) else {}


def _normalize_cninfo_page(
    announcements: list[dict[str, Any]],
    *,
    code: str,
    symbol: str,
    org_id: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(announcements)
    if frame.empty:
        return frame
    if "secCode" in frame.columns:
        frame = frame.loc[frame["secCode"].astype(str).eq(code)].copy()
    frame = frame.rename(
        columns={
            "announcementId": "announcement_id",
            "announcementTitle": "title",
            "announcementTime": "publish_time",
            "announcementType": "announcement_type_codes",
            "adjunctSize": "file_size_kb",
        }
    )
    if "publish_time" in frame.columns:
        values = pd.to_numeric(frame["publish_time"], errors="coerce")
        parsed_ms = pd.to_datetime(values, unit="ms", utc=True, errors="coerce")
        parsed_text = pd.to_datetime(frame["publish_time"], utc=True, errors="coerce")
        parsed = parsed_ms.fillna(parsed_text).dt.tz_convert("Asia/Shanghai")
        frame["publish_time"] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
        frame["trade_date"] = parsed.dt.strftime("%Y-%m-%d")
    frame["source_date"] = frame.get("trade_date", "")
    frame["feature_available_date"] = ""
    frame["symbol"] = symbol
    frame["normalized_title"] = frame.get("title", "")
    frame["category"] = frame.get("announcement_type_codes", "")
    frame["cninfo_announcement_id"] = frame.get("announcement_id", "")
    frame["eastmoney_art_code"] = ""
    frame["org_id"] = org_id
    adjunct = frame.get("adjunctUrl", pd.Series("", index=frame.index)).fillna("").astype(str)
    frame["pdf_url"] = adjunct.map(
        lambda value: f"http://static.cninfo.com.cn/{value.lstrip('/')}" if value else ""
    )
    frame["url"] = frame.apply(
        lambda row: (
            "http://www.cninfo.com.cn/new/disclosure/detail?"
            f"stockCode={code}&announcementId={row.get('announcement_id', '')}"
            f"&orgId={org_id}&announcementTime={row.get('trade_date', '')}"
        ),
        axis=1,
    )
    frame["cninfo_present"] = True
    frame["eastmoney_present"] = False
    frame["source_disagreement"] = False
    frame["source"] = "cninfo"
    return frame


def _fetch_cninfo_announcements(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    page_size: int,
    max_pages: int,
    org_id: str = "",
) -> pd.DataFrame:
    url = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
    headers = {
        "User-Agent": "Mozilla/5.0 qdp-cninfo-provider",
        "Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch",
    }
    normalized_symbol = str(symbol).strip().upper()
    code = _strip_suffix(normalized_symbol)
    resolved_org_id = str(org_id or _fetch_cninfo_org_id(code)).strip()
    if not resolved_org_id:
        raise RuntimeError(f"cninfo_org_id_missing:{normalized_symbol}")
    rows: list[pd.DataFrame] = []
    page = 1
    total_pages: int | None = None
    session = requests.Session()
    try:
        while total_pages is None or page <= total_pages:
            if int(max_pages) > 0 and page > int(max_pages):
                break
            payload = _cninfo_page_payload(
                page=page,
                page_size=int(page_size),
                symbol=normalized_symbol,
                code=code,
                org_id=resolved_org_id,
                start_date=start_date,
                end_date=end_date,
            )
            body = _cninfo_response_body(session.post(url, headers=headers, data=payload, timeout=20))
            announcements = list(body.get("announcements", []) or [])
            if total_pages is None:
                total = int(body.get("totalAnnouncement", 0) or 0)
                total_pages = max(1, math.ceil(total / 30)) if total else 0
            frame = _normalize_cninfo_page(
                announcements,
                code=code,
                symbol=normalized_symbol,
                org_id=resolved_org_id,
            )
            if frame.empty:
                break
            rows.append(frame)
            page += 1
            if total_pages == 0:
                break
    finally:
        session.close()
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


_CNINFO_ORG_MAP: dict[str, str] | None = None


_CNINFO_ORG_MAP_LOCK = threading.Lock()


def _cninfo_org_map() -> dict[str, str]:
    global _CNINFO_ORG_MAP
    with _CNINFO_ORG_MAP_LOCK:
        if _CNINFO_ORG_MAP is not None:
            return dict(_CNINFO_ORG_MAP)
        response = requests.get(
            "http://www.cninfo.com.cn/new/data/szse_stock.json",
            headers={"User-Agent": "Mozilla/5.0 qdp-cninfo-provider"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        _CNINFO_ORG_MAP = {
            str(item.get("code", "")).strip(): str(item.get("orgId", "")).strip()
            for item in list(payload.get("stockList", []) or [])
            if str(item.get("code", "")).strip() and str(item.get("orgId", "")).strip()
        }
        return dict(_CNINFO_ORG_MAP)


def _fetch_cninfo_org_id(symbol: str) -> str:
    code = _strip_suffix(symbol)
    resolved = _cninfo_org_map().get(code, "")
    if resolved:
        return resolved
    response = requests.post(
        "https://irm.cninfo.com.cn/newircs/index/queryKeyboardInfo",
        params={"_t": str(int(time.time() * 1000))},
        data={"keyWord": code},
        headers={"User-Agent": "Mozilla/5.0 qdp-cninfo-provider"},
        timeout=30,
    )
    response.raise_for_status()
    candidates = list((response.json() or {}).get("data", []) or [])
    exact = [item for item in candidates if str(item.get("stockcode", item.get("secCode", ""))).strip() == code]
    selected = exact[0] if exact else (candidates[0] if candidates else {})
    return str(selected.get("secid", selected.get("orgId", ""))).strip()
