from __future__ import annotations

import os
import multiprocessing
import queue as queue_module
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from daily_research.data_platform.contracts import (
    DataDomain,
    DomainFetchRequest,
    FetchRequest,
    ProviderResult,
    normalize_domain_frame,
    normalize_market_frame,
    validate_provider_name,
)
from daily_research.progress import create_progress, progress_write


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
    DataDomain.MONEY_FLOW_HOTSPOT,
)
FORMAL_FREE_V3_RESEARCH_FUTURE_DOMAINS: tuple[str, ...] = (
    DataDomain.NEWS_EVENT,
    DataDomain.ANNOUNCEMENT,
    DataDomain.RESEARCH_REPORT,
    DataDomain.IWENCAI_SEMANTIC,
)


_PROVIDER_CAPABILITIES: dict[str, dict[str, Any]] = {
    "baostock": {
        "domains": (DataDomain.MARKET_DAILY, DataDomain.TRADING_CALENDAR, DataDomain.UNIVERSE_SNAPSHOT, DataDomain.SECURITY_STATUS),
        "requires_token": False,
        "formal_eligible": True,
        "notes": "stable free source for daily bars, calendar and universe/status basics",
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
    "tushare_http_optional": {
        "domains": (DataDomain.MARKET_DAILY, DataDomain.TRADING_CALENDAR, DataDomain.UNIVERSE_SNAPSHOT, DataDomain.VALUATION, DataDomain.LIMIT_STATUS),
        "requires_token": True,
        "formal_eligible": False,
        "notes": "token-gated optional source; never required by formal_free_v3",
    },
    "sina_tencent_realtime": {
        "domains": (),
        "requires_token": False,
        "formal_eligible": False,
        "notes": "legacy placeholder for future same-day supplement",
    },
}


def _strip_suffix(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    if "." in raw:
        return raw.split(".", 1)[0]
    if raw.startswith(("SH", "SZ", "BJ")) and raw[2:].isdigit():
        return raw[2:]
    return raw


def _formal_requirement(domain: str) -> str:
    if domain in FORMAL_FREE_V3_REQUIRED_DOMAINS:
        return "required"
    if domain in FORMAL_FREE_V3_OPTIONAL_DOMAINS:
        return "optional"
    if domain in FORMAL_FREE_V3_RESEARCH_FUTURE_DOMAINS:
        return "research_future"
    return "unsupported"


def provider_capability_matrix(provider_plan: str = "formal_free_v3") -> list[dict[str, Any]]:
    plan = str(provider_plan or "formal_free_v3").strip().lower()
    if plan == "formal_free_v3":
        provider_names = ("baostock", "eastmoney_efinance", "akshare_eastmoney", "tencent_finance", "tonghuashun_hotspot", "tushare_http_optional")
    else:
        provider_names = tuple(str(getattr(provider, "name", "")) for provider in build_default_providers(plan))
    rows: list[dict[str, Any]] = []
    all_domains = (
        *FORMAL_FREE_V3_REQUIRED_DOMAINS,
        *FORMAL_FREE_V3_OPTIONAL_DOMAINS,
        *FORMAL_FREE_V3_RESEARCH_FUTURE_DOMAINS,
    )
    for provider_name in provider_names:
        meta = _PROVIDER_CAPABILITIES.get(provider_name, {"domains": (), "requires_token": False, "formal_eligible": False, "notes": ""})
        supported = set(str(item) for item in meta.get("domains", ()))
        for domain in all_domains:
            formal_refresh = bool(plan == "formal_free_v3" and meta.get("formal_eligible", False) and _formal_requirement(domain) in {"required", "optional"})
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
                    "requirement": _formal_requirement(domain),
                    "notes": str(meta.get("notes", "")),
                }
            )
    return rows


@dataclass
class EastmoneyEfinanceProvider:
    name: str = "eastmoney_efinance"

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        validate_provider_name(self.name)
        try:
            import efinance as ef  # type: ignore
        except Exception as exc:
            raise RuntimeError("efinance is not installed in the yolos environment") from exc
        request = request.normalized()
        frames: list[pd.DataFrame] = []
        for symbol in request.symbols:
            payload = ef.stock.get_quote_history(
                stock_codes=_strip_suffix(symbol),
                beg=request.start_date.replace("-", ""),
                end=request.end_date.replace("-", ""),
                klt=101,
                fqt=1 if request.adjusted_flag in {"front", "qfq"} else 0,
            )
            if isinstance(payload, dict):
                for value in payload.values():
                    if isinstance(value, pd.DataFrame):
                        frame = value.copy()
                        frame["symbol"] = symbol
                        frames.append(frame)
            elif isinstance(payload, pd.DataFrame):
                frame = payload.copy()
                frame["symbol"] = symbol
                frames.append(frame)
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        data = normalize_market_frame(raw, source=self.name, adjusted_flag=request.adjusted_flag, require_columns=False)
        return ProviderResult(provider=self.name, data=data)

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
        if request.domain in {DataDomain.UNIVERSE_SNAPSHOT, DataDomain.VALUATION}:
            try:
                import efinance as ef  # type: ignore
            except Exception as exc:
                raise RuntimeError("efinance is not installed in the yolos environment") from exc
            raw = ef.stock.get_realtime_quotes()
            frame = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame()
            if not frame.empty:
                frame["trade_date"] = request.end_date
            data = normalize_domain_frame(frame, domain=request.domain, source=self.name, as_of_date=request.end_date, require_columns=False)
            return ProviderResult(provider=self.name, data=data)
        raise RuntimeError(f"unsupported_domain: {self.name} does not support {request.domain}")


@dataclass
class AkshareEastmoneyProvider:
    name: str = "akshare_eastmoney"

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        validate_provider_name(self.name)
        try:
            import akshare as ak  # type: ignore
        except Exception as exc:
            raise RuntimeError("akshare is not installed in the yolos environment") from exc
        request = request.normalized()
        frames: list[pd.DataFrame] = []
        for symbol in request.symbols:
            raw = ak.stock_zh_a_hist(
                symbol=_strip_suffix(symbol),
                period="daily",
                start_date=request.start_date.replace("-", ""),
                end_date=request.end_date.replace("-", ""),
                adjust="qfq" if request.adjusted_flag in {"front", "qfq"} else "",
            )
            if isinstance(raw, pd.DataFrame) and not raw.empty:
                raw = raw.copy()
                raw["symbol"] = symbol
                frames.append(raw)
        data = normalize_market_frame(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), source=self.name, adjusted_flag=request.adjusted_flag, require_columns=False)
        return ProviderResult(provider=self.name, data=data)

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
        try:
            import akshare as ak  # type: ignore
        except Exception as exc:
            raise RuntimeError("akshare is not installed in the yolos environment") from exc
        frame = pd.DataFrame()
        if request.domain in {DataDomain.UNIVERSE_SNAPSHOT, DataDomain.VALUATION}:
            frame = ak.stock_zh_a_spot_em()
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                frame = frame.copy()
                frame["trade_date"] = request.end_date
        elif request.domain == DataDomain.INDUSTRY_CONCEPT:
            frame = _akshare_industry_members(ak, request)
        elif request.domain == DataDomain.LIMIT_STATUS:
            frame = _akshare_limit_status(ak, request)
        elif request.domain == DataDomain.MONEY_FLOW_HOTSPOT:
            frame = _akshare_money_flow(ak, request)
        else:
            raise RuntimeError(f"unsupported_domain: {self.name} does not support {request.domain}")
        data = normalize_domain_frame(frame, domain=request.domain, source=self.name, as_of_date=request.end_date, require_columns=False)
        return ProviderResult(provider=self.name, data=data)


@dataclass
class BaostockProvider:
    name: str = "baostock"

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        validate_provider_name(self.name)
        try:
            import baostock as bs  # type: ignore
        except Exception as exc:
            raise RuntimeError("baostock is not installed in the yolos environment") from exc
        request = request.normalized()
        login = bs.login()
        if getattr(login, "error_code", "1") != "0":
            raise RuntimeError(f"baostock login failed: {getattr(login, 'error_msg', '')}")
        rows: list[pd.DataFrame] = []
        try:
            with create_progress(total=len(request.symbols), desc="Baostock market_daily", unit="symbol", leave=False) as progress:
                for idx, symbol in enumerate(request.symbols, start=1):
                    if idx == 1 or idx % 50 == 0 or idx == len(request.symbols):
                        progress_write(f"baostock_market_daily={idx}/{len(request.symbols)} symbol={symbol}")
                    progress.set_description_str(f"Baostock market_daily {symbol}")
                    progress.update(1)
                    code = _to_baostock_code(symbol)
                    query = bs.query_history_k_data_plus(
                        code,
                        "date,code,open,high,low,close,volume,amount",
                        start_date=request.start_date,
                        end_date=request.end_date,
                        frequency="d",
                        adjustflag="2" if request.adjusted_flag in {"front", "qfq"} else "3",
                    )
                    data: list[list[Any]] = []
                    while getattr(query, "error_code", "1") == "0" and query.next():
                        data.append(query.get_row_data())
                    frame = pd.DataFrame(data, columns=["trade_date", "symbol", "open", "high", "low", "close", "volume", "amount"])
                    if not frame.empty:
                        frame["symbol"] = symbol
                        rows.append(frame)
        finally:
            bs.logout()
        data = normalize_market_frame(pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), source=self.name, adjusted_flag=request.adjusted_flag, require_columns=False)
        return ProviderResult(provider=self.name, data=data)

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
        if request.domain == DataDomain.TRADING_CALENDAR:
            frame = _fetch_baostock_trade_calendar_frame_with_timeout(
                start_date=request.start_date,
                end_date=request.end_date,
                exchange=request.exchange,
            )
        elif request.domain in {DataDomain.UNIVERSE_SNAPSHOT, DataDomain.SECURITY_STATUS}:
            frame = _fetch_baostock_stock_basic_frame_with_timeout(trade_date=request.end_date)
        else:
            raise RuntimeError(f"unsupported_domain: {self.name} does not support {request.domain}")
        data = normalize_domain_frame(frame, domain=request.domain, source=self.name, as_of_date=request.end_date, require_columns=False)
        return ProviderResult(provider=self.name, data=data)


@dataclass
class TushareHttpOptionalProvider:
    name: str = "tushare_http_optional"
    token: str = ""

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        validate_provider_name(self.name)
        token = str(self.token or os.environ.get("TUSHARE_TOKEN", "")).strip()
        if not token:
            raise RuntimeError("TUSHARE_TOKEN is not configured; tushare_http_optional is disabled")
        request = request.normalized()
        frames: list[pd.DataFrame] = []
        for symbol in request.symbols:
            response = requests.post(
                "http://api.tushare.pro",
                json={
                    "api_name": "daily",
                    "token": token,
                    "params": {
                        "ts_code": symbol,
                        "start_date": request.start_date.replace("-", ""),
                        "end_date": request.end_date.replace("-", ""),
                    },
                    "fields": "ts_code,trade_date,open,high,low,close,vol,amount",
                },
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") != 0:
                raise RuntimeError(payload.get("msg") or f"tushare error code {payload.get('code')}")
            data = payload.get("data") or {}
            frame = pd.DataFrame(data.get("items") or [], columns=data.get("fields") or [])
            if not frame.empty:
                frames.append(frame.rename(columns={"ts_code": "symbol", "vol": "volume"}))
        data = normalize_market_frame(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), source=self.name, adjusted_flag=request.adjusted_flag, require_columns=False)
        return ProviderResult(provider=self.name, data=data)

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
        token = str(self.token or os.environ.get("TUSHARE_TOKEN", "")).strip()
        if not token:
            raise RuntimeError("TUSHARE_TOKEN is not configured; tushare_http_optional is disabled")
        if request.domain == DataDomain.TRADING_CALENDAR:
            frame = self._post_tushare(
                token=token,
                api_name="trade_cal",
                params={"start_date": request.start_date.replace("-", ""), "end_date": request.end_date.replace("-", "")},
                fields="cal_date,is_open,exchange",
            ).rename(columns={"cal_date": "trade_date"})
        elif request.domain == DataDomain.UNIVERSE_SNAPSHOT:
            frame = self._post_tushare(
                token=token,
                api_name="stock_basic",
                params={"list_status": "L"},
                fields="ts_code,name,market,list_status,list_date,delist_date",
            ).rename(columns={"ts_code": "symbol", "market": "board"})
            frame["trade_date"] = request.end_date
        elif request.domain == DataDomain.VALUATION:
            frames = []
            for trade_date in _date_range_strings(request.start_date, request.end_date):
                frames.append(
                    self._post_tushare(
                        token=token,
                        api_name="daily_basic",
                        params={"trade_date": trade_date.replace("-", "")},
                        fields="ts_code,trade_date,total_mv,circ_mv,pe,pb,turnover_rate",
                    ).rename(columns={"ts_code": "symbol"})
                )
            frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        elif request.domain == DataDomain.LIMIT_STATUS:
            frames = []
            for trade_date in _date_range_strings(request.start_date, request.end_date):
                frames.append(
                    self._post_tushare(
                        token=token,
                        api_name="stk_limit",
                        params={"trade_date": trade_date.replace("-", "")},
                        fields="ts_code,trade_date,up_limit,down_limit",
                    ).rename(columns={"ts_code": "symbol"})
                )
            frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        else:
            raise RuntimeError(f"unsupported_domain: {self.name} does not support {request.domain}")
        data = normalize_domain_frame(frame, domain=request.domain, source=self.name, as_of_date=request.end_date, require_columns=False)
        return ProviderResult(provider=self.name, data=data)

    def _post_tushare(self, *, token: str, api_name: str, params: dict[str, Any], fields: str) -> pd.DataFrame:
        response = requests.post(
            "http://api.tushare.pro",
            json={"api_name": api_name, "token": token, "params": params, "fields": fields},
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError(payload.get("msg") or f"tushare error code {payload.get('code')}")
        data = payload.get("data") or {}
        return pd.DataFrame(data.get("items") or [], columns=data.get("fields") or [])


@dataclass
class SinaTencentRealtimeProvider:
    name: str = "sina_tencent_realtime"

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        validate_provider_name(self.name)
        raise RuntimeError("sina_tencent_realtime only supports realtime snapshots; daily refresh uses it as an optional same-day supplement in a later phase")

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        raise RuntimeError("sina_tencent_realtime only supports realtime supplement domains in a later phase")


@dataclass
class TencentFinanceProvider:
    name: str = "tencent_finance"

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        raise RuntimeError("tencent_finance is an optional valuation/realtime supplement; it does not provide formal daily bars")

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        request = request.normalized()
        if request.domain != DataDomain.VALUATION:
            raise RuntimeError(f"unsupported_domain: {self.name} does not support {request.domain}")
        symbols = tuple(request.symbols or ())
        if not symbols:
            return ProviderResult(provider=self.name, data=pd.DataFrame())
        query = ",".join(_to_tencent_simple_code(symbol) for symbol in symbols)
        response = requests.get(f"https://qt.gtimg.cn/q={query}")
        response.encoding = response.encoding or "gbk"
        rows: list[dict[str, Any]] = []
        for line in str(response.text or "").splitlines():
            parts = line.split("~")
            if len(parts) < 4:
                continue
            raw_code = parts[0].split("=", 1)[0].replace("v_s_", "").replace("v_", "").strip()
            symbol = _from_tencent_code(raw_code)
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": request.end_date,
                    "total_mv": float("nan"),
                    "circ_mv": float("nan"),
                    "pe": float("nan"),
                    "pb": float("nan"),
                    "turnover_rate": float("nan"),
                    "source": self.name,
                }
            )
        data = normalize_domain_frame(pd.DataFrame(rows), domain=request.domain, source=self.name, as_of_date=request.end_date, require_columns=False)
        return ProviderResult(provider=self.name, data=data)


@dataclass
class TonghuashunHotspotProvider:
    name: str = "tonghuashun_hotspot"

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        raise RuntimeError("tonghuashun_hotspot only supports optional concept/hotspot domains")

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        request = request.normalized()
        try:
            import akshare as ak  # type: ignore
        except Exception as exc:
            raise RuntimeError("akshare is required for tonghuashun_hotspot optional endpoints") from exc
        if request.domain == DataDomain.INDUSTRY_CONCEPT:
            frame = _ths_concept_frame(ak, request)
        elif request.domain == DataDomain.MONEY_FLOW_HOTSPOT:
            frame = _ths_hotspot_frame(ak, request)
        else:
            raise RuntimeError(f"unsupported_domain: {self.name} does not support {request.domain}")
        data = normalize_domain_frame(frame, domain=request.domain, source=self.name, as_of_date=request.end_date, require_columns=False)
        return ProviderResult(provider=self.name, data=data)


def build_default_providers(provider_plan: str = "default_free") -> list:
    plan = str(provider_plan or "default_free").strip().lower()
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
        return [EastmoneyEfinanceProvider(), AkshareEastmoneyProvider(), BaostockProvider(), SinaTencentRealtimeProvider()]
    if plan == "tushare_optional":
        return [TushareHttpOptionalProvider()]
    raise ValueError(f"Unsupported provider_plan: {provider_plan}")


def _to_baostock_code(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    if raw.endswith(".SH"):
        return f"sh.{raw[:6]}"
    if raw.endswith(".SZ"):
        return f"sz.{raw[:6]}"
    if raw.startswith(("5", "6", "9")):
        return f"sh.{raw[:6]}"
    return f"sz.{raw[:6]}"


def _to_tencent_simple_code(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    code = raw.split(".", 1)[0] if "." in raw else raw[-6:]
    exchange = raw.split(".", 1)[1] if "." in raw else ("SH" if code.startswith(("5", "6", "9")) else "SZ")
    prefix = "sh" if exchange == "SH" else "sz"
    return f"s_{prefix}{code}"


def _from_tencent_code(value: Any) -> str:
    raw = str(value or "").strip().lower()
    raw = raw.removeprefix("s_")
    if raw.startswith("sh"):
        return f"{raw[2:8].upper()}.SH"
    if raw.startswith("sz"):
        return f"{raw[2:8].upper()}.SZ"
    return str(value or "").strip().upper()


def _from_baostock_code(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw.startswith("sh."):
        return f"{raw[3:].upper()}.SH"
    if raw.startswith("sz."):
        return f"{raw[3:].upper()}.SZ"
    if raw.startswith("bj."):
        return f"{raw[3:].upper()}.BJ"
    return str(value or "").strip().upper()


def _baostock_stock_basic_worker(queue: Any, trade_date: str) -> None:
    try:
        import baostock as bs  # type: ignore

        login = bs.login()
        if getattr(login, "error_code", "1") != "0":
            queue.put({"status": "error", "error_type": "RuntimeError", "error": f"baostock login failed: {getattr(login, 'error_msg', '')}"})
            return
        try:
            frame = _baostock_stock_basic_frame(bs.query_stock_basic(), trade_date=trade_date)
        finally:
            bs.logout()
        queue.put({"status": "ok", "data": frame})
    except Exception as exc:
        queue.put({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})


def _baostock_trade_calendar_worker(queue: Any, start_date: str, end_date: str, exchange: str) -> None:
    try:
        import baostock as bs  # type: ignore

        login = bs.login()
        if getattr(login, "error_code", "1") != "0":
            queue.put({"status": "error", "error_type": "RuntimeError", "error": f"baostock login failed: {getattr(login, 'error_msg', '')}"})
            return
        try:
            query = bs.query_trade_dates(start_date=start_date, end_date=end_date)
            rows: list[list[Any]] = []
            while getattr(query, "error_code", "1") == "0" and query.next():
                rows.append(query.get_row_data())
            frame = pd.DataFrame(rows, columns=["trade_date", "is_open"])
            frame["exchange"] = exchange
        finally:
            bs.logout()
        queue.put({"status": "ok", "data": frame})
    except Exception as exc:
        queue.put({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})


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
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(5)
        raise TimeoutError(f"{timeout_label}: exceeded {int(timeout)} seconds")
    try:
        payload = payload_queue.get(timeout=1.0)
    except queue_module.Empty:
        if process.exitcode not in {0, None}:
            raise RuntimeError(f"{failure_label}_worker_failed: exitcode={process.exitcode}")
        raise RuntimeError(f"{failure_label}_worker_returned_no_payload")
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


def _baostock_stock_basic_frame(query: Any, *, trade_date: str) -> pd.DataFrame:
    fields = [str(item) for item in (getattr(query, "fields", None) or [])]
    rows: list[list[Any]] = []
    while getattr(query, "error_code", "1") == "0" and query.next():
        rows.append(query.get_row_data())
    frame = pd.DataFrame(rows, columns=fields) if fields else pd.DataFrame(rows)
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


def _date_range_strings(start_date: str, end_date: str) -> list[str]:
    return [pd.Timestamp(item).strftime("%Y-%m-%d") for item in pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")]


def _akshare_industry_members(ak: Any, request: DomainFetchRequest) -> pd.DataFrame:
    boards = ak.stock_board_industry_name_em()
    if not isinstance(boards, pd.DataFrame) or boards.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    name_col = "板块名称" if "板块名称" in boards.columns else "名称" if "名称" in boards.columns else boards.columns[0]
    for industry in boards[name_col].dropna().astype(str).head(200):
        try:
            members = ak.stock_board_industry_cons_em(symbol=industry)
        except Exception:
            continue
        if not isinstance(members, pd.DataFrame) or members.empty:
            continue
        code_col = "代码" if "代码" in members.columns else "symbol" if "symbol" in members.columns else members.columns[0]
        for code in members[code_col].dropna().astype(str):
            rows.append({"symbol": code, "trade_date": request.end_date, "industry": industry, "concept_tags": ""})
    return pd.DataFrame(rows)


def _akshare_limit_status(ak: Any, request: DomainFetchRequest) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for trade_date in _date_range_strings(request.start_date, request.end_date):
        try:
            up = ak.stock_zt_pool_em(date=trade_date.replace("-", ""))
        except Exception:
            up = pd.DataFrame()
        if isinstance(up, pd.DataFrame) and not up.empty:
            up = up.copy()
            up["trade_date"] = trade_date
            up["is_limit_up"] = True
            frames.append(up.rename(columns={"代码": "symbol", "最新价": "up_limit"}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _akshare_money_flow(ak: Any, request: DomainFetchRequest) -> pd.DataFrame:
    try:
        frame = ak.stock_sector_fund_flow_rank(indicator="今日")
    except Exception:
        frame = pd.DataFrame()
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        frame = frame.copy()
        frame["trade_date"] = request.end_date
        frame = frame.rename(columns={"名称": "hotspot_tags", "今日主力净流入-净额": "main_net_inflow"})
        frame["sector_rank"] = range(1, len(frame) + 1)
    return frame


def _ths_concept_frame(ak: Any, request: DomainFetchRequest) -> pd.DataFrame:
    boards = ak.stock_board_concept_name_ths()
    if not isinstance(boards, pd.DataFrame) or boards.empty:
        return pd.DataFrame()
    name_col = "概念名称" if "概念名称" in boards.columns else "名称" if "名称" in boards.columns else boards.columns[0]
    rows = [
        {
            "symbol": "HOTSPOT",
            "trade_date": request.end_date,
            "industry": "",
            "concept_tags": str(name),
            "source": "tonghuashun_hotspot",
        }
        for name in boards[name_col].dropna().astype(str).head(300)
    ]
    return pd.DataFrame(rows)


def _ths_hotspot_frame(ak: Any, request: DomainFetchRequest) -> pd.DataFrame:
    boards = ak.stock_board_concept_name_ths()
    if not isinstance(boards, pd.DataFrame) or boards.empty:
        return pd.DataFrame()
    name_col = "概念名称" if "概念名称" in boards.columns else "名称" if "名称" in boards.columns else boards.columns[0]
    rows: list[dict[str, Any]] = []
    for rank, name in enumerate(boards[name_col].dropna().astype(str).head(100), start=1):
        rows.append(
            {
                "symbol": "HOTSPOT",
                "trade_date": request.end_date,
                "main_net_inflow": float("nan"),
                "sector_rank": rank,
                "hotspot_tags": str(name),
                "source": "tonghuashun_hotspot",
            }
        )
    return pd.DataFrame(rows)
