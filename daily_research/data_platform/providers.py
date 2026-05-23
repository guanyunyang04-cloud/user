from __future__ import annotations

import os
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


def _strip_suffix(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    if "." in raw:
        return raw.split(".", 1)[0]
    if raw.startswith(("SH", "SZ", "BJ")) and raw[2:].isdigit():
        return raw[2:]
    return raw


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
        try:
            import baostock as bs  # type: ignore
        except Exception as exc:
            raise RuntimeError("baostock is not installed in the yolos environment") from exc
        login = bs.login()
        if getattr(login, "error_code", "1") != "0":
            raise RuntimeError(f"baostock login failed: {getattr(login, 'error_msg', '')}")
        try:
            if request.domain == DataDomain.TRADING_CALENDAR:
                query = bs.query_trade_dates(start_date=request.start_date, end_date=request.end_date)
                rows: list[list[Any]] = []
                while getattr(query, "error_code", "1") == "0" and query.next():
                    rows.append(query.get_row_data())
                frame = pd.DataFrame(rows, columns=["trade_date", "is_open"])
                frame["exchange"] = request.exchange
            elif request.domain in {DataDomain.UNIVERSE_SNAPSHOT, DataDomain.SECURITY_STATUS}:
                frame = _baostock_stock_basic_frame(bs.query_stock_basic(), trade_date=request.end_date)
            else:
                raise RuntimeError(f"unsupported_domain: {self.name} does not support {request.domain}")
        finally:
            bs.logout()
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


def build_default_providers(provider_plan: str = "default_free") -> list:
    plan = str(provider_plan or "default_free").strip().lower()
    if plan == "default_free":
        return [EastmoneyEfinanceProvider(), AkshareEastmoneyProvider(), BaostockProvider()]
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


def _from_baostock_code(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw.startswith("sh."):
        return f"{raw[3:].upper()}.SH"
    if raw.startswith("sz."):
        return f"{raw[3:].upper()}.SZ"
    if raw.startswith("bj."):
        return f"{raw[3:].upper()}.BJ"
    return str(value or "").strip().upper()


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
