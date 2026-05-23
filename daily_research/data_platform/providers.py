from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from daily_research.data_platform.contracts import FetchRequest, ProviderResult, normalize_market_frame, validate_provider_name


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
            for symbol in request.symbols:
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


@dataclass
class TushareHttpOptionalProvider:
    name: str = "tushare_http_optional"
    token: str = ""
    timeout_seconds: int = 30

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
                timeout=self.timeout_seconds,
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


@dataclass
class SinaTencentRealtimeProvider:
    name: str = "sina_tencent_realtime"

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        validate_provider_name(self.name)
        raise RuntimeError("sina_tencent_realtime only supports realtime snapshots; daily refresh uses it as an optional same-day supplement in a later phase")


def build_default_providers(provider_plan: str = "default_free") -> list:
    plan = str(provider_plan or "default_free").strip().lower()
    if plan == "default_free":
        return [EastmoneyEfinanceProvider(), AkshareEastmoneyProvider(), BaostockProvider()]
    if plan == "default_free_no_realtime":
        return [EastmoneyEfinanceProvider(), AkshareEastmoneyProvider(), BaostockProvider()]
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
