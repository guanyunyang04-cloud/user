"""Optional public-web and realtime supplement providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from quantlab.data.domains.contracts import (
    DataDomain,
    DomainFetchRequest,
    FetchRequest,
    ProviderResult,
    normalize_domain_frame,
    normalize_market_frame,
    validate_provider_name,
)
from quantlab.data.provider_symbols import (
    from_tencent_code as _from_tencent_code,
)
from quantlab.data.provider_symbols import (
    strip_suffix as _strip_suffix,
)
from quantlab.data.provider_symbols import (
    to_tencent_simple_code as _to_tencent_simple_code,
)


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
            data = normalize_domain_frame(
                frame, domain=request.domain, source=self.name, as_of_date=request.end_date, require_columns=False
            )
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
        data = normalize_market_frame(
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
            source=self.name,
            adjusted_flag=request.adjusted_flag,
            require_columns=False,
        )
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
        data = normalize_domain_frame(
            frame, domain=request.domain, source=self.name, as_of_date=request.end_date, require_columns=False
        )
        return ProviderResult(provider=self.name, data=data)


@dataclass
class SinaTencentRealtimeProvider:
    name: str = "sina_tencent_realtime"

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        validate_provider_name(self.name)
        raise RuntimeError(
            "sina_tencent_realtime only supports realtime snapshots; daily refresh uses it as an optional same-day supplement in a later phase"
        )

    def fetch_domain(self, request: DomainFetchRequest) -> ProviderResult:
        raise RuntimeError("sina_tencent_realtime only supports realtime supplement domains in a later phase")


@dataclass
class TencentFinanceProvider:
    name: str = "tencent_finance"

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        raise RuntimeError(
            "tencent_finance is an optional valuation/realtime supplement; it does not provide formal daily bars"
        )

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
        data = normalize_domain_frame(
            pd.DataFrame(rows),
            domain=request.domain,
            source=self.name,
            as_of_date=request.end_date,
            require_columns=False,
        )
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
        data = normalize_domain_frame(
            frame, domain=request.domain, source=self.name, as_of_date=request.end_date, require_columns=False
        )
        return ProviderResult(provider=self.name, data=data)


def _date_range_strings(start_date: str, end_date: str) -> list[str]:
    return [
        pd.Timestamp(item).strftime("%Y-%m-%d")
        for item in pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")
    ]


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
        code_col = (
            "代码" if "代码" in members.columns else "symbol" if "symbol" in members.columns else members.columns[0]
        )
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
