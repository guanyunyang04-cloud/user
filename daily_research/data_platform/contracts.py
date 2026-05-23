from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol

import numpy as np
import pandas as pd


STANDARD_MARKET_COLUMNS = [
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "source",
    "adjusted_flag",
]

PRICE_COLUMNS = ["open", "high", "low", "close"]
NUMERIC_MARKET_COLUMNS = [*PRICE_COLUMNS, "volume", "amount"]
TDX_FAMILY_PROVIDER_NAMES = frozenset({"tq", "tqcenter", "tdx", "pytdx", "mootdx"})


@dataclass(frozen=True)
class FetchRequest:
    symbols: tuple[str, ...]
    start_date: str
    end_date: str
    domain: str = "market_daily"
    adjusted_flag: str = "none"
    fields: tuple[str, ...] = tuple(STANDARD_MARKET_COLUMNS)

    def normalized(self) -> "FetchRequest":
        return FetchRequest(
            symbols=tuple(_normalize_symbol(item) for item in self.symbols if str(item or "").strip()),
            start_date=_normalize_date(self.start_date),
            end_date=_normalize_date(self.end_date),
            domain=str(self.domain or "market_daily"),
            adjusted_flag=str(self.adjusted_flag or "none"),
            fields=tuple(self.fields or tuple(STANDARD_MARKET_COLUMNS)),
        )


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    data: pd.DataFrame
    coverage_report: dict[str, Any] = field(default_factory=dict)
    error_report: list[dict[str, Any]] = field(default_factory=list)


class MarketProvider(Protocol):
    name: str

    def fetch_market_bars(self, request: FetchRequest) -> ProviderResult:
        ...


MarketProviderCallable = Callable[[FetchRequest], ProviderResult]


def validate_provider_name(provider_name: str, *, allow_tdx_family: bool = False) -> str:
    normalized = str(provider_name or "").strip().lower()
    if not normalized:
        raise ValueError("provider name cannot be empty")
    if normalized in TDX_FAMILY_PROVIDER_NAMES and not allow_tdx_family:
        raise ValueError(
            "TDX-family provider is disabled by default. "
            f"provider={provider_name}; use daily_research.data_platform.refresh_daily with non-TDX providers."
        )
    return normalized


def ensure_tdx_free_data_source(data_source: str, *, allow_legacy: bool = False) -> str:
    normalized = str(data_source or "lake").strip().lower()
    aliases = {
        "data_lake": "lake",
        "csv_imported_lake": "lake",
        "csv_lake": "lake",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in TDX_FAMILY_PROVIDER_NAMES and not allow_legacy:
        raise ValueError(
            "TDX-family data_source is no longer allowed in formal daily_research paths. "
            "Run `python -m daily_research.data_platform.refresh_daily ...` to update the lake, "
            "then pass `--data-source lake --lake-dataset-id <explicit_id>`."
        )
    return normalized


def market_business_dates(start_date: str, end_date: str) -> list[str]:
    start_ts = pd.Timestamp(_normalize_date(start_date))
    end_ts = pd.Timestamp(_normalize_date(end_date))
    if end_ts < start_ts:
        return []
    return [pd.Timestamp(item).strftime("%Y-%m-%d") for item in pd.bdate_range(start_ts, end_ts)]


def next_business_date(date_value: str) -> str:
    return (pd.Timestamp(_normalize_date(date_value)) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")


def latest_completed_business_date(
    reference_ts: pd.Timestamp | str | None = None,
    *,
    close_time: str = "15:05",
) -> str:
    now_ts = pd.Timestamp(reference_ts).tz_localize(None) if reference_ts is not None else pd.Timestamp.now().tz_localize(None)
    close_clock = pd.Timestamp(close_time).time()
    include_today = now_ts.time() >= close_clock
    offset = 0 if include_today else 1
    return (now_ts.normalize() - pd.offsets.BDay(offset)).strftime("%Y-%m-%d")


def normalize_market_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    adjusted_flag: str = "none",
    require_columns: bool = True,
) -> pd.DataFrame:
    provider = validate_provider_name(source)
    if frame is None or frame.empty:
        return pd.DataFrame(columns=STANDARD_MARKET_COLUMNS)
    working = frame.copy()
    rename_map: dict[str, str] = {}
    aliases = _column_aliases()
    lower_lookup = {str(column).strip().lower(): column for column in working.columns}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            if candidate.lower() in lower_lookup:
                rename_map[lower_lookup[candidate.lower()]] = canonical
                break
    working = working.rename(columns=rename_map)
    if "source" not in working.columns:
        working["source"] = provider
    if "adjusted_flag" not in working.columns:
        working["adjusted_flag"] = str(adjusted_flag or "none")
    required = {"symbol", "trade_date", *NUMERIC_MARKET_COLUMNS}
    missing = sorted(required - set(working.columns))
    if missing and require_columns:
        raise ValueError(f"provider_frame_schema_error: missing standard market columns {missing}")
    for column in missing:
        working[column] = np.nan
    working["symbol"] = working["symbol"].map(_normalize_symbol)
    working["trade_date"] = pd.to_datetime(working["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in NUMERIC_MARKET_COLUMNS:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working["source"] = working["source"].fillna(provider).astype(str).str.strip().str.lower().replace("", provider)
    working["adjusted_flag"] = working["adjusted_flag"].fillna(str(adjusted_flag or "none")).astype(str).str.strip().replace("", "none")
    out = working.loc[:, [column for column in STANDARD_MARKET_COLUMNS if column in working.columns]].copy()
    for column in STANDARD_MARKET_COLUMNS:
        if column not in out.columns:
            out[column] = "" if column in {"symbol", "trade_date", "source", "adjusted_flag"} else np.nan
    out = out[STANDARD_MARKET_COLUMNS]
    out = out.loc[out["symbol"].astype(str).str.len() > 0]
    out = out.loc[out["trade_date"].astype(str).str.lower() != "nat"]
    return out.sort_values(["trade_date", "symbol", "source"]).reset_index(drop=True)


def valid_market_rows(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series([], dtype=bool)
    data = normalize_market_frame(frame, source=str(frame["source"].iloc[0] if "source" in frame.columns and len(frame) else "unknown"), require_columns=False)
    price_positive = data[PRICE_COLUMNS].gt(0).all(axis=1)
    range_valid = data["high"].ge(data[["open", "close", "low"]].max(axis=1)) & data["low"].le(data[["open", "close", "high"]].min(axis=1))
    activity_valid = data[["volume", "amount"]].ge(0).all(axis=1)
    return (price_positive & range_valid & activity_valid).reindex(data.index, fill_value=False)


def coverage_report_for_frame(frame: pd.DataFrame, request: FetchRequest, *, provider: str) -> dict[str, Any]:
    dates = market_business_dates(request.start_date, request.end_date)
    expected_rows = int(len(request.symbols) * len(dates))
    row_count = int(len(frame))
    unique_symbols = int(frame["symbol"].nunique()) if "symbol" in frame.columns and not frame.empty else 0
    unique_dates = int(frame["trade_date"].nunique()) if "trade_date" in frame.columns and not frame.empty else 0
    valid_rows = int(valid_market_rows(frame).sum()) if not frame.empty else 0
    coverage_ratio = float(row_count / expected_rows) if expected_rows else 0.0
    return {
        "provider": str(provider),
        "expected_rows": expected_rows,
        "row_count": row_count,
        "valid_rows": valid_rows,
        "symbol_count": unique_symbols,
        "trade_date_count": unique_dates,
        "coverage_ratio": coverage_ratio,
        "status": "ok" if row_count > 0 and valid_rows > 0 else "empty",
    }


def _normalize_date(value: Any) -> str:
    if value is None or str(value).strip() == "":
        raise ValueError("date cannot be empty")
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _normalize_symbol(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return ""
    raw = raw.replace("SH.", "").replace("SZ.", "").replace("BJ.", "")
    if "." in raw:
        base, suffix = raw.rsplit(".", 1)
        suffix = suffix.upper()
        if suffix == "SS":
            suffix = "SH"
        if suffix in {"SH", "SZ", "BJ"}:
            return f"{base.zfill(6) if base.isdigit() and len(base) < 6 else base}.{suffix}"
        return raw
    if raw.startswith(("SH", "SZ", "BJ")) and len(raw) >= 8 and raw[2:].isdigit():
        return f"{raw[2:]}.{raw[:2]}"
    if raw.isdigit() and len(raw) == 6:
        if raw.startswith(("5", "6", "9")):
            return f"{raw}.SH"
        if raw.startswith(("4", "8", "92")):
            return f"{raw}.BJ"
        return f"{raw}.SZ"
    return raw


def _column_aliases() -> Mapping[str, Iterable[str]]:
    return {
        "symbol": ("symbol", "stock", "code", "ts_code", "股票代码", "证券代码"),
        "trade_date": ("trade_date", "date", "datetime", "日期", "交易日期"),
        "open": ("open", "Open", "开盘", "开盘价"),
        "high": ("high", "High", "最高", "最高价"),
        "low": ("low", "Low", "最低", "最低价"),
        "close": ("close", "Close", "收盘", "收盘价", "最新价"),
        "volume": ("volume", "Volume", "vol", "成交量"),
        "amount": ("amount", "Amount", "成交额"),
        "source": ("source", "provider", "数据源"),
        "adjusted_flag": ("adjusted_flag", "adjust", "复权"),
    }
