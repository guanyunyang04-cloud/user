"""Build v2 point-in-time Shanghai/Shenzhen mainboard daily datasets."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .baostock_source import BaostockSource, BaostockSourceConfig, BaostockSourceError
from .dataset_v2 import DAILY_SIZE_COLUMNS, DEFAULT_V2_SNAPSHOT_ROOT
from .size_source import (
    AKSHARE_CNINFO_RECONSTRUCTED_SOURCE,
    PROXY_AMOUNT_SOURCE,
    TUSHARE_DAILY_BASIC_SOURCE,
    daily_size_cache_path,
    daily_size_meta_path,
    daily_size_source_grade,
    fetch_tushare_daily_size_cache,
    load_cached_daily_size,
    normalize_cninfo_share_change_events,
    normalize_trade_dates,
    reconstruct_daily_size_from_share_events,
    standardize_proxy_amount_daily_size,
)
from .universe import is_sh_sz_mainboard_a_share


PIT_BIAS_STATEMENT = (
    "v2 constructs the universe date by date from Baostock point-in-time stock lists, "
    "security master dates, daily tradestatus, and daily isST flags. It is intended "
    "to reduce survivorship bias versus v1, while remaining limited by Baostock field "
    "coverage and historical data quality."
)
BAOSTOCK_DAILY_FIELDS = ("date", "code", "open", "high", "low", "close", "volume", "amount", "tradestatus", "isST")
BAOSTOCK_DAILY_METRICS_FIELDS = ("date", "code", "turn", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM")
SECURITY_MASTER_COLUMNS = [
    "code",
    "baostock_code",
    "name",
    "ipo_date",
    "out_date",
    "security_type",
    "status",
    "first_seen_date",
    "last_seen_date",
    "basic_error",
]
RAW_STOCK_LIST_COLUMNS = ["date", "code", "name_on_date", "query_all_trade_status"]
DAILY_BARS_COLUMNS = [*BAOSTOCK_DAILY_FIELDS, "source"]
DAILY_METRICS_COLUMNS = [*BAOSTOCK_DAILY_METRICS_FIELDS, "source"]
STOCK_INDUSTRY_COLUMNS = [
    "date",
    "code",
    "name_on_date",
    "industry",
    "industry_classification",
    "industry_update_date",
    "source",
]
DAILY_UNIVERSE_COLUMNS = [
    "date",
    "code",
    "name_on_date",
    "ipo_date",
    "out_date",
    "is_listed_on_date",
    "is_mainboard",
    "is_common_a_share",
    "is_st_on_date",
    "is_suspended_on_date",
    "has_bar",
    "is_tradeable",
    "reject_reason",
]
DAILY_STATUS_COLUMNS = ["date", "code", "has_bar", "is_suspended_like", "is_tradeable"]
SIZE_SOURCE_TUSHARE_DAILY_BASIC = "tushare_daily_basic"
SIZE_SOURCE_AKSHARE_CNINFO_RECONSTRUCTED = "akshare_cninfo_reconstructed"
SIZE_SOURCE_PROXY_AMOUNT = "proxy_amount"
SIZE_SOURCE_CHOICES = (
    SIZE_SOURCE_TUSHARE_DAILY_BASIC,
    SIZE_SOURCE_AKSHARE_CNINFO_RECONSTRUCTED,
    SIZE_SOURCE_PROXY_AMOUNT,
)


@dataclass(frozen=True)
class PitBuildConfig:
    output_root: Path = DEFAULT_V2_SNAPSHOT_ROOT
    start_date: str = "2016-01-01"
    end_date: str = "2026-06-01"
    symbols: tuple[str, ...] = ()
    max_symbols: int = 0
    command: str = "build"
    snapshot_id: str = ""
    year: int = 0
    trade_dates: tuple[str, ...] = ()
    force_refresh: bool = False
    request_interval_seconds: float = 0.0
    discovery_mode: str = "stock-basic"
    include_industry: bool = False
    industry_frequency: str = "daily"
    include_metrics: bool = False
    include_size: bool = False
    size_source: str = SIZE_SOURCE_TUSHARE_DAILY_BASIC
    size_token: str | None = None


def _cache_root(output_root: Path) -> Path:
    return output_root / "cache"


def _stock_list_cache_path(output_root: Path, year: int) -> Path:
    return _cache_root(output_root) / "daily_stock_lists" / f"year={year}.parquet"


def _daily_bars_cache_path(output_root: Path, year: int) -> Path:
    return _cache_root(output_root) / "daily_bars" / f"year={year}.parquet"


def _daily_metrics_cache_path(
    output_root: Path,
    year: int,
    config: PitBuildConfig | None = None,
    start_date: str = "",
    end_date: str = "",
) -> Path:
    root = _cache_root(output_root) / "daily_metrics"
    scope = _sample_cache_scope(config, year=year, start_date=start_date, end_date=end_date)
    if scope:
        return root / "samples" / scope / f"year={year}.parquet"
    return root / f"year={year}.parquet"


def _stock_industry_cache_path(output_root: Path, year: int) -> Path:
    return _cache_root(output_root) / "stock_industry" / f"year={year}.parquet"


def _security_master_cache_path(output_root: Path) -> Path:
    return _cache_root(output_root) / "security_master.parquet"


def _stock_basic_all_cache_path(output_root: Path) -> Path:
    return _cache_root(output_root) / "stock_basic_all.parquet"


def _trade_dates_cache_path(output_root: Path) -> Path:
    return _cache_root(output_root) / "trade_dates.parquet"


def _trade_dates_meta_path(output_root: Path) -> Path:
    return _cache_root(output_root) / "cache_meta" / "trade_dates.json"


def _progress_path(output_root: Path, year: int) -> Path:
    return _cache_root(output_root) / "progress" / f"year={year}.json"


def _daily_metrics_progress_path(
    output_root: Path,
    year: int,
    config: PitBuildConfig | None = None,
    start_date: str = "",
    end_date: str = "",
) -> Path:
    root = _cache_root(output_root) / "progress" / "daily_metrics"
    scope = _sample_cache_scope(config, year=year, start_date=start_date, end_date=end_date)
    if scope:
        return root / "samples" / scope / f"year={year}.json"
    return root / f"year={year}.json"


def _monthly_stock_list_cache_path(output_root: Path, year: int, month: int) -> Path:
    return _cache_root(output_root) / "daily_stock_lists" / "parts" / f"year={year}" / f"month={month:02d}.parquet"


def _daily_bars_part_cache_path(output_root: Path, year: int, batch_index: int) -> Path:
    return _cache_root(output_root) / "daily_bars" / "parts" / f"year={year}" / f"batch={batch_index:04d}.parquet"


def _daily_metrics_part_cache_path(
    output_root: Path,
    year: int,
    batch_index: int,
    config: PitBuildConfig | None = None,
    start_date: str = "",
    end_date: str = "",
) -> Path:
    return _daily_metrics_parts_dir(output_root, year, config=config, start_date=start_date, end_date=end_date) / f"batch={batch_index:04d}.parquet"


def _monthly_stock_industry_cache_path(output_root: Path, year: int, month: int) -> Path:
    return _cache_root(output_root) / "stock_industry" / "parts" / f"year={year}" / f"month={month:02d}.parquet"


def _stock_list_meta_path(output_root: Path, year: int) -> Path:
    return _cache_root(output_root) / "cache_meta" / "daily_stock_lists" / f"year={year}.json"


def _daily_bars_meta_path(output_root: Path, year: int) -> Path:
    return _cache_root(output_root) / "cache_meta" / "daily_bars" / f"year={year}.json"


def _daily_metrics_meta_path(
    output_root: Path,
    year: int,
    config: PitBuildConfig | None = None,
    start_date: str = "",
    end_date: str = "",
) -> Path:
    root = _cache_root(output_root) / "cache_meta" / "daily_metrics"
    scope = _sample_cache_scope(config, year=year, start_date=start_date, end_date=end_date)
    if scope:
        return root / "samples" / scope / f"year={year}.json"
    return root / f"year={year}.json"


def _daily_metrics_parts_dir(
    output_root: Path,
    year: int,
    config: PitBuildConfig | None = None,
    start_date: str = "",
    end_date: str = "",
) -> Path:
    root = _cache_root(output_root) / "daily_metrics" / "parts"
    scope = _sample_cache_scope(config, year=year, start_date=start_date, end_date=end_date)
    if scope:
        return root / "samples" / scope / f"year={year}"
    return root / f"year={year}"


def _sample_cache_scope(config: PitBuildConfig | None, *, year: int, start_date: str, end_date: str) -> str:
    if config is None or (not config.symbols and config.max_symbols <= 0):
        return ""
    payload = {
        "year": year,
        "start_date": start_date,
        "end_date": end_date,
        "symbols": sorted(config.symbols),
        "max_symbols": int(config.max_symbols),
        "discovery_mode": config.discovery_mode,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"sample={digest}"


def _stock_industry_meta_path(output_root: Path, year: int) -> Path:
    return _cache_root(output_root) / "cache_meta" / "stock_industry" / f"year={year}.json"


def _config_cache_signature(config: PitBuildConfig, *, year: int, start_date: str, end_date: str) -> dict[str, Any]:
    return {
        "year": year,
        "start_date": start_date,
        "end_date": end_date,
        "symbols": sorted(config.symbols),
        "max_symbols": int(config.max_symbols),
        "discovery_mode": config.discovery_mode,
    }


def _industry_cache_signature(config: PitBuildConfig, *, year: int, start_date: str, end_date: str) -> dict[str, Any]:
    return {
        **_config_cache_signature(config, year=year, start_date=start_date, end_date=end_date),
        "industry_frequency": config.industry_frequency,
    }


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _is_cache_compatible(meta: dict[str, Any], config: PitBuildConfig, *, year: int, start_date: str, end_date: str) -> bool:
    if not meta or not meta.get("complete"):
        return False
    if int(meta.get("year", -1)) != int(year):
        return False
    cached_start = pd.Timestamp(meta.get("start_date", "1900-01-01"))
    cached_end = pd.Timestamp(meta.get("end_date", "1900-01-01"))
    if cached_start > pd.Timestamp(start_date) or cached_end < pd.Timestamp(end_date):
        return False
    if not _symbols_cache_compatible(meta.get("symbols", []), config.symbols):
        return False
    if not _max_symbols_cache_compatible(int(meta.get("max_symbols", 0) or 0), int(config.max_symbols)):
        return False
    if str(meta.get("discovery_mode", "")) != config.discovery_mode:
        return False
    return True


def _symbols_cache_compatible(cached_symbols: Any, requested_symbols: tuple[str, ...]) -> bool:
    cached = set(str(symbol) for symbol in (cached_symbols or []))
    requested = set(str(symbol) for symbol in requested_symbols)
    if not requested:
        return not cached
    if not cached:
        return True
    return requested.issubset(cached)


def _max_symbols_cache_compatible(cached_max_symbols: int, requested_max_symbols: int) -> bool:
    if requested_max_symbols <= 0:
        return cached_max_symbols <= 0
    if cached_max_symbols <= 0:
        return True
    return cached_max_symbols >= requested_max_symbols


def _codes_cache_compatible(cached_codes: Any, expected_codes: list[str]) -> bool:
    cached = set(str(code) for code in (cached_codes or []))
    expected = set(str(code) for code in expected_codes)
    if not expected:
        return True
    if not cached:
        return False
    return expected.issubset(cached)


def _is_industry_cache_compatible(meta: dict[str, Any], config: PitBuildConfig, *, year: int, start_date: str, end_date: str) -> bool:
    if not _is_cache_compatible(meta, config, year=year, start_date=start_date, end_date=end_date):
        return False
    return str(meta.get("industry_frequency", "daily")) == config.industry_frequency


def _cache_compat_summary(meta: dict[str, Any], compatible: bool) -> dict[str, Any]:
    if compatible:
        return {"cache_meta_compatible": 1}
    if not meta:
        return {"cache_meta_compatible": 0, "cache_meta_reason": "missing_or_invalid"}
    return {
        "cache_meta_compatible": 0,
        "cache_meta_reason": "signature_mismatch_or_incomplete",
        "cache_meta": meta,
    }


def _date_span_from_series(dates: pd.Series, fallback_start: str, fallback_end: str) -> tuple[str, str]:
    if len(dates) == 0:
        return fallback_start, fallback_end
    values = pd.to_datetime(dates)
    return str(values.min().date()), str(values.max().date())


def _is_date_cache_compatible(meta: dict[str, Any], start_date: str, end_date: str) -> bool:
    if not meta or not meta.get("complete"):
        return False
    cached_start = pd.Timestamp(meta.get("start_date", "1900-01-01"))
    cached_end = pd.Timestamp(meta.get("end_date", "1900-01-01"))
    return cached_start <= pd.Timestamp(start_date) and cached_end >= pd.Timestamp(end_date)


def _read_progress(output_root: Path, year: int) -> dict[str, Any]:
    path = _progress_path(output_root, year)
    if not path.exists():
        return _new_progress(year)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_progress(output_root: Path, year: int, progress: dict[str, Any]) -> None:
    path = _progress_path(output_root, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_daily_metrics_progress(config: PitBuildConfig, year: int, start_date: str, end_date: str) -> dict[str, Any]:
    path = _daily_metrics_progress_path(config.output_root, year, config=config, start_date=start_date, end_date=end_date)
    if not path.exists():
        return _new_progress(year)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_daily_metrics_progress(config: PitBuildConfig, year: int, start_date: str, end_date: str, progress: dict[str, Any]) -> None:
    path = _daily_metrics_progress_path(config.output_root, year, config=config, start_date=start_date, end_date=end_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _new_progress(year: int, signature: dict[str, Any] | None = None) -> dict[str, Any]:
    progress = {
        "year": year,
        "completed_stock_dates": [],
        "completed_bar_codes": [],
        "completed_metric_codes": [],
        "completed_industry_dates": [],
        "failures": [],
    }
    if signature is not None:
        progress["signature"] = signature
    return progress


def _progress_compatible(progress: dict[str, Any], config: PitBuildConfig, *, year: int, start_date: str, end_date: str) -> bool:
    return progress.get("signature") == _config_cache_signature(config, year=year, start_date=start_date, end_date=end_date)


def _baostock_to_std_code(code: str) -> str:
    market, symbol = str(code).split(".", 1)
    suffix = {"sh": "SH", "sz": "SZ"}.get(market.lower(), market.upper())
    return f"{symbol}.{suffix}"


def _std_to_baostock_code(code: str) -> str:
    symbol, suffix = str(code).upper().split(".", 1)
    market = {"SH": "sh", "SZ": "sz"}[suffix]
    return f"{market}.{symbol}"


def _normalize_date(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return str(pd.to_datetime(text).date())


def _years_between(start_date: str, end_date: str) -> list[int]:
    start = pd.Timestamp(start_date).year
    end = pd.Timestamp(end_date).year
    return list(range(start, end + 1))


def _year_bounds(year: int, start_date: str, end_date: str) -> tuple[str, str]:
    start = max(pd.Timestamp(start_date), pd.Timestamp(year=year, month=1, day=1))
    end = min(pd.Timestamp(end_date), pd.Timestamp(year=year, month=12, day=31))
    return str(start.date()), str(end.date())


def _read_parquet_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _filter_dates(frame: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if frame.empty or "date" not in frame.columns:
        return frame
    dates = pd.to_datetime(frame["date"])
    mask = (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))
    return frame.loc[mask].reset_index(drop=True)


def _trade_dates(source: BaostockSource, config: PitBuildConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    path = _trade_dates_cache_path(config.output_root)
    meta_path = _trade_dates_meta_path(config.output_root)
    meta = _read_json_or_empty(meta_path)
    compatible = _is_date_cache_compatible(meta, config.start_date, config.end_date)
    cache_hit = int(path.exists() and not config.force_refresh and compatible)
    if cache_hit:
        cached = pd.read_parquet(path)
        return _filter_dates(cached, config.start_date, config.end_date), {"trade_dates_cache_hit": 1, "trade_dates_cache_miss": 0}
    if path.exists() and not compatible and not config.force_refresh:
        path.unlink(missing_ok=True)

    raw = source.query_trade_dates(config.start_date, config.end_date)
    if raw.empty:
        frame = pd.DataFrame(columns=["date", "is_trading_day"])
    else:
        frame = raw.rename(columns={"calendar_date": "date"}).copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["is_trading_day"] = frame["is_trading_day"].astype(str) == "1"
        frame = frame.loc[frame["is_trading_day"], ["date", "is_trading_day"]].reset_index(drop=True)
    _write_parquet(path, frame)
    _write_json(
        meta_path,
        {
            "start_date": config.start_date,
            "end_date": config.end_date,
            "complete": True,
            "row_count": int(len(frame)),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return frame, {"trade_dates_cache_hit": 0, "trade_dates_cache_miss": 1}


def _stock_basic_all(source: BaostockSource, config: PitBuildConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    path = _stock_basic_all_cache_path(config.output_root)
    if path.exists() and not config.force_refresh:
        return pd.read_parquet(path), {"stock_basic_all_cache_hit": 1, "stock_basic_all_cache_miss": 0}
    raw = source.query_stock_basic()
    frame = normalize_stock_basic(raw)
    _write_parquet(path, frame)
    return frame, {"stock_basic_all_cache_hit": 0, "stock_basic_all_cache_miss": 1}


def normalize_stock_basic(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=SECURITY_MASTER_COLUMNS)
    frame = raw.copy()
    frame["code"] = frame["code"].map(_baostock_to_std_code)
    frame["baostock_code"] = raw["code"].astype(str)
    frame["name"] = frame["code_name"].astype(str)
    frame["ipo_date"] = frame["ipoDate"].map(_normalize_date)
    frame["out_date"] = frame["outDate"].map(_normalize_date)
    frame["security_type"] = frame["type"].astype(str)
    frame["status"] = frame["status"].astype(str)
    frame["first_seen_date"] = ""
    frame["last_seen_date"] = ""
    frame["basic_error"] = ""
    return frame[SECURITY_MASTER_COLUMNS]


def derive_stock_lists_from_basic(
    security_master: pd.DataFrame,
    dates: pd.Series,
    *,
    explicit_symbols: set[str] | None = None,
) -> pd.DataFrame:
    if security_master.empty or len(dates) == 0:
        return pd.DataFrame(columns=RAW_STOCK_LIST_COLUMNS)
    base = security_master.copy()
    base = base[base["code"].map(is_sh_sz_mainboard_a_share)]
    base = base[base["security_type"].astype(str) == "1"]
    if explicit_symbols is not None:
        base = base[base["code"].isin(explicit_symbols)]
    if base.empty:
        return pd.DataFrame(columns=RAW_STOCK_LIST_COLUMNS)
    ipo = pd.to_datetime(base["ipo_date"], errors="coerce")
    out = pd.to_datetime(base["out_date"], errors="coerce")
    parts: list[pd.DataFrame] = []
    for date in pd.to_datetime(dates):
        mask = ipo.notna() & (ipo <= date) & (out.isna() | (date < out))
        daily = base.loc[mask, ["code", "name"]].copy()
        if daily.empty:
            continue
        daily["date"] = pd.Timestamp(date)
        daily["name_on_date"] = daily["name"]
        daily["query_all_trade_status"] = "1"
        parts.append(daily[RAW_STOCK_LIST_COLUMNS])
    return _concat(parts, RAW_STOCK_LIST_COLUMNS).drop_duplicates(["date", "code"]).reset_index(drop=True)


def discover_daily_stock_lists(
    source: BaostockSource,
    dates: pd.Series,
    *,
    config: PitBuildConfig,
    year: int,
    explicit_symbols: set[str] | None,
    security_master: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _stock_list_cache_path(config.output_root, year)
    meta_path = _stock_list_meta_path(config.output_root, year)
    meta = _read_json_or_empty(meta_path)
    compatible = _is_cache_compatible(meta, config, year=year, start_date=config.start_date, end_date=config.end_date)
    if path.exists() and not config.force_refresh:
        if compatible:
            frame = pd.read_parquet(path)
            frame = _filter_dates(frame, config.start_date, config.end_date)
            if explicit_symbols is not None:
                frame = frame[frame["code"].isin(explicit_symbols)].reset_index(drop=True)
            return frame, {
                "stock_list_cache_hit": 1,
                "stock_list_cache_miss": 0,
                "empty_stock_list_dates": [],
                **_cache_compat_summary(meta, compatible),
            }
        path.unlink(missing_ok=True)

    parts: list[pd.DataFrame] = []
    empty_dates: list[str] = []
    if config.discovery_mode == "stock-basic":
        if security_master is None:
            security_master, _ = _stock_basic_all(source, config)
        output = derive_stock_lists_from_basic(security_master, dates, explicit_symbols=explicit_symbols)
        _write_parquet(path, output)
        _write_json(
            meta_path,
            {
                **_config_cache_signature(config, year=year, start_date=config.start_date, end_date=config.end_date),
                "complete": True,
                "row_count": int(len(output)),
                "code_count": int(output["code"].nunique()) if not output.empty else 0,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        return output, {
            "stock_list_cache_hit": 0,
            "stock_list_cache_miss": 1,
            "empty_stock_list_dates": [],
            "discovery_mode": "stock-basic",
            **_cache_compat_summary(meta, compatible),
        }

    if config.command == "dry-run" and explicit_symbols is not None:
        for date in dates:
            date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
            for code in sorted(explicit_symbols):
                parts.append(
                    pd.DataFrame(
                        [
                            {
                                "date": pd.Timestamp(date),
                                "code": code,
                                "name_on_date": "",
                                "query_all_trade_status": "1",
                            }
                        ]
                    )
                )
        output = _concat(parts, RAW_STOCK_LIST_COLUMNS).drop_duplicates(["date", "code"])
        _write_parquet(path, output)
        _write_json(
            meta_path,
            {
                **_config_cache_signature(config, year=year, start_date=config.start_date, end_date=config.end_date),
                "complete": True,
                "row_count": int(len(output)),
                "code_count": int(output["code"].nunique()) if not output.empty else 0,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        return output, {
            "stock_list_cache_hit": 0,
            "stock_list_cache_miss": 1,
            "empty_stock_list_dates": [],
            "dry_run_symbol_mode": 1,
            **_cache_compat_summary(meta, compatible),
        }

    signature = _config_cache_signature(config, year=year, start_date=config.start_date, end_date=config.end_date)
    progress = _new_progress(year, signature) if config.force_refresh else _read_progress(config.output_root, year)
    resume_parts = not config.force_refresh and _progress_compatible(progress, config, year=year, start_date=config.start_date, end_date=config.end_date)
    if not resume_parts:
        progress = _new_progress(year, signature)
    completed_dates = set(progress.get("completed_stock_dates", []))
    touched_months: set[int] = set()
    for date in dates:
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        month_path = _monthly_stock_list_cache_path(config.output_root, year, pd.Timestamp(date).month)
        if date_str in completed_dates and month_path.exists():
            continue
        raw = source.query_all_stock(day=date_str)
        if raw.empty:
            empty_dates.append(date_str)
            continue
        frame = raw.copy()
        frame["date"] = pd.Timestamp(date)
        frame["code"] = frame["code"].map(_baostock_to_std_code)
        frame["name_on_date"] = frame["code_name"].astype(str)
        frame["query_all_trade_status"] = frame["tradeStatus"].astype(str)
        frame = frame[RAW_STOCK_LIST_COLUMNS]
        frame = frame[frame["code"].map(is_sh_sz_mainboard_a_share)]
        if explicit_symbols is not None:
            frame = frame[frame["code"].isin(explicit_symbols)]
        parts.append(frame)
        touched_months.add(pd.Timestamp(date).month)
        existing_month = _read_parquet_or_empty(month_path, RAW_STOCK_LIST_COLUMNS) if resume_parts else pd.DataFrame(columns=RAW_STOCK_LIST_COLUMNS)
        month_output = _concat([existing_month, frame], RAW_STOCK_LIST_COLUMNS).drop_duplicates(["date", "code"])
        _write_parquet(month_path, month_output)
        progress["completed_stock_dates"] = sorted(set(progress.get("completed_stock_dates", [])) | {date_str})
        _write_progress(config.output_root, year, progress)

    months_to_read = sorted(touched_months) if config.force_refresh else list(range(1, 13))
    month_parts = [
        (
            _read_parquet_or_empty(_monthly_stock_list_cache_path(config.output_root, year, month), RAW_STOCK_LIST_COLUMNS)
            if resume_parts
            else pd.DataFrame(columns=RAW_STOCK_LIST_COLUMNS)
        )
        for month in months_to_read
    ]
    output = _concat([*month_parts, *parts], RAW_STOCK_LIST_COLUMNS).drop_duplicates(["date", "code"])
    output = _filter_dates(output, config.start_date, config.end_date)
    _write_parquet(path, output)
    _write_json(
        meta_path,
        {
            **_config_cache_signature(config, year=year, start_date=config.start_date, end_date=config.end_date),
            "complete": True,
            "row_count": int(len(output)),
            "code_count": int(output["code"].nunique()) if not output.empty else 0,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return output, {
        "stock_list_cache_hit": 0,
        "stock_list_cache_miss": 1,
        "empty_stock_list_dates": empty_dates,
        **_cache_compat_summary(meta, compatible),
    }


def update_security_master(
    source: BaostockSource,
    stock_lists: pd.DataFrame,
    *,
    config: PitBuildConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _security_master_cache_path(config.output_root)
    existing = _read_parquet_or_empty(path, SECURITY_MASTER_COLUMNS)
    existing_codes = set(existing["code"].dropna().astype(str).tolist()) if not existing.empty else set()
    seen = _seen_ranges(stock_lists)
    required_codes = set(seen["code"].astype(str).tolist())
    missing_codes = sorted(required_codes - existing_codes)
    new_parts: list[pd.DataFrame] = []

    for code in missing_codes:
        try:
            raw = source.query_stock_basic(_std_to_baostock_code(code))
        except BaostockSourceError as exc:
            raw = pd.DataFrame()
            basic_error = str(exc)
        else:
            basic_error = ""

        if raw.empty:
            new_parts.append(
                pd.DataFrame(
                    [
                        {
                            "code": code,
                            "baostock_code": _std_to_baostock_code(code),
                            "name": "",
                            "ipo_date": "",
                            "out_date": "",
                            "security_type": "",
                            "status": "",
                            "first_seen_date": "",
                            "last_seen_date": "",
                            "basic_error": basic_error or "empty_stock_basic",
                        }
                    ]
                )
            )
            continue

        item = raw.iloc[0].to_dict()
        new_parts.append(
            pd.DataFrame(
                [
                    {
                        "code": code,
                        "baostock_code": item.get("code", ""),
                        "name": item.get("code_name", ""),
                        "ipo_date": _normalize_date(item.get("ipoDate", "")),
                        "out_date": _normalize_date(item.get("outDate", "")),
                        "security_type": str(item.get("type", "") or ""),
                        "status": str(item.get("status", "") or ""),
                        "first_seen_date": "",
                        "last_seen_date": "",
                        "basic_error": "",
                    }
                ]
            )
        )

    combined = _concat([existing, *new_parts], SECURITY_MASTER_COLUMNS).drop_duplicates("code", keep="last")
    if not combined.empty and not seen.empty:
        combined = combined.drop(columns=["first_seen_date", "last_seen_date"], errors="ignore").merge(seen, on="code", how="left")
        combined["first_seen_date"] = combined["first_seen_date"].fillna("")
        combined["last_seen_date"] = combined["last_seen_date"].fillna("")
        combined = combined[SECURITY_MASTER_COLUMNS]
    _write_parquet(path, combined)
    return combined[combined["code"].isin(required_codes)].reset_index(drop=True), {
        "security_master_cache_existing": len(existing_codes),
        "security_master_new_queries": len(missing_codes),
        "basic_error_count": int((combined["basic_error"].fillna("") != "").sum()) if not combined.empty else 0,
    }


def update_security_master_from_basic(stock_lists: pd.DataFrame, security_master_all: pd.DataFrame, *, config: PitBuildConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _security_master_cache_path(config.output_root)
    seen = _seen_ranges(stock_lists)
    required_codes = set(seen["code"].astype(str).tolist())
    selected = security_master_all[security_master_all["code"].isin(required_codes)].copy()
    if not selected.empty and not seen.empty:
        selected = selected.drop(columns=["first_seen_date", "last_seen_date"], errors="ignore").merge(seen, on="code", how="left")
        selected["first_seen_date"] = selected["first_seen_date"].fillna("")
        selected["last_seen_date"] = selected["last_seen_date"].fillna("")
        selected = selected[SECURITY_MASTER_COLUMNS]
    existing = _read_parquet_or_empty(path, SECURITY_MASTER_COLUMNS)
    combined = _concat([existing, selected], SECURITY_MASTER_COLUMNS).drop_duplicates("code", keep="last")
    _write_parquet(path, combined)
    return selected.reset_index(drop=True), {
        "security_master_cache_existing": len(existing) if not existing.empty else 0,
        "security_master_new_queries": 0,
        "basic_error_count": int((selected["basic_error"].fillna("") != "").sum()) if not selected.empty else 0,
        "security_master_mode": "stock-basic-all",
    }


def _seen_ranges(stock_lists: pd.DataFrame) -> pd.DataFrame:
    if stock_lists.empty:
        return pd.DataFrame(columns=["code", "first_seen_date", "last_seen_date"])
    grouped = stock_lists.groupby("code")["date"].agg(["min", "max"]).reset_index()
    grouped["first_seen_date"] = pd.to_datetime(grouped["min"]).dt.strftime("%Y-%m-%d")
    grouped["last_seen_date"] = pd.to_datetime(grouped["max"]).dt.strftime("%Y-%m-%d")
    return grouped[["code", "first_seen_date", "last_seen_date"]]


def fetch_daily_bars(
    source: BaostockSource,
    codes: list[str],
    *,
    config: PitBuildConfig,
    year: int,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, int]]:
    path = _daily_bars_cache_path(config.output_root, year)
    meta_path = _daily_bars_meta_path(config.output_root, year)
    meta = _read_json_or_empty(meta_path)
    compatible = _is_cache_compatible(meta, config, year=year, start_date=start_date, end_date=end_date)
    expected_codes = sorted(set(codes))
    if compatible and not _codes_cache_compatible(meta.get("codes", []), expected_codes):
        compatible = False
    if path.exists() and not config.force_refresh:
        if compatible:
            frame = pd.read_parquet(path)
            frame = _filter_dates(frame, start_date, end_date)
            if config.symbols:
                frame = frame[frame["code"].isin(set(config.symbols))].reset_index(drop=True)
            return frame, [], {
                "daily_bars_cache_hit": 1,
                "daily_bars_cache_miss": 0,
                **_cache_compat_summary(meta, compatible),
            }
        path.unlink(missing_ok=True)

    parts: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    signature = _config_cache_signature(config, year=year, start_date=start_date, end_date=end_date)
    progress = _new_progress(year, signature) if config.force_refresh else _read_progress(config.output_root, year)
    resume_parts = not config.force_refresh and _progress_compatible(progress, config, year=year, start_date=start_date, end_date=end_date)
    if not resume_parts:
        progress = _new_progress(year, signature)
    completed_codes = set(progress.get("completed_bar_codes", []))
    batch_size = 50
    for batch_index, start in enumerate(range(0, len(codes), batch_size)):
        batch_codes = codes[start : start + batch_size]
        part_path = _daily_bars_part_cache_path(config.output_root, year, batch_index)
        existing_part = (
            _read_parquet_or_empty(part_path, DAILY_BARS_COLUMNS)
            if resume_parts
            else pd.DataFrame(columns=DAILY_BARS_COLUMNS)
        )
        batch_parts: list[pd.DataFrame] = [existing_part] if not existing_part.empty else []
        for code in batch_codes:
            if code in completed_codes and not existing_part.empty and code in set(existing_part["code"].astype(str)):
                continue
            progress["active_bar_code"] = code
            _write_progress(config.output_root, year, progress)
            try:
                raw = source.query_daily_bars(_std_to_baostock_code(code), start_date, end_date)
            except BaostockSourceError as exc:
                failure = {"year": year, "code": code, "kind": "daily_fetch_error", "error": str(exc)}
                failures.append(failure)
                progress["failures"] = [*progress.get("failures", []), failure]
                progress["active_bar_code"] = ""
                _write_progress(config.output_root, year, progress)
                continue
            if raw.empty:
                failure = {"year": year, "code": code, "kind": "empty_daily_bars"}
                failures.append(failure)
                progress["failures"] = [*progress.get("failures", []), failure]
                progress["active_bar_code"] = ""
                _write_progress(config.output_root, year, progress)
                continue
            frame = normalize_daily_bars(raw)
            batch_parts.append(frame)
            batch_output = _concat(batch_parts, DAILY_BARS_COLUMNS).drop_duplicates(["date", "code"])
            if not batch_output.empty:
                _write_parquet(part_path, batch_output)
            progress["completed_bar_codes"] = sorted(set(progress.get("completed_bar_codes", [])) | {code})
            progress["active_bar_code"] = ""
            _write_progress(config.output_root, year, progress)
        batch_output = _concat(batch_parts, DAILY_BARS_COLUMNS).drop_duplicates(["date", "code"])
        if not batch_output.empty:
            _write_parquet(part_path, batch_output)
            parts.append(batch_output)

    part_paths = (
        []
        if not resume_parts
        else sorted((_cache_root(config.output_root) / "daily_bars" / "parts" / f"year={year}").glob("batch=*.parquet"))
    )
    part_frames = [_read_parquet_or_empty(part_path, DAILY_BARS_COLUMNS) for part_path in part_paths]
    output = _concat([*part_frames, *parts], DAILY_BARS_COLUMNS).drop_duplicates(["date", "code"])
    output = _filter_dates(output, start_date, end_date)
    if codes:
        output = output[output["code"].isin(set(codes))].reset_index(drop=True)
    _write_parquet(path, output)
    _write_json(
        meta_path,
        {
            **_config_cache_signature(config, year=year, start_date=start_date, end_date=end_date),
            "complete": True,
            "codes": expected_codes,
            "row_count": int(len(output)),
            "code_count": int(output["code"].nunique()) if not output.empty else 0,
            "failure_count": int(len(failures)),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return output, failures, {
        "daily_bars_cache_hit": 0,
        "daily_bars_cache_miss": 1,
        **_cache_compat_summary(meta, compatible),
    }


def normalize_daily_bars(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    for column in BAOSTOCK_DAILY_FIELDS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame["code"] = frame["code"].map(_baostock_to_std_code)
    frame["date"] = pd.to_datetime(frame["date"])
    for field in ("open", "high", "low", "close", "volume", "amount"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame["tradestatus"] = frame["tradestatus"].fillna("").astype(str)
    frame["isST"] = frame["isST"].fillna("0").astype(str)
    frame["source"] = "baostock"
    return frame[DAILY_BARS_COLUMNS]


def fetch_daily_metrics(
    source: BaostockSource,
    codes: list[str],
    *,
    config: PitBuildConfig,
    year: int,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, int]]:
    path = _daily_metrics_cache_path(config.output_root, year, config=config, start_date=start_date, end_date=end_date)
    meta_path = _daily_metrics_meta_path(config.output_root, year, config=config, start_date=start_date, end_date=end_date)
    meta = _read_json_or_empty(meta_path)
    compatible = _is_cache_compatible(meta, config, year=year, start_date=start_date, end_date=end_date)
    expected_codes = sorted(set(codes))
    if compatible and not _codes_cache_compatible(meta.get("codes", []), expected_codes):
        compatible = False
    if path.exists() and not config.force_refresh:
        if compatible:
            frame = pd.read_parquet(path)
            frame = _filter_dates(frame, start_date, end_date)
            if config.symbols:
                frame = frame[frame["code"].isin(set(config.symbols))].reset_index(drop=True)
            return frame, [], {
                "daily_metrics_cache_hit": 1,
                "daily_metrics_cache_miss": 0,
                **_cache_compat_summary(meta, compatible),
            }
        path.unlink(missing_ok=True)

    parts: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    signature = _config_cache_signature(config, year=year, start_date=start_date, end_date=end_date)
    progress = _new_progress(year, signature) if config.force_refresh else _read_daily_metrics_progress(config, year, start_date, end_date)
    resume_parts = not config.force_refresh and _progress_compatible(progress, config, year=year, start_date=start_date, end_date=end_date)
    if not resume_parts:
        progress = _new_progress(year, signature)
    completed_codes = set(progress.get("completed_metric_codes", []))
    batch_size = 50
    for batch_index, start in enumerate(range(0, len(codes), batch_size)):
        batch_codes = codes[start : start + batch_size]
        part_path = _daily_metrics_part_cache_path(config.output_root, year, batch_index, config=config, start_date=start_date, end_date=end_date)
        existing_part = (
            _read_parquet_or_empty(part_path, DAILY_METRICS_COLUMNS)
            if resume_parts
            else pd.DataFrame(columns=DAILY_METRICS_COLUMNS)
        )
        batch_parts: list[pd.DataFrame] = [existing_part] if not existing_part.empty else []
        for code in batch_codes:
            if code in completed_codes and not existing_part.empty and code in set(existing_part["code"].astype(str)):
                continue
            progress["active_metric_code"] = code
            _write_daily_metrics_progress(config, year, start_date, end_date, progress)
            try:
                raw = source.query_daily_metrics(_std_to_baostock_code(code), start_date, end_date)
            except BaostockSourceError as exc:
                failure = {"year": year, "code": code, "kind": "daily_metrics_fetch_error", "error": str(exc)}
                failures.append(failure)
                progress["failures"] = [*progress.get("failures", []), failure]
                progress["active_metric_code"] = ""
                _write_daily_metrics_progress(config, year, start_date, end_date, progress)
                continue
            if raw.empty:
                failure = {"year": year, "code": code, "kind": "empty_daily_metrics"}
                failures.append(failure)
                progress["failures"] = [*progress.get("failures", []), failure]
                progress["active_metric_code"] = ""
                _write_daily_metrics_progress(config, year, start_date, end_date, progress)
                continue
            frame = normalize_daily_metrics(raw)
            batch_parts.append(frame)
            batch_output = _concat(batch_parts, DAILY_METRICS_COLUMNS).drop_duplicates(["date", "code"])
            if not batch_output.empty:
                _write_parquet(part_path, batch_output)
            progress["completed_metric_codes"] = sorted(set(progress.get("completed_metric_codes", [])) | {code})
            progress["active_metric_code"] = ""
            _write_daily_metrics_progress(config, year, start_date, end_date, progress)
        batch_output = _concat(batch_parts, DAILY_METRICS_COLUMNS).drop_duplicates(["date", "code"])
        if not batch_output.empty:
            _write_parquet(part_path, batch_output)
            parts.append(batch_output)

    part_paths = (
        []
        if not resume_parts
        else sorted(_daily_metrics_parts_dir(config.output_root, year, config=config, start_date=start_date, end_date=end_date).glob("batch=*.parquet"))
    )
    part_frames = [_read_parquet_or_empty(part_path, DAILY_METRICS_COLUMNS) for part_path in part_paths]
    output = _concat([*part_frames, *parts], DAILY_METRICS_COLUMNS).drop_duplicates(["date", "code"])
    output = _filter_dates(output, start_date, end_date)
    if codes:
        output = output[output["code"].isin(set(codes))].reset_index(drop=True)
    _write_parquet(path, output)
    _write_json(
        meta_path,
        {
            **_config_cache_signature(config, year=year, start_date=start_date, end_date=end_date),
            "complete": True,
            "codes": expected_codes,
            "row_count": int(len(output)),
            "code_count": int(output["code"].nunique()) if not output.empty else 0,
            "failure_count": int(len(failures)),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return output, failures, {
        "daily_metrics_cache_hit": 0,
        "daily_metrics_cache_miss": 1,
        **_cache_compat_summary(meta, compatible),
    }


def normalize_daily_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    for column in BAOSTOCK_DAILY_METRICS_FIELDS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame["code"] = frame["code"].map(_baostock_to_std_code)
    frame["date"] = pd.to_datetime(frame["date"])
    for field in ("turn", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame["source"] = "baostock"
    return frame[DAILY_METRICS_COLUMNS]


def normalize_stock_industry(raw: pd.DataFrame, query_date: str) -> pd.DataFrame:
    frame = raw.copy()
    for column in ("updateDate", "code", "code_name", "industry", "industryClassification"):
        if column not in frame.columns:
            frame[column] = pd.NA
    if frame.empty:
        return pd.DataFrame(columns=STOCK_INDUSTRY_COLUMNS)
    frame["date"] = pd.Timestamp(query_date)
    frame["code"] = frame["code"].map(_baostock_to_std_code)
    frame["name_on_date"] = frame["code_name"].fillna("").astype(str)
    frame["industry"] = frame["industry"].fillna("").astype(str)
    frame["industry_classification"] = frame["industryClassification"].fillna("").astype(str)
    frame["industry_update_date"] = frame["updateDate"].map(_normalize_date)
    frame["source"] = "baostock"
    return frame[STOCK_INDUSTRY_COLUMNS]


def industry_query_dates(stock_lists: pd.DataFrame, frequency: str) -> list[pd.Timestamp]:
    if stock_lists.empty:
        return []
    dates = pd.Series(pd.to_datetime(stock_lists["date"]).dropna().unique()).sort_values().reset_index(drop=True)
    if frequency == "daily":
        return [pd.Timestamp(date) for date in dates]
    if frequency == "month-start":
        grouped = dates.groupby(dates.dt.to_period("M")).min()
        return [pd.Timestamp(date) for date in grouped.tolist()]
    raise ValueError(f"unsupported industry_frequency: {frequency}")


def expand_stock_industry_observations(
    observations: pd.DataFrame,
    stock_lists: pd.DataFrame,
    *,
    frequency: str,
) -> pd.DataFrame:
    observations = _ensure_columns(observations, STOCK_INDUSTRY_COLUMNS)
    stock_lists = _ensure_columns(stock_lists, RAW_STOCK_LIST_COLUMNS)
    if frequency == "daily" or observations.empty or stock_lists.empty:
        return _filter_stock_industry_to_stock_lists(observations, stock_lists)
    keys = stock_lists[["date", "code", "name_on_date"]].drop_duplicates().copy()
    keys["date"] = pd.to_datetime(keys["date"]).astype("datetime64[ns]")
    obs = observations.copy()
    obs["date"] = pd.to_datetime(obs["date"]).astype("datetime64[ns]")
    parts: list[pd.DataFrame] = []
    obs_by_code = {code: frame.sort_values("date") for code, frame in obs.groupby("code")}
    for code, code_keys in keys.sort_values(["code", "date"]).groupby("code"):
        code_obs = obs_by_code.get(code)
        if code_obs is None or code_obs.empty:
            empty = code_keys.copy()
            empty["industry"] = ""
            empty["industry_classification"] = ""
            empty["industry_update_date"] = ""
            empty["source"] = ""
            parts.append(empty[STOCK_INDUSTRY_COLUMNS])
            continue
        merged = pd.merge_asof(
            code_keys.sort_values("date"),
            code_obs[["date", "industry", "industry_classification", "industry_update_date", "source"]].sort_values("date"),
            on="date",
            direction="backward",
        )
        merged["code"] = code
        merged["industry"] = merged["industry"].fillna("")
        merged["industry_classification"] = merged["industry_classification"].fillna("")
        merged["industry_update_date"] = merged["industry_update_date"].fillna("")
        merged["source"] = merged["source"].where(merged["source"].fillna("") == "", merged["source"].astype(str) + f":{frequency}-ffill")
        merged["source"] = merged["source"].fillna("")
        parts.append(merged[STOCK_INDUSTRY_COLUMNS])
    return _concat(parts, STOCK_INDUSTRY_COLUMNS).sort_values(["date", "code"]).reset_index(drop=True)


def fetch_stock_industry(
    source: BaostockSource,
    stock_lists: pd.DataFrame,
    *,
    config: PitBuildConfig,
    year: int,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, int]]:
    path = _stock_industry_cache_path(config.output_root, year)
    meta_path = _stock_industry_meta_path(config.output_root, year)
    meta = _read_json_or_empty(meta_path)
    compatible = _is_industry_cache_compatible(meta, config, year=year, start_date=start_date, end_date=end_date)
    if path.exists() and not config.force_refresh:
        if compatible:
            frame = _filter_stock_industry_to_stock_lists(pd.read_parquet(path), stock_lists)
            return _filter_dates(frame, start_date, end_date), [], {
                "stock_industry_cache_hit": 1,
                "stock_industry_cache_miss": 0,
                **_cache_compat_summary(meta, compatible),
            }
        path.unlink(missing_ok=True)

    stock_lists = _filter_dates(_ensure_columns(stock_lists, RAW_STOCK_LIST_COLUMNS), start_date, end_date)
    if stock_lists.empty:
        return pd.DataFrame(columns=STOCK_INDUSTRY_COLUMNS), [], {
            "stock_industry_cache_hit": 0,
            "stock_industry_cache_miss": 1,
            "stock_industry_empty_stock_lists": 1,
            **_cache_compat_summary(meta, compatible),
        }

    failures: list[dict[str, Any]] = []
    parts: list[pd.DataFrame] = []
    signature = _config_cache_signature(config, year=year, start_date=start_date, end_date=end_date)
    progress = _new_progress(year, signature) if config.force_refresh else _read_progress(config.output_root, year)
    resume_parts = not config.force_refresh and _progress_compatible(progress, config, year=year, start_date=start_date, end_date=end_date)
    if not resume_parts:
        progress = _new_progress(year, signature)
    completed_dates = set(progress.get("completed_industry_dates", []))
    touched_months: set[int] = set()
    for date in industry_query_dates(stock_lists, config.industry_frequency):
        date_ts = pd.Timestamp(date)
        date_str = date_ts.strftime("%Y-%m-%d")
        month_path = _monthly_stock_industry_cache_path(config.output_root, year, date_ts.month)
        if date_str in completed_dates and month_path.exists():
            continue
        progress["active_industry_date"] = date_str
        _write_progress(config.output_root, year, progress)
        try:
            raw = source.query_stock_industry(date=date_str)
        except BaostockSourceError as exc:
            failure = {"year": year, "date": date_str, "kind": "industry_fetch_error", "error": str(exc)}
            failures.append(failure)
            progress["failures"] = [*progress.get("failures", []), failure]
            progress["active_industry_date"] = ""
            _write_progress(config.output_root, year, progress)
            continue
        if raw.empty:
            failure = {"year": year, "date": date_str, "kind": "empty_stock_industry"}
            failures.append(failure)
            progress["failures"] = [*progress.get("failures", []), failure]
            progress["active_industry_date"] = ""
            _write_progress(config.output_root, year, progress)
            continue
        frame = normalize_stock_industry(raw, date_str)
        frame = frame[frame["code"].map(is_sh_sz_mainboard_a_share)].reset_index(drop=True)
        frame = _filter_stock_industry_to_stock_lists(frame, stock_lists)
        existing_month = _read_parquet_or_empty(month_path, STOCK_INDUSTRY_COLUMNS) if resume_parts else pd.DataFrame(columns=STOCK_INDUSTRY_COLUMNS)
        month_output = _concat([existing_month, frame], STOCK_INDUSTRY_COLUMNS).drop_duplicates(["date", "code"])
        _write_parquet(month_path, month_output)
        parts.append(frame)
        touched_months.add(date_ts.month)
        progress["completed_industry_dates"] = sorted(set(progress.get("completed_industry_dates", [])) | {date_str})
        progress["active_industry_date"] = ""
        _write_progress(config.output_root, year, progress)

    months_to_read = sorted(touched_months) if config.force_refresh else list(range(1, 13))
    month_parts = [
        (
            _read_parquet_or_empty(_monthly_stock_industry_cache_path(config.output_root, year, month), STOCK_INDUSTRY_COLUMNS)
            if resume_parts
            else pd.DataFrame(columns=STOCK_INDUSTRY_COLUMNS)
        )
        for month in months_to_read
    ]
    observations = _concat([*month_parts, *parts], STOCK_INDUSTRY_COLUMNS).drop_duplicates(["date", "code"])
    observations = _filter_dates(observations, start_date, end_date)
    output = expand_stock_industry_observations(
        observations,
        stock_lists,
        frequency=config.industry_frequency,
    )
    _write_parquet(path, output)
    _write_json(
        meta_path,
        {
            **_industry_cache_signature(config, year=year, start_date=start_date, end_date=end_date),
            "complete": True,
            "row_count": int(len(output)),
            "date_count": int(output["date"].nunique()) if not output.empty else 0,
            "query_date_count": int(observations["date"].nunique()) if not observations.empty else 0,
            "failure_count": int(len(failures)),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return output, failures, {
        "stock_industry_cache_hit": 0,
        "stock_industry_cache_miss": 1,
        "stock_industry_failure_count": len(failures),
        **_cache_compat_summary(meta, compatible),
    }


def _filter_stock_industry_to_stock_lists(stock_industry: pd.DataFrame, stock_lists: pd.DataFrame) -> pd.DataFrame:
    stock_industry = _ensure_columns(stock_industry, STOCK_INDUSTRY_COLUMNS)
    if stock_industry.empty or stock_lists.empty:
        return stock_industry
    keys = stock_lists[["date", "code"]].drop_duplicates().copy()
    keys["date"] = pd.to_datetime(keys["date"]).astype("datetime64[ns]")
    output = stock_industry.copy()
    output["date"] = pd.to_datetime(output["date"]).astype("datetime64[ns]")
    return output.merge(keys, on=["date", "code"], how="inner")[STOCK_INDUSTRY_COLUMNS].reset_index(drop=True)


def build_daily_universe(
    stock_lists: pd.DataFrame,
    security_master: pd.DataFrame,
    daily_bars: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock_lists = _ensure_columns(stock_lists, RAW_STOCK_LIST_COLUMNS)
    security_master = _ensure_columns(security_master, SECURITY_MASTER_COLUMNS)
    daily_bars = _ensure_columns(daily_bars, DAILY_BARS_COLUMNS)
    if stock_lists.empty:
        return pd.DataFrame(columns=DAILY_UNIVERSE_COLUMNS), pd.DataFrame(columns=DAILY_STATUS_COLUMNS)

    merged = stock_lists.merge(security_master, on="code", how="left")
    bar_flags = daily_bars[["date", "code", "open", "high", "low", "close", "volume", "amount", "tradestatus", "isST"]].copy()
    merged = merged.merge(bar_flags, on=["date", "code"], how="left")

    date_series = pd.to_datetime(merged["date"])
    ipo_dates = pd.to_datetime(merged["ipo_date"], errors="coerce")
    out_dates = pd.to_datetime(merged["out_date"], errors="coerce")
    has_bar = merged[["open", "high", "low", "close"]].notna().any(axis=1)
    volume = pd.to_numeric(merged["volume"], errors="coerce")
    amount = pd.to_numeric(merged["amount"], errors="coerce")
    is_stock_type = merged["security_type"].fillna("") == "1"
    is_listed = ipo_dates.notna() & (ipo_dates <= date_series) & (out_dates.isna() | (date_series < out_dates))
    is_mainboard = merged["code"].map(is_sh_sz_mainboard_a_share)
    is_trade_status_ok = merged["tradestatus"].fillna(merged["query_all_trade_status"]).astype(str) == "1"
    is_st = merged["isST"].fillna("0").astype(str) == "1"
    missing_bar = ~has_bar
    is_suspended = (~is_trade_status_ok) | missing_bar | volume.isna() | amount.isna() | (volume <= 0) | (amount <= 0)

    merged["is_listed_on_date"] = is_listed.astype(bool)
    merged["is_mainboard"] = is_mainboard.astype(bool)
    merged["is_common_a_share"] = is_stock_type.astype(bool)
    merged["is_st_on_date"] = is_st.astype(bool)
    merged["has_bar"] = has_bar.astype(bool)
    merged["is_suspended_on_date"] = is_suspended.astype(bool)
    merged["is_tradeable"] = (
        merged["is_listed_on_date"]
        & merged["is_mainboard"]
        & merged["is_common_a_share"]
        & (~merged["is_st_on_date"])
        & (~merged["is_suspended_on_date"])
    ).astype(bool)
    merged["reject_reason"] = _reject_reasons(merged, missing_bar)

    universe = merged[DAILY_UNIVERSE_COLUMNS].sort_values(["date", "code"]).reset_index(drop=True)
    status = universe[["date", "code", "has_bar", "is_suspended_on_date", "is_tradeable"]].rename(
        columns={"is_suspended_on_date": "is_suspended_like"}
    )
    return universe, status[DAILY_STATUS_COLUMNS].sort_values(["date", "code"]).reset_index(drop=True)


def _reject_reasons(frame: pd.DataFrame, missing_bar: pd.Series) -> list[str]:
    reasons: list[str] = []
    missing_values = missing_bar.reset_index(drop=True).tolist()
    for index, row in enumerate(frame.itertuples(index=False)):
        if not bool(row.is_listed_on_date):
            reasons.append("not_listed_on_date")
        elif not bool(row.is_mainboard):
            reasons.append("not_mainboard")
        elif not bool(row.is_common_a_share):
            reasons.append("not_common_a_share")
        elif bool(row.is_st_on_date):
            reasons.append("st_on_date")
        elif bool(row.is_suspended_on_date):
            reasons.append("missing_bar" if bool(missing_values[index]) else "suspended_on_date")
        else:
            reasons.append("")
    return reasons


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = pd.NA
    return output[columns]


def _concat(parts: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    valid = [part for part in parts if part is not None and not part.empty]
    if not valid:
        return pd.DataFrame(columns=columns)
    return pd.concat(valid, ignore_index=True)


def _source_config(config: PitBuildConfig) -> BaostockSourceConfig:
    return BaostockSourceConfig(request_interval_seconds=config.request_interval_seconds)


def discover(config: PitBuildConfig) -> dict[str, Any]:
    with BaostockSource(_source_config(config)) as source:
        trade_dates, trade_cache = _trade_dates(source, config)
        explicit = set(config.symbols) if config.symbols else None
        security_master_all = None
        basic_cache: dict[str, Any] = {}
        if config.discovery_mode == "stock-basic":
            security_master_all, basic_cache = _stock_basic_all(source, config)
        summaries: list[dict[str, Any]] = []
        for year in _years_between(config.start_date, config.end_date):
            start, end = _year_bounds(year, config.start_date, config.end_date)
            year_dates = _filter_dates(trade_dates, start, end)["date"]
            stock_lists, summary = discover_daily_stock_lists(
                source,
                year_dates,
                config=config,
                year=year,
                explicit_symbols=explicit,
                security_master=security_master_all,
            )
            summaries.append({"year": year, "rows": len(stock_lists), **summary})
        stock_lists = load_cached_stock_lists(config.output_root, _years_between(config.start_date, config.end_date), config.start_date, config.end_date, config=config)
        if security_master_all is not None:
            security_master, security_summary = update_security_master_from_basic(stock_lists, security_master_all, config=config)
        else:
            security_master, security_summary = update_security_master(source, stock_lists, config=config)
        return {
            "command": "discover",
            "trade_dates": len(trade_dates),
            "stock_list_years": summaries,
            "security_master_rows": len(security_master),
            "cache": {**trade_cache, **basic_cache},
            "security": security_summary,
            "source_errors": source.errors,
        }


def fetch_bars(config: PitBuildConfig) -> dict[str, Any]:
    with BaostockSource(_source_config(config)) as source:
        years = [config.year] if config.year else _years_between(config.start_date, config.end_date)
        all_failures: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for year in years:
            start, end = _year_bounds(year, config.start_date, config.end_date)
            stock_lists = load_cached_stock_lists(config.output_root, [year], start, end, config=config)
            codes = sorted(stock_lists["code"].unique().tolist())
            if config.max_symbols > 0:
                codes = codes[: config.max_symbols]
            if config.symbols:
                codes = [code for code in codes if code in set(config.symbols)]
            bars, failures, cache = fetch_daily_bars(source, codes, config=config, year=year, start_date=start, end_date=end)
            all_failures.extend(failures)
            summaries.append({"year": year, "codes": len(codes), "rows": len(bars), "failures": len(failures), **cache})
        return {"command": "fetch-bars", "years": summaries, "failure_count": len(all_failures), "failures": all_failures, "source_errors": source.errors}


def fetch_industry(config: PitBuildConfig) -> dict[str, Any]:
    with BaostockSource(_source_config(config)) as source:
        years = [config.year] if config.year else _years_between(config.start_date, config.end_date)
        all_failures: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for year in years:
            start, end = _year_bounds(year, config.start_date, config.end_date)
            stock_lists = load_cached_stock_lists(config.output_root, [year], start, end, config=config)
            if config.max_symbols > 0:
                codes = sorted(stock_lists["code"].unique().tolist())[: config.max_symbols]
                stock_lists = stock_lists[stock_lists["code"].isin(codes)].reset_index(drop=True)
            if config.symbols:
                stock_lists = stock_lists[stock_lists["code"].isin(set(config.symbols))].reset_index(drop=True)
            industry, failures, cache = fetch_stock_industry(
                source,
                stock_lists,
                config=config,
                year=year,
                start_date=start,
                end_date=end,
            )
            all_failures.extend(failures)
            summaries.append({"year": year, "dates": int(industry["date"].nunique()) if not industry.empty else 0, "rows": len(industry), "failures": len(failures), **cache})
        return {
            "command": "fetch-industry",
            "years": summaries,
            "failure_count": len(all_failures),
            "failures": all_failures,
            "source_errors": source.errors,
        }


def fetch_metrics(config: PitBuildConfig) -> dict[str, Any]:
    with BaostockSource(_source_config(config)) as source:
        years = [config.year] if config.year else _years_between(config.start_date, config.end_date)
        all_failures: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for year in years:
            start, end = _year_bounds(year, config.start_date, config.end_date)
            stock_lists = load_cached_stock_lists(config.output_root, [year], start, end, config=config)
            codes = sorted(stock_lists["code"].unique().tolist())
            if config.max_symbols > 0:
                codes = codes[: config.max_symbols]
            if config.symbols:
                codes = [code for code in codes if code in set(config.symbols)]
            metrics, failures, cache = fetch_daily_metrics(source, codes, config=config, year=year, start_date=start, end_date=end)
            all_failures.extend(failures)
            summaries.append({"year": year, "codes": len(codes), "rows": len(metrics), "failures": len(failures), **cache})
        return {
            "command": "fetch-metrics",
            "years": summaries,
            "failure_count": len(all_failures),
            "failures": all_failures,
            "source_errors": source.errors,
        }


def fetch_size(config: PitBuildConfig) -> dict[str, Any]:
    """Fetch/cache external daily size fields without assembling a snapshot."""

    years = [config.year] if config.year else _years_between(config.start_date, config.end_date)
    if config.size_source == SIZE_SOURCE_PROXY_AMOUNT:
        return _fetch_proxy_amount_size(config, years)
    if config.size_source == SIZE_SOURCE_AKSHARE_CNINFO_RECONSTRUCTED:
        return _fetch_akshare_cninfo_reconstructed_size(config, years)
    if config.size_source != SIZE_SOURCE_TUSHARE_DAILY_BASIC:
        raise ValueError(f"unsupported size_source: {config.size_source}")

    requested_trade_dates = set(normalize_trade_dates(config.trade_dates)) if config.trade_dates else set()
    cached_trade_dates = _cached_trade_dates_for_size(config)
    summaries: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []

    for year in years:
        start, end = _year_bounds(year, config.start_date, config.end_date)
        trade_dates = _select_size_trade_dates(
            year=year,
            start_date=start,
            end_date=end,
            requested_trade_dates=requested_trade_dates,
            cached_trade_dates=cached_trade_dates,
            output_root=config.output_root,
        )
        stock_lists = load_cached_stock_lists(config.output_root, [year], start, end, config=None)
        codes = sorted(stock_lists["code"].unique().tolist()) if not stock_lists.empty else sorted(config.symbols)
        if config.max_symbols > 0:
            codes = codes[: config.max_symbols]
        if config.symbols:
            codes = [code for code in codes if code in set(config.symbols)]
        if not codes and config.symbols:
            codes = sorted(config.symbols)
        if not trade_dates:
            failure = {"year": year, "trade_date": "", "error_type": "missing_trade_dates", "message": "No trade dates found from --trade-dates, trade_dates cache, or daily_stock_lists cache."}
            all_failures.append(failure)
            summaries.append(
                {
                    "command": "fetch-size",
                    "source": TUSHARE_DAILY_BASIC_SOURCE,
                    "size_source": config.size_source,
                    "status": "failed",
                    "year": year,
                    "rows": 0,
                    "date_count": 0,
                    "code_count": len(codes),
                    "failure_count": 1,
                    "failures": [failure],
                    "cache_hit": 0,
                    "cache_miss": 0,
                }
            )
            continue
        summary = fetch_tushare_daily_size_cache(
            output_root=config.output_root,
            year=year,
            trade_dates=trade_dates,
            start_date=start,
            end_date=end,
            symbols=codes,
            token=config.size_token,
            force_refresh=config.force_refresh,
            request_interval_seconds=config.request_interval_seconds,
        )
        summaries.append(summary)
        all_failures.extend(summary.get("failures", []))

    return {
        "command": "fetch-size",
        "source": TUSHARE_DAILY_BASIC_SOURCE,
        "size_source": config.size_source,
        "status": _fetch_size_status(summaries),
        "years": summaries,
        "failure_count": len(all_failures),
        "failures": all_failures,
    }


def _fetch_proxy_amount_size(config: PitBuildConfig, years: list[int]) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    for year in years:
        start, end = _year_bounds(year, config.start_date, config.end_date)
        bars = _filter_dates(_read_parquet_or_empty(_daily_bars_cache_path(config.output_root, year), DAILY_BARS_COLUMNS), start, end)
        codes = _size_request_codes(config, year, start, end, bars)
        if codes:
            bars = bars.loc[bars["code"].isin(set(codes))].reset_index(drop=True)
        if bars.empty:
            failure = {
                "year": year,
                "trade_date": "",
                "error_type": "missing_daily_bars",
                "message": "proxy_amount requires cached daily_bars for the requested year/date/symbol scope.",
            }
            all_failures.append(failure)
            summaries.append(_failed_size_year_summary(year, PROXY_AMOUNT_SOURCE, config.size_source, [failure], len(codes)))
            continue
        frame = standardize_proxy_amount_daily_size(bars[["date", "code", "amount"]])
        path_symbols = codes if (config.symbols or config.max_symbols > 0) else None
        path = daily_size_cache_path(config.output_root, year, symbols=path_symbols)
        meta_path = daily_size_meta_path(config.output_root, year, symbols=path_symbols)
        _write_parquet(path, frame)
        _write_json(
            meta_path,
            {
                "source": PROXY_AMOUNT_SOURCE,
                "source_grade": daily_size_source_grade(PROXY_AMOUNT_SOURCE),
                "size_source": config.size_source,
                "year": int(year),
                "start_date": start,
                "end_date": end,
                "symbols": list(path_symbols or []),
                "complete": True,
                "row_count": int(len(frame)),
                "date_count": int(frame["date"].nunique()) if not frame.empty else 0,
                "code_count": int(frame["code"].nunique()) if not frame.empty else 0,
                "status": "passed",
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        summaries.append(
            {
                "command": "fetch-size",
                "source": PROXY_AMOUNT_SOURCE,
                "size_source": config.size_source,
                "source_grade": daily_size_source_grade(PROXY_AMOUNT_SOURCE),
                "status": "passed",
                "year": int(year),
                "rows": int(len(frame)),
                "date_count": int(frame["date"].nunique()) if not frame.empty else 0,
                "code_count": int(frame["code"].nunique()) if not frame.empty else 0,
                "failure_count": 0,
                "failures": [],
                "cache_path": str(path),
                "cache_hit": 0,
                "cache_miss": 1,
                "promotion_eligible": False,
            }
        )
    return {
        "command": "fetch-size",
        "source": PROXY_AMOUNT_SOURCE,
        "size_source": config.size_source,
        "status": _fetch_size_status(summaries),
        "years": summaries,
        "failure_count": len(all_failures),
        "failures": all_failures,
    }


def _fetch_akshare_cninfo_reconstructed_size(config: PitBuildConfig, years: list[int]) -> dict[str, Any]:
    try:
        akshare_module = importlib.import_module("akshare")
        import_error = ""
    except Exception as exc:  # noqa: BLE001
        akshare_module = None
        import_error = str(exc)

    summaries: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    for year in years:
        start, end = _year_bounds(year, config.start_date, config.end_date)
        bars = _filter_dates(_read_parquet_or_empty(_daily_bars_cache_path(config.output_root, year), DAILY_BARS_COLUMNS), start, end)
        codes = _size_request_codes(config, year, start, end, bars)
        if codes:
            bars = bars.loc[bars["code"].isin(set(codes))].reset_index(drop=True)
        path_symbols = codes if (config.symbols or config.max_symbols > 0) else None
        path = daily_size_cache_path(config.output_root, year, symbols=path_symbols)
        if path.exists() and not config.force_refresh:
            cached = pd.read_parquet(path)
            summaries.append(
                {
                    "command": "fetch-size",
                    "source": AKSHARE_CNINFO_RECONSTRUCTED_SOURCE,
                    "size_source": config.size_source,
                    "source_grade": daily_size_source_grade(AKSHARE_CNINFO_RECONSTRUCTED_SOURCE),
                    "status": "cache_hit",
                    "year": int(year),
                    "rows": int(len(cached)),
                    "date_count": int(pd.to_datetime(cached["date"]).nunique()) if not cached.empty and "date" in cached.columns else 0,
                    "code_count": int(cached["code"].nunique()) if not cached.empty and "code" in cached.columns else 0,
                    "failure_count": 0,
                    "failures": [],
                    "cache_path": str(path),
                    "cache_hit": 1,
                    "cache_miss": 0,
                    "promotion_eligible": True,
                }
            )
            continue
        if akshare_module is None:
            failure = {"year": year, "trade_date": "", "error_type": "package_missing", "message": import_error}
            all_failures.append(failure)
            summaries.append(_failed_size_year_summary(year, AKSHARE_CNINFO_RECONSTRUCTED_SOURCE, config.size_source, [failure], len(codes)))
            continue
        if bars.empty:
            failure = {
                "year": year,
                "trade_date": "",
                "error_type": "missing_daily_bars",
                "message": "akshare_cninfo_reconstructed requires cached daily_bars close prices.",
            }
            all_failures.append(failure)
            summaries.append(_failed_size_year_summary(year, AKSHARE_CNINFO_RECONSTRUCTED_SOURCE, config.size_source, [failure], len(codes)))
            continue

        event_frames: list[pd.DataFrame] = []
        failures: list[dict[str, Any]] = []
        for code in codes:
            try:
                raw = akshare_module.stock_share_change_cninfo(
                    symbol=code.split(".", 1)[0],
                    start_date="19000101",
                    end_date=pd.Timestamp(end).strftime("%Y%m%d"),
                )
            except Exception as exc:  # noqa: BLE001
                failures.append({"year": year, "code": code, "trade_date": "", "error_type": type(exc).__name__, "message": str(exc)})
                continue
            events = normalize_cninfo_share_change_events(raw, code=code)
            if events.empty:
                failures.append({"year": year, "code": code, "trade_date": "", "error_type": "empty_share_events", "message": "stock_share_change_cninfo returned no parseable share events."})
                continue
            event_frames.append(events)
            if config.request_interval_seconds > 0:
                time.sleep(float(config.request_interval_seconds))
        share_events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
        frame = reconstruct_daily_size_from_share_events(bars, share_events)
        if frame.empty:
            failure = {"year": year, "trade_date": "", "error_type": "empty_reconstructed_size", "message": "No daily_size rows could be reconstructed from share events and daily bars."}
            failures.append(failure)
        if not frame.empty:
            _write_parquet(path, frame)
            _write_json(
                daily_size_meta_path(config.output_root, year, symbols=path_symbols),
                {
                    "source": AKSHARE_CNINFO_RECONSTRUCTED_SOURCE,
                    "source_grade": daily_size_source_grade(AKSHARE_CNINFO_RECONSTRUCTED_SOURCE),
                    "size_source": config.size_source,
                    "year": int(year),
                    "start_date": start,
                    "end_date": end,
                    "symbols": list(path_symbols or []),
                    "complete": True,
                    "row_count": int(len(frame)),
                    "date_count": int(frame["date"].nunique()) if not frame.empty else 0,
                    "code_count": int(frame["code"].nunique()) if not frame.empty else 0,
                    "failure_count": int(len(failures)),
                    "status": "passed" if not failures else "partial",
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
        all_failures.extend(failures)
        summaries.append(
            {
                "command": "fetch-size",
                "source": AKSHARE_CNINFO_RECONSTRUCTED_SOURCE,
                "size_source": config.size_source,
                "source_grade": daily_size_source_grade(AKSHARE_CNINFO_RECONSTRUCTED_SOURCE),
                "status": "passed" if frame is not None and not frame.empty and not failures else ("partial" if frame is not None and not frame.empty else "failed"),
                "year": int(year),
                "rows": int(len(frame)),
                "date_count": int(frame["date"].nunique()) if frame is not None and not frame.empty else 0,
                "code_count": int(frame["code"].nunique()) if frame is not None and not frame.empty else 0,
                "failure_count": int(len(failures)),
                "failures": failures,
                "cache_path": str(path),
                "cache_hit": 0,
                "cache_miss": 1,
                "promotion_eligible": True,
            }
        )
    return {
        "command": "fetch-size",
        "source": AKSHARE_CNINFO_RECONSTRUCTED_SOURCE,
        "size_source": config.size_source,
        "status": _fetch_size_status(summaries),
        "years": summaries,
        "failure_count": len(all_failures),
        "failures": all_failures,
    }


def _size_request_codes(config: PitBuildConfig, year: int, start: str, end: str, fallback_frame: pd.DataFrame) -> list[str]:
    stock_lists = load_cached_stock_lists(config.output_root, [year], start, end, config=None)
    codes = sorted(stock_lists["code"].unique().tolist()) if not stock_lists.empty else sorted(fallback_frame["code"].dropna().astype(str).unique().tolist())
    if config.max_symbols > 0:
        codes = codes[: config.max_symbols]
    if config.symbols:
        codes = [code for code in codes if code in set(config.symbols)]
    if not codes and config.symbols:
        codes = sorted(config.symbols)
    return codes


def _failed_size_year_summary(year: int, source: str, size_source: str, failures: list[dict[str, Any]], code_count: int) -> dict[str, Any]:
    return {
        "command": "fetch-size",
        "source": source,
        "size_source": size_source,
        "source_grade": daily_size_source_grade(source),
        "status": "failed",
        "year": int(year),
        "rows": 0,
        "date_count": 0,
        "code_count": int(code_count),
        "failure_count": int(len(failures)),
        "failures": failures,
        "cache_hit": 0,
        "cache_miss": 0,
    }


def _fetch_size_status(summaries: list[dict[str, Any]]) -> str:
    if not summaries:
        return "failed"
    if all(item.get("status") == "cache_hit" for item in summaries):
        return "cache_hit"
    if all(item.get("status") == "skipped" for item in summaries):
        return "skipped"
    if any(item.get("status") == "failed" for item in summaries):
        return "failed"
    if any(item.get("status") in {"partial", "skipped"} for item in summaries):
        return "partial"
    return "passed"


def _cached_trade_dates_for_size(config: PitBuildConfig) -> list[str]:
    trade_dates = _filter_dates(
        _read_parquet_or_empty(_trade_dates_cache_path(config.output_root), ["date", "is_trading_day"]),
        config.start_date,
        config.end_date,
    )
    if not trade_dates.empty:
        return pd.to_datetime(trade_dates["date"]).dt.strftime("%Y%m%d").sort_values().drop_duplicates().tolist()
    stock_lists = load_cached_stock_lists(config.output_root, _years_between(config.start_date, config.end_date), config.start_date, config.end_date, config=None)
    if stock_lists.empty:
        return []
    return pd.to_datetime(stock_lists["date"]).dt.strftime("%Y%m%d").sort_values().drop_duplicates().tolist()


def _select_size_trade_dates(
    *,
    year: int,
    start_date: str,
    end_date: str,
    requested_trade_dates: set[str],
    cached_trade_dates: list[str],
    output_root: Path,
) -> list[str]:
    if requested_trade_dates:
        source = sorted(requested_trade_dates)
    else:
        source = cached_trade_dates
    selected = [
        item
        for item in normalize_trade_dates(source)
        if pd.Timestamp(start_date) <= pd.Timestamp(item) <= pd.Timestamp(end_date)
        and pd.Timestamp(item).year == int(year)
    ]
    if selected:
        return selected
    stock_lists = _filter_dates(_read_parquet_or_empty(_stock_list_cache_path(output_root, year), RAW_STOCK_LIST_COLUMNS), start_date, end_date)
    if stock_lists.empty:
        return []
    return pd.to_datetime(stock_lists["date"]).dt.strftime("%Y%m%d").sort_values().drop_duplicates().tolist()


def build_year(config: PitBuildConfig) -> dict[str, Any]:
    year = config.year or pd.Timestamp(config.start_date).year
    start, end = _year_bounds(year, config.start_date, config.end_date)
    year_config = PitBuildConfig(
        output_root=config.output_root,
        start_date=start,
        end_date=end,
        symbols=config.symbols,
        max_symbols=config.max_symbols,
        command="build-year",
        snapshot_id=config.snapshot_id,
        year=year,
        trade_dates=config.trade_dates,
        force_refresh=config.force_refresh,
        request_interval_seconds=config.request_interval_seconds,
        discovery_mode=config.discovery_mode,
        include_industry=config.include_industry,
        industry_frequency=config.industry_frequency,
        include_metrics=config.include_metrics,
        include_size=config.include_size,
        size_source=config.size_source,
        size_token=config.size_token,
    )
    with BaostockSource(_source_config(year_config)) as source:
        trade_dates, trade_cache = _trade_dates(source, year_config)
        explicit = set(config.symbols) if config.symbols else None
        security_master_all = None
        basic_cache: dict[str, Any] = {}
        if year_config.discovery_mode == "stock-basic":
            security_master_all, basic_cache = _stock_basic_all(source, year_config)
        stock_lists, stock_cache = discover_daily_stock_lists(
            source,
            trade_dates["date"],
            config=year_config,
            year=year,
            explicit_symbols=explicit,
            security_master=security_master_all,
        )
        if stock_lists.empty and not trade_dates.empty:
            raise BaostockSourceError(f"empty stock list for year {year}")
        if config.max_symbols > 0:
            codes = sorted(stock_lists["code"].unique().tolist())[: config.max_symbols]
            stock_lists = stock_lists[stock_lists["code"].isin(codes)].reset_index(drop=True)
        if security_master_all is not None:
            security_master, security_summary = update_security_master_from_basic(stock_lists, security_master_all, config=year_config)
        else:
            security_master, security_summary = update_security_master(source, stock_lists, config=year_config)
        codes = sorted(stock_lists["code"].unique().tolist())
        daily_bars, failures, bar_cache = fetch_daily_bars(source, codes, config=year_config, year=year, start_date=start, end_date=end)
        _assert_failure_rate(year, len(codes), failures)
        daily_metrics = pd.DataFrame(columns=DAILY_METRICS_COLUMNS)
        metrics_failures: list[dict[str, Any]] = []
        metrics_cache: dict[str, Any] = {}
        if year_config.include_metrics:
            daily_metrics, metrics_failures, metrics_cache = fetch_daily_metrics(
                source,
                codes,
                config=year_config,
                year=year,
                start_date=start,
                end_date=end,
            )
            _assert_failure_rate(year, len(codes), metrics_failures)
        stock_industry = pd.DataFrame(columns=STOCK_INDUSTRY_COLUMNS)
        industry_failures: list[dict[str, Any]] = []
        industry_cache: dict[str, Any] = {}
        if year_config.include_industry:
            stock_industry, industry_failures, industry_cache = fetch_stock_industry(
                source,
                stock_lists,
                config=year_config,
                year=year,
                start_date=start,
                end_date=end,
            )
        daily_size = (
            load_cached_daily_size(year_config.output_root, [year], start, end, symbols=codes)
            if year_config.include_size
            else pd.DataFrame(columns=DAILY_SIZE_COLUMNS)
        )
        manifest = assemble_snapshot_from_frames(
            config=year_config,
            trade_dates=trade_dates,
            security_master=security_master,
            stock_lists=stock_lists,
            daily_bars=daily_bars,
            daily_metrics=daily_metrics,
            stock_industry=stock_industry,
            daily_size=daily_size,
            failures=failures,
            source_version=source.version,
            source_errors=source.errors,
            extra_quality={
                **trade_cache,
                **basic_cache,
                **stock_cache,
                **bar_cache,
                **metrics_cache,
                **industry_cache,
                **security_summary,
                "daily_metrics_failures": metrics_failures,
                "stock_industry_failures": industry_failures,
                "daily_size_rows": int(len(daily_size)),
            },
        )
        return manifest


def assemble(config: PitBuildConfig) -> dict[str, Any]:
    years = [config.year] if config.year else _years_between(config.start_date, config.end_date)
    trade_dates = _filter_dates(_read_parquet_or_empty(_trade_dates_cache_path(config.output_root), ["date", "is_trading_day"]), config.start_date, config.end_date)
    stock_lists = load_cached_stock_lists(config.output_root, years, config.start_date, config.end_date, config=config)
    if config.max_symbols > 0:
        codes = sorted(stock_lists["code"].unique().tolist())[: config.max_symbols]
        stock_lists = stock_lists[stock_lists["code"].isin(codes)].reset_index(drop=True)
    if config.symbols:
        stock_lists = stock_lists[stock_lists["code"].isin(set(config.symbols))].reset_index(drop=True)
    security_master = _read_parquet_or_empty(_security_master_cache_path(config.output_root), SECURITY_MASTER_COLUMNS)
    if not stock_lists.empty:
        security_master = security_master[security_master["code"].isin(set(stock_lists["code"]))].reset_index(drop=True)
    daily_bars = load_cached_daily_bars(config.output_root, years, config.start_date, config.end_date, config=config)
    if not stock_lists.empty:
        daily_bars = daily_bars[daily_bars["code"].isin(set(stock_lists["code"]))].reset_index(drop=True)
    daily_metrics = (
        load_cached_daily_metrics(config.output_root, years, config.start_date, config.end_date, config=config)
        if config.include_metrics
        else pd.DataFrame(columns=DAILY_METRICS_COLUMNS)
    )
    if not stock_lists.empty:
        daily_metrics = daily_metrics[daily_metrics["code"].isin(set(stock_lists["code"]))].reset_index(drop=True)
    stock_industry = load_cached_stock_industry(config.output_root, years, config.start_date, config.end_date, config=config)
    stock_industry = _filter_stock_industry_to_stock_lists(stock_industry, stock_lists)
    daily_size = (
        load_cached_daily_size(config.output_root, years, config.start_date, config.end_date, symbols=sorted(stock_lists["code"].unique().tolist()))
        if config.include_size
        else pd.DataFrame(columns=DAILY_SIZE_COLUMNS)
    )
    daily_size = _filter_daily_size_to_stock_lists(daily_size, stock_lists)
    trade_dates = _repair_trade_dates_from_stock_lists(trade_dates, stock_lists, config.start_date, config.end_date)
    if trade_dates.empty:
        raise BaostockSourceError("cannot assemble: trade date cache is empty")
    if stock_lists.empty:
        raise BaostockSourceError("cannot assemble: stock list cache is empty")
    return assemble_snapshot_from_frames(
        config=config,
        trade_dates=trade_dates,
        security_master=security_master,
        stock_lists=stock_lists,
        daily_bars=daily_bars,
        daily_metrics=daily_metrics,
        stock_industry=stock_industry,
        daily_size=daily_size,
        failures=[],
        source_version="cache",
        source_errors=[],
        extra_quality={
            "assembled_from_cache": 1,
            "stock_industry_rows": int(len(stock_industry)),
            "daily_metrics_rows": int(len(daily_metrics)),
            "daily_size_rows": int(len(daily_size)),
        },
    )


def _repair_trade_dates_from_stock_lists(
    trade_dates: pd.DataFrame,
    stock_lists: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    if stock_lists.empty:
        return trade_dates
    derived_dates = pd.Series(pd.to_datetime(stock_lists["date"]).dropna().unique())
    derived_dates = derived_dates[(derived_dates >= pd.Timestamp(start_date)) & (derived_dates <= pd.Timestamp(end_date))]
    if derived_dates.empty:
        return trade_dates
    if not trade_dates.empty:
        cached_dates = set(pd.to_datetime(trade_dates["date"]).dt.normalize())
        derived_set = set(pd.to_datetime(derived_dates).dt.normalize())
        if derived_set.issubset(cached_dates):
            return trade_dates.sort_values("date").reset_index(drop=True)
    return pd.DataFrame({"date": pd.to_datetime(sorted(derived_dates)), "is_trading_day": True})


def build_snapshot(config: PitBuildConfig) -> dict[str, Any]:
    with BaostockSource(_source_config(config)) as source:
        trade_dates, trade_cache = _trade_dates(source, config)
        explicit = set(config.symbols) if config.symbols else None
        security_master_all = None
        basic_cache: dict[str, Any] = {}
        if config.discovery_mode == "stock-basic":
            security_master_all, basic_cache = _stock_basic_all(source, config)
        stock_parts: list[pd.DataFrame] = []
        stock_quality: dict[str, Any] = {}
        for year in _years_between(config.start_date, config.end_date):
            start, end = _year_bounds(year, config.start_date, config.end_date)
            year_dates = _filter_dates(trade_dates, start, end)["date"]
            stock_lists, summary = discover_daily_stock_lists(
                source,
                year_dates,
                config=config,
                year=year,
                explicit_symbols=explicit,
                security_master=security_master_all,
            )
            stock_quality[f"stock_list_year_{year}"] = summary
            stock_parts.append(stock_lists)
        stock_lists = _concat(stock_parts, RAW_STOCK_LIST_COLUMNS)
        if stock_lists.empty and not trade_dates.empty:
            raise BaostockSourceError("empty stock list during build")
        if config.max_symbols > 0:
            codes = sorted(stock_lists["code"].unique().tolist())[: config.max_symbols]
            stock_lists = stock_lists[stock_lists["code"].isin(codes)].reset_index(drop=True)
        if security_master_all is not None:
            security_master, security_summary = update_security_master_from_basic(stock_lists, security_master_all, config=config)
        else:
            security_master, security_summary = update_security_master(source, stock_lists, config=config)
        bar_parts: list[pd.DataFrame] = []
        metric_parts: list[pd.DataFrame] = []
        industry_parts: list[pd.DataFrame] = []
        failures: list[dict[str, Any]] = []
        metrics_failures: list[dict[str, Any]] = []
        industry_failures: list[dict[str, Any]] = []
        bar_quality: dict[str, Any] = {}
        metric_quality: dict[str, Any] = {}
        industry_quality: dict[str, Any] = {}
        for year in _years_between(config.start_date, config.end_date):
            start, end = _year_bounds(year, config.start_date, config.end_date)
            year_stock_lists = _filter_dates(stock_lists, start, end)
            codes = sorted(year_stock_lists["code"].unique().tolist())
            bars, year_failures, cache = fetch_daily_bars(source, codes, config=config, year=year, start_date=start, end_date=end)
            _assert_failure_rate(year, len(codes), year_failures)
            bar_parts.append(bars)
            failures.extend(year_failures)
            bar_quality[f"daily_bars_year_{year}"] = cache
            if config.include_metrics:
                metrics, year_metric_failures, metric_cache = fetch_daily_metrics(
                    source,
                    codes,
                    config=config,
                    year=year,
                    start_date=start,
                    end_date=end,
                )
                _assert_failure_rate(year, len(codes), year_metric_failures)
                metric_parts.append(metrics)
                metrics_failures.extend(year_metric_failures)
                metric_quality[f"daily_metrics_year_{year}"] = metric_cache
            if config.include_industry:
                industry, year_industry_failures, industry_cache = fetch_stock_industry(
                    source,
                    year_stock_lists,
                    config=config,
                    year=year,
                    start_date=start,
                    end_date=end,
                )
                industry_parts.append(industry)
                industry_failures.extend(year_industry_failures)
                industry_quality[f"stock_industry_year_{year}"] = industry_cache
        daily_bars = _concat(bar_parts, DAILY_BARS_COLUMNS)
        daily_metrics = _concat(metric_parts, DAILY_METRICS_COLUMNS)
        stock_industry = _concat(industry_parts, STOCK_INDUSTRY_COLUMNS)
        daily_size = (
            load_cached_daily_size(
                config.output_root,
                _years_between(config.start_date, config.end_date),
                config.start_date,
                config.end_date,
                symbols=sorted(stock_lists["code"].unique().tolist()),
            )
            if config.include_size
            else pd.DataFrame(columns=DAILY_SIZE_COLUMNS)
        )
        return assemble_snapshot_from_frames(
            config=config,
            trade_dates=trade_dates,
            security_master=security_master,
            stock_lists=stock_lists,
            daily_bars=daily_bars,
            daily_metrics=daily_metrics,
            stock_industry=stock_industry,
            daily_size=daily_size,
            failures=failures,
            source_version=source.version,
            source_errors=source.errors,
            extra_quality={
                **trade_cache,
                **basic_cache,
                **security_summary,
                **stock_quality,
                **bar_quality,
                **metric_quality,
                **industry_quality,
                "daily_metrics_failures": metrics_failures,
                "stock_industry_failures": industry_failures,
                "daily_size_rows": int(len(daily_size)),
            },
        )


def assemble_snapshot_from_frames(
    *,
    config: PitBuildConfig,
    trade_dates: pd.DataFrame,
    security_master: pd.DataFrame,
    stock_lists: pd.DataFrame,
    daily_bars: pd.DataFrame,
    daily_metrics: pd.DataFrame,
    stock_industry: pd.DataFrame,
    daily_size: pd.DataFrame,
    failures: list[dict[str, Any]],
    source_version: str,
    source_errors: list[dict[str, str]],
    extra_quality: dict[str, Any],
) -> dict[str, Any]:
    if len(trade_dates) == 0:
        raise BaostockSourceError("trade date count is zero")
    daily_bars = _filter_bars_to_stock_lists(daily_bars, stock_lists)
    daily_metrics = _filter_metrics_to_stock_lists(
        daily_metrics if daily_metrics is not None else pd.DataFrame(columns=DAILY_METRICS_COLUMNS),
        stock_lists,
    )
    stock_industry = _filter_stock_industry_to_stock_lists(
        stock_industry if stock_industry is not None else pd.DataFrame(columns=STOCK_INDUSTRY_COLUMNS),
        stock_lists,
    )
    daily_size = _filter_daily_size_to_stock_lists(
        daily_size if daily_size is not None else pd.DataFrame(columns=DAILY_SIZE_COLUMNS),
        stock_lists,
    )
    daily_universe, daily_status = build_daily_universe(stock_lists, security_master, daily_bars)
    snapshot_id = config.snapshot_id or f"baostock_daily_mainboard_v2_pit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    snapshot_root = config.output_root / snapshot_id
    snapshot_root.mkdir(parents=True, exist_ok=False)
    manifest = _manifest(
        config=config,
        snapshot_id=snapshot_id,
        snapshot_root=snapshot_root,
        source_version=source_version,
        trade_dates=trade_dates,
        security_master=security_master,
        daily_universe=daily_universe,
        daily_bars=daily_bars,
        daily_status=daily_status,
        daily_metrics=daily_metrics,
        stock_industry=stock_industry,
        daily_size=daily_size,
        failures=failures,
        source_errors=source_errors,
        extra_quality=extra_quality,
    )
    quality_report = _quality_report(daily_universe, daily_status, failures, source_errors, extra_quality)
    _write_outputs(
        snapshot_root,
        config.output_root,
        security_master,
        stock_lists,
        daily_universe,
        daily_bars,
        daily_status,
        daily_metrics,
        stock_industry,
        daily_size,
        manifest,
        quality_report,
    )
    return manifest


def _filter_bars_to_stock_lists(daily_bars: pd.DataFrame, stock_lists: pd.DataFrame) -> pd.DataFrame:
    if daily_bars.empty or stock_lists.empty:
        return daily_bars
    keys = stock_lists[["date", "code"]].drop_duplicates()
    return daily_bars.merge(keys, on=["date", "code"], how="inner").reset_index(drop=True)


def _filter_metrics_to_stock_lists(daily_metrics: pd.DataFrame, stock_lists: pd.DataFrame) -> pd.DataFrame:
    daily_metrics = _ensure_columns(daily_metrics, DAILY_METRICS_COLUMNS)
    if daily_metrics.empty or stock_lists.empty:
        return daily_metrics
    keys = stock_lists[["date", "code"]].drop_duplicates()
    return daily_metrics.merge(keys, on=["date", "code"], how="inner")[DAILY_METRICS_COLUMNS].reset_index(drop=True)


def _filter_daily_size_to_stock_lists(daily_size: pd.DataFrame, stock_lists: pd.DataFrame) -> pd.DataFrame:
    daily_size = _ensure_columns(daily_size, DAILY_SIZE_COLUMNS)
    if daily_size.empty or stock_lists.empty:
        return daily_size
    keys = stock_lists[["date", "code"]].drop_duplicates().copy()
    keys["date"] = pd.to_datetime(keys["date"]).astype("datetime64[ns]")
    output = daily_size.copy()
    output["date"] = pd.to_datetime(output["date"]).astype("datetime64[ns]")
    return output.merge(keys, on=["date", "code"], how="inner")[DAILY_SIZE_COLUMNS].reset_index(drop=True)


def _assert_failure_rate(year: int, code_count: int, failures: list[dict[str, Any]]) -> None:
    if code_count <= 0:
        return
    rate = len(failures) / code_count
    if rate > 0.05:
        raise BaostockSourceError(f"daily bar failure rate too high for {year}: {rate:.2%}")


def _manifest(
    *,
    config: PitBuildConfig,
    snapshot_id: str,
    snapshot_root: Path,
    source_version: str,
    trade_dates: pd.DataFrame,
    security_master: pd.DataFrame,
    daily_universe: pd.DataFrame,
    daily_bars: pd.DataFrame,
    daily_status: pd.DataFrame,
    daily_metrics: pd.DataFrame,
    stock_industry: pd.DataFrame,
    daily_size: pd.DataFrame,
    failures: list[dict[str, Any]],
    source_errors: list[dict[str, str]],
    extra_quality: dict[str, Any],
) -> dict[str, Any]:
    years = _years_between(config.start_date, config.end_date)
    return {
        "schema_version": 2,
        "snapshot_id": snapshot_id,
        "snapshot_path": str(snapshot_root),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": config.command,
        "source": {"name": "baostock", "version": source_version, "cache_policy": "yearly_cache_remote_read_only"},
        "dataset": {
            "frequency": "1d",
            "start_date_requested": config.start_date,
            "end_date_requested": config.end_date,
            "years": years,
            "date_min": str(daily_bars["date"].min().date()) if not daily_bars.empty else "",
            "date_max": str(daily_bars["date"].max().date()) if not daily_bars.empty else "",
            "trade_date_count": int(len(trade_dates)),
            "security_count": int(len(security_master)),
            "daily_universe_rows": int(len(daily_universe)),
            "daily_bar_rows": int(len(daily_bars)),
            "daily_status_rows": int(len(daily_status)),
            "daily_metrics_rows": int(len(daily_metrics)),
            "stock_industry_rows": int(len(stock_industry)),
            "daily_size_rows": int(len(daily_size)),
            "fields": list(BAOSTOCK_DAILY_FIELDS),
            "daily_metrics_fields": DAILY_METRICS_COLUMNS,
            "industry_fields": STOCK_INDUSTRY_COLUMNS,
            "daily_size_fields": DAILY_SIZE_COLUMNS,
        },
        "quality": {
            "failure_count": len(failures),
            "source_error_count": len(source_errors),
            "tradeable_rows": int(daily_status["is_tradeable"].sum()) if not daily_status.empty else 0,
            "suspended_like_rows": int(daily_status["is_suspended_like"].sum()) if not daily_status.empty else 0,
            "st_rows": int(daily_universe["is_st_on_date"].sum()) if not daily_universe.empty else 0,
            **extra_quality,
        },
        "bias_statement": PIT_BIAS_STATEMENT,
    }


def _quality_report(
    daily_universe: pd.DataFrame,
    daily_status: pd.DataFrame,
    failures: list[dict[str, Any]],
    source_errors: list[dict[str, str]],
    extra_quality: dict[str, Any],
) -> dict[str, Any]:
    daily_counts = pd.DataFrame(
        columns=["date", "universe_rows", "tradeable_rows", "st_rows", "suspended_rows", "missing_bar_rows"]
    )
    if not daily_universe.empty:
        enriched = daily_universe.copy()
        enriched["missing_bar"] = enriched["reject_reason"] == "missing_bar"
        daily_counts = (
            enriched.groupby("date")
            .agg(
                universe_rows=("code", "count"),
                tradeable_rows=("is_tradeable", "sum"),
                st_rows=("is_st_on_date", "sum"),
                suspended_rows=("is_suspended_on_date", "sum"),
                missing_bar_rows=("missing_bar", "sum"),
            )
            .reset_index()
        )
        daily_counts["date"] = pd.to_datetime(daily_counts["date"]).dt.strftime("%Y-%m-%d")
    yearly_counts = pd.DataFrame(
        columns=["year", "trade_dates", "universe_rows", "daily_bar_rows", "tradeable_rows", "st_rows", "suspended_rows", "missing_bar_rows", "failure_count"]
    )
    if not daily_universe.empty:
        enriched = daily_universe.copy()
        enriched["year"] = pd.to_datetime(enriched["date"]).dt.year
        enriched["missing_bar"] = enriched["reject_reason"] == "missing_bar"
        yearly_counts = (
            enriched.groupby("year")
            .agg(
                trade_dates=("date", "nunique"),
                universe_rows=("code", "count"),
                tradeable_rows=("is_tradeable", "sum"),
                st_rows=("is_st_on_date", "sum"),
                suspended_rows=("is_suspended_on_date", "sum"),
                missing_bar_rows=("missing_bar", "sum"),
            )
            .reset_index()
        )
        if not daily_status.empty:
            bar_counts = daily_status.copy()
            bar_counts["year"] = pd.to_datetime(bar_counts["date"]).dt.year
            bar_counts = bar_counts.groupby("year").size().rename("daily_bar_rows").reset_index()
            yearly_counts = yearly_counts.merge(bar_counts, on="year", how="left")
        else:
            yearly_counts["daily_bar_rows"] = 0
        failure_counts = pd.DataFrame(failures)
        if not failure_counts.empty and "year" in failure_counts.columns:
            failure_counts = failure_counts.groupby("year").size().rename("failure_count").reset_index()
            yearly_counts = yearly_counts.merge(failure_counts, on="year", how="left")
        else:
            yearly_counts["failure_count"] = 0
        yearly_counts["daily_bar_rows"] = yearly_counts["daily_bar_rows"].fillna(0).astype(int)
        yearly_counts["failure_count"] = yearly_counts["failure_count"].fillna(0).astype(int)
    return {
        "failure_count": len(failures),
        "failures": failures,
        "source_errors": source_errors,
        "daily_summary": daily_counts.to_dict(orient="records"),
        "yearly_summary": yearly_counts.to_dict(orient="records"),
        "st_rows": int(daily_universe["is_st_on_date"].sum()) if not daily_universe.empty else 0,
        "suspended_like_rows": int(daily_status["is_suspended_like"].sum()) if not daily_status.empty else 0,
        "missing_bar_rows": int((daily_universe["reject_reason"] == "missing_bar").sum()) if not daily_universe.empty else 0,
        "missing_basic_rows": int((daily_universe["reject_reason"] == "not_common_a_share").sum()) if not daily_universe.empty else 0,
        "extra": extra_quality,
    }


def _write_outputs(
    snapshot_root: Path,
    output_root: Path,
    security_master: pd.DataFrame,
    raw_daily_stock_lists: pd.DataFrame,
    daily_universe: pd.DataFrame,
    daily_bars: pd.DataFrame,
    daily_status: pd.DataFrame,
    daily_metrics: pd.DataFrame,
    stock_industry: pd.DataFrame,
    daily_size: pd.DataFrame,
    manifest: dict[str, Any],
    quality_report: dict[str, Any],
) -> None:
    security_master.to_parquet(snapshot_root / "security_master.parquet", index=False)
    raw_daily_stock_lists.to_parquet(snapshot_root / "raw_daily_stock_lists.parquet", index=False)
    daily_universe.to_parquet(snapshot_root / "daily_universe.parquet", index=False)
    daily_bars.to_parquet(snapshot_root / "daily_bars.parquet", index=False)
    daily_status.to_parquet(snapshot_root / "daily_status.parquet", index=False)
    if not daily_metrics.empty:
        daily_metrics.to_parquet(snapshot_root / "daily_metrics.parquet", index=False)
    if not stock_industry.empty:
        stock_industry.to_parquet(snapshot_root / "stock_industry.parquet", index=False)
    if not daily_size.empty:
        daily_size.to_parquet(snapshot_root / "daily_size.parquet", index=False)
    (snapshot_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (snapshot_root / "quality_report.json").write_text(json.dumps(quality_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    latest = {
        "snapshot_id": manifest["snapshot_id"],
        "snapshot_path": str(snapshot_root),
        "manifest_path": str(snapshot_root / "manifest.json"),
        "created_at": manifest["created_at"],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "latest_manifest.json").write_text(json.dumps(latest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_validated_year_cache(
    output_root: Path,
    year: int,
    start_date: str,
    end_date: str,
    *,
    config: PitBuildConfig | None,
    data_path_fn: Any,
    meta_path_fn: Any,
    columns: list[str],
) -> pd.DataFrame:
    path = _call_cache_path_fn(data_path_fn, output_root, year, config=config, start_date=start_date, end_date=end_date)
    if config is not None:
        meta = _read_json_or_empty(_call_cache_path_fn(meta_path_fn, output_root, year, config=config, start_date=start_date, end_date=end_date))
        if not _is_cache_compatible(meta, config, year=year, start_date=start_date, end_date=end_date):
            return pd.DataFrame(columns=columns)
    return _read_parquet_or_empty(path, columns)


def _call_cache_path_fn(
    path_fn: Any,
    output_root: Path,
    year: int,
    *,
    config: PitBuildConfig | None,
    start_date: str,
    end_date: str,
) -> Path:
    try:
        return path_fn(output_root, year, config=config, start_date=start_date, end_date=end_date)
    except TypeError:
        return path_fn(output_root, year)


def load_cached_stock_lists(
    output_root: Path,
    years: list[int],
    start_date: str,
    end_date: str,
    *,
    config: PitBuildConfig | None = None,
) -> pd.DataFrame:
    parts = [
        _read_validated_year_cache(
            output_root,
            year,
            max(start_date, f"{year}-01-01"),
            min(end_date, f"{year}-12-31"),
            config=config,
            data_path_fn=_stock_list_cache_path,
            meta_path_fn=_stock_list_meta_path,
            columns=RAW_STOCK_LIST_COLUMNS,
        )
        for year in years
    ]
    return _filter_dates(_concat(parts, RAW_STOCK_LIST_COLUMNS), start_date, end_date)


def load_cached_daily_bars(
    output_root: Path,
    years: list[int],
    start_date: str,
    end_date: str,
    *,
    config: PitBuildConfig | None = None,
) -> pd.DataFrame:
    parts = [
        _read_validated_year_cache(
            output_root,
            year,
            max(start_date, f"{year}-01-01"),
            min(end_date, f"{year}-12-31"),
            config=config,
            data_path_fn=_daily_bars_cache_path,
            meta_path_fn=_daily_bars_meta_path,
            columns=DAILY_BARS_COLUMNS,
        )
        for year in years
    ]
    return _filter_dates(_concat(parts, DAILY_BARS_COLUMNS), start_date, end_date)


def load_cached_daily_metrics(
    output_root: Path,
    years: list[int],
    start_date: str,
    end_date: str,
    *,
    config: PitBuildConfig | None = None,
) -> pd.DataFrame:
    parts = [
        _read_validated_year_cache(
            output_root,
            year,
            max(start_date, f"{year}-01-01"),
            min(end_date, f"{year}-12-31"),
            config=config,
            data_path_fn=_daily_metrics_cache_path,
            meta_path_fn=_daily_metrics_meta_path,
            columns=DAILY_METRICS_COLUMNS,
        )
        for year in years
    ]
    return _filter_dates(_concat(parts, DAILY_METRICS_COLUMNS), start_date, end_date)


def load_cached_stock_industry(
    output_root: Path,
    years: list[int],
    start_date: str,
    end_date: str,
    *,
    config: PitBuildConfig | None = None,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for year in years:
        year_start = max(start_date, f"{year}-01-01")
        year_end = min(end_date, f"{year}-12-31")
        path = _stock_industry_cache_path(output_root, year)
        if config is not None:
            meta = _read_json_or_empty(_stock_industry_meta_path(output_root, year))
            if not _is_industry_cache_compatible(meta, config, year=year, start_date=year_start, end_date=year_end):
                parts.append(pd.DataFrame(columns=STOCK_INDUSTRY_COLUMNS))
                continue
        parts.append(_read_parquet_or_empty(path, STOCK_INDUSTRY_COLUMNS))
    return _filter_dates(_concat(parts, STOCK_INDUSTRY_COLUMNS), start_date, end_date)


def _parse_symbols(raw: str) -> tuple[str, ...]:
    if not raw.strip():
        return ()
    return tuple(item.strip().upper() for item in raw.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Baostock point-in-time daily research snapshots.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("dry-run", "build", "refresh", "rebuild", "discover", "fetch-bars", "fetch-metrics", "fetch-industry", "fetch-size", "assemble", "build-year"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--symbols", default="", help="Comma-separated symbols such as 600000.SH,000001.SZ.")
        cmd.add_argument("--max-symbols", type=int, default=0, help="Limit selected symbols after PIT universe discovery.")
        cmd.add_argument("--output-root", default=str(DEFAULT_V2_SNAPSHOT_ROOT))
        cmd.add_argument("--start-date", default="2016-01-01")
        cmd.add_argument("--end-date", default="2026-06-01")
        cmd.add_argument("--trade-dates", default="", help="Comma-separated YYYYMMDD dates for fetch-size smoke runs.")
        cmd.add_argument("--snapshot-id", default="")
        cmd.add_argument("--year", type=int, default=0)
        cmd.add_argument("--force-refresh", action="store_true")
        cmd.add_argument("--request-interval-seconds", type=float, default=0.0)
        cmd.add_argument("--discovery-mode", choices=["stock-basic", "daily"], default="stock-basic")
        cmd.add_argument("--include-industry", action="store_true", help="Fetch/write optional Baostock stock_industry table.")
        cmd.add_argument("--industry-frequency", choices=["daily", "month-start"], default="daily")
        cmd.add_argument("--include-metrics", action="store_true", help="Fetch/write optional Baostock daily_metrics table.")
        cmd.add_argument("--include-size", action="store_true", help="Read/write optional external daily_size table from cached size data.")
        cmd.add_argument(
            "--size-source",
            choices=SIZE_SOURCE_CHOICES,
            default=SIZE_SOURCE_TUSHARE_DAILY_BASIC,
            help="Source backend for fetch-size. Tushare remains the default; free/proxy sources are diagnostic until audited.",
        )
        cmd.add_argument("--size-token", default="", help="Tushare token for fetch-size; falls back to TUSHARE_TOKEN or TS_TOKEN.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    symbols = _parse_symbols(args.symbols)
    trade_dates = tuple(item.strip() for item in args.trade_dates.split(",") if item.strip())
    if args.command == "dry-run" and not symbols and args.max_symbols <= 0:
        symbols = ("600000.SH", "000001.SZ")
    config = PitBuildConfig(
        output_root=Path(args.output_root),
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=symbols,
        max_symbols=args.max_symbols,
        command=args.command,
        snapshot_id=args.snapshot_id,
        year=args.year,
        trade_dates=trade_dates,
        force_refresh=args.force_refresh,
        request_interval_seconds=args.request_interval_seconds,
        discovery_mode=args.discovery_mode,
        include_industry=args.include_industry,
        industry_frequency=args.industry_frequency,
        include_metrics=args.include_metrics,
        include_size=args.include_size,
        size_source=args.size_source,
        size_token=args.size_token or None,
    )
    if args.command in {"dry-run", "build", "refresh", "rebuild"}:
        manifest = build_snapshot(config)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    elif args.command == "discover":
        print(json.dumps(discover(config), ensure_ascii=False, indent=2))
    elif args.command == "fetch-bars":
        print(json.dumps(fetch_bars(config), ensure_ascii=False, indent=2))
    elif args.command == "fetch-metrics":
        print(json.dumps(fetch_metrics(config), ensure_ascii=False, indent=2))
    elif args.command == "fetch-industry":
        print(json.dumps(fetch_industry(config), ensure_ascii=False, indent=2))
    elif args.command == "fetch-size":
        print(json.dumps(fetch_size(config), ensure_ascii=False, indent=2))
    elif args.command == "assemble":
        print(json.dumps(assemble(config), ensure_ascii=False, indent=2))
    elif args.command == "build-year":
        print(json.dumps(build_year(config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
