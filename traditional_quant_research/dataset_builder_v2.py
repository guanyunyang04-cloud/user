"""Build v2 point-in-time Shanghai/Shenzhen mainboard daily datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .baostock_source import BaostockSource, BaostockSourceConfig, BaostockSourceError
from .dataset_v2 import DEFAULT_V2_SNAPSHOT_ROOT
from .universe import is_sh_sz_mainboard_a_share


PIT_BIAS_STATEMENT = (
    "v2 constructs the universe date by date from Baostock point-in-time stock lists, "
    "security master dates, daily tradestatus, and daily isST flags. It is intended "
    "to reduce survivorship bias versus v1, while remaining limited by Baostock field "
    "coverage and historical data quality."
)
BAOSTOCK_DAILY_FIELDS = ("date", "code", "open", "high", "low", "close", "volume", "amount", "tradestatus", "isST")
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
    force_refresh: bool = False
    request_interval_seconds: float = 0.0


def _cache_root(output_root: Path) -> Path:
    return output_root / "cache"


def _stock_list_cache_path(output_root: Path, year: int) -> Path:
    return _cache_root(output_root) / "daily_stock_lists" / f"year={year}.parquet"


def _daily_bars_cache_path(output_root: Path, year: int) -> Path:
    return _cache_root(output_root) / "daily_bars" / f"year={year}.parquet"


def _security_master_cache_path(output_root: Path) -> Path:
    return _cache_root(output_root) / "security_master.parquet"


def _trade_dates_cache_path(output_root: Path) -> Path:
    return _cache_root(output_root) / "trade_dates.parquet"


def _progress_path(output_root: Path, year: int) -> Path:
    return _cache_root(output_root) / "progress" / f"year={year}.json"


def _monthly_stock_list_cache_path(output_root: Path, year: int, month: int) -> Path:
    return _cache_root(output_root) / "daily_stock_lists" / "parts" / f"year={year}" / f"month={month:02d}.parquet"


def _daily_bars_part_cache_path(output_root: Path, year: int, batch_index: int) -> Path:
    return _cache_root(output_root) / "daily_bars" / "parts" / f"year={year}" / f"batch={batch_index:04d}.parquet"


def _read_progress(output_root: Path, year: int) -> dict[str, Any]:
    path = _progress_path(output_root, year)
    if not path.exists():
        return {"year": year, "completed_stock_dates": [], "completed_bar_codes": [], "failures": []}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_progress(output_root: Path, year: int, progress: dict[str, Any]) -> None:
    path = _progress_path(output_root, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    cache_hit = int(path.exists() and not config.force_refresh)
    if cache_hit:
        cached = pd.read_parquet(path)
        return _filter_dates(cached, config.start_date, config.end_date), {"trade_dates_cache_hit": 1, "trade_dates_cache_miss": 0}

    raw = source.query_trade_dates(config.start_date, config.end_date)
    if raw.empty:
        frame = pd.DataFrame(columns=["date", "is_trading_day"])
    else:
        frame = raw.rename(columns={"calendar_date": "date"}).copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["is_trading_day"] = frame["is_trading_day"].astype(str) == "1"
        frame = frame.loc[frame["is_trading_day"], ["date", "is_trading_day"]].reset_index(drop=True)
    _write_parquet(path, frame)
    return frame, {"trade_dates_cache_hit": 0, "trade_dates_cache_miss": 1}


def discover_daily_stock_lists(
    source: BaostockSource,
    dates: pd.Series,
    *,
    config: PitBuildConfig,
    year: int,
    explicit_symbols: set[str] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _stock_list_cache_path(config.output_root, year)
    if path.exists() and not config.force_refresh:
        frame = pd.read_parquet(path)
        frame = _filter_dates(frame, config.start_date, config.end_date)
        if explicit_symbols is not None:
            frame = frame[frame["code"].isin(explicit_symbols)].reset_index(drop=True)
        return frame, {"stock_list_cache_hit": 1, "stock_list_cache_miss": 0, "empty_stock_list_dates": []}

    parts: list[pd.DataFrame] = []
    empty_dates: list[str] = []
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
        return output, {"stock_list_cache_hit": 0, "stock_list_cache_miss": 1, "empty_stock_list_dates": [], "dry_run_symbol_mode": 1}

    progress = {"year": year, "completed_stock_dates": [], "completed_bar_codes": [], "failures": []} if config.force_refresh else _read_progress(config.output_root, year)
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
        existing_month = pd.DataFrame(columns=RAW_STOCK_LIST_COLUMNS) if config.force_refresh else _read_parquet_or_empty(month_path, RAW_STOCK_LIST_COLUMNS)
        month_output = _concat([existing_month, frame], RAW_STOCK_LIST_COLUMNS).drop_duplicates(["date", "code"])
        _write_parquet(month_path, month_output)
        progress["completed_stock_dates"] = sorted(set(progress.get("completed_stock_dates", [])) | {date_str})
        _write_progress(config.output_root, year, progress)

    months_to_read = sorted(touched_months) if config.force_refresh else list(range(1, 13))
    month_parts = [
        _read_parquet_or_empty(_monthly_stock_list_cache_path(config.output_root, year, month), RAW_STOCK_LIST_COLUMNS)
        for month in months_to_read
    ]
    output = _concat([*month_parts, *parts], RAW_STOCK_LIST_COLUMNS).drop_duplicates(["date", "code"])
    output = _filter_dates(output, config.start_date, config.end_date)
    _write_parquet(path, output)
    return output, {"stock_list_cache_hit": 0, "stock_list_cache_miss": 1, "empty_stock_list_dates": empty_dates}


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
    if path.exists() and not config.force_refresh:
        frame = pd.read_parquet(path)
        frame = _filter_dates(frame, start_date, end_date)
        if config.symbols:
            frame = frame[frame["code"].isin(set(config.symbols))].reset_index(drop=True)
        return frame, [], {"daily_bars_cache_hit": 1, "daily_bars_cache_miss": 0}

    parts: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    progress = {"year": year, "completed_stock_dates": [], "completed_bar_codes": [], "failures": []} if config.force_refresh else _read_progress(config.output_root, year)
    completed_codes = set(progress.get("completed_bar_codes", []))
    batch_size = 50
    for batch_index, start in enumerate(range(0, len(codes), batch_size)):
        batch_codes = codes[start : start + batch_size]
        part_path = _daily_bars_part_cache_path(config.output_root, year, batch_index)
        existing_part = pd.DataFrame(columns=DAILY_BARS_COLUMNS) if config.force_refresh else _read_parquet_or_empty(part_path, DAILY_BARS_COLUMNS)
        batch_parts: list[pd.DataFrame] = [existing_part] if not existing_part.empty else []
        for code in batch_codes:
            if code in completed_codes and not existing_part.empty and code in set(existing_part["code"].astype(str)):
                continue
            try:
                raw = source.query_daily_bars(_std_to_baostock_code(code), start_date, end_date)
            except BaostockSourceError as exc:
                failure = {"year": year, "code": code, "kind": "daily_fetch_error", "error": str(exc)}
                failures.append(failure)
                progress["failures"] = [*progress.get("failures", []), failure]
                _write_progress(config.output_root, year, progress)
                continue
            if raw.empty:
                failure = {"year": year, "code": code, "kind": "empty_daily_bars"}
                failures.append(failure)
                progress["failures"] = [*progress.get("failures", []), failure]
                _write_progress(config.output_root, year, progress)
                continue
            frame = normalize_daily_bars(raw)
            batch_parts.append(frame)
            progress["completed_bar_codes"] = sorted(set(progress.get("completed_bar_codes", [])) | {code})
            _write_progress(config.output_root, year, progress)
        batch_output = _concat(batch_parts, DAILY_BARS_COLUMNS).drop_duplicates(["date", "code"])
        if not batch_output.empty:
            _write_parquet(part_path, batch_output)
            parts.append(batch_output)

    part_paths = [] if config.force_refresh else sorted((_cache_root(config.output_root) / "daily_bars" / "parts" / f"year={year}").glob("batch=*.parquet"))
    part_frames = [_read_parquet_or_empty(part_path, DAILY_BARS_COLUMNS) for part_path in part_paths]
    output = _concat([*part_frames, *parts], DAILY_BARS_COLUMNS).drop_duplicates(["date", "code"])
    output = _filter_dates(output, start_date, end_date)
    if codes:
        output = output[output["code"].isin(set(codes))].reset_index(drop=True)
    _write_parquet(path, output)
    return output, failures, {"daily_bars_cache_hit": 0, "daily_bars_cache_miss": 1}


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
        summaries: list[dict[str, Any]] = []
        for year in _years_between(config.start_date, config.end_date):
            start, end = _year_bounds(year, config.start_date, config.end_date)
            year_dates = _filter_dates(trade_dates, start, end)["date"]
            stock_lists, summary = discover_daily_stock_lists(source, year_dates, config=config, year=year, explicit_symbols=explicit)
            summaries.append({"year": year, "rows": len(stock_lists), **summary})
        stock_lists = load_cached_stock_lists(config.output_root, _years_between(config.start_date, config.end_date), config.start_date, config.end_date)
        security_master, security_summary = update_security_master(source, stock_lists, config=config)
        return {
            "command": "discover",
            "trade_dates": len(trade_dates),
            "stock_list_years": summaries,
            "security_master_rows": len(security_master),
            "cache": trade_cache,
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
            stock_lists = load_cached_stock_lists(config.output_root, [year], start, end)
            codes = sorted(stock_lists["code"].unique().tolist())
            if config.max_symbols > 0:
                codes = codes[: config.max_symbols]
            if config.symbols:
                codes = [code for code in codes if code in set(config.symbols)]
            bars, failures, cache = fetch_daily_bars(source, codes, config=config, year=year, start_date=start, end_date=end)
            all_failures.extend(failures)
            summaries.append({"year": year, "codes": len(codes), "rows": len(bars), "failures": len(failures), **cache})
        return {"command": "fetch-bars", "years": summaries, "failure_count": len(all_failures), "failures": all_failures, "source_errors": source.errors}


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
        force_refresh=config.force_refresh,
        request_interval_seconds=config.request_interval_seconds,
    )
    with BaostockSource(_source_config(year_config)) as source:
        trade_dates, trade_cache = _trade_dates(source, year_config)
        explicit = set(config.symbols) if config.symbols else None
        stock_lists, stock_cache = discover_daily_stock_lists(source, trade_dates["date"], config=year_config, year=year, explicit_symbols=explicit)
        if stock_lists.empty and not trade_dates.empty:
            raise BaostockSourceError(f"empty stock list for year {year}")
        if config.max_symbols > 0:
            codes = sorted(stock_lists["code"].unique().tolist())[: config.max_symbols]
            stock_lists = stock_lists[stock_lists["code"].isin(codes)].reset_index(drop=True)
        security_master, security_summary = update_security_master(source, stock_lists, config=year_config)
        codes = sorted(stock_lists["code"].unique().tolist())
        daily_bars, failures, bar_cache = fetch_daily_bars(source, codes, config=year_config, year=year, start_date=start, end_date=end)
        _assert_failure_rate(year, len(codes), failures)
        manifest = assemble_snapshot_from_frames(
            config=year_config,
            trade_dates=trade_dates,
            security_master=security_master,
            stock_lists=stock_lists,
            daily_bars=daily_bars,
            failures=failures,
            source_version=source.version,
            source_errors=source.errors,
            extra_quality={**trade_cache, **stock_cache, **bar_cache, **security_summary},
        )
        return manifest


def assemble(config: PitBuildConfig) -> dict[str, Any]:
    years = [config.year] if config.year else _years_between(config.start_date, config.end_date)
    trade_dates = _filter_dates(_read_parquet_or_empty(_trade_dates_cache_path(config.output_root), ["date", "is_trading_day"]), config.start_date, config.end_date)
    stock_lists = load_cached_stock_lists(config.output_root, years, config.start_date, config.end_date)
    if config.max_symbols > 0:
        codes = sorted(stock_lists["code"].unique().tolist())[: config.max_symbols]
        stock_lists = stock_lists[stock_lists["code"].isin(codes)].reset_index(drop=True)
    if config.symbols:
        stock_lists = stock_lists[stock_lists["code"].isin(set(config.symbols))].reset_index(drop=True)
    security_master = _read_parquet_or_empty(_security_master_cache_path(config.output_root), SECURITY_MASTER_COLUMNS)
    if not stock_lists.empty:
        security_master = security_master[security_master["code"].isin(set(stock_lists["code"]))].reset_index(drop=True)
    daily_bars = load_cached_daily_bars(config.output_root, years, config.start_date, config.end_date)
    if not stock_lists.empty:
        daily_bars = daily_bars[daily_bars["code"].isin(set(stock_lists["code"]))].reset_index(drop=True)
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
        failures=[],
        source_version="cache",
        source_errors=[],
        extra_quality={"assembled_from_cache": 1},
    )


def build_snapshot(config: PitBuildConfig) -> dict[str, Any]:
    with BaostockSource(_source_config(config)) as source:
        trade_dates, trade_cache = _trade_dates(source, config)
        explicit = set(config.symbols) if config.symbols else None
        stock_parts: list[pd.DataFrame] = []
        stock_quality: dict[str, Any] = {}
        for year in _years_between(config.start_date, config.end_date):
            start, end = _year_bounds(year, config.start_date, config.end_date)
            year_dates = _filter_dates(trade_dates, start, end)["date"]
            stock_lists, summary = discover_daily_stock_lists(source, year_dates, config=config, year=year, explicit_symbols=explicit)
            stock_quality[f"stock_list_year_{year}"] = summary
            stock_parts.append(stock_lists)
        stock_lists = _concat(stock_parts, RAW_STOCK_LIST_COLUMNS)
        if stock_lists.empty and not trade_dates.empty:
            raise BaostockSourceError("empty stock list during build")
        if config.max_symbols > 0:
            codes = sorted(stock_lists["code"].unique().tolist())[: config.max_symbols]
            stock_lists = stock_lists[stock_lists["code"].isin(codes)].reset_index(drop=True)
        security_master, security_summary = update_security_master(source, stock_lists, config=config)
        bar_parts: list[pd.DataFrame] = []
        failures: list[dict[str, Any]] = []
        bar_quality: dict[str, Any] = {}
        for year in _years_between(config.start_date, config.end_date):
            start, end = _year_bounds(year, config.start_date, config.end_date)
            year_stock_lists = _filter_dates(stock_lists, start, end)
            codes = sorted(year_stock_lists["code"].unique().tolist())
            bars, year_failures, cache = fetch_daily_bars(source, codes, config=config, year=year, start_date=start, end_date=end)
            _assert_failure_rate(year, len(codes), year_failures)
            bar_parts.append(bars)
            failures.extend(year_failures)
            bar_quality[f"daily_bars_year_{year}"] = cache
        daily_bars = _concat(bar_parts, DAILY_BARS_COLUMNS)
        return assemble_snapshot_from_frames(
            config=config,
            trade_dates=trade_dates,
            security_master=security_master,
            stock_lists=stock_lists,
            daily_bars=daily_bars,
            failures=failures,
            source_version=source.version,
            source_errors=source.errors,
            extra_quality={**trade_cache, **security_summary, **stock_quality, **bar_quality},
        )


def assemble_snapshot_from_frames(
    *,
    config: PitBuildConfig,
    trade_dates: pd.DataFrame,
    security_master: pd.DataFrame,
    stock_lists: pd.DataFrame,
    daily_bars: pd.DataFrame,
    failures: list[dict[str, Any]],
    source_version: str,
    source_errors: list[dict[str, str]],
    extra_quality: dict[str, Any],
) -> dict[str, Any]:
    if len(trade_dates) == 0:
        raise BaostockSourceError("trade date count is zero")
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
        failures=failures,
        source_errors=source_errors,
        extra_quality=extra_quality,
    )
    quality_report = _quality_report(daily_universe, daily_status, failures, source_errors, extra_quality)
    _write_outputs(snapshot_root, config.output_root, security_master, stock_lists, daily_universe, daily_bars, daily_status, manifest, quality_report)
    return manifest


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
            "fields": list(BAOSTOCK_DAILY_FIELDS),
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
    manifest: dict[str, Any],
    quality_report: dict[str, Any],
) -> None:
    security_master.to_parquet(snapshot_root / "security_master.parquet", index=False)
    raw_daily_stock_lists.to_parquet(snapshot_root / "raw_daily_stock_lists.parquet", index=False)
    daily_universe.to_parquet(snapshot_root / "daily_universe.parquet", index=False)
    daily_bars.to_parquet(snapshot_root / "daily_bars.parquet", index=False)
    daily_status.to_parquet(snapshot_root / "daily_status.parquet", index=False)
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


def load_cached_stock_lists(output_root: Path, years: list[int], start_date: str, end_date: str) -> pd.DataFrame:
    parts = [_read_parquet_or_empty(_stock_list_cache_path(output_root, year), RAW_STOCK_LIST_COLUMNS) for year in years]
    return _filter_dates(_concat(parts, RAW_STOCK_LIST_COLUMNS), start_date, end_date)


def load_cached_daily_bars(output_root: Path, years: list[int], start_date: str, end_date: str) -> pd.DataFrame:
    parts = [_read_parquet_or_empty(_daily_bars_cache_path(output_root, year), DAILY_BARS_COLUMNS) for year in years]
    return _filter_dates(_concat(parts, DAILY_BARS_COLUMNS), start_date, end_date)


def _parse_symbols(raw: str) -> tuple[str, ...]:
    if not raw.strip():
        return ()
    return tuple(item.strip().upper() for item in raw.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Baostock point-in-time daily research snapshots.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("dry-run", "build", "refresh", "rebuild", "discover", "fetch-bars", "assemble", "build-year"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--symbols", default="", help="Comma-separated symbols such as 600000.SH,000001.SZ.")
        cmd.add_argument("--max-symbols", type=int, default=0, help="Limit selected symbols after PIT universe discovery.")
        cmd.add_argument("--output-root", default=str(DEFAULT_V2_SNAPSHOT_ROOT))
        cmd.add_argument("--start-date", default="2016-01-01")
        cmd.add_argument("--end-date", default="2026-06-01")
        cmd.add_argument("--snapshot-id", default="")
        cmd.add_argument("--year", type=int, default=0)
        cmd.add_argument("--force-refresh", action="store_true")
        cmd.add_argument("--request-interval-seconds", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    symbols = _parse_symbols(args.symbols)
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
        force_refresh=args.force_refresh,
        request_interval_seconds=args.request_interval_seconds,
    )
    if args.command in {"dry-run", "build", "refresh", "rebuild"}:
        manifest = build_snapshot(config)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    elif args.command == "discover":
        print(json.dumps(discover(config), ensure_ascii=False, indent=2))
    elif args.command == "fetch-bars":
        print(json.dumps(fetch_bars(config), ensure_ascii=False, indent=2))
    elif args.command == "assemble":
        print(json.dumps(assemble(config), ensure_ascii=False, indent=2))
    elif args.command == "build-year":
        print(json.dumps(build_year(config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
