"""Normalize and cache external PIT size sources into the v2 daily_size schema."""

from __future__ import annotations

import importlib
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .dataset_v2 import DAILY_SIZE_COLUMNS


TUSHARE_DAILY_BASIC_SOURCE = "tushare.daily_basic"
TUSHARE_MARKET_CAP_UNIT = "10k CNY"
TUSHARE_SHARE_UNIT = "10k shares"
TUSHARE_DAILY_BASIC_SIZE_FIELDS = (
    "ts_code",
    "trade_date",
    "total_mv",
    "circ_mv",
    "total_share",
    "float_share",
    "free_share",
)
TUSHARE_DAILY_SIZE_FAILURE_COLUMNS = ("trade_date", "error_type", "message")

_TUSHARE_TO_DAILY_SIZE = {
    "total_mv": "total_market_cap",
    "circ_mv": "float_market_cap",
    "total_share": "total_share",
    "float_share": "float_share",
    "free_share": "free_share",
}
_DAILY_SIZE_NUMERIC_COLUMNS = [
    "total_market_cap",
    "float_market_cap",
    "total_share",
    "float_share",
    "free_share",
]


def standardize_tushare_daily_basic_size(
    frame: pd.DataFrame | None,
    *,
    source: str = TUSHARE_DAILY_BASIC_SOURCE,
) -> pd.DataFrame:
    """Map Tushare daily_basic rows into the project daily_size contract.

    Tushare field names and units stay in the ingestion layer. Research code
    should consume only the vendor-neutral columns returned here.
    """

    if frame is None or frame.empty:
        return empty_daily_size_frame()

    raw = frame.copy()
    for column in ["ts_code", "trade_date", *_TUSHARE_TO_DAILY_SIZE.keys()]:
        if column not in raw.columns:
            raw[column] = pd.NA

    output = pd.DataFrame()
    output["source_trade_date"] = raw["trade_date"].map(_normalize_yyyymmdd)
    output["date"] = pd.to_datetime(output["source_trade_date"], format="%Y%m%d", errors="coerce")
    output["code"] = raw["ts_code"].map(normalize_project_symbol)
    for src_col, dst_col in _TUSHARE_TO_DAILY_SIZE.items():
        output[dst_col] = pd.to_numeric(raw[src_col], errors="coerce")
    output["market_cap_unit"] = TUSHARE_MARKET_CAP_UNIT
    output["share_unit"] = TUSHARE_SHARE_UNIT
    output["source"] = source
    return standardize_daily_size_frame(output)


def standardize_daily_size_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Return a sorted DataFrame with the exact v2 daily_size columns."""

    if frame is None or frame.empty:
        return empty_daily_size_frame()

    output = frame.copy()
    for column in DAILY_SIZE_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output["code"] = output["code"].map(normalize_project_symbol)
    for column in _DAILY_SIZE_NUMERIC_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["source_trade_date"] = output["source_trade_date"].map(_normalize_yyyymmdd)
    output = output.loc[:, DAILY_SIZE_COLUMNS].drop_duplicates(["date", "code"], keep="last")
    return output.sort_values(["date", "code"]).reset_index(drop=True)


def empty_daily_size_frame() -> pd.DataFrame:
    """Return the fixed-schema empty daily_size table."""

    return pd.DataFrame(columns=DAILY_SIZE_COLUMNS)


def write_daily_size_parquet(
    frame: pd.DataFrame,
    snapshot_root: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a daily_size.parquet table under a snapshot root."""

    root = Path(snapshot_root)
    path = root / "daily_size.parquet"
    if path.exists() and not overwrite:
        raise FileExistsError(f"daily_size already exists: {path}")
    root.mkdir(parents=True, exist_ok=True)
    standardize_daily_size_frame(frame).to_parquet(path, index=False)
    return path


def resolve_tushare_token(token: str | None = None, *, env: Mapping[str, str] | None = None) -> str:
    """Resolve an explicit or environment Tushare token."""

    if token:
        return token
    env_map = os.environ if env is None else env
    return str(env_map.get("TUSHARE_TOKEN") or env_map.get("TS_TOKEN") or "")


def normalize_trade_dates(trade_dates: str | Sequence[Any] | pd.Series) -> list[str]:
    """Normalize trade date inputs into YYYYMMDD strings."""

    if isinstance(trade_dates, str):
        raw_dates: Sequence[Any] = [item.strip() for item in trade_dates.split(",")]
    else:
        raw_dates = list(trade_dates)
    output: list[str] = []
    for item in raw_dates:
        if item is None or str(item).strip() == "":
            continue
        digits = _normalize_yyyymmdd(item)
        if len(digits) != 8 or not digits.isdigit():
            raise ValueError(f"Invalid trade date: {item!r}")
        output.append(digits)
    return sorted(set(output))


def daily_size_cache_path(output_root: str | Path, year: int, *, symbols: Sequence[str] | None = None) -> Path:
    root = Path(output_root) / "cache" / "daily_size"
    scope = _daily_size_sample_scope(year=year, symbols=symbols)
    if scope:
        return root / "samples" / scope / f"year={year}.parquet"
    return root / f"year={year}.parquet"


def daily_size_meta_path(output_root: str | Path, year: int, *, symbols: Sequence[str] | None = None) -> Path:
    root = Path(output_root) / "cache" / "cache_meta" / "daily_size"
    scope = _daily_size_sample_scope(year=year, symbols=symbols)
    if scope:
        return root / "samples" / scope / f"year={year}.json"
    return root / f"year={year}.json"


def daily_size_progress_path(output_root: str | Path, year: int, *, symbols: Sequence[str] | None = None) -> Path:
    root = Path(output_root) / "cache" / "progress" / "daily_size"
    scope = _daily_size_sample_scope(year=year, symbols=symbols)
    if scope:
        return root / "samples" / scope / f"year={year}.json"
    return root / f"year={year}.json"


def daily_size_parts_dir(output_root: str | Path, year: int, *, symbols: Sequence[str] | None = None) -> Path:
    root = Path(output_root) / "cache" / "daily_size" / "parts"
    scope = _daily_size_sample_scope(year=year, symbols=symbols)
    if scope:
        return root / "samples" / scope / f"year={year}"
    return root / f"year={year}"


def daily_size_part_path(output_root: str | Path, year: int, trade_date: str, *, symbols: Sequence[str] | None = None) -> Path:
    return daily_size_parts_dir(output_root, year, symbols=symbols) / f"trade_date={trade_date}.parquet"


def load_cached_daily_size(
    output_root: str | Path,
    years: Sequence[int],
    start_date: str,
    end_date: str,
    *,
    symbols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load yearly cached daily_size rows and filter by date/symbol."""

    parts = [pd.read_parquet(daily_size_cache_path(output_root, int(year))) for year in years if daily_size_cache_path(output_root, int(year)).exists()]
    frame = standardize_daily_size_frame(pd.concat(parts, ignore_index=True)) if parts else empty_daily_size_frame()
    return _filter_daily_size(frame, start_date=start_date, end_date=end_date, symbols=symbols)


def fetch_tushare_daily_size_cache(
    *,
    output_root: str | Path,
    year: int,
    trade_dates: str | Sequence[Any] | pd.Series,
    start_date: str,
    end_date: str,
    symbols: Sequence[str] | None = None,
    token: str | None = None,
    force_refresh: bool = False,
    request_interval_seconds: float = 0.0,
    tushare_module: Any | None = None,
) -> dict[str, Any]:
    """Fetch Tushare daily_basic rows by trade date into an annual daily_size cache.

    The function is deliberately vendor-facing and should not be used by research
    code. Research code reads the standardized snapshot table through dataset_v2.
    """

    output_root = Path(output_root)
    year = int(year)
    trade_date_list = normalize_trade_dates(trade_dates)
    requested_symbols = sorted({normalize_project_symbol(symbol) for symbol in symbols or [] if normalize_project_symbol(symbol)})
    path = daily_size_cache_path(output_root, year, symbols=requested_symbols)
    meta_path = daily_size_meta_path(output_root, year, symbols=requested_symbols)
    meta = _read_json_or_empty(meta_path)
    compatible = _is_daily_size_cache_compatible(
        meta,
        year=year,
        start_date=start_date,
        end_date=end_date,
        symbols=requested_symbols,
    )
    if path.exists() and compatible and not force_refresh:
        cached = _filter_daily_size(pd.read_parquet(path), start_date=start_date, end_date=end_date, symbols=requested_symbols)
        return {
            "command": "fetch-size",
            "source": TUSHARE_DAILY_BASIC_SOURCE,
            "status": "cache_hit",
            "year": year,
            "rows": int(len(cached)),
            "date_count": int(cached["date"].nunique()) if not cached.empty else 0,
            "code_count": int(cached["code"].nunique()) if not cached.empty else 0,
            "failure_count": 0,
            "failures": [],
            "cache_path": str(path),
            "cache_hit": 1,
            "cache_miss": 0,
        }

    module, import_error = _load_tushare_module(tushare_module)
    resolved_token = resolve_tushare_token(token)
    if module is None:
        return _skipped_size_summary(
            year=year,
            skip_reason="package_missing",
            message=import_error,
            path=path,
        )
    if not resolved_token:
        return _skipped_size_summary(
            year=year,
            skip_reason="auth_missing",
            message="Set TUSHARE_TOKEN or TS_TOKEN, or pass --size-token, before fetching Tushare daily_size.",
            path=path,
        )

    pro = _tushare_pro(module, resolved_token)
    progress_path = daily_size_progress_path(output_root, year, symbols=requested_symbols)
    progress = _new_daily_size_progress(year, trade_date_list, requested_symbols) if force_refresh else _read_json_or_empty(progress_path)
    if not _is_daily_size_progress_compatible(progress, year=year, trade_dates=trade_date_list, symbols=requested_symbols):
        progress = _new_daily_size_progress(year, trade_date_list, requested_symbols)
    completed = set(progress.get("completed_trade_dates", []))
    failures: list[dict[str, Any]] = list(progress.get("failures", [])) if not force_refresh else []
    touched_parts: list[pd.DataFrame] = []
    fields = ",".join(TUSHARE_DAILY_BASIC_SIZE_FIELDS)
    for trade_date in trade_date_list:
        part_path = daily_size_part_path(output_root, year, trade_date, symbols=requested_symbols)
        if trade_date in completed and part_path.exists() and not force_refresh:
            continue
        progress["active_trade_date"] = trade_date
        _write_json(progress_path, progress)
        try:
            raw = pro.daily_basic(trade_date=trade_date, fields=fields)
        except Exception as exc:  # noqa: BLE001
            failure = {"trade_date": trade_date, "error_type": type(exc).__name__, "message": str(exc)}
            failures.append(failure)
            progress["failures"] = failures
            progress["active_trade_date"] = ""
            _write_json(progress_path, progress)
            continue
        if raw is None or raw.empty:
            failure = {"trade_date": trade_date, "error_type": "empty_result", "message": "daily_basic returned no rows for trade_date."}
            failures.append(failure)
            progress["failures"] = failures
            progress["active_trade_date"] = ""
            _write_json(progress_path, progress)
            continue
        raw = raw.copy()
        if "trade_date" not in raw.columns:
            raw["trade_date"] = trade_date
        frame = standardize_tushare_daily_basic_size(raw)
        frame = _filter_daily_size(frame, start_date=start_date, end_date=end_date, symbols=requested_symbols)
        _write_parquet(part_path, frame)
        touched_parts.append(frame)
        progress["completed_trade_dates"] = sorted(set(progress.get("completed_trade_dates", [])) | {trade_date})
        progress["failures"] = failures
        progress["active_trade_date"] = ""
        _write_json(progress_path, progress)
        if request_interval_seconds > 0:
            time.sleep(float(request_interval_seconds))

    part_frames = [
        pd.read_parquet(part)
        for part in sorted(daily_size_parts_dir(output_root, year, symbols=requested_symbols).glob("trade_date=*.parquet"))
        if part.exists()
    ]
    output = standardize_daily_size_frame(pd.concat([*part_frames, *touched_parts], ignore_index=True)) if (part_frames or touched_parts) else empty_daily_size_frame()
    output = _filter_daily_size(output, start_date=start_date, end_date=end_date, symbols=requested_symbols)
    _write_parquet(path, output)
    status = "passed" if not failures else ("partial" if not output.empty else "failed")
    _write_json(
        meta_path,
        {
            "source": TUSHARE_DAILY_BASIC_SOURCE,
            "year": year,
            "start_date": start_date,
            "end_date": end_date,
            "symbols": requested_symbols,
            "complete": True,
            "row_count": int(len(output)),
            "date_count": int(output["date"].nunique()) if not output.empty else 0,
            "code_count": int(output["code"].nunique()) if not output.empty else 0,
            "failure_count": int(len(failures)),
            "status": status,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return {
        "command": "fetch-size",
        "source": TUSHARE_DAILY_BASIC_SOURCE,
        "status": status,
        "year": year,
        "rows": int(len(output)),
        "date_count": int(output["date"].nunique()) if not output.empty else 0,
        "code_count": int(output["code"].nunique()) if not output.empty else 0,
        "failure_count": int(len(failures)),
        "failures": failures,
        "cache_path": str(path),
        "cache_hit": 0,
        "cache_miss": 1,
    }


def normalize_project_symbol(value: Any) -> str:
    """Normalize vendor symbols into project format, e.g. 600000.SH."""

    text = str(value).strip().upper()
    if not text or text in {"<NA>", "NAN", "NONE"}:
        return ""
    if text.startswith("SH.") or text.startswith("SZ."):
        market, code = text.split(".", 1)
        return f"{code}.{market}"
    if "." in text:
        code, market = text.split(".", 1)
        if market in {"SH", "SZ"}:
            return f"{code}.{market}"
    return text


def _daily_size_sample_scope(*, year: int, symbols: Sequence[str] | None) -> str:
    normalized = sorted({normalize_project_symbol(symbol) for symbol in symbols or [] if normalize_project_symbol(symbol)})
    if not normalized:
        return ""
    payload = {"year": int(year), "symbols": normalized}
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"sample={digest}"


def _filter_daily_size(
    frame: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    symbols: Sequence[str] | None = None,
) -> pd.DataFrame:
    output = standardize_daily_size_frame(frame)
    if output.empty:
        return output
    dates = pd.to_datetime(output["date"])
    output = output.loc[(dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))]
    if symbols:
        output = output.loc[output["code"].isin(set(symbols))]
    return output.reset_index(drop=True)


def _load_tushare_module(module: Any | None) -> tuple[Any | None, str]:
    if module is not None:
        return module, ""
    try:
        return importlib.import_module("tushare"), ""
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _tushare_pro(module: Any, token: str) -> Any:
    if hasattr(module, "set_token"):
        module.set_token(token)
    if callable(getattr(module, "pro_api", None)):
        try:
            return module.pro_api(token)
        except TypeError:
            return module.pro_api()
    return module.pro_api


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


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _new_daily_size_progress(year: int, trade_dates: Sequence[str], symbols: Sequence[str] | None = None) -> dict[str, Any]:
    return {
        "year": int(year),
        "trade_dates": list(trade_dates),
        "symbols": list(symbols or []),
        "completed_trade_dates": [],
        "active_trade_date": "",
        "failures": [],
    }


def _is_daily_size_progress_compatible(
    progress: dict[str, Any],
    *,
    year: int,
    trade_dates: Sequence[str],
    symbols: Sequence[str],
) -> bool:
    if not progress or int(progress.get("year", -1)) != int(year):
        return False
    if not set(trade_dates).issubset(set(progress.get("trade_dates", []))):
        return False
    cached_symbols = set(str(item) for item in progress.get("symbols", []))
    requested_symbols = set(symbols)
    if not requested_symbols:
        return not cached_symbols
    return not cached_symbols or requested_symbols.issubset(cached_symbols)


def _is_daily_size_cache_compatible(
    meta: dict[str, Any],
    *,
    year: int,
    start_date: str,
    end_date: str,
    symbols: Sequence[str],
) -> bool:
    if not meta or not meta.get("complete"):
        return False
    if str(meta.get("source", "")) != TUSHARE_DAILY_BASIC_SOURCE:
        return False
    if int(meta.get("year", -1)) != int(year):
        return False
    if pd.Timestamp(meta.get("start_date", "1900-01-01")) > pd.Timestamp(start_date):
        return False
    if pd.Timestamp(meta.get("end_date", "1900-01-01")) < pd.Timestamp(end_date):
        return False
    cached_symbols = set(str(item) for item in meta.get("symbols", []))
    requested_symbols = set(symbols)
    if not requested_symbols:
        return not cached_symbols
    return not cached_symbols or requested_symbols.issubset(cached_symbols)


def _skipped_size_summary(*, year: int, skip_reason: str, message: str, path: Path) -> dict[str, Any]:
    failure = {"trade_date": "", "error_type": skip_reason, "message": message}
    return {
        "command": "fetch-size",
        "source": TUSHARE_DAILY_BASIC_SOURCE,
        "status": "skipped",
        "skip_reason": skip_reason,
        "year": int(year),
        "rows": 0,
        "date_count": 0,
        "code_count": 0,
        "failure_count": 1,
        "failures": [failure],
        "cache_path": str(path),
        "cache_hit": 0,
        "cache_miss": 0,
    }


def _normalize_yyyymmdd(value: Any) -> str:
    if pd.isna(value):
        return ""
    digits = str(value).strip().replace("-", "")
    if len(digits) == 8 and digits.isdigit():
        return digits
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%Y%m%d")
