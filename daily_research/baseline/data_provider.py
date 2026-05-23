from __future__ import annotations

import sys
import io
import os
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from daily_research.progress import create_progress


MAINLAND_MAIN_BOARD_PREFIXES = {
    "SH": ("600", "601", "603", "605"),
    "SZ": ("000", "001", "002", "003"),
}
ST_NAME_PREFIXES = ("ST", "*ST", "SST", "S*ST")

STYLE_SECTOR_CODES = {
    "financial": [
        "881386.SH",
        "881389.SH",
        "881394.SH",
        "881395.SH",
        "881396.SH",
    ],
    "high_dividend": [
        "880526.SH",
    ],
}

MARKET_DATA_FIELDS = ("Open", "High", "Low", "Close", "Volume", "Amount")
TQ_FETCH_BATCH_SIZE = 64
_TQ_SESSION_COUNTER = 0
TQ_FETCH_EMPTY_BATCH_RETRY_COUNT = 2


def _run_tq_quietly(func, *args, **kwargs):
    buffer = io.StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        return func(*args, **kwargs)


def _tq_session_path(attempt: int = 0) -> str:
    global _TQ_SESSION_COUNTER
    _TQ_SESSION_COUNTER += 1
    session_dir = Path(__file__).resolve().parents[1] / "cache" / "tq_sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    session_path = session_dir / (
        f"data_provider_{os.getpid()}_{int(time.time() * 1000)}_{_TQ_SESSION_COUNTER}_{attempt}.session"
    )
    if not session_path.exists():
        session_path.write_text("daily_research tq session\n", encoding="utf-8")
    return str(session_path)


def _record_tq_init_failure(exc: Exception, *, attempt: int) -> None:
    log_path = Path(__file__).resolve().parents[1] / "cache" / "tq_sessions" / "tq_init_failures.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{pd.Timestamp.now().isoformat()} attempt={attempt} error={type(exc).__name__}: {exc}\n"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def _initialize_tq_client(tq) -> None:
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            session_path = _tq_session_path(attempt)
            _run_tq_quietly(tq.initialize, session_path)
            return
        except Exception as exc:
            last_error = exc
            _record_tq_init_failure(exc, attempt=attempt + 1)
            try:
                _run_tq_quietly(tq.close)
            except Exception:
                pass
            if attempt < 5:
                time.sleep(min(8.0, 0.75 * (attempt + 1)))
    if last_error is not None:
        raise last_error


def _close_tq_client(tq) -> None:
    _run_tq_quietly(tq.close)


def _try_import_tq():
    allow = str(os.environ.get("DAILY_RESEARCH_ALLOW_TDX_FAMILY", "") or "").strip().lower()
    if allow not in {"1", "true", "yes", "on"}:
        return None
    try:
        from tqcenter import tq  # type: ignore
        return tq
    except Exception:
        root = Path(__file__).resolve().parents[2]
        candidate = root / "t0_project"
        if candidate.exists():
            sys.path.append(str(candidate))
            try:
                from tqcenter import tq  # type: ignore
                return tq
            except Exception:
                return None
        return None


def _fetch_tq_market_data_batch(
    tq,
    *,
    stock_list: List[str],
    start_date: str,
    end_date: str,
    count: int,
    dividend_type: str,
    batch_start: int,
    reinitialize_callback=None,
) -> dict:
    for attempt in range(TQ_FETCH_EMPTY_BATCH_RETRY_COUNT + 1):
        if attempt > 0 and reinitialize_callback is not None:
            reinitialize_callback(tq)
        batch_dict = tq.get_market_data(
            field_list=list(MARKET_DATA_FIELDS),
            stock_list=stock_list,
            start_time=start_date,
            end_time=end_date,
            count=count,
            dividend_type=dividend_type,
            period="1d",
        )
        if batch_dict and "Close" in batch_dict:
            return batch_dict
        if attempt < TQ_FETCH_EMPTY_BATCH_RETRY_COUNT:
            time.sleep(0.35 * (attempt + 1))
    if len(stock_list) <= 1:
        stock_text = stock_list[0] if stock_list else "<empty>"
        raise RuntimeError(
            "TDX returned empty data for singleton batch "
            f"stock={stock_text} start_date={start_date} end_date={end_date or '<latest>'} batch_start={batch_start}."
        )
    midpoint = max(1, len(stock_list) // 2)
    left = _fetch_tq_market_data_batch(
        tq,
        stock_list=stock_list[:midpoint],
        start_date=start_date,
        end_date=end_date,
        count=count,
        dividend_type=dividend_type,
        batch_start=batch_start,
        reinitialize_callback=reinitialize_callback,
    )
    right = _fetch_tq_market_data_batch(
        tq,
        stock_list=stock_list[midpoint:],
        start_date=start_date,
        end_date=end_date,
        count=count,
        dividend_type=dividend_type,
        batch_start=batch_start + midpoint,
        reinitialize_callback=reinitialize_callback,
    )
    merged: dict = {}
    for field in MARKET_DATA_FIELDS:
        frames = [payload.get(field) for payload in (left, right) if payload.get(field) is not None]
        if frames:
            merged[field] = pd.concat(frames, axis=1)
    return merged


def get_universe_cache_file(universe_scope: str = "all_a") -> Path:
    safe_scope = str(universe_scope or "all_a").strip().lower().replace("/", "_").replace("\\", "_")
    cache_file = Path(__file__).resolve().parents[1] / "cache" / f"universe_{safe_scope}_tq.csv"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    return cache_file


def load_cached_universe_from_tq(universe_scope: str = "all_a") -> List[str]:
    cache_file = get_universe_cache_file(universe_scope)
    if not cache_file.exists():
        cached_manifest = _load_cached_universe_from_raw_manifest(universe_scope)
        if cached_manifest:
            _save_universe_cache(cached_manifest, universe_scope)
        return cached_manifest
    try:
        cached = pd.read_csv(cache_file, dtype={"stock": str})
    except Exception:
        return _load_cached_universe_from_raw_manifest(universe_scope)
    if cached.empty or "stock" not in cached.columns:
        return _load_cached_universe_from_raw_manifest(universe_scope)
    return _unique_preserve_order([str(item).strip().upper() for item in cached["stock"].dropna().tolist()])


def _load_cached_universe_from_raw_manifest(universe_scope: str = "all_a") -> List[str]:
    if str(universe_scope or "all_a").strip().lower() != "all_a":
        return []
    manifest_dir = Path(__file__).resolve().parents[1] / "cache" / "advanced_ml"
    if not manifest_dir.exists():
        return []
    candidates = sorted(
        manifest_dir.glob("all_a_cached_from_raw_*.txt"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for manifest in candidates:
        try:
            values = [
                line.strip().upper()
                for line in manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except Exception:
            continue
        cached = _unique_preserve_order([value for value in values if _is_a_share_stock(value)])
        if cached:
            return cached
    return []


def _save_universe_cache(universe: List[str], universe_scope: str = "all_a") -> None:
    stocks = _unique_preserve_order(universe)
    if not stocks:
        return
    cache_file = get_universe_cache_file(universe_scope)
    pd.DataFrame({"stock": stocks}).to_csv(cache_file, index=False, encoding="utf-8-sig")


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    if out.index.name is None:
        out.index.name = "Date"
    return out.sort_index()


def align_data_dict(df_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    if not df_dict:
        return df_dict
    index = None
    for _, df in df_dict.items():
        if df is None or df.empty:
            continue
        index = df.index if index is None else index.intersection(df.index)
    if index is None:
        return df_dict
    return {k: _ensure_datetime_index(v.loc[index].copy()) for k, v in df_dict.items()}


def resolve_benchmark_symbol(benchmark: str) -> str:
    benchmark = str(benchmark or "").strip().upper()
    if not benchmark:
        raise ValueError("benchmark cannot be empty.")
    return benchmark


def _normalize_stock_code(code: str) -> str:
    return str(code or "").strip().upper()


def _unique_preserve_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _name_map_lookup(stock_name_map: pd.Series | Dict[str, str] | None) -> Dict[str, str] | None:
    if stock_name_map is None:
        return None
    if isinstance(stock_name_map, pd.Series):
        return {str(idx).upper(): str(value or "") for idx, value in stock_name_map.dropna().items()}
    return {str(key).upper(): str(value or "") for key, value in stock_name_map.items()}


def _looks_like_st_stock(name: str) -> bool:
    normalized = str(name or "").strip().upper().replace(" ", "")
    if not normalized:
        return False
    return normalized.startswith(ST_NAME_PREFIXES)


def _is_a_share_stock(code: str) -> bool:
    if not code or "." not in code:
        return False
    prefix, market = code.split(".", 1)
    market = market.upper()
    if market not in MAINLAND_MAIN_BOARD_PREFIXES:
        return False
    prefix = prefix.upper()
    return any(prefix.startswith(item) for item in MAINLAND_MAIN_BOARD_PREFIXES[market])


def get_stock_name_cache_file(cache_path: str | Path | None = None) -> Path:
    if cache_path:
        cache_file = Path(cache_path)
    else:
        cache_file = Path(__file__).resolve().parents[1] / "cache" / "stock_name_map_tq.csv"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    return cache_file


def load_cached_stock_name_map(cache_path: str | Path | None = None) -> pd.Series:
    cache_file = get_stock_name_cache_file(cache_path)
    if not cache_file.exists():
        return pd.Series(dtype=str, name="name")
    cached = pd.read_csv(cache_file, dtype={"stock": str, "name": str})
    if cached.empty or not {"stock", "name"}.issubset(cached.columns):
        return pd.Series(dtype=str, name="name")
    series = cached.drop_duplicates(subset=["stock"]).set_index("stock")["name"].fillna("").astype(str)
    series.index = series.index.map(lambda item: str(item).upper())
    return series.sort_index()


def _save_stock_name_map_cache(stock_name_map: pd.Series, cache_path: str | Path | None = None) -> None:
    cache_file = get_stock_name_cache_file(cache_path)
    series = stock_name_map.dropna().astype(str)
    if series.empty:
        return
    series.rename_axis("stock").rename("name").reset_index().to_csv(cache_file, index=False, encoding="utf-8-sig")


def find_universe_violations(
    stock_list: List[str],
    stock_name_map: pd.Series | Dict[str, str] | None = None,
) -> Dict[str, str]:
    name_lookup = _name_map_lookup(stock_name_map)
    violations: Dict[str, str] = {}
    for raw in stock_list:
        stock = _normalize_stock_code(raw)
        if not stock:
            continue
        if not _is_a_share_stock(stock):
            violations[stock] = "market_or_board"
            continue
        if name_lookup is not None and _looks_like_st_stock(name_lookup.get(stock, "")):
            violations[stock] = "st"
    return violations


def filter_a_share_universe(
    stock_list: List[str],
    universe_scope: str = "all_a",
    stock_name_map: pd.Series | Dict[str, str] | None = None,
) -> List[str]:
    universe_scope = str(universe_scope or "all_a").lower()
    normalized = [_normalize_stock_code(stock) for stock in stock_list if _normalize_stock_code(stock)]
    if universe_scope != "all_a":
        return _unique_preserve_order(normalized)
    violations = find_universe_violations(normalized, stock_name_map=stock_name_map)
    filtered = [stock for stock in normalized if stock not in violations]
    return _unique_preserve_order(filtered)


def load_universe_from_tq(universe_scope: str = "all_a") -> List[str]:
    tq = _try_import_tq()
    if tq is None:
        cached_universe = load_cached_universe_from_tq(universe_scope)
        if cached_universe:
            return cached_universe
        raise RuntimeError("tqcenter not available. Please ensure t0_project/tqcenter.py exists.")

    try:
        _initialize_tq_client(tq)
        stock_list = tq.get_stock_list()
        if not stock_list:
            raise RuntimeError("TDX returned empty stock list.")
        code_filtered = filter_a_share_universe(list(stock_list), universe_scope=universe_scope)
        stock_name_map = load_cached_stock_name_map()
        missing = [stock for stock in code_filtered if stock not in set(stock_name_map.index)]
        if missing:
            fetched: Dict[str, str] = {}
            for stock in missing:
                try:
                    info = tq.get_stock_info(stock, ["Name"])
                except Exception:
                    continue
                name = str((info or {}).get("Name", "")).strip()
                if name:
                    fetched[stock] = name
            if fetched:
                stock_name_map = (
                    pd.concat([stock_name_map, pd.Series(fetched, name="name")])
                    .groupby(level=0)
                    .last()
                    .sort_index()
                )
                _save_stock_name_map_cache(stock_name_map)
        filtered = filter_a_share_universe(
            list(stock_list),
            universe_scope=universe_scope,
            stock_name_map=stock_name_map if not stock_name_map.empty else None,
        )
        if not filtered:
            raise RuntimeError(f"No valid stocks found for universe_scope={universe_scope}.")
        _save_universe_cache(filtered, universe_scope)
        return filtered
    except Exception:
        cached_universe = load_cached_universe_from_tq(universe_scope)
        if cached_universe:
            return cached_universe
        raise
    finally:
        try:
            _close_tq_client(tq)
        except Exception:
            pass


def load_industry_map_from_tq(
    universe: List[str] | None = None,
    cache_path: str | Path | None = None,
) -> pd.Series:
    cache_file = Path(cache_path) if cache_path else Path(__file__).resolve().parents[1] / "cache" / "industry_map_tq.csv"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    requested = sorted(set(universe or []))
    if cache_file.exists():
        cached = pd.read_csv(cache_file, dtype={"stock": str, "industry": str})
        if not cached.empty and {"stock", "industry"}.issubset(cached.columns):
            cached_series = cached.drop_duplicates(subset=["stock"]).set_index("stock")["industry"]
            if not requested:
                return cached_series.sort_index()
            if set(requested).issubset(set(cached_series.index)):
                return cached_series.reindex(requested).dropna()

    tq = _try_import_tq()
    if tq is None:
        raise RuntimeError("tqcenter not available. Please ensure t0_project/tqcenter.py exists.")

    _initialize_tq_client(tq)
    try:
        sector_codes = [code for code in tq.get_sector_list() if str(code).startswith("881")]
        if not sector_codes:
            raise RuntimeError("No industry-like sector codes found from TQ.")

        stock_to_industry: Dict[str, str] = {}
        for sector_code in sector_codes:
            try:
                sector_name = tq.get_stock_info(sector_code, ["Name"]).get("Name", "")
                members = tq.get_stock_list_in_sector(sector_code)
            except Exception:
                continue
            if not sector_name or not members:
                continue
            for stock in members:
                if requested and stock not in requested:
                    continue
                stock_to_industry.setdefault(stock, sector_name)

        series = pd.Series(stock_to_industry, name="industry").sort_index()
        if not series.empty:
            series.rename_axis("stock").reset_index().to_csv(cache_file, index=False, encoding="utf-8-sig")
        if requested:
            return series.reindex(requested).dropna()
        return series
    finally:
        try:
            _close_tq_client(tq)
        except Exception:
            pass


def load_style_map_from_tq(
    universe: List[str] | None = None,
    cache_path: str | Path | None = None,
    style_sector_codes: Dict[str, List[str]] | None = None,
) -> pd.DataFrame:
    cache_file = Path(cache_path) if cache_path else Path(__file__).resolve().parents[1] / "cache" / "style_map_tq.csv"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    style_sector_codes = style_sector_codes or STYLE_SECTOR_CODES
    requested = sorted(set(universe or []))

    if cache_file.exists():
        cached = pd.read_csv(cache_file, dtype={"stock": str})
        if not cached.empty and "stock" in cached.columns:
            cached = cached.drop_duplicates(subset=["stock"]).set_index("stock")
            cols = [style for style in style_sector_codes if style in cached.columns]
            if cols:
                cached = cached[cols].fillna(False).astype(bool)
                if not requested:
                    return cached.sort_index()
                if set(requested).issubset(set(cached.index)):
                    return cached.reindex(requested).astype("boolean").fillna(False).astype(bool)

    tq = _try_import_tq()
    if tq is None:
        raise RuntimeError("tqcenter not available. Please ensure t0_project/tqcenter.py exists.")

    _initialize_tq_client(tq)
    try:
        rows: Dict[str, Dict[str, bool]] = {}
        for style_name, sector_codes in style_sector_codes.items():
            for sector_code in sector_codes:
                try:
                    members = tq.get_stock_list_in_sector(sector_code)
                except Exception:
                    continue
                for stock in members:
                    if requested and stock not in requested:
                        continue
                    rows.setdefault(stock, {name: False for name in style_sector_codes.keys()})
                    rows[stock][style_name] = True

        style_df = pd.DataFrame.from_dict(rows, orient="index").fillna(False).astype(bool)
        style_df.index.name = "stock"
        if not style_df.empty:
            style_df.reset_index().to_csv(cache_file, index=False, encoding="utf-8-sig")
        if requested:
            return style_df.reindex(requested).astype("boolean").fillna(False).astype(bool)
        return style_df.sort_index()
    finally:
        try:
            _close_tq_client(tq)
        except Exception:
            pass


def _fetch_tq_data(
    stock_list: List[str],
    start_date: str,
    end_date: str = "",
    count: int = 0,
    dividend_type: str = "front",
    progress_desc: str = "Load daily stock bars",
    progress_position: int = 0,
) -> Dict[str, pd.DataFrame]:
    tq = _try_import_tq()
    if tq is None:
        raise RuntimeError("tqcenter not available. Please ensure t0_project/tqcenter.py exists.")

    _initialize_tq_client(tq)
    try:
        chunk_size = max(1, min(TQ_FETCH_BATCH_SIZE, len(stock_list)))
        collected: Dict[str, list[pd.DataFrame]] = {field: [] for field in MARKET_DATA_FIELDS}
        with create_progress(
            total=len(stock_list),
            desc=str(progress_desc or "Load daily stock bars"),
            unit="stock",
            leave=False,
            position=progress_position,
        ) as progress:
            for start in range(0, len(stock_list), chunk_size):
                batch = stock_list[start:start + chunk_size]
                batch_dict = _fetch_tq_market_data_batch(
                    tq,
                    stock_list=batch,
                    start_date=start_date,
                    end_date=end_date,
                    count=count,
                    dividend_type=dividend_type,
                    batch_start=start,
                    reinitialize_callback=_initialize_tq_client,
                )
                for field in MARKET_DATA_FIELDS:
                    frame = batch_dict.get(field)
                    if frame is None or frame.empty:
                        continue
                    collected[field].append(_ensure_datetime_index(frame))
                progress.update(len(batch))

        merged: Dict[str, pd.DataFrame] = {}
        for field, frames in collected.items():
            if not frames:
                continue
            merged_frame = pd.concat(frames, axis=1)
            merged[field] = merged_frame.loc[:, ~merged_frame.columns.duplicated(keep="last")]
        if not merged or "Close" not in merged:
            raise RuntimeError("TDX returned empty data.")
        return align_data_dict({k: _ensure_datetime_index(v) for k, v in merged.items()})
    finally:
        try:
            _close_tq_client(tq)
        except Exception:
            pass


def load_daily_from_tq(
    stock_list: List[str],
    start_date: str,
    end_date: str = "",
    count: int = 0,
    dividend_type: str = "front",
    benchmark: str = "",
    progress_desc: str = "Load daily stock bars",
    progress_position: int = 0,
) -> Dict[str, pd.DataFrame]:
    to_fetch = [s for s in stock_list if s]
    benchmark_symbol = resolve_benchmark_symbol(benchmark) if benchmark else ""
    if benchmark_symbol:
        to_fetch.append(benchmark_symbol)
    to_fetch = sorted(set(to_fetch))
    if not to_fetch:
        raise ValueError("No stocks provided to load_daily_from_tq.")
    return _fetch_tq_data(
        stock_list=to_fetch,
        start_date=start_date,
        end_date=end_date,
        count=count,
        dividend_type=dividend_type,
        progress_desc=progress_desc,
        progress_position=progress_position,
    )


def load_daily_from_csv(
    folder: str,
    *,
    progress_desc: str = "Load CSV market data",
    progress_position: int = 0,
) -> Dict[str, pd.DataFrame]:
    path = Path(folder)
    if not path.exists():
        raise FileNotFoundError(f"CSV folder not found: {folder}")

    fields = {"Open": {}, "High": {}, "Low": {}, "Close": {}, "Volume": {}, "Amount": {}}
    required_fields = tuple(fields.keys())
    csv_files = sorted(path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {folder}")

    with create_progress(
        total=len(csv_files),
        desc=str(progress_desc or "Load CSV market data"),
        unit="file",
        leave=False,
        position=progress_position,
    ) as progress:
        for csv_file in csv_files:
            stock = csv_file.stem
            df = pd.read_csv(csv_file)

            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
            elif "Datetime" in df.columns:
                df["Datetime"] = pd.to_datetime(df["Datetime"])
                df = df.set_index("Datetime")
            else:
                raise ValueError(f"{csv_file.name} missing Date/Datetime column.")

            missing = [field for field in required_fields if field not in df.columns]
            if missing:
                raise ValueError(f"{csv_file.name} missing fields: {missing}")

            df = df.sort_index()
            for field in required_fields:
                fields[field][stock] = df[field].copy()
            progress.update(1)

    df_dict = {k: pd.DataFrame(v) for k, v in fields.items()}
    return align_data_dict({k: _ensure_datetime_index(v) for k, v in df_dict.items()})


def split_benchmark_from_universe(
    df_dict: Dict[str, pd.DataFrame],
    benchmark_symbol: str,
) -> Tuple[Dict[str, pd.DataFrame], pd.Series]:
    benchmark_symbol = resolve_benchmark_symbol(benchmark_symbol)
    if "Close" not in df_dict or benchmark_symbol not in df_dict["Close"].columns:
        raise KeyError(f"Benchmark {benchmark_symbol} not found in Close data.")

    benchmark_close = df_dict["Close"][benchmark_symbol].copy()
    universe_dict: Dict[str, pd.DataFrame] = {}
    for field, df in df_dict.items():
        universe_dict[field] = df.drop(columns=[benchmark_symbol], errors="ignore").copy()
    return universe_dict, benchmark_close


def get_next_trading_date(anchor_date: str | pd.Timestamp, market: str = "SH") -> str | None:
    anchor_ts = pd.Timestamp(anchor_date).normalize()
    tq = _try_import_tq()
    if tq is None:
        return (anchor_ts + pd.offsets.BDay(1)).strftime("%Y-%m-%d")

    try:
        _initialize_tq_client(tq)
        end_ts = anchor_ts + timedelta(days=40)
        dates = tq.get_trading_dates(
            market=str(market).upper(),
            start_time=anchor_ts.strftime("%Y%m%d"),
            end_time=end_ts.strftime("%Y%m%d"),
            count=-1,
        ) or []
        normalized = sorted({pd.Timestamp(item).normalize() for item in dates})
        for dt in normalized:
            if dt > anchor_ts:
                return dt.strftime("%Y-%m-%d")
    except Exception:
        return (anchor_ts + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    finally:
        try:
            _close_tq_client(tq)
        except Exception:
            pass
    return (anchor_ts + pd.offsets.BDay(1)).strftime("%Y-%m-%d")


def get_latest_completed_trading_date(
    reference_ts: pd.Timestamp | None = None,
    market: str = "SH",
    close_time: str = "15:05",
) -> str:
    del market
    from daily_research.data_platform.contracts import latest_completed_business_date

    return latest_completed_business_date(reference_ts=reference_ts, close_time=close_time)
