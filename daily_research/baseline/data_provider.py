from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from daily_research.progress import create_progress


A_SHARE_PREFIXES = (
    "00",
    "30",
    "60",
    "68",
    "43",
    "82",
    "83",
    "87",
    "88",
    "89",
    "92",
)

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


def _try_import_tq():
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
        raise ValueError("benchmark 不能为空。")
    return benchmark


def _is_a_share_stock(code: str) -> bool:
    if not code or "." not in code:
        return False
    prefix, market = code.split(".", 1)
    if market.upper() not in {"SH", "SZ", "BJ"}:
        return False
    return prefix.startswith(A_SHARE_PREFIXES)


def filter_a_share_universe(stock_list: List[str], universe_scope: str = "all_a") -> List[str]:
    universe_scope = str(universe_scope or "all_a").lower()
    if universe_scope != "all_a":
        return [s for s in stock_list if s]
    filtered = [s for s in stock_list if _is_a_share_stock(s)]
    return sorted(set(filtered))


def load_universe_from_tq(universe_scope: str = "all_a") -> List[str]:
    tq = _try_import_tq()
    if tq is None:
        raise RuntimeError("tqcenter not available. Please ensure t0_project/tqcenter.py exists.")

    tq.initialize(__file__)
    try:
        stock_list = tq.get_stock_list()
        if not stock_list:
            raise RuntimeError("TDX returned empty stock list.")
        filtered = filter_a_share_universe(list(stock_list), universe_scope=universe_scope)
        if not filtered:
            raise RuntimeError(f"No valid stocks found for universe_scope={universe_scope}.")
        return filtered
    finally:
        try:
            tq.close()
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

    tq.initialize(__file__)
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
            tq.close()
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

    tq.initialize(__file__)
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
            tq.close()
        except Exception:
            pass


def _fetch_tq_data(
    stock_list: List[str],
    start_date: str,
    end_date: str = "",
    count: int = 0,
    dividend_type: str = "front",
    progress_desc: str = "读取股票日线",
    progress_position: int = 0,
) -> Dict[str, pd.DataFrame]:
    tq = _try_import_tq()
    if tq is None:
        raise RuntimeError("tqcenter not available. Please ensure t0_project/tqcenter.py exists.")

    tq.initialize(__file__)
    try:
        chunk_size = max(1, min(TQ_FETCH_BATCH_SIZE, len(stock_list)))
        collected: Dict[str, list[pd.DataFrame]] = {field: [] for field in MARKET_DATA_FIELDS}
        with create_progress(
            total=len(stock_list),
            desc=str(progress_desc or "读取股票日线"),
            unit="stock",
            leave=False,
            position=progress_position,
        ) as progress:
            for start in range(0, len(stock_list), chunk_size):
                batch = stock_list[start:start + chunk_size]
                batch_dict = tq.get_market_data(
                    field_list=list(MARKET_DATA_FIELDS),
                    stock_list=batch,
                    start_time=start_date,
                    end_time=end_date,
                    count=count,
                    dividend_type=dividend_type,
                    period="1d",
                )
                if not batch_dict or "Close" not in batch_dict:
                    raise RuntimeError(f"TDX returned empty data for batch starting at {start}.")
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
            tq.close()
        except Exception:
            pass


def load_daily_from_tq(
    stock_list: List[str],
    start_date: str,
    end_date: str = "",
    count: int = 0,
    dividend_type: str = "front",
    benchmark: str = "",
    progress_desc: str = "读取股票日线",
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
    progress_desc: str = "读取CSV行情",
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
        desc=str(progress_desc or "读取CSV行情"),
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

    tq.initialize(__file__)
    try:
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
    finally:
        try:
            tq.close()
        except Exception:
            pass
    return (anchor_ts + pd.offsets.BDay(1)).strftime("%Y-%m-%d")


def get_latest_completed_trading_date(
    reference_ts: pd.Timestamp | None = None,
    market: str = "SH",
    close_time: str = "15:05",
) -> str:
    now_ts = pd.Timestamp(reference_ts).tz_localize(None) if reference_ts is not None else pd.Timestamp.now().tz_localize(None)
    close_clock = pd.Timestamp(close_time).time()
    include_today = now_ts.time() >= close_clock
    anchor_ts = now_ts.normalize()

    tq = _try_import_tq()
    if tq is None:
        offset = 0 if include_today else 1
        return (anchor_ts - pd.offsets.BDay(offset)).strftime("%Y-%m-%d")

    tq.initialize(__file__)
    try:
        start_ts = anchor_ts - timedelta(days=40)
        dates = tq.get_trading_dates(
            market=str(market).upper(),
            start_time=start_ts.strftime("%Y%m%d"),
            end_time=anchor_ts.strftime("%Y%m%d"),
            count=-1,
        ) or []
        normalized = sorted({pd.Timestamp(item).normalize() for item in dates})
        if include_today:
            eligible = [dt for dt in normalized if dt <= anchor_ts]
        else:
            eligible = [dt for dt in normalized if dt < anchor_ts]
        if eligible:
            return eligible[-1].strftime("%Y-%m-%d")
    finally:
        try:
            tq.close()
        except Exception:
            pass

    offset = 0 if include_today else 1
    return (anchor_ts - pd.offsets.BDay(offset)).strftime("%Y-%m-%d")
