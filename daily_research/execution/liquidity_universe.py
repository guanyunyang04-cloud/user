from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from daily_research.baseline.advanced_ml_runtime import HistoryWindow, load_raw_data_with_cache
from daily_research.baseline.data_provider import get_latest_completed_trading_date, load_universe_from_tq


DEFAULT_POOL_SIZES: tuple[int, ...] = (300, 500, 800)
DEFAULT_SIGNAL_LOOKBACK_DAYS = 80
DEFAULT_POOL_SIZE = 500
DEFAULT_POOL_NAMES: tuple[str, ...] = tuple(f"liquid{size}" for size in DEFAULT_POOL_SIZES)


@dataclass(frozen=True)
class LiquidityPoolArtifacts:
    signal_date: str
    universe_dir: Path
    latest_files: dict[int, Path]
    snapshot_files: dict[int, Path]
    ranking_csv: Path
    summary_csv: Path
    cache_meta: dict[str, Any]


@dataclass(frozen=True)
class RollingLiquidityPoolArtifacts:
    pool_name: str
    pool_size: int
    signal_start_date: str
    signal_end_date: str
    rebalance_every_days: int
    adv_window: int
    membership_frame: pd.DataFrame
    schedule_df: pd.DataFrame
    summary_df: pd.DataFrame


def get_universe_dir() -> Path:
    universe_dir = Path(__file__).resolve().parent / "universe"
    universe_dir.mkdir(parents=True, exist_ok=True)
    (universe_dir / "history").mkdir(parents=True, exist_ok=True)
    return universe_dir


def get_default_pool_file(pool_size: int = DEFAULT_POOL_SIZE) -> Path:
    return get_universe_dir() / f"liquid{int(pool_size)}_latest.txt"


def get_liquidity_pool_summary_file() -> Path:
    return get_universe_dir() / "liquidity_pool_summary_latest.csv"


def get_latest_pool_snapshot(
    pool_size: int = DEFAULT_POOL_SIZE,
) -> dict[str, Any]:
    summary_csv = get_liquidity_pool_summary_file()
    if not summary_csv.exists():
        return {}
    try:
        summary_df = pd.read_csv(summary_csv)
    except Exception:
        return {}
    if summary_df.empty or "pool_name" not in summary_df.columns:
        return {}
    pool_name = f"liquid{int(pool_size)}"
    working = summary_df.copy()
    working["pool_name"] = working["pool_name"].astype(str).str.strip().str.lower()
    matched = working.loc[working["pool_name"].eq(pool_name)]
    if matched.empty:
        return {}
    row = matched.iloc[-1].to_dict()
    payload: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            payload[str(key)] = ""
        else:
            payload[str(key)] = value
    return payload


def get_named_pool_file(pool_name: str) -> Path:
    name = str(pool_name or "").strip().lower()
    if not name:
        raise ValueError("pool_name is required.")
    if name.startswith("liquid") and name[6:].isdigit():
        return get_universe_dir() / f"{name}_latest.txt"
    if name.isdigit():
        return get_default_pool_file(int(name))
    raise ValueError(f"Unsupported liquidity pool name: {pool_name}")


def parse_pool_name(pool_name: str | int) -> tuple[str, int]:
    if isinstance(pool_name, int):
        size = int(pool_name)
        return f"liquid{size}", size
    name = str(pool_name or "").strip().lower()
    if not name:
        raise ValueError("pool_name is required.")
    if name.startswith("liquid") and name[6:].isdigit():
        return name, int(name[6:])
    if name.isdigit():
        size = int(name)
        return f"liquid{size}", size
    raise ValueError(f"Unsupported liquidity pool name: {pool_name}")


def parse_pool_sizes(raw: str | None, fallback: Iterable[int] = DEFAULT_POOL_SIZES) -> tuple[int, ...]:
    if not raw:
        return tuple(int(size) for size in fallback)
    values = tuple(sorted({int(item.strip()) for item in str(raw).split(",") if item.strip()}))
    return values or tuple(int(size) for size in fallback)


def _summarize_membership_frame(
    membership_frame: pd.DataFrame,
    schedule_df: pd.DataFrame,
) -> pd.DataFrame:
    member_counts = membership_frame.sum(axis=1).rename("member_count")
    unique_rebalance_dates = pd.to_datetime(schedule_df["rebalance_date"]) if not schedule_df.empty else pd.DatetimeIndex([])
    rebalance_flag = membership_frame.index.isin(unique_rebalance_dates)
    summary = pd.DataFrame(
        {
            "signal_date": membership_frame.index.strftime("%Y-%m-%d"),
            "member_count": member_counts.to_numpy(dtype=int),
            "is_rebalance_date": rebalance_flag.astype(bool),
        }
    )
    return summary


def build_rolling_liquidity_membership(
    *,
    close_frame: pd.DataFrame,
    amount_frame: pd.DataFrame,
    pool_name: str | int,
    signal_start_date: str | None = None,
    signal_end_date: str | None = None,
    rebalance_every_days: int = 21,
    adv_window: int = 20,
    min_price: float = 2.0,
    max_price: float = 300.0,
) -> RollingLiquidityPoolArtifacts:
    resolved_pool_name, pool_size = parse_pool_name(pool_name)
    if int(rebalance_every_days) <= 0:
        raise ValueError("rebalance_every_days must be positive.")
    if int(adv_window) <= 0:
        raise ValueError("adv_window must be positive.")

    close = close_frame.astype(float).copy()
    amount = amount_frame.astype(float).reindex_like(close)
    close.index = pd.to_datetime(close.index)
    amount.index = pd.to_datetime(amount.index)
    close = close.sort_index()
    amount = amount.sort_index()

    signal_start = pd.Timestamp(signal_start_date) if str(signal_start_date or "").strip() else close.index.min()
    signal_end = pd.Timestamp(signal_end_date) if str(signal_end_date or "").strip() else close.index.max()
    signal_dates = close.index[(close.index >= signal_start) & (close.index <= signal_end)]
    if len(signal_dates) == 0:
        raise RuntimeError("No signal dates available for rolling liquidity pool.")

    adv = amount.rolling(int(adv_window)).mean().reindex(index=close.index, columns=close.columns)
    membership_frame = pd.DataFrame(False, index=signal_dates, columns=close.columns, dtype=bool)
    schedule_rows: list[dict[str, Any]] = []

    rebalance_dates = signal_dates[:: int(rebalance_every_days)]
    previous_members: set[str] = set()

    for idx, rebalance_dt in enumerate(rebalance_dates):
        adv_row = adv.loc[rebalance_dt]
        close_row = close.loc[rebalance_dt]
        valid_mask = adv_row.notna() & (adv_row > 0) & close_row.notna()
        if float(min_price) > 0:
            valid_mask &= close_row >= float(min_price)
        if float(max_price) > 0:
            valid_mask &= close_row <= float(max_price)

        ranked = pd.DataFrame(
            {
                "stock": adv_row.index,
                "adv": adv_row.values,
                "close": close_row.reindex(adv_row.index).values,
            }
        )
        ranked = ranked.loc[valid_mask.reindex(ranked["stock"]).fillna(False).to_numpy()].copy()
        ranked = ranked.sort_values(["adv", "close", "stock"], ascending=[False, False, True]).reset_index(drop=True)
        members = ranked["stock"].head(int(pool_size)).tolist()
        member_set = set(members)

        next_rebalance_dt = rebalance_dates[idx + 1] if idx + 1 < len(rebalance_dates) else None
        active_mask = signal_dates >= rebalance_dt
        if next_rebalance_dt is not None:
            active_mask &= signal_dates < next_rebalance_dt
        active_dates = signal_dates[active_mask]
        if len(active_dates) > 0 and members:
            membership_frame.loc[active_dates, members] = True

        turnover = np.nan
        if previous_members:
            turnover = 1.0 - (len(previous_members & member_set) / max(len(previous_members | member_set), 1))
        previous_members = member_set

        subset = ranked.head(int(pool_size))
        schedule_rows.append(
            {
                "rebalance_date": pd.Timestamp(rebalance_dt).strftime("%Y-%m-%d"),
                "effective_from": pd.Timestamp(active_dates.min()).strftime("%Y-%m-%d") if len(active_dates) else pd.Timestamp(rebalance_dt).strftime("%Y-%m-%d"),
                "effective_to": pd.Timestamp(active_dates.max()).strftime("%Y-%m-%d") if len(active_dates) else pd.Timestamp(rebalance_dt).strftime("%Y-%m-%d"),
                "pool_name": resolved_pool_name,
                "target_pool_size": int(pool_size),
                "actual_pool_size": int(len(members)),
                "pool_turnover": float(turnover) if np.isfinite(turnover) else np.nan,
                "min_adv": float(subset["adv"].min()) if not subset.empty else np.nan,
                "median_adv": float(subset["adv"].median()) if not subset.empty else np.nan,
                "max_adv": float(subset["adv"].max()) if not subset.empty else np.nan,
                "min_close": float(subset["close"].min()) if not subset.empty else np.nan,
                "max_close": float(subset["close"].max()) if not subset.empty else np.nan,
            }
        )

    schedule_df = pd.DataFrame(schedule_rows)
    summary_df = _summarize_membership_frame(membership_frame, schedule_df)
    return RollingLiquidityPoolArtifacts(
        pool_name=resolved_pool_name,
        pool_size=int(pool_size),
        signal_start_date=pd.Timestamp(signal_dates.min()).strftime("%Y-%m-%d"),
        signal_end_date=pd.Timestamp(signal_dates.max()).strftime("%Y-%m-%d"),
        rebalance_every_days=int(rebalance_every_days),
        adv_window=int(adv_window),
        membership_frame=membership_frame,
        schedule_df=schedule_df,
        summary_df=summary_df,
    )


def _build_history_window(start_date: str, signal_date: str, lookback_days: int) -> HistoryWindow:
    effective_start = max(pd.Timestamp(start_date), pd.Timestamp(signal_date) - pd.offsets.BDay(int(lookback_days) + 10))
    return HistoryWindow(
        mode="liquidity_pool",
        requested_start_date=str(start_date),
        effective_start_date=effective_start.strftime("%Y%m%d"),
        end_date=pd.Timestamp(signal_date).strftime("%Y%m%d"),
        required_trading_days=int(lookback_days),
    )


def _format_stock_file(stocks: list[str]) -> str:
    return "\n".join(stocks) + ("\n" if stocks else "")


def build_liquidity_rankings(
    *,
    start_date: str = "20240101",
    signal_date: str | None = None,
    universe_scope: str = "all_a",
    lookback_days: int = DEFAULT_SIGNAL_LOOKBACK_DAYS,
    min_price: float = 2.0,
    max_price: float = 300.0,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    signal_date_str = signal_date or get_latest_completed_trading_date()
    universe = load_universe_from_tq(universe_scope)
    history_window = _build_history_window(start_date, signal_date_str, lookback_days)
    raw_df_dict, cache_meta = load_raw_data_with_cache(
        data_source="tq",
        csv_folder=None,
        universe=universe,
        benchmark="",
        history_window=history_window,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        progress_desc="Load liquidity-pool market data",
        progress_position=1,
    )

    close_frame = raw_df_dict["Close"].copy()
    amount_frame = raw_df_dict["Amount"].copy()
    latest_date = close_frame.dropna(how="all").index.max()
    if pd.isna(latest_date):
        raise RuntimeError("No valid latest trading date found while building liquidity pool.")

    adv20 = amount_frame.rolling(20).mean().loc[latest_date]
    last_close = close_frame.loc[latest_date]
    valid_mask = adv20.notna() & (adv20 > 0) & last_close.notna()
    if min_price > 0:
        valid_mask &= last_close >= float(min_price)
    if max_price > 0:
        valid_mask &= last_close <= float(max_price)

    ranked = pd.DataFrame(
        {
            "stock": adv20.index,
            "adv20": adv20.values,
            "close": last_close.reindex(adv20.index).values,
        }
    )
    ranked = ranked.loc[valid_mask.reindex(ranked["stock"]).fillna(False).to_numpy()].copy()
    ranked = ranked.sort_values(["adv20", "close", "stock"], ascending=[False, False, True]).reset_index(drop=True)
    ranked["signal_date"] = latest_date.strftime("%Y-%m-%d")
    ranked["rank"] = range(1, len(ranked) + 1)
    ranked["adv20_pct_rank"] = ranked["adv20"].rank(method="first", pct=True, ascending=False)
    return ranked, latest_date.strftime("%Y-%m-%d"), cache_meta


def write_liquidity_pool_files(
    *,
    ranked: pd.DataFrame,
    signal_date: str,
    pool_sizes: Iterable[int] = DEFAULT_POOL_SIZES,
    universe_dir: Path | None = None,
    cache_meta: dict[str, Any] | None = None,
) -> LiquidityPoolArtifacts:
    universe_dir = universe_dir or get_universe_dir()
    history_dir = universe_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    signal_stamp = pd.Timestamp(signal_date).strftime("%Y%m%d")

    latest_files: dict[int, Path] = {}
    snapshot_files: dict[int, Path] = {}
    summary_rows: list[dict[str, Any]] = []

    for size in tuple(sorted({int(s) for s in pool_sizes if int(s) > 0})):
        stocks = ranked["stock"].head(size).tolist()
        latest_path = universe_dir / f"liquid{size}_latest.txt"
        snapshot_path = history_dir / f"liquid{size}_{signal_stamp}.txt"
        content = _format_stock_file(stocks)
        latest_path.write_text(content, encoding="utf-8")
        snapshot_path.write_text(content, encoding="utf-8")
        latest_files[size] = latest_path
        snapshot_files[size] = snapshot_path
        subset = ranked.head(size)
        summary_rows.append(
            {
                "signal_date": signal_date,
                "pool_name": f"liquid{size}",
                "pool_size": int(len(stocks)),
                "min_adv20": float(subset["adv20"].min()) if not subset.empty else 0.0,
                "median_adv20": float(subset["adv20"].median()) if not subset.empty else 0.0,
                "max_adv20": float(subset["adv20"].max()) if not subset.empty else 0.0,
                "min_close": float(subset["close"].min()) if not subset.empty else 0.0,
                "max_close": float(subset["close"].max()) if not subset.empty else 0.0,
                "latest_file": str(latest_path),
                "snapshot_file": str(snapshot_path),
            }
        )

    ranking_csv = universe_dir / "liquidity_rank_latest.csv"
    summary_csv = universe_dir / "liquidity_pool_summary_latest.csv"
    ranked.to_csv(ranking_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    return LiquidityPoolArtifacts(
        signal_date=signal_date,
        universe_dir=universe_dir,
        latest_files=latest_files,
        snapshot_files=snapshot_files,
        ranking_csv=ranking_csv,
        summary_csv=summary_csv,
        cache_meta=cache_meta or {},
    )


def update_liquidity_pool_files(
    *,
    pool_sizes: Iterable[int] = DEFAULT_POOL_SIZES,
    start_date: str = "20240101",
    signal_date: str | None = None,
    universe_scope: str = "all_a",
    lookback_days: int = DEFAULT_SIGNAL_LOOKBACK_DAYS,
    min_price: float = 2.0,
    max_price: float = 300.0,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> LiquidityPoolArtifacts:
    ranked, resolved_signal_date, cache_meta = build_liquidity_rankings(
        start_date=start_date,
        signal_date=signal_date,
        universe_scope=universe_scope,
        lookback_days=lookback_days,
        min_price=min_price,
        max_price=max_price,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
    )
    return write_liquidity_pool_files(
        ranked=ranked,
        signal_date=resolved_signal_date,
        pool_sizes=pool_sizes,
        cache_meta=cache_meta,
    )


def ensure_default_pool_file(
    *,
    pool_size: int = DEFAULT_POOL_SIZE,
    start_date: str = "20240101",
    signal_date: str | None = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    auto_refresh_stale: bool = True,
) -> Path:
    default_file = get_default_pool_file(pool_size)
    expected_signal_date = pd.Timestamp(signal_date or get_latest_completed_trading_date()).strftime("%Y-%m-%d")
    snapshot = get_latest_pool_snapshot(pool_size)
    current_signal_date = str(snapshot.get("signal_date", "")).strip()
    latest_file_text = str(snapshot.get("latest_file", "")).strip()
    latest_file_matches = False
    if latest_file_text:
        try:
            latest_file_matches = Path(latest_file_text).resolve() == default_file.resolve()
        except Exception:
            latest_file_matches = False
    is_fresh = (
        default_file.exists()
        and not refresh_cache
        and current_signal_date == expected_signal_date
        and latest_file_matches
    )
    if is_fresh:
        return default_file
    if not auto_refresh_stale:
        if not default_file.exists():
            raise FileNotFoundError(
                f"Default liquid{int(pool_size)} universe file not found: {default_file}. "
                "Please run daily_research/execution/update_liquid_pool.py after close first."
            )
        raise RuntimeError(
            f"Default liquid{int(pool_size)} universe file is stale: "
            f"current_signal_date={current_signal_date or 'missing'} expected_signal_date={expected_signal_date}. "
            "Please run daily_research/execution/update_liquid_pool.py after close first."
        )
    artifacts = update_liquidity_pool_files(
        pool_sizes=DEFAULT_POOL_SIZES,
        start_date=start_date,
        signal_date=expected_signal_date,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
    )
    refreshed_file = artifacts.latest_files[int(pool_size)]
    refreshed_signal_date = pd.Timestamp(artifacts.signal_date).strftime("%Y-%m-%d")
    if refreshed_signal_date != expected_signal_date:
        raise RuntimeError(
            f"Auto-refresh produced an unexpected liquid{int(pool_size)} signal date: "
            f"refreshed_signal_date={refreshed_signal_date} expected_signal_date={expected_signal_date}."
        )
    if not refreshed_file.exists():
        raise FileNotFoundError(
            f"Auto-refresh completed but default liquid{int(pool_size)} universe file is still missing: {refreshed_file}"
        )
    return refreshed_file
