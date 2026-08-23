"""Structural and cross-source audits for canonical minute imports."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from quantlab.data.core.paths import qdp_paths
from quantlab.data.minute_archive.contracts import (
    AUCTION_TIME,
    BAR_TIME_BY_HHMM,
    CONTINUOUS_TIMES,
    RAW_COLUMNS,
    ArchiveMember,
    MinuteArchiveError,
)
from quantlab.data.qdp_v2.manifest import (
    dataset_manifest_for_id,
    read_active_manifest,
    read_dataset_manifest,
)

PRICE_COLUMNS = ["open", "high", "low", "close"]


def repair_mislabeled_1300_as_1130(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    """Repair a source bug only when 13:00 replaces the missing 11:30 bar."""

    if frame.empty:
        return pd.DataFrame()
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    dates = timestamps.to_numpy(dtype="datetime64[D]").astype(str)
    clock = timestamps.dt.hour * 100 + timestamps.dt.minute
    clocks = pd.DataFrame({"trade_date": dates, "clock": clock})
    flags = clocks.groupby("trade_date")["clock"].agg(
        has_1300=lambda values: bool(values.eq(1300).any()),
        has_1130=lambda values: bool(values.eq(1130).any()),
    )
    repair_dates = set(flags.index[flags["has_1300"] & ~flags["has_1130"]])
    repairable = clock.eq(1300) & pd.Series(dates, index=frame.index).isin(repair_dates)
    if not repairable.any():
        return pd.DataFrame()
    repaired = timestamps.loc[repairable].dt.normalize() + pd.Timedelta(hours=11, minutes=30)
    ledger = pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": dates[repairable.to_numpy()],
            "original_bar_time": "130000000",
            "repaired_bar_time": "113000000",
            "repair_reason": "13:00_present_while_11:30_missing",
        }
    )
    frame.loc[repairable, "timestamp"] = repaired
    return ledger.reset_index(drop=True)


def repair_zero_price_placeholders(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    """Repair only zero-flow, all-zero quote placeholders and return a sparse ledger."""

    if frame.empty:
        return pd.DataFrame()
    for column in [*PRICE_COLUMNS, "volume", "amount"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    candidates = frame["volume"].eq(0) & frame["amount"].eq(0) & frame[PRICE_COLUMNS].eq(0).all(axis=1)
    if not candidates.any():
        return pd.DataFrame()
    replacement = frame["close"].where(frame["close"] > 0).ffill().bfill()
    repairable = candidates & replacement.notna()
    if not repairable.any():
        return pd.DataFrame()
    timestamps = pd.to_datetime(frame.loc[repairable, "timestamp"], errors="coerce")
    ledger = pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": timestamps.dt.strftime("%Y-%m-%d"),
            "bar_time": timestamps.dt.strftime("%H%M00000"),
            "original_open": frame.loc[repairable, "open"].astype("float64"),
            "original_high": frame.loc[repairable, "high"].astype("float64"),
            "original_low": frame.loc[repairable, "low"].astype("float64"),
            "original_close": frame.loc[repairable, "close"].astype("float64"),
            "repaired_price": replacement.loc[repairable].astype("float64"),
            "repair_reason": "all_zero_ohlc_with_zero_volume_and_amount",
        }
    )
    for column in PRICE_COLUMNS:
        frame.loc[repairable, column] = replacement.loc[repairable]
    return ledger.reset_index(drop=True)


def prepare_member_frame(
    frame: pd.DataFrame,
    *,
    member: ArchiveMember,
    year_label: str,
) -> None:
    """Normalize numeric/time keys once and reject non-canonical source rows."""

    if frame.empty:
        return
    timestamp = pd.to_datetime(frame["timestamp"], errors="coerce")
    invalid_timestamp = int(timestamp.isna().sum())
    frame["timestamp"] = timestamp
    frame["trade_date"] = timestamp.to_numpy(dtype="datetime64[D]").astype(str)
    clock = timestamp.dt.hour * 100 + timestamp.dt.minute
    frame["bar_time"] = clock.map(BAR_TIME_BY_HHMM)
    for column in RAW_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    values = frame[["open", "high", "low", "close", "volume", "amount"]].to_numpy(
        dtype="float64",
        copy=False,
    )
    failures = {
        "invalid_timestamp": invalid_timestamp,
        "nonfinite_ohlcva": int((~np.isfinite(values)).any(axis=1).sum()),
        "impossible_ohlc": int(
            (
                (frame["low"] > frame[["open", "close"]].min(axis=1))
                | (frame["high"] < frame[["open", "close"]].max(axis=1))
                | (frame["low"] > frame["high"])
                | (frame[["open", "high", "low", "close"]] <= 0).any(axis=1)
            ).sum()
        ),
        "negative_flow": int(((frame["volume"] < 0) | (frame["amount"] < 0)).sum()),
        "duplicate_timestamp": int(frame.duplicated(["timestamp"]).sum()),
        "invalid_clock": int((~frame["bar_time"].isin(CONTINUOUS_TIMES | {AUCTION_TIME})).sum()),
    }
    if any(failures.values()):
        raise MinuteArchiveError(f"member_quality_failed:{member.symbol}:{year_label}:{json.dumps(failures)}")


def member_frame_audit(
    frame: pd.DataFrame,
    *,
    member: ArchiveMember,
    year: int,
) -> dict[str, Any]:
    identity = {
        "symbol": member.symbol,
        "member": member.normalized_member,
        "crc32": f"{member.crc:08x}",
        "compressed_size": member.compressed_size,
        "uncompressed_size": member.uncompressed_size,
        "year": int(year),
    }
    if frame.empty:
        return {
            **identity,
            "row_count": 0,
            "date_count": 0,
            "date_min": "",
            "date_max": "",
        }
    dates = frame["trade_date"]
    return {
        **identity,
        "row_count": len(frame),
        "date_count": int(dates.nunique()),
        "date_min": str(dates.min()),
        "date_max": str(dates.max()),
    }


def validate_member_frame(
    frame: pd.DataFrame,
    *,
    member: ArchiveMember,
    year: int,
) -> dict[str, Any]:
    """Backwards-compatible one-year validation and audit helper."""

    prepare_member_frame(frame, member=member, year_label=str(int(year)))
    return member_frame_audit(frame, member=member, year=year)


def daily_evidence(frame: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    continuous = frame.loc[frame["bar_time"].isin(CONTINUOUS_TIMES)]
    auction = frame.loc[frame["bar_time"].eq(AUCTION_TIME)]
    counts = continuous.groupby("trade_date", sort=True).agg(
        continuous_bar_count=("bar_time", "size"),
        continuous_volume=("volume", "sum"),
        continuous_amount=("amount", "sum"),
    )
    auction_summary = auction.groupby("trade_date", sort=True).agg(
        auction_row_count=("bar_time", "size"),
        auction_volume=("volume", "sum"),
        auction_amount=("amount", "sum"),
    )
    counts = counts.join(auction_summary, how="left")
    counts["auction_row_count"] = counts["auction_row_count"].fillna(0).astype("int64")
    counts[["auction_volume", "auction_amount"]] = counts[["auction_volume", "auction_amount"]].fillna(0.0)
    counts = counts.reset_index()
    counts.insert(0, "symbol", symbol)
    shares = (
        frame.groupby("trade_date", sort=True)[["turnover_pct", "float_shares", "total_shares"]].first().reset_index()
    )
    shares.insert(0, "symbol", symbol)
    return counts, shares


def qdp_daily_paths(workspace: Path) -> list[Path]:
    root = qdp_paths(workspace).qdp_v2_dir
    active = read_active_manifest(root)
    dataset_id = str(dict(active.get("datasets", {}) or {}).get("market_daily_raw", ""))
    manifest_path = dataset_manifest_for_id(root, dataset_id, "market_daily_raw")
    if manifest_path is None:
        raise MinuteArchiveError("active_market_daily_raw_missing")
    manifest = read_dataset_manifest(manifest_path)
    return [(root / shard.path).resolve() for shard in manifest.shards]


def parquet_scan(paths: Sequence[Path]) -> str:
    quoted = ",".join("'" + path.as_posix().replace("'", "''") + "'" for path in paths)
    return f"read_parquet([{quoted}], union_by_name=true, hive_partitioning=false)"


def _create_daily_comparison(
    connection: duckdb.DuckDBPyConnection,
    *,
    bars: str,
    daily: str,
    year: int,
) -> None:
    connection.execute(
        f"""
        CREATE TEMP TABLE aggregate AS
            SELECT symbol,trade_date,
                   coalesce(arg_min(open,bar_time) FILTER(WHERE volume>0 OR amount>0),arg_min(open,bar_time)) agg_open,
                   coalesce(max(high) FILTER(WHERE volume>0 OR amount>0),max(high)) agg_high,
                   coalesce(min(low) FILTER(WHERE volume>0 OR amount>0),min(low)) agg_low,
                   coalesce(arg_max(close,bar_time) FILTER(WHERE volume>0 OR amount>0),arg_max(close,bar_time)) agg_close,
                   sum(volume) agg_volume,sum(amount) agg_amount
            FROM {bars} GROUP BY symbol,trade_date
        """
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE daily_year AS
        SELECT symbol,trade_date,open,high,low,close,volume,amount
        FROM {daily}
        WHERE trade_date BETWEEN '{year}-01-01' AND '{year}-12-31'
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE compared AS
        SELECT a.*,d.open d_open,d.high d_high,d.low d_low,d.close d_close,
               d.volume d_volume,d.amount d_amount,
               greatest(abs(agg_open-d_open),abs(agg_high-d_high),
                        abs(agg_low-d_low),abs(agg_close-d_close)) price_max_abs_error,
               abs(agg_volume-d_volume)/greatest(abs(d_volume),1.0) volume_relative_error,
               abs(agg_amount-d_amount)/greatest(abs(d_amount),1.0) amount_relative_error
        FROM aggregate a JOIN daily_year d USING(symbol,trade_date)
        """
    )


def _write_sparse_parity_evidence(
    connection: duckdb.DuckDBPyConnection,
    *,
    material_path: Path,
    exclusions_path: Path,
    missing_path: Path,
) -> None:
    connection.execute(
        f"COPY (SELECT * FROM compared WHERE price_max_abs_error>0.0100001 "
        "OR volume_relative_error>0.001 OR amount_relative_error>0.001 "
        f"ORDER BY trade_date,symbol) TO '{material_path.as_posix()}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.execute(
        f"COPY (SELECT symbol,trade_date,price_max_abs_error,"
        "'daily_ohlc_mismatch_over_one_cent' AS exclusion_reason FROM compared "
        "WHERE price_max_abs_error>0.0100001 ORDER BY trade_date,symbol) "
        f"TO '{exclusions_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.execute(
        "COPY (SELECT d.symbol,d.trade_date,'daily_reference_has_no_minute_rows' AS reason "
        "FROM daily_year d ANTI JOIN aggregate a USING(symbol,trade_date) "
        f"ORDER BY d.trade_date,d.symbol) TO '{missing_path.as_posix()}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )


def parity_audit(
    *,
    continuous_paths: Sequence[Path],
    auction_paths: Sequence[Path],
    daily_paths: Sequence[Path],
    year: int,
    output_dir: Path,
    final_quality_dir: Path,
) -> dict[str, Any]:
    bars = parquet_scan([*continuous_paths, *auction_paths])
    daily = parquet_scan(daily_paths)
    material_path = output_dir / "daily_parity_material_mismatches.parquet"
    exclusions_path = output_dir / "minute_feature_exclusions.parquet"
    missing_path = output_dir / "daily_reference_missing_minute.parquet"
    for path in (material_path, exclusions_path, missing_path):
        path.unlink(missing_ok=True)
    with duckdb.connect() as connection:
        _create_daily_comparison(connection, bars=bars, daily=daily, year=int(year))
        row = connection.execute(
            """
            SELECT count(*),
                   avg((abs(agg_open-d_open)<1e-8)::INT),avg((abs(agg_high-d_high)<1e-8)::INT),
                   avg((abs(agg_low-d_low)<1e-8)::INT),avg((abs(agg_close-d_close)<1e-8)::INT),
                   quantile_cont(volume_relative_error,0.5),quantile_cont(volume_relative_error,0.95),
                   quantile_cont(amount_relative_error,0.5),quantile_cont(amount_relative_error,0.95),
                   count(*) FILTER(WHERE price_max_abs_error>0.0100001),
                   count(*) FILTER(WHERE price_max_abs_error>0.0100001
                                          OR volume_relative_error>0.001
                                          OR amount_relative_error>0.001)
            FROM compared
            """
        ).fetchone()
        _write_sparse_parity_evidence(
            connection,
            material_path=material_path,
            exclusions_path=exclusions_path,
            missing_path=missing_path,
        )
        coverage = connection.execute(
            """
            SELECT
                (SELECT count(*) FROM aggregate),
                (SELECT count(*) FROM daily_year),
                (SELECT count(*) FROM compared),
                (SELECT count(*) FROM daily_year d ANTI JOIN aggregate a USING(symbol,trade_date)),
                (SELECT count(*) FROM aggregate a ANTI JOIN daily_year d USING(symbol,trade_date))
            """
        ).fetchone()
        duplicate = int(
            connection.execute(
                f"SELECT count(*) FROM (SELECT symbol,trade_date,bar_time,count(*) n "
                f"FROM {bars} GROUP BY 1,2,3 HAVING n>1)"
            ).fetchone()[0]
        )
    return {
        "source_stock_days": int(coverage[0] or 0),
        "daily_reference_stock_days": int(coverage[1] or 0),
        "common_stock_days": int(row[0] or 0),
        "daily_reference_missing_minute_rows": int(coverage[3] or 0),
        "source_only_stock_days": int(coverage[4] or 0),
        "daily_reference_coverage_rate": (float((coverage[2] or 0) / coverage[1]) if coverage[1] else 0.0),
        "daily_reference_missing_minute_path": str((final_quality_dir / missing_path.name).resolve()),
        "open_exact_rate": float(row[1] or 0),
        "high_exact_rate": float(row[2] or 0),
        "low_exact_rate": float(row[3] or 0),
        "close_exact_rate": float(row[4] or 0),
        "volume_relative_error_median": float(row[5] or 0),
        "volume_relative_error_p95": float(row[6] or 0),
        "amount_relative_error_median": float(row[7] or 0),
        "amount_relative_error_p95": float(row[8] or 0),
        "price_over_one_cent_rows": int(row[9] or 0),
        "material_daily_mismatch_rows": int(row[10] or 0),
        "minute_feature_exclusion_rows": int(row[9] or 0),
        "minute_feature_exclusions_path": str((final_quality_dir / exclusions_path.name).resolve()),
        "duplicate_primary_keys": duplicate,
    }


__all__ = [
    "daily_evidence",
    "member_frame_audit",
    "parity_audit",
    "prepare_member_frame",
    "qdp_daily_paths",
    "repair_mislabeled_1300_as_1130",
    "repair_zero_price_placeholders",
    "validate_member_frame",
]
