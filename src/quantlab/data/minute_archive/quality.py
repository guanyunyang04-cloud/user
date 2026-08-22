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


def validate_member_frame(
    frame: pd.DataFrame,
    *,
    member: ArchiveMember,
    year: int,
) -> dict[str, Any]:
    if frame.empty:
        return {"row_count": 0, "date_count": 0, "date_min": "", "date_max": ""}
    timestamp = pd.to_datetime(frame["timestamp"], errors="coerce")
    invalid_timestamp = int(timestamp.isna().sum())
    frame["timestamp"] = timestamp
    frame["trade_date"] = timestamp.dt.strftime("%Y-%m-%d")
    frame["bar_time"] = timestamp.dt.strftime("%H%M00000")
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
        raise MinuteArchiveError(f"member_quality_failed:{member.symbol}:{json.dumps(failures)}")
    dates = frame["trade_date"]
    return {
        "symbol": member.symbol,
        "member": member.normalized_member,
        "crc32": f"{member.crc:08x}",
        "compressed_size": member.compressed_size,
        "uncompressed_size": member.uncompressed_size,
        "row_count": len(frame),
        "date_count": int(dates.nunique()),
        "date_min": str(dates.min()),
        "date_max": str(dates.max()),
        "year": int(year),
    }


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
    counts[["auction_volume", "auction_amount"]] = counts[
        ["auction_volume", "auction_amount"]
    ].fillna(0.0)
    counts = counts.reset_index()
    counts.insert(0, "symbol", symbol)
    shares = (
        frame.groupby("trade_date", sort=True)[["turnover_pct", "float_shares", "total_shares"]]
        .first()
        .reset_index()
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
        CREATE TEMP TABLE compared AS
        WITH aggregate AS (
            SELECT symbol,trade_date,
                   coalesce(arg_min(open,bar_time) FILTER(WHERE volume>0 OR amount>0),arg_min(open,bar_time)) agg_open,
                   coalesce(max(high) FILTER(WHERE volume>0 OR amount>0),max(high)) agg_high,
                   coalesce(min(low) FILTER(WHERE volume>0 OR amount>0),min(low)) agg_low,
                   coalesce(arg_max(close,bar_time) FILTER(WHERE volume>0 OR amount>0),arg_max(close,bar_time)) agg_close,
                   sum(volume) agg_volume,sum(amount) agg_amount
            FROM {bars} GROUP BY symbol,trade_date
        )
        SELECT a.*,d.open d_open,d.high d_high,d.low d_low,d.close d_close,
               d.volume d_volume,d.amount d_amount,
               greatest(abs(agg_open-d_open),abs(agg_high-d_high),
                        abs(agg_low-d_low),abs(agg_close-d_close)) price_max_abs_error,
               abs(agg_volume-d_volume)/greatest(abs(d_volume),1.0) volume_relative_error,
               abs(agg_amount-d_amount)/greatest(abs(d_amount),1.0) amount_relative_error
        FROM aggregate a JOIN {daily} d USING(symbol,trade_date)
        WHERE a.trade_date BETWEEN '{year}-01-01' AND '{year}-12-31'
        """
    )


def _write_sparse_parity_evidence(
    connection: duckdb.DuckDBPyConnection,
    *,
    material_path: Path,
    exclusions_path: Path,
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
        )
        duplicate = int(
            connection.execute(
                f"SELECT count(*) FROM (SELECT symbol,trade_date,bar_time,count(*) n "
                f"FROM {bars} GROUP BY 1,2,3 HAVING n>1)"
            ).fetchone()[0]
        )
    return {
        "common_stock_days": int(row[0] or 0),
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


__all__ = ["daily_evidence", "parity_audit", "qdp_daily_paths", "validate_member_frame"]
