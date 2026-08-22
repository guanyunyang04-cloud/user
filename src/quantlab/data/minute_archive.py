"""Read and standardize the downloaded minute-bar ZIP archives.

The archive is kept immutable.  This module only reads CSV members, removes
the standalone 09:30 opening-auction row when requested, and writes a compact
Parquet extract with an explicit raw-price contract.  It deliberately does
not write to the iQuant cache.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quantlab.core.io import sha256_file, write_json

SYMBOL_PATTERN = re.compile(r"(?:^|/)(?P<market>sh|sz|bj)(?P<code>\d{6})\.csv$", re.I)
PERIOD_PATTERN = re.compile(r"(?P<period>\d+)\s*(?:分钟|min)", re.I)
CANONICAL_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "price_change",
    "pct_change",
    "turnover_pct",
    "float_shares",
    "total_shares",
)
OUTPUT_COLUMNS = (
    "symbol",
    "trade_date",
    "bar_time",
    *CANONICAL_COLUMNS[1:],
    "source_archive",
    "source_member",
)
PRICE_MODE = "raw_unadjusted"


class MinuteArchiveError(ValueError):
    """Raised when an archive member violates the minute-bar contract."""


@dataclass(frozen=True)
class ArchiveMember:
    archive: Path
    member: str
    normalized_member: str
    symbol: str
    period: str


def normalize_member_name(name: str) -> str:
    """Recover GBK names from ZIPs that omit the UTF-8 filename flag."""

    text = str(name).replace("\\", "/")
    try:
        decoded = text.encode("cp437").decode("gbk")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    return decoded if "�" not in decoded else text


def _member_symbol(name: str) -> str | None:
    match = SYMBOL_PATTERN.search(name.lower())
    if not match:
        return None
    return f"{match.group('code')}.{match.group('market').upper()}"


def _member_period(name: str, archive: Path) -> str | None:
    match = PERIOD_PATTERN.search(name)
    if match:
        return f"{match.group('period')}m"
    # The older archive contains only a directory named 1分钟.  Keep the
    # fallback explicit rather than guessing for arbitrary future archives.
    if "1分钟" in normalize_member_name(archive.name):
        return "1m"
    return None


def list_members(
    archives: Iterable[str | Path],
    *,
    symbols: Iterable[str] | None = None,
    period: str = "1m",
) -> list[ArchiveMember]:
    wanted = {str(value).upper() for value in symbols or ()}
    normalized_period = str(period).strip().lower()
    if normalized_period != "1m":
        raise MinuteArchiveError(f"unsupported_archive_period:{period}")
    result: list[ArchiveMember] = []
    for archive_value in archives:
        archive = Path(archive_value).resolve()
        if not archive.is_file():
            raise MinuteArchiveError(f"archive_missing:{archive}")
        with ZipFile(archive) as zipped:
            for info in zipped.infolist():
                if info.is_dir():
                    continue
                normalized = normalize_member_name(info.filename)
                symbol = _member_symbol(normalized)
                member_period = _member_period(normalized, archive)
                if symbol is None or member_period != normalized_period:
                    continue
                if wanted and symbol not in wanted:
                    continue
                result.append(
                    ArchiveMember(
                        archive=archive,
                        member=info.filename,
                        normalized_member=normalized,
                        symbol=symbol,
                        period=member_period,
                    )
                )
    result.sort(key=lambda item: (item.symbol, str(item.archive), item.normalized_member))
    return result


def _read_member(zipped: ZipFile, info: ZipInfo, *, symbol: str) -> pd.DataFrame:
    with zipped.open(info) as stream:
        payload = stream.read()
    # The downloaded CSV payloads are UTF-8 with BOM even though the ZIP
    # member names use a legacy GBK encoding.
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = payload.decode("gb18030")
    from io import StringIO

    frame = pd.read_csv(
        StringIO(text),
        header=0,
        names=list(CANONICAL_COLUMNS),
        low_memory=False,
    )
    if len(frame.columns) < len(CANONICAL_COLUMNS):
        raise MinuteArchiveError(f"columns_missing:{symbol}:{len(frame.columns)}")
    frame = frame.iloc[:, : len(CANONICAL_COLUMNS)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.loc[frame["timestamp"].notna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=[*CANONICAL_COLUMNS, "symbol"])
    for column in CANONICAL_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["symbol"] = str(symbol)
    return frame


def standardize_frame(
    frame: pd.DataFrame,
    *,
    start_date: str = "",
    end_date: str = "",
    exclude_0930: bool = True,
    start_bar: str = "",
    end_bar: str = "",
    source_archive: str = "",
    source_member: str = "",
) -> pd.DataFrame:
    """Return canonical local-time bars with an explicit auction policy."""

    required = {"symbol", *CANONICAL_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise MinuteArchiveError(f"member_columns_missing:{','.join(missing)}")
    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], errors="coerce")
    output = output.loc[output["timestamp"].notna()].copy()
    output["trade_date"] = output["timestamp"].dt.strftime("%Y-%m-%d")
    if start_date:
        output = output.loc[output["trade_date"] >= str(start_date)]
    if end_date:
        output = output.loc[output["trade_date"] <= str(end_date)]
    output["bar_time"] = output["timestamp"].dt.strftime("%H%M00000")
    if exclude_0930:
        output = output.loc[output["bar_time"] != "093000000"].copy()
    if start_bar:
        output = output.loc[output["bar_time"] >= str(start_bar)].copy()
    if end_bar:
        output = output.loc[output["bar_time"] <= str(end_bar)].copy()
    output = output.sort_values(["symbol", "timestamp"], kind="stable")
    output = output.drop_duplicates(["symbol", "timestamp"], keep="last")
    numeric = list(CANONICAL_COLUMNS[1:])
    output[numeric] = output[numeric].replace([np.inf, -np.inf], np.nan).astype("float64")
    output["source_archive"] = str(source_archive)
    output["source_member"] = str(source_member)
    return output.loc[:, list(OUTPUT_COLUMNS)].reset_index(drop=True)


def _source_info(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path) if stat.st_size < 512 * (1 << 20) else None,
        "sha256_skipped_for_bytes": bool(stat.st_size >= 512 * (1 << 20)),
    }


def extract_to_parquet(
    archives: Iterable[str | Path],
    *,
    symbols: Iterable[str],
    output_path: str | Path,
    start_date: str = "",
    end_date: str = "",
    exclude_0930: bool = True,
    start_bar: str = "",
    end_bar: str = "",
    overwrite: bool = False,
) -> dict[str, object]:
    """Extract selected symbols/dates without unpacking the source archives."""

    archive_values = list(archives)
    output = Path(output_path).resolve()
    if output.exists() and not overwrite:
        raise MinuteArchiveError(f"output_exists:{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = list_members(archive_values, symbols=symbols)
    if not selected:
        raise MinuteArchiveError("no_matching_1m_members")
    if output.exists():
        output.unlink()
    writer: pq.ParquetWriter | None = None
    row_count = 0
    member_rows: list[dict[str, object]] = []
    try:
        for member in selected:
            with ZipFile(member.archive) as zipped:
                info = zipped.getinfo(member.member)
                raw = _read_member(zipped, info, symbol=member.symbol)
            standardized = standardize_frame(
                raw,
                start_date=start_date,
                end_date=end_date,
                exclude_0930=exclude_0930,
                start_bar=start_bar,
                end_bar=end_bar,
                source_archive=str(member.archive),
                source_member=member.normalized_member,
            )
            if standardized.empty:
                continue
            table = pa.Table.from_pandas(standardized, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
            row_count += len(standardized)
            counts = standardized.groupby("trade_date", sort=True).size()
            member_rows.append(
                {
                    "archive": str(member.archive),
                    "member": member.normalized_member,
                    "symbol": member.symbol,
                    "row_count": int(len(standardized)),
                    "date_count": int(counts.size),
                    "date_min": str(counts.index.min()),
                    "date_max": str(counts.index.max()),
                    "minimum_bars_per_day": int(counts.min()),
                    "maximum_bars_per_day": int(counts.max()),
                    "opening_auction_rows_removed": int(
                        raw["timestamp"].dt.strftime("%H%M").eq("0930").sum()
                    )
                    if exclude_0930
                    else 0,
                }
            )
    finally:
        if writer is not None:
            writer.close()
    if row_count == 0:
        output.unlink(missing_ok=True)
        raise MinuteArchiveError("selected_extract_is_empty")
    dates = [str(row["date_min"]) for row in member_rows] + [str(row["date_max"]) for row in member_rows]
    manifest = {
        "schema": "quantlab.minute_archive/1",
        "status": "ready",
        "price_mode": PRICE_MODE,
        "bar_interval": "1m",
        "opening_auction_policy": "exclude_0930" if exclude_0930 else "retain_0930",
        "bar_time_range": [str(start_bar), str(end_bar)],
        "source_archives": [_source_info(Path(value).resolve()) for value in archive_values],
        "selected_symbols": sorted({str(value).upper() for value in symbols}),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "row_count": int(row_count),
        "member_count": len(member_rows),
        "date_min": min(dates) if dates else None,
        "date_max": max(dates) if dates else None,
        "output": {"path": str(output), "size": int(output.stat().st_size)},
        "members": member_rows,
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract selected minute CSV members to Parquet")
    parser.add_argument("--archive", action="append", required=True)
    parser.add_argument("--symbol", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--retain-0930", action="store_true")
    parser.add_argument("--start-bar", default="")
    parser.add_argument("--end-bar", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = extract_to_parquet(
        args.archive,
        symbols=args.symbol,
        output_path=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        exclude_0930=not args.retain_0930,
        start_bar=args.start_bar,
        end_bar=args.end_bar,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ArchiveMember",
    "CANONICAL_COLUMNS",
    "MinuteArchiveError",
    "OUTPUT_COLUMNS",
    "extract_to_parquet",
    "list_members",
    "normalize_member_name",
    "standardize_frame",
]
