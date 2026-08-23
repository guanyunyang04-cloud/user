"""ZIP discovery, bounded reads, normalization, and small research extracts."""

from __future__ import annotations

from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from quantlab.core.io import write_json
from quantlab.data.minute_archive.contracts import (
    AUCTION_TIME,
    BAR_SCHEMA,
    BAR_TIME_BY_HHMM,
    CORE_COLUMNS,
    OUTPUT_COLUMNS,
    PERIOD_PATTERN,
    PRICE_MODE,
    RAW_COLUMNS,
    SOURCE_NAME,
    SYMBOL_PATTERN,
    ArchiveMember,
    MinuteArchiveError,
)


def normalize_member_name(name: str) -> str:
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
                        crc=int(info.CRC),
                        compressed_size=int(info.compress_size),
                        uncompressed_size=int(info.file_size),
                    )
                )
    result.sort(key=lambda item: (item.symbol, str(item.archive), item.normalized_member))
    return result


def read_member(zipped: ZipFile, info: ZipInfo, *, symbol: str) -> pd.DataFrame:
    with zipped.open(info) as stream:
        payload = stream.read()
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = payload.decode("gb18030")
    frame = pd.read_csv(
        BytesIO(text.encode("utf-8")),
        header=0,
        names=list(RAW_COLUMNS),
        low_memory=False,
    )
    frame["symbol"] = symbol
    return frame


def _find_selected_year_start(data: bytes, *, years: tuple[int, ...]) -> int:
    positions = [position for year in years if (position := data.find(f"\n{year:04d}-".encode("ascii"))) >= 0]
    for year in years:
        if data.startswith(f"{year:04d}-".encode("ascii")):
            positions.append(0)
            break
    return min(positions) if positions else -1


def _find_year_end(data: bytes, *, year: int) -> int:
    position = data.find(f"\n{int(year) + 1:04d}-".encode("ascii"))
    return -1 if position < 0 else position + 1


def _parse_selected_rows(payload: bytes) -> pd.DataFrame:
    column_types = {
        "timestamp": pa.timestamp("s"),
        **{column: pa.float64() for column in RAW_COLUMNS[1:]},
    }
    try:
        table = pacsv.read_csv(
            BytesIO(payload),
            read_options=pacsv.ReadOptions(
                column_names=list(RAW_COLUMNS),
                block_size=8 << 20,
                use_threads=True,
            ),
            convert_options=pacsv.ConvertOptions(
                column_types=column_types,
                null_values=["", "nan", "NaN", "None", "null", "NULL", "--"],
                strings_can_be_null=True,
            ),
        )
        frame = table.to_pandas()
    except pa.ArrowInvalid as exc:
        message = str(exc)
        marker = "Expected 12 columns, got "
        try:
            observed_columns = int(message.split(marker, 1)[1].split(":", 1)[0])
        except (IndexError, ValueError):
            observed_columns = 0
        if not 7 <= observed_columns < len(RAW_COLUMNS):
            raise
        frame = pd.read_csv(
            BytesIO(payload),
            header=None,
            names=list(RAW_COLUMNS),
            low_memory=False,
        )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame


def read_member_year(
    zipped: ZipFile,
    info: ZipInfo,
    *,
    symbol: str,
    year: int,
) -> pd.DataFrame:
    """Read exactly one sorted source year and stop before later rows."""

    return read_member_years(
        zipped,
        info,
        symbol=symbol,
        years=[int(year)],
    )


def read_member_years(
    zipped: ZipFile,
    info: ZipInfo,
    *,
    symbol: str,
    years: Iterable[int],
) -> pd.DataFrame:
    """Read selected sorted source years with one pass through a ZIP member."""

    selected_years = tuple(sorted({int(value) for value in years}))
    if not selected_years:
        raise MinuteArchiveError("member_years_empty")
    end_year = selected_years[-1]
    found = False
    finished = False
    tail = b""
    selected: list[bytes] = []
    keep = 32
    with zipped.open(info) as stream:
        stream.readline()
        while block := stream.read(8 << 20):
            data = tail + block
            if not found:
                start = _find_selected_year_start(data, years=selected_years)
                if start < 0:
                    tail = data[-keep:]
                    continue
                if data[start : start + 1] == b"\n":
                    start += 1
                data = data[start:]
                found = True
            end = _find_year_end(data, year=end_year)
            if end >= 0:
                selected.append(data[:end])
                finished = True
                break
            if len(data) > keep:
                selected.append(data[:-keep])
                tail = data[-keep:]
            else:
                tail = data
    if found and not finished:
        selected.append(tail)
    if not selected:
        return pd.DataFrame(columns=[*RAW_COLUMNS, "symbol"])
    try:
        frame = _parse_selected_rows(b"".join(selected))
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        raise MinuteArchiveError(f"member_parse_failed:{symbol}:{info.filename}:{exc}") from exc
    frame = frame.loc[frame["timestamp"].dt.year.isin(selected_years)].copy()
    frame["symbol"] = symbol
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
    """Normalize a source frame to the core OHLCVA contract."""

    del source_archive, source_member
    required = {"symbol", *RAW_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise MinuteArchiveError(f"member_columns_missing:{','.join(missing)}")
    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], errors="coerce")
    output = output.loc[output["timestamp"].notna()].copy()
    if "trade_date" not in output:
        output["trade_date"] = output["timestamp"].to_numpy(dtype="datetime64[D]").astype(str)
    if start_date:
        output = output.loc[output["trade_date"] >= str(start_date)]
    if end_date:
        output = output.loc[output["trade_date"] <= str(end_date)]
    if "bar_time" not in output:
        clock = output["timestamp"].dt.hour * 100 + output["timestamp"].dt.minute
        output["bar_time"] = clock.map(BAR_TIME_BY_HHMM)
    if exclude_0930:
        output = output.loc[output["bar_time"] != AUCTION_TIME].copy()
    if start_bar:
        output = output.loc[output["bar_time"] >= str(start_bar)].copy()
    if end_bar:
        output = output.loc[output["bar_time"] <= str(end_bar)].copy()
    output = output.sort_values(["symbol", "timestamp"], kind="stable")
    if output.duplicated(["symbol", "timestamp"]).any():
        raise MinuteArchiveError("duplicate_symbol_timestamp")
    for column in ("open", "high", "low", "close", "volume", "amount"):
        output[column] = pd.to_numeric(output[column], errors="coerce").astype("float64")
    output["source"] = SOURCE_NAME
    output["adjusted_flag"] = PRICE_MODE
    return output.loc[:, list(CORE_COLUMNS)].reset_index(drop=True)


def bar_table(frame: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(
        frame.loc[:, list(CORE_COLUMNS)],
        schema=BAR_SCHEMA,
        preserve_index=False,
        safe=True,
    )


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
    """Create a compact research extract; formal QDP imports use ``import_year``."""

    archive_values = list(archives)
    output = Path(output_path).resolve()
    if output.exists() and not overwrite:
        raise MinuteArchiveError(f"output_exists:{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    members = list_members(archive_values, symbols=symbols)
    if not members:
        raise MinuteArchiveError("no_matching_1m_members")
    output.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    row_count = 0
    member_count = 0
    try:
        grouped: dict[Path, list[ArchiveMember]] = {}
        for member in members:
            grouped.setdefault(member.archive, []).append(member)
        for archive, selected in grouped.items():
            with ZipFile(archive) as zipped:
                for member in selected:
                    raw = read_member(zipped, zipped.getinfo(member.member), symbol=member.symbol)
                    data = standardize_frame(
                        raw,
                        start_date=start_date,
                        end_date=end_date,
                        exclude_0930=exclude_0930,
                        start_bar=start_bar,
                        end_bar=end_bar,
                    )
                    if data.empty:
                        continue
                    if writer is None:
                        writer = pq.ParquetWriter(output, BAR_SCHEMA, compression="zstd", use_dictionary=True)
                    writer.write_table(bar_table(data))
                    row_count += len(data)
                    member_count += 1
    finally:
        if writer is not None:
            writer.close()
    if not row_count:
        output.unlink(missing_ok=True)
        raise MinuteArchiveError("selected_extract_is_empty")
    manifest = {
        "schema": "quantlab.minute_archive_extract/2",
        "status": "ready",
        "price_mode": PRICE_MODE,
        "opening_auction_policy": "exclude_0930" if exclude_0930 else "retain_0930",
        "row_count": row_count,
        "member_count": member_count,
        "output": {"path": str(output), "size": output.stat().st_size},
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    return manifest


__all__ = [
    "OUTPUT_COLUMNS",
    "bar_table",
    "extract_to_parquet",
    "list_members",
    "normalize_member_name",
    "read_member_year",
    "read_member_years",
    "standardize_frame",
]
