"""Historical Intraday External Archive Repair: inventory responsibilities."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
)
from quantlab.data.qdp_v2.recent_market_repair import (
    INTRADAY_DOMAIN,
)
from quantlab.data.qdp_v2.research_event_update import (
    _sha256,
)
from quantlab.data.qdp_v2.status import active_dataset_map

from .config import (
    _MEMBER_PATTERN,
    _NUMERIC_COLUMNS,
    END_DATE,
    START_DATE,
    ArchiveMember,
    ExternalArchiveRepairError,
)
from .context import (
    _baseline_manifest,
)


def _gap_inventory(workspace: Path) -> tuple[pd.DataFrame, Path, str]:
    baseline = _baseline_manifest(workspace)
    item = dict(dict(baseline.get("files", {})).get("historical_intraday_5m_gap", {}))
    path = Path(str(item.get("path", ""))).resolve()
    expected_hash = str(item.get("sha256", ""))
    if not path.is_file() or _sha256(path) != expected_hash:
        raise ExternalArchiveRepairError("historical_intraday_gap_inventory_changed")
    frame = pd.read_parquet(path)
    required = {
        "symbol",
        "trade_date",
        *_NUMERIC_COLUMNS,
        "quality_liquidity_keep",
    }
    if not required.issubset(frame.columns):
        raise ExternalArchiveRepairError("historical_intraday_gap_schema_invalid")
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["trade_date"] = frame["trade_date"].astype(str).str[:10]
    if frame.duplicated(["symbol", "trade_date"]).any():
        raise ExternalArchiveRepairError("historical_intraday_gap_key_duplicate")
    if not frame["trade_date"].between(START_DATE, END_DATE).all():
        raise ExternalArchiveRepairError("historical_intraday_gap_reads_2026")
    if len(frame) != 491_775:
        raise ExternalArchiveRepairError(f"historical_intraday_gap_count:{len(frame)}")
    return frame, path, expected_hash


def _active_intraday(workspace: Path) -> tuple[DatasetManifest, Path]:
    root = qdp_v2_root(workspace)
    dataset_id = active_dataset_map(read_active_manifest(root)).get(INTRADAY_DOMAIN, "")
    path = dataset_manifest_for_id(root, dataset_id, INTRADAY_DOMAIN)
    if path is None:
        raise ExternalArchiveRepairError("active_intraday_manifest_missing")
    return read_dataset_manifest(path), path


def _symbol_from_member(value: str) -> str:
    basename = str(value).replace("\\", "/").rsplit("/", 1)[-1]
    match = _MEMBER_PATTERN.fullmatch(basename)
    if not match:
        return ""
    return f"{match.group('code')}.{match.group('exchange').upper()}"


def _scan_archive(archive_path: Path) -> dict[str, ArchiveMember]:
    members: dict[str, ArchiveMember] = {}
    with zipfile.ZipFile(archive_path, mode="r", metadata_encoding="gbk") as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            symbol = _symbol_from_member(info.filename)
            if not symbol:
                continue
            if symbol in members:
                raise ExternalArchiveRepairError(f"archive_symbol_duplicate:{symbol}")
            members[symbol] = ArchiveMember(
                symbol=symbol,
                member_name=info.filename,
                file_size=int(info.file_size),
                compressed_size=int(info.compress_size),
                crc32=f"{int(info.CRC):08x}",
            )
    return members
