from __future__ import annotations

"""Repair exact historical 5-minute gaps from the external local archive.

Only stock-days frozen in the pre-repair inventory are considered.  A source
file never changes PIT eligibility, malformed days are never interpolated, and
the active dataset is switched only after a new immutable version is complete.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import time
import zipfile
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from quantlab.data.core.json_io import json_safe
from quantlab.data.core.paths import qdp_paths
from quantlab.data.qdp_v2.manifest import (
    EXPECTED_BAR_TIMES,
    DatasetManifest,
    ShardManifestEntry,
    atomic_write_json,
    dataset_manifest_for_id,
    path_for_manifest,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    utc_now,
    write_active_manifest,
    write_dataset_manifest,
)
from quantlab.data.qdp_v2.recent_market_repair import (
    INTRADAY_COLUMNS,
    INTRADAY_DOMAIN,
    _write_final_parquet,
)
from quantlab.data.qdp_v2.research_event_update import (
    _assert_credential_free,
    _sha256,
)
from quantlab.data.qdp_v2.status import active_dataset_map

REPAIR_ID = "historical_intraday_5m_external_archive_repair_v2"
BASELINE_ID = "pretraining_data_repair_baseline_v1"
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"
DEFAULT_ARCHIVE = Path(r"H:\BaiduNetdiskDownload\量化数据\5分钟(2000-2025).zip")
DEFAULT_WORKERS = 4
PRICE_RELATIVE_TOLERANCE = 0.02
VOLUME_RELATIVE_TOLERANCE = 0.05
AMOUNT_RELATIVE_TOLERANCE = 0.05
SOURCE_NAME = "external_quant_archive_daily_validated_v2"
EXPECTED_CANDIDATE_DAYS = 130_060
EXPECTED_CANDIDATE_SYMBOLS = 174
EXPECTED_ACCEPTED_DAYS = 129_116
EXPECTED_FORMAL_QUALITY_ACCEPTED_DAYS = 35_450
EXPECTED_FORMAL_QUALITY_REMAINING_DAYS = 152
EXPECTED_RECENT_QUALITY_REMAINING_DAYS = 3
EXPECTED_FORMAL_QUALITY_POOL_ROWS = 4_476_997
_MEMBER_PATTERN = re.compile(
    r"^(?P<exchange>sh|sz)(?P<code>\d{6})\.csv$", re.IGNORECASE
)
_NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume", "amount")
_PRICE_COLUMNS = ("open", "high", "low", "close")
_DECISION_COLUMNS = (
    "symbol",
    "trade_date",
    "quality_liquidity_keep",
    "archive_member",
    "decision",
    "rejection_reason",
    "source_bar_count",
    "normalized_bar_count",
    "price_relative_error",
    "volume_relative_error",
    "amount_relative_error",
    "accepted_row_count",
    "source",
)


class ExternalArchiveRepairError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArchiveMember:
    symbol: str
    member_name: str
    file_size: int
    compressed_size: int
    crc32: str


@dataclass(frozen=True)
class SymbolResult:
    symbol: str
    requested_day_count: int
    accepted_day_count: int
    rejected_day_count: int
    accepted_path: str
    decision_path: str
    receipt_path: str
    accepted_sha256: str
    decision_sha256: str
    reused: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "requested_day_count": self.requested_day_count,
            "accepted_day_count": self.accepted_day_count,
            "rejected_day_count": self.rejected_day_count,
            "accepted_path": self.accepted_path,
            "decision_path": self.decision_path,
            "receipt_path": self.receipt_path,
            "accepted_sha256": self.accepted_sha256,
            "decision_sha256": self.decision_sha256,
            "reused": self.reused,
        }


def _workspace(value: str | Path | None) -> Path:
    return Path(value or Path.cwd()).resolve()


def _runtime(workspace: Path) -> Path:
    path = qdp_paths(workspace).data_dir / "qdp_runtime" / REPAIR_ID
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _state_path(workspace: Path) -> Path:
    return _runtime(workspace) / "state.json"


def _read_state(workspace: Path) -> dict[str, Any]:
    path = _state_path(workspace)
    if not path.is_file():
        return {"repair_id": REPAIR_ID, "status": "pending"}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_state(workspace: Path, state: Mapping[str, Any]) -> None:
    payload = {**dict(state), "updated_at": utc_now()}
    _assert_credential_free(payload)
    last_error: PermissionError | None = None
    for attempt in range(6):
        try:
            atomic_write_json(_state_path(workspace), payload)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.10 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _baseline_manifest(workspace: Path) -> dict[str, Any]:
    path = qdp_paths(workspace).data_dir / "qdp_runtime" / BASELINE_ID / "manifest.json"
    if not path.is_file():
        raise ExternalArchiveRepairError(f"repair_baseline_missing:{path}")
    payload = dict(json.loads(path.read_text(encoding="utf-8")))
    if payload.get("status") != "frozen":
        raise ExternalArchiveRepairError("repair_baseline_not_frozen")
    if int(payload.get("read_2026_rows", -1)) != 0:
        raise ExternalArchiveRepairError("repair_baseline_read_2026_nonzero")
    return payload


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


def _positive_finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number > 0)


def _normalize_day(day: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    ordered = day.sort_values("bar_time", kind="stable").reset_index(drop=True)
    actual = tuple(ordered["bar_time"].astype(str))
    expected_colon = tuple(f"{item[:2]}:{item[2:4]}" for item in EXPECTED_BAR_TIMES)
    if len(ordered) == 48 and actual == expected_colon:
        normalized = ordered.copy()
    elif len(ordered) == 49 and actual == ("09:30", *expected_colon):
        auction = ordered.iloc[0]
        first = ordered.iloc[1]
        normalized = ordered.iloc[1:].copy().reset_index(drop=True)
        normalized.loc[0, "open"] = (
            float(auction["open"])
            if _positive_finite(auction["open"])
            else float(first["open"])
        )
        highs = [
            float(value)
            for value in (auction["high"], first["high"])
            if _positive_finite(value)
        ]
        lows = [
            float(value)
            for value in (auction["low"], first["low"])
            if _positive_finite(value)
        ]
        if not highs or not lows:
            return pd.DataFrame(), "auction_price_invalid"
        normalized.loc[0, "high"] = max(highs)
        normalized.loc[0, "low"] = min(lows)
        normalized.loc[0, "close"] = float(first["close"])
        normalized.loc[0, "volume"] = float(auction["volume"]) + float(first["volume"])
        normalized.loc[0, "amount"] = float(auction["amount"]) + float(first["amount"])
    else:
        if ordered["bar_time"].duplicated().any():
            return pd.DataFrame(), "duplicate_bar_time"
        return pd.DataFrame(), "invalid_bar_time_set"
    numeric = normalized.loc[:, _NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype="float64")).all():
        return pd.DataFrame(), "non_finite_numeric"
    if numeric.loc[:, _PRICE_COLUMNS].le(0).any().any():
        return pd.DataFrame(), "non_positive_price"
    if numeric.loc[:, ["volume", "amount"]].lt(0).any().any():
        return pd.DataFrame(), "negative_volume_or_amount"
    if (numeric["high"] < numeric[["open", "low", "close"]].max(axis=1)).any():
        return pd.DataFrame(), "invalid_high_relation"
    if (numeric["low"] > numeric[["open", "high", "close"]].min(axis=1)).any():
        return pd.DataFrame(), "invalid_low_relation"
    normalized.loc[:, _NUMERIC_COLUMNS] = numeric
    normalized["bar_time"] = list(EXPECTED_BAR_TIMES)
    return normalized.reset_index(drop=True), ""


def _relative_error(observed: float, reference: float) -> float:
    if not np.isfinite(observed) or not np.isfinite(reference):
        return float("inf")
    return abs(float(observed) - float(reference)) / max(abs(float(reference)), 1e-12)


def _validate_daily(
    day: pd.DataFrame,
    reference: Mapping[str, Any],
) -> tuple[bool, str, float, float, float]:
    price_errors = [
        _relative_error(
            float(day[column].iloc[0] if column == "open" else day[column].iloc[-1])
            if column in {"open", "close"}
            else float(day[column].max() if column == "high" else day[column].min()),
            float(reference[column]),
        )
        for column in _PRICE_COLUMNS
    ]
    price_error = max(price_errors)
    volume_error = _relative_error(
        float(day["volume"].sum()), float(reference["volume"])
    )
    amount_error = _relative_error(
        float(day["amount"].sum()), float(reference["amount"])
    )
    if price_error > PRICE_RELATIVE_TOLERANCE:
        reason = "daily_price_mismatch"
    elif (
        volume_error > VOLUME_RELATIVE_TOLERANCE
        and amount_error > AMOUNT_RELATIVE_TOLERANCE
    ):
        reason = "daily_volume_and_amount_mismatch"
    else:
        reason = ""
    return not reason, reason, price_error, volume_error, amount_error


def _symbol_paths(workspace: Path, symbol: str) -> tuple[Path, Path, Path]:
    safe = symbol.replace(".", "_")
    root = _runtime(workspace) / "symbols" / safe
    return root / "accepted.parquet", root / "decisions.parquet", root / "receipt.json"


def _load_reusable_result(
    workspace: Path,
    *,
    symbol: str,
    requested_day_count: int,
    member: ArchiveMember,
) -> SymbolResult | None:
    accepted_path, decision_path, receipt_path = _symbol_paths(workspace, symbol)
    if not receipt_path.is_file() or not decision_path.is_file():
        return None
    try:
        receipt = dict(json.loads(receipt_path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None
    accepted_count = int(receipt.get("accepted_day_count", -1))
    accepted_hash = str(receipt.get("accepted_sha256", ""))
    if (
        receipt.get("status") != "completed"
        or receipt.get("symbol") != symbol
        or receipt.get("archive_member") != member.member_name
        or receipt.get("member_crc32") != member.crc32
        or int(receipt.get("requested_day_count", -1)) != requested_day_count
        or _sha256(decision_path) != str(receipt.get("decision_sha256", ""))
        or int(pq.ParquetFile(decision_path).metadata.num_rows) != requested_day_count
    ):
        return None
    if accepted_count:
        if (
            not accepted_path.is_file()
            or _sha256(accepted_path) != accepted_hash
            or int(pq.ParquetFile(accepted_path).metadata.num_rows)
            != accepted_count * len(EXPECTED_BAR_TIMES)
        ):
            return None
    elif accepted_path.exists():
        return None
    return SymbolResult(
        symbol=symbol,
        requested_day_count=requested_day_count,
        accepted_day_count=accepted_count,
        rejected_day_count=requested_day_count - accepted_count,
        accepted_path=str(accepted_path) if accepted_count else "",
        decision_path=str(decision_path),
        receipt_path=str(receipt_path),
        accepted_sha256=accepted_hash,
        decision_sha256=str(receipt["decision_sha256"]),
        reused=True,
    )


def _read_candidate_rows(
    archive_path: Path,
    member: ArchiveMember,
    wanted_dates: set[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(archive_path, mode="r", metadata_encoding="gbk") as archive:
        try:
            stream = archive.open(member.member_name, mode="r")
        except (KeyError, zipfile.BadZipFile) as exc:
            raise ExternalArchiveRepairError(
                f"archive_member_unreadable:{member.symbol}:{member.member_name}"
            ) from exc
        with stream:
            try:
                chunks = pd.read_csv(
                    stream,
                    encoding="utf-8-sig",
                    usecols=list(range(7)),
                    chunksize=250_000,
                    low_memory=False,
                )
                for chunk in chunks:
                    if len(chunk.columns) != 7:
                        raise ExternalArchiveRepairError(
                            f"archive_member_schema:{member.symbol}"
                        )
                    chunk.columns = ["timestamp", *_NUMERIC_COLUMNS]
                    timestamp = chunk["timestamp"].astype(str)
                    mask = timestamp.str[:10].isin(wanted_dates)
                    if not mask.any():
                        continue
                    selected = chunk.loc[mask].copy()
                    selected["trade_date"] = timestamp.loc[mask].str[:10].to_numpy()
                    selected["bar_time"] = timestamp.loc[mask].str[11:16].to_numpy()
                    for column in _NUMERIC_COLUMNS:
                        selected[column] = pd.to_numeric(
                            selected[column], errors="coerce"
                        ).astype("float64")
                    frames.append(
                        selected.loc[:, ["trade_date", "bar_time", *_NUMERIC_COLUMNS]]
                    )
            except UnicodeDecodeError as exc:
                raise ExternalArchiveRepairError(
                    f"archive_member_encoding:{member.symbol}"
                ) from exc
    if not frames:
        return pd.DataFrame(columns=["trade_date", "bar_time", *_NUMERIC_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def _atomic_frame_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _process_symbol(
    workspace: Path,
    *,
    archive_path: Path,
    member: ArchiveMember,
    reference: pd.DataFrame,
) -> SymbolResult:
    symbol = member.symbol
    requested_day_count = len(reference)
    reusable = _load_reusable_result(
        workspace,
        symbol=symbol,
        requested_day_count=requested_day_count,
        member=member,
    )
    if reusable is not None:
        return reusable
    accepted_path, decision_path, receipt_path = _symbol_paths(workspace, symbol)
    raw = _read_candidate_rows(
        archive_path,
        member,
        set(reference["trade_date"].astype(str)),
    )
    grouped = {
        str(trade_date): day.copy()
        for trade_date, day in raw.groupby("trade_date", sort=False)
    }
    accepted: list[pd.DataFrame] = []
    decisions: list[dict[str, Any]] = []
    for row in reference.sort_values("trade_date", kind="stable").to_dict("records"):
        trade_date = str(row["trade_date"])
        day = grouped.get(trade_date)
        source_bar_count = len(day) if day is not None else 0
        normalized = pd.DataFrame()
        if day is None:
            reason = "source_day_absent"
        else:
            normalized, reason = _normalize_day(day)
        price_error = float("nan")
        volume_error = float("nan")
        amount_error = float("nan")
        if not reason:
            valid, reason, price_error, volume_error, amount_error = _validate_daily(
                normalized, row
            )
            assert valid == (not reason)
        decision = "rejected" if reason else "accepted"
        if not reason:
            normalized["symbol"] = symbol
            normalized["trade_date"] = trade_date
            normalized["source"] = SOURCE_NAME
            normalized["adjusted_flag"] = "none"
            accepted.append(normalized.loc[:, INTRADAY_COLUMNS])
        decisions.append(
            {
                "symbol": symbol,
                "trade_date": trade_date,
                "quality_liquidity_keep": bool(row["quality_liquidity_keep"]),
                "archive_member": member.member_name,
                "decision": decision,
                "rejection_reason": reason,
                "source_bar_count": source_bar_count,
                "normalized_bar_count": len(normalized),
                "price_relative_error": price_error,
                "volume_relative_error": volume_error,
                "amount_relative_error": amount_error,
                "accepted_row_count": len(normalized) if not reason else 0,
                "source": SOURCE_NAME,
            }
        )
    accepted_frame = (
        pd.concat(accepted, ignore_index=True)
        if accepted
        else pd.DataFrame(columns=INTRADAY_COLUMNS)
    )
    decision_frame = pd.DataFrame(decisions, columns=_DECISION_COLUMNS)
    if len(decision_frame) != requested_day_count:
        raise ExternalArchiveRepairError(f"symbol_decision_count:{symbol}")
    if not accepted_frame.empty:
        if accepted_frame.duplicated(["symbol", "trade_date", "bar_time"]).any():
            raise ExternalArchiveRepairError(f"symbol_accepted_key_duplicate:{symbol}")
        _write_final_parquet(
            accepted_frame.sort_values(["trade_date", "bar_time"], kind="stable"),
            accepted_path,
            domain=INTRADAY_DOMAIN,
        )
        accepted_hash = _sha256(accepted_path)
    else:
        accepted_path.unlink(missing_ok=True)
        accepted_hash = ""
    _atomic_frame_parquet(decision_frame, decision_path)
    decision_hash = _sha256(decision_path)
    accepted_days = int((decision_frame["decision"] == "accepted").sum())
    receipt = {
        "status": "completed",
        "symbol": symbol,
        "archive_path": str(archive_path),
        "archive_member": member.member_name,
        "member_file_size": member.file_size,
        "member_compressed_size": member.compressed_size,
        "member_crc32": member.crc32,
        "requested_day_count": requested_day_count,
        "accepted_day_count": accepted_days,
        "rejected_day_count": requested_day_count - accepted_days,
        "accepted_sha256": accepted_hash,
        "decision_sha256": decision_hash,
        "completed_at": utc_now(),
    }
    _assert_credential_free(receipt)
    atomic_write_json(receipt_path, receipt)
    return SymbolResult(
        symbol=symbol,
        requested_day_count=requested_day_count,
        accepted_day_count=accepted_days,
        rejected_day_count=requested_day_count - accepted_days,
        accepted_path=str(accepted_path) if accepted_days else "",
        decision_path=str(decision_path),
        receipt_path=str(receipt_path),
        accepted_sha256=accepted_hash,
        decision_sha256=decision_hash,
    )


def _initialize(
    workspace: Path,
    archive_path: Path,
) -> tuple[pd.DataFrame, dict[str, ArchiveMember], DatasetManifest, dict[str, Any]]:
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    inventory, inventory_path, inventory_hash = _gap_inventory(workspace)
    active_manifest, _ = _active_intraday(workspace)
    baseline = _baseline_manifest(workspace)
    baseline_intraday = str(
        dict(baseline.get("active_dataset_ids", {})).get(INTRADAY_DOMAIN, "")
    )
    if active_manifest.dataset_id != baseline_intraday:
        state = _read_state(workspace)
        installed = str(state.get("installed_dataset_id", ""))
        if not installed or active_manifest.dataset_id != installed:
            raise ExternalArchiveRepairError(
                "active_intraday_drifted_since_repair_baseline"
            )
    members = _scan_archive(archive_path)
    matched = sorted(set(inventory["symbol"]).intersection(members))
    candidate_count = int(inventory["symbol"].isin(matched).sum())
    if (
        len(matched) != EXPECTED_CANDIDATE_SYMBOLS
        or candidate_count != EXPECTED_CANDIDATE_DAYS
    ):
        raise ExternalArchiveRepairError(
            f"archive_candidate_contract:{len(matched)}:{candidate_count}"
        )
    state = _read_state(workspace)
    state.update(
        {
            "repair_id": REPAIR_ID,
            "status": "initialized",
            "archive_path": str(archive_path),
            "archive_size": archive_path.stat().st_size,
            "archive_mtime_ns": archive_path.stat().st_mtime_ns,
            "archive_member_count": len(members),
            "input_intraday_dataset_id": baseline_intraday,
            "inventory_path": str(inventory_path),
            "inventory_sha256": inventory_hash,
            "inventory_day_count": len(inventory),
            "candidate_symbol_count": len(matched),
            "candidate_day_count": candidate_count,
            "no_file_symbol_count": int(
                inventory.loc[~inventory["symbol"].isin(matched), "symbol"].nunique()
            ),
            "no_file_day_count": int((~inventory["symbol"].isin(matched)).sum()),
            "request_2026_count": 0,
            "write_2026_count": 0,
        }
    )
    _write_state(workspace, state)
    return inventory, members, active_manifest, state


def _no_file_decisions(inventory: pd.DataFrame, matched: set[str]) -> pd.DataFrame:
    frame = inventory.loc[
        ~inventory["symbol"].isin(matched),
        ["symbol", "trade_date", "quality_liquidity_keep"],
    ].copy()
    frame["archive_member"] = ""
    frame["decision"] = "no_file"
    frame["rejection_reason"] = "symbol_file_absent"
    frame["source_bar_count"] = 0
    frame["normalized_bar_count"] = 0
    frame["price_relative_error"] = np.nan
    frame["volume_relative_error"] = np.nan
    frame["amount_relative_error"] = np.nan
    frame["accepted_row_count"] = 0
    frame["source"] = SOURCE_NAME
    return frame.loc[:, _DECISION_COLUMNS]


def _assemble_decisions(
    workspace: Path,
    inventory: pd.DataFrame,
    results: Sequence[SymbolResult],
) -> Path:
    matched = {item.symbol for item in results}
    frames = [pd.read_parquet(item.decision_path) for item in results]
    frames.append(_no_file_decisions(inventory, matched))
    decisions = pd.concat(frames, ignore_index=True)
    decisions = decisions.sort_values(["trade_date", "symbol"], kind="stable")
    if len(decisions) != len(inventory):
        raise ExternalArchiveRepairError(
            f"decision_inventory_count:{len(decisions)}!={len(inventory)}"
        )
    if decisions.duplicated(["symbol", "trade_date"]).any():
        raise ExternalArchiveRepairError("decision_key_duplicate")
    if set(decisions["decision"].unique()) != {"accepted", "rejected", "no_file"}:
        raise ExternalArchiveRepairError("decision_state_contract")
    path = _runtime(workspace) / "inventory" / "repair_decisions.parquet"
    _atomic_frame_parquet(decisions, path)
    return path


def _sql_paths(paths: Sequence[Path]) -> str:
    return ",".join(
        f"'{path.resolve().as_posix().replace(chr(39), chr(39) * 2)}'" for path in paths
    )


def _bundle_accepted(workspace: Path, results: Sequence[SymbolResult]) -> list[Path]:
    accepted_paths = [
        Path(item.accepted_path) for item in results if item.accepted_path
    ]
    if not accepted_paths:
        raise ExternalArchiveRepairError("archive_repair_no_accepted_rows")
    scan = (
        f"read_parquet([{_sql_paths(accepted_paths)}], union_by_name=false, "
        "hive_partitioning=false)"
    )
    output: list[Path] = []
    connection = duckdb.connect()
    connection.execute("SET threads=4")
    spill = _runtime(workspace) / "bundle_spill"
    spill.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"SET temp_directory='{spill.resolve().as_posix().replace(chr(39), chr(39) * 2)}'"
    )
    try:
        years = [
            int(row[0])
            for row in connection.execute(
                f"SELECT DISTINCT substr(trade_date,1,4) FROM {scan} ORDER BY 1"
            ).fetchall()
        ]
        for year in years:
            path = (
                _runtime(workspace)
                / "prepared"
                / INTRADAY_DOMAIN
                / f"year={year}"
                / "part-0000.parquet"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".partial")
            temporary.unlink(missing_ok=True)
            quoted = temporary.resolve().as_posix().replace("'", "''")
            connection.execute(
                f"""
                COPY (
                  SELECT {",".join(INTRADAY_COLUMNS)} FROM {scan}
                  WHERE trade_date BETWEEN '{year}-01-01' AND '{year}-12-31'
                  ORDER BY trade_date,symbol,bar_time
                ) TO '{quoted}' (
                  FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000
                )
                """
            )
            os.replace(temporary, path)
            output.append(path)
    finally:
        connection.close()
        shutil.rmtree(spill, ignore_errors=True)
    return output


def _validate_prepared(
    workspace: Path,
    *,
    inventory: pd.DataFrame,
    decisions_path: Path,
    bundle_paths: Sequence[Path],
    input_manifest: DatasetManifest,
) -> dict[str, Any]:
    root = qdp_v2_root(workspace)
    input_paths = [
        resolve_manifest_path(item.path, root=root) for item in input_manifest.shards
    ]
    decisions_scan = (
        f"read_parquet('{decisions_path.resolve().as_posix()}', "
        "hive_partitioning=false)"
    )
    bundles_scan = (
        f"read_parquet([{_sql_paths(bundle_paths)}], union_by_name=false, "
        "hive_partitioning=false)"
    )
    input_scan = (
        f"read_parquet([{_sql_paths(input_paths)}], union_by_name=true, "
        "hive_partitioning=false)"
    )
    connection = duckdb.connect()
    connection.execute("SET threads=4")
    try:
        decision_stats = connection.execute(
            f"""
            SELECT count(*) AS total,
              count(*) FILTER(WHERE decision='accepted') AS accepted,
              count(*) FILTER(WHERE decision='rejected') AS rejected,
              count(*) FILTER(WHERE decision='no_file') AS no_file,
              count(*) FILTER(WHERE decision='accepted'
                AND quality_liquidity_keep AND trade_date>='2011-01-01') AS formal_quality_accepted,
              count(*) FILTER(WHERE decision!='accepted'
                AND quality_liquidity_keep AND trade_date>='2011-01-01') AS formal_quality_remaining,
              count(*) FILTER(WHERE decision!='accepted'
                AND quality_liquidity_keep AND trade_date BETWEEN '2023-01-01' AND '2025-12-31')
                AS recent_quality_remaining,
              count(*) FILTER(WHERE trade_date>'{END_DATE}') AS forbidden_2026
            FROM {decisions_scan}
            """
        ).fetchone()
        bundle_stats = connection.execute(
            f"""
            SELECT count(*) AS rows,
              count(*)-count(DISTINCT symbol||'|'||trade_date||'|'||bar_time) AS duplicate_rows,
              count(DISTINCT symbol||'|'||trade_date) AS days,
              count(*) FILTER(WHERE trade_date>'{END_DATE}') AS forbidden_2026,
              count(*) FILTER(WHERE bar_time NOT IN ({",".join(repr(x) for x in EXPECTED_BAR_TIMES)}))
                AS invalid_bar_time
            FROM {bundles_scan}
            """
        ).fetchone()
        invalid_day_count = int(
            connection.execute(
                f"""
                SELECT count(*) FROM (
                  SELECT symbol,trade_date,count(*) AS bars,count(DISTINCT bar_time) AS clocks
                  FROM {bundles_scan} GROUP BY symbol,trade_date
                  HAVING bars<>48 OR clocks<>48
                )
                """
            ).fetchone()[0]
        )
        overlap_count = int(
            connection.execute(
                f"""
                SELECT count(*) FROM {bundles_scan} n JOIN (
                  SELECT * FROM {input_scan} WHERE trade_date<='{END_DATE}'
                ) o USING(symbol,trade_date,bar_time)
                """
            ).fetchone()[0]
        )
    finally:
        connection.close()
    formal_quality_total = int(
        inventory.loc[
            inventory["quality_liquidity_keep"]
            & inventory["trade_date"].ge("2011-01-01")
        ].shape[0]
    )
    checks = {
        "decision_inventory_exhaustive": int(decision_stats[0]) == len(inventory),
        "decision_states_mutually_exclusive": sum(map(int, decision_stats[1:4]))
        == len(inventory),
        "accepted_day_count_matches_audit": int(decision_stats[1])
        == EXPECTED_ACCEPTED_DAYS,
        "formal_quality_accepted_matches_audit": int(decision_stats[4])
        == EXPECTED_FORMAL_QUALITY_ACCEPTED_DAYS,
        "formal_quality_remaining_matches_audit": int(decision_stats[5])
        == EXPECTED_FORMAL_QUALITY_REMAINING_DAYS,
        "recent_quality_remaining_matches_audit": int(decision_stats[6])
        == EXPECTED_RECENT_QUALITY_REMAINING_DAYS,
        "decision_2026_rows_zero": int(decision_stats[7]) == 0,
        "accepted_rows_equal_48_per_day": int(bundle_stats[0])
        == int(decision_stats[1]) * 48,
        "accepted_primary_key_unique": int(bundle_stats[1]) == 0,
        "accepted_distinct_days_match": int(bundle_stats[2]) == int(decision_stats[1]),
        "accepted_2026_rows_zero": int(bundle_stats[3]) == 0,
        "accepted_bar_times_valid": int(bundle_stats[4]) == 0,
        "accepted_day_contract_valid": invalid_day_count == 0,
        "existing_intraday_keys_not_overwritten": overlap_count == 0,
        "baseline_formal_quality_gap_count": formal_quality_total == 35_602,
        "formal_complete_pool_expected_rows": EXPECTED_FORMAL_QUALITY_POOL_ROWS
        - int(decision_stats[5])
        == 4_476_845,
        "request_2026_count_zero": int(
            _read_state(workspace).get("request_2026_count", -1)
        )
        == 0,
    }
    if not all(checks.values()):
        raise ExternalArchiveRepairError(f"external_archive_prepared_contract:{checks}")
    reason_counts = {
        str(reason): int(count)
        for reason, count in pd.read_parquet(
            decisions_path, columns=["decision", "rejection_reason"]
        )
        .groupby(["decision", "rejection_reason"], dropna=False)
        .size()
        .items()
    }
    return {
        "checks": checks,
        "decision_counts": {
            "accepted": int(decision_stats[1]),
            "rejected": int(decision_stats[2]),
            "no_file": int(decision_stats[3]),
        },
        "accepted_row_count": int(bundle_stats[0]),
        "formal_quality_accepted_day_count": int(decision_stats[4]),
        "formal_quality_remaining_day_count": int(decision_stats[5]),
        "recent_quality_remaining_day_count": int(decision_stats[6]),
        "formal_quality_complete_pool_row_count": EXPECTED_FORMAL_QUALITY_POOL_ROWS
        - int(decision_stats[5]),
        "rejection_reason_counts": reason_counts,
    }


def _dataset_id(input_dataset_id: str, bundle_paths: Sequence[Path]) -> str:
    digest = hashlib.sha256(input_dataset_id.encode("utf-8"))
    for path in bundle_paths:
        digest.update(_sha256(path).encode("ascii"))
    return f"{INTRADAY_DOMAIN}__{digest.hexdigest()[:24]}"


def _entry_for_installed(
    path: Path, *, root: Path, source_hash: str
) -> ShardManifestEntry:
    parquet = pq.ParquetFile(path)
    connection = duckdb.connect()
    try:
        start_date, end_date = connection.execute(
            "SELECT min(trade_date),max(trade_date) FROM read_parquet(?,hive_partitioning=false)",
            [str(path)],
        ).fetchone()
    finally:
        connection.close()
    return ShardManifestEntry(
        path=path_for_manifest(path, root=root),
        row_count=int(parquet.metadata.num_rows),
        start_date=str(start_date),
        end_date=str(end_date),
        status="stored",
        file_size=path.stat().st_size,
        metadata={
            "source": SOURCE_NAME,
            "source_sha256": source_hash,
            "created_at": utc_now(),
        },
    )


def _install_immutable_version(
    workspace: Path,
    *,
    input_manifest: DatasetManifest,
    bundle_paths: Sequence[Path],
    validation: Mapping[str, Any],
) -> tuple[str, list[Path]]:
    root = qdp_v2_root(workspace)
    dataset_id = _dataset_id(input_manifest.dataset_id, bundle_paths)
    target_root = root / "datasets" / INTRADAY_DOMAIN / dataset_id / "shards"
    installed: list[Path] = []
    entries: list[ShardManifestEntry] = []
    for source_path in bundle_paths:
        year = source_path.parent.name
        target = target_root / "external_archive_v2" / year / "part-0000.parquet"
        source_hash = _sha256(source_path)
        if target.is_file():
            if _sha256(target) != source_hash:
                raise ExternalArchiveRepairError(
                    f"installed_bundle_hash_conflict:{target}"
                )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".partial")
            temporary.unlink(missing_ok=True)
            shutil.copy2(source_path, temporary)
            if _sha256(temporary) != source_hash:
                temporary.unlink(missing_ok=True)
                raise ExternalArchiveRepairError(
                    f"installed_bundle_copy_failed:{target}"
                )
            os.replace(temporary, target)
        installed.append(target)
        entries.append(_entry_for_installed(target, root=root, source_hash=source_hash))
    new_manifest = replace(
        input_manifest,
        dataset_id=dataset_id,
        row_count=input_manifest.row_count + sum(item.row_count for item in entries),
        shards=[*input_manifest.shards, *entries],
        source={
            **input_manifest.source,
            "historical_external_archive_repair": REPAIR_ID,
            "historical_external_archive_provider": SOURCE_NAME,
            "historical_external_archive_checked_through": END_DATE,
            "historical_external_archive_eligibility_filter": False,
            "historical_external_archive_request_2026_count": 0,
        },
        quality={
            **input_manifest.quality,
            "historical_external_archive_candidate_days": EXPECTED_CANDIDATE_DAYS,
            "historical_external_archive_accepted_days": int(
                validation["decision_counts"]["accepted"]
            ),
            "historical_external_archive_rejected_days": int(
                validation["decision_counts"]["rejected"]
            ),
            "historical_external_archive_no_file_days": int(
                validation["decision_counts"]["no_file"]
            ),
            "quality_liquidity_complete_pit_rows_2011_2025": int(
                validation["formal_quality_complete_pool_row_count"]
            ),
            "external_archive_existing_keys_overwritten": 0,
            "external_archive_2026_rows_written": 0,
        },
        created_at=utc_now(),
        notes=[
            *input_manifest.notes,
            "External archive is used only to fill the frozen missing stock-day inventory.",
            "A 09:30 auction row is merged into 09:35 only for the exact canonical 49-row pattern.",
            "Daily OHLC must pass 2%; volume or amount must pass 5%; no bars are interpolated.",
            "Archive file presence never changes PIT universe membership.",
        ],
    )
    _assert_credential_free(new_manifest.to_dict())
    write_dataset_manifest(root, new_manifest)
    reread = read_dataset_manifest(
        dataset_manifest_for_id(root, dataset_id, INTRADAY_DOMAIN)
    )
    if (
        reread.dataset_id != dataset_id
        or reread.row_count != new_manifest.row_count
        or len(reread.shards) != len(new_manifest.shards)
    ):
        raise ExternalArchiveRepairError(
            "immutable_intraday_manifest_validation_failed"
        )
    active = read_active_manifest(root)
    current = active_dataset_map(active)
    if current.get(INTRADAY_DOMAIN) != input_manifest.dataset_id:
        if current.get(INTRADAY_DOMAIN) == dataset_id:
            return dataset_id, installed
        raise ExternalArchiveRepairError("active_intraday_drifted_before_commit")
    active["datasets"] = {**current, INTRADAY_DOMAIN: dataset_id}
    active["updated_at"] = utc_now()
    write_active_manifest(root, active)
    if (
        active_dataset_map(read_active_manifest(root)).get(INTRADAY_DOMAIN)
        != dataset_id
    ):
        raise ExternalArchiveRepairError("active_intraday_atomic_switch_failed")
    return dataset_id, installed


def run_pending(
    *,
    workspace_root: str | Path | None = None,
    archive_path: str | Path = DEFAULT_ARCHIVE,
    max_workers: int = DEFAULT_WORKERS,
    seal_runtime: bool = True,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    if state.get("status") == "applied":
        return state
    archive = Path(archive_path).resolve()
    inventory, members, input_manifest, state = _initialize(workspace, archive)
    candidate_symbols = sorted(set(inventory["symbol"]).intersection(members))
    grouped = {
        str(symbol): frame.copy()
        for symbol, frame in inventory.loc[
            inventory["symbol"].isin(candidate_symbols)
        ].groupby("symbol", sort=True)
    }
    workers = max(1, min(DEFAULT_WORKERS, int(max_workers)))
    results: dict[str, dict[str, Any]] = dict(state.get("symbols", {}) or {})
    failures: dict[str, str] = {}
    futures: dict[Future[SymbolResult], str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for symbol in candidate_symbols:
            futures[
                pool.submit(
                    _process_symbol,
                    workspace,
                    archive_path=archive,
                    member=members[symbol],
                    reference=grouped[symbol],
                )
            ] = symbol
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                results[symbol] = future.result().to_dict()
            except Exception as exc:  # noqa: BLE001 - preserve task error in ledger
                failures[symbol] = f"{type(exc).__name__}:{str(exc)[:500]}"
            if completed % 5 == 0 or completed == len(futures):
                state.update(
                    {
                        "status": "processing",
                        "maximum_workers": workers,
                        "completed_symbol_count": len(results),
                        "failed_symbols": failures,
                        "symbols": results,
                    }
                )
                _write_state(workspace, state)
    if failures:
        state.update({"status": "failed", "failed_symbols": failures})
        _write_state(workspace, state)
        raise ExternalArchiveRepairError(
            f"external_archive_symbol_failures:{len(failures)}"
        )
    ordered_results = [
        SymbolResult(
            **{**results[symbol], "reused": bool(results[symbol].get("reused", False))}
        )
        for symbol in candidate_symbols
    ]
    decisions_path = _assemble_decisions(workspace, inventory, ordered_results)
    bundle_paths = _bundle_accepted(workspace, ordered_results)
    validation = _validate_prepared(
        workspace,
        inventory=inventory,
        decisions_path=decisions_path,
        bundle_paths=bundle_paths,
        input_manifest=input_manifest,
    )
    state.update(
        {
            "status": "prepared",
            "symbols": results,
            "failed_symbols": {},
            "decision_path": str(decisions_path),
            "decision_sha256": _sha256(decisions_path),
            "prepared_bundle_paths": [str(path) for path in bundle_paths],
            "prepared_bundle_sha256": {
                str(path): _sha256(path) for path in bundle_paths
            },
            **validation,
        }
    )
    _write_state(workspace, state)
    dataset_id, installed = _install_immutable_version(
        workspace,
        input_manifest=input_manifest,
        bundle_paths=bundle_paths,
        validation=validation,
    )
    state.update(
        {
            "status": "applied",
            "installed_dataset_id": dataset_id,
            "installed_bundle_paths": [str(path) for path in installed],
            "output_row_count": input_manifest.row_count
            + int(validation["accepted_row_count"]),
        }
    )
    _write_state(workspace, state)
    audit = {key: value for key, value in state.items() if key not in {"symbols"}} | {
        "symbol_task_count": len(results),
        "reused_symbol_task_count": sum(
            bool(item.get("reused", False)) for item in results.values()
        ),
    }
    atomic_write_json(qdp_v2_root(workspace) / "audits" / f"{REPAIR_ID}.json", audit)
    if seal_runtime:
        from quantlab.data.qdp_v2.runtime_archive import (
            seal_completed_workflow,
        )

        return {
            **state,
            "runtime_archive": seal_completed_workflow(
                REPAIR_ID,
                workspace_root=workspace,
            ),
        }
    return state


def evaluate(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    active = active_dataset_map(read_active_manifest(qdp_v2_root(workspace)))
    checks = {
        **dict(state.get("checks", {}) or {}),
        "repair_applied": state.get("status") == "applied",
        "immutable_dataset_active": active.get(INTRADAY_DOMAIN)
        == state.get("installed_dataset_id"),
        "request_2026_count_zero": int(state.get("request_2026_count", -1)) == 0,
        "write_2026_count_zero": int(state.get("write_2026_count", -1)) == 0,
    }
    return {
        "status": "ok" if checks and all(checks.values()) else "error",
        "repair_id": REPAIR_ID,
        "checks": checks,
        "installed_dataset_id": state.get("installed_dataset_id", ""),
        "input_intraday_dataset_id": state.get("input_intraday_dataset_id", ""),
        "decision_counts": state.get("decision_counts", {}),
        "accepted_row_count": state.get("accepted_row_count", 0),
        "formal_quality_complete_pool_row_count": state.get(
            "formal_quality_complete_pool_row_count", 0
        ),
        "rejection_reason_counts": state.get("rejection_reason_counts", {}),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp external-archive-5m-repair")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--archive-path", default=str(DEFAULT_ARCHIVE))
    parser.add_argument("--max-workers", type=int, default=DEFAULT_WORKERS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-pending", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--status", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = str(args.workspace_root or "") or None
    if args.run_pending:
        payload = run_pending(
            workspace_root=workspace,
            archive_path=args.archive_path,
            max_workers=args.max_workers,
        )
    elif args.evaluate:
        payload = evaluate(workspace_root=workspace)
    else:
        payload = _read_state(_workspace(workspace))
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    return 0 if payload.get("status") not in {"error", "failed"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
