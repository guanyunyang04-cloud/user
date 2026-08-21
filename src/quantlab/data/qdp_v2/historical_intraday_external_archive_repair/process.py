"""Historical Intraday External Archive Repair: process responsibilities."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from quantlab.data.qdp_v2.manifest import (
    EXPECTED_BAR_TIMES,
    atomic_write_json,
    utc_now,
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

from .config import (
    _DECISION_COLUMNS,
    _NUMERIC_COLUMNS,
    SOURCE_NAME,
    ArchiveMember,
    ExternalArchiveRepairError,
    SymbolResult,
)
from .context import (
    _runtime,
)
from .normalize import (
    _normalize_day,
    _validate_daily,
)


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
            or int(pq.ParquetFile(accepted_path).metadata.num_rows) != accepted_count * len(EXPECTED_BAR_TIMES)
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
            raise ExternalArchiveRepairError(f"archive_member_unreadable:{member.symbol}:{member.member_name}") from exc
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
                        raise ExternalArchiveRepairError(f"archive_member_schema:{member.symbol}")
                    chunk.columns = ["timestamp", *_NUMERIC_COLUMNS]
                    timestamp = chunk["timestamp"].astype(str)
                    mask = timestamp.str[:10].isin(wanted_dates)
                    if not mask.any():
                        continue
                    selected = chunk.loc[mask].copy()
                    selected["trade_date"] = timestamp.loc[mask].str[:10].to_numpy()
                    selected["bar_time"] = timestamp.loc[mask].str[11:16].to_numpy()
                    for column in _NUMERIC_COLUMNS:
                        selected[column] = pd.to_numeric(selected[column], errors="coerce").astype("float64")
                    frames.append(selected.loc[:, ["trade_date", "bar_time", *_NUMERIC_COLUMNS]])
            except UnicodeDecodeError as exc:
                raise ExternalArchiveRepairError(f"archive_member_encoding:{member.symbol}") from exc
    if not frames:
        return pd.DataFrame(columns=["trade_date", "bar_time", *_NUMERIC_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def _atomic_frame_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _evaluate_symbol_days(
    symbol: str,
    *,
    member: ArchiveMember,
    reference: pd.DataFrame,
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = {str(trade_date): day.copy() for trade_date, day in raw.groupby("trade_date", sort=False)}
    accepted: list[pd.DataFrame] = []
    decisions: list[dict[str, Any]] = []
    for row in reference.sort_values("trade_date", kind="stable").to_dict("records"):
        trade_date = str(row["trade_date"])
        day = grouped.get(trade_date)
        source_bar_count = len(day) if day is not None else 0
        normalized = pd.DataFrame()
        reason = "source_day_absent"
        if day is not None:
            normalized, reason = _normalize_day(day)
        price_error = volume_error = amount_error = float("nan")
        if not reason:
            valid, reason, price_error, volume_error, amount_error = _validate_daily(normalized, row)
            assert valid == (not reason)
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
                "decision": "rejected" if reason else "accepted",
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
    accepted_frame = pd.concat(accepted, ignore_index=True) if accepted else pd.DataFrame(columns=INTRADAY_COLUMNS)
    return accepted_frame, pd.DataFrame(decisions, columns=_DECISION_COLUMNS)


def _write_symbol_result_frames(
    symbol: str,
    *,
    accepted_frame: pd.DataFrame,
    decision_frame: pd.DataFrame,
    accepted_path: Path,
    decision_path: Path,
) -> tuple[str, str, int]:
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
    accepted_days = int((decision_frame["decision"] == "accepted").sum())
    return accepted_hash, _sha256(decision_path), accepted_days


def _write_symbol_receipt(
    receipt_path: Path,
    *,
    archive_path: Path,
    member: ArchiveMember,
    requested_day_count: int,
    accepted_days: int,
    accepted_hash: str,
    decision_hash: str,
) -> None:
    receipt = {
        "status": "completed",
        "symbol": member.symbol,
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
    accepted_frame, decision_frame = _evaluate_symbol_days(
        symbol,
        member=member,
        reference=reference,
        raw=raw,
    )
    if len(decision_frame) != requested_day_count:
        raise ExternalArchiveRepairError(f"symbol_decision_count:{symbol}")
    accepted_hash, decision_hash, accepted_days = _write_symbol_result_frames(
        symbol,
        accepted_frame=accepted_frame,
        decision_frame=decision_frame,
        accepted_path=accepted_path,
        decision_path=decision_path,
    )
    _write_symbol_receipt(
        receipt_path,
        archive_path=archive_path,
        member=member,
        requested_day_count=requested_day_count,
        accepted_days=accepted_days,
        accepted_hash=accepted_hash,
        decision_hash=decision_hash,
    )
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
