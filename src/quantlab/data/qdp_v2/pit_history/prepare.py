"""PIT history prepare operations."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.data.qdp_v2.active import resolve_active_domain
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb
from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    atomic_write_json,
    path_for_manifest,
    utc_now,
    write_dataset_manifest,
)

from .config import (
    PART_SEMANTIC_VERSION,
    RESTORE_DOMAINS,
    _factor_rows,
    _historical_names,
    _identity_exchange,
    _security_id,
    _short_exchange,
)
from .context import (
    PitHistoryContext,
    PitHistoryError,
    _atomic_parquet,
)
from .download import (
    _historical_st_status,
)


@dataclass
class _PreparedHistory:
    frame: pd.DataFrame
    suspended_input: pd.Series
    observed_bar: pd.Series
    names: pd.Series
    is_st: pd.Series
    is_suspended: pd.Series
    is_delisted: pd.Series


@dataclass
class _PreparedShares:
    total_share: pd.Series
    float_share: pd.Series
    restricted_share: pd.Series
    total_source_date: pd.Series
    float_source_date: pd.Series
    turnover: pd.Series
    total_valid: pd.Series
    event_float_valid: pd.Series


def _symbol_part_is_current(done: Path) -> bool:
    if not done.is_file():
        return False
    try:
        completed = json.loads(done.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return str(completed.get("semantic_version", "")) == PART_SEMANTIC_VERSION


def _load_symbol_inputs(
    ctx: PitHistoryContext,
    token: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(ctx.runtime / "history_parts" / f"{token}.parquet"),
        pd.read_parquet(ctx.runtime / "factor_parts" / f"{token}.parquet"),
        pd.read_parquet(ctx.runtime / "share_event_parts" / f"{token}.parquet"),
        pd.read_parquet(ctx.runtime / "name_interval_parts" / f"{token}.parquet"),
    )


def _prepare_history(
    history: pd.DataFrame,
    *,
    symbol: str,
    row: Mapping[str, Any],
    intervals: pd.DataFrame,
) -> _PreparedHistory:
    history = history.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)
    if "history_source" not in history:
        history["history_source"] = "legacy_baostock_history_fixture"
    for column in (
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "turn",
        "peTTM",
        "pbMRQ",
        "psTTM",
        "pcfNcfTTM",
    ):
        history[column] = pd.to_numeric(history[column], errors="coerce")
    trade_status = history["tradestatus"].fillna("").astype(str).str.strip()
    is_suspended = trade_status.map({"0": True, "1": False}).astype("boolean")
    suspended_input = is_suspended.fillna(False).astype(bool)
    for column in ("volume", "amount", "turn"):
        history.loc[suspended_input & history[column].isna(), column] = 0.0
    history["trade_date"] = pd.to_datetime(history["trade_date"], errors="raise").dt.strftime("%Y-%m-%d")
    history["symbol"] = symbol
    values = history.loc[:, ["open", "high", "low", "close", "volume", "amount"]]
    valid_bar = (
        values.notna().all(axis=1)
        & values[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & values[["volume", "amount"]].ge(0).all(axis=1)
        & history["high"].ge(history[["open", "low", "close"]].max(axis=1))
        & history["low"].le(history[["open", "high", "close"]].min(axis=1))
    )
    invalid_observed = (~suspended_input) & (~valid_bar)
    if bool(invalid_observed.any()):
        raise PitHistoryError(f"pit_history_invalid_bar:{symbol}:{int(invalid_observed.sum())}")
    names = _historical_names(history["trade_date"], intervals, fallback=str(row.get("name", "")))
    archive_names = (
        history.get("name_on_date", pd.Series("", index=history.index, dtype="object"))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    names = archive_names.where(archive_names.ne(""), names)
    delist_date = str(row.get("delist_date", "") or "")
    is_delisted = history["trade_date"].ge(delist_date) if delist_date else pd.Series(False, index=history.index)
    is_st = _historical_st_status(history, symbol=symbol, names=names)
    observed_bar = is_suspended.eq(False).fillna(valid_bar).astype(bool) & valid_bar
    return _PreparedHistory(
        frame=history,
        suspended_input=suspended_input,
        observed_bar=observed_bar,
        names=names,
        is_st=is_st,
        is_suspended=is_suspended,
        is_delisted=is_delisted,
    )


def _market_domain_frames(
    prepared: _PreparedHistory,
    *,
    symbol: str,
    row: Mapping[str, Any],
    factors: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    history = prepared.frame
    delist_date = str(row.get("delist_date", "") or "")
    daily = (
        pd.DataFrame(
            {
                "symbol": symbol,
                "semantic_version": PART_SEMANTIC_VERSION,
                "trade_date": history["trade_date"],
                "open": history["open"].astype("float64"),
                "high": history["high"].astype("float64"),
                "low": history["low"].astype("float64"),
                "close": history["close"].astype("float64"),
                "volume": history["volume"].astype("float64"),
                "amount": history["amount"].astype("float64"),
                "source": history["history_source"].astype(str),
                "adjusted_flag": "none",
            }
        )
        .loc[prepared.observed_bar]
        .reset_index(drop=True)
    )
    universe = pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": history["trade_date"],
            "name": prepared.names,
            "exchange": _short_exchange(symbol),
            "board": "main",
            "list_status": np.where(prepared.is_delisted, "D", "L"),
            "list_date": str(row.get("list_date", "") or ""),
            "delist_date": delist_date,
            "source": "protected_archive+exchange_name_history+akshare_pit_restore",
        }
    )
    known_st = prepared.is_st.fillna(False).astype(bool)
    known_suspended = prepared.is_suspended.fillna(False).astype(bool)
    unknown_status = prepared.is_st.isna() | prepared.is_suspended.isna()
    reasons = np.select(
        (
            prepared.is_delisted,
            unknown_status & known_st,
            unknown_status & known_suspended,
            unknown_status,
            known_st & known_suspended,
            known_st,
            known_suspended,
        ),
        (
            "delisted",
            "st;status_unknown",
            "suspended;status_unknown",
            "status_unknown",
            "st;suspended",
            "st",
            "suspended",
        ),
        default="",
    )
    status = pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": history["trade_date"],
            "is_st": prepared.is_st.astype("boolean"),
            "is_suspended": prepared.is_suspended.astype("boolean"),
            "is_delisted": prepared.is_delisted.astype(bool),
            "status_reason": reasons,
            "source": "protected_archive+exchange_status_evidence+akshare_pit_restore",
        }
    )
    return {
        "market_daily_raw": daily,
        "universe_snapshot": universe,
        "security_status": status,
        "adjust_factor": _factor_rows(history, factors).loc[prepared.observed_bar].reset_index(drop=True),
    }


def _event_share_series(
    history: pd.DataFrame,
    share_events: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    event = share_events.copy()
    if not event.empty:
        event["source_date"] = pd.to_datetime(event["source_date"], errors="coerce")
        event["total_share"] = pd.to_numeric(event["total_share"], errors="coerce")
        event["float_share"] = pd.to_numeric(event["float_share"], errors="coerce")
        event = (
            event.dropna(subset=["source_date"]).sort_values("source_date").drop_duplicates("source_date", keep="last")
        )
    if event.empty:
        index = history.index
        return (
            pd.Series(np.nan, index=index, dtype="float64"),
            pd.Series(np.nan, index=index, dtype="float64"),
            pd.Series("", index=index, dtype="object"),
        )
    history_dates = pd.to_datetime(history["trade_date"], errors="raise")
    positions = (
        np.searchsorted(
            event["source_date"].to_numpy(dtype="datetime64[ns]"),
            history_dates.to_numpy(dtype="datetime64[ns]"),
            side="right",
        )
        - 1
    )
    valid_event = positions >= 0
    safe_position = np.maximum(positions, 0)
    total_values = event["total_share"].to_numpy(dtype=np.float64)[safe_position]
    float_values = event["float_share"].to_numpy(dtype=np.float64)[safe_position]
    source_values = event["source_date"].dt.strftime("%Y-%m-%d").to_numpy()[safe_position]
    total_values[~valid_event] = np.nan
    float_values[~valid_event] = np.nan
    source_values = np.where(valid_event, source_values, "")
    return (
        pd.Series(total_values, index=history.index, dtype="float64"),
        pd.Series(float_values, index=history.index, dtype="float64"),
        pd.Series(source_values, index=history.index, dtype="object"),
    )


def _prepare_share_series(prepared: _PreparedHistory, share_events: pd.DataFrame) -> _PreparedShares:
    history = prepared.frame
    total_share, event_float, event_source = _event_share_series(history, share_events)
    turnover = history["turn"].copy()
    turnover.loc[prepared.suspended_input & turnover.isna()] = 0.0
    known_float_from_turnover = (
        history["volume"].astype("float64").mul(100.0).div(turnover.astype("float64").where(turnover.gt(0.0)))
    )
    past_float_for_turnover = event_float.where(
        np.isfinite(event_float) & event_float.gt(0.0), known_float_from_turnover
    ).ffill()
    implied_turnover = (
        history["volume"]
        .astype("float64")
        .mul(100.0)
        .div(past_float_for_turnover.where(past_float_for_turnover.gt(0.0)))
    )
    turnover = turnover.fillna(implied_turnover)
    if turnover.isna().any():
        symbol = str(history["symbol"].iloc[0])
        raise PitHistoryError(f"pit_history_turnover_missing:{symbol}:{int(turnover.isna().sum())}")
    inferred_float_direct = (
        history["volume"].astype("float64").mul(100.0).div(turnover.astype("float64").where(turnover.gt(0.0)))
    )
    inferred_float_direct = inferred_float_direct.where(
        np.isfinite(inferred_float_direct) & inferred_float_direct.gt(0.0)
    )
    inferred_source_date = (
        pd.Series(
            np.where(np.isfinite(inferred_float_direct), history["trade_date"], None),
            index=history.index,
            dtype="object",
        )
        .ffill()
        .fillna("")
    )
    event_float_valid = np.isfinite(event_float) & event_float.gt(0.0)
    float_share = event_float.where(event_float_valid, inferred_float_direct.ffill()).astype("float64")
    float_source_date = event_source.where(event_float_valid, inferred_source_date)
    total_valid = np.isfinite(total_share) & total_share.gt(0.0)
    total_source_date = event_source.where(total_valid, "")
    restricted = pd.Series(
        np.where(
            total_valid & np.isfinite(float_share),
            np.maximum(total_share - float_share, 0.0),
            np.nan,
        ),
        index=history.index,
        dtype="float64",
    )
    return _PreparedShares(
        total_share=total_share,
        float_share=float_share,
        restricted_share=restricted,
        total_source_date=total_source_date,
        float_source_date=float_source_date,
        turnover=turnover,
        total_valid=total_valid,
        event_float_valid=event_float_valid,
    )


def _share_and_valuation_frames(
    prepared: _PreparedHistory,
    shares: _PreparedShares,
    *,
    symbol: str,
) -> dict[str, pd.DataFrame]:
    history = prepared.frame
    share_capital = (
        pd.DataFrame(
            {
                "symbol": symbol,
                "trade_date": history["trade_date"],
                "total_share": shares.total_share,
                "float_share": shares.float_share,
                "restricted_share": shares.restricted_share,
                "total_share_source_date": shares.total_source_date.astype(str),
                "float_share_source_date": shares.float_source_date.astype(str),
                "restricted_share_source_date": shares.total_source_date.astype(str),
                "share_fill_method": np.where(
                    shares.total_valid & shares.event_float_valid,
                    "past_only_cninfo_event",
                    "same_day_turnover_float_inference",
                ),
                "source": "cninfo_event+archive_or_eastmoney_turnover_pit_restore",
            }
        )
        .loc[prepared.observed_bar]
        .reset_index(drop=True)
    )
    valuation = (
        pd.DataFrame(
            {
                "symbol": symbol,
                "trade_date": history["trade_date"],
                "total_mv": history["close"].to_numpy(dtype=np.float64) * shares.total_share.to_numpy(dtype=np.float64),
                "circ_mv": history["close"].to_numpy(dtype=np.float64) * shares.float_share.to_numpy(dtype=np.float64),
                "pe": history["peTTM"].astype("float64"),
                "pb": history["pbMRQ"].astype("float64"),
                "turnover_rate": shares.turnover.astype("float64"),
                "source": "archive_or_eastmoney+cninfo_or_turnover_share_pit_restore",
            }
        )
        .loc[prepared.observed_bar]
        .reset_index(drop=True)
    )
    return {"share_capital": share_capital, "valuation": valuation}


def _identity_frames(
    *,
    symbol: str,
    row: Mapping[str, Any],
    delist_date: str,
    identity_already_known: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    identity = pd.DataFrame(
        {
            "security_id": [_security_id(symbol)],
            "official_org_id": [""],
            "issuer_name": [str(row.get("name", ""))],
            "exchange": [_identity_exchange(symbol)],
            "list_date": [str(row.get("list_date", ""))],
            "current_symbol": [symbol],
            "identity_source": ["protected_archive_security_master+pit_history_restore"],
        }
    )
    symbol_history = pd.DataFrame(
        {
            "security_id": [_security_id(symbol)],
            "symbol": [symbol],
            "effective_from": [str(row.get("list_date", ""))],
            "effective_to": [delist_date or "9999-12-31"],
            "name_on_date": [str(row.get("name", ""))],
            "board_on_date": ["MainBoard"],
            "evidence_source": ["protected_archive_security_master+pit_history_restore"],
            "official_document_hash": [""],
        }
    )
    if identity_already_known:
        identity = identity.iloc[0:0].copy()
        symbol_history = symbol_history.iloc[0:0].copy()
    return identity, symbol_history


def _name_event_frame(
    intervals: pd.DataFrame,
    *,
    symbol: str,
    start_date: str,
) -> pd.DataFrame:
    columns = ["symbol", "trade_date", "old_name", "new_name", "change_type", "source"]
    if intervals.empty:
        return pd.DataFrame(columns=columns)
    ordered = intervals.sort_values("start_date").copy()
    ordered["old_name"] = ordered["name"].shift(1)
    ordered = ordered.loc[
        ordered["old_name"].notna() & ordered["old_name"].ne(ordered["name"]) & ordered["start_date"].ge(start_date)
    ]
    return pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": ordered["start_date"].astype(str),
            "old_name": ordered["old_name"].astype(str),
            "new_name": ordered["name"].astype(str),
            "change_type": "short_name",
            "source": "exchange_name_intervals+pit_history_restore",
        }
    )


def _write_symbol_domain_parts(
    ctx: PitHistoryContext,
    *,
    symbol: str,
    token: str,
    domain_frames: Mapping[str, pd.DataFrame],
    done: Path,
) -> None:
    output = ctx.runtime / "domain_parts"
    for domain, frame in domain_frames.items():
        _atomic_parquet(frame, output / domain / f"{token}.parquet")
    done.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        done,
        {
            "symbol": symbol,
            "semantic_version": PART_SEMANTIC_VERSION,
            "row_counts": {key: int(len(value)) for key, value in domain_frames.items()},
            "completed_at": utc_now(),
        },
    )


def _prepare_symbol_parts(
    ctx: PitHistoryContext,
    *,
    row: Mapping[str, Any],
    identity_already_known: bool = False,
) -> None:
    symbol = str(row["symbol"])
    token = symbol.replace(".", "_")
    done = ctx.runtime / "domain_parts" / "done" / f"{token}.json"
    if _symbol_part_is_current(done):
        return
    history, factors, share_events, intervals = _load_symbol_inputs(ctx, token)
    prepared = _prepare_history(history, symbol=symbol, row=row, intervals=intervals)
    domain_frames = _market_domain_frames(prepared, symbol=symbol, row=row, factors=factors)
    daily = domain_frames["market_daily_raw"]
    domain_frames["industry_concept"] = pd.DataFrame(
        {
            "symbol": daily["symbol"],
            "trade_date": daily["trade_date"],
            "industry": "Unknown",
            "source": "pit_history_restore_industry_unavailable",
            "original_industry": "",
            "original_source": "",
            "industry_fill_method": "unavailable",
            "industry_source_date": daily["trade_date"],
            "industry_standard": "Unclassified",
        }
    )
    shares = _prepare_share_series(prepared, share_events)
    domain_frames.update(_share_and_valuation_frames(prepared, shares, symbol=symbol))
    delist_date = str(row.get("delist_date", "") or "")
    identity, symbol_history = _identity_frames(
        symbol=symbol,
        row=row,
        delist_date=delist_date,
        identity_already_known=identity_already_known,
    )
    domain_frames.update(
        {
            "security_identity": identity,
            "symbol_history": symbol_history,
            "name_change": _name_event_frame(intervals, symbol=symbol, start_date=ctx.start_date),
        }
    )
    ordered = {domain: domain_frames[domain] for domain in RESTORE_DOMAINS}
    _write_symbol_domain_parts(
        ctx,
        symbol=symbol,
        token=token,
        domain_frames=ordered,
        done=done,
    )


def _combine_domain_parts(ctx: PitHistoryContext, domain: str) -> Path:
    parts = sorted((ctx.runtime / "domain_parts" / domain).glob("*.parquet"))
    if not parts:
        raise PitHistoryError(f"pit_history_domain_parts_missing:{domain}")
    current = resolve_active_domain(domain, workspace_root=ctx.workspace)
    keys = list(current.manifest.primary_key)
    if not keys:
        raise PitHistoryError(f"pit_history_primary_key_missing:{domain}")
    prepared = ctx.runtime / "prepared" / f"{domain}.parquet"
    prepared.parent.mkdir(parents=True, exist_ok=True)
    quoted = ",".join(f'"{item}"' for item in keys)
    temporary = prepared.with_name(f".{prepared.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with open_guarded_duckdb(
            temp_directory=ctx.runtime / "combine_spill" / domain,
            threads=4,
        ) as con:
            con.execute(
                f"COPY ("
                "WITH incoming AS ("
                " SELECT * FROM read_parquet(?, union_by_name=true)"
                f" QUALIFY row_number() OVER (PARTITION BY {quoted} ORDER BY {quoted})=1"
                "), existing AS ("
                f" SELECT {quoted} FROM read_parquet(?, union_by_name=true)"
                ") SELECT i.* FROM incoming i ANTI JOIN existing e "
                f"USING ({quoted}) ORDER BY {quoted}"
                f") TO '{str(temporary).replace(chr(39), chr(39) * 2)}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)",
                [
                    [str(item) for item in parts],
                    [str(item) for item in current.shard_paths],
                ],
            )
        temporary.replace(prepared)
    finally:
        temporary.unlink(missing_ok=True)
        shutil.rmtree(ctx.runtime / "combine_spill" / domain, ignore_errors=True)
    return prepared


def _copy_restore_shard(*, prepared: Path, shard: Path, expected: pd.DataFrame, domain: str) -> None:
    if shard.is_file():
        return
    shard.parent.mkdir(parents=True, exist_ok=True)
    temporary = shard.with_name(f".{shard.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        shutil.copy2(prepared, temporary)
        copied = pd.read_parquet(temporary)
        if len(copied) != len(expected) or list(copied.columns) != list(expected.columns):
            raise PitHistoryError(f"pit_history_copy_validation_failed:{domain}")
        temporary.replace(shard)
    finally:
        temporary.unlink(missing_ok=True)


def _restore_domain_metadata(
    domain: str,
    frame: pd.DataFrame,
    *,
    source: dict[str, Any],
    quality: dict[str, Any],
) -> None:
    if domain == "industry_concept":
        quality["restored_unclassified_rows"] = int(frame["industry_fill_method"].astype(str).eq("unavailable").sum())
        source["source_contract"] = (
            "existing strict-PIT industry plus explicit Unclassified for restored "
            "historical securities where no dated industry evidence is available"
        )
    elif domain == "share_capital":
        quality["total_share_null_rows"] = int(quality.get("total_share_null_rows", 0) or 0) + int(
            frame["total_share"].isna().sum()
        )
        quality["float_share_null_rows"] = int(quality.get("float_share_null_rows", 0) or 0) + int(
            frame["float_share"].isna().sum()
        )
        quality["restored_float_share_inference"] = "same_day_volume_divided_by_archive_or_eastmoney_turnover_rate"
        source["source_contract"] = (
            "existing strict-PIT shares plus past-only CNInfo events; missing float "
            "shares inferred from same-day volume and turnover without future fill"
        )
    elif domain == "valuation":
        quality["total_mv_null_rows"] = int(quality.get("total_mv_null_rows", 0) or 0) + int(
            frame["total_mv"].isna().sum()
        )
        quality["circ_mv_null_rows"] = int(quality.get("circ_mv_null_rows", 0) or 0) + int(
            frame["circ_mv"].isna().sum()
        )
        source["source_contract"] = (
            "existing strict-PIT valuation plus protected-archive PE/PB/turnover, "
            "Eastmoney boundary turnover, and price-times-past-only-or-turnover-inferred shares"
        )
    elif domain == "name_change":
        quality["restored_name_history_scope"] = (
            "dated Shenzhen exchange short-name history; Shanghai exact-date ST state is "
            "restored separately from official SSE factbooks"
        )


def _composite_manifest(
    *,
    current: Any,
    dataset_id: str,
    entry: ShardManifestEntry,
    domain: str,
    frame: pd.DataFrame,
) -> DatasetManifest:
    payload = current.manifest.to_dict()
    payload["dataset_id"] = dataset_id
    payload["shards"] = [item.to_dict() for item in current.manifest.shards] + [entry.to_dict()]
    payload["row_count"] = int(current.manifest.row_count) + int(len(frame))
    starts = [item.start_date for item in current.manifest.shards if item.start_date]
    ends = [item.end_date for item in current.manifest.shards if item.end_date]
    if entry.start_date:
        starts.append(entry.start_date)
    if entry.end_date:
        ends.append(entry.end_date)
    payload.update(
        {
            "start_date": min(starts) if starts else "",
            "end_date": max(ends) if ends else "",
            "created_at": utc_now(),
            "notes": [
                item
                for item in list(payload.get("notes", []) or [])
                if "permanent_st_delisting_exclusion" not in str(item)
            ]
            + ["composite_restore:pit_historical_mainboard_v2"],
        }
    )
    source = dict(payload.get("source", {}) or {})
    source.update(
        {
            "scope": "point_in_time_historical_mainboard",
            "survivorship_policy": "include_when_listed_then_apply_same_day_status",
            "pit_history_restored_at": utc_now(),
        }
    )
    quality = dict(payload.get("quality", {}) or {})
    quality.update(
        {
            "scope": "point_in_time_historical_mainboard",
            "survivorship_bias_free_mainboard_daily": True,
            "pit_status_is_signal_date_asof": True,
        }
    )
    _restore_domain_metadata(domain, frame, source=source, quality=quality)
    payload.update({"source": source, "quality": quality})
    return DatasetManifest.from_mapping(payload)


def _create_composite_dataset(
    ctx: PitHistoryContext,
    *,
    domain: str,
    prepared: Path,
) -> tuple[str, dict[str, Any]]:
    current = resolve_active_domain(domain, workspace_root=ctx.workspace)
    frame = pd.read_parquet(prepared)
    if frame.empty:
        return current.dataset_id, {"status": "unchanged", "added_rows": 0}
    token = utc_now().replace("-", "").replace(":", "").replace("+00:00", "Z")
    dataset_id = f"{domain}__pit_history_{token}"
    dataset_dir = ctx.root / "datasets" / domain / dataset_id
    shard = dataset_dir / "shards" / f"pit_restore_{token}.parquet"
    _copy_restore_shard(prepared=prepared, shard=shard, expected=frame, domain=domain)
    date_column = "trade_date" if "trade_date" in frame.columns else ""
    start = str(frame[date_column].min()) if date_column and len(frame) else ""
    end = str(frame[date_column].max()) if date_column and len(frame) else ""
    entry = ShardManifestEntry(
        path=path_for_manifest(shard, root=ctx.root),
        row_count=int(len(frame)),
        start_date=start,
        end_date=end,
        status="stored",
        file_size=shard.stat().st_size,
        metadata={
            "restore_policy": "pit_historical_mainboard_v2",
            "source_path": str(prepared),
        },
    )
    manifest = _composite_manifest(
        current=current,
        dataset_id=dataset_id,
        entry=entry,
        domain=domain,
        frame=frame,
    )
    write_dataset_manifest(ctx.root, manifest)
    return dataset_id, {
        "status": "created",
        "added_rows": int(len(frame)),
        "dataset_id": dataset_id,
        "manifest_path": str(ctx.root / "datasets" / domain / dataset_id / "dataset.json"),
    }


def _validate_prepared(
    ctx: PitHistoryContext,
    *,
    symbols: Sequence[str],
    prepared: Mapping[str, Path],
) -> dict[str, Any]:
    scans = {domain: "read_parquet('" + str(path).replace("'", "''") + "')" for domain, path in prepared.items()}
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "validate_spill",
        threads=4,
    ) as con:
        counts = {
            domain: int(con.execute(f"SELECT count(*) FROM {scan}").fetchone()[0]) for domain, scan in scans.items()
        }
        duplicates = _prepared_duplicate_counts(con, ctx=ctx, scans=scans)
        stats = _prepared_validation_counts(con, scans=scans, end_date=ctx.end_date)
    shutil.rmtree(ctx.runtime / "validate_spill", ignore_errors=True)
    errors = _prepared_validation_errors(
        duplicates=duplicates,
        stats=stats,
        expected_symbol_count=len(symbols),
    )
    return {
        "status": "ok" if not errors else "blocked",
        "row_counts": counts,
        "duplicate_key_groups": duplicates,
        **stats,
        "errors": errors,
    }


def _prepared_duplicate_counts(
    con: Any,
    *,
    ctx: PitHistoryContext,
    scans: Mapping[str, str],
) -> dict[str, int]:
    duplicates: dict[str, int] = {}
    for domain in RESTORE_DOMAINS:
        current = resolve_active_domain(domain, workspace_root=ctx.workspace)
        keys = ",".join(f'"{item}"' for item in current.manifest.primary_key)
        duplicates[domain] = int(
            con.execute(
                f"SELECT count(*) FROM (SELECT {keys},count(*) n FROM {scans[domain]} GROUP BY {keys} HAVING n>1)"
            ).fetchone()[0]
        )
    return duplicates


def _prepared_validation_counts(
    con: Any,
    *,
    scans: Mapping[str, str],
    end_date: str,
) -> dict[str, int]:
    daily = scans["market_daily_raw"]
    status = scans["security_status"]
    universe = scans["universe_snapshot"]

    def count(query: str) -> int:
        return int(con.execute(query).fetchone()[0])

    return {
        "missing_factor_keys": count(
            f"SELECT count(*) FROM {daily} d ANTI JOIN {scans['adjust_factor']} f USING(symbol,trade_date)"
        ),
        "missing_status_keys": count(f"SELECT count(*) FROM {daily} d ANTI JOIN {status} s USING(symbol,trade_date)"),
        "missing_universe_keys": count(
            f"SELECT count(*) FROM {daily} d ANTI JOIN {universe} u USING(symbol,trade_date)"
        ),
        "missing_turnover_rows": count(f"SELECT count(*) FROM {scans['valuation']} WHERE turnover_rate IS NULL"),
        "missing_industry_keys": count(
            f"SELECT count(*) FROM {daily} d ANTI JOIN {scans['industry_concept']} i USING(symbol,trade_date)"
        ),
        "missing_share_keys": count(
            f"SELECT count(*) FROM {daily} d ANTI JOIN {scans['share_capital']} s USING(symbol,trade_date)"
        ),
        "missing_valuation_keys": count(
            f"SELECT count(*) FROM {daily} d ANTI JOIN {scans['valuation']} v USING(symbol,trade_date)"
        ),
        "invalid_factor_rows": count(
            f"SELECT count(*) FROM {scans['adjust_factor']} WHERE "
            "adjust_factor IS NULL OR adjust_factor<=0 OR "
            "try_cast(factor_source_date AS DATE)>try_cast(trade_date AS DATE)"
        ),
        "future_share_source_rows": count(
            f"SELECT count(*) FROM {scans['share_capital']} WHERE "
            "try_cast(total_share_source_date AS DATE)>try_cast(trade_date AS DATE) OR "
            "try_cast(float_share_source_date AS DATE)>try_cast(trade_date AS DATE) OR "
            "try_cast(restricted_share_source_date AS DATE)>try_cast(trade_date AS DATE)"
        ),
        "daily_ineligible_status_rows": count(
            f"SELECT count(*) FROM {daily} d JOIN {status} s USING(symbol,trade_date) "
            "WHERE s.is_suspended OR s.is_delisted"
        ),
        "status_without_universe_rows": count(
            f"SELECT count(*) FROM {status} s ANTI JOIN {universe} u USING(symbol,trade_date)"
        ),
        "future_delist_state_rows": count(
            f"SELECT count(*) FROM {status} s JOIN {universe} u USING(symbol,trade_date) "
            "WHERE s.is_delisted AND (u.delist_date='' OR s.trade_date<u.delist_date)"
        ),
        "st_status_rows": count(f"SELECT count(*) FROM {status} WHERE is_st"),
        "suspended_status_rows": count(f"SELECT count(*) FROM {status} WHERE is_suspended"),
        "delisted_status_rows": count(f"SELECT count(*) FROM {status} WHERE is_delisted"),
        "float_share_null_rows": count(f"SELECT count(*) FROM {scans['share_capital']} WHERE float_share IS NULL"),
        "future_name_events": count(f"SELECT count(*) FROM {scans['name_change']} WHERE trade_date>'{end_date}'"),
        "restored_symbol_count": count(f"SELECT count(DISTINCT symbol) FROM {universe}"),
        "restored_daily_symbol_count": count(f"SELECT count(DISTINCT symbol) FROM {daily}"),
    }


def _prepared_validation_errors(
    *,
    duplicates: Mapping[str, int],
    stats: Mapping[str, int],
    expected_symbol_count: int,
) -> list[str]:
    errors = []
    if any(duplicates.values()):
        errors.append(f"duplicate_keys:{duplicates}")
    if (
        stats["missing_factor_keys"]
        or stats["missing_status_keys"]
        or stats["missing_universe_keys"]
        or stats["missing_turnover_rows"]
        or stats["missing_industry_keys"]
        or stats["missing_share_keys"]
        or stats["missing_valuation_keys"]
    ):
        errors.append(
            "cross_domain_missing:"
            f"factor={stats['missing_factor_keys']}:status={stats['missing_status_keys']}:"
            f"universe={stats['missing_universe_keys']}:turnover={stats['missing_turnover_rows']}:"
            f"industry={stats['missing_industry_keys']}:share={stats['missing_share_keys']}:"
            f"valuation={stats['missing_valuation_keys']}"
        )
    if stats["invalid_factor_rows"] or stats["future_share_source_rows"] or stats["daily_ineligible_status_rows"]:
        errors.append(
            "pit_semantic_error:"
            f"invalid_factor={stats['invalid_factor_rows']}:"
            f"future_share={stats['future_share_source_rows']}:"
            f"daily_ineligible={stats['daily_ineligible_status_rows']}"
        )
    if stats["status_without_universe_rows"] or stats["future_delist_state_rows"]:
        errors.append(
            "lifecycle_semantic_error:"
            f"status_without_universe={stats['status_without_universe_rows']}:"
            f"future_delist_state={stats['future_delist_state_rows']}"
        )
    if stats["future_name_events"]:
        errors.append(f"future_name_events:{stats['future_name_events']}")
    if stats["restored_symbol_count"] != expected_symbol_count:
        errors.append(f"restored_symbol_count:{stats['restored_symbol_count']}!={expected_symbol_count}")
    return errors
