"""Evidence-gated adapter for purchased local one-minute Parquet files."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from quantlab.core.io import sha256_file, stable_hash, write_json
from quantlab.data.minute_archive.quality import (
    PRICE_ABSOLUTE_TOLERANCE,
    PRICE_COMPARISON_EPSILON,
    PRICE_RELATIVE_UNRELIABLE_THRESHOLD,
)

from .candidate import (
    EXPECTED_SESSION_TIMES,
    KEY_COLUMNS,
    PRICE_COLUMNS,
    PROVIDER_FIELDS,
    MinuteRepairError,
    TargetBatch,
    _domain_shards,
    _saved_batch_valid,
    evaluate_target_set,
    load_local_target_minutes,
    load_provider_target_minutes,
)

LOCAL_PROVIDER_ID = "external_quant_data_stock_1min"
LOCAL_SOURCE_TAG = "local_minute_zip+external_quant_data_price_repair"
LOCAL_REPAIR_REASON = "targeted external local-Parquet historical minute price repair"
_BASE_FILE_PATTERN = re.compile(r"^(?P<symbol>\d{6}\.(?:SH|SZ|BJ))\.parquet$")
_DUPLICATE_FILE_PATTERN = re.compile(r"^\d{6}\.(?:SH|SZ|BJ)\(\d+\)\.parquet$")
_AGGREGATE_COLUMNS = (
    "symbol",
    "trade_date",
    "row_count",
    "unique_time_count",
    "bad_time_count",
    "bad_price_row_count",
    "ext_open",
    "ext_high",
    "ext_low",
    "ext_close",
    "ext_volume",
    "ext_amount",
)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_json_with_retry(path: Path, payload: Mapping[str, Any]) -> None:
    """Tolerate short-lived Windows scanner locks on newly written evidence files."""

    for attempt in range(6):
        try:
            write_json(path, payload)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.1 * (2**attempt))


def _replace_with_retry(source: Path, target: Path) -> None:
    for attempt in range(8):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.1 * (2**attempt))


def _sql_text(value: str | Path) -> str:
    return "'" + Path(value).as_posix().replace("'", "''") + "'"


def local_source_inventory(source_root: str | Path) -> tuple[dict[str, Path], dict[str, Any]]:
    """Inventory only canonical symbol files and explicitly ignore ``(1)`` copies."""

    root = Path(source_root).resolve()
    if not root.is_dir():
        raise MinuteRepairError(f"minute_repair_local_source_root_missing:{root}")
    files: dict[str, Path] = {}
    ignored_duplicates: list[str] = []
    ignored_other: list[str] = []
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.parquet"), key=lambda item: item.name):
        match = _BASE_FILE_PATTERN.fullmatch(path.name)
        if match:
            symbol = match.group("symbol")
            if symbol in files:
                raise MinuteRepairError(f"minute_repair_local_source_symbol_duplicate:{symbol}")
            stat = path.stat()
            files[symbol] = path
            records.append(
                {
                    "symbol": symbol,
                    "name": path.name,
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
        elif _DUPLICATE_FILE_PATTERN.fullmatch(path.name):
            ignored_duplicates.append(path.name)
        else:
            ignored_other.append(path.name)
    if not files:
        raise MinuteRepairError(f"minute_repair_local_source_base_files_empty:{root}")
    fingerprint = stable_hash(
        {
            "schema": "quantlab.local_minute_source_inventory/v1",
            "root": str(root),
            "files": records,
        }
    )
    return files, {
        "schema": "quantlab.local_minute_source_inventory/v1",
        "created_at": _utc_now(),
        "root": str(root),
        "base_file_count": len(files),
        "ignored_duplicate_file_count": len(ignored_duplicates),
        "ignored_duplicate_files": ignored_duplicates,
        "ignored_other_parquet_count": len(ignored_other),
        "ignored_other_parquet_files": ignored_other,
        "fingerprint": fingerprint,
    }


def _target_key_hash(targets: pd.DataFrame) -> str:
    records = (
        targets.loc[:, ["symbol", "trade_date"]]
        .astype(str)
        .drop_duplicates()
        .sort_values(["symbol", "trade_date"], kind="stable")
        .to_dict("records")
    )
    return stable_hash({"schema": "quantlab.local_minute_aggregate_targets/v1", "targets": records})


def _normalize_aggregate_frame(frame: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(_AGGREGATE_COLUMNS).difference(frame.columns))
    if missing:
        raise MinuteRepairError(
            f"minute_repair_local_aggregate_columns_missing:{','.join(missing)}"
        )
    data = frame.loc[:, list(_AGGREGATE_COLUMNS)].copy()
    data["symbol"] = data["symbol"].astype(str)
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    if data["trade_date"].isna().any():
        raise MinuteRepairError("minute_repair_local_aggregate_trade_date_invalid")
    if data.duplicated(["symbol", "trade_date"]).any():
        raise MinuteRepairError("minute_repair_local_aggregate_duplicate_key")
    target_keys = set(
        map(tuple, targets.loc[:, ["symbol", "trade_date"]].astype(str).to_numpy())
    )
    keep = [
        (str(symbol), str(trade_date)) in target_keys
        for symbol, trade_date in zip(data["symbol"], data["trade_date"], strict=True)
    ]
    data = data.loc[keep].copy()
    for column in _AGGREGATE_COLUMNS[2:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.sort_values(["symbol", "trade_date"], kind="stable").reset_index(drop=True)


def _compute_local_aggregates(
    targets: pd.DataFrame,
    source_files: Mapping[str, Path],
) -> pd.DataFrame:
    expected_sql = ",".join(f"'{value[:6]}'" for value in sorted(EXPECTED_SESSION_TIMES))
    parts: list[pd.DataFrame] = []
    with duckdb.connect(":memory:") as connection:
        connection.execute("SET enable_progress_bar=false")
        connection.execute("SET threads=4")
        for symbol, selected in targets.groupby("symbol", sort=True):
            path = source_files.get(str(symbol))
            if path is None:
                continue
            dates = pd.DataFrame(
                {
                    "trade_date": pd.to_datetime(
                        selected["trade_date"].astype(str), errors="raise"
                    ).dt.date
                }
            ).drop_duplicates()
            connection.register("target_dates", dates)
            frame = connection.execute(
                f"""
                WITH selected AS (
                    SELECT {_sql_text(str(symbol))}::VARCHAR AS symbol,
                           cast(p.trade_date AS DATE) AS trade_date,
                           cast(p.trade_time AS TIMESTAMP) AS trade_time,
                           cast(p.open AS DOUBLE) AS open,
                           cast(p.high AS DOUBLE) AS high,
                           cast(p.low AS DOUBLE) AS low,
                           cast(p.close AS DOUBLE) AS close,
                           cast(p.vol AS DOUBLE) AS volume,
                           cast(p.amount AS DOUBLE) AS amount
                    FROM read_parquet({_sql_text(path)}) p
                    INNER JOIN target_dates t
                      ON cast(p.trade_date AS DATE)=t.trade_date
                ), marked AS (
                    SELECT *,
                           max(CASE WHEN volume>0 OR amount>0 THEN 1 ELSE 0 END)
                               OVER (PARTITION BY trade_date) AS has_flow
                    FROM selected
                ), eligible AS (
                    SELECT *, (has_flow=0 OR volume>0 OR amount>0) AS use_for_price
                    FROM marked
                )
                SELECT symbol,trade_date,
                       count(*)::BIGINT AS row_count,
                       count(DISTINCT trade_time)::BIGINT AS unique_time_count,
                       sum(CASE WHEN strftime(trade_time,'%H%M%S') NOT IN ({expected_sql})
                                THEN 1 ELSE 0 END)::BIGINT AS bad_time_count,
                       sum(CASE WHEN NOT isfinite(open) OR NOT isfinite(high)
                                      OR NOT isfinite(low) OR NOT isfinite(close)
                                      OR open<=0 OR high<=0 OR low<=0 OR close<=0
                                      OR low>least(open,close) OR high<greatest(open,close)
                                      OR low>high
                                THEN 1 ELSE 0 END)::BIGINT AS bad_price_row_count,
                       arg_min(open,trade_time) FILTER (WHERE use_for_price) AS ext_open,
                       max(high) FILTER (WHERE use_for_price) AS ext_high,
                       min(low) FILTER (WHERE use_for_price) AS ext_low,
                       arg_max(close,trade_time) FILTER (WHERE use_for_price) AS ext_close,
                       sum(volume) AS ext_volume,
                       sum(amount) AS ext_amount
                FROM eligible
                GROUP BY symbol,trade_date
                """
            ).df()
            connection.unregister("target_dates")
            if not frame.empty:
                parts.append(frame)
    if not parts:
        return pd.DataFrame(columns=_AGGREGATE_COLUMNS)
    return _normalize_aggregate_frame(pd.concat(parts, ignore_index=True), targets)


def prepare_local_aggregate_evidence(
    targets: pd.DataFrame,
    *,
    source_root: str | Path,
    run_dir: str | Path,
    seed_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    """Prepare a run-scoped aggregate cache, optionally adopting a validated prior scan."""

    output = Path(run_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    aggregate_path = output / "source_aggregate_evidence.parquet"
    metadata_path = output / "source_aggregate_evidence.json"
    inventory_path = output / "source_inventory.json"
    source_files, inventory = local_source_inventory(source_root)
    target_hash = _target_key_hash(targets)
    if aggregate_path.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            reusable = bool(
                metadata.get("source_inventory_fingerprint") == inventory["fingerprint"]
                and metadata.get("target_key_hash") == target_hash
                and metadata.get("sha256") == sha256_file(aggregate_path)
            )
        except (OSError, ValueError, json.JSONDecodeError):
            reusable = False
        if reusable:
            aggregates = _normalize_aggregate_frame(pd.read_parquet(aggregate_path), targets)
            write_json(inventory_path, inventory)
            return aggregates, metadata, {
                "source_aggregates": str(aggregate_path),
                "source_aggregate_metadata": str(metadata_path),
                "source_inventory": str(inventory_path),
            }

    if seed_path is not None:
        seed = Path(seed_path).resolve()
        if not seed.is_file():
            raise MinuteRepairError(f"minute_repair_local_aggregate_seed_missing:{seed}")
        aggregates = _normalize_aggregate_frame(pd.read_parquet(seed), targets)
        method = "validated_seed_subset"
        seed_record = {"path": str(seed), "sha256": sha256_file(seed)}
    else:
        aggregates = _compute_local_aggregates(targets, source_files)
        method = "scanned_canonical_base_files"
        seed_record = None
    temporary = aggregate_path.with_suffix(aggregate_path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    aggregates.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
    _replace_with_retry(temporary, aggregate_path)
    target_keys = set(
        map(tuple, targets.loc[:, ["symbol", "trade_date"]].astype(str).to_numpy())
    )
    aggregate_keys = set(map(tuple, aggregates.loc[:, ["symbol", "trade_date"]].to_numpy()))
    metadata = {
        "schema": "quantlab.local_minute_source_aggregate_evidence/v1",
        "created_at": _utc_now(),
        "method": method,
        "source_root": str(Path(source_root).resolve()),
        "source_inventory_fingerprint": inventory["fingerprint"],
        "target_key_hash": target_hash,
        "target_stock_days": len(target_keys),
        "covered_stock_days": len(aggregate_keys),
        "missing_stock_days": len(target_keys.difference(aggregate_keys)),
        "seed": seed_record,
        "sha256": sha256_file(aggregate_path),
    }
    write_json(metadata_path, metadata)
    write_json(inventory_path, inventory)
    return aggregates, metadata, {
        "source_aggregates": str(aggregate_path),
        "source_aggregate_metadata": str(metadata_path),
        "source_inventory": str(inventory_path),
    }


def _field_excluded(value: float, reference: float, scale: float) -> bool:
    error = abs(float(value) - float(reference))
    return bool(
        error > PRICE_ABSOLUTE_TOLERANCE + PRICE_COMPARISON_EPSILON
        and error / max(abs(float(scale)), 1e-12)
        > PRICE_RELATIVE_UNRELIABLE_THRESHOLD
    )


def prefilter_local_candidates(
    targets: pd.DataFrame,
    aggregates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reject missing, malformed, or non-improving source days before minute loading."""

    evidence = targets.merge(
        aggregates,
        on=["symbol", "trade_date"],
        how="left",
        validate="one_to_one",
    )
    accepted_indexes: list[int] = []
    rejected: list[dict[str, Any]] = []
    prefilter_rows: list[dict[str, Any]] = []
    for index, row in evidence.iterrows():
        reason = ""
        if pd.isna(row["row_count"]):
            reason = "local_source_target_day_missing"
        elif int(row["row_count"]) != len(EXPECTED_SESSION_TIMES):
            reason = f"local_source_session_row_count:{int(row['row_count'])}"
        elif int(row["unique_time_count"]) != len(EXPECTED_SESSION_TIMES) or int(
            row["bad_time_count"]
        ):
            reason = "local_source_session_times_invalid"
        elif int(row["bad_price_row_count"]):
            reason = "local_source_prices_nonpositive_or_invalid_ohlc"
        else:
            candidate_fields: list[str] = []
            for field in PRICE_COLUMNS:
                if not bool(row[f"exclude_{field}"]):
                    continue
                provider_value = float(row[f"ext_{field}"])
                daily_value = float(row[f"d_{field}"])
                baseline_value = float(row[f"agg_{field}"])
                scale = float(row["price_reference_scale"])
                if (
                    not _field_excluded(provider_value, daily_value, scale)
                    and abs(provider_value - daily_value) + PRICE_COMPARISON_EPSILON
                    < abs(baseline_value - daily_value)
                ):
                    candidate_fields.append(field)
            if not candidate_fields:
                reason = "local_source_does_not_clear_selected_field"
        prefilter_rows.append(
            {
                "symbol": str(row["symbol"]),
                "trade_date": str(row["trade_date"]),
                "status": "candidate" if not reason else "rejected",
                "reason": reason,
                "candidate_fields": ",".join(candidate_fields) if not reason else "",
            }
        )
        if not reason:
            accepted_indexes.append(index)
            continue
        decision: dict[str, Any] = {
            "symbol": str(row["symbol"]),
            "trade_date": str(row["trade_date"]),
            "status": "rejected",
            "reason": reason,
            "repaired_fields": "",
            "changed_row_count": 0,
            "selection_reason": str(row["selection_reason"]),
            "provider_raw_path": "",
            "provider_raw_sha256": "",
        }
        for field in PRICE_COLUMNS:
            decision[f"baseline_{field}"] = float(row[f"agg_{field}"])
            decision[f"daily_{field}"] = float(row[f"d_{field}"])
            value = row.get(f"ext_{field}")
            decision[f"provider_{field}"] = float(value) if pd.notna(value) else np.nan
        rejected.append(decision)
    candidates = targets.loc[accepted_indexes].copy().reset_index(drop=True)
    return candidates, pd.DataFrame(rejected), pd.DataFrame(prefilter_rows)


def _local_batch(
    symbol: str,
    dates: Sequence[str],
    *,
    source_path: Path,
    inventory_fingerprint: str,
    raw_dir: Path,
) -> TargetBatch:
    target_dates = tuple(sorted({str(value) for value in dates}))
    batch_id = stable_hash(
        {
            "schema": "quantlab.local_minute_capture_batch/v1",
            "provider": LOCAL_PROVIDER_ID,
            "symbol": symbol,
            "target_dates": target_dates,
            "source_path": str(source_path),
            "source_size": int(source_path.stat().st_size),
            "source_mtime_ns": int(source_path.stat().st_mtime_ns),
            "inventory_fingerprint": inventory_fingerprint,
        }
    )[:20]
    symbol_dir = raw_dir / symbol.replace(".", "_")
    stem = f"{target_dates[0].replace('-', '')}_{target_dates[-1].replace('-', '')}_{batch_id}"
    return TargetBatch(
        batch_id=batch_id,
        symbol=symbol,
        start_date=target_dates[0],
        end_date=target_dates[-1],
        target_dates=target_dates,
        raw_path=str(symbol_dir / f"{stem}.parquet"),
        metadata_path=str(symbol_dir / f"{stem}.json"),
    )


def capture_local_candidate_batches(
    targets: pd.DataFrame,
    *,
    source_root: str | Path,
    raw_dir: str | Path,
) -> tuple[list[TargetBatch], dict[str, Any]]:
    """Capture only qualified target days from each canonical symbol file."""

    if targets.empty:
        return [], {"cached_batches": 0, "new_batches": 0, "raw_rows": 0}
    source_files, inventory = local_source_inventory(source_root)
    output = Path(raw_dir).resolve()
    batches: list[TargetBatch] = []
    cached_count = 0
    new_count = 0
    raw_rows = 0
    with duckdb.connect(":memory:") as connection:
        connection.execute("SET enable_progress_bar=false")
        connection.execute("SET threads=4")
        for symbol, selected in targets.groupby("symbol", sort=True):
            source_path = source_files.get(str(symbol))
            if source_path is None:
                raise MinuteRepairError(f"minute_repair_local_source_symbol_missing:{symbol}")
            batch = _local_batch(
                str(symbol),
                selected["trade_date"].astype(str).tolist(),
                source_path=source_path,
                inventory_fingerprint=str(inventory["fingerprint"]),
                raw_dir=output,
            )
            raw_path = Path(batch.raw_path)
            metadata_path = Path(batch.metadata_path)
            reusable = _saved_batch_valid(batch, raw_path, metadata_path)
            if reusable:
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    stat = source_path.stat()
                    reusable = bool(
                        metadata.get("provider") == LOCAL_PROVIDER_ID
                        and metadata.get("source_inventory_fingerprint")
                        == inventory["fingerprint"]
                        and int(metadata.get("source_file_size", -1)) == int(stat.st_size)
                        and int(metadata.get("source_file_mtime_ns", -1)) == int(stat.st_mtime_ns)
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    reusable = False
            if reusable:
                cached_count += 1
                raw_rows += int(metadata.get("row_count", 0))
                batches.append(batch)
                continue
            dates = pd.DataFrame(
                {
                    "trade_date": pd.to_datetime(
                        selected["trade_date"].astype(str), errors="raise"
                    ).dt.date
                }
            ).drop_duplicates()
            connection.register("target_dates", dates)
            frame = connection.execute(
                f"""
                SELECT {_sql_text(str(symbol))}::VARCHAR AS ts_code,
                       strftime(cast(p.trade_time AS TIMESTAMP),'%Y-%m-%d %H:%M:%S')
                           AS trade_time,
                       cast(p.open AS DOUBLE) AS open,
                       cast(p.high AS DOUBLE) AS high,
                       cast(p.low AS DOUBLE) AS low,
                       cast(p.close AS DOUBLE) AS close,
                       cast(p.vol AS DOUBLE) AS vol,
                       cast(p.amount AS DOUBLE) AS amount
                FROM read_parquet({_sql_text(source_path)}) p
                INNER JOIN target_dates t ON cast(p.trade_date AS DATE)=t.trade_date
                ORDER BY trade_time
                """
            ).df()
            connection.unregister("target_dates")
            frame = frame.loc[:, list(PROVIDER_FIELDS)]
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = raw_path.with_suffix(raw_path.suffix + ".partial")
            temporary.unlink(missing_ok=True)
            frame.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
            _replace_with_retry(temporary, raw_path)
            stat = source_path.stat()
            metadata = {
                "schema": "quantlab.local_minute_capture/v1",
                "created_at": _utc_now(),
                "provider": LOCAL_PROVIDER_ID,
                "batch_id": batch.batch_id,
                "symbol": batch.symbol,
                "start_date": batch.start_date,
                "end_date": batch.end_date,
                "target_dates": list(batch.target_dates),
                "source_file": str(source_path),
                "source_file_size": int(stat.st_size),
                "source_file_mtime_ns": int(stat.st_mtime_ns),
                "source_inventory_fingerprint": inventory["fingerprint"],
                "row_count": int(len(frame)),
                "sha256": sha256_file(raw_path),
            }
            _write_json_with_retry(metadata_path, metadata)
            new_count += 1
            raw_rows += len(frame)
            batches.append(batch)
    return batches, {
        "cached_batches": cached_count,
        "new_batches": new_count,
        "raw_rows": int(raw_rows),
        "source_inventory_fingerprint": inventory["fingerprint"],
    }


def _narrow_batches(
    batches: Sequence[TargetBatch], targets: pd.DataFrame
) -> list[TargetBatch]:
    dates_by_symbol = {
        str(symbol): set(group["trade_date"].astype(str))
        for symbol, group in targets.groupby("symbol", sort=False)
    }
    result: list[TargetBatch] = []
    for batch in batches:
        selected = tuple(value for value in batch.target_dates if value in dates_by_symbol.get(batch.symbol, set()))
        if not selected:
            continue
        result.append(
            TargetBatch(
                batch_id=batch.batch_id,
                symbol=batch.symbol,
                start_date=batch.start_date,
                end_date=batch.end_date,
                target_dates=selected,
                raw_path=batch.raw_path,
                metadata_path=batch.metadata_path,
            )
        )
    return result


def evaluate_local_candidates(
    targets: pd.DataFrame,
    batches: Sequence[TargetBatch],
    *,
    workspace_root: str | Path | None = None,
    chunk_stock_days: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the existing exact-session evaluator in bounded-memory chunks."""

    decision_parts: list[pd.DataFrame] = []
    change_parts: list[pd.DataFrame] = []
    ordered = targets.sort_values(["trade_date", "symbol"], kind="stable").reset_index(drop=True)
    for start in range(0, len(ordered), int(chunk_stock_days)):
        selected = ordered.iloc[start : start + int(chunk_stock_days)].copy()
        local = load_local_target_minutes(selected, workspace_root=workspace_root)
        narrowed = _narrow_batches(batches, selected)
        provider, sources = load_provider_target_minutes(narrowed)
        decisions, changes = evaluate_target_set(
            selected,
            local,
            provider,
            sources,
            source_tag=LOCAL_SOURCE_TAG,
        )
        decision_parts.append(decisions)
        if not changes.empty:
            change_parts.append(changes)
    decisions = pd.concat(decision_parts, ignore_index=True) if decision_parts else pd.DataFrame()
    changes = pd.concat(change_parts, ignore_index=True) if change_parts else pd.DataFrame()
    if not changes.empty and changes.duplicated(list(KEY_COLUMNS)).any():
        raise MinuteRepairError("minute_repair_local_accepted_change_duplicate_key")
    return decisions, changes


def _five_minute_bucket(bar_time: str) -> str:
    hour = int(str(bar_time)[:2])
    minute = int(str(bar_time)[2:4])
    if hour == 9 and minute <= 35:
        return "093500000"
    bucket_minute = ((minute + 4) // 5) * 5
    if bucket_minute == 60:
        hour += 1
        bucket_minute = 0
    return f"{hour:02d}{bucket_minute:02d}00000"


def _load_active_five_minute(keys: pd.DataFrame, workspace_root: str | Path | None) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path, start_date, end_date in _domain_shards(
        "market_intraday_5m", workspace_root=workspace_root
    ):
        selected = keys.loc[keys["trade_date"].between(start_date, end_date)]
        if selected.empty:
            continue
        symbols = ",".join(_sql_text(value) for value in sorted(selected["symbol"].unique()))
        dates = ",".join(_sql_text(value) for value in sorted(selected["trade_date"].unique()))
        with duckdb.connect(":memory:") as connection:
            connection.execute("SET enable_progress_bar=false")
            connection.register("target_keys", selected)
            frame = connection.execute(
                f"""
                SELECT b.symbol,b.trade_date,b.bar_time,b.high,b.low,b.source
                FROM read_parquet({_sql_text(path)}) b
                INNER JOIN target_keys t USING(symbol,trade_date)
                WHERE b.symbol IN ({symbols}) AND b.trade_date IN ({dates})
                """
            ).df()
        if not frame.empty:
            parts.append(frame)
    data = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not data.empty and data.duplicated(list(KEY_COLUMNS)).any():
        raise MinuteRepairError("minute_repair_active_five_minute_duplicate_key")
    return data


def validate_extreme_timing_sample(
    decisions: pd.DataFrame,
    batches: Sequence[TargetBatch],
    *,
    workspace_root: str | Path | None = None,
    samples_per_field_year: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare external 1m high/low buckets with the independent active 5m archive."""

    accepted = decisions.loc[decisions["status"].eq("accepted")].copy()
    samples: list[dict[str, str]] = []
    for field in ("high", "low"):
        selected = accepted.loc[
            accepted["repaired_fields"]
            .astype(str)
            .str.split(",")
            .map(lambda values, selected_field=field: selected_field in values)
        ].copy()
        selected["year"] = selected["trade_date"].astype(str).str[:4].astype(int)
        selected["sample_order"] = [
            stable_hash({"symbol": symbol, "trade_date": trade_date, "field": field})
            for symbol, trade_date in zip(selected["symbol"], selected["trade_date"], strict=True)
        ]
        for _, group in selected.groupby("year", sort=True):
            sampled = group.sort_values("sample_order", kind="stable").head(
                int(samples_per_field_year)
            )
            for row in sampled.to_dict("records"):
                samples.append(
                    {
                        "symbol": str(row["symbol"]),
                        "trade_date": str(row["trade_date"]),
                        "field": field,
                    }
                )
    sample_keys = pd.DataFrame(samples)
    if sample_keys.empty:
        raise MinuteRepairError("minute_repair_extreme_timing_sample_empty")
    unique_keys = sample_keys.loc[:, ["symbol", "trade_date"]].drop_duplicates()
    provider, _ = load_provider_target_minutes(_narrow_batches(batches, unique_keys))
    active = _load_active_five_minute(unique_keys, workspace_root)
    provider_groups = {
        (str(key[0]), str(key[1])): value
        for key, value in provider.groupby(["symbol", "trade_date"])
    }
    active_groups = {
        (str(key[0]), str(key[1])): value
        for key, value in active.groupby(["symbol", "trade_date"])
    }
    records: list[dict[str, Any]] = []
    for sample in samples:
        key = (sample["symbol"], sample["trade_date"])
        field = sample["field"]
        one_minute = provider_groups.get(key, pd.DataFrame())
        five_minute = active_groups.get(key, pd.DataFrame())
        if one_minute.empty or five_minute.empty:
            records.append(
                {
                    **sample,
                    "year": int(sample["trade_date"][:4]),
                    "reference_available": False,
                    "timing_match": False,
                    "reason": "one_or_five_minute_sample_missing",
                }
            )
            continue
        flow = one_minute["provider_volume"].gt(0) | one_minute["provider_amount"].gt(0)
        eligible = one_minute.loc[flow] if bool(flow.any()) else one_minute
        external_extreme = float(eligible[field].max() if field == "high" else eligible[field].min())
        extreme_rows = eligible.loc[eligible[field].eq(external_extreme)]
        external_buckets = sorted({_five_minute_bucket(value) for value in extreme_rows["bar_time"]})
        active_daily_extreme = float(
            five_minute[field].max() if field == "high" else five_minute[field].min()
        )
        price_extreme_match = bool(
            abs(active_daily_extreme - external_extreme)
            <= PRICE_ABSOLUTE_TOLERANCE + PRICE_COMPARISON_EPSILON
        )
        active_extreme_rows = five_minute.loc[five_minute[field].eq(active_daily_extreme)]
        active_extreme_buckets = sorted(set(active_extreme_rows["bar_time"].astype(str)))
        reference_available = bool(
            len(five_minute) == 48 and five_minute["bar_time"].astype(str).nunique() == 48
        )
        timing_match = bool(
            reference_available
            and set(external_buckets).intersection(active_extreme_buckets)
        )
        records.append(
            {
                **sample,
                "year": int(sample["trade_date"][:4]),
                "reference_available": reference_available,
                "timing_match": timing_match,
                "reason": "" if reference_available else "five_minute_session_incomplete",
                "external_extreme": external_extreme,
                "active_five_minute_extreme": active_daily_extreme,
                "price_extreme_match": price_extreme_match,
                "external_extreme_buckets": ",".join(external_buckets),
                "active_five_minute_extreme_buckets": ",".join(active_extreme_buckets),
                "active_five_minute_sources": ",".join(sorted(set(five_minute["source"].astype(str)))),
            }
        )
    frame = pd.DataFrame(records).sort_values(["year", "field", "trade_date", "symbol"])
    available = frame.loc[frame["reference_available"]]
    match_rate = float(available["timing_match"].mean()) if not available.empty else 0.0
    summary = {
        "schema": "quantlab.local_minute_extreme_timing_validation/v1",
        "created_at": _utc_now(),
        "sample_rows": int(len(frame)),
        "reference_available_rows": int(len(available)),
        "timing_match_rows": int(available["timing_match"].sum()),
        "timing_match_rate": match_rate,
        "gate_threshold": 0.98,
        "gate_passed": bool(len(available) >= 10 and match_rate >= 0.98),
        "by_year_field": [
            {
                "year": int(year),
                "field": str(field),
                "sample_rows": int(len(group)),
                "reference_available_rows": int(group["reference_available"].sum()),
                "timing_match_rows": int(group.loc[group["reference_available"], "timing_match"].sum()),
            }
            for (year, field), group in frame.groupby(["year", "field"], sort=True)
        ],
    }
    return frame.reset_index(drop=True), summary


__all__ = [
    "LOCAL_PROVIDER_ID",
    "LOCAL_REPAIR_REASON",
    "LOCAL_SOURCE_TAG",
    "capture_local_candidate_batches",
    "evaluate_local_candidates",
    "local_source_inventory",
    "prefilter_local_candidates",
    "prepare_local_aggregate_evidence",
    "validate_extreme_timing_sample",
]
