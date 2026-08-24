"""Candidate selection, raw capture, and field-level minute repair evaluation."""

from __future__ import annotations

import json
import os
import time
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from quantlab.core.io import sha256_file, stable_hash, write_json
from quantlab.data.core.paths import qdp_paths
from quantlab.data.minute_archive.contracts import AUCTION_TIME, CONTINUOUS_TIMES
from quantlab.data.minute_archive.quality import (
    PRICE_ABSOLUTE_TOLERANCE,
    PRICE_COMPARISON_EPSILON,
    PRICE_RELATIVE_UNRELIABLE_THRESHOLD,
)
from quantlab.data.qdp_v2.auxiliary_update.context import (
    TUSHARE_MAX_RPM,
    TUSHARE_WORKERS,
    TushareRateLimitError,
    _resolve_tushare_token,
    _TushareClient,
)
from quantlab.data.qdp_v2.manifest import (
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
)

PRICE_COLUMNS = ("open", "high", "low", "close")
FLOW_COLUMNS = ("volume", "amount")
KEY_COLUMNS = ("symbol", "trade_date", "bar_time")
PROVIDER_FIELDS = ("ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount")
SOURCE_REPAIR_TAG = "local_minute_zip+tushare_compatible_price_repair"
DEFAULT_PRIORITY_RELATIVE_ERROR = 0.05
DEFAULT_BATCH_CALENDAR_DAYS = 44
DEFAULT_BATCH_TRADING_DAYS = 33
EXPECTED_SESSION_TIMES = frozenset(CONTINUOUS_TIMES | {AUCTION_TIME})
KNOWN_PROBE_TARGETS = frozenset(
    {
        ("600000.SH", "2010-05-17"),
        ("601098.SH", "2015-07-08"),
        ("603607.SH", "2020-07-02"),
        ("003017.SZ", "2024-03-20"),
        ("603388.SH", "2025-03-14"),
        ("605081.SH", "2026-02-09"),
    }
)


class MinuteRepairError(RuntimeError):
    """Raised when a targeted minute repair cannot meet its evidence contract."""


@dataclass(frozen=True)
class TargetBatch:
    batch_id: str
    symbol: str
    start_date: str
    end_date: str
    target_dates: tuple[str, ...]
    raw_path: str = ""
    metadata_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _quality_root(workspace_root: str | Path | None) -> Path:
    return qdp_paths(workspace_root).source_archives_dir / "minute" / "quality"


def _target_columns() -> list[str]:
    return [
        "symbol",
        "trade_date",
        "agg_open",
        "agg_high",
        "agg_low",
        "agg_close",
        "agg_volume",
        "agg_amount",
        "d_open",
        "d_high",
        "d_low",
        "d_close",
        "d_volume",
        "d_amount",
        "open_abs_error",
        "high_abs_error",
        "low_abs_error",
        "close_abs_error",
        "price_reference_scale",
        "price_max_abs_error",
        "price_max_relative_error",
        "price_quality_class",
        "exclude_open",
        "exclude_high",
        "exclude_low",
        "exclude_close",
        "year",
        "selection_reason",
    ]


def load_priority_targets(
    workspace_root: str | Path | None = None,
    *,
    minimum_relative_error: float = DEFAULT_PRIORITY_RELATIVE_ERROR,
    maximum_relative_error: float | None = None,
    include_known_probes: bool = True,
    selection_mode: str = "priority",
) -> pd.DataFrame:
    """Load the small, high-value subset of already audited minute anomalies.

    The priority set contains every price anomaly whose minute aggregate falls
    outside the trusted daily high/low envelope, plus anomalies above the
    requested relative-error threshold.  Previously hand-checked cases remain
    in scope even when they fall below that threshold.
    """

    paths = sorted(_quality_root(workspace_root).glob("year=*/daily_parity_material_mismatches.parquet"))
    if not paths:
        raise MinuteRepairError("minute_repair_quality_evidence_missing")
    parts: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["year"] = int(path.parent.name.split("=", 1)[1])
        parts.append(frame)
    material = pd.concat(parts, ignore_index=True)
    for column in ("trade_date", "symbol"):
        material[column] = material[column].astype(str)
    for column in ("exclude_open", "exclude_high", "exclude_low", "exclude_close"):
        material[column] = material[column].fillna(False).astype(bool)

    audited_anomaly = (
        material["price_max_abs_error"].gt(PRICE_ABSOLUTE_TOLERANCE + PRICE_COMPARISON_EPSILON)
        & material["price_max_relative_error"].gt(PRICE_RELATIVE_UNRELIABLE_THRESHOLD)
    )
    high_overshoot = material["agg_high"].sub(material["d_high"]).gt(
        PRICE_ABSOLUTE_TOLERANCE + PRICE_COMPARISON_EPSILON
    )
    low_overshoot = material["d_low"].sub(material["agg_low"]).gt(
        PRICE_ABSOLUTE_TOLERANCE + PRICE_COMPARISON_EPSILON
    )
    large_error = material["price_max_relative_error"].gt(float(minimum_relative_error))
    known = pd.Series(False, index=material.index)
    if include_known_probes:
        known = pd.Series(
            [
                (str(symbol), str(trade_date)) in KNOWN_PROBE_TARGETS
                for symbol, trade_date in zip(material["symbol"], material["trade_date"], strict=True)
            ],
            index=material.index,
        )
    mode = str(selection_mode).strip().lower()
    if mode == "priority":
        selected_mask = audited_anomaly & (high_overshoot | low_overshoot | large_error | known)
    elif mode == "open":
        open_relative_error = material["open_abs_error"].div(material["price_reference_scale"].clip(lower=1e-12))
        selected_mask = audited_anomaly & material["exclude_open"] & open_relative_error.gt(
            float(minimum_relative_error)
        )
        if maximum_relative_error is not None:
            if float(maximum_relative_error) <= float(minimum_relative_error):
                raise ValueError("minute_repair_maximum_relative_error_must_exceed_minimum")
            selected_mask &= open_relative_error.le(float(maximum_relative_error))
    elif mode == "high-low":
        selected_mask = audited_anomaly & (material["exclude_high"] | material["exclude_low"])
    else:
        raise ValueError(f"minute_repair_selection_mode_invalid:{selection_mode}")
    selected = material.loc[selected_mask].copy()
    if selected.empty:
        raise MinuteRepairError("minute_repair_priority_targets_empty")

    reasons: list[str] = []
    for index in selected.index:
        if mode == "open":
            reasons.append("open_relative_error_above_threshold")
            continue
        if mode == "high-low":
            labels = []
            if bool(material.at[index, "exclude_high"]):
                labels.append("high_quality_mask")
            if bool(material.at[index, "exclude_low"]):
                labels.append("low_quality_mask")
            reasons.append("+".join(labels))
            continue
        labels: list[str] = []
        if bool(high_overshoot.loc[index]) or bool(low_overshoot.loc[index]):
            labels.append("outside_daily_price_envelope")
        if bool(large_error.loc[index]):
            labels.append("relative_error_above_priority_threshold")
        if bool(known.loc[index]):
            labels.append("previously_probed_case")
        reasons.append("+".join(labels))
    selected["selection_reason"] = reasons
    selected = selected.loc[:, _target_columns()].sort_values(["symbol", "trade_date"], kind="stable")
    if selected.duplicated(["symbol", "trade_date"]).any():
        raise MinuteRepairError("minute_repair_priority_target_duplicate")
    return selected.reset_index(drop=True)


def target_set_id(
    targets: pd.DataFrame,
    *,
    minimum_relative_error: float,
    maximum_relative_error: float | None = None,
    selection_mode: str = "priority",
) -> str:
    columns = [
        "symbol",
        "trade_date",
        "price_max_abs_error",
        "price_max_relative_error",
        "exclude_open",
        "exclude_high",
        "exclude_low",
        "exclude_close",
        "selection_reason",
    ]
    records = targets.loc[:, columns].sort_values(["symbol", "trade_date"]).to_dict("records")
    digest = stable_hash(
        {
            "schema": "quantlab.targeted_minute_repair_targets/v1",
            "minimum_relative_error": float(minimum_relative_error),
            "maximum_relative_error": (
                None if maximum_relative_error is None else float(maximum_relative_error)
            ),
            "selection_mode": str(selection_mode),
            "targets": records,
        }
    )
    return f"{str(selection_mode).strip().lower()}_{digest[:16]}"


def plan_target_batches(
    targets: pd.DataFrame,
    *,
    max_calendar_days: int = DEFAULT_BATCH_CALENDAR_DAYS,
    trading_dates: Sequence[str] | None = None,
    max_trading_days: int = DEFAULT_BATCH_TRADING_DAYS,
) -> list[TargetBatch]:
    """Group nearby target dates without reaching the provider's 8,000-row cap."""

    required = {"symbol", "trade_date"}
    missing = sorted(required.difference(targets.columns))
    if missing:
        raise MinuteRepairError(f"minute_repair_target_columns_missing:{','.join(missing)}")
    if int(max_calendar_days) < 0 or int(max_calendar_days) > 44:
        raise ValueError("minute_repair_max_calendar_days_must_be_between_0_and_44")
    trading_ordinal: dict[str, int] = {}
    if trading_dates is not None:
        if int(max_trading_days) < 1 or int(max_trading_days) > 33:
            raise ValueError("minute_repair_max_trading_days_must_be_between_1_and_33")
        normalized_trading_dates = sorted({str(value) for value in trading_dates})
        trading_ordinal = {
            trade_date: index for index, trade_date in enumerate(normalized_trading_dates)
        }
        missing_dates = sorted(set(targets["trade_date"].astype(str)).difference(trading_ordinal))
        if missing_dates:
            raise MinuteRepairError(
                f"minute_repair_target_dates_missing_from_calendar:{missing_dates[:10]}"
            )
    batches: list[TargetBatch] = []
    ordered = targets.assign(_date=pd.to_datetime(targets["trade_date"], errors="raise")).sort_values(
        ["symbol", "_date"], kind="stable"
    )
    for symbol, group in ordered.groupby("symbol", sort=True):
        dates = sorted(set(group["_date"].tolist()))
        start = dates[0]
        selected = [dates[0]]
        for trade_date in dates[1:]:
            exceeds_limit = (
                trading_ordinal[trade_date.strftime("%Y-%m-%d")]
                - trading_ordinal[start.strftime("%Y-%m-%d")]
                >= int(max_trading_days)
                if trading_ordinal
                else int((trade_date - start).days) > int(max_calendar_days)
            )
            if exceeds_limit:
                batches.append(_target_batch(str(symbol), selected))
                start = trade_date
                selected = [trade_date]
            else:
                selected.append(trade_date)
        batches.append(_target_batch(str(symbol), selected))
    return batches


def _target_batch(symbol: str, dates: Sequence[pd.Timestamp]) -> TargetBatch:
    normalized = tuple(value.strftime("%Y-%m-%d") for value in dates)
    start_date = normalized[0]
    end_date = normalized[-1]
    batch_id = stable_hash(
        {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "target_dates": normalized,
        }
    )[:20]
    return TargetBatch(
        batch_id=batch_id,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        target_dates=normalized,
    )


def _batch_paths(raw_dir: Path, batch: TargetBatch) -> tuple[Path, Path]:
    symbol_dir = raw_dir / batch.symbol.replace(".", "_")
    stem = f"{batch.start_date.replace('-', '')}_{batch.end_date.replace('-', '')}_{batch.batch_id}"
    return symbol_dir / f"{stem}.parquet", symbol_dir / f"{stem}.json"


def _batch_from_metadata(metadata_path: Path) -> TargetBatch | None:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        batch = TargetBatch(
            batch_id=str(payload["batch_id"]),
            symbol=str(payload["symbol"]),
            start_date=str(payload["start_date"]),
            end_date=str(payload["end_date"]),
            target_dates=tuple(str(value) for value in payload["target_dates"]),
            raw_path=str(metadata_path.with_suffix(".parquet")),
            metadata_path=str(metadata_path),
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return batch


def _saved_batch_valid(batch: TargetBatch, raw_path: Path, metadata_path: Path) -> bool:
    if not raw_path.is_file() or not metadata_path.is_file():
        return False
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("batch_id") == batch.batch_id
        and payload.get("symbol") == batch.symbol
        and payload.get("start_date") == batch.start_date
        and payload.get("end_date") == batch.end_date
        and all(batch.start_date <= value <= batch.end_date for value in batch.target_dates)
        and int(payload.get("row_count", -1)) >= 0
        and payload.get("sha256") == sha256_file(raw_path)
    )


def load_cached_target_batches(
    targets: pd.DataFrame,
    *,
    raw_dir: str | Path,
) -> list[TargetBatch]:
    """Recover valid captures already written for the current target set."""

    target_keys = set(
        map(tuple, targets.loc[:, ["symbol", "trade_date"]].astype(str).to_numpy())
    )
    covered: set[tuple[str, str]] = set()
    captured: list[TargetBatch] = []
    for metadata_path in sorted(Path(raw_dir).glob("**/*.json")):
        batch = _batch_from_metadata(metadata_path)
        if batch is None or not _saved_batch_valid(
            batch, Path(batch.raw_path), Path(batch.metadata_path)
        ):
            continue
        keys = {(batch.symbol, trade_date) for trade_date in batch.target_dates}
        if not keys or not keys.issubset(target_keys):
            continue
        duplicates = keys.intersection(covered)
        if duplicates:
            raise MinuteRepairError(
                f"minute_repair_cached_target_duplicate:{sorted(duplicates)[:10]}"
            )
        covered.update(keys)
        captured.append(batch)
    return sorted(captured, key=lambda item: (item.symbol, item.start_date, item.batch_id))


def load_reusable_prior_batches(
    targets: pd.DataFrame,
    *,
    run_dir: str | Path,
    workspace_root: str | Path | None = None,
) -> list[TargetBatch]:
    """Slice target dates from valid range captures made by earlier repair runs."""

    if targets.empty:
        return []
    dates_by_symbol = {
        str(symbol): sorted(set(group["trade_date"].astype(str)))
        for symbol, group in targets.groupby("symbol", sort=True)
    }
    assigned: set[tuple[str, str]] = set()
    reused: list[TargetBatch] = []
    current_run = Path(run_dir).resolve()
    repair_root = (
        qdp_paths(workspace_root).source_archives_dir
        / "tushare_compatible"
        / "minute_repair"
    ).resolve()
    prior_runs = sorted(
        (
            path
            for path in repair_root.iterdir()
            if path.is_dir() and path.resolve() != current_run and (path / "batches.json").is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for prior_run in prior_runs:
        try:
            payload = json.loads((prior_run / "batches.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item in list(payload.get("batches", []) or []):
            symbol = str(item.get("symbol", ""))
            dates = dates_by_symbol.get(symbol, [])
            if not dates:
                continue
            start_date = str(item.get("start_date", ""))
            end_date = str(item.get("end_date", ""))
            left = bisect_left(dates, start_date)
            right = bisect_right(dates, end_date)
            selected_dates = tuple(
                value for value in dates[left:right] if (symbol, value) not in assigned
            )
            if not selected_dates:
                continue
            parent = TargetBatch(
                batch_id=str(item.get("batch_id", "")),
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                target_dates=tuple(str(value) for value in list(item.get("target_dates", []) or [])),
                raw_path=str(item.get("raw_path", "")),
                metadata_path=str(item.get("metadata_path", "")),
            )
            if not _saved_batch_valid(
                parent, Path(parent.raw_path), Path(parent.metadata_path)
            ):
                continue
            reused.append(
                TargetBatch(
                    batch_id=parent.batch_id,
                    symbol=parent.symbol,
                    start_date=parent.start_date,
                    end_date=parent.end_date,
                    target_dates=selected_dates,
                    raw_path=parent.raw_path,
                    metadata_path=parent.metadata_path,
                )
            )
            assigned.update((symbol, value) for value in selected_dates)
    return sorted(reused, key=lambda item: (item.symbol, item.start_date, item.batch_id))


def _download_one_batch(
    client: _TushareClient,
    batch: TargetBatch,
    *,
    raw_dir: Path,
    api_url: str,
) -> TargetBatch:
    raw_path, metadata_path = _batch_paths(raw_dir, batch)
    if _saved_batch_valid(batch, raw_path, metadata_path):
        return TargetBatch(**{**batch.to_dict(), "raw_path": str(raw_path), "metadata_path": str(metadata_path)})

    params = {
        "ts_code": batch.symbol,
        "freq": "1min",
        "start_date": f"{batch.start_date} 00:00:00",
        "end_date": f"{batch.end_date} 23:59:59",
    }
    request_strategy = "range"
    actual_requests = [params]
    range_fallback_error = ""
    try:
        frame = client.fetch(
            "stk_mins",
            params=params,
            fields=PROVIDER_FIELDS,
            retries=1 if len(batch.target_dates) > 1 else 4,
        )
        if len(frame) >= 8000:
            raise MinuteRepairError(
                f"minute_repair_provider_response_may_be_truncated:{batch.batch_id}:{len(frame)}"
            )
    except TushareRateLimitError:
        raise
    except Exception as exc:
        if len(batch.target_dates) <= 1:
            raise MinuteRepairError(
                f"minute_repair_provider_batch_failed:{batch.batch_id}:{type(exc).__name__}:{exc}"
            ) from exc
        request_strategy = "target_dates_fallback"
        range_fallback_error = f"{type(exc).__name__}:{str(exc)[:500]}"
        actual_requests = []
        frames: list[pd.DataFrame] = []
        for trade_date in batch.target_dates:
            exact_params = {
                "ts_code": batch.symbol,
                "freq": "1min",
                "start_date": f"{trade_date} 00:00:00",
                "end_date": f"{trade_date} 23:59:59",
            }
            actual_requests.append(exact_params)
            try:
                frames.append(
                    client.fetch(
                        "stk_mins",
                        params=exact_params,
                        fields=PROVIDER_FIELDS,
                    )
                )
            except TushareRateLimitError:
                raise
            except Exception as exact_exc:
                raise MinuteRepairError(
                    "minute_repair_provider_exact_date_failed:"
                    f"{batch.batch_id}:{trade_date}:{type(exact_exc).__name__}:{exact_exc}"
                ) from exact_exc
        frame = pd.concat(frames, ignore_index=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_suffix(raw_path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    frame.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
    os.replace(temporary, raw_path)
    metadata = {
        "schema": "quantlab.tushare_compatible_stk_mins_raw/v1",
        "batch_id": batch.batch_id,
        "symbol": batch.symbol,
        "start_date": batch.start_date,
        "end_date": batch.end_date,
        "target_dates": list(batch.target_dates),
        "api_name": "stk_mins",
        "api_url": api_url,
        "params": params,
        "request_strategy": request_strategy,
        "actual_requests": actual_requests,
        "range_fallback_error": range_fallback_error,
        "fields": list(PROVIDER_FIELDS),
        "row_count": int(len(frame)),
        "sha256": sha256_file(raw_path),
        "fetched_at": _utc_now(),
    }
    write_json(metadata_path, metadata)
    return TargetBatch(**{**batch.to_dict(), "raw_path": str(raw_path), "metadata_path": str(metadata_path)})


def download_target_batches(
    batches: Sequence[TargetBatch],
    *,
    raw_dir: str | Path,
    workspace_root: str | Path | None = None,
    rpm: int = TUSHARE_MAX_RPM,
    workers: int = TUSHARE_WORKERS,
) -> list[TargetBatch]:
    """Capture source responses atomically and reuse valid prior captures."""

    output = Path(raw_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    client = _TushareClient(
        _resolve_tushare_token(workspace_root),
        rpm=int(rpm),
        workspace_root=workspace_root,
    )
    api_url = str(client._url)  # The endpoint is provenance, not a credential.
    completed: dict[str, TargetBatch] = {}
    pending = list(batches)
    last_errors: dict[str, str] = {}
    for round_index in range(3):
        failed: list[TargetBatch] = []
        rate_limit_error: TushareRateLimitError | None = None
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
            futures = {
                executor.submit(
                    _download_one_batch,
                    client,
                    batch,
                    raw_dir=output,
                    api_url=api_url,
                ): batch
                for batch in pending
            }
            for future in as_completed(futures):
                requested = futures[future]
                try:
                    captured = future.result()
                except TushareRateLimitError as exc:
                    rate_limit_error = exc
                    for pending_future in futures:
                        pending_future.cancel()
                    break
                except Exception as exc:
                    failed.append(requested)
                    last_errors[requested.batch_id] = f"{type(exc).__name__}:{str(exc)[:500]}"
                else:
                    completed[captured.batch_id] = captured
                    last_errors.pop(captured.batch_id, None)
        if rate_limit_error is not None:
            raise MinuteRepairError(
                f"minute_repair_provider_quota_exhausted:{rate_limit_error}"
            ) from rate_limit_error
        pending = failed
        if not pending:
            break
        if round_index < 2:
            time.sleep(min(2 ** (round_index + 1), 4))
    if pending:
        examples = ";".join(
            f"{item.batch_id}={last_errors.get(item.batch_id, 'unknown')}" for item in pending[:5]
        )
        raise MinuteRepairError(
            f"minute_repair_provider_batches_failed:{len(pending)}:{examples}"
        )
    return [completed[item.batch_id] for item in batches]


def normalize_provider_minutes(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    required = set(PROVIDER_FIELDS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise MinuteRepairError(f"minute_repair_provider_columns_missing:{','.join(missing)}")
    data = frame.copy()
    data["symbol"] = data["ts_code"].astype(str)
    if not data.empty and not data["symbol"].eq(str(symbol)).all():
        raise MinuteRepairError(f"minute_repair_provider_symbol_mismatch:{symbol}")
    timestamps = pd.to_datetime(data["trade_time"], errors="coerce")
    if timestamps.isna().any():
        raise MinuteRepairError(f"minute_repair_provider_timestamp_invalid:{symbol}")
    data["trade_date"] = timestamps.dt.strftime("%Y-%m-%d")
    data["bar_time"] = timestamps.dt.strftime("%H%M00000")
    for column in PRICE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype("float64")
    data["provider_volume"] = pd.to_numeric(data["vol"], errors="coerce").astype("float64")
    data["provider_amount"] = pd.to_numeric(data["amount"], errors="coerce").astype("float64")
    if data.duplicated(list(KEY_COLUMNS)).any():
        raise MinuteRepairError(f"minute_repair_provider_duplicate_key:{symbol}")
    return data.loc[
        :, [*KEY_COLUMNS, *PRICE_COLUMNS, "provider_volume", "provider_amount"]
    ].sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)


def _sql_text(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _active_manifest(workspace_root: str | Path | None) -> tuple[Path, dict[str, Any]]:
    root = qdp_paths(workspace_root).qdp_v2_dir.resolve()
    active = read_active_manifest(root)
    if not active:
        raise MinuteRepairError("minute_repair_active_manifest_missing")
    return root, active


def _domain_shards(
    domain: str,
    *,
    workspace_root: str | Path | None,
) -> list[tuple[Path, str, str]]:
    root, active = _active_manifest(workspace_root)
    dataset_id = str(dict(active.get("datasets", {}) or {}).get(domain, ""))
    if not dataset_id:
        raise MinuteRepairError(f"minute_repair_active_domain_missing:{domain}")
    manifest_path = root / "datasets" / domain / dataset_id / "dataset.json"
    manifest = read_dataset_manifest(manifest_path)
    return [
        (resolve_manifest_path(entry.path, root=root), str(entry.start_date), str(entry.end_date))
        for entry in manifest.shards
    ]


def load_open_trade_dates(
    workspace_root: str | Path | None = None,
) -> tuple[str, ...]:
    """Load the canonical SSE open dates used to cap provider range rows."""

    parts = [
        pd.read_parquet(path, columns=["trade_date", "exchange", "is_open"])
        for path, _, _ in _domain_shards("trading_calendar", workspace_root=workspace_root)
    ]
    if not parts:
        raise MinuteRepairError("minute_repair_trading_calendar_empty")
    calendar = pd.concat(parts, ignore_index=True)
    selected = calendar.loc[
        calendar["exchange"].astype(str).eq("SSE")
        & calendar["is_open"].fillna(False).astype(bool),
        "trade_date",
    ]
    dates = tuple(sorted(set(selected.astype(str))))
    if not dates:
        raise MinuteRepairError("minute_repair_open_trade_dates_empty")
    return dates


def load_local_target_minutes(
    targets: pd.DataFrame,
    *,
    workspace_root: str | Path | None = None,
) -> pd.DataFrame:
    """Read only audited stock-days from the two active one-minute domains."""

    keys = targets.loc[:, ["symbol", "trade_date"]].drop_duplicates().copy()
    keys["symbol"] = keys["symbol"].astype(str)
    keys["trade_date"] = keys["trade_date"].astype(str)
    parts: list[pd.DataFrame] = []
    for domain in ("market_intraday_1m", "market_opening_auction"):
        for path, start_date, end_date in _domain_shards(domain, workspace_root=workspace_root):
            selected = keys.loc[keys["trade_date"].between(start_date, end_date)]
            if selected.empty:
                continue
            symbols = ",".join(_sql_text(value) for value in sorted(selected["symbol"].unique()))
            dates = ",".join(_sql_text(value) for value in sorted(selected["trade_date"].unique()))
            path_sql = _sql_text(path.as_posix())
            with duckdb.connect(":memory:") as connection:
                connection.execute("SET enable_progress_bar=false")
                connection.register("target_keys", selected)
                frame = connection.execute(
                    f"""
                    SELECT b.symbol,b.trade_date,b.bar_time,b.open,b.high,b.low,b.close,
                           b.volume,b.amount,b.source,b.adjusted_flag
                    FROM read_parquet({path_sql}) b
                    INNER JOIN target_keys t USING(symbol,trade_date)
                    WHERE b.symbol IN ({symbols}) AND b.trade_date IN ({dates})
                    """
                ).df()
            if not frame.empty:
                frame["domain"] = domain
                frame["active_shard_path"] = str(path)
                parts.append(frame)
    if not parts:
        raise MinuteRepairError("minute_repair_local_target_rows_empty")
    data = pd.concat(parts, ignore_index=True)
    if data.duplicated(list(KEY_COLUMNS)).any():
        raise MinuteRepairError("minute_repair_local_duplicate_key")
    expected = set(map(tuple, keys.to_numpy()))
    actual = set(map(tuple, data.loc[:, ["symbol", "trade_date"]].drop_duplicates().to_numpy()))
    missing = sorted(expected.difference(actual))
    if missing:
        raise MinuteRepairError(f"minute_repair_local_target_days_missing:{missing[:10]}")
    return data.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)


def load_provider_target_minutes(
    batches: Sequence[TargetBatch],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    source_rows: list[dict[str, Any]] = []
    for batch in batches:
        raw_path = Path(batch.raw_path)
        metadata_path = Path(batch.metadata_path)
        if not _saved_batch_valid(batch, raw_path, metadata_path):
            raise MinuteRepairError(f"minute_repair_raw_capture_invalid:{batch.batch_id}")
        normalized = normalize_provider_minutes(pd.read_parquet(raw_path), symbol=batch.symbol)
        selected = normalized.loc[normalized["trade_date"].isin(batch.target_dates)].copy()
        selected["provider_raw_path"] = str(raw_path)
        selected["provider_raw_sha256"] = sha256_file(raw_path)
        parts.append(selected)
        source_rows.extend(
            {
                "symbol": batch.symbol,
                "trade_date": trade_date,
                "batch_id": batch.batch_id,
                "provider_raw_path": str(raw_path),
                "provider_raw_sha256": sha256_file(raw_path),
                "provider_metadata_path": str(metadata_path),
            }
            for trade_date in batch.target_dates
        )
    data = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not data.empty and data.duplicated(list(KEY_COLUMNS)).any():
        raise MinuteRepairError("minute_repair_provider_target_duplicate_key")
    sources = pd.DataFrame(source_rows)
    return data.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True), sources


def aggregate_day_prices(frame: pd.DataFrame) -> dict[str, float]:
    """Aggregate prices exactly like the canonical daily parity audit."""

    if frame.empty:
        raise MinuteRepairError("minute_repair_day_empty")
    ordered = frame.sort_values("bar_time", kind="stable")
    flow = ordered["volume"].gt(0) | ordered["amount"].gt(0)
    eligible = ordered.loc[flow] if bool(flow.any()) else ordered
    return {
        "open": float(eligible.iloc[0]["open"]),
        "high": float(eligible["high"].max()),
        "low": float(eligible["low"].min()),
        "close": float(eligible.iloc[-1]["close"]),
    }


def _field_excluded(value: float, reference: float, scale: float) -> bool:
    error = abs(float(value) - float(reference))
    return bool(
        error > PRICE_ABSOLUTE_TOLERANCE + PRICE_COMPARISON_EPSILON
        and error / max(abs(float(scale)), 1e-12) > PRICE_RELATIVE_UNRELIABLE_THRESHOLD
    )


def _candidate_session_error(local: pd.DataFrame, candidate: pd.DataFrame) -> str:
    if len(local) != 241:
        return f"local_session_row_count:{len(local)}"
    if len(candidate) != 241:
        return f"provider_session_row_count:{len(candidate)}"
    local_times = set(local["bar_time"].astype(str))
    candidate_times = set(candidate["bar_time"].astype(str))
    if local_times != EXPECTED_SESSION_TIMES:
        return "local_session_times_invalid"
    if candidate_times != EXPECTED_SESSION_TIMES:
        return "provider_session_times_invalid"
    values = candidate.loc[:, PRICE_COLUMNS].to_numpy(dtype="float64", copy=False)
    if not np.isfinite(values).all() or bool((values <= 0).any()):
        return "provider_prices_nonpositive_or_nonfinite"
    invalid = (
        candidate["low"].gt(candidate[["open", "close"]].min(axis=1))
        | candidate["high"].lt(candidate[["open", "close"]].max(axis=1))
        | candidate["low"].gt(candidate["high"])
    )
    if bool(invalid.any()):
        return "provider_ohlc_invariant_failed"
    return ""


def _rows_for_field(
    field: str,
    local: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    daily_value: float,
) -> set[str]:
    flow = local["volume"].gt(0) | local["amount"].gt(0)
    eligible = local.index[flow] if bool(flow.any()) else local.index
    if field == "open":
        return {str(local.loc[eligible].sort_values("bar_time", kind="stable").iloc[0]["bar_time"])}
    if field == "close":
        return {str(local.loc[eligible].sort_values("bar_time", kind="stable").iloc[-1]["bar_time"])}

    local_values = local.loc[eligible, field]
    candidate_values = candidate.loc[eligible, field]
    result: set[str] = set()
    if field == "high":
        candidate_extreme = float(candidate_values.max())
        if float(local_values.max()) > float(daily_value) + PRICE_ABSOLUTE_TOLERANCE:
            result.update(
                local.loc[
                    eligible[local_values.gt(float(daily_value) + PRICE_ABSOLUTE_TOLERANCE)], "bar_time"
                ].astype(str)
            )
        elif float(local_values.max()) < float(daily_value) - PRICE_ABSOLUTE_TOLERANCE:
            first = candidate_values[candidate_values.eq(candidate_extreme)].index[0]
            result.add(str(local.at[first, "bar_time"]))
    elif field == "low":
        candidate_extreme = float(candidate_values.min())
        if float(local_values.min()) < float(daily_value) - PRICE_ABSOLUTE_TOLERANCE:
            result.update(
                local.loc[
                    eligible[local_values.lt(float(daily_value) - PRICE_ABSOLUTE_TOLERANCE)], "bar_time"
                ].astype(str)
            )
        elif float(local_values.min()) > float(daily_value) + PRICE_ABSOLUTE_TOLERANCE:
            first = candidate_values[candidate_values.eq(candidate_extreme)].index[0]
            result.add(str(local.at[first, "bar_time"]))
    return result


def _proposal_for_fields(
    local: pd.DataFrame,
    candidate: pd.DataFrame,
    fields: Iterable[str],
    *,
    daily: Mapping[str, float],
    source_tag: str,
) -> tuple[pd.DataFrame, set[str]]:
    proposed = local.copy()
    selected_by_field: dict[str, set[str]] = {}
    for field in fields:
        selected_by_field[field] = _rows_for_field(
            field,
            local,
            candidate,
            daily_value=float(daily[field]),
        )
    changed_times = set().union(*selected_by_field.values()) if selected_by_field else set()
    if not changed_times:
        return proposed, changed_times
    for field, times in selected_by_field.items():
        index = proposed["bar_time"].isin(times)
        proposed.loc[index, field] = candidate.loc[index, field].to_numpy()

    for row_index in proposed.index[proposed["bar_time"].isin(changed_times)]:
        changed_fields = {
            field for field, times in selected_by_field.items() if str(proposed.at[row_index, "bar_time"]) in times
        }
        for _ in range(4):
            open_value = float(proposed.at[row_index, "open"])
            high_value = float(proposed.at[row_index, "high"])
            low_value = float(proposed.at[row_index, "low"])
            close_value = float(proposed.at[row_index, "close"])
            if open_value > high_value:
                companion = "high" if "open" in changed_fields else "open"
            elif close_value > high_value:
                companion = "high" if "close" in changed_fields else "close"
            elif open_value < low_value:
                companion = "low" if "open" in changed_fields else "open"
            elif close_value < low_value:
                companion = "low" if "close" in changed_fields else "close"
            elif low_value > high_value:
                companion = "low" if "high" in changed_fields else "high"
            else:
                break
            proposed.at[row_index, companion] = candidate.at[row_index, companion]
            changed_fields.add(companion)

    actually_changed = proposed.loc[:, PRICE_COLUMNS].ne(local.loc[:, PRICE_COLUMNS]).any(axis=1)
    proposed.loc[actually_changed, "source"] = str(source_tag)
    return proposed, set(proposed.loc[actually_changed, "bar_time"].astype(str))


def _proposal_valid(
    proposed: pd.DataFrame,
    *,
    baseline_aggregate: Mapping[str, float],
    daily: Mapping[str, float],
    scale: float,
    candidate_fields: set[str],
) -> tuple[bool, dict[str, float], set[str], str]:
    values = proposed.loc[:, PRICE_COLUMNS].to_numpy(dtype="float64", copy=False)
    if not np.isfinite(values).all() or bool((values <= 0).any()):
        return False, {}, set(), "post_patch_prices_nonpositive_or_nonfinite"
    invalid = (
        proposed["low"].gt(proposed[["open", "close"]].min(axis=1))
        | proposed["high"].lt(proposed[["open", "close"]].max(axis=1))
        | proposed["low"].gt(proposed["high"])
    )
    if bool(invalid.any()):
        return False, {}, set(), "post_patch_ohlc_invariant_failed"
    aggregate = aggregate_day_prices(proposed)
    repaired: set[str] = set()
    for field in PRICE_COLUMNS:
        before_error = abs(float(baseline_aggregate[field]) - float(daily[field]))
        after_error = abs(float(aggregate[field]) - float(daily[field]))
        if after_error > before_error + PRICE_COMPARISON_EPSILON:
            return False, aggregate, set(), f"post_patch_daily_{field}_regressed"
        if field in candidate_fields:
            before_excluded = _field_excluded(baseline_aggregate[field], daily[field], scale)
            after_excluded = _field_excluded(aggregate[field], daily[field], scale)
            if before_excluded and not after_excluded and after_error + PRICE_COMPARISON_EPSILON < before_error:
                repaired.add(field)
    if not repaired:
        return False, aggregate, set(), "no_flagged_field_repaired"
    return True, aggregate, repaired, ""


def evaluate_target_day(
    target: Mapping[str, Any],
    local_day: pd.DataFrame,
    provider_day: pd.DataFrame,
    *,
    source_tag: str = SOURCE_REPAIR_TAG,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return one decision and the minimal accepted row-level price overlay."""

    symbol = str(target["symbol"])
    trade_date = str(target["trade_date"])
    local = local_day.sort_values("bar_time", kind="stable").reset_index(drop=True).copy()
    provider = provider_day.sort_values("bar_time", kind="stable").reset_index(drop=True).copy()
    decision: dict[str, Any] = {
        "symbol": symbol,
        "trade_date": trade_date,
        "status": "rejected",
        "reason": "",
        "repaired_fields": "",
        "changed_row_count": 0,
    }
    session_error = _candidate_session_error(local, provider)
    if session_error:
        decision["reason"] = session_error
        return decision, pd.DataFrame()
    if not local.loc[:, list(KEY_COLUMNS)].equals(provider.loc[:, list(KEY_COLUMNS)]):
        decision["reason"] = "provider_local_keys_mismatch"
        return decision, pd.DataFrame()

    baseline = aggregate_day_prices(local)
    scale = float(target["price_reference_scale"])
    daily = {field: float(target[f"d_{field}"]) for field in PRICE_COLUMNS}
    flagged = {field for field in PRICE_COLUMNS if bool(target[f"exclude_{field}"])}
    if not flagged:
        decision["reason"] = "target_has_no_flagged_price_field"
        return decision, pd.DataFrame()

    full_provider = local.copy()
    for column in PRICE_COLUMNS:
        full_provider[column] = provider[column].to_numpy()
    provider_aggregate = aggregate_day_prices(full_provider)
    candidate_fields = {
        field
        for field in flagged
        if not _field_excluded(provider_aggregate[field], daily[field], scale)
        and abs(provider_aggregate[field] - daily[field]) + PRICE_COMPARISON_EPSILON
        < abs(baseline[field] - daily[field])
    }
    if not candidate_fields:
        decision["reason"] = "provider_does_not_clear_any_flagged_field"
        for field in PRICE_COLUMNS:
            decision[f"baseline_{field}"] = baseline[field]
            decision[f"provider_{field}"] = provider_aggregate[field]
            decision[f"daily_{field}"] = daily[field]
        return decision, pd.DataFrame()

    proposed, changed_times = _proposal_for_fields(
        local,
        provider,
        candidate_fields,
        daily=daily,
        source_tag=source_tag,
    )
    valid, post, repaired, error = _proposal_valid(
        proposed,
        baseline_aggregate=baseline,
        daily=daily,
        scale=scale,
        candidate_fields=candidate_fields,
    )
    if not valid and len(candidate_fields) > 1:
        accepted = local.copy()
        repaired = set()
        for field in sorted(candidate_fields, key=lambda item: abs(baseline[item] - daily[item]), reverse=True):
            trial, _ = _proposal_for_fields(
                accepted,
                provider,
                {field},
                daily=daily,
                source_tag=source_tag,
            )
            ok, _, fixed, _ = _proposal_valid(
                trial,
                baseline_aggregate=aggregate_day_prices(accepted),
                daily=daily,
                scale=scale,
                candidate_fields={field},
            )
            if ok:
                accepted = trial
                repaired.update(fixed)
        proposed = accepted
        post = aggregate_day_prices(proposed)
        changed_times = set(
            proposed.loc[
                proposed.loc[:, PRICE_COLUMNS].ne(local.loc[:, PRICE_COLUMNS]).any(axis=1), "bar_time"
            ].astype(str)
        )
        valid = bool(repaired)
        error = "" if valid else error
    if not valid:
        decision["reason"] = error
        return decision, pd.DataFrame()

    changed = proposed.loc[proposed["bar_time"].isin(changed_times)].copy()
    if changed.empty:
        decision["reason"] = "accepted_proposal_has_no_changed_rows"
        return decision, pd.DataFrame()
    provider_index = provider.set_index("bar_time")
    local_index = local.set_index("bar_time")
    changed["old_open"] = changed["bar_time"].map(local_index["open"])
    changed["old_high"] = changed["bar_time"].map(local_index["high"])
    changed["old_low"] = changed["bar_time"].map(local_index["low"])
    changed["old_close"] = changed["bar_time"].map(local_index["close"])
    changed["provider_open"] = changed["bar_time"].map(provider_index["open"])
    changed["provider_high"] = changed["bar_time"].map(provider_index["high"])
    changed["provider_low"] = changed["bar_time"].map(provider_index["low"])
    changed["provider_close"] = changed["bar_time"].map(provider_index["close"])
    changed["repaired_fields"] = ",".join(sorted(repaired))

    decision.update(
        {
            "status": "accepted",
            "reason": "provider_price_overlay_clears_daily_quality_mask",
            "repaired_fields": ",".join(sorted(repaired)),
            "changed_row_count": int(len(changed)),
        }
    )
    for field in PRICE_COLUMNS:
        decision[f"baseline_{field}"] = baseline[field]
        decision[f"provider_{field}"] = provider_aggregate[field]
        decision[f"post_{field}"] = post[field]
        decision[f"daily_{field}"] = daily[field]
    return decision, changed.reset_index(drop=True)


def evaluate_target_set(
    targets: pd.DataFrame,
    local: pd.DataFrame,
    provider: pd.DataFrame,
    provider_sources: pd.DataFrame,
    *,
    source_tag: str = SOURCE_REPAIR_TAG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    local_groups = {(str(key[0]), str(key[1])): value for key, value in local.groupby(["symbol", "trade_date"])}
    provider_groups = {
        (str(key[0]), str(key[1])): value for key, value in provider.groupby(["symbol", "trade_date"])
    }
    sources = provider_sources.set_index(["symbol", "trade_date"]).to_dict("index")
    decisions: list[dict[str, Any]] = []
    changes: list[pd.DataFrame] = []
    for target in targets.to_dict("records"):
        key = (str(target["symbol"]), str(target["trade_date"]))
        local_day = local_groups.get(key, pd.DataFrame())
        provider_day = provider_groups.get(key, pd.DataFrame())
        if local_day.empty:
            decision = {
                "symbol": key[0],
                "trade_date": key[1],
                "status": "rejected",
                "reason": "local_target_day_missing",
                "repaired_fields": "",
                "changed_row_count": 0,
            }
            changed = pd.DataFrame()
        else:
            decision, changed = evaluate_target_day(
                target,
                local_day,
                provider_day,
                source_tag=source_tag,
            )
        source = sources.get(key, {})
        decision.update(
            {
                "selection_reason": str(target["selection_reason"]),
                "provider_raw_path": str(source.get("provider_raw_path", "")),
                "provider_raw_sha256": str(source.get("provider_raw_sha256", "")),
            }
        )
        decisions.append(decision)
        if not changed.empty:
            changed["provider_raw_path"] = str(source.get("provider_raw_path", ""))
            changed["provider_raw_sha256"] = str(source.get("provider_raw_sha256", ""))
            changes.append(changed)
    decision_frame = pd.DataFrame(decisions).sort_values(["trade_date", "symbol"], kind="stable")
    change_frame = pd.concat(changes, ignore_index=True) if changes else pd.DataFrame()
    if not change_frame.empty:
        if change_frame.duplicated(list(KEY_COLUMNS)).any():
            raise MinuteRepairError("minute_repair_accepted_change_duplicate_key")
        change_frame = change_frame.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)
    return decision_frame.reset_index(drop=True), change_frame


def write_candidate_artifacts(
    run_dir: str | Path,
    *,
    targets: pd.DataFrame,
    batches: Sequence[TargetBatch],
    decisions: pd.DataFrame,
    changes: pd.DataFrame,
) -> dict[str, str]:
    output = Path(run_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "targets": output / "targets.parquet",
        "batches": output / "batches.json",
        "decisions": output / "candidate_decisions.parquet",
        "changes": output / "accepted_row_changes.parquet",
    }
    targets.to_parquet(paths["targets"], index=False, engine="pyarrow", compression="zstd")
    write_json(
        paths["batches"],
        {
            "schema": "quantlab.targeted_minute_repair_batches/v1",
            "created_at": _utc_now(),
            "batches": [item.to_dict() for item in batches],
        },
    )
    decisions.to_parquet(paths["decisions"], index=False, engine="pyarrow", compression="zstd")
    if changes.empty:
        pd.DataFrame(columns=[*KEY_COLUMNS, *PRICE_COLUMNS]).to_parquet(
            paths["changes"], index=False, engine="pyarrow", compression="zstd"
        )
    else:
        changes.to_parquet(paths["changes"], index=False, engine="pyarrow", compression="zstd")
    return {key: str(value) for key, value in paths.items()}


__all__ = [
    "DEFAULT_BATCH_CALENDAR_DAYS",
    "DEFAULT_BATCH_TRADING_DAYS",
    "DEFAULT_PRIORITY_RELATIVE_ERROR",
    "KEY_COLUMNS",
    "MinuteRepairError",
    "PRICE_COLUMNS",
    "SOURCE_REPAIR_TAG",
    "TargetBatch",
    "aggregate_day_prices",
    "download_target_batches",
    "evaluate_target_day",
    "evaluate_target_set",
    "load_cached_target_batches",
    "load_local_target_minutes",
    "load_open_trade_dates",
    "load_priority_targets",
    "load_provider_target_minutes",
    "load_reusable_prior_batches",
    "normalize_provider_minutes",
    "plan_target_batches",
    "target_set_id",
    "write_candidate_artifacts",
]
