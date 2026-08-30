"""Recent Market Repair: mootdx responsibilities."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data.domains.contracts.requests import DomainFetchRequest
from quantlab.data.providers.mootdx.provider import MootdxOnlineProvider
from quantlab.data.qdp_v2.manifest import (
    atomic_write_json,
)

from .commit import (
    _write_final_parquet,
)
from .config import (
    BUCKET_COUNT,
    DAILY_COLUMNS,
    DAILY_DOMAIN,
    INTRADAY_COLUMNS,
    INTRADAY_DOMAIN,
    RecentMarketRepairError,
    _SystemMemoryGuard,
)
from .state import (
    _requested_pairs,
    _reused_bucket_is_valid,
    _split_symbols,
    _stable_bucket,
    _unresolved_mapping,
    _utc_now,
)
from .validation import (
    _validate_daily_frame,
    _validate_intraday_frame,
)


def _download_buckets(
    *,
    runtime: Path,
    state: dict[str, Any],
    wanted: Mapping[str, Sequence[str]],
    provider_domain: str,
    output_domain: str,
    start_date: str,
    end_date: str,
    workers: int,
    guard: _SystemMemoryGuard,
) -> dict[str, Any]:
    bundles = runtime / "bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    completed = dict(state.get("buckets", {}) or {})
    for bucket in range(BUCKET_COUNT):
        key = f"{bucket:02d}"
        guard.check(f"bucket_{key}_start")
        bucket_wanted = {
            symbol: tuple(sorted({str(item) for item in dates}))
            for symbol, dates in wanted.items()
            if _stable_bucket(symbol) == bucket
        }
        previous = completed.get(key)
        if (
            isinstance(previous, dict)
            and previous.get("status") == "completed"
            and _reused_bucket_is_valid(previous, runtime=runtime, wanted=bucket_wanted)
        ):
            continue
        frame, unresolved, errors = _fetch_bucket(
            bucket_wanted,
            provider_domain=provider_domain,
            output_domain=output_domain,
            start_date=start_date,
            end_date=end_date,
            workers=workers,
            guard=guard,
        )
        bundle_path: Path | None = None
        if not frame.empty:
            bundle_path = bundles / f"{output_domain}_bucket_{key}.parquet"
            _write_final_parquet(frame, bundle_path, domain=output_domain)
        completed[key] = {
            "status": "completed",
            "symbol_count": len(bucket_wanted),
            "requested_day_count": sum(len(item) for item in bucket_wanted.values()),
            "accepted_day_count": int(len(frame) if output_domain == DAILY_DOMAIN else len(frame) // 48),
            "row_count": len(frame),
            "bundle_path": str(bundle_path) if bundle_path else "",
            "file_size": bundle_path.stat().st_size if bundle_path else 0,
            "requested_pairs": _requested_pairs(bucket_wanted),
            "unresolved": [
                {"symbol": symbol, "trade_date": trade_date, "reason": reason}
                for (symbol, trade_date), reason in sorted(unresolved.items())
            ],
            "provider_error_count": len(errors),
            "provider_error_sample": errors[:10],
        }
        state.update(
            {
                "status": "downloading",
                "buckets": completed,
                "completed_bucket_count": sum(item.get("status") == "completed" for item in completed.values()),
                "updated_at": _utc_now(),
            }
        )
        atomic_write_json(runtime / "state.json", state)
        print(
            json.dumps(
                {
                    "bucket": key,
                    "symbols": len(bucket_wanted),
                    "rows": len(frame),
                    "unresolved_days": len(unresolved),
                    "provider_errors": len(errors),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    state["status"] = "downloaded"
    state["buckets"] = completed
    state["row_count"] = sum(int(item.get("row_count", 0)) for item in completed.values())
    state["accepted_day_count"] = sum(int(item.get("accepted_day_count", 0)) for item in completed.values())
    state["unresolved_day_count"] = sum(len(item.get("unresolved", []) or []) for item in completed.values())
    state["provider_error_count"] = sum(int(item.get("provider_error_count", 0)) for item in completed.values())
    state["updated_at"] = _utc_now()
    atomic_write_json(runtime / "state.json", state)
    return state


def _fetch_bucket(
    wanted: Mapping[str, Sequence[str]],
    *,
    provider_domain: str,
    output_domain: str,
    start_date: str,
    end_date: str,
    workers: int,
    guard: _SystemMemoryGuard,
) -> tuple[pd.DataFrame, dict[tuple[str, str], str], list[dict[str, Any]]]:
    pending = {symbol: tuple(dates) for symbol, dates in wanted.items()}
    accepted: list[pd.DataFrame] = []
    all_errors: list[dict[str, Any]] = []
    last_reasons: dict[tuple[str, str], str] = {
        (symbol, trade_date): "not_requested" for symbol, dates in pending.items() for trade_date in dates
    }
    for attempt in range(1, 3):
        guard.check(f"provider_round_{attempt}")
        symbols = tuple(sorted(pending))
        if not symbols:
            break
        chunks = _split_symbols(symbols, max(1, int(workers)))
        frames: list[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = {
                executor.submit(
                    _fetch_mootdx_chunk,
                    chunk,
                    domain=provider_domain,
                    start_date=start_date,
                    end_date=end_date,
                ): chunk
                for chunk in chunks
            }
            for future in as_completed(futures):
                guard.check(f"provider_round_{attempt}_result")
                try:
                    frame, errors = future.result()
                except BaseException as exc:  # noqa: BLE001 - retain worker failure
                    all_errors.append(
                        {
                            "code": "chunk_fetch_error",
                            "error_type": type(exc).__name__,
                            "message": str(exc)[:500],
                            "symbol_count": len(futures[future]),
                        }
                    )
                    continue
                if not frame.empty:
                    frames.append(frame)
                all_errors.extend(dict(item) for item in errors)
        downloaded = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        if output_domain == DAILY_DOMAIN:
            valid, unresolved = _validate_daily_frame(downloaded, pending)
        else:
            valid, unresolved = _validate_intraday_frame(downloaded, pending)
        if not valid.empty:
            accepted.append(valid)
        pending = _unresolved_mapping(unresolved)
        last_reasons = unresolved
    if accepted:
        combined = pd.concat(accepted, ignore_index=True, sort=False)
        key_columns = ["symbol", "trade_date"] + (["bar_time"] if output_domain == INTRADAY_DOMAIN else [])
        if combined.duplicated(key_columns).any():
            raise RecentMarketRepairError("recent_repair_duplicate_accepted_primary_key")
        combined = combined.sort_values(key_columns, kind="stable").reset_index(drop=True)
    else:
        columns = INTRADAY_COLUMNS if output_domain == INTRADAY_DOMAIN else DAILY_COLUMNS
        combined = pd.DataFrame(columns=columns)
    return combined, last_reasons, all_errors


def _fetch_mootdx_chunk(
    symbols: Sequence[str],
    *,
    domain: str,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    provider = MootdxOnlineProvider(page_size=800, max_pages=3)
    try:
        result = provider.fetch_domain(
            DomainFetchRequest(
                domain=domain,
                symbols=tuple(symbols),
                start_date=start_date,
                end_date=end_date,
                adjusted_flag="none",
            )
        )
        return result.data, list(result.error_report or [])
    finally:
        provider.close()
