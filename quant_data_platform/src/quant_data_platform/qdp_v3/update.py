from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from quant_data_platform.core.json_io import read_json
from quant_data_platform.qdp_v3.audit import audit_candidate
from quant_data_platform.qdp_v3.build import build_candidate
from quant_data_platform.qdp_v3.constants import (
    BOOTSTRAP_CUTOFF,
    RAW_ADJUST_FACTOR_EVENT,
    RAW_EXTERNAL_QUANT_INTRADAY_5M,
    RAW_INTRADAY_5M_BAOSTOCK,
    RAW_INTRADAY_5M_MOOTDX,
    RAW_INTRADAY_5M_SELECTED,
    RAW_LEGACY_V2_INTRADAY_5M,
    RAW_TRADING_CALENDAR,
    RAW_TUSHARE_PROXY_DAILY,
    RAW_TUSHARE_PROXY_INTRADAY_5M,
    RAW_TUSHARE_PROXY_TRADE_CALENDAR,
    V2_RETIREMENT_GATE_FILENAME,
)
from quant_data_platform.qdp_v3.compaction import compact_raw_domain
from quant_data_platform.qdp_v3.freeze import freeze_v2, validate_v2_freeze_proof
from quant_data_platform.qdp_v3.historical import (
    ingest_tushare_proxy_intraday,
    ingest_tushare_proxy_reference,
    plan_intraday_residuals,
    proxy_symbol_inventory,
)
from quant_data_platform.qdp_v3.identity import SecurityIdentityRegistry, board_for_symbol, normalize_symbol
from quant_data_platform.qdp_v3.ingest import (
    ingest_baostock_date_partitions,
    ingest_factor_symbol_histories,
    ingest_security_master,
    ingest_trading_calendar,
)
from quant_data_platform.qdp_v3.intraday import ingest_intraday_5m
from quant_data_platform.qdp_v3.manifest import active_manifest_sha256, atomic_write_json, stable_hash, utc_now
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout, qdp_v3_paths
from quant_data_platform.qdp_v3.proxy_compatibility import lock_bootstrap_cutoff, run_tushare_proxy_compatibility_gate
from quant_data_platform.qdp_v3.release import (
    diff_candidate,
    publish_candidate,
    validate_published_active,
)
from quant_data_platform.qdp_v3.retirement import retire_v2_intraday
from quant_data_platform.qdp_v3.storage import iter_raw_partitions, read_raw_partition
from quant_data_platform.qdp_v3.supervisor import requeue_stale_jobs, supervise_job


TRUSTED_BOOTSTRAP_PROVIDER = "tushare-proxy"
LOCAL_ARCHIVE_BOOTSTRAP_PROVIDER = "external-quant-archive"
TRUSTED_BOOTSTRAP_START = "2010-01-01"
INCREMENTAL_CALENDAR_LOOKBACK_DAYS = 120
INCREMENTAL_DAILY_REFRESH_DAYS = 10
INCREMENTAL_FACTOR_REFRESH_DAYS = 60


def _proxy_daily_expected_trade_dates(
    *,
    symbols: Iterable[str],
    trade_dates: Iterable[str],
    identity_registry: SecurityIdentityRegistry,
    workspace_root: str | Path | None,
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    """Build exact traded stock-days from the trusted daily date partitions.

    Absence from a successfully captured full-market daily partition is the
    local proof that a listed security did not trade that day. The mapping is
    returned only when every requested exchange date has a daily partition;
    partial evidence must never turn an unknown day into a suspension.
    """

    dates = sorted({_normalize_date(item) for item in trade_dates if str(item)})
    requested_symbols = sorted({normalize_symbol(item) for item in symbols if normalize_symbol(item)})
    security_ids = {
        str(security_id)
        for symbol in requested_symbols
        if (security_id := identity_registry.security_id_for_provider_symbol(symbol))
    }
    expected: dict[str, set[str]] = {str(security_id): set() for security_id in security_ids}
    if not dates or not expected:
        return {}, {
            "status": "unavailable",
            "reason": "trade_dates_or_identity_empty",
            "requested_trade_date_count": len(dates),
        }
    covered_dates: set[str] = set()
    refs = iter_raw_partitions(
        RAW_TUSHARE_PROXY_DAILY,
        workspace_root=workspace_root,
        start_value=dates[0],
        end_value=dates[-1],
    )
    date_set = set(dates)
    for ref in refs:
        partition_date = str(ref.partition_value)[:10]
        if partition_date not in date_set:
            continue
        frame = read_raw_partition(ref)
        covered_dates.add(partition_date)
        if frame.empty:
            continue
        symbol_column = (
            "ts_code"
            if "ts_code" in frame.columns
            else "code"
            if "code" in frame.columns
            else "provider_symbol"
            if "provider_symbol" in frame.columns
            else ""
        )
        if not symbol_column:
            raise RuntimeError(f"proxy_daily_symbol_column_missing:{ref.partition_value}")
        for provider_symbol in frame[symbol_column].map(normalize_symbol).drop_duplicates():
            security_id = identity_registry.security_id_for_provider_symbol(provider_symbol)
            if security_id in expected:
                expected[str(security_id)].add(partition_date)
    missing_dates = sorted(date_set.difference(covered_dates))
    evidence = {
        "status": "complete" if not missing_dates else "partial",
        "requested_trade_date_count": len(dates),
        "covered_trade_date_count": len(covered_dates),
        "missing_trade_date_count": len(missing_dates),
        "missing_trade_dates_sample": missing_dates[:20],
        "security_count": len(expected),
        "expected_stock_day_count": sum(len(value) for value in expected.values()),
    }
    if missing_dates:
        return {}, evidence
    return {key: tuple(sorted(value)) for key, value in expected.items()}, evidence


def _advance_retirement_gate(
    *,
    paths: Any,
    bootstrap: bool,
    run_id: str,
    candidate_id: str,
    active_sha256: str,
    active: dict[str, Any],
) -> dict[str, Any]:
    gate_path = paths.metadata / V2_RETIREMENT_GATE_FILENAME
    coverage = dict(active.get("coverage", {}) or {})
    daily_watermark = str(coverage.get("market_daily_watermark", "") or "")
    five_watermark = str(coverage.get("market_intraday_5m_watermark", "") or "")
    if bootstrap:
        payload = {
            "contract": "qdp_v3_post_bootstrap_incremental_retirement_gate_v1",
            "status": "awaiting_post_cutoff_incremental",
            "bootstrap_cutoff": BOOTSTRAP_CUTOFF,
            "bootstrap_run_id": run_id,
            "bootstrap_candidate_id": candidate_id,
            "bootstrap_active_sha256": active_sha256,
            "bootstrap_daily_watermark": daily_watermark,
            "bootstrap_5m_watermark": five_watermark,
            "updated_at": utc_now(),
        }
        atomic_write_json(gate_path, payload)
        return {**payload, "gate_path": str(gate_path.resolve())}

    gate = read_json(gate_path)
    if gate.get("status") not in {"awaiting_post_cutoff_incremental", "incremental_verification_blocked"}:
        return {
            "status": "not_applicable",
            "reason": "bootstrap_retirement_gate_not_waiting",
            "gate_path": str(gate_path.resolve()),
        }
    blockers: list[dict[str, Any]] = []
    if candidate_id == str(gate.get("bootstrap_candidate_id", "") or ""):
        blockers.append({"code": "incremental_candidate_same_as_bootstrap"})
    if not daily_watermark or daily_watermark <= BOOTSTRAP_CUTOFF:
        blockers.append({"code": "daily_watermark_not_post_cutoff", "watermark": daily_watermark})
    if not five_watermark or five_watermark <= BOOTSTRAP_CUTOFF:
        blockers.append({"code": "intraday_5m_watermark_not_post_cutoff", "watermark": five_watermark})
    if daily_watermark != five_watermark:
        blockers.append(
            {
                "code": "incremental_watermark_mismatch",
                "market_daily_watermark": daily_watermark,
                "market_intraday_5m_watermark": five_watermark,
            }
        )
    if blockers:
        payload = {
            **gate,
            "status": "incremental_verification_blocked",
            "last_incremental_run_id": run_id,
            "last_incremental_candidate_id": candidate_id,
            "last_incremental_active_sha256": active_sha256,
            "blockers": blockers,
            "updated_at": utc_now(),
        }
        atomic_write_json(gate_path, payload)
        return {**payload, "gate_path": str(gate_path.resolve())}
    payload = {
        **gate,
        "status": "incremental_verified",
        "incremental_run_id": run_id,
        "incremental_candidate_id": candidate_id,
        "incremental_active_sha256": active_sha256,
        "incremental_daily_watermark": daily_watermark,
        "incremental_5m_watermark": five_watermark,
        "verified_at": utc_now(),
        "updated_at": utc_now(),
    }
    payload.pop("blockers", None)
    atomic_write_json(gate_path, payload)
    return {**payload, "gate_path": str(gate_path.resolve())}


def _normalize_date(value: Any) -> str:
    text = str(value or "")[:10]
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def _incremental_lookback_start(as_of_date: str, requested_start: str = "") -> str:
    required = (
        pd.Timestamp(str(as_of_date)[:10])
        - pd.Timedelta(days=INCREMENTAL_CALENDAR_LOOKBACK_DAYS)
    ).strftime("%Y-%m-%d")
    explicit = str(requested_start or "")[:10]
    return min(explicit, required) if explicit else required


def _incremental_trade_date_windows(
    dates: Iterable[str],
    *,
    require_full_factor_window: bool = False,
) -> tuple[list[str], list[str]]:
    normalized = sorted({_normalize_date(item) for item in dates if _normalize_date(item)})
    if require_full_factor_window and len(normalized) < INCREMENTAL_FACTOR_REFRESH_DAYS:
        raise RuntimeError(
            "incremental_calendar_lookback_insufficient:"
            f"required={INCREMENTAL_FACTOR_REFRESH_DAYS} actual={len(normalized)}"
        )
    return (
        normalized[-INCREMENTAL_DAILY_REFRESH_DAYS:],
        normalized[-INCREMENTAL_FACTOR_REFRESH_DAYS:],
    )


def _calendar_from_raw_domain(
    raw_domain: str,
    *,
    start_date: str,
    end_date: str,
    workspace_root: str | Path | None,
) -> list[str]:
    frames: list[pd.DataFrame] = []
    for ref in iter_raw_partitions(raw_domain, workspace_root=workspace_root):
        frame = read_raw_partition(ref)
        date_column = "cal_date" if "cal_date" in frame.columns else "trade_date" if "trade_date" in frame.columns else ""
        if not date_column or "is_open" not in frame.columns:
            continue
        selected = pd.DataFrame(
            {
                "trade_date": frame[date_column].map(_normalize_date),
                "is_open": frame["is_open"],
            }
        )
        frames.append(selected)
    if not frames:
        return []
    calendar = pd.concat(frames, ignore_index=True)
    opened = calendar["is_open"].astype(str).str.lower().isin({"1", "true", "t", "yes"})
    dates = calendar.loc[
        opened & calendar["trade_date"].between(str(start_date), str(end_date)),
        "trade_date",
    ]
    return sorted(set(dates.astype(str)))


def _calendar_dates_read_only(
    *,
    start_date: str,
    end_date: str,
    workspace_root: str | Path | None,
    prefer_proxy: bool = False,
) -> tuple[list[str], str]:
    ordered = (
        ((RAW_TUSHARE_PROXY_TRADE_CALENDAR, "trusted_tushare_proxy_calendar"), (RAW_TRADING_CALENDAR, "stored_baostock_calendar"))
        if prefer_proxy
        else ((RAW_TRADING_CALENDAR, "stored_baostock_calendar"), (RAW_TUSHARE_PROXY_TRADE_CALENDAR, "trusted_tushare_proxy_calendar"))
    )
    for raw_domain, source in ordered:
        dates = _calendar_from_raw_domain(
            raw_domain,
            start_date=start_date,
            end_date=end_date,
            workspace_root=workspace_root,
        )
        if dates:
            return dates, source
    dates = [item.strftime("%Y-%m-%d") for item in pd.bdate_range(start_date, end_date)]
    return dates, "business_day_fallback_dry_run_only"


def _date_task_summary(dates: list[str]) -> dict[str, Any]:
    if not dates:
        return {"task_count": 0, "start_date": "", "end_date": "", "sample_dates": []}
    sample = dates if len(dates) <= 10 else [*dates[:5], *dates[-5:]]
    return {
        "task_count": len(dates),
        "start_date": dates[0],
        "end_date": dates[-1],
        "sample_dates": sample,
    }


def plan_update(
    *,
    as_of_date: str,
    workspace_root: str | Path | None = None,
    bootstrap: bool = False,
    start_date: str = "",
    historical_provider: str = "",
    source_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    paths = qdp_v3_paths(workspace_root)
    active_validation = validate_published_active(workspace_root)
    if not active_validation["valid"] and not bootstrap:
        return {
            "status": "blocked",
            "blocker": "qdp_v3_bootstrap_requires_explicit_bootstrap",
            "message": "No v3 active exists. Use --bootstrap for the initial trusted-source rebuild.",
            "active_sha256": active_manifest_sha256(workspace_root),
        }
    normalized_source_paths = tuple(str(Path(item)) for item in source_paths if str(item).strip())
    normalized_provider = str(
        historical_provider
        or (
            LOCAL_ARCHIVE_BOOTSTRAP_PROVIDER
            if bootstrap and normalized_source_paths
            else TRUSTED_BOOTSTRAP_PROVIDER if bootstrap else ""
        )
    ).strip().lower()
    if bootstrap and normalized_provider not in {TRUSTED_BOOTSTRAP_PROVIDER, LOCAL_ARCHIVE_BOOTSTRAP_PROVIDER}:
        return {
            "status": "blocked",
            "blocker": "qdp_v3_bootstrap_historical_provider_unsupported",
            "historical_provider": normalized_provider,
            "active_sha256": active_manifest_sha256(workspace_root),
        }
    if bootstrap and normalized_provider == LOCAL_ARCHIVE_BOOTSTRAP_PROVIDER and not normalized_source_paths:
        return {
            "status": "blocked",
            "blocker": "qdp_v3_local_archive_bootstrap_requires_source_path",
            "historical_provider": normalized_provider,
            "active_sha256": active_manifest_sha256(workspace_root),
        }
    effective_start = (
        str(start_date or TRUSTED_BOOTSTRAP_START)
        if bootstrap
        else _incremental_lookback_start(str(as_of_date), str(start_date))
    )
    dates, calendar_source = _calendar_dates_read_only(
        start_date=effective_start,
        end_date=str(as_of_date),
        workspace_root=workspace_root,
        prefer_proxy=bool(bootstrap),
    )
    freeze_state = read_json(paths.metadata / "v2_freeze_20260713.json")
    freeze_validation = validate_v2_freeze_proof(freeze_state)
    common_tail = [
        {"stage": "build_candidate"},
        {"stage": "semantic_audit"},
        {"stage": "diff_candidate", "against": "active"},
        {"stage": "cas_publish"},
        {"stage": "post_publish_semantic_hash_audit"},
    ]
    if bootstrap:
        legacy_migration_stage = {
            "stage": "legacy_v2_intraday_5m_migration",
            "provider": "qdp_v2_active",
            "priority": 0,
            "mode": "reuse_or_migrate_strict_complete_stock_days",
        }
        intraday_stage = (
            {
                "stage": "intraday_5m_local_archive_then_tushare_residual",
                "provider": "external_quant_archive_with_tushare_proxy_residual",
                "priority": 1,
                "source_paths": list(normalized_source_paths),
                "source_priority": [
                    "external_quant_archive",
                    "qdp_v2_migration",
                    "mootdx_or_baostock_complete_day",
                    "tushare_proxy_residual",
                ],
                "archive_scope": "direct_plus_legacy_residual_only",
                **_date_task_summary(dates),
            }
            if normalized_provider == LOCAL_ARCHIVE_BOOTSTRAP_PROVIDER
            else {
                "stage": "intraday_5m",
                "provider": "tushare_proxy",
                "priority": 1,
                "routing": "existing_direct_plus_v2_then_free_2020_plus_then_tushare_residual",
                **_date_task_summary(dates),
            }
        )
        stages = [
            {"stage": "freeze_v2", "enabled": not bool(freeze_validation.get("acceptable", False))},
            {"stage": "tushare_proxy_compatibility", "enabled": True},
            {"stage": "tushare_proxy_cutoff", "bootstrap_cutoff": str(as_of_date)},
            {"stage": "trusted_reference_prerequisites", "domains": ["stock-basic", "trade-calendar"]},
            legacy_migration_stage,
            intraday_stage,
            {
                "stage": "quota_tail_reference",
                "priority": 2,
                "domains": ["status", "factor"],
                "skip_domains": ["dividend", "financial"],
            },
            {
                "stage": "trusted_reference_completion",
                "domains": ["daily", "daily-basic", "identity", "status", "factor"],
            },
            *common_tail,
            {"stage": "await_post_cutoff_incremental_publish", "enabled": True, "bootstrap_cutoff": BOOTSTRAP_CUTOFF},
            {
                "stage": "retire_v2_intraday",
                "enabled": False,
                "delete": True,
                "guarded": True,
                "reason": "requires_later_post_cutoff_incremental_publish",
            },
        ]
    else:
        daily_dates, factor_dates = _incremental_trade_date_windows(dates)
        stages = [
            {"stage": "baostock_incremental_daily", "cross_check": False, **_date_task_summary(daily_dates)},
            {"stage": "baostock_incremental_factor_events", **_date_task_summary(factor_dates)},
            {
                "stage": "incremental_intraday_5m",
                "provider": "mootdx_with_baostock_complete_day_fallback",
                **_date_task_summary(daily_dates),
            },
            *common_tail,
            {"stage": "verify_post_cutoff_incremental_publish", "enabled": True, "bootstrap_cutoff": BOOTSTRAP_CUTOFF},
            {"stage": "retire_v2_intraday", "enabled": True, "delete": True, "guarded": True},
        ]
    return {
        "status": "planned",
        "bootstrap": bool(bootstrap),
        "start_date": effective_start,
        "as_of_date": str(as_of_date),
        "historical_provider": normalized_provider,
        "source_paths": list(normalized_source_paths),
        "active_sha256": active_manifest_sha256(workspace_root),
        "calendar_source": calendar_source,
        "stages": stages,
    }


def _mainboard_symbols(security_master: pd.DataFrame) -> list[str]:
    if security_master.empty:
        return []
    column = "symbol" if "symbol" in security_master.columns else "provider_symbol" if "provider_symbol" in security_master.columns else ""
    if not column:
        return []
    provider_symbols = sorted({symbol for symbol in security_master[column].map(normalize_symbol) if symbol})
    registry = SecurityIdentityRegistry.from_sources(provider_symbols=provider_symbols, security_master=security_master)
    identities, history = registry.identity_frames()
    eligible_security_ids = {
        str(security_id)
        for security_id, group in history.groupby("security_id", sort=False)
        if group["symbol"].map(board_for_symbol).eq("MainBoard").any()
        or group["board_on_date"].astype(str).eq("MainBoard").any()
    }
    return sorted(
        {
            normalize_symbol(row.current_symbol)
            for row in identities.itertuples(index=False)
            if str(row.security_id) in eligible_security_ids and normalize_symbol(row.current_symbol)
        }
    )


def _proxy_reference_call(
    *,
    domain: str,
    dates: Iterable[str],
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    workspace_root: str | Path | None,
) -> dict[str, Any]:
    return ingest_tushare_proxy_reference(
        domain=domain,
        start_date=start_date,
        end_date=end_date,
        trade_dates=tuple(dates),
        symbols=tuple(symbols),
        workspace_root=workspace_root,
        resume=True,
        max_workers=3,
    )


def _finish_run(
    state: dict[str, Any],
    *,
    run_path: Path,
    status: str,
    paused_stage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state["status"] = status
    state["finished_at"] = utc_now()
    if paused_stage is not None:
        state["paused_stage"] = paused_stage
    atomic_write_json(run_path, state)
    return {**state, "run_path": str(run_path.resolve())}


def _result_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, dict):
            return dict(payload)
    if is_dataclass(value):
        return dict(asdict(value))
    raise TypeError(f"bootstrap_stage_result_not_serializable:{type(value).__name__}")


def _migrate_v2_intraday_5m(
    *,
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    workspace_root: str | Path | None,
) -> dict[str, Any]:
    """Delay the legacy adapter import so read-only commands stay lightweight."""

    from quant_data_platform.qdp_v3.legacy_v2_5m import migrate_v2_active_intraday_5m

    return _result_payload(
        migrate_v2_active_intraday_5m(
            workspace_root=workspace_root,
            start_date=start_date,
            end_date=end_date,
            symbols=tuple(symbols),
            workers=8,
            strict=True,
            min_available_gib=0.5,
            low_memory_seconds=5.0,
            min_free_disk_gib=200.0,
        )
    )


def _intraday_primary_raw_domains() -> tuple[str, ...]:
    """Ordered complete-day evidence used before the Tushare residual."""

    return (
        RAW_EXTERNAL_QUANT_INTRADAY_5M,
        RAW_LEGACY_V2_INTRADAY_5M,
        RAW_INTRADAY_5M_SELECTED,
    )


def _intraday_compact_raw_domains() -> tuple[str, ...]:
    return (
        RAW_EXTERNAL_QUANT_INTRADAY_5M,
        RAW_LEGACY_V2_INTRADAY_5M,
        RAW_INTRADAY_5M_MOOTDX,
        RAW_INTRADAY_5M_BAOSTOCK,
        RAW_INTRADAY_5M_SELECTED,
        RAW_TUSHARE_PROXY_INTRADAY_5M,
    )


def _residual_route_scope(
    residual: dict[str, Any],
    *,
    trade_dates: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    symbols = tuple(sorted(set(residual.get("download_symbols", ()) or ())))
    available_dates = tuple(sorted({_normalize_date(item) for item in trade_dates if _normalize_date(item)}))
    route_dates: set[str] = set()
    for group in residual.get("download_groups", ()) or ():
        explicit = {_normalize_date(item) for item in group.get("trade_dates", ()) or () if _normalize_date(item)}
        if explicit:
            route_dates.update(explicit)
            continue
        group_start = str(group.get("start_date", "") or "")[:10]
        group_end = str(group.get("end_date", "") or "")[:10]
        if group_start and group_end:
            route_dates.update(item for item in available_dates if group_start <= item <= group_end)
    return symbols, tuple(sorted(route_dates))


def _residual_archive_batches(
    residual: dict[str, Any],
    *,
    trade_dates: Iterable[str],
) -> list[dict[str, Any]]:
    """Keep partial-coverage imports exact without re-indexing per security.

    Securities with no positive source coverage can share one broad envelope:
    every date in that envelope is still residual for them.  Partially covered
    securities remain grouped by their exact missing-date set so a short gap
    does not turn back into a full-history archive parse/output revision.
    """

    available_dates = tuple(sorted({_normalize_date(item) for item in trade_dates if _normalize_date(item)}))
    download_symbols = set(residual.get("download_symbols", ()) or ())
    partial_symbols = download_symbols.intersection(
        set(residual.get("partially_covered_symbols", ()) or ())
    )
    fully_uncovered_symbols = download_symbols.difference(partial_symbols)
    full_dates: set[str] = set()
    partial_groups: dict[tuple[str, ...], set[str]] = {}
    for group in residual.get("download_groups", ()) or ():
        group_symbols = set(group.get("symbols", ()) or ())
        explicit_dates = tuple(
            sorted({_normalize_date(item) for item in group.get("trade_dates", ()) or () if _normalize_date(item)})
        )
        if not explicit_dates:
            group_start = str(group.get("start_date", "") or "")[:10]
            group_end = str(group.get("end_date", "") or "")[:10]
            explicit_dates = tuple(item for item in available_dates if group_start <= item <= group_end)
        if not explicit_dates:
            continue
        if group_symbols.intersection(fully_uncovered_symbols):
            full_dates.update(explicit_dates)
        partial = group_symbols.intersection(partial_symbols)
        if partial:
            partial_groups.setdefault(explicit_dates, set()).update(partial)

    batches: list[dict[str, Any]] = []
    if fully_uncovered_symbols:
        dates = tuple(sorted(full_dates)) or available_dates
        if dates:
            batches.append(
                {
                    "kind": "fully_uncovered_envelope",
                    "symbols": tuple(sorted(fully_uncovered_symbols)),
                    "trade_dates": dates,
                    "start_date": dates[0],
                    "end_date": dates[-1],
                }
            )
    for dates, symbols in sorted(partial_groups.items()):
        batches.append(
            {
                "kind": "partial_exact_gap",
                "symbols": tuple(sorted(symbols)),
                "trade_dates": dates,
                "start_date": dates[0],
                "end_date": dates[-1],
            }
        )
    return batches


def _plan_bootstrap_intraday_residual(
    *,
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    workspace_root: str | Path | None,
    lifecycle_ranges: dict[str, tuple[str, str]],
    trade_dates: Iterable[str],
    expected_trade_dates_by_symbol: dict[str, tuple[str, ...]] | None = None,
    treat_tushare_empty_as_known: bool,
) -> dict[str, Any]:
    return plan_intraday_residuals(
        symbols=tuple(symbols),
        start_date=start_date,
        end_date=end_date,
        primary_raw_domains=_intraday_primary_raw_domains(),
        fallback_raw_domain=RAW_TUSHARE_PROXY_INTRADAY_5M,
        workspace_root=workspace_root,
        lifecycle_ranges=lifecycle_ranges,
        trade_dates=tuple(trade_dates),
        expected_trade_dates_by_symbol=expected_trade_dates_by_symbol,
        treat_fallback_empty_as_known=treat_tushare_empty_as_known,
    )


def _run_bootstrap_capture(
    *,
    plan: dict[str, Any],
    record: Callable[[str, dict[str, Any]], None],
    workspace_root: str | Path | None,
) -> tuple[str, dict[str, Any] | None]:
    as_of_date = str(plan["as_of_date"])
    start_date = str(plan["start_date"])
    compatibility = run_tushare_proxy_compatibility_gate(
        workspace_root=workspace_root,
        as_of_date=as_of_date,
        smoke=False,
        reuse_passed=True,
    )
    record("tushare_proxy_compatibility", compatibility)
    if compatibility.get("status") != "passed":
        raise RuntimeError("tushare_proxy_compatibility_gate_failed")
    cutoff = lock_bootstrap_cutoff(requested_cutoff=as_of_date, workspace_root=workspace_root)
    record("tushare_proxy_cutoff", cutoff)
    cutoff_date = str(cutoff["bootstrap_cutoff"])
    for domain in ("stock-basic", "trade-calendar"):
        result = _proxy_reference_call(
            domain=domain,
            dates=(),
            symbols=(),
            start_date=start_date,
            end_date=cutoff_date,
            workspace_root=workspace_root,
        )
        record(f"tushare_proxy_{domain}", result)
        if result.get("status") != "completed":
            return str(result.get("status") or "partial"), result
    dates, _ = _calendar_dates_read_only(
        start_date=start_date,
        end_date=cutoff_date,
        workspace_root=workspace_root,
        prefer_proxy=True,
    )
    all_symbols, _ = proxy_symbol_inventory(workspace_root=workspace_root, mainboard_only=False)
    mainboard_symbols, lifecycle = proxy_symbol_inventory(workspace_root=workspace_root, mainboard_only=True)
    if not dates or not all_symbols or not mainboard_symbols:
        raise RuntimeError("trusted_bootstrap_inventory_or_calendar_missing")

    # The readable v2 leaf is valuable migration input even though its lost
    # ancestors cannot be represented as complete source lineage.  Reuse it
    # before touching local archives or consuming any minute entitlement.
    try:
        migration = _migrate_v2_intraday_5m(
            symbols=mainboard_symbols,
            start_date=start_date,
            end_date=cutoff_date,
            workspace_root=workspace_root,
        )
    except Exception as exc:
        if type(exc).__name__.endswith("ResourceGuardError"):
            migration = {
                "status": "paused_resource_guard",
                "stage": "legacy_v2_intraday_5m_migration",
                "error_type": type(exc).__name__,
            }
            record("legacy_v2_intraday_5m_migration", migration)
            return "paused_resource_guard", migration
        raise
    migration.setdefault("status", "completed")
    record("legacy_v2_intraday_5m_migration", migration)
    if str(migration.get("status", "")) not in {"completed", "success"}:
        return str(migration.get("status") or "partial"), migration

    # Exact stock-day residuals require a complete trusted daily/status base so
    # legitimate suspensions are classified locally instead of sent to minute
    # providers. These jobs are resumable and normally reuse prior captures.
    for domain in ("daily", "status"):
        result = _proxy_reference_call(
            domain=domain,
            dates=dates,
            symbols=(),
            start_date=start_date,
            end_date=cutoff_date,
            workspace_root=workspace_root,
        )
        record(f"tushare_proxy_pre_intraday_{domain}", result)
        if result.get("status") != "completed":
            return str(result.get("status") or "partial"), result
    identity_registry = SecurityIdentityRegistry.from_sources(
        provider_symbols=all_symbols,
        workspace_root=workspace_root,
    )
    intraday_status = "completed"
    intraday_result: dict[str, Any] | None = None
    historical_provider = str(plan.get("historical_provider", TRUSTED_BOOTSTRAP_PROVIDER) or TRUSTED_BOOTSTRAP_PROVIDER)
    waves = (
        (start_date, min("2019-12-31", cutoff_date), "2010_2019"),
        ("2020-01-01", cutoff_date, "2020_cutoff"),
    )
    for wave_start, wave_end, wave_name in waves:
        if wave_start > wave_end:
            continue
        wave_dates = tuple(item for item in dates if wave_start <= item <= wave_end)
        residual = _plan_bootstrap_intraday_residual(
            symbols=mainboard_symbols,
            start_date=wave_start,
            end_date=wave_end,
            workspace_root=workspace_root,
            lifecycle_ranges=lifecycle,
            trade_dates=wave_dates,
            # A historical successful-empty Tushare receipt must not suppress
            # a viable local/free-source attempt.
            treat_tushare_empty_as_known=False,
        )
        record(f"intraday_5m_residual_after_v2_{wave_name}", residual)

        if historical_provider == LOCAL_ARCHIVE_BOOTSTRAP_PROVIDER and int(
            residual.get("download_count", 0) or 0
        ):
            from quant_data_platform.qdp_v3.external_quant_5m import import_external_quant_5m

            archive_batches = _residual_archive_batches(residual, trade_dates=wave_dates)
            completed_batches: list[dict[str, Any]] = []
            for batch_index, batch in enumerate(archive_batches, start=1):
                local_symbols = tuple(batch["symbols"])
                local_dates = tuple(batch["trade_dates"])
                try:
                    local_result = _result_payload(
                        import_external_quant_5m(
                            source_paths=tuple(plan.get("source_paths", ()) or ()),
                            symbols=local_symbols,
                            workspace_root=workspace_root,
                            raw_domain=RAW_EXTERNAL_QUANT_INTRADAY_5M,
                            workers=8,
                            start_date=local_dates[0],
                            end_date=local_dates[-1],
                            min_available_gib=0.5,
                            min_free_disk_gib=200.0,
                        )
                    )
                except Exception as exc:
                    if type(exc).__name__.endswith("ResourceGuardError"):
                        local_result = {
                            "status": "paused_resource_guard",
                            "stage": "external_quant_archive_intraday_5m_residual",
                            "error_type": type(exc).__name__,
                            "batch_index": batch_index,
                            "batch_kind": str(batch["kind"]),
                        }
                        record(
                            f"external_quant_archive_intraday_5m_residual_{wave_name}_{batch_index}",
                            local_result,
                        )
                        return "paused_resource_guard", local_result
                    raise
                local_result.setdefault("status", "completed")
                local_result["residual_batch_index"] = batch_index
                local_result["residual_batch_kind"] = str(batch["kind"])
                local_result["residual_input_symbol_count"] = len(local_symbols)
                local_result["residual_input_trade_date_count"] = len(local_dates)
                record(
                    f"external_quant_archive_intraday_5m_residual_{wave_name}_{batch_index}",
                    local_result,
                )
                local_status = str(local_result.get("status") or "completed")
                if local_status not in {"completed", "success"}:
                    return local_status, local_result
                completed_batches.append(
                    {
                        "batch_index": batch_index,
                        "kind": str(batch["kind"]),
                        "symbol_count": len(local_symbols),
                        "trade_date_count": len(local_dates),
                        "start_date": local_dates[0],
                        "end_date": local_dates[-1],
                    }
                )
            if completed_batches:
                record(
                    f"external_quant_archive_intraday_5m_residual_{wave_name}",
                    {
                        "status": "completed",
                        "batch_count": len(completed_batches),
                        "batches": completed_batches,
                    },
                )
                residual = _plan_bootstrap_intraday_residual(
                    symbols=mainboard_symbols,
                    start_date=wave_start,
                    end_date=wave_end,
                    workspace_root=workspace_root,
                    lifecycle_ranges=lifecycle,
                    trade_dates=wave_dates,
                    treat_tushare_empty_as_known=False,
                )
                record(f"intraday_5m_residual_after_local_{wave_name}", residual)

        route_symbols, route_dates = _residual_route_scope(residual, trade_dates=wave_dates)
        expected_trade_dates: dict[str, tuple[str, ...]] = {}
        if route_symbols and route_dates:
            expected_trade_dates, expected_evidence = _proxy_daily_expected_trade_dates(
                symbols=route_symbols,
                trade_dates=route_dates,
                identity_registry=identity_registry,
                workspace_root=workspace_root,
            )
            record(f"intraday_5m_expected_stock_days_{wave_name}", expected_evidence)
            residual = _plan_bootstrap_intraday_residual(
                symbols=route_symbols,
                start_date=wave_start,
                end_date=wave_end,
                workspace_root=workspace_root,
                lifecycle_ranges=lifecycle,
                trade_dates=route_dates,
                expected_trade_dates_by_symbol=expected_trade_dates or None,
                # There is no additional pre-Tushare source for the older
                # wave, so a prior successful-empty response is a stable gap.
                treat_tushare_empty_as_known=wave_start < "2020-01-01",
            )
        record(f"intraday_5m_residual_exact_{wave_name}", residual)

        # Free sources are useful for the recent range, but only complete
        # stock-days enter RAW_INTRADAY_5M_SELECTED.  No cross-source values are
        # compared and no partial day is spliced into another provider.
        if wave_start >= "2020-01-01":
            for group_index, group in enumerate(residual.get("download_groups", ()) or (), start=1):
                group_start = str(group.get("start_date", "") or "")[:10]
                group_end = str(group.get("end_date", "") or "")[:10]
                group_symbols = tuple(group.get("symbols", ()) or ())
                group_dates = list(group.get("trade_dates", ()) or ())
                if not group_dates:
                    group_dates = [item for item in route_dates if group_start <= item <= group_end]
                if not group_symbols or not group_dates:
                    continue
                free_result = ingest_intraday_5m(
                    symbols=group_symbols,
                    trade_dates=group_dates,
                    workspace_root=workspace_root,
                    refresh=False,
                    audit_cross_sources=False,
                )
                record(f"free_intraday_5m_residual_{wave_name}_{group_index}", free_result)
            residual = _plan_bootstrap_intraday_residual(
                symbols=route_symbols,
                start_date=wave_start,
                end_date=wave_end,
                workspace_root=workspace_root,
                lifecycle_ranges=lifecycle,
                trade_dates=route_dates,
                expected_trade_dates_by_symbol=expected_trade_dates,
                treat_tushare_empty_as_known=True,
            )
            record(f"intraday_5m_residual_after_free_{wave_name}", residual)

        for group_index, group in enumerate(residual.get("download_groups", ()) or (), start=1):
            residual_start = str(group.get("start_date", "") or "")[:10]
            residual_end = str(group.get("end_date", "") or "")[:10]
            residual_symbols = tuple(group.get("symbols", ()) or ())
            if not residual_start or not residual_end or not residual_symbols:
                continue
            residual_hash = stable_hash(
                {"start": residual_start, "end": residual_end, "symbols": residual_symbols},
                length=16,
            )
            intraday_result = ingest_tushare_proxy_intraday(
                symbols=residual_symbols,
                start_date=residual_start,
                end_date=residual_end,
                workspace_root=workspace_root,
                lifecycle_ranges=lifecycle,
                resume=True,
                job_id=(
                    "tushare_proxy__intraday_5m__final_residual_"
                    f"{wave_name}_{residual_start.replace('-', '')}_"
                    f"{residual_end.replace('-', '')}_{residual_hash}"
                ),
                max_workers=3,
            )
            record(f"tushare_proxy_intraday_5m_residual_{wave_name}_{group_index}", intraday_result)
            intraday_status = str(intraday_result.get("status") or "partial")
            if intraday_status != "completed":
                break
        if intraday_status != "completed":
            break
        final_residual = _plan_bootstrap_intraday_residual(
            symbols=route_symbols,
            start_date=wave_start,
            end_date=wave_end,
            workspace_root=workspace_root,
            lifecycle_ranges=lifecycle,
            trade_dates=route_dates,
            expected_trade_dates_by_symbol=expected_trade_dates,
            treat_tushare_empty_as_known=True,
        )
        record(f"intraday_5m_residual_final_{wave_name}", final_residual)
        if int(final_residual.get("download_count", 0) or 0):
            return "partial", final_residual
    # Minute quota is the scarce entitlement. Once exhausted, use the
    # remaining low-frequency capacity only for core status and factor facts.
    if intraday_status == "paused_quota":
        for domain in ("status", "factor"):
            result = _proxy_reference_call(
                domain=domain,
                dates=dates if domain == "status" else (),
                symbols=all_symbols if domain == "factor" else (),
                start_date=start_date,
                end_date=cutoff_date,
                workspace_root=workspace_root,
            )
            record(f"tushare_proxy_quota_tail_{domain}", result)
        return intraday_status, intraday_result
    if intraday_status != "completed":
        return intraday_status, intraday_result
    for domain in ("daily-basic", "identity", "factor"):
        result = _proxy_reference_call(
            domain=domain,
            dates=dates if domain in {"daily", "daily-basic", "status"} else (),
            symbols=all_symbols if domain in {"identity", "factor"} else (),
            start_date=start_date,
            end_date=cutoff_date,
            workspace_root=workspace_root,
        )
        record(f"tushare_proxy_{domain}", result)
        if result.get("status") != "completed":
            return str(result.get("status") or "partial"), result
    # The minute capture paths are intentionally resumable per symbol/month.
    # Before candidate construction, collapse those many small immutable files
    # into verified year x 16-bucket bundles and export a portable raw index.
    # Source archives and v2 are untouched; only verified v3 legacy partitions
    # are removed after their bundle/hash round-trip succeeds.
    compact_results: list[dict[str, Any]] = []
    for raw_domain in _intraday_compact_raw_domains():
        compacted = compact_raw_domain(
            raw_domain,
            workspace_root=workspace_root,
            export_index_dir=qdp_v3_paths(workspace_root).metadata / "raw_index",
            delete_sources=True,
            yes=True,
        )
        compact_results.append(asdict(compacted))
    record(
        "compact_intraday_raw",
        {
            "status": "completed",
            "domain_count": len(compact_results),
            "results": compact_results,
        },
    )
    return "completed", None


def _run_incremental_capture(
    *,
    plan: dict[str, Any],
    run_id: str,
    record: Callable[[str, dict[str, Any]], None],
    workspace_root: str | Path | None,
) -> None:
    calendar, _ = ingest_trading_calendar(
        start_date=str(plan["start_date"]),
        end_date=str(plan["as_of_date"]),
        workspace_root=workspace_root,
        refresh=True,
    )
    opened = calendar["is_open"].astype(str).str.lower().isin({"1", "true", "t", "yes"})
    dates = sorted(set(calendar.loc[opened, "trade_date"].map(_normalize_date)))
    daily_dates, factor_dates = _incremental_trade_date_windows(
        dates,
        require_full_factor_window=True,
    )
    security_ref = ingest_security_master(
        as_of_date=str(plan["as_of_date"]),
        workspace_root=workspace_root,
        refresh=True,
    )
    security_master = read_raw_partition(security_ref)
    record(
        "security_master",
        {"status": "completed", "content_sha256": security_ref.content_sha256, "row_count": security_ref.row_count},
    )
    daily = ingest_baostock_date_partitions(
        mode="date-snapshot",
        trade_dates=daily_dates,
        workspace_root=workspace_root,
        refresh=True,
        cross_check=False,
        security_master=security_master,
        job_id=f"{run_id}__daily",
    )
    record("baostock_incremental_daily", daily)
    if daily.get("failed_count"):
        raise RuntimeError("baostock_incremental_daily_failed")
    factor = ingest_baostock_date_partitions(
        mode="date-events",
        trade_dates=factor_dates,
        workspace_root=workspace_root,
        refresh=True,
        cross_check=False,
        job_id=f"{run_id}__factor",
    )
    record("baostock_incremental_factor_events", factor)
    if factor.get("failed_count"):
        raise RuntimeError("baostock_incremental_factor_events_failed")
    event_symbols: set[str] = set()
    for ref in iter_raw_partitions(
        RAW_ADJUST_FACTOR_EVENT,
        workspace_root=workspace_root,
        start_value=max(BOOTSTRAP_CUTOFF, factor_dates[0]),
        end_value=factor_dates[-1],
    ):
        if str(ref.partition_value) <= BOOTSTRAP_CUTOFF:
            continue
        event_frame = read_raw_partition(ref)
        if "code" in event_frame.columns:
            event_symbols.update(
                normalize_symbol(value)
                for value in event_frame["code"].astype(str)
                if normalize_symbol(value)
            )
    if event_symbols:
        factor_history = ingest_factor_symbol_histories(
            symbols=sorted(event_symbols),
            start_date="1990-01-01",
            end_date=str(plan["as_of_date"]),
            workspace_root=workspace_root,
            refresh=True,
            job_id=f"{run_id}__factor_history",
        )
        record("baostock_incremental_factor_history_anchors", factor_history)
        if factor_history.get("failed_count"):
            raise RuntimeError("baostock_incremental_factor_history_anchor_failed")
    intraday = ingest_intraday_5m(
        symbols=_mainboard_symbols(security_master),
        trade_dates=daily_dates,
        workspace_root=workspace_root,
        refresh=True,
        audit_cross_sources=False,
    )
    record("incremental_intraday_5m", intraday)


def run_update(
    *,
    as_of_date: str,
    workspace_root: str | Path | None = None,
    bootstrap: bool = False,
    start_date: str = "",
    historical_provider: str = "",
    source_paths: Iterable[str | Path] = (),
    publish: bool = True,
    hash_v2_shards: bool = False,
) -> dict[str, Any]:
    paths = ensure_qdp_v3_layout(workspace_root)
    requeue_stale_jobs(workspace_root=workspace_root)
    plan = plan_update(
        as_of_date=as_of_date,
        workspace_root=workspace_root,
        bootstrap=bootstrap,
        start_date=start_date,
        historical_provider=historical_provider,
        source_paths=source_paths,
    )
    if plan.get("status") != "planned":
        return plan
    run_id = (
        f"bootstrap__{stable_hash({'as_of_date': as_of_date, 'start_date': plan['start_date'], 'historical_provider': plan['historical_provider'], 'source_paths': plan.get('source_paths', [])}, length=20)}"
        if bootstrap
        else f"update__{stable_hash({'as_of_date': as_of_date}, length=20)}"
    )
    run_path = paths.jobs / f"{run_id}.json"
    with supervise_job(run_id, workspace_root=workspace_root):
        previous = read_json(run_path)
        state: dict[str, Any] = {
            "status": "running",
            "run_id": run_id,
            "plan": plan,
            "stages": [],
            "started_at": utc_now(),
            "attempt": int(previous.get("attempt", 0) or 0) + 1,
            "resumed_from_status": str(previous.get("status", "") or ""),
        }
        atomic_write_json(run_path, state)

        def record(stage: str, result: dict[str, Any]) -> None:
            state["stages"].append({"stage": stage, "result": result})
            state["updated_at"] = utc_now()
            atomic_write_json(run_path, state)

        try:
            freeze_path = paths.metadata / "v2_freeze_20260713.json"
            freeze_state = read_json(freeze_path) if freeze_path.exists() else {}
            if not validate_v2_freeze_proof(freeze_state).get("acceptable", False):
                freeze_result = freeze_v2(workspace_root=workspace_root, hash_shards=bool(hash_v2_shards))
                record("freeze_v2", freeze_result)
                if freeze_result.get("status") == "metadata_only" and not hash_v2_shards:
                    freeze_result = freeze_v2(workspace_root=workspace_root, hash_shards=True)
                    record("freeze_v2_full_hash", freeze_result)
                if not validate_v2_freeze_proof(read_json(freeze_path)).get("acceptable", False):
                    raise RuntimeError(f"v2_freeze_stage_incomplete:{freeze_result.get('status')}")
            if bootstrap:
                capture_status, paused_stage = _run_bootstrap_capture(
                    plan=plan,
                    record=record,
                    workspace_root=workspace_root,
                )
                if capture_status != "completed":
                    return _finish_run(
                        state,
                        run_path=run_path,
                        status=capture_status,
                        paused_stage=paused_stage,
                    )
            else:
                _run_incremental_capture(
                    plan=plan,
                    run_id=run_id,
                    record=record,
                    workspace_root=workspace_root,
                )
            candidate = build_candidate(
                workspace_root=workspace_root,
                start_date=TRUSTED_BOOTSTRAP_START,
                end_date=str(plan["as_of_date"]),
                require_factor_dual_path=False,
            )
            record(
                "build_candidate",
                {"status": "completed", "candidate_id": candidate.candidate_id, "blocker_count": len(candidate.blockers)},
            )
            audit = audit_candidate(
                candidate_id=candidate.candidate_id,
                mode="semantic",
                workspace_root=workspace_root,
            )
            record("semantic_audit", audit)
            candidate_diff = diff_candidate(candidate.candidate_id, workspace_root=workspace_root, against="active")
            record("diff_candidate", candidate_diff)
            atomic_write_json(paths.audits / candidate.candidate_id / "v2_v3_diff.json", candidate_diff)
            state["diff"] = candidate_diff
            if publish and audit.get("status") == "passed":
                publish_result = publish_candidate(
                    candidate_id=candidate.candidate_id,
                    expect_active_sha=str(plan["active_sha256"]),
                    workspace_root=workspace_root,
                )
                record("cas_publish", publish_result)
            elif publish:
                publish_result = {"status": "blocked", "reason": "semantic_audit_failed"}
            else:
                publish_result = {"status": "not_requested"}
            state["candidate_id"] = candidate.candidate_id
            state["publish"] = publish_result
            if publish_result.get("status") == "published":
                post_publish = audit_candidate(
                    candidate_id=candidate.candidate_id,
                    mode="semantic",
                    workspace_root=workspace_root,
                )
                record("post_publish_semantic_hash_audit", post_publish)
                if post_publish.get("status") != "passed":
                    return _finish_run(state, run_path=run_path, status="published_validation_failed")
                retirement_gate = _advance_retirement_gate(
                    paths=paths,
                    bootstrap=bootstrap,
                    run_id=run_id,
                    candidate_id=candidate.candidate_id,
                    active_sha256=str(publish_result.get("active_sha256", "") or ""),
                    active=read_json(paths.active_manifest),
                )
                record("post_cutoff_incremental_retirement_gate", retirement_gate)
                state["retirement_gate"] = retirement_gate
                if bootstrap:
                    final_status = "published_awaiting_incremental_update"
                elif retirement_gate.get("status") == "incremental_verified":
                    state["status"] = "published_pending_cleanup"
                    atomic_write_json(run_path, state)
                    retirement = retire_v2_intraday(
                        expect_active_sha=str(publish_result.get("active_sha256", "") or ""),
                        workspace_root=workspace_root,
                        delete=True,
                        yes=True,
                    )
                    record("retire_v2_intraday", retirement)
                    state["retirement"] = retirement
                    final_status = "completed" if retirement.get("status") == "retired" else "published_cleanup_blocked"
                elif retirement_gate.get("status") == "incremental_verification_blocked":
                    final_status = "published_incremental_verification_blocked"
                else:
                    final_status = "completed"
            elif publish_result.get("status") == "not_requested":
                final_status = "completed"
            else:
                final_status = "blocked"
            return _finish_run(state, run_path=run_path, status=final_status)
        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
            return _finish_run(state, run_path=run_path, status="failed")
