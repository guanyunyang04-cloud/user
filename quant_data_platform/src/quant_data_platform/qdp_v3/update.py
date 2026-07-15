from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from quant_data_platform.core.json_io import read_json
from quant_data_platform.qdp_v3.audit import audit_candidate
from quant_data_platform.qdp_v3.build import build_candidate
from quant_data_platform.qdp_v3.constants import (
    BOOTSTRAP_CUTOFF,
    RAW_ADJUST_FACTOR_EVENT,
    RAW_TRADING_CALENDAR,
    RAW_TUSHARE_PROXY_TRADE_CALENDAR,
    V2_RETIREMENT_GATE_FILENAME,
)
from quant_data_platform.qdp_v3.freeze import freeze_v2, validate_v2_freeze_proof
from quant_data_platform.qdp_v3.historical import (
    ingest_tushare_proxy_intraday,
    ingest_tushare_proxy_reference,
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
TRUSTED_BOOTSTRAP_START = "2010-01-01"
INCREMENTAL_CALENDAR_LOOKBACK_DAYS = 120
INCREMENTAL_DAILY_REFRESH_DAYS = 10
INCREMENTAL_FACTOR_REFRESH_DAYS = 60


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
    normalized_provider = str(historical_provider or (TRUSTED_BOOTSTRAP_PROVIDER if bootstrap else "")).strip().lower()
    if bootstrap and normalized_provider != TRUSTED_BOOTSTRAP_PROVIDER:
        return {
            "status": "blocked",
            "blocker": "qdp_v3_bootstrap_requires_trusted_tushare_proxy",
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
        stages = [
            {"stage": "freeze_v2", "enabled": not bool(freeze_validation.get("acceptable", False))},
            {"stage": "tushare_proxy_compatibility", "enabled": True},
            {"stage": "tushare_proxy_cutoff", "bootstrap_cutoff": str(as_of_date)},
            {"stage": "trusted_reference_prerequisites", "domains": ["stock-basic", "trade-calendar"]},
            {
                "stage": "intraday_5m",
                "provider": "tushare_proxy",
                "priority": 1,
                **_date_task_summary(dates),
            },
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
    intraday_status = "completed"
    intraday_result: dict[str, Any] | None = None
    for wave_start, wave_end, wave_name in (
        (start_date, min("2019-12-31", cutoff_date), "2010_2019"),
        ("2020-01-01", cutoff_date, "2020_cutoff"),
    ):
        if wave_start > wave_end:
            continue
        intraday_result = ingest_tushare_proxy_intraday(
            symbols=mainboard_symbols,
            start_date=wave_start,
            end_date=wave_end,
            workspace_root=workspace_root,
            lifecycle_ranges=lifecycle,
            resume=True,
            job_id=f"tushare_proxy__intraday_5m__bootstrap_{wave_name}_{cutoff_date.replace('-', '')}",
            max_workers=3,
        )
        record(f"tushare_proxy_intraday_5m_{wave_name}", intraday_result)
        intraday_status = str(intraday_result.get("status") or "partial")
        if intraday_status != "completed":
            break
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
    for domain in ("daily", "daily-basic", "identity", "status", "factor"):
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
    )
    if plan.get("status") != "planned":
        return plan
    run_id = (
        f"bootstrap__{stable_hash({'as_of_date': as_of_date, 'start_date': plan['start_date'], 'historical_provider': plan['historical_provider']}, length=20)}"
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
