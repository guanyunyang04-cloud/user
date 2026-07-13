from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quant_data_platform.core.json_io import read_json
from quant_data_platform.qdp_v3.audit import audit_candidate
from quant_data_platform.qdp_v3.build import build_candidate
from quant_data_platform.qdp_v3.corporate_actions import ingest_mootdx_xdxr
from quant_data_platform.qdp_v3.freeze import freeze_v2
from quant_data_platform.qdp_v3.identity import SecurityIdentityRegistry, board_for_symbol, normalize_symbol
from quant_data_platform.qdp_v3.ingest import (
    ingest_baostock_date_partitions,
    ingest_factor_symbol_histories,
    ingest_security_master,
    ingest_trading_calendar,
)
from quant_data_platform.qdp_v3.intraday import ingest_intraday_5m
from quant_data_platform.qdp_v3.constants import (
    RAW_ADJUST_FACTOR_EVENT,
    RAW_ADJUST_FACTOR_SYMBOL_HISTORY,
    RAW_CORPORATE_ACTION_XDXR,
    RAW_DAILY_ASTOCK,
)
from quant_data_platform.qdp_v3.manifest import active_manifest_sha256, atomic_write_json, stable_hash, utc_now
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout, qdp_v3_paths
from quant_data_platform.qdp_v3.release import diff_candidate, publish_candidate
from quant_data_platform.qdp_v3.secondary import ingest_baostock_report_domain, ingest_baostock_snapshot_domain
from quant_data_platform.qdp_v3.storage import iter_raw_partitions, read_raw_partition


def _calendar_dates_read_only(*, start_date: str, end_date: str, workspace_root: str | Path | None) -> tuple[list[str], str]:
    paths = qdp_v3_paths(workspace_root)
    frames: list[pd.DataFrame] = []
    calendar_root = paths.raw / "baostock_trading_calendar_raw"
    if calendar_root.exists():
        for latest_path in sorted(calendar_root.glob("*/latest.json")):
            latest = read_json(latest_path)
            version_path = Path(str(latest.get("version_path", "") or ""))
            if not version_path.is_absolute():
                version_path = paths.root / version_path
            payload = version_path / "payload.parquet"
            if payload.exists():
                frames.append(pd.read_parquet(payload, engine="pyarrow"))
    if frames:
        calendar = pd.concat(frames, ignore_index=True)
        calendar["trade_date"] = calendar["trade_date"].astype(str).str.slice(0, 10)
        opened = calendar["is_open"].astype(str).str.lower().isin({"1", "true", "t", "yes"})
        dates = calendar.loc[opened & calendar["trade_date"].between(start_date, end_date), "trade_date"].tolist()
        return sorted(set(dates)), "stored_baostock_calendar"
    dates = [item.strftime("%Y-%m-%d") for item in pd.bdate_range(start_date, end_date)]
    return dates, "business_day_fallback_dry_run_only"


def plan_update(
    *,
    as_of_date: str,
    workspace_root: str | Path | None = None,
    bootstrap: bool = False,
    start_date: str = "",
    include_5m: bool = True,
    include_secondary: bool = True,
) -> dict[str, Any]:
    paths = qdp_v3_paths(workspace_root)
    active = read_json(paths.active_manifest)
    active_end = str(dict(active.get("coverage", {}) or {}).get("end_date", "") or "")
    if not active and not bootstrap:
        return {
            "status": "blocked",
            "blocker": "qdp_v3_bootstrap_requires_explicit_bootstrap",
            "message": "No v3 active exists. Use --bootstrap --start-date 2010-01-01 for the initial rebuild.",
            "active_sha256": active_manifest_sha256(workspace_root),
        }
    effective_start = str(start_date or (active_end if active_end else "2010-01-01"))
    dates, calendar_source = _calendar_dates_read_only(start_date=effective_start, end_date=str(as_of_date), workspace_root=workspace_root)
    daily_dates = dates if bootstrap else dates[-10:]
    factor_dates = dates if bootstrap else dates[-60:]
    intraday_dates = [date for date in dates if date >= "2020-01-01"] if bootstrap and include_5m else dates[-1:] if include_5m and dates else []
    freeze_state = read_json(paths.metadata / "v2_freeze_20260713.json")
    return {
        "status": "planned",
        "bootstrap": bool(bootstrap),
        "start_date": effective_start,
        "as_of_date": str(as_of_date),
        "active_sha256": active_manifest_sha256(workspace_root),
        "calendar_source": calendar_source,
        "stages": [
            {"stage": "freeze_v2", "enabled": freeze_state.get("status") != "complete", "current_status": freeze_state.get("status", "missing"), "missing_dataset_count": len(freeze_state.get("missing_dataset_ids", []) or [])},
            {"stage": "calendar", "start_date": effective_start, "end_date": str(as_of_date)},
            {"stage": "security_master", "as_of_date": str(as_of_date)},
            {"stage": "daily_date_snapshot", "task_count": len(daily_dates), "dates": daily_dates},
            {"stage": "factor_date_events", "task_count": len(factor_dates), "dates": factor_dates},
            {"stage": "factor_symbol_history", "scope": "all securities once; resumable"},
            {"stage": "corporate_action_xdxr", "scope": "all securities on bootstrap; recent factor-event securities on incremental updates"},
            {"stage": "secondary_pit", "enabled": bool(include_secondary), "domains": ["financial_quarterly", "performance_forecast", "performance_express", "industry_concept", "index_constituents"]},
            {"stage": "intraday_5m", "enabled": bool(include_5m), "trade_date_count": len(intraday_dates), "dates": intraday_dates},
            {"stage": "build_candidate"},
            {"stage": "semantic_audit"},
            {"stage": "diff_candidate", "against": "active"},
            {"stage": "cas_publish"},
        ],
    }


def _mainboard_symbols(security_master: pd.DataFrame) -> list[str]:
    if security_master.empty:
        return []
    column = "symbol" if "symbol" in security_master.columns else "provider_symbol" if "provider_symbol" in security_master.columns else ""
    if not column:
        return []
    provider_symbols = sorted({symbol for symbol in security_master[column].map(normalize_symbol) if symbol})
    registry = SecurityIdentityRegistry.from_sources(
        provider_symbols=provider_symbols,
        security_master=security_master,
    )
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


def _all_stock_symbols(security_master: pd.DataFrame) -> list[str]:
    if security_master.empty:
        return []
    column = "symbol" if "symbol" in security_master.columns else "provider_symbol" if "provider_symbol" in security_master.columns else ""
    if not column:
        return []
    symbols = security_master[column].map(normalize_symbol)
    return sorted({symbol for symbol in symbols if board_for_symbol(symbol) in {"MainBoard", "ChiNext", "STAR"}})


def _recent_factor_event_symbols(*, trade_dates: list[str], workspace_root: str | Path | None) -> list[str]:
    if not trade_dates:
        return []
    refs = iter_raw_partitions(
        RAW_ADJUST_FACTOR_EVENT,
        workspace_root=workspace_root,
        start_value=min(trade_dates),
        end_value=max(trade_dates),
    )
    symbols: set[str] = set()
    for ref in refs:
        raw = read_raw_partition(ref)
        if "code" in raw.columns:
            symbols.update(normalize_symbol(item) for item in raw["code"] if normalize_symbol(item))
    return sorted(symbols)


def _missing_symbol_partitions(
    *,
    symbols: list[str],
    raw_domain: str,
    workspace_root: str | Path | None,
) -> list[str]:
    existing = {
        normalize_symbol(ref.partition_value)
        for ref in iter_raw_partitions(raw_domain, workspace_root=workspace_root)
        if normalize_symbol(ref.partition_value)
    }
    return sorted(set(symbols) - existing)


def _monthly_intraday_sampling_universe(
    *,
    trade_dates: list[str],
    workspace_root: str | Path | None,
) -> pd.DataFrame:
    refs = iter_raw_partitions(RAW_DAILY_ASTOCK, workspace_root=workspace_root)
    if not refs:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for month in sorted({date[:7] for date in trade_dates}):
        first_day = f"{month}-01"
        prior = [ref for ref in refs if ref.partition_value < first_day]
        current = [ref for ref in refs if ref.partition_value.startswith(month)]
        anchor = prior[-1] if prior else current[0] if current else None
        if anchor is None:
            continue
        raw = read_raw_partition(anchor)
        if not {"code", "amount"}.issubset(raw.columns):
            continue
        frame = pd.DataFrame(
            {
                "provider_symbol": raw["code"].map(normalize_symbol),
                "liquidity": pd.to_numeric(raw["amount"], errors="coerce"),
                "month": month,
            }
        )
        frame = frame.loc[frame["provider_symbol"].map(board_for_symbol).eq("MainBoard")]
        frame["exchange"] = frame["provider_symbol"].str.rsplit(".", n=1).str[-1]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_update(
    *,
    as_of_date: str,
    workspace_root: str | Path | None = None,
    bootstrap: bool = False,
    start_date: str = "",
    include_5m: bool = True,
    include_secondary: bool = True,
    publish: bool = True,
    hash_v2_shards: bool = False,
) -> dict[str, Any]:
    paths = ensure_qdp_v3_layout(workspace_root)
    plan = plan_update(
        as_of_date=as_of_date,
        workspace_root=workspace_root,
        bootstrap=bootstrap,
        start_date=start_date,
        include_5m=include_5m,
        include_secondary=include_secondary,
    )
    if plan.get("status") != "planned":
        return plan
    run_id = f"update__{stable_hash({'as_of_date': as_of_date, 'bootstrap': bootstrap, 'started_at': utc_now()}, length=20)}"
    run_path = paths.jobs / f"{run_id}.json"
    state: dict[str, Any] = {"status": "running", "run_id": run_id, "plan": plan, "stages": [], "started_at": utc_now()}
    atomic_write_json(run_path, state)

    def record(stage: str, result: dict[str, Any]) -> None:
        state["stages"].append({"stage": stage, "result": result})
        atomic_write_json(run_path, state)

    try:
        freeze_path = paths.metadata / "v2_freeze_20260713.json"
        if not freeze_path.exists() or read_json(freeze_path).get("status") != "complete":
            # Re-check reachability cheaply before hashing hundreds of GB.  A
            # missing v2 ancestor cannot be repaired by hashing extant shards.
            freeze_result = freeze_v2(workspace_root=workspace_root, hash_shards=False)
            record("freeze_v2", freeze_result)
            if freeze_result.get("status") == "metadata_only":
                freeze_result = freeze_v2(workspace_root=workspace_root, hash_shards=True)
                record("freeze_v2_full_hash", freeze_result)
            if freeze_result.get("status") != "complete":
                raise RuntimeError(f"v2_freeze_stage_incomplete:{freeze_result.get('status')}")
        calendar, _ = ingest_trading_calendar(
            start_date=str(plan["start_date"]),
            end_date=str(as_of_date),
            workspace_root=workspace_root,
            refresh=True,
        )
        opened = calendar["is_open"].astype(str).str.lower().isin({"1", "true", "t", "yes"})
        dates = sorted(set(calendar.loc[opened, "trade_date"].astype(str).str.slice(0, 10)))
        dates = [date for date in dates if str(plan["start_date"]) <= date <= str(as_of_date)]
        daily_dates = dates if bootstrap else dates[-10:]
        factor_dates = dates if bootstrap else dates[-60:]
        security_ref = ingest_security_master(as_of_date=str(as_of_date), workspace_root=workspace_root, refresh=True)
        security_master = read_raw_partition(security_ref)
        record("security_master", {"status": "completed", "content_sha256": security_ref.content_sha256, "row_count": security_ref.row_count})
        daily_result = ingest_baostock_date_partitions(
            mode="date-snapshot",
            trade_dates=daily_dates,
            workspace_root=workspace_root,
            refresh=True,
            cross_check=True,
            security_master=security_master,
            job_id=f"{run_id}__daily",
        )
        record("daily_date_snapshot", daily_result)
        if daily_result.get("failed_count"):
            raise RuntimeError("daily_date_snapshot_stage_failed")
        factor_result = ingest_baostock_date_partitions(
            mode="date-events",
            trade_dates=factor_dates,
            workspace_root=workspace_root,
            refresh=True,
            cross_check=False,
            job_id=f"{run_id}__factor",
        )
        record("factor_date_events", factor_result)
        if factor_result.get("failed_count"):
            raise RuntimeError("factor_date_events_stage_failed")
        all_symbols = _all_stock_symbols(security_master)
        factor_event_symbols = _recent_factor_event_symbols(trade_dates=factor_dates, workspace_root=workspace_root)
        missing_factor_symbols = _missing_symbol_partitions(
            symbols=all_symbols,
            raw_domain=RAW_ADJUST_FACTOR_SYMBOL_HISTORY,
            workspace_root=workspace_root,
        )
        reconciliation_symbols = (
            all_symbols
            if bootstrap
            else sorted(set(factor_event_symbols) | set(missing_factor_symbols))
        )
        factor_history_result = (
            ingest_factor_symbol_histories(
                symbols=reconciliation_symbols,
                start_date="1990-01-01",
                end_date=str(as_of_date),
                workspace_root=workspace_root,
                refresh=not bootstrap,
                job_id=f"{run_id}__factor_symbol_history",
            )
            if reconciliation_symbols
            else {"status": "completed", "symbol_count": 0, "completed_count": 0, "skipped_count": 0, "failed_count": 0}
        )
        record("factor_symbol_history", factor_history_result)
        if factor_history_result.get("failed_count"):
            raise RuntimeError("factor_symbol_history_stage_failed")
        missing_xdxr_symbols = _missing_symbol_partitions(
            symbols=all_symbols,
            raw_domain=RAW_CORPORATE_ACTION_XDXR,
            workspace_root=workspace_root,
        )
        xdxr_symbols = all_symbols if bootstrap else sorted(set(factor_event_symbols) | set(missing_xdxr_symbols))
        xdxr_result = (
            ingest_mootdx_xdxr(
                symbols=xdxr_symbols,
                workspace_root=workspace_root,
                refresh=not bootstrap,
                job_id=f"{run_id}__corporate_action_xdxr",
            )
            if xdxr_symbols
            else {"status": "completed", "symbol_count": 0, "completed_count": 0, "skipped_count": 0, "failed_count": 0}
        )
        record("corporate_action_xdxr", xdxr_result)
        if xdxr_result.get("failed_count"):
            raise RuntimeError("corporate_action_xdxr_stage_failed")
        if include_secondary:
            for snapshot_domain in ("industry_concept", "index_constituents"):
                snapshot_result = ingest_baostock_snapshot_domain(
                    domain=snapshot_domain,
                    as_of_date=str(as_of_date),
                    workspace_root=workspace_root,
                    refresh=True,
                )
                record(f"secondary_{snapshot_domain}", snapshot_result)
                if snapshot_result.get("status") == "partial":
                    raise RuntimeError(f"secondary_{snapshot_domain}_stage_failed")
            report_start = str(plan["start_date"]) if bootstrap else (pd.Timestamp(as_of_date) - pd.Timedelta(days=550)).strftime("%Y-%m-%d")
            for report_domain in ("financial_quarterly", "performance_forecast", "performance_express"):
                report_result = ingest_baostock_report_domain(
                    domain=report_domain,
                    symbols=all_symbols,
                    start_date=report_start,
                    end_date=str(as_of_date),
                    workspace_root=workspace_root,
                    refresh=True,
                    chunk_size=8 if report_domain == "financial_quarterly" else 32,
                    job_id=f"{run_id}__secondary_{report_domain}",
                )
                record(f"secondary_{report_domain}", report_result)
                if report_result.get("failed_count"):
                    raise RuntimeError(f"secondary_{report_domain}_stage_failed")
        if include_5m:
            intraday_dates = [date for date in dates if date >= "2020-01-01"] if bootstrap else dates[-1:]
            intraday_result = ingest_intraday_5m(
                symbols=_mainboard_symbols(security_master),
                trade_dates=intraday_dates,
                workspace_root=workspace_root,
                sampling_universe=_monthly_intraday_sampling_universe(trade_dates=intraday_dates, workspace_root=workspace_root),
                refresh=not bootstrap,
            )
            record("intraday_5m", intraday_result)
        candidate = build_candidate(
            workspace_root=workspace_root,
            start_date=str(plan["start_date"] if bootstrap else "2010-01-01"),
            end_date=str(as_of_date),
            require_factor_dual_path=bool(bootstrap),
        )
        record("build_candidate", {"status": "completed", "candidate_id": candidate.candidate_id, "blocker_count": len(candidate.blockers)})
        audit = audit_candidate(candidate_id=candidate.candidate_id, mode="semantic", workspace_root=workspace_root)
        record("semantic_audit", audit)
        candidate_diff = diff_candidate(candidate.candidate_id, workspace_root=workspace_root, against="active")
        record("diff_candidate", candidate_diff)
        state["diff"] = candidate_diff
        if publish:
            publish_result = publish_candidate(
                candidate_id=candidate.candidate_id,
                expect_active_sha=str(plan["active_sha256"]),
                workspace_root=workspace_root,
            )
            record("cas_publish", publish_result)
        else:
            publish_result = {"status": "not_requested"}
        state["status"] = "completed" if publish_result.get("status") in {"published", "not_requested"} else "blocked"
        state["candidate_id"] = candidate.candidate_id
        state["publish"] = publish_result
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
    state["finished_at"] = utc_now()
    atomic_write_json(run_path, state)
    return {**state, "run_path": str(run_path.resolve())}
