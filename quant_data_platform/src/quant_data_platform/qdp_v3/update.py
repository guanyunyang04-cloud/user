from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quant_data_platform.core.json_io import read_json
from quant_data_platform.qdp_v3.audit import audit_candidate
from quant_data_platform.qdp_v3.build import build_candidate
from quant_data_platform.qdp_v3.corporate_actions import ingest_mootdx_xdxr
from quant_data_platform.qdp_v3.freeze import freeze_v2, validate_v2_freeze_proof
from quant_data_platform.qdp_v3.historical import (
    ingest_tushare_proxy_intraday_pair,
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
from quant_data_platform.qdp_v3.mootdx_1m import ingest_mootdx_intraday_1m
from quant_data_platform.qdp_v3.proxy_compatibility import lock_bootstrap_cutoff, run_tushare_proxy_compatibility_gate
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


def _date_task_summary(dates: list[str]) -> dict[str, Any]:
    """Keep plans useful without serializing thousands of redundant dates."""

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
    include_5m: bool = True,
    include_1m: bool = False,
    include_secondary: bool = True,
    historical_provider: str = "",
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
    intraday_dates = dates if bootstrap and include_5m else dates[-1:] if include_5m and dates else []
    incremental_1m_dates = dates[-10:] if include_1m and not bootstrap else dates if include_1m else []
    use_proxy_bootstrap = bool(bootstrap and str(historical_provider).strip().lower() == "tushare-proxy")
    freeze_state = read_json(paths.metadata / "v2_freeze_20260713.json")
    freeze_validation = validate_v2_freeze_proof(freeze_state)
    return {
        "status": "planned",
        "bootstrap": bool(bootstrap),
        "start_date": effective_start,
        "as_of_date": str(as_of_date),
        "active_sha256": active_manifest_sha256(workspace_root),
        "calendar_source": calendar_source,
        "stages": [
            {"stage": "freeze_v2", "enabled": not bool(freeze_validation.get("acceptable", False)), "current_status": freeze_state.get("status", "missing"), "proof_kind": freeze_validation.get("proof_kind", ""), "missing_dataset_count": len(freeze_state.get("missing_dataset_ids", []) or [])},
            {"stage": "calendar", "start_date": effective_start, "end_date": str(as_of_date)},
            {"stage": "security_master", "as_of_date": str(as_of_date)},
            {"stage": "tushare_proxy_compatibility", "enabled": use_proxy_bootstrap},
            {"stage": "tushare_proxy_cutoff", "enabled": use_proxy_bootstrap, "bootstrap_cutoff": str(as_of_date)},
            {"stage": "tushare_proxy_reference", "enabled": use_proxy_bootstrap, "domains": ["stock-basic", "trade-calendar", "daily", "daily-basic", "identity", "status", "factor", "dividend", "financial"]},
            {"stage": "daily_date_snapshot", **_date_task_summary(daily_dates)},
            {"stage": "factor_date_events", **_date_task_summary(factor_dates)},
            {"stage": "factor_symbol_history", "scope": "all securities once; resumable"},
            {"stage": "corporate_action_xdxr", "scope": "all securities on bootstrap; recent factor-event securities on incremental updates"},
            {"stage": "secondary_pit", "enabled": bool(include_secondary), "domains": ["financial_quarterly", "performance_forecast", "performance_express", "industry_concept", "index_constituents"]},
            {"stage": "intraday_5m", "enabled": bool(include_5m), "trade_date_count": len(intraday_dates), **{key: value for key, value in _date_task_summary(intraday_dates).items() if key != "task_count"}},
            {"stage": "intraday_1m", "enabled": bool(include_1m), "provider": "tushare_proxy" if use_proxy_bootstrap else "mootdx_online", **_date_task_summary(incremental_1m_dates)},
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


def _security_lifecycle_ranges(security_master: pd.DataFrame) -> dict[str, tuple[str, str]]:
    if security_master.empty:
        return {}
    symbol_column = "symbol" if "symbol" in security_master.columns else "provider_symbol" if "provider_symbol" in security_master.columns else ""
    if not symbol_column:
        return {}
    out: dict[str, tuple[str, str]] = {}
    for row in security_master.itertuples(index=False):
        payload = row._asdict()
        symbol = normalize_symbol(payload.get(symbol_column, ""))
        if not symbol:
            continue
        list_date = str(payload.get("list_date", "") or "")[:10]
        delist_date = str(payload.get("delist_date", "") or "")[:10]
        out[symbol] = (list_date, delist_date)
    return out


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
    include_1m: bool = False,
    include_secondary: bool = True,
    historical_provider: str = "",
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
        include_1m=include_1m,
        include_secondary=include_secondary,
        historical_provider=historical_provider,
    )
    if plan.get("status") != "planned":
        return plan
    if bootstrap:
        run_id = f"bootstrap__{stable_hash({'as_of_date': as_of_date, 'start_date': plan['start_date'], 'historical_provider': historical_provider}, length=20)}"
    else:
        run_id = f"update__{stable_hash({'as_of_date': as_of_date, 'started_at': utc_now()}, length=20)}"
    run_path = paths.jobs / f"{run_id}.json"
    state: dict[str, Any] = {"status": "running", "run_id": run_id, "plan": plan, "stages": [], "started_at": utc_now()}
    atomic_write_json(run_path, state)

    def record(stage: str, result: dict[str, Any]) -> None:
        state["stages"].append({"stage": stage, "result": result})
        atomic_write_json(run_path, state)

    try:
        freeze_path = paths.metadata / "v2_freeze_20260713.json"
        freeze_state = read_json(freeze_path) if freeze_path.exists() else {}
        if not validate_v2_freeze_proof(freeze_state).get("acceptable", False):
            # Re-check reachability cheaply before hashing hundreds of GB.  A
            # missing v2 ancestor cannot be repaired by hashing extant shards.
            freeze_result = freeze_v2(workspace_root=workspace_root, hash_shards=bool(hash_v2_shards))
            record("freeze_v2", freeze_result)
            if freeze_result.get("status") == "metadata_only" and not hash_v2_shards:
                freeze_result = freeze_v2(workspace_root=workspace_root, hash_shards=True)
                record("freeze_v2_full_hash", freeze_result)
            refreshed_freeze = read_json(freeze_path)
            if not validate_v2_freeze_proof(refreshed_freeze).get("acceptable", False):
                raise RuntimeError(f"v2_freeze_stage_incomplete:{freeze_result.get('status')}")
        use_proxy_bootstrap = bool(
            bootstrap and str(historical_provider or "").strip().lower() == "tushare-proxy"
        )
        if use_proxy_bootstrap:
            compatibility = run_tushare_proxy_compatibility_gate(
                workspace_root=workspace_root,
                as_of_date=str(as_of_date),
                smoke=False,
            )
            record("tushare_proxy_compatibility", compatibility)
            if compatibility.get("status") != "passed":
                raise RuntimeError("tushare_proxy_compatibility_gate_failed")
            cutoff = lock_bootstrap_cutoff(
                requested_cutoff=str(as_of_date),
                workspace_root=workspace_root,
            )
            record("tushare_proxy_cutoff", cutoff)
            for reference_domain in ("stock-basic", "trade-calendar"):
                result = ingest_tushare_proxy_reference(
                    domain=reference_domain,
                    start_date=str(plan["start_date"]),
                    end_date=str(cutoff["bootstrap_cutoff"]),
                    workspace_root=workspace_root,
                    resume=True,
                )
                record(f"tushare_proxy_{reference_domain}", result)
                if result.get("status") != "completed":
                    raise RuntimeError(f"tushare_proxy_{reference_domain}_incomplete")
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
        proxy_all_symbols: list[str] = []
        proxy_mainboard_symbols: list[str] = []
        proxy_lifecycle: dict[str, tuple[str, str]] = {}
        if use_proxy_bootstrap:
            proxy_all_symbols, _ = proxy_symbol_inventory(
                workspace_root=workspace_root,
                mainboard_only=False,
            )
            proxy_mainboard_symbols, proxy_lifecycle = proxy_symbol_inventory(
                workspace_root=workspace_root,
                mainboard_only=True,
            )
            reference_plan = (
                ("daily", dates, ()),
                ("daily-basic", dates, ()),
                ("status", dates, ()),
                ("factor", (), proxy_all_symbols),
                ("identity", (), proxy_all_symbols),
                ("dividend", (), proxy_all_symbols),
                ("financial", (), proxy_all_symbols),
            )
            for reference_domain, reference_dates, reference_symbols in reference_plan:
                result = ingest_tushare_proxy_reference(
                    domain=reference_domain,
                    start_date="1990-01-01" if reference_domain in {"factor", "identity", "dividend"} else str(plan["start_date"]),
                    end_date=str(as_of_date),
                    trade_dates=reference_dates,
                    symbols=reference_symbols,
                    workspace_root=workspace_root,
                    resume=True,
                    max_workers=4,
                )
                record(f"tushare_proxy_{reference_domain}", result)
                if result.get("status") != "completed":
                    raise RuntimeError(f"tushare_proxy_{reference_domain}_incomplete:{result.get('status')}")
            if include_1m:
                for wave_start, wave_end, wave_name in (
                    (str(plan["start_date"]), min("2019-12-31", str(as_of_date)), "2010_2019"),
                    ("2020-01-01", str(as_of_date), "2020_cutoff"),
                ):
                    if wave_start > wave_end:
                        continue
                    proxy_intraday = ingest_tushare_proxy_intraday_pair(
                        symbols=proxy_mainboard_symbols,
                        start_date=wave_start,
                        end_date=wave_end,
                        workspace_root=workspace_root,
                        lifecycle_ranges=proxy_lifecycle,
                        resume=True,
                        run_label=f"bootstrap_{wave_name}_{str(as_of_date).replace('-', '')}",
                    )
                    record(f"tushare_proxy_intraday_{wave_name}", proxy_intraday)
                    if proxy_intraday.get("status") != "completed":
                        state["status"] = str(proxy_intraday.get("status") or "paused")
                        state["finished_at"] = utc_now()
                        atomic_write_json(run_path, state)
                        return {**state, "run_path": str(run_path.resolve())}
        security_ref = ingest_security_master(as_of_date=str(as_of_date), workspace_root=workspace_root, refresh=True)
        security_master = read_raw_partition(security_ref)
        record("security_master", {"status": "completed", "content_sha256": security_ref.content_sha256, "row_count": security_ref.row_count})
        daily_result = ingest_baostock_date_partitions(
            mode="date-snapshot",
            trade_dates=daily_dates,
            workspace_root=workspace_root,
            refresh=not bootstrap,
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
            refresh=not bootstrap,
            cross_check=False,
            job_id=f"{run_id}__factor",
        )
        record("factor_date_events", factor_result)
        if factor_result.get("failed_count"):
            raise RuntimeError("factor_date_events_stage_failed")
        all_symbols = _all_stock_symbols(security_master)
        lifecycle_ranges = _security_lifecycle_ranges(security_master)
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
                    symbol_lifecycle_ranges=lifecycle_ranges,
                )
                record(f"secondary_{report_domain}", report_result)
                if report_result.get("failed_count"):
                    raise RuntimeError(f"secondary_{report_domain}_stage_failed")
        if include_5m and not use_proxy_bootstrap:
            intraday_dates = [date for date in dates if date >= "2020-01-01"] if bootstrap else dates[-1:]
            intraday_result = ingest_intraday_5m(
                symbols=_mainboard_symbols(security_master),
                trade_dates=intraday_dates,
                workspace_root=workspace_root,
                sampling_universe=_monthly_intraday_sampling_universe(trade_dates=intraday_dates, workspace_root=workspace_root),
                refresh=not bootstrap,
            )
            record("intraday_5m", intraday_result)
        if include_1m and not use_proxy_bootstrap:
            one_minute_result = ingest_mootdx_intraday_1m(
                trade_dates=dates[-10:],
                workspace_root=workspace_root,
                refresh=True,
                job_id=f"{run_id}__intraday_1m",
            )
            record("intraday_1m", one_minute_result)
            if one_minute_result.get("status") not in {"completed", "pending_provider_unhealthy"}:
                raise RuntimeError(f"intraday_1m_stage_failed:{one_minute_result.get('status')}")
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
