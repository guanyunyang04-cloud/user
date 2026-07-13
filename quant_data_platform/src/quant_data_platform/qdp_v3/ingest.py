from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from quant_data_platform.core.json_io import read_json
from quant_data_platform.domains.contracts import DataDomain, DatePartitionFetchRequest, DomainFetchRequest
from quant_data_platform.providers import (
    BAOSTOCK_BATCH_VERSION,
    BAOSTOCK_BATCH_WHEEL_SHA256,
    BaostockProvider,
    assert_baostock_batch_runtime,
    _fetch_baostock_stock_basic_frame_with_timeout,
)
from quant_data_platform.qdp_v3.constants import (
    QUALITY_PROVISIONAL,
    RAW_ADJUST_FACTOR_EVENT,
    RAW_ADJUST_FACTOR_SYMBOL_HISTORY,
    RAW_ALL_STOCK,
    RAW_DAILY_ASTOCK,
    RAW_DAILY_ETF,
    RAW_SECURITY_MASTER,
    RAW_TRADING_CALENDAR,
)
from quant_data_platform.qdp_v3.manifest import atomic_write_json, stable_hash, utc_now
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout, qdp_v3_paths
from quant_data_platform.qdp_v3.quality import audit_baostock_daily_raw, audit_baostock_factor_event_raw
from quant_data_platform.qdp_v3.storage import RawPartitionRef, iter_raw_partitions, read_raw_partition, write_raw_partition


@dataclass
class IngestTaskState:
    trade_date: str
    status: str = "pending"
    raw_domain: str = ""
    content_sha256: str = ""
    row_count: int = 0
    quality_tier: str = ""
    created: bool = False
    error: str = ""


@dataclass
class IngestJobState:
    job_id: str
    provider: str
    mode: str
    universe_kind: str
    trade_dates: list[str]
    tasks: dict[str, IngestTaskState]
    status: str = "pending"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    package_lock: dict[str, Any] = field(
        default_factory=lambda: {
            "baostock": {"version": BAOSTOCK_BATCH_VERSION, "wheel_sha256": BAOSTOCK_BATCH_WHEEL_SHA256}
        }
    )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tasks"] = {key: asdict(value) for key, value in self.tasks.items()}
        return payload


def _job_path(job_id: str, *, workspace_root: str | Path | None = None) -> Path:
    return qdp_v3_paths(workspace_root).jobs / f"{job_id}.json"


def _load_or_create_job(
    *,
    provider: str,
    mode: str,
    universe_kind: str,
    trade_dates: list[str],
    workspace_root: str | Path | None,
    job_id: str,
) -> IngestJobState:
    path = _job_path(job_id, workspace_root=workspace_root)
    payload = read_json(path)
    if payload:
        expected = {
            "provider": provider,
            "mode": mode,
            "universe_kind": universe_kind,
            "trade_dates": trade_dates,
        }
        actual = {key: payload.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(f"ingest_job_contract_conflict:{job_id}")
        tasks = {
            key: IngestTaskState(**dict(value))
            for key, value in dict(payload.get("tasks", {}) or {}).items()
            if isinstance(value, dict)
        }
        return IngestJobState(
            job_id=job_id,
            provider=provider,
            mode=mode,
            universe_kind=universe_kind,
            trade_dates=list(trade_dates),
            tasks=tasks,
            status=str(payload.get("status", "pending") or "pending"),
            created_at=str(payload.get("created_at", "") or utc_now()),
            updated_at=str(payload.get("updated_at", "") or utc_now()),
            package_lock=dict(payload.get("package_lock", {}) or {}),
        )
    return IngestJobState(
        job_id=job_id,
        provider=provider,
        mode=mode,
        universe_kind=universe_kind,
        trade_dates=list(trade_dates),
        tasks={date: IngestTaskState(trade_date=date) for date in trade_dates},
    )


def _save_job(job: IngestJobState, *, workspace_root: str | Path | None = None) -> None:
    job.updated_at = utc_now()
    atomic_write_json(_job_path(job.job_id, workspace_root=workspace_root), job.to_dict())


def ingest_trading_calendar(
    *,
    start_date: str,
    end_date: str,
    workspace_root: str | Path | None = None,
    provider: BaostockProvider | None = None,
    refresh: bool = False,
) -> tuple[pd.DataFrame, RawPartitionRef]:
    ensure_qdp_v3_layout(workspace_root)
    partition_value = f"{start_date}_{end_date}"
    existing = iter_raw_partitions(RAW_TRADING_CALENDAR, workspace_root=workspace_root)
    for ref in existing:
        if ref.partition_value == partition_value and not refresh:
            return pd.read_parquet(ref.payload_path, engine="pyarrow"), ref
    source = provider or BaostockProvider()
    result = source.fetch_domain(
        DomainFetchRequest(
            domain=DataDomain.TRADING_CALENDAR,
            start_date=start_date,
            end_date=end_date,
            exchange="SSE",
        )
    )
    frame = result.data.copy()
    if frame.empty or not {"trade_date", "is_open"}.issubset(frame.columns):
        raise RuntimeError("baostock_trading_calendar_empty_or_invalid")
    ref, _ = write_raw_partition(
        raw_domain=RAW_TRADING_CALENDAR,
        partition_field="request_range",
        partition_value=partition_value,
        frame=frame,
        receipt={
            "provider": "baostock",
            "endpoint": "query_trade_dates",
            "package_version": BAOSTOCK_BATCH_VERSION,
            "wheel_sha256": BAOSTOCK_BATCH_WHEEL_SHA256,
            "request": {"start_date": start_date, "end_date": end_date},
            "error_code": "0",
            "quality_tier": "strict",
        },
        workspace_root=workspace_root,
    )
    return frame, ref


def resolve_trade_dates(
    *,
    start_date: str,
    end_date: str,
    workspace_root: str | Path | None = None,
    provider: BaostockProvider | None = None,
    refresh_calendar: bool = False,
) -> list[str]:
    calendar, _ = ingest_trading_calendar(
        start_date=start_date,
        end_date=end_date,
        workspace_root=workspace_root,
        provider=provider,
        refresh=refresh_calendar,
    )
    mask = calendar["is_open"].astype(str).str.lower().isin({"1", "true", "t", "yes"})
    dates = calendar.loc[mask, "trade_date"].astype(str).str.slice(0, 10)
    return sorted(set(dates.tolist()))


def ingest_security_master(
    *,
    as_of_date: str,
    workspace_root: str | Path | None = None,
    refresh: bool = False,
) -> RawPartitionRef:
    existing = iter_raw_partitions(RAW_SECURITY_MASTER, workspace_root=workspace_root)
    for ref in existing:
        if ref.partition_value == str(as_of_date) and not refresh:
            return ref
    frame = _fetch_baostock_stock_basic_frame_with_timeout(trade_date=str(as_of_date), timeout_seconds=120)
    if frame.empty:
        raise RuntimeError("baostock_security_master_empty")
    ref, _ = write_raw_partition(
        raw_domain=RAW_SECURITY_MASTER,
        partition_value=str(as_of_date),
        frame=frame,
        receipt={
            "provider": "baostock",
            "endpoint": "query_stock_basic",
            "package_version": importlib_metadata.version("baostock"),
            "wheel_sha256": BAOSTOCK_BATCH_WHEEL_SHA256,
            "request": {"as_of_date": str(as_of_date)},
            "error_code": "0",
            "quality_tier": QUALITY_PROVISIONAL,
            "quality_note": "BaoStock stock-basic identity evidence; official code-change documents override it.",
        },
        workspace_root=workspace_root,
    )
    return ref


def ingest_factor_symbol_histories(
    *,
    symbols: Iterable[str],
    end_date: str,
    start_date: str = "1990-01-01",
    workspace_root: str | Path | None = None,
    provider: BaostockProvider | None = None,
    refresh: bool = False,
    chunk_size: int = 64,
    job_id: str = "",
) -> dict[str, Any]:
    """Fetch the legacy per-symbol factor history once for dual-path proof."""

    from quant_data_platform.qdp_v3.identity import normalize_symbol

    ensure_qdp_v3_layout(workspace_root)
    assert_baostock_batch_runtime()
    if provider is None or isinstance(provider, BaostockProvider):
        from quant_data_platform.qdp_v3.compatibility import assert_baostock_compatibility_gate

        assert_baostock_compatibility_gate(workspace_root)
    normalized = sorted({normalize_symbol(item) for item in symbols if normalize_symbol(item)})
    if not normalized:
        raise ValueError("factor_symbol_history_symbols_empty")
    size = max(1, int(chunk_size))
    if not job_id:
        job_id = f"ingest_factor_symbol_history__{stable_hash({'symbols': normalized, 'start': start_date, 'end': end_date}, length=20)}"
    job_path = _job_path(job_id, workspace_root=workspace_root)
    state = read_json(job_path) or {
        "job_id": job_id,
        "provider": "baostock",
        "mode": "factor-symbol-history",
        "start_date": str(start_date),
        "end_date": str(end_date),
        "symbols": normalized,
        "tasks": {symbol: {"status": "pending", "error": ""} for symbol in normalized},
        "created_at": utc_now(),
    }
    contract = {key: state.get(key) for key in ("start_date", "end_date", "symbols")}
    if contract != {"start_date": str(start_date), "end_date": str(end_date), "symbols": normalized}:
        raise RuntimeError(f"factor_symbol_history_job_contract_conflict:{job_id}")
    tasks = dict(state.get("tasks", {}) or {})
    source = provider or BaostockProvider()
    existing = {ref.partition_value: ref for ref in iter_raw_partitions(RAW_ADJUST_FACTOR_SYMBOL_HISTORY, workspace_root=workspace_root)}
    pending: list[str] = []
    for symbol in normalized:
        if not refresh and symbol in existing:
            tasks[symbol] = {
                "status": "skipped",
                "content_sha256": existing[symbol].content_sha256,
                "row_count": existing[symbol].row_count,
                "error": "",
            }
        elif not refresh and str(dict(tasks.get(symbol, {}) or {}).get("status", "")) == "completed":
            continue
        else:
            pending.append(symbol)
    state.update({"status": "running", "tasks": tasks, "updated_at": utc_now()})
    atomic_write_json(job_path, state)
    for offset in range(0, len(pending), size):
        chunk = pending[offset : offset + size]
        try:
            result = source.fetch_domain(
                DomainFetchRequest(
                    domain=DataDomain.ADJUST_FACTOR,
                    symbols=tuple(chunk),
                    start_date=str(start_date),
                    end_date=str(end_date),
                )
            )
            error_symbols = {
                normalize_symbol(item.get("symbol", ""))
                for item in list(result.error_report or [])
                if isinstance(item, dict) and item.get("symbol")
            }
            data = result.data.copy()
            for symbol in chunk:
                if symbol in error_symbols:
                    tasks[symbol] = {"status": "failed", "error": "provider_symbol_error"}
                    continue
                if not data.empty and "symbol" in data.columns:
                    symbol_frame = data.loc[data["symbol"].astype(str).eq(symbol)].copy()
                else:
                    symbol_frame = data.iloc[0:0].copy()
                ref, created = write_raw_partition(
                    raw_domain=RAW_ADJUST_FACTOR_SYMBOL_HISTORY,
                    partition_field="provider_symbol",
                    partition_value=symbol,
                    frame=symbol_frame,
                    receipt={
                        "provider": "baostock",
                        "endpoint": "query_adjust_factor",
                        "package_version": importlib_metadata.version("baostock"),
                        "wheel_sha256": BAOSTOCK_BATCH_WHEEL_SHA256,
                        "request": {"symbol": symbol, "start_date": str(start_date), "end_date": str(end_date)},
                        "error_code": "0",
                        "quality_tier": QUALITY_PROVISIONAL,
                        "quality_note": "Legacy per-symbol path retained for event-set reconciliation; not sufficient alone for canonical strict factors.",
                    },
                    workspace_root=workspace_root,
                )
                tasks[symbol] = {
                    "status": "completed",
                    "content_sha256": ref.content_sha256,
                    "row_count": ref.row_count,
                    "created": bool(created),
                    "error": "",
                }
        except Exception as exc:
            for symbol in chunk:
                tasks[symbol] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        state.update({"tasks": tasks, "updated_at": utc_now()})
        atomic_write_json(job_path, state)
    failed = [{"symbol": symbol, **dict(task)} for symbol, task in tasks.items() if str(dict(task).get("status", "")) == "failed"]
    state.update({"status": "partial" if failed else "completed", "tasks": tasks, "updated_at": utc_now()})
    atomic_write_json(job_path, state)
    return {
        "status": state["status"],
        "job_id": job_id,
        "symbol_count": len(normalized),
        "completed_count": sum(str(dict(item).get("status", "")) == "completed" for item in tasks.values()),
        "skipped_count": sum(str(dict(item).get("status", "")) == "skipped" for item in tasks.values()),
        "failed_count": len(failed),
        "failed": failed[:50],
        "job_path": str(job_path.resolve()),
    }


def _previous_open_date(trade_date: str, *, workspace_root: str | Path | None) -> str:
    refs = iter_raw_partitions(RAW_TRADING_CALENDAR, workspace_root=workspace_root)
    frames = [read_raw_partition(ref) for ref in refs]
    if not frames:
        return ""
    calendar = pd.concat(frames, ignore_index=True)
    if not {"trade_date", "is_open"}.issubset(calendar.columns):
        return ""
    opened = calendar["is_open"].astype(str).str.lower().isin({"1", "true", "t", "yes"})
    dates = sorted(
        set(
            calendar.loc[opened, "trade_date"]
            .astype(str)
            .str.slice(0, 10)
            .loc[lambda values: values < str(trade_date)]
        )
    )
    return dates[-1] if dates else ""


def _latest_neighbor_partition(raw_domain: str, trade_date: str, *, workspace_root: str | Path | None) -> RawPartitionRef | None:
    previous_date = _previous_open_date(str(trade_date), workspace_root=workspace_root)
    if not previous_date:
        return None
    refs = iter_raw_partitions(
        raw_domain,
        workspace_root=workspace_root,
        start_value=previous_date,
        end_value=previous_date,
    )
    return refs[-1] if refs else None


def _latest_security_master_frame(*, workspace_root: str | Path | None) -> pd.DataFrame:
    refs = iter_raw_partitions(RAW_SECURITY_MASTER, workspace_root=workspace_root)
    return read_raw_partition(refs[-1]) if refs else pd.DataFrame()


def _existing_for_date(raw_domain: str, trade_date: str, *, workspace_root: str | Path | None) -> RawPartitionRef | None:
    refs = iter_raw_partitions(raw_domain, workspace_root=workspace_root, start_value=str(trade_date), end_value=str(trade_date))
    return refs[-1] if refs else None


def ingest_baostock_date_partitions(
    *,
    mode: str,
    trade_dates: Iterable[str],
    universe_kind: str = "all_a",
    workspace_root: str | Path | None = None,
    provider: BaostockProvider | None = None,
    refresh: bool = False,
    cross_check: bool = True,
    security_master: pd.DataFrame | None = None,
    job_id: str = "",
) -> dict[str, Any]:
    ensure_qdp_v3_layout(workspace_root)
    dates = sorted({str(item)[:10] for item in trade_dates if str(item).strip()})
    if not dates:
        raise ValueError("ingest_trade_dates_empty")
    if provider is None or isinstance(provider, BaostockProvider):
        from quant_data_platform.qdp_v3.compatibility import assert_baostock_compatibility_gate

        assert_baostock_compatibility_gate(workspace_root)
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"date-snapshot", "date-events"}:
        raise ValueError(f"unsupported_ingest_mode:{mode}")
    universe = str(universe_kind or "all_a").strip().lower()
    if normalized_mode == "date-events" and universe != "all_a":
        raise ValueError("date-events only supports all_a")
    raw_domain = (
        RAW_ADJUST_FACTOR_EVENT
        if normalized_mode == "date-events"
        else RAW_DAILY_ETF
        if universe == "etf"
        else RAW_DAILY_ASTOCK
    )
    if not job_id:
        seed = {"provider": "baostock", "mode": normalized_mode, "universe": universe, "dates": dates, "refresh": bool(refresh)}
        job_id = f"ingest_{normalized_mode.replace('-', '_')}__{stable_hash(seed, length=20)}"
    job = _load_or_create_job(
        provider="baostock",
        mode=normalized_mode,
        universe_kind=universe,
        trade_dates=dates,
        workspace_root=workspace_root,
        job_id=job_id,
    )
    source = provider or BaostockProvider()
    master_evidence = (
        security_master.copy()
        if isinstance(security_master, pd.DataFrame)
        else _latest_security_master_frame(workspace_root=workspace_root)
    )
    identity_registry: Any | None = None
    identity_provider_symbols: set[str] = set()
    if normalized_mode == "date-snapshot" and universe == "all_a":
        from quant_data_platform.qdp_v3.identity import SecurityIdentityRegistry

        master_symbol_column = (
            "symbol"
            if "symbol" in master_evidence.columns
            else "provider_symbol"
            if "provider_symbol" in master_evidence.columns
            else ""
        )
        if master_symbol_column:
            identity_provider_symbols.update(master_evidence[master_symbol_column].astype(str).tolist())
        identity_registry = SecurityIdentityRegistry.from_sources(
            provider_symbols=identity_provider_symbols,
            security_master=master_evidence,
            workspace_root=workspace_root,
        )
    job.status = "running"
    _save_job(job, workspace_root=workspace_root)
    for trade_date in dates:
        task = job.tasks.setdefault(trade_date, IngestTaskState(trade_date=trade_date))
        if task.status in {"completed", "skipped"} and not refresh:
            continue
        existing = _existing_for_date(raw_domain, trade_date, workspace_root=workspace_root)
        if existing is not None and not refresh:
            task.status = "skipped"
            task.raw_domain = raw_domain
            task.content_sha256 = existing.content_sha256
            task.row_count = existing.row_count
            task.quality_tier = existing.quality_tier
            _save_job(job, workspace_root=workspace_root)
            continue
        task.status = "running"
        task.error = ""
        _save_job(job, workspace_root=workspace_root)
        try:
            request = DatePartitionFetchRequest(
                domain=DataDomain.ADJUST_FACTOR_EVENT if normalized_mode == "date-events" else DataDomain.MARKET_DAILY,
                trade_date=trade_date,
                universe_kind=universe,
                fetch_mode="date_events" if normalized_mode == "date-events" else "date_snapshot",
            )
            result = source.fetch_date_partition(request)
            expected_codes: list[str] | None = None
            cross_check_ref: RawPartitionRef | None = None
            if normalized_mode == "date-snapshot" and universe == "all_a" and cross_check:
                if type(source) is BaostockProvider:
                    universe_frame = source.fetch_all_stock_audit_evidence(trade_date=trade_date)
                    universe_errors: list[dict[str, Any]] = []
                else:
                    universe_result = source.fetch_domain(
                        DomainFetchRequest(
                            domain=DataDomain.UNIVERSE_SNAPSHOT,
                            start_date=trade_date,
                            end_date=trade_date,
                        )
                    )
                    universe_frame = universe_result.data
                    universe_errors = list(universe_result.error_report)
                if universe_errors:
                    raise RuntimeError(f"query_all_stock_cross_check_failed:{universe_errors[:3]}")
                expected_codes = universe_frame.get("symbol", pd.Series(dtype=str)).astype(str).tolist()
                cross_report = {
                    "domain": RAW_ALL_STOCK,
                    "quality_tier": "strict" if not universe_errors else "quarantined",
                    "errors": universe_errors,
                }
                cross_check_ref, _ = write_raw_partition(
                    raw_domain=RAW_ALL_STOCK,
                    partition_value=trade_date,
                    frame=universe_frame,
                    receipt={
                        "provider": "baostock",
                        "endpoint": "query_all_stock",
                        "package_version": BAOSTOCK_BATCH_VERSION,
                        "wheel_sha256": BAOSTOCK_BATCH_WHEEL_SHA256,
                        "request": {"day": trade_date},
                        "error_code": "0" if not universe_errors else "provider_error",
                        "quality_tier": cross_report["quality_tier"],
                        "quality_report": cross_report,
                        "normalized_evidence": True,
                    },
                    workspace_root=workspace_root,
                )
            if normalized_mode == "date-events":
                quality = audit_baostock_factor_event_raw(result.raw_data, query_date=trade_date)
            elif universe == "etf":
                quality = audit_baostock_daily_raw(
                    result.raw_data,
                    query_date=trade_date,
                    raw_domain=RAW_DAILY_ETF,
                    expected_codes=None,
                )
            else:
                from quant_data_platform.qdp_v3.identity import SecurityIdentityRegistry

                code_evidence = set(expected_codes or []) | set(result.raw_data.get("code", pd.Series(dtype=str)).astype(str))
                if not master_evidence.empty:
                    master_symbol_column = (
                        "symbol"
                        if "symbol" in master_evidence.columns
                        else "provider_symbol"
                        if "provider_symbol" in master_evidence.columns
                        else ""
                    )
                    if master_symbol_column:
                        code_evidence.update(master_evidence[master_symbol_column].astype(str).tolist())
                if identity_registry is None or any(
                    not identity_registry.security_id_for_provider_symbol(symbol) for symbol in code_evidence
                ):
                    identity_provider_symbols.update(code_evidence)
                    identity_registry = SecurityIdentityRegistry.from_sources(
                        provider_symbols=identity_provider_symbols,
                        security_master=master_evidence,
                        workspace_root=workspace_root,
                    )
                identity_map = {
                    symbol: identity_registry.security_id_for_provider_symbol(symbol)
                    for symbol in code_evidence
                    if identity_registry.security_id_for_provider_symbol(symbol)
                }
                neighbor_daily = _latest_neighbor_partition(raw_domain, trade_date, workspace_root=workspace_root)
                neighbor_evidence = read_raw_partition(neighbor_daily) if neighbor_daily else pd.DataFrame()
                quality = audit_baostock_daily_raw(
                    result.raw_data,
                    query_date=trade_date,
                    expected_codes=expected_codes,
                    expected_code_evidence=universe_frame if cross_check else None,
                    security_master=master_evidence,
                    neighbor_row_count=neighbor_daily.row_count if neighbor_daily else None,
                    neighbor_expected_codes=(
                        neighbor_evidence.get("symbol", pd.Series(dtype=str)).astype(str).tolist()
                        if not neighbor_evidence.empty
                        else None
                    ),
                    neighbor_date=neighbor_daily.partition_value if neighbor_daily else "",
                    identity_security_by_symbol=identity_map,
                )
            receipt = {
                "provider": "baostock",
                "endpoint": str(result.coverage_report.get("endpoint", "")),
                "package_version": str(result.coverage_report.get("package_version", BAOSTOCK_BATCH_VERSION)),
                "wheel_sha256": BAOSTOCK_BATCH_WHEEL_SHA256,
                "request": asdict(result.request),
                "requested_at": utc_now(),
                "error_code": str(result.coverage_report.get("error_code", "0")),
                "error_msg": str(result.coverage_report.get("error_msg", "")),
                "per_page_count": int(result.coverage_report.get("per_page_count", 0) or 0),
                "elapsed_seconds": float(result.coverage_report.get("elapsed_seconds", 0.0) or 0.0),
                "attempt_count": int(result.coverage_report.get("attempt_count", 1) or 1),
                "quality_tier": quality.quality_tier,
                "quality_report": quality.to_dict(),
                "cross_check_content_sha256": cross_check_ref.content_sha256 if cross_check_ref else "",
            }
            ref, created = write_raw_partition(
                raw_domain=raw_domain,
                partition_value=trade_date,
                frame=result.raw_data,
                receipt=receipt,
                workspace_root=workspace_root,
            )
            task.status = "completed"
            task.raw_domain = raw_domain
            task.content_sha256 = ref.content_sha256
            task.row_count = ref.row_count
            task.quality_tier = quality.quality_tier
            task.created = bool(created)
        except Exception as exc:
            task.status = "failed"
            task.error = f"{type(exc).__name__}: {exc}"
        _save_job(job, workspace_root=workspace_root)
    failed = [asdict(task) for task in job.tasks.values() if task.status == "failed"]
    job.status = "partial" if failed else "completed"
    _save_job(job, workspace_root=workspace_root)
    return {
        "status": job.status,
        "job_id": job.job_id,
        "provider": job.provider,
        "mode": job.mode,
        "universe_kind": job.universe_kind,
        "task_count": len(job.tasks),
        "completed_count": sum(task.status == "completed" for task in job.tasks.values()),
        "skipped_count": sum(task.status == "skipped" for task in job.tasks.values()),
        "failed_count": len(failed),
        "failed": failed[:50],
        "job_path": str(_job_path(job.job_id, workspace_root=workspace_root).resolve()),
    }
