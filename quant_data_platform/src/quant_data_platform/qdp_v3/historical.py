from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

from quant_data_platform.core.json_io import read_json
from quant_data_platform.domains.contracts import HistoryPageFetchRequest
from quant_data_platform.qdp_v3.constants import (
    EXPECTED_5M_BAR_ENDS,
    QUALITY_PROVISIONAL,
    QUALITY_QUARANTINED,
    QUALITY_STRICT,
    RAW_INTRADAY_5M_SELECTED,
    RAW_TUSHARE_PROXY_ADJ_FACTOR,
    RAW_TUSHARE_PROXY_DAILY,
    RAW_TUSHARE_PROXY_DAILY_BASIC,
    RAW_TUSHARE_PROXY_DIVIDEND,
    RAW_TUSHARE_PROXY_FINANCIAL,
    RAW_TUSHARE_PROXY_INTRADAY_5M,
    RAW_TUSHARE_PROXY_NAMECHANGE,
    RAW_TUSHARE_PROXY_STOCK_BASIC,
    RAW_TUSHARE_PROXY_SUSPEND,
    RAW_TUSHARE_PROXY_TRADE_CALENDAR,
)
from quant_data_platform.qdp_v3.manifest import atomic_write_json, sha256_file, stable_hash, utc_now
from quant_data_platform.qdp_v3.identity import SecurityIdentityRegistry, board_for_symbol, normalize_symbol
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout, qdp_v3_paths
from quant_data_platform.qdp_v3.runtime_index import RuntimeIndex
from quant_data_platform.qdp_v3.storage import (
    RawPartitionRef,
    atomic_write_parquet,
    get_raw_partition,
    iter_raw_partitions,
    read_raw_partition,
    write_empty_raw_partition,
    write_raw_partition,
    read_raw_receipt,
)
from quant_data_platform.qdp_v3.supervisor import supervise_job
from quant_data_platform.tushare_proxy import (
    TUSHARE_PROXY_HISTORY_PAGE_SIZE,
    TUSHARE_PROXY_NAME,
    TushareProxyClient,
    TushareProxyConfig,
    TushareProxyError,
    TushareProxyProtocolError,
    TushareProxyQuotaError,
    history_page_result_to_receipt,
    redact_secrets,
)


GIB = 1024**3
MIN_FREE_SPACE_BYTES = 200 * GIB
MAX_HISTORY_WORKERS = 3
HISTORY_WORKER_ENV = "QDP_TUSHARE_PROXY_HISTORY_WORKERS"
PROTOCOL_CIRCUIT_THRESHOLD = 5
ERROR_WINDOW_SIZE = 200
ERROR_RATE_WORKER_REDUCTION = 0.02
PRODUCTION_GATE_MIN_REQUESTS = 1_000
# Network attempts are retried per page.  Use the same sustained-error
# threshold as worker reduction so occasional TLS EOFs do not repeatedly stop
# a multi-day historical capture that is otherwise making forward progress.
PRODUCTION_GATE_MAX_ERROR_RATE = 0.02
INTRADAY_CAPTURE_EVIDENCE_CONTRACT = "qdp_v3_intraday_capture_evidence_v1"


def _proxy_performance_gate_failed(metrics: Mapping[str, Any]) -> bool:
    return (
        int(metrics.get("request_count", 0) or 0) >= PRODUCTION_GATE_MIN_REQUESTS
        and float(metrics.get("error_rate", 0.0) or 0.0) > PRODUCTION_GATE_MAX_ERROR_RATE
    )


def _history_worker_limit(requested_workers: int) -> int:
    """Clamp history concurrency, allowing an operational low-memory override."""

    configured = os.environ.get(HISTORY_WORKER_ENV, "").strip()
    if not configured:
        return max(1, min(int(requested_workers), MAX_HISTORY_WORKERS))
    try:
        override = int(configured)
    except ValueError as exc:
        raise ValueError(f"history_worker_env_invalid:{configured}") from exc
    return max(1, min(int(requested_workers), MAX_HISTORY_WORKERS, override))


@dataclass(frozen=True)
class ProxyRawSpec:
    name: str
    api_name: str
    raw_domain: str
    partition_field: str
    fields: tuple[str, ...]
    request_kind: str


REFERENCE_SPECS: dict[str, tuple[ProxyRawSpec, ...]] = {
    "stock-basic": (
        ProxyRawSpec(
            name="stock_basic",
            api_name="stock_basic",
            raw_domain=RAW_TUSHARE_PROXY_STOCK_BASIC,
            partition_field="as_of_date",
            fields=(
                "ts_code",
                "symbol",
                "name",
                "area",
                "industry",
                "fullname",
                "enname",
                "cnspell",
                "market",
                "exchange",
                "curr_type",
                "list_status",
                "list_date",
                "delist_date",
                "is_hs",
                "act_name",
                "act_ent_type",
            ),
            request_kind="stock_status",
        ),
    ),
    "trade-calendar": (
        ProxyRawSpec(
            name="trade_cal",
            api_name="trade_cal",
            raw_domain=RAW_TUSHARE_PROXY_TRADE_CALENDAR,
            partition_field="request_range",
            fields=("exchange", "cal_date", "is_open", "pretrade_date"),
            request_kind="range",
        ),
    ),
    "daily": (
        ProxyRawSpec(
            name="daily",
            api_name="daily",
            raw_domain=RAW_TUSHARE_PROXY_DAILY,
            partition_field="trade_date",
            fields=(
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "pre_close",
                "change",
                "pct_chg",
                "vol",
                "amount",
            ),
            request_kind="trade_date",
        ),
    ),
    "daily-basic": (
        ProxyRawSpec(
            name="daily_basic",
            api_name="daily_basic",
            raw_domain=RAW_TUSHARE_PROXY_DAILY_BASIC,
            partition_field="trade_date",
            fields=(
                "ts_code",
                "trade_date",
                "close",
                "turnover_rate",
                "turnover_rate_f",
                "volume_ratio",
                "pe",
                "pe_ttm",
                "pb",
                "ps",
                "ps_ttm",
                "dv_ratio",
                "dv_ttm",
                "total_share",
                "float_share",
                "free_share",
                "total_mv",
                "circ_mv",
            ),
            request_kind="trade_date",
        ),
    ),
    "factor": (
        ProxyRawSpec(
            name="adj_factor",
            api_name="adj_factor",
            raw_domain=RAW_TUSHARE_PROXY_ADJ_FACTOR,
            partition_field="provider_symbol",
            fields=("ts_code", "trade_date", "adj_factor"),
            request_kind="symbol_range",
        ),
    ),
    "identity": (
        ProxyRawSpec(
            name="namechange",
            api_name="namechange",
            raw_domain=RAW_TUSHARE_PROXY_NAMECHANGE,
            partition_field="provider_symbol",
            fields=("ts_code", "name", "start_date", "end_date", "ann_date", "change_reason"),
            request_kind="symbol_range",
        ),
    ),
    "status": (
        ProxyRawSpec(
            name="suspend_d",
            api_name="suspend_d",
            raw_domain=RAW_TUSHARE_PROXY_SUSPEND,
            partition_field="trade_date",
            fields=("ts_code", "trade_date", "suspend_timing", "suspend_type"),
            request_kind="trade_date",
        ),
    ),
    "dividend": (
        ProxyRawSpec(
            name="dividend",
            api_name="dividend",
            raw_domain=RAW_TUSHARE_PROXY_DIVIDEND,
            partition_field="provider_symbol",
            fields=(
                "ts_code",
                "end_date",
                "ann_date",
                "div_proc",
                "stk_div",
                "stk_bo_rate",
                "stk_co_rate",
                "cash_div",
                "cash_div_tax",
                "record_date",
                "ex_date",
                "pay_date",
                "div_listdate",
                "imp_ann_date",
                "base_date",
                "base_share",
            ),
            request_kind="symbol_range",
        ),
    ),
}


FINANCIAL_APIS: tuple[ProxyRawSpec, ...] = tuple(
    ProxyRawSpec(
        name=api_name,
        api_name=api_name,
        raw_domain=RAW_TUSHARE_PROXY_FINANCIAL,
        partition_field="api_symbol",
        # Empty fields requests every gateway-supported field. Raw finance is
        # deliberately lossless; canonical mapping happens in a later stage.
        fields=(),
        request_kind="symbol_range",
    )
    for api_name in (
        "income",
        "balancesheet",
        "cashflow",
        "fina_indicator",
        "forecast",
        "express",
    )
)


@dataclass
class HistoricalTaskState:
    task_id: str
    status: str = "pending"
    cursor_end_at: str = ""
    page_count: int = 0
    row_count: int = 0
    content_sha256: str = ""
    error_code: str = ""
    error: str = ""
    updated_at: str = field(default_factory=utc_now)


@dataclass
class HistoricalJobState:
    job_id: str
    provider: str
    mode: str
    domain: str
    start_date: str
    end_date: str
    frequency: str
    task_ids: list[str]
    tasks: dict[str, HistoricalTaskState]
    status: str = "pending"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    bootstrap_cutoff: str = ""
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    protocol_error_streak: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tasks"] = {key: asdict(value) for key, value in self.tasks.items()}
        return payload


def _historical_job_path(job_id: str, *, workspace_root: str | Path | None) -> Path:
    return qdp_v3_paths(workspace_root).jobs / f"{job_id}.json"


def proxy_symbol_inventory(
    *,
    workspace_root: str | Path | None = None,
    mainboard_only: bool = False,
) -> tuple[list[str], dict[str, tuple[str, str]]]:
    """Read the latest L/D/P proxy security inventory without survivor bias."""

    refs = iter_raw_partitions(
        RAW_TUSHARE_PROXY_STOCK_BASIC,
        workspace_root=workspace_root,
    )
    if not refs:
        return [], {}
    frame = read_raw_partition(refs[-1])
    symbol_column = "ts_code" if "ts_code" in frame.columns else "symbol" if "symbol" in frame.columns else ""
    if not symbol_column:
        raise RuntimeError("tushare_proxy_stock_basic_symbol_column_missing")
    frame = frame.copy()
    frame["_symbol"] = frame[symbol_column].map(normalize_symbol)
    frame = frame.loc[
        frame["_symbol"].map(
            lambda value: len(str(value).split(".", 1)[0]) == 6
            and str(value).split(".", 1)[0].isdigit()
            and str(value).endswith((".SH", ".SZ", ".BJ"))
        )
    ]
    if mainboard_only:
        frame = frame.loc[frame["_symbol"].map(board_for_symbol).eq("MainBoard")]
    symbols = sorted(set(frame["_symbol"]))
    lifecycle: dict[str, tuple[str, str]] = {}
    for payload in frame.to_dict("records"):
        symbol = normalize_symbol(payload.get("_symbol", "") or payload.get(symbol_column, ""))
        if not symbol:
            continue
        listed = str(payload.get("list_date", "") or "")[:10]
        delisted = str(payload.get("delist_date", "") or "")[:10]
        if listed and "-" not in listed and len(listed) == 8:
            listed = f"{listed[:4]}-{listed[4:6]}-{listed[6:]}"
        if delisted and "-" not in delisted and len(delisted) == 8:
            delisted = f"{delisted[:4]}-{delisted[4:6]}-{delisted[6:]}"
        lifecycle[symbol] = (listed, delisted)
    return symbols, lifecycle


def _staging_root(job_id: str, *, workspace_root: str | Path | None) -> Path:
    return qdp_v3_paths(workspace_root).staging / str(job_id)


def _load_or_create_historical_job(
    *,
    job_id: str,
    domain: str,
    start_date: str,
    end_date: str,
    frequency: str,
    task_ids: Iterable[str],
    provider_metadata: Mapping[str, Any],
    workspace_root: str | Path | None,
) -> HistoricalJobState:
    normalized_tasks = sorted({str(item) for item in task_ids if str(item)})
    path = _historical_job_path(job_id, workspace_root=workspace_root)
    payload = read_json(path)
    contract = {
        "provider": TUSHARE_PROXY_NAME,
        "mode": "historical",
        "domain": str(domain),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "frequency": str(frequency),
        "task_ids": normalized_tasks,
    }
    if payload:
        actual = {key: payload.get(key) for key in contract}
        if actual != contract:
            raise RuntimeError(f"historical_job_contract_conflict:{job_id}")
        tasks = {
            key: HistoricalTaskState(**dict(value))
            for key, value in dict(payload.get("tasks", {}) or {}).items()
            if isinstance(value, Mapping)
        }
        return HistoricalJobState(
            job_id=job_id,
            provider=TUSHARE_PROXY_NAME,
            mode="historical",
            domain=str(domain),
            start_date=str(start_date),
            end_date=str(end_date),
            frequency=str(frequency),
            task_ids=normalized_tasks,
            tasks=tasks,
            status=str(payload.get("status", "pending") or "pending"),
            created_at=str(payload.get("created_at", "") or utc_now()),
            updated_at=str(payload.get("updated_at", "") or utc_now()),
            bootstrap_cutoff=str(payload.get("bootstrap_cutoff", "") or ""),
            provider_metadata=dict(payload.get("provider_metadata", {}) or {}),
            metrics=dict(payload.get("metrics", {}) or {}),
            protocol_error_streak=int(payload.get("protocol_error_streak", 0) or 0),
        )
    return HistoricalJobState(
        job_id=job_id,
        provider=TUSHARE_PROXY_NAME,
        mode="historical",
        domain=str(domain),
        start_date=str(start_date),
        end_date=str(end_date),
        frequency=str(frequency),
        task_ids=normalized_tasks,
        tasks={item: HistoricalTaskState(task_id=item) for item in normalized_tasks},
        provider_metadata=dict(redact_secrets(provider_metadata)),
    )


def _save_historical_job(job: HistoricalJobState, *, workspace_root: str | Path | None) -> None:
    job.updated_at = utc_now()
    atomic_write_json(_historical_job_path(job.job_id, workspace_root=workspace_root), job.to_dict())


def _disk_has_capacity(*, workspace_root: str | Path | None, minimum_free_bytes: int) -> tuple[bool, int]:
    paths = ensure_qdp_v3_layout(workspace_root)
    free = int(shutil.disk_usage(paths.root).free)
    return free >= int(minimum_free_bytes), free


def _safe_task_component(value: str) -> str:
    normalized = "".join(item if item.isalnum() or item in "._=-" else "_" for item in str(value))
    if not normalized or normalized in {".", ".."}:
        raise ValueError(f"invalid_historical_task_component:{value!r}")
    return normalized


def _request_params(
    spec: ProxyRawSpec,
    *,
    task_id: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    if spec.request_kind == "trade_date":
        return {"trade_date": str(task_id).replace("-", "")}
    if spec.request_kind == "range":
        return {"start_date": str(start_date).replace("-", ""), "end_date": str(end_date).replace("-", "")}
    if spec.request_kind == "stock_status":
        return {"list_status": str(task_id)}
    if spec.request_kind == "symbol_range":
        return {
            "ts_code": str(task_id),
            "start_date": str(start_date).replace("-", ""),
            "end_date": str(end_date).replace("-", ""),
        }
    raise ValueError(f"unsupported_proxy_request_kind:{spec.request_kind}")


def _reference_partition_value(spec: ProxyRawSpec, *, task_id: str, start_date: str, end_date: str) -> str:
    if spec.request_kind == "range":
        return f"{start_date}_{end_date}"
    if spec.request_kind == "stock_status":
        return str(end_date)
    if spec.raw_domain == RAW_TUSHARE_PROXY_FINANCIAL:
        return f"{spec.api_name}__{task_id}"
    return str(task_id)


def _reference_task_ids(
    spec: ProxyRawSpec,
    *,
    trade_dates: Iterable[str],
    symbols: Iterable[str],
) -> list[str]:
    if spec.request_kind == "trade_date":
        return sorted({str(item)[:10] for item in trade_dates if str(item)})
    if spec.request_kind == "symbol_range":
        return sorted({str(item).strip().upper() for item in symbols if str(item).strip()})
    if spec.request_kind == "stock_status":
        return ["L", "D", "P"]
    if spec.request_kind == "range":
        return ["range"]
    return []


def _fetch_and_store_reference_task(
    *,
    client: TushareProxyClient,
    spec: ProxyRawSpec,
    task_id: str,
    start_date: str,
    end_date: str,
    workspace_root: str | Path | None,
) -> RawPartitionRef:
    partition_value = _reference_partition_value(
        spec,
        task_id=task_id,
        start_date=start_date,
        end_date=end_date,
    )
    if spec.request_kind != "stock_status":
        existing = get_raw_partition(
            spec.raw_domain,
            partition_field=spec.partition_field,
            partition_value=partition_value,
            workspace_root=workspace_root,
        )
        if existing is not None:
            return existing
    params = _request_params(spec, task_id=task_id, start_date=start_date, end_date=end_date)
    result = client.fetch_frame(api_name=spec.api_name, params=params, fields=spec.fields)
    frame = result.frame.copy()
    # stock_basic is queried three times but stored as one as-of partition;
    # merge the immutable latest content rather than overwriting another status.
    if spec.request_kind == "stock_status":
        existing = get_raw_partition(
            spec.raw_domain,
            partition_field=spec.partition_field,
            partition_value=partition_value,
            workspace_root=workspace_root,
        )
        if existing is not None:
            prior = read_raw_partition(existing)
            frame = pd.concat([prior, frame], ignore_index=True, sort=False)
            if "ts_code" in frame.columns:
                frame = frame.drop_duplicates(["ts_code"], keep="last")
    raw_receipt = {
        **client.config.public_metadata(),
        "api_name": spec.api_name,
        "request": result.request_metadata_without_token,
        "response_sha256": result.response_sha256,
        "elapsed_seconds": result.elapsed_seconds,
        "attempts": result.attempts,
        "quality_tier": QUALITY_PROVISIONAL,
        "quality_note": "trusted_historical_source_pending_structural_identity_and_pit_gates",
    }
    writer = write_empty_raw_partition if frame.empty else write_raw_partition
    ref, _ = writer(
        raw_domain=spec.raw_domain,
        partition_field=spec.partition_field,
        partition_value=partition_value,
        frame=frame,
        receipt=raw_receipt,
        workspace_root=workspace_root,
    )
    return ref


def _ingest_tushare_proxy_reference_impl(
    *,
    domain: str,
    start_date: str,
    end_date: str,
    trade_dates: Iterable[str] = (),
    symbols: Iterable[str] = (),
    workspace_root: str | Path | None = None,
    client: TushareProxyClient | None = None,
    resume: bool = True,
    job_id: str = "",
    max_workers: int = MAX_HISTORY_WORKERS,
) -> dict[str, Any]:
    ensure_qdp_v3_layout(workspace_root)
    normalized_domain = str(domain or "").strip().lower().replace("_", "-")
    specs = FINANCIAL_APIS if normalized_domain == "financial" else REFERENCE_SPECS.get(normalized_domain, ())
    if not specs:
        raise ValueError(f"unsupported_tushare_proxy_historical_domain:{domain}")
    source = client or TushareProxyClient()
    all_tasks: list[tuple[ProxyRawSpec, str]] = []
    for spec in specs:
        for task_id in _reference_task_ids(spec, trade_dates=trade_dates, symbols=symbols):
            all_tasks.append((spec, task_id))
    if not all_tasks:
        raise ValueError(f"tushare_proxy_historical_tasks_empty:{normalized_domain}")
    task_ids = [f"{spec.api_name}::{task_id}" for spec, task_id in all_tasks]
    resolved_job_id = str(job_id or f"tushare_proxy__{normalized_domain}__{stable_hash({'start': start_date, 'end': end_date, 'tasks': task_ids}, length=20)}")
    job = _load_or_create_historical_job(
        job_id=resolved_job_id,
        domain=normalized_domain,
        start_date=start_date,
        end_date=end_date,
        frequency="",
        task_ids=task_ids,
        provider_metadata=source.config.public_metadata(),
        workspace_root=workspace_root,
    )
    if not resume:
        for state in job.tasks.values():
            if state.status != "completed":
                state.status = "pending"
                state.error = ""
                state.error_code = ""
    job.status = "running"
    _save_historical_job(job, workspace_root=workspace_root)
    task_lookup = {f"{spec.api_name}::{task_id}": (spec, task_id) for spec, task_id in all_tasks}
    pending = [item for item in job.task_ids if job.tasks[item].status != "completed"]
    lock = threading.Lock()

    def run_one(composite_id: str) -> tuple[str, RawPartitionRef | None, BaseException | None]:
        spec, task_id = task_lookup[composite_id]
        try:
            ref = _fetch_and_store_reference_task(
                client=source,
                spec=spec,
                task_id=task_id,
                start_date=start_date,
                end_date=end_date,
                workspace_root=workspace_root,
            )
            return composite_id, ref, None
        except BaseException as exc:
            return composite_id, None, exc
        finally:
            source.close_thread_session()

    effective_workers = 1 if normalized_domain == "stock-basic" else max(1, min(int(max_workers), MAX_HISTORY_WORKERS))
    completed_since_checkpoint = 0
    performance_gate_failed = False
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        pending_iter = iter(pending)
        futures: dict[Future[tuple[str, RawPartitionRef | None, BaseException | None]], str] = {}

        def fill_queue() -> None:
            while len(futures) < effective_workers * 2:
                try:
                    task_id = next(pending_iter)
                except StopIteration:
                    break
                futures[executor.submit(run_one, task_id)] = task_id

        fill_queue()
        while futures:
            done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
            stop = False
            for future in done:
                futures.pop(future, None)
                composite_id, ref, error = future.result()
                completed_since_checkpoint += 1
                force_checkpoint = error is not None
                with lock:
                    state = job.tasks[composite_id]
                    state.updated_at = utc_now()
                    if error is None and ref is not None:
                        state.status = "completed"
                        state.row_count = int(ref.row_count)
                        state.content_sha256 = ref.content_sha256
                        state.error = ""
                        state.error_code = ""
                        job.protocol_error_streak = 0
                    else:
                        state.status = "pending" if isinstance(error, TushareProxyQuotaError) else "failed"
                        state.error_code = type(error).__name__ if error is not None else "unknown_error"
                        state.error = str(redact_secrets(error, secrets=(source.config.token,)))
                        if isinstance(error, TushareProxyProtocolError):
                            job.protocol_error_streak += 1
                    job.metrics = source.operational_metrics()
                    # Raw partitions are themselves immutable checkpoints.  A
                    # coarse job checkpoint avoids O(task_count^2) JSON I/O;
                    # after a crash, completed-but-uncheckpointed tasks are
                    # recognized from their raw partition without a request.
                    if force_checkpoint or completed_since_checkpoint >= 32:
                        _save_historical_job(job, workspace_root=workspace_root)
                        completed_since_checkpoint = 0
                    if job.protocol_error_streak >= PROTOCOL_CIRCUIT_THRESHOLD:
                        stop = True
                        break
                    if _proxy_performance_gate_failed(job.metrics):
                        performance_gate_failed = True
                        stop = True
                        break
            if stop:
                for queued in futures:
                    queued.cancel()
                break
            fill_queue()
    statuses = [state.status for state in job.tasks.values()]
    export_payload: dict[str, Any] = {}
    export_error: BaseException | None = None
    if all(item == "completed" for item in statuses):
        job.status = "completed"
        try:
            export = RuntimeIndex(workspace_root=workspace_root).export_raw_index(
                qdp_v3_paths(workspace_root).metadata / "raw_index"
            )
            export_payload = {
                "path": str(export.parquet_path.resolve()),
                "sha256": export.sha256,
                "row_count": int(export.row_count),
                "receipts_path": str(export.receipts_path.resolve()),
                "receipts_sha256": export.receipts_sha256,
                "receipt_count": int(export.receipt_count),
            }
        except BaseException as exc:
            export_error = exc
            job.status = "failed_raw_index_export"
    elif any(item == "pending" for item in statuses) and any(
        isinstance_text in state.error.lower()
        for state in job.tasks.values()
        for isinstance_text in ("quota", "entitlement", "过期", "额度")
    ):
        job.status = "paused_quota"
    elif performance_gate_failed:
        job.status = "paused_rate_limit_quality_gate"
    elif job.protocol_error_streak >= PROTOCOL_CIRCUIT_THRESHOLD:
        job.status = "circuit_open_protocol"
    else:
        job.status = "partial"
    job.metrics = source.operational_metrics()
    if export_payload:
        job.metrics["raw_index_export"] = export_payload
    if export_error is not None:
        job.metrics["raw_index_export_error"] = type(export_error).__name__
    _save_historical_job(job, workspace_root=workspace_root)
    if export_error is not None:
        raise RuntimeError(
            f"tushare_proxy_reference_raw_index_export_failed:{type(export_error).__name__}"
        ) from export_error
    return {
        "status": job.status,
        "run_id": job.job_id,
        "domain": normalized_domain,
        "task_count": len(job.tasks),
        "completed_count": sum(state.status == "completed" for state in job.tasks.values()),
        "pending_count": sum(state.status == "pending" for state in job.tasks.values()),
        "failed_count": sum(state.status == "failed" for state in job.tasks.values()),
        "metrics": job.metrics,
        "raw_index_export": export_payload,
        "job_path": str(_historical_job_path(job.job_id, workspace_root=workspace_root).resolve()),
    }


def ingest_tushare_proxy_reference(
    *,
    domain: str,
    start_date: str,
    end_date: str,
    trade_dates: Iterable[str] = (),
    symbols: Iterable[str] = (),
    workspace_root: str | Path | None = None,
    client: TushareProxyClient | None = None,
    resume: bool = True,
    job_id: str = "",
    max_workers: int = MAX_HISTORY_WORKERS,
) -> dict[str, Any]:
    normalized_domain = str(domain or "").strip().lower().replace("_", "-")
    materialized_dates = tuple(str(item) for item in trade_dates)
    materialized_symbols = tuple(str(item) for item in symbols)
    specs = FINANCIAL_APIS if normalized_domain == "financial" else REFERENCE_SPECS.get(normalized_domain, ())
    if not specs:
        raise ValueError(f"unsupported_tushare_proxy_historical_domain:{domain}")
    tasks = [
        f"{spec.api_name}::{task_id}"
        for spec in specs
        for task_id in _reference_task_ids(
            spec,
            trade_dates=materialized_dates,
            symbols=materialized_symbols,
        )
    ]
    if not tasks:
        raise ValueError(f"tushare_proxy_historical_tasks_empty:{normalized_domain}")
    resolved_job_id = str(
        job_id
        or f"tushare_proxy__{normalized_domain}__{stable_hash({'start': start_date, 'end': end_date, 'tasks': tasks}, length=20)}"
    )
    with supervise_job(resolved_job_id, workspace_root=workspace_root):
        return _ingest_tushare_proxy_reference_impl(
            domain=normalized_domain,
            start_date=start_date,
            end_date=end_date,
            trade_dates=materialized_dates,
            symbols=materialized_symbols,
            workspace_root=workspace_root,
            client=client,
            resume=resume,
            job_id=resolved_job_id,
            max_workers=max_workers,
        )


def _timestamp_column(frame: pd.DataFrame) -> str:
    for column in ("trade_time", "datetime", "trade_datetime", "trade_date", "date", "time"):
        if column in frame.columns:
            return column
    return ""


def _staged_page_paths(task_root: Path) -> list[Path]:
    return sorted(task_root.glob("page_*.parquet")) if task_root.exists() else []


def _intraday_task_state_path(task_root: Path) -> Path:
    return task_root / "task_state.json"


def _save_intraday_task_state(task_root: Path, state: HistoricalTaskState) -> None:
    """Persist a small per-security cursor instead of rewriting the global job.

    The minute bootstrap has roughly 380k network pages.  Serialising the
    several-thousand-security global job after every page creates more local
    I/O than the actual payload.  A task-local cursor remains fully durable;
    the global job is a coarse progress index and is checkpointed when a
    security completes or fails.
    """

    task_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(_intraday_task_state_path(task_root), asdict(state))


def _hydrate_intraday_task_state(task_root: Path, state: HistoricalTaskState) -> None:
    payload = read_json(_intraday_task_state_path(task_root))
    if not payload:
        return
    task_id = str(payload.get("task_id", "") or "")
    if task_id and task_id != state.task_id:
        raise RuntimeError(f"historical_task_state_identity_mismatch:{task_root}")
    # Local state is authoritative only while staging pages exist.  A stale
    # file cannot turn an already completed immutable raw partition backwards.
    page_count = int(payload.get("page_count", 0) or 0)
    if page_count < int(state.page_count):
        return
    for field_name in (
        "status",
        "cursor_end_at",
        "page_count",
        "row_count",
        "content_sha256",
        "error_code",
        "error",
        "updated_at",
    ):
        if field_name in payload:
            setattr(state, field_name, payload[field_name])


def _validate_staged_pages(task_root: Path, *, expected_page_count: int) -> list[dict[str, Any]]:
    pages = _staged_page_paths(task_root)
    if len(pages) != int(expected_page_count):
        raise RuntimeError(
            f"historical_staging_page_count_mismatch:{task_root}:{len(pages)}:{expected_page_count}"
        )
    receipts: list[dict[str, Any]] = []
    for index, page_path in enumerate(pages, start=1):
        expected_prefix = f"page_{index:06d}__"
        if not page_path.name.startswith(expected_prefix):
            raise RuntimeError(f"historical_staging_page_sequence_gap:{page_path}")
        receipt_path = page_path.with_suffix(".json")
        receipt = read_json(receipt_path)
        if not receipt:
            raise RuntimeError(f"historical_staging_receipt_missing:{receipt_path}")
        if sha256_file(page_path) != str(receipt.get("parquet_sha256", "") or ""):
            raise RuntimeError(f"historical_staging_page_hash_mismatch:{page_path}")
        receipts.append(receipt)
    return receipts


def _write_staged_page(
    *,
    task_root: Path,
    page_number: int,
    result: Any,
) -> tuple[Path, dict[str, Any]]:
    page_id = f"page_{int(page_number):06d}__{str(result.response_sha256)[:16]}"
    page_path = task_root / f"{page_id}.parquet"
    receipt_path = task_root / f"{page_id}.json"
    task_root.mkdir(parents=True, exist_ok=True)
    if page_path.exists() or receipt_path.exists():
        if not page_path.exists() or not receipt_path.exists():
            raise RuntimeError(f"historical_staging_partial_page:{page_id}")
        receipt = read_json(receipt_path)
        if str(receipt.get("response_sha256", "") or "") != str(result.response_sha256):
            raise RuntimeError(f"historical_staging_response_hash_conflict:{page_id}")
        return page_path, receipt
    atomic_write_parquet(page_path, result.raw_data)
    receipt = {
        **history_page_result_to_receipt(result),
        "page_number": int(page_number),
        "parquet_sha256": sha256_file(page_path),
        "stored_at": utc_now(),
    }
    atomic_write_json(receipt_path, receipt)
    return page_path, receipt


def _combine_intraday_staging(
    *,
    task_root: Path,
    expected_page_count: int,
    start_at: str,
    end_at: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    receipts = _validate_staged_pages(task_root, expected_page_count=expected_page_count)
    frames = [pd.read_parquet(path, engine="pyarrow") for path in _staged_page_paths(task_root)]
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if combined.empty:
        return combined, receipts
    timestamp_column = _timestamp_column(combined)
    if not timestamp_column:
        raise RuntimeError("historical_staging_timestamp_field_missing")
    timestamps = pd.to_datetime(combined[timestamp_column], errors="coerce")
    if timestamps.isna().any():
        raise RuntimeError("historical_staging_timestamp_invalid")
    if timestamps.duplicated().any():
        raise RuntimeError("historical_staging_page_overlap_or_duplicate")
    if len(timestamps) > 1 and not timestamps.reset_index(drop=True).is_monotonic_decreasing:
        raise RuntimeError("historical_staging_global_order_invalid")
    if pd.Timestamp(timestamps.min()) < pd.Timestamp(start_at) or pd.Timestamp(timestamps.max()) > pd.Timestamp(end_at):
        raise RuntimeError("historical_staging_outside_contract_range")
    return combined, receipts


def _merge_intraday_range_frames(
    prior: pd.DataFrame,
    current: pd.DataFrame,
    *,
    symbol: str,
) -> tuple[pd.DataFrame, str]:
    """Merge adjacent immutable captures without inventing an empty timestamp schema."""

    left = prior.copy()
    right = current.copy()
    timestamp_column = _timestamp_column(right) or _timestamp_column(left)
    if not timestamp_column:
        if left.empty and right.empty:
            return pd.concat([left, right], ignore_index=True, sort=False), ""
        raise RuntimeError(f"intraday_range_merge_timestamp_missing:{symbol}")
    for frame in (left, right):
        frame_timestamp_column = _timestamp_column(frame)
        if frame_timestamp_column and frame_timestamp_column != timestamp_column:
            frame.rename(columns={frame_timestamp_column: timestamp_column}, inplace=True)
    merged = pd.concat([left, right], ignore_index=True, sort=False)
    if merged.empty:
        return merged, timestamp_column
    timestamps = pd.to_datetime(merged[timestamp_column], errors="coerce")
    if timestamps.isna().any():
        raise RuntimeError(f"intraday_range_merge_timestamp_invalid:{symbol}")
    duplicate = timestamps.duplicated(keep=False)
    if duplicate.any():
        compare_columns = [column for column in merged.columns if column != "provider_symbol"]
        duplicate_rows = merged.loc[duplicate].assign(_timestamp=timestamps.loc[duplicate])
        for _, group in duplicate_rows.groupby("_timestamp", sort=False):
            values = group.drop(columns=["_timestamp"])[compare_columns]
            if len(values.astype("string").fillna("<NULL>").drop_duplicates()) != 1:
                raise RuntimeError(f"intraday_range_merge_value_conflict:{symbol}")
        merged = (
            merged.assign(_timestamp=timestamps)
            .drop_duplicates("_timestamp", keep="last")
            .drop(columns=["_timestamp"])
        )
    return merged, timestamp_column


def _lifecycle_overlaps(
    symbol: str,
    *,
    start_date: str,
    end_date: str,
    lifecycle_ranges: Mapping[str, tuple[str, str]] | None,
) -> bool:
    if not lifecycle_ranges or symbol not in lifecycle_ranges:
        return True
    listed, delisted = lifecycle_ranges[symbol]
    effective_start = max(str(start_date), str(listed or start_date)[:10])
    effective_end = min(str(end_date), str(delisted or end_date)[:10])
    return effective_start <= effective_end


def _task_time_range(
    symbol: str,
    *,
    start_date: str,
    end_date: str,
    lifecycle_ranges: Mapping[str, tuple[str, str]] | None,
) -> tuple[str, str]:
    listed, delisted = (lifecycle_ranges or {}).get(symbol, ("", ""))
    effective_start = max(str(start_date), str(listed or start_date)[:10])
    effective_end = min(str(end_date), str(delisted or end_date)[:10])
    return f"{effective_start} 00:00:00", f"{effective_end} 23:59:59"


def _receipt_range_bounds(receipt: Mapping[str, Any]) -> tuple[str, str]:
    """Return an explicitly captured request range from a raw receipt.

    Historical residual planning must not infer completeness merely from a
    partition's existence.  Prefer request/coverage bounds over observed
    minima because a security can legitimately have no bar on the first or
    last calendar day of its lifecycle.
    """

    containers: list[Mapping[str, Any]] = [receipt]
    for key in ("coverage", "captured_range", "request", "request_range", "import_range"):
        value = receipt.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    pairs = (
        ("coverage_start_date", "coverage_end_date"),
        ("requested_start_date", "requested_end_date"),
        ("start_date", "end_date"),
        ("start_at", "end_at"),
        ("min_trade_date", "max_trade_date"),
        ("min_timestamp", "max_timestamp"),
    )
    for container in containers:
        for start_key, end_key in pairs:
            start = str(container.get(start_key, "") or "")[:10]
            end = str(container.get(end_key, "") or "")[:10]
            if start and end:
                return start, end
    return "", ""


def _intraday_capture_evidence_root(
    raw_domain: str,
    *,
    workspace_root: str | Path | None,
) -> Path:
    paths = ensure_qdp_v3_layout(workspace_root)
    return paths.metadata / "intraday_capture_evidence" / _safe_task_component(raw_domain)


def _write_intraday_capture_evidence(
    *,
    raw_domain: str,
    provider_symbol: str,
    capture_start_at: str,
    capture_end_at: str,
    capture_kind: str,
    captured_row_count: int,
    result_content_sha256: str,
    network_pages: Iterable[Mapping[str, Any]] = (),
    empty_gap_evidence: Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
) -> Path:
    """Append immutable evidence for one successful residual request.

    Raw receipts are content-addressed and intentionally immutable.  A request
    that returns no new rows may therefore reuse an older raw version and its
    older receipt.  This sidecar records the capture interval without changing
    either object and lets residual planning distinguish "not requested" from
    "requested successfully but empty/duplicate".
    """

    symbol = normalize_symbol(provider_symbol)
    allowed_kinds = {"positive_capture", "empty_provider_gap", "empty_restatement"}
    if capture_kind not in allowed_kinds:
        raise ValueError(f"intraday_capture_evidence_kind_invalid:{capture_kind}")
    start = str(capture_start_at or "")[:10]
    end = str(capture_end_at or "")[:10]
    if not symbol or not start or not end or pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError(f"intraday_capture_evidence_range_invalid:{symbol}:{start}:{end}")
    pages = [
        {
            "page_number": int(item.get("page_number", 0) or 0),
            "response_sha256": str(item.get("response_sha256", "") or ""),
            "row_count": int(item.get("row_count", 0) or 0),
            "min_timestamp": str(item.get("min_timestamp", "") or ""),
            "max_timestamp": str(item.get("max_timestamp", "") or ""),
        }
        for item in network_pages
    ]
    identity = {
        "contract": INTRADAY_CAPTURE_EVIDENCE_CONTRACT,
        "raw_domain": str(raw_domain),
        "provider_symbol": symbol,
        "capture_start_date": start,
        "capture_end_date": end,
        "capture_kind": capture_kind,
        "captured_row_count": int(captured_row_count),
        "result_content_sha256": str(result_content_sha256 or ""),
        "network_pages": pages,
        "empty_gap_evidence": dict(empty_gap_evidence or {}),
    }
    evidence_id = stable_hash(identity)
    destination = (
        _intraday_capture_evidence_root(raw_domain, workspace_root=workspace_root)
        / _safe_task_component(symbol)
        / f"{evidence_id}.json"
    )
    if destination.exists():
        existing = read_json(destination)
        existing_identity = {
            key: existing.get(key)
            for key in identity
        }
        if stable_hash(existing_identity) != evidence_id:
            raise RuntimeError(f"intraday_capture_evidence_identity_mismatch:{destination}")
        return destination
    atomic_write_json(
        destination,
        {
            **identity,
            "evidence_id": evidence_id,
            "captured_at": utc_now(),
        },
    )
    return destination


def _iter_intraday_capture_evidence(
    raw_domain: str,
    *,
    workspace_root: str | Path | None,
) -> list[dict[str, Any]]:
    root = _intraday_capture_evidence_root(raw_domain, workspace_root=workspace_root)
    if not root.exists():
        return []
    evidence: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/*.json")):
        payload = read_json(path)
        if str(payload.get("contract", "") or "") != INTRADAY_CAPTURE_EVIDENCE_CONTRACT:
            raise RuntimeError(f"intraday_capture_evidence_contract_invalid:{path}")
        evidence_id = str(payload.get("evidence_id", "") or "")
        identity = {
            key: payload.get(key)
            for key in (
                "contract",
                "raw_domain",
                "provider_symbol",
                "capture_start_date",
                "capture_end_date",
                "capture_kind",
                "captured_row_count",
                "result_content_sha256",
                "network_pages",
                "empty_gap_evidence",
            )
        }
        if not evidence_id or stable_hash(identity) != evidence_id:
            raise RuntimeError(f"intraday_capture_evidence_hash_mismatch:{path}")
        evidence.append(payload)
    return evidence


def raw_partition_covers_historical_request(
    ref: RawPartitionRef | None,
    receipt: Mapping[str, Any],
    *,
    start_at: str,
    end_at: str,
) -> bool:
    """Whether a positive, usable raw partition proves the requested range.

    In particular, a successful empty response is evidence of a provider gap,
    not evidence that historical market data was captured.  This distinction
    is important for delisted securities and prevents index-only quarantined
    receipts from silently satisfying a resumed job.
    """

    if ref is None or int(ref.row_count) <= 0:
        return False
    quality = str(receipt.get("quality_tier", "") or ref.quality_tier or "").strip().lower()
    if quality == QUALITY_QUARANTINED:
        return False
    if bool(receipt.get("quarantined_empty", False)) or receipt.get("quarantined_empty_evidence"):
        return False
    captured_start, captured_end = _receipt_range_bounds(receipt)
    if not captured_start or not captured_end:
        return False
    return (
        pd.Timestamp(captured_start) <= pd.Timestamp(str(start_at)[:10])
        and pd.Timestamp(captured_end) >= pd.Timestamp(str(end_at)[:10])
    )


def _explicit_empty_provider_gap_covers(
    ref: RawPartitionRef,
    receipt: Mapping[str, Any],
    *,
    start_at: str,
    end_at: str,
) -> bool:
    if int(ref.row_count) != 0:
        return False
    evidence = receipt.get("quarantined_empty_evidence")
    if not bool(receipt.get("quarantined_empty", False)) or not isinstance(evidence, Mapping):
        return False
    if str(evidence.get("reason", "") or "") != "provider_successful_empty_with_lifecycle_overlap":
        return False
    captured_start, captured_end = _receipt_range_bounds(receipt)
    if not captured_start or not captured_end:
        return False
    return (
        pd.Timestamp(captured_start) <= pd.Timestamp(str(start_at)[:10])
        and pd.Timestamp(captured_end) >= pd.Timestamp(str(end_at)[:10])
    )


def _merge_date_intervals(intervals: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    ordered = sorted(
        (pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())
        for start, end in intervals
        if start and end and pd.Timestamp(start) <= pd.Timestamp(end)
    )
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1] + pd.Timedelta(days=1):
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return [(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")) for start, end in merged]


def _subtract_date_intervals(
    target: tuple[str, str],
    covered: Iterable[tuple[str, str]],
) -> list[tuple[str, str]]:
    target_start, target_end = pd.Timestamp(target[0]), pd.Timestamp(target[1])
    cursor = target_start
    gaps: list[tuple[str, str]] = []
    for start_text, end_text in _merge_date_intervals(covered):
        start = max(pd.Timestamp(start_text), target_start)
        end = min(pd.Timestamp(end_text), target_end)
        if end < cursor or start > target_end:
            continue
        if start > cursor:
            gaps.append((cursor.strftime("%Y-%m-%d"), (start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")))
        cursor = max(cursor, end + pd.Timedelta(days=1))
        if cursor > target_end:
            break
    if cursor <= target_end:
        gaps.append((cursor.strftime("%Y-%m-%d"), target_end.strftime("%Y-%m-%d")))
    return gaps


def _intersect_date_intervals(
    target: tuple[str, str],
    intervals: Iterable[tuple[str, str]],
) -> list[tuple[str, str]]:
    target_start, target_end = pd.Timestamp(target[0]), pd.Timestamp(target[1])
    intersections: list[tuple[str, str]] = []
    for start_text, end_text in _merge_date_intervals(intervals):
        start = max(target_start, pd.Timestamp(start_text))
        end = min(target_end, pd.Timestamp(end_text))
        if start <= end:
            intersections.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    return intersections


def _normalize_trade_date_values(values: Iterable[Any]) -> set[str]:
    normalized: set[str] = set()
    for value in values:
        text = str(value or "")[:10]
        try:
            timestamp = pd.Timestamp(text).normalize()
        except (TypeError, ValueError):
            continue
        if pd.isna(timestamp):
            continue
        normalized.add(timestamp.strftime("%Y-%m-%d"))
    return normalized


def _receipt_day_result_dates(
    receipt: Mapping[str, Any],
    *,
    statuses: set[str],
) -> set[str]:
    results = receipt.get("day_results")
    if not isinstance(results, (list, tuple)):
        return set()
    dates: set[str] = set()
    for item in results:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("status", "") or "").strip().lower() not in statuses:
            continue
        dates.update(_normalize_trade_date_values((item.get("trade_date", ""),)))
    return dates


def _receipt_explicit_complete_dates(receipt: Mapping[str, Any]) -> set[str]:
    """Read exact complete-day evidence without treating min/max as coverage.

    The external archive writer evolved while the bootstrap was running, so
    accept the small set of explicit field spellings that appeared in its
    resumable receipts.  An envelope such as ``coverage_start_date`` is
    intentionally not accepted here: it cannot prove internal days.
    """

    dates: set[str] = set()
    for key in (
        "complete_trade_dates",
        "complete_stock_days",
        "complete_stock_day_dates",
        "strict_trade_dates",
    ):
        values = receipt.get(key)
        if isinstance(values, (list, tuple, set, frozenset)):
            dates.update(_normalize_trade_date_values(values))
    dates.update(_receipt_day_result_dates(receipt, statuses={QUALITY_STRICT}))
    return dates


def _frame_complete_5m_dates(frame: pd.DataFrame) -> set[str]:
    """Return only stock-days that physically satisfy the strict 48-bar shape."""

    if frame is None or frame.empty or "trade_date" not in frame.columns:
        return set()
    bar_column = "bar_end" if "bar_end" in frame.columns else "bar_time" if "bar_time" in frame.columns else ""
    if not bar_column:
        return set()
    expected = set(EXPECTED_5M_BAR_ENDS)
    complete: set[str] = set()
    for trade_date, day in frame.groupby("trade_date", sort=False):
        times = day[bar_column].astype(str).str.slice(0, 5)
        if len(day) != len(EXPECTED_5M_BAR_ENDS) or times.duplicated().any() or set(times) != expected:
            continue
        required_numeric = [
            column
            for column in ("open", "high", "low", "close", "volume", "amount")
            if column in day.columns
        ]
        if not {"open", "high", "low", "close"}.issubset(required_numeric):
            continue
        numeric = day.loc[:, required_numeric].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any() or numeric.loc[:, ["open", "high", "low", "close"]].le(0).any().any():
            continue
        if "volume" in numeric and numeric["volume"].lt(0).any():
            continue
        if "amount" in numeric and numeric["amount"].lt(0).any():
            continue
        if (numeric["high"] < numeric.loc[:, ["open", "low", "close"]].max(axis=1)).any():
            continue
        if (numeric["low"] > numeric.loc[:, ["open", "high", "close"]].min(axis=1)).any():
            continue
        complete.update(_normalize_trade_date_values((trade_date,)))
    return complete


def _partition_complete_trade_dates(
    ref: RawPartitionRef,
    receipt: Mapping[str, Any],
    *,
    raw_domain: str,
    workspace_root: str | Path | None,
) -> set[str]:
    """Resolve exact positive coverage for one minute partition.

    Selected source-month partitions are allowed to be quarantined as a
    whole: their strict ``day_results`` and the physical 48-bar payload still
    prove the good days.  All other quarantined partitions remain unusable.
    """

    if int(ref.row_count) <= 0:
        return set()
    quality = str(receipt.get("quality_tier", "") or ref.quality_tier or "").strip().lower()
    if raw_domain != RAW_INTRADAY_5M_SELECTED and quality == QUALITY_QUARANTINED:
        return set()
    explicit_dates = _receipt_explicit_complete_dates(receipt)
    if raw_domain != RAW_INTRADAY_5M_SELECTED and explicit_dates:
        expected_count = int(receipt.get("complete_stock_day_count", 0) or 0)
        if not expected_count or expected_count == len(explicit_dates):
            return explicit_dates

    cache_root = ensure_qdp_v3_layout(workspace_root).jobs / "intraday_residual_coverage"
    cache_path = cache_root / f"{ref.content_sha256}.json"
    if cache_path.exists():
        cached = read_json(cache_path)
        if (
            str(cached.get("contract", "") or "") == "qdp_v3_exact_5m_coverage_v1"
            and str(cached.get("content_sha256", "") or "") == ref.content_sha256
            and int(cached.get("row_count", -1) or -1) == int(ref.row_count)
        ):
            frame_dates = _normalize_trade_date_values(cached.get("complete_trade_dates", ()) or ())
        else:
            frame_dates = set()
    else:
        frame_dates = set()
    if not frame_dates:
        frame_dates = _frame_complete_5m_dates(read_raw_partition(ref))
        cache_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            cache_path,
            {
                "contract": "qdp_v3_exact_5m_coverage_v1",
                "content_sha256": ref.content_sha256,
                "row_count": int(ref.row_count),
                "complete_trade_dates": sorted(frame_dates),
            },
        )
    if not frame_dates:
        return set()
    if raw_domain == RAW_INTRADAY_5M_SELECTED:
        # A selected month can contain strict and quarantined days.  Require
        # both the selector decision and the actual strict bar shape.
        return frame_dates.intersection(explicit_dates)
    # For archive/proxy partitions the payload itself is exact evidence.  If
    # the writer also emitted a complete-day list, use the conservative
    # intersection so a stale or malformed receipt cannot broaden coverage.
    return frame_dates.intersection(explicit_dates) if explicit_dates else frame_dates


def _target_trade_dates(
    *,
    start_date: str,
    end_date: str,
    trade_dates: Iterable[str] | None,
) -> tuple[list[str], str]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if trade_dates is None:
        # This fallback deliberately excludes weekends.  Production callers
        # pass the QDP trading calendar so exchange holidays are excluded too.
        values = pd.bdate_range(start, end)
        source = "weekday_fallback"
    else:
        values = sorted(
            pd.Timestamp(value).normalize()
            for value in _normalize_trade_date_values(trade_dates)
            if start <= pd.Timestamp(value).normalize() <= end
        )
        source = "provided_trade_dates"
    return [value.strftime("%Y-%m-%d") for value in values], source


def _contiguous_trade_date_runs(
    dates: Iterable[str],
    *,
    calendar_dates: list[str],
) -> list[list[str]]:
    requested = sorted(_normalize_trade_date_values(dates))
    if not requested:
        return []
    positions = {value: index for index, value in enumerate(calendar_dates)}
    runs: list[list[str]] = []
    for value in requested:
        if (
            not runs
            or value not in positions
            or runs[-1][-1] not in positions
            or positions[value] != positions[runs[-1][-1]] + 1
        ):
            runs.append([value])
        else:
            runs[-1].append(value)
    return runs


def plan_intraday_residuals(
    *,
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    primary_raw_domains: Iterable[str],
    workspace_root: str | Path | None = None,
    lifecycle_ranges: Mapping[str, tuple[str, str]] | None = None,
    fallback_raw_domain: str = RAW_TUSHARE_PROXY_INTRADAY_5M,
    identity_registry: SecurityIdentityRegistry | None = None,
    supplemental_coverage: Iterable[Mapping[str, Any]] = (),
    trade_dates: Iterable[str] | None = None,
    expected_trade_dates_by_symbol: Mapping[str, Iterable[str]] | None = None,
    treat_fallback_empty_as_known: bool = True,
) -> dict[str, Any]:
    """Plan only the stable-identity gaps left after local historical import.

    A raw partition for any officially configured alias can satisfy another
    alias of the same ``security_id``.  Positive fallback data also satisfies
    coverage, while a previously successful-empty fallback response becomes
    an explicit provider gap and is not pointlessly downloaded again.
    """

    requested = sorted(
        {
            normalize_symbol(item)
            for item in symbols
            if normalize_symbol(item)
            and _lifecycle_overlaps(
                normalize_symbol(item),
                start_date=str(start_date),
                end_date=str(end_date),
                lifecycle_ranges=lifecycle_ranges,
            )
        }
    )
    domains = tuple(dict.fromkeys(str(item) for item in primary_raw_domains if str(item)))
    source_domains = tuple(dict.fromkeys((*domains, str(fallback_raw_domain))))
    refs_by_domain: dict[str, list[RawPartitionRef]] = {
        domain: iter_raw_partitions(domain, workspace_root=workspace_root)
        for domain in source_domains
    }
    capture_evidence_by_domain: dict[str, list[dict[str, Any]]] = {
        domain: _iter_intraday_capture_evidence(domain, workspace_root=workspace_root)
        for domain in source_domains
    }
    provider_symbols = set(requested)
    for refs in refs_by_domain.values():
        provider_symbols.update(
            normalize_symbol(ref.partition_value)
            for ref in refs
            if ref.partition_field == "provider_symbol" and normalize_symbol(ref.partition_value)
        )
    for evidence in capture_evidence_by_domain.values():
        provider_symbols.update(
            normalize_symbol(item.get("provider_symbol", ""))
            for item in evidence
            if normalize_symbol(item.get("provider_symbol", ""))
        )
    registry = identity_registry or SecurityIdentityRegistry.from_sources(
        provider_symbols=sorted(provider_symbols),
        workspace_root=workspace_root,
    )

    calendar_dates, calendar_source = _target_trade_dates(
        start_date=str(start_date),
        end_date=str(end_date),
        trade_dates=trade_dates,
    )
    calendar_set = set(calendar_dates)
    positive_by_security: dict[str, list[tuple[set[str], str, str]]] = {}
    empty_gaps_by_security: dict[str, set[str]] = {}
    for domain, refs in refs_by_domain.items():
        for ref in refs:
            receipt = read_raw_receipt(ref)
            request = receipt.get("request") if isinstance(receipt.get("request"), Mapping) else {}
            provider_symbol = normalize_symbol(
                ref.partition_value
                if ref.partition_field == "provider_symbol"
                else receipt.get("provider_symbol", "")
                or request.get("symbol", "")
            )
            security_id = registry.security_id_for_provider_symbol(provider_symbol)
            if not security_id:
                continue
            captured_start, captured_end = _receipt_range_bounds(receipt)
            complete_dates = _partition_complete_trade_dates(
                ref,
                receipt,
                raw_domain=domain,
                workspace_root=workspace_root,
            )
            if complete_dates:
                positive_by_security.setdefault(security_id, []).append(
                    (complete_dates, domain, provider_symbol)
                )
            elif (
                treat_fallback_empty_as_known
                and domain == str(fallback_raw_domain)
                and _explicit_empty_provider_gap_covers(
                ref,
                receipt,
                start_at=captured_start or start_date,
                end_at=captured_end or end_date,
                )
            ):
                empty_gaps_by_security.setdefault(security_id, set()).update(
                    date
                    for date in calendar_dates
                    if captured_start <= date <= captured_end
                )
    for domain, evidence_items in capture_evidence_by_domain.items():
        for evidence in evidence_items:
            if str(evidence.get("raw_domain", "") or "") != domain:
                raise RuntimeError(
                    "intraday_capture_evidence_domain_mismatch:"
                    f"expected={domain}:actual={evidence.get('raw_domain', '')}"
                )
            provider_symbol = normalize_symbol(evidence.get("provider_symbol", ""))
            security_id = registry.security_id_for_provider_symbol(provider_symbol)
            captured_start = str(evidence.get("capture_start_date", "") or "")[:10]
            captured_end = str(evidence.get("capture_end_date", "") or "")[:10]
            capture_kind = str(evidence.get("capture_kind", "") or "")
            if not security_id or not captured_start or not captured_end:
                continue
            # Positive capture envelopes do not prove every internal stock-day;
            # the immutable raw payload above is the exact evidence.  Empty
            # provider responses, on the other hand, explicitly prove a known
            # provider gap for the requested trading dates.
            if treat_fallback_empty_as_known and domain == str(fallback_raw_domain) and capture_kind in {
                "empty_provider_gap",
                "empty_restatement",
            }:
                empty_gaps_by_security.setdefault(security_id, set()).update(
                    date
                    for date in calendar_dates
                    if captured_start <= date <= captured_end
                )
    for item in supplemental_coverage:
        provider_symbol = normalize_symbol(item.get("provider_symbol", "") or item.get("symbol", ""))
        security_id = registry.security_id_for_provider_symbol(provider_symbol)
        explicit_dates = _normalize_trade_date_values(
            item.get("complete_trade_dates", ()) or item.get("trade_dates", ()) or ()
        )
        captured_start = str(item.get("start_date", "") or "")[:10]
        captured_end = str(item.get("end_date", "") or "")[:10]
        if not explicit_dates and captured_start and captured_end:
            explicit_dates = {
                date
                for date in calendar_dates
                if captured_start <= date <= captured_end
            }
        if security_id and explicit_dates:
            positive_by_security.setdefault(security_id, []).append(
                (
                    explicit_dates,
                    str(item.get("raw_domain", "") or "supplemental_complete_source_route"),
                    provider_symbol,
                )
            )

    covered: list[str] = []
    partially_covered: list[str] = []
    download_spans: list[dict[str, Any]] = []
    known_gaps: list[dict[str, Any]] = []
    alias_coverage: list[dict[str, str]] = []
    for symbol in requested:
        target_start_at, target_end_at = _task_time_range(
            symbol,
            start_date=str(start_date),
            end_date=str(end_date),
            lifecycle_ranges=lifecycle_ranges,
        )
        target_start, target_end = target_start_at[:10], target_end_at[:10]
        security_id = registry.security_id_for_provider_symbol(symbol)
        target_dates = {
            date
            for date in calendar_dates
            if target_start <= date <= target_end
        }
        expected_map = expected_trade_dates_by_symbol or {}
        explicitly_expected = expected_map.get(symbol)
        if explicitly_expected is None:
            explicitly_expected = expected_map.get(security_id)
        if explicitly_expected is not None:
            target_dates.intersection_update(_normalize_trade_date_values(explicitly_expected))
        positive = list(positive_by_security.get(security_id, ()))
        positive_dates: set[str] = set()
        for dates, _domain, _provider_symbol in positive:
            positive_dates.update(dates)
        covered_target_dates = target_dates.intersection(positive_dates)
        uncovered = target_dates.difference(positive_dates)
        if not uncovered:
            covered.append(symbol)
        elif covered_target_dates:
            partially_covered.append(symbol)
        aliases = sorted(
            {
                (item[1], item[2])
                for item in positive
                if item[2] != symbol and target_dates.intersection(item[0])
            }
        )
        for raw_domain, alias in aliases:
            alias_coverage.append(
                {
                    "security_id": security_id,
                    "requested_symbol": symbol,
                    "covered_by_symbol": alias,
                    "raw_domain": raw_domain,
                }
            )
        known_gap_dates = uncovered.intersection(empty_gaps_by_security.get(security_id, set()))
        missing_download_dates = uncovered.difference(known_gap_dates)
        for run in _contiguous_trade_date_runs(known_gap_dates, calendar_dates=calendar_dates):
            known_gaps.append(
                {
                    "security_id": security_id,
                    "provider_symbol": symbol,
                    "start_date": run[0],
                    "end_date": run[-1],
                    "trade_dates": run,
                    "reason": "tushare_proxy_successful_empty_provider_gap",
                }
            )
        for run in _contiguous_trade_date_runs(missing_download_dates, calendar_dates=calendar_dates):
            download_spans.append(
                {
                    "security_id": security_id,
                    "provider_symbol": symbol,
                    "start_date": run[0],
                    "end_date": run[-1],
                    "trade_dates": run,
                }
            )
    grouped: dict[tuple[str, ...], set[str]] = {}
    for item in download_spans:
        key = tuple(str(value) for value in item["trade_dates"])
        grouped.setdefault(key, set()).add(item["provider_symbol"])
    download_groups = [
        {
            "start_date": dates[0],
            "end_date": dates[-1],
            "trade_dates": list(dates),
            "symbols": sorted(symbols),
        }
        for dates, symbols in sorted(grouped.items())
        if dates
    ]
    download = sorted({item["provider_symbol"] for item in download_spans})
    residual = sorted({*download, *(item["provider_symbol"] for item in known_gaps)})
    return {
        "contract": "qdp_v3_intraday_residual_plan_v1",
        "start_date": str(start_date),
        "end_date": str(end_date),
        "trade_calendar_source": calendar_source,
        "trade_date_count": len(calendar_dates),
        "requested_count": len(requested),
        "covered_count": len(covered),
        "partially_covered_count": len(partially_covered),
        "residual_count": len(residual),
        "download_count": len(download),
        "download_span_count": len(download_spans),
        "download_group_count": len(download_groups),
        "known_provider_gap_count": len(known_gaps),
        "covered_symbols": covered,
        "partially_covered_symbols": partially_covered,
        "residual_symbols": residual,
        "download_symbols": download,
        "download_spans": download_spans,
        "download_groups": download_groups,
        "known_provider_gaps": known_gaps,
        "alias_coverage": alias_coverage,
        "primary_raw_domains": list(domains),
        "fallback_raw_domain": str(fallback_raw_domain),
        "treat_fallback_empty_as_known": bool(treat_fallback_empty_as_known),
    }


def _official_restatement_aliases(
    symbols: Iterable[str],
    *,
    workspace_root: str | Path | None,
) -> dict[str, tuple[str, ...]]:
    normalized = sorted({normalize_symbol(item) for item in symbols if normalize_symbol(item)})
    registry = SecurityIdentityRegistry.from_sources(
        provider_symbols=normalized,
        workspace_root=workspace_root,
    )
    available = set(normalized)
    aliases: dict[str, tuple[str, ...]] = {}
    for _, group in registry.symbol_history.groupby("security_id", sort=False):
        group_symbols = sorted(set(group["symbol"].astype(str)) & available)
        if len(group_symbols) < 2:
            continue
        for symbol in group_symbols:
            aliases[symbol] = tuple(item for item in group_symbols if item != symbol)
    return aliases


def _prove_provider_symbol_restatement(
    *,
    client: TushareProxyClient,
    provider_symbol: str,
    aliases: Iterable[str],
    start_at: str,
    end_at: str,
) -> dict[str, Any]:
    for alias in aliases:
        result = client.fetch_history_page(
            HistoryPageFetchRequest(
                provider_symbol=alias,
                start_at=start_at,
                end_at=end_at,
                page_size=1,
            )
        )
        if result.row_count > 0:
            return {
                "reason": "provider_current_code_restatement",
                "empty_provider_symbol": provider_symbol,
                "history_provider_symbol": alias,
                "evidence_response_sha256": result.response_sha256,
                "evidence_timestamp": result.max_timestamp,
            }
    return {}


def _run_intraday_symbol(
    *,
    symbol: str,
    client: TushareProxyClient,
    job: HistoricalJobState,
    job_guard: threading.RLock,
    workspace_root: str | Path | None,
    lifecycle_ranges: Mapping[str, tuple[str, str]] | None,
    restatement_aliases: Mapping[str, tuple[str, ...]],
    minimum_free_bytes: int,
) -> tuple[str, str]:
    state = job.tasks[symbol]
    start_at, end_at = _task_time_range(
        symbol,
        start_date=job.start_date,
        end_date=job.end_date,
        lifecycle_ranges=lifecycle_ranges,
    )
    task_root = _staging_root(job.job_id, workspace_root=workspace_root) / _safe_task_component(symbol)
    _hydrate_intraday_task_state(task_root, state)
    raw_domain = RAW_TUSHARE_PROXY_INTRADAY_5M
    existing = get_raw_partition(
        raw_domain,
        partition_field="provider_symbol",
        partition_value=symbol,
        workspace_root=workspace_root,
    )
    existing_receipt = read_raw_receipt(existing) if existing is not None else {}
    existing_start = str(existing_receipt.get("start_at", "") or "")
    existing_end = str(existing_receipt.get("end_at", "") or "")
    existing_covers_request = raw_partition_covers_historical_request(
        existing,
        existing_receipt,
        start_at=start_at,
        end_at=end_at,
    )
    if existing_covers_request and state.status != "completed":
        with job_guard:
            state.status = "completed"
            state.content_sha256 = existing.content_sha256
            state.row_count = int(existing.row_count)
            state.updated_at = utc_now()
        return symbol, "completed_existing_raw"
    staged_receipts: list[dict[str, Any]] = []
    staged_capture_complete = False
    if state.page_count:
        staged_receipts = _validate_staged_pages(
            task_root,
            expected_page_count=state.page_count,
        )
        expected_cursor = str(staged_receipts[-1].get("next_end_at", "") or "")
        if str(state.cursor_end_at or "") != expected_cursor:
            raise RuntimeError(f"historical_cursor_staging_mismatch:{symbol}")
        staged_capture_complete = bool(staged_receipts[-1].get("is_complete", False))
    cursor_end_at = str(state.cursor_end_at or end_at)
    previous_min = ""
    if state.page_count:
        previous_min = str(staged_receipts[-1].get("min_timestamp", "") or "")
    while not staged_capture_complete:
        has_capacity, free_bytes = _disk_has_capacity(
            workspace_root=workspace_root,
            minimum_free_bytes=minimum_free_bytes,
        )
        if not has_capacity:
            with job_guard:
                state.status = "pending"
                state.error_code = "disk_free_space_gate"
                state.error = f"free_bytes={free_bytes};required={minimum_free_bytes}"
                state.updated_at = utc_now()
                _save_intraday_task_state(task_root, state)
            return symbol, "paused_disk"
        request = HistoryPageFetchRequest(
            provider_symbol=symbol,
            start_at=start_at,
            end_at=cursor_end_at,
            page_size=TUSHARE_PROXY_HISTORY_PAGE_SIZE,
        )
        result = client.fetch_history_page(request)
        if previous_min and result.max_timestamp and pd.Timestamp(result.max_timestamp) >= pd.Timestamp(previous_min):
            raise TushareProxyProtocolError(f"history_page_cross_page_overlap:{symbol}")
        page_number = state.page_count + 1
        _write_staged_page(task_root=task_root, page_number=page_number, result=result)
        with job_guard:
            state.page_count = int(page_number)
            state.row_count += int(result.row_count)
            state.cursor_end_at = str(result.next_end_at)
            state.status = "running"
            state.error_code = ""
            state.error = ""
            state.updated_at = utc_now()
            job.metrics = client.operational_metrics()
            _save_intraday_task_state(task_root, state)
        previous_min = str(result.min_timestamp or previous_min)
        if result.is_complete:
            break
        if not result.next_end_at:
            raise TushareProxyProtocolError(f"history_page_missing_next_cursor:{symbol}")
        if pd.Timestamp(result.next_end_at) >= pd.Timestamp(cursor_end_at):
            raise TushareProxyProtocolError(f"history_page_cursor_not_decreasing:{symbol}")
        cursor_end_at = result.next_end_at
    combined, receipts = _combine_intraday_staging(
        task_root=task_root,
        expected_page_count=state.page_count,
        start_at=start_at,
        end_at=end_at,
    )
    captured_row_count = int(len(combined))
    legitimate_empty_evidence: dict[str, Any] = {}
    quarantined_empty_evidence: dict[str, Any] = {}
    if combined.empty:
        legitimate_empty_evidence = _prove_provider_symbol_restatement(
            client=client,
            provider_symbol=symbol,
            aliases=restatement_aliases.get(symbol, ()),
            start_at=start_at,
            end_at=end_at,
        )
        if not legitimate_empty_evidence:
            # A successful empty response is a provider coverage gap, not a
            # protocol failure.  Keep it explicit and quarantined so a small
            # number of delisted/special securities cannot stall the entire
            # personal-research bootstrap.  Candidate coverage still counts
            # the missing stock-days and can enforce the configured 98% gate.
            quarantined_empty_evidence = {
                "reason": "provider_successful_empty_with_lifecycle_overlap",
                "provider_symbol": symbol,
                "requested_start_at": start_at,
                "requested_end_at": end_at,
                "response_sha256": str(
                    receipts[-1].get("response_sha256", "") if receipts else ""
                ),
            }
    timestamp_column = _timestamp_column(combined)
    combined = combined.copy()
    if "provider_symbol" not in combined.columns:
        combined["provider_symbol"] = symbol
    overall_start_at = start_at
    overall_end_at = end_at
    merged_from_content_sha256 = ""
    if existing is not None and not existing_covers_request:
        prior = read_raw_partition(existing)
        combined, timestamp_column = _merge_intraday_range_frames(
            prior,
            combined,
            symbol=symbol,
        )
        overall_start_at = min(filter(None, (existing_start, start_at)), default=start_at)
        overall_end_at = max(filter(None, (existing_end, end_at)), default=end_at)
        merged_from_content_sha256 = existing.content_sha256
    if timestamp_column and not combined.empty:
        combined = combined.sort_values(timestamp_column, ascending=False).reset_index(drop=True)
    else:
        combined = combined.reset_index(drop=True)
    network_pages = [
        {
            "page_number": int(item.get("page_number", 0) or 0),
            "request": item.get("request", {}),
            "response_sha256": item.get("response_sha256", ""),
            "row_count": int(item.get("row_count", 0) or 0),
            "min_timestamp": item.get("min_timestamp", ""),
            "max_timestamp": item.get("max_timestamp", ""),
        }
        for item in receipts
    ]
    receipt_payload = {
            **client.config.public_metadata(),
            "endpoint": "stk_mins",
            "provider_symbol": symbol,
            "frequency": "5m",
            "start_at": overall_start_at,
            "end_at": overall_end_at,
            "captured_range": {"start_at": start_at, "end_at": end_at},
            "merged_from_content_sha256": merged_from_content_sha256,
            "page_size": TUSHARE_PROXY_HISTORY_PAGE_SIZE,
            "page_count": int(state.page_count),
            "network_pages": network_pages,
            "quality_tier": (
                QUALITY_QUARANTINED
                if quarantined_empty_evidence
                else QUALITY_PROVISIONAL
            ),
            "quality_note": (
                "trusted_source_successful_empty_provider_gap"
                if quarantined_empty_evidence
                else "trusted_historical_source_pending_daily_identity_and_5m_structural_gates"
            ),
            "legitimate_empty": bool(legitimate_empty_evidence),
            "legitimate_empty_evidence": legitimate_empty_evidence,
            "quarantined_empty": bool(quarantined_empty_evidence),
            "quarantined_empty_evidence": quarantined_empty_evidence,
        }
    writer = write_empty_raw_partition if combined.empty else write_raw_partition
    ref, _ = writer(
        raw_domain=raw_domain,
        partition_field="provider_symbol",
        partition_value=symbol,
        frame=combined,
        receipt=receipt_payload,
        workspace_root=workspace_root,
    )
    capture_kind = (
        "positive_capture"
        if captured_row_count > 0
        else "empty_restatement"
        if legitimate_empty_evidence
        else "empty_provider_gap"
    )
    _write_intraday_capture_evidence(
        raw_domain=raw_domain,
        provider_symbol=symbol,
        capture_start_at=start_at,
        capture_end_at=end_at,
        capture_kind=capture_kind,
        captured_row_count=captured_row_count,
        result_content_sha256=ref.content_sha256,
        network_pages=network_pages,
        empty_gap_evidence=(
            legitimate_empty_evidence
            if legitimate_empty_evidence
            else quarantined_empty_evidence
        ),
        workspace_root=workspace_root,
    )
    with job_guard:
        state.status = "completed"
        state.content_sha256 = ref.content_sha256
        state.row_count = int(ref.row_count)
        state.cursor_end_at = ""
        state.error_code = ""
        state.error = ""
        state.updated_at = utc_now()
        job.protocol_error_streak = 0
        job.metrics = client.operational_metrics()
    shutil.rmtree(task_root, ignore_errors=True)
    return symbol, "completed"


def _ingest_tushare_proxy_intraday_impl(
    *,
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    workspace_root: str | Path | None = None,
    client: TushareProxyClient | None = None,
    lifecycle_ranges: Mapping[str, tuple[str, str]] | None = None,
    resume: bool = True,
    job_id: str = "",
    max_workers: int = MAX_HISTORY_WORKERS,
    minimum_free_bytes: int = MIN_FREE_SPACE_BYTES,
) -> dict[str, Any]:
    ensure_qdp_v3_layout(workspace_root)
    normalized_symbols = sorted(
        {
            str(item).strip().upper()
            for item in symbols
            if str(item).strip()
            and _lifecycle_overlaps(
                str(item).strip().upper(),
                start_date=start_date,
                end_date=end_date,
                lifecycle_ranges=lifecycle_ranges,
            )
        }
    )
    if not normalized_symbols:
        raise ValueError("tushare_proxy_intraday_symbols_empty")
    source = client or TushareProxyClient()
    resolved_job_id = str(
        job_id
        or f"tushare_proxy__intraday_5m__{stable_hash({'start': start_date, 'end': end_date, 'symbols': normalized_symbols}, length=20)}"
    )
    job = _load_or_create_historical_job(
        job_id=resolved_job_id,
        domain="intraday",
        start_date=str(start_date),
        end_date=str(end_date),
        frequency="5m",
        task_ids=normalized_symbols,
        provider_metadata=source.config.public_metadata(),
        workspace_root=workspace_root,
    )
    if not resume:
        for state in job.tasks.values():
            if state.status != "completed":
                state.status = "pending"
                state.error = ""
                state.error_code = ""
    job.bootstrap_cutoff = str(end_date)
    job.status = "running"
    _save_historical_job(job, workspace_root=workspace_root)
    pending = [item for item in job.task_ids if job.tasks[item].status != "completed"]
    restatement_aliases = _official_restatement_aliases(
        normalized_symbols,
        workspace_root=workspace_root,
    )
    job_guard = threading.RLock()
    current_worker_limit = _history_worker_limit(max_workers)
    next_index = 0
    live: dict[Future[tuple[str, str]], str] = {}
    stop_reason = ""
    recent_outcomes: list[bool] = []

    def submit_until_limit(executor: ThreadPoolExecutor) -> None:
        nonlocal next_index
        while not stop_reason and len(live) < current_worker_limit and next_index < len(pending):
            symbol = pending[next_index]
            next_index += 1
            future = executor.submit(
                _run_intraday_symbol,
                symbol=symbol,
                client=source,
                job=job,
                job_guard=job_guard,
                workspace_root=workspace_root,
                lifecycle_ranges=lifecycle_ranges,
                restatement_aliases=restatement_aliases,
                minimum_free_bytes=int(minimum_free_bytes),
            )
            live[future] = symbol

    with ThreadPoolExecutor(max_workers=current_worker_limit) as executor:
        submit_until_limit(executor)
        while live:
            done, _ = wait(tuple(live), return_when=FIRST_COMPLETED)
            for future in done:
                symbol = live.pop(future)
                try:
                    _, outcome = future.result()
                    success = outcome.startswith("completed")
                    recent_outcomes.append(success)
                    if outcome in {"paused_disk"}:
                        stop_reason = outcome
                except TushareProxyQuotaError as exc:
                    recent_outcomes.append(False)
                    stop_reason = "paused_quota"
                    with job_guard:
                        state = job.tasks[symbol]
                        state.status = "pending"
                        state.error_code = type(exc).__name__
                        state.error = str(redact_secrets(exc, secrets=(source.config.token,)))
                        state.updated_at = utc_now()
                except TushareProxyProtocolError as exc:
                    recent_outcomes.append(False)
                    with job_guard:
                        state = job.tasks[symbol]
                        state.status = "failed"
                        state.error_code = type(exc).__name__
                        state.error = str(redact_secrets(exc, secrets=(source.config.token,)))
                        state.updated_at = utc_now()
                        job.protocol_error_streak += 1
                        if job.protocol_error_streak >= PROTOCOL_CIRCUIT_THRESHOLD:
                            stop_reason = "circuit_open_protocol"
                except BaseException as exc:
                    recent_outcomes.append(False)
                    with job_guard:
                        state = job.tasks[symbol]
                        state.status = "failed"
                        state.error_code = type(exc).__name__
                        state.error = str(redact_secrets(exc, secrets=(source.config.token,)))
                        state.updated_at = utc_now()
                finally:
                    source.close_thread_session()
                    with job_guard:
                        job.metrics = source.operational_metrics()
                        _save_historical_job(job, workspace_root=workspace_root)
            if _proxy_performance_gate_failed(source.operational_metrics()):
                stop_reason = "paused_rate_limit_quality_gate"
            recent_outcomes = recent_outcomes[-ERROR_WINDOW_SIZE:]
            if len(recent_outcomes) >= ERROR_WINDOW_SIZE:
                error_rate = 1.0 - (sum(recent_outcomes) / len(recent_outcomes))
                if error_rate > ERROR_RATE_WORKER_REDUCTION and current_worker_limit > 1:
                    current_worker_limit = max(1, current_worker_limit // 2)
            if stop_reason:
                for future in live:
                    future.cancel()
            else:
                submit_until_limit(executor)
    statuses = [state.status for state in job.tasks.values()]
    if all(item == "completed" for item in statuses):
        job.status = "completed"
    elif stop_reason:
        job.status = stop_reason
    elif any(item == "failed" for item in statuses):
        job.status = "partial"
    else:
        job.status = "pending"
    job.metrics = source.operational_metrics()
    _save_historical_job(job, workspace_root=workspace_root)
    return {
        "status": job.status,
        "run_id": job.job_id,
        "domain": "intraday",
        "frequency": "5m",
        "start_date": str(start_date),
        "bootstrap_cutoff": str(end_date),
        "task_count": len(job.tasks),
        "completed_count": sum(state.status == "completed" for state in job.tasks.values()),
        "pending_count": sum(state.status in {"pending", "running"} for state in job.tasks.values()),
        "failed_count": sum(state.status == "failed" for state in job.tasks.values()),
        "metrics": job.metrics,
        "worker_limit_final": int(current_worker_limit),
        "minimum_free_bytes": int(minimum_free_bytes),
        "job_path": str(_historical_job_path(job.job_id, workspace_root=workspace_root).resolve()),
    }


def ingest_tushare_proxy_intraday(
    *,
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    workspace_root: str | Path | None = None,
    client: TushareProxyClient | None = None,
    lifecycle_ranges: Mapping[str, tuple[str, str]] | None = None,
    resume: bool = True,
    job_id: str = "",
    max_workers: int = MAX_HISTORY_WORKERS,
    minimum_free_bytes: int = MIN_FREE_SPACE_BYTES,
) -> dict[str, Any]:
    materialized_symbols = tuple(str(item) for item in symbols)
    normalized_symbols = sorted(
        {
            str(item).strip().upper()
            for item in materialized_symbols
            if str(item).strip()
            and _lifecycle_overlaps(
                str(item).strip().upper(),
                start_date=start_date,
                end_date=end_date,
                lifecycle_ranges=lifecycle_ranges,
            )
        }
    )
    if not normalized_symbols:
        raise ValueError("tushare_proxy_intraday_symbols_empty")
    resolved_job_id = str(
        job_id
        or f"tushare_proxy__intraday_5m__{stable_hash({'start': start_date, 'end': end_date, 'symbols': normalized_symbols}, length=20)}"
    )
    with supervise_job(resolved_job_id, workspace_root=workspace_root):
        return _ingest_tushare_proxy_intraday_impl(
            symbols=materialized_symbols,
            start_date=start_date,
            end_date=end_date,
            workspace_root=workspace_root,
            client=client,
            lifecycle_ranges=lifecycle_ranges,
            resume=resume,
            job_id=resolved_job_id,
            max_workers=max_workers,
            minimum_free_bytes=minimum_free_bytes,
        )
