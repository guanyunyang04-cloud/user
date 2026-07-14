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
from quant_data_platform.domains.contracts import DataDomain, HistoryPageFetchRequest
from quant_data_platform.qdp_v3.constants import (
    QUALITY_PROVISIONAL,
    QUALITY_QUARANTINED,
    RAW_TUSHARE_PROXY_ADJ_FACTOR,
    RAW_TUSHARE_PROXY_DAILY,
    RAW_TUSHARE_PROXY_DAILY_BASIC,
    RAW_TUSHARE_PROXY_DIVIDEND,
    RAW_TUSHARE_PROXY_FINANCIAL,
    RAW_TUSHARE_PROXY_INTRADAY_1M,
    RAW_TUSHARE_PROXY_INTRADAY_5M,
    RAW_TUSHARE_PROXY_NAMECHANGE,
    RAW_TUSHARE_PROXY_STK_LIMIT,
    RAW_TUSHARE_PROXY_STOCK_BASIC,
    RAW_TUSHARE_PROXY_SUSPEND,
    RAW_TUSHARE_PROXY_TRADE_CALENDAR,
)
from quant_data_platform.qdp_v3.manifest import atomic_write_json, sha256_file, stable_hash, utc_now
from quant_data_platform.qdp_v3.identity import board_for_symbol, normalize_symbol
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout, qdp_v3_paths
from quant_data_platform.qdp_v3.storage import (
    RawPartitionRef,
    atomic_write_parquet,
    get_raw_partition,
    write_raw_partition,
    read_raw_receipt,
)
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
MAX_HISTORY_WORKERS = 4
PROTOCOL_CIRCUIT_THRESHOLD = 5
ERROR_WINDOW_SIZE = 200
ERROR_RATE_WORKER_REDUCTION = 0.02


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
        ProxyRawSpec(
            name="stk_limit",
            api_name="stk_limit",
            raw_domain=RAW_TUSHARE_PROXY_STK_LIMIT,
            partition_field="trade_date",
            fields=("trade_date", "ts_code", "pre_close", "up_limit", "down_limit"),
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

    paths = qdp_v3_paths(workspace_root)
    domain_root = paths.raw / RAW_TUSHARE_PROXY_STOCK_BASIC
    refs: list[RawPartitionRef] = []
    if domain_root.exists():
        for latest_path in sorted(domain_root.glob("*/latest.json")):
            latest = read_json(latest_path)
            version_path = Path(str(latest.get("version_path", "") or ""))
            if not version_path.is_absolute():
                version_path = paths.root / version_path
            receipt = read_json(version_path / "receipt.json")
            payload_path = version_path / "payload.parquet"
            if receipt and payload_path.exists():
                refs.append(
                    RawPartitionRef(
                        raw_domain=RAW_TUSHARE_PROXY_STOCK_BASIC,
                        partition_field=str(receipt.get("partition_field", "") or "as_of_date"),
                        partition_value=str(receipt.get("partition_value", "") or ""),
                        content_sha256=str(receipt.get("content_sha256", "") or ""),
                        payload_path=payload_path,
                        receipt_path=version_path / "receipt.json",
                        row_count=int(receipt.get("row_count", 0) or 0),
                        quality_tier=str(receipt.get("quality_tier", "") or QUALITY_PROVISIONAL),
                    )
                )
    if not refs:
        return [], {}
    frame = pd.read_parquet(refs[-1].payload_path, engine="pyarrow")
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
    return qdp_v3_paths(workspace_root).jobs / "staging" / str(job_id)


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
            prior = pd.read_parquet(existing.payload_path, engine="pyarrow")
            frame = pd.concat([prior, frame], ignore_index=True, sort=False)
            if "ts_code" in frame.columns:
                frame = frame.drop_duplicates(["ts_code"], keep="last")
    ref, _ = write_raw_partition(
        raw_domain=spec.raw_domain,
        partition_field=spec.partition_field,
        partition_value=partition_value,
        frame=frame,
        receipt={
            **client.config.public_metadata(),
            "api_name": spec.api_name,
            "request": result.request_metadata_without_token,
            "response_sha256": result.response_sha256,
            "elapsed_seconds": result.elapsed_seconds,
            "attempts": result.attempts,
            "quality_tier": QUALITY_PROVISIONAL,
            "quality_note": "third_party_historical_bootstrap_requires_cross_source_canonical_audit",
        },
        workspace_root=workspace_root,
    )
    return ref


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
                    job.metrics = source.metrics.snapshot()
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
            if stop:
                for queued in futures:
                    queued.cancel()
                break
            fill_queue()
    statuses = [state.status for state in job.tasks.values()]
    if all(item == "completed" for item in statuses):
        job.status = "completed"
    elif any(item == "pending" for item in statuses) and any(
        isinstance_text in state.error.lower()
        for state in job.tasks.values()
        for isinstance_text in ("quota", "entitlement", "过期", "额度")
    ):
        job.status = "paused_quota"
    elif job.protocol_error_streak >= PROTOCOL_CIRCUIT_THRESHOLD:
        job.status = "circuit_open_protocol"
    else:
        job.status = "partial"
    job.metrics = source.metrics.snapshot()
    _save_historical_job(job, workspace_root=workspace_root)
    return {
        "status": job.status,
        "run_id": job.job_id,
        "domain": normalized_domain,
        "task_count": len(job.tasks),
        "completed_count": sum(state.status == "completed" for state in job.tasks.values()),
        "pending_count": sum(state.status == "pending" for state in job.tasks.values()),
        "failed_count": sum(state.status == "failed" for state in job.tasks.values()),
        "metrics": job.metrics,
        "job_path": str(_historical_job_path(job.job_id, workspace_root=workspace_root).resolve()),
    }


def _intraday_raw_domain(frequency: str) -> str:
    normalized = str(frequency).strip().lower()
    if normalized == "1m":
        return RAW_TUSHARE_PROXY_INTRADAY_1M
    if normalized == "5m":
        return RAW_TUSHARE_PROXY_INTRADAY_5M
    raise ValueError(f"unsupported_intraday_frequency:{frequency}")


def _intraday_data_domain(frequency: str) -> str:
    return DataDomain.MARKET_INTRADAY_1M if frequency == "1m" else DataDomain.MARKET_INTRADAY_5M


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


def _run_intraday_symbol(
    *,
    symbol: str,
    frequency: str,
    client: TushareProxyClient,
    job: HistoricalJobState,
    job_guard: threading.RLock,
    workspace_root: str | Path | None,
    lifecycle_ranges: Mapping[str, tuple[str, str]] | None,
    minimum_free_bytes: int,
) -> tuple[str, str]:
    state = job.tasks[symbol]
    start_at, end_at = _task_time_range(
        symbol,
        start_date=job.start_date,
        end_date=job.end_date,
        lifecycle_ranges=lifecycle_ranges,
    )
    task_root = _staging_root(job.job_id, workspace_root=workspace_root) / _safe_task_component(f"{symbol}__{frequency}")
    _hydrate_intraday_task_state(task_root, state)
    raw_domain = _intraday_raw_domain(frequency)
    existing = get_raw_partition(
        raw_domain,
        partition_field="provider_symbol_frequency",
        partition_value=f"{symbol}__{frequency}",
        workspace_root=workspace_root,
    )
    existing_receipt = read_raw_receipt(existing) if existing is not None else {}
    existing_start = str(existing_receipt.get("start_at", "") or "")
    existing_end = str(existing_receipt.get("end_at", "") or "")
    existing_covers_request = bool(
        existing is not None
        and existing_start
        and existing_end
        and pd.Timestamp(existing_start) <= pd.Timestamp(start_at)
        and pd.Timestamp(existing_end) >= pd.Timestamp(end_at)
    )
    if existing_covers_request and state.status != "completed":
        with job_guard:
            state.status = "completed"
            state.content_sha256 = existing.content_sha256
            state.row_count = int(existing.row_count)
            state.updated_at = utc_now()
        return symbol, "completed_existing_raw"
    if state.page_count:
        receipts = _validate_staged_pages(task_root, expected_page_count=state.page_count)
        expected_cursor = str(receipts[-1].get("next_end_at", "") or "")
        if str(state.cursor_end_at or "") != expected_cursor:
            raise RuntimeError(f"historical_cursor_staging_mismatch:{symbol}:{frequency}")
    cursor_end_at = str(state.cursor_end_at or end_at)
    previous_min = ""
    if state.page_count:
        staged_receipts = _validate_staged_pages(task_root, expected_page_count=state.page_count)
        previous_min = str(staged_receipts[-1].get("min_timestamp", "") or "")
    while True:
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
            domain=_intraday_data_domain(frequency),
            provider_symbol=symbol,
            frequency=frequency,
            start_at=start_at,
            end_at=cursor_end_at,
            page_size=TUSHARE_PROXY_HISTORY_PAGE_SIZE,
        )
        result = client.fetch_history_page(request)
        if previous_min and result.max_timestamp and pd.Timestamp(result.max_timestamp) >= pd.Timestamp(previous_min):
            raise TushareProxyProtocolError(f"history_page_cross_page_overlap:{symbol}:{frequency}")
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
            job.metrics = client.metrics.snapshot()
            _save_intraday_task_state(task_root, state)
        previous_min = str(result.min_timestamp or previous_min)
        if result.is_complete:
            break
        if not result.next_end_at:
            raise TushareProxyProtocolError(f"history_page_missing_next_cursor:{symbol}:{frequency}")
        if pd.Timestamp(result.next_end_at) >= pd.Timestamp(cursor_end_at):
            raise TushareProxyProtocolError(f"history_page_cursor_not_decreasing:{symbol}:{frequency}")
        cursor_end_at = result.next_end_at
    combined, receipts = _combine_intraday_staging(
        task_root=task_root,
        expected_page_count=state.page_count,
        start_at=start_at,
        end_at=end_at,
    )
    if combined.empty:
        raise RuntimeError(f"intraday_empty_despite_lifecycle_overlap:{symbol}:{frequency}")
    timestamp_column = _timestamp_column(combined)
    combined = combined.copy()
    if "provider_symbol" not in combined.columns:
        combined["provider_symbol"] = symbol
    overall_start_at = start_at
    overall_end_at = end_at
    merged_from_content_sha256 = ""
    if existing is not None and not existing_covers_request:
        prior = read_raw_partition(existing)
        prior_timestamp_column = _timestamp_column(prior)
        if not prior.empty and prior_timestamp_column != timestamp_column:
            prior = prior.rename(columns={prior_timestamp_column: timestamp_column})
        merged = pd.concat([prior, combined], ignore_index=True, sort=False)
        timestamps = pd.to_datetime(merged[timestamp_column], errors="coerce")
        if timestamps.isna().any():
            raise RuntimeError(f"intraday_range_merge_timestamp_invalid:{symbol}:{frequency}")
        duplicate = timestamps.duplicated(keep=False)
        if duplicate.any():
            compare_columns = [column for column in merged.columns if column != "provider_symbol"]
            for _, group in merged.loc[duplicate].assign(_timestamp=timestamps.loc[duplicate]).groupby("_timestamp", sort=False):
                if len(group.drop(columns=["_timestamp"])[compare_columns].astype("string").fillna("<NULL>").drop_duplicates()) != 1:
                    raise RuntimeError(f"intraday_range_merge_value_conflict:{symbol}:{frequency}")
            merged = merged.assign(_timestamp=timestamps).drop_duplicates("_timestamp", keep="last").drop(columns=["_timestamp"])
        combined = merged
        overall_start_at = min(filter(None, (existing_start, start_at)), default=start_at)
        overall_end_at = max(filter(None, (existing_end, end_at)), default=end_at)
        merged_from_content_sha256 = existing.content_sha256
    combined = combined.sort_values(timestamp_column, ascending=False).reset_index(drop=True)
    ref, _ = write_raw_partition(
        raw_domain=raw_domain,
        partition_field="provider_symbol_frequency",
        partition_value=f"{symbol}__{frequency}",
        frame=combined,
        receipt={
            **client.config.public_metadata(),
            "endpoint": "stk_mins",
            "provider_symbol": symbol,
            "frequency": frequency,
            "start_at": overall_start_at,
            "end_at": overall_end_at,
            "captured_range": {"start_at": start_at, "end_at": end_at},
            "merged_from_content_sha256": merged_from_content_sha256,
            "page_size": TUSHARE_PROXY_HISTORY_PAGE_SIZE,
            "page_count": int(state.page_count),
            "network_pages": [
                {
                    "page_number": int(item.get("page_number", 0) or 0),
                    "request": item.get("request", {}),
                    "response_sha256": item.get("response_sha256", ""),
                    "row_count": int(item.get("row_count", 0) or 0),
                    "min_timestamp": item.get("min_timestamp", ""),
                    "max_timestamp": item.get("max_timestamp", ""),
                }
                for item in receipts
            ],
            "quality_tier": QUALITY_PROVISIONAL,
            "quality_note": "raw_complete_symbol_history_pending_daily_identity_and_1m_5m_semantic_gates",
        },
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
        job.metrics = client.metrics.snapshot()
    shutil.rmtree(task_root, ignore_errors=True)
    return symbol, "completed"


def ingest_tushare_proxy_intraday(
    *,
    symbols: Iterable[str],
    frequency: str,
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
    normalized_frequency = str(frequency or "").strip().lower()
    _intraday_raw_domain(normalized_frequency)
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
        or f"tushare_proxy__intraday_{normalized_frequency}__{stable_hash({'start': start_date, 'end': end_date, 'symbols': normalized_symbols}, length=20)}"
    )
    job = _load_or_create_historical_job(
        job_id=resolved_job_id,
        domain="intraday",
        start_date=str(start_date),
        end_date=str(end_date),
        frequency=normalized_frequency,
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
    job_guard = threading.RLock()
    current_worker_limit = max(1, min(int(max_workers), MAX_HISTORY_WORKERS))
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
                frequency=normalized_frequency,
                client=source,
                job=job,
                job_guard=job_guard,
                workspace_root=workspace_root,
                lifecycle_ranges=lifecycle_ranges,
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
                        job.metrics = source.metrics.snapshot()
                        _save_historical_job(job, workspace_root=workspace_root)
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
    job.metrics = source.metrics.snapshot()
    _save_historical_job(job, workspace_root=workspace_root)
    return {
        "status": job.status,
        "run_id": job.job_id,
        "domain": "intraday",
        "frequency": normalized_frequency,
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


def ingest_tushare_proxy_intraday_pair(
    *,
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    workspace_root: str | Path | None = None,
    lifecycle_ranges: Mapping[str, tuple[str, str]] | None = None,
    resume: bool = True,
    run_label: str = "",
    minimum_free_bytes: int = MIN_FREE_SPACE_BYTES,
) -> dict[str, Any]:
    """Run 1m/5m capture together with an approximately 5:1 request bias."""

    shared_client = TushareProxyClient()
    normalized_symbols = tuple(sorted({str(item).strip().upper() for item in symbols if str(item).strip()}))
    label = str(run_label or stable_hash({"start": start_date, "end": end_date}, length=12))

    def run(frequency: str, workers: int) -> dict[str, Any]:
        return ingest_tushare_proxy_intraday(
            symbols=normalized_symbols,
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            workspace_root=workspace_root,
            client=shared_client,
            lifecycle_ranges=lifecycle_ranges,
            resume=resume,
            job_id=f"tushare_proxy__intraday_{frequency}__{label}",
            max_workers=workers,
            minimum_free_bytes=minimum_free_bytes,
        )

    # Four 1m workers and one 5m worker share a three-request semaphore and
    # one token bucket. This produces the requested ~83/17 allocation without
    # ever exceeding the account's actual concurrency contract.
    with ThreadPoolExecutor(max_workers=2) as executor:
        one_future = executor.submit(run, "1m", 4)
        five_future = executor.submit(run, "5m", 1)
        one = one_future.result()
        five = five_future.result()
    statuses = {str(one.get("status", "")), str(five.get("status", ""))}
    if statuses == {"completed"}:
        status = "completed"
    elif "paused_quota" in statuses:
        status = "paused_quota"
    elif "paused_disk" in statuses:
        status = "paused_disk"
    elif "circuit_open_protocol" in statuses:
        status = "circuit_open_protocol"
    else:
        status = "partial"
    return {
        "status": status,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "allocation_policy": "four_1m_workers_one_5m_worker_shared_135rpm_limiter",
        "one_minute": one,
        "five_minute": five,
        "metrics": shared_client.metrics.snapshot(),
    }
