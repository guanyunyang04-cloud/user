"""Tushare Extended Backfill: download responsibilities."""

from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from quantlab.data.qdp_v2.auxiliary_update import (
    _resolve_tushare_token,
)
from quantlab.data.qdp_v2.manifest import (
    atomic_write_json,
    utc_now,
)

from .config import (
    END_DATE,
    FORBIDDEN_YEAR,
    MAX_PAGES_PER_TASK,
    MAX_REQUESTS_PER_MINUTE,
    MAX_WORKERS,
    PAGE_SIZE,
    SPECS,
    EndpointSpec,
    ResilientTushareClient,
    TushareExtendedBackfillError,
)
from .context import (
    _assert_credential_free,
    _open_dates,
    _provider_cache_frame,
    _read_state,
    _runtime,
    _sha256,
    _stable_hash,
    _validate_schema,
    _write_parquet,
    _write_state,
)
from .probe import (
    probe,
)


def _task_keys(spec: EndpointSpec, open_dates: Sequence[str]) -> list[str]:
    if spec.mode == "year":
        return [str(year) for year in range(2010, 2026)]
    return list(open_dates)


def _task_params(spec: EndpointSpec, task_key: str, offset: int) -> dict[str, Any]:
    if spec.mode == "year":
        year = int(task_key)
        return {
            "start_date": f"{year}0101",
            "end_date": f"{year}1231",
            "limit": PAGE_SIZE,
            "offset": int(offset),
        }
    compact = str(task_key).replace("-", "")
    params: dict[str, Any] = {
        "trade_date": compact,
        "limit": PAGE_SIZE,
        "offset": int(offset),
    }
    if spec.name == "margin-secs":
        params["exchange"] = ""
    return params


def _task_dir(workspace: Path, spec: EndpointSpec, task_key: str) -> Path:
    year = str(task_key)[:4]
    stable_key = str(task_key).replace("-", "")
    return _runtime(workspace) / "raw" / spec.name / f"year={year}" / f"task={stable_key}"


def _page_paths(workspace: Path, spec: EndpointSpec, task_key: str, offset: int) -> tuple[Path, Path]:
    root = _task_dir(workspace, spec, task_key)
    stem = f"offset={int(offset):09d}"
    return root / f"{stem}.parquet", root / f"{stem}.json"


def _success_path(workspace: Path, spec: EndpointSpec, task_key: str) -> Path:
    return _task_dir(workspace, spec, task_key) / "_SUCCESS.json"


def _request_fingerprint(spec: EndpointSpec, task_key: str) -> str:
    return _stable_hash(
        {
            "endpoint": spec.api_name,
            "task_key": task_key,
            "fields": list(spec.fields),
            "page_size": PAGE_SIZE,
            "end_date": END_DATE,
        }
    )


def _valid_cached_page(path: Path, sidecar: Path) -> dict[str, Any] | None:
    if not path.is_file() or not sidecar.is_file():
        return None
    try:
        record = dict(json.loads(sidecar.read_text(encoding="utf-8")))
        if record.get("sha256") != _sha256(path):
            return None
        if int(record.get("row_count", -1)) != int(pq.ParquetFile(path).metadata.num_rows):
            return None
        return record
    except Exception:  # noqa: BLE001 - corrupt cache is treated as a miss
        return None


def _valid_success(workspace: Path, spec: EndpointSpec, task_key: str) -> dict[str, Any] | None:
    path = _success_path(workspace, spec, task_key)
    if not path.is_file():
        return None
    try:
        record = dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001 - corrupt cache is treated as a miss
        return None
    if record.get("request_fingerprint") != _request_fingerprint(spec, task_key):
        return None
    for page in list(record.get("pages", []) or []):
        parquet = Path(str(page.get("path", "")))
        if not parquet.is_file() or page.get("sha256") != _sha256(parquet):
            return None
    return record


def _download_task(
    workspace: Path,
    spec: EndpointSpec,
    task_key: str,
    client: ResilientTushareClient,
) -> dict[str, Any]:
    cached = _valid_success(workspace, spec, task_key)
    if cached is not None:
        return {**cached, "reused": True}
    pages: list[dict[str, Any]] = []
    offset = 0
    while True:
        if offset > PAGE_SIZE * MAX_PAGES_PER_TASK:
            raise TushareExtendedBackfillError(f"pagination_guard_exceeded:{spec.name}:{task_key}")
        parquet, sidecar = _page_paths(workspace, spec, task_key, offset)
        record = _valid_cached_page(parquet, sidecar)
        if record is None:
            params = _task_params(spec, task_key, offset)
            frame = client.fetch(
                spec.api_name,
                params=params,
                fields=spec.fields,
            )
            schema_digest = _validate_schema(spec, frame.columns)
            if "trade_date" in frame.columns and len(frame):
                dates = frame["trade_date"].astype(str).str.replace("-", "", regex=False)
                if bool(dates.str.startswith(str(FORBIDDEN_YEAR)).any()):
                    raise TushareExtendedBackfillError("forbidden_2026_provider_row")
                if str(dates.max()) > END_DATE.replace("-", ""):
                    raise TushareExtendedBackfillError("provider_row_after_end_date")
            cache = _provider_cache_frame(frame)
            _write_parquet(cache, parquet)
            record = {
                "endpoint": spec.api_name,
                "task_key": task_key,
                "offset": int(offset),
                "row_count": len(cache),
                "field_count": len(cache.columns),
                "schema_hash": schema_digest,
                "request_hash": _stable_hash({"api_name": spec.api_name, "params": params, "fields": spec.fields}),
                "path": str(parquet.resolve()),
                "sha256": _sha256(parquet),
                "downloaded_at": utc_now(),
                "credential_persisted": False,
            }
            _assert_credential_free(record)
            atomic_write_json(sidecar, record)
        pages.append(record)
        if int(record["row_count"]) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    success = {
        "status": "completed",
        "endpoint": spec.name,
        "task_key": task_key,
        "year": int(str(task_key)[:4]),
        "burn_in_only": int(str(task_key)[:4]) == 2010,
        "request_fingerprint": _request_fingerprint(spec, task_key),
        "row_count": sum(int(item["row_count"]) for item in pages),
        "page_count": len(pages),
        "pages": pages,
        "credential_persisted": False,
    }
    _assert_credential_free(success)
    atomic_write_json(_success_path(workspace, spec, task_key), success)
    return success


def _selected_specs(domains: str | Sequence[str] | None) -> list[EndpointSpec]:
    if domains is None or domains == "":
        names = list(SPECS)
    elif isinstance(domains, str):
        names = [item.strip() for item in domains.split(",") if item.strip()]
    else:
        names = [str(item).strip() for item in domains if str(item).strip()]
    unknown = sorted(set(names).difference(SPECS))
    if unknown:
        raise TushareExtendedBackfillError(f"unknown_domains:{','.join(unknown)}")
    return [SPECS[name] for name in names]


def _download_sources(
    workspace: Path,
    specs: Sequence[EndpointSpec],
) -> tuple[dict[str, Any], dict[str, Any], ResilientTushareClient]:
    state = _read_state(workspace)
    probe_state = dict(state.get("probe", {}) or {})
    if probe_state.get("status") != "completed":
        probe_state = probe(workspace_root=workspace)
        state = _read_state(workspace)
    endpoint_probe = dict(probe_state.get("endpoints", {}) or {})
    unavailable = [
        spec.name
        for spec in specs
        if not spec.optional
        and dict(endpoint_probe.get(spec.name, {}) or {}).get("status")
        not in {"available", "available_with_partial_probe"}
    ]
    if unavailable:
        raise TushareExtendedBackfillError(f"mandatory_source_unavailable:{','.join(unavailable)}")
    token = _resolve_tushare_token(workspace)
    if not token:
        raise TushareExtendedBackfillError("tushare_token_required")
    client = ResilientTushareClient(token, rpm=MAX_REQUESTS_PER_MINUTE, workspace_root=workspace)
    return state, endpoint_probe, client


def _pending_download_tasks(
    *,
    workspace: Path,
    specs: Sequence[EndpointSpec],
    endpoint_probe: dict[str, Any],
) -> tuple[list[tuple[EndpointSpec, str]], dict[str, Any]]:
    tasks: list[tuple[EndpointSpec, str]] = []
    skipped: dict[str, Any] = {}
    open_dates = _open_dates(workspace)
    for spec in specs:
        availability = dict(endpoint_probe.get(spec.name, {}) or {}).get("status")
        if spec.optional and availability not in {"available", "available_with_partial_probe"}:
            skipped[spec.name] = {"status": "source_unavailable", "reason": "probe_failed_or_empty"}
            continue
        tasks.extend(
            (spec, task_key)
            for task_key in _task_keys(spec, open_dates)
            if _valid_success(workspace, spec, task_key) is None
        )
    return tasks, skipped


def _checkpoint_download(
    *,
    workspace: Path,
    planned: int,
    completed: int,
    failed: int,
) -> None:
    state = _read_state(workspace)
    state["download_progress"] = {
        "planned": planned,
        "completed": completed,
        "failed": failed,
        "last_checkpoint_at": utc_now(),
    }
    _write_state(workspace, state)
    print(json.dumps(state["download_progress"], ensure_ascii=False, sort_keys=True), flush=True)


def _run_download_tasks(
    *,
    workspace: Path,
    tasks: list[tuple[EndpointSpec, str]],
    client: ResilientTushareClient,
    worker_count: int,
) -> tuple[list[dict[str, Any]], list[tuple[EndpointSpec, str, str]]]:
    completed: list[dict[str, Any]] = []
    failures: list[tuple[EndpointSpec, str, str]] = []

    def run(item: tuple[EndpointSpec, str]) -> dict[str, Any]:
        spec, task_key = item
        return _download_task(workspace, spec, task_key, client)

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        future_map = {pool.submit(run, item): item for item in tasks}
        for index, future in enumerate(as_completed(future_map), start=1):
            spec, task_key = future_map[future]
            try:
                completed.append(future.result())
            except Exception as exc:  # noqa: BLE001 - isolate task failures for resume
                failures.append((spec, task_key, f"{type(exc).__name__}:{str(exc)[:300]}"))
            if index % 20 == 0 or index == len(tasks):
                _checkpoint_download(
                    workspace=workspace,
                    planned=len(tasks),
                    completed=len(completed),
                    failed=len(failures),
                )
    return completed, failures


def _retry_download_failures(
    *,
    workspace: Path,
    failures: list[tuple[EndpointSpec, str, str]],
    completed: list[dict[str, Any]],
    client: ResilientTushareClient,
) -> list[tuple[EndpointSpec, str, str]]:
    remaining: list[tuple[EndpointSpec, str, str]] = []
    for spec, task_key, _ in failures:
        try:
            completed.append(_download_task(workspace, spec, task_key, client))
        except Exception as exc:  # noqa: BLE001 - isolate task failures for resume
            remaining.append((spec, task_key, f"{type(exc).__name__}:{str(exc)[:300]}"))
    return remaining


def _download_inventory(
    *,
    workspace: Path,
    specs: Sequence[EndpointSpec],
    workers: int,
) -> dict[str, Any]:
    state, endpoint_probe, client = _download_sources(workspace, specs)
    tasks, skipped = _pending_download_tasks(
        workspace=workspace,
        specs=specs,
        endpoint_probe=endpoint_probe,
    )
    worker_count = max(1, min(int(workers), MAX_WORKERS))
    state["status"] = "downloading"
    state["download_plan"] = {
        "task_count": len(tasks),
        "workers": worker_count,
        "selected_domains": [spec.name for spec in specs],
        "optional_sources": skipped,
        "forbidden_2026_tasks": 0,
    }
    _write_state(workspace, state)
    completed, failures = _run_download_tasks(
        workspace=workspace,
        tasks=tasks,
        client=client,
        worker_count=worker_count,
    )
    if failures and worker_count > 1:
        failures = _retry_download_failures(
            workspace=workspace,
            failures=failures,
            completed=completed,
            client=client,
        )
    state = _read_state(workspace)
    state["download_progress"] = {
        "planned": len(tasks),
        "completed": len(completed),
        "failed": len(failures),
        "optional_sources": skipped,
    }
    state["download_failures"] = [
        {"endpoint": spec.name, "task_key": key, "error": error} for spec, key, error in failures[:200]
    ]
    state["status"] = "downloaded" if not failures else "download_incomplete"
    _write_state(workspace, state)
    if failures:
        raise TushareExtendedBackfillError(f"download_incomplete:{len(failures)}:resume_with_run_pending")
    return dict(state["download_progress"])
