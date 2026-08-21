"""Tushare Extended Backfill: probe responsibilities."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from quantlab.data.qdp_v2.auxiliary_update import (
    _resolve_tushare_token,
)

from .config import (
    MAX_REQUESTS_PER_MINUTE,
    MAX_WORKERS,
    SPECS,
    EndpointSpec,
    ResilientTushareClient,
    TushareExtendedBackfillError,
)
from .context import (
    _read_state,
    _validate_schema,
    _workspace,
    _write_state,
)


def _probe_params(spec: EndpointSpec, date: str) -> dict[str, Any]:
    compact = date.replace("-", "")
    if spec.mode == "year":
        return {
            "start_date": compact,
            "end_date": compact,
            "limit": 1,
            "offset": 0,
        }
    params: dict[str, Any] = {"trade_date": compact, "limit": 1, "offset": 0}
    if spec.name == "margin-secs":
        params["exchange"] = "SSE"
    return params


def probe(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    token = _resolve_tushare_token(workspace)
    if not token:
        raise TushareExtendedBackfillError("tushare_token_required")
    client = ResilientTushareClient(token, rpm=MAX_REQUESTS_PER_MINUTE, workspace_root=workspace)
    previous = dict(dict(_read_state(workspace).get("probe", {}) or {}).get("endpoints", {}) or {})
    endpoint_checks: dict[str, dict[str, Any]] = {name: {} for name in SPECS}
    jobs: list[tuple[EndpointSpec, str]] = []
    for spec in SPECS.values():
        dates = ("2011-01-04", "2024-01-02") if not spec.optional else ("2024-01-02",)
        for date in dates:
            jobs.append((spec, date))

    def run(item: tuple[EndpointSpec, str]) -> tuple[str, str, dict[str, Any]]:
        spec, date = item
        try:
            frame = client.fetch(
                spec.api_name,
                params=_probe_params(spec, date),
                fields=spec.fields,
                retries=2,
            )
            schema_digest = _validate_schema(spec, frame.columns)
            result = {
                "status": "available" if len(frame) else "source_unavailable",
                "row_count": len(frame),
                "field_count": len(frame.columns),
                "schema_hash": schema_digest,
            }
        except Exception as exc:  # noqa: BLE001 - probe records provider failures
            result = {
                "status": "source_unavailable",
                "error_type": type(exc).__name__,
            }
        return spec.name, date, result

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(run, item) for item in jobs]
        for future in as_completed(futures):
            name, date, result = future.result()
            old = dict(dict(previous.get(name, {}) or {}).get("checks", {}).get(date, {}) or {})
            if result.get("status") != "available" and old.get("status") == "available":
                result = {**old, "reused_successful_probe": True}
            endpoint_checks[name][date] = result

    endpoints: dict[str, Any] = {}
    for spec in SPECS.values():
        checks = endpoint_checks[spec.name]
        available_count = sum(item.get("status") == "available" for item in checks.values())
        if available_count == len(checks):
            endpoint_status = "available"
        elif available_count:
            endpoint_status = "available_with_partial_probe"
        else:
            endpoint_status = "source_unavailable"
        endpoints[spec.name] = {
            "status": endpoint_status,
            "optional": spec.optional,
            "checks": checks,
        }
    state = _read_state(workspace)
    state["probe"] = {
        "status": "completed",
        "probed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "maximum_workers": MAX_WORKERS,
        "maximum_requests_per_minute": MAX_REQUESTS_PER_MINUTE,
        "endpoints": endpoints,
    }
    state["status"] = "probed"
    _write_state(workspace, state)
    return dict(state["probe"])
