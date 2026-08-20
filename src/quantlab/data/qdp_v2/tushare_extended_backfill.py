from __future__ import annotations

"""Backfill full-history technical, margin, and traditional money-flow data."""

import argparse
import getpass
import hashlib
import json
import os
import shutil
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow.parquet as pq
import requests

from quantlab.data.core.json_io import json_safe
from quantlab.data.core.paths import qdp_paths
from quantlab.data.domains.contracts import DataDomain
from quantlab.data.qdp_v2.auxiliary_update import (
    _MinuteLimiter,
    _resolve_tushare_token,
)
from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    _manifest_schema_from_arrow,
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    utc_now,
    write_active_manifest,
    write_dataset_manifest,
)
from quantlab.data.qdp_v2.provider_credentials import (
    ProviderCredentialError,
    resolve_tushare_api_url,
    resolve_tushare_rate_limit,
    tushare_credential_values,
    tushare_provider_status,
)
from quantlab.data.qdp_v2.status import active_dataset_map

UPDATE_ID = "tushare_extended_backfill_v1"
BURN_IN_START = "2010-01-01"
RESEARCH_START = "2011-01-01"
END_DATE = "2025-12-31"
FORBIDDEN_YEAR = 2026
PAGE_SIZE = 5_000
MAX_PAGES_PER_TASK = 1_000
MAX_WORKERS = 3
MAX_REQUESTS_PER_MINUTE = 120
EXPECTED_FACTOR_FIELD_COUNT = 261
EXPECTED_FACTOR_SCHEMA_HASH = (
    "14cf668f77da06b5ddaffe59057e98c25107a870d7ec006ec5e9ee1ae574d488"
)
PREPARED_TRANSFORM_VERSION = "fixed_provider_types_v2"

MARGIN_FIELDS = (
    "trade_date",
    "exchange_id",
    "rzye",
    "rzmre",
    "rzche",
    "rqye",
    "rqmcl",
    "rzrqye",
    "rqyl",
)
MARGIN_DETAIL_FIELDS = (
    "trade_date",
    "ts_code",
    "rzye",
    "rqye",
    "rzmre",
    "rqyl",
    "rzche",
    "rqchl",
    "rqmcl",
    "rzrqye",
)
MARGIN_SECS_FIELDS = ("trade_date", "ts_code", "name", "exchange")
MONEYFLOW_FIELDS = (
    "ts_code",
    "trade_date",
    "buy_sm_vol",
    "buy_sm_amount",
    "sell_sm_vol",
    "sell_sm_amount",
    "buy_md_vol",
    "buy_md_amount",
    "sell_md_vol",
    "sell_md_amount",
    "buy_lg_vol",
    "buy_lg_amount",
    "sell_lg_vol",
    "sell_lg_amount",
    "buy_elg_vol",
    "buy_elg_amount",
    "sell_elg_vol",
    "sell_elg_amount",
    "net_mf_vol",
    "net_mf_amount",
)


class TushareExtendedBackfillError(RuntimeError):
    pass


def _resolve_tushare_api_url(
    workspace_root: str | Path | None = None,
) -> str:
    try:
        return resolve_tushare_api_url(workspace_root)
    except ProviderCredentialError as exc:
        raise TushareExtendedBackfillError(str(exc)) from exc


class ResilientTushareClient:
    """Use one HTTP session per worker while sharing a global rate limiter."""

    def __init__(
        self,
        token: str,
        *,
        rpm: int = MAX_REQUESTS_PER_MINUTE,
        timeout: int = 60,
        workspace_root: str | Path | None = None,
    ) -> None:
        secret = str(token or "").strip()
        if not secret:
            raise TushareExtendedBackfillError("tushare_token_missing")
        self._token = secret
        self._timeout = int(timeout)
        self._limiter = _MinuteLimiter(
            resolve_tushare_rate_limit(rpm, workspace_root=workspace_root)
        )
        self._local = threading.local()
        self._url = _resolve_tushare_api_url(workspace_root)

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"Accept-Encoding": "gzip"})
            self._local.session = session
        return session

    def fetch(
        self,
        api_name: str,
        *,
        params: Mapping[str, Any],
        fields: Sequence[str],
        retries: int = 5,
    ) -> pd.DataFrame:
        last_error = ""
        for attempt in range(max(1, int(retries))):
            self._limiter.wait()
            try:
                response = self._session().post(
                    self._url,
                    json={
                        "api_name": str(api_name),
                        "token": self._token,
                        "params": dict(params),
                        "fields": ",".join(fields),
                    },
                    timeout=self._timeout,
                )
                response.raise_for_status()
                payload = response.json()
                if int(payload.get("code", -1)) != 0:
                    raise TushareExtendedBackfillError(
                        f"tushare_api_error:{api_name}:code={payload.get('code')}"
                    )
                data = dict(payload.get("data", {}) or {})
                names = [str(item) for item in list(data.get("fields", []) or [])]
                rows = list(data.get("items", []) or [])
                return pd.DataFrame(rows, columns=names or list(fields))
            except Exception as exc:  # noqa: BLE001 - bounded provider retry
                last_error = f"{type(exc).__name__}:{str(exc)[:200]}"
                if attempt + 1 < max(1, int(retries)):
                    time.sleep(min(2**attempt, 8))
        raise TushareExtendedBackfillError(
            f"tushare_request_failed:{api_name}:{last_error}"
        )


@dataclass(frozen=True)
class EndpointSpec:
    name: str
    api_name: str
    domain: str
    mode: str
    fields: tuple[str, ...]
    primary_key: tuple[str, ...]
    frequency: str
    contract_version: str
    optional: bool = False


SPECS = {
    "stk-factor-pro": EndpointSpec(
        name="stk-factor-pro",
        api_name="stk_factor_pro",
        domain=DataDomain.STK_FACTOR_PRO_RAW,
        mode="trade_date",
        fields=(),
        primary_key=("trade_date", "security_id"),
        frequency="1d",
        contract_version="qdp_v2_tushare_stk_factor_pro_raw_v1",
    ),
    "margin": EndpointSpec(
        name="margin",
        api_name="margin",
        domain=DataDomain.MARGIN_MARKET,
        mode="year",
        fields=MARGIN_FIELDS,
        primary_key=("trade_date", "exchange_id"),
        frequency="1d_market",
        contract_version="qdp_v2_tushare_margin_market_raw_v1",
    ),
    "margin-detail": EndpointSpec(
        name="margin-detail",
        api_name="margin_detail",
        domain=DataDomain.MARGIN_DETAIL,
        mode="trade_date",
        fields=MARGIN_DETAIL_FIELDS,
        primary_key=("trade_date", "security_id"),
        frequency="1d",
        contract_version="qdp_v2_tushare_margin_detail_raw_v1",
    ),
    "margin-secs": EndpointSpec(
        name="margin-secs",
        api_name="margin_secs",
        domain=DataDomain.MARGIN_SECS,
        mode="trade_date",
        fields=MARGIN_SECS_FIELDS,
        primary_key=("trade_date", "security_id"),
        frequency="1d_membership",
        contract_version="qdp_v2_tushare_margin_secs_raw_v1",
        optional=True,
    ),
    "moneyflow": EndpointSpec(
        name="moneyflow",
        api_name="moneyflow",
        domain=DataDomain.MONEYFLOW_RAW,
        mode="trade_date",
        fields=MONEYFLOW_FIELDS,
        primary_key=("trade_date", "security_id"),
        frequency="1d",
        contract_version="qdp_v2_tushare_moneyflow_raw_v1",
    ),
}


def _workspace(value: str | Path | None) -> Path:
    return Path(value or Path.cwd()).resolve()


def _runtime(workspace: Path) -> Path:
    path = (qdp_paths(workspace).data_dir / "qdp_runtime" / UPDATE_ID).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _state_path(workspace: Path) -> Path:
    return _runtime(workspace) / "state.json"


def _credential_values() -> tuple[str, ...]:
    return tushare_credential_values()


def _assert_credential_free(payload: Any) -> None:
    encoded = json.dumps(json_safe(payload), ensure_ascii=False, default=str)
    for secret in _credential_values():
        if secret and secret in encoded:
            raise TushareExtendedBackfillError("credential_persistence_blocked")


def _read_state(workspace: Path) -> dict[str, Any]:
    path = _state_path(workspace)
    if not path.is_file():
        return {
            "update_id": UPDATE_ID,
            "status": "pending",
            "burn_in_start": BURN_IN_START,
            "research_start": RESEARCH_START,
            "end_date": END_DATE,
            "training_performed": False,
        }
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_state(workspace: Path, state: Mapping[str, Any]) -> None:
    payload = {**dict(state), "updated_at": utc_now()}
    _assert_credential_free(payload)
    atomic_write_json(_state_path(workspace), payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        json_safe(payload), sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _provider_cache_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.select_dtypes(include=["object"]).columns:
        output[column] = output[column].map(
            lambda value: (
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (list, tuple, dict))
                else ""
                if value is None
                else str(value)
            )
        )
    return output


def _active_paths(workspace: Path, domain: str) -> list[Path]:
    root = qdp_v2_root(workspace)
    datasets = active_dataset_map(read_active_manifest(root))
    dataset_id = datasets.get(domain, "")
    manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
    if manifest_path is None:
        raise TushareExtendedBackfillError(f"active_domain_missing:{domain}")
    manifest = read_dataset_manifest(manifest_path)
    paths = [resolve_manifest_path(item.path, root=root) for item in manifest.shards]
    if not paths or not all(path.is_file() for path in paths):
        raise TushareExtendedBackfillError(f"active_domain_shards_missing:{domain}")
    return paths


def _scan(paths: Sequence[Path]) -> str:
    literals = ",".join("'" + str(path).replace("'", "''") + "'" for path in paths)
    return f"read_parquet([{literals}], union_by_name=true)"


def _open_dates(workspace: Path) -> list[str]:
    paths = _active_paths(workspace, DataDomain.TRADING_CALENDAR)
    with duckdb.connect() as connection:
        rows = connection.execute(
            f"""
            SELECT DISTINCT cast(trade_date AS VARCHAR)
            FROM {_scan(paths)}
            WHERE is_open=true
              AND cast(trade_date AS VARCHAR) BETWEEN '{BURN_IN_START}' AND '{END_DATE}'
            ORDER BY 1
            """
        ).fetchall()
    dates = [str(row[0])[:10] for row in rows]
    if not dates or dates[0] < BURN_IN_START or dates[-1] > END_DATE:
        raise TushareExtendedBackfillError("open_date_inventory_invalid")
    if any(date.startswith(str(FORBIDDEN_YEAR)) for date in dates):
        raise TushareExtendedBackfillError("forbidden_2026_date_planned")
    return dates


def _schema_hash(columns: Sequence[str]) -> str:
    return hashlib.sha256(",".join(str(item) for item in columns).encode()).hexdigest()


def _validate_schema(spec: EndpointSpec, columns: Sequence[str]) -> str:
    actual = tuple(str(item) for item in columns)
    digest = _schema_hash(actual)
    if spec.name == "stk-factor-pro":
        if (
            len(actual) != EXPECTED_FACTOR_FIELD_COUNT
            or digest != EXPECTED_FACTOR_SCHEMA_HASH
        ):
            raise TushareExtendedBackfillError(
                f"factor_schema_changed:count={len(actual)}:hash={digest}"
            )
    elif actual != spec.fields:
        raise TushareExtendedBackfillError(
            f"endpoint_schema_changed:{spec.name}:actual={actual}"
        )
    return digest


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
    client = ResilientTushareClient(
        token, rpm=MAX_REQUESTS_PER_MINUTE, workspace_root=workspace
    )
    previous = dict(
        dict(_read_state(workspace).get("probe", {}) or {}).get("endpoints", {}) or {}
    )
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
            old = dict(
                dict(previous.get(name, {}) or {}).get("checks", {}).get(date, {}) or {}
            )
            if result.get("status") != "available" and old.get("status") == "available":
                result = {**old, "reused_successful_probe": True}
            endpoint_checks[name][date] = result

    endpoints: dict[str, Any] = {}
    for spec in SPECS.values():
        checks = endpoint_checks[spec.name]
        available_count = sum(
            item.get("status") == "available" for item in checks.values()
        )
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
    return (
        _runtime(workspace) / "raw" / spec.name / f"year={year}" / f"task={stable_key}"
    )


def _page_paths(
    workspace: Path, spec: EndpointSpec, task_key: str, offset: int
) -> tuple[Path, Path]:
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
        if int(record.get("row_count", -1)) != int(
            pq.ParquetFile(path).metadata.num_rows
        ):
            return None
        return record
    except Exception:  # noqa: BLE001 - corrupt cache is treated as a miss
        return None


def _valid_success(
    workspace: Path, spec: EndpointSpec, task_key: str
) -> dict[str, Any] | None:
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
            raise TushareExtendedBackfillError(
                f"pagination_guard_exceeded:{spec.name}:{task_key}"
            )
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
                dates = (
                    frame["trade_date"].astype(str).str.replace("-", "", regex=False)
                )
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
                "request_hash": _stable_hash(
                    {"api_name": spec.api_name, "params": params, "fields": spec.fields}
                ),
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


def _download_inventory(
    *,
    workspace: Path,
    specs: Sequence[EndpointSpec],
    workers: int,
) -> dict[str, Any]:
    state = _read_state(workspace)
    probe_state = dict(state.get("probe", {}) or {})
    if probe_state.get("status") != "completed":
        probe_state = probe(workspace_root=workspace)
        state = _read_state(workspace)
    endpoint_probe = dict(probe_state.get("endpoints", {}) or {})
    unavailable_mandatory = [
        spec.name
        for spec in specs
        if not spec.optional
        and dict(endpoint_probe.get(spec.name, {}) or {}).get("status")
        not in {"available", "available_with_partial_probe"}
    ]
    if unavailable_mandatory:
        raise TushareExtendedBackfillError(
            f"mandatory_source_unavailable:{','.join(unavailable_mandatory)}"
        )
    token = _resolve_tushare_token(workspace)
    if not token:
        raise TushareExtendedBackfillError("tushare_token_required")
    client = ResilientTushareClient(
        token, rpm=MAX_REQUESTS_PER_MINUTE, workspace_root=workspace
    )
    open_dates = _open_dates(workspace)
    tasks: list[tuple[EndpointSpec, str]] = []
    skipped: dict[str, Any] = {}
    for spec in specs:
        availability = dict(endpoint_probe.get(spec.name, {}) or {}).get("status")
        if spec.optional and availability not in {
            "available",
            "available_with_partial_probe",
        }:
            skipped[spec.name] = {
                "status": "source_unavailable",
                "reason": "probe_failed_or_empty",
            }
            continue
        for task_key in _task_keys(spec, open_dates):
            if _valid_success(workspace, spec, task_key) is None:
                tasks.append((spec, task_key))
    completed: list[dict[str, Any]] = []
    failures: list[tuple[EndpointSpec, str, str]] = []
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
                failures.append(
                    (spec, task_key, f"{type(exc).__name__}:{str(exc)[:300]}")
                )
            if index % 20 == 0 or index == len(tasks):
                state = _read_state(workspace)
                state["download_progress"] = {
                    "planned": len(tasks),
                    "completed": len(completed),
                    "failed": len(failures),
                    "last_checkpoint_at": utc_now(),
                }
                _write_state(workspace, state)
                print(
                    json.dumps(
                        state["download_progress"], ensure_ascii=False, sort_keys=True
                    ),
                    flush=True,
                )
    if failures and worker_count > 1:
        retry_failures: list[tuple[EndpointSpec, str, str]] = []
        for spec, task_key, _ in failures:
            try:
                completed.append(_download_task(workspace, spec, task_key, client))
            except Exception as exc:  # noqa: BLE001 - isolate task failures for resume
                retry_failures.append(
                    (spec, task_key, f"{type(exc).__name__}:{str(exc)[:300]}")
                )
        failures = retry_failures
    state = _read_state(workspace)
    state["download_progress"] = {
        "planned": len(tasks),
        "completed": len(completed),
        "failed": len(failures),
        "optional_sources": skipped,
    }
    state["download_failures"] = [
        {"endpoint": spec.name, "task_key": key, "error": error}
        for spec, key, error in failures[:200]
    ]
    state["status"] = "downloaded" if not failures else "download_incomplete"
    _write_state(workspace, state)
    if failures:
        raise TushareExtendedBackfillError(
            f"download_incomplete:{len(failures)}:resume_with_run_pending"
        )
    return dict(state["download_progress"])


def _year_page_paths(workspace: Path, spec: EndpointSpec, year: int) -> list[Path]:
    root = _runtime(workspace) / "raw" / spec.name / f"year={year}"
    if not root.is_dir():
        return []
    if spec.mode == "year":
        paths = sorted(root.glob(f"task={year}/offset=*.parquet"))
    else:
        paths = sorted(
            path
            for task_root in root.glob("task=*")
            if (task_key := task_root.name.removeprefix("task=")).isdigit()
            and len(task_key) == 8
            for path in task_root.glob("offset=*.parquet")
        )
    return [path for path in paths if path.is_file()]


def _sql_date(column: str) -> str:
    value = f"cast({column} AS VARCHAR)"
    return (
        f"CASE WHEN length({value})=8 THEN substr({value},1,4)||'-'||"
        f"substr({value},5,2)||'-'||substr({value},7,2) ELSE substr({value},1,10) END"
    )


def _copy_query(connection: duckdb.DuckDBPyConnection, sql: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    if temporary.exists():
        temporary.unlink()
    quoted = str(temporary).replace("'", "''")
    connection.execute(
        f"COPY ({sql}) TO '{quoted}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072)"
    )
    os.replace(temporary, path)


def _provider_fields(paths: Sequence[Path]) -> tuple[str, ...]:
    if not paths:
        return ()
    return tuple(pq.read_schema(paths[0]).names)


def _quoted(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _provider_value_expression(spec: EndpointSpec, *, alias: str, field: str) -> str:
    reference = f"{alias}.{_quoted(field)}"
    text_fields = {"exchange_id"}
    if spec.name == "margin-secs":
        text_fields.update({"name", "exchange"})
    target_type = "VARCHAR" if field in text_fields else "DOUBLE"
    return f"try_cast({reference} AS {target_type}) AS {_quoted(field)}"


def _prepared_sql(
    *,
    spec: EndpointSpec,
    pages: Sequence[Path],
    history_paths: Sequence[Path],
    calendar_paths: Sequence[Path],
) -> str:
    fields = _provider_fields(pages)
    _validate_schema(spec, fields)
    date_expr = _sql_date("r.trade_date")
    provider_values = [
        field for field in fields if field not in {"ts_code", "trade_date"}
    ]
    provider_sql = ",\n             ".join(
        _provider_value_expression(spec, alias="r", field=field)
        for field in provider_values
    )
    provider_sql = (",\n             " + provider_sql) if provider_sql else ""
    page_scan = _scan(pages)
    calendar_scan = _scan(calendar_paths)
    availability = (
        "n.trade_date"
        if spec.name == "stk-factor-pro"
        else "CASE WHEN c.next_trade_date<='2025-12-31' "
        "THEN c.next_trade_date ELSE '' END"
    )
    common = f"""
    WITH open_calendar AS (
      SELECT trade_date,
             lead(trade_date) OVER (ORDER BY trade_date) AS next_trade_date
      FROM (
        SELECT DISTINCT cast(trade_date AS VARCHAR) AS trade_date
        FROM {calendar_scan}
        WHERE is_open=true AND cast(trade_date AS VARCHAR)<='2026-01-31'
      )
    ), raw AS (
      SELECT * FROM {page_scan}
    ), normalized AS (
      SELECT cast(r.ts_code AS VARCHAR) AS ts_code,
             upper(cast(r.ts_code AS VARCHAR)) AS symbol,
             {date_expr} AS trade_date{provider_sql}
      FROM raw r
    ), history AS (
      SELECT symbol,min(security_id) AS security_id
      FROM {_scan(history_paths)}
      GROUP BY symbol
      HAVING count(DISTINCT security_id)=1
    ), mapped AS (
      SELECT h.security_id,n.*,n.trade_date AS source_date,
             {availability} AS feature_available_date,
             n.trade_date<'{RESEARCH_START}' AS burn_in_only,
             'tushare_compatible.{spec.api_name}' AS source
      FROM normalized n
      JOIN history h USING(symbol)
      LEFT JOIN open_calendar c USING(trade_date)
      WHERE n.trade_date BETWEEN '{BURN_IN_START}' AND '{END_DATE}'
    )
    """
    if spec.name == "margin":
        values = [field for field in fields if field != "trade_date"]
        selected = ",\n                 ".join(
            _provider_value_expression(spec, alias="r", field=field) for field in values
        )
        return f"""
        WITH open_calendar AS (
          SELECT trade_date,
                 lead(trade_date) OVER (ORDER BY trade_date) AS next_trade_date
          FROM (
            SELECT DISTINCT cast(trade_date AS VARCHAR) AS trade_date
            FROM {calendar_scan}
            WHERE is_open=true AND cast(trade_date AS VARCHAR)<='2026-01-31'
          )
        ), raw AS (
          SELECT * FROM {page_scan}
        ), normalized AS (
          SELECT {_sql_date("r.trade_date")} AS trade_date,
                 {selected}
          FROM raw r
        )
        SELECT trade_date,{",".join(_quoted(field) for field in values)},
               trade_date AS source_date,
               CASE WHEN c.next_trade_date<='{END_DATE}'
                    THEN c.next_trade_date ELSE '' END AS feature_available_date,
               trade_date<'{RESEARCH_START}' AS burn_in_only,
               'tushare_compatible.margin' AS source
        FROM normalized n
        LEFT JOIN open_calendar c USING(trade_date)
        WHERE trade_date BETWEEN '{BURN_IN_START}' AND '{END_DATE}'
        QUALIFY row_number() OVER (
          PARTITION BY trade_date,exchange_id ORDER BY trade_date
        )=1
        ORDER BY trade_date,exchange_id
        """
    output_fields = ["security_id", "symbol", "ts_code", "trade_date", *provider_values]
    return (
        common
        + f"""
        SELECT {",".join(_quoted(field) for field in output_fields)},
               source_date,feature_available_date,burn_in_only,source
        FROM mapped
        QUALIFY row_number() OVER (
          PARTITION BY trade_date,security_id ORDER BY trade_date
        )=1
        ORDER BY trade_date,security_id
        """
    )


def _validate_prepared(path: Path, spec: EndpointSpec, year: int) -> dict[str, Any]:
    numeric_fields: tuple[str, ...] = ()
    allowed_negative_fields: tuple[str, ...] = ()
    if spec.name == "margin":
        numeric_fields = tuple(
            field
            for field in MARGIN_FIELDS
            if field not in {"trade_date", "exchange_id", "rzche"}
        )
    elif spec.name == "margin-detail":
        numeric_fields = tuple(
            field
            for field in MARGIN_DETAIL_FIELDS
            if field not in {"trade_date", "ts_code", "rzche", "rqchl"}
        )
    elif spec.name == "moneyflow":
        numeric_fields = tuple(
            field
            for field in MONEYFLOW_FIELDS
            if field not in {"trade_date", "ts_code", "net_mf_vol", "net_mf_amount"}
        )
        allowed_negative_fields = ("net_mf_vol", "net_mf_amount")
    negative_sql = (
        "+".join(
            f"count(*) FILTER(WHERE try_cast({_quoted(field)} AS DOUBLE)<0)"
            for field in numeric_fields
        )
        or "0"
    )
    allowed_negative_sql = (
        "+".join(
            f"count(*) FILTER(WHERE try_cast({_quoted(field)} AS DOUBLE)<0)"
            for field in allowed_negative_fields
        )
        or "0"
    )
    relation_sql = "0"
    source_exception_sql = "0"
    if spec.name == "margin":
        negative_sql = (
            f"({negative_sql})+count(*) FILTER(WHERE "
            "try_cast(rzche AS DOUBLE)<0 AND exchange_id<>'BSE')"
        )
        source_exception_sql = (
            "count(*) FILTER(WHERE try_cast(rzche AS DOUBLE)<0 AND exchange_id='BSE')"
        )
    elif spec.name == "margin-detail":
        source_exception_sql = (
            "count(*) FILTER(WHERE try_cast(rzche AS DOUBLE)<0)+"
            "count(*) FILTER(WHERE try_cast(rqchl AS DOUBLE)<0)"
        )
    if spec.name == "moneyflow":
        relation_sql = """
        count(*) FILTER(
          WHERE net_mf_amount IS NOT NULL
            AND buy_lg_amount IS NOT NULL AND buy_elg_amount IS NOT NULL
            AND sell_lg_amount IS NOT NULL AND sell_elg_amount IS NOT NULL
            AND abs(try_cast(net_mf_amount AS DOUBLE)-(
              try_cast(buy_lg_amount AS DOUBLE)+try_cast(buy_elg_amount AS DOUBLE)-
              try_cast(sell_lg_amount AS DOUBLE)-try_cast(sell_elg_amount AS DOUBLE)
            ))>greatest(1.0,abs(try_cast(net_mf_amount AS DOUBLE))*0.000001)
        )
        """
    with duckdb.connect() as connection:
        quoted = str(path).replace("'", "''")
        keys = ",".join(_quoted(item) for item in spec.primary_key)
        row = connection.execute(
            f"""
            SELECT count(*),count(*)-count(DISTINCT ({keys})),
                   min(trade_date),max(trade_date),
                   count(*) FILTER(WHERE trade_date>='{FORBIDDEN_YEAR}-01-01'),
                   count(*) FILTER(WHERE feature_available_date<>'' AND
                     feature_available_date<source_date),
                   count(*) FILTER(WHERE burn_in_only<>(trade_date<'{RESEARCH_START}')),
                   {negative_sql},
                   {allowed_negative_sql},
                   {relation_sql},
                   {source_exception_sql}
            FROM read_parquet('{quoted}')
            """
        ).fetchone()
    if (
        int(row[1] or 0)
        or int(row[4] or 0)
        or int(row[5] or 0)
        or int(row[6] or 0)
        or int(row[7] or 0)
    ):
        raise TushareExtendedBackfillError(
            f"prepared_contract_failed:{spec.domain}:{year}:{tuple(row)}"
        )
    return {
        "year": int(year),
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "row_count": int(row[0]),
        "start_date": str(row[2] or ""),
        "end_date": str(row[3] or ""),
        "primary_key_unique": True,
        "forbidden_2026_rows": int(row[4] or 0),
        "availability_before_source_rows": int(row[5] or 0),
        "burn_in_flag_mismatch_rows": int(row[6] or 0),
        "disallowed_negative_value_count": int(row[7] or 0),
        "allowed_negative_net_value_count": int(row[8] or 0),
        "moneyflow_main_net_relation_mismatch_count": int(row[9] or 0),
        "source_exception_negative_value_count": int(row[10] or 0),
    }


def _prepare_domain(
    workspace: Path, spec: EndpointSpec, open_dates: Sequence[str]
) -> list[dict[str, Any]]:
    history_paths = _active_paths(workspace, DataDomain.SYMBOL_HISTORY)
    calendar_paths = _active_paths(workspace, DataDomain.TRADING_CALENDAR)
    records: list[dict[str, Any]] = []
    for year in range(2010, 2026):
        expected_keys = _task_keys(
            spec, [date for date in open_dates if date.startswith(str(year))]
        )
        if spec.mode == "year":
            expected_keys = [str(year)]
        incomplete = [
            key for key in expected_keys if _valid_success(workspace, spec, key) is None
        ]
        if incomplete:
            raise TushareExtendedBackfillError(
                f"raw_tasks_incomplete:{spec.name}:{year}:{len(incomplete)}"
            )
        pages = _year_page_paths(workspace, spec, year)
        if not pages:
            raise TushareExtendedBackfillError(f"raw_pages_missing:{spec.name}:{year}")
        path = (
            _runtime(workspace)
            / "prepared"
            / spec.domain
            / f"year={year}"
            / "part-0000.parquet"
        )
        input_hash = _stable_hash(
            {
                "prepared_transform_version": PREPARED_TRANSFORM_VERSION,
                "endpoint": spec.name,
                "contract_version": spec.contract_version,
                "page_sha256": [_sha256(page) for page in pages],
            }
        )
        sidecar = path.with_suffix(".json")
        reusable = False
        if path.is_file() and sidecar.is_file():
            previous = json.loads(sidecar.read_text(encoding="utf-8"))
            reusable = previous.get("input_hash") == input_hash and previous.get(
                "sha256"
            ) == _sha256(path)
        if not reusable:
            with duckdb.connect() as connection:
                connection.execute("SET threads=4")
                connection.execute("SET memory_limit='12GB'")
                _copy_query(
                    connection,
                    _prepared_sql(
                        spec=spec,
                        pages=pages,
                        history_paths=history_paths,
                        calendar_paths=calendar_paths,
                    ),
                    path,
                )
            validation = _validate_prepared(path, spec, year)
            atomic_write_json(sidecar, {**validation, "input_hash": input_hash})
        records.append(
            {**_validate_prepared(path, spec, year), "input_hash": input_hash}
        )
        print(json.dumps({"prepared": spec.domain, "year": year}), flush=True)
    return records


def _assert_uniform_prepared_schema(
    spec: EndpointSpec, records: Sequence[Mapping[str, Any]]
) -> None:
    schemas: dict[str, list[int]] = {}
    for record in records:
        path = Path(str(record["path"]))
        digest = _stable_hash(_manifest_schema_from_arrow(pq.read_schema(path)))
        schemas.setdefault(digest, []).append(int(record["year"]))
    if len(schemas) != 1:
        detail = ";".join(
            f"{digest[:12]}={','.join(str(year) for year in years)}"
            for digest, years in sorted(schemas.items())
        )
        raise TushareExtendedBackfillError(
            f"prepared_schema_drift:{spec.domain}:{detail}"
        )


def _factor_validation(
    workspace: Path, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    factor_paths = [Path(str(item["path"])) for item in records]
    daily_paths = _active_paths(workspace, "market_daily_raw")
    adjust_paths = _active_paths(workspace, DataDomain.ADJUST_FACTOR)
    with duckdb.connect() as connection:
        connection.execute("SET threads=4")
        connection.execute("SET memory_limit='12GB'")
        row = connection.execute(
            f"""
            WITH f AS (
              SELECT symbol,trade_date,
                     try_cast(open AS DOUBLE) AS open,
                     try_cast(high AS DOUBLE) AS high,
                     try_cast(low AS DOUBLE) AS low,
                     try_cast(close AS DOUBLE) AS close,
                     try_cast(close_hfq AS DOUBLE) AS close_hfq,
                     try_cast(adj_factor AS DOUBLE) AS adj_factor
              FROM {_scan(factor_paths)}
            ), first_factor AS (
              SELECT symbol,arg_min(adj_factor,trade_date) AS first_adj_factor
              FROM f WHERE adj_factor>0 GROUP BY symbol
            ), pairs AS (
              SELECT f.*,d.open AS d_open,d.high AS d_high,d.low AS d_low,
                     d.close AS d_close,a.back_adjust_factor,ff.first_adj_factor
              FROM f
              JOIN {_scan(daily_paths)} d USING(symbol,trade_date)
              JOIN {_scan(adjust_paths)} a USING(symbol,trade_date)
              JOIN first_factor ff USING(symbol)
              WHERE f.trade_date BETWEEN '{BURN_IN_START}' AND '{END_DATE}'
                AND abs(hash(f.symbol||f.trade_date)%97)=0
            )
            SELECT count(*),
              count(*) FILTER(WHERE
                abs(open/d_open-1)>0.005 OR abs(high/d_high-1)>0.005 OR
                abs(low/d_low-1)>0.005 OR abs(close/d_close-1)>0.005),
              count(*) FILTER(WHERE close>0 AND adj_factor>0 AND
                abs(close_hfq/(close*adj_factor)-1)>0.0001),
              count(*) FILTER(WHERE first_adj_factor>0 AND back_adjust_factor>0 AND
                abs((adj_factor/first_adj_factor)/back_adjust_factor-1)>0.005)
            FROM pairs
            """
        ).fetchone()
    compared = int(row[0] or 0)
    raw_mismatch = int(row[1] or 0)
    hfq_identity_mismatch = int(row[2] or 0)
    factor_mismatch = int(row[3] or 0)
    raw_rate = raw_mismatch / compared if compared else 1.0
    hfq_rate = (hfq_identity_mismatch + factor_mismatch) / compared if compared else 1.0
    return {
        "sample_compared_count": compared,
        "raw_price_mismatch_count": raw_mismatch,
        "raw_price_mismatch_rate": raw_rate,
        "hfq_identity_mismatch_count": hfq_identity_mismatch,
        "normalized_adjust_factor_mismatch_count": factor_mismatch,
        "hfq_validation_mismatch_rate": hfq_rate,
        "raw_price_validation_passed": compared > 0 and raw_rate <= 0.005,
        "hfq_validation_passed": compared > 0 and hfq_rate <= 0.005,
    }


def _install_domain(
    workspace: Path,
    *,
    spec: EndpointSpec,
    records: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    root = qdp_v2_root(workspace)
    fingerprint = _stable_hash(
        {"domain": spec.domain, "shards": [item["sha256"] for item in records]}
    )[:24]
    dataset_id = f"{spec.domain}__{fingerprint}"
    dataset_dir = root / "datasets" / spec.domain / dataset_id
    shards: list[ShardManifestEntry] = []
    for record in records:
        year = int(record["year"])
        source = Path(str(record["path"]))
        target = dataset_dir / "shards" / f"year={year}" / "part-0000.parquet"
        if not target.is_file() or _sha256(target) != str(record["sha256"]):
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp.parquet")
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
        shards.append(
            ShardManifestEntry(
                path=str(target.relative_to(root)).replace("\\", "/"),
                row_count=int(record["row_count"]),
                start_date=str(record["start_date"]),
                end_date=str(record["end_date"]),
                file_size=target.stat().st_size,
                metadata={
                    "year": year,
                    "sha256": _sha256(target),
                    "burn_in_only": year == 2010,
                },
            )
        )
    nonempty = [item for item in records if int(item["row_count"]) > 0]
    if not nonempty:
        raise TushareExtendedBackfillError(f"prepared_domain_empty:{spec.domain}")
    first_shard = resolve_manifest_path(shards[0].path, root=root)
    source_inventory_hash = _stable_hash([item["input_hash"] for item in records])
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        domain=spec.domain,
        layer="raw",
        frequency=spec.frequency,
        contract_version=spec.contract_version,
        primary_key=list(spec.primary_key),
        start_date=min(str(item["start_date"]) for item in nonempty),
        end_date=max(str(item["end_date"]) for item in nonempty),
        row_count=sum(int(item["row_count"]) for item in records),
        shards=shards,
        source={
            "provider": f"tushare_compatible.{spec.api_name}",
            "query_granularity": spec.mode,
            "checked_through": END_DATE,
            "scope": "point_in_time_historical_mainboard",
            "source_inventory_hash": source_inventory_hash,
            "credential_persisted": False,
            "request_concurrency_maximum": MAX_WORKERS,
            "request_rate_per_minute_maximum": MAX_REQUESTS_PER_MINUTE,
        },
        quality={
            "primary_key_unique": True,
            "strict_point_in_time": True,
            "burn_in_year": 2010,
            "burn_in_eligible_for_training": False,
            "forbidden_2026_rows": 0,
            **dict(quality),
        },
        schema=_manifest_schema_from_arrow(pq.read_schema(first_shard)),
        notes=[
            "2010 is retained only for causal feature burn-in",
            "research rows end on 2025-12-31 and never read 2026",
            "provider missingness is preserved rather than filled with zero",
        ],
    )
    _assert_credential_free(manifest.to_dict())
    write_dataset_manifest(root, manifest)
    return {
        "dataset_id": dataset_id,
        "domain": spec.domain,
        "row_count": int(manifest.row_count),
        "start_date": manifest.start_date,
        "end_date": manifest.end_date,
        "shard_count": len(shards),
        "source_inventory_hash": source_inventory_hash,
        "quality": dict(manifest.quality),
    }


def prepare_and_install(
    *,
    workspace_root: str | Path | None = None,
    domains: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    specs = _selected_specs(domains)
    state = _read_state(workspace)
    probe_endpoints = dict(
        dict(state.get("probe", {}) or {}).get("endpoints", {}) or {}
    )
    open_dates = _open_dates(workspace)
    installed: dict[str, Any] = dict(state.get("installed_domains", {}) or {})
    dataset_ids: dict[str, str] = {}
    for spec in specs:
        if spec.optional and dict(probe_endpoints.get(spec.name, {}) or {}).get(
            "status"
        ) not in {"available", "available_with_partial_probe"}:
            installed[spec.domain] = {
                "status": "source_unavailable",
                "optional": True,
            }
            continue
        records = _prepare_domain(workspace, spec, open_dates)
        _assert_uniform_prepared_schema(spec, records)
        quality: dict[str, Any] = {
            "provider_field_count": len(
                _provider_fields(_year_page_paths(workspace, spec, 2011))
            ),
            "provider_schema_hash": _schema_hash(
                _provider_fields(_year_page_paths(workspace, spec, 2011))
            ),
            "yearly_row_count": {
                str(item["year"]): int(item["row_count"]) for item in records
            },
            "disallowed_negative_value_count": sum(
                int(item.get("disallowed_negative_value_count", 0)) for item in records
            ),
            "allowed_negative_net_value_count": sum(
                int(item.get("allowed_negative_net_value_count", 0)) for item in records
            ),
            "provider_net_amount_unreconciled_count": sum(
                int(item.get("moneyflow_main_net_relation_mismatch_count", 0))
                for item in records
            ),
            "source_exception_negative_value_count": sum(
                int(item.get("source_exception_negative_value_count", 0))
                for item in records
            ),
        }
        if spec.name == "stk-factor-pro":
            quality.update(_factor_validation(workspace, records))
            quality["qfq_formal_eligibility"] = False
        record = _install_domain(
            workspace,
            spec=spec,
            records=records,
            quality=quality,
        )
        installed[spec.domain] = {"status": "installed", **record}
        dataset_ids[spec.domain] = str(record["dataset_id"])
    if dataset_ids:
        root = qdp_v2_root(workspace)
        active = read_active_manifest(root)
        active["datasets"] = {
            **dict(active.get("datasets", {}) or {}),
            **dataset_ids,
        }
        active["updated_at"] = utc_now()
        write_active_manifest(root, active)
    state = _read_state(workspace)
    state["installed_domains"] = installed
    state["status"] = "applied"
    state["training_performed"] = False
    _write_state(workspace, state)
    audit_path = qdp_v2_root(workspace) / "audits" / f"{UPDATE_ID}.json"
    _assert_credential_free(state)
    atomic_write_json(audit_path, state)
    return {"status": "applied", "domains": installed, "audit_path": str(audit_path)}


def run_pending(
    *,
    workspace_root: str | Path | None = None,
    domains: str | Sequence[str] | None = None,
    workers: int = MAX_WORKERS,
    seal_runtime: bool = True,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    specs = _selected_specs(domains)
    downloads = _download_inventory(
        workspace=workspace,
        specs=specs,
        workers=workers,
    )
    installed = prepare_and_install(workspace_root=workspace, domains=domains)
    result = {"status": "completed", "downloads": downloads, "install": installed}
    if seal_runtime:
        from quantlab.data.qdp_v2.runtime_archive import (
            seal_completed_workflow,
        )

        result["runtime_archive"] = seal_completed_workflow(
            UPDATE_ID,
            workspace_root=workspace,
        )
    return result


def _credential_file_hits(workspace: Path) -> list[str]:
    secrets = tuple(
        value.encode() for value in tushare_credential_values(workspace) if value
    )
    if not secrets:
        return []
    candidates = [Path(__file__).resolve()]
    candidates.extend(
        path
        for path in _runtime(workspace).rglob("*")
        if path.is_file() and path.suffix.lower() != ".parquet"
    )
    hits: list[str] = []
    for path in candidates:
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        if any(secret in payload for secret in secrets):
            hits.append(str(path))
    return sorted(set(hits))


def _physical_domain_audit(
    workspace: Path, spec: EndpointSpec, dataset_id: str
) -> dict[str, Any]:
    root = qdp_v2_root(workspace)
    manifest_path = dataset_manifest_for_id(root, dataset_id, spec.domain)
    if manifest_path is None:
        raise TushareExtendedBackfillError(
            f"installed_manifest_missing:{spec.domain}:{dataset_id}"
        )
    manifest = read_dataset_manifest(manifest_path)
    records: list[dict[str, Any]] = []
    for shard in manifest.shards:
        path = resolve_manifest_path(shard.path, root=root)
        year = int(dict(shard.metadata or {}).get("year") or str(shard.start_date)[:4])
        records.append(_validate_prepared(path, spec, year))
    row_count = sum(int(item["row_count"]) for item in records)
    if row_count != int(manifest.row_count):
        raise TushareExtendedBackfillError(
            f"installed_row_count_mismatch:{spec.domain}:{row_count}:{manifest.row_count}"
        )
    return {
        "dataset_id": dataset_id,
        "row_count": row_count,
        "shard_count": len(records),
        "start_date": min(
            (str(item["start_date"]) for item in records if item["start_date"]),
            default="",
        ),
        "end_date": max(
            (str(item["end_date"]) for item in records if item["end_date"]),
            default="",
        ),
        "primary_key_unique": all(item["primary_key_unique"] for item in records),
        "forbidden_2026_rows": sum(
            int(item["forbidden_2026_rows"]) for item in records
        ),
        "availability_before_source_rows": sum(
            int(item["availability_before_source_rows"]) for item in records
        ),
        "burn_in_flag_mismatch_rows": sum(
            int(item["burn_in_flag_mismatch_rows"]) for item in records
        ),
        "disallowed_negative_value_count": sum(
            int(item["disallowed_negative_value_count"]) for item in records
        ),
        "allowed_negative_net_value_count": sum(
            int(item["allowed_negative_net_value_count"]) for item in records
        ),
        "provider_net_amount_unreconciled_count": sum(
            int(item["moneyflow_main_net_relation_mismatch_count"]) for item in records
        ),
        "source_exception_negative_value_count": sum(
            int(item["source_exception_negative_value_count"]) for item in records
        ),
        "yearly_row_count": {
            str(item["year"]): int(item["row_count"]) for item in records
        },
    }


def _legacy_year_cache_summary(
    workspace: Path, specs: Sequence[EndpointSpec]
) -> dict[str, Any]:
    endpoints: dict[str, Any] = {}
    for spec in specs:
        if spec.mode != "trade_date":
            continue
        paths: list[Path] = []
        for year in range(2010, 2026):
            root = (
                _runtime(workspace)
                / "raw"
                / spec.name
                / f"year={year}"
                / f"task={year}"
            )
            paths.extend(sorted(root.glob("offset=*.parquet")))
        endpoints[spec.name] = {
            "file_count": len(paths),
            "row_count": sum(
                int(pq.ParquetFile(path).metadata.num_rows) for path in paths
            ),
            "excluded_from_prepared_inventory": all(
                path not in _year_page_paths(workspace, spec, int(path.parts[-3][5:]))
                for path in paths
            ),
        }
    return endpoints


def audit(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    installed = dict(state.get("installed_domains", {}) or {})
    root = qdp_v2_root(workspace)
    active = active_dataset_map(read_active_manifest(root))
    mandatory_domains = [spec.domain for spec in SPECS.values() if not spec.optional]
    physical_domains: dict[str, Any] = {}
    for spec in SPECS.values():
        record = dict(installed.get(spec.domain, {}) or {})
        if record.get("status") != "installed":
            continue
        physical_domains[spec.domain] = _physical_domain_audit(
            workspace, spec, str(record["dataset_id"])
        )
    credential_hits = _credential_file_hits(workspace)
    provider_profile = tushare_provider_status(workspace)
    query_granularity: dict[str, str] = {}
    for spec in SPECS.values():
        record = dict(installed.get(spec.domain, {}) or {})
        if record.get("status") != "installed":
            continue
        manifest_path = dataset_manifest_for_id(
            root, str(record["dataset_id"]), spec.domain
        )
        if manifest_path is None:
            query_granularity[spec.domain] = ""
            continue
        manifest = read_dataset_manifest(manifest_path).to_dict()
        query_granularity[spec.domain] = str(
            dict(manifest.get("source", {}) or {}).get("query_granularity", "")
        )
    expected_query_granularity = {
        spec.domain: spec.mode for spec in SPECS.values() if not spec.optional
    }
    legacy_year_cache = _legacy_year_cache_summary(workspace, list(SPECS.values()))
    checks = {
        "applied": state.get("status") == "applied",
        "mandatory_domains_installed": all(
            dict(installed.get(domain, {}) or {}).get("status") == "installed"
            for domain in mandatory_domains
        ),
        "active_manifest_matches": all(
            active.get(domain)
            == dict(installed.get(domain, {}) or {}).get("dataset_id")
            for domain in mandatory_domains
        ),
        "factor_field_count_exact": dict(
            dict(installed.get(DataDomain.STK_FACTOR_PRO_RAW, {}) or {}).get(
                "quality", {}
            )
            or {}
        ).get("provider_field_count")
        == EXPECTED_FACTOR_FIELD_COUNT,
        "factor_schema_hash_exact": dict(
            dict(installed.get(DataDomain.STK_FACTOR_PRO_RAW, {}) or {}).get(
                "quality", {}
            )
            or {}
        ).get("provider_schema_hash")
        == EXPECTED_FACTOR_SCHEMA_HASH,
        "query_granularity_exact": query_granularity == expected_query_granularity,
        "legacy_year_range_cache_excluded": all(
            bool(record["excluded_from_prepared_inventory"])
            for record in legacy_year_cache.values()
        ),
        "forbidden_2026_rows": all(
            str(dict(installed.get(domain, {}) or {}).get("end_date", "")) <= END_DATE
            for domain in mandatory_domains
        )
        and all(
            int(record["forbidden_2026_rows"]) == 0
            for record in physical_domains.values()
        ),
        "physical_primary_keys_unique": all(
            bool(record["primary_key_unique"]) for record in physical_domains.values()
        ),
        "physical_pit_dates_valid": all(
            int(record["availability_before_source_rows"]) == 0
            and int(record["burn_in_flag_mismatch_rows"]) == 0
            for record in physical_domains.values()
        ),
        "physical_nonnegative_fields_valid": all(
            int(record["disallowed_negative_value_count"]) == 0
            for record in physical_domains.values()
        ),
        "credential_not_persisted": not credential_hits,
        "credential_not_persisted_outside_private_profile": not credential_hits,
        "training_not_performed": state.get("training_performed") is False,
    }
    status_value = "ok" if all(checks.values()) else "error"
    result = {
        "status": status_value,
        "update_id": UPDATE_ID,
        "checks": checks,
        "domains": installed,
        "physical_domains": physical_domains,
        "credential_file_hits": credential_hits,
        "provider_profile": provider_profile,
        "query_granularity": query_granularity,
        "legacy_year_range_cache": legacy_year_cache,
    }
    _assert_credential_free(result)
    atomic_write_json(_runtime(workspace) / "evaluation.json", result)
    if status_value != "ok":
        raise TushareExtendedBackfillError(f"extended_backfill_audit_failed:{checks}")
    return result


def cleanup_prepared_cache(
    *,
    workspace_root: str | Path | None = None,
    delete: bool = False,
    yes: bool = False,
) -> dict[str, Any]:
    """Remove only prepared shards already copied into the active QDP datasets."""

    if delete and not yes:
        raise ValueError("prepared_cache_cleanup_requires_yes")
    workspace = _workspace(workspace_root)
    runtime = _runtime(workspace).resolve()
    prepared_root = (runtime / "prepared").resolve()
    prepared_root.relative_to(runtime)
    if not prepared_root.is_dir():
        return {
            "status": "absent",
            "prepared_root": str(prepared_root),
            "deletable": True,
            "bytes": 0,
            "file_count": 0,
            "verified_domains": [],
            "blockers": [],
            "destructive_actions_performed": False,
        }

    state = _read_state(workspace)
    installed = dict(state.get("installed_domains", {}) or {})
    qdp_root = qdp_v2_root(workspace)
    active = active_dataset_map(read_active_manifest(qdp_root))
    blockers: list[str] = []
    verified_domains: list[dict[str, Any]] = []
    prepared_files = sorted(path for path in prepared_root.rglob("*") if path.is_file())

    for domain_dir in sorted(path for path in prepared_root.iterdir() if path.is_dir()):
        domain = domain_dir.name
        record = dict(installed.get(domain, {}) or {})
        dataset_id = str(record.get("dataset_id", "") or "")
        if record.get("status") != "installed" or not dataset_id:
            blockers.append(f"prepared_domain_not_recorded_as_installed:{domain}")
            continue
        if active.get(domain) != dataset_id:
            blockers.append(f"prepared_domain_not_active:{domain}:{dataset_id}")
            continue
        manifest_path = dataset_manifest_for_id(qdp_root, dataset_id, domain)
        if manifest_path is None:
            blockers.append(f"prepared_domain_manifest_missing:{domain}:{dataset_id}")
            continue
        manifest = read_dataset_manifest(manifest_path)
        installed_by_year = {
            int(
                dict(shard.metadata or {}).get("year") or str(shard.start_date)[:4]
            ): shard
            for shard in manifest.shards
        }
        domain_prepared = sorted(domain_dir.rglob("*.parquet"))
        verified_years: list[int] = []
        for path in domain_prepared:
            try:
                year = int(path.parent.name.removeprefix("year="))
            except ValueError:
                blockers.append(f"prepared_year_invalid:{path}")
                continue
            sidecar = path.with_suffix(".json")
            shard = installed_by_year.get(year)
            if shard is None:
                blockers.append(f"prepared_year_not_installed:{domain}:{year}")
                continue
            if not sidecar.is_file():
                blockers.append(f"prepared_sidecar_missing:{domain}:{year}")
                continue
            try:
                profile = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                blockers.append(f"prepared_sidecar_invalid:{domain}:{year}:{exc}")
                continue
            installed_path = resolve_manifest_path(shard.path, root=qdp_root)
            prepared_hash = str(profile.get("sha256", "") or "")
            installed_hash = str(dict(shard.metadata or {}).get("sha256", "") or "")
            expected_size = int(shard.file_size or 0)
            if not installed_path.is_file():
                blockers.append(f"installed_shard_missing:{domain}:{year}")
            elif not prepared_hash or prepared_hash != installed_hash:
                blockers.append(f"prepared_installed_hash_mismatch:{domain}:{year}")
            elif path.stat().st_size != installed_path.stat().st_size:
                blockers.append(f"prepared_installed_size_mismatch:{domain}:{year}")
            elif expected_size and installed_path.stat().st_size != expected_size:
                blockers.append(f"installed_manifest_size_mismatch:{domain}:{year}")
            else:
                verified_years.append(year)
        if set(verified_years) != set(installed_by_year):
            blockers.append(
                f"prepared_year_inventory_mismatch:{domain}:"
                f"{sorted(verified_years)}!={sorted(installed_by_year)}"
            )
        verified_domains.append(
            {
                "domain": domain,
                "dataset_id": dataset_id,
                "verified_years": sorted(verified_years),
                "prepared_parquet_count": len(domain_prepared),
            }
        )

    size = sum(path.stat().st_size for path in prepared_files)
    payload: dict[str, Any] = {
        "status": "blocked" if blockers else ("deleted" if delete else "dry_run"),
        "prepared_root": str(prepared_root),
        "deletable": not blockers,
        "bytes": int(size),
        "file_count": len(prepared_files),
        "verified_domains": verified_domains,
        "blockers": sorted(set(blockers)),
        "destructive_actions_performed": False,
    }
    if blockers:
        return payload
    if delete:
        shutil.rmtree(prepared_root)
        payload["destructive_actions_performed"] = True
        receipt = (
            qdp_root
            / "audits"
            / f"{UPDATE_ID}_prepared_cleanup_{utc_now().replace(':', '')}.json"
        )
        atomic_write_json(receipt, payload)
        payload["receipt_path"] = str(receipt)
    return payload


def status(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    probe_state = dict(state.get("probe", {}) or {})
    return {
        "update_id": UPDATE_ID,
        "status": state.get("status", "pending"),
        "probe_status": probe_state.get("status", "pending"),
        "endpoint_status": {
            name: dict(record).get("status", "pending")
            for name, record in dict(probe_state.get("endpoints", {}) or {}).items()
        },
        "download_progress": state.get("download_progress", {}),
        "installed_domains": state.get("installed_domains", {}),
        "provider_profile": tushare_provider_status(workspace),
        "training_performed": state.get("training_performed", False),
    }


def self_test() -> dict[str, Any]:
    if len(SPECS) != 5 or MAX_WORKERS != 3:
        raise AssertionError("endpoint or concurrency policy changed")
    if PAGE_SIZE != 5_000 or EXPECTED_FACTOR_FIELD_COUNT != 261:
        raise AssertionError("provider pagination or factor schema policy changed")
    if any(str(year) == str(FORBIDDEN_YEAR) for year in range(2010, 2026)):
        raise AssertionError("forbidden year entered download inventory")
    if _task_params(SPECS["margin"], "2011", 0)["end_date"] != "20111231":
        raise AssertionError("year task boundary changed")
    if _task_params(SPECS["moneyflow"], "2011-01-04", 0)["trade_date"] != "20110104":
        raise AssertionError("daily task normalization changed")
    return {
        "status": "ok",
        "checks": {
            "endpoint_count": len(SPECS),
            "factor_field_count": EXPECTED_FACTOR_FIELD_COUNT,
            "factor_schema_hash": EXPECTED_FACTOR_SCHEMA_HASH,
            "page_size": PAGE_SIZE,
            "maximum_workers": MAX_WORKERS,
            "maximum_requests_per_minute": MAX_REQUESTS_PER_MINUTE,
            "burn_in_year": 2010,
            "research_start_year": 2011,
            "forbidden_2026": True,
            "training_performed": False,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tushare-extended-backfill")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--domains", default="")
    parser.add_argument("--workers", "--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument(
        "--credential-stdin",
        action="store_true",
        help="Read the provider token without echo and keep it process-local.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--run-pending", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--cleanup-prepared", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--yes", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = str(args.workspace_root or "") or None
    if args.credential_stdin:
        token = getpass.getpass("Token: ").strip()
        if not token:
            raise TushareExtendedBackfillError("tushare_token_required")
        os.environ["QDP_TUSHARE_PROXY_TOKEN"] = token
        os.environ["QDP_TUSHARE_PREFER_ENV"] = "1"
        try:
            _resolve_tushare_api_url(workspace)
        except TushareExtendedBackfillError:
            api_url = getpass.getpass("API URL: ").strip()
            if not api_url:
                raise TushareExtendedBackfillError("tushare_api_url_missing") from None
            os.environ["QDP_TUSHARE_API_URL"] = api_url
    domains = str(args.domains or "") or None
    if args.probe:
        payload = probe(workspace_root=workspace)
    elif args.status:
        payload = status(workspace_root=workspace)
    elif args.run_pending:
        payload = run_pending(
            workspace_root=workspace,
            domains=domains,
            workers=int(args.workers),
        )
    elif args.prepare:
        payload = prepare_and_install(workspace_root=workspace, domains=domains)
    elif args.audit:
        payload = audit(workspace_root=workspace)
    elif args.cleanup_prepared:
        payload = cleanup_prepared_cache(
            workspace_root=workspace,
            delete=bool(args.delete),
            yes=bool(args.yes),
        )
    else:
        payload = self_test()
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
