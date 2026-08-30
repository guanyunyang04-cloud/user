"""Auxiliary update context operations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quantlab.core.io import json_safe
from quantlab.data.core.paths import qdp_paths
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb
from quantlab.data.qdp_v2.manifest import (
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    utc_now,
)
from quantlab.data.qdp_v2.provider_credentials import (
    ProviderCredentialError,
    resolve_tushare_api_url,
    resolve_tushare_rate_limit,
    resolve_tushare_token,
)
from quantlab.data.qdp_v2.repair.common import _sql_literal
from quantlab.data.qdp_v2.repair.mutation import replace_active_table_from_parquet
from quantlab.data.qdp_v2.status import active_dataset_map

AUXILIARY_DOMAINS = (
    "industry_concept",
    "name_change",
    "corporate_actions",
    "index_constituents",
    "share_capital",
    "valuation",
)


DAILY_AUXILIARY_DOMAINS = (
    "industry_concept",
    "share_capital",
    "valuation",
)

INDEX_SPECS = (
    ("000016.SH", "SSE 50", "sz50"),
    ("000300.SH", "CSI 300", "hs300"),
    ("000905.SH", "CSI 500", "zz500"),
)


TUSHARE_MAX_RPM = 96


TUSHARE_WORKERS = 3


BAOSTOCK_WORKERS = 4


SECONDARY_VALIDATION_WORKERS = 4


SECONDARY_VALIDATION_SAMPLE_SIZE = 200


SECONDARY_VALIDATION_SEED = 20260716


FREE_SOURCE_POLICY = "baostock+mootdx_detection+cninfo_confirmation; no_tushare"


LEGACY_SOURCE_POLICY = "legacy_tushare_plus_free_source_validation"


class AuxiliaryUpdateError(RuntimeError):
    pass


class TushareRateLimitError(AuxiliaryUpdateError):
    pass


@dataclass(frozen=True)
class AuxiliaryContext:
    workspace: Path
    root: Path
    active: dict[str, Any]
    datasets: dict[str, str]
    target_date: str
    runtime: Path


class _MinuteLimiter:
    def __init__(self, rpm: int) -> None:
        self._interval = 60.0 / max(1, int(rpm))
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self._interval
        if delay:
            time.sleep(delay)


class _TushareClient:
    def __init__(
        self,
        token: str,
        *,
        rpm: int = TUSHARE_MAX_RPM,
        workspace_root: str | Path | None = None,
    ) -> None:
        secret = str(token or "").strip()
        if not secret:
            raise AuxiliaryUpdateError("tushare_token_missing")
        self._token = secret
        self._url = _resolve_tushare_api_url(workspace_root)
        self._limiter = _MinuteLimiter(resolve_tushare_rate_limit(rpm, workspace_root=workspace_root))

    def fetch(
        self,
        api_name: str,
        *,
        params: Mapping[str, Any],
        fields: Sequence[str],
        retries: int = 4,
    ) -> pd.DataFrame:
        last_error = ""
        for attempt in range(max(1, int(retries))):
            self._limiter.wait()
            try:
                response = requests.post(
                    self._url,
                    json={
                        "api_name": str(api_name),
                        "token": self._token,
                        "params": dict(params),
                        "fields": ",".join(fields),
                    },
                    headers={"Accept-Encoding": "gzip"},
                    timeout=45,
                )
                if response.status_code == 429:
                    try:
                        rate_payload = dict(response.json() or {})
                    except (TypeError, ValueError):
                        rate_payload = {}
                    raise TushareRateLimitError(
                        f"tushare_rate_limited:{api_name}:status=429:"
                        f"code={rate_payload.get('code')}:{str(rate_payload.get('msg', ''))[:300]}"
                    )
                response.raise_for_status()
                payload = response.json()
                if int(payload.get("code", -1)) != 0:
                    raise AuxiliaryUpdateError(f"tushare_api_error:{api_name}:code={payload.get('code')}")
                data = dict(payload.get("data", {}) or {})
                names = [str(item) for item in list(data.get("fields", []) or [])]
                rows = list(data.get("items", []) or [])
                return pd.DataFrame(rows, columns=names or list(fields))
            except TushareRateLimitError:
                raise
            except Exception as exc:
                last_error = f"{type(exc).__name__}:{str(exc)[:300]}"
                if attempt + 1 >= max(1, int(retries)):
                    break
                time.sleep(min(2**attempt, 8))
        raise AuxiliaryUpdateError(f"tushare_request_failed:{api_name}:{last_error}")


def _resolve_tushare_token(workspace_root: str | Path | None = None) -> str:
    return resolve_tushare_token(workspace_root)


def _resolve_tushare_api_url(workspace_root: str | Path | None = None) -> str:
    try:
        return resolve_tushare_api_url(workspace_root)
    except ProviderCredentialError as exc:
        raise AuxiliaryUpdateError(str(exc)) from exc


def _context(
    *,
    as_of_date: str,
    workspace_root: str | Path | None,
) -> AuxiliaryContext:
    workspace = Path(workspace_root or Path.cwd()).resolve()
    root = qdp_v2_root(workspace).resolve()
    active = read_active_manifest(root)
    if not active:
        raise AuxiliaryUpdateError("active_manifest_missing")
    datasets = active_dataset_map(active)
    required = {
        "market_daily_raw",
        "trading_calendar",
        "security_identity",
        "symbol_history",
        "universe_snapshot",
        *AUXILIARY_DOMAINS,
    }
    missing = sorted(required.difference(datasets))
    if missing:
        raise AuxiliaryUpdateError(f"auxiliary_required_domains_missing:{','.join(missing)}")
    target = _date_text(as_of_date)
    daily_manifest = _manifest(root, datasets, "market_daily_raw")
    if target > str(daily_manifest.end_date):
        raise AuxiliaryUpdateError(f"auxiliary_target_after_market_daily:{target}>{daily_manifest.end_date}")
    runtime = (qdp_paths(workspace).data_dir / "qdp_runtime" / "auxiliary_repair").resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ["QDP_MOOTDX_LAST_GOOD_PATH"] = str(runtime / "mootdx_last_good_5m.json")
    return AuxiliaryContext(
        workspace=workspace,
        root=root,
        active=active,
        datasets=datasets,
        target_date=target,
        runtime=runtime,
    )


def _manifest(root: Path, datasets: Mapping[str, str], domain: str) -> Any:
    path = dataset_manifest_for_id(root, str(datasets[domain]), domain)
    if path is None:
        raise AuxiliaryUpdateError(f"active_dataset_manifest_missing:{domain}")
    return read_dataset_manifest(path)


def _identity_scope_signature(ctx: AuxiliaryContext) -> str:
    manifest = _manifest(ctx.root, ctx.datasets, "security_identity")
    payload = {
        "dataset_id": manifest.dataset_id,
        "row_count": int(manifest.row_count),
        "start_date": manifest.start_date,
        "end_date": manifest.end_date,
        "shards": [
            {
                "path": item.path,
                "row_count": int(item.row_count),
                "start_date": item.start_date,
                "end_date": item.end_date,
                "file_size": int(item.file_size),
            }
            for item in manifest.shards
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity_dependent_repair_required(
    ctx: AuxiliaryContext,
    domain: str,
) -> bool:
    if domain not in {"industry_concept", "index_constituents"}:
        return False
    manifest = _manifest(ctx.root, ctx.datasets, domain)
    recorded = str(dict(manifest.source or {}).get("identity_scope_signature", ""))
    return recorded != _identity_scope_signature(ctx)


def _paths(ctx: AuxiliaryContext, domain: str) -> list[Path]:
    manifest = _manifest(ctx.root, ctx.datasets, domain)
    return [resolve_manifest_path(item.path, root=ctx.root) for item in manifest.shards]


def _scan_sql(paths: Sequence[Path]) -> str:
    if not paths:
        raise AuxiliaryUpdateError("parquet_paths_empty")
    values = ",".join(_sql_literal(str(Path(item).resolve())) for item in paths)
    return f"read_parquet([{values}], union_by_name=true)"


def _date_text(value: Any) -> str:
    parsed = pd.Timestamp(value).normalize()
    return parsed.strftime("%Y-%m-%d")


def _runtime_state_path(ctx: AuxiliaryContext) -> Path:
    return ctx.runtime / "state.json"


def _write_state(ctx: AuxiliaryContext, payload: Mapping[str, Any]) -> None:
    atomic_write_json(
        _runtime_state_path(ctx),
        {"updated_at": utc_now(), **dict(payload)},
    )


def _parquet_schema_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.read_schema(path).names)


def _copy_query(
    ctx: AuxiliaryContext,
    *,
    sql: str,
    target: Path,
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    target.unlink(missing_ok=True)
    spill = ctx.runtime / "duckdb_spill"
    try:
        with open_guarded_duckdb(temp_directory=spill, threads=4) as con:
            con.execute(
                f"COPY ({sql}) TO {_sql_literal(str(temporary))} "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)"
            )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
        shutil.rmtree(spill, ignore_errors=True)
    return target


def _replace_domain(
    ctx: AuxiliaryContext,
    *,
    domain: str,
    prepared: Path,
    primary_key: Sequence[str],
    contract_version: str,
    source_contract: str,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    material_mismatches = int(validation.get("secondary_material_mismatch_count", 0) or 0)
    validation_source_updates = {
        str(key): json_safe(value) for key, value in validation.items() if str(key).startswith("secondary_")
    }
    identity_dependent = domain in {"industry_concept", "index_constituents"}
    scope_updates = (
        {
            "scope": "point_in_time_historical_mainboard",
            "survivorship_policy": ("include_when_listed_then_apply_same_day_status"),
            "identity_scope_signature": _identity_scope_signature(ctx),
        }
        if identity_dependent
        else {}
    )
    quality_scope_updates = (
        {
            "scope": "point_in_time_historical_mainboard",
            "survivorship_bias_free_mainboard_daily": True,
        }
        if identity_dependent
        else {}
    )
    if domain == "index_constituents":
        quality_scope_updates.update(
            {
                "permanent_exclusions_applied": False,
                "permanent_exclusions": {
                    "applied": False,
                    "reason": ("historical membership follows dated published index snapshots"),
                },
            }
        )
    result = replace_active_table_from_parquet(
        domain,
        prepared,
        reason="strict PIT auxiliary repair 2010 through current market date",
        workspace_root=ctx.workspace,
        primary_key=primary_key,
        contract_version=contract_version,
        source_updates={
            **validation_source_updates,
            **scope_updates,
            "checked_through": ctx.target_date,
            "source_contract": source_contract,
            "missing_daily_keys": (0 if domain in DAILY_AUXILIARY_DOMAINS else None),
            "secondary_validation_at": utc_now(),
            "secondary_compared_count": int(validation.get("secondary_compared_count", 0) or 0),
            "secondary_material_mismatch_count": material_mismatches,
            "secondary_validation_status": str(validation.get("secondary_validation_status", "ok")),
        },
        quality_updates={
            **quality_scope_updates,
            "strict_point_in_time": True,
            "future_source_dates": 0,
        },
    )
    prepared.unlink(missing_ok=True)
    return result


def plan_auxiliary_update(
    *,
    as_of_date: str,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    ctx = _context(as_of_date=as_of_date, workspace_root=workspace_root)
    daily = _scan_sql(_paths(ctx, "market_daily_raw"))
    coverage: dict[str, Any] = {}
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "plan_spill",
        threads=2,
    ) as con:
        for domain in AUXILIARY_DOMAINS:
            manifest = _manifest(ctx.root, ctx.datasets, domain)
            item: dict[str, Any] = {
                "start_date": manifest.start_date,
                "end_date": manifest.end_date,
                "row_count": manifest.row_count,
                "checked_through": str(dict(manifest.source or {}).get("checked_through", "")),
                "identity_scope_refresh_required": (_identity_dependent_repair_required(ctx, domain)),
            }
            if domain in DAILY_AUXILIARY_DOMAINS:
                current = _scan_sql(_paths(ctx, domain))
                missing = con.execute(
                    f"SELECT count(*) FROM ("
                    f"SELECT symbol, trade_date FROM {daily} "
                    f"WHERE trade_date <= ? EXCEPT "
                    f"SELECT symbol, trade_date FROM {current}"
                    ")",
                    [ctx.target_date],
                ).fetchone()[0]
                item["missing_daily_keys"] = int(missing)
            coverage[domain] = item
    shutil.rmtree(ctx.runtime / "plan_spill", ignore_errors=True)
    return {
        "status": "planned",
        "as_of_date": ctx.target_date,
        "domains": list(AUXILIARY_DOMAINS),
        "provider_policy": FREE_SOURCE_POLICY,
        "coverage_before": coverage,
        "provider_order": {
            "industry_concept": ["baostock", "cninfo_validation"],
            "share_capital": ["mootdx_detection", "cninfo_confirmation", "state_carry"],
            "valuation": ["baostock", "qdp_share_price_formula"],
            "name_change": ["universe_transitions"],
            "corporate_actions": ["mootdx_detection", "cninfo_confirmation"],
            "index_constituents": ["baostock"],
        },
    }


def _open_dates(ctx: AuxiliaryContext) -> list[str]:
    calendar = _scan_sql(_paths(ctx, "trading_calendar"))
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "calendar_spill",
        threads=1,
    ) as con:
        rows = con.execute(
            f"SELECT DISTINCT trade_date FROM {calendar} "
            "WHERE is_open AND exchange='SSE' AND trade_date BETWEEN '2010-01-04' AND ? "
            "ORDER BY trade_date",
            [ctx.target_date],
        ).fetchall()
    shutil.rmtree(ctx.runtime / "calendar_spill", ignore_errors=True)
    return [str(item[0]) for item in rows]


def _snapshot_dates(open_dates: Sequence[str]) -> list[str]:
    if not open_dates:
        return []
    frame = pd.DataFrame({"trade_date": pd.to_datetime(list(open_dates))})
    month_end = (
        frame.assign(month=frame["trade_date"].dt.to_period("M"))
        .groupby("month", sort=True)["trade_date"]
        .max()
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )
    values = {str(open_dates[0]), str(open_dates[-1]), *month_end}
    return sorted(values)


def _split_evenly(items: Sequence[str], parts: int) -> list[list[str]]:
    buckets = [[] for _ in range(max(1, min(int(parts), len(items) or 1)))]
    for index, item in enumerate(items):
        buckets[index % len(buckets)].append(str(item))
    return [item for item in buckets if item]


def _external_with_retry(call: Any, *, label: str, retries: int = 3) -> Any:
    last_error = ""
    for attempt in range(max(1, int(retries))):
        try:
            return call()
        except Exception as exc:
            last_error = f"{type(exc).__name__}:{str(exc)[:200]}"
            if attempt + 1 < max(1, int(retries)):
                time.sleep(min(2**attempt, 4))
    raise AuxiliaryUpdateError(f"secondary_validation_request_failed:{label}:{last_error}")
