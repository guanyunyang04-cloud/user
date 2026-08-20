from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from quantlab.data.core.json_io import json_safe
from quantlab.data.core.paths import qdp_paths
from quantlab.data.qdp_v2 import normalization as _normalization
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
from quantlab.data.qdp_v2.repair import (
    _sql_literal,
    replace_active_table_from_parquet,
    update_active_manifest_metadata,
)
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


class AuxiliaryUpdateError(RuntimeError):
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
        self._limiter = _MinuteLimiter(
            resolve_tushare_rate_limit(rpm, workspace_root=workspace_root)
        )

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
                response.raise_for_status()
                payload = response.json()
                if int(payload.get("code", -1)) != 0:
                    raise AuxiliaryUpdateError(
                        f"tushare_api_error:{api_name}:code={payload.get('code')}"
                    )
                data = dict(payload.get("data", {}) or {})
                names = [str(item) for item in list(data.get("fields", []) or [])]
                rows = list(data.get("items", []) or [])
                return pd.DataFrame(rows, columns=names or list(fields))
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
        raise AuxiliaryUpdateError(
            f"auxiliary_required_domains_missing:{','.join(missing)}"
        )
    target = _date_text(as_of_date)
    daily_manifest = _manifest(root, datasets, "market_daily_raw")
    if target > str(daily_manifest.end_date):
        raise AuxiliaryUpdateError(
            f"auxiliary_target_after_market_daily:{target}>{daily_manifest.end_date}"
        )
    runtime = (
        qdp_paths(workspace).data_dir / "qdp_runtime" / "auxiliary_repair"
    ).resolve()
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
    material_mismatches = int(
        validation.get("secondary_material_mismatch_count", 0) or 0
    )
    validation_source_updates = {
        str(key): json_safe(value)
        for key, value in validation.items()
        if str(key).startswith("secondary_")
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
                    "reason": (
                        "historical membership follows dated published index snapshots"
                    ),
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
            "secondary_compared_count": int(
                validation.get("secondary_compared_count", 0) or 0
            ),
            "secondary_material_mismatch_count": material_mismatches,
            "secondary_validation_status": str(
                validation.get("secondary_validation_status", "ok")
            ),
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
                "checked_through": str(
                    dict(manifest.source or {}).get("checked_through", "")
                ),
                "identity_scope_refresh_required": (
                    _identity_dependent_repair_required(ctx, domain)
                ),
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
        "coverage_before": coverage,
        "provider_order": {
            "industry_concept": ["baostock", "cninfo_validation"],
            "share_capital": ["tushare_daily_basic", "cninfo_validation"],
            "valuation": ["existing_baostock", "tushare_daily_basic_gap_fill"],
            "name_change": ["universe_transitions", "tushare_validation"],
            "corporate_actions": ["tushare_dividend", "cninfo_validation"],
            "index_constituents": ["baostock", "tushare_validation"],
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


def _baostock_snapshot_worker(
    kind: str,
    dates: Sequence[str],
    output_dir: str,
) -> dict[str, Any]:
    import baostock as bs

    columns = _baostock_snapshot_columns(kind)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    row_count = 0
    failures: list[str] = []
    outputs: list[str] = []
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        login = bs.login()
    if str(getattr(login, "error_code", "1")) != "0":
        raise AuxiliaryUpdateError("baostock_snapshot_login_failed")
    try:
        for trade_date in dates:
            success = False
            for attempt in range(3):
                date_rows: list[dict[str, Any]] = []
                try:
                    if kind == "industry":
                        query = bs.query_stock_industry(date=str(trade_date))
                        if str(query.error_code) != "0":
                            raise AuxiliaryUpdateError(
                                f"baostock_industry_query_failed:{trade_date}"
                            )
                        while query.error_code == "0" and query.next():
                            row = dict(zip(query.fields, query.get_row_data(), strict=True))
                            code = _from_baostock_code(row.get("code", ""))
                            if code:
                                date_rows.append(
                                    {
                                        "symbol": code,
                                        "snapshot_query_date": str(trade_date),
                                        "industry": str(
                                            row.get("industry", "") or ""
                                        ).strip(),
                                        "industry_standard": str(
                                            row.get("industryClassification", "")
                                            or "证监会行业分类"
                                        ).strip(),
                                        "source_date": _provider_date(
                                            row.get("updateDate"), trade_date
                                        ),
                                        "source": "baostock.query_stock_industry",
                                    }
                                )
                        if not date_rows:
                            raise AuxiliaryUpdateError(
                                f"baostock_industry_query_empty:{trade_date}"
                            )
                    elif kind == "index":
                        index_counts: dict[str, int] = {}
                        for index_symbol, index_name, endpoint in INDEX_SPECS:
                            before = len(date_rows)
                            query_func = getattr(bs, f"query_{endpoint}_stocks")
                            query = query_func(date=str(trade_date))
                            if str(query.error_code) != "0":
                                raise AuxiliaryUpdateError(
                                    f"baostock_index_query_failed:{endpoint}:{trade_date}"
                                )
                            while query.error_code == "0" and query.next():
                                row = dict(zip(query.fields, query.get_row_data(), strict=True))
                                code = _from_baostock_code(row.get("code", ""))
                                if code:
                                    date_rows.append(
                                        {
                                            "index_symbol": index_symbol,
                                            "symbol": code,
                                            "index_name": index_name,
                                            "snapshot_query_date": str(trade_date),
                                            "source_snapshot_date": str(trade_date),
                                            "source": "baostock",
                                        }
                                    )
                            index_counts[index_symbol] = len(date_rows) - before
                        missing_indexes = [
                            symbol
                            for symbol, _, _ in INDEX_SPECS
                            if index_counts.get(symbol, 0) <= 0
                        ]
                        if missing_indexes:
                            raise AuxiliaryUpdateError(
                                "baostock_index_query_empty:"
                                f"{trade_date}:{','.join(missing_indexes)}"
                            )
                    else:
                        raise ValueError(f"unsupported_baostock_snapshot_kind:{kind}")
                    frame = pd.DataFrame(date_rows, columns=columns)
                    output = destination / _baostock_snapshot_part_name(trade_date)
                    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
                    temporary.unlink(missing_ok=True)
                    try:
                        frame.to_parquet(
                            temporary,
                            index=False,
                            compression="zstd",
                        )
                        temporary.replace(output)
                    finally:
                        temporary.unlink(missing_ok=True)
                    row_count += int(len(frame))
                    outputs.append(str(output))
                    success = True
                    break
                except Exception:
                    if attempt == 2:
                        failures.append(str(trade_date))
                    else:
                        with (
                            redirect_stdout(io.StringIO()),
                            redirect_stderr(io.StringIO()),
                        ):
                            try:
                                bs.logout()
                            except Exception:
                                pass
                            login = bs.login()
                        if str(getattr(login, "error_code", "1")) != "0":
                            time.sleep(1 + attempt)
                            continue
                        time.sleep(1 + attempt)
            if not success:
                continue
    finally:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                bs.logout()
            except Exception:
                pass
    return {
        "row_count": row_count,
        "failure_dates": failures,
        "output_paths": outputs,
    }


def _baostock_snapshot_columns(kind: str) -> list[str]:
    if kind == "industry":
        return [
            "symbol",
            "snapshot_query_date",
            "industry",
            "industry_standard",
            "source_date",
            "source",
        ]
    if kind == "index":
        return [
            "index_symbol",
            "symbol",
            "index_name",
            "snapshot_query_date",
            "source_snapshot_date",
            "source",
        ]
    raise ValueError(f"unsupported_baostock_snapshot_kind:{kind}")


def _baostock_snapshot_part_name(trade_date: str) -> str:
    return f"snapshot_{str(trade_date).replace('-', '')}.parquet"


def _valid_parquet_columns(path: Path, expected: Sequence[str]) -> bool:
    if not path.is_file():
        return False
    try:
        return _parquet_schema_columns(path) == list(expected)
    except Exception:
        return False


def _provider_date(value: Any, fallback: Any) -> str:
    return _normalization.provider_date(value, fallback)


def _from_baostock_code(value: Any) -> str:
    return _normalization.from_baostock_code(value)


def _normalize_comparison_text(value: Any) -> str:
    return _normalization.normalize_comparison_text(value)


def _normalize_industry_comparison(value: Any) -> str:
    return _normalization.normalize_industry_comparison(value)


def _external_with_retry(call: Any, *, label: str, retries: int = 3) -> Any:
    last_error = ""
    for attempt in range(max(1, int(retries))):
        try:
            return call()
        except Exception as exc:
            last_error = f"{type(exc).__name__}:{str(exc)[:200]}"
            if attempt + 1 < max(1, int(retries)):
                time.sleep(min(2**attempt, 4))
    raise AuxiliaryUpdateError(
        f"secondary_validation_request_failed:{label}:{last_error}"
    )


def validate_industry_secondary(ctx: AuxiliaryContext) -> dict[str, Any]:
    import akshare as ak

    industry = _scan_sql(_paths(ctx, "industry_concept"))
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "industry_secondary_spill",
        threads=1,
    ) as con:
        latest = con.execute(
            f"SELECT symbol, industry FROM {industry} WHERE trade_date=? "
            "ORDER BY symbol",
            [ctx.target_date],
        ).fetchdf()
    shutil.rmtree(ctx.runtime / "industry_secondary_spill", ignore_errors=True)
    if len(latest) < SECONDARY_VALIDATION_SAMPLE_SIZE:
        raise AuxiliaryUpdateError(
            f"industry_secondary_population_too_small:{len(latest)}"
        )
    sample = latest.sample(
        n=SECONDARY_VALIDATION_SAMPLE_SIZE,
        random_state=SECONDARY_VALIDATION_SEED,
    ).sort_values("symbol")

    def fetch_one(row: Any) -> dict[str, str]:
        symbol = str(row.symbol)
        code = symbol.split(".", 1)[0]
        profile = _external_with_retry(
            lambda: ak.stock_profile_cninfo(symbol=code),
            label=f"industry_profile:{symbol}",
        )
        secondary = ""
        if isinstance(profile, pd.DataFrame) and not profile.empty:
            secondary = _normalize_comparison_text(profile.iloc[-1].get("所属行业", ""))
        if not secondary:
            changes = _external_with_retry(
                lambda: ak.stock_industry_change_cninfo(
                    symbol=code,
                    start_date="20100101",
                    end_date=ctx.target_date.replace("-", ""),
                ),
                label=f"industry_change:{symbol}",
            )
            if isinstance(changes, pd.DataFrame) and not changes.empty:
                values = changes.copy()
                if "分类标准编码" in values:
                    csrc = values["分类标准编码"].astype(str).eq("008001")
                    if csrc.any():
                        values = values.loc[csrc]
                if "变更日期" in values:
                    values = values.sort_values("变更日期")
                latest_change = values.iloc[-1]
                for column in ("行业大类", "行业中类", "行业门类"):
                    candidate = _normalize_comparison_text(
                        latest_change.get(column, "")
                    )
                    if candidate:
                        secondary = candidate
                        break
        return {
            "symbol": symbol,
            "primary": str(row.industry),
            "secondary": secondary,
        }

    rows: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=SECONDARY_VALIDATION_WORKERS) as pool:
        futures = {
            pool.submit(fetch_one, row): str(row.symbol)
            for row in sample.itertuples(index=False)
        }
        for future in as_completed(futures):
            rows.append(future.result())
    comparable = [item for item in rows if item["secondary"]]
    if len(comparable) != SECONDARY_VALIDATION_SAMPLE_SIZE:
        raise AuxiliaryUpdateError(
            "industry_secondary_incomplete:"
            f"compared={len(comparable)}:expected={SECONDARY_VALIDATION_SAMPLE_SIZE}"
        )
    mismatches = [
        item
        for item in comparable
        if _normalize_industry_comparison(item["primary"])
        != _normalize_industry_comparison(item["secondary"])
    ]
    match_rate = 1.0 - (len(mismatches) / len(comparable))
    if match_rate < 0.95:
        examples = ";".join(
            f"{item['symbol']}:{item['primary']}!={item['secondary']}"
            for item in mismatches[:5]
        )
        raise AuxiliaryUpdateError(
            f"industry_secondary_match_rate_failed:{match_rate:.6f}:{examples}"
        )
    metadata = update_active_manifest_metadata(
        "industry_concept",
        reason="CNInfo deterministic current-industry sample validation",
        workspace_root=ctx.workspace,
        source_updates={
            "secondary_validation_at": utc_now(),
            "secondary_compared_count": len(comparable),
            "secondary_material_mismatch_count": 0,
            "secondary_validation_status": "ok",
            "secondary_match_rate": round(match_rate, 8),
            "secondary_raw_mismatch_count": len(mismatches),
        },
    )
    return {
        **metadata,
        "compared_count": len(comparable),
        "raw_mismatch_count": len(mismatches),
        "match_rate": match_rate,
        "mismatch_examples": mismatches[:5],
    }


def validate_share_capital_secondary(ctx: AuxiliaryContext) -> dict[str, Any]:
    import akshare as ak

    share = _scan_sql(_paths(ctx, "share_capital"))
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "share_secondary_spill",
        threads=1,
    ) as con:
        latest = con.execute(
            f"SELECT symbol, total_share, float_share FROM {share} "
            "WHERE trade_date=? ORDER BY symbol",
            [ctx.target_date],
        ).fetchdf()
    shutil.rmtree(ctx.runtime / "share_secondary_spill", ignore_errors=True)
    if len(latest) < SECONDARY_VALIDATION_SAMPLE_SIZE:
        raise AuxiliaryUpdateError(
            f"share_secondary_population_too_small:{len(latest)}"
        )
    sample = latest.sample(
        n=SECONDARY_VALIDATION_SAMPLE_SIZE,
        random_state=SECONDARY_VALIDATION_SEED,
    ).sort_values("symbol")

    def fetch_one(row: Any) -> dict[str, Any]:
        symbol = str(row.symbol)
        code = symbol.split(".", 1)[0]
        frame = _external_with_retry(
            lambda: ak.stock_share_change_cninfo(
                symbol=code,
                start_date="19900101",
                end_date=ctx.target_date.replace("-", ""),
            ),
            label=f"share_change:{symbol}",
        )
        total = None
        floating = None
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            data = frame.copy()
            if "变动日期" in data:
                data["变动日期"] = pd.to_datetime(data["变动日期"], errors="coerce")
                data = data.loc[data["变动日期"].le(pd.Timestamp(ctx.target_date))]
            if "公告日期" in data:
                announcement = pd.to_datetime(data["公告日期"], errors="coerce")
                data = data.loc[
                    announcement.isna() | announcement.le(pd.Timestamp(ctx.target_date))
                ]
            if not data.empty:
                if "变动日期" in data:
                    data = data.sort_values("变动日期")
                last = data.iloc[-1]
                total_value = pd.to_numeric(
                    pd.Series([last.get("总股本")]), errors="coerce"
                ).iloc[0]
                float_value = pd.to_numeric(
                    pd.Series(
                        [
                            last.get("人民币普通股")
                            if pd.notna(last.get("人民币普通股"))
                            else last.get("已流通股份")
                        ]
                    ),
                    errors="coerce",
                ).iloc[0]
                if pd.notna(total_value):
                    total = float(total_value) * 10_000.0
                if pd.notna(float_value):
                    floating = float(float_value) * 10_000.0
        return {
            "symbol": symbol,
            "primary_total": float(row.total_share),
            "primary_float": float(row.float_share),
            "secondary_total": total,
            "secondary_float": floating,
        }

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=SECONDARY_VALIDATION_WORKERS) as pool:
        futures = {
            pool.submit(fetch_one, row): str(row.symbol)
            for row in sample.itertuples(index=False)
        }
        for future in as_completed(futures):
            rows.append(future.result())
    total_rows = [item for item in rows if item["secondary_total"] is not None]
    float_rows = [item for item in rows if item["secondary_float"] is not None]
    if len(total_rows) != SECONDARY_VALIDATION_SAMPLE_SIZE or len(float_rows) < 190:
        raise AuxiliaryUpdateError(
            "share_secondary_incomplete:"
            f"total={len(total_rows)}:float={len(float_rows)}:"
            f"expected={SECONDARY_VALIDATION_SAMPLE_SIZE}"
        )

    def relative_error(primary: float, secondary: float) -> float:
        return abs(primary - secondary) / max(abs(secondary), 1.0)

    total_mismatches = [
        item
        for item in total_rows
        if relative_error(item["primary_total"], item["secondary_total"]) > 0.0001
    ]
    float_mismatches = [
        item
        for item in float_rows
        if relative_error(item["primary_float"], item["secondary_float"]) > 0.01
    ]
    total_match_rate = 1.0 - len(total_mismatches) / len(total_rows)
    float_match_rate = 1.0 - len(float_mismatches) / len(float_rows)
    if total_match_rate < 0.999 or float_match_rate < 0.99:
        raise AuxiliaryUpdateError(
            "share_secondary_match_rate_failed:"
            f"total={total_match_rate:.8f}:float={float_match_rate:.8f}"
        )
    metadata = update_active_manifest_metadata(
        "share_capital",
        reason="CNInfo deterministic current-share sample validation",
        workspace_root=ctx.workspace,
        source_updates={
            "secondary_validation_at": utc_now(),
            "secondary_compared_count": len(total_rows),
            "secondary_material_mismatch_count": 0,
            "secondary_validation_status": "ok",
            "secondary_total_share_match_rate": round(total_match_rate, 8),
            "secondary_float_share_match_rate": round(float_match_rate, 8),
            "secondary_raw_mismatch_count": (
                len(total_mismatches) + len(float_mismatches)
            ),
        },
    )
    return {
        **metadata,
        "total_compared_count": len(total_rows),
        "float_compared_count": len(float_rows),
        "total_match_rate": total_match_rate,
        "float_match_rate": float_match_rate,
        "total_mismatch_examples": total_mismatches[:5],
        "float_mismatch_examples": float_mismatches[:5],
    }


def validate_index_secondary(ctx: AuxiliaryContext) -> dict[str, Any]:
    import akshare as ak

    index = _scan_sql(_paths(ctx, "index_constituents"))
    identity = _scan_sql(_paths(ctx, "security_identity"))
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "index_secondary_spill",
        threads=1,
    ) as con:
        current = con.execute(
            f"SELECT index_symbol, symbol FROM {index} WHERE trade_date=?",
            [ctx.target_date],
        ).fetchdf()
        allowed = {
            str(item[0])
            for item in con.execute(f"SELECT current_symbol FROM {identity}").fetchall()
        }
    shutil.rmtree(ctx.runtime / "index_secondary_spill", ignore_errors=True)

    comparisons: dict[str, dict[str, Any]] = {}
    total_compared = 0
    total_raw_mismatches = 0
    for index_symbol, _, _ in INDEX_SPECS:
        code = index_symbol.split(".", 1)[0]
        official = _external_with_retry(
            lambda code=code: ak.index_stock_cons_csindex(symbol=code),
            label=f"index_constituents:{index_symbol}",
        )
        if not isinstance(official, pd.DataFrame) or official.empty:
            raise AuxiliaryUpdateError(f"index_secondary_empty:{index_symbol}")
        official_members: set[str] = set()
        for row in official.itertuples(index=False):
            payload = row._asdict()
            raw_code = str(payload.get("成分券代码", "") or "").zfill(6)
            exchange = str(payload.get("交易所", "") or "")
            suffix = (
                "SH"
                if "上海" in exchange
                else "SZ"
                if "深圳" in exchange
                else "SH"
                if raw_code.startswith(("6", "9"))
                else "SZ"
            )
            symbol = f"{raw_code}.{suffix}"
            if symbol in allowed:
                official_members.add(symbol)
        primary_members = set(
            current.loc[
                current["index_symbol"].astype(str).eq(index_symbol),
                "symbol",
            ].astype(str)
        )
        union = primary_members | official_members
        intersection = primary_members & official_members
        jaccard = len(intersection) / len(union) if union else 0.0
        raw_mismatches = len(primary_members ^ official_members)
        comparisons[index_symbol] = {
            "primary_count": len(primary_members),
            "secondary_count": len(official_members),
            "jaccard": jaccard,
            "raw_mismatch_count": raw_mismatches,
            "only_primary": sorted(primary_members - official_members)[:10],
            "only_secondary": sorted(official_members - primary_members)[:10],
        }
        total_compared += len(official_members)
        total_raw_mismatches += raw_mismatches
    failed = [
        symbol
        for symbol, payload in comparisons.items()
        if float(payload["jaccard"]) < 0.99
    ]
    if failed:
        raise AuxiliaryUpdateError(
            "index_secondary_jaccard_failed:"
            + ",".join(
                f"{symbol}={comparisons[symbol]['jaccard']:.6f}" for symbol in failed
            )
        )
    metadata = update_active_manifest_metadata(
        "index_constituents",
        reason="CSIndex latest constituent-set validation",
        workspace_root=ctx.workspace,
        source_updates={
            "secondary_validation_at": utc_now(),
            "secondary_compared_count": total_compared,
            "secondary_material_mismatch_count": 0,
            "secondary_validation_status": "ok",
            "secondary_raw_mismatch_count": total_raw_mismatches,
            "secondary_minimum_jaccard": round(
                min(item["jaccard"] for item in comparisons.values()), 8
            ),
        },
    )
    return {
        **metadata,
        "compared_count": total_compared,
        "raw_mismatch_count": total_raw_mismatches,
        "comparisons": comparisons,
    }


def _fetch_baostock_snapshots(
    ctx: AuxiliaryContext,
    *,
    kind: str,
    dates: Sequence[str],
) -> list[Path]:
    unique_dates = sorted(set(str(item) for item in dates))
    if not unique_dates:
        raise AuxiliaryUpdateError(f"baostock_{kind}_snapshot_dates_empty")
    output_dir = ctx.runtime / f"baostock_{kind}_parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_columns = _baostock_snapshot_columns(kind)
    outputs = [
        output_dir / _baostock_snapshot_part_name(trade_date)
        for trade_date in unique_dates
    ]
    pending_dates: list[str] = []
    for trade_date, output in zip(unique_dates, outputs, strict=True):
        if _valid_parquet_columns(output, expected_columns):
            continue
        output.unlink(missing_ok=True)
        pending_dates.append(trade_date)
    chunks = _split_evenly(pending_dates, BAOSTOCK_WORKERS)
    if chunks:
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            futures = {
                pool.submit(
                    _baostock_snapshot_worker,
                    kind,
                    chunk,
                    str(output_dir),
                ): chunk
                for chunk in chunks
            }
            failures: list[str] = []
            for future in as_completed(futures):
                result = future.result()
                failures.extend(list(result.get("failure_dates", []) or []))
        if failures:
            raise AuxiliaryUpdateError(
                f"baostock_{kind}_snapshot_failures:{len(failures)}:"
                + ",".join(failures[:10])
            )
    incomplete = [
        trade_date
        for trade_date, output in zip(unique_dates, outputs, strict=True)
        if not _valid_parquet_columns(output, expected_columns)
    ]
    if incomplete:
        raise AuxiliaryUpdateError(
            f"baostock_{kind}_snapshot_incomplete:{len(incomplete)}:"
            + ",".join(incomplete[:10])
        )
    return outputs


def _industry_query_dates(ctx: AuxiliaryContext) -> list[str]:
    open_dates = _open_dates(ctx)
    monthly = set(_snapshot_dates(open_dates))
    daily = _scan_sql(_paths(ctx, "market_daily_raw"))
    current = _scan_sql(_paths(ctx, "industry_concept"))
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "industry_plan_spill",
        threads=2,
    ) as con:
        seed_dates = con.execute(
            f"""
            WITH daily_first AS (
              SELECT symbol, min(trade_date) AS first_date
              FROM {daily}
              WHERE trade_date<=?
              GROUP BY symbol
            ), valid_sources AS (
              SELECT symbol,
                     min(coalesce(nullif(industry_source_date, ''), trade_date))
                       AS first_source_date
              FROM {current}
              WHERE coalesce(trim(industry), '')<>''
                AND lower(trim(industry)) NOT IN ('unknown', 'unclassified')
                AND coalesce(industry_fill_method, '')<>'unavailable'
                AND coalesce(original_source, '')<>
                    'qdp_v2_industry_initial_bfill'
                AND try_cast(
                      coalesce(nullif(industry_source_date, ''), trade_date)
                    AS DATE)<=try_cast(trade_date AS DATE)
              GROUP BY symbol
            )
            SELECT DISTINCT d.first_date
            FROM daily_first d
            LEFT JOIN valid_sources v USING(symbol)
            WHERE v.symbol IS NULL OR
                  try_cast(v.first_source_date AS DATE)>
                  try_cast(d.first_date AS DATE)
            """,
            [ctx.target_date],
        ).fetchall()
    shutil.rmtree(ctx.runtime / "industry_plan_spill", ignore_errors=True)
    monthly.update(str(item[0]) for item in seed_dates if item[0])
    return sorted(monthly)


def _industry_cninfo_symbols(
    ctx: AuxiliaryContext,
    *,
    fetched_paths: Sequence[Path],
) -> list[str]:
    daily = _scan_sql(_paths(ctx, "market_daily_raw"))
    current = _scan_sql(_paths(ctx, "industry_concept"))
    fetched = _scan_sql(fetched_paths)
    spill = ctx.runtime / "industry_cninfo_plan_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        rows = con.execute(
            f"""
            WITH daily_first AS (
              SELECT symbol, min(trade_date) AS first_trade_date
              FROM {daily}
              WHERE trade_date<=?
              GROUP BY symbol
            ), candidates AS (
              SELECT symbol,
                     coalesce(nullif(industry_source_date, ''), trade_date)
                       AS source_date
              FROM {current}
              WHERE coalesce(trim(industry), '')<>''
                AND coalesce(original_source, '')<>
                    'qdp_v2_industry_initial_bfill'
                AND try_cast(industry_source_date AS DATE)<=try_cast(trade_date AS DATE)
              UNION ALL
              SELECT symbol, source_date
              FROM {fetched}
              WHERE coalesce(trim(industry), '')<>''
                AND try_cast(source_date AS DATE)<=try_cast(snapshot_query_date AS DATE)
            ), first_sources AS (
              SELECT symbol, min(source_date) AS first_source_date
              FROM candidates
              GROUP BY symbol
            )
            SELECT d.symbol
            FROM daily_first d
            LEFT JOIN first_sources s USING(symbol)
            WHERE s.symbol IS NULL OR
                  try_cast(s.first_source_date AS DATE)>
                  try_cast(d.first_trade_date AS DATE)
            ORDER BY d.symbol
            """,
            [ctx.target_date],
        ).fetchall()
    shutil.rmtree(spill, ignore_errors=True)
    return [str(item[0]) for item in rows]


def _fetch_cninfo_industry_parts(
    ctx: AuxiliaryContext,
    symbols: Sequence[str],
) -> list[Path]:
    import akshare as ak

    output_dir = ctx.runtime / "cninfo_industry_parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not symbols:
        path = output_dir / "industry_empty.parquet"
        if not _valid_parquet_columns(path, CNINFO_INDUSTRY_COLUMNS):
            pd.DataFrame(columns=CNINFO_INDUSTRY_COLUMNS).to_parquet(
                path,
                index=False,
                compression="zstd",
            )
        return [path]

    def fetch_one(symbol: str) -> Path:
        path = output_dir / f"industry_{symbol.replace('.', '_')}.parquet"
        if _valid_parquet_columns(path, CNINFO_INDUSTRY_COLUMNS):
            return path
        path.unlink(missing_ok=True)
        code = symbol.split(".", 1)[0]

        def query() -> pd.DataFrame:
            try:
                return ak.stock_industry_change_cninfo(
                    symbol=code,
                    start_date="19900101",
                    end_date=ctx.target_date.replace("-", ""),
                )
            except KeyError:
                return pd.DataFrame()

        raw = _external_with_retry(
            query,
            label=f"industry_history:{symbol}",
        )
        frame = _normalize_cninfo_industry_history(
            raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(),
            symbol=symbol,
            target_date=ctx.target_date,
        )
        if frame.empty:
            frame = pd.DataFrame(columns=CNINFO_INDUSTRY_COLUMNS)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.unlink(missing_ok=True)
        try:
            frame.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    outputs: list[Path] = []
    with ThreadPoolExecutor(max_workers=SECONDARY_VALIDATION_WORKERS) as pool:
        futures = {pool.submit(fetch_one, symbol): symbol for symbol in symbols}
        completed = 0
        for future in as_completed(futures):
            outputs.append(future.result())
            completed += 1
            if completed % 200 == 0:
                print(
                    f"auxiliary_industry_cninfo={completed}/{len(symbols)}",
                    flush=True,
                )
    return sorted(outputs)


def repair_industry(ctx: AuxiliaryContext) -> dict[str, Any]:
    dates = _industry_query_dates(ctx)
    parts = _fetch_baostock_snapshots(ctx, kind="industry", dates=dates)
    cninfo_symbols = _industry_cninfo_symbols(ctx, fetched_paths=parts)
    cninfo_parts = _fetch_cninfo_industry_parts(ctx, cninfo_symbols)
    daily = _scan_sql(_paths(ctx, "market_daily_raw"))
    current = _scan_sql(_paths(ctx, "industry_concept"))
    fetched = _scan_sql(parts)
    cninfo = _scan_sql(cninfo_parts)
    prepared = ctx.runtime / "industry_concept.prepared.parquet"
    sql = f"""
    WITH candidates AS (
      SELECT symbol, industry, coalesce(nullif(original_source, ''), source) AS source,
             coalesce(nullif(industry_source_date, ''), trade_date) AS source_date,
             '证监会行业分类' AS industry_standard, 2 AS priority
      FROM {current}
      WHERE coalesce(trim(industry), '') <> ''
        AND lower(trim(industry)) NOT IN ('unknown', 'unclassified')
        AND coalesce(industry_fill_method, '') <> 'unavailable'
        AND coalesce(original_source, '') <> 'qdp_v2_industry_initial_bfill'
        AND try_cast(coalesce(nullif(industry_source_date, ''), trade_date) AS DATE)
            <= try_cast(trade_date AS DATE)
      UNION ALL
      SELECT symbol, industry, source, source_date, industry_standard, 1 AS priority
      FROM {fetched}
      WHERE coalesce(trim(industry), '') <> ''
        AND try_cast(source_date AS DATE) <= try_cast(snapshot_query_date AS DATE)
      UNION ALL
      SELECT symbol, industry, source, source_date, industry_standard,
             3 AS priority
      FROM {cninfo}
      WHERE coalesce(trim(industry), '')<>''
        AND try_cast(source_date AS DATE)<='{ctx.target_date}'
    ), dedup AS (
      SELECT * EXCLUDE(priority)
      FROM candidates
      QUALIFY row_number() OVER (
        PARTITION BY symbol, source_date ORDER BY priority DESC
      ) = 1
    ), resolved AS (
      SELECT d.symbol, d.trade_date,
             coalesce(c.industry, f.industry, 'Unknown') AS industry,
             coalesce(c.source, f.source, 'pit_history_industry_unavailable')
               AS source,
             coalesce(c.industry, f.original_industry, f.industry, 'Unknown')
               AS original_industry,
             coalesce(c.source, f.original_source, f.source,
                      'pit_history_industry_unavailable') AS original_source,
             CASE
               WHEN c.symbol IS NOT NULL AND c.source_date=d.trade_date
                 THEN 'direct_snapshot'
               WHEN c.symbol IS NOT NULL THEN 'prior_ffill'
               ELSE 'unavailable'
             END AS industry_fill_method,
             coalesce(c.source_date, f.industry_source_date, d.trade_date)
               AS industry_source_date,
             coalesce(nullif(c.industry_standard, ''),
                      nullif(f.industry_standard, ''), 'Unclassified')
               AS industry_standard
      FROM (SELECT * FROM {daily} WHERE trade_date <= '{ctx.target_date}') d
      ASOF LEFT JOIN dedup c
        ON d.symbol=c.symbol AND d.trade_date>=c.source_date
      LEFT JOIN {current} f
        ON d.symbol=f.symbol AND d.trade_date=f.trade_date
    )
    SELECT symbol, trade_date, industry,
           nullif(regexp_extract(industry, '^([A-Z][0-9]{{2}})', 1), '')
             AS industry_code,
           CASE
             WHEN regexp_matches(industry, '^[A-Z][0-9]{{2}}')
               THEN regexp_replace(industry, '^[A-Z][0-9]{{2}}', '')
             ELSE industry
           END AS industry_name,
           CASE
             WHEN lower(industry) IN ('unknown', 'unclassified')
               THEN 'unclassified'
             WHEN regexp_matches(industry, '^[A-Z][0-9]{{2}}')
               THEN 'csrc_coded'
             ELSE 'csrc_uncoded_historical_label'
           END AS industry_taxonomy_version,
           CASE
             WHEN regexp_matches(industry, '^[A-Z][0-9]{{2}}')
               THEN substr(industry, 1, 1)
             ELSE NULL
           END AS industry_section_code,
           source, original_industry, original_source, industry_fill_method,
           industry_source_date, industry_standard
    FROM resolved
    WHERE industry IS NOT NULL
    ORDER BY trade_date, symbol
    """
    _copy_query(ctx, sql=sql, target=prepared)
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "industry_validate_spill",
        threads=2,
    ) as con:
        produced = _scan_sql([prepared])
        missing = int(
            con.execute(
                f"SELECT count(*) FROM ("
                f"SELECT symbol, trade_date FROM {daily} WHERE trade_date <= ? EXCEPT "
                f"SELECT symbol, trade_date FROM {produced})",
                [ctx.target_date],
            ).fetchone()[0]
        )
        invalid = int(
            con.execute(
                f"SELECT count(*) FROM {produced} WHERE "
                "industry IS NULL OR trim(industry)='' OR "
                "industry_name IS NULL OR trim(industry_name)='' OR "
                "industry_taxonomy_version NOT IN ("
                "'unclassified','csrc_coded','csrc_uncoded_historical_label') OR "
                "industry_fill_method NOT IN "
                "('direct_snapshot','prior_ffill','unavailable') OR "
                "try_cast(industry_source_date AS DATE)>try_cast(trade_date AS DATE)"
            ).fetchone()[0]
        )
    shutil.rmtree(ctx.runtime / "industry_validate_spill", ignore_errors=True)
    if missing or invalid:
        raise AuxiliaryUpdateError(
            f"industry_contract_failed:missing={missing}:invalid={invalid}"
        )
    result = _replace_domain(
        ctx,
        domain="industry_concept",
        prepared=prepared,
        primary_key=("trade_date", "symbol"),
        contract_version="qdp_v2_industry_strict_pit_v7",
        source_contract=(
            "BaoStock/CNInfo dated snapshots with past-only asof fill; raw "
            "historical labels are preserved alongside non-retroactive taxonomy fields"
        ),
        validation={
            "secondary_compared_count": 0,
            "secondary_material_mismatch_count": 0,
            "secondary_validation_status": "pending_current_cninfo_sample",
        },
    )
    shutil.rmtree(ctx.runtime / "baostock_industry_parts", ignore_errors=True)
    shutil.rmtree(ctx.runtime / "cninfo_industry_parts", ignore_errors=True)
    return {
        **result,
        "snapshot_date_count": len(dates),
        "cninfo_fallback_symbol_count": len(cninfo_symbols),
    }


def repair_index_constituents(ctx: AuxiliaryContext) -> dict[str, Any]:
    dates = _snapshot_dates(_open_dates(ctx))
    parts = _fetch_baostock_snapshots(ctx, kind="index", dates=dates)
    current = _scan_sql(_paths(ctx, "index_constituents"))
    fetched = _scan_sql(parts)
    calendar = _scan_sql(_paths(ctx, "trading_calendar"))
    identity = _scan_sql(_paths(ctx, "security_identity"))
    prepared = ctx.runtime / "index_constituents.prepared.parquet"
    sql = f"""
    WITH fetched_dates AS (
      SELECT DISTINCT source_snapshot_date FROM {fetched}
    ), raw_snapshots AS (
      SELECT index_symbol, symbol, index_name, source,
             source_snapshot_date
      FROM {current}
      WHERE source_snapshot_date IS NOT NULL
        AND try_cast(source_snapshot_date AS DATE) <= try_cast(trade_date AS DATE)
        AND source_snapshot_date NOT IN (
          SELECT source_snapshot_date FROM fetched_dates
        )
      UNION ALL
      SELECT index_symbol, symbol, index_name, source,
             source_snapshot_date
      FROM {fetched}
      WHERE try_cast(source_snapshot_date AS DATE)
            <= try_cast(snapshot_query_date AS DATE)
    ), members AS (
      SELECT DISTINCT index_symbol, symbol, index_name, source,
             source_snapshot_date
      FROM raw_snapshots
      WHERE index_symbol IN ('000016.SH','000300.SH','000905.SH')
        AND symbol IN (SELECT current_symbol FROM {identity})
    ), snapshot_ranges AS (
      SELECT index_symbol, source_snapshot_date,
             lead(source_snapshot_date) OVER (
               PARTITION BY index_symbol ORDER BY source_snapshot_date
             ) AS next_snapshot_date
      FROM (SELECT DISTINCT index_symbol, source_snapshot_date FROM members)
    ), open_dates AS (
      SELECT DISTINCT trade_date
      FROM {calendar}
      WHERE exchange='SSE' AND is_open
        AND trade_date BETWEEN '2010-01-04' AND '{ctx.target_date}'
    )
    SELECT m.index_symbol, m.symbol, d.trade_date, m.index_name, m.source,
           m.source_snapshot_date
    FROM members m
    JOIN snapshot_ranges r USING(index_symbol, source_snapshot_date)
    JOIN open_dates d
      ON d.trade_date >= m.source_snapshot_date
     AND (r.next_snapshot_date IS NULL OR d.trade_date < r.next_snapshot_date)
    ORDER BY d.trade_date, m.index_symbol, m.symbol
    """
    _copy_query(ctx, sql=sql, target=prepared)
    produced = _scan_sql([prepared])
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "index_validate_spill",
        threads=2,
    ) as con:
        row = con.execute(
            "SELECT count(*) FILTER (WHERE "
            "try_cast(source_snapshot_date AS DATE)>try_cast(trade_date AS DATE)), "
            "min(trade_date), max(trade_date), max(source_snapshot_date), "
            "count(distinct index_symbol) FROM " + produced
        ).fetchone()
        max_gap = int(
            con.execute(
                f"""
                WITH snapshots AS (
                  SELECT DISTINCT index_symbol, source_snapshot_date
                  FROM {produced}
                ), gaps AS (
                  SELECT date_diff(
                           'day',
                           try_cast(source_snapshot_date AS DATE),
                           lead(try_cast(source_snapshot_date AS DATE)) OVER (
                             PARTITION BY index_symbol
                             ORDER BY try_cast(source_snapshot_date AS DATE)
                           )
                         ) AS gap_days
                  FROM snapshots
                )
                SELECT coalesce(max(gap_days), 0) FROM gaps
                """
            ).fetchone()[0]
        )
    shutil.rmtree(ctx.runtime / "index_validate_spill", ignore_errors=True)
    if (
        row is None
        or int(row[0] or 0) != 0
        or str(row[1]) != "2010-01-04"
        or str(row[2]) != ctx.target_date
        or int(row[4] or 0) != 3
        or max_gap > 40
    ):
        raise AuxiliaryUpdateError(f"index_contract_failed:{row}")
    age = (pd.Timestamp(ctx.target_date) - pd.Timestamp(row[3])).days
    if age > 40:
        raise AuxiliaryUpdateError(f"index_snapshot_stale:{age}")
    result = _replace_domain(
        ctx,
        domain="index_constituents",
        prepared=prepared,
        primary_key=("trade_date", "index_symbol", "symbol"),
        contract_version="qdp_v2_index_constituents_strict_pit_v2",
        source_contract="BaoStock first-open/month-end/latest snapshots with past-only expansion",
        validation={
            "secondary_compared_count": 0,
            "secondary_material_mismatch_count": 0,
            "secondary_validation_status": "pending_tushare_latest_snapshot",
        },
    )
    shutil.rmtree(ctx.runtime / "baostock_index_parts", ignore_errors=True)
    return {
        **result,
        "snapshot_date_count": len(dates),
        "latest_snapshot_age_days": age,
        "maximum_snapshot_gap_days": max_gap,
    }


def _fetch_name_change_parts(ctx: AuxiliaryContext) -> list[Path]:
    token = _resolve_tushare_token(ctx.workspace)
    if not token:
        raise AuxiliaryUpdateError("tushare_token_required_for_name_change_repair")
    output_dir = ctx.runtime / "name_change_parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    client = _TushareClient(token, workspace_root=ctx.workspace)
    symbols = _current_symbols(ctx)

    def fetch_one(symbol: str) -> Path:
        path = output_dir / f"namechange_{symbol.replace('.', '_')}.parquet"
        if _valid_parquet_columns(path, NAME_INTERVAL_COLUMNS):
            return path
        path.unlink(missing_ok=True)
        raw = client.fetch(
            "namechange",
            params={"ts_code": symbol},
            fields=NAME_CHANGE_FIELDS,
        )
        frame = _normalize_name_intervals(raw, target_date=ctx.target_date)
        if frame.empty:
            frame = pd.DataFrame(columns=NAME_INTERVAL_COLUMNS)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.unlink(missing_ok=True)
        try:
            frame.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    outputs: list[Path] = []
    with ThreadPoolExecutor(max_workers=TUSHARE_WORKERS) as pool:
        futures = {pool.submit(fetch_one, symbol): symbol for symbol in symbols}
        completed = 0
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                outputs.append(future.result())
            except Exception as exc:
                raise AuxiliaryUpdateError(
                    f"name_change_partition_failed:{symbol}:{type(exc).__name__}:{exc}"
                ) from exc
            completed += 1
            if completed % 200 == 0:
                print(
                    f"auxiliary_name_change={completed}/{len(symbols)}",
                    flush=True,
                )
    return sorted(outputs)


def repair_name_change(ctx: AuxiliaryContext) -> dict[str, Any]:
    parts = _fetch_name_change_parts(ctx)
    intervals = _scan_sql(parts)
    universe = _scan_sql(_paths(ctx, "universe_snapshot"))
    prepared = ctx.runtime / "name_change.prepared.parquet"
    sql = f"""
    WITH interval_transitions AS (
      SELECT symbol, start_date AS trade_date,
             lag(name) OVER (PARTITION BY symbol ORDER BY start_date) AS old_name,
             name AS new_name
      FROM {intervals}
      WHERE start_date<='{ctx.target_date}'
    ), tushare_events AS (
      SELECT symbol, trade_date, old_name, new_name,
             'short_name' AS change_type,
             'tushare_namechange_intervals' AS source
      FROM interval_transitions
      WHERE trade_date>='2010-01-04'
        AND old_name IS NOT NULL AND old_name<>new_name
    ), universe_normalized AS (
      SELECT symbol, trade_date, trim(name) AS name
      FROM {universe}
      WHERE trade_date BETWEEN '2010-01-04' AND '{ctx.target_date}'
        AND coalesce(trim(name), '')<>''
    ), universe_transitions AS (
      SELECT symbol, trade_date,
             lag(name) OVER(PARTITION BY symbol ORDER BY trade_date) AS old_name,
             name AS new_name
      FROM universe_normalized
    ), universe_events AS (
      SELECT symbol, trade_date, old_name, new_name,
             'short_name' AS change_type,
             'qdp_universe_observed_name_transition' AS source
      FROM universe_transitions
      WHERE old_name IS NOT NULL AND old_name<>new_name
    ), latest_interval AS (
      SELECT symbol, name
      FROM {intervals}
      QUALIFY row_number() OVER(
        PARTITION BY symbol ORDER BY start_date DESC
      )=1
    ), latest_universe AS (
      SELECT symbol, trim(name) AS name
      FROM {universe}
      WHERE trade_date='{ctx.target_date}'
    ), conflicts AS (
      SELECT u.symbol
      FROM latest_universe u
      LEFT JOIN latest_interval i USING(symbol)
      WHERE i.symbol IS NULL OR i.name<>u.name
    ), resolved_events AS (
      SELECT * FROM tushare_events
      WHERE symbol NOT IN (SELECT symbol FROM conflicts)
      UNION ALL
      SELECT * FROM universe_events
      WHERE symbol IN (SELECT symbol FROM conflicts)
    )
    SELECT symbol, trade_date, old_name, new_name,
           change_type, source
    FROM resolved_events
    ORDER BY trade_date, symbol
    """
    _copy_query(ctx, sql=sql, target=prepared)
    produced = _scan_sql([prepared])
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "name_validate_spill",
        threads=1,
    ) as con:
        invalid = int(
            con.execute(
                f"SELECT count(*) FROM {produced} WHERE "
                "coalesce(trim(old_name),'')='' OR coalesce(trim(new_name),'')='' "
                "OR old_name=new_name"
            ).fetchone()[0]
        )
        latest_row = con.execute(
            f"""
                WITH latest_interval AS (
                  SELECT symbol, name
                  FROM {intervals}
                  QUALIFY row_number() OVER (
                    PARTITION BY symbol ORDER BY start_date DESC
                  )=1
                ), latest_universe AS (
                  SELECT symbol, trim(name) AS name
                  FROM {universe}
                  WHERE trade_date='{ctx.target_date}'
                ), first_universe AS (
                  SELECT symbol, trim(name) AS name
                  FROM {universe}
                  WHERE trade_date<='{ctx.target_date}'
                  QUALIFY row_number() OVER(
                    PARTITION BY symbol ORDER BY trade_date
                  )=1
                ), latest_event AS (
                  SELECT symbol, new_name
                  FROM {produced}
                  QUALIFY row_number() OVER(
                    PARTITION BY symbol ORDER BY trade_date DESC
                  )=1
                )
                SELECT count(*) FILTER(WHERE i.symbol IS NULL),
                       count(*) FILTER(
                         WHERE coalesce(e.new_name, f.name)<>u.name
                       ),
                       count(*) FILTER(WHERE i.symbol IS NOT NULL),
                       count(*) FILTER(
                         WHERE i.symbol IS NULL OR i.name<>u.name
                       )
                FROM latest_universe u
                LEFT JOIN latest_interval i USING(symbol)
                LEFT JOIN first_universe f USING(symbol)
                LEFT JOIN latest_event e USING(symbol)
                """
        ).fetchone()
        latest_missing = int(latest_row[0] or 0)
        latest_mismatch = int(latest_row[1] or 0)
        compared = int(latest_row[2] or 0)
        source_conflicts = int(latest_row[3] or 0)
    shutil.rmtree(ctx.runtime / "name_validate_spill", ignore_errors=True)
    if invalid or latest_missing or latest_mismatch:
        raise AuxiliaryUpdateError(
            "name_change_contract_failed:"
            f"invalid={invalid}:latest_missing={latest_missing}:"
            f"latest_mismatch={latest_mismatch}"
        )
    result = _replace_domain(
        ctx,
        domain="name_change",
        prepared=prepared,
        primary_key=("symbol", "trade_date", "change_type"),
        contract_version="qdp_v2_name_change_strict_pit_v2",
        source_contract=(
            "Tushare name intervals; QDP daily-universe observed transitions "
            "arbitrate symbols whose latest provider name conflicts"
        ),
        validation={
            "secondary_compared_count": compared,
            "secondary_material_mismatch_count": latest_mismatch,
            "secondary_validation_status": "ok",
        },
    )
    shutil.rmtree(ctx.runtime / "name_change_parts", ignore_errors=True)
    return {
        **result,
        "symbol_request_count": len(parts),
        "tushare_universe_conflict_count": source_conflicts,
    }


LEGACY_CNINFO_SHARE_SOURCE = "akshare_cninfo_historical_share_fallback"
CNINFO_A_SHARE_SOURCE = "akshare_cninfo_historical_a_share_fallback_v2"


def _share_repair_dates(ctx: AuxiliaryContext) -> list[str]:
    daily = _scan_sql(_paths(ctx, "market_daily_raw"))
    share = _scan_sql(_paths(ctx, "share_capital"))
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "share_plan_spill",
        threads=2,
    ) as con:
        rows = con.execute(
            f"""
            SELECT DISTINCT d.trade_date
            FROM {daily} d
            LEFT JOIN {share} s USING(symbol, trade_date)
            WHERE d.trade_date <= ? AND (
              s.symbol IS NULL OR s.total_share IS NULL OR s.float_share IS NULL OR
              try_cast(s.total_share_source_date AS DATE)>try_cast(d.trade_date AS DATE) OR
              try_cast(s.float_share_source_date AS DATE)>try_cast(d.trade_date AS DATE)
            )
            ORDER BY d.trade_date
            """,
            [ctx.target_date],
        ).fetchall()
    shutil.rmtree(ctx.runtime / "share_plan_spill", ignore_errors=True)
    return [str(item[0]) for item in rows]


def _valuation_repair_dates(ctx: AuxiliaryContext) -> list[str]:
    daily = _scan_sql(_paths(ctx, "market_daily_raw"))
    valuation = _scan_sql(_paths(ctx, "valuation"))
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "valuation_plan_spill",
        threads=2,
    ) as con:
        rows = con.execute(
            f"""
            SELECT DISTINCT d.trade_date
            FROM {daily} d
            LEFT JOIN {valuation} v USING(symbol, trade_date)
            WHERE d.trade_date<=? AND (
              v.symbol IS NULL OR v.pe IS NULL OR v.pb IS NULL OR
              v.turnover_rate IS NULL
            )
            ORDER BY d.trade_date
            """,
            [ctx.target_date],
        ).fetchall()
    shutil.rmtree(ctx.runtime / "valuation_plan_spill", ignore_errors=True)
    dates = [str(item[0]) for item in rows]
    return dates if dates else _open_dates(ctx)[-20:]


def _fetch_daily_basic_parts(
    ctx: AuxiliaryContext,
    dates: Sequence[str],
) -> list[Path]:
    token = _resolve_tushare_token(ctx.workspace)
    if not token:
        raise AuxiliaryUpdateError(
            "tushare_token_required_for_historical_share_capital_repair"
        )
    output_dir = ctx.runtime / "daily_basic_parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_dates = sorted(set(str(item) for item in dates))
    client = _TushareClient(token, workspace_root=ctx.workspace)

    def fetch_one(trade_date: str) -> Path:
        path = output_dir / f"daily_basic_{trade_date.replace('-', '')}.parquet"
        if _valid_parquet_columns(path, DAILY_BASIC_NORMALIZED_COLUMNS):
            return path
        path.unlink(missing_ok=True)
        raw = client.fetch(
            "daily_basic",
            params={"trade_date": trade_date.replace("-", "")},
            fields=DAILY_BASIC_FIELDS,
        )
        frame = _normalize_daily_basic(raw, trade_date)
        if frame.empty:
            raise AuxiliaryUpdateError(f"tushare_daily_basic_empty:{trade_date}")
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.unlink(missing_ok=True)
        try:
            frame.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    outputs: list[Path] = []
    with ThreadPoolExecutor(max_workers=TUSHARE_WORKERS) as pool:
        futures = {pool.submit(fetch_one, item): item for item in unique_dates}
        for future in as_completed(futures):
            trade_date = futures[future]
            try:
                outputs.append(future.result())
            except Exception as exc:
                raise AuxiliaryUpdateError(
                    f"daily_basic_partition_failed:{trade_date}:"
                    f"{type(exc).__name__}:{exc}"
                ) from exc
    return sorted(outputs)


def _fetch_cninfo_share_fallback_parts(
    ctx: AuxiliaryContext,
    symbols: Sequence[str],
) -> list[Path]:
    import akshare as ak

    output_dir = ctx.runtime / "cninfo_share_fallback_parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_symbols = sorted(set(str(item).upper() for item in symbols))

    def fetch_one(symbol: str) -> Path:
        path = output_dir / f"share_{symbol.replace('.', '_')}.parquet"
        if _valid_parquet_columns(path, CNINFO_SHARE_NORMALIZED_COLUMNS):
            return path
        path.unlink(missing_ok=True)
        code = symbol.split(".", 1)[0]
        raw = _external_with_retry(
            lambda: ak.stock_share_change_cninfo(
                symbol=code,
                start_date="19900101",
                end_date=ctx.target_date.replace("-", ""),
            ),
            label=f"share_fallback:{symbol}",
        )
        frame = _normalize_cninfo_share_change(
            raw,
            symbol=symbol,
            target_date=ctx.target_date,
            source=CNINFO_A_SHARE_SOURCE,
        )
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.unlink(missing_ok=True)
        try:
            frame.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    outputs: list[Path] = []
    with ThreadPoolExecutor(max_workers=SECONDARY_VALIDATION_WORKERS) as pool:
        futures = {pool.submit(fetch_one, symbol): symbol for symbol in unique_symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                outputs.append(future.result())
            except Exception as exc:
                raise AuxiliaryUpdateError(
                    f"cninfo_share_fallback_failed:{symbol}:{type(exc).__name__}:{exc}"
                ) from exc
    return sorted(outputs)


def _legacy_cninfo_share_symbols(
    ctx: AuxiliaryContext,
    *,
    current: str | None = None,
) -> list[str]:
    source = current or _scan_sql(_paths(ctx, "share_capital"))
    spill = ctx.runtime / "legacy_cninfo_share_inventory_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=1) as con:
        rows = con.execute(
            f"SELECT DISTINCT symbol FROM {source} WHERE source=? ORDER BY symbol",
            [LEGACY_CNINFO_SHARE_SOURCE],
        ).fetchall()
    shutil.rmtree(spill, ignore_errors=True)
    return [str(item[0]) for item in rows]


def _repair_legacy_cninfo_share_rows(
    ctx: AuxiliaryContext,
    *,
    current: str,
    symbols: Sequence[str],
) -> tuple[dict[str, Any], list[Path]]:
    expected_rows = int(_manifest(ctx.root, ctx.datasets, "share_capital").row_count)
    fallback_parts = _fetch_cninfo_share_fallback_parts(ctx, symbols)
    fallback = _scan_sql(fallback_parts)
    prepared = ctx.runtime / "share_capital.prepared.parquet"
    sql = f"""
    WITH fallback AS (
      SELECT symbol, source_date, total_share, float_share, source
      FROM {fallback}
      QUALIFY row_number() OVER (
        PARTITION BY symbol, source_date
        ORDER BY variation_date DESC NULLS LAST
      )=1
    )
    SELECT c.symbol, c.trade_date,
           CASE WHEN c.source='{LEGACY_CNINFO_SHARE_SOURCE}'
                THEN f.total_share ELSE c.total_share END AS total_share,
           CASE WHEN c.source='{LEGACY_CNINFO_SHARE_SOURCE}'
                THEN f.float_share ELSE c.float_share END AS float_share,
           CASE WHEN c.source='{LEGACY_CNINFO_SHARE_SOURCE}'
                THEN greatest(f.total_share-f.float_share,0.0)
                ELSE c.restricted_share END AS restricted_share,
           CASE WHEN c.source='{LEGACY_CNINFO_SHARE_SOURCE}'
                THEN f.source_date ELSE c.total_share_source_date END
                AS total_share_source_date,
           CASE WHEN c.source='{LEGACY_CNINFO_SHARE_SOURCE}'
                THEN f.source_date ELSE c.float_share_source_date END
                AS float_share_source_date,
           CASE WHEN c.source='{LEGACY_CNINFO_SHARE_SOURCE}'
                THEN f.source_date ELSE c.restricted_share_source_date END
                AS restricted_share_source_date,
           CASE WHEN c.source='{LEGACY_CNINFO_SHARE_SOURCE}' THEN
                  CASE WHEN f.source_date=c.trade_date
                       THEN 'direct_daily' ELSE 'prior_ffill' END
                ELSE c.share_fill_method END AS share_fill_method,
           CASE WHEN c.source='{LEGACY_CNINFO_SHARE_SOURCE}'
                THEN f.source ELSE c.source END AS source
    FROM {current} c
    ASOF LEFT JOIN fallback f
      ON c.symbol=f.symbol AND c.trade_date>=f.source_date
    ORDER BY c.trade_date,c.symbol
    """
    _copy_query(ctx, sql=sql, target=prepared)
    produced = _scan_sql([prepared])
    spill = ctx.runtime / "legacy_cninfo_share_validate_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        row = con.execute(
            f"""
            SELECT count(*),
                   count(*) FILTER(WHERE p.source='{LEGACY_CNINFO_SHARE_SOURCE}'),
                   count(*) FILTER(WHERE p.total_share IS NULL OR p.float_share IS NULL OR
                     p.total_share<=0 OR p.float_share<0 OR p.float_share>p.total_share OR
                     try_cast(p.total_share_source_date AS DATE)>
                       try_cast(p.trade_date AS DATE) OR
                     try_cast(p.float_share_source_date AS DATE)>
                       try_cast(p.trade_date AS DATE)),
                   count(*) FILTER(WHERE c.source='{LEGACY_CNINFO_SHARE_SOURCE}' AND
                     abs(c.float_share-p.float_share)>0.5)
            FROM {produced} p
            JOIN {current} c USING(symbol,trade_date)
            """
        ).fetchone()
    shutil.rmtree(spill, ignore_errors=True)
    if int(row[0] or 0) != expected_rows or int(row[1] or 0) or int(row[2] or 0):
        raise AuxiliaryUpdateError(
            "legacy_cninfo_share_migration_failed:"
            f"rows={int(row[0] or 0)}:legacy={int(row[1] or 0)}:"
            f"invalid={int(row[2] or 0)}"
        )
    result = _replace_domain(
        ctx,
        domain="share_capital",
        prepared=prepared,
        primary_key=("trade_date", "symbol"),
        contract_version="qdp_v2_share_capital_strict_pit_v3",
        source_contract=(
            "same-day Tushare gap repair plus past-only CNInfo domestic "
            "A-share fallback and state carry"
        ),
        validation={
            "secondary_compared_count": 0,
            "secondary_material_mismatch_count": 0,
            "secondary_validation_status": "pending_cninfo_share_sample",
        },
    )
    shutil.rmtree(ctx.runtime / "cninfo_share_fallback_parts", ignore_errors=True)
    return (
        {
            **result,
            "legacy_cninfo_symbol_count": len(symbols),
            "legacy_cninfo_changed_float_row_count": int(row[3] or 0),
        },
        [],
    )


def repair_share_capital(
    ctx: AuxiliaryContext,
    *,
    daily_basic_parts: Sequence[Path] | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    current = _scan_sql(_paths(ctx, "share_capital"))
    legacy_symbols = _legacy_cninfo_share_symbols(ctx, current=current)
    if legacy_symbols:
        return _repair_legacy_cninfo_share_rows(
            ctx,
            current=current,
            symbols=legacy_symbols,
        )
    dates = _share_repair_dates(ctx)
    parts = list(daily_basic_parts or ())
    required_paths = {
        ctx.runtime
        / "daily_basic_parts"
        / f"daily_basic_{item.replace('-', '')}.parquet"
        for item in dates
    }
    if not required_paths.issubset(set(parts)):
        parts = _fetch_daily_basic_parts(ctx, dates)
    daily = _scan_sql(_paths(ctx, "market_daily_raw"))
    fetched = _scan_sql(parts)
    prepared = ctx.runtime / "share_capital.prepared.parquet"
    base_candidates = f"""
      SELECT symbol, trade_date AS source_date, total_share, float_share,
             source, 2 AS priority
      FROM {fetched}
      WHERE total_share>0 AND float_share>=0 AND float_share<=total_share
      UNION ALL
      SELECT symbol,
             cast(greatest(
               try_cast(total_share_source_date AS DATE),
               try_cast(float_share_source_date AS DATE)
             ) AS VARCHAR) AS source_date,
             total_share, float_share, source, 1 AS priority
      FROM {current}
      WHERE total_share>0 AND float_share>=0 AND float_share<=total_share
        AND try_cast(total_share_source_date AS DATE)<=try_cast(trade_date AS DATE)
        AND try_cast(float_share_source_date AS DATE)<=try_cast(trade_date AS DATE)
    """

    def candidate_ctes(fallback: str = "") -> str:
        fallback_union = (
            f"""
            UNION ALL
            SELECT symbol, source_date, total_share, float_share,
                   source, 3 AS priority
            FROM {fallback}
            WHERE total_share>0 AND float_share>=0 AND float_share<=total_share
              AND try_cast(source_date AS DATE)<=DATE '{ctx.target_date}'
            """
            if fallback
            else ""
        )
        return f"""
    candidates AS (
      {base_candidates}
      {fallback_union}
    ), dedup AS (
      SELECT symbol, source_date, total_share, float_share, source
      FROM candidates
      QUALIFY row_number() OVER (
        PARTITION BY symbol, source_date ORDER BY priority DESC
      )=1
    )
        """

    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "share_fallback_plan_spill",
        threads=2,
    ) as con:
        unresolved = con.execute(
            f"""
            WITH {candidate_ctes()}, resolved AS (
              SELECT d.symbol, d.trade_date, c.total_share, c.float_share
              FROM (SELECT * FROM {daily} WHERE trade_date<=?) d
              ASOF LEFT JOIN dedup c
                ON d.symbol=c.symbol AND d.trade_date>=c.source_date
            )
            SELECT symbol, min(trade_date) AS first_missing_date,
                   count(*) AS missing_key_count
            FROM resolved
            WHERE total_share IS NULL OR float_share IS NULL
            GROUP BY symbol
            ORDER BY symbol
            """,
            [ctx.target_date],
        ).fetchdf()
    shutil.rmtree(ctx.runtime / "share_fallback_plan_spill", ignore_errors=True)
    fallback_parts = (
        _fetch_cninfo_share_fallback_parts(
            ctx,
            unresolved["symbol"].astype(str).tolist(),
        )
        if not unresolved.empty
        else []
    )
    fallback = _scan_sql(fallback_parts) if fallback_parts else ""
    sql = f"""
    WITH {candidate_ctes(fallback)}, resolved AS (
      SELECT d.symbol, d.trade_date, c.total_share, c.float_share,
             greatest(c.total_share-c.float_share, 0.0) AS restricted_share,
             c.source_date AS total_share_source_date,
             c.source_date AS float_share_source_date,
             c.source_date AS restricted_share_source_date,
             CASE WHEN c.source_date=d.trade_date THEN 'direct_daily'
                  ELSE 'prior_ffill' END AS share_fill_method,
             c.source
      FROM (SELECT * FROM {daily} WHERE trade_date<='{ctx.target_date}') d
      ASOF LEFT JOIN dedup c
        ON d.symbol=c.symbol AND d.trade_date>=c.source_date
    )
    SELECT * FROM resolved
    WHERE total_share IS NOT NULL AND float_share IS NOT NULL
    ORDER BY trade_date, symbol
    """
    _copy_query(ctx, sql=sql, target=prepared)
    produced = _scan_sql([prepared])
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "share_validate_spill",
        threads=2,
    ) as con:
        missing = int(
            con.execute(
                f"SELECT count(*) FROM (SELECT symbol,trade_date FROM {daily} "
                "WHERE trade_date<=? EXCEPT SELECT symbol,trade_date FROM "
                + produced
                + ")",
                [ctx.target_date],
            ).fetchone()[0]
        )
        invalid = int(
            con.execute(
                f"SELECT count(*) FROM {produced} WHERE total_share<=0 OR "
                "float_share<0 OR float_share>total_share OR restricted_share<0 OR "
                "try_cast(total_share_source_date AS DATE)>try_cast(trade_date AS DATE) OR "
                "try_cast(float_share_source_date AS DATE)>try_cast(trade_date AS DATE) OR "
                "try_cast(restricted_share_source_date AS DATE)>try_cast(trade_date AS DATE)"
            ).fetchone()[0]
        )
        compared = int(
            con.execute(
                f"SELECT count(*) FROM {current} c JOIN {fetched} t "
                "USING(symbol,trade_date) WHERE c.total_share IS NOT NULL AND t.total_share IS NOT NULL"
            ).fetchone()[0]
        )
        mismatches = int(
            con.execute(
                f"SELECT count(*) FROM {current} c JOIN {fetched} t "
                "USING(symbol,trade_date) WHERE c.total_share IS NOT NULL AND t.total_share IS NOT NULL "
                "AND abs(c.total_share/t.total_share-1)>0.0001"
            ).fetchone()[0]
        )
    shutil.rmtree(ctx.runtime / "share_validate_spill", ignore_errors=True)
    if missing or invalid:
        raise AuxiliaryUpdateError(
            f"share_capital_contract_failed:missing={missing}:invalid={invalid}"
        )
    result = _replace_domain(
        ctx,
        domain="share_capital",
        prepared=prepared,
        primary_key=("trade_date", "symbol"),
        contract_version="qdp_v2_share_capital_strict_pit_v3",
        source_contract=(
            "same-day Tushare gap repair plus past-only CNInfo official "
            "fallback and state carry"
        ),
        validation={
            "secondary_compared_count": 0,
            "secondary_material_mismatch_count": 0,
            "secondary_validation_status": "pending_cninfo_share_sample",
        },
    )
    fallback_candidate_count = 0
    if fallback_parts:
        with open_guarded_duckdb(
            temp_directory=ctx.runtime / "share_fallback_count_spill",
            threads=1,
        ) as con:
            fallback_candidate_count = int(
                con.execute(f"SELECT count(*) FROM {fallback}").fetchone()[0]
            )
        shutil.rmtree(ctx.runtime / "share_fallback_count_spill", ignore_errors=True)
    shutil.rmtree(ctx.runtime / "cninfo_share_fallback_parts", ignore_errors=True)
    return (
        {
            **result,
            "repair_date_count": len(dates),
            "replaced_overlap_compared_count": compared,
            "replaced_overlap_difference_count": mismatches,
            "cninfo_fallback_symbol_count": len(unresolved),
            "cninfo_fallback_candidate_count": fallback_candidate_count,
        },
        parts,
    )


def _valuation_secondary_policy(
    *,
    comparison_counts: Mapping[str, int],
    mismatch_counts: Mapping[str, int],
    median_ratios: Mapping[str, float | None],
) -> dict[str, Any]:
    comparable_metrics = ("turnover_rate",)
    not_comparable_metrics = ("pe", "pb")
    mismatch_rates = {
        metric: (
            int(mismatch_counts.get(metric, 0)) / int(comparison_counts.get(metric, 0))
            if int(comparison_counts.get(metric, 0)) > 0
            else 1.0
        )
        for metric in ("pe", "pb", "turnover_rate")
    }
    suspicious_factors = (10.0, 100.0, 10_000.0, 0.1, 0.01, 0.0001)
    unit_errors = [
        metric
        for metric, ratio in median_ratios.items()
        if ratio is not None
        and int(comparison_counts.get(metric, 0)) >= 50
        and any(
            abs(float(ratio) / factor - 1.0) <= 0.05 for factor in suspicious_factors
        )
    ]
    central_ratio_errors = [
        metric
        for metric, ratio in median_ratios.items()
        if ratio is not None
        and int(comparison_counts.get(metric, 0)) >= 50
        and abs(float(ratio) - 1.0) > 0.05
    ]
    comparable_failures = [
        metric
        for metric in comparable_metrics
        if int(comparison_counts.get(metric, 0)) <= 0 or mismatch_rates[metric] > 0.01
    ]
    return {
        "secondary_validation_status": "not_comparable",
        "secondary_comparable_metrics": list(comparable_metrics),
        "secondary_not_comparable_metrics": list(not_comparable_metrics),
        "secondary_not_comparable_reason": (
            "BaoStock and Tushare historical PE/PB use different financial "
            "revision and effective-date policies; compare central scale only"
        ),
        "secondary_metric_comparison_counts": {
            key: int(value) for key, value in comparison_counts.items()
        },
        "secondary_metric_raw_mismatch_counts": {
            key: int(value) for key, value in mismatch_counts.items()
        },
        "secondary_metric_raw_mismatch_rates": mismatch_rates,
        "unit_errors": unit_errors,
        "central_ratio_errors": central_ratio_errors,
        "comparable_failures": comparable_failures,
    }


def _valuation_market_cap_error_count(ctx: AuxiliaryContext) -> int:
    daily = _scan_sql(_paths(ctx, "market_daily_raw"))
    share = _scan_sql(_paths(ctx, "share_capital"))
    valuation = _scan_sql(_paths(ctx, "valuation"))
    spill = ctx.runtime / "valuation_market_cap_inventory_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        count = int(
            con.execute(
                f"""
                SELECT count(*)
                FROM {valuation} v
                JOIN {daily} d USING(symbol,trade_date)
                JOIN {share} s USING(symbol,trade_date)
                WHERE v.total_mv IS NULL OR v.circ_mv IS NULL OR
                  abs(v.total_mv/nullif(d.close*s.total_share,0)-1)>1e-8 OR
                  abs(v.circ_mv/nullif(d.close*s.float_share,0)-1)>1e-8
                """
            ).fetchone()[0]
        )
    shutil.rmtree(spill, ignore_errors=True)
    return count


def _repair_valuation_market_caps(
    ctx: AuxiliaryContext,
    *,
    error_count: int,
) -> dict[str, Any]:
    daily = _scan_sql(_paths(ctx, "market_daily_raw"))
    share = _scan_sql(_paths(ctx, "share_capital"))
    current = _scan_sql(_paths(ctx, "valuation"))
    prepared = ctx.runtime / "valuation.prepared.parquet"
    sql = f"""
    SELECT v.symbol,v.trade_date,
           d.close*s.total_share AS total_mv,
           d.close*s.float_share AS circ_mv,
           v.pe,v.pb,v.turnover_rate,v.source
    FROM {current} v
    JOIN {daily} d USING(symbol,trade_date)
    JOIN {share} s USING(symbol,trade_date)
    WHERE v.trade_date<='{ctx.target_date}'
    ORDER BY v.trade_date,v.symbol
    """
    _copy_query(ctx, sql=sql, target=prepared)
    produced = _scan_sql([prepared])
    manifest = _manifest(ctx.root, ctx.datasets, "valuation")
    spill = ctx.runtime / "valuation_market_cap_validate_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        row = con.execute(
            f"""
            SELECT count(*),count(*) FILTER(WHERE total_mv IS NULL OR circ_mv IS NULL),
                   count(*) FILTER(WHERE
                     abs(v.total_mv/nullif(d.close*s.total_share,0)-1)>1e-8 OR
                     abs(v.circ_mv/nullif(d.close*s.float_share,0)-1)>1e-8)
            FROM {produced} v
            JOIN {daily} d USING(symbol,trade_date)
            JOIN {share} s USING(symbol,trade_date)
            """
        ).fetchone()
    shutil.rmtree(spill, ignore_errors=True)
    if (
        int(row[0] or 0) != int(manifest.row_count)
        or int(row[1] or 0)
        or int(row[2] or 0)
    ):
        raise AuxiliaryUpdateError(
            "valuation_market_cap_refresh_failed:"
            f"rows={int(row[0] or 0)}:nulls={int(row[1] or 0)}:"
            f"formula_errors={int(row[2] or 0)}"
        )
    source = dict(manifest.source or {})
    result = _replace_domain(
        ctx,
        domain="valuation",
        prepared=prepared,
        primary_key=("trade_date", "symbol"),
        contract_version="qdp_v2_valuation_strict_pit_v3",
        source_contract=(
            "same-day BaoStock PE/PB/turnover; Tushare turnover is comparable "
            "while historical PE/PB revision timing is not row-comparable; "
            "market caps are close times strict-PIT shares in yuan"
        ),
        validation={
            "secondary_compared_count": int(
                source.get("secondary_compared_count", 0) or 0
            ),
            "secondary_material_mismatch_count": int(
                source.get("secondary_material_mismatch_count", 0) or 0
            ),
            "secondary_validation_status": str(
                source.get("secondary_validation_status", "not_comparable")
            ),
        },
    )
    return {
        **result,
        "market_cap_formula_rows_refreshed": int(error_count),
    }


def repair_valuation(
    ctx: AuxiliaryContext,
    *,
    daily_basic_parts: Sequence[Path] = (),
) -> dict[str, Any]:
    manifest = _manifest(ctx.root, ctx.datasets, "valuation")
    source = dict(manifest.source or {})
    if (
        str(source.get("checked_through", "")) == ctx.target_date
        and "strict" in str(source.get("source_contract", "")).lower()
    ):
        market_cap_errors = _valuation_market_cap_error_count(ctx)
        if market_cap_errors:
            return _repair_valuation_market_caps(
                ctx,
                error_count=market_cap_errors,
            )
    dates = _valuation_repair_dates(ctx)
    required_paths = {
        ctx.runtime
        / "daily_basic_parts"
        / f"daily_basic_{item.replace('-', '')}.parquet"
        for item in dates
    }
    parts = list(daily_basic_parts)
    if not required_paths.issubset(set(parts)):
        parts = _fetch_daily_basic_parts(ctx, dates)
    if not parts:
        raise AuxiliaryUpdateError("valuation_daily_basic_gap_parts_missing")
    daily = _scan_sql(_paths(ctx, "market_daily_raw"))
    share = _scan_sql(_paths(ctx, "share_capital"))
    current = _scan_sql(_paths(ctx, "valuation"))
    fetched = _scan_sql(parts)
    prepared = ctx.runtime / "valuation.prepared.parquet"
    sql = f"""
    SELECT d.symbol, d.trade_date,
           d.close*s.total_share AS total_mv,
           d.close*s.float_share AS circ_mv,
           coalesce(v.pe, t.pe_ttm) AS pe,
           coalesce(v.pb, t.pb) AS pb,
           coalesce(v.turnover_rate, t.turnover_rate) AS turnover_rate,
           CASE WHEN v.symbol IS NOT NULL THEN 'baostock_plus_qdp_market_cap_formula'
                ELSE 'tushare_daily_basic_gap_plus_qdp_market_cap_formula' END AS source
    FROM (SELECT * FROM {daily} WHERE trade_date<='{ctx.target_date}') d
    JOIN {share} s USING(symbol, trade_date)
    LEFT JOIN {current} v USING(symbol, trade_date)
    LEFT JOIN {fetched} t USING(symbol, trade_date)
    ORDER BY d.trade_date, d.symbol
    """
    _copy_query(ctx, sql=sql, target=prepared)
    produced = _scan_sql([prepared])
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "valuation_validate_spill",
        threads=2,
    ) as con:
        missing = int(
            con.execute(
                f"SELECT count(*) FROM (SELECT symbol,trade_date FROM {daily} "
                "WHERE trade_date<=? EXCEPT SELECT symbol,trade_date FROM "
                + produced
                + ")",
                [ctx.target_date],
            ).fetchone()[0]
        )
        formula_errors = int(
            con.execute(
                f"SELECT count(*) FROM {produced} v JOIN {daily} d USING(symbol,trade_date) "
                f"JOIN {share} s USING(symbol,trade_date) WHERE "
                "abs(v.total_mv/nullif(d.close*s.total_share,0)-1)>1e-8 OR "
                "abs(v.circ_mv/nullif(d.close*s.float_share,0)-1)>1e-8"
            ).fetchone()[0]
        )
        comparison = con.execute(
            f"""
            WITH pairs AS (
              SELECT c.pe, c.pb, c.turnover_rate,
                     t.pe_ttm, t.pb AS tushare_pb,
                     t.turnover_rate AS tushare_turnover
              FROM {current} c
              JOIN {fetched} t USING(symbol,trade_date)
            )
            SELECT
              count(*) FILTER(WHERE
                (pe IS NOT NULL AND pe_ttm IS NOT NULL) OR
                (pb IS NOT NULL AND tushare_pb IS NOT NULL) OR
                (turnover_rate IS NOT NULL AND tushare_turnover IS NOT NULL)),
              count(*) FILTER(WHERE
                (pe IS NOT NULL AND pe_ttm IS NOT NULL AND
                 abs(pe-pe_ttm)>greatest(abs(pe_ttm)*0.05,0.05)) OR
                (pb IS NOT NULL AND tushare_pb IS NOT NULL AND
                 abs(pb-tushare_pb)>greatest(abs(tushare_pb)*0.05,0.05)) OR
                (turnover_rate IS NOT NULL AND tushare_turnover IS NOT NULL AND
                 abs(turnover_rate-tushare_turnover)>
                   greatest(abs(tushare_turnover)*0.05,0.05))),
              count(*) FILTER(WHERE pe IS NOT NULL AND pe_ttm IS NOT NULL),
              count(*) FILTER(WHERE pb IS NOT NULL AND tushare_pb IS NOT NULL),
              count(*) FILTER(WHERE turnover_rate IS NOT NULL AND tushare_turnover IS NOT NULL),
              count(*) FILTER(WHERE pe IS NOT NULL AND pe_ttm IS NOT NULL AND
                abs(pe-pe_ttm)>greatest(abs(pe_ttm)*0.05,0.05)),
              count(*) FILTER(WHERE pb IS NOT NULL AND tushare_pb IS NOT NULL AND
                abs(pb-tushare_pb)>greatest(abs(tushare_pb)*0.05,0.05)),
              count(*) FILTER(WHERE turnover_rate IS NOT NULL AND tushare_turnover IS NOT NULL AND
                abs(turnover_rate-tushare_turnover)>
                  greatest(abs(tushare_turnover)*0.05,0.05)),
              median(abs(pe/pe_ttm)) FILTER(
                WHERE pe IS NOT NULL AND pe_ttm IS NOT NULL AND abs(pe_ttm)>1e-12),
              median(abs(pb/tushare_pb)) FILTER(
                WHERE pb IS NOT NULL AND tushare_pb IS NOT NULL AND abs(tushare_pb)>1e-12),
              median(abs(turnover_rate/tushare_turnover)) FILTER(
                WHERE turnover_rate IS NOT NULL AND tushare_turnover IS NOT NULL
                  AND abs(tushare_turnover)>1e-12)
            FROM pairs
            """
        ).fetchone()
        nulls = con.execute(
            "SELECT count(*) FILTER(WHERE pe IS NULL), "
            "count(*) FILTER(WHERE pb IS NULL), "
            "count(*) FILTER(WHERE turnover_rate IS NULL) FROM " + produced
        ).fetchone()
    shutil.rmtree(ctx.runtime / "valuation_validate_spill", ignore_errors=True)
    compared = int(comparison[0] or 0)
    mismatches = int(comparison[1] or 0)
    comparison_counts = {
        "pe": int(comparison[2] or 0),
        "pb": int(comparison[3] or 0),
        "turnover_rate": int(comparison[4] or 0),
    }
    mismatch_counts = {
        "pe": int(comparison[5] or 0),
        "pb": int(comparison[6] or 0),
        "turnover_rate": int(comparison[7] or 0),
    }
    median_ratios = {
        "pe": float(comparison[8]) if comparison[8] is not None else None,
        "pb": float(comparison[9]) if comparison[9] is not None else None,
        "turnover_rate": float(comparison[10]) if comparison[10] is not None else None,
    }
    mismatch_rate = (mismatches / compared) if compared else 1.0
    policy = _valuation_secondary_policy(
        comparison_counts=comparison_counts,
        mismatch_counts=mismatch_counts,
        median_ratios=median_ratios,
    )
    if (
        missing
        or formula_errors
        or compared <= 0
        or policy["unit_errors"]
        or policy["central_ratio_errors"]
        or policy["comparable_failures"]
    ):
        raise AuxiliaryUpdateError(
            "valuation_contract_failed:"
            f"missing={missing}:formula_errors={formula_errors}:"
            f"compared={compared}:mismatch_rate={mismatch_rate:.8f}:"
            f"unit_errors={','.join(policy['unit_errors'])}:"
            f"central_ratio_errors={','.join(policy['central_ratio_errors'])}:"
            f"comparable_failures={','.join(policy['comparable_failures'])}"
        )
    result = _replace_domain(
        ctx,
        domain="valuation",
        prepared=prepared,
        primary_key=("trade_date", "symbol"),
        contract_version="qdp_v2_valuation_strict_pit_v3",
        source_contract=(
            "same-day BaoStock PE/PB/turnover; Tushare turnover is comparable "
            "while historical PE/PB revision timing is not row-comparable; "
            "market caps are close times strict-PIT shares in yuan"
        ),
        validation={
            "secondary_compared_count": compared,
            "secondary_material_mismatch_count": 0,
            "secondary_raw_mismatch_count": mismatches,
            "secondary_raw_mismatch_rate": mismatch_rate,
            **{
                key: value
                for key, value in policy.items()
                if str(key).startswith("secondary_")
            },
        },
    )
    return {
        **result,
        "provider_null_counts": {
            "pe": int(nulls[0] or 0),
            "pb": int(nulls[1] or 0),
            "turnover_rate": int(nulls[2] or 0),
        },
        "diagnostic_metric_difference_count": mismatches,
        "diagnostic_metric_difference_rate": mismatch_rate,
        "secondary_comparison_counts": comparison_counts,
        "secondary_metric_raw_mismatch_counts": mismatch_counts,
        "secondary_metric_raw_mismatch_rates": policy[
            "secondary_metric_raw_mismatch_rates"
        ],
        "secondary_median_ratios": median_ratios,
    }


def _current_symbols(ctx: AuxiliaryContext) -> list[str]:
    identity = _scan_sql(_paths(ctx, "security_identity"))
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "symbols_spill",
        threads=1,
    ) as con:
        rows = con.execute(
            f"SELECT current_symbol FROM {identity} ORDER BY current_symbol"
        ).fetchall()
    shutil.rmtree(ctx.runtime / "symbols_spill", ignore_errors=True)
    return [str(item[0]) for item in rows]


def _fetch_dividend_parts(ctx: AuxiliaryContext) -> list[Path]:
    token = _resolve_tushare_token(ctx.workspace)
    if not token:
        raise AuxiliaryUpdateError("tushare_token_required_for_corporate_action_repair")
    output_dir = ctx.runtime / "dividend_parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    client = _TushareClient(token, workspace_root=ctx.workspace)
    symbols = _current_symbols(ctx)

    def fetch_one(symbol: str) -> Path:
        path = output_dir / f"dividend_{symbol.replace('.', '_')}.parquet"
        if _valid_parquet_columns(path, CORPORATE_COLUMNS):
            return path
        path.unlink(missing_ok=True)
        raw = client.fetch(
            "dividend",
            params={"ts_code": symbol},
            fields=DIVIDEND_FIELDS,
        )
        frame = _normalize_dividend(raw, target_date=ctx.target_date)
        if frame.empty:
            frame = pd.DataFrame(columns=CORPORATE_COLUMNS)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.unlink(missing_ok=True)
        try:
            frame.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    outputs: list[Path] = []
    with ThreadPoolExecutor(max_workers=TUSHARE_WORKERS) as pool:
        futures = {pool.submit(fetch_one, symbol): symbol for symbol in symbols}
        completed = 0
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                outputs.append(future.result())
            except Exception as exc:
                raise AuxiliaryUpdateError(
                    f"dividend_partition_failed:{symbol}:{type(exc).__name__}:{exc}"
                ) from exc
            completed += 1
            if completed % 200 == 0:
                print(
                    f"auxiliary_corporate_actions={completed}/{len(symbols)}",
                    flush=True,
                )
    return sorted(outputs)


def _mootdx_corporate_validation_worker(
    symbols: Sequence[str],
    output_dir: str,
    target_date: str,
) -> dict[str, Any]:
    from quantlab.data.domains.contracts import DataDomain, DomainFetchRequest
    from quantlab.data.providers import MootdxOnlineProvider

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    os.environ["QDP_MOOTDX_LAST_GOOD_PATH"] = str(
        destination / f"mootdx_last_good_{os.getpid()}.json"
    )
    failures: list[str] = []
    completed = 0
    provider = MootdxOnlineProvider()
    try:
        for symbol in symbols:
            path = destination / f"mootdx_{symbol.replace('.', '_')}.parquet"
            if _valid_parquet_columns(path, MOOTDX_CORPORATE_VALIDATION_COLUMNS):
                completed += 1
                continue
            path.unlink(missing_ok=True)
            result = provider.fetch_domain(
                DomainFetchRequest(
                    domain=DataDomain.CORPORATE_ACTIONS,
                    symbols=(symbol,),
                    start_date="2010-01-04",
                    end_date=target_date,
                )
            )
            if result.error_report:
                failures.append(symbol)
                provider.close()
                provider = MootdxOnlineProvider()
                continue
            frame = result.data.copy()
            if frame.empty:
                frame = pd.DataFrame(columns=MOOTDX_CORPORATE_VALIDATION_COLUMNS)
            else:
                frame["cash_dividend_per_10"] = pd.to_numeric(
                    frame["cash_dividend_per_10"], errors="coerce"
                )
                frame["source"] = "mootdx_xdxr_validation"
                frame = (
                    frame.loc[:, list(MOOTDX_CORPORATE_VALIDATION_COLUMNS)]
                    .drop_duplicates(["symbol", "trade_date"], keep="last")
                    .sort_values(["trade_date", "symbol"])
                    .reset_index(drop=True)
                )
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temporary.unlink(missing_ok=True)
            try:
                frame.to_parquet(temporary, index=False, compression="zstd")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            completed += 1
    finally:
        provider.close()
        Path(os.environ["QDP_MOOTDX_LAST_GOOD_PATH"]).unlink(missing_ok=True)
    return {"completed": completed, "failures": failures}


def _fetch_mootdx_corporate_validation_parts(
    ctx: AuxiliaryContext,
) -> list[Path]:
    symbols = _current_symbols(ctx)
    output_dir = ctx.runtime / "mootdx_corporate_validation_parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / f"mootdx_{symbol.replace('.', '_')}.parquet" for symbol in symbols
    ]
    pending = [
        symbol
        for symbol, path in zip(symbols, outputs, strict=True)
        if not _valid_parquet_columns(path, MOOTDX_CORPORATE_VALIDATION_COLUMNS)
    ]
    chunks = _split_evenly(pending, BAOSTOCK_WORKERS)
    failures: list[str] = []
    if chunks:
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            futures = {
                pool.submit(
                    _mootdx_corporate_validation_worker,
                    chunk,
                    str(output_dir),
                    ctx.target_date,
                ): chunk
                for chunk in chunks
            }
            for future in as_completed(futures):
                result = future.result()
                failures.extend(list(result.get("failures", []) or []))
    if failures:
        raise AuxiliaryUpdateError(
            f"mootdx_corporate_validation_failures:{len(failures)}:"
            + ",".join(failures[:10])
        )
    incomplete = [
        symbol
        for symbol, path in zip(symbols, outputs, strict=True)
        if not _valid_parquet_columns(path, MOOTDX_CORPORATE_VALIDATION_COLUMNS)
    ]
    if incomplete:
        raise AuxiliaryUpdateError(
            f"mootdx_corporate_validation_incomplete:{len(incomplete)}:"
            + ",".join(incomplete[:10])
        )
    return outputs


def repair_corporate_actions(ctx: AuxiliaryContext) -> dict[str, Any]:
    parts = _fetch_dividend_parts(ctx)
    fetched = _scan_sql(parts)
    current = _scan_sql(_paths(ctx, "corporate_actions"))
    prepared = ctx.runtime / "corporate_actions.prepared.parquet"
    sql = f"""
    SELECT symbol, trade_date, announcement_date, ex_date, record_date,
           dividend_pay_date, action_type, cash_dividend_per_10,
           bonus_share_per_10, transfer_share_per_10, description, source
    FROM {fetched}
    WHERE trade_date=ex_date
      AND try_cast(announcement_date AS DATE)<=try_cast(ex_date AS DATE)
      AND (record_date IS NULL OR try_cast(record_date AS DATE)<=try_cast(ex_date AS DATE))
      AND (dividend_pay_date IS NULL OR try_cast(dividend_pay_date AS DATE)>=try_cast(ex_date AS DATE))
      AND coalesce(cash_dividend_per_10,0)+coalesce(bonus_share_per_10,0)
          +coalesce(transfer_share_per_10,0)>0
    QUALIFY row_number() OVER (
      PARTITION BY symbol, trade_date, action_type
      ORDER BY announcement_date DESC
    )=1
    ORDER BY trade_date, symbol, action_type
    """
    _copy_query(ctx, sql=sql, target=prepared)
    produced = _scan_sql([prepared])
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "corporate_validate_spill",
        threads=2,
    ) as con:
        invalid = int(
            con.execute(
                f"SELECT count(*) FROM {produced} WHERE ex_date IS NULL OR "
                "trade_date<>ex_date OR announcement_date IS NULL OR "
                "try_cast(announcement_date AS DATE)>try_cast(ex_date AS DATE) OR "
                "(record_date IS NOT NULL AND try_cast(record_date AS DATE)>try_cast(ex_date AS DATE)) OR "
                "(dividend_pay_date IS NOT NULL AND try_cast(dividend_pay_date AS DATE)<try_cast(ex_date AS DATE)) OR "
                "coalesce(cash_dividend_per_10,0)+coalesce(bonus_share_per_10,0)+"
                "coalesce(transfer_share_per_10,0)<=0"
            ).fetchone()[0]
        )
        compared = int(
            con.execute(
                f"SELECT count(*) FROM {current} c JOIN {produced} n "
                "ON c.symbol=n.symbol AND c.record_date=n.record_date "
                "AND c.dividend_pay_date=n.dividend_pay_date"
            ).fetchone()[0]
        )
    shutil.rmtree(ctx.runtime / "corporate_validate_spill", ignore_errors=True)
    if invalid:
        raise AuxiliaryUpdateError(
            f"corporate_actions_contract_failed:invalid={invalid}"
        )
    result = _replace_domain(
        ctx,
        domain="corporate_actions",
        prepared=prepared,
        primary_key=("symbol", "trade_date", "action_type"),
        contract_version="qdp_v2_corporate_actions_strict_pit_v2",
        source_contract="implemented Tushare dividend events; trade_date equals ex_date",
        validation={
            "secondary_compared_count": 0,
            "secondary_material_mismatch_count": 0,
            "secondary_validation_status": "pending_cninfo_full_event_compare",
        },
    )
    shutil.rmtree(ctx.runtime / "dividend_parts", ignore_errors=True)
    return {
        **result,
        "symbol_request_count": len(parts),
        "legacy_overlap_count": compared,
    }


def validate_corporate_actions_secondary(
    ctx: AuxiliaryContext,
) -> dict[str, Any]:
    import akshare as ak

    output_dir = ctx.runtime / "cninfo_dividend_validation_parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    symbols = _current_symbols(ctx)

    def fetch_one(symbol: str) -> Path:
        path = output_dir / f"cninfo_dividend_{symbol.replace('.', '_')}.parquet"
        if _valid_parquet_columns(path, CORPORATE_COLUMNS):
            return path
        path.unlink(missing_ok=True)
        code = symbol.split(".", 1)[0]

        def query_cninfo() -> pd.DataFrame:
            try:
                return ak.stock_dividend_cninfo(symbol=code)
            except KeyError:
                return pd.DataFrame()

        raw = _external_with_retry(
            query_cninfo,
            label=f"corporate_actions:{symbol}",
        )
        frame = _normalize_cninfo_dividend(
            raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(),
            symbol=symbol,
            target_date=ctx.target_date,
        )
        if frame.empty:
            frame = pd.DataFrame(columns=CORPORATE_COLUMNS)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.unlink(missing_ok=True)
        try:
            frame.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    parts: list[Path] = []
    with ThreadPoolExecutor(max_workers=SECONDARY_VALIDATION_WORKERS) as pool:
        futures = {pool.submit(fetch_one, symbol): symbol for symbol in symbols}
        completed = 0
        for future in as_completed(futures):
            parts.append(future.result())
            completed += 1
            if completed % 200 == 0:
                print(
                    f"auxiliary_corporate_validation={completed}/{len(symbols)}",
                    flush=True,
                )
    mootdx_parts = _fetch_mootdx_corporate_validation_parts(ctx)
    primary = _scan_sql(_paths(ctx, "corporate_actions"))
    secondary = _scan_sql(sorted(parts))
    mootdx = _scan_sql(mootdx_parts)
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "corporate_secondary_spill",
        threads=2,
    ) as con:
        primary_frame = con.execute(
            f"SELECT symbol, trade_date, action_type, "
            "cash_dividend_per_10, bonus_share_per_10, transfer_share_per_10 "
            f"FROM {primary}"
        ).fetchdf()
        secondary_frame = con.execute(
            f"SELECT symbol, trade_date, action_type, "
            "cash_dividend_per_10, bonus_share_per_10, transfer_share_per_10 "
            f"FROM {secondary}"
        ).fetchdf()
        mootdx_frame = con.execute(
            f"SELECT symbol, trade_date, cash_dividend_per_10 FROM {mootdx}"
        ).fetchdf()
    shutil.rmtree(ctx.runtime / "corporate_secondary_spill", ignore_errors=True)

    def date_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
        return set(
            zip(
                frame["symbol"].astype(str),
                frame["trade_date"].astype(str),
                strict=True,
            )
        )

    primary_dates = date_set(primary_frame)
    cninfo_dates = date_set(secondary_frame)
    mootdx_dates = date_set(mootdx_frame)
    secondary_dates = cninfo_dates | mootdx_dates
    union = primary_dates | secondary_dates
    intersection = primary_dates & secondary_dates
    jaccard = len(intersection) / len(union) if union else 1.0

    keys = ["symbol", "trade_date", "action_type"]
    common_cninfo = primary_frame.merge(
        secondary_frame,
        on=keys,
        suffixes=("_primary", "_secondary"),
    )

    def materially_different(
        frame: pd.DataFrame,
        primary_column: str,
        secondary_column: str,
    ) -> pd.Series:
        primary_value = pd.to_numeric(frame[primary_column], errors="coerce").fillna(
            0.0
        )
        secondary_value = pd.to_numeric(
            frame[secondary_column], errors="coerce"
        ).fillna(0.0)
        return (
            (primary_value - secondary_value)
            .abs()
            .gt(np.maximum(secondary_value.abs() * 0.05, 0.05))
        )

    cninfo_amount_mismatch = pd.Series(False, index=common_cninfo.index, dtype=bool)
    for metric in (
        "cash_dividend_per_10",
        "bonus_share_per_10",
        "transfer_share_per_10",
    ):
        cninfo_amount_mismatch |= materially_different(
            common_cninfo,
            f"{metric}_primary",
            f"{metric}_secondary",
        )
    cninfo_amount_mismatch_count = int(cninfo_amount_mismatch.sum())
    cninfo_amount_mismatch_rate = (
        cninfo_amount_mismatch_count / len(common_cninfo) if len(common_cninfo) else 1.0
    )

    common_mootdx = primary_frame.merge(
        mootdx_frame,
        on=["symbol", "trade_date"],
        suffixes=("_primary", "_secondary"),
    )
    comparable_mootdx_cash = common_mootdx.loc[
        common_mootdx["cash_dividend_per_10_secondary"].notna()
    ].copy()
    mootdx_cash_mismatch = materially_different(
        comparable_mootdx_cash,
        "cash_dividend_per_10_primary",
        "cash_dividend_per_10_secondary",
    )
    mootdx_cash_mismatch_count = int(mootdx_cash_mismatch.sum())
    mootdx_cash_mismatch_rate = (
        mootdx_cash_mismatch_count / len(comparable_mootdx_cash)
        if len(comparable_mootdx_cash)
        else 1.0
    )
    raw_mismatches = (
        len(primary_dates ^ secondary_dates)
        + cninfo_amount_mismatch_count
        + mootdx_cash_mismatch_count
    )
    if (
        jaccard < 0.99
        or cninfo_amount_mismatch_rate > 0.01
        or mootdx_cash_mismatch_rate > 0.01
    ):
        raise AuxiliaryUpdateError(
            "corporate_secondary_validation_failed:"
            f"jaccard={jaccard:.8f}:primary={len(primary_dates)}:"
            f"secondary={len(secondary_dates)}:"
            f"cninfo_amount_rate={cninfo_amount_mismatch_rate:.8f}:"
            f"mootdx_cash_rate={mootdx_cash_mismatch_rate:.8f}"
        )
    metadata = update_active_manifest_metadata(
        "corporate_actions",
        reason="CNInfo plus Mootdx full implemented-dividend validation",
        workspace_root=ctx.workspace,
        source_updates={
            "secondary_validation_at": utc_now(),
            "secondary_compared_count": len(secondary_dates),
            "secondary_material_mismatch_count": 0,
            "secondary_validation_status": "ok",
            "secondary_event_jaccard": round(jaccard, 8),
            "secondary_cninfo_amount_mismatch_rate": round(
                cninfo_amount_mismatch_rate, 8
            ),
            "secondary_mootdx_cash_mismatch_rate": round(mootdx_cash_mismatch_rate, 8),
            "secondary_raw_mismatch_count": raw_mismatches,
        },
    )
    only_primary = sorted(primary_dates - secondary_dates)[:10]
    only_secondary = sorted(secondary_dates - primary_dates)[:10]
    shutil.rmtree(output_dir, ignore_errors=True)
    shutil.rmtree(
        ctx.runtime / "mootdx_corporate_validation_parts",
        ignore_errors=True,
    )
    return {
        **metadata,
        "primary_event_count": len(primary_dates),
        "secondary_event_count": len(secondary_dates),
        "jaccard": jaccard,
        "cninfo_amount_compared_count": len(common_cninfo),
        "cninfo_amount_mismatch_count": cninfo_amount_mismatch_count,
        "cninfo_amount_mismatch_rate": cninfo_amount_mismatch_rate,
        "mootdx_cash_compared_count": len(comparable_mootdx_cash),
        "mootdx_cash_mismatch_count": mootdx_cash_mismatch_count,
        "mootdx_cash_mismatch_rate": mootdx_cash_mismatch_rate,
        "raw_mismatch_count": raw_mismatches,
        "only_primary_examples": only_primary,
        "only_secondary_examples": only_secondary,
    }


def _validate_auxiliary_domains(
    ctx: AuxiliaryContext,
    domains: Sequence[str],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for domain in domains:
        if domain == "industry_concept":
            results[domain] = validate_industry_secondary(ctx)
        elif domain == "share_capital":
            results[domain] = validate_share_capital_secondary(ctx)
        elif domain == "corporate_actions":
            results[domain] = validate_corporate_actions_secondary(ctx)
        elif domain == "index_constituents":
            results[domain] = validate_index_secondary(ctx)
        else:
            manifest = _manifest(ctx.root, ctx.datasets, domain)
            status = str(
                dict(manifest.source or {}).get("secondary_validation_status", "")
            )
            if status not in {"ok", "not_comparable"}:
                raise AuxiliaryUpdateError(
                    f"auxiliary_secondary_validation_pending:{domain}:{status}"
                )
            results[domain] = {
                "status": "already_validated",
                "secondary_validation_status": status,
            }
    return results


def run_auxiliary_validation(
    *,
    as_of_date: str,
    workspace_root: str | Path | None = None,
    domains: Sequence[str] = AUXILIARY_DOMAINS,
) -> dict[str, Any]:
    ctx = _context(as_of_date=as_of_date, workspace_root=workspace_root)
    selected = tuple(str(item) for item in domains)
    unknown = sorted(set(selected).difference(AUXILIARY_DOMAINS))
    if unknown:
        raise ValueError(f"unknown_auxiliary_domains:{','.join(unknown)}")
    return {
        "status": "validated",
        "as_of_date": ctx.target_date,
        "domains": _validate_auxiliary_domains(ctx, selected),
    }


def run_auxiliary_repair(
    *,
    as_of_date: str,
    workspace_root: str | Path | None = None,
    domains: Sequence[str] = AUXILIARY_DOMAINS,
    force: bool = False,
) -> dict[str, Any]:
    ctx = _context(as_of_date=as_of_date, workspace_root=workspace_root)
    selected = tuple(str(item) for item in domains)
    unknown = sorted(set(selected).difference(AUXILIARY_DOMAINS))
    if unknown:
        raise ValueError(f"unknown_auxiliary_domains:{','.join(unknown)}")
    state: dict[str, Any] = {
        "status": "repairing",
        "as_of_date": ctx.target_date,
        "domains": list(selected),
        "completed": {},
        "failed_stage": "",
    }
    if _runtime_state_path(ctx).is_file():
        try:
            prior = json.loads(_runtime_state_path(ctx).read_text(encoding="utf-8"))
            if str(prior.get("as_of_date", "")) == ctx.target_date:
                state["completed"] = dict(prior.get("completed", {}) or {})
        except (OSError, ValueError):
            pass
    _write_state(ctx, state)
    daily_basic_parts: list[Path] = []
    try:
        for domain in selected:
            manifest = _manifest(ctx.root, ctx.datasets, domain)
            checked = str(dict(manifest.source or {}).get("checked_through", ""))
            source_contract = str(
                dict(manifest.source or {}).get("source_contract", "")
            )
            requires_repair = False
            if domain == "share_capital":
                requires_repair = bool(_legacy_cninfo_share_symbols(ctx))
            elif domain == "valuation":
                requires_repair = _valuation_market_cap_error_count(ctx) > 0
            elif domain in {"industry_concept", "index_constituents"}:
                requires_repair = _identity_dependent_repair_required(
                    ctx,
                    domain,
                )
            if (
                checked == ctx.target_date
                and "strict" in source_contract.lower()
                and not requires_repair
                and not force
            ):
                state["completed"][domain] = {
                    "status": "already_current",
                    "checked_through": checked,
                }
                _write_state(ctx, state)
                continue
            state["stage"] = domain
            _write_state(ctx, state)
            if domain == "industry_concept":
                result = repair_industry(ctx)
            elif domain == "name_change":
                result = repair_name_change(ctx)
            elif domain == "corporate_actions":
                result = repair_corporate_actions(ctx)
            elif domain == "index_constituents":
                result = repair_index_constituents(ctx)
            elif domain == "share_capital":
                result, daily_basic_parts = repair_share_capital(ctx)
            elif domain == "valuation":
                result = repair_valuation(
                    ctx,
                    daily_basic_parts=daily_basic_parts,
                )
            else:  # pragma: no cover - guarded above
                raise AssertionError(domain)
            state["completed"][domain] = result
            _write_state(ctx, state)
        state["stage"] = "secondary_validation"
        _write_state(ctx, state)
        state["validation"] = _validate_auxiliary_domains(ctx, selected)
        _write_state(ctx, state)
        state["status"] = "repaired"
        state["stage"] = ""
        _write_state(ctx, state)
        shutil.rmtree(ctx.runtime / "daily_basic_parts", ignore_errors=True)
        shutil.rmtree(ctx.runtime / "duckdb_spill", ignore_errors=True)
        _runtime_state_path(ctx).unlink(missing_ok=True)
        try:
            ctx.runtime.rmdir()
        except OSError:
            pass
        return state
    except Exception as exc:
        state.update(
            {
                "status": "failed",
                "failed_stage": str(state.get("stage", "")),
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
        )
        _write_state(ctx, state)
        return state


def run_auxiliary_update(
    *,
    as_of_date: str,
    workspace_root: str | Path | None = None,
    baostock_valuation_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Append a free-source auxiliary tail after strict contracts are installed."""

    ctx = _context(as_of_date=as_of_date, workspace_root=workspace_root)
    stale: list[str] = []
    for domain in AUXILIARY_DOMAINS:
        manifest = _manifest(ctx.root, ctx.datasets, domain)
        if str(
            dict(manifest.source or {}).get("checked_through", "")
        ) != ctx.target_date or _identity_dependent_repair_required(ctx, domain):
            stale.append(domain)
    if not stale:
        if baostock_valuation_cache_path is not None:
            Path(baostock_valuation_cache_path).resolve().unlink(missing_ok=True)
        return {
            "status": "current",
            "as_of_date": ctx.target_date,
            "domains": list(AUXILIARY_DOMAINS),
        }
    from quantlab.data.qdp_v2.auxiliary_tail_update import (
        run_auxiliary_tail_update,
    )

    return run_auxiliary_tail_update(
        as_of_date=ctx.target_date,
        workspace_root=ctx.workspace,
        domains=stale,
        baostock_valuation_cache_path=baostock_valuation_cache_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qdp auxiliary-update",
        description="Repair or update the six strict-PIT auxiliary domains.",
    )
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--domains", default=",".join(AUXILIARY_DOMAINS))
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = str(args.workspace_root or "") or None
    domains = tuple(
        item.strip() for item in str(args.domains).split(",") if item.strip()
    )
    if args.dry_run and args.validate_only:
        raise ValueError("dry_run_and_validate_only_are_mutually_exclusive")
    payload = (
        plan_auxiliary_update(
            as_of_date=str(args.as_of_date),
            workspace_root=workspace,
        )
        if args.dry_run
        else run_auxiliary_validation(
            as_of_date=str(args.as_of_date),
            workspace_root=workspace,
            domains=domains,
        )
        if args.validate_only
        else run_auxiliary_repair(
            as_of_date=str(args.as_of_date),
            workspace_root=workspace,
            domains=domains,
            force=bool(args.force),
        )
    )
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    return (
        0
        if payload.get("status")
        in {
            "planned",
            "repaired",
            "validated",
        }
        else 2
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


# Keep the historical private names as a compatibility surface while making
# the deterministic provider-to-QDP conversions live in one dependency-free
# module.  Existing callers can migrate without changing update orchestration.
_provider_date = _normalization.provider_date
_from_baostock_code = _normalization.from_baostock_code
_normalize_comparison_text = _normalization.normalize_comparison_text
_normalize_industry_comparison = _normalization.normalize_industry_comparison
_normalize_cninfo_industry_history = _normalization.normalize_cninfo_industry_history
_normalize_name_intervals = _normalization.normalize_name_intervals
_normalize_daily_basic = _normalization.normalize_daily_basic
_normalize_cninfo_share_change = _normalization.normalize_cninfo_share_change
_tushare_date_series = _normalization.tushare_date_series
_normalize_dividend = _normalization.normalize_dividend
_normalize_cninfo_dividend = _normalization.normalize_cninfo_dividend
CORPORATE_COLUMNS = _normalization.CORPORATE_COLUMNS
CNINFO_INDUSTRY_COLUMNS = _normalization.CNINFO_INDUSTRY_COLUMNS
CNINFO_SHARE_NORMALIZED_COLUMNS = _normalization.CNINFO_SHARE_NORMALIZED_COLUMNS
DAILY_BASIC_FIELDS = _normalization.DAILY_BASIC_FIELDS
DAILY_BASIC_NORMALIZED_COLUMNS = _normalization.DAILY_BASIC_NORMALIZED_COLUMNS
DIVIDEND_FIELDS = _normalization.DIVIDEND_FIELDS
MOOTDX_CORPORATE_VALIDATION_COLUMNS = _normalization.MOOTDX_CORPORATE_VALIDATION_COLUMNS
NAME_CHANGE_FIELDS = _normalization.NAME_CHANGE_FIELDS
NAME_INTERVAL_COLUMNS = _normalization.NAME_INTERVAL_COLUMNS


__all__ = [
    "AUXILIARY_DOMAINS",
    "DAILY_AUXILIARY_DOMAINS",
    "plan_auxiliary_update",
    "repair_corporate_actions",
    "repair_index_constituents",
    "repair_industry",
    "repair_name_change",
    "repair_share_capital",
    "repair_valuation",
    "run_auxiliary_repair",
    "run_auxiliary_update",
    "run_auxiliary_validation",
    "validate_corporate_actions_secondary",
    "validate_index_secondary",
    "validate_industry_secondary",
    "validate_share_capital_secondary",
]
