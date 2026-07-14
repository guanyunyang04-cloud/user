from __future__ import annotations

from bisect import bisect_right
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from quant_data_platform.core.json_io import read_json
from quant_data_platform.domains.contracts import DOMAIN_STANDARD_COLUMNS, DataDomain, DomainFetchRequest
from quant_data_platform.providers import BAOSTOCK_BATCH_VERSION, BAOSTOCK_BATCH_WHEEL_SHA256, BaostockProvider
from quant_data_platform.qdp_v3.constants import (
    DOMAIN_FINANCIAL_QUARTERLY,
    DOMAIN_INDEX_CONSTITUENTS,
    DOMAIN_INDUSTRY,
    DOMAIN_PERFORMANCE_EXPRESS,
    DOMAIN_PERFORMANCE_FORECAST,
    QUALITY_PROVISIONAL,
    RAW_FINANCIAL_QUARTERLY,
    RAW_INDEX_CONSTITUENTS,
    RAW_INDUSTRY_SNAPSHOT,
    RAW_PERFORMANCE_EXPRESS,
    RAW_PERFORMANCE_FORECAST,
)
from quant_data_platform.qdp_v3.identity import SecurityIdentityRegistry, normalize_symbol
from quant_data_platform.qdp_v3.manifest import atomic_write_json, stable_hash, utc_now
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout, qdp_v3_paths
from quant_data_platform.qdp_v3.storage import RawPartitionRef, iter_raw_partitions, read_raw_partition, write_raw_partition


REPORT_DOMAINS = {
    DOMAIN_FINANCIAL_QUARTERLY: (DataDomain.FINANCIAL_QUARTERLY, RAW_FINANCIAL_QUARTERLY),
    DOMAIN_PERFORMANCE_FORECAST: (DataDomain.PERFORMANCE_FORECAST, RAW_PERFORMANCE_FORECAST),
    DOMAIN_PERFORMANCE_EXPRESS: (DataDomain.PERFORMANCE_EXPRESS, RAW_PERFORMANCE_EXPRESS),
}

SNAPSHOT_DOMAINS = {
    DOMAIN_INDUSTRY: (DataDomain.INDUSTRY_CONCEPT, RAW_INDUSTRY_SNAPSHOT),
    DOMAIN_INDEX_CONSTITUENTS: (DataDomain.INDEX_CONSTITUENTS, RAW_INDEX_CONSTITUENTS),
}


def _assert_runtime_and_gate(workspace_root: str | Path | None, provider: BaostockProvider | None) -> None:
    runtime = importlib_metadata.version("baostock")
    if runtime != BAOSTOCK_BATCH_VERSION:
        raise RuntimeError(f"baostock_version_mismatch:expected={BAOSTOCK_BATCH_VERSION}:actual={runtime}")
    if provider is None or isinstance(provider, BaostockProvider):
        from quant_data_platform.qdp_v3.compatibility import assert_baostock_compatibility_gate

        assert_baostock_compatibility_gate(workspace_root)


def ingest_baostock_snapshot_domain(
    *,
    domain: str,
    as_of_date: str,
    workspace_root: str | Path | None = None,
    provider: BaostockProvider | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    normalized = str(domain).strip().lower()
    if normalized not in SNAPSHOT_DOMAINS:
        raise ValueError(f"unsupported_secondary_snapshot_domain:{domain}")
    ensure_qdp_v3_layout(workspace_root)
    _assert_runtime_and_gate(workspace_root, provider)
    provider_domain, raw_domain = SNAPSHOT_DOMAINS[normalized]
    existing = iter_raw_partitions(raw_domain, workspace_root=workspace_root, start_value=str(as_of_date), end_value=str(as_of_date))
    if existing and not refresh:
        ref = existing[-1]
        return {"status": "completed", "domain": normalized, "created": False, "row_count": ref.row_count, "content_sha256": ref.content_sha256}
    source = provider or BaostockProvider()
    result = source.fetch_domain(DomainFetchRequest(domain=provider_domain, start_date=str(as_of_date), end_date=str(as_of_date)))
    quality_tier = "quarantined" if result.error_report else QUALITY_PROVISIONAL
    ref, created = write_raw_partition(
        raw_domain=raw_domain,
        partition_value=str(as_of_date),
        frame=result.data,
        receipt={
            "provider": "baostock",
            "endpoint": "query_stock_industry" if normalized == DOMAIN_INDUSTRY else "query_sz50_stocks+query_hs300_stocks+query_zz500_stocks",
            "package_version": importlib_metadata.version("baostock"),
            "wheel_sha256": BAOSTOCK_BATCH_WHEEL_SHA256,
            "request": {"as_of_date": str(as_of_date), "domain": normalized},
            "error_code": "0" if not result.error_report else "provider_error",
            "error_report": result.error_report,
            "quality_tier": quality_tier,
            "quality_note": "Single-date provider snapshot; unknown history is never backfilled from this observation.",
        },
        workspace_root=workspace_root,
    )
    return {"status": "completed" if not result.error_report else "partial", "domain": normalized, "created": bool(created), "row_count": ref.row_count, "content_sha256": ref.content_sha256, "error_report": result.error_report}


def ingest_baostock_report_domain(
    *,
    domain: str,
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    workspace_root: str | Path | None = None,
    provider: BaostockProvider | None = None,
    refresh: bool = False,
    chunk_size: int = 8,
    job_id: str = "",
    symbol_lifecycle_ranges: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, Any]:
    normalized = str(domain).strip().lower()
    if normalized not in REPORT_DOMAINS:
        raise ValueError(f"unsupported_secondary_report_domain:{domain}")
    ensure_qdp_v3_layout(workspace_root)
    _assert_runtime_and_gate(workspace_root, provider)
    provider_domain, raw_domain = REPORT_DOMAINS[normalized]
    normalized_symbols = sorted({normalize_symbol(item) for item in symbols if normalize_symbol(item)})
    if not normalized_symbols:
        raise ValueError("secondary_report_symbols_empty")
    lifecycle = {
        normalize_symbol(symbol): (str(values[0])[:10], str(values[1])[:10])
        for symbol, values in dict(symbol_lifecycle_ranges or {}).items()
        if normalize_symbol(symbol) and isinstance(values, (tuple, list)) and len(values) >= 2
    }
    effective_ranges: dict[str, tuple[str, str]] = {}
    for symbol in normalized_symbols:
        list_date, delist_date = lifecycle.get(symbol, ("", ""))
        effective_start = str(start_date)
        effective_end = str(end_date)
        if normalized == DOMAIN_FINANCIAL_QUARTERLY and len(list_date) == 10:
            evidence_start = pd.Timestamp(list_date) - pd.Timedelta(days=550)
            evidence_start = evidence_start.to_period("Q").start_time.strftime("%Y-%m-%d")
            effective_start = max(effective_start, evidence_start)
        if normalized == DOMAIN_FINANCIAL_QUARTERLY and len(delist_date) == 10 and delist_date < str(end_date):
            effective_end = pd.offsets.QuarterEnd().rollback(pd.Timestamp(delist_date)).strftime("%Y-%m-%d")
        effective_ranges[symbol] = (effective_start, effective_end)
    range_contract_sha = stable_hash(effective_ranges, length=32)
    if not job_id:
        job_id = f"ingest_{normalized}__{stable_hash({'symbols': normalized_symbols, 'start': start_date, 'end': end_date, 'ranges': range_contract_sha}, length=20)}"
    job_path = qdp_v3_paths(workspace_root).jobs / f"{job_id}.json"
    state = read_json(job_path) or {
        "job_id": job_id,
        "provider": "baostock",
        "mode": normalized,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "symbols": normalized_symbols,
        "effective_ranges_sha256": range_contract_sha,
        "tasks": {symbol: {"status": "pending", "error": ""} for symbol in normalized_symbols},
        "created_at": utc_now(),
    }
    signature = {key: state.get(key) for key in ("mode", "start_date", "end_date", "symbols", "effective_ranges_sha256")}
    expected = {"mode": normalized, "start_date": str(start_date), "end_date": str(end_date), "symbols": normalized_symbols, "effective_ranges_sha256": range_contract_sha}
    if signature != expected:
        raise RuntimeError(f"secondary_report_job_contract_conflict:{job_id}")
    tasks = dict(state.get("tasks", {}) or {})
    range_key = lambda symbol: f"{symbol}__{str(start_date)}__{str(end_date)}"
    existing = {ref.partition_value: ref for ref in iter_raw_partitions(raw_domain, workspace_root=workspace_root)}
    pending: list[str] = []
    for symbol in normalized_symbols:
        token = range_key(symbol)
        if not refresh and token in existing:
            tasks[symbol] = {"status": "skipped", "row_count": existing[token].row_count, "content_sha256": existing[token].content_sha256, "error": ""}
        elif not refresh and str(dict(tasks.get(symbol, {}) or {}).get("status", "")) == "completed":
            continue
        elif effective_ranges[symbol][0] > effective_ranges[symbol][1]:
            tasks[symbol] = {"status": "completed", "row_count": 0, "not_applicable": True, "error": ""}
        else:
            pending.append(symbol)
    source = provider or BaostockProvider(_reuse_symbol_range_session=True)
    state.update({"status": "running", "tasks": tasks, "updated_at": utc_now()})
    atomic_write_json(job_path, state)
    pending_by_range: dict[tuple[str, str], list[str]] = {}
    for symbol in pending:
        pending_by_range.setdefault(effective_ranges[symbol], []).append(symbol)
    chunks: list[tuple[tuple[str, str], list[str]]] = []
    size = max(1, int(chunk_size))
    for effective_range, range_symbols in sorted(pending_by_range.items()):
        chunks.extend((effective_range, range_symbols[offset : offset + size]) for offset in range(0, len(range_symbols), size))
    for (effective_start, effective_end), chunk in chunks:
        try:
            result = source.fetch_domain(
                DomainFetchRequest(domain=provider_domain, symbols=tuple(chunk), start_date=effective_start, end_date=effective_end)
            )
            errors = {normalize_symbol(item.get("symbol", "")): item for item in result.error_report if isinstance(item, dict) and item.get("symbol")}
            for symbol in chunk:
                if symbol in errors:
                    tasks[symbol] = {"status": "failed", "error": str(errors[symbol].get("message", "provider_symbol_error"))}
                    continue
                frame = result.data.loc[result.data.get("symbol", pd.Series(dtype=str)).astype(str).eq(symbol)].copy() if not result.data.empty else result.data.copy()
                ref, created = write_raw_partition(
                    raw_domain=raw_domain,
                    partition_field="provider_symbol_range",
                    partition_value=range_key(symbol),
                    frame=frame,
                    receipt={
                        "provider": "baostock",
                        "endpoint": normalized,
                        "package_version": importlib_metadata.version("baostock"),
                        "wheel_sha256": BAOSTOCK_BATCH_WHEEL_SHA256,
                        "request": {
                            "symbol": symbol,
                            "start_date": str(start_date),
                            "end_date": str(end_date),
                            "provider_start_date": effective_start,
                            "provider_end_date": effective_end,
                        },
                        "error_code": "0",
                        "quality_tier": QUALITY_PROVISIONAL,
                        "quality_note": "PIT availability is rebuilt from an explicit publish date and the next exchange trading day; inferred publish dates remain unavailable.",
                    },
                    workspace_root=workspace_root,
                )
                tasks[symbol] = {"status": "completed", "row_count": ref.row_count, "content_sha256": ref.content_sha256, "created": bool(created), "error": ""}
        except Exception as exc:
            for symbol in chunk:
                tasks[symbol] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        state.update({"tasks": tasks, "updated_at": utc_now()})
        atomic_write_json(job_path, state)
    failed = [{"symbol": symbol, **dict(task)} for symbol, task in tasks.items() if str(dict(task).get("status", "")) == "failed"]
    state.update({"status": "partial" if failed else "completed", "tasks": tasks, "updated_at": utc_now()})
    atomic_write_json(job_path, state)
    if provider is None:
        source.close()
    return {
        "status": state["status"],
        "domain": normalized,
        "job_id": job_id,
        "symbol_count": len(normalized_symbols),
        "completed_count": sum(str(dict(item).get("status", "")) == "completed" for item in tasks.values()),
        "skipped_count": sum(str(dict(item).get("status", "")) == "skipped" for item in tasks.values()),
        "failed_count": len(failed),
        "provider_query_chunk_count": len(chunks),
        "lifecycle_clipped_symbol_count": sum(effective_ranges[symbol] != (str(start_date), str(end_date)) for symbol in normalized_symbols),
        "failed": failed[:50],
        "job_path": str(job_path.resolve()),
    }


def _next_open_dates(calendar: pd.DataFrame) -> dict[str, str]:
    if calendar is None or calendar.empty:
        return {}
    opened = calendar.loc[calendar["is_open"].astype(str).str.lower().isin({"1", "true", "t", "yes"}), "trade_date"].astype(str).str.slice(0, 10)
    dates = sorted(set(opened))
    # A disclosure without an intraday timestamp becomes usable strictly on
    # the following exchange trading day, never on its calendar date.
    out: dict[str, str] = {}
    for date in dates:
        later = next((candidate for candidate in dates if candidate > date), "")
        out[date] = later
    return out


def _next_open_after(date: str, open_dates: list[str]) -> str:
    index = bisect_right(open_dates, str(date))
    return open_dates[index] if index < len(open_dates) else ""


def canonicalize_secondary_domain(
    *,
    domain: str,
    refs: Iterable[RawPartitionRef],
    identity_registry: SecurityIdentityRegistry,
    calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized = str(domain).strip().lower()
    frames = [read_raw_partition(ref) for ref in refs]
    raw = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if raw.empty:
        provider_domain = REPORT_DOMAINS[normalized][0] if normalized in REPORT_DOMAINS else SNAPSHOT_DOMAINS[normalized][0]
        source_columns = [column for column in DOMAIN_STANDARD_COLUMNS[provider_domain] if column != "symbol"]
        if normalized in REPORT_DOMAINS:
            prefix = ["security_id", "report_date", "symbol_on_date", "provider_symbol", "publish_date", "available_date", "availability_status", "identity_mapping_status", "quality_tier"]
        else:
            prefix = ["security_id", "trade_date", "symbol_on_date", "provider_symbol", "identity_mapping_status", "quality_tier"]
        return pd.DataFrame(columns=[*prefix, *[column for column in source_columns if column not in prefix]]), pd.DataFrame()
    if "symbol" not in raw.columns:
        raise ValueError(f"secondary_domain_symbol_missing:{normalized}")
    raw = raw.copy()
    raw["provider_symbol"] = raw["symbol"].map(normalize_symbol)
    mapping_date = "report_date" if normalized in REPORT_DOMAINS else "trade_date"
    mapped = identity_registry.map_frame(raw, provider_symbol_column="provider_symbol", date_column=mapping_date)
    conflicts = mapped.loc[mapped["identity_mapping_status"].ne("mapped")].copy()
    mapped = mapped.loc[mapped["identity_mapping_status"].eq("mapped")].copy()
    mapped = mapped.drop(columns=["symbol"], errors="ignore")
    if normalized in REPORT_DOMAINS:
        open_dates = sorted(
            set(
                calendar.loc[
                    calendar["is_open"].astype(str).str.lower().isin({"1", "true", "t", "yes"}),
                    "trade_date",
                ].astype(str).str.slice(0, 10)
            )
        ) if calendar is not None and not calendar.empty else []
        explicit = mapped.get("lag_policy", pd.Series(index=mapped.index, dtype=str)).astype(str).eq("publish_date_plus_1d_in_features")
        publish = mapped.get("publish_date", pd.Series(index=mapped.index, dtype=str)).fillna("").astype(str).str.slice(0, 10)
        mapped["available_date"] = ""
        mapped.loc[explicit, "available_date"] = publish.loc[explicit].map(lambda date: _next_open_after(date, open_dates))
        mapped["availability_status"] = "publish_date_unproven"
        mapped.loc[explicit & mapped["available_date"].ne(""), "availability_status"] = "next_trading_day_after_publish_date"
        mapped.loc[explicit & mapped["available_date"].eq(""), "availability_status"] = "calendar_horizon_missing"
        mapped["quality_tier"] = QUALITY_PROVISIONAL
        prefix = ["security_id", "report_date", "symbol_on_date", "provider_symbol", "publish_date", "available_date", "availability_status", "identity_mapping_status", "quality_tier"]
        remaining = [column for column in mapped.columns if column not in prefix]
        mapped = mapped.loc[:, [*prefix, *remaining]]
        key = ["security_id", "report_date", "publish_date", "source"]
        mapped = mapped.drop_duplicates(key, keep="last").sort_values(["report_date", "security_id", "publish_date"])
    else:
        mapped["quality_tier"] = QUALITY_PROVISIONAL
        prefix = ["security_id", "trade_date", "symbol_on_date", "provider_symbol", "identity_mapping_status", "quality_tier"]
        remaining = [column for column in mapped.columns if column not in prefix]
        mapped = mapped.loc[:, [*prefix, *remaining]]
        if normalized == DOMAIN_INDEX_CONSTITUENTS:
            key = ["trade_date", "index_symbol", "security_id", "source"]
        else:
            key = ["trade_date", "security_id", "source"]
        mapped = mapped.drop_duplicates(key, keep="last").sort_values(key)
    return mapped.reset_index(drop=True), conflicts.reset_index(drop=True)


def refs_for_secondary_domain(domain: str, *, workspace_root: str | Path | None = None) -> list[RawPartitionRef]:
    normalized = str(domain).strip().lower()
    if normalized in REPORT_DOMAINS:
        raw_domain = REPORT_DOMAINS[normalized][1]
    elif normalized in SNAPSHOT_DOMAINS:
        raw_domain = SNAPSHOT_DOMAINS[normalized][1]
    else:
        raise ValueError(f"unsupported_secondary_domain:{domain}")
    return iter_raw_partitions(raw_domain, workspace_root=workspace_root)
