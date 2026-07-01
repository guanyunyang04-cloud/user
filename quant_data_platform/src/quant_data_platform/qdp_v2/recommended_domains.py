from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pandas as pd

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.domains.contracts import DataDomain, DomainFetchRequest, normalize_domain_frame
from quant_data_platform.qdp_v2.environment import runtime_environment
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    atomic_write_json,
    dataset_manifest_for_id,
    path_for_manifest,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    schema_hash,
    stable_hash,
    utc_now,
    write_active_manifest,
    write_dataset_manifest,
)
from quant_data_platform.qdp_v2.runtime import resolve_runtime_profile


SUPPORTED_DOMAINS = (
    DataDomain.LIMIT_STATUS,
    DataDomain.ANNOUNCEMENT,
    DataDomain.FINANCIAL_QUARTERLY,
    DataDomain.PERFORMANCE_FORECAST,
    DataDomain.PERFORMANCE_EXPRESS,
    DataDomain.CORPORATE_ACTIONS,
    DataDomain.SHARE_CAPITAL,
    DataDomain.NAME_CHANGE,
)

LOW_REQUEST_DOMAINS = (
    DataDomain.CORPORATE_ACTIONS,
    DataDomain.NAME_CHANGE,
    DataDomain.LIMIT_STATUS,
)

PRIMARY_KEYS = {
    DataDomain.LIMIT_STATUS: ["trade_date", "symbol"],
    DataDomain.ANNOUNCEMENT: ["trade_date", "symbol", "title", "url"],
    DataDomain.FINANCIAL_QUARTERLY: ["symbol", "report_date", "source"],
    DataDomain.PERFORMANCE_FORECAST: ["symbol", "report_date", "publish_date", "source"],
    DataDomain.PERFORMANCE_EXPRESS: ["symbol", "report_date", "publish_date", "source"],
    DataDomain.CORPORATE_ACTIONS: ["symbol", "trade_date", "action_type", "description", "source"],
    DataDomain.SHARE_CAPITAL: ["trade_date", "symbol", "source"],
    DataDomain.NAME_CHANGE: ["trade_date", "symbol", "change_type", "source"],
}

CONTRACTS = {
    DataDomain.LIMIT_STATUS: "qdp_v2_limit_status_events_v1",
    DataDomain.ANNOUNCEMENT: "qdp_v2_announcement_raw_v1",
    DataDomain.FINANCIAL_QUARTERLY: "qdp_v2_financial_quarterly_raw_v1",
    DataDomain.PERFORMANCE_FORECAST: "qdp_v2_performance_forecast_raw_v1",
    DataDomain.PERFORMANCE_EXPRESS: "qdp_v2_performance_express_raw_v1",
    DataDomain.CORPORATE_ACTIONS: "qdp_v2_corporate_actions_raw_v1",
    DataDomain.SHARE_CAPITAL: "qdp_v2_share_capital_raw_v1",
    DataDomain.NAME_CHANGE: "qdp_v2_name_change_raw_v1",
}


def build_recommended_domains(
    *,
    workspace_root: str | Path | None = None,
    domains: Iterable[str] | None = None,
    start_date: str = "",
    end_date: str = "",
    runtime: str = "balanced",
    workers: int = 0,
    max_symbols: int = 0,
    max_dates: int = 0,
    chunk_size: int = 200,
    activate: bool = False,
    force: bool = False,
    dry_run: bool = False,
    include_calendar_days: bool = False,
) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    active = read_active_manifest(root)
    if not active:
        return {"status": "error", "errors": ["active_manifest_missing"], "qdp_v2_root": str(root.resolve())}
    profile = resolve_runtime_profile(runtime)
    worker_count = max(1, int(workers or profile.duckdb_threads or 1))
    active_start = str(start_date or active.get("active_start_date", "") or "").strip()
    active_end = str(end_date or active.get("active_as_of_date", "") or "").strip()
    if not active_start or not active_end:
        return {"status": "error", "errors": ["active_start_or_end_date_missing"], "active_start_date": active_start, "active_end_date": active_end}
    selected_domains = _parse_domains(domains)
    symbols = _active_symbols(root=root, active=active, max_symbols=max_symbols)
    dates = _calendar_dates(root=root, active=active, start_date=active_start, end_date=active_end, max_dates=max_dates, include_calendar_days=include_calendar_days)
    plan = {
        "status": "planned",
        "qdp_v2_root": str(root.resolve()),
        "domains": selected_domains,
        "start_date": active_start,
        "end_date": active_end,
        "symbol_count": len(symbols),
        "date_count": len(dates),
        "runtime": profile.name,
        "workers": worker_count,
        "activate": bool(activate),
        "force": bool(force),
        "include_calendar_days": bool(include_calendar_days),
        "runtime_environment": runtime_environment(),
    }
    if dry_run:
        plan["fetch_execution"] = "not_started"
        _write_run(root, "recommended_domains_dry_run", plan)
        return plan

    active_raw = dict(active.get("raw", {}) or {})
    results: list[dict[str, Any]] = []
    for domain in selected_domains:
        if active_raw.get(domain) and not force:
            results.append({"domain": domain, "status": "skipped", "reason": "already_active", "dataset_id": active_raw.get(domain)})
            continue
        started = time.time()
        try:
            frame, fetch_report = _fetch_domain(
                root=root,
                active=active,
                domain=domain,
                symbols=symbols,
                dates=dates,
                start_date=active_start,
                end_date=active_end,
                workers=worker_count,
                chunk_size=max(1, int(chunk_size or 1)),
            )
            if frame.empty:
                results.append(
                    {
                        "domain": domain,
                        "status": "empty",
                        "fetch_report": fetch_report,
                        "elapsed_seconds": round(time.time() - started, 3),
                    }
                )
                continue
            manifest, manifest_path = _write_domain_dataset(root=root, domain=domain, frame=frame, source_report=fetch_report)
            result_status = "partial" if int(fetch_report.get("error_count", 0) or 0) else "ok"
            results.append(
                {
                    "domain": domain,
                    "status": result_status,
                    "dataset_id": manifest.dataset_id,
                    "manifest_path": str(manifest_path.resolve()),
                    "row_count": manifest.row_count,
                    "start_date": manifest.start_date,
                    "end_date": manifest.end_date,
                    "duplicate_key_rows": manifest.quality.get("duplicate_key_rows"),
                    "fetch_report": fetch_report,
                    "elapsed_seconds": round(time.time() - started, 3),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "domain": domain,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "elapsed_seconds": round(time.time() - started, 3),
                }
            )

    active_path = ""
    if activate:
        successful = {str(item["domain"]): str(item["dataset_id"]) for item in results if item.get("status") == "ok" and item.get("dataset_id")}
        if successful:
            updated = dict(active)
            raw = dict(updated.get("raw", {}) or {})
            raw.update(successful)
            updated["raw"] = raw
            source = dict(updated.get("source", {}) or {})
            previous_recommended = set()
            if isinstance(source.get("recommended_domains"), Mapping):
                previous_recommended = {str(item) for item in list(source["recommended_domains"].get("domains", []) or []) if str(item)}
            source["recommended_domains"] = {
                "updated_at": utc_now(),
                "domains": sorted(previous_recommended.union(successful)),
                "created_by": "qdp supplement recommended",
            }
            updated["source"] = source
            active_path = str(write_active_manifest(root, updated).resolve())

    payload = {
        **plan,
        "status": "ok" if all(str(item.get("status")) in {"ok", "empty", "skipped"} for item in results) else "needs_attention",
        "fetch_execution": "completed",
        "active_manifest": active_path,
        "results": results,
        "ok_domains": [item["domain"] for item in results if item.get("status") == "ok"],
        "empty_domains": [item["domain"] for item in results if item.get("status") == "empty"],
        "error_domains": [item["domain"] for item in results if item.get("status") == "error"],
    }
    run_path = _write_run(root, "recommended_domains", payload)
    payload["run_manifest"] = str(run_path.resolve())
    return payload


def _parse_domains(domains: Iterable[str] | None) -> list[str]:
    raw = [item for domain in list(domains or []) for item in str(domain).split(",")]
    selected = [str(item).strip().lower() for item in raw if str(item).strip()]
    if not selected:
        selected = list(SUPPORTED_DOMAINS)
    aliases = {
        "all": ",".join(SUPPORTED_DOMAINS),
        "low_request": ",".join(LOW_REQUEST_DOMAINS),
        "low": ",".join(LOW_REQUEST_DOMAINS),
    }
    expanded: list[str] = []
    for item in selected:
        if item in aliases:
            expanded.extend([part for part in aliases[item].split(",") if part])
        else:
            expanded.append(item)
    invalid = sorted(set(expanded).difference(SUPPORTED_DOMAINS))
    if invalid:
        raise ValueError(f"unsupported_recommended_domains:{invalid}")
    return list(dict.fromkeys(expanded))


def _fetch_domain(
    *,
    root: Path,
    active: Mapping[str, Any],
    domain: str,
    symbols: list[str],
    dates: list[str],
    start_date: str,
    end_date: str,
    workers: int,
    chunk_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if domain == DataDomain.LIMIT_STATUS:
        return _derive_limit_status_from_daily(root=root, active=active, symbols=symbols, start_date=start_date, end_date=end_date)
    if domain == DataDomain.ANNOUNCEMENT:
        return _fetch_announcements(symbols=symbols, dates=dates, workers=workers)
    if domain == DataDomain.CORPORATE_ACTIONS:
        return _fetch_corporate_actions(symbols=symbols, start_date=start_date, end_date=end_date, workers=workers)
    if domain == DataDomain.SHARE_CAPITAL:
        return _fetch_share_capital(symbols=symbols, start_date=start_date, end_date=end_date, workers=workers)
    if domain == DataDomain.NAME_CHANGE:
        return _fetch_name_change(symbols=symbols, start_date=start_date, end_date=end_date)
    if domain in {DataDomain.FINANCIAL_QUARTERLY, DataDomain.PERFORMANCE_FORECAST, DataDomain.PERFORMANCE_EXPRESS}:
        return _fetch_baostock_report_domain(domain=domain, symbols=symbols, start_date=start_date, end_date=end_date, chunk_size=chunk_size)
    raise ValueError(f"unsupported_domain:{domain}")


def _derive_limit_status_from_daily(*, root: Path, active: Mapping[str, Any], symbols: list[str], start_date: str, end_date: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = dict(active.get("raw", {}) or {})
    daily_id = str(raw.get("market_daily_raw", "") or "")
    status_id = str(raw.get("security_status", "") or "")
    daily_path = dataset_manifest_for_id(root, daily_id, "market_daily_raw")
    status_path = dataset_manifest_for_id(root, status_id, "security_status")
    if daily_path is None or status_path is None:
        return pd.DataFrame(), {"provider": "qdp_v2_derived", "method": "daily_close_limit_rule", "error_count": 1, "errors": ["market_daily_raw_or_security_status_missing"]}
    daily = read_dataset_manifest(daily_path)
    status = read_dataset_manifest(status_path)
    daily_paths = [str((root / shard.path).resolve() if not Path(shard.path).is_absolute() else Path(shard.path).resolve()) for shard in daily.shards]
    status_paths = [str((root / shard.path).resolve() if not Path(shard.path).is_absolute() else Path(shard.path).resolve()) for shard in status.shards]
    import duckdb  # type: ignore

    with duckdb.connect(":memory:") as con:
        frame = con.execute(
            f"""
            with daily as (
              select
                symbol,
                trade_date,
                close,
                lag(close) over (partition by symbol order by trade_date) as prev_close
              from read_parquet({_path_list_sql(daily_paths)}, union_by_name=true)
              where trade_date >= ? and trade_date <= ?
            ),
            st as (
              select symbol, trade_date, is_st
              from read_parquet({_path_list_sql(status_paths)}, union_by_name=true)
            ),
            calc as (
              select
                d.symbol,
                d.trade_date,
                round(cast(d.prev_close as double) * (1 + case when coalesce(st.is_st, false) then 0.05 else 0.10 end), 2) as up_limit,
                round(cast(d.prev_close as double) * (1 - case when coalesce(st.is_st, false) then 0.05 else 0.10 end), 2) as down_limit,
                cast(d.close as double) as close
              from daily d
              left join st using(symbol, trade_date)
              where d.prev_close is not null and d.prev_close > 0 and d.close > 0
            )
            select
              symbol,
              trade_date,
              up_limit,
              down_limit,
              close >= up_limit - 0.011 as is_limit_up,
              close <= down_limit + 0.011 as is_limit_down,
              'qdp_v2_daily_close_limit_rule' as source
            from calc
            where close >= up_limit - 0.011 or close <= down_limit + 0.011
            order by trade_date, symbol
            """,
            [start_date, end_date],
        ).fetchdf()
    normalized = normalize_domain_frame(frame, domain=DataDomain.LIMIT_STATUS, source="qdp_v2_daily_close_limit_rule", as_of_date=end_date, require_columns=False)
    if symbols:
        normalized = normalized.loc[normalized["symbol"].isin(set(symbols))].copy()
    normalized = _dedupe_limit_status(normalized)
    return normalized, {
        "provider": "qdp_v2_derived",
        "method": "daily_close_limit_rule",
        "scope": "close_at_limit_events_only",
        "rule_note": "Mainboard close-to-limit approximation using prior close and historical ST flag; it does not claim intraday touch/open-board events.",
        "error_count": 0,
        "errors": [],
    }


def _fetch_announcements(*, symbols: list[str], dates: list[str], workers: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    import akshare as ak  # type: ignore

    symbol_set = set(symbols)

    def fetch_day(trade_date: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        try:
            raw = ak.stock_notice_report(symbol="全部", date=trade_date.replace("-", ""))
        except Exception as exc:
            if type(exc).__name__ == "KeyError" and "代码" in str(exc):
                return pd.DataFrame(), [{"date": trade_date, "status": "empty_schema", "message": "no_standard_notice_rows"}]
            return pd.DataFrame(), [{"date": trade_date, "error_type": type(exc).__name__, "message": str(exc)}]
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            return pd.DataFrame(), []
        frame = pd.DataFrame(
            {
                "symbol": raw.get("代码"),
                "trade_date": raw.get("公告日期", trade_date),
                "title": raw.get("公告标题"),
                "url": raw.get("网址"),
                "category": raw.get("公告类型"),
                "source": "akshare_eastmoney_notice",
            }
        )
        return frame, []

    frame, errors = _parallel_collect(dates, fetch_day, workers=workers)
    empty_schema = [item for item in errors if item.get("status") == "empty_schema"]
    real_errors = [item for item in errors if item.get("status") != "empty_schema"]
    normalized = normalize_domain_frame(frame, domain=DataDomain.ANNOUNCEMENT, source="akshare_eastmoney_notice", as_of_date=dates[-1] if dates else "", require_columns=False)
    normalized = normalized.loc[normalized["symbol"].isin(symbol_set)].copy()
    return normalized, {
        "provider": "akshare_eastmoney_notice",
        "date_count": len(dates),
        "empty_schema_date_count": len(empty_schema),
        "empty_schema_examples": empty_schema[:20],
        "error_count": len(real_errors),
        "errors": real_errors[:50],
    }


def _fetch_corporate_actions(*, symbols: list[str], start_date: str, end_date: str, workers: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    import akshare as ak  # type: ignore

    def fetch_symbol(symbol: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        code = _plain_code(symbol)
        try:
            raw = ak.stock_dividend_cninfo(symbol=code)
        except Exception as exc:
            if type(exc).__name__ == "KeyError" and "实施方案公告日期" in str(exc):
                return pd.DataFrame(), [{"symbol": symbol, "status": "empty_schema", "message": "no_standard_dividend_rows"}]
            return pd.DataFrame(), [{"symbol": symbol, "error_type": type(exc).__name__, "message": str(exc)}]
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            return pd.DataFrame(), []
        raw = raw.copy()
        raw["symbol"] = symbol
        raw["source"] = "akshare_cninfo_dividend"
        return raw, []

    frame, errors = _parallel_collect(symbols, fetch_symbol, workers=workers)
    empty_schema = [item for item in errors if item.get("status") == "empty_schema"]
    real_errors = [item for item in errors if item.get("status") != "empty_schema"]
    normalized = normalize_domain_frame(frame, domain=DataDomain.CORPORATE_ACTIONS, source="akshare_cninfo_dividend", as_of_date=end_date, require_columns=False)
    normalized = _filter_date_window(normalized, start_date, end_date)
    return normalized, {
        "provider": "akshare_cninfo_dividend",
        "symbol_count": len(symbols),
        "empty_schema_symbol_count": len(empty_schema),
        "empty_schema_examples": empty_schema[:20],
        "error_count": len(real_errors),
        "errors": real_errors[:50],
    }


def _fetch_share_capital(*, symbols: list[str], start_date: str, end_date: str, workers: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    import akshare as ak  # type: ignore

    start_arg = start_date.replace("-", "")
    end_arg = end_date.replace("-", "")

    def fetch_symbol(symbol: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        code = _plain_code(symbol)
        try:
            raw = ak.stock_share_change_cninfo(symbol=code, start_date=start_arg, end_date=end_arg)
        except Exception as exc:
            if type(exc).__name__ == "KeyError" and "公告日期" in str(exc):
                return pd.DataFrame(), [{"symbol": symbol, "status": "empty_schema", "message": "no_standard_share_change_rows"}]
            return pd.DataFrame(), [{"symbol": symbol, "error_type": type(exc).__name__, "message": str(exc)}]
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            return pd.DataFrame(), []
        frame = raw.copy()
        frame["symbol"] = symbol
        for column in ["总股本", "已流通股份", "人民币普通股", "流通受限股份"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce") * 10000.0
        frame["source"] = "akshare_cninfo_share_change"
        return frame, []

    frame, errors = _parallel_collect(symbols, fetch_symbol, workers=workers)
    empty_schema = [item for item in errors if item.get("status") == "empty_schema"]
    real_errors = [item for item in errors if item.get("status") != "empty_schema"]
    normalized = normalize_domain_frame(frame, domain=DataDomain.SHARE_CAPITAL, source="akshare_cninfo_share_change", as_of_date=end_date, require_columns=False)
    normalized = _filter_date_window(normalized, start_date, end_date)
    return normalized, {
        "provider": "akshare_cninfo_share_change",
        "symbol_count": len(symbols),
        "share_unit": "shares",
        "empty_schema_symbol_count": len(empty_schema),
        "empty_schema_examples": empty_schema[:20],
        "error_count": len(real_errors),
        "errors": real_errors[:50],
    }


def _fetch_name_change(*, symbols: list[str], start_date: str, end_date: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    import akshare as ak  # type: ignore

    frames: list[pd.DataFrame] = []
    errors: list[dict[str, Any]] = []
    for change_type in ("简称变更", "全称变更"):
        try:
            raw = ak.stock_info_sz_change_name(symbol=change_type)
        except Exception as exc:
            errors.append({"change_type": change_type, "error_type": type(exc).__name__, "message": str(exc)})
            continue
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            continue
        frame = raw.copy()
        frame["change_type"] = "short_name" if change_type == "简称变更" else "full_name"
        frame["source"] = "akshare_szse_name_change"
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    normalized = normalize_domain_frame(raw, domain=DataDomain.NAME_CHANGE, source="akshare_szse_name_change", as_of_date=end_date, require_columns=False)
    normalized = normalized.loc[normalized["symbol"].isin(set(symbols))].copy()
    normalized = _filter_date_window(normalized, start_date, end_date)
    return normalized, {"provider": "akshare_szse_name_change", "coverage_note": "SZSE official batch endpoint; SH dated name history not available through this low-request fetcher", "error_count": len(errors), "errors": errors[:50]}


def _fetch_baostock_report_domain(*, domain: str, symbols: list[str], start_date: str, end_date: str, chunk_size: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    from quant_data_platform.providers import BaostockProvider

    provider = BaostockProvider()
    frames: list[pd.DataFrame] = []
    errors: list[dict[str, Any]] = []
    chunks = [symbols[index : index + chunk_size] for index in range(0, len(symbols), chunk_size)]
    for index, chunk in enumerate(chunks, start=1):
        try:
            result = provider.fetch_domain(
                DomainFetchRequest(
                    domain=domain,
                    symbols=tuple(chunk),
                    start_date=start_date,
                    end_date=end_date,
                )
            )
            if result.data is not None and not result.data.empty:
                frames.append(result.data)
        except Exception as exc:
            errors.append({"chunk_index": index, "symbol_count": len(chunk), "error_type": type(exc).__name__, "message": str(exc)})
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    normalized = normalize_domain_frame(frame, domain=domain, source="baostock", as_of_date=end_date, require_columns=False)
    normalized = _filter_date_window(normalized, start_date, end_date)
    return normalized, {"provider": "baostock", "chunk_count": len(chunks), "chunk_size": chunk_size, "error_count": len(errors), "errors": errors[:50]}


def _parallel_collect(items: list[str], fn: Callable[[str], tuple[pd.DataFrame, list[dict[str, Any]]]], *, workers: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    errors: list[dict[str, Any]] = []
    if not items:
        return pd.DataFrame(), []
    with ThreadPoolExecutor(max_workers=max(1, int(workers or 1))) as executor:
        futures = {executor.submit(fn, item): item for item in items}
        for future in as_completed(futures):
            try:
                frame, item_errors = future.result()
                if isinstance(frame, pd.DataFrame) and not frame.empty:
                    frames.append(frame)
                errors.extend(list(item_errors or []))
            except Exception as exc:
                errors.append({"item": futures[future], "error_type": type(exc).__name__, "message": str(exc)})
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), errors


def _write_domain_dataset(*, root: Path, domain: str, frame: pd.DataFrame, source_report: Mapping[str, Any]) -> tuple[DatasetManifest, Path]:
    primary_key = list(PRIMARY_KEYS[domain])
    frame = frame.copy()
    start_date = str(frame["trade_date"].min()) if "trade_date" in frame.columns and not frame.empty else ""
    end_date = str(frame["trade_date"].max()) if "trade_date" in frame.columns and not frame.empty else ""
    duplicate_key_rows = int(frame.duplicated(subset=[col for col in primary_key if col in frame.columns]).sum()) if primary_key else 0
    if duplicate_key_rows:
        frame = frame.drop_duplicates(subset=[col for col in primary_key if col in frame.columns], keep="last").reset_index(drop=True)
        duplicate_key_rows = int(frame.duplicated(subset=[col for col in primary_key if col in frame.columns]).sum())
    dataset_id = f"{domain}__{stable_hash({'domain': domain, 'contract': CONTRACTS[domain], 'start': start_date, 'end': end_date, 'rows': len(frame), 'source': source_report.get('provider', '')})}"
    target_dir = root / "datasets" / domain / dataset_id
    staging = target_dir / ".staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging_shards = staging / "shards"
    staging_shards.mkdir(parents=True, exist_ok=True)
    target = staging_shards / f"part_000000_{domain}.parquet"
    frame.to_parquet(target, index=False)
    schema, row_count = _parquet_schema_and_rows(target)
    final_shards = target_dir / "shards"
    target_dir.mkdir(parents=True, exist_ok=True)
    if final_shards.exists():
        shutil.rmtree(final_shards)
    staging_shards.replace(final_shards)
    shutil.rmtree(staging, ignore_errors=True)
    final_path = final_shards / target.name
    shard = ShardManifestEntry(
        path=path_for_manifest(final_path, root=root),
        row_count=int(row_count),
        start_date=start_date,
        end_date=end_date,
        status="stored",
        file_size=int(final_path.stat().st_size),
        schema_hash=schema_hash(schema),
        source_path="",
        content_key=CONTRACTS[domain],
        metadata={"source_report": dict(source_report)},
    )
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        domain=domain,
        layer="raw",
        frequency="event" if domain in {DataDomain.ANNOUNCEMENT, DataDomain.CORPORATE_ACTIONS, DataDomain.NAME_CHANGE} else "1d",
        contract_version=CONTRACTS[domain],
        primary_key=primary_key,
        start_date=start_date,
        end_date=end_date,
        row_count=int(row_count),
        schema_hash=schema_hash(schema),
        schema=schema,
        shards=[shard],
        source={"provider": source_report.get("provider", ""), "created_by": "qdp supplement recommended", "created_at": utc_now()},
        quality={"path_refs_exist": True, "primary_key_unique": duplicate_key_rows == 0, "duplicate_key_rows": duplicate_key_rows},
    )
    manifest_path = write_dataset_manifest(root, manifest)
    return manifest, manifest_path


def _parquet_schema_and_rows(path: Path) -> tuple[list[dict[str, str]], int]:
    import pyarrow.parquet as pq  # type: ignore

    parquet_file = pq.ParquetFile(path)
    schema = [{"name": field.name, "type": str(field.type)} for field in parquet_file.schema_arrow]
    return schema, int(parquet_file.metadata.num_rows)


def _active_symbols(*, root: Path, active: Mapping[str, Any], max_symbols: int = 0) -> list[str]:
    raw = dict(active.get("raw", {}) or {})
    dataset_id = str(raw.get("market_daily_raw", "") or "")
    manifest_path = dataset_manifest_for_id(root, dataset_id, "market_daily_raw")
    if manifest_path is None:
        return []
    manifest = read_dataset_manifest(manifest_path)
    paths = [str((root / shard.path).resolve() if not Path(shard.path).is_absolute() else Path(shard.path).resolve()) for shard in manifest.shards]
    import duckdb  # type: ignore

    with duckdb.connect(":memory:") as con:
        rows = con.execute(f"select distinct symbol from read_parquet({_path_list_sql(paths)}, union_by_name=true) order by symbol").fetchall()
    symbols = [str(row[0]) for row in rows if row and str(row[0]).strip()]
    return symbols[: int(max_symbols)] if int(max_symbols or 0) > 0 else symbols


def _calendar_dates(*, root: Path, active: Mapping[str, Any], start_date: str, end_date: str, max_dates: int = 0, include_calendar_days: bool = False) -> list[str]:
    if include_calendar_days:
        dates = [item.strftime("%Y-%m-%d") for item in pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")]
        return dates[: int(max_dates)] if int(max_dates or 0) > 0 else dates
    raw = dict(active.get("raw", {}) or {})
    dataset_id = str(raw.get("trading_calendar", "") or "")
    manifest_path = dataset_manifest_for_id(root, dataset_id, "trading_calendar")
    if manifest_path is None:
        return []
    manifest = read_dataset_manifest(manifest_path)
    paths = [str((root / shard.path).resolve() if not Path(shard.path).is_absolute() else Path(shard.path).resolve()) for shard in manifest.shards]
    import duckdb  # type: ignore

    with duckdb.connect(":memory:") as con:
        rows = con.execute(
            f"""
            select distinct trade_date
            from read_parquet({_path_list_sql(paths)}, union_by_name=true)
            where (is_open = true or is_open = 1)
              and trade_date >= ?
              and trade_date <= ?
            order by trade_date
            """,
            [start_date, end_date],
        ).fetchall()
    dates = [str(row[0]) for row in rows if row and str(row[0]).strip()]
    return dates[: int(max_dates)] if int(max_dates or 0) > 0 else dates


def _filter_date_window(frame: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if frame.empty or "trade_date" not in frame.columns:
        return frame
    dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    mask = dates.ge(pd.Timestamp(start_date)) & dates.le(pd.Timestamp(end_date))
    return frame.loc[mask].reset_index(drop=True)


def _dedupe_limit_status(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    grouped = (
        frame.groupby(["trade_date", "symbol"], as_index=False)
        .agg(
            {
                "up_limit": "max",
                "down_limit": "min",
                "is_limit_up": "max",
                "is_limit_down": "max",
                "source": lambda values: ",".join(sorted(set(str(item) for item in values if str(item)))),
            }
        )
        .sort_values(["trade_date", "symbol"])
        .reset_index(drop=True)
    )
    return grouped


def _plain_code(symbol: str) -> str:
    return str(symbol or "").strip().upper().split(".", 1)[0]


def _path_list_sql(paths: list[str]) -> str:
    return "[" + ", ".join("'" + str(path).replace("'", "''").replace("\\", "/") + "'" for path in paths) + "]"


def _write_run(root: Path, prefix: str, payload: Mapping[str, Any]) -> Path:
    run_path = root / "runs" / f"{prefix}_{utc_now().replace(':', '').replace('-', '')}.json"
    return atomic_write_json(run_path, payload)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp supplement recommended", description="Fetch and register qdp_v2 recommended raw domains.")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--domains", default="all", help="Comma-separated domains, all, or low_request.")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--runtime", default="balanced", choices=("safe", "balanced", "fast"))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--max-dates", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--include-calendar-days", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = build_recommended_domains(
        workspace_root=str(args.workspace_root or "") or None,
        domains=str(args.domains or "all").split(","),
        start_date=str(args.start_date or ""),
        end_date=str(args.end_date or ""),
        runtime=str(args.runtime or "balanced"),
        workers=int(args.workers or 0),
        max_symbols=int(args.max_symbols or 0),
        max_dates=int(args.max_dates or 0),
        chunk_size=int(args.chunk_size or 200),
        include_calendar_days=bool(args.include_calendar_days),
        dry_run=bool(args.dry_run),
        activate=bool(args.activate),
        force=bool(args.force),
    )
    if bool(args.json):
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print(_format(payload))
    return 0 if str(payload.get("status", "")) in {"ok", "planned"} else 2


def _format(payload: Mapping[str, Any]) -> str:
    lines = [
        f"status: {payload.get('status')}",
        f"domains: {','.join(str(item) for item in payload.get('domains', []) or [])}",
        f"symbol_count: {payload.get('symbol_count', 0)}",
        f"date_count: {payload.get('date_count', 0)}",
    ]
    for item in list(payload.get("results", []) or []):
        lines.append(f"{item.get('domain')}: {item.get('status')} rows={item.get('row_count', 0)} dataset={item.get('dataset_id', '')}")
    if payload.get("active_manifest"):
        lines.append(f"active_manifest: {payload.get('active_manifest')}")
    if payload.get("run_manifest"):
        lines.append(f"run_manifest: {payload.get('run_manifest')}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
