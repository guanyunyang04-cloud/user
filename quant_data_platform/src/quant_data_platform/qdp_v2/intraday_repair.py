from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.domains.contracts import DataDomain, DomainFetchRequest, normalize_domain_frame
from quant_data_platform.ingest.import_external_quant_zip import normalize_intraday_1m_to_mootdx_240_frame
from quant_data_platform.providers import MootdxOnlineProvider
from quant_data_platform.qdp_v2.environment import runtime_environment
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    atomic_write_json,
    dataset_manifest_for_id,
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


PRICE_COLUMNS = ("open", "high", "low", "close")
ONE_MINUTE_COLUMNS = [
    "symbol",
    "trade_date",
    "bar_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover_rate",
    "float_share",
    "total_share",
    "source",
    "adjusted_flag",
]


def repair_intraday_zero_bars(
    *,
    workspace_root: str | Path | None = None,
    source_dataset_id: str = "",
    external_source_root: str | Path = r"H:\BaiduNetdiskDownload\量化数据",
    years: tuple[int, ...] = (2023, 2024, 2026),
    runtime: str = "balanced",
    duckdb_memory_limit: str = "",
    dry_run: bool = False,
    activate_domain: bool = False,
    no_mootdx: bool = False,
    max_repair_symbol_days: int = 0,
    copy_mode: str = "hardlink",
) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    source_manifest_path = _resolve_source_manifest(root, source_dataset_id)
    source = read_dataset_manifest(source_manifest_path)
    profile = resolve_runtime_profile(runtime)
    memory_limit = str(duckdb_memory_limit or "").strip() or profile.duckdb_memory_limit
    selected_shards = _select_shards_by_year(root=root, manifest=source, years=years)
    run_id = f"intraday_zero_repair_{utc_now().replace(':', '').replace('-', '')}"
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    issue_rows = _collect_zero_price_rows(
        shard_paths=[item["path"] for item in selected_shards],
        memory_limit=memory_limit,
        threads=max(1, int(profile.duckdb_threads or 1)),
    )
    if issue_rows.empty:
        payload = {
            "status": "ok",
            "message": "no_zero_price_rows_found",
            "source_dataset_id": source.dataset_id,
            "selected_shards": len(selected_shards),
            "years": list(years),
            "runtime_environment": runtime_environment(),
        }
        atomic_write_json(run_dir / "result.json", payload)
        return payload
    issues = _classify_issue_symbol_days(root=root, issue_rows=issue_rows)
    if max_repair_symbol_days:
        issues = issues.sort_values(["trade_date", "symbol"]).head(int(max_repair_symbol_days)).reset_index(drop=True)
        issue_rows = issue_rows.merge(issues[["symbol", "trade_date"]].drop_duplicates(), on=["symbol", "trade_date"], how="inner")
    issues["action"] = issues["repair_class"].map(
        {
            "suspended_with_zero_bars": "drop",
            "traded_day_partial_zero_bars": "replace",
            "no_daily_row_not_suspended": "drop",
        }
    ).fillna("review")
    issues["repair_provider"] = issues.apply(_repair_provider_for_issue, axis=1)
    issue_hash = stable_hash(
        {
            "source_dataset_id": source.dataset_id,
            "issues": issues[["symbol", "trade_date", "repair_class", "action", "repair_provider"]].to_dict("records"),
        },
        length=24,
    )
    target_dataset_id = f"market_intraday_1m__{stable_hash({'source': source.dataset_id, 'repair': 'zero_price_v1', 'issue_hash': issue_hash})}"
    issues_path = run_dir / "intraday_zero_issues.csv"
    issue_rows_path = run_dir / "intraday_zero_bad_rows.parquet"
    issues.to_csv(issues_path, index=False, encoding="utf-8-sig")
    issue_rows.to_parquet(issue_rows_path, index=False)
    external_root = Path(external_source_root)
    replacement_result = _build_replacement_rows(
        issues=issues,
        external_source_root=external_root,
        use_mootdx=not bool(no_mootdx),
        runtime=runtime,
        run_dir=run_dir,
    )
    replacements = replacement_result["rows"]
    unresolved = list(replacement_result["unresolved"])
    if not replacements.empty:
        replacements_path = run_dir / "intraday_zero_replacements.parquet"
        replacements.to_parquet(replacements_path, index=False)
    else:
        replacements_path = None
    blockers = _repair_blockers(issues=issues, replacements=replacements, unresolved=unresolved)
    affected_shards = sorted(set(str(item) for item in issue_rows["shard_path"].dropna().astype(str).unique()))
    dry_payload = {
        "status": "planned" if not blockers else "blocked",
        "dry_run": bool(dry_run),
        "run_id": run_id,
        "source_dataset_id": source.dataset_id,
        "target_dataset_id": target_dataset_id,
        "selected_shards": len(selected_shards),
        "affected_shards": len(affected_shards),
        "zero_price_rows": int(len(issue_rows)),
        "issue_symbol_days": int(len(issues)),
        "issues_by_class": _count_records(issues, "repair_class"),
        "issues_by_provider": _count_records(issues, "repair_provider"),
        "replacement_rows": int(len(replacements)),
        "replacement_symbol_days": int(replacements[["symbol", "trade_date"]].drop_duplicates().shape[0]) if not replacements.empty else 0,
        "unresolved": unresolved[:50],
        "unresolved_count": len(unresolved),
        "blockers": blockers,
        "issues_path": str(issues_path.resolve()),
        "bad_rows_path": str(issue_rows_path.resolve()),
        "replacements_path": str(replacements_path.resolve()) if replacements_path else "",
        "runtime_environment": runtime_environment(),
    }
    if dry_run or blockers:
        atomic_write_json(run_dir / "result.json", dry_payload)
        return dry_payload
    target_dir = root / "datasets" / "market_intraday_1m" / target_dataset_id
    staging_dir = target_dir / ".staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_shards = staging_dir / "shards"
    staging_shards.mkdir(parents=True, exist_ok=True)
    keys = issues[["shard_path", "symbol", "trade_date", "repair_class", "action"]].copy()
    replacements_with_shards = _attach_shard_paths_to_replacements(replacements, keys)
    entries: list[ShardManifestEntry] = []
    affected_set = set(affected_shards)
    rewritten = 0
    linked = 0
    dropped_rows = 0
    inserted_rows = 0
    for index, shard in enumerate(source.shards):
        source_path = _resolve_data_path(root, shard.path)
        target_path = staging_shards / Path(shard.path).name
        if str(source_path.resolve()) in affected_set:
            shard_keys = keys.loc[keys["shard_path"].eq(str(source_path.resolve()))].copy()
            shard_replacements = replacements_with_shards.loc[
                replacements_with_shards["shard_path"].eq(str(source_path.resolve()))
            ].copy()
            stats = _rewrite_1m_shard(
                source_path=source_path,
                target_path=target_path,
                remove_keys=shard_keys,
                replacements=shard_replacements,
                columns=[item["name"] for item in source.schema] or ONE_MINUTE_COLUMNS,
            )
            rewritten += 1
            dropped_rows += int(stats["dropped_rows"])
            inserted_rows += int(stats["inserted_rows"])
            entry = _entry_for_repaired_shard(
                target_dataset_id=target_dataset_id,
                target_path=target_path,
                source_path=source_path,
                source_shard=shard,
                row_count=int(stats["row_count"]),
            )
        else:
            _copy_or_link_file(source_path, target_path, mode=copy_mode)
            linked += 1
            entry = ShardManifestEntry(
                path=f"datasets/market_intraday_1m/{target_dataset_id}/shards/{target_path.name}",
                row_count=shard.row_count,
                start_date=shard.start_date,
                end_date=shard.end_date,
                status="stored",
                file_size=int(target_path.stat().st_size),
                schema_hash=shard.schema_hash,
                source_path=str(source_path.resolve()),
                content_key=shard.content_key,
                metadata={**dict(shard.metadata or {}), "repair_copy_mode": copy_mode},
            )
        entries.append(entry)
    final_shards = target_dir / "shards"
    target_dir.mkdir(parents=True, exist_ok=True)
    if final_shards.exists():
        shutil.rmtree(final_shards)
    staging_shards.replace(final_shards)
    shutil.rmtree(staging_dir, ignore_errors=True)
    schema = _parquet_schema(final_shards / Path(entries[0].path).name) if entries else list(source.schema)
    manifest = DatasetManifest(
        dataset_id=target_dataset_id,
        domain="market_intraday_1m",
        layer="raw",
        frequency="1m",
        contract_version="mootdx_1m_240_v1",
        primary_key=["trade_date", "symbol", "bar_time"],
        start_date=min([item.start_date for item in entries if item.start_date], default=""),
        end_date=max([item.end_date for item in entries if item.end_date], default=""),
        row_count=sum(int(item.row_count) for item in entries),
        schema_hash=schema_hash(schema),
        schema=schema,
        shards=entries,
        source={
            "provider": "qdp_v2",
            "created_by": "repair_intraday_zero_bars",
            "created_at": utc_now(),
            "source_dataset_id": source.dataset_id,
            "issue_hash": issue_hash,
        },
        quality={
            "path_refs_exist": True,
            "zero_price_repair": "applied",
            "zero_price_rows_before": int(len(issue_rows)),
            "affected_symbol_days": int(len(issues)),
            "primary_key_unique": "not_checked_after_repair",
        },
        notes=[
            "zero-price suspended intraday pseudo bars removed",
            "traded zero-price symbol-days replaced from external CSV or mootdx when available",
        ],
    )
    manifest_path = write_dataset_manifest(root, manifest)
    active_path = ""
    if activate_domain:
        active = read_active_manifest(root)
        raw = dict(active.get("raw", {}) or {})
        raw["market_intraday_1m"] = target_dataset_id
        active["raw"] = raw
        active.setdefault("source", {})
        active["updated_at"] = utc_now()
        active_path = str(write_active_manifest(root, active).resolve())
    payload = {
        **dry_payload,
        "status": "ok",
        "dry_run": False,
        "manifest_path": str(manifest_path.resolve()),
        "row_count": manifest.row_count,
        "start_date": manifest.start_date,
        "end_date": manifest.end_date,
        "rewritten_shards": rewritten,
        "linked_shards": linked,
        "dropped_rows": dropped_rows,
        "inserted_rows": inserted_rows,
        "active_manifest": active_path,
    }
    atomic_write_json(run_dir / "result.json", payload)
    return payload


def _resolve_source_manifest(root: Path, source_dataset_id: str) -> Path:
    dataset_id = str(source_dataset_id or "").strip()
    if not dataset_id:
        active = read_active_manifest(root)
        dataset_id = str(dict(active.get("raw", {}) or {}).get("market_intraday_1m", "") or "")
    path = dataset_manifest_for_id(root, dataset_id, "market_intraday_1m")
    if path is None:
        raise FileNotFoundError(f"market_intraday_1m dataset manifest not found: {dataset_id}")
    return path


def _select_shards_by_year(*, root: Path, manifest: DatasetManifest, years: tuple[int, ...]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    year_set = {int(item) for item in years if int(item) > 0}
    for shard in manifest.shards:
        start_year = _year(shard.start_date)
        end_year = _year(shard.end_date)
        if year_set and not any(start_year <= year <= end_year for year in year_set):
            continue
        selected.append({"path": _resolve_data_path(root, shard.path), "shard": shard})
    return selected


def _collect_zero_price_rows(*, shard_paths: list[Path], memory_limit: str, threads: int) -> pd.DataFrame:
    if not shard_paths:
        return pd.DataFrame(columns=["shard_path", "symbol", "trade_date", "bar_time", "source"])
    import duckdb  # type: ignore

    paths = [str(Path(item).resolve()) for item in shard_paths]
    with duckdb.connect(":memory:") as con:
        con.execute(f"set memory_limit='{memory_limit}'")
        con.execute(f"set threads={max(1, int(threads))}")
        rows = con.execute(
            """
            select
              filename as shard_path,
              symbol,
              trade_date,
              bar_time,
              source,
              open,
              high,
              low,
              close,
              volume,
              amount
            from read_parquet(?, union_by_name=true, filename=true)
            where open <= 0 or high <= 0 or low <= 0 or close <= 0
            """,
            [paths],
        ).fetchdf()
    if rows.empty:
        return rows
    rows["shard_path"] = rows["shard_path"].map(lambda value: str(Path(str(value)).resolve()))
    rows["symbol"] = rows["symbol"].astype(str).str.upper()
    rows["trade_date"] = rows["trade_date"].astype(str)
    rows["bar_time"] = rows["bar_time"].astype(str).str.zfill(9)
    return rows


def _classify_issue_symbol_days(*, root: Path, issue_rows: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        issue_rows.groupby(["shard_path", "symbol", "trade_date"], dropna=False)
        .agg(
            bad_rows=("bar_time", "size"),
            first_bad_bar=("bar_time", "min"),
            last_bad_bar=("bar_time", "max"),
            sources=("source", lambda values: ",".join(sorted({str(item) for item in values if str(item)}))),
        )
        .reset_index()
    )
    keys = grouped[["symbol", "trade_date"]].drop_duplicates().reset_index(drop=True)
    daily = _lookup_daily_rows(root, keys)
    status = _lookup_security_status(root, keys)
    classified = grouped.merge(daily, on=["symbol", "trade_date"], how="left").merge(status, on=["symbol", "trade_date"], how="left")
    classified["has_daily_bar"] = classified["daily_open"].notna() & classified["daily_open"].gt(0) & classified["daily_volume"].fillna(0).gt(0)
    classified["is_suspended"] = classified["is_suspended"].fillna(False).astype(bool)
    classified["repair_class"] = "no_daily_row_not_suspended"
    classified.loc[classified["has_daily_bar"], "repair_class"] = "traded_day_partial_zero_bars"
    classified.loc[classified["is_suspended"], "repair_class"] = "suspended_with_zero_bars"
    classified["year"] = classified["trade_date"].astype(str).str.slice(0, 4).astype(int)
    return classified.sort_values(["trade_date", "symbol", "shard_path"]).reset_index(drop=True)


def _lookup_daily_rows(root: Path, keys: pd.DataFrame) -> pd.DataFrame:
    active = read_active_manifest(root)
    dataset_id = str(dict(active.get("raw", {}) or {}).get("market_daily_raw", "") or "")
    manifest_path = dataset_manifest_for_id(root, dataset_id, "market_daily_raw")
    if manifest_path is None:
        out = keys.copy()
        out["daily_open"] = pd.NA
        out["daily_volume"] = pd.NA
        return out
    manifest = read_dataset_manifest(manifest_path)
    return _lookup_parquet_values(
        root=root,
        manifest=manifest,
        keys=keys,
        select_sql="d.open as daily_open, d.volume as daily_volume",
    )


def _lookup_security_status(root: Path, keys: pd.DataFrame) -> pd.DataFrame:
    active = read_active_manifest(root)
    dataset_id = str(dict(active.get("raw", {}) or {}).get("security_status", "") or "")
    manifest_path = dataset_manifest_for_id(root, dataset_id, "security_status")
    if manifest_path is None:
        out = keys.copy()
        out["is_suspended"] = False
        return out
    manifest = read_dataset_manifest(manifest_path)
    return _lookup_parquet_values(
        root=root,
        manifest=manifest,
        keys=keys,
        select_sql="coalesce(d.is_suspended, false) as is_suspended",
    )


def _lookup_parquet_values(*, root: Path, manifest: DatasetManifest, keys: pd.DataFrame, select_sql: str) -> pd.DataFrame:
    import duckdb  # type: ignore

    paths = [str(_resolve_data_path(root, item.path)) for item in manifest.shards]
    with duckdb.connect(":memory:") as con:
        con.register("keys", keys)
        return con.execute(
            f"""
            select k.symbol, k.trade_date, {select_sql}
            from keys k
            left join read_parquet(?, union_by_name=true) d
              on d.symbol = k.symbol and d.trade_date = k.trade_date
            """,
            [paths],
        ).fetchdf()


def _repair_provider_for_issue(row: pd.Series) -> str:
    if str(row.get("repair_class", "")) == "suspended_with_zero_bars":
        return "none_drop_suspended"
    if str(row.get("repair_class", "")) == "no_daily_row_not_suspended":
        return "none_drop_no_daily"
    year = int(row.get("year", 0) or 0)
    if year <= 2024:
        return "external_csv"
    return "mootdx"


def _build_replacement_rows(
    *,
    issues: pd.DataFrame,
    external_source_root: Path,
    use_mootdx: bool,
    runtime: str,
    run_dir: Path,
) -> dict[str, Any]:
    replace = issues.loc[issues["action"].isin(["replace", "replace_or_drop_if_unavailable"])].copy()
    frames: list[pd.DataFrame] = []
    unresolved: list[dict[str, Any]] = []
    external_issues = replace.loc[replace["repair_provider"].eq("external_csv")].copy()
    if not external_issues.empty:
        external_frame, external_unresolved = _external_replacements(external_issues, external_source_root)
        if not external_frame.empty:
            frames.append(external_frame)
        unresolved.extend(external_unresolved)
    mootdx_issues = replace.loc[replace["repair_provider"].eq("mootdx")].copy()
    if not mootdx_issues.empty and use_mootdx:
        mootdx_frame, mootdx_unresolved = _mootdx_replacements(mootdx_issues, runtime=runtime)
        if not mootdx_frame.empty:
            frames.append(mootdx_frame)
        unresolved.extend(mootdx_unresolved)
        missing = _missing_replacement_keys(mootdx_issues, mootdx_frame)
        if missing and external_source_root.exists():
            fallback_issues = pd.DataFrame(missing)
            fallback_frame, fallback_unresolved = _external_replacements(fallback_issues, external_source_root)
            if not fallback_frame.empty:
                frames.append(fallback_frame)
                (run_dir / "mootdx_external_fallback_used.txt").write_text(
                    json.dumps(json_safe(missing), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            unresolved.extend(fallback_unresolved)
    elif not mootdx_issues.empty:
        unresolved.extend(
            {
                "symbol": str(row.symbol),
                "trade_date": str(row.trade_date),
                "reason": "mootdx_disabled",
                "required": str(row.action),
            }
            for row in mootdx_issues.itertuples(index=False)
        )
    if frames:
        rows = pd.concat(frames, ignore_index=True)
        rows = _normalize_replacement_frame(rows)
    else:
        rows = pd.DataFrame(columns=ONE_MINUTE_COLUMNS)
    unresolved = _dedupe_unresolved(unresolved)
    return {"rows": rows, "unresolved": unresolved}


def _external_replacements(issues: pd.DataFrame, external_source_root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    unresolved: list[dict[str, Any]] = []
    for (year, symbol), group in issues.groupby(["year", "symbol"], dropna=False):
        dates = sorted({str(item) for item in group["trade_date"].tolist()})
        csv_path = _find_external_csv(external_source_root, int(year), str(symbol))
        if csv_path is None:
            unresolved.extend({"symbol": str(symbol), "trade_date": date, "reason": "external_csv_missing"} for date in dates)
            continue
        try:
            frame = _read_external_1m_csv(csv_path, symbol=str(symbol), dates=dates)
        except Exception as exc:
            unresolved.extend(
                {"symbol": str(symbol), "trade_date": date, "reason": "external_csv_read_failed", "error": str(exc)}
                for date in dates
            )
            continue
        frames.append(frame)
        present = set(frame["trade_date"].astype(str).unique()) if not frame.empty else set()
        for date in dates:
            if date not in present:
                unresolved.append({"symbol": str(symbol), "trade_date": date, "reason": "external_csv_date_missing", "path": str(csv_path)})
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ONE_MINUTE_COLUMNS)
    rows = _normalize_replacement_frame(rows)
    unresolved.extend(_invalid_replacement_records(issues, rows, provider="external_csv"))
    return rows, unresolved


def _find_external_csv(root: Path, year: int, symbol: str) -> Path | None:
    code = str(symbol).split(".")[0].zfill(6)
    suffix = str(symbol).split(".")[-1].upper()
    prefix = "sh" if suffix == "SH" else "sz" if suffix == "SZ" else "bj"
    year_roots = [root / str(year)]
    if root.exists():
        year_roots.extend(sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith(str(year)) and path not in year_roots))
    candidates = [
        year_root / str(year) / f"{prefix}{code}.csv"
        for year_root in year_roots
    ]
    candidates.extend(year_root / f"{prefix}{code}.csv" for year_root in year_roots)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches: list[Path] = []
    for year_root in year_roots:
        if year_root.exists():
            matches.extend(year_root.rglob(f"{prefix}{code}.csv"))
    return matches[0] if matches else None


def _read_external_1m_csv(path: Path, *, symbol: str, dates: list[str]) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    raw["symbol"] = str(symbol).upper()
    if "日期" in raw.columns:
        day = raw["日期"].astype(str).str.slice(0, 10)
    elif "datetime" in raw.columns:
        day = raw["datetime"].astype(str).str.slice(0, 10)
    else:
        day = pd.Series([""] * len(raw))
    raw = raw.loc[day.isin(set(dates))].copy()
    if raw.empty:
        return pd.DataFrame(columns=ONE_MINUTE_COLUMNS)
    normalized = normalize_domain_frame(
        raw,
        domain=DataDomain.MARKET_INTRADAY_1M,
        source="external_quant_csv_repair",
        adjusted_flag="none",
        require_columns=False,
    )
    normalized = normalize_intraday_1m_to_mootdx_240_frame(normalized)
    return _normalize_replacement_frame(normalized)


def _mootdx_replacements(issues: pd.DataFrame, *, runtime: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if issues.empty:
        return pd.DataFrame(columns=ONE_MINUTE_COLUMNS), []
    profile = resolve_runtime_profile(runtime)
    symbols = tuple(sorted({str(item) for item in issues["symbol"].tolist()}))
    start_date = str(issues["trade_date"].min())
    end_date = str(issues["trade_date"].max())
    provider = MootdxOnlineProvider()
    provider.page_size = max(200, int(profile.mootdx_page_size or 800))
    provider.max_pages = 240
    started = time.time()
    result = provider.fetch_domain(
        DomainFetchRequest(
            domain=DataDomain.MARKET_INTRADAY_1M,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            adjusted_flag="none",
        )
    )
    rows = _normalize_replacement_frame(result.data)
    rows["source"] = "mootdx_online_repair"
    needed = issues[["symbol", "trade_date"]].drop_duplicates()
    rows = rows.merge(needed, on=["symbol", "trade_date"], how="inner") if not rows.empty else rows
    unresolved = [
        {
            "symbol": str(item.get("symbol", "")),
            "trade_date": str(item.get("trade_date", "")),
            "reason": "mootdx_error",
            "error": str(item.get("message", item)),
        }
        for item in list(result.error_report or [])
    ]
    unresolved.extend(_invalid_replacement_records(issues, rows, provider="mootdx"))
    rows.attrs["elapsed_seconds"] = round(time.time() - started, 3)
    return rows, unresolved


def _normalize_replacement_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=ONE_MINUTE_COLUMNS)
    data = frame.copy()
    for column in ONE_MINUTE_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA
    data = data.loc[:, ONE_MINUTE_COLUMNS].copy()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    data["trade_date"] = data["trade_date"].astype(str)
    data["bar_time"] = data["bar_time"].astype(str).str.zfill(9)
    for column in ("open", "high", "low", "close", "volume", "amount", "turnover_rate", "float_share", "total_share"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["adjusted_flag"] = data["adjusted_flag"].fillna("none").astype(str)
    data["source"] = data["source"].fillna("repair").astype(str)
    data = _fill_zero_price_no_trade_rows(data)
    return data.drop_duplicates(["trade_date", "symbol", "bar_time"], keep="last").sort_values(["trade_date", "symbol", "bar_time"]).reset_index(drop=True)


def _fill_zero_price_no_trade_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not set(PRICE_COLUMNS).issubset(frame.columns):
        return frame
    data = frame.copy()
    bad_price = (data["open"] <= 0) | (data["high"] <= 0) | (data["low"] <= 0) | (data["close"] <= 0)
    no_trade = data["volume"].fillna(0).eq(0) & data["amount"].fillna(0).eq(0)
    eligible = bad_price & no_trade
    if not bool(eligible.any()):
        return data
    data["_qdp_original_index"] = range(len(data))
    data = data.sort_values(["symbol", "trade_date", "bar_time"]).reset_index(drop=True)
    good_close = data["close"].where((data["open"] > 0) & (data["high"] > 0) & (data["low"] > 0) & (data["close"] > 0))
    fill_price = good_close.groupby([data["symbol"], data["trade_date"]]).ffill()
    fill_price = fill_price.fillna(good_close.groupby([data["symbol"], data["trade_date"]]).bfill())
    eligible = ((data["open"] <= 0) | (data["high"] <= 0) | (data["low"] <= 0) | (data["close"] <= 0)) & data["volume"].fillna(0).eq(0) & data["amount"].fillna(0).eq(0) & fill_price.notna()
    for column in PRICE_COLUMNS:
        data.loc[eligible, column] = fill_price.loc[eligible]
    if "source" in data.columns:
        data.loc[eligible, "source"] = data.loc[eligible, "source"].astype(str) + "_zero_no_trade_ffill"
    data = data.sort_values("_qdp_original_index").drop(columns=["_qdp_original_index"])
    return data


def _invalid_replacement_records(issues: pd.DataFrame, rows: pd.DataFrame, *, provider: str) -> list[dict[str, Any]]:
    unresolved: list[dict[str, Any]] = []
    required = issues.loc[issues["action"].eq("replace"), ["symbol", "trade_date"]].drop_duplicates()
    if required.empty:
        return unresolved
    if rows.empty:
        return [
            {"symbol": str(row.symbol), "trade_date": str(row.trade_date), "reason": f"{provider}_replacement_missing"}
            for row in required.itertuples(index=False)
        ]
    counts = rows.groupby(["symbol", "trade_date"]).agg(
        bars=("bar_time", "nunique"),
        bad_price_rows=("open", lambda values: 0),
    )
    bad_prices = (
        rows.loc[(rows["open"] <= 0) | (rows["high"] <= 0) | (rows["low"] <= 0) | (rows["close"] <= 0)]
        .groupby(["symbol", "trade_date"])
        .size()
        .rename("bad_price_rows")
    )
    counts = counts.drop(columns=["bad_price_rows"]).join(bad_prices, how="left").fillna({"bad_price_rows": 0}).reset_index()
    observed = {(str(row.symbol), str(row.trade_date)): (int(row.bars), int(row.bad_price_rows)) for row in counts.itertuples(index=False)}
    for row in required.itertuples(index=False):
        key = (str(row.symbol), str(row.trade_date))
        bars, bad = observed.get(key, (0, 0))
        if bars != 240 or bad:
            unresolved.append(
                {
                    "symbol": key[0],
                    "trade_date": key[1],
                    "reason": f"{provider}_replacement_invalid",
                    "bars": bars,
                    "bad_price_rows": bad,
                }
            )
    return unresolved


def _missing_replacement_keys(issues: pd.DataFrame, rows: pd.DataFrame) -> list[dict[str, Any]]:
    needed = issues[["symbol", "trade_date", "year", "action"]].drop_duplicates()
    present = set()
    if not rows.empty:
        present = {(str(row.symbol), str(row.trade_date)) for row in rows[["symbol", "trade_date"]].drop_duplicates().itertuples(index=False)}
    missing: list[dict[str, Any]] = []
    for row in needed.itertuples(index=False):
        key = (str(row.symbol), str(row.trade_date))
        if key not in present:
            missing.append({"symbol": key[0], "trade_date": key[1], "year": int(row.year), "action": str(row.action)})
    return missing


def _dedupe_unresolved(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    output: list[dict[str, Any]] = []
    for item in items:
        key = (str(item.get("symbol", "")), str(item.get("trade_date", "")), str(item.get("reason", "")))
        if key in seen:
            continue
        seen.add(key)
        output.append(dict(item))
    return output


def _repair_blockers(*, issues: pd.DataFrame, replacements: pd.DataFrame, unresolved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    required = issues.loc[issues["action"].eq("replace"), ["symbol", "trade_date"]].drop_duplicates()
    present = set()
    if not replacements.empty:
        valid_counts = replacements.groupby(["symbol", "trade_date"])["bar_time"].nunique().reset_index(name="bars")
        present = {
            (str(row.symbol), str(row.trade_date))
            for row in valid_counts.itertuples(index=False)
            if int(row.bars) == 240
        }
    for row in required.itertuples(index=False):
        key = (str(row.symbol), str(row.trade_date))
        if key not in present:
            blockers.append({"symbol": key[0], "trade_date": key[1], "reason": "required_replacement_missing"})
    hard_unresolved = [
        item
        for item in unresolved
        if (
            str(item.get("reason", "")).endswith("_replacement_invalid")
            or str(item.get("reason", "")).endswith("_replacement_missing")
        )
        and (str(item.get("symbol", "")), str(item.get("trade_date", ""))) not in present
    ]
    blockers.extend(hard_unresolved[:50])
    return blockers


def _attach_shard_paths_to_replacements(replacements: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    if replacements.empty:
        out = replacements.copy()
        out["shard_path"] = ""
        return out
    replace_keys = keys.loc[keys["action"].isin(["replace", "replace_or_drop_if_unavailable"]), ["shard_path", "symbol", "trade_date", "action"]].drop_duplicates()
    return replacements.merge(replace_keys, on=["symbol", "trade_date"], how="inner")


def _rewrite_1m_shard(
    *,
    source_path: Path,
    target_path: Path,
    remove_keys: pd.DataFrame,
    replacements: pd.DataFrame,
    columns: list[str],
) -> dict[str, Any]:
    data = pd.read_parquet(source_path)
    for column in columns:
        if column not in data.columns:
            data[column] = pd.NA
    data = data.loc[:, columns].copy()
    key_set = {(str(row.symbol), str(row.trade_date)) for row in remove_keys[["symbol", "trade_date"]].drop_duplicates().itertuples(index=False)}
    before = len(data)
    if key_set:
        mask = pd.MultiIndex.from_frame(data[["symbol", "trade_date"]].astype(str)).isin(key_set)
        data = data.loc[~mask].copy()
    dropped = before - len(data)
    insert = _normalize_replacement_frame(replacements.drop(columns=["shard_path", "action"], errors="ignore")) if not replacements.empty else pd.DataFrame(columns=columns)
    if not insert.empty:
        share_fill = _share_fill_values(pd.read_parquet(source_path, columns=[column for column in ("symbol", "trade_date", "float_share", "total_share") if column in columns]))
        insert = _fill_replacement_share_fields(insert, share_fill)
        for column in columns:
            if column not in insert.columns:
                insert[column] = pd.NA
        insert = insert.loc[:, columns]
        data = pd.concat([data, insert], ignore_index=True)
    data = data.drop_duplicates(["trade_date", "symbol", "bar_time"], keep="last").sort_values(["trade_date", "symbol", "bar_time"]).reset_index(drop=True)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(target_path, index=False)
    return {"row_count": int(len(data)), "dropped_rows": int(dropped), "inserted_rows": int(len(insert))}


def _share_fill_values(frame: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    if frame.empty or not {"symbol", "trade_date"}.issubset(frame.columns):
        return {}
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for (symbol, trade_date), group in frame.groupby(["symbol", "trade_date"], dropna=False):
        values: dict[str, Any] = {}
        for column in ("float_share", "total_share"):
            if column in group.columns:
                non_null = group[column].dropna()
                if len(non_null):
                    values[column] = non_null.iloc[0]
        if values:
            output[(str(symbol), str(trade_date))] = values
    return output


def _fill_replacement_share_fields(frame: pd.DataFrame, fills: dict[tuple[str, str], dict[str, Any]]) -> pd.DataFrame:
    if not fills or frame.empty:
        return frame
    data = frame.copy()
    for idx, row in data.iterrows():
        values = fills.get((str(row["symbol"]), str(row["trade_date"])), {})
        for column, value in values.items():
            if column in data.columns and pd.isna(data.at[idx, column]):
                data.at[idx, column] = value
    return data


def _entry_for_repaired_shard(
    *,
    target_dataset_id: str,
    target_path: Path,
    source_path: Path,
    source_shard: ShardManifestEntry,
    row_count: int,
) -> ShardManifestEntry:
    return ShardManifestEntry(
        path=f"datasets/market_intraday_1m/{target_dataset_id}/shards/{target_path.name}",
        row_count=int(row_count),
        start_date=source_shard.start_date,
        end_date=source_shard.end_date,
        status="stored",
        file_size=int(target_path.stat().st_size),
        schema_hash=schema_hash(_parquet_schema(target_path)),
        source_path=str(source_path.resolve()),
        content_key=source_shard.content_key,
        metadata={**dict(source_shard.metadata or {}), "repair": "zero_price_intraday_bars"},
    )


def _copy_or_link_file(source_path: Path, target_path: Path, *, mode: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()
    if str(mode or "").lower() == "copy":
        shutil.copy2(source_path, target_path)
        return
    try:
        os.link(source_path, target_path)
    except Exception:
        shutil.copy2(source_path, target_path)


def _resolve_data_path(root: Path, path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()


def _parquet_schema(path: Path) -> list[dict[str, str]]:
    import pyarrow.parquet as pq  # type: ignore

    schema = pq.ParquetFile(str(path)).schema_arrow
    return [{"name": str(field.name), "type": str(field.type)} for field in schema]


def _year(value: str) -> int:
    try:
        return int(str(value)[:4])
    except Exception:
        return 0


def _count_records(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {str(k): int(v) for k, v in frame[column].value_counts(dropna=False).sort_index().items()}
