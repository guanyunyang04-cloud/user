from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    atomic_write_json,
    dataset_manifest_for_id,
    path_for_manifest,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    schema_hash,
    stable_hash,
    utc_now,
    write_dataset_manifest,
)


CONTRACT_VERSION = "qdp_v2_pit_market_substrate_v2"
VIEW_SCHEMA_VERSION = 1
OVERRIDE_DOMAINS = (
    "market_daily_raw",
    "security_status",
    "limit_status",
    "adjust_factor",
    "pit_signal_universe",
)


@dataclass(frozen=True)
class SnapshotSource:
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    priority: int
    daily_universe_paths: tuple[Path, ...]
    daily_bars_path: Path | None
    source_kind: str
    security_master_path: Path | None
    source_hash: str


@dataclass(frozen=True)
class DataFileSource:
    path: Path
    priority: int
    source_hash: str


def build_pit_market_substrate(
    *,
    snapshot_roots: Sequence[str | Path],
    market_bar_sources: Sequence[str | Path] = (),
    factor_events_path: str | Path | Sequence[str | Path] = "",
    workspace_root: str | Path | None = None,
    qdp_root: str | Path | None = None,
    reuse_active_factor: bool = True,
    active_factor_dataset_id: str = "",
    allow_no_event_factor_symbols: Sequence[str] = (),
    view_name: str = "pit_market_substrate",
    duckdb_memory_limit: str = "8GB",
    threads: int = 4,
) -> dict[str, Any]:
    """Build a non-active, PIT-complete daily market substrate and dataset view.

    Every output is content addressed. The canonical ``active/active.json`` is read
    for the base view and optional dense factor reuse, but is never written.
    """

    root = Path(qdp_root).resolve() if qdp_root else qdp_v2_root(workspace_root).resolve()
    active_path = root / "active" / "active.json"
    active_before = active_path.read_bytes()
    active = read_active_manifest(root)
    snapshots = _resolve_snapshots(snapshot_roots)
    if not snapshots:
        raise ValueError("at least one traditional PIT snapshot root is required")

    external_market_bars = _resolve_data_files(
        market_bar_sources,
        qdp_root=root,
        wanted_domain="market_daily",
    )
    if not external_market_bars and not any(item.daily_bars_path is not None for item in snapshots):
        raise ValueError("no daily market bar source was found")
    factor_inputs = (
        list(factor_events_path)
        if isinstance(factor_events_path, Sequence) and not isinstance(factor_events_path, (str, bytes, Path))
        else ([factor_events_path] if str(factor_events_path or "").strip() else [])
    )
    factor_event_files = _resolve_data_files(
        factor_inputs,
        qdp_root=root,
        wanted_domain="adjust_factor",
    )
    active_factor = _resolve_active_factor(
        root=root,
        active=active,
        reuse_active_factor=bool(reuse_active_factor),
        dataset_id=str(active_factor_dataset_id or ""),
    )
    if active_factor is None and not factor_event_files:
        raise ValueError("factor coverage requires --factor-events and/or active factor reuse")

    identity = {
        "contract_version": CONTRACT_VERSION,
        "snapshot_sources": [
            {
                "root": str(item.root),
                "source_hash": item.source_hash,
                "created_at": str(item.manifest.get("created_at", "") or ""),
            }
            for item in snapshots
        ],
        "market_bar_sources": [
            {"path": str(item.path), "sha256": item.source_hash}
            for item in external_market_bars
        ],
        "factor_events": {
            "paths": [str(item.path) for item in factor_event_files],
            "sha256": [item.source_hash for item in factor_event_files],
        },
        "active_factor_dataset_id": active_factor.dataset_id if active_factor else "",
        "allow_no_event_factor_symbols": sorted({str(item).upper() for item in allow_no_event_factor_symbols}),
        "active_factor_manifest_hash": (
            _sha256_file(root / "datasets" / "adjust_factor" / active_factor.dataset_id / "dataset.json")
            if active_factor
            else ""
        ),
        "overlap_policy": "latest_snapshot_created_at_then_resolved_path_wins",
        "factor_precedence": "active_dense_exact_key_then_active_dense_asof_then_external_event_asof",
    }
    build_hash = stable_hash(identity, length=24)
    dataset_ids = {domain: f"{domain}__{build_hash}" for domain in OVERRIDE_DOMAINS}
    view_id = f"{_safe_name(view_name)}__{build_hash}"
    view_path = root / "views" / f"{view_id}.json"
    existing = _existing_result(root=root, dataset_ids=dataset_ids, view_path=view_path)
    if existing is not None:
        _assert_unchanged(active_path, active_before)
        return existing

    stage = root / "tmp" / f"pit_market_substrate_{build_hash}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)

    import duckdb  # type: ignore

    con = duckdb.connect(":memory:")
    try:
        con.execute(f"set memory_limit={_sql_literal(str(duckdb_memory_limit or '8GB'))}")
        con.execute(f"set threads={max(1, int(threads or 1))}")
        _create_snapshot_views(con, snapshots, external_market_bars)

        market_stage = stage / "market_daily_raw.parquet"
        status_stage = stage / "security_status.parquet"
        limit_stage = stage / "limit_status.parquet"
        factor_stage = stage / "adjust_factor.parquet"
        pit_universe_stage = stage / "pit_signal_universe.parquet"
        _write_market_daily(con, market_stage)
        _write_security_status(con, status_stage)
        _write_pit_signal_universe(con, pit_universe_stage)
        _create_factor_sources(
            con,
            root=root,
            active_factor=active_factor,
            factor_events=[item.path for item in factor_event_files],
        )
        _write_adjust_factor(
            con,
            factor_stage,
            allow_no_event_factor_symbols=allow_no_event_factor_symbols,
        )
        _write_limit_status(con, limit_stage, factor_stage)

        stats = {
            "market_daily_raw": _dataset_stats(con, market_stage, ["trade_date", "symbol"]),
            "security_status": _dataset_stats(con, status_stage, ["trade_date", "symbol"]),
            "limit_status": _dataset_stats(con, limit_stage, ["trade_date", "symbol"]),
            "adjust_factor": _factor_stats(con, factor_stage),
            "pit_signal_universe": _pit_universe_stats(con, pit_universe_stage, market_stage),
        }
        blockers = _quality_blockers(stats)
        if blockers:
            raise ValueError("pit_market_substrate_blocked:" + ",".join(blockers))

        manifests: dict[str, DatasetManifest] = {}
        for domain, source_path in (
            ("market_daily_raw", market_stage),
            ("security_status", status_stage),
            ("limit_status", limit_stage),
            ("adjust_factor", factor_stage),
            ("pit_signal_universe", pit_universe_stage),
        ):
            final_path = _commit_parquet(root, domain, dataset_ids[domain], source_path)
            manifests[domain] = _build_manifest(
                con=con,
                root=root,
                domain=domain,
                dataset_id=dataset_ids[domain],
                final_path=final_path,
                stats=stats[domain],
                identity=identity,
                snapshot_count=len(snapshots),
            )
            write_dataset_manifest(root, manifests[domain])

        dataset_map = dict(active.get("datasets", {}) or {})
        dataset_map.update(dataset_ids)
        view_payload = {
            "schema_version": VIEW_SCHEMA_VERSION,
            "view_id": view_id,
            "kind": "qdp_v2_research_dataset_view",
            "created_at": utc_now(),
            "base_active_manifest": path_for_manifest(active_path, root=root),
            "base_active_updated_at": str(active.get("updated_at", "") or ""),
            "datasets": dict(sorted(dataset_map.items())),
            "overrides": {domain: dataset_ids[domain] for domain in OVERRIDE_DOMAINS},
            "scope": {
                "name": "pit_mainboard_non_st_daily_v1",
                "eligibility": "date-local listed Shanghai/Shenzhen mainboard common A; non-ST; non-delisted",
                "signal": "eligibility plus observed bar and non-suspended",
                "snapshot_count": len(snapshots),
            },
            "quality": {
                "status": "ok",
                "active_manifest_unchanged": True,
                "factor_key_coverage_complete": True,
                "factor_positive": True,
                "override_domains": list(OVERRIDE_DOMAINS),
                "stats": stats,
            },
            "source": identity,
        }
        atomic_write_json(view_path, view_payload)
    finally:
        con.close()
        shutil.rmtree(stage, ignore_errors=True)

    _assert_unchanged(active_path, active_before)
    return {
        "status": "ok",
        "view_id": view_id,
        "view_path": str(view_path.resolve()),
        "dataset_ids": dataset_ids,
        "snapshot_count": len(snapshots),
        "active_manifest_unchanged": True,
        "stats": stats,
    }


def verify_dataset_view(*, view_path: str | Path, qdp_root: str | Path | None = None) -> dict[str, Any]:
    path = Path(view_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    root = Path(qdp_root).resolve() if qdp_root else path.parents[1]
    findings: list[str] = []
    if int(payload.get("schema_version", 0) or 0) != VIEW_SCHEMA_VERSION:
        findings.append("unsupported_schema_version")
    overrides = dict(payload.get("overrides", {}) or {})
    for domain in OVERRIDE_DOMAINS:
        dataset_id = str(overrides.get(domain, "") or "")
        manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
        if not dataset_id or manifest_path is None:
            findings.append(f"missing_override:{domain}")
            continue
        manifest = read_dataset_manifest(manifest_path)
        for shard in manifest.shards:
            if not resolve_manifest_path(shard.path, root=root).exists():
                findings.append(f"missing_shard:{domain}:{shard.path}")
    return {"status": "ok" if not findings else "blocked", "findings": findings, "view_path": str(path)}


def _resolve_snapshots(roots: Sequence[str | Path]) -> list[SnapshotSource]:
    resolved: list[tuple[Path, Path, dict[str, Any], tuple[Path, ...], Path | None, str, Path | None, str]] = []
    seen: set[str] = set()
    for raw in roots:
        candidate = Path(raw).resolve()
        cache_root = candidate / "cache" if (candidate / "cache" / "daily_stock_lists").exists() else candidate
        if (cache_root / "daily_stock_lists").exists() and (cache_root / "security_master.parquet").exists():
            snapshot = candidate
            manifest_path = cache_root / "security_master.parquet"
            universe_paths = tuple(sorted((cache_root / "daily_stock_lists").glob("*.parquet")))
            if not universe_paths:
                raise FileNotFoundError(f"traditional PIT recovery cache has no daily stock lists: {cache_root}")
            security_master = cache_root / "security_master.parquet"
            created_at = max([item.stat().st_mtime for item in (*universe_paths, security_master)])
            manifest = {
                "schema_version": 2,
                "snapshot_id": candidate.name,
                "created_at": str(created_at),
                "source_kind": "traditional_pit_recovery_cache",
            }
            bars_path = candidate / "daily_bars.parquet"
            bars_path = bars_path if bars_path.exists() else None
            digest = hashlib.sha256()
            for item in (*universe_paths, security_master, bars_path):
                if item is None:
                    continue
                digest.update(item.name.encode("utf-8"))
                digest.update(_sha256_file(item).encode("ascii"))
            resolved.append(
                (snapshot, manifest_path, manifest, universe_paths, bars_path, "cache", security_master, digest.hexdigest())
            )
            continue
        if (candidate / "manifest.json").exists():
            snapshot = candidate
        else:
            latest_path = candidate / "latest_manifest.json"
            if not latest_path.exists():
                raise FileNotFoundError(f"traditional PIT snapshot manifest not found: {candidate}")
            latest = json.loads(latest_path.read_text(encoding="utf-8-sig"))
            snapshot_text = str(latest.get("snapshot_path", "") or "")
            snapshot = Path(snapshot_text)
            if not snapshot.is_absolute():
                # Traditional manifests store workspace-relative paths.
                workspace = candidate
                while workspace.parent != workspace and not (workspace / ".git").exists():
                    workspace = workspace.parent
                snapshot = workspace / snapshot
            snapshot = snapshot.resolve()
        key = str(snapshot).lower()
        if key in seen:
            continue
        seen.add(key)
        manifest_path = snapshot / "manifest.json"
        universe = snapshot / "daily_universe.parquet"
        bars = snapshot / "daily_bars.parquet"
        for required in (manifest_path, universe):
            if not required.exists():
                raise FileNotFoundError(f"traditional PIT source file not found: {required}")
        bars_path = bars if bars.exists() else None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        digest = hashlib.sha256()
        for item in (manifest_path, universe, bars_path):
            if item is None:
                continue
            digest.update(item.name.encode("utf-8"))
            digest.update(_sha256_file(item).encode("ascii"))
        resolved.append((snapshot, manifest_path, manifest, (universe,), bars_path, "snapshot", None, digest.hexdigest()))
    resolved.sort(key=lambda item: (str(item[2].get("created_at", "") or ""), str(item[0]).lower()))
    return [
        SnapshotSource(
            root=item[0],
            manifest_path=item[1],
            manifest=item[2],
            priority=index,
            daily_universe_paths=item[3],
            daily_bars_path=item[4],
            source_kind=item[5],
            security_master_path=item[6],
            source_hash=item[7],
        )
        for index, item in enumerate(resolved, start=1)
    ]


def _resolve_data_files(
    sources: Sequence[str | Path],
    *,
    qdp_root: Path,
    wanted_domain: str,
) -> list[DataFileSource]:
    collected: dict[str, Path] = {}

    def add_path(raw: str | Path, *, context: Path | None = None) -> None:
        text = str(raw or "").strip()
        if not text:
            return
        candidates = [Path(item) for item in glob.glob(text)] if any(ch in text for ch in "*?[") else [Path(text)]
        for candidate in candidates:
            if not candidate.is_absolute():
                local = (context / candidate).resolve() if context is not None else candidate.resolve()
                qdp_local = (qdp_root / candidate).resolve()
                candidate = local if local.exists() else qdp_local
            candidate = candidate.resolve()
            if candidate.is_dir():
                conventional = candidate / "daily_bars.parquet" if wanted_domain == "market_daily" else None
                if conventional is not None and conventional.exists():
                    add_path(conventional)
                    continue
                latest = candidate / "manifest_latest.json"
                if latest.exists():
                    add_path(latest)
                    continue
                raise FileNotFoundError(f"no supported {wanted_domain} data source in directory: {candidate}")
            if not candidate.exists():
                raise FileNotFoundError(f"data source not found: {candidate}")
            if candidate.suffix.lower() == ".parquet":
                collected[str(candidate).lower()] = candidate
                continue
            if candidate.suffix.lower() != ".json":
                raise ValueError(f"unsupported data source type: {candidate}")
            payload = json.loads(candidate.read_text(encoding="utf-8-sig"))
            if isinstance(payload.get("datasets"), list):
                matches = [item for item in payload["datasets"] if _descriptor_matches_domain(item, wanted_domain)]
                if not matches:
                    raise ValueError(f"manifest contains no {wanted_domain} dataset: {candidate}")
                for item in matches:
                    consume_descriptor(dict(item), candidate)
                continue
            if isinstance(payload.get("shards"), list):
                if not _descriptor_matches_domain(payload, wanted_domain):
                    raise ValueError(f"shard manifest domain mismatch for {wanted_domain}: {candidate}")
                for shard in payload["shards"]:
                    shard_path = str(dict(shard).get("path", "") or "")
                    if shard_path:
                        add_path(shard_path, context=candidate.parent)
                continue
            consume_descriptor(payload, candidate)

    def consume_descriptor(payload: Mapping[str, Any], manifest_path: Path) -> None:
        content = dict(payload.get("content_paths", {}) or {})
        shard_manifest = str(content.get("shard_manifest", "") or "")
        if shard_manifest:
            add_path(shard_manifest, context=manifest_path.parent)
            return
        silver = str(content.get("silver_domain_data", "") or "")
        if silver:
            add_path(silver, context=manifest_path.parent)
            return
        raise ValueError(f"dataset descriptor has no parquet/shard path: {manifest_path}")

    for source in sources:
        add_path(source)
    paths = sorted(collected.values(), key=lambda item: str(item).lower())
    return [
        DataFileSource(path=path, priority=index, source_hash=_sha256_file(path))
        for index, path in enumerate(paths, start=1)
    ]


def _descriptor_matches_domain(payload: Mapping[str, Any], wanted_domain: str) -> bool:
    text = " ".join(
        str(payload.get(key, "") or "").lower()
        for key in ("domain", "dataset_kind", "dataset_id")
    )
    if wanted_domain == "market_daily":
        return "market_daily" in text or "market_daily_raw" in text
    if wanted_domain == "adjust_factor":
        return "adjust_factor" in text
    return wanted_domain.lower() in text


def _resolve_active_factor(
    *, root: Path, active: Mapping[str, Any], reuse_active_factor: bool, dataset_id: str
) -> DatasetManifest | None:
    if not reuse_active_factor and not dataset_id:
        return None
    resolved_id = str(dataset_id or dict(active.get("datasets", {}) or {}).get("adjust_factor", "") or "")
    if not resolved_id:
        return None
    path = dataset_manifest_for_id(root, resolved_id, "adjust_factor")
    if path is None:
        raise FileNotFoundError(f"active/reused adjust_factor manifest not found: {resolved_id}")
    manifest = read_dataset_manifest(path)
    if "back_adjust_factor" not in {item.get("name") for item in manifest.schema}:
        raise ValueError(f"reused adjust_factor lacks back_adjust_factor: {resolved_id}")
    return manifest


def _create_snapshot_views(
    con: Any,
    snapshots: Sequence[SnapshotSource],
    external_market_bars: Sequence[DataFileSource],
) -> None:
    universe_parts: list[str] = []
    bar_parts: list[str] = []
    for item in snapshots:
        if item.source_kind == "cache":
            assert item.security_master_path is not None
            lists_sql = _path_list_sql(item.daily_universe_paths)
            master_sql = _sql_literal(str(item.security_master_path))
            mainboard_sql = (
                "regexp_matches(upper(trim(cast(l.code as varchar))), '^(600|601|603|605)[0-9]{3}[.]SH$') "
                "or regexp_matches(upper(trim(cast(l.code as varchar))), '^(000|001|002|003)[0-9]{3}[.]SZ$')"
            )
            universe_parts.append(
                f"""
                select
                  upper(trim(cast(l.code as varchar))) as symbol,
                  cast(l.date as date) as trade_date,
                  (cast(l.date as date) >= coalesce(try_cast(s.ipo_date as date), cast(l.date as date))
                    and (try_cast(s.out_date as date) is null or cast(l.date as date) < try_cast(s.out_date as date))) as is_listed,
                  ({mainboard_sql}) as is_mainboard,
                  coalesce(trim(cast(s.security_type as varchar)) = '1', true) as is_common_a_share,
                  regexp_matches(upper(trim(cast(l.name_on_date as varchar))), '^(\\*ST|ST|SST|S\\*ST)') as is_st,
                  coalesce(trim(cast(l.query_all_trade_status as varchar)) <> '1', false) as is_suspended,
                  false as has_bar,
                  try_cast(s.out_date as date) as out_date,
                  case when coalesce(trim(cast(l.query_all_trade_status as varchar)) <> '1', false) then 'suspended' else '' end as reject_reason,
                  {item.priority}::integer as _source_priority,
                  {_sql_literal(str(item.root))}::varchar as _snapshot_root
                from read_parquet({lists_sql}, union_by_name=true) l
                left join read_parquet({master_sql}) s
                  on upper(trim(cast(l.code as varchar))) = upper(trim(cast(s.code as varchar)))
                """
            )
        else:
            universe_parts.append(
                f"""
                select
                  upper(trim(cast(code as varchar))) as symbol,
                  cast(date as date) as trade_date,
                  coalesce(try_cast(is_listed_on_date as boolean), false) as is_listed,
                  coalesce(try_cast(is_mainboard as boolean), false) as is_mainboard,
                  coalesce(try_cast(is_common_a_share as boolean), false) as is_common_a_share,
                  coalesce(try_cast(is_st_on_date as boolean), false) as is_st,
                  coalesce(try_cast(is_suspended_on_date as boolean), false) as is_suspended,
                  coalesce(try_cast(has_bar as boolean), false) as has_bar,
                  try_cast(nullif(trim(cast(out_date as varchar)), '') as date) as out_date,
                  cast(coalesce(reject_reason, '') as varchar) as reject_reason,
                  {item.priority}::integer as _source_priority,
                  {_sql_literal(str(item.root))}::varchar as _snapshot_root
                from read_parquet({_path_list_sql(item.daily_universe_paths)}, union_by_name=true)
                """
            )
        if item.daily_bars_path is not None:
            bar_parts.append(
                "select code as _raw_symbol, date as _raw_date, open, high, low, close, volume, amount, "
                f"{item.priority}::integer as _source_priority, "
                f"{_sql_literal(str(item.root))}::varchar as _snapshot_root from read_parquet({_sql_literal(str(item.daily_bars_path))})"
            )
    base_priority = len(snapshots)
    for item in external_market_bars:
        columns = {
            str(row[0]).lower(): str(row[0])
            for row in con.execute(
                f"describe select * from read_parquet({_sql_literal(str(item.path))}, union_by_name=true)"
            ).fetchall()
        }
        symbol_column = columns.get("symbol") or columns.get("code")
        date_column = columns.get("trade_date") or columns.get("date")
        if symbol_column is None or date_column is None:
            raise ValueError(
                "external market bars require symbol/trade_date or code/date columns: "
                f"{item.path}"
            )
        bar_parts.append(
            f"select {_ident(symbol_column)} as _raw_symbol, {_ident(date_column)} as _raw_date, "
            "open, high, low, close, volume, amount, "
            f"{base_priority + item.priority}::integer as _source_priority, "
            f"{_sql_literal(str(item.path))}::varchar as _snapshot_root from read_parquet({_sql_literal(str(item.path))}, union_by_name=true)"
        )
    con.execute("create temp view snapshot_universe_raw as " + " union all by name ".join(universe_parts))
    con.execute("create temp view snapshot_bars_raw as " + " union all by name ".join(bar_parts))
    con.execute(
        """
        create temp table pit_universe as
        select symbol, trade_date, is_listed, is_mainboard, is_common_a_share,
               is_st, is_suspended, has_bar, out_date, reject_reason, _snapshot_root
        from snapshot_universe_raw
        where trade_date is not null and trim(coalesce(symbol, '')) <> ''
        qualify row_number() over (
          partition by trade_date, symbol
          order by _source_priority desc, _snapshot_root desc
        ) = 1
        """
    )
    con.execute(
        """
        create temp table pit_bars as
        select
          upper(trim(cast(_raw_symbol as varchar))) as symbol,
          cast(_raw_date as date) as trade_date,
          try_cast(open as double) as open,
          try_cast(high as double) as high,
          try_cast(low as double) as low,
          try_cast(close as double) as close,
          try_cast(volume as double) as volume,
          try_cast(amount as double) as amount,
          _snapshot_root
        from snapshot_bars_raw
        where try_cast(_raw_date as date) is not null and trim(coalesce(cast(_raw_symbol as varchar), '')) <> ''
        qualify row_number() over (
          partition by cast(_raw_date as date), upper(trim(cast(_raw_symbol as varchar)))
          order by _source_priority desc, _snapshot_root desc
        ) = 1
        """
    )


def _write_market_daily(con: Any, target: Path) -> None:
    con.execute(
        f"""
        copy (
          select
            b.symbol,
            strftime(b.trade_date, '%Y-%m-%d') as trade_date,
            b.open, b.high, b.low, b.close,
            greatest(b.volume, 0.0) as volume,
            greatest(b.amount, 0.0) as amount,
            'traditional_baostock_pit_snapshot' as source,
            'none' as adjusted_flag
          from pit_bars b
          inner join pit_universe u using(symbol, trade_date)
          where u.is_listed and u.is_mainboard and u.is_common_a_share
            and not (u.out_date is not null and b.trade_date >= u.out_date)
            and b.open > 0 and b.high > 0 and b.low > 0 and b.close > 0
            and b.high >= greatest(b.open, b.close, b.low)
            and b.low <= least(b.open, b.close, b.high)
            and b.volume >= 0 and b.amount >= 0
          order by b.trade_date, b.symbol
        ) to {_sql_literal(str(target))} (format parquet, compression zstd)
        """
    )
    con.execute(f"create temp view built_market as select * from read_parquet({_sql_literal(str(target))})")


def _write_security_status(con: Any, target: Path) -> None:
    con.execute(
        f"""
        copy (
          select
            u.symbol,
            strftime(u.trade_date, '%Y-%m-%d') as trade_date,
            u.is_st,
            u.is_suspended,
            (u.out_date is not null and u.trade_date >= u.out_date) as is_delisted,
            (m.symbol is not null) as has_bar,
            (m.symbol is not null) as has_valid_market_bar,
            case
              when u.out_date is not null and u.trade_date >= u.out_date then 'delisted'
              when u.is_st then 'st'
              when u.is_suspended then 'suspended'
              when m.symbol is null then 'missing_or_invalid_bar'
              else coalesce(nullif(u.reject_reason, ''), 'tradeable')
            end as status_reason,
            'traditional_baostock_pit_snapshot' as source
          from pit_universe u
          left join built_market m
            on m.symbol = u.symbol and cast(m.trade_date as date) = u.trade_date
          order by u.trade_date, u.symbol
        ) to {_sql_literal(str(target))} (format parquet, compression zstd)
        """
    )


def _write_pit_signal_universe(con: Any, target: Path) -> None:
    con.execute(
        f"""
        copy (
          select
            u.symbol,
            strftime(u.trade_date, '%Y-%m-%d') as trade_date,
            u.is_listed,
            u.is_mainboard,
            u.is_common_a_share,
            u.is_st,
            u.is_suspended,
            (m.symbol is not null) as has_bar,
            (u.out_date is not null and u.trade_date >= u.out_date) as is_delisted,
            true as status_valid,
            (u.is_listed and u.is_mainboard and u.is_common_a_share
              and not u.is_st
              and not (u.out_date is not null and u.trade_date >= u.out_date)) as eligible_for_research,
            (u.is_listed and u.is_mainboard and u.is_common_a_share
              and not u.is_st and not u.is_suspended
              and not (u.out_date is not null and u.trade_date >= u.out_date)
              and m.symbol is not null) as eligible_for_signal,
            case
              when not u.is_listed then 'not_listed'
              when not u.is_mainboard then 'non_mainboard'
              when not u.is_common_a_share then 'non_common_a_share'
              when u.out_date is not null and u.trade_date >= u.out_date then 'delisted'
              when u.is_st then 'st'
              when u.is_suspended then 'suspended'
              when m.symbol is null then 'missing_or_invalid_bar'
              else ''
            end as primary_exclusion_reason,
            'traditional_baostock_pit_snapshot' as source
          from pit_universe u
          left join built_market m
            on m.symbol = u.symbol and cast(m.trade_date as date) = u.trade_date
          order by u.trade_date, u.symbol
        ) to {_sql_literal(str(target))} (format parquet, compression zstd)
        """
    )


def _write_limit_status(con: Any, target: Path, factor_path: Path) -> None:
    con.execute(
        f"""
        copy (
          with daily as (
            select
              m.*,
              f.back_adjust_factor,
              lag(m.close * f.back_adjust_factor) over (
                partition by m.symbol order by cast(m.trade_date as date)
              ) as prior_valid_adjusted_close
            from built_market m
            inner join read_parquet({_sql_literal(str(factor_path))}) f using(trade_date, symbol)
          ), joined as (
            select d.*, coalesce(s.is_st, false) as is_st
            from daily d
            left join read_parquet({_sql_literal(str(target.parent / 'security_status.parquet'))}) s
              on s.symbol = d.symbol and s.trade_date = d.trade_date
          ), limits as (
            select
              *,
              prior_valid_adjusted_close / back_adjust_factor as reference_raw_today,
              floor((prior_valid_adjusted_close / back_adjust_factor)
                * case when is_st then 1.05 else 1.10 end * 100.0 + 0.5) / 100.0 as up_limit,
              floor((prior_valid_adjusted_close / back_adjust_factor)
                * case when is_st then 0.95 else 0.90 end * 100.0 + 0.5) / 100.0 as down_limit
            from joined
          )
          select
            trade_date, symbol, prior_valid_adjusted_close, reference_raw_today, up_limit, down_limit,
            (up_limit is not null and floor(close * 100.0 + 0.5) / 100.0 >= up_limit) as is_limit_up,
            (down_limit is not null and floor(close * 100.0 + 0.5) / 100.0 <= down_limit) as is_limit_down,
            'pit_adjusted_prior_close_to_current_raw_current_st_tick_half_up' as source
          from limits
          order by cast(trade_date as date), symbol
        ) to {_sql_literal(str(target))} (format parquet, compression zstd)
        """
    )


def _create_factor_sources(
    con: Any, *, root: Path, active_factor: DatasetManifest | None, factor_events: Sequence[Path]
) -> None:
    if active_factor is None:
        con.execute(
            "create temp view active_factor_rows as select null::varchar symbol, null::date trade_date, "
            "null::double fore_adjust_factor, null::double back_adjust_factor where false"
        )
    else:
        paths = [_resolve_shard(root, item.path) for item in active_factor.shards]
        con.execute(
            f"""
            create temp view active_factor_rows as
            select
              upper(trim(cast(symbol as varchar))) as symbol,
              cast(trade_date as date) as trade_date,
              try_cast(fore_adjust_factor as double) as fore_adjust_factor,
              try_cast(back_adjust_factor as double) as back_adjust_factor
            from read_parquet({_path_list_sql(paths)}, union_by_name=true)
            """
        )
    if not factor_events:
        con.execute(
            "create temp view external_factor_events as select null::varchar symbol, null::date factor_date, "
            "null::double fore_adjust_factor, null::double back_adjust_factor where false"
        )
        return
    columns = {str(row[0]).lower(): str(row[0]) for row in con.execute(
        f"describe select * from read_parquet({_path_list_sql(factor_events)}, union_by_name=true)"
    ).fetchall()}
    symbol_col = _pick_column(columns, "symbol", "code", "ts_code")
    date_col = _pick_column(columns, "factor_source_date", "trade_date", "dividoperatedate", "date")
    back_col = _pick_column(columns, "back_adjust_factor", "backadjustfactor", "adjust_factor", "adjustfactor")
    fore_col = _pick_column(columns, "fore_adjust_factor", "foreadjustfactor", required=False)
    symbol_expr = _normalized_symbol_sql(_ident(symbol_col))
    fore_expr = f"try_cast({_ident(fore_col)} as double)" if fore_col else "null::double"
    con.execute(
        f"""
        create temp view external_factor_events as
        select
          {symbol_expr} as symbol,
          try_cast({_ident(date_col)} as date) as factor_date,
          {fore_expr} as fore_adjust_factor,
          try_cast({_ident(back_col)} as double) as back_adjust_factor
        from read_parquet({_path_list_sql(factor_events)}, union_by_name=true)
        where try_cast({_ident(date_col)} as date) is not null
          and try_cast({_ident(back_col)} as double) > 0
        qualify row_number() over (
          partition by {symbol_expr}, try_cast({_ident(date_col)} as date)
          order by try_cast({_ident(back_col)} as double) desc
        ) = 1
        """
    )


def _write_adjust_factor(
    con: Any,
    target: Path,
    *,
    allow_no_event_factor_symbols: Sequence[str] = (),
) -> None:
    allowed = sorted({str(item).upper().strip() for item in allow_no_event_factor_symbols if str(item).strip()})
    allowed_sql = "[" + ",".join(_sql_literal(item) for item in allowed) + "]"
    con.execute(
        f"""
        copy (
          with active_expanded as (
            select
              m.symbol,
              m.trade_date,
              a.fore_adjust_factor,
              a.back_adjust_factor,
              a.trade_date as factor_date
            from built_market m
            asof left join active_factor_rows a
              on m.symbol = a.symbol and cast(m.trade_date as date) >= a.trade_date
          ), event_expanded as (
            select
              m.symbol,
              m.trade_date,
              e.fore_adjust_factor,
              e.back_adjust_factor,
              e.factor_date
            from built_market m
            asof left join external_factor_events e
              on m.symbol = e.symbol and cast(m.trade_date as date) >= e.factor_date
          )
          select
            m.symbol,
            m.trade_date,
            coalesce(a.fore_adjust_factor, ae.fore_adjust_factor, e.fore_adjust_factor,
              case when m.symbol in {allowed_sql} then 1.0 end) as fore_adjust_factor,
            coalesce(a.back_adjust_factor, ae.back_adjust_factor, e.back_adjust_factor,
              case when m.symbol in {allowed_sql} then 1.0 end) as back_adjust_factor,
            coalesce(a.back_adjust_factor, ae.back_adjust_factor, e.back_adjust_factor,
              case when m.symbol in {allowed_sql} then 1.0 end) as adjust_factor,
            case
              when a.back_adjust_factor is not null then 'active_qdp_dense'
              when ae.back_adjust_factor is not null then 'active_qdp_dense_asof'
              when e.back_adjust_factor is not null then 'external_factor_events'
              else 'explicit_no_event_factor'
            end as factor_provider,
            'back_adjust_factor_exact_or_event_ffill' as factor_semantics,
            case
              when a.back_adjust_factor is not null then 'active_adjust_factor_exact_key'
              when ae.back_adjust_factor is not null then 'active_adjust_factor_asof'
              when e.back_adjust_factor is not null then 'external_factor_event_ffill'
              else 'explicit_no_event_factor_1'
            end as source,
            strftime(coalesce(a.trade_date, ae.factor_date, e.factor_date,
              case when m.symbol in {allowed_sql} then cast(m.trade_date as date) end), '%Y-%m-%d') as factor_source_date,
            case
              when a.trade_date is not null then 0
              else date_diff('day', coalesce(ae.factor_date, e.factor_date), cast(m.trade_date as date))
            end as ffill_days
          from built_market m
          left join active_factor_rows a
            on a.symbol = m.symbol and a.trade_date = cast(m.trade_date as date)
          left join active_expanded ae
            on ae.symbol = m.symbol and ae.trade_date = m.trade_date
          left join event_expanded e
            on e.symbol = m.symbol and e.trade_date = m.trade_date
          order by cast(m.trade_date as date), m.symbol
        ) to {_sql_literal(str(target))} (format parquet, compression zstd)
        """
    )


def _dataset_stats(con: Any, path: Path, keys: Sequence[str]) -> dict[str, Any]:
    key_expr = ", ".join(_ident(item) for item in keys)
    row = con.execute(
        f"""
        select count(*) row_count,
               count(*) - count(distinct ({key_expr})) duplicate_keys,
               min(trade_date) start_date, max(trade_date) end_date,
               count(distinct symbol) symbol_count, count(distinct trade_date) date_count
        from read_parquet({_sql_literal(str(path))})
        """
    ).fetchone()
    return {
        "row_count": int(row[0] or 0),
        "duplicate_keys": int(row[1] or 0),
        "start_date": str(row[2] or ""),
        "end_date": str(row[3] or ""),
        "symbol_count": int(row[4] or 0),
        "date_count": int(row[5] or 0),
    }


def _factor_stats(con: Any, path: Path) -> dict[str, Any]:
    stats = _dataset_stats(con, path, ["trade_date", "symbol"])
    row = con.execute(
        f"""
        select
          sum(case when back_adjust_factor is null then 1 else 0 end),
          sum(case when back_adjust_factor is not null and back_adjust_factor <= 0 then 1 else 0 end),
          sum(case when factor_source_date is null then 1 else 0 end),
          sum(case when source = 'active_adjust_factor_exact_key' then 1 else 0 end),
          sum(case when source = 'active_adjust_factor_asof' then 1 else 0 end),
          sum(case when source = 'external_factor_event_ffill' then 1 else 0 end),
          max(ffill_days)
        from read_parquet({_sql_literal(str(path))})
        """
    ).fetchone()
    stats.update(
        missing_factor_rows=int(row[0] or 0),
        nonpositive_factor_rows=int(row[1] or 0),
        missing_factor_source_rows=int(row[2] or 0),
        active_exact_rows=int(row[3] or 0),
        active_asof_rows=int(row[4] or 0),
        external_ffill_rows=int(row[5] or 0),
        max_ffill_days=int(row[6] or 0),
    )
    return stats


def _pit_universe_stats(con: Any, path: Path, market_path: Path) -> dict[str, Any]:
    stats = _dataset_stats(con, path, ["trade_date", "symbol"])
    row = con.execute(
        f"""
        select
          sum(case when eligible_for_research then 1 else 0 end),
          sum(case when eligible_for_signal then 1 else 0 end),
          sum(case when eligible_for_signal and m.symbol is null then 1 else 0 end),
          sum(case when eligible_for_research and not is_suspended and m.symbol is null then 1 else 0 end),
          min(case when eligible_for_research and not is_suspended then u.trade_date end),
          max(case when eligible_for_research and not is_suspended then u.trade_date end),
          count(distinct case when eligible_for_research and not is_suspended then u.trade_date end),
          sum(case when is_st then 1 else 0 end),
          sum(case when is_suspended then 1 else 0 end),
          sum(case when is_delisted then 1 else 0 end)
        from read_parquet({_sql_literal(str(path))}) u
        left join read_parquet({_sql_literal(str(market_path))}) m using(trade_date, symbol)
        """
    ).fetchone()
    stats.update(
        eligible_for_research_rows=int(row[0] or 0),
        eligible_for_signal_rows=int(row[1] or 0),
        eligible_signal_without_market_rows=int(row[2] or 0),
        eligible_research_non_suspended_without_market_rows=int(row[3] or 0),
        expected_market_start_date=str(row[4] or ""),
        expected_market_end_date=str(row[5] or ""),
        expected_market_date_count=int(row[6] or 0),
        st_rows=int(row[7] or 0),
        suspended_rows=int(row[8] or 0),
        delisted_rows=int(row[9] or 0),
    )
    return stats


def _quality_blockers(stats: Mapping[str, Mapping[str, Any]]) -> list[str]:
    blockers: list[str] = []
    market_rows = int(stats["market_daily_raw"].get("row_count", 0) or 0)
    if market_rows <= 0:
        blockers.append("empty_market_daily_raw")
    for domain, item in stats.items():
        if int(item.get("duplicate_keys", 0) or 0):
            blockers.append(f"duplicate_primary_keys:{domain}")
    factors = stats["adjust_factor"]
    if int(factors.get("row_count", 0) or 0) != market_rows:
        blockers.append("factor_market_row_count_mismatch")
    if int(factors.get("missing_factor_rows", 0) or 0):
        blockers.append("missing_factor_rows")
    if int(factors.get("nonpositive_factor_rows", 0) or 0):
        blockers.append("nonpositive_factor_rows")
    if int(factors.get("missing_factor_source_rows", 0) or 0):
        blockers.append("missing_factor_provenance")
    if int(stats["pit_signal_universe"].get("eligible_signal_without_market_rows", 0) or 0):
        blockers.append("pit_gate0_eligible_signal_without_market")
    pit_stats = stats["pit_signal_universe"]
    if int(stats["security_status"].get("row_count", 0) or 0) != int(pit_stats.get("row_count", 0) or 0):
        blockers.append("security_status_pit_row_count_mismatch")
    if int(pit_stats.get("eligible_research_non_suspended_without_market_rows", 0) or 0):
        blockers.append("pit_gate0_research_non_suspended_without_market")
    expected_start = str(pit_stats.get("expected_market_start_date", "") or "")
    expected_end = str(pit_stats.get("expected_market_end_date", "") or "")
    market_start = str(stats["market_daily_raw"].get("start_date", "") or "")
    market_end = str(stats["market_daily_raw"].get("end_date", "") or "")
    if expected_start and (not market_start or market_start > expected_start):
        blockers.append("pit_gate0_market_start_coverage")
    if expected_end and (not market_end or market_end < expected_end):
        blockers.append("pit_gate0_market_end_coverage")
    return blockers


def _commit_parquet(root: Path, domain: str, dataset_id: str, source: Path) -> Path:
    target_dir = root / "datasets" / domain / dataset_id
    final = target_dir / "shards" / f"part_000000_{domain}.parquet"
    if final.exists():
        return final
    staging = target_dir / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    staged = staging / final.name
    shutil.move(str(source), str(staged))
    final.parent.mkdir(parents=True, exist_ok=True)
    staged.replace(final)
    shutil.rmtree(staging, ignore_errors=True)
    return final


def _build_manifest(
    *,
    con: Any,
    root: Path,
    domain: str,
    dataset_id: str,
    final_path: Path,
    stats: Mapping[str, Any],
    identity: Mapping[str, Any],
    snapshot_count: int,
) -> DatasetManifest:
    schema = [{"name": str(row[0]), "type": str(row[1])} for row in con.execute(
        f"describe select * from read_parquet({_sql_literal(str(final_path))})"
    ).fetchall()]
    primary_key = ["trade_date", "symbol"]
    entry = ShardManifestEntry(
        path=path_for_manifest(final_path, root=root),
        row_count=int(stats.get("row_count", 0) or 0),
        start_date=str(stats.get("start_date", "") or ""),
        end_date=str(stats.get("end_date", "") or ""),
        file_size=int(final_path.stat().st_size),
        schema_hash=schema_hash(schema),
        content_key="pit_mainboard_daily_market_substrate",
        metadata={"snapshot_count": int(snapshot_count)},
    )
    contracts = {
        "market_daily_raw": "qdp_v2_market_daily_raw_pit_v1",
        "security_status": "qdp_v2_security_status_pit_v1",
        "limit_status": "qdp_v2_limit_status_dense_tick_v1",
        "adjust_factor": "qdp_v2_adjust_factor_pit_complete_v1",
        "pit_signal_universe": "qdp_v2_pit_signal_universe_v2",
    }
    return DatasetManifest(
        dataset_id=dataset_id,
        domain=domain,
        layer="raw" if domain in {"market_daily_raw", "security_status", "adjust_factor"} else "derived",
        frequency="1d",
        contract_version=contracts[domain],
        primary_key=primary_key,
        start_date=entry.start_date,
        end_date=entry.end_date,
        row_count=entry.row_count,
        schema_hash=schema_hash(schema),
        schema=schema,
        shards=[entry],
        source={
            "provider": "traditional_baostock_pit_snapshot",
            "created_by": "build_pit_market_substrate",
            "created_at": utc_now(),
            "identity": dict(identity),
        },
        quality={
            "status": "ok",
            "path_refs_exist": True,
            "primary_key_unique": int(stats.get("duplicate_keys", 0) or 0) == 0,
            "stats": dict(stats),
            "active_pointer_changed": False,
        },
        notes=[
            "Non-active PIT research dataset; canonical active.json is unchanged.",
            "Snapshot overlap resolves by created_at then resolved path, with the greatest priority winning.",
        ],
    )


def _existing_result(*, root: Path, dataset_ids: Mapping[str, str], view_path: Path) -> dict[str, Any] | None:
    if not view_path.exists():
        return None
    if any(dataset_manifest_for_id(root, dataset_id, domain) is None for domain, dataset_id in dataset_ids.items()):
        return None
    verified = verify_dataset_view(view_path=view_path, qdp_root=root)
    if verified["status"] != "ok":
        return None
    payload = json.loads(view_path.read_text(encoding="utf-8-sig"))
    return {
        "status": "ok_reused",
        "view_id": str(payload.get("view_id", "") or ""),
        "view_path": str(view_path.resolve()),
        "dataset_ids": dict(dataset_ids),
        "snapshot_count": int(dict(payload.get("scope", {}) or {}).get("snapshot_count", 0) or 0),
        "active_manifest_unchanged": True,
        "stats": dict(dict(payload.get("quality", {}) or {}).get("stats", {}) or {}),
    }


def _pick_column(columns: Mapping[str, str], *names: str, required: bool = True) -> str:
    for name in names:
        if name.lower() in columns:
            return columns[name.lower()]
    if required:
        raise ValueError(f"factor events missing required column; expected one of {names}")
    return ""


def _normalized_symbol_sql(expr: str) -> str:
    return (
        "case "
        f"when regexp_matches(lower(trim(cast({expr} as varchar))), '^(sh|sz)[.]?[0-9]{{6}}$') then "
        f"substr(regexp_replace(lower(trim(cast({expr} as varchar))), '[.]', '', 'g'), 3, 6) || '.' || upper(substr(trim(cast({expr} as varchar)), 1, 2)) "
        f"when regexp_matches(upper(trim(cast({expr} as varchar))), '^[0-9]{{6}}[.](SH|SZ)$') then upper(trim(cast({expr} as varchar))) "
        f"when regexp_matches(trim(cast({expr} as varchar)), '^[0-9]{{6}}$') then trim(cast({expr} as varchar)) || case when substr(trim(cast({expr} as varchar)),1,1)='6' then '.SH' else '.SZ' end "
        f"else upper(trim(cast({expr} as varchar))) end"
    )


def _resolve_shard(root: Path, path: str) -> Path:
    return resolve_manifest_path(path, root=root)


def _path_list_sql(paths: Iterable[Path]) -> str:
    return "[" + ",".join(_sql_literal(str(path)) for path in paths) + "]"


def _sha256_file(path: Path | None, chunk_size: int = 4 * 1024 * 1024) -> str:
    if path is None:
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_unchanged(path: Path, before: bytes) -> None:
    if path.read_bytes() != before:
        raise RuntimeError("active_manifest_changed_during_non_active_build")


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value or ""))
    return cleaned.strip("_") or "pit_market_substrate"


def _ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("\\", "/").replace("'", "''") + "'"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build/verify a non-active QDP PIT daily market substrate")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--snapshot-root", action="append", required=True)
    build.add_argument(
        "--market-bars-source",
        action="append",
        default=[],
        help="Optional parquet, shard_manifest.json, QDP dataset.json, or data-lake manifest_latest.json; repeatable.",
    )
    build.add_argument(
        "--factor-events",
        action="append",
        default=[],
        help="Factor-event parquet/manifest; repeatable. Active dense factors have exact-key precedence.",
    )
    build.add_argument("--workspace-root", default="")
    build.add_argument("--qdp-root", default="")
    build.add_argument("--active-factor-dataset-id", default="")
    build.add_argument(
        "--allow-no-event-factor-symbol",
        action="append",
        default=[],
        help="Explicit audited no-corporate-action exception; emits factor=1 with provenance. Repeatable.",
    )
    build.add_argument("--reuse-active-factor", dest="reuse_active_factor", action="store_true", default=True)
    build.add_argument("--no-reuse-active-factor", dest="reuse_active_factor", action="store_false")
    build.add_argument("--view-name", default="pit_market_substrate")
    build.add_argument("--duckdb-memory-limit", default="8GB")
    build.add_argument("--threads", type=int, default=4)
    verify = sub.add_parser("verify-view")
    verify.add_argument("--view", required=True)
    verify.add_argument("--qdp-root", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        payload = build_pit_market_substrate(
            snapshot_roots=args.snapshot_root,
            market_bar_sources=args.market_bars_source,
            factor_events_path=args.factor_events,
            workspace_root=args.workspace_root or None,
            qdp_root=args.qdp_root or None,
            reuse_active_factor=bool(args.reuse_active_factor),
            active_factor_dataset_id=args.active_factor_dataset_id,
            allow_no_event_factor_symbols=args.allow_no_event_factor_symbol,
            view_name=args.view_name,
            duckdb_memory_limit=args.duckdb_memory_limit,
            threads=args.threads,
        )
    else:
        payload = verify_dataset_view(view_path=args.view, qdp_root=args.qdp_root or None)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload.get("status") in {"ok", "ok_reused"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
