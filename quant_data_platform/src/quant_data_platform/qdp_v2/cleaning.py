from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.qdp_v2.duckdb_resources import open_guarded_duckdb
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


def derive_5m_from_1m(
    *,
    workspace_root: str | Path | None = None,
    source_dataset_id: str = "",
    max_shards: int = 0,
    runtime: str = "balanced",
    workers: int = 1,
    resume: bool = False,
    trust_existing: bool = False,
    duckdb_memory_limit: str = "",
    shard_modulo: int = 0,
    shard_remainder: int = 0,
    stage_only: bool = False,
    activate_domain: bool = False,
    reuse_from_dataset_id: str = "",
    reuse_copy_mode: str = "hardlink",
) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    source_manifest = _resolve_source_manifest(root, source_dataset_id, "market_intraday_1m")
    source = read_dataset_manifest(source_manifest)
    profile = resolve_runtime_profile(runtime)
    memory_limit = str(duckdb_memory_limit or "").strip() or profile.duckdb_memory_limit
    target_dataset_id = f"market_intraday_5m__{stable_hash({'source': source.dataset_id, 'contract': 'mootdx_5m_48_v1'})}"
    target_dir = root / "datasets" / "market_intraday_5m" / target_dataset_id
    staging = target_dir / ".staging"
    if staging.exists() and not resume:
        shutil.rmtree(staging)
    staging_shards = staging / "shards"
    staging_shards.mkdir(parents=True, exist_ok=True)
    requested_workers = max(1, int(workers or 1))
    worker_count = min(requested_workers, max(1, int(profile.duckdb_threads or 1)), max(1, len(source.shards)))
    worker_memory_limit = _parallel_duckdb_memory_limit(memory_limit, worker_count)
    worker_threads = max(1, min(2, int(profile.duckdb_threads or 1)))
    reuse_manifest = _resolve_optional_manifest(root, reuse_from_dataset_id, "market_intraday_5m")
    reuse_source = read_dataset_manifest(reuse_manifest) if reuse_manifest is not None else None
    shard_entries_by_index: dict[int, ShardManifestEntry] = {}
    errors: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    reused = 0
    reused_from_dataset = 0
    selected = 0
    modulo = max(0, int(shard_modulo or 0))
    remainder = int(shard_remainder or 0)
    for index, shard in enumerate(source.shards):
        if modulo and index % modulo != remainder:
            continue
        if max_shards and selected >= max_shards:
            break
        source_path = root / shard.path if not Path(shard.path).is_absolute() else Path(shard.path)
        reuse_shard = reuse_source.shards[index] if reuse_source is not None and index < len(reuse_source.shards) else None
        target_name = Path(reuse_shard.path).name if reuse_shard is not None else f"part_{index:06d}_5m.parquet"
        target_path = staging_shards / target_name
        selected += 1
        if reuse_shard is not None and _can_reuse_5m_shard_from_prior(source_shard=shard, reuse_shard=reuse_shard):
            try:
                reuse_path = root / reuse_shard.path if not Path(reuse_shard.path).is_absolute() else Path(reuse_shard.path)
                _copy_or_link_file(reuse_path, target_path, mode=reuse_copy_mode)
                shard_entries_by_index[index] = ShardManifestEntry(
                    path=f"datasets/market_intraday_5m/{target_dataset_id}/shards/{target_path.name}",
                    row_count=reuse_shard.row_count,
                    start_date=reuse_shard.start_date or shard.start_date,
                    end_date=reuse_shard.end_date or shard.end_date,
                    status="stored",
                    file_size=int(target_path.stat().st_size),
                    schema_hash=reuse_shard.schema_hash,
                    source_path=str(reuse_path.resolve()),
                    content_key=reuse_shard.content_key or "derived_from_1m_240",
                    metadata={**dict(reuse_shard.metadata or {}), "reused_from_dataset_id": reuse_source.dataset_id, "source_1m_dataset_id": source.dataset_id},
                )
                reused_from_dataset += 1
                continue
            except Exception as exc:
                errors.append({"source_path": str(source_path), "target_path": str(target_path), "error": f"reuse_5m_shard_failed: {exc}"})
                continue
        if resume and target_path.exists():
            if trust_existing and int(target_path.stat().st_size) > 0:
                reused += 1
                continue
            try:
                stats = _existing_5m_shard_stats(target_path)
                if int(stats["row_count"]) > 0:
                    shard_entries_by_index[index] = _five_minute_shard_entry(
                        target_dataset_id=target_dataset_id,
                        target_path=target_path,
                        source_path=source_path,
                        row_count=int(stats["row_count"]),
                        schema_hash_value=str(stats.get("schema_hash", "") or ""),
                        shard=shard,
                    )
                    reused += 1
                    continue
            except Exception:
                try:
                    _remove_file_with_retries(target_path)
                except Exception as exc:
                    errors.append({"source_path": str(source_path), "target_path": str(target_path), "error": f"remove_corrupt_resume_file_failed: {exc}"})
                    continue
        tasks.append(
            {
                "index": index,
                "source_path": str(source_path),
                "target_path": str(target_path),
                "memory_limit": worker_memory_limit,
                "threads": worker_threads,
                "target_dataset_id": target_dataset_id,
                "shard": shard.model_dump() if hasattr(shard, "model_dump") else shard.__dict__,
            }
        )
    if worker_count <= 1 or len(tasks) <= 1:
        for task in tasks:
            result = _derive_5m_shard_task(task)
            if result.get("status") == "ok":
                shard_entries_by_index[int(result["index"])] = result["entry"]
            elif result.get("status") != "empty":
                errors.append(result)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(_derive_5m_shard_task, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                if result.get("status") == "ok":
                    shard_entries_by_index[int(result["index"])] = result["entry"]
                elif result.get("status") != "empty":
                    errors.append(result)
    shard_entries = [shard_entries_by_index[key] for key in sorted(shard_entries_by_index)]
    processed = len(shard_entries)
    if errors:
        return {"status": "error", "target_dataset_id": target_dataset_id, "processed_shards": processed, "errors": errors[:50], "error_count": len(errors)}
    if stage_only:
        payload = {
            "status": "ok",
            "stage_only": True,
            "target_dataset_id": target_dataset_id,
            "source_dataset_id": source.dataset_id,
            "processed_shards": processed,
            "new_shards": len(tasks),
            "reused_shards": reused,
            "reused_from_dataset_shards": reused_from_dataset,
            "selected_shards": selected,
            "trust_existing": bool(trust_existing),
            "max_shards": int(max_shards or 0),
            "shard_modulo": modulo,
            "shard_remainder": remainder,
            "workers": worker_count,
            "worker_memory_limit": worker_memory_limit,
            "worker_threads": worker_threads,
            "runtime_environment": runtime_environment(),
        }
        atomic_write_json(root / "runs" / f"derive_5m_stage_{remainder}_{utc_now().replace(':', '').replace('-', '')}.json", payload)
        return payload
    final_shards = target_dir / "shards"
    target_dir.mkdir(parents=True, exist_ok=True)
    if final_shards.exists():
        shutil.rmtree(final_shards)
    if staging_shards.exists():
        staging_shards.replace(final_shards)
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    schema = _duckdb_schema(final_shards / shard_entries[0].path.split("/")[-1]) if shard_entries else []
    manifest = DatasetManifest(
        dataset_id=target_dataset_id,
        domain="market_intraday_5m",
        layer="raw",
        frequency="5m",
        contract_version="mootdx_5m_48_v1",
        primary_key=["trade_date", "symbol", "bar_time"],
        start_date=min([item.start_date for item in shard_entries if item.start_date], default=""),
        end_date=max([item.end_date for item in shard_entries if item.end_date], default=""),
        row_count=sum(int(item.row_count) for item in shard_entries),
        schema_hash=schema_hash(schema),
        schema=schema,
        shards=shard_entries,
        source={"provider": "qdp_v2", "created_by": "derive_5m_from_1m", "created_at": utc_now(), "source_dataset_id": source.dataset_id},
        quality={"path_refs_exist": True, "bar_count_contract": "48", "primary_key_unique": "not_checked"},
        notes=["derived from mootdx-style 1m 240 bars using 5-minute end labels"],
    )
    manifest_path = write_dataset_manifest(root, manifest)
    active_path = ""
    if activate_domain:
        active = read_active_manifest(root)
        if active:
            raw = dict(active.get("raw", {}) or {})
            raw["market_intraday_5m"] = target_dataset_id
            active["raw"] = raw
            active_path = str(write_active_manifest(root, active).resolve())
    payload = {
        "status": "ok",
        "target_dataset_id": target_dataset_id,
        "manifest_path": str(manifest_path.resolve()),
        "source_dataset_id": source.dataset_id,
        "processed_shards": processed,
        "new_shards": len(tasks),
        "reused_shards": reused,
        "reused_from_dataset_shards": reused_from_dataset,
        "row_count": manifest.row_count,
        "start_date": manifest.start_date,
        "end_date": manifest.end_date,
        "active_manifest": active_path,
        "max_shards": int(max_shards or 0),
        "workers": worker_count,
        "worker_memory_limit": worker_memory_limit,
        "worker_threads": worker_threads,
        "runtime_environment": runtime_environment(),
    }
    atomic_write_json(root / "runs" / f"derive_5m_{utc_now().replace(':', '').replace('-', '')}.json", payload)
    return payload


def split_daily_market(
    *,
    workspace_root: str | Path | None = None,
    source_dataset_id: str = "",
    source_domain: str = "market_daily_panel",
    max_shards: int = 0,
    runtime: str = "balanced",
    activate_domain: bool = False,
) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    source_manifest = _resolve_source_manifest(root, source_dataset_id, source_domain)
    source = read_dataset_manifest(source_manifest)
    profile = resolve_runtime_profile(runtime)
    raw_dataset_id = f"market_daily_raw__{stable_hash({'source': source.dataset_id, 'contract': 'qdp_v2_market_daily_raw_v1'})}"
    panel_dataset_id = f"market_daily_panel__{stable_hash({'source': source.dataset_id, 'contract': 'qdp_v2_market_daily_panel_v1'})}"
    raw_staging = root / "datasets" / "market_daily_raw" / raw_dataset_id / ".staging" / "shards"
    panel_staging = root / "datasets" / "market_daily_panel" / panel_dataset_id / ".staging" / "shards"
    for path in (raw_staging.parent, panel_staging.parent):
        if path.exists():
            shutil.rmtree(path)
    raw_staging.mkdir(parents=True, exist_ok=True)
    panel_staging.mkdir(parents=True, exist_ok=True)
    raw_entries: list[ShardManifestEntry] = []
    panel_entries: list[ShardManifestEntry] = []
    errors: list[dict[str, Any]] = []
    processed = 0
    for index, shard in enumerate(source.shards):
        if max_shards and processed >= max_shards:
            break
        source_path = root / shard.path if not Path(shard.path).is_absolute() else Path(shard.path)
        raw_path = raw_staging / f"part_{index:06d}_daily_raw.parquet"
        panel_path = panel_staging / f"part_{index:06d}_daily_panel.parquet"
        try:
            stats = _split_daily_market_shard(
                source_path=source_path,
                raw_target_path=raw_path,
                panel_target_path=panel_path,
                memory_limit=profile.duckdb_memory_limit,
                threads=profile.duckdb_threads,
            )
            raw_entries.append(
                ShardManifestEntry(
                    path=f"datasets/market_daily_raw/{raw_dataset_id}/shards/{raw_path.name}",
                    row_count=int(stats["raw_row_count"]),
                    start_date=shard.start_date,
                    end_date=shard.end_date,
                    status="stored",
                    file_size=int(raw_path.stat().st_size),
                    schema_hash=schema_hash(_duckdb_schema(raw_path)),
                    source_path=str(source_path.resolve()),
                    content_key="market_daily_raw",
                )
            )
            panel_entries.append(
                ShardManifestEntry(
                    path=f"datasets/market_daily_panel/{panel_dataset_id}/shards/{panel_path.name}",
                    row_count=int(stats["panel_row_count"]),
                    start_date=shard.start_date,
                    end_date=shard.end_date,
                    status="stored",
                    file_size=int(panel_path.stat().st_size),
                    schema_hash=schema_hash(_duckdb_schema(panel_path)),
                    source_path=str(source_path.resolve()),
                    content_key="market_daily_panel",
                )
            )
            processed += 1
        except Exception as exc:
            errors.append({"source_path": str(source_path), "error": str(exc)})
    if errors:
        return {"status": "error", "processed_shards": processed, "errors": errors[:50], "error_count": len(errors)}
    raw_final = root / "datasets" / "market_daily_raw" / raw_dataset_id / "shards"
    panel_final = root / "datasets" / "market_daily_panel" / panel_dataset_id / "shards"
    _replace_staging_shards(raw_staging, raw_final)
    _replace_staging_shards(panel_staging, panel_final)
    raw_schema = _duckdb_schema(raw_final / raw_entries[0].path.split("/")[-1]) if raw_entries else []
    panel_schema = _duckdb_schema(panel_final / panel_entries[0].path.split("/")[-1]) if panel_entries else []
    starts = [item.start_date for item in panel_entries if item.start_date]
    ends = [item.end_date for item in panel_entries if item.end_date]
    raw_manifest = DatasetManifest(
        dataset_id=raw_dataset_id,
        domain="market_daily_raw",
        layer="raw",
        frequency="1d",
        contract_version="qdp_v2_market_daily_raw_v1",
        primary_key=["trade_date", "symbol"],
        start_date=min(starts) if starts else "",
        end_date=max(ends) if ends else "",
        row_count=sum(int(item.row_count) for item in raw_entries),
        schema_hash=schema_hash(raw_schema),
        schema=raw_schema,
        shards=raw_entries,
        source={"provider": "qdp_v2", "created_by": "split_daily_market", "created_at": utc_now(), "source_dataset_id": source.dataset_id},
        quality={"path_refs_exist": True, "ohlcv_non_null": True, "primary_key_unique": "not_checked"},
        notes=["raw daily facts filtered from legacy rectangular market panel; null OHLCV rows are excluded"],
    )
    panel_manifest = DatasetManifest(
        dataset_id=panel_dataset_id,
        domain="market_daily_panel",
        layer="research_panel",
        frequency="1d",
        contract_version="qdp_v2_market_daily_panel_v1",
        primary_key=["trade_date", "symbol"],
        start_date=min(starts) if starts else "",
        end_date=max(ends) if ends else "",
        row_count=sum(int(item.row_count) for item in panel_entries),
        schema_hash=schema_hash(panel_schema),
        schema=panel_schema,
        shards=panel_entries,
        source={"provider": "qdp_v2", "created_by": "split_daily_market", "created_at": utc_now(), "source_dataset_id": source.dataset_id},
        quality={"path_refs_exist": True, "has_bar_contract": True, "primary_key_unique": "not_checked"},
        notes=["research panel keeps rectangular rows and marks missing bars with has_bar=false"],
    )
    raw_manifest_path = write_dataset_manifest(root, raw_manifest)
    panel_manifest_path = write_dataset_manifest(root, panel_manifest)
    active_path = ""
    if activate_domain:
        active = read_active_manifest(root)
        if active:
            raw = dict(active.get("raw", {}) or {})
            research_panels = dict(active.get("research_panels", {}) or {})
            raw["market_daily_raw"] = raw_dataset_id
            research_panels["market_daily_panel"] = panel_dataset_id
            active["raw"] = raw
            active["research_panels"] = research_panels
            active_path = str(write_active_manifest(root, active).resolve())
    payload = {
        "status": "ok",
        "source_dataset_id": source.dataset_id,
        "raw_dataset_id": raw_dataset_id,
        "panel_dataset_id": panel_dataset_id,
        "raw_manifest_path": str(raw_manifest_path.resolve()),
        "panel_manifest_path": str(panel_manifest_path.resolve()),
        "processed_shards": processed,
        "raw_row_count": raw_manifest.row_count,
        "panel_row_count": panel_manifest.row_count,
        "start_date": panel_manifest.start_date,
        "end_date": panel_manifest.end_date,
        "active_manifest": active_path,
        "max_shards": int(max_shards or 0),
        "runtime_environment": runtime_environment(),
    }
    atomic_write_json(root / "runs" / f"split_daily_market_{utc_now().replace(':', '').replace('-', '')}.json", payload)
    return payload


def normalize_valuation(
    *,
    workspace_root: str | Path | None = None,
    source_dataset_id: str = "",
    max_shards: int = 0,
    runtime: str = "balanced",
    activate_domain: bool = False,
) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    source_manifest = _resolve_source_manifest(root, source_dataset_id, "valuation")
    source = read_dataset_manifest(source_manifest)
    profile = resolve_runtime_profile(runtime)
    target_dataset_id = f"valuation__{stable_hash({'source': source.dataset_id, 'contract': 'qdp_v2_valuation_v1'})}"
    target_dir = root / "datasets" / "valuation" / target_dataset_id
    staging = target_dir / ".staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging_shards = staging / "shards"
    staging_shards.mkdir(parents=True, exist_ok=True)
    processed = 0
    entries: list[ShardManifestEntry] = []
    errors: list[dict[str, Any]] = []
    for index, shard in enumerate(source.shards):
        if max_shards and processed >= max_shards:
            break
        source_path = root / shard.path if not Path(shard.path).is_absolute() else Path(shard.path)
        target_path = staging_shards / f"part_{index:06d}_valuation.parquet"
        try:
            stats = _normalize_valuation_shard(source_path=source_path, target_path=target_path, memory_limit=profile.duckdb_memory_limit, threads=profile.duckdb_threads)
            entries.append(
                ShardManifestEntry(
                    path=f"datasets/valuation/{target_dataset_id}/shards/{target_path.name}",
                    row_count=int(stats["row_count"]),
                    start_date=shard.start_date,
                    end_date=shard.end_date,
                    status="stored",
                    file_size=int(target_path.stat().st_size),
                    schema_hash=str(stats.get("schema_hash", "") or ""),
                    source_path=str(source_path.resolve()),
                    content_key="normalized_valuation",
                )
            )
            processed += 1
        except Exception as exc:
            errors.append({"source_path": str(source_path), "target_path": str(target_path), "error": str(exc)})
    if errors:
        return {"status": "error", "target_dataset_id": target_dataset_id, "processed_shards": processed, "errors": errors[:50], "error_count": len(errors)}
    final_shards = target_dir / "shards"
    target_dir.mkdir(parents=True, exist_ok=True)
    if final_shards.exists():
        shutil.rmtree(final_shards)
    if staging_shards.exists():
        staging_shards.replace(final_shards)
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    schema = _duckdb_schema(final_shards / entries[0].path.split("/")[-1]) if entries else []
    manifest = DatasetManifest(
        dataset_id=target_dataset_id,
        domain="valuation",
        layer="raw",
        frequency="1d",
        contract_version="qdp_v2_valuation_v1",
        primary_key=["trade_date", "symbol"],
        start_date=min([item.start_date for item in entries if item.start_date], default=""),
        end_date=max([item.end_date for item in entries if item.end_date], default=""),
        row_count=sum(int(item.row_count) for item in entries),
        schema_hash=schema_hash(schema),
        schema=schema,
        shards=entries,
        source={"provider": "qdp_v2", "created_by": "normalize_valuation", "created_at": utc_now(), "source_dataset_id": source.dataset_id},
        quality={"path_refs_exist": True, "schema_contract": "qdp_v2_valuation_v1", "primary_key_unique": "not_checked"},
    )
    manifest_path = write_dataset_manifest(root, manifest)
    active_path = ""
    if activate_domain:
        active = read_active_manifest(root)
        if active:
            raw = dict(active.get("raw", {}) or {})
            raw["valuation"] = target_dataset_id
            active["raw"] = raw
            active_path = str(write_active_manifest(root, active).resolve())
    payload = {
        "status": "ok",
        "target_dataset_id": target_dataset_id,
        "manifest_path": str(manifest_path.resolve()),
        "source_dataset_id": source.dataset_id,
        "processed_shards": processed,
        "row_count": manifest.row_count,
        "start_date": manifest.start_date,
        "end_date": manifest.end_date,
        "active_manifest": active_path,
        "max_shards": int(max_shards or 0),
        "runtime_environment": runtime_environment(),
    }
    atomic_write_json(root / "runs" / f"normalize_valuation_{utc_now().replace(':', '').replace('-', '')}.json", payload)
    return payload


def repair_daily_from_intraday_missing(
    *,
    workspace_root: str | Path | None = None,
    raw_dataset_id: str = "",
    panel_dataset_id: str = "",
    five_minute_dataset_id: str = "",
    runtime: str = "balanced",
    duckdb_memory_limit: str = "",
    activate_domain: bool = False,
) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    active = read_active_manifest(root)
    raw_id = raw_dataset_id or str(dict(active.get("raw", {}) or {}).get("market_daily_raw", "") or "")
    panel_id = panel_dataset_id or str(dict(active.get("research_panels", {}) or {}).get("market_daily_panel", "") or "")
    five_id = five_minute_dataset_id or str(dict(active.get("raw", {}) or {}).get("market_intraday_5m", "") or "")
    raw = read_dataset_manifest(_require_dataset_manifest(root, raw_id, "market_daily_raw"))
    panel = read_dataset_manifest(_require_dataset_manifest(root, panel_id, "market_daily_panel"))
    five = read_dataset_manifest(_require_dataset_manifest(root, five_id, "market_intraday_5m"))
    profile = resolve_runtime_profile(runtime)
    memory_limit = str(duckdb_memory_limit or "").strip() or profile.duckdb_memory_limit
    missing = _daily_missing_from_intraday_rows(root=root, raw=raw, five=five, memory_limit=memory_limit, threads=profile.duckdb_threads)
    if missing.empty:
        return {
            "status": "ok",
            "message": "no_nonzero_intraday_daily_missing_rows",
            "raw_dataset_id": raw.dataset_id,
            "panel_dataset_id": panel.dataset_id,
            "five_minute_dataset_id": five.dataset_id,
        }
    missing_hash = stable_hash({"rows": missing.to_dict("records")}, length=24)
    raw_target_id = f"market_daily_raw__{stable_hash({'source': raw.dataset_id, 'repair': 'intraday_missing_v1', 'five': five.dataset_id, 'missing': missing_hash})}"
    panel_target_id = f"market_daily_panel__{stable_hash({'source': panel.dataset_id, 'repair': 'intraday_missing_v1', 'five': five.dataset_id, 'missing': missing_hash})}"
    raw_dir = root / "datasets" / "market_daily_raw" / raw_target_id
    panel_dir = root / "datasets" / "market_daily_panel" / panel_target_id
    raw_staging = raw_dir / ".staging" / "shards"
    panel_staging = panel_dir / ".staging" / "shards"
    for path in (raw_dir / ".staging", panel_dir / ".staging"):
        if path.exists():
            shutil.rmtree(path)
    raw_staging.mkdir(parents=True, exist_ok=True)
    panel_staging.mkdir(parents=True, exist_ok=True)
    raw_target = raw_staging / "part_000000_market_daily_raw_mainboard.parquet"
    panel_target = panel_staging / "part_000000_market_daily_panel_mainboard.parquet"
    stats = _write_daily_missing_repair_outputs(
        root=root,
        raw=raw,
        panel=panel,
        missing=missing,
        raw_target=raw_target,
        panel_target=panel_target,
        memory_limit=memory_limit,
        threads=profile.duckdb_threads,
    )
    raw_final = raw_dir / "shards"
    panel_final = panel_dir / "shards"
    raw_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    if raw_final.exists():
        shutil.rmtree(raw_final)
    if panel_final.exists():
        shutil.rmtree(panel_final)
    raw_staging.replace(raw_final)
    panel_staging.replace(panel_final)
    shutil.rmtree(raw_dir / ".staging", ignore_errors=True)
    shutil.rmtree(panel_dir / ".staging", ignore_errors=True)
    raw_path = raw_final / raw_target.name
    panel_path = panel_final / panel_target.name
    raw_schema = _duckdb_schema(raw_path)
    panel_schema = _duckdb_schema(panel_path)
    raw_manifest = DatasetManifest(
        dataset_id=raw_target_id,
        domain="market_daily_raw",
        layer="raw",
        frequency="1d",
        contract_version=raw.contract_version or "qdp_v2_market_daily_raw_v1",
        primary_key=["trade_date", "symbol"],
        start_date=str(stats["raw_start_date"]),
        end_date=str(stats["raw_end_date"]),
        row_count=int(stats["raw_row_count"]),
        schema_hash=schema_hash(raw_schema),
        schema=raw_schema,
        shards=[
            ShardManifestEntry(
                path=f"datasets/market_daily_raw/{raw_target_id}/shards/{raw_path.name}",
                row_count=int(stats["raw_row_count"]),
                start_date=str(stats["raw_start_date"]),
                end_date=str(stats["raw_end_date"]),
                status="stored",
                file_size=int(raw_path.stat().st_size),
                schema_hash=schema_hash(raw_schema),
                source_path=str(_resolve_data_path(root, raw.shards[0].path).resolve()) if raw.shards else "",
                content_key="market_daily_raw",
                metadata={"repair": "daily_missing_from_intraday_5m"},
            )
        ],
        source={"provider": "qdp_v2", "created_by": "repair_daily_from_intraday_missing", "created_at": utc_now(), "source_dataset_id": raw.dataset_id, "five_minute_dataset_id": five.dataset_id, "missing_hash": missing_hash},
        quality={**dict(raw.quality or {}), "intraday_missing_daily_repair": "applied", "repaired_rows": int(len(missing)), "primary_key_unique": "not_checked_after_repair"},
        notes=[*list(raw.notes or []), "nonzero intraday symbol-days missing from daily raw were filled from 5m aggregation"],
    )
    panel_manifest = DatasetManifest(
        dataset_id=panel_target_id,
        domain="market_daily_panel",
        layer="research_panel",
        frequency="1d",
        contract_version=panel.contract_version or "qdp_v2_market_daily_panel_v1",
        primary_key=["trade_date", "symbol"],
        start_date=str(stats["panel_start_date"]),
        end_date=str(stats["panel_end_date"]),
        row_count=int(stats["panel_row_count"]),
        schema_hash=schema_hash(panel_schema),
        schema=panel_schema,
        shards=[
            ShardManifestEntry(
                path=f"datasets/market_daily_panel/{panel_target_id}/shards/{panel_path.name}",
                row_count=int(stats["panel_row_count"]),
                start_date=str(stats["panel_start_date"]),
                end_date=str(stats["panel_end_date"]),
                status="stored",
                file_size=int(panel_path.stat().st_size),
                schema_hash=schema_hash(panel_schema),
                source_path=str(_resolve_data_path(root, panel.shards[0].path).resolve()) if panel.shards else "",
                content_key="market_daily_panel",
                metadata={"repair": "daily_missing_from_intraday_5m"},
            )
        ],
        source={"provider": "qdp_v2", "created_by": "repair_daily_from_intraday_missing", "created_at": utc_now(), "source_dataset_id": panel.dataset_id, "five_minute_dataset_id": five.dataset_id, "missing_hash": missing_hash},
        quality={**dict(panel.quality or {}), "intraday_missing_daily_repair": "applied", "repaired_rows": int(len(missing)), "primary_key_unique": "not_checked_after_repair"},
        notes=[*list(panel.notes or []), "panel no_bar rows were updated when nonzero intraday bars exist"],
    )
    raw_manifest_path = write_dataset_manifest(root, raw_manifest)
    panel_manifest_path = write_dataset_manifest(root, panel_manifest)
    active_path = ""
    if activate_domain:
        active = read_active_manifest(root)
        raw_section = dict(active.get("raw", {}) or {})
        panel_section = dict(active.get("research_panels", {}) or {})
        raw_section["market_daily_raw"] = raw_target_id
        panel_section["market_daily_panel"] = panel_target_id
        active["raw"] = raw_section
        active["research_panels"] = panel_section
        active["updated_at"] = utc_now()
        active_path = str(write_active_manifest(root, active).resolve())
    payload = {
        "status": "ok",
        "raw_dataset_id": raw_target_id,
        "panel_dataset_id": panel_target_id,
        "raw_manifest_path": str(raw_manifest_path.resolve()),
        "panel_manifest_path": str(panel_manifest_path.resolve()),
        "five_minute_dataset_id": five.dataset_id,
        "repaired_rows": int(len(missing)),
        "symbols": sorted(missing["symbol"].astype(str).unique().tolist()),
        "dates": sorted(missing["trade_date"].astype(str).unique().tolist()),
        "raw_row_count": raw_manifest.row_count,
        "panel_row_count": panel_manifest.row_count,
        "active_manifest": active_path,
    }
    atomic_write_json(root / "runs" / f"repair_daily_from_intraday_missing_{utc_now().replace(':', '').replace('-', '')}.json", payload)
    return payload


def scope_mainboard_active(
    *,
    workspace_root: str | Path | None = None,
    as_of_date: str = "",
    domains: str = "",
    max_shards: int = 0,
    runtime: str = "balanced",
    workers: int = 1,
    resume: bool = False,
    trust_existing: bool = False,
    duckdb_memory_limit: str = "",
) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    active = read_active_manifest(root)
    active_as_of = str(as_of_date or active.get("active_as_of_date", "") or "").strip()
    if not active_as_of:
        raise ValueError("active_as_of_date_required")
    profile = resolve_runtime_profile(runtime)
    memory_limit = str(duckdb_memory_limit or "").strip() or profile.duckdb_memory_limit
    selected_domains = _scope_domain_list(domains, active)
    scope_symbols_path, scope_stats = _write_mainboard_scope_symbols(root=root, active=active, active_as_of=active_as_of)
    processed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for domain in selected_domains:
        source_id = _active_dataset_id_for_domain(active, domain)
        if not source_id:
            skipped.append({"domain": domain, "reason": "domain_not_active"})
            continue
        manifest_path = dataset_manifest_for_id(root, source_id, domain)
        if not manifest_path:
            errors.append({"domain": domain, "dataset_id": source_id, "error": "manifest_not_found"})
            continue
        source = read_dataset_manifest(manifest_path)
        if not _manifest_has_column(source, "symbol"):
            skipped.append({"domain": domain, "dataset_id": source_id, "reason": "no_symbol_column"})
            continue
        result = _scope_dataset_by_symbols(
            root=root,
            source=source,
            scope_symbols_path=scope_symbols_path,
            scope_name="mainboard_hs_a_ex_current_st_delisted_v1",
            active_as_of=active_as_of,
            max_shards=max_shards,
            memory_limit=memory_limit,
            duckdb_threads=profile.duckdb_threads,
            workers=workers,
            resume=resume,
            trust_existing=trust_existing,
            scope_stats=scope_stats,
        )
        if result.get("status") == "ok":
            processed.append(result)
        else:
            errors.append({"domain": domain, "dataset_id": source_id, "error": result})
    payload = {
        "status": "error" if errors else "ok",
        "active_as_of_date": active_as_of,
        "scope": "mainboard_hs_a_ex_current_st_delisted_v1",
        "scope_symbols_path": str(scope_symbols_path.resolve()),
        "scope_stats": scope_stats,
        "processed": processed,
        "skipped": skipped,
        "errors": errors[:50],
        "error_count": len(errors),
        "runtime_environment": runtime_environment(),
    }
    atomic_write_json(root / "runs" / f"scope_mainboard_{utc_now().replace(':', '').replace('-', '')}.json", payload)
    return payload


def _resolve_source_manifest(root: Path, dataset_id: str, domain: str) -> Path:
    if dataset_id:
        path = dataset_manifest_for_id(root, dataset_id, domain)
        if path is None:
            raise FileNotFoundError(f"dataset_manifest_not_found:{domain}:{dataset_id}")
        return path
    active = read_active_manifest(root)
    for section in ("raw", "derived", "research_panels"):
        active_id = str(dict(active.get(section, {}) or {}).get(domain, "") or "")
        if active_id:
            path = dataset_manifest_for_id(root, active_id, domain)
            if path is not None:
                return path
    candidates = sorted((root / "datasets" / domain).glob("*/dataset.json"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"source_dataset_required:{domain}")


def _replace_staging_shards(staging_shards: Path, final_shards: Path) -> None:
    staging_root = staging_shards.parent
    final_shards.parent.mkdir(parents=True, exist_ok=True)
    if final_shards.exists():
        shutil.rmtree(final_shards)
    if staging_shards.exists():
        staging_shards.replace(final_shards)
    if staging_root.exists():
        shutil.rmtree(staging_root, ignore_errors=True)


def _scope_domain_list(domains: str, active: dict[str, Any]) -> list[str]:
    text = str(domains or "").strip()
    if text:
        return [item.strip() for item in text.split(",") if item.strip()]
    ordered: list[str] = []
    for section in ("raw", "derived", "research_panels"):
        for domain in dict(active.get(section, {}) or {}):
            if domain not in ordered and domain != "trading_calendar":
                ordered.append(domain)
    return ordered


def _active_dataset_id_for_domain(active: dict[str, Any], domain: str) -> str:
    for section in ("raw", "derived", "research_panels"):
        value = dict(active.get(section, {}) or {}).get(domain)
        if value:
            return str(value)
    return ""


def _manifest_has_column(manifest: DatasetManifest, column: str) -> bool:
    wanted = str(column).lower()
    return any(str(item.get("name", "")).lower() == wanted for item in manifest.schema)


def _write_mainboard_scope_symbols(*, root: Path, active: dict[str, Any], active_as_of: str) -> tuple[Path, dict[str, Any]]:
    universe_id = _active_dataset_id_for_domain(active, "universe_snapshot")
    status_id = _active_dataset_id_for_domain(active, "security_status")
    universe = read_dataset_manifest(_require_dataset_manifest(root, universe_id, "universe_snapshot"))
    status = read_dataset_manifest(_require_dataset_manifest(root, status_id, "security_status"))
    universe_paths = [str(_resolve_qdp_v2_path(root, shard.path)) for shard in universe.shards]
    status_paths = [str(_resolve_qdp_v2_path(root, shard.path)) for shard in status.shards]
    target = root / "runs" / f"mainboard_scope_symbols_{stable_hash({'as_of': active_as_of, 'universe': universe.dataset_id, 'status': status.dataset_id})}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    with open_guarded_duckdb(
        temp_directory=root / "tmp" / "mainboard_scope_duckdb",
        threads=2,
    ) as con:
        con.execute(
            f"""
            copy (
                with u as (
                    select
                        upper(cast(symbol as varchar)) as symbol,
                        max(coalesce(cast(name as varchar), '')) as name
                    from read_parquet(?, union_by_name=true)
                    where cast(trade_date as varchar)[:10] = ?
                      and (
                        regexp_matches(upper(cast(symbol as varchar)), '^(600|601|603|605)[0-9]{{3}}\\.SH$')
                        or regexp_matches(upper(cast(symbol as varchar)), '^(000|001|002|003)[0-9]{{3}}\\.SZ$')
                      )
                    group by 1
                ),
                s as (
                    select
                        upper(cast(symbol as varchar)) as symbol,
                        bool_or(coalesce(cast(is_st as boolean), false)) as is_st,
                        bool_or(coalesce(cast(is_delisted as boolean), false)) as is_delisted
                    from read_parquet(?, union_by_name=true)
                    where cast(trade_date as varchar)[:10] = ?
                    group by 1
                )
                select u.symbol
                from u
                left join s using(symbol)
                where coalesce(s.is_st, false) = false
                  and coalesce(s.is_delisted, false) = false
                  and coalesce(u.name, '') not like '%退市%'
                order by u.symbol
            ) to {_sql_literal(str(target))} (format parquet)
            """,
            [universe_paths, active_as_of, status_paths, active_as_of],
        )
        stats = con.execute(
            """
            select
                count(*) as symbol_count,
                sum(case when symbol like '%.SH' then 1 else 0 end) as sh_count,
                sum(case when symbol like '%.SZ' then 1 else 0 end) as sz_count
            from read_parquet(?)
            """,
            [str(target)],
        ).fetchone()
    return target, {"symbol_count": int(stats[0] or 0), "sh_count": int(stats[1] or 0), "sz_count": int(stats[2] or 0)}


def _require_dataset_manifest(root: Path, dataset_id: str, domain: str) -> Path:
    path = dataset_manifest_for_id(root, dataset_id, domain)
    if not path:
        raise FileNotFoundError(f"dataset_manifest_not_found: {domain}:{dataset_id}")
    return path


def _resolve_qdp_v2_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _resolve_data_path(root: Path, path: str | Path) -> Path:
    return _resolve_qdp_v2_path(root, path).resolve()


def _daily_missing_from_intraday_rows(
    *,
    root: Path,
    raw: DatasetManifest,
    five: DatasetManifest,
    memory_limit: str,
    threads: int,
) -> "pd.DataFrame":
    import duckdb  # type: ignore

    raw_paths = [str(_resolve_data_path(root, item.path)) for item in raw.shards]
    five_paths = [str(_resolve_data_path(root, item.path)) for item in five.shards]
    with duckdb.connect(":memory:") as con:
        con.execute(f"set memory_limit='{memory_limit}'")
        con.execute(f"set threads={max(1, int(threads or 1))}")
        return con.execute(
            """
            with intraday as (
              select
                symbol,
                trade_date,
                arg_min(open, bar_time) as open,
                max(high) as high,
                min(low) as low,
                arg_max(close, bar_time) as close,
                sum(volume) as volume,
                sum(amount) as amount
              from read_parquet($five_paths, union_by_name=true)
              group by 1, 2
            ),
            daily as (
              select symbol, trade_date
              from read_parquet($raw_paths, union_by_name=true)
              group by 1, 2
            )
            select
              i.symbol,
              i.trade_date,
              cast(i.open as double) as open,
              cast(i.high as double) as high,
              cast(i.low as double) as low,
              cast(i.close as double) as close,
              cast(i.volume as double) as volume,
              cast(i.amount as double) as amount,
              'intraday_5m_missing_daily_repair' as source,
              'none' as adjusted_flag
            from intraday i
            left join daily d on d.symbol = i.symbol and d.trade_date = i.trade_date
            where d.symbol is null
              and (abs(coalesce(i.volume, 0)) > 0.0001 or abs(coalesce(i.amount, 0)) > 0.0001)
            order by i.trade_date, i.symbol
            """,
            {"five_paths": five_paths, "raw_paths": raw_paths},
        ).fetchdf()


def _write_daily_missing_repair_outputs(
    *,
    root: Path,
    raw: DatasetManifest,
    panel: DatasetManifest,
    missing: "pd.DataFrame",
    raw_target: Path,
    panel_target: Path,
    memory_limit: str,
    threads: int,
) -> dict[str, Any]:
    import duckdb  # type: ignore

    raw_paths = [str(_resolve_data_path(root, item.path)) for item in raw.shards]
    panel_paths = [str(_resolve_data_path(root, item.path)) for item in panel.shards]
    raw_target.parent.mkdir(parents=True, exist_ok=True)
    panel_target.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(":memory:") as con:
        con.execute(f"set memory_limit='{memory_limit}'")
        con.execute(f"set threads={max(1, int(threads or 1))}")
        con.register("missing_daily", missing)
        con.execute(
            f"""
            copy (
              select symbol, trade_date, open, high, low, close, volume, amount, source, adjusted_flag
              from read_parquet($raw_paths, union_by_name=true)
              union all
              select symbol, trade_date, open, high, low, close, volume, amount, source, adjusted_flag
              from missing_daily
              order by trade_date, symbol
            ) to {_sql_literal(str(raw_target))} (format parquet)
            """,
            {"raw_paths": raw_paths},
        )
        con.execute(
            f"""
            copy (
              with panel as (
                select *
                from read_parquet($panel_paths, union_by_name=true)
              ),
              updated as (
                select
                  p.symbol,
                  p.trade_date,
                  coalesce(m.open, p.open) as open,
                  coalesce(m.high, p.high) as high,
                  coalesce(m.low, p.low) as low,
                  coalesce(m.close, p.close) as close,
                  coalesce(m.volume, p.volume) as volume,
                  coalesce(m.amount, p.amount) as amount,
                  case when m.symbol is not null then m.source else p.source end as source,
                  case when m.symbol is not null then m.adjusted_flag else p.adjusted_flag end as adjusted_flag,
                  p.ingest_batch_id,
                  case when m.symbol is not null then true else p.has_bar end as has_bar,
                  case when m.symbol is not null then '' else p.reject_reason end as reject_reason
                from panel p
                left join missing_daily m on m.symbol = p.symbol and m.trade_date = p.trade_date
              ),
              appended as (
                select
                  m.symbol,
                  m.trade_date,
                  m.open,
                  m.high,
                  m.low,
                  m.close,
                  m.volume,
                  m.amount,
                  m.source,
                  m.adjusted_flag,
                  'qdp_v2_daily_from_intraday_missing' as ingest_batch_id,
                  true as has_bar,
                  '' as reject_reason
                from missing_daily m
                left join panel p on p.symbol = m.symbol and p.trade_date = m.trade_date
                where p.symbol is null
              )
              select * from updated
              union all
              select * from appended
              order by trade_date, symbol
            ) to {_sql_literal(str(panel_target))} (format parquet)
            """,
            {"panel_paths": panel_paths},
        )
        raw_count = con.execute("select count(*) from read_parquet(?)", [str(raw_target)]).fetchone()
        panel_count = con.execute("select count(*) from read_parquet(?)", [str(panel_target)]).fetchone()
        raw_start = con.execute("select trade_date from read_parquet(?) order by trade_date asc limit 1", [str(raw_target)]).fetchone()
        raw_end = con.execute("select trade_date from read_parquet(?) order by trade_date desc limit 1", [str(raw_target)]).fetchone()
        panel_start = con.execute("select trade_date from read_parquet(?) order by trade_date asc limit 1", [str(panel_target)]).fetchone()
        panel_end = con.execute("select trade_date from read_parquet(?) order by trade_date desc limit 1", [str(panel_target)]).fetchone()
    return {
        "raw_row_count": int(raw_count[0] or 0),
        "raw_start_date": str(raw_start[0] if raw_start else ""),
        "raw_end_date": str(raw_end[0] if raw_end else ""),
        "panel_row_count": int(panel_count[0] or 0),
        "panel_start_date": str(panel_start[0] if panel_start else ""),
        "panel_end_date": str(panel_end[0] if panel_end else ""),
    }


def _scope_dataset_by_symbols(
    *,
    root: Path,
    source: DatasetManifest,
    scope_symbols_path: Path,
    scope_name: str,
    active_as_of: str,
    max_shards: int,
    memory_limit: str,
    duckdb_threads: int,
    workers: int,
    resume: bool,
    trust_existing: bool,
    scope_stats: dict[str, Any],
) -> dict[str, Any]:
    target_dataset_id = f"{source.domain}__{stable_hash({'source': source.dataset_id, 'scope': scope_name, 'active_as_of': active_as_of})}"
    target_dir = root / "datasets" / source.domain / target_dataset_id
    staging = target_dir / ".staging"
    if staging.exists() and not resume:
        shutil.rmtree(staging)
    staging_shards = staging / "shards"
    staging_shards.mkdir(parents=True, exist_ok=True)
    requested_workers = max(1, int(workers or 1))
    worker_count = min(requested_workers, max(1, int(duckdb_threads or 1)), max(1, len(source.shards)))
    worker_memory_limit = _parallel_duckdb_memory_limit(memory_limit, worker_count)
    worker_threads = max(1, min(2, int(duckdb_threads or 1)))
    tasks: list[dict[str, Any]] = []
    entries_by_index: dict[int, ShardManifestEntry] = {}
    reused = 0
    selected = 0
    errors: list[dict[str, Any]] = []
    for index, shard in enumerate(source.shards):
        if max_shards and selected >= max_shards:
            break
        selected += 1
        source_path = _resolve_qdp_v2_path(root, shard.path)
        target_path = staging_shards / f"part_{index:06d}_{_safe_file_stem(source.domain)}_mainboard.parquet"
        if resume and target_path.exists():
            if trust_existing and int(target_path.stat().st_size) > 0:
                reused += 1
                continue
            try:
                stats = _parquet_file_stats(target_path)
                if int(stats["row_count"]) > 0:
                    entries_by_index[index] = _scoped_shard_entry(
                        source_domain=source.domain,
                        target_dataset_id=target_dataset_id,
                        target_path=target_path,
                        source_path=source_path,
                        row_count=int(stats["row_count"]),
                        schema_hash_value=str(stats.get("schema_hash", "") or ""),
                        shard=shard,
                        scope_name=scope_name,
                    )
                    reused += 1
                    continue
            except Exception:
                _remove_file_with_retries(target_path)
        tasks.append(
            {
                "index": index,
                "source_path": str(source_path),
                "target_path": str(target_path),
                "scope_symbols_path": str(scope_symbols_path),
                "memory_limit": worker_memory_limit,
                "threads": worker_threads,
                "source_domain": source.domain,
                "target_dataset_id": target_dataset_id,
                "scope_name": scope_name,
                "shard": shard.to_dict(),
            }
        )
    if worker_count <= 1 or len(tasks) <= 1:
        for task in tasks:
            result = _filter_symbol_shard_task(task)
            if result.get("status") == "ok":
                entries_by_index[int(result["index"])] = result["entry"]
            elif result.get("status") != "empty":
                errors.append(result)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(_filter_symbol_shard_task, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                if result.get("status") == "ok":
                    entries_by_index[int(result["index"])] = result["entry"]
                elif result.get("status") != "empty":
                    errors.append(result)
    if errors:
        return {"status": "error", "domain": source.domain, "source_dataset_id": source.dataset_id, "target_dataset_id": target_dataset_id, "errors": errors[:50], "error_count": len(errors)}
    entries = [entries_by_index[key] for key in sorted(entries_by_index)]
    final_shards = target_dir / "shards"
    target_dir.mkdir(parents=True, exist_ok=True)
    if final_shards.exists():
        shutil.rmtree(final_shards)
    if staging_shards.exists():
        staging_shards.replace(final_shards)
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    schema = _duckdb_schema(final_shards / entries[0].path.split("/")[-1]) if entries else source.schema
    manifest = DatasetManifest(
        dataset_id=target_dataset_id,
        domain=source.domain,
        layer=source.layer,
        frequency=source.frequency,
        contract_version=source.contract_version,
        primary_key=source.primary_key,
        start_date=min([item.start_date for item in entries if item.start_date], default=source.start_date),
        end_date=max([item.end_date for item in entries if item.end_date], default=source.end_date),
        row_count=sum(int(item.row_count) for item in entries),
        schema_hash=schema_hash(schema),
        schema=schema,
        shards=entries,
        source={"provider": "qdp_v2", "created_by": "scope_mainboard_active", "created_at": utc_now(), "source_dataset_id": source.dataset_id, "scope": scope_name, "active_as_of_date": active_as_of},
        quality=_scoped_manifest_quality(source.quality, scope_name=scope_name, scope_stats=scope_stats),
        notes=[*source.notes, f"filtered to {scope_name} as of {active_as_of}"],
    )
    manifest_path = write_dataset_manifest(root, manifest)
    return {
        "status": "ok",
        "domain": source.domain,
        "source_dataset_id": source.dataset_id,
        "target_dataset_id": target_dataset_id,
        "manifest_path": str(manifest_path.resolve()),
        "processed_shards": len(entries),
        "new_shards": len(tasks),
        "reused_shards": reused,
        "row_count_before": source.row_count,
        "row_count_after": manifest.row_count,
        "row_count_removed": int(source.row_count) - int(manifest.row_count),
        "size_bytes": sum(int(item.file_size) for item in entries),
        "start_date": manifest.start_date,
        "end_date": manifest.end_date,
        "workers": worker_count,
        "worker_memory_limit": worker_memory_limit,
        "worker_threads": worker_threads,
    }


def _scoped_manifest_quality(source_quality: Mapping[str, Any], *, scope_name: str, scope_stats: Mapping[str, Any]) -> dict[str, Any]:
    stale_keys = {
        "date_coverage_audit",
        "meta_domain_quality",
        "primary_key_audit",
        "source_factor_quality",
    }
    quality = {str(key): value for key, value in dict(source_quality or {}).items() if str(key) not in stale_keys}
    quality.update(
        {
            "scope": scope_name,
            "scope_symbol_count": int(scope_stats.get("symbol_count", 0) or 0),
            "path_refs_exist": True,
            "primary_key_unique": bool(dict(source_quality or {}).get("primary_key_unique", False)),
            "primary_key_audit": {
                "method": "inherited_from_unique_source_after_symbol_scope_filter",
                "status": "passed" if bool(dict(source_quality or {}).get("primary_key_unique", False)) else "not_proven",
            },
        }
    )
    return quality


def _filter_symbol_shard_task(task: dict[str, Any]) -> dict[str, Any]:
    import duckdb  # type: ignore

    index = int(task["index"])
    source_path = Path(str(task["source_path"]))
    target_path = Path(str(task["target_path"]))
    scope_symbols_path = Path(str(task["scope_symbols_path"]))
    shard = ShardManifestEntry.from_mapping(task.get("shard", {}) or {})
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(":memory:") as con:
            con.execute(f"set memory_limit='{str(task['memory_limit'])}'")
            con.execute(f"set threads={max(1, int(task['threads']))}")
            con.execute(
                f"""
                copy (
                    select src.*
                    from read_parquet({_sql_literal(str(source_path))}, union_by_name=true) src
                    inner join read_parquet({_sql_literal(str(scope_symbols_path))}) scope
                      on upper(cast(src.symbol as varchar)) = scope.symbol
                ) to {_sql_literal(str(target_path))} (format parquet)
                """
            )
        stats = _parquet_file_stats(target_path)
        if int(stats["row_count"]) <= 0:
            _remove_file_with_retries(target_path)
            return {"status": "empty", "index": index, "source_path": str(source_path)}
        entry = _scoped_shard_entry(
            source_domain=str(task["source_domain"]),
            target_dataset_id=str(task["target_dataset_id"]),
            target_path=target_path,
            source_path=source_path,
            row_count=int(stats["row_count"]),
            schema_hash_value=str(stats.get("schema_hash", "") or ""),
            shard=shard,
            scope_name=str(task["scope_name"]),
        )
        return {"status": "ok", "index": index, "entry": entry}
    except Exception as exc:
        return {"status": "error", "index": index, "source_path": str(source_path), "target_path": str(target_path), "error": str(exc)}


def _scoped_shard_entry(
    *,
    source_domain: str,
    target_dataset_id: str,
    target_path: Path,
    source_path: Path,
    row_count: int,
    schema_hash_value: str,
    shard: ShardManifestEntry,
    scope_name: str,
) -> ShardManifestEntry:
    return ShardManifestEntry(
        path=f"datasets/{source_domain}/{target_dataset_id}/shards/{target_path.name}",
        row_count=int(row_count),
        start_date=shard.start_date,
        end_date=shard.end_date,
        status="stored",
        file_size=int(target_path.stat().st_size),
        schema_hash=schema_hash_value,
        source_path=str(source_path.resolve()),
        content_key=scope_name,
        metadata={"source_content_key": shard.content_key},
    )


def _parquet_file_stats(target_path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        parquet_file = pq.ParquetFile(str(target_path))
        row_count = int(parquet_file.metadata.num_rows)
        schema = [{"name": str(field.name), "type": str(field.type)} for field in parquet_file.schema_arrow]
        return {"row_count": row_count, "schema_hash": schema_hash(schema)}
    except Exception:
        import duckdb  # type: ignore

        with duckdb.connect(":memory:") as con:
            row = con.execute("select count(*) as n from read_parquet(?)", [str(target_path)]).fetchone()
        return {"row_count": int(row[0] or 0), "schema_hash": schema_hash(_duckdb_schema(target_path))}


def _safe_file_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value or "dataset"))


def _split_daily_market_shard(
    *,
    source_path: Path,
    raw_target_path: Path,
    panel_target_path: Path,
    memory_limit: str,
    threads: int,
) -> dict[str, Any]:
    import duckdb  # type: ignore

    raw_target_path.parent.mkdir(parents=True, exist_ok=True)
    panel_target_path.parent.mkdir(parents=True, exist_ok=True)
    source_literal = _sql_literal(str(source_path))
    raw_literal = _sql_literal(str(raw_target_path))
    panel_literal = _sql_literal(str(panel_target_path))
    with duckdb.connect(":memory:") as con:
        con.execute(f"set memory_limit='{memory_limit}'")
        con.execute(f"set threads={max(1, int(threads))}")
        cols = {str(row[0]) for row in con.execute("describe select * from read_parquet(?)", [str(source_path)]).fetchall()}
        symbol_expr = _column_expr(cols, "symbol", aliases=("stock", "code", "stock_code"))
        date_expr = _column_expr(cols, "trade_date", aliases=("date",))
        source_expr = "source" if "source" in cols else "'legacy_market_panel'"
        adjusted_expr = "adjusted_flag" if "adjusted_flag" in cols else "''"
        ingest_expr = "ingest_batch_id" if "ingest_batch_id" in cols else "''"
        has_bar_expr = (
            "open is not null and high is not null and low is not null and close is not null "
            "and volume is not null and amount is not null"
        )
        base_select = f"""
            select
                cast({symbol_expr} as varchar) as symbol,
                cast({date_expr} as varchar) as trade_date,
                cast(open as double) as open,
                cast(high as double) as high,
                cast(low as double) as low,
                cast(close as double) as close,
                cast(volume as double) as volume,
                cast(amount as double) as amount,
                cast({source_expr} as varchar) as source,
                cast({adjusted_expr} as varchar) as adjusted_flag,
                cast({ingest_expr} as varchar) as ingest_batch_id,
                ({has_bar_expr}) as has_bar
            from read_parquet({source_literal})
        """
        con.execute(
            f"""
            copy (
                select
                    symbol,
                    trade_date,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    amount,
                    source,
                    adjusted_flag
                from ({base_select}) base
                where has_bar
            ) to {raw_literal} (format parquet)
            """
        )
        con.execute(
            f"""
            copy (
                select
                    symbol,
                    trade_date,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    amount,
                    source,
                    adjusted_flag,
                    ingest_batch_id,
                    has_bar,
                    case when has_bar then '' else 'no_bar' end as reject_reason
                from ({base_select}) base
            ) to {panel_literal} (format parquet)
            """
        )
    with duckdb.connect(":memory:") as stats_con:
        raw_row = stats_con.execute("select count(*) as n, min(trade_date), max(trade_date) from read_parquet(?)", [str(raw_target_path)]).fetchone()
        panel_row = stats_con.execute("select count(*) as n, min(trade_date), max(trade_date) from read_parquet(?)", [str(panel_target_path)]).fetchone()
    return {
        "raw_row_count": int(raw_row[0] or 0),
        "panel_row_count": int(panel_row[0] or 0),
        "start_date": str(panel_row[1] or raw_row[1] or ""),
        "end_date": str(panel_row[2] or raw_row[2] or ""),
    }


def _derive_5m_shard(*, source_path: Path, target_path: Path, memory_limit: str, threads: int) -> dict[str, Any]:
    import duckdb  # type: ignore

    target_path.parent.mkdir(parents=True, exist_ok=True)
    source_literal = _sql_literal(str(source_path))
    target_literal = _sql_literal(str(target_path))
    with duckdb.connect(":memory:") as con:
        con.execute(f"set memory_limit='{memory_limit}'")
        con.execute(f"set threads={max(1, int(threads))}")
        columns = {row[0] for row in con.execute("describe select * from read_parquet(?)", [str(source_path)]).fetchall()}
        source_expr = "any_value(source) as source" if "source" in columns else "'derived_from_1m_240' as source"
        adjusted_expr = "any_value(adjusted_flag) as adjusted_flag" if "adjusted_flag" in columns else "'' as adjusted_flag"
        sql = f"""
        copy (
            with raw as (
                select
                    *,
                    regexp_replace(cast(bar_time as varchar), '[^0-9]', '', 'g') as bt
                from read_parquet({source_literal})
            ),
            parsed as (
                select
                    *,
                    cast(substr(bt, 1, 2) as integer) * 60 + cast(substr(bt, 3, 2) as integer) as minute_of_day
                from raw
                where length(bt) >= 4
            ),
            bucketed as (
                select
                    *,
                    case
                        when minute_of_day between 571 and 690 then 570 + cast(ceil((minute_of_day - 570) / 5.0) * 5 as integer)
                        when minute_of_day between 781 and 900 then 780 + cast(ceil((minute_of_day - 780) / 5.0) * 5 as integer)
                        else null
                    end as bucket_end
                from parsed
            ),
            grouped as (
                select
                    symbol,
                    trade_date,
                    printf('%02d%02d00000', cast(floor(bucket_end / 60) as integer), cast(bucket_end % 60 as integer)) as bar_time,
                    arg_min(open, minute_of_day) as open,
                    max(high) as high,
                    min(low) as low,
                    arg_max(close, minute_of_day) as close,
                    sum(volume) as volume,
                    sum(amount) as amount,
                    {source_expr},
                    {adjusted_expr}
                from bucketed
                where bucket_end is not null
                group by symbol, trade_date, bucket_end
            )
            select * from grouped order by trade_date, symbol, bar_time
        ) to {target_literal} (format parquet)
        """
        con.execute(sql)
    with duckdb.connect(":memory:") as stats_con:
        row = stats_con.execute("select count(*) as n, min(trade_date) as start_date, max(trade_date) as end_date from read_parquet(?)", [str(target_path)]).fetchone()
    return {
        "row_count": int(row[0] or 0),
        "start_date": str(row[1] or ""),
        "end_date": str(row[2] or ""),
        "schema_hash": schema_hash(_duckdb_schema(target_path)),
    }


def _derive_5m_shard_task(task: dict[str, Any]) -> dict[str, Any]:
    index = int(task["index"])
    source_path = Path(str(task["source_path"]))
    target_path = Path(str(task["target_path"]))
    shard = ShardManifestEntry.from_mapping(task.get("shard", {}) or {})
    try:
        stats = _derive_5m_shard(
            source_path=source_path,
            target_path=target_path,
            memory_limit=str(task["memory_limit"]),
            threads=int(task["threads"]),
        )
        if int(stats["row_count"]) <= 0:
            return {"status": "empty", "index": index, "source_path": str(source_path), "target_path": str(target_path)}
        return {
            "status": "ok",
            "index": index,
            "entry": _five_minute_shard_entry(
                target_dataset_id=str(task["target_dataset_id"]),
                target_path=target_path,
                source_path=source_path,
                row_count=int(stats["row_count"]),
                schema_hash_value=str(stats.get("schema_hash", "") or ""),
                shard=shard,
            ),
        }
    except Exception as exc:
        return {"status": "error", "index": index, "source_path": str(source_path), "target_path": str(target_path), "error": str(exc)}


def _existing_5m_shard_stats(target_path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        import duckdb  # type: ignore

        with duckdb.connect(":memory:") as con:
            row = con.execute("select count(*) as n from read_parquet(?)", [str(target_path)]).fetchone()
        return {"row_count": int(row[0] or 0), "schema_hash": schema_hash(_duckdb_schema(target_path))}

    parquet_file = pq.ParquetFile(str(target_path))
    row_count = int(parquet_file.metadata.num_rows)
    arrow_schema = parquet_file.schema_arrow
    schema = [{"name": str(field.name), "type": str(field.type)} for field in arrow_schema]
    return {"row_count": row_count, "schema_hash": schema_hash(schema)}


def _remove_file_with_retries(path: Path, *, attempts: int = 8, delay_seconds: float = 0.25) -> None:
    last_error: Exception | None = None
    for _ in range(max(1, attempts)):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError as exc:
            last_error = exc
            gc.collect()
            time.sleep(delay_seconds)
    if last_error is not None:
        raise last_error


def _resolve_optional_manifest(root: Path, dataset_id: str, domain: str) -> Path | None:
    text = str(dataset_id or "").strip()
    if not text:
        return None
    return dataset_manifest_for_id(root, text, domain)


def _can_reuse_5m_shard_from_prior(*, source_shard: ShardManifestEntry, reuse_shard: ShardManifestEntry) -> bool:
    metadata = dict(source_shard.metadata or {})
    if metadata.get("repair") in {"zero_price_intraday_bars", "quality_intraday_bars"}:
        return False
    if metadata.get("quality_repair") == "applied":
        return False
    if int(reuse_shard.row_count or 0) <= 0:
        return False
    return True


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


def _five_minute_shard_entry(
    *,
    target_dataset_id: str,
    target_path: Path,
    source_path: Path,
    row_count: int,
    schema_hash_value: str,
    shard: ShardManifestEntry,
) -> ShardManifestEntry:
    return ShardManifestEntry(
        path=f"datasets/market_intraday_5m/{target_dataset_id}/shards/{target_path.name}",
        row_count=int(row_count),
        start_date=shard.start_date,
        end_date=shard.end_date,
        status="stored",
        file_size=int(target_path.stat().st_size),
        schema_hash=schema_hash_value,
        source_path=str(source_path.resolve()),
        content_key="derived_from_1m_240",
    )


def _parallel_duckdb_memory_limit(profile_limit: str, workers: int) -> str:
    worker_count = max(1, int(workers or 1))
    profile_gb = _memory_limit_gb(profile_limit)
    if worker_count <= 1:
        return f"{profile_gb}GB"
    total_budget_gb = min(14, max(profile_gb, 12))
    per_worker_gb = max(2, int(total_budget_gb / worker_count))
    return f"{per_worker_gb}GB"


def _memory_limit_gb(value: str) -> int:
    text = str(value or "").strip().upper()
    if text.endswith("GB"):
        return max(1, int(float(text[:-2])))
    if text.endswith("G"):
        return max(1, int(float(text[:-1])))
    return 4


def _normalize_valuation_shard(*, source_path: Path, target_path: Path, memory_limit: str, threads: int) -> dict[str, Any]:
    import duckdb  # type: ignore

    target_path.parent.mkdir(parents=True, exist_ok=True)
    source_literal = _sql_literal(str(source_path))
    target_literal = _sql_literal(str(target_path))
    with duckdb.connect(":memory:") as con:
        con.execute(f"set memory_limit='{memory_limit}'")
        con.execute(f"set threads={max(1, int(threads))}")
        cols = {str(row[0]) for row in con.execute("describe select * from read_parquet(?)", [str(source_path)]).fetchall()}
        exprs = [
            _column_expr(cols, "symbol", aliases=("code", "stock_code")),
            _column_expr(cols, "trade_date", aliases=("date",)),
            _column_expr(cols, "total_mv", aliases=("totalMarketValue", "total_mv")),
            _column_expr(cols, "circ_mv", aliases=("circulatingMarketValue", "circ_mv")),
            _column_expr(cols, "pe", aliases=("peTTM", "pe_ttm", "pe")),
            _column_expr(cols, "pb", aliases=("pbMRQ", "pb_mrq", "pb")),
            _column_expr(cols, "turnover_rate", aliases=("turn", "turnover", "turnover_rate")),
        ]
        source_expr = "source" if "source" in cols else "'legacy'"
        sql = f"""
        copy (
            select
                {exprs[0]} as symbol,
                {exprs[1]} as trade_date,
                cast({exprs[2]} as double) as total_mv,
                cast({exprs[3]} as double) as circ_mv,
                cast({exprs[4]} as double) as pe,
                cast({exprs[5]} as double) as pb,
                cast({exprs[6]} as double) as turnover_rate,
                cast({source_expr} as varchar) as source
            from read_parquet({source_literal})
        ) to {target_literal} (format parquet)
        """
        con.execute(sql)
    with duckdb.connect(":memory:") as stats_con:
        row = stats_con.execute("select count(*) as n, min(trade_date) as start_date, max(trade_date) as end_date from read_parquet(?)", [str(target_path)]).fetchone()
    return {
        "row_count": int(row[0] or 0),
        "start_date": str(row[1] or ""),
        "end_date": str(row[2] or ""),
        "schema_hash": schema_hash(_duckdb_schema(target_path)),
    }


def _column_expr(columns: set[str], canonical: str, *, aliases: tuple[str, ...]) -> str:
    for name in (canonical, *aliases):
        if name in columns:
            return name
    return "null"


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _duckdb_schema(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    import duckdb  # type: ignore

    with duckdb.connect(":memory:") as con:
        rows = con.execute("describe select * from read_parquet(?)", [str(path)]).fetchall()
    return [{"name": str(row[0]), "type": str(row[1])} for row in rows]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp rebuild", description="Rebuild qdp_v2 cache and derived tables.")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--runtime", default="balanced", choices=("safe", "balanced", "fast"))
    sub = parser.add_subparsers(dest="clean_command", required=True)
    five = sub.add_parser("5m-from-1m", help="Derive standard mootdx_5m_48_v1 shards from qdp_v2 1m 240 shards.")
    five.add_argument("--source-dataset-id", default="")
    five.add_argument("--max-shards", type=int, default=0)
    five.add_argument("--workers", type=int, default=1)
    five.add_argument("--resume", action="store_true")
    five.add_argument("--trust-existing", action="store_true")
    five.add_argument("--duckdb-memory-limit", default="")
    five.add_argument("--shard-modulo", type=int, default=0)
    five.add_argument("--shard-remainder", type=int, default=0)
    five.add_argument("--stage-only", action="store_true")
    five.add_argument("--activate-domain", action="store_true")
    five.add_argument("--reuse-from-dataset-id", default="")
    five.add_argument("--reuse-copy-mode", default="hardlink", choices=("hardlink", "copy"))
    five.add_argument("--json", action="store_true")
    daily = sub.add_parser("daily-market", help="Rebuild raw daily facts and research panel from an input daily panel.")
    daily.add_argument("--source-dataset-id", default="")
    daily.add_argument("--source-domain", default="market_daily_panel")
    daily.add_argument("--max-shards", type=int, default=0)
    daily.add_argument("--activate-domain", action="store_true")
    daily.add_argument("--json", action="store_true")
    daily_missing = sub.add_parser("daily-from-intraday-missing", help="Fill nonzero intraday symbol-days missing from daily raw and panel.")
    daily_missing.add_argument("--raw-dataset-id", default="")
    daily_missing.add_argument("--panel-dataset-id", default="")
    daily_missing.add_argument("--five-minute-dataset-id", default="")
    daily_missing.add_argument("--duckdb-memory-limit", default="")
    daily_missing.add_argument("--activate-domain", action="store_true")
    daily_missing.add_argument("--json", action="store_true")
    valuation = sub.add_parser("valuation", help="Normalize valuation shards to qdp_v2 valuation schema.")
    valuation.add_argument("--source-dataset-id", default="")
    valuation.add_argument("--max-shards", type=int, default=0)
    valuation.add_argument("--activate-domain", action="store_true")
    valuation.add_argument("--json", action="store_true")
    scope = sub.add_parser("scope-mainboard", help="Filter active symbol datasets to Shanghai/Shenzhen mainboard A-share scope.")
    scope.add_argument("--as-of-date", default="")
    scope.add_argument("--domains", default="")
    scope.add_argument("--max-shards", type=int, default=0)
    scope.add_argument("--workers", type=int, default=1)
    scope.add_argument("--resume", action="store_true")
    scope.add_argument("--trust-existing", action="store_true")
    scope.add_argument("--duckdb-memory-limit", default="")
    scope.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = str(args.workspace_root or "") or None
    if args.clean_command == "5m-from-1m":
        payload = derive_5m_from_1m(
            workspace_root=workspace,
            source_dataset_id=str(args.source_dataset_id or ""),
            max_shards=int(args.max_shards or 0),
            runtime=str(args.runtime or "balanced"),
            workers=int(args.workers or 1),
            resume=bool(args.resume),
            trust_existing=bool(args.trust_existing),
            duckdb_memory_limit=str(args.duckdb_memory_limit or ""),
            shard_modulo=int(args.shard_modulo or 0),
            shard_remainder=int(args.shard_remainder or 0),
            stage_only=bool(args.stage_only),
            activate_domain=bool(args.activate_domain),
            reuse_from_dataset_id=str(args.reuse_from_dataset_id or ""),
            reuse_copy_mode=str(args.reuse_copy_mode or "hardlink"),
        )
    elif args.clean_command == "daily-market":
        payload = split_daily_market(
            workspace_root=workspace,
            source_dataset_id=str(args.source_dataset_id or ""),
            source_domain=str(args.source_domain or "market_daily_panel"),
            max_shards=int(args.max_shards or 0),
            runtime=str(args.runtime or "balanced"),
            activate_domain=bool(args.activate_domain),
        )
    elif args.clean_command == "daily-from-intraday-missing":
        payload = repair_daily_from_intraday_missing(
            workspace_root=workspace,
            raw_dataset_id=str(args.raw_dataset_id or ""),
            panel_dataset_id=str(args.panel_dataset_id or ""),
            five_minute_dataset_id=str(args.five_minute_dataset_id or ""),
            runtime=str(args.runtime or "balanced"),
            duckdb_memory_limit=str(args.duckdb_memory_limit or ""),
            activate_domain=bool(args.activate_domain),
        )
    elif args.clean_command == "valuation":
        payload = normalize_valuation(
            workspace_root=workspace,
            source_dataset_id=str(args.source_dataset_id or ""),
            max_shards=int(args.max_shards or 0),
            runtime=str(args.runtime or "balanced"),
            activate_domain=bool(args.activate_domain),
        )
    elif args.clean_command == "scope-mainboard":
        payload = scope_mainboard_active(
            workspace_root=workspace,
            as_of_date=str(args.as_of_date or ""),
            domains=str(args.domains or ""),
            max_shards=int(args.max_shards or 0),
            runtime=str(args.runtime or "balanced"),
            workers=int(args.workers or 1),
            resume=bool(args.resume),
            trust_existing=bool(args.trust_existing),
            duckdb_memory_limit=str(args.duckdb_memory_limit or ""),
        )
    else:
        raise ValueError(f"unsupported_clean_command:{args.clean_command}")
    if bool(args.json):
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print("\n".join(f"{key}: {value}" for key, value in payload.items()))
    return 0 if payload.get("status") == "ok" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
