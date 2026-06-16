from __future__ import annotations

import gc
import glob
import hashlib
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from daily_research.data_lake.catalog import ResearchDataLake
from daily_research.data_lake.pool_views import load_pool_view
from daily_research.data_lake.policy_input_loader import load_policy_inputs_from_lake
from daily_research.data_lake.sector_board_views import load_sector_board_view
from daily_research.data_platform.contracts import DataDomain
from daily_research.path_policy.forecast_features import (
    DEFAULT_FORECAST_MAX_FEATURE_COLUMNS,
    build_forecast_feature_store,
)
from daily_research.path_policy.forecast_dataset import (
    _cross_section_bucket,
    _static_context_for_universe,
    build_static_context_vocab,
    normalize_static_context_fields,
    static_context_id_columns,
)
from daily_research.path_policy.labels import (
    PATH20_HORIZON,
    build_path20_labels,
    normalize_cumulative_horizons,
)

from quant_data_platform.core.json_io import read_json, utc_now, write_json
from quant_data_platform.core.paths import QdpPaths, qdp_paths
from quant_data_platform.core.registry import load_root_manifest, write_root_manifest_update
from quant_data_platform.features.profiles import DEFAULT_PROFILE


_STORED_SHARD_STATUS = "completed"
_EMPTY_SHARD_STATUSES = {"empty", "no_coverage"}
_TERMINAL_SHARD_STATUSES = {_STORED_SHARD_STATUS, *_EMPTY_SHARD_STATUSES}
_SCHEMA_CANDIDATE_STATUSES = {_STORED_SHARD_STATUS, "schema_mismatch"}
_RESUMABLE_SHARD_STATUSES = {*_TERMINAL_SHARD_STATUSES, "schema_mismatch"}


@dataclass(frozen=True)
class ShardedMemmapConfig:
    profile: str = DEFAULT_PROFILE
    start_year: int = 0
    end_year: int = 0
    symbol_block_size: int = 300
    max_universe_size: int = 0
    max_shards: int = 0
    lookback_days: int = 60
    horizon: int = PATH20_HORIZON
    cumulative_horizons: str = "1,3,5,10,20"
    execution_mode: str = "next_open"
    max_feature_columns: int = DEFAULT_FORECAST_MAX_FEATURE_COLUMNS
    min_lookback_valid_ratio: float = 0.80
    tag: str = ""
    resume: bool = True
    workers: int = 1
    year_input_cache: bool = False
    pool_view_id: str = ""
    sector_board_view_id: str = ""
    include_static_context: bool = False
    static_context_fields: str = "symbol,exchange,industry"

    def normalized(self) -> "ShardedMemmapConfig":
        return ShardedMemmapConfig(
            profile=str(self.profile or DEFAULT_PROFILE).strip(),
            start_year=max(int(self.start_year or 0), 0),
            end_year=max(int(self.end_year or 0), 0),
            symbol_block_size=max(int(self.symbol_block_size or 300), 1),
            max_universe_size=max(int(self.max_universe_size or 0), 0),
            max_shards=max(int(self.max_shards or 0), 0),
            lookback_days=max(int(self.lookback_days or 60), 1),
            horizon=max(int(self.horizon or PATH20_HORIZON), 1),
            cumulative_horizons=str(self.cumulative_horizons or "1,3,5,10,20"),
            execution_mode=str(self.execution_mode or "next_open").strip(),
            max_feature_columns=max(int(self.max_feature_columns or DEFAULT_FORECAST_MAX_FEATURE_COLUMNS), 1),
            min_lookback_valid_ratio=float(self.min_lookback_valid_ratio),
            tag=str(self.tag or "").strip(),
            resume=bool(self.resume),
            workers=max(int(self.workers or 1), 1),
            year_input_cache=bool(self.year_input_cache),
            pool_view_id=str(self.pool_view_id or "").strip(),
            sector_board_view_id=str(self.sector_board_view_id or "").strip(),
            include_static_context=bool(self.include_static_context),
            static_context_fields=",".join(normalize_static_context_fields(self.static_context_fields)),
        )


def write_sharded_memmap_plan(
    paths: QdpPaths | None = None,
    *,
    profile: str = DEFAULT_PROFILE,
    max_universe_size: int = 0,
    workers: int = 1,
    year_input_cache: bool = False,
    pool_view_id: str = "",
    sector_board_view_id: str = "",
    include_static_context: bool = False,
    static_context_fields: str = "symbol,exchange,industry",
    write: bool = True,
) -> dict[str, Any]:
    resolved_static_fields = normalize_static_context_fields(static_context_fields)
    resolved = paths or qdp_paths()
    payload = {
        "status": "plan_only",
        "artifact_type": "sharded_memmap_build_plan",
        "profile": str(profile or DEFAULT_PROFILE),
        "max_universe_size": int(max_universe_size or 0),
        "workers": max(int(workers or 1), 1),
        "shard_policy": "year_symbol_block_feature_label_shards",
        "memory_policy": "never_build_full_universe_single_process_memmap",
        "worker_policy": "single_worker_by_default_use_build_sharded_memmap_workers_for_controlled_parallelism",
        "year_input_cache": bool(year_input_cache),
        "pool_view_id": str(pool_view_id or "").strip(),
        "sector_board_view_id": str(sector_board_view_id or "").strip(),
        "include_static_context": bool(include_static_context),
        "static_context_fields": list(resolved_static_fields),
        "created_at": utc_now(),
    }
    if write:
        path = resolved.memmap_dir / "sharded_memmap_plan.json"
        write_json(path, payload)
        payload["plan_json"] = str(path.as_posix())
    return payload


def build_sharded_memmap(
    paths: QdpPaths | None = None,
    *,
    config: ShardedMemmapConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = paths or qdp_paths()
    cfg = _coerce_config(config)
    root_manifest = load_root_manifest(resolved)
    dataset_id = str(root_manifest.get("canonical_dataset_id", "") or "")
    if not dataset_id:
        raise ValueError("canonical_dataset_id_required")
    lake = ResearchDataLake(resolved.lake_root)
    canonical_meta = lake.describe_dataset(dataset_id)
    canonical_start = pd.Timestamp(root_manifest.get("canonical_start_date", "2010-01-01")).normalize()
    canonical_end = pd.Timestamp(canonical_meta.get("end_date", "") or pd.Timestamp.today()).normalize()
    years = _year_values(
        start_year=cfg.start_year or int(canonical_start.year),
        end_year=cfg.end_year or int(canonical_end.year),
        canonical_start=canonical_start,
        canonical_end=canonical_end,
    )
    universe = _canonical_universe(canonical_meta)
    source_pool_view_meta = _pool_view_meta(lake=lake, pool_view_id=cfg.pool_view_id, dataset_id=dataset_id)
    source_sector_board_meta = _sector_board_view_meta(lake=lake, sector_board_view_id=cfg.sector_board_view_id, dataset_id=dataset_id)
    if source_pool_view_meta:
        universe = _filter_universe_by_pool_view(lake=lake, pool_view_id=cfg.pool_view_id, universe=universe)
    if cfg.max_universe_size > 0:
        universe = universe[: cfg.max_universe_size]
    blocks = list(_symbol_blocks(universe, cfg.symbol_block_size))
    static_schema = _static_context_schema_for_universe(
        lake=lake,
        canonical_meta=canonical_meta,
        universe=universe,
        enabled=bool(cfg.include_static_context),
        static_context_fields=cfg.static_context_fields,
        sector_board_view_id=cfg.sector_board_view_id,
    )
    tag = cfg.tag or f"{cfg.profile}_{years[0]}_{years[-1]}_{utc_now().replace(':', '').replace('-', '')}"
    out_root = resolved.memmap_dir / "sharded" / _safe_name(tag)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "sharded_memmap_manifest.json"
    progress_path = out_root / "sharded_memmap_progress.json"
    cumulative_horizons = normalize_cumulative_horizons(cfg.cumulative_horizons, horizon=cfg.horizon)
    planned_count = int(len(years) * len(blocks))
    completed: list[dict[str, Any]] = []
    skipped = 0
    schema_hash = ""
    schema_columns: list[str] = []

    existing = read_json(manifest_path)
    existing_by_key = {
        str(item.get("shard_key", "")): dict(item)
        for item in list(existing.get("shards", []) or [])
        if str(item.get("status", "")) in _TERMINAL_SHARD_STATUSES
    }
    for row in _discover_existing_shards(out_root):
        existing_by_key.setdefault(str(row.get("shard_key", "")), row)
    schema_columns = _select_initial_schema_columns(existing_by_key.values())
    if not schema_columns and cfg.max_shards <= 0:
        reference = _build_schema_reference_shard(
            lake=lake,
            dataset_id=dataset_id,
            years=years,
            blocks=blocks,
            canonical_start=canonical_start,
            canonical_end=canonical_end,
            cfg=cfg,
            out_root=out_root,
            cumulative_horizons=cumulative_horizons,
            static_schema=static_schema,
        )
        if reference:
            existing_by_key[str(reference.get("shard_key", ""))] = reference
            schema_columns = _select_initial_schema_columns(existing_by_key.values())
    schema_hash = _sequence_hash(schema_columns) if schema_columns else ""
    pending_specs: list[dict[str, Any]] = []
    build_mode = "serial_year_input_cache" if bool(cfg.year_input_cache) and int(cfg.workers) <= 1 else "parallel_threads" if int(cfg.workers) > 1 and schema_columns else "serial"
    if int(cfg.workers) > 1 and not schema_columns:
        build_mode = "serial_schema_unavailable"
    for year in years:
        year_prepared: Any | None = None
        year_load_error = ""
        target_start, target_end, context_start, context_end = _shard_windows(
            year=int(year),
            canonical_start=canonical_start,
            canonical_end=canonical_end,
            cfg=cfg,
        )
        for block_id, symbols in enumerate(blocks):
            if cfg.max_shards > 0 and len(completed) + len(pending_specs) >= cfg.max_shards:
                break
            shard_key = f"year={year}/block={block_id:04d}"
            if cfg.resume and shard_key in existing_by_key and _shard_is_resumable(existing_by_key[shard_key]):
                row = _normalize_resumed_shard(existing_by_key[shard_key], expected_feature_columns=schema_columns)
                completed.append(row)
                if str(row.get("status", "")) == _STORED_SHARD_STATUS and not schema_hash:
                    schema_hash = str(row.get("feature_schema_hash", "") or "")
                    schema_columns = list(row.get("feature_columns", []) or [])
                skipped += 1
                continue
            if build_mode == "parallel_threads":
                pending_specs.append(
                    {
                        "shard_key": shard_key,
                        "year": int(year),
                        "block_id": int(block_id),
                        "symbols": list(symbols),
                    }
                )
                continue
            if build_mode == "serial_year_input_cache":
                shard_dir = out_root / f"year={int(year)}" / f"block={int(block_id):04d}"
                shard_manifest_path = shard_dir / "shard_manifest.json"
                if year_prepared is None and not year_load_error:
                    try:
                        year_prepared = load_policy_inputs_from_lake(
                            lake=lake,
                            dataset_id=dataset_id,
                            start_date=context_start.strftime("%Y-%m-%d"),
                            end_date=context_end.strftime("%Y-%m-%d"),
                            universe=list(universe),
                            pool_view_id=str(cfg.pool_view_id or ""),
                            intersect_pool_view_with_universe=bool(cfg.pool_view_id),
                            sector_board_view_id=str(cfg.sector_board_view_id or ""),
                            min_trading_days=2,
                            require_benchmark_open=str(cfg.execution_mode).strip().lower() == "next_open",
                        )
                    except ValueError as exc:
                        message = str(exc)
                        if "lake_coverage_blocker: no requested symbols are available in lake market data" not in message:
                            raise
                        year_load_error = message
                if year_load_error:
                    shard = _write_terminal_shard(
                        shard_manifest_path=shard_manifest_path,
                        status="no_coverage",
                        empty_reason="no_requested_symbols_available_in_lake_market_data",
                        year=year,
                        block_id=block_id,
                        symbols=list(symbols),
                        target_start=target_start,
                        target_end=target_end,
                        context_start=context_start,
                        context_end=context_end,
                        error_summary=year_load_error,
                    )
                else:
                    block_prepared = _slice_prepared_for_symbols(year_prepared, list(symbols)) if year_prepared is not None else None
                    if block_prepared is None:
                        shard = _write_terminal_shard(
                            shard_manifest_path=shard_manifest_path,
                            status="no_coverage",
                            empty_reason="no_requested_symbols_available_in_cached_year_inputs",
                            year=year,
                            block_id=block_id,
                            symbols=list(symbols),
                            target_start=target_start,
                            target_end=target_end,
                            context_start=context_start,
                            context_end=context_end,
                            error_summary="no requested symbols remain after slicing cached year inputs",
                        )
                    else:
                        shard = _build_one_shard_from_prepared(
                            prepared=block_prepared,
                            year=year,
                            block_id=block_id,
                            target_start=target_start,
                            target_end=target_end,
                            context_start=context_start,
                            context_end=context_end,
                            cfg=cfg,
                            out_root=out_root,
                            cumulative_horizons=cumulative_horizons,
                            expected_feature_columns=schema_columns,
                            static_schema=static_schema,
                        )
                schema_hash, schema_columns = _accept_shard_schema(
                    shard=shard,
                    schema_hash=schema_hash,
                    schema_columns=schema_columns,
                )
                completed.append(shard)
                _write_progress(
                    progress_path,
                    status="running",
                    planned_shards=planned_count,
                    completed_shards=len(completed),
                    latest_shard=shard,
                )
                gc.collect()
                continue
            shard = _build_one_shard(
                lake=lake,
                dataset_id=dataset_id,
                year=year,
                block_id=block_id,
                symbols=list(symbols),
                canonical_start=canonical_start,
                canonical_end=canonical_end,
                cfg=cfg,
                out_root=out_root,
                cumulative_horizons=cumulative_horizons,
                expected_feature_columns=schema_columns,
                static_schema=static_schema,
            )
            schema_hash, schema_columns = _accept_shard_schema(
                shard=shard,
                schema_hash=schema_hash,
                schema_columns=schema_columns,
            )
            completed.append(shard)
            _write_progress(
                progress_path,
                status="running",
                planned_shards=planned_count,
                completed_shards=len(completed),
                latest_shard=shard,
            )
            gc.collect()
        if cfg.max_shards > 0 and len(completed) >= cfg.max_shards:
            break
    if pending_specs:
        parallel_results = _build_shards_parallel(
            lake=lake,
            dataset_id=dataset_id,
            shard_specs=pending_specs,
            canonical_start=canonical_start,
            canonical_end=canonical_end,
            cfg=cfg,
            out_root=out_root,
            cumulative_horizons=cumulative_horizons,
            expected_feature_columns=schema_columns,
            static_schema=static_schema,
            progress_path=progress_path,
            planned_count=planned_count,
            completed_so_far=len(completed),
        )
        for spec in pending_specs:
            shard = parallel_results[str(spec["shard_key"])]
            schema_hash, schema_columns = _accept_shard_schema(
                shard=shard,
                schema_hash=schema_hash,
                schema_columns=schema_columns,
            )
            completed.append(shard)

    stored = [item for item in completed if str(item.get("status", "")) == _STORED_SHARD_STATUS]
    empty = [item for item in completed if str(item.get("status", "")) in _EMPTY_SHARD_STATUSES]
    failed = [item for item in completed if str(item.get("status", "")) not in _TERMINAL_SHARD_STATUSES]
    status = "completed" if len(completed) == planned_count and not failed else "partial"
    scope = _scope_for_build(cfg=cfg, universe_size=len(universe), full_universe_size=_canonical_universe_size(canonical_meta), years=years, canonical_start=canonical_start, canonical_end=canonical_end)
    manifest = {
        "artifact_type": "qdp_sharded_memmap",
        "schema_version": 1,
        "status": status,
        "scope": scope,
        "canonical_dataset_id": dataset_id,
        "canonical_alias": str(root_manifest.get("alias", "canonical_data_v1") or "canonical_data_v1"),
        "source_market_dataset_id": dataset_id,
        "source_pool_view_id": str(source_pool_view_meta.get("dataset_id", "")),
        "source_pool_view_kind": str(source_pool_view_meta.get("view_kind", "")),
        "source_pool_view_name": str(source_pool_view_meta.get("view_name", "")),
        "source_sector_board_view_id": str(source_sector_board_meta.get("dataset_id", "")),
        "source_sector_board_view_kind": str(source_sector_board_meta.get("view_kind", "")),
        "source_sector_board_snapshot_semantics": str(source_sector_board_meta.get("snapshot_semantics", "")),
        "profile": cfg.profile,
        "feature_profile": cfg.profile,
        "feature_schema_hash": schema_hash,
        "feature_columns": schema_columns,
        "feature_count": int(len(schema_columns)),
        "start_year": int(years[0]),
        "end_year": int(years[-1]),
        "years": [int(year) for year in years],
        "symbol_count": int(len(universe)),
        "full_universe_size": int(_canonical_universe_size(canonical_meta)),
        "symbol_block_size": int(cfg.symbol_block_size),
        "planned_shard_count": planned_count,
        "processed_shard_count": int(len(completed)),
        "stored_shard_count": int(len(stored)),
        "empty_shard_count": int(len(empty)),
        "failed_shard_count": int(len(failed)),
        "shards": completed,
        "lookback_days": int(cfg.lookback_days),
        "horizon": int(cfg.horizon),
        "cumulative_horizons": [int(item) for item in cumulative_horizons],
        "label_schema_name": "path20_basic_v2",
        "label_schema_version": 2,
        "execution_mode": cfg.execution_mode,
        "max_feature_columns": int(cfg.max_feature_columns),
        "min_lookback_valid_ratio": float(cfg.min_lookback_valid_ratio),
        "build_mode": build_mode,
        "requested_worker_count": int(cfg.workers),
        "effective_worker_count": int(min(max(int(cfg.workers), 1), max(len(pending_specs), 1))) if build_mode == "parallel_threads" else 1,
        "year_input_cache_requested": bool(cfg.year_input_cache),
        "year_input_cache_effective": bool(build_mode == "serial_year_input_cache"),
        "year_input_cache_policy": "serial_only_full_year_prepared_inputs_sliced_by_symbol_block",
        "static_context_schema": _static_schema_manifest(static_schema, enabled=bool(cfg.include_static_context)),
        "static_context_vocab": _static_vocab_manifest(static_schema) if bool(cfg.include_static_context) else {},
        "skipped_shard_count": int(skipped),
        "created_at": utc_now(),
        "manifest_json": str(manifest_path.resolve()),
    }
    write_json(manifest_path, manifest)
    _write_progress(progress_path, status=status, planned_shards=planned_count, completed_shards=len(completed), latest_shard={})
    _register_sharded_manifest(resolved, manifest)
    return manifest


def validate_sharded_memmap_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = read_json(path)
    blockers: list[str] = []
    if not manifest:
        return {"status": "blocked", "blockers": ["missing_manifest"], "manifest_json": str(path)}
    if str(manifest.get("artifact_type", "")) != "qdp_sharded_memmap":
        blockers.append("not_qdp_sharded_memmap")
    shards = list(manifest.get("shards", []) or [])
    stored = [dict(item) for item in shards if str(dict(item).get("status", "")) == "completed"]
    empty = [dict(item) for item in shards if str(dict(item).get("status", "")) in _EMPTY_SHARD_STATUSES]
    processed = stored + empty
    if not processed:
        blockers.append("no_processed_shards")
    schema = str(manifest.get("feature_schema_hash", "") or "")
    for shard in stored:
        if schema and str(shard.get("feature_schema_hash", "") or "") != schema:
            blockers.append("feature_schema_mismatch")
        if not _shard_files_exist(shard):
            blockers.append("missing_shard_file")
    for shard in empty:
        shard_manifest_path = Path(str(shard.get("shard_manifest_json", "") or ""))
        if not shard_manifest_path.exists():
            blockers.append("missing_empty_shard_manifest")
    return {
        "status": "ok" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "manifest_json": str(path.resolve()),
        "scope": str(manifest.get("scope", "") or ""),
        "canonical_dataset_id": str(manifest.get("canonical_dataset_id", "") or ""),
        "profile": str(manifest.get("profile", "") or ""),
        "planned_shard_count": int(manifest.get("planned_shard_count", 0) or 0),
        "processed_shard_count": int(len(processed)),
        "stored_shard_count": int(len(stored)),
        "empty_shard_count": int(len(empty)),
        "feature_count": int(manifest.get("feature_count", 0) or 0),
        "symbol_count": int(manifest.get("symbol_count", 0) or 0),
        "years": list(manifest.get("years", []) or []),
    }


def _build_one_shard(
    *,
    lake: ResearchDataLake,
    dataset_id: str,
    year: int,
    block_id: int,
    symbols: list[str],
    canonical_start: pd.Timestamp,
    canonical_end: pd.Timestamp,
    cfg: ShardedMemmapConfig,
    out_root: Path,
    cumulative_horizons: tuple[int, ...],
    expected_feature_columns: list[str] | None = None,
    static_schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target_start, target_end, context_start, context_end = _shard_windows(
        year=int(year),
        canonical_start=canonical_start,
        canonical_end=canonical_end,
        cfg=cfg,
    )
    shard_dir = out_root / f"year={int(year)}" / f"block={int(block_id):04d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_manifest_path = shard_dir / "shard_manifest.json"
    try:
        prepared = load_policy_inputs_from_lake(
            lake=lake,
            dataset_id=dataset_id,
            start_date=context_start.strftime("%Y-%m-%d"),
            end_date=context_end.strftime("%Y-%m-%d"),
            universe=symbols,
            pool_view_id=str(cfg.pool_view_id or ""),
            intersect_pool_view_with_universe=bool(cfg.pool_view_id),
            sector_board_view_id=str(cfg.sector_board_view_id or ""),
            min_trading_days=2,
            require_benchmark_open=str(cfg.execution_mode).strip().lower() == "next_open",
        )
    except ValueError as exc:
        message = str(exc)
        if "lake_coverage_blocker: no requested symbols are available in lake market data" not in message:
            raise
        return _write_terminal_shard(
            shard_manifest_path=shard_manifest_path,
            status="no_coverage",
            empty_reason="no_requested_symbols_available_in_lake_market_data",
            year=year,
            block_id=block_id,
            symbols=list(symbols),
            target_start=target_start,
            target_end=target_end,
            context_start=context_start,
            context_end=context_end,
            error_summary=message,
        )
    return _build_one_shard_from_prepared(
        prepared=prepared,
        year=year,
        block_id=block_id,
        target_start=target_start,
        target_end=target_end,
        context_start=context_start,
        context_end=context_end,
        cfg=cfg,
        out_root=out_root,
        cumulative_horizons=cumulative_horizons,
        expected_feature_columns=expected_feature_columns,
        static_schema=static_schema,
    )


def _build_one_shard_from_prepared(
    *,
    prepared: Any,
    year: int,
    block_id: int,
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
    context_start: pd.Timestamp,
    context_end: pd.Timestamp,
    cfg: ShardedMemmapConfig,
    out_root: Path,
    cumulative_horizons: tuple[int, ...],
    expected_feature_columns: list[str] | None = None,
    static_schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    shard_dir = out_root / f"year={int(year)}" / f"block={int(block_id):04d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_manifest_path = shard_dir / "shard_manifest.json"
    target_dates = [
        pd.Timestamp(dt).normalize()
        for dt in prepared.close.index
        if pd.Timestamp(dt).normalize() >= target_start and pd.Timestamp(dt).normalize() <= target_end
    ]
    if not target_dates:
        return _write_terminal_shard(
            shard_manifest_path=shard_manifest_path,
            status="empty",
            empty_reason="no_target_dates",
            year=year,
            block_id=block_id,
            symbols=list(prepared.universe),
            target_start=target_start,
            target_end=target_end,
            context_start=context_start,
            context_end=context_end,
        )
    feature_store_path, feature_columns, feature_manifest, history_ratio = build_forecast_feature_store(
        prepared,
        target_dates,
        root=shard_dir,
        feature_profile=cfg.profile,
        max_feature_columns=int(cfg.max_feature_columns),
        lookback_days=int(cfg.lookback_days),
        min_lookback_valid_ratio=float(cfg.min_lookback_valid_ratio),
    )
    if expected_feature_columns:
        feature_columns, feature_manifest = _project_feature_store_to_schema(
            feature_store_path=feature_store_path,
            feature_columns=list(feature_columns),
            feature_manifest=dict(feature_manifest),
            expected_feature_columns=list(expected_feature_columns),
        )
    labels = build_path20_labels(
        prepared,
        execution_mode=cfg.execution_mode,
        horizon=int(cfg.horizon),
        cumulative_horizons=cumulative_horizons,
    )
    label_manifest = _write_label_store(
        shard_dir=shard_dir,
        labels=labels,
        dates=target_dates,
        symbols=list(prepared.universe),
        horizon=int(cfg.horizon),
        cumulative_horizons=cumulative_horizons,
    )
    sample_index_path, sample_count = _write_sample_index(
        shard_dir=shard_dir,
        prepared=prepared,
        dates=target_dates,
        symbols=list(prepared.universe),
        history_ratio=history_ratio,
        label_manifest=label_manifest,
        min_lookback_valid_ratio=float(cfg.min_lookback_valid_ratio),
        include_static_context=bool(cfg.include_static_context),
        static_schema=static_schema,
    )
    feature_shape = [int(item) for item in list(feature_manifest.get("feature_store_shape", []) or [])]
    feature_schema_hash = _sequence_hash(feature_columns)
    shard = {
        "status": "completed",
        "shard_key": f"year={year}/block={block_id:04d}",
        "year": int(year),
        "block_id": int(block_id),
        "start_date": target_start.strftime("%Y-%m-%d"),
        "end_date": target_end.strftime("%Y-%m-%d"),
        "context_start_date": context_start.strftime("%Y-%m-%d"),
        "context_end_date": context_end.strftime("%Y-%m-%d"),
        "symbol_count": int(len(prepared.universe)),
        "date_count": int(len(target_dates)),
        "sample_count": int(sample_count),
        "feature_store_path": str(feature_store_path.resolve()),
        "feature_store_shape": feature_shape,
        "feature_columns": list(feature_columns),
        "feature_schema_hash": feature_schema_hash,
        "feature_manifest": dict(feature_manifest),
        "label_manifest_json": str((shard_dir / "labels_manifest.json").resolve()),
        "sample_index_path": str(sample_index_path.resolve()),
        "static_context_schema": _static_schema_manifest(static_schema, enabled=bool(cfg.include_static_context)),
        "shard_manifest_json": str(shard_manifest_path.resolve()),
        "created_at": utc_now(),
    }
    write_json(shard_manifest_path, shard)
    return shard


def _shard_windows(
    *,
    year: int,
    canonical_start: pd.Timestamp,
    canonical_end: pd.Timestamp,
    cfg: ShardedMemmapConfig,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    target_start = max(pd.Timestamp(year=int(year), month=1, day=1), canonical_start)
    target_end = min(pd.Timestamp(year=int(year), month=12, day=31), canonical_end)
    context_start = max(target_start - pd.Timedelta(days=int(cfg.lookback_days) * 3 + 45), canonical_start)
    context_end = min(target_end + pd.Timedelta(days=int(cfg.horizon) * 4 + 21), canonical_end)
    return target_start, target_end, context_start, context_end


def _write_terminal_shard(
    *,
    shard_manifest_path: Path,
    status: str,
    empty_reason: str,
    year: int,
    block_id: int,
    symbols: list[str],
    target_start: pd.Timestamp,
    target_end: pd.Timestamp,
    context_start: pd.Timestamp,
    context_end: pd.Timestamp,
    error_summary: str = "",
) -> dict[str, Any]:
    shard_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    shard: dict[str, Any] = {
        "status": str(status),
        "empty_reason": str(empty_reason),
        "shard_key": f"year={int(year)}/block={int(block_id):04d}",
        "year": int(year),
        "block_id": int(block_id),
        "start_date": target_start.strftime("%Y-%m-%d"),
        "end_date": target_end.strftime("%Y-%m-%d"),
        "context_start_date": context_start.strftime("%Y-%m-%d"),
        "context_end_date": context_end.strftime("%Y-%m-%d"),
        "symbol_count": int(len(symbols)),
        "date_count": 0,
        "sample_count": 0,
        "shard_manifest_json": str(shard_manifest_path.resolve()),
        "created_at": utc_now(),
    }
    if str(error_summary or "").strip():
        shard["error_summary"] = str(error_summary)[:500]
    write_json(shard_manifest_path, shard)
    return shard


def _slice_prepared_for_symbols(prepared: Any, symbols: list[str]) -> Any | None:
    requested = _normalize_symbols(symbols)
    available = set(_normalize_symbols(list(getattr(prepared, "universe", []) or [])))
    selected = [symbol for symbol in requested if symbol in available]
    if not selected:
        return None

    def slice_frame(frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return frame.copy() if isinstance(frame, pd.DataFrame) else frame
        column_map = {str(column).strip().upper(): column for column in frame.columns}
        selected_columns = [column_map[symbol] for symbol in selected if symbol in column_map]
        if selected_columns:
            out = frame.reindex(columns=selected_columns).copy()
            out.columns = [str(column).strip().upper() for column in out.columns]
            return out
        for symbol_column in ("symbol", "stock"):
            if symbol_column in frame.columns:
                out = frame.loc[frame[symbol_column].astype(str).str.strip().str.upper().isin(set(selected))].copy()
                return out
        return frame.copy()

    def slice_frame_dict(frames: Mapping[str, Any]) -> dict[str, Any]:
        return {
            str(name): slice_frame(frame) if isinstance(frame, pd.DataFrame) else frame
            for name, frame in dict(frames or {}).items()
        }

    return replace(
        prepared,
        universe=tuple(selected),
        close=slice_frame(prepared.close),
        open_=slice_frame(prepared.open_),
        high=slice_frame(prepared.high),
        low=slice_frame(prepared.low),
        volume=slice_frame(prepared.volume),
        amount=slice_frame(prepared.amount),
        score_none=slice_frame(prepared.score_none),
        score_v2=slice_frame(prepared.score_v2),
        score_blend=slice_frame(prepared.score_blend),
        feature_frames=slice_frame_dict(prepared.feature_frames),
        membership_frame=slice_frame(prepared.membership_frame),
        derived_frames=slice_frame_dict(prepared.derived_frames),
        metadata_frames=slice_frame_dict(getattr(prepared, "metadata_frames", {})),
    )


def _normalize_symbols(symbols: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in list(symbols or []):
        symbol = str(raw or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def _build_shards_parallel(
    *,
    lake: ResearchDataLake,
    dataset_id: str,
    shard_specs: list[dict[str, Any]],
    canonical_start: pd.Timestamp,
    canonical_end: pd.Timestamp,
    cfg: ShardedMemmapConfig,
    out_root: Path,
    cumulative_horizons: tuple[int, ...],
    expected_feature_columns: list[str],
    static_schema: Mapping[str, Any] | None,
    progress_path: Path,
    planned_count: int,
    completed_so_far: int,
) -> dict[str, dict[str, Any]]:
    if not expected_feature_columns:
        raise ValueError("parallel_sharded_memmap_requires_locked_feature_schema")
    worker_count = min(max(int(cfg.workers), 1), max(len(shard_specs), 1))
    results: dict[str, dict[str, Any]] = {}
    lake_root = Path(lake.root)
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="qdp-shard") as executor:
        futures = {
            executor.submit(
                _build_one_shard_worker,
                lake_root=str(lake_root),
                dataset_id=str(dataset_id),
                year=int(spec["year"]),
                block_id=int(spec["block_id"]),
                symbols=list(spec["symbols"]),
                canonical_start=str(pd.Timestamp(canonical_start).strftime("%Y-%m-%d")),
                canonical_end=str(pd.Timestamp(canonical_end).strftime("%Y-%m-%d")),
                cfg_payload=asdict(cfg),
                out_root=str(out_root),
                cumulative_horizons=tuple(int(item) for item in cumulative_horizons),
                expected_feature_columns=list(expected_feature_columns),
                static_schema=dict(static_schema or {}),
            ): str(spec["shard_key"])
            for spec in shard_specs
        }
        for future in as_completed(futures):
            shard_key = futures[future]
            try:
                shard = future.result()
            except Exception as exc:  # pragma: no cover - exercised by integration failures.
                shard = _failed_shard_record(
                    shard_key=shard_key,
                    spec=next(item for item in shard_specs if str(item["shard_key"]) == shard_key),
                    out_root=out_root,
                    error=exc,
                )
            results[shard_key] = shard
            _write_progress(
                progress_path,
                status="running",
                planned_shards=planned_count,
                completed_shards=int(completed_so_far + len(results)),
                latest_shard=shard,
                worker_count=int(worker_count),
            )
    return results


def _build_one_shard_worker(
    *,
    lake_root: str,
    dataset_id: str,
    year: int,
    block_id: int,
    symbols: list[str],
    canonical_start: str,
    canonical_end: str,
    cfg_payload: Mapping[str, Any],
    out_root: str,
    cumulative_horizons: tuple[int, ...],
    expected_feature_columns: list[str],
    static_schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    lake = ResearchDataLake(Path(lake_root))
    cfg = _coerce_config(cfg_payload)
    return _build_one_shard(
        lake=lake,
        dataset_id=str(dataset_id),
        year=int(year),
        block_id=int(block_id),
        symbols=list(symbols),
        canonical_start=pd.Timestamp(canonical_start).normalize(),
        canonical_end=pd.Timestamp(canonical_end).normalize(),
        cfg=cfg,
        out_root=Path(out_root),
        cumulative_horizons=tuple(int(item) for item in cumulative_horizons),
        expected_feature_columns=list(expected_feature_columns),
        static_schema=static_schema,
    )


def _failed_shard_record(
    *,
    shard_key: str,
    spec: Mapping[str, Any],
    out_root: Path,
    error: Exception,
) -> dict[str, Any]:
    year = int(spec.get("year", 0) or 0)
    block_id = int(spec.get("block_id", 0) or 0)
    shard_dir = out_root / f"year={year}" / f"block={block_id:04d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_manifest_path = shard_dir / "shard_manifest.json"
    shard = {
        "status": "failed",
        "shard_key": str(shard_key),
        "year": year,
        "block_id": block_id,
        "symbol_count": int(len(list(spec.get("symbols", []) or []))),
        "sample_count": 0,
        "shard_manifest_json": str(shard_manifest_path.resolve()),
        "error_type": type(error).__name__,
        "error_summary": str(error)[:1000],
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__))[-4000:],
        "created_at": utc_now(),
    }
    write_json(shard_manifest_path, shard)
    return shard


def _accept_shard_schema(
    *,
    shard: Mapping[str, Any],
    schema_hash: str,
    schema_columns: list[str],
) -> tuple[str, list[str]]:
    if str(shard.get("status", "")) != "completed":
        return schema_hash, schema_columns
    current_hash = str(shard.get("feature_schema_hash", "") or "")
    if schema_hash and current_hash and current_hash != schema_hash:
        row = {**dict(shard), "status": "schema_mismatch", "expected_feature_schema_hash": schema_hash}
        write_json(Path(str(row["shard_manifest_json"])), row)
        raise ValueError(
            "sharded_memmap_feature_schema_mismatch: "
            f"expected={schema_hash} actual={current_hash} shard={row.get('shard_key', '')}"
        )
    return schema_hash or current_hash, schema_columns or list(shard.get("feature_columns", []) or [])


def _write_label_store(
    *,
    shard_dir: Path,
    labels: Any,
    dates: list[pd.Timestamp],
    symbols: list[str],
    horizon: int,
    cumulative_horizons: tuple[int, ...],
) -> dict[str, Any]:
    label_dir = shard_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "daily_return": _stack_panel_map(labels.daily_return, range(1, int(horizon) + 1), dates=dates, symbols=symbols),
        "benchmark_daily_return": _stack_series_map(labels.benchmark_daily_return, range(1, int(horizon) + 1), dates=dates, symbols=symbols),
        "daily_excess_return": _stack_panel_map(labels.daily_excess_return, range(1, int(horizon) + 1), dates=dates, symbols=symbols),
        "cumulative_return": _stack_panel_map(labels.cumulative_return, cumulative_horizons, dates=dates, symbols=symbols),
        "benchmark_cumulative_return": _stack_series_map(labels.benchmark_cumulative_return, cumulative_horizons, dates=dates, symbols=symbols),
        "cumulative_excess_return": _stack_panel_map(labels.cumulative_excess_return, cumulative_horizons, dates=dates, symbols=symbols),
        "cumulative_return_1to20": _stack_panel_map(labels.cumulative_return_1to20, range(1, int(horizon) + 1), dates=dates, symbols=symbols),
        "benchmark_cumulative_return_1to20": _stack_series_map(labels.benchmark_cumulative_return_1to20, range(1, int(horizon) + 1), dates=dates, symbols=symbols),
        "cumulative_excess_return_1to20": _stack_panel_map(labels.cumulative_excess_return_1to20, range(1, int(horizon) + 1), dates=dates, symbols=symbols),
        "rank_1to20": _stack_panel_map(labels.forward_rank_1to20, range(1, int(horizon) + 1), dates=dates, symbols=symbols),
        "rank_by_horizon": _stack_panel_map(labels.forward_rank, cumulative_horizons, dates=dates, symbols=symbols),
        "industry_rank_by_horizon": _stack_panel_map(labels.industry_rank_by_horizon, cumulative_horizons, dates=dates, symbols=symbols),
        "drawdown_by_horizon": _stack_panel_map(labels.path_max_drawdown_by_horizon, cumulative_horizons, dates=dates, symbols=symbols),
        "worst_by_horizon": _stack_panel_map(labels.path_worst_1d_by_horizon, cumulative_horizons, dates=dates, symbols=symbols),
        "upside_by_horizon": _stack_panel_map(labels.path_upside_capture_by_horizon, cumulative_horizons, dates=dates, symbols=symbols),
        "rank_20d": _panel_to_array(labels.forward_rank.get(int(horizon), labels.forward_rank[int(cumulative_horizons[-1])]), dates=dates, symbols=symbols),
        "max_drawdown_20d": _panel_to_array(labels.path_max_drawdown_20d, dates=dates, symbols=symbols),
        "worst_1d_20d": _panel_to_array(labels.path_worst_1d_20d, dates=dates, symbols=symbols),
        "upside_20d": _panel_to_array(labels.path_upside_capture_20d, dates=dates, symbols=symbols),
        "entry_tradeable": _panel_to_array(labels.entry_tradeable, dates=dates, symbols=symbols),
        "entry_limit_up_buy_blocked": _panel_to_array(labels.entry_limit_up_buy_blocked, dates=dates, symbols=symbols),
        "entry_suspended_or_no_open": _panel_to_array(labels.entry_suspended_or_no_open, dates=dates, symbols=symbols),
        "forward_tradeable_ratio_by_horizon": _stack_panel_map(labels.forward_tradeable_ratio_by_horizon, cumulative_horizons, dates=dates, symbols=symbols),
    }
    entries: dict[str, Any] = {}
    for name, array in arrays.items():
        path = label_dir / f"{name}.dat"
        _write_array_memmap(path, array.astype("float32", copy=False))
        entries[name] = {
            "path": str(path.resolve()),
            "shape": [int(item) for item in array.shape],
            "dtype": "float32",
        }
    manifest = {
        "status": "completed",
        "label_store_kind": "path20_sharded_daily_symbol_panels",
        "label_schema_name": "path20_basic_v2",
        "label_schema_version": 2,
        "date_values": [pd.Timestamp(dt).strftime("%Y-%m-%d") for dt in dates],
        "stock_values": list(symbols),
        "horizon": int(horizon),
        "cumulative_horizons": [int(item) for item in cumulative_horizons],
        "arrays": entries,
        "label_metadata": dict(labels.metadata),
    }
    write_json(label_dir.parent / "labels_manifest.json", manifest)
    return manifest


def _write_sample_index(
    *,
    shard_dir: Path,
    prepared: Any,
    dates: list[pd.Timestamp],
    symbols: list[str],
    history_ratio: pd.DataFrame,
    label_manifest: Mapping[str, Any],
    min_lookback_valid_ratio: float,
    include_static_context: bool = False,
    static_schema: Mapping[str, Any] | None = None,
) -> tuple[Path, int]:
    membership = prepared.membership_frame.reindex(index=dates, columns=symbols, fill_value=False).astype(bool).to_numpy(dtype=bool)
    history = history_ratio.reindex(index=dates, columns=symbols).to_numpy(dtype=float)
    valid = membership & np.isfinite(history) & (history >= float(min_lookback_valid_ratio))
    arrays = dict(label_manifest.get("arrays", {}) or {})
    for name in ("daily_excess_return", "cumulative_excess_return", "rank_by_horizon", "max_drawdown_20d", "worst_1d_20d", "upside_20d"):
        meta = dict(arrays.get(name, {}) or {})
        path = Path(str(meta.get("path", "") or ""))
        shape = tuple(int(item) for item in list(meta.get("shape", []) or []))
        if not path.exists() or not shape:
            valid &= False
            continue
        arr = np.memmap(path, dtype="float32", mode="r", shape=shape)
        finite = np.isfinite(np.asarray(arr)).all(axis=2) if len(shape) == 3 else np.isfinite(np.asarray(arr))
        valid &= finite
        del arr
    date_pos, stock_pos = np.nonzero(valid)
    if len(date_pos):
        date_values = np.asarray([pd.Timestamp(dt).strftime("%Y-%m-%d") for dt in dates], dtype=object)
        symbol_values = np.asarray([str(symbol) for symbol in symbols], dtype=object)
        sample_index = pd.DataFrame(
            {
                "date": date_values[date_pos],
                "stock": symbol_values[stock_pos],
                "date_pos": date_pos.astype(np.int32, copy=False),
                "stock_pos": stock_pos.astype(np.int32, copy=False),
                "history_valid_ratio": history[date_pos, stock_pos].astype(float, copy=False),
            }
        )
    else:
        sample_index = pd.DataFrame(
            {
                "date": pd.Series(dtype="object"),
                "stock": pd.Series(dtype="object"),
                "date_pos": pd.Series(dtype="int32"),
                "stock_pos": pd.Series(dtype="int32"),
                "history_valid_ratio": pd.Series(dtype="float64"),
            }
        )
    if bool(include_static_context):
        sample_index = _attach_static_context_to_sample_index(
            sample_index=sample_index,
            prepared=prepared,
            dates=dates,
            symbols=symbols,
            static_schema=static_schema,
        )
    path = shard_dir / "sample_index.parquet"
    sample_index.to_parquet(path, index=False)
    return path, int(len(sample_index))


def _attach_static_context_to_sample_index(
    *,
    sample_index: pd.DataFrame,
    prepared: Any,
    dates: list[pd.Timestamp],
    symbols: list[str],
    static_schema: Mapping[str, Any] | None,
) -> pd.DataFrame:
    schema = _static_schema_manifest(static_schema, enabled=True)
    fields = normalize_static_context_fields(schema.get("fields") or None)
    id_columns = list(static_context_id_columns(fields))
    out = sample_index.copy()
    for column in id_columns:
        if column not in out.columns:
            out[column] = pd.Series(np.zeros(len(out), dtype=np.int64), index=out.index)
    if out.empty:
        return out
    vocab = dict(static_schema or {})
    static_by_stock = _static_context_for_universe(prepared, universe=[str(symbol).strip().upper() for symbol in symbols], vocab=vocab)
    if "stock" in out.columns:
        stock_values = out["stock"].astype(str).str.strip().str.upper()
        for column in id_columns:
            if column in {"liquidity_bucket_id", "price_bucket_id"}:
                continue
            values = static_by_stock.reindex(stock_values)[column] if column in static_by_stock.columns else pd.Series(0, index=stock_values.index)
            out[column] = np.asarray(values.fillna(0).astype("int64"), dtype=np.int64)
    if "liquidity_bucket_id" in id_columns:
        buckets = _cross_section_bucket(prepared.amount.reindex(index=dates, columns=symbols), window=20, buckets=5)
        out["liquidity_bucket_id"] = [
            int(buckets.get((pd.Timestamp(row["date"]), str(row["stock"]).strip().upper()), 0))
            for _, row in out.iterrows()
        ]
    if "price_bucket_id" in id_columns:
        buckets = _cross_section_bucket(prepared.close.reindex(index=dates, columns=symbols), window=20, buckets=5)
        out["price_bucket_id"] = [
            int(buckets.get((pd.Timestamp(row["date"]), str(row["stock"]).strip().upper()), 0))
            for _, row in out.iterrows()
        ]
    for column in id_columns:
        out[column] = out[column].fillna(0).astype("int64")
    return out


def _stack_panel_map(
    panels: Mapping[int, pd.DataFrame],
    keys: Iterable[int],
    *,
    dates: list[pd.Timestamp],
    symbols: list[str],
) -> np.ndarray:
    arrays = [_panel_to_array(panels[int(key)], dates=dates, symbols=symbols) for key in keys]
    return np.stack(arrays, axis=2).astype("float32") if arrays else np.empty((len(dates), len(symbols), 0), dtype="float32")


def _stack_series_map(
    series_by_key: Mapping[int, pd.Series],
    keys: Iterable[int],
    *,
    dates: list[pd.Timestamp],
    symbols: list[str],
) -> np.ndarray:
    arrays = [_series_to_symbol_array(series_by_key[int(key)], dates=dates, symbols=symbols) for key in keys]
    return np.stack(arrays, axis=2).astype("float32") if arrays else np.empty((len(dates), len(symbols), 0), dtype="float32")


def _series_to_symbol_array(series: pd.Series, *, dates: list[pd.Timestamp], symbols: list[str]) -> np.ndarray:
    values = series.reindex(index=dates).to_numpy(dtype=np.float32).reshape((len(dates), 1))
    if not symbols:
        return np.empty((len(dates), 0), dtype=np.float32)
    return np.broadcast_to(values, (len(dates), len(symbols))).astype("float32", copy=True)


def _panel_to_array(panel: pd.DataFrame, *, dates: list[pd.Timestamp], symbols: list[str]) -> np.ndarray:
    return panel.reindex(index=dates, columns=symbols).to_numpy(dtype=np.float32)


def _write_array_memmap(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    store = np.memmap(path, dtype=str(array.dtype), mode="w+", shape=array.shape)
    if array.size:
        store[...] = array
    store.flush()
    del store


def _project_feature_store_to_schema(
    *,
    feature_store_path: Path,
    feature_columns: list[str],
    feature_manifest: dict[str, Any],
    expected_feature_columns: list[str],
) -> tuple[list[str], dict[str, Any]]:
    expected = list(expected_feature_columns)
    current = list(feature_columns)
    if current == expected:
        return current, feature_manifest
    shape = tuple(int(item) for item in list(feature_manifest.get("feature_store_shape", []) or []))
    if len(shape) != 3:
        raise ValueError(f"invalid_feature_store_shape_for_projection: {shape}")
    rows, cols, current_count = shape
    if int(current_count) != len(current):
        raise ValueError(
            "feature_store_column_count_mismatch: "
            f"shape_count={current_count} feature_columns={len(current)} path={feature_store_path}"
        )
    current_store = np.memmap(feature_store_path, dtype="float32", mode="r", shape=shape)
    projected_path = feature_store_path.with_name(f"{feature_store_path.stem}.schema_projected{feature_store_path.suffix}")
    projected_shape = (int(rows), int(cols), int(len(expected)))
    projected = np.memmap(projected_path, dtype="float32", mode="w+", shape=projected_shape)
    projected[...] = np.nan
    current_pos = {column: idx for idx, column in enumerate(current)}
    missing: list[str] = []
    for target_idx, column in enumerate(expected):
        source_idx = current_pos.get(column)
        if source_idx is None:
            missing.append(column)
            continue
        projected[:, :, target_idx] = current_store[:, :, source_idx]
    projected.flush()
    nan_count = int(np.isnan(np.asarray(projected)).sum())
    value_count = int(np.asarray(projected).size)
    del projected
    del current_store
    projected_path.replace(feature_store_path)
    dropped = [column for column in current if column not in set(expected)]
    feature_manifest.update(
        {
            "feature_columns": list(expected),
            "feature_count_after_cap": int(len(expected)),
            "feature_store_shape": [int(item) for item in projected_shape],
            "feature_nan_ratio": float(nan_count / value_count) if value_count else 0.0,
            "schema_projection": {
                "status": "projected_to_canonical_shard_schema",
                "source_feature_count": int(len(current)),
                "target_feature_count": int(len(expected)),
                "missing_feature_count": int(len(missing)),
                "dropped_feature_count": int(len(dropped)),
                "missing_features": missing[:50],
                "dropped_features": dropped[:50],
                "created_at": utc_now(),
            },
        }
    )
    return expected, feature_manifest


def _canonical_universe(metadata: Mapping[str, Any]) -> list[str]:
    membership_path = Path(str(dict(metadata.get("content_paths", {}) or {}).get("silver_membership", "") or ""))
    if not membership_path.exists():
        raise FileNotFoundError(f"canonical_membership_not_found: {membership_path}")
    frame = pd.read_parquet(membership_path)
    return [str(column).strip().upper() for column in frame.columns if str(column).strip().lower() not in {"date", "trade_date"}]


def _canonical_universe_size(metadata: Mapping[str, Any]) -> int:
    try:
        return int(len(_canonical_universe(metadata)))
    except Exception:
        return 0


def _pool_view_meta(*, lake: ResearchDataLake, pool_view_id: str, dataset_id: str) -> dict[str, Any]:
    if not str(pool_view_id or "").strip():
        return {}
    view = load_pool_view(lake=lake, pool_view_id=str(pool_view_id).strip())
    parameters = dict(view.metadata.get("parameters", {}) or {})
    source_market_dataset_id = str(parameters.get("source_market_dataset_id", "") or "")
    if source_market_dataset_id and source_market_dataset_id != str(dataset_id):
        raise ValueError(
            "pool_view_source_mismatch: "
            f"pool_view_id={view.dataset_id} source_market_dataset_id={source_market_dataset_id} "
            f"requested_dataset_id={dataset_id}"
        )
    return {
        "dataset_id": view.dataset_id,
        "view_kind": str(parameters.get("view_kind", "") or ""),
        "view_name": str(parameters.get("view_name", "") or ""),
        "source_market_dataset_id": source_market_dataset_id,
    }


def _sector_board_view_meta(*, lake: ResearchDataLake, sector_board_view_id: str, dataset_id: str) -> dict[str, Any]:
    if not str(sector_board_view_id or "").strip():
        return {}
    view = load_sector_board_view(lake=lake, sector_board_view_id=str(sector_board_view_id).strip())
    parameters = dict(view.metadata.get("parameters", {}) or {})
    source_market_dataset_id = str(parameters.get("source_market_dataset_id", "") or "")
    if source_market_dataset_id and source_market_dataset_id != str(dataset_id):
        raise ValueError(
            "sector_board_view_source_mismatch: "
            f"sector_board_view_id={view.dataset_id} source_market_dataset_id={source_market_dataset_id} "
            f"requested_dataset_id={dataset_id}"
        )
    source_cache = dict(view.metadata.get("source_cache", {}) or {})
    return {
        "dataset_id": view.dataset_id,
        "view_kind": str(parameters.get("view_kind", "") or ""),
        "view_name": str(parameters.get("view_name", "") or ""),
        "source_market_dataset_id": source_market_dataset_id,
        "snapshot_semantics": str(parameters.get("snapshot_semantics", "") or source_cache.get("snapshot_semantics", "")),
    }


def _filter_universe_by_pool_view(*, lake: ResearchDataLake, pool_view_id: str, universe: list[str]) -> list[str]:
    view = load_pool_view(lake=lake, pool_view_id=str(pool_view_id).strip())
    membership = view.membership_frame.copy()
    membership.columns = [str(column).strip().upper() for column in membership.columns]
    active = {str(column).strip().upper() for column in membership.columns[membership.any(axis=0)]}
    return [symbol for symbol in _normalize_symbols(list(universe)) if symbol in active]


def _sidecar_dataset_ids(metadata: Mapping[str, Any]) -> dict[str, str]:
    parameters = dict(metadata.get("parameters", {}) or {})
    sidecars = dict(parameters.get("sidecar_dataset_ids", {}) or {})
    return {str(key): str(value) for key, value in sidecars.items() if str(value or "").strip()}


def _static_context_schema_for_universe(
    *,
    lake: ResearchDataLake,
    canonical_meta: Mapping[str, Any],
    universe: list[str],
    enabled: bool,
    static_context_fields: str,
    sector_board_view_id: str = "",
) -> dict[str, Any]:
    fields = normalize_static_context_fields(static_context_fields)
    if not bool(enabled):
        return _static_schema_disabled(fields)
    universe = _normalize_symbols(list(universe))
    metadata_frames = _static_metadata_frames(
        lake=lake,
        canonical_meta=canonical_meta,
        universe=universe,
        sector_board_view_id=sector_board_view_id,
    )
    prepared = _StaticPrepared(universe=tuple(universe), metadata_frames=metadata_frames)
    schema = build_static_context_vocab(prepared, static_context_fields=fields)
    schema["embedding_defaults"] = {
        "symbol": 16,
        "exchange": 4,
        "industry": 8,
        "board": 4,
        "liquidity_bucket": 4,
        "price_bucket": 4,
        "dropout": 0.20,
    }
    return schema


@dataclass(frozen=True)
class _StaticPrepared:
    universe: tuple[str, ...]
    metadata_frames: dict[str, pd.DataFrame]


def _static_metadata_frames(
    *,
    lake: ResearchDataLake,
    canonical_meta: Mapping[str, Any],
    universe: list[str],
    sector_board_view_id: str = "",
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    universe_set = set(_normalize_symbols(list(universe)))
    if str(sector_board_view_id or "").strip():
        view = load_sector_board_view(lake=lake, sector_board_view_id=str(sector_board_view_id).strip())
        industry_map = view.industry_map_frame.copy()
        if "symbol" in industry_map.columns:
            industry_map["symbol"] = industry_map["symbol"].astype(str).str.strip().str.upper()
            frames["industry_map"] = industry_map.loc[industry_map["symbol"].isin(universe_set)].reset_index(drop=True)
        board_membership = view.board_membership_frame.copy()
        if "symbol" in board_membership.columns:
            board_membership["symbol"] = board_membership["symbol"].astype(str).str.strip().str.upper()
            frames["board_membership"] = board_membership.loc[board_membership["symbol"].isin(universe_set)].reset_index(drop=True)
        return frames

    industry_id = _sidecar_dataset_ids(canonical_meta).get(DataDomain.INDUSTRY_CONCEPT, "")
    if not industry_id:
        return frames
    meta = lake.describe_dataset(industry_id)
    paths = dict(meta.get("content_paths", {}) or {})
    candidates = _latest_sidecar_shard_paths(paths)
    raw_path = str(paths.get("silver_domain_data", "") or "")
    if not candidates:
        candidates = sorted(glob.glob(raw_path)) if "*" in raw_path else ([raw_path] if raw_path else [])
    existing = [path for path in candidates if Path(path).exists()]
    if not existing:
        return frames
    columns = ["symbol", "trade_date", "industry"]
    try:
        industry = pd.concat([pd.read_parquet(path, columns=columns) for path in existing], ignore_index=True)
    except Exception:
        industry = pd.concat([pd.read_parquet(path) for path in existing], ignore_index=True)
        industry = industry.reindex(columns=[column for column in columns if column in industry.columns])
    if industry.empty or not {"symbol", "industry"}.issubset(industry.columns):
        return frames
    industry["symbol"] = industry["symbol"].astype(str).str.strip().str.upper()
    industry["industry"] = industry["industry"].fillna("").astype(str).str.strip()
    industry = industry.loc[industry["symbol"].isin(universe_set) & industry["industry"].ne("")].copy()
    if industry.empty:
        return frames
    if "trade_date" in industry.columns:
        industry["trade_date"] = pd.to_datetime(industry["trade_date"], errors="coerce")
        industry = industry.sort_values(["symbol", "trade_date"])
        latest = industry.drop_duplicates(subset=["symbol"], keep="last").copy()
        latest["as_of_date"] = latest["trade_date"].dt.strftime("%Y-%m-%d")
        latest["trade_date"] = latest["as_of_date"]
    else:
        latest = industry.drop_duplicates(subset=["symbol"], keep="last").copy()
    frames["industry_map"] = latest.reset_index(drop=True)
    return frames


def _latest_sidecar_shard_paths(paths: Mapping[str, Any]) -> list[str]:
    manifest_path = Path(str(dict(paths).get("shard_manifest", "") or ""))
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [
        dict(row)
        for row in list(manifest.get("shards", []) or [])
        if str(dict(row).get("status", "") or "") in {"stored", "skipped"}
        and int(dict(row).get("row_count", 0) or 0) > 0
        and str(dict(row).get("path", "") or "").strip()
    ]
    if not rows:
        return []
    rows = sorted(rows, key=lambda row: (str(row.get("end_date", "") or ""), str(row.get("start_date", "") or "")))
    return [str(rows[-1].get("path", "") or "")]


def _static_schema_disabled(fields: tuple[str, ...] | list[str] | str | None = None) -> dict[str, Any]:
    resolved = normalize_static_context_fields(fields)
    return {
        "enabled": False,
        "fields": list(resolved),
        "id_columns": list(static_context_id_columns(resolved)),
        "vocab_sizes": {},
    }


def _static_schema_manifest(schema: Mapping[str, Any] | None, *, enabled: bool) -> dict[str, Any]:
    raw = dict(schema or {})
    fields = normalize_static_context_fields(raw.get("fields") or None)
    return {
        "enabled": bool(enabled and raw.get("enabled", True)),
        "fields": list(fields),
        "id_columns": list(static_context_id_columns(fields)),
        "vocab_sizes": dict(raw.get("vocab_sizes", {}) or {}),
        "embedding_defaults": dict(raw.get("embedding_defaults", {}) or {}),
        "symbol_vocab_fingerprint": str(raw.get("symbol_vocab_fingerprint", "") or ""),
        "industry_vocab_fingerprint": str(raw.get("industry_vocab_fingerprint", "") or ""),
        "board_vocab_fingerprint": str(raw.get("board_vocab_fingerprint", "") or ""),
        "exchange_vocab_fingerprint": str(raw.get("exchange_vocab_fingerprint", "") or ""),
        "liquidity_bucket_vocab_fingerprint": str(raw.get("liquidity_bucket_vocab_fingerprint", "") or ""),
        "price_bucket_vocab_fingerprint": str(raw.get("price_bucket_vocab_fingerprint", "") or ""),
    }


def _static_vocab_manifest(schema: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(schema or {})
    return {
        key: value
        for key, value in raw.items()
        if key.endswith("_vocab") or key == "vocab_sizes"
    }


def _year_values(*, start_year: int, end_year: int, canonical_start: pd.Timestamp, canonical_end: pd.Timestamp) -> list[int]:
    start = max(int(start_year), int(canonical_start.year))
    end = min(int(end_year), int(canonical_end.year))
    if end < start:
        raise ValueError(f"invalid_year_range: {start_year}->{end_year}")
    return list(range(start, end + 1))


def _symbol_blocks(symbols: list[str], block_size: int) -> Iterable[list[str]]:
    size = max(int(block_size), 1)
    for start in range(0, len(symbols), size):
        yield list(symbols[start : start + size])


def _scope_for_build(
    *,
    cfg: ShardedMemmapConfig,
    universe_size: int,
    full_universe_size: int,
    years: list[int],
    canonical_start: pd.Timestamp,
    canonical_end: pd.Timestamp,
) -> str:
    full_years = list(range(int(canonical_start.year), int(canonical_end.year) + 1))
    if cfg.max_shards > 0:
        return "validation_partial"
    if cfg.max_universe_size > 0 and universe_size < full_universe_size:
        return "validation_capped_universe"
    if years != full_years:
        return "validation_year_subset"
    return "full_canonical_candidate"


def _register_sharded_manifest(paths: QdpPaths, manifest: Mapping[str, Any]) -> None:
    registry_path = paths.registry_dir / "sharded_memmap_registry.json"
    registry = read_json(registry_path)
    entries = [
        item
        for item in list(registry.get("entries", []) or [])
        if str(dict(item).get("manifest_json", "") or "") != str(manifest.get("manifest_json", "") or "")
    ]
    entry = {
        "manifest_json": str(manifest.get("manifest_json", "") or ""),
        "status": str(manifest.get("status", "") or ""),
        "scope": str(manifest.get("scope", "") or ""),
        "canonical_dataset_id": str(manifest.get("canonical_dataset_id", "") or ""),
        "profile": str(manifest.get("profile", "") or ""),
        "feature_schema_hash": str(manifest.get("feature_schema_hash", "") or ""),
        "stored_shard_count": int(manifest.get("stored_shard_count", 0) or 0),
        "processed_shard_count": int(manifest.get("processed_shard_count", 0) or 0),
        "empty_shard_count": int(manifest.get("empty_shard_count", 0) or 0),
        "planned_shard_count": int(manifest.get("planned_shard_count", 0) or 0),
        "registered_at": utc_now(),
    }
    entries.append(entry)
    registry.update(
        {
            "schema_version": 1,
            "status": "ready",
            "entries": entries,
            "latest_manifest_json": str(manifest.get("manifest_json", "") or ""),
            "latest_scope": str(manifest.get("scope", "") or ""),
            "updated_at": utc_now(),
        }
    )
    if str(manifest.get("scope", "")) == "full_canonical_candidate" and str(manifest.get("status", "")) == "completed":
        registry["active_manifest_json"] = str(manifest.get("manifest_json", "") or "")
    write_json(registry_path, registry)
    if str(manifest.get("status", "")) == "completed" and str(manifest.get("scope", "")) == "full_canonical_candidate":
        sharded_status = "sharded_full_ready"
    elif str(manifest.get("status", "")) == "completed":
        sharded_status = "sharded_validation_ready"
    else:
        sharded_status = "sharded_partial"
    status = {
        "status": sharded_status,
        "latest_manifest_json": str(manifest.get("manifest_json", "") or ""),
        "latest_scope": str(manifest.get("scope", "") or ""),
        "latest_stored_shard_count": int(manifest.get("stored_shard_count", 0) or 0),
        "latest_processed_shard_count": int(manifest.get("processed_shard_count", 0) or 0),
        "latest_empty_shard_count": int(manifest.get("empty_shard_count", 0) or 0),
        "latest_planned_shard_count": int(manifest.get("planned_shard_count", 0) or 0),
        "latest_profile": str(manifest.get("profile", "") or ""),
        "latest_canonical_dataset_id": str(manifest.get("canonical_dataset_id", "") or ""),
        "updated_at": utc_now(),
    }
    if str(manifest.get("scope", "")) == "full_canonical_candidate" and str(manifest.get("status", "")) == "completed":
        status["active_manifest_json"] = str(manifest.get("manifest_json", "") or "")
    write_root_manifest_update({"canonical_sharded_memmap_status": status}, paths=paths)


def _shard_files_exist(shard: Mapping[str, Any]) -> bool:
    for key in ("feature_store_path", "label_manifest_json", "sample_index_path", "shard_manifest_json"):
        value = str(shard.get(key, "") or "")
        if value and not Path(value).exists():
            return False
    label_manifest = read_json(Path(str(shard.get("label_manifest_json", "") or "")))
    for meta in dict(label_manifest.get("arrays", {}) or {}).values():
        path = Path(str(dict(meta).get("path", "") or ""))
        if not path.exists():
            return False
    return True


def _shard_is_resumable(shard: Mapping[str, Any]) -> bool:
    status = str(shard.get("status", "") or "")
    if status == _STORED_SHARD_STATUS:
        return _shard_files_exist(shard)
    if status == "schema_mismatch":
        return _shard_files_exist(shard)
    if status in _EMPTY_SHARD_STATUSES:
        path = Path(str(shard.get("shard_manifest_json", "") or ""))
        return bool(path.exists())
    return False


def _discover_existing_shards(out_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not out_root.exists():
        return rows
    for path in out_root.glob("year=*/block=*/shard_manifest.json"):
        row = read_json(path)
        if str(row.get("status", "")) in _RESUMABLE_SHARD_STATUSES:
            rows.append(dict(row))
    return rows


def _select_initial_schema_columns(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    candidates = [dict(row) for row in rows if str(dict(row).get("status", "")) in _SCHEMA_CANDIDATE_STATUSES and list(dict(row).get("feature_columns", []) or [])]
    if not candidates:
        return []
    best = max(candidates, key=_schema_candidate_score)
    return [str(column) for column in list(best.get("feature_columns", []) or [])]


def _schema_candidate_score(row: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    columns = [str(column) for column in list(row.get("feature_columns", []) or [])]
    history_state_prefixes = (
        "ret_",
        "vol_",
        "ret_accel_",
        "volume_ratio_",
        "adv_ratio_",
        "volatility_expansion",
    )
    history_state_score = sum(1 for column in columns if column.startswith(history_state_prefixes))
    year = int(row.get("year", 0) or 0)
    block_id = int(row.get("block_id", 0) or 0)
    status_score = 1 if str(row.get("status", "")) == _STORED_SHARD_STATUS else 0
    return (history_state_score, len(columns), year, status_score, -block_id)


def _normalize_resumed_shard(shard: Mapping[str, Any], *, expected_feature_columns: list[str]) -> dict[str, Any]:
    row = dict(shard)
    status = str(row.get("status", "") or "")
    if status in _EMPTY_SHARD_STATUSES:
        return row
    if status not in _SCHEMA_CANDIDATE_STATUSES:
        return row
    if not expected_feature_columns:
        if status == "schema_mismatch":
            row["status"] = _STORED_SHARD_STATUS
            row.pop("expected_feature_schema_hash", None)
            write_json(Path(str(row["shard_manifest_json"])), row)
        return row
    current_columns = [str(column) for column in list(row.get("feature_columns", []) or [])]
    if current_columns != list(expected_feature_columns):
        feature_path = Path(str(row.get("feature_store_path", "") or ""))
        feature_columns, feature_manifest = _project_feature_store_to_schema(
            feature_store_path=feature_path,
            feature_columns=current_columns,
            feature_manifest=dict(row.get("feature_manifest", {}) or {}),
            expected_feature_columns=list(expected_feature_columns),
        )
        row["feature_columns"] = list(feature_columns)
        row["feature_manifest"] = dict(feature_manifest)
        row["feature_store_shape"] = [int(item) for item in list(feature_manifest.get("feature_store_shape", []) or [])]
    row["status"] = _STORED_SHARD_STATUS
    row["feature_schema_hash"] = _sequence_hash(expected_feature_columns)
    row.pop("expected_feature_schema_hash", None)
    row["schema_normalized_at"] = utc_now()
    write_json(Path(str(row["shard_manifest_json"])), row)
    return row


def _build_schema_reference_shard(
    *,
    lake: ResearchDataLake,
    dataset_id: str,
    years: list[int],
    blocks: list[list[str]],
    canonical_start: pd.Timestamp,
    canonical_end: pd.Timestamp,
    cfg: ShardedMemmapConfig,
    out_root: Path,
    cumulative_horizons: tuple[int, ...],
    static_schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not years or not blocks:
        return {}
    complete_years = [year for year in sorted(years, reverse=True) if int(year) < int(canonical_end.year)]
    tail_years = [year for year in sorted(years, reverse=True) if int(year) >= int(canonical_end.year)]
    candidate_years = complete_years + tail_years
    best: dict[str, Any] = {}
    for year in candidate_years:
        for block_id, symbols in enumerate(blocks):
            shard = _build_one_shard(
                lake=lake,
                dataset_id=dataset_id,
                year=int(year),
                block_id=int(block_id),
                symbols=list(symbols),
                canonical_start=canonical_start,
                canonical_end=canonical_end,
                cfg=cfg,
                out_root=out_root,
                cumulative_horizons=cumulative_horizons,
                expected_feature_columns=None,
                static_schema=static_schema,
            )
            if str(shard.get("status", "")) != _STORED_SHARD_STATUS:
                continue
            if not best or _schema_candidate_score(shard) > _schema_candidate_score(best):
                best = dict(shard)
            if _schema_candidate_score(best)[0] > 0:
                return best
        if best:
            return best
    return best


def _write_progress(path: Path, **payload: Any) -> None:
    write_json(path, {"updated_at": utc_now(), **payload})


def _coerce_config(config: ShardedMemmapConfig | Mapping[str, Any] | None) -> ShardedMemmapConfig:
    if isinstance(config, ShardedMemmapConfig):
        return config.normalized()
    if isinstance(config, Mapping):
        return ShardedMemmapConfig(**dict(config)).normalized()
    return ShardedMemmapConfig().normalized()


def _sequence_hash(values: Iterable[Any]) -> str:
    payload = "\n".join(str(item) for item in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(value or "").strip()) or "sharded_memmap"
