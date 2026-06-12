from __future__ import annotations

import gc
import hashlib
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from daily_research.data_lake.catalog import ResearchDataLake
from daily_research.data_lake.policy_input_loader import load_policy_inputs_from_lake
from daily_research.path_policy.forecast_features import (
    DEFAULT_FORECAST_MAX_FEATURE_COLUMNS,
    build_forecast_feature_store,
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
        )


def write_sharded_memmap_plan(
    paths: QdpPaths | None = None,
    *,
    profile: str = DEFAULT_PROFILE,
    max_universe_size: int = 0,
    workers: int = 1,
    write: bool = True,
) -> dict[str, Any]:
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
    if cfg.max_universe_size > 0:
        universe = universe[: cfg.max_universe_size]
    blocks = list(_symbol_blocks(universe, cfg.symbol_block_size))
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
        )
        if reference:
            existing_by_key[str(reference.get("shard_key", ""))] = reference
            schema_columns = _select_initial_schema_columns(existing_by_key.values())
    schema_hash = _sequence_hash(schema_columns) if schema_columns else ""
    pending_specs: list[dict[str, Any]] = []
    build_mode = "parallel_threads" if int(cfg.workers) > 1 and schema_columns else "serial"
    if int(cfg.workers) > 1 and not schema_columns:
        build_mode = "serial_schema_unavailable"
    for year in years:
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
        "execution_mode": cfg.execution_mode,
        "max_feature_columns": int(cfg.max_feature_columns),
        "min_lookback_valid_ratio": float(cfg.min_lookback_valid_ratio),
        "build_mode": build_mode,
        "requested_worker_count": int(cfg.workers),
        "effective_worker_count": int(min(max(int(cfg.workers), 1), max(len(pending_specs), 1))) if build_mode == "parallel_threads" else 1,
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
) -> dict[str, Any]:
    target_start = max(pd.Timestamp(year=int(year), month=1, day=1), canonical_start)
    target_end = min(pd.Timestamp(year=int(year), month=12, day=31), canonical_end)
    context_start = max(target_start - pd.Timedelta(days=int(cfg.lookback_days) * 3 + 45), canonical_start)
    context_end = min(target_end + pd.Timedelta(days=int(cfg.horizon) * 4 + 21), canonical_end)
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
            min_trading_days=2,
            require_benchmark_open=str(cfg.execution_mode).strip().lower() == "next_open",
        )
    except ValueError as exc:
        message = str(exc)
        if "lake_coverage_blocker: no requested symbols are available in lake market data" not in message:
            raise
        shard = {
            "status": "no_coverage",
            "empty_reason": "no_requested_symbols_available_in_lake_market_data",
            "shard_key": f"year={year}/block={block_id:04d}",
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
            "error_summary": message[:500],
            "created_at": utc_now(),
        }
        write_json(shard_manifest_path, shard)
        return shard
    target_dates = [
        pd.Timestamp(dt).normalize()
        for dt in prepared.close.index
        if pd.Timestamp(dt).normalize() >= target_start and pd.Timestamp(dt).normalize() <= target_end
    ]
    if not target_dates:
        shard = {
            "status": "empty",
            "empty_reason": "no_target_dates",
            "shard_key": f"year={year}/block={block_id:04d}",
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
        write_json(shard_manifest_path, shard)
        return shard
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
        "shard_manifest_json": str(shard_manifest_path.resolve()),
        "created_at": utc_now(),
    }
    write_json(shard_manifest_path, shard)
    return shard


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
        "daily_excess_return": _stack_panel_map(labels.daily_excess_return, range(1, int(horizon) + 1), dates=dates, symbols=symbols),
        "cumulative_excess_return": _stack_panel_map(labels.cumulative_excess_return, cumulative_horizons, dates=dates, symbols=symbols),
        "rank_by_horizon": _stack_panel_map(labels.forward_rank, cumulative_horizons, dates=dates, symbols=symbols),
        "drawdown_by_horizon": _stack_panel_map(labels.path_max_drawdown_by_horizon, cumulative_horizons, dates=dates, symbols=symbols),
        "worst_by_horizon": _stack_panel_map(labels.path_worst_1d_by_horizon, cumulative_horizons, dates=dates, symbols=symbols),
        "upside_by_horizon": _stack_panel_map(labels.path_upside_capture_by_horizon, cumulative_horizons, dates=dates, symbols=symbols),
        "rank_20d": _panel_to_array(labels.forward_rank.get(int(horizon), labels.forward_rank[int(cumulative_horizons[-1])]), dates=dates, symbols=symbols),
        "max_drawdown_20d": _panel_to_array(labels.path_max_drawdown_20d, dates=dates, symbols=symbols),
        "worst_1d_20d": _panel_to_array(labels.path_worst_1d_20d, dates=dates, symbols=symbols),
        "upside_20d": _panel_to_array(labels.path_upside_capture_20d, dates=dates, symbols=symbols),
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
    rows = []
    for date_pos, stock_pos in np.argwhere(valid):
        rows.append(
            {
                "date": pd.Timestamp(dates[int(date_pos)]).strftime("%Y-%m-%d"),
                "stock": str(symbols[int(stock_pos)]),
                "date_pos": int(date_pos),
                "stock_pos": int(stock_pos),
                "history_valid_ratio": float(history[int(date_pos), int(stock_pos)]),
            }
        )
    sample_index = pd.DataFrame(rows)
    path = shard_dir / "sample_index.parquet"
    sample_index.to_parquet(path, index=False)
    return path, int(len(sample_index))


def _stack_panel_map(
    panels: Mapping[int, pd.DataFrame],
    keys: Iterable[int],
    *,
    dates: list[pd.Timestamp],
    symbols: list[str],
) -> np.ndarray:
    arrays = [_panel_to_array(panels[int(key)], dates=dates, symbols=symbols) for key in keys]
    return np.stack(arrays, axis=2).astype("float32") if arrays else np.empty((len(dates), len(symbols), 0), dtype="float32")


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
) -> dict[str, Any]:
    if not years or not blocks:
        return {}
    preferred_years = [year for year in years if int(year) > int(canonical_start.year)]
    candidate_years = preferred_years + [year for year in years if year not in set(preferred_years)]
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
