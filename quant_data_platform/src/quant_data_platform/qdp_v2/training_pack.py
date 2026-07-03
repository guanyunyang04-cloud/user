from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.ml_alpha import MARKET_STATE_LABELS
from daily_research.baseline.regime import compute_market_regime_state
from daily_research.continuous_policy.state_builder import (
    ALPHA_PRIOR_FRAME_NAMES,
    DEFAULT_SCORE_BLEND_WEIGHTS,
    STATE_SEQUENCE_BASES,
    STATE_SEQUENCE_LAGS,
    PreparedPolicyInputs,
)
from daily_research.baseline.advanced_ml_runtime import HistoryWindow
from daily_research.path_policy.forecast_dataset import (
    build_qdp_training_pack,
    build_static_context_vocab,
    normalize_static_context_fields,
    static_context_id_columns,
)
from daily_research.path_policy.labels import PATH20_HORIZON, normalize_cumulative_horizons
from quant_data_platform.core.json_io import json_safe, read_json, write_json
from quant_data_platform.memmap.sharded import (
    ShardedMemmapConfig,
    _EMPTY_SHARD_STATUSES,
    _STORED_SHARD_STATUS,
    _TERMINAL_SHARD_STATUSES,
    _accept_shard_schema,
    _build_one_shard_from_prepared,
    _safe_name,
    _scope_for_build,
    _sequence_hash,
    _shard_windows,
    _static_schema_manifest,
    _static_vocab_manifest,
    _symbol_blocks,
    _write_progress,
)
from quant_data_platform.qdp_v2.manifest import (
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    utc_now,
)
from quant_data_platform.qdp_v2.status import active_dataset_map


STYLE_ALPHA_PROFILE = "style_structural_alpha_v2"
DEFAULT_BENCHMARK = "000300.SH"
DEFAULT_START_YEAR = 2012
DEFAULT_END_YEAR = 2025
DEFAULT_TRAIN_START_YEAR = 2012
DEFAULT_TRAIN_END_YEAR = 2023
DEFAULT_VALIDATION_YEAR = 2024
DEFAULT_TEST_YEAR = 2025


@dataclass(frozen=True)
class V2TrainingPackConfig:
    workspace_root: str = ""
    tag: str = ""
    feature_profile: str = STYLE_ALPHA_PROFILE
    start_year: int = DEFAULT_START_YEAR
    end_year: int = DEFAULT_END_YEAR
    symbol_block_size: int = 300
    max_universe_size: int = 0
    max_shards: int = 0
    lookback_days: int = 252
    horizon: int = PATH20_HORIZON
    cumulative_horizons: str = "1,3,5,10,20"
    execution_mode: str = "next_open"
    max_feature_columns: int = 384
    min_lookback_valid_ratio: float = 0.80
    include_static_context: bool = True
    static_context_fields: str = "symbol,exchange,industry"
    train_start_year: int = DEFAULT_TRAIN_START_YEAR
    train_end_year: int = DEFAULT_TRAIN_END_YEAR
    validation_year: int = DEFAULT_VALIDATION_YEAR
    test_year: int = DEFAULT_TEST_YEAR
    feature_dtype: str = "float16"
    training_pack_stock_chunk_size: int = 96
    resume: bool = True
    build_training_pack: bool = True
    benchmark: str = DEFAULT_BENCHMARK
    benchmark_source: str = "baostock"
    duckdb_memory_limit: str = "12GB"
    duckdb_threads: int = 4
    json: bool = False

    def normalized(self) -> "V2TrainingPackConfig":
        return V2TrainingPackConfig(
            workspace_root=str(self.workspace_root or "").strip(),
            tag=str(self.tag or "").strip(),
            feature_profile=str(self.feature_profile or STYLE_ALPHA_PROFILE).strip(),
            start_year=max(int(self.start_year or DEFAULT_START_YEAR), 0),
            end_year=max(int(self.end_year or DEFAULT_END_YEAR), 0),
            symbol_block_size=max(int(self.symbol_block_size or 300), 1),
            max_universe_size=max(int(self.max_universe_size or 0), 0),
            max_shards=max(int(self.max_shards or 0), 0),
            lookback_days=max(int(self.lookback_days or 252), 1),
            horizon=max(int(self.horizon or PATH20_HORIZON), 1),
            cumulative_horizons=str(self.cumulative_horizons or "1,3,5,10,20"),
            execution_mode=str(self.execution_mode or "next_open").strip().lower(),
            max_feature_columns=max(int(self.max_feature_columns or 384), 1),
            min_lookback_valid_ratio=float(self.min_lookback_valid_ratio),
            include_static_context=bool(self.include_static_context),
            static_context_fields=str(self.static_context_fields or "symbol,exchange,industry").strip(),
            train_start_year=max(int(self.train_start_year or DEFAULT_TRAIN_START_YEAR), 0),
            train_end_year=max(int(self.train_end_year or DEFAULT_TRAIN_END_YEAR), 0),
            validation_year=max(int(self.validation_year or DEFAULT_VALIDATION_YEAR), 0),
            test_year=max(int(self.test_year or DEFAULT_TEST_YEAR), 0),
            feature_dtype=str(self.feature_dtype or "float16").strip().lower(),
            training_pack_stock_chunk_size=max(int(self.training_pack_stock_chunk_size or 96), 1),
            resume=bool(self.resume),
            build_training_pack=bool(self.build_training_pack),
            benchmark=_normalize_symbol(self.benchmark or DEFAULT_BENCHMARK),
            benchmark_source=str(self.benchmark_source or "baostock").strip().lower(),
            duckdb_memory_limit=str(self.duckdb_memory_limit or "12GB").strip(),
            duckdb_threads=max(int(self.duckdb_threads or 4), 1),
            json=bool(self.json),
        )


@dataclass(frozen=True)
class _StaticPrepared:
    universe: tuple[str, ...]
    metadata_frames: dict[str, pd.DataFrame]


def build_v2_training_pack(config: V2TrainingPackConfig | Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = _coerce_config(config)
    root = qdp_v2_root(cfg.workspace_root or None)
    active = read_active_manifest(root)
    active_map = active_dataset_map(active)
    if not active_map:
        raise ValueError(f"qdp_v2_active_missing: {root / 'active' / 'active.json'}")

    canonical_start = pd.Timestamp(dict(active.get("scope", {}) or {}).get("start_date", "2011-11-22")).normalize()
    canonical_end = pd.Timestamp(active.get("active_as_of_date") or dict(active.get("scope", {}) or {}).get("end_date", "2026-06-26")).normalize()
    years = [year for year in range(int(cfg.start_year), int(cfg.end_year) + 1) if year >= int(canonical_start.year) and year <= int(canonical_end.year)]
    if not years:
        raise ValueError("training_pack_no_years_in_active_scope")

    universe = _load_universe(root=root, active_map=active_map, cfg=cfg)
    if cfg.max_universe_size > 0:
        universe = universe[: int(cfg.max_universe_size)]
    if not universe:
        raise ValueError("training_pack_empty_universe")

    sharded_root = root / "research" / "sharded_memmap"
    pack_root = root / "research" / "training_pack"
    tag = cfg.tag or f"qdp_v2_{cfg.feature_profile}_{years[0]}_{years[-1]}_{_compact_now()}"
    out_root = sharded_root / _safe_name(tag)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "sharded_memmap_manifest.json"
    progress_path = out_root / "sharded_memmap_progress.json"

    sharded_cfg = ShardedMemmapConfig(
        profile=cfg.feature_profile,
        start_year=years[0],
        end_year=years[-1],
        symbol_block_size=cfg.symbol_block_size,
        max_universe_size=cfg.max_universe_size,
        max_shards=cfg.max_shards,
        lookback_days=cfg.lookback_days,
        horizon=cfg.horizon,
        cumulative_horizons=cfg.cumulative_horizons,
        execution_mode=cfg.execution_mode,
        max_feature_columns=cfg.max_feature_columns,
        min_lookback_valid_ratio=cfg.min_lookback_valid_ratio,
        tag=tag,
        resume=cfg.resume,
        include_static_context=cfg.include_static_context,
        static_context_fields=cfg.static_context_fields,
    ).normalized()

    blocks = list(_symbol_blocks(universe, sharded_cfg.symbol_block_size))
    cumulative_horizons = normalize_cumulative_horizons(sharded_cfg.cumulative_horizons, horizon=sharded_cfg.horizon)
    static_schema = _static_context_schema_for_v2_universe(
        root=root,
        active_map=active_map,
        universe=universe,
        enabled=bool(cfg.include_static_context),
        static_context_fields=cfg.static_context_fields,
        cfg=cfg,
    )

    planned_specs: list[dict[str, Any]] = []
    for year in years:
        for block_id, symbols in enumerate(blocks):
            planned_specs.append({"year": int(year), "block_id": int(block_id), "symbols": list(symbols)})
    if cfg.max_shards > 0:
        planned_specs = planned_specs[: int(cfg.max_shards)]

    completed: list[dict[str, Any]] = []
    schema_hash = ""
    schema_columns: list[str] = []
    started_at = utc_now()
    _write_progress(
        progress_path,
        status="running",
        build_mode="qdp_v2_manifest_first",
        planned_shards=len(planned_specs),
        completed_shards=0,
        tag=tag,
    )
    for idx, spec in enumerate(planned_specs, start=1):
        year = int(spec["year"])
        block_id = int(spec["block_id"])
        symbols = [str(item) for item in spec["symbols"]]
        target_start, target_end, context_start, context_end = _shard_windows(
            year=year,
            canonical_start=canonical_start,
            canonical_end=canonical_end,
            cfg=sharded_cfg,
        )
        shard_dir = out_root / f"year={year}" / f"block={block_id:04d}"
        shard_manifest_path = shard_dir / "shard_manifest.json"
        if cfg.resume and shard_manifest_path.exists():
            existing = read_json(shard_manifest_path)
            if str(existing.get("status", "")) in _TERMINAL_SHARD_STATUSES:
                schema_hash, schema_columns = _accept_shard_schema(
                    shard=existing,
                    schema_hash=schema_hash,
                    schema_columns=schema_columns,
                )
                completed.append(existing)
                _write_progress(progress_path, status="running", planned_shards=len(planned_specs), completed_shards=len(completed), latest_shard=existing)
                continue

        prepared = _load_prepared_from_v2(
            root=root,
            active=active,
            active_map=active_map,
            symbols=symbols,
            start_date=context_start,
            end_date=context_end,
            cfg=cfg,
        )
        shard = _build_one_shard_from_prepared(
            prepared=prepared,
            year=year,
            block_id=block_id,
            target_start=target_start,
            target_end=target_end,
            context_start=context_start,
            context_end=context_end,
            cfg=sharded_cfg,
            out_root=out_root,
            cumulative_horizons=tuple(int(item) for item in cumulative_horizons),
            expected_feature_columns=schema_columns or None,
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
            planned_shards=len(planned_specs),
            completed_shards=len(completed),
            latest_shard=shard,
            elapsed_sec=round(time.perf_counter(), 3),
            progress_ratio=float(idx / max(len(planned_specs), 1)),
        )

    stored = [item for item in completed if str(item.get("status", "")) == _STORED_SHARD_STATUS]
    empty = [item for item in completed if str(item.get("status", "")) in _EMPTY_SHARD_STATUSES]
    failed = [item for item in completed if str(item.get("status", "")) not in _TERMINAL_SHARD_STATUSES]
    status = "completed" if len(completed) == len(planned_specs) and not failed else "partial"
    scope = _scope_for_build(
        cfg=sharded_cfg,
        universe_size=len(universe),
        full_universe_size=int(dict(active.get("scope", {}) or {}).get("symbol_count", len(universe)) or len(universe)),
        years=years,
        canonical_start=canonical_start,
        canonical_end=canonical_end,
    )
    sharded_manifest = {
        "artifact_type": "qdp_sharded_memmap",
        "schema_version": 1,
        "status": status,
        "scope": scope,
        "canonical_dataset_id": "qdp_v2_active",
        "canonical_alias": "qdp_v2_active_manifest_first",
        "canonical_dataset_source": "qdp_v2_active_json",
        "source_market_dataset_id": str(active_map.get("market_daily_raw", "")),
        "source_active_manifest_json": str((root / "active" / "active.json").resolve()),
        "source_data_base": "qdp_v2",
        "profile": sharded_cfg.profile,
        "feature_profile": sharded_cfg.profile,
        "feature_schema_hash": schema_hash,
        "feature_columns": schema_columns,
        "feature_count": int(len(schema_columns)),
        "start_year": int(years[0]),
        "end_year": int(years[-1]),
        "years": [int(year) for year in years],
        "symbol_count": int(len(universe)),
        "full_universe_size": int(dict(active.get("scope", {}) or {}).get("symbol_count", len(universe)) or len(universe)),
        "symbol_block_size": int(sharded_cfg.symbol_block_size),
        "planned_shard_count": len(planned_specs),
        "processed_shard_count": int(len(completed)),
        "stored_shard_count": int(len(stored)),
        "empty_shard_count": int(len(empty)),
        "failed_shard_count": int(len(failed)),
        "shards": completed,
        "lookback_days": int(sharded_cfg.lookback_days),
        "horizon": int(sharded_cfg.horizon),
        "cumulative_horizons": [int(item) for item in cumulative_horizons],
        "label_schema_name": "path20_basic_v2",
        "label_schema_version": 2,
        "execution_mode": sharded_cfg.execution_mode,
        "max_feature_columns": int(sharded_cfg.max_feature_columns),
        "min_lookback_valid_ratio": float(sharded_cfg.min_lookback_valid_ratio),
        "build_mode": "qdp_v2_manifest_first",
        "requested_worker_count": 1,
        "effective_worker_count": 1,
        "year_input_cache_requested": False,
        "year_input_cache_effective": False,
        "static_context_schema": _static_schema_manifest(static_schema, enabled=bool(cfg.include_static_context)),
        "static_context_vocab": _static_vocab_manifest(static_schema) if bool(cfg.include_static_context) else {},
        "created_at": started_at,
        "manifest_json": str(manifest_path.resolve()),
    }
    write_json(manifest_path, sharded_manifest)
    _write_progress(progress_path, status=status, planned_shards=len(planned_specs), completed_shards=len(completed), latest_shard={})

    payload: dict[str, Any] = {
        "status": status,
        "qdp_v2_root": str(root.resolve()),
        "sharded_manifest_json": str(manifest_path.resolve()),
        "planned_shard_count": len(planned_specs),
        "stored_shard_count": len(stored),
        "empty_shard_count": len(empty),
        "failed_shard_count": len(failed),
        "feature_count": int(len(schema_columns)),
        "feature_schema_hash": schema_hash,
        "symbol_count": int(len(universe)),
        "years": years,
        "tag": tag,
    }
    if cfg.build_training_pack and status in {"completed", "partial"} and stored:
        training_manifest = build_qdp_training_pack(
            manifest_path,
            output_root=pack_root,
            tag=f"{_safe_name(tag)}_training_pack",
            train_start_year=cfg.train_start_year,
            train_end_year=cfg.train_end_year,
            validation_year=cfg.validation_year,
            test_year=cfg.test_year,
            feature_dtype=cfg.feature_dtype,
            stock_chunk_size=cfg.training_pack_stock_chunk_size,
            resume=cfg.resume,
        )
        payload["training_pack_manifest_json"] = str(training_manifest.get("manifest_json", ""))
        payload["training_pack_status"] = str(training_manifest.get("status", ""))
        payload["training_pack_sample_count"] = int(training_manifest.get("sample_count", 0) or 0)
        payload["training_pack_feature_count"] = int(training_manifest.get("feature_count", 0) or 0)
        payload["training_pack_sample_count_by_role"] = dict(training_manifest.get("sample_count_by_role", {}) or {})
    return payload


def _load_prepared_from_v2(
    *,
    root: Path,
    active: Mapping[str, Any],
    active_map: Mapping[str, str],
    symbols: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cfg: V2TrainingPackConfig,
) -> PreparedPolicyInputs:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    dates = _load_open_dates(root=root, active_map=active_map, start_date=start, end_date=end)
    symbols = [str(item).strip().upper() for item in symbols if str(item).strip()]
    market = _read_domain_frame(
        root=root,
        active_map=active_map,
        domain="market_daily_raw",
        columns=["trade_date", "symbol", "open", "high", "low", "close", "volume", "amount"],
        start_date=start,
        end_date=end,
        symbols=symbols,
        cfg=cfg,
    )
    panels = {
        field: _pivot_panel(market, field, dates=dates, symbols=symbols)
        for field in ("open", "high", "low", "close", "volume", "amount")
    }
    close = panels["close"]
    open_ = panels["open"]
    high = panels["high"]
    low = panels["low"]
    volume = panels["volume"]
    amount = panels["amount"]
    membership = close.notna() & open_.notna() & high.notna() & low.notna()

    benchmark = _load_benchmark_frame(root=root, start_date=start, end_date=end, cfg=cfg)
    benchmark = benchmark.set_index("trade_date").sort_index() if not benchmark.empty else pd.DataFrame(index=dates)
    benchmark.index = pd.to_datetime(benchmark.index).normalize()
    benchmark_close = pd.to_numeric(benchmark.get("close", pd.Series(index=dates, dtype=float)), errors="coerce").reindex(dates)
    benchmark_open = pd.to_numeric(benchmark.get("open", benchmark_close), errors="coerce").reindex(dates)

    zero = pd.DataFrame(0.0, index=dates, columns=symbols, dtype=float)
    score_none = zero.copy()
    score_v2 = zero.copy()
    score_blend = score_none * float(DEFAULT_SCORE_BLEND_WEIGHTS[0]) + score_v2 * float(DEFAULT_SCORE_BLEND_WEIGHTS[1])

    derived_frames = _base_derived_frames(
        close=close,
        volume=volume,
        amount=amount,
        score_blend=score_blend,
        zero=zero,
    )
    derived_frames.update(_load_valuation_frames(root=root, active_map=active_map, dates=dates, symbols=symbols, start=start, end=end, cfg=cfg))
    derived_frames.update(_load_adjust_frames(root=root, active_map=active_map, dates=dates, symbols=symbols, start=start, end=end, cfg=cfg))
    derived_frames.update(_load_intraday_feature_frames(root=root, active_map=active_map, dates=dates, symbols=symbols, start=start, end=end, cfg=cfg))
    derived_frames.update(_load_index_constituent_frames(root=root, active_map=active_map, dates=dates, symbols=symbols, start=start, end=end, cfg=cfg))

    feature_frames: dict[str, pd.DataFrame] = {}
    market_features: dict[str, pd.Series] = {}
    industry_daily = _read_domain_frame(
        root=root,
        active_map=active_map,
        domain="industry_concept",
        columns=["trade_date", "symbol", "industry"],
        start_date=start,
        end_date=end,
        symbols=symbols,
        cfg=cfg,
    )
    if not industry_daily.empty:
        industry_daily["trade_date"] = pd.to_datetime(industry_daily["trade_date"], errors="coerce").dt.normalize()
        industry_daily["symbol"] = industry_daily["symbol"].astype(str).str.strip().str.upper()
        industry_daily["industry"] = industry_daily["industry"].fillna("").astype(str).str.strip()
    industry_map = (
        industry_daily.sort_values("trade_date").drop_duplicates("symbol", keep="last")[["symbol", "industry"]].reset_index(drop=True)
        if not industry_daily.empty
        else pd.DataFrame({"symbol": symbols, "industry": [""] * len(symbols)})
    )
    history_window = HistoryWindow(
        mode="train",
        requested_start_date=start.strftime("%Y%m%d"),
        effective_start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        required_trading_days=int(cfg.lookback_days),
    )
    return PreparedPolicyInputs(
        universe=tuple(symbols),
        pool_name="qdp_v2_active_scope",
        benchmark=cfg.benchmark,
        data_source="qdp_v2",
        csv_folder="",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        requested_start_date=start.strftime("%Y%m%d"),
        history_window=history_window,
        raw_cache_meta={"source": "qdp_v2_active", "active_as_of_date": str(active.get("active_as_of_date", ""))},
        prepared_cache_meta={"source": "qdp_v2_training_pack_adapter", "created_at": utc_now()},
        close=close,
        open_=open_,
        high=high,
        low=low,
        volume=volume,
        amount=amount,
        benchmark_close=benchmark_close,
        benchmark_open=benchmark_open,
        score_none=score_none,
        score_v2=score_v2,
        score_blend=score_blend,
        feature_frames=feature_frames,
        market_features=market_features,
        membership_frame=membership.astype(bool),
        rolling_pool_summary={
            "pool_name": "qdp_v2_active_scope",
            "selection_scope": "qdp_v2_mainboard_scope",
            "selection_mode": "pit_universe_with_real_daily_bar",
            "member_count_median": int(membership.sum(axis=1).median()) if not membership.empty else 0,
        },
        alpha_prior_summary={"source": "none", "status": "disabled"},
        derived_frames=derived_frames,
        metadata_frames={
            "industry_daily": industry_daily,
            "industry_map": industry_map,
        },
        metadata_summary={
            "qdp_v2_active_manifest_json": str((root / "active" / "active.json").resolve()),
            "industry_source": str(active_map.get("industry_concept", "")),
        },
    )


def _base_derived_frames(
    *,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    amount: pd.DataFrame,
    score_blend: pd.DataFrame,
    zero: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    returns_1d = close.pct_change(1, fill_method=None)
    rolling_high_20 = close.rolling(20, min_periods=1).max()
    rolling_high_60 = close.rolling(60, min_periods=1).max()
    rolling_low_20 = close.rolling(20, min_periods=1).min()
    derived: dict[str, pd.DataFrame] = {
        "ret_1d": returns_1d,
        "ret_3d": close.pct_change(3, fill_method=None),
        "ret_5d": close.pct_change(5, fill_method=None),
        "ret_10d": close.pct_change(10, fill_method=None),
        "ret_20d": close.pct_change(20, fill_method=None),
        "vol_5d": returns_1d.rolling(5).std(),
        "vol_20d": returns_1d.rolling(20).std(),
        "adv20": amount.rolling(20).mean(),
        "volume_ratio_5_20": volume.rolling(5).mean().div(volume.rolling(20).mean().replace(0, np.nan)),
        "score_blend": score_blend,
        "score_delta_1d": score_blend.diff(1),
        "score_delta_5d": score_blend.diff(5),
        "score_delta_accel": score_blend.diff(1).sub(score_blend.diff(5).div(5.0)),
        "ret_accel_5_20": close.pct_change(5, fill_method=None).sub(close.pct_change(20, fill_method=None).div(4.0)),
        "distance_to_20d_high": close.div(rolling_high_20.replace(0, np.nan)).sub(1.0),
        "distance_to_60d_high": close.div(rolling_high_60.replace(0, np.nan)).sub(1.0),
        "distance_to_20d_low": close.div(rolling_low_20.replace(0, np.nan)).sub(1.0),
        "volatility_expansion": returns_1d.rolling(5).std().div(returns_1d.rolling(20).std().replace(0, np.nan)).sub(1.0),
        "adv_ratio_5_20": amount.rolling(5).mean().div(amount.rolling(20).mean().replace(0, np.nan)).sub(1.0),
    }
    for name in ALPHA_PRIOR_FRAME_NAMES:
        derived[str(name)] = zero.copy()
    for base_name in STATE_SEQUENCE_BASES:
        frame = derived.get(base_name)
        if frame is None:
            continue
        for lag in STATE_SEQUENCE_LAGS:
            derived[f"{base_name}_lag{int(lag)}"] = frame.shift(int(lag))
    return {key: value.replace([np.inf, -np.inf], np.nan) for key, value in derived.items()}


def _feature_frames(
    *,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    amount: pd.DataFrame,
    score_none: pd.DataFrame,
    score_v2: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()
    adv20 = amount.rolling(20).mean()
    vol20 = volume.rolling(20).mean()
    return {
        "z_score_none": _zscore_cs(score_none),
        "z_score_v2": _zscore_cs(score_v2),
        "adv20_rank": adv20.rank(axis=1, pct=True),
        "price_rank": close.rank(axis=1, pct=True),
        "ma20_gap": close.div(ema20).sub(1.0).replace([np.inf, -np.inf], np.nan),
        "ma60_gap": close.div(ema60).sub(1.0).replace([np.inf, -np.inf], np.nan),
        "volume_rank": vol20.rank(axis=1, pct=True),
    }


def _market_features(*, benchmark_close: pd.Series, universe: list[str]) -> dict[str, pd.Series]:
    cfg = ResearchConfig(universe=list(universe), universe_scope="qdp_v2", benchmark=DEFAULT_BENCHMARK)
    regime = compute_market_regime_state(benchmark_close, cfg)
    out: dict[str, pd.Series] = {
        "benchmark_trend_gap": regime["benchmark_trend_gap"],
        "benchmark_annual_vol": regime["benchmark_annual_vol"],
        "benchmark_vol_gap": regime["benchmark_vol_gap"],
        "benchmark_vol_ratio": regime["benchmark_vol_ratio"],
        "is_up_low": regime["quadrant"].eq("trend_up_low_vol").astype(float),
        "is_up_high": regime["quadrant"].eq("trend_up_high_vol").astype(float),
        "is_down_low": regime["quadrant"].eq("trend_down_low_vol").astype(float),
        "is_down_high": regime["quadrant"].eq("trend_down_high_vol").astype(float),
        "is_trend_up": regime["trend_bucket"].eq("trend_up").astype(float),
        "is_trend_flat": regime["trend_bucket"].eq("trend_flat").astype(float),
        "is_trend_down": regime["trend_bucket"].eq("trend_down").astype(float),
        "is_vol_low": regime["vol_bucket"].eq("vol_low").astype(float),
        "is_vol_mid": regime["vol_bucket"].eq("vol_mid").astype(float),
        "is_vol_high": regime["vol_bucket"].eq("vol_high").astype(float),
        "regime_on": regime["regime_on"].astype(float),
    }
    for label in MARKET_STATE_LABELS:
        out[f"is_{label}"] = regime["market_state"].eq(label).astype(float)
    return out


def _load_valuation_frames(
    *,
    root: Path,
    active_map: Mapping[str, str],
    dates: pd.DatetimeIndex,
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: V2TrainingPackConfig,
) -> dict[str, pd.DataFrame]:
    data = _read_domain_frame(
        root=root,
        active_map=active_map,
        domain="valuation",
        columns=["trade_date", "symbol", "turnover_rate", "pe", "pb"],
        start_date=start,
        end_date=end,
        symbols=symbols,
        cfg=cfg,
        required=False,
    )
    if data.empty:
        return {}
    return {
        "turn": _pivot_panel(data, "turnover_rate", dates=dates, symbols=symbols),
        "peTTM": _pivot_panel(data, "pe", dates=dates, symbols=symbols),
        "pbMRQ": _pivot_panel(data, "pb", dates=dates, symbols=symbols),
        "psTTM": pd.DataFrame(np.nan, index=dates, columns=symbols, dtype=float),
        "pcfNcfTTM": pd.DataFrame(np.nan, index=dates, columns=symbols, dtype=float),
    }


def _load_adjust_frames(
    *,
    root: Path,
    active_map: Mapping[str, str],
    dates: pd.DatetimeIndex,
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: V2TrainingPackConfig,
) -> dict[str, pd.DataFrame]:
    data = _read_domain_frame(
        root=root,
        active_map=active_map,
        domain="adjust_factor",
        columns=["trade_date", "symbol", "adjust_factor", "fore_adjust_factor", "back_adjust_factor"],
        start_date=start,
        end_date=end,
        symbols=symbols,
        cfg=cfg,
        required=False,
    )
    if data.empty:
        return {}
    out = {
        "adjust_factor": _pivot_panel(data, "adjust_factor", dates=dates, symbols=symbols),
        "adjust_factor_fore_adjust_factor": _pivot_panel(data, "fore_adjust_factor", dates=dates, symbols=symbols),
        "adjust_factor_back_adjust_factor": _pivot_panel(data, "back_adjust_factor", dates=dates, symbols=symbols),
    }
    return out


def _load_intraday_feature_frames(
    *,
    root: Path,
    active_map: Mapping[str, str],
    dates: pd.DatetimeIndex,
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: V2TrainingPackConfig,
) -> dict[str, pd.DataFrame]:
    manifest = _manifest_for_domain(root=root, active_map=active_map, domain="intraday_daily_features", required=False)
    if manifest is None:
        return {}
    fields = [
        str(item.get("name", ""))
        for item in list(manifest.schema or [])
        if str(item.get("name", "")) not in {"trade_date", "symbol", "source", "adjusted_flag"}
    ]
    if not fields:
        return {}
    data = _read_domain_frame(
        root=root,
        active_map=active_map,
        domain="intraday_daily_features",
        columns=["trade_date", "symbol", *fields],
        start_date=start,
        end_date=end,
        symbols=symbols,
        cfg=cfg,
        required=False,
    )
    if data.empty:
        return {}
    return {
        f"intraday_daily_features_{field}": _pivot_panel(data, field, dates=dates, symbols=symbols)
        for field in fields
        if field in data.columns
    }


def _load_index_constituent_frames(
    *,
    root: Path,
    active_map: Mapping[str, str],
    dates: pd.DatetimeIndex,
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: V2TrainingPackConfig,
) -> dict[str, pd.DataFrame]:
    data = _read_domain_frame(
        root=root,
        active_map=active_map,
        domain="index_constituents",
        columns=["trade_date", "symbol", "index_symbol"],
        start_date=start,
        end_date=end,
        symbols=symbols,
        cfg=cfg,
        required=False,
    )
    if data.empty:
        return {}
    data["member"] = 1.0
    out: dict[str, pd.DataFrame] = {}
    for index_symbol, group in data.groupby("index_symbol", sort=True):
        suffix = str(index_symbol).strip().upper().replace(".", "_")
        out[f"index_constituents_{suffix}_member"] = _pivot_panel(group, "member", dates=dates, symbols=symbols).fillna(0.0)
    return out


def _load_open_dates(*, root: Path, active_map: Mapping[str, str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DatetimeIndex:
    cfg = V2TrainingPackConfig()
    data = _read_domain_frame(
        root=root,
        active_map=active_map,
        domain="trading_calendar",
        columns=["trade_date", "is_open"],
        start_date=start_date,
        end_date=end_date,
        symbols=[],
        cfg=cfg,
    )
    if data.empty:
        return pd.DatetimeIndex([])
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    data = data.loc[data["trade_date"].notna()]
    if "is_open" in data.columns:
        data = data.loc[data["is_open"].astype(bool)]
    return pd.DatetimeIndex(sorted(data["trade_date"].drop_duplicates().tolist()))


def _load_universe(*, root: Path, active_map: Mapping[str, str], cfg: V2TrainingPackConfig) -> list[str]:
    data = _read_domain_frame(
        root=root,
        active_map=active_map,
        domain="universe_snapshot",
        columns=["symbol"],
        start_date=pd.Timestamp("1900-01-01"),
        end_date=pd.Timestamp("2100-01-01"),
        symbols=[],
        cfg=cfg,
        distinct=True,
    )
    symbols = sorted({str(item).strip().upper() for item in data.get("symbol", pd.Series(dtype=str)).tolist() if str(item).strip()})
    return [symbol for symbol in symbols if symbol != _normalize_symbol(cfg.benchmark)]


def _static_context_schema_for_v2_universe(
    *,
    root: Path,
    active_map: Mapping[str, str],
    universe: list[str],
    enabled: bool,
    static_context_fields: str,
    cfg: V2TrainingPackConfig,
) -> dict[str, Any]:
    fields = normalize_static_context_fields(static_context_fields)
    if not bool(enabled):
        return {
            "enabled": False,
            "fields": list(fields),
            "id_columns": list(static_context_id_columns(fields)),
            "vocab_sizes": {},
        }
    symbols = [str(item).strip().upper() for item in universe if str(item).strip()]
    industry_map = _load_latest_industry_map(root=root, active_map=active_map, symbols=symbols, cfg=cfg)
    prepared = _StaticPrepared(
        universe=tuple(symbols),
        metadata_frames={"industry_map": industry_map},
    )
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


def _load_latest_industry_map(
    *,
    root: Path,
    active_map: Mapping[str, str],
    symbols: list[str],
    cfg: V2TrainingPackConfig,
) -> pd.DataFrame:
    manifest = _manifest_for_domain(root=root, active_map=active_map, domain="industry_concept", required=False)
    if manifest is None or not symbols:
        return pd.DataFrame({"symbol": symbols, "industry": [""] * len(symbols)})
    paths = _paths_for_manifest(root=root, manifest=manifest, start_date=pd.Timestamp("1900-01-01"), end_date=pd.Timestamp("2100-01-01"))
    if not paths or not _domain_has_column(manifest, "industry"):
        return pd.DataFrame({"symbol": symbols, "industry": [""] * len(symbols)})

    import duckdb  # type: ignore

    con = duckdb.connect(":memory:")
    con.execute(f"set memory_limit='{cfg.duckdb_memory_limit}'")
    con.execute(f"set threads={int(cfg.duckdb_threads)}")
    con.register("_qdp_symbols", pd.DataFrame({"symbol": symbols}))
    try:
        result = con.execute(
            """
            with latest as (
              select
                cast(t.symbol as varchar) as symbol,
                coalesce(cast(t.industry as varchar), '') as industry,
                row_number() over (
                  partition by cast(t.symbol as varchar)
                  order by cast(t.trade_date as date) desc
                ) as rn
              from read_parquet(?) t
              where cast(t.symbol as varchar) in (select symbol from _qdp_symbols)
                and t.trade_date is not null
            )
            select symbol, industry
            from latest
            where rn = 1
            """,
            [paths],
        ).fetchdf()
    finally:
        con.close()
    if result.empty:
        return pd.DataFrame({"symbol": symbols, "industry": [""] * len(symbols)})
    result["symbol"] = result["symbol"].astype(str).str.strip().str.upper()
    result["industry"] = result["industry"].fillna("").astype(str).str.strip()
    merged = pd.DataFrame({"symbol": symbols}).merge(result, on="symbol", how="left")
    merged["industry"] = merged["industry"].fillna("").astype(str)
    return merged[["symbol", "industry"]]


def _read_domain_frame(
    *,
    root: Path,
    active_map: Mapping[str, str],
    domain: str,
    columns: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    symbols: list[str],
    cfg: V2TrainingPackConfig,
    required: bool = True,
    distinct: bool = False,
) -> pd.DataFrame:
    manifest = _manifest_for_domain(root=root, active_map=active_map, domain=domain, required=required)
    if manifest is None:
        return pd.DataFrame(columns=columns)
    paths = _paths_for_manifest(root=root, manifest=manifest, start_date=start_date, end_date=end_date)
    if not paths:
        return pd.DataFrame(columns=columns)
    import duckdb  # type: ignore

    con = duckdb.connect(":memory:")
    con.execute(f"set memory_limit='{cfg.duckdb_memory_limit}'")
    con.execute(f"set threads={int(cfg.duckdb_threads)}")
    select_cols = ", ".join(f"t.{col}" for col in columns)
    if distinct:
        select_cols = "distinct " + select_cols
    where = []
    params: list[Any] = [paths]
    if "trade_date" in columns or _domain_has_column(manifest, "trade_date"):
        where.append("cast(t.trade_date as varchar) >= ? and cast(t.trade_date as varchar) <= ?")
        params.extend([start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")])
    if symbols and _domain_has_column(manifest, "symbol"):
        symbol_frame = pd.DataFrame({"symbol": [str(item).strip().upper() for item in symbols]})
        con.register("_qdp_symbols", symbol_frame)
        where.append("t.symbol in (select symbol from _qdp_symbols)")
    query = f"select {select_cols} from read_parquet(?) t"
    if where:
        query += " where " + " and ".join(where)
    try:
        return con.execute(query, params).fetchdf()
    finally:
        con.close()


def _manifest_for_domain(*, root: Path, active_map: Mapping[str, str], domain: str, required: bool = True):
    dataset_id = str(active_map.get(domain, "") or "")
    if not dataset_id:
        if required:
            raise ValueError(f"qdp_v2_active_missing_domain:{domain}")
        return None
    path = dataset_manifest_for_id(root, dataset_id, domain)
    if path is None:
        if required:
            raise ValueError(f"qdp_v2_dataset_manifest_missing:{domain}:{dataset_id}")
        return None
    return read_dataset_manifest(path)


def _paths_for_manifest(*, root: Path, manifest: Any, start_date: pd.Timestamp, end_date: pd.Timestamp) -> list[str]:
    start_s = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    paths: list[str] = []
    for shard in list(manifest.shards or []):
        shard_start = str(getattr(shard, "start_date", "") or start_s)
        shard_end = str(getattr(shard, "end_date", "") or end_s)
        if shard_end < start_s or shard_start > end_s:
            continue
        path = resolve_manifest_path(getattr(shard, "path", ""), root=root)
        if path.exists():
            paths.append(str(path.resolve()))
    return paths


def _domain_has_column(manifest: Any, column: str) -> bool:
    return str(column) in {str(item.get("name", "")) for item in list(getattr(manifest, "schema", []) or [])}


def _pivot_panel(data: pd.DataFrame, field: str, *, dates: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    if data.empty or field not in data.columns:
        return pd.DataFrame(np.nan, index=dates, columns=symbols, dtype=float)
    working = data.loc[:, ["trade_date", "symbol", field]].copy()
    working["trade_date"] = pd.to_datetime(working["trade_date"], errors="coerce").dt.normalize()
    working["symbol"] = working["symbol"].astype(str).str.strip().str.upper()
    working[field] = pd.to_numeric(working[field], errors="coerce")
    panel = working.pivot_table(index="trade_date", columns="symbol", values=field, aggfunc="last")
    return panel.reindex(index=dates, columns=symbols).astype(float)


def _load_benchmark_frame(*, root: Path, start_date: pd.Timestamp, end_date: pd.Timestamp, cfg: V2TrainingPackConfig) -> pd.DataFrame:
    cache_path = root / "runs" / f"benchmark_daily_{_safe_name(cfg.benchmark)}.parquet"
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached["trade_date"] = pd.to_datetime(cached["trade_date"], errors="coerce").dt.normalize()
        if not cached.empty and _benchmark_cache_covers(root=root, cached=cached, start_date=start_date, end_date=end_date, cfg=cfg):
            return cached.loc[(cached["trade_date"] >= start_date) & (cached["trade_date"] <= end_date)].copy()
    if cfg.benchmark_source not in {"baostock", "auto"}:
        raise ValueError(f"unsupported_benchmark_source:{cfg.benchmark_source}")
    fetched = _fetch_benchmark_baostock(symbol=cfg.benchmark, start_date=start_date, end_date=end_date)
    existing = pd.DataFrame()
    if cache_path.exists():
        existing = pd.read_parquet(cache_path)
    combined = pd.concat([existing, fetched], ignore_index=True) if not existing.empty else fetched
    if not combined.empty:
        combined["trade_date"] = pd.to_datetime(combined["trade_date"], errors="coerce").dt.normalize()
        combined = combined.dropna(subset=["trade_date"]).sort_values("trade_date").drop_duplicates("trade_date", keep="last")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(cache_path, index=False)
    return combined.loc[(combined["trade_date"] >= start_date) & (combined["trade_date"] <= end_date)].copy()


def _benchmark_cache_covers(
    *,
    root: Path,
    cached: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cfg: V2TrainingPackConfig,
) -> bool:
    if cached.empty or "trade_date" not in cached.columns:
        return False
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    dates = pd.to_datetime(cached["trade_date"], errors="coerce").dt.normalize().dropna().drop_duplicates()
    if dates.empty or dates.min() > start or dates.max() < end:
        return False
    actual = set(dates.loc[(dates >= start) & (dates <= end)].tolist())
    try:
        active = read_active_manifest(root)
        expected_dates = _load_open_dates(root=root, active_map=active_dataset_map(active), start_date=start, end_date=end)
    except Exception:
        expected_dates = pd.DatetimeIndex(pd.bdate_range(start, end))
    expected = {pd.Timestamp(item).normalize() for item in expected_dates}
    if not expected:
        return bool(actual)
    return len(actual.intersection(expected)) >= int(len(expected) * 0.98)


def _fetch_benchmark_baostock(*, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    import baostock as bs  # type: ignore

    code = _baostock_code(symbol)
    fields = "date,code,open,high,low,close,volume,amount"
    login = bs.login()
    if getattr(login, "error_code", "") not in {"0", 0}:
        raise RuntimeError(f"baostock_login_failed:{getattr(login, 'error_msg', '')}")
    rows: list[list[str]] = []
    try:
        rs = bs.query_history_k_data_plus(
            code,
            fields,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag="3",
        )
        if getattr(rs, "error_code", "") not in {"0", 0}:
            raise RuntimeError(f"baostock_benchmark_query_failed:{getattr(rs, 'error_msg', '')}")
        while rs.next():
            rows.append(rs.get_row_data())
    finally:
        try:
            bs.logout()
        except Exception:
            pass
    data = pd.DataFrame(rows, columns=fields.split(","))
    if data.empty:
        raise ValueError(f"benchmark_empty:{symbol}:{start_date:%Y-%m-%d}:{end_date:%Y-%m-%d}")
    data = data.rename(columns={"date": "trade_date"})
    data["symbol"] = _normalize_symbol(symbol)
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    for field in ("open", "high", "low", "close", "volume", "amount"):
        data[field] = pd.to_numeric(data[field], errors="coerce")
    return data[["trade_date", "symbol", "open", "high", "low", "close", "volume", "amount"]]


def _zscore_cs(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0).replace([np.inf, -np.inf], np.nan)


def _normalize_symbol(value: str) -> str:
    raw = str(value or "").strip().upper()
    if "." in raw:
        code, suffix = raw.split(".", 1)
        return f"{code.zfill(6)}.{suffix}"
    code = raw.zfill(6)
    if code.startswith(("6", "5", "9", "000")):
        return f"{code}.SH"
    return f"{code}.SZ"


def _baostock_code(symbol: str) -> str:
    normalized = _normalize_symbol(symbol)
    code, suffix = normalized.split(".", 1)
    return f"{suffix.lower()}.{code}"


def _compact_now() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _coerce_config(config: V2TrainingPackConfig | Mapping[str, Any] | None) -> V2TrainingPackConfig:
    if isinstance(config, V2TrainingPackConfig):
        return config.normalized()
    if isinstance(config, Mapping):
        return V2TrainingPackConfig(**dict(config)).normalized()
    return V2TrainingPackConfig().normalized()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp rebuild training-pack", description="Build a research training pack from qdp_v2 active data.")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--feature-profile", default=STYLE_ALPHA_PROFILE)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--symbol-block-size", type=int, default=300)
    parser.add_argument("--max-universe-size", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--lookback-days", type=int, default=252)
    parser.add_argument("--horizon", type=int, default=PATH20_HORIZON)
    parser.add_argument("--cumulative-horizons", default="1,3,5,10,20")
    parser.add_argument("--execution-mode", default="next_open")
    parser.add_argument("--max-feature-columns", type=int, default=384)
    parser.add_argument("--min-lookback-valid-ratio", type=float, default=0.80)
    parser.add_argument("--include-static-context", dest="include_static_context", action="store_true", default=True)
    parser.add_argument("--no-static-context", dest="include_static_context", action="store_false")
    parser.add_argument("--static-context-fields", default="symbol,exchange,industry")
    parser.add_argument("--train-start-year", type=int, default=DEFAULT_TRAIN_START_YEAR)
    parser.add_argument("--train-end-year", type=int, default=DEFAULT_TRAIN_END_YEAR)
    parser.add_argument("--validation-year", type=int, default=DEFAULT_VALIDATION_YEAR)
    parser.add_argument("--test-year", type=int, default=DEFAULT_TEST_YEAR)
    parser.add_argument("--feature-dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--training-pack-stock-chunk-size", type=int, default=96)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--benchmark-source", default="baostock", choices=("baostock", "auto"))
    parser.add_argument("--duckdb-memory-limit", default="12GB")
    parser.add_argument("--duckdb-threads", type=int, default=4)
    parser.add_argument("--no-training-pack", dest="build_training_pack", action="store_false", default=True)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true", default=True)
    resume.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw = list(argv or [])
    if raw and raw[0] == "training-pack":
        raw = raw[1:]
    args = build_arg_parser().parse_args(raw)
    cfg = V2TrainingPackConfig(
        workspace_root=str(args.workspace_root or ""),
        tag=str(args.tag or ""),
        feature_profile=str(args.feature_profile or STYLE_ALPHA_PROFILE),
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        symbol_block_size=int(args.symbol_block_size),
        max_universe_size=int(args.max_universe_size),
        max_shards=int(args.max_shards),
        lookback_days=int(args.lookback_days),
        horizon=int(args.horizon),
        cumulative_horizons=str(args.cumulative_horizons),
        execution_mode=str(args.execution_mode),
        max_feature_columns=int(args.max_feature_columns),
        min_lookback_valid_ratio=float(args.min_lookback_valid_ratio),
        include_static_context=bool(args.include_static_context),
        static_context_fields=str(args.static_context_fields),
        train_start_year=int(args.train_start_year),
        train_end_year=int(args.train_end_year),
        validation_year=int(args.validation_year),
        test_year=int(args.test_year),
        feature_dtype=str(args.feature_dtype),
        training_pack_stock_chunk_size=int(args.training_pack_stock_chunk_size),
        resume=bool(args.resume),
        build_training_pack=bool(args.build_training_pack),
        benchmark=str(args.benchmark),
        benchmark_source=str(args.benchmark_source),
        duckdb_memory_limit=str(args.duckdb_memory_limit),
        duckdb_threads=int(args.duckdb_threads),
        json=bool(args.json),
    )
    payload = build_v2_training_pack(cfg)
    if bool(args.json):
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print(f"status: {payload.get('status')}")
        print(f"sharded_manifest_json: {payload.get('sharded_manifest_json')}")
        if payload.get("training_pack_manifest_json"):
            print(f"training_pack_manifest_json: {payload.get('training_pack_manifest_json')}")
        print(f"feature_count: {payload.get('feature_count')}")
        print(f"sample_count: {payload.get('training_pack_sample_count', 0)}")
    return 0 if str(payload.get("status", "")) in {"completed", "partial"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
