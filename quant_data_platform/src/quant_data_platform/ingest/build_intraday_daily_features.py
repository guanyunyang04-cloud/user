from __future__ import annotations

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from quant_data_platform.lake.canonical import DEFAULT_CANONICAL_START_DATE
from quant_data_platform.lake.catalog import DEFAULT_DATA_LAKE_ROOT, ResearchDataLake
from quant_data_platform.domains.contracts import (
    DataDomain,
    DOMAIN_STANDARD_COLUMNS,
    normalize_intraday_5m_frame,
    normalize_intraday_daily_features_frame,
    normalize_domain,
)


_PROGRESS_LOCK = threading.Lock()
_INTRADAY_5M_REQUIRED_COLUMNS = {
    "symbol",
    "trade_date",
    "bar_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
}
INTRADAY_DAILY_FEATURE_CONTRACT_VERSION = "intraday_daily_features_v3"
LAST_5M_RET_POLICY = "last_close_to_previous_5m_close_return"


@dataclass(frozen=True)
class BuildIntradayDailyFeaturesConfig:
    lake_root: Path = DEFAULT_DATA_LAKE_ROOT
    source_dataset_id: str = ""
    existing_feature_dataset_id: str = ""
    auction_1m_dataset_id: str = ""
    start_date: str = DEFAULT_CANONICAL_START_DATE
    end_date: str = ""
    years: tuple[int, ...] = ()
    source_name: str = "external_1m_derived_5m"
    adjusted_flag: str = "none"
    workers: int = 1
    max_shards: int = 0
    progress_path: Path | None = None
    dry_run: bool = False
    resume: bool = True
    reuse: bool = True

    def normalized(self) -> "BuildIntradayDailyFeaturesConfig":
        return BuildIntradayDailyFeaturesConfig(
            lake_root=Path(self.lake_root),
            source_dataset_id=str(self.source_dataset_id or "").strip(),
            existing_feature_dataset_id=str(self.existing_feature_dataset_id or "").strip(),
            auction_1m_dataset_id=str(self.auction_1m_dataset_id or "").strip(),
            start_date=str(self.start_date or DEFAULT_CANONICAL_START_DATE),
            end_date=str(self.end_date or ""),
            years=tuple(sorted({int(item) for item in self.years})),
            source_name=str(self.source_name or "external_1m_derived_5m").strip().lower(),
            adjusted_flag=str(self.adjusted_flag or "none").strip().lower(),
            workers=max(1, int(self.workers or 1)),
            max_shards=max(0, int(self.max_shards or 0)),
            progress_path=Path(self.progress_path) if self.progress_path else None,
            dry_run=bool(self.dry_run),
            resume=bool(self.resume),
            reuse=bool(self.reuse),
        )


@dataclass(frozen=True)
class BuildIntradayDailyFeaturesResult:
    status: str
    dataset_id: str = ""
    shard_count: int = 0
    row_count: int = 0
    error_count: int = 0
    progress_path: Path | None = None
    errors: list[str] = field(default_factory=list)


def build_intraday_daily_features(config: BuildIntradayDailyFeaturesConfig) -> BuildIntradayDailyFeaturesResult:
    cfg = config.normalized()
    if cfg.existing_feature_dataset_id:
        return _overlay_existing_intraday_daily_features(cfg)
    if not cfg.source_dataset_id:
        raise ValueError("source_dataset_id_required")
    lake = ResearchDataLake(cfg.lake_root)
    source_metadata = lake.describe_dataset(cfg.source_dataset_id)
    source_domain = normalize_domain(str(dict(source_metadata.get("parameters", {}) or {}).get("domain", "") or ""))
    if source_domain != DataDomain.MARKET_INTRADAY_5M:
        raise ValueError(f"source_domain_mismatch: {source_domain}; expected={DataDomain.MARKET_INTRADAY_5M}")
    auction_1m_metadata: dict[str, Any] = {}
    auction_1m_shards: list[dict[str, Any]] = []
    if cfg.auction_1m_dataset_id:
        auction_1m_metadata = lake.describe_dataset(cfg.auction_1m_dataset_id)
        auction_1m_domain = normalize_domain(str(dict(auction_1m_metadata.get("parameters", {}) or {}).get("domain", "") or ""))
        if auction_1m_domain != DataDomain.MARKET_INTRADAY_1M:
            raise ValueError(f"auction_1m_source_domain_mismatch: {auction_1m_domain}; expected={DataDomain.MARKET_INTRADAY_1M}")
        auction_1m_manifest_path = Path(str(dict(auction_1m_metadata.get("content_paths", {}) or {}).get("shard_manifest", "") or ""))
        if not auction_1m_manifest_path.exists():
            raise FileNotFoundError(f"auction_1m_shard_manifest_not_found: {auction_1m_manifest_path}")
        auction_1m_manifest = json.loads(auction_1m_manifest_path.read_text(encoding="utf-8"))
        auction_1m_shards = [
            dict(item)
            for item in list(auction_1m_manifest.get("shards", []) or [])
            if str(dict(item).get("status", "") or "") == "stored" and int(dict(item).get("row_count", 0) or 0) > 0
        ]
    auction_1m_index = _build_auction_1m_shard_index(auction_1m_shards)
    source_manifest_path = Path(str(dict(source_metadata.get("content_paths", {}) or {}).get("shard_manifest", "") or ""))
    if not source_manifest_path.exists():
        raise FileNotFoundError(f"source_shard_manifest_not_found: {source_manifest_path}")

    progress_path = cfg.progress_path or _default_progress_path(lake)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_shards = [
        dict(item)
        for item in list(source_manifest.get("shards", []) or [])
        if str(dict(item).get("status", "") or "") == "stored" and int(dict(item).get("row_count", 0) or 0) > 0
    ]
    if cfg.years:
        allowed_years = set(int(item) for item in cfg.years)
        source_shards = [record for record in source_shards if _shard_intersects_years(record, allowed_years)]
    if cfg.max_shards:
        source_shards = source_shards[: cfg.max_shards]

    target_spec = _feature_dataset_spec(cfg, source_metadata=source_metadata)
    identity_spec = {**target_spec, "domain": DataDomain.INTRADAY_DAILY_FEATURES, "sharded": True}
    identity = lake.build_domain_dataset_identity(domain=DataDomain.INTRADAY_DAILY_FEATURES, spec=identity_spec)
    shard_dir = Path(identity["dataset_dir"]) / "shards"
    _write_progress(
        progress_path,
        {
            "event": "start",
            "source_dataset_id": cfg.source_dataset_id,
            "auction_1m_dataset_id": cfg.auction_1m_dataset_id,
            "source_shard_count": len(source_shards),
            "auction_1m_shard_count": len(auction_1m_shards),
            "workers": cfg.workers,
            "dry_run": cfg.dry_run,
        },
    )
    if cfg.dry_run:
        _write_progress(progress_path, {"event": "completed", "dry_run": True, "shard_count": len(source_shards)})
        return BuildIntradayDailyFeaturesResult(
            status="dry_run",
            shard_count=len(source_shards),
            progress_path=progress_path,
        )

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    if cfg.workers <= 1 or len(source_shards) <= 1:
        for idx, source_record in enumerate(source_shards, start=1):
            records.append(_build_one_shard(lake=lake, cfg=cfg, shard_dir=shard_dir, source_record=source_record, progress_path=progress_path, index=idx, total=len(source_shards), auction_1m_index=auction_1m_index))
    else:
        with ThreadPoolExecutor(max_workers=min(cfg.workers, len(source_shards))) as executor:
            futures = {
                executor.submit(
                    _build_one_shard,
                    lake=lake,
                    cfg=cfg,
                    shard_dir=shard_dir,
                    source_record=source_record,
                    progress_path=progress_path,
                    index=idx,
                    total=len(source_shards),
                    auction_1m_index=auction_1m_index,
                ): source_record
                for idx, source_record in enumerate(source_shards, start=1)
            }
            for future in as_completed(futures):
                try:
                    records.append(future.result())
                except Exception as exc:
                    source_record = futures[future]
                    message = f"{source_record.get('path', '')}: {exc}"
                    errors.append(message)
                    _write_progress(progress_path, {"event": "shard_error", "source_path": str(source_record.get("path", "") or ""), "error": str(exc)})

    records = sorted(records, key=lambda item: str(item.get("path", "")))
    record = lake.save_sharded_domain_dataset(
        domain=DataDomain.INTRADAY_DAILY_FEATURES,
        spec=target_spec,
        shard_records=records,
        source="intraday_daily_features_from_5m",
        reuse=cfg.reuse,
    )
    row_count = sum(int(item.get("row_count", 0) or 0) for item in records)
    error_count = len(errors) + sum(int(item.get("error_count", 0) or 0) for item in records)
    _write_progress(
        progress_path,
        {
            "event": "dataset_registered",
            "dataset_id": record.dataset_id,
            "shard_count": len(records),
            "row_count": row_count,
            "error_count": error_count,
        },
    )
    _write_progress(progress_path, {"event": "completed", "dataset_id": record.dataset_id})
    return BuildIntradayDailyFeaturesResult(
        status="completed",
        dataset_id=record.dataset_id,
        shard_count=len(records),
        row_count=row_count,
        error_count=error_count,
        progress_path=progress_path,
        errors=errors,
    )


def _overlay_existing_intraday_daily_features(cfg: BuildIntradayDailyFeaturesConfig) -> BuildIntradayDailyFeaturesResult:
    if not cfg.auction_1m_dataset_id:
        raise ValueError("auction_1m_dataset_id_required_for_existing_feature_overlay")
    lake = ResearchDataLake(cfg.lake_root)
    feature_metadata = lake.describe_dataset(cfg.existing_feature_dataset_id)
    feature_domain = normalize_domain(str(dict(feature_metadata.get("parameters", {}) or {}).get("domain", "") or ""))
    if feature_domain != DataDomain.INTRADAY_DAILY_FEATURES:
        raise ValueError(f"existing_feature_domain_mismatch: {feature_domain}; expected={DataDomain.INTRADAY_DAILY_FEATURES}")
    feature_manifest_path = Path(str(dict(feature_metadata.get("content_paths", {}) or {}).get("shard_manifest", "") or ""))
    if not feature_manifest_path.exists():
        raise FileNotFoundError(f"existing_feature_shard_manifest_not_found: {feature_manifest_path}")

    auction_1m_metadata = lake.describe_dataset(cfg.auction_1m_dataset_id)
    auction_1m_domain = normalize_domain(str(dict(auction_1m_metadata.get("parameters", {}) or {}).get("domain", "") or ""))
    if auction_1m_domain != DataDomain.MARKET_INTRADAY_1M:
        raise ValueError(f"auction_1m_source_domain_mismatch: {auction_1m_domain}; expected={DataDomain.MARKET_INTRADAY_1M}")
    auction_1m_manifest_path = Path(str(dict(auction_1m_metadata.get("content_paths", {}) or {}).get("shard_manifest", "") or ""))
    if not auction_1m_manifest_path.exists():
        raise FileNotFoundError(f"auction_1m_shard_manifest_not_found: {auction_1m_manifest_path}")
    auction_1m_manifest = json.loads(auction_1m_manifest_path.read_text(encoding="utf-8"))
    auction_1m_shards = [
        dict(item)
        for item in list(auction_1m_manifest.get("shards", []) or [])
        if str(dict(item).get("status", "") or "") == "stored" and int(dict(item).get("row_count", 0) or 0) > 0
    ]
    auction_1m_index = _build_auction_1m_shard_index(auction_1m_shards)

    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    source_shards = [
        dict(item)
        for item in list(feature_manifest.get("shards", []) or [])
        if str(dict(item).get("status", "") or "") in {"stored", "skipped"} and int(dict(item).get("row_count", 0) or 0) > 0
    ]
    if cfg.years:
        allowed_years = set(int(item) for item in cfg.years)
        source_shards = [record for record in source_shards if _shard_intersects_years(record, allowed_years)]
    if cfg.max_shards:
        source_shards = source_shards[: cfg.max_shards]

    progress_path = cfg.progress_path or _default_progress_path(lake)
    target_spec = _feature_overlay_dataset_spec(
        cfg,
        feature_metadata=feature_metadata,
        auction_1m_metadata=auction_1m_metadata,
    )
    identity_spec = {**target_spec, "domain": DataDomain.INTRADAY_DAILY_FEATURES, "sharded": True}
    identity = lake.build_domain_dataset_identity(domain=DataDomain.INTRADAY_DAILY_FEATURES, spec=identity_spec)
    shard_dir = Path(identity["dataset_dir"]) / "shards"
    _write_progress(
        progress_path,
        {
            "event": "start",
            "mode": "existing_feature_1m_auction_overlay",
            "existing_feature_dataset_id": cfg.existing_feature_dataset_id,
            "auction_1m_dataset_id": cfg.auction_1m_dataset_id,
            "source_shard_count": len(source_shards),
            "auction_1m_shard_count": len(auction_1m_shards),
            "workers": cfg.workers,
            "dry_run": cfg.dry_run,
        },
    )
    if cfg.dry_run:
        _write_progress(progress_path, {"event": "completed", "dry_run": True, "shard_count": len(source_shards)})
        return BuildIntradayDailyFeaturesResult(
            status="dry_run",
            shard_count=len(source_shards),
            progress_path=progress_path,
        )

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    if cfg.workers <= 1 or len(source_shards) <= 1:
        for idx, source_record in enumerate(source_shards, start=1):
            records.append(
                _overlay_one_feature_shard(
                    cfg=cfg,
                    shard_dir=shard_dir,
                    source_record=source_record,
                    progress_path=progress_path,
                    index=idx,
                    total=len(source_shards),
                    auction_1m_index=auction_1m_index,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=min(cfg.workers, len(source_shards))) as executor:
            futures = {
                executor.submit(
                    _overlay_one_feature_shard,
                    cfg=cfg,
                    shard_dir=shard_dir,
                    source_record=source_record,
                    progress_path=progress_path,
                    index=idx,
                    total=len(source_shards),
                    auction_1m_index=auction_1m_index,
                ): source_record
                for idx, source_record in enumerate(source_shards, start=1)
            }
            for future in as_completed(futures):
                try:
                    records.append(future.result())
                except Exception as exc:
                    source_record = futures[future]
                    message = f"{source_record.get('path', '')}: {exc}"
                    errors.append(message)
                    _write_progress(progress_path, {"event": "shard_error", "source_path": str(source_record.get("path", "") or ""), "error": str(exc)})

    records = sorted(records, key=lambda item: str(item.get("path", "")))
    record = lake.save_sharded_domain_dataset(
        domain=DataDomain.INTRADAY_DAILY_FEATURES,
        spec=target_spec,
        shard_records=records,
        source="intraday_daily_features_auction_1m_overlay",
        reuse=cfg.reuse,
    )
    row_count = sum(int(item.get("row_count", 0) or 0) for item in records)
    error_count = len(errors) + sum(int(item.get("error_count", 0) or 0) for item in records)
    _write_progress(
        progress_path,
        {
            "event": "dataset_registered",
            "dataset_id": record.dataset_id,
            "shard_count": len(records),
            "row_count": row_count,
            "error_count": error_count,
        },
    )
    _write_progress(progress_path, {"event": "completed", "dataset_id": record.dataset_id})
    return BuildIntradayDailyFeaturesResult(
        status="completed",
        dataset_id=record.dataset_id,
        shard_count=len(records),
        row_count=row_count,
        error_count=error_count,
        progress_path=progress_path,
        errors=errors,
    )


def _overlay_one_feature_shard(
    *,
    cfg: BuildIntradayDailyFeaturesConfig,
    shard_dir: Path,
    source_record: dict[str, Any],
    progress_path: Path,
    index: int,
    total: int,
    auction_1m_index: dict[str, Any],
) -> dict[str, Any]:
    source_path = Path(str(source_record.get("path", "") or ""))
    if not source_path.exists():
        raise FileNotFoundError(f"existing_feature_shard_not_found: {source_path}")
    shard_dir.mkdir(parents=True, exist_ok=True)
    target_path = shard_dir / f"{_safe_stem(source_path.stem)}__auction_1m_overlay.parquet"
    _write_progress(progress_path, {"event": "shard_start", "index": index, "total": total, "source_path": str(source_path.resolve())})
    if cfg.resume and target_path.exists():
        features = pd.read_parquet(target_path, columns=["trade_date", "symbol"])
        return _overlay_feature_shard_record(
            target_path=target_path,
            features=features,
            source_record=source_record,
            source_path=source_path,
            status="stored",
            materialization="resume_hit",
        )
    features = pd.read_parquet(source_path)
    features = _apply_auction_1m_overlay(features, auction_1m_index=auction_1m_index, source_record=source_record)
    features.to_parquet(target_path, index=False)
    record = _overlay_feature_shard_record(
        target_path=target_path,
        features=features,
        source_record=source_record,
        source_path=source_path,
        status="stored",
        materialization="auction_1m_overlay",
    )
    _write_progress(
        progress_path,
        {
            "event": "shard_stored",
            "index": index,
            "total": total,
            "path": str(target_path.resolve()),
            "source_path": str(source_path.resolve()),
            "row_count": record["row_count"],
            "start_date": record["start_date"],
            "end_date": record["end_date"],
        },
    )
    return record


def _build_one_shard(
    *,
    lake: ResearchDataLake,
    cfg: BuildIntradayDailyFeaturesConfig,
    shard_dir: Path,
    source_record: dict[str, Any],
    progress_path: Path,
    index: int,
    total: int,
    auction_1m_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_path = Path(str(source_record.get("path", "") or ""))
    if not source_path.exists():
        raise FileNotFoundError(f"source_shard_not_found: {source_path}")
    shard_dir.mkdir(parents=True, exist_ok=True)
    target_path = shard_dir / f"{_safe_stem(source_path.stem)}__intraday_daily_features.parquet"
    _write_progress(progress_path, {"event": "shard_start", "index": index, "total": total, "source_path": str(source_path.resolve())})
    if cfg.resume and target_path.exists():
        features = pd.read_parquet(target_path, columns=["trade_date", "symbol"])
        return _feature_shard_record(
            target_path=target_path,
            features=features,
            source_record=source_record,
            source_path=source_path,
            status="stored",
            materialization="resume_hit",
        )
    raw = pd.read_parquet(source_path)
    features = _build_intraday_daily_feature_frame_fast(raw, source=cfg.source_name, adjusted_flag=cfg.adjusted_flag)
    if cfg.auction_1m_dataset_id and auction_1m_index:
        features = _apply_auction_1m_overlay(features, auction_1m_index=auction_1m_index, source_record=source_record)
    features.to_parquet(target_path, index=False)
    record = _feature_shard_record(
        target_path=target_path,
        features=features,
        source_record=source_record,
        source_path=source_path,
        status="stored",
        materialization="built",
    )
    _write_progress(
        progress_path,
        {
            "event": "shard_stored",
            "index": index,
            "total": total,
            "path": str(target_path.resolve()),
            "source_path": str(source_path.resolve()),
            "row_count": record["row_count"],
            "start_date": record["start_date"],
            "end_date": record["end_date"],
        },
    )
    return record


def _build_intraday_daily_feature_frame_fast(
    intraday_frame: pd.DataFrame,
    *,
    source: str,
    adjusted_flag: str,
) -> pd.DataFrame:
    bars = _prepare_intraday_5m_bars(intraday_frame, source=source, adjusted_flag=adjusted_flag)
    if bars.empty:
        return pd.DataFrame(columns=DOMAIN_STANDARD_COLUMNS[DataDomain.INTRADAY_DAILY_FEATURES])

    keys = ["trade_date", "symbol"]
    bars = bars.sort_values(keys + ["bar_time"]).reset_index(drop=True)
    numeric_columns = ["open", "high", "low", "close", "volume", "amount"]
    for column in numeric_columns:
        bars[column] = pd.to_numeric(bars[column], errors="coerce").replace([np.inf, -np.inf], np.nan)

    grouped = bars.groupby(keys, sort=True, dropna=False)
    bars["_pos"] = grouped.cumcount()
    bars["_revpos"] = grouped.cumcount(ascending=False)
    bars["_clock"] = pd.to_numeric(bars["bar_time"].astype(str).str[:6], errors="coerce").fillna(0).astype(int)

    base = grouped.agg(
        first_open=("open", "first"),
        last_close=("close", "last"),
        high_max=("high", "max"),
        low_min=("low", "min"),
        total_volume=("volume", "sum"),
        total_amount=("amount", "sum"),
        bar_count=("close", "size"),
    )
    out = pd.DataFrame(index=base.index)
    out["bar_count"] = base["bar_count"].astype(float)
    out["intraday_vwap"] = _safe_ratio(base["total_amount"], base["total_volume"])
    out["intraday_ret"] = _safe_return_series(base["last_close"], base["first_open"])
    out["close_to_vwap"] = _safe_return_series(base["last_close"], out["intraday_vwap"])
    out["intraday_range"] = _safe_return_series(base["high_max"], base["low_min"])
    out["close_position"] = (base["last_close"] - base["low_min"]) / (base["high_max"] - base["low_min"])
    out.loc[(base["high_max"] <= base["low_min"]) | base["last_close"].isna(), "close_position"] = np.nan
    out["open_to_high_ret"] = _safe_return_series(base["high_max"], base["first_open"])
    out["open_to_low_ret"] = _safe_return_series(base["low_min"], base["first_open"])
    out["high_to_close_ret"] = _safe_return_series(base["last_close"], base["high_max"])
    out["low_to_close_ret"] = _safe_return_series(base["last_close"], base["low_min"])

    pos0 = _rows_at(bars, bars["_pos"].eq(0), keys)
    pos2 = _rows_at(bars, bars["_pos"].eq(2), keys)
    pos5 = _rows_at(bars, bars["_pos"].eq(5), keys)
    last = _rows_at(bars, bars["_revpos"].eq(0), keys)
    tail5 = _rows_at(bars, bars["_revpos"].eq(5), keys)
    out["first_5m_ret"] = _safe_return_series(pos0["close"], pos0["open"]).reindex(out.index)
    out["first_15m_ret"] = _safe_return_series(pos2["close"], pos0["open"]).reindex(out.index)
    out["first_30m_ret"] = _safe_return_series(pos5["close"], pos0["open"]).reindex(out.index)
    out["last_5m_ret"] = _last_close_to_previous_close_return(bars, keys).reindex(out.index)
    out["last_30m_ret"] = _safe_return_series(last["close"], tail5["open"]).reindex(out.index)

    head6 = bars.loc[bars["_pos"].lt(6)]
    tail6 = bars.loc[bars["_revpos"].lt(6)]
    head6_amount = head6.groupby(keys, sort=True, dropna=False)["amount"].sum().reindex(out.index)
    tail6_amount = tail6.groupby(keys, sort=True, dropna=False)["amount"].sum().reindex(out.index)
    first_amount = pos0["amount"].reindex(out.index)
    first_volume = pos0["volume"].reindex(out.index)
    last_amount = last["amount"].reindex(out.index)
    last_volume = last["volume"].reindex(out.index)
    out["first_30m_amount_share"] = _safe_ratio(head6_amount, base["total_amount"])
    out["last_30m_amount_share"] = _safe_ratio(tail6_amount, base["total_amount"])
    out["first_5m_amount_share"] = _safe_ratio(first_amount, base["total_amount"])
    out["last_5m_amount_share"] = _safe_ratio(last_amount, base["total_amount"])
    out["opening_auction_ret"] = out["first_5m_ret"]
    out["opening_auction_amount"] = first_amount
    out["opening_auction_volume"] = first_volume
    out["opening_auction_amount_share"] = out["first_5m_amount_share"]
    out["opening_auction_vwap"] = _safe_ratio(first_amount, first_volume)
    out["opening_auction_pressure"] = out["opening_auction_ret"] * out["opening_auction_amount_share"]
    out["closing_auction_ret"] = out["last_5m_ret"]
    out["closing_auction_amount"] = last_amount
    out["closing_auction_volume"] = last_volume
    out["closing_auction_amount_share"] = out["last_5m_amount_share"]
    out["closing_auction_vwap"] = _safe_ratio(last_amount, last_volume)
    out["closing_auction_pressure"] = out["closing_auction_ret"] * out["closing_auction_amount_share"]

    head_range = head6.groupby(keys, sort=True, dropna=False).agg(high=("high", "max"), low=("low", "min")).reindex(out.index)
    tail_range = tail6.groupby(keys, sort=True, dropna=False).agg(high=("high", "max"), low=("low", "min")).reindex(out.index)
    first_range = _safe_return_series(pos0["high"], pos0["low"]).reindex(out.index)
    last_range = _safe_return_series(last["high"], last["low"]).reindex(out.index)
    enough = base["bar_count"].ge(6)
    out["first_30m_range"] = _safe_return_series(head_range["high"], head_range["low"])
    out["last_30m_range"] = _safe_return_series(tail_range["high"], tail_range["low"])
    out["opening_auction_range"] = first_range
    out["closing_auction_range"] = last_range
    out.loc[~enough, ["first_30m_range", "last_30m_range"]] = np.nan

    prev_close = base["last_close"].groupby(level="symbol", sort=False).shift(1)
    out["open_gap"] = _safe_return_series(base["first_open"], prev_close)
    gap_sign = np.sign(out["open_gap"])
    out["open_gap_first_30m_follow_through"] = gap_sign * out["first_30m_ret"]
    out["open_gap_first_30m_reversal"] = -gap_sign * out["first_30m_ret"]
    out.loc[out["open_gap"].isna() | out["first_30m_ret"].isna(), ["open_gap_first_30m_follow_through", "open_gap_first_30m_reversal"]] = np.nan

    close_ret = grouped["close"].pct_change().replace([np.inf, -np.inf], np.nan)
    bars["_close_ret"] = close_ret
    out["intraday_realized_vol"] = close_ret.groupby([bars["trade_date"], bars["symbol"]], sort=True, dropna=False).std().reindex(out.index)
    out["intraday_price_volume_corr"] = _group_corr(bars, keys, "_close_ret", "volume").reindex(out.index)

    valid_high = bars["high"].notna()
    valid_low = bars["low"].notna()
    high_pos = _extreme_pos(bars.loc[valid_high], keys, "high", "max").reindex(out.index)
    low_pos = _extreme_pos(bars.loc[valid_low], keys, "low", "min").reindex(out.index)
    denom = (base["bar_count"] - 1).replace(0, np.nan)
    out["high_time_frac"] = high_pos / denom
    out["low_time_frac"] = low_pos / denom
    out.loc[base["bar_count"].eq(1) & high_pos.notna(), "high_time_frac"] = 0.0
    out.loc[base["bar_count"].eq(1) & low_pos.notna(), "low_time_frac"] = 0.0
    out["high_before_low"] = np.where(
        high_pos.notna() & low_pos.notna(),
        np.where(high_pos.eq(low_pos), 0.5, (high_pos < low_pos).astype(float)),
        np.nan,
    )

    valid_close_count = grouped["close"].count().reindex(out.index)
    running_max = grouped["close"].cummax()
    running_min = grouped["close"].cummin()
    drawdown = bars["close"] / running_max - 1.0
    runup = bars["close"] / running_min - 1.0
    out["intraday_max_drawdown"] = drawdown.groupby([bars["trade_date"], bars["symbol"]], sort=True, dropna=False).min().reindex(out.index)
    out["intraday_max_runup"] = runup.groupby([bars["trade_date"], bars["symbol"]], sort=True, dropna=False).max().reindex(out.index)
    out.loc[valid_close_count.lt(2), ["intraday_max_drawdown", "intraday_max_runup"]] = np.nan

    vwap_by_row = out["intraday_vwap"].reindex(pd.MultiIndex.from_frame(bars[keys])).to_numpy()
    above = pd.Series(bars["close"].to_numpy() > vwap_by_row, index=bars.index)
    out["price_above_vwap_share"] = above.groupby([bars["trade_date"], bars["symbol"]], sort=True, dropna=False).mean().reindex(out.index)
    out.loc[out["intraday_vwap"].isna(), "price_above_vwap_share"] = np.nan

    amount_share = _safe_ratio(bars["amount"], base["total_amount"].reindex(pd.MultiIndex.from_frame(bars[keys])).to_numpy())
    bars["_amount_share"] = amount_share.to_numpy() if isinstance(amount_share, pd.Series) else amount_share
    out["amount_top_bar_share"] = bars["_amount_share"].groupby([bars["trade_date"], bars["symbol"]], sort=True, dropna=False).max().reindex(out.index)
    out["amount_concentration_hhi"] = (bars["_amount_share"] ** 2).groupby([bars["trade_date"], bars["symbol"]], sort=True, dropna=False).sum().reindex(out.index)
    out.loc[base["total_amount"].le(0) | base["total_amount"].isna(), ["amount_top_bar_share", "amount_concentration_hhi"]] = np.nan

    cum_amount = grouped["amount"].cumsum()
    cum_volume = grouped["volume"].cumsum()
    bars["_cum_vwap"] = cum_amount / cum_volume.where(cum_volume > 0)
    out["cum_vwap_slope"] = _linear_slope_by_group(bars, keys, "_cum_vwap").reindex(out.index)

    am = bars.loc[bars["_clock"].le(113000)]
    pm = bars.loc[bars["_clock"].ge(130000)]
    am_first = _rows_at(am.assign(_session_pos=am.groupby(keys, sort=True, dropna=False).cumcount()), am.groupby(keys, sort=True, dropna=False).cumcount().eq(0), keys)
    am_last = _rows_at(am.assign(_session_revpos=am.groupby(keys, sort=True, dropna=False).cumcount(ascending=False)), am.groupby(keys, sort=True, dropna=False).cumcount(ascending=False).eq(0), keys)
    pm_first = _rows_at(pm.assign(_session_pos=pm.groupby(keys, sort=True, dropna=False).cumcount()), pm.groupby(keys, sort=True, dropna=False).cumcount().eq(0), keys)
    pm_last = _rows_at(pm.assign(_session_revpos=pm.groupby(keys, sort=True, dropna=False).cumcount(ascending=False)), pm.groupby(keys, sort=True, dropna=False).cumcount(ascending=False).eq(0), keys)
    out["am_ret"] = _safe_return_series(am_last["close"], am_first["open"]).reindex(out.index)
    out["pm_ret"] = _safe_return_series(pm_last["close"], pm_first["open"]).reindex(out.index)
    out["lunch_gap_ret"] = _safe_return_series(pm_first["open"], am_last["close"]).reindex(out.index)
    out["am_pm_ret_spread"] = out["pm_ret"] - out["am_ret"]
    am_close_ret = am.groupby(keys, sort=True, dropna=False)["close"].pct_change().replace([np.inf, -np.inf], np.nan)
    pm_close_ret = pm.groupby(keys, sort=True, dropna=False)["close"].pct_change().replace([np.inf, -np.inf], np.nan)
    am_vol = am_close_ret.groupby([am["trade_date"], am["symbol"]], sort=True, dropna=False).std().reindex(out.index)
    pm_vol = pm_close_ret.groupby([pm["trade_date"], pm["symbol"]], sort=True, dropna=False).std().reindex(out.index)
    out["am_pm_vol_spread"] = pm_vol - am_vol
    am_amount = am.groupby(keys, sort=True, dropna=False)["amount"].sum().reindex(out.index)
    out["am_amount_share"] = _safe_ratio(am_amount, base["total_amount"])
    pm_amount_share = 1.0 - out["am_amount_share"]
    out["am_pm_amount_spread"] = out["am_amount_share"] - pm_amount_share

    out["early_strength_late_weak"] = out["first_30m_ret"] - out["last_30m_ret"]
    out["close_pressure_30m"] = out["last_30m_ret"] * out["last_30m_amount_share"]
    out["source"] = source
    out["adjusted_flag"] = str(adjusted_flag or "none")
    out = out.reset_index()
    return normalize_intraday_daily_features_frame(out, source=source, adjusted_flag=adjusted_flag, require_columns=False)


def _apply_auction_1m_overlay(
    features: pd.DataFrame,
    *,
    auction_1m_index: dict[str, Any],
    source_record: dict[str, Any],
) -> pd.DataFrame:
    if features is None or features.empty or not auction_1m_index:
        return features
    if not {"trade_date", "symbol"}.issubset(features.columns):
        return features
    feature_dates = pd.to_datetime(features["trade_date"], errors="coerce").dropna()
    if feature_dates.empty:
        return features
    start_date = feature_dates.min().strftime("%Y-%m-%d")
    end_date = feature_dates.max().strftime("%Y-%m-%d")
    feature_symbols = set(features["symbol"].astype(str).str.strip().str.upper())
    auction_1m_shards = _matching_auction_1m_shards(
        source_record=source_record,
        auction_1m_index=auction_1m_index,
        start_date=start_date,
        end_date=end_date,
        feature_symbols=feature_symbols,
    )
    overlay = _auction_1m_overlay_frame_duckdb(
        auction_1m_shards,
        start_date=start_date,
        end_date=end_date,
        feature_symbols=feature_symbols,
    )
    if overlay.empty:
        return _apply_auction_1m_overlay_pandas_slow(
            features,
            auction_1m_shards=auction_1m_shards,
            start_date=start_date,
            end_date=end_date,
            feature_symbols=feature_symbols,
        )
    overlay = overlay.drop_duplicates(["trade_date", "symbol"], keep="last").set_index(["trade_date", "symbol"])
    out = features.copy()
    key_index = pd.MultiIndex.from_frame(out[["trade_date", "symbol"]].astype(str))
    for column in overlay.columns:
        values = overlay[column].reindex(key_index).to_numpy()
        mask = pd.notna(values)
        if column not in out.columns:
            out[column] = np.nan
        out.loc[mask, column] = values[mask]
    return normalize_intraday_daily_features_frame(out, source=str(out["source"].iloc[0] if "source" in out.columns and len(out) else "intraday_daily_features"), adjusted_flag=str(out["adjusted_flag"].iloc[0] if "adjusted_flag" in out.columns and len(out) else "none"), require_columns=False)


def _auction_1m_overlay_frame_duckdb(
    auction_1m_shards: Sequence[dict[str, Any]],
    *,
    start_date: str,
    end_date: str,
    feature_symbols: set[str],
) -> pd.DataFrame:
    paths = [str(record.get("path", "") or "") for record in auction_1m_shards if Path(str(record.get("path", "") or "")).exists()]
    if not paths:
        return pd.DataFrame()
    try:
        import duckdb
    except Exception:
        return pd.DataFrame()
    symbols = sorted(str(symbol).strip().upper() for symbol in feature_symbols if str(symbol).strip())
    if not symbols:
        return pd.DataFrame()
    sql = """
    with filtered as (
      select
        cast(trade_date as varchar)[:10] as trade_date,
        upper(cast(symbol as varchar)) as symbol,
        lpad(replace(replace(cast(bar_time as varchar), ':', ''), '.', ''), 9, '0') as bar_time,
        try_cast(open as double) as open,
        try_cast(high as double) as high,
        try_cast(low as double) as low,
        try_cast(close as double) as close,
        try_cast(volume as double) as volume,
        try_cast(amount as double) as amount
      from read_parquet(?, union_by_name=true)
      where cast(trade_date as date) >= cast(? as date)
        and cast(trade_date as date) <= cast(? as date)
        and upper(cast(symbol as varchar)) in (select upper(x) from unnest(?) as t(x))
    ),
    totals as (
      select trade_date, symbol, sum(amount) as total_amount
      from filtered
      group by trade_date, symbol
    ),
    opening as (
      select distinct on (trade_date, symbol)
        trade_date, symbol, open, high, low, close, volume, amount
      from filtered
      where bar_time = '093100000'
      order by trade_date, symbol
    ),
    closing as (
      select distinct on (trade_date, symbol)
        trade_date, symbol, open, high, low, close, volume, amount
      from filtered
      where bar_time = '150000000'
      order by trade_date, symbol
    ),
    keys as (
      select trade_date, symbol from opening
      union
      select trade_date, symbol from closing
    )
    select
      keys.trade_date,
      keys.symbol,
      case when opening.open > 0 and opening.close is not null then opening.close / opening.open - 1.0 else null end as opening_auction_ret,
      opening.amount as opening_auction_amount,
      opening.volume as opening_auction_volume,
      case when totals.total_amount > 0 then opening.amount / totals.total_amount else null end as opening_auction_amount_share,
      case when opening.low > 0 and opening.high is not null then opening.high / opening.low - 1.0 else null end as opening_auction_range,
      case when opening.volume > 0 then opening.amount / opening.volume else null end as opening_auction_vwap,
      case
        when opening.open > 0 and opening.close is not null and totals.total_amount > 0
        then (opening.close / opening.open - 1.0) * (opening.amount / totals.total_amount)
        else null
      end as opening_auction_pressure,
      case when closing.open > 0 and closing.close is not null then closing.close / closing.open - 1.0 else null end as closing_auction_ret,
      closing.amount as closing_auction_amount,
      closing.volume as closing_auction_volume,
      case when totals.total_amount > 0 then closing.amount / totals.total_amount else null end as closing_auction_amount_share,
      case when closing.low > 0 and closing.high is not null then closing.high / closing.low - 1.0 else null end as closing_auction_range,
      case when closing.volume > 0 then closing.amount / closing.volume else null end as closing_auction_vwap,
      case
        when closing.open > 0 and closing.close is not null and totals.total_amount > 0
        then (closing.close / closing.open - 1.0) * (closing.amount / totals.total_amount)
        else null
      end as closing_auction_pressure
    from keys
    left join totals using (trade_date, symbol)
    left join opening using (trade_date, symbol)
    left join closing using (trade_date, symbol)
    """
    try:
        con = duckdb.connect(":memory:")
        try:
            con.execute("set memory_limit='2GB'")
            con.execute("set threads=1")
        except Exception:
            pass
        return con.execute(sql, [paths, start_date, end_date, symbols]).fetchdf()
    except Exception:
        return pd.DataFrame()


def _apply_auction_1m_overlay_pandas_slow(
    features: pd.DataFrame,
    *,
    auction_1m_shards: Sequence[dict[str, Any]],
    start_date: str,
    end_date: str,
    feature_symbols: set[str],
) -> pd.DataFrame:
    overlay_frames: list[pd.DataFrame] = []
    for record in auction_1m_shards:
        shard_start = str(record.get("start_date", "") or "")
        shard_end = str(record.get("end_date", "") or "")
        if shard_start and shard_end:
            if pd.Timestamp(shard_end) < pd.Timestamp(start_date) or pd.Timestamp(shard_start) > pd.Timestamp(end_date):
                continue
        path = Path(str(record.get("path", "") or ""))
        if not path.exists():
            continue
        try:
            raw = pd.read_parquet(path, columns=["trade_date", "symbol", "bar_time", "open", "high", "low", "close", "volume", "amount"])
        except Exception:
            continue
        raw_dates = pd.to_datetime(raw["trade_date"], errors="coerce")
        raw = raw.loc[(raw_dates >= pd.Timestamp(start_date)) & (raw_dates <= pd.Timestamp(end_date))].copy()
        if raw.empty:
            continue
        raw["trade_date"] = raw_dates.loc[raw.index].dt.strftime("%Y-%m-%d").to_numpy()
        raw["symbol"] = raw["symbol"].astype(str).str.strip().str.upper()
        if feature_symbols:
            raw = raw.loc[raw["symbol"].isin(feature_symbols)].copy()
            if raw.empty:
                continue
        overlay = _auction_1m_overlay_frame(raw)
        if not overlay.empty:
            overlay_frames.append(overlay)
    if not overlay_frames:
        return features
    overlay = pd.concat(overlay_frames, ignore_index=True)
    overlay = overlay.drop_duplicates(["trade_date", "symbol"], keep="last").set_index(["trade_date", "symbol"])
    out = features.copy()
    key_index = pd.MultiIndex.from_frame(out[["trade_date", "symbol"]].astype(str))
    for column in overlay.columns:
        values = overlay[column].reindex(key_index).to_numpy()
        mask = pd.notna(values)
        if column not in out.columns:
            out[column] = np.nan
        out.loc[mask, column] = values[mask]
    return normalize_intraday_daily_features_frame(out, source=str(out["source"].iloc[0] if "source" in out.columns and len(out) else "intraday_daily_features"), adjusted_flag=str(out["adjusted_flag"].iloc[0] if "adjusted_flag" in out.columns and len(out) else "none"), require_columns=False)


def _build_auction_1m_shard_index(shards: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_source_member: dict[str, list[dict[str, Any]]] = {}
    by_chunk_suffix: dict[str, list[dict[str, Any]]] = {}
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    by_year: dict[int, list[dict[str, Any]]] = {}
    ranged: list[tuple[int, int, dict[str, Any]]] = []
    all_shards: list[dict[str, Any]] = []
    for raw in shards:
        record = dict(raw)
        all_shards.append(record)
        source_member = _one_minute_source_member_key(str(record.get("source_member", "") or ""))
        if source_member:
            by_source_member.setdefault(source_member, []).append(record)
        chunk_suffix = _chunk_suffix_key(record)
        if chunk_suffix:
            by_chunk_suffix.setdefault(chunk_suffix, []).append(record)
        chunk_range = _chunk_ordinal_range(record)
        if chunk_range:
            ranged.append((chunk_range[0], chunk_range[1], record))
        for symbol in _sample_symbols_for_record(record):
            by_symbol.setdefault(symbol, []).append(record)
        for year in _years_for_record(record):
            by_year.setdefault(year, []).append(record)
    return {
        "all": all_shards,
        "by_source_member": by_source_member,
        "by_chunk_suffix": by_chunk_suffix,
        "by_symbol": by_symbol,
        "ranged": ranged,
        "by_year": by_year,
    }


def _matching_auction_1m_shards(
    *,
    source_record: dict[str, Any],
    auction_1m_index: dict[str, Any],
    start_date: str,
    end_date: str,
    feature_symbols: set[str] | None = None,
) -> list[dict[str, Any]]:
    source_member = _one_minute_source_member_key(str(source_record.get("source_member", "") or ""))
    if source_member:
        matches = list(dict(auction_1m_index.get("by_source_member", {}) or {}).get(source_member, []) or [])
        if matches:
            return _date_overlapping_shards(matches, start_date=start_date, end_date=end_date)
    chunk_suffix = _chunk_suffix_key(source_record)
    if chunk_suffix:
        matches = list(dict(auction_1m_index.get("by_chunk_suffix", {}) or {}).get(chunk_suffix, []) or [])
        if matches:
            return _date_overlapping_shards(matches, start_date=start_date, end_date=end_date)
    source_range = _chunk_ordinal_range(source_record)
    if source_range:
        matches = [
            record
            for start, end, record in list(auction_1m_index.get("ranged", []) or [])
            if start <= source_range[1] and end >= source_range[0]
        ]
        if matches:
            return _date_overlapping_shards(matches, start_date=start_date, end_date=end_date)
    if feature_symbols:
        matches = []
        seen_symbol_paths: set[str] = set()
        by_symbol = dict(auction_1m_index.get("by_symbol", {}) or {})
        for symbol in sorted(feature_symbols):
            for record in list(by_symbol.get(symbol, []) or []):
                path = str(record.get("path", "") or "")
                if path in seen_symbol_paths:
                    continue
                seen_symbol_paths.add(path)
                matches.append(record)
        if matches:
            overlapping = _date_overlapping_shards(matches, start_date=start_date, end_date=end_date)
            if _records_have_complete_symbol_coverage(overlapping, feature_symbols):
                return overlapping
    start_year = int(pd.Timestamp(start_date).year)
    end_year = int(pd.Timestamp(end_date).year)
    years = list(range(start_year, end_year + 1))
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    by_year = dict(auction_1m_index.get("by_year", {}) or {})
    for year in sorted(set(years)):
        for record in list(by_year.get(year, []) or []):
            key = str(record.get("path", "") or "")
            if key in seen:
                continue
            seen.add(key)
            candidates.append(record)
    if not candidates:
        candidates = list(auction_1m_index.get("all", []) or [])
    return _date_overlapping_shards(candidates, start_date=start_date, end_date=end_date)


def _records_have_complete_symbol_coverage(records: Sequence[dict[str, Any]], feature_symbols: set[str]) -> bool:
    requested = {str(symbol).strip().upper() for symbol in feature_symbols if str(symbol).strip()}
    if not requested or not records:
        return False
    covered: set[str] = set()
    for record in records:
        sample_symbols = _sample_symbols_for_record(record)
        if not sample_symbols:
            return False
        member_count = int(record.get("source_member_count", 0) or record.get("symbol_count", 0) or 0)
        if member_count and len(sample_symbols) < member_count:
            return False
        covered.update(sample_symbols)
    return requested.issubset(covered)


def _one_minute_source_member_key(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.replace("market_intraday_5m", "market_intraday_1m").replace("_derived_5m", "")


def _chunk_suffix_key(record: dict[str, Any]) -> str:
    for raw in (
        record.get("chunk_id", ""),
        record.get("source_raw_chunk_id", ""),
        Path(str(record.get("source_raw_path", "") or "")).stem,
        Path(str(record.get("path", "") or "")).stem,
    ):
        text = str(raw or "")
        match = re.search(r"__(s\d+_\d+_[0-9a-f]{10})", text)
        if match:
            return match.group(1)
    return ""


def _chunk_ordinal_range(record: dict[str, Any]) -> tuple[int, int] | None:
    for raw in (
        record.get("source_raw_chunk_id", ""),
        Path(str(record.get("source_raw_path", "") or "")).stem,
        record.get("chunk_id", ""),
        Path(str(record.get("path", "") or "")).stem,
    ):
        text = str(raw or "")
        match = re.search(r"__s(\d+)_(\d+)_", text)
        if match:
            index = int(match.group(1))
            count = int(match.group(2))
            if index <= 0 or count <= 0:
                return None
            stride = _chunk_ordinal_stride(text, count)
            start = (index - 1) * stride + 1
            return start, start + count - 1
    return None


def _chunk_ordinal_stride(text: str, count: int) -> int:
    lower = str(text or "").lower()
    if "market_intraday_1m" in lower:
        return max(int(count), 50)
    if "market_intraday_5m" in lower:
        return max(int(count), 100)
    return int(count)


def _sample_symbols_for_record(record: dict[str, Any]) -> set[str]:
    raw_values = list(record.get("source_members_sample", []) or [])
    out: set[str] = set()
    for raw in raw_values:
        symbol = _symbol_from_member_sample(str(raw or ""))
        if symbol:
            out.add(symbol)
    return out


def _symbol_from_member_sample(value: str) -> str:
    name = Path(str(value or "").replace("\\", "/")).name.lower()
    stem = name.rsplit(".", 1)[0]
    match = re.search(r"(sh|sz|bj)(\d{6})", stem)
    if not match:
        return ""
    prefix, code = match.group(1), match.group(2)
    suffix = {"sh": "SH", "sz": "SZ", "bj": "BJ"}.get(prefix, "")
    return f"{code}.{suffix}" if suffix else ""


def _years_for_record(record: dict[str, Any]) -> set[int]:
    out: set[int] = set()
    for key in ("start_date", "end_date"):
        value = str(record.get(key, "") or "")
        if value:
            try:
                out.add(int(pd.Timestamp(value).year))
            except Exception:
                pass
    return out


def _date_overlapping_shards(shards: Sequence[dict[str, Any]], *, start_date: str, end_date: str) -> list[dict[str, Any]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    out: list[dict[str, Any]] = []
    for record in shards:
        shard_start = str(record.get("start_date", "") or "")
        shard_end = str(record.get("end_date", "") or "")
        if shard_start and shard_end:
            if pd.Timestamp(shard_end) < start or pd.Timestamp(shard_start) > end:
                continue
        out.append(dict(record))
    return out


def _auction_1m_overlay_frame(raw_1m: pd.DataFrame) -> pd.DataFrame:
    if raw_1m is None or raw_1m.empty:
        return pd.DataFrame()
    data = raw_1m.copy()
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["trade_date"] = data["trade_date"].astype(str).str.slice(0, 10)
    data["symbol"] = data["symbol"].astype(str).str.strip().str.upper()
    data["bar_time"] = _fast_bar_time(data["bar_time"])
    total_amount = data.groupby(["trade_date", "symbol"], sort=True, dropna=False)["amount"].sum()
    opening = data.loc[data["bar_time"].eq("093100000")].drop_duplicates(["trade_date", "symbol"], keep="last").set_index(["trade_date", "symbol"])
    closing = data.loc[data["bar_time"].eq("150000000")].drop_duplicates(["trade_date", "symbol"], keep="last").set_index(["trade_date", "symbol"])
    if opening.empty and closing.empty:
        return pd.DataFrame()
    index = opening.index.union(closing.index)
    out = pd.DataFrame(index=index)
    if not opening.empty:
        opening_amount = opening["amount"].reindex(index)
        opening_volume = opening["volume"].reindex(index)
        out["opening_auction_ret"] = _safe_return_series(opening["close"].reindex(index), opening["open"].reindex(index))
        out["opening_auction_amount"] = opening_amount
        out["opening_auction_volume"] = opening_volume
        out["opening_auction_amount_share"] = _safe_ratio(opening_amount, total_amount.reindex(index))
        out["opening_auction_range"] = _safe_return_series(opening["high"].reindex(index), opening["low"].reindex(index))
        out["opening_auction_vwap"] = _safe_ratio(opening_amount, opening_volume)
        out["opening_auction_pressure"] = out["opening_auction_ret"] * out["opening_auction_amount_share"]
    if not closing.empty:
        closing_amount = closing["amount"].reindex(index)
        closing_volume = closing["volume"].reindex(index)
        out["closing_auction_ret"] = _safe_return_series(closing["close"].reindex(index), closing["open"].reindex(index))
        out["closing_auction_amount"] = closing_amount
        out["closing_auction_volume"] = closing_volume
        out["closing_auction_amount_share"] = _safe_ratio(closing_amount, total_amount.reindex(index))
        out["closing_auction_range"] = _safe_return_series(closing["high"].reindex(index), closing["low"].reindex(index))
        out["closing_auction_vwap"] = _safe_ratio(closing_amount, closing_volume)
        out["closing_auction_pressure"] = out["closing_auction_ret"] * out["closing_auction_amount_share"]
    return out.reset_index()


def _prepare_intraday_5m_bars(intraday_frame: pd.DataFrame, *, source: str, adjusted_flag: str) -> pd.DataFrame:
    if intraday_frame is None or intraday_frame.empty:
        return pd.DataFrame(columns=list(_INTRADAY_5M_REQUIRED_COLUMNS) + ["source", "adjusted_flag"])
    if not _INTRADAY_5M_REQUIRED_COLUMNS.issubset(set(intraday_frame.columns)):
        return normalize_intraday_5m_frame(
            intraday_frame,
            source=source,
            adjusted_flag=adjusted_flag,
            require_columns=False,
        )

    columns = [
        "symbol",
        "trade_date",
        "bar_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    data = intraday_frame.loc[:, columns].copy()
    data["symbol"] = data["symbol"].astype(str).str.strip().str.upper()
    data["trade_date"] = data["trade_date"].astype(str).str.strip().str.slice(0, 10)
    data["bar_time"] = _fast_bar_time(data["bar_time"])
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["source"] = source
    data["adjusted_flag"] = str(adjusted_flag or "none")
    valid = (
        data["symbol"].str.len().gt(0)
        & data["trade_date"].str.lower().ne("nat")
        & data["trade_date"].str.lower().ne("nan")
        & data["bar_time"].str.len().gt(0)
    )
    return data.loc[valid].drop_duplicates(["symbol", "trade_date", "bar_time", "source"]).reset_index(drop=True)


def _fast_bar_time(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    has_colon = text.str.contains(":", regex=False, na=False)
    if bool(has_colon.any()):
        text = text.str.replace(":", "", regex=False)
    text = text.str.replace(".", "", regex=False)
    text = text.str.slice(0, 9)
    short = text.str.len().eq(6)
    if bool(short.any()):
        text.loc[short] = text.loc[short] + "000"
    return text.str.zfill(9)


def _rows_at(frame: pd.DataFrame, mask: pd.Series, keys: list[str]) -> pd.DataFrame:
    value_columns = ["open", "high", "low", "close", "volume", "amount", "_pos"]
    if frame.empty:
        return pd.DataFrame(columns=keys + value_columns).set_index(keys)
    subset = frame.loc[mask, keys + value_columns].drop_duplicates(keys, keep="first")
    return subset.set_index(keys)


def _safe_ratio(numerator: Any, denominator: Any) -> pd.Series:
    num = pd.Series(numerator, copy=False)
    den = pd.Series(denominator, index=num.index if not isinstance(denominator, pd.Series) else denominator.index, copy=False)
    if isinstance(numerator, pd.Series) and isinstance(denominator, pd.Series):
        den = denominator.reindex(numerator.index)
    out = pd.to_numeric(num, errors="coerce") / pd.to_numeric(den, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out.loc[pd.to_numeric(den, errors="coerce").le(0) | pd.to_numeric(den, errors="coerce").isna()] = np.nan
    return out


def _safe_return_series(close_value: Any, open_value: Any) -> pd.Series:
    close = pd.Series(close_value, copy=False)
    open_ = pd.Series(open_value, index=close.index if not isinstance(open_value, pd.Series) else open_value.index, copy=False)
    if isinstance(close_value, pd.Series) and isinstance(open_value, pd.Series):
        open_ = open_value.reindex(close_value.index)
    out = pd.to_numeric(close, errors="coerce") / pd.to_numeric(open_, errors="coerce") - 1.0
    out = out.replace([np.inf, -np.inf], np.nan)
    out.loc[pd.to_numeric(open_, errors="coerce").le(0) | pd.to_numeric(open_, errors="coerce").isna() | pd.to_numeric(close, errors="coerce").isna()] = np.nan
    return out


def _last_close_to_previous_close_return(frame: pd.DataFrame, keys: list[str]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    data = frame.loc[:, [*keys, "close"]].copy()
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data["_previous_close"] = data.groupby(keys, sort=True, dropna=False)["close"].shift(1)
    last_rows = data.groupby(keys, sort=True, dropna=False).tail(1).set_index(keys)
    return _safe_return_series(last_rows["close"], last_rows["_previous_close"])


def _extreme_pos(frame: pd.DataFrame, keys: list[str], column: str, mode: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    idx = frame.groupby(keys, sort=True, dropna=False)[column].idxmax() if mode == "max" else frame.groupby(keys, sort=True, dropna=False)[column].idxmin()
    return frame.loc[idx.dropna().astype(int), keys + ["_pos"]].drop_duplicates(keys, keep="first").set_index(keys)["_pos"].astype(float)


def _group_corr(frame: pd.DataFrame, keys: list[str], x_col: str, y_col: str) -> pd.Series:
    valid = frame[[*keys, x_col, y_col]].copy()
    valid[x_col] = pd.to_numeric(valid[x_col], errors="coerce")
    valid[y_col] = pd.to_numeric(valid[y_col], errors="coerce")
    valid = valid.loc[valid[x_col].notna() & valid[y_col].notna()]
    if valid.empty:
        return pd.Series(dtype=float)
    valid["_xy"] = valid[x_col] * valid[y_col]
    valid["_x2"] = valid[x_col] * valid[x_col]
    valid["_y2"] = valid[y_col] * valid[y_col]
    agg = valid.groupby(keys, sort=True, dropna=False).agg(
        n=(x_col, "size"),
        sx=(x_col, "sum"),
        sy=(y_col, "sum"),
        sxy=("_xy", "sum"),
        sx2=("_x2", "sum"),
        sy2=("_y2", "sum"),
    )
    num = agg["n"] * agg["sxy"] - agg["sx"] * agg["sy"]
    den_sq = (agg["n"] * agg["sx2"] - agg["sx"] ** 2) * (agg["n"] * agg["sy2"] - agg["sy"] ** 2)
    den = np.sqrt(den_sq.where(den_sq > 0))
    corr = num / den
    corr = corr.replace([np.inf, -np.inf], np.nan)
    corr.loc[agg["n"].lt(2)] = np.nan
    return corr


def _linear_slope_by_group(frame: pd.DataFrame, keys: list[str], column: str) -> pd.Series:
    valid = frame.loc[pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).notna(), keys + [column]].copy()
    if valid.empty:
        return pd.Series(dtype=float)
    grouped = valid.groupby(keys, sort=True, dropna=False)
    valid["_x"] = grouped.cumcount().astype(float)
    first = grouped[column].transform("first").astype(float)
    scale = first.abs().where(first.ne(0.0), 1.0)
    valid["_y"] = valid[column].astype(float) / scale
    valid["_xy"] = valid["_x"] * valid["_y"]
    valid["_x2"] = valid["_x"] * valid["_x"]
    agg = valid.groupby(keys, sort=True, dropna=False).agg(
        n=("_y", "size"),
        sx=("_x", "sum"),
        sy=("_y", "sum"),
        sxy=("_xy", "sum"),
        sx2=("_x2", "sum"),
    )
    denom = agg["n"] * agg["sx2"] - agg["sx"] ** 2
    slope = (agg["n"] * agg["sxy"] - agg["sx"] * agg["sy"]) / denom
    slope = slope.replace([np.inf, -np.inf], np.nan)
    slope.loc[agg["n"].lt(2) | denom.eq(0)] = np.nan
    return slope


def _feature_shard_record(
    *,
    target_path: Path,
    features: pd.DataFrame,
    source_record: dict[str, Any],
    source_path: Path,
    status: str,
    materialization: str,
) -> dict[str, Any]:
    starts = pd.to_datetime(features["trade_date"], errors="coerce") if "trade_date" in features.columns else pd.Series(dtype="datetime64[ns]")
    return {
        "domain": DataDomain.INTRADAY_DAILY_FEATURES,
        "status": status,
        "path": str(target_path.resolve()),
        "row_count": int(len(features)),
        "start_date": starts.min().strftime("%Y-%m-%d") if starts.notna().any() else str(source_record.get("start_date", "") or ""),
        "end_date": starts.max().strftime("%Y-%m-%d") if starts.notna().any() else str(source_record.get("end_date", "") or ""),
        "symbol_count": int(features["symbol"].nunique()) if "symbol" in features.columns and not features.empty else 0,
        "source_raw_path": str(source_path.resolve()),
        "source_raw_row_count": int(source_record.get("row_count", 0) or 0),
        "source_zip": str(source_record.get("source_zip", "") or ""),
        "source_member": str(source_record.get("source_member", "") or ""),
        "error_count": 0,
        "error": "",
        "materialization": materialization,
        "auction_process_policy": "unobservable_use_open_as_opening_result_only",
    }


def _overlay_feature_shard_record(
    *,
    target_path: Path,
    features: pd.DataFrame,
    source_record: dict[str, Any],
    source_path: Path,
    status: str,
    materialization: str,
) -> dict[str, Any]:
    starts = pd.to_datetime(features["trade_date"], errors="coerce") if "trade_date" in features.columns else pd.Series(dtype="datetime64[ns]")
    return {
        "domain": DataDomain.INTRADAY_DAILY_FEATURES,
        "status": status,
        "path": str(target_path.resolve()),
        "row_count": int(len(features)),
        "start_date": starts.min().strftime("%Y-%m-%d") if starts.notna().any() else str(source_record.get("start_date", "") or ""),
        "end_date": starts.max().strftime("%Y-%m-%d") if starts.notna().any() else str(source_record.get("end_date", "") or ""),
        "symbol_count": int(features["symbol"].nunique()) if "symbol" in features.columns and not features.empty else 0,
        "source_feature_path": str(source_path.resolve()),
        "source_feature_row_count": int(source_record.get("row_count", 0) or 0),
        "source_raw_path": str(source_record.get("source_raw_path", "") or ""),
        "source_raw_chunk_id": str(source_record.get("source_raw_chunk_id", "") or ""),
        "source_member": str(source_record.get("source_member", "") or ""),
        "combined_from_dataset_id": str(source_record.get("combined_from_dataset_id", "") or ""),
        "error_count": 0,
        "error": "",
        "materialization": materialization,
        "auction_process_policy": "mootdx_240_0931_opening_and_1500_closing_overlay",
        "auction_feature_policy": "override_opening_closing_auction_fields_from_mootdx_240_1m_0931_1500",
    }


def _feature_dataset_spec(cfg: BuildIntradayDailyFeaturesConfig, *, source_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "intraday_daily_features_from_5m",
        "domain": DataDomain.INTRADAY_DAILY_FEATURES,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "years": list(cfg.years),
        "source_5m_dataset_id": cfg.source_dataset_id,
        "source_5m_fingerprint": str(source_metadata.get("fingerprint", "") or ""),
        "source_5m_dataset_kind": str(source_metadata.get("dataset_kind", "") or ""),
        "auction_1m_dataset_id": cfg.auction_1m_dataset_id,
        "auction_feature_policy": (
            "override_opening_closing_auction_fields_from_mootdx_240_1m_0931_1500"
            if cfg.auction_1m_dataset_id
            else "proxy_from_first_last_5m_bar"
        ),
        "feature_source_name": cfg.source_name,
        "feature_contract_version": INTRADAY_DAILY_FEATURE_CONTRACT_VERSION,
        "last_5m_ret_policy": LAST_5M_RET_POLICY,
        "adjusted_flag": cfg.adjusted_flag,
        "raw_ohlcv_policy": "raw_ohlcv_never_overwritten",
        "auction_process_policy": "unobservable_use_open_as_opening_result_only",
        "prediction_policy": "daily_features_for_next_day_or_multi_day_prediction_not_intraday_realtime",
    }


def _feature_overlay_dataset_spec(
    cfg: BuildIntradayDailyFeaturesConfig,
    *,
    feature_metadata: dict[str, Any],
    auction_1m_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source": "intraday_daily_features_auction_1m_overlay",
        "domain": DataDomain.INTRADAY_DAILY_FEATURES,
        "start_date": cfg.start_date or str(feature_metadata.get("start_date", "") or ""),
        "end_date": cfg.end_date or str(feature_metadata.get("end_date", "") or ""),
        "years": list(cfg.years),
        "existing_feature_dataset_id": cfg.existing_feature_dataset_id,
        "existing_feature_fingerprint": str(feature_metadata.get("fingerprint", "") or ""),
        "existing_feature_dataset_kind": str(feature_metadata.get("dataset_kind", "") or ""),
        "auction_1m_dataset_id": cfg.auction_1m_dataset_id,
        "auction_1m_fingerprint": str(auction_1m_metadata.get("fingerprint", "") or ""),
        "auction_feature_policy": "override_opening_closing_auction_fields_from_mootdx_240_1m_0931_1500",
        "feature_contract_version": INTRADAY_DAILY_FEATURE_CONTRACT_VERSION,
        "last_5m_ret_policy": LAST_5M_RET_POLICY,
        "raw_ohlcv_policy": "raw_ohlcv_never_overwritten",
        "auction_process_policy": "mootdx_240_0931_opening_and_1500_closing_overlay",
        "prediction_policy": "daily_features_for_next_day_or_multi_day_prediction_not_intraday_realtime",
    }


def _shard_intersects_years(record: dict[str, Any], years: set[int]) -> bool:
    if not years:
        return True
    start = pd.to_datetime(str(record.get("start_date", "") or ""), errors="coerce")
    end = pd.to_datetime(str(record.get("end_date", "") or ""), errors="coerce")
    if pd.isna(start) and pd.isna(end):
        return True
    if pd.isna(end):
        end = start
    if pd.isna(start):
        start = end
    return any(int(start.year) <= year <= int(end.year) for year in years)


def _safe_stem(raw: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(raw or "shard"))
    return safe[:180] if len(safe) > 180 else safe


def _default_progress_path(lake: ResearchDataLake) -> Path:
    return lake.root / "canonical" / "imports" / f"intraday_daily_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}_progress.jsonl"


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": datetime.now(timezone.utc).replace(microsecond=0).isoformat(), **payload}
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with _PROGRESS_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build daily intraday feature sidecar from 5m data or overlay 1m auction fields onto an existing feature dataset.")
    parser.add_argument("--lake-root", default=str(DEFAULT_DATA_LAKE_ROOT))
    parser.add_argument("--source-dataset-id", default="")
    parser.add_argument("--existing-feature-dataset-id", default="")
    parser.add_argument("--auction-1m-dataset-id", default="")
    parser.add_argument("--start-date", default=DEFAULT_CANONICAL_START_DATE)
    parser.add_argument("--end-date", default="")
    parser.add_argument("--years", default="")
    parser.add_argument("--source-name", default="external_1m_derived_5m")
    parser.add_argument("--adjusted-flag", default="none")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--progress-path", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--reuse", dest="reuse", action="store_true", default=True)
    parser.add_argument("--no-reuse", dest="reuse", action="store_false")
    return parser


def _parse_years(text: str) -> tuple[int, ...]:
    years: list[int] = []
    for raw_item in str(text or "").split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if end < start:
                raise ValueError(f"invalid_year_range: {item}")
            years.extend(range(start, end + 1))
        else:
            years.append(int(item))
    return tuple(dict.fromkeys(years))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_intraday_daily_features(
        BuildIntradayDailyFeaturesConfig(
            lake_root=Path(args.lake_root),
            source_dataset_id=args.source_dataset_id,
            existing_feature_dataset_id=args.existing_feature_dataset_id,
            auction_1m_dataset_id=args.auction_1m_dataset_id,
            start_date=args.start_date,
            end_date=args.end_date,
            years=_parse_years(args.years),
            source_name=args.source_name,
            adjusted_flag=args.adjusted_flag,
            workers=args.workers,
            max_shards=args.max_shards,
            progress_path=Path(args.progress_path) if str(args.progress_path or "").strip() else None,
            dry_run=args.dry_run,
            resume=args.resume,
            reuse=args.reuse,
        )
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "dataset_id": result.dataset_id,
                "shard_count": result.shard_count,
                "row_count": result.row_count,
                "error_count": result.error_count,
                "progress_path": str(result.progress_path.resolve()) if result.progress_path else "",
                "errors": result.errors,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
