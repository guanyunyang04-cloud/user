from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from daily_research.data_lake.canonical import DEFAULT_CANONICAL_START_DATE
from daily_research.data_lake.catalog import DEFAULT_DATA_LAKE_ROOT, ResearchDataLake
from daily_research.data_platform.contracts import (
    DataDomain,
    build_intraday_daily_feature_frame,
    normalize_domain,
)


_PROGRESS_LOCK = threading.Lock()


@dataclass(frozen=True)
class BuildIntradayDailyFeaturesConfig:
    lake_root: Path = DEFAULT_DATA_LAKE_ROOT
    source_dataset_id: str = ""
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
    if not cfg.source_dataset_id:
        raise ValueError("source_dataset_id_required")
    lake = ResearchDataLake(cfg.lake_root)
    source_metadata = lake.describe_dataset(cfg.source_dataset_id)
    source_domain = normalize_domain(str(dict(source_metadata.get("parameters", {}) or {}).get("domain", "") or ""))
    if source_domain != DataDomain.MARKET_INTRADAY_5M:
        raise ValueError(f"source_domain_mismatch: {source_domain}; expected={DataDomain.MARKET_INTRADAY_5M}")
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
            "source_shard_count": len(source_shards),
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
            records.append(_build_one_shard(lake=lake, cfg=cfg, shard_dir=shard_dir, source_record=source_record, progress_path=progress_path, index=idx, total=len(source_shards)))
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


def _build_one_shard(
    *,
    lake: ResearchDataLake,
    cfg: BuildIntradayDailyFeaturesConfig,
    shard_dir: Path,
    source_record: dict[str, Any],
    progress_path: Path,
    index: int,
    total: int,
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
    features = build_intraday_daily_feature_frame(raw, source=cfg.source_name, adjusted_flag=cfg.adjusted_flag)
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
        "feature_source_name": cfg.source_name,
        "adjusted_flag": cfg.adjusted_flag,
        "raw_ohlcv_policy": "raw_ohlcv_never_overwritten",
        "auction_process_policy": "unobservable_use_open_as_opening_result_only",
        "prediction_policy": "daily_features_for_next_day_or_multi_day_prediction_not_intraday_realtime",
    }


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
    parser = argparse.ArgumentParser(description="Build daily intraday feature sidecar from a sharded 5m dataset.")
    parser.add_argument("--lake-root", default=str(DEFAULT_DATA_LAKE_ROOT))
    parser.add_argument("--source-dataset-id", required=True)
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
