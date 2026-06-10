from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from daily_research.data_lake.canonical import DEFAULT_CANONICAL_START_DATE, DEFAULT_EXTERNAL_QUANT_DATA_ROOT
from daily_research.data_lake.catalog import DEFAULT_DATA_LAKE_ROOT, ResearchDataLake
from daily_research.data_platform.contracts import (
    DataDomain,
    aggregate_intraday_1m_to_5m_frame,
    normalize_domain,
    normalize_domain_frame,
)


DEFAULT_DOMAINS = (
    DataDomain.MARKET_INTRADAY_5M,
    DataDomain.ADJUST_FACTOR,
)
DEFAULT_ENCODINGS = ("utf-8-sig", "gb18030", "gbk")
_PROGRESS_LOCK = threading.Lock()


@dataclass
class ImportConfig:
    lake_root: Path = DEFAULT_DATA_LAKE_ROOT
    source_root: Path = DEFAULT_EXTERNAL_QUANT_DATA_ROOT
    domains: tuple[str, ...] = DEFAULT_DOMAINS
    start_date: str = DEFAULT_CANONICAL_START_DATE
    end_date: str = ""
    years: tuple[int, ...] = ()
    csv_sep: str = ","
    chunksize: int = 200_000
    max_files_per_zip: int = 0
    dry_run: bool = False
    derive_5m_from_1m: bool = False
    hash_zips: bool = True
    reuse: bool = True
    progress_path: Path | None = None
    workers: int = 1
    shard_batch_members: int = 1
    shard_batch_rows: int = 0

    def normalized(self) -> "ImportConfig":
        domains = tuple(normalize_domain(item) for item in self.domains)
        return ImportConfig(
            lake_root=Path(self.lake_root),
            source_root=Path(self.source_root),
            domains=domains,
            start_date=_date_text(self.start_date or DEFAULT_CANONICAL_START_DATE),
            end_date=_date_text(self.end_date) if str(self.end_date or "").strip() else "",
            years=tuple(sorted({int(item) for item in self.years})),
            csv_sep=str(self.csv_sep or ","),
            chunksize=max(1, int(self.chunksize or 200_000)),
            max_files_per_zip=max(0, int(self.max_files_per_zip or 0)),
            dry_run=bool(self.dry_run),
            derive_5m_from_1m=bool(self.derive_5m_from_1m),
            hash_zips=bool(self.hash_zips),
            reuse=bool(self.reuse),
            progress_path=Path(self.progress_path) if self.progress_path else None,
            workers=max(1, int(self.workers or 1)),
            shard_batch_members=max(1, int(self.shard_batch_members or 1)),
            shard_batch_rows=max(0, int(self.shard_batch_rows or 0)),
        )


@dataclass
class ImportResult:
    status: str
    lake_root: Path
    source_root: Path
    dataset_ids: dict[str, str] = field(default_factory=dict)
    shard_counts: dict[str, int] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    error_counts: dict[str, int] = field(default_factory=dict)
    plan_path: Path | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import external native 5m/adjust-factor yearly zips into the research data lake.")
    parser.add_argument("--lake-root", default=str(DEFAULT_DATA_LAKE_ROOT))
    parser.add_argument("--source-root", default=str(DEFAULT_EXTERNAL_QUANT_DATA_ROOT))
    parser.add_argument("--domains", default="market_intraday_5m,adjust_factor")
    parser.add_argument("--start-date", default=DEFAULT_CANONICAL_START_DATE)
    parser.add_argument("--end-date", default="")
    parser.add_argument("--years", default="", help="Comma separated years. Empty means all years >= start-date year.")
    parser.add_argument("--csv-sep", default=",")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--max-files-per-zip", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--derive-5m-from-1m", dest="derive_5m_from_1m", action="store_true", default=False)
    parser.add_argument("--no-derive-5m-from-1m", dest="derive_5m_from_1m", action="store_false")
    parser.add_argument("--hash-zips", dest="hash_zips", action="store_true", default=True)
    parser.add_argument("--no-hash-zips", dest="hash_zips", action="store_false")
    parser.add_argument("--reuse", dest="reuse", action="store_true", default=True)
    parser.add_argument("--no-reuse", dest="reuse", action="store_false")
    parser.add_argument("--progress-path", default="")
    parser.add_argument("--workers", type=int, default=1, help="Parallel zip workers. Catalog registration remains single-process.")
    parser.add_argument("--shard-batch-members", type=int, default=1, help="CSV members per parquet shard within each zip.")
    parser.add_argument("--shard-batch-rows", type=int, default=0, help="Flush a parquet shard after this many rows. 0 disables row-based flushing.")
    return parser


def run_import(config: ImportConfig) -> ImportResult:
    cfg = config.normalized()
    lake = ResearchDataLake(cfg.lake_root)
    progress_path = cfg.progress_path or _default_progress_path(lake)
    zip_paths_by_domain = _zip_paths_by_domain(cfg)
    plan: dict[str, Any] = {
        "status": "dry_run" if cfg.dry_run else "running",
        "generated_at": _utc_now(),
        "lake_root": str(lake.root.resolve()),
        "source_root": str(cfg.source_root),
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "domains": list(cfg.domains),
        "zip_paths_by_domain": {domain: [str(path) for path in paths] for domain, paths in zip_paths_by_domain.items()},
        "datasets": {},
        "progress_path": str(progress_path.resolve()),
    }
    _write_progress(
        progress_path,
        {
            "event": "start",
            "dry_run": cfg.dry_run,
            "domains": list(cfg.domains),
            "zip_count": int(sum(len(paths) for paths in zip_paths_by_domain.values())),
            "workers": int(cfg.workers),
            "shard_batch_members": int(cfg.shard_batch_members),
            "shard_batch_rows": int(cfg.shard_batch_rows),
        },
    )
    if cfg.dry_run:
        plan_path = _write_plan(lake, plan)
        _write_progress(progress_path, {"event": "dry_run_written", "plan_path": str(plan_path.resolve())})
        return ImportResult(status="dry_run", lake_root=lake.root, source_root=cfg.source_root, plan_path=plan_path)

    shard_records_by_domain: dict[str, list[dict[str, Any]]] = {domain: [] for domain in cfg.domains}
    if cfg.derive_5m_from_1m and DataDomain.MARKET_INTRADAY_1M in cfg.domains:
        shard_records_by_domain.setdefault(DataDomain.MARKET_INTRADAY_5M, [])

    for domain in cfg.domains:
        domain_zip_paths = list(zip_paths_by_domain.get(domain, []))
        if cfg.workers <= 1 or len(domain_zip_paths) <= 1:
            for zip_path in domain_zip_paths:
                _merge_shard_records(
                    shard_records_by_domain,
                    _import_zip_for_domain(
                        lake=lake,
                        cfg=cfg,
                        domain=domain,
                        zip_path=zip_path,
                        progress_path=progress_path,
                    ),
                )
        else:
            with ThreadPoolExecutor(max_workers=min(cfg.workers, len(domain_zip_paths))) as executor:
                futures = {
                    executor.submit(
                        _import_zip_for_domain,
                        lake=lake,
                        cfg=cfg,
                        domain=domain,
                        zip_path=zip_path,
                        progress_path=progress_path,
                    ): zip_path
                    for zip_path in domain_zip_paths
                }
                for future in as_completed(futures):
                    zip_path = futures[future]
                    try:
                        _merge_shard_records(shard_records_by_domain, future.result())
                    except Exception as exc:
                        shard_records_by_domain.setdefault(domain, []).append(
                            _shard_record(
                                domain=domain,
                                zip_path=zip_path,
                                member_name="",
                                zip_sha256="",
                                status="error",
                                row_count=0,
                                error_count=1,
                                error=str(exc),
                            )
                        )
                        _write_progress(
                            progress_path,
                            {"event": "zip_error", "domain": domain, "zip_path": str(zip_path.resolve()), "error": str(exc)},
                        )

    dataset_ids: dict[str, str] = {}
    shard_counts: dict[str, int] = {}
    row_counts: dict[str, int] = {}
    error_counts: dict[str, int] = {}
    for domain, shard_records in shard_records_by_domain.items():
        if not shard_records:
            continue
        spec = _dataset_spec(cfg, domain=domain)
        record = lake.save_sharded_domain_dataset(
            domain=domain,
            spec=spec,
            shard_records=shard_records,
            source="external_quant_zip",
            reuse=cfg.reuse,
        )
        dataset_ids[domain] = record.dataset_id
        shard_counts[domain] = int(len(shard_records))
        row_counts[domain] = int(sum(int(item.get("row_count", 0) or 0) for item in shard_records))
        error_counts[domain] = int(sum(int(item.get("error_count", 0) or 0) for item in shard_records))
        plan["datasets"][domain] = {
            "dataset_id": record.dataset_id,
            "shard_count": shard_counts[domain],
            "row_count": row_counts[domain],
            "error_count": error_counts[domain],
        }
        _write_progress(
            progress_path,
            {
                "event": "dataset_registered",
                "domain": domain,
                "dataset_id": record.dataset_id,
                "shard_count": shard_counts[domain],
                "row_count": row_counts[domain],
                "error_count": error_counts[domain],
            },
        )
    plan["status"] = "completed"
    plan_path = _write_plan(lake, plan)
    _write_progress(progress_path, {"event": "completed", "plan_path": str(plan_path.resolve()), "dataset_ids": dataset_ids})
    return ImportResult(
        status="completed",
        lake_root=lake.root,
        source_root=cfg.source_root,
        dataset_ids=dataset_ids,
        shard_counts=shard_counts,
        row_counts=row_counts,
        error_counts=error_counts,
        plan_path=plan_path,
    )


def _import_zip_for_domain(
    *,
    lake: ResearchDataLake,
    cfg: ImportConfig,
    domain: str,
    zip_path: Path,
    progress_path: Path,
) -> dict[str, list[dict[str, Any]]]:
    shard_records_by_domain: dict[str, list[dict[str, Any]]] = {}
    _write_progress(progress_path, {"event": "zip_start", "domain": domain, "zip_path": str(zip_path.resolve())})
    zip_sha256 = _sha256_file(zip_path) if cfg.hash_zips else ""
    batch_frames: dict[str, list[pd.DataFrame]] = {}
    batch_members: dict[str, list[str]] = {}
    batch_derivation: dict[str, str] = {}
    batch_rows: dict[str, int] = {}
    batch_index: dict[str, int] = {}

    def flush_batch(target_domain: str) -> None:
        frames = batch_frames.pop(target_domain, [])
        members_for_batch = batch_members.pop(target_domain, [])
        derivation = str(batch_derivation.pop(target_domain, "") or "")
        batch_rows.pop(target_domain, None)
        if not frames:
            return
        batch_index[target_domain] = int(batch_index.get(target_domain, 0)) + 1
        data = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        member_label = f"{zip_path.stem}_{target_domain}_batch_{batch_index[target_domain]:05d}"
        shard_path = _write_domain_shard(
            lake,
            cfg,
            domain=target_domain,
            frame=data,
            zip_path=zip_path,
            member_name=member_label,
            suffix="derived_5m" if derivation else "",
        )
        shard_records_by_domain.setdefault(target_domain, []).append(
            _shard_record(
                domain=target_domain,
                zip_path=zip_path,
                member_name=member_label,
                zip_sha256=zip_sha256,
                status="stored",
                row_count=len(data),
                path=shard_path,
                start_date=str(data["trade_date"].min()) if "trade_date" in data.columns and not data.empty else "",
                end_date=str(data["trade_date"].max()) if "trade_date" in data.columns and not data.empty else "",
                symbol_count=int(data["symbol"].nunique()) if "symbol" in data.columns and not data.empty else 0,
                derivation=derivation,
                source_member_count=len(members_for_batch),
                source_members_sample=members_for_batch[:20],
            )
        )
        _write_progress(
            progress_path,
            {
                "event": "batch_stored",
                "domain": target_domain,
                "zip_path": str(zip_path.resolve()),
                "batch": member_label,
                "row_count": int(len(data)),
                "source_member_count": int(len(members_for_batch)),
                "derivation": derivation,
            },
        )

    def add_batch_frame(target_domain: str, frame: pd.DataFrame, *, member_name: str, derivation: str = "") -> None:
        batch_frames.setdefault(target_domain, []).append(frame)
        batch_members.setdefault(target_domain, []).append(member_name)
        if derivation:
            batch_derivation[target_domain] = derivation
        batch_rows[target_domain] = int(batch_rows.get(target_domain, 0)) + int(len(frame))
        member_limit_hit = len(batch_members[target_domain]) >= int(cfg.shard_batch_members)
        row_limit = int(cfg.shard_batch_rows)
        row_limit_hit = bool(row_limit > 0 and batch_rows[target_domain] >= row_limit)
        if member_limit_hit or row_limit_hit:
            flush_batch(target_domain)

    with zipfile.ZipFile(zip_path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir() and item.filename.lower().endswith((".csv", ".txt"))]
        members = sorted(members, key=lambda item: item.filename)
        if cfg.max_files_per_zip:
            members = members[: cfg.max_files_per_zip]
        for member in members:
            try:
                _write_progress(progress_path, {"event": "member_start", "domain": domain, "zip_path": str(zip_path.resolve()), "member": member.filename})
                normalized = _read_member_normalized(archive, member.filename, domain=domain, cfg=cfg, zip_path=zip_path)
                normalized = _filter_date_window(normalized, start_date=cfg.start_date, end_date=cfg.end_date)
                if normalized.empty:
                    shard_records_by_domain.setdefault(domain, []).append(
                        _shard_record(
                            domain=domain,
                            zip_path=zip_path,
                            member_name=member.filename,
                            zip_sha256=zip_sha256,
                            status="skipped",
                            row_count=0,
                            error_count=0,
                        )
                    )
                    _write_progress(progress_path, {"event": "member_skipped", "domain": domain, "zip_path": str(zip_path.resolve()), "member": member.filename})
                    continue
                add_batch_frame(domain, normalized, member_name=member.filename)
                _write_progress(
                    progress_path,
                    {
                        "event": "member_buffered",
                        "domain": domain,
                        "zip_path": str(zip_path.resolve()),
                        "member": member.filename,
                        "row_count": int(len(normalized)),
                    },
                )
                if domain == DataDomain.MARKET_INTRADAY_1M and cfg.derive_5m_from_1m:
                    derived_5m = aggregate_intraday_1m_to_5m_frame(normalized, source="external_1m", adjusted_flag="none")
                    derived_5m = _filter_date_window(derived_5m, start_date=cfg.start_date, end_date=cfg.end_date)
                    if not derived_5m.empty:
                        add_batch_frame(
                            DataDomain.MARKET_INTRADAY_5M,
                            derived_5m,
                            member_name=member.filename,
                            derivation="aggregate_1m_to_5m",
                        )
                        _write_progress(
                            progress_path,
                            {
                                "event": "member_derived_5m_buffered",
                                "domain": DataDomain.MARKET_INTRADAY_5M,
                                "source_domain": domain,
                                "zip_path": str(zip_path.resolve()),
                                "member": member.filename,
                                "row_count": int(len(derived_5m)),
                            },
                        )
            except Exception as exc:
                shard_records_by_domain.setdefault(domain, []).append(
                    _shard_record(
                        domain=domain,
                        zip_path=zip_path,
                        member_name=member.filename,
                        zip_sha256=zip_sha256,
                        status="error",
                        row_count=0,
                        error_count=1,
                        error=str(exc),
                    )
                )
                _write_progress(
                    progress_path,
                    {
                        "event": "member_error",
                        "domain": domain,
                        "zip_path": str(zip_path.resolve()),
                        "member": member.filename,
                        "error": str(exc),
                    },
                )
        for target_domain in list(batch_frames):
            flush_batch(target_domain)
    _write_progress(progress_path, {"event": "zip_done", "domain": domain, "zip_path": str(zip_path.resolve()), "member_count": int(len(members))})
    return shard_records_by_domain


def _merge_shard_records(target: dict[str, list[dict[str, Any]]], source: dict[str, list[dict[str, Any]]]) -> None:
    for domain, records in source.items():
        target.setdefault(domain, []).extend(list(records))


def _read_member_normalized(
    archive: zipfile.ZipFile,
    member_name: str,
    *,
    domain: str,
    cfg: ImportConfig,
    zip_path: Path,
) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in DEFAULT_ENCODINGS:
        try:
            frames: list[pd.DataFrame] = []
            for chunk in pd.read_csv(
                archive.open(member_name),
                sep=cfg.csv_sep,
                encoding=encoding,
                chunksize=cfg.chunksize,
                low_memory=False,
            ):
                chunk = _prepare_external_chunk(chunk, domain=domain, zip_path=zip_path, member_name=member_name)
                normalized = normalize_domain_frame(
                    chunk,
                    domain=domain,
                    source="external_quant_zip",
                    adjusted_flag="none",
                    require_columns=False,
                )
                if not normalized.empty:
                    frames.append(normalized)
            if not frames:
                return pd.DataFrame()
            data = pd.concat(frames, ignore_index=True)
            sort_columns = [column for column in ("trade_date", "symbol", "bar_time", "source") if column in data.columns]
            dedupe_columns = [column for column in ("trade_date", "symbol", "bar_time", "factor_provider", "source") if column in data.columns]
            return data.drop_duplicates(subset=dedupe_columns or None, keep="last").sort_values(sort_columns).reset_index(drop=True)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
        except ValueError as exc:
            errors.append(f"{encoding}: {exc}")
            break
    raise RuntimeError(f"csv_member_read_failed: member={member_name}; errors={errors}")


def _prepare_external_chunk(chunk: pd.DataFrame, *, domain: str, zip_path: Path, member_name: str) -> pd.DataFrame:
    data = chunk.copy()
    if "symbol" not in {str(column).strip().lower() for column in data.columns}:
        symbol = _symbol_from_member_name(member_name)
        if symbol:
            data["symbol"] = symbol
    if domain == DataDomain.ADJUST_FACTOR:
        provider, semantics = _adjust_factor_provider(zip_path, member_name)
        if "factor_provider" not in data.columns:
            data["factor_provider"] = provider
        if "factor_semantics" not in data.columns:
            data["factor_semantics"] = semantics
    return data


def _filter_date_window(frame: pd.DataFrame, *, start_date: str, end_date: str) -> pd.DataFrame:
    if frame.empty or "trade_date" not in frame.columns:
        return frame
    dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    mask = dates.notna() & dates.ge(pd.Timestamp(start_date))
    if end_date:
        mask &= dates.le(pd.Timestamp(end_date))
    return frame.loc[mask].reset_index(drop=True)


def _write_domain_shard(
    lake: ResearchDataLake,
    cfg: ImportConfig,
    *,
    domain: str,
    frame: pd.DataFrame,
    zip_path: Path,
    member_name: str,
    suffix: str = "",
) -> str:
    spec = _dataset_spec(cfg, domain=domain)
    identity_spec = {**spec, "domain": domain, "sharded": True}
    identity = lake.build_domain_dataset_identity(domain=domain, spec=identity_spec)
    shard_dir = Path(identity["dataset_dir"]) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_name = _safe_shard_name(zip_path, member_name, suffix=suffix)
    shard_path = shard_dir / f"{shard_name}.parquet"
    frame.to_parquet(shard_path, index=False)
    return str(shard_path.resolve())


def _dataset_spec(cfg: ImportConfig, *, domain: str) -> dict[str, Any]:
    return {
        "source": "external_quant_zip",
        "domain": domain,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "years": list(cfg.years),
        "source_root": str(cfg.source_root),
        "canonical_start_date": DEFAULT_CANONICAL_START_DATE,
        "raw_archive_policy": "keep_2000_2009_outside_default_research_view",
        "original_ohlcv_policy": "raw_ohlcv_never_overwritten",
        "derive_5m_from_1m": bool(cfg.derive_5m_from_1m),
        "one_minute_policy": "cold_archive_not_default_research_view",
        "five_minute_policy": "native_5m_preferred_reuse_existing_derived_5m_if_equivalent",
        "shard_batch_members": int(cfg.shard_batch_members),
        "shard_batch_rows": int(cfg.shard_batch_rows),
    }


def _shard_record(
    *,
    domain: str,
    zip_path: Path,
    member_name: str,
    zip_sha256: str,
    status: str,
    row_count: int,
    path: str | Path = "",
    start_date: str = "",
    end_date: str = "",
    symbol_count: int = 0,
    error_count: int = 0,
    error: str = "",
    derivation: str = "",
    source_member_count: int = 1,
    source_members_sample: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "domain": domain,
        "status": status,
        "path": str(path or ""),
        "row_count": int(row_count),
        "start_date": str(start_date or ""),
        "end_date": str(end_date or ""),
        "symbol_count": int(symbol_count),
        "source_zip": str(zip_path.resolve()),
        "source_zip_sha256": str(zip_sha256 or ""),
        "source_member": str(member_name),
        "source_member_count": int(source_member_count),
        "source_members_sample": list(source_members_sample or []),
        "error_count": int(error_count),
        "error": str(error or ""),
        "derivation": str(derivation or ""),
    }


def _zip_paths_by_domain(cfg: ImportConfig) -> dict[str, list[Path]]:
    paths_by_domain = {domain: [] for domain in cfg.domains}
    if not cfg.source_root.exists():
        return paths_by_domain
    start_year = pd.Timestamp(cfg.start_date).year
    for path in sorted(cfg.source_root.rglob("*.zip")):
        domain = _classify_zip_path(path)
        if domain not in paths_by_domain:
            continue
        year = _year_from_path(path)
        if cfg.years and year not in cfg.years:
            continue
        if not cfg.years and year is not None and year < start_year:
            continue
        paths_by_domain[domain].append(path)
    return paths_by_domain


def _classify_zip_path(path: Path) -> str:
    text = str(path).lower()
    if "1分钟" in text or "1min" in text or "1m" in text:
        return DataDomain.MARKET_INTRADAY_1M
    if "5分钟" in text or "5min" in text or "5m" in text:
        return DataDomain.MARKET_INTRADAY_5M
    if "复权" in text or "adjust" in text:
        return DataDomain.ADJUST_FACTOR
    return ""


def _symbol_from_member_name(member_name: str) -> str:
    name = Path(member_name).stem.strip().lower()
    match = re.search(r"(sh|sz|bj)[._-]?(\d{6})", name)
    if match:
        exchange = {"sh": "SH", "sz": "SZ", "bj": "BJ"}[match.group(1)]
        return f"{match.group(2)}.{exchange}"
    match = re.search(r"(\d{6})[._-]?(sh|sz|bj)", name)
    if match:
        exchange = {"sh": "SH", "sz": "SZ", "bj": "BJ"}[match.group(2)]
        return f"{match.group(1)}.{exchange}"
    return ""


def _adjust_factor_provider(zip_path: Path, member_name: str) -> tuple[str, str]:
    text = f"{zip_path} {member_name}".lower()
    if "新浪" in text or "sina" in text:
        return "sina", "external_sina_event_factor"
    if "同花顺" in text or "ths" in text or "tonghuashun" in text:
        return "tonghuashun", "external_tonghuashun_raw_factor"
    return "external", "external_raw_factor"


def _year_from_path(path: Path) -> int | None:
    matches = re.findall(r"(?:19|20)\d{2}", str(path))
    if not matches:
        return None
    return int(matches[-1])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_shard_name(zip_path: Path, member_name: str, *, suffix: str) -> str:
    raw = f"{zip_path.stem}_{member_name}_{suffix}".strip("_")
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in raw)
    if len(safe) > 180:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        safe = f"{safe[:150]}_{digest}"
    return safe


def _date_text(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _parse_domains(text: str) -> tuple[str, ...]:
    values = [item.strip() for item in str(text or "").split(",") if item.strip()]
    return tuple(normalize_domain(item) for item in values) if values else DEFAULT_DOMAINS


def _parse_years(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in str(text or "").split(",") if item.strip())


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_plan(lake: ResearchDataLake, plan: dict[str, Any]) -> Path:
    path = lake.root / "canonical" / "imports" / f"external_quant_zip_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _default_progress_path(lake: ResearchDataLake) -> Path:
    return lake.root / "canonical" / "imports" / f"external_quant_zip_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}_progress.jsonl"


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": _utc_now(), **payload}
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with _PROGRESS_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_import(
        ImportConfig(
            lake_root=Path(args.lake_root),
            source_root=Path(args.source_root),
            domains=_parse_domains(args.domains),
            start_date=args.start_date,
            end_date=args.end_date,
            years=_parse_years(args.years),
            csv_sep=args.csv_sep,
            chunksize=args.chunksize,
            max_files_per_zip=args.max_files_per_zip,
            dry_run=args.dry_run,
            derive_5m_from_1m=args.derive_5m_from_1m,
            hash_zips=args.hash_zips,
            reuse=args.reuse,
            progress_path=Path(args.progress_path) if str(args.progress_path or "").strip() else None,
            workers=args.workers,
            shard_batch_members=args.shard_batch_members,
            shard_batch_rows=args.shard_batch_rows,
        )
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "dataset_ids": result.dataset_ids,
                "shard_counts": result.shard_counts,
                "row_counts": result.row_counts,
                "error_counts": result.error_counts,
                "plan_path": str(result.plan_path.resolve()) if result.plan_path else "",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
