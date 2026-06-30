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

from quant_data_platform.lake.canonical import DEFAULT_CANONICAL_START_DATE, DEFAULT_EXTERNAL_QUANT_DATA_ROOT
from quant_data_platform.lake.catalog import DEFAULT_DATA_LAKE_ROOT, ResearchDataLake
from quant_data_platform.domains.contracts import (
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
    max_files_per_source: int = 0
    dry_run: bool = False
    include_unpacked_csv: bool = False
    derive_5m_from_1m: bool = False
    normalize_intraday_1m_to_mootdx_240: bool = True
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
            max_files_per_source=max(0, int(self.max_files_per_source or self.max_files_per_zip or 0)),
            dry_run=bool(self.dry_run),
            include_unpacked_csv=bool(self.include_unpacked_csv),
            derive_5m_from_1m=bool(self.derive_5m_from_1m),
            normalize_intraday_1m_to_mootdx_240=bool(self.normalize_intraday_1m_to_mootdx_240),
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


@dataclass(frozen=True)
class ExternalSourcePath:
    path: Path
    kind: str


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
    parser.add_argument("--max-files-per-source", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-unpacked-csv", action="store_true")
    parser.add_argument("--derive-5m-from-1m", dest="derive_5m_from_1m", action="store_true", default=False)
    parser.add_argument("--no-derive-5m-from-1m", dest="derive_5m_from_1m", action="store_false")
    parser.add_argument("--normalize-1m-to-mootdx-240", dest="normalize_intraday_1m_to_mootdx_240", action="store_true", default=True)
    parser.add_argument("--preserve-source-1m-bars", dest="normalize_intraday_1m_to_mootdx_240", action="store_false")
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
    if cfg.reuse and not cfg.dry_run:
        reuse_hit = _existing_import_result(lake=lake, cfg=cfg)
        expected_domains = _expected_output_domains(cfg)
        if expected_domains and all(domain in reuse_hit["dataset_ids"] for domain in expected_domains):
            plan = {
                "status": "completed_reused",
                "generated_at": _utc_now(),
                "lake_root": str(lake.root.resolve()),
                "source_root": str(cfg.source_root),
                "start_date": cfg.start_date,
                "end_date": cfg.end_date,
                "domains": list(cfg.domains),
                "datasets": {
                    domain: {
                        "dataset_id": reuse_hit["dataset_ids"][domain],
                        "shard_count": int(reuse_hit["shard_counts"].get(domain, 0) or 0),
                        "row_count": int(reuse_hit["row_counts"].get(domain, 0) or 0),
                        "error_count": int(reuse_hit["error_counts"].get(domain, 0) or 0),
                        "reuse": True,
                    }
                    for domain in expected_domains
                },
                "progress_path": str(progress_path.resolve()),
                "reuse": True,
            }
            plan_path = _write_plan(lake, plan)
            _write_progress(
                progress_path,
                {
                    "event": "completed_reused",
                    "dataset_ids": reuse_hit["dataset_ids"],
                    "plan_path": str(plan_path.resolve()),
                },
            )
            return ImportResult(
                status="completed_reused",
                lake_root=lake.root,
                source_root=cfg.source_root,
                dataset_ids=dict(reuse_hit["dataset_ids"]),
                shard_counts=dict(reuse_hit["shard_counts"]),
                row_counts=dict(reuse_hit["row_counts"]),
                error_counts=dict(reuse_hit["error_counts"]),
                plan_path=plan_path,
            )
    source_paths_by_domain = _source_paths_by_domain(cfg)
    plan: dict[str, Any] = {
        "status": "dry_run" if cfg.dry_run else "running",
        "generated_at": _utc_now(),
        "lake_root": str(lake.root.resolve()),
        "source_root": str(cfg.source_root),
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "domains": list(cfg.domains),
        "zip_paths_by_domain": {
            domain: [str(item.path) for item in paths if item.kind == "zip"]
            for domain, paths in source_paths_by_domain.items()
        },
        "source_paths_by_domain": {
            domain: [{"path": str(item.path), "kind": item.kind} for item in paths]
            for domain, paths in source_paths_by_domain.items()
        },
        "include_unpacked_csv": bool(cfg.include_unpacked_csv),
        "normalize_intraday_1m_to_mootdx_240": bool(cfg.normalize_intraday_1m_to_mootdx_240),
        "datasets": {},
        "progress_path": str(progress_path.resolve()),
    }
    _write_progress(
        progress_path,
        {
            "event": "start",
            "dry_run": cfg.dry_run,
            "domains": list(cfg.domains),
            "source_count": int(sum(len(paths) for paths in source_paths_by_domain.values())),
            "zip_count": int(sum(1 for paths in source_paths_by_domain.values() for item in paths if item.kind == "zip")),
            "directory_count": int(sum(1 for paths in source_paths_by_domain.values() for item in paths if item.kind == "directory")),
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
        domain_source_paths = list(source_paths_by_domain.get(domain, []))
        if cfg.workers <= 1 or len(domain_source_paths) <= 1:
            for source_path in domain_source_paths:
                _merge_shard_records(
                    shard_records_by_domain,
                    _import_source_for_domain(
                        lake=lake,
                        cfg=cfg,
                        domain=domain,
                        source_path=source_path,
                        progress_path=progress_path,
                    ),
                )
        else:
            with ThreadPoolExecutor(max_workers=min(cfg.workers, len(domain_source_paths))) as executor:
                futures = {
                    executor.submit(
                        _import_source_for_domain,
                        lake=lake,
                        cfg=cfg,
                        domain=domain,
                        source_path=source_path,
                        progress_path=progress_path,
                    ): source_path
                    for source_path in domain_source_paths
                }
                for future in as_completed(futures):
                    source_path = futures[future]
                    try:
                        _merge_shard_records(shard_records_by_domain, future.result())
                    except Exception as exc:
                        shard_records_by_domain.setdefault(domain, []).append(
                            _shard_record(
                                domain=domain,
                                source_path=source_path.path,
                                source_kind=source_path.kind,
                                member_name="",
                                source_sha256="",
                                status="error",
                                row_count=0,
                                error_count=1,
                                error=str(exc),
                            )
                        )
                        _write_progress(
                            progress_path,
                            {
                                "event": "source_error",
                                "domain": domain,
                                "source_path": str(source_path.path.resolve()),
                                "source_kind": source_path.kind,
                                "error": str(exc),
                            },
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


def _import_source_for_domain(
    *,
    lake: ResearchDataLake,
    cfg: ImportConfig,
    domain: str,
    source_path: ExternalSourcePath,
    progress_path: Path,
) -> dict[str, list[dict[str, Any]]]:
    shard_records_by_domain: dict[str, list[dict[str, Any]]] = {}
    source_container_path = source_path.path
    source_kind = str(source_path.kind or "zip")
    _write_progress(
        progress_path,
        {
            "event": "source_start",
            "domain": domain,
            "source_path": str(source_container_path.resolve()),
            "source_kind": source_kind,
            "zip_path": str(source_container_path.resolve()) if source_kind == "zip" else "",
        },
    )
    source_sha256 = _sha256_file(source_container_path) if source_kind == "zip" and cfg.hash_zips else ""
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
        member_label = f"{source_container_path.stem}_{target_domain}_batch_{batch_index[target_domain]:05d}"
        shard_path = _write_domain_shard(
            lake,
            cfg,
            domain=target_domain,
            frame=data,
            source_path=source_container_path,
            member_name=member_label,
            suffix="derived_5m" if derivation else "",
        )
        shard_records_by_domain.setdefault(target_domain, []).append(
            _shard_record(
                domain=target_domain,
                source_path=source_container_path,
                source_kind=source_kind,
                member_name=member_label,
                source_sha256=source_sha256,
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
                "source_path": str(source_container_path.resolve()),
                "source_kind": source_kind,
                "zip_path": str(source_container_path.resolve()) if source_kind == "zip" else "",
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

    def handle_member(member_name: str, open_binary: Any) -> None:
        try:
            _write_progress(
                progress_path,
                {
                    "event": "member_start",
                    "domain": domain,
                    "source_path": str(source_container_path.resolve()),
                    "source_kind": source_kind,
                    "member": member_name,
                },
            )
            normalized = _read_member_normalized(
                open_binary,
                member_name,
                domain=domain,
                cfg=cfg,
                source_path=source_container_path,
                source_kind=source_kind,
            )
            normalized = _filter_date_window(normalized, start_date=cfg.start_date, end_date=cfg.end_date)
            if normalized.empty:
                shard_records_by_domain.setdefault(domain, []).append(
                    _shard_record(
                        domain=domain,
                        source_path=source_container_path,
                        source_kind=source_kind,
                        member_name=member_name,
                        source_sha256=source_sha256,
                        status="skipped",
                        row_count=0,
                        error_count=0,
                    )
                )
                _write_progress(
                    progress_path,
                    {
                        "event": "member_skipped",
                        "domain": domain,
                        "source_path": str(source_container_path.resolve()),
                        "source_kind": source_kind,
                        "member": member_name,
                    },
                )
                return
            add_batch_frame(domain, normalized, member_name=member_name)
            _write_progress(
                progress_path,
                {
                    "event": "member_buffered",
                    "domain": domain,
                    "source_path": str(source_container_path.resolve()),
                    "source_kind": source_kind,
                    "member": member_name,
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
                        member_name=member_name,
                        derivation="aggregate_1m_to_5m",
                    )
                    _write_progress(
                        progress_path,
                        {
                            "event": "member_derived_5m_buffered",
                            "domain": DataDomain.MARKET_INTRADAY_5M,
                            "source_domain": domain,
                            "source_path": str(source_container_path.resolve()),
                            "source_kind": source_kind,
                            "member": member_name,
                            "row_count": int(len(derived_5m)),
                        },
                    )
        except Exception as exc:
            shard_records_by_domain.setdefault(domain, []).append(
                _shard_record(
                    domain=domain,
                    source_path=source_container_path,
                    source_kind=source_kind,
                    member_name=member_name,
                    source_sha256=source_sha256,
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
                    "source_path": str(source_container_path.resolve()),
                    "source_kind": source_kind,
                    "member": member_name,
                    "error": str(exc),
                },
            )

    member_count = 0
    if source_kind == "zip":
        with zipfile.ZipFile(source_container_path) as archive:
            members = [item.filename for item in archive.infolist() if not item.is_dir() and item.filename.lower().endswith((".csv", ".txt"))]
            members = sorted(members)
            if cfg.max_files_per_source:
                members = members[: cfg.max_files_per_source]
            member_count = len(members)
            for member_name in members:
                handle_member(member_name, lambda member_name=member_name: archive.open(member_name))
    else:
        members = [
            path
            for path in sorted(source_container_path.iterdir(), key=lambda item: item.name.lower())
            if path.is_file() and path.suffix.lower() in {".csv", ".txt"}
        ]
        if cfg.max_files_per_source:
            members = members[: cfg.max_files_per_source]
        member_count = len(members)
        for member_path in members:
            member_name = member_path.relative_to(source_container_path).as_posix()
            handle_member(member_name, lambda member_path=member_path: member_path.open("rb"))

    for target_domain in list(batch_frames):
        flush_batch(target_domain)
    _write_progress(
        progress_path,
        {
            "event": "source_done",
            "domain": domain,
            "source_path": str(source_container_path.resolve()),
            "source_kind": source_kind,
            "zip_path": str(source_container_path.resolve()) if source_kind == "zip" else "",
            "member_count": int(member_count),
        },
    )
    if source_kind == "zip":
        _write_progress(
            progress_path,
            {
                "event": "zip_done",
                "domain": domain,
                "zip_path": str(source_container_path.resolve()),
                "member_count": int(member_count),
            },
        )
    return shard_records_by_domain


def _merge_shard_records(target: dict[str, list[dict[str, Any]]], source: dict[str, list[dict[str, Any]]]) -> None:
    for domain, records in source.items():
        target.setdefault(domain, []).extend(list(records))


def _read_member_normalized(
    open_binary: Any,
    member_name: str,
    *,
    domain: str,
    cfg: ImportConfig,
    source_path: Path,
    source_kind: str,
) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in DEFAULT_ENCODINGS:
        try:
            frames: list[pd.DataFrame] = []
            with open_binary() as handle:
                for chunk in pd.read_csv(
                    handle,
                    sep=cfg.csv_sep,
                    encoding=encoding,
                    chunksize=cfg.chunksize,
                    low_memory=False,
                ):
                    chunk = _prepare_external_chunk(chunk, domain=domain, source_path=source_path, member_name=member_name)
                    normalized = normalize_domain_frame(
                        chunk,
                        domain=domain,
                        source=_external_source_name(source_kind),
                        adjusted_flag="none",
                        require_columns=False,
                    )
                    if not normalized.empty:
                        frames.append(normalized)
            if not frames:
                return pd.DataFrame()
            data = pd.concat(frames, ignore_index=True)
            if domain == DataDomain.MARKET_INTRADAY_1M and cfg.normalize_intraday_1m_to_mootdx_240:
                data = normalize_intraday_1m_to_mootdx_240_frame(data)
            sort_columns = [column for column in ("trade_date", "symbol", "bar_time", "source") if column in data.columns]
            dedupe_columns = [column for column in ("trade_date", "symbol", "bar_time", "factor_provider", "source") if column in data.columns]
            return data.drop_duplicates(subset=dedupe_columns or None, keep="last").sort_values(sort_columns).reset_index(drop=True)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
        except ValueError as exc:
            errors.append(f"{encoding}: {exc}")
            break
    raise RuntimeError(f"csv_member_read_failed: member={member_name}; errors={errors}")


def _prepare_external_chunk(chunk: pd.DataFrame, *, domain: str, source_path: Path, member_name: str) -> pd.DataFrame:
    data = chunk.copy()
    if "symbol" not in {str(column).strip().lower() for column in data.columns}:
        symbol = _symbol_from_member_name(member_name)
        if symbol:
            data["symbol"] = symbol
    if domain == DataDomain.ADJUST_FACTOR:
        provider, semantics = _adjust_factor_provider(source_path, member_name)
        if "factor_provider" not in data.columns:
            data["factor_provider"] = provider
        if "factor_semantics" not in data.columns:
            data["factor_semantics"] = semantics
    return data


def normalize_intraday_1m_to_mootdx_240_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    required = {"trade_date", "symbol", "bar_time"}
    if not required.issubset(set(frame.columns)):
        return frame
    data = frame.copy()
    data["_qdp_original_order"] = range(len(data))
    keys = ["trade_date", "symbol"]
    source_like = [column for column in ("source", "adjusted_flag") if column in data.columns]
    if source_like:
        keys.extend(source_like)
    data["_qdp_bar_time"] = data["bar_time"].astype(str).str.strip().str.zfill(9)
    opening = data.loc[data["_qdp_bar_time"].eq("093000000")].drop_duplicates(keys, keep="last").copy()
    first_continuous = data.loc[data["_qdp_bar_time"].eq("093100000")].drop_duplicates(keys, keep="last").copy()
    if opening.empty or first_continuous.empty:
        return data.drop(columns=["_qdp_original_order", "_qdp_bar_time"], errors="ignore")

    merge_columns = keys + [column for column in frame.columns if column not in keys]
    paired = first_continuous.loc[:, merge_columns].merge(
        opening.loc[:, merge_columns],
        on=keys,
        how="inner",
        suffixes=("_0931", "_0930"),
    )
    if paired.empty:
        return data.drop(columns=["_qdp_original_order", "_qdp_bar_time"], errors="ignore")

    merged = paired.loc[:, keys].copy()
    for column in frame.columns:
        if column in keys:
            continue
        c0931 = f"{column}_0931"
        c0930 = f"{column}_0930"
        if column == "bar_time":
            merged[column] = "093100000"
        elif column == "open" and c0930 in paired and c0931 in paired:
            open_0930 = pd.to_numeric(paired[c0930], errors="coerce")
            open_0931 = pd.to_numeric(paired[c0931], errors="coerce")
            merged[column] = open_0930.where(open_0930.gt(0), open_0931)
        elif column == "high" and c0930 in paired and c0931 in paired:
            high_0930 = pd.to_numeric(paired[c0930], errors="coerce")
            high_0931 = pd.to_numeric(paired[c0931], errors="coerce")
            merged[column] = pd.concat(
                [high_0930.where(high_0930.gt(0)), high_0931.where(high_0931.gt(0))],
                axis=1,
            ).max(axis=1, skipna=True)
        elif column == "low" and c0930 in paired and c0931 in paired:
            low_0930 = pd.to_numeric(paired[c0930], errors="coerce")
            low_0931 = pd.to_numeric(paired[c0931], errors="coerce")
            merged[column] = pd.concat(
                [low_0930.where(low_0930.gt(0)), low_0931.where(low_0931.gt(0))],
                axis=1,
            ).min(axis=1, skipna=True)
        elif column in {"volume", "amount", "turnover_rate"} and c0930 in paired and c0931 in paired:
            merged[column] = pd.to_numeric(paired[c0930], errors="coerce").fillna(0.0) + pd.to_numeric(paired[c0931], errors="coerce").fillna(0.0)
        elif column in {"float_share", "total_share"} and c0930 in paired and c0931 in paired:
            value_0931 = pd.to_numeric(paired[c0931], errors="coerce")
            value_0930 = pd.to_numeric(paired[c0930], errors="coerce")
            merged[column] = value_0931.where(value_0931.notna(), value_0930)
        elif c0931 in paired:
            merged[column] = paired[c0931]
    merge_keys = paired.loc[:, keys].drop_duplicates()
    merge_keys["_qdp_merge_pair"] = True
    data_with_pair = data.merge(merge_keys, on=keys, how="left")
    remove_opening = data_with_pair["_qdp_bar_time"].eq("093000000") & data_with_pair["_qdp_merge_pair"].eq(True)
    remove_first_continuous = data_with_pair["_qdp_bar_time"].eq("093100000") & data_with_pair["_qdp_merge_pair"].eq(True)
    output = data_with_pair.loc[~(remove_opening | remove_first_continuous), frame.columns].copy()
    output = pd.concat([output, merged.loc[:, frame.columns]], ignore_index=True)
    sort_columns = [column for column in ("trade_date", "symbol", "bar_time", "source") if column in output.columns]
    if sort_columns:
        output = output.sort_values(sort_columns)
    return output.reset_index(drop=True)


def _external_source_name(source_kind: str) -> str:
    return "external_quant_csv" if str(source_kind or "").lower() == "directory" else "external_quant_zip"


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
    source_path: Path,
    member_name: str,
    suffix: str = "",
) -> str:
    spec = _dataset_spec(cfg, domain=domain)
    identity_spec = {**spec, "domain": domain, "sharded": True}
    identity = lake.build_domain_dataset_identity(domain=domain, spec=identity_spec)
    shard_dir = Path(identity["dataset_dir"]) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_name = _safe_shard_name(source_path, member_name, suffix=suffix)
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
        "include_unpacked_csv": bool(cfg.include_unpacked_csv),
        "canonical_start_date": DEFAULT_CANONICAL_START_DATE,
        "raw_archive_policy": "keep_2000_2009_outside_default_research_view",
        "original_ohlcv_policy": "raw_ohlcv_never_overwritten",
        "derive_5m_from_1m": bool(cfg.derive_5m_from_1m),
        "normalize_intraday_1m_to_mootdx_240": bool(cfg.normalize_intraday_1m_to_mootdx_240),
        "one_minute_policy": (
            "mootdx_240_0930_merged_into_0931"
            if cfg.normalize_intraday_1m_to_mootdx_240
            else "store_raw_1m_bars_preserving_source_bar_time_contract"
        ),
        "one_minute_bar_count_contracts": {
            "external_quant_csv": (
                "240 bars/full trading day; 09:30 opening auction merged into 09:31"
                if cfg.normalize_intraday_1m_to_mootdx_240
                else "241 bars/full trading day when 09:30 is present"
            ),
            "external_quant_zip": (
                "normalized to mootdx 240 bars when 09:30 is present"
                if cfg.normalize_intraday_1m_to_mootdx_240
                else "source archive convention; validated per source"
            ),
            "mootdx_online": "240 bars/full trading day, starts at 09:31",
        },
        "five_minute_policy": "native_5m_preferred_reuse_existing_derived_5m_if_equivalent",
        "shard_batch_members": int(cfg.shard_batch_members),
        "shard_batch_rows": int(cfg.shard_batch_rows),
    }


def _expected_output_domains(cfg: ImportConfig) -> tuple[str, ...]:
    domains = list(cfg.domains)
    if cfg.derive_5m_from_1m and DataDomain.MARKET_INTRADAY_1M in domains:
        domains.append(DataDomain.MARKET_INTRADAY_5M)
    return tuple(dict.fromkeys(domains))


def _existing_import_result(*, lake: ResearchDataLake, cfg: ImportConfig) -> dict[str, dict[str, Any]]:
    dataset_ids: dict[str, str] = {}
    shard_counts: dict[str, int] = {}
    row_counts: dict[str, int] = {}
    error_counts: dict[str, int] = {}
    for domain in _expected_output_domains(cfg):
        spec = _dataset_spec(cfg, domain=domain)
        identity = lake.build_domain_dataset_identity(domain=domain, spec={**spec, "domain": domain, "sharded": True})
        dataset_id = str(identity.get("dataset_id", "") or "")
        if not dataset_id:
            continue
        try:
            metadata = lake.describe_dataset(dataset_id)
        except Exception:
            continue
        manifest_path = Path(str(dict(metadata.get("content_paths", {}) or {}).get("shard_manifest", "") or ""))
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        row_count = int(dict(metadata.get("row_counts", {}) or {}).get("silver_domain_data", 0) or 0)
        shards = list(manifest.get("shards", []) or [])
        dataset_ids[domain] = dataset_id
        shard_counts[domain] = int(len(shards))
        row_counts[domain] = row_count
        error_counts[domain] = int(sum(int(dict(item).get("error_count", 0) or 0) for item in shards))
    return {
        "dataset_ids": dataset_ids,
        "shard_counts": shard_counts,
        "row_counts": row_counts,
        "error_counts": error_counts,
    }


def _shard_record(
    *,
    domain: str,
    source_path: Path,
    source_kind: str,
    member_name: str,
    source_sha256: str,
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
        "source_container": str(source_path.resolve()),
        "source_container_kind": str(source_kind or ""),
        "source_zip": str(source_path.resolve()) if str(source_kind or "") == "zip" else "",
        "source_zip_sha256": str(source_sha256 or "") if str(source_kind or "") == "zip" else "",
        "source_sha256": str(source_sha256 or ""),
        "source_member": str(member_name),
        "source_member_count": int(source_member_count),
        "source_members_sample": list(source_members_sample or []),
        "error_count": int(error_count),
        "error": str(error or ""),
        "derivation": str(derivation or ""),
    }


def _source_paths_by_domain(cfg: ImportConfig) -> dict[str, list[ExternalSourcePath]]:
    paths_by_domain = {domain: [] for domain in cfg.domains}
    if not cfg.source_root.exists():
        return paths_by_domain
    start_year = pd.Timestamp(cfg.start_date).year
    for path in sorted(cfg.source_root.rglob("*.zip")):
        domain = _classify_source_path(path)
        if domain not in paths_by_domain:
            continue
        year = _year_from_path(path)
        if cfg.years and year not in cfg.years:
            continue
        if not cfg.years and year is not None and year < start_year:
            continue
        paths_by_domain[domain].append(ExternalSourcePath(path=path, kind="zip"))
    if cfg.include_unpacked_csv:
        for path in sorted((item for item in cfg.source_root.rglob("*") if item.is_dir()), key=lambda item: str(item).lower()):
            domain = _classify_source_path(path)
            if domain not in paths_by_domain:
                continue
            year = _year_from_path(path)
            if cfg.years and year not in cfg.years:
                continue
            if not cfg.years and year is not None and year < start_year:
                continue
            if not _directory_has_external_csv_members(path):
                continue
            paths_by_domain[domain].append(ExternalSourcePath(path=path, kind="directory"))
    return paths_by_domain


def _classify_source_path(path: Path) -> str:
    text = str(path).lower()
    if "1分钟" in text or "1min" in text or "1m" in text:
        return DataDomain.MARKET_INTRADAY_1M
    if "5分钟" in text or "5min" in text or "5m" in text:
        return DataDomain.MARKET_INTRADAY_5M
    if "复权" in text or "adjust" in text:
        return DataDomain.ADJUST_FACTOR
    return ""


def _directory_has_external_csv_members(path: Path) -> bool:
    try:
        for child in path.iterdir():
            if child.is_file() and child.suffix.lower() in {".csv", ".txt"}:
                return True
    except OSError:
        return False
    return False


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
            max_files_per_source=args.max_files_per_source,
            dry_run=args.dry_run,
            include_unpacked_csv=args.include_unpacked_csv,
            derive_5m_from_1m=args.derive_5m_from_1m,
            normalize_intraday_1m_to_mootdx_240=args.normalize_intraday_1m_to_mootdx_240,
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
