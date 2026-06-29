from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quant_data_platform.lake.canonical import DEFAULT_CANONICAL_START_DATE, DEFAULT_EXTERNAL_QUANT_DATA_ROOT
from quant_data_platform.lake.catalog import DEFAULT_DATA_LAKE_ROOT, ResearchDataLake
from quant_data_platform.domains.contracts import DataDomain, normalize_domain
from quant_data_platform.ingest.import_external_quant_zip import (
    ImportConfig,
    _dataset_spec,
    _safe_shard_name,
    _year_from_path,
)


DEFAULT_RECOVERY_DOMAINS = (DataDomain.MARKET_INTRADAY_5M,)


@dataclass(frozen=True)
class RecoverExternalImportConfig:
    lake_root: Path = DEFAULT_DATA_LAKE_ROOT
    source_root: Path = DEFAULT_EXTERNAL_QUANT_DATA_ROOT
    progress_path: Path = Path()
    domains: tuple[str, ...] = DEFAULT_RECOVERY_DOMAINS
    start_date: str = DEFAULT_CANONICAL_START_DATE
    end_date: str = ""
    years: tuple[int, ...] = ()
    source_years: tuple[int, ...] = ()
    shard_batch_members: int = 1
    shard_batch_rows: int = 0
    derive_5m_from_1m: bool = False
    normalize_intraday_1m_to_mootdx_240: bool = True
    link_mode: str = "manifest"
    dry_run: bool = False
    reuse: bool = True

    def normalized(self) -> "RecoverExternalImportConfig":
        return RecoverExternalImportConfig(
            lake_root=Path(self.lake_root),
            source_root=Path(self.source_root),
            progress_path=Path(self.progress_path),
            domains=tuple(normalize_domain(item) for item in self.domains),
            start_date=str(self.start_date or DEFAULT_CANONICAL_START_DATE),
            end_date=str(self.end_date or ""),
            years=tuple(sorted({int(item) for item in self.years})),
            source_years=tuple(sorted({int(item) for item in self.source_years})),
            shard_batch_members=max(1, int(self.shard_batch_members or 1)),
            shard_batch_rows=max(0, int(self.shard_batch_rows or 0)),
            derive_5m_from_1m=bool(self.derive_5m_from_1m),
            normalize_intraday_1m_to_mootdx_240=bool(self.normalize_intraday_1m_to_mootdx_240),
            link_mode=str(self.link_mode or "manifest").strip().lower(),
            dry_run=bool(self.dry_run),
            reuse=bool(self.reuse),
        )


@dataclass(frozen=True)
class RecoverExternalImportResult:
    status: str
    dataset_ids: dict[str, str] = field(default_factory=dict)
    shard_counts: dict[str, int] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    linked_counts: dict[str, int] = field(default_factory=dict)
    existing_counts: dict[str, int] = field(default_factory=dict)
    completed_years: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def recover_import(config: RecoverExternalImportConfig) -> RecoverExternalImportResult:
    cfg = config.normalized()
    if cfg.link_mode not in {"manifest", "hardlink", "hardlink-or-copy", "copy"}:
        raise ValueError(f"unsupported link_mode: {cfg.link_mode}")
    if not cfg.progress_path.exists():
        raise FileNotFoundError(f"progress_path_not_found: {cfg.progress_path}")

    lake = ResearchDataLake(cfg.lake_root)
    events = _read_progress_events(cfg.progress_path)
    complete_zip_paths = _completed_zip_paths(events, years=cfg.years)
    completed_years = sorted({year for path in complete_zip_paths if (year := _year_from_path(path)) is not None})
    if cfg.years:
        missing_years = sorted(set(cfg.years).difference(completed_years))
        if missing_years:
            raise RuntimeError(f"recovery_blocked_incomplete_years: {missing_years}")

    source_import_cfg = ImportConfig(
        lake_root=cfg.lake_root,
        source_root=cfg.source_root,
        domains=cfg.domains,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        years=cfg.source_years,
        derive_5m_from_1m=cfg.derive_5m_from_1m,
        normalize_intraday_1m_to_mootdx_240=bool(cfg.normalize_intraday_1m_to_mootdx_240),
        hash_zips=False,
        shard_batch_members=cfg.shard_batch_members,
        shard_batch_rows=cfg.shard_batch_rows,
    ).normalized()
    target_import_cfg = ImportConfig(
        lake_root=cfg.lake_root,
        source_root=cfg.source_root,
        domains=cfg.domains,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        years=cfg.years,
        derive_5m_from_1m=cfg.derive_5m_from_1m,
        normalize_intraday_1m_to_mootdx_240=bool(cfg.normalize_intraday_1m_to_mootdx_240),
        hash_zips=False,
        shard_batch_members=cfg.shard_batch_members,
        shard_batch_rows=cfg.shard_batch_rows,
    ).normalized()

    shard_records_by_domain: dict[str, list[dict[str, Any]]] = {domain: [] for domain in cfg.domains}
    linked_counts = {domain: 0 for domain in cfg.domains}
    existing_counts = {domain: 0 for domain in cfg.domains}
    errors: list[str] = []
    complete_zip_keys = {str(path.resolve()).lower() for path in complete_zip_paths}

    for event in events:
        if str(event.get("event", "")) != "batch_stored":
            continue
        domain = normalize_domain(str(event.get("domain", "") or ""))
        if domain not in shard_records_by_domain:
            continue
        zip_path = Path(str(event.get("zip_path", "") or ""))
        if str(zip_path.resolve()).lower() not in complete_zip_keys:
            continue
        batch = str(event.get("batch", "") or "")
        if not batch:
            continue
        source_path = _shard_path_for_event(
            lake=lake,
            import_cfg=source_import_cfg,
            domain=domain,
            zip_path=zip_path,
            batch=batch,
            derivation=str(event.get("derivation", "") or ""),
        )
        target_path = (
            source_path
            if cfg.link_mode == "manifest"
            else _shard_path_for_event(
                lake=lake,
                import_cfg=target_import_cfg,
                domain=domain,
                zip_path=zip_path,
                batch=batch,
                derivation=str(event.get("derivation", "") or ""),
            )
        )
        if not source_path.exists():
            errors.append(f"missing_source_shard: {source_path}")
            continue
        status = "dry_run"
        if cfg.link_mode == "manifest":
            status = "manifest_reference"
        elif not cfg.dry_run:
            status = _materialize_shard(source_path, target_path, link_mode=cfg.link_mode)
            if status == "existing":
                existing_counts[domain] += 1
            else:
                linked_counts[domain] += 1
        year = _year_from_path(zip_path)
        shard_records_by_domain[domain].append(
            {
                "domain": domain,
                "status": "stored",
                "path": str(target_path.resolve()),
                "row_count": int(event.get("row_count", 0) or 0),
                "start_date": f"{year}-01-01" if year else "",
                "end_date": f"{year}-12-31" if year else "",
                "symbol_count": 0,
                "source_zip": str(zip_path.resolve()),
                "source_zip_sha256": "",
                "source_member": batch,
                "source_member_count": int(event.get("source_member_count", 0) or 0),
                "source_members_sample": [],
                "error_count": 0,
                "error": "",
                "derivation": str(event.get("derivation", "") or ""),
                "recovery_materialization": status,
                "recovered_from_progress_path": str(cfg.progress_path.resolve()),
            }
        )

    if errors:
        raise RuntimeError("; ".join(errors[:20]))

    dataset_ids: dict[str, str] = {}
    shard_counts: dict[str, int] = {}
    row_counts: dict[str, int] = {}
    if not cfg.dry_run:
        for domain, records in shard_records_by_domain.items():
            if not records:
                continue
            record = lake.save_sharded_domain_dataset(
                domain=domain,
                spec=_dataset_spec(target_import_cfg, domain=domain),
                shard_records=records,
                source="external_quant_zip_recovery",
                reuse=cfg.reuse,
            )
            dataset_ids[domain] = record.dataset_id
            shard_counts[domain] = len(records)
            row_counts[domain] = sum(int(item.get("row_count", 0) or 0) for item in records)
    else:
        for domain, records in shard_records_by_domain.items():
            shard_counts[domain] = len(records)
            row_counts[domain] = sum(int(item.get("row_count", 0) or 0) for item in records)

    return RecoverExternalImportResult(
        status="dry_run" if cfg.dry_run else "completed",
        dataset_ids=dataset_ids,
        shard_counts=shard_counts,
        row_counts=row_counts,
        linked_counts=linked_counts,
        existing_counts=existing_counts,
        completed_years=completed_years,
        errors=errors,
    )


def _read_progress_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _completed_zip_paths(events: list[dict[str, Any]], *, years: tuple[int, ...]) -> list[Path]:
    wanted = set(years)
    paths: dict[str, Path] = {}
    for event in events:
        if str(event.get("event", "")) != "zip_done":
            continue
        path = Path(str(event.get("zip_path", "") or ""))
        if not str(path):
            continue
        year = _year_from_path(path)
        if wanted and year not in wanted:
            continue
        paths[str(path.resolve()).lower()] = path
    return sorted(paths.values(), key=lambda item: str(item))


def _shard_path_for_event(
    *,
    lake: ResearchDataLake,
    import_cfg: ImportConfig,
    domain: str,
    zip_path: Path,
    batch: str,
    derivation: str,
) -> Path:
    spec = _dataset_spec(import_cfg, domain=domain)
    identity_spec = {**spec, "domain": domain, "sharded": True}
    identity = lake.build_domain_dataset_identity(domain=domain, spec=identity_spec)
    suffix = "derived_5m" if derivation else ""
    return Path(identity["dataset_dir"]) / "shards" / f"{_safe_shard_name(zip_path, batch, suffix=suffix)}.parquet"


def _materialize_shard(source_path: Path, target_path: Path, *, link_mode: str) -> str:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        if target_path.stat().st_size == source_path.stat().st_size:
            return "existing"
        raise RuntimeError(f"target_shard_size_mismatch: {target_path}")
    if link_mode in {"hardlink", "hardlink-or-copy"}:
        try:
            os.link(source_path, target_path)
            return "hardlink"
        except OSError:
            if link_mode == "hardlink":
                raise
    shutil.copy2(source_path, target_path)
    return "copy"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recover completed external quant zip shards from a progress JSONL.")
    parser.add_argument("--lake-root", default=str(DEFAULT_DATA_LAKE_ROOT))
    parser.add_argument("--source-root", default=str(DEFAULT_EXTERNAL_QUANT_DATA_ROOT))
    parser.add_argument("--progress-path", required=True)
    parser.add_argument("--domains", default="market_intraday_5m")
    parser.add_argument("--start-date", default=DEFAULT_CANONICAL_START_DATE)
    parser.add_argument("--end-date", default="")
    parser.add_argument("--years", required=True, help="Comma separated completed years to recover.")
    parser.add_argument("--source-years", default="", help="Original import --years value. Empty matches the full-run spec.")
    parser.add_argument("--shard-batch-members", type=int, default=1)
    parser.add_argument("--shard-batch-rows", type=int, default=0)
    parser.add_argument("--normalize-1m-to-mootdx-240", dest="normalize_intraday_1m_to_mootdx_240", action="store_true", default=True)
    parser.add_argument("--preserve-source-1m-bars", dest="normalize_intraday_1m_to_mootdx_240", action="store_false")
    parser.add_argument("--link-mode", choices=("manifest", "hardlink", "hardlink-or-copy", "copy"), default="manifest")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse", dest="reuse", action="store_true", default=True)
    parser.add_argument("--no-reuse", dest="reuse", action="store_false")
    return parser


def _parse_domains(text: str) -> tuple[str, ...]:
    values = [item.strip() for item in str(text or "").split(",") if item.strip()]
    return tuple(normalize_domain(item) for item in values) if values else DEFAULT_RECOVERY_DOMAINS


def _parse_years(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in str(text or "").split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = recover_import(
        RecoverExternalImportConfig(
            lake_root=Path(args.lake_root),
            source_root=Path(args.source_root),
            progress_path=Path(args.progress_path),
            domains=_parse_domains(args.domains),
            start_date=args.start_date,
            end_date=args.end_date,
            years=_parse_years(args.years),
            source_years=_parse_years(args.source_years),
            shard_batch_members=args.shard_batch_members,
            shard_batch_rows=args.shard_batch_rows,
            normalize_intraday_1m_to_mootdx_240=args.normalize_intraday_1m_to_mootdx_240,
            link_mode=args.link_mode,
            dry_run=args.dry_run,
            reuse=args.reuse,
        )
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "dataset_ids": result.dataset_ids,
                "shard_counts": result.shard_counts,
                "row_counts": result.row_counts,
                "linked_counts": result.linked_counts,
                "existing_counts": result.existing_counts,
                "completed_years": result.completed_years,
                "errors": result.errors,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
