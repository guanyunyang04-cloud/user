from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from quant_data_platform.lake.canonical import DEFAULT_CANONICAL_START_DATE, DEFAULT_EXTERNAL_QUANT_DATA_ROOT
from quant_data_platform.lake.catalog import DEFAULT_DATA_LAKE_ROOT, ResearchDataLake
from quant_data_platform.domains.contracts import DataDomain, normalize_domain
from quant_data_platform.ingest.import_external_quant_zip import ImportConfig, _dataset_spec


@dataclass(frozen=True)
class CombineShardedDomainConfig:
    lake_root: Path = DEFAULT_DATA_LAKE_ROOT
    source_root: Path = DEFAULT_EXTERNAL_QUANT_DATA_ROOT
    domain: str = DataDomain.MARKET_INTRADAY_5M
    source_dataset_ids: tuple[str, ...] = ()
    start_date: str = DEFAULT_CANONICAL_START_DATE
    end_date: str = ""
    years: tuple[int, ...] = ()
    shard_batch_members: int = 1
    shard_batch_rows: int = 0
    derive_5m_from_1m: bool = False
    link_mode: str = "manifest"
    dry_run: bool = False
    reuse: bool = True

    def normalized(self) -> "CombineShardedDomainConfig":
        return CombineShardedDomainConfig(
            lake_root=Path(self.lake_root),
            source_root=Path(self.source_root),
            domain=normalize_domain(self.domain),
            source_dataset_ids=tuple(str(item).strip() for item in self.source_dataset_ids if str(item).strip()),
            start_date=str(self.start_date or DEFAULT_CANONICAL_START_DATE),
            end_date=str(self.end_date or ""),
            years=tuple(sorted({int(item) for item in self.years})),
            shard_batch_members=max(1, int(self.shard_batch_members or 1)),
            shard_batch_rows=max(0, int(self.shard_batch_rows or 0)),
            derive_5m_from_1m=bool(self.derive_5m_from_1m),
            link_mode=str(self.link_mode or "manifest").strip().lower(),
            dry_run=bool(self.dry_run),
            reuse=bool(self.reuse),
        )


@dataclass(frozen=True)
class CombineShardedDomainResult:
    status: str
    dataset_id: str = ""
    shard_count: int = 0
    row_count: int = 0
    linked_count: int = 0
    existing_count: int = 0
    source_dataset_ids: tuple[str, ...] = ()
    errors: list[str] = field(default_factory=list)


def combine_sharded_domain_datasets(config: CombineShardedDomainConfig) -> CombineShardedDomainResult:
    cfg = config.normalized()
    if cfg.link_mode not in {"manifest", "hardlink", "hardlink-or-copy", "copy"}:
        raise ValueError(f"unsupported link_mode: {cfg.link_mode}")
    if not cfg.source_dataset_ids:
        raise ValueError("source_dataset_ids_required")

    lake = ResearchDataLake(cfg.lake_root)
    target_spec = _target_spec(cfg)
    identity_spec = {**target_spec, "domain": cfg.domain, "sharded": True}
    identity = lake.build_domain_dataset_identity(domain=cfg.domain, spec=identity_spec)
    target_shard_dir = Path(identity["dataset_dir"]) / "shards"

    shard_records: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    linked_count = 0
    existing_count = 0
    errors: list[str] = []
    for dataset_id in cfg.source_dataset_ids:
        metadata = lake.describe_dataset(dataset_id)
        source_domain = str(dict(metadata.get("parameters", {}) or {}).get("domain", "") or "")
        if normalize_domain(source_domain) != cfg.domain:
            raise ValueError(f"source_domain_mismatch: dataset_id={dataset_id} domain={source_domain} expected={cfg.domain}")
        manifest_path = Path(str(dict(metadata.get("content_paths", {}) or {}).get("shard_manifest", "") or ""))
        if not manifest_path.exists():
            raise FileNotFoundError(f"source_shard_manifest_not_found: {dataset_id} {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for raw_record in list(manifest.get("shards", []) or []):
            record = dict(raw_record)
            if str(record.get("status", "") or "") not in {"stored", "skipped"}:
                errors.append(f"skipping_unstored_shard: dataset_id={dataset_id} source_member={record.get('source_member', '')}")
                continue
            source_path = Path(str(record.get("path", "") or ""))
            if int(record.get("row_count", 0) or 0) > 0 and not source_path.exists():
                errors.append(f"missing_source_shard: dataset_id={dataset_id} path={source_path}")
                continue
            dedupe_key = _dedupe_key(record, fallback=str(source_path.resolve()).lower())
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            target_path = (
                source_path
                if cfg.link_mode == "manifest"
                else target_shard_dir / source_path.name
                if source_path.name
                else target_shard_dir / f"empty_{len(shard_records):06d}.parquet"
            )
            materialization = "dry_run"
            if cfg.link_mode == "manifest":
                materialization = "manifest_reference"
            elif not cfg.dry_run and source_path.exists():
                materialization = _materialize_shard(source_path, target_path, link_mode=cfg.link_mode)
                if materialization == "existing":
                    existing_count += 1
                else:
                    linked_count += 1
            record["path"] = str(target_path.resolve())
            record["combined_from_dataset_id"] = str(dataset_id)
            record["combine_materialization"] = materialization
            shard_records.append(record)

    if errors:
        raise RuntimeError("; ".join(errors[:20]))

    dataset_id = ""
    if not cfg.dry_run and shard_records:
        saved = lake.save_sharded_domain_dataset(
            domain=cfg.domain,
            spec=target_spec,
            shard_records=shard_records,
            source="external_quant_zip_combined",
            reuse=cfg.reuse,
        )
        dataset_id = saved.dataset_id

    return CombineShardedDomainResult(
        status="dry_run" if cfg.dry_run else "completed",
        dataset_id=dataset_id,
        shard_count=len(shard_records),
        row_count=sum(int(item.get("row_count", 0) or 0) for item in shard_records),
        linked_count=linked_count,
        existing_count=existing_count,
        source_dataset_ids=cfg.source_dataset_ids,
        errors=errors,
    )


def _target_spec(cfg: CombineShardedDomainConfig) -> dict[str, Any]:
    import_cfg = ImportConfig(
        lake_root=cfg.lake_root,
        source_root=cfg.source_root,
        domains=(cfg.domain,),
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        years=cfg.years,
        derive_5m_from_1m=cfg.derive_5m_from_1m,
        hash_zips=False,
        shard_batch_members=cfg.shard_batch_members,
        shard_batch_rows=cfg.shard_batch_rows,
    ).normalized()
    spec = _dataset_spec(import_cfg, domain=cfg.domain)
    spec["combined_from_dataset_ids"] = list(cfg.source_dataset_ids)
    spec["combine_policy"] = "hardlink_completed_shards_no_rewrite"
    return spec


def _dedupe_key(record: Mapping[str, Any], *, fallback: str) -> str:
    values = [
        str(record.get("domain", "") or ""),
        str(record.get("source_zip", "") or ""),
        str(record.get("source_member", "") or ""),
        str(record.get("derivation", "") or ""),
    ]
    key = "|".join(values)
    return key if any(values) else fallback


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
    parser = argparse.ArgumentParser(description="Combine compatible sharded domain datasets without rewriting parquet rows.")
    parser.add_argument("--lake-root", default=str(DEFAULT_DATA_LAKE_ROOT))
    parser.add_argument("--source-root", default=str(DEFAULT_EXTERNAL_QUANT_DATA_ROOT))
    parser.add_argument("--domain", required=True)
    parser.add_argument("--source-dataset-ids", required=True, help="Comma separated source dataset ids.")
    parser.add_argument("--start-date", default=DEFAULT_CANONICAL_START_DATE)
    parser.add_argument("--end-date", default="")
    parser.add_argument("--years", required=True)
    parser.add_argument("--shard-batch-members", type=int, default=1)
    parser.add_argument("--shard-batch-rows", type=int, default=0)
    parser.add_argument("--link-mode", choices=("manifest", "hardlink", "hardlink-or-copy", "copy"), default="manifest")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse", dest="reuse", action="store_true", default=True)
    parser.add_argument("--no-reuse", dest="reuse", action="store_false")
    return parser


def _parse_csv(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(text or "").split(",") if item.strip())


def _parse_years(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in str(text or "").split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = combine_sharded_domain_datasets(
        CombineShardedDomainConfig(
            lake_root=Path(args.lake_root),
            source_root=Path(args.source_root),
            domain=args.domain,
            source_dataset_ids=_parse_csv(args.source_dataset_ids),
            start_date=args.start_date,
            end_date=args.end_date,
            years=_parse_years(args.years),
            shard_batch_members=args.shard_batch_members,
            shard_batch_rows=args.shard_batch_rows,
            link_mode=args.link_mode,
            dry_run=args.dry_run,
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
                "linked_count": result.linked_count,
                "existing_count": result.existing_count,
                "source_dataset_ids": list(result.source_dataset_ids),
                "errors": result.errors,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
