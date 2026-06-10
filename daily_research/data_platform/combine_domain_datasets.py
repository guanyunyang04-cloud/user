from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from daily_research.data_lake.canonical import DEFAULT_CANONICAL_START_DATE
from daily_research.data_lake.catalog import DEFAULT_DATA_LAKE_ROOT, ResearchDataLake
from daily_research.data_platform.contracts import DataDomain, normalize_domain


@dataclass(frozen=True)
class CombineDomainDatasetsConfig:
    lake_root: Path = DEFAULT_DATA_LAKE_ROOT
    domain: str = DataDomain.MARKET_DAILY
    source_dataset_ids: tuple[str, ...] = ()
    start_date: str = DEFAULT_CANONICAL_START_DATE
    end_date: str = ""
    source: str = "canonical_reuse_combined"
    merge_policy: str = "manifest_reference_no_row_rewrite"
    allow_policy_bundle_market: bool = True
    dry_run: bool = False
    reuse: bool = True

    def normalized(self) -> "CombineDomainDatasetsConfig":
        return CombineDomainDatasetsConfig(
            lake_root=Path(self.lake_root),
            domain=normalize_domain(self.domain),
            source_dataset_ids=tuple(str(item).strip() for item in self.source_dataset_ids if str(item).strip()),
            start_date=str(self.start_date or DEFAULT_CANONICAL_START_DATE),
            end_date=str(self.end_date or ""),
            source=str(self.source or "canonical_reuse_combined"),
            merge_policy=str(self.merge_policy or "manifest_reference_no_row_rewrite"),
            allow_policy_bundle_market=bool(self.allow_policy_bundle_market),
            dry_run=bool(self.dry_run),
            reuse=bool(self.reuse),
        )


@dataclass(frozen=True)
class CombineDomainDatasetsResult:
    status: str
    dataset_id: str = ""
    shard_count: int = 0
    row_count: int = 0
    source_dataset_ids: tuple[str, ...] = ()
    source_summaries: list[dict[str, Any]] = field(default_factory=list)


def combine_domain_datasets(config: CombineDomainDatasetsConfig) -> CombineDomainDatasetsResult:
    cfg = config.normalized()
    if not cfg.source_dataset_ids:
        raise ValueError("source_dataset_ids_required")
    lake = ResearchDataLake(cfg.lake_root)
    shard_records: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    for dataset_id in cfg.source_dataset_ids:
        metadata = lake.describe_dataset(dataset_id)
        records, summary = _source_shard_records(lake=lake, cfg=cfg, metadata=metadata)
        shard_records.extend(records)
        source_summaries.append(summary)
    shard_records = _dedupe_shards(shard_records)
    row_count = sum(int(item.get("row_count", 0) or 0) for item in shard_records)
    if cfg.dry_run:
        return CombineDomainDatasetsResult(
            status="dry_run",
            shard_count=len(shard_records),
            row_count=row_count,
            source_dataset_ids=cfg.source_dataset_ids,
            source_summaries=source_summaries,
        )
    record = lake.save_sharded_domain_dataset(
        domain=cfg.domain,
        spec=_target_spec(cfg, source_summaries=source_summaries),
        shard_records=shard_records,
        source=cfg.source,
        reuse=cfg.reuse,
    )
    return CombineDomainDatasetsResult(
        status="completed",
        dataset_id=record.dataset_id,
        shard_count=len(shard_records),
        row_count=row_count,
        source_dataset_ids=cfg.source_dataset_ids,
        source_summaries=source_summaries,
    )


def _source_shard_records(
    *,
    lake: ResearchDataLake,
    cfg: CombineDomainDatasetsConfig,
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_id = str(metadata.get("dataset_id", "") or "")
    dataset_kind = str(metadata.get("dataset_kind", "") or "")
    params = dict(metadata.get("parameters", {}) or {})
    paths = dict(metadata.get("content_paths", {}) or {})
    source_domain = str(params.get("domain", "") or "")
    if dataset_kind == "policy_input_bundle" and cfg.domain == DataDomain.MARKET_DAILY and cfg.allow_policy_bundle_market:
        path_key = "bronze_market_data"
    else:
        normalized_source_domain = normalize_domain(source_domain or cfg.domain)
        if normalized_source_domain != cfg.domain:
            raise ValueError(f"source_domain_mismatch: dataset_id={dataset_id} domain={source_domain} expected={cfg.domain}")
        path_key = "silver_domain_data"

    manifest_raw = str(paths.get("shard_manifest", "") or "")
    manifest_path = Path(manifest_raw) if manifest_raw else Path()
    records: list[dict[str, Any]]
    if manifest_raw and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = [
            _manifest_record_to_combined_record(raw_record=dict(item), dataset_id=dataset_id)
            for item in list(manifest.get("shards", []) or [])
            if str(dict(item).get("status", "") or "") in {"stored", "skipped"}
        ]
    else:
        records = _path_records(metadata=metadata, path_key=path_key, dataset_id=dataset_id, domain=cfg.domain)

    summary = {
        "dataset_id": dataset_id,
        "dataset_kind": dataset_kind,
        "start_date": str(metadata.get("start_date", "") or ""),
        "end_date": str(metadata.get("end_date", "") or ""),
        "source": str(metadata.get("source", "") or ""),
        "path_key": path_key,
        "shard_count": len(records),
        "row_count": sum(int(item.get("row_count", 0) or 0) for item in records),
    }
    return records, summary


def _manifest_record_to_combined_record(*, raw_record: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    out = dict(raw_record)
    out["combined_from_dataset_id"] = dataset_id
    out["combine_materialization"] = "manifest_reference"
    return out


def _path_records(*, metadata: dict[str, Any], path_key: str, dataset_id: str, domain: str) -> list[dict[str, Any]]:
    paths = dict(metadata.get("content_paths", {}) or {})
    row_counts = dict(metadata.get("row_counts", {}) or {})
    raw = str(paths.get(path_key, "") or "")
    if not raw:
        return []
    candidates = sorted(glob.glob(raw)) if "*" in raw else [raw]
    existing = [Path(path) for path in candidates if Path(path).exists()]
    if not existing:
        return []
    total_rows = int(row_counts.get(path_key, row_counts.get("silver_domain_data", row_counts.get("bronze_market_data", 0))) or 0)
    rows_per_path = total_rows // len(existing) if total_rows and existing else 0
    records: list[dict[str, Any]] = []
    for idx, path in enumerate(existing, start=1):
        row_count = rows_per_path
        if idx == len(existing) and total_rows:
            row_count = total_rows - rows_per_path * (len(existing) - 1)
        records.append(
            {
                "domain": domain,
                "status": "stored",
                "path": str(path.resolve()),
                "row_count": row_count,
                "start_date": str(metadata.get("start_date", "") or ""),
                "end_date": str(metadata.get("end_date", "") or ""),
                "symbol_count": int(dict(metadata.get("parameters", {}) or {}).get("symbol_count", 0) or 0),
                "source_dataset_id": dataset_id,
                "source_dataset_kind": str(metadata.get("dataset_kind", "") or ""),
                "source_path_key": path_key,
                "source_member": path.name,
                "error_count": 0,
                "error": "",
                "combined_from_dataset_id": dataset_id,
                "combine_materialization": "manifest_reference",
            }
        )
    return records


def _dedupe_shards(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        key = "|".join(
            [
                str(record.get("path", "") or ""),
                str(record.get("source_zip", "") or ""),
                str(record.get("source_member", "") or ""),
                str(record.get("combined_from_dataset_id", "") or ""),
            ]
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def _target_spec(cfg: CombineDomainDatasetsConfig, *, source_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dataset": f"data_platform_{cfg.domain}",
        "source": cfg.source,
        "domain": cfg.domain,
        "provider_plan": "baostock_first_reuse",
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "canonical_start_date": DEFAULT_CANONICAL_START_DATE,
        "source_dataset_ids": list(cfg.source_dataset_ids),
        "source_summaries": source_summaries,
        "merge_policy": cfg.merge_policy,
        "sharded": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine existing domain datasets into a manifest-only sharded dataset.")
    parser.add_argument("--lake-root", default=str(DEFAULT_DATA_LAKE_ROOT))
    parser.add_argument("--domain", required=True)
    parser.add_argument("--source-dataset-ids", required=True)
    parser.add_argument("--start-date", default=DEFAULT_CANONICAL_START_DATE)
    parser.add_argument("--end-date", default="")
    parser.add_argument("--source", default="canonical_reuse_combined")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse", dest="reuse", action="store_true", default=True)
    parser.add_argument("--no-reuse", dest="reuse", action="store_false")
    return parser


def _parse_csv(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(text or "").split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = combine_domain_datasets(
        CombineDomainDatasetsConfig(
            lake_root=Path(args.lake_root),
            domain=args.domain,
            source_dataset_ids=_parse_csv(args.source_dataset_ids),
            start_date=args.start_date,
            end_date=args.end_date,
            source=args.source,
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
                "source_dataset_ids": list(result.source_dataset_ids),
                "source_summaries": result.source_summaries,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
