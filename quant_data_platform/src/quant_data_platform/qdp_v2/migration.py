from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from quant_data_platform.core.json_io import json_safe, read_json
from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.core.registry import load_root_manifest
from quant_data_platform.lake.catalog import ResearchDataLake
from quant_data_platform.qdp_v2.manifest import (
    ACTIVE_MANIFEST_VERSION,
    DatasetManifest,
    ShardManifestEntry,
    atomic_write_json,
    path_for_manifest,
    qdp_v2_root,
    schema_hash,
    stable_hash,
    utc_now,
    write_active_manifest,
    write_dataset_manifest,
)
from quant_data_platform.qdp_v2.runtime import resolve_runtime_profile


@dataclass(frozen=True)
class LegacyFilePlan:
    source_path: str
    target_path: str
    content_key: str
    row_count: int = 0
    start_date: str = ""
    end_date: str = ""
    source_schema_hash: str = ""
    file_size: int = 0


@dataclass(frozen=True)
class DatasetMigrationPlan:
    legacy_dataset_id: str
    target_dataset_id: str
    domain: str
    layer: str
    frequency: str
    contract_version: str
    primary_key: list[str]
    source: dict[str, Any]
    legacy_metadata: dict[str, Any]
    files: list[LegacyFilePlan] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MigrationPlan:
    status: str
    qdp_v2_root: str
    active_as_of_date: str
    dataset_count: int
    file_count: int
    total_bytes: int
    datasets: list[DatasetMigrationPlan]
    active: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["datasets"] = [asdict(item) for item in self.datasets]
        return json_safe(payload)


def build_migration_plan(
    *,
    workspace_root: str | Path | None = None,
    domains: set[str] | None = None,
    max_files: int = 0,
) -> MigrationPlan:
    paths = qdp_paths(workspace_root)
    root = qdp_v2_root(paths.workspace_root)
    legacy_root = load_root_manifest(paths)
    lake = ResearchDataLake(paths.lake_root)
    active_components = dict(legacy_root.get("canonical_component_dataset_ids", {}) or {})
    canonical_dataset_id = str(legacy_root.get("canonical_dataset_id", "") or "")
    if canonical_dataset_id and "market_daily" not in active_components:
        active_components["market_daily"] = canonical_dataset_id
    if not active_components:
        raise RuntimeError("legacy_active_components_not_found")

    include_domains = {str(item).strip() for item in domains or set() if str(item).strip()}
    dataset_plans: list[DatasetMigrationPlan] = []
    for legacy_domain, legacy_dataset_id in sorted(active_components.items()):
        if include_domains and str(legacy_domain) not in include_domains and _domain_target(str(legacy_domain), str(legacy_dataset_id)) not in include_domains:
            continue
        legacy_id = str(legacy_dataset_id or "").strip()
        if not legacy_id:
            continue
        try:
            metadata = lake.describe_dataset(legacy_id)
        except Exception as exc:
            dataset_plans.append(
                DatasetMigrationPlan(
                    legacy_dataset_id=legacy_id,
                    target_dataset_id=legacy_id,
                    domain=_domain_target(str(legacy_domain), legacy_id),
                    layer="raw",
                    frequency="",
                    contract_version="legacy_unavailable",
                    primary_key=_default_primary_key(_domain_target(str(legacy_domain), legacy_id)),
                    source={"provider": "legacy", "created_by": "qdp v2 migration", "created_at": utc_now()},
                    legacy_metadata={"error": str(exc)},
                    notes=[f"legacy_dataset_describe_failed: {exc}"],
                )
            )
            continue

        dataset_plans.extend(_dataset_plans_from_legacy(root=root, legacy_domain=str(legacy_domain), metadata=metadata, max_files=max_files))

    active = _build_active_manifest(dataset_plans, legacy_root)
    file_count = sum(len(item.files) for item in dataset_plans)
    total_bytes = sum(sum(file.file_size for file in item.files) for item in dataset_plans)
    return MigrationPlan(
        status="planned",
        qdp_v2_root=str(root.resolve()),
        active_as_of_date=str(active.get("active_as_of_date", "") or ""),
        dataset_count=len(dataset_plans),
        file_count=file_count,
        total_bytes=total_bytes,
        datasets=dataset_plans,
        active=active,
        notes=[
            "migration uses old lake catalog only to inventory legacy dataset metadata",
            "source parquet is preserved; cleanup is a separate qdp lake gc --delete --yes step",
        ],
        blockers=_plan_blockers(dataset_plans),
    )


def execute_migration_plan(
    plan: MigrationPlan,
    *,
    verify: bool,
    activate: bool,
    reuse_existing: bool = True,
) -> dict[str, Any]:
    root = Path(plan.qdp_v2_root)
    copied_files = 0
    skipped_files = 0
    verified_files = 0
    dataset_manifests: list[str] = []
    errors: list[dict[str, Any]] = []

    for dataset in plan.datasets:
        staging_dir = root / "datasets" / dataset.domain / dataset.target_dataset_id / ".staging"
        final_dir = root / "datasets" / dataset.domain / dataset.target_dataset_id
        shard_entries: list[ShardManifestEntry] = []
        schema_payload: list[dict[str, str]] = []
        for file_plan in dataset.files:
            source = Path(file_plan.source_path)
            target = Path(file_plan.target_path)
            staging_target = staging_dir / "shards" / target.name
            try:
                staging_target.parent.mkdir(parents=True, exist_ok=True)
                if reuse_existing and staging_target.exists() and staging_target.stat().st_size == source.stat().st_size:
                    skipped_files += 1
                else:
                    shutil.copy2(source, staging_target)
                    copied_files += 1
                target_schema = _parquet_schema(staging_target)
                target_rows = _parquet_row_count(staging_target)
                target_hash = schema_hash(target_schema)
                if verify:
                    _verify_copy(source, staging_target, expected_rows=file_plan.row_count, expected_schema_hash=file_plan.source_schema_hash)
                    verified_files += 1
                if not schema_payload and target_schema:
                    schema_payload = target_schema
                shard_entries.append(
                    ShardManifestEntry(
                        path=path_for_manifest(final_dir / "shards" / staging_target.name, root=root),
                        row_count=int(target_rows or file_plan.row_count),
                        start_date=file_plan.start_date,
                        end_date=file_plan.end_date,
                        status="stored",
                        file_size=int(staging_target.stat().st_size),
                        schema_hash=target_hash,
                        source_path=file_plan.source_path,
                        content_key=file_plan.content_key,
                    )
                )
            except Exception as exc:
                errors.append(
                    {
                        "dataset_id": dataset.target_dataset_id,
                        "source_path": file_plan.source_path,
                        "target_path": str(staging_target),
                        "error": str(exc),
                    }
                )
        if errors:
            continue
        _swap_staging_shards(staging_dir, final_dir)
        total_rows = int(sum(item.row_count for item in shard_entries))
        starts = [item.start_date for item in shard_entries if item.start_date]
        ends = [item.end_date for item in shard_entries if item.end_date]
        manifest = DatasetManifest(
            dataset_id=dataset.target_dataset_id,
            domain=dataset.domain,
            layer=dataset.layer,
            frequency=dataset.frequency,
            contract_version=dataset.contract_version,
            primary_key=dataset.primary_key,
            start_date=min(starts) if starts else str(dataset.legacy_metadata.get("start_date", "") or ""),
            end_date=max(ends) if ends else str(dataset.legacy_metadata.get("end_date", "") or ""),
            row_count=total_rows,
            schema_hash=schema_hash(schema_payload),
            schema=schema_payload,
            shards=shard_entries,
            source={**dataset.source, "migrated_at": utc_now()},
            quality={
                "path_refs_exist": all((root / item.path).exists() for item in shard_entries),
                "primary_key_unique": "not_checked_in_migration",
                "date_coverage_ok": "not_checked_in_migration",
                "footer_verified": bool(verify),
            },
            legacy={
                "dataset_id": dataset.legacy_dataset_id,
                "metadata": dataset.legacy_metadata,
            },
            notes=list(dataset.notes),
        )
        dataset_manifests.append(str(write_dataset_manifest(root, manifest).resolve()))

    status = "error" if errors else "migrated"
    active_path = ""
    blockers: list[str] = []
    if not errors and activate:
        blockers = list(plan.blockers)
        if blockers:
            status = "blocked"
        else:
            active_path = str(write_active_manifest(root, plan.active).resolve())
    run_payload = {
        "status": status,
        "qdp_v2_root": str(root.resolve()),
        "copied_files": copied_files,
        "skipped_files": skipped_files,
        "verified_files": verified_files,
        "dataset_manifests": dataset_manifests,
        "active_manifest": active_path,
        "errors": errors,
        "blockers": blockers,
        "source_parquet_preserved": True,
    }
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    run_id = f"migrate_v2_{utc_now().replace(':', '').replace('-', '')}"
    atomic_write_json(runs / f"{run_id}.json", run_payload)
    return run_payload


def migration_status_from_plan(plan: MigrationPlan, *, as_json: bool = True) -> dict[str, Any]:
    payload = plan.to_dict()
    if not as_json:
        payload = {
            "status": payload["status"],
            "qdp_v2_root": payload["qdp_v2_root"],
            "dataset_count": payload["dataset_count"],
            "file_count": payload["file_count"],
            "total_gb": round(int(payload["total_bytes"]) / 1024**3, 4),
            "active_as_of_date": payload["active_as_of_date"],
            "notes": payload["notes"],
            "blockers": payload.get("blockers", []),
        }
    return payload


def _dataset_plans_from_legacy(
    *,
    root: Path,
    legacy_domain: str,
    metadata: Mapping[str, Any],
    max_files: int,
) -> list[DatasetMigrationPlan]:
    legacy_dataset_id = str(metadata.get("dataset_id", "") or "")
    domain = _domain_target(legacy_domain, legacy_dataset_id)
    notes: list[str] = []
    files = _legacy_files(root=root, domain=domain, target_dataset_id=legacy_dataset_id, metadata=metadata, max_files=max_files)
    plans = [
        DatasetMigrationPlan(
            legacy_dataset_id=legacy_dataset_id,
            target_dataset_id=legacy_dataset_id,
            domain=domain,
            layer=_default_layer(domain),
            frequency=_default_frequency(domain),
            contract_version=_default_contract(domain, metadata),
            primary_key=_default_primary_key(domain),
            source={
                "provider": str(metadata.get("source", "") or "legacy"),
                "created_by": "qdp v2 migration",
                "created_at": utc_now(),
            },
            legacy_metadata=_legacy_metadata_subset(metadata),
            files=files,
            notes=notes,
        )
    ]
    if legacy_domain == "market_daily":
        raw_files = [item for item in files if item.content_key == "bronze_market_data"]
        if raw_files:
            raw_dataset_id = f"market_daily_raw__{stable_hash({'legacy_dataset_id': legacy_dataset_id, 'content': 'bronze_market_data'})}"
            raw_files = _retarget_files(root=root, domain="market_daily_raw", target_dataset_id=raw_dataset_id, files=raw_files)
            plans.append(
                DatasetMigrationPlan(
                    legacy_dataset_id=legacy_dataset_id,
                    target_dataset_id=raw_dataset_id,
                    domain="market_daily_raw",
                    layer="raw",
                    frequency="1d",
                    contract_version="qdp_v2_market_daily_raw_v1",
                    primary_key=_default_primary_key("market_daily_raw"),
                    source={
                        "provider": str(metadata.get("source", "") or "legacy"),
                        "created_by": "qdp v2 migration",
                        "created_at": utc_now(),
                    },
                    legacy_metadata=_legacy_metadata_subset(metadata),
                    files=raw_files,
                    notes=["split from legacy policy_input_bundle bronze_market_data"],
                )
            )
    return plans


def _legacy_files(
    *,
    root: Path,
    domain: str,
    target_dataset_id: str,
    metadata: Mapping[str, Any],
    max_files: int,
) -> list[LegacyFilePlan]:
    out: list[LegacyFilePlan] = []
    content_paths = dict(metadata.get("content_paths", {}) or {})
    shard_manifest = str(content_paths.get("shard_manifest", "") or "").strip()
    if shard_manifest and Path(shard_manifest).exists():
        manifest = read_json(shard_manifest)
        for index, shard in enumerate(list(manifest.get("shards", []) or [])):
            if max_files and len(out) >= max_files:
                break
            if not isinstance(shard, Mapping):
                continue
            source_path = _first_existing_path(shard, ("path", "file_path", "shard_path", "source_path", "source_feature_path"))
            if not source_path:
                continue
            out.append(_file_plan(root=root, domain=domain, target_dataset_id=target_dataset_id, source_path=source_path, content_key="shard", index=index, metadata=shard))
        return out
    index = 0
    for key, value in sorted(content_paths.items()):
        for source_path in _expand_content_path(str(value or "")):
            if max_files and len(out) >= max_files:
                break
            out.append(_file_plan(root=root, domain=domain, target_dataset_id=target_dataset_id, source_path=source_path, content_key=str(key), index=index, metadata=metadata))
            index += 1
        if max_files and len(out) >= max_files:
            break
    return out


def _file_plan(
    *,
    root: Path,
    domain: str,
    target_dataset_id: str,
    source_path: Path,
    content_key: str,
    index: int,
    metadata: Mapping[str, Any],
) -> LegacyFilePlan:
    source = Path(source_path).resolve()
    rows = int(metadata.get("row_count", 0) or 0)
    if rows <= 0:
        rows = _parquet_row_count(source)
    schema_payload = _parquet_schema(source)
    suffix = source.suffix or ".parquet"
    safe_key = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(content_key))
    target_name = f"part_{index:06d}_{safe_key}{suffix}"
    target_path = root / "datasets" / domain / target_dataset_id / "shards" / target_name
    return LegacyFilePlan(
        source_path=str(source),
        target_path=str(target_path.resolve()),
        content_key=str(content_key),
        row_count=rows,
        start_date=str(metadata.get("start_date", "") or _date_from_metadata(metadata, "start")),
        end_date=str(metadata.get("end_date", "") or _date_from_metadata(metadata, "end")),
        source_schema_hash=schema_hash(schema_payload),
        file_size=int(source.stat().st_size) if source.exists() else 0,
    )


def _retarget_files(*, root: Path, domain: str, target_dataset_id: str, files: list[LegacyFilePlan]) -> list[LegacyFilePlan]:
    out: list[LegacyFilePlan] = []
    for index, file_plan in enumerate(files):
        suffix = Path(file_plan.source_path).suffix or ".parquet"
        target_path = root / "datasets" / domain / target_dataset_id / "shards" / f"part_{index:06d}_{file_plan.content_key}{suffix}"
        out.append(
            LegacyFilePlan(
                source_path=file_plan.source_path,
                target_path=str(target_path.resolve()),
                content_key=file_plan.content_key,
                row_count=file_plan.row_count,
                start_date=file_plan.start_date,
                end_date=file_plan.end_date,
                source_schema_hash=file_plan.source_schema_hash,
                file_size=file_plan.file_size,
            )
        )
    return out


def _expand_content_path(value: str) -> list[Path]:
    text = str(value or "").strip()
    if not text:
        return []
    if "*" in text:
        return sorted(Path(item) for item in Path().glob(text) if Path(item).is_file()) if not Path(text).is_absolute() else sorted(Path(text).parent.glob(Path(text).name))
    path = Path(text)
    if path.exists() and path.is_file() and path.suffix.lower() == ".parquet":
        return [path]
    return []


def _first_existing_path(payload: Mapping[str, Any], keys: tuple[str, ...]) -> Path | None:
    for key in keys:
        text = str(payload.get(key, "") or "").strip()
        if text and Path(text).exists() and Path(text).is_file():
            return Path(text)
    return None


def _swap_staging_shards(staging_dir: Path, final_dir: Path) -> None:
    staging_shards = staging_dir / "shards"
    final_shards = final_dir / "shards"
    final_dir.mkdir(parents=True, exist_ok=True)
    if final_shards.exists():
        shutil.rmtree(final_shards)
    if staging_shards.exists():
        staging_shards.replace(final_shards)
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)


def _verify_copy(source: Path, target: Path, *, expected_rows: int, expected_schema_hash: str) -> None:
    if not target.exists():
        raise RuntimeError(f"target_missing_after_copy: {target}")
    if source.stat().st_size != target.stat().st_size:
        raise RuntimeError(f"copied_file_size_mismatch: {source} -> {target}")
    rows = _parquet_row_count(target)
    if expected_rows > 0 and int(rows) != int(expected_rows):
        raise RuntimeError(f"copied_row_count_mismatch: {target}; expected={expected_rows}; actual={rows}")
    actual_schema_hash = schema_hash(_parquet_schema(target))
    if expected_schema_hash and actual_schema_hash != expected_schema_hash:
        raise RuntimeError(f"copied_schema_hash_mismatch: {target}")


def _parquet_row_count(path: Path) -> int:
    if not path.exists() or path.suffix.lower() != ".parquet":
        return 0
    try:
        import pyarrow.parquet as pq  # type: ignore

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        try:
            import duckdb  # type: ignore

            with duckdb.connect(":memory:") as con:
                return int(con.execute("select count(*) as n from read_parquet(?)", [str(path)]).fetchone()[0])
        except Exception:
            return 0


def _parquet_schema(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.suffix.lower() != ".parquet":
        return []
    try:
        import pyarrow.parquet as pq  # type: ignore

        schema = pq.ParquetFile(path).schema_arrow
        return [{"name": str(field.name), "type": str(field.type)} for field in schema]
    except Exception:
        try:
            import duckdb  # type: ignore

            with duckdb.connect(":memory:") as con:
                rows = con.execute("describe select * from read_parquet(?)", [str(path)]).fetchall()
            return [{"name": str(row[0]), "type": str(row[1])} for row in rows]
        except Exception:
            return []


def _domain_target(legacy_domain: str, legacy_dataset_id: str) -> str:
    if legacy_domain == "market_daily" and legacy_dataset_id.startswith("policy_input_bundle__"):
        return "market_daily_panel"
    return str(legacy_domain).strip().lower()


def _default_layer(domain: str) -> str:
    if domain in {"intraday_daily_features", "v2_status_sidecar"}:
        return "derived"
    if domain == "market_daily_panel":
        return "research_panel"
    return "raw"


def _default_frequency(domain: str) -> str:
    if domain.endswith("_1m"):
        return "1m"
    if domain.endswith("_5m"):
        return "5m"
    if domain in {"market_daily", "market_daily_raw", "market_daily_panel", "intraday_daily_features", "valuation", "adjust_factor"}:
        return "1d"
    return ""


def _default_contract(domain: str, metadata: Mapping[str, Any]) -> str:
    if domain == "market_intraday_1m":
        return "mootdx_1m_240_v1"
    if domain == "market_intraday_5m":
        return "legacy_5m_pending_rebuild_to_mootdx_5m_48_v1"
    if domain == "market_daily_raw":
        return "qdp_v2_market_daily_raw_v1"
    if domain == "market_daily_panel":
        return "legacy_policy_bundle_research_panel_v1"
    if domain == "valuation":
        return "legacy_valuation_pending_qdp_v2_normalization"
    return str(dict(metadata.get("parameters", {}) or {}).get("contract_version", "") or f"qdp_v2_{domain}_v1")


def _default_primary_key(domain: str) -> list[str]:
    if domain in {"market_intraday_1m", "market_intraday_5m"}:
        return ["trade_date", "symbol", "bar_time"]
    if domain in {"market_daily", "market_daily_raw", "market_daily_panel", "intraday_daily_features", "valuation", "adjust_factor", "universe_snapshot", "security_status"}:
        return ["trade_date", "symbol"]
    if domain == "trading_calendar":
        return ["trade_date"]
    return []


def _legacy_metadata_subset(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": str(metadata.get("dataset_id", "") or ""),
        "dataset_kind": str(metadata.get("dataset_kind", "") or ""),
        "domain": str(metadata.get("domain", "") or ""),
        "source": str(metadata.get("source", "") or ""),
        "start_date": str(metadata.get("start_date", "") or ""),
        "end_date": str(metadata.get("end_date", "") or ""),
        "row_counts": dict(metadata.get("row_counts", {}) or {}),
        "parameters": dict(metadata.get("parameters", {}) or {}),
    }


def _date_from_metadata(metadata: Mapping[str, Any], bound: str) -> str:
    bounds = dict(metadata.get("_date_bounds", {}) or {})
    key = "start_date" if bound == "start" else "end_date"
    return str(bounds.get(key, "") or dict(metadata.get("parameters", {}) or {}).get(key, "") or metadata.get(key, "") or "")


def _build_active_manifest(dataset_plans: list[DatasetMigrationPlan], legacy_root: Mapping[str, Any]) -> dict[str, Any]:
    raw: dict[str, str] = {}
    derived: dict[str, str] = {}
    research_panels: dict[str, str] = {}
    for plan in dataset_plans:
        if plan.domain in {"intraday_daily_features", "v2_status_sidecar"}:
            derived[plan.domain] = plan.target_dataset_id
        elif plan.domain == "market_daily_panel":
            research_panels[plan.domain] = plan.target_dataset_id
        else:
            raw[plan.domain] = plan.target_dataset_id
    as_of = str(legacy_root.get("canonical_dataset_end_date", "") or legacy_root.get("as_of_date", "") or "")
    if not as_of:
        ends = [str(plan.legacy_metadata.get("end_date", "") or "") for plan in dataset_plans if str(plan.legacy_metadata.get("end_date", "") or "")]
        as_of = max(ends) if ends else ""
    return {
        "version": ACTIVE_MANIFEST_VERSION,
        "active_as_of_date": as_of,
        "raw": raw,
        "derived": derived,
        "research_panels": research_panels,
        "memmap": {
            "active_manifest": "",
            "status": "not_part_of_data_base",
        },
        "source": {
            "created_by": "qdp migrate-v2",
            "created_at": utc_now(),
            "legacy_canonical_dataset_id": str(legacy_root.get("canonical_dataset_id", "") or ""),
        },
    }


def _plan_blockers(dataset_plans: list[DatasetMigrationPlan]) -> list[str]:
    blockers: list[str] = []
    for plan in dataset_plans:
        contract = str(plan.contract_version or "")
        if "pending_rebuild" in contract or "pending_qdp_v2_normalization" in contract:
            blockers.append(f"{plan.domain}:{plan.target_dataset_id}:{contract}")
    return blockers


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp migrate-v2", description="Migrate legacy QDP parquet into manifest-first qdp_v2 layout.")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--runtime", default="balanced", choices=("safe", "balanced", "fast"))
    parser.add_argument("--move", action="store_true", help="Copy legacy parquet into qdp_v2 and activate when verification passes. Source is preserved.")
    parser.add_argument("--verify", action="store_true", help="Verify copied parquet size, row count, and schema hash.")
    parser.add_argument("--activate", action="store_true", default=True)
    parser.add_argument("--no-activate", dest="activate", action="store_false")
    parser.add_argument("--domains", default="", help="Comma-separated legacy or v2 domains to migrate.")
    parser.add_argument("--max-files", type=int, default=0, help="Test limiter; 0 means no limit.")
    parser.add_argument("--no-reuse-existing", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    profile = resolve_runtime_profile(str(args.runtime or "balanced"))
    domains = {item.strip() for item in str(args.domains or "").split(",") if item.strip()}
    plan = build_migration_plan(workspace_root=str(args.workspace_root or "") or None, domains=domains or None, max_files=int(args.max_files or 0))
    if not bool(args.move):
        payload = migration_status_from_plan(plan, as_json=bool(args.json))
        payload["runtime"] = asdict(profile)
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
        return 0
    activation_allowed = bool(args.activate) and int(args.max_files or 0) <= 0
    payload = execute_migration_plan(
        plan,
        verify=bool(args.verify),
        activate=activation_allowed,
        reuse_existing=not bool(args.no_reuse_existing),
    )
    if bool(args.activate) and not activation_allowed:
        payload["activation_skipped"] = "max_files_limiter_requires_full_migration"
    payload["runtime"] = asdict(profile)
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    return 0 if payload.get("status") == "migrated" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
