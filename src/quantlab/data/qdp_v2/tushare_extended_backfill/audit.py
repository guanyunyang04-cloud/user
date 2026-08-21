"""Tushare Extended Backfill: audit responsibilities."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from quantlab.data.domains.contracts import DataDomain
from quantlab.data.qdp_v2.manifest import (
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    utc_now,
)
from quantlab.data.qdp_v2.provider_credentials import (
    tushare_credential_values,
    tushare_provider_status,
)
from quantlab.data.qdp_v2.status import active_dataset_map

from .config import (
    END_DATE,
    EXPECTED_FACTOR_FIELD_COUNT,
    EXPECTED_FACTOR_SCHEMA_HASH,
    SPECS,
    UPDATE_ID,
    EndpointSpec,
    TushareExtendedBackfillError,
)
from .context import (
    _assert_credential_free,
    _read_state,
    _runtime,
    _workspace,
)
from .prepare import (
    _validate_prepared,
    _year_page_paths,
)


def _credential_file_hits(workspace: Path) -> list[str]:
    secrets = tuple(value.encode() for value in tushare_credential_values(workspace) if value)
    if not secrets:
        return []
    candidates = [Path(__file__).resolve()]
    candidates.extend(
        path for path in _runtime(workspace).rglob("*") if path.is_file() and path.suffix.lower() != ".parquet"
    )
    hits: list[str] = []
    for path in candidates:
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        if any(secret in payload for secret in secrets):
            hits.append(str(path))
    return sorted(set(hits))


def _physical_domain_audit(workspace: Path, spec: EndpointSpec, dataset_id: str) -> dict[str, Any]:
    root = qdp_v2_root(workspace)
    manifest_path = dataset_manifest_for_id(root, dataset_id, spec.domain)
    if manifest_path is None:
        raise TushareExtendedBackfillError(f"installed_manifest_missing:{spec.domain}:{dataset_id}")
    manifest = read_dataset_manifest(manifest_path)
    records: list[dict[str, Any]] = []
    for shard in manifest.shards:
        path = resolve_manifest_path(shard.path, root=root)
        year = int(dict(shard.metadata or {}).get("year") or str(shard.start_date)[:4])
        records.append(_validate_prepared(path, spec, year))
    row_count = sum(int(item["row_count"]) for item in records)
    if row_count != int(manifest.row_count):
        raise TushareExtendedBackfillError(
            f"installed_row_count_mismatch:{spec.domain}:{row_count}:{manifest.row_count}"
        )
    return {
        "dataset_id": dataset_id,
        "row_count": row_count,
        "shard_count": len(records),
        "start_date": min(
            (str(item["start_date"]) for item in records if item["start_date"]),
            default="",
        ),
        "end_date": max(
            (str(item["end_date"]) for item in records if item["end_date"]),
            default="",
        ),
        "primary_key_unique": all(item["primary_key_unique"] for item in records),
        "forbidden_2026_rows": sum(int(item["forbidden_2026_rows"]) for item in records),
        "availability_before_source_rows": sum(int(item["availability_before_source_rows"]) for item in records),
        "burn_in_flag_mismatch_rows": sum(int(item["burn_in_flag_mismatch_rows"]) for item in records),
        "disallowed_negative_value_count": sum(int(item["disallowed_negative_value_count"]) for item in records),
        "allowed_negative_net_value_count": sum(int(item["allowed_negative_net_value_count"]) for item in records),
        "provider_net_amount_unreconciled_count": sum(
            int(item["moneyflow_main_net_relation_mismatch_count"]) for item in records
        ),
        "source_exception_negative_value_count": sum(
            int(item["source_exception_negative_value_count"]) for item in records
        ),
        "yearly_row_count": {str(item["year"]): int(item["row_count"]) for item in records},
    }


def _legacy_year_cache_summary(workspace: Path, specs: Sequence[EndpointSpec]) -> dict[str, Any]:
    endpoints: dict[str, Any] = {}
    for spec in specs:
        if spec.mode != "trade_date":
            continue
        paths: list[Path] = []
        for year in range(2010, 2026):
            root = _runtime(workspace) / "raw" / spec.name / f"year={year}" / f"task={year}"
            paths.extend(sorted(root.glob("offset=*.parquet")))
        endpoints[spec.name] = {
            "file_count": len(paths),
            "row_count": sum(int(pq.ParquetFile(path).metadata.num_rows) for path in paths),
            "excluded_from_prepared_inventory": all(
                path not in _year_page_paths(workspace, spec, int(path.parts[-3][5:])) for path in paths
            ),
        }
    return endpoints


def audit(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    installed = dict(state.get("installed_domains", {}) or {})
    root = qdp_v2_root(workspace)
    active = active_dataset_map(read_active_manifest(root))
    mandatory_domains = [spec.domain for spec in SPECS.values() if not spec.optional]
    physical_domains: dict[str, Any] = {}
    for spec in SPECS.values():
        record = dict(installed.get(spec.domain, {}) or {})
        if record.get("status") != "installed":
            continue
        physical_domains[spec.domain] = _physical_domain_audit(workspace, spec, str(record["dataset_id"]))
    credential_hits = _credential_file_hits(workspace)
    provider_profile = tushare_provider_status(workspace)
    query_granularity: dict[str, str] = {}
    for spec in SPECS.values():
        record = dict(installed.get(spec.domain, {}) or {})
        if record.get("status") != "installed":
            continue
        manifest_path = dataset_manifest_for_id(root, str(record["dataset_id"]), spec.domain)
        if manifest_path is None:
            query_granularity[spec.domain] = ""
            continue
        manifest = read_dataset_manifest(manifest_path).to_dict()
        query_granularity[spec.domain] = str(dict(manifest.get("source", {}) or {}).get("query_granularity", ""))
    expected_query_granularity = {spec.domain: spec.mode for spec in SPECS.values() if not spec.optional}
    legacy_year_cache = _legacy_year_cache_summary(workspace, list(SPECS.values()))
    checks = {
        "applied": state.get("status") == "applied",
        "mandatory_domains_installed": all(
            dict(installed.get(domain, {}) or {}).get("status") == "installed" for domain in mandatory_domains
        ),
        "active_manifest_matches": all(
            active.get(domain) == dict(installed.get(domain, {}) or {}).get("dataset_id")
            for domain in mandatory_domains
        ),
        "factor_field_count_exact": dict(
            dict(installed.get(DataDomain.STK_FACTOR_PRO_RAW, {}) or {}).get("quality", {}) or {}
        ).get("provider_field_count")
        == EXPECTED_FACTOR_FIELD_COUNT,
        "factor_schema_hash_exact": dict(
            dict(installed.get(DataDomain.STK_FACTOR_PRO_RAW, {}) or {}).get("quality", {}) or {}
        ).get("provider_schema_hash")
        == EXPECTED_FACTOR_SCHEMA_HASH,
        "query_granularity_exact": query_granularity == expected_query_granularity,
        "legacy_year_range_cache_excluded": all(
            bool(record["excluded_from_prepared_inventory"]) for record in legacy_year_cache.values()
        ),
        "forbidden_2026_rows": all(
            str(dict(installed.get(domain, {}) or {}).get("end_date", "")) <= END_DATE for domain in mandatory_domains
        )
        and all(int(record["forbidden_2026_rows"]) == 0 for record in physical_domains.values()),
        "physical_primary_keys_unique": all(bool(record["primary_key_unique"]) for record in physical_domains.values()),
        "physical_pit_dates_valid": all(
            int(record["availability_before_source_rows"]) == 0 and int(record["burn_in_flag_mismatch_rows"]) == 0
            for record in physical_domains.values()
        ),
        "physical_nonnegative_fields_valid": all(
            int(record["disallowed_negative_value_count"]) == 0 for record in physical_domains.values()
        ),
        "credential_not_persisted": not credential_hits,
        "credential_not_persisted_outside_private_profile": not credential_hits,
        "training_not_performed": state.get("training_performed") is False,
    }
    status_value = "ok" if all(checks.values()) else "error"
    result = {
        "status": status_value,
        "update_id": UPDATE_ID,
        "checks": checks,
        "domains": installed,
        "physical_domains": physical_domains,
        "credential_file_hits": credential_hits,
        "provider_profile": provider_profile,
        "query_granularity": query_granularity,
        "legacy_year_range_cache": legacy_year_cache,
    }
    _assert_credential_free(result)
    atomic_write_json(_runtime(workspace) / "evaluation.json", result)
    if status_value != "ok":
        raise TushareExtendedBackfillError(f"extended_backfill_audit_failed:{checks}")
    return result


def _absent_prepared_cache(prepared_root: Path) -> dict[str, Any]:
    return {
        "status": "absent",
        "prepared_root": str(prepared_root),
        "deletable": True,
        "bytes": 0,
        "file_count": 0,
        "verified_domains": [],
        "blockers": [],
        "destructive_actions_performed": False,
    }


def _prepared_year(path: Path) -> int | None:
    try:
        return int(path.parent.name.removeprefix("year="))
    except ValueError:
        return None


def _verify_prepared_domain(
    domain_dir: Path,
    *,
    installed: dict[str, Any],
    active: dict[str, str],
    qdp_root: Path,
) -> tuple[dict[str, Any] | None, list[str], set[Path]]:
    domain = domain_dir.name
    record = dict(installed.get(domain, {}) or {})
    dataset_id = str(record.get("dataset_id", "") or "")
    if record.get("status") != "installed" or not dataset_id:
        return None, [f"prepared_domain_not_recorded_as_installed:{domain}"], set()
    if active.get(domain) != dataset_id:
        return None, [f"prepared_domain_not_active:{domain}:{dataset_id}"], set()
    manifest_path = dataset_manifest_for_id(qdp_root, dataset_id, domain)
    if manifest_path is None:
        return None, [f"prepared_domain_manifest_missing:{domain}:{dataset_id}"], set()

    manifest = read_dataset_manifest(manifest_path)
    installed_by_year: dict[int, Any] = {}
    blockers: list[str] = []
    for shard in manifest.shards:
        year = int(dict(shard.metadata or {}).get("year") or str(shard.start_date)[:4])
        if year in installed_by_year:
            blockers.append(f"installed_year_not_unique:{domain}:{year}")
        installed_by_year[year] = shard

    domain_prepared = sorted(domain_dir.rglob("*.parquet"))
    verified_years: list[int] = []
    verified_files: set[Path] = set()
    seen_years: set[int] = set()
    for path in domain_prepared:
        year = _prepared_year(path)
        if year is None:
            blockers.append(f"prepared_year_invalid:{path}")
            continue
        if year in seen_years:
            blockers.append(f"prepared_year_not_unique:{domain}:{year}")
            continue
        seen_years.add(year)
        sidecar = path.with_suffix(".json")
        shard = installed_by_year.get(year)
        if shard is None:
            blockers.append(f"prepared_year_not_installed:{domain}:{year}")
            continue
        if not sidecar.is_file():
            blockers.append(f"prepared_sidecar_missing:{domain}:{year}")
            continue
        try:
            profile = json.loads(sidecar.read_text(encoding="utf-8"))
            if not isinstance(profile, dict):
                raise TypeError("sidecar root must be an object")
        except (OSError, TypeError, json.JSONDecodeError) as exc:
            blockers.append(f"prepared_sidecar_invalid:{domain}:{year}:{exc}")
            continue
        installed_path = resolve_manifest_path(shard.path, root=qdp_root)
        prepared_hash = str(profile.get("sha256", "") or "")
        installed_hash = str(dict(shard.metadata or {}).get("sha256", "") or "")
        expected_size = int(shard.file_size or 0)
        if not installed_path.is_file():
            blockers.append(f"installed_shard_missing:{domain}:{year}")
        elif not prepared_hash or prepared_hash != installed_hash:
            blockers.append(f"prepared_installed_hash_mismatch:{domain}:{year}")
        elif path.stat().st_size != installed_path.stat().st_size:
            blockers.append(f"prepared_installed_size_mismatch:{domain}:{year}")
        elif expected_size and installed_path.stat().st_size != expected_size:
            blockers.append(f"installed_manifest_size_mismatch:{domain}:{year}")
        else:
            verified_years.append(year)
            verified_files.update((path, sidecar))
    if set(verified_years) != set(installed_by_year):
        blockers.append(
            f"prepared_year_inventory_mismatch:{domain}:{sorted(verified_years)}!={sorted(installed_by_year)}"
        )
    return (
        {
            "domain": domain,
            "dataset_id": dataset_id,
            "verified_years": sorted(verified_years),
            "prepared_parquet_count": len(domain_prepared),
        },
        blockers,
        verified_files,
    )


def _prepared_cache_inventory(
    workspace: Path,
    prepared_root: Path,
) -> tuple[Path, list[Path], list[dict[str, Any]], list[str]]:
    state = _read_state(workspace)
    installed = dict(state.get("installed_domains", {}) or {})
    qdp_root = qdp_v2_root(workspace)
    active = active_dataset_map(read_active_manifest(qdp_root))
    prepared_files = sorted(path for path in prepared_root.rglob("*") if path.is_file())
    verified_domains: list[dict[str, Any]] = []
    blockers: list[str] = []
    verified_files: set[Path] = set()
    for domain_dir in sorted(path for path in prepared_root.iterdir() if path.is_dir()):
        domain_record, domain_blockers, domain_files = _verify_prepared_domain(
            domain_dir,
            installed=installed,
            active=active,
            qdp_root=qdp_root,
        )
        if domain_record is not None:
            verified_domains.append(domain_record)
        blockers.extend(domain_blockers)
        verified_files.update(domain_files)
    for path in sorted(set(prepared_files).difference(verified_files)):
        blockers.append(f"unverified_prepared_file:{path}")
    return qdp_root, prepared_files, verified_domains, sorted(set(blockers))


def cleanup_prepared_cache(
    *,
    workspace_root: str | Path | None = None,
    delete: bool = False,
    yes: bool = False,
) -> dict[str, Any]:
    """Remove only prepared shards already copied into the active QDP datasets."""

    if delete and not yes:
        raise ValueError("prepared_cache_cleanup_requires_yes")
    workspace = _workspace(workspace_root)
    runtime = _runtime(workspace).resolve()
    prepared_root = (runtime / "prepared").resolve()
    prepared_root.relative_to(runtime)
    if not prepared_root.is_dir():
        return _absent_prepared_cache(prepared_root)

    qdp_root, prepared_files, verified_domains, blockers = _prepared_cache_inventory(workspace, prepared_root)
    payload: dict[str, Any] = {
        "status": "blocked" if blockers else ("deleted" if delete else "dry_run"),
        "prepared_root": str(prepared_root),
        "deletable": not blockers,
        "bytes": int(sum(path.stat().st_size for path in prepared_files)),
        "file_count": len(prepared_files),
        "verified_domains": verified_domains,
        "blockers": blockers,
        "destructive_actions_performed": False,
    }
    if blockers or not delete:
        return payload
    shutil.rmtree(prepared_root)
    payload["destructive_actions_performed"] = True
    receipt = qdp_root / "audits" / f"{UPDATE_ID}_prepared_cleanup_{utc_now().replace(':', '')}.json"
    atomic_write_json(receipt, payload)
    payload["receipt_path"] = str(receipt)
    return payload
