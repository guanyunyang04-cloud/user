"""Tushare Extended Backfill: install responsibilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from quantlab.core.io import atomic_copy_file
from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    _manifest_schema_from_arrow,
    atomic_write_json,
    qdp_v2_root,
    read_active_manifest,
    resolve_manifest_path,
    utc_now,
    write_active_manifest,
    write_dataset_manifest,
)

from .config import (
    END_DATE,
    MAX_REQUESTS_PER_MINUTE,
    MAX_WORKERS,
    UPDATE_ID,
    EndpointSpec,
    TushareExtendedBackfillError,
)
from .context import (
    _assert_credential_free,
    _open_dates,
    _read_state,
    _schema_hash,
    _sha256,
    _stable_hash,
    _workspace,
    _write_state,
)
from .download import (
    _selected_specs,
)
from .prepare import (
    _assert_uniform_prepared_schema,
    _factor_validation,
    _prepare_domain,
    _provider_fields,
    _year_page_paths,
)


def _install_domain(
    workspace: Path,
    *,
    spec: EndpointSpec,
    records: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    root = qdp_v2_root(workspace)
    fingerprint = _stable_hash({"domain": spec.domain, "shards": [item["sha256"] for item in records]})[:24]
    dataset_id = f"{spec.domain}__{fingerprint}"
    dataset_dir = root / "datasets" / spec.domain / dataset_id
    shards: list[ShardManifestEntry] = []
    for record in records:
        year = int(record["year"])
        source = Path(str(record["path"]))
        target = dataset_dir / "shards" / f"year={year}" / "part-0000.parquet"
        if not target.is_file() or _sha256(target) != str(record["sha256"]):
            atomic_copy_file(source, target)
        shards.append(
            ShardManifestEntry(
                path=str(target.relative_to(root)).replace("\\", "/"),
                row_count=int(record["row_count"]),
                start_date=str(record["start_date"]),
                end_date=str(record["end_date"]),
                file_size=target.stat().st_size,
                metadata={
                    "year": year,
                    "sha256": _sha256(target),
                    "burn_in_only": year == 2010,
                },
            )
        )
    nonempty = [item for item in records if int(item["row_count"]) > 0]
    if not nonempty:
        raise TushareExtendedBackfillError(f"prepared_domain_empty:{spec.domain}")
    first_shard = resolve_manifest_path(shards[0].path, root=root)
    source_inventory_hash = _stable_hash([item["input_hash"] for item in records])
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        domain=spec.domain,
        layer="raw",
        frequency=spec.frequency,
        contract_version=spec.contract_version,
        primary_key=list(spec.primary_key),
        start_date=min(str(item["start_date"]) for item in nonempty),
        end_date=max(str(item["end_date"]) for item in nonempty),
        row_count=sum(int(item["row_count"]) for item in records),
        shards=shards,
        source={
            "provider": f"tushare_compatible.{spec.api_name}",
            "query_granularity": spec.mode,
            "checked_through": END_DATE,
            "scope": "point_in_time_historical_mainboard",
            "source_inventory_hash": source_inventory_hash,
            "credential_persisted": False,
            "request_concurrency_maximum": MAX_WORKERS,
            "request_rate_per_minute_maximum": MAX_REQUESTS_PER_MINUTE,
        },
        quality={
            "primary_key_unique": True,
            "strict_point_in_time": True,
            "burn_in_year": 2010,
            "burn_in_eligible_for_training": False,
            "forbidden_2026_rows": 0,
            **dict(quality),
        },
        schema=_manifest_schema_from_arrow(pq.read_schema(first_shard)),
        notes=[
            "2010 is retained only for causal feature burn-in",
            "research rows end on 2025-12-31 and never read 2026",
            "provider missingness is preserved rather than filled with zero",
        ],
    )
    _assert_credential_free(manifest.to_dict())
    write_dataset_manifest(root, manifest)
    return {
        "dataset_id": dataset_id,
        "domain": spec.domain,
        "row_count": int(manifest.row_count),
        "start_date": manifest.start_date,
        "end_date": manifest.end_date,
        "shard_count": len(shards),
        "source_inventory_hash": source_inventory_hash,
        "quality": dict(manifest.quality),
    }


def prepare_and_install(
    *,
    workspace_root: str | Path | None = None,
    domains: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    specs = _selected_specs(domains)
    state = _read_state(workspace)
    probe_endpoints = dict(dict(state.get("probe", {}) or {}).get("endpoints", {}) or {})
    open_dates = _open_dates(workspace)
    installed: dict[str, Any] = dict(state.get("installed_domains", {}) or {})
    dataset_ids: dict[str, str] = {}
    for spec in specs:
        if spec.optional and dict(probe_endpoints.get(spec.name, {}) or {}).get("status") not in {
            "available",
            "available_with_partial_probe",
        }:
            installed[spec.domain] = {
                "status": "source_unavailable",
                "optional": True,
            }
            continue
        records = _prepare_domain(workspace, spec, open_dates)
        _assert_uniform_prepared_schema(spec, records)
        quality: dict[str, Any] = {
            "provider_field_count": len(_provider_fields(_year_page_paths(workspace, spec, 2011))),
            "provider_schema_hash": _schema_hash(_provider_fields(_year_page_paths(workspace, spec, 2011))),
            "yearly_row_count": {str(item["year"]): int(item["row_count"]) for item in records},
            "disallowed_negative_value_count": sum(
                int(item.get("disallowed_negative_value_count", 0)) for item in records
            ),
            "allowed_negative_net_value_count": sum(
                int(item.get("allowed_negative_net_value_count", 0)) for item in records
            ),
            "provider_net_amount_unreconciled_count": sum(
                int(item.get("moneyflow_main_net_relation_mismatch_count", 0)) for item in records
            ),
            "source_exception_negative_value_count": sum(
                int(item.get("source_exception_negative_value_count", 0)) for item in records
            ),
        }
        if spec.name == "stk-factor-pro":
            quality.update(_factor_validation(workspace, records))
            quality["qfq_formal_eligibility"] = False
        record = _install_domain(
            workspace,
            spec=spec,
            records=records,
            quality=quality,
        )
        installed[spec.domain] = {"status": "installed", **record}
        dataset_ids[spec.domain] = str(record["dataset_id"])
    if dataset_ids:
        root = qdp_v2_root(workspace)
        active = read_active_manifest(root)
        active["datasets"] = {
            **dict(active.get("datasets", {}) or {}),
            **dataset_ids,
        }
        active["updated_at"] = utc_now()
        write_active_manifest(root, active)
    state = _read_state(workspace)
    state["installed_domains"] = installed
    state["status"] = "applied"
    state["training_performed"] = False
    _write_state(workspace, state)
    audit_path = qdp_v2_root(workspace) / "audits" / f"{UPDATE_ID}.json"
    _assert_credential_free(state)
    atomic_write_json(audit_path, state)
    return {"status": "applied", "domains": installed, "audit_path": str(audit_path)}
