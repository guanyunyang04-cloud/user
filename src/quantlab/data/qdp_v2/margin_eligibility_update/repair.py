"""Margin Eligibility Update: repair responsibilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from quantlab.data.core.paths import qdp_paths
from quantlab.data.domains.contracts import DataDomain
from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    utc_now,
    write_active_manifest,
    write_dataset_manifest,
)
from quantlab.data.qdp_v2.research_event_update import (
    _assert_credential_free,
)
from quantlab.data.qdp_v2.status import active_dataset_map

from .config import (
    END_DATE,
    MARGIN_ELIGIBILITY_CONTRACT_V1,
    MARGIN_ELIGIBILITY_CONTRACT_V2,
    PROVENANCE_REPAIR_ID,
    SSE_SEMANTICS_REPAIR_ID,
    UPDATE_ID,
    MarginEligibilityUpdateError,
)
from .context import (
    _read_state,
    _sql_paths,
    _workspace,
)
from .install import (
    _install,
)
from .prepare import (
    _copy_query,
)


def _sse_semantics_query(scan: str) -> str:
    return f"""
      SELECT symbol,trade_date,exchange,
        CASE WHEN exchange='SH' AND detail_observed THEN 'eligible_observed'
             WHEN exchange='SH' THEN 'source_unavailable'
             ELSE eligibility_state END AS eligibility_state,
        CASE WHEN exchange='SH' AND detail_observed THEN true
             WHEN exchange='SH' THEN NULL::BOOLEAN
             ELSE eligible END AS eligible,
        CASE WHEN exchange='SH' THEN NULL::BOOLEAN
             ELSE finance_eligible END AS finance_eligible,
        CASE WHEN exchange='SH' THEN NULL::BOOLEAN
             ELSE securities_lending_eligible END
          AS securities_lending_eligible,
        detail_observed,
        CASE WHEN exchange='SH' THEN detail_observed
             ELSE source_available END AS source_available,
        CASE WHEN exchange='SH' THEN detail_observed
             ELSE eligibility_source_available END
          AS eligibility_source_available,
        detail_source_available,source_date,feature_available_date,burn_in_only,
        CASE WHEN exchange='SH' THEN 'sse_official_margin_detail_only'
             ELSE source END AS source
      FROM {scan}
      ORDER BY trade_date,symbol
    """


def _prepare_sse_semantics_shards(
    workspace: Path,
    *,
    root: Path,
    current_manifest: DatasetManifest,
) -> list[Path]:
    prepared_root = (
        qdp_paths(workspace).data_dir
        / "qdp_runtime"
        / SSE_SEMANTICS_REPAIR_ID
        / "prepared"
        / DataDomain.MARGIN_ELIGIBILITY
    )
    prepared_paths: list[Path] = []
    for shard in current_manifest.shards:
        source_path = resolve_manifest_path(shard.path, root=root)
        year = str(shard.start_date)[:4]
        output_path = prepared_root / f"year={year}" / "part-0000.parquet"
        scan = "read_parquet('" + source_path.resolve().as_posix().replace("'", "''") + "')"
        with duckdb.connect() as connection:
            _copy_query(
                connection,
                query=_sse_semantics_query(scan),
                path=output_path,
            )
        prepared_paths.append(output_path)
    return prepared_paths


@dataclass(frozen=True)
class _SseSemanticsStats:
    row_count: int
    duplicate_count: int
    forbidden_rows: int
    sse_known_ineligible: int
    sse_source_unavailable: int
    sse_unobserved: int
    sz_source_unavailable: int
    nullable_state_errors: int
    availability_errors: int


def _sse_semantics_stats(prepared_paths: list[Path]) -> _SseSemanticsStats:
    repaired_scan = f"read_parquet([{_sql_paths(prepared_paths)}], union_by_name=true, hive_partitioning=false)"
    with duckdb.connect() as connection:
        row = connection.execute(
            "SELECT count(*),"
            "count(*)-count(DISTINCT trade_date||'|'||symbol),"
            f"count(*) FILTER(WHERE trade_date>'{END_DATE}'),"
            "count(*) FILTER(WHERE exchange='SH' AND eligibility_state='known_ineligible'),"
            "count(*) FILTER(WHERE exchange='SH' AND eligibility_state='source_unavailable'),"
            "count(*) FILTER(WHERE exchange='SH' AND NOT detail_observed),"
            "count(*) FILTER(WHERE exchange='SZ' AND eligibility_state='source_unavailable'),"
            "count(*) FILTER(WHERE NOT coalesce(("
            " (eligibility_state='eligible_observed' AND eligible IS true) OR"
            " (eligibility_state='known_ineligible' AND eligible IS false) OR"
            " (eligibility_state='source_unavailable' AND eligible IS NULL)),false)),"
            "count(*) FILTER(WHERE NOT coalesce("
            " source_available=eligibility_source_available AND "
            " source_available=(eligibility_state<>'source_unavailable') AND "
            " detail_source_available,false)) "
            f"FROM {repaired_scan}"
        ).fetchone()
    return _SseSemanticsStats(*(int(value or 0) for value in row))


def _sse_semantics_checks(
    stats: _SseSemanticsStats,
    *,
    expected_rows: int,
) -> dict[str, bool]:
    return {
        "row_count_unchanged": stats.row_count == expected_rows,
        "primary_key_unique": stats.duplicate_count == 0,
        "forbidden_2026_rows": stats.forbidden_rows == 0,
        "sse_known_ineligible_removed": stats.sse_known_ineligible == 0,
        "sse_absence_is_source_unavailable": (stats.sse_source_unavailable == stats.sse_unobserved),
        "szse_explicit_states_unchanged": stats.sz_source_unavailable == 0,
        "eligibility_state_matches_nullable_value": stats.nullable_state_errors == 0,
        "source_availability_flags_are_consistent": stats.availability_errors == 0,
    }


def _install_sse_semantics_repair(
    workspace: Path,
    *,
    root: Path,
    current_id: str,
    prepared_paths: list[Path],
    stats: _SseSemanticsStats,
    checks: dict[str, bool],
) -> tuple[str, Any]:
    domain = DataDomain.MARGIN_ELIGIBILITY
    dataset_id, installed = _install(
        workspace,
        domain=domain,
        paths=prepared_paths,
        input_manifest=None,
    )
    installed_path = dataset_manifest_for_id(root, dataset_id, domain)
    assert installed_path is not None
    installed_manifest = read_dataset_manifest(installed_path)
    repaired_manifest = DatasetManifest.from_mapping(
        {
            **installed_manifest.to_dict(),
            "source": {
                **dict(installed_manifest.source or {}),
                "semantic_repair_id": SSE_SEMANTICS_REPAIR_ID,
                "upstream_dataset_id": current_id,
            },
            "quality": {
                **dict(installed_manifest.quality or {}),
                "sse_known_ineligible_count": 0,
                "sse_source_unavailable_count": stats.sse_source_unavailable,
                "semantic_repair_checks": checks,
            },
            "notes": [
                *list(installed_manifest.notes or []),
                "immutable successor of the v1 dataset; only SSE eligibility semantics changed",
            ],
        }
    )
    _assert_credential_free(repaired_manifest.to_dict())
    write_dataset_manifest(root, repaired_manifest)

    latest_active = read_active_manifest(root)
    latest = active_dataset_map(latest_active)
    if latest.get(domain) != current_id:
        raise MarginEligibilityUpdateError("active_margin_eligibility_drifted")
    latest_active["datasets"] = {**latest, domain: dataset_id}
    latest_active["updated_at"] = utc_now()
    write_active_manifest(root, latest_active)
    if active_dataset_map(read_active_manifest(root)).get(domain) != dataset_id:
        raise MarginEligibilityUpdateError("sse_semantics_active_switch_failed")
    return dataset_id, installed


def repair_sse_eligibility_semantics(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    """Replace unsupported SSE negative states with explicit unknown states.

    The repair only rewrites ``margin_eligibility``.  Existing v1 shards and
    manifests remain immutable and the active pointer switches only after the
    successor has passed the state/value invariants below.
    """

    workspace = _workspace(workspace_root)
    root = qdp_v2_root(workspace)
    active = read_active_manifest(root)
    current = active_dataset_map(active)
    current_id = str(current.get(DataDomain.MARGIN_ELIGIBILITY, ""))
    current_path = dataset_manifest_for_id(root, current_id, DataDomain.MARGIN_ELIGIBILITY)
    if current_path is None:
        raise MarginEligibilityUpdateError("active_margin_eligibility_missing")
    current_manifest = read_dataset_manifest(current_path)
    if current_manifest.contract_version == MARGIN_ELIGIBILITY_CONTRACT_V2:
        return {
            "status": "already_repaired",
            "repair_id": SSE_SEMANTICS_REPAIR_ID,
            "dataset_id": current_id,
        }
    if current_manifest.contract_version != MARGIN_ELIGIBILITY_CONTRACT_V1:
        raise MarginEligibilityUpdateError(
            f"unsupported_margin_eligibility_contract:{current_manifest.contract_version}"
        )

    prepared_paths = _prepare_sse_semantics_shards(
        workspace,
        root=root,
        current_manifest=current_manifest,
    )
    stats = _sse_semantics_stats(prepared_paths)
    checks = _sse_semantics_checks(stats, expected_rows=current_manifest.row_count)
    if not all(checks.values()):
        raise MarginEligibilityUpdateError(f"sse_semantics_repair_contract_failed:{checks}")

    dataset_id, installed = _install_sse_semantics_repair(
        workspace,
        root=root,
        current_id=current_id,
        prepared_paths=prepared_paths,
        stats=stats,
        checks=checks,
    )

    payload = {
        "status": "applied",
        "repair_id": SSE_SEMANTICS_REPAIR_ID,
        "input_dataset_id": current_id,
        "installed_domains": {DataDomain.MARGIN_ELIGIBILITY: installed},
        "checks": checks,
        "statistics": {
            "row_count": stats.row_count,
            "sse_source_unavailable_count": stats.sse_source_unavailable,
            "request_2026_count": 0,
        },
        "updated_at": utc_now(),
    }
    _assert_credential_free(payload)
    atomic_write_json(root / "audits" / f"{SSE_SEMANTICS_REPAIR_ID}.json", payload)
    return payload


def repair_margin_detail_provenance(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    """Create a metadata-correct immutable version without copying data shards."""

    workspace = _workspace(workspace_root)
    root = qdp_v2_root(workspace)
    state = _read_state(workspace)
    active = read_active_manifest(root)
    current = active_dataset_map(active)
    current_id = str(current.get(DataDomain.MARGIN_DETAIL, ""))
    current_path = dataset_manifest_for_id(root, current_id, DataDomain.MARGIN_DETAIL)
    if current_path is None:
        raise MarginEligibilityUpdateError("active_margin_detail_manifest_missing")
    current_manifest = read_dataset_manifest(current_path)
    if (
        current_manifest.contract_version == "qdp_v2_tushare_margin_detail_raw_v3_mixed_provenance"
        and current_manifest.source.get("query_granularity") == "trade_date"
    ):
        return {
            "status": "already_repaired",
            "dataset_id": current_id,
            "repair_id": PROVENANCE_REPAIR_ID,
        }
    upstream_id = str(dict(state.get("input_dataset_ids", {}) or {}).get(DataDomain.MARGIN_DETAIL, ""))
    upstream_path = dataset_manifest_for_id(root, upstream_id, DataDomain.MARGIN_DETAIL)
    if upstream_path is None:
        raise MarginEligibilityUpdateError("margin_detail_provenance_upstream_manifest_missing")
    upstream = read_dataset_manifest(upstream_path)
    digest = hashlib.sha256(
        (f"{current_id}|{upstream_id}|qdp_v2_tushare_margin_detail_raw_v3_mixed_provenance").encode("ascii")
    ).hexdigest()
    dataset_id = f"{DataDomain.MARGIN_DETAIL}__{digest[:24]}"
    source = {
        **dict(upstream.source or {}),
        "provider": "tushare_compatible_primary+sse+szse_official_gap_repair",
        "query_granularity": "trade_date",
        "upstream_dataset_id": upstream_id,
        "repaired_dataset_id": current_id,
        "exchange_gap_repair_update_id": UPDATE_ID,
        "exchange_gap_repair_row_count": int(state.get("exchange_detail_missing_count", 0)),
        "availability_semantics": "next_exchange_open_day",
        "request_2026_count": 0,
        "credential_persisted": False,
    }
    repaired = DatasetManifest(
        dataset_id=dataset_id,
        domain=current_manifest.domain,
        layer=current_manifest.layer,
        frequency=current_manifest.frequency,
        contract_version="qdp_v2_tushare_margin_detail_raw_v3_mixed_provenance",
        primary_key=list(current_manifest.primary_key),
        start_date=current_manifest.start_date,
        end_date=current_manifest.end_date,
        row_count=current_manifest.row_count,
        shards=list(current_manifest.shards),
        source=source,
        quality={
            **dict(current_manifest.quality or {}),
            "provenance_repair_only": True,
            "physical_rows_unchanged": True,
        },
        schema=list(current_manifest.schema),
        notes=[
            *list(current_manifest.notes or []),
            "metadata-only immutable successor restores Tushare primary provenance and daily query granularity",
            "all physical shards and values are unchanged from the exchange-gap-repaired dataset",
        ],
    )
    _assert_credential_free(repaired.to_dict())
    write_dataset_manifest(root, repaired)
    active["datasets"] = {**current, DataDomain.MARGIN_DETAIL: dataset_id}
    active["updated_at"] = utc_now()
    write_active_manifest(root, active)
    if active_dataset_map(read_active_manifest(root)).get(DataDomain.MARGIN_DETAIL) != dataset_id:
        raise MarginEligibilityUpdateError("margin_detail_provenance_switch_failed")
    payload = {
        "status": "applied",
        "repair_id": PROVENANCE_REPAIR_ID,
        "input_dataset_id": current_id,
        "upstream_dataset_id": upstream_id,
        "dataset_id": dataset_id,
        "row_count": repaired.row_count,
        "shards_reused": len(repaired.shards),
        "physical_rows_unchanged": True,
        "query_granularity": "trade_date",
        "request_2026_count": 0,
        "applied_at": utc_now(),
    }
    atomic_write_json(root / "audits" / f"{PROVENANCE_REPAIR_ID}.json", payload)
    return payload
