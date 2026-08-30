"""Database audit asof checks."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from quantlab.data.core.paths import qdp_paths
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb
from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    atomic_write_json,
    qdp_v2_root,
    read_active_manifest,
    resolve_manifest_path,
    utc_now,
    write_active_manifest,
    write_dataset_manifest,
)
from quantlab.data.qdp_v2.repair.common import _sql_literal
from quantlab.data.qdp_v2.status import active_dataset_map

from .common import (
    _finding,
    _path_texts,
    _q,
)
from .config import (
    AS_OF_DATE_COLUMNS,
)
from .identity import (
    _active_manifest,
)
from .orchestrator import (
    audit_database,
)


def _audit_ephemeral_snapshot(
    *,
    workspace: Path,
    source_root: Path,
    active: dict[str, Any],
    cutoff: str,
    selected_threads: int,
    deep: bool,
    runtime: str,
    max_shards: int,
    sample_limit: int,
    skip_global_primary_key: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    datasets = active_dataset_map(active)
    runtime_root = qdp_paths(workspace).runtime_dir
    runtime_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"qdp_as_of_{cutoff.replace('-', '')}_",
        dir=str(runtime_root),
    ) as temporary:
        temporary_root = Path(temporary)
        snapshot_workspace = temporary_root / "workspace"
        snapshot_root = qdp_v2_root(snapshot_workspace)
        materialized_root = temporary_root / "boundary_shards"
        profiles: dict[str, Any] = {}
        with open_guarded_duckdb(
            temp_directory=temporary_root / "duckdb_spill",
            threads=selected_threads,
        ) as con:
            for domain, dataset_id in sorted(datasets.items()):
                source_manifest = _active_manifest(source_root, dataset_id, domain)
                snapshot_manifest, profile = _snapshot_manifest_as_of(
                    con,
                    source_root=source_root,
                    source_manifest=source_manifest,
                    as_of_date=cutoff,
                    materialized_root=materialized_root,
                )
                write_dataset_manifest(snapshot_root, snapshot_manifest)
                profiles[domain] = profile
        snapshot_active = {
            **active,
            "active_as_of_date": cutoff,
            "updated_at": utc_now(),
            "source_active_manifest": str((source_root / "active" / "active.json").resolve()),
        }
        snapshot_active["scope"] = {
            **dict(active.get("scope", {}) or {}),
            "end_date": cutoff,
            "as_of_snapshot": True,
        }
        write_active_manifest(snapshot_root, snapshot_active)
        payload = audit_database(
            workspace_root=snapshot_workspace,
            deep=deep,
            runtime=runtime,
            threads=selected_threads,
            max_shards=max_shards,
            sample_limit=sample_limit,
            skip_global_primary_key=skip_global_primary_key,
            write=False,
        )
    return payload, profiles


def _as_of_boundary_findings(profiles: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for domain, profile in profiles.items():
        if int(profile["forbidden_after_cutoff_row_count_in_audit_snapshot"]):
            findings.append(
                _finding(
                    "high",
                    "as_of_boundary",
                    domain,
                    "post_cutoff_row_reached_audit_snapshot",
                    profile,
                )
            )
        if int(profile["null_cutoff_date_row_count"]):
            findings.append(
                _finding(
                    "high",
                    "as_of_boundary",
                    domain,
                    "cutoff_date_is_null",
                    {
                        "date_column": profile["date_column"],
                        "null_cutoff_date_row_count": profile["null_cutoff_date_row_count"],
                    },
                )
            )
    return findings


def _decorate_as_of_audit(
    payload: dict[str, Any],
    *,
    source_root: Path,
    cutoff: str,
    profiles: dict[str, Any],
) -> None:
    payload["findings"] = [
        *list(payload.get("findings", []) or []),
        *_as_of_boundary_findings(profiles),
    ]
    payload["errors"] = [item for item in payload["findings"] if item.get("severity") == "high"]
    payload["warnings"] = [item for item in payload["findings"] if item.get("severity") == "medium"]
    payload["finding_count"] = len(payload["findings"])
    payload["status"] = "needs_attention" if payload["errors"] else "ok"
    payload["mode"] = "full_as_of"
    payload["qdp_v2_root"] = str(source_root.resolve())
    payload["as_of"] = {
        "date": cutoff,
        "cutoff_applied_before_semantic_checks": True,
        "physical_row_read_cutoff": cutoff,
        "forbidden_after_cutoff_row_count_in_audit_snapshot": sum(
            int(profile["forbidden_after_cutoff_row_count_in_audit_snapshot"]) for profile in profiles.values()
        ),
        "null_cutoff_date_row_count": sum(int(profile["null_cutoff_date_row_count"]) for profile in profiles.values()),
        "source_rows_after_cutoff_excluded": sum(
            int(profile["source_rows_after_cutoff_excluded"]) for profile in profiles.values()
        ),
        "ephemeral_snapshot_removed": True,
        "datasets": profiles,
    }


def audit_database_as_of(
    *,
    as_of_date: str,
    workspace_root: str | Path | None = None,
    deep: bool = True,
    runtime: str = "balanced",
    threads: int | None = None,
    max_shards: int = 0,
    sample_limit: int = 20,
    skip_global_primary_key: bool = False,
    write_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the full database audit on an ephemeral row-level as-of snapshot.

    Shards wholly before the cutoff are referenced in place. Shards crossing
    the cutoff are filtered into a temporary workspace before any semantic
    check runs, so downstream checks cannot consume post-cutoff rows.
    """

    cutoff = _normalized_cutoff(as_of_date)
    workspace = Path(workspace_root or Path.cwd()).resolve()
    source_root = qdp_v2_root(workspace)
    active = read_active_manifest(source_root)
    if not active:
        raise ValueError("active_manifest_missing")
    selected_threads = int(threads or {"safe": 2, "balanced": 4, "fast": 8}.get(runtime, 4))
    selected_threads = max(1, min(selected_threads, os.cpu_count() or 1))
    payload, profiles = _audit_ephemeral_snapshot(
        workspace=workspace,
        source_root=source_root,
        active=active,
        cutoff=cutoff,
        selected_threads=selected_threads,
        deep=deep,
        runtime=runtime,
        max_shards=max_shards,
        sample_limit=sample_limit,
        skip_global_primary_key=skip_global_primary_key,
    )
    _decorate_as_of_audit(
        payload,
        source_root=source_root,
        cutoff=cutoff,
        profiles=profiles,
    )
    if write_path is not None:
        resolved = Path(write_path).resolve()
        atomic_write_json(resolved, payload)
        payload["audit_path"] = str(resolved)
    return payload


def _normalized_cutoff(value: str) -> str:
    parsed = pd.Timestamp(str(value))
    if pd.isna(parsed) or parsed.time() != pd.Timestamp(parsed.date()).time():
        raise ValueError(f"invalid_as_of_date:{value}")
    return parsed.date().isoformat()


@dataclass
class _AsOfShardSelection:
    retained: list[ShardManifestEntry]
    direct_count: int = 0
    materialized_count: int = 0
    excluded_count: int = 0
    source_rows_after: int = 0


def _direct_reference_entry(
    entry: ShardManifestEntry,
    source_path: Path,
) -> ShardManifestEntry:
    rows = int(pq.ParquetFile(source_path).metadata.num_rows)
    return ShardManifestEntry(
        path=str(source_path),
        row_count=rows,
        start_date=str(entry.start_date or ""),
        end_date=str(entry.end_date or ""),
        status=entry.status,
        file_size=int(source_path.stat().st_size),
        metadata={**dict(entry.metadata), "as_of_direct_reference": True},
    )


def _boundary_shard_profile(
    con: Any,
    *,
    source_path: Path,
    date_column: str,
    as_of_date: str,
) -> tuple[int, int, str, str]:
    quoted_source = str(source_path).replace("'", "''")
    date_expr = f"try_cast({_q(date_column)} AS DATE)"
    row = con.execute(
        f"""
        SELECT count(*) FILTER (
                   WHERE {date_expr} IS NULL
                      OR {date_expr}<=try_cast(? AS DATE)
               ),
               count(*) FILTER (WHERE {date_expr}>try_cast(? AS DATE)),
               min(cast({_q(date_column)} AS VARCHAR)) FILTER (
                   WHERE {date_expr}<=try_cast(? AS DATE)
               ),
               max(cast({_q(date_column)} AS VARCHAR)) FILTER (
                   WHERE {date_expr}<=try_cast(? AS DATE)
               )
        FROM read_parquet('{quoted_source}', union_by_name=true)
        """,
        [as_of_date, as_of_date, as_of_date, as_of_date],
    ).fetchone()
    return int(row[0] or 0), int(row[1] or 0), str(row[2] or ""), str(row[3] or "")


def _materialize_boundary_entry(
    con: Any,
    *,
    source_manifest: DatasetManifest,
    entry: ShardManifestEntry,
    source_path: Path,
    target: Path,
    shard_index: int,
    date_column: str,
    as_of_date: str,
    expected_rows: int,
    source_rows_after: int,
    filtered_start: str,
    filtered_end: str,
) -> ShardManifestEntry:
    target.parent.mkdir(parents=True, exist_ok=True)
    quoted_source = str(source_path).replace("'", "''")
    quoted_target = str(target).replace("'", "''")
    date_expr = f"try_cast({_q(date_column)} AS DATE)"
    ordering = ",".join(_q(item) for item in source_manifest.primary_key)
    order_clause = f" ORDER BY {ordering}" if ordering else ""
    con.execute(
        f"""
        COPY (
          SELECT * FROM read_parquet('{quoted_source}', union_by_name=true)
          WHERE {date_expr} IS NULL OR {date_expr}<=DATE '{as_of_date}'
          {order_clause}
        ) TO '{quoted_target}' (
          FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072
        )
        """
    )
    actual_rows = int(pq.ParquetFile(target).metadata.num_rows)
    if actual_rows != expected_rows:
        raise ValueError(
            f"as_of_materialized_row_mismatch:{source_manifest.domain}:{shard_index}:{expected_rows}:{actual_rows}"
        )
    return ShardManifestEntry(
        path=str(target.resolve()),
        row_count=actual_rows,
        start_date=filtered_start,
        end_date=filtered_end,
        status="stored",
        file_size=int(target.stat().st_size),
        metadata={
            **dict(entry.metadata),
            "as_of_materialized": True,
            "source_path": str(source_path),
            "source_rows_after_cutoff_excluded": source_rows_after,
        },
    )


def _select_as_of_shards(
    con: Any,
    *,
    source_root: Path,
    source_manifest: DatasetManifest,
    date_column: str,
    as_of_date: str,
    materialized_root: Path,
) -> _AsOfShardSelection:
    selection = _AsOfShardSelection(retained=[])
    for index, entry in enumerate(source_manifest.shards):
        source_path = resolve_manifest_path(entry.path, root=source_root).resolve()
        start = str(entry.start_date or "")
        end = str(entry.end_date or "")
        if start and start > as_of_date:
            selection.excluded_count += 1
            selection.source_rows_after += int(entry.row_count)
            continue
        if end and end <= as_of_date:
            selection.retained.append(_direct_reference_entry(entry, source_path))
            selection.direct_count += 1
            continue
        retained_rows, rows_after, filtered_start, filtered_end = _boundary_shard_profile(
            con,
            source_path=source_path,
            date_column=date_column,
            as_of_date=as_of_date,
        )
        selection.source_rows_after += rows_after
        if not retained_rows:
            selection.excluded_count += 1
            continue
        target = materialized_root / source_manifest.domain / f"shard_{index:04d}.parquet"
        selection.retained.append(
            _materialize_boundary_entry(
                con,
                source_manifest=source_manifest,
                entry=entry,
                source_path=source_path,
                target=target,
                shard_index=index,
                date_column=date_column,
                as_of_date=as_of_date,
                expected_rows=retained_rows,
                source_rows_after=rows_after,
                filtered_start=filtered_start,
                filtered_end=filtered_end,
            )
        )
        selection.materialized_count += 1
    return selection


def _snapshot_boundary_profile(
    con: Any,
    *,
    retained: list[ShardManifestEntry],
    date_column: str,
    as_of_date: str,
) -> tuple[int, int, str, str]:
    date_expr = f"try_cast({_q(date_column)} AS DATE)"
    audit_paths = [Path(item.path) for item in retained]
    boundary = con.execute(
        f"""
        SELECT count(*) FILTER (WHERE {date_expr}>{_sql_literal(as_of_date)}),
               count(*) FILTER (WHERE {date_expr} IS NULL),
               min(cast({_q(date_column)} AS VARCHAR)) FILTER (
                   WHERE {date_expr} IS NOT NULL
               ),
               max(cast({_q(date_column)} AS VARCHAR)) FILTER (
                   WHERE {date_expr} IS NOT NULL
               )
        FROM read_parquet(?, union_by_name=true)
        """,
        [_path_texts(audit_paths)],
    ).fetchone()
    return int(boundary[0] or 0), int(boundary[1] or 0), str(boundary[2] or ""), str(boundary[3] or "")


def _snapshot_manifest_as_of(
    con: Any,
    *,
    source_root: Path,
    source_manifest: DatasetManifest,
    as_of_date: str,
    materialized_root: Path,
) -> tuple[DatasetManifest, dict[str, Any]]:
    domain = source_manifest.domain
    date_column = AS_OF_DATE_COLUMNS.get(domain, "trade_date")
    schema_columns = [str(item.get("name", "")) for item in source_manifest.schema]
    if date_column not in schema_columns:
        raise ValueError(f"as_of_date_column_missing:{domain}:{date_column}")
    selection = _select_as_of_shards(
        con,
        source_root=source_root,
        source_manifest=source_manifest,
        date_column=date_column,
        as_of_date=as_of_date,
        materialized_root=materialized_root,
    )
    if not selection.retained:
        raise ValueError(f"as_of_snapshot_empty:{domain}:{as_of_date}")
    forbidden, null_dates, audit_min_date, audit_max_date = _snapshot_boundary_profile(
        con,
        retained=selection.retained,
        date_column=date_column,
        as_of_date=as_of_date,
    )
    if forbidden:
        raise ValueError(f"as_of_snapshot_contains_future_rows:{domain}:{forbidden}")

    source = dict(source_manifest.source)
    checked_through = str(source.get("checked_through", "") or "")
    if checked_through and checked_through > as_of_date:
        source["physical_source_checked_through"] = checked_through
        source["checked_through"] = as_of_date
    source["as_of_snapshot_date"] = as_of_date
    quality = dict(source_manifest.quality)
    if "unknown_industry_rows" in quality:
        quality["physical_source_unknown_industry_rows"] = quality.pop("unknown_industry_rows")
    use_declared_range = domain not in AS_OF_DATE_COLUMNS
    start_date = audit_min_date if use_declared_range else ""
    end_date = audit_max_date if use_declared_range else ""
    snapshot = DatasetManifest(
        dataset_id=source_manifest.dataset_id,
        domain=domain,
        layer=source_manifest.layer,
        frequency=source_manifest.frequency,
        contract_version=source_manifest.contract_version,
        primary_key=list(source_manifest.primary_key),
        start_date=start_date,
        end_date=end_date,
        row_count=sum(int(item.row_count) for item in selection.retained),
        shards=selection.retained,
        source=source,
        quality=quality,
        schema=list(source_manifest.schema),
        notes=[
            *list(source_manifest.notes),
            f"Ephemeral audit snapshot filtered through {as_of_date} on {date_column}.",
        ],
    )
    profile = {
        "dataset_id": source_manifest.dataset_id,
        "date_column": date_column,
        "source_manifest_start_date": source_manifest.start_date,
        "source_manifest_end_date": source_manifest.end_date,
        "audit_snapshot_row_count": snapshot.row_count,
        "audit_snapshot_min_date": audit_min_date,
        "audit_snapshot_max_date": audit_max_date,
        "forbidden_after_cutoff_row_count_in_audit_snapshot": forbidden,
        "null_cutoff_date_row_count": null_dates,
        "source_rows_after_cutoff_excluded": selection.source_rows_after,
        "direct_reference_shard_count": selection.direct_count,
        "materialized_boundary_shard_count": selection.materialized_count,
        "excluded_after_cutoff_shard_count": selection.excluded_count,
        "physical_row_read_cutoff": as_of_date,
    }
    return snapshot, profile
