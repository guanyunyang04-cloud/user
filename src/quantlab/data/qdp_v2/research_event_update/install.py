"""Research Event Update: install responsibilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from quantlab.core.io import atomic_copy_file
from quantlab.data.domains.contracts import DataDomain
from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    _manifest_schema_from_arrow,
    atomic_write_json,
    qdp_v2_root,
    read_active_manifest,
    utc_now,
    write_active_manifest,
    write_dataset_manifest,
)

from .config import (
    END_DATE,
    REPORT_RC_PAGE_SIZE,
    UPDATE_ID,
    ResearchEventUpdateError,
)
from .context import (
    _assert_credential_free,
    _read_state,
    _runtime,
    _sha256,
    _workspace,
    _write_state,
)


def _install_domain(
    workspace: Path,
    *,
    domain: str,
    prepared: Path,
    contract_version: str,
    primary_key: Sequence[str],
    frequency: str,
    source: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    root = qdp_v2_root(workspace)
    digest = _sha256(prepared)[:24]
    dataset_id = f"{domain}__{digest}"
    dataset_dir = root / "datasets" / domain / dataset_id
    shard = dataset_dir / "shards" / "part-0000.parquet"
    if not shard.is_file():
        atomic_copy_file(prepared, shard)
    connection = duckdb.connect()
    try:
        quoted = str(shard).replace("'", "''")
        key_sql = ",".join(f'"{item}"' for item in primary_key)
        row_count = int(connection.execute(f"SELECT count(*) FROM read_parquet('{quoted}')").fetchone()[0])
        duplicate_count = int(
            connection.execute(
                f"SELECT count(*) FROM (SELECT {key_sql},count(*) n "
                f"FROM read_parquet('{quoted}') GROUP BY {key_sql} HAVING n>1)"
            ).fetchone()[0]
        )
        dates = connection.execute(
            f"SELECT min(source_date),max(source_date),"
            f"count(*) FILTER(WHERE source_date>'{END_DATE}'),"
            "count(*) FILTER(WHERE feature_available_date<>'' "
            "AND try_cast(feature_available_date AS DATE)<=try_cast(source_date AS DATE)) "
            f"FROM read_parquet('{quoted}')"
        ).fetchone()
    finally:
        connection.close()
    if duplicate_count or int(dates[2]) or int(dates[3]):
        raise ResearchEventUpdateError(
            f"event_contract_failed:{domain}:duplicates={duplicate_count}:"
            f"future={int(dates[2])}:availability={int(dates[3])}"
        )
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        domain=domain,
        layer="raw",
        frequency=frequency,
        contract_version=contract_version,
        primary_key=list(primary_key),
        start_date=str(dates[0]),
        end_date=str(dates[1]),
        row_count=row_count,
        shards=[
            ShardManifestEntry(
                path=str(shard.relative_to(root)).replace("\\", "/"),
                row_count=row_count,
                start_date=str(dates[0]),
                end_date=str(dates[1]),
                file_size=shard.stat().st_size,
                metadata={"prepared_sha256": _sha256(prepared)},
            )
        ],
        source={
            **dict(source),
            "checked_through": END_DATE,
            "scope": "point_in_time_historical_mainboard",
            "credential_persisted": False,
        },
        quality={
            "primary_key_unique": True,
            "future_source_rows": 0,
            "availability_not_after_source_rows": 0,
            "strict_point_in_time": True,
            "forbidden_2026_rows": 0,
        },
        schema=_manifest_schema_from_arrow(pq.read_schema(shard)),
        notes=[
            "source_date is the visible report/announcement date; features consume from feature_available_date",
            "PDF URLs are evidence metadata only; no PDF body was downloaded",
        ],
    )
    _assert_credential_free(manifest.to_dict())
    write_dataset_manifest(root, manifest)
    return dataset_id, {
        "dataset_id": dataset_id,
        "row_count": row_count,
        "start_date": str(dates[0]),
        "end_date": str(dates[1]),
        "sha256": _sha256(shard),
    }


@dataclass(frozen=True)
class _PreparedDomain:
    domain: str
    path: Path
    contract_version: str
    primary_key: list[str]
    frequency: str
    source: dict[str, Any]


def _annual_report_statistics(prepared: Path) -> list[dict[str, Any]]:
    return pd.read_parquet(
        prepared / "report_annual_statistics.parquet",
        columns=[
            "year",
            "tushare_source_coverage_status",
            "tushare_source_coverage_start",
            "tushare_source_coverage_end",
            "requested_date_count",
            "terminal_date_count",
            "observed_date_count",
            "confirmed_empty_date_count",
            "failed_date_count",
            "pending_date_count",
        ],
    ).to_dict(orient="records")


def _prepared_domain_specs(
    *,
    workspace: Path,
    prepared: Path,
    include_announcements: bool,
) -> list[_PreparedDomain]:
    annual_statistics = _annual_report_statistics(prepared)
    specs = [
        _PreparedDomain(
            domain=DataDomain.RESEARCH_REPORT,
            path=prepared / "research_report.parquet",
            contract_version="qdp_v2_research_report_pit_v2",
            primary_key=["report_id"],
            frequency="event",
            source={
                "provider": "tushare_report_rc+eastmoney_report_metadata",
                "availability_semantics": "report date; consume next exchange-open day",
                "eastmoney_forecasts_used": False,
                "request_granularity": "report_date",
                "page_size": REPORT_RC_PAGE_SIZE,
                "coverage_contract": "complete_daily_task_ledger",
                "tushare_annual_source_coverage": annual_statistics,
            },
        ),
    ]
    if include_announcements:
        policy = dict(_read_state(workspace).get("announcement_source_policy", {}) or {})
        specs.append(
            _PreparedDomain(
                domain=DataDomain.ANNOUNCEMENT,
                path=prepared / "announcement.parquet",
                contract_version="qdp_v2_announcement_metadata_pit_v1",
                primary_key=["announcement_id"],
                frequency="event",
                source={
                    "provider": "cninfo_primary_with_eastmoney_whole_symbol_fallback",
                    "availability_semantics": "announcement date; consume next exchange-open day",
                    "source_selection_policy": policy,
                },
            )
        )
    return specs


def _install_prepared_domains(
    workspace: Path,
    specs: Sequence[_PreparedDomain],
) -> tuple[dict[str, str], dict[str, Any]]:
    missing = [str(spec.path) for spec in specs if not spec.path.is_file()]
    if missing:
        raise ResearchEventUpdateError(f"prepared_domain_missing:{missing}")
    dataset_ids: dict[str, str] = {}
    installed: dict[str, Any] = {}
    for spec in specs:
        dataset_id, record = _install_domain(
            workspace,
            domain=spec.domain,
            prepared=spec.path,
            contract_version=spec.contract_version,
            primary_key=spec.primary_key,
            frequency=spec.frequency,
            source=spec.source,
        )
        dataset_ids[spec.domain] = dataset_id
        installed[spec.domain] = record
    return dataset_ids, installed


def commit_prepared(
    *,
    workspace_root: str | Path | None = None,
    include_announcements: bool = True,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    prepared = _runtime(workspace) / "prepared"
    specs = _prepared_domain_specs(
        workspace=workspace,
        prepared=prepared,
        include_announcements=include_announcements,
    )
    dataset_ids, installed = _install_prepared_domains(workspace, specs)
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
    _write_state(workspace, state)
    audit_path = root / "audits" / f"{UPDATE_ID}.json"
    _assert_credential_free(state)
    atomic_write_json(audit_path, state)
    return {"status": "applied", "domains": installed, "audit_path": str(audit_path)}
