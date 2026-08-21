"""Financial Statement Update: install responsibilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
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
from quantlab.data.qdp_v2.research_event_update import (
    _assert_credential_free,
    _sha256,
)

from .config import (
    BALANCE_SEMANTIC_CONFLICT_COLUMN,
    END_DATE,
    PRIMARY_KEY,
    SOURCE_SCHEMA_VERSION,
    UPDATE_ID,
    V2_STATEMENT_SPECS,
    FinancialStatementUpdateError,
    StatementSpec,
)
from .context import (
    _read_state,
    _workspace,
    _write_state,
)


@dataclass(frozen=True)
class _StatementAudit:
    row_count: int
    start_date: str
    end_date: str


def _copy_statement_shard(prepared: Path, shard: Path) -> None:
    if shard.is_file():
        return
    atomic_copy_file(prepared, shard)


def _audit_statement_shard(shard: Path, spec: StatementSpec) -> _StatementAudit:
    quoted = str(shard).replace("'", "''")
    key_sql = ",".join(f'"{column}"' for column in PRIMARY_KEY)
    connection = duckdb.connect()
    try:
        row_count = int(connection.execute(f"SELECT count(*) FROM read_parquet('{quoted}')").fetchone()[0])
        duplicate_count = int(
            connection.execute(
                f"SELECT count(*) FROM (SELECT {key_sql},count(*) n "
                f"FROM read_parquet('{quoted}') GROUP BY {key_sql} HAVING n>1)"
            ).fetchone()[0]
        )
        dates = connection.execute(
            "SELECT min(publish_date),max(publish_date),"
            f"count(*) FILTER(WHERE publish_date>'{END_DATE}'),"
            "count(*) FILTER(WHERE report_date>publish_date),"
            "count(*) FILTER(WHERE feature_available_date<>'' "
            "AND feature_available_date<=publish_date) "
            f"FROM read_parquet('{quoted}')"
        ).fetchone()
    finally:
        connection.close()
    if duplicate_count or any(int(value or 0) for value in dates[2:]):
        raise FinancialStatementUpdateError(
            f"statement_contract_failed:{spec.domain}:duplicates={duplicate_count}:"
            f"future={int(dates[2] or 0)}:report_after_publish={int(dates[3] or 0)}:"
            f"availability={int(dates[4] or 0)}"
        )
    return _StatementAudit(row_count, str(dates[0]), str(dates[1]))


def _statement_contract_version(spec: StatementSpec, shard: Path) -> str:
    if (
        spec.domain == DataDomain.BALANCE_SHEET_QUARTERLY
        and BALANCE_SEMANTIC_CONFLICT_COLUMN in pq.read_schema(shard).names
    ):
        return f"qdp_v2_{spec.domain}_pit_v5"
    return f"qdp_v2_{spec.domain}_pit_v2"


def _statement_manifest(
    *,
    root: Path,
    dataset_id: str,
    spec: StatementSpec,
    prepared: Path,
    shard: Path,
    audit: _StatementAudit,
) -> DatasetManifest:
    return DatasetManifest(
        dataset_id=dataset_id,
        domain=spec.domain,
        layer="raw",
        frequency="quarterly_event",
        contract_version=_statement_contract_version(spec, shard),
        primary_key=list(PRIMARY_KEY),
        start_date=audit.start_date,
        end_date=audit.end_date,
        row_count=audit.row_count,
        shards=[
            ShardManifestEntry(
                path=str(shard.relative_to(root)).replace("\\", "/"),
                row_count=audit.row_count,
                start_date=audit.start_date,
                end_date=audit.end_date,
                file_size=shard.stat().st_size,
                metadata={"prepared_sha256": _sha256(prepared)},
            )
        ],
        source={
            "provider": f"tushare_compatible_{spec.api_name}",
            "source_schema_version": SOURCE_SCHEMA_VERSION,
            "checked_through": END_DATE,
            "scope": "point_in_time_historical_mainboard",
            "availability_semantics": "actual announcement date; consume next exchange-open day",
            "credential_persisted": False,
            "provider_revision_history_timestamp_complete": False,
        },
        quality={
            "primary_key_unique": True,
            "future_publish_rows": 0,
            "report_after_publish_rows": 0,
            "strict_feature_availability_lag": True,
            "provider_future_response_rows_not_persisted": True,
        },
        schema=_manifest_schema_from_arrow(pq.read_schema(shard)),
        notes=[
            "f_ann_date is the PIT publication date when present; ann_date is the fallback",
            "duplicate provider rows prefer update_flag=1, then completeness, with conflicts retained as flags",
            "the provider does not expose complete correction timestamps for every restatement",
            "v2 preserves every v1 key and existing value; only audited balance-sheet fields are appended",
            "balance extension source conflicts are retained separately from legacy core-field conflicts",
            "combined statement fields are retained as reported and are never split into fabricated components",
            "stable trade-receivable, trade-payable, fixed-asset, and construction-in-progress measures retain explicit source states",
            "semantic-field conflicts are retained separately from other extension conflicts",
            "special financial-company statement structures retain not_applicable rather than numeric zero",
        ],
    )


def _install_domain(
    workspace: Path,
    *,
    spec: StatementSpec,
    prepared: Path,
) -> tuple[str, dict[str, Any]]:
    root = qdp_v2_root(workspace)
    digest = _sha256(prepared)[:24]
    dataset_id = f"{spec.domain}__{digest}"
    dataset_dir = root / "datasets" / spec.domain / dataset_id
    shard = dataset_dir / "shards" / "part-0000.parquet"
    _copy_statement_shard(prepared, shard)
    audit = _audit_statement_shard(shard, spec)
    manifest = _statement_manifest(
        root=root,
        dataset_id=dataset_id,
        spec=spec,
        prepared=prepared,
        shard=shard,
        audit=audit,
    )
    _assert_credential_free(manifest.to_dict())
    write_dataset_manifest(root, manifest)
    return dataset_id, {
        "dataset_id": dataset_id,
        "row_count": audit.row_count,
        "start_date": audit.start_date,
        "end_date": audit.end_date,
        "sha256": _sha256(shard),
    }


def commit(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    if state.get("status") == "applied" and state.get("source_schema_version") == SOURCE_SCHEMA_VERSION:
        return state
    if state.get("status") != "prepared":
        raise FinancialStatementUpdateError(f"statement_not_prepared:{state.get('status')}")
    installed: dict[str, Any] = {}
    dataset_ids: dict[str, str] = {}
    for spec in V2_STATEMENT_SPECS:
        path = Path(state["prepared_domains"][spec.domain]["path"])
        dataset_id, record = _install_domain(
            workspace,
            spec=spec,
            prepared=path,
        )
        dataset_ids[spec.domain] = dataset_id
        installed[spec.domain] = record
    root = qdp_v2_root(workspace)
    active = read_active_manifest(root)
    expected_unchanged = dict(state.get("input_dataset_ids", {}) or {})
    if str(dict(active.get("datasets", {}) or {}).get(DataDomain.BALANCE_SHEET_QUARTERLY, "")) != str(
        expected_unchanged.get(DataDomain.BALANCE_SHEET_QUARTERLY, "")
    ):
        raise FinancialStatementUpdateError("balance_statement_domain_drifted")
    for domain in (
        DataDomain.INCOME_STATEMENT_QUARTERLY,
        DataDomain.CASH_FLOW_STATEMENT_QUARTERLY,
    ):
        if str(dict(active.get("datasets", {}) or {}).get(domain, "")) != str(expected_unchanged.get(domain, "")):
            raise FinancialStatementUpdateError(f"unchanged_statement_domain_drifted:{domain}")
    active["datasets"] = {
        **dict(active.get("datasets", {}) or {}),
        **dataset_ids,
    }
    active["updated_at"] = utc_now()
    write_active_manifest(root, active)
    state.update({"status": "applied", "installed_domains": installed})
    _write_state(workspace, state)
    atomic_write_json(root / "audits" / f"{UPDATE_ID}.json", state)
    return state
