"""Financial Statement Update: repair responsibilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from quantlab.data.core.paths import qdp_paths
from quantlab.data.domains.contracts.schema import DataDomain
from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    utc_now,
    write_active_manifest,
    write_dataset_manifest,
)
from quantlab.data.qdp_v2.research_event_update.context import _assert_credential_free
from quantlab.data.qdp_v2.status import active_dataset_map

from .config import (
    BALANCE_CONFLICT_REPAIR_ID,
    BALANCE_EXTENSION_CONFLICT_COLUMN,
    END_DATE,
    PRIMARY_KEY,
    V2_STATEMENT_SPECS,
    FinancialStatementUpdateError,
)
from .context import (
    _normalized_path,
    _report_periods,
    _workspace,
)
from .install import (
    _install_domain,
)
from .prepare import (
    _balance_extension_hash_sql,
    _dataset_paths,
    _sql_paths,
)


@dataclass(frozen=True)
class _BalanceRepairPlan:
    spec: Any
    output: Path
    temporary: Path
    base_scan: str
    key_sql: str
    conflict_query: str


@dataclass(frozen=True)
class _BalanceRepairStats:
    row_count: int
    duplicate_count: int
    conflict_count: int
    forbidden_rows: int
    core_mismatch: int
    exact_conflicts: int
    missing_normalized_keys: int
    previous_conflict_count: int


def _balance_repair_plan(
    workspace: Path,
    *,
    current_id: str,
) -> _BalanceRepairPlan:
    domain = DataDomain.BALANCE_SHEET_QUARTERLY
    spec = next(item for item in V2_STATEMENT_SPECS if item.domain == domain)
    normalized_paths = [_normalized_path(workspace, spec, period) for period in _report_periods()]
    if any(not path.is_file() for path in normalized_paths):
        raise FinancialStatementUpdateError("balance_extension_normalized_evidence_missing")
    base_paths = _dataset_paths(workspace, domain=domain, dataset_id=current_id)
    base_scan = f"read_parquet([{_sql_paths(base_paths)}], union_by_name=true)"
    normalized_scan = f"read_parquet([{_sql_paths(normalized_paths)}], union_by_name=true, hive_partitioning=false)"
    output = (
        qdp_paths(workspace).data_dir / "qdp_runtime" / BALANCE_CONFLICT_REPAIR_ID / "prepared" / f"{domain}.parquet"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    key_sql = ",".join(f'"{column}"' for column in PRIMARY_KEY)
    conflict_query = f"""
      SELECT {key_sql},
        count(DISTINCT {_balance_extension_hash_sql()}) > 1
          AS {BALANCE_EXTENSION_CONFLICT_COLUMN}
      FROM {normalized_scan}
      GROUP BY {key_sql}
    """
    return _BalanceRepairPlan(
        spec=spec,
        output=output,
        temporary=output.with_suffix(".tmp.parquet"),
        base_scan=base_scan,
        key_sql=key_sql,
        conflict_query=conflict_query,
    )


def _materialize_balance_repair(plan: _BalanceRepairPlan) -> _BalanceRepairStats:
    with duckdb.connect() as connection:
        base_columns = [
            str(row[0]) for row in connection.execute(f"DESCRIBE SELECT * FROM {plan.base_scan}").fetchall()
        ]
        preserved_columns = [column for column in base_columns if column != BALANCE_EXTENSION_CONFLICT_COLUMN]
        base_select = ",".join(f'"{column}"' for column in preserved_columns)
        joined_select = ",".join(f'b."{column}"' for column in preserved_columns)
        previous_conflict_count = (
            int(
                connection.execute(
                    f"SELECT count(*) FILTER(WHERE {BALANCE_EXTENSION_CONFLICT_COLUMN}) FROM {plan.base_scan}"
                ).fetchone()[0]
                or 0
            )
            if BALANCE_EXTENSION_CONFLICT_COLUMN in base_columns
            else 0
        )
        quoted_temporary = str(plan.temporary).replace("'", "''")
        connection.execute(
            f"""
            COPY (
              SELECT {joined_select},
                coalesce(r.{BALANCE_EXTENSION_CONFLICT_COLUMN},false)
                  AS {BALANCE_EXTENSION_CONFLICT_COLUMN}
              FROM {plan.base_scan} b
              LEFT JOIN ({plan.conflict_query}) r USING({plan.key_sql})
              ORDER BY b.publish_date,b.symbol,b.report_date,b.report_type,
                       b.company_type,b.period_type
            ) TO '{quoted_temporary}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        output_scan = f"read_parquet('{quoted_temporary}')"
        missing_normalized_keys = int(
            connection.execute(
                "SELECT count(*) FILTER(WHERE r.symbol IS NULL) "
                f"FROM {plan.base_scan} b LEFT JOIN ({plan.conflict_query}) r "
                f"USING({plan.key_sql})"
            ).fetchone()[0]
        )
        output_stats = connection.execute(
            "SELECT count(*),"
            f"count(*)-count(DISTINCT concat_ws('|',{plan.key_sql})),"
            f"count(*) FILTER(WHERE {BALANCE_EXTENSION_CONFLICT_COLUMN}),"
            f"count(*) FILTER(WHERE publish_date>'{END_DATE}') "
            f"FROM {output_scan}"
        ).fetchone()
        core_mismatch = int(
            connection.execute(
                f"""
                SELECT count(*) FROM (
                  (SELECT {base_select} FROM {plan.base_scan})
                  EXCEPT ALL
                  (SELECT {base_select} FROM {output_scan})
                )
                """
            ).fetchone()[0]
        )
        exact_conflicts = int(
            connection.execute(
                f"SELECT count(*) FROM ({plan.conflict_query}) WHERE {BALANCE_EXTENSION_CONFLICT_COLUMN}"
            ).fetchone()[0]
        )
    return _BalanceRepairStats(
        row_count=int(output_stats[0]),
        duplicate_count=int(output_stats[1]),
        conflict_count=int(output_stats[2]),
        forbidden_rows=int(output_stats[3]),
        core_mismatch=core_mismatch,
        exact_conflicts=exact_conflicts,
        missing_normalized_keys=missing_normalized_keys,
        previous_conflict_count=previous_conflict_count,
    )


def _balance_repair_checks(
    stats: _BalanceRepairStats,
    *,
    expected_rows: int,
) -> dict[str, bool]:
    return {
        "row_count_unchanged": stats.row_count == expected_rows,
        "primary_key_unique": stats.duplicate_count == 0,
        "existing_non_conflict_columns_unchanged": stats.core_mismatch == 0,
        "extension_conflicts_are_exact": stats.conflict_count == stats.exact_conflicts,
        "all_active_keys_have_normalized_evidence": stats.missing_normalized_keys == 0,
        "forbidden_2026_rows": stats.forbidden_rows == 0,
    }


def _install_balance_repair(
    workspace: Path,
    *,
    root: Path,
    current_id: str,
    plan: _BalanceRepairPlan,
    stats: _BalanceRepairStats,
    checks: dict[str, bool],
) -> tuple[str, Any]:
    domain = DataDomain.BALANCE_SHEET_QUARTERLY
    dataset_id, installed = _install_domain(workspace, spec=plan.spec, prepared=plan.output)
    installed_path = dataset_manifest_for_id(root, dataset_id, domain)
    assert installed_path is not None
    installed_manifest = read_dataset_manifest(installed_path)
    repaired_manifest = DatasetManifest.from_mapping(
        {
            **installed_manifest.to_dict(),
            "source": {
                **dict(installed_manifest.source or {}),
                "semantic_repair_id": BALANCE_CONFLICT_REPAIR_ID,
                "upstream_dataset_id": current_id,
            },
            "quality": {
                **dict(installed_manifest.quality or {}),
                "balance_extension_source_conflict_count": stats.conflict_count,
                "prior_overbroad_balance_source_conflict_count": stats.previous_conflict_count,
                "semantic_repair_checks": checks,
            },
            "notes": [
                *list(installed_manifest.notes or []),
                "immutable successor; non-conflict columns and values are unchanged",
                "the extension conflict flag compares only the four extension numeric values",
            ],
        }
    )
    _assert_credential_free(repaired_manifest.to_dict())
    write_dataset_manifest(root, repaired_manifest)

    latest_active = read_active_manifest(root)
    latest = active_dataset_map(latest_active)
    if latest.get(domain) != current_id:
        raise FinancialStatementUpdateError("active_balance_dataset_drifted")
    latest_active["datasets"] = {**latest, domain: dataset_id}
    latest_active["updated_at"] = utc_now()
    write_active_manifest(root, latest_active)
    if active_dataset_map(read_active_manifest(root)).get(domain) != dataset_id:
        raise FinancialStatementUpdateError("balance_conflict_active_switch_failed")
    return dataset_id, installed


def repair_balance_extension_conflict_metadata(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    """Install an exact four-field extension-conflict flag without changing core data."""

    workspace = _workspace(workspace_root)
    root = qdp_v2_root(workspace)
    active = read_active_manifest(root)
    current = active_dataset_map(active)
    domain = DataDomain.BALANCE_SHEET_QUARTERLY
    current_id = str(current.get(domain, ""))
    current_path = dataset_manifest_for_id(root, current_id, domain)
    if current_path is None:
        raise FinancialStatementUpdateError("active_balance_manifest_missing")
    current_manifest = read_dataset_manifest(current_path)
    existing_columns = [str(item.get("name", "")) for item in current_manifest.schema]
    if BALANCE_EXTENSION_CONFLICT_COLUMN in existing_columns and current_manifest.contract_version in {
        "qdp_v2_balance_sheet_quarterly_pit_v4",
        "qdp_v2_balance_sheet_quarterly_pit_v5",
    }:
        return {
            "status": "already_repaired",
            "repair_id": BALANCE_CONFLICT_REPAIR_ID,
            "dataset_id": current_id,
        }

    plan = _balance_repair_plan(workspace, current_id=current_id)
    stats = _materialize_balance_repair(plan)
    checks = _balance_repair_checks(stats, expected_rows=current_manifest.row_count)
    if not all(checks.values()):
        plan.temporary.unlink(missing_ok=True)
        raise FinancialStatementUpdateError(f"balance_conflict_metadata_contract_failed:{checks}")
    os.replace(plan.temporary, plan.output)
    dataset_id, installed = _install_balance_repair(
        workspace,
        root=root,
        current_id=current_id,
        plan=plan,
        stats=stats,
        checks=checks,
    )

    payload = {
        "status": "applied",
        "repair_id": BALANCE_CONFLICT_REPAIR_ID,
        "input_dataset_id": current_id,
        "installed_domains": {domain: installed},
        "checks": checks,
        "statistics": {
            "row_count": stats.row_count,
            "balance_extension_source_conflict_count": stats.conflict_count,
            "prior_overbroad_balance_source_conflict_count": stats.previous_conflict_count,
            "generic_only_conflict_count": max(0, stats.previous_conflict_count - stats.conflict_count),
            "request_2026_count": 0,
        },
        "updated_at": utc_now(),
    }
    _assert_credential_free(payload)
    atomic_write_json(root / "audits" / f"{BALANCE_CONFLICT_REPAIR_ID}.json", payload)
    return payload
