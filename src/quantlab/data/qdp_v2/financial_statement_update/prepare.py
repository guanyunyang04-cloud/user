"""Financial Statement Update: prepare responsibilities."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from quantlab.core.io import sha256_file as _sha256
from quantlab.data.domains.contracts.schema import DataDomain
from quantlab.data.qdp_v2.manifest import (
    dataset_manifest_for_id,
    qdp_v2_root,
    read_dataset_manifest,
    resolve_manifest_path,
)
from quantlab.data.qdp_v2.research_event_update.context import _identity_symbols, _open_dates, _write_parquet

from .config import (
    BALANCE_DERIVED_COLUMNS,
    BALANCE_EXTENSION_CONFLICT_COLUMN,
    BALANCE_EXTENSION_NUMERIC_COLUMNS,
    BALANCE_SEMANTIC_CONFLICT_COLUMN,
    BALANCE_SEMANTIC_NUMERIC_COLUMNS,
    BALANCE_SEMANTIC_SOURCE_COLUMNS,
    COMMON_OUTPUT_COLUMNS,
    PRIMARY_KEY,
    SOURCE_SCHEMA_VERSION,
    V2_STATEMENT_SPECS,
    FinancialStatementUpdateError,
    StatementSpec,
)
from .context import (
    _normalized_path,
    _raw_path,
    _read_state,
    _report_periods,
    _runtime,
    _workspace,
    _write_state,
)
from .normalize import (
    _normalize_statement_part,
)


def _sql_paths(paths: Sequence[Path]) -> str:
    return ",".join(f"'{str(path).replace(chr(39), chr(39) * 2)}'" for path in paths)


def _balance_extension_hash_sql(*, prefix: str = "") -> str:
    values = ",".join(
        f"coalesce(cast({prefix}\"{column}\" AS VARCHAR),'<NA>')" for column in BALANCE_EXTENSION_NUMERIC_COLUMNS
    )
    return f"md5(concat_ws('|',{values}))"


def _balance_semantic_hash_sql(*, prefix: str = "") -> str:
    values = ",".join(
        f"coalesce(cast({prefix}\"{column}\" AS VARCHAR),'<NA>')" for column in BALANCE_SEMANTIC_NUMERIC_COLUMNS
    )
    return f"md5(concat_ws('|',{values}))"


def _normalized_statement_parts(
    workspace: Path,
    *,
    spec: StatementSpec,
    identity_symbols: set[str],
    open_dates: np.ndarray,
) -> tuple[list[Path], int]:
    required = {
        *COMMON_OUTPUT_COLUMNS,
        *spec.metric_map.values(),
        *(BALANCE_DERIVED_COLUMNS if spec.domain == DataDomain.BALANCE_SHEET_QUARTERLY else ()),
    }
    paths: list[Path] = []
    row_count = 0
    for period in _report_periods():
        raw_path = _raw_path(workspace, spec, period)
        if not raw_path.is_file():
            raise FinancialStatementUpdateError(f"statement_raw_part_missing:{spec.name}:{period}")
        output_path = _normalized_path(workspace, spec, period)
        stale = (
            not output_path.is_file()
            or raw_path.stat().st_mtime_ns > output_path.stat().st_mtime_ns
            or not required.issubset(pq.read_schema(output_path).names)
        )
        if stale:
            normalized = _normalize_statement_part(
                pd.read_parquet(raw_path),
                spec=spec,
                identity_symbols=identity_symbols,
                open_dates=open_dates,
            )
            _write_parquet(normalized, output_path)
        row_count += int(pq.ParquetFile(output_path).metadata.num_rows)
        paths.append(output_path)
    return paths, row_count


def _prepared_statement_columns(spec: StatementSpec) -> list[str]:
    is_balance = spec.domain == DataDomain.BALANCE_SHEET_QUARTERLY
    return [
        *COMMON_OUTPUT_COLUMNS,
        *spec.metric_map.values(),
        *(BALANCE_DERIVED_COLUMNS if is_balance else ()),
        "source_duplicate_count",
        "source_conflict",
        *([BALANCE_EXTENSION_CONFLICT_COLUMN] if is_balance else []),
        *([BALANCE_SEMANTIC_CONFLICT_COLUMN] if is_balance else []),
        "lag_policy",
        "source",
    ]


def _statement_conflict_sql(spec: StatementSpec, key_sql: str) -> tuple[str, str]:
    if spec.domain != DataDomain.BALANCE_SHEET_QUARTERLY:
        return "", ""
    extension = (
        ",\n          count(DISTINCT "
        f"{_balance_extension_hash_sql()}) OVER (PARTITION BY {key_sql}) > 1 "
        f"AS {BALANCE_EXTENSION_CONFLICT_COLUMN}"
    )
    semantic = (
        ",\n          count(DISTINCT "
        f"{_balance_semantic_hash_sql()}) OVER (PARTITION BY {key_sql}) > 1 "
        f"AS {BALANCE_SEMANTIC_CONFLICT_COLUMN}"
    )
    return extension, semantic


def _write_prepared_statement(
    *,
    spec: StatementSpec,
    normalized_paths: list[Path],
    temporary: Path,
) -> tuple[Any, ...]:
    key_sql = ",".join(f'"{column}"' for column in PRIMARY_KEY)
    select_sql = ",".join(f'"{column}"' for column in _prepared_statement_columns(spec))
    extension_conflict_sql, semantic_conflict_sql = _statement_conflict_sql(spec, key_sql)
    escaped = str(temporary).replace("'", "''")
    sql = f"""
    COPY (
      WITH source AS (
        SELECT * FROM read_parquet([{_sql_paths(normalized_paths)}], union_by_name=true)
      ), ranked AS (
        SELECT *,
          count(*) OVER (PARTITION BY {key_sql}) AS source_duplicate_count,
          count(DISTINCT _metric_hash) OVER (PARTITION BY {key_sql}) > 1 AS source_conflict{extension_conflict_sql}{semantic_conflict_sql},
          row_number() OVER (
            PARTITION BY {key_sql}
            ORDER BY update_flag DESC, _completeness DESC, _source_row_hash DESC
          ) AS rn
        FROM source
      )
      SELECT {select_sql}
      FROM ranked WHERE rn=1
      ORDER BY publish_date,symbol,report_date,report_type,company_type,period_type
    ) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    extension_sum = (
        f"sum({BALANCE_EXTENSION_CONFLICT_COLUMN})"
        if spec.domain == DataDomain.BALANCE_SHEET_QUARTERLY
        else "NULL"
    )
    semantic_sum = (
        f"sum({BALANCE_SEMANTIC_CONFLICT_COLUMN})"
        if spec.domain == DataDomain.BALANCE_SHEET_QUARTERLY
        else "NULL"
    )
    connection = duckdb.connect()
    try:
        connection.execute(sql)
        return connection.execute(
            "SELECT count(*),sum(source_duplicate_count>1),sum(source_conflict),"
            f"{extension_sum},{semantic_sum},min(publish_date),max(publish_date) "
            f"FROM read_parquet('{escaped}')"
        ).fetchone()
    finally:
        connection.close()


def _prepare_statement(
    workspace: Path,
    *,
    spec: StatementSpec,
    identity_symbols: set[str],
    open_dates: np.ndarray,
    prepared_filename: str | None = None,
) -> dict[str, Any]:
    normalized_paths, normalized_rows = _normalized_statement_parts(
        workspace,
        spec=spec,
        identity_symbols=identity_symbols,
        open_dates=open_dates,
    )
    prepared = _runtime(workspace) / "prepared" / (prepared_filename or f"{spec.domain}.parquet")
    prepared.parent.mkdir(parents=True, exist_ok=True)
    temporary = prepared.with_suffix(".tmp.parquet")
    row = _write_prepared_statement(spec=spec, normalized_paths=normalized_paths, temporary=temporary)
    os.replace(temporary, prepared)
    return {
        "status": "completed",
        "domain": spec.domain,
        "path": str(prepared),
        "normalized_row_count": normalized_rows,
        "row_count": int(row[0]),
        "deduplicated_row_count": normalized_rows - int(row[0]),
        "rows_from_duplicate_source_groups": int(row[1] or 0),
        "source_conflict_row_count": int(row[2] or 0),
        "balance_extension_source_conflict_count": int(row[3] or 0),
        "balance_semantic_source_conflict_count": int(row[4] or 0),
        "start_date": str(row[5]),
        "end_date": str(row[6]),
        "sha256": _sha256(prepared),
    }


def _dataset_paths(
    workspace: Path,
    *,
    domain: str,
    dataset_id: str,
) -> list[Path]:
    root = qdp_v2_root(workspace)
    manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
    if manifest_path is None:
        raise FinancialStatementUpdateError(f"statement_input_manifest_missing:{domain}:{dataset_id}")
    manifest = read_dataset_manifest(manifest_path)
    paths = [resolve_manifest_path(item.path, root=root) for item in manifest.shards]
    if not paths or any(not path.is_file() for path in paths):
        raise FinancialStatementUpdateError(f"statement_input_shard_missing:{domain}:{dataset_id}")
    return paths


def _balance_refreshed_columns() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *BALANCE_EXTENSION_NUMERIC_COLUMNS,
                *BALANCE_SEMANTIC_SOURCE_COLUMNS,
                *BALANCE_DERIVED_COLUMNS,
                BALANCE_EXTENSION_CONFLICT_COLUMN,
                BALANCE_SEMANTIC_CONFLICT_COLUMN,
            )
        )
    )


def _balance_alignment_columns(
    connection: duckdb.DuckDBPyConnection,
    *,
    base_scan: str,
    refreshed_scan: str,
    refreshed_columns: tuple[str, ...],
) -> list[str]:
    base_columns = [str(row[0]) for row in connection.execute(f"DESCRIBE SELECT * FROM {base_scan}").fetchall()]
    refreshed_schema = {
        str(row[0]) for row in connection.execute(f"DESCRIBE SELECT * FROM {refreshed_scan}").fetchall()
    }
    missing = sorted(set(refreshed_columns).difference(refreshed_schema))
    if missing:
        raise FinancialStatementUpdateError(f"balance_refreshed_columns_missing:{missing}")
    return [column for column in base_columns if column not in set(refreshed_columns)]


def _write_aligned_balance(
    connection: duckdb.DuckDBPyConnection,
    *,
    base_scan: str,
    refreshed_scan: str,
    temporary: Path,
    preserved_columns: list[str],
    refreshed_columns: tuple[str, ...],
) -> None:
    key_sql = ",".join(f'"{column}"' for column in PRIMARY_KEY)
    output_select = ",".join(
        [
            *(f'b."{column}"' for column in preserved_columns),
            *(f'r."{column}"' for column in refreshed_columns),
        ]
    )
    escaped = str(temporary).replace("'", "''")
    connection.execute(
        f"""
        COPY (
          SELECT {output_select}
          FROM {base_scan} b
          LEFT JOIN {refreshed_scan} r USING({key_sql})
          ORDER BY b.publish_date,b.symbol,b.report_date,b.report_type,
                   b.company_type,b.period_type
        ) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def _balance_alignment_diagnostics(
    connection: duckdb.DuckDBPyConnection,
    *,
    base_scan: str,
    refreshed_scan: str,
    output_scan: str,
    preserved_columns: list[str],
) -> tuple[int, int, int, int, int]:
    key_sql = ",".join(f'"{column}"' for column in PRIMARY_KEY)
    base_select = ",".join(f'"{column}"' for column in preserved_columns)
    base_count = int(connection.execute(f"SELECT count(*) FROM {base_scan}").fetchone()[0])
    output_count = int(connection.execute(f"SELECT count(*) FROM {output_scan}").fetchone()[0])
    unmatched = int(
        connection.execute(
            f"""
            SELECT count(*) FROM {base_scan} b
            LEFT JOIN (
              SELECT {key_sql},true AS refreshed_present FROM {refreshed_scan}
            ) r USING({key_sql})
            WHERE NOT coalesce(r.refreshed_present,false)
            """
        ).fetchone()[0]
    )
    core_mismatch = int(
        connection.execute(
            f"""
            SELECT count(*) FROM (
              (SELECT {base_select} FROM {base_scan})
              EXCEPT ALL
              (SELECT {base_select} FROM {output_scan})
            )
            """
        ).fetchone()[0]
    )
    duplicate_count = int(
        connection.execute(
            f"SELECT count(*) FROM (SELECT {key_sql},count(*) n FROM {output_scan} GROUP BY {key_sql} HAVING n>1)"
        ).fetchone()[0]
    )
    return base_count, output_count, unmatched, core_mismatch, duplicate_count


def _align_balance_to_active(
    workspace: Path,
    *,
    refreshed: Path,
    input_dataset_id: str,
) -> dict[str, Any]:
    domain = DataDomain.BALANCE_SHEET_QUARTERLY
    base_paths = _dataset_paths(
        workspace,
        domain=domain,
        dataset_id=input_dataset_id,
    )
    output = _runtime(workspace) / "prepared" / f"{domain}.parquet"
    temporary = output.with_suffix(".tmp.parquet")
    refreshed_columns = _balance_refreshed_columns()
    base_scan = f"read_parquet([{_sql_paths(base_paths)}], union_by_name=true)"
    refreshed_scan = f"read_parquet('{str(refreshed).replace(chr(39), chr(39) * 2)}')"
    connection = duckdb.connect()
    try:
        preserved_columns = _balance_alignment_columns(
            connection,
            base_scan=base_scan,
            refreshed_scan=refreshed_scan,
            refreshed_columns=refreshed_columns,
        )
        _write_aligned_balance(
            connection,
            base_scan=base_scan,
            refreshed_scan=refreshed_scan,
            temporary=temporary,
            preserved_columns=preserved_columns,
            refreshed_columns=refreshed_columns,
        )
        output_scan = f"read_parquet('{str(temporary).replace(chr(39), chr(39) * 2)}')"
        base_count, output_count, unmatched, core_mismatch, duplicate_count = _balance_alignment_diagnostics(
            connection,
            base_scan=base_scan,
            refreshed_scan=refreshed_scan,
            output_scan=output_scan,
            preserved_columns=preserved_columns,
        )
    finally:
        connection.close()
    if base_count != output_count or unmatched or core_mismatch or duplicate_count:
        raise FinancialStatementUpdateError(
            "balance_alignment_contract_failed:"
            f"base={base_count}:output={output_count}:"
            f"unmatched={unmatched}:core_mismatch={core_mismatch}:"
            f"duplicates={duplicate_count}"
        )
    os.replace(temporary, output)
    return {
        "status": "completed",
        "domain": domain,
        "path": str(output),
        "input_dataset_id": input_dataset_id,
        "refreshed_path": str(refreshed),
        "row_count": output_count,
        "input_row_count": base_count,
        "unmatched_refreshed_key_count": unmatched,
        "existing_core_value_mismatch_count": core_mismatch,
        "primary_key_duplicate_count": duplicate_count,
        "sha256": _sha256(output),
    }


def prepare(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    if state.get("status") in {"prepared", "applied"} and state.get("source_schema_version") == SOURCE_SCHEMA_VERSION:
        return state
    if state.get("status") != "downloaded":
        raise FinancialStatementUpdateError(f"statement_not_downloaded:{state.get('status')}")
    identity = set(_identity_symbols(workspace))
    open_dates = _open_dates(workspace)
    balance_spec = V2_STATEMENT_SPECS[0]
    refreshed = _prepare_statement(
        workspace,
        spec=balance_spec,
        identity_symbols=identity,
        open_dates=open_dates,
        prepared_filename="balance_sheet_quarterly_refreshed.parquet",
    )
    input_dataset_id = str(dict(state.get("input_dataset_ids", {}) or {}).get(DataDomain.BALANCE_SHEET_QUARTERLY, ""))
    if not input_dataset_id:
        raise FinancialStatementUpdateError("balance_input_dataset_id_missing")
    domains = {
        balance_spec.domain: {
            **_align_balance_to_active(
                workspace,
                refreshed=Path(refreshed["path"]),
                input_dataset_id=input_dataset_id,
            ),
            "refreshed_normalization": refreshed,
        }
    }
    state.update(
        {
            "status": "prepared",
            "identity_symbol_count": len(identity),
            "prepared_domains": domains,
        }
    )
    _write_state(workspace, state)
    return state
