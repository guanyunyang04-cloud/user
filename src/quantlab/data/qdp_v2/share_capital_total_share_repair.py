from __future__ import annotations

"""Repair missing total shares from same-day raw factors and rebuild market value."""

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from quantlab.data.core.json_io import json_safe
from quantlab.data.core.paths import qdp_paths
from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    _manifest_schema_from_arrow,
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
from quantlab.data.qdp_v2.pretraining_repair_baseline import (
    _quality_diagnostics,
)
from quantlab.data.qdp_v2.research_event_update import (
    _assert_credential_free,
    _sha256,
)
from quantlab.data.qdp_v2.status import active_dataset_map

REPAIR_ID = "share_capital_total_share_repair_v1"
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"
TOTAL_SHARE_UNIT_MULTIPLIER = 10_000.0
BOUNDARY_RELATIVE_TOLERANCE = 0.0001
SHARE_DOMAIN = "share_capital"
VALUATION_DOMAIN = "valuation"
FACTOR_DOMAIN = "stk_factor_pro_raw"
DAILY_DOMAIN = "market_daily_raw"


class ShareCapitalRepairError(RuntimeError):
    pass


def _workspace(value: str | Path | None) -> Path:
    return Path(value or Path.cwd()).resolve()


def _runtime(workspace: Path) -> Path:
    path = qdp_paths(workspace).data_dir / "qdp_runtime" / REPAIR_ID
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _state_path(workspace: Path) -> Path:
    return _runtime(workspace) / "state.json"


def _read_state(workspace: Path) -> dict[str, Any]:
    path = _state_path(workspace)
    if not path.is_file():
        return {
            "repair_id": REPAIR_ID,
            "status": "pending",
            "start_date": START_DATE,
            "end_date": END_DATE,
        }
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_state(workspace: Path, state: Mapping[str, Any]) -> None:
    payload = {**dict(state), "updated_at": utc_now()}
    _assert_credential_free(payload)
    atomic_write_json(_state_path(workspace), payload)


def _sql_paths(paths: Sequence[Path]) -> str:
    return ",".join(
        f"'{path.resolve().as_posix().replace(chr(39), chr(39) * 2)}'"
        for path in paths
    )


def _dataset(
    workspace: Path,
    *,
    domain: str,
    dataset_id: str | None = None,
) -> tuple[DatasetManifest, list[Path]]:
    root = qdp_v2_root(workspace)
    selected_id = dataset_id or active_dataset_map(read_active_manifest(root)).get(domain)
    if not selected_id:
        raise ShareCapitalRepairError(f"active_dataset_missing:{domain}")
    manifest_path = dataset_manifest_for_id(root, selected_id, domain)
    if manifest_path is None:
        raise ShareCapitalRepairError(f"dataset_manifest_missing:{domain}:{selected_id}")
    manifest = read_dataset_manifest(manifest_path)
    paths = [resolve_manifest_path(item.path, root=root) for item in manifest.shards]
    if not paths or any(not path.is_file() for path in paths):
        raise ShareCapitalRepairError(f"dataset_shard_missing:{domain}:{selected_id}")
    return manifest, paths


def _copy_query(
    connection: duckdb.DuckDBPyConnection,
    *,
    query: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    quoted = temporary.resolve().as_posix().replace("'", "''")
    connection.execute(
        f"COPY ({query}) TO '{quoted}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    os.replace(temporary, path)


def _content_dataset_id(domain: str, paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return f"{domain}__{digest.hexdigest()[:24]}"


def _repair_value(existing: Any, provider_total_share_wan: Any) -> float | None:
    if pd.notna(existing):
        return float(existing)
    provider = pd.to_numeric(
        pd.Series([provider_total_share_wan]), errors="coerce"
    ).iloc[0]
    if pd.isna(provider) or float(provider) <= 0:
        return None
    return float(provider) * TOTAL_SHARE_UNIT_MULTIPLIER


def _market_value(close: Any, shares: Any) -> float | None:
    price = pd.to_numeric(pd.Series([close]), errors="coerce").iloc[0]
    count = pd.to_numeric(pd.Series([shares]), errors="coerce").iloc[0]
    if pd.isna(price) or pd.isna(count):
        return None
    return float(price) * float(count)


def prepare(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    if state.get("status") in {"prepared", "applied"}:
        return state
    root = qdp_v2_root(workspace)
    active = active_dataset_map(read_active_manifest(root))
    input_ids = {
        domain: active[domain]
        for domain in (SHARE_DOMAIN, VALUATION_DOMAIN, FACTOR_DOMAIN, DAILY_DOMAIN)
    }
    share_manifest, share_paths = _dataset(workspace, domain=SHARE_DOMAIN)
    valuation_manifest, valuation_paths = _dataset(workspace, domain=VALUATION_DOMAIN)
    _, factor_paths = _dataset(workspace, domain=FACTOR_DOMAIN)
    _, daily_paths = _dataset(workspace, domain=DAILY_DOMAIN)
    runtime = _runtime(workspace)
    temp = runtime / "duckdb_tmp"
    temp.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET threads=4")
    connection.execute(
        f"SET temp_directory='{temp.resolve().as_posix().replace(chr(39), chr(39) * 2)}'"
    )
    share_scan = (
        f"read_parquet([{_sql_paths(share_paths)}], union_by_name=true, "
        "hive_partitioning=false)"
    )
    factor_scan = (
        f"read_parquet([{_sql_paths(factor_paths)}], union_by_name=true, "
        "hive_partitioning=false)"
    )
    valuation_scan = (
        f"read_parquet([{_sql_paths(valuation_paths)}], union_by_name=true, "
        "hive_partitioning=false)"
    )
    daily_scan = (
        f"read_parquet([{_sql_paths(daily_paths)}], union_by_name=true, "
        "hive_partitioning=false)"
    )
    try:
        connection.execute(
            f"""
            CREATE TEMP TABLE repair_candidates AS
            SELECT s.symbol,s.trade_date,
                   try_cast(f.total_share AS DOUBLE) AS provider_total_share_wan,
                   try_cast(f.total_share AS DOUBLE)*{TOTAL_SHARE_UNIT_MULTIPLIER}
                     AS repaired_total_share,
                   s.total_share AS prior_total_share,
                   s.total_share_source_date AS prior_total_share_source_date,
                   'observed' AS status,
                   'stk_factor_pro_raw.total_share' AS fill_source,
                   'same_day_wan_shares_x10000' AS fill_method,
                   s.trade_date AS source_date
            FROM {share_scan} s
            JOIN {factor_scan} f USING(symbol,trade_date)
            WHERE s.trade_date BETWEEN '{START_DATE}' AND '{END_DATE}'
              AND s.total_share IS NULL
              AND try_cast(f.total_share AS DOUBLE)>0
            """
        )
        candidate_count = int(
            connection.execute("SELECT count(*) FROM repair_candidates").fetchone()[0]
        )
        if candidate_count != 559_513:
            raise ShareCapitalRepairError(
                f"unexpected_repair_candidate_count:{candidate_count}"
            )
        candidate_path = runtime / "inventory" / "repair_candidates.parquet"
        _copy_query(
            connection,
            query="SELECT * FROM repair_candidates ORDER BY trade_date,symbol",
            path=candidate_path,
        )
        unresolved_query = f"""
          SELECT s.symbol,s.trade_date,'unknown' AS status,
                 'same_day_stk_factor_total_share_unavailable' AS reason
          FROM {share_scan} s
          LEFT JOIN repair_candidates c USING(symbol,trade_date)
          WHERE s.trade_date BETWEEN '{START_DATE}' AND '{END_DATE}'
            AND s.total_share IS NULL AND c.symbol IS NULL
          ORDER BY s.trade_date,s.symbol
        """
        unresolved_path = runtime / "inventory" / "unresolved.parquet"
        _copy_query(connection, query=unresolved_query, path=unresolved_path)
        unresolved_count = int(
            connection.execute(f"SELECT count(*) FROM ({unresolved_query})").fetchone()[0]
        )

        boundary_query = f"""
          WITH ordered AS (
            SELECT symbol,trade_date,total_share,
              last_value(total_share IGNORE NULLS) OVER (
                PARTITION BY symbol ORDER BY trade_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
              ) AS previous_known_total_share,
              first_value(total_share IGNORE NULLS) OVER (
                PARTITION BY symbol ORDER BY trade_date
                ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING
              ) AS next_known_total_share
            FROM {share_scan}
            WHERE trade_date<='{END_DATE}'
          )
          SELECT c.symbol,c.trade_date,c.repaired_total_share,
                 o.previous_known_total_share,o.next_known_total_share,
                 CASE WHEN o.previous_known_total_share>0 THEN
                   abs(c.repaired_total_share-o.previous_known_total_share)
                     /o.previous_known_total_share END AS previous_relative_change,
                 CASE WHEN o.next_known_total_share>0 THEN
                   abs(c.repaired_total_share-o.next_known_total_share)
                     /o.next_known_total_share END AS next_relative_change,
                 'review_share_change_boundary' AS status
          FROM repair_candidates c JOIN ordered o USING(symbol,trade_date)
          WHERE (o.previous_known_total_share>0 AND
                 abs(c.repaired_total_share-o.previous_known_total_share)
                   /o.previous_known_total_share>{BOUNDARY_RELATIVE_TOLERANCE})
             OR (o.next_known_total_share>0 AND
                 abs(c.repaired_total_share-o.next_known_total_share)
                   /o.next_known_total_share>{BOUNDARY_RELATIVE_TOLERANCE})
          ORDER BY c.trade_date,c.symbol
        """
        boundary_path = runtime / "inventory" / "share_change_boundary_review.parquet"
        _copy_query(connection, query=boundary_query, path=boundary_path)
        boundary_count = int(
            connection.execute(f"SELECT count(*) FROM ({boundary_query})").fetchone()[0]
        )

        prepared_share_root = runtime / "prepared" / SHARE_DOMAIN
        prepared_share_paths: list[Path] = []
        share_fill_method = "same_day_stk_factor_pro_raw_total_share_wan_x10000"
        for year in range(2010, 2027):
            path = prepared_share_root / f"year={year}" / "part-0000.parquet"
            query = f"""
              SELECT s.symbol,s.trade_date,
                CASE WHEN c.symbol IS NOT NULL THEN c.repaired_total_share
                     ELSE s.total_share END AS total_share,
                s.float_share,s.restricted_share,
                CASE WHEN c.symbol IS NOT NULL THEN c.source_date
                     ELSE s.total_share_source_date END AS total_share_source_date,
                s.float_share_source_date,s.restricted_share_source_date,
                CASE WHEN c.symbol IS NOT NULL THEN '{share_fill_method}'
                     ELSE s.share_fill_method END AS share_fill_method,
                CASE WHEN c.symbol IS NOT NULL THEN
                     concat(s.source,'+stk_factor_pro_raw_total_share_repair')
                     ELSE s.source END AS source
              FROM {share_scan} s
              LEFT JOIN repair_candidates c USING(symbol,trade_date)
              WHERE s.trade_date BETWEEN '{year}-01-01' AND '{year}-12-31'
              ORDER BY s.trade_date,s.symbol
            """
            _copy_query(connection, query=query, path=path)
            prepared_share_paths.append(path)

        prepared_share_scan = (
            f"read_parquet([{_sql_paths(prepared_share_paths)}], "
            "union_by_name=true, hive_partitioning=false)"
        )
        prepared_valuation_root = runtime / "prepared" / VALUATION_DOMAIN
        prepared_valuation_paths: list[Path] = []
        for year in range(2010, 2027):
            path = prepared_valuation_root / f"year={year}" / "part-0000.parquet"
            query = f"""
              SELECT v.symbol,v.trade_date,
                CASE WHEN v.trade_date<='{END_DATE}'
                     THEN d.close*s.total_share ELSE v.total_mv END AS total_mv,
                CASE WHEN v.trade_date<='{END_DATE}'
                     THEN d.close*s.float_share ELSE v.circ_mv END AS circ_mv,
                v.pe,v.pb,v.turnover_rate,
                CASE WHEN c.symbol IS NOT NULL THEN
                  concat(v.source,'+qdp_close_x_repaired_total_share')
                  ELSE v.source END AS source
              FROM {valuation_scan} v
              JOIN {prepared_share_scan} s USING(symbol,trade_date)
              JOIN {daily_scan} d USING(symbol,trade_date)
              LEFT JOIN repair_candidates c USING(symbol,trade_date)
              WHERE v.trade_date BETWEEN '{year}-01-01' AND '{year}-12-31'
              ORDER BY v.trade_date,v.symbol
            """
            _copy_query(connection, query=query, path=path)
            prepared_valuation_paths.append(path)

        prepared_valuation_scan = (
            f"read_parquet([{_sql_paths(prepared_valuation_paths)}], "
            "union_by_name=true, hive_partitioning=false)"
        )
        share_stats = connection.execute(
            f"""
            SELECT count(*),min(trade_date),max(trade_date),
              count(*) FILTER(WHERE trade_date<='{END_DATE}' AND total_share IS NULL),
              count(*) FILTER(WHERE trade_date>'{END_DATE}')
            FROM {prepared_share_scan}
            """
        ).fetchone()
        valuation_stats = connection.execute(
            f"SELECT count(*),min(trade_date),max(trade_date),"
            f"count(*) FILTER(WHERE trade_date>'{END_DATE}') "
            f"FROM {prepared_valuation_scan}"
        ).fetchone()
        existing_share_mismatch = int(
            connection.execute(
                f"""
                SELECT count(*) FROM {share_scan} old
                JOIN {prepared_share_scan} new USING(symbol,trade_date)
                WHERE old.trade_date<='{END_DATE}' AND old.total_share IS NOT NULL
                  AND new.total_share IS DISTINCT FROM old.total_share
                """
            ).fetchone()[0]
        )
        share_2026_mismatch = int(
            connection.execute(
                f"""
                SELECT count(*) FROM (
                  (SELECT * FROM {share_scan} WHERE trade_date>'{END_DATE}')
                  EXCEPT ALL
                  (SELECT * FROM {prepared_share_scan} WHERE trade_date>'{END_DATE}')
                )
                """
            ).fetchone()[0]
        )
        valuation_2026_mismatch = int(
            connection.execute(
                f"""
                SELECT count(*) FROM (
                  (SELECT * FROM {valuation_scan} WHERE trade_date>'{END_DATE}')
                  EXCEPT ALL
                  (SELECT * FROM {prepared_valuation_scan} WHERE trade_date>'{END_DATE}')
                )
                """
            ).fetchone()[0]
        )
        formula_errors = connection.execute(
            f"""
            SELECT
              count(*) FILTER(WHERE s.total_share IS NOT NULL AND
                (v.total_mv IS NULL OR abs(v.total_mv-d.close*s.total_share)>
                  greatest(1.0,abs(d.close*s.total_share))*1e-10)),
              count(*) FILTER(WHERE s.float_share IS NOT NULL AND
                (v.circ_mv IS NULL OR abs(v.circ_mv-d.close*s.float_share)>
                  greatest(1.0,abs(d.close*s.float_share))*1e-10))
            FROM {prepared_valuation_scan} v
            JOIN {prepared_share_scan} s USING(symbol,trade_date)
            JOIN {daily_scan} d USING(symbol,trade_date)
            WHERE v.trade_date BETWEEN '{START_DATE}' AND '{END_DATE}'
            """
        ).fetchone()
        quality_paths = _quality_diagnostics(workspace)
        quality_missing = None
        if quality_paths:
            quality_scan = (
                f"read_parquet([{_sql_paths(quality_paths)}], "
                "union_by_name=true, hive_partitioning=false)"
            )
            quality_missing = int(
                connection.execute(
                    f"""
                    SELECT count(*) FROM {quality_scan} q
                    JOIN {prepared_share_scan} s USING(symbol,trade_date)
                    WHERE q.trade_date BETWEEN '2011-01-01' AND '{END_DATE}'
                      AND q.quality_liquidity_keep AND s.total_share IS NULL
                    """
                ).fetchone()[0]
            )
    finally:
        connection.close()

    checks = {
        "candidate_count_expected": candidate_count == 559_513,
        "unresolved_count_expected": unresolved_count == 3_290,
        "share_row_count_unchanged": int(share_stats[0]) == share_manifest.row_count,
        "share_date_range_unchanged": str(share_stats[1]) == share_manifest.start_date
        and str(share_stats[2]) == share_manifest.end_date,
        "valuation_row_count_unchanged": int(valuation_stats[0])
        == valuation_manifest.row_count,
        "valuation_date_range_unchanged": str(valuation_stats[1])
        == valuation_manifest.start_date
        and str(valuation_stats[2]) == valuation_manifest.end_date,
        "existing_nonnull_total_share_unchanged": existing_share_mismatch == 0,
        "quality_liquidity_total_share_missing_zero": quality_missing == 0,
        "total_mv_formula_valid": int(formula_errors[0] or 0) == 0,
        "circ_mv_formula_valid": int(formula_errors[1] or 0) == 0,
        "out_of_scope_2026_share_values_unchanged": share_2026_mismatch == 0,
        "out_of_scope_2026_valuation_values_unchanged": valuation_2026_mismatch == 0,
        "provider_requests_2026": True,
    }
    if not all(checks.values()):
        raise ShareCapitalRepairError(f"prepared_contract_failed:{checks}")
    state.update(
        {
            "status": "prepared",
            "input_dataset_ids": input_ids,
            "repair_candidate_count": candidate_count,
            "unresolved_count": unresolved_count,
            "boundary_review_count": boundary_count,
            "formal_quality_missing_count": quality_missing,
            "existing_nonnull_total_share_mismatch_count": existing_share_mismatch,
            "formula_error_count": {
                "total_mv": int(formula_errors[0] or 0),
                "circ_mv": int(formula_errors[1] or 0),
            },
            "out_of_scope_2026": {
                "provider_request_count": 0,
                "repair_candidate_count": 0,
                "modified_share_row_count": share_2026_mismatch,
                "modified_valuation_row_count": valuation_2026_mismatch,
                "pass_through_share_row_count": int(share_stats[4] or 0),
                "pass_through_valuation_row_count": int(valuation_stats[3] or 0),
            },
            "prepared": {
                SHARE_DOMAIN: [str(path) for path in prepared_share_paths],
                VALUATION_DOMAIN: [str(path) for path in prepared_valuation_paths],
            },
            "inventories": {
                "repair_candidates": str(candidate_path),
                "unresolved": str(unresolved_path),
                "share_change_boundary_review": str(boundary_path),
            },
            "checks": checks,
        }
    )
    _write_state(workspace, state)
    return state


def _install(
    workspace: Path,
    *,
    domain: str,
    prepared_paths: Sequence[Path],
    input_manifest: DatasetManifest,
    state: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    root = qdp_v2_root(workspace)
    dataset_id = _content_dataset_id(domain, prepared_paths)
    dataset_root = root / "datasets" / domain / dataset_id
    entries: list[ShardManifestEntry] = []
    for prepared in prepared_paths:
        year = prepared.parent.name
        target = dataset_root / "shards" / year / "part-0000.parquet"
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp.parquet")
            shutil.copy2(prepared, temporary)
            os.replace(temporary, target)
        parquet = pq.ParquetFile(target)
        year_value = int(year.split("=", 1)[1])
        entries.append(
            ShardManifestEntry(
                path=str(target.relative_to(root)).replace("\\", "/"),
                row_count=int(parquet.metadata.num_rows),
                start_date=f"{year_value}-01-01",
                end_date=f"{year_value}-12-31",
                file_size=target.stat().st_size,
                metadata={
                    "prepared_sha256": _sha256(prepared),
                    "repair_id": REPAIR_ID,
                },
            )
        )
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        domain=domain,
        layer=input_manifest.layer,
        frequency=input_manifest.frequency,
        contract_version=(
            "qdp_v2_share_capital_strict_pit_v4"
            if domain == SHARE_DOMAIN
            else "qdp_v2_valuation_formula_v4"
        ),
        primary_key=list(input_manifest.primary_key),
        start_date=input_manifest.start_date,
        end_date=input_manifest.end_date,
        row_count=sum(item.row_count for item in entries),
        shards=entries,
        source={
            **dict(input_manifest.source or {}),
            "repair_id": REPAIR_ID,
            "repair_source": "same_day_stk_factor_pro_raw.total_share",
            "total_share_provider_unit": "ten_thousand_shares",
            "total_share_qdp_unit": "shares",
            "valuation_formula": "unadjusted_qdp_close_x_qdp_share_count",
            "checked_through": END_DATE,
            "provider_request_2026_count": 0,
            "out_of_scope_2026_policy": "byte-equivalent_value_pass_through",
        },
        quality={
            **dict(input_manifest.quality or {}),
            "primary_key_unique": True,
            "formal_quality_total_share_missing_rows": int(
                state.get("formal_quality_missing_count", 0) or 0
            ),
            "unresolved_total_share_rows_through_2025": int(
                state.get("unresolved_count", 0) or 0
            ),
            "existing_nonnull_total_share_mismatch_count": int(
                state.get("existing_nonnull_total_share_mismatch_count", 0) or 0
            ),
            "total_mv_formula_error_count": int(
                dict(state.get("formula_error_count", {}) or {}).get("total_mv", 0)
            ),
            "circ_mv_formula_error_count": int(
                dict(state.get("formula_error_count", {}) or {}).get("circ_mv", 0)
            ),
        },
        schema=_manifest_schema_from_arrow(pq.read_schema(prepared_paths[0])),
        notes=[
            "only null total_share values through 2025 are repaired",
            "same-day stk_factor_pro_raw total_share is converted from 万股 to shares",
            "existing non-null share values are preserved exactly",
            "2026 values are pass-through only and are neither repaired nor used by research",
        ],
    )
    _assert_credential_free(manifest.to_dict())
    write_dataset_manifest(root, manifest)
    return dataset_id, {
        "dataset_id": dataset_id,
        "row_count": manifest.row_count,
        "start_date": manifest.start_date,
        "end_date": manifest.end_date,
        "shard_count": len(entries),
    }


def commit(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    if state.get("status") == "applied":
        return state
    if state.get("status") != "prepared":
        raise ShareCapitalRepairError(f"repair_not_prepared:{state.get('status')}")
    installed: dict[str, Any] = {}
    ids: dict[str, str] = {}
    inputs = dict(state.get("input_dataset_ids", {}) or {})
    for domain in (SHARE_DOMAIN, VALUATION_DOMAIN):
        input_manifest, _ = _dataset(
            workspace,
            domain=domain,
            dataset_id=str(inputs[domain]),
        )
        paths = [Path(path) for path in state["prepared"][domain]]
        dataset_id, record = _install(
            workspace,
            domain=domain,
            prepared_paths=paths,
            input_manifest=input_manifest,
            state=state,
        )
        ids[domain] = dataset_id
        installed[domain] = record
    root = qdp_v2_root(workspace)
    active = read_active_manifest(root)
    current = active_dataset_map(active)
    for domain, dataset_id in inputs.items():
        if domain not in {SHARE_DOMAIN, VALUATION_DOMAIN} and current.get(domain) != dataset_id:
            raise ShareCapitalRepairError(f"unexpected_input_domain_drift:{domain}")
    active["datasets"] = {**current, **ids}
    active["updated_at"] = utc_now()
    write_active_manifest(root, active)
    state.update({"status": "applied", "installed_domains": installed})
    _write_state(workspace, state)
    audit = root / "audits" / f"{REPAIR_ID}.json"
    atomic_write_json(audit, state)
    return state


def evaluate(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    checks = dict(state.get("checks", {}) or {})
    active = active_dataset_map(read_active_manifest(qdp_v2_root(workspace)))
    installed = dict(state.get("installed_domains", {}) or {})
    checks.update(
        {
            "share_dataset_active": active.get(SHARE_DOMAIN)
            == dict(installed.get(SHARE_DOMAIN, {}) or {}).get("dataset_id"),
            "valuation_dataset_active": active.get(VALUATION_DOMAIN)
            == dict(installed.get(VALUATION_DOMAIN, {}) or {}).get("dataset_id"),
            "repair_performed_2026_rows_zero": int(
                dict(state.get("out_of_scope_2026", {}) or {}).get(
                    "repair_candidate_count", -1
                )
            )
            == 0,
            "provider_requests_2026_zero": int(
                dict(state.get("out_of_scope_2026", {}) or {}).get(
                    "provider_request_count", -1
                )
            )
            == 0,
        }
    )
    return {
        "status": "ok" if checks and all(checks.values()) else "error",
        "repair_id": REPAIR_ID,
        "checks": checks,
        "repair_candidate_count": state.get("repair_candidate_count", 0),
        "unresolved_count": state.get("unresolved_count", 0),
        "boundary_review_count": state.get("boundary_review_count", 0),
        "installed_domains": installed,
        "out_of_scope_2026": state.get("out_of_scope_2026", {}),
    }


def run_pending(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    prepare(workspace_root=workspace_root)
    return commit(workspace_root=workspace_root)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp share-capital-total-share-repair")
    parser.add_argument("--workspace-root", default="")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-pending", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--status", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = str(args.workspace_root or "") or None
    if args.run_pending:
        payload = run_pending(workspace_root=workspace)
    elif args.evaluate:
        payload = evaluate(workspace_root=workspace)
    else:
        payload = _read_state(_workspace(workspace))
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    return 0 if payload.get("status") not in {"error", "failed"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
