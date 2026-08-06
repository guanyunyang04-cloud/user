from __future__ import annotations

"""Small, auditable repairs for PIT metadata that already has a safe source.

This module intentionally does not rebuild a QDP domain.  It replaces only
active shards that contain a targeted correction and records the before/after
inventory beside the normal QDP runtime state.
"""

import argparse
import hashlib
import json
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.qdp_v2.auxiliary_update import (
    _fetch_cninfo_industry_parts,
)
from quant_data_platform.qdp_v2.duckdb_resources import open_guarded_duckdb
from quant_data_platform.qdp_v2.manifest import (
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    utc_now,
)
from quant_data_platform.qdp_v2.repair import (
    _sql_literal,
    mutate_active_shards_from_parquet,
    update_active_manifest_metadata,
)
from quant_data_platform.qdp_v2.status import active_dataset_map

REPAIR_ID = "pit_metadata_repair_v1"
SUPPORTED_DOMAINS = (
    "universe-list-date",
    "universe-name-pit",
    "industry-unknown",
)


class PitMetadataRepairError(RuntimeError):
    pass


def _workspace(value: str | Path | None) -> Path:
    return Path(value or Path.cwd()).resolve()


def _root(workspace: Path) -> Path:
    return qdp_v2_root(workspace).resolve()


def _active_paths(root: Path, domain: str) -> tuple[Path, ...]:
    active = read_active_manifest(root)
    datasets = active_dataset_map(active)
    manifest_path = dataset_manifest_for_id(root, datasets[domain], domain)
    if manifest_path is None:
        raise PitMetadataRepairError(f"active_manifest_missing:{domain}")
    manifest = read_dataset_manifest(manifest_path)
    paths = tuple(resolve_manifest_path(item.path, root=root) for item in manifest.shards)
    if not paths or any(not path.is_file() for path in paths):
        raise PitMetadataRepairError(f"active_shards_missing:{domain}")
    return paths


def _runtime(workspace: Path) -> Path:
    path = (qdp_paths(workspace).data_dir / "qdp_runtime" / REPAIR_ID).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _schema_columns(path: Path) -> list[str]:
    return list(pq.read_schema(path).names)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parquet_count(path: Path, predicate: str = "TRUE") -> int:
    with open_guarded_duckdb(
        temp_directory=path.parent / "_count_spill", threads=2
    ) as con:
        return int(
            con.execute(
                f"SELECT count(*) FROM read_parquet(?) WHERE {predicate}",
                [str(path)],
            ).fetchone()[0]
        )


def _assert_untargeted_columns_equal(
    old_path: Path,
    new_path: Path,
    columns: Sequence[str],
    changed_columns: Sequence[str],
) -> None:
    stable = [column for column in columns if column not in set(changed_columns)]
    if not stable:
        return
    projection = ",".join(_sql_identifier(column) for column in stable)
    with open_guarded_duckdb(
        temp_directory=old_path.parent / "_repair_compare_spill", threads=2
    ) as con:
        left_only = int(
            con.execute(
                f"SELECT count(*) FROM (SELECT {projection} FROM read_parquet(?) "
                f"EXCEPT ALL SELECT {projection} FROM read_parquet(?))",
                [str(old_path), str(new_path)],
            ).fetchone()[0]
        )
        right_only = int(
            con.execute(
                f"SELECT count(*) FROM (SELECT {projection} FROM read_parquet(?) "
                f"EXCEPT ALL SELECT {projection} FROM read_parquet(?))",
                [str(new_path), str(old_path)],
            ).fetchone()[0]
        )
    if left_only or right_only:
        raise PitMetadataRepairError(
            f"repair_untargeted_columns_changed:{old_path.name}:{left_only}:{right_only}"
        )


def _sql_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _build_identity_map(root: Path) -> tuple[Path, Path]:
    active = read_active_manifest(root)
    datasets = active_dataset_map(active)
    history_manifest = dataset_manifest_for_id(root, datasets["symbol_history"], "symbol_history")
    identity_manifest = dataset_manifest_for_id(root, datasets["security_identity"], "security_identity")
    if history_manifest is None or identity_manifest is None:
        raise PitMetadataRepairError("identity_map_inputs_missing")
    history = read_dataset_manifest(history_manifest)
    identity = read_dataset_manifest(identity_manifest)
    history_path = root / "tmp" / f"{REPAIR_ID}_symbol_history.parquet"
    identity_path = root / "tmp" / f"{REPAIR_ID}_security_identity.parquet"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    # A list of active shards is accepted by DuckDB, but a stable copied map
    # makes the repair manifest independent of future active-pointer changes.
    for target, manifest in ((history_path, history), (identity_path, identity)):
        if target.is_file():
            continue
        paths = [resolve_manifest_path(item.path, root=root) for item in manifest.shards]
        with open_guarded_duckdb(temp_directory=root / "tmp" / f"{REPAIR_ID}_map_spill", threads=2) as con:
            con.execute(
                f"COPY (SELECT * FROM read_parquet(?) ORDER BY ALL) TO {_sql_literal(str(target))} (FORMAT PARQUET, COMPRESSION ZSTD)",
                [[str(path) for path in paths]],
            )
    return history_path, identity_path


def _prepare_list_date(workspace: Path, *, apply: bool) -> dict[str, Any]:
    root = _root(workspace)
    paths = _active_paths(root, "universe_snapshot")
    history_path, identity_path = _build_identity_map(root)
    runtime = _runtime(workspace) / "universe_list_date"
    prepared = runtime / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    replacements: list[tuple[Path, Path]] = []
    filled_rows = 0
    for index, old_path in enumerate(paths):
        columns = _schema_columns(old_path)
        if "list_date" not in columns:
            raise PitMetadataRepairError("universe_snapshot_list_date_column_missing")
        expressions: list[str] = []
        for column in columns:
            qcol = _sql_identifier(column)
            if column == "list_date":
                expressions.append(
                    "CASE WHEN nullif(trim(cast(u.\"list_date\" AS VARCHAR)), '') IS NULL "
                    "THEN coalesce(m.list_date, u.\"list_date\") ELSE u.\"list_date\" END AS \"list_date\""
                )
            else:
                expressions.append(f"u.{qcol} AS {qcol}")
        target = prepared / f"universe_{index:04d}.parquet"
        with open_guarded_duckdb(temp_directory=runtime / "spill", threads=2) as con:
            count = int(
                con.execute(
                    "SELECT count(*) FROM read_parquet(?) u "
                    "LEFT JOIN (SELECT sh.symbol, min(nullif(trim(cast(si.list_date AS VARCHAR)), '')) AS list_date "
                    "FROM read_parquet(?) sh JOIN read_parquet(?) si USING(security_id) "
                    "GROUP BY sh.symbol) m USING(symbol) "
                    "WHERE nullif(trim(cast(u.list_date AS VARCHAR)), '') IS NULL AND m.list_date IS NOT NULL",
                    [str(old_path), str(history_path), str(identity_path)],
                ).fetchone()[0]
            )
            if not count:
                continue
            con.execute(
                f"COPY (SELECT {','.join(expressions)} FROM read_parquet(?) u "
                f"LEFT JOIN (SELECT sh.symbol, min(nullif(trim(cast(si.list_date AS VARCHAR)), '')) AS list_date "
                f"FROM read_parquet(?) sh JOIN read_parquet(?) si USING(security_id) GROUP BY sh.symbol) m USING(symbol) "
                f"ORDER BY u.trade_date,u.symbol) TO {_sql_literal(str(target))} (FORMAT PARQUET, COMPRESSION ZSTD)",
                [str(old_path), str(history_path), str(identity_path)],
            )
            future_count = int(
                con.execute(
                    "SELECT count(*) FROM read_parquet(?) "
                    "WHERE try_cast(list_date AS DATE)>try_cast(trade_date AS DATE)",
                    [str(target)],
                ).fetchone()[0]
            )
        if _schema_columns(target) != columns:
            raise PitMetadataRepairError(f"universe_list_date_schema_changed:{old_path}")
        _assert_untargeted_columns_equal(old_path, target, columns, ("list_date",))
        if future_count:
            raise PitMetadataRepairError(
                f"universe_list_date_future_values:{old_path}:{future_count}"
            )
        replacements.append((old_path, target))
        filled_rows += count
    payload: dict[str, Any] = {
        "domain": "universe_snapshot",
        "status": "planned" if not apply else "prepared",
        "replacement_count": len(replacements),
        "filled_rows": int(filled_rows),
        "old_shard_count": len(paths),
        "preconditions": {
            "only_blank_list_date": True,
            "identity_source": "symbol_history.security_id -> security_identity.list_date",
            "future_dates_forbidden": True,
        },
        "replacements": [(str(old), str(new)) for old, new in replacements],
        "replacement_hashes": {str(old): _sha256(new) for old, new in replacements},
        "created_at": utc_now(),
    }
    if apply and replacements:
        mutation = mutate_active_shards_from_parquet(
            "universe_snapshot",
            replacements=replacements,
            reason="fill blank PIT universe listing dates from security identity",
            workspace_root=workspace,
        )
        metadata = update_active_manifest_metadata(
            "universe_snapshot",
            reason="record PIT listing-date repair",
            workspace_root=workspace,
            source_updates={"list_date_repaired_at": utc_now(), "list_date_repair_id": REPAIR_ID},
            quality_updates={"list_date_blank_rows": 0, "list_date_repair_filled_rows": int(filled_rows)},
        )
        payload.update({"status": "applied", "mutation": mutation, "metadata": metadata})
    atomic_write_json(runtime / "plan.json", json_safe(payload))
    shutil.rmtree(runtime / "prepared", ignore_errors=not apply)
    return json_safe(payload)


def _name_event_contract(root: Path) -> tuple[tuple[Path, ...], dict[str, Any]]:
    paths = _active_paths(root, "name_change")
    required = {
        "symbol",
        "trade_date",
        "old_name",
        "new_name",
        "change_type",
    }
    available = set().union(*(_schema_columns(path) for path in paths))
    missing = sorted(required.difference(available))
    if missing:
        raise PitMetadataRepairError(
            f"name_change_columns_missing:{','.join(missing)}"
        )
    spill = root / "tmp" / f"{REPAIR_ID}_name_event_contract_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        row = con.execute(
            """
            WITH events AS (
              SELECT symbol,trade_date,trim(old_name) AS old_name,
                     trim(new_name) AS new_name,change_type
              FROM read_parquet(?, union_by_name=true)
            ), ordered AS (
              SELECT *,lag(new_name) OVER(
                PARTITION BY symbol ORDER BY trade_date
              ) AS previous_new_name
              FROM events
            ), duplicate_dates AS (
              SELECT symbol,trade_date
              FROM events
              GROUP BY symbol,trade_date
              HAVING count(*)>1
            )
            SELECT count(*) AS event_rows,
                   count(DISTINCT symbol) AS event_symbols,
                   min(trade_date) AS min_date,
                   max(trade_date) AS max_date,
                   count(*) FILTER(
                     WHERE coalesce(symbol,'')='' OR
                           try_cast(trade_date AS DATE) IS NULL OR
                           coalesce(old_name,'')='' OR
                           coalesce(new_name,'')='' OR old_name=new_name
                   ) AS invalid_rows,
                   (SELECT count(*) FROM duplicate_dates) AS duplicate_dates,
                   count(*) FILTER(
                     WHERE previous_new_name IS NOT NULL AND
                           previous_new_name<>old_name
                   ) AS broken_chain_rows
            FROM ordered
            """,
            [[str(path) for path in paths]],
        ).fetchone()
    shutil.rmtree(spill, ignore_errors=True)
    contract = {
        "event_row_count": int(row[0] or 0),
        "event_symbol_count": int(row[1] or 0),
        "minimum_event_date": str(row[2] or ""),
        "maximum_event_date": str(row[3] or ""),
        "invalid_event_row_count": int(row[4] or 0),
        "duplicate_symbol_date_count": int(row[5] or 0),
        "broken_event_chain_count": int(row[6] or 0),
    }
    if (
        not contract["event_row_count"]
        or contract["invalid_event_row_count"]
        or contract["duplicate_symbol_date_count"]
        or contract["broken_event_chain_count"]
    ):
        raise PitMetadataRepairError(
            "name_change_event_contract_failed:"
            f"{json.dumps(contract, ensure_ascii=True, sort_keys=True)}"
        )
    return paths, contract


def _resolved_universe_name_cte() -> str:
    return """
    WITH name_events AS (
      SELECT symbol,trade_date,trim(old_name) AS old_name,
             trim(new_name) AS new_name
      FROM read_parquet(?, union_by_name=true)
    ), first_events AS (
      SELECT symbol,arg_min(old_name,trade_date) AS first_old_name
      FROM name_events
      GROUP BY symbol
    ), resolved AS (
      SELECT u.*,
             trim(cast(u.name AS VARCHAR)) AS current_name,
             CASE WHEN e.symbol IS NOT NULL THEN e.new_name
                  WHEN f.symbol IS NOT NULL THEN f.first_old_name
                  ELSE u.name END AS expected_name
      FROM read_parquet(?, union_by_name=true) u
      ASOF LEFT JOIN name_events e
        ON u.symbol=e.symbol AND u.trade_date>=e.trade_date
      LEFT JOIN first_events f ON u.symbol=f.symbol
    )
    """


def _prepare_universe_name_pit(workspace: Path, *, apply: bool) -> dict[str, Any]:
    root = _root(workspace)
    universe_paths = _active_paths(root, "universe_snapshot")
    event_paths, event_contract = _name_event_contract(root)
    active = read_active_manifest(root)
    datasets = active_dataset_map(active)
    runtime = _runtime(workspace) / "universe_name_pit"
    prepared = runtime / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    replacements: list[tuple[Path, Path]] = []
    replacement_hashes: dict[str, str] = {}
    changed_rows = 0
    changed_symbols: set[str] = set()
    changed_rows_by_year: dict[str, int] = {}
    changed_date_min = ""
    changed_date_max = ""
    event_arguments = [str(path) for path in event_paths]
    cte = _resolved_universe_name_cte()

    for index, old_path in enumerate(universe_paths):
        columns = _schema_columns(old_path)
        required = {"symbol", "trade_date", "name"}
        missing = sorted(required.difference(columns))
        if missing:
            raise PitMetadataRepairError(
                f"universe_name_columns_missing:{old_path}:{','.join(missing)}"
            )
        spill = runtime / f"profile_spill_{index:04d}"
        with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
            profile = con.execute(
                cte
                + """
                SELECT count(*) AS row_count,
                       count(*) FILTER(
                         WHERE current_name IS DISTINCT FROM expected_name
                       ) AS changed_rows,
                       count(DISTINCT symbol) FILTER(
                         WHERE current_name IS DISTINCT FROM expected_name
                       ) AS changed_symbols,
                       min(trade_date) FILTER(
                         WHERE current_name IS DISTINCT FROM expected_name
                       ) AS changed_date_min,
                       max(trade_date) FILTER(
                         WHERE current_name IS DISTINCT FROM expected_name
                       ) AS changed_date_max
                FROM resolved
                """,
                [event_arguments, str(old_path)],
            ).fetchone()
            shard_changed = int(profile[1] or 0)
            if not shard_changed:
                shutil.rmtree(spill, ignore_errors=True)
                continue
            year_rows = con.execute(
                cte
                + """
                SELECT substr(trade_date,1,4) AS year,count(*) AS changed_rows
                FROM resolved
                WHERE current_name IS DISTINCT FROM expected_name
                GROUP BY year ORDER BY year
                """,
                [event_arguments, str(old_path)],
            ).fetchall()
            symbols = con.execute(
                cte
                + """
                SELECT DISTINCT symbol
                FROM resolved
                WHERE current_name IS DISTINCT FROM expected_name
                """,
                [event_arguments, str(old_path)],
            ).fetchall()
            changed_rows += shard_changed
            changed_symbols.update(str(row[0]) for row in symbols)
            for year, count in year_rows:
                changed_rows_by_year[str(year)] = (
                    changed_rows_by_year.get(str(year), 0) + int(count)
                )
            shard_min = str(profile[3] or "")
            shard_max = str(profile[4] or "")
            if shard_min and (not changed_date_min or shard_min < changed_date_min):
                changed_date_min = shard_min
            if shard_max and (not changed_date_max or shard_max > changed_date_max):
                changed_date_max = shard_max

            if not apply:
                replacements.append((old_path, old_path))
                shutil.rmtree(spill, ignore_errors=True)
                continue

            projection = ",".join(
                (
                    'r.expected_name AS "name"'
                    if column == "name"
                    else f"r.{_sql_identifier(column)} AS {_sql_identifier(column)}"
                )
                for column in columns
            )
            target = prepared / f"universe_{index:04d}.parquet"
            temporary = target.with_suffix(target.suffix + ".tmp")
            target.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)
            con.execute(
                f"COPY ({cte} SELECT {projection} FROM resolved r "
                f"ORDER BY r.trade_date,r.symbol) TO {_sql_literal(str(temporary))} "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)",
                [event_arguments, str(old_path)],
            )
            temporary.replace(target)
            post = con.execute(
                cte
                + """
                SELECT count(*),count(*) FILTER(
                  WHERE current_name IS DISTINCT FROM expected_name
                )
                FROM resolved
                """,
                [event_arguments, str(target)],
            ).fetchone()
        shutil.rmtree(spill, ignore_errors=True)
        if int(post[0] or 0) != int(profile[0] or 0) or int(post[1] or 0):
            raise PitMetadataRepairError(
                f"universe_name_postcheck_failed:{old_path}:"
                f"rows={post[0]}:mismatches={post[1]}"
            )
        if _schema_columns(target) != columns:
            raise PitMetadataRepairError(
                f"universe_name_schema_changed:{old_path}"
            )
        _assert_untargeted_columns_equal(old_path, target, columns, ("name",))
        replacements.append((old_path, target))
        replacement_hashes[str(old_path)] = _sha256(target)

    payload: dict[str, Any] = {
        "domain": "universe_snapshot",
        "status": "planned" if not apply else "prepared",
        "dataset_id": str(datasets["universe_snapshot"]),
        "name_change_dataset_id": str(datasets["name_change"]),
        "old_shard_count": len(universe_paths),
        "replacement_count": len(replacements),
        "changed_row_count": int(changed_rows),
        "changed_symbol_count": len(changed_symbols),
        "changed_rows_by_year": dict(sorted(changed_rows_by_year.items())),
        "changed_date_min": changed_date_min,
        "changed_date_max": changed_date_max,
        "event_contract": event_contract,
        "resolution_contract": {
            "on_or_after_event_date": "latest_new_name",
            "before_first_event_date": "first_old_name",
            "symbol_without_event": "preserve_universe_name",
            "only_changed_column": "name",
        },
        "replacement_hashes": replacement_hashes,
        "created_at": utc_now(),
    }
    if apply and replacements:
        mutation = mutate_active_shards_from_parquet(
            "universe_snapshot",
            replacements=replacements,
            reason="restore PIT universe names from dated name-change events",
            workspace_root=workspace,
        )
        metadata = update_active_manifest_metadata(
            "universe_snapshot",
            reason="record PIT universe-name repair",
            workspace_root=workspace,
            source_updates={
                "name_pit_repaired_at": utc_now(),
                "name_pit_repair_id": REPAIR_ID,
                "name_pit_source_dataset_id": str(datasets["name_change"]),
                "name_pit_source_contract": (
                    "latest dated new_name; first old_name before first event; "
                    "preserve original name without event evidence"
                ),
            },
            quality_updates={
                "name_pit_mismatch_rows": 0,
                "name_pit_repaired_rows": int(changed_rows),
                "name_pit_repaired_symbols": len(changed_symbols),
            },
        )
        current_paths = _active_paths(root, "universe_snapshot")
        with open_guarded_duckdb(
            temp_directory=runtime / "post_commit_spill", threads=2
        ) as con:
            post_mismatch = int(
                con.execute(
                    cte
                    + """
                    SELECT count(*) FROM resolved
                    WHERE current_name IS DISTINCT FROM expected_name
                    """,
                    [event_arguments, [str(path) for path in current_paths]],
                ).fetchone()[0]
            )
        shutil.rmtree(runtime / "post_commit_spill", ignore_errors=True)
        if post_mismatch:
            raise PitMetadataRepairError(
                f"universe_name_post_commit_mismatch:{post_mismatch}"
            )
        payload.update(
            {
                "status": "applied",
                "post_commit_mismatch_rows": post_mismatch,
                "mutation": mutation,
                "metadata": metadata,
            }
        )
    elif apply:
        payload["status"] = "already_consistent"
        payload["post_commit_mismatch_rows"] = 0
    atomic_write_json(runtime / "plan.json", json_safe(payload))
    shutil.rmtree(prepared, ignore_errors=True)
    return json_safe(payload)


def _unknown_symbols(root: Path) -> list[str]:
    paths = _active_paths(root, "industry_concept")
    with open_guarded_duckdb(temp_directory=root / "tmp" / f"{REPAIR_ID}_unknown_spill", threads=2) as con:
        return [
            str(row[0])
            for row in con.execute(
                "SELECT DISTINCT symbol FROM read_parquet(?, union_by_name=true) "
                "WHERE lower(trim(industry)) IN ('unknown','unclassified') OR industry_fill_method='unavailable' ORDER BY symbol",
                [[str(path) for path in paths]],
            ).fetchall()
        ]


def _prepare_industry(workspace: Path, *, apply: bool) -> dict[str, Any]:
    root = _root(workspace)
    paths = _active_paths(root, "industry_concept")
    if len(paths) != 1:
        raise PitMetadataRepairError("industry_repair_expected_single_active_shard")
    runtime = _runtime(workspace) / "industry_unknown"
    runtime.mkdir(parents=True, exist_ok=True)
    symbols = _unknown_symbols(root)
    active = read_active_manifest(root)
    datasets = active_dataset_map(active)
    industry_manifest_path = dataset_manifest_for_id(root, datasets["industry_concept"], "industry_concept")
    if industry_manifest_path is None:
        raise PitMetadataRepairError("industry_manifest_missing")
    from quant_data_platform.qdp_v2.auxiliary_update import AuxiliaryContext

    ctx = AuxiliaryContext(
        workspace=workspace,
        root=root,
        active=active,
        datasets=datasets,
        target_date=str(read_dataset_manifest(industry_manifest_path).end_date),
        runtime=runtime,
    )
    evidence_parts = _fetch_cninfo_industry_parts(ctx, symbols)
    evidence = pd.concat(
        [pd.read_parquet(path) for path in evidence_parts],
        ignore_index=True,
    ) if evidence_parts else pd.DataFrame()
    evidence_path = runtime / "cninfo_evidence.parquet"
    evidence.to_parquet(evidence_path, index=False, compression="zstd")
    old_path = paths[0]
    target = runtime / "industry_prepared.parquet"
    columns = _schema_columns(old_path)
    if evidence.empty:
        resolved_count = 0
        unresolved_count = len(pd.read_parquet(old_path).query("industry_fill_method == 'unavailable'"))
        payload = {
            "domain": "industry_concept",
            "status": "no_evidence" if not apply else "applied_no_change",
            "symbol_count": len(symbols),
            "resolved_rows": resolved_count,
            "unresolved_rows": unresolved_count,
            "evidence_path": str(evidence_path),
            "created_at": utc_now(),
        }
        atomic_write_json(runtime / "plan.json", json_safe(payload))
        return json_safe(payload)
    with open_guarded_duckdb(temp_directory=runtime / "spill", threads=2) as con:
        unknown = "(lower(trim(u.industry)) IN ('unknown','unclassified') OR u.industry_fill_method='unavailable')"
        evidence_cte = (
            "SELECT symbol,industry,industry_standard,source,source_date "
            "FROM read_parquet(?) WHERE try_cast(source_date AS DATE) IS NOT NULL"
        )
        candidates = (
            "SELECT u.trade_date,u.symbol,e.industry,e.industry_standard,e.source,e.source_date "
            "FROM read_parquet(?) u JOIN (" + evidence_cte + ") e "
            "ON e.symbol=u.symbol AND try_cast(e.source_date AS DATE)<=try_cast(u.trade_date AS DATE) "
            f"WHERE {unknown} "
            "QUALIFY row_number() OVER(PARTITION BY u.trade_date,u.symbol ORDER BY try_cast(e.source_date AS DATE) DESC)=1"
        )
        counts = con.execute(
            f"SELECT count(*) FROM ({candidates}) c", [str(old_path), str(evidence_path)]
        ).fetchone()[0]
        all_expr: list[str] = []
        for column in columns:
            qcol = _sql_identifier(column)
            if column == "industry":
                all_expr.append("coalesce(c.industry,u.\"industry\") AS \"industry\"")
            elif column == "industry_code":
                all_expr.append("CASE WHEN c.symbol IS NULL THEN u.\"industry_code\" ELSE NULL END AS \"industry_code\"")
            elif column == "industry_name":
                all_expr.append("coalesce(c.industry,u.\"industry_name\") AS \"industry_name\"")
            elif column == "industry_taxonomy_version":
                all_expr.append("CASE WHEN c.symbol IS NULL THEN u.\"industry_taxonomy_version\" ELSE 'csrc_uncoded_historical_label' END AS \"industry_taxonomy_version\"")
            elif column == "industry_section_code":
                all_expr.append("CASE WHEN c.symbol IS NULL THEN u.\"industry_section_code\" ELSE NULL END AS \"industry_section_code\"")
            elif column == "source":
                all_expr.append("coalesce(c.source,u.\"source\") AS \"source\"")
            elif column == "original_industry":
                all_expr.append("coalesce(c.industry,u.\"original_industry\") AS \"original_industry\"")
            elif column == "original_source":
                all_expr.append("coalesce(c.source,u.\"original_source\") AS \"original_source\"")
            elif column == "industry_fill_method":
                all_expr.append("CASE WHEN c.symbol IS NULL THEN u.\"industry_fill_method\" WHEN c.source_date=u.trade_date THEN 'direct_snapshot' ELSE 'prior_ffill' END AS \"industry_fill_method\"")
            elif column == "industry_source_date":
                all_expr.append("coalesce(c.source_date,u.\"industry_source_date\") AS \"industry_source_date\"")
            elif column == "industry_standard":
                all_expr.append("coalesce(c.industry_standard,u.\"industry_standard\") AS \"industry_standard\"")
            else:
                all_expr.append(f"u.{qcol} AS {qcol}")
        con.execute(
            f"COPY (SELECT {','.join(all_expr)} FROM read_parquet(?) u LEFT JOIN ({candidates}) c USING(symbol,trade_date) "
            f"ORDER BY u.trade_date,u.symbol) TO {_sql_literal(str(target))} (FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(old_path), str(old_path), str(evidence_path)],
        )
        future_count = int(
            con.execute(
                "SELECT count(*) FROM read_parquet(?) "
                "WHERE try_cast(industry_source_date AS DATE)>try_cast(trade_date AS DATE)",
                [str(target)],
            ).fetchone()[0]
        )
    if future_count:
        raise PitMetadataRepairError(f"industry_future_source_dates:{future_count}")
    if _schema_columns(target) != columns:
        raise PitMetadataRepairError("industry_repair_schema_changed")
    _assert_untargeted_columns_equal(
        old_path,
        target,
        columns,
        (
            "industry",
            "industry_code",
            "industry_name",
            "industry_taxonomy_version",
            "industry_section_code",
            "source",
            "original_industry",
            "original_source",
            "industry_fill_method",
            "industry_source_date",
            "industry_standard",
        ),
    )
    unresolved_path = runtime / "unresolved.parquet"
    resolved_path = runtime / "resolved.parquet"
    with open_guarded_duckdb(temp_directory=runtime / "unresolved_spill", threads=2) as con:
        con.execute(
            f"COPY (SELECT u.symbol,u.trade_date FROM read_parquet(?) u LEFT JOIN ({candidates}) c USING(symbol,trade_date) "
            f"WHERE {unknown} AND c.symbol IS NULL ORDER BY u.symbol,u.trade_date) TO {_sql_literal(str(unresolved_path))} (FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(old_path), str(old_path), str(evidence_path)],
        )
        con.execute(
            f"COPY (SELECT u.symbol,u.trade_date,c.industry,c.industry_standard,c.source,c.source_date "
            f"FROM read_parquet(?) u JOIN ({candidates}) c USING(symbol,trade_date) "
            f"ORDER BY u.symbol,u.trade_date) TO {_sql_literal(str(resolved_path))} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(old_path), str(old_path), str(evidence_path)],
        )
    unresolved_count = _parquet_count(unresolved_path)
    resolved_count = _parquet_count(resolved_path)
    original_unknown_count = _parquet_count(
        old_path,
        "lower(trim(industry)) IN ('unknown','unclassified') OR industry_fill_method='unavailable'",
    )
    if resolved_count != int(counts) or unresolved_count != original_unknown_count - int(counts):
        raise PitMetadataRepairError(
            f"industry_repair_count_mismatch:{counts}:{resolved_count}:{unresolved_count}"
        )
    payload = {
        "domain": "industry_concept",
        "status": "planned" if not apply else "prepared",
        "symbol_count": len(symbols),
        "resolved_rows": int(counts),
        "unresolved_rows": int(unresolved_count),
        "resolved_path": str(resolved_path),
        "unresolved_path": str(unresolved_path),
        "evidence_path": str(evidence_path),
        "evidence_sha256": _sha256(evidence_path),
        "resolved_sha256": _sha256(resolved_path),
        "unresolved_sha256": _sha256(unresolved_path),
        "created_at": utc_now(),
    }
    if apply:
        mutation = mutate_active_shards_from_parquet(
            "industry_concept",
            replacements=[(old_path, target)],
            reason="fill dated PIT industry Unknown rows from CNINFO evidence",
            workspace_root=workspace,
        )
        payload.update({"status": "applied", "mutation": mutation})
        update_active_manifest_metadata(
            "industry_concept",
            reason="record dated CNINFO industry Unknown repair",
            workspace_root=workspace,
            source_updates={"industry_unknown_repair_at": utc_now(), "industry_unknown_repair_id": REPAIR_ID},
            quality_updates={
                "industry_unknown_rows": int(unresolved_count),
                "unknown_industry_rows": int(unresolved_count),
                "restored_unclassified_rows": int(counts),
                "industry_repair_resolved_rows": int(counts),
                "industry_repair_unresolved_rows": int(unresolved_count),
            },
        )
    atomic_write_json(runtime / "plan.json", json_safe(payload))
    if apply:
        shutil.rmtree(runtime / "spill", ignore_errors=True)
    return json_safe(payload)


def run_repair(
    *,
    workspace_root: str | Path | None = None,
    domains: Sequence[str] = SUPPORTED_DOMAINS,
    apply: bool = False,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    selected = tuple(str(item) for item in domains)
    unknown = sorted(set(selected).difference(SUPPORTED_DOMAINS))
    if unknown:
        raise ValueError(f"unknown_pit_metadata_domains:{','.join(unknown)}")
    results: dict[str, Any] = {}
    if "universe-list-date" in selected:
        results["universe-list-date"] = _prepare_list_date(workspace, apply=apply)
    if "universe-name-pit" in selected:
        results["universe-name-pit"] = _prepare_universe_name_pit(
            workspace, apply=apply
        )
    if "industry-unknown" in selected:
        results["industry-unknown"] = _prepare_industry(workspace, apply=apply)
    return json_safe({"status": "applied" if apply else "planned", "domains": results})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qdp pit-metadata-repair")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--domains", default=",".join(SUPPORTED_DOMAINS))
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.plan) == bool(args.apply):
        parser.error("exactly one of --plan or --apply is required")
    domains = tuple(item.strip() for item in str(args.domains).split(",") if item.strip())
    payload = run_repair(
        workspace_root=str(args.workspace_root or "") or None,
        domains=domains,
        apply=bool(args.apply),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["SUPPORTED_DOMAINS", "PitMetadataRepairError", "run_repair"]
