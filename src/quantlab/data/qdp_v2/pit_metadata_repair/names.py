"""Pit Metadata Repair: names responsibilities."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quantlab.core.io import json_safe
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb
from quantlab.data.qdp_v2.manifest import (
    atomic_write_json,
    read_active_manifest,
    utc_now,
)
from quantlab.data.qdp_v2.repair.common import _sql_literal
from quantlab.data.qdp_v2.repair.mutation import (
    mutate_active_shards_from_parquet,
    update_active_manifest_metadata,
)
from quantlab.data.qdp_v2.status import active_dataset_map

from .config import (
    REPAIR_ID,
    PitMetadataRepairError,
)
from .context import (
    _active_paths,
    _assert_untargeted_columns_equal,
    _root,
    _runtime,
    _schema_columns,
    _sha256,
    _sql_identifier,
)


@dataclass(frozen=True)
class _NameShardResult:
    replacement: tuple[Path, Path] | None
    replacement_hash: str | None
    changed_rows: int
    changed_symbols: set[str]
    changed_rows_by_year: dict[str, int]
    changed_date_min: str
    changed_date_max: str


@dataclass(frozen=True)
class _NameProfile:
    row_count: int
    changed_rows: int
    changed_symbols: set[str]
    changed_rows_by_year: dict[str, int]
    changed_date_min: str
    changed_date_max: str


def _validate_name_columns(path: Path) -> set[str]:
    columns = _schema_columns(path)
    missing = sorted({"symbol", "trade_date", "name"}.difference(columns))
    if missing:
        raise PitMetadataRepairError(f"universe_name_columns_missing:{path}:{','.join(missing)}")
    return columns


def _profile_name_shard(
    *,
    old_path: Path,
    event_arguments: list[str],
    cte: str,
    spill: Path,
) -> _NameProfile:
    try:
        with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
            profile = con.execute(
                cte
                + """
                SELECT count(*) AS row_count,
                       count(*) FILTER(WHERE current_name IS DISTINCT FROM expected_name),
                       count(DISTINCT symbol) FILTER(WHERE current_name IS DISTINCT FROM expected_name),
                       min(trade_date) FILTER(WHERE current_name IS DISTINCT FROM expected_name),
                       max(trade_date) FILTER(WHERE current_name IS DISTINCT FROM expected_name)
                FROM resolved
                """,
                [event_arguments, str(old_path)],
            ).fetchone()
            changed_rows = int(profile[1] or 0)
            if not changed_rows:
                return _NameProfile(int(profile[0] or 0), 0, set(), {}, "", "")
            year_rows = con.execute(
                cte
                + """
                SELECT substr(trade_date,1,4),count(*)
                FROM resolved
                WHERE current_name IS DISTINCT FROM expected_name
                GROUP BY 1 ORDER BY 1
                """,
                [event_arguments, str(old_path)],
            ).fetchall()
            symbols = {
                str(row[0])
                for row in con.execute(
                    cte
                    + """
                    SELECT DISTINCT symbol
                    FROM resolved
                    WHERE current_name IS DISTINCT FROM expected_name
                    """,
                    [event_arguments, str(old_path)],
                ).fetchall()
            }
            return _NameProfile(
                int(profile[0] or 0),
                changed_rows,
                symbols,
                {str(year): int(count) for year, count in year_rows},
                str(profile[3] or ""),
                str(profile[4] or ""),
            )
    finally:
        shutil.rmtree(spill, ignore_errors=True)


def _write_name_shard(
    *,
    old_path: Path,
    target: Path,
    columns: set[str],
    event_arguments: list[str],
    cte: str,
    spill: Path,
) -> tuple[int, int]:
    temporary = target.with_suffix(target.suffix + ".tmp")
    target.unlink(missing_ok=True)
    temporary.unlink(missing_ok=True)
    try:
        with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
            con.execute(
                f"COPY ({cte} SELECT {_name_projection(columns)} FROM resolved r "
                f"ORDER BY r.trade_date,r.symbol) TO {_sql_literal(str(temporary))} "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)",
                [event_arguments, str(old_path)],
            )
            temporary.replace(target)
            post = con.execute(
                cte
                + """
                SELECT count(*),count(*) FILTER(WHERE current_name IS DISTINCT FROM expected_name)
                FROM resolved
                """,
                [event_arguments, str(target)],
            ).fetchone()
            return int(post[0] or 0), int(post[1] or 0)
    finally:
        shutil.rmtree(spill, ignore_errors=True)


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
        raise PitMetadataRepairError(f"name_change_columns_missing:{','.join(missing)}")
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
            f"name_change_event_contract_failed:{json.dumps(contract, ensure_ascii=True, sort_keys=True)}"
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


def _name_projection(columns: set[str]) -> str:
    return ",".join(
        (
            'r.expected_name AS "name"'
            if column == "name"
            else f"r.{_sql_identifier(column)} AS {_sql_identifier(column)}"
        )
        for column in columns
    )


def _prepare_name_shard(
    *,
    old_path: Path,
    index: int,
    event_arguments: list[str],
    cte: str,
    runtime: Path,
    prepared: Path,
    apply: bool,
) -> _NameShardResult:
    columns = _validate_name_columns(old_path)
    spill = runtime / f"profile_spill_{index:04d}"
    profile = _profile_name_shard(
        old_path=old_path,
        event_arguments=event_arguments,
        cte=cte,
        spill=spill,
    )
    if not profile.changed_rows:
        return _NameShardResult(None, None, 0, set(), {}, "", "")
    if not apply:
        return _NameShardResult(
            (old_path, old_path),
            None,
            profile.changed_rows,
            profile.changed_symbols,
            profile.changed_rows_by_year,
            profile.changed_date_min,
            profile.changed_date_max,
        )
    target = prepared / f"universe_{index:04d}.parquet"
    post_rows, post_mismatches = _write_name_shard(
        old_path=old_path,
        target=target,
        columns=columns,
        event_arguments=event_arguments,
        cte=cte,
        spill=runtime / f"write_spill_{index:04d}",
    )
    if post_rows != profile.row_count or post_mismatches:
        raise PitMetadataRepairError(
            f"universe_name_postcheck_failed:{old_path}:rows={post_rows}:mismatches={post_mismatches}"
        )
    if _schema_columns(target) != columns:
        raise PitMetadataRepairError(f"universe_name_schema_changed:{old_path}")
    _assert_untargeted_columns_equal(old_path, target, columns, ("name",))
    return _NameShardResult(
        (old_path, target),
        _sha256(target),
        profile.changed_rows,
        profile.changed_symbols,
        profile.changed_rows_by_year,
        profile.changed_date_min,
        profile.changed_date_max,
    )


def _name_repair_payload(
    *,
    datasets: dict[str, str],
    universe_paths: tuple[Path, ...],
    event_contract: dict[str, Any],
    results: list[_NameShardResult],
    apply: bool,
) -> dict[str, Any]:
    changed_symbols = set().union(*(result.changed_symbols for result in results))
    changed_by_year: dict[str, int] = {}
    for result in results:
        for year, count in result.changed_rows_by_year.items():
            changed_by_year[year] = changed_by_year.get(year, 0) + count
    minimums = [result.changed_date_min for result in results if result.changed_date_min]
    maximums = [result.changed_date_max for result in results if result.changed_date_max]
    replacements = [result.replacement for result in results if result.replacement]
    hashes = {
        str(result.replacement[0]): result.replacement_hash
        for result in results
        if result.replacement and result.replacement_hash
    }
    return {
        "domain": "universe_snapshot",
        "status": "planned" if not apply else "prepared",
        "dataset_id": str(datasets["universe_snapshot"]),
        "name_change_dataset_id": str(datasets["name_change"]),
        "old_shard_count": len(universe_paths),
        "replacement_count": len(replacements),
        "changed_row_count": sum(result.changed_rows for result in results),
        "changed_symbol_count": len(changed_symbols),
        "changed_rows_by_year": dict(sorted(changed_by_year.items())),
        "changed_date_min": min(minimums) if minimums else "",
        "changed_date_max": max(maximums) if maximums else "",
        "event_contract": event_contract,
        "resolution_contract": {
            "on_or_after_event_date": "latest_new_name",
            "before_first_event_date": "first_old_name",
            "symbol_without_event": "preserve_universe_name",
            "only_changed_column": "name",
        },
        "replacement_hashes": hashes,
        "created_at": utc_now(),
        "_replacements": replacements,
    }


def _apply_name_repair(
    *,
    workspace: Path,
    root: Path,
    runtime: Path,
    datasets: dict[str, str],
    event_arguments: list[str],
    cte: str,
    payload: dict[str, Any],
) -> None:
    replacements = payload.pop("_replacements")
    if not replacements:
        payload["status"] = "already_consistent"
        payload["post_commit_mismatch_rows"] = 0
        return
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
            "name_pit_repaired_rows": int(payload["changed_row_count"]),
            "name_pit_repaired_symbols": int(payload["changed_symbol_count"]),
        },
    )
    current_paths = _active_paths(root, "universe_snapshot")
    with open_guarded_duckdb(temp_directory=runtime / "post_commit_spill", threads=2) as con:
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
        raise PitMetadataRepairError(f"universe_name_post_commit_mismatch:{post_mismatch}")
    payload.update(
        {
            "status": "applied",
            "post_commit_mismatch_rows": post_mismatch,
            "mutation": mutation,
            "metadata": metadata,
        }
    )


def _prepare_universe_name_pit(workspace: Path, *, apply: bool) -> dict[str, Any]:
    root = _root(workspace)
    universe_paths = _active_paths(root, "universe_snapshot")
    event_paths, event_contract = _name_event_contract(root)
    active = read_active_manifest(root)
    datasets = active_dataset_map(active)
    runtime = _runtime(workspace) / "universe_name_pit"
    prepared = runtime / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    event_arguments = [str(path) for path in event_paths]
    cte = _resolved_universe_name_cte()
    results = [
        _prepare_name_shard(
            old_path=old_path,
            index=index,
            event_arguments=event_arguments,
            cte=cte,
            runtime=runtime,
            prepared=prepared,
            apply=apply,
        )
        for index, old_path in enumerate(universe_paths)
    ]
    payload = _name_repair_payload(
        datasets=datasets,
        universe_paths=universe_paths,
        event_contract=event_contract,
        results=results,
        apply=apply,
    )
    if apply:
        _apply_name_repair(
            workspace=workspace,
            root=root,
            runtime=runtime,
            datasets=datasets,
            event_arguments=event_arguments,
            cte=cte,
            payload=payload,
        )
    else:
        payload.pop("_replacements")
    atomic_write_json(runtime / "plan.json", json_safe(payload))
    shutil.rmtree(prepared, ignore_errors=True)
    return json_safe(payload)
