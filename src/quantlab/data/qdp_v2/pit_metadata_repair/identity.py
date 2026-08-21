"""Pit Metadata Repair: identity responsibilities."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from quantlab.data.core.json_io import json_safe
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb
from quantlab.data.qdp_v2.manifest import (
    atomic_write_json,
    dataset_manifest_for_id,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    utc_now,
)
from quantlab.data.qdp_v2.repair import (
    _sql_literal,
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
                    'THEN coalesce(m.list_date, u."list_date") ELSE u."list_date" END AS "list_date"'
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
            raise PitMetadataRepairError(f"universe_list_date_future_values:{old_path}:{future_count}")
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
