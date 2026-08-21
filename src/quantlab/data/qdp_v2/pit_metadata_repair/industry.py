"""Pit Metadata Repair: industry responsibilities."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data.core.json_io import json_safe
from quantlab.data.qdp_v2.auxiliary_update import (
    AuxiliaryContext,
    _fetch_cninfo_industry_parts,
)
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb
from quantlab.data.qdp_v2.manifest import (
    atomic_write_json,
    dataset_manifest_for_id,
    read_active_manifest,
    read_dataset_manifest,
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
    _parquet_count,
    _root,
    _runtime,
    _schema_columns,
    _sha256,
    _sql_identifier,
)


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


def _industry_evidence(
    *,
    workspace: Path,
    root: Path,
    runtime: Path,
    symbols: list[str],
    active: dict[str, Any],
    datasets: dict[str, str],
) -> tuple[pd.DataFrame, Path]:
    manifest_path = dataset_manifest_for_id(
        root,
        datasets["industry_concept"],
        "industry_concept",
    )
    if manifest_path is None:
        raise PitMetadataRepairError("industry_manifest_missing")
    context = AuxiliaryContext(
        workspace=workspace,
        root=root,
        active=active,
        datasets=datasets,
        target_date=str(read_dataset_manifest(manifest_path).end_date),
        runtime=runtime,
    )
    parts = _fetch_cninfo_industry_parts(context, symbols)
    evidence = pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True) if parts else pd.DataFrame()
    path = runtime / "cninfo_evidence.parquet"
    evidence.to_parquet(path, index=False, compression="zstd")
    return evidence, path


def _industry_candidate_sql() -> tuple[str, str]:
    unknown = "(lower(trim(u.industry)) IN ('unknown','unclassified') OR u.industry_fill_method='unavailable')"
    evidence = (
        "SELECT symbol,industry,industry_standard,source,source_date "
        "FROM read_parquet(?) WHERE try_cast(source_date AS DATE) IS NOT NULL"
    )
    candidates = (
        "SELECT u.trade_date,u.symbol,e.industry,e.industry_standard,e.source,e.source_date "
        "FROM read_parquet(?) u JOIN (" + evidence + ") e "
        "ON e.symbol=u.symbol AND try_cast(e.source_date AS DATE)<=try_cast(u.trade_date AS DATE) "
        f"WHERE {unknown} "
        "QUALIFY row_number() OVER(PARTITION BY u.trade_date,u.symbol "
        "ORDER BY try_cast(e.source_date AS DATE) DESC)=1"
    )
    return unknown, candidates


def _industry_projection(columns: list[str]) -> str:
    replacements = {
        "industry": 'coalesce(c.industry,u."industry") AS "industry"',
        "industry_code": ('CASE WHEN c.symbol IS NULL THEN u."industry_code" ELSE NULL END AS "industry_code"'),
        "industry_name": 'coalesce(c.industry,u."industry_name") AS "industry_name"',
        "industry_taxonomy_version": (
            'CASE WHEN c.symbol IS NULL THEN u."industry_taxonomy_version" '
            "ELSE 'csrc_uncoded_historical_label' END AS \"industry_taxonomy_version\""
        ),
        "industry_section_code": (
            'CASE WHEN c.symbol IS NULL THEN u."industry_section_code" ELSE NULL END AS "industry_section_code"'
        ),
        "source": 'coalesce(c.source,u."source") AS "source"',
        "original_industry": ('coalesce(c.industry,u."original_industry") AS "original_industry"'),
        "original_source": 'coalesce(c.source,u."original_source") AS "original_source"',
        "industry_fill_method": (
            'CASE WHEN c.symbol IS NULL THEN u."industry_fill_method" '
            "WHEN c.source_date=u.trade_date THEN 'direct_snapshot' "
            "ELSE 'prior_ffill' END AS \"industry_fill_method\""
        ),
        "industry_source_date": ('coalesce(c.source_date,u."industry_source_date") AS "industry_source_date"'),
        "industry_standard": ('coalesce(c.industry_standard,u."industry_standard") AS "industry_standard"'),
    }
    return ",".join(
        replacements.get(
            column,
            f"u.{_sql_identifier(column)} AS {_sql_identifier(column)}",
        )
        for column in columns
    )


def _write_industry_replacement(
    *,
    runtime: Path,
    old_path: Path,
    evidence_path: Path,
    target: Path,
    columns: list[str],
    candidates: str,
) -> int:
    with open_guarded_duckdb(temp_directory=runtime / "spill", threads=2) as con:
        count = int(
            con.execute(
                f"SELECT count(*) FROM ({candidates}) c",
                [str(old_path), str(evidence_path)],
            ).fetchone()[0]
        )
        con.execute(
            f"COPY (SELECT {_industry_projection(columns)} FROM read_parquet(?) u "
            f"LEFT JOIN ({candidates}) c USING(symbol,trade_date) "
            f"ORDER BY u.trade_date,u.symbol) TO {_sql_literal(str(target))} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)",
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
    return count


def _write_industry_inventories(
    *,
    runtime: Path,
    old_path: Path,
    evidence_path: Path,
    unknown: str,
    candidates: str,
) -> tuple[Path, Path]:
    unresolved_path = runtime / "unresolved.parquet"
    resolved_path = runtime / "resolved.parquet"
    params = [str(old_path), str(old_path), str(evidence_path)]
    with open_guarded_duckdb(temp_directory=runtime / "unresolved_spill", threads=2) as con:
        con.execute(
            f"COPY (SELECT u.symbol,u.trade_date FROM read_parquet(?) u "
            f"LEFT JOIN ({candidates}) c USING(symbol,trade_date) "
            f"WHERE {unknown} AND c.symbol IS NULL ORDER BY u.symbol,u.trade_date) "
            f"TO {_sql_literal(str(unresolved_path))} (FORMAT PARQUET, COMPRESSION ZSTD)",
            params,
        )
        con.execute(
            f"COPY (SELECT u.symbol,u.trade_date,c.industry,c.industry_standard,c.source,c.source_date "
            f"FROM read_parquet(?) u JOIN ({candidates}) c USING(symbol,trade_date) "
            f"ORDER BY u.symbol,u.trade_date) TO {_sql_literal(str(resolved_path))} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)",
            params,
        )
    return resolved_path, unresolved_path


def _apply_industry_repair(
    *,
    workspace: Path,
    old_path: Path,
    target: Path,
    resolved_count: int,
    unresolved_count: int,
    payload: dict[str, Any],
) -> None:
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
        source_updates={
            "industry_unknown_repair_at": utc_now(),
            "industry_unknown_repair_id": REPAIR_ID,
        },
        quality_updates={
            "industry_unknown_rows": unresolved_count,
            "unknown_industry_rows": unresolved_count,
            "restored_unclassified_rows": resolved_count,
            "industry_repair_resolved_rows": resolved_count,
            "industry_repair_unresolved_rows": unresolved_count,
        },
    )


def _no_industry_evidence_payload(
    *,
    runtime: Path,
    evidence_path: Path,
    old_path: Path,
    symbol_count: int,
    apply: bool,
) -> dict[str, Any]:
    unresolved_count = len(pd.read_parquet(old_path).query("industry_fill_method == 'unavailable'"))
    payload = {
        "domain": "industry_concept",
        "status": "no_evidence" if not apply else "applied_no_change",
        "symbol_count": symbol_count,
        "resolved_rows": 0,
        "unresolved_rows": unresolved_count,
        "evidence_path": str(evidence_path),
        "created_at": utc_now(),
    }
    atomic_write_json(runtime / "plan.json", json_safe(payload))
    return json_safe(payload)


def _validate_industry_replacement(
    *,
    old_path: Path,
    target: Path,
    columns: list[str],
) -> None:
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


def _industry_inventory_counts(
    *,
    old_path: Path,
    resolved_path: Path,
    unresolved_path: Path,
    resolved_count: int,
) -> tuple[int, int]:
    unresolved_count = _parquet_count(unresolved_path)
    inventory_resolved_count = _parquet_count(resolved_path)
    original_unknown_count = _parquet_count(
        old_path,
        "lower(trim(industry)) IN ('unknown','unclassified') OR industry_fill_method='unavailable'",
    )
    if inventory_resolved_count != resolved_count or unresolved_count != original_unknown_count - resolved_count:
        raise PitMetadataRepairError(
            f"industry_repair_count_mismatch:{resolved_count}:{inventory_resolved_count}:{unresolved_count}"
        )
    return int(unresolved_count), int(inventory_resolved_count)


def _industry_payload(
    *,
    runtime: Path,
    evidence_path: Path,
    resolved_path: Path,
    unresolved_path: Path,
    symbol_count: int,
    resolved_count: int,
    unresolved_count: int,
    apply: bool,
) -> dict[str, Any]:
    payload = {
        "domain": "industry_concept",
        "status": "planned" if not apply else "prepared",
        "symbol_count": symbol_count,
        "resolved_rows": resolved_count,
        "unresolved_rows": unresolved_count,
        "resolved_path": str(resolved_path),
        "unresolved_path": str(unresolved_path),
        "evidence_path": str(evidence_path),
        "evidence_sha256": _sha256(evidence_path),
        "resolved_sha256": _sha256(resolved_path),
        "unresolved_sha256": _sha256(unresolved_path),
        "created_at": utc_now(),
    }
    atomic_write_json(runtime / "plan.json", json_safe(payload))
    return payload


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
    evidence, evidence_path = _industry_evidence(
        workspace=workspace,
        root=root,
        runtime=runtime,
        symbols=symbols,
        active=active,
        datasets=datasets,
    )
    old_path = paths[0]
    target = runtime / "industry_prepared.parquet"
    columns = _schema_columns(old_path)
    if evidence.empty:
        return _no_industry_evidence_payload(
            runtime=runtime,
            evidence_path=evidence_path,
            old_path=old_path,
            symbol_count=len(symbols),
            apply=apply,
        )
    unknown, candidates = _industry_candidate_sql()
    resolved_count = _write_industry_replacement(
        runtime=runtime,
        old_path=old_path,
        evidence_path=evidence_path,
        target=target,
        columns=columns,
        candidates=candidates,
    )
    _validate_industry_replacement(
        old_path=old_path,
        target=target,
        columns=columns,
    )
    resolved_path, unresolved_path = _write_industry_inventories(
        runtime=runtime,
        old_path=old_path,
        evidence_path=evidence_path,
        unknown=unknown,
        candidates=candidates,
    )
    unresolved_count, _ = _industry_inventory_counts(
        old_path=old_path,
        resolved_path=resolved_path,
        unresolved_path=unresolved_path,
        resolved_count=resolved_count,
    )
    payload = _industry_payload(
        runtime=runtime,
        evidence_path=evidence_path,
        resolved_path=resolved_path,
        unresolved_path=unresolved_path,
        symbol_count=len(symbols),
        resolved_count=resolved_count,
        unresolved_count=unresolved_count,
        apply=apply,
    )
    if apply:
        _apply_industry_repair(
            workspace=workspace,
            old_path=old_path,
            target=target,
            resolved_count=resolved_count,
            unresolved_count=unresolved_count,
            payload=payload,
        )
    if apply:
        shutil.rmtree(runtime / "spill", ignore_errors=True)
    return json_safe(payload)
