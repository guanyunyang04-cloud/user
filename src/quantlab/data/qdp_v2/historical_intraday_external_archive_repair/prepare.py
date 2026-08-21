"""Historical Intraday External Archive Repair: prepare responsibilities."""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from quantlab.data.qdp_v2.manifest import (
    EXPECTED_BAR_TIMES,
    DatasetManifest,
    qdp_v2_root,
    resolve_manifest_path,
)
from quantlab.data.qdp_v2.recent_market_repair import (
    INTRADAY_COLUMNS,
    INTRADAY_DOMAIN,
)

from .config import (
    _DECISION_COLUMNS,
    END_DATE,
    EXPECTED_ACCEPTED_DAYS,
    EXPECTED_CANDIDATE_DAYS,
    EXPECTED_CANDIDATE_SYMBOLS,
    EXPECTED_FORMAL_QUALITY_ACCEPTED_DAYS,
    EXPECTED_FORMAL_QUALITY_POOL_ROWS,
    EXPECTED_FORMAL_QUALITY_REMAINING_DAYS,
    EXPECTED_RECENT_QUALITY_REMAINING_DAYS,
    REPAIR_ID,
    SOURCE_NAME,
    ArchiveMember,
    ExternalArchiveRepairError,
    SymbolResult,
)
from .context import (
    _baseline_manifest,
    _read_state,
    _runtime,
    _write_state,
)
from .inventory import (
    _active_intraday,
    _gap_inventory,
    _scan_archive,
)
from .process import (
    _atomic_frame_parquet,
)


def _initialize(
    workspace: Path,
    archive_path: Path,
) -> tuple[pd.DataFrame, dict[str, ArchiveMember], DatasetManifest, dict[str, Any]]:
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    inventory, inventory_path, inventory_hash = _gap_inventory(workspace)
    active_manifest, _ = _active_intraday(workspace)
    baseline = _baseline_manifest(workspace)
    baseline_intraday = str(dict(baseline.get("active_dataset_ids", {})).get(INTRADAY_DOMAIN, ""))
    if active_manifest.dataset_id != baseline_intraday:
        state = _read_state(workspace)
        installed = str(state.get("installed_dataset_id", ""))
        if not installed or active_manifest.dataset_id != installed:
            raise ExternalArchiveRepairError("active_intraday_drifted_since_repair_baseline")
    members = _scan_archive(archive_path)
    matched = sorted(set(inventory["symbol"]).intersection(members))
    candidate_count = int(inventory["symbol"].isin(matched).sum())
    if len(matched) != EXPECTED_CANDIDATE_SYMBOLS or candidate_count != EXPECTED_CANDIDATE_DAYS:
        raise ExternalArchiveRepairError(f"archive_candidate_contract:{len(matched)}:{candidate_count}")
    state = _read_state(workspace)
    state.update(
        {
            "repair_id": REPAIR_ID,
            "status": "initialized",
            "archive_path": str(archive_path),
            "archive_size": archive_path.stat().st_size,
            "archive_mtime_ns": archive_path.stat().st_mtime_ns,
            "archive_member_count": len(members),
            "input_intraday_dataset_id": baseline_intraday,
            "inventory_path": str(inventory_path),
            "inventory_sha256": inventory_hash,
            "inventory_day_count": len(inventory),
            "candidate_symbol_count": len(matched),
            "candidate_day_count": candidate_count,
            "no_file_symbol_count": int(inventory.loc[~inventory["symbol"].isin(matched), "symbol"].nunique()),
            "no_file_day_count": int((~inventory["symbol"].isin(matched)).sum()),
            "request_2026_count": 0,
            "write_2026_count": 0,
        }
    )
    _write_state(workspace, state)
    return inventory, members, active_manifest, state


def _no_file_decisions(inventory: pd.DataFrame, matched: set[str]) -> pd.DataFrame:
    frame = inventory.loc[
        ~inventory["symbol"].isin(matched),
        ["symbol", "trade_date", "quality_liquidity_keep"],
    ].copy()
    frame["archive_member"] = ""
    frame["decision"] = "no_file"
    frame["rejection_reason"] = "symbol_file_absent"
    frame["source_bar_count"] = 0
    frame["normalized_bar_count"] = 0
    frame["price_relative_error"] = np.nan
    frame["volume_relative_error"] = np.nan
    frame["amount_relative_error"] = np.nan
    frame["accepted_row_count"] = 0
    frame["source"] = SOURCE_NAME
    return frame.loc[:, _DECISION_COLUMNS]


def _assemble_decisions(
    workspace: Path,
    inventory: pd.DataFrame,
    results: Sequence[SymbolResult],
) -> Path:
    matched = {item.symbol for item in results}
    frames = [pd.read_parquet(item.decision_path) for item in results]
    frames.append(_no_file_decisions(inventory, matched))
    decisions = pd.concat(frames, ignore_index=True)
    decisions = decisions.sort_values(["trade_date", "symbol"], kind="stable")
    if len(decisions) != len(inventory):
        raise ExternalArchiveRepairError(f"decision_inventory_count:{len(decisions)}!={len(inventory)}")
    if decisions.duplicated(["symbol", "trade_date"]).any():
        raise ExternalArchiveRepairError("decision_key_duplicate")
    if set(decisions["decision"].unique()) != {"accepted", "rejected", "no_file"}:
        raise ExternalArchiveRepairError("decision_state_contract")
    path = _runtime(workspace) / "inventory" / "repair_decisions.parquet"
    _atomic_frame_parquet(decisions, path)
    return path


def _sql_paths(paths: Sequence[Path]) -> str:
    return ",".join(f"'{path.resolve().as_posix().replace(chr(39), chr(39) * 2)}'" for path in paths)


def _bundle_accepted(workspace: Path, results: Sequence[SymbolResult]) -> list[Path]:
    accepted_paths = [Path(item.accepted_path) for item in results if item.accepted_path]
    if not accepted_paths:
        raise ExternalArchiveRepairError("archive_repair_no_accepted_rows")
    scan = f"read_parquet([{_sql_paths(accepted_paths)}], union_by_name=false, hive_partitioning=false)"
    output: list[Path] = []
    connection = duckdb.connect()
    connection.execute("SET threads=4")
    spill = _runtime(workspace) / "bundle_spill"
    spill.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET temp_directory='{spill.resolve().as_posix().replace(chr(39), chr(39) * 2)}'")
    try:
        years = [
            int(row[0])
            for row in connection.execute(f"SELECT DISTINCT substr(trade_date,1,4) FROM {scan} ORDER BY 1").fetchall()
        ]
        for year in years:
            path = _runtime(workspace) / "prepared" / INTRADAY_DOMAIN / f"year={year}" / "part-0000.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".partial")
            temporary.unlink(missing_ok=True)
            quoted = temporary.resolve().as_posix().replace("'", "''")
            connection.execute(
                f"""
                COPY (
                  SELECT {",".join(INTRADAY_COLUMNS)} FROM {scan}
                  WHERE trade_date BETWEEN '{year}-01-01' AND '{year}-12-31'
                  ORDER BY trade_date,symbol,bar_time
                ) TO '{quoted}' (
                  FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000
                )
                """
            )
            os.replace(temporary, path)
            output.append(path)
    finally:
        connection.close()
        shutil.rmtree(spill, ignore_errors=True)
    return output


def _prepared_archive_stats(
    *,
    decisions_scan: str,
    bundles_scan: str,
    input_scan: str,
) -> tuple[tuple[Any, ...], tuple[Any, ...], int, int]:
    connection = duckdb.connect()
    connection.execute("SET threads=4")
    try:
        decision_stats = connection.execute(
            f"""
            SELECT count(*) AS total,
              count(*) FILTER(WHERE decision='accepted') AS accepted,
              count(*) FILTER(WHERE decision='rejected') AS rejected,
              count(*) FILTER(WHERE decision='no_file') AS no_file,
              count(*) FILTER(WHERE decision='accepted'
                AND quality_liquidity_keep AND trade_date>='2011-01-01') AS formal_quality_accepted,
              count(*) FILTER(WHERE decision!='accepted'
                AND quality_liquidity_keep AND trade_date>='2011-01-01') AS formal_quality_remaining,
              count(*) FILTER(WHERE decision!='accepted'
                AND quality_liquidity_keep AND trade_date BETWEEN '2023-01-01' AND '2025-12-31')
                AS recent_quality_remaining,
              count(*) FILTER(WHERE trade_date>'{END_DATE}') AS forbidden_2026
            FROM {decisions_scan}
            """
        ).fetchone()
        bundle_stats = connection.execute(
            f"""
            SELECT count(*) AS rows,
              count(*)-count(DISTINCT symbol||'|'||trade_date||'|'||bar_time) AS duplicate_rows,
              count(DISTINCT symbol||'|'||trade_date) AS days,
              count(*) FILTER(WHERE trade_date>'{END_DATE}') AS forbidden_2026,
              count(*) FILTER(WHERE bar_time NOT IN ({','.join(repr(x) for x in EXPECTED_BAR_TIMES)}))
                AS invalid_bar_time
            FROM {bundles_scan}
            """
        ).fetchone()
        invalid_day_count = int(
            connection.execute(
                f"""
                SELECT count(*) FROM (
                  SELECT symbol,trade_date,count(*) AS bars,count(DISTINCT bar_time) AS clocks
                  FROM {bundles_scan} GROUP BY symbol,trade_date
                  HAVING bars<>48 OR clocks<>48
                )
                """
            ).fetchone()[0]
        )
        overlap_count = int(
            connection.execute(
                f"""
                SELECT count(*) FROM {bundles_scan} n JOIN (
                  SELECT * FROM {input_scan} WHERE trade_date<='{END_DATE}'
                ) o USING(symbol,trade_date,bar_time)
                """
            ).fetchone()[0]
        )
    finally:
        connection.close()
    return decision_stats, bundle_stats, invalid_day_count, overlap_count


def _prepared_archive_checks(
    workspace: Path,
    *,
    inventory: pd.DataFrame,
    decision_stats: tuple[Any, ...],
    bundle_stats: tuple[Any, ...],
    invalid_day_count: int,
    overlap_count: int,
) -> dict[str, bool]:
    formal_quality_total = int(
        inventory.loc[inventory["quality_liquidity_keep"] & inventory["trade_date"].ge("2011-01-01")].shape[0]
    )
    return {
        "decision_inventory_exhaustive": int(decision_stats[0]) == len(inventory),
        "decision_states_mutually_exclusive": sum(map(int, decision_stats[1:4])) == len(inventory),
        "accepted_day_count_matches_audit": int(decision_stats[1]) == EXPECTED_ACCEPTED_DAYS,
        "formal_quality_accepted_matches_audit": int(decision_stats[4]) == EXPECTED_FORMAL_QUALITY_ACCEPTED_DAYS,
        "formal_quality_remaining_matches_audit": int(decision_stats[5]) == EXPECTED_FORMAL_QUALITY_REMAINING_DAYS,
        "recent_quality_remaining_matches_audit": int(decision_stats[6]) == EXPECTED_RECENT_QUALITY_REMAINING_DAYS,
        "decision_2026_rows_zero": int(decision_stats[7]) == 0,
        "accepted_rows_equal_48_per_day": int(bundle_stats[0]) == int(decision_stats[1]) * 48,
        "accepted_primary_key_unique": int(bundle_stats[1]) == 0,
        "accepted_distinct_days_match": int(bundle_stats[2]) == int(decision_stats[1]),
        "accepted_2026_rows_zero": int(bundle_stats[3]) == 0,
        "accepted_bar_times_valid": int(bundle_stats[4]) == 0,
        "accepted_day_contract_valid": invalid_day_count == 0,
        "existing_intraday_keys_not_overwritten": overlap_count == 0,
        "baseline_formal_quality_gap_count": formal_quality_total == 35_602,
        "formal_complete_pool_expected_rows": EXPECTED_FORMAL_QUALITY_POOL_ROWS - int(decision_stats[5]) == 4_476_845,
        "request_2026_count_zero": int(_read_state(workspace).get("request_2026_count", -1)) == 0,
    }


def _validate_prepared(
    workspace: Path,
    *,
    inventory: pd.DataFrame,
    decisions_path: Path,
    bundle_paths: Sequence[Path],
    input_manifest: DatasetManifest,
) -> dict[str, Any]:
    root = qdp_v2_root(workspace)
    input_paths = [resolve_manifest_path(item.path, root=root) for item in input_manifest.shards]
    decisions_scan = f"read_parquet('{decisions_path.resolve().as_posix()}', hive_partitioning=false)"
    bundles_scan = f"read_parquet([{_sql_paths(bundle_paths)}], union_by_name=false, hive_partitioning=false)"
    input_scan = f"read_parquet([{_sql_paths(input_paths)}], union_by_name=true, hive_partitioning=false)"
    decision_stats, bundle_stats, invalid_day_count, overlap_count = _prepared_archive_stats(
        decisions_scan=decisions_scan,
        bundles_scan=bundles_scan,
        input_scan=input_scan,
    )
    checks = _prepared_archive_checks(
        workspace,
        inventory=inventory,
        decision_stats=decision_stats,
        bundle_stats=bundle_stats,
        invalid_day_count=invalid_day_count,
        overlap_count=overlap_count,
    )
    if not all(checks.values()):
        raise ExternalArchiveRepairError(f"external_archive_prepared_contract:{checks}")
    reason_counts = {
        str(reason): int(count)
        for reason, count in pd.read_parquet(decisions_path, columns=["decision", "rejection_reason"])
        .groupby(["decision", "rejection_reason"], dropna=False)
        .size()
        .items()
    }
    return {
        "checks": checks,
        "decision_counts": {
            "accepted": int(decision_stats[1]),
            "rejected": int(decision_stats[2]),
            "no_file": int(decision_stats[3]),
        },
        "accepted_row_count": int(bundle_stats[0]),
        "formal_quality_accepted_day_count": int(decision_stats[4]),
        "formal_quality_remaining_day_count": int(decision_stats[5]),
        "recent_quality_remaining_day_count": int(decision_stats[6]),
        "formal_quality_complete_pool_row_count": EXPECTED_FORMAL_QUALITY_POOL_ROWS - int(decision_stats[5]),
        "rejection_reason_counts": reason_counts,
    }
