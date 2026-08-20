from __future__ import annotations

"""Cap inferred float shares at repaired total shares and refresh market value."""

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
from quantlab.data.qdp_v2.share_capital_total_share_repair import (
    DAILY_DOMAIN,
    END_DATE,
    SHARE_DOMAIN,
    VALUATION_DOMAIN,
    _copy_query,
    _dataset,
    _sql_paths,
)
from quantlab.data.qdp_v2.status import active_dataset_map

REPAIR_ID = "share_capital_float_share_invariant_repair_v1"
TARGET_FILL_METHOD = "same_day_stk_factor_pro_raw_total_share_wan_x10000"
EXPECTED_REPAIR_ROWS = 95_556


class FloatShareInvariantRepairError(RuntimeError):
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
        return {"repair_id": REPAIR_ID, "status": "pending"}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_state(workspace: Path, state: Mapping[str, Any]) -> None:
    payload = {**dict(state), "updated_at": utc_now()}
    _assert_credential_free(payload)
    atomic_write_json(_state_path(workspace), payload)


def _capped_float_share(total_share: Any, float_share: Any) -> float | None:
    total = pd.to_numeric(pd.Series([total_share]), errors="coerce").iloc[0]
    floating = pd.to_numeric(pd.Series([float_share]), errors="coerce").iloc[0]
    if pd.isna(floating):
        return None
    if pd.isna(total):
        return float(floating)
    return min(float(total), float(floating))


def _scan(paths: Sequence[Path]) -> str:
    return (
        f"read_parquet([{_sql_paths(paths)}], union_by_name=true, "
        "hive_partitioning=false)"
    )


def prepare(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    if state.get("status") in {"prepared", "applied"}:
        return state
    active = active_dataset_map(read_active_manifest(qdp_v2_root(workspace)))
    input_ids = {
        domain: active[domain]
        for domain in (SHARE_DOMAIN, VALUATION_DOMAIN, DAILY_DOMAIN)
    }
    share_manifest, share_paths = _dataset(workspace, domain=SHARE_DOMAIN)
    valuation_manifest, valuation_paths = _dataset(workspace, domain=VALUATION_DOMAIN)
    _, daily_paths = _dataset(workspace, domain=DAILY_DOMAIN)
    share_scan = _scan(share_paths)
    valuation_scan = _scan(valuation_paths)
    daily_scan = _scan(daily_paths)
    target = (
        f"trade_date<='{END_DATE}' AND share_fill_method='{TARGET_FILL_METHOD}' "
        "AND total_share IS NOT NULL AND float_share>total_share"
    )
    runtime = _runtime(workspace)
    connection = duckdb.connect()
    connection.execute("SET threads=4")
    spill = runtime / "duckdb_tmp"
    spill.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"SET temp_directory='{spill.resolve().as_posix().replace(chr(39), chr(39) * 2)}'"
    )
    try:
        target_path = runtime / "inventory" / "float_share_above_total_share.parquet"
        _copy_query(
            connection,
            query=f"SELECT * FROM {share_scan} WHERE {target} ORDER BY trade_date,symbol",
            path=target_path,
        )
        target_count = int(
            connection.execute(
                f"SELECT count(*) FROM {share_scan} WHERE {target}"
            ).fetchone()[0]
        )
        prepared_share_paths: list[Path] = []
        for year in range(2010, 2027):
            path = (
                runtime
                / "prepared"
                / SHARE_DOMAIN
                / f"year={year}"
                / "part-0000.parquet"
            )
            query = f"""
              SELECT symbol,trade_date,total_share,
                CASE WHEN {target} THEN total_share ELSE float_share END AS float_share,
                CASE WHEN {target} THEN 0.0 ELSE restricted_share END AS restricted_share,
                total_share_source_date,
                CASE WHEN {target} THEN coalesce(total_share_source_date,trade_date)
                     ELSE float_share_source_date END AS float_share_source_date,
                CASE WHEN {target} THEN coalesce(total_share_source_date,trade_date)
                     ELSE restricted_share_source_date END AS restricted_share_source_date,
                CASE WHEN {target} THEN concat(share_fill_method,'+float_capped_by_total_share')
                     ELSE share_fill_method END AS share_fill_method,
                CASE WHEN {target} THEN concat(source,'+float_share_invariant_repair')
                     ELSE source END AS source
              FROM {share_scan}
              WHERE trade_date BETWEEN '{year}-01-01' AND '{year}-12-31'
              ORDER BY trade_date,symbol
            """
            _copy_query(connection, query=query, path=path)
            prepared_share_paths.append(path)
        prepared_share_scan = _scan(prepared_share_paths)
        prepared_valuation_paths: list[Path] = []
        for year in range(2010, 2027):
            path = (
                runtime
                / "prepared"
                / VALUATION_DOMAIN
                / f"year={year}"
                / "part-0000.parquet"
            )
            query = f"""
              SELECT v.symbol,v.trade_date,
                CASE WHEN v.trade_date<='{END_DATE}' THEN d.close*s.total_share
                     ELSE v.total_mv END AS total_mv,
                CASE WHEN v.trade_date<='{END_DATE}' THEN d.close*s.float_share
                     ELSE v.circ_mv END AS circ_mv,
                v.pe,v.pb,v.turnover_rate,
                CASE WHEN old.float_share>old.total_share
                           AND old.trade_date<='{END_DATE}'
                           AND old.share_fill_method='{TARGET_FILL_METHOD}'
                     THEN concat(v.source,'+qdp_close_x_capped_float_share')
                     ELSE v.source END AS source
              FROM {valuation_scan} v
              JOIN {prepared_share_scan} s USING(symbol,trade_date)
              JOIN {share_scan} old USING(symbol,trade_date)
              JOIN {daily_scan} d USING(symbol,trade_date)
              WHERE v.trade_date BETWEEN '{year}-01-01' AND '{year}-12-31'
              ORDER BY v.trade_date,v.symbol
            """
            _copy_query(connection, query=query, path=path)
            prepared_valuation_paths.append(path)
        prepared_valuation_scan = _scan(prepared_valuation_paths)
        share_stats = connection.execute(
            f"SELECT count(*),min(trade_date),max(trade_date),"
            f"count(*) FILTER(WHERE float_share>total_share) FROM {prepared_share_scan}"
        ).fetchone()
        valuation_stats = connection.execute(
            f"SELECT count(*),min(trade_date),max(trade_date) FROM {prepared_valuation_scan}"
        ).fetchone()
        non_target_mismatch = int(
            connection.execute(
                f"""
                SELECT count(*) FROM {share_scan} old
                JOIN {prepared_share_scan} new USING(symbol,trade_date)
                WHERE NOT (old.trade_date<='{END_DATE}'
                  AND old.share_fill_method='{TARGET_FILL_METHOD}'
                  AND old.total_share IS NOT NULL
                  AND old.float_share>old.total_share)
                  AND (old.total_share IS DISTINCT FROM new.total_share
                    OR old.float_share IS DISTINCT FROM new.float_share
                    OR old.restricted_share IS DISTINCT FROM new.restricted_share
                    OR old.total_share_source_date IS DISTINCT FROM new.total_share_source_date
                    OR old.float_share_source_date IS DISTINCT FROM new.float_share_source_date
                    OR old.restricted_share_source_date IS DISTINCT FROM new.restricted_share_source_date
                    OR old.share_fill_method IS DISTINCT FROM new.share_fill_method
                    OR old.source IS DISTINCT FROM new.source)
                """
            ).fetchone()[0]
        )
        share_2026_mismatch = int(
            connection.execute(
                f"SELECT count(*) FROM ((SELECT * FROM {share_scan} WHERE trade_date>'{END_DATE}') "
                f"EXCEPT ALL (SELECT * FROM {prepared_share_scan} WHERE trade_date>'{END_DATE}'))"
            ).fetchone()[0]
        )
        valuation_2026_mismatch = int(
            connection.execute(
                f"SELECT count(*) FROM ((SELECT * FROM {valuation_scan} WHERE trade_date>'{END_DATE}') "
                f"EXCEPT ALL (SELECT * FROM {prepared_valuation_scan} WHERE trade_date>'{END_DATE}'))"
            ).fetchone()[0]
        )
        formula_errors = connection.execute(
            f"""
            SELECT count(*) FILTER(WHERE s.total_share IS NOT NULL AND
                     abs(v.total_mv-d.close*s.total_share)>
                     greatest(abs(d.close*s.total_share)*1e-8,1e-6)),
                   count(*) FILTER(WHERE s.float_share IS NOT NULL AND
                     abs(v.circ_mv-d.close*s.float_share)>
                     greatest(abs(d.close*s.float_share)*1e-8,1e-6))
            FROM {prepared_valuation_scan} v
            JOIN {prepared_share_scan} s USING(symbol,trade_date)
            JOIN {daily_scan} d USING(symbol,trade_date)
            """
        ).fetchone()
    finally:
        connection.close()
        shutil.rmtree(spill, ignore_errors=True)
    checks = {
        "target_count_expected": target_count == EXPECTED_REPAIR_ROWS,
        "share_row_count_unchanged": int(share_stats[0]) == share_manifest.row_count,
        "share_date_range_unchanged": str(share_stats[1]) == share_manifest.start_date
        and str(share_stats[2]) == share_manifest.end_date,
        "float_share_never_exceeds_total_share": int(share_stats[3]) == 0,
        "valuation_row_count_unchanged": int(valuation_stats[0])
        == valuation_manifest.row_count,
        "valuation_date_range_unchanged": str(valuation_stats[1])
        == valuation_manifest.start_date
        and str(valuation_stats[2]) == valuation_manifest.end_date,
        "non_target_share_rows_unchanged": non_target_mismatch == 0,
        "total_mv_formula_valid": int(formula_errors[0]) == 0,
        "circ_mv_formula_valid": int(formula_errors[1]) == 0,
        "out_of_scope_2026_share_values_unchanged": share_2026_mismatch == 0,
        "out_of_scope_2026_valuation_values_unchanged": valuation_2026_mismatch == 0,
        "provider_request_2026_count_zero": True,
    }
    if not all(checks.values()):
        raise FloatShareInvariantRepairError(f"float_share_prepare_failed:{checks}")
    state.update(
        {
            "status": "prepared",
            "input_dataset_ids": input_ids,
            "repair_row_count": target_count,
            "inventory_path": str(target_path),
            "inventory_sha256": _sha256(target_path),
            "prepared": {
                SHARE_DOMAIN: [str(path) for path in prepared_share_paths],
                VALUATION_DOMAIN: [str(path) for path in prepared_valuation_paths],
            },
            "formula_error_count": {
                "total_mv": int(formula_errors[0]),
                "circ_mv": int(formula_errors[1]),
            },
            "out_of_scope_2026": {
                "provider_request_count": 0,
                "modified_share_row_count": share_2026_mismatch,
                "modified_valuation_row_count": valuation_2026_mismatch,
            },
            "checks": checks,
        }
    )
    _write_state(workspace, state)
    return state


def _content_id(domain: str, paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(_sha256(path).encode("ascii"))
    return f"{domain}__{digest.hexdigest()[:24]}"


def _install(
    workspace: Path,
    *,
    domain: str,
    paths: Sequence[Path],
    input_manifest: DatasetManifest,
    state: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    root = qdp_v2_root(workspace)
    dataset_id = _content_id(domain, paths)
    target_root = root / "datasets" / domain / dataset_id
    entries: list[ShardManifestEntry] = []
    for source_path in paths:
        year = source_path.parent.name
        target = target_root / "shards" / year / "part-0000.parquet"
        source_hash = _sha256(source_path)
        if target.is_file():
            if _sha256(target) != source_hash:
                raise FloatShareInvariantRepairError(
                    f"installed_hash_conflict:{target}"
                )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".partial")
            shutil.copy2(source_path, temporary)
            if _sha256(temporary) != source_hash:
                temporary.unlink(missing_ok=True)
                raise FloatShareInvariantRepairError(f"install_copy_failed:{target}")
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
                    "source_sha256": source_hash,
                    "repair_id": REPAIR_ID,
                },
            )
        )
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        domain=domain,
        layer=input_manifest.layer,
        frequency=input_manifest.frequency,
        contract_version=input_manifest.contract_version,
        primary_key=list(input_manifest.primary_key),
        start_date=input_manifest.start_date,
        end_date=input_manifest.end_date,
        row_count=sum(item.row_count for item in entries),
        shards=entries,
        source={
            **dict(input_manifest.source or {}),
            "checked_through": input_manifest.end_date,
            "repair_id": REPAIR_ID,
            "repair_checked_through": END_DATE,
            "float_share_invariant": "min(inferred_float_share,total_share)",
            "provider_request_2026_count": 0,
            "out_of_scope_2026_policy": "byte-equivalent_value_pass_through",
        },
        quality={
            **dict(input_manifest.quality or {}),
            "primary_key_unique": True,
            "float_share_above_total_share_rows": 0,
            "float_share_capped_by_total_share_rows": int(
                state.get("repair_row_count", 0)
            ),
            "total_mv_formula_error_count": int(
                dict(state.get("formula_error_count", {}) or {}).get("total_mv", 0)
            ),
            "circ_mv_formula_error_count": int(
                dict(state.get("formula_error_count", {}) or {}).get("circ_mv", 0)
            ),
        },
        schema=_manifest_schema_from_arrow(pq.read_schema(paths[0])),
        notes=[
            *list(input_manifest.notes or []),
            "Only rows filled by the same-day total-share repair are eligible for the float-share cap.",
            "Turnover-inferred float shares above total shares are capped at total shares; total shares are unchanged.",
            "Non-target rows and all 2026 values are preserved.",
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
        raise FloatShareInvariantRepairError(
            f"float_share_not_prepared:{state.get('status')}"
        )
    inputs = dict(state.get("input_dataset_ids", {}) or {})
    installed: dict[str, Any] = {}
    ids: dict[str, str] = {}
    for domain in (SHARE_DOMAIN, VALUATION_DOMAIN):
        input_manifest, _ = _dataset(
            workspace, domain=domain, dataset_id=str(inputs[domain])
        )
        dataset_id, record = _install(
            workspace,
            domain=domain,
            paths=[Path(path) for path in state["prepared"][domain]],
            input_manifest=input_manifest,
            state=state,
        )
        ids[domain] = dataset_id
        installed[domain] = record
    root = qdp_v2_root(workspace)
    active = read_active_manifest(root)
    current = active_dataset_map(active)
    for domain in (SHARE_DOMAIN, VALUATION_DOMAIN, DAILY_DOMAIN):
        if current.get(domain) != inputs.get(domain):
            raise FloatShareInvariantRepairError(f"active_input_drifted:{domain}")
    active["datasets"] = {**current, **ids}
    active["updated_at"] = utc_now()
    write_active_manifest(root, active)
    state.update({"status": "applied", "installed_domains": installed})
    _write_state(workspace, state)
    atomic_write_json(root / "audits" / f"{REPAIR_ID}.json", state)
    return state


def evaluate(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    active = active_dataset_map(read_active_manifest(qdp_v2_root(workspace)))
    installed = dict(state.get("installed_domains", {}) or {})
    checks = {
        **dict(state.get("checks", {}) or {}),
        "share_dataset_active": active.get(SHARE_DOMAIN)
        == dict(installed.get(SHARE_DOMAIN, {}) or {}).get("dataset_id"),
        "valuation_dataset_active": active.get(VALUATION_DOMAIN)
        == dict(installed.get(VALUATION_DOMAIN, {}) or {}).get("dataset_id"),
    }
    return {
        "status": "ok" if checks and all(checks.values()) else "error",
        "repair_id": REPAIR_ID,
        "checks": checks,
        "repair_row_count": state.get("repair_row_count", 0),
        "installed_domains": installed,
        "out_of_scope_2026": state.get("out_of_scope_2026", {}),
    }


def run_pending(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    prepare(workspace_root=workspace_root)
    return commit(workspace_root=workspace_root)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp float-share-invariant-repair")
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
