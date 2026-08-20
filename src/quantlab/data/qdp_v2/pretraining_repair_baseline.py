from __future__ import annotations

"""Freeze the QDP state immediately before the training-data repair programme."""

import argparse
import hashlib
import json
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from quantlab.data.core.json_io import json_safe
from quantlab.data.core.paths import qdp_paths
from quantlab.data.qdp_v2.manifest import (
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    utc_now,
)

BASELINE_ID = "pretraining_data_repair_baseline_v1"
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"


class PretrainingRepairBaselineError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_hash(payload: Any) -> str:
    encoded = json.dumps(
        json_safe(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _scan(paths: Iterable[Path]) -> str:
    values = ",".join(f"'{_sql_path(path)}'" for path in paths)
    if not values:
        raise PretrainingRepairBaselineError("empty_dataset_path_set")
    return f"read_parquet([{values}], union_by_name=true, hive_partitioning=false)"


def _copy(con: duckdb.DuckDBPyConnection, query: str, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    con.execute(
        f"COPY ({query}) TO '{_sql_path(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    temporary.replace(path)
    return int(con.execute(f"SELECT count(*) FROM ({query})").fetchone()[0])


def _latest_audit(root: Path, prefix: str) -> Path | None:
    paths = sorted((root / "audits").glob(f"{prefix}*.json"))
    return paths[-1] if paths else None


def _quality_diagnostics(workspace: Path) -> list[Path]:
    state_path = (
        workspace
        / "research"
        / "legacy"
        / "data_prep"
        / "seq100_quality_liquidity_data_prep"
        / "state.json"
    )
    if not state_path.is_file():
        return []
    state = json.loads(state_path.read_text(encoding="utf-8"))
    paths = []
    for year, record in dict(state.get("membership_years", {}) or {}).items():
        if 2010 <= int(year) <= 2025:
            path = Path(dict(record or {}).get("diagnostics_path", ""))
            if path.is_file():
                paths.append(path)
    return paths


def _dataset_inventory(
    *, root: Path, active: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[Path]]]:
    dataset_rows: list[dict[str, Any]] = []
    shard_rows: list[dict[str, Any]] = []
    paths_by_domain: dict[str, list[Path]] = {}
    for domain, dataset_id in sorted(dict(active.get("datasets", {}) or {}).items()):
        manifest_path = dataset_manifest_for_id(root, str(dataset_id), str(domain))
        if manifest_path is None:
            raise PretrainingRepairBaselineError(
                f"active_dataset_manifest_missing:{domain}:{dataset_id}"
            )
        manifest = read_dataset_manifest(manifest_path)
        schema = [dict(item) for item in manifest.schema]
        dataset_rows.append(
            {
                "domain": str(domain),
                "dataset_id": str(dataset_id),
                "contract_version": manifest.contract_version,
                "start_date": manifest.start_date,
                "end_date": manifest.end_date,
                "row_count": int(manifest.row_count),
                "shard_count": len(manifest.shards),
                "schema_hash": _payload_hash(schema),
                "dataset_manifest_path": str(manifest_path.resolve()),
                "dataset_manifest_sha256": _sha256(manifest_path),
            }
        )
        domain_paths: list[Path] = []
        for shard in manifest.shards:
            path = resolve_manifest_path(shard.path, root=root)
            if not path.is_file():
                raise PretrainingRepairBaselineError(
                    f"active_shard_missing:{domain}:{path}"
                )
            domain_paths.append(path)
            shard_rows.append(
                {
                    "domain": str(domain),
                    "dataset_id": str(dataset_id),
                    "path": str(path),
                    "relative_manifest_path": shard.path,
                    "row_count": int(shard.row_count),
                    "file_size": int(path.stat().st_size),
                    "sha256": _sha256(path),
                    "start_date": shard.start_date,
                    "end_date": shard.end_date,
                }
            )
        paths_by_domain[str(domain)] = domain_paths
    return pd.DataFrame(dataset_rows), pd.DataFrame(shard_rows), paths_by_domain


def freeze_baseline(
    *, workspace_root: str | Path | None = None, force: bool = False
) -> dict[str, Any]:
    workspace = Path(workspace_root or Path.cwd()).resolve()
    root = qdp_v2_root(workspace)
    output = qdp_paths(workspace).data_dir / "qdp_runtime" / BASELINE_ID
    manifest_path = output / "manifest.json"
    active_path = root / "active" / "active.json"
    active = read_active_manifest(root)
    if not active or len(dict(active.get("datasets", {}) or {})) != 26:
        raise PretrainingRepairBaselineError("expected_26_active_domains")
    active_hash = _sha256(active_path)
    if manifest_path.is_file() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("active_manifest_sha256") != active_hash:
            raise PretrainingRepairBaselineError(
                "baseline_already_frozen_for_different_active_manifest"
            )
        return existing

    output.mkdir(parents=True, exist_ok=True)
    dataset_frame, shard_frame, paths = _dataset_inventory(root=root, active=active)
    dataset_path = output / "active_dataset_inventory.parquet"
    shard_path = output / "active_shard_inventory.parquet"
    dataset_frame.to_parquet(dataset_path, index=False, compression="zstd")
    shard_frame.to_parquet(shard_path, index=False, compression="zstd")
    shutil.copy2(active_path, output / "active_manifest.json")

    temp = output / "duckdb_tmp"
    temp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute(f"SET temp_directory='{_sql_path(temp)}'")
    try:
        daily = _scan(paths["market_daily_raw"])
        intraday = _scan(paths["market_intraday_5m"])
        balance = _scan(paths["balance_sheet_quarterly"])
        share = _scan(paths["share_capital"])
        universe = _scan(paths["universe_snapshot"])
        margin = _scan(paths["margin_detail"])

        v1_report_state_path = (
            qdp_paths(workspace).data_dir
            / "qdp_runtime"
            / "research_report_rc_backfill_v1"
            / "state.json"
        )
        v1_report_state = (
            json.loads(v1_report_state_path.read_text(encoding="utf-8"))
            if v1_report_state_path.is_file()
            else {}
        )
        v1_years = dict(
            dict(v1_report_state.get("tushare_report_rc", {}) or {}).get("years", {})
            or {}
        )
        report_gaps = pd.DataFrame(
            {
                "report_date": pd.date_range(START_DATE, END_DATE, freq="D").strftime(
                    "%Y-%m-%d"
                )
            }
        )
        report_gaps["year"] = report_gaps["report_date"].str[:4].astype(int)
        report_gaps["status"] = "pending"
        report_gaps["reason"] = "v1_has_no_daily_task_ledger"
        report_gaps["v1_year_status"] = report_gaps["year"].map(
            lambda year: str(dict(v1_years.get(str(year), {}) or {}).get("status", ""))
        )
        report_gap_path = output / "research_report_daily_task_gap.parquet"
        report_gaps.to_parquet(report_gap_path, index=False, compression="zstd")

        financial_query = f"""
          SELECT symbol,report_date,feature_available_date,field_name,
                 'pending' AS status,'field_absent_from_v1_contract' AS reason
          FROM {balance}
          CROSS JOIN (VALUES
            ('other_receivables_total'),
            ('other_payables_total'),
            ('contract_liabilities')
          ) AS fields(field_name)
          WHERE report_date BETWEEN '{START_DATE}' AND '{END_DATE}'
        """
        financial_gap_path = output / "financial_field_gap.parquet"
        financial_rows = _copy(con, financial_query, financial_gap_path)

        share_query = f"""
          SELECT symbol,trade_date,total_share,total_share_source_date,
                 share_fill_method,source,'pending' AS status
          FROM {share}
          WHERE trade_date BETWEEN '{START_DATE}' AND '{END_DATE}'
            AND total_share IS NULL
          ORDER BY trade_date,symbol
        """
        share_gap_path = output / "share_capital_total_share_gap.parquet"
        share_rows = _copy(con, share_query, share_gap_path)

        margin_query = f"""
          SELECT u.symbol,u.trade_date,u.exchange,
                 m.symbol IS NOT NULL AS detail_observed,
                 'pending' AS eligibility_state,
                 false AS source_available
          FROM {universe} u
          LEFT JOIN {margin} m USING(symbol,trade_date)
          WHERE u.trade_date BETWEEN '2011-01-01' AND '{END_DATE}'
            AND upper(coalesce(u.board,'')) IN ('MAIN','MAINBOARD','MAIN_BOARD','主板')
          ORDER BY u.trade_date,u.symbol
        """
        margin_gap_path = output / "margin_eligibility_gap.parquet"
        margin_rows = _copy(con, margin_query, margin_gap_path)

        quality_paths = _quality_diagnostics(workspace)
        quality_join = "false"
        if quality_paths:
            quality = _scan(quality_paths)
            con.execute(
                f"""
                CREATE TEMP TABLE quality_keys AS
                SELECT DISTINCT symbol,trade_date
                FROM {quality}
                WHERE quality_liquidity_keep
                  AND trade_date BETWEEN '{START_DATE}' AND '{END_DATE}'
                """
            )
            quality_join = (
                "EXISTS (SELECT 1 FROM quality_keys q "
                "WHERE q.symbol=d.symbol AND q.trade_date=d.trade_date)"
            )
        minute_query = f"""
          WITH five AS (
            SELECT symbol,trade_date,count(*) AS bar_count
            FROM {intraday}
            WHERE trade_date BETWEEN '{START_DATE}' AND '{END_DATE}'
            GROUP BY symbol,trade_date
          )
          SELECT d.symbol,d.trade_date,d.open,d.high,d.low,d.close,d.volume,d.amount,
                 coalesce(f.bar_count,0) AS existing_bar_count,
                 {quality_join} AS quality_liquidity_keep,
                 'pending' AS status
          FROM {daily} d
          LEFT JOIN five f USING(symbol,trade_date)
          WHERE d.trade_date BETWEEN '{START_DATE}' AND '{END_DATE}'
            AND d.volume>0
            AND coalesce(f.bar_count,0)<>48
          ORDER BY d.trade_date,d.symbol
        """
        minute_gap_path = output / "historical_intraday_5m_gap.parquet"
        minute_rows = _copy(con, minute_query, minute_gap_path)
        quality_minute_rows = int(
            con.execute(
                f"SELECT count(*) FROM read_parquet('{_sql_path(minute_gap_path)}') "
                "WHERE quality_liquidity_keep"
            ).fetchone()[0]
        )
    finally:
        con.close()

    latest_active_audit = _latest_audit(root, "active_audit_")
    latest_database_audit = _latest_audit(root, "database_audit_")
    copied_audits: dict[str, Any] = {}
    for name, source in (
        ("active", latest_active_audit),
        ("database", latest_database_audit),
    ):
        if source is not None:
            target = output / f"pre_repair_{name}_audit.json"
            shutil.copy2(source, target)
            copied_audits[name] = {
                "source_path": str(source),
                "copied_path": str(target),
                "sha256": _sha256(target),
            }

    files = {
        "active_dataset_inventory": dataset_path,
        "active_shard_inventory": shard_path,
        "research_report_daily_task_gap": report_gap_path,
        "financial_field_gap": financial_gap_path,
        "share_capital_total_share_gap": share_gap_path,
        "margin_eligibility_gap": margin_gap_path,
        "historical_intraday_5m_gap": minute_gap_path,
    }
    result = {
        "baseline_id": BASELINE_ID,
        "status": "frozen",
        "created_at": utc_now(),
        "scope": {
            "start_date": START_DATE,
            "end_date": END_DATE,
            "burn_in_year": 2010,
            "formal_years": "2011-2025",
            "forbidden_year": 2026,
        },
        "active_manifest_sha256": active_hash,
        "active_dataset_count": len(dataset_frame),
        "active_shard_count": len(shard_frame),
        "active_dataset_ids": dict(active.get("datasets", {}) or {}),
        "gap_counts": {
            "research_report_daily_tasks": len(report_gaps),
            "financial_field_rows": financial_rows,
            "share_capital_total_share_rows": share_rows,
            "margin_eligibility_rows": margin_rows,
            "historical_intraday_5m_stock_days": minute_rows,
            "quality_liquidity_intraday_5m_stock_days": quality_minute_rows,
        },
        "state_semantics": [
            "observed",
            "confirmed_empty",
            "source_unavailable",
            "rejected",
            "pending",
        ],
        "files": {
            name: {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in files.items()
        },
        "pre_repair_audits": copied_audits,
        "credential_persisted": False,
        "read_2026_rows": 0,
        "write_2026_rows": 0,
    }
    atomic_write_json(manifest_path, result)
    audit_path = root / "audits" / f"{BASELINE_ID}.json"
    atomic_write_json(audit_path, result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp pretraining-repair-baseline")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = freeze_baseline(
        workspace_root=str(args.workspace_root or "") or None,
        force=bool(args.force),
    )
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
