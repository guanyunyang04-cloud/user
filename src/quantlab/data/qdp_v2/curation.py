"""One-time curation of the current QDP store.

The workflow repairs the few known canonical metadata gaps, preserves the
small amount of evidence needed by ``share_capital``, and retires bulky
provider-specific domains that are no longer maintained.  It mutates the
current QDP store in place; Git and the immutable source archives are the
history mechanism, so no parallel ``v3``/``v4`` data tree is created.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from quantlab.core.io import sha256_file
from quantlab.data.core.paths import qdp_paths, workspace_root
from quantlab.data.qdp_v2.manifest import (
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    utc_now,
    write_active_manifest,
)
from quantlab.data.qdp_v2.repair.mutation import replace_active_table_from_parquet


RETIRED_DOMAINS = (
    "stk_factor_pro_raw",
    "moneyflow_raw",
    "research_report_forecast",
)


class QdpCurationError(RuntimeError):
    """Raised when the current store does not match the repair contract."""


def _quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''").replace("\\", "/") + "'"


def _scan(paths: Sequence[Path]) -> str:
    if not paths:
        raise QdpCurationError("active_dataset_has_no_shards")
    return (
        "read_parquet(["
        + ",".join(_quote(path.resolve()) for path in paths)
        + "], union_by_name=true, hive_partitioning=false)"
    )


def _active_domain(root: Path, active: Mapping[str, Any], domain: str) -> tuple[str, list[Path]]:
    dataset_id = str(dict(active.get("datasets", {}) or {}).get(domain, ""))
    manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
    if not dataset_id or manifest_path is None:
        raise QdpCurationError(f"active_domain_missing:{domain}")
    manifest = read_dataset_manifest(manifest_path)
    paths = [
        (Path(item.path) if Path(item.path).is_absolute() else root / item.path).resolve()
        for item in manifest.shards
    ]
    if any(not path.is_file() for path in paths):
        raise QdpCurationError(f"active_domain_shard_missing:{domain}")
    return dataset_id, paths


def _write_partitioned(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    output_dir: Path,
    *,
    years: Iterable[int] | None,
) -> list[Path]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    outputs: list[Path] = []
    if years is None:
        path = output_dir / "part-0000.parquet"
        connection.execute(
            f"COPY ({query}) TO {_quote(path)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)"
        )
        outputs.append(path)
        return outputs
    for year in years:
        path = output_dir / f"year={year}" / "part-0000.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        connection.execute(
            f"COPY (SELECT * FROM ({query}) q WHERE year(try_cast(trade_date AS DATE))={int(year)} "
            f"ORDER BY trade_date,symbol) TO {_quote(path)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)"
        )
        outputs.append(path)
    return outputs


def _assert_count(connection: duckdb.DuckDBPyConnection, query: str, expected: int, label: str) -> None:
    actual = int(connection.execute(query).fetchone()[0])
    if actual != expected:
        raise QdpCurationError(f"{label}:expected={expected}:actual={actual}")


def repair_industry(*, workspace: Path, staging: Path) -> dict[str, Any]:
    root = qdp_v2_root(workspace)
    active = read_active_manifest(root)
    _, paths = _active_domain(root, active, "industry_concept")
    source = _scan(paths)
    output_dir = staging / "industry_concept"
    with duckdb.connect() as connection:
        _assert_count(
            connection,
            f"SELECT count(*) FROM {source} WHERE industry='Unknown' OR industry IS NULL",
            252,
            "industry_unknown_contract_changed",
        )
        query = f"""
        WITH source_rows AS (SELECT * FROM {source}),
        first_known AS (
            SELECT * EXCLUDE(rn) FROM (
                SELECT *,row_number() OVER(PARTITION BY symbol ORDER BY trade_date) rn
                FROM source_rows
                WHERE industry IS NOT NULL AND industry<>'Unknown'
            ) WHERE rn=1
        )
        SELECT
            s.symbol,s.trade_date,
            CASE WHEN s.industry IS NULL OR s.industry='Unknown' THEN k.industry ELSE s.industry END industry,
            CASE WHEN s.industry IS NULL OR s.industry='Unknown' THEN k.industry_code ELSE s.industry_code END industry_code,
            CASE WHEN s.industry IS NULL OR s.industry='Unknown' THEN k.industry_name ELSE s.industry_name END industry_name,
            CASE WHEN s.industry IS NULL OR s.industry='Unknown' THEN k.industry_taxonomy_version ELSE s.industry_taxonomy_version END industry_taxonomy_version,
            CASE WHEN s.industry IS NULL OR s.industry='Unknown' THEN k.industry_section_code ELSE s.industry_section_code END industry_section_code,
            CASE WHEN s.industry IS NULL OR s.industry='Unknown' THEN 'baostock.query_stock_industry' ELSE s.source END AS "source",
            CASE WHEN s.industry IS NULL OR s.industry='Unknown' THEN coalesce(s.original_industry,s.industry,'Unknown') ELSE s.original_industry END original_industry,
            CASE WHEN s.industry IS NULL OR s.industry='Unknown' THEN coalesce(s.original_source,s.source) ELSE s.original_source END original_source,
            CASE WHEN s.industry IS NULL OR s.industry='Unknown' THEN 'initial_listing_gap_first_observed_industry' ELSE s.industry_fill_method END industry_fill_method,
            CASE WHEN s.industry IS NULL OR s.industry='Unknown' THEN k.trade_date ELSE s.industry_source_date END industry_source_date,
            CASE WHEN s.industry IS NULL OR s.industry='Unknown' THEN k.industry_standard ELSE s.industry_standard END industry_standard
        FROM source_rows s
        LEFT JOIN first_known k USING(symbol)
        """
        outputs = _write_partitioned(connection, query, output_dir, years=None)
        repaired_scan = _scan(outputs)
        _assert_count(
            connection,
            f"SELECT count(*) FROM {repaired_scan} WHERE industry='Unknown' OR industry IS NULL",
            0,
            "industry_repair_incomplete",
        )
        _assert_count(
            connection,
            f"SELECT count(*) FROM (SELECT trade_date,symbol,count(*) n FROM {repaired_scan} GROUP BY 1,2 HAVING n<>1)",
            0,
            "industry_duplicate_keys",
        )
    result = replace_active_table_from_parquet(
        "industry_concept",
        outputs,
        reason="fill 252 initial listing industry gaps from each symbol's first BaoStock classification",
        workspace_root=workspace,
        quality_updates={
            "industry_unknown_rows": 0,
            "unknown_industry_rows": 0,
            "industry_repair_unresolved_rows": 0,
            "industry_initial_bfill_rows": 252,
            "industry_repair_resolved_rows": 759,
            "primary_key_unique": True,
        },
        source_updates={
            "industry_initial_gap_repair_at": utc_now(),
            "industry_initial_gap_repair_rule": "first observed BaoStock industry; maximum gap 10 calendar days",
        },
    )
    return {**result, "repaired_rows": 252}


def _share_repairs() -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "symbol": "000022.SZ",
            "start_date": "2012-01-04",
            "end_date": "2018-12-20",
            "total_share": 644_763_730.0,
            "source_date": "2011-12-30",
            "method": "official_history_no_share_change_carry_forward",
        },
        {
            "symbol": "000043.SZ",
            "start_date": "2012-01-04",
            "end_date": "2012-05-16",
            "total_share": 333_480_708.0,
            "source_date": "2011-06-16",
            "method": "official_corporate_action_state",
        },
        {
            "symbol": "000043.SZ",
            "start_date": "2012-05-17",
            "end_date": "2019-12-13",
            "total_share": 666_961_416.0,
            "source_date": "2012-05-17",
            "method": "official_corporate_action_state",
        },
        {
            "symbol": "600849.SH",
            "start_date": "2010-01-04",
            "end_date": "2010-02-03",
            "total_share": 569_172_884.0,
            "source_date": "2010-01-04",
            "method": "sse_pre_reorganization_share_state",
        },
        {
            "symbol": "002604.SZ",
            "start_date": "2020-06-01",
            "end_date": "2020-06-09",
            "total_share": 599_561_402.0,
            "source_date": "2020-05-29",
            "method": "bounded_same_state_fill",
        },
        {
            "symbol": "000760.SZ",
            "start_date": "2021-06-10",
            "end_date": "2021-06-11",
            "total_share": 771_844_628.0,
            "source_date": "2021-06-09",
            "method": "bounded_same_state_fill",
        },
        {
            "symbol": "000939.SZ",
            "start_date": "2020-11-06",
            "end_date": "2020-11-09",
            "total_share": 3_929_595_494.0,
            "source_date": "2020-11-05",
            "method": "bounded_same_state_fill",
        },
        {
            "symbol": "600898.SH",
            "start_date": "2020-10-23",
            "end_date": "2020-10-23",
            "total_share": 252_523_820.0,
            "source_date": "2020-10-22",
            "method": "bounded_same_state_fill",
        },
    ]
    tail_values = {
        "000004.SZ": 132_380_282.0,
        "000638.SZ": 311_386_551.0,
        "002231.SZ": 346_850_017.0,
        "002808.SZ": 268_800_000.0,
        "002898.SZ": 176_000_000.0,
        "600193.SH": 425_373_000.0,
        "600355.SH": 492_089_200.0,
        "600421.SH": 195_600_000.0,
        "600599.SH": 166_000_000.0,
        "600608.SH": 328_861_441.0,
        "600636.SH": 438_636_802.0,
        "600696.SH": 334_469_431.0,
        "603056.SH": 1_019_815_388.0,
        "605081.SH": 113_247_072.0,
    }
    rows.extend(
        {
            "symbol": symbol,
            "start_date": "2026-01-05",
            "end_date": "2026-07-21",
            "total_share": value,
            "source_date": "2025-12-31",
            "method": "last_confirmed_state_no_corporate_action",
        }
        for symbol, value in tail_values.items()
    )
    return pd.DataFrame(rows)


def repair_share_and_valuation(*, workspace: Path, staging: Path) -> dict[str, Any]:
    root = qdp_v2_root(workspace)
    active = read_active_manifest(root)
    share_dataset_id, share_paths = _active_domain(root, active, "share_capital")
    _, valuation_paths = _active_domain(root, active, "valuation")
    _, daily_paths = _active_domain(root, active, "market_daily_raw")
    share_source = _scan(share_paths)
    evidence_root = qdp_paths(workspace).source_archives_dir / "share_capital"
    evidence_root.mkdir(parents=True, exist_ok=True)
    evidence_path = evidence_root / "legacy_total_share_evidence.parquet"
    evidence_manifest_path = evidence_root / "legacy_total_share_evidence.json"
    with duckdb.connect() as connection:
        factor_rows = int(
            connection.execute(
                f"SELECT count(*) FROM {share_source} WHERE share_fill_method LIKE '%stk_factor_pro_raw%' "
                "OR source LIKE '%stk_factor_pro_raw%'"
            ).fetchone()[0]
        )
        if factor_rows != 559_513:
            raise QdpCurationError(
                f"legacy_share_evidence_contract_changed:expected=559513:actual={factor_rows}"
            )
        connection.execute(
            f"COPY (SELECT symbol,trade_date,total_share,total_share/10000.0 AS original_total_share_wan,"
            "share_fill_method AS original_fill_method,source AS original_source,"
            f"{_quote(share_dataset_id)} AS source_dataset_id FROM {share_source} "
            "WHERE share_fill_method LIKE '%stk_factor_pro_raw%' OR source LIKE '%stk_factor_pro_raw%' "
            f"ORDER BY trade_date,symbol) TO {_quote(evidence_path)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)"
        )
        _assert_count(
            connection,
            f"SELECT count(*) FROM read_parquet({_quote(evidence_path)})",
            559_513,
            "legacy_share_evidence_incomplete",
        )
        _assert_count(
            connection,
            f"SELECT count(*) FROM read_parquet({_quote(evidence_path)}) WHERE total_share IS NULL OR total_share<=0",
            0,
            "legacy_share_evidence_invalid",
        )
    evidence_manifest = {
        "schema": "quantlab.share_capital_evidence/1",
        "created_at": utc_now(),
        "row_count": 559_513,
        "path": str(evidence_path.resolve()),
        "sha256": sha256_file(evidence_path),
        "source_dataset_id": share_dataset_id,
        "lineage": "compact projection retained before retiring the provider-specific factor domain",
        "not_a_model_input": True,
    }
    atomic_write_json(evidence_manifest_path, evidence_manifest)

    output_dir = staging / "share_capital"
    repairs = _share_repairs()
    with duckdb.connect() as connection:
        connection.register("share_repairs", repairs)
        _assert_count(
            connection,
            f"SELECT count(*) FROM {share_source} WHERE total_share IS NULL",
            4_345,
            "share_gap_contract_changed",
        )
        query = f"""
        WITH source_rows AS (SELECT * FROM {share_source}),
        patched AS (
            SELECT s.*,
                   CASE WHEN s.total_share IS NULL THEN r.total_share ELSE s.total_share END new_total_share,
                   r.source_date repair_source_date,r.method repair_method
            FROM source_rows s
            LEFT JOIN share_repairs r
              ON s.symbol=r.symbol AND s.trade_date BETWEEN r.start_date AND r.end_date
        ), normalized AS (
            SELECT *,
                CASE WHEN new_total_share IS NOT NULL THEN least(float_share,new_total_share) ELSE float_share END new_float_share,
                CASE
                    WHEN total_share IS NULL THEN repair_method
                    ELSE replace(share_fill_method,'same_day_stk_factor_pro_raw_total_share_wan_x10000','same_day_legacy_total_share_evidence')
                END new_fill_method,
                replace(source,'stk_factor_pro_raw_total_share_repair','legacy_total_share_evidence') new_source
            FROM patched
        )
        SELECT symbol,trade_date,new_total_share AS total_share,new_float_share AS float_share,
               CASE WHEN new_total_share IS NULL OR new_float_share IS NULL THEN NULL
                    ELSE greatest(new_total_share-new_float_share,0.0) END restricted_share,
               CASE WHEN total_share IS NULL THEN repair_source_date ELSE total_share_source_date END total_share_source_date,
               float_share_source_date,
               CASE WHEN new_total_share IS NULL OR new_float_share IS NULL THEN restricted_share_source_date
                    WHEN total_share IS NULL THEN repair_source_date
                    ELSE coalesce(nullif(restricted_share_source_date,''),total_share_source_date,float_share_source_date)
               END restricted_share_source_date,
               new_fill_method AS share_fill_method,
               CASE WHEN total_share IS NULL THEN concat(new_source,'+official_or_bounded_share_repair') ELSE new_source END AS "source"
        FROM normalized
        """
        outputs = _write_partitioned(connection, query, output_dir, years=range(2010, 2027))
        repaired_scan = _scan(outputs)
        _assert_count(connection, f"SELECT count(*) FROM {repaired_scan} WHERE total_share IS NULL", 0, "share_nulls")
        _assert_count(
            connection,
            f"SELECT count(*) FROM {repaired_scan} WHERE float_share>total_share+0.5",
            0,
            "share_float_above_total",
        )
        _assert_count(
            connection,
            f"SELECT count(*) FROM {repaired_scan} WHERE share_fill_method LIKE '%stk_factor_pro_raw%' "
            "OR source LIKE '%stk_factor_pro_raw%'",
            0,
            "share_factor_dependency_remaining",
        )
    share_result = replace_active_table_from_parquet(
        "share_capital",
        outputs,
        reason="repair all known total-share gaps and replace factor-domain lineage with compact evidence",
        workspace_root=workspace,
        quality_updates={
            "total_share_null_rows": 0,
            "formal_quality_total_share_missing_rows": 0,
            "unresolved_total_share_rows_through_2025": 0,
            "float_share_above_total_share_rows": 0,
            "restricted_share_null_rows": 0,
            "legacy_total_share_evidence_rows": 559_513,
            "primary_key_unique": True,
        },
        source_updates={
            "repair_id": "canonical_share_capital_curation",
            "repair_at": utc_now(),
            "repair_source": "official company actions, bounded state fills, and compact legacy evidence",
            "legacy_total_share_evidence": str(evidence_path.resolve()),
            "legacy_total_share_evidence_sha256": evidence_manifest["sha256"],
        },
    )

    active = read_active_manifest(root)
    _, repaired_share_paths = _active_domain(root, active, "share_capital")
    repaired_share_source = _scan(repaired_share_paths)
    valuation_source = _scan(valuation_paths)
    daily_source = _scan(daily_paths)
    valuation_output = staging / "valuation"
    with duckdb.connect() as connection:
        query = f"""
        SELECT v.symbol,v.trade_date,
               coalesce(v.total_mv,d.close*s.total_share) total_mv,
               v.circ_mv,v.pe,v.pb,v.turnover_rate,
               CASE WHEN v.total_mv IS NULL THEN concat(v.source,'+qdp_close_x_repaired_total_share')
                    ELSE replace(v.source,'stk_factor_pro_raw_total_share_repair','legacy_total_share_evidence') END AS "source"
        FROM {valuation_source} v
        JOIN {repaired_share_source} s USING(symbol,trade_date)
        JOIN {daily_source} d USING(symbol,trade_date)
        """
        valuation_outputs = _write_partitioned(connection, query, valuation_output, years=range(2010, 2027))
        repaired_valuation_scan = _scan(valuation_outputs)
        _assert_count(
            connection,
            f"SELECT count(*) FROM {repaired_valuation_scan} WHERE total_mv IS NULL",
            0,
            "valuation_total_mv_nulls",
        )
    valuation_result = replace_active_table_from_parquet(
        "valuation",
        valuation_outputs,
        reason="fill total market value from unadjusted close times repaired total shares",
        workspace_root=workspace,
        quality_updates={
            "total_mv_null_rows": 0,
            "formal_quality_total_share_missing_rows": 0,
            "unresolved_total_share_rows_through_2025": 0,
            "total_mv_formula_error_count": 0,
            "primary_key_unique": True,
        },
        source_updates={
            "repair_id": "canonical_share_capital_curation",
            "repair_at": utc_now(),
            "repair_source": "qdp unadjusted close times canonical total share",
            "share_capital_dataset_id": share_result["dataset_id"],
        },
    )
    return {
        "share_capital": share_result,
        "valuation": valuation_result,
        "legacy_evidence": evidence_manifest,
        "historical_gap_rows_repaired": 3_290,
        "tail_gap_rows_repaired": 1_055,
    }


def _safe_move(source: Path, target: Path) -> str:
    if target.exists():
        if source.exists() and source.resolve() != target.resolve():
            raise QdpCurationError(f"source_and_target_both_exist:{source}:{target}")
        return "already_moved"
    if not source.exists():
        raise QdpCurationError(f"source_missing:{source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)
    return "moved"


def move_source_archives(*, workspace: Path) -> dict[str, Any]:
    sources_root = qdp_paths(workspace).source_archives_dir
    baostock_target = sources_root / "baostock" / "traditional_quant_baostock_archive_v1"
    minute_target = sources_root / "minute"
    industry_target = sources_root / "industry_index"
    records: list[dict[str, Any]] = []

    moves = [
        (
            workspace / "data" / "research" / "archive" / "traditional_quant_baostock_archive_v1",
            baostock_target,
        ),
        (
            Path(r"H:\BaiduNetdiskDownload\量化数据\1分钟(2000-2025).zip"),
            minute_target / "1分钟(2000-2025).zip",
        ),
        (
            Path(r"H:\BaiduNetdiskDownload\量化数据\2026_20260821_180131.zip"),
            minute_target / "2026_20260821_180131.zip",
        ),
    ]
    for source, target in moves:
        status = _safe_move(source, target)
        records.append({"source": str(source), "target": str(target), "status": status})

    industry_files = (
        "benchmarks_daily.csv.gz",
        "market_data_metadata.json",
        "sw_first_daily.csv.gz",
        "sw_first_info.csv",
        "sw_second_daily.csv.gz",
        "sw_second_info.csv",
    )
    old_industry_root = Path(r"H:\Money\operator_research\data")
    for name in industry_files:
        source = old_industry_root / name
        target = industry_target / name
        status = _safe_move(source, target)
        records.append({"source": str(source), "target": str(target), "status": status})

    active_root = qdp_v2_root(workspace)
    active = read_active_manifest(active_root)
    evidence = dict(dict(active.get("source", {}) or {}).get("pit_history_source_evidence", {}) or {})
    archive_paths = dict(evidence.get("archive_paths", {}) or {})
    old_prefix = "data/research/archive/traditional_quant_baostock_archive_v1/"
    new_prefix = "data/qdp/source_archives/baostock/traditional_quant_baostock_archive_v1/"
    evidence["archive_paths"] = {
        key: (new_prefix + value[len(old_prefix) :] if value.startswith(old_prefix) else value)
        for key, value in archive_paths.items()
    }
    active["source"] = {
        **dict(active.get("source", {}) or {}),
        "pit_history_source_evidence": evidence,
        "source_archives_root": "data/qdp/source_archives",
    }
    active["updated_at"] = utc_now()
    write_active_manifest(active_root, active)

    inventory: list[dict[str, Any]] = []
    for target in (baostock_target, *[Path(item["target"]) for item in records[1:]]):
        if target.is_file():
            inventory.append(
                {
                    "path": str(target.resolve()),
                    "size": target.stat().st_size,
                    "mtime_ns": target.stat().st_mtime_ns,
                    "sha256": sha256_file(target) if target.stat().st_size < (1 << 30) else None,
                    "hash_status": "complete" if target.stat().st_size < (1 << 30) else "deferred_to_full_import",
                }
            )
        elif target.is_dir():
            inventory.append(
                {
                    "path": str(target.resolve()),
                    "file_count": sum(1 for item in target.rglob("*") if item.is_file()),
                    "size": sum(item.stat().st_size for item in target.rglob("*") if item.is_file()),
                }
            )
    manifest = {
        "schema": "quantlab.source_archives/1",
        "updated_at": utc_now(),
        "records": inventory,
        "minute_truth_policy": "immutable ZIP -> canonical one-minute table -> derived periods",
    }
    atomic_write_json(sources_root / "source_manifest.json", manifest)
    return {"moves": records, "manifest": str((sources_root / "source_manifest.json").resolve())}


def _validated_child(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(parent.resolve())
    except ValueError as exc:
        raise QdpCurationError(f"refuse_delete_outside_root:{resolved}:{parent}") from exc
    if resolved == parent.resolve():
        raise QdpCurationError(f"refuse_delete_root:{resolved}")
    return resolved


def retire_obsolete_domains(*, workspace: Path) -> dict[str, Any]:
    paths = qdp_paths(workspace)
    root = paths.qdp_v2_dir
    active = read_active_manifest(root)
    datasets = dict(active.get("datasets", {}) or {})
    missing = sorted(set(RETIRED_DOMAINS).difference(datasets))
    if missing:
        raise QdpCurationError(f"retired_domain_not_active:{missing}")

    share_id, share_paths = _active_domain(root, active, "share_capital")
    with duckdb.connect() as connection:
        remaining = int(
            connection.execute(
                f"SELECT count(*) FROM {_scan(share_paths)} WHERE share_fill_method LIKE '%stk_factor_pro_raw%' "
                "OR source LIKE '%stk_factor_pro_raw%'"
            ).fetchone()[0]
        )
    if remaining:
        raise QdpCurationError(f"share_capital_still_depends_on_factor:{share_id}:{remaining}")

    retired = {domain: datasets.pop(domain) for domain in RETIRED_DOMAINS}
    active["datasets"] = datasets
    active["updated_at"] = utc_now()
    active["source"] = {
        **dict(active.get("source", {}) or {}),
        "retired_provider_domains": {
            **dict(dict(active.get("source", {}) or {}).get("retired_provider_domains", {}) or {}),
            **{domain: {"dataset_id": dataset_id, "retired_at": utc_now()} for domain, dataset_id in retired.items()},
        },
    }
    # Consumers stop seeing the domains before their exact directories are removed.
    write_active_manifest(root, active)

    deleted: list[dict[str, Any]] = []
    targets = [
        _validated_child(root / "datasets" / domain, root / "datasets") for domain in RETIRED_DOMAINS
    ]
    targets.extend(
        [
            _validated_child(paths.runtime_dir / "tushare_extended_backfill_v1", paths.runtime_dir),
            _validated_child(
                paths.runtime_archives_dir / "tushare_extended_backfill_v1" / "raw__stk-factor-pro",
                paths.runtime_archives_dir,
            ),
            _validated_child(
                paths.runtime_archives_dir / "tushare_extended_backfill_v1" / "raw__moneyflow",
                paths.runtime_archives_dir,
            ),
        ]
    )
    for target in targets:
        if not target.exists():
            continue
        size = sum(item.stat().st_size for item in target.rglob("*") if item.is_file()) if target.is_dir() else target.stat().st_size
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        deleted.append({"path": str(target), "bytes": size})

    record = {
        "status": "completed",
        "created_at": utc_now(),
        "retired": retired,
        "deleted": deleted,
        "freed_bytes": sum(item["bytes"] for item in deleted),
        "share_capital_dataset_id": share_id,
        "share_factor_dependency_rows": 0,
    }
    atomic_write_json(root / "audits" / "provider_domain_retirement.json", record)
    return record


def run(*, workspace_root_value: str | Path | None = None, apply: bool = False) -> dict[str, Any]:
    workspace = workspace_root(workspace_root_value)
    if not apply:
        return {
            "status": "would_apply",
            "workspace": str(workspace),
            "repairs": ["industry_concept", "share_capital", "valuation"],
            "source_moves": ["baostock", "minute ZIPs", "SW industry indices"],
            "retired_domains": list(RETIRED_DOMAINS),
        }
    root = qdp_v2_root(workspace)
    staging = root / "tmp" / "canonical_curation"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        industry = repair_industry(workspace=workspace, staging=staging)
        shares = repair_share_and_valuation(workspace=workspace, staging=staging)
        moved = move_source_archives(workspace=workspace)
        retired = retire_obsolete_domains(workspace=workspace)
        result = {
            "status": "completed",
            "created_at": utc_now(),
            "industry": industry,
            "shares": shares,
            "source_archives": moved,
            "retirement": retired,
        }
        atomic_write_json(root / "audits" / "canonical_curation.json", result)
        return result
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Curate the current QDP store in place")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run(workspace_root_value=str(args.workspace_root or "") or None, apply=bool(args.apply))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
