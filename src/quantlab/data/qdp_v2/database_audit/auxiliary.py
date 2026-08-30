"""Database audit auxiliary checks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data.qdp_v2.auxiliary_update.context import AUXILIARY_DOMAINS
from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    resolve_manifest_path,
)

from .common import (
    _q,
)


def _register_tables(
    con: Any,
    *,
    root: Path,
    manifests: Mapping[str, DatasetManifest],
) -> dict[str, str]:
    tables: dict[str, str] = {}
    domains = (
        "market_daily_raw",
        "universe_snapshot",
        "adjust_factor",
        *AUXILIARY_DOMAINS,
    )
    for domain in domains:
        name = "aux_" + domain.replace("-", "_")
        paths = [resolve_manifest_path(item.path, root=root).resolve() for item in manifests[domain].shards]
        literals = ",".join("'" + str(path).replace("'", "''") + "'" for path in paths)
        table = _q(name)
        con.execute(
            f"CREATE OR REPLACE TEMP VIEW {table} AS SELECT * FROM read_parquet([{literals}], union_by_name=true)"
        )
        tables[domain] = table
    return tables


def _daily_key_checks(
    con: Any,
    *,
    daily: str,
    tables: Mapping[str, str],
    target: str,
) -> tuple[dict[str, int], int]:
    checks: dict[str, int] = {}
    blocking = 0
    for domain in ("industry_concept", "share_capital", "valuation"):
        table = tables[domain]
        missing = int(
            con.execute(
                f"SELECT count(*) FROM (SELECT symbol,trade_date FROM {daily} "
                "WHERE trade_date<=? EXCEPT SELECT symbol,trade_date FROM " + table + ")",
                [target],
            ).fetchone()[0]
        )
        extra = int(
            con.execute(
                f"SELECT count(*) FROM (SELECT symbol,trade_date FROM {table} "
                "EXCEPT SELECT symbol,trade_date FROM " + daily + " WHERE trade_date<=?)",
                [target],
            ).fetchone()[0]
        )
        checks[f"{domain}_missing_daily_keys"] = missing
        checks[f"{domain}_extra_daily_keys"] = extra
        blocking += missing + extra
    return checks, blocking


def _industry_checks(
    con: Any,
    *,
    table: str,
    manifest: DatasetManifest,
) -> tuple[dict[str, Any], int]:
    checks: dict[str, Any] = {}
    blocking = 0
    industry_invalid = int(
        con.execute(
            f"SELECT count(*) FROM {table} WHERE "
            "coalesce(trim(industry),'')='' OR "
            "coalesce(trim(industry_standard),'')='' OR "
            "industry_fill_method NOT IN ('direct_snapshot','prior_ffill','unavailable') OR "
            "try_cast(industry_source_date AS DATE)>try_cast(trade_date AS DATE)"
        ).fetchone()[0]
    )
    checks["industry_invalid_or_future_count"] = industry_invalid
    blocking += industry_invalid
    industry_unknown = int(
        con.execute(
            f"SELECT count(*) FROM {table} WHERE "
            "lower(trim(industry)) IN ('unknown','unclassified') OR "
            "industry_fill_method='unavailable'"
        ).fetchone()[0]
    )
    checks["industry_unknown_rows_physical"] = industry_unknown
    declared_unknown = manifest.quality.get("unknown_industry_rows")
    if declared_unknown is not None and int(declared_unknown) != industry_unknown:
        checks["industry_unknown_manifest_mismatch"] = {
            "declared": int(declared_unknown),
            "physical": industry_unknown,
        }
        blocking += 1
    else:
        checks["industry_unknown_manifest_mismatch"] = None
    return checks, blocking


def _share_checks(con: Any, table: str) -> tuple[dict[str, int], int]:
    share_invalid = int(
        con.execute(
            f"SELECT count(*) FROM {table} WHERE total_share<=0 OR float_share<0 OR "
            "float_share>total_share OR restricted_share<0 OR "
            "abs(restricted_share-greatest(total_share-float_share,0))>1e-6 OR "
            "try_cast(total_share_source_date AS DATE)>try_cast(trade_date AS DATE) OR "
            "try_cast(float_share_source_date AS DATE)>try_cast(trade_date AS DATE) OR "
            "try_cast(restricted_share_source_date AS DATE)>try_cast(trade_date AS DATE)"
        ).fetchone()[0]
    )
    return {"share_capital_invalid_or_future_count": share_invalid}, share_invalid


def _valuation_checks(
    con: Any,
    *,
    valuation: str,
    daily: str,
    share: str,
) -> tuple[dict[str, Any], int]:
    valuation_formula = int(
        con.execute(
            f"""
            SELECT count(*)
            FROM {valuation} v
            JOIN {daily} d USING(symbol,trade_date)
            JOIN {share} s USING(symbol,trade_date)
            WHERE abs(v.total_mv-d.close*s.total_share)>
                    greatest(abs(d.close*s.total_share)*1e-8,1e-6)
               OR abs(v.circ_mv-d.close*s.float_share)>
                    greatest(abs(d.close*s.float_share)*1e-8,1e-6)
            """
        ).fetchone()[0]
    )
    valuation_nulls = con.execute(
        "SELECT count(*) FILTER(WHERE pe IS NULL), "
        "count(*) FILTER(WHERE pb IS NULL), "
        "count(*) FILTER(WHERE turnover_rate IS NULL) FROM " + valuation
    ).fetchone()
    checks = {
        "valuation_formula_error_count": valuation_formula,
        "valuation_provider_null_counts": {
            "pe": int(valuation_nulls[0] or 0),
            "pb": int(valuation_nulls[1] or 0),
            "turnover_rate": int(valuation_nulls[2] or 0),
        },
    }
    return checks, valuation_formula


def _name_checks(
    con: Any,
    *,
    name_change: str,
    universe: str,
    target: str,
) -> tuple[dict[str, int], int]:
    name_invalid = int(
        con.execute(
            f"SELECT count(*) FROM {name_change} WHERE "
            "coalesce(trim(old_name),'')='' OR coalesce(trim(new_name),'')='' OR "
            "old_name=new_name"
        ).fetchone()[0]
    )
    name_latest_mismatch = int(
        con.execute(
            f"""
            WITH e AS (
              SELECT symbol,new_name FROM {name_change}
              QUALIFY row_number() OVER(PARTITION BY symbol ORDER BY trade_date DESC)=1
            ), u AS (
              SELECT symbol,trim(name) AS name FROM {universe} WHERE trade_date=?
            )
            SELECT count(*) FROM e JOIN u USING(symbol) WHERE e.new_name<>u.name
            """,
            [target],
        ).fetchone()[0]
    )
    checks = {
        "name_change_invalid_count": name_invalid,
        "name_change_latest_name_mismatch_count": name_latest_mismatch,
    }
    return checks, name_invalid + name_latest_mismatch


def _corporate_action_checks(
    con: Any,
    *,
    corporate: str,
    factor: str,
) -> tuple[dict[str, int], int]:
    corporate_invalid = int(
        con.execute(
            f"SELECT count(*) FROM {corporate} WHERE ex_date IS NULL OR "
            "trade_date<>ex_date OR announcement_date IS NULL OR "
            "try_cast(announcement_date AS DATE)>try_cast(ex_date AS DATE) OR "
            "(record_date IS NOT NULL AND try_cast(record_date AS DATE)>try_cast(ex_date AS DATE)) OR "
            "(dividend_pay_date IS NOT NULL AND try_cast(dividend_pay_date AS DATE)<try_cast(ex_date AS DATE)) OR "
            "coalesce(cash_dividend_per_10,0)+coalesce(bonus_share_per_10,0)+"
            "coalesce(transfer_share_per_10,0)<=0"
        ).fetchone()[0]
    )
    corporate_without_factor = int(
        con.execute(
            f"""
            WITH changes AS (
              SELECT symbol,trade_date,adjust_factor,
                     lag(adjust_factor) OVER(PARTITION BY symbol ORDER BY trade_date) AS prior
              FROM {factor}
            )
            SELECT count(*) FROM {corporate} c
            LEFT JOIN changes f USING(symbol,trade_date)
            WHERE f.prior IS NULL OR abs(f.adjust_factor/f.prior-1)<1e-12
            """
        ).fetchone()[0]
    )
    checks = {
        "corporate_action_invalid_count": corporate_invalid,
        "corporate_action_without_factor_change_count": corporate_without_factor,
    }
    return checks, corporate_invalid


def _index_checks(con: Any, *, table: str, target: str) -> tuple[dict[str, int], int]:
    index_row = con.execute(
        "SELECT count(*) FILTER(WHERE try_cast(source_snapshot_date AS DATE)>"
        "try_cast(trade_date AS DATE)), min(trade_date), max(trade_date), "
        "max(source_snapshot_date), count(distinct index_symbol) FROM " + table
    ).fetchone()
    index_future = int(index_row[0] or 0)
    latest_snapshot = str(index_row[3] or "")
    snapshot_age = (pd.Timestamp(target) - pd.Timestamp(latest_snapshot)).days if latest_snapshot else 999999
    maximum_snapshot_gap = int(
        con.execute(
            f"""
            WITH snapshots AS (
              SELECT DISTINCT index_symbol, source_snapshot_date FROM {table}
            ), gaps AS (
              SELECT date_diff(
                       'day',
                       try_cast(source_snapshot_date AS DATE),
                       lead(try_cast(source_snapshot_date AS DATE)) OVER (
                         PARTITION BY index_symbol
                         ORDER BY try_cast(source_snapshot_date AS DATE)
                       )
                     ) AS gap_days
              FROM snapshots
            )
            SELECT coalesce(max(gap_days), 0) FROM gaps
            """
        ).fetchone()[0]
    )
    index_contract_errors = (
        index_future
        + int(str(index_row[1] or "") != "2010-01-04")
        + int(str(index_row[2] or "") != target)
        + int(int(index_row[4] or 0) != 3)
        + int(snapshot_age > 40)
        + int(maximum_snapshot_gap > 40)
    )
    checks = {
        "index_future_snapshot_count": index_future,
        "index_latest_snapshot_age_days": snapshot_age,
        "index_maximum_snapshot_gap_days": maximum_snapshot_gap,
        "index_contract_error_count": index_contract_errors,
    }
    return checks, index_contract_errors


def _source_contract_checks(
    *,
    manifests: Mapping[str, DatasetManifest],
    target: str,
) -> tuple[dict[str, Any], int]:
    watermark_errors: list[str] = []
    secondary_status_errors: list[str] = []
    secondary_mismatches = 0
    for domain in AUXILIARY_DOMAINS:
        source = dict(manifests[domain].source or {})
        checked = str(source.get("checked_through", "") or "")
        if checked != target:
            watermark_errors.append(f"{domain}:{checked or 'missing'}")
        validation_status = str(source.get("secondary_validation_status", "") or "")
        if validation_status not in {"ok", "not_comparable"}:
            secondary_status_errors.append(f"{domain}:{validation_status or 'missing'}")
        secondary_mismatches += int(source.get("secondary_material_mismatch_count", 0) or 0)
    checks = {
        "watermark_errors": watermark_errors,
        "secondary_validation_status_errors": secondary_status_errors,
        "secondary_material_mismatch_count": secondary_mismatches,
    }
    blocking = len(watermark_errors) + len(secondary_status_errors) + secondary_mismatches
    return checks, blocking


def _auxiliary_semantics_check(
    con: Any,
    *,
    root: Path,
    manifests: Mapping[str, DatasetManifest],
    sample_limit: int,
) -> dict[str, Any]:
    del sample_limit
    tables = _register_tables(con, root=root, manifests=manifests)
    target = str(manifests["market_daily_raw"].end_date or "")
    daily = tables["market_daily_raw"]
    checks: dict[str, Any] = {}
    blocking = 0
    check_groups = (
        _daily_key_checks(con, daily=daily, tables=tables, target=target),
        _industry_checks(
            con,
            table=tables["industry_concept"],
            manifest=manifests["industry_concept"],
        ),
        _share_checks(con, tables["share_capital"]),
        _valuation_checks(
            con,
            valuation=tables["valuation"],
            daily=daily,
            share=tables["share_capital"],
        ),
        _name_checks(
            con,
            name_change=tables["name_change"],
            universe=tables["universe_snapshot"],
            target=target,
        ),
        _corporate_action_checks(
            con,
            corporate=tables["corporate_actions"],
            factor=tables["adjust_factor"],
        ),
        _index_checks(con, table=tables["index_constituents"], target=target),
        _source_contract_checks(manifests=manifests, target=target),
    )
    for group_checks, group_blocking in check_groups:
        checks.update(group_checks)
        blocking += group_blocking
    return {
        "status": "ok" if blocking == 0 else "needs_attention",
        "target_date": target,
        "blocking_error_count": int(blocking),
        "checks": checks,
    }
