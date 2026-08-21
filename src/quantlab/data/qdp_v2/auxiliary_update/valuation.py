"""Auxiliary update valuation operations."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb

from .context import (
    AuxiliaryContext,
    AuxiliaryUpdateError,
    _copy_query,
    _manifest,
    _open_dates,
    _paths,
    _replace_domain,
    _scan_sql,
)
from .shares import (
    _fetch_daily_basic_parts,
)


def _valuation_repair_dates(ctx: AuxiliaryContext) -> list[str]:
    daily = _scan_sql(_paths(ctx, "market_daily_raw"))
    valuation = _scan_sql(_paths(ctx, "valuation"))
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "valuation_plan_spill",
        threads=2,
    ) as con:
        rows = con.execute(
            f"""
            SELECT DISTINCT d.trade_date
            FROM {daily} d
            LEFT JOIN {valuation} v USING(symbol, trade_date)
            WHERE d.trade_date<=? AND (
              v.symbol IS NULL OR v.pe IS NULL OR v.pb IS NULL OR
              v.turnover_rate IS NULL
            )
            ORDER BY d.trade_date
            """,
            [ctx.target_date],
        ).fetchall()
    shutil.rmtree(ctx.runtime / "valuation_plan_spill", ignore_errors=True)
    dates = [str(item[0]) for item in rows]
    return dates if dates else _open_dates(ctx)[-20:]


def _valuation_secondary_policy(
    *,
    comparison_counts: Mapping[str, int],
    mismatch_counts: Mapping[str, int],
    median_ratios: Mapping[str, float | None],
) -> dict[str, Any]:
    comparable_metrics = ("turnover_rate",)
    not_comparable_metrics = ("pe", "pb")
    mismatch_rates = {
        metric: (
            int(mismatch_counts.get(metric, 0)) / int(comparison_counts.get(metric, 0))
            if int(comparison_counts.get(metric, 0)) > 0
            else 1.0
        )
        for metric in ("pe", "pb", "turnover_rate")
    }
    suspicious_factors = (10.0, 100.0, 10_000.0, 0.1, 0.01, 0.0001)
    unit_errors = [
        metric
        for metric, ratio in median_ratios.items()
        if ratio is not None
        and int(comparison_counts.get(metric, 0)) >= 50
        and any(abs(float(ratio) / factor - 1.0) <= 0.05 for factor in suspicious_factors)
    ]
    central_ratio_errors = [
        metric
        for metric, ratio in median_ratios.items()
        if ratio is not None and int(comparison_counts.get(metric, 0)) >= 50 and abs(float(ratio) - 1.0) > 0.05
    ]
    comparable_failures = [
        metric
        for metric in comparable_metrics
        if int(comparison_counts.get(metric, 0)) <= 0 or mismatch_rates[metric] > 0.01
    ]
    return {
        "secondary_validation_status": "not_comparable",
        "secondary_comparable_metrics": list(comparable_metrics),
        "secondary_not_comparable_metrics": list(not_comparable_metrics),
        "secondary_not_comparable_reason": (
            "BaoStock and Tushare historical PE/PB use different financial "
            "revision and effective-date policies; compare central scale only"
        ),
        "secondary_metric_comparison_counts": {key: int(value) for key, value in comparison_counts.items()},
        "secondary_metric_raw_mismatch_counts": {key: int(value) for key, value in mismatch_counts.items()},
        "secondary_metric_raw_mismatch_rates": mismatch_rates,
        "unit_errors": unit_errors,
        "central_ratio_errors": central_ratio_errors,
        "comparable_failures": comparable_failures,
    }


def _valuation_market_cap_error_count(ctx: AuxiliaryContext) -> int:
    daily = _scan_sql(_paths(ctx, "market_daily_raw"))
    share = _scan_sql(_paths(ctx, "share_capital"))
    valuation = _scan_sql(_paths(ctx, "valuation"))
    spill = ctx.runtime / "valuation_market_cap_inventory_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        count = int(
            con.execute(
                f"""
                SELECT count(*)
                FROM {valuation} v
                JOIN {daily} d USING(symbol,trade_date)
                JOIN {share} s USING(symbol,trade_date)
                WHERE v.total_mv IS NULL OR v.circ_mv IS NULL OR
                  abs(v.total_mv/nullif(d.close*s.total_share,0)-1)>1e-8 OR
                  abs(v.circ_mv/nullif(d.close*s.float_share,0)-1)>1e-8
                """
            ).fetchone()[0]
        )
    shutil.rmtree(spill, ignore_errors=True)
    return count


def _repair_valuation_market_caps(
    ctx: AuxiliaryContext,
    *,
    error_count: int,
) -> dict[str, Any]:
    daily = _scan_sql(_paths(ctx, "market_daily_raw"))
    share = _scan_sql(_paths(ctx, "share_capital"))
    current = _scan_sql(_paths(ctx, "valuation"))
    prepared = ctx.runtime / "valuation.prepared.parquet"
    sql = f"""
    SELECT v.symbol,v.trade_date,
           d.close*s.total_share AS total_mv,
           d.close*s.float_share AS circ_mv,
           v.pe,v.pb,v.turnover_rate,v.source
    FROM {current} v
    JOIN {daily} d USING(symbol,trade_date)
    JOIN {share} s USING(symbol,trade_date)
    WHERE v.trade_date<='{ctx.target_date}'
    ORDER BY v.trade_date,v.symbol
    """
    _copy_query(ctx, sql=sql, target=prepared)
    produced = _scan_sql([prepared])
    manifest = _manifest(ctx.root, ctx.datasets, "valuation")
    spill = ctx.runtime / "valuation_market_cap_validate_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        row = con.execute(
            f"""
            SELECT count(*),count(*) FILTER(WHERE total_mv IS NULL OR circ_mv IS NULL),
                   count(*) FILTER(WHERE
                     abs(v.total_mv/nullif(d.close*s.total_share,0)-1)>1e-8 OR
                     abs(v.circ_mv/nullif(d.close*s.float_share,0)-1)>1e-8)
            FROM {produced} v
            JOIN {daily} d USING(symbol,trade_date)
            JOIN {share} s USING(symbol,trade_date)
            """
        ).fetchone()
    shutil.rmtree(spill, ignore_errors=True)
    if int(row[0] or 0) != int(manifest.row_count) or int(row[1] or 0) or int(row[2] or 0):
        raise AuxiliaryUpdateError(
            "valuation_market_cap_refresh_failed:"
            f"rows={int(row[0] or 0)}:nulls={int(row[1] or 0)}:"
            f"formula_errors={int(row[2] or 0)}"
        )
    source = dict(manifest.source or {})
    result = _replace_domain(
        ctx,
        domain="valuation",
        prepared=prepared,
        primary_key=("trade_date", "symbol"),
        contract_version="qdp_v2_valuation_strict_pit_v3",
        source_contract=(
            "same-day BaoStock PE/PB/turnover; Tushare turnover is comparable "
            "while historical PE/PB revision timing is not row-comparable; "
            "market caps are close times strict-PIT shares in yuan"
        ),
        validation={
            "secondary_compared_count": int(source.get("secondary_compared_count", 0) or 0),
            "secondary_material_mismatch_count": int(source.get("secondary_material_mismatch_count", 0) or 0),
            "secondary_validation_status": str(source.get("secondary_validation_status", "not_comparable")),
        },
    )
    return {
        **result,
        "market_cap_formula_rows_refreshed": int(error_count),
    }


@dataclass(frozen=True)
class _ValuationDiagnostics:
    missing: int
    formula_errors: int
    compared: int
    mismatches: int
    comparison_counts: dict[str, int]
    mismatch_counts: dict[str, int]
    median_ratios: dict[str, float | None]
    null_counts: dict[str, int]

    @property
    def mismatch_rate(self) -> float:
        return self.mismatches / self.compared if self.compared else 1.0


def _valuation_repair_query(
    *,
    target_date: str,
    daily: str,
    share: str,
    current: str,
    fetched: str,
) -> str:
    return f"""
    SELECT d.symbol, d.trade_date,
           d.close*s.total_share AS total_mv,
           d.close*s.float_share AS circ_mv,
           coalesce(v.pe, t.pe_ttm) AS pe,
           coalesce(v.pb, t.pb) AS pb,
           coalesce(v.turnover_rate, t.turnover_rate) AS turnover_rate,
           CASE WHEN v.symbol IS NOT NULL THEN 'baostock_plus_qdp_market_cap_formula'
                ELSE 'tushare_daily_basic_gap_plus_qdp_market_cap_formula' END AS source
    FROM (SELECT * FROM {daily} WHERE trade_date<='{target_date}') d
    JOIN {share} s USING(symbol, trade_date)
    LEFT JOIN {current} v USING(symbol, trade_date)
    LEFT JOIN {fetched} t USING(symbol, trade_date)
    ORDER BY d.trade_date, d.symbol
    """


def _valuation_comparison(con: Any, *, current: str, fetched: str) -> tuple[Any, ...]:
    return con.execute(
        f"""
        WITH pairs AS (
          SELECT c.pe, c.pb, c.turnover_rate,
                 t.pe_ttm, t.pb AS tushare_pb,
                 t.turnover_rate AS tushare_turnover
          FROM {current} c
          JOIN {fetched} t USING(symbol,trade_date)
        )
        SELECT
          count(*) FILTER(WHERE
            (pe IS NOT NULL AND pe_ttm IS NOT NULL) OR
            (pb IS NOT NULL AND tushare_pb IS NOT NULL) OR
            (turnover_rate IS NOT NULL AND tushare_turnover IS NOT NULL)),
          count(*) FILTER(WHERE
            (pe IS NOT NULL AND pe_ttm IS NOT NULL AND
             abs(pe-pe_ttm)>greatest(abs(pe_ttm)*0.05,0.05)) OR
            (pb IS NOT NULL AND tushare_pb IS NOT NULL AND
             abs(pb-tushare_pb)>greatest(abs(tushare_pb)*0.05,0.05)) OR
            (turnover_rate IS NOT NULL AND tushare_turnover IS NOT NULL AND
             abs(turnover_rate-tushare_turnover)>greatest(abs(tushare_turnover)*0.05,0.05))),
          count(*) FILTER(WHERE pe IS NOT NULL AND pe_ttm IS NOT NULL),
          count(*) FILTER(WHERE pb IS NOT NULL AND tushare_pb IS NOT NULL),
          count(*) FILTER(WHERE turnover_rate IS NOT NULL AND tushare_turnover IS NOT NULL),
          count(*) FILTER(WHERE pe IS NOT NULL AND pe_ttm IS NOT NULL AND
            abs(pe-pe_ttm)>greatest(abs(pe_ttm)*0.05,0.05)),
          count(*) FILTER(WHERE pb IS NOT NULL AND tushare_pb IS NOT NULL AND
            abs(pb-tushare_pb)>greatest(abs(tushare_pb)*0.05,0.05)),
          count(*) FILTER(WHERE turnover_rate IS NOT NULL AND tushare_turnover IS NOT NULL AND
            abs(turnover_rate-tushare_turnover)>greatest(abs(tushare_turnover)*0.05,0.05)),
          median(abs(pe/pe_ttm)) FILTER(WHERE pe IS NOT NULL AND pe_ttm IS NOT NULL AND abs(pe_ttm)>1e-12),
          median(abs(pb/tushare_pb)) FILTER(WHERE pb IS NOT NULL AND tushare_pb IS NOT NULL AND abs(tushare_pb)>1e-12),
          median(abs(turnover_rate/tushare_turnover)) FILTER(
            WHERE turnover_rate IS NOT NULL AND tushare_turnover IS NOT NULL
              AND abs(tushare_turnover)>1e-12)
        FROM pairs
        """
    ).fetchone()


def _valuation_integrity_counts(
    con: Any,
    *,
    target_date: str,
    daily: str,
    share: str,
    produced: str,
) -> tuple[int, int]:
    missing = int(
        con.execute(
            f"SELECT count(*) FROM (SELECT symbol,trade_date FROM {daily} "
            "WHERE trade_date<=? EXCEPT SELECT symbol,trade_date FROM " + produced + ")",
            [target_date],
        ).fetchone()[0]
    )
    formula_errors = int(
        con.execute(
            f"SELECT count(*) FROM {produced} v JOIN {daily} d USING(symbol,trade_date) "
            f"JOIN {share} s USING(symbol,trade_date) WHERE "
            "abs(v.total_mv/nullif(d.close*s.total_share,0)-1)>1e-8 OR "
            "abs(v.circ_mv/nullif(d.close*s.float_share,0)-1)>1e-8"
        ).fetchone()[0]
    )
    return missing, formula_errors


def _valuation_diagnostics(
    ctx: AuxiliaryContext,
    *,
    daily: str,
    share: str,
    current: str,
    fetched: str,
    produced: str,
) -> _ValuationDiagnostics:
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "valuation_validate_spill",
        threads=2,
    ) as con:
        missing, formula_errors = _valuation_integrity_counts(
            con,
            target_date=ctx.target_date,
            daily=daily,
            share=share,
            produced=produced,
        )
        comparison = _valuation_comparison(con, current=current, fetched=fetched)
        nulls = con.execute(
            "SELECT count(*) FILTER(WHERE pe IS NULL), "
            "count(*) FILTER(WHERE pb IS NULL), "
            "count(*) FILTER(WHERE turnover_rate IS NULL) FROM " + produced
        ).fetchone()
    shutil.rmtree(ctx.runtime / "valuation_validate_spill", ignore_errors=True)
    return _ValuationDiagnostics(
        missing=missing,
        formula_errors=formula_errors,
        compared=int(comparison[0] or 0),
        mismatches=int(comparison[1] or 0),
        comparison_counts={
            "pe": int(comparison[2] or 0),
            "pb": int(comparison[3] or 0),
            "turnover_rate": int(comparison[4] or 0),
        },
        mismatch_counts={
            "pe": int(comparison[5] or 0),
            "pb": int(comparison[6] or 0),
            "turnover_rate": int(comparison[7] or 0),
        },
        median_ratios={
            "pe": float(comparison[8]) if comparison[8] is not None else None,
            "pb": float(comparison[9]) if comparison[9] is not None else None,
            "turnover_rate": float(comparison[10]) if comparison[10] is not None else None,
        },
        null_counts={
            "pe": int(nulls[0] or 0),
            "pb": int(nulls[1] or 0),
            "turnover_rate": int(nulls[2] or 0),
        },
    )


def _validate_valuation_diagnostics(
    diagnostics: _ValuationDiagnostics,
    policy: Mapping[str, Any],
) -> None:
    if not (
        diagnostics.missing
        or diagnostics.formula_errors
        or diagnostics.compared <= 0
        or policy["unit_errors"]
        or policy["central_ratio_errors"]
        or policy["comparable_failures"]
    ):
        return
    raise AuxiliaryUpdateError(
        "valuation_contract_failed:"
        f"missing={diagnostics.missing}:formula_errors={diagnostics.formula_errors}:"
        f"compared={diagnostics.compared}:mismatch_rate={diagnostics.mismatch_rate:.8f}:"
        f"unit_errors={','.join(policy['unit_errors'])}:"
        f"central_ratio_errors={','.join(policy['central_ratio_errors'])}:"
        f"comparable_failures={','.join(policy['comparable_failures'])}"
    )


def repair_valuation(
    ctx: AuxiliaryContext,
    *,
    daily_basic_parts: Sequence[Path] = (),
) -> dict[str, Any]:
    manifest = _manifest(ctx.root, ctx.datasets, "valuation")
    source = dict(manifest.source or {})
    if (
        str(source.get("checked_through", "")) == ctx.target_date
        and "strict" in str(source.get("source_contract", "")).lower()
    ):
        market_cap_errors = _valuation_market_cap_error_count(ctx)
        if market_cap_errors:
            return _repair_valuation_market_caps(
                ctx,
                error_count=market_cap_errors,
            )
    dates = _valuation_repair_dates(ctx)
    required_paths = {
        ctx.runtime / "daily_basic_parts" / f"daily_basic_{item.replace('-', '')}.parquet" for item in dates
    }
    parts = list(daily_basic_parts)
    if not required_paths.issubset(set(parts)):
        parts = _fetch_daily_basic_parts(ctx, dates)
    if not parts:
        raise AuxiliaryUpdateError("valuation_daily_basic_gap_parts_missing")
    daily = _scan_sql(_paths(ctx, "market_daily_raw"))
    share = _scan_sql(_paths(ctx, "share_capital"))
    current = _scan_sql(_paths(ctx, "valuation"))
    fetched = _scan_sql(parts)
    prepared = ctx.runtime / "valuation.prepared.parquet"
    _copy_query(
        ctx,
        sql=_valuation_repair_query(
            target_date=ctx.target_date,
            daily=daily,
            share=share,
            current=current,
            fetched=fetched,
        ),
        target=prepared,
    )
    produced = _scan_sql([prepared])
    diagnostics = _valuation_diagnostics(
        ctx,
        daily=daily,
        share=share,
        current=current,
        fetched=fetched,
        produced=produced,
    )
    policy = _valuation_secondary_policy(
        comparison_counts=diagnostics.comparison_counts,
        mismatch_counts=diagnostics.mismatch_counts,
        median_ratios=diagnostics.median_ratios,
    )
    _validate_valuation_diagnostics(diagnostics, policy)
    result = _replace_domain(
        ctx,
        domain="valuation",
        prepared=prepared,
        primary_key=("trade_date", "symbol"),
        contract_version="qdp_v2_valuation_strict_pit_v3",
        source_contract=(
            "same-day BaoStock PE/PB/turnover; Tushare turnover is comparable "
            "while historical PE/PB revision timing is not row-comparable; "
            "market caps are close times strict-PIT shares in yuan"
        ),
        validation={
            "secondary_compared_count": diagnostics.compared,
            "secondary_material_mismatch_count": 0,
            "secondary_raw_mismatch_count": diagnostics.mismatches,
            "secondary_raw_mismatch_rate": diagnostics.mismatch_rate,
            **{key: value for key, value in policy.items() if str(key).startswith("secondary_")},
        },
    )
    return {
        **result,
        "provider_null_counts": diagnostics.null_counts,
        "diagnostic_metric_difference_count": diagnostics.mismatches,
        "diagnostic_metric_difference_rate": diagnostics.mismatch_rate,
        "secondary_comparison_counts": diagnostics.comparison_counts,
        "secondary_metric_raw_mismatch_counts": diagnostics.mismatch_counts,
        "secondary_metric_raw_mismatch_rates": policy["secondary_metric_raw_mismatch_rates"],
        "secondary_median_ratios": diagnostics.median_ratios,
    }
