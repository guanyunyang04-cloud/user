"""Auxiliary update secondary operations."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

from quantlab.data.qdp_v2 import normalization as _normalization
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb
from quantlab.data.qdp_v2.manifest import (
    utc_now,
)
from quantlab.data.qdp_v2.repair.mutation import update_active_manifest_metadata

from .context import (
    INDEX_SPECS,
    SECONDARY_VALIDATION_SAMPLE_SIZE,
    SECONDARY_VALIDATION_SEED,
    SECONDARY_VALIDATION_WORKERS,
    AuxiliaryContext,
    AuxiliaryUpdateError,
    _external_with_retry,
    _paths,
    _scan_sql,
)


def _normalize_comparison_text(value: Any) -> str:
    return _normalization.normalize_comparison_text(value)


def _normalize_industry_comparison(value: Any) -> str:
    return _normalization.normalize_industry_comparison(value)


def _parallel_secondary_rows(
    sample: pd.DataFrame,
    fetch_one: Callable[[Any], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=SECONDARY_VALIDATION_WORKERS) as pool:
        futures = [pool.submit(fetch_one, row) for row in sample.itertuples(index=False)]
        for future in as_completed(futures):
            rows.append(future.result())
    return rows


def validate_industry_secondary(ctx: AuxiliaryContext) -> dict[str, Any]:
    import akshare as ak

    industry = _scan_sql(_paths(ctx, "industry_concept"))
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "industry_secondary_spill",
        threads=1,
    ) as con:
        latest = con.execute(
            f"SELECT symbol, industry FROM {industry} WHERE trade_date=? ORDER BY symbol",
            [ctx.target_date],
        ).fetchdf()
    shutil.rmtree(ctx.runtime / "industry_secondary_spill", ignore_errors=True)
    if len(latest) < SECONDARY_VALIDATION_SAMPLE_SIZE:
        raise AuxiliaryUpdateError(f"industry_secondary_population_too_small:{len(latest)}")
    sample = latest.sample(
        n=SECONDARY_VALIDATION_SAMPLE_SIZE,
        random_state=SECONDARY_VALIDATION_SEED,
    ).sort_values("symbol")

    def fetch_one(row: Any) -> dict[str, str]:
        symbol = str(row.symbol)
        code = symbol.split(".", 1)[0]
        profile = _external_with_retry(
            lambda: ak.stock_profile_cninfo(symbol=code),
            label=f"industry_profile:{symbol}",
        )
        secondary = ""
        if isinstance(profile, pd.DataFrame) and not profile.empty:
            secondary = _normalize_comparison_text(profile.iloc[-1].get("所属行业", ""))
        if not secondary:
            changes = _external_with_retry(
                lambda: ak.stock_industry_change_cninfo(
                    symbol=code,
                    start_date="20100101",
                    end_date=ctx.target_date.replace("-", ""),
                ),
                label=f"industry_change:{symbol}",
            )
            if isinstance(changes, pd.DataFrame) and not changes.empty:
                values = changes.copy()
                if "分类标准编码" in values:
                    csrc = values["分类标准编码"].astype(str).eq("008001")
                    if csrc.any():
                        values = values.loc[csrc]
                if "变更日期" in values:
                    values = values.sort_values("变更日期")
                latest_change = values.iloc[-1]
                for column in ("行业大类", "行业中类", "行业门类"):
                    candidate = _normalize_comparison_text(latest_change.get(column, ""))
                    if candidate:
                        secondary = candidate
                        break
        return {
            "symbol": symbol,
            "primary": str(row.industry),
            "secondary": secondary,
        }

    rows = _parallel_secondary_rows(sample, fetch_one)
    comparable = [item for item in rows if item["secondary"]]
    if len(comparable) != SECONDARY_VALIDATION_SAMPLE_SIZE:
        raise AuxiliaryUpdateError(
            f"industry_secondary_incomplete:compared={len(comparable)}:expected={SECONDARY_VALIDATION_SAMPLE_SIZE}"
        )
    mismatches = [
        item
        for item in comparable
        if _normalize_industry_comparison(item["primary"]) != _normalize_industry_comparison(item["secondary"])
    ]
    match_rate = 1.0 - (len(mismatches) / len(comparable))
    if match_rate < 0.95:
        examples = ";".join(f"{item['symbol']}:{item['primary']}!={item['secondary']}" for item in mismatches[:5])
        raise AuxiliaryUpdateError(f"industry_secondary_match_rate_failed:{match_rate:.6f}:{examples}")
    metadata = update_active_manifest_metadata(
        "industry_concept",
        reason="CNInfo deterministic current-industry sample validation",
        workspace_root=ctx.workspace,
        source_updates={
            "secondary_validation_at": utc_now(),
            "secondary_compared_count": len(comparable),
            "secondary_material_mismatch_count": 0,
            "secondary_validation_status": "ok",
            "secondary_match_rate": round(match_rate, 8),
            "secondary_raw_mismatch_count": len(mismatches),
        },
    )
    return {
        **metadata,
        "compared_count": len(comparable),
        "raw_mismatch_count": len(mismatches),
        "match_rate": match_rate,
        "mismatch_examples": mismatches[:5],
    }


def _share_secondary_row(ak: Any, ctx: AuxiliaryContext, row: Any) -> dict[str, Any]:
    symbol = str(row.symbol)
    code = symbol.split(".", 1)[0]
    frame = _external_with_retry(
        lambda: ak.stock_share_change_cninfo(
            symbol=code,
            start_date="19900101",
            end_date=ctx.target_date.replace("-", ""),
        ),
        label=f"share_change:{symbol}",
    )
    total = None
    floating = None
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        data = frame.copy()
        if "变动日期" in data:
            data["变动日期"] = pd.to_datetime(data["变动日期"], errors="coerce")
            data = data.loc[data["变动日期"].le(pd.Timestamp(ctx.target_date))]
        if "公告日期" in data:
            announcement = pd.to_datetime(data["公告日期"], errors="coerce")
            data = data.loc[announcement.isna() | announcement.le(pd.Timestamp(ctx.target_date))]
        if not data.empty:
            if "变动日期" in data:
                data = data.sort_values("变动日期")
            last = data.iloc[-1]
            total_value = pd.to_numeric(pd.Series([last.get("总股本")]), errors="coerce").iloc[0]
            raw_float = last.get("人民币普通股") if pd.notna(last.get("人民币普通股")) else last.get("已流通股份")
            float_value = pd.to_numeric(pd.Series([raw_float]), errors="coerce").iloc[0]
            total = float(total_value) * 10_000.0 if pd.notna(total_value) else None
            floating = float(float_value) * 10_000.0 if pd.notna(float_value) else None
    return {
        "symbol": symbol,
        "primary_total": float(row.total_share),
        "primary_float": float(row.float_share),
        "secondary_total": total,
        "secondary_float": floating,
    }


def _share_secondary_matches(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    total_rows = [item for item in rows if item["secondary_total"] is not None]
    float_rows = [item for item in rows if item["secondary_float"] is not None]
    if len(total_rows) != SECONDARY_VALIDATION_SAMPLE_SIZE or len(float_rows) < 190:
        raise AuxiliaryUpdateError(
            "share_secondary_incomplete:"
            f"total={len(total_rows)}:float={len(float_rows)}:"
            f"expected={SECONDARY_VALIDATION_SAMPLE_SIZE}"
        )

    total_mismatches = [
        item
        for item in total_rows
        if abs(item["primary_total"] - item["secondary_total"]) / max(abs(item["secondary_total"]), 1.0) > 0.0001
    ]
    float_mismatches = [
        item
        for item in float_rows
        if abs(item["primary_float"] - item["secondary_float"]) / max(abs(item["secondary_float"]), 1.0) > 0.01
    ]
    return total_rows, float_rows, total_mismatches, float_mismatches


def validate_share_capital_secondary(ctx: AuxiliaryContext) -> dict[str, Any]:
    import akshare as ak

    share = _scan_sql(_paths(ctx, "share_capital"))
    spill = ctx.runtime / "share_secondary_spill"
    try:
        with open_guarded_duckdb(temp_directory=spill, threads=1) as con:
            latest = con.execute(
                f"SELECT symbol, total_share, float_share FROM {share} WHERE trade_date=? ORDER BY symbol",
                [ctx.target_date],
            ).fetchdf()
    finally:
        shutil.rmtree(spill, ignore_errors=True)
    if len(latest) < SECONDARY_VALIDATION_SAMPLE_SIZE:
        raise AuxiliaryUpdateError(f"share_secondary_population_too_small:{len(latest)}")
    sample = latest.sample(
        n=SECONDARY_VALIDATION_SAMPLE_SIZE,
        random_state=SECONDARY_VALIDATION_SEED,
    ).sort_values("symbol")
    rows = _parallel_secondary_rows(sample, lambda row: _share_secondary_row(ak, ctx, row))
    total_rows, float_rows, total_mismatches, float_mismatches = _share_secondary_matches(rows)
    total_match_rate = 1.0 - len(total_mismatches) / len(total_rows)
    float_match_rate = 1.0 - len(float_mismatches) / len(float_rows)
    if total_match_rate < 0.999 or float_match_rate < 0.99:
        raise AuxiliaryUpdateError(
            f"share_secondary_match_rate_failed:total={total_match_rate:.8f}:float={float_match_rate:.8f}"
        )
    metadata = update_active_manifest_metadata(
        "share_capital",
        reason="CNInfo deterministic current-share sample validation",
        workspace_root=ctx.workspace,
        source_updates={
            "secondary_validation_at": utc_now(),
            "secondary_compared_count": len(total_rows),
            "secondary_material_mismatch_count": 0,
            "secondary_validation_status": "ok",
            "secondary_total_share_match_rate": round(total_match_rate, 8),
            "secondary_float_share_match_rate": round(float_match_rate, 8),
            "secondary_raw_mismatch_count": (len(total_mismatches) + len(float_mismatches)),
        },
    )
    return {
        **metadata,
        "total_compared_count": len(total_rows),
        "float_compared_count": len(float_rows),
        "total_match_rate": total_match_rate,
        "float_match_rate": float_match_rate,
        "total_mismatch_examples": total_mismatches[:5],
        "float_mismatch_examples": float_mismatches[:5],
    }


def validate_index_secondary(ctx: AuxiliaryContext) -> dict[str, Any]:
    import akshare as ak

    index = _scan_sql(_paths(ctx, "index_constituents"))
    identity = _scan_sql(_paths(ctx, "security_identity"))
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "index_secondary_spill",
        threads=1,
    ) as con:
        current = con.execute(
            f"SELECT index_symbol, symbol FROM {index} WHERE trade_date=?",
            [ctx.target_date],
        ).fetchdf()
        allowed = {str(item[0]) for item in con.execute(f"SELECT current_symbol FROM {identity}").fetchall()}
    shutil.rmtree(ctx.runtime / "index_secondary_spill", ignore_errors=True)

    comparisons: dict[str, dict[str, Any]] = {}
    total_compared = 0
    total_raw_mismatches = 0
    for index_symbol, _, _ in INDEX_SPECS:
        code = index_symbol.split(".", 1)[0]
        official = _external_with_retry(
            lambda code=code: ak.index_stock_cons_csindex(symbol=code),
            label=f"index_constituents:{index_symbol}",
        )
        if not isinstance(official, pd.DataFrame) or official.empty:
            raise AuxiliaryUpdateError(f"index_secondary_empty:{index_symbol}")
        official_members: set[str] = set()
        for row in official.itertuples(index=False):
            payload = row._asdict()
            raw_code = str(payload.get("成分券代码", "") or "").zfill(6)
            exchange = str(payload.get("交易所", "") or "")
            suffix = (
                "SH"
                if "上海" in exchange
                else "SZ"
                if "深圳" in exchange
                else "SH"
                if raw_code.startswith(("6", "9"))
                else "SZ"
            )
            symbol = f"{raw_code}.{suffix}"
            if symbol in allowed:
                official_members.add(symbol)
        primary_members = set(
            current.loc[
                current["index_symbol"].astype(str).eq(index_symbol),
                "symbol",
            ].astype(str)
        )
        union = primary_members | official_members
        intersection = primary_members & official_members
        jaccard = len(intersection) / len(union) if union else 0.0
        raw_mismatches = len(primary_members ^ official_members)
        comparisons[index_symbol] = {
            "primary_count": len(primary_members),
            "secondary_count": len(official_members),
            "jaccard": jaccard,
            "raw_mismatch_count": raw_mismatches,
            "only_primary": sorted(primary_members - official_members)[:10],
            "only_secondary": sorted(official_members - primary_members)[:10],
        }
        total_compared += len(official_members)
        total_raw_mismatches += raw_mismatches
    failed = [symbol for symbol, payload in comparisons.items() if float(payload["jaccard"]) < 0.99]
    if failed:
        raise AuxiliaryUpdateError(
            "index_secondary_jaccard_failed:"
            + ",".join(f"{symbol}={comparisons[symbol]['jaccard']:.6f}" for symbol in failed)
        )
    metadata = update_active_manifest_metadata(
        "index_constituents",
        reason="CSIndex latest constituent-set validation",
        workspace_root=ctx.workspace,
        source_updates={
            "secondary_validation_at": utc_now(),
            "secondary_compared_count": total_compared,
            "secondary_material_mismatch_count": 0,
            "secondary_validation_status": "ok",
            "secondary_raw_mismatch_count": total_raw_mismatches,
            "secondary_minimum_jaccard": round(min(item["jaccard"] for item in comparisons.values()), 8),
        },
    )
    return {
        **metadata,
        "compared_count": total_compared,
        "raw_mismatch_count": total_raw_mismatches,
        "comparisons": comparisons,
    }


_normalize_comparison_text = _normalization.normalize_comparison_text


_normalize_industry_comparison = _normalization.normalize_industry_comparison
