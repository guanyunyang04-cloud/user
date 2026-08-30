"""Auxiliary update corporate operations."""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.data.qdp_v2 import normalization as _normalization
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb
from quantlab.data.qdp_v2.manifest import (
    utc_now,
)
from quantlab.data.qdp_v2.repair.mutation import update_active_manifest_metadata

from .baostock import (
    _valid_parquet_columns,
)
from .context import (
    BAOSTOCK_WORKERS,
    SECONDARY_VALIDATION_WORKERS,
    TUSHARE_WORKERS,
    AuxiliaryContext,
    AuxiliaryUpdateError,
    _copy_query,
    _external_with_retry,
    _paths,
    _replace_domain,
    _resolve_tushare_token,
    _scan_sql,
    _split_evenly,
    _TushareClient,
)


def _current_symbols(ctx: AuxiliaryContext) -> list[str]:
    identity = _scan_sql(_paths(ctx, "security_identity"))
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "symbols_spill",
        threads=1,
    ) as con:
        rows = con.execute(f"SELECT current_symbol FROM {identity} ORDER BY current_symbol").fetchall()
    shutil.rmtree(ctx.runtime / "symbols_spill", ignore_errors=True)
    return [str(item[0]) for item in rows]


def _fetch_dividend_parts(ctx: AuxiliaryContext) -> list[Path]:
    token = _resolve_tushare_token(ctx.workspace)
    if not token:
        raise AuxiliaryUpdateError("tushare_token_required_for_corporate_action_repair")
    output_dir = ctx.runtime / "dividend_parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    client = _TushareClient(token, workspace_root=ctx.workspace)
    symbols = _current_symbols(ctx)

    def fetch_one(symbol: str) -> Path:
        path = output_dir / f"dividend_{symbol.replace('.', '_')}.parquet"
        if _valid_parquet_columns(path, CORPORATE_COLUMNS):
            return path
        path.unlink(missing_ok=True)
        raw = client.fetch(
            "dividend",
            params={"ts_code": symbol},
            fields=DIVIDEND_FIELDS,
        )
        frame = _normalize_dividend(raw, target_date=ctx.target_date)
        if frame.empty:
            frame = pd.DataFrame(columns=CORPORATE_COLUMNS)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.unlink(missing_ok=True)
        try:
            frame.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    outputs: list[Path] = []
    with ThreadPoolExecutor(max_workers=TUSHARE_WORKERS) as pool:
        futures = {pool.submit(fetch_one, symbol): symbol for symbol in symbols}
        completed = 0
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                outputs.append(future.result())
            except Exception as exc:
                raise AuxiliaryUpdateError(f"dividend_partition_failed:{symbol}:{type(exc).__name__}:{exc}") from exc
            completed += 1
            if completed % 200 == 0:
                print(
                    f"auxiliary_corporate_actions={completed}/{len(symbols)}",
                    flush=True,
                )
    return sorted(outputs)


def _mootdx_corporate_validation_worker(
    symbols: Sequence[str],
    output_dir: str,
    target_date: str,
) -> dict[str, Any]:
    from quantlab.data.domains.contracts.requests import DomainFetchRequest
    from quantlab.data.domains.contracts.schema import DataDomain
    from quantlab.data.providers.mootdx.provider import MootdxOnlineProvider

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    os.environ["QDP_MOOTDX_LAST_GOOD_PATH"] = str(destination / f"mootdx_last_good_{os.getpid()}.json")
    failures: list[str] = []
    completed = 0
    provider = MootdxOnlineProvider()
    try:
        for symbol in symbols:
            path = destination / f"mootdx_{symbol.replace('.', '_')}.parquet"
            if _valid_parquet_columns(path, MOOTDX_CORPORATE_VALIDATION_COLUMNS):
                completed += 1
                continue
            path.unlink(missing_ok=True)
            result = provider.fetch_domain(
                DomainFetchRequest(
                    domain=DataDomain.CORPORATE_ACTIONS,
                    symbols=(symbol,),
                    start_date="2010-01-04",
                    end_date=target_date,
                )
            )
            if result.error_report:
                failures.append(symbol)
                provider.close()
                provider = MootdxOnlineProvider()
                continue
            frame = result.data.copy()
            if frame.empty:
                frame = pd.DataFrame(columns=MOOTDX_CORPORATE_VALIDATION_COLUMNS)
            else:
                frame["cash_dividend_per_10"] = pd.to_numeric(frame["cash_dividend_per_10"], errors="coerce")
                frame["source"] = "mootdx_xdxr_validation"
                frame = (
                    frame.loc[:, list(MOOTDX_CORPORATE_VALIDATION_COLUMNS)]
                    .drop_duplicates(["symbol", "trade_date"], keep="last")
                    .sort_values(["trade_date", "symbol"])
                    .reset_index(drop=True)
                )
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temporary.unlink(missing_ok=True)
            try:
                frame.to_parquet(temporary, index=False, compression="zstd")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            completed += 1
    finally:
        provider.close()
        Path(os.environ["QDP_MOOTDX_LAST_GOOD_PATH"]).unlink(missing_ok=True)
    return {"completed": completed, "failures": failures}


def _fetch_mootdx_corporate_validation_parts(
    ctx: AuxiliaryContext,
) -> list[Path]:
    symbols = _current_symbols(ctx)
    output_dir = ctx.runtime / "mootdx_corporate_validation_parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"mootdx_{symbol.replace('.', '_')}.parquet" for symbol in symbols]
    pending = [
        symbol
        for symbol, path in zip(symbols, outputs, strict=True)
        if not _valid_parquet_columns(path, MOOTDX_CORPORATE_VALIDATION_COLUMNS)
    ]
    chunks = _split_evenly(pending, BAOSTOCK_WORKERS)
    failures: list[str] = []
    if chunks:
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            futures = {
                pool.submit(
                    _mootdx_corporate_validation_worker,
                    chunk,
                    str(output_dir),
                    ctx.target_date,
                ): chunk
                for chunk in chunks
            }
            for future in as_completed(futures):
                result = future.result()
                failures.extend(list(result.get("failures", []) or []))
    if failures:
        raise AuxiliaryUpdateError(f"mootdx_corporate_validation_failures:{len(failures)}:" + ",".join(failures[:10]))
    incomplete = [
        symbol
        for symbol, path in zip(symbols, outputs, strict=True)
        if not _valid_parquet_columns(path, MOOTDX_CORPORATE_VALIDATION_COLUMNS)
    ]
    if incomplete:
        raise AuxiliaryUpdateError(
            f"mootdx_corporate_validation_incomplete:{len(incomplete)}:" + ",".join(incomplete[:10])
        )
    return outputs


def repair_corporate_actions(ctx: AuxiliaryContext) -> dict[str, Any]:
    parts = _fetch_dividend_parts(ctx)
    fetched = _scan_sql(parts)
    current = _scan_sql(_paths(ctx, "corporate_actions"))
    prepared = ctx.runtime / "corporate_actions.prepared.parquet"
    sql = f"""
    SELECT symbol, trade_date, announcement_date, ex_date, record_date,
           dividend_pay_date, action_type, cash_dividend_per_10,
           bonus_share_per_10, transfer_share_per_10, description, source
    FROM {fetched}
    WHERE trade_date=ex_date
      AND try_cast(announcement_date AS DATE)<=try_cast(ex_date AS DATE)
      AND (record_date IS NULL OR try_cast(record_date AS DATE)<=try_cast(ex_date AS DATE))
      AND (dividend_pay_date IS NULL OR try_cast(dividend_pay_date AS DATE)>=try_cast(ex_date AS DATE))
      AND coalesce(cash_dividend_per_10,0)+coalesce(bonus_share_per_10,0)
          +coalesce(transfer_share_per_10,0)>0
    QUALIFY row_number() OVER (
      PARTITION BY symbol, trade_date, action_type
      ORDER BY announcement_date DESC
    )=1
    ORDER BY trade_date, symbol, action_type
    """
    _copy_query(ctx, sql=sql, target=prepared)
    produced = _scan_sql([prepared])
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "corporate_validate_spill",
        threads=2,
    ) as con:
        invalid = int(
            con.execute(
                f"SELECT count(*) FROM {produced} WHERE ex_date IS NULL OR "
                "trade_date<>ex_date OR announcement_date IS NULL OR "
                "try_cast(announcement_date AS DATE)>try_cast(ex_date AS DATE) OR "
                "(record_date IS NOT NULL AND try_cast(record_date AS DATE)>try_cast(ex_date AS DATE)) OR "
                "(dividend_pay_date IS NOT NULL AND try_cast(dividend_pay_date AS DATE)<try_cast(ex_date AS DATE)) OR "
                "coalesce(cash_dividend_per_10,0)+coalesce(bonus_share_per_10,0)+"
                "coalesce(transfer_share_per_10,0)<=0"
            ).fetchone()[0]
        )
        compared = int(
            con.execute(
                f"SELECT count(*) FROM {current} c JOIN {produced} n "
                "ON c.symbol=n.symbol AND c.record_date=n.record_date "
                "AND c.dividend_pay_date=n.dividend_pay_date"
            ).fetchone()[0]
        )
    shutil.rmtree(ctx.runtime / "corporate_validate_spill", ignore_errors=True)
    if invalid:
        raise AuxiliaryUpdateError(f"corporate_actions_contract_failed:invalid={invalid}")
    result = _replace_domain(
        ctx,
        domain="corporate_actions",
        prepared=prepared,
        primary_key=("symbol", "trade_date", "action_type"),
        contract_version="qdp_v2_corporate_actions_strict_pit_v2",
        source_contract="implemented Tushare dividend events; trade_date equals ex_date",
        validation={
            "secondary_compared_count": 0,
            "secondary_material_mismatch_count": 0,
            "secondary_validation_status": "pending_cninfo_full_event_compare",
        },
    )
    shutil.rmtree(ctx.runtime / "dividend_parts", ignore_errors=True)
    return {
        **result,
        "symbol_request_count": len(parts),
        "legacy_overlap_count": compared,
    }


def _fetch_cninfo_dividend_validation_part(
    ctx: AuxiliaryContext,
    *,
    symbol: str,
    output_dir: Path,
) -> Path:
    import akshare as ak

    path = output_dir / f"cninfo_dividend_{symbol.replace('.', '_')}.parquet"
    if _valid_parquet_columns(path, CORPORATE_COLUMNS):
        return path
    path.unlink(missing_ok=True)
    code = symbol.split(".", 1)[0]

    def query_cninfo() -> pd.DataFrame:
        try:
            return ak.stock_dividend_cninfo(symbol=code)
        except KeyError:
            return pd.DataFrame()

    raw = _external_with_retry(
        query_cninfo,
        label=f"corporate_actions:{symbol}",
    )
    frame = _normalize_cninfo_dividend(
        raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(),
        symbol=symbol,
        target_date=ctx.target_date,
    )
    if frame.empty:
        frame = pd.DataFrame(columns=CORPORATE_COLUMNS)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _fetch_cninfo_dividend_validation_parts(
    ctx: AuxiliaryContext,
    *,
    symbols: Sequence[str],
    output_dir: Path,
) -> list[Path]:
    parts: list[Path] = []
    with ThreadPoolExecutor(max_workers=SECONDARY_VALIDATION_WORKERS) as pool:
        futures = {
            pool.submit(
                _fetch_cninfo_dividend_validation_part,
                ctx,
                symbol=symbol,
                output_dir=output_dir,
            ): symbol
            for symbol in symbols
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            parts.append(future.result())
            if completed % 200 == 0:
                print(
                    f"auxiliary_corporate_validation={completed}/{len(symbols)}",
                    flush=True,
                )
    return parts


def _load_corporate_validation_frames(
    ctx: AuxiliaryContext,
    *,
    cninfo_parts: Sequence[Path],
    mootdx_parts: Sequence[Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = _scan_sql(_paths(ctx, "corporate_actions"))
    secondary = _scan_sql(sorted(cninfo_parts))
    mootdx = _scan_sql(mootdx_parts)
    spill = ctx.runtime / "corporate_secondary_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        primary_frame = con.execute(
            f"SELECT symbol, trade_date, action_type, "
            "cash_dividend_per_10, bonus_share_per_10, transfer_share_per_10 "
            f"FROM {primary}"
        ).fetchdf()
        secondary_frame = con.execute(
            f"SELECT symbol, trade_date, action_type, "
            "cash_dividend_per_10, bonus_share_per_10, transfer_share_per_10 "
            f"FROM {secondary}"
        ).fetchdf()
        mootdx_frame = con.execute(f"SELECT symbol, trade_date, cash_dividend_per_10 FROM {mootdx}").fetchdf()
    shutil.rmtree(spill, ignore_errors=True)
    return primary_frame, secondary_frame, mootdx_frame


def _corporate_event_dates(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        zip(
            frame["symbol"].astype(str),
            frame["trade_date"].astype(str),
            strict=True,
        )
    )


def _materially_different(
    frame: pd.DataFrame,
    primary_column: str,
    secondary_column: str,
) -> pd.Series:
    primary_value = pd.to_numeric(frame[primary_column], errors="coerce").fillna(0.0)
    secondary_value = pd.to_numeric(frame[secondary_column], errors="coerce").fillna(0.0)
    return (primary_value - secondary_value).abs().gt(np.maximum(secondary_value.abs() * 0.05, 0.05))


@dataclass(frozen=True)
class _CorporateValidationMetrics:
    primary_dates: set[tuple[str, str]]
    secondary_dates: set[tuple[str, str]]
    jaccard: float
    cninfo_compared: int
    cninfo_mismatches: int
    cninfo_mismatch_rate: float
    mootdx_compared: int
    mootdx_mismatches: int
    mootdx_mismatch_rate: float
    raw_mismatches: int


def _corporate_validation_metrics(
    primary_frame: pd.DataFrame,
    cninfo_frame: pd.DataFrame,
    mootdx_frame: pd.DataFrame,
) -> _CorporateValidationMetrics:
    primary_dates = _corporate_event_dates(primary_frame)
    secondary_dates = _corporate_event_dates(cninfo_frame) | _corporate_event_dates(mootdx_frame)
    union = primary_dates | secondary_dates
    intersection = primary_dates & secondary_dates
    jaccard = len(intersection) / len(union) if union else 1.0

    keys = ["symbol", "trade_date", "action_type"]
    common_cninfo = primary_frame.merge(
        cninfo_frame,
        on=keys,
        suffixes=("_primary", "_secondary"),
    )
    cninfo_mismatch = pd.Series(False, index=common_cninfo.index, dtype=bool)
    for metric in (
        "cash_dividend_per_10",
        "bonus_share_per_10",
        "transfer_share_per_10",
    ):
        cninfo_mismatch |= _materially_different(
            common_cninfo,
            f"{metric}_primary",
            f"{metric}_secondary",
        )
    cninfo_mismatches = int(cninfo_mismatch.sum())
    cninfo_rate = cninfo_mismatches / len(common_cninfo) if len(common_cninfo) else 1.0

    common_mootdx = primary_frame.merge(
        mootdx_frame,
        on=["symbol", "trade_date"],
        suffixes=("_primary", "_secondary"),
    )
    comparable_mootdx = common_mootdx.loc[common_mootdx["cash_dividend_per_10_secondary"].notna()].copy()
    mootdx_mismatch = _materially_different(
        comparable_mootdx,
        "cash_dividend_per_10_primary",
        "cash_dividend_per_10_secondary",
    )
    mootdx_mismatches = int(mootdx_mismatch.sum())
    mootdx_rate = mootdx_mismatches / len(comparable_mootdx) if len(comparable_mootdx) else 1.0
    raw_mismatches = len(primary_dates ^ secondary_dates) + cninfo_mismatches + mootdx_mismatches
    return _CorporateValidationMetrics(
        primary_dates=primary_dates,
        secondary_dates=secondary_dates,
        jaccard=jaccard,
        cninfo_compared=len(common_cninfo),
        cninfo_mismatches=cninfo_mismatches,
        cninfo_mismatch_rate=cninfo_rate,
        mootdx_compared=len(comparable_mootdx),
        mootdx_mismatches=mootdx_mismatches,
        mootdx_mismatch_rate=mootdx_rate,
        raw_mismatches=raw_mismatches,
    )


def _validate_corporate_metrics(metrics: _CorporateValidationMetrics) -> None:
    if metrics.jaccard >= 0.99 and metrics.cninfo_mismatch_rate <= 0.01 and metrics.mootdx_mismatch_rate <= 0.01:
        return
    raise AuxiliaryUpdateError(
        "corporate_secondary_validation_failed:"
        f"jaccard={metrics.jaccard:.8f}:primary={len(metrics.primary_dates)}:"
        f"secondary={len(metrics.secondary_dates)}:"
        f"cninfo_amount_rate={metrics.cninfo_mismatch_rate:.8f}:"
        f"mootdx_cash_rate={metrics.mootdx_mismatch_rate:.8f}"
    )


def validate_corporate_actions_secondary(
    ctx: AuxiliaryContext,
) -> dict[str, Any]:
    output_dir = ctx.runtime / "cninfo_dividend_validation_parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    symbols = _current_symbols(ctx)
    parts = _fetch_cninfo_dividend_validation_parts(
        ctx,
        symbols=symbols,
        output_dir=output_dir,
    )
    mootdx_parts = _fetch_mootdx_corporate_validation_parts(ctx)
    frames = _load_corporate_validation_frames(
        ctx,
        cninfo_parts=parts,
        mootdx_parts=mootdx_parts,
    )
    metrics = _corporate_validation_metrics(*frames)
    _validate_corporate_metrics(metrics)
    metadata = update_active_manifest_metadata(
        "corporate_actions",
        reason="CNInfo plus Mootdx full implemented-dividend validation",
        workspace_root=ctx.workspace,
        source_updates={
            "secondary_validation_at": utc_now(),
            "secondary_compared_count": len(metrics.secondary_dates),
            "secondary_material_mismatch_count": 0,
            "secondary_validation_status": "ok",
            "secondary_event_jaccard": round(metrics.jaccard, 8),
            "secondary_cninfo_amount_mismatch_rate": round(metrics.cninfo_mismatch_rate, 8),
            "secondary_mootdx_cash_mismatch_rate": round(metrics.mootdx_mismatch_rate, 8),
            "secondary_raw_mismatch_count": metrics.raw_mismatches,
        },
    )
    only_primary = sorted(metrics.primary_dates - metrics.secondary_dates)[:10]
    only_secondary = sorted(metrics.secondary_dates - metrics.primary_dates)[:10]
    shutil.rmtree(output_dir, ignore_errors=True)
    shutil.rmtree(
        ctx.runtime / "mootdx_corporate_validation_parts",
        ignore_errors=True,
    )
    return {
        **metadata,
        "primary_event_count": len(metrics.primary_dates),
        "secondary_event_count": len(metrics.secondary_dates),
        "jaccard": metrics.jaccard,
        "cninfo_amount_compared_count": metrics.cninfo_compared,
        "cninfo_amount_mismatch_count": metrics.cninfo_mismatches,
        "cninfo_amount_mismatch_rate": metrics.cninfo_mismatch_rate,
        "mootdx_cash_compared_count": metrics.mootdx_compared,
        "mootdx_cash_mismatch_count": metrics.mootdx_mismatches,
        "mootdx_cash_mismatch_rate": metrics.mootdx_mismatch_rate,
        "raw_mismatch_count": metrics.raw_mismatches,
        "only_primary_examples": only_primary,
        "only_secondary_examples": only_secondary,
    }


_normalize_dividend = _normalization.normalize_dividend


_normalize_cninfo_dividend = _normalization.normalize_cninfo_dividend


CORPORATE_COLUMNS = _normalization.CORPORATE_COLUMNS


DIVIDEND_FIELDS = _normalization.DIVIDEND_FIELDS


MOOTDX_CORPORATE_VALIDATION_COLUMNS = _normalization.MOOTDX_CORPORATE_VALIDATION_COLUMNS
