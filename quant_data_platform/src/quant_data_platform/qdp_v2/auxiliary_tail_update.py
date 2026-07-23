from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from quant_data_platform.domains.contracts import (
    DataDomain,
    DatePartitionFetchRequest,
)
from quant_data_platform.providers import (
    BaostockProvider,
    MootdxOnlineProvider,
    _mootdx_xdxr_domain_frame,
)
from quant_data_platform.qdp_v2.auxiliary_update import (
    AUXILIARY_DOMAINS,
    CORPORATE_COLUMNS,
    INDEX_SPECS,
    SECONDARY_VALIDATION_WORKERS,
    AuxiliaryContext,
    AuxiliaryUpdateError,
    _context,
    _current_symbols,
    _external_with_retry,
    _fetch_baostock_snapshots,
    _manifest,
    _normalize_cninfo_share_change,
    _normalize_cninfo_dividend,
    _paths,
    _scan_sql,
)
from quant_data_platform.qdp_v2.duckdb_resources import open_guarded_duckdb
from quant_data_platform.qdp_v2.repair import (
    append_active_shard,
    update_active_manifest_metadata,
)


class AuxiliaryTailUpdateError(RuntimeError):
    pass


def _only_changed_share_events(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    data = frame.sort_values(["symbol", "trade_date"], kind="stable").copy()
    prior_total = data.groupby("symbol", sort=False)["total_share"].shift()
    prior_float = data.groupby("symbol", sort=False)["float_share"].shift()
    changed = prior_total.isna() | prior_float.isna()
    changed |= ~np.isclose(
        pd.to_numeric(data["total_share"], errors="coerce"),
        pd.to_numeric(prior_total, errors="coerce"),
        rtol=1e-12,
        atol=0.5,
        equal_nan=True,
    )
    changed |= ~np.isclose(
        pd.to_numeric(data["float_share"], errors="coerce"),
        pd.to_numeric(prior_float, errors="coerce"),
        rtol=1e-12,
        atol=0.5,
        equal_nan=True,
    )
    return data.loc[changed].reset_index(drop=True)


def _unconfirmed_share_detections(
    detected: pd.DataFrame,
    confirmed: pd.DataFrame,
) -> list[tuple[str, str]]:
    detected_keys = (
        set(
            zip(
                detected["symbol"].astype(str),
                detected["trade_date"].astype(str),
                strict=True,
            )
        )
        if not detected.empty
        else set()
    )
    confirmed_keys = (
        set(
            zip(
                confirmed["symbol"].astype(str),
                confirmed["variation_date"].astype(str),
                strict=True,
            )
        )
        if not confirmed.empty
        else set()
    )
    return sorted(detected_keys - confirmed_keys)


def _missing_daily_keys(ctx: AuxiliaryContext, domain: str) -> pd.DataFrame:
    daily = _scan_sql(_paths(ctx, "market_daily_raw"))
    current = _scan_sql(_paths(ctx, domain))
    spill = ctx.runtime / f"tail_{domain}_inventory_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        frame = con.execute(
            f"""
            SELECT d.symbol, d.trade_date
            FROM {daily} d
            ANTI JOIN {current} c USING(symbol, trade_date)
            WHERE d.trade_date<=?
            ORDER BY d.trade_date, d.symbol
            """,
            [ctx.target_date],
        ).fetchdf()
    shutil.rmtree(spill, ignore_errors=True)
    return frame


def _checked_through(ctx: AuxiliaryContext, domain: str) -> str:
    manifest = _manifest(ctx.root, ctx.datasets, domain)
    return str(
        dict(manifest.source or {}).get("checked_through", "")
        or manifest.end_date
        or "2010-01-03"
    )


def _mark_checked(
    ctx: AuxiliaryContext,
    domain: str,
    *,
    missing_daily_keys: int | None = None,
    source_updates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    updates: dict[str, Any] = {
        "checked_through": ctx.target_date,
        **dict(source_updates or {}),
    }
    if missing_daily_keys is not None:
        updates["missing_daily_keys"] = int(missing_daily_keys)
    return update_active_manifest_metadata(
        domain,
        reason=f"free-source auxiliary tail checked through {ctx.target_date}",
        workspace_root=ctx.workspace,
        source_updates=updates,
        quality_updates={"future_source_dates": 0},
    )


def _append_if_any(
    ctx: AuxiliaryContext,
    domain: str,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    if frame.empty:
        return {"status": "already_complete", "row_count": 0}
    commit = append_active_shard(
        domain,
        frame,
        f"append free-source auxiliary tail through {ctx.target_date}",
        workspace_root=ctx.workspace,
    )
    return {"status": "updated", "row_count": int(len(frame)), "commit": commit}


def update_industry_tail(ctx: AuxiliaryContext) -> dict[str, Any]:
    missing = _missing_daily_keys(ctx, "industry_concept")
    if missing.empty:
        metadata = _mark_checked(ctx, "industry_concept", missing_daily_keys=0)
        return {"status": "already_complete", "row_count": 0, "metadata": metadata}
    dates = sorted(missing["trade_date"].astype(str).unique())
    parts = _fetch_baostock_snapshots(ctx, kind="industry", dates=dates)
    current = _scan_sql(_paths(ctx, "industry_concept"))
    fetched = _scan_sql(parts)
    spill = ctx.runtime / "tail_industry_build_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        con.register("tail_missing_keys", missing)
        frame = con.execute(
            f"""
            WITH candidates AS (
              SELECT symbol, industry,
                     coalesce(nullif(original_source, ''), source) AS source,
                     coalesce(nullif(industry_source_date, ''), trade_date)
                       AS source_date,
                     coalesce(nullif(industry_standard, ''), '证监会行业分类')
                       AS industry_standard,
                     1 AS priority
              FROM {current}
              WHERE coalesce(trim(industry), '')<>''
                AND industry_fill_method IN ('direct_snapshot','prior_ffill')
                AND try_cast(industry_source_date AS DATE)<=try_cast(trade_date AS DATE)
              UNION ALL
              SELECT symbol, industry, source, source_date,
                     coalesce(nullif(industry_standard, ''), '证监会行业分类'),
                     2 AS priority
              FROM {fetched}
              WHERE coalesce(trim(industry), '')<>''
                AND try_cast(source_date AS DATE)<=try_cast(snapshot_query_date AS DATE)
            ), dedup AS (
              SELECT * EXCLUDE(priority)
              FROM candidates
              QUALIFY row_number() OVER(
                PARTITION BY symbol, source_date ORDER BY priority DESC
              )=1
            )
            SELECT m.symbol, m.trade_date, c.industry, c.source,
                   c.industry AS original_industry,
                   c.source AS original_source,
                   CASE WHEN c.source_date=m.trade_date THEN 'direct_snapshot'
                        ELSE 'prior_ffill' END AS industry_fill_method,
                   c.source_date AS industry_source_date,
                   c.industry_standard
            FROM tail_missing_keys m
            ASOF LEFT JOIN dedup c
              ON m.symbol=c.symbol AND m.trade_date>=c.source_date
            ORDER BY m.trade_date, m.symbol
            """
        ).fetchdf()
    shutil.rmtree(spill, ignore_errors=True)
    if len(frame) != len(missing) or frame["industry"].isna().any():
        raise AuxiliaryTailUpdateError(
            f"industry_tail_unresolved:{len(missing)-frame['industry'].notna().sum()}"
        )
    result = _append_if_any(ctx, "industry_concept", frame)
    shutil.rmtree(ctx.runtime / "baostock_industry_parts", ignore_errors=True)
    remaining = _missing_daily_keys(ctx, "industry_concept")
    if not remaining.empty:
        raise AuxiliaryTailUpdateError(
            f"industry_tail_post_commit_missing:{len(remaining)}"
        )
    result["metadata"] = _mark_checked(
        ctx, "industry_concept", missing_daily_keys=0
    )
    return result


def update_name_change_tail(ctx: AuxiliaryContext) -> dict[str, Any]:
    universe = _scan_sql(_paths(ctx, "universe_snapshot"))
    current = _scan_sql(_paths(ctx, "name_change"))
    spill = ctx.runtime / "tail_name_change_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        frame = con.execute(
            f"""
            WITH normalized AS (
              SELECT symbol, trade_date, trim(name) AS name
              FROM {universe}
              WHERE trade_date<='{ctx.target_date}'
                AND coalesce(trim(name), '')<>''
            ), transitions AS (
              SELECT symbol, trade_date,
                     lag(name) OVER(PARTITION BY symbol ORDER BY trade_date)
                       AS old_name,
                     name AS new_name
              FROM normalized
            ), events AS (
              SELECT symbol, trade_date, old_name, new_name,
                     'short_name' AS change_type,
                     'qdp_universe_tail_transition' AS source
              FROM transitions
              WHERE old_name IS NOT NULL AND old_name<>new_name
            )
            SELECT e.* FROM events e
            ANTI JOIN {current} c USING(symbol, trade_date, change_type)
            ORDER BY e.trade_date, e.symbol
            """
        ).fetchdf()
    shutil.rmtree(spill, ignore_errors=True)
    result = _append_if_any(ctx, "name_change", frame)
    result["metadata"] = _mark_checked(ctx, "name_change")
    return result


def _index_tail_dates(ctx: AuxiliaryContext) -> list[str]:
    current = _scan_sql(_paths(ctx, "index_constituents"))
    calendar = _scan_sql(_paths(ctx, "trading_calendar"))
    spill = ctx.runtime / "tail_index_inventory_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=1) as con:
        dates = [
            str(item[0])
            for item in con.execute(
                f"""
                SELECT DISTINCT c.trade_date
                FROM {calendar} c
                WHERE c.exchange='SSE' AND c.is_open
                  AND try_cast(c.trade_date AS DATE)>(
                    SELECT coalesce(
                      max(try_cast(trade_date AS DATE)),
                      DATE '2010-01-03'
                    )
                    FROM {current}
                  )
                  AND try_cast(c.trade_date AS DATE)<=try_cast(? AS DATE)
                ORDER BY c.trade_date
                """,
                [ctx.target_date],
            ).fetchall()
        ]
    shutil.rmtree(spill, ignore_errors=True)
    return dates


def update_index_tail(ctx: AuxiliaryContext) -> dict[str, Any]:
    current = _scan_sql(_paths(ctx, "index_constituents"))
    dates = _index_tail_dates(ctx)
    if not dates:
        metadata = _mark_checked(ctx, "index_constituents")
        return {"status": "already_complete", "row_count": 0, "metadata": metadata}
    parts = _fetch_baostock_snapshots(ctx, kind="index", dates=dates)
    fetched = _scan_sql(parts)
    identity = _scan_sql(_paths(ctx, "security_identity"))
    spill = ctx.runtime / "tail_index_build_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        frame = con.execute(
            f"""
            SELECT index_symbol, symbol,
                   snapshot_query_date AS trade_date,
                   index_name, source, source_snapshot_date
            FROM {fetched}
            WHERE symbol IN (SELECT current_symbol FROM {identity})
            QUALIFY row_number() OVER(
              PARTITION BY index_symbol, symbol, snapshot_query_date
              ORDER BY source_snapshot_date DESC
            )=1
            ORDER BY trade_date, index_symbol, symbol
            """
        ).fetchdf()
    shutil.rmtree(spill, ignore_errors=True)
    coverage = frame.groupby("trade_date")["index_symbol"].nunique()
    if coverage.empty or int(coverage.min()) != 3 or int(coverage.max()) != 3:
        raise AuxiliaryTailUpdateError(
            f"index_tail_incomplete:{coverage.to_dict()}"
        )
    result = _append_if_any(ctx, "index_constituents", frame)
    shutil.rmtree(ctx.runtime / "baostock_index_parts", ignore_errors=True)
    result["metadata"] = _mark_checked(ctx, "index_constituents")
    return result


def _fetch_mootdx_evidence(
    ctx: AuxiliaryContext,
    *,
    start_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = _current_symbols(ctx)
    provider = MootdxOnlineProvider()
    try:
        raw = provider.fetch_xdxr_raw(symbols)
    finally:
        provider.close()
    if raw.error_report:
        raise AuxiliaryTailUpdateError(
            f"mootdx_xdxr_symbol_errors:{len(raw.error_report)}"
        )
    share = _mootdx_xdxr_domain_frame(
        raw.data,
        domain=DataDomain.SHARE_CAPITAL,
        source="mootdx_online",
        as_of_date=ctx.target_date,
    )
    corporate = _mootdx_xdxr_domain_frame(
        raw.data,
        domain=DataDomain.CORPORATE_ACTIONS,
        source="mootdx_online",
        as_of_date=ctx.target_date,
    )
    share = _only_changed_share_events(share)
    for frame in (share, corporate):
        if not frame.empty:
            frame.drop(
                frame.index[
                    ~frame["trade_date"].astype(str).between(
                        start_date, ctx.target_date, inclusive="right"
                    )
                ],
                inplace=True,
            )
            frame.reset_index(drop=True, inplace=True)
    return share, corporate


def _fetch_cninfo_share_events(
    ctx: AuxiliaryContext,
    symbols: Sequence[str],
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(
            columns=[
                "symbol",
                "variation_date",
                "source_date",
                "total_share",
                "float_share",
                "source",
            ]
        )
    import akshare as ak

    def fetch_one(symbol: str) -> pd.DataFrame:
        code = symbol.split(".", 1)[0]
        raw = _external_with_retry(
            lambda: ak.stock_share_change_cninfo(
                symbol=code,
                start_date="19900101",
                end_date=ctx.target_date.replace("-", ""),
            ),
            label=f"tail_share_change:{symbol}",
        )
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            return pd.DataFrame()
        return _normalize_cninfo_share_change(
            raw,
            symbol=symbol,
            target_date=ctx.target_date,
            source="cninfo_confirmed_mootdx_xdxr",
        )

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=SECONDARY_VALIDATION_WORKERS) as pool:
        futures = {pool.submit(fetch_one, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            frame = future.result()
            if not frame.empty:
                frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def update_share_capital_tail(
    ctx: AuxiliaryContext,
    *,
    detected_events: pd.DataFrame,
) -> dict[str, Any]:
    missing = _missing_daily_keys(ctx, "share_capital")
    if missing.empty:
        metadata = _mark_checked(ctx, "share_capital", missing_daily_keys=0)
        return {"status": "already_complete", "row_count": 0, "metadata": metadata}
    current = _scan_sql(_paths(ctx, "share_capital"))
    spill = ctx.runtime / "tail_share_prior_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=1) as con:
        con.register("tail_missing_keys", missing)
        unseeded = {
            str(item[0])
            for item in con.execute(
                f"""
                SELECT DISTINCT m.symbol
                FROM tail_missing_keys m
                WHERE NOT EXISTS (
                  SELECT 1 FROM {current} c
                  WHERE c.symbol=m.symbol AND c.trade_date<m.trade_date
                )
                """
            ).fetchall()
        }
    shutil.rmtree(spill, ignore_errors=True)
    affected = set(unseeded)
    if not detected_events.empty:
        affected.update(detected_events["symbol"].astype(str))
    confirmed = _fetch_cninfo_share_events(ctx, sorted(affected))
    if affected and confirmed.empty:
        raise AuxiliaryTailUpdateError(
            f"share_tail_cninfo_confirmation_empty:{len(affected)}"
        )
    # Mootdx xdxr is a detector, not the PIT authority.  Its dates can be a
    # weekend publication/update date and the first row returned for a symbol
    # can simply repeat the already-known share count.  Requiring an exact
    # (symbol, date) match would therefore block a healthy tail or, worse,
    # tempt callers to apply an unconfirmed secondary-source value.  Only the
    # CNInfo rows below are eligible to change the share history; unmatched
    # detector rows are retained as audit evidence while the last officially
    # visible value is carried forward.
    unconfirmed_detections = _unconfirmed_share_detections(
        detected_events,
        confirmed,
    )
    spill = ctx.runtime / "tail_share_build_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        con.register("tail_missing_keys", missing)
        empty_confirmed = pd.DataFrame(
            {
                "symbol": pd.Series(dtype="object"),
                "variation_date": pd.Series(dtype="object"),
                "source_date": pd.Series(dtype="object"),
                "total_share": pd.Series(dtype="float64"),
                "float_share": pd.Series(dtype="float64"),
                "source": pd.Series(dtype="object"),
            }
        )
        con.register(
            "tail_confirmed_share",
            confirmed if not confirmed.empty else empty_confirmed,
        )
        frame = con.execute(
            f"""
            WITH candidates AS (
              SELECT symbol,
                     cast(greatest(
                       try_cast(total_share_source_date AS DATE),
                       try_cast(float_share_source_date AS DATE)
                     ) AS VARCHAR) AS source_date,
                     total_share, float_share, source, 1 AS priority
              FROM {current}
              WHERE total_share>0 AND float_share>=0 AND float_share<=total_share
                AND try_cast(total_share_source_date AS DATE)<=try_cast(trade_date AS DATE)
                AND try_cast(float_share_source_date AS DATE)<=try_cast(trade_date AS DATE)
              UNION ALL
              SELECT symbol, source_date, total_share, float_share, source,
                     2 AS priority
              FROM tail_confirmed_share
            ), dedup AS (
              SELECT symbol, source_date, total_share, float_share, source
              FROM candidates
              QUALIFY row_number() OVER(
                PARTITION BY symbol, source_date ORDER BY priority DESC
              )=1
            )
            SELECT m.symbol, m.trade_date, c.total_share, c.float_share,
                   greatest(c.total_share-c.float_share, 0.0) AS restricted_share,
                   c.source_date AS total_share_source_date,
                   c.source_date AS float_share_source_date,
                   c.source_date AS restricted_share_source_date,
                   CASE WHEN c.source_date=m.trade_date THEN 'direct_daily'
                        ELSE 'prior_ffill' END AS share_fill_method,
                   c.source
            FROM tail_missing_keys m
            ASOF LEFT JOIN dedup c
              ON m.symbol=c.symbol AND m.trade_date>=c.source_date
            ORDER BY m.trade_date, m.symbol
            """
        ).fetchdf()
    shutil.rmtree(spill, ignore_errors=True)
    if len(frame) != len(missing) or frame["total_share"].isna().any():
        raise AuxiliaryTailUpdateError(
            f"share_tail_unresolved:{len(missing)-frame['total_share'].notna().sum()}"
        )
    result = _append_if_any(ctx, "share_capital", frame)
    remaining = _missing_daily_keys(ctx, "share_capital")
    if not remaining.empty:
        raise AuxiliaryTailUpdateError(
            f"share_tail_post_commit_missing:{len(remaining)}"
        )
    result["detected_event_count"] = int(len(detected_events))
    result["confirmed_symbol_count"] = len(affected)
    result["unconfirmed_detection_count"] = len(unconfirmed_detections)
    result["unconfirmed_detection_examples"] = [
        {"symbol": symbol, "detector_date": trade_date}
        for symbol, trade_date in unconfirmed_detections[:20]
    ]
    result["metadata"] = _mark_checked(
        ctx,
        "share_capital",
        missing_daily_keys=0,
        source_updates={
            "tail_unconfirmed_detector_event_count": len(unconfirmed_detections),
            "tail_unconfirmed_detector_event_examples": [
                {"symbol": symbol, "detector_date": trade_date}
                for symbol, trade_date in unconfirmed_detections[:20]
            ],
        },
    )
    return result


def update_valuation_tail(
    ctx: AuxiliaryContext,
    *,
    baostock_valuation_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    missing = _missing_daily_keys(ctx, "valuation")
    if missing.empty:
        metadata = _mark_checked(ctx, "valuation", missing_daily_keys=0)
        return {"status": "already_complete", "row_count": 0, "metadata": metadata}
    frames: list[pd.DataFrame] = []
    cache = (
        Path(baostock_valuation_cache_path).resolve()
        if baostock_valuation_cache_path is not None
        else None
    )
    if cache is not None and cache.is_file():
        cached = pd.read_parquet(cache)
        required = {"symbol", "trade_date", "pe", "pb", "turnover_rate"}
        if required.issubset(cached.columns):
            frames.append(cached.loc[:, sorted(required)].copy())
    cached_dates = (
        set(pd.concat(frames, ignore_index=True)["trade_date"].astype(str))
        if frames
        else set()
    )
    missing_dates = sorted(
        set(missing["trade_date"].astype(str)).difference(cached_dates)
    )
    provider = BaostockProvider()
    try:
        for trade_date in missing_dates:
            result = provider.fetch_date_partition(
                DatePartitionFetchRequest(
                    domain=DataDomain.VALUATION,
                    trade_date=trade_date,
                    universe_kind="all_a",
                    fetch_mode="date_snapshot",
                )
            )
            if result.error_report:
                raise AuxiliaryTailUpdateError(
                    f"valuation_tail_provider_errors:{trade_date}"
                )
            if not result.data.empty:
                normalized = result.data.rename(
                    columns={
                        "provider_symbol": "symbol",
                        "pe_ttm": "pe",
                        "pb_mrq": "pb",
                    }
                )
                frames.append(
                    normalized.loc[
                        :, ["symbol", "trade_date", "pe", "pb", "turnover_rate"]
                    ].copy()
                )
    finally:
        provider.close()
    fetched = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if fetched.empty:
        raise AuxiliaryTailUpdateError("valuation_tail_provider_empty")
    daily = _scan_sql(_paths(ctx, "market_daily_raw"))
    share = _scan_sql(_paths(ctx, "share_capital"))
    spill = ctx.runtime / "tail_valuation_build_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        con.register("tail_missing_keys", missing)
        con.register("tail_valuation_fetched", fetched)
        frame = con.execute(
            f"""
            SELECT m.symbol, m.trade_date,
                   d.close*s.total_share AS total_mv,
                   d.close*s.float_share AS circ_mv,
                   v.pe, v.pb, v.turnover_rate,
                   'baostock_tail_plus_qdp_market_cap_formula' AS source
            FROM tail_missing_keys m
            JOIN {daily} d USING(symbol, trade_date)
            JOIN {share} s USING(symbol, trade_date)
            LEFT JOIN tail_valuation_fetched v USING(symbol, trade_date)
            ORDER BY m.trade_date, m.symbol
            """
        ).fetchdf()
    shutil.rmtree(spill, ignore_errors=True)
    if len(frame) != len(missing) or frame[["total_mv", "circ_mv"]].isna().any().any():
        raise AuxiliaryTailUpdateError("valuation_tail_market_cap_unresolved")
    result = _append_if_any(ctx, "valuation", frame)
    remaining = _missing_daily_keys(ctx, "valuation")
    if not remaining.empty:
        raise AuxiliaryTailUpdateError(
            f"valuation_tail_post_commit_missing:{len(remaining)}"
        )
    result["metadata"] = _mark_checked(ctx, "valuation", missing_daily_keys=0)
    return result


def update_corporate_actions_tail(
    ctx: AuxiliaryContext,
    *,
    detected_events: pd.DataFrame,
) -> dict[str, Any]:
    if detected_events.empty:
        metadata = _mark_checked(ctx, "corporate_actions")
        return {"status": "already_complete", "row_count": 0, "metadata": metadata}
    import akshare as ak

    symbols = sorted(set(detected_events["symbol"].astype(str)))

    def fetch_one(symbol: str) -> pd.DataFrame:
        code = symbol.split(".", 1)[0]

        def query() -> pd.DataFrame:
            try:
                return ak.stock_dividend_cninfo(symbol=code)
            except KeyError:
                return pd.DataFrame()

        raw = _external_with_retry(
            query,
            label=f"tail_corporate_actions:{symbol}",
        )
        return _normalize_cninfo_dividend(
            raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(),
            symbol=symbol,
            target_date=ctx.target_date,
        )

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=SECONDARY_VALIDATION_WORKERS) as pool:
        futures = {pool.submit(fetch_one, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            frame = future.result()
            if not frame.empty:
                frames.append(frame)
    confirmed = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not confirmed.empty:
        announcement = pd.to_datetime(
            confirmed["announcement_date"], errors="coerce"
        )
        ex_date = pd.to_datetime(confirmed["ex_date"], errors="coerce")
        record_date = pd.to_datetime(confirmed["record_date"], errors="coerce")
        pay_date = pd.to_datetime(
            confirmed["dividend_pay_date"], errors="coerce"
        )
        implemented = (
            pd.to_numeric(
                confirmed["cash_dividend_per_10"], errors="coerce"
            ).fillna(0.0)
            + pd.to_numeric(
                confirmed["bonus_share_per_10"], errors="coerce"
            ).fillna(0.0)
            + pd.to_numeric(
                confirmed["transfer_share_per_10"], errors="coerce"
            ).fillna(0.0)
        )
        confirmed = confirmed.loc[
            announcement.notna()
            & ex_date.notna()
            & announcement.le(ex_date)
            & (record_date.isna() | record_date.le(ex_date))
            & (pay_date.isna() | pay_date.ge(ex_date))
            & implemented.gt(0)
        ].reset_index(drop=True)
    detected_keys = set(
        zip(
            detected_events["symbol"].astype(str),
            detected_events["trade_date"].astype(str),
            strict=True,
        )
    )
    confirmed_keys = (
        set(zip(confirmed["symbol"], confirmed["trade_date"], strict=True))
        if not confirmed.empty
        else set()
    )
    unresolved = sorted(detected_keys - confirmed_keys)
    if unresolved:
        raise AuxiliaryTailUpdateError(
            f"corporate_tail_unconfirmed_events:{len(unresolved)}:{unresolved[:5]}"
        )
    current = _scan_sql(_paths(ctx, "corporate_actions"))
    spill = ctx.runtime / "tail_corporate_build_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=1) as con:
        con.register("tail_corporate_confirmed", confirmed)
        frame = con.execute(
            f"""
            SELECT n.* FROM tail_corporate_confirmed n
            ANTI JOIN {current} c USING(symbol, trade_date, action_type)
            WHERE n.trade_date>'{_checked_through(ctx, 'corporate_actions')}'
              AND n.trade_date<='{ctx.target_date}'
            ORDER BY n.trade_date, n.symbol, n.action_type
            """
        ).fetchdf()
    shutil.rmtree(spill, ignore_errors=True)
    frame = frame.loc[:, list(CORPORATE_COLUMNS)] if not frame.empty else frame
    result = _append_if_any(ctx, "corporate_actions", frame)
    result["detected_event_count"] = len(detected_keys)
    result["confirmed_event_count"] = len(confirmed_keys)
    result["metadata"] = _mark_checked(
        ctx,
        "corporate_actions",
        source_updates={
            "secondary_validation_at": pd.Timestamp.utcnow().isoformat(),
            "secondary_compared_count": len(confirmed_keys),
            "secondary_material_mismatch_count": 0,
            "secondary_validation_status": "ok",
        },
    )
    return result


def run_auxiliary_tail_update(
    *,
    as_of_date: str,
    workspace_root: str | Path | None = None,
    domains: Sequence[str] = AUXILIARY_DOMAINS,
    baostock_valuation_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    ctx = _context(as_of_date=as_of_date, workspace_root=workspace_root)
    selected = tuple(str(item) for item in domains)
    unknown = sorted(set(selected).difference(AUXILIARY_DOMAINS))
    if unknown:
        raise ValueError(f"unknown_auxiliary_domains:{','.join(unknown)}")
    results: dict[str, Any] = {}
    if "industry_concept" in selected:
        results["industry_concept"] = update_industry_tail(ctx)
    if "name_change" in selected:
        results["name_change"] = update_name_change_tail(ctx)
    if "index_constituents" in selected:
        results["index_constituents"] = update_index_tail(ctx)

    needs_xdxr = bool(
        set(selected).intersection({"share_capital", "corporate_actions"})
    )
    share_events = pd.DataFrame()
    corporate_events = pd.DataFrame()
    if needs_xdxr:
        start = min(
            _checked_through(ctx, domain)
            for domain in ("share_capital", "corporate_actions")
            if domain in selected
        )
        share_events, corporate_events = _fetch_mootdx_evidence(
            ctx,
            start_date=start,
        )
    if "share_capital" in selected:
        results["share_capital"] = update_share_capital_tail(
            ctx,
            detected_events=share_events,
        )
    if "valuation" in selected:
        results["valuation"] = update_valuation_tail(
            ctx,
            baostock_valuation_cache_path=baostock_valuation_cache_path,
        )
    if "corporate_actions" in selected:
        results["corporate_actions"] = update_corporate_actions_tail(
            ctx,
            detected_events=corporate_events,
        )
    (ctx.runtime / "mootdx_last_good_5m.json").unlink(missing_ok=True)
    if baostock_valuation_cache_path is not None:
        Path(baostock_valuation_cache_path).resolve().unlink(missing_ok=True)
    return {
        "status": "updated",
        "as_of_date": ctx.target_date,
        "domains": results,
        "provider_policy": "baostock+mootdx_detection+cninfo_confirmation; no_tushare",
    }


__all__ = [
    "AuxiliaryTailUpdateError",
    "run_auxiliary_tail_update",
    "update_corporate_actions_tail",
    "update_index_tail",
    "update_industry_tail",
    "update_name_change_tail",
    "update_share_capital_tail",
    "update_valuation_tail",
]
