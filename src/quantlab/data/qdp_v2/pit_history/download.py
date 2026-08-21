"""PIT history download operations."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data.qdp_v2.auxiliary_update import _external_with_retry
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb
from quantlab.data.qdp_v2.manifest import (
    atomic_write_json,
    utc_now,
)
from quantlab.data.qdp_v2.repair import (
    resolve_active_domain,
)

from .config import (
    CNINFO_SHARE_NORMALIZED_COLUMNS,
    NAME_INTERVAL_COLUMNS,
    PART_SEMANTIC_VERSION,
    SSE_ST_TRANSITIONS,
    _name_implies_st,
    _normalize_cninfo_share_change,
    _normalize_eastmoney_history,
    _normalize_sina_factors,
)
from .context import (
    PitHistoryContext,
    PitHistoryError,
    _archive_paths,
    _atomic_parquet,
    _valid_parquet,
)


def _archive_cache_is_current(output: Path, metadata: Path, symbols: list[str]) -> bool:
    if _valid_parquet(
        output,
        ("trade_date", "symbol", "tradestatus", "isST", "history_source"),
    ) and metadata.is_file():
        try:
            cached = json.loads(metadata.read_text(encoding="utf-8"))
            return cached.get("semantic_version") == PART_SEMANTIC_VERSION and cached.get("symbols") == symbols
        except (OSError, ValueError, TypeError):
            return False
    return False


def _archive_cache_query() -> str:
    return """
        WITH old_rows AS (
            SELECT
                cast(b.date AS DATE) AS trade_date, cast(b.code AS VARCHAR) AS symbol,
                cast(b.open AS DOUBLE) AS open, cast(b.high AS DOUBLE) AS high,
                cast(b.low AS DOUBLE) AS low, cast(b.close AS DOUBLE) AS close,
                cast(b.volume AS DOUBLE) AS volume, cast(b.amount AS DOUBLE) AS amount,
                cast(b.tradestatus AS VARCHAR) AS tradestatus, cast(b.isST AS VARCHAR) AS isST,
                NULL::DOUBLE AS turn, NULL::DOUBLE AS pctChg, NULL::DOUBLE AS peTTM,
                NULL::DOUBLE AS pbMRQ, NULL::DOUBLE AS psTTM, NULL::DOUBLE AS pcfNcfTTM,
                cast(n.name_on_date AS VARCHAR) AS name_on_date,
                'protected_baostock_archive_2012_2015' AS history_source
            FROM read_parquet(?) b
            JOIN targets t ON cast(b.code AS VARCHAR)=t.symbol
            LEFT JOIN read_parquet(?, union_by_name=true) n ON b.date=n.date AND b.code=n.code
        ), recent_rows AS (
            SELECT
                cast(b.date AS DATE) AS trade_date, cast(b.code AS VARCHAR) AS symbol,
                cast(b.open AS DOUBLE) AS open, cast(b.high AS DOUBLE) AS high,
                cast(b.low AS DOUBLE) AS low, cast(b.close AS DOUBLE) AS close,
                cast(b.volume AS DOUBLE) AS volume, cast(b.amount AS DOUBLE) AS amount,
                cast(b.tradestatus AS VARCHAR) AS tradestatus, cast(b.isST AS VARCHAR) AS isST,
                cast(m.turn AS DOUBLE) AS turn, cast(m.pctChg AS DOUBLE) AS pctChg,
                cast(m.peTTM AS DOUBLE) AS peTTM, cast(m.pbMRQ AS DOUBLE) AS pbMRQ,
                cast(m.psTTM AS DOUBLE) AS psTTM, cast(m.pcfNcfTTM AS DOUBLE) AS pcfNcfTTM,
                cast(coalesce(u.name_on_date,n.name_on_date) AS VARCHAR) AS name_on_date,
                'protected_baostock_archive_2016_2026' AS history_source
            FROM read_parquet(?) b
            JOIN targets t ON cast(b.code AS VARCHAR)=t.symbol
            LEFT JOIN read_parquet(?) m ON b.date=m.date AND b.code=m.code
            LEFT JOIN read_parquet(?) u ON b.date=u.date AND b.code=u.code
            LEFT JOIN read_parquet(?) n ON b.date=n.date AND b.code=n.code
        )
        SELECT * FROM old_rows UNION ALL SELECT * FROM recent_rows
        ORDER BY symbol,trade_date
    """


def _write_archive_cache(
    ctx: PitHistoryContext,
    *,
    paths: dict[str, Path],
    symbols: Sequence[str],
    temporary: Path,
) -> None:
    recovery_names = str(paths["recovery_names"] / "year=*.parquet")
    targets = pd.DataFrame({"symbol": list(symbols)})
    with open_guarded_duckdb(temp_directory=ctx.runtime / "archive_cache_spill", threads=4) as con:
        con.register("targets", targets)
        escaped = str(temporary).replace("'", "''")
        con.execute(
            f"COPY ({_archive_cache_query()}) TO '{escaped}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)",
            [
                str(paths["recovery_bars"]),
                recovery_names,
                str(paths["recent_bars"]),
                str(paths["recent_metrics"]),
                str(paths["recent_universe"]),
                str(paths["recent_names"]),
            ],
        )


def _build_archive_cache(
    ctx: PitHistoryContext,
    *,
    symbols: Sequence[str],
) -> Path:
    output = ctx.runtime / "archive_history.parquet"
    metadata = ctx.runtime / "archive_history.json"
    requested_symbols = sorted(str(item) for item in symbols)
    if _archive_cache_is_current(output, metadata, requested_symbols):
        return output
    paths = _archive_paths(ctx)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        _write_archive_cache(ctx, paths=paths, symbols=symbols, temporary=temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
        shutil.rmtree(ctx.runtime / "archive_cache_spill", ignore_errors=True)
    atomic_write_json(
        metadata,
        {
            "semantic_version": PART_SEMANTIC_VERSION,
            "symbols": requested_symbols,
            "sources": {key: str(value) for key, value in paths.items()},
            "created_at": utc_now(),
        },
    )
    return output


def _download_market_supplements(
    ctx: PitHistoryContext,
    *,
    symbols: Sequence[str],
    workers: int,
) -> None:
    import akshare as ak

    output = ctx.runtime / "supplement_parts"
    output.mkdir(parents=True, exist_ok=True)
    required = ("trade_date", "symbol", "history_source")
    pending = [
        symbol for symbol in symbols if not _valid_parquet(output / f"{symbol.replace('.', '_')}.parquet", required)
    ]

    def fetch_one(symbol: str) -> str:
        raw = _external_with_retry(
            lambda: ak.stock_zh_a_hist(
                symbol=symbol.split(".", 1)[0],
                period="daily",
                start_date=ctx.start_date.replace("-", ""),
                end_date=ctx.end_date.replace("-", ""),
                adjust="",
            ),
            label=f"pit_history_eastmoney:{symbol}",
        )
        normalized = _normalize_eastmoney_history(raw, symbol=symbol)
        _atomic_parquet(normalized, output / f"{symbol.replace('.', '_')}.parquet")
        return symbol

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 3))) as pool:
        futures = {pool.submit(fetch_one, symbol): symbol for symbol in pending}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                future.result()
            except Exception as exc:
                raise PitHistoryError(
                    f"pit_history_market_supplement_failed:{symbol}:{type(exc).__name__}:{exc}"
                ) from exc
            completed += 1
            if completed == 1 or completed % 25 == 0 or completed == len(pending):
                print(f"pit_history_market={completed}/{len(pending)}", flush=True)


def _download_sina_factors(
    ctx: PitHistoryContext,
    *,
    symbols: Sequence[str],
    workers: int,
) -> None:
    import akshare as ak

    output = ctx.runtime / "factor_parts"
    output.mkdir(parents=True, exist_ok=True)
    required = ("symbol", "trade_date", "back_adjust_factor", "factor_provider")
    pending = [
        symbol for symbol in symbols if not _valid_parquet(output / f"{symbol.replace('.', '_')}.parquet", required)
    ]

    def fetch_one(symbol: str) -> str:
        code, suffix = symbol.split(".", 1)
        raw = _external_with_retry(
            lambda: ak.stock_zh_a_daily(
                symbol=("sh" if suffix == "SH" else "sz") + code,
                start_date="19900101",
                end_date=ctx.end_date.replace("-", ""),
                adjust="hfq-factor",
            ),
            label=f"pit_history_sina_factor:{symbol}",
        )
        normalized = _normalize_sina_factors(raw, symbol=symbol)
        _atomic_parquet(normalized, output / f"{symbol.replace('.', '_')}.parquet")
        return symbol

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 3))) as pool:
        futures = {pool.submit(fetch_one, symbol): symbol for symbol in pending}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                future.result()
            except Exception as exc:
                raise PitHistoryError(
                    f"pit_history_factor_supplement_failed:{symbol}:{type(exc).__name__}:{exc}"
                ) from exc
            completed += 1
            if completed == 1 or completed % 25 == 0 or completed == len(pending):
                print(f"pit_history_factors={completed}/{len(pending)}", flush=True)


def _trade_calendar(ctx: PitHistoryContext) -> pd.DataFrame:
    output = ctx.runtime / "trade_calendar.parquet"
    if _valid_parquet(output, ("trade_date",)):
        return pd.read_parquet(output)
    current = resolve_active_domain("universe_snapshot", workspace_root=ctx.workspace)
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "calendar_spill",
        threads=2,
    ) as con:
        frame = con.execute(
            "SELECT DISTINCT cast(trade_date AS DATE) trade_date "
            "FROM read_parquet(?, union_by_name=true) "
            "WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
            [[str(item) for item in current.shard_paths], ctx.start_date, ctx.end_date],
        ).fetchdf()
    shutil.rmtree(ctx.runtime / "calendar_spill", ignore_errors=True)
    _atomic_parquet(frame, output)
    return frame


def _history_observations(archive: pd.DataFrame, supplement: pd.DataFrame) -> pd.DataFrame:
    archive = archive.copy()
    supplement = supplement.copy()
    archive["trade_date"] = pd.to_datetime(archive["trade_date"], errors="coerce")
    supplement["trade_date"] = pd.to_datetime(supplement["trade_date"], errors="coerce")
    if not archive.empty and not supplement.empty:
        supplement_metrics = supplement.loc[:, ["trade_date", "turn", "pctChg"]].rename(
            columns={"turn": "turn_supplement", "pctChg": "pctChg_supplement"}
        )
        archive = archive.merge(supplement_metrics, on="trade_date", how="left")
        archive["turn"] = pd.to_numeric(archive["turn"], errors="coerce").fillna(
            pd.to_numeric(archive["turn_supplement"], errors="coerce")
        )
        archive["pctChg"] = pd.to_numeric(archive["pctChg"], errors="coerce").fillna(
            pd.to_numeric(archive["pctChg_supplement"], errors="coerce")
        )
        archive = archive.drop(columns=["turn_supplement", "pctChg_supplement"])
    archive["priority"] = 2
    supplement["priority"] = 1
    observations = (
        pd.concat([supplement, archive], ignore_index=True, sort=False)
        .sort_values(["trade_date", "priority"])
        .drop_duplicates("trade_date", keep="last")
    )
    observations["trade_date"] = pd.to_datetime(observations["trade_date"], errors="coerce")
    return observations


def _symbol_history_dates(
    *,
    ctx: PitHistoryContext,
    symbol: str,
    security: pd.Series,
    calendar_dates: pd.Series,
    observations: pd.DataFrame,
) -> tuple[pd.DataFrame, int | None]:
    list_date = pd.Timestamp(str(security["list_date"]))
    mask = calendar_dates.ge(max(pd.Timestamp(ctx.start_date), list_date))
    delist_date = str(security.get("delist_date", "") or "")
    if delist_date:
        mask &= calendar_dates.lt(pd.Timestamp(delist_date))
    dates = pd.DataFrame({"trade_date": calendar_dates.loc[mask].reset_index(drop=True)})
    history = dates.merge(observations, on="trade_date", how="left")
    first_observed = history["close"].first_valid_index()
    if first_observed is not None:
        history = history.loc[first_observed:].reset_index(drop=True)
    elif history.empty:
        raise PitHistoryError(f"pit_history_no_lifecycle_dates:{symbol}")
    return history, first_observed


def _normalize_history_rows(history: pd.DataFrame, *, symbol: str, first_observed: int | None) -> pd.DataFrame:
    history["symbol"] = symbol
    history["semantic_version"] = PART_SEMANTIC_VERSION
    history["tradestatus"] = history["tradestatus"].fillna("0").astype(str)
    observed_fields = history.loc[:, ["open", "high", "low", "close", "volume", "amount"]].apply(
        pd.to_numeric, errors="coerce"
    )
    observed = (
        history["tradestatus"].eq("1")
        & observed_fields.notna().all(axis=1)
        & observed_fields[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & observed_fields[["volume", "amount"]].ge(0).all(axis=1)
    )
    history.loc[~observed, "tradestatus"] = "0"
    history["close"] = pd.to_numeric(history["close"], errors="coerce").ffill()
    for column in ("open", "high", "low"):
        history[column] = pd.to_numeric(history[column], errors="coerce")
        missing = ~observed & history[column].isna()
        history.loc[missing, column] = history.loc[missing, "close"]
    for column in ("volume", "amount", "turn"):
        history[column] = pd.to_numeric(history[column], errors="coerce")
        history.loc[~observed & history[column].isna(), column] = 0.0
    for column in ("peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"):
        history[column] = pd.to_numeric(history.get(column), errors="coerce")
    history["preclose"] = history["close"].shift(1).fillna(history["close"])
    history["pctChg"] = pd.to_numeric(history.get("pctChg"), errors="coerce")
    history["pctChg"] = history["pctChg"].fillna((history["close"] / history["preclose"] - 1.0) * 100.0)
    history["isST"] = history.get("isST", "").fillna("").astype(str)
    history["name_on_date"] = history.get("name_on_date", "").fillna("").astype(str)
    default_source = (
        "calendar_suspension_inference_no_observed_bar"
        if first_observed is None
        else "calendar_suspension_inference"
    )
    history["history_source"] = history.get("history_source", "").fillna(default_source).replace("", default_source)
    history["provider_code"] = ("sh." if symbol.endswith(".SH") else "sz.") + symbol[:6]
    history["adjustflag"] = "3"
    history["trade_date"] = history["trade_date"].dt.strftime("%Y-%m-%d")
    return history.drop(columns=["priority"], errors="ignore")


def _materialize_history_parts(
    ctx: PitHistoryContext,
    *,
    symbols: Sequence[str],
    basic: pd.DataFrame,
    archive_cache: Path,
) -> None:
    output = ctx.runtime / "history_parts"
    output.mkdir(parents=True, exist_ok=True)
    calendar = _trade_calendar(ctx)
    calendar_dates = pd.to_datetime(calendar["trade_date"], errors="raise")
    basic_index = basic.set_index("symbol", drop=False)
    required = (
        "trade_date",
        "symbol",
        "tradestatus",
        "isST",
        "turn",
        "history_source",
        "semantic_version",
    )
    for index, symbol in enumerate(symbols, start=1):
        target = output / f"{symbol.replace('.', '_')}.parquet"
        if _valid_parquet(target, required):
            continue
        row = basic_index.loc[symbol]
        archive = pd.read_parquet(archive_cache, filters=[("symbol", "==", symbol)])
        supplement = pd.read_parquet(ctx.runtime / "supplement_parts" / f"{symbol.replace('.', '_')}.parquet")
        observations = _history_observations(archive, supplement)
        history, first_observed = _symbol_history_dates(
            ctx=ctx,
            symbol=symbol,
            security=row,
            calendar_dates=calendar_dates,
            observations=observations,
        )
        _atomic_parquet(_normalize_history_rows(history, symbol=symbol, first_observed=first_observed), target)
        if index == 1 or index % 25 == 0 or index == len(symbols):
            print(f"pit_history_materialize={index}/{len(symbols)}", flush=True)


def _fetch_tencent_daily_metrics(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    import requests
    from akshare.utils import demjson

    code, suffix = symbol.split(".", 1)
    provider_symbol = ("sh" if suffix == "SH" else "sz") + code
    rows: list[list[Any]] = []
    start_year = pd.Timestamp(start_date).year
    end_year = pd.Timestamp(end_date).year
    for year in range(start_year, end_year + 1):
        params = {
            "_var": f"kline_day{year}",
            "param": f"{provider_symbol},day,{year}-01-01,{year + 1}-12-31,640,",
            "r": "0.8205512681390605",
        }
        response = requests.get(
            "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = demjson.decode(response.text[response.text.find("={") + 1 :])
        security = dict(payload.get("data", {}).get(provider_symbol, {}) or {})
        rows.extend(list(security.get("day", []) or []))
    if not rows:
        return pd.DataFrame(columns=["trade_date", "volume", "amount", "turn"])
    usable = [row for row in rows if isinstance(row, list) and len(row) >= 9]
    data = pd.DataFrame(
        {
            "trade_date": [row[0] for row in usable],
            "volume": [row[5] for row in usable],
            "turn": [row[7] for row in usable],
            "amount": [row[8] for row in usable],
        }
    )
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["volume"] = pd.to_numeric(data["volume"], errors="coerce") * 100.0
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce") * 10_000.0
    data["turn"] = pd.to_numeric(data["turn"], errors="coerce")
    return (
        data.loc[data["trade_date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))]
        .drop_duplicates("trade_date", keep="last")
        .sort_values("trade_date")
        .reset_index(drop=True)
    )


def _fill_missing_turnover_from_tencent(
    ctx: PitHistoryContext,
    *,
    symbols: Sequence[str],
) -> None:
    history_paths = [ctx.runtime / "history_parts" / f"{symbol.replace('.', '_')}.parquet" for symbol in symbols]
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "missing_turnover_spill",
        threads=2,
    ) as con:
        missing = con.execute(
            "SELECT symbol,min(trade_date) first_date,max(trade_date) last_date,count(*) n "
            "FROM read_parquet(?, union_by_name=true) "
            "WHERE cast(tradestatus AS VARCHAR)='1' AND turn IS NULL "
            "GROUP BY symbol ORDER BY symbol",
            [[str(path) for path in history_paths]],
        ).fetchdf()
    shutil.rmtree(ctx.runtime / "missing_turnover_spill", ignore_errors=True)
    if missing.empty:
        return
    cache = ctx.runtime / "tencent_metric_parts"
    cache.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(missing.itertuples(index=False), start=1):
        symbol = str(row.symbol)
        token = symbol.replace(".", "_")
        metric_path = cache / f"{token}.parquet"
        if _valid_parquet(metric_path, ("trade_date", "volume", "amount", "turn")):
            metrics = pd.read_parquet(metric_path)
        else:
            metrics = _external_with_retry(
                lambda symbol=symbol, row=row: _fetch_tencent_daily_metrics(
                    symbol=symbol,
                    start_date=str(row.first_date)[:10],
                    end_date=str(row.last_date)[:10],
                ),
                label=f"pit_history_tencent_metrics:{symbol}",
            )
            _atomic_parquet(metrics, metric_path)
        history_path = ctx.runtime / "history_parts" / f"{token}.parquet"
        history = pd.read_parquet(history_path)
        history["trade_date"] = pd.to_datetime(history["trade_date"], errors="raise")
        metrics["trade_date"] = pd.to_datetime(metrics["trade_date"], errors="coerce")
        history = history.merge(
            metrics.rename(
                columns={
                    "turn": "turn_tencent",
                    "volume": "volume_tencent",
                    "amount": "amount_tencent",
                }
            ),
            on="trade_date",
            how="left",
        )
        history["turn"] = pd.to_numeric(history["turn"], errors="coerce").fillna(
            pd.to_numeric(history["turn_tencent"], errors="coerce")
        )
        history = history.drop(columns=["turn_tencent", "volume_tencent", "amount_tencent"])
        history["trade_date"] = history["trade_date"].dt.strftime("%Y-%m-%d")
        _atomic_parquet(history, history_path)
        print(f"pit_history_tencent_turnover={index}/{len(missing)}:{symbol}", flush=True)


def _load_sz_name_events(ctx: PitHistoryContext) -> pd.DataFrame:
    path = ctx.runtime / "sz_name_events.parquet"
    required = ("symbol", "trade_date", "old_name", "new_name")
    if _valid_parquet(path, required):
        return pd.read_parquet(path)
    import akshare as ak

    raw = _external_with_retry(
        lambda: ak.stock_info_sz_change_name(symbol="简称变更"),
        label="pit_history_sz_name_change",
    )
    data = raw.rename(
        columns={
            "变更日期": "trade_date",
            "证券代码": "code",
            "变更前简称": "old_name",
            "变更后简称": "new_name",
        }
    ).copy()
    for column in ("trade_date", "code", "old_name", "new_name"):
        if column not in data:
            data[column] = ""
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    data["symbol"] = data["code"].fillna("").astype(str).str.zfill(6) + ".SZ"
    data["old_name"] = data["old_name"].fillna("").astype(str).str.strip()
    data["new_name"] = data["new_name"].fillna("").astype(str).str.strip()
    data = (
        data.loc[
            data["symbol"].str.match(r"^\d{6}\.SZ$", na=False) & data["trade_date"].notna() & data["new_name"].ne(""),
            list(required),
        ]
        .drop_duplicates(["symbol", "trade_date"], keep="last")
        .sort_values(["symbol", "trade_date"])
        .reset_index(drop=True)
    )
    _atomic_parquet(data, path)
    return data


def _name_intervals_from_events(
    symbol: str,
    events: pd.DataFrame,
) -> pd.DataFrame:
    selected = events.loc[events["symbol"].eq(symbol)].copy()
    if selected.empty:
        return pd.DataFrame(columns=list(NAME_INTERVAL_COLUMNS))
    selected = selected.sort_values("trade_date").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    first = selected.iloc[0]
    first_old_name = str(first["old_name"]).strip() if pd.notna(first["old_name"]) else ""
    if first_old_name:
        rows.append(
            {
                "symbol": symbol,
                "name": first_old_name,
                "start_date": "1900-01-01",
                "end_date": (pd.Timestamp(first["trade_date"]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "announcement_date": str(first["trade_date"]),
                "change_reason": "exchange_short_name_history",
                "source": "akshare_szse_short_name_change",
            }
        )
    for index, row in selected.iterrows():
        next_start = pd.Timestamp(selected.iloc[index + 1]["trade_date"]) if index + 1 < len(selected) else None
        rows.append(
            {
                "symbol": symbol,
                "name": str(row["new_name"]).strip(),
                "start_date": str(row["trade_date"]),
                "end_date": (
                    (next_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d") if next_start is not None else ""
                ),
                "announcement_date": str(row["trade_date"]),
                "change_reason": "exchange_short_name_history",
                "source": "akshare_szse_short_name_change",
            }
        )
    return pd.DataFrame(rows, columns=list(NAME_INTERVAL_COLUMNS))


def _download_reference_parts(
    ctx: PitHistoryContext,
    *,
    symbols: Sequence[str],
    workers: int,
) -> None:
    import akshare as ak

    share_output = ctx.runtime / "share_event_parts"
    name_output = ctx.runtime / "name_interval_parts"
    share_output.mkdir(parents=True, exist_ok=True)
    name_output.mkdir(parents=True, exist_ok=True)
    sz_events = _load_sz_name_events(ctx)

    def fetch_one(symbol: str) -> str:
        token_name = symbol.replace(".", "_")
        share_path = share_output / f"{token_name}.parquet"
        if not _valid_parquet(
            share_path,
            ("symbol", "variation_date", "source_date", "total_share", "float_share"),
        ):
            code = symbol.split(".", 1)[0]
            try:
                raw = _external_with_retry(
                    lambda: ak.stock_share_change_cninfo(
                        symbol=code,
                        start_date="19900101",
                        end_date=ctx.end_date.replace("-", ""),
                    ),
                    label=f"pit_history_share_change:{symbol}",
                )
                normalized = _normalize_cninfo_share_change(
                    raw,
                    symbol=symbol,
                    target_date=ctx.end_date,
                    source="akshare_cninfo_share_history_pit_restore",
                )
            except Exception:
                normalized = pd.DataFrame(columns=list(CNINFO_SHARE_NORMALIZED_COLUMNS))
            _atomic_parquet(normalized, share_path)
        name_path = name_output / f"{token_name}.parquet"
        if not _valid_parquet(name_path, ("symbol", "name", "start_date")):
            normalized = _name_intervals_from_events(symbol, sz_events)
            _atomic_parquet(normalized, name_path)
        return symbol

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 3))) as pool:
        futures = {pool.submit(fetch_one, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                future.result()
            except Exception as exc:
                raise PitHistoryError(f"pit_reference_partition_failed:{symbol}:{type(exc).__name__}:{exc}") from exc
            completed += 1
            if completed == 1 or completed % 25 == 0 or completed == len(symbols):
                print(f"pit_history_reference={completed}/{len(symbols)}", flush=True)


def _load_sse_st_transitions() -> pd.DataFrame:
    if not SSE_ST_TRANSITIONS.is_file():
        raise PitHistoryError(f"pit_history_sse_status_resource_missing:{SSE_ST_TRANSITIONS}")
    frame = pd.read_csv(SSE_ST_TRANSITIONS, dtype=str)
    frame["effective_date"] = pd.to_datetime(frame["effective_date"], errors="raise")
    frame["is_st"] = frame["is_st"].str.lower().map({"true": True, "false": False})
    if frame["is_st"].isna().any():
        raise PitHistoryError("pit_history_sse_status_resource_invalid_boolean")
    return frame.sort_values(["symbol", "effective_date"]).reset_index(drop=True)


def _historical_st_status(
    history: pd.DataFrame,
    *,
    symbol: str,
    names: pd.Series,
) -> pd.Series:
    raw = history["isST"].fillna("").astype(str).str.strip().str.lower()
    result = pd.Series(pd.NA, index=history.index, dtype="boolean")
    result.loc[raw.isin({"1", "true"})] = True
    result.loc[raw.isin({"0", "false"})] = False
    dates = pd.to_datetime(history["trade_date"], errors="raise")
    if symbol.endswith(".SH"):
        events = _load_sse_st_transitions()
        events = events.loc[events["symbol"].eq(symbol)].copy()
        if not events.empty:
            initial = not bool(events.iloc[0]["is_st"])
            inferred = pd.Series(initial, index=history.index, dtype="boolean")
            for event in events.itertuples(index=False):
                inferred.loc[dates.ge(pd.Timestamp(event.effective_date))] = bool(event.is_st)
            pre_archive = dates.lt(pd.Timestamp("2012-01-04"))
            result.loc[pre_archive] = result.loc[pre_archive].fillna(inferred.loc[pre_archive])
        elif result.notna().any():
            first_known = bool(result.loc[result.notna()].iloc[0])
            result.loc[dates.lt(dates.loc[result.notna()].iloc[0])] = first_known
    # Shenzhen's dated exchange name intervals carry the exact ST prefix. The
    # name fallback also covers archive-edge current rows after 2026-06-01.
    result = result.fillna(_name_implies_st(names).astype("boolean"))
    return result.astype("boolean")
