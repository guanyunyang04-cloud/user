from __future__ import annotations

import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.providers import BaostockProvider
from quant_data_platform.qdp_v2.auxiliary_update import (
    CNINFO_SHARE_NORMALIZED_COLUMNS,
    NAME_INTERVAL_COLUMNS,
    _external_with_retry,
    _normalize_cninfo_share_change,
)
from quant_data_platform.qdp_v2.duckdb_resources import open_guarded_duckdb
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    atomic_write_json,
    canonical_manifest_sha256,
    path_for_manifest,
    qdp_v2_root,
    read_active_manifest,
    stable_hash,
    utc_now,
    write_active_manifest,
    write_dataset_manifest,
)
from quant_data_platform.qdp_v2.repair import (
    mutate_active_shards_from_parquet,
    resolve_active_domain,
    update_active_manifest_metadata,
)


DEFAULT_START_DATE = "2010-01-04"
PART_SEMANTIC_VERSION = "pit_historical_mainboard_daily_v5"
MAINBOARD_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")
ARCHIVE_ROOT = Path(
    "daily_research/data/research_store/traditional_quant_baostock_archive_v1/raw"
)
ARCHIVE_RECENT = ARCHIVE_ROOT / "baostock_daily_mainboard_v2_pit"
ARCHIVE_RECOVERY = (
    ARCHIVE_ROOT / "baostock_daily_mainboard_v2_pit_recovery_2012_2015"
)
SSE_ST_TRANSITIONS = Path(__file__).with_name("resources") / "sse_st_transitions_2010_2011.csv"
SSE_FACTBOOK_EVIDENCE = {
    "2011": {
        "url": "https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170571/files/f43f33c247f242d48780c3097548281b.pdf",
        "sha256": "77E4D1F8024F47456A608C119F2E634E3936F5AE1C245F4AEFA45BA59F0D0BBE",
    },
    "2012": {
        "url": "https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170570/files/9a7b8e0d00e84d358d02fbba056f7bda.pdf",
        "sha256": "D1283B557D51B298300B3CB46C1125AABEAC0AE86E94CEA7ECB15DE9BEFFF0BB",
    },
}
PRE_ARCHIVE_SECURITIES = (
    ("000578.SZ", "盐湖集团", "1995-03-03", "2011-03-22"),
    ("600003.SH", "ST东北高", "1999-08-10", "2010-02-26"),
    ("600553.SH", "太行水泥", "2002-08-22", "2011-02-18"),
    ("600591.SH", "*ST上航", "2002-10-11", "2010-01-25"),
    ("600607.SH", "上实医药", "1992-03-27", "2010-02-12"),
    ("600631.SH", "百联股份", "1993-02-19", "2011-08-23"),
    ("600842.SH", "中西药业", "1994-03-11", "2010-02-12"),
    ("600849.SH", "上药转换", "1994-03-24", "2010-03-05"),
)
RESTORE_DOMAINS = (
    "security_identity",
    "symbol_history",
    "market_daily_raw",
    "universe_snapshot",
    "security_status",
    "adjust_factor",
    "industry_concept",
    "share_capital",
    "valuation",
    "name_change",
)

# Code changes are a ticker lifecycle event, not a new economic security.  The
# daily research domains below must use the effective ticker for each date.  The
# 5-minute table is intentionally excluded: historical 5m restoration is not a
# requirement for the daily-only research contract.
LIFECYCLE_NORMALIZE_DOMAINS = (
    "market_daily_raw",
    "universe_snapshot",
    "security_status",
    "adjust_factor",
    "industry_concept",
    "share_capital",
    "valuation",
    "name_change",
    "corporate_actions",
)


class PitHistoryError(RuntimeError):
    pass


@dataclass(frozen=True)
class PitHistoryContext:
    workspace: Path
    root: Path
    runtime: Path
    start_date: str
    end_date: str


def _context(
    *,
    start_date: str,
    end_date: str,
    workspace_root: str | Path | None,
) -> PitHistoryContext:
    workspace = Path(workspace_root or Path.cwd()).resolve()
    root = qdp_v2_root(workspace).resolve()
    runtime = (
        qdp_paths(workspace).data_dir / "qdp_runtime" / "pit_history_restore"
    ).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    if start > end:
        raise ValueError(f"pit_history_invalid_range:{start}>{end}")
    return PitHistoryContext(workspace, root, runtime, start, end)


def _is_mainboard(symbol: object) -> bool:
    text = str(symbol or "").strip().upper()
    code = text.split(".", 1)[0]
    return text.endswith((".SH", ".SZ")) and code.startswith(MAINBOARD_PREFIXES)


def _security_id(symbol: str) -> str:
    code, exchange = str(symbol).upper().split(".", 1)
    venue = "SSE" if exchange == "SH" else "SZSE"
    return f"QDP-CN-{venue}-{code}"


def _identity_exchange(symbol: str) -> str:
    return "SSE" if str(symbol).upper().endswith(".SH") else "SZSE"


def _short_exchange(symbol: str) -> str:
    return "SH" if str(symbol).upper().endswith(".SH") else "SZ"


def _read_active_symbols(ctx: PitHistoryContext, domain: str) -> set[str]:
    current = resolve_active_domain(domain, workspace_root=ctx.workspace)
    column = "current_symbol" if domain == "security_identity" else "symbol"
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "symbol_inventory_spill",
        threads=2,
    ) as con:
        frame = con.execute(
            f"SELECT DISTINCT cast({column} AS VARCHAR) AS symbol "
            "FROM read_parquet(?, union_by_name=true)",
            [[str(item) for item in current.shard_paths]],
        ).fetchdf()
    shutil.rmtree(ctx.runtime / "symbol_inventory_spill", ignore_errors=True)
    return set(frame["symbol"].astype(str).str.upper())


def _read_symbol_history_symbols(ctx: PitHistoryContext) -> set[str]:
    current = resolve_active_domain("symbol_history", workspace_root=ctx.workspace)
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "symbol_history_inventory_spill",
        threads=2,
    ) as con:
        frame = con.execute(
            "SELECT DISTINCT upper(cast(symbol AS VARCHAR)) AS symbol "
            "FROM read_parquet(?, union_by_name=true)",
            [[str(item) for item in current.shard_paths]],
        ).fetchdf()
    shutil.rmtree(
        ctx.runtime / "symbol_history_inventory_spill",
        ignore_errors=True,
    )
    return set(frame["symbol"].astype(str))


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _valid_parquet(path: Path, required: Sequence[str]) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        import pyarrow.parquet as pq

        return set(required).issubset(pq.read_schema(path).names)
    except Exception:
        return False


def _archive_paths(ctx: PitHistoryContext) -> dict[str, Path]:
    base = ctx.workspace / ARCHIVE_ROOT
    recent_root = base / ARCHIVE_RECENT.name
    latest_path = recent_root / "latest_manifest.json"
    if not latest_path.is_file():
        raise PitHistoryError(f"pit_history_archive_manifest_missing:{latest_path}")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    snapshot_id = str(latest.get("snapshot_id", "")).strip()
    recent = recent_root / snapshot_id
    recovery = base / ARCHIVE_RECOVERY.name
    required = {
        "recent_bars": recent / "daily_bars.parquet",
        "recent_metrics": recent / "daily_metrics.parquet",
        "recent_universe": recent / "daily_universe.parquet",
        "recent_names": recent / "raw_daily_stock_lists.parquet",
        "recent_master": recent / "security_master.parquet",
        "recovery_bars": recovery / "daily_bars.parquet",
        "recovery_names": recovery / "cache" / "daily_stock_lists",
        "recovery_master": recovery / "cache" / "security_master.parquet",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise PitHistoryError(f"pit_history_archive_incomplete:{missing}")
    return required


def _local_stock_basic(ctx: PitHistoryContext) -> pd.DataFrame:
    paths = _archive_paths(ctx)
    rows: list[pd.DataFrame] = []
    for key in ("recovery_master", "recent_master"):
        frame = pd.read_parquet(paths[key]).copy()
        frame = frame.loc[frame.get("security_type", "1").astype(str).eq("1")]
        rows.append(
            pd.DataFrame(
                {
                    "symbol": frame["code"].astype(str).str.upper(),
                    "name": frame.get("name", "").fillna("").astype(str),
                    "list_date": frame.get("ipo_date", "").fillna("").astype(str),
                    "delist_date": frame.get("out_date", "").fillna("").astype(str),
                    "priority": 1 if key == "recovery_master" else 2,
                }
            )
        )
    try:
        current = resolve_active_domain("security_identity", workspace_root=ctx.workspace)
        with open_guarded_duckdb(
            temp_directory=ctx.runtime / "stock_basic_spill",
            threads=2,
        ) as con:
            identity = con.execute(
                "SELECT cast(current_symbol AS VARCHAR) symbol, "
                "cast(issuer_name AS VARCHAR) name, cast(list_date AS VARCHAR) list_date "
                "FROM read_parquet(?, union_by_name=true)",
                [[str(item) for item in current.shard_paths]],
            ).fetchdf()
        shutil.rmtree(ctx.runtime / "stock_basic_spill", ignore_errors=True)
        identity["delist_date"] = ""
        identity["priority"] = 3
        rows.append(identity)
    except Exception:
        shutil.rmtree(ctx.runtime / "stock_basic_spill", ignore_errors=True)
    rows.append(
        pd.DataFrame(
            PRE_ARCHIVE_SECURITIES,
            columns=["symbol", "name", "list_date", "delist_date"],
        ).assign(priority=4)
    )
    combined = pd.concat(rows, ignore_index=True)
    combined["symbol"] = combined["symbol"].astype(str).str.upper()
    combined = combined.loc[combined["symbol"].map(_is_mainboard)].copy()
    combined = combined.sort_values(["symbol", "priority"])

    def last_text(values: pd.Series) -> str:
        valid = [str(item).strip() for item in values if str(item).strip() not in {"", "nan", "NaT"}]
        return valid[-1] if valid else ""

    result = (
        combined.groupby("symbol", as_index=False)
        .agg(
            name=("name", last_text),
            list_date=("list_date", lambda values: min(
                [str(item)[:10] for item in values if str(item).strip() not in {"", "nan", "NaT"}],
                default="",
            )),
            delist_date=("delist_date", last_text),
        )
        .sort_values("symbol")
        .reset_index(drop=True)
    )
    result["type"] = "1"
    result["status"] = np.where(result["delist_date"].ne(""), "0", "1")
    result["list_status"] = result["status"]
    result["trade_date"] = ctx.end_date
    result["board"] = "1"
    result["is_st"] = result["name"].str.upper().str.contains("ST", regex=False)
    result["is_suspended"] = False
    result["is_delisted"] = result["delist_date"].ne("")
    result["status_reason"] = ""
    return result


def _stock_basic(ctx: PitHistoryContext, provider: BaostockProvider) -> pd.DataFrame:
    path = ctx.runtime / "stock_basic.parquet"
    if _valid_parquet(path, ("symbol", "list_date", "delist_date")):
        frame = pd.read_parquet(path)
    else:
        frame = _local_stock_basic(ctx)
        if frame.empty:
            frame = provider.fetch_stock_basic_snapshot(trade_date=ctx.end_date).copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame = frame.loc[frame["symbol"].map(_is_mainboard)].copy()
    frame["list_date"] = pd.to_datetime(frame["list_date"], errors="coerce")
    frame["delist_date"] = pd.to_datetime(frame["delist_date"], errors="coerce")
    frame = frame.loc[
        frame["list_date"].notna()
        & frame["list_date"].le(ctx.end_date)
        & (frame["delist_date"].isna() | frame["delist_date"].ge(ctx.start_date))
    ].copy()
    frame["list_date"] = frame["list_date"].dt.strftime("%Y-%m-%d")
    frame["delist_date"] = frame["delist_date"].dt.strftime("%Y-%m-%d").fillna("")
    frame = frame.drop_duplicates("symbol", keep="last").sort_values("symbol")
    _atomic_parquet(frame, path)
    return frame.reset_index(drop=True)


def inventory_pit_history(
    *,
    start_date: str = DEFAULT_START_DATE,
    end_date: str,
    workspace_root: str | Path | None = None,
    provider: BaostockProvider | None = None,
) -> dict[str, Any]:
    ctx = _context(
        start_date=start_date,
        end_date=end_date,
        workspace_root=workspace_root,
    )
    source = provider or BaostockProvider()
    owned = provider is None
    try:
        basic = _stock_basic(ctx, source)
    finally:
        if owned:
            source.close()
    active_daily = _read_active_symbols(ctx, "market_daily_raw")
    active_lifecycle = _read_active_symbols(ctx, "universe_snapshot")
    missing = sorted(set(basic["symbol"].astype(str)) - active_lifecycle)
    missing_frame = basic.loc[basic["symbol"].isin(missing)].copy()
    payload = {
        "status": "planned" if missing else "already_complete",
        "start_date": ctx.start_date,
        "end_date": ctx.end_date,
        "pit_mainboard_symbol_count": int(len(basic)),
        "active_daily_symbol_count": int(len(active_daily)),
        "active_lifecycle_symbol_count": int(len(active_lifecycle)),
        "missing_symbol_count": int(len(missing)),
        "missing_delisted_symbol_count": int(
            missing_frame["delist_date"].astype(str).ne("").sum()
        ),
        "missing_current_or_st_symbol_count": int(
            missing_frame["delist_date"].astype(str).eq("").sum()
        ),
        "missing_symbols": missing,
        "runtime": str(ctx.runtime),
    }
    atomic_write_json(ctx.runtime / "inventory.json", payload)
    return payload


def _build_archive_cache(
    ctx: PitHistoryContext,
    *,
    symbols: Sequence[str],
) -> Path:
    output = ctx.runtime / "archive_history.parquet"
    metadata = ctx.runtime / "archive_history.json"
    identity = stable_hash(
        {
            "semantic_version": PART_SEMANTIC_VERSION,
            "symbols": sorted(str(item) for item in symbols),
        }
    )
    if _valid_parquet(
        output,
        ("trade_date", "symbol", "tradestatus", "isST", "history_source"),
    ) and metadata.is_file():
        try:
            if json.loads(metadata.read_text(encoding="utf-8")).get("identity") == identity:
                return output
        except (OSError, ValueError, TypeError):
            pass
    paths = _archive_paths(ctx)
    targets = pd.DataFrame({"symbol": list(symbols)})
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    recovery_names = str(paths["recovery_names"] / "year=*.parquet")
    query = """
        WITH old_rows AS (
            SELECT
                cast(b.date AS DATE) AS trade_date,
                cast(b.code AS VARCHAR) AS symbol,
                cast(b.open AS DOUBLE) AS open,
                cast(b.high AS DOUBLE) AS high,
                cast(b.low AS DOUBLE) AS low,
                cast(b.close AS DOUBLE) AS close,
                cast(b.volume AS DOUBLE) AS volume,
                cast(b.amount AS DOUBLE) AS amount,
                cast(b.tradestatus AS VARCHAR) AS tradestatus,
                cast(b.isST AS VARCHAR) AS isST,
                NULL::DOUBLE AS turn,
                NULL::DOUBLE AS pctChg,
                NULL::DOUBLE AS peTTM,
                NULL::DOUBLE AS pbMRQ,
                NULL::DOUBLE AS psTTM,
                NULL::DOUBLE AS pcfNcfTTM,
                cast(n.name_on_date AS VARCHAR) AS name_on_date,
                'protected_baostock_archive_2012_2015' AS history_source
            FROM read_parquet(?) b
            JOIN targets t ON cast(b.code AS VARCHAR)=t.symbol
            LEFT JOIN read_parquet(?, union_by_name=true) n
              ON b.date=n.date AND b.code=n.code
        ), recent_rows AS (
            SELECT
                cast(b.date AS DATE) AS trade_date,
                cast(b.code AS VARCHAR) AS symbol,
                cast(b.open AS DOUBLE) AS open,
                cast(b.high AS DOUBLE) AS high,
                cast(b.low AS DOUBLE) AS low,
                cast(b.close AS DOUBLE) AS close,
                cast(b.volume AS DOUBLE) AS volume,
                cast(b.amount AS DOUBLE) AS amount,
                cast(b.tradestatus AS VARCHAR) AS tradestatus,
                cast(b.isST AS VARCHAR) AS isST,
                cast(m.turn AS DOUBLE) AS turn,
                cast(m.pctChg AS DOUBLE) AS pctChg,
                cast(m.peTTM AS DOUBLE) AS peTTM,
                cast(m.pbMRQ AS DOUBLE) AS pbMRQ,
                cast(m.psTTM AS DOUBLE) AS psTTM,
                cast(m.pcfNcfTTM AS DOUBLE) AS pcfNcfTTM,
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
    try:
        with open_guarded_duckdb(
            temp_directory=ctx.runtime / "archive_cache_spill",
            threads=4,
        ) as con:
            con.register("targets", targets)
            escaped = str(temporary).replace("'", "''")
            con.execute(
                f"COPY ({query}) TO '{escaped}' "
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
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
        shutil.rmtree(ctx.runtime / "archive_cache_spill", ignore_errors=True)
    atomic_write_json(
        metadata,
        {
            "identity": identity,
            "semantic_version": PART_SEMANTIC_VERSION,
            "symbols": len(symbols),
            "sources": {key: str(value) for key, value in paths.items()},
            "created_at": utc_now(),
        },
    )
    return output


def _normalize_eastmoney_history(raw: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    columns = [
        "trade_date", "symbol", "open", "high", "low", "close", "volume",
        "amount", "tradestatus", "isST", "turn", "pctChg", "peTTM",
        "pbMRQ", "psTTM", "pcfNcfTTM", "name_on_date", "history_source",
    ]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=columns)
    data = raw.rename(
        columns={
            "日期": "trade_date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turn",
            "涨跌幅": "pctChg",
        }
    ).copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["symbol"] = symbol
    for column in ("open", "high", "low", "close", "volume", "amount", "turn", "pctChg"):
        data[column] = pd.to_numeric(data.get(column), errors="coerce")
    # Eastmoney reports A-share daily volume in lots; QDP and BaoStock use shares.
    data["volume"] = data["volume"] * 100.0
    data["tradestatus"] = "1"
    data["isST"] = ""
    for column in ("peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"):
        data[column] = np.nan
    data["name_on_date"] = ""
    data["history_source"] = "akshare_eastmoney_unadjusted_history"
    return (
        data.loc[data["trade_date"].notna(), columns]
        .drop_duplicates("trade_date", keep="last")
        .sort_values("trade_date")
        .reset_index(drop=True)
    )


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
        symbol for symbol in symbols
        if not _valid_parquet(output / f"{symbol.replace('.', '_')}.parquet", required)
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
                    f"pit_history_market_supplement_failed:{symbol}:"
                    f"{type(exc).__name__}:{exc}"
                ) from exc
            completed += 1
            if completed == 1 or completed % 25 == 0 or completed == len(pending):
                print(f"pit_history_market={completed}/{len(pending)}", flush=True)


def _normalize_sina_factors(raw: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    columns = [
        "symbol", "trade_date", "fore_adjust_factor", "back_adjust_factor",
        "adjust_factor", "factor_provider", "source",
    ]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=columns)
    data = raw.rename(columns={"date": "trade_date", "hfq_factor": "factor"}).copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["factor"] = pd.to_numeric(data["factor"], errors="coerce")
    data = data.loc[data["trade_date"].notna() & data["factor"].gt(0)].copy()
    data["symbol"] = symbol
    data["fore_adjust_factor"] = data["factor"]
    data["back_adjust_factor"] = data["factor"]
    data["adjust_factor"] = data["factor"]
    data["factor_provider"] = "sina_via_akshare"
    data["source"] = "akshare_sina_hfq_factor_event"
    return data.loc[:, columns].sort_values("trade_date").reset_index(drop=True)


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
        symbol for symbol in symbols
        if not _valid_parquet(output / f"{symbol.replace('.', '_')}.parquet", required)
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
                    f"pit_history_factor_supplement_failed:{symbol}:"
                    f"{type(exc).__name__}:{exc}"
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
        "trade_date", "symbol", "tradestatus", "isST", "turn", "history_source",
        "semantic_version",
    )
    for index, symbol in enumerate(symbols, start=1):
        target = output / f"{symbol.replace('.', '_')}.parquet"
        if _valid_parquet(target, required):
            continue
        row = basic_index.loc[symbol]
        archive = pd.read_parquet(archive_cache, filters=[("symbol", "==", symbol)])
        supplement = pd.read_parquet(
            ctx.runtime / "supplement_parts" / f"{symbol.replace('.', '_')}.parquet"
        )
        archive["trade_date"] = pd.to_datetime(archive["trade_date"], errors="coerce")
        supplement["trade_date"] = pd.to_datetime(
            supplement["trade_date"], errors="coerce"
        )
        if not archive.empty and not supplement.empty:
            supplement_metrics = supplement.loc[
                :, ["trade_date", "turn", "pctChg"]
            ].rename(columns={"turn": "turn_supplement", "pctChg": "pctChg_supplement"})
            archive = archive.merge(supplement_metrics, on="trade_date", how="left")
            archive["turn"] = pd.to_numeric(archive["turn"], errors="coerce").fillna(
                pd.to_numeric(archive["turn_supplement"], errors="coerce")
            )
            archive["pctChg"] = pd.to_numeric(
                archive["pctChg"], errors="coerce"
            ).fillna(pd.to_numeric(archive["pctChg_supplement"], errors="coerce"))
            archive = archive.drop(
                columns=["turn_supplement", "pctChg_supplement"]
            )
        archive["priority"] = 2
        supplement["priority"] = 1
        observations = (
            pd.concat([supplement, archive], ignore_index=True, sort=False)
            .sort_values(["trade_date", "priority"])
            .drop_duplicates("trade_date", keep="last")
        )
        observations["trade_date"] = pd.to_datetime(
            observations["trade_date"], errors="coerce"
        )
        list_date = pd.Timestamp(str(row["list_date"]))
        mask = calendar_dates.ge(max(pd.Timestamp(ctx.start_date), list_date))
        delist_date = str(row.get("delist_date", "") or "")
        if delist_date:
            mask &= calendar_dates.lt(pd.Timestamp(delist_date))
        dates = pd.DataFrame({"trade_date": calendar_dates.loc[mask].reset_index(drop=True)})
        history = dates.merge(observations, on="trade_date", how="left")
        first_observed = history["close"].first_valid_index()
        if first_observed is not None:
            history = history.loc[first_observed:].reset_index(drop=True)
        elif history.empty:
            raise PitHistoryError(f"pit_history_no_lifecycle_dates:{symbol}")
        history["symbol"] = symbol
        history["semantic_version"] = PART_SEMANTIC_VERSION
        history["tradestatus"] = history["tradestatus"].fillna("0").astype(str)
        observed_fields = history.loc[
            :, ["open", "high", "low", "close", "volume", "amount"]
        ].apply(pd.to_numeric, errors="coerce")
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
            history.loc[~observed & history[column].isna(), column] = history.loc[
                ~observed & history[column].isna(), "close"
            ]
        for column in ("volume", "amount", "turn"):
            history[column] = pd.to_numeric(history[column], errors="coerce")
            history.loc[~observed & history[column].isna(), column] = 0.0
        for column in ("peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"):
            history[column] = pd.to_numeric(history.get(column), errors="coerce")
        history["preclose"] = history["close"].shift(1).fillna(history["close"])
        history["pctChg"] = pd.to_numeric(history.get("pctChg"), errors="coerce")
        history["pctChg"] = history["pctChg"].fillna(
            (history["close"] / history["preclose"] - 1.0) * 100.0
        )
        history["isST"] = history.get("isST", "").fillna("").astype(str)
        history["name_on_date"] = history.get("name_on_date", "").fillna("").astype(str)
        default_source = (
            "calendar_suspension_inference_no_observed_bar"
            if first_observed is None
            else "calendar_suspension_inference"
        )
        history["history_source"] = history.get("history_source", "").fillna(
            default_source
        ).replace("", default_source)
        history["provider_code"] = (
            ("sh." if symbol.endswith(".SH") else "sz.") + symbol[:6]
        )
        history["adjustflag"] = "3"
        history["trade_date"] = history["trade_date"].dt.strftime("%Y-%m-%d")
        _atomic_parquet(history.drop(columns=["priority"], errors="ignore"), target)
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
        data.loc[
            data["trade_date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
        ]
        .drop_duplicates("trade_date", keep="last")
        .sort_values("trade_date")
        .reset_index(drop=True)
    )


def _fill_missing_turnover_from_tencent(
    ctx: PitHistoryContext,
    *,
    symbols: Sequence[str],
) -> None:
    history_paths = [
        ctx.runtime / "history_parts" / f"{symbol.replace('.', '_')}.parquet"
        for symbol in symbols
    ]
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
                lambda: _fetch_tencent_daily_metrics(
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
        history = history.drop(
            columns=["turn_tencent", "volume_tencent", "amount_tencent"]
        )
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
    data["trade_date"] = pd.to_datetime(
        data["trade_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    data["symbol"] = data["code"].fillna("").astype(str).str.zfill(6) + ".SZ"
    data["old_name"] = data["old_name"].fillna("").astype(str).str.strip()
    data["new_name"] = data["new_name"].fillna("").astype(str).str.strip()
    data = (
        data.loc[
            data["symbol"].str.match(r"^\d{6}\.SZ$", na=False)
            & data["trade_date"].notna()
            & data["new_name"].ne(""),
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
    first_old_name = (
        str(first["old_name"]).strip() if pd.notna(first["old_name"]) else ""
    )
    if first_old_name:
        rows.append(
            {
                "symbol": symbol,
                "name": first_old_name,
                "start_date": "1900-01-01",
                "end_date": (
                    pd.Timestamp(first["trade_date"]) - pd.Timedelta(days=1)
                ).strftime("%Y-%m-%d"),
                "announcement_date": str(first["trade_date"]),
                "change_reason": "exchange_short_name_history",
                "source": "akshare_szse_short_name_change",
            }
        )
    for index, row in selected.iterrows():
        next_start = (
            pd.Timestamp(selected.iloc[index + 1]["trade_date"])
            if index + 1 < len(selected)
            else None
        )
        rows.append(
            {
                "symbol": symbol,
                "name": str(row["new_name"]).strip(),
                "start_date": str(row["trade_date"]),
                "end_date": (
                    (next_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    if next_start is not None
                    else ""
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
                normalized = pd.DataFrame(
                    columns=list(CNINFO_SHARE_NORMALIZED_COLUMNS)
                )
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
                raise PitHistoryError(
                    f"pit_reference_partition_failed:{symbol}:"
                    f"{type(exc).__name__}:{exc}"
                ) from exc
            completed += 1
            if completed == 1 or completed % 25 == 0 or completed == len(symbols):
                print(f"pit_history_reference={completed}/{len(symbols)}", flush=True)


def _historical_names(
    dates: pd.Series,
    intervals: pd.DataFrame,
    *,
    fallback: str,
) -> pd.Series:
    result = pd.Series(str(fallback), index=dates.index, dtype="object")
    if intervals.empty:
        return result
    normalized = intervals.copy()
    normalized["start_date"] = pd.to_datetime(normalized["start_date"], errors="coerce")
    normalized["end_date"] = pd.to_datetime(normalized["end_date"], errors="coerce")
    normalized = normalized.dropna(subset=["start_date"]).sort_values("start_date")
    date_values = pd.to_datetime(dates, errors="coerce")
    for row in normalized.itertuples(index=False):
        start = pd.Timestamp(row.start_date)
        end = pd.Timestamp(row.end_date) if pd.notna(row.end_date) else pd.Timestamp.max
        mask = date_values.ge(start) & date_values.le(end)
        result.loc[mask] = str(row.name)
    return result


def _load_sse_st_transitions() -> pd.DataFrame:
    if not SSE_ST_TRANSITIONS.is_file():
        raise PitHistoryError(f"pit_history_sse_status_resource_missing:{SSE_ST_TRANSITIONS}")
    frame = pd.read_csv(SSE_ST_TRANSITIONS, dtype=str)
    frame["effective_date"] = pd.to_datetime(frame["effective_date"], errors="raise")
    frame["is_st"] = frame["is_st"].str.lower().map({"true": True, "false": False})
    if frame["is_st"].isna().any():
        raise PitHistoryError("pit_history_sse_status_resource_invalid_boolean")
    return frame.sort_values(["symbol", "effective_date"]).reset_index(drop=True)


def _name_implies_st(names: pd.Series) -> pd.Series:
    return names.fillna("").astype(str).str.upper().str.contains("ST", regex=False)


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
    return result.fillna(False).astype(bool)


def _factor_rows(history: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(history["trade_date"], errors="raise")
    event = events.copy()
    factor_provider = "identity_no_factor_event"
    factor_source = "identity_factor_pit_history_restore"
    if event.empty:
        factor = np.ones(len(history), dtype=np.float64)
        source_dates = dates.copy()
    else:
        event["trade_date"] = pd.to_datetime(event["trade_date"], errors="coerce")
        event["back_adjust_factor"] = pd.to_numeric(
            event["back_adjust_factor"], errors="coerce"
        )
        event = event.dropna(subset=["trade_date", "back_adjust_factor"])
        event = event.loc[event["back_adjust_factor"].gt(0)].sort_values("trade_date")
        if event.empty:
            factor = np.ones(len(history), dtype=np.float64)
            source_dates = dates.copy()
        else:
            factor_provider = str(
                event.get("factor_provider", pd.Series(["sina_via_akshare"])).iloc[0]
            )
            factor_source = str(
                event.get("source", pd.Series(["akshare_sina_hfq_factor_event"])).iloc[0]
            )
            event_dates = event["trade_date"].to_numpy(dtype="datetime64[ns]")
            event_values = event["back_adjust_factor"].to_numpy(dtype=np.float64)
            position = np.searchsorted(
                event_dates,
                dates.to_numpy(dtype="datetime64[ns]"),
                side="right",
            ) - 1
            first_position = int(position[0])
            baseline = (
                float(event_values[first_position]) if first_position >= 0 else 1.0
            )
            current = np.where(
                position >= 0,
                event_values[np.maximum(position, 0)],
                baseline,
            )
            factor = current / baseline
            source_values = np.where(
                position >= 0,
                event_dates[np.maximum(position, 0)],
                dates.to_numpy(dtype="datetime64[ns]"),
            )
            source_dates = pd.Series(
                pd.to_datetime(source_values), index=history.index
            )
    symbol = str(history["symbol"].iloc[0])
    trade_date = history["trade_date"].astype(str)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": trade_date,
            "fore_adjust_factor": factor,
            "back_adjust_factor": factor,
            "adjust_factor": factor,
            "factor_provider": factor_provider,
            "factor_semantics": "sina_hfq_factor_ratio_normalized_to_first_qdp_observation",
            "source": factor_source + "+pit_history_restore",
            "factor_source_date": source_dates.dt.strftime("%Y-%m-%d"),
            "ffill_days": (dates - source_dates).dt.days.astype("int64"),
        }
    )


def _prepare_symbol_parts(
    ctx: PitHistoryContext,
    *,
    row: Mapping[str, Any],
    identity_already_known: bool = False,
) -> None:
    symbol = str(row["symbol"])
    token = symbol.replace(".", "_")
    output = ctx.runtime / "domain_parts"
    done = output / "done" / f"{token}.json"
    if done.is_file():
        try:
            completed = json.loads(done.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            completed = {}
        if str(completed.get("semantic_version", "")) == PART_SEMANTIC_VERSION:
            return
    history = pd.read_parquet(ctx.runtime / "history_parts" / f"{token}.parquet")
    factors = pd.read_parquet(ctx.runtime / "factor_parts" / f"{token}.parquet")
    share_events = pd.read_parquet(ctx.runtime / "share_event_parts" / f"{token}.parquet")
    intervals = pd.read_parquet(ctx.runtime / "name_interval_parts" / f"{token}.parquet")
    history = (
        history.sort_values("trade_date")
        .drop_duplicates("trade_date", keep="last")
        .reset_index(drop=True)
    )
    if "history_source" not in history:
        history["history_source"] = "legacy_baostock_history_fixture"
    numeric_columns = (
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "turn",
        "peTTM",
        "pbMRQ",
        "psTTM",
        "pcfNcfTTM",
    )
    for column in numeric_columns:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    suspended_input = history["tradestatus"].fillna("").astype(str).ne("1")
    for column in ("volume", "amount", "turn"):
        history.loc[suspended_input & history[column].isna(), column] = 0.0
    history["trade_date"] = pd.to_datetime(
        history["trade_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    history["symbol"] = symbol
    values = history.loc[:, ["open", "high", "low", "close", "volume", "amount"]]
    valid_bar = (
        values.notna().all(axis=1)
        & values[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & values[["volume", "amount"]].ge(0).all(axis=1)
        & history["high"].ge(history[["open", "low", "close"]].max(axis=1))
        & history["low"].le(history[["open", "high", "close"]].min(axis=1))
    )
    invalid_observed = (~suspended_input) & (~valid_bar)
    if bool(invalid_observed.any()):
        raise PitHistoryError(
            f"pit_history_invalid_bar:{symbol}:{int(invalid_observed.sum())}"
        )
    names = _historical_names(
        history["trade_date"], intervals, fallback=str(row.get("name", ""))
    )
    archive_names = history.get(
        "name_on_date", pd.Series("", index=history.index, dtype="object")
    ).fillna("").astype(str).str.strip()
    names = archive_names.where(archive_names.ne(""), names)
    delist_date = str(row.get("delist_date", "") or "")
    is_delisted = (
        history["trade_date"].ge(delist_date)
        if delist_date
        else pd.Series(False, index=history.index)
    )
    is_suspended = suspended_input
    is_st = _historical_st_status(history, symbol=symbol, names=names)
    observed_bar = ~is_suspended

    daily = pd.DataFrame(
        {
            "symbol": symbol,
            "semantic_version": PART_SEMANTIC_VERSION,
            "trade_date": history["trade_date"],
            "open": history["open"].astype("float64"),
            "high": history["high"].astype("float64"),
            "low": history["low"].astype("float64"),
            "close": history["close"].astype("float64"),
            "volume": history["volume"].astype("float64"),
            "amount": history["amount"].astype("float64"),
            "source": history["history_source"].astype(str),
            "adjusted_flag": "none",
        }
    ).loc[observed_bar].reset_index(drop=True)
    universe = pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": history["trade_date"],
            "name": names,
            "exchange": _short_exchange(symbol),
            "board": "main",
            "list_status": np.where(is_delisted, "D", "L"),
            "list_date": str(row.get("list_date", "") or ""),
            "delist_date": delist_date,
            "source": "protected_archive+exchange_name_history+akshare_pit_restore",
        }
    )
    reasons = np.where(
        is_delisted,
        "delisted",
        np.where(is_st, "st", np.where(is_suspended, "suspended", "")),
    )
    status = pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": history["trade_date"],
            "is_st": is_st.astype(bool),
            "is_suspended": is_suspended.astype(bool),
            "is_delisted": is_delisted.astype(bool),
            "status_reason": reasons,
            "source": "protected_archive+exchange_status_evidence+akshare_pit_restore",
        }
    )
    factor = _factor_rows(history, factors).loc[observed_bar].reset_index(drop=True)

    event = share_events.copy()
    if not event.empty:
        event["source_date"] = pd.to_datetime(event["source_date"], errors="coerce")
        event["total_share"] = pd.to_numeric(event["total_share"], errors="coerce")
        event["float_share"] = pd.to_numeric(event["float_share"], errors="coerce")
        event = (
            event.dropna(subset=["source_date"])
            .sort_values("source_date")
            .drop_duplicates("source_date", keep="last")
        )
    history_dates = pd.to_datetime(history["trade_date"], errors="raise")
    if event.empty:
        total_share = pd.Series(np.nan, index=history.index, dtype="float64")
        event_float = pd.Series(np.nan, index=history.index, dtype="float64")
        event_source = pd.Series("", index=history.index, dtype="object")
    else:
        positions = np.searchsorted(
            event["source_date"].to_numpy(dtype="datetime64[ns]"),
            history_dates.to_numpy(dtype="datetime64[ns]"),
            side="right",
        ) - 1
        valid_event = positions >= 0
        safe_position = np.maximum(positions, 0)
        total_values = event["total_share"].to_numpy(dtype=np.float64)[safe_position]
        float_values = event["float_share"].to_numpy(dtype=np.float64)[safe_position]
        source_values = event["source_date"].dt.strftime("%Y-%m-%d").to_numpy()[safe_position]
        total_values[~valid_event] = np.nan
        float_values[~valid_event] = np.nan
        source_values = np.where(valid_event, source_values, "")
        total_share = pd.Series(total_values, index=history.index, dtype="float64")
        event_float = pd.Series(float_values, index=history.index, dtype="float64")
        event_source = pd.Series(source_values, index=history.index, dtype="object")

    turnover = history["turn"].copy()
    turnover.loc[is_suspended & turnover.isna()] = 0.0
    known_float_from_turnover = history["volume"].astype("float64").mul(100.0).div(
        turnover.astype("float64").where(turnover.gt(0.0))
    )
    past_float_for_turnover = event_float.where(
        np.isfinite(event_float) & event_float.gt(0.0),
        known_float_from_turnover,
    ).ffill()
    implied_turnover = history["volume"].astype("float64").mul(100.0).div(
        past_float_for_turnover.where(past_float_for_turnover.gt(0.0))
    )
    turnover = turnover.fillna(implied_turnover)
    if turnover.isna().any():
        raise PitHistoryError(
            f"pit_history_turnover_missing:{symbol}:{int(turnover.isna().sum())}"
        )
    inferred_float_direct = history["volume"].astype("float64").mul(100.0).div(
        turnover.astype("float64").where(turnover.gt(0.0))
    )
    inferred_float_direct = inferred_float_direct.where(
        np.isfinite(inferred_float_direct) & inferred_float_direct.gt(0.0)
    )
    inferred_source_date = pd.Series(
        np.where(
            np.isfinite(inferred_float_direct),
            history["trade_date"],
            None,
        ),
        index=history.index,
        dtype="object",
    ).ffill().fillna("")
    inferred_float = inferred_float_direct.ffill()
    use_event_float = np.isfinite(event_float) & event_float.gt(0.0)
    float_share = event_float.where(use_event_float, inferred_float).astype("float64")
    float_source_date = event_source.where(
        use_event_float,
        inferred_source_date,
    )
    total_valid = np.isfinite(total_share) & total_share.gt(0.0)
    total_source_date = event_source.where(total_valid, "")
    restricted = pd.Series(
        np.where(
            total_valid & np.isfinite(float_share),
            np.maximum(total_share - float_share, 0.0),
            np.nan,
        ),
        index=history.index,
        dtype="float64",
    )
    share_capital = pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": history["trade_date"],
            "total_share": total_share,
            "float_share": float_share,
            "restricted_share": restricted,
            "total_share_source_date": total_source_date.astype(str),
            "float_share_source_date": float_source_date.astype(str),
            "restricted_share_source_date": total_source_date.astype(str),
            "share_fill_method": np.where(
                total_valid & use_event_float,
                "past_only_cninfo_event",
                "same_day_turnover_float_inference",
            ),
            "source": "cninfo_event+archive_or_eastmoney_turnover_pit_restore",
        }
    ).loc[observed_bar].reset_index(drop=True)
    valuation = pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": history["trade_date"],
            "total_mv": history["close"].to_numpy(dtype=np.float64)
            * total_share.to_numpy(dtype=np.float64),
            "circ_mv": history["close"].to_numpy(dtype=np.float64)
            * float_share.to_numpy(dtype=np.float64),
            "pe": history["peTTM"].astype("float64"),
            "pb": history["pbMRQ"].astype("float64"),
            "turnover_rate": turnover.astype("float64"),
            "source": "archive_or_eastmoney+cninfo_or_turnover_share_pit_restore",
        }
    ).loc[observed_bar].reset_index(drop=True)
    industry = pd.DataFrame(
        {
            "symbol": daily["symbol"],
            "trade_date": daily["trade_date"],
            "industry": "Unknown",
            "source": "pit_history_restore_industry_unavailable",
            "original_industry": "",
            "original_source": "",
            "industry_fill_method": "unavailable",
            "industry_source_date": daily["trade_date"],
            "industry_standard": "Unclassified",
        }
    )
    identity = pd.DataFrame(
        {
            "security_id": [_security_id(symbol)],
            "official_org_id": [""],
            "issuer_name": [str(row.get("name", ""))],
            "exchange": [_identity_exchange(symbol)],
            "list_date": [str(row.get("list_date", ""))],
            "current_symbol": [symbol],
            "identity_source": ["protected_archive_security_master+pit_history_restore"],
        }
    )
    history_identity = pd.DataFrame(
        {
            "security_id": [_security_id(symbol)],
            "symbol": [symbol],
            "effective_from": [str(row.get("list_date", ""))],
            "effective_to": [delist_date or "9999-12-31"],
            "name_on_date": [str(row.get("name", ""))],
            "board_on_date": ["MainBoard"],
            "evidence_source": ["protected_archive_security_master+pit_history_restore"],
            "official_document_hash": [""],
        }
    )
    if identity_already_known:
        # A historical ticker can already belong to an officially documented
        # code-change identity even when it is absent from the active daily
        # universe.  Restore its price/status history, but do not invent a
        # second security_id or an overlapping symbol-history interval.
        identity = identity.iloc[0:0].copy()
        history_identity = history_identity.iloc[0:0].copy()
    name_events = pd.DataFrame(
        columns=["symbol", "trade_date", "old_name", "new_name", "change_type", "source"]
    )
    if not intervals.empty:
        ordered = intervals.sort_values("start_date").copy()
        ordered["old_name"] = ordered["name"].shift(1)
        ordered = ordered.loc[
            ordered["old_name"].notna()
            & ordered["old_name"].ne(ordered["name"])
            & ordered["start_date"].ge(ctx.start_date)
        ]
        name_events = pd.DataFrame(
            {
                "symbol": symbol,
                "trade_date": ordered["start_date"].astype(str),
                "old_name": ordered["old_name"].astype(str),
                "new_name": ordered["name"].astype(str),
                "change_type": "short_name",
                "source": "exchange_name_intervals+pit_history_restore",
            }
        )

    domain_frames = {
        "security_identity": identity,
        "symbol_history": history_identity,
        "market_daily_raw": daily,
        "universe_snapshot": universe,
        "security_status": status,
        "adjust_factor": factor,
        "industry_concept": industry,
        "share_capital": share_capital,
        "valuation": valuation,
        "name_change": name_events,
    }
    for domain, frame in domain_frames.items():
        _atomic_parquet(frame, output / domain / f"{token}.parquet")
    done.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        done,
        {
            "symbol": symbol,
            "semantic_version": PART_SEMANTIC_VERSION,
            "row_counts": {key: int(len(value)) for key, value in domain_frames.items()},
            "completed_at": utc_now(),
        },
    )


def _combine_domain_parts(ctx: PitHistoryContext, domain: str) -> Path:
    parts = sorted((ctx.runtime / "domain_parts" / domain).glob("*.parquet"))
    if not parts:
        raise PitHistoryError(f"pit_history_domain_parts_missing:{domain}")
    current = resolve_active_domain(domain, workspace_root=ctx.workspace)
    keys = list(current.manifest.primary_key)
    if not keys:
        raise PitHistoryError(f"pit_history_primary_key_missing:{domain}")
    prepared = ctx.runtime / "prepared" / f"{domain}.parquet"
    prepared.parent.mkdir(parents=True, exist_ok=True)
    quoted = ",".join(f'"{item}"' for item in keys)
    temporary = prepared.with_name(f".{prepared.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with open_guarded_duckdb(
            temp_directory=ctx.runtime / "combine_spill" / domain,
            threads=4,
        ) as con:
            con.execute(
                f"COPY ("
                "WITH incoming AS ("
                " SELECT * FROM read_parquet(?, union_by_name=true)"
                f" QUALIFY row_number() OVER (PARTITION BY {quoted} ORDER BY {quoted})=1"
                "), existing AS ("
                f" SELECT {quoted} FROM read_parquet(?, union_by_name=true)"
                ") SELECT i.* FROM incoming i ANTI JOIN existing e "
                f"USING ({quoted}) ORDER BY {quoted}"
                f") TO '{str(temporary).replace(chr(39), chr(39) * 2)}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)",
                [
                    [str(item) for item in parts],
                    [str(item) for item in current.shard_paths],
                ],
            )
        temporary.replace(prepared)
    finally:
        temporary.unlink(missing_ok=True)
        shutil.rmtree(ctx.runtime / "combine_spill" / domain, ignore_errors=True)
    return prepared


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _create_composite_dataset(
    ctx: PitHistoryContext,
    *,
    domain: str,
    prepared: Path,
) -> tuple[str, dict[str, Any]]:
    current = resolve_active_domain(domain, workspace_root=ctx.workspace)
    frame = pd.read_parquet(prepared)
    if frame.empty:
        return current.dataset_id, {"status": "unchanged", "added_rows": 0}
    content_sha = _file_sha256(prepared)
    dataset_id = f"{domain}__{stable_hash({'policy': 'pit_historical_mainboard_v2', 'old': current.dataset_id, 'sha256': content_sha})}"
    dataset_dir = ctx.root / "datasets" / domain / dataset_id
    shard = dataset_dir / "shards" / f"pit_restore_{content_sha[:16]}.parquet"
    if not shard.is_file():
        shard.parent.mkdir(parents=True, exist_ok=True)
        temporary = shard.with_name(f".{shard.name}.{os.getpid()}.tmp")
        temporary.unlink(missing_ok=True)
        try:
            shutil.copy2(prepared, temporary)
            if _file_sha256(temporary) != content_sha:
                raise PitHistoryError(f"pit_history_copy_hash_mismatch:{domain}")
            temporary.replace(shard)
        finally:
            temporary.unlink(missing_ok=True)
    date_column = "trade_date" if "trade_date" in frame.columns else ""
    start = str(frame[date_column].min()) if date_column and len(frame) else ""
    end = str(frame[date_column].max()) if date_column and len(frame) else ""
    entry = ShardManifestEntry(
        path=path_for_manifest(shard, root=ctx.root),
        row_count=int(len(frame)),
        start_date=start,
        end_date=end,
        status="stored",
        file_size=shard.stat().st_size,
        schema_hash=current.manifest.schema_hash,
        source_path=str(prepared),
        content_key=f"pit-history:{content_sha[:24]}",
        metadata={
            "restore_policy": "pit_historical_mainboard_v2",
            "restore_content_sha256": content_sha,
        },
    )
    payload = current.manifest.to_dict()
    payload["dataset_id"] = dataset_id
    payload["shards"] = [item.to_dict() for item in current.manifest.shards] + [entry.to_dict()]
    payload["row_count"] = int(current.manifest.row_count) + int(len(frame))
    starts = [item.start_date for item in current.manifest.shards if item.start_date]
    ends = [item.end_date for item in current.manifest.shards if item.end_date]
    if start:
        starts.append(start)
    if end:
        ends.append(end)
    payload["start_date"] = min(starts) if starts else ""
    payload["end_date"] = max(ends) if ends else ""
    payload["created_at"] = utc_now()
    payload["notes"] = [
        item
        for item in list(payload.get("notes", []) or [])
        if "permanent_st_delisting_exclusion" not in str(item)
    ] + ["composite_restore:pit_historical_mainboard_v2"]
    source = dict(payload.get("source", {}) or {})
    source.update(
        {
            "scope": "point_in_time_historical_mainboard",
            "survivorship_policy": "include_when_listed_then_apply_same_day_status",
            "pit_history_restored_at": utc_now(),
        }
    )
    payload["source"] = source
    quality = dict(payload.get("quality", {}) or {})
    quality.pop("permanent_exclusions", None)
    quality.update(
        {
            "scope": "point_in_time_historical_mainboard",
            "survivorship_bias_free_mainboard_daily": True,
            "pit_status_is_signal_date_asof": True,
        }
    )
    if domain == "industry_concept":
        quality["restored_unclassified_rows"] = int(
            frame["industry_fill_method"].astype(str).eq("unavailable").sum()
        )
        source["source_contract"] = (
            "existing strict-PIT industry plus explicit Unclassified for restored "
            "historical securities where no dated industry evidence is available"
        )
    elif domain == "share_capital":
        quality["total_share_null_rows"] = int(
            quality.get("total_share_null_rows", 0) or 0
        ) + int(frame["total_share"].isna().sum())
        quality["float_share_null_rows"] = int(
            quality.get("float_share_null_rows", 0) or 0
        ) + int(frame["float_share"].isna().sum())
        quality["restored_float_share_inference"] = (
            "same_day_volume_divided_by_archive_or_eastmoney_turnover_rate"
        )
        source["source_contract"] = (
            "existing strict-PIT shares plus past-only CNInfo events; missing float "
            "shares inferred from same-day volume and turnover without future fill"
        )
    elif domain == "valuation":
        quality["total_mv_null_rows"] = int(
            quality.get("total_mv_null_rows", 0) or 0
        ) + int(frame["total_mv"].isna().sum())
        quality["circ_mv_null_rows"] = int(
            quality.get("circ_mv_null_rows", 0) or 0
        ) + int(frame["circ_mv"].isna().sum())
        source["source_contract"] = (
            "existing strict-PIT valuation plus protected-archive PE/PB/turnover, "
            "Eastmoney boundary turnover, and price-times-past-only-or-turnover-inferred shares"
        )
    elif domain == "name_change":
        quality["restored_name_history_scope"] = (
            "dated Shenzhen exchange short-name history; Shanghai exact-date ST state is "
            "restored separately from official SSE factbooks"
        )
    payload["quality"] = quality
    manifest = DatasetManifest.from_mapping(payload)
    write_dataset_manifest(ctx.root, manifest)
    return dataset_id, {
        "status": "created",
        "added_rows": int(len(frame)),
        "dataset_id": dataset_id,
        "manifest_sha256": canonical_manifest_sha256(manifest.to_dict()),
    }


def _validate_prepared(
    ctx: PitHistoryContext,
    *,
    symbols: Sequence[str],
    prepared: Mapping[str, Path],
) -> dict[str, Any]:
    scans = {
        domain: "read_parquet('" + str(path).replace("'", "''") + "')"
        for domain, path in prepared.items()
    }
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "validate_spill",
        threads=4,
    ) as con:
        counts = {
            domain: int(con.execute(f"SELECT count(*) FROM {scan}").fetchone()[0])
            for domain, scan in scans.items()
        }
        duplicates = {}
        for domain in RESTORE_DOMAINS:
            current = resolve_active_domain(domain, workspace_root=ctx.workspace)
            keys = ",".join(f'"{item}"' for item in current.manifest.primary_key)
            duplicates[domain] = int(
                con.execute(
                    f"SELECT count(*) FROM (SELECT {keys},count(*) n FROM {scans[domain]} "
                    f"GROUP BY {keys} HAVING n>1)"
                ).fetchone()[0]
            )
        missing_factor = int(
            con.execute(
                f"SELECT count(*) FROM {scans['market_daily_raw']} d ANTI JOIN "
                f"{scans['adjust_factor']} f USING(symbol,trade_date)"
            ).fetchone()[0]
        )
        missing_status = int(
            con.execute(
                f"SELECT count(*) FROM {scans['market_daily_raw']} d ANTI JOIN "
                f"{scans['security_status']} s USING(symbol,trade_date)"
            ).fetchone()[0]
        )
        missing_universe = int(
            con.execute(
                f"SELECT count(*) FROM {scans['market_daily_raw']} d ANTI JOIN "
                f"{scans['universe_snapshot']} u USING(symbol,trade_date)"
            ).fetchone()[0]
        )
        missing_turnover = int(
            con.execute(
                f"SELECT count(*) FROM {scans['valuation']} WHERE turnover_rate IS NULL"
            ).fetchone()[0]
        )
        missing_industry = int(
            con.execute(
                f"SELECT count(*) FROM {scans['market_daily_raw']} d ANTI JOIN "
                f"{scans['industry_concept']} i USING(symbol,trade_date)"
            ).fetchone()[0]
        )
        missing_share = int(
            con.execute(
                f"SELECT count(*) FROM {scans['market_daily_raw']} d ANTI JOIN "
                f"{scans['share_capital']} s USING(symbol,trade_date)"
            ).fetchone()[0]
        )
        missing_valuation = int(
            con.execute(
                f"SELECT count(*) FROM {scans['market_daily_raw']} d ANTI JOIN "
                f"{scans['valuation']} v USING(symbol,trade_date)"
            ).fetchone()[0]
        )
        invalid_factor = int(
            con.execute(
                f"SELECT count(*) FROM {scans['adjust_factor']} WHERE "
                "adjust_factor IS NULL OR adjust_factor<=0 OR "
                "try_cast(factor_source_date AS DATE)>try_cast(trade_date AS DATE)"
            ).fetchone()[0]
        )
        future_share_source = int(
            con.execute(
                f"SELECT count(*) FROM {scans['share_capital']} WHERE "
                "try_cast(total_share_source_date AS DATE)>try_cast(trade_date AS DATE) OR "
                "try_cast(float_share_source_date AS DATE)>try_cast(trade_date AS DATE) OR "
                "try_cast(restricted_share_source_date AS DATE)>try_cast(trade_date AS DATE)"
            ).fetchone()[0]
        )
        invalid_daily_status = int(
            con.execute(
                f"SELECT count(*) FROM {scans['market_daily_raw']} d JOIN "
                f"{scans['security_status']} s USING(symbol,trade_date) "
                "WHERE s.is_suspended OR s.is_delisted"
            ).fetchone()[0]
        )
        status_without_universe = int(
            con.execute(
                f"SELECT count(*) FROM {scans['security_status']} s ANTI JOIN "
                f"{scans['universe_snapshot']} u USING(symbol,trade_date)"
            ).fetchone()[0]
        )
        future_delist_state = int(
            con.execute(
                f"SELECT count(*) FROM {scans['security_status']} s JOIN "
                f"{scans['universe_snapshot']} u USING(symbol,trade_date) "
                "WHERE s.is_delisted AND (u.delist_date='' OR s.trade_date<u.delist_date)"
            ).fetchone()[0]
        )
        st_rows = int(
            con.execute(
                f"SELECT count(*) FROM {scans['security_status']} WHERE is_st"
            ).fetchone()[0]
        )
        suspended_rows = int(
            con.execute(
                f"SELECT count(*) FROM {scans['security_status']} WHERE is_suspended"
            ).fetchone()[0]
        )
        delisted_rows = int(
            con.execute(
                f"SELECT count(*) FROM {scans['security_status']} WHERE is_delisted"
            ).fetchone()[0]
        )
        float_share_null_rows = int(
            con.execute(
                f"SELECT count(*) FROM {scans['share_capital']} WHERE float_share IS NULL"
            ).fetchone()[0]
        )
        future_name = int(
            con.execute(
                f"SELECT count(*) FROM {scans['name_change']} WHERE trade_date>'{ctx.end_date}'"
            ).fetchone()[0]
        )
        restored_daily_symbols = int(
            con.execute(
                f"SELECT count(DISTINCT symbol) FROM {scans['market_daily_raw']}"
            ).fetchone()[0]
        )
        restored_symbols = int(
            con.execute(
                f"SELECT count(DISTINCT symbol) FROM {scans['universe_snapshot']}"
            ).fetchone()[0]
        )
    shutil.rmtree(ctx.runtime / "validate_spill", ignore_errors=True)
    errors = []
    if any(duplicates.values()):
        errors.append(f"duplicate_keys:{duplicates}")
    if (
        missing_factor
        or missing_status
        or missing_universe
        or missing_turnover
        or missing_industry
        or missing_share
        or missing_valuation
    ):
        errors.append(
            "cross_domain_missing:"
            f"factor={missing_factor}:status={missing_status}:"
            f"universe={missing_universe}:turnover={missing_turnover}:"
            f"industry={missing_industry}:share={missing_share}:"
            f"valuation={missing_valuation}"
        )
    if invalid_factor or future_share_source or invalid_daily_status:
        errors.append(
            "pit_semantic_error:"
            f"invalid_factor={invalid_factor}:future_share={future_share_source}:"
            f"daily_ineligible={invalid_daily_status}"
        )
    if status_without_universe or future_delist_state:
        errors.append(
            "lifecycle_semantic_error:"
            f"status_without_universe={status_without_universe}:"
            f"future_delist_state={future_delist_state}"
        )
    if future_name:
        errors.append(f"future_name_events:{future_name}")
    if restored_symbols != len(symbols):
        errors.append(
            f"restored_symbol_count:{restored_symbols}!={len(symbols)}"
        )
    return {
        "status": "ok" if not errors else "blocked",
        "row_counts": counts,
        "duplicate_key_groups": duplicates,
        "missing_factor_keys": missing_factor,
        "missing_status_keys": missing_status,
        "missing_universe_keys": missing_universe,
        "missing_turnover_rows": missing_turnover,
        "missing_industry_keys": missing_industry,
        "missing_share_keys": missing_share,
        "missing_valuation_keys": missing_valuation,
        "invalid_factor_rows": invalid_factor,
        "future_share_source_rows": future_share_source,
        "daily_ineligible_status_rows": invalid_daily_status,
        "status_without_universe_rows": status_without_universe,
        "future_delist_state_rows": future_delist_state,
        "st_status_rows": st_rows,
        "suspended_status_rows": suspended_rows,
        "delisted_status_rows": delisted_rows,
        "float_share_null_rows": float_share_null_rows,
        "future_name_events": future_name,
        "restored_symbol_count": restored_symbols,
        "restored_daily_symbol_count": restored_daily_symbols,
        "errors": errors,
    }


def _sql_literal(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _quoted_identifier(value: object) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _parquet_scan(paths: Sequence[str | Path]) -> str:
    values = ",".join(_sql_literal(Path(item).resolve()) for item in paths)
    return f"read_parquet([{values}], union_by_name=true)"


def _symbol_lifecycle_tables(
    ctx: PitHistoryContext,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    history = resolve_active_domain("symbol_history", workspace_root=ctx.workspace)
    identity = resolve_active_domain("security_identity", workspace_root=ctx.workspace)
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "lifecycle_inventory_spill",
        threads=2,
    ) as con:
        intervals = con.execute(
            "SELECT cast(security_id AS VARCHAR) security_id, "
            "upper(cast(symbol AS VARCHAR)) symbol, "
            "cast(effective_from AS VARCHAR) effective_from, "
            "cast(effective_to AS VARCHAR) effective_to, "
            "cast(name_on_date AS VARCHAR) name_on_date "
            "FROM read_parquet(?, union_by_name=true)",
            [[str(item) for item in history.shard_paths]],
        ).fetchdf()
        identities = con.execute(
            "SELECT cast(security_id AS VARCHAR) security_id, "
            "cast(list_date AS VARCHAR) list_date "
            "FROM read_parquet(?, union_by_name=true)",
            [[str(item) for item in identity.shard_paths]],
        ).fetchdf()
    shutil.rmtree(ctx.runtime / "lifecycle_inventory_spill", ignore_errors=True)
    intervals = intervals.drop_duplicates(
        ["security_id", "symbol", "effective_from", "effective_to"]
    ).copy()
    intervals["effective_from"] = intervals["effective_from"].astype(str).str[:10]
    intervals["effective_to"] = intervals["effective_to"].astype(str).str[:10]
    multi_ids = (
        intervals.groupby("security_id")["symbol"].nunique().loc[lambda x: x > 1].index
    )
    intervals = intervals.loc[intervals["security_id"].isin(multi_ids)].copy()
    if intervals.empty:
        return intervals, pd.DataFrame(columns=["security_id", "symbol"]), ""
    intervals = intervals.merge(
        identities.drop_duplicates("security_id", keep="last"),
        on="security_id",
        how="left",
        validate="many_to_one",
    )
    intervals["list_date"] = intervals["list_date"].fillna("").astype(str).str[:10]
    intervals["canonical_delist_date"] = ""
    finite = intervals["effective_to"].ne("9999-12-31")
    intervals.loc[finite, "canonical_delist_date"] = (
        pd.to_datetime(intervals.loc[finite, "effective_to"], errors="raise")
        + pd.Timedelta(days=1)
    ).dt.strftime("%Y-%m-%d")
    ordered = intervals.sort_values(["security_id", "effective_from", "effective_to"])
    prior_end = ordered.groupby("security_id")["effective_to"].shift(1)
    overlap = prior_end.notna() & ordered["effective_from"].le(prior_end)
    if overlap.any():
        examples = ordered.loc[overlap, ["security_id", "symbol", "effective_from"]]
        raise PitHistoryError(
            "pit_history_symbol_lifecycle_interval_overlap:"
            f"{examples.head(10).to_dict('records')}"
        )
    symbol_map = intervals[["security_id", "symbol"]].drop_duplicates()
    conflicts = symbol_map.groupby("symbol")["security_id"].nunique()
    if (conflicts > 1).any():
        raise PitHistoryError(
            "pit_history_symbol_lifecycle_symbol_identity_conflict:"
            f"{conflicts.loc[conflicts > 1].index.tolist()[:10]}"
        )
    transitions = ordered.groupby("security_id").tail(-1)
    cutoff = str(transitions["effective_from"].max()) if not transitions.empty else ""
    return ordered.reset_index(drop=True), symbol_map.reset_index(drop=True), cutoff


def _lifecycle_domain_findings(
    con: Any,
    *,
    context: Any,
    sample_limit: int = 20,
) -> dict[str, Any]:
    if "symbol" not in context.manifest.primary_key:
        return {
            "status": "not_applicable",
            "domain": context.domain,
            "outside_effective_interval_rows": 0,
            "examples": [],
        }
    scan = _parquet_scan(context.shard_paths)
    base = f"""
      WITH source_rows AS (
        SELECT upper(cast(d.symbol AS VARCHAR)) source_symbol,
               cast(d.trade_date AS VARCHAR) trade_date,
               m.security_id
        FROM {scan} d
        JOIN lifecycle_symbol_map m
          ON upper(cast(d.symbol AS VARCHAR))=m.symbol
      ), resolved AS (
        SELECT s.*,i.symbol canonical_symbol
        FROM source_rows s
        LEFT JOIN lifecycle_intervals i
          ON i.security_id=s.security_id
         AND s.trade_date BETWEEN i.effective_from AND i.effective_to
      )
    """
    row = con.execute(
        base
        + "SELECT count(*) FILTER (WHERE canonical_symbol IS NULL), "
        "count(*) FILTER (WHERE canonical_symbol IS NOT NULL "
        "AND canonical_symbol<>source_symbol) FROM resolved"
    ).fetchone()
    unmapped = int(row[0] or 0)
    outside = int(row[1] or 0)
    examples: list[dict[str, Any]] = []
    if unmapped or outside:
        examples = con.execute(
            base
            + "SELECT security_id,source_symbol,trade_date,canonical_symbol "
            "FROM resolved WHERE canonical_symbol IS NULL "
            "OR canonical_symbol<>source_symbol "
            "ORDER BY trade_date,source_symbol LIMIT ?",
            [int(sample_limit)],
        ).fetchdf().to_dict("records")
    return {
        "status": "ok" if not unmapped and not outside else "needs_repair",
        "domain": context.domain,
        "outside_effective_interval_rows": outside,
        "unmapped_lifecycle_rows": unmapped,
        "examples": examples,
    }


def audit_symbol_lifecycle_effectivity(
    *,
    workspace_root: str | Path | None = None,
    domains: Sequence[str] = LIFECYCLE_NORMALIZE_DOMAINS,
    sample_limit: int = 20,
) -> dict[str, Any]:
    ctx = _context(
        start_date=DEFAULT_START_DATE,
        end_date=str(
            read_active_manifest(qdp_v2_root(workspace_root)).get(
                "active_as_of_date", DEFAULT_START_DATE
            )
        ),
        workspace_root=workspace_root,
    )
    intervals, symbol_map, cutoff = _symbol_lifecycle_tables(ctx)
    if intervals.empty:
        return {
            "status": "ok",
            "multi_symbol_identity_count": 0,
            "transition_count": 0,
            "domains": {},
        }
    reports: dict[str, Any] = {}
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "lifecycle_audit_spill",
        threads=4,
    ) as con:
        con.register("lifecycle_intervals", intervals)
        con.register("lifecycle_symbol_map", symbol_map)
        for domain in domains:
            context = resolve_active_domain(domain, workspace_root=ctx.workspace)
            reports[domain] = _lifecycle_domain_findings(
                con,
                context=context,
                sample_limit=sample_limit,
            )
    shutil.rmtree(ctx.runtime / "lifecycle_audit_spill", ignore_errors=True)
    outside = sum(
        int(item.get("outside_effective_interval_rows", 0) or 0)
        + int(item.get("unmapped_lifecycle_rows", 0) or 0)
        for item in reports.values()
    )
    return {
        "status": "ok" if outside == 0 else "needs_repair",
        "multi_symbol_identity_count": int(intervals["security_id"].nunique()),
        "transition_count": int(len(intervals) - intervals["security_id"].nunique()),
        "effective_cutoff": cutoff,
        "outside_effective_interval_rows": int(outside),
        "domains": reports,
    }


def _copy_lifecycle_query(con: Any, query: str, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        con.execute(
            f"COPY ({query}) TO {_sql_literal(temporary)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)"
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(target).metadata.num_rows)


def _normalized_lifecycle_projection(
    *,
    context: Any,
) -> tuple[str, str]:
    columns = [str(item.get("name", "")) for item in context.manifest.schema]
    columns = [item for item in columns if item]
    expressions: list[str] = []
    for column in columns:
        quoted = _quoted_identifier(column)
        if column == "symbol":
            expressions.append(f"m.canonical_symbol AS {quoted}")
        elif context.domain == "universe_snapshot" and column == "name":
            expressions.append(
                f"CASE WHEN m.remap_priority=1 THEN "
                f"coalesce(nullif(m.target_name,''),m.{quoted}) "
                f"ELSE m.{quoted} END AS {quoted}"
            )
        elif context.domain == "universe_snapshot" and column == "list_date":
            expressions.append(
                f"CASE WHEN m.remap_priority=1 THEN "
                f"coalesce(nullif(m.target_list_date,''),m.{quoted}) "
                f"ELSE m.{quoted} END AS {quoted}"
            )
        elif context.domain == "universe_snapshot" and column == "delist_date":
            expressions.append(
                f"CASE WHEN m.remap_priority=1 THEN m.target_delist_date "
                f"ELSE m.{quoted} END AS {quoted}"
            )
        else:
            expressions.append(f"m.{quoted}")
    projection = ",".join(expressions)
    partition = ",".join(
        _quoted_identifier(item) for item in context.manifest.primary_key
    )
    return projection, partition


def _prepare_lifecycle_domain_mutation(
    ctx: PitHistoryContext,
    *,
    domain: str,
    intervals: pd.DataFrame,
    symbol_map: pd.DataFrame,
    cutoff: str,
) -> tuple[list[tuple[Path, Path]], list[Path], list[Path], dict[str, Any]]:
    context = resolve_active_domain(domain, workspace_root=ctx.workspace)
    prepared_dir = ctx.runtime / "lifecycle_effectivity" / "prepared" / domain
    prepared_dir.mkdir(parents=True, exist_ok=True)
    replacements: list[tuple[Path, Path]] = []
    removals: list[Path] = []
    affected: list[Path] = []
    kept_rows = 0
    source_rows = 0
    with open_guarded_duckdb(
        temp_directory=ctx.runtime / "lifecycle_effectivity" / "spill" / domain,
        threads=4,
    ) as con:
        con.register("lifecycle_intervals", intervals)
        con.register("lifecycle_symbol_map", symbol_map)
        for index, old_path in enumerate(context.shard_paths):
            row = con.execute(
                f"SELECT count(*) FROM {_parquet_scan([old_path])} d "
                "JOIN lifecycle_symbol_map s "
                "ON upper(cast(d.symbol AS VARCHAR))=s.symbol "
                "WHERE cast(d.trade_date AS VARCHAR)<=?",
                [cutoff],
            ).fetchone()
            if not int(row[0] or 0):
                continue
            affected.append(old_path)
            target = prepared_dir / f"retained_{index:04d}.parquet"
            query = (
                f"SELECT d.* FROM {_parquet_scan([old_path])} d "
                "LEFT JOIN lifecycle_symbol_map s "
                "ON upper(cast(d.symbol AS VARCHAR))=s.symbol "
                "WHERE s.symbol IS NULL OR cast(d.trade_date AS VARCHAR)>"
                f"{_sql_literal(cutoff)}"
            )
            count = _copy_lifecycle_query(con, query, target)
            if count:
                replacements.append((old_path, target))
                kept_rows += count
            else:
                target.unlink(missing_ok=True)
                removals.append(old_path)
        if not affected:
            return [], [], [], {
                "status": "unchanged",
                "affected_shard_count": 0,
                "source_row_count": 0,
            }
        scans = _parquet_scan(affected)
        missing_mapping = int(
            con.execute(
                f"WITH source_rows AS ("
                f" SELECT d.*,upper(cast(d.symbol AS VARCHAR)) source_symbol,"
                f" s.security_id FROM {scans} d JOIN lifecycle_symbol_map s "
                " ON upper(cast(d.symbol AS VARCHAR))=s.symbol "
                " WHERE cast(d.trade_date AS VARCHAR)<=?"
                ") SELECT count(*) FROM source_rows r LEFT JOIN lifecycle_intervals i "
                " ON i.security_id=r.security_id AND cast(r.trade_date AS VARCHAR) "
                " BETWEEN i.effective_from AND i.effective_to WHERE i.symbol IS NULL",
                [cutoff],
            ).fetchone()[0]
            or 0
        )
        if missing_mapping:
            raise PitHistoryError(
                f"pit_history_symbol_lifecycle_mapping_missing:{domain}:{missing_mapping}"
            )
        projection, partition = _normalized_lifecycle_projection(context=context)
        canonical = prepared_dir / "canonicalized.parquet"
        query = f"""
          WITH source_rows AS (
            SELECT d.*,upper(cast(d.symbol AS VARCHAR)) source_symbol,
                   s.security_id
            FROM {scans} d
            JOIN lifecycle_symbol_map s
              ON upper(cast(d.symbol AS VARCHAR))=s.symbol
            WHERE cast(d.trade_date AS VARCHAR)<={_sql_literal(cutoff)}
          ), mapped AS (
            SELECT r.*,i.symbol canonical_symbol,
                   i.name_on_date target_name,
                   i.list_date target_list_date,
                   i.canonical_delist_date target_delist_date,
                   CASE WHEN r.source_symbol=i.symbol THEN 0 ELSE 1 END remap_priority
            FROM source_rows r
            JOIN lifecycle_intervals i
              ON i.security_id=r.security_id
             AND cast(r.trade_date AS VARCHAR)
                 BETWEEN i.effective_from AND i.effective_to
          ), normalized AS (
            SELECT {projection},m.remap_priority,m.source_symbol
            FROM mapped m
          )
          SELECT * EXCLUDE(remap_priority,source_symbol)
          FROM normalized
          QUALIFY row_number() OVER(
            PARTITION BY {partition}
            ORDER BY remap_priority,source_symbol
          )=1
          ORDER BY {partition}
        """
        canonical_rows = _copy_lifecycle_query(con, query, canonical)
        source_rows = int(
            con.execute(
                f"SELECT count(*) FROM {scans} d JOIN lifecycle_symbol_map s "
                "ON upper(cast(d.symbol AS VARCHAR))=s.symbol "
                "WHERE cast(d.trade_date AS VARCHAR)<=?",
                [cutoff],
            ).fetchone()[0]
            or 0
        )
    shutil.rmtree(
        ctx.runtime / "lifecycle_effectivity" / "spill" / domain,
        ignore_errors=True,
    )
    return replacements, removals, [canonical], {
        "status": "prepared",
        "affected_shard_count": len(affected),
        "replacement_count": len(replacements),
        "removal_count": len(removals),
        "source_row_count": source_rows,
        "canonical_row_count": canonical_rows,
        "deduplicated_row_count": source_rows - canonical_rows,
        "retained_row_count": kept_rows,
    }


def normalize_symbol_lifecycle_effectivity(
    *,
    workspace_root: str | Path | None = None,
    domains: Sequence[str] = LIFECYCLE_NORMALIZE_DOMAINS,
    apply: bool = True,
) -> dict[str, Any]:
    active = read_active_manifest(qdp_v2_root(workspace_root))
    ctx = _context(
        start_date=DEFAULT_START_DATE,
        end_date=str(active.get("active_as_of_date", DEFAULT_START_DATE)),
        workspace_root=workspace_root,
    )
    before = audit_symbol_lifecycle_effectivity(
        workspace_root=ctx.workspace,
        domains=domains,
    )
    if before["status"] == "ok":
        return {"status": "already_normalized", "before": before, "domains": {}}
    if not apply:
        return {"status": "planned", "before": before, "domains": {}}
    intervals, symbol_map, cutoff = _symbol_lifecycle_tables(ctx)
    results: dict[str, Any] = {}
    for domain in domains:
        finding = dict(before["domains"].get(domain, {}) or {})
        if finding.get("status") == "ok":
            results[domain] = {"status": "already_normalized"}
            continue
        replacements, removals, appends, prepared = _prepare_lifecycle_domain_mutation(
            ctx,
            domain=domain,
            intervals=intervals,
            symbol_map=symbol_map,
            cutoff=cutoff,
        )
        mutation = mutate_active_shards_from_parquet(
            domain,
            replacements=replacements,
            removals=removals,
            appends=appends,
            reason="canonicalize ticker rows to symbol_history effective intervals",
            workspace_root=ctx.workspace,
        )
        update_active_manifest_metadata(
            domain,
            reason="record symbol_history effective-interval normalization",
            workspace_root=ctx.workspace,
            source_updates={
                "symbol_lifecycle_semantics": "date_effective_symbol_history",
                "symbol_lifecycle_normalized_at": utc_now(),
            },
            quality_updates={
                "symbol_history_effective_intervals": True,
                "multi_symbol_identity_count": int(
                    intervals["security_id"].nunique()
                ),
            },
        )
        results[domain] = {
            **prepared,
            "status": str(mutation.get("status", "")),
            "mutation_id": str(mutation.get("mutation_id", "")),
            "manifest_row_count": int(mutation.get("manifest_row_count", 0) or 0),
        }
        prepared_dir = (
            ctx.runtime / "lifecycle_effectivity" / "prepared" / domain
        )
        if prepared_dir.exists():
            shutil.rmtree(prepared_dir.resolve(strict=True))
    after = audit_symbol_lifecycle_effectivity(
        workspace_root=ctx.workspace,
        domains=domains,
    )
    if after["status"] != "ok":
        raise PitHistoryError(
            "pit_history_symbol_lifecycle_normalization_incomplete:"
            f"{after['outside_effective_interval_rows']}"
        )
    payload = {
        "status": "normalized",
        "before": before,
        "after": after,
        "domains": results,
        "intraday_5m_changed": False,
        "completed_at": utc_now(),
    }
    atomic_write_json(
        ctx.runtime / "lifecycle_effectivity" / "normalization.json",
        payload,
    )
    return json_safe(payload)


def run_pit_history_restore(
    *,
    start_date: str = DEFAULT_START_DATE,
    end_date: str,
    workspace_root: str | Path | None = None,
    workers: int = 3,
    chunk_size: int = 20,
    apply: bool = True,
) -> dict[str, Any]:
    ctx = _context(
        start_date=start_date,
        end_date=end_date,
        workspace_root=workspace_root,
    )
    before = read_active_manifest(ctx.root)
    provider = BaostockProvider(
        _market_daily_max_workers=1,
        _reuse_symbol_range_session=True,
    )
    try:
        inventory = inventory_pit_history(
            start_date=ctx.start_date,
            end_date=ctx.end_date,
            workspace_root=ctx.workspace,
            provider=provider,
        )
        symbols = list(inventory["missing_symbols"])
        if not symbols:
            lifecycle = normalize_symbol_lifecycle_effectivity(
                workspace_root=ctx.workspace,
                apply=apply,
            )
            return {
                **inventory,
                "status": (
                    "updated"
                    if lifecycle.get("status") == "normalized"
                    else "already_complete"
                ),
                "symbol_lifecycle": lifecycle,
            }
        basic = _stock_basic(ctx, provider).set_index("symbol", drop=False)
        archive_cache = _build_archive_cache(ctx, symbols=symbols)
        _download_market_supplements(ctx, symbols=symbols, workers=workers)
        _download_sina_factors(ctx, symbols=symbols, workers=workers)
        _materialize_history_parts(
            ctx,
            symbols=symbols,
            basic=basic.reset_index(drop=True),
            archive_cache=archive_cache,
        )
        _fill_missing_turnover_from_tencent(ctx, symbols=symbols)
    finally:
        provider.close()
    _download_reference_parts(ctx, symbols=symbols, workers=workers)
    known_history_symbols = _read_symbol_history_symbols(ctx)
    for index, symbol in enumerate(symbols, start=1):
        _prepare_symbol_parts(
            ctx,
            row=basic.loc[symbol].to_dict(),
            identity_already_known=symbol in known_history_symbols,
        )
        if index == 1 or index % 25 == 0 or index == len(symbols):
            print(f"pit_history_prepare={index}/{len(symbols)}", flush=True)
    prepared = {
        domain: _combine_domain_parts(ctx, domain) for domain in RESTORE_DOMAINS
    }
    validation = _validate_prepared(ctx, symbols=symbols, prepared=prepared)
    if validation["status"] != "ok":
        payload = {
            "status": "blocked",
            "inventory": inventory,
            "validation": validation,
            "active_unchanged": True,
        }
        atomic_write_json(ctx.runtime / "result.json", payload)
        return payload
    if not apply:
        payload = {
            "status": "prepared",
            "inventory": inventory,
            "validation": validation,
            "active_unchanged": True,
        }
        atomic_write_json(ctx.runtime / "result.json", payload)
        return payload

    datasets: dict[str, str] = dict(before.get("datasets", {}) or {})
    commits: dict[str, Any] = {}
    for domain in RESTORE_DOMAINS:
        dataset_id, commit = _create_composite_dataset(
            ctx,
            domain=domain,
            prepared=prepared[domain],
        )
        datasets[domain] = dataset_id
        commits[domain] = commit
    scope = {
        "name": "point_in_time_historical_mainboard",
        "universe": (
            "Shanghai/Shenzhen main-board securities are included on dates when "
            "listed; same-day ST, suspension, and delisting state controls eligibility."
        ),
        "start_date": ctx.start_date,
        "end_date": ctx.end_date,
        "symbol_count": int(inventory["pit_mainboard_symbol_count"]),
        "restored_symbol_count": int(inventory["missing_symbol_count"]),
        "restored_delisted_symbol_count": int(
            inventory["missing_delisted_symbol_count"]
        ),
        "survivorship_policy": "point_in_time_no_future_exclusion",
        "intraday_5m_restored_for_historical_symbols": False,
    }
    archive_paths = _archive_paths(ctx)
    source_evidence = {
        "archive_paths": {key: str(value) for key, value in archive_paths.items()},
        "sse_factbooks": SSE_FACTBOOK_EVIDENCE,
        "sse_transition_resource": str(SSE_ST_TRANSITIONS),
        "sse_transition_resource_sha256": _file_sha256(SSE_ST_TRANSITIONS),
        "factor_provider": "sina_via_akshare_hfq_factor_event",
        "boundary_market_provider": "eastmoney_via_akshare_unadjusted_daily",
    }
    after = {
        **before,
        "datasets": datasets,
        "scope": scope,
        "source": {
            **dict(before.get("source", {}) or {}),
            "pit_history_restore": (
                "protected_baostock_archive+akshare_eastmoney+sina_factor+"
                "cninfo+szse+sse_factbook"
            ),
            "pit_history_restored_at": utc_now(),
            "pit_history_source_evidence": source_evidence,
        },
        "updated_at": utc_now(),
    }
    audit = {
        "schema_version": 1,
        "audit_type": "qdp_pit_historical_mainboard_activation",
        "created_at": utc_now(),
        "before_active": before,
        "after_active": after,
        "inventory": inventory,
        "validation": validation,
        "commits": commits,
        "source_evidence": source_evidence,
    }
    audit_id = stable_hash(audit)
    audit_path = ctx.root / "audits" / f"pit_history_restore_{audit_id}.json"
    atomic_write_json(audit_path, audit)
    write_active_manifest(ctx.root, after)
    lifecycle = normalize_symbol_lifecycle_effectivity(
        workspace_root=ctx.workspace,
        apply=True,
    )
    payload = {
        "status": "updated",
        "inventory": inventory,
        "validation": validation,
        "commits": commits,
        "symbol_lifecycle": lifecycle,
        "audit_path": str(audit_path),
        "active_manifest_sha256": canonical_manifest_sha256(after),
    }
    atomic_write_json(ctx.runtime / "result.json", payload)
    return json_safe(payload)
