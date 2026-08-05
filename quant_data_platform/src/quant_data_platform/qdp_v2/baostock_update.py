from __future__ import annotations

"""Append new main-board market facts directly from BaoStock."""

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from quant_data_platform.core.security_status import st_status_from_name
from quant_data_platform.domains.contracts import (
    DataDomain,
    DatePartitionFetchRequest,
    DomainFetchRequest,
)
from quant_data_platform.providers import BaostockProvider
from quant_data_platform.qdp_v2.duckdb_resources import open_guarded_duckdb
from quant_data_platform.qdp_v2.repair import (
    append_active_shard,
    resolve_active_domain,
)


CORE_UPDATE_DOMAINS = (
    "trading_calendar",
    "security_identity",
    "symbol_history",
    "universe_snapshot",
    "security_status",
    "market_daily_raw",
)

MAINBOARD_PREFIXES = {
    "SH": ("600", "601", "603", "605"),
    "SZ": ("000", "001", "002", "003"),
}


class BaostockCoreUpdateError(RuntimeError):
    pass


def run_baostock_core_update(
    *,
    start_date: str,
    end_date: str,
    workspace_root: str | Path | None = None,
    provider: Any | None = None,
    apply: bool = True,
    valuation_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    start = _date_text(start_date)
    end = _date_text(end_date)
    if start > end:
        raise ValueError(f"baostock_core_update_invalid_range:{start}>{end}")
    workspace = Path(workspace_root or Path.cwd()).resolve()
    source = provider or BaostockProvider()
    owned = provider is None
    try:
        calendar_result = source.fetch_domain(
            DomainFetchRequest(
                domain=DataDomain.TRADING_CALENDAR,
                start_date=start,
                end_date=end,
                exchange="SSE",
            )
        )
        if list(getattr(calendar_result, "error_report", []) or []):
            raise BaostockCoreUpdateError("baostock_calendar_provider_errors")
        calendar = _calendar_frame(calendar_result.data)
        if calendar.empty:
            raise BaostockCoreUpdateError("baostock_calendar_empty")
        open_dates = tuple(
            calendar.loc[calendar["is_open"], "trade_date"].astype(str).sort_values()
        )

        stock_basic = source.fetch_stock_basic_snapshot(trade_date=end)
        identity, history = _identity_additions(
            stock_basic,
            workspace=workspace,
        )

        daily_frames: list[pd.DataFrame] = []
        valuation_frames: list[pd.DataFrame] = []
        universe_frames: list[pd.DataFrame] = []
        status_frames: list[pd.DataFrame] = []
        for trade_date in open_dates:
            result, all_stock = source.fetch_date_partition_with_all_stock(
                DatePartitionFetchRequest(
                    domain=DataDomain.MARKET_DAILY,
                    trade_date=trade_date,
                    universe_kind="all_a",
                    fetch_mode="date_snapshot",
                )
            )
            if list(getattr(result, "error_report", []) or []):
                raise BaostockCoreUpdateError(
                    f"baostock_daily_partition_errors:{trade_date}"
                )
            daily = _daily_frame(result.data, trade_date=trade_date)
            valuation = _valuation_frame(
                getattr(result, "raw_data", pd.DataFrame()),
                trade_date=trade_date,
            )
            universe, status = _universe_and_status_frames(
                all_stock,
                daily_status=getattr(result, "raw_data", pd.DataFrame()),
                stock_basic=stock_basic,
                trade_date=trade_date,
            )
            if universe.empty or status.empty:
                raise BaostockCoreUpdateError(
                    f"baostock_reference_partition_empty:{trade_date}"
                )
            daily_frames.append(daily)
            valuation_frames.append(valuation)
            universe_frames.append(universe)
            status_frames.append(status)

        candidates = {
            "trading_calendar": calendar,
            "security_identity": identity,
            "symbol_history": history,
            "universe_snapshot": _concat(universe_frames),
            "security_status": _concat(status_frames),
            "market_daily_raw": _concat(daily_frames),
        }
        missing = {
            domain: _only_missing_keys(frame, domain=domain, workspace=workspace)
            for domain, frame in candidates.items()
        }
        counts = {domain: int(len(frame)) for domain, frame in missing.items()}
        payload: dict[str, Any] = {
            "status": "planned" if any(counts.values()) else "already_complete",
            "start_date": start,
            "end_date": end,
            "open_date_count": len(open_dates),
            "missing_rows": counts,
            "provider": "baostock",
        }
        if valuation_cache_path is not None:
            cache = Path(valuation_cache_path).resolve()
            if workspace not in cache.parents:
                raise BaostockCoreUpdateError(
                    f"valuation_cache_outside_workspace:{cache}"
                )
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_name(f".{cache.name}.tmp")
            temporary.unlink(missing_ok=True)
            cache.unlink(missing_ok=True)
            valuation = _concat(valuation_frames)
            try:
                valuation.to_parquet(
                    temporary,
                    index=False,
                    compression="zstd",
                )
                temporary.replace(cache)
            finally:
                temporary.unlink(missing_ok=True)
            payload["valuation_cache_path"] = str(cache)
            payload["valuation_cache_row_count"] = int(len(valuation))
        if not apply or not any(counts.values()):
            return payload

        commits: dict[str, Any] = {}
        for domain in CORE_UPDATE_DOMAINS:
            frame = missing[domain]
            if frame.empty:
                continue
            commits[domain] = append_active_shard(
                domain,
                frame,
                f"append BaoStock core facts {start}..{end}",
                workspace_root=workspace,
            )
        payload["status"] = "updated"
        payload["commits"] = commits
        return payload
    finally:
        if owned:
            close = getattr(source, "close", None)
            if callable(close):
                close()


def _calendar_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["trade_date", "is_open", "exchange", "source"])
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    is_open = data["is_open"].map(_to_nullable_bool).astype("boolean")
    if bool(is_open.isna().any()):
        raise BaostockCoreUpdateError("baostock_calendar_is_open_invalid")
    data["is_open"] = is_open.astype(bool)
    data["exchange"] = data.get("exchange", "SSE").fillna("SSE").astype(str).replace("", "SSE")
    data["source"] = "baostock"
    data = data.dropna(subset=["trade_date"])
    return data.loc[:, ["trade_date", "is_open", "exchange", "source"]].drop_duplicates(
        ["trade_date", "exchange"], keep="last"
    ).reset_index(drop=True)


def _identity_additions(
    stock_basic: pd.DataFrame,
    *,
    workspace: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if stock_basic is None or stock_basic.empty:
        raise BaostockCoreUpdateError("baostock_stock_basic_empty")
    basic = stock_basic.copy()
    basic["symbol"] = basic["symbol"].astype(str).str.upper()
    basic = basic.loc[basic["symbol"].map(_is_supported_mainboard_symbol)].copy()
    history_context = resolve_active_domain("symbol_history", workspace_root=workspace)
    with open_guarded_duckdb(
        temp_directory=_runtime_root(workspace) / "identity_spill",
        threads=2,
    ) as con:
        existing = con.execute(
            "SELECT DISTINCT cast(symbol AS VARCHAR) AS symbol, cast(security_id AS VARCHAR) AS security_id FROM read_parquet(?, union_by_name=true)",
            [[str(item) for item in history_context.shard_paths]],
        ).fetchdf()
    existing_symbols = set(existing["symbol"].astype(str))
    basic = basic.loc[~basic["symbol"].isin(existing_symbols)].copy()
    if basic.empty:
        return _empty_identity(), _empty_history()
    basic["list_date"] = pd.to_datetime(basic.get("list_date", ""), errors="coerce").dt.strftime("%Y-%m-%d")
    basic["list_date"] = basic["list_date"].fillna("")
    basic["name"] = basic.get("name", "").fillna("").astype(str)
    basic["security_id"] = basic["symbol"].map(_security_id)
    identity = pd.DataFrame(
        {
            "security_id": basic["security_id"],
            "official_org_id": "",
            "issuer_name": basic["name"],
            "exchange": basic["symbol"].map(_identity_exchange),
            "list_date": basic["list_date"],
            "current_symbol": basic["symbol"],
            "identity_source": "baostock.query_stock_basic",
        }
    )
    history = pd.DataFrame(
        {
            "security_id": basic["security_id"],
            "symbol": basic["symbol"],
            "effective_from": basic["list_date"].replace("", pd.Timestamp.today().strftime("%Y-%m-%d")),
            "effective_to": "9999-12-31",
            "name_on_date": basic["name"],
            "board_on_date": basic["symbol"].map(_board_name),
            "evidence_source": "baostock.query_stock_basic",
            "official_document_hash": "",
        }
    )
    return identity.reset_index(drop=True), history.reset_index(drop=True)


def _daily_frame(frame: pd.DataFrame, *, trade_date: str) -> pd.DataFrame:
    columns = [
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "source",
        "adjusted_flag",
    ]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    data = frame.rename(columns={"provider_symbol": "symbol"}).copy()
    if "symbol" not in data:
        raise BaostockCoreUpdateError(f"baostock_daily_symbol_missing:{trade_date}")
    data["symbol"] = data["symbol"].astype(str).str.upper()
    data["trade_date"] = str(trade_date)
    numeric = data.loc[:, ["open", "high", "low", "close", "volume", "amount"]].apply(
        pd.to_numeric, errors="coerce"
    ).astype("float64")
    data[numeric.columns] = numeric
    values = numeric.to_numpy(dtype="float64", na_value=np.nan)
    valid = (
        np.isfinite(values).all(axis=1)
        & (values[:, :4] > 0).all(axis=1)
        & (values[:, 4:] >= 0).all(axis=1)
        & (values[:, 1] >= values[:, [0, 2, 3]].max(axis=1))
        & (values[:, 2] <= values[:, [0, 1, 3]].min(axis=1))
    )
    data = data.loc[
        valid & data["symbol"].map(_is_supported_mainboard_symbol)
    ].copy()
    data["source"] = "baostock"
    data["adjusted_flag"] = "none"
    return data.loc[:, columns].drop_duplicates(["trade_date", "symbol"], keep="last").reset_index(drop=True)


def _valuation_frame(frame: pd.DataFrame, *, trade_date: str) -> pd.DataFrame:
    columns = ["symbol", "trade_date", "pe", "pb", "turnover_rate"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    required = {"code", "peTTM", "pbMRQ", "turn"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise BaostockCoreUpdateError(
            f"baostock_daily_valuation_fields_missing:{trade_date}:{missing}"
        )
    data = frame.copy()
    code = data["code"].fillna("").astype(str).str.lower()
    data["symbol"] = np.where(
        code.str.startswith("sh."),
        code.str.slice(3) + ".SH",
        np.where(
            code.str.startswith("sz."),
            code.str.slice(3) + ".SZ",
            "",
        ),
    )
    data["trade_date"] = str(trade_date)
    data["pe"] = pd.to_numeric(data["peTTM"], errors="coerce")
    data["pb"] = pd.to_numeric(data["pbMRQ"], errors="coerce")
    data["turnover_rate"] = pd.to_numeric(data["turn"], errors="coerce")
    data = data.loc[data["symbol"].map(_is_supported_mainboard_symbol)]
    return (
        data.loc[:, columns]
        .drop_duplicates(["trade_date", "symbol"], keep="last")
        .reset_index(drop=True)
    )


def _universe_and_status_frames(
    all_stock: pd.DataFrame,
    *,
    daily_status: pd.DataFrame | None = None,
    stock_basic: pd.DataFrame,
    trade_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if all_stock is None or all_stock.empty:
        return pd.DataFrame(), pd.DataFrame()
    data = all_stock.copy()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    data = data.loc[data["symbol"].map(_is_supported_mainboard_symbol)].copy()
    basic = stock_basic.copy()
    basic["symbol"] = basic["symbol"].astype(str).str.upper()
    basic = basic.drop_duplicates("symbol", keep="last").set_index("symbol")
    for column, default in (("name", ""), ("list_date", ""), ("delist_date", "")):
        lookup = basic[column] if column in basic else pd.Series(dtype=object)
        current = data[column] if column in data else pd.Series("", index=data.index)
        data[column] = current.fillna("").astype(str)
        missing = data[column].eq("")
        data.loc[missing, column] = data.loc[missing, "symbol"].map(lookup).fillna("")
    data["trade_date"] = str(trade_date)
    data["exchange"] = data["symbol"].map(_short_exchange)
    data["board"] = data["symbol"].map(_board_code)
    data["list_status"] = "L"
    data["source"] = "baostock"
    universe = data.loc[
        :, [
            "symbol",
            "trade_date",
            "name",
            "exchange",
            "board",
            "list_status",
            "list_date",
            "delist_date",
            "source",
        ]
    ].drop_duplicates(["trade_date", "symbol"], keep="last")
    if "is_suspended" in data:
        is_suspended = data["is_suspended"].map(_to_nullable_bool).astype("boolean")
    elif "trade_status" in data:
        trade_status = data["trade_status"].fillna("").astype(str).str.strip()
        is_suspended = trade_status.map({"0": True, "1": False}).astype("boolean")
    else:
        is_suspended = pd.Series(pd.NA, index=data.index, dtype="boolean")
    name_is_st = data["name"].map(st_status_from_name).astype("boolean")
    daily_is_st = data["symbol"].map(
        _daily_is_st_lookup(daily_status)
    ).astype("boolean")
    is_st = daily_is_st.fillna(name_is_st).astype("boolean")
    known_st = is_st.fillna(False).astype(bool)
    known_suspended = is_suspended.fillna(False).astype(bool)
    unknown_status = is_st.isna() | is_suspended.isna()
    status_reason = np.select(
        (
            unknown_status & known_st,
            unknown_status & known_suspended,
            unknown_status,
            known_st & known_suspended,
            known_st,
            known_suspended,
        ),
        (
            "st;status_unknown",
            "suspended;status_unknown",
            "status_unknown",
            "st;suspended",
            "st",
            "suspended",
        ),
        default="tradeable",
    )
    status_source = np.where(
        daily_is_st.notna(),
        "baostock.daily_isST",
        np.where(name_is_st.notna(), "baostock.all_stock_name", "status_unknown"),
    )
    status = pd.DataFrame(
        {
            "symbol": data["symbol"],
            "trade_date": str(trade_date),
            "is_st": is_st,
            "is_suspended": is_suspended,
            "is_delisted": False,
            "status_reason": status_reason,
            "source": status_source,
        }
    ).drop_duplicates(["trade_date", "symbol"], keep="last")
    return universe.reset_index(drop=True), status.reset_index(drop=True)


def _only_missing_keys(
    frame: pd.DataFrame,
    *,
    domain: str,
    workspace: Path,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    context = resolve_active_domain(domain, workspace_root=workspace)
    keys = list(context.manifest.primary_key)
    if not keys:
        raise BaostockCoreUpdateError(f"active_primary_key_missing:{domain}")
    data = frame.drop_duplicates(keys, keep="last").reset_index(drop=True)
    date_column = "trade_date" if "trade_date" in data.columns else ""
    where = ""
    params: list[Any] = [[str(item) for item in context.shard_paths]]
    if date_column:
        start = str(data[date_column].min())
        end = str(data[date_column].max())
        where = "WHERE cast(trade_date AS VARCHAR) BETWEEN ? AND ?"
        params.extend([start, end])
    quoted = ",".join(f'"{item}"' for item in keys)
    with open_guarded_duckdb(
        temp_directory=_runtime_root(workspace) / "missing_key_spill",
        threads=4,
    ) as con:
        con.register("qdp_candidate_rows", data)
        result = con.execute(
            f"""
            WITH existing AS (
              SELECT {quoted} FROM read_parquet(?, union_by_name=true) {where}
            )
            SELECT c.* FROM qdp_candidate_rows c
            ANTI JOIN existing e USING ({quoted})
            """,
            params,
        ).fetchdf()
    return result.loc[:, list(data.columns)].reset_index(drop=True)


def _concat(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(list(frames), ignore_index=True, sort=False) if frames else pd.DataFrame()


def _empty_identity() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "security_id", "official_org_id", "issuer_name", "exchange",
        "list_date", "current_symbol", "identity_source",
    ])


def _empty_history() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "security_id", "symbol", "effective_from", "effective_to",
        "name_on_date", "board_on_date", "evidence_source", "official_document_hash",
    ])


def _runtime_root(workspace: Path) -> Path:
    root = (workspace / "quant_data_platform" / "data" / "qdp_runtime" / "baostock_update").resolve()
    if workspace not in root.parents:
        raise BaostockCoreUpdateError(f"runtime_outside_workspace:{root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _security_id(symbol: str) -> str:
    code, suffix = str(symbol).split(".", 1)
    exchange = {"SH": "SSE", "SZ": "SZSE", "BJ": "BSE"}.get(suffix, suffix)
    return f"QDP-CN-{exchange}-{code}"


def _identity_exchange(symbol: str) -> str:
    return {"SH": "SSE", "SZ": "SZSE", "BJ": "BSE"}.get(str(symbol)[-2:], "")


def _short_exchange(symbol: str) -> str:
    return str(symbol)[-2:]


def _board_code(symbol: str) -> str:
    code = str(symbol)[:6]
    if code.startswith(("300", "301")):
        return "chinext"
    if code.startswith("688"):
        return "star"
    if str(symbol).endswith(".BJ"):
        return "beijing"
    return "main"


def _board_name(symbol: str) -> str:
    return {
        "chinext": "ChiNext",
        "star": "STAR",
        "beijing": "Beijing",
        "main": "MainBoard",
    }[_board_code(symbol)]


def _is_supported_mainboard_symbol(symbol: str) -> bool:
    text = str(symbol).upper()
    if len(text) != 9 or text[6] != "." or not text[:6].isdigit():
        return False
    code, suffix = text[:6], text[-2:]
    return suffix in MAINBOARD_PREFIXES and code.startswith(MAINBOARD_PREFIXES[suffix])


def _daily_is_st_lookup(frame: pd.DataFrame | None) -> pd.Series:
    if frame is None or frame.empty or "isST" not in frame.columns:
        return pd.Series(dtype="boolean")
    raw = frame.copy()
    if "symbol" in raw.columns:
        symbol = raw["symbol"].fillna("").astype(str).str.upper().str.strip()
    elif "provider_symbol" in raw.columns:
        symbol = (
            raw["provider_symbol"].fillna("").astype(str).str.upper().str.strip()
        )
    elif "code" in raw.columns:
        code = raw["code"].fillna("").astype(str).str.lower().str.strip()
        symbol = pd.Series(
            np.where(
                code.str.startswith("sh."),
                code.str.slice(3) + ".SH",
                np.where(
                    code.str.startswith("sz."),
                    code.str.slice(3) + ".SZ",
                    "",
                ),
            ),
            index=raw.index,
        )
    else:
        return pd.Series(dtype="boolean")
    values = raw["isST"].fillna("").astype(str).str.strip().str.lower()
    parsed = values.map(
        {"1": True, "true": True, "0": False, "false": False}
    ).astype("boolean")
    result = pd.DataFrame({"symbol": symbol, "is_st": parsed})
    result = result.loc[result["symbol"].ne("")].drop_duplicates(
        "symbol", keep="last"
    )
    return result.set_index("symbol")["is_st"]


def _to_nullable_bool(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return None


def _date_text(value: str) -> str:
    return pd.Timestamp(str(value)).strftime("%Y-%m-%d")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append BaoStock low-frequency facts to the current QDP tables."
    )
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_baostock_core_update(
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        workspace_root=str(args.workspace_root or "") or None,
        apply=not bool(args.dry_run),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("status") in {
        "planned",
        "updated",
        "already_complete",
    } else 2


__all__ = ["BaostockCoreUpdateError", "run_baostock_core_update"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
