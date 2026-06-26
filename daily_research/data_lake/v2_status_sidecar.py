from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from daily_research.data_lake.catalog import ResearchDataLake
from daily_research.data_platform.contracts import DataDomain


V2_STATUS_SIDECAR_DOMAIN = "v2_status_sidecar"
V2_STATUS_COLUMNS = [
    "symbol",
    "trade_date",
    "is_listed_on_date",
    "is_mainboard",
    "is_common_a_share",
    "is_st",
    "is_suspended",
    "is_delisted",
    "is_tradeable",
    "has_bar",
    "reject_reason",
    "source",
]


def is_mainboard_symbol(symbol: Any) -> bool:
    text = str(symbol or "").strip().upper()
    if "." not in text:
        return False
    code, suffix = text.split(".", 1)
    if suffix == "SH":
        return code.startswith(("600", "601", "603", "605"))
    if suffix == "SZ":
        return code.startswith(("000", "001", "002", "003"))
    return False


def _date_text(value: Any) -> str:
    if value is None or str(value).strip() == "":
        return ""
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return ""


def _bool_series(
    frame: pd.DataFrame,
    column: str,
    default: bool = False,
    *,
    true_values: set[str] | None = None,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(default).astype(bool)
    normalized = values.astype("object").where(values.notna(), default).astype(str).str.strip().str.lower()
    markers = true_values or {"1", "true", "t", "yes", "y", "是"}
    return normalized.isin(markers)


def _first_existing_dataset(lake: ResearchDataLake, dataset_kind: str) -> str:
    rows = lake.list_datasets(dataset_kind=dataset_kind)
    if rows.empty:
        return ""
    sort_cols = [column for column in ("end_date", "created_at") if column in rows.columns]
    if sort_cols:
        rows = rows.sort_values(sort_cols)
    return str(rows.iloc[-1]["dataset_id"])


def _read_domain_dataset(lake: ResearchDataLake, dataset_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not dataset_id:
        return pd.DataFrame(), {}
    metadata = lake.describe_dataset(str(dataset_id))
    paths = dict(metadata.get("content_paths", {}) or {})
    existing = _domain_table_paths(paths)
    if not existing:
        return pd.DataFrame(), metadata
    frames = [pd.read_parquet(item) for item in existing]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0], metadata


def _domain_table_paths(paths: Mapping[str, Any]) -> list[str]:
    path = str(dict(paths).get("silver_domain_data", "") or "")
    candidates = sorted(glob.glob(path)) if "*" in path else ([path] if path else [])
    existing = [item for item in candidates if Path(item).exists()]
    if existing:
        return existing
    manifest_path = Path(str(dict(paths).get("shard_manifest", "") or ""))
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_paths: list[str] = []
    for item in list(manifest.get("shards", []) or []):
        row = dict(item)
        if str(row.get("status", "") or "") not in {"stored", "skipped"}:
            continue
        if int(row.get("row_count", 0) or 0) <= 0:
            continue
        shard_path = str(row.get("path", "") or "")
        if shard_path and Path(shard_path).exists():
            shard_paths.append(shard_path)
    return shard_paths


def _read_market_dataset(lake: ResearchDataLake, dataset_id: str, start_date: str, end_date: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = lake.describe_dataset(str(dataset_id))
    path = str(dict(metadata.get("content_paths", {}) or {}).get("bronze_market_data", "") or "")
    candidates = sorted(glob.glob(path)) if "*" in path else ([path] if path else [])
    existing = [item for item in candidates if Path(item).exists()]
    if not existing:
        raise ValueError(f"v2_status_sidecar_blocker: missing bronze_market_data for {dataset_id}")
    market = pd.concat([pd.read_parquet(item) for item in existing], ignore_index=True) if len(existing) > 1 else pd.read_parquet(existing[0])
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="coerce")
    start = pd.Timestamp(start_date or metadata.get("start_date", "") or market["trade_date"].min())
    end = pd.Timestamp(end_date or metadata.get("end_date", "") or market["trade_date"].max())
    market = market.loc[(market["trade_date"] >= start) & (market["trade_date"] <= end)].copy()
    market["trade_date"] = market["trade_date"].dt.strftime("%Y-%m-%d")
    market["symbol"] = market["symbol"].astype(str).str.strip().str.upper()
    return market, metadata


def _align_domain_to_market(domain: pd.DataFrame, market_keys: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if domain is None or domain.empty:
        out = market_keys.copy()
        for column in columns:
            if column not in out.columns:
                out[column] = pd.NA
        return out
    data = domain.copy()
    data["symbol"] = data["symbol"].astype(str).str.strip().str.upper()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    data = data.drop_duplicates(subset=["trade_date", "symbol"], keep="last")
    keep = ["trade_date", "symbol", *[column for column in columns if column in data.columns]]
    return market_keys.merge(data[keep], on=["trade_date", "symbol"], how="left")


def build_v2_status_sidecar_frame(
    *,
    market: pd.DataFrame,
    universe_snapshot: pd.DataFrame | None = None,
    security_status: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if market is None or market.empty:
        empty = pd.DataFrame(columns=V2_STATUS_COLUMNS)
        return empty, {"status": "blocked", "blockers": ["empty_market"]}
    market_data = market.copy()
    market_data["symbol"] = market_data["symbol"].astype(str).str.strip().str.upper()
    market_data["trade_date"] = pd.to_datetime(market_data["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    market_data = market_data.dropna(subset=["trade_date", "symbol"])
    keys = market_data[["trade_date", "symbol"]].drop_duplicates().sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    market_flags = keys.merge(
        market_data[["trade_date", "symbol", "open", "high", "low", "close", "volume", "amount"]],
        on=["trade_date", "symbol"],
        how="left",
    )
    price_cols = ["open", "high", "low", "close"]
    for column in [*price_cols, "volume", "amount"]:
        market_flags[column] = pd.to_numeric(market_flags[column], errors="coerce")
    has_bar = market_flags[price_cols].notna().any(axis=1)
    fallback_suspended = (~has_bar) | market_flags["volume"].isna() | market_flags["amount"].isna() | (market_flags["volume"] <= 0) | (market_flags["amount"] <= 0)

    universe = _align_domain_to_market(
        universe_snapshot if universe_snapshot is not None else pd.DataFrame(),
        keys,
        ["name", "exchange", "board", "list_status", "list_date", "delist_date"],
    )
    status = _align_domain_to_market(
        security_status if security_status is not None else pd.DataFrame(),
        keys,
        ["is_st", "is_suspended", "is_delisted", "status_reason"],
    )

    data = keys.copy()
    data["list_date"] = universe.get("list_date", pd.Series("", index=data.index)).fillna("").astype(str)
    data["delist_date"] = universe.get("delist_date", pd.Series("", index=data.index)).fillna("").astype(str)
    trade_date = pd.to_datetime(data["trade_date"], errors="coerce")
    list_date = pd.to_datetime(data["list_date"].map(_date_text).replace("", pd.NA), errors="coerce")
    delist_date = pd.to_datetime(data["delist_date"].map(_date_text).replace("", pd.NA), errors="coerce")
    list_status = universe.get("list_status", pd.Series("", index=data.index)).fillna("").astype(str).str.upper()

    listed_by_dates = list_date.isna() | (list_date <= trade_date)
    not_delisted_by_dates = delist_date.isna() | (trade_date < delist_date)
    listed_by_status = ~list_status.isin({"D", "DELIST", "退市"})
    data["is_listed_on_date"] = (listed_by_dates & not_delisted_by_dates & listed_by_status).astype(bool)
    data["is_mainboard"] = data["symbol"].map(is_mainboard_symbol).astype(bool)
    data["is_common_a_share"] = True
    data.loc[list_status.isin({"", "L", "LIST", "上市", "1"}), "is_common_a_share"] = True
    data["is_st"] = _bool_series(
        status,
        "is_st",
        default=False,
        true_values={"1", "true", "t", "yes", "y", "是", "st", "*st"},
    ).to_numpy(dtype=bool)
    status_suspended = _bool_series(
        status,
        "is_suspended",
        default=False,
        true_values={"1", "true", "t", "yes", "y", "是", "停牌", "暂停", "suspended", "halted"},
    )
    has_status_suspension = "is_suspended" in status.columns and status["is_suspended"].notna()
    data["is_suspended_from_status"] = has_status_suspension.to_numpy(dtype=bool)
    data["is_suspended"] = np.where(has_status_suspension.to_numpy(dtype=bool), status_suspended.to_numpy(dtype=bool), fallback_suspended.to_numpy(dtype=bool))
    data["is_delisted"] = (
        _bool_series(
            status,
            "is_delisted",
            default=False,
            true_values={"1", "true", "t", "yes", "y", "是", "d", "delist", "delisted", "退市", "摘牌"},
        ).to_numpy(dtype=bool)
        | (~not_delisted_by_dates).to_numpy(dtype=bool)
        | list_status.isin({"D", "DELIST", "退市"}).to_numpy(dtype=bool)
    )
    data["has_bar"] = has_bar.to_numpy(dtype=bool)
    data["is_tradeable"] = (
        data["is_listed_on_date"]
        & data["is_mainboard"]
        & data["is_common_a_share"]
        & (~data["is_st"])
        & (~data["is_suspended"])
        & (~data["is_delisted"])
        & data["has_bar"]
    ).astype(bool)
    data["reject_reason"] = reject_reasons_for_status_frame(data)
    data["source"] = "daily_research_v2_status_sidecar"
    data = data[V2_STATUS_COLUMNS].sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    summary = summarize_v2_status_sidecar(data)
    return data, summary


def reject_reasons_for_status_frame(frame: pd.DataFrame) -> list[str]:
    reasons: list[str] = []
    for row in frame.itertuples(index=False):
        if not bool(getattr(row, "is_listed_on_date")):
            reasons.append("not_listed_on_date")
        elif not bool(getattr(row, "is_mainboard")):
            reasons.append("not_mainboard")
        elif not bool(getattr(row, "is_common_a_share")):
            reasons.append("not_common_a_share")
        elif bool(getattr(row, "is_st")):
            reasons.append("st_on_date")
        elif bool(getattr(row, "is_delisted")):
            reasons.append("delisted_on_date")
        elif bool(getattr(row, "is_suspended")) and (bool(getattr(row, "is_suspended_from_status", False)) or bool(getattr(row, "has_bar"))):
            reasons.append("suspended_on_date")
        elif not bool(getattr(row, "has_bar")):
            reasons.append("missing_bar")
        else:
            reasons.append("")
    return reasons


def summarize_v2_status_sidecar(frame: pd.DataFrame) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {"status": "blocked", "row_count": 0, "blockers": ["empty_status_sidecar"]}
    daily = frame.groupby("trade_date").agg(
        universe_rows=("symbol", "count"),
        tradeable_rows=("is_tradeable", "sum"),
        st_rows=("is_st", "sum"),
        suspended_rows=("is_suspended", "sum"),
        delisted_rows=("is_delisted", "sum"),
        not_listed_rows=("is_listed_on_date", lambda s: int((~s.astype(bool)).sum())),
    )
    blockers: list[str] = []
    if int(frame["is_tradeable"].sum()) <= 0:
        blockers.append("no_tradeable_rows")
    return {
        "status": "ok" if not blockers else "blocked",
        "row_count": int(len(frame)),
        "trade_date_count": int(frame["trade_date"].nunique()),
        "symbol_count": int(frame["symbol"].nunique()),
        "tradeable_rows": int(frame["is_tradeable"].sum()),
        "st_rows": int(frame["is_st"].sum()),
        "suspended_rows": int(frame["is_suspended"].sum()),
        "delisted_rows": int(frame["is_delisted"].sum()),
        "not_listed_rows": int((~frame["is_listed_on_date"].astype(bool)).sum()),
        "reject_reason_counts": {str(k): int(v) for k, v in frame["reject_reason"].fillna("").value_counts().to_dict().items()},
        "daily_summary_head": daily.reset_index().head(20).to_dict("records"),
        "daily_summary_tail": daily.reset_index().tail(20).to_dict("records"),
        "blockers": blockers,
    }


def build_v2_status_sidecar(
    *,
    lake: ResearchDataLake,
    source_market_dataset_id: str,
    start_date: str = "",
    end_date: str = "",
    universe_snapshot_dataset_id: str = "",
    security_status_dataset_id: str = "",
    reuse: bool = True,
) -> Any:
    market, market_metadata = _read_market_dataset(lake, source_market_dataset_id, start_date, end_date)
    parameters = dict(market_metadata.get("parameters", {}) or {})
    sidecars = dict(parameters.get("sidecar_dataset_ids", {}) or {})
    universe_id = universe_snapshot_dataset_id or str(sidecars.get(DataDomain.UNIVERSE_SNAPSHOT, "") or "") or _first_existing_dataset(lake, "data_platform_universe_snapshot")
    status_id = security_status_dataset_id or str(sidecars.get(DataDomain.SECURITY_STATUS, "") or "") or _first_existing_dataset(lake, "data_platform_security_status")
    universe, universe_meta = _read_domain_dataset(lake, universe_id)
    status, status_meta = _read_domain_dataset(lake, status_id)
    frame, summary = build_v2_status_sidecar_frame(market=market, universe_snapshot=universe, security_status=status)
    resolved_start_date = str(start_date or (frame["trade_date"].min() if not frame.empty else ""))
    resolved_end_date = str(end_date or (frame["trade_date"].max() if not frame.empty else ""))
    spec: dict[str, Any] = {
        "dataset": f"data_platform_{V2_STATUS_SIDECAR_DOMAIN}",
        "source": "daily_research_v2_status_sidecar",
        "source_market_dataset_id": str(source_market_dataset_id),
        "universe_snapshot_dataset_id": str(universe_id),
        "security_status_dataset_id": str(status_id),
        "start_date": resolved_start_date,
        "end_date": resolved_end_date,
        "coverage_report": summary,
    }
    record = lake.save_domain_dataset(
        domain=V2_STATUS_SIDECAR_DOMAIN,
        frame=frame,
        spec=spec,
        source="daily_research_v2_status_sidecar",
        reuse=reuse,
    )
    return record, frame, {
        **summary,
        "dataset_id": record.dataset_id,
        "source_market_dataset_id": str(source_market_dataset_id),
        "universe_snapshot_dataset_id": str(universe_id),
        "security_status_dataset_id": str(status_id),
        "universe_snapshot_rows": int(len(universe)),
        "security_status_rows": int(len(status)),
        "universe_snapshot_metadata": {"dataset_id": universe_meta.get("dataset_id", "")} if universe_meta else {},
        "security_status_metadata": {"dataset_id": status_meta.get("dataset_id", "")} if status_meta else {},
    }
