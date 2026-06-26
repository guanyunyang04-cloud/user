from __future__ import annotations

import glob
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from quant_data_platform.lake.catalog import LakeDatasetRecord, ResearchDataLake
from daily_research.execution.liquidity_universe import build_rolling_liquidity_membership


@dataclass(frozen=True)
class PoolViewSpec:
    source_market_dataset_id: str
    view_kind: str
    view_name: str
    start_date: str = ""
    end_date: str = ""
    pool_name: str = ""
    rebalance_every_days: int = 21
    adv_window: int = 20
    min_price: float = 2.0
    max_price: float = 300.0
    exchange_suffix: str = ""
    symbols: tuple[str, ...] = ()
    exclude_symbol_prefixes: tuple[str, ...] = ()
    status_sidecar_dataset_id: str = ""
    require_tradeable: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["view_kind"] = str(payload["view_kind"] or "").strip().lower()
        payload["view_name"] = str(payload["view_name"] or "").strip().lower()
        payload["pool_name"] = str(payload["pool_name"] or "").strip().lower()
        payload["exchange_suffix"] = str(payload["exchange_suffix"] or "").strip().upper()
        payload["symbols"] = tuple(_normalize_symbols(payload.get("symbols", ())))
        payload["exclude_symbol_prefixes"] = tuple(_normalize_symbol_prefixes(payload.get("exclude_symbol_prefixes", ())))
        payload["status_sidecar_dataset_id"] = str(payload.get("status_sidecar_dataset_id", "") or "").strip()
        payload["require_tradeable"] = bool(payload.get("require_tradeable", False))
        payload["rebalance_every_days"] = int(payload["rebalance_every_days"] or 21)
        payload["adv_window"] = int(payload["adv_window"] or 20)
        payload["min_price"] = float(payload["min_price"])
        payload["max_price"] = float(payload["max_price"])
        return payload


@dataclass(frozen=True)
class PoolViewRecord:
    dataset_id: str
    dataset_kind: str
    fingerprint: str
    status: str
    content_paths: dict[str, str]
    metadata: dict[str, Any]
    membership_frame: pd.DataFrame
    schedule_frame: pd.DataFrame
    summary_frame: pd.DataFrame


def _normalize_symbols(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip().upper()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _normalize_symbol_prefixes(values: Iterable[str] | str) -> list[str]:
    if isinstance(values, str):
        raw_values = values.split(",")
    else:
        raw_values = list(values or ())
    out: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        value = str(raw or "").strip().upper()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _symbol_code(symbol: str) -> str:
    return str(symbol or "").strip().upper().split(".", 1)[0]


def _exclude_prefixed_symbols(symbols: Iterable[str], prefixes: Iterable[str]) -> list[str]:
    normalized_prefixes = tuple(_normalize_symbol_prefixes(prefixes))
    if not normalized_prefixes:
        return [str(symbol).strip().upper() for symbol in symbols]
    out: list[str] = []
    for symbol in symbols:
        value = str(symbol or "").strip().upper()
        code = _symbol_code(value)
        if any(code.startswith(prefix) for prefix in normalized_prefixes):
            continue
        out.append(value)
    return out


def _read_market_panel(lake: ResearchDataLake, dataset_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    metadata = lake.describe_dataset(dataset_id)
    if str(metadata.get("dataset_kind", "")) != "policy_input_bundle":
        raise ValueError(f"pool_view_blocker: source dataset is not policy_input_bundle: {dataset_id}")
    market_path = str(dict(metadata.get("content_paths", {}) or {}).get("bronze_market_data", "") or "")
    candidates = sorted(glob.glob(market_path)) if "*" in market_path else ([market_path] if market_path else [])
    existing = [item for item in candidates if Path(item).exists()]
    if not existing:
        raise ValueError(f"pool_view_blocker: source bundle has no bronze_market_data: {dataset_id}")
    market = pd.concat([pd.read_parquet(item) for item in existing], ignore_index=True) if len(existing) > 1 else pd.read_parquet(existing[0])
    if "trade_date" not in market.columns:
        raise ValueError(f"pool_view_blocker: bronze_market_data has no trade_date column: {market_path}")
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    start = pd.Timestamp(start_date or metadata.get("start_date", "") or market["trade_date"].min())
    end = pd.Timestamp(end_date or metadata.get("end_date", "") or market["trade_date"].max())
    market = market.loc[(market["trade_date"] >= start) & (market["trade_date"] <= end)].copy()
    if market.empty:
        raise ValueError(f"pool_view_blocker: no market rows for view window {start.date()} -> {end.date()}")
    market["symbol"] = market["symbol"].astype(str).str.strip().str.upper()
    return market


def _pivot_market(market: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in market.columns:
        raise ValueError(f"pool_view_blocker: bronze_market_data has no {column} column.")
    out = market.pivot(index="trade_date", columns="symbol", values=column).sort_index()
    out.index.name = None
    out.columns.name = None
    return out


def _read_status_sidecar_tradeable(
    lake: ResearchDataLake,
    dataset_id: str,
    *,
    source_market_dataset_id: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    metadata = lake.describe_dataset(str(dataset_id))
    parameters = dict(metadata.get("parameters", {}) or {})
    sidecar_source_id = str(parameters.get("source_market_dataset_id", "") or "")
    if sidecar_source_id and sidecar_source_id != str(source_market_dataset_id):
        raise ValueError(
            "pool_view_blocker: status sidecar source_market_dataset_id mismatch: "
            f"status_sidecar_dataset_id={dataset_id} source_market_dataset_id={sidecar_source_id} "
            f"expected={source_market_dataset_id}"
        )
    paths = dict(metadata.get("content_paths", {}) or {})
    table_paths = _domain_table_paths(paths)
    if not table_paths:
        raise ValueError(f"pool_view_blocker: status sidecar has no silver_domain_data: {dataset_id}")
    frames = [pd.read_parquet(path) for path in table_paths]
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    required = {"trade_date", "symbol", "is_tradeable"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"pool_view_blocker: status sidecar missing columns {missing}: {dataset_id}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    start = pd.Timestamp(start_date or metadata.get("start_date", "") or frame["trade_date"].min())
    end = pd.Timestamp(end_date or metadata.get("end_date", "") or frame["trade_date"].max())
    frame = frame.loc[(frame["trade_date"] >= start) & (frame["trade_date"] <= end)].copy()
    pivot = frame.pivot(index="trade_date", columns="symbol", values="is_tradeable").sort_index()
    pivot.index.name = None
    pivot.columns.name = None
    return pivot.fillna(False).astype(bool)


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


def _metadata_summary(membership: pd.DataFrame, view_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(membership.index).strftime("%Y-%m-%d"),
            "view_name": str(view_name),
            "member_count": membership.sum(axis=1).astype(int).to_numpy(),
        }
    )


def _build_membership_for_spec(
    *,
    lake: ResearchDataLake,
    market: pd.DataFrame,
    spec: PoolViewSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    payload = spec.to_dict()
    view_kind = str(payload["view_kind"])
    view_name = str(payload["view_name"] or view_kind)
    close = _pivot_market(market, "close")
    available_symbols = _exclude_prefixed_symbols(close.columns, payload.get("exclude_symbol_prefixes", ()))
    close = close.reindex(columns=available_symbols)
    dates = close.index
    schedule = pd.DataFrame()
    source_cache: dict[str, Any] = {
        "source_market_dataset_id": payload["source_market_dataset_id"],
        "view_kind": view_kind,
        "view_name": view_name,
        "exclude_symbol_prefixes": list(payload.get("exclude_symbol_prefixes", ())),
    }
    status_tradeable = pd.DataFrame(True, index=dates, columns=available_symbols)
    status_sidecar_dataset_id = str(payload.get("status_sidecar_dataset_id", "") or "")
    if status_sidecar_dataset_id:
        status_tradeable = _read_status_sidecar_tradeable(
            lake,
            status_sidecar_dataset_id,
            source_market_dataset_id=str(payload["source_market_dataset_id"]),
            start_date=str(payload.get("start_date", "") or ""),
            end_date=str(payload.get("end_date", "") or ""),
        ).reindex(index=dates, columns=available_symbols, fill_value=False)
        source_cache["status_sidecar_dataset_id"] = status_sidecar_dataset_id
        source_cache["status_tradeable_true_cells"] = int(status_tradeable.to_numpy(dtype=bool).sum())

    if view_kind in {"learned_all_a", "all_a"}:
        membership = close.notna().astype(bool)
    elif view_kind == "tradeable_mainboard":
        if not status_sidecar_dataset_id:
            raise ValueError("pool_view_blocker: tradeable_mainboard requires status_sidecar_dataset_id")
        membership = (close.notna() & status_tradeable.reindex(index=dates, columns=available_symbols, fill_value=False)).astype(bool)
        daily_counts = membership.sum(axis=1).astype(int)
        source_cache["tradeable_mainboard_summary"] = {
            "true_cells": int(membership.to_numpy(dtype=bool).sum()),
            "active_symbol_count": int((membership.sum(axis=0) > 0).sum()),
            "min_daily_member_count": int(daily_counts.min()) if len(daily_counts) else 0,
            "max_daily_member_count": int(daily_counts.max()) if len(daily_counts) else 0,
            "mean_daily_member_count": float(daily_counts.mean()) if len(daily_counts) else 0.0,
        }
    elif view_kind in {"rolling_liquidity", "rolling_liquidity_tradeable_mainboard"}:
        amount = _pivot_market(market, "amount").reindex(columns=available_symbols)
        close_for_pool = close
        amount_for_pool = amount
        if view_kind == "rolling_liquidity_tradeable_mainboard":
            if not status_sidecar_dataset_id:
                raise ValueError("pool_view_blocker: rolling_liquidity_tradeable_mainboard requires status_sidecar_dataset_id")
            close_for_pool = close.where(status_tradeable)
            amount_for_pool = amount.where(status_tradeable)
        artifact = build_rolling_liquidity_membership(
            close_frame=close_for_pool,
            amount_frame=amount_for_pool,
            pool_name=str(payload["pool_name"] or payload["view_name"]).replace("rolling_", ""),
            signal_start_date=str(payload["start_date"] or ""),
            signal_end_date=str(payload["end_date"] or ""),
            rebalance_every_days=int(payload["rebalance_every_days"]),
            adv_window=int(payload["adv_window"]),
            min_price=float(payload["min_price"]),
            max_price=float(payload["max_price"]),
        )
        membership = artifact.membership_frame.reindex(index=dates, columns=available_symbols, fill_value=False).astype(bool)
        if view_kind == "rolling_liquidity_tradeable_mainboard":
            membership = (membership & status_tradeable.reindex(index=dates, columns=available_symbols, fill_value=False)).astype(bool)
        schedule = artifact.schedule_df
        source_cache["rolling_pool_summary"] = {
            "pool_name": artifact.pool_name,
            "pool_size": int(artifact.pool_size),
            "rebalance_every_days": int(artifact.rebalance_every_days),
            "adv_window": int(artifact.adv_window),
        }
    elif view_kind == "exchange":
        suffix = str(payload["exchange_suffix"] or "").upper()
        if suffix and not suffix.startswith("."):
            suffix = f".{suffix}"
        if suffix not in {".SH", ".SZ"}:
            raise ValueError(f"pool_view_blocker: unsupported exchange suffix: {suffix}")
        keep = [symbol for symbol in available_symbols if symbol.upper().endswith(suffix)]
        membership = pd.DataFrame(False, index=dates, columns=available_symbols)
        if keep:
            membership.loc[:, keep] = close.reindex(columns=keep).notna()
    elif view_kind == "static_symbols":
        keep = [symbol for symbol in _normalize_symbols(payload["symbols"]) if symbol in set(available_symbols)]
        membership = pd.DataFrame(False, index=dates, columns=available_symbols)
        if keep:
            membership.loc[:, keep] = close.reindex(columns=keep).notna()
    else:
        raise ValueError(f"pool_view_blocker: unsupported view_kind: {view_kind}")

    membership = membership.fillna(False).astype(bool)
    if int(membership.to_numpy(dtype=bool).sum()) <= 0:
        raise ValueError(f"pool_view_blocker: view produced empty membership: {view_name}")
    summary = _metadata_summary(membership, view_name)
    return membership, schedule, summary, source_cache


def build_pool_view_from_policy_bundle(
    *,
    lake: ResearchDataLake,
    spec: PoolViewSpec | Mapping[str, Any],
    reuse: bool = True,
) -> LakeDatasetRecord:
    resolved = spec if isinstance(spec, PoolViewSpec) else PoolViewSpec(**dict(spec))
    payload = resolved.to_dict()
    market = _read_market_panel(
        lake,
        dataset_id=str(payload["source_market_dataset_id"]),
        start_date=str(payload.get("start_date", "") or ""),
        end_date=str(payload.get("end_date", "") or ""),
    )
    membership, schedule, summary, source_cache = _build_membership_for_spec(lake=lake, market=market, spec=resolved)
    return lake.save_pool_view(
        spec=payload,
        membership_frame=membership,
        schedule_frame=schedule,
        summary_frame=summary,
        source_cache=source_cache,
        reuse=bool(reuse),
    )


def load_pool_view(*, lake: ResearchDataLake, pool_view_id: str) -> PoolViewRecord:
    metadata = lake.describe_dataset(str(pool_view_id))
    if str(metadata.get("dataset_kind", "")) != "policy_pool_view":
        raise ValueError(f"pool_view_blocker: dataset is not policy_pool_view: {pool_view_id}")
    paths = dict(metadata.get("content_paths", {}) or {})
    membership_path = Path(str(paths.get("membership_frame", "") or ""))
    if not membership_path.exists():
        raise ValueError(f"pool_view_blocker: missing membership frame: {membership_path}")
    membership_raw = pd.read_parquet(membership_path)
    date_col = "date" if "date" in membership_raw.columns else membership_raw.columns[0]
    membership_raw[date_col] = pd.to_datetime(membership_raw[date_col])
    membership = membership_raw.set_index(date_col).sort_index().fillna(False).astype(bool)
    membership.index.name = None
    schedule_path = Path(str(paths.get("schedule_frame", "") or ""))
    summary_path = Path(str(paths.get("summary_frame", "") or ""))
    schedule = pd.read_parquet(schedule_path) if schedule_path.exists() else pd.DataFrame()
    summary = pd.read_parquet(summary_path) if summary_path.exists() else pd.DataFrame()
    return PoolViewRecord(
        dataset_id=str(metadata["dataset_id"]),
        dataset_kind=str(metadata["dataset_kind"]),
        fingerprint=str(metadata["fingerprint"]),
        status=str(metadata.get("status", "") or "loaded"),
        content_paths=paths,
        metadata=metadata,
        membership_frame=membership,
        schedule_frame=schedule,
        summary_frame=summary,
    )


def resolve_pool_view_for_policy_inputs(
    *,
    lake: ResearchDataLake,
    dataset_id: str,
    pool_view_id: str = "",
    pool_view_spec: PoolViewSpec | Mapping[str, Any] | None = None,
) -> PoolViewRecord | None:
    if str(pool_view_id or "").strip():
        return load_pool_view(lake=lake, pool_view_id=str(pool_view_id).strip())
    if pool_view_spec is None:
        return None
    payload = pool_view_spec if isinstance(pool_view_spec, PoolViewSpec) else PoolViewSpec(**dict(pool_view_spec))
    spec_dict = payload.to_dict()
    if not str(spec_dict.get("source_market_dataset_id", "") or "").strip():
        spec_dict["source_market_dataset_id"] = str(dataset_id)
        payload = PoolViewSpec(**spec_dict)
    record = build_pool_view_from_policy_bundle(lake=lake, spec=payload)
    return load_pool_view(lake=lake, pool_view_id=record.dataset_id)
