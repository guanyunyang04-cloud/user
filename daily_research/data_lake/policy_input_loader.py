from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.baseline.advanced_ml_runtime import HistoryWindow
from daily_research.continuous_policy.state_builder import (
    STATE_SEQUENCE_BASES,
    STATE_SEQUENCE_LAGS,
    DEFAULT_SCORE_BLEND_WEIGHTS,
    PreparedPolicyInputs,
    _build_alpha_prior_frames,
    _safe_pct_change,
)
from daily_research.data_lake.catalog import ResearchDataLake
from daily_research.data_lake.canonical import DEFAULT_CANONICAL_ALIAS, resolve_canonical_dataset_id
from daily_research.data_lake.pool_views import PoolViewSpec, resolve_pool_view_for_policy_inputs
from daily_research.data_lake.sector_board_views import SectorBoardViewSpec, resolve_sector_board_view_for_policy_inputs
from daily_research.data_platform.contracts import DataDomain

LEGACY_DEFAULT_POLICY_INPUT_LAKE_DATASET_ID = "policy_input_bundle__7c8f58d851bce8179e1e9e2d"
DEFAULT_POLICY_INPUT_LAKE_DATASET_ID = DEFAULT_CANONICAL_ALIAS


def resolve_policy_input_dataset_id(lake: ResearchDataLake, dataset_id: str = "") -> str:
    requested = str(dataset_id or "").strip()
    if requested and requested != DEFAULT_CANONICAL_ALIAS:
        return requested
    canonical_id = resolve_canonical_dataset_id(lake, alias=DEFAULT_CANONICAL_ALIAS)
    return canonical_id or LEGACY_DEFAULT_POLICY_INPUT_LAKE_DATASET_ID


def _read_feature_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "trade_date" in frame.columns:
        frame = frame.rename(columns={"trade_date": "date"})
    if "date" not in frame.columns:
        raise ValueError(f"Feature panel has no date/trade_date column: {path}")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date").sort_index()


def _slice_wide(frame: pd.DataFrame, *, universe: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    out = frame.copy()
    out.index = pd.to_datetime(out.index)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    out = out.loc[(out.index >= start_ts) & (out.index <= end_ts)]
    return out.reindex(columns=universe).sort_index()


def _load_feature_panels(feature_glob: str) -> dict[str, pd.DataFrame]:
    pattern = Path(str(feature_glob).replace("*.parquet", ""))
    if pattern.name:
        feature_dir = pattern
    else:
        feature_dir = pattern.parent
    if str(feature_glob).endswith("*.parquet"):
        feature_dir = Path(str(feature_glob)).parent
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(feature_dir.glob("*.parquet")):
        frames[path.stem] = _read_feature_panel(path)
    return frames


def _read_domain_sidecar_frame(lake: ResearchDataLake, dataset_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not str(dataset_id or "").strip():
        return pd.DataFrame(), {}
    metadata = lake.describe_dataset(str(dataset_id))
    paths = dict(metadata.get("content_paths", {}) or {})
    frame = _read_sidecar_table_paths(paths)
    return frame, metadata


def _read_domain_sidecar_frame_filtered(
    lake: ResearchDataLake,
    dataset_id: str,
    *,
    start_date: str,
    end_date: str,
    columns: list[str] | None = None,
    symbols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not str(dataset_id or "").strip():
        return pd.DataFrame(), {}
    metadata = lake.describe_dataset(str(dataset_id))
    paths = dict(metadata.get("content_paths", {}) or {})
    frame = _read_table_paths_filtered(
        paths,
        data_key="silver_domain_data",
        manifest_keys=("shard_manifest",),
        start_date=start_date,
        end_date=end_date,
        columns=columns,
        symbols=symbols,
    )
    return frame, metadata


def _read_sidecar_table_paths(paths: dict[str, Any]) -> pd.DataFrame:
    return _read_table_paths(paths, data_key="silver_domain_data", manifest_keys=("shard_manifest",))


def _read_table_paths(
    paths: dict[str, Any],
    *,
    data_key: str,
    manifest_keys: tuple[str, ...],
) -> pd.DataFrame:
    raw = str(paths.get(data_key, "") or "")
    if raw:
        candidates = sorted(glob.glob(raw)) if "*" in raw else [raw]
        existing = [path for path in candidates if Path(path).exists()]
        if existing:
            frames = [pd.read_parquet(path) for path in existing]
            return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    for manifest_key in manifest_keys:
        manifest_path = Path(str(paths.get(manifest_key, "") or ""))
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard_paths = [
            str(dict(item).get("path", "") or "")
            for item in list(manifest.get("shards", []) or [])
            if str(dict(item).get("status", "") or "") in {"stored", "skipped"}
            and int(dict(item).get("row_count", 0) or 0) > 0
        ]
        existing = [path for path in shard_paths if Path(path).exists()]
        if existing:
            frames = [pd.read_parquet(path) for path in existing]
            return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return pd.DataFrame()


def _read_table_paths_filtered(
    paths: dict[str, Any],
    *,
    data_key: str,
    manifest_keys: tuple[str, ...],
    start_date: str,
    end_date: str,
    columns: list[str] | None = None,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    candidates: list[str] = []
    for manifest_key in manifest_keys:
        manifest_raw = str(paths.get(manifest_key, "") or "").strip()
        if not manifest_raw:
            continue
        manifest_path = Path(manifest_raw)
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in list(manifest.get("shards", []) or []):
            row = dict(item)
            if str(row.get("status", "") or "") not in {"stored", "skipped"}:
                continue
            if int(row.get("row_count", 0) or 0) <= 0:
                continue
            if not _shard_date_overlaps(row, start_ts=start_ts, end_ts=end_ts):
                continue
            if symbols and not _shard_symbol_overlaps(row, symbols=symbols):
                continue
            path = str(row.get("path", "") or "")
            if path:
                candidates.append(path)
        if candidates:
            break
    raw = str(paths.get(data_key, "") or "")
    if not candidates and raw:
        candidates.extend(sorted(glob.glob(raw)) if "*" in raw else [raw])
    requested_columns = list(dict.fromkeys(str(column) for column in list(columns or []) if str(column).strip()))
    if len(candidates) > 1:
        fast = _read_table_candidates_duckdb(
            candidates,
            start_date=start_ts.strftime("%Y-%m-%d"),
            end_date=end_ts.strftime("%Y-%m-%d"),
            columns=requested_columns,
            symbols=symbols,
        )
        if fast is not None:
            return fast
    frames: list[pd.DataFrame] = []
    for path in candidates:
        parquet_path = Path(path)
        if not parquet_path.exists():
            continue
        try:
            frame = pd.read_parquet(parquet_path, columns=requested_columns or None)
        except Exception as exc:
            message = str(exc).lower()
            if not requested_columns or not any(token in message for token in ("no match", "not found", "missing", "fieldref")):
                raise
            frame = pd.read_parquet(parquet_path)
            if requested_columns:
                frame = frame.reindex(columns=[column for column in requested_columns if column in frame.columns])
        if "trade_date" in frame.columns:
            frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
            frame = frame.loc[(frame["trade_date"] >= start_ts) & (frame["trade_date"] <= end_ts)].copy()
        elif "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.loc[(frame["date"] >= start_ts) & (frame["date"] <= end_ts)].copy()
        if not frame.empty:
            if symbols and "symbol" in frame.columns:
                allowed = {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}
                frame = frame.loc[frame["symbol"].astype(str).str.strip().str.upper().isin(allowed)].copy()
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else (frames[0] if frames else pd.DataFrame())


def _read_table_candidates_duckdb(
    candidates: list[str],
    *,
    start_date: str,
    end_date: str,
    columns: list[str],
    symbols: list[str] | None,
) -> pd.DataFrame | None:
    try:
        import duckdb
    except Exception:
        return None
    select_expr = ", ".join(_quote_sql_identifier(column) for column in columns) if columns else "*"
    where = ["CAST(trade_date AS DATE) >= CAST(? AS DATE)", "CAST(trade_date AS DATE) <= CAST(? AS DATE)"]
    params: list[Any] = [candidates, start_date, end_date]
    normalized_symbols = [str(symbol).strip().upper() for symbol in list(symbols or []) if str(symbol).strip()]
    if normalized_symbols:
        where.append("upper(symbol) in (select upper(x) from unnest(?) as t(x))")
        params.append(normalized_symbols)
    sql = f"select {select_expr} from read_parquet(?, union_by_name=true) where {' and '.join(where)}"
    try:
        con = duckdb.connect(":memory:")
        return con.execute(sql, params).fetchdf()
    except Exception as exc:
        message = str(exc).lower()
        if columns and any(token in message for token in ("no match", "not found", "missing", "fieldref")):
            return None
        raise


def _quote_sql_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _shard_date_overlaps(row: dict[str, Any], *, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> bool:
    row_start = str(row.get("start_date", "") or row.get("date_start", "") or "").strip()
    row_end = str(row.get("end_date", "") or row.get("date_end", "") or "").strip()
    if not row_start and not row_end:
        return True
    shard_start = pd.Timestamp(row_start) if row_start else pd.Timestamp.min
    shard_end = pd.Timestamp(row_end) if row_end else pd.Timestamp.max
    return bool(shard_start <= end_ts and shard_end >= start_ts)


def _shard_symbol_overlaps(row: dict[str, Any], *, symbols: list[str]) -> bool:
    requested = {str(symbol).strip().upper() for symbol in list(symbols or []) if str(symbol).strip()}
    if not requested:
        return True
    parsed: set[str] = set()
    for key in ("symbols", "symbol_sample", "symbols_sample", "source_symbols", "source_members_sample"):
        value = row.get(key)
        if isinstance(value, (list, tuple, set)):
            for item in value:
                symbol = _symbol_from_manifest_token(item)
                if symbol:
                    parsed.add(symbol)
        elif isinstance(value, str):
            symbol = _symbol_from_manifest_token(value)
            if symbol:
                parsed.add(symbol)
    if not parsed:
        return True
    member_count = int(row.get("source_member_count", 0) or row.get("symbol_count", 0) or 0)
    if member_count and len(parsed) < member_count:
        return True
    return bool(parsed & requested)


def _symbol_from_manifest_token(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    name = Path(text.replace("\\", "/")).name
    stem = name.rsplit(".", 1)[0].strip().upper()
    if not stem:
        return ""
    if "." in stem:
        left, right = stem.split(".", 1)
        if left.isdigit() and right in {"SH", "SZ", "BJ"}:
            return f"{left.zfill(6)}.{right}"
    lower = stem.lower()
    if len(lower) >= 8 and lower[:2] in {"sh", "sz", "bj"} and lower[2:8].isdigit():
        suffix = {"sh": "SH", "sz": "SZ", "bj": "BJ"}[lower[:2]]
        return f"{lower[2:8]}.{suffix}"
    if len(stem) == 6 and stem.isdigit():
        if stem.startswith(("60", "68", "90")):
            return f"{stem}.SH"
        if stem.startswith(("00", "30", "20")):
            return f"{stem}.SZ"
        if stem.startswith(("43", "83", "87", "88", "92")):
            return f"{stem}.BJ"
    return ""


def _sidecar_dataset_ids(metadata: dict[str, Any]) -> dict[str, str]:
    parameters = dict(metadata.get("parameters", {}) or {})
    sidecars = dict(parameters.get("sidecar_dataset_ids", {}) or {})
    return {str(key): str(value) for key, value in sidecars.items() if str(value or "").strip()}


def _load_valuation_sidecar_frames(
    *,
    lake: ResearchDataLake,
    metadata: dict[str, Any],
    universe: list[str],
    dates: pd.Index,
    start_date: str,
    end_date: str,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    sidecar_id = _sidecar_dataset_ids(metadata).get(DataDomain.VALUATION, "")
    frame, sidecar_metadata = _read_domain_sidecar_frame(lake, sidecar_id)
    if frame.empty:
        return {}, pd.DataFrame(), {"dataset_id": str(sidecar_id), "available": False, "reason": "missing_or_empty"}
    data = frame.copy()
    if "trade_date" not in data.columns or "symbol" not in data.columns:
        return {}, pd.DataFrame(), {"dataset_id": str(sidecar_id), "available": False, "reason": "missing_trade_date_or_symbol"}
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["symbol"] = data["symbol"].astype(str).str.strip().str.upper()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    data = data.loc[(data["trade_date"] >= start) & (data["trade_date"] <= end) & data["symbol"].isin(set(universe))].copy()
    if data.empty:
        return {}, pd.DataFrame(), {"dataset_id": str(sidecar_id), "available": False, "reason": "no_overlap"}
    metric_fields = [column for column in ("turn", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM") if column in data.columns]
    frames: dict[str, pd.DataFrame] = {}
    for column in metric_fields:
        wide = data.pivot(index="trade_date", columns="symbol", values=column).sort_index()
        wide.index.name = None
        wide.columns.name = None
        frames[column] = wide.reindex(index=dates, columns=universe)
    row_count = max(int(len(data)), 1)
    coverage = {
        column: float(pd.to_numeric(data[column], errors="coerce").notna().sum() / row_count)
        for column in metric_fields
    }
    summary = {
        "dataset_id": str(sidecar_id),
        "available": bool(frames),
        "dataset_kind": str(sidecar_metadata.get("dataset_kind", "")),
        "metric_fields": metric_fields,
        "row_count": int(len(data)),
        "symbol_count": int(data["symbol"].nunique()),
        "trade_date_count": int(data["trade_date"].nunique()),
        "coverage": coverage,
        "valuation_lag_policy": str(dict(sidecar_metadata.get("parameters", {}) or {}).get("valuation_lag_policy", "")),
    }
    return frames, data.reset_index(drop=True), summary


def _load_industry_sidecar_metadata(
    *,
    lake: ResearchDataLake,
    metadata: dict[str, Any],
    universe: list[str],
    start_date: str,
    end_date: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    sidecar_id = _sidecar_dataset_ids(metadata).get(DataDomain.INDUSTRY_CONCEPT, "")
    frame, sidecar_metadata = _read_domain_sidecar_frame(lake, sidecar_id)
    if frame.empty:
        return {}, {"dataset_id": str(sidecar_id), "available": False, "reason": "missing_or_empty"}
    data = frame.copy()
    if "trade_date" not in data.columns or "symbol" not in data.columns or "industry" not in data.columns:
        return {}, {"dataset_id": str(sidecar_id), "available": False, "reason": "missing_required_columns"}
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["symbol"] = data["symbol"].astype(str).str.strip().str.upper()
    data["industry"] = data["industry"].fillna("").astype(str).str.strip()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    data = data.loc[
        (data["trade_date"] >= start)
        & (data["trade_date"] <= end)
        & data["symbol"].isin(set(universe))
        & data["industry"].ne("")
    ].copy()
    if data.empty:
        return {}, {"dataset_id": str(sidecar_id), "available": False, "reason": "no_overlap"}
    data = data.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    latest = data.drop_duplicates(subset=["symbol"], keep="last").copy()
    latest["as_of_date"] = latest["trade_date"].dt.strftime("%Y-%m-%d")
    latest["trade_date"] = latest["as_of_date"]
    summary = {
        "dataset_id": str(sidecar_id),
        "available": True,
        "dataset_kind": str(sidecar_metadata.get("dataset_kind", "")),
        "row_count": int(len(data)),
        "symbol_count": int(data["symbol"].nunique()),
        "trade_date_count": int(data["trade_date"].nunique()),
        "industry_count": int(data["industry"].nunique()),
        "coverage_ratio": float(data["symbol"].nunique() / max(len(universe), 1)),
        "industry_frequency": str(dict(sidecar_metadata.get("parameters", {}) or {}).get("industry_frequency", "")),
        "pit_semantics": str(dict(sidecar_metadata.get("parameters", {}) or {}).get("pit_semantics", "")),
    }
    data["trade_date"] = data["trade_date"].dt.strftime("%Y-%m-%d")
    return {"industry_daily": data.reset_index(drop=True), "industry_map": latest.reset_index(drop=True)}, summary


def _load_intraday_daily_feature_sidecar_frames(
    *,
    lake: ResearchDataLake,
    metadata: dict[str, Any],
    universe: list[str],
    dates: pd.Index,
    start_date: str,
    end_date: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    sidecar_id = _sidecar_dataset_ids(metadata).get(DataDomain.INTRADAY_DAILY_FEATURES, "")
    fields = [
        "first_5m_ret",
        "first_15m_ret",
        "first_30m_ret",
        "first_30m_amount_share",
        "open_gap",
        "open_gap_first_30m_follow_through",
        "open_gap_first_30m_reversal",
        "last_5m_ret",
        "last_30m_ret",
        "last_30m_amount_share",
        "intraday_ret",
        "intraday_vwap",
        "close_to_vwap",
        "intraday_range",
        "close_position",
        "intraday_realized_vol",
        "intraday_price_volume_corr",
        "bar_count",
        "high_time_frac",
        "low_time_frac",
        "high_before_low",
        "open_to_high_ret",
        "open_to_low_ret",
        "high_to_close_ret",
        "low_to_close_ret",
        "intraday_max_drawdown",
        "intraday_max_runup",
        "price_above_vwap_share",
        "cum_vwap_slope",
        "first_5m_amount_share",
        "last_5m_amount_share",
        "first_30m_range",
        "last_30m_range",
        "amount_top_bar_share",
        "amount_concentration_hhi",
        "lunch_gap_ret",
        "am_ret",
        "pm_ret",
        "am_pm_ret_spread",
        "am_pm_vol_spread",
        "am_amount_share",
        "am_pm_amount_spread",
        "early_strength_late_weak",
        "close_pressure_30m",
    ]
    frame, sidecar_metadata = _read_domain_sidecar_frame_filtered(
        lake,
        sidecar_id,
        start_date=start_date,
        end_date=end_date,
        columns=["symbol", "trade_date", *fields],
        symbols=universe,
    )
    if frame.empty:
        return {}, {"dataset_id": str(sidecar_id), "available": False, "reason": "missing_or_empty"}
    data = frame.copy()
    if "trade_date" not in data.columns or "symbol" not in data.columns:
        return {}, {"dataset_id": str(sidecar_id), "available": False, "reason": "missing_trade_date_or_symbol"}
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["symbol"] = data["symbol"].astype(str).str.strip().str.upper()
    data = data.loc[data["trade_date"].notna() & data["symbol"].isin(set(universe))].copy()
    if data.empty:
        return {}, {"dataset_id": str(sidecar_id), "available": False, "reason": "no_overlap"}
    frames: dict[str, pd.DataFrame] = {}
    for column in fields:
        if column not in data.columns:
            continue
        data[column] = pd.to_numeric(data[column], errors="coerce")
        wide = data.pivot_table(index="trade_date", columns="symbol", values=column, aggfunc="last").sort_index()
        wide.index.name = None
        wide.columns.name = None
        frames[f"intraday_daily_features_{column}"] = wide.reindex(index=dates, columns=universe).astype("float32")
    summary = {
        "dataset_id": str(sidecar_id),
        "available": bool(frames),
        "dataset_kind": str(sidecar_metadata.get("dataset_kind", "")),
        "field_count": int(len(frames)),
        "row_count": int(len(data)),
        "symbol_count": int(data["symbol"].nunique()),
        "trade_date_count": int(data["trade_date"].nunique()),
    }
    return frames, summary


def _load_adjust_factor_sidecar_frames(
    *,
    lake: ResearchDataLake,
    metadata: dict[str, Any],
    universe: list[str],
    dates: pd.Index,
    start_date: str,
    end_date: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    sidecar_id = _sidecar_dataset_ids(metadata).get(DataDomain.ADJUST_FACTOR, "")
    fields = ["fore_adjust_factor", "back_adjust_factor", "adjust_factor"]
    frame, sidecar_metadata = _read_domain_sidecar_frame_filtered(
        lake,
        sidecar_id,
        start_date=start_date,
        end_date=end_date,
        columns=["symbol", "trade_date", *fields],
        symbols=universe,
    )
    if frame.empty:
        return {}, {"dataset_id": str(sidecar_id), "available": False, "reason": "missing_or_empty"}
    data = frame.copy()
    if "trade_date" not in data.columns or "symbol" not in data.columns:
        return {}, {"dataset_id": str(sidecar_id), "available": False, "reason": "missing_trade_date_or_symbol"}
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["symbol"] = data["symbol"].astype(str).str.strip().str.upper()
    data = data.loc[data["trade_date"].notna() & data["symbol"].isin(set(universe))].copy()
    if data.empty:
        return {}, {"dataset_id": str(sidecar_id), "available": False, "reason": "no_overlap"}
    frames: dict[str, pd.DataFrame] = {}
    for column in fields:
        if column not in data.columns:
            continue
        data[column] = pd.to_numeric(data[column], errors="coerce")
        wide = data.pivot_table(index="trade_date", columns="symbol", values=column, aggfunc="last").sort_index()
        wide.index.name = None
        wide.columns.name = None
        frames[f"adjust_factor_{column}"] = wide.reindex(index=dates, columns=universe).ffill().astype("float32")
    if "adjust_factor_fore_adjust_factor" in frames and "adjust_factor" not in frames:
        frames["adjust_factor"] = frames["adjust_factor_fore_adjust_factor"]
    summary = {
        "dataset_id": str(sidecar_id),
        "available": bool(frames),
        "dataset_kind": str(sidecar_metadata.get("dataset_kind", "")),
        "field_count": int(len(frames)),
        "row_count": int(len(data)),
        "symbol_count": int(data["symbol"].nunique()),
        "trade_date_count": int(data["trade_date"].nunique()),
    }
    return frames, summary


def _normalize_symbol_list(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip().upper()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _resolve_lake_universe(
    *,
    available_symbols: list[str],
    requested_symbols: list[str] | None,
    extra_stocks: list[str] | None,
    max_universe_size: int,
) -> list[str]:
    available = _normalize_symbol_list(available_symbols)
    available_set = set(available)
    requested = _normalize_symbol_list(requested_symbols or [])
    extras = _normalize_symbol_list(extra_stocks or [])
    if requested:
        universe = [stock for stock in requested if stock in available_set]
        missing_requested = [stock for stock in requested if stock not in available_set]
    else:
        universe = list(available)
        missing_requested = []
    if int(max_universe_size or 0) > 0:
        capped = universe[: int(max_universe_size)]
        universe = _normalize_symbol_list([*capped, *extras])
    else:
        universe = _normalize_symbol_list([*universe, *extras])
    missing_extras = [stock for stock in extras if stock not in available_set]
    universe = [stock for stock in universe if stock in available_set]
    if not universe:
        raise ValueError(
            "lake_coverage_blocker: no requested symbols are available in lake market data; "
            f"missing_requested={missing_requested[:10]}, missing_extras={missing_extras[:10]}"
        )
    return universe


def _lake_coverage_report(
    *,
    close: pd.DataFrame,
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    amount: pd.DataFrame,
    benchmark_close: pd.Series,
    benchmark_open: pd.Series,
    benchmark_open_source: str,
    membership_frame: pd.DataFrame,
    start_date: str,
    end_date: str,
    dataset_id: str,
    min_trading_days: int,
    require_benchmark_open: bool = False,
) -> dict[str, Any]:
    price_frames = {
        "close": close,
        "open": open_,
        "high": high,
        "low": low,
        "volume": volume,
        "amount": amount,
    }
    missing_fields = [
        name
        for name, frame in price_frames.items()
        if frame.empty or int(frame.notna().sum().sum()) <= 0
    ]
    frame_nan_cells = {
        name: int(frame.isna().sum().sum()) if not frame.empty else 0
        for name, frame in price_frames.items()
    }
    trading_days = int(len(close.index))
    membership_true_rows = int(membership_frame.any(axis=1).sum()) if not membership_frame.empty else 0
    blockers: list[str] = []
    if trading_days < int(min_trading_days or 1):
        blockers.append("insufficient_trading_days")
    if missing_fields:
        blockers.append("missing_market_fields")
    if benchmark_close.empty or int(benchmark_close.notna().sum()) < max(1, trading_days):
        blockers.append("missing_benchmark")
    benchmark_open_rows = int(benchmark_open.notna().sum()) if not benchmark_open.empty else 0
    if bool(require_benchmark_open):
        if str(benchmark_open_source) != "silver_benchmark.open" or benchmark_open_rows < max(1, trading_days):
            blockers.append("missing_benchmark_open")
    if membership_frame.empty or membership_true_rows <= 0:
        blockers.append("missing_membership")
    return {
        "status": "ok" if not blockers else "lake_coverage_blocker",
        "dataset_id": str(dataset_id),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "trading_days": trading_days,
        "min_trading_days": int(min_trading_days or 1),
        "universe_size": int(len(close.columns)),
        "missing_fields": missing_fields,
        "market_nan_cells": frame_nan_cells,
        "benchmark_rows": int(benchmark_close.notna().sum()) if not benchmark_close.empty else 0,
        "benchmark_close_rows": int(benchmark_close.notna().sum()) if not benchmark_close.empty else 0,
        "benchmark_open_rows": benchmark_open_rows,
        "benchmark_open_source": str(benchmark_open_source),
        "require_benchmark_open": bool(require_benchmark_open),
        "membership_rows": int(len(membership_frame.index)),
        "membership_true_rows": membership_true_rows,
        "blockers": blockers,
    }


def _raise_lake_coverage_blocker(report: dict[str, Any]) -> None:
    if str(report.get("status", "")) == "ok":
        return
    raise ValueError(
        "lake_coverage_blocker: "
        f"dataset_id={report.get('dataset_id')} "
        f"start_date={report.get('start_date')} end_date={report.get('end_date')} "
        f"blockers={report.get('blockers')} missing_fields={report.get('missing_fields')} "
        f"trading_days={report.get('trading_days')} benchmark_rows={report.get('benchmark_rows')} "
        f"membership_true_rows={report.get('membership_true_rows')}"
    )


def load_policy_inputs_from_lake(
    *,
    lake: ResearchDataLake,
    dataset_id: str,
    start_date: str,
    end_date: str,
    pool_name: str = "",
    benchmark: str = "000300.SH",
    universe: list[str] | None = None,
    pool_view_id: str = "",
    pool_view_spec: PoolViewSpec | dict[str, Any] | None = None,
    sector_board_view_id: str = "",
    sector_board_view_spec: SectorBoardViewSpec | dict[str, Any] | None = None,
    extra_stocks: list[str] | None = None,
    max_universe_size: int = 0,
    min_trading_days: int = 2,
    alpha_prior_source: str = "none",
    alpha_prior_score_panel: str = "",
    alpha_prior_target_weight_panel: str = "",
    require_benchmark_open: bool = False,
) -> PreparedPolicyInputs:
    dataset_id = resolve_policy_input_dataset_id(lake, dataset_id)
    metadata = lake.describe_dataset(dataset_id)
    start_date = str(start_date or metadata.get("start_date", "") or "").strip()
    end_date = str(end_date or metadata.get("end_date", "") or start_date).strip()
    if not start_date or not end_date:
        raise ValueError(f"lake_coverage_blocker: start/end date is required for lake dataset {dataset_id}")
    paths = dict(metadata.get("content_paths", {}) or {})
    market_path = str(paths.get("bronze_market_data", "") or "")
    if not market_path:
        raise ValueError(f"lake_coverage_blocker: dataset has no bronze_market_data path: {dataset_id}")
    market = _read_table_paths_filtered(
        paths,
        data_key="bronze_market_data",
        manifest_keys=("bronze_market_shard_manifest", "shard_manifest"),
        start_date=start_date,
        end_date=end_date,
        columns=["trade_date", "symbol", "open", "high", "low", "close", "volume", "amount"],
    )
    if market.empty:
        raise ValueError(f"lake_coverage_blocker: no readable bronze_market_data rows for lake dataset {dataset_id}")
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    all_trade_dates = pd.Index(sorted(pd.to_datetime(market["trade_date"].dropna().unique())))
    market = market.loc[(market["trade_date"] >= start_ts) & (market["trade_date"] <= end_ts)].copy()
    if market.empty:
        raise ValueError(f"lake_coverage_blocker: no market rows in lake dataset {dataset_id} for {start_date} -> {end_date}")
    available_symbols = sorted(str(item).strip().upper() for item in market["symbol"].dropna().astype(str).unique())
    pool_view = resolve_pool_view_for_policy_inputs(
        lake=lake,
        dataset_id=dataset_id,
        pool_view_id=str(pool_view_id or ""),
        pool_view_spec=pool_view_spec,
    )
    sector_board_view = resolve_sector_board_view_for_policy_inputs(
        lake=lake,
        dataset_id=dataset_id,
        sector_board_view_id=str(sector_board_view_id or ""),
        sector_board_view_spec=sector_board_view_spec,
    )
    if pool_view is not None:
        view_source_dataset_id = str(pool_view.metadata.get("parameters", {}).get("source_market_dataset_id", "") or "")
        if view_source_dataset_id and view_source_dataset_id != dataset_id:
            raise ValueError(
                "pool_view_source_mismatch: "
                f"pool_view_id={pool_view.dataset_id} source_market_dataset_id={view_source_dataset_id} "
                f"requested_dataset_id={dataset_id}"
            )
    if sector_board_view is not None:
        sector_source_dataset_id = str(sector_board_view.metadata.get("parameters", {}).get("source_market_dataset_id", "") or "")
        if sector_source_dataset_id and sector_source_dataset_id != dataset_id:
            raise ValueError(
                "sector_board_view_source_mismatch: "
                f"sector_board_view_id={sector_board_view.dataset_id} source_market_dataset_id={sector_source_dataset_id} "
                f"requested_dataset_id={dataset_id}"
            )
    if pool_view is not None:
        available_set = set(available_symbols)
        view_membership = pool_view.membership_frame.copy()
        view_membership.index = pd.to_datetime(view_membership.index)
        view_columns = [str(column).strip().upper() for column in view_membership.columns]
        view_membership.columns = view_columns
        view_membership = _slice_wide(
            view_membership,
            universe=[symbol for symbol in view_columns if symbol in available_set],
            start_date=start_date,
            end_date=end_date,
        ).fillna(False).astype(bool)
        requested_symbols = [symbol for symbol in view_membership.columns if bool(view_membership[symbol].any())]
    else:
        view_membership = pd.DataFrame()
        requested_symbols = universe
    resolved_universe = _resolve_lake_universe(
        available_symbols=available_symbols,
        requested_symbols=requested_symbols,
        extra_stocks=extra_stocks,
        max_universe_size=0 if pool_view is not None else max_universe_size,
    )
    market["symbol"] = market["symbol"].astype(str).str.strip().str.upper()
    market = market.loc[market["symbol"].isin(resolved_universe)].copy()
    if market.empty:
        raise ValueError(f"lake_coverage_blocker: no selected market rows in lake dataset {dataset_id} for {start_date} -> {end_date}")

    def pivot(column: str) -> pd.DataFrame:
        out = market.pivot(index="trade_date", columns="symbol", values=column)
        out = out.reindex(columns=resolved_universe).sort_index()
        out.index.name = None
        out.columns.name = None
        return out

    open_ = pivot("open")
    high = pivot("high")
    low = pivot("low")
    close = pivot("close")
    volume = pivot("volume")
    amount = pivot("amount")

    benchmark_path = str(paths.get("silver_benchmark", "") or "")
    if not benchmark_path:
        raise ValueError(f"lake_coverage_blocker: dataset has no silver_benchmark path: {dataset_id}")
    benchmark_frame = pd.read_parquet(benchmark_path)
    benchmark_frame["trade_date"] = pd.to_datetime(benchmark_frame["trade_date"])
    benchmark_frame = benchmark_frame.loc[(benchmark_frame["trade_date"] >= start_ts) & (benchmark_frame["trade_date"] <= end_ts)].copy()
    if "benchmark" in benchmark_frame.columns and str(benchmark or "").strip():
        requested_benchmark = str(benchmark or "").strip().upper()
        matched = benchmark_frame.loc[benchmark_frame["benchmark"].astype(str).str.upper() == requested_benchmark].copy()
        if not matched.empty:
            benchmark_frame = matched
    benchmark_close = pd.Series(dtype=float, name=str(benchmark or metadata.get("benchmark", "") or "000300.SH"))
    if "close" in benchmark_frame.columns and not benchmark_frame.empty:
        benchmark_close = pd.Series(
            pd.to_numeric(benchmark_frame["close"], errors="coerce").to_numpy(dtype=float),
            index=pd.to_datetime(benchmark_frame["trade_date"]),
            name=str(benchmark or metadata.get("benchmark", "") or "000300.SH"),
        ).sort_index()
        benchmark_close.index.name = None
    if "open" in benchmark_frame.columns and not benchmark_frame.empty:
        benchmark_open = pd.Series(
            pd.to_numeric(benchmark_frame["open"], errors="coerce").to_numpy(dtype=float),
            index=pd.to_datetime(benchmark_frame["trade_date"]),
            name=str(benchmark_close.name),
        ).sort_index()
        benchmark_open.index.name = None
        benchmark_open_source = "silver_benchmark.open"
    else:
        benchmark_open = benchmark_close.copy()
        benchmark_open_source = "fallback_close"

    if pool_view is not None:
        membership_frame = view_membership.reindex(columns=resolved_universe, fill_value=False)
    else:
        membership_path = str(paths.get("silver_membership", "") or "")
        if not membership_path:
            raise ValueError(f"lake_coverage_blocker: dataset has no silver_membership path: {dataset_id}")
        membership_raw = pd.read_parquet(membership_path)
        if "trade_date" in membership_raw.columns:
            membership_raw = membership_raw.rename(columns={"trade_date": "date"})
        if "date" in membership_raw.columns:
            membership_raw["date"] = pd.to_datetime(membership_raw["date"])
            membership_frame = membership_raw.set_index("date").sort_index()
        else:
            membership_frame = membership_raw.copy()
            if len(membership_frame) == len(all_trade_dates):
                membership_frame.index = all_trade_dates
            elif len(membership_frame) == len(close.index):
                membership_frame.index = close.index
            else:
                raise ValueError(
                    "Silver membership has no date column and row count does not match market dates: "
                    f"membership_rows={len(membership_frame)}, all_trade_dates={len(all_trade_dates)}, selected_dates={len(close.index)}"
                )
        membership_frame = _slice_wide(membership_frame, universe=resolved_universe, start_date=start_date, end_date=end_date).fillna(False).astype(bool)

    panels = _load_feature_panels(str(paths["silver_feature_panels"]))
    sliced_panels = {
        name: _slice_wide(frame, universe=resolved_universe, start_date=start_date, end_date=end_date)
        for name, frame in panels.items()
    }
    score_none = sliced_panels.get("score_none", pd.DataFrame(index=close.index, columns=resolved_universe, dtype=float)).copy()
    score_v2 = sliced_panels.get("score_v2", pd.DataFrame(index=close.index, columns=resolved_universe, dtype=float)).copy()
    score_blend = sliced_panels.get("score_blend")
    if score_blend is None:
        score_blend = score_none * float(DEFAULT_SCORE_BLEND_WEIGHTS[0]) + score_v2 * float(DEFAULT_SCORE_BLEND_WEIGHTS[1])

    coverage_report = _lake_coverage_report(
        close=close,
        open_=open_,
        high=high,
        low=low,
        volume=volume,
        amount=amount,
        benchmark_close=benchmark_close,
        benchmark_open=benchmark_open,
        benchmark_open_source=benchmark_open_source,
        membership_frame=membership_frame,
        start_date=start_date,
        end_date=end_date,
        dataset_id=dataset_id,
        min_trading_days=int(min_trading_days or 1),
        require_benchmark_open=bool(require_benchmark_open),
    )
    _raise_lake_coverage_blocker(coverage_report)

    feature_names = {
        "z_score_none",
        "z_score_v2",
        "adv20_rank",
        "price_rank",
        "ma20_gap",
        "ma60_gap",
        "volume_rank",
    }
    feature_frames = {name: frame for name, frame in sliced_panels.items() if name in feature_names}
    alpha_prior_frames, alpha_prior_summary = _build_alpha_prior_frames(
        close=close,
        source=alpha_prior_source,
        score_panel=alpha_prior_score_panel,
        target_weight_panel=alpha_prior_target_weight_panel,
    )
    derived_frames = _build_state_derived_frames_from_market(
        close=close,
        volume=volume,
        amount=amount,
        score_blend=score_blend,
        alpha_prior_frames=alpha_prior_frames,
    )
    derived_frames.update({name: frame for name, frame in sliced_panels.items() if name not in {"score_none", "score_v2"}})
    valuation_frames, valuation_metrics_frame, valuation_sidecar_summary = _load_valuation_sidecar_frames(
        lake=lake,
        metadata=metadata,
        universe=resolved_universe,
        dates=close.index,
        start_date=start_date,
        end_date=end_date,
    )
    derived_frames.update(valuation_frames)
    intraday_feature_frames, intraday_sidecar_summary = _load_intraday_daily_feature_sidecar_frames(
        lake=lake,
        metadata=metadata,
        universe=resolved_universe,
        dates=close.index,
        start_date=start_date,
        end_date=end_date,
    )
    derived_frames.update(intraday_feature_frames)
    adjust_factor_frames, adjust_factor_sidecar_summary = _load_adjust_factor_sidecar_frames(
        lake=lake,
        metadata=metadata,
        universe=resolved_universe,
        dates=close.index,
        start_date=start_date,
        end_date=end_date,
    )
    derived_frames.update(adjust_factor_frames)
    derived_frames.update(alpha_prior_frames)
    derived_frames["score_blend"] = score_blend
    metadata_frames: dict[str, pd.DataFrame] = {}
    if not valuation_metrics_frame.empty:
        metadata_frames["valuation_metrics"] = valuation_metrics_frame
    industry_metadata_frames, industry_sidecar_summary = _load_industry_sidecar_metadata(
        lake=lake,
        metadata=metadata,
        universe=resolved_universe,
        start_date=start_date,
        end_date=end_date,
    )
    metadata_frames.update(industry_metadata_frames)
    metadata_summary: dict[str, Any] = {
        "sidecar_dataset_ids": _sidecar_dataset_ids(metadata),
        "valuation_sidecar": valuation_sidecar_summary,
        "industry_sidecar": industry_sidecar_summary,
        "intraday_daily_features_sidecar": intraday_sidecar_summary,
        "adjust_factor_sidecar": adjust_factor_sidecar_summary,
    }
    sector_meta_for_cache: dict[str, Any] = {}
    if sector_board_view is not None:
        universe_set = set(resolved_universe)
        industry_map = sector_board_view.industry_map_frame.copy()
        if "symbol" in industry_map.columns:
            industry_map["symbol"] = industry_map["symbol"].astype(str).str.strip().str.upper()
            industry_map = industry_map.loc[industry_map["symbol"].isin(universe_set)].sort_values("symbol").reset_index(drop=True)
        board_membership = sector_board_view.board_membership_frame.copy()
        if "symbol" in board_membership.columns:
            board_membership["symbol"] = board_membership["symbol"].astype(str).str.strip().str.upper()
            board_membership = board_membership.loc[board_membership["symbol"].isin(universe_set)].reset_index(drop=True)
        board_summary = sector_board_view.board_summary_frame.copy()
        if not board_membership.empty and {"board_kind", "board_name", "board_code"}.issubset(board_membership.columns):
            keys = board_membership[["board_kind", "board_name", "board_code"]].drop_duplicates()
            if not board_summary.empty and {"board_kind", "board_name", "board_code"}.issubset(board_summary.columns):
                board_summary = board_summary.merge(keys, on=["board_kind", "board_name", "board_code"], how="inner")
        metadata_frames.update(
            {
                "industry_map": industry_map,
                "board_membership": board_membership,
                "board_summary": board_summary,
            }
        )
        source_cache = dict(sector_board_view.metadata.get("source_cache", {}) or {})
        parameters = dict(sector_board_view.metadata.get("parameters", {}) or {})
        sector_meta_for_cache = {
            "dataset_id": sector_board_view.dataset_id,
            "view_kind": str(parameters.get("view_kind", "") or ""),
            "view_name": str(parameters.get("view_name", "") or ""),
            "source_market_dataset_id": str(parameters.get("source_market_dataset_id", "") or ""),
            "snapshot_semantics": str(parameters.get("snapshot_semantics", "") or source_cache.get("snapshot_semantics", "")),
            "as_of_date": str(parameters.get("as_of_date", "") or source_cache.get("as_of_date", "")),
            "industry_coverage": source_cache.get("industry_coverage", {}),
            "board_coverage": source_cache.get("board_coverage", {}),
        }
        metadata_summary["sector_board_view"] = dict(sector_meta_for_cache)

    history_window = HistoryWindow(
        mode="train",
        requested_start_date=pd.Timestamp(start_date).strftime("%Y%m%d"),
        effective_start_date=pd.Timestamp(start_date).strftime("%Y%m%d"),
        end_date=pd.Timestamp(end_date).strftime("%Y%m%d"),
        required_trading_days=int(len(close.index)),
    )
    return PreparedPolicyInputs(
        universe=tuple(resolved_universe),
        pool_name=str(
            (pool_view.metadata.get("parameters", {}).get("view_name", "") if pool_view is not None else "")
            or pool_name
            or metadata.get("universe_name", "")
            or metadata.get("parameters", {}).get("pool_name", "")
            or metadata.get("parameters", {}).get("universe", "")
            or "learned_all_a"
        ),
        benchmark=str(benchmark or metadata.get("benchmark", "") or "000300.SH"),
        data_source="lake",
        csv_folder="",
        start_date=pd.Timestamp(start_date).strftime("%Y-%m-%d"),
        end_date=pd.Timestamp(end_date).strftime("%Y-%m-%d"),
        requested_start_date=pd.Timestamp(start_date).strftime("%Y%m%d"),
        history_window=history_window,
        raw_cache_meta={
            "cache_hit": True,
            "source": "data_lake",
            "dataset_id": dataset_id,
            "data_lake_root": str(lake.root.resolve()),
            "lake_coverage_report": coverage_report,
            "pool_view": (
                {
                    "dataset_id": pool_view.dataset_id,
                    "view_kind": str(pool_view.metadata.get("parameters", {}).get("view_kind", "") or ""),
                    "view_name": str(pool_view.metadata.get("parameters", {}).get("view_name", "") or ""),
                    "source_market_dataset_id": str(pool_view.metadata.get("parameters", {}).get("source_market_dataset_id", "") or ""),
                }
                if pool_view is not None
                else {}
            ),
            "sector_board_view": dict(sector_meta_for_cache),
        },
        prepared_cache_meta={
            "cache_hit": True,
            "source": "data_lake",
            "dataset_id": dataset_id,
            "data_lake_root": str(lake.root.resolve()),
            "lake_coverage_report": coverage_report,
            "pool_view": (
                {
                    "dataset_id": pool_view.dataset_id,
                    "view_kind": str(pool_view.metadata.get("parameters", {}).get("view_kind", "") or ""),
                    "view_name": str(pool_view.metadata.get("parameters", {}).get("view_name", "") or ""),
                    "source_market_dataset_id": str(pool_view.metadata.get("parameters", {}).get("source_market_dataset_id", "") or ""),
                }
                if pool_view is not None
                else {}
            ),
            "sector_board_view": dict(sector_meta_for_cache),
        },
        close=close,
        open_=open_,
        high=high,
        low=low,
        volume=volume,
        amount=amount,
        benchmark_close=benchmark_close,
        benchmark_open=benchmark_open,
        score_none=score_none,
        score_v2=score_v2,
        score_blend=score_blend,
        feature_frames=feature_frames,
        market_features={},
        membership_frame=membership_frame.reindex(index=close.index, columns=close.columns, fill_value=False),
        rolling_pool_summary={
            "source": "data_lake",
            "dataset_id": dataset_id,
            "pool_view_id": pool_view.dataset_id if pool_view is not None else "",
            "pool_view_kind": str(pool_view.metadata.get("parameters", {}).get("view_kind", "") or "") if pool_view is not None else "",
            "pool_view_name": str(pool_view.metadata.get("parameters", {}).get("view_name", "") or "") if pool_view is not None else "",
        },
        alpha_prior_summary=alpha_prior_summary,
        derived_frames=derived_frames,
        metadata_frames=metadata_frames,
        metadata_summary=metadata_summary,
    )


def _build_state_derived_frames_from_market(
    *,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    amount: pd.DataFrame,
    score_blend: pd.DataFrame,
    alpha_prior_frames: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    returns_1d = _safe_pct_change(close, 1)
    rolling_high_20 = close.rolling(20, min_periods=1).max()
    rolling_high_60 = close.rolling(60, min_periods=1).max()
    rolling_low_20 = close.rolling(20, min_periods=1).min()
    derived_frames: dict[str, pd.DataFrame] = {
        "ret_1d": returns_1d,
        "ret_3d": _safe_pct_change(close, 3),
        "ret_5d": _safe_pct_change(close, 5),
        "ret_10d": _safe_pct_change(close, 10),
        "ret_20d": _safe_pct_change(close, 20),
        "vol_5d": returns_1d.rolling(5).std(),
        "vol_20d": returns_1d.rolling(20).std(),
        "adv20": amount.rolling(20).mean(),
        "volume_ratio_5_20": volume.rolling(5).mean().div(volume.rolling(20).mean().replace(0, np.nan)),
        "score_blend": score_blend,
        "score_delta_1d": score_blend.diff(1),
        "score_delta_5d": score_blend.diff(5),
        "score_delta_accel": score_blend.diff(1).sub(score_blend.diff(5).div(5.0)),
        "ret_accel_5_20": _safe_pct_change(close, 5).sub(_safe_pct_change(close, 20).div(4.0)),
        "distance_to_20d_high": close.div(rolling_high_20.replace(0, np.nan)).sub(1.0),
        "distance_to_60d_high": close.div(rolling_high_60.replace(0, np.nan)).sub(1.0),
        "distance_to_20d_low": close.div(rolling_low_20.replace(0, np.nan)).sub(1.0),
        "volatility_expansion": returns_1d.rolling(5).std().div(returns_1d.rolling(20).std().replace(0, np.nan)).sub(1.0),
        "adv_ratio_5_20": amount.rolling(5).mean().div(amount.rolling(20).mean().replace(0, np.nan)).sub(1.0),
        **alpha_prior_frames,
    }
    for base_name in STATE_SEQUENCE_BASES:
        frame = derived_frames.get(base_name)
        if frame is None:
            continue
        for lag in STATE_SEQUENCE_LAGS:
            derived_frames[f"{base_name}_lag{int(lag)}"] = frame.shift(int(lag))
    return derived_frames
