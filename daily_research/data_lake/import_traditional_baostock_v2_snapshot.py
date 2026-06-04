from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.data_lake.catalog import (
    DATA_LAKE_SCHEMA_VERSION,
    LakeDatasetRecord,
    ResearchDataLake,
    _stable_hash,
)
from daily_research.data_lake.v2_status_sidecar import V2_STATUS_COLUMNS
from daily_research.data_platform.contracts import DataDomain


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_ARTIFACT = "daily_research/output/active_execution_strategy.json"
DEFAULT_SOURCE_ROOT = (
    "traditional_quant_research/data/raw/baostock_daily_mainboard_v2_pit/"
    "baostock_v2_pit_20160101_20260601_industry_metrics_month_start_20260603"
)
DEFAULT_BENCHMARK_SOURCE_DATASET_ID = "policy_input_bundle__45e3d8c059ba718426a9f887"
DEFAULT_RUN_TAG = "daily_research_traditional_baostock_v2_1_import_20260604_01"
IMPORT_SOURCE = "traditional_baostock_v2_1_import"
SOURCE_BRIDGE_NAME = "traditional_quant_research_baostock_v2_1_pit"
METRIC_FIELDS = ("turn", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(dict(payload)), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return dict(payload) if isinstance(payload, dict) else {}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _active_artifact_has_diff() -> bool:
    result = subprocess.run(["git", "diff", "--", ACTIVE_ARTIFACT], check=False, capture_output=True, text=True)
    return bool(str(result.stdout or "").strip() or str(result.stderr or "").strip())


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def resolve_source_root(source_root: str | Path = DEFAULT_SOURCE_ROOT) -> Path:
    base = _resolve_project_path(source_root)
    if (base / "manifest.json").exists():
        return base
    latest = base / "latest_manifest.json"
    if latest.exists():
        payload = _read_json(latest)
        snapshot_path = str(payload.get("snapshot_path", "") or "").strip()
        if snapshot_path:
            resolved = _resolve_project_path(snapshot_path)
            if (resolved / "manifest.json").exists():
                return resolved
    raise FileNotFoundError(f"traditional Baostock v2 snapshot manifest not found under: {base}")


def _normalize_symbol(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    text = text.replace("-", ".")
    if text.startswith("SH.") or text.startswith("SZ."):
        suffix, code = text.split(".", 1)
        return f"{code}.{suffix}"
    if "." in text:
        code, suffix = text.split(".", 1)
        return f"{code.zfill(6)}.{suffix.upper()}"
    code = text.zfill(6)
    if code.startswith(("600", "601", "603", "605", "688", "689")):
        return f"{code}.SH"
    if code.startswith(("000", "001", "002", "003", "300", "301")):
        return f"{code}.SZ"
    return code


def _symbol_exchange(symbol: Any) -> str:
    text = str(symbol or "").strip().upper()
    return text.split(".", 1)[1] if "." in text else ""


def _symbol_board(symbol: Any) -> str:
    code = str(symbol or "").strip().upper().split(".", 1)[0]
    if code.startswith(("600", "601", "603", "605", "000", "001", "002", "003")):
        return "mainboard"
    if code.startswith(("300", "301")):
        return "chinext"
    if code.startswith(("688", "689")):
        return "star"
    return ""


def _to_bool_series(series: pd.Series, *, default: bool = False) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(default).astype(bool)
    normalized = series.astype("object").where(series.notna(), default).astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y", "是", "st", "*st"})


def _read_required_parquet(snapshot_root: Path, table_name: str, required_columns: set[str]) -> pd.DataFrame:
    path = snapshot_root / f"{table_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"traditional_import_blocker: missing {table_name}.parquet: {path}")
    frame = pd.read_parquet(path)
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"traditional_import_blocker: {table_name}.parquet missing columns {missing}")
    return frame


def _normalize_market_frame(daily_bars: pd.DataFrame, *, start_date: str, end_date: str) -> pd.DataFrame:
    data = daily_bars.copy()
    data["trade_date"] = pd.to_datetime(data["date"], errors="coerce")
    if start_date:
        data = data.loc[data["trade_date"] >= pd.Timestamp(start_date)]
    if end_date:
        data = data.loc[data["trade_date"] <= pd.Timestamp(end_date)]
    data["symbol"] = data["code"].map(_normalize_symbol)
    data = data.loc[data["trade_date"].notna() & data["symbol"].ne("")].copy()
    for column in ("open", "high", "low", "close", "volume", "amount"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["source"] = IMPORT_SOURCE
    data["ingest_batch_id"] = _stable_hash(
        {
            "source": IMPORT_SOURCE,
            "trade_date_min": data["trade_date"].min().strftime("%Y-%m-%d") if not data.empty else "",
            "trade_date_max": data["trade_date"].max().strftime("%Y-%m-%d") if not data.empty else "",
            "symbol_count": int(data["symbol"].nunique()) if not data.empty else 0,
        }
    )
    data["trade_date"] = data["trade_date"].dt.strftime("%Y-%m-%d")
    ordered = ["trade_date", "symbol", "open", "high", "low", "close", "volume", "amount", "source", "ingest_batch_id"]
    return data[ordered].drop_duplicates(["trade_date", "symbol"], keep="last").sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _normalize_universe_frame(daily_universe: pd.DataFrame, *, start_date: str, end_date: str) -> pd.DataFrame:
    data = daily_universe.copy()
    data["trade_date"] = pd.to_datetime(data["date"], errors="coerce")
    if start_date:
        data = data.loc[data["trade_date"] >= pd.Timestamp(start_date)]
    if end_date:
        data = data.loc[data["trade_date"] <= pd.Timestamp(end_date)]
    data["symbol"] = data["code"].map(_normalize_symbol)
    listed = _to_bool_series(data.get("is_listed_on_date", pd.Series(True, index=data.index)), default=True)
    out = pd.DataFrame(
        {
            "symbol": data["symbol"],
            "trade_date": data["trade_date"].dt.strftime("%Y-%m-%d"),
            "name": data.get("name_on_date", pd.Series("", index=data.index)).fillna("").astype(str),
            "exchange": data["symbol"].map(_symbol_exchange),
            "board": data["symbol"].map(_symbol_board),
            "list_status": np.where(listed.to_numpy(dtype=bool), "L", "D"),
            "list_date": data.get("ipo_date", pd.Series("", index=data.index)).fillna("").astype(str),
            "delist_date": data.get("out_date", pd.Series("", index=data.index)).fillna("").astype(str),
            "is_mainboard": _to_bool_series(data.get("is_mainboard", pd.Series(False, index=data.index))),
            "is_common_a_share": _to_bool_series(data.get("is_common_a_share", pd.Series(False, index=data.index))),
            "source": IMPORT_SOURCE,
        }
    )
    return out.loc[out["symbol"].ne("")].drop_duplicates(["trade_date", "symbol"], keep="last").sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _normalize_status_frame(daily_universe: pd.DataFrame, daily_status: pd.DataFrame, *, start_date: str, end_date: str) -> pd.DataFrame:
    data = daily_universe.copy()
    data["trade_date"] = pd.to_datetime(data["date"], errors="coerce")
    if start_date:
        data = data.loc[data["trade_date"] >= pd.Timestamp(start_date)]
    if end_date:
        data = data.loc[data["trade_date"] <= pd.Timestamp(end_date)]
    data["symbol"] = data["code"].map(_normalize_symbol)
    status = daily_status.copy()
    if not status.empty and {"date", "code"}.issubset(status.columns):
        status["trade_date_text"] = pd.to_datetime(status["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        status["symbol"] = status["code"].map(_normalize_symbol)
        status = status.rename(
            columns={
                "has_bar": "status_has_bar",
                "is_suspended_like": "status_is_suspended_like",
                "is_tradeable": "status_is_tradeable",
            }
        )
        keep = [
            column
            for column in ("trade_date_text", "symbol", "status_has_bar", "status_is_suspended_like", "status_is_tradeable")
            if column in status.columns
        ]
        status = status[keep].drop_duplicates(["trade_date_text", "symbol"], keep="last")
        data["trade_date_text"] = data["trade_date"].dt.strftime("%Y-%m-%d")
        data = data.merge(status, on=["trade_date_text", "symbol"], how="left")
    out_dates = pd.to_datetime(data.get("out_date", pd.Series("", index=data.index)).replace("", pd.NA), errors="coerce")
    trade_dates = pd.to_datetime(data["trade_date"], errors="coerce")
    is_listed = _to_bool_series(data.get("is_listed_on_date", pd.Series(True, index=data.index)), default=True)
    has_bar = _to_bool_series(data.get("status_has_bar", data.get("has_bar", pd.Series(False, index=data.index))))
    is_suspended = _to_bool_series(
        data.get("status_is_suspended_like", data.get("is_suspended_on_date", pd.Series(False, index=data.index)))
    )
    is_tradeable = _to_bool_series(data.get("status_is_tradeable", data.get("is_tradeable", pd.Series(False, index=data.index))))
    out = pd.DataFrame(
        {
            "symbol": data["symbol"],
            "trade_date": trade_dates.dt.strftime("%Y-%m-%d"),
            "is_listed_on_date": is_listed,
            "is_mainboard": _to_bool_series(data.get("is_mainboard", pd.Series(False, index=data.index))),
            "is_common_a_share": _to_bool_series(data.get("is_common_a_share", pd.Series(False, index=data.index))),
            "is_st": _to_bool_series(data.get("is_st_on_date", pd.Series(False, index=data.index))),
            "is_suspended": is_suspended,
            "is_delisted": out_dates.notna() & (trade_dates >= out_dates),
            "is_tradeable": is_tradeable,
            "has_bar": has_bar,
            "reject_reason": data.get("reject_reason", pd.Series("", index=data.index)).fillna("").astype(str),
            "source": IMPORT_SOURCE,
        }
    )
    return out[V2_STATUS_COLUMNS].loc[out["symbol"].ne("")].drop_duplicates(["trade_date", "symbol"], keep="last").sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _normalize_industry_frame(stock_industry: pd.DataFrame, *, start_date: str, end_date: str) -> pd.DataFrame:
    data = stock_industry.copy()
    data["trade_date"] = pd.to_datetime(data["date"], errors="coerce")
    if start_date:
        data = data.loc[data["trade_date"] >= pd.Timestamp(start_date)]
    if end_date:
        data = data.loc[data["trade_date"] <= pd.Timestamp(end_date)]
    data["symbol"] = data["code"].map(_normalize_symbol)
    out = pd.DataFrame(
        {
            "symbol": data["symbol"],
            "trade_date": data["trade_date"].dt.strftime("%Y-%m-%d"),
            "industry": data.get("industry", pd.Series("", index=data.index)).fillna("").astype(str).str.strip(),
            "concept_tags": "",
            "industry_classification": data.get("industry_classification", pd.Series("", index=data.index)).fillna("").astype(str),
            "industry_update_date": data.get("industry_update_date", pd.Series("", index=data.index)).fillna("").astype(str),
            "source": IMPORT_SOURCE,
        }
    )
    out = out.loc[out["symbol"].ne("") & out["industry"].ne("")]
    return out.drop_duplicates(["trade_date", "symbol"], keep="last").sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _normalize_metrics_frame(daily_metrics: pd.DataFrame, *, start_date: str, end_date: str) -> pd.DataFrame:
    data = daily_metrics.copy()
    data["trade_date"] = pd.to_datetime(data["date"], errors="coerce")
    if start_date:
        data = data.loc[data["trade_date"] >= pd.Timestamp(start_date)]
    if end_date:
        data = data.loc[data["trade_date"] <= pd.Timestamp(end_date)]
    data["symbol"] = data["code"].map(_normalize_symbol)
    out = pd.DataFrame({"symbol": data["symbol"], "trade_date": data["trade_date"].dt.strftime("%Y-%m-%d")})
    for field in METRIC_FIELDS:
        out[field] = pd.to_numeric(data.get(field, pd.Series(np.nan, index=data.index)), errors="coerce")
    out["source"] = IMPORT_SOURCE
    return out.loc[out["symbol"].ne("")].drop_duplicates(["trade_date", "symbol"], keep="last").sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _membership_from_status(status_frame: pd.DataFrame) -> pd.DataFrame:
    data = status_frame[["trade_date", "symbol", "is_tradeable"]].copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data["symbol"] = data["symbol"].astype(str).str.strip().str.upper()
    membership = data.pivot(index="trade_date", columns="symbol", values="is_tradeable").sort_index()
    membership.index.name = "trade_date"
    membership.columns.name = None
    return membership.fillna(False).astype(bool)


def _load_benchmark_from_source(
    lake: ResearchDataLake,
    *,
    benchmark_source_dataset_id: str,
    benchmark: str,
    start_date: str,
    end_date: str,
) -> tuple[pd.Series, pd.Series | None, dict[str, Any]]:
    if not str(benchmark_source_dataset_id or "").strip():
        return pd.Series(dtype=float, name=benchmark), None, {"status": "missing_source_dataset_id", "coverage_ratio": 0.0}
    metadata = lake.describe_dataset(str(benchmark_source_dataset_id))
    path = str(dict(metadata.get("content_paths", {}) or {}).get("silver_benchmark", "") or "")
    if not path or not Path(path).exists():
        return pd.Series(dtype=float, name=benchmark), None, {"status": "missing_silver_benchmark", "coverage_ratio": 0.0}
    frame = pd.read_parquet(path)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    if "benchmark" in frame.columns:
        requested = str(benchmark or "").strip().upper()
        matched = frame.loc[frame["benchmark"].astype(str).str.upper() == requested].copy()
        if not matched.empty:
            frame = matched
    if start_date:
        frame = frame.loc[frame["trade_date"] >= pd.Timestamp(start_date)]
    if end_date:
        frame = frame.loc[frame["trade_date"] <= pd.Timestamp(end_date)]
    frame = frame.dropna(subset=["trade_date"]).drop_duplicates("trade_date", keep="last").sort_values("trade_date")
    close = pd.Series(pd.to_numeric(frame.get("close", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float), index=frame["trade_date"], name=benchmark)
    open_series: pd.Series | None = None
    if "open" in frame.columns:
        open_series = pd.Series(pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float), index=frame["trade_date"], name=benchmark)
    expected_dates: pd.DatetimeIndex
    if start_date and end_date:
        expected_dates = pd.bdate_range(pd.Timestamp(start_date), pd.Timestamp(end_date))
    else:
        expected_dates = pd.DatetimeIndex(frame["trade_date"].dropna().unique())
    coverage_ratio = float(close.notna().sum() / max(int(len(expected_dates)), 1))
    return close, open_series, {
        "status": "ok" if int(close.notna().sum()) > 0 else "missing_benchmark_rows",
        "source_dataset_id": str(benchmark_source_dataset_id),
        "benchmark": str(benchmark),
        "rows": int(len(close)),
        "non_null_close_rows": int(close.notna().sum()),
        "non_null_open_rows": int(open_series.notna().sum()) if open_series is not None else 0,
        "date_min": close.index.min().strftime("%Y-%m-%d") if len(close.index) else "",
        "date_max": close.index.max().strftime("%Y-%m-%d") if len(close.index) else "",
        "rough_business_day_coverage_ratio": coverage_ratio,
    }


def _save_policy_input_bundle_long(
    *,
    lake: ResearchDataLake,
    spec: Mapping[str, Any],
    market_long: pd.DataFrame,
    benchmark_close: pd.Series,
    benchmark_open: pd.Series | None,
    membership_frame: pd.DataFrame,
    source: str,
    reuse: bool,
) -> LakeDatasetRecord:
    dataset_kind = "policy_input_bundle"
    zone = "research"
    effective_spec = dict(spec)
    if benchmark_open is not None and "benchmark_fields" not in effective_spec:
        effective_spec["benchmark_fields"] = ["open", "close"]
    fingerprint = _stable_hash(
        {
            "schema_version": DATA_LAKE_SCHEMA_VERSION,
            "dataset_kind": dataset_kind,
            "zone": zone,
            "spec": effective_spec,
        }
    )
    dataset_id = f"{dataset_kind}__{fingerprint}"
    dataset_dir = lake.parquet_root / "bronze_silver" / dataset_kind / fingerprint
    content_paths = {
        "bronze_market_data": str((dataset_dir / "bronze_market_data.parquet").resolve()),
        "silver_benchmark": str((dataset_dir / "silver_benchmark.parquet").resolve()),
        "silver_membership": str((dataset_dir / "silver_membership.parquet").resolve()),
        "silver_feature_values": str((dataset_dir / "silver_feature_panels" / "*.parquet").resolve()),
        "silver_feature_panels": str((dataset_dir / "silver_feature_panels" / "*.parquet").resolve()),
    }
    existing = lake._existing_by_fingerprint(fingerprint)
    if reuse and existing is not None and Path(content_paths["bronze_market_data"]).exists():
        metadata = lake.describe_dataset(str(existing["dataset_id"]))
        return LakeDatasetRecord(
            dataset_id=str(metadata["dataset_id"]),
            dataset_kind=str(metadata["dataset_kind"]),
            zone=str(metadata["zone"]),
            fingerprint=str(metadata["fingerprint"]),
            status="hit",
            root=lake.root,
            content_paths=dict(metadata.get("content_paths", {}) or {}),
            row_counts={str(key): int(value) for key, value in dict(metadata.get("row_counts", {}) or {}).items()},
            metadata=metadata,
        )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    market_out = market_long.copy()
    market_out.to_parquet(content_paths["bronze_market_data"], index=False)
    benchmark_dates = pd.Index(pd.to_datetime(benchmark_close.index)).dropna().unique()
    open_series: pd.Series | None = None
    if benchmark_open is not None:
        open_series = pd.Series(pd.to_numeric(benchmark_open, errors="coerce"), index=pd.to_datetime(benchmark_open.index), name=benchmark_open.name)
        benchmark_dates = benchmark_dates.union(pd.Index(pd.to_datetime(open_series.index)).dropna().unique())
    benchmark_dates = pd.Index(sorted(benchmark_dates))
    close_series = pd.Series(pd.to_numeric(benchmark_close, errors="coerce"), index=pd.to_datetime(benchmark_close.index), name=benchmark_close.name)
    benchmark = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(benchmark_dates).strftime("%Y-%m-%d"),
            "benchmark": str(effective_spec.get("benchmark", benchmark_close.name or "") or ""),
            "close": close_series.reindex(benchmark_dates).to_numpy(dtype=float),
        }
    )
    if open_series is not None:
        benchmark["open"] = open_series.reindex(benchmark_dates).to_numpy(dtype=float)
    benchmark.to_parquet(content_paths["silver_benchmark"], index=False)
    membership_out = membership_frame.copy()
    membership_out.index = pd.to_datetime(membership_out.index)
    membership_out = membership_out.sort_index()
    membership_out.insert(0, "trade_date", membership_out.index.strftime("%Y-%m-%d"))
    membership_out.reset_index(drop=True).to_parquet(content_paths["silver_membership"], index=False)
    feature_dir = dataset_dir / "silver_feature_panels"
    feature_dir.mkdir(parents=True, exist_ok=True)
    start = str(market_out["trade_date"].min()) if not market_out.empty else ""
    end = str(market_out["trade_date"].max()) if not market_out.empty else ""
    row_counts = {
        "bronze_market_data": int(len(market_out)),
        "silver_benchmark": int(len(benchmark)),
        "silver_membership": int(len(membership_out)),
        "silver_feature_panels": 0,
        "silver_feature_cells": 0,
        "_date_bounds": {"start_date": start, "end_date": end},
    }
    merged_spec = {**effective_spec, "start_date": str(effective_spec.get("start_date", "") or start), "end_date": str(effective_spec.get("end_date", "") or end)}
    lake._upsert_dataset(
        dataset_id=dataset_id,
        dataset_kind=dataset_kind,
        domain="bronze_silver",
        zone=zone,
        source=str(source or ""),
        spec=merged_spec,
        label_completeness_summary={},
        content_paths=content_paths,
        row_counts=row_counts,
        source_cache={},
        fingerprint=fingerprint,
        status="stored",
    )
    metadata = lake.describe_dataset(dataset_id)
    return LakeDatasetRecord(
        dataset_id=dataset_id,
        dataset_kind=dataset_kind,
        zone=zone,
        fingerprint=fingerprint,
        status="stored",
        root=lake.root,
        content_paths=dict(metadata.get("content_paths", {}) or {}),
        row_counts={str(key): int(value) for key, value in dict(metadata.get("row_counts", {}) or {}).items()},
        metadata=metadata,
    )


def _attach_sidecars_to_bundle(
    *,
    lake: ResearchDataLake,
    bundle_dataset_id: str,
    sidecar_dataset_ids: Mapping[str, str],
    import_summary: Mapping[str, Any],
) -> LakeDatasetRecord:
    metadata = lake.describe_dataset(str(bundle_dataset_id))
    parameters = dict(metadata.get("parameters", {}) or {})
    existing_sidecars = dict(parameters.get("sidecar_dataset_ids", {}) or {})
    existing_sidecars.update({str(key): str(value) for key, value in sidecar_dataset_ids.items() if str(value or "").strip()})
    parameters["sidecar_dataset_ids"] = existing_sidecars
    parameters["sidecar_linkage_policy"] = "attached_after_bundle_materialization"
    source_cache = dict(metadata.get("source_cache", {}) or {})
    source_cache["traditional_baostock_v2_1_import"] = dict(import_summary)
    lake._upsert_dataset(
        dataset_id=str(metadata["dataset_id"]),
        dataset_kind=str(metadata["dataset_kind"]),
        domain=str(metadata.get("domain", "") or "bronze_silver"),
        zone=str(metadata.get("zone", "") or "research"),
        source=str(metadata.get("source", "") or IMPORT_SOURCE),
        spec=parameters,
        label_completeness_summary=dict(metadata.get("label_completeness_summary", {}) or {}),
        content_paths=dict(metadata.get("content_paths", {}) or {}),
        row_counts=dict(metadata.get("row_counts", {}) or {}),
        source_cache=source_cache,
        fingerprint=str(metadata["fingerprint"]),
        status=str(metadata.get("status", "") or "stored"),
    )
    updated = lake.describe_dataset(str(bundle_dataset_id))
    return LakeDatasetRecord(
        dataset_id=str(updated["dataset_id"]),
        dataset_kind=str(updated["dataset_kind"]),
        zone=str(updated["zone"]),
        fingerprint=str(updated["fingerprint"]),
        status=str(updated.get("status", "") or "stored"),
        root=lake.root,
        content_paths=dict(updated.get("content_paths", {}) or {}),
        row_counts={str(key): int(value) for key, value in dict(updated.get("row_counts", {}) or {}).items()},
        metadata=updated,
    )


def _coverage_ratio(frame: pd.DataFrame, value_columns: list[str], *, on_tradeable: pd.DataFrame | None = None) -> dict[str, Any]:
    data = frame.copy()
    if on_tradeable is not None and not on_tradeable.empty:
        keys = on_tradeable.loc[on_tradeable["is_tradeable"].astype(bool), ["trade_date", "symbol"]].copy()
        data = keys.merge(data, on=["trade_date", "symbol"], how="left")
    total = max(int(len(data)), 1)
    out: dict[str, Any] = {}
    for column in value_columns:
        series = data.get(column, pd.Series(np.nan, index=data.index))
        if str(column) == "industry":
            valid = series.fillna("").astype(str).str.strip().ne("")
        else:
            valid = pd.to_numeric(series, errors="coerce").notna()
        out[str(column)] = float(valid.sum() / total)
    return out


def import_traditional_baostock_v2_snapshot(
    *,
    lake: ResearchDataLake,
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    benchmark_source_dataset_id: str = DEFAULT_BENCHMARK_SOURCE_DATASET_ID,
    benchmark: str = "000300.SH",
    start_date: str = "",
    end_date: str = "",
    run_tag: str = DEFAULT_RUN_TAG,
    reuse: bool = True,
) -> dict[str, Any]:
    snapshot_root = resolve_source_root(source_root)
    manifest_path = snapshot_root / "manifest.json"
    quality_path = snapshot_root / "quality_report.json"
    manifest = _read_json(manifest_path)
    quality = _read_json(quality_path)
    snapshot_id = str(manifest.get("snapshot_id", snapshot_root.name) or snapshot_root.name)
    dataset_meta = dict(manifest.get("dataset", {}) or {})
    resolved_start = str(start_date or dataset_meta.get("date_min", "") or "")
    resolved_end = str(end_date or dataset_meta.get("date_max", "") or "")

    daily_bars = _read_required_parquet(snapshot_root, "daily_bars", {"date", "code", "open", "high", "low", "close", "volume", "amount"})
    daily_universe = _read_required_parquet(snapshot_root, "daily_universe", {"date", "code", "is_tradeable", "is_listed_on_date", "is_mainboard", "is_common_a_share", "is_st_on_date", "is_suspended_on_date", "has_bar", "reject_reason"})
    daily_status = _read_required_parquet(snapshot_root, "daily_status", {"date", "code", "has_bar", "is_suspended_like", "is_tradeable"})
    stock_industry = _read_required_parquet(snapshot_root, "stock_industry", {"date", "code", "industry", "industry_classification", "industry_update_date"})
    daily_metrics = _read_required_parquet(snapshot_root, "daily_metrics", {"date", "code", *METRIC_FIELDS})

    market = _normalize_market_frame(daily_bars, start_date=resolved_start, end_date=resolved_end)
    universe = _normalize_universe_frame(daily_universe, start_date=resolved_start, end_date=resolved_end)
    status = _normalize_status_frame(daily_universe, daily_status, start_date=resolved_start, end_date=resolved_end)
    industry = _normalize_industry_frame(stock_industry, start_date=resolved_start, end_date=resolved_end)
    metrics = _normalize_metrics_frame(daily_metrics, start_date=resolved_start, end_date=resolved_end)
    if market.empty:
        raise ValueError("traditional_import_blocker: normalized market frame is empty")
    if status.empty:
        raise ValueError("traditional_import_blocker: normalized status frame is empty")
    membership = _membership_from_status(status)
    benchmark_close, benchmark_open, benchmark_summary = _load_benchmark_from_source(
        lake,
        benchmark_source_dataset_id=benchmark_source_dataset_id,
        benchmark=benchmark,
        start_date=resolved_start,
        end_date=resolved_end,
    )
    if int(benchmark_summary.get("non_null_close_rows", 0) or 0) <= 0:
        raise ValueError(
            "traditional_import_blocker: benchmark source did not provide any benchmark rows; "
            f"benchmark_source_dataset_id={benchmark_source_dataset_id}"
        )

    import_summary: dict[str, Any] = {
        "run_tag": str(run_tag),
        "source": IMPORT_SOURCE,
        "source_bridge_name": SOURCE_BRIDGE_NAME,
        "source_snapshot_id": snapshot_id,
        "source_snapshot_path": str(snapshot_root),
        "source_manifest_fingerprint": _sha256_file(manifest_path),
        "source_quality_fingerprint": _sha256_file(quality_path) if quality_path.exists() else "",
        "source_manifest_dataset": dataset_meta,
        "source_manifest_quality": dict(manifest.get("quality", {}) or {}),
        "source_quality_report": quality,
        "date_range": {"start_date": str(market["trade_date"].min()), "end_date": str(market["trade_date"].max())},
        "symbol_count": int(market["symbol"].nunique()),
        "trade_date_count": int(pd.Series(market["trade_date"]).nunique()),
        "imported_table_row_counts": {
            "daily_universe": int(len(universe)),
            "daily_bars": int(len(market)),
            "daily_status": int(len(status)),
            "stock_industry": int(len(industry)),
            "daily_metrics": int(len(metrics)),
        },
        "source_table_row_counts": {
            "daily_universe": int(len(daily_universe)),
            "daily_bars": int(len(daily_bars)),
            "daily_status": int(len(daily_status)),
            "stock_industry": int(len(stock_industry)),
            "daily_metrics": int(len(daily_metrics)),
        },
        "row_count_alignment": {
            "manifest_daily_universe_rows": int(dataset_meta.get("daily_universe_rows", 0) or 0),
            "manifest_daily_bar_rows": int(dataset_meta.get("daily_bar_rows", 0) or 0),
            "manifest_stock_industry_rows": int(dataset_meta.get("stock_industry_rows", 0) or 0),
            "manifest_daily_metrics_rows": int(dataset_meta.get("daily_metrics_rows", 0) or 0),
            "imported_rows_match_manifest": bool(
                int(dataset_meta.get("daily_bar_rows", len(market)) or len(market)) == int(len(market))
                and int(dataset_meta.get("daily_universe_rows", len(universe)) or len(universe)) == int(len(universe))
            ),
        },
        "industry_frequency": "month-start-ffill",
        "industry_pit_semantics": "approximate_pit_month_start_forward_fill_not_exact_daily_industry_change",
        "metrics_fields": list(METRIC_FIELDS),
        "valuation_lag_policy": "feature_profile_lags_pe_pb_ps_pcf_by_at_least_one_trading_day",
        "pit_limitations": [
            "industry is Baostock month-start forward filled, not exact daily change history",
            "valuation metrics are imported as daily Baostock metrics and must be lagged in formal features",
            "2025-2026 rows are shadow/forward-review until label completeness protocol is explicit",
            "benchmark coverage is inherited from benchmark_source_dataset_id and may be partial outside formal train windows",
        ],
        "formal_protocol": {
            "same_period_ab": {"start_date": "2018-01-01", "end_date": "2024-12-31"},
            "long_history_ab": {"start_date": "2016-01-01", "end_date": "2024-12-31", "formal_train_start_date": "2017-01-01"},
            "shadow_forward_review": {"start_date": "2025-01-01", "end_date": "2026-06-01", "formal_model_quality_evidence": False},
        },
        "benchmark_summary": benchmark_summary,
        "tradeable_rows": int(status["is_tradeable"].astype(bool).sum()),
        "industry_coverage_on_tradeable": _coverage_ratio(industry, ["industry"], on_tradeable=status),
        "metrics_coverage_on_tradeable": _coverage_ratio(metrics, list(METRIC_FIELDS), on_tradeable=status),
    }

    bundle_spec = {
        "dataset": "traditional_baostock_v2_1_policy_input_bundle",
        "source_snapshot_id": snapshot_id,
        "source_manifest_fingerprint": import_summary["source_manifest_fingerprint"],
        "source": IMPORT_SOURCE,
        "pool_name": "traditional_baostock_v2_1_mainboard_pit",
        "benchmark": str(benchmark),
        "start_date": str(market["trade_date"].min()),
        "end_date": str(market["trade_date"].max()),
        "feature_profile_candidates": ["raw_kline_context_v2_tradeable_local_state_industry_metrics_v1"],
        "shadow_forward_only_after": "2024-12-31",
    }
    bundle = _save_policy_input_bundle_long(
        lake=lake,
        spec=bundle_spec,
        market_long=market,
        benchmark_close=benchmark_close,
        benchmark_open=benchmark_open,
        membership_frame=membership,
        source=IMPORT_SOURCE,
        reuse=reuse,
    )

    sidecar_common = {
        "source": IMPORT_SOURCE,
        "source_market_dataset_id": bundle.dataset_id,
        "source_snapshot_id": snapshot_id,
        "source_manifest_fingerprint": import_summary["source_manifest_fingerprint"],
        "start_date": str(market["trade_date"].min()),
        "end_date": str(market["trade_date"].max()),
        "run_tag": str(run_tag),
    }
    universe_record = lake.save_domain_dataset(
        domain=DataDomain.UNIVERSE_SNAPSHOT,
        frame=universe,
        spec={"dataset": f"data_platform_{DataDomain.UNIVERSE_SNAPSHOT}", **sidecar_common},
        source=IMPORT_SOURCE,
        reuse=reuse,
    )
    status_record = lake.save_domain_dataset(
        domain="v2_status_sidecar",
        frame=status,
        spec={"dataset": "data_platform_v2_status_sidecar", **sidecar_common},
        source=IMPORT_SOURCE,
        reuse=reuse,
    )
    industry_record = lake.save_domain_dataset(
        domain=DataDomain.INDUSTRY_CONCEPT,
        frame=industry,
        spec={
            "dataset": f"data_platform_{DataDomain.INDUSTRY_CONCEPT}",
            **sidecar_common,
            "industry_frequency": "month-start-ffill",
            "pit_semantics": "approximate_pit_month_start_forward_fill",
        },
        source=IMPORT_SOURCE,
        reuse=reuse,
    )
    metrics_record = lake.save_domain_dataset(
        domain=DataDomain.VALUATION,
        frame=metrics,
        spec={
            "dataset": f"data_platform_{DataDomain.VALUATION}",
            **sidecar_common,
            "metrics_fields": list(METRIC_FIELDS),
            "valuation_lag_policy": "lag_at_least_one_trading_day_in_feature_profile",
        },
        source=IMPORT_SOURCE,
        reuse=reuse,
    )
    sidecar_ids = {
        DataDomain.UNIVERSE_SNAPSHOT: universe_record.dataset_id,
        "v2_status_sidecar": status_record.dataset_id,
        DataDomain.INDUSTRY_CONCEPT: industry_record.dataset_id,
        DataDomain.VALUATION: metrics_record.dataset_id,
    }
    bundle = _attach_sidecars_to_bundle(
        lake=lake,
        bundle_dataset_id=bundle.dataset_id,
        sidecar_dataset_ids=sidecar_ids,
        import_summary=import_summary,
    )
    import_summary["dataset_id"] = bundle.dataset_id
    import_summary["sidecar_dataset_ids"] = sidecar_ids
    import_summary["content_paths"] = bundle.content_paths
    import_summary["row_counts"] = bundle.row_counts
    return {
        "status": "ok",
        "run_tag": str(run_tag),
        "dataset_id": bundle.dataset_id,
        "dataset_kind": bundle.dataset_kind,
        "fingerprint": bundle.fingerprint,
        "content_paths": bundle.content_paths,
        "row_counts": bundle.row_counts,
        "sidecar_dataset_ids": sidecar_ids,
        "summary": import_summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import traditional_quant_research Baostock v2.1 PIT snapshot into daily_research lake.")
    parser.add_argument("--data-lake-root", default="")
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--traditional-snapshot-root", default="", help="Alias for --source-root.")
    parser.add_argument("--benchmark-source-dataset-id", default=DEFAULT_BENCHMARK_SOURCE_DATASET_ID)
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_ARTIFACT} has uncommitted diff.")
    lake = ResearchDataLake(str(args.data_lake_root or "").strip() or None)
    source_root = str(args.traditional_snapshot_root or args.source_root or DEFAULT_SOURCE_ROOT).strip()
    payload = import_traditional_baostock_v2_snapshot(
        lake=lake,
        source_root=source_root,
        benchmark_source_dataset_id=str(args.benchmark_source_dataset_id or "").strip(),
        benchmark=str(args.benchmark or "000300.SH").strip().upper(),
        start_date=str(args.start_date or "").strip(),
        end_date=str(args.end_date or "").strip(),
        run_tag=str(args.run_tag or DEFAULT_RUN_TAG).strip(),
        reuse=not bool(args.refresh),
    )
    output_root = PROJECT_ROOT / "daily_research/output/data_lake/imports" / str(args.run_tag or DEFAULT_RUN_TAG)
    manifest_path = output_root / "import_manifest.json"
    payload["import_manifest_path"] = str(manifest_path.resolve())
    _write_json(manifest_path, payload)
    lake.write_catalog_manifest()
    print(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    main()
