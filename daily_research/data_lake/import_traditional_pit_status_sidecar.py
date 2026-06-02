from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.data_lake.catalog import ResearchDataLake
from daily_research.data_lake.v2_status_sidecar import (
    V2_STATUS_COLUMNS,
    V2_STATUS_SIDECAR_DOMAIN,
    summarize_v2_status_sidecar,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_ARTIFACT = "daily_research/output/active_execution_strategy.json"
DEFAULT_DATASET_ID = "policy_input_bundle__45e3d8c059ba718426a9f887"
DEFAULT_TRADITIONAL_SNAPSHOT_ROOT = "traditional_quant_research/data/raw/baostock_daily_mainboard_v2_pit"
DEFAULT_RUN_TAG = "traditional_pit_status_sidecar_import_20260602_01"
BRIDGE_SOURCE = "traditional_quant_research_v2_pit"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    return value


def _active_artifact_has_diff() -> bool:
    result = subprocess.run(["git", "diff", "--", ACTIVE_ARTIFACT], check=False, capture_output=True, text=True)
    return bool(str(result.stdout or "").strip() or str(result.stderr or "").strip())


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def resolve_traditional_snapshot_root(snapshot_root: str | Path = DEFAULT_TRADITIONAL_SNAPSHOT_ROOT) -> Path:
    base = _resolve_project_path(snapshot_root)
    if (base / "manifest.json").exists():
        return base
    latest = base / "latest_manifest.json"
    if not latest.exists():
        raise FileNotFoundError(f"traditional PIT latest_manifest not found: {latest}")
    payload = json.loads(latest.read_text(encoding="utf-8-sig"))
    snapshot_path = str(payload.get("snapshot_path", "") or "").strip()
    if not snapshot_path:
        raise ValueError(f"traditional PIT latest_manifest has no snapshot_path: {latest}")
    resolved = _resolve_project_path(snapshot_path)
    if not (resolved / "manifest.json").exists():
        raise FileNotFoundError(f"traditional PIT manifest not found: {resolved / 'manifest.json'}")
    return resolved


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return dict(payload) if isinstance(payload, dict) else {}


def _to_bool_series(series: pd.Series, *, default: bool = False) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(default).astype(bool)
    normalized = series.astype("object").where(series.notna(), default).astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y", "是", "st", "*st"})


def _read_market_keys(
    lake: ResearchDataLake,
    *,
    source_market_dataset_id: str,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = lake.describe_dataset(str(source_market_dataset_id))
    path = str(dict(metadata.get("content_paths", {}) or {}).get("bronze_market_data", "") or "")
    if not path or not Path(path).exists():
        raise ValueError(f"traditional_pit_import_blocker: missing bronze_market_data for {source_market_dataset_id}")
    try:
        market = pd.read_parquet(path, columns=["trade_date", "symbol"])
    except Exception:
        market = pd.read_parquet(path)[["trade_date", "symbol"]]
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="coerce")
    start = pd.Timestamp(start_date or metadata.get("start_date", "") or market["trade_date"].min())
    end = pd.Timestamp(end_date or metadata.get("end_date", "") or market["trade_date"].max())
    market = market.loc[(market["trade_date"] >= start) & (market["trade_date"] <= end)].copy()
    market["trade_date"] = market["trade_date"].dt.strftime("%Y-%m-%d")
    market["symbol"] = market["symbol"].astype(str).str.strip().str.upper()
    market = market.dropna(subset=["trade_date", "symbol"]).drop_duplicates().reset_index(drop=True)
    return market, metadata


def _load_traditional_daily_universe(snapshot_root: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    manifest = _read_json_or_empty(snapshot_root / "manifest.json")
    quality = _read_json_or_empty(snapshot_root / "quality_report.json")
    path = snapshot_root / "daily_universe.parquet"
    if not path.exists():
        raise FileNotFoundError(f"traditional PIT daily_universe not found: {path}")
    universe = pd.read_parquet(path)
    required = {"date", "code", "is_listed_on_date", "is_mainboard", "is_common_a_share", "is_st_on_date", "is_suspended_on_date", "has_bar", "is_tradeable", "reject_reason"}
    missing = sorted(required - set(universe.columns))
    if missing:
        raise ValueError(f"traditional_pit_import_blocker: daily_universe missing columns {missing}")
    return universe, manifest, quality


def build_traditional_pit_status_sidecar_frame(
    *,
    snapshot_root: str | Path = DEFAULT_TRADITIONAL_SNAPSHOT_ROOT,
    market_keys: pd.DataFrame | None = None,
    start_date: str = "",
    end_date: str = "",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved_snapshot = resolve_traditional_snapshot_root(snapshot_root)
    universe, manifest, quality = _load_traditional_daily_universe(resolved_snapshot)
    data = universe.copy()
    data["trade_date"] = pd.to_datetime(data["date"], errors="coerce")
    if start_date:
        data = data.loc[data["trade_date"] >= pd.Timestamp(start_date)]
    if end_date:
        data = data.loc[data["trade_date"] <= pd.Timestamp(end_date)]
    data["trade_date"] = data["trade_date"].dt.strftime("%Y-%m-%d")
    data["symbol"] = data["code"].astype(str).str.strip().str.upper()
    data = data.dropna(subset=["trade_date", "symbol"])
    if market_keys is not None and not market_keys.empty:
        keys = market_keys.copy()
        keys["trade_date"] = pd.to_datetime(keys["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        keys["symbol"] = keys["symbol"].astype(str).str.strip().str.upper()
        keys = keys.dropna(subset=["trade_date", "symbol"]).drop_duplicates()
        before = int(len(data))
        data = data.merge(keys[["trade_date", "symbol"]], on=["trade_date", "symbol"], how="inner")
        market_key_rows = int(len(keys))
    else:
        before = int(len(data))
        market_key_rows = 0

    out = pd.DataFrame()
    out["symbol"] = data["symbol"]
    out["trade_date"] = data["trade_date"]
    out["is_listed_on_date"] = _to_bool_series(data["is_listed_on_date"])
    out["is_mainboard"] = _to_bool_series(data["is_mainboard"])
    out["is_common_a_share"] = _to_bool_series(data["is_common_a_share"])
    out["is_st"] = _to_bool_series(data["is_st_on_date"])
    out["is_suspended"] = _to_bool_series(data["is_suspended_on_date"])
    out["has_bar"] = _to_bool_series(data["has_bar"])
    trade_dates = pd.to_datetime(out["trade_date"], errors="coerce")
    out_dates = pd.to_datetime(data.get("out_date", pd.Series("", index=data.index)).replace("", pd.NA), errors="coerce")
    out["is_delisted"] = out_dates.notna() & (trade_dates >= out_dates)
    out["is_tradeable"] = _to_bool_series(data["is_tradeable"])
    out["reject_reason"] = data["reject_reason"].fillna("").astype(str)
    out["source"] = BRIDGE_SOURCE
    out = out[V2_STATUS_COLUMNS].sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    summary = summarize_v2_status_sidecar(out)
    summary.update(
        {
            "source": BRIDGE_SOURCE,
            "traditional_snapshot_id": str(manifest.get("snapshot_id", resolved_snapshot.name) or resolved_snapshot.name),
            "traditional_snapshot_path": str(resolved_snapshot),
            "traditional_manifest_dataset": dict(manifest.get("dataset", {}) or {}),
            "traditional_manifest_quality": dict(manifest.get("quality", {}) or {}),
            "traditional_quality_failure_count": int(quality.get("failure_count", 0) or 0),
            "traditional_rows_before_market_join": before,
            "market_key_rows": market_key_rows,
        }
    )
    return out, summary


def import_traditional_pit_status_sidecar(
    *,
    lake: ResearchDataLake,
    source_market_dataset_id: str,
    traditional_snapshot_root: str | Path = DEFAULT_TRADITIONAL_SNAPSHOT_ROOT,
    start_date: str = "2018-01-01",
    end_date: str = "2024-12-31",
    reuse: bool = True,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    market_keys, market_metadata = _read_market_keys(
        lake,
        source_market_dataset_id=str(source_market_dataset_id),
        start_date=str(start_date or ""),
        end_date=str(end_date or ""),
    )
    frame, summary = build_traditional_pit_status_sidecar_frame(
        snapshot_root=traditional_snapshot_root,
        market_keys=market_keys,
        start_date=str(start_date or ""),
        end_date=str(end_date or ""),
    )
    if frame.empty:
        raise ValueError("traditional_pit_import_blocker: imported status sidecar is empty")
    spec = {
        "dataset": f"data_platform_{V2_STATUS_SIDECAR_DOMAIN}",
        "source": BRIDGE_SOURCE,
        "source_market_dataset_id": str(source_market_dataset_id),
        "traditional_snapshot_id": str(summary.get("traditional_snapshot_id", "")),
        "traditional_snapshot_path": str(summary.get("traditional_snapshot_path", "")),
        "start_date": str(start_date or frame["trade_date"].min()),
        "end_date": str(end_date or frame["trade_date"].max()),
        "coverage_report": summary,
    }
    record = lake.save_domain_dataset(
        domain=V2_STATUS_SIDECAR_DOMAIN,
        frame=frame,
        spec=spec,
        source=BRIDGE_SOURCE,
        reuse=reuse,
    )
    summary.update(
        {
            "dataset_id": record.dataset_id,
            "source_market_dataset_id": str(source_market_dataset_id),
            "source_market_dataset_date_bounds": {
                "start_date": str(market_metadata.get("start_date", "")),
                "end_date": str(market_metadata.get("end_date", "")),
            },
        }
    )
    return record, frame, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import traditional_quant_research PIT universe as daily_research v2 status sidecar.")
    parser.add_argument("--data-lake-root", default="")
    parser.add_argument("--source-market-dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--traditional-snapshot-root", default=DEFAULT_TRADITIONAL_SNAPSHOT_ROOT)
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_ARTIFACT} has uncommitted diff.")
    lake = ResearchDataLake(str(args.data_lake_root or "").strip() or None)
    record, _frame, summary = import_traditional_pit_status_sidecar(
        lake=lake,
        source_market_dataset_id=str(args.source_market_dataset_id).strip(),
        traditional_snapshot_root=str(args.traditional_snapshot_root or "").strip(),
        start_date=str(args.start_date or "").strip(),
        end_date=str(args.end_date or "").strip(),
        reuse=not bool(args.refresh),
    )
    payload = {
        "status": "ok",
        "run_tag": str(args.run_tag),
        "dataset_id": record.dataset_id,
        "dataset_kind": record.dataset_kind,
        "fingerprint": record.fingerprint,
        "content_paths": record.content_paths,
        "row_counts": record.row_counts,
        "summary": summary,
    }
    lake.write_catalog_manifest()
    print(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    main()
