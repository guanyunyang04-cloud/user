from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from daily_research.data_lake.catalog import DEFAULT_DATA_LAKE_ROOT, ResearchDataLake


DEFAULT_CANONICAL_ALIAS = "canonical_data_v1"
DEFAULT_CANONICAL_START_DATE = "2010-01-01"
DEFAULT_EXTERNAL_QUANT_DATA_ROOT = Path("H:/BaiduNetdiskDownload/量化数据")
DEFAULT_PATH_POLICY_ROOT = Path("daily_research/output/path_policy")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(dict(payload)), ensure_ascii=False, indent=2), encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_date_text(value: Any) -> str:
    if value is None or str(value).strip() == "":
        return ""
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def canonical_dir(lake: ResearchDataLake | str | Path | None = None) -> Path:
    root = lake.root if isinstance(lake, ResearchDataLake) else Path(lake or DEFAULT_DATA_LAKE_ROOT)
    return root / "canonical"


def canonical_manifest_path(
    lake: ResearchDataLake | str | Path | None = None,
    *,
    alias: str = DEFAULT_CANONICAL_ALIAS,
) -> Path:
    root = canonical_dir(lake)
    if str(alias or DEFAULT_CANONICAL_ALIAS) == DEFAULT_CANONICAL_ALIAS:
        return root / "canonical_manifest.json"
    return root / f"{_safe_name(alias)}_manifest.json"


def write_canonical_manifest(
    lake: ResearchDataLake | str | Path | None,
    *,
    dataset_id: str,
    alias: str = DEFAULT_CANONICAL_ALIAS,
    start_date: str = DEFAULT_CANONICAL_START_DATE,
    end_date: str = "",
    sidecar_dataset_ids: Mapping[str, str] | None = None,
    coverage_report: Mapping[str, Any] | None = None,
    notes: str = "",
    status: str = "active",
) -> Path:
    payload = {
        "alias": str(alias or DEFAULT_CANONICAL_ALIAS),
        "canonical_dataset_id": str(dataset_id),
        "dataset_id": str(dataset_id),
        "start_date": _normalize_date_text(start_date or DEFAULT_CANONICAL_START_DATE),
        "end_date": _normalize_date_text(end_date),
        "status": str(status or "active"),
        "sidecar_dataset_ids": dict(sidecar_dataset_ids or {}),
        "coverage_report": dict(coverage_report or {}),
        "notes": str(notes or ""),
        "updated_at": _utc_now(),
    }
    path = canonical_manifest_path(lake, alias=alias)
    _write_json(path, payload)
    return path


def load_canonical_manifest(
    lake: ResearchDataLake | str | Path | None = None,
    *,
    alias: str = DEFAULT_CANONICAL_ALIAS,
) -> dict[str, Any]:
    return _read_json(canonical_manifest_path(lake, alias=alias))


def resolve_canonical_dataset_id(
    lake: ResearchDataLake | str | Path | None = None,
    *,
    alias: str = DEFAULT_CANONICAL_ALIAS,
) -> str:
    manifest = load_canonical_manifest(lake, alias=alias)
    return str(manifest.get("canonical_dataset_id", "") or manifest.get("dataset_id", "") or "")


def build_coverage_report(
    lake: ResearchDataLake,
    *,
    start_date: str = DEFAULT_CANONICAL_START_DATE,
) -> dict[str, Any]:
    rows = lake.list_datasets()
    expected_start = pd.Timestamp(start_date)
    domains: dict[str, Any] = {}
    if rows.empty:
        return {"status": "empty_catalog", "start_date": start_date, "domains": domains}
    for _, row in rows.iterrows():
        dataset_kind = str(row.get("dataset_kind", "") or "")
        domain = str(dataset_kind.removeprefix("data_platform_") if dataset_kind.startswith("data_platform_") else row.get("domain", "") or "")
        if not domain:
            domain = dataset_kind
        start = _normalize_date_text(row.get("start_date", ""))
        end = _normalize_date_text(row.get("end_date", ""))
        if not start and not end:
            continue
        entry = domains.setdefault(domain, {"dataset_count": 0, "min_start_date": "", "max_end_date": "", "gaps": []})
        entry["dataset_count"] = int(entry["dataset_count"]) + 1
        if start:
            entry["min_start_date"] = min([item for item in [entry["min_start_date"], start] if item]) if entry["min_start_date"] else start
        if end:
            entry["max_end_date"] = max([item for item in [entry["max_end_date"], end] if item]) if entry["max_end_date"] else end
    for domain, entry in domains.items():
        min_start = str(entry.get("min_start_date", "") or "")
        if not min_start:
            entry["gaps"].append({"kind": "missing_start_date", "required_start_date": start_date})
        elif pd.Timestamp(min_start) > expected_start:
            entry["gaps"].append(
                {
                    "kind": "starts_after_canonical_start",
                    "required_start_date": start_date,
                    "observed_start_date": min_start,
                }
            )
        entry["status"] = "ok" if not entry["gaps"] else "gap"
    return {
        "status": "ok",
        "start_date": _normalize_date_text(start_date),
        "domains": domains,
        "generated_at": _utc_now(),
    }


def build_lake_inventory(
    lake: ResearchDataLake | str | Path | None = None,
    *,
    external_data_root: str | Path | None = None,
    path_policy_root: str | Path | None = None,
    start_date: str = DEFAULT_CANONICAL_START_DATE,
) -> dict[str, Any]:
    resolved_lake = lake if isinstance(lake, ResearchDataLake) else ResearchDataLake(lake or DEFAULT_DATA_LAKE_ROOT)
    rows = resolved_lake.list_datasets()
    data_lake_root = resolved_lake.root
    external_root = Path(external_data_root or DEFAULT_EXTERNAL_QUANT_DATA_ROOT)
    memmap_root = Path(path_policy_root or DEFAULT_PATH_POLICY_ROOT)
    catalog_fingerprints = {str(item) for item in rows["fingerprint"].dropna().tolist()} if "fingerprint" in rows.columns else set()
    datasets = _dataset_rows(rows)
    duplicate_groups = _duplicate_dataset_groups(rows)
    filesystem = _filesystem_inventory(data_lake_root, catalog_fingerprints)
    external_zips = _external_zip_inventory(external_root)
    memmaps = _memmap_inventory(memmap_root)
    dry_run = _build_cleanup_dry_run(filesystem=filesystem, memmaps=memmaps, rows=rows)
    return {
        "status": "ok",
        "generated_at": _utc_now(),
        "canonical_alias": DEFAULT_CANONICAL_ALIAS,
        "canonical_start_date": _normalize_date_text(start_date),
        "data_lake_root": str(data_lake_root.resolve()),
        "external_data_root": str(external_root.resolve()) if external_root.exists() else str(external_root),
        "path_policy_root": str(memmap_root.resolve()) if memmap_root.exists() else str(memmap_root),
        "dataset_count": int(len(datasets)),
        "datasets": datasets,
        "duplicate_dataset_groups": duplicate_groups,
        "filesystem": filesystem,
        "external_zips": external_zips,
        "memmaps": memmaps,
        "cleanup_dry_run": dry_run,
        "coverage_report": build_coverage_report(resolved_lake, start_date=start_date),
    }


def write_inventory_report(
    inventory: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> Path:
    path = Path(output_dir) / f"canonical_inventory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    _write_json(path, dict(inventory))
    return path


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value or "").strip())


def _dataset_rows(rows: pd.DataFrame) -> list[dict[str, Any]]:
    datasets: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        datasets.append(
            {
                "dataset_id": str(row.get("dataset_id", "") or ""),
                "dataset_kind": str(row.get("dataset_kind", "") or ""),
                "domain": str(row.get("domain", "") or ""),
                "zone": str(row.get("zone", "") or ""),
                "source": str(row.get("source", "") or ""),
                "start_date": _normalize_date_text(row.get("start_date", "")),
                "end_date": _normalize_date_text(row.get("end_date", "")),
                "fingerprint": str(row.get("fingerprint", "") or ""),
                "status": str(row.get("status", "") or ""),
            }
        )
    return datasets


def _duplicate_dataset_groups(rows: pd.DataFrame) -> list[dict[str, Any]]:
    if rows.empty:
        return []
    working = rows.copy()
    for column in ("dataset_kind", "domain", "source", "start_date", "end_date"):
        if column not in working.columns:
            working[column] = ""
    groups: list[dict[str, Any]] = []
    for key, group in working.groupby(["dataset_kind", "domain", "source", "start_date", "end_date"], dropna=False):
        if len(group) <= 1:
            continue
        dataset_ids = [str(item) for item in group["dataset_id"].tolist()]
        groups.append(
            {
                "dataset_kind": str(key[0]),
                "domain": str(key[1]),
                "source": str(key[2]),
                "start_date": _normalize_date_text(key[3]),
                "end_date": _normalize_date_text(key[4]),
                "dataset_count": int(len(dataset_ids)),
                "dataset_ids": dataset_ids,
            }
        )
    return groups


def _filesystem_inventory(root: Path, catalog_fingerprints: set[str]) -> dict[str, Any]:
    parquet_root = root / "parquet"
    empty_dirs: list[str] = []
    orphan_fingerprint_dirs: list[str] = []
    parquet_files = 0
    parquet_bytes = 0
    if parquet_root.exists():
        for path in parquet_root.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".parquet":
                parquet_files += 1
                try:
                    parquet_bytes += int(path.stat().st_size)
                except OSError:
                    pass
            elif path.is_dir():
                try:
                    is_empty = not any(path.iterdir())
                except OSError:
                    is_empty = False
                if is_empty:
                    empty_dirs.append(str(path.resolve()))
                if _looks_like_fingerprint(path.name) and path.name not in catalog_fingerprints:
                    orphan_fingerprint_dirs.append(str(path.resolve()))
    return {
        "parquet_root": str(parquet_root.resolve()) if parquet_root.exists() else str(parquet_root),
        "parquet_files": int(parquet_files),
        "parquet_bytes": int(parquet_bytes),
        "empty_dirs": empty_dirs,
        "orphan_fingerprint_dirs": orphan_fingerprint_dirs,
    }


def _looks_like_fingerprint(name: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{16,64}", str(name or "")))


def _external_zip_inventory(root: Path) -> dict[str, Any]:
    domains: dict[str, Any] = {
        "market_intraday_1m": {"zip_count": 0, "bytes": 0, "years": [], "paths": []},
        "market_intraday_5m": {"zip_count": 0, "bytes": 0, "years": [], "paths": []},
        "adjust_factor": {"zip_count": 0, "bytes": 0, "years": [], "paths": []},
        "unknown": {"zip_count": 0, "bytes": 0, "years": [], "paths": []},
    }
    if not root.exists():
        return {"status": "missing_root", "domains": domains}
    for path in root.rglob("*.zip"):
        domain = _classify_external_zip(path)
        entry = domains.setdefault(domain, {"zip_count": 0, "bytes": 0, "years": [], "paths": []})
        entry["zip_count"] = int(entry["zip_count"]) + 1
        try:
            entry["bytes"] = int(entry["bytes"]) + int(path.stat().st_size)
        except OSError:
            pass
        year = _year_from_path(path)
        if year:
            entry["years"].append(year)
        entry["paths"].append(str(path.resolve()))
    for entry in domains.values():
        entry["years"] = sorted({int(item) for item in entry["years"]})
        entry["paths"] = sorted(entry["paths"])
    return {"status": "ok", "domains": domains}


def _classify_external_zip(path: Path) -> str:
    text = str(path).lower()
    if "1分钟" in text or "1min" in text or "1m" in text:
        return "market_intraday_1m"
    if "5分钟" in text or "5min" in text or "5m" in text:
        return "market_intraday_5m"
    if "复权" in text or "adjust" in text:
        return "adjust_factor"
    return "unknown"


def _year_from_path(path: Path) -> int | None:
    matches = re.findall(r"(?:19|20)\d{2}", path.name)
    if not matches:
        matches = re.findall(r"(?:19|20)\d{2}", str(path.parent))
    if not matches:
        return None
    year = int(matches[-1])
    return year if 1900 <= year <= 2100 else None


def _memmap_inventory(root: Path) -> dict[str, Any]:
    manifests: list[dict[str, Any]] = []
    dat_files: list[dict[str, Any]] = []
    total_dat_bytes = 0
    if root.exists():
        for manifest_path in root.rglob("forecast_dataset_manifest.json"):
            manifest = _read_json(manifest_path)
            manifests.append(
                {
                    "manifest_json": str(manifest_path.resolve()),
                    "status": str(manifest.get("status", "") or ""),
                    "dataset_mode": str(manifest.get("dataset_mode", "") or ""),
                    "source_market_dataset_id": str(manifest.get("source_market_dataset_id", "") or ""),
                    "feature_store_shape": list(manifest.get("feature_store_shape", []) or []),
                    "sample_count": int(manifest.get("sample_count", 0) or 0),
                }
            )
        for dat_path in root.rglob("forecast_*.dat"):
            size = 0
            try:
                size = int(dat_path.stat().st_size)
            except OSError:
                pass
            total_dat_bytes += size
            dat_files.append({"path": str(dat_path.resolve()), "bytes": size})
    return {
        "root": str(root.resolve()) if root.exists() else str(root),
        "manifest_count": int(len(manifests)),
        "manifests": sorted(manifests, key=lambda item: item["manifest_json"]),
        "forecast_dat_count": int(len(dat_files)),
        "forecast_dat_bytes": int(total_dat_bytes),
        "forecast_dat_files": sorted(dat_files, key=lambda item: item["path"]),
    }


def _build_cleanup_dry_run(*, filesystem: Mapping[str, Any], memmaps: Mapping[str, Any], rows: pd.DataFrame) -> dict[str, Any]:
    deprecated_bundles: list[str] = []
    if not rows.empty and "dataset_kind" in rows.columns:
        bundles = rows.loc[rows["dataset_kind"].astype(str).eq("policy_input_bundle")].copy()
        for _, row in bundles.iterrows():
            deprecated_bundles.append(str(row.get("dataset_id", "") or ""))
    return {
        "destructive_actions_performed": False,
        "empty_dirs_to_remove": list(filesystem.get("empty_dirs", []) or []),
        "orphan_fingerprint_dirs_to_review": list(filesystem.get("orphan_fingerprint_dirs", []) or []),
        "old_forecast_dat_files_to_replace_after_registry_validation": [
            str(item.get("path", "") or "") for item in list(memmaps.get("forecast_dat_files", []) or [])
        ],
        "policy_input_bundles_to_mark_deprecated_after_canonical_cutover": deprecated_bundles,
    }
