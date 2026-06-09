from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from daily_research.path_policy.forecast_dataset import normalize_static_context_fields


CANONICAL_MEMMAP_REGISTRY_SCHEMA_VERSION = 1
DEFAULT_CANONICAL_MEMMAP_ALIAS = "canonical_memmap_v1"
DEFAULT_CANONICAL_MEMMAP_REGISTRY = Path("daily_research/output/path_policy/canonical_memmaps/registry.json")


def default_registry_path() -> Path:
    return DEFAULT_CANONICAL_MEMMAP_REGISTRY


def manifest_signature(manifest: Mapping[str, Any]) -> dict[str, Any]:
    role_years = dict(manifest.get("role_years", {}) or {})
    static_schema = dict(manifest.get("static_context_schema", {}) or {})
    return {
        "source_market_dataset_id": str(manifest.get("source_market_dataset_id", "") or ""),
        "source_pool_view_id": str(manifest.get("source_pool_view_id", "") or ""),
        "source_sector_board_view_id": str(manifest.get("source_sector_board_view_id", "") or ""),
        "lookback_days": int(manifest.get("lookback_days", 0) or 0),
        "horizon": int(manifest.get("horizon", manifest.get("forecast_horizon", 0)) or 0),
        "cumulative_horizons": [int(item) for item in list(manifest.get("cumulative_horizons", []) or [])],
        "execution_mode": str(manifest.get("execution_mode", "") or ""),
        "feature_profile": str(manifest.get("feature_profile", "") or ""),
        "max_feature_columns": int(
            dict(manifest.get("feature_manifest", {}) or {}).get(
                "max_feature_columns",
                manifest.get("feature_count_after_cap", 0),
            )
            or 0
        ),
        "min_lookback_valid_ratio": float(manifest.get("min_lookback_valid_ratio", 0.0) or 0.0),
        "max_samples_per_role": int(manifest.get("max_samples_per_role", 0) or 0),
        "max_samples_per_date_per_role": int(manifest.get("max_samples_per_date_per_role", 0) or 0),
        "role_years": {
            "train_start_year": int(role_years.get("train_start_year", 0) or 0),
            "train_end_year": int(role_years.get("train_end_year", 0) or 0),
            "validation_year": int(role_years.get("validation_year", 0) or 0),
            "test_year": int(role_years.get("test_year", 0) or 0),
        },
        "static_context": {
            "enabled": bool(static_schema.get("enabled", False)),
            "fields": list(normalize_static_context_fields(static_schema.get("fields") or None)),
        },
    }


def request_signature(
    *,
    prepared: Any,
    args: Any,
    horizon: int,
    cumulative_horizons: tuple[int, ...] | list[int],
) -> dict[str, Any]:
    raw_cache_meta = dict(getattr(prepared, "raw_cache_meta", {}) or {}) if prepared is not None else {}
    pool_view = dict(raw_cache_meta.get("pool_view", {}) or {})
    sector_board = dict(raw_cache_meta.get("sector_board_view", {}) or {})
    source_market_dataset_id = str(
        pool_view.get("source_market_dataset_id", "")
        or raw_cache_meta.get("dataset_id", "")
        or getattr(args, "lake_dataset_id", "")
        or ""
    )
    return {
        "source_market_dataset_id": source_market_dataset_id,
        "source_pool_view_id": str(pool_view.get("dataset_id", "") or ""),
        "source_sector_board_view_id": str(sector_board.get("dataset_id", "") or ""),
        "lookback_days": int(getattr(args, "forecast_lookback_days", 0) or 0),
        "horizon": int(horizon),
        "cumulative_horizons": [int(item) for item in list(cumulative_horizons or [])],
        "execution_mode": str(getattr(args, "execution_mode", "") or ""),
        "feature_profile": str(getattr(args, "forecast_feature_profile", "") or ""),
        "max_feature_columns": int(getattr(args, "forecast_max_feature_columns", 0) or 0),
        "min_lookback_valid_ratio": float(getattr(args, "forecast_min_lookback_valid_ratio", 0.0) or 0.0),
        "max_samples_per_role": int(getattr(args, "forecast_max_samples_per_role", 0) or 0),
        "max_samples_per_date_per_role": int(getattr(args, "forecast_max_samples_per_date_per_role", 0) or 0),
        "role_years": {
            "train_start_year": int(getattr(args, "forecast_train_start_year", 0) or 0),
            "train_end_year": int(getattr(args, "forecast_train_end_year", 0) or 0),
            "validation_year": int(getattr(args, "forecast_validation_year", 0) or 0),
            "test_year": int(getattr(args, "forecast_test_year", 0) or 0),
        },
        "static_context": {
            "enabled": bool(getattr(args, "forecast_include_static_context", False)),
            "fields": list(normalize_static_context_fields(str(getattr(args, "forecast_static_fields", "") or ""))),
        },
    }


def signature_hash(signature: Mapping[str, Any]) -> str:
    payload = json.dumps(_json_safe(dict(signature)), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def register_memmap_manifest(
    manifest_path: str | Path,
    *,
    registry_path: str | Path | None = None,
    alias: str = DEFAULT_CANONICAL_MEMMAP_ALIAS,
) -> Path:
    path = Path(manifest_path)
    manifest = _read_json(path)
    if not manifest:
        raise FileNotFoundError(f"forecast_memmap_manifest_not_found: {path}")
    if str(manifest.get("dataset_mode", "") or "") != "memmap":
        raise ValueError(f"forecast_memmap_manifest_not_memmap: {path}")
    if not _manifest_files_exist(manifest):
        raise FileNotFoundError(f"forecast_memmap_manifest_files_missing: {path}")
    registry = _read_registry(registry_path)
    signature = manifest_signature(manifest)
    sig_hash = signature_hash(signature)
    entry = {
        "alias": str(alias or DEFAULT_CANONICAL_MEMMAP_ALIAS),
        "signature_hash": sig_hash,
        "signature": signature,
        "manifest_json": str(path.resolve()),
        "status": str(manifest.get("status", "") or ""),
        "sample_count": int(manifest.get("sample_count", 0) or 0),
        "feature_store_shape": list(manifest.get("feature_store_shape", []) or []),
        "source_market_dataset_id": str(manifest.get("source_market_dataset_id", "") or ""),
        "registered_at": _utc_now(),
    }
    entries = [
        item
        for item in list(registry.get("entries", []) or [])
        if not (
            str(dict(item).get("alias", "") or "") == entry["alias"]
            and str(dict(item).get("signature_hash", "") or "") == sig_hash
        )
    ]
    entries.append(entry)
    registry["entries"] = sorted(entries, key=lambda item: (str(item.get("alias", "")), str(item.get("signature_hash", ""))))
    registry["schema_version"] = CANONICAL_MEMMAP_REGISTRY_SCHEMA_VERSION
    registry["updated_at"] = _utc_now()
    out = Path(registry_path) if registry_path else default_registry_path()
    _write_json(out, registry)
    return out


def resolve_registered_memmap_manifest(
    *,
    prepared: Any,
    args: Any,
    horizon: int,
    cumulative_horizons: tuple[int, ...] | list[int],
    registry_path: str | Path | None = None,
    alias: str = DEFAULT_CANONICAL_MEMMAP_ALIAS,
) -> Path | None:
    registry_file = Path(registry_path) if registry_path else default_registry_path()
    if not registry_file.exists():
        return None
    registry = _read_registry(registry_file)
    signature = request_signature(prepared=prepared, args=args, horizon=horizon, cumulative_horizons=cumulative_horizons)
    sig_hash = signature_hash(signature)
    for entry in list(registry.get("entries", []) or []):
        row = dict(entry)
        if str(row.get("alias", "") or "") != str(alias or DEFAULT_CANONICAL_MEMMAP_ALIAS):
            continue
        if str(row.get("signature_hash", "") or "") != sig_hash:
            continue
        manifest_path = Path(str(row.get("manifest_json", "") or ""))
        if not manifest_path.exists():
            continue
        manifest = _read_json(manifest_path)
        if _manifest_files_exist(manifest):
            return manifest_path
    return None


def _manifest_files_exist(manifest: Mapping[str, Any]) -> bool:
    manifest_path = Path(str(manifest.get("manifest_json", "") or ""))
    root = manifest_path.parent if manifest_path else Path(str(manifest.get("feature_store_path", ".") or ".")).parent
    required = [
        "feature_store_path",
        "sample_index_csv",
        "manifest_json",
    ]
    for key in required:
        value = str(manifest.get(key, "") or "")
        if value and not Path(value).exists():
            return False
    for name in (
        "forecast_y_daily_excess.dat",
        "forecast_y_cum_excess.dat",
        "forecast_y_rank_by_horizon.dat",
        "forecast_y_rank_20d.dat",
        "forecast_y_max_drawdown_20d.dat",
        "forecast_y_worst_1d_20d.dat",
        "forecast_y_upside_20d.dat",
    ):
        if not (root / name).exists():
            return False
    for key in (
        "static_context_path",
        "y_daily_excess_path",
        "y_cum_excess_path",
        "y_rank_by_horizon_path",
        "y_rank_20d_path",
        "y_drawdown_by_horizon_path",
        "y_worst_by_horizon_path",
        "y_upside_by_horizon_path",
        "y_max_drawdown_20d_path",
        "y_worst_1d_20d_path",
        "y_upside_20d_path",
    ):
        value = str(manifest.get(key, "") or "")
        if value and not Path(value).exists():
            return False
    return True


def _read_registry(registry_path: str | Path | None) -> dict[str, Any]:
    path = Path(registry_path) if registry_path else default_registry_path()
    if not path.exists():
        return {"schema_version": CANONICAL_MEMMAP_REGISTRY_SCHEMA_VERSION, "entries": []}
    payload = _read_json(path)
    if not payload:
        return {"schema_version": CANONICAL_MEMMAP_REGISTRY_SCHEMA_VERSION, "entries": []}
    payload.setdefault("schema_version", CANONICAL_MEMMAP_REGISTRY_SCHEMA_VERSION)
    payload.setdefault("entries", [])
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(dict(payload)), ensure_ascii=False, indent=2), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
