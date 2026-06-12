from __future__ import annotations

from pathlib import Path
from typing import Any

from daily_research.path_policy.validate_forecast_memmap import validate_forecast_memmap_manifest

from quant_data_platform.core.json_io import read_json
from quant_data_platform.core.registry import load_memmap_registry, load_root_manifest
from quant_data_platform.core.paths import QdpPaths, qdp_paths
from quant_data_platform.memmap.sharded import validate_sharded_memmap_manifest


def validate_active_memmap(
    paths: QdpPaths | None = None,
    *,
    manifest: str | Path | None = None,
    min_universe_size: int = 0,
    min_train_rows: int = 0,
) -> dict[str, Any]:
    resolved = paths or qdp_paths()
    registry = load_memmap_registry(resolved)
    sharded_registry = read_json(resolved.registry_dir / "sharded_memmap_registry.json")
    root_manifest = load_root_manifest(resolved)
    raw_manifest = str(
        manifest
        or sharded_registry.get("active_manifest_json", "")
        or registry.get("active_manifest_json", "")
        or ""
    ).strip()
    if not raw_manifest:
        return {"status": "blocked", "blockers": ["missing_active_memmap_manifest"]}
    manifest_path = Path(raw_manifest)
    payload = read_json(manifest_path)
    if str(payload.get("artifact_type", "")) == "qdp_sharded_memmap":
        report = validate_sharded_memmap_manifest(manifest_path)
        expected = str(root_manifest.get("canonical_dataset_id", "") or "")
        if expected and str(report.get("canonical_dataset_id", "") or "") != expected:
            report = {**report, "status": "blocked", "blockers": sorted(set(list(report.get("blockers", []) or []) + ["source_market_dataset_mismatch"]))}
        return report
    return validate_forecast_memmap_manifest(
        manifest=manifest_path,
        expect_source_market_dataset_id=str(root_manifest.get("canonical_dataset_id", "") or ""),
        min_universe_size=int(min_universe_size or 0),
        min_train_rows=int(min_train_rows or 0),
    )
