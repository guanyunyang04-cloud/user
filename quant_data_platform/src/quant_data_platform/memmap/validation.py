from __future__ import annotations

from pathlib import Path
from typing import Any

from daily_research.path_policy.validate_forecast_memmap import validate_forecast_memmap_manifest

from quant_data_platform.core.registry import load_memmap_registry, load_root_manifest
from quant_data_platform.core.paths import QdpPaths, qdp_paths


def validate_active_memmap(
    paths: QdpPaths | None = None,
    *,
    manifest: str | Path | None = None,
    min_universe_size: int = 0,
    min_train_rows: int = 0,
) -> dict[str, Any]:
    resolved = paths or qdp_paths()
    registry = load_memmap_registry(resolved)
    root_manifest = load_root_manifest(resolved)
    manifest_path = Path(str(manifest or registry.get("active_manifest_json", "") or ""))
    if not str(manifest_path):
        return {"status": "blocked", "blockers": ["missing_active_memmap_manifest"]}
    return validate_forecast_memmap_manifest(
        manifest=manifest_path,
        expect_source_market_dataset_id=str(root_manifest.get("canonical_dataset_id", "") or ""),
        min_universe_size=int(min_universe_size or 0),
        min_train_rows=int(min_train_rows or 0),
    )
