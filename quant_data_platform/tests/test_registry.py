from __future__ import annotations

import json
from pathlib import Path

from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.core.registry import load_root_manifest, migrate_legacy_registry, registry_status


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_migrate_legacy_registry_writes_qdp_registry(tmp_path: Path) -> None:
    _write(tmp_path / "brain" / "brain_manifest.json", {"schema_version": 1})
    _write(
        tmp_path / "canonical_data" / "registry" / "root_manifest.json",
        {
            "schema_version": 1,
            "alias": "canonical_data_v1",
            "canonical_dataset_id": "policy_input_bundle__old",
            "canonical_memmap_registry": "legacy",
        },
    )
    _write(
        tmp_path / "canonical_data" / "registry" / "memmap_registry.json",
        {"schema_version": 1, "active_manifest_json": "manifest.json", "entries": []},
    )
    paths = qdp_paths(tmp_path)
    result = migrate_legacy_registry(paths)

    assert result["status"] == "written"
    assert paths.root_manifest.exists()
    assert paths.memmap_registry.exists()
    manifest = load_root_manifest(paths, fallback_legacy=False)
    assert manifest["registry_owner"] == "quant_data_platform"
    assert manifest["canonical_memmap_registry"].endswith("quant_data_platform/registry/memmap_registry.json")
    status = registry_status(paths)
    assert status["legacy_fallback_used"] is False
    assert status["canonical_dataset_id"] == "policy_input_bundle__old"
