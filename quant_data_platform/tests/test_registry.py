from __future__ import annotations

import json
from pathlib import Path

from quant_data_platform.core.paths import qdp_paths, workspace_root
from quant_data_platform.core.registry import load_root_manifest, migrate_legacy_registry, registry_status


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_migrate_legacy_registry_writes_qdp_registry(tmp_path: Path) -> None:
    _write(tmp_path / "brain" / "brain_manifest.json", {"schema_version": 1, "brain_type": "main"})
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


def test_registry_status_prefers_full_sharded_active_manifest(tmp_path: Path) -> None:
    _write(tmp_path / "brain" / "brain_manifest.json", {"schema_version": 1, "brain_type": "main"})
    paths = qdp_paths(tmp_path)
    _write(
        paths.root_manifest,
        {
            "schema_version": 1,
            "alias": "canonical_data_v1",
            "canonical_dataset_id": "policy_input_bundle__new",
            "canonical_sharded_memmap_status": {
                "status": "sharded_full_ready",
                "active_manifest_json": "H:/qdp/full/sharded_memmap_manifest.json",
                "latest_manifest_json": "H:/qdp/full/sharded_memmap_manifest.json",
                "latest_scope": "full_canonical_candidate",
                "latest_profile": "short_horizon_core_v1",
                "latest_canonical_dataset_id": "policy_input_bundle__new",
                "latest_stored_shard_count": 262,
                "latest_planned_shard_count": 323,
            },
        },
    )
    _write(
        paths.memmap_registry,
        {
            "schema_version": 1,
            "active_manifest_json": "H:/old/capped/forecast_dataset_manifest.json",
            "active_source_market_dataset_id": "policy_input_bundle__old",
            "active_scope": "capped_validation",
            "active_feature_profile": "old_profile",
        },
    )
    _write(
        paths.registry_dir / "sharded_memmap_registry.json",
        {
            "schema_version": 1,
            "active_manifest_json": "H:/qdp/full/sharded_memmap_manifest.json",
            "latest_manifest_json": "H:/qdp/full/sharded_memmap_manifest.json",
            "latest_scope": "full_canonical_candidate",
        },
    )

    status = registry_status(paths)
    assert status["active_memmap_manifest"] == "H:/qdp/full/sharded_memmap_manifest.json"
    assert status["active_memmap_source_market_dataset_id"] == "policy_input_bundle__new"
    assert status["active_memmap_state"] == "current"
    assert status["active_memmap_matches_canonical_bundle"] is True
    assert status["active_feature_profile"] == "short_horizon_core_v1"
    assert status["active_scope"] == "full_canonical_candidate"
    assert status["active_sample_count"] == 0


def test_explicit_workspace_root_does_not_inherit_parent_main_brain(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("QDP_WORKSPACE_ROOT", raising=False)
    _write(tmp_path / "brain" / "brain_manifest.json", {"schema_version": 1, "brain_type": "main"})
    child = tmp_path / "quant_data_platform"
    _write(child / "brain" / "brain_manifest.json", {"schema_version": 1, "brain_type": "sub_brain"})

    assert workspace_root(child) == child.resolve()
    assert qdp_paths(child).workspace_root == child.resolve()


def test_implicit_workspace_root_discovers_parent_main_brain(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("QDP_WORKSPACE_ROOT", raising=False)
    _write(tmp_path / "brain" / "brain_manifest.json", {"schema_version": 1, "brain_type": "main"})
    child = tmp_path / "quant_data_platform"
    _write(child / "brain" / "brain_manifest.json", {"schema_version": 1, "brain_type": "sub_brain"})
    monkeypatch.chdir(child)

    assert workspace_root() == tmp_path.resolve()
    assert qdp_paths().workspace_root == tmp_path.resolve()


def test_explicit_unmarked_workspace_is_an_isolation_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("QDP_WORKSPACE_ROOT", raising=False)

    assert workspace_root(tmp_path) == tmp_path.resolve()
    assert qdp_paths(tmp_path).workspace_root == tmp_path.resolve()


def test_env_workspace_root_overrides_explicit_start(tmp_path: Path, monkeypatch) -> None:
    env_root = tmp_path / "env_workspace"
    explicit_root = tmp_path / "explicit_workspace"
    monkeypatch.setenv("QDP_WORKSPACE_ROOT", str(env_root))

    assert workspace_root(explicit_root) == env_root.resolve()
    assert qdp_paths(explicit_root).workspace_root == env_root.resolve()
