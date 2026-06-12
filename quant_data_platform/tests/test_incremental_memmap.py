from __future__ import annotations

from pathlib import Path

from quant_data_platform.core.json_io import read_json, write_json
from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.memmap.incremental import (
    IncrementalPlanConfig,
    compose_sharded_memmap,
    freeze_sharded_memmap,
    plan_incremental_memmap,
)


def _workspace(tmp_path: Path) -> Path:
    (tmp_path / "brain").mkdir(parents=True)
    (tmp_path / "brain" / "brain_manifest.json").write_text('{"brain_type": "main"}', encoding="utf-8")
    paths = qdp_paths(tmp_path)
    paths.registry_dir.mkdir(parents=True, exist_ok=True)
    paths.memmap_dir.mkdir(parents=True, exist_ok=True)
    return tmp_path


def _manifest(path: Path, *, dataset_id: str = "policy_input_bundle__base", years: list[int] | None = None) -> Path:
    years = years or [2024, 2025]
    shards = []
    for year in years:
        for block in range(2):
            shards.append(
                {
                    "status": "completed",
                    "shard_key": f"year={year}/block={block:04d}",
                    "year": year,
                    "block_id": block,
                    "feature_store_path": str(path.parent / f"feature_{year}_{block}.dat"),
                    "label_manifest_json": str(path.parent / f"labels_{year}_{block}.json"),
                    "sample_index_path": str(path.parent / f"sample_{year}_{block}.parquet"),
                    "shard_manifest_json": str(path.parent / f"shard_{year}_{block}.json"),
                }
            )
    write_json(
        path,
        {
            "artifact_type": "qdp_sharded_memmap",
            "status": "completed",
            "scope": "full_canonical_candidate",
            "canonical_dataset_id": dataset_id,
            "profile": "short_horizon_core_v1",
            "feature_profile": "short_horizon_core_v1",
            "feature_schema_hash": "schema",
            "feature_count": 3,
            "feature_columns": ["a", "b", "c"],
            "symbol_block_size": 300,
            "lookback_days": 60,
            "horizon": 20,
            "execution_mode": "next_open",
            "start_year": min(years),
            "end_year": max(years),
            "years": years,
            "planned_shard_count": len(shards),
            "processed_shard_count": len(shards),
            "stored_shard_count": len(shards),
            "empty_shard_count": 0,
            "failed_shard_count": 0,
            "shards": shards,
            "manifest_json": str(path),
        },
    )
    return path


def test_freeze_sharded_memmap_registers_reusable_base(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QDP_WORKSPACE_ROOT", str(_workspace(tmp_path)))
    paths = qdp_paths(tmp_path)
    manifest = _manifest(tmp_path / "base" / "sharded_memmap_manifest.json")
    write_json(paths.registry_dir / "sharded_memmap_registry.json", {"active_manifest_json": str(manifest)})

    payload = freeze_sharded_memmap(paths, tag="base")

    assert payload["status"] == "frozen"
    registry = read_json(paths.registry_dir / "frozen_sharded_memmap_registry.json")
    assert registry["active_frozen_manifest_json"] == str(manifest.resolve())
    root = read_json(paths.root_manifest)
    assert root["canonical_sharded_memmap_freeze"]["status"] == "frozen_base_ready"


def test_plan_incremental_memmap_defaults_to_latest_tail_year(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QDP_WORKSPACE_ROOT", str(_workspace(tmp_path)))
    paths = qdp_paths(tmp_path)
    manifest = _manifest(tmp_path / "base" / "sharded_memmap_manifest.json", years=[2023, 2024, 2025])
    write_json(paths.registry_dir / "sharded_memmap_registry.json", {"active_manifest_json": str(manifest)})
    write_json(paths.root_manifest, {"canonical_dataset_id": "policy_input_bundle__new"})

    payload = plan_incremental_memmap(paths, config=IncrementalPlanConfig(recent_years=1), write=False)

    assert payload["rebuild_years"] == [2025]
    assert payload["frozen_years"] == [2023, 2024]
    assert payload["source_equivalence_required"] is True
    assert "--start-year 2025 --end-year 2025" in payload["recommended_build_command"]


def test_compose_sharded_memmap_overlays_matching_shard_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QDP_WORKSPACE_ROOT", str(_workspace(tmp_path)))
    paths = qdp_paths(tmp_path)
    base = _manifest(tmp_path / "base" / "sharded_memmap_manifest.json", years=[2024, 2025])
    overlay = _manifest(tmp_path / "overlay" / "sharded_memmap_manifest.json", dataset_id="policy_input_bundle__new", years=[2025])
    write_json(paths.root_manifest, {"canonical_dataset_id": "policy_input_bundle__new"})

    payload = compose_sharded_memmap(
        paths,
        base_manifest=base,
        overlay_manifests=[overlay],
        tag="composite",
        activate=True,
    )

    assert payload["status"] == "completed"
    assert payload["scope"] == "full_canonical_incremental_composite"
    assert payload["canonical_dataset_id"] == "policy_input_bundle__new"
    source_by_key = payload["shard_source_manifest_by_key"]
    assert source_by_key["year=2024/block=0000"] == str(base.resolve())
    assert source_by_key["year=2025/block=0000"] == str(overlay.resolve())
    registry = read_json(paths.registry_dir / "sharded_memmap_registry.json")
    assert registry["active_manifest_json"] == payload["manifest_json"]
