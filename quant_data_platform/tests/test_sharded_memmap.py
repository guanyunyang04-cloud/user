from __future__ import annotations

import numpy as np

from quant_data_platform.cli import main as cli_main
from quant_data_platform.core.json_io import read_json, write_json
from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.memmap import sharded
from quant_data_platform.memmap.sharded import (
    ShardedMemmapConfig,
    _build_shards_parallel,
    _project_feature_store_to_schema,
    _symbol_blocks,
    validate_sharded_memmap_manifest,
)


def test_symbol_blocks_preserve_order() -> None:
    assert list(_symbol_blocks(["A", "B", "C", "D", "E"], 2)) == [["A", "B"], ["C", "D"], ["E"]]


def test_validate_sharded_memmap_manifest_checks_files(tmp_path) -> None:
    feature = tmp_path / "feature_store.dat"
    np.memmap(feature, dtype="float32", mode="w+", shape=(2, 3, 4)).flush()
    label = tmp_path / "labels" / "daily_excess_return.dat"
    label.parent.mkdir(parents=True, exist_ok=True)
    np.memmap(label, dtype="float32", mode="w+", shape=(2, 3, 1)).flush()
    labels_manifest = tmp_path / "labels_manifest.json"
    write_json(labels_manifest, {"arrays": {"daily_excess_return": {"path": str(label), "shape": [2, 3, 1]}}})
    sample = tmp_path / "sample_index.parquet"
    sample.write_bytes(b"placeholder")
    shard_manifest = tmp_path / "shard_manifest.json"
    write_json(shard_manifest, {"status": "completed"})
    manifest = tmp_path / "sharded_memmap_manifest.json"
    write_json(
        manifest,
        {
            "artifact_type": "qdp_sharded_memmap",
            "canonical_dataset_id": "policy_input_bundle__x",
            "feature_schema_hash": "abc",
            "feature_count": 4,
            "symbol_count": 3,
            "years": [2022],
            "planned_shard_count": 1,
            "shards": [
                {
                    "status": "completed",
                    "feature_schema_hash": "abc",
                    "feature_store_path": str(feature),
                    "label_manifest_json": str(labels_manifest),
                    "sample_index_path": str(sample),
                    "shard_manifest_json": str(shard_manifest),
                }
            ],
        },
    )

    report = validate_sharded_memmap_manifest(manifest)
    assert report["status"] == "ok"
    assert report["stored_shard_count"] == 1


def test_validate_sharded_memmap_manifest_allows_empty_processed_shards(tmp_path) -> None:
    shard_manifest = tmp_path / "year=2010" / "block=0001" / "shard_manifest.json"
    write_json(
        shard_manifest,
        {
            "status": "no_coverage",
            "shard_key": "year=2010/block=0001",
            "empty_reason": "no_requested_symbols_available_in_lake_market_data",
        },
    )
    manifest = tmp_path / "sharded_memmap_manifest.json"
    write_json(
        manifest,
        {
            "artifact_type": "qdp_sharded_memmap",
            "canonical_dataset_id": "policy_input_bundle__x",
            "feature_schema_hash": "abc",
            "feature_count": 4,
            "symbol_count": 3,
            "years": [2010],
            "planned_shard_count": 1,
            "processed_shard_count": 1,
            "stored_shard_count": 0,
            "empty_shard_count": 1,
            "shards": [
                {
                    "status": "no_coverage",
                    "shard_key": "year=2010/block=0001",
                    "shard_manifest_json": str(shard_manifest),
                }
            ],
        },
    )

    report = validate_sharded_memmap_manifest(manifest)
    assert report["status"] == "ok"
    assert report["processed_shard_count"] == 1
    assert report["empty_shard_count"] == 1
    assert report["manifest_json"] == str(manifest.resolve())


def test_project_feature_store_to_schema_reorders_and_fills_missing(tmp_path) -> None:
    path = tmp_path / "forecast_feature_store.dat"
    store = np.memmap(path, dtype="float32", mode="w+", shape=(2, 1, 2))
    store[:, :, :] = np.array([[[1.0, 10.0]], [[2.0, 20.0]]], dtype=np.float32)
    store.flush()
    del store

    columns, manifest = _project_feature_store_to_schema(
        feature_store_path=path,
        feature_columns=["a", "b"],
        feature_manifest={"feature_store_shape": [2, 1, 2]},
        expected_feature_columns=["b", "c", "a"],
    )

    projected = np.memmap(path, dtype="float32", mode="r", shape=(2, 1, 3))
    assert columns == ["b", "c", "a"]
    assert manifest["feature_store_shape"] == [2, 1, 3]
    np.testing.assert_allclose(np.asarray(projected)[:, 0, 0], [10.0, 20.0])
    assert np.isnan(np.asarray(projected)[:, 0, 1]).all()
    np.testing.assert_allclose(np.asarray(projected)[:, 0, 2], [1.0, 2.0])


def test_cli_sharded_dry_run_writes_plan(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QDP_WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "brain").mkdir(parents=True)
    (tmp_path / "brain" / "brain_manifest.json").write_text('{"brain_type": "main"}', encoding="utf-8")
    assert cli_main(["build-sharded-memmap", "--dry-run", "--workers", "4", "--json"]) == 0
    plan = qdp_paths(tmp_path).memmap_dir / "sharded_memmap_plan.json"
    assert plan.exists()
    assert read_json(plan)["workers"] == 4


def test_sharded_config_normalizes_workers() -> None:
    assert ShardedMemmapConfig(workers=0).normalized().workers == 1
    assert ShardedMemmapConfig(workers=4).normalized().workers == 4


def test_build_shards_parallel_uses_worker_safe_shard_writes(tmp_path, monkeypatch) -> None:
    class FakeLake:
        root = tmp_path / "lake"

    def fake_worker(**kwargs):
        year = int(kwargs["year"])
        block_id = int(kwargs["block_id"])
        out_root = tmp_path / "out"
        shard_dir = out_root / f"year={year}" / f"block={block_id:04d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_manifest = shard_dir / "shard_manifest.json"
        payload = {
            "status": "completed",
            "shard_key": f"year={year}/block={block_id:04d}",
            "year": year,
            "block_id": block_id,
            "feature_schema_hash": "schema",
            "feature_columns": ["a", "b"],
            "feature_store_path": str(shard_dir / "feature.dat"),
            "label_manifest_json": str(shard_dir / "labels.json"),
            "sample_index_path": str(shard_dir / "sample.parquet"),
            "shard_manifest_json": str(shard_manifest),
        }
        write_json(shard_manifest, payload)
        return payload

    monkeypatch.setattr(sharded, "_build_one_shard_worker", fake_worker)

    results = _build_shards_parallel(
        lake=FakeLake(),
        dataset_id="policy_input_bundle__x",
        shard_specs=[
            {"shard_key": "year=2024/block=0000", "year": 2024, "block_id": 0, "symbols": ["A"]},
            {"shard_key": "year=2024/block=0001", "year": 2024, "block_id": 1, "symbols": ["B"]},
        ],
        canonical_start=np.datetime64("2024-01-01"),
        canonical_end=np.datetime64("2024-12-31"),
        cfg=ShardedMemmapConfig(workers=2).normalized(),
        out_root=tmp_path / "out",
        cumulative_horizons=(1, 3, 5),
        expected_feature_columns=["a", "b"],
        progress_path=tmp_path / "progress.json",
        planned_count=2,
        completed_so_far=0,
    )

    assert sorted(results) == ["year=2024/block=0000", "year=2024/block=0001"]
    assert (tmp_path / "progress.json").exists()
    assert all(item["status"] == "completed" for item in results.values())
