from __future__ import annotations

import numpy as np

from quant_data_platform.cli import main as cli_main
from quant_data_platform.core.json_io import write_json
from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.memmap.sharded import _symbol_blocks, validate_sharded_memmap_manifest


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


def test_cli_sharded_dry_run_writes_plan(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("QDP_WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "brain").mkdir(parents=True)
    (tmp_path / "brain" / "brain_manifest.json").write_text("{}", encoding="utf-8")
    assert cli_main(["build-sharded-memmap", "--dry-run", "--json"]) == 0
    assert (qdp_paths(tmp_path).memmap_dir / "sharded_memmap_plan.json").exists()
