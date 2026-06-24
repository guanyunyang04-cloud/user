from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from daily_research.path_policy.shortline_raw_upside_diagnostic import build_shortline_raw_upside_diagnostic


def _write_memmap(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mmap = np.memmap(path, dtype=array.dtype, mode="w+", shape=array.shape)
    mmap[:] = array[:]
    mmap.flush()


def test_raw_upside_diagnostic_profiles_unnormalized_features(tmp_path: Path) -> None:
    root = tmp_path / "sharded"
    feature_columns = ["good_raw_signal", "bad_raw_signal"]
    shards = []
    for year in (2023, 2024, 2025):
        shard_dir = root / f"year={year}" / "block=0000"
        features = np.zeros((4, 20, 2), dtype=np.float32)
        labels = np.zeros((4, 20, 20), dtype=np.float32)
        for date_idx in range(4):
            for stock_idx in range(20):
                signal = float(stock_idx)
                target = (signal - 9.5) / 100.0
                features[date_idx, stock_idx, 0] = signal
                features[date_idx, stock_idx, 1] = -signal
                labels[date_idx, stock_idx, :] = target
        feature_path = shard_dir / "forecast_feature_store.dat"
        return_path = shard_dir / "labels" / "cumulative_return_1to20.dat"
        excess_path = shard_dir / "labels" / "cumulative_excess_return_1to20.dat"
        _write_memmap(feature_path, features)
        _write_memmap(return_path, labels)
        _write_memmap(excess_path, labels - 0.001)
        label_manifest = {
            "status": "completed",
            "label_store_kind": "path20_sharded_daily_symbol_panels",
            "label_schema_name": "path20_basic_v2",
            "label_schema_version": 2,
            "arrays": {
                "cumulative_return_1to20": {"path": str(return_path), "shape": list(labels.shape), "dtype": "float32"},
                "cumulative_excess_return_1to20": {"path": str(excess_path), "shape": list(labels.shape), "dtype": "float32"},
            },
        }
        label_manifest_path = shard_dir / "labels_manifest.json"
        label_manifest_path.write_text(json.dumps(label_manifest), encoding="utf-8")
        shards.append(
            {
                "status": "completed",
                "year": year,
                "block_id": 0,
                "feature_store_path": str(feature_path),
                "feature_store_shape": list(features.shape),
                "feature_columns": feature_columns,
                "label_manifest_json": str(label_manifest_path),
            }
        )
    manifest = {
        "artifact_type": "qdp_sharded_memmap",
        "feature_profile": "unit_test",
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "label_schema_name": "path20_basic_v2",
        "label_schema_version": 2,
        "shards": shards,
    }
    manifest_path = root / "sharded_memmap_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = build_shortline_raw_upside_diagnostic(
        sharded_manifest_json=manifest_path,
        output_root=tmp_path / "out",
        run_tag="unit",
        roles=("train", "validation", "test"),
        role_years="train:2023;validation:2024;test:2025",
        features=("good_raw_signal", "bad_raw_signal"),
        label_horizons=(1,),
        target_kinds=("net_abs",),
        round_trip_cost_bps=0.0,
        n_bins=3,
        max_sample_per_feature=1000,
    )

    metrics = pd.read_csv(report["outputs"]["feature_metrics_csv"])
    bins = pd.read_csv(report["outputs"]["feature_bins_csv"])
    stable = pd.read_csv(report["outputs"]["stable_corr_features_csv"])
    common = pd.read_csv(report["outputs"]["common_stable_corr_features_csv"])
    good = metrics.loc[metrics["feature"].eq("good_raw_signal")].iloc[0]
    bad = metrics.loc[metrics["feature"].eq("bad_raw_signal")].iloc[0]
    assert good["pearson_corr_raw"] > 0.99
    assert bad["pearson_corr_raw"] < -0.99
    assert good["future_top10_minus_bottom10_feature_mean"] > 0.0
    assert set(bins["feature"].astype(str)) == {"good_raw_signal", "bad_raw_signal"}
    assert set(stable["feature"].astype(str)) == {"good_raw_signal", "bad_raw_signal"}
    assert common.loc[common["feature"].eq("good_raw_signal"), "sign"].iloc[0] == "pos"
    assert common.loc[common["feature"].eq("bad_raw_signal"), "sign"].iloc[0] == "neg"
    assert Path(report["outputs"]["report_json"]).exists()


def test_raw_upside_diagnostic_rejects_missing_features(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"artifact_type": "qdp_sharded_memmap", "feature_columns": ["x"], "shards": []}),
        encoding="utf-8",
    )
    try:
        build_shortline_raw_upside_diagnostic(
            sharded_manifest_json=manifest_path,
            output_root=tmp_path / "out",
            features=("missing",),
        )
    except ValueError as exc:
        assert "none of requested features" in str(exc)
    else:
        raise AssertionError("expected ValueError")
