from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from daily_research.path_policy.shortline_raw_condition_matrix import build_shortline_raw_condition_matrix


def _write_memmap(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mmap = np.memmap(path, dtype=array.dtype, mode="w+", shape=array.shape)
    mmap[:] = array[:]
    mmap.flush()


def _fixture_manifest(tmp_path: Path, feature_columns: list[str]) -> Path:
    root = tmp_path / "sharded"
    shards = []
    date_values = ["2023-01-02", "2023-01-03", "2023-02-01", "2023-02-02"]
    for year, offset in ((2023, 0.0), (2024, 0.001), (2025, 0.002)):
        shard_dir = root / f"year={year}" / "block=0000"
        features = np.zeros((4, 40, len(feature_columns)), dtype=np.float32)
        labels = np.zeros((4, 40, 20), dtype=np.float32)
        optional = np.ones((4, 40), dtype=np.float32)
        for date_idx in range(features.shape[0]):
            for stock_idx in range(features.shape[1]):
                anti_hot = stock_idx < 24
                late_low = stock_idx % 4 >= 2
                target = 0.012 + offset if anti_hot and late_low else -0.006 + offset
                values = {
                    "ret_20d": -0.20 if anti_hot else 0.25,
                    "ret_10d": -0.10 if anti_hot else 0.15,
                    "ret_5d": -0.06 if anti_hot else 0.09,
                    "turn": 1.0 if anti_hot else 8.0,
                    "turn_z20": -0.5 if anti_hot else 1.0,
                    "raw_open_gap_1d": 0.01 if stock_idx % 3 else -0.01,
                    "raw_intraday_range_1d": 0.03,
                    "distance_to_20d_high": -0.12 if anti_hot else -0.01,
                    "local_drawdown_20d": -0.18 if anti_hot else -0.02,
                    "vol_20d": 0.015 if anti_hot else 0.04,
                    "intraday_last_30m_ret": 0.01 if late_low else -0.01,
                    "intraday_close_position": 0.75 if late_low else 0.25,
                    "intraday_low_time_frac": 0.85 if late_low else 0.20,
                    "intraday_high_before_low": 1.0 if late_low else 0.0,
                    "intraday_intraday_range": 0.025,
                    "market_positive_share_1d": 0.55,
                    "market_above_ma20_share": 0.50,
                    "market_amount_expansion_share": 0.45,
                    "industry_ret_5_excess": 0.01,
                    "industry_ret_20_excess": 0.02,
                    "stock_ret_5_minus_industry": -0.04 if anti_hot else 0.08,
                    "stock_ret_20_minus_industry": -0.12 if anti_hot else 0.20,
                }
                for col_idx, col in enumerate(feature_columns):
                    features[date_idx, stock_idx, col_idx] = float(values[col])
                labels[date_idx, stock_idx, :] = target
        feature_path = shard_dir / "forecast_feature_store.dat"
        return_path = shard_dir / "labels" / "cumulative_return_1to20.dat"
        excess_path = shard_dir / "labels" / "cumulative_excess_return_1to20.dat"
        tradeable_path = shard_dir / "labels" / "entry_tradeable.dat"
        limit_path = shard_dir / "labels" / "entry_limit_up_buy_blocked.dat"
        suspended_path = shard_dir / "labels" / "entry_suspended_or_no_open.dat"
        _write_memmap(feature_path, features)
        _write_memmap(return_path, labels)
        _write_memmap(excess_path, labels - 0.001)
        _write_memmap(tradeable_path, optional)
        _write_memmap(limit_path, np.zeros_like(optional))
        _write_memmap(suspended_path, np.zeros_like(optional))
        label_manifest = {
            "status": "completed",
            "label_store_kind": "path20_sharded_daily_symbol_panels",
            "label_schema_name": "path20_basic_v2",
            "label_schema_version": 2,
            "date_values": [value.replace("2023", str(year)) for value in date_values],
            "arrays": {
                "cumulative_return_1to20": {"path": str(return_path), "shape": list(labels.shape), "dtype": "float32"},
                "cumulative_excess_return_1to20": {"path": str(excess_path), "shape": list(labels.shape), "dtype": "float32"},
                "entry_tradeable": {"path": str(tradeable_path), "shape": list(optional.shape), "dtype": "float32"},
                "entry_limit_up_buy_blocked": {"path": str(limit_path), "shape": list(optional.shape), "dtype": "float32"},
                "entry_suspended_or_no_open": {"path": str(suspended_path), "shape": list(optional.shape), "dtype": "float32"},
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
    return manifest_path


def test_raw_condition_matrix_uses_train_thresholds_and_reports_stability(tmp_path: Path) -> None:
    feature_columns = [
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "turn",
        "turn_z20",
        "raw_open_gap_1d",
        "raw_intraday_range_1d",
        "distance_to_20d_high",
        "local_drawdown_20d",
        "vol_20d",
        "intraday_last_30m_ret",
        "intraday_close_position",
        "intraday_low_time_frac",
        "intraday_high_before_low",
        "intraday_intraday_range",
        "market_positive_share_1d",
        "market_above_ma20_share",
        "market_amount_expansion_share",
        "industry_ret_5_excess",
        "industry_ret_20_excess",
        "stock_ret_5_minus_industry",
        "stock_ret_20_minus_industry",
    ]
    manifest_path = _fixture_manifest(tmp_path, feature_columns)

    report = build_shortline_raw_condition_matrix(
        sharded_manifest_json=manifest_path,
        output_root=tmp_path / "out",
        run_tag="unit",
        roles=("train", "validation", "test"),
        role_years="train:2023;validation:2024;test:2025",
        label_horizons=(1,),
        target_kinds=("net_abs",),
        round_trip_cost_bps=0.0,
        min_count_for_stability=10,
    )

    metrics = pd.read_csv(report["outputs"]["condition_metrics_csv"])
    stable = pd.read_csv(report["outputs"]["stable_conditions_csv"])
    thresholds = pd.read_csv(report["outputs"]["thresholds_csv"])
    condition = metrics.loc[
        metrics["condition"].eq("low_runup_late_intraday_low")
        & metrics["role"].eq("test")
        & metrics["target_kind"].eq("net_abs")
    ].iloc[0]
    assert condition["target_mean"] > condition["baseline_target_mean"]
    assert condition["target_mean"] > 0.0
    assert condition["entry_tradeable_rate"] == 1.0
    assert "low_runup_late_intraday_low" in set(stable["condition"].astype(str))
    ret20 = thresholds.loc[thresholds["feature"].eq("ret_20d")].iloc[0]
    assert ret20["q40"] < 0.0
    assert Path(report["outputs"]["report_json"]).exists()


def test_raw_condition_matrix_rejects_missing_condition_features(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"artifact_type": "qdp_sharded_memmap", "feature_columns": ["ret_20d"], "shards": []}),
        encoding="utf-8",
    )
    try:
        build_shortline_raw_condition_matrix(
            sharded_manifest_json=manifest_path,
            output_root=tmp_path / "out",
        )
    except ValueError as exc:
        assert "default conditions require missing features" in str(exc)
    else:
        raise AssertionError("expected ValueError")
