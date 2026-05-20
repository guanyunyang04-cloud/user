from __future__ import annotations

import json

import numpy as np
import torch
import pytest

from daily_research.path_policy.forecast_dataset import (
    ForecastDateBatchTorchDataset,
    build_forecast_memmap_dataset,
    build_static_context_vocab,
    load_forecast_memmap_dataset,
)
from daily_research.path_policy.tests.fixtures import make_prepared_policy_inputs


def test_forecast_memmap_dataset_builds_lazy_store_and_batches(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=420, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2019-07-01")
    prepared.raw_cache_meta["dataset_id"] = "policy_input_bundle__unit"
    prepared.raw_cache_meta["pool_view"] = {
        "dataset_id": "policy_pool_view__unit",
        "view_kind": "rolling_liquidity",
        "view_name": "rolling_liquid500",
        "source_market_dataset_id": "policy_input_bundle__unit",
    }

    dataset = build_forecast_memmap_dataset(
        prepared,
        root=tmp_path,
        train_start_year=2019,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        lookback_days=5,
        horizon=20,
        max_samples_per_role=8,
        min_lookback_valid_ratio=0.80,
    )

    assert not hasattr(dataset, "x")
    assert dataset.manifest["source_market_dataset_id"] == "policy_input_bundle__unit"
    assert dataset.manifest["source_pool_view_id"] == "policy_pool_view__unit"
    assert dataset.manifest["source_pool_view_name"] == "rolling_liquid500"
    assert dataset.row_count > 0
    assert dataset.lookback_days == 5
    assert dataset.input_dim == len(dataset.feature_columns)
    assert dataset.manifest["dataset_mode"] == "memmap"
    assert dataset.manifest["role_purge_trading_days"] == 21
    assert dataset.normalization_manifest["fit_role"] == "train_only"
    assert dataset.manifest["feature_store_path"]
    assert dataset.manifest["sample_index_csv"]
    assert (tmp_path / "forecast_feature_store.dat").exists()
    assert (tmp_path / "forecast_sample_index.csv").exists()
    assert dataset.manifest["feature_store_shape"][2] == dataset.input_dim

    train_indices = dataset.role_indices("train")
    assert train_indices.size > 0
    torch_dataset = dataset.torch_dataset(train_indices[:3], target_scale=100.0)
    x, y_daily, y_cum, y_risk, row_idx = torch_dataset[0]

    assert tuple(x.shape) == (5, dataset.input_dim)
    assert tuple(y_daily.shape) == (20,)
    assert tuple(y_cum.shape) == (5,)
    assert tuple(y_risk.shape) == (3,)
    assert int(row_idx.item()) == int(train_indices[0])
    assert torch.isfinite(x).all()


def test_static_context_vocab_is_stable_and_memmap_samples_are_aligned(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=420, stocks=("AAA.SZ", "BBB.SH", "CCC.SZ"), start_date="2019-07-01")
    prepared.metadata_frames["industry_map"] = __import__("pandas").DataFrame(
        {
            "symbol": ["AAA.SZ", "BBB.SH"],
            "industry": ["bank", "electronics"],
        }
    )
    prepared.metadata_frames["board_membership"] = __import__("pandas").DataFrame(
        {
            "symbol": ["AAA.SZ", "AAA.SZ", "BBB.SH"],
            "board_kind": ["GN", "FG", "GN"],
            "board_name": ["value", "dividend", "semiconductor"],
            "board_code": ["GN001", "FG001", "GN002"],
        }
    )
    prepared.raw_cache_meta["sector_board_view"] = {
        "dataset_id": "policy_sector_board_view__unit",
        "view_kind": "latest_static_snapshot",
        "snapshot_semantics": "latest_static_snapshot",
    }

    first = build_static_context_vocab(prepared)
    reversed_prepared = make_prepared_policy_inputs(days=420, stocks=("CCC.SZ", "BBB.SH", "AAA.SZ"), start_date="2019-07-01")
    reversed_prepared.metadata_frames.update(prepared.metadata_frames)
    second = build_static_context_vocab(reversed_prepared)

    assert first["symbol_vocab_fingerprint"] == second["symbol_vocab_fingerprint"]
    assert first["industry_vocab_fingerprint"] == second["industry_vocab_fingerprint"]
    assert first["board_vocab_fingerprint"] == second["board_vocab_fingerprint"]

    dataset = build_forecast_memmap_dataset(
        prepared,
        root=tmp_path,
        train_start_year=2019,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        lookback_days=5,
        horizon=20,
        max_samples_per_role=6,
        min_lookback_valid_ratio=0.80,
        include_static_context=True,
    )

    assert dataset.manifest["static_context_schema"]["enabled"] is True
    assert dataset.manifest["symbol_vocab_fingerprint"] == first["symbol_vocab_fingerprint"]
    assert dataset.manifest["industry_vocab_fingerprint"] == first["industry_vocab_fingerprint"]
    assert dataset.static_context_ids is not None
    assert dataset.static_context_ids.shape == (dataset.row_count, 5)
    assert "static_context_path" in dataset.manifest
    assert dataset.sample_index["liquidity_bucket_id"].between(1, 5).all()
    assert dataset.sample_index["price_bucket_id"].between(1, 5).all()

    train_indices = dataset.role_indices("train")
    x, y_daily, y_cum, y_risk, row_idx, static_ids = dataset.torch_dataset(train_indices[:1], target_scale=100.0)[0]

    assert tuple(x.shape) == (5, dataset.input_dim)
    assert tuple(static_ids.shape) == (5,)
    assert int(static_ids[0].item()) == int(dataset.sample_index.iloc[int(row_idx.item())]["symbol_id"])
    assert tuple(y_daily.shape) == (20,)
    assert tuple(y_cum.shape) == (5,)
    assert tuple(y_risk.shape) == (3,)

    loaded = load_forecast_memmap_dataset(tmp_path / "forecast_dataset_manifest.json")
    assert loaded.static_context_ids is not None
    _, _, _, _, loaded_row_idx, loaded_static_ids = loaded.torch_dataset(loaded.role_indices("train")[:1], target_scale=100.0)[0]
    assert int(loaded_row_idx.item()) == int(train_indices[0])
    assert tuple(loaded_static_ids.shape) == (5,)


def test_date_batch_view_groups_memmap_samples_by_signal_date(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=420, stocks=("AAA.SZ", "BBB.SH", "CCC.SZ"), start_date="2019-07-01")
    dataset = build_forecast_memmap_dataset(
        prepared,
        root=tmp_path,
        train_start_year=2019,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        lookback_days=5,
        horizon=20,
        max_samples_per_role=9,
        min_lookback_valid_ratio=0.80,
        include_static_context=True,
    )

    date_view = ForecastDateBatchTorchDataset(dataset, dataset.role_indices("train"), target_scale=100.0)
    x, mask, y_daily, y_cum, y_risk, row_indices, static_ids = date_view[0]

    assert x.ndim == 3
    assert tuple(x.shape[1:]) == (5, dataset.input_dim)
    assert mask.dtype == torch.bool
    assert int(mask.sum().item()) == x.shape[0]
    assert tuple(y_daily.shape) == (x.shape[0], 20)
    assert tuple(y_cum.shape) == (x.shape[0], 5)
    assert tuple(y_risk.shape) == (x.shape[0], 3)
    assert tuple(static_ids.shape) == (x.shape[0], 5)
    assert len(set(dataset.sample_index.iloc[row_indices.numpy().astype(int)]["date"].astype(str))) == 1


def test_forecast_memmap_dataset_filters_low_history_samples(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=420, stocks=("AAA", "BBB", "CCC"), start_date="2019-07-01")
    prepared.open_.iloc[:6, prepared.open_.columns.get_loc("AAA")] = np.nan
    prepared.close.iloc[:6, prepared.close.columns.get_loc("AAA")] = np.nan
    prepared.high.iloc[:6, prepared.high.columns.get_loc("AAA")] = np.nan
    prepared.low.iloc[:6, prepared.low.columns.get_loc("AAA")] = np.nan
    prepared.volume.iloc[:6, prepared.volume.columns.get_loc("AAA")] = np.nan
    prepared.amount.iloc[:6, prepared.amount.columns.get_loc("AAA")] = np.nan

    dataset = build_forecast_memmap_dataset(
        prepared,
        root=tmp_path,
        train_start_year=2019,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        lookback_days=10,
        horizon=20,
        max_samples_per_role=12,
        min_lookback_valid_ratio=0.90,
    )

    assert dataset.manifest["dropped_low_history"] > 0
    assert dataset.manifest["min_lookback_valid_ratio"] == 0.90
    assert "history_valid_ratio_summary" in dataset.manifest
    assert set(dataset.sample_index["history_bucket"]).issubset({"low", "medium", "high"})


def test_forecast_memmap_dataset_loads_existing_manifest_without_rebuild(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=420, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2019-07-01")
    built = build_forecast_memmap_dataset(
        prepared,
        root=tmp_path,
        train_start_year=2019,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        lookback_days=5,
        horizon=20,
        max_samples_per_role=8,
        min_lookback_valid_ratio=0.80,
    )
    feature_store_mtime = (tmp_path / "forecast_feature_store.dat").stat().st_mtime_ns

    loaded = load_forecast_memmap_dataset(tmp_path / "forecast_dataset_manifest.json")

    assert loaded.manifest["artifact_reused"] is True
    assert loaded.row_count == built.row_count
    assert loaded.feature_store_shape == built.feature_store_shape
    assert loaded.feature_columns == built.feature_columns
    assert loaded.normalization_manifest["fit_role"] == "train_only"
    assert (tmp_path / "forecast_feature_store.dat").stat().st_mtime_ns == feature_store_mtime
    x, y_daily, y_cum, y_risk, row_idx = loaded.torch_dataset(loaded.role_indices("train")[:1], target_scale=100.0)[0]
    assert tuple(x.shape) == (5, loaded.input_dim)
    assert tuple(y_daily.shape) == (20,)
    assert tuple(y_cum.shape) == (5,)
    assert tuple(y_risk.shape) == (3,)
    assert int(row_idx.item()) >= 0


def test_forecast_memmap_dataset_rejects_inconsistent_reuse_manifest(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=420, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2019-07-01")
    built = build_forecast_memmap_dataset(
        prepared,
        root=tmp_path,
        train_start_year=2019,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        lookback_days=5,
        horizon=20,
        max_samples_per_role=8,
        min_lookback_valid_ratio=0.80,
    )
    manifest_path = tmp_path / "forecast_dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    bad_sample_manifest = {**manifest, "sample_count": int(built.row_count) - 1}
    bad_sample_path = tmp_path / "bad_sample_manifest.json"
    bad_sample_path.write_text(json.dumps(bad_sample_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="sample_count"):
        load_forecast_memmap_dataset(bad_sample_path)

    bad_shape_manifest = {**manifest, "feature_store_shape": [built.feature_store_shape[0], built.feature_store_shape[1], built.input_dim + 1]}
    bad_shape_path = tmp_path / "bad_shape_manifest.json"
    bad_shape_path.write_text(json.dumps(bad_shape_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="feature_store_shape"):
        load_forecast_memmap_dataset(bad_shape_path)
