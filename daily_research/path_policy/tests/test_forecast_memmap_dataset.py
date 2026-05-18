from __future__ import annotations

import numpy as np
import torch

from daily_research.path_policy.forecast_dataset import build_forecast_memmap_dataset
from daily_research.path_policy.tests.fixtures import make_prepared_policy_inputs


def test_forecast_memmap_dataset_builds_lazy_store_and_batches(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=420, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2019-07-01")

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
