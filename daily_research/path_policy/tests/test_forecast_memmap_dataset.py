from __future__ import annotations

import json
from argparse import Namespace

import numpy as np
import pandas as pd
import torch
import pytest

from daily_research.path_policy.canonical_memmap import (
    register_memmap_manifest,
    resolve_registered_memmap_manifest,
)
from daily_research.path_policy.forecast_dataset import (
    ForecastDateBatchTorchDataset,
    _fit_memmap_train_normalization,
    build_forecast_memmap_dataset,
    build_qdp_training_pack,
    build_qdp_training_pack_date_major_layout,
    build_static_context_vocab,
    ForecastShardedMemmapDataset,
    ForecastTrainingPackDataset,
    load_forecast_memmap_dataset,
)
from daily_research.path_policy.tests.fixtures import make_prepared_policy_inputs
from daily_research.path_policy.validate_forecast_memmap import main as validate_memmap_main
from daily_research.path_policy.validate_forecast_memmap import validate_forecast_memmap_manifest


def _write_float_memmap(path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    store = np.memmap(path, dtype="float32", mode="w+", shape=values.shape)
    store[...] = values.astype(np.float32, copy=False)
    store.flush()


def _write_qdp_fixture_shard(root, *, year: int, stocks: tuple[str, ...], feature_columns: list[str], include_label_v2: bool = False) -> dict:
    dates = [pd.Timestamp(item).strftime("%Y-%m-%d") for item in pd.bdate_range(f"{year}-01-01", periods=6)]
    shard_dir = root / f"year={year}" / "block=0000"
    feature_shape = (len(dates), len(stocks), len(feature_columns))
    features = np.zeros(feature_shape, dtype=np.float32)
    for date_pos in range(feature_shape[0]):
        for stock_pos in range(feature_shape[1]):
            for feature_pos in range(feature_shape[2]):
                features[date_pos, stock_pos, feature_pos] = year * 100 + date_pos * 10 + stock_pos + feature_pos / 10.0
    feature_path = shard_dir / "forecast_feature_store.dat"
    _write_float_memmap(feature_path, features)

    horizon = 1
    cumulative_horizons = [1]
    labels_dir = shard_dir / "labels"
    arrays = {
        "daily_excess_return": np.full((len(dates), len(stocks), horizon), 0.01, dtype=np.float32),
        "cumulative_excess_return": np.full((len(dates), len(stocks), 1), 0.01, dtype=np.float32),
        "rank_by_horizon": np.tile(np.arange(len(stocks), dtype=np.float32).reshape(1, len(stocks), 1), (len(dates), 1, 1)),
        "drawdown_by_horizon": np.full((len(dates), len(stocks), 1), -0.02, dtype=np.float32),
        "worst_by_horizon": np.full((len(dates), len(stocks), 1), -0.01, dtype=np.float32),
        "upside_by_horizon": np.full((len(dates), len(stocks), 1), 0.03, dtype=np.float32),
        "rank_20d": np.tile(np.arange(len(stocks), dtype=np.float32).reshape(1, len(stocks)), (len(dates), 1)),
        "max_drawdown_20d": np.full((len(dates), len(stocks)), -0.02, dtype=np.float32),
        "worst_1d_20d": np.full((len(dates), len(stocks)), -0.01, dtype=np.float32),
        "upside_20d": np.full((len(dates), len(stocks)), 0.03, dtype=np.float32),
    }
    if include_label_v2:
        rank = np.tile(np.arange(len(stocks), dtype=np.float32).reshape(1, len(stocks), 1), (len(dates), 1, 1))
        arrays.update(
            {
                "daily_return": np.full((len(dates), len(stocks), horizon), 0.02, dtype=np.float32),
                "benchmark_daily_return": np.full((len(dates), len(stocks), horizon), 0.01, dtype=np.float32),
                "cumulative_return": np.full((len(dates), len(stocks), 1), 0.02, dtype=np.float32),
                "benchmark_cumulative_return": np.full((len(dates), len(stocks), 1), 0.01, dtype=np.float32),
                "cumulative_return_1to20": np.full((len(dates), len(stocks), horizon), 0.02, dtype=np.float32),
                "benchmark_cumulative_return_1to20": np.full((len(dates), len(stocks), horizon), 0.01, dtype=np.float32),
                "cumulative_excess_return_1to20": np.full((len(dates), len(stocks), horizon), 0.01, dtype=np.float32),
                "rank_1to20": rank,
                "industry_rank_by_horizon": rank,
                "entry_tradeable": np.ones((len(dates), len(stocks)), dtype=np.float32),
                "entry_limit_up_buy_blocked": np.zeros((len(dates), len(stocks)), dtype=np.float32),
                "entry_suspended_or_no_open": np.zeros((len(dates), len(stocks)), dtype=np.float32),
                "forward_tradeable_ratio_by_horizon": np.ones((len(dates), len(stocks), 1), dtype=np.float32),
            }
        )
    label_entries = {}
    for name, values in arrays.items():
        path = labels_dir / f"{name}.dat"
        _write_float_memmap(path, values)
        label_entries[name] = {"path": str(path), "shape": list(values.shape), "dtype": "float32"}
    label_manifest = {
        "status": "completed",
        "date_values": dates,
        "stock_values": list(stocks),
        "horizon": horizon,
        "cumulative_horizons": cumulative_horizons,
        "arrays": label_entries,
        "label_schema_name": "path20_basic_v2" if include_label_v2 else "path20_legacy_v1",
        "label_schema_version": 2 if include_label_v2 else 1,
        "label_metadata": {"execution_mode": "next_open"},
    }
    label_manifest_path = shard_dir / "labels_manifest.json"
    label_manifest_path.write_text(json.dumps(label_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    sample_rows = []
    for date_pos, date in enumerate(dates):
        for stock_pos, stock in enumerate(stocks):
            sample_rows.append(
                {
                    "date": date,
                    "stock": stock,
                    "date_pos": date_pos,
                    "stock_pos": stock_pos,
                    "history_valid_ratio": 1.0,
                    "symbol_id": stock_pos + 1,
                    "exchange_id": 1 if stock.endswith(".SH") else 2,
                    "industry_id": stock_pos + 10,
                }
            )
    sample_index_path = shard_dir / "sample_index.parquet"
    pd.DataFrame(sample_rows).to_parquet(sample_index_path, index=False)
    shard = {
        "status": "completed",
        "shard_key": f"year={year}/block=0000",
        "year": year,
        "block_id": 0,
        "start_date": dates[0],
        "end_date": dates[-1],
        "context_start_date": dates[0],
        "context_end_date": dates[-1],
        "symbol_count": len(stocks),
        "date_count": len(dates),
        "sample_count": len(sample_rows),
        "feature_store_path": str(feature_path),
        "feature_store_shape": list(feature_shape),
        "feature_columns": list(feature_columns),
        "feature_schema_hash": "unit",
        "label_manifest_json": str(label_manifest_path),
        "sample_index_path": str(sample_index_path),
        "static_context_schema": {
            "enabled": True,
            "fields": ["symbol", "exchange", "industry"],
            "id_columns": ["symbol_id", "exchange_id", "industry_id"],
            "vocab_sizes": {"symbol": 3, "exchange": 3, "industry": 12},
        },
        "shard_manifest_json": str(shard_dir / "shard_manifest.json"),
    }
    (shard_dir / "shard_manifest.json").write_text(json.dumps(shard, ensure_ascii=False, indent=2), encoding="utf-8")
    return shard


def _make_fast_prepared(
    *,
    days: int = 170,
    stocks: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD"),
    start_date: str = "2019-10-15",
):
    return make_prepared_policy_inputs(days=days, stocks=stocks, start_date=start_date)


def _build_fast_forecast_memmap(
    root,
    *,
    prepared=None,
    days: int = 170,
    stocks: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD"),
    start_date: str = "2019-10-15",
    train_start_year: int = 2019,
    train_end_year: int = 2019,
    validation_year: int = 2020,
    test_year: int = 2021,
    lookback_days: int = 5,
    horizon: int = 20,
    cumulative_horizons=None,
    max_samples_per_role: int = 8,
    max_samples_per_date_per_role: int = 0,
    min_lookback_valid_ratio: float = 0.80,
    max_feature_columns: int = 32,
    include_static_context: bool = False,
    static_context_fields=None,
):
    prepared = prepared if prepared is not None else _make_fast_prepared(days=days, stocks=stocks, start_date=start_date)
    return build_forecast_memmap_dataset(
        prepared,
        root=root,
        train_start_year=train_start_year,
        train_end_year=train_end_year,
        validation_year=validation_year,
        test_year=test_year,
        lookback_days=lookback_days,
        horizon=horizon,
        cumulative_horizons=cumulative_horizons,
        max_samples_per_role=max_samples_per_role,
        max_samples_per_date_per_role=max_samples_per_date_per_role,
        max_feature_columns=max_feature_columns,
        min_lookback_valid_ratio=min_lookback_valid_ratio,
        include_static_context=include_static_context,
        static_context_fields=static_context_fields,
    )


def test_forecast_memmap_dataset_builds_lazy_store_and_batches(tmp_path) -> None:
    prepared = _make_fast_prepared()
    prepared.raw_cache_meta["dataset_id"] = "policy_input_bundle__unit"
    prepared.raw_cache_meta["pool_view"] = {
        "dataset_id": "policy_pool_view__unit",
        "view_kind": "rolling_liquidity",
        "view_name": "rolling_liquid500",
        "source_market_dataset_id": "policy_input_bundle__unit",
    }

    dataset = _build_fast_forecast_memmap(
        tmp_path,
        prepared=prepared,
        test_year=2020,
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
    assert tuple(y_risk.shape) == (5, 3)
    assert int(row_idx.item()) == int(train_indices[0])
    assert torch.isfinite(x).all()


def test_qdp_sharded_memmap_loader_builds_role_view_and_cross_year_windows(tmp_path) -> None:
    stocks = ("AAA.SZ", "BBB.SH")
    feature_columns = ["feature_a", "feature_b"]
    shards = [
        _write_qdp_fixture_shard(tmp_path, year=year, stocks=stocks, feature_columns=feature_columns)
        for year in (2019, 2020, 2021)
    ]
    manifest = {
        "artifact_type": "qdp_sharded_memmap",
        "profile": "style_structural_v1",
        "feature_profile": "style_structural_v1",
        "canonical_dataset_id": "policy_input_bundle__unit",
        "source_pool_view_id": "policy_pool_view__unit",
        "source_pool_view_kind": "tradeable_mainboard",
        "lookback_days": 3,
        "horizon": 1,
        "forecast_horizon": 1,
        "execution_mode": "next_open",
        "cumulative_horizons": [1],
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "static_context_schema": {
            "enabled": True,
            "fields": ["symbol", "exchange", "industry"],
            "id_columns": ["symbol_id", "exchange_id", "industry_id"],
            "vocab_sizes": {"symbol": 3, "exchange": 3, "industry": 12},
            "embedding_defaults": {"symbol": 16, "exchange": 4, "industry": 8, "dropout": 0.2},
        },
        "shards": shards,
    }
    manifest_path = tmp_path / "sharded_memmap_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_forecast_memmap_dataset(
        manifest_path,
        train_start_year=2019,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        max_samples_per_role=4,
    )

    assert isinstance(loaded, ForecastShardedMemmapDataset)
    assert loaded.manifest["qdp_sharded_memmap_reused"] is True
    assert loaded.manifest["sample_count_by_role"] == {"train": 4, "validation": 4, "test": 4}
    assert loaded.static_context_ids is not None
    assert loaded.static_context_ids.shape == (loaded.row_count, 3)

    validation_row = int(loaded.role_indices("validation")[0])
    raw_window = loaded.raw_input_window(validation_row)
    assert tuple(raw_window.shape) == (3, 2)
    assert raw_window[:, 0].tolist() == [2019 * 100 + 40, 2019 * 100 + 50, 2020 * 100]
    batch_rows = loaded.cache_friendly_indices(loaded.role_indices("validation")[:3])
    batch_raw = loaded.raw_input_windows(batch_rows)
    assert tuple(batch_raw.shape) == (3, 3, 2)
    np.testing.assert_allclose(batch_raw[0], loaded.raw_input_window(int(batch_rows[0])), equal_nan=True)
    batch_x = loaded.input_windows(batch_rows)
    np.testing.assert_allclose(batch_x[0], loaded.input_window(int(batch_rows[0])), rtol=1e-6, atol=1e-6)

    x, y_daily, y_cum, y_risk, row_idx, static_ids = loaded.torch_dataset(np.asarray([validation_row]), target_scale=100.0)[0]
    assert tuple(x.shape) == (3, 2)
    assert tuple(y_daily.shape) == (1,)
    assert tuple(y_cum.shape) == (1,)
    assert tuple(y_risk.shape) == (1, 3)
    assert int(row_idx.item()) == validation_row
    assert tuple(static_ids.shape) == (3,)
    assert torch.isfinite(x).all()


def test_qdp_training_pack_loads_stock_major_features_and_sample_labels(tmp_path) -> None:
    stocks = ("AAA.SZ", "BBB.SH")
    feature_columns = ["feature_a", "feature_b"]
    shards = [
        _write_qdp_fixture_shard(tmp_path, year=year, stocks=stocks, feature_columns=feature_columns)
        for year in (2019, 2020, 2021)
    ]
    manifest = {
        "artifact_type": "qdp_sharded_memmap",
        "profile": "style_structural_v1",
        "feature_profile": "style_structural_v1",
        "canonical_dataset_id": "policy_input_bundle__unit",
        "source_pool_view_id": "policy_pool_view__unit",
        "source_pool_view_kind": "tradeable_mainboard",
        "lookback_days": 3,
        "horizon": 1,
        "forecast_horizon": 1,
        "execution_mode": "next_open",
        "cumulative_horizons": [1],
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "static_context_schema": {
            "enabled": True,
            "fields": ["symbol", "exchange", "industry"],
            "id_columns": ["symbol_id", "exchange_id", "industry_id"],
            "vocab_sizes": {"symbol": 3, "exchange": 3, "industry": 12},
            "embedding_defaults": {"symbol": 16, "exchange": 4, "industry": 8, "dropout": 0.2},
        },
        "shards": shards,
    }
    source_manifest_path = tmp_path / "sharded_memmap_manifest.json"
    source_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    sharded = load_forecast_memmap_dataset(
        source_manifest_path,
        train_start_year=2019,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        max_samples_per_role=4,
    )
    pack_manifest = build_qdp_training_pack(
        source_manifest_path,
        output_root=tmp_path / "training_pack",
        train_start_year=2019,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        max_samples_per_role=4,
        feature_dtype="float32",
    )
    loaded = load_forecast_memmap_dataset(pack_manifest["manifest_json"])

    assert isinstance(loaded, ForecastTrainingPackDataset)
    assert loaded.manifest["artifact_type"] == "qdp_training_pack_v1"
    assert loaded.manifest["sample_count_by_role"] == {"train": 4, "validation": 4, "test": 4}
    validation_row = int(loaded.role_indices("validation")[0])
    sharded_validation_row = int(sharded.role_indices("validation")[0])
    np.testing.assert_allclose(loaded.input_window(validation_row), sharded.input_window(sharded_validation_row), rtol=1e-6, atol=1e-6)
    batch_rows = loaded.cache_friendly_indices(loaded.role_indices("validation")[:3])
    assert tuple(loaded.input_windows(batch_rows).shape) == (3, 3, 2)
    x, y_daily, y_cum, y_risk, row_idx, static_ids = loaded.batch_torch_dataset(
        np.asarray([validation_row]),
        batch_size=1,
        target_scale=100.0,
    )[0]
    assert tuple(x.shape) == (1, 3, 2)
    assert tuple(y_daily.shape) == (1, 1)
    assert tuple(y_cum.shape) == (1, 1)
    assert tuple(y_risk.shape) == (1, 1, 3)
    assert int(row_idx[0].item()) == validation_row
    assert tuple(static_ids.shape) == (1, 3)


def test_qdp_training_pack_date_major_layout_powers_date_batches(tmp_path) -> None:
    stocks = ("AAA.SZ", "BBB.SH", "CCC.SZ")
    feature_columns = ["feature_a", "feature_b"]
    shards = [
        _write_qdp_fixture_shard(tmp_path, year=year, stocks=stocks, feature_columns=feature_columns)
        for year in (2019, 2020, 2021)
    ]
    manifest = {
        "artifact_type": "qdp_sharded_memmap",
        "profile": "style_structural_v1",
        "feature_profile": "style_structural_v1",
        "canonical_dataset_id": "policy_input_bundle__unit",
        "source_pool_view_id": "policy_pool_view__unit",
        "source_pool_view_kind": "tradeable_mainboard",
        "lookback_days": 3,
        "horizon": 1,
        "forecast_horizon": 1,
        "execution_mode": "next_open",
        "cumulative_horizons": [1],
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "static_context_schema": {
            "enabled": True,
            "fields": ["symbol", "exchange", "industry"],
            "id_columns": ["symbol_id", "exchange_id", "industry_id"],
            "vocab_sizes": {"symbol": 4, "exchange": 3, "industry": 12},
            "embedding_defaults": {"symbol": 16, "exchange": 4, "industry": 8, "dropout": 0.2},
        },
        "shards": shards,
    }
    source_manifest_path = tmp_path / "sharded_memmap_manifest.json"
    source_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pack_manifest = build_qdp_training_pack(
        source_manifest_path,
        output_root=tmp_path / "training_pack",
        train_start_year=2019,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        max_samples_per_role=6,
        feature_dtype="float32",
    )
    stock_major = load_forecast_memmap_dataset(pack_manifest["manifest_json"])
    assert isinstance(stock_major, ForecastTrainingPackDataset)
    validation = stock_major.sample_index.loc[stock_major.sample_index["role"].astype(str).eq("validation")]
    rows = next(iter(validation.groupby("date", sort=True)))[1].index.to_numpy(dtype=np.int64)
    expected = np.stack([stock_major.input_window(int(row_idx)) for row_idx in rows], axis=0)
    np.testing.assert_allclose(stock_major.date_input_windows(rows), expected, rtol=1e-6, atol=1e-6)

    updated_manifest = build_qdp_training_pack_date_major_layout(
        pack_manifest["manifest_json"],
        feature_dtype="float32",
        stock_chunk_size=1,
        resume=False,
    )
    loaded = load_forecast_memmap_dataset(updated_manifest["manifest_json"])

    assert isinstance(loaded, ForecastTrainingPackDataset)
    assert loaded.has_date_major_feature_panel
    assert updated_manifest["training_pack_contract"]["date_batch_feature_layout"] == "date_stock_feature"
    assert updated_manifest["regime_auxiliary_layout"]["enabled"] is True
    assert (tmp_path / "training_pack" / "cross_section_index.parquet").exists()
    np.testing.assert_allclose(loaded.date_input_windows(rows), expected, rtol=1e-6, atol=1e-6)
    date_view = ForecastDateBatchTorchDataset(loaded, rows, target_scale=1.0)
    x, mask, y_daily, y_cum, y_risk, row_indices, static_ids = date_view[0]
    np.testing.assert_allclose(x.numpy(), expected, rtol=1e-6, atol=1e-6)
    assert tuple(mask.shape) == (len(rows),)
    assert tuple(y_daily.shape) == (len(rows), 1)
    assert tuple(y_cum.shape) == (len(rows), 1)
    assert tuple(y_risk.shape) == (len(rows), 1, 3)
    assert row_indices.numpy().astype(np.int64).tolist() == rows.tolist()
    assert tuple(static_ids.shape) == (len(rows), 3)


def test_qdp_training_pack_preserves_optional_basic_v2_labels(tmp_path) -> None:
    stocks = ("AAA.SZ", "BBB.SH")
    feature_columns = ["feature_a", "feature_b"]
    shards = [
        _write_qdp_fixture_shard(tmp_path, year=year, stocks=stocks, feature_columns=feature_columns, include_label_v2=True)
        for year in (2019, 2020, 2021)
    ]
    manifest = {
        "artifact_type": "qdp_sharded_memmap",
        "profile": "style_structural_alpha_v2",
        "feature_profile": "style_structural_alpha_v2",
        "canonical_dataset_id": "policy_input_bundle__unit",
        "source_pool_view_id": "policy_pool_view__unit",
        "source_pool_view_kind": "tradeable_mainboard",
        "lookback_days": 3,
        "horizon": 1,
        "forecast_horizon": 1,
        "execution_mode": "next_open",
        "cumulative_horizons": [1],
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "label_schema_name": "path20_basic_v2",
        "label_schema_version": 2,
        "static_context_schema": {
            "enabled": True,
            "fields": ["symbol", "exchange", "industry"],
            "id_columns": ["symbol_id", "exchange_id", "industry_id"],
            "vocab_sizes": {"symbol": 3, "exchange": 3, "industry": 12},
            "embedding_defaults": {"symbol": 16, "exchange": 4, "industry": 8, "dropout": 0.2},
        },
        "shards": shards,
    }
    source_manifest_path = tmp_path / "sharded_memmap_manifest.json"
    source_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    pack_manifest = build_qdp_training_pack(
        source_manifest_path,
        output_root=tmp_path / "training_pack",
        train_start_year=2019,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        max_samples_per_role=4,
        feature_dtype="float32",
    )
    loaded = load_forecast_memmap_dataset(pack_manifest["manifest_json"])

    assert pack_manifest["label_schema_version"] == 2
    assert "daily_return" in pack_manifest["label_arrays"]
    assert "cumulative_excess_return_1to20" in pack_manifest["label_arrays"]
    assert "entry_tradeable" in pack_manifest["label_arrays"]
    assert isinstance(loaded, ForecastTrainingPackDataset)
    assert loaded.y_daily_return is not None
    assert loaded.y_cum_excess_1to20 is not None
    assert loaded.y_entry_tradeable is not None
    np.testing.assert_allclose(loaded.y_daily_return[0], np.asarray([0.02], dtype=np.float32))
    np.testing.assert_allclose(loaded.y_cum_excess_1to20[0], np.asarray([0.01], dtype=np.float32))
    assert float(loaded.y_entry_tradeable[0]) == 1.0


def test_qdp_training_pack_loader_applies_sample_cap_without_misalignment(tmp_path) -> None:
    stocks = ("AAA.SZ", "BBB.SH")
    feature_columns = ["feature_a", "feature_b"]
    shards = [
        _write_qdp_fixture_shard(tmp_path, year=year, stocks=stocks, feature_columns=feature_columns)
        for year in (2019, 2020, 2021)
    ]
    manifest = {
        "artifact_type": "qdp_sharded_memmap",
        "profile": "style_structural_v1",
        "feature_profile": "style_structural_v1",
        "canonical_dataset_id": "policy_input_bundle__unit",
        "source_pool_view_id": "policy_pool_view__unit",
        "source_pool_view_kind": "tradeable_mainboard",
        "lookback_days": 3,
        "horizon": 1,
        "forecast_horizon": 1,
        "execution_mode": "next_open",
        "cumulative_horizons": [1],
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "static_context_schema": {
            "enabled": True,
            "fields": ["symbol", "exchange", "industry"],
            "id_columns": ["symbol_id", "exchange_id", "industry_id"],
            "vocab_sizes": {"symbol": 3, "exchange": 3, "industry": 12},
            "embedding_defaults": {"symbol": 16, "exchange": 4, "industry": 8, "dropout": 0.2},
        },
        "shards": shards,
    }
    source_manifest_path = tmp_path / "sharded_memmap_manifest.json"
    source_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pack_manifest = build_qdp_training_pack(
        source_manifest_path,
        output_root=tmp_path / "training_pack",
        train_start_year=2019,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        max_samples_per_role=4,
        feature_dtype="float32",
    )

    full = load_forecast_memmap_dataset(pack_manifest["manifest_json"])
    capped = load_forecast_memmap_dataset(pack_manifest["manifest_json"], max_samples_per_role=2)

    assert isinstance(capped, ForecastTrainingPackDataset)
    assert capped.manifest["qdp_training_pack_sample_cap_applied"] is True
    assert capped.manifest["sample_count_by_role"] == {"train": 2, "validation": 2, "test": 2}
    assert capped.row_count == 6
    assert capped.static_context_ids is not None
    assert capped.static_context_ids.shape[0] == capped.row_count

    for capped_row, source_row in enumerate(capped.sample_index["_pack_row_idx"].to_numpy(dtype=np.int64)):
        assert capped.sample_index.iloc[capped_row]["stock"] == full.sample_index.iloc[int(source_row)]["stock"]
        np.testing.assert_allclose(capped.input_window(capped_row), full.input_window(int(source_row)), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(capped.y_daily_excess[capped_row], full.y_daily_excess[int(source_row)], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(capped.y_rank_by_horizon[capped_row], full.y_rank_by_horizon[int(source_row)], rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(capped.static_context_ids[capped_row], full.static_context_ids[int(source_row)])


def test_canonical_memmap_registry_resolves_matching_request(tmp_path) -> None:
    prepared = _make_fast_prepared()
    prepared.raw_cache_meta["dataset_id"] = "policy_input_bundle__unit"
    prepared.raw_cache_meta["pool_view"] = {
        "dataset_id": "policy_pool_view__unit",
        "view_kind": "rolling_liquidity",
        "view_name": "rolling_liquid500",
        "source_market_dataset_id": "policy_input_bundle__unit",
    }
    dataset = _build_fast_forecast_memmap(
        tmp_path / "memmap",
        prepared=prepared,
        test_year=2020,
    )
    registry_path = tmp_path / "registry.json"
    register_memmap_manifest(tmp_path / "memmap" / "forecast_dataset_manifest.json", registry_path=registry_path)
    args = Namespace(
        lake_dataset_id="policy_input_bundle__unit",
        forecast_lookback_days=5,
        execution_mode="next_open",
        forecast_feature_profile=dataset.manifest["feature_profile"],
        forecast_max_feature_columns=32,
        forecast_min_lookback_valid_ratio=0.80,
        forecast_max_samples_per_role=8,
        forecast_max_samples_per_date_per_role=0,
        forecast_train_start_year=2019,
        forecast_train_end_year=2019,
        forecast_validation_year=2020,
        forecast_test_year=2020,
        forecast_include_static_context=False,
        forecast_static_fields="symbol,exchange,industry,liquidity_bucket,price_bucket",
    )

    resolved = resolve_registered_memmap_manifest(
        prepared=prepared,
        args=args,
        horizon=20,
        cumulative_horizons=tuple(dataset.manifest["cumulative_horizons"]),
        registry_path=registry_path,
    )

    assert resolved == tmp_path / "memmap" / "forecast_dataset_manifest.json"


def test_canonical_memmap_registry_does_not_reuse_different_universe(tmp_path) -> None:
    prepared = _make_fast_prepared()
    prepared.raw_cache_meta["dataset_id"] = "policy_input_bundle__unit"
    dataset = _build_fast_forecast_memmap(
        tmp_path / "memmap",
        prepared=prepared,
        test_year=2020,
    )
    registry_path = tmp_path / "registry.json"
    register_memmap_manifest(tmp_path / "memmap" / "forecast_dataset_manifest.json", registry_path=registry_path)
    other_prepared = _make_fast_prepared(stocks=("AAA", "BBB"))
    other_prepared.raw_cache_meta["dataset_id"] = "policy_input_bundle__unit"
    args = Namespace(
        lake_dataset_id="policy_input_bundle__unit",
        forecast_lookback_days=5,
        execution_mode="next_open",
        forecast_feature_profile=dataset.manifest["feature_profile"],
        forecast_max_feature_columns=32,
        forecast_min_lookback_valid_ratio=0.80,
        forecast_max_samples_per_role=8,
        forecast_max_samples_per_date_per_role=0,
        forecast_train_start_year=2019,
        forecast_train_end_year=2019,
        forecast_validation_year=2020,
        forecast_test_year=2020,
        forecast_include_static_context=False,
        forecast_static_fields="symbol,exchange,industry,liquidity_bucket,price_bucket",
    )

    resolved = resolve_registered_memmap_manifest(
        prepared=other_prepared,
        args=args,
        horizon=20,
        cumulative_horizons=tuple(dataset.manifest["cumulative_horizons"]),
        registry_path=registry_path,
    )

    assert resolved is None


def test_forecast_memmap_dataset_writes_progress_files(tmp_path) -> None:
    _build_fast_forecast_memmap(
        tmp_path,
        max_samples_per_role=4,
    )

    memmap_progress = json.loads((tmp_path / "forecast_memmap_build_progress.json").read_text(encoding="utf-8"))
    feature_progress = json.loads((tmp_path / "forecast_feature_store_progress.json").read_text(encoding="utf-8"))

    assert memmap_progress["stage"] == "manifest_written"
    assert memmap_progress["sample_count"] > 0
    assert feature_progress["stage"] == "feature_profile_audit_done"
    assert feature_progress["feature_nan_ratio"] >= 0.0


def test_forecast_memmap_dataset_can_cap_samples_per_date(tmp_path) -> None:
    dataset = _build_fast_forecast_memmap(
        tmp_path,
        days=370,
        start_date="2019-11-01",
        train_start_year=2019,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        max_samples_per_role=0,
        max_samples_per_date_per_role=1,
    )

    assert dataset.manifest["max_samples_per_date_per_role"] == 1
    for role in ("train", "validation", "test"):
        rows = dataset.sample_index.loc[dataset.sample_index["role"].astype(str).eq(role)]
        assert rows["date"].nunique() > 3
        assert rows.groupby("date").size().max() == 1
        assert rows["stock"].nunique() > 1


def test_memmap_train_normalization_matches_naive_window_scan(tmp_path) -> None:
    feature_shape = (7, 3, 4)
    values = np.arange(np.prod(feature_shape), dtype=np.float32).reshape(feature_shape)
    values[2, 1, 0] = np.nan
    values[5, 2, 3] = np.nan
    feature_store = np.memmap(tmp_path / "forecast_feature_store.dat", dtype="float32", mode="w+", shape=feature_shape)
    feature_store[...] = values
    feature_store.flush()
    sample_index = __import__("pandas").DataFrame(
        [
            {"role": "train", "sequence_start_pos": 0, "date_pos": 2, "stock_pos": 1},
            {"role": "train", "sequence_start_pos": 1, "date_pos": 4, "stock_pos": 1},
            {"role": "train", "sequence_start_pos": 2, "date_pos": 6, "stock_pos": 2},
            {"role": "validation", "sequence_start_pos": 0, "date_pos": 6, "stock_pos": 0},
        ]
    )

    feature_mean, feature_std = _fit_memmap_train_normalization(
        feature_store_path=tmp_path / "forecast_feature_store.dat",
        feature_shape=feature_shape,
        sample_index=sample_index,
    )

    windows = []
    for row in sample_index[sample_index["role"].eq("train")].itertuples(index=False):
        windows.append(values[int(row.sequence_start_pos) : int(row.date_pos) + 1, int(row.stock_pos), :])
    stacked = np.concatenate(windows, axis=0)
    expected_mean = np.nanmean(stacked, axis=0).astype(np.float32)
    expected_std = np.nanstd(stacked, axis=0).astype(np.float32)
    expected_std = np.where(np.isfinite(expected_std) & (np.abs(expected_std) > 1.0e-8), expected_std, 1.0).astype(np.float32)

    np.testing.assert_allclose(feature_mean, expected_mean, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(feature_std, expected_std, rtol=1.0e-6, atol=1.0e-6)


def test_validate_forecast_memmap_manifest_reports_ok_and_source_binding(tmp_path) -> None:
    prepared = _make_fast_prepared()
    prepared.raw_cache_meta["dataset_id"] = "policy_input_bundle__unit"
    prepared.raw_cache_meta["pool_view"] = {
        "dataset_id": "policy_pool_view__unit",
        "view_kind": "rolling_liquidity",
        "view_name": "rolling_liquid500",
        "source_market_dataset_id": "policy_input_bundle__unit",
    }
    _build_fast_forecast_memmap(
        tmp_path,
        prepared=prepared,
        test_year=2020,
    )

    report = validate_forecast_memmap_manifest(
        manifest=tmp_path / "forecast_dataset_manifest.json",
        expect_source_market_dataset_id="policy_input_bundle__unit",
        expect_pool_view_id="policy_pool_view__unit",
        expect_pool_view_kind="rolling_liquidity",
        expect_pool_view_name="rolling_liquid500",
        min_universe_size=4,
        min_train_rows=1,
        require_roles=("train",),
    )

    assert report["status"] == "ok"
    assert report["blockers"] == []
    assert report["source_market_dataset_id"] == "policy_input_bundle__unit"
    assert report["source_pool_view_id"] == "policy_pool_view__unit"
    assert report["universe_size"] == 4
    assert report["sample_count_by_role"]["train"] > 0


def test_validate_forecast_memmap_manifest_blocks_pool_mismatch_and_low_train_rows(tmp_path) -> None:
    prepared = _make_fast_prepared()
    prepared.raw_cache_meta["dataset_id"] = "policy_input_bundle__unit"
    prepared.raw_cache_meta["pool_view"] = {
        "dataset_id": "policy_pool_view__unit",
        "view_kind": "rolling_liquidity",
        "view_name": "rolling_liquid500",
        "source_market_dataset_id": "policy_input_bundle__unit",
    }
    _build_fast_forecast_memmap(
        tmp_path,
        prepared=prepared,
        test_year=2020,
    )

    report = validate_forecast_memmap_manifest(
        manifest=tmp_path / "forecast_dataset_manifest.json",
        expect_pool_view_id="policy_pool_view__other",
        min_train_rows=999999,
        require_roles=("train", "validation", "test"),
    )

    assert report["status"] == "blocked"
    assert "source_pool_view_mismatch" in report["blockers"]
    assert "train_rows_below_minimum" in report["blockers"]


def test_validate_forecast_memmap_cli_writes_json_and_returns_nonzero_on_blocker(tmp_path) -> None:
    _build_fast_forecast_memmap(tmp_path)
    json_output = tmp_path / "validator.json"

    with pytest.raises(SystemExit) as exc:
        validate_memmap_main(
            [
                "--manifest",
                str(tmp_path / "forecast_dataset_manifest.json"),
                "--min-universe-size",
                "999",
                "--json-output",
                str(json_output),
            ]
        )

    assert exc.value.code == 1
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert "universe_size_below_minimum" in payload["blockers"]


def test_validate_forecast_memmap_manifest_classifies_missing_memmap_file(tmp_path) -> None:
    _build_fast_forecast_memmap(tmp_path)
    (tmp_path / "forecast_y_daily_excess.dat").unlink()

    report = validate_forecast_memmap_manifest(manifest=tmp_path / "forecast_dataset_manifest.json")

    assert report["status"] == "blocked"
    assert report["blockers"] == ["missing_memmap_file"]


def test_forecast_memmap_dataset_accepts_custom_horizon_grid_and_horizon_risk(tmp_path) -> None:
    horizons = (1, 2, 3, 5, 8, 10, 15, 20, 30)

    dataset = _build_fast_forecast_memmap(
        tmp_path,
        horizon=30,
        cumulative_horizons=horizons,
    )

    assert dataset.manifest["forecast_horizon"] == 30
    assert dataset.manifest["cumulative_horizons"] == list(horizons)
    assert dataset.manifest["risk_horizons"] == list(horizons)
    assert dataset.y_cum_excess.shape == (dataset.row_count, len(horizons))
    assert dataset.y_drawdown_by_horizon.shape == (dataset.row_count, len(horizons))
    train_indices = dataset.role_indices("train")
    x, y_daily, y_cum, y_risk, row_idx = dataset.torch_dataset(train_indices[:1], target_scale=100.0)[0]

    assert tuple(x.shape) == (5, dataset.input_dim)
    assert tuple(y_daily.shape) == (30,)
    assert tuple(y_cum.shape) == (len(horizons),)
    assert tuple(y_risk.shape) == (len(horizons), 3)
    assert int(row_idx.item()) == int(train_indices[0])

    loaded = load_forecast_memmap_dataset(tmp_path / "forecast_dataset_manifest.json")
    assert loaded.manifest["cumulative_horizons"] == list(horizons)
    assert loaded.y_drawdown_by_horizon.shape == (loaded.row_count, len(horizons))


def test_forecast_memmap_loader_resolves_copied_absolute_manifest_paths(tmp_path) -> None:
    dataset = _build_fast_forecast_memmap(
        tmp_path,
        stocks=("AAA", "BBB", "CCC"),
        max_samples_per_role=6,
    )
    manifest_path = tmp_path / "forecast_dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_store_path"] = str(tmp_path / "missing_old_root" / "forecast_feature_store.dat")
    manifest["sample_index_csv"] = str(tmp_path / "missing_old_root" / "forecast_sample_index.csv")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    loaded = load_forecast_memmap_dataset(manifest_path)

    assert loaded.row_count == dataset.row_count
    assert loaded.feature_store_path == tmp_path / "forecast_feature_store.dat"


def test_static_context_vocab_is_stable_and_memmap_samples_are_aligned(tmp_path) -> None:
    prepared = _make_fast_prepared(stocks=("AAA.SZ", "BBB.SH", "CCC.SZ"))
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
    reversed_prepared = _make_fast_prepared(stocks=("CCC.SZ", "BBB.SH", "AAA.SZ"))
    reversed_prepared.metadata_frames.update(prepared.metadata_frames)
    second = build_static_context_vocab(reversed_prepared)

    assert first["symbol_vocab_fingerprint"] == second["symbol_vocab_fingerprint"]
    assert first["industry_vocab_fingerprint"] == second["industry_vocab_fingerprint"]
    assert first["board_vocab_fingerprint"] == second["board_vocab_fingerprint"]

    dataset = _build_fast_forecast_memmap(
        tmp_path,
        prepared=prepared,
        max_samples_per_role=6,
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
    assert tuple(y_risk.shape) == (5, 3)

    loaded = load_forecast_memmap_dataset(tmp_path / "forecast_dataset_manifest.json")
    assert loaded.static_context_ids is not None
    _, _, _, _, loaded_row_idx, loaded_static_ids = loaded.torch_dataset(loaded.role_indices("train")[:1], target_scale=100.0)[0]
    assert int(loaded_row_idx.item()) == int(train_indices[0])
    assert tuple(loaded_static_ids.shape) == (5,)


def test_static_context_can_include_stable_primary_board_id(tmp_path) -> None:
    prepared = _make_fast_prepared(stocks=("AAA.SZ", "BBB.SH", "CCC.SZ"))
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

    dataset = _build_fast_forecast_memmap(
        tmp_path,
        prepared=prepared,
        max_samples_per_role=6,
        include_static_context=True,
        static_context_fields=("symbol", "exchange", "industry", "board", "liquidity_bucket", "price_bucket"),
    )

    assert dataset.manifest["static_context_schema"]["fields"] == [
        "symbol",
        "exchange",
        "industry",
        "board",
        "liquidity_bucket",
        "price_bucket",
    ]
    assert dataset.manifest["static_context_shape"] == [dataset.row_count, 6]
    assert "board_id" in dataset.sample_index.columns
    aaa_rows = dataset.sample_index[dataset.sample_index["stock"] == "AAA.SZ"]
    ccc_rows = dataset.sample_index[dataset.sample_index["stock"] == "CCC.SZ"]
    assert int(aaa_rows["board_id"].iloc[0]) > 0
    assert int(ccc_rows["board_id"].iloc[0]) == 0

    loaded = load_forecast_memmap_dataset(tmp_path / "forecast_dataset_manifest.json")
    assert loaded.static_context_ids is not None
    _, _, _, _, _, static_ids = loaded.torch_dataset(loaded.role_indices("train")[:1], target_scale=100.0)[0]
    assert tuple(static_ids.shape) == (6,)


def test_date_batch_view_groups_memmap_samples_by_signal_date(tmp_path) -> None:
    dataset = _build_fast_forecast_memmap(
        tmp_path,
        stocks=("AAA.SZ", "BBB.SH", "CCC.SZ"),
        max_samples_per_role=9,
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
    assert tuple(y_risk.shape) == (x.shape[0], 5, 3)
    assert tuple(static_ids.shape) == (x.shape[0], 5)
    assert len(set(dataset.sample_index.iloc[row_indices.numpy().astype(int)]["date"].astype(str))) == 1


def test_forecast_memmap_dataset_filters_low_history_samples(tmp_path) -> None:
    prepared = _make_fast_prepared(stocks=("AAA", "BBB", "CCC"))
    prepared.open_.iloc[:6, prepared.open_.columns.get_loc("AAA")] = np.nan
    prepared.close.iloc[:6, prepared.close.columns.get_loc("AAA")] = np.nan
    prepared.high.iloc[:6, prepared.high.columns.get_loc("AAA")] = np.nan
    prepared.low.iloc[:6, prepared.low.columns.get_loc("AAA")] = np.nan
    prepared.volume.iloc[:6, prepared.volume.columns.get_loc("AAA")] = np.nan
    prepared.amount.iloc[:6, prepared.amount.columns.get_loc("AAA")] = np.nan

    dataset = _build_fast_forecast_memmap(
        tmp_path,
        prepared=prepared,
        lookback_days=10,
        max_samples_per_role=12,
        min_lookback_valid_ratio=0.90,
    )

    assert dataset.manifest["dropped_low_history"] > 0
    assert dataset.manifest["min_lookback_valid_ratio"] == 0.90
    assert "history_valid_ratio_summary" in dataset.manifest
    assert set(dataset.sample_index["history_bucket"]).issubset({"low", "medium", "high"})


def test_forecast_memmap_dataset_loads_existing_manifest_without_rebuild(tmp_path) -> None:
    built = _build_fast_forecast_memmap(tmp_path)
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
    assert tuple(y_risk.shape) == (5, 3)
    assert int(row_idx.item()) >= 0


def test_forecast_memmap_dataset_rejects_inconsistent_reuse_manifest(tmp_path) -> None:
    built = _build_fast_forecast_memmap(tmp_path)
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
