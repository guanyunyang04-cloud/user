from __future__ import annotations

import json
from argparse import Namespace

import numpy as np
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
    build_static_context_vocab,
    load_forecast_memmap_dataset,
)
from daily_research.path_policy.tests.fixtures import make_prepared_policy_inputs
from daily_research.path_policy.validate_forecast_memmap import main as validate_memmap_main
from daily_research.path_policy.validate_forecast_memmap import validate_forecast_memmap_manifest


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
        test_year=2020,
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
    assert tuple(y_risk.shape) == (5, 3)
    assert int(row_idx.item()) == int(train_indices[0])
    assert torch.isfinite(x).all()


def test_canonical_memmap_registry_resolves_matching_request(tmp_path) -> None:
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
        root=tmp_path / "memmap",
        train_start_year=2019,
        train_end_year=2019,
        validation_year=2020,
        test_year=2020,
        lookback_days=5,
        horizon=20,
        max_samples_per_role=8,
        max_feature_columns=32,
        min_lookback_valid_ratio=0.80,
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


def test_forecast_memmap_dataset_writes_progress_files(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=420, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2019-07-01")
    build_forecast_memmap_dataset(
        prepared,
        root=tmp_path,
        train_start_year=2019,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        lookback_days=5,
        horizon=20,
        max_samples_per_role=4,
        min_lookback_valid_ratio=0.80,
    )

    memmap_progress = json.loads((tmp_path / "forecast_memmap_build_progress.json").read_text(encoding="utf-8"))
    feature_progress = json.loads((tmp_path / "forecast_feature_store_progress.json").read_text(encoding="utf-8"))

    assert memmap_progress["stage"] == "manifest_written"
    assert memmap_progress["sample_count"] > 0
    assert feature_progress["stage"] == "feature_profile_audit_done"
    assert feature_progress["feature_nan_ratio"] >= 0.0


def test_forecast_memmap_dataset_can_cap_samples_per_date(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=820, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2018-01-02")
    dataset = build_forecast_memmap_dataset(
        prepared,
        root=tmp_path,
        train_start_year=2018,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        lookback_days=5,
        horizon=20,
        max_samples_per_role=0,
        max_samples_per_date_per_role=1,
        min_lookback_valid_ratio=0.80,
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
    prepared = make_prepared_policy_inputs(days=420, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2019-07-01")
    prepared.raw_cache_meta["dataset_id"] = "policy_input_bundle__unit"
    prepared.raw_cache_meta["pool_view"] = {
        "dataset_id": "policy_pool_view__unit",
        "view_kind": "rolling_liquidity",
        "view_name": "rolling_liquid500",
        "source_market_dataset_id": "policy_input_bundle__unit",
    }
    build_forecast_memmap_dataset(
        prepared,
        root=tmp_path,
        train_start_year=2019,
        train_end_year=2019,
        validation_year=2020,
        test_year=2020,
        lookback_days=5,
        horizon=20,
        max_samples_per_role=8,
        min_lookback_valid_ratio=0.80,
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
    prepared = make_prepared_policy_inputs(days=420, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2019-07-01")
    prepared.raw_cache_meta["dataset_id"] = "policy_input_bundle__unit"
    prepared.raw_cache_meta["pool_view"] = {
        "dataset_id": "policy_pool_view__unit",
        "view_kind": "rolling_liquidity",
        "view_name": "rolling_liquid500",
        "source_market_dataset_id": "policy_input_bundle__unit",
    }
    build_forecast_memmap_dataset(
        prepared,
        root=tmp_path,
        train_start_year=2019,
        train_end_year=2019,
        validation_year=2020,
        test_year=2020,
        lookback_days=5,
        horizon=20,
        max_samples_per_role=8,
        min_lookback_valid_ratio=0.80,
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
    prepared = make_prepared_policy_inputs(days=420, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2019-07-01")
    build_forecast_memmap_dataset(
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
    prepared = make_prepared_policy_inputs(days=420, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2019-07-01")
    build_forecast_memmap_dataset(
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
    (tmp_path / "forecast_y_daily_excess.dat").unlink()

    report = validate_forecast_memmap_manifest(manifest=tmp_path / "forecast_dataset_manifest.json")

    assert report["status"] == "blocked"
    assert report["blockers"] == ["missing_memmap_file"]


def test_forecast_memmap_dataset_accepts_custom_horizon_grid_and_horizon_risk(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=900, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2018-01-02")
    horizons = (1, 2, 3, 5, 8, 10, 15, 20, 30)

    dataset = build_forecast_memmap_dataset(
        prepared,
        root=tmp_path,
        train_start_year=2018,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        lookback_days=5,
        horizon=30,
        cumulative_horizons=horizons,
        max_samples_per_role=8,
        min_lookback_valid_ratio=0.80,
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
    prepared = make_prepared_policy_inputs(days=420, stocks=("AAA", "BBB", "CCC"), start_date="2019-07-01")
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
    assert tuple(y_risk.shape) == (5, 3)

    loaded = load_forecast_memmap_dataset(tmp_path / "forecast_dataset_manifest.json")
    assert loaded.static_context_ids is not None
    _, _, _, _, loaded_row_idx, loaded_static_ids = loaded.torch_dataset(loaded.role_indices("train")[:1], target_scale=100.0)[0]
    assert int(loaded_row_idx.item()) == int(train_indices[0])
    assert tuple(loaded_static_ids.shape) == (5,)


def test_static_context_can_include_stable_primary_board_id(tmp_path) -> None:
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
    assert tuple(y_risk.shape) == (x.shape[0], 5, 3)
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
    assert tuple(y_risk.shape) == (5, 3)
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
