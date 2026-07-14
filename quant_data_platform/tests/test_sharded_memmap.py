from __future__ import annotations

import json

import numpy as np
import pandas as pd

from daily_research.baseline.advanced_ml_runtime import HistoryWindow
from daily_research.continuous_policy.state_builder import PreparedPolicyInputs
from daily_research.path_policy.labels import build_path20_labels
from daily_research.path_policy.tests.fixtures import make_prepared_policy_inputs
from quant_data_platform.cli import main as cli_main
from quant_data_platform.core.json_io import read_json, write_json
from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.memmap import sharded
from quant_data_platform.memmap.sharded import (
    ShardedMemmapConfig,
    _build_shards_parallel_from_prepared,
    _build_shards_parallel,
    _chunk_shard_specs,
    _parse_year_filter,
    _project_feature_store_to_schema,
    _slice_prepared_for_symbols,
    _symbols_for_shard_specs,
    _symbol_blocks,
    _write_label_store,
    _write_sample_index,
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


def test_cli_sharded_dry_run_is_archived(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("QDP_WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "brain").mkdir(parents=True)
    (tmp_path / "brain" / "brain_manifest.json").write_text('{"brain_type": "main"}', encoding="utf-8")
    assert cli_main(
        [
            "build-sharded-memmap",
            "--dry-run",
            "--workers",
            "4",
            "--year-input-cache",
            "--canonical-dataset-id",
            "policy_input_bundle__override",
            "--pool-view-id",
            "policy_pool_view__unit",
            "--include-static-context",
            "--static-context-fields",
            "symbol,exchange,industry",
            "--json",
        ]
    ) == 2
    plan = qdp_paths(tmp_path).memmap_dir / "sharded_memmap_plan.json"
    assert not plan.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "archived"
    assert payload["command"] == "build-sharded-memmap"


def test_sharded_config_normalizes_workers() -> None:
    assert ShardedMemmapConfig(canonical_dataset_id=" policy_input_bundle__override ").normalized().canonical_dataset_id == "policy_input_bundle__override"
    assert ShardedMemmapConfig(workers=0).normalized().workers == 1
    assert ShardedMemmapConfig(workers=4).normalized().workers == 4
    assert ShardedMemmapConfig(year_input_cache=True).normalized().year_input_cache is True
    assert ShardedMemmapConfig(force_years=" 2010-2015 ").normalized().force_years == "2010-2015"
    assert ShardedMemmapConfig(static_context_fields="symbol_id,exchange_id").normalized().static_context_fields == "symbol,exchange"


def test_slice_prepared_for_symbols_filters_wide_frames_and_metadata() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    symbols = ["AAA.SZ", "BBB.SZ", "CCC.SZ"]
    close = pd.DataFrame(
        [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]],
        index=dates,
        columns=symbols,
    )
    prepared = PreparedPolicyInputs(
        universe=tuple(symbols),
        pool_name="unit",
        benchmark="000300.SH",
        data_source="lake",
        csv_folder="",
        start_date="2024-01-02",
        end_date="2024-01-03",
        requested_start_date="20240102",
        history_window=HistoryWindow(mode="train", requested_start_date="20240102", effective_start_date="20240102", end_date="20240103", required_trading_days=2),
        raw_cache_meta={},
        prepared_cache_meta={},
        close=close,
        open_=close + 0.1,
        high=close + 0.2,
        low=close - 0.2,
        volume=close * 100,
        amount=close * 1000,
        benchmark_close=pd.Series([1.0, 1.1], index=dates),
        benchmark_open=pd.Series([1.0, 1.1], index=dates),
        score_none=close * 0.0,
        score_v2=close * 0.0,
        score_blend=close * 0.0,
        feature_frames={"score": close * 2},
        market_features={},
        membership_frame=pd.DataFrame(True, index=dates, columns=symbols),
        rolling_pool_summary={},
        alpha_prior_summary={},
        derived_frames={"ret_1d": close.pct_change()},
        metadata_frames={
            "industry_daily": pd.DataFrame(
                {
                    "trade_date": ["2024-01-02", "2024-01-02", "2024-01-02"],
                    "symbol": symbols,
                    "industry": ["a", "b", "c"],
                }
            )
        },
        metadata_summary={},
    )

    sliced = _slice_prepared_for_symbols(prepared, ["BBB.SZ", "MISSING.SZ"])

    assert sliced is not None
    assert sliced.universe == ("BBB.SZ",)
    assert list(sliced.close.columns) == ["BBB.SZ"]
    assert list(sliced.feature_frames["score"].columns) == ["BBB.SZ"]
    assert list(sliced.derived_frames["ret_1d"].columns) == ["BBB.SZ"]
    assert sliced.metadata_frames["industry_daily"]["symbol"].tolist() == ["BBB.SZ"]


def test_shard_spec_batch_helpers_preserve_block_order_and_deduplicate_symbols() -> None:
    specs = [
        {"shard_key": "year=2024/block=0000", "symbols": ["AAA.SZ", "BBB.SH"]},
        {"shard_key": "year=2024/block=0001", "symbols": ["BBB.SH", "CCC.SZ"]},
        {"shard_key": "year=2024/block=0002", "symbols": ["DDD.SH"]},
    ]

    batches = _chunk_shard_specs(specs, 2)

    assert [[item["shard_key"] for item in batch] for batch in batches] == [
        ["year=2024/block=0000", "year=2024/block=0001"],
        ["year=2024/block=0002"],
    ]
    assert _symbols_for_shard_specs(batches[0]) == ["AAA.SZ", "BBB.SH", "CCC.SZ"]


def test_parse_year_filter_supports_ranges_and_commas() -> None:
    assert _parse_year_filter("2010-2012,2024") == {2010, 2011, 2012, 2024}
    assert _parse_year_filter("") == set()


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
        static_schema={},
        progress_path=tmp_path / "progress.json",
        planned_count=2,
        completed_so_far=0,
    )

    assert sorted(results) == ["year=2024/block=0000", "year=2024/block=0001"]
    assert (tmp_path / "progress.json").exists()
    assert all(item["status"] == "completed" for item in results.values())


def test_build_shards_parallel_from_prepared_uses_cached_year_inputs(tmp_path, monkeypatch) -> None:
    calls = []
    prepared = make_prepared_policy_inputs(days=30, stocks=("AAA.SZ", "BBB.SH"))

    def fake_cached_worker(**kwargs):
        calls.append((kwargs["prepared"], tuple(kwargs["symbols"]), int(kwargs["block_id"])))
        year = int(kwargs["year"])
        block_id = int(kwargs["block_id"])
        shard_dir = tmp_path / "out" / f"year={year}" / f"block={block_id:04d}"
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

    monkeypatch.setattr(sharded, "_build_one_shard_from_cached_year_worker", fake_cached_worker)

    results = _build_shards_parallel_from_prepared(
        prepared=prepared,
        year_load_error="",
        shard_specs=[
            {"shard_key": "year=2024/block=0000", "year": 2024, "block_id": 0, "symbols": ["AAA.SZ"]},
            {"shard_key": "year=2024/block=0001", "year": 2024, "block_id": 1, "symbols": ["BBB.SH"]},
        ],
        target_start=pd.Timestamp("2024-01-01"),
        target_end=pd.Timestamp("2024-12-31"),
        context_start=pd.Timestamp("2023-01-01"),
        context_end=pd.Timestamp("2025-01-31"),
        cfg=ShardedMemmapConfig(workers=2, year_input_cache=True).normalized(),
        out_root=tmp_path / "out",
        cumulative_horizons=(1, 3, 5),
        expected_feature_columns=["a", "b"],
        static_schema={},
        progress_path=tmp_path / "progress.json",
        planned_count=2,
        completed_so_far=0,
    )

    assert sorted(results) == ["year=2024/block=0000", "year=2024/block=0001"]
    assert all(item["status"] == "completed" for item in results.values())
    assert sorted(call[1:] for call in calls) == [(("AAA.SZ",), 0), (("BBB.SH",), 1)]
    assert all(call[0] is prepared for call in calls)
    assert read_json(tmp_path / "progress.json")["worker_count"] == 2


def test_build_shards_parallel_from_prepared_writes_year_load_no_coverage(tmp_path) -> None:
    results = _build_shards_parallel_from_prepared(
        prepared=None,
        year_load_error="lake_coverage_blocker: no requested symbols are available in lake market data",
        shard_specs=[
            {"shard_key": "year=2010/block=0000", "year": 2010, "block_id": 0, "symbols": ["AAA.SZ"]},
            {"shard_key": "year=2010/block=0001", "year": 2010, "block_id": 1, "symbols": ["BBB.SH"]},
        ],
        target_start=pd.Timestamp("2010-01-01"),
        target_end=pd.Timestamp("2010-12-31"),
        context_start=pd.Timestamp("2010-01-01"),
        context_end=pd.Timestamp("2011-01-31"),
        cfg=ShardedMemmapConfig(workers=2, year_input_cache=True).normalized(),
        out_root=tmp_path / "out",
        cumulative_horizons=(1, 3, 5),
        expected_feature_columns=["a", "b"],
        static_schema={},
        progress_path=tmp_path / "progress.json",
        planned_count=2,
        completed_so_far=0,
    )

    assert sorted(results) == ["year=2010/block=0000", "year=2010/block=0001"]
    assert {item["status"] for item in results.values()} == {"no_coverage"}
    assert read_json(tmp_path / "progress.json")["completed_shards"] == 2


def test_write_label_store_writes_basic_v2_arrays(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=30, stocks=("AAA.SZ", "BBB.SH"))
    labels = build_path20_labels(prepared, execution_mode="next_open", horizon=3, cumulative_horizons=(1, 3))
    dates = list(prepared.close.index[:5])
    symbols = list(prepared.close.columns)

    manifest = _write_label_store(
        shard_dir=tmp_path,
        labels=labels,
        dates=dates,
        symbols=symbols,
        horizon=3,
        cumulative_horizons=(1, 3),
    )

    arrays = dict(manifest["arrays"])
    assert manifest["label_schema_version"] == 2
    assert arrays["daily_return"]["shape"] == [5, 2, 3]
    assert arrays["benchmark_daily_return"]["shape"] == [5, 2, 3]
    assert arrays["cumulative_excess_return_1to20"]["shape"] == [5, 2, 3]
    assert arrays["rank_1to20"]["shape"] == [5, 2, 3]
    assert arrays["industry_rank_by_horizon"]["shape"] == [5, 2, 2]
    assert arrays["entry_tradeable"]["shape"] == [5, 2]
    benchmark = np.memmap(
        arrays["benchmark_daily_return"]["path"],
        dtype="float32",
        mode="r",
        shape=tuple(arrays["benchmark_daily_return"]["shape"]),
    )
    np.testing.assert_allclose(np.asarray(benchmark)[:, 0, :], np.asarray(benchmark)[:, 1, :])


def test_write_sample_index_vectorized_filters_valid_rows(tmp_path) -> None:
    class Prepared:
        membership_frame = pd.DataFrame(
            [[True, True], [True, False]],
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
            columns=["AAA.SZ", "BBB.SZ"],
        )

    dates = list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    symbols = ["AAA.SZ", "BBB.SZ"]
    history_ratio = pd.DataFrame(
        [[0.9, 0.7], [0.8, 0.95]],
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        columns=symbols,
    )
    label_dir = tmp_path / "labels"
    label_dir.mkdir(parents=True)
    arrays = {}
    for name, shape in {
        "daily_excess_return": (2, 2, 1),
        "cumulative_excess_return": (2, 2, 1),
        "rank_by_horizon": (2, 2, 1),
        "max_drawdown_20d": (2, 2),
        "worst_1d_20d": (2, 2),
        "upside_20d": (2, 2),
    }.items():
        path = label_dir / f"{name}.dat"
        store = np.memmap(path, dtype="float32", mode="w+", shape=shape)
        store[...] = 1.0
        store.flush()
        del store
        arrays[name] = {"path": str(path), "shape": list(shape)}

    sample_path, sample_count = _write_sample_index(
        shard_dir=tmp_path,
        prepared=Prepared(),
        dates=dates,
        symbols=symbols,
        history_ratio=history_ratio,
        label_manifest={"arrays": arrays},
        min_lookback_valid_ratio=0.8,
    )

    frame = pd.read_parquet(sample_path)
    assert sample_count == 2
    assert frame[["date", "stock", "date_pos", "stock_pos"]].to_dict("records") == [
        {"date": "2024-01-02", "stock": "AAA.SZ", "date_pos": 0, "stock_pos": 0},
        {"date": "2024-01-03", "stock": "AAA.SZ", "date_pos": 1, "stock_pos": 0},
    ]


def test_write_sample_index_adds_static_context_columns(tmp_path) -> None:
    class Prepared:
        universe = ("AAA.SZ", "BBB.SH")
        membership_frame = pd.DataFrame(
            [[True, True], [True, True]],
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
            columns=["AAA.SZ", "BBB.SH"],
        )
        amount = pd.DataFrame(
            [[100.0, 200.0], [110.0, 210.0]],
            index=membership_frame.index,
            columns=membership_frame.columns,
        )
        close = pd.DataFrame(
            [[10.0, 20.0], [11.0, 21.0]],
            index=membership_frame.index,
            columns=membership_frame.columns,
        )
        metadata_frames = {
            "industry_map": pd.DataFrame(
                {
                    "symbol": ["AAA.SZ", "BBB.SH"],
                    "industry": ["tech", "bank"],
                }
            )
        }

    dates = list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    symbols = ["AAA.SZ", "BBB.SH"]
    history_ratio = pd.DataFrame(0.9, index=dates, columns=symbols)
    label_dir = tmp_path / "labels"
    label_dir.mkdir(parents=True)
    arrays = {}
    for name, shape in {
        "daily_excess_return": (2, 2, 1),
        "cumulative_excess_return": (2, 2, 1),
        "rank_by_horizon": (2, 2, 1),
        "max_drawdown_20d": (2, 2),
        "worst_1d_20d": (2, 2),
        "upside_20d": (2, 2),
    }.items():
        path = label_dir / f"{name}.dat"
        store = np.memmap(path, dtype="float32", mode="w+", shape=shape)
        store[...] = 1.0
        store.flush()
        del store
        arrays[name] = {"path": str(path), "shape": list(shape)}
    static_schema = sharded.build_static_context_vocab(Prepared(), static_context_fields=("symbol", "exchange", "industry"))

    sample_path, sample_count = _write_sample_index(
        shard_dir=tmp_path,
        prepared=Prepared(),
        dates=dates,
        symbols=symbols,
        history_ratio=history_ratio,
        label_manifest={"arrays": arrays},
        min_lookback_valid_ratio=0.8,
        include_static_context=True,
        static_schema=static_schema,
    )

    frame = pd.read_parquet(sample_path)
    assert sample_count == 4
    assert {"symbol_id", "exchange_id", "industry_id"}.issubset(frame.columns)
    assert frame.loc[frame["stock"] == "AAA.SZ", "symbol_id"].nunique() == 1
    assert int(frame.loc[frame["stock"] == "AAA.SZ", "symbol_id"].iloc[0]) > 0
    assert int(frame.loc[frame["stock"] == "BBB.SH", "exchange_id"].iloc[0]) > 0
